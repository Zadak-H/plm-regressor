#!/usr/bin/env python
import argparse
import json
import os

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import T5EncoderModel, AutoTokenizer


def is_cuda_oom_error(exc: RuntimeError) -> bool:
    msg = str(exc).lower()
    return (
        "cuda out of memory" in msg
        or "outofmemoryerror" in msg
        or "cublas_status_alloc_failed" in msg
    )


def print_header(title: str):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def configure_torch_for_cpu(num_threads: int, num_interop_threads: int):
    if num_threads < 1:
        raise ValueError("--num-threads must be >= 1")
    if num_interop_threads < 1:
        raise ValueError("--num-interop-threads must be >= 1")
    torch.set_num_threads(num_threads)
    torch.set_num_interop_threads(num_interop_threads)
    print(f"torch num threads: {torch.get_num_threads()}")
    print(f"torch num interop threads: {torch.get_num_interop_threads()}")


def load_prot_t5_model(model_id: str, device: torch.device, cuda_dtype: str = "fp16"):
    print(f"Loading tokenizer from {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        do_lower_case=False,
        use_fast=False,
    )

    print(f"Loading model from {model_id}")
    model_kwargs = {"low_cpu_mem_usage": True}
    if device.type == "cuda":
        dtype_map = {
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
            "fp32": torch.float32,
        }
        if cuda_dtype not in dtype_map:
            raise ValueError("--cuda-dtype must be one of: fp16, bf16, fp32")
        model_kwargs["torch_dtype"] = dtype_map[cuda_dtype]

    model = T5EncoderModel.from_pretrained(model_id, **model_kwargs)
    model.to(device)
    model.eval()
    if device.type == "cuda":
        print(f"Model loaded on CUDA with dtype: {next(model.parameters()).dtype}")
    return tokenizer, model


def prepare_sequence_for_t5(sequence: str) -> str:
    return " ".join(list(sequence))


def extract_prot_t5_embeddings(
    sequences: list,
    tokenizer,
    model: T5EncoderModel,
    device: torch.device,
    batch_size: int,
    use_cuda_amp: bool = True,
) -> np.ndarray:
    embeddings = []
    prepped_sequences = [prepare_sequence_for_t5(seq) for seq in sequences]
    special_ids = set(int(tok_id) for tok_id in tokenizer.all_special_ids)

    amp_enabled = use_cuda_amp and device.type == "cuda"
    with torch.inference_mode():
        for i in tqdm(range(0, len(prepped_sequences), batch_size), desc="Processing Batches"):
            batch_seqs = prepped_sequences[i : i + batch_size]

            ids = tokenizer(
                batch_seqs,
                add_special_tokens=True,
                padding="longest",
                return_tensors="pt",
            )

            input_ids = ids["input_ids"].to(device)
            attention_mask = ids["attention_mask"].to(device)

            if amp_enabled:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    embedding_output = model(input_ids=input_ids, attention_mask=attention_mask)
            else:
                embedding_output = model(input_ids=input_ids, attention_mask=attention_mask)

            residue_embs = embedding_output.last_hidden_state  # (B, L, D)
            token_mask = attention_mask.bool()
            if special_ids:
                special_mask = torch.zeros_like(token_mask)
                for tok_id in special_ids:
                    special_mask |= input_ids.eq(tok_id)
                token_mask &= ~special_mask
            if not token_mask.any():
                token_mask = attention_mask.bool()
            token_mask = token_mask.unsqueeze(-1).to(dtype=residue_embs.dtype)
            token_counts = token_mask.sum(dim=1).clamp_min(1.0)
            seq_embeddings = (residue_embs * token_mask).sum(dim=1) / token_counts

            embeddings.append(seq_embeddings.detach().cpu().numpy())

    return np.concatenate(embeddings, axis=0).astype(np.float32)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract ProtT5/ProstT5 (T5) embeddings for protein sequences.")
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--seq-col", required=True)
    parser.add_argument("--output-npz", required=True)
    parser.add_argument("--model-id", default="Rostlab/ProstT5")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--no-cuda-amp", action="store_true")
    parser.add_argument("--cuda-dtype", choices=["fp16", "bf16", "fp32"], default="fp16")
    parser.add_argument("--cpu-fallback-on-oom", action="store_true")
    parser.add_argument("--num-threads", type=int, default=16)
    parser.add_argument("--num-interop-threads", type=int, default=1)
    args = parser.parse_args()

    print_header(f"Loading data from {args.input_csv}")
    df = pd.read_csv(args.input_csv)

    if args.seq_col not in df.columns:
        raise ValueError(f"Sequence column '{args.seq_col}' not found in CSV")

    df[args.seq_col] = df[args.seq_col].astype(str).str.strip()
    df = df[df[args.seq_col].notna() & (df[args.seq_col].str.len() > 0)].copy()

    unique_seqs = df[args.seq_col].unique().tolist()
    print("Rows with non-null and non-empty sequences:", df.shape[0])
    print("Unique sequences:", len(unique_seqs))

    seq_to_index = {seq: idx for idx, seq in enumerate(unique_seqs)}

    if torch.cuda.is_available() and not args.cpu:
        device = torch.device("cuda")
        print("Using CUDA")
    else:
        device = torch.device("cpu")
        print("Using CPU")

    configure_torch_for_cpu(args.num_threads, args.num_interop_threads)

    print_header(f"Loading ProtT5/ProstT5 model ({args.model_id})")
    tokenizer, model = load_prot_t5_model(args.model_id, device, cuda_dtype=args.cuda_dtype)

    print_header("Extracting ProtT5/ProstT5 embeddings")
    try:
        embeddings = extract_prot_t5_embeddings(
            unique_seqs, tokenizer, model, device=device,
            batch_size=args.batch_size, use_cuda_amp=not args.no_cuda_amp
        )
    except RuntimeError as exc:
        if not (args.cpu_fallback_on_oom and device.type == "cuda" and is_cuda_oom_error(exc)):
            raise
        print("[WARN] CUDA OOM during extraction. Falling back to CPU.")
        del model
        torch.cuda.empty_cache()
        cpu_device = torch.device("cpu")
        tokenizer, model = load_prot_t5_model(args.model_id, cpu_device, cuda_dtype=args.cuda_dtype)
        embeddings = extract_prot_t5_embeddings(
            unique_seqs, tokenizer, model, device=cpu_device,
            batch_size=max(1, min(args.batch_size, 8)), use_cuda_amp=False
        )

    print("Embeddings shape:", embeddings.shape)

    out_dir = os.path.dirname(args.output_npz)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    print_header("Saving to NPZ")
    np.savez_compressed(
        args.output_npz,
        embeddings=embeddings,
        sequences=np.array(unique_seqs, dtype=object),
        seq_to_index=seq_to_index,  # keep as dict
    )
    print(f"Successfully saved embeddings to {args.output_npz}")
