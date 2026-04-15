#!/usr/bin/env python
"""
Extract ESM1v embeddings for PETase sequences (FAST for H100 + sane CPU-8).

Key speedups:
- H100: TF32 enabled + BF16 autocast (default), optional torch.compile
- Token-budget batching (--max-tokens) to reduce padding & increase throughput
- Preallocated output tensor on CPU (pinned when CUDA) for faster transfers
- Better CPU thread settings (8 cores) + avoid CPU thread contention on GPU runs
- Cluster caching: honor TORCH_HOME / XDG_CACHE_HOME (set in environment)

Examples
--------
Single model (GPU, H100-friendly):
python scripts/embeds_scripts/extract_esm1v_embeddings.py \
  --input-csv data_processed/activity_dataset_clean.csv \
  --seq-col Sequence_clean \
  --output-npz embeddings/esm1v_activity_unique_seqs_650M.npz \
  --model-names esm1v_t33_650M_UR90S_1 \
  --max-tokens 12000 \
  --dtype bf16 \
  --compile

Ensemble avg (5 models):
python scripts/embeds_scripts/extract_esm1v_embeddings.py \
  --input-csv data_processed/activity_dataset_clean.csv \
  --seq-col Sequence_clean \
  --output-npz embeddings/esm1v_ensemble5_activity_unique_seqs_650M.npz \
  --model-names esm1v_t33_650M_UR90S_1,esm1v_t33_650M_UR90S_2,esm1v_t33_650M_UR90S_3,esm1v_t33_650M_UR90S_4,esm1v_t33_650M_UR90S_5 \
  --max-tokens 12000 \
  --dtype bf16

CPU (8 cores):
python scripts/embeds_scripts/extract_esm1v_embeddings.py \
  --cpu \
  --num-threads 8 \
  --num-interop-threads 1 \
  --input-csv data_processed/activity_dataset_clean.csv \
  --seq-col Sequence_clean \
  --output-npz embeddings/esm1v_cpu.npz \
  --max-tokens 2000
"""

import argparse
import json
import os
from typing import Iterable, List, Optional

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

import esm  # from fair-esm


# ------------------------- utils -------------------------


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
    """
    Configure torch CPU threading for faster inference.
    """
    if num_threads < 1:
        raise ValueError("--num-threads must be >= 1")
    if num_interop_threads < 1:
        raise ValueError("--num-interop-threads must be >= 1")

    torch.set_num_threads(num_threads)
    torch.set_num_interop_threads(num_interop_threads)
    print(f"torch num threads: {torch.get_num_threads()}")
    print(f"torch num interop threads: {torch.get_num_interop_threads()}")


def configure_torch_for_cuda():
    """
    CUDA speed settings (good defaults for H100).
    """
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass


def pick_autocast_dtype(dtype_str: str) -> Optional[torch.dtype]:
    if dtype_str == "bf16":
        return torch.bfloat16
    if dtype_str == "fp16":
        return torch.float16
    # fp32 => disable autocast
    return None


def load_esm1v_model(model_name: str, device: torch.device):
    """
    Load a single ESM1v model by name via esm.pretrained.

    E.g. model_name = "esm1v_t33_650M_UR90S_1"
    """
    if not hasattr(esm, "pretrained"):
        raise RuntimeError(
            "esm.pretrained not found. Make sure you installed `fair-esm` "
            "and uninstalled any conflicting `esm` package:\n"
            "  pip uninstall -y esm\n"
            "  pip install fair-esm\n"
        )

    print(f"Using esm.pretrained.{model_name}()")
    if not hasattr(esm.pretrained, model_name):
        raise ValueError(
            f"esm.pretrained has no attribute '{model_name}'. "
            f"Check the model name (e.g. esm1v_t33_650M_UR90S_1 ... _5)."
        )

    load_fn = getattr(esm.pretrained, model_name)
    model, alphabet = load_fn()
    model.to(device)
    model.eval()
    return model, alphabet


def make_token_batches(
    sorted_indices: List[int],
    sequences: List[str],
    max_tokens: int,
    fallback_batch_size: int,
) -> Iterable[List[int]]:
    """
    Yield batches of indices.
    If max_tokens>0: dynamic batching so sum(len(seq)+2) <= max_tokens (roughly accounts BOS/EOS).
    Else: fixed batch size.
    """
    if max_tokens and max_tokens > 0:
        batch: List[int] = []
        tok_sum = 0
        for idx in sorted_indices:
            n_tok = len(sequences[idx]) + 2
            # If single sequence exceeds max_tokens, still yield it alone.
            if batch and (tok_sum + n_tok) > max_tokens:
                yield batch
                batch = [idx]
                tok_sum = n_tok
            else:
                batch.append(idx)
                tok_sum += n_tok
        if batch:
            yield batch
    else:
        bs = max(1, fallback_batch_size)
        for i in range(0, len(sorted_indices), bs):
            yield sorted_indices[i : i + bs]


def extract_embeddings_for_sequences(
    sequences: List[str],
    model,
    alphabet,
    device: torch.device,
    batch_size: int = 8,
    min_batch_size: int = 1,
    use_cuda_amp: bool = True,
    layer: Optional[int] = None,
    max_tokens: int = 0,
    amp_dtype: Optional[torch.dtype] = torch.bfloat16,
):
    """
    sequences: list of str (unique sequences)
    Returns: np.ndarray of shape (N, D) float32
    """
    model.eval()

    if layer is None:
        layer = model.num_layers  # final layer

    batch_converter = alphabet.get_batch_converter()

    # Sort by length to reduce padding
    sorted_indices = sorted(
        range(len(sequences)),
        key=lambda idx: len(sequences[idx]),
        reverse=True,
    )

    # Determine embedding dim (robust-ish)
    D = getattr(model, "embed_dim", None)
    if D is None and hasattr(model, "args") and hasattr(model.args, "embed_dim"):
        D = int(model.args.embed_dim)
    if D is None:
        # fallback guess for ESM1v 650M
        D = 1280

    # Preallocate output on CPU; pin memory for faster D2H copies
    out_cpu = torch.empty(
        (len(sequences), D),
        dtype=torch.float32,
        pin_memory=(device.type == "cuda"),
    )

    use_amp = use_cuda_amp and device.type == "cuda" and (amp_dtype is not None)

    with torch.inference_mode():
        with tqdm(total=len(sorted_indices), desc="Embedding batches") as pbar:
            for batch_indices in make_token_batches(
                sorted_indices, sequences, max_tokens=max_tokens, fallback_batch_size=batch_size
            ):
                data = [(f"seq_{idx}", sequences[idx]) for idx in batch_indices]
                _, _, tokens = batch_converter(data)

                if device.type == "cuda":
                    tokens = tokens.pin_memory().to(device, non_blocking=True)
                else:
                    tokens = tokens.to(device)

                try:
                    if use_amp:
                        with torch.autocast(device_type="cuda", dtype=amp_dtype):
                            out = model(tokens, repr_layers=[layer], return_contacts=False)
                    else:
                        out = model(tokens, repr_layers=[layer], return_contacts=False)
                except RuntimeError as exc:
                    # Only do auto batch-size halving when using fixed batch size mode.
                    if device.type == "cuda" and is_cuda_oom_error(exc) and (max_tokens <= 0):
                        torch.cuda.empty_cache()
                        if len(batch_indices) <= min_batch_size:
                            raise
                        new_bs = max(min_batch_size, len(batch_indices) // 2)
                        if new_bs == len(batch_indices):
                            raise
                        print(
                            f"[WARN] CUDA OOM at batch_size={len(batch_indices)}; "
                            f"retrying by splitting into batch_size={new_bs}"
                        )
                        # Split and retry each chunk (simple & robust)
                        for k in range(0, len(batch_indices), new_bs):
                            sub = batch_indices[k : k + new_bs]
                            sub_data = [(f"seq_{idx}", sequences[idx]) for idx in sub]
                            _, _, sub_tokens = batch_converter(sub_data)
                            sub_tokens = sub_tokens.pin_memory().to(device, non_blocking=True)

                            if use_amp:
                                with torch.autocast(device_type="cuda", dtype=amp_dtype):
                                    sub_out = model(
                                        sub_tokens, repr_layers=[layer], return_contacts=False
                                    )
                            else:
                                sub_out = model(sub_tokens, repr_layers=[layer], return_contacts=False)

                            sub_reps = sub_out["representations"][layer]
                            for j, seq_idx in enumerate(sub):
                                seq_len = len(sequences[seq_idx])
                                rep = sub_reps[j, 1 : 1 + seq_len].mean(0)
                                out_cpu[seq_idx].copy_(
                                    rep.float().to("cpu", non_blocking=True)
                                )
                            pbar.update(len(sub))
                        continue
                    raise

                reps = out["representations"][layer]  # (B, L, D)

                for j, seq_idx in enumerate(batch_indices):
                    seq_len = len(sequences[seq_idx])
                    rep = reps[j, 1 : 1 + seq_len].mean(0)
                    out_cpu[seq_idx].copy_(rep.float().to("cpu", non_blocking=True))

                pbar.update(len(batch_indices))

    return out_cpu.numpy()


# ------------------------- main -------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Extract ESM1v embeddings for sequences in a CSV (fast batching + H100 defaults)."
    )
    parser.add_argument(
        "--input-csv",
        type=str,
        required=True,
        help="Path to input CSV (e.g., activity_dataset_clean.csv)",
    )
    parser.add_argument(
        "--seq-col",
        type=str,
        default="Sequence_clean",
        help="Name of sequence column in CSV",
    )
    parser.add_argument(
        "--output-npz",
        type=str,
        required=True,
        help="Path to output .npz file with embeddings",
    )
    parser.add_argument(
        "--model-names",
        type=str,
        default="esm1v_t33_650M_UR90S_1",
        help=(
            "Comma-separated list of ESM1v model names from esm.pretrained. "
            "Default: esm1v_t33_650M_UR90S_1. "
            "For ensemble, pass several names separated by commas."
        ),
    )

    # Batching controls
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Fixed batch size (used when --max-tokens=0).",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=0,
        help=(
            "If >0, use token-budget batching so sum(len(seq)+2) per batch <= max_tokens "
            "(usually faster than fixed batch size, especially on GPU)."
        ),
    )
    parser.add_argument(
        "--min-batch-size",
        type=int,
        default=1,
        help="Minimum batch size when auto-reducing after CUDA OOM (fixed-batch mode only).",
    )

    # Device / precision
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU even if CUDA is available",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bf16",
        choices=["bf16", "fp16", "fp32"],
        help="CUDA autocast dtype. H100: bf16 recommended. fp32 disables autocast.",
    )
    parser.add_argument(
        "--no-cuda-amp",
        action="store_true",
        help="Disable CUDA autocast mixed precision.",
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        help="Use torch.compile(model) (PyTorch 2.x). May speed up long runs.",
    )

    # OOM fallback
    parser.add_argument(
        "--cpu-fallback-on-oom",
        action="store_true",
        help="If CUDA still OOMs, retry that model on CPU.",
    )
    parser.add_argument(
        "--cpu-fallback-batch-size",
        type=int,
        default=8,
        help="Batch size to use when falling back to CPU after CUDA OOM.",
    )

    # CPU threading
    parser.add_argument(
        "--num-threads",
        type=int,
        default=0,
        help=(
            "Torch intra-op CPU threads (used when running on CPU). "
            "Use 0 to auto-select min(8, os.cpu_count())."
        ),
    )
    parser.add_argument(
        "--num-interop-threads",
        type=int,
        default=1,
        help="Torch inter-op CPU threads (used when running on CPU). Default: 1",
    )

    args = parser.parse_args()

    print_header("Loading CSV")
    print("Input CSV:", args.input_csv)
    df = pd.read_csv(args.input_csv)

    if args.seq_col not in df.columns:
        raise ValueError(f"Sequence column '{args.seq_col}' not found in CSV.")

    df = df[df[args.seq_col].notna()].copy()
    print("Rows with non-null sequences:", df.shape[0])

    unique_seqs = df[args.seq_col].unique().tolist()
    print("Unique sequences:", len(unique_seqs))

    seq_to_index = {seq: idx for idx, seq in enumerate(unique_seqs)}

    # Device selection + torch config
    if torch.cuda.is_available() and not args.cpu:
        device = torch.device("cuda")
        print("Using CUDA")
        configure_torch_for_cuda()
        # Prevent CPU thread contention while GPU is working
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
    else:
        device = torch.device("cpu")
        print("Using CPU")
        cpu_count = os.cpu_count() or 8
        num_threads = args.num_threads if args.num_threads > 0 else min(8, cpu_count)
        configure_torch_for_cpu(num_threads=num_threads, num_interop_threads=args.num_interop_threads)

    model_names = [m.strip() for m in args.model_names.split(",") if m.strip()]
    if not model_names:
        raise ValueError("No model names provided via --model-names.")
    num_models = len(model_names)

    print_header("ESM1v models to use")
    for m in model_names:
        print(" -", m)

    # Precision
    amp_dtype = pick_autocast_dtype(args.dtype)
    use_cuda_amp = (not args.no_cuda_amp)

    emb_sum = None
    alphabet = None

    for mi, model_name in enumerate(model_names):
        print_header(f"Loading ESM1v model [{mi+1}/{num_models}]: {model_name}")
        model, alphabet_m = load_esm1v_model(model_name, device)

        if args.compile:
            try:
                model = torch.compile(model, mode="max-autotune", fullgraph=False)
                print("[INFO] torch.compile enabled")
            except Exception as e:
                print(f"[WARN] torch.compile failed; continuing uncompiled: {e}")

        if alphabet is None:
            alphabet = alphabet_m
        else:
            if alphabet_m.tok_to_idx != alphabet.tok_to_idx:
                print(
                    f"[WARN] Alphabet for '{model_name}' differs from first model's alphabet. "
                    "Using the first one for batch conversion."
                )

        print_header(f"Extracting embeddings with {model_name}")
        try:
            emb_model = extract_embeddings_for_sequences(
                unique_seqs,
                model,
                alphabet,
                device=device,
                batch_size=args.batch_size,
                min_batch_size=args.min_batch_size,
                use_cuda_amp=use_cuda_amp,
                layer=None,
                max_tokens=args.max_tokens,
                amp_dtype=amp_dtype,
            )
        except RuntimeError as exc:
            if not (
                args.cpu_fallback_on_oom
                and device.type == "cuda"
                and is_cuda_oom_error(exc)
            ):
                raise

            print(
                f"[WARN] CUDA OOM persisted for model {model_name}. Falling back to CPU for this model."
            )
            del model
            torch.cuda.empty_cache()

            cpu_device = torch.device("cpu")
            # CPU thread config for fallback
            cpu_count = os.cpu_count() or 8
            num_threads = args.num_threads if args.num_threads > 0 else min(8, cpu_count)
            configure_torch_for_cpu(num_threads=num_threads, num_interop_threads=args.num_interop_threads)

            model, alphabet_m = load_esm1v_model(model_name, cpu_device)
            emb_model = extract_embeddings_for_sequences(
                unique_seqs,
                model,
                alphabet_m,
                device=cpu_device,
                batch_size=args.cpu_fallback_batch_size,
                min_batch_size=1,
                use_cuda_amp=False,
                layer=None,
                max_tokens=max(0, min(args.max_tokens, 3000)) if args.max_tokens else 0,
                amp_dtype=None,
            )

        print(f"Embeddings from {model_name}: shape {emb_model.shape}")

        if emb_sum is None:
            emb_sum = emb_model.astype(np.float32, copy=False)
        else:
            emb_sum += emb_model.astype(np.float32, copy=False)

        # Free memory
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    embeddings = emb_sum / float(num_models)

    print_header("Final averaged embeddings")
    print("Embeddings shape:", embeddings.shape, "dtype:", embeddings.dtype)

    out_dir = os.path.dirname(args.output_npz)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    print_header("Saving to NPZ")
    seq_to_index_json = json.dumps(seq_to_index)

    np.savez_compressed(
        args.output_npz,
        embeddings=embeddings,
        sequences=np.array(unique_seqs, dtype=object),
        seq_to_index=seq_to_index_json,
    )
    print("Saved embeddings to:", args.output_npz)

    print_header("Cluster cache tip")
    print("To speed up model downloads on a cluster, set one of these before running:")
    print("  export TORCH_HOME=/shared/cache/torch    # shared cache across jobs")
    print("  export TORCH_HOME=$SLURM_TMPDIR/torch    # node-local cache (fast)")
    print("Optionally warm cache once per node:")
    print("  python -c \"import esm; esm.pretrained.esm1v_t33_650M_UR90S_1()\"")

    print_header("Done")


if __name__ == "__main__":
    main()
