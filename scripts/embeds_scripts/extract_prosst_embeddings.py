#!/usr/bin/env python
"""
Extract ProSST embeddings for PETase sequences.

Example:

python scripts/extract_prosst_embeddings.py \
  --input-csv data_processed/activity_dataset_clean.csv \
  --seq-col Sequence_clean \
  --output-npz embeddings/prosst_activity_unique_seqs_2048.npz \
  --model-id AI4Protein/ProSST-2048 \
  --batch-size 8
"""

import argparse
import json
import os

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForMaskedLM


def print_header(title: str):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def load_prosst(model_id: str, device: torch.device):
    """
    Load ProSST model + tokenizer from Hugging Face.

    model_id: e.g. "AI4Protein/ProSST-2048"
    """
    print(f"Loading tokenizer from {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=True,
    )

    print(f"Loading model from {model_id}")
    model = AutoModelForMaskedLM.from_pretrained(
        model_id,
        trust_remote_code=True,
    )
    model.to(device)
    model.eval()
    return tokenizer, model


def extract_prosst_embeddings(
    sequences,
    tokenizer,
    model,
    device,
    batch_size: int = 8,
):
    """
    sequences: list of str (unique sequences)

    Returns: np.ndarray of shape (N, D)
             D = hidden size of ProSST
    """
    all_embs = []

    with torch.no_grad():
        for i in tqdm(
            range(0, len(sequences), batch_size),
            desc="Embedding batches",
            total=(len(sequences) + batch_size - 1) // batch_size,
        ):
            batch_seqs = sequences[i : i + batch_size]

            enc = tokenizer(
                batch_seqs,
                padding=True,
                truncation=True,
                return_tensors="pt",
            )
            enc = {k: v.to(device) for k, v in enc.items()}

            # ProSST expects ss_input_ids; we don't have SS, so use zeros as dummy.
            ss_input_ids = torch.zeros_like(enc["input_ids"]).to(device)

            outputs = model(
                input_ids=enc["input_ids"],
                attention_mask=enc["attention_mask"],
                ss_input_ids=ss_input_ids,
                output_hidden_states=True,   # we need hidden_states
                return_dict=True,
            )

            # Use final layer from hidden_states tuple
            hidden = outputs.hidden_states[-1]           # (batch, seq_len, hidden_dim)
            mask = enc["attention_mask"].unsqueeze(-1)   # (batch, seq_len, 1)

            # Mean-pool over non-padding tokens
            masked_hidden = hidden * mask
            sum_hidden = masked_hidden.sum(dim=1)        # (batch, hidden_dim)
            lengths = mask.sum(dim=1).clamp(min=1)       # (batch, 1)
            avg_hidden = sum_hidden / lengths            # (batch, hidden_dim)

            all_embs.append(avg_hidden.cpu())

    embs = torch.cat(all_embs, dim=0).numpy()
    return embs


def main():
    parser = argparse.ArgumentParser(
        description="Extract ProSST embeddings for sequences in a CSV."
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
        help="Name of sequence column in CSV (default: Sequence_clean)",
    )
    parser.add_argument(
        "--output-npz",
        type=str,
        required=True,
        help="Path to output .npz file with embeddings",
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default="AI4Protein/ProSST-2048",
        help="HuggingFace model ID (default: AI4Protein/ProSST-2048)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for embedding extraction (default: 8)",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU even if CUDA is available",
    )

    args = parser.parse_args()

    print_header("Loading CSV")
    print("Input CSV:", args.input_csv)
    df = pd.read_csv(args.input_csv)

    if args.seq_col not in df.columns:
        raise ValueError(
            f"Sequence column '{args.seq_col}' not found in CSV."
        )

    df = df[df[args.seq_col].notna()].copy()
    print("Rows with non-null sequences:", df.shape[0])

    unique_seqs = df[args.seq_col].unique().tolist()
    print("Unique sequences:", len(unique_seqs))

    # Map sequence -> index in unique_seqs (for later alignment)
    seq_to_index = {seq: idx for idx, seq in enumerate(unique_seqs)}

    # Device
    if torch.cuda.is_available() and not args.cpu:
        device = torch.device("cuda")
        print("Using CUDA")
    else:
        device = torch.device("cpu")
        print("Using CPU")

    print_header(f"Loading ProSST model ({args.model_id})")
    tokenizer, model = load_prosst(args.model_id, device)

    print_header("Extracting ProSST embeddings")
    embeddings = extract_prosst_embeddings(
        unique_seqs,
        tokenizer,
        model,
        device=device,
        batch_size=args.batch_size,
    )
    print("Embeddings shape:", embeddings.shape)

    # Ensure output directory exists
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

    print_header("Done")


if __name__ == "__main__":
    main()
