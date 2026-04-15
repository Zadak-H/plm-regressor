#!/usr/bin/env python
"""
Extract ESM2 embeddings for PETase sequences.

Example:

python scripts/extract_esm2_embeddings.py \
  --input-csv data_processed/activity_dataset_clean.csv \
  --seq-col Sequence_clean \
  --output-npz embeddings/esm2_activity_unique_seqs_650M.npz \
  --model-size 650M \
  --batch-size 8
"""

import argparse
import json
import os

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

import esm  # make sure this is from `fair-esm`


def print_header(title):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def load_esm2_model(model_size):
    """
    model_size: one of ["8M", "35M", "150M", "650M", "3B"]
    Returns: model, alphabet

    Tries esm.pretrained first; if unavailable, falls back to torch.hub.load().
    """
    size = model_size.upper()

    # Map our size string to the canonical model name used by esm / torch.hub
    size_to_name = {
        "8M": "esm2_t6_8M_UR50D",
        "35M": "esm2_t12_35M_UR50D",
        "150M": "esm2_t30_150M_UR50D",
        "650M": "esm2_t33_650M_UR50D",
        "3B": "esm2_t36_3B_UR50D",
    }
    if size not in size_to_name:
        raise ValueError(
            "Unsupported model_size '{}'. Use one of: 8M, 35M, 150M, 650M, 3B".format(
                model_size
            )
        )
    model_name = size_to_name[size]

    # ---- Try esm.pretrained (preferred if fair-esm is installed correctly) ----
    try:
        if hasattr(esm, "pretrained"):
            print("Using esm.pretrained.{}()".format(model_name))
            load_fn = getattr(esm.pretrained, model_name)
            model, alphabet = load_fn()
            return model, alphabet
        else:
            print("[WARN] esm has no attribute 'pretrained'; falling back to torch.hub.")
    except Exception as e:
        print(
            "[WARN] Failed to load via esm.pretrained ({}). "
            "Falling back to torch.hub.load(...).".format(repr(e))
        )

    # ---- Fallback: torch.hub.load from GitHub repo ----
    print("Loading model via torch.hub.load('facebookresearch/esm:main', '{}')".format(
        model_name
    ))
    model, alphabet = torch.hub.load(
        "facebookresearch/esm:main", model_name
    )
    return model, alphabet


def extract_embeddings_for_sequences(
    sequences,
    model,
    alphabet,
    device,
    batch_size=8,
    layer=None,
):
    """
    sequences: list of str (unique sequences)
    Returns: np.ndarray of shape (N, D)
    """
    model.eval()
    model = model.to(device)

    if layer is None:
        layer = model.num_layers  # final layer

    batch_converter = alphabet.get_batch_converter()

    all_embeddings = []
    with torch.no_grad():
        for i in tqdm(
            range(0, len(sequences), batch_size),
            desc="Embedding batches",
            total=(len(sequences) + batch_size - 1) // batch_size,
        ):
            batch_seqs = sequences[i : i + batch_size]
            data = [("seq_{}".format(j), s) for j, s in enumerate(batch_seqs)]
            _, _, tokens = batch_converter(data)
            tokens = tokens.to(device)

            out = model(tokens, repr_layers=[layer], return_contacts=False)
            token_representations = out["representations"][layer]

            for j, seq in enumerate(batch_seqs):
                seq_len = len(seq)
                rep = token_representations[j, 1 : 1 + seq_len].mean(0)
                all_embeddings.append(rep.cpu())

    embeddings = torch.stack(all_embeddings, dim=0).numpy()
    return embeddings


def main():
    parser = argparse.ArgumentParser(
        description="Extract ESM2 embeddings for sequences in a CSV."
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
        "--model-size",
        type=str,
        default="650M",
        help="ESM2 model size: one of [8M, 35M, 150M, 650M, 3B]. Default: 650M",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for embedding extraction",
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
            "Sequence column '{}' not found in CSV".format(args.seq_col)
        )

    df = df[df[args.seq_col].notna()].copy()
    print("Rows with non-null sequences:", df.shape[0])

    unique_seqs = df[args.seq_col].unique().tolist()
    print("Unique sequences:", len(unique_seqs))

    seq_to_index = {seq: idx for idx, seq in enumerate(unique_seqs)}

    if torch.cuda.is_available() and not args.cpu:
        device = torch.device("cuda")
        print("Using CUDA")
    else:
        device = torch.device("cpu")
        print("Using CPU")

    print_header("Loading ESM2 model ({})".format(args.model_size))
    model, alphabet = load_esm2_model(args.model_size)

    print_header("Extracting embeddings")
    embeddings = extract_embeddings_for_sequences(
        unique_seqs,
        model,
        alphabet,
        device=device,
        batch_size=args.batch_size,
        layer=None,
    )
    print("Embeddings shape:", embeddings.shape)

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
