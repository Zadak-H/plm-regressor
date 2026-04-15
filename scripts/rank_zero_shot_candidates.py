#!/usr/bin/env python3
"""
Rank zero-shot candidates from a saved MLDE run or from an existing prediction CSV.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import joblib
import numpy as np
import pandas as pd

from mlde_utils import (
    add_rank_column,
    assemble_feature_matrices,
    build_prediction_dataframe,
    load_embedding_banks,
    load_json,
    transform_feature_mode,
)


def write_top_tables(df: pd.DataFrame, out_dir: Path, top_n: int) -> None:
    valid = df[df["rank"].notna()].copy()
    requested_sizes = []
    for size in [10, 50, 100, top_n]:
        if size not in requested_sizes:
            requested_sizes.append(size)
    for size in requested_sizes:
        valid.head(size).to_csv(out_dir / f"top_{size}.csv", index=False)


def score_candidates(args: argparse.Namespace) -> Path:
    run_dir = Path(args.run_dir)
    summary = load_json(run_dir / "run_summary.json")
    best_model = joblib.load(run_dir / "best_model.joblib")
    uncertainty_enabled = bool(summary.get("uncertainty_enabled", True))
    ensemble = None
    ensemble_path = run_dir / "uncertainty_ensemble.joblib"
    if uncertainty_enabled and ensemble_path.exists():
        ensemble = joblib.load(ensemble_path)

    candidate_csv = Path(args.candidate_csv)
    candidate_df = pd.read_csv(candidate_csv)
    predict_seq_col = args.predict_seq_col or summary["predict_seq_col"]
    if predict_seq_col not in candidate_df.columns:
        raise ValueError(f"Prediction sequence column '{predict_seq_col}' not found in {candidate_csv}")

    feature_sources = summary.get("deployment_feature_sources") or summary["feature_sources"]
    candidate_embedding_dir = args.candidate_embedding_dir or summary.get("predict_embedding_dir")
    embedding_banks = load_embedding_banks(
        feature_sources=feature_sources,
        embedding_dir=candidate_embedding_dir,
        explicit_embedding_paths=None,
    )
    X_raw, missing_any, _, _ = assemble_feature_matrices(
        df=candidate_df,
        seq_col=predict_seq_col,
        feature_sources=feature_sources,
        embedding_banks=embedding_banks,
        expected_sequence_length=summary.get("expected_sequence_length"),
    )

    feature_modes = {best_model.feature_mode}
    if ensemble is not None:
        feature_modes |= {member.feature_mode for member in ensemble.fitted_models}
    feature_modes = sorted(feature_modes)
    X_by_mode: Dict[str, Dict[str, np.ndarray]] = {
        mode: transform_feature_mode(
            X_raw,
            wt_index=None,
            feature_mode=mode,
            wt_by_source=best_model.wt_by_source or None,
        )
        for mode in feature_modes
    }

    valid = ~missing_any
    y_pred = np.full(len(candidate_df), np.nan, dtype=float)
    native_std = np.full(len(candidate_df), np.nan, dtype=float)
    ensemble_mean = np.full(len(candidate_df), np.nan, dtype=float)
    ensemble_std = np.full(len(candidate_df), np.nan, dtype=float)

    if valid.any():
        best_pred, best_native_std = best_model.predict_with_uncertainty(
            {key: value[valid] for key, value in X_by_mode[best_model.feature_mode].items()}
        )
        y_pred[valid] = best_pred
        if uncertainty_enabled and best_native_std is not None:
            native_std[valid] = best_native_std
        if uncertainty_enabled and ensemble is not None:
            ens_mean_valid, ens_std_valid = ensemble.predict_with_uncertainty(
                {mode: {key: value[valid] for key, value in source_map.items()} for mode, source_map in X_by_mode.items()}
            )
            ensemble_mean[valid] = ens_mean_valid
            ensemble_std[valid] = ens_std_valid

    seen_in_train = candidate_df[predict_seq_col].astype(str).str.strip().isin(set(best_model.train_sequences)).to_numpy()
    ranked = build_prediction_dataframe(
        df=candidate_df,
        run_name=summary["run_name"],
        y_pred=y_pred,
        conformal_qhat=best_model.conformal_qhat,
        ensemble_mean=ensemble_mean,
        native_std=native_std,
        ensemble_std=ensemble_std,
        seen_in_train=seen_in_train,
        missing_any_feature=missing_any,
        ascending=args.ascending,
    )

    out_dir = Path(args.out_dir) if args.out_dir else run_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "candidate_predictions.csv"
    ranked.to_csv(out_path, index=False)
    write_top_tables(ranked, out_dir=out_dir, top_n=args.top_n)
    print(f"Saved ranked candidates to: {out_path}")
    return out_path


def rerank_existing_predictions(args: argparse.Namespace) -> Path:
    pred_csv = Path(args.pred_csv)
    df = pd.read_csv(pred_csv)
    if "y_pred" not in df.columns:
        raise ValueError(f"{pred_csv} must contain a 'y_pred' column")
    ranked = add_rank_column(df, pred_col="y_pred", ascending=args.ascending)
    out_dir = Path(args.out_dir) if args.out_dir else pred_csv.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "ranked_predictions.csv"
    ranked.to_csv(out_path, index=False)
    write_top_tables(ranked, out_dir=out_dir, top_n=args.top_n)
    print(f"Saved reranked predictions to: {out_path}")
    return out_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", default=None, help="Run directory containing saved model artifacts")
    parser.add_argument("--candidate-csv", default=None, help="Candidate CSV to score with the saved run")
    parser.add_argument("--predict-seq-col", default=None, help="Candidate sequence column override")
    parser.add_argument("--predict-id-col", default=None, help="Candidate ID column override")
    parser.add_argument("--candidate-embedding-dir", default=None, help="Directory containing candidate embedding NPZs")
    parser.add_argument("--pred-csv", default=None, help="Existing prediction CSV to rerank")
    parser.add_argument("--top-n", type=int, default=100)
    parser.add_argument("--ascending", action="store_true", help="Rank smaller predictions first")
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()

    if bool(args.run_dir) == bool(args.pred_csv):
        raise ValueError("Provide exactly one of --run-dir or --pred-csv")
    if args.run_dir and args.candidate_csv is None:
        raise ValueError("--candidate-csv is required with --run-dir")

    if args.run_dir:
        score_candidates(args)
    else:
        rerank_existing_predictions(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
