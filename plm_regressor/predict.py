#!/usr/bin/env python3
"""Score / rank candidate sequences with a saved run (tabular-aware)."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import joblib
import numpy as np
import pandas as pd

from .core import (
    TABULAR_FEATURE_SOURCE,
    add_rank_column,
    assemble_feature_matrices,
    build_prediction_dataframe,
    load_embedding_banks,
    load_json,
    transform_feature_mode,
)


def write_top_tables(df: pd.DataFrame, out_dir: Path, top_n: int) -> None:
    valid = df[df["rank"].notna()].copy()
    sizes: list[int] = []
    for size in [10, 50, 100, top_n]:
        if size not in sizes:
            sizes.append(size)
    for size in sizes:
        valid.head(size).to_csv(out_dir / f"top_{size}.csv", index=False)


def score_candidates_from_run(
    run_dir: str | Path,
    candidate_csv: str | Path,
    predict_seq_col: Optional[str] = None,
    candidate_embedding_dir: Optional[str] = None,
    top_n: int = 100,
    out_dir: Optional[str | Path] = None,
    ascending: bool = False,
) -> Path:
    run_dir = Path(run_dir)
    summary = load_json(run_dir / "run_summary.json")
    best_model = joblib.load(run_dir / "best_model.joblib")
    uncertainty_enabled = bool(summary.get("uncertainty_enabled", True))

    ensemble = None
    ensemble_path = run_dir / "uncertainty_ensemble.joblib"
    if uncertainty_enabled and ensemble_path.exists():
        ensemble = joblib.load(ensemble_path)

    tabular_encoder = None
    tab_path = run_dir / "tabular_encoder.joblib"
    if summary.get("has_tabular") and tab_path.exists():
        tabular_encoder = joblib.load(tab_path)

    candidate_df = pd.read_csv(candidate_csv)
    predict_seq_col = predict_seq_col or summary.get("predict_seq_col") or summary["train_seq_col"]
    if predict_seq_col not in candidate_df.columns:
        raise ValueError(f"Prediction sequence column '{predict_seq_col}' not found in {candidate_csv}")

    feature_sources = summary.get("deployment_feature_sources") or summary["feature_sources"]
    embedding_dir = candidate_embedding_dir or summary.get("predict_embedding_dir") or summary.get("embedding_dir")
    embedding_banks = load_embedding_banks(feature_sources=feature_sources, embedding_dir=embedding_dir)

    tabular_matrix = None
    if TABULAR_FEATURE_SOURCE in [s.lower() for s in feature_sources]:
        if tabular_encoder is None:
            raise ValueError("Run uses tabular features but no tabular_encoder.joblib was found")
        tabular_matrix = tabular_encoder.transform(candidate_df)

    X_raw, missing_any, _, _ = assemble_feature_matrices(
        df=candidate_df, seq_col=predict_seq_col, feature_sources=feature_sources,
        embedding_banks=embedding_banks, expected_sequence_length=summary.get("expected_sequence_length"),
        tabular_matrix=tabular_matrix,
    )

    feature_modes = {best_model.feature_mode}
    if ensemble is not None:
        feature_modes |= {m.feature_mode for m in ensemble.fitted_models}
    X_by_mode: Dict[str, Dict[str, np.ndarray]] = {
        mode: transform_feature_mode(X_raw, None, mode, best_model.wt_by_source or None)
        for mode in sorted(feature_modes)
    }

    valid = ~missing_any
    n = len(candidate_df)
    y_pred = np.full(n, np.nan, dtype=float)
    native_std = np.full(n, np.nan, dtype=float)
    ensemble_mean = np.full(n, np.nan, dtype=float)
    ensemble_std = np.full(n, np.nan, dtype=float)

    if valid.any():
        best_pred, best_native = best_model.predict_with_uncertainty(
            {k: v[valid] for k, v in X_by_mode[best_model.feature_mode].items()}
        )
        y_pred[valid] = best_pred
        if uncertainty_enabled and best_native is not None:
            native_std[valid] = best_native
        if uncertainty_enabled and ensemble is not None:
            em, es = ensemble.predict_with_uncertainty(
                {mode: {k: v[valid] for k, v in smap.items()} for mode, smap in X_by_mode.items()}
            )
            ensemble_mean[valid] = em
            ensemble_std[valid] = es

    seen = candidate_df[predict_seq_col].astype(str).str.strip().isin(set(best_model.train_sequences)).to_numpy()
    ranked = build_prediction_dataframe(
        df=candidate_df, run_name=summary["run_name"], y_pred=y_pred, conformal_qhat=best_model.conformal_qhat,
        ensemble_mean=ensemble_mean, native_std=native_std, ensemble_std=ensemble_std,
        seen_in_train=seen, missing_any_feature=missing_any, ascending=ascending,
    )

    out_dir = Path(out_dir) if out_dir else run_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "candidate_predictions.csv"
    ranked.to_csv(out_path, index=False)
    write_top_tables(ranked, out_dir=out_dir, top_n=top_n)
    print(f"Saved ranked candidates to: {out_path}")
    return out_path


def rerank_predictions(pred_csv: str | Path, top_n: int = 100, ascending: bool = False,
                       out_dir: Optional[str | Path] = None) -> Path:
    pred_csv = Path(pred_csv)
    df = pd.read_csv(pred_csv)
    if "y_pred" not in df.columns:
        raise ValueError(f"{pred_csv} must contain a 'y_pred' column")
    ranked = add_rank_column(df, pred_col="y_pred", ascending=ascending)
    out_dir = Path(out_dir) if out_dir else pred_csv.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "ranked_predictions.csv"
    ranked.to_csv(out_path, index=False)
    write_top_tables(ranked, out_dir=out_dir, top_n=top_n)
    print(f"Saved reranked predictions to: {out_path}")
    return out_path
