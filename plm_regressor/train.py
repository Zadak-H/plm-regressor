#!/usr/bin/env python3
"""Training orchestrator: RunConfig -> standardized run directory + report.

Generalizes the original ``training3_optuna_mlde_uncertainty.py`` with:
- the size engine (CV strategy / trial budget / model gating / subsample tuning)
- the model registry (classical + deep)
- extra "tabular" feature columns
- the expanded metric/plot report
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import optuna
import pandas as pd
from optuna.samplers import TPESampler

from .config import RunConfig
from .core import (
    TABULAR_FEATURE_SOURCE,
    FittedEnsemble,
    FittedRunModel,
    all_nonempty_feature_subsets,
    assemble_feature_matrices,
    build_oof_dataframe,
    build_prediction_dataframe,
    conformal_q,
    json_default,
    load_embedding_banks,
    make_cv_splits,
    normalize_feature_source,
    prepare_supervised_dataframe,
    print_header,
    save_json,
    seed_everything,
    transform_feature_mode,
)
from .features import build_tabular_encoder
from .metrics import metric_direction, sort_descending_for_metric
from .registry import MODEL_REGISTRY, eligible_models
from .search import (
    SearchContext,
    build_fixed_trial_estimator,
    cross_val_predict_with_uncertainty,
    make_objective,
)
from .sizing import SizeProfile, profile_for_n


def _resolve_profile(cfg: RunConfig, n: int) -> SizeProfile:
    profile = profile_for_n(n)
    if not cfg.auto_size:
        # honor manual overrides while keeping size-derived big-data safeguards
        profile.cv_strategy = cfg.cv_strategy
        profile.cv_splits = cfg.cv_splits
    if cfg.n_trials is not None:
        profile.trial_budget = int(cfg.n_trials)
    return profile


def run_training(cfg: RunConfig) -> Path:
    cfg.validate()
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(cfg.random_state)

    feature_sources = [normalize_feature_source(s) for s in cfg.feature_sources]
    has_tabular = TABULAR_FEATURE_SOURCE in feature_sources

    print_header("Loading supervised CSV")
    raw_df = pd.read_csv(cfg.csv)
    if cfg.seq_col not in raw_df.columns:
        raise ValueError(f"Sequence column '{cfg.seq_col}' not found in {cfg.csv}")
    if cfg.target_col not in raw_df.columns:
        raise ValueError(f"Target column '{cfg.target_col}' not found in {cfg.csv}")

    keep_cols = list(cfg.extra_feature_cols) + list(cfg.categorical_cols)
    prepared_df = prepare_supervised_dataframe(
        df=raw_df,
        seq_col=cfg.seq_col,
        target_col=cfg.target_col,
        replicate_policy=cfg.replicate_policy,
        id_col=cfg.id_col,
        keep_cols=keep_cols,
    )
    y = prepared_df[cfg.target_col].astype(float).to_numpy()

    if cfg.group_col and cfg.group_col in prepared_df.columns:
        groups = prepared_df[cfg.group_col].astype(str).to_numpy()
        group_col_used = cfg.group_col
    else:
        groups = prepared_df[cfg.seq_col].astype(str).to_numpy()
        group_col_used = cfg.seq_col

    print(f"Rows in raw training CSV: {len(raw_df)}")
    print(f"Rows after replicate policy '{cfg.replicate_policy}': {len(prepared_df)}")

    # tabular encoder (fit on training frame only)
    tabular_encoder = None
    if has_tabular:
        tabular_encoder = build_tabular_encoder(prepared_df, cfg.extra_feature_cols, cfg.categorical_cols)
        if tabular_encoder is None:
            raise ValueError("tabular feature source requested but no extra/categorical columns provided")

    print_header("Loading feature banks")
    profile = _resolve_profile(cfg, len(prepared_df))
    print(f"Dataset size profile: {profile.describe()}")
    embedding_banks = load_embedding_banks(
        feature_sources=feature_sources,
        embedding_dir=cfg.embedding_dir,
        explicit_embedding_paths=None,
        mmap=profile.mmap_embeddings,
    )
    for name, bank in embedding_banks.items():
        print(f"- {name}: dim={bank.dim} | {bank.path}")

    tabular_matrix = tabular_encoder.transform(prepared_df) if tabular_encoder is not None else None
    X_by_source_raw, missing_any, source_missing_counts, expected_sequence_length = assemble_feature_matrices(
        df=prepared_df,
        seq_col=cfg.seq_col,
        feature_sources=feature_sources,
        embedding_banks=embedding_banks,
        tabular_matrix=tabular_matrix,
    )
    rows_before_feature_drop = len(prepared_df)
    if missing_any.any():
        print(f"Dropping {int(missing_any.sum())} training rows with missing learned features")
        keep = ~missing_any
        prepared_df = prepared_df.loc[keep].reset_index(drop=True)
        y = y[keep]
        groups = groups[keep]
        for source in list(X_by_source_raw.keys()):
            X_by_source_raw[source] = X_by_source_raw[source][keep]
    if len(prepared_df) < 2:
        raise RuntimeError("Not enough rows remain after feature coverage filtering")

    # WT handling for delta modes
    wt_index: Optional[int] = None
    wt_by_source: Optional[Dict[str, np.ndarray]] = None
    if cfg.wt_sequence is not None:
        wt_sequence = str(cfg.wt_sequence).strip()
        seqs_after_filter = prepared_df[cfg.seq_col].astype(str).tolist()
        if wt_sequence not in seqs_after_filter:
            raise ValueError("WT sequence was not found in the filtered training data")
        wt_index = seqs_after_filter.index(wt_sequence)
        wt_by_source = {source: matrix[wt_index].copy() for source, matrix in X_by_source_raw.items()}

    if cfg.standard_search:
        feature_modes = ["raw"]
    elif cfg.feature_mode_options:
        feature_modes = list(dict.fromkeys(cfg.feature_mode_options))
    elif wt_index is not None:
        feature_modes = ["raw", "delta", "raw_plus_delta"]
    else:
        feature_modes = ["raw"]
    if any(mode != "raw" for mode in feature_modes) and wt_index is None:
        raise ValueError("Delta-based feature modes require wt_sequence")

    X_by_mode: Dict[str, Dict[str, np.ndarray]] = {
        mode: transform_feature_mode(X_by_source_raw, wt_index, mode, wt_by_source) for mode in feature_modes
    }

    if cfg.standard_search:
        feature_subsets = [(source,) for source in feature_sources]
    else:
        feature_subsets = all_nonempty_feature_subsets(feature_sources)

    # model gating by dataset size
    n_rows = len(prepared_df)
    elig = eligible_models(cfg.models, n_rows)
    dropped = [m for m in cfg.models if m not in elig and m in MODEL_REGISTRY]
    if dropped:
        print(f"Models excluded for n={n_rows} (size/availability): {', '.join(dropped)}")
    if not elig:
        raise RuntimeError("No eligible models remain for this dataset size; relax the model list")

    # subsample for tuning on big data; deploy on full
    rng = np.random.RandomState(cfg.random_state)
    if profile.tune_subsample and n_rows > profile.tune_subsample:
        tune_idx = np.sort(rng.choice(n_rows, size=profile.tune_subsample, replace=False))
        print(f"Tuning on a random subsample of {len(tune_idx)} / {n_rows} rows; deploying on all rows")
    else:
        tune_idx = np.arange(n_rows)

    X_by_mode_tune = {mode: {s: X_by_mode[mode][s][tune_idx] for s in X_by_mode[mode]} for mode in feature_modes}
    y_tune = y[tune_idx]
    groups_tune = groups[tune_idx]
    splits, uses_group_cv = make_cv_splits(
        n_samples=len(tune_idx),
        groups=groups_tune,
        cv_splits=profile.cv_splits,
        random_state=cfg.random_state,
        strategy=profile.cv_strategy,
        n_repeats=profile.n_repeats,
        holdout_fraction=profile.holdout_fraction,
    )

    print_header("Search space summary")
    print(f"Feature sources: {', '.join(feature_sources)}")
    print(f"Feature subsets: {len(feature_subsets)} | Feature modes: {', '.join(feature_modes)}")
    print(f"Eligible models: {', '.join(elig)}")
    print(f"Metric: {cfg.metric} | CV: {profile.cv_strategy} ({len(splits)} folds) | trials: {profile.trial_budget}")

    ctx = SearchContext(
        X_by_mode=X_by_mode_tune,
        y=y_tune,
        groups=groups_tune,
        splits=splits,
        feature_subsets=feature_subsets,
        feature_modes=feature_modes,
        eligible_models=elig,
        metric_name=cfg.metric,
        random_state=cfg.random_state,
        use_gpu=cfg.use_gpu,
        standard_search=cfg.standard_search,
    )

    print_header("Running Optuna search")
    sampler = TPESampler(seed=cfg.random_state, multivariate=True)
    study = optuna.create_study(direction=metric_direction(cfg.metric), sampler=sampler)
    study.optimize(make_objective(ctx), n_trials=profile.trial_budget, timeout=cfg.timeout, show_progress_bar=True)

    complete = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if not complete:
        raise RuntimeError("No completed Optuna trials (all pruned/failed)")
    descending = sort_descending_for_metric(cfg.metric)
    complete = sorted(complete, key=lambda t: float(t.value), reverse=descending)
    best_trial = complete[0]
    best_subset = tuple(best_trial.user_attrs["feature_subset"])
    best_mode = str(best_trial.user_attrs["feature_mode"])

    # full + tuning matrices for the best subset/mode
    best_X_full = np.concatenate([X_by_mode[best_mode][s] for s in best_subset], axis=1)
    best_X_tune = np.concatenate([X_by_mode_tune[best_mode][s] for s in best_subset], axis=1)

    best_estimator, best_meta = build_fixed_trial_estimator(
        frozen_trial=best_trial,
        eligible_models=elig,
        X=best_X_tune,
        n_splits=len(splits),
        random_state=cfg.random_state + best_trial.number,
        use_gpu=cfg.use_gpu,
        standard_search=cfg.standard_search,
    )

    print_header("Evaluating best single model (OOF on tuning set)")
    best_metrics, best_oof, best_native_std, best_fold_ids, best_fold_rows = cross_val_predict_with_uncertainty(
        estimator=best_estimator, X=best_X_tune, y=y_tune, splits=splits
    )
    best_abs_resid = np.abs(y_tune - best_oof)
    if cfg.no_uncertainty:
        qhat = float("nan")
        best_native_std = np.full(len(best_native_std), np.nan, dtype=float)
    else:
        qhat = conformal_q(best_abs_resid, alpha=cfg.conformal_alpha)

    # search history
    history_rows = []
    for t in complete:
        history_rows.append(
            {
                "trial_number": t.number,
                "value": t.value,
                "feature_subset": "+".join(t.user_attrs.get("feature_subset", [])),
                "feature_mode": t.user_attrs.get("feature_mode"),
                "model_name": t.user_attrs.get("meta", {}).get("model_name"),
                **{f"metric__{k}": v for k, v in t.user_attrs.get("metrics", {}).items()},
                "meta_json": json.dumps(t.user_attrs.get("meta", {}), default=json_default),
            }
        )
    pd.DataFrame(history_rows).sort_values("value", ascending=not descending).to_csv(
        out_dir / "search_history.csv", index=False
    )

    print_header("Fitting best model (full data) and uncertainty ensemble")
    best_estimator.fit(best_X_full, y)
    best_model_artifact = FittedRunModel(
        run_name=out_dir.name,
        trial_number=best_trial.number,
        score=float(best_trial.value),
        metric_name=cfg.metric,
        model_name=best_meta["model_name"],
        feature_sources=best_subset,
        feature_mode=best_mode,
        estimator=best_estimator,
        conformal_alpha=cfg.conformal_alpha,
        conformal_qhat=qhat,
        train_sequences=tuple(prepared_df[cfg.seq_col].astype(str).tolist()),
        expected_sequence_length=expected_sequence_length,
        wt_by_source={} if wt_by_source is None else wt_by_source,
    )
    joblib.dump(best_model_artifact, out_dir / "best_model.joblib")
    if tabular_encoder is not None:
        joblib.dump(tabular_encoder, out_dir / "tabular_encoder.joblib")

    ensemble_artifact: Optional[FittedEnsemble] = None
    ensemble_members: List[FittedRunModel] = []
    if not cfg.no_uncertainty:
        ensemble_oof_predictions: List[np.ndarray] = []
        ensemble_rows: List[Dict[str, Any]] = []
        for t in complete[: max(1, min(cfg.top_ensemble, len(complete)))]:
            fsub = tuple(t.user_attrs["feature_subset"])
            fmode = str(t.user_attrs["feature_mode"])
            X_tr_tune = np.concatenate([X_by_mode_tune[fmode][s] for s in fsub], axis=1)
            X_tr_full = np.concatenate([X_by_mode[fmode][s] for s in fsub], axis=1)
            est_t, meta_t = build_fixed_trial_estimator(
                frozen_trial=t, eligible_models=elig, X=X_tr_tune, n_splits=len(splits),
                random_state=cfg.random_state + t.number, use_gpu=cfg.use_gpu, standard_search=cfg.standard_search,
            )
            _, oof_t, _, _, _ = cross_val_predict_with_uncertainty(est_t, X_tr_tune, y_tune, splits)
            ensemble_oof_predictions.append(oof_t)
            est_t.fit(X_tr_full, y)
            ensemble_members.append(
                FittedRunModel(
                    run_name=out_dir.name, trial_number=t.number, score=float(t.value), metric_name=cfg.metric,
                    model_name=meta_t["model_name"], feature_sources=fsub, feature_mode=fmode, estimator=est_t,
                    conformal_alpha=cfg.conformal_alpha, conformal_qhat=qhat,
                    train_sequences=tuple(prepared_df[cfg.seq_col].astype(str).tolist()),
                    expected_sequence_length=expected_sequence_length,
                    wt_by_source={} if wt_by_source is None else wt_by_source,
                )
            )
            ensemble_rows.append(
                {"trial_number": t.number, "score": float(t.value), "feature_subset": "+".join(fsub),
                 "feature_mode": fmode, "model_name": meta_t["model_name"],
                 "meta_json": json.dumps(meta_t, default=json_default)}
            )
        ensemble_artifact = FittedEnsemble(run_name=out_dir.name, fitted_models=ensemble_members)
        joblib.dump(ensemble_artifact, out_dir / "uncertainty_ensemble.joblib")
        pd.DataFrame(ensemble_rows).to_csv(out_dir / "top_ensemble_members.csv", index=False)
        deployment_feature_sources = sorted({s for m in ensemble_members for s in m.feature_sources})
        ensemble_oof_mean = np.mean(np.vstack(ensemble_oof_predictions), axis=0)
        ensemble_oof_std = np.std(np.vstack(ensemble_oof_predictions), axis=0, ddof=0)
    else:
        deployment_feature_sources = sorted(best_subset)
        ensemble_oof_mean = np.full(len(best_oof), np.nan, dtype=float)
        ensemble_oof_std = np.full(len(best_oof), np.nan, dtype=float)

    oof_df = build_oof_dataframe(
        df=prepared_df.iloc[tune_idx].reset_index(drop=True),
        run_name=out_dir.name, target_col=cfg.target_col, y_pred=best_oof, fold_ids=best_fold_ids,
        group_values=groups_tune, conformal_qhat=qhat, native_std=best_native_std,
        ensemble_mean=ensemble_oof_mean, ensemble_std=ensemble_oof_std,
    )
    oof_df.to_csv(out_dir / "oof_predictions.csv", index=False)
    pd.DataFrame(best_fold_rows).to_csv(out_dir / "fold_metrics.csv", index=False)

    # train-set predictions on full data
    train_pred, train_native_std = best_model_artifact.predict_with_uncertainty(X_by_mode[best_mode])
    if cfg.no_uncertainty or ensemble_artifact is None:
        train_native_std = np.full(len(train_pred), np.nan, dtype=float)
        train_ens_mean = np.full(len(train_pred), np.nan, dtype=float)
        train_ens_std = np.full(len(train_pred), np.nan, dtype=float)
    else:
        train_ens_mean, train_ens_std = ensemble_artifact.predict_with_uncertainty(X_by_mode)
    tp_input = prepared_df.copy()
    tp_input["y_true"] = y
    train_predictions = build_prediction_dataframe(
        df=tp_input, run_name=out_dir.name, y_pred=train_pred, conformal_qhat=qhat,
        ensemble_mean=train_ens_mean, native_std=train_native_std, ensemble_std=train_ens_std,
        seen_in_train=np.ones(len(prepared_df), dtype=bool),
        missing_any_feature=np.zeros(len(prepared_df), dtype=bool), ascending=False,
    )
    train_predictions["residual"] = train_predictions["y_true"] - train_predictions["y_pred"]
    train_predictions.to_csv(out_dir / "train_predictions.csv", index=False)

    save_json(out_dir / "coverage_report.json", {
        "feature_sources": feature_sources,
        "source_missing_counts_before_drop": source_missing_counts,
        "rows_before_feature_drop": rows_before_feature_drop,
        "rows_after_feature_drop": int(len(prepared_df)),
        "rows_dropped_for_missing_features": int(rows_before_feature_drop - len(prepared_df)),
        "expected_sequence_length": int(expected_sequence_length),
    })

    run_summary = {
        "run_name": out_dir.name, "csv": cfg.csv, "predict_csv": cfg.predict_csv,
        "train_seq_col": cfg.seq_col, "predict_seq_col": cfg.predict_seq_col or cfg.seq_col,
        "target_col": cfg.target_col, "id_col": cfg.id_col, "predict_id_col": cfg.predict_id_col,
        "group_col": group_col_used, "replicate_policy": cfg.replicate_policy,
        "feature_sources": feature_sources, "deployment_feature_sources": deployment_feature_sources,
        "extra_feature_cols": cfg.extra_feature_cols, "categorical_cols": cfg.categorical_cols,
        "has_tabular": has_tabular, "feature_modes_searched": feature_modes,
        "standard_search": bool(cfg.standard_search), "uncertainty_enabled": bool(not cfg.no_uncertainty),
        "size_tier": profile.tier, "cv_strategy": profile.cv_strategy, "trial_budget": profile.trial_budget,
        "tuned_on_rows": int(len(tune_idx)), "metric": cfg.metric, "metric_direction": metric_direction(cfg.metric),
        "eligible_models": elig, "models_requested": cfg.models, "models_dropped": dropped,
        "best_trial_number": int(best_trial.number), "best_trial_value": float(best_trial.value),
        "best_feature_subset": list(best_subset), "best_feature_mode": best_mode,
        "best_model_name": best_meta["model_name"], "best_params": best_trial.params, "best_meta": best_meta,
        "best_oof_metrics": best_metrics, "conformal_alpha": float(cfg.conformal_alpha), "conformal_qhat": qhat,
        "top_ensemble_size": int(len(ensemble_members)), "n_raw_rows": int(len(raw_df)),
        "n_training_rows": int(len(prepared_df)), "n_feature_subsets": int(len(feature_subsets)),
        "embedding_dir": cfg.embedding_dir, "predict_embedding_dir": cfg.predict_embedding_dir or cfg.embedding_dir,
        "expected_sequence_length": int(expected_sequence_length), "wt_sequence": cfg.wt_sequence,
        "uses_group_cv": bool(uses_group_cv), "coverage_report_file": "coverage_report.json",
    }
    save_json(out_dir / "run_summary.json", run_summary)
    cfg.to_yaml(out_dir / "run_config.yaml")

    # optional inline candidate scoring
    if cfg.predict_csv:
        from .predict import score_candidates_from_run

        print_header(f"Scoring candidates: {cfg.predict_csv}")
        score_candidates_from_run(
            run_dir=out_dir, candidate_csv=cfg.predict_csv,
            predict_seq_col=cfg.predict_seq_col, candidate_embedding_dir=cfg.predict_embedding_dir,
            top_n=100, out_dir=out_dir,
        )

    # report (plots + html); best-effort, never fails the run
    try:
        from .report import build_report

        build_report(out_dir, cfg.target_col, cfg.metric)
    except Exception as exc:  # pragma: no cover
        print(f"[warn] report generation skipped: {exc}")

    print_header("Done")
    print(f"Best {cfg.metric}: {best_trial.value:.4f} | model: {best_meta['model_name']} | "
          f"features: {'+'.join(best_subset)} ({best_mode})")
    print(f"Outputs: {out_dir}")
    return out_dir


def main(argv: Optional[List[str]] = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Train from a RunConfig YAML")
    parser.add_argument("config", help="Path to a RunConfig YAML")
    args = parser.parse_args(argv)
    run_training(RunConfig.from_yaml(args.config))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
