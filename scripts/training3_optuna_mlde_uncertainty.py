#!/usr/bin/env python3
"""
Optuna-driven MLDE training/search workflow with standardized outputs.

Highlights
----------
- Searches any non-empty subset of feature sources
- Supports Protein LKanguage Models embeddings plus one-hot and BLOSUM62 encodings
- Supports raw / delta / raw+delta feature modes when WT is supplied
- Uses OOF predictions for model selection
- Saves both a best single model and a top-model uncertainty ensemble
- Writes standardized CSV outputs for downstream comparison and zero-shot ranking
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator
from sklearn.compose import TransformedTargetRegressor
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.feature_selection import VarianceThreshold
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import BayesianRidge, ElasticNet, HuberRegressor, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import QuantileTransformer, RobustScaler, StandardScaler
from sklearn.svm import SVR

from mlde_utils import (
    FittedEnsemble,
    FittedRunModel,
    ColumnSelectorKBest,
    IdentityTransformer,
    all_nonempty_feature_subsets,
    assemble_feature_matrices,
    build_oof_dataframe,
    build_prediction_dataframe,
    compute_metrics,
    conformal_q,
    cross_val_predict_with_uncertainty,
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

try:
    import optuna
    from optuna.samplers import TPESampler
except Exception:
    optuna = None
    TPESampler = None

try:
    from xgboost import XGBRegressor

    HAS_XGB = True
except Exception:
    XGBRegressor = None
    HAS_XGB = False

try:
    from lightgbm import LGBMRegressor

    HAS_LGB = True
except Exception:
    LGBMRegressor = None
    HAS_LGB = False


def metric_direction(metric_name: str) -> str:
    return "minimize" if metric_name in {"rmse", "mae"} else "maximize"


def sort_descending_for_metric(metric_name: str) -> bool:
    return metric_direction(metric_name) == "maximize"


def make_model(
    trial: "optuna.trial.Trial",
    random_state: int,
    use_gpu: bool,
    n_features: int,
    n_samples_train: int,
) -> Tuple[str, BaseEstimator]:
    choices = [
        "ridge",
        "elasticnet",
        "huber",
        "bayesian_ridge",
        "svr_rbf",
        "knn",
        "mlp",
        "rf",
        "extra_trees",
        "hist_gb",
        "pls",
        "kernel_ridge",
        "gpr",
    ]
    if HAS_XGB:
        choices.append("xgboost")
    if HAS_LGB:
        choices.append("lightgbm")

    model_name = trial.suggest_categorical("model_name", choices)

    if model_name == "ridge":
        model = Ridge(alpha=trial.suggest_float("ridge_alpha", 1e-4, 1e3, log=True), random_state=random_state)
    elif model_name == "elasticnet":
        model = ElasticNet(
            alpha=trial.suggest_float("enet_alpha", 1e-5, 1e1, log=True),
            l1_ratio=trial.suggest_float("enet_l1_ratio", 0.05, 0.95),
            max_iter=20000,
            random_state=random_state,
        )
    elif model_name == "huber":
        model = HuberRegressor(
            epsilon=trial.suggest_float("huber_epsilon", 1.05, 2.0),
            alpha=trial.suggest_float("huber_alpha", 1e-6, 1e-1, log=True),
            max_iter=2000,
        )
    elif model_name == "bayesian_ridge":
        model = BayesianRidge(
            alpha_1=trial.suggest_float("br_alpha_1", 1e-8, 1e-2, log=True),
            alpha_2=trial.suggest_float("br_alpha_2", 1e-8, 1e-2, log=True),
            lambda_1=trial.suggest_float("br_lambda_1", 1e-8, 1e-2, log=True),
            lambda_2=trial.suggest_float("br_lambda_2", 1e-8, 1e-2, log=True),
        )
    elif model_name == "svr_rbf":
        model = SVR(
            kernel="rbf",
            C=trial.suggest_float("svr_C", 1e-2, 1e3, log=True),
            epsilon=trial.suggest_float("svr_epsilon", 1e-3, 1.0, log=True),
            gamma=trial.suggest_categorical("svr_gamma", ["scale", "auto"]),
        )
    elif model_name == "knn":
        model = KNeighborsRegressor(
            n_neighbors=trial.suggest_int("knn_n_neighbors", 2, 25),
            weights=trial.suggest_categorical("knn_weights", ["uniform", "distance"]),
            p=trial.suggest_int("knn_p", 1, 2),
        )
    elif model_name == "mlp":
        hidden = trial.suggest_categorical("mlp_hidden", ["64", "128", "256", "128_64", "256_128"])
        hidden_map = {
            "64": (64,),
            "128": (128,),
            "256": (256,),
            "128_64": (128, 64),
            "256_128": (256, 128),
        }
        model = MLPRegressor(
            hidden_layer_sizes=hidden_map[hidden],
            alpha=trial.suggest_float("mlp_alpha", 1e-6, 1e-1, log=True),
            learning_rate_init=trial.suggest_float("mlp_lr", 1e-4, 1e-2, log=True),
            max_iter=3000,
            early_stopping=True,
            random_state=random_state,
        )
    elif model_name == "rf":
        model = RandomForestRegressor(
            n_estimators=trial.suggest_int("rf_n_estimators", 100, 1000, step=100),
            max_depth=trial.suggest_int("rf_max_depth", 3, 20),
            min_samples_leaf=trial.suggest_int("rf_min_samples_leaf", 1, 10),
            max_features=trial.suggest_categorical("rf_max_features", ["sqrt", "log2", None]),
            n_jobs=1,
            random_state=random_state,
        )
    elif model_name == "extra_trees":
        model = ExtraTreesRegressor(
            n_estimators=trial.suggest_int("et_n_estimators", 100, 1000, step=100),
            max_depth=trial.suggest_int("et_max_depth", 3, 20),
            min_samples_leaf=trial.suggest_int("et_min_samples_leaf", 1, 10),
            max_features=trial.suggest_categorical("et_max_features", ["sqrt", "log2", None]),
            n_jobs=1,
            random_state=random_state,
        )
    elif model_name == "hist_gb":
        model = HistGradientBoostingRegressor(
            learning_rate=trial.suggest_float("hgb_learning_rate", 1e-3, 0.2, log=True),
            max_depth=trial.suggest_int("hgb_max_depth", 2, 12),
            max_leaf_nodes=trial.suggest_int("hgb_max_leaf_nodes", 15, 255),
            l2_regularization=trial.suggest_float("hgb_l2", 1e-8, 1e1, log=True),
            min_samples_leaf=trial.suggest_int("hgb_min_samples_leaf", 5, 50),
            random_state=random_state,
        )
    elif model_name == "pls":
        pls_max_components = max(2, min(20, n_features, max(2, n_samples_train - 1)))
        model = PLSRegression(n_components=trial.suggest_int("pls_n_components", 2, pls_max_components))
    elif model_name == "kernel_ridge":
        model = KernelRidge(
            alpha=trial.suggest_float("kr_alpha", 1e-4, 1e2, log=True),
            kernel=trial.suggest_categorical("kr_kernel", ["rbf", "laplacian", "poly"]),
            gamma=trial.suggest_float("kr_gamma", 1e-6, 1e-1, log=True),
        )
    elif model_name == "gpr":
        length_scale = trial.suggest_float("gpr_length_scale", 1e-2, 1e2, log=True)
        noise = trial.suggest_float("gpr_noise", 1e-8, 1e0, log=True)
        nu = trial.suggest_categorical("gpr_nu", [0.5, 1.5, 2.5])
        kernel = 1.0 * Matern(length_scale=length_scale, nu=nu) + WhiteKernel(noise_level=noise)
        model = GaussianProcessRegressor(kernel=kernel, random_state=random_state, normalize_y=False)
    elif model_name == "xgboost":
        params = {
            "n_estimators": trial.suggest_int("xgb_n_estimators", 100, 1000, step=100),
            "learning_rate": trial.suggest_float("xgb_learning_rate", 1e-3, 0.2, log=True),
            "max_depth": trial.suggest_int("xgb_max_depth", 2, 10),
            "subsample": trial.suggest_float("xgb_subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("xgb_colsample", 0.5, 1.0),
            "reg_lambda": trial.suggest_float("xgb_reg_lambda", 1e-6, 10.0, log=True),
            "reg_alpha": trial.suggest_float("xgb_reg_alpha", 1e-8, 1.0, log=True),
            "objective": "reg:squarederror",
            "random_state": random_state,
            "n_jobs": 1,
            "tree_method": "hist",
        }
        if use_gpu:
            params["device"] = "cuda"
        model = XGBRegressor(**params)
    elif model_name == "lightgbm":
        params = {
            "n_estimators": trial.suggest_int("lgb_n_estimators", 100, 1000, step=100),
            "learning_rate": trial.suggest_float("lgb_learning_rate", 1e-3, 0.2, log=True),
            "num_leaves": trial.suggest_int("lgb_num_leaves", 15, 255),
            "subsample": trial.suggest_float("lgb_subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("lgb_colsample", 0.5, 1.0),
            "reg_lambda": trial.suggest_float("lgb_reg_lambda", 1e-6, 10.0, log=True),
            "random_state": random_state,
            "n_jobs": 1,
            "verbose": -1,
        }
        if use_gpu:
            params["device"] = "gpu"
        model = LGBMRegressor(**params)
    else:
        raise ValueError(model_name)

    return model_name, model


def choose_feature_subset(
    trial: "optuna.trial.Trial",
    feature_subsets: Sequence[Tuple[str, ...]],
) -> Tuple[str, ...]:
    labels = ["+".join(subset) for subset in feature_subsets]
    chosen = trial.suggest_categorical("feature_subset", labels)
    return tuple(chosen.split("+"))


def build_preprocess_and_model(
    trial: "optuna.trial.Trial",
    n_features: int,
    n_samples_train: int,
    random_state: int,
    use_gpu: bool,
    force_reducer: Optional[str] = None,
    force_target_transform: Optional[str] = None,
) -> Tuple[BaseEstimator, Dict[str, Any]]:
    steps: List[Tuple[str, Any]] = []
    meta: Dict[str, Any] = {}

    steps.append(("var", VarianceThreshold()))

    scaler_choice = trial.suggest_categorical("scaler", ["none", "standard", "robust"])
    if scaler_choice == "standard":
        steps.append(("scale", StandardScaler()))
    elif scaler_choice == "robust":
        steps.append(("scale", RobustScaler()))
    else:
        steps.append(("scale", IdentityTransformer()))
    meta["scaler"] = scaler_choice

    selector_choice = trial.suggest_categorical("selector", ["none", "f_regression", "mutual_info"])
    if selector_choice == "none":
        steps.append(("select", IdentityTransformer()))
        meta["selector"] = "none"
        selector_k_eff = n_features
    else:
        upper = max(5, min(n_features, 512))
        selector_k = trial.suggest_int("selector_k", 5, upper)
        steps.append(("select", ColumnSelectorKBest(score_func=selector_choice, k=selector_k)))
        meta["selector"] = selector_choice
        meta["selector_k"] = int(selector_k)
        selector_k_eff = min(selector_k, n_features)

    reducer_choice = force_reducer or trial.suggest_categorical("reducer", ["none", "pca_fixed", "pca_var", "svd"])
    max_components = max(2, min(selector_k_eff, n_samples_train - 1))
    if reducer_choice == "none" or max_components < 2:
        steps.append(("reduce", IdentityTransformer()))
        meta["reducer"] = "none"
    elif reducer_choice == "pca_fixed":
        n_components = trial.suggest_int("pca_n_components", 2, max_components)
        steps.append(("reduce", PCA(n_components=n_components, random_state=random_state)))
        meta["reducer"] = "pca_fixed"
        meta["pca_n_components"] = int(n_components)
    elif reducer_choice == "pca_var":
        variance_keep = trial.suggest_categorical("pca_var_keep", [0.90, 0.95, 0.99])
        steps.append(("reduce", PCA(n_components=variance_keep, random_state=random_state)))
        meta["reducer"] = "pca_var"
        meta["pca_var_keep"] = float(variance_keep)
    elif reducer_choice == "svd":
        n_components = trial.suggest_int("svd_n_components", 2, max_components)
        steps.append(("reduce", TruncatedSVD(n_components=n_components, random_state=random_state)))
        meta["reducer"] = "svd"
        meta["svd_n_components"] = int(n_components)
    else:
        raise ValueError(reducer_choice)

    model_name, model = make_model(
        trial,
        random_state=random_state,
        use_gpu=use_gpu,
        n_features=n_features,
        n_samples_train=n_samples_train,
    )
    steps.append(("model", model))
    pipeline = Pipeline(steps)

    target_transform = force_target_transform or trial.suggest_categorical("target_transform", ["none", "quantile"])
    if target_transform == "quantile":
        n_quantiles = int(min(max(10, n_samples_train - 1), 200))
        estimator: BaseEstimator = TransformedTargetRegressor(
            regressor=pipeline,
            transformer=QuantileTransformer(
                n_quantiles=n_quantiles,
                output_distribution="normal",
                random_state=random_state,
                subsample=int(1e9),
            ),
        )
    else:
        estimator = pipeline

    meta["target_transform"] = target_transform
    meta["model_name"] = model_name
    return estimator, meta


@dataclass
class SearchContext:
    X_by_mode: Dict[str, Dict[str, np.ndarray]]
    y: np.ndarray
    groups: np.ndarray
    splits: List[Tuple[np.ndarray, np.ndarray]]
    feature_subsets: List[Tuple[str, ...]]
    feature_modes: List[str]
    metric_name: str
    random_state: int
    use_gpu: bool
    standard_search: bool = False


def make_objective(ctx: SearchContext):
    def objective(trial: "optuna.trial.Trial") -> float:
        feature_subset = choose_feature_subset(trial, ctx.feature_subsets)
        if ctx.standard_search:
            feature_mode = "raw"
        else:
            feature_mode = trial.suggest_categorical("feature_mode", ctx.feature_modes)
        X = np.concatenate([ctx.X_by_mode[feature_mode][source] for source in feature_subset], axis=1)
        effective_n_features = max(1, int(np.sum(np.var(X, axis=0) > 0.0)))
        try:
            estimator, meta = build_preprocess_and_model(
                trial=trial,
                n_features=effective_n_features,
                n_samples_train=max(10, int(len(ctx.y) * (len(ctx.splits) - 1) / len(ctx.splits))),
                random_state=ctx.random_state + trial.number,
                use_gpu=ctx.use_gpu,
                force_reducer="none" if ctx.standard_search else None,
                force_target_transform="none" if ctx.standard_search else None,
            )
            metrics, _, _, _, fold_rows = cross_val_predict_with_uncertainty(
                estimator=estimator,
                X=X,
                y=ctx.y,
                splits=ctx.splits,
            )
        except Exception as exc:
            trial.set_user_attr("failed_reason", str(exc))
            raise optuna.TrialPruned(str(exc))
        score = float(metrics[ctx.metric_name])
        trial.set_user_attr("feature_subset", list(feature_subset))
        trial.set_user_attr("feature_mode", feature_mode)
        trial.set_user_attr("metrics", metrics)
        trial.set_user_attr("fold_rows", fold_rows)
        trial.set_user_attr("meta", meta)
        return score

    return objective


def build_fixed_trial_estimator(
    frozen_trial: "optuna.trial.FrozenTrial",
    X: np.ndarray,
    n_splits: int,
    random_state: int,
    use_gpu: bool,
    standard_search: bool = False,
) -> Tuple[BaseEstimator, Dict[str, Any]]:
    fixed_trial = optuna.trial.FixedTrial(frozen_trial.params)
    effective_n_features = max(1, int(np.sum(np.var(X, axis=0) > 0.0)))
    estimator, meta = build_preprocess_and_model(
        trial=fixed_trial,
        n_features=effective_n_features,
        n_samples_train=max(10, int(X.shape[0] * (n_splits - 1) / n_splits)),
        random_state=random_state,
        use_gpu=use_gpu,
        force_reducer="none" if standard_search else None,
        force_target_transform="none" if standard_search else None,
    )
    return estimator, meta


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="Supervised training CSV")
    parser.add_argument("--target-col", required=True, help="Name of supervised target column")
    parser.add_argument("--train-seq-col", default=None, help="Sequence column for training CSV")
    parser.add_argument("--seq-col", default=None, help="Backward-compatible alias for --train-seq-col")
    parser.add_argument("--predict-seq-col", default=None, help="Sequence column for optional prediction CSV")
    parser.add_argument("--id-col", default=None, help="Optional supervised row identifier column")
    parser.add_argument("--predict-id-col", default=None, help="Optional prediction row identifier column")
    parser.add_argument("--group-col", default=None, help="Optional explicit leakage-safe grouping column")
    parser.add_argument("--embedding-dir", default=None, help="Directory containing <feature_source>.npz files")
    parser.add_argument(
        "--feature-sources",
        nargs="+",
        default=None,
        help="Feature sources to search over, e.g. esm2 onehot blosum62",
    )
    parser.add_argument(
        "--embeddings",
        nargs="+",
        default=None,
        help="Backward-compatible explicit embedding NPZ paths. Feature names are inferred from file stems.",
    )
    parser.add_argument(
        "--feature-mode-options",
        nargs="+",
        default=None,
        choices=["raw", "delta", "raw_plus_delta"],
        help="Allowed feature modes. Defaults to raw unless --wt-sequence is supplied.",
    )
    parser.add_argument(
        "--replicate-policy",
        default="mean_by_sequence",
        choices=["mean_by_sequence", "median_by_sequence", "keep_rows"],
        help="How to handle repeated rows for identical amino-acid sequences",
    )
    parser.add_argument("--cv-splits", type=int, default=5)
    parser.add_argument("--n-trials", type=int, default=100)
    parser.add_argument("--timeout", type=int, default=None)
    parser.add_argument("--metric", default="spearman", choices=["spearman", "kendall", "r2", "rmse", "mae"])
    parser.add_argument("--top-ensemble", type=int, default=5)
    parser.add_argument("--conformal-alpha", type=float, default=0.10)
    parser.add_argument(
        "--no-uncertainty",
        action="store_true",
        help="Skip uncertainty ensemble fitting and leave uncertainty columns blank",
    )
    parser.add_argument(
        "--standard-search",
        "--standard",
        dest="standard_search",
        action="store_true",
        help="Restrict search to single provided feature sources, raw features only, no PCA/SVD, and no quantile target transform",
    )
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--use-gpu", action="store_true")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--predict-csv", default=None, help="Optional CSV to score after fitting")
    parser.add_argument("--predict-embedding-dir", default=None, help="Embedding directory for prediction CSV")
    parser.add_argument("--wt-sequence", default=None, help="Required for delta-based feature modes")
    parser.add_argument("--study-name", default="protein_optuna_automl")
    parser.add_argument("--storage", default=None, help="Optional Optuna storage URL, e.g. sqlite:///study.db")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if optuna is None or TPESampler is None:
        raise ImportError(
            "This script requires optuna in the active environment. "
            "Install it with `pip install optuna` or use the updated environment files."
        )

    train_seq_col = args.train_seq_col or args.seq_col
    if train_seq_col is None:
        raise ValueError("Provide --train-seq-col (or the backward-compatible --seq-col)")
    predict_seq_col = args.predict_seq_col or train_seq_col

    if args.feature_sources is None and args.embeddings is None:
        raise ValueError("Provide either --feature-sources or --embeddings")

    if args.feature_sources is None:
        feature_sources = [Path(path).stem for path in args.embeddings]
    else:
        feature_sources = [normalize_feature_source(source) for source in args.feature_sources]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(args.random_state)

    print_header("Loading supervised CSV")
    raw_df = pd.read_csv(args.csv)
    if train_seq_col not in raw_df.columns:
        raise ValueError(f"Training sequence column '{train_seq_col}' not found in {args.csv}")
    if args.target_col not in raw_df.columns:
        raise ValueError(f"Target column '{args.target_col}' not found in {args.csv}")

    prepared_df = prepare_supervised_dataframe(
        df=raw_df,
        seq_col=train_seq_col,
        target_col=args.target_col,
        replicate_policy=args.replicate_policy,
        id_col=args.id_col,
    )
    y = prepared_df[args.target_col].astype(float).to_numpy()
    if args.group_col:
        if args.group_col not in prepared_df.columns:
            raise ValueError(f"Group column '{args.group_col}' not found in prepared training data")
        groups = prepared_df[args.group_col].astype(str).to_numpy()
        group_col_used = args.group_col
    else:
        groups = prepared_df[train_seq_col].astype(str).to_numpy()
        group_col_used = train_seq_col

    print(f"Rows in raw training CSV: {len(raw_df)}")
    print(f"Rows after replicate policy '{args.replicate_policy}': {len(prepared_df)}")

    print_header("Loading feature banks")
    embedding_banks = load_embedding_banks(
        feature_sources=feature_sources,
        embedding_dir=args.embedding_dir,
        explicit_embedding_paths=args.embeddings,
    )
    for name, bank in embedding_banks.items():
        print(f"- {name}: dim={bank.dim} | {bank.path}")

    X_by_source_raw, missing_any, source_missing_counts, expected_sequence_length = assemble_feature_matrices(
        df=prepared_df,
        seq_col=train_seq_col,
        feature_sources=feature_sources,
        embedding_banks=embedding_banks,
    )
    rows_before_feature_drop = len(prepared_df)
    if missing_any.any():
        print(f"Dropping {int(missing_any.sum())} training rows with missing learned features")
        prepared_df = prepared_df.loc[~missing_any].reset_index(drop=True)
        y = y[~missing_any]
        groups = groups[~missing_any]
        for source in list(X_by_source_raw.keys()):
            X_by_source_raw[source] = X_by_source_raw[source][~missing_any]
    if len(prepared_df) < 2:
        raise RuntimeError("Not enough rows remain after feature coverage filtering")

    wt_index: Optional[int] = None
    wt_by_source: Optional[Dict[str, np.ndarray]] = None
    if args.wt_sequence is not None:
        wt_sequence = str(args.wt_sequence).strip()
        seqs_after_filter = prepared_df[train_seq_col].astype(str).tolist()
        if wt_sequence not in seqs_after_filter:
            raise ValueError("WT sequence was not found in the filtered training data")
        wt_index = seqs_after_filter.index(wt_sequence)
        wt_by_source = {source: matrix[wt_index].copy() for source, matrix in X_by_source_raw.items()}

    if args.standard_search:
        feature_modes = ["raw"]
    elif args.feature_mode_options:
        feature_modes = list(dict.fromkeys(args.feature_mode_options))
    elif wt_index is not None:
        feature_modes = ["raw", "delta", "raw_plus_delta"]
    else:
        feature_modes = ["raw"]

    if any(mode != "raw" for mode in feature_modes) and wt_index is None:
        raise ValueError("Delta-based feature modes require --wt-sequence")

    X_by_mode: Dict[str, Dict[str, np.ndarray]] = {
        mode: transform_feature_mode(
            X_by_source_raw,
            wt_index=wt_index,
            feature_mode=mode,
            wt_by_source=wt_by_source,
        )
        for mode in feature_modes
    }
    if args.standard_search:
        feature_subsets = [(source,) for source in feature_sources]
    else:
        feature_subsets = all_nonempty_feature_subsets(feature_sources)
    splits, uses_group_cv = make_cv_splits(
        n_samples=len(prepared_df),
        groups=groups,
        cv_splits=args.cv_splits,
        random_state=args.random_state,
    )

    print_header("Search space summary")
    print(f"Feature sources: {', '.join(feature_sources)}")
    print(f"Feature subsets: {len(feature_subsets)}")
    print(f"Feature modes: {', '.join(feature_modes)}")
    print(f"Metric: {args.metric}")
    print(f"CV strategy: {'GroupKFold' if uses_group_cv else 'KFold'} with {len(splits)} splits")

    ctx = SearchContext(
        X_by_mode=X_by_mode,
        y=y,
        groups=groups,
        splits=splits,
        feature_subsets=feature_subsets,
        feature_modes=feature_modes,
        metric_name=args.metric,
        random_state=args.random_state,
        use_gpu=args.use_gpu,
        standard_search=args.standard_search,
    )

    print_header("Running Optuna search")
    sampler = TPESampler(seed=args.random_state, multivariate=True)
    study = optuna.create_study(
        study_name=args.study_name,
        storage=args.storage,
        direction=metric_direction(args.metric),
        sampler=sampler,
        load_if_exists=True,
    )
    study.optimize(
        make_objective(ctx),
        n_trials=args.n_trials,
        timeout=args.timeout,
        show_progress_bar=True,
    )

    complete_trials = [trial for trial in study.trials if trial.state == optuna.trial.TrialState.COMPLETE]
    if not complete_trials:
        raise RuntimeError("No completed Optuna trials")

    descending = sort_descending_for_metric(args.metric)
    complete_trials = sorted(complete_trials, key=lambda trial: float(trial.value), reverse=descending)
    best_trial = complete_trials[0]
    best_subset = tuple(best_trial.user_attrs["feature_subset"])
    best_mode = str(best_trial.user_attrs["feature_mode"])
    best_X = np.concatenate([X_by_mode[best_mode][source] for source in best_subset], axis=1)
    best_estimator, best_meta = build_fixed_trial_estimator(
        frozen_trial=best_trial,
        X=best_X,
        n_splits=len(splits),
        random_state=args.random_state + best_trial.number,
        use_gpu=args.use_gpu,
        standard_search=args.standard_search,
    )

    print_header("Evaluating best single model")
    best_metrics, best_oof, best_native_std, best_fold_ids, best_fold_rows = cross_val_predict_with_uncertainty(
        estimator=best_estimator,
        X=best_X,
        y=y,
        splits=splits,
    )
    best_abs_resid = np.abs(y - best_oof)
    if args.no_uncertainty:
        qhat = float("nan")
        best_native_std = np.full(len(best_native_std), np.nan, dtype=float)
    else:
        qhat = conformal_q(best_abs_resid, alpha=args.conformal_alpha)

    history_rows = []
    for trial in complete_trials:
        history_rows.append(
            {
                "trial_number": trial.number,
                "value": trial.value,
                "feature_subset": "+".join(trial.user_attrs.get("feature_subset", [])),
                "feature_mode": trial.user_attrs.get("feature_mode"),
                **{f"param__{key}": value for key, value in trial.params.items()},
                **{f"metric__{key}": value for key, value in trial.user_attrs.get("metrics", {}).items()},
                "meta_json": json.dumps(trial.user_attrs.get("meta", {}), default=json_default),
            }
        )
    search_history = pd.DataFrame(history_rows).sort_values("value", ascending=not descending)
    search_history.to_csv(out_dir / "search_history.csv", index=False)

    top_trials = complete_trials[: max(1, min(args.top_ensemble, len(complete_trials)))]
    ensemble_members: List[FittedRunModel] = []
    ensemble_oof_predictions: List[np.ndarray] = []
    ensemble_rows: List[Dict[str, Any]] = []

    print_header("Fitting best model and uncertainty ensemble")
    best_estimator.fit(best_X, y)
    best_model_artifact = FittedRunModel(
        run_name=out_dir.name,
        trial_number=best_trial.number,
        score=float(best_trial.value),
        metric_name=args.metric,
        model_name=best_meta["model_name"],
        feature_sources=best_subset,
        feature_mode=best_mode,
        estimator=best_estimator,
        conformal_alpha=args.conformal_alpha,
        conformal_qhat=qhat,
        train_sequences=tuple(prepared_df[train_seq_col].astype(str).tolist()),
        expected_sequence_length=expected_sequence_length,
        wt_by_source={} if wt_by_source is None else wt_by_source,
    )
    joblib.dump(best_model_artifact, out_dir / "best_model.joblib")

    ensemble_artifact: Optional[FittedEnsemble] = None
    if not args.no_uncertainty:
        for trial in top_trials:
            feature_subset = tuple(trial.user_attrs["feature_subset"])
            feature_mode = str(trial.user_attrs["feature_mode"])
            X_trial = np.concatenate([X_by_mode[feature_mode][source] for source in feature_subset], axis=1)
            estimator_trial, meta_trial = build_fixed_trial_estimator(
                frozen_trial=trial,
                X=X_trial,
                n_splits=len(splits),
                random_state=args.random_state + trial.number,
                use_gpu=args.use_gpu,
                standard_search=args.standard_search,
            )
            _, trial_oof, _, _, _ = cross_val_predict_with_uncertainty(
                estimator=estimator_trial,
                X=X_trial,
                y=y,
                splits=splits,
            )
            ensemble_oof_predictions.append(trial_oof)
            estimator_trial.fit(X_trial, y)
            ensemble_members.append(
                FittedRunModel(
                    run_name=out_dir.name,
                    trial_number=trial.number,
                    score=float(trial.value),
                    metric_name=args.metric,
                    model_name=meta_trial["model_name"],
                    feature_sources=feature_subset,
                    feature_mode=feature_mode,
                    estimator=estimator_trial,
                    conformal_alpha=args.conformal_alpha,
                    conformal_qhat=qhat,
                    train_sequences=tuple(prepared_df[train_seq_col].astype(str).tolist()),
                    expected_sequence_length=expected_sequence_length,
                    wt_by_source={} if wt_by_source is None else wt_by_source,
                )
            )
            ensemble_rows.append(
                {
                    "trial_number": trial.number,
                    "score": float(trial.value),
                    "feature_subset": "+".join(feature_subset),
                    "feature_mode": feature_mode,
                    "model_name": meta_trial["model_name"],
                    "meta_json": json.dumps(meta_trial, default=json_default),
                }
            )

        ensemble_artifact = FittedEnsemble(run_name=out_dir.name, fitted_models=ensemble_members)
        joblib.dump(ensemble_artifact, out_dir / "uncertainty_ensemble.joblib")
        pd.DataFrame(ensemble_rows).to_csv(out_dir / "top_ensemble_members.csv", index=False)
        deployment_feature_sources = sorted({source for member in ensemble_members for source in member.feature_sources})
        ensemble_oof_mean = np.mean(np.vstack(ensemble_oof_predictions), axis=0)
        ensemble_oof_std = np.std(np.vstack(ensemble_oof_predictions), axis=0, ddof=0)
    else:
        deployment_feature_sources = sorted(best_subset)
        ensemble_oof_mean = np.full(len(best_oof), np.nan, dtype=float)
        ensemble_oof_std = np.full(len(best_oof), np.nan, dtype=float)

    oof_df = build_oof_dataframe(
        df=prepared_df,
        run_name=out_dir.name,
        target_col=args.target_col,
        y_pred=best_oof,
        fold_ids=best_fold_ids,
        group_values=groups,
        conformal_qhat=qhat,
        native_std=best_native_std,
        ensemble_mean=ensemble_oof_mean,
        ensemble_std=ensemble_oof_std,
    )
    oof_df.to_csv(out_dir / "oof_predictions.csv", index=False)
    pd.DataFrame(best_fold_rows).to_csv(out_dir / "fold_metrics.csv", index=False)

    train_best_pred, train_best_native_std = best_model_artifact.predict_with_uncertainty(X_by_mode[best_mode])
    if args.no_uncertainty or ensemble_artifact is None:
        train_best_native_std = np.full(len(train_best_pred), np.nan, dtype=float)
        train_ensemble_mean = np.full(len(train_best_pred), np.nan, dtype=float)
        train_ensemble_std = np.full(len(train_best_pred), np.nan, dtype=float)
    else:
        train_ensemble_mean, train_ensemble_std = ensemble_artifact.predict_with_uncertainty(X_by_mode)
    train_predictions_input = prepared_df.copy()
    train_predictions_input["y_true"] = y
    train_predictions = build_prediction_dataframe(
        df=train_predictions_input,
        run_name=out_dir.name,
        y_pred=train_best_pred,
        conformal_qhat=qhat,
        ensemble_mean=train_ensemble_mean,
        native_std=train_best_native_std,
        ensemble_std=train_ensemble_std,
        seen_in_train=np.ones(len(prepared_df), dtype=bool),
        missing_any_feature=np.zeros(len(prepared_df), dtype=bool),
        ascending=False,
    )
    train_predictions["residual"] = train_predictions["y_true"] - train_predictions["y_pred"]
    train_predictions.to_csv(out_dir / "train_predictions.csv", index=False)

    coverage_report = {
        "feature_sources": feature_sources,
        "source_missing_counts_before_drop": source_missing_counts,
        "rows_before_feature_drop": rows_before_feature_drop,
        "rows_after_feature_drop": int(len(prepared_df)),
        "rows_dropped_for_missing_features": int(rows_before_feature_drop - len(prepared_df)),
        "expected_sequence_length": int(expected_sequence_length),
    }
    save_json(out_dir / "coverage_report.json", coverage_report)

    run_summary = {
        "run_name": out_dir.name,
        "csv": args.csv,
        "predict_csv": args.predict_csv,
        "train_seq_col": train_seq_col,
        "predict_seq_col": predict_seq_col,
        "target_col": args.target_col,
        "id_col": args.id_col,
        "predict_id_col": args.predict_id_col,
        "group_col": group_col_used,
        "replicate_policy": args.replicate_policy,
        "feature_sources": feature_sources,
        "deployment_feature_sources": deployment_feature_sources,
        "feature_modes_searched": feature_modes,
        "standard_search": bool(args.standard_search),
        "uncertainty_enabled": bool(not args.no_uncertainty),
        "metric": args.metric,
        "metric_direction": metric_direction(args.metric),
        "best_trial_number": int(best_trial.number),
        "best_trial_value": float(best_trial.value),
        "best_feature_subset": list(best_subset),
        "best_feature_mode": best_mode,
        "best_model_name": best_meta["model_name"],
        "best_params": best_trial.params,
        "best_meta": best_meta,
        "best_oof_metrics": best_metrics,
        "conformal_alpha": float(args.conformal_alpha),
        "conformal_qhat": qhat,
        "top_ensemble_size": int(len(ensemble_members)),
        "n_raw_rows": int(len(raw_df)),
        "n_training_rows": int(len(prepared_df)),
        "n_feature_subsets": int(len(feature_subsets)),
        "embedding_dir": args.embedding_dir,
        "embedding_paths": args.embeddings,
        "predict_embedding_dir": args.predict_embedding_dir or args.embedding_dir,
        "expected_sequence_length": int(expected_sequence_length),
        "wt_sequence": args.wt_sequence,
        "coverage_report_file": "coverage_report.json",
    }
    save_json(out_dir / "run_summary.json", run_summary)

    if args.predict_csv:
        print_header(f"Predicting {args.predict_csv}")
        predict_df = pd.read_csv(args.predict_csv)
        if predict_seq_col not in predict_df.columns:
            raise ValueError(f"Prediction sequence column '{predict_seq_col}' not found in {args.predict_csv}")

        predict_embedding_dir = args.predict_embedding_dir or args.embedding_dir
        predict_embedding_banks = load_embedding_banks(
            feature_sources=deployment_feature_sources,
            embedding_dir=predict_embedding_dir,
            explicit_embedding_paths=None if predict_embedding_dir is not None else args.embeddings,
        )
        predict_X_raw, predict_missing_any, _, _ = assemble_feature_matrices(
            df=predict_df,
            seq_col=predict_seq_col,
            feature_sources=deployment_feature_sources,
            embedding_banks=predict_embedding_banks,
            expected_sequence_length=expected_sequence_length,
        )
        predict_X_by_mode = {
            mode: transform_feature_mode(
                predict_X_raw,
                wt_index=None,
                feature_mode=mode,
                wt_by_source=best_model_artifact.wt_by_source or None,
            )
            for mode in feature_modes
        }

        predict_best_matrix_sources = predict_X_by_mode[best_mode]
        valid_mask = ~predict_missing_any
        y_pred = np.full(len(predict_df), np.nan, dtype=float)
        native_std = np.full(len(predict_df), np.nan, dtype=float)
        ensemble_mean = np.full(len(predict_df), np.nan, dtype=float)
        ensemble_std = np.full(len(predict_df), np.nan, dtype=float)
        if valid_mask.any():
            pred_valid, native_std_valid = best_model_artifact.predict_with_uncertainty(
                {key: value[valid_mask] for key, value in predict_best_matrix_sources.items()}
            )
            y_pred[valid_mask] = pred_valid
            if (not args.no_uncertainty) and native_std_valid is not None:
                native_std[valid_mask] = native_std_valid
            if not args.no_uncertainty and ensemble_artifact is not None:
                ensemble_mean_valid, ensemble_std_valid = ensemble_artifact.predict_with_uncertainty(
                    {
                        mode: {key: value[valid_mask] for key, value in source_map.items()}
                        for mode, source_map in predict_X_by_mode.items()
                    }
                )
                ensemble_mean[valid_mask] = ensemble_mean_valid
                ensemble_std[valid_mask] = ensemble_std_valid

        seen_in_train = predict_df[predict_seq_col].astype(str).str.strip().isin(set(best_model_artifact.train_sequences)).to_numpy()
        candidate_predictions = build_prediction_dataframe(
            df=predict_df,
            run_name=out_dir.name,
            y_pred=y_pred,
            conformal_qhat=qhat,
            ensemble_mean=ensemble_mean,
            native_std=native_std,
            ensemble_std=ensemble_std,
            seen_in_train=seen_in_train,
            missing_any_feature=predict_missing_any,
            ascending=False,
        )
        candidate_predictions.to_csv(out_dir / "candidate_predictions.csv", index=False)

    print_header("Done")
    print(f"Best trial: {best_trial.number}")
    print(f"Best {args.metric}: {best_trial.value:.4f}")
    print(f"Best feature subset: {' + '.join(best_subset)}")
    print(f"Best feature mode: {best_mode}")
    print(f"Standard search: {args.standard_search}")
    print(f"Uncertainty enabled: {not args.no_uncertainty}")
    print(f"Outputs saved to: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
