#!/usr/bin/env python3
"""Optuna search: preprocessing + model pipeline construction and the objective.

Generalized from the original ``training3_optuna_mlde_uncertainty.py``:
- model selection comes from the size-filtered :data:`MODEL_REGISTRY`
- deep models are supported; CNN is constrained to positional encodings via
  trial pruning
- CV strategy / arrays are supplied by the caller (size engine in ``train.py``)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

import optuna

from sklearn.base import BaseEstimator
from sklearn.compose import TransformedTargetRegressor
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.feature_selection import VarianceThreshold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import QuantileTransformer, RobustScaler, StandardScaler

from .core import ColumnSelectorKBest, IdentityTransformer, cross_val_predict_with_uncertainty
from .registry import MODEL_REGISTRY, build_model


def choose_feature_subset(trial, feature_subsets: Sequence[Tuple[str, ...]]) -> Tuple[str, ...]:
    labels = ["+".join(subset) for subset in feature_subsets]
    chosen = trial.suggest_categorical("feature_subset", labels)
    return tuple(chosen.split("+"))


def build_preprocess_and_model(
    trial,
    eligible_models: Sequence[str],
    n_features: int,
    n_samples_train: int,
    random_state: int,
    use_gpu: bool,
    force_reducer: Optional[str] = None,
    force_target_transform: Optional[str] = None,
) -> Tuple[BaseEstimator, Dict[str, Any]]:
    """Suggest model + preprocessing and assemble an estimator.

    Models whose ``requires_source`` is set (e.g. CNN) get a *bare* pipeline
    (no variance/scale/select/reduce) so the flat L*C layout stays intact; the
    deep wrapper standardizes internally.
    """
    meta: Dict[str, Any] = {}
    model_name = trial.suggest_categorical("model_name", list(eligible_models))
    spec = MODEL_REGISTRY[model_name]
    bare = spec.requires_source is not None

    steps: List[Tuple[str, Any]] = []
    if not bare:
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
    else:
        meta["scaler"] = "none"
        meta["selector"] = "none"
        meta["reducer"] = "none"

    _, model = build_model(
        model_name,
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
    meta["bare_pipeline"] = bare
    return estimator, meta


@dataclass
class SearchContext:
    X_by_mode: Dict[str, Dict[str, np.ndarray]]
    y: np.ndarray
    groups: np.ndarray
    splits: List[Tuple[np.ndarray, np.ndarray]]
    feature_subsets: List[Tuple[str, ...]]
    feature_modes: List[str]
    eligible_models: List[str]
    metric_name: str
    random_state: int
    use_gpu: bool
    standard_search: bool = False


def _subset_ok_for_model(model_name: str, subset: Tuple[str, ...]) -> bool:
    spec = MODEL_REGISTRY[model_name]
    if spec.requires_source is None:
        return True
    return len(subset) == 1 and subset[0] in spec.requires_source


def make_objective(ctx: SearchContext):
    def objective(trial) -> float:
        feature_subset = choose_feature_subset(trial, ctx.feature_subsets)
        feature_mode = "raw" if ctx.standard_search else trial.suggest_categorical("feature_mode", ctx.feature_modes)
        X = np.concatenate([ctx.X_by_mode[feature_mode][source] for source in feature_subset], axis=1)
        effective_n_features = max(1, int(np.sum(np.var(X, axis=0) > 0.0)))
        try:
            estimator, meta = build_preprocess_and_model(
                trial=trial,
                eligible_models=ctx.eligible_models,
                n_features=effective_n_features,
                n_samples_train=max(10, int(len(ctx.y) * (len(ctx.splits) - 1) / max(1, len(ctx.splits)))),
                random_state=ctx.random_state + trial.number,
                use_gpu=ctx.use_gpu,
                force_reducer="none" if ctx.standard_search else None,
                force_target_transform="none" if ctx.standard_search else None,
            )
            if not _subset_ok_for_model(meta["model_name"], feature_subset):
                raise optuna.TrialPruned(
                    f"{meta['model_name']} requires a positional encoding source; subset={feature_subset}"
                )
            metrics, _, _, _, fold_rows = cross_val_predict_with_uncertainty(
                estimator=estimator, X=X, y=ctx.y, splits=ctx.splits
            )
        except optuna.TrialPruned:
            raise
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
    frozen_trial,
    eligible_models: Sequence[str],
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
        eligible_models=eligible_models,
        n_features=effective_n_features,
        n_samples_train=max(10, int(X.shape[0] * (n_splits - 1) / max(1, n_splits))),
        random_state=random_state,
        use_gpu=use_gpu,
        force_reducer="none" if standard_search else None,
        force_target_transform="none" if standard_search else None,
    )
    return estimator, meta
