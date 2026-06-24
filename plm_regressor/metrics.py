#!/usr/bin/env python3
"""Regression + ranking metrics for the PLM-Regressor framework.

Centralizes every metric the framework can report or optimize:
- point metrics: rmse, mse, mae, r2
- rank metrics: spearman, kendall, pearson
- top-of-list metrics: ndcg, ndcg@k, topk_recall, topk_precision

All metric functions are defensive: they never raise on degenerate input
(constant predictions, <2 rows, NaNs) so an Optuna trial is never killed by a
metric edge case.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
from scipy.stats import kendalltau, pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, ndcg_score, r2_score

# Metrics usable as a single scalar optimization objective.
METRIC_CHOICES = (
    "spearman",
    "kendall",
    "pearson",
    "r2",
    "ndcg",
    "rmse",
    "mse",
    "mae",
)

_MINIMIZE = {"rmse", "mse", "mae"}


def metric_direction(metric_name: str) -> str:
    """'minimize' for error metrics, 'maximize' otherwise."""
    return "minimize" if metric_name in _MINIMIZE else "maximize"


def sort_descending_for_metric(metric_name: str) -> bool:
    return metric_direction(metric_name) == "maximize"


def _clean_pair(y_true: np.ndarray, y_pred: np.ndarray):
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    return y_true, y_pred


def safe_spearman(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true, y_pred = _clean_pair(y_true, y_pred)
    if len(y_true) < 2 or np.allclose(np.std(y_pred), 0.0):
        return 0.0
    corr = spearmanr(y_true, y_pred).correlation
    return 0.0 if corr is None or np.isnan(corr) else float(corr)


def safe_kendall(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    try:
        corr = kendalltau(y_true, y_pred, nan_policy="omit").correlation
        return 0.0 if corr is None or np.isnan(corr) else float(corr)
    except Exception:
        return 0.0


def safe_pearson(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true, y_pred = _clean_pair(y_true, y_pred)
    if len(y_true) < 2 or np.allclose(np.std(y_pred), 0.0) or np.allclose(np.std(y_true), 0.0):
        return 0.0
    try:
        corr = pearsonr(y_true, y_pred)[0]
    except Exception:
        return 0.0
    return 0.0 if corr is None or np.isnan(corr) else float(corr)


def ndcg_at_k(y_true: np.ndarray, y_pred: np.ndarray, k: int | None = None) -> float:
    """Single-query NDCG. Gains are shifted to be non-negative (NDCG requirement)."""
    y_true, y_pred = _clean_pair(y_true, y_pred)
    if len(y_true) < 2 or np.allclose(np.std(y_pred), 0.0):
        return 0.0
    gains = y_true - np.min(y_true)
    if np.allclose(gains.sum(), 0.0):
        return 0.0
    try:
        return float(ndcg_score(gains.reshape(1, -1), y_pred.reshape(1, -1), k=k))
    except Exception:
        return 0.0


def topk_recall(y_true: np.ndarray, y_pred: np.ndarray, k: int = 10) -> float:
    """Fraction of the true top-k that appears in the predicted top-k."""
    y_true, y_pred = _clean_pair(y_true, y_pred)
    n = len(y_true)
    if n == 0:
        return 0.0
    k = max(1, min(k, n))
    true_top = set(np.argsort(-y_true)[:k].tolist())
    pred_top = set(np.argsort(-y_pred)[:k].tolist())
    return float(len(true_top & pred_top) / k)


def topk_precision(y_true: np.ndarray, y_pred: np.ndarray, k: int = 10) -> float:
    # For equal-size top-k sets, precision == recall; kept for naming clarity.
    return topk_recall(y_true, y_pred, k=k)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, topk: int = 10) -> Dict[str, float]:
    """Full metric bundle. Keys are stable so downstream CSV schemas stay fixed.

    Superset of the legacy bundle (rmse/mae/r2/spearman/kendall) plus
    mse/pearson/ndcg/topk_recall, so existing readers keep working.
    """
    y_true, y_pred = _clean_pair(y_true, y_pred)
    mse = float(mean_squared_error(y_true, y_pred))
    return {
        "rmse": float(np.sqrt(mse)),
        "mse": mse,
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
        "spearman": safe_spearman(y_true, y_pred),
        "kendall": safe_kendall(y_true, y_pred),
        "pearson": safe_pearson(y_true, y_pred),
        "ndcg": ndcg_at_k(y_true, y_pred, k=None),
        "topk_recall": topk_recall(y_true, y_pred, k=topk),
    }
