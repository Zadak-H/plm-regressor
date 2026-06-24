#!/usr/bin/env python3
"""Plotting helpers for the run report. All functions save a PNG and return its path."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib import gridspec  # noqa: E402
from scipy import stats  # noqa: E402


def scatter_pred_vs_true(pred_csv, target_col, out_png, pred_col="y_pred",
                         subset_min=None, subset_max=None) -> Path:
    """Measured vs predicted scatter with marginal histograms (port of the
    original ``scripts/scatter_plot.py::make_scatter``)."""
    df = pd.read_csv(pred_csv)
    subset = df.copy()
    if subset_min is not None:
        subset = subset[subset[target_col] >= subset_min]
    if subset_max is not None:
        subset = subset[subset[target_col] <= subset_max]
    subset = subset[subset[target_col].notna() & subset[pred_col].notna()]
    if len(subset) < 2:
        raise ValueError("Scatter requires >= 2 valid rows")

    rho = stats.spearmanr(subset[target_col], subset[pred_col]).correlation
    pear = stats.pearsonr(subset[target_col], subset[pred_col])[0]
    slope, intercept = np.polyfit(subset[target_col], subset[pred_col], 1)
    x_vals = np.linspace(subset[target_col].min(), subset[target_col].max(), 200)

    fig = plt.figure(figsize=(7, 7))
    gs = gridspec.GridSpec(2, 2, width_ratios=[4, 1.2], height_ratios=[1.2, 4], wspace=0.05, hspace=0.05)
    ax_main = fig.add_subplot(gs[1, 0])
    ax_top = fig.add_subplot(gs[0, 0], sharex=ax_main)
    ax_right = fig.add_subplot(gs[1, 1], sharey=ax_main)

    ax_main.scatter(subset[target_col], subset[pred_col], s=55, alpha=0.6, color="blue")
    lo = min(subset[target_col].min(), subset[pred_col].min())
    hi = max(subset[target_col].max(), subset[pred_col].max())
    pad = 0.05 * (hi - lo) if hi > lo else 1.0
    lims = [lo - pad, hi + pad]
    ax_main.plot(lims, lims, "--", lw=1.5, color="black")
    ax_main.plot(x_vals, slope * x_vals + intercept, "-", lw=1.5, color="red")
    ax_main.set_xlim(lims); ax_main.set_ylim(lims)
    ax_main.set_xlabel(f"Measured {target_col}"); ax_main.set_ylabel(f"Predicted {target_col}")
    ax_main.grid(True, linestyle=":", linewidth=0.7)
    ax_main.text(0.05, 0.92, f"Spearman={rho:.3f}\nPearson={pear:.3f}", transform=ax_main.transAxes,
                 va="top", fontsize=12, bbox=dict(boxstyle="round", fc="white", alpha=0.7))

    ax_top.hist(subset[target_col], bins=15, alpha=0.8, color="#7DC9FF")
    ax_top.tick_params(bottom=False, labelbottom=False, left=False, labelleft=False)
    ax_right.hist(subset[pred_col], bins=15, orientation="horizontal", alpha=0.8, color="#FF9B7D")
    ax_right.tick_params(left=False, labelleft=False, bottom=False, labelbottom=False)
    fig.suptitle(f"Measured vs Predicted {target_col}", y=0.96)
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out_png, dpi=200)
    plt.close(fig)
    return Path(out_png)


def residual_plot(pred_csv, out_png, pred_col="y_pred", true_col="y_true") -> Path:
    df = pd.read_csv(pred_csv)
    df = df[df[pred_col].notna() & df[true_col].notna()]
    resid = df[true_col] - df[pred_col]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axes[0].scatter(df[pred_col], resid, s=30, alpha=0.6, color="purple")
    axes[0].axhline(0, color="black", lw=1)
    axes[0].set_xlabel("Predicted"); axes[0].set_ylabel("Residual"); axes[0].set_title("Residuals vs predicted")
    axes[0].grid(True, linestyle=":", linewidth=0.7)
    axes[1].hist(resid, bins=25, color="#9B7DFF", alpha=0.85)
    axes[1].set_xlabel("Residual"); axes[1].set_ylabel("Count"); axes[1].set_title("Residual distribution")
    plt.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)
    return Path(out_png)


def model_comparison_bar(search_history_csv, out_png, metric="spearman") -> Optional[Path]:
    df = pd.read_csv(search_history_csv)
    if "model_name" not in df.columns or df.empty:
        return None
    # best trial value per model
    best = df.groupby("model_name")["value"].max().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(max(6, 0.7 * len(best)), 4.5))
    ax.bar(best.index.astype(str), best.values, color="#2C7FB8")
    ax.set_ylabel(f"best {metric} (OOF)")
    ax.set_title("Best score by model family")
    ax.tick_params(axis="x", rotation=45)
    plt.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)
    return Path(out_png)


def calibration_plot(pred_csv, out_png, pred_col="y_pred", true_col="y_true",
                     std_col="pred_ensemble_std") -> Optional[Path]:
    df = pd.read_csv(pred_csv)
    if std_col not in df.columns or df[std_col].isna().all():
        return None
    df = df[df[pred_col].notna() & df[true_col].notna() & df[std_col].notna() & (df[std_col] > 0)]
    if len(df) < 5:
        return None
    z = np.abs(df[true_col] - df[pred_col]) / df[std_col]
    levels = np.linspace(0.1, 0.99, 20)
    from scipy.stats import norm
    empirical = [float(np.mean(z <= norm.ppf(0.5 + lvl / 2))) for lvl in levels]
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "--", color="black", label="ideal")
    ax.plot(levels, empirical, "-o", color="#E6550D", label="observed")
    ax.set_xlabel("Expected coverage"); ax.set_ylabel("Observed coverage")
    ax.set_title("Uncertainty calibration"); ax.legend(); ax.grid(True, linestyle=":")
    plt.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)
    return Path(out_png)
