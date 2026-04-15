#!/usr/bin/env python
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import gridspec
from scipy import stats


def make_scatter(pred_csv, target_col, out_png, pred_col="y_pred",
                             subset_min=None, subset_max=None):
    df = pd.read_csv(pred_csv)

    subset = df.copy()
    if subset_min is not None:
        subset = subset[subset[target_col] >= subset_min].copy()
    if subset_max is not None:
        subset = subset[subset[target_col] <= subset_max].copy()

    if pred_col not in subset.columns:
        if "y_pred" in subset.columns:
            pred_col = "y_pred"
        elif "oof_pred" in subset.columns:
            pred_col = "oof_pred"
        elif "y_pred_oof" in subset.columns:
            pred_col = "y_pred_oof"
        elif "prediction" in subset.columns:
            pred_col = "prediction"

    if pred_col not in subset.columns:
        raise ValueError(f"Prediction column '{pred_col}' not found in CSV. Columns: {subset.columns.tolist()}")
    if target_col not in subset.columns:
        raise ValueError(f"Target column '{target_col}' not found in CSV. Columns: {subset.columns.tolist()}")
    if len(subset) < 2:
        raise ValueError("Scatter plot requires at least two rows after optional subsetting.")

    rho, pval = stats.spearmanr(subset[target_col], subset[pred_col])
    pearson_r, pearson_p = stats.pearsonr(subset[target_col], subset[pred_col])

    slope, intercept = np.polyfit(subset[target_col], subset[pred_col], 1)
    x_vals = np.linspace(subset[target_col].min(), subset[target_col].max(), 200)
    y_reg = slope * x_vals + intercept

    plt.rcParams.update({
        "font.size": 15,
        "axes.labelsize": 18,
        "axes.titlesize": 20,
        "xtick.labelsize": 13,
        "ytick.labelsize": 13,
    })

    fig = plt.figure(figsize=(7, 7))
    gs = gridspec.GridSpec(
        2, 2,
        width_ratios=[4, 1.2],
        height_ratios=[1.2, 4],
        wspace=0.05,
        hspace=0.05
    )

    ax_main = fig.add_subplot(gs[1, 0])
    ax_top = fig.add_subplot(gs[0, 0], sharex=ax_main)
    ax_right = fig.add_subplot(gs[1, 1], sharey=ax_main)

    ax_main.scatter(
        subset[target_col],
        subset[pred_col],
        s=65,
        alpha=0.6,
        color="blue",
        marker="o",
    )

    min_val = min(subset[target_col].min(), subset[pred_col].min())
    max_val = max(subset[target_col].max(), subset[pred_col].max())
    pad = 0.05 * (max_val - min_val) if max_val > min_val else 1.0
    lims = [min_val - pad, max_val + pad]

    ax_main.plot(
        lims,
        lims,
        linestyle="--",
        linewidth=1.5,
        color="black"
    )

    ax_main.plot(
        x_vals,
        y_reg,
        linestyle="-",
        linewidth=1.5,
        color="red"
    )

    ax_main.set_xlim(lims)
    ax_main.set_ylim(lims)

    ax_main.set_xlabel(f"Measured {target_col}")
    ax_main.set_ylabel(f"Predicted {target_col}")

    for spine in ["top", "right", "bottom", "left"]:
        ax_main.spines[spine].set_visible(True)
        ax_main.spines[spine].set_linewidth(1.2)

    ax_main.grid(True, linestyle=":", linewidth=0.7)

    fig.text(
        0.45, 0.12,
        f"Spearman ρ = {rho:.2f}",
        ha="left",
        va="bottom",
        fontsize=13,
    )

    ax_top.hist(subset[target_col], bins=15, alpha=0.8, color="#7DC9FF")
    ax_top.grid(True, linestyle=":", linewidth=0.7)
    ax_top.tick_params(
        bottom=False,
        labelbottom=False,
        left=False,
        labelleft=False
    )

    ax_right.hist(
        subset[pred_col],
        bins=15,
        orientation="horizontal",
        alpha=0.8,
        color="#FF9B7D"
    )
    ax_right.grid(True, linestyle=":", linewidth=0.7)
    ax_right.tick_params(
        left=False,
        labelleft=False,
        bottom=False,
        labelbottom=False
    )

    fig.suptitle(f"Model Performance: Measured vs Predicted {target_col}", y=0.96)

    plt.tight_layout(rect=[0, 0.15, 1, 0.94])
    fig.savefig(out_png, dpi=300)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pred-csv", required=True, help="CSV with true + prediction columns")
    p.add_argument("--target-col", required=True, help="Name of true target column (e.g. Activity)")
    p.add_argument("--out-png", required=True, help="Output PNG path")
    p.add_argument("--subset-min", type=float, default=None)
    p.add_argument("--subset-max", type=float, default=None)
    p.add_argument("--pred-col", default="y_pred", help="Name of prediction column")
    args = p.parse_args()

    make_scatter(
        args.pred_csv,
        args.target_col,
        args.out_png,
        pred_col=args.pred_col,
        subset_min=args.subset_min,
        subset_max=args.subset_max,
    )


if __name__ == "__main__":
    main()
