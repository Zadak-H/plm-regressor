#!/usr/bin/env python3
"""
Compare standardized supervised OOF outputs and pick the best run.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

from mlde_utils import compute_metrics, load_json
from scatter_plot import make_scatter


def metric_direction(metric_name: str) -> str:
    return "minimize" if metric_name in {"rmse", "mae"} else "maximize"


def resolve_inputs(run_dirs: List[str], pred_csvs: List[str]) -> List[Tuple[str, Path, Path | None]]:
    resolved: List[Tuple[str, Path, Path | None]] = []
    for run_dir in run_dirs:
        run_path = Path(run_dir)
        pred_path = run_path / "oof_predictions.csv"
        if not pred_path.exists():
            raise FileNotFoundError(f"Expected OOF CSV at {pred_path}")
        resolved.append((run_path.name, pred_path, run_path / "run_summary.json"))
    for pred_csv in pred_csvs:
        pred_path = Path(pred_csv)
        if not pred_path.exists():
            raise FileNotFoundError(pred_path)
        resolved.append((pred_path.stem, pred_path, None))
    if not resolved:
        raise ValueError("Provide at least one --run-dir or --pred-csv")
    return resolved


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dirs", nargs="*", default=[], help="Run directories containing oof_predictions.csv")
    parser.add_argument("--pred-csvs", nargs="*", default=[], help="Explicit OOF prediction CSVs")
    parser.add_argument("--metric", default="spearman", choices=["spearman", "kendall", "r2", "rmse", "mae"])
    parser.add_argument("--out-csv", default="model_comparison.csv")
    parser.add_argument("--plot-best", action="store_true")
    parser.add_argument("--plot-out", default=None, help="Optional explicit PNG path for the best-run scatter plot")
    parser.add_argument("--subset-min", type=float, default=None)
    parser.add_argument("--subset-max", type=float, default=None)
    args = parser.parse_args()

    inputs = resolve_inputs(args.run_dirs, args.pred_csvs)
    descending = metric_direction(args.metric) == "maximize"
    rows: List[Dict[str, object]] = []

    for default_name, pred_path, summary_path in inputs:
        df = pd.read_csv(pred_path)
        if "y_true" not in df.columns or "y_pred" not in df.columns:
            raise ValueError(f"{pred_path} must contain 'y_true' and 'y_pred' columns")
        valid = df["y_true"].notna() & df["y_pred"].notna()
        metrics = compute_metrics(df.loc[valid, "y_true"].to_numpy(), df.loc[valid, "y_pred"].to_numpy())
        run_name = default_name
        best_model_name = None
        best_subset = None
        if summary_path is not None and summary_path.exists():
            summary = load_json(summary_path)
            run_name = summary.get("run_name", run_name)
            best_model_name = summary.get("best_model_name")
            best_subset = "+".join(summary.get("best_feature_subset", []))
        rows.append(
            {
                "run_name": run_name,
                "pred_csv": str(pred_path),
                "rmse": metrics["rmse"],
                "mae": metrics["mae"],
                "r2": metrics["r2"],
                "spearman": metrics["spearman"],
                "kendall": metrics["kendall"],
                "best_model_name": best_model_name,
                "best_feature_subset": best_subset,
                "n_rows": int(valid.sum()),
            }
        )

    comparison = pd.DataFrame(rows).sort_values(args.metric, ascending=not descending).reset_index(drop=True)
    comparison.to_csv(args.out_csv, index=False)

    best_row = comparison.iloc[0]
    best_pred_csv = Path(best_row["pred_csv"])
    if args.plot_best:
        if args.plot_out is not None:
            plot_path = Path(args.plot_out)
        elif best_pred_csv.parent.is_dir():
            plot_path = best_pred_csv.parent / "best_supervised_scatter.png"
        else:
            plot_path = Path("best_supervised_scatter.png")
        make_scatter(
            pred_csv=str(best_pred_csv),
            target_col="y_true",
            out_png=str(plot_path),
            pred_col="y_pred",
            subset_min=args.subset_min,
            subset_max=args.subset_max,
        )
        print(f"Saved scatter plot to: {plot_path}")

    print(comparison.to_string(index=False))
    print(f"Saved comparison table to: {args.out_csv}")
    print(f"Best run by {args.metric}: {best_row['run_name']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
