#!/usr/bin/env python3
"""Assemble the per-run report: metric table + plots -> run_report.json + .html."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import pandas as pd

from .core import load_json, save_json
from .metrics import compute_metrics
from . import plots


def _oof_metrics(out_dir: Path) -> Dict[str, float]:
    oof = pd.read_csv(out_dir / "oof_predictions.csv")
    valid = oof["y_true"].notna() & oof["y_pred"].notna()
    return compute_metrics(oof.loc[valid, "y_true"].to_numpy(), oof.loc[valid, "y_pred"].to_numpy())


def build_report(out_dir: str | Path, target_col: str, metric: str = "spearman") -> Path:
    out_dir = Path(out_dir)
    oof_csv = out_dir / "oof_predictions.csv"
    summary = load_json(out_dir / "run_summary.json") if (out_dir / "run_summary.json").exists() else {}
    metrics = _oof_metrics(out_dir)

    images: List[str] = []
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    def _try(fn, *args, **kwargs):
        try:
            path = fn(*args, **kwargs)
            if path is not None:
                images.append(Path(path).name if Path(path).parent == plots_dir else str(Path(path)))
        except Exception as exc:  # pragma: no cover
            print(f"[warn] plot {fn.__name__} skipped: {exc}")

    _try(plots.scatter_pred_vs_true, oof_csv, "y_true", plots_dir / "scatter.png", "y_pred")
    _try(plots.residual_plot, oof_csv, plots_dir / "residuals.png")
    if (out_dir / "search_history.csv").exists():
        _try(plots.model_comparison_bar, out_dir / "search_history.csv", plots_dir / "model_comparison.png", metric)
    _try(plots.calibration_plot, oof_csv, plots_dir / "calibration.png")

    report = {
        "run_name": summary.get("run_name", out_dir.name),
        "primary_metric": metric,
        "oof_metrics": metrics,
        "best_model_name": summary.get("best_model_name"),
        "best_feature_subset": summary.get("best_feature_subset"),
        "best_feature_mode": summary.get("best_feature_mode"),
        "size_tier": summary.get("size_tier"),
        "cv_strategy": summary.get("cv_strategy"),
        "eligible_models": summary.get("eligible_models"),
        "plots": images,
    }
    save_json(out_dir / "run_report.json", report)
    _write_html(out_dir, report, metrics, plots_dir)
    return out_dir / "run_report.html"


def _write_html(out_dir: Path, report: dict, metrics: Dict[str, float], plots_dir: Path) -> None:
    rows = "".join(
        f"<tr><td>{k}</td><td>{v:.4f}</td></tr>" for k, v in metrics.items() if isinstance(v, (int, float))
    )
    imgs = "".join(
        f'<div class="card"><img src="plots/{Path(name).name}"></div>'
        for name in report.get("plots", [])
    )
    html = f"""<!doctype html><html><head><meta charset="utf-8"><title>PLM-Regressor report: {report['run_name']}</title>
<style>
body{{font-family:system-ui,Arial,sans-serif;margin:24px;color:#222}}
h1{{margin-bottom:0}} .sub{{color:#666;margin-top:4px}}
table{{border-collapse:collapse;margin:16px 0}} td,th{{border:1px solid #ccc;padding:6px 12px;text-align:left}}
.card{{display:inline-block;margin:10px;vertical-align:top}} img{{max-width:560px;border:1px solid #eee;border-radius:6px}}
.kv{{background:#f6f8fa;padding:10px 14px;border-radius:6px;display:inline-block;margin:4px}}
</style></head><body>
<h1>PLM-Regressor run report</h1>
<div class="sub">{report['run_name']} &middot; tier: {report.get('size_tier')} &middot; CV: {report.get('cv_strategy')}</div>
<p>
<span class="kv">best model: <b>{report.get('best_model_name')}</b></span>
<span class="kv">features: <b>{'+'.join(report.get('best_feature_subset') or [])}</b> ({report.get('best_feature_mode')})</span>
<span class="kv">primary metric: <b>{report.get('primary_metric')}</b></span>
</p>
<h2>OOF metrics</h2>
<table><tr><th>metric</th><th>value</th></tr>{rows}</table>
<h2>Plots</h2>
<div>{imgs}</div>
</body></html>"""
    (out_dir / "run_report.html").write_text(html, encoding="utf-8")
