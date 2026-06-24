<h1 align="center">PLM-Regressor</h1>

<p align="center">
  <b>General sequence → property regression for protein engineering.</b><br>
  Turn a CSV of sequences + a numeric property (activity, Kd, Tm, pH, …) into a trained model
  and a ranked candidate list — with a no-code web GUI.
</p>

<p align="center">
  <a href="https://zadak-h.github.io/plm-regressor/"><img src="https://img.shields.io/badge/docs-website-teal.svg" alt="Docs"></a>
  <a href="requirements.txt"><img src="https://img.shields.io/badge/python-3.9%2B-blue.svg" alt="Python"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-green.svg" alt="License: MIT"></a>
</p>

<p align="center">
  📖 <b><a href="https://zadak-h.github.io/plm-regressor/">Documentation &amp; GUI guide</a></b>
</p>

---

## What it does

- **Many feature sources** — protein language model embeddings (ESM2 8M–15B, ESM1/1b/1v,
  **ESM C / Cambrian** + the HF mirror **ESM++**, ProtT5, ProstT5, ProtBert, ProSST, Ankh, CARP),
  simple encodings (one-hot, BLOSUM62), and **extra tabular columns** (pH, temperature, assay
  conditions) fed alongside sequence features.
- **Many regressors** — a large classical zoo (ridge, elasticnet, SVR, KNN, RF, ExtraTrees,
  HistGB, XGBoost, LightGBM, PLS, KernelRidge, GPR, BayesianRidge, SGD) **plus deep models**:
  `mlp_torch` (FNN over embeddings) and `cnn1d` (1D-CNN over one-hot/BLOSUM).
- **Size-aware Optuna search** — auto-picks CV strategy, trial budget, and eligible models from
  the dataset size (100 → 1M+); big data uses subsample-tuning + full-data refit.
- **Leakage-safe CV + OOF model selection**, conformal + ensemble **uncertainty**, and a rich
  report (Spearman, Pearson, Kendall, R², RMSE, MSE, MAE, NDCG, top-k recall + plots).
- **Streamlit GUI** and a thin CLI over one config object.

## Install

```bash
git clone https://github.com/Zadak-H/plm-regressor.git
cd plm-regressor
python -m pip install -e ".[all]"      # or pick extras: .[deep] .[esm] .[t5] .[gui]
```

## Quick start

**GUI (no code):**

```bash
plm-regressor gui
```

Upload a CSV → pick the target + any extra columns → tick pLMs + regressors → **Run** → read the
report → rank candidates. See the [GUI guide](https://zadak-h.github.io/plm-regressor/gui/).

**CLI:**

```bash
plm-regressor train run.yaml        # train from a config
plm-regressor predict --run-dir runs/activity --candidate-csv cand.csv --predict-seq-col Sequence
plm-regressor embed --plm esm2 --input-csv cand.csv --seq-col Sequence --output-npz embeddings/esm2.npz
plm-regressor list                  # every model + pLM and whether it's available
```

A minimal `run.yaml`:

```yaml
csv: data/activity.csv
seq_col: Protein_Seq
target_col: Activity
embedding_dir: embeddings
feature_sources: [esm2, tabular]
extra_feature_cols: [pH, temp]
models: [ridge, svr_rbf, hist_gb, mlp_torch]
metric: spearman
auto_size: true
out_dir: runs/activity
```

## Documentation

Full docs (install, GUI usage, CLI, model/pLM tables, how it works) live at
**<https://zadak-h.github.io/plm-regressor/>** (source in [`docs/`](docs/)).

## Repository layout

- `plm_regressor/` — the framework package (config, registry, sizing, features, models, search, train,
  predict, metrics, plots, report, cli) + `plm_regressor/embeddings/` extractors
- `app.py` — Streamlit GUI
- `docs/` — documentation site (MkDocs Material)
- `data/`, `embeddings/` — example datasets and precomputed embedding banks
- `scripts/` — the original standalone CLI (still works)

## License

[MIT](LICENSE).
