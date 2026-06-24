# PLM-Regressor

> **General sequence → property regression for protein engineering.** Turn a CSV of
> sequences + a numeric property (activity, Kd, Tm, pH, …) into a trained model and a
> ranked candidate list — with a no-code web GUI.

[Get started](install.md){ .md-button .md-button--primary }
[GUI usage](gui.md){ .md-button }
[GitHub](https://github.com/Zadak-H/plm-regressor){ .md-button }

## What it does

- **Many feature sources** — protein language model embeddings (ESM2 8M–15B, ESM1/1b/1v,
  **ESM C / Cambrian** and the HF mirror **ESM++**, ProtT5, ProstT5, ProtBert, ProSST, Ankh,
  CARP), simple encodings (one-hot, BLOSUM62), and **extra tabular columns** (pH, temperature,
  assay conditions) fed alongside sequence features.
- **Many regressors** — a large classical zoo (ridge, elasticnet, SVR, KNN, RF, ExtraTrees,
  HistGB, XGBoost, LightGBM, PLS, KernelRidge, GPR, BayesianRidge, SGD) **plus deep models**:
  `mlp_torch` (FNN over embeddings) and `cnn1d` (1D-CNN over one-hot/BLOSUM).
- **Size-aware Optuna search** — auto-picks CV strategy, trial budget, and eligible models from
  the dataset size (100 → 1M+); big data uses subsample-tuning + full-data refit.
- **Leakage-safe CV + OOF model selection**, conformal + ensemble **uncertainty**, and a rich
  report (Spearman, Pearson, Kendall, R², RMSE, MSE, MAE, NDCG, top-k recall + plots).
- **Streamlit GUI** and a thin CLI over one config object.

## 60-second tour

```bash
pip install -e ".[all]"     # or pick extras: .[deep] .[esm] .[t5] .[gui]
plm-regressor gui                # point-and-click: upload CSV → pick features/models → Run → rank
```

Prefer the command line? Write a tiny `run.yaml` and:

```bash
plm-regressor train run.yaml
plm-regressor predict --run-dir runs/activity --candidate-csv candidates.csv --predict-seq-col Sequence
```

See [GUI usage](gui.md) and [CLI usage](cli.md).

## How model selection scales with data size

| rows | CV | tuned models |
|------|----|--------------|
| <300 | RepeatedKFold 5×3 | all |
| 300–1k | 5-fold (group-aware) | all |
| 1k–5k | 5-fold | all but exact GPR |
| 5k–50k | 3-fold / holdout | drop GPR/SVR-rbf/KernelRidge |
| 50k–500k | holdout | scalable only (SGD/Ridge, HistGB/XGB/LGB, torch) |
| 500k–1M+ | holdout | scalable only (mmap embeddings) |

Full list of regressors and pLMs: [Models & pLMs](models.md).
