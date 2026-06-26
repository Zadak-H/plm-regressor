# How it works

PLM-Regressor is a two-stage, rank-first regression workflow.

```
sequence ──► frozen protein LM ──► pooled embedding ─┐
(+ extra columns) ──────────────► tabular features ──┼─► preprocessing ─► regressor ─► property
(one-hot / blosum) ─────────────► positional enc. ───┘        (Optuna-searched pipeline)
```

The big pretrained pLM is a **frozen feature extractor** — only a light regressor head is trained,
which is the right call for small data.

## Feature assembly

Each chosen source becomes a matrix; subsets are searched (`esm2`, `esm2+tabular`, …). With a
`wt_sequence`, features can be `raw`, `delta` (variant − WT), or `raw+delta`. Tabular columns are
numeric pass-through + one-hot categoricals; scaling happens **inside** each CV fold to avoid
leakage.

## Search

A single Optuna **TPE** study jointly tunes: feature subset, feature mode, model family,
preprocessing (variance filter, scaler, feature selection, PCA/SVD), optional quantile target
transform, and per-model hyperparameters. The CNN is constrained to positional encodings
(invalid combinations are pruned).

## Size-aware engine

The number of (replicate-aggregated) rows selects a **profile** that sets the CV strategy, trial
budget, and which models are eligible:

| tier | rows | CV | budget | gating |
|------|------|----|--------|--------|
| tiny | <300 | RepeatedKFold 5×3 | ~100 | all |
| small | 300–1k | 5-fold (group-aware) | ~80 | all |
| medium | 1k–5k | 5-fold | ~60 | drop exact GPR |
| large | 5k–50k | 3-fold / holdout | ~40 | drop GPR/SVR-rbf/KernelRidge |
| xlarge | 50k–500k | holdout | ~30 | scalable only; tune on ≤20k rows, refit on all |
| huge | 500k–1M+ | holdout | ~20 | scalable only; mmap embeddings |

For big data, hyperparameters are tuned on a representative **subsample**, then the single best
configuration is **refit on the full dataset** for deployment.

## Model selection & evaluation

Selection uses **out-of-fold (OOF)** predictions — never training-set fit — with a leakage-safe
splitter (GroupKFold by sequence when enough groups exist). Reported metrics: Spearman, Pearson,
Kendall, R², RMSE, MSE, MAE, NDCG, top-k recall.

## Uncertainty

Optional and non-mandatory:

- **Conformal** prediction intervals from OOF residuals.
- **Ensemble disagreement** across the top-N trial models.
- **Native** predictive std for models that expose it (GPR, BayesianRidge).

## Deployment

`best_model.joblib` is the primary artifact (a self-contained `FittedRunModel` holding the
estimator, feature spec, WT reference, and conformal width). `plm-regressor predict` scores new
candidate sequences and ranks them, writing `top_{10,50,100}.csv`.
