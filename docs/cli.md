# CLI usage

One command, four subcommands:

```bash
plm-regressor train   run.yaml
plm-regressor predict --run-dir runs/X --candidate-csv cand.csv --predict-seq-col Sequence
plm-regressor embed   --plm esm2 --input-csv X.csv --seq-col Sequence --output-npz embeddings/esm2.npz
plm-regressor list    # every model + pLM and whether it's available
```

## 1 · (Optional) extract embeddings

Computed once and cached by sequence hash, so re-runs are cheap:

```bash
plm-regressor embed --plm esm2 \
  --input-csv data/candidates.csv --seq-col Sequence \
  --output-npz zeroshot_embeds/esm2.npz
```

Swap `--plm` for any available model (`plm-regressor list`): `esm2_150m`, `esmplusplus_small`,
`protT5`, `prostT5`, `protbert`, `ankh_base`, …

## 2 · Train from a config

```yaml
# run.yaml
csv: data/activity.csv
seq_col: Protein_Seq
target_col: Activity        # any numeric property: Kd, Tm, pH, ...
id_col: Code
embedding_dir: embeddings

feature_sources: [esm2, tabular]   # tabular = use the extra columns below
extra_feature_cols: [pH, temp]
categorical_cols: [buffer]

models: [ridge, svr_rbf, hist_gb, mlp_torch]
metric: spearman            # spearman|pearson|kendall|r2|ndcg|rmse|mse|mae
auto_size: true             # size engine picks CV / budget / model gating
out_dir: runs/activity
```

```bash
plm-regressor train run.yaml
```

Useful config keys: `standard_search` (single sources, raw features only), `no_uncertainty`,
`top_ensemble`, `n_trials` (override the auto budget), `group_col`, `wt_sequence`
(enables delta/raw+delta feature modes), `use_gpu`.

## 3 · Rank candidates

```bash
plm-regressor predict --run-dir runs/activity \
  --candidate-csv data/candidates.csv \
  --predict-seq-col Sequence \
  --candidate-embedding-dir zeroshot_embeds \
  --top-n 100
```

Writes `candidate_predictions.csv` + `top_{10,50,100}.csv`. Re-rank an existing prediction CSV
with `--pred-csv ...` instead of `--run-dir`.

## Outputs per run

`best_model.joblib`, `oof_predictions.csv`, `train_predictions.csv`, `search_history.csv`,
`fold_metrics.csv`, `run_summary.json`, `coverage_report.json`, `run_config.yaml`,
`run_report.json` + `run_report.html` (with a `plots/` folder). With uncertainty:
`uncertainty_ensemble.joblib`, `top_ensemble_members.csv`. With tabular features:
`tabular_encoder.joblib`.
