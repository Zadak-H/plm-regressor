# HelixForge Field Guide

This repo supports two practical MLDE workflows from the command line:

- `Standard` mode:
  single provided feature sources only, no feature-combination search, raw features only, no PCA/SVD, no quantile target transform
- `Full` mode:
  non-empty feature-subset search across the provided feature sources, optional uncertainty outputs, optional WT-aware feature modes

This guide assumes you run commands from the repo root:

```bash
cd <repo-root>
```

## 1. Environment Setup

Use one of the following.

### Option A: conda

```bash
conda env create -f PET.yml
conda activate pet
```

### Option B: venv + pip

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Optional check:

```bash
python scripts/training3_optuna_mlde_uncertainty.py --help
python scripts/select_best_supervised_model.py --help
python scripts/rank_zero_shot_candidates.py --help
```

## 2. Expected Input Files

### Supervised training CSV

Typical columns:

- `Code`: row ID
- `Protein_Seq`: amino-acid sequence
- target column such as `Activity`, `Weight_Loss`, or `Tm`

Examples already in the repo:

- `data/21h_activity_TPA_release_0_1MKPB.csv`
- `data/21h_activity_TPA_release_1MKPB.csv`
- `data/PETdegrad_weight_loss_0_1MKPB_.csv`
- `data/PETdegrad_weight_loss_1MKPB_.csv`
- `data/Tm_for_variants_1-335.csv`

### Zero-shot candidate CSV

Typical columns:

- `ID`
- `Sequence`

Example already in the repo:

- `data/zero_shot_mutants.csv`

### Embedding NPZ files

For the training script, learned embeddings are expected as:

- `embeddings/esm1v.npz`
- `embeddings/esm2.npz`
- `embeddings/prosst.npz`
- `embeddings/protT5.npz`
- `embeddings/protT5_half.npz`
- `embeddings/prostT5.npz`

Each NPZ should contain `embeddings` and `sequences`.

## 3. Feature Sources You Can Search

Learned embeddings:

- `esm1v`
- `esm2`
- `prosst`
- `protT5`
- `protT5_half`
- `prostT5`

Simple sequence encodings:

- `onehot`
- `blosum62`

## 4. Embedding Generation

You only need zero-shot candidate embeddings for the learned embedding sources you plan to use at prediction time.

### Local single-script examples

ESM2:

```bash
python scripts/embeds_scripts/extract_esm2_embeddings.py \
  --input-csv data/zero_shot_mutants.csv \
  --seq-col Sequence \
  --output-npz zeroshot_embeds/esm2.npz \
  --model-size 650M \
  --batch-size 8
```

ESM1v ensemble:

```bash
python scripts/embeds_scripts/extract_esm1v_embeddings.py \
  --input-csv data/zero_shot_mutants.csv \
  --seq-col Sequence \
  --output-npz zeroshot_embeds/esm1v.npz \
  --model-names esm1v_t33_650M_UR90S_1,esm1v_t33_650M_UR90S_2,esm1v_t33_650M_UR90S_3,esm1v_t33_650M_UR90S_4,esm1v_t33_650M_UR90S_5 \
  --max-tokens 12000
```

ProSST:

```bash
python scripts/embeds_scripts/extract_prosst_embeddings.py \
  --input-csv data/zero_shot_mutants.csv \
  --seq-col Sequence \
  --output-npz zeroshot_embeds/prosst.npz \
  --model-id AI4Protein/ProSST-2048 \
  --batch-size 8
```

ProtT5:

```bash
python scripts/embeds_scripts/extract_prot_t5_embeddings.py \
  --input-csv data/zero_shot_mutants.csv \
  --seq-col Sequence \
  --output-npz zeroshot_embeds/protT5.npz \
  --model-id Rostlab/prot_t5_xl_uniref50 \
  --batch-size 4
```

ProtT5 half encoder:

```bash
python scripts/embeds_scripts/extract_prot_t5_embeddings.py \
  --input-csv data/zero_shot_mutants.csv \
  --seq-col Sequence \
  --output-npz zeroshot_embeds/protT5_half.npz \
  --model-id Rostlab/prot_t5_xl_half_uniref50-enc \
  --batch-size 4
```

ProstT5:

```bash
python scripts/embeds_scripts/extract_prot_t5_embeddings.py \
  --input-csv data/zero_shot_mutants.csv \
  --seq-col Sequence \
  --output-npz zeroshot_embeds/prostT5.npz \
  --model-id Rostlab/ProstT5 \
  --batch-size 4
```

### Cluster array runner

```bash
sbatch scripts/embeds_scripts/zeroshot.sh
```

The runner writes outputs into `zeroshot_embeds/`.

## 5. Training Modes

### Standard mode

Use this when you want:

- single embedding or single feature source only
- no feature combinations
- no PCA/SVD
- no quantile target transform
- optionally no uncertainty

If you pass multiple feature sources, each is tested alone.

Example:

```bash
--feature-sources prostT5 esm2
```

This evaluates:

- `prostT5`
- `esm2`

It does not evaluate:

- `prostT5 + esm2`

### Full mode

Use this when you want:

- any non-empty subset of the provided feature sources
- optional uncertainty outputs
- optional WT-aware feature modes
- broader preprocessing and model-family search

## 6. Standard Mode Commands

### Standard, no uncertainty, compare `prostT5` vs `esm2`

```bash
python scripts/training3_optuna_mlde_uncertainty.py \
  --csv data/21h_activity_TPA_release_0_1MKPB.csv \
  --train-seq-col Protein_Seq \
  --target-col Activity \
  --id-col Code \
  --embedding-dir embeddings \
  --feature-sources prostT5 esm2 \
  --replicate-policy mean_by_sequence \
  --cv-splits 5 \
  --n-trials 50 \
  --metric spearman \
  --standard \
  --no-uncertainty \
  --out-dir runs/activity_standard_no_unc
```

### Standard, no uncertainty, single embedding only

```bash
python scripts/training3_optuna_mlde_uncertainty.py \
  --csv data/21h_activity_TPA_release_0_1MKPB.csv \
  --train-seq-col Protein_Seq \
  --target-col Activity \
  --id-col Code \
  --embedding-dir embeddings \
  --feature-sources esm2 \
  --replicate-policy mean_by_sequence \
  --cv-splits 5 \
  --n-trials 30 \
  --metric spearman \
  --standard \
  --no-uncertainty \
  --out-dir runs/activity_esm2_standard_no_unc
```

### Standard, single simple encoding only

```bash
python scripts/training3_optuna_mlde_uncertainty.py \
  --csv data/21h_activity_TPA_release_0_1MKPB.csv \
  --train-seq-col Protein_Seq \
  --target-col Activity \
  --id-col Code \
  --feature-sources onehot blosum62 \
  --replicate-policy mean_by_sequence \
  --cv-splits 5 \
  --n-trials 30 \
  --metric spearman \
  --standard \
  --no-uncertainty \
  --out-dir runs/activity_simple_standard_no_unc
```

## 7. Full Mode Commands

### Full MLDE search with combinations and uncertainty

```bash
python scripts/training3_optuna_mlde_uncertainty.py \
  --csv data/21h_activity_TPA_release_0_1MKPB.csv \
  --train-seq-col Protein_Seq \
  --target-col Activity \
  --id-col Code \
  --embedding-dir embeddings \
  --feature-sources esm1v esm2 prosst protT5 protT5_half prostT5 onehot blosum62 \
  --replicate-policy mean_by_sequence \
  --cv-splits 5 \
  --n-trials 100 \
  --top-ensemble 5 \
  --metric spearman \
  --out-dir runs/activity_full_unc
```

### Full MLDE search with WT-aware feature modes

Replace the WT sequence with your exact sequence string.

```bash
python scripts/training3_optuna_mlde_uncertainty.py \
  --csv data/21h_activity_TPA_release_0_1MKPB.csv \
  --train-seq-col Protein_Seq \
  --target-col Activity \
  --id-col Code \
  --embedding-dir embeddings \
  --feature-sources esm1v esm2 prosst protT5 protT5_half prostT5 onehot blosum62 \
  --replicate-policy mean_by_sequence \
  --cv-splits 5 \
  --n-trials 100 \
  --top-ensemble 5 \
  --metric spearman \
  --wt-sequence "PUT_WT_SEQUENCE_HERE" \
  --out-dir runs/activity_full_unc_wt
```

### Train and immediately score a candidate CSV in the same run

```bash
python scripts/training3_optuna_mlde_uncertainty.py \
  --csv data/21h_activity_TPA_release_0_1MKPB.csv \
  --train-seq-col Protein_Seq \
  --target-col Activity \
  --id-col Code \
  --embedding-dir embeddings \
  --feature-sources esm2 prostT5 onehot blosum62 \
  --replicate-policy mean_by_sequence \
  --cv-splits 5 \
  --n-trials 50 \
  --top-ensemble 5 \
  --metric spearman \
  --predict-csv data/zero_shot_mutants.csv \
  --predict-seq-col Sequence \
  --predict-id-col ID \
  --predict-embedding-dir zeroshot_embeds \
  --out-dir runs/activity_train_and_score
```

## 8. Model Selection and Plotting

Compare one or more run directories and automatically create the scatter plot for the best run:

```bash
python scripts/select_best_supervised_model.py \
  --run-dirs runs/activity_standard_no_unc runs/activity_full_unc \
  --metric spearman \
  --plot-best \
  --out-csv runs/model_comparison.csv
```

You can also directly plot one OOF CSV:

```bash
python scripts/scatter_plot.py \
  --pred-csv runs/activity_full_unc/oof_predictions.csv \
  --target-col y_true \
  --pred-col y_pred \
  --out-png runs/activity_full_unc/scatter_plot.png
```

Or use the shell wrapper:

```bash
bash scripts/scatter_plot.sh runs/activity_full_unc
```

## 9. Zero-Shot Ranking

### Rank candidates from a saved run

```bash
python scripts/rank_zero_shot_candidates.py \
  --run-dir runs/activity_full_unc \
  --candidate-csv data/zero_shot_mutants.csv \
  --predict-seq-col Sequence \
  --predict-id-col ID \
  --candidate-embedding-dir zeroshot_embeds \
  --top-n 100
```

This writes:

- `candidate_predictions.csv`
- `top_10.csv`
- `top_50.csv`
- `top_100.csv`

### Rerank an existing prediction CSV

```bash
python scripts/rank_zero_shot_candidates.py \
  --pred-csv runs/activity_full_unc/candidate_predictions.csv \
  --top-n 100
```

## 10. Output Files Per Training Run

Each training run directory typically contains:

- `best_model.joblib`
- `oof_predictions.csv`
- `train_predictions.csv`
- `search_history.csv`
- `fold_metrics.csv`
- `run_summary.json`
- `coverage_report.json`

If uncertainty is enabled, it also contains:

- `uncertainty_ensemble.joblib`
- `top_ensemble_members.csv`

If `--predict-csv` was used during training, it also contains:

- `candidate_predictions.csv`

## 11. What `--no-uncertainty` Changes

When you use `--no-uncertainty`:

- no `uncertainty_ensemble.joblib` is written
- no `top_ensemble_members.csv` is written
- prediction still works from `best_model.joblib`
- uncertainty columns stay present in CSV outputs for schema stability, but they are `NaN`

## 12. What `--standard` Changes

When you use `--standard` or `--standard-search`:

- the script evaluates only single provided feature sources
- feature mode is fixed to `raw`
- dimensionality reduction is forced to `none`
- target transform is forced to `none`
- model family and lighter preprocessing choices are still searched

Example:

```bash
--feature-sources prostT5 esm2
```

means:

- test `prostT5`
- test `esm2`

not:

- `prostT5 + esm2`

## 13. Recommended Run Order

### Minimal standard workflow

1. Generate only the candidate embeddings you may deploy

```bash
python scripts/embeds_scripts/extract_esm2_embeddings.py \
  --input-csv data/zero_shot_mutants.csv \
  --seq-col Sequence \
  --output-npz zeroshot_embeds/esm2.npz \
  --model-size 650M \
  --batch-size 8

python scripts/embeds_scripts/extract_prot_t5_embeddings.py \
  --input-csv data/zero_shot_mutants.csv \
  --seq-col Sequence \
  --output-npz zeroshot_embeds/prostT5.npz \
  --model-id Rostlab/ProstT5 \
  --batch-size 4
```

2. Run standard supervised search

```bash
python scripts/training3_optuna_mlde_uncertainty.py \
  --csv data/21h_activity_TPA_release_0_1MKPB.csv \
  --train-seq-col Protein_Seq \
  --target-col Activity \
  --id-col Code \
  --embedding-dir embeddings \
  --feature-sources esm2 prostT5 \
  --replicate-policy mean_by_sequence \
  --cv-splits 5 \
  --n-trials 50 \
  --metric spearman \
  --standard \
  --no-uncertainty \
  --out-dir runs/activity_standard_no_unc
```

3. Inspect OOF performance

```bash
python scripts/select_best_supervised_model.py \
  --run-dirs runs/activity_standard_no_unc \
  --metric spearman \
  --plot-best \
  --out-csv runs/activity_standard_no_unc_comparison.csv
```

4. Rank zero-shot candidates

```bash
python scripts/rank_zero_shot_candidates.py \
  --run-dir runs/activity_standard_no_unc \
  --candidate-csv data/zero_shot_mutants.csv \
  --predict-seq-col Sequence \
  --predict-id-col ID \
  --candidate-embedding-dir zeroshot_embeds \
  --top-n 100
```

### Full MLDE workflow

1. Generate candidate embeddings for the learned embedding sources you want in the search space
2. Run the full Optuna search with combinations
3. Compare run directories with `select_best_supervised_model.py`
4. Rank zero-shot candidates from the chosen run with `rank_zero_shot_candidates.py`

## 14. Practical Notes

- Use OOF results in `oof_predictions.csv` for supervised model selection. Do not select from training predictions.
- Default `replicate-policy` recommendation is `mean_by_sequence`.
- Default primary metric recommendation is `spearman`.
- For `onehot` and `blosum62`, sequences must be aligned and all the same length.
- If you use `--predict-csv` or `rank_zero_shot_candidates.py`, the candidate sequence column is usually `Sequence`, not `Protein_Seq`.
- In standard mode, the deployed candidate embeddings only need to cover the best selected feature source.

## 15. Quick Reference

Standard, no uncertainty:

```bash
python scripts/training3_optuna_mlde_uncertainty.py \
  --csv data/21h_activity_TPA_release_0_1MKPB.csv \
  --train-seq-col Protein_Seq \
  --target-col Activity \
  --id-col Code \
  --embedding-dir embeddings \
  --feature-sources esm2 prostT5 \
  --standard \
  --no-uncertainty \
  --out-dir runs/activity_standard_no_unc
```

Full search with uncertainty:

```bash
python scripts/training3_optuna_mlde_uncertainty.py \
  --csv data/21h_activity_TPA_release_0_1MKPB.csv \
  --train-seq-col Protein_Seq \
  --target-col Activity \
  --id-col Code \
  --embedding-dir embeddings \
  --feature-sources esm1v esm2 prosst protT5 protT5_half prostT5 onehot blosum62 \
  --top-ensemble 5 \
  --out-dir runs/activity_full_unc
```

Compare runs:

```bash
python scripts/select_best_supervised_model.py \
  --run-dirs runs/activity_standard_no_unc runs/activity_full_unc \
  --metric spearman \
  --plot-best
```

Rank candidates:

```bash
python scripts/rank_zero_shot_candidates.py \
  --run-dir runs/activity_standard_no_unc \
  --candidate-csv data/zero_shot_mutants.csv \
  --predict-seq-col Sequence \
  --candidate-embedding-dir zeroshot_embeds \
  --top-n 100
```
