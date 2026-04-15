# Catalytica MLDE

> Rank-first MLDE for small-data protein engineering.

[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](requirements.txt)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Workflow Guide](https://img.shields.io/badge/guide-field--guide-orange.svg)](field-guide/MLDE_WORKFLOW.md)

Catalytica MLDE is a practical machine-learning-directed evolution workflow for protein engineering when labels are scarce and ranking quality matters more than everything else.

It is designed for:

- selecting the best supervised model from out-of-fold predictions
- searching across learned embeddings and simple sequence encodings
- scoring zero-shot candidate sequences with the chosen saved model
- optionally adding uncertainty without making uncertainty mandatory

## Why Catalytica MLDE

Most small-data protein MLDE setups need a few things to work well in practice:

- leakage-safe cross-validation
- rank-aware model selection
- flexible but debuggable feature search
- clean zero-shot deployment
- a lightweight path when you do not want uncertainty or feature-combination search

This repo now supports all of those in a single workflow.

## Quick Start

### 1. Install the environment

Conda:

```bash
conda env create -f PET.yml
conda activate pet
```

or venv:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 2. Run a standard supervised search

This compares the provided feature sources as single models only, with no feature-combination search and no uncertainty layer:

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

### 3. Compare supervised runs

```bash
python scripts/select_best_supervised_model.py \
  --run-dirs runs/activity_standard_no_unc \
  --metric spearman \
  --plot-best
```

### 4. Rank zero-shot candidates

```bash
python scripts/rank_zero_shot_candidates.py \
  --run-dir runs/activity_standard_no_unc \
  --candidate-csv data/zero_shot_mutants.csv \
  --predict-seq-col Sequence \
  --candidate-embedding-dir zeroshot_embeds \
  --top-n 100
```

## Repository Map

- `data/`: supervised datasets and zero-shot candidate CSVs
- `embeddings/`: learned embedding banks used for supervised search
- `scripts/`: training, evaluation, plotting, and ranking entrypoints
- `scripts/embeds_scripts/`: embedding extractors and cluster runner
- `field-guide/`: long-form usage docs and workflow reference

## Main Scripts

- `scripts/training3_optuna_mlde_uncertainty.py`
- `scripts/select_best_supervised_model.py`
- `scripts/rank_zero_shot_candidates.py`
- `scripts/scatter_plot.py`

## Docs

Start here for the full run order, commands, and output-file descriptions:

- [field-guide/MLDE_WORKFLOW.md](field-guide/MLDE_WORKFLOW.md)

## Notes

- Use `oof_predictions.csv` for supervised model selection.
- `--standard` means single provided feature sources only, raw features only, no PCA/SVD, and no quantile target transform.
- `--no-uncertainty` skips ensemble disagreement and interval generation while keeping the same deployment path.
- `best_model.joblib` is always the primary deployment artifact.

## Contributing

Bug reports, feature requests, and pull requests are welcome.

- Contribution guide: [CONTRIBUTING.md](CONTRIBUTING.md)
- License: [MIT](LICENSE)

## License

This project is released under the [MIT License](LICENSE).
