# Install

Requires Python 3.9+.

## From source (recommended for now)

```bash
git clone https://github.com/Zadak-H/plm-regressor.git
cd plm-regressor
python -m pip install -e ".[all]"
```

`.[all]` pulls core + boosting + deep + ESM + ProtT5 + GUI. Pick smaller extras if you prefer:

| extra | brings | for |
|-------|--------|-----|
| (core) | numpy, pandas, scikit-learn, scipy, optuna, matplotlib, pyyaml | always |
| `.[deep]` | torch | `mlp_torch`, `cnn1d` |
| `.[boost]` | xgboost, lightgbm | gradient boosting |
| `.[esm]` | fair-esm | ESM2 / ESM1 embeddings |
| `.[t5]` | transformers, sentencepiece | ProtT5 / ProstT5 / ProtBert / Ankh / ESM++ |
| `.[gui]` | streamlit | the web GUI |

## Conda

```bash
conda env create -f PET.yml
conda activate pet
```

## Verify

```bash
plm-regressor list      # prints every regressor + pLM and whether it is available
plm-regressor gui       # launches the web app
```

## Notes on optional model backends

- **ESM C native** (`esmc_300m`, `esmc_600m`) needs EvolutionaryScale's `esm` SDK, which imports
  as `esm` and **collides with `fair-esm`**. Install it in a *separate* environment, or just use
  **ESM++** (`esmplusplus_small`, `esmplusplus_large`) which gives the same embeddings through
  `transformers` with no conflict.
- **LightGBM** and **CARP** are optional; if not installed, those models simply don't appear in
  `plm-regressor list`.
- A CUDA GPU is used automatically when available (embedding extraction and deep models).
