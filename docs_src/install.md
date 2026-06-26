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

## ESM-C setup (one time)

ESM-C (`esmc_300m`, `esmc_600m`) uses EvolutionaryScale's `esm` SDK which conflicts with
`fair-esm` and requires Python ≥ 3.10. **You do not need to switch environments manually** —
the tool auto-discovers any conda env that has the SDK and uses it via subprocess.

Set it up once:

```bash
conda create -n esmc python=3.10 -y
conda activate esmc
pip install esm httpx
```

After that, select `esmc_300m` or `esmc_600m` anywhere (GUI or CLI) and it works automatically.
You will see `[ESM-C] using: /path/to/esmc/bin/python3` in the log.

## Notes on other optional backends

- **LightGBM** and **CARP** are optional; if not installed they simply don't appear in
  `plm-regressor list`.
- A CUDA GPU is used automatically when available (embedding extraction and deep models).
