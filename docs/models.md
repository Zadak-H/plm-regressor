# Models & pLMs

Run `plm-regressor list` to see exactly what is installed/available in your environment.

## Regressors

`kind` = classical (sklearn / xgboost / lightgbm) or torch (deep). `max rows` is the dataset-size
ceiling the size engine uses to auto-skip models that would be too slow.

| model | kind | max rows | notes |
|-------|------|----------|-------|
| `ridge` | classical | unlimited | |
| `sgd` | classical | unlimited | scalable linear |
| `elasticnet` | classical | unlimited | |
| `huber` | classical | ≤ 200,000 | robust linear |
| `bayesian_ridge` | classical | ≤ 200,000 | native std |
| `svr_rbf` | classical | ≤ 20,000 | |
| `kernel_ridge` | classical | ≤ 20,000 | |
| `gpr` | classical | ≤ 2,000 | native std |
| `knn` | classical | ≤ 50,000 | |
| `pls` | classical | unlimited | |
| `rf` | classical | ≤ 200,000 | |
| `extra_trees` | classical | ≤ 200,000 | |
| `hist_gb` | classical | unlimited | |
| `xgboost` | classical | unlimited | optional |
| `lightgbm` | classical | unlimited | optional |
| `mlp` | classical | ≤ 50,000 | sklearn MLP |
| `mlp_torch` | torch | unlimited | FNN over embeddings (+tabular) |
| `cnn1d` | torch | unlimited | 1D-CNN; one-hot/blosum only |

## Protein language models

| name | backend | dim | model id | needs |
|------|---------|-----|----------|-------|
| `esm2_8m` | esm | 320 | esm2_t6_8M_UR50D | fair-esm |
| `esm2_35m` | esm | 480 | esm2_t12_35M_UR50D | fair-esm |
| `esm2_150m` | esm | 640 | esm2_t30_150M_UR50D | fair-esm |
| `esm2` | esm | 1280 | esm2_t33_650M_UR50D | fair-esm |
| `esm2_3b` | esm | 2560 | esm2_t36_3B_UR50D | fair-esm |
| `esm2_15b` | esm | 5120 | esm2_t48_15B_UR50D | fair-esm |
| `esm1` | esm | 1280 | esm1_t34_670M_UR50S | fair-esm |
| `esm1b` | esm | 1280 | esm1b_t33_650M_UR50S | fair-esm |
| `esm1v` | esm | 1280 | esm1v_t33_650M_UR90S_1 | fair-esm |
| `esmc_300m` | esmc | 960 | esmc_300m | esm (EvolutionaryScale SDK) |
| `esmc_600m` | esmc | 1152 | esmc_600m | esm (EvolutionaryScale SDK) |
| `esmplusplus_small` | hf_auto | 960 | Synthyra/ESMplusplus_small | transformers |
| `esmplusplus_large` | hf_auto | 1152 | Synthyra/ESMplusplus_large | transformers |
| `protT5` | t5 | 1024 | Rostlab/prot_t5_xl_uniref50 | transformers + sentencepiece |
| `protT5_half` | t5 | 1024 | Rostlab/prot_t5_xl_half_uniref50-enc | transformers + sentencepiece |
| `protT5_bfd` | t5 | 1024 | Rostlab/prot_t5_xl_bfd | transformers + sentencepiece |
| `protT5_xxl` | t5 | 1024 | Rostlab/prot_t5_xxl_uniref50 | transformers + sentencepiece |
| `prostT5` | t5 | 1024 | Rostlab/ProstT5 | transformers + sentencepiece |
| `protbert` | bert | 1024 | Rostlab/prot_bert | transformers |
| `protbert_bfd` | bert | 1024 | Rostlab/prot_bert_bfd | transformers |
| `ankh_base` | ankh | 768 | ElnaggarLab/ankh-base | transformers |
| `ankh_large` | ankh | 1536 | ElnaggarLab/ankh-large | transformers |
| `prosst` | prosst | 768 | AI4Protein/ProSST-2048 | transformers |
| `carp_640m` | carp | 1280 | carp_640M | sequence-models |

!!! note "ESM C"
    `esmc_*` use EvolutionaryScale's `esm` SDK, which collides with `fair-esm` (both import as
    `esm`). To use ESM C today without a separate environment, pick **`esmplusplus_small`** or
    **`esmplusplus_large`** — the same architecture/weights via `transformers`.

## Simple encodings + tabular

- `onehot`, `blosum62` — positional encodings (require aligned, equal-length sequences). The only
  inputs `cnn1d` accepts.
- `tabular` — your extra numeric/categorical columns, combined with any other source.
