# GUI usage

The Streamlit GUI is the no-code path: upload data, pick a language model, train, and rank
candidates — no Python knowledge required.

```bash
plm-regressor gui          # or: streamlit run app.py
```

Opens at `http://localhost:8501`. Eight tabs, left to right:

```
┌ PLM-Regressor ──────────────────────────────────────────────────────────────────────┐
│ Embed │ Data │ Features │ Models │ Search │ Run │ Results │ Predict │
└─────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 0 · Embed — generate embeddings without training

!!! tip "Start here if you just need embedding files"
    This tab works completely standalone. No training data, no labels needed.
    Upload any CSV of sequences → pick a pLM → click **Generate embeddings** → download the `.npz`.

- Upload a **CSV** with a column of protein sequences (any column name).
- Pick the **sequence column**.
- Choose a **protein language model** from the dropdown — every model shows a plain-English
  description with recommended size and speed information. Default is **ESM-2 650M** (best
  general-purpose starting point).
- *Advanced options* (collapsed by default): batch size and CPU-only mode.
- Click **Generate embeddings**. The first run downloads the model weights (~minutes). Progress
  is shown in the spinner.
- When done, click **Download `<plm>.npz`** to save the embedding file.

!!! note "Using the file for training"
    Put the downloaded `<plm>.npz` in your embedding directory (default `embeddings/`) and
    select that pLM as a feature source in the **Features** tab.

### Which model to pick?

| Model | Speed | Memory | When to use |
|-------|-------|---------|-------------|
| `esm2_8m` | ⚡⚡⚡ | ~200 MB | Quick prototyping, CPU only |
| `esm2_35m` | ⚡⚡⚡ | ~300 MB | Good starting point on CPU |
| `esm2_150m` | ⚡⚡ | ~600 MB | Balanced quality/speed |
| `esm2` (650M) | ⚡⚡ | ~2.5 GB | **Recommended default** — strong across benchmarks |
| `esmc_300m` | ⚡⚡ | ~1.5 GB | Latest ESM-C; auto-detected and runs transparently |
| `esmc_600m` | ⚡ | ~3 GB | Larger ESM-C; highest ESM quality without 3B+ size |
| `protT5` | ⚡ | ~5 GB | Strong alternative to ESM2; T5-based |
| `ankh_base` | ⚡⚡ | ~400 MB | Compact, efficient |

### ESM-C (esmc_300m / esmc_600m)

ESM-C is EvolutionaryScale's newest architecture (better than ESM-2 on many benchmarks).
The SDK conflicts with `fair-esm` and requires Python ≥ 3.10, but **you don't have to manage
any of this yourself**:

!!! success "ESM-C just works"
    The tool automatically finds a conda environment on your machine that has the ESM SDK
    installed. If one exists, ESM-C extraction runs transparently — you'll see
    `[ESM-C] using: /path/to/env/python3` in the log.

To set it up once:

```bash
conda create -n esmc python=3.10 -y
conda activate esmc
pip install esm httpx
```

After that, any time you pick `esmc_300m` or `esmc_600m` in either the GUI or CLI,
the tool finds and uses that environment automatically. No manual activation needed.

---

## 1 · Data

- Upload your training **CSV**.
- Pick the **sequence column** and the **target column** (your property: activity, Kd, Tm, pH, …).
- Optional: an **ID column**, a **group column** (for leakage-safe CV), and **extra inputs** —
  numeric columns (pH, temperature, …) and/or categorical columns (buffer, assay, …).
- Choose a **replicate policy** (default: average repeated measurements per sequence).

---

## 2 · Features

- Multiselect **feature sources**: protein LM embeddings, `onehot`/`blosum62`, and `tabular`
  (your extra columns). Mix freely — e.g. `esm2` + `tabular`.
- Set the **embedding directory** (where `*.npz` banks live).
- For any selected pLM whose bank is missing, click **Extract** — embeddings are computed and
  cached (first run downloads the model).

!!! tip "Already generated embeddings in the Embed tab?"
    Place the downloaded `.npz` in the embedding directory, then select the matching pLM here.

---

## 3 · Models

- Multiselect **regressors** (classical + deep `mlp_torch`/`cnn1d`).
- Models too costly for the dataset size are auto-skipped — you don't have to think about it.

---

## 4 · Search

- **Primary metric**: `spearman | pearson | kendall | r2 | ndcg | rmse | mse | mae`.
- **Auto-tune by dataset size** (recommended): CV strategy, trial budget, and model gating are
  chosen automatically.
- Or override the trial budget manually.
- Toggle **uncertainty** (conformal + ensemble intervals) and ensemble size.

---

## 5 · Run

- Set the **output directory** and click **🚀 Run training**.
- A live log streams the Optuna search. When it finishes, go to **Results**.

---

## 6 · Results

- Best model, chosen features, dataset tier, and the **OOF metric table**.
- **Plots**: measured-vs-predicted scatter, residuals, per-model comparison bar, uncertainty
  calibration.
- **Download** `best_model.joblib`, `oof_predictions.csv`, and the candidate ranking.

---

## 7 · Predict

- Point at a saved **run directory**.
- Upload a **candidate CSV** — the sequence column is auto-detected from the file.
- The **embedding directory** defaults to the one used during training.
- Set **Top N** and click **Rank candidates**.

!!! warning "Missing embeddings"
    If candidates don't have embeddings yet, the tool shows a clear warning and the predictions
    will be blank. Use the **Embed tab (Tab 0)** to generate embeddings for your candidates
    first, then place the `.npz` in the candidate embedding directory.

!!! tip "Everything the GUI does is reproducible"
    Each run writes a `run_config.yaml`. Re-run it headless with
    `plm-regressor train run_config.yaml` — handy for clusters or scripting.
