# GUI usage

The Streamlit GUI is the no-code path: upload data, tick what you want, click **Run**, read the
report, rank candidates.

```bash
plm-regressor gui          # or: streamlit run app.py
```

It opens at `http://localhost:8501`. The app has seven tabs, left to right.

```
┌ PLM-Regressor ────────────────────────────────────────────────────────────┐
│ 1.Data │ 2.Features │ 3.Models │ 4.Search │ 5.Run │ 6.Results │ 7.Predict │
└───────────────────────────────────────────────────────────────────────────┘
```

## 1 · Data

- Upload your training **CSV**.
- Pick the **sequence column** and the **target column** (your property: activity, Kd, Tm, pH, …).
- Optional: an **ID column**, a **group column** (for leakage-safe CV), and **extra inputs** —
  numeric columns (pH, temperature, …) and/or categorical columns (buffer, assay, …).
- Choose a **replicate policy** (default: average repeated measurements per sequence).

## 2 · Features

- Multiselect **feature sources**: protein LM embeddings, `onehot`/`blosum62`, and `tabular`
  (your extra columns). Mix freely — e.g. `esm2` + `tabular`.
- Set the **embedding directory** (where `*.npz` banks live).
- For any selected pLM whose bank is missing, click **Extract** — embeddings are computed and
  cached (first run downloads the model). ESM C → use `esmplusplus_small/large`.

## 3 · Models

- Multiselect **regressors** (classical + deep `mlp_torch`/`cnn1d`). Models too costly for the
  dataset size are auto-skipped — you don't have to think about it.

## 4 · Search

- **Primary metric**: `spearman | pearson | kendall | r2 | ndcg | rmse | mse | mae`.
- **Auto-tune by dataset size** (recommended): CV strategy, trial budget, and model gating are
  chosen for you. Or override the trial budget manually.
- Toggle **uncertainty** (conformal + ensemble intervals) and ensemble size.

## 5 · Run

- Set the **output directory** and click **🚀 Run training**.
- A live log streams the Optuna search. When it finishes, go to **Results**.

## 6 · Results

- Best model, chosen features, dataset tier, and the **OOF metric table**.
- **Plots**: measured-vs-predicted scatter, residuals, per-model comparison bar, uncertainty
  calibration.
- **Download** `best_model.joblib`, `oof_predictions.csv`, and the candidate ranking.

## 7 · Predict

- Point at a saved run, upload a **candidate CSV**, set the candidate sequence column (and
  embedding directory if needed), and **Rank candidates**.
- Get a ranked table + `top_{10,50,100}.csv` to download.

!!! tip "Everything the GUI does is reproducible"
    Each run writes a `run_config.yaml`. Re-run it headless with
    `plm-regressor train run_config.yaml` — handy for clusters or scripting.
