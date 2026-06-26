#!/usr/bin/env python3
"""PLM-Regressor — Streamlit GUI.

A point-and-click front end over the config-driven core: upload a CSV, pick the
target + (optional) extra columns, choose pLMs and regressors, run a size-aware
Optuna search, view the report, and rank candidate sequences. Launch with::

    plm_regressor gui
    # or: streamlit run app.py
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

from plm_regressor.config import RunConfig
from plm_regressor.core import COMPUTED_FEATURE_SOURCES, TABULAR_FEATURE_SOURCE
from plm_regressor.registry import DEFAULT_MODELS, available_models, available_plms

st.set_page_config(page_title="PLM-Regressor", layout="wide")

_PLM_DESCRIPTIONS = {
    "esm2_8m":         "ESM-2 8M  — fastest, lowest memory (320-dim). Good for quick tests.",
    "esm2_35m":        "ESM-2 35M — fast, low memory (480-dim). Recommended starting point.",
    "esm2_150m":       "ESM-2 150M — good balance of speed and quality (640-dim).",
    "esm2":            "ESM-2 650M — strong general-purpose model (1280-dim). ⭐ Most popular.",
    "esm2_3b":         "ESM-2 3B  — high quality, needs ~12 GB VRAM (2560-dim).",
    "esm2_15b":        "ESM-2 15B — highest ESM2 quality, needs ~60 GB VRAM (5120-dim).",
    "esm1":            "ESM-1 670M — original ESM (1280-dim).",
    "esm1b":           "ESM-1b 650M — improved ESM-1 (1280-dim).",
    "esm1v":           "ESM-1v — trained for variant effect prediction (1280-dim).",
    "esmc_300m":       "ESM-C 300M — latest EvolutionaryScale model (960-dim). Needs `esm` SDK.",
    "esmc_600m":       "ESM-C 600M — larger Cambrian model (1152-dim). Needs `esm` SDK.",
    "esmplusplus_small": "ESM++ small — ESM-C reimplemented in HF Transformers (960-dim). No SDK clash.",
    "esmplusplus_large": "ESM++ large — larger ESM-C HF variant (1152-dim). No SDK clash.",
    "protT5":          "ProtT5 XL UniRef50 — T5 encoder, strong across benchmarks (1024-dim).",
    "protT5_half":     "ProtT5 XL half-precision encoder — faster ProtT5 (1024-dim).",
    "protT5_bfd":      "ProtT5 XL BFD — ProtT5 trained on BFD database (1024-dim).",
    "protT5_xxl":      "ProtT5 XXL — largest T5 encoder, highest quality (1024-dim). Slow.",
    "prostT5":         "ProstT5 — structure-sequence T5 model (1024-dim).",
    "protbert":        "ProtBert UniRef100 — BERT-based protein model (1024-dim).",
    "protbert_bfd":    "ProtBert BFD — ProtBert trained on BFD (1024-dim).",
    "ankh_base":       "Ankh Base — compact efficient protein LM (768-dim).",
    "ankh_large":      "Ankh Large — larger Ankh model (1536-dim).",
    "carp_640m":       "CARP 640M — Microsoft sequence-models protein LM (1280-dim).",
}
_PLM_NO_AUTO = {"prosst"}


def _idx(options, value, default=0):
    try:
        return options.index(value)
    except (ValueError, AttributeError):
        return default


WORKSPACE = Path("plm_regressor_workspace")
WORKSPACE.mkdir(exist_ok=True)
ss = st.session_state
ss.setdefault("train_csv", None)
ss.setdefault("columns", [])

st.title("🧬 PLM-Regressor — sequence → property regression")
st.caption("Upload data, pick features + models, train, and rank candidates. No coding required.")

tabs = st.tabs(["0. Embed", "1. Data", "2. Features", "3. Models", "4. Search", "5. Run", "6. Results", "7. Predict"])

# --------------------------------------------------------------------------- #
# 0. Embed  — standalone embedding generation (no training required)
# --------------------------------------------------------------------------- #
with tabs[0]:
    st.header("Generate protein embeddings")
    st.caption(
        "Upload any CSV with protein sequences, pick a language model, and download the "
        "embedding file (.npz). No training needed — use this output as a pre-built bank "
        "for the Features tab, or for any other analysis."
    )

    from plm_regressor.registry import PLM_REGISTRY as _PLM_REG

    embed_up = st.file_uploader("Upload a CSV with protein sequences", type=["csv"], key="embed_up")
    embed_csv_path = None
    embed_cols = []
    if embed_up is not None:
        embed_csv_path = WORKSPACE / "embed_input.csv"
        embed_csv_path.write_bytes(embed_up.getbuffer())
        _edf = pd.read_csv(embed_csv_path)
        embed_cols = list(_edf.columns)
        st.dataframe(_edf.head(10), use_container_width=True)
        st.caption(f"{len(_edf)} sequences in file.")

    runnable_plms = [n for n in _PLM_REG if n not in _PLM_NO_AUTO]
    plm_labels = {n: f"{n}  —  {_PLM_DESCRIPTIONS.get(n, '')}" for n in runnable_plms}
    default_plm = "esm2" if "esm2" in runnable_plms else runnable_plms[0]

    ec1, ec2 = st.columns(2)
    embed_seq_col = ec1.selectbox(
        "Sequence column", embed_cols if embed_cols else ["(upload a CSV first)"], key="embed_seq_col"
    )
    embed_plm = ec2.selectbox(
        "Protein language model",
        runnable_plms,
        index=runnable_plms.index(default_plm),
        format_func=lambda n: plm_labels[n],
        key="embed_plm_select",
    )
    with st.expander("Advanced options"):
        embed_batch = st.number_input("Batch size", 1, 64, 8, key="embed_batch")
        embed_cpu = st.checkbox("Force CPU (slower but works without a GPU)", value=False, key="embed_cpu")

    embed_out_npz = WORKSPACE / f"{embed_plm}.npz"
    if st.button("Generate embeddings", type="primary", key="embed_run"):
        if embed_csv_path is None:
            st.error("Upload a CSV file first.")
        elif not embed_cols or embed_seq_col not in embed_cols:
            st.error("Select a valid sequence column.")
        else:
            from plm_regressor.embeddings.extract import extract_from_csv

            with st.spinner(f"Running {embed_plm} — first run downloads the model weights (~minutes)…"):
                try:
                    out_path, n_new, n_cached = extract_from_csv(
                        embed_plm, str(embed_csv_path), embed_seq_col,
                        str(embed_out_npz), batch_size=embed_batch, force_cpu=embed_cpu,
                    )
                    st.success(f"Done! {n_new} embeddings computed, {n_cached} reused from cache.")
                    ss["embed_ready_npz"] = str(out_path)
                except Exception as exc:
                    st.error(f"Extraction failed: {exc}")

    if ss.get("embed_ready_npz") and Path(ss["embed_ready_npz"]).exists():
        npz_bytes = Path(ss["embed_ready_npz"]).read_bytes()
        st.download_button(
            f"Download {embed_plm}.npz",
            npz_bytes,
            file_name=f"{embed_plm}.npz",
            mime="application/octet-stream",
        )
        st.info(
            f"To use in training: put `{embed_plm}.npz` in your embedding directory "
            f"(default: `embeddings/`) and select `{embed_plm}` as a feature source in Tab 2."
        )

# --------------------------------------------------------------------------- #
# 1. Data
# --------------------------------------------------------------------------- #
with tabs[1]:
    st.header("Training data")
    up = st.file_uploader("Upload a training CSV", type=["csv"])
    if up is not None:
        path = WORKSPACE / "train.csv"
        path.write_bytes(up.getbuffer())
        ss["train_csv"] = str(path)
    if ss.get("train_csv"):
        df = pd.read_csv(ss["train_csv"])
        ss["columns"] = list(df.columns)
        st.dataframe(df.head(20), use_container_width=True)
        cols = ss["columns"]
        numeric_cols = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
        c1, c2, c3 = st.columns(3)
        ss["seq_col"] = c1.selectbox("Sequence column", cols, index=_idx(cols, ss.get("seq_col", "Protein_Seq")))
        ss["target_col"] = c2.selectbox("Target column", cols, index=_idx(cols, ss.get("target_col", cols[-1])))
        ss["id_col"] = c3.selectbox("ID column (optional)", ["<none>"] + cols, index=0)
        c4, c5, c6 = st.columns(3)
        ss["group_col"] = c4.selectbox("Group column (optional, leakage-safe CV)", ["<none>"] + cols, index=0)
        ss["extra_feature_cols"] = c5.multiselect("Extra numeric inputs (pH, temp, …)", numeric_cols)
        ss["categorical_cols"] = c6.multiselect("Extra categorical inputs", [c for c in cols])
        ss["replicate_policy"] = st.selectbox(
            "Replicate policy", ["mean_by_sequence", "median_by_sequence", "keep_rows"], index=0
        )
    else:
        st.info("Upload a CSV to begin.")

# --------------------------------------------------------------------------- #
# 2. Features
# --------------------------------------------------------------------------- #
with tabs[2]:
    st.header("Feature sources")
    plm_choices = available_plms()
    encodings = sorted(COMPUTED_FEATURE_SOURCES)
    has_extra = bool(ss.get("extra_feature_cols") or ss.get("categorical_cols"))
    extra_opt = [TABULAR_FEATURE_SOURCE] if has_extra else []
    default_feats = [f for f in ["esm2"] if f in plm_choices] or plm_choices[:1]
    ss["feature_sources"] = st.multiselect(
        "Choose feature sources (pLMs / encodings / tabular)",
        plm_choices + encodings + extra_opt,
        default=ss.get("feature_sources", default_feats),
    )
    ss["embedding_dir"] = st.text_input("Embedding directory (npz banks)", ss.get("embedding_dir", "embeddings"))

    st.subheader("Embedding extraction")
    emb_dir = Path(ss.get("embedding_dir", "embeddings"))
    learned = [f for f in ss["feature_sources"] if f not in COMPUTED_FEATURE_SOURCES and f != TABULAR_FEATURE_SOURCE]
    for f in learned:
        present = (emb_dir / f"{f}.npz").exists()
        cols = st.columns([3, 1])
        cols[0].write(f"`{f}.npz` — {'✅ found' if present else '⚠️ missing'}")
        if not present and ss.get("train_csv") and ss.get("seq_col"):
            if cols[1].button(f"Extract {f}", key=f"extract_{f}"):
                from plm_regressor.embeddings.extract import extract_from_csv

                with st.spinner(f"Extracting {f} (first run downloads the model)…"):
                    try:
                        _, n_new, n_cached = extract_from_csv(
                            f, ss["train_csv"], ss["seq_col"], str(emb_dir / f"{f}.npz")
                        )
                        st.success(f"{f}: {n_new} computed, {n_cached} cached")
                    except Exception as exc:
                        st.error(f"Extraction failed: {exc}")
    st.caption("onehot / blosum62 need aligned, equal-length sequences; tabular uses your extra columns.")

# --------------------------------------------------------------------------- #
# 3. Models
# --------------------------------------------------------------------------- #
with tabs[3]:
    st.header("Regressors")
    models = available_models()
    ss["models"] = st.multiselect(
        "Choose regressors (classical + deep). Models too costly for the dataset size are auto-skipped.",
        models, default=[m for m in (ss.get("models") or DEFAULT_MODELS) if m in models],
    )
    st.caption("Deep models: `mlp_torch` (FNN on embeddings), `cnn1d` (1D-CNN on onehot/blosum).")

# --------------------------------------------------------------------------- #
# 4. Search
# --------------------------------------------------------------------------- #
with tabs[4]:
    st.header("Search settings")
    c1, c2, c3 = st.columns(3)
    ss["metric"] = c1.selectbox("Primary metric",
                                ["spearman", "pearson", "kendall", "r2", "ndcg", "rmse", "mse", "mae"],
                                index=0)
    ss["auto_size"] = c2.checkbox("Auto-tune by dataset size", value=ss.get("auto_size", True))
    ss["standard_search"] = c3.checkbox("Standard (single sources only)", value=ss.get("standard_search", False))
    c4, c5, c6 = st.columns(3)
    override = c4.checkbox("Override trial budget", value=False)
    ss["n_trials"] = c4.number_input("Trials", 5, 1000, ss.get("n_trials") or 50) if override else None
    ss["top_ensemble"] = c5.number_input("Ensemble size (uncertainty)", 1, 20, ss.get("top_ensemble", 5))
    ss["no_uncertainty"] = c6.checkbox("Disable uncertainty", value=ss.get("no_uncertainty", False))
    ss["use_gpu"] = st.checkbox("Use GPU for boosting (xgb/lgb)", value=ss.get("use_gpu", False))

# --------------------------------------------------------------------------- #
# 5. Run
# --------------------------------------------------------------------------- #
with tabs[5]:
    st.header("Run training")
    ss["out_dir"] = st.text_input("Output run directory", ss.get("out_dir", "runs/gui_run"))
    if st.button("🚀 Run training", type="primary"):
        if not ss.get("train_csv"):
            st.error("Upload a CSV first (tab 1).")
        else:
            cfg = RunConfig(
                csv=ss["train_csv"], seq_col=ss["seq_col"], target_col=ss["target_col"],
                id_col=None if ss.get("id_col", "<none>") == "<none>" else ss["id_col"],
                group_col=None if ss.get("group_col", "<none>") == "<none>" else ss["group_col"],
                extra_feature_cols=ss.get("extra_feature_cols", []),
                categorical_cols=ss.get("categorical_cols", []),
                replicate_policy=ss.get("replicate_policy", "mean_by_sequence"),
                feature_sources=ss["feature_sources"], embedding_dir=ss.get("embedding_dir", "embeddings"),
                models=ss["models"], metric=ss["metric"], auto_size=ss["auto_size"],
                standard_search=ss["standard_search"], n_trials=ss.get("n_trials"),
                top_ensemble=int(ss["top_ensemble"]), no_uncertainty=ss["no_uncertainty"],
                use_gpu=ss["use_gpu"], out_dir=ss["out_dir"],
            )
            try:
                cfg.validate()
            except Exception as exc:
                st.error(f"Invalid config: {exc}")
                st.stop()
            cfg_path = WORKSPACE / "run_config.yaml"
            cfg.to_yaml(cfg_path)
            log_path = WORKSPACE / "train.log"
            st.info("Training started. Live log below.")
            log_box = st.empty()
            with open(log_path, "w") as logf:
                proc = subprocess.Popen(
                    [sys.executable, "-m", "plm_regressor.cli", "train", str(cfg_path)],
                    stdout=logf, stderr=subprocess.STDOUT,
                )
                import time

                while proc.poll() is None:
                    log_box.code(Path(log_path).read_text()[-4000:])
                    time.sleep(1.5)
                log_box.code(Path(log_path).read_text()[-4000:])
            if proc.returncode == 0:
                st.success(f"Done. Open tab 6 (Results) and point it at {ss['out_dir']}.")
            else:
                st.error("Training failed; see log above.")

# --------------------------------------------------------------------------- #
# 6. Results
# --------------------------------------------------------------------------- #
with tabs[6]:
    st.header("Results")
    run_dir = Path(st.text_input("Run directory", ss.get("out_dir", "runs/gui_run")))
    if (run_dir / "run_report.json").exists():
        report = json.loads((run_dir / "run_report.json").read_text())
        st.subheader(f"{report.get('run_name')} — best model: {report.get('best_model_name')}")
        st.write(f"Features: **{'+'.join(report.get('best_feature_subset') or [])}** "
                 f"({report.get('best_feature_mode')}) · tier: {report.get('size_tier')} · CV: {report.get('cv_strategy')}")
        st.table(pd.DataFrame([report.get("oof_metrics", {})]).T.rename(columns={0: "value"}))
        for img in report.get("plots", []):
            p = run_dir / "plots" / Path(img).name
            if p.exists():
                st.image(str(p))
        for fname in ["best_model.joblib", "oof_predictions.csv", "candidate_predictions.csv"]:
            fp = run_dir / fname
            if fp.exists():
                st.download_button(f"Download {fname}", fp.read_bytes(), file_name=fname)
    else:
        st.info("No run_report.json found in that directory yet.")

# --------------------------------------------------------------------------- #
# 7. Predict
# --------------------------------------------------------------------------- #
with tabs[7]:
    st.header("Rank candidate sequences")
    pred_run_dir = st.text_input("Saved run directory", ss.get("out_dir", "runs/gui_run"), key="pred_run")
    cand = st.file_uploader("Upload candidate CSV", type=["csv"], key="cand")
    pseq = st.text_input("Candidate sequence column", "Sequence")
    cand_emb = st.text_input("Candidate embedding directory (optional)", "")
    top_n = st.number_input("Top N", 1, 100000, 100)
    if st.button("Rank candidates"):
        if cand is None:
            st.error("Upload a candidate CSV first.")
        else:
            cand_path = WORKSPACE / "candidates.csv"
            cand_path.write_bytes(cand.getbuffer())
            from plm_regressor.predict import score_candidates_from_run

            with st.spinner("Scoring candidates…"):
                try:
                    out = score_candidates_from_run(
                        run_dir=pred_run_dir, candidate_csv=str(cand_path), predict_seq_col=pseq,
                        candidate_embedding_dir=cand_emb or None, top_n=int(top_n),
                    )
                    ranked = pd.read_csv(out)
                    st.success(f"Ranked {len(ranked)} candidates.")
                    st.dataframe(ranked.head(50), use_container_width=True)
                    st.download_button("Download candidate_predictions.csv", out.read_bytes(),
                                       file_name="candidate_predictions.csv")
                except Exception as exc:
                    st.error(f"Ranking failed: {exc}")
