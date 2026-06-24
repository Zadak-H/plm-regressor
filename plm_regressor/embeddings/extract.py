#!/usr/bin/env python3
"""Unified, cached protein-LM embedding extraction.

One entry point (:func:`extract_to_npz`) that:
- dedupes sequences
- reuses any already-cached vectors in the target npz (so re-runs are cheap)
- computes only the missing sequences with the right backend
- mean-pools per-residue states to one vector per sequence
- writes the standard ``{embeddings, sequences, seq_to_index}`` npz schema

Backends: ``esm`` (fair-esm), ``t5`` / ``ankh`` / ``bert`` (HF transformers).
ProSST extraction needs structure tokens and is intentionally not auto-run here
(use the precomputed bank or ``scripts/embeds_scripts/extract_prosst_embeddings.py``).
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ..registry import PLM_REGISTRY, PLMSpec


def _device(force_cpu: bool = False) -> str:
    if force_cpu:
        return "cpu"
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


# --------------------------------------------------------------------------- #
# Backends: each returns an (N, D) float32 array for the given sequence list
# --------------------------------------------------------------------------- #


def _embed_esm(spec: PLMSpec, sequences: List[str], batch_size: int, device: str) -> np.ndarray:
    import esm
    import torch

    load_fn = getattr(esm.pretrained, spec.model_id)
    model, alphabet = load_fn()
    model = model.eval().to(device)
    layer = model.num_layers
    batch_converter = alphabet.get_batch_converter()
    out: List[np.ndarray] = []
    with torch.no_grad():
        for i in range(0, len(sequences), batch_size):
            batch = sequences[i : i + batch_size]
            data = [(f"seq_{j}", s) for j, s in enumerate(batch)]
            _, _, tokens = batch_converter(data)
            tokens = tokens.to(device)
            reps = model(tokens, repr_layers=[layer], return_contacts=False)["representations"][layer]
            for j, seq in enumerate(batch):
                out.append(reps[j, 1 : 1 + len(seq)].mean(0).cpu().numpy())
    return np.asarray(out, dtype=np.float32)


def _embed_t5(spec: PLMSpec, sequences: List[str], batch_size: int, device: str) -> np.ndarray:
    import torch
    from transformers import T5EncoderModel, T5Tokenizer

    tokenizer = T5Tokenizer.from_pretrained(spec.model_id, do_lower_case=False)
    model = T5EncoderModel.from_pretrained(spec.model_id).eval().to(device)
    prefix = "<AA2fold> " if "prostt5" in spec.model_id.lower() else ""
    out: List[np.ndarray] = []
    with torch.no_grad():
        for i in range(0, len(sequences), batch_size):
            batch = sequences[i : i + batch_size]
            spaced = [prefix + " ".join(re.sub(r"[UZOB]", "X", s)) for s in batch]
            enc = tokenizer(spaced, add_special_tokens=True, padding="longest", return_tensors="pt")
            ids = enc["input_ids"].to(device)
            mask = enc["attention_mask"].to(device)
            reps = model(input_ids=ids, attention_mask=mask).last_hidden_state
            for j, seq in enumerate(batch):
                length = int(mask[j].sum().item())
                # drop trailing EOS/special token; pooling over real residues
                vec = reps[j, : max(1, length - 1)].mean(0)
                out.append(vec.cpu().numpy())
    return np.asarray(out, dtype=np.float32)


def _embed_bert(spec: PLMSpec, sequences: List[str], batch_size: int, device: str) -> np.ndarray:
    import torch
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(spec.model_id, do_lower_case=False)
    model = AutoModel.from_pretrained(spec.model_id).eval().to(device)
    out: List[np.ndarray] = []
    with torch.no_grad():
        for i in range(0, len(sequences), batch_size):
            batch = sequences[i : i + batch_size]
            spaced = [" ".join(re.sub(r"[UZOB]", "X", s)) for s in batch]
            enc = tokenizer(spaced, add_special_tokens=True, padding="longest", return_tensors="pt")
            ids = enc["input_ids"].to(device)
            mask = enc["attention_mask"].to(device)
            reps = model(input_ids=ids, attention_mask=mask).last_hidden_state
            for j in range(len(batch)):
                length = int(mask[j].sum().item())
                vec = reps[j, 1 : max(2, length - 1)].mean(0)  # skip CLS and trailing SEP
                out.append(vec.cpu().numpy())
    return np.asarray(out, dtype=np.float32)


def _embed_ankh(spec: PLMSpec, sequences: List[str], batch_size: int, device: str) -> np.ndarray:
    import torch
    from transformers import AutoTokenizer, T5EncoderModel

    tokenizer = AutoTokenizer.from_pretrained(spec.model_id)
    model = T5EncoderModel.from_pretrained(spec.model_id).eval().to(device)
    out: List[np.ndarray] = []
    with torch.no_grad():
        for i in range(0, len(sequences), batch_size):
            batch = [list(s) for s in sequences[i : i + batch_size]]
            enc = tokenizer.batch_encode_plus(
                batch, add_special_tokens=True, padding=True, is_split_into_words=True, return_tensors="pt"
            )
            ids = enc["input_ids"].to(device)
            mask = enc["attention_mask"].to(device)
            reps = model(input_ids=ids, attention_mask=mask).last_hidden_state
            for j in range(len(batch)):
                length = int(mask[j].sum().item())
                out.append(reps[j, : max(1, length - 1)].mean(0).cpu().numpy())
    return np.asarray(out, dtype=np.float32)


def _embed_esmc(spec: PLMSpec, sequences: List[str], batch_size: int, device: str) -> np.ndarray:
    """ESM C / Cambrian via the EvolutionaryScale `esm` SDK (per-sequence)."""
    import torch
    from esm.models.esmc import ESMC
    from esm.sdk.api import ESMProtein, LogitsConfig

    client = ESMC.from_pretrained(spec.model_id).to(device).eval()
    cfg = LogitsConfig(sequence=True, return_embeddings=True)
    out: List[np.ndarray] = []
    with torch.no_grad():
        for seq in sequences:
            tensor = client.encode(ESMProtein(sequence=seq))
            emb = client.logits(tensor, cfg).embeddings  # (1, L+special, D)
            out.append(emb[0, 1:-1].mean(0).float().cpu().numpy())  # drop BOS/EOS, mean-pool
    return np.asarray(out, dtype=np.float32)


def _embed_hf_auto(spec: PLMSpec, sequences: List[str], batch_size: int, device: str) -> np.ndarray:
    """Generic HF model (e.g. ESM++ / Synthyra). Attention-masked mean pooling.

    Robust to models (like ESM++) that bundle their tokenizer as ``model.tokenizer``
    and aren't resolvable via ``AutoTokenizer.from_pretrained``.
    """
    import torch
    from transformers import AutoModel, AutoModelForMaskedLM, AutoTokenizer

    try:
        model = AutoModel.from_pretrained(spec.model_id, trust_remote_code=True)
    except Exception:
        model = AutoModelForMaskedLM.from_pretrained(spec.model_id, trust_remote_code=True)
    tokenizer = getattr(model, "tokenizer", None)
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(spec.model_id, trust_remote_code=True)
    model = model.eval().to(device)

    out: List[np.ndarray] = []
    with torch.no_grad():
        for i in range(0, len(sequences), batch_size):
            batch = [re.sub(r"[UZOB]", "X", s) for s in sequences[i : i + batch_size]]
            enc = tokenizer(batch, add_special_tokens=True, padding=True, return_tensors="pt")
            enc = {k: v.to(device) for k, v in enc.items() if hasattr(v, "to")}
            res = model(**enc, output_hidden_states=True)
            reps = getattr(res, "last_hidden_state", None)
            if reps is None:
                reps = res.hidden_states[-1]
            mask = enc["attention_mask"].unsqueeze(-1).float()  # (B, T, 1)
            pooled = (reps * mask).sum(1) / mask.sum(1).clamp(min=1.0)
            out.extend(pooled.float().cpu().numpy())
    return np.asarray(out, dtype=np.float32)


def _embed_carp(spec: PLMSpec, sequences: List[str], batch_size: int, device: str) -> np.ndarray:
    """CARP via Microsoft sequence-models."""
    import torch
    from sequence_models.pretrained import load_model_and_alphabet

    model, collater = load_model_and_alphabet(spec.model_id)
    model = model.eval().to(device)
    out: List[np.ndarray] = []
    with torch.no_grad():
        for i in range(0, len(sequences), batch_size):
            batch = [[s] for s in sequences[i : i + batch_size]]
            x, = collater(batch)[:1]
            x = x.to(device)
            reps = model(x, repr_layers=[model.n_layers] if hasattr(model, "n_layers") else None)["representations"]
            rep = reps[max(reps)] if isinstance(reps, dict) else reps
            for j, s in enumerate(sequences[i : i + batch_size]):
                out.append(rep[j, : len(s)].mean(0).float().cpu().numpy())
    return np.asarray(out, dtype=np.float32)


_BACKENDS: Dict[str, Callable[..., np.ndarray]] = {
    "esm": _embed_esm,
    "esmc": _embed_esmc,
    "t5": _embed_t5,
    "bert": _embed_bert,
    "ankh": _embed_ankh,
    "hf_auto": _embed_hf_auto,
    "carp": _embed_carp,
}


# --------------------------------------------------------------------------- #
# Cache-aware driver
# --------------------------------------------------------------------------- #


def _load_cache(npz_path: Path) -> Dict[str, np.ndarray]:
    if not npz_path.exists():
        return {}
    try:
        data = np.load(npz_path, allow_pickle=True)
        seqs = [str(s).strip() for s in data["sequences"].tolist()]
        emb = np.asarray(data["embeddings"], dtype=np.float32)
        return {s: emb[i] for i, s in enumerate(seqs)}
    except Exception:
        return {}


def _save(npz_path: Path, seq_to_vec: Dict[str, np.ndarray]) -> None:
    sequences = list(seq_to_vec.keys())
    embeddings = np.asarray([seq_to_vec[s] for s in sequences], dtype=np.float32)
    seq_to_index = json.dumps({s: i for i, s in enumerate(sequences)})
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        npz_path, embeddings=embeddings,
        sequences=np.array(sequences, dtype=object), seq_to_index=seq_to_index,
    )


def extract_to_npz(
    plm_name: str,
    sequences: List[str],
    output_npz: str | Path,
    batch_size: int = 8,
    force_cpu: bool = False,
) -> Tuple[Path, int, int]:
    """Compute (with caching) embeddings for ``sequences`` and write ``output_npz``.

    Returns (path, n_computed, n_from_cache).
    """
    spec = PLM_REGISTRY.get(plm_name)
    if spec is None:
        raise ValueError(f"Unknown pLM '{plm_name}'. Known: {sorted(PLM_REGISTRY)}")
    backend = _BACKENDS.get(spec.backend)
    if backend is None:
        raise NotImplementedError(
            f"Backend '{spec.backend}' for '{plm_name}' is not auto-runnable here. "
            "Use a precomputed bank or the dedicated script in scripts/embeds_scripts/."
        )

    output_npz = Path(output_npz)
    clean = [str(s).strip() for s in sequences]
    unique = list(dict.fromkeys(clean))
    cache = _load_cache(output_npz)
    missing = [s for s in unique if s not in cache]

    n_from_cache = len(unique) - len(missing)
    if missing:
        device = _device(force_cpu=force_cpu)
        vecs = backend(spec, missing, batch_size, device)
        for s, v in zip(missing, vecs):
            cache[s] = np.asarray(v, dtype=np.float32)
    _save(output_npz, cache)
    return output_npz, len(missing), n_from_cache


def extract_from_csv(
    plm_name: str,
    input_csv: str | Path,
    seq_col: str,
    output_npz: str | Path,
    batch_size: int = 8,
    force_cpu: bool = False,
) -> Tuple[Path, int, int]:
    df = pd.read_csv(input_csv)
    if seq_col not in df.columns:
        raise ValueError(f"Sequence column '{seq_col}' not found in {input_csv}")
    sequences = df[seq_col].dropna().astype(str).tolist()
    return extract_to_npz(plm_name, sequences, output_npz, batch_size=batch_size, force_cpu=force_cpu)
