#!/usr/bin/env python3
"""Core utilities for the PLM-Regressor framework (relocated + extended from the
original ``scripts/mlde_utils.py``).

Centralizes:
- feature-bank loading for learned embeddings (with optional mmap for big data)
- simple positional sequence encodings (one-hot and BLOSUM62)
- CV helpers and metric computation (metrics live in :mod:`plm_regressor.metrics`)
- standardized OOF / prediction table formatting
- picklable fitted-model artifacts used across train/predict/GUI
"""

from __future__ import annotations

import json
import math
import os
import random
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import GroupKFold, KFold, RepeatedKFold
from sklearn.compose import TransformedTargetRegressor
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression

import warnings

from .metrics import compute_metrics  # re-exported for back-compat

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning)


COMPUTED_FEATURE_SOURCES = {"onehot", "blosum62"}
# Feature source name reserved for user-supplied extra numeric/categorical columns.
TABULAR_FEATURE_SOURCE = "tabular"
AA_ALPHABET: Tuple[str, ...] = tuple("ACDEFGHIKLMNPQRSTVWY")
AA_TO_INDEX = {aa: idx for idx, aa in enumerate(AA_ALPHABET)}

# Standard 20x20 BLOSUM62 restricted to canonical amino acids in AA_ALPHABET order.
_BLOSUM62_ROWS = {
    "A": [4, 0, -2, -1, -2, 0, -2, -1, -1, -1, -1, -1, -1, -2, -1, 1, 0, -3, -2, 0],
    "C": [0, 9, -3, -4, -2, -3, -3, -1, -3, -1, -1, -3, -1, -2, -3, -1, -1, -2, -2, -1],
    "D": [-2, -3, 6, 2, -3, -1, -1, -3, -1, -3, -4, -1, -3, -3, -1, 0, -1, -4, -3, -3],
    "E": [-1, -4, 2, 5, -3, -2, 0, -3, 1, -3, -3, 1, -2, -3, -1, 0, -1, -3, -2, -2],
    "F": [-2, -2, -3, -3, 6, -3, -3, -1, -3, 0, 0, -3, 0, 1, -3, -2, -2, 1, 3, -1],
    "G": [0, -3, -1, -2, -3, 6, -2, -4, -2, -4, -4, -2, -3, -3, -2, 0, -2, -2, -3, -3],
    "H": [-2, -3, -1, 0, -3, -2, 8, -3, -1, -3, -3, -1, -2, -1, -2, -1, -2, -2, 2, -3],
    "I": [-1, -1, -3, -3, -1, -4, -3, 4, -3, 2, 1, -3, 1, 0, -3, -2, -1, -3, -1, 3],
    "K": [-1, -3, -1, 1, -3, -2, -1, -3, 5, -2, -3, 1, -1, -3, -1, 0, -1, -3, -2, -2],
    "L": [-1, -1, -3, -3, 0, -4, -3, 2, -2, 4, 2, -2, 2, 0, -3, -2, -1, -2, -1, 1],
    "M": [-1, -1, -4, -3, 0, -4, -3, 1, -3, 2, 5, -2, 3, 0, -2, -1, -1, -1, -1, 1],
    "N": [-1, -3, -1, 1, -3, -2, -1, -3, 1, -2, -2, 6, -2, -4, -2, 0, -1, -4, -2, -3],
    "P": [-1, -1, -3, -2, 0, -3, -2, 1, -1, 2, 3, -2, 7, -1, -2, -1, -1, -1, -1, 1],
    "Q": [-2, -2, -3, -3, 1, -3, -1, 0, -3, 0, 0, -4, -1, 5, -1, -2, -2, 1, 3, -1],
    "R": [-1, -3, -1, -1, -3, -2, -2, -3, -1, -3, -2, -2, -2, -1, 5, -1, -1, -3, -2, -3],
    "S": [1, -1, 0, 0, -2, 0, -1, -2, 0, -2, -1, 0, -1, -2, -1, 4, 1, -3, -2, -2],
    "T": [0, -1, -1, -1, -2, -2, -2, -1, -1, -1, -1, -1, -1, -2, -1, 1, 5, -2, -2, 0],
    "W": [-3, -2, -4, -3, 1, -2, -2, -3, -3, -2, -1, -4, -1, 1, -3, -3, -2, 11, 2, -3],
    "Y": [-2, -2, -3, -2, 3, -3, 2, -1, -2, -1, -1, -2, -1, 3, -2, -2, -2, 2, 7, -1],
    "V": [0, -1, -3, -2, -1, -3, -3, 3, -2, 1, 1, -3, 1, -1, -3, -2, 0, -3, -1, 4],
}
BLOSUM62_MATRIX = np.asarray([_BLOSUM62_ROWS[aa] for aa in AA_ALPHABET], dtype=np.float32)


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def json_default(obj: Any) -> Any:
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (Path,)):
        return str(obj)
    return str(obj)


def save_json(path: str | Path, payload: Dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=json_default)


def load_json(path: str | Path) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def print_header(title: str) -> None:
    print("\n" + "=" * 92)
    print(title)
    print("=" * 92)


def conformal_q(abs_residuals: np.ndarray, alpha: float) -> float:
    scores = np.asarray(abs_residuals, dtype=float).ravel()
    scores = np.sort(scores[~np.isnan(scores)])
    if len(scores) == 0:
        return float("nan")
    q_idx = int(math.ceil((len(scores) + 1) * (1 - alpha))) - 1
    q_idx = min(max(q_idx, 0), len(scores) - 1)
    return float(scores[q_idx])


@dataclass
class EmbeddingBank:
    name: str
    path: str
    seq_to_vec: Dict[str, np.ndarray]
    dim: int


class IdentityTransformer(TransformerMixin, BaseEstimator):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X


class ColumnSelectorKBest(TransformerMixin, BaseEstimator):
    def __init__(self, score_func: str = "f_regression", k: int = 100):
        self.score_func = score_func
        self.k = int(k)
        self.selector_: Optional[SelectKBest] = None

    def fit(self, X, y):
        k_eff = min(self.k, X.shape[1])
        func = f_regression if self.score_func == "f_regression" else mutual_info_regression
        self.selector_ = SelectKBest(score_func=func, k=k_eff)
        self.selector_.fit(X, y)
        return self

    def transform(self, X):
        if self.selector_ is None:
            raise RuntimeError("ColumnSelectorKBest not fitted")
        return self.selector_.transform(X)


def normalize_feature_source(name: str) -> str:
    lowered = name.strip().lower()
    if lowered in COMPUTED_FEATURE_SOURCES or lowered == TABULAR_FEATURE_SOURCE:
        return lowered
    return name.strip()


def load_embedding_npz(npz_path: str, mmap: bool = False) -> Tuple[List[str], np.ndarray]:
    data = np.load(npz_path, allow_pickle=True, mmap_mode="r" if mmap else None)
    keys = set(data.files)

    def to_list(values: Any) -> List[str]:
        if isinstance(values, np.ndarray):
            values = values.tolist()
        return [str(item).strip() for item in values]

    if "sequences" in keys and "embeddings" in keys:
        sequences = to_list(data["sequences"])
        embeddings = np.asarray(data["embeddings"])
    elif "seqs" in keys and "embeddings" in keys:
        sequences = to_list(data["seqs"])
        embeddings = np.asarray(data["embeddings"])
    elif "arr_0" in keys:
        obj = data["arr_0"]
        try:
            obj = obj.item()
        except Exception:
            pass
        if isinstance(obj, dict) and "embeddings" in obj and "sequences" in obj:
            sequences = to_list(obj["sequences"])
            embeddings = np.asarray(obj["embeddings"])
        elif isinstance(obj, dict):
            sequences = to_list(obj.keys())
            embeddings = np.stack([obj[seq] for seq in sequences], axis=0)
        else:
            raise ValueError(f"Unsupported arr_0 layout in {npz_path}")
    elif "seq_to_index" in keys and "embeddings" in keys:
        raw = data["seq_to_index"]
        if hasattr(raw, "item"):
            try:
                raw = raw.item()
            except Exception:
                pass
        if isinstance(raw, str):
            raw = json.loads(raw)
        if not isinstance(raw, dict):
            raise ValueError(f"Unsupported seq_to_index type in {npz_path}")
        sequences = [str(key).strip() for key, _ in sorted(raw.items(), key=lambda item: int(item[1]))]
        embeddings = np.asarray(data["embeddings"])
    else:
        raise ValueError(f"Unsupported embedding format in {npz_path}. Keys={sorted(keys)}")

    if embeddings.ndim == 3:
        embeddings = embeddings.mean(axis=1)
    if embeddings.ndim != 2:
        raise ValueError(f"Expected 2D embeddings after pooling, got {embeddings.shape} in {npz_path}")
    return sequences, embeddings.astype(np.float32, copy=False)


def build_embedding_bank(npz_path: str, name: Optional[str] = None, mmap: bool = False) -> EmbeddingBank:
    sequences, embeddings = load_embedding_npz(npz_path, mmap=mmap)
    seq_to_vec = {str(seq).strip(): embeddings[idx] for idx, seq in enumerate(sequences)}
    return EmbeddingBank(
        name=name or Path(npz_path).stem,
        path=str(npz_path),
        seq_to_vec=seq_to_vec,
        dim=int(embeddings.shape[1]),
    )


def resolve_embedding_paths(
    feature_sources: Sequence[str],
    embedding_dir: Optional[str] = None,
    explicit_embedding_paths: Optional[Sequence[str]] = None,
) -> Dict[str, str]:
    learned_sources = [
        normalize_feature_source(source)
        for source in feature_sources
        if normalize_feature_source(source) not in COMPUTED_FEATURE_SOURCES
        and normalize_feature_source(source) != TABULAR_FEATURE_SOURCE
    ]
    paths: Dict[str, str] = {}

    if explicit_embedding_paths:
        for raw_path in explicit_embedding_paths:
            path = Path(raw_path)
            paths[path.stem] = str(path)

    if embedding_dir:
        embedding_dir_path = Path(embedding_dir)
        for source in learned_sources:
            if source in paths:
                continue
            candidate = embedding_dir_path / f"{source}.npz"
            if not candidate.exists():
                raise FileNotFoundError(
                    f"Expected embedding file for feature source '{source}' at {candidate}"
                )
            paths[source] = str(candidate)

    missing = [source for source in learned_sources if source not in paths]
    if missing:
        raise ValueError(
            "Missing embedding paths for learned feature sources: " + ", ".join(sorted(missing))
        )
    return paths


def load_embedding_banks(
    feature_sources: Sequence[str],
    embedding_dir: Optional[str] = None,
    explicit_embedding_paths: Optional[Sequence[str]] = None,
    mmap: bool = False,
) -> Dict[str, EmbeddingBank]:
    resolved = resolve_embedding_paths(
        feature_sources=feature_sources,
        embedding_dir=embedding_dir,
        explicit_embedding_paths=explicit_embedding_paths,
    )
    return {name: build_embedding_bank(path, name=name, mmap=mmap) for name, path in resolved.items()}


def infer_sequence_length(sequences: Sequence[str], expected_length: Optional[int] = None) -> int:
    lengths = {len(str(sequence).strip()) for sequence in sequences}
    if not lengths:
        raise ValueError("Cannot infer sequence length from an empty sequence collection")
    if len(lengths) != 1:
        raise ValueError(
            "One-hot and BLOSUM62 encodings require aligned fixed-length sequences. "
            f"Observed lengths: {sorted(lengths)}"
        )
    length = next(iter(lengths))
    if expected_length is not None and length != expected_length:
        raise ValueError(f"Sequence length mismatch: expected {expected_length}, observed {length}")
    return length


def encode_onehot_sequences(sequences: Sequence[str], expected_length: Optional[int] = None) -> np.ndarray:
    sequences = [str(sequence).strip() for sequence in sequences]
    seq_len = infer_sequence_length(sequences, expected_length=expected_length)
    encoded = np.zeros((len(sequences), seq_len, len(AA_ALPHABET)), dtype=np.float32)
    for row_idx, sequence in enumerate(sequences):
        for pos_idx, aa in enumerate(sequence):
            aa_idx = AA_TO_INDEX.get(aa)
            if aa_idx is not None:
                encoded[row_idx, pos_idx, aa_idx] = 1.0
    return encoded.reshape(len(sequences), seq_len * len(AA_ALPHABET))


def encode_blosum62_sequences(sequences: Sequence[str], expected_length: Optional[int] = None) -> np.ndarray:
    sequences = [str(sequence).strip() for sequence in sequences]
    seq_len = infer_sequence_length(sequences, expected_length=expected_length)
    encoded = np.zeros((len(sequences), seq_len, len(AA_ALPHABET)), dtype=np.float32)
    for row_idx, sequence in enumerate(sequences):
        for pos_idx, aa in enumerate(sequence):
            aa_idx = AA_TO_INDEX.get(aa)
            if aa_idx is not None:
                encoded[row_idx, pos_idx, :] = BLOSUM62_MATRIX[aa_idx]
    return encoded.reshape(len(sequences), seq_len * len(AA_ALPHABET))


def assemble_single_embedding(sequences: Sequence[str], bank: EmbeddingBank) -> Tuple[np.ndarray, np.ndarray]:
    clean_sequences = [str(sequence).strip() for sequence in sequences]
    matrix = np.zeros((len(clean_sequences), bank.dim), dtype=np.float32)
    missing = np.zeros(len(clean_sequences), dtype=bool)
    for idx, sequence in enumerate(clean_sequences):
        vec = bank.seq_to_vec.get(sequence)
        if vec is None:
            missing[idx] = True
        else:
            matrix[idx] = vec
    return matrix, missing


def assemble_feature_matrices(
    df: pd.DataFrame,
    seq_col: str,
    feature_sources: Sequence[str],
    embedding_banks: Optional[Dict[str, EmbeddingBank]] = None,
    expected_sequence_length: Optional[int] = None,
    tabular_matrix: Optional[np.ndarray] = None,
) -> Tuple[Dict[str, np.ndarray], np.ndarray, Dict[str, int], int]:
    """Build a per-source feature matrix dict.

    ``tabular_matrix`` (if given) is used verbatim for the reserved ``tabular``
    feature source; the caller is responsible for building it leakage-safely
    (see :func:`plm_regressor.features.build_tabular_matrix`).
    """
    sequences = df[seq_col].astype(str).str.strip().tolist()
    matrices: Dict[str, np.ndarray] = {}
    missing_any = np.zeros(len(df), dtype=bool)
    source_missing_counts: Dict[str, int] = {}
    computed_len: Optional[int] = None

    for raw_source in feature_sources:
        source = normalize_feature_source(raw_source)
        if source == "onehot":
            matrices[source] = encode_onehot_sequences(sequences, expected_length=expected_sequence_length)
            source_missing_counts[source] = 0
            computed_len = matrices[source].shape[1] // len(AA_ALPHABET)
            continue
        if source == "blosum62":
            matrices[source] = encode_blosum62_sequences(sequences, expected_length=expected_sequence_length)
            source_missing_counts[source] = 0
            computed_len = matrices[source].shape[1] // len(AA_ALPHABET)
            continue
        if source == TABULAR_FEATURE_SOURCE:
            if tabular_matrix is None:
                raise ValueError("tabular feature source requested but no tabular_matrix supplied")
            matrices[source] = np.asarray(tabular_matrix, dtype=np.float32)
            source_missing_counts[source] = 0
            continue

        if embedding_banks is None or source not in embedding_banks:
            raise ValueError(f"Missing embedding bank for feature source '{source}'")
        matrix, missing = assemble_single_embedding(sequences, embedding_banks[source])
        matrices[source] = matrix
        missing_any |= missing
        source_missing_counts[source] = int(missing.sum())

    if computed_len is None:
        try:
            computed_len = infer_sequence_length(sequences, expected_length=expected_sequence_length)
        except ValueError:
            # Variable-length sequences are fine when no positional encoding is used.
            computed_len = expected_sequence_length or 0
    return matrices, missing_any, source_missing_counts, computed_len


def transform_feature_mode(
    X_by_source_raw: Dict[str, np.ndarray],
    wt_index: Optional[int],
    feature_mode: str,
    wt_by_source: Optional[Dict[str, np.ndarray]] = None,
) -> Dict[str, np.ndarray]:
    if feature_mode not in {"raw", "delta", "raw_plus_delta"}:
        raise ValueError(f"Unsupported feature mode: {feature_mode}")
    if feature_mode != "raw" and wt_index is None and wt_by_source is None:
        raise ValueError("WT reference is required for delta-based feature modes")

    transformed: Dict[str, np.ndarray] = {}
    for source, matrix in X_by_source_raw.items():
        if feature_mode == "raw":
            transformed[source] = matrix
            continue
        if wt_by_source is not None and source in wt_by_source:
            wt_vec = wt_by_source[source]
        else:
            wt_vec = matrix[wt_index]
        delta = matrix - wt_vec
        if feature_mode == "delta":
            transformed[source] = delta.astype(np.float32, copy=False)
        else:
            transformed[source] = np.concatenate([matrix, delta], axis=1).astype(np.float32, copy=False)
    return transformed


def all_nonempty_feature_subsets(names: Sequence[str]) -> List[Tuple[str, ...]]:
    clean_names = [normalize_feature_source(name) for name in names]
    subsets: List[Tuple[str, ...]] = []
    for size in range(1, len(clean_names) + 1):
        subsets.extend(combinations(clean_names, size))
    return subsets


def make_cv_splits(
    n_samples: int,
    groups: np.ndarray,
    cv_splits: int,
    random_state: int,
    strategy: str = "auto",
    n_repeats: int = 1,
    holdout_fraction: float = 0.2,
) -> Tuple[List[Tuple[np.ndarray, np.ndarray]], bool]:
    """Build CV splits. ``strategy`` (from the size engine):

    - ``auto``      : GroupKFold if enough groups, else shuffled KFold (legacy behavior)
    - ``group``     : force GroupKFold when possible
    - ``kfold``     : plain shuffled KFold
    - ``repeated``  : RepeatedKFold (small-data, ``n_repeats`` repeats)
    - ``holdout``   : single shuffled train/val split (big-data)
    """
    groups = np.asarray(groups)
    n_unique_groups = len(np.unique(groups))

    if strategy == "holdout":
        rng = np.random.RandomState(random_state)
        perm = rng.permutation(n_samples)
        n_val = max(1, int(round(n_samples * holdout_fraction)))
        valid_idx = np.sort(perm[:n_val])
        train_idx = np.sort(perm[n_val:])
        return [(train_idx, valid_idx)], False

    if strategy == "repeated":
        n_splits_eff = max(2, min(cv_splits, n_samples))
        splitter = RepeatedKFold(n_splits=n_splits_eff, n_repeats=max(1, n_repeats), random_state=random_state)
        splits = list(splitter.split(np.zeros((n_samples, 1))))
        return splits, False

    if strategy in {"auto", "group"} and n_unique_groups >= cv_splits:
        splitter = GroupKFold(n_splits=cv_splits)
        splits = list(splitter.split(np.zeros((n_samples, 1)), np.zeros(n_samples), groups=groups))
        return splits, True

    n_splits_eff = max(2, min(cv_splits, n_samples))
    splitter = KFold(n_splits=n_splits_eff, shuffle=True, random_state=random_state)
    splits = list(splitter.split(np.zeros((n_samples, 1))))
    return splits, False


def predict_estimator_with_optional_std(
    estimator: BaseEstimator,
    X: np.ndarray,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    if isinstance(estimator, TransformedTargetRegressor):
        return np.asarray(estimator.predict(X)).ravel(), None

    pred = np.asarray(estimator.predict(X)).ravel()
    native_std: Optional[np.ndarray] = None
    if hasattr(estimator, "named_steps") and "model" in estimator.named_steps:
        final_model = estimator.named_steps["model"]
        try:
            transformed_X = estimator[:-1].transform(X)
            pred_native, native_std = final_model.predict(transformed_X, return_std=True)
            pred = np.asarray(pred_native).ravel()
            native_std = np.asarray(native_std).ravel()
        except TypeError:
            native_std = None
        except Exception:
            native_std = None
    return pred, native_std


def cross_val_predict_with_uncertainty(
    estimator: BaseEstimator,
    X: np.ndarray,
    y: np.ndarray,
    splits: Sequence[Tuple[np.ndarray, np.ndarray]],
) -> Tuple[Dict[str, float], np.ndarray, np.ndarray, np.ndarray, List[Dict[str, float]]]:
    oof = np.full(len(y), np.nan, dtype=float)
    native_std = np.full(len(y), np.nan, dtype=float)
    fold_ids = np.full(len(y), -1, dtype=int)
    fold_rows: List[Dict[str, float]] = []
    counts = np.zeros(len(y), dtype=int)
    oof_sum = np.zeros(len(y), dtype=float)

    for fold_number, (train_idx, valid_idx) in enumerate(splits, start=1):
        model = clone(estimator)
        model.fit(X[train_idx], y[train_idx])
        pred, pred_std = predict_estimator_with_optional_std(model, X[valid_idx])
        oof_sum[valid_idx] += pred
        counts[valid_idx] += 1
        fold_ids[valid_idx] = fold_number
        if pred_std is not None:
            native_std[valid_idx] = pred_std
        metrics = compute_metrics(y[valid_idx], pred)
        metrics["fold"] = fold_number
        metrics["n_train"] = int(len(train_idx))
        metrics["n_valid"] = int(len(valid_idx))
        fold_rows.append(metrics)

    # RepeatedKFold visits each row more than once; average those predictions.
    # Holdout leaves train rows unseen on purpose -> they stay NaN and are simply
    # excluded from the OOF metric (metrics are computed on validation rows only).
    seen = counts > 0
    oof[seen] = oof_sum[seen] / counts[seen]
    if not seen.any():
        raise RuntimeError("No validation predictions were produced")

    return compute_metrics(y[seen], oof[seen]), oof, native_std, fold_ids, fold_rows


def prepare_supervised_dataframe(
    df: pd.DataFrame,
    seq_col: str,
    target_col: str,
    replicate_policy: str = "mean_by_sequence",
    id_col: Optional[str] = None,
    keep_cols: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """Clean + (optionally) aggregate replicate rows.

    ``keep_cols`` are extra columns (e.g. tabular feature columns) preserved
    through aggregation; for mean/median policies the first value per sequence
    group is kept.
    """
    work = df.copy()
    work[seq_col] = work[seq_col].astype(str).str.strip()
    work[target_col] = pd.to_numeric(work[target_col], errors="coerce")
    work = work[work[target_col].notna() & work[seq_col].notna()].copy()
    work = work[work[seq_col].str.len() > 0].reset_index(drop=True)

    if replicate_policy == "keep_rows":
        work["is_aggregated_row"] = False
        work["n_source_rows"] = 1
        if id_col and id_col in work.columns:
            work["source_ids"] = work[id_col].astype(str)
        return work

    if replicate_policy not in {"mean_by_sequence", "median_by_sequence"}:
        raise ValueError(f"Unsupported replicate policy: {replicate_policy}")

    # Vectorized aggregation: keep the first row per sequence for all non-target
    # columns, and replace the target with the per-sequence mean/median. This is
    # O(N) in pandas rather than a Python loop over groups (critical for big data).
    agg_func = "mean" if replicate_policy == "mean_by_sequence" else "median"
    grouped = work.groupby(seq_col, sort=False, dropna=False)
    agg_target = grouped[target_col].agg(agg_func)
    counts = grouped.size()

    base = work.drop_duplicates(subset=[seq_col], keep="first").reset_index(drop=True)
    base[target_col] = base[seq_col].map(agg_target).astype(float)
    base["is_aggregated_row"] = True
    base["n_source_rows"] = base[seq_col].map(counts).astype(int)
    if id_col and id_col in work.columns:
        if len(work) <= 200_000:
            ids = grouped[id_col].agg(lambda s: "|".join(map(str, s)))
            base["source_ids"] = base[seq_col].map(ids)
        else:  # avoid the per-group join cost on very large frames
            base["source_ids"] = base[id_col].astype(str)
    return base


def add_rank_column(df: pd.DataFrame, pred_col: str = "y_pred", ascending: bool = False) -> pd.DataFrame:
    ranked = df.copy()
    valid_mask = ranked[pred_col].notna()
    ranks = pd.Series(np.nan, index=ranked.index, dtype=float)
    ranks.loc[valid_mask] = ranked.loc[valid_mask, pred_col].rank(method="first", ascending=ascending)
    ranked["rank"] = ranks
    return ranked.sort_values(["rank", pred_col], ascending=[True, ascending], na_position="last").reset_index(drop=True)


def build_oof_dataframe(
    df: pd.DataFrame,
    run_name: str,
    target_col: str,
    y_pred: np.ndarray,
    fold_ids: np.ndarray,
    group_values: Sequence[str],
    conformal_qhat: Optional[float],
    native_std: Optional[np.ndarray] = None,
    ensemble_mean: Optional[np.ndarray] = None,
    ensemble_std: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    out = df.copy()
    y_true = pd.to_numeric(out[target_col], errors="coerce").to_numpy(dtype=float)
    out["run_name"] = run_name
    out["group_id"] = np.asarray(group_values).astype(str)
    out["fold"] = fold_ids.astype(int)
    out["y_true"] = y_true
    out["y_pred"] = np.asarray(y_pred).ravel()
    out["residual"] = out["y_true"] - out["y_pred"]
    out["pred_native_std"] = np.nan if native_std is None else np.asarray(native_std).ravel()
    out["ensemble_mean"] = np.nan if ensemble_mean is None else np.asarray(ensemble_mean).ravel()
    out["pred_ensemble_std"] = np.nan if ensemble_std is None else np.asarray(ensemble_std).ravel()
    if conformal_qhat is None or np.isnan(conformal_qhat):
        out["pi_lower"] = np.nan
        out["pi_upper"] = np.nan
    else:
        out["pi_lower"] = out["y_pred"] - float(conformal_qhat)
        out["pi_upper"] = out["y_pred"] + float(conformal_qhat)
    if "is_aggregated_row" not in out.columns:
        out["is_aggregated_row"] = False
    return out


def build_prediction_dataframe(
    df: pd.DataFrame,
    run_name: str,
    y_pred: np.ndarray,
    conformal_qhat: Optional[float],
    ensemble_mean: Optional[np.ndarray] = None,
    native_std: Optional[np.ndarray] = None,
    ensemble_std: Optional[np.ndarray] = None,
    seen_in_train: Optional[np.ndarray] = None,
    missing_any_feature: Optional[np.ndarray] = None,
    ascending: bool = False,
) -> pd.DataFrame:
    out = df.copy()
    out["run_name"] = run_name
    out["y_pred"] = np.asarray(y_pred).ravel()
    out["ensemble_mean"] = np.nan if ensemble_mean is None else np.asarray(ensemble_mean).ravel()
    out["pred_native_std"] = np.nan if native_std is None else np.asarray(native_std).ravel()
    out["pred_ensemble_std"] = np.nan if ensemble_std is None else np.asarray(ensemble_std).ravel()
    if conformal_qhat is None or np.isnan(conformal_qhat):
        out["pi_lower"] = np.nan
        out["pi_upper"] = np.nan
    else:
        out["pi_lower"] = out["y_pred"] - float(conformal_qhat)
        out["pi_upper"] = out["y_pred"] + float(conformal_qhat)
    out["seen_in_train"] = False if seen_in_train is None else np.asarray(seen_in_train).astype(bool)
    out["missing_any_feature"] = False if missing_any_feature is None else np.asarray(missing_any_feature).astype(bool)
    return add_rank_column(out, pred_col="y_pred", ascending=ascending)


@dataclass
class FittedRunModel:
    run_name: str
    trial_number: int
    score: float
    metric_name: str
    model_name: str
    feature_sources: Tuple[str, ...]
    feature_mode: str
    estimator: BaseEstimator
    conformal_alpha: float
    conformal_qhat: Optional[float]
    train_sequences: Tuple[str, ...] = field(default_factory=tuple)
    expected_sequence_length: Optional[int] = None
    wt_by_source: Dict[str, np.ndarray] = field(default_factory=dict)

    def build_matrix(self, X_by_source: Dict[str, np.ndarray]) -> np.ndarray:
        return np.concatenate([X_by_source[source] for source in self.feature_sources], axis=1)

    def predict(self, X_by_source: Dict[str, np.ndarray]) -> np.ndarray:
        pred, _ = self.predict_with_uncertainty(X_by_source)
        return pred

    def predict_with_uncertainty(
        self,
        X_by_source: Dict[str, np.ndarray],
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        matrix = self.build_matrix(X_by_source)
        return predict_estimator_with_optional_std(self.estimator, matrix)


@dataclass
class FittedEnsemble:
    run_name: str
    fitted_models: List[FittedRunModel]

    def predict_with_uncertainty(
        self,
        X_by_mode: Dict[str, Dict[str, np.ndarray]],
    ) -> Tuple[np.ndarray, np.ndarray]:
        if not self.fitted_models:
            raise ValueError("Ensemble has no fitted models")
        pred_matrix = []
        for model in self.fitted_models:
            pred, _ = model.predict_with_uncertainty(X_by_mode[model.feature_mode])
            pred_matrix.append(np.asarray(pred).ravel())
        stacked = np.vstack(pred_matrix)
        return stacked.mean(axis=0), stacked.std(axis=0, ddof=0)
