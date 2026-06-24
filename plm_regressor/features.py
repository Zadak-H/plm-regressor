#!/usr/bin/env python3
"""Extra-column ("tabular") feature support.

Lets the framework feed non-sequence inputs (pH, temperature, assay conditions,
...) to the regressor alongside pLM / one-hot features. The encoder is fit on the
training frame and reused for prediction frames so columns line up.

Design choices for leakage-safety:
- numeric columns are passed through as-is; the per-trial pipeline scaler does any
  scaling *inside* each CV fold (so no target leakage)
- categorical columns are one-hot encoded against categories observed in training
  (label-free, so encoding the prediction frame with the same encoder is fine)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd


@dataclass
class TabularEncoder:
    numeric_cols: List[str] = field(default_factory=list)
    categorical_cols: List[str] = field(default_factory=list)
    categories_: Dict[str, List[str]] = field(default_factory=dict)
    numeric_means_: Dict[str, float] = field(default_factory=dict)
    feature_names_: List[str] = field(default_factory=list)

    def fit(self, df: pd.DataFrame) -> "TabularEncoder":
        self.categories_ = {}
        self.numeric_means_ = {}
        self.feature_names_ = []
        for col in self.numeric_cols:
            vals = pd.to_numeric(df[col], errors="coerce")
            self.numeric_means_[col] = float(np.nanmean(vals.to_numpy())) if len(vals) else 0.0
            self.feature_names_.append(f"num__{col}")
        for col in self.categorical_cols:
            cats = sorted(df[col].astype(str).fillna("NA").unique().tolist())
            self.categories_[col] = cats
            self.feature_names_.extend([f"cat__{col}={c}" for c in cats])
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        cols: List[np.ndarray] = []
        n = len(df)
        for col in self.numeric_cols:
            if col in df.columns:
                vals = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
            else:
                vals = np.full(n, np.nan)
            mean = self.numeric_means_.get(col, 0.0)
            vals = np.where(np.isnan(vals), mean, vals)
            cols.append(vals.reshape(-1, 1))
        for col in self.categorical_cols:
            cats = self.categories_.get(col, [])
            series = df[col].astype(str).fillna("NA") if col in df.columns else pd.Series(["NA"] * n)
            onehot = np.zeros((n, len(cats)), dtype=np.float32)
            cat_index = {c: i for i, c in enumerate(cats)}
            for row_idx, value in enumerate(series.tolist()):
                j = cat_index.get(value)
                if j is not None:
                    onehot[row_idx, j] = 1.0
            cols.append(onehot)
        if not cols:
            return np.zeros((n, 0), dtype=np.float32)
        return np.concatenate(cols, axis=1).astype(np.float32)

    @property
    def dim(self) -> int:
        return len(self.feature_names_)


def build_tabular_encoder(
    df: pd.DataFrame,
    numeric_cols: Optional[Sequence[str]] = None,
    categorical_cols: Optional[Sequence[str]] = None,
) -> Optional[TabularEncoder]:
    numeric_cols = list(numeric_cols or [])
    categorical_cols = list(categorical_cols or [])
    if not numeric_cols and not categorical_cols:
        return None
    return TabularEncoder(numeric_cols=numeric_cols, categorical_cols=categorical_cols).fit(df)
