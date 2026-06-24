#!/usr/bin/env python3
"""RunConfig: the single config object the GUI writes and the CLI/trainer read."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List, Optional

import yaml

from .metrics import METRIC_CHOICES
from .registry import DEFAULT_MODELS


@dataclass
class RunConfig:
    # data
    csv: str = ""
    seq_col: str = "Protein_Seq"
    target_col: str = "Activity"
    id_col: Optional[str] = None
    group_col: Optional[str] = None
    extra_feature_cols: List[str] = field(default_factory=list)   # numeric extra inputs
    categorical_cols: List[str] = field(default_factory=list)     # categorical extra inputs
    replicate_policy: str = "mean_by_sequence"

    # features
    feature_sources: List[str] = field(default_factory=lambda: ["esm2"])
    embedding_dir: Optional[str] = "embeddings"
    feature_mode_options: Optional[List[str]] = None
    wt_sequence: Optional[str] = None

    # models
    models: List[str] = field(default_factory=lambda: list(DEFAULT_MODELS))

    # search
    metric: str = "spearman"
    auto_size: bool = True          # let the size engine pick CV / budget / gating
    standard_search: bool = False   # single sources, raw, no reducer/target-transform
    cv_splits: int = 5
    cv_strategy: str = "auto"       # auto|group|kfold|repeated|holdout (used if auto_size is False)
    n_trials: Optional[int] = None  # None -> size engine default
    timeout: Optional[int] = None
    top_ensemble: int = 5
    no_uncertainty: bool = False
    conformal_alpha: float = 0.10
    random_state: int = 42
    use_gpu: bool = False

    # output
    out_dir: str = "runs/run"

    # optional inline prediction
    predict_csv: Optional[str] = None
    predict_seq_col: Optional[str] = None
    predict_id_col: Optional[str] = None
    predict_embedding_dir: Optional[str] = None

    def validate(self) -> None:
        if not self.csv:
            raise ValueError("config.csv (training CSV path) is required")
        if not self.seq_col:
            raise ValueError("config.seq_col is required")
        if not self.target_col:
            raise ValueError("config.target_col is required")
        if self.metric not in METRIC_CHOICES:
            raise ValueError(f"metric must be one of {METRIC_CHOICES}, got {self.metric}")
        if not self.feature_sources:
            raise ValueError("at least one feature source is required")
        if not self.models:
            raise ValueError("at least one model is required")

    def to_dict(self) -> dict:
        return asdict(self)

    def to_yaml(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(self.to_dict(), handle, sort_keys=False)

    @classmethod
    def from_dict(cls, data: dict) -> "RunConfig":
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in data.items() if k in known})

    @classmethod
    def from_yaml(cls, path: str | Path) -> "RunConfig":
        with Path(path).open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
        return cls.from_dict(data)
