#!/usr/bin/env python3
"""Dataset-size engine.

Maps the number of (replicate-aggregated) training rows to a :class:`SizeProfile`
that drives CV strategy, Optuna trial budget, big-data memory handling, and
model eligibility. Everything here is a *default* the user can override from the
config / GUI.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class SizeProfile:
    tier: str
    cv_strategy: str          # auto | group | kfold | repeated | holdout
    cv_splits: int
    n_repeats: int
    holdout_fraction: float
    trial_budget: int
    tune_subsample: Optional[int]  # cap rows used during HP search (None = use all)
    refit_full: bool               # refit best config on full data after subsampled tuning
    mmap_embeddings: bool          # load embedding banks with mmap to bound RAM

    def describe(self) -> str:
        cv = self.cv_strategy
        if cv == "repeated":
            cv = f"RepeatedKFold({self.cv_splits}x{self.n_repeats})"
        elif cv == "holdout":
            cv = f"Holdout({int(self.holdout_fraction * 100)}% val)"
        else:
            cv = f"{cv}({self.cv_splits} splits)"
        extra = ""
        if self.tune_subsample:
            extra = f", tune on <= {self.tune_subsample} rows" + (", refit full" if self.refit_full else "")
        return f"[{self.tier}] {cv}, ~{self.trial_budget} trials{extra}"


# (upper_bound_exclusive, kwargs) ordered small -> large
_TIERS = [
    (300,      dict(tier="tiny",   cv_strategy="repeated", cv_splits=5, n_repeats=3,
                     holdout_fraction=0.2, trial_budget=100, tune_subsample=None,
                     refit_full=False, mmap_embeddings=False)),
    (1_000,    dict(tier="small",  cv_strategy="auto", cv_splits=5, n_repeats=1,
                     holdout_fraction=0.2, trial_budget=80, tune_subsample=None,
                     refit_full=False, mmap_embeddings=False)),
    (5_000,    dict(tier="medium", cv_strategy="auto", cv_splits=5, n_repeats=1,
                     holdout_fraction=0.2, trial_budget=60, tune_subsample=None,
                     refit_full=False, mmap_embeddings=False)),
    (50_000,   dict(tier="large",  cv_strategy="kfold", cv_splits=3, n_repeats=1,
                     holdout_fraction=0.2, trial_budget=40, tune_subsample=None,
                     refit_full=False, mmap_embeddings=False)),
    (500_000,  dict(tier="xlarge", cv_strategy="holdout", cv_splits=3, n_repeats=1,
                     holdout_fraction=0.15, trial_budget=30, tune_subsample=20_000,
                     refit_full=True, mmap_embeddings=True)),
]
_HUGE = dict(tier="huge", cv_strategy="holdout", cv_splits=3, n_repeats=1,
             holdout_fraction=0.1, trial_budget=20, tune_subsample=20_000,
             refit_full=True, mmap_embeddings=True)


def profile_for_n(n: int) -> SizeProfile:
    for upper, kwargs in _TIERS:
        if n < upper:
            return SizeProfile(**kwargs)
    return SizeProfile(**_HUGE)
