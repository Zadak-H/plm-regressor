#!/usr/bin/env python3
"""Single source of truth for the model + pLM menus.

- ``MODEL_REGISTRY``: every regressor (classical + deep), with availability,
  a size ceiling (``max_n``) used by the size engine, and any feature-source
  requirement (e.g. CNN only over positional encodings).
- ``PLM_REGISTRY``: every protein language model the extractor knows how to run,
  grouped by backend, with output dim and the package it needs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, FrozenSet, List, Optional, Tuple

from sklearn.base import BaseEstimator

from .models import classical as _classical
from .models import torch_models as _torch

# --------------------------------------------------------------------------- #
# Regressor registry
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ModelSpec:
    name: str
    kind: str                       # "classical" | "torch"
    available: bool
    max_n: Optional[int]            # rows above which this model is skipped (None = no cap)
    requires_source: Optional[FrozenSet[str]] = None  # only valid with these feature sources

    def eligible_for_n(self, n: int) -> bool:
        return self.available and (self.max_n is None or n <= self.max_n)


_POSITIONAL = frozenset({"onehot", "blosum62"})

_SPECS: List[ModelSpec] = [
    # linear / scalable
    ModelSpec("ridge", "classical", True, None),
    ModelSpec("sgd", "classical", True, None),
    ModelSpec("elasticnet", "classical", True, None),
    ModelSpec("huber", "classical", True, 200_000),
    ModelSpec("bayesian_ridge", "classical", True, 200_000),
    # kernel / instance (super-linear cost)
    ModelSpec("svr_rbf", "classical", True, 20_000),
    ModelSpec("kernel_ridge", "classical", True, 20_000),
    ModelSpec("gpr", "classical", True, 2_000),
    ModelSpec("knn", "classical", True, 50_000),
    ModelSpec("pls", "classical", True, None),
    # trees / boosting
    ModelSpec("rf", "classical", True, 200_000),
    ModelSpec("extra_trees", "classical", True, 200_000),
    ModelSpec("hist_gb", "classical", True, None),
    ModelSpec("xgboost", "classical", _classical.HAS_XGB, None),
    ModelSpec("lightgbm", "classical", _classical.HAS_LGB, None),
    # sklearn MLP (full-batch-ish, modest scale)
    ModelSpec("mlp", "classical", True, 50_000),
    # deep (torch, mini-batch -> scalable)
    ModelSpec("mlp_torch", "torch", _torch.HAS_TORCH, None),
    ModelSpec("cnn1d", "torch", _torch.HAS_TORCH, None, requires_source=_POSITIONAL),
]

MODEL_REGISTRY: Dict[str, ModelSpec] = {spec.name: spec for spec in _SPECS}

# Sensible default menu when the user does not specify models.
DEFAULT_MODELS: Tuple[str, ...] = (
    "ridge",
    "elasticnet",
    "svr_rbf",
    "knn",
    "rf",
    "hist_gb",
    "gpr",
    "mlp_torch",
)


def available_models() -> List[str]:
    return [name for name, spec in MODEL_REGISTRY.items() if spec.available]


def eligible_models(requested: List[str], n: int) -> List[str]:
    """Filter a requested model list to those available and size-eligible for n."""
    out = []
    for name in requested:
        spec = MODEL_REGISTRY.get(name)
        if spec is None or not spec.eligible_for_n(n):
            continue
        out.append(name)
    return out


def build_model(
    name: str,
    trial: "object",
    random_state: int,
    use_gpu: bool,
    n_features: int,
    n_samples_train: int,
) -> Tuple[str, BaseEstimator]:
    spec = MODEL_REGISTRY.get(name)
    if spec is None:
        raise ValueError(f"Unknown model '{name}'")
    if spec.kind == "torch":
        return _torch.build_torch_model(name, trial, random_state, use_gpu, n_features, n_samples_train)
    return _classical.build_classical_model(name, trial, random_state, use_gpu, n_features, n_samples_train)


# --------------------------------------------------------------------------- #
# Protein language model registry
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class PLMSpec:
    name: str
    backend: str          # esm | esmc | t5 | bert | ankh | hf_auto | prosst | carp
    model_id: str         # canonical model name / HF id / esm size
    dim: int
    needs: str            # pip package needed


_PLM_SPECS: List[PLMSpec] = [
    # --- ESM2 (fair-esm) ---
    PLMSpec("esm2_8m", "esm", "esm2_t6_8M_UR50D", 320, "fair-esm"),
    PLMSpec("esm2_35m", "esm", "esm2_t12_35M_UR50D", 480, "fair-esm"),
    PLMSpec("esm2_150m", "esm", "esm2_t30_150M_UR50D", 640, "fair-esm"),
    PLMSpec("esm2", "esm", "esm2_t33_650M_UR50D", 1280, "fair-esm"),
    PLMSpec("esm2_3b", "esm", "esm2_t36_3B_UR50D", 2560, "fair-esm"),
    PLMSpec("esm2_15b", "esm", "esm2_t48_15B_UR50D", 5120, "fair-esm"),
    # --- ESM1 / 1b / 1v (fair-esm) ---
    PLMSpec("esm1", "esm", "esm1_t34_670M_UR50S", 1280, "fair-esm"),
    PLMSpec("esm1b", "esm", "esm1b_t33_650M_UR50S", 1280, "fair-esm"),
    PLMSpec("esm1v", "esm", "esm1v_t33_650M_UR90S_1", 1280, "fair-esm"),
    # --- ESM C / Cambrian (EvolutionaryScale `esm` SDK; collides with fair-esm) ---
    PLMSpec("esmc_300m", "esmc", "esmc_300m", 960, "esm (evolutionaryscale)"),
    PLMSpec("esmc_600m", "esmc", "esmc_600m", 1152, "esm (evolutionaryscale)"),
    # --- ESM++ : HF re-implementation of ESM C, runs via transformers (no SDK clash) ---
    PLMSpec("esmplusplus_small", "hf_auto", "Synthyra/ESMplusplus_small", 960, "transformers"),
    PLMSpec("esmplusplus_large", "hf_auto", "Synthyra/ESMplusplus_large", 1152, "transformers"),
    # --- ProtT5 / ProstT5 (HF transformers, T5 encoder) ---
    PLMSpec("protT5", "t5", "Rostlab/prot_t5_xl_uniref50", 1024, "transformers+sentencepiece"),
    PLMSpec("protT5_half", "t5", "Rostlab/prot_t5_xl_half_uniref50-enc", 1024, "transformers+sentencepiece"),
    PLMSpec("protT5_bfd", "t5", "Rostlab/prot_t5_xl_bfd", 1024, "transformers+sentencepiece"),
    PLMSpec("protT5_xxl", "t5", "Rostlab/prot_t5_xxl_uniref50", 1024, "transformers+sentencepiece"),
    PLMSpec("prostT5", "t5", "Rostlab/ProstT5", 1024, "transformers+sentencepiece"),
    # --- ProtBert ---
    PLMSpec("protbert", "bert", "Rostlab/prot_bert", 1024, "transformers"),
    PLMSpec("protbert_bfd", "bert", "Rostlab/prot_bert_bfd", 1024, "transformers"),
    # --- Ankh ---
    PLMSpec("ankh_base", "ankh", "ElnaggarLab/ankh-base", 768, "transformers"),
    PLMSpec("ankh_large", "ankh", "ElnaggarLab/ankh-large", 1536, "transformers"),
    # --- ProSST (structure-aware; extractor needs structure tokens, see notes) ---
    PLMSpec("prosst", "prosst", "AI4Protein/ProSST-2048", 768, "transformers"),
    # --- CARP (Microsoft sequence-models) ---
    PLMSpec("carp_640m", "carp", "carp_640M", 1280, "sequence-models"),
]

PLM_REGISTRY: Dict[str, PLMSpec] = {spec.name: spec for spec in _PLM_SPECS}


def _backend_available(backend: str) -> bool:
    try:
        if backend == "esm":
            import esm  # fair-esm  # noqa: F401

            return hasattr(__import__("esm"), "pretrained")
        if backend == "esmc":
            from esm.models.esmc import ESMC  # noqa: F401

            return True
        if backend == "carp":
            import sequence_models  # noqa: F401

            return True
        # t5 | bert | ankh | hf_auto | prosst all run through HF transformers
        import transformers  # noqa: F401

        return True
    except Exception:
        return False


def available_plms() -> List[str]:
    return [name for name, spec in PLM_REGISTRY.items() if _backend_available(spec.backend)]
