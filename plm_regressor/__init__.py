"""PLM-Regressor: general sequence -> property regression framework.

A config-driven, size-aware MLDE / property-prediction toolkit:
- many protein language model embeddings + simple sequence encodings + extra tabular columns
- a large regressor zoo (classical + deep MLP/FNN/CNN)
- Optuna search that adapts to dataset size (100 -> 1M+)
- leakage-safe CV, OOF model selection, conformal/ensemble uncertainty
- rich metrics + plots, a Streamlit GUI, and a thin CLI
"""

__version__ = "0.2.0"
