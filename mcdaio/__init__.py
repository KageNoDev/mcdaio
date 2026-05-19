from .analysis import sensitivity_analysis
from .methods import ARAS, MABAC, METHOD_REGISTRY, TOPSIS, VIKOR
from .metrics import spearman_corr, weighted_rank_corr
from .normalization import minmax_normalize

__version__ = "0.2.1"

__all__ = [
    "ARAS",
    "MABAC",
    "METHOD_REGISTRY",
    "TOPSIS",
    "VIKOR",
    "minmax_normalize",
    "sensitivity_analysis",
    "spearman_corr",
    "weighted_rank_corr",
]
