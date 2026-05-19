from __future__ import annotations
import numpy as np
from .methods import EPS

def spearman_corr(r1, r2) -> float:
    
    a = np.asarray(r1, dtype=float)
    b = np.asarray(r2, dtype=float)
    if a.shape != b.shape:
        raise ValueError("Rankings must be the same size.")
    n = len(a)
    if n < 2:
        return 1.0
    d2 = np.sum((a - b) ** 2)
    return float(1 - (6 * d2) / (n * (n ** 2 - 1) + EPS))

def weighted_rank_corr(r1, r2) -> float:

    a = np.asarray(r1, dtype=float)
    b = np.asarray(r2, dtype=float)
    if a.shape != b.shape:
        raise ValueError("Rankings must be the same size.")
    n = len(a)
    weights = (n - a + 1) + (n - b + 1)
    numerator = np.sum((a - b) ** 2 * weights)
    denominator = n ** 4 + n ** 3 - n ** 2 - n
    if denominator == 0:
        return 1.0
    return float(1 - (6 * numerator) / denominator)
