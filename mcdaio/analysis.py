from __future__ import annotations
from typing import List, Tuple, Type
import numpy as np
from .methods import _MCDAMethod
from .metrics import spearman_corr

def sensitivity_analysis(
    reference_rank: np.ndarray,
    matrix: np.ndarray,
    weights: np.ndarray,
    types: np.ndarray,
    method_class: Type[_MCDAMethod],
) -> Tuple[List[np.ndarray], List[float]]:

    matrix = np.asarray(matrix, dtype=float)
    weights = np.asarray(weights, dtype=float)
    types = np.asarray(types, dtype=int)

    n_criteria = matrix.shape[1]
    rankings: List[np.ndarray] = []
    correlations: List[float] = []

    for j in range(n_criteria):
        reduced_matrix = np.delete(matrix, j, axis=1)
        reduced_weights = np.delete(weights, j)
        reduced_types = np.delete(types, j)

        model = method_class(reduced_matrix, reduced_weights, reduced_types)
        _, ranks = model.run()

        rankings.append(ranks)
        correlations.append(spearman_corr(reference_rank, ranks))

    return rankings, correlations
