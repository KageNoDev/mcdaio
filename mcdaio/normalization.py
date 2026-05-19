from __future__ import annotations
import numpy as np

def minmax_normalize(matrix: np.ndarray, types: np.ndarray) -> np.ndarray:

    mat = np.asarray(matrix, dtype=float)
    types = np.asarray(types, dtype=int)
    norm = np.zeros_like(mat)

    for j in range(mat.shape[1]):
        col = mat[:, j]
        c_min, c_max = float(np.min(col)), float(np.max(col))
        if np.isclose(c_max, c_min):
            norm[:, j] = 0.0
            continue
        if types[j] == 0:
            norm[:, j] = (c_max - col) / (c_max - c_min)
        else:
            norm[:, j] = (col - c_min) / (c_max - c_min)
    return norm
