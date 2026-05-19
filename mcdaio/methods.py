from __future__ import annotations
from typing import Tuple
import numpy as np
from scipy.stats import rankdata
from .normalization import minmax_normalize

EPS = 1e-12

class _MCDAMethod:
    def __init__(
        self,
        matrix: np.ndarray,
        weights: np.ndarray,
        types: np.ndarray,
    ) -> None:
        self.matrix = np.asarray(matrix, dtype=float)
        self.weights = np.asarray(weights, dtype=float)
        self.types = np.asarray(types, dtype=int)

    def run(self) -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError

class MABAC(_MCDAMethod):

    def run(self) -> Tuple[np.ndarray, np.ndarray]:
        normalized = minmax_normalize(self.matrix, self.types)
        weighted = normalized * self.weights
        border = np.prod(weighted + EPS, axis=0) ** (1.0 / weighted.shape[0])
        distance = weighted - border
        scores = np.sum(distance, axis=1)
        ranks = rankdata(-scores, method="ordinal")
        return scores, ranks

class TOPSIS(_MCDAMethod):

    def run(self) -> Tuple[np.ndarray, np.ndarray]:
        norms = np.sqrt((self.matrix ** 2).sum(axis=0))
        norms[norms == 0] = EPS
        normalized = self.matrix / norms
        weighted = normalized * self.weights

        ideal_pos = (
            np.max(weighted, axis=0) * self.types
            + np.min(weighted, axis=0) * (1 - self.types)
        )
        ideal_neg = (
            np.min(weighted, axis=0) * self.types
            + np.max(weighted, axis=0) * (1 - self.types)
        )
        d_pos = np.sqrt(((weighted - ideal_pos) ** 2).sum(axis=1))
        d_neg = np.sqrt(((weighted - ideal_neg) ** 2).sum(axis=1))
        scores = d_neg / (d_pos + d_neg + EPS)
        ranks = rankdata(-scores, method="ordinal")
        return scores, ranks

class VIKOR(_MCDAMethod):

    def __init__(
        self,
        matrix: np.ndarray,
        weights: np.ndarray,
        types: np.ndarray,
        v: float = 0.5,
    ) -> None:
        super().__init__(matrix, weights, types)
        self.v = float(v)

    def run(self) -> Tuple[np.ndarray, np.ndarray]:
        f_star = (
            np.max(self.matrix, axis=0) * self.types
            + np.min(self.matrix, axis=0) * (1 - self.types)
        )
        f_minus = (
            np.min(self.matrix, axis=0) * self.types
            + np.max(self.matrix, axis=0) * (1 - self.types)
        )
        denom = f_star - f_minus
        denom[denom == 0] = EPS

        weighted_dev = self.weights * (f_star - self.matrix) / denom
        s = np.sum(weighted_dev, axis=1)
        r = np.max(weighted_dev, axis=1)

        s_star, s_minus = np.min(s), np.max(s)
        r_star, r_minus = np.min(r), np.max(r)

        q = (
            self.v * (s - s_star) / (s_minus - s_star + EPS)
            + (1 - self.v) * (r - r_star) / (r_minus - r_star + EPS)
        )
        ranks = rankdata(q, method="ordinal")
        return q, ranks

class ARAS(_MCDAMethod):

    def run(self) -> Tuple[np.ndarray, np.ndarray]:
        mat = self.matrix.copy().astype(float)
        n_criteria = mat.shape[1]

        best = np.array(
            [
                np.max(mat[:, j]) if self.types[j] == 1 else np.min(mat[:, j])
                for j in range(n_criteria)
            ]
        )
        extended = np.vstack([best, mat])

        for j in range(n_criteria):
            if self.types[j] == 0:
                col = extended[:, j]
                col = np.where(col == 0, EPS, col)
                extended[:, j] = 1.0 / col

        col_sums = extended.sum(axis=0)
        col_sums[col_sums == 0] = EPS
        normalized = extended / col_sums

        weighted = normalized * self.weights
        s = weighted.sum(axis=1)

        scores = s[1:] / (s[0] + EPS)
        ranks = rankdata(-scores, method="ordinal")
        return scores, ranks

METHOD_REGISTRY = {
    "MABAC": MABAC,
    "TOPSIS": TOPSIS,
    "VIKOR": VIKOR,
    "ARAS": ARAS,
}
