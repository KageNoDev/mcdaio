from __future__ import annotations
import numpy as np
import pytest
from mcdaio import (
    ARAS,
    MABAC,
    METHOD_REGISTRY,
    TOPSIS,
    VIKOR,
    minmax_normalize,
    sensitivity_analysis,
    spearman_corr,
    weighted_rank_corr,
)


@pytest.fixture
def sample_problem():
    matrix = np.array(
        [
            [3, 4, 2],
            [2, 5, 3],
            [4, 3, 1],
        ],
        dtype=float,
    )
    weights = np.array([0.3, 0.4, 0.3])
    types = np.array([1, 1, 0])
    return matrix, weights, types

@pytest.mark.parametrize("method_cls", [MABAC, TOPSIS, VIKOR, ARAS])
def test_method_returns_scores_and_ranks(method_cls, sample_problem):
    matrix, weights, types = sample_problem
    scores, ranks = method_cls(matrix, weights, types).run()

    assert scores.shape == (3,)
    assert ranks.shape == (3,)
    assert set(ranks.tolist()) == {1, 2, 3}


def test_method_registry_contains_all_methods():
    assert set(METHOD_REGISTRY) == {"MABAC", "TOPSIS", "VIKOR", "ARAS"}
    assert METHOD_REGISTRY["MABAC"] is MABAC
    assert METHOD_REGISTRY["TOPSIS"] is TOPSIS
    assert METHOD_REGISTRY["VIKOR"] is VIKOR
    assert METHOD_REGISTRY["ARAS"] is ARAS

def test_minmax_normalize_range(sample_problem):
    matrix, _, types = sample_problem
    norm = minmax_normalize(matrix, types)

    assert norm.shape == matrix.shape
    assert np.all(norm >= 0)
    assert np.all(norm <= 1)

def test_minmax_normalize_constant_column():
    matrix = np.array([[1.0, 5.0], [1.0, 6.0], [1.0, 7.0]])
    types = np.array([1, 1])
    norm = minmax_normalize(matrix, types)

    assert np.allclose(norm[:, 0], 0.0)

def test_aras_utility_bounded_by_one_with_cost_criteria():
    matrix = np.array(
        [
            [1800, 60, 12, 24, 75, 3, 65],
            [1500, 55, 10, 20, 65, 4, 60],
            [900, 40, 8, 18, 55, 5, 45],
            [2200, 70, 18, 30, 80, 3, 75],
            [2600, 80, 25, 36, 85, 4, 85],
            [2400, 95, 22, 34, 80, 3, 95],
            [2300, 85, 28, 32, 90, 4, 90],
            [1700, 65, 15, 28, 70, 5, 70],
            [2800, 75, 20, 30, 88, 3, 80],
            [1600, 60, 14, 26, 68, 4, 62],
        ],
        dtype=float,
    )
    weights = np.array([0.20, 0.20, 0.15, 0.10, 0.15, 0.10, 0.10])
    types = np.array([1, 1, 0, 0, 1, 0, 1])

    scores, _ = ARAS(matrix, weights, types).run()
    assert np.all(scores <= 1.0 + 1e-9), f"K_i values must be <= 1, got max={scores.max()}"
    assert np.all(scores >= 0.0)


def test_aras_pure_benefit_problem():
    matrix = np.array([[3, 4, 2], [2, 5, 3], [4, 3, 1]], dtype=float)
    weights = np.array([0.3, 0.4, 0.3])
    types = np.array([1, 1, 1])
    scores, _ = ARAS(matrix, weights, types).run()
    assert np.all(scores <= 1.0 + 1e-9)

def test_spearman_corr_bounds():
    assert spearman_corr([1, 2, 3], [1, 2, 3]) == pytest.approx(1.0)
    assert spearman_corr([1, 2, 3], [3, 2, 1]) == pytest.approx(-1.0)


def test_weighted_rank_corr_bounds():
    assert weighted_rank_corr([1, 2, 3], [1, 2, 3]) == pytest.approx(1.0)
    assert -1.0 <= weighted_rank_corr([1, 2, 3], [3, 1, 2]) <= 1.0


def test_metrics_reject_mismatched_shapes():
    with pytest.raises(ValueError):
        spearman_corr([1, 2, 3], [1, 2])
    with pytest.raises(ValueError):
        weighted_rank_corr([1, 2, 3], [1, 2])

def test_sensitivity_analysis(sample_problem):
    matrix, weights, types = sample_problem
    _, reference_ranks = MABAC(matrix, weights, types).run()

    rankings, correlations = sensitivity_analysis(
        reference_ranks, matrix, weights, types, MABAC
    )

    assert len(rankings) == matrix.shape[1]
    assert len(correlations) == matrix.shape[1]
    for ranks in rankings:
        assert ranks.shape == (matrix.shape[0],)
    for corr in correlations:
        assert -1.0 <= corr <= 1.0
