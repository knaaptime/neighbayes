"""Tests for the conditional Gaussian kernel in :mod:`neighbayes._prediction`."""

from __future__ import annotations

import arviz as az
import numpy as np
import pytest
import scipy.sparse as sp
from libpysal.graph import Graph

from neighbayes._prediction import GaussianConditional
from neighbayes.dgp import simulate_sar
from neighbayes.diagnostics.spatial_cv import _fold_elpd


@pytest.fixture(scope="module")
def grid():
    """6x6 SAR sample with its rook W, design, and design with WX."""
    gdf = simulate_sar(
        n_side=6,
        rho=0.5,
        beta=np.array([1.0, 2.0]),
        sigma=1.0,
        seed=0,
        create_gdf=True,
        geometry_type="polygon",
    )
    W = sp.csr_matrix(Graph.build_contiguity(gdf, rook=True).transform("r").sparse)
    y = gdf["y"].to_numpy(float)
    X = np.column_stack([np.ones(len(y)), gdf["X_1"].to_numpy(float)])
    XW = np.hstack([X, W @ X[:, 1:]])
    return y, W, X, XW


def _dense_conditional(W, theta, sigma, mu, y, O):
    """Goldberger form: μ_O + Σ_OS Σ_SS⁻¹ (y_S − μ_S) and its covariance."""
    n = W.shape[0]
    S = np.setdiff1d(np.arange(n), O)
    Ai = np.linalg.inv(np.eye(n) - theta * W.toarray())
    Sig = sigma**2 * Ai @ Ai.T
    K = Sig[np.ix_(O, S)] @ np.linalg.inv(Sig[np.ix_(S, S)])
    mean = mu[O] + K @ (y[S] - mu[S])
    cov = Sig[np.ix_(O, O)] - K @ Sig[np.ix_(S, O)]
    return mean, cov


# Held-out log density from the pre-kernel ``_fold_elpd`` (commit b6fc409),
# one fixed posterior per model with seed = position in this dict.
_PINNED_ELPD = {
    "OLS": -35.93431127072498,
    "SLX": -39.257240363341,
    "SAR": -22.518423697404216,
    "SDM": -24.200809180505804,
    "SEM": -22.682541766833722,
    "SDEM": -24.08205293224506,
}
_KINDS = {
    "OLS": ("iid", False, None),
    "SLX": ("iid", True, None),
    "SAR": ("lag", False, "rho"),
    "SDM": ("lag", True, "rho"),
    "SEM": ("error", False, "lam"),
    "SDEM": ("error", True, "lam"),
}


class _Refit:
    def __init__(self, posterior):
        self.inference_data = az.from_dict({"posterior": posterior})


@pytest.mark.parametrize("name", list(_PINNED_ELPD))
def test_fold_elpd_matches_pre_kernel_pin(grid, name):
    y, W, X, XW = grid
    kind, wx, spatial = _KINDS[name]
    design = XW if wx else X
    rng = np.random.default_rng(list(_PINNED_ELPD).index(name))
    G, k = 5, design.shape[1]
    post = {
        "beta": rng.normal(0.5, 0.3, size=(1, G, k)),
        "sigma": rng.uniform(0.8, 1.2, size=(1, G)),
    }
    if spatial:
        post[spatial] = rng.uniform(0.3, 0.6, size=(1, G))
    test = np.flatnonzero(np.arange(len(y)) % 3 == 1)
    got = _fold_elpd(
        _Refit(post), y_full=y, design_full=design, W_full=W, test_idx=test, kind=kind
    )
    np.testing.assert_allclose(got, _PINNED_ELPD[name], rtol=1e-12)


def _knn_W(n_side=6, k=3):
    """Row-standardized, non-symmetric KNN weights on a lattice."""
    xy = np.array([(i, j) for i in range(n_side) for j in range(n_side)], float)
    xy += np.random.default_rng(1).uniform(-0.2, 0.2, xy.shape)
    return sp.csr_matrix(Graph.build_knn(xy, k=k).transform("r").sparse)


@pytest.mark.parametrize("which", ["rook", "knn"])
@pytest.mark.parametrize("theta", [0.0, 0.45, -0.3, 0.9])
def test_mean_and_logpdf_match_dense_goldberger(grid, which, theta):
    y, W_rook, X, _ = grid
    W = W_rook if which == "rook" else _knn_W()
    n = W.shape[0]
    O = np.array([20, 3, 4, 17, 35, 9])  # unsorted, mixed interior/boundary
    mu = X @ np.array([0.7, -1.2])
    sigma = 1.3
    cond = GaussianConditional(W, O, n)
    cond.update(theta, sigma)
    mean = cond.mean(mu, y)
    ref_mean, ref_cov = _dense_conditional(W, theta, sigma, mu, y, O)
    np.testing.assert_allclose(mean, ref_mean, rtol=1e-10, atol=1e-12)

    from scipy.stats import multivariate_normal

    ref_lp = multivariate_normal(ref_mean, ref_cov).logpdf(y[O])
    np.testing.assert_allclose(cond.logpdf(y[O], mean), ref_lp, rtol=1e-10)


def test_mean_ignores_held_out_outcomes(grid):
    y, W, X, _ = grid
    O = np.array([1, 14, 30])
    cond = GaussianConditional(W, O, len(y))
    cond.update(0.5, 1.0)
    mu = X @ np.array([1.0, 2.0])
    y2 = y.copy()
    y2[O] = 1e6
    np.testing.assert_array_equal(cond.mean(mu, y), cond.mean(mu, y2))


def test_sample_covariance_is_conditional_covariance(grid):
    y, W, X, _ = grid
    O = np.array([7, 8, 14, 21])
    theta, sigma = 0.6, 0.9
    cond = GaussianConditional(W, O, len(y))
    cond.update(theta, sigma)
    mu = X @ np.array([1.0, 2.0])
    mean = cond.mean(mu, y)
    rng = np.random.default_rng(0)
    draws = np.stack([cond.sample(mean, rng) for _ in range(20000)])
    _, ref_cov = _dense_conditional(W, theta, sigma, mu, y, O)
    np.testing.assert_allclose(draws.mean(0), mean, atol=0.03)
    np.testing.assert_allclose(np.cov(draws.T), ref_cov, atol=0.02)


def test_iid_is_the_marginal(grid):
    y, _, X, _ = grid
    O = np.array([0, 5, 11])
    cond = GaussianConditional(None, O, len(y))
    assert cond.is_iid
    cond.update(0.0, 2.0)
    mu = X @ np.array([1.0, 2.0])
    mean = cond.mean(mu, y)
    np.testing.assert_array_equal(mean, mu[O])
    from scipy.stats import norm

    np.testing.assert_allclose(
        cond.logpdf(y[O], mean), norm(mu[O], 2.0).logpdf(y[O]).sum(), rtol=1e-12
    )


@pytest.mark.parametrize(
    "O, msg",
    [
        (np.array([], dtype=int), "empty"),
        ([1, 1], "duplicates"),
        ([36], "out of range"),
    ],
)
def test_rejects_bad_oos_idx(grid, O, msg):
    _, W, _, _ = grid
    with pytest.raises(ValueError, match=msg):
        GaussianConditional(W, O, 36)
