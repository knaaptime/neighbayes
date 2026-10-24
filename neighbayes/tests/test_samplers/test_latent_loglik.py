"""The latent models' pointwise log-likelihood, each latent value integrated out.

``p(y_i | θ, η_{-i}) = ∫ p(y_i | η_i) p(η_i | η_{-i}, θ) dη_i`` must match a
brute-force integral on a fine grid, and a fitted latent model must store it
for ``az.loo`` on either backend.
"""

from __future__ import annotations

import arviz as az
import numpy as np
import pytest
import scipy.sparse as sp
from scipy.special import gammaln

import neighbayes.models as M
from neighbayes import dgp
from neighbayes.dgp.utils import rook_grid_weights
from neighbayes.samplers._utils._latent_loglik import (
    LOGIT,
    NEGBIN,
    POISSON,
    TRUNCATED,
    UNOBSERVED,
    integrated_loglik,
)
from neighbayes.samplers.negbin_latent._core import filter_columns

W = sp.csr_matrix(rook_grid_weights(4)[0])
N = W.shape[0]
GRID = np.linspace(-60.0, 25.0, 1_000_001)


def _brute(fam, eta, c, par, alpha, y, i, kind=POISSON):
    """The integral on a fine grid; the logit's threshold at a grid node."""
    A = (sp.eye(N) - par * W).toarray()
    E = (A @ eta - c)[:, None] + A[:, i][:, None] * (GRID[None, :] - eta[i])
    if fam == NEGBIN:
        logf = (alpha * E - alpha * np.exp(E)).sum(0)
        obs = y[i] * GRID - np.exp(GRID) - gammaln(y[i] + 1)
        if kind == TRUNCATED:
            obs -= np.log(-np.expm1(-np.exp(GRID)))
    else:
        a = np.abs(E)
        logf = (-a - 2 * np.log1p(np.exp(-a))).sum(0)
        obs = np.where((GRID > 0) == (y[i] > 0), 0.0, -np.inf)

    def log_trapz(v):
        top = v[np.isfinite(v)].max()
        return top + np.log(np.trapezoid(np.exp(v - top), GRID))

    return log_trapz(logf + obs) - log_trapz(logf)


@pytest.mark.parametrize(
    "fam, kind",
    [(NEGBIN, POISSON), (NEGBIN, TRUNCATED), (LOGIT, POISSON)],
    ids=["negbin", "negbin-truncated", "logit"],
)
@pytest.mark.parametrize("par, alpha", [(0.7, 0.3), (-0.4, 5.0)])
def test_matches_the_integral(fam, kind, par, alpha):
    rng = np.random.default_rng(3)
    eta = rng.normal(1.0 if fam == NEGBIN else 0.0, 2.0, N)
    c = rng.normal(0.0, 1.0, N)
    if fam == NEGBIN:
        y = rng.poisson(np.exp(np.clip(eta, -5, 3.7))).astype(float)
        if kind == TRUNCATED:
            y = np.maximum(y, 1.0)
    else:
        y = (rng.random(N) < 0.5).astype(float)
    kinds = np.full(N, kind, dtype=np.int8)
    kinds[0] = UNOBSERVED
    e = ((sp.eye(N) - par * W) @ eta - c)[None, :]
    indptr, rows, ones, wvals = filter_columns(W)
    got = integrated_loglik(
        eta[None, :], e, y, indptr, rows, ones, wvals,
        np.array([par]), np.array([alpha]), fam, gammaln(y + 1), kinds,
    )[0]  # fmt: skip
    want = np.array([_brute(fam, eta, c, par, alpha, y, i, kind) for i in range(N)])
    if fam == NEGBIN:
        want[0] = 0.0  # an unobserved count contributes nothing
    np.testing.assert_allclose(got, want, atol=2e-4)


@pytest.mark.parametrize(
    "backend", ["numpy", pytest.param("jax", marks=pytest.mark.requires_jax)]
)
@pytest.mark.parametrize(
    "cls, simulate, kw",
    [
        (M.SARNegBin, dgp.simulate_sar_negbin, {"latent": True}),
        (M.SEMLogit, dgp.simulate_sem_logit, {}),
    ],
    ids=["SARNegBin latent", "SEMLogit"],
)
def test_a_fit_stores_it_for_loo(cls, simulate, kw, backend):
    d = simulate(n_side=8, seed=2, **kw)
    m = cls(y=d["y"], X=d["X"], W=d["W_sparse"], **kw)
    idata = m.fit(
        draws=200, tune=200, chains=2, progressbar=False, random_seed=0,
        backend=backend, n_jobs=1, idata_kwargs={"log_likelihood": True},
    )  # fmt: skip
    ll = idata.log_likelihood["obs"].values
    assert ll.shape == (2, 200, 64) and np.all(np.isfinite(ll)) and np.all(ll <= 0)
    loo = az.loo(idata, pointwise=True)
    assert np.mean(np.asarray(loo.pareto_k) > 0.7) < 0.05
