"""Tests for the reduced-form SAR-logit Pólya-Gamma Gibbs sampler.

Covers the NumPy chain runner (``binary_reduced._core.run_chain``) and its
NumPy↔JAX parity via the model API — both backends fit the *same* reduced-form
model (η = (I − ρW)⁻¹Xβ, Bernoulli), so their posteriors should agree within
Monte-Carlo noise.
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp

from neighbayes.models.priors import BinaryGibbsPriors
from neighbayes.samplers._utils._slice import SliceWidthState
from neighbayes.samplers.binary_reduced._core import (
    ReducedBinaryGibbsState,
    _logit_loglik_pointwise,
    run_chain,
)
from neighbayes.samplers.negbin_reduced._core import (
    ReducedGibbsCache,
    _make_cholmod_pattern,
)
from neighbayes.tests.helpers import W_to_graph, make_rook_W


def _sar_logit_data(side: int = 12, rho: float = 0.5, seed: int = 3):
    """Simulate reduced-form SAR-logit data on a rook lattice."""
    W_dense = make_rook_W(side)
    rs = W_dense.sum(1, keepdims=True)
    rs[rs == 0] = 1
    W = sp.csr_matrix(W_dense / rs).astype(np.float64)
    n = W.shape[0]
    rng = np.random.default_rng(seed)
    beta = np.array([0.2, 1.0])
    X = np.column_stack([np.ones(n), rng.normal(size=n)])
    eta = sp.linalg.spsolve((sp.eye(n) - rho * W).tocsc(), X @ beta)
    y = (rng.uniform(size=n) < 1.0 / (1.0 + np.exp(-eta))).astype(np.float64)
    return y, X, W, beta, rho


def _cache(W, *, slice_width=0.4):
    W_csc = W.tocsc()
    W_sym, WtW, pattern = _make_cholmod_pattern(W_csc, W.shape[0])
    return ReducedGibbsCache(
        W_sparse=W.tocsr(),
        W_csc=W_csc,
        rho_lower=-0.99,
        rho_upper=0.99,
        rho_adaptive_width=True,
        rho_slice_width_state=SliceWidthState(w=slice_width),
        krylov_degree=12,
        krylov_dmax=0.4,
        cholmod_pattern=pattern,
        W_sym=W_sym,
        WtW=WtW,
        W_eig_max=1.0,
        W_eig_min=-1.0,
        n_rho_omega_cycles=1,
    )


def _run(y, X, W, seed, draws=1200, tune=800):
    priors = BinaryGibbsPriors(
        beta_mu=0.0, beta_sigma=10.0, rho_lower=-0.99, rho_upper=0.99
    )
    n, k = X.shape
    r = np.random.default_rng(seed)
    init = ReducedBinaryGibbsState(
        beta=r.normal(0, 0.2, k),
        rho=float(r.uniform(-0.1, 0.1)),
        omega=0.25 * np.ones(n),
    )
    return run_chain(
        y=y,
        X=X,
        W_sparse=W.tocsr(),
        priors=priors,
        cache=_cache(W),
        init=init,
        draws=draws,
        tune=tune,
        rng=r,
    )


class TestReducedLogitNumpy:
    def test_shapes_and_finite(self):
        y, X, W, _, _ = _sar_logit_data(side=8)
        res = _run(y, X, W, seed=0, draws=200, tune=100)
        n, k = X.shape
        assert res["rho"].shape == (200,)
        assert res["beta"].shape == (200, k)
        assert res["log_lik"].shape == (200, n)
        assert np.all(np.isfinite(res["rho"]))
        assert np.all(np.isfinite(res["beta"]))
        assert np.all(res["rho"] > -1.0) and np.all(res["rho"] < 1.0)

    def test_recovers_parameters(self):
        # Binary reduced-form ρ attenuates strongly toward 0 at small n (the
        # model warns n≥900), so assert the sampler detects clear positive
        # spatial dependence and recovers β, not a tight ρ band.
        y, X, W, beta_t, rho_t = _sar_logit_data(side=14, rho=0.5, seed=3)
        res = [_run(y, X, W, seed=s) for s in range(3)]
        rho = np.concatenate([r["rho"] for r in res])
        beta = np.vstack([r["beta"] for r in res])
        assert 0.1 < rho.mean() < rho_t + 0.2  # positive, not over-shooting
        assert np.allclose(beta.mean(0), beta_t, atol=0.4)

    def test_loglik_pointwise_matches_formula(self):
        y = np.array([0.0, 1.0, 1.0])
        eta = np.array([-0.5, 0.3, 2.0])
        expected = y * eta - np.logaddexp(0.0, eta)
        np.testing.assert_allclose(_logit_loglik_pointwise(y, eta), expected)


@pytest.mark.requires_jax
class TestReducedLogitBackendParity:
    """The model's NumPy and JAX backends fit the same reduced-form model."""

    def test_numpy_jax_agree(self):
        import neighbayes  # configure CPU devices before jax import
        from neighbayes.models import SARLogit

        _ = neighbayes
        y, X, W, beta_t, rho_t = _sar_logit_data(side=12, rho=0.5, seed=5)
        graph = W_to_graph(W.toarray())

        common = dict(draws=1000, tune=800, chains=4, random_seed=0, progressbar=False)
        inp = SARLogit(y=y, X=X, W=graph).fit(backend="numpy", n_jobs=1, **common)
        ij = SARLogit(y=y, X=X, W=graph).fit(backend="jax", **common)

        rho_np = float(inp.posterior["rho"].mean())
        rho_jx = float(ij.posterior["rho"].mean())
        beta_np = inp.posterior["beta"].mean(("chain", "draw")).values
        beta_jx = ij.posterior["beta"].mean(("chain", "draw")).values

        # The real correctness signal: both backends fit the SAME model, so
        # (different RNG streams notwithstanding) their posteriors agree.
        assert abs(rho_np - rho_jx) < 0.08
        assert np.allclose(beta_np, beta_jx, atol=0.15)
        # Both detect clear positive spatial dependence (ρ attenuates at small n).
        assert rho_np > 0.1 and rho_jx > 0.1
        assert set(ij.posterior.data_vars) == {"beta", "rho"}
