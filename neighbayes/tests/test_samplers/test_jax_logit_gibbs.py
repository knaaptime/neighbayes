"""Unit tests for JAX-accelerated SAR/SEM-logit Gibbs sampler runners.

Tests the de-chunked ``run_chains_jax_vectorized`` and
``run_chains_jax_sem_vectorized`` functions in
``neighbayes.samplers.logit._jax`` for correctness, shape, and
concrete-int safety (no traced iteration counts through JIT).

All tests are skipped when JAX is not installed.
"""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest
import scipy.sparse as sp

# Skip entire module if JAX is not available
pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("jax") is None,
    reason="JAX not installed",
)

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from neighbayes._logdet import make_logdet_jax_fn
from neighbayes.samplers.logit._core import (
    JAXLogitGibbsState,
    JAXSEMLogitGibbsState,
    LogitREGibbsPriors,
    SEMLogitGibbsPriors,
)
from neighbayes.samplers.logit._jax import (
    run_chains_jax_sem_vectorized,
    run_chains_jax_vectorized,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_small_problem(n: int = 20, k: int = 3, rho: float = 0.3, seed: int = 42):
    """Create a small SAR-logit test problem with known parameters."""
    rng = np.random.default_rng(seed)
    X = np.column_stack([np.ones(n), rng.standard_normal((n, k - 1))])
    beta = np.array([0.5, -1.0, 0.8])[:k]

    # Row-standardized ring-lattice W
    W_dense = np.zeros((n, n))
    for i in range(n):
        W_dense[i, (i + 1) % n] = 1.0
        W_dense[i, (i - 1) % n] = 1.0
    row_sums = W_dense.sum(axis=1, keepdims=True)
    W_dense = W_dense / row_sums
    W_sparse = sp.csr_matrix(W_dense)

    # Generate latent field and binary response
    nu = rng.standard_normal(n)
    Xbeta = X @ beta
    A_rho_inv = sp.linalg.spsolve(sp.eye(n, format="csr") - rho * W_sparse, Xbeta + nu)
    eta = A_rho_inv
    probs = 1.0 / (1.0 + np.exp(-eta))
    y = rng.binomial(1, probs).astype(float)

    return y, X, W_sparse, W_dense, beta, rho


def _make_logdet_jax(W_sparse):
    """Create a JAX-native logdet function."""
    return make_logdet_jax_fn(
        W_sparse,
        method="eigenvalue",
        rho_min=-0.999,
        rho_max=0.999,
    )


def _make_sar_inits(y, X, W_dense, n_chains: int = 2, rho_init: float = 0.1):
    """Create per-chain initial states for SAR-logit."""
    n, k = X.shape
    inits = []
    for c in range(n_chains):
        inits.append(
            JAXLogitGibbsState(
                eta=jnp.zeros(n, dtype=jnp.float64),
                beta=jnp.zeros(k, dtype=jnp.float64),
                rho=jnp.float64(rho_init + 0.01 * c),
                omega=jnp.ones(n, dtype=jnp.float64),
                sigma=jnp.float64(1.0),
            )
        )
    return inits


def _make_sem_inits(y, X, W_dense, n_chains: int = 2, lam_init: float = 0.1):
    """Create per-chain initial states for SEM-logit."""
    n, k = X.shape
    inits = []
    for c in range(n_chains):
        inits.append(
            JAXSEMLogitGibbsState(
                eta=jnp.zeros(n, dtype=jnp.float64),
                beta=jnp.zeros(k, dtype=jnp.float64),
                lam=jnp.float64(lam_init + 0.01 * c),
                omega=jnp.ones(n, dtype=jnp.float64),
                sigma=jnp.float64(1.0),
            )
        )
    return inits


# ---------------------------------------------------------------------------
# SAR-logit runner tests
# ---------------------------------------------------------------------------


class TestRunChainsJaxVectorized:
    """Tests for run_chains_jax_vectorized (SAR-logit)."""

    @pytest.fixture
    def problem(self):
        y, X, W_sparse, W_dense, beta, rho = _make_small_problem()
        logdet_jax = _make_logdet_jax(W_sparse)
        W_sym_dense = jnp.asarray(W_dense + W_dense.T, dtype=jnp.float64)
        WtW_dense = jnp.asarray(W_dense.T @ W_dense, dtype=jnp.float64)
        priors = LogitREGibbsPriors()
        return (
            y,
            X,
            W_sparse,
            W_dense,
            W_sym_dense,
            WtW_dense,
            logdet_jax,
            priors,
            beta,
            rho,
        )

    def test_sar_output_shapes(self, problem):
        """Output shapes should match (chains, draws) for rho and (chains, draws, k) for beta."""
        y, X, W_sparse, W_dense, W_sym_dense, WtW_dense, logdet_jax, priors, _, _ = (
            problem
        )
        n_chains = 2
        draws = 15
        tune = 10
        inits = _make_sar_inits(y, X, W_dense, n_chains=n_chains)

        results = run_chains_jax_vectorized(
            y=y,
            X=X,
            W_sparse=W_sparse,
            W_sym_dense=W_sym_dense,
            WtW_dense=WtW_dense,
            logdet_jax=logdet_jax,
            priors=priors,
            inits=inits,
            draws=draws,
            tune=tune,
            progressbar=False,
        )

        assert len(results) == n_chains
        for chain_result in results:
            assert "rho" in chain_result
            assert "beta" in chain_result
            assert chain_result["rho"].shape == (draws,)
            assert chain_result["beta"].shape == (draws, X.shape[1])

    def test_sar_rho_in_bounds(self, problem):
        """All rho draws should be within prior bounds."""
        y, X, W_sparse, W_dense, W_sym_dense, WtW_dense, logdet_jax, priors, _, _ = (
            problem
        )
        inits = _make_sar_inits(y, X, W_dense)

        results = run_chains_jax_vectorized(
            y=y,
            X=X,
            W_sparse=W_sparse,
            W_sym_dense=W_sym_dense,
            WtW_dense=WtW_dense,
            logdet_jax=logdet_jax,
            priors=priors,
            inits=inits,
            draws=20,
            tune=10,
            progressbar=False,
        )

        for chain_result in results:
            assert np.all(chain_result["rho"] >= priors.rho_lower)
            assert np.all(chain_result["rho"] <= priors.rho_upper)

    def test_sar_finite_draws(self, problem):
        """All draws should be finite (no NaN or inf)."""
        y, X, W_sparse, W_dense, W_sym_dense, WtW_dense, logdet_jax, priors, _, _ = (
            problem
        )
        inits = _make_sar_inits(y, X, W_dense)

        results = run_chains_jax_vectorized(
            y=y,
            X=X,
            W_sparse=W_sparse,
            W_sym_dense=W_sym_dense,
            WtW_dense=WtW_dense,
            logdet_jax=logdet_jax,
            priors=priors,
            inits=inits,
            draws=20,
            tune=10,
            progressbar=False,
        )

        for chain_result in results:
            assert np.all(np.isfinite(chain_result["rho"]))
            assert np.all(np.isfinite(chain_result["beta"]))

    def test_sar_distinct_chains(self, problem):
        """Different chains should produce different draws."""
        y, X, W_sparse, W_dense, W_sym_dense, WtW_dense, logdet_jax, priors, _, _ = (
            problem
        )
        inits = _make_sar_inits(y, X, W_dense, n_chains=2)

        results = run_chains_jax_vectorized(
            y=y,
            X=X,
            W_sparse=W_sparse,
            W_sym_dense=W_sym_dense,
            WtW_dense=WtW_dense,
            logdet_jax=logdet_jax,
            priors=priors,
            inits=inits,
            draws=20,
            tune=10,
            progressbar=False,
        )

        assert not np.allclose(results[0]["rho"], results[1]["rho"])

    def test_sar_single_chain(self, problem):
        """Should work with a single chain."""
        y, X, W_sparse, W_dense, W_sym_dense, WtW_dense, logdet_jax, priors, _, _ = (
            problem
        )
        inits = _make_sar_inits(y, X, W_dense, n_chains=1)

        results = run_chains_jax_vectorized(
            y=y,
            X=X,
            W_sparse=W_sparse,
            W_sym_dense=W_sym_dense,
            WtW_dense=WtW_dense,
            logdet_jax=logdet_jax,
            priors=priors,
            inits=inits,
            draws=10,
            tune=5,
            progressbar=False,
        )

        assert len(results) == 1
        assert results[0]["rho"].shape == (10,)

    def test_sar_thin(self, problem):
        """Thinning should reduce the number of stored draws."""
        y, X, W_sparse, W_dense, W_sym_dense, WtW_dense, logdet_jax, priors, _, _ = (
            problem
        )
        inits = _make_sar_inits(y, X, W_dense)

        results = run_chains_jax_vectorized(
            y=y,
            X=X,
            W_sparse=W_sparse,
            W_sym_dense=W_sym_dense,
            WtW_dense=WtW_dense,
            logdet_jax=logdet_jax,
            priors=priors,
            inits=inits,
            draws=20,
            tune=10,
            thin=2,
            progressbar=False,
        )

        assert results[0]["rho"].shape == (10,)  # 20 / 2

    def test_sar_tune_zero(self, problem):
        """Should work with tune=0 (no warmup)."""
        y, X, W_sparse, W_dense, W_sym_dense, WtW_dense, logdet_jax, priors, _, _ = (
            problem
        )
        inits = _make_sar_inits(y, X, W_dense)

        results = run_chains_jax_vectorized(
            y=y,
            X=X,
            W_sparse=W_sparse,
            W_sym_dense=W_sym_dense,
            WtW_dense=WtW_dense,
            logdet_jax=logdet_jax,
            priors=priors,
            inits=inits,
            draws=10,
            tune=0,
            progressbar=False,
        )

        assert results[0]["rho"].shape == (10,)

    def test_sar_accept_rate_in_result(self, problem):
        """Results should contain mh_accept_rate field."""
        y, X, W_sparse, W_dense, W_sym_dense, WtW_dense, logdet_jax, priors, _, _ = (
            problem
        )
        inits = _make_sar_inits(y, X, W_dense)

        results = run_chains_jax_vectorized(
            y=y,
            X=X,
            W_sparse=W_sparse,
            W_sym_dense=W_sym_dense,
            WtW_dense=WtW_dense,
            logdet_jax=logdet_jax,
            priors=priors,
            inits=inits,
            draws=20,
            tune=10,
            progressbar=False,
        )

        for chain_result in results:
            assert "mh_accept_rate" in chain_result
            assert 0.0 <= chain_result["mh_accept_rate"] <= 1.0


# ---------------------------------------------------------------------------
# SEM-logit runner tests
# ---------------------------------------------------------------------------


class TestRunChainsJaxSemVectorized:
    """Tests for run_chains_jax_sem_vectorized (SEM-logit)."""

    @pytest.fixture
    def problem(self):
        y, X, W_sparse, W_dense, beta, rho = _make_small_problem()
        logdet_jax = _make_logdet_jax(W_sparse)
        W_sym_dense = jnp.asarray(W_dense + W_dense.T, dtype=jnp.float64)
        WtW_dense = jnp.asarray(W_dense.T @ W_dense, dtype=jnp.float64)
        priors = SEMLogitGibbsPriors()
        return (
            y,
            X,
            W_sparse,
            W_dense,
            W_sym_dense,
            WtW_dense,
            logdet_jax,
            priors,
            beta,
            rho,
        )

    def test_sem_output_shapes(self, problem):
        """Output shapes should match (chains, draws) for lam and (chains, draws, k) for beta."""
        y, X, W_sparse, W_dense, W_sym_dense, WtW_dense, logdet_jax, priors, _, _ = (
            problem
        )
        n_chains = 2
        draws = 15
        tune = 10
        inits = _make_sem_inits(y, X, W_dense, n_chains=n_chains)

        results = run_chains_jax_sem_vectorized(
            y=y,
            X=X,
            W_sparse=W_sparse,
            W_sym_dense=W_sym_dense,
            WtW_dense=WtW_dense,
            logdet_jax=logdet_jax,
            priors=priors,
            inits=inits,
            draws=draws,
            tune=tune,
            progressbar=False,
        )

        assert len(results) == n_chains
        for chain_result in results:
            assert "lam" in chain_result
            assert "beta" in chain_result
            assert chain_result["lam"].shape == (draws,)
            assert chain_result["beta"].shape == (draws, X.shape[1])

    def test_sem_lam_in_bounds(self, problem):
        """All lam draws should be within prior bounds."""
        y, X, W_sparse, W_dense, W_sym_dense, WtW_dense, logdet_jax, priors, _, _ = (
            problem
        )
        inits = _make_sem_inits(y, X, W_dense)

        results = run_chains_jax_sem_vectorized(
            y=y,
            X=X,
            W_sparse=W_sparse,
            W_sym_dense=W_sym_dense,
            WtW_dense=WtW_dense,
            logdet_jax=logdet_jax,
            priors=priors,
            inits=inits,
            draws=20,
            tune=10,
            progressbar=False,
        )

        for chain_result in results:
            assert np.all(chain_result["lam"] >= priors.lam_lower)
            assert np.all(chain_result["lam"] <= priors.lam_upper)

    def test_sem_finite_draws(self, problem):
        """All draws should be finite (no NaN or inf)."""
        y, X, W_sparse, W_dense, W_sym_dense, WtW_dense, logdet_jax, priors, _, _ = (
            problem
        )
        inits = _make_sem_inits(y, X, W_dense)

        results = run_chains_jax_sem_vectorized(
            y=y,
            X=X,
            W_sparse=W_sparse,
            W_sym_dense=W_sym_dense,
            WtW_dense=WtW_dense,
            logdet_jax=logdet_jax,
            priors=priors,
            inits=inits,
            draws=20,
            tune=10,
            progressbar=False,
        )

        for chain_result in results:
            assert np.all(np.isfinite(chain_result["lam"]))
            assert np.all(np.isfinite(chain_result["beta"]))

    def test_sem_distinct_chains(self, problem):
        """Different chains should produce different draws."""
        y, X, W_sparse, W_dense, W_sym_dense, WtW_dense, logdet_jax, priors, _, _ = (
            problem
        )
        inits = _make_sem_inits(y, X, W_dense, n_chains=2)

        results = run_chains_jax_sem_vectorized(
            y=y,
            X=X,
            W_sparse=W_sparse,
            W_sym_dense=W_sym_dense,
            WtW_dense=WtW_dense,
            logdet_jax=logdet_jax,
            priors=priors,
            inits=inits,
            draws=20,
            tune=10,
            progressbar=False,
        )

        assert not np.allclose(results[0]["lam"], results[1]["lam"])

    def test_sem_single_chain(self, problem):
        """Should work with a single chain."""
        y, X, W_sparse, W_dense, W_sym_dense, WtW_dense, logdet_jax, priors, _, _ = (
            problem
        )
        inits = _make_sem_inits(y, X, W_dense, n_chains=1)

        results = run_chains_jax_sem_vectorized(
            y=y,
            X=X,
            W_sparse=W_sparse,
            W_sym_dense=W_sym_dense,
            WtW_dense=WtW_dense,
            logdet_jax=logdet_jax,
            priors=priors,
            inits=inits,
            draws=10,
            tune=5,
            progressbar=False,
        )

        assert len(results) == 1
        assert results[0]["lam"].shape == (10,)

    def test_sem_tune_zero(self, problem):
        """Should work with tune=0 (no warmup)."""
        y, X, W_sparse, W_dense, W_sym_dense, WtW_dense, logdet_jax, priors, _, _ = (
            problem
        )
        inits = _make_sem_inits(y, X, W_dense)

        results = run_chains_jax_sem_vectorized(
            y=y,
            X=X,
            W_sparse=W_sparse,
            W_sym_dense=W_sym_dense,
            WtW_dense=WtW_dense,
            logdet_jax=logdet_jax,
            priors=priors,
            inits=inits,
            draws=10,
            tune=0,
            progressbar=False,
        )

        assert results[0]["lam"].shape == (10,)

    def test_sem_accept_rate_in_result(self, problem):
        """Results should contain mh_accept_rate field."""
        y, X, W_sparse, W_dense, W_sym_dense, WtW_dense, logdet_jax, priors, _, _ = (
            problem
        )
        inits = _make_sem_inits(y, X, W_dense)

        results = run_chains_jax_sem_vectorized(
            y=y,
            X=X,
            W_sparse=W_sparse,
            W_sym_dense=W_sym_dense,
            WtW_dense=WtW_dense,
            logdet_jax=logdet_jax,
            priors=priors,
            inits=inits,
            draws=20,
            tune=10,
            progressbar=False,
        )

        for chain_result in results:
            assert "mh_accept_rate" in chain_result
            assert 0.0 <= chain_result["mh_accept_rate"] <= 1.0
