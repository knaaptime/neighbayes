"""Unit tests for JAX-accelerated reduced-form SAR-NB Gibbs sampler.

Tests the slice+Krylov architecture in
``neighbayes.samplers.negbin_reduced._jax`` against the NumPy
factorize path for correctness and shape.

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

from neighbayes.samplers.negbin_reduced import (
    ReducedGibbsPriors,
    ReducedGibbsState,
)
from neighbayes.samplers.negbin_reduced._jax import (
    _build_krylov_basis_jax,
    _build_sparse_ctx,
    _eval_U_from_basis_jax,
    _make_sparse_solvers,
    _rho_log_density_marginal_jax,
    _slice_sample_rho_jax,
    run_chains_jax_reduced,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_problem(n: int = 25, k: int = 3, seed: int = 42):
    """Build a small test problem."""
    rng = np.random.default_rng(seed)
    X = np.column_stack([np.ones(n)] + [rng.standard_normal(n) for _ in range(k - 1)])
    W = sp.random(n, n, density=0.2, format="csr", random_state=seed)
    W = W + W.T
    # Row-standardize
    row_sums = np.array(W.sum(axis=1)).flatten()
    row_sums[row_sums == 0] = 1.0
    W = sp.diags(1.0 / row_sums) @ W
    y = rng.poisson(5, size=n).astype(np.float64)
    return y, X, W


def _krylov_basis(W, X_jax, rho_c, n, k, degree):
    """Build a Krylov basis via the sparse sparsax path (W never densified)."""
    ctx = _build_sparse_ctx(sp.csr_matrix(W), n)
    solve, matvec_W = _make_sparse_solvers(ctx)
    return _build_krylov_basis_jax(
        lambda rhs: solve(rho_c, rhs), X_jax, matvec_W, n, k, degree
    )


# ---------------------------------------------------------------------------
# Krylov basis tests
# ---------------------------------------------------------------------------


class TestKrylovBasis:
    """Tests for _build_krylov_basis_jax and _eval_U_from_basis_jax."""

    @pytest.fixture
    def problem(self):
        return _make_problem(n=25, k=3)

    def test_krylov_basis_shape(self, problem):
        """V_stack should have shape (m+1, n, k)."""
        y, X, W = problem
        n, k = X.shape
        degree = 4
        X_jax = jnp.asarray(X)
        rho_c = jnp.float64(0.3)

        V_stack = _krylov_basis(W, X_jax, rho_c, n, k, degree)
        assert V_stack.shape == (degree + 1, n, k)

    def test_krylov_basis_v0_matches_solve(self, problem):
        """V_0 should equal (I - ρW)⁻¹ X."""
        y, X, W = problem
        n, k = X.shape
        X_jax = jnp.asarray(X)
        W_jax = jnp.asarray(W.toarray())
        rho_c = jnp.float64(0.3)

        V_stack = _krylov_basis(W, X_jax, rho_c, n, k, 8)
        I_n = jnp.eye(n)
        U_exact = jnp.linalg.solve(I_n - rho_c * W_jax, X_jax)

        np.testing.assert_allclose(
            np.asarray(V_stack[0]), np.asarray(U_exact), atol=1e-10
        )

    def test_horner_eval_at_centre(self, problem):
        """Horner eval at Δρ=0 should return V_0."""
        y, X, W = problem
        n, k = X.shape
        X_jax = jnp.asarray(X)
        rho_c = jnp.float64(0.3)

        V_stack = _krylov_basis(W, X_jax, rho_c, n, k, 8)
        U_eval = _eval_U_from_basis_jax(V_stack, jnp.float64(0.0))

        np.testing.assert_allclose(
            np.asarray(U_eval), np.asarray(V_stack[0]), atol=1e-10
        )

    def test_horner_eval_accuracy(self, problem):
        """Horner eval at small Δρ should match direct solve."""
        y, X, W = problem
        n, k = X.shape
        X_jax = jnp.asarray(X)
        W_jax = jnp.asarray(W.toarray())
        rho_c = jnp.float64(0.3)
        drho = jnp.float64(0.05)

        V_stack = _krylov_basis(W, X_jax, rho_c, n, k, 8)
        U_krylov = _eval_U_from_basis_jax(V_stack, drho)

        rho_new = rho_c + drho
        I_n = jnp.eye(n)
        U_exact = jnp.linalg.solve(I_n - rho_new * W_jax, X_jax)

        np.testing.assert_allclose(np.asarray(U_krylov), np.asarray(U_exact), atol=1e-6)


# ---------------------------------------------------------------------------
# ρ log-density tests
# ---------------------------------------------------------------------------


class TestRhoLogDensity:
    """Tests for _rho_log_density_marginal_jax."""

    @pytest.fixture
    def problem(self):
        return _make_problem(n=25, k=3)

    def test_log_density_finite(self, problem):
        """Log-density should be finite at a reasonable ρ."""
        y, X, W = problem
        n, k = X.shape
        X_jax = jnp.asarray(X)
        y_jax = jnp.asarray(y)
        rho_c = jnp.float64(0.3)

        priors = ReducedGibbsPriors()
        V0_inv_diag = jnp.full(k, 1.0 / priors.beta_sigma**2)
        mu0 = jnp.zeros(k)
        omega = jnp.full(n, 0.25)
        alpha = jnp.float64(1.0)

        V_stack = _krylov_basis(W, X_jax, rho_c, n, k, 8)

        log_dens = _rho_log_density_marginal_jax(
            rho_c,
            V_stack,
            rho_c,
            omega,
            y_jax,
            alpha,
            V0_inv_diag,
            mu0,
            intercept_col=0,
            krylov_dmax=0.15,
        )
        assert jnp.isfinite(log_dens)

    def test_log_density_rejects_outside_krylov_radius(self, problem):
        """Log-density should be -inf when |Δρ| > dmax."""
        y, X, W = problem
        n, k = X.shape
        X_jax = jnp.asarray(X)
        y_jax = jnp.asarray(y)
        rho_c = jnp.float64(0.3)

        priors = ReducedGibbsPriors()
        V0_inv_diag = jnp.full(k, 1.0 / priors.beta_sigma**2)
        mu0 = jnp.zeros(k)
        omega = jnp.full(n, 0.25)
        alpha = jnp.float64(1.0)

        V_stack = _krylov_basis(W, X_jax, rho_c, n, k, 8)

        # Test with Δρ = 0.5 > dmax = 0.15
        rho_far = jnp.float64(0.8)
        log_dens = _rho_log_density_marginal_jax(
            rho_far,
            V_stack,
            rho_c,
            omega,
            y_jax,
            alpha,
            V0_inv_diag,
            mu0,
            intercept_col=0,
            krylov_dmax=0.15,
        )
        assert log_dens == -jnp.inf


# ---------------------------------------------------------------------------
# Slice sampler tests
# ---------------------------------------------------------------------------


class TestSliceSampleRho:
    """Tests for _slice_sample_rho_jax."""

    @pytest.fixture
    def problem(self):
        return _make_problem(n=25, k=3)

    def test_slice_returns_valid_rho(self, problem):
        """Slice sampler should return ρ in [rho_lower, rho_upper]."""
        y, X, W = problem
        n, k = X.shape
        X_jax = jnp.asarray(X)
        y_jax = jnp.asarray(y)
        rho_c = jnp.float64(0.3)

        priors = ReducedGibbsPriors()
        V0_inv_diag = jnp.full(k, 1.0 / priors.beta_sigma**2)
        mu0 = jnp.zeros(k)
        omega = jnp.full(n, 0.25)
        alpha = jnp.float64(1.0)

        V_stack = _krylov_basis(W, X_jax, rho_c, n, k, 8)

        key = jax.random.PRNGKey(0)
        rho_new = _slice_sample_rho_jax(
            rho_current=rho_c,
            V_stack=V_stack,
            rho_basis=rho_c,
            omega=omega,
            y_jax=y_jax,
            alpha=alpha,
            V0_inv_diag=V0_inv_diag,
            mu0=mu0,
            intercept_col=0,
            rho_lower=jnp.float64(-0.999),
            rho_upper=jnp.float64(0.999),
            krylov_dmax=jnp.float64(0.15),
            slice_width=jnp.float64(0.2),
            key=key,
        )

        assert float(rho_new) > -0.999
        assert float(rho_new) < 0.999
        assert jnp.isfinite(rho_new)


# ---------------------------------------------------------------------------
# Full chain runner tests
# ---------------------------------------------------------------------------


class TestRunChainsJaxReduced:
    """Tests for run_chains_jax_reduced."""

    def test_single_chain_shapes(self):
        """Output shapes should match draws × parameters."""
        y, X, W = _make_problem(n=25, k=3)
        priors = ReducedGibbsPriors()
        inits = [
            ReducedGibbsState(
                beta=np.zeros(3), rho=0.1, alpha=1.0, omega=0.25 * np.ones(25)
            )
        ]

        results = run_chains_jax_reduced(
            y=y,
            X=X,
            W_sparse=W,
            priors=priors,
            inits=inits,
            draws=50,
            tune=50,
            krylov_degree=4,
            progressbar=False,
        )

        assert len(results) == 1
        assert results[0]["rho"].shape == (50,)
        assert results[0]["beta"].shape == (50, 3)
        assert results[0]["alpha"].shape == (50,)
        assert results[0]["log_lik"].shape == (50, 25)

    def test_output_is_finite(self):
        """All output values should be finite."""
        y, X, W = _make_problem(n=25, k=3)
        priors = ReducedGibbsPriors()
        inits = [
            ReducedGibbsState(
                beta=np.zeros(3), rho=0.1, alpha=1.0, omega=0.25 * np.ones(25)
            )
        ]

        results = run_chains_jax_reduced(
            y=y,
            X=X,
            W_sparse=W,
            priors=priors,
            inits=inits,
            draws=50,
            tune=50,
            krylov_degree=4,
            progressbar=False,
        )

        assert np.all(np.isfinite(results[0]["rho"]))
        assert np.all(np.isfinite(results[0]["beta"]))
        assert np.all(np.isfinite(results[0]["alpha"]))
        assert np.all(np.isfinite(results[0]["log_lik"]))

    def test_alpha_positive(self):
        """α should always be positive."""
        y, X, W = _make_problem(n=25, k=3)
        priors = ReducedGibbsPriors()
        inits = [
            ReducedGibbsState(
                beta=np.zeros(3), rho=0.1, alpha=1.0, omega=0.25 * np.ones(25)
            )
        ]

        results = run_chains_jax_reduced(
            y=y,
            X=X,
            W_sparse=W,
            priors=priors,
            inits=inits,
            draws=50,
            tune=50,
            krylov_degree=4,
            progressbar=False,
        )

        assert np.all(results[0]["alpha"] > 0)

    def test_thinning(self):
        """Thinning should reduce the number of stored draws."""
        y, X, W = _make_problem(n=25, k=3)
        priors = ReducedGibbsPriors()
        inits = [
            ReducedGibbsState(
                beta=np.zeros(3), rho=0.1, alpha=1.0, omega=0.25 * np.ones(25)
            )
        ]

        results = run_chains_jax_reduced(
            y=y,
            X=X,
            W_sparse=W,
            priors=priors,
            inits=inits,
            draws=100,
            tune=50,
            thin=2,
            krylov_degree=4,
            progressbar=False,
        )

        assert results[0]["rho"].shape == (50,)  # 100/2 = 50
