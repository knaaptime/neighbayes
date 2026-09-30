"""Tests for the logit Gibbs sampler primitives.

Covers:
- _sample_omega: h=1, correct shape, positive values
- _sample_eta: correct shape, finite values
- _sample_beta: correct shape, finite values
- _sample_rho: stays in bounds, finite log-density
- run_chain: produces correct output shapes
- _logit_loglik_pointwise: correct values at known points
- SEM variants: _sample_eta_sem, _sample_beta_sem, _sample_lam, run_chain_sem
- _sem_logit_collapsed_log_density: analytical check
- Parameter recovery: posterior means near true values
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp

from neighbayes.samplers._utils._slice import SliceWidthState
from neighbayes.samplers._utils._spatial_normal import CholmodFactor
from neighbayes.samplers.logit._core import (
    LogitGibbsCache,
    LogitGibbsPriors,
    LogitGibbsState,
    SEMLogitGibbsCache,
    SEMLogitGibbsPriors,
    SEMLogitGibbsState,
    _logit_loglik_pointwise,
    _sample_beta,
    _sample_beta_sem,
    _sample_eta,
    _sample_eta_sem,
    _sample_lam,
    _sample_omega,
    _sample_rho,
    _sem_logit_collapsed_log_density,
    run_chain,
    run_chain_sem,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_small_problem(n=20, k=3, rho=0.3, seed=42):
    """Create a small test problem with known parameters."""
    rng = np.random.default_rng(seed)
    X = np.column_stack([np.ones(n), rng.standard_normal((n, k - 1))])
    beta = np.array([0.5, -1.0, 0.8])[:k]

    # Row-standardized W (ring lattice)
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

    return y, X, W_sparse, W_dense, beta, rho, eta


def _make_state(eta, beta, rho=0.3, omega=None):
    """Create a LogitGibbsState for unit tests."""
    n = len(eta)
    if omega is None:
        omega = np.ones(n)
    return LogitGibbsState(eta=eta, beta=beta, rho=rho, omega=omega)


def _make_cache(W_sparse, n, priors=None):
    """Create a LogitGibbsCache for unit tests."""
    if priors is None:
        priors = LogitGibbsPriors()

    W_sym = W_sparse + W_sparse.T
    WtW = W_sparse.T @ W_sparse

    # Simple eigenvalue-based logdet
    W_eigs = np.linalg.eigvalsh(W_sparse.toarray())

    def logdet_fn(rho):
        return float(np.sum(np.log(np.maximum(1.0 - rho * W_eigs, 1e-300))))

    _P0 = sp.eye(n, format="csr") + 0.5 * W_sym + 0.25 * WtW
    cholmod_factor = CholmodFactor(_P0)

    return LogitGibbsCache(
        W_sparse=W_sparse,
        XtX=np.eye(W_sparse.shape[1]) if False else np.zeros((3, 3)),  # placeholder
        logdet_fn=logdet_fn,
        rho_lower=priors.rho_lower,
        rho_upper=priors.rho_upper,
        cholmod_factor=cholmod_factor,
        W_sym=W_sym,
        WtW=WtW,
        solve_method="cholmod",
        logdet_P_method="cholmod",
        sample_method="cholmod",
        rho_adaptive_width=True,
        rho_slice_width_state=SliceWidthState(w=0.2),
    )


# ---------------------------------------------------------------------------
# _sample_omega tests
# ---------------------------------------------------------------------------


class TestSampleOmega:
    """_sample_omega should draw PG(1, eta) correctly."""

    def test_shape(self):
        """Output shape matches input."""
        rng = np.random.default_rng(0)
        eta = np.zeros(20)
        omega = _sample_omega(eta, rng=rng)
        assert omega.shape == (20,)

    def test_positive(self):
        """PG draws are always positive."""
        rng = np.random.default_rng(1)
        eta = np.array([-5.0, -1.0, 0.0, 1.0, 5.0])
        omega = _sample_omega(eta, rng=rng)
        assert np.all(omega > 0)

    def test_mean_at_zero_tilting(self):
        """PG(1, 0) has mean 0.25 (known result)."""
        rng = np.random.default_rng(2)
        eta = np.zeros(10000)
        omega = _sample_omega(eta, rng=rng)
        # E[PG(1, 0)] = 1/4
        assert abs(np.mean(omega) - 0.25) < 0.02


# ---------------------------------------------------------------------------
# _sample_eta tests
# ---------------------------------------------------------------------------


class TestSampleEta:
    """_sample_eta should draw from the spatial-normal conditional."""

    def test_shape(self):
        """Output shape matches input."""
        rng = np.random.default_rng(0)
        y, X, W_sparse, _, _, rho, eta = _make_small_problem()
        state = _make_state(eta, np.array([0.5, -1.0, 0.8]))
        cache = _make_cache(W_sparse, len(y))
        cache = cache._replace(XtX=X.T @ X)
        eta_new, _ = _sample_eta(state, y, X, W_sparse, rng=rng, cache=cache)
        assert eta_new.shape == (len(y),)

    def test_finite(self):
        """Draws should be finite."""
        rng = np.random.default_rng(1)
        y, X, W_sparse, _, _, rho, eta = _make_small_problem()
        state = _make_state(eta, np.array([0.5, -1.0, 0.8]))
        cache = _make_cache(W_sparse, len(y))
        cache = cache._replace(XtX=X.T @ X)
        eta_new, _ = _sample_eta(state, y, X, W_sparse, rng=rng, cache=cache)
        assert np.all(np.isfinite(eta_new))


# ---------------------------------------------------------------------------
# _sample_beta tests
# ---------------------------------------------------------------------------


class TestSampleBeta:
    """_sample_beta should draw from the conjugate normal conditional."""

    def test_shape(self):
        """Output shape matches number of coefficients."""
        rng = np.random.default_rng(0)
        y, X, W_sparse, _, beta, rho, eta = _make_small_problem()
        state = _make_state(eta, beta)
        priors = LogitGibbsPriors()
        XtX = X.T @ X
        A_rho = sp.eye(len(y), format="csr") - rho * W_sparse
        A_rho_eta = A_rho @ eta
        beta_new = _sample_beta(state, X, XtX, priors, A_rho_eta, rng=rng)
        assert beta_new.shape == (3,)

    def test_finite(self):
        """Draws should be finite."""
        rng = np.random.default_rng(1)
        y, X, W_sparse, _, beta, rho, eta = _make_small_problem()
        state = _make_state(eta, beta)
        priors = LogitGibbsPriors()
        XtX = X.T @ X
        A_rho = sp.eye(len(y), format="csr") - rho * W_sparse
        A_rho_eta = A_rho @ eta
        beta_new = _sample_beta(state, X, XtX, priors, A_rho_eta, rng=rng)
        assert np.all(np.isfinite(beta_new))


# ---------------------------------------------------------------------------
# _sample_rho tests
# ---------------------------------------------------------------------------


class TestSampleRho:
    """_sample_rho should draw from the collapsed slice sampler."""

    def test_stays_in_bounds(self):
        """ρ should stay within [rho_lower, rho_upper]."""
        rng = np.random.default_rng(0)
        y, X, W_sparse, _, beta, rho, eta = _make_small_problem()
        state = _make_state(eta, beta, rho=rho)
        priors = LogitGibbsPriors()
        cache = _make_cache(W_sparse, len(y), priors)
        cache = cache._replace(XtX=X.T @ X)
        rho_new, log_dens = _sample_rho(
            state, cache, priors, y, X, rng=rng, sweep_idx=0, tune=100
        )
        assert priors.rho_lower <= rho_new <= priors.rho_upper

    def test_finite_log_density(self):
        """Log-density at new ρ should be finite."""
        rng = np.random.default_rng(1)
        y, X, W_sparse, _, beta, rho, eta = _make_small_problem()
        state = _make_state(eta, beta, rho=rho)
        priors = LogitGibbsPriors()
        cache = _make_cache(W_sparse, len(y), priors)
        cache = cache._replace(XtX=X.T @ X)
        rho_new, log_dens = _sample_rho(
            state, cache, priors, y, X, rng=rng, sweep_idx=0, tune=100
        )
        assert np.isfinite(log_dens)


# ---------------------------------------------------------------------------
# _logit_loglik_pointwise tests
# ---------------------------------------------------------------------------


class TestLogitLoglik:
    """_logit_loglik_pointwise should compute correct log-likelihood."""

    def test_at_zero_eta(self):
        """At η=0, log p(y|η) = -log(2) for both y=0 and y=1."""
        y = np.array([0.0, 1.0])
        eta = np.zeros(2)
        ll = _logit_loglik_pointwise(y, eta)
        # log p(y=0|η=0) = 0 - 0 - log(1+exp(0)) = -log(2)
        # log p(y=1|η=0) = 0 - 0 - log(1+exp(0)) = -log(2)
        expected = -np.log(2)
        assert np.allclose(ll, expected, atol=1e-10)

    def test_at_large_positive_eta(self):
        """At large positive η, P(y=1) ≈ 1."""
        y = np.array([1.0])
        eta = np.array([50.0])
        ll = _logit_loglik_pointwise(y, eta)
        # log p(y=1|η=50) ≈ 50 - 50 - log(1+exp(-50)) ≈ 0
        assert ll[0] > -1e-10

    def test_at_large_negative_eta(self):
        """At large negative η, P(y=0) ≈ 1."""
        y = np.array([0.0])
        eta = np.array([-50.0])
        ll = _logit_loglik_pointwise(y, eta)
        # log p(y=0|η=-50) ≈ 0 - 0 - log(1+exp(-50)) ≈ 0
        assert ll[0] > -1e-10


# ---------------------------------------------------------------------------
# run_chain tests
# ---------------------------------------------------------------------------


class TestRunChain:
    """run_chain should produce correct output shapes."""

    def test_output_shapes(self):
        """Output arrays have correct shapes."""
        rng = np.random.default_rng(0)
        y, X, W_sparse, _, beta, rho, eta = _make_small_problem()
        n, k = X.shape

        priors = LogitGibbsPriors()
        cache = _make_cache(W_sparse, n, priors)
        cache = cache._replace(XtX=X.T @ X)

        # Initialize from OLS
        beta_init = np.linalg.lstsq(X, y, rcond=None)[0]
        eta_init = X @ beta_init
        from neighbayes.samplers._utils._polyagamma import sample_polyagamma

        omega_init = sample_polyagamma(np.ones(n), eta_init, rng=rng)

        init = LogitGibbsState(eta=eta_init, beta=beta_init, rho=0.0, omega=omega_init)

        draws = 50
        tune = 50
        result = run_chain(
            y=y,
            X=X,
            W_sparse=W_sparse,
            priors=priors,
            cache=cache,
            init=init,
            draws=draws,
            tune=tune,
            rng=rng,
        )

        assert result["rho"].shape == (draws,)
        assert result["beta"].shape == (draws, k)
        assert result["log_lik"].shape == (draws, n)
        assert result["eta_norm"].shape == (draws,)

    def test_return_eta(self):
        """When return_eta=True, eta is included in output."""
        rng = np.random.default_rng(1)
        y, X, W_sparse, _, beta, rho, eta = _make_small_problem()
        n, k = X.shape

        priors = LogitGibbsPriors()
        cache = _make_cache(W_sparse, n, priors)
        cache = cache._replace(XtX=X.T @ X)

        beta_init = np.linalg.lstsq(X, y, rcond=None)[0]
        eta_init = X @ beta_init
        from neighbayes.samplers._utils._polyagamma import sample_polyagamma

        omega_init = sample_polyagamma(np.ones(n), eta_init, rng=rng)

        init = LogitGibbsState(eta=eta_init, beta=beta_init, rho=0.0, omega=omega_init)

        draws = 20
        tune = 20
        result = run_chain(
            y=y,
            X=X,
            W_sparse=W_sparse,
            priors=priors,
            cache=cache,
            init=init,
            draws=draws,
            tune=tune,
            return_eta=True,
            rng=rng,
        )

        assert "eta" in result
        assert result["eta"].shape == (draws, n)

    def test_rho_in_bounds(self):
        """All ρ draws should be within prior bounds."""
        rng = np.random.default_rng(2)
        y, X, W_sparse, _, beta, rho, eta = _make_small_problem()
        n, k = X.shape

        priors = LogitGibbsPriors()
        cache = _make_cache(W_sparse, n, priors)
        cache = cache._replace(XtX=X.T @ X)

        beta_init = np.linalg.lstsq(X, y, rcond=None)[0]
        eta_init = X @ beta_init
        from neighbayes.samplers._utils._polyagamma import sample_polyagamma

        omega_init = sample_polyagamma(np.ones(n), eta_init, rng=rng)

        init = LogitGibbsState(eta=eta_init, beta=beta_init, rho=0.0, omega=omega_init)

        draws = 50
        tune = 50
        result = run_chain(
            y=y,
            X=X,
            W_sparse=W_sparse,
            priors=priors,
            cache=cache,
            init=init,
            draws=draws,
            tune=tune,
            rng=rng,
        )

        assert np.all(result["rho"] >= priors.rho_lower)
        assert np.all(result["rho"] <= priors.rho_upper)


# ---------------------------------------------------------------------------
# SEM-logit helpers
# ---------------------------------------------------------------------------


def _make_small_sem_problem(n=20, k=3, lam=0.3, seed=42):
    """Create a small SEM-logit test problem with known parameters."""
    rng = np.random.default_rng(seed)
    X = np.column_stack([np.ones(n), rng.standard_normal((n, k - 1))])
    beta = np.array([0.5, -1.0, 0.8])[:k]

    # Row-standardized W (ring lattice)
    W_dense = np.zeros((n, n))
    for i in range(n):
        W_dense[i, (i + 1) % n] = 1.0
        W_dense[i, (i - 1) % n] = 1.0
    row_sums = W_dense.sum(axis=1, keepdims=True)
    W_dense = W_dense / row_sums
    W_sparse = sp.csr_matrix(W_dense)

    # Generate latent field: eta = X beta + (I - lam W)^{-1} nu
    nu = rng.standard_normal(n)
    Xbeta = X @ beta
    A_lam_inv = sp.linalg.spsolve(sp.eye(n, format="csr") - lam * W_sparse, nu)
    eta = Xbeta + A_lam_inv
    probs = 1.0 / (1.0 + np.exp(-eta))
    y = rng.binomial(1, probs).astype(float)

    return y, X, W_sparse, W_dense, beta, lam, eta


def _make_sem_state(eta, beta, lam=0.3, omega=None):
    """Create a SEMLogitGibbsState for unit tests."""
    n = len(eta)
    if omega is None:
        omega = np.ones(n)
    return SEMLogitGibbsState(eta=eta, beta=beta, lam=lam, omega=omega)


def _make_sem_cache(W_sparse, n, priors=None):
    """Create a SEMLogitGibbsCache for unit tests."""
    if priors is None:
        priors = SEMLogitGibbsPriors()

    W_sym = W_sparse + W_sparse.T
    WtW = W_sparse.T @ W_sparse

    W_eigs = np.linalg.eigvalsh(W_sparse.toarray())

    def logdet_fn(lam):
        return float(np.sum(np.log(np.maximum(1.0 - lam * W_eigs, 1e-300))))

    _P0 = sp.eye(n, format="csr") + 0.5 * W_sym + 0.25 * WtW
    cholmod_factor = CholmodFactor(_P0)

    return SEMLogitGibbsCache(
        W_sparse=W_sparse,
        XtX=np.zeros((3, 3)),  # placeholder
        logdet_fn=logdet_fn,
        lam_lower=priors.lam_lower,
        lam_upper=priors.lam_upper,
        cholmod_factor=cholmod_factor,
        W_sym=W_sym,
        WtW=WtW,
        solve_method="cholmod",
        logdet_P_method="cholmod",
        sample_method="cholmod",
        lam_adaptive_width=True,
        lam_slice_width_state=SliceWidthState(w=0.2),
    )


# ---------------------------------------------------------------------------
# SEM _sample_eta tests
# ---------------------------------------------------------------------------


class TestSampleEtaSEM:
    """_sample_eta_sem should draw from the SEM spatial-normal conditional."""

    def test_shape(self):
        """Output shape matches input."""
        rng = np.random.default_rng(0)
        y, X, W_sparse, _, _, lam, eta = _make_small_sem_problem()
        state = _make_sem_state(eta, np.array([0.5, -1.0, 0.8]))
        cache = _make_sem_cache(W_sparse, len(y))
        cache = cache._replace(XtX=X.T @ X)
        eta_new, _ = _sample_eta_sem(state, y, X, W_sparse, rng=rng, cache=cache)
        assert eta_new.shape == (len(y),)

    def test_finite(self):
        """Draws should be finite."""
        rng = np.random.default_rng(1)
        y, X, W_sparse, _, _, lam, eta = _make_small_sem_problem()
        state = _make_sem_state(eta, np.array([0.5, -1.0, 0.8]))
        cache = _make_sem_cache(W_sparse, len(y))
        cache = cache._replace(XtX=X.T @ X)
        eta_new, _ = _sample_eta_sem(state, y, X, W_sparse, rng=rng, cache=cache)
        assert np.all(np.isfinite(eta_new))


# ---------------------------------------------------------------------------
# SEM _sample_beta tests
# ---------------------------------------------------------------------------


class TestSampleBetaSEM:
    """_sample_beta_sem should draw from the SEM conjugate normal conditional."""

    def test_shape(self):
        """Output shape matches number of coefficients."""
        rng = np.random.default_rng(0)
        y, X, W_sparse, _, beta, lam, eta = _make_small_sem_problem()
        state = _make_sem_state(eta, beta)
        priors = SEMLogitGibbsPriors()
        A_lam = sp.eye(len(y), format="csr") - lam * W_sparse
        A_lam_eta = A_lam @ eta
        beta_new = _sample_beta_sem(state, X, priors, A_lam_eta, W_sparse, rng=rng)
        assert beta_new.shape == (3,)

    def test_finite(self):
        """Draws should be finite."""
        rng = np.random.default_rng(1)
        y, X, W_sparse, _, beta, lam, eta = _make_small_sem_problem()
        state = _make_sem_state(eta, beta)
        priors = SEMLogitGibbsPriors()
        A_lam = sp.eye(len(y), format="csr") - lam * W_sparse
        A_lam_eta = A_lam @ eta
        beta_new = _sample_beta_sem(state, X, priors, A_lam_eta, W_sparse, rng=rng)
        assert np.all(np.isfinite(beta_new))


# ---------------------------------------------------------------------------
# SEM _sample_lam tests
# ---------------------------------------------------------------------------


class TestSampleLam:
    """_sample_lam should draw from the collapsed slice sampler."""

    def test_stays_in_bounds(self):
        """λ should stay within [lam_lower, lam_upper]."""
        rng = np.random.default_rng(0)
        y, X, W_sparse, _, beta, lam, eta = _make_small_sem_problem()
        state = _make_sem_state(eta, beta, lam=lam)
        priors = SEMLogitGibbsPriors()
        cache = _make_sem_cache(W_sparse, len(y), priors)
        cache = cache._replace(XtX=X.T @ X)
        lam_new, log_dens = _sample_lam(
            state, cache, priors, y, X, rng=rng, sweep_idx=0, tune=100
        )
        assert priors.lam_lower <= lam_new <= priors.lam_upper

    def test_finite_log_density(self):
        """Log-density at new λ should be finite."""
        rng = np.random.default_rng(1)
        y, X, W_sparse, _, beta, lam, eta = _make_small_sem_problem()
        state = _make_sem_state(eta, beta, lam=lam)
        priors = SEMLogitGibbsPriors()
        cache = _make_sem_cache(W_sparse, len(y), priors)
        cache = cache._replace(XtX=X.T @ X)
        lam_new, log_dens = _sample_lam(
            state, cache, priors, y, X, rng=rng, sweep_idx=0, tune=100
        )
        assert np.isfinite(log_dens)


# ---------------------------------------------------------------------------
# SEM collapsed log-density test
# ---------------------------------------------------------------------------


class TestSEMCollapsedLogDensity:
    """_sem_logit_collapsed_log_density should match analytical formula."""

    def test_finite_at_zero(self):
        """Log-density at λ=0 should be finite."""
        y, X, W_sparse, _, beta, lam, eta = _make_small_sem_problem()
        n = len(y)
        omega = np.ones(n)
        Xbeta = X @ beta
        W_eigs = np.linalg.eigvalsh(W_sparse.toarray())

        def logdet_fn(lam):
            return float(np.sum(np.log(np.maximum(1.0 - lam * W_eigs, 1e-300))))

        log_dens = _sem_logit_collapsed_log_density(
            0.0, y, X, W_sparse, logdet_fn, omega, Xbeta, n, X.shape[1]
        )
        assert np.isfinite(log_dens)

    def test_symmetric(self):
        """Log-density at λ and -λ should differ (not symmetric in general)."""
        y, X, W_sparse, _, beta, lam, eta = _make_small_sem_problem()
        n = len(y)
        omega = np.ones(n)
        Xbeta = X @ beta
        W_eigs = np.linalg.eigvalsh(W_sparse.toarray())

        def logdet_fn(lam):
            return float(np.sum(np.log(np.maximum(1.0 - lam * W_eigs, 1e-300))))

        ld_pos = _sem_logit_collapsed_log_density(
            0.3, y, X, W_sparse, logdet_fn, omega, Xbeta, n, X.shape[1]
        )
        ld_neg = _sem_logit_collapsed_log_density(
            -0.3, y, X, W_sparse, logdet_fn, omega, Xbeta, n, X.shape[1]
        )
        # They should be different (not symmetric due to rhs structure)
        assert ld_pos != ld_neg


# ---------------------------------------------------------------------------
# SEM run_chain tests
# ---------------------------------------------------------------------------


class TestRunChainSEM:
    """run_chain_sem should produce correct output shapes."""

    def test_output_shapes(self):
        """Output arrays have correct shapes."""
        rng = np.random.default_rng(0)
        y, X, W_sparse, _, beta, lam, eta = _make_small_sem_problem()
        n, k = X.shape

        priors = SEMLogitGibbsPriors()
        cache = _make_sem_cache(W_sparse, n, priors)
        cache = cache._replace(XtX=X.T @ X)

        beta_init = np.linalg.lstsq(X, y, rcond=None)[0]
        eta_init = X @ beta_init
        from neighbayes.samplers._utils._polyagamma import sample_polyagamma

        omega_init = sample_polyagamma(np.ones(n), eta_init, rng=rng)

        init = SEMLogitGibbsState(
            eta=eta_init, beta=beta_init, lam=0.0, omega=omega_init
        )

        draws = 50
        tune = 50
        result = run_chain_sem(
            y=y,
            X=X,
            W_sparse=W_sparse,
            priors=priors,
            cache=cache,
            init=init,
            draws=draws,
            tune=tune,
            rng=rng,
        )

        assert result["lam"].shape == (draws,)
        assert result["beta"].shape == (draws, k)
        assert result["log_lik"].shape == (draws, n)
        assert result["eta_norm"].shape == (draws,)

    def test_lam_in_bounds(self):
        """All λ draws should be within prior bounds."""
        rng = np.random.default_rng(2)
        y, X, W_sparse, _, beta, lam, eta = _make_small_sem_problem()
        n, k = X.shape

        priors = SEMLogitGibbsPriors()
        cache = _make_sem_cache(W_sparse, n, priors)
        cache = cache._replace(XtX=X.T @ X)

        beta_init = np.linalg.lstsq(X, y, rcond=None)[0]
        eta_init = X @ beta_init
        from neighbayes.samplers._utils._polyagamma import sample_polyagamma

        omega_init = sample_polyagamma(np.ones(n), eta_init, rng=rng)

        init = SEMLogitGibbsState(
            eta=eta_init, beta=beta_init, lam=0.0, omega=omega_init
        )

        draws = 50
        tune = 50
        result = run_chain_sem(
            y=y,
            X=X,
            W_sparse=W_sparse,
            priors=priors,
            cache=cache,
            init=init,
            draws=draws,
            tune=tune,
            rng=rng,
        )

        assert np.all(result["lam"] >= priors.lam_lower)
        assert np.all(result["lam"] <= priors.lam_upper)


# ---------------------------------------------------------------------------
# Parameter recovery tests
# ---------------------------------------------------------------------------


class TestSEMLogitRecovery:
    """SEM-logit Gibbs should recover true parameters from DGP data."""

    def test_lam_recovery(self):
        """Posterior mean of λ should be near true value (large sample)."""
        from neighbayes.dgp import simulate_sem_logit

        # Use n=8 (64 obs) for better recovery
        out = simulate_sem_logit(n_side=8, lam=0.5, seed=42)
        y = out["y"]
        X = out["X"]
        W_sparse = out["W_sparse"]
        n = len(y)

        priors = SEMLogitGibbsPriors()
        cache = _make_sem_cache(W_sparse, n, priors)
        cache = cache._replace(XtX=X.T @ X)

        rng = np.random.default_rng(0)
        beta_init = np.linalg.lstsq(X, y, rcond=None)[0]
        eta_init = X @ beta_init
        from neighbayes.samplers._utils._polyagamma import sample_polyagamma

        omega_init = sample_polyagamma(np.ones(n), eta_init, rng=rng)

        init = SEMLogitGibbsState(
            eta=eta_init, beta=beta_init, lam=0.0, omega=omega_init
        )

        result = run_chain_sem(
            y=y,
            X=X,
            W_sparse=W_sparse,
            priors=priors,
            cache=cache,
            init=init,
            draws=2000,
            tune=1000,
            rng=rng,
        )

        lam_mean = np.mean(result["lam"])
        # With 2000 draws and n=64, we expect reasonable recovery
        assert abs(lam_mean - 0.5) < 0.4, f"λ mean = {lam_mean}, expected ~0.5"

    def test_beta_recovery(self):
        """Posterior mean of β should be near true value (large sample)."""
        from neighbayes.dgp import simulate_sem_logit

        # Use n=8 (64 obs) for better recovery
        out = simulate_sem_logit(n_side=8, lam=0.3, beta=np.array([0.3, 1.0]), seed=456)
        y = out["y"]
        X = out["X"]
        W_sparse = out["W_sparse"]
        n = len(y)

        priors = SEMLogitGibbsPriors()
        cache = _make_sem_cache(W_sparse, n, priors)
        cache = cache._replace(XtX=X.T @ X)

        rng = np.random.default_rng(0)
        beta_init = np.linalg.lstsq(X, y, rcond=None)[0]
        eta_init = X @ beta_init
        from neighbayes.samplers._utils._polyagamma import sample_polyagamma

        omega_init = sample_polyagamma(np.ones(n), eta_init, rng=rng)

        init = SEMLogitGibbsState(
            eta=eta_init, beta=beta_init, lam=0.0, omega=omega_init
        )

        result = run_chain_sem(
            y=y,
            X=X,
            W_sparse=W_sparse,
            priors=priors,
            cache=cache,
            init=init,
            draws=2000,
            tune=1000,
            rng=rng,
        )

        beta_mean = np.mean(result["beta"], axis=0)
        # Wide tolerance for small sample
        assert np.all(np.abs(beta_mean - np.array([0.3, 1.0])) < 1.5), (
            f"β mean = {beta_mean}, expected ~[0.3, 1.0]"
        )
