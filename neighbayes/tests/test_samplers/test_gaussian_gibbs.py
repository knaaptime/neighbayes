"""Tests for the Gaussian Gibbs sampler primitives and integration.

Covers:
- Unit tests for collapsed/conditional log-densities
- Unit tests for block samplers (β, σ²)
- Unit tests for pointwise log-likelihood functions
- Integration tests: Gibbs vs NUTS posterior convergence
- InferenceData compatibility (LOO, WAIC, spatial diagnostics)
- JAX JIT path tests (skipped when JAX unavailable)
- Edge cases (tiny n, intercept-only, robust raises)
"""

from __future__ import annotations

import importlib.util

import arviz as az
import numpy as np
import pytest
import scipy.sparse as sp

from neighbayes._logdet import make_logdet_numpy_fn, make_logdet_numpy_vec_fn
from neighbayes.samplers.gaussian._core import (
    GaussianGibbsCache,
    GaussianGibbsPriors,
    GaussianGibbsState,
    _draw_beta_from_cho,
    _initialize_gaussian_gibbs,
    _sample_beta_conjugate,
    _sample_sigma2,
    _sar_log_density_given_sigma2,
    _sar_sweep_terms,
    _sem_log_density_given_sigma2,
    _sem_precision_terms,
    gaussian_sweep,
    run_gaussian_chain,
)
from neighbayes.samplers.gaussian._loglik import (
    ols_pointwise_loglik_numpy,
    sar_pointwise_loglik_numpy,
    sar_pointwise_loglik_vectorized,
    sem_pointwise_loglik_numpy,
    sem_pointwise_loglik_vectorized,
)
from neighbayes.tests.helpers import W_to_graph, make_rook_W

# Skip JAX tests when JAX is not installed
_JAX_AVAILABLE = importlib.util.find_spec("jax") is not None
requires_jax = pytest.mark.skipif(not _JAX_AVAILABLE, reason="JAX not installed")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SIDE = 4  # 16 cross-sectional units (small for speed)


def _make_sar_data(side=SIDE, rho_true=0.4, beta_true=None, sigma_true=1.0, seed=42):
    """Create a small synthetic SAR dataset using the DGP module."""
    from neighbayes.dgp.cross_sectional import simulate_sar

    rng = np.random.default_rng(seed)
    W_dense = make_rook_W(side)
    out = simulate_sar(
        W=sp.csr_matrix(W_dense),
        rho=rho_true,
        beta=beta_true,
        sigma=sigma_true,
        rng=rng,
    )
    return out["y"], out["X"], W_dense, W_dense.shape[0]


def _make_sem_data(side=SIDE, lam_true=0.4, beta_true=None, sigma_true=1.0, seed=42):
    """Create a small synthetic SEM dataset using the DGP module."""
    from neighbayes.dgp.cross_sectional import simulate_sem

    rng = np.random.default_rng(seed)
    W_dense = make_rook_W(side)
    out = simulate_sem(
        W=sp.csr_matrix(W_dense),
        lam=lam_true,
        beta=beta_true,
        sigma=sigma_true,
        rng=rng,
    )
    return out["y"], out["X"], W_dense, W_dense.shape[0]


def _build_cache(W_dense, X, model_type="sar", y=None, Wy=None, method="eigenvalue"):
    """Build a GaussianGibbsCache from data."""
    W_sparse = sp.csr_matrix(W_dense)
    eigs = np.linalg.eigvals(W_dense).real
    rho_min = float(1.0 / eigs.min() + 1e-6)
    rho_max = float(1.0 / eigs.max() - 1e-6)
    if rho_min > rho_max:
        rho_min, rho_max = rho_max, rho_min

    logdet_fn = make_logdet_numpy_fn(
        W_sparse, eigs=eigs, method=method, rho_min=rho_min, rho_max=rho_max
    )
    logdet_vec_fn = make_logdet_numpy_vec_fn(
        W_sparse, eigs=eigs, method=method, rho_min=rho_min, rho_max=rho_max
    )
    XtX = X.T @ X
    from scipy.linalg import cho_factor

    XtX_cho = cho_factor(XtX)

    # Precompute Wy for all model types
    if Wy is None and y is not None:
        Wy = W_dense @ y
    elif Wy is None:
        Wy = W_dense @ np.ones(X.shape[0])  # placeholder; caller should override

    # Precompute y-side inner products for ALL model types
    WX = None
    XtWX = None
    WXtWX = None
    yty = None
    yTWy = None
    WyTWy = None
    XTy = None
    XTWy = None
    WXTy = None
    WXTWy = None
    if y is not None:
        yty = float(y @ y)
        yTWy = float(y @ Wy)
        WyTWy = float(Wy @ Wy)
        XTy = X.T @ y
        XTWy = X.T @ Wy
    if model_type in ("sem", "sdem"):
        WX = W_sparse @ X
        XtWX = X.T @ WX
        WXtWX = WX.T @ WX
        if y is not None:
            WXTy = WX.T @ y
            WXTWy = WX.T @ Wy

    return GaussianGibbsCache(
        XtX=XtX,
        XtX_cho=XtX_cho,
        logdet_fn=logdet_fn,
        logdet_vec_fn=logdet_vec_fn,
        rho_lower=rho_min,
        rho_upper=rho_max,
        model_type=model_type,
        Wy=Wy,
        W_sparse=W_sparse,
        WX=WX,
        XtWX=XtWX,
        WXtWX=WXtWX,
        yty=yty,
        yTWy=yTWy,
        WyTWy=WyTWy,
        XTy=XTy,
        XTWy=XTWy,
        WXTy=WXTy,
        WXTWy=WXTWy,
    )


# ===================================================================
# Unit tests: log-density functions
# ===================================================================


def _exact_log_density(param, y, X, W_dense, sigma2, priors, model):
    """Brute-force log p(ρ | σ², y) with β integrated out, up to a constant.

    Given ρ and σ², the filtered response ``y* = A y`` is
    ``N(X* μ₀, σ² I + X* Λ₀ X*ᵀ)``; the Jacobian of ``y ↦ y*`` is ``|A|``.
    """
    n, k = X.shape
    A = np.eye(n) - param * W_dense
    y_star = A @ y
    X_star = X if model == "sar" else A @ X
    mu = np.full(k, priors.beta_mu, dtype=float)
    Lam0 = np.diag(np.full(k, priors.beta_sigma, dtype=float) ** 2)
    C = sigma2 * np.eye(n) + X_star @ Lam0 @ X_star.T
    resid = y_star - X_star @ mu
    _, logdet_C = np.linalg.slogdet(C)
    _, logdet_A = np.linalg.slogdet(A)
    return logdet_A - 0.5 * logdet_C - 0.5 * resid @ np.linalg.solve(C, resid)


class TestSpatialLogDensityGivenSigma2:
    """ρ/λ | σ², y with β integrated out under its Normal prior."""

    PRIORS = GaussianGibbsPriors(beta_mu=0.3, beta_sigma=2.0)

    @pytest.mark.parametrize("sigma2", [0.5, 1.7])
    def test_sar_matches_brute_force(self, sigma2):
        y, X, W_dense, n = _make_sar_data()
        cache = _build_cache(W_dense, X, model_type="sar", y=y, Wy=W_dense @ y)
        terms = _sar_sweep_terms(cache, sigma2, self.PRIORS)
        grid = (-0.6, -0.2, 0.0, 0.3, 0.7)
        got = np.array(
            [_sar_log_density_given_sigma2(r, cache, sigma2, terms) for r in grid]
        )
        want = np.array(
            [
                _exact_log_density(r, y, X, W_dense, sigma2, self.PRIORS, "sar")
                for r in grid
            ]
        )
        # Equal up to an additive constant in ρ.
        np.testing.assert_allclose(got - got[0], want - want[0], atol=1e-8)

    @pytest.mark.parametrize("sigma2", [0.5, 1.7])
    def test_sem_matches_brute_force(self, sigma2):
        y, X, W_dense, n = _make_sem_data()
        cache = _build_cache(W_dense, X, model_type="sem", y=y)
        grid = (-0.6, -0.2, 0.0, 0.3, 0.7)
        got = np.array(
            [_sem_log_density_given_sigma2(l, cache, sigma2, self.PRIORS) for l in grid]
        )
        want = np.array(
            [
                _exact_log_density(l, y, X, W_dense, sigma2, self.PRIORS, "sem")
                for l in grid
            ]
        )
        np.testing.assert_allclose(got - got[0], want - want[0], atol=1e-8)

    def test_sar_decreases_at_boundaries(self):
        """log|I-ρW| → -∞ at the spectral bound, so the density falls there."""
        y, X, W_dense, n = _make_sar_data()
        cache = _build_cache(W_dense, X, model_type="sar", y=y, Wy=W_dense @ y)
        terms = _sar_sweep_terms(cache, 1.0, self.PRIORS)
        ld_mid = _sar_log_density_given_sigma2(0.0, cache, 1.0, terms)
        ld_edge = _sar_log_density_given_sigma2(
            cache.rho_upper - 1e-4, cache, 1.0, terms
        )
        assert ld_edge < ld_mid


# ===================================================================
# Unit tests: block samplers
# ===================================================================


class TestBetaBlock:
    """Tests for conjugate normal β draw."""

    @pytest.mark.parametrize("model", ["sar", "sem"])
    def test_beta_given_rho_sigma2_matches_analytical(self, model):
        """β | ρ, σ² from the sweep factor has the analytical mean and covariance."""
        make = _make_sar_data if model == "sar" else _make_sem_data
        y, X, W_dense, n = make()
        cache = _build_cache(W_dense, X, model_type=model, y=y, Wy=W_dense @ y)
        priors = GaussianGibbsPriors(beta_mu=0.3, beta_sigma=2.0)
        rho, sigma2 = 0.35, 0.8
        if model == "sar":
            terms = _sar_sweep_terms(cache, sigma2, priors)
            P_cho, mean = terms.P_cho, terms.m0 - rho * terms.m1
        else:
            from scipy.linalg import cho_solve

            P_cho, b, _ = _sem_precision_terms(rho, cache, sigma2, priors)
            mean = cho_solve(P_cho, b)

        A = np.eye(n) - rho * W_dense
        y_star = A @ y
        X_star = X if model == "sar" else A @ X
        k = X.shape[1]
        prec0 = np.eye(k) / priors.beta_sigma**2
        post_prec = X_star.T @ X_star / sigma2 + prec0
        post_cov = np.linalg.inv(post_prec)
        post_mean = post_cov @ (X_star.T @ y_star / sigma2 + prec0 @ np.full(k, 0.3))
        np.testing.assert_allclose(mean, post_mean, atol=1e-10)

        rng = np.random.default_rng(0)
        draws = np.array([_draw_beta_from_cho(P_cho, mean, rng) for _ in range(20000)])
        # Monte Carlo SE of each covariance entry is about sqrt(Σii Σjj / N).
        sd = np.sqrt(np.diag(post_cov))
        se = np.outer(sd, sd) / np.sqrt(len(draws))
        assert np.all(np.abs(np.cov(draws.T) - post_cov) < 5 * se)

    def test_conjugate_normal_matches_analytical(self):
        """Posterior mean of many draws should match analytical posterior mean."""
        rng = np.random.default_rng(42)
        n, k = 50, 3
        X = np.column_stack([np.ones(n), rng.standard_normal((n, 2))])
        beta_true = np.array([1.0, 2.0, -0.5])
        r = X @ beta_true + 0.5 * rng.standard_normal(n)
        sigma2 = 0.25
        priors = GaussianGibbsPriors(beta_mu=0.0, beta_sigma=100.0)
        XtX = X.T @ X

        # Analytical posterior
        prior_prec = np.diag(1.0 / np.full(k, 100.0) ** 2)
        post_prec = XtX / sigma2 + prior_prec
        post_cov = np.linalg.inv(post_prec)
        post_mean = post_cov @ (X.T @ r / sigma2 + prior_prec @ np.zeros(k))

        # Draw many samples
        draws = np.array(
            [
                _sample_beta_conjugate(r, X, XtX, sigma2, priors, rng)
                for _ in range(5000)
            ]
        )

        # Sample mean should be close to analytical posterior mean
        np.testing.assert_allclose(draws.mean(axis=0), post_mean, atol=0.1)


class TestSigma2Block:
    """Tests for the σ² conjugate Inverse-Gamma update.

    With prior ``σ² ~ InverseGamma(α, β)`` and Gaussian likelihood the
    full conditional is ``InverseGamma(α + n/2, β + ss/2)`` — drawn
    exactly in one step.
    """

    def test_sample_sigma2_sar_positive(self):
        y, X, W_dense, n = _make_sar_data()
        Wy = W_dense @ y
        priors = GaussianGibbsPriors()
        rng = np.random.default_rng(42)
        beta = np.array([1.0, 2.0])
        sigma2 = _sample_sigma2(0.3, beta, y, Wy, None, X, priors, "sar", rng)
        assert sigma2 > 0
        assert isinstance(sigma2, float)

    def test_sample_sigma2_sem_positive(self):
        y, X, W_dense, n = _make_sem_data()
        W_sparse = sp.csr_matrix(W_dense)
        Wy = W_dense @ y
        priors = GaussianGibbsPriors()
        rng = np.random.default_rng(42)
        beta = np.array([1.0, 2.0])
        sigma2 = _sample_sigma2(0.3, beta, y, Wy, W_sparse, X, priors, "sem", rng)
        assert sigma2 > 0
        assert isinstance(sigma2, float)

    def test_sigma2_draws_converge_to_analytical(self):
        """Mean of σ² draws matches the analytical Inv-Γ posterior mean."""
        rng = np.random.default_rng(42)
        n = 50
        X = np.column_stack([np.ones(n), rng.standard_normal(n)])
        beta_true = np.array([1.0, 2.0])
        y = X @ beta_true + 0.5 * rng.standard_normal(n)
        resid = y - X @ beta_true
        Wy = np.zeros(n)  # ρ=0 so Wy doesn't matter, but must be provided
        priors = GaussianGibbsPriors(sigma2_alpha=2.0, sigma2_beta=1.0)

        # Posterior: Inv-Γ(α + n/2, β + ss/2), mean = b_post / (a_post - 1).
        a_post = priors.sigma2_alpha + n / 2.0
        b_post = priors.sigma2_beta + 0.5 * np.dot(resid, resid)
        expected_mean = b_post / (a_post - 1)

        draws = np.empty(5000)
        for i in range(5000):
            draws[i] = _sample_sigma2(
                0.0, beta_true, y, Wy, None, X, priors, "sar", rng
            )

        np.testing.assert_allclose(draws.mean(), expected_mean, rtol=0.05)


# ===================================================================
# Unit tests: pointwise log-likelihood
# ===================================================================


class TestPointwiseLogLik:
    """Tests for pointwise log-likelihood functions."""

    def test_sar_loglik_sums_to_total(self):
        """Per-obs LL sums to total LL (Gaussian + Jacobian)."""
        y, X, W_dense, n = _make_sar_data()
        Wy = W_dense @ y
        W_sparse = sp.csr_matrix(W_dense)
        eigs = np.linalg.eigvals(W_dense).real
        logdet_fn = make_logdet_numpy_fn(W_sparse, eigs=eigs, method="eigenvalue")
        beta = np.array([1.0, 2.0])
        sigma = 1.0
        rho = 0.3

        ll = sar_pointwise_loglik_numpy(y, X, Wy, beta, sigma, rho, logdet_fn, n)
        total_ll = ll.sum()

        # Manual computation
        mu = rho * Wy + X @ beta
        resid = y - mu
        gauss_total = (
            -0.5 * np.dot(resid, resid) / sigma**2
            - n * np.log(sigma)
            - n / 2 * np.log(2 * np.pi)
        )
        jacobian = logdet_fn(rho)
        expected_total = gauss_total + jacobian

        np.testing.assert_allclose(total_ll, expected_total, rtol=1e-10)

    def test_sem_loglik_sums_to_total(self):
        """Per-obs LL sums to total LL (Gaussian + Jacobian)."""
        y, X, W_dense, n = _make_sem_data()
        W_sparse = sp.csr_matrix(W_dense)
        eigs = np.linalg.eigvals(W_dense).real
        logdet_fn = make_logdet_numpy_fn(W_sparse, eigs=eigs, method="eigenvalue")
        beta = np.array([1.0, 2.0])
        sigma = 1.0
        lam = 0.3

        ll = sem_pointwise_loglik_numpy(y, X, W_sparse, beta, sigma, lam, logdet_fn, n)
        total_ll = ll.sum()

        # Manual computation
        resid = y - X @ beta
        eps = resid - lam * (W_sparse @ resid)
        gauss_total = (
            -0.5 * np.dot(eps, eps) / sigma**2
            - n * np.log(sigma)
            - n / 2 * np.log(2 * np.pi)
        )
        jacobian = logdet_fn(lam)
        expected_total = gauss_total + jacobian

        np.testing.assert_allclose(total_ll, expected_total, rtol=1e-10)

    def test_sar_vectorized_matches_per_draw(self):
        """Vectorized LL should match per-draw computation."""
        y, X, W_dense, n = _make_sar_data()
        Wy = W_dense @ y
        W_sparse = sp.csr_matrix(W_dense)
        eigs = np.linalg.eigvals(W_dense).real
        logdet_fn = make_logdet_numpy_fn(W_sparse, eigs=eigs, method="eigenvalue")
        logdet_vec_fn = make_logdet_numpy_vec_fn(
            W_sparse, eigs=eigs, method="eigenvalue"
        )

        n_keep = 10
        rng = np.random.default_rng(42)
        rho_draws = rng.uniform(0.1, 0.5, n_keep)
        beta_draws = rng.standard_normal((n_keep, X.shape[1]))
        sigma_draws = np.abs(rng.standard_normal(n_keep)) + 0.5

        # Vectorized
        ll_vec = sar_pointwise_loglik_vectorized(
            rho_draws, beta_draws, sigma_draws, y, X, Wy, logdet_vec_fn, n
        )

        # Per-draw
        ll_per = np.empty((n_keep, n))
        for g in range(n_keep):
            ll_per[g] = sar_pointwise_loglik_numpy(
                y, X, Wy, beta_draws[g], sigma_draws[g], rho_draws[g], logdet_fn, n
            )

        np.testing.assert_allclose(ll_vec, ll_per, rtol=1e-10)

    def test_sem_vectorized_matches_per_draw(self):
        """Vectorized LL should match per-draw computation."""
        y, X, W_dense, n = _make_sem_data()
        W_sparse = sp.csr_matrix(W_dense)
        eigs = np.linalg.eigvals(W_dense).real
        logdet_fn = make_logdet_numpy_fn(W_sparse, eigs=eigs, method="eigenvalue")
        logdet_vec_fn = make_logdet_numpy_vec_fn(
            W_sparse, eigs=eigs, method="eigenvalue"
        )

        n_keep = 10
        rng = np.random.default_rng(42)
        lam_draws = rng.uniform(0.1, 0.5, n_keep)
        beta_draws = rng.standard_normal((n_keep, X.shape[1]))
        sigma_draws = np.abs(rng.standard_normal(n_keep)) + 0.5

        # Vectorized
        ll_vec = sem_pointwise_loglik_vectorized(
            lam_draws, beta_draws, sigma_draws, y, X, W_sparse, logdet_vec_fn, n
        )

        # Per-draw
        ll_per = np.empty((n_keep, n))
        for g in range(n_keep):
            ll_per[g] = sem_pointwise_loglik_numpy(
                y,
                X,
                W_sparse,
                beta_draws[g],
                sigma_draws[g],
                lam_draws[g],
                logdet_fn,
                n,
            )

        np.testing.assert_allclose(ll_vec, ll_per, rtol=1e-10)

    def test_ols_loglik_no_jacobian(self):
        """OLS LL should not include Jacobian term."""
        rng = np.random.default_rng(42)
        n = 20
        X = np.column_stack([np.ones(n), rng.standard_normal(n)])
        beta = np.array([1.0, 2.0])
        sigma = 1.0
        y = X @ beta + sigma * rng.standard_normal(n)

        ll = ols_pointwise_loglik_numpy(y, X, beta, sigma)
        # Sum should equal standard Gaussian LL (no Jacobian)
        resid = y - X @ beta
        expected = -0.5 * (resid / sigma) ** 2 - np.log(sigma) - 0.5 * np.log(2 * np.pi)
        np.testing.assert_allclose(ll, expected, rtol=1e-10)


# ===================================================================
# Unit tests: initialization
# ===================================================================


class TestInitialization:
    """Tests for _initialize_gaussian_gibbs."""

    def test_ols_warm_start(self):
        y, X, W_dense, n = _make_sar_data()
        from scipy.linalg import cho_factor

        XtX_cho = cho_factor(X.T @ X)
        priors = GaussianGibbsPriors()
        rng = np.random.default_rng(42)
        state = _initialize_gaussian_gibbs(y, X, XtX_cho, priors, rng)

        assert isinstance(state, GaussianGibbsState)
        assert state.beta.shape == (X.shape[1],)
        assert state.sigma2 > 0
        assert state.rho == 0.0  # starts at 0

    def test_ols_beta_close_to_lstsq(self):
        y, X, W_dense, n = _make_sar_data()
        from scipy.linalg import cho_factor

        XtX_cho = cho_factor(X.T @ X)
        priors = GaussianGibbsPriors()
        rng = np.random.default_rng(42)
        state = _initialize_gaussian_gibbs(y, X, XtX_cho, priors, rng)

        beta_lstsq = np.linalg.lstsq(X, y, rcond=None)[0]
        np.testing.assert_allclose(state.beta, beta_lstsq, rtol=1e-10)


# ===================================================================
# Integration tests: chain runner
# ===================================================================


class TestChainRunner:
    """Tests for run_gaussian_chain."""

    def test_sar_chain_produces_valid_output(self):
        y, X, W_dense, n = _make_sar_data()
        Wy = W_dense @ y
        cache = _build_cache(W_dense, X, model_type="sar", y=y, Wy=Wy)
        priors = GaussianGibbsPriors()
        rng = np.random.default_rng(42)
        init = _initialize_gaussian_gibbs(y, X, cache.XtX_cho, priors, rng)

        result = run_gaussian_chain(
            y=y,
            X=X,
            cache=cache,
            priors=priors,
            init=init,
            draws=50,
            tune=20,
            thin=1,
            rng=rng,
        )

        assert "rho" in result
        assert "beta" in result
        assert "sigma" in result
        assert "log_lik" in result
        assert result["rho"].shape == (50,)
        assert result["beta"].shape == (50, X.shape[1])
        assert result["sigma"].shape == (50,)
        assert result["log_lik"].shape == (50, n)
        assert np.all(result["sigma"] > 0)
        assert np.all(np.isfinite(result["log_lik"]))

    def test_sem_chain_produces_valid_output(self):
        y, X, W_dense, n = _make_sem_data()
        cache = _build_cache(W_dense, X, model_type="sem", y=y)
        priors = GaussianGibbsPriors()
        rng = np.random.default_rng(42)
        init = _initialize_gaussian_gibbs(y, X, cache.XtX_cho, priors, rng)

        result = run_gaussian_chain(
            y=y,
            X=X,
            cache=cache,
            priors=priors,
            init=init,
            draws=50,
            tune=20,
            thin=1,
            rng=rng,
        )

        assert "lam" in result
        assert "beta" in result
        assert "sigma" in result
        assert "log_lik" in result
        assert result["lam"].shape == (50,)
        assert result["beta"].shape == (50, X.shape[1])
        assert result["sigma"].shape == (50,)
        assert result["log_lik"].shape == (50, n)

    def test_thinning(self):
        y, X, W_dense, n = _make_sar_data()
        Wy = W_dense @ y
        cache = _build_cache(W_dense, X, model_type="sar", y=y, Wy=Wy)
        priors = GaussianGibbsPriors()
        rng = np.random.default_rng(42)
        init = _initialize_gaussian_gibbs(y, X, cache.XtX_cho, priors, rng)

        result = run_gaussian_chain(
            y=y,
            X=X,
            cache=cache,
            priors=priors,
            init=init,
            draws=100,
            tune=20,
            thin=2,
            rng=rng,
        )

        # 100 draws with thin=2 → 50 kept
        assert result["rho"].shape == (50,)


# ===================================================================
# Integration tests: Gibbs vs NUTS posterior convergence
# ===================================================================


@pytest.mark.slow
class TestGibbsVsNUTS:
    """Gibbs and NUTS posteriors should agree for Gaussian models.

    These tests use moderate draws and check that posterior means
    are within a tolerance.  They are marked ``slow`` because they
    run MCMC chains.
    """

    def test_sar_gibbs_vs_nuts(self):
        from neighbayes.models.cross_section.sar import SAR

        y, X, W_dense, n = _make_sar_data(rho_true=0.4)
        W = W_to_graph(W_dense)

        model_nuts = SAR(y=y, X=X, W=W)
        idata_nuts = model_nuts.fit(
            sampler="nuts",
            draws=500,
            tune=500,
            chains=2,
            random_seed=42,
            target_accept=0.9,
            progressbar=False,
            idata_kwargs={"log_likelihood": True},
        )

        model_gibbs = SAR(y=y, X=X, W=W)
        idata_gibbs = model_gibbs.fit(
            sampler="gibbs",
            draws=500,
            tune=500,
            chains=2,
            random_seed=42,
            n_jobs=1,
            progressbar=False,
        )

        for param in ["rho", "sigma"]:
            mean_nuts = float(idata_nuts.posterior[param].mean())
            mean_gibbs = float(idata_gibbs.posterior[param].mean())
            np.testing.assert_allclose(mean_nuts, mean_gibbs, atol=0.15)

        # Beta: compare element-wise
        beta_nuts = idata_nuts.posterior["beta"].mean(dim=["chain", "draw"]).values
        beta_gibbs = idata_gibbs.posterior["beta"].mean(dim=["chain", "draw"]).values
        np.testing.assert_allclose(beta_nuts, beta_gibbs, atol=0.15)

    def test_sem_gibbs_vs_nuts(self):
        from neighbayes.models.cross_section.sem import SEM

        y, X, W_dense, n = _make_sem_data(lam_true=0.4)
        W = W_to_graph(W_dense)

        model_nuts = SEM(y=y, X=X, W=W)
        idata_nuts = model_nuts.fit(
            sampler="nuts",
            draws=500,
            tune=500,
            chains=2,
            random_seed=42,
            target_accept=0.9,
            progressbar=False,
            idata_kwargs={"log_likelihood": True},
        )

        model_gibbs = SEM(y=y, X=X, W=W)
        idata_gibbs = model_gibbs.fit(
            sampler="gibbs",
            draws=500,
            tune=500,
            chains=2,
            random_seed=42,
            n_jobs=1,
            progressbar=False,
        )

        for param in ["lam", "sigma"]:
            mean_nuts = float(idata_nuts.posterior[param].mean())
            mean_gibbs = float(idata_gibbs.posterior[param].mean())
            np.testing.assert_allclose(mean_nuts, mean_gibbs, atol=0.15)

    def test_sdm_gibbs_vs_nuts(self):
        from neighbayes.models.cross_section.sdm import SDM

        y, X, W_dense, n = _make_sar_data(rho_true=0.3)
        W = W_to_graph(W_dense)

        model_nuts = SDM(y=y, X=X, W=W)
        idata_nuts = model_nuts.fit(
            sampler="nuts",
            draws=500,
            tune=500,
            chains=2,
            random_seed=42,
            target_accept=0.9,
            progressbar=False,
            idata_kwargs={"log_likelihood": True},
        )

        model_gibbs = SDM(y=y, X=X, W=W)
        idata_gibbs = model_gibbs.fit(
            sampler="gibbs",
            draws=500,
            tune=500,
            chains=2,
            random_seed=42,
            n_jobs=1,
            progressbar=False,
        )

        mean_nuts = float(idata_nuts.posterior["rho"].mean())
        mean_gibbs = float(idata_gibbs.posterior["rho"].mean())
        np.testing.assert_allclose(mean_nuts, mean_gibbs, atol=0.15)

    def test_sdem_gibbs_vs_nuts(self):
        from neighbayes.models.cross_section.sdem import SDEM

        y, X, W_dense, n = _make_sem_data(lam_true=0.3)
        W = W_to_graph(W_dense)

        model_nuts = SDEM(y=y, X=X, W=W)
        idata_nuts = model_nuts.fit(
            sampler="nuts",
            draws=500,
            tune=500,
            chains=2,
            random_seed=42,
            target_accept=0.9,
            progressbar=False,
            idata_kwargs={"log_likelihood": True},
        )

        model_gibbs = SDEM(y=y, X=X, W=W)
        idata_gibbs = model_gibbs.fit(
            sampler="gibbs",
            draws=500,
            tune=500,
            chains=2,
            random_seed=42,
            n_jobs=1,
            progressbar=False,
        )

        mean_nuts = float(idata_nuts.posterior["lam"].mean())
        mean_gibbs = float(idata_gibbs.posterior["lam"].mean())
        np.testing.assert_allclose(mean_nuts, mean_gibbs, atol=0.15)


class TestJointPosterior:
    """The stored (β, σ, ρ) triples must be joint draws, not just good marginals.

    A sweep that draws ρ from a density with β integrated out and then keeps
    the β drawn under the previous ρ reproduces every marginal but decouples
    β from ρ.  In a SAR with an intercept, β₀ and ρ are strongly negatively
    correlated (the fitted mean is β₀ / (1 − ρ)); a decoupled sampler reports
    a correlation near zero.
    """

    def test_sar_intercept_rho_correlation_survives(self):
        y, X, W_dense, n = _make_sar_data(side=10, rho_true=0.5, beta_true=[3.0, 1.0])
        cache = _build_cache(W_dense, X, model_type="sar", y=y, Wy=W_dense @ y)
        priors = GaussianGibbsPriors()
        rng = np.random.default_rng(1)
        init = _initialize_gaussian_gibbs(y, X, cache.XtX_cho, priors, rng)
        out = run_gaussian_chain(
            y=y,
            X=X,
            cache=cache,
            priors=priors,
            init=init,
            draws=4000,
            tune=500,
            rng=rng,
            progressbar=False,
            store_log_lik=False,
        )
        corr = np.corrcoef(out["beta"][:, 0], out["rho"])[0, 1]
        assert corr < -0.5, corr

    def test_sweep_redraws_beta_after_rho(self):
        """β after a sweep is drawn at the sweep's own ρ, not the previous one."""
        y, X, W_dense, n = _make_sar_data(side=6)
        cache = _build_cache(W_dense, X, model_type="sar", y=y, Wy=W_dense @ y)
        priors = GaussianGibbsPriors()
        from neighbayes.samplers._utils._slice import SliceWidthState

        state = GaussianGibbsState(beta=np.zeros(X.shape[1]), sigma2=1.0, rho=0.0)
        rng = np.random.default_rng(3)
        gaussian_sweep(state, y, X, cache, priors, rng, SliceWidthState(w=0.3))
        # Recompute the conditional mean at the new (ρ, σ²): the drawn β must be
        # within a few posterior SDs of it.
        terms = _sar_sweep_terms(cache, state.sigma2, priors)
        mean = terms.m0 - state.rho * terms.m1
        cov = np.linalg.inv(terms.P_cho[0] @ terms.P_cho[0].T)
        z = (state.beta - mean) / np.sqrt(np.diag(cov))
        assert np.all(np.abs(z) < 5), z


@pytest.mark.slow
class TestRobustGibbsVsNUTS:
    """Robust Gibbs (normal scale mixture) matches NUTS on Student-t data."""

    @pytest.mark.parametrize(
        "backend", ["numpy", pytest.param("jax", marks=requires_jax)]
    )
    @pytest.mark.parametrize("model_name", ["SAR", "SEM"])
    def test_robust_posterior_matches_nuts(self, model_name, backend):
        from neighbayes.models import SAR, SEM

        rng = np.random.default_rng(11)
        W_dense = make_rook_W(12)
        n = W_dense.shape[0]
        X = np.column_stack([np.ones(n), rng.standard_normal(n)])
        eps = 0.8 * rng.standard_t(3.0, size=n)
        A = np.eye(n) - 0.5 * W_dense
        if model_name == "SAR":
            y = np.linalg.solve(A, X @ np.array([1.0, 2.0]) + eps)
        else:
            y = X @ np.array([1.0, 2.0]) + np.linalg.solve(A, eps)
        Model = {"SAR": SAR, "SEM": SEM}[model_name]
        W = W_to_graph(W_dense)
        param = "rho" if model_name == "SAR" else "lam"

        kw = dict(draws=3000, tune=1000, chains=4, random_seed=3, progressbar=False)
        nuts = Model(y=y, X=X, W=W, robust=True).fit(sampler="nuts", **kw)
        gibbs = Model(y=y, X=X, W=W, robust=True).fit(
            sampler="gibbs", gibbs_backend=backend, **kw
        )
        # The NUTS run is the reference, so check it before blaming Gibbs.
        rhat = az.rhat(nuts, var_names=[param, "sigma", "beta"])
        assert float(rhat.to_array().max()) < 1.01, "NUTS reference did not converge"
        for name in (param, "sigma", "beta"):
            a = nuts.posterior[name].values.reshape(-1, *nuts.posterior[name].shape[2:])
            b = gibbs.posterior[name].values.reshape(a.shape)
            # Five combined Monte Carlo standard errors, floored at 5% of the
            # posterior SD so a very long run does not demand digits it can't
            # resolve.
            se = np.sqrt(
                az.mcse(nuts, var_names=[name])[name].values ** 2
                + az.mcse(gibbs, var_names=[name])[name].values ** 2
            )
            tol = np.maximum(5 * se, 0.05 * a.std(axis=0))
            assert np.all(np.abs(b.mean(axis=0) - a.mean(axis=0)) < tol), name
            np.testing.assert_allclose(b.std(axis=0), a.std(axis=0), rtol=0.15)


@pytest.mark.slow
class TestGibbsJointVsNUTS:
    """Gibbs reproduces the NUTS joint posterior, including cross-correlations."""

    @pytest.mark.parametrize(
        "backend", ["numpy", pytest.param("jax", marks=requires_jax)]
    )
    @pytest.mark.parametrize("model_name", ["SAR", "SEM"])
    def test_correlations_match_nuts(self, model_name, backend):
        from neighbayes.models import SAR, SEM

        Model = {"SAR": SAR, "SEM": SEM}[model_name]
        make = _make_sar_data if model_name == "SAR" else _make_sem_data
        y, X, W_dense, n = make(side=10, beta_true=[3.0, 1.0])
        W = W_to_graph(W_dense)
        param = "rho" if model_name == "SAR" else "lam"

        def _stats(idata):
            post = idata.posterior
            sp_ = post[param].values.ravel()
            beta = post["beta"].values.reshape(-1, post["beta"].shape[-1])
            sig = post["sigma"].values.ravel()
            return (
                np.corrcoef(beta[:, 0], sp_)[0, 1],
                np.corrcoef(sig, sp_)[0, 1],
                sp_.mean(),
                sp_.std(),
            )

        kw = dict(draws=3000, tune=1000, chains=4, random_seed=7, progressbar=False)
        nuts = _stats(Model(y=y, X=X, W=W).fit(sampler="nuts", **kw))
        gibbs = _stats(
            Model(y=y, X=X, W=W).fit(sampler="gibbs", gibbs_backend=backend, **kw)
        )
        np.testing.assert_allclose(gibbs[:2], nuts[:2], atol=0.08)
        np.testing.assert_allclose(gibbs[2], nuts[2], atol=0.02)
        np.testing.assert_allclose(gibbs[3], nuts[3], rtol=0.15)


# ===================================================================
# InferenceData compatibility
# ===================================================================


class TestInferenceDataCompat:
    """Gibbs-produced InferenceData should work with ArviZ diagnostics."""

    @pytest.fixture
    def sar_idata(self):
        from neighbayes.models.cross_section.sar import SAR

        y, X, W_dense, n = _make_sar_data()
        W = W_to_graph(W_dense)
        model = SAR(y=y, X=X, W=W)
        return model.fit(
            sampler="gibbs",
            draws=100,
            tune=50,
            chains=2,
            random_seed=42,
            n_jobs=1,
            progressbar=False,
            idata_kwargs={"log_likelihood": True},
        )

    def test_idata_groups(self, sar_idata):
        assert "posterior" in sar_idata.groups()
        assert "log_likelihood" in sar_idata.groups()
        assert "observed_data" in sar_idata.groups()

    def test_posterior_vars(self, sar_idata):
        assert "rho" in sar_idata.posterior.data_vars
        assert "beta" in sar_idata.posterior.data_vars
        assert "sigma" in sar_idata.posterior.data_vars

    def test_log_likelihood_dims(self, sar_idata):
        """obs should be a data variable, not a coordinate."""
        assert "obs" in sar_idata.log_likelihood.data_vars
        ll = sar_idata.log_likelihood["obs"]
        assert ll.ndim == 3
        assert "chain" in ll.dims
        assert "draw" in ll.dims

    def test_loo_works(self, sar_idata):
        loo = az.loo(sar_idata)
        assert np.isfinite(loo.elpd_loo)

    def test_waic_works(self, sar_idata):
        waic = az.waic(sar_idata)
        assert np.isfinite(waic.elpd_waic)

    def test_summary_works(self, sar_idata):
        summary = az.summary(sar_idata)
        assert len(summary) > 0

    def test_sigma_not_sigma2(self, sar_idata):
        """Posterior should store σ (not σ²)."""
        sigma_mean = float(sar_idata.posterior["sigma"].mean())
        assert sigma_mean > 0  # σ is positive
        # σ should be on the order of the true value (1.0), not σ²
        assert sigma_mean < 10.0  # would be ~1, not ~1 if stored as σ²


# ===================================================================
# JAX JIT path tests
# ===================================================================


@requires_jax
class TestJAXGaussianGibbs:
    """Tests for the JAX JIT Gaussian Gibbs path."""

    def test_sar_jax_produces_valid_idata(self):
        from neighbayes.models.cross_section.sar import SAR

        y, X, W_dense, n = _make_sar_data()
        W = W_to_graph(W_dense)
        model = SAR(y=y, X=X, W=W)
        idata = model.fit(
            sampler="gibbs",
            draws=50,
            tune=20,
            chains=1,
            random_seed=42,
            n_jobs=1,
            progressbar=False,
            gibbs_backend="jax",
        )
        assert "posterior" in idata.groups()
        assert "rho" in idata.posterior.data_vars
        assert "sigma" in idata.posterior.data_vars
        assert float(idata.posterior["rho"].mean()) != 0  # not stuck at init

    def test_sem_jax_produces_valid_idata(self):
        from neighbayes.models.cross_section.sem import SEM

        y, X, W_dense, n = _make_sem_data()
        W = W_to_graph(W_dense)
        model = SEM(y=y, X=X, W=W)
        idata = model.fit(
            sampler="gibbs",
            draws=50,
            tune=20,
            chains=1,
            random_seed=42,
            n_jobs=1,
            progressbar=False,
            gibbs_backend="jax",
        )
        assert "lam" in idata.posterior.data_vars

    def test_sdm_jax_produces_valid_idata(self):
        from neighbayes.models.cross_section.sdm import SDM

        y, X, W_dense, n = _make_sar_data()
        W = W_to_graph(W_dense)
        model = SDM(y=y, X=X, W=W)
        idata = model.fit(
            sampler="gibbs",
            draws=50,
            tune=20,
            chains=1,
            random_seed=42,
            n_jobs=1,
            progressbar=False,
            gibbs_backend="jax",
        )
        assert "rho" in idata.posterior.data_vars
        assert idata.posterior["beta"].shape[-1] == 3  # intercept + x + W*x

    def test_sdem_jax_produces_valid_idata(self):
        from neighbayes.models.cross_section.sdem import SDEM

        y, X, W_dense, n = _make_sem_data()
        W = W_to_graph(W_dense)
        model = SDEM(y=y, X=X, W=W)
        idata = model.fit(
            sampler="gibbs",
            draws=50,
            tune=20,
            chains=1,
            random_seed=42,
            n_jobs=1,
            progressbar=False,
            gibbs_backend="jax",
        )
        assert "lam" in idata.posterior.data_vars

    def test_jax_loo_works(self):
        """LOO should work with JAX-produced InferenceData."""
        from neighbayes.models.cross_section.sar import SAR

        y, X, W_dense, n = _make_sar_data()
        W = W_to_graph(W_dense)
        model = SAR(y=y, X=X, W=W)
        idata = model.fit(
            sampler="gibbs",
            draws=100,
            tune=50,
            chains=2,
            random_seed=42,
            n_jobs=1,
            progressbar=False,
            gibbs_backend="jax",
            idata_kwargs={"log_likelihood": True},
        )
        loo = az.loo(idata)
        assert np.isfinite(loo.elpd_loo)

    def test_chebyshev_logdet_with_jax(self):
        """Chebyshev logdet method should work with JAX path."""
        from neighbayes.models.cross_section.sar import SAR

        y, X, W_dense, n = _make_sar_data()
        W = W_to_graph(W_dense)
        model = SAR(y=y, X=X, W=W, logdet_method="chebyshev")
        idata = model.fit(
            sampler="gibbs",
            draws=50,
            tune=20,
            chains=1,
            random_seed=42,
            n_jobs=1,
            progressbar=False,
            gibbs_backend="jax",
        )
        assert "rho" in idata.posterior.data_vars


# ===================================================================
# Edge cases
# ===================================================================


class TestEdgeCases:
    """Edge case tests for the Gibbs sampler."""

    @pytest.mark.parametrize(
        "backend", ["numpy", pytest.param("jax", marks=requires_jax)]
    )
    @pytest.mark.parametrize("model_name", ["SAR", "SEM", "SDM", "SDEM"])
    def test_robust_gibbs_runs(self, model_name, backend):
        """Robust (Student-t) models sample with Gibbs on both backends."""
        from neighbayes.models import SAR, SDEM, SDM, SEM

        Model = {"SAR": SAR, "SEM": SEM, "SDM": SDM, "SDEM": SDEM}[model_name]
        y, X, W_dense, n = _make_sar_data()
        model = Model(y=y, X=X, W=W_to_graph(W_dense), robust=True)
        idata = model.fit(
            sampler="gibbs",
            gibbs_backend=backend,
            draws=30,
            tune=20,
            chains=2,
            progressbar=False,
            idata_kwargs={"log_likelihood": True},
        )
        # The mixing variances are internal; the contract matches NUTS.
        assert "v" not in idata.posterior
        assert np.all(np.isfinite(idata.log_likelihood["obs"].values))

    def test_robust_pointwise_loglik_is_student_t(self):
        """Robust Gibbs stores the Student-t marginal, as the NUTS path does."""
        from scipy import stats

        from neighbayes.samplers.gaussian._core import _pointwise_loglik

        y, X, W_dense, n = _make_sar_data()
        cache = _build_cache(W_dense, X, model_type="sar", y=y, Wy=W_dense @ y)
        cache.nu = 4.0
        state = GaussianGibbsState(beta=np.array([1.0, 2.0]), sigma2=0.7, rho=0.3)
        eps = y - 0.3 * cache.Wy - X @ state.beta
        want = stats.t.logpdf(eps, df=4.0, scale=np.sqrt(0.7)) + cache.logdet_fn(
            0.3
        ) / len(y)
        np.testing.assert_allclose(_pointwise_loglik(state, y, X, cache), want)

    def test_tiny_n(self):
        """Gibbs should work with very small n."""
        from neighbayes.models.cross_section.sar import SAR

        y, X, W_dense, n = _make_sar_data(side=3)  # n=9
        W = W_to_graph(W_dense)
        model = SAR(y=y, X=X, W=W)
        idata = model.fit(
            sampler="gibbs",
            draws=50,
            tune=20,
            chains=1,
            random_seed=42,
            n_jobs=1,
            progressbar=False,
        )
        assert "rho" in idata.posterior.data_vars

    def test_intercept_only(self):
        """Gibbs should work with k=1 (intercept only)."""
        rng = np.random.default_rng(42)
        W_dense = make_rook_W(4)
        n = 16
        W = W_to_graph(W_dense)
        X = np.ones((n, 1))
        y = 1.0 + 0.3 * (W_dense @ np.ones(n)) + 0.5 * rng.standard_normal(n)

        from neighbayes.models.cross_section.sar import SAR

        model = SAR(y=y, X=X, W=W)
        idata = model.fit(
            sampler="gibbs",
            draws=50,
            tune=20,
            chains=1,
            random_seed=42,
            n_jobs=1,
            progressbar=False,
        )
        assert idata.posterior["beta"].shape[-1] == 1

    def test_invalid_sampler_raises(self):
        """Invalid sampler name should raise ValueError."""
        from neighbayes.models.cross_section.sar import SAR

        y, X, W_dense, n = _make_sar_data()
        W = W_to_graph(W_dense)
        model = SAR(y=y, X=X, W=W)
        with pytest.raises(ValueError, match="sampler"):
            model.fit(sampler="invalid")

    def test_invalid_gibbs_backend_raises(self):
        """An invalid gibbs_backend value raises (strict; no silent fallback)."""
        from neighbayes.models.cross_section.sar import SAR

        y, X, W_dense, n = _make_sar_data()
        W = W_to_graph(W_dense)
        model = SAR(y=y, X=X, W=W)
        with pytest.raises(ValueError, match="gibbs_backend must be one of"):
            model.fit(
                sampler="gibbs",
                draws=10,
                tune=5,
                chains=1,
                random_seed=42,
                n_jobs=1,
                progressbar=False,
                gibbs_backend="invalid",
            )

    def test_removed_gibbs_method_kwarg_rejected(self):
        """The old gibbs_method kwarg was renamed to gibbs_backend (strict)."""
        from neighbayes.models.cross_section.sar import SAR

        y, X, W_dense, n = _make_sar_data()
        W = W_to_graph(W_dense)
        model = SAR(y=y, X=X, W=W)
        with pytest.raises(TypeError, match="unsupported keyword"):
            model.fit(
                sampler="gibbs",
                draws=10,
                tune=5,
                chains=1,
                random_seed=42,
                n_jobs=1,
                progressbar=False,
                gibbs_method="jax",
            )


# ---------------------------------------------------------------------------
# Tests for progress bars, informative output, chain parallelism, mp context
# ---------------------------------------------------------------------------


class TestGibbsProgressBarManager:
    """Tests for GibbsProgressBarManager."""

    def test_context_manager_enter_exit(self):
        """Manager enters and exits cleanly with progressbar=True."""
        from neighbayes.samplers._utils._progress import GibbsProgressBarManager

        pm = GibbsProgressBarManager(chains=2, draws=10, tune=5, progressbar=True)
        with pm:
            assert pm._show is True
            assert len(pm._tasks) == 2

    def test_context_manager_no_progressbar(self):
        """Manager is a no-op when progressbar=False."""
        from neighbayes.samplers._utils._progress import GibbsProgressBarManager

        pm = GibbsProgressBarManager(chains=2, draws=10, tune=5, progressbar=False)
        with pm:
            assert pm._show is False
            assert len(pm._tasks) == 0

    def test_update_advances_iteration(self):
        """update() advances completed progress in tune and draw phases."""
        from neighbayes.samplers._utils._progress import GibbsProgressBarManager

        pm = GibbsProgressBarManager(chains=1, draws=5, tune=2, progressbar=True)
        with pm:
            task = pm._progress.tasks[pm._tasks[0]]
            pm.update(0, 0, tuning=True)
            assert task.completed == 1
            pm.update(0, 1, tuning=True)
            assert task.completed == 2
            pm.update(0, 2, tuning=False)
            assert task.completed == 3

    def test_update_noop_when_disabled(self):
        """update() is a no-op when progressbar=False."""
        from neighbayes.samplers._utils._progress import GibbsProgressBarManager

        pm = GibbsProgressBarManager(chains=1, draws=5, tune=2, progressbar=False)
        with pm:
            pm.update(0, 0, tuning=True)
            # Should not raise

    def test_no_accept_rate_column(self):
        """There is no accept-rate column, by design.

        Every Gibbs block here is a conjugate draw or a slice step, both of
        which accept by construction, so the column only ever showed ``--`` or
        a vacuous 100%.  Pins the removal so it is not reintroduced.
        """
        import inspect

        from neighbayes.samplers._utils._progress import GibbsProgressBarManager

        assert (
            "accept" not in inspect.signature(GibbsProgressBarManager.update).parameters
        )

        pm = GibbsProgressBarManager(chains=1, draws=5, tune=2, progressbar=True)
        with pm:
            assert "accept_rate" not in pm._progress.tasks[pm._tasks[0]].fields

    def test_exit_forces_final_refresh(self):
        """__exit__ forces one final refresh for notebook rendering."""
        from neighbayes.samplers._utils._progress import GibbsProgressBarManager

        pm = GibbsProgressBarManager(chains=1, draws=5, tune=2, progressbar=True)
        pm._progress.live.auto_refresh = False
        refresh_calls = {"count": 0}

        with pm:
            pm._progress.refresh = lambda: refresh_calls.__setitem__(
                "count", refresh_calls["count"] + 1
            )

        assert refresh_calls["count"] == 1


class TestInformativeOutput:
    """Tests for logging output from GibbsEstimation."""

    def test_numpy_path_logs_sampling_info(self, caplog):
        """NumPy path logs sampling start and completion."""
        from neighbayes.models.cross_section.sar import SAR

        y, X, W_dense, n = _make_sar_data()
        W = W_to_graph(W_dense)
        model = SAR(y=y, X=X, W=W)
        with caplog.at_level("INFO", logger="neighbayes.samplers.gaussian._estimation"):
            model.fit(
                sampler="gibbs",
                draws=10,
                tune=5,
                chains=1,
                random_seed=42,
                n_jobs=1,
                progressbar=False,
            )
        assert "Gibbs sampling" in caplog.text
        assert "took" in caplog.text

    def test_jax_path_logs_sampler_name(self, caplog):
        """JAX path logs the slice sampler name."""
        pytest.importorskip("jax")
        from neighbayes.models.cross_section.sar import SAR

        y, X, W_dense, n = _make_sar_data()
        W = W_to_graph(W_dense)
        model = SAR(y=y, X=X, W=W)
        with caplog.at_level("INFO", logger="neighbayes.samplers.gaussian._estimation"):
            model.fit(
                sampler="gibbs",
                draws=10,
                tune=5,
                chains=1,
                random_seed=42,
                n_jobs=1,
                progressbar=False,
                gibbs_backend="jax",
            )
        assert "JAX Gibbs sampling" in caplog.text
        assert "slice" in caplog.text


class TestChainParallelism:
    """Tests for chain parallelism (n_jobs for NumPy, chain_method for JAX)."""

    def test_sequential_numpy(self):
        """n_jobs=1 runs NumPy chains sequentially."""
        from neighbayes.models.cross_section.sar import SAR

        y, X, W_dense, n = _make_sar_data()
        W = W_to_graph(W_dense)
        model = SAR(y=y, X=X, W=W)
        idata = model.fit(
            sampler="gibbs",
            draws=10,
            tune=5,
            chains=2,
            random_seed=42,
            n_jobs=1,
            progressbar=False,
        )
        assert "posterior" in idata.groups()
        assert idata.posterior["beta"].shape[0] == 2  # 2 chains

    def test_parallel_numpy(self):
        """n_jobs=2 runs NumPy chains in parallel via joblib."""
        from neighbayes.models.cross_section.sar import SAR

        y, X, W_dense, n = _make_sar_data()
        W = W_to_graph(W_dense)
        model = SAR(y=y, X=X, W=W)
        idata = model.fit(
            sampler="gibbs",
            draws=10,
            tune=5,
            chains=2,
            random_seed=42,
            n_jobs=2,
            progressbar=False,
        )
        assert "posterior" in idata.groups()
        assert idata.posterior["beta"].shape[0] == 2  # 2 chains

    def test_parallel_numpy_all_cpus(self):
        """n_jobs=-1 runs NumPy chains in parallel using all CPUs."""
        from neighbayes.models.cross_section.sar import SAR

        y, X, W_dense, n = _make_sar_data()
        W = W_to_graph(W_dense)
        model = SAR(y=y, X=X, W=W)
        idata = model.fit(
            sampler="gibbs",
            draws=10,
            tune=5,
            chains=2,
            random_seed=42,
            n_jobs=-1,
            progressbar=False,
        )
        assert "posterior" in idata.groups()
        assert idata.posterior["beta"].shape[0] == 2  # 2 chains

    def test_vectorized_jax(self):
        """chain_method='vectorized' works for JAX path."""
        pytest.importorskip("jax")
        from neighbayes.models.cross_section.sar import SAR

        y, X, W_dense, n = _make_sar_data()
        W = W_to_graph(W_dense)
        model = SAR(y=y, X=X, W=W)
        idata = model.fit(
            sampler="gibbs",
            draws=10,
            tune=5,
            chains=2,
            random_seed=42,
            progressbar=False,
            gibbs_backend="jax",
            chain_method="vectorized",
        )
        assert "posterior" in idata.groups()
        assert idata.posterior["beta"].shape[0] == 2  # 2 chains

    def test_jax_default_is_vectorized(self):
        """JAX path defaults to chain_method='vectorized'."""
        pytest.importorskip("jax")
        from neighbayes.models.cross_section.sar import SAR

        y, X, W_dense, n = _make_sar_data()
        W = W_to_graph(W_dense)
        model = SAR(y=y, X=X, W=W)
        idata = model.fit(
            sampler="gibbs",
            draws=10,
            tune=5,
            chains=2,
            random_seed=42,
            progressbar=False,
            gibbs_backend="jax",
        )
        assert "posterior" in idata.groups()
        assert idata.posterior["beta"].shape[0] == 2  # 2 chains

    def test_parallel_jax_raises(self):
        """chain_method='parallel' raises NotImplementedError for JAX."""
        pytest.importorskip("jax")
        from neighbayes.models.cross_section.sar import SAR

        y, X, W_dense, n = _make_sar_data()
        W = W_to_graph(W_dense)
        model = SAR(y=y, X=X, W=W)
        with pytest.raises(NotImplementedError, match="vectorized"):
            model.fit(
                sampler="gibbs",
                draws=10,
                tune=5,
                chains=2,
                random_seed=42,
                progressbar=False,
                gibbs_backend="jax",
                chain_method="parallel",
            )

    def test_vectorized_runner_valid_results(self):
        """JAX vectorized runner returns valid multi-chain slice results."""
        jax = pytest.importorskip("jax")
        import jax.numpy as jnp
        import scipy.sparse as sp

        from neighbayes._logdet import (
            make_logdet_jax_fn,
            make_logdet_numpy_fn,
            make_logdet_numpy_vec_fn,
        )
        from neighbayes.samplers.gaussian._core import (
            GaussianGibbsCache,
            GaussianGibbsState,
            _initialize_gaussian_gibbs,
        )
        from neighbayes.samplers.gaussian._estimation import GaussianGibbsPriors
        from neighbayes.samplers.gaussian._jax import (
            run_chains_jax_gibbs_vectorized,
        )

        y, X, W_dense, n = _make_sar_data()
        W = W_to_graph(W_dense)
        Wy = W_dense @ y
        W_sparse = sp.csr_matrix(W_dense)

        priors = GaussianGibbsPriors()
        W_sparse = sp.csr_matrix(W_dense)
        eigs = np.linalg.eigvals(W_dense).real
        logdet_fn = make_logdet_numpy_fn(
            W_sparse,
            eigs=eigs,
            method="eigenvalue",
            rho_min=priors.rho_lower,
            rho_max=priors.rho_upper,
        )
        logdet_vec_fn = make_logdet_numpy_vec_fn(
            W_sparse,
            eigs=eigs,
            method="eigenvalue",
            rho_min=priors.rho_lower,
            rho_max=priors.rho_upper,
        )
        from scipy.linalg import cho_factor

        cache = GaussianGibbsCache(
            XtX=X.T @ X,
            XtX_cho=cho_factor(X.T @ X),
            logdet_fn=logdet_fn,
            logdet_vec_fn=logdet_vec_fn,
            rho_lower=priors.rho_lower,
            rho_upper=priors.rho_upper,
            model_type="sar",
            Wy=Wy,
            W_sparse=W_sparse,
        )

        # Initialize 4 chains with different seeds
        chains = 4
        inits = []
        for i in range(chains):
            rng = np.random.default_rng(42 + i)
            inits.append(_initialize_gaussian_gibbs(y, X, cache.XtX_cho, priors, rng))

        logdet_jax = make_logdet_jax_fn(
            W=W_dense,
            method="eigenvalue",
            rho_min=priors.rho_lower,
            rho_max=priors.rho_upper,
        )

        results = run_chains_jax_gibbs_vectorized(
            y=y,
            X=X,
            W_sparse=W_sparse,
            Wy=Wy,
            logdet_jax=logdet_jax,
            logdet_vec_fn=cache.logdet_vec_fn,
            priors=priors,
            inits=inits,
            draws=50,
            tune=50,
            thin=1,
            jax_seeds=list(range(chains)),
            model_type="sar",
            progressbar=False,
        )

        # Verify results are valid
        assert len(results) == chains
        for r in results:
            assert "rho" in r
            assert "beta" in r
            assert "sigma" in r
            assert "mh_accept_rate" in r
            # Accept rate should be between 0 and 1 (inclusive)
            assert 0 <= r["mh_accept_rate"] <= 1
