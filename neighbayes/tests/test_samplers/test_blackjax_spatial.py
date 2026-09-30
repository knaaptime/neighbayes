"""Tests for the native blackjax spatial-NUTS path (``gaussian/_blackjax.py``).

Validates that:

1. the joint ``(β, σ², ρ)`` log-density is correct — it recovers known
   parameters and, on identical data + priors, targets the same posterior as
   the package's own SAR Gibbs sampler (within MC error);
2. a surrogate logdet (chol-cheb) matches the exact eigenvalue reference;
3. the ``spatial=False`` OLS yardstick recovers ``β`` from OLS data;
4. the backend-neutral log-density is differentiable and matches a direct
   numpy evaluation at a point.
"""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest
import scipy.sparse as sp

_HAS_JAX = importlib.util.find_spec("jax") is not None
_HAS_BLACKJAX = importlib.util.find_spec("blackjax") is not None

pytestmark = pytest.mark.skipif(
    not (_HAS_JAX and _HAS_BLACKJAX), reason="JAX + blackjax required"
)

from neighbayes.dgp.cross_sectional import simulate_ols, simulate_sar  # noqa: E402
from neighbayes.models.priors import OLSPriors, SARPriors  # noqa: E402
from neighbayes.samplers.gaussian._blackjax import (  # noqa: E402
    make_sar_joint_logdensity,
    run_chain_blackjax_gaussian,
)

_PRIORS = {"beta_mu": 0.0, "beta_sigma": 10.0, "sigma2_beta": 1.0}


def _sar_priors():
    return SARPriors(rho_lower=-1.0, rho_upper=1.0, **_PRIORS)


# ---------------------------------------------------------------------------
# Parameter recovery + posterior agreement with Gibbs
# ---------------------------------------------------------------------------


class TestRecovery:
    def test_recovers_known_parameters(self):
        d = simulate_sar(
            n_side=15, rho=0.6, beta=np.array([1.0, 2.0]), sigma=1.0, seed=3
        )
        res = run_chain_blackjax_gaussian(
            d["y"],
            d["X"],
            d["W_sparse"],
            _sar_priors(),
            spatial=True,
            logdet_method="eigenvalue",
            draws=1500,
            tune=1000,
            chains=2,
            seed=0,
        )
        assert res.num_divergent == 0
        # n = 225 concentrates the posterior near truth.
        assert abs(res.rho.mean() - 0.6) < 0.1
        beta_mean = res.beta.mean(axis=(0, 1))
        np.testing.assert_allclose(beta_mean, [1.0, 2.0], atol=0.2)
        assert abs(res.sigma2.mean() - 1.0) < 0.25

    def test_matches_package_gibbs_posterior(self):
        """blackjax NUTS and package SAR Gibbs target the same posterior."""
        from neighbayes.models import SAR

        d = simulate_sar(
            n_side=15, rho=0.6, beta=np.array([1.0, 2.0]), sigma=1.0, seed=3
        )
        y, X, Wd = d["y"], d["X"], d["W_sparse"]

        res = run_chain_blackjax_gaussian(
            y,
            X,
            Wd,
            _sar_priors(),
            spatial=True,
            logdet_method="eigenvalue",
            draws=3000,
            tune=1000,
            chains=2,
            seed=0,
        )
        m = SAR(y=y, X=X, W=Wd, priors=_PRIORS, logdet_method="eigenvalue")
        idata = m.fit(
            sampler="gibbs", draws=3000, tune=1000, chains=2, progressbar=False
        )
        post = idata.posterior

        # ρ and σ² posterior means agree within a few MC standard errors.
        assert abs(res.rho.mean() - float(post["rho"].mean())) < 0.02
        assert abs(res.sigma2.mean() - float(post["sigma2"].mean())) < 0.05
        # β posterior means agree (Gibbs stores the 2-vector under "beta").
        gibbs_beta = np.asarray(post["beta"].mean(axis=(0, 1)))
        np.testing.assert_allclose(res.beta.mean(axis=(0, 1)), gibbs_beta, atol=0.1)


# ---------------------------------------------------------------------------
# Surrogate logdet matches the eigenvalue reference
# ---------------------------------------------------------------------------


class TestSurrogateAgreesWithReference:
    def test_cheb_cholesky_matches_eigenvalue(self):
        d = simulate_sar(
            n_side=15, rho=0.5, beta=np.array([0.5, 1.5]), sigma=1.0, seed=7
        )
        y, X, W = d["y"], d["X"], d["W_sparse"]
        common = dict(draws=2000, tune=1000, chains=2, seed=1, spatial=True)

        ref = run_chain_blackjax_gaussian(
            y, X, W, _sar_priors(), logdet_method="eigenvalue", **common
        )
        cheb = run_chain_blackjax_gaussian(
            y,
            X,
            W,
            _sar_priors(),
            logdet_method="cheb_cholesky",
            rho_lower=0.0,
            rho_upper=0.95,
            **common,
        )
        # Same posterior up to MC error — the surrogate is exact for chol-cheb.
        assert abs(ref.rho.mean() - cheb.rho.mean()) < 0.03
        np.testing.assert_allclose(
            ref.beta.mean(axis=(0, 1)), cheb.beta.mean(axis=(0, 1)), atol=0.1
        )


# ---------------------------------------------------------------------------
# OLS yardstick
# ---------------------------------------------------------------------------


class TestOLSYardstick:
    def test_ols_matches_least_squares(self):
        """OLS-mode posterior mean concentrates on the closed-form LS estimate.

        This validates the ``spatial=False`` log-density directly against the
        analytic MLE (the object the near-flat-prior OLS posterior concentrates
        on), independent of finite-sample deviation from the true β.
        """
        d = simulate_ols(n_side=22, beta=np.array([1.0, -2.0, 0.5]), sigma=1.0, seed=5)
        y, X = d["y"], d["X"]
        bhat = np.linalg.lstsq(X, y, rcond=None)[0]

        res = run_chain_blackjax_gaussian(
            y,
            X,
            None,
            OLSPriors(**_PRIORS),
            spatial=False,
            draws=1500,
            tune=1000,
            chains=2,
            seed=0,
        )
        assert res.rho is None
        assert res.num_divergent == 0
        np.testing.assert_allclose(res.beta.mean(axis=(0, 1)), bhat, atol=0.03)


# ---------------------------------------------------------------------------
# Backend-neutral log-density: differentiability + direct-eval agreement
# ---------------------------------------------------------------------------


class TestJointLogDensity:
    def test_grad_finite_and_matches_manual(self):
        import jax
        import jax.numpy as jnp

        jax.config.update("jax_enable_x64", True)

        d = simulate_sar(
            n_side=8, rho=0.4, beta=np.array([1.0, 1.0]), sigma=1.0, seed=2
        )
        y, X, W = d["y"], d["X"], d["W_sparse"]
        Wy = W @ y
        eigs = np.linalg.eigvals(d["W_sparse"].toarray()).real

        from neighbayes._logdet import make_logdet_jax_fn

        logdet_fn = make_logdet_jax_fn(eigs, method="eigenvalue")
        jld = make_sar_joint_logdensity(
            y, X, Wy, _sar_priors(), logdet_fn, spatial=True
        )

        k = X.shape[1]
        theta = jnp.asarray(
            np.concatenate([[0.5, 1.5], [np.log(0.9)], [0.2]])
        )  # beta, tau, u

        val = float(jld.logdensity_fn(theta))
        assert np.isfinite(val)
        g = jax.grad(jld.logdensity_fn)(theta)
        assert np.all(np.isfinite(np.asarray(g)))
        assert g.shape == (k + 2,)

        # Manual numpy re-evaluation of the joint log-density at theta.
        beta = np.array([0.5, 1.5])
        tau = np.log(0.9)
        sigma2 = np.exp(tau)
        u = 0.2
        s = 1.0 / (1.0 + np.exp(-u))
        rho = -1.0 + 2.0 * s
        resid = y - rho * Wy - X @ beta
        n = len(y)
        half_log_2pi = 0.5 * np.log(2 * np.pi)
        logdet = float(np.sum(np.log(np.abs(1.0 - rho * eigs))))
        loglik = (
            logdet - n * half_log_2pi - 0.5 * n * tau - 0.5 * (resid @ resid) / sigma2
        )
        bs = 10.0
        log_prior_beta = np.sum(
            -half_log_2pi - np.log(bs) - 0.5 * ((beta - 0.0) / bs) ** 2
        )
        from scipy.special import gammaln

        a0, b0 = 2.0, 1.0
        log_prior_sigma2 = a0 * np.log(b0) - gammaln(a0) - (a0 + 1) * tau - b0 / sigma2
        log_width = np.log(2.0)
        log_prior_rho = -log_width
        jac_sigma2 = tau
        jac_rho = log_width + (-np.logaddexp(0.0, -u)) + (-np.logaddexp(0.0, u))
        manual = (
            loglik
            + log_prior_beta
            + log_prior_sigma2
            + log_prior_rho
            + jac_sigma2
            + jac_rho
        )
        np.testing.assert_allclose(val, manual, rtol=1e-10, atol=1e-10)

    def test_to_constrained_roundtrip(self):
        d = simulate_sar(
            n_side=8, rho=0.4, beta=np.array([1.0, 1.0]), sigma=1.0, seed=2
        )
        y, X, W = d["y"], d["X"], d["W_sparse"]
        Wy = W @ y
        eigs = np.linalg.eigvals(d["W_sparse"].toarray()).real

        from neighbayes._logdet import make_logdet_jax_fn

        logdet_fn = make_logdet_jax_fn(eigs, method="eigenvalue")
        jld = make_sar_joint_logdensity(
            y, X, Wy, _sar_priors(), logdet_fn, spatial=True
        )
        theta = np.concatenate([[0.5, 1.5], [np.log(0.9)], [0.2]])
        out = jld.to_constrained(theta)
        np.testing.assert_allclose(out["beta"], [0.5, 1.5])
        np.testing.assert_allclose(out["sigma2"], 0.9)
        s = 1.0 / (1.0 + np.exp(-0.2))
        np.testing.assert_allclose(out["rho"], -1.0 + 2.0 * s)
