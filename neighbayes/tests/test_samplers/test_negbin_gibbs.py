"""Tests for the NB Gibbs sampler primitives.

Covers:
- _sample_alpha numerical stability (no -inf from log(0))
- GibbsPriors data-dependent sigma2_beta default
- Gibbs-vs-NUTS parity: σ² prior scale matches data scale
"""

from __future__ import annotations

import numpy as np
import pytest

from neighbayes.samplers.negbin._core import GibbsPriors, GibbsState, _sample_alpha

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_state(eta, alpha=2.0, beta=None, sigma2=0.5, rho=0.3):
    """Create a minimal GibbsState for unit tests."""
    n = len(eta)
    return GibbsState(
        eta=eta,
        beta=beta if beta is not None else np.zeros(2),
        sigma2=sigma2,
        rho=rho,
        alpha=alpha,
        omega=np.ones(n),
    )


# ---------------------------------------------------------------------------
# _sample_alpha numerical stability
# ---------------------------------------------------------------------------


class TestSampleAlphaNumericalStability:
    """_sample_alpha must not produce -inf log-density from log(0)."""

    def test_extreme_negative_eta(self):
        """Very negative η (tiny mu) should not crash or return -inf alpha."""
        rng = np.random.default_rng(0)
        # eta = -50 → mu = exp(-50) ≈ 5e-22, so mu/(mu+alpha) ≈ 0
        eta = np.full(10, -50.0)
        y = np.ones(10, dtype=float)
        state = _make_state(eta, alpha=2.0)
        priors = GibbsPriors(alpha_sigma=10.0)

        alpha_new = _sample_alpha(state, y, priors, rng=rng)
        assert np.isfinite(alpha_new), f"alpha_new={alpha_new} is not finite"
        assert alpha_new > 0, f"alpha_new={alpha_new} should be positive"

    def test_extreme_positive_eta(self):
        """Very positive η (huge mu) should not crash or return -inf alpha."""
        rng = np.random.default_rng(1)
        # eta = 50 → mu = exp(50) ≈ 5e21, so alpha/(mu+alpha) ≈ 0
        eta = np.full(10, 50.0)
        y = np.full(10, 100.0)
        state = _make_state(eta, alpha=2.0)
        priors = GibbsPriors(alpha_sigma=10.0)

        alpha_new = _sample_alpha(state, y, priors, rng=rng)
        assert np.isfinite(alpha_new), f"alpha_new={alpha_new} is not finite"
        assert alpha_new > 0, f"alpha_new={alpha_new} should be positive"

    def test_mixed_eta(self):
        """Mix of extreme positive and negative η should not crash."""
        rng = np.random.default_rng(2)
        eta = np.array([-50.0, 50.0, -30.0, 30.0, 0.0] * 2)
        y = np.array([0, 100, 1, 50, 5] * 2, dtype=float)
        state = _make_state(eta, alpha=2.0)
        priors = GibbsPriors(alpha_sigma=10.0)

        alpha_new = _sample_alpha(state, y, priors, rng=rng)
        assert np.isfinite(alpha_new), f"alpha_new={alpha_new} is not finite"
        assert alpha_new > 0, f"alpha_new={alpha_new} should be positive"

    def test_zero_y_extreme_eta(self):
        """y=0 with extreme η should not crash (log(0/(0+alpha)) edge case)."""
        rng = np.random.default_rng(3)
        eta = np.full(10, -50.0)
        y = np.zeros(10, dtype=float)
        state = _make_state(eta, alpha=2.0)
        priors = GibbsPriors(alpha_sigma=10.0)

        alpha_new = _sample_alpha(state, y, priors, rng=rng)
        assert np.isfinite(alpha_new), f"alpha_new={alpha_new} is not finite"

    def test_repeated_draws_stable(self):
        """Multiple draws from _sample_alpha remain finite and positive."""
        rng = np.random.default_rng(42)
        eta = np.array([-20.0, 0.0, 20.0, -10.0, 5.0] * 2)
        y = np.array([0, 3, 50, 1, 10] * 2, dtype=float)
        state = _make_state(eta, alpha=2.0)
        priors = GibbsPriors(alpha_sigma=10.0)

        for _ in range(50):
            alpha_new = _sample_alpha(state, y, priors, rng=rng)
            assert np.isfinite(alpha_new) and alpha_new > 0
            # Update state for next draw
            state = _make_state(eta, alpha=alpha_new)


# ---------------------------------------------------------------------------
# GibbsPriors data-dependent sigma2_beta
# ---------------------------------------------------------------------------


class TestGibbsPriorsSigma2Beta:
    """Verify GibbsPriors accepts data-dependent sigma2_beta."""

    def test_custom_sigma2_beta(self):
        """GibbsPriors stores a custom sigma2_beta value."""
        priors = GibbsPriors(sigma2_beta=5.0)
        assert priors.sigma2_beta == 5.0

    def test_default_sigma2_beta(self):
        """GibbsPriors default sigma2_beta is 1.0 (for backward compat)."""
        priors = GibbsPriors()
        assert priors.sigma2_beta == 1.0

    def test_data_dependent_sigma2_beta(self):
        """GibbsPriors can be constructed with np.var(y) as sigma2_beta."""
        y = np.random.default_rng(0).normal(0, 3, size=100)
        sigma2_beta = max(float(np.var(y)), 0.1)
        priors = GibbsPriors(sigma2_beta=sigma2_beta)
        assert priors.sigma2_beta > 1.0  # var(y) with scale=3 should be ~9


# ---------------------------------------------------------------------------
# Integration: rho mixing after stale-cache fix
# ---------------------------------------------------------------------------


class TestSARNegBinStructuralRhoMixing:
    """Verify that the collapsed rho slice sampler mixes (not stuck).

    Regression test for the bug where log_density_rho was cached across
    Gibbs iterations, causing the slice level to be computed from a stale
    conditional and rho to get stuck at a single value.
    """

    def test_rho_not_stuck(self):
        """Rho should have more than 1 unique value after 200 draws."""
        from neighbayes import dgp
        from neighbayes.models import SARNegBinStructural
        from neighbayes.tests.helpers import W_to_graph, _as_sparse, make_rook_W

        rng = np.random.default_rng(42)
        data = dgp.simulate_sar_negbin(
            W=_as_sparse(W_to_graph(make_rook_W(5))),
            rho=0.4,
            beta=np.array([1.0, 0.5]),
            alpha=2.0,
            sigma2=0.5,
            rng=rng,
        )
        model = SARNegBinStructural(y=data["y"], X=data["X"], W=data["W_graph"])
        idata = model.fit(
            draws=200, tune=200, chains=1, random_seed=42, n_jobs=1, progressbar=False
        )
        rho = idata.posterior["rho"].values[0]
        n_unique = len(np.unique(rho.round(6)))
        # Before the fix, rho was stuck at a single value (n_unique=1).
        # After the fix, rho should mix freely (n_unique >> 1).
        assert n_unique > 10, (
            f"rho appears stuck: only {n_unique} unique values in 200 draws"
        )
