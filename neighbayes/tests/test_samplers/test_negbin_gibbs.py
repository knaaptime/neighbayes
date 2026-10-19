"""Tests for the NB samplers' shared α step: numerical stability (no -inf from log(0))."""

from __future__ import annotations

import numpy as np
import pytest

from neighbayes.models.priors import ReducedGibbsPriors
from neighbayes.samplers.negbin._core import GibbsState, _sample_alpha

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
        priors = ReducedGibbsPriors(alpha_sigma=10.0)

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
        priors = ReducedGibbsPriors(alpha_sigma=10.0)

        alpha_new = _sample_alpha(state, y, priors, rng=rng)
        assert np.isfinite(alpha_new), f"alpha_new={alpha_new} is not finite"
        assert alpha_new > 0, f"alpha_new={alpha_new} should be positive"

    def test_mixed_eta(self):
        """Mix of extreme positive and negative η should not crash."""
        rng = np.random.default_rng(2)
        eta = np.array([-50.0, 50.0, -30.0, 30.0, 0.0] * 2)
        y = np.array([0, 100, 1, 50, 5] * 2, dtype=float)
        state = _make_state(eta, alpha=2.0)
        priors = ReducedGibbsPriors(alpha_sigma=10.0)

        alpha_new = _sample_alpha(state, y, priors, rng=rng)
        assert np.isfinite(alpha_new), f"alpha_new={alpha_new} is not finite"
        assert alpha_new > 0, f"alpha_new={alpha_new} should be positive"

    def test_zero_y_extreme_eta(self):
        """y=0 with extreme η should not crash (log(0/(0+alpha)) edge case)."""
        rng = np.random.default_rng(3)
        eta = np.full(10, -50.0)
        y = np.zeros(10, dtype=float)
        state = _make_state(eta, alpha=2.0)
        priors = ReducedGibbsPriors(alpha_sigma=10.0)

        alpha_new = _sample_alpha(state, y, priors, rng=rng)
        assert np.isfinite(alpha_new), f"alpha_new={alpha_new} is not finite"

    def test_repeated_draws_stable(self):
        """Multiple draws from _sample_alpha remain finite and positive."""
        rng = np.random.default_rng(42)
        eta = np.array([-20.0, 0.0, 20.0, -10.0, 5.0] * 2)
        y = np.array([0, 3, 50, 1, 10] * 2, dtype=float)
        state = _make_state(eta, alpha=2.0)
        priors = ReducedGibbsPriors(alpha_sigma=10.0)

        for _ in range(50):
            alpha_new = _sample_alpha(state, y, priors, rng=rng)
            assert np.isfinite(alpha_new) and alpha_new > 0
            # Update state for next draw
            state = _make_state(eta, alpha=alpha_new)
