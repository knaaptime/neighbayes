"""Parameter recovery tests for SARZINB (zero-inflated SAR Negative Binomial).

Generates count data with zero-inflation from known parameters, fits the
model via the Gibbs sampler, and verifies that posterior means recover
the spatial and regression parameters for both the count and selection
equations.

The ZINB DGP has two equations:
- **Selection** (SAR-logit): ``d ~ Bernoulli(logit^{-1}(eta^sel))`` where
  ``eta^sel = (I - lam W)^{-1} Z gamma`` — the reduced form the model fits
  (the DGP's former latent ``nu`` was removed on 2026-10-03)
- **Count** (reduced-form SAR-NB): ``y|d=1 ~ NegBin(exp(eta^cnt), alpha)``
  where ``eta^cnt = (I - rho W)^{-1} X beta``

Five parameters to recover: ``rho``, ``lam``, ``beta``, ``gamma``, ``alpha``.

Run with::

    pytest tests/test_models/test_sar_zinb_recovery.py -m slow -v
"""

from __future__ import annotations

import numpy as np
import pytest

from neighbayes import dgp
from neighbayes.models import SARZINB
from neighbayes.tests.helpers import W_to_graph, _as_sparse, make_rook_W

pytestmark = [pytest.mark.slow, pytest.mark.recovery]

# Grid size: NB dispersion + zero-inflation need large N for reliable recovery
SIDE = 32  # 1024 cross-sectional units

# True parameters
RHO_TRUE = 0.4
LAM_TRUE = 0.3
BETA_TRUE = np.array([1.0, 0.6])
GAMMA_TRUE = np.array([0.3, 1.0])
ALPHA_TRUE = 2.0

# Recovery tolerances
ABS_TOL_RHO = 0.20
ABS_TOL_LAM = 0.25
ABS_TOL_BETA = 0.40
ABS_TOL_GAMMA = 0.40
ABS_TOL_ALPHA = 1.5

DRAWS = 1000
TUNE = 1000
CHAINS = 2


@pytest.fixture(scope="module")
def zinb_data():
    """Simulated ZINB data on a 32x32 rook grid (n=1024)."""
    rng = np.random.default_rng(42)
    W_dense = make_rook_W(SIDE)
    W_graph = W_to_graph(W_dense)
    return dgp.simulate_sar_zinb(
        W=_as_sparse(W_graph),
        rho=RHO_TRUE,
        lam=LAM_TRUE,
        beta=BETA_TRUE,
        gamma=GAMMA_TRUE,
        alpha=ALPHA_TRUE,
        rng=rng,
    )


class TestSARZINBRecovery:
    """Parameter recovery for SARZINB (Gibbs sampler)."""

    def test_recovers_rho(self, zinb_data):
        y = zinb_data["y"]
        X = zinb_data["X"]
        Z = zinb_data["Z"]
        W = zinb_data["W_graph"]

        model = SARZINB(y=y, X=X, Z=Z, W=W)
        idata = model.fit(
            draws=DRAWS,
            tune=TUNE,
            chains=CHAINS,
            random_seed=42,
            progressbar=False,
        )

        rho_hat = float(idata.posterior["rho"].mean())
        assert abs(rho_hat - RHO_TRUE) < ABS_TOL_RHO, (
            f"rho: {rho_hat:.3f} vs {RHO_TRUE}"
        )

    def test_recovers_lam(self, zinb_data):
        y = zinb_data["y"]
        X = zinb_data["X"]
        Z = zinb_data["Z"]
        W = zinb_data["W_graph"]

        model = SARZINB(y=y, X=X, Z=Z, W=W)
        idata = model.fit(
            draws=DRAWS,
            tune=TUNE,
            chains=CHAINS,
            random_seed=42,
            progressbar=False,
        )

        lam_hat = float(idata.posterior["lam"].mean())
        assert abs(lam_hat - LAM_TRUE) < ABS_TOL_LAM, (
            f"lam: {lam_hat:.3f} vs {LAM_TRUE}"
        )

    def test_recovers_beta(self, zinb_data):
        y = zinb_data["y"]
        X = zinb_data["X"]
        Z = zinb_data["Z"]
        W = zinb_data["W_graph"]

        model = SARZINB(y=y, X=X, Z=Z, W=W)
        idata = model.fit(
            draws=DRAWS,
            tune=TUNE,
            chains=CHAINS,
            random_seed=42,
            progressbar=False,
        )

        beta_hat = idata.posterior["beta"].mean(("chain", "draw")).values
        for j, (bhat, btrue) in enumerate(zip(beta_hat, BETA_TRUE)):
            assert abs(bhat - btrue) < ABS_TOL_BETA, f"beta[{j}]: {bhat:.3f} vs {btrue}"

    def test_recovers_gamma(self, zinb_data):
        y = zinb_data["y"]
        X = zinb_data["X"]
        Z = zinb_data["Z"]
        W = zinb_data["W_graph"]

        model = SARZINB(y=y, X=X, Z=Z, W=W)
        idata = model.fit(
            draws=DRAWS,
            tune=TUNE,
            chains=CHAINS,
            random_seed=42,
            progressbar=False,
        )

        gamma_hat = idata.posterior["gamma"].mean(("chain", "draw")).values
        for j, (ghat, gtrue) in enumerate(zip(gamma_hat, GAMMA_TRUE)):
            assert abs(ghat - gtrue) < ABS_TOL_GAMMA, (
                f"gamma[{j}]: {ghat:.3f} vs {gtrue}"
            )

    def test_recovers_alpha(self, zinb_data):
        y = zinb_data["y"]
        X = zinb_data["X"]
        Z = zinb_data["Z"]
        W = zinb_data["W_graph"]

        model = SARZINB(y=y, X=X, Z=Z, W=W)
        idata = model.fit(
            draws=DRAWS,
            tune=TUNE,
            chains=CHAINS,
            random_seed=42,
            progressbar=False,
        )

        alpha_hat = float(idata.posterior["alpha"].mean())
        assert alpha_hat > 0, f"alpha={alpha_hat:.3f} should be positive"
        assert abs(alpha_hat - ALPHA_TRUE) < ABS_TOL_ALPHA, (
            f"alpha: {alpha_hat:.3f} vs {ALPHA_TRUE}"
        )
