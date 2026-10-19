"""Parameter recovery tests for the reduced-form spatial logit.

Generates binary outcomes from known parameters,
``eta = (I - rho W)^{-1} X beta``, fits ``SARLogit`` once through the public
``.fit()`` API, and verifies that posterior means recover ρ and β within a
small tolerance.

Run with::

    pytest tests/test_models/test_spatial_logit_recovery.py -m slow -v
"""

from __future__ import annotations

import numpy as np
import pytest

from neighbayes.models import SARLogit
from neighbayes.tests.helpers import (
    SAMPLE_KWARGS,
    W_to_graph,
    make_rook_W,
    make_sar_logit_data,
)

pytestmark = [pytest.mark.slow, pytest.mark.recovery]

# ---------------------------------------------------------------------------
# Grid size: binary spatial models need large N for reliable rho recovery
# (the model warns n >= 900).  30 x 30 rook grid = 900 observations.
# ---------------------------------------------------------------------------
SIDE = 30  # 900 cross-sectional units

# True parameters
RHO_TRUE = 0.4
BETA_TRUE = np.array([0.3, 1.0])

# Recovery tolerances — the model matches its DGP, so tight tolerances apply
# to ρ and the slopes.
ABS_TOL_SPATIAL = 0.15
ABS_TOL_BETA = 0.25


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def sar_logit_reduced_data():
    """Reduced-form SAR-logit binary data (noise-free) on a 30x30 rook grid."""
    rng = np.random.default_rng(42)
    W_dense = make_rook_W(SIDE)
    y, X = make_sar_logit_data(rng, W_dense, rho=RHO_TRUE, beta=BETA_TRUE)
    return y, X, W_dense


# ---------------------------------------------------------------------------
# SARLogit (reduced-form, noise-free DGP)
# ---------------------------------------------------------------------------


class TestSARLogitRecovery:
    """Parameter recovery for the reduced-form SARLogit model."""

    def test_recovers_rho_and_beta(self, sar_logit_reduced_data):
        y, X, W_dense = sar_logit_reduced_data
        W_graph = W_to_graph(W_dense)
        model = SARLogit(y=y, X=X, W=W_graph)
        idata = model.fit(**SAMPLE_KWARGS)

        rho_hat = float(idata.posterior["rho"].mean())
        assert abs(rho_hat - RHO_TRUE) < ABS_TOL_SPATIAL, (
            f"SARLogit rho: expected ~{RHO_TRUE}, got {rho_hat:.3f}"
        )

        beta_hat = idata.posterior["beta"].mean(("chain", "draw")).values
        for j, (bhat, btrue) in enumerate(zip(beta_hat, BETA_TRUE)):
            assert abs(bhat - btrue) < ABS_TOL_BETA, (
                f"SARLogit beta[{j}]: expected ~{btrue}, got {bhat:.3f}"
            )

    def test_fitted_probabilities_in_range(self, sar_logit_reduced_data):
        y, X, W_dense = sar_logit_reduced_data
        W_graph = W_to_graph(W_dense)
        model = SARLogit(y=y, X=X, W=W_graph)
        model.fit(**SAMPLE_KWARGS)

        p_hat = model.fitted_probabilities()
        assert p_hat.shape == (len(y),)
        assert np.all((p_hat >= 0.0) & (p_hat <= 1.0))
