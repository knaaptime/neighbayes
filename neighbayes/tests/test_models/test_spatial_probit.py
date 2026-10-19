"""Parameter recovery tests for SEMProbit.

Generates binary outcomes from known parameters, fits the model once, and
verifies that posterior means recover the spatial / regression parameters
and that fitted probabilities are in [0, 1].

Run with::

    pytest tests/test_spatial_probit.py -m slow -v
"""

from __future__ import annotations

import numpy as np
import pytest

from neighbayes.models import SEMProbit
from neighbayes.tests.helpers import (
    SAMPLE_KWARGS,
    W_to_graph,
    make_line_W,
    make_spatial_probit_data,
)

pytestmark = [pytest.mark.slow, pytest.mark.recovery]

# Data generating process parameters
M_REGIONS = 10
N_PER_REGION = 25
RHO_TRUE = 0.35
BETA_TRUE = np.array([0.3, 1.0])
SIGMA_A_TRUE = 0.8

ABS_TOL_RHO = 0.30
ABS_TOL_BETA = 0.45
ABS_TOL_SIGMA_A = 0.40


def test_spatialprobit_recovery_and_fitted_probs(rng):
    W_dense = make_line_W(M_REGIONS)
    W_graph = W_to_graph(W_dense)
    y, X, region_ids = make_spatial_probit_data(
        rng,
        W_dense,
        rho=RHO_TRUE,
        beta=BETA_TRUE,
        sigma_a=SIGMA_A_TRUE,
        n_per_region=N_PER_REGION,
    )
    model = SEMProbit(group_effects="random", y=y, X=X, W=W_graph, groups=region_ids)
    idata = model.fit(**SAMPLE_KWARGS)

    rho_hat = float(idata.posterior["lam"].mean())
    assert abs(rho_hat - RHO_TRUE) < ABS_TOL_RHO, (
        f"SEMProbit lam: expected ≈{RHO_TRUE}, got {rho_hat:.3f}"
    )

    beta_hat = idata.posterior["beta"].mean(("chain", "draw")).values
    for j, (bhat, btrue) in enumerate(zip(beta_hat, BETA_TRUE)):
        assert abs(bhat - btrue) < ABS_TOL_BETA, (
            f"SEMProbit beta[{j}]: expected ≈{btrue}, got {bhat:.3f}"
        )

    sa_hat = float(idata.posterior["group_effect_sd"].mean())
    assert abs(sa_hat - SIGMA_A_TRUE) < ABS_TOL_SIGMA_A, (
        f"SEMProbit sigma_a: expected ≈{SIGMA_A_TRUE}, got {sa_hat:.3f}"
    )

    p_hat = model.fitted_probabilities()
    assert np.all((p_hat >= 0.0) & (p_hat <= 1.0))
