"""Parameter recovery tests for panel spatial Tobit models."""

from __future__ import annotations

import numpy as np
import pytest

from neighbayes.models import SARTobit, SEMTobit
from neighbayes.tests.helpers import (
    PANEL_N,
    PANEL_T,
    SAMPLE_KWARGS,
    make_panel_sar_tobit_data,
    make_panel_sem_tobit_data,
)

pytestmark = [pytest.mark.slow, pytest.mark.recovery]

RHO_TRUE = 0.35
LAM_TRUE = 0.35
BETA_TRUE = np.array([1.0, 1.4])
SIGMA_TRUE = 0.8
CENSOR_TRUE = 0.0

ABS_TOL_SPATIAL = 0.35
ABS_TOL_BETA = 0.60


def test_sar_panel_tobit_recovers_rho_and_beta(rng, W_panel_dense, W_panel_graph):
    y, X = make_panel_sar_tobit_data(
        rng,
        W_panel_dense,
        PANEL_N,
        PANEL_T,
        rho=RHO_TRUE,
        beta=BETA_TRUE,
        sigma=SIGMA_TRUE,
        censoring=CENSOR_TRUE,
    )
    model = SARTobit(
        y=y, X=X, W=W_panel_graph, N=PANEL_N, T=PANEL_T, censoring=CENSOR_TRUE
    )
    idata = model.fit(**SAMPLE_KWARGS)

    rho_hat = float(idata.posterior["rho"].mean())
    assert abs(rho_hat - RHO_TRUE) < ABS_TOL_SPATIAL

    beta_hat = idata.posterior["beta"].mean(("chain", "draw")).values
    for bhat, btrue in zip(beta_hat, BETA_TRUE):
        assert abs(bhat - btrue) < ABS_TOL_BETA


def test_sem_panel_tobit_recovers_lam_and_beta(rng, W_panel_dense, W_panel_graph):
    y, X = make_panel_sem_tobit_data(
        rng,
        W_panel_dense,
        PANEL_N,
        PANEL_T,
        lam=LAM_TRUE,
        beta=BETA_TRUE,
        sigma=SIGMA_TRUE,
        censoring=CENSOR_TRUE,
    )
    model = SEMTobit(
        y=y, X=X, W=W_panel_graph, N=PANEL_N, T=PANEL_T, censoring=CENSOR_TRUE
    )
    idata = model.fit(**SAMPLE_KWARGS)

    lam_hat = float(idata.posterior["lam"].mean())
    assert abs(lam_hat - LAM_TRUE) < ABS_TOL_SPATIAL

    beta_hat = idata.posterior["beta"].mean(("chain", "draw")).values
    for bhat, btrue in zip(beta_hat, BETA_TRUE):
        assert abs(bhat - btrue) < ABS_TOL_BETA
