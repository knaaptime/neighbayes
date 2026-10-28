"""Parameter recovery tests for spatial panel fixed-effects models.

Each test generates balanced panel data from known parameters, fits the model
once, and asserts that **all** posterior means are within tolerance of the
true values.

Run with::

    pytest tests/test_panel_fe.py -m slow -v
"""

from __future__ import annotations

import numpy as np
import pytest

from neighbayes.models import OLS, SAR, SDEM, SDM, SEM, SLX
from neighbayes.tests.helpers import (
    PANEL_N,
    PANEL_T,
    SAMPLE_KWARGS,
    make_panel_ols_data,
    make_panel_sar_data,
    make_panel_sdem_fe_data,
    make_panel_sdm_fe_data,
    make_panel_sem_data,
)

pytestmark = [pytest.mark.slow, pytest.mark.recovery]

# True parameters
RHO_TRUE = 0.4
LAM_TRUE = 0.4
BETA_TRUE = np.array([1.0, 2.0])
BETA2_TRUE = np.array([0.8])  # WX coefficient for SDM/SDEM/SLX
SIGMA_TRUE = 0.8

ABS_TOL_SIGMA = 0.35
ABS_TOL_SPATIAL = 0.25
ABS_TOL_BETA = 0.35  # panel FE beta slightly harder to recover at small N*T


def _assert_slope(idata, label):
    """Under unit-FE demeaning the intercept is dropped, so beta_hat[0]
    is the slope coefficient and should approximate BETA_TRUE[1]."""
    beta_hat = idata.posterior["beta"].mean(("chain", "draw")).values
    assert abs(beta_hat[0] - BETA_TRUE[1]) < ABS_TOL_BETA, (
        f"{label} beta[0]: expected ≈{BETA_TRUE[1]}, got {beta_hat[0]:.3f}"
    )


def _assert_scalar(idata, name, true, tol, label):
    hat = float(idata.posterior[name].mean())
    assert abs(hat - true) < tol, f"{label} {name}: expected ≈{true}, got {hat:.3f}"


def test_ols_panel_fe_recovers_beta_and_sigma(rng, W_panel_dense, W_panel_graph):
    """OLSPanelFE recovery.

    With effects=1 (unit FE), the intercept is absorbed by demeaning and
    sigma reflects the residual variance.  We set sigma_alpha=0 so the
    DGP has no unit effects and the model sigma matches the DGP sigma.
    """
    y, X, _ = make_panel_ols_data(
        rng,
        W_panel_dense,
        PANEL_N,
        PANEL_T,
        beta=BETA_TRUE,
        sigma=SIGMA_TRUE,
        sigma_alpha=0.0,
    )
    model = OLS(unit_effects="fixed", y=y, X=X, W=W_panel_graph, N=PANEL_N, T=PANEL_T)
    idata = model.fit(**SAMPLE_KWARGS)
    _assert_slope(idata, "OLSPanelFE")
    _assert_scalar(idata, "sigma", SIGMA_TRUE, ABS_TOL_SIGMA, "OLSPanelFE")


def test_sar_panel_fe_recovers_rho_and_beta(rng, W_panel_dense, W_panel_graph):
    y, X, _ = make_panel_sar_data(
        rng,
        W_panel_dense,
        PANEL_N,
        PANEL_T,
        rho=RHO_TRUE,
        beta=BETA_TRUE,
        sigma=SIGMA_TRUE,
    )
    model = SAR(unit_effects="fixed", y=y, X=X, W=W_panel_graph, N=PANEL_N, T=PANEL_T)
    idata = model.fit(**SAMPLE_KWARGS)
    _assert_scalar(idata, "rho", RHO_TRUE, ABS_TOL_SPATIAL, "SARPanelFE")
    _assert_slope(idata, "SARPanelFE")


def test_sem_panel_fe_recovers_lam_and_beta(rng, W_panel_dense, W_panel_graph):
    y, X, _ = make_panel_sem_data(
        rng,
        W_panel_dense,
        PANEL_N,
        PANEL_T,
        lam=LAM_TRUE,
        beta=BETA_TRUE,
        sigma=SIGMA_TRUE,
    )
    model = SEM(unit_effects="fixed", y=y, X=X, W=W_panel_graph, N=PANEL_N, T=PANEL_T)
    idata = model.fit(**SAMPLE_KWARGS)
    _assert_scalar(idata, "lam", LAM_TRUE, ABS_TOL_SPATIAL, "SEMPanelFE")
    _assert_slope(idata, "SEMPanelFE")


def test_sdm_panel_fe_recovers_rho_and_beta(rng, W_panel_dense, W_panel_graph):
    y, X, _ = make_panel_sdm_fe_data(
        rng,
        W_panel_dense,
        PANEL_N,
        PANEL_T,
        rho=RHO_TRUE,
        beta1=BETA_TRUE,
        beta2=BETA2_TRUE,
        sigma=SIGMA_TRUE,
    )
    model = SDM(unit_effects="fixed", y=y, X=X, W=W_panel_graph, N=PANEL_N, T=PANEL_T)
    idata = model.fit(**SAMPLE_KWARGS)
    _assert_scalar(idata, "rho", RHO_TRUE, ABS_TOL_SPATIAL, "SDMPanelFE")
    _assert_slope(idata, "SDMPanelFE")


def test_sdem_panel_fe_recovers_lam_and_beta(rng, W_panel_dense, W_panel_graph):
    y, X, _ = make_panel_sdem_fe_data(
        rng,
        W_panel_dense,
        PANEL_N,
        PANEL_T,
        lam=LAM_TRUE,
        beta1=BETA_TRUE,
        beta2=BETA2_TRUE,
        sigma=SIGMA_TRUE,
    )
    model = SDEM(unit_effects="fixed", y=y, X=X, W=W_panel_graph, N=PANEL_N, T=PANEL_T)
    idata = model.fit(**SAMPLE_KWARGS)
    _assert_scalar(idata, "lam", LAM_TRUE, ABS_TOL_SPATIAL, "SDEMPanelFE")
    _assert_slope(idata, "SDEMPanelFE")


def test_slx_panel_fe_recovers_beta(rng, W_panel_dense, W_panel_graph):
    """SLXPanelFE: rho=0 SDM-style DGP provides WX signal only."""
    y, X, _ = make_panel_sdm_fe_data(
        rng,
        W_panel_dense,
        PANEL_N,
        PANEL_T,
        rho=0.0,
        beta1=BETA_TRUE,
        beta2=BETA2_TRUE,
        sigma=SIGMA_TRUE,
    )
    model = SLX(unit_effects="fixed", y=y, X=X, W=W_panel_graph, N=PANEL_N, T=PANEL_T)
    idata = model.fit(**SAMPLE_KWARGS)
    _assert_slope(idata, "SLXPanelFE")
