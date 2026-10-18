"""Parameter recovery tests for dynamic panel model classes.

Each test generates balanced panel data from known parameters, fits the
model once, and asserts that **all** posterior means are within tolerance
of the true values.  This is a deliberate one-fit-per-model design to
minimize MCMC wall time; the per-parameter splits this file used to
maintain were equivalent assertions on independent refits and added cost
without coverage.

**Design notes**

Dynamic panel models with ``effects=1`` (unit FE demeaning) suffer from
Nickell bias: the demeaned lagged dependent variable is correlated with
the demeaned error, biasing φ toward zero.  To obtain clean parameter
recovery, we use ``effects=0`` (pooled) and generate data **without** unit
effects (``sigma_alpha=0``), so the DGP matches the model specification
exactly.

Run with::

    pytest tests/test_panel_dynamic_recovery.py -m slow -v
"""

from __future__ import annotations

import numpy as np
import pytest

from neighbayes.models import OLS, SAR, SDEM, SDM, SEM, SLX
from neighbayes.models.panel._dynamic import (
    OLSPanelDynamic,
    SARPanelDynamic,
    SDEMPanelDynamic,
    SDMPanelDynamic,
    SDMPanelDynamicRestricted,
    SEMPanelDynamic,
    SLXPanelDynamic,
)
from neighbayes.tests.helpers import (
    PANEL_N,
    PANEL_T,
    SAMPLE_KWARGS,
    make_panel_dlm_data,
    make_panel_sar_dynamic_data,
    make_panel_sdem_dynamic_data,
    make_panel_sdmr_data,
    make_panel_sdmu_data,
    make_panel_sem_dynamic_data,
    make_panel_slx_dynamic_data,
)

pytestmark = [pytest.mark.slow, pytest.mark.recovery]

# True parameters
PHI_TRUE = 0.4
RHO_TRUE = 0.3
LAM_TRUE = 0.3
THETA_TRUE = -0.1
BETA_TRUE = np.array([1.0, 2.0])
SIGMA_TRUE = 1.0
# sigma_alpha=0 so DGP matches the pooled (effects=0) specification exactly
SIGMA_ALPHA_TRUE = 0.0

# Tolerances — dynamic panels with pooled specification
ABS_TOL_PHI = 0.25
ABS_TOL_SPATIAL = 0.35
ABS_TOL_BETA = 0.50
ABS_TOL_THETA = 0.40


def _fit(model_cls, y, X, W_panel_graph):
    model = model_cls(y=y, X=X, W=W_panel_graph, N=PANEL_N, T=PANEL_T, effects=0)
    return model.fit(**SAMPLE_KWARGS)


def _assert_scalar(idata, name, true, tol, label):
    hat = float(idata.posterior[name].mean())
    assert abs(hat - true) < tol, f"{label} {name}: expected ≈{true}, got {hat:.3f}"


def _assert_beta(idata, label):
    beta_hat = idata.posterior["beta"].mean(("chain", "draw")).values
    for j, (bhat, btrue) in enumerate(zip(beta_hat, BETA_TRUE)):
        assert abs(bhat - btrue) < ABS_TOL_BETA, (
            f"{label} beta[{j}]: expected ≈{btrue}, got {bhat:.3f}"
        )


# ---------------------------------------------------------------------------
# DLM Panel  (non-spatial dynamic)
# ---------------------------------------------------------------------------


def test_dlm_panel_fe_recovers_phi_and_beta(rng, W_panel_dense, W_panel_graph):
    y, X, _ = make_panel_dlm_data(
        rng,
        W_panel_dense,
        PANEL_N,
        PANEL_T,
        phi=PHI_TRUE,
        beta=BETA_TRUE,
        sigma=SIGMA_TRUE,
        sigma_alpha=SIGMA_ALPHA_TRUE,
    )
    idata = _fit(OLSPanelDynamic, y, X, W_panel_graph)
    _assert_scalar(idata, "phi", PHI_TRUE, ABS_TOL_PHI, "OLSPanelDynamic")
    _assert_beta(idata, "OLSPanelDynamic")


# ---------------------------------------------------------------------------
# SDMR Panel  (restricted SDM dynamic)
# ---------------------------------------------------------------------------


def test_sdmr_panel_fe_recovers_all(rng, W_panel_dense, W_panel_graph):
    y, X, _ = make_panel_sdmr_data(
        rng,
        W_panel_dense,
        PANEL_N,
        PANEL_T,
        rho=RHO_TRUE,
        phi=PHI_TRUE,
        beta=BETA_TRUE,
        sigma=SIGMA_TRUE,
        sigma_alpha=SIGMA_ALPHA_TRUE,
    )
    idata = _fit(SDMPanelDynamicRestricted, y, X, W_panel_graph)
    _assert_scalar(idata, "rho", RHO_TRUE, ABS_TOL_SPATIAL, "SDMPanelDynamicRestricted")
    _assert_scalar(idata, "phi", PHI_TRUE, ABS_TOL_PHI, "SDMPanelDynamicRestricted")
    _assert_beta(idata, "SDMPanelDynamicRestricted")


# ---------------------------------------------------------------------------
# SDMU Panel  (unrestricted SDM dynamic)
# ---------------------------------------------------------------------------


def test_sdmu_panel_fe_recovers_all(rng, W_panel_dense, W_panel_graph):
    y, X, _ = make_panel_sdmu_data(
        rng,
        W_panel_dense,
        PANEL_N,
        PANEL_T,
        rho=RHO_TRUE,
        phi=PHI_TRUE,
        theta=THETA_TRUE,
        beta=BETA_TRUE,
        sigma=SIGMA_TRUE,
        sigma_alpha=SIGMA_ALPHA_TRUE,
    )
    idata = _fit(SDMPanelDynamic, y, X, W_panel_graph)
    _assert_scalar(idata, "rho", RHO_TRUE, ABS_TOL_SPATIAL, "SDMPanelDynamic")
    _assert_scalar(idata, "phi", PHI_TRUE, ABS_TOL_PHI, "SDMPanelDynamic")
    _assert_scalar(idata, "theta", THETA_TRUE, ABS_TOL_THETA, "SDMPanelDynamic")
    _assert_beta(idata, "SDMPanelDynamic")


# ---------------------------------------------------------------------------
# SAR Panel DE Dynamic  (SAR with lagged DV)
# ---------------------------------------------------------------------------


def test_sar_panel_de_dynamic_recovers_all(rng, W_panel_dense, W_panel_graph):
    y, X, _ = make_panel_sar_dynamic_data(
        rng,
        W_panel_dense,
        PANEL_N,
        PANEL_T,
        rho=RHO_TRUE,
        phi=PHI_TRUE,
        beta=BETA_TRUE,
        sigma=SIGMA_TRUE,
        sigma_alpha=SIGMA_ALPHA_TRUE,
    )
    idata = _fit(SARPanelDynamic, y, X, W_panel_graph)
    _assert_scalar(idata, "rho", RHO_TRUE, ABS_TOL_SPATIAL, "SARPanelDynamic")
    _assert_scalar(idata, "phi", PHI_TRUE, ABS_TOL_PHI, "SARPanelDynamic")
    _assert_beta(idata, "SARPanelDynamic")


# ---------------------------------------------------------------------------
# SEM Panel DE Dynamic
# ---------------------------------------------------------------------------


def test_sem_panel_de_dynamic_recovers_all(rng, W_panel_dense, W_panel_graph):
    y, X, _ = make_panel_sem_dynamic_data(
        rng,
        W_panel_dense,
        PANEL_N,
        PANEL_T,
        lam=LAM_TRUE,
        phi=PHI_TRUE,
        beta=BETA_TRUE,
        sigma=SIGMA_TRUE,
        sigma_alpha=SIGMA_ALPHA_TRUE,
    )
    idata = _fit(SEMPanelDynamic, y, X, W_panel_graph)
    _assert_scalar(idata, "lam", LAM_TRUE, ABS_TOL_SPATIAL, "SEMPanelDynamic")
    _assert_scalar(idata, "phi", PHI_TRUE, ABS_TOL_PHI, "SEMPanelDynamic")
    _assert_beta(idata, "SEMPanelDynamic")


# ---------------------------------------------------------------------------
# SDEM Panel DE Dynamic
# ---------------------------------------------------------------------------


def test_sdem_panel_de_dynamic_recovers_all(rng, W_panel_dense, W_panel_graph):
    y, X, _ = make_panel_sdem_dynamic_data(
        rng,
        W_panel_dense,
        PANEL_N,
        PANEL_T,
        lam=LAM_TRUE,
        phi=PHI_TRUE,
        beta=BETA_TRUE,
        sigma=SIGMA_TRUE,
        sigma_alpha=SIGMA_ALPHA_TRUE,
    )
    idata = _fit(SDEMPanelDynamic, y, X, W_panel_graph)
    _assert_scalar(idata, "lam", LAM_TRUE, ABS_TOL_SPATIAL, "SDEMPanelDynamic")
    _assert_scalar(idata, "phi", PHI_TRUE, ABS_TOL_PHI, "SDEMPanelDynamic")
    _assert_beta(idata, "SDEMPanelDynamic")


# ---------------------------------------------------------------------------
# SLX Panel DE Dynamic
# ---------------------------------------------------------------------------


def test_slx_panel_de_dynamic_recovers_phi_and_beta(rng, W_panel_dense, W_panel_graph):
    y, X, _ = make_panel_slx_dynamic_data(
        rng,
        W_panel_dense,
        PANEL_N,
        PANEL_T,
        phi=PHI_TRUE,
        beta=BETA_TRUE,
        sigma=SIGMA_TRUE,
        sigma_alpha=SIGMA_ALPHA_TRUE,
    )
    idata = _fit(SLXPanelDynamic, y, X, W_panel_graph)
    _assert_scalar(idata, "phi", PHI_TRUE, ABS_TOL_PHI, "SLXPanelDynamic")
    _assert_beta(idata, "SLXPanelDynamic")
