"""Parameter recovery tests for spatial panel random-effects models.

Each test generates balanced panel data (including unit random effects),
fits the RE model once, and asserts that **all** posterior means are within
tolerance of the true values.

Run with::

    pytest tests/test_panel_re.py -m slow -v
"""

from __future__ import annotations

import numpy as np
import pytest

from neighbayes.models import OLS, SAR, SDEM, SEM
from neighbayes.tests.helpers import (
    PANEL_N,
    PANEL_T,
    SAMPLE_KWARGS,
    make_panel_ols_data,
    make_panel_sar_data,
    make_panel_sdem_fe_data,
    make_panel_sem_data,
)

pytestmark = [pytest.mark.slow, pytest.mark.recovery]

# True parameters
RHO_TRUE = 0.4
LAM_TRUE = 0.4
BETA_TRUE = np.array([1.0, 2.0])
BETA2_TRUE = np.array([0.8])  # WX coefficient for SDEM
SIGMA_TRUE = 0.8
SIGMA_ALPHA_TRUE = 0.5

ABS_TOL_SPATIAL = 0.25
ABS_TOL_BETA = 0.35
ABS_TOL_SIGMA_ALPHA = 0.30
# SEM-RE has weak identification between λ and σ_α² — the random
# effects absorb spatial correlation, so σ_α² recovery is inherently
# noisy.  Use a wider tolerance for the SEM-RE σ_α² test.
ABS_TOL_SIGMA_ALPHA_SEM_RE = 0.40


def _assert_beta(idata, label):
    beta_hat = idata.posterior["beta"].mean(("chain", "draw")).values
    for j, (bhat, btrue) in enumerate(zip(beta_hat, BETA_TRUE)):
        assert abs(bhat - btrue) < ABS_TOL_BETA, (
            f"{label} beta[{j}]: expected ≈{btrue}, got {bhat:.3f}"
        )


def _assert_scalar(idata, name, true, tol, label):
    hat = float(idata.posterior[name].mean())
    assert abs(hat - true) < tol, f"{label} {name}: expected ≈{true}, got {hat:.3f}"


def _assert_sdem_beta(idata, label):
    """SDEM beta covers [X coefficients, WX coefficients]."""
    beta_hat = idata.posterior["beta"].mean(("chain", "draw")).values
    combined = np.concatenate([BETA_TRUE, BETA2_TRUE])
    for j, (bhat, btrue) in enumerate(zip(beta_hat, combined)):
        assert abs(bhat - btrue) < ABS_TOL_BETA, (
            f"{label} beta[{j}]: expected ≈{btrue}, got {bhat:.3f}"
        )


def test_ols_panel_re_recovers_all(rng, W_panel_dense, W_panel_graph):
    y, X, _ = make_panel_ols_data(
        rng,
        W_panel_dense,
        PANEL_N,
        PANEL_T,
        beta=BETA_TRUE,
        sigma=SIGMA_TRUE,
        sigma_alpha=SIGMA_ALPHA_TRUE,
    )
    model = OLS(unit_effects="random", y=y, X=X, W=W_panel_graph, N=PANEL_N, T=PANEL_T)
    idata = model.fit(**SAMPLE_KWARGS)
    _assert_beta(idata, "OLSPanelRE")
    _assert_scalar(
        idata, "sigma_alpha", SIGMA_ALPHA_TRUE, ABS_TOL_SIGMA_ALPHA, "OLSPanelRE"
    )


def test_sar_panel_re_recovers_all(rng, W_panel_dense, W_panel_graph):
    y, X, _ = make_panel_sar_data(
        rng,
        W_panel_dense,
        PANEL_N,
        PANEL_T,
        rho=RHO_TRUE,
        beta=BETA_TRUE,
        sigma=SIGMA_TRUE,
        sigma_alpha=SIGMA_ALPHA_TRUE,
    )
    model = SAR(unit_effects="random", y=y, X=X, W=W_panel_graph, N=PANEL_N, T=PANEL_T)
    idata = model.fit(**SAMPLE_KWARGS)
    _assert_scalar(idata, "rho", RHO_TRUE, ABS_TOL_SPATIAL, "SARPanelRE")
    _assert_beta(idata, "SARPanelRE")
    _assert_scalar(
        idata, "sigma_alpha", SIGMA_ALPHA_TRUE, ABS_TOL_SIGMA_ALPHA, "SARPanelRE"
    )


def test_sem_panel_re_recovers_all(rng, W_panel_dense, W_panel_graph):
    """SEMPanelRE posterior mean recovery.

    NOTE: σ_α² is weakly identified in SEM-RE models because the random
    effects absorb spatial correlation that λ would otherwise capture.
    The posterior for σ_α² is diffuse and biased downward even with
    correct samplers — this is a model identification issue, not a bug.
    A wider tolerance is used to reflect this inherent difficulty.
    """
    y, X, _ = make_panel_sem_data(
        rng,
        W_panel_dense,
        PANEL_N,
        PANEL_T,
        lam=LAM_TRUE,
        beta=BETA_TRUE,
        sigma=SIGMA_TRUE,
        sigma_alpha=SIGMA_ALPHA_TRUE,
    )
    model = SEM(unit_effects="random", y=y, X=X, W=W_panel_graph, N=PANEL_N, T=PANEL_T)
    idata = model.fit(**SAMPLE_KWARGS)
    _assert_scalar(idata, "lam", LAM_TRUE, ABS_TOL_SPATIAL, "SEMPanelRE")
    _assert_beta(idata, "SEMPanelRE")
    _assert_scalar(
        idata, "sigma_alpha", SIGMA_ALPHA_TRUE, ABS_TOL_SIGMA_ALPHA_SEM_RE, "SEMPanelRE"
    )


def test_sdem_panel_re_recovers_all(rng, W_panel_dense, W_panel_graph):
    """SDEMPanelRE posterior mean recovery.

    Like SEM-RE, σ_α² is weakly identified because random effects absorb
    spatial correlation.  The beta vector includes both X and WX
    coefficients.
    """
    y, X, _ = make_panel_sdem_fe_data(
        rng,
        W_panel_dense,
        PANEL_N,
        PANEL_T,
        lam=LAM_TRUE,
        beta1=BETA_TRUE,
        beta2=BETA2_TRUE,
        sigma=SIGMA_TRUE,
        sigma_alpha=SIGMA_ALPHA_TRUE,
    )
    model = SDEM(unit_effects="random", y=y, X=X, W=W_panel_graph, N=PANEL_N, T=PANEL_T)
    idata = model.fit(**SAMPLE_KWARGS)
    _assert_scalar(idata, "lam", LAM_TRUE, ABS_TOL_SPATIAL, "SDEMPanelRE")
    _assert_sdem_beta(idata, "SDEMPanelRE")
    _assert_scalar(
        idata,
        "sigma_alpha",
        SIGMA_ALPHA_TRUE,
        ABS_TOL_SIGMA_ALPHA_SEM_RE,
        "SDEMPanelRE",
    )
