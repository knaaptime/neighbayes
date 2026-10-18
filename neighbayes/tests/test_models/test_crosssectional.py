"""Parameter recovery tests for cross-sectional spatial models.

Each test generates synthetic data from known parameters, fits the model
once, and asserts that **all** posterior means are within tolerance of the
true values.  One fit per model — per-parameter splits were independent
refits that duplicated MCMC cost without adding coverage.

Run with::

    pytest tests/test_crosssectional.py -m slow -v
"""

from __future__ import annotations

import numpy as np
import pytest

from neighbayes.models import SAR, SDEM, SDM, SEM, SLX
from neighbayes.tests.helpers import (
    SAMPLE_KWARGS,
    make_sar_data,
    make_sdem_data,
    make_sdm_data,
    make_sem_data,
    make_slx_data,
)

pytestmark = [pytest.mark.slow, pytest.mark.recovery]

# True parameters used across all cross-sectional tests
RHO_TRUE = 0.5
LAM_TRUE = 0.5
BETA_TRUE = np.array([1.0, 2.0])
BETA2_TRUE = np.array([0.8])  # spatially-lagged X coefficient (SLX/SDM/SDEM)
SIGMA_TRUE = 0.8

# Recovery tolerance: posterior mean must be within this absolute distance
# of the true value.  At n=225 (SIDE=15) Gaussian Gibbs recovers tightly:
# observed max errors ~0.05 (spatial), ~0.10 (β), ~0.15 (WX).
ABS_TOL_SPATIAL = 0.10  # for rho / lambda
ABS_TOL_BETA = 0.15  # for regression coefficients
ABS_TOL_WX = 0.20  # for lagged-X coefficients


def _assert_beta(idata, true_beta, label, tol_wx=ABS_TOL_WX):
    beta_hat = idata.posterior["beta"].mean(("chain", "draw")).values
    n_x = len(BETA_TRUE)
    for j, (bhat, btrue) in enumerate(zip(beta_hat, true_beta)):
        tol = tol_wx if j >= n_x else ABS_TOL_BETA
        assert abs(bhat - btrue) < tol, (
            f"{label} beta[{j}]: expected ≈{btrue}, got {bhat:.3f}"
        )


def _assert_scalar(idata, name, true, tol, label):
    hat = float(idata.posterior[name].mean())
    assert abs(hat - true) < tol, f"{label} {name}: expected ≈{true}, got {hat:.3f}"


def test_sar_recovers_rho_and_beta(rng, W_dense, W_graph):
    y, X = make_sar_data(rng, W_dense, rho=RHO_TRUE, beta=BETA_TRUE, sigma=SIGMA_TRUE)
    idata = SAR(y=y, X=X, W=W_graph).fit(**SAMPLE_KWARGS)
    _assert_scalar(idata, "rho", RHO_TRUE, ABS_TOL_SPATIAL, "SAR")
    _assert_beta(idata, BETA_TRUE, "SAR")


def test_sem_recovers_lam_and_beta(rng, W_dense, W_graph):
    y, X = make_sem_data(rng, W_dense, lam=LAM_TRUE, beta=BETA_TRUE, sigma=SIGMA_TRUE)
    idata = SEM(y=y, X=X, W=W_graph).fit(**SAMPLE_KWARGS)
    _assert_scalar(idata, "lam", LAM_TRUE, ABS_TOL_SPATIAL, "SEM")
    _assert_beta(idata, BETA_TRUE, "SEM")


def test_slx_recovers_beta(rng, W_dense, W_graph):
    y, X = make_slx_data(
        rng, W_dense, beta1=BETA_TRUE, beta2=BETA2_TRUE, sigma=SIGMA_TRUE
    )
    idata = SLX(y=y, X=X, W=W_graph).fit(**SAMPLE_KWARGS)
    combined = np.concatenate([BETA_TRUE, BETA2_TRUE])
    _assert_beta(idata, combined, "SLX")


def test_sdm_recovers_rho_and_beta(rng, W_dense, W_graph):
    y, X = make_sdm_data(
        rng, W_dense, rho=RHO_TRUE, beta1=BETA_TRUE, beta2=BETA2_TRUE, sigma=SIGMA_TRUE
    )
    idata = SDM(y=y, X=X, W=W_graph).fit(**SAMPLE_KWARGS)
    _assert_scalar(idata, "rho", RHO_TRUE, ABS_TOL_SPATIAL, "SDM")
    combined = np.concatenate([BETA_TRUE, BETA2_TRUE])
    _assert_beta(idata, combined, "SDM")


def test_sdem_recovers_lam_and_beta(rng, W_dense, W_graph):
    y, X = make_sdem_data(
        rng, W_dense, lam=LAM_TRUE, beta1=BETA_TRUE, beta2=BETA2_TRUE, sigma=SIGMA_TRUE
    )
    idata = SDEM(y=y, X=X, W=W_graph).fit(**SAMPLE_KWARGS)
    _assert_scalar(idata, "lam", LAM_TRUE, ABS_TOL_SPATIAL, "SDEM")
    combined = np.concatenate([BETA_TRUE, BETA2_TRUE])
    _assert_beta(idata, combined, "SDEM")


# ---------------------------------------------------------------------------
# Parameterized backend recovery: each model is fit on both numpy and jax
# backends to ensure the Gibbs sampler produces equivalent posteriors.
# SLX has no spatial parameter, so it is excluded from the spatial-param
# check but still tested for β recovery on both backends.
# ---------------------------------------------------------------------------

BACKENDS = ["numpy", "jax"]


def _backend_kwargs(backend):
    kw = dict(SAMPLE_KWARGS)
    kw["backend"] = backend
    if backend == "numpy":
        kw["n_jobs"] = 1
    return kw


@pytest.mark.parametrize("backend", BACKENDS)
def test_sar_backend_recovery(rng, W_dense, W_graph, backend):
    y, X = make_sar_data(rng, W_dense, rho=RHO_TRUE, beta=BETA_TRUE, sigma=SIGMA_TRUE)
    idata = SAR(y=y, X=X, W=W_graph).fit(**_backend_kwargs(backend))
    _assert_scalar(idata, "rho", RHO_TRUE, ABS_TOL_SPATIAL, f"SAR[{backend}]")
    _assert_beta(idata, BETA_TRUE, f"SAR[{backend}]")


@pytest.mark.parametrize("backend", BACKENDS)
def test_sem_backend_recovery(rng, W_dense, W_graph, backend):
    y, X = make_sem_data(rng, W_dense, lam=LAM_TRUE, beta=BETA_TRUE, sigma=SIGMA_TRUE)
    idata = SEM(y=y, X=X, W=W_graph).fit(**_backend_kwargs(backend))
    _assert_scalar(idata, "lam", LAM_TRUE, ABS_TOL_SPATIAL, f"SEM[{backend}]")
    _assert_beta(idata, BETA_TRUE, f"SEM[{backend}]")


def test_slx_recovery(rng, W_dense, W_graph):
    """SLX has no Gibbs sampler, so no Gibbs backend to vary: NUTS once."""
    y, X = make_slx_data(
        rng, W_dense, beta1=BETA_TRUE, beta2=BETA2_TRUE, sigma=SIGMA_TRUE
    )
    idata = SLX(y=y, X=X, W=W_graph).fit(**SAMPLE_KWARGS)
    combined = np.concatenate([BETA_TRUE, BETA2_TRUE])
    _assert_beta(idata, combined, "SLX")


@pytest.mark.parametrize("backend", BACKENDS)
def test_sdm_backend_recovery(rng, W_dense, W_graph, backend):
    y, X = make_sdm_data(
        rng, W_dense, rho=RHO_TRUE, beta1=BETA_TRUE, beta2=BETA2_TRUE, sigma=SIGMA_TRUE
    )
    idata = SDM(y=y, X=X, W=W_graph).fit(**_backend_kwargs(backend))
    _assert_scalar(idata, "rho", RHO_TRUE, ABS_TOL_SPATIAL, f"SDM[{backend}]")
    combined = np.concatenate([BETA_TRUE, BETA2_TRUE])
    _assert_beta(idata, combined, f"SDM[{backend}]")


@pytest.mark.parametrize("backend", BACKENDS)
def test_sdem_backend_recovery(rng, W_dense, W_graph, backend):
    y, X = make_sdem_data(
        rng, W_dense, lam=LAM_TRUE, beta1=BETA_TRUE, beta2=BETA2_TRUE, sigma=SIGMA_TRUE
    )
    idata = SDEM(y=y, X=X, W=W_graph).fit(**_backend_kwargs(backend))
    _assert_scalar(idata, "lam", LAM_TRUE, ABS_TOL_SPATIAL, f"SDEM[{backend}]")
    combined = np.concatenate([BETA_TRUE, BETA2_TRUE])
    _assert_beta(idata, combined, f"SDEM[{backend}]")
