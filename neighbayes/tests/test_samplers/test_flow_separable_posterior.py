"""The separable NB flow sampler must match an independent reference posterior.

The production sampler slices the exact NB2 likelihood
(:mod:`neighbayes.samplers.negbin_reduced._flow_exact`); the reference is
Pólya–Gamma Gibbs with exact Kronecker solves at every slice candidate
(:func:`neighbayes.tests.helpers.separable_flow_pg_reference`).  The two share
no block, so their agreement on the posterior means and SDs of ρ, on both
backends, for the cross-section and a panel, checks each.  (A sampler that once
reused a ρ Krylov basis across sweeps got the means right and the SDs 10–20%
too wide.)
"""

from __future__ import annotations

import arviz as az
import numpy as np
import pytest

from neighbayes.dgp import simulate_sar_negbin_flow
from neighbayes.models import SARNegBinFlow
from neighbayes.models._base._shared import gelman_default_beta_prior
from neighbayes.models.priors import FlowReducedGibbsPriors
from neighbayes.samplers.negbin_reduced._flow import FlowReducedGibbsState
from neighbayes.tests.helpers import separable_flow_pg_reference

N_REGIONS = 25
DRAWS, TUNE, CHAINS = 4000, 800, 4


@pytest.fixture(scope="module")
def data():
    return simulate_sar_negbin_flow(
        n=N_REGIONS, rho_d=0.35, rho_o=0.25, seed=3, separable=True
    )


def _model(data):
    y = data["y_vec"] if "y_vec" in data else data["y"]
    return SARNegBinFlow(
        y, data["X"], data["G"], separable=True, col_names=data["col_names"]
    )


def _pg_reference(m, n, T=1):
    """The Pólya–Gamma reference: exact Kronecker solves at every slice candidate."""
    mu, sd = gelman_default_beta_prior(m._y, m._X, list(m._feature_names))
    priors = FlowReducedGibbsPriors(
        beta_mu=mu,
        beta_sigma=sd,
        alpha_sigma=2.5,
        alpha_nu=3.0,
        rho_lower=-0.999,
        rho_upper=0.999,
    )
    W = m._W_sparse.tocsc()
    k = m._X.shape[1]
    draws = {"rho_d": [], "rho_o": []}
    for c in range(CHAINS):
        rng = np.random.default_rng(1000 + c)
        init = FlowReducedGibbsState(
            beta=rng.normal(0, 0.1, k),
            rho_d=rng.uniform(-0.1, 0.1),
            rho_o=rng.uniform(-0.1, 0.1),
            rho_w=None,
            alpha=1.0,
            omega=np.full(n * n * T, 0.5),
        )
        out = separable_flow_pg_reference(
            m._y_int_vec.astype(float), m._X, W, n, priors, init, DRAWS, TUNE, rng, T
        )
        for name in draws:
            draws[name].append(out[name])
    return {name: np.concatenate(v) for name, v in draws.items()}


def _assert_matches(idata, reference):
    for name in ("rho_d", "rho_o"):
        got = idata.posterior[name].values.ravel()
        ref = reference[name]
        ess = float(az.ess(idata, var_names=[name])[name])
        assert ess > 1000, f"{name} ESS {ess:.0f}: sampler is not mixing"
        # SE of a mean is sd/√ESS; allow 5 SE across both runs.
        np.testing.assert_allclose(got.mean(), ref.mean(), atol=5 * ref.std() / 30)
        np.testing.assert_allclose(got.std(), ref.std(), rtol=0.08)


@pytest.fixture(scope="module")
def reference(data):
    return _pg_reference(_model(data), N_REGIONS)


PANEL_N, PANEL_T = 20, 4


@pytest.fixture(scope="module")
def panel_model():
    d = simulate_sar_negbin_flow(
        n=PANEL_N, T=PANEL_T, rho_d=0.35, rho_o=0.25, seed=5, separable=True
    )
    return SARNegBinFlow(
        d["y"], d["X"], d["G"], separable=True, T=PANEL_T, col_names=d["col_names"]
    )


@pytest.fixture(scope="module")
def panel_reference(panel_model):
    return _pg_reference(panel_model, PANEL_N, PANEL_T)


BACKENDS = ["numpy", pytest.param("jax", marks=pytest.mark.requires_jax)]


@pytest.mark.slow
@pytest.mark.parametrize("backend", BACKENDS)
def test_separable_posterior_matches_pg_reference(data, reference, backend):
    idata = _model(data).fit(
        draws=DRAWS,
        tune=TUNE,
        chains=CHAINS,
        random_seed=11,
        progressbar=False,
        backend=backend,
    )
    _assert_matches(idata, reference)


@pytest.mark.slow
@pytest.mark.parametrize("backend", BACKENDS)
def test_separable_panel_posterior_matches_pg_reference(
    panel_model, panel_reference, backend
):
    idata = panel_model.fit(
        draws=DRAWS,
        tune=TUNE,
        chains=CHAINS,
        random_seed=11,
        progressbar=False,
        backend=backend,
    )
    _assert_matches(idata, panel_reference)
