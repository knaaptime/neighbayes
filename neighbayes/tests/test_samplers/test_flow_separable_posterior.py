"""Separable NB flow samplers must match an exact-solve reference posterior.

Both separable backends once reused a ρ Krylov basis across sweeps.  A basis for
ρ_d is built at (ρ_d, ρ_o) and ρ_o is re-sliced every sweep, so the reused basis
evaluated the density at a stale ρ_o: ρ posterior SDs came out 10–20% too wide
with ESS near 50 from 40,000 draws, while every marginal mean still looked
right.  This pins the posterior SDs of the NumPy and JAX (structured) backends, for the
cross-section and a panel, to the reference sampler run with exact solves at
every candidate.
"""

from __future__ import annotations

import arviz as az
import numpy as np
import pytest

from neighbayes.dgp.flows import (
    generate_negbin_flow_data_separable,
    generate_panel_negbin_flow_data_separable,
)
from neighbayes.models import SARNegBinFlowSeparable, SARNegBinFlowSeparablePanel
from neighbayes.models._base._shared import gelman_default_beta_prior
from neighbayes.models.priors import FlowReducedGibbsPriors
from neighbayes.samplers.negbin_reduced._flow import (
    FlowReducedGibbsCache,
    FlowReducedGibbsState,
    run_chain_separable,
)

N_REGIONS = 25
DRAWS, TUNE, CHAINS = 4000, 800, 4


@pytest.fixture(scope="module")
def data():
    return generate_negbin_flow_data_separable(
        n=N_REGIONS, rho_d=0.35, rho_o=0.25, seed=3
    )


def _model(data):
    y = data["y_vec"] if "y_vec" in data else data["y"]
    return SARNegBinFlowSeparable(y, data["X"], data["G"], col_names=data["col_names"])


def _exact_reference(m, n, T=1):
    """Exact Kronecker solves at every slice candidate (no Krylov basis)."""
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
        cache = FlowReducedGibbsCache(
            None, None, None, W, n, separable=True, T=T, krylov_degree=0
        )
        out = run_chain_separable(
            m._y_int_vec.astype(float),
            m._X,
            W,
            n,
            priors,
            cache,
            init,
            DRAWS,
            TUNE,
            rng=rng,
            store_log_lik=False,
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
    return _exact_reference(_model(data), N_REGIONS)


PANEL_N, PANEL_T = 20, 4


@pytest.fixture(scope="module")
def panel_model():
    d = generate_panel_negbin_flow_data_separable(
        n=PANEL_N, T=PANEL_T, rho_d=0.35, rho_o=0.25, seed=5
    )
    return SARNegBinFlowSeparablePanel(
        d["y"], d["X"], d["G"], T=PANEL_T, col_names=d["col_names"]
    )


@pytest.fixture(scope="module")
def panel_reference(panel_model):
    return _exact_reference(panel_model, PANEL_N, PANEL_T)


BACKENDS = ["numpy", pytest.param("jax", marks=pytest.mark.requires_jax)]


@pytest.mark.slow
@pytest.mark.parametrize("backend", BACKENDS)
def test_separable_posterior_matches_exact_reference(data, reference, backend):
    idata = _model(data).fit(
        draws=DRAWS,
        tune=TUNE,
        chains=CHAINS,
        random_seed=11,
        progressbar=False,
        gibbs_backend=backend,
    )
    _assert_matches(idata, reference)


@pytest.mark.slow
@pytest.mark.parametrize("backend", BACKENDS)
def test_separable_panel_posterior_matches_exact_reference(
    panel_model, panel_reference, backend
):
    idata = panel_model.fit(
        draws=DRAWS,
        tune=TUNE,
        chains=CHAINS,
        random_seed=11,
        progressbar=False,
        gibbs_backend=backend,
    )
    _assert_matches(idata, panel_reference)
