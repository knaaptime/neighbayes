"""Fast method-level tests for panel random-effects model classes."""

from __future__ import annotations

import arviz as az
import numpy as np
import pymc as pm
import xarray as xr

from neighbayes.models import OLSPanelRE, SARPanelRE, SDEMPanelRE, SEMPanelRE
from neighbayes.tests.helpers import W_to_graph, make_line_W


def _idata(vars_dict: dict[str, np.ndarray]) -> xr.DataTree:
    payload = {k: np.asarray(v)[None, ...] for k, v in vars_dict.items()}
    return az.from_dict({"posterior": payload})


def _panel_data(seed: int = 70):
    rng = np.random.default_rng(seed)
    N, T = 4, 3
    n = N * T
    x1 = rng.normal(size=n)
    X = np.column_stack([np.ones(n), x1])
    y = 0.3 + 0.7 * x1 + rng.normal(scale=0.25, size=n)
    W = W_to_graph(make_line_W(N))
    return y, X, W, N, T


def test_panel_re_build_pymc_models_and_force_model_zero():
    y, X, W, N, T = _panel_data()

    for cls in [OLSPanelRE, SARPanelRE, SEMPanelRE]:
        model = cls(y=y, X=X, W=W, N=N, T=T, effects=1)
        assert model.model == 0
        assert model._unit_idx.shape[0] == N * T

        pymc_model = model._build_pymc_model()
        assert isinstance(pymc_model, pm.Model)


def test_panel_re_fitted_values_and_effects_with_mock_posteriors():
    y, X, W, N, T = _panel_data(seed=71)
    alpha = np.array([0.1, -0.05, 0.02, -0.01])
    beta = np.array([0.2, 0.8])

    ols = OLSPanelRE(y=y, X=X, W=W, N=N, T=T)
    ols._idata = _idata(
        {
            "beta": np.stack([beta, beta + 1e-3]),
            "alpha": np.stack([alpha, alpha + 1e-3]),
        }
    )

    sar = SARPanelRE(y=y, X=X, W=W, N=N, T=T)
    sar._idata = _idata(
        {
            "beta": np.stack([beta, beta + 1e-3]),
            "alpha": np.stack([alpha, alpha + 1e-3]),
            "rho": np.array([0.2, 0.201]),
        }
    )

    sem = SEMPanelRE(y=y, X=X, W=W, N=N, T=T)
    sem._idata = _idata(
        {
            "beta": np.stack([beta, beta + 1e-3]),
            "alpha": np.stack([alpha, alpha + 1e-3]),
            "lam": np.array([0.1, 0.101]),
        }
    )

    for model in [ols, sar, sem]:
        fitted = model.fitted_values()
        effects = model.spatial_effects()

        assert fitted.shape == y.shape
        assert np.all(np.isfinite(fitted))
        assert set(effects.columns) == {
            "direct",
            "direct_ci_lower",
            "direct_ci_upper",
            "direct_pvalue",
            "indirect",
            "indirect_ci_lower",
            "indirect_ci_upper",
            "indirect_pvalue",
            "total",
            "total_ci_lower",
            "total_ci_upper",
            "total_pvalue",
        }
        assert np.all(np.isfinite(effects["direct"].values))


def test_sdem_panel_re_builds_and_computes_effects():
    y, X, W, N, T = _panel_data(seed=72)

    sdem = SDEMPanelRE(y=y, X=X, W=W, N=N, T=T, effects=2)
    assert sdem.model == 0
    pymc_model = sdem._build_pymc_model()
    assert isinstance(pymc_model, pm.Model)

    k = X.shape[1]
    k_wx = len(sdem._wx_column_indices)
    beta_full = np.concatenate([np.array([0.2, 0.8]), np.full(k_wx, 0.1)])
    alpha = np.array([0.1, -0.05, 0.02, -0.01])
    sdem._idata = _idata(
        {
            "beta": np.stack([beta_full, beta_full + 1e-3]),
            "alpha": np.stack([alpha, alpha + 1e-3]),
            "lam": np.array([0.1, 0.101]),
        }
    )

    fitted = sdem.fitted_values()
    effects = sdem.spatial_effects()
    assert fitted.shape == y.shape
    assert np.all(np.isfinite(fitted))
    assert np.all(np.isfinite(effects["direct"].values))
