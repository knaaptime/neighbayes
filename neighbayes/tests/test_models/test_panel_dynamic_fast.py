"""Fast tests for dynamic panel model methods without full MCMC."""

from __future__ import annotations

import arviz as az
import numpy as np
import pymc as pm
import pytest
import xarray as xr

from neighbayes.models import (
    OLSPanelDynamic,
    SARPanelDynamic,
    SDEMPanelDynamic,
    SDMRPanelDynamic,
    SDMUPanelDynamic,
    SEMPanelDynamic,
    SLXPanelDynamic,
)
from neighbayes.tests.helpers import W_to_graph, make_line_W


def _idata(vars_dict: dict[str, np.ndarray]) -> xr.DataTree:
    payload = {k: np.asarray(v)[None, ...] for k, v in vars_dict.items()}
    return az.from_dict({"posterior": payload})


def _panel_data(seed: int = 50):
    rng = np.random.default_rng(seed)
    N, T = 4, 3
    n = N * T
    x1 = rng.normal(size=n)
    X = np.column_stack([np.ones(n), x1])
    y = 0.2 + 0.9 * x1 + rng.normal(scale=0.25, size=n)
    W = W_to_graph(make_line_W(N))
    return y, X, W, N, T


def test_dynamic_panel_models_fitted_values_and_effects_with_mock_posteriors():
    y, X, W, N, T = _panel_data()

    # k=2, kw=1 => beta length is 3
    beta = np.array([0.25, 0.85, 0.10])

    dlm = OLSPanelDynamic(y=y, X=X, W=W, N=N, T=T, effects=0)
    dlm._idata = _idata(
        {
            "beta": np.stack([beta, beta + 1e-3]),
            "phi": np.array([0.4, 0.401]),
        }
    )

    sdmr = SDMRPanelDynamic(y=y, X=X, W=W, N=N, T=T, effects=0)
    sdmr._idata = _idata(
        {
            "beta": np.stack([beta, beta + 1e-3]),
            "phi": np.array([0.4, 0.401]),
            "rho": np.array([0.2, 0.201]),
        }
    )

    sdmu = SDMUPanelDynamic(y=y, X=X, W=W, N=N, T=T, effects=0)
    sdmu._idata = _idata(
        {
            "beta": np.stack([beta, beta + 1e-3]),
            "phi": np.array([0.4, 0.401]),
            "rho": np.array([0.2, 0.201]),
            "theta": np.array([-0.1, -0.099]),
        }
    )

    for model in [dlm, sdmr, sdmu]:
        fitted = model.fitted_values()
        effects = model.spatial_effects()

        assert fitted.shape[0] == N * (T - 1)
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


def test_dynamic_dlm_no_wx_branch_uses_feature_names():
    y, _, W, N, T = _panel_data(seed=51)
    X = np.ones((N * T, 1), dtype=float)  # only intercept => no WX columns

    model = OLSPanelDynamic(y=y, X=X, W=W, N=N, T=T, effects=0)
    model._idata = _idata(
        {
            "beta": np.array([[0.3], [0.301]]),
            "phi": np.array([0.5, 0.501]),
        }
    )

    effects = model.spatial_effects()
    assert list(effects.index) == ["x0"]
    assert np.allclose(effects["indirect"].values, 0.0)


def test_dynamic_models_require_at_least_two_periods():
    rng = np.random.default_rng(52)
    N, T = 4, 1
    y = rng.normal(size=N * T)
    X = np.column_stack([np.ones(N * T), rng.normal(size=N * T)])
    W = W_to_graph(make_line_W(N))

    model = OLSPanelDynamic(y=y, X=X, W=W, N=N, T=T, effects=0)
    model._idata = _idata(
        {
            "beta": np.array([[0.2, 0.8], [0.201, 0.801]]),
            "phi": np.array([0.3, 0.301]),
        }
    )

    with pytest.raises(ValueError, match="T >= 2"):
        model.fitted_values()


def test_dynamic_sar_panel_fitted_values_and_effects():
    y, X, W, N, T = _panel_data()
    beta = np.array([0.25, 0.85])

    model = SARPanelDynamic(y=y, X=X, W=W, N=N, T=T, effects=0)
    model._idata = _idata(
        {
            "beta": np.stack([beta, beta + 1e-3]),
            "phi": np.array([0.4, 0.401]),
            "rho": np.array([0.2, 0.201]),
        }
    )

    fitted = model.fitted_values()
    effects = model.spatial_effects()

    assert fitted.shape[0] == N * (T - 1)
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
    # SAR has indirect effects via rho
    assert not np.allclose(effects["indirect"].values, 0.0)


def test_dynamic_sem_panel_fitted_values_and_effects():
    y, X, W, N, T = _panel_data()
    beta = np.array([0.25, 0.85])

    model = SEMPanelDynamic(y=y, X=X, W=W, N=N, T=T, effects=0)
    model._idata = _idata(
        {
            "beta": np.stack([beta, beta + 1e-3]),
            "phi": np.array([0.4, 0.401]),
            "lam": np.array([0.2, 0.201]),
        }
    )

    fitted = model.fitted_values()
    effects = model.spatial_effects()

    assert fitted.shape[0] == N * (T - 1)
    assert np.all(np.isfinite(fitted))
    # SEM has no indirect effects
    assert np.allclose(effects["indirect"].values, 0.0)


def test_dynamic_sdem_panel_fitted_values_and_effects():
    y, X, W, N, T = _panel_data()
    beta = np.array([0.25, 0.85, 0.10])  # k=2, kw=1

    model = SDEMPanelDynamic(y=y, X=X, W=W, N=N, T=T, effects=0)
    model._idata = _idata(
        {
            "beta": np.stack([beta, beta + 1e-3]),
            "phi": np.array([0.4, 0.401]),
            "lam": np.array([0.2, 0.201]),
        }
    )

    fitted = model.fitted_values()
    effects = model.spatial_effects()

    assert fitted.shape[0] == N * (T - 1)
    assert np.all(np.isfinite(fitted))
    # SDEM panel: effects DataFrame should have valid columns
    assert "direct" in effects.columns
    assert "indirect" in effects.columns
    assert "total" in effects.columns
    assert np.all(np.isfinite(effects["direct"].values))


def test_dynamic_slx_panel_fitted_values_and_effects():
    y, X, W, N, T = _panel_data()
    beta = np.array([0.25, 0.85, 0.10])  # k=2, kw=1

    model = SLXPanelDynamic(y=y, X=X, W=W, N=N, T=T, effects=0)
    model._idata = _idata(
        {
            "beta": np.stack([beta, beta + 1e-3]),
            "phi": np.array([0.4, 0.401]),
        }
    )

    fitted = model.fitted_values()
    effects = model.spatial_effects()

    assert fitted.shape[0] == N * (T - 1)
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


@pytest.mark.parametrize(
    "cls,expected_present,expected_absent",
    [
        (SARPanelDynamic, ("rho", "phi"), ()),
        (SEMPanelDynamic, ("lam", "phi"), ()),
        (SDEMPanelDynamic, ("lam", "phi"), ()),
        (SLXPanelDynamic, ("phi",), ("rho", "lam")),
    ],
    ids=lambda v: getattr(v, "__name__", str(v)),
)
def test_dynamic_panel_builds_pymc_model(cls, expected_present, expected_absent):
    y, X, W, N, T = _panel_data()
    model = cls(y=y, X=X, W=W, N=N, T=T, effects=0)
    pymc_model = model._build_pymc_model()
    rv_names = [v.name for v in pymc_model.free_RVs]
    for name in expected_present:
        assert name in rv_names
    for name in expected_absent:
        assert name not in rv_names


# ---------------------------------------------------------------------------
# Nickell bias guard: effects=1 must raise ValueError for all dynamic models
# ---------------------------------------------------------------------------

_DYNAMIC_CLASSES = [
    OLSPanelDynamic,
    SDMRPanelDynamic,
    SDMUPanelDynamic,
    SARPanelDynamic,
    SEMPanelDynamic,
    SDEMPanelDynamic,
    SLXPanelDynamic,
]


@pytest.mark.parametrize("cls", _DYNAMIC_CLASSES, ids=lambda c: c.__name__)
def test_dynamic_panel_model1_raises_nickell_bias(cls):
    """effects=1 (unit FE) must raise ValueError for dynamic panel models.

    The within-transformation creates correlation between the demeaned
    lagged dependent variable and the demeaned error, biasing the
    autoregressive coefficient toward zero (Nickell, 1981).
    """
    y, X, W, N, T = _panel_data()
    with pytest.raises(ValueError, match="Nickell|effects=1|unit fixed effects"):
        cls(y=y, X=X, W=W, N=N, T=T, effects=1)


# ---------------------------------------------------------------------------
# Build / branch coverage (formerly test_panel_dynamic_builds_fast.py)
# ---------------------------------------------------------------------------


def _builds_data(seed: int = 100):
    rng = np.random.default_rng(seed)
    N, T = 4, 4
    n = N * T
    x1 = rng.normal(size=n)
    X = np.column_stack([np.ones(n), x1])
    y = 0.3 + 0.7 * x1 + rng.normal(scale=0.2, size=n)
    W = W_to_graph(make_line_W(N))
    return y, X, W, N, T


def test_dynamic_panel_build_pymc_models_and_prepare_dynamic_design_cache():
    y, X, W, N, T = _builds_data()

    models = [
        OLSPanelDynamic(y=y, X=X, W=W, N=N, T=T, effects=0),
        SDMRPanelDynamic(y=y, X=X, W=W, N=N, T=T, effects=0),
        SDMUPanelDynamic(y=y, X=X, W=W, N=N, T=T, effects=0),
    ]

    for model in models:
        pymc_model = model._build_pymc_model()
        assert isinstance(pymc_model, pm.Model)

        # Exercise the cached branch in _prepare_dynamic_design.
        z0 = model._Z_dyn
        model._prepare_dynamic_design()
        assert model._Z_dyn is z0


def test_dynamic_sdm_models_no_wx_branch_effects_and_names():
    y, _, W, N, T = _builds_data(seed=101)

    # Use X with 2 columns (intercept + x1) so WX has 1 column
    X = np.column_stack([np.ones(N * T), np.linspace(-1, 1, N * T)])

    sdmr = SDMRPanelDynamic(y=y, X=X, W=W, N=N, T=T, effects=0)
    sdmr._idata = _idata(
        {
            "beta": np.array([[0.2, 0.5, 0.1], [0.201, 0.501, 0.101]]),
            "phi": np.array([0.4, 0.401]),
            "rho": np.array([0.1, 0.101]),
        }
    )

    sdmu = SDMUPanelDynamic(y=y, X=X, W=W, N=N, T=T, effects=0)
    sdmu._idata = _idata(
        {
            "beta": np.array([[0.2, 0.5, 0.1], [0.201, 0.501, 0.101]]),
            "phi": np.array([0.4, 0.401]),
            "rho": np.array([0.1, 0.101]),
            "theta": np.array([0.0, 0.001]),
        }
    )

    for model in [sdmr, sdmu]:
        eff = model.spatial_effects()
        # SDM models with WX terms report effects for lagged covariates only
        assert len(eff.index) >= 1
        assert np.all(np.isfinite(eff["direct"].values))
        assert np.all(np.isfinite(eff["indirect"].values))
        assert np.all(np.isfinite(eff["total"].values))


def test_dynamic_beta_names_include_wx_labels_when_present():
    y, X, W, N, T = _builds_data(seed=102)

    model = OLSPanelDynamic(y=y, X=X, W=W, N=N, T=T, effects=0)
    names = model._beta_names()

    assert any(name.startswith("W*") for name in names)
    assert len(names) > X.shape[1]
