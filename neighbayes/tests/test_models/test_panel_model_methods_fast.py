"""Fast method-level tests for static panel FE model classes."""

from __future__ import annotations

import warnings

import arviz as az
import numpy as np
import pymc as pm
import pytest
import xarray as xr
from libpysal.graph import Graph

from neighbayes.models import OLS, SAR, SDEM, SDM, SEM, SLX
from neighbayes.models.panel_base import SpatialPanelModel
from neighbayes.tests.helpers import W_to_graph, make_line_W


def _idata(vars_dict: dict[str, np.ndarray]) -> xr.DataTree:
    payload = {k: np.asarray(v)[None, ...] for k, v in vars_dict.items()}
    return az.from_dict({"posterior": payload})


def _panel_data(seed: int = 60):
    rng = np.random.default_rng(seed)
    N, T = 4, 3
    n = N * T
    x1 = rng.normal(size=n)
    X = np.column_stack([np.ones(n), x1])
    y = 0.2 + 0.8 * x1 + rng.normal(scale=0.25, size=n)
    W = W_to_graph(make_line_W(N))
    return y, X, W, N, T


def _graph_from_dense_without_transform(W_dense: np.ndarray) -> Graph:
    rows, cols = np.nonzero(W_dense)
    weights = W_dense[rows, cols]
    return Graph.from_arrays(rows, cols, weights.astype(float))


def test_panel_fe_build_pymc_models():
    y, X, W, N, T = _panel_data()

    models = [
        OLS(unit_effects="fixed", y=y, X=X, W=W, N=N, T=T),
        SAR(unit_effects="fixed", y=y, X=X, W=W, N=N, T=T),
        SEM(unit_effects="fixed", y=y, X=X, W=W, N=N, T=T),
        SDM(unit_effects="fixed", y=y, X=X, W=W, N=N, T=T),
        SDEM(unit_effects="fixed", y=y, X=X, W=W, N=N, T=T),
        SLX(unit_effects="fixed", y=y, X=X, W=W, N=N, T=T),
    ]

    for model in models:
        pymc_model = model._build_pymc_model()
        assert isinstance(pymc_model, pm.Model)


def test_panel_fe_fitted_values_and_effects_with_mock_posteriors():
    y, X, W, N, T = _panel_data(seed=61)

    # FE models drop the intercept column, so beta has k-1 elements
    # (slope only) for OLS/SAR/SEM, and (k-1 + kw) for SDM/SDEM/SLX.
    beta_1 = np.array([0.9])  # slope only (intercept dropped)
    beta_2_fe = np.array([0.9, 0.15])  # slope + WX (intercept dropped)

    ols = OLS(unit_effects="fixed", y=y, X=X, W=W, N=N, T=T)
    ols._idata = _idata({"beta": np.stack([beta_1, beta_1 + 1e-3])})

    sar = SAR(unit_effects="fixed", y=y, X=X, W=W, N=N, T=T)
    sar._idata = _idata(
        {
            "beta": np.stack([beta_1, beta_1 + 1e-3]),
            "rho": np.array([0.2, 0.201]),
        }
    )

    sem = SEM(unit_effects="fixed", y=y, X=X, W=W, N=N, T=T)
    sem._idata = _idata(
        {
            "beta": np.stack([beta_1, beta_1 + 1e-3]),
            "lam": np.array([0.1, 0.101]),
        }
    )

    sdm = SDM(unit_effects="fixed", y=y, X=X, W=W, N=N, T=T)
    sdm._idata = _idata(
        {
            "beta": np.stack([beta_2_fe, beta_2_fe + 1e-3]),
            "rho": np.array([0.2, 0.201]),
        }
    )

    sdem = SDEM(unit_effects="fixed", y=y, X=X, W=W, N=N, T=T)
    sdem._idata = _idata(
        {
            "beta": np.stack([beta_2_fe, beta_2_fe + 1e-3]),
            "lam": np.array([0.1, 0.101]),
        }
    )

    slx = SLX(unit_effects="fixed", y=y, X=X, W=W, N=N, T=T)
    slx._idata = _idata(
        {
            "beta": np.stack([beta_2_fe, beta_2_fe + 1e-3]),
        }
    )

    for model in [ols, sar, sem, sdm, sdem, slx]:
        fitted = model.fitted_values()
        effects = model.impacts()
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


def test_panel_fe_impacts_accept_numeric_row_standardized_graph_without_transform():
    y, X, _, N, T = _panel_data(seed=65)
    # FE models drop the intercept, so beta has (k-1 + kw) = 2 elements
    beta_2_fe = np.array([0.9, 0.15])
    W = _graph_from_dense_without_transform(make_line_W(N))

    assert getattr(W, "transformation", None) == "O"

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model = SDM(unit_effects="fixed", y=y, X=X, W=W, N=N, T=T)

    model._idata = _idata(
        {
            "beta": np.stack([beta_2_fe, beta_2_fe + 1e-3]),
            "rho": np.array([0.2, 0.201]),
        }
    )

    assert model._is_row_std is True
    assert not any("row-standardized" in str(w.message) for w in caught)

    effects = model.impacts()
    assert np.all(np.isfinite(effects[["direct", "indirect", "total"]].to_numpy()))


def test_panel_fe_warns_on_non_row_standardized_graph_without_transform():
    y, X, _, N, T = _panel_data(seed=66)
    W = _graph_from_dense_without_transform(
        np.array(
            [
                [0.0, 1.0, 1.0, 0.0],
                [1.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 1.0, 0.0],
            ]
        )
    )

    assert getattr(W, "transformation", None) == "O"

    with pytest.warns(UserWarning, match="row-standardized"):
        model = OLS(unit_effects="fixed", y=y, X=X, W=W, N=N, T=T)

    assert model._is_row_std is False


def test_sem_panel_fe_fit_adds_log_likelihood_when_missing(monkeypatch):
    y, X, W, N, T = _panel_data(seed=62)
    model = SEM(unit_effects="fixed", y=y, X=X, W=W, N=N, T=T)

    posterior = {
        "lam": np.array([[0.1, 0.11]]),
        "beta": np.array([[[0.9], [0.91]]]),
        "sigma": np.array([[1.0, 1.1]]),
    }
    fake_idata = az.from_dict({"posterior": posterior})

    import pymc as pm

    monkeypatch.setattr(pm, "sample", lambda **kw: fake_idata)
    out = model.fit(
        draws=2,
        tune=1,
        chains=1,
        progressbar=False,
        idata_kwargs={"log_likelihood": True},
    )

    assert "log_likelihood" in out.children
    assert "obs" in out.log_likelihood


def test_sem_panel_fe_fit_returns_early_when_log_likelihood_exists(monkeypatch):
    y, X, W, N, T = _panel_data(seed=63)
    model = SEM(unit_effects="fixed", y=y, X=X, W=W, N=N, T=T)

    n = y.shape[0]
    fake_idata = az.from_dict(
        {
            "posterior": {
                "lam": np.array([[0.1, 0.11]]),
                "beta": np.array([[[0.9], [0.91]]]),
                "sigma": np.array([[1.0, 1.1]]),
            },
            "log_likelihood": {"obs": np.zeros((1, 2, n), dtype=float)},
        }
    )

    import pymc as pm

    monkeypatch.setattr(pm, "sample", lambda **kw: fake_idata)
    out = model.fit(
        draws=2,
        tune=1,
        chains=1,
        progressbar=False,
        sampler="nuts",
        idata_kwargs={"log_likelihood": True},
    )

    assert out is fake_idata


def test_sdem_panel_fe_fit_adds_log_likelihood_when_missing(monkeypatch):
    y, X, W, N, T = _panel_data(seed=64)
    model = SDEM(unit_effects="fixed", y=y, X=X, W=W, N=N, T=T)

    posterior = {
        "lam": np.array([[0.1, 0.11]]),
        "beta": np.array([[[0.9, 0.15], [0.91, 0.16]]]),
        "sigma": np.array([[1.0, 1.1]]),
    }
    fake_idata = az.from_dict({"posterior": posterior})

    import pymc as pm

    monkeypatch.setattr(pm, "sample", lambda **kw: fake_idata)
    out = model.fit(
        draws=2,
        tune=1,
        chains=1,
        progressbar=False,
        idata_kwargs={"log_likelihood": True},
    )

    assert "log_likelihood" in out.children
    assert "obs" in out.log_likelihood


def test_sar_panel_fe_fit_applies_jacobian_when_loglik_requested(monkeypatch):
    """Jacobian correction should run when log_likelihood is requested."""
    y, X, W, N, T = _panel_data(seed=67)
    model = SAR(unit_effects="fixed", y=y, X=X, W=W, N=N, T=T)

    n = y.shape[0]
    fake_idata = az.from_dict(
        {
            "posterior": {
                "rho": np.array([[0.2, 0.21]]),
                "beta": np.array([[[0.9], [0.91]]]),
                "sigma": np.array([[1.0, 1.1]]),
            },
            "log_likelihood": {"obs": np.zeros((1, 2, n), dtype=float)},
        }
    )

    called = {"ok": False}

    import pymc as pm

    def _fake_sample(**kw):
        return fake_idata

    def _fake_attach(spatial_param, nuts_sampler, T_eff):
        called["ok"] = True
        assert spatial_param == "rho"
        assert T_eff == model._T

    monkeypatch.setattr(pm, "sample", _fake_sample)
    monkeypatch.setattr(model, "_reconstruct_panel_log_likelihood", _fake_attach)

    out = model.fit(
        draws=2,
        tune=1,
        chains=1,
        progressbar=False,
        sampler="nuts",
        idata_kwargs={"log_likelihood": True},
    )

    assert out is fake_idata
    assert called["ok"]
