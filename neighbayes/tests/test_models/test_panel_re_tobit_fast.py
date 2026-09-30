"""Fast tests for panel RE and panel Tobit model methods."""

from __future__ import annotations

import arviz as az
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from neighbayes.models import (
    OLSPanelRE,
    SARPanelRE,
    SARPanelTobit,
    SEMPanelRE,
    SEMPanelTobit,
)
from neighbayes.tests.helpers import W_to_graph, make_line_W


def _idata(vars_dict: dict[str, np.ndarray]) -> xr.DataTree:
    payload = {k: np.asarray(v)[None, ...] for k, v in vars_dict.items()}
    return az.from_dict({"posterior": payload})


def _panel_data(seed: int = 30):
    rng = np.random.default_rng(seed)
    N, T = 4, 3
    n = N * T
    x1 = rng.normal(size=n)
    X = np.column_stack([np.ones(n), x1])
    y = 0.1 + 0.8 * x1 + rng.normal(scale=0.25, size=n)
    return y, X, W_to_graph(make_line_W(N)), N, T


def test_panel_re_fitted_values_and_effects_run_with_mock_posteriors():
    y, X, W, N, T = _panel_data()

    alpha = np.linspace(-0.2, 0.2, N)
    beta = np.array([0.3, 1.0])

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

    for m in [ols, sar, sem]:
        fitted = m.fitted_values()
        effects = m.spatial_effects()
        assert fitted.shape == y.shape
        assert np.all(np.isfinite(fitted))
        assert isinstance(effects, pd.DataFrame)
        assert "direct" in effects.columns
        assert "indirect" in effects.columns
        assert "total" in effects.columns

    sem_eff = sem.spatial_effects()
    assert np.allclose(sem_eff["indirect"].values, 0.0)


@pytest.mark.skip(reason="Spatial effects not yet implemented for panel Tobit models")
def test_panel_tobit_fitted_values_and_effects_run_with_latent_gap_draws():
    y, X, W, N, T = _panel_data(seed=31)
    y = y.copy()
    y[[0, 3]] = 0.0  # force censoring at default threshold

    sar_tobit = SARPanelTobit(y=y, X=X, W=W, N=N, T=T)
    sem_tobit = SEMPanelTobit(y=y, X=X, W=W, N=N, T=T)

    beta = np.array([0.2, 0.9])
    yc_sar = np.vstack(
        [
            np.linspace(0.05, 0.15, sar_tobit._censored_idx.size),
            np.linspace(0.06, 0.16, sar_tobit._censored_idx.size),
        ]
    )
    yc_sem = np.vstack(
        [
            np.linspace(0.05, 0.15, sem_tobit._censored_idx.size),
            np.linspace(0.06, 0.16, sem_tobit._censored_idx.size),
        ]
    )

    sar_tobit._idata = _idata(
        {
            "beta": np.stack([beta, beta + 1e-3]),
            "rho": np.array([0.15, 0.151]),
            "y_cens_gap": yc_sar,
        }
    )
    sem_tobit._idata = _idata(
        {
            "beta": np.stack([beta, beta + 1e-3]),
            "lam": np.array([0.05, 0.051]),
            "y_cens_gap": yc_sem,
        }
    )

    for m in [sar_tobit, sem_tobit]:
        fitted = m.fitted_values()
        effects = m.spatial_effects()
        assert fitted.shape == y.shape
        assert np.all(np.isfinite(fitted))
        assert np.all(np.isfinite(effects["direct"]))

    sem_eff = sem_tobit.spatial_effects()
    assert np.allclose(sem_eff["indirect"], 0.0)
