"""Fast construction tests for Tobit model classes."""

from __future__ import annotations

import arviz as az
import numpy as np
import pymc as pm
import xarray as xr

from neighbayes.models import SARPanelTobit, SARTobit, SDMTobit, SEMPanelTobit, SEMTobit
from neighbayes.tests.helpers import W_to_graph, make_line_W


def _idata(vars_dict: dict[str, np.ndarray]) -> xr.DataTree:
    payload = {k: np.asarray(v)[None, ...] for k, v in vars_dict.items()}
    return az.from_dict({"posterior": payload})


def _cs_data(seed: int = 80):
    rng = np.random.default_rng(seed)
    n = 8
    x1 = rng.normal(size=n)
    X = np.column_stack([np.ones(n), x1])
    y = 1.0 + 0.5 * x1 + rng.normal(scale=0.2, size=n)  # strictly above 0
    W = W_to_graph(make_line_W(n))
    return y, X, W


def _panel_data(seed: int = 81):
    rng = np.random.default_rng(seed)
    N, T = 4, 3
    n = N * T
    x1 = rng.normal(size=n)
    X = np.column_stack([np.ones(n), x1])
    y = 1.2 + 0.6 * x1 + rng.normal(scale=0.2, size=n)  # strictly above 0
    W = W_to_graph(make_line_W(N))
    return y, X, W, N, T


def test_cross_section_tobit_models_build_and_handle_no_censoring():
    y, X, W = _cs_data()

    sar = SARTobit(y=y, X=X, W=W)
    sem = SEMTobit(y=y, X=X, W=W)
    sdm = SDMTobit(y=y, X=X, W=W)

    assert sar._censored_idx.size == 0
    assert sem._censored_idx.size == 0
    assert sdm._censored_idx.size == 0

    for model in [sar, sem, sdm]:
        pymc_model = model._build_pymc_model()
        assert isinstance(pymc_model, pm.Model)

    # No censored draws provided should return observed y unchanged.
    sar._idata = _idata({"beta": np.array([[0.2, 0.8]]), "rho": np.array([0.1])})
    assert np.allclose(sar._posterior_latent_y_mean(), sar._y)

    names = sdm._beta_names()
    assert any(name.startswith("W*") for name in names)


def test_panel_tobit_models_build_force_model_zero_and_handle_no_censoring():
    y, X, W, N, T = _panel_data()

    sar = SARPanelTobit(y=y, X=X, W=W, N=N, T=T, effects=1)
    sem = SEMPanelTobit(y=y, X=X, W=W, N=N, T=T, effects=1)

    assert sar.model == 0
    assert sem.model == 0
    assert sar._censored_idx.size == 0
    assert sem._censored_idx.size == 0

    for model in [sar, sem]:
        pymc_model = model._build_pymc_model()
        assert isinstance(pymc_model, pm.Model)

    sem._idata = _idata({"beta": np.array([[0.2, 0.8]]), "lam": np.array([0.1])})
    assert np.allclose(sem._posterior_latent_y_mean(), sem._y)
