"""Fast method/build tests for cross-sectional spatial models (SLX/SEM/SDEM/SDM).

These tests inject a mock posterior into ``model._idata`` to exercise
``fitted_values``, ``spatial_effects``, ``_beta_names``, and ``_build_pymc_model``
without running MCMC.
"""

from __future__ import annotations

import arviz as az
import numpy as np
import pymc as pm
import pytest
import xarray as xr

from neighbayes.models import SDEM, SDM, SEM, SLX
from neighbayes.tests.helpers import W_to_graph, make_line_W

_EXPECTED_EFFECT_COLUMNS = {
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


def _idata(vars_dict: dict[str, np.ndarray]) -> xr.DataTree:
    payload = {k: np.asarray(v)[None, ...] for k, v in vars_dict.items()}
    return az.from_dict({"posterior": payload})


def _cs_data(seed: int = 90, *, intercept: bool = True):
    rng = np.random.default_rng(seed)
    n = 8
    x1 = rng.normal(size=n)
    if intercept:
        X = np.column_stack([np.ones(n), x1])
        y = 0.5 + 0.9 * x1 + rng.normal(scale=0.3, size=n)
    else:
        x2 = rng.normal(size=n)
        X = np.column_stack([x1, x2])
        y = 0.8 * x1 - 0.3 * x2 + rng.normal(scale=0.3, size=n)
    W = W_to_graph(make_line_W(n))
    return y, X, W


@pytest.mark.parametrize("model_cls", [SLX, SEM, SDEM, SDM])
def test_build_pymc_model(model_cls):
    y, X, W = _cs_data()
    model = model_cls(y=y, X=X, W=W)
    assert isinstance(model._build_pymc_model(), pm.Model)


@pytest.mark.parametrize("model_cls", [SLX, SDEM, SDM])
def test_beta_names_include_spatially_lagged_labels(model_cls):
    y, X, W = _cs_data(seed=92)
    model = model_cls(y=y, X=X, W=W)
    names = model._beta_names()
    assert len(names) > X.shape[1]
    assert any(name.startswith("W*") for name in names)


def test_slx_fitted_values_and_effects_with_mock_posterior():
    y, X, W = _cs_data(seed=111, intercept=False)
    model = SLX(y=y, X=X, W=W)

    k = model._X.shape[1]
    beta = np.linspace(0.3, 0.3 + 0.1 * (2 * k - 1), 2 * k)
    model._idata = _idata({"beta": np.stack([beta, beta + 1e-3])})

    fitted = model.fitted_values()
    effects = model.spatial_effects()

    assert fitted.shape == y.shape
    assert np.all(np.isfinite(fitted))
    assert set(effects.columns) == _EXPECTED_EFFECT_COLUMNS
    assert np.all(np.isfinite(effects["direct"].values))


def test_slx_fitted_values_raises_on_unexpected_beta_dimension():
    y, X, W = _cs_data(seed=112)
    model = SLX(y=y, X=X, W=W)

    bad_beta = np.array([0.3, 0.9])  # expected length is 3
    model._idata = _idata({"beta": np.stack([bad_beta, bad_beta + 1e-3])})

    with pytest.raises(ValueError, match="Unexpected beta dimension"):
        model.fitted_values()


def test_sem_sdem_sdm_fitted_values_and_effects_with_mock_posteriors():
    y, X, W = _cs_data(seed=91)

    sem = SEM(y=y, X=X, W=W)
    sem._idata = _idata(
        {
            "beta": np.stack([np.array([0.3, 0.8]), np.array([0.301, 0.801])]),
            "lam": np.array([0.1, 0.101]),
        }
    )

    # k=2, kw=1 -> beta length 3 for SDEM/SDM
    beta3 = np.array([0.3, 0.8, 0.15])

    sdem = SDEM(y=y, X=X, W=W)
    sdem._idata = _idata(
        {
            "beta": np.stack([beta3, beta3 + 1e-3]),
            "lam": np.array([0.1, 0.101]),
        }
    )

    sdm = SDM(y=y, X=X, W=W)
    sdm._idata = _idata(
        {
            "beta": np.stack([beta3, beta3 + 1e-3]),
            "rho": np.array([0.2, 0.201]),
        }
    )

    for model in [sem, sdem, sdm]:
        fitted = model.fitted_values()
        effects = model.spatial_effects()
        assert fitted.shape == y.shape
        assert np.all(np.isfinite(fitted))
        assert set(effects.columns) == _EXPECTED_EFFECT_COLUMNS
        assert np.all(np.isfinite(effects["direct"].values))

    sem_eff = sem.spatial_effects()
    assert np.allclose(sem_eff["indirect"].values, 0.0)
