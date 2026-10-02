"""Fast branch and utility tests for SARProbit."""

from __future__ import annotations

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm
import pytest
import scipy.sparse as sp
import xarray as xr

from neighbayes.models import SARProbit
from neighbayes.tests.helpers import W_to_graph, make_line_W


class _LegacyW:
    def __init__(self):
        self.sparse = sp.eye(2, format="csr")

    def transform(self, *_args, **_kwargs):
        return self


def _idata_for_model(m: SARProbit) -> xr.DataTree:
    n = m._X.shape[0]
    r = m._m
    posterior = {
        "beta": np.array(
            [[[0.2] * m._X.shape[1], [0.21] * m._X.shape[1]]], dtype=float
        ),
        "a": np.array([[[0.1] * r, [0.11] * r]], dtype=float),
        "p": np.array([[[0.6] * n, [0.61] * n]], dtype=float),
        "rho": np.array([[0.1, 0.11]], dtype=float),
        "sigma_a": np.array([[0.9, 0.91]], dtype=float),
    }
    return az.from_dict({"posterior": posterior})


def test_as_dense_region_W_warns_for_non_row_standardized_sparse():
    W = sp.csr_matrix(np.array([[0.0, 2.0], [1.0, 0.0]], dtype=float))
    with pytest.warns(UserWarning, match="row-standardized"):
        dense = SARProbit._as_dense_region_W(W)
    assert dense.shape == (2, 2)


def test_as_dense_region_W_rejects_legacy_and_dense_types():
    with pytest.raises(TypeError, match="legacy libpysal.weights.W"):
        SARProbit._as_dense_region_W(_LegacyW())

    with pytest.raises(TypeError, match="Graph or a scipy sparse"):
        SARProbit._as_dense_region_W(np.eye(2))


def test_matrix_mode_region_parsing_with_ids_and_mobs():
    y = np.array([0, 1, 0, 1], dtype=float)
    X = np.column_stack([np.ones(4), np.arange(4, dtype=float)])
    W = W_to_graph(make_line_W(2))

    m_ids = SARProbit(y=y, X=X, W=W, region_ids=np.array([10, 10, 20, 20]))
    assert m_ids._region_codes.tolist() == [0, 0, 1, 1]
    assert m_ids._region_names == ["10", "20"]

    m_mobs = SARProbit(y=y, X=X, W=W, mobs=[2, 2])
    assert m_mobs._region_codes.tolist() == [0, 0, 1, 1]
    assert m_mobs._region_names == ["region_0", "region_1"]


def test_spatial_probit_constructor_validation_errors():
    y = np.array([0, 1, 0, 1], dtype=float)
    X = np.column_stack([np.ones(4), np.arange(4, dtype=float)])
    W = W_to_graph(make_line_W(2))

    with pytest.raises(ValueError, match="Provide either region_ids or mobs"):
        SARProbit(y=y, X=X, W=W)

    with pytest.raises(ValueError, match="region_ids must have one entry"):
        SARProbit(y=y, X=X, W=W, region_ids=np.array([0, 1, 0]))

    with pytest.raises(ValueError, match=r"sum\(mobs\) must equal"):
        SARProbit(y=y, X=X, W=W, mobs=[1, 1])

    with pytest.raises(ValueError, match="y must be binary"):
        SARProbit(y=np.array([0, 2, 0, 1], dtype=float), X=X, W=W, mobs=[2, 2])

    with pytest.raises(ValueError, match="must match W dimension"):
        SARProbit(y=y, X=X, W=W, mobs=[1, 1, 2])


def test_formula_mode_validation_and_parsing():
    df = pd.DataFrame(
        {
            "y": [0, 1, 0, 1],
            "x": [1.0, 2.0, 3.0, 4.0],
            "region": ["a", "a", "b", "b"],
        }
    )
    W = W_to_graph(make_line_W(2))

    with pytest.raises(ValueError, match="data is required"):
        SARProbit(formula="y ~ x", W=W, region_col="region")

    with pytest.raises(ValueError, match="region_col is required"):
        SARProbit(formula="y ~ x", data=df, W=W)

    model = SARProbit(formula="y ~ x", data=df, W=W, region_col="region")
    assert model._X.shape[0] == 4
    assert model._y.shape[0] == 4
    assert model._region_names == ["a", "b"]


def test_build_model_requires_predictor_columns():
    y = np.array([0, 1, 0, 1], dtype=float)
    X = np.empty((4, 0), dtype=float)
    W = W_to_graph(make_line_W(2))
    model = SARProbit(y=y, X=X, W=W, mobs=[2, 2])

    with pytest.raises(ValueError, match="at least one predictor"):
        model._build_pymc_model()


def test_post_fit_accessors_and_rename_helper_and_pymc_property(monkeypatch):
    y = np.array([0, 1, 0, 1], dtype=float)
    X = np.column_stack([np.ones(4), np.arange(4, dtype=float)])
    W = W_to_graph(make_line_W(2))
    model = SARProbit(y=y, X=X, W=W, mobs=[2, 2])

    # pre-fit guard coverage
    with pytest.raises(RuntimeError, match="Model has not been fit yet"):
        model.summary()
    with pytest.raises(RuntimeError, match="Model has not been fit yet"):
        model.random_effects_mean()
    with pytest.raises(RuntimeError, match="Model has not been fit yet"):
        model.fitted_probabilities()

    fake_idata = _idata_for_model(model)

    def _fake_sample(**_kwargs):
        return fake_idata

    monkeypatch.setattr(pm, "sample", _fake_sample)
    out = model.fit(draws=2, tune=1, chains=1, progressbar=False)

    assert out is fake_idata
    assert model.pymc_model is not None

    # monkeypatch summary to avoid depending on ArviZ summary internals
    def _fake_summary(_idata, var_names=None, **_kwargs):
        idx = ["beta[Intercept]", "a[a]", "rho"]
        return pd.DataFrame({"mean": [0.2, 0.1, 0.1]}, index=idx)

    monkeypatch.setattr(az, "summary", _fake_summary)
    smry = model.summary(var_names=["beta", "a", "rho"])
    assert list(smry.index) == ["Intercept", "a:a", "rho"]

    a_mean = model.random_effects_mean()
    p_hat = model.fitted_probabilities()
    assert isinstance(a_mean, pd.Series)
    assert a_mean.shape[0] == 2
    assert p_hat.shape[0] == 4
    assert np.all((p_hat >= 0.0) & (p_hat <= 1.0))
