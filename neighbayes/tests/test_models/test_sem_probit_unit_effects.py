"""The SEM probit with random unit effects (``SEMProbit(unit_effects="random")``).

Each unit, a row of ``W``, is observed several times, and the error process
carries one normal effect per unit; the model is fitted by NUTS.
"""

from __future__ import annotations

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm
import pytest
import scipy.sparse as sp
import xarray as xr

from neighbayes import dgp
from neighbayes.models import SEMProbit
from neighbayes.tests.helpers import SAMPLE_KWARGS, W_to_graph, make_line_W

Y4 = np.array([0, 1, 0, 1], dtype=float)
X4 = np.column_stack([np.ones(4), np.arange(4, dtype=float)])


class _LegacyW:
    def __init__(self):
        self.sparse = sp.eye(2, format="csr")

    def transform(self, *_args, **_kwargs):
        return self


def _W2():
    return W_to_graph(make_line_W(2))


def _idata_for_model(m: SEMProbit) -> xr.DataTree:
    n = m._X.shape[0]
    r = m._m
    posterior = {
        "beta": np.array(
            [[[0.2] * m._X.shape[1], [0.21] * m._X.shape[1]]], dtype=float
        ),
        "unit_effect": np.array([[[0.1] * r, [0.11] * r]], dtype=float),
        "p": np.array([[[0.6] * n, [0.61] * n]], dtype=float),
        "lam": np.array([[0.1, 0.11]], dtype=float),
        "unit_effect_sd": np.array([[0.9, 0.91]], dtype=float),
    }
    return az.from_dict({"posterior": posterior})


def test_the_design():
    m = SEMProbit(unit_effects="random", y=Y4, X=X4, W=_W2(), units=[10, 10, 20, 20])
    assert m._design == "random"
    assert m._gibbs_key is None
    assert m._unit_codes.tolist() == [0, 0, 1, 1]
    assert m._unit_names == ["10", "20"]


def test_unit_W_warns_for_non_row_standardized_sparse():
    W = sp.csr_matrix(np.array([[0.0, 2.0], [1.0, 0.0]], dtype=float))
    with pytest.warns(UserWarning, match="row-standardized"):
        dense = SEMProbit._as_dense_unit_W(W)
    assert dense.shape == (2, 2)


def test_unit_W_rejects_legacy_and_dense_types():
    with pytest.raises(TypeError, match="legacy libpysal.weights.W"):
        SEMProbit._as_dense_unit_W(_LegacyW())
    with pytest.raises(TypeError, match="Graph or a scipy sparse"):
        SEMProbit._as_dense_unit_W(np.eye(2))


def test_constructor_validation_errors():
    W = _W2()
    with pytest.raises(ValueError, match="need unit_col"):
        SEMProbit(unit_effects="random", y=Y4, X=X4, W=W)
    with pytest.raises(ValueError, match="units must have one entry"):
        SEMProbit(unit_effects="random", y=Y4, X=X4, W=W, units=[0, 1, 0])
    with pytest.raises(ValueError, match="y must be binary"):
        SEMProbit(
            unit_effects="random",
            y=np.array([0, 2, 0, 1], dtype=float),
            X=X4,
            W=W,
            units=[0, 0, 1, 1],
        )
    with pytest.raises(ValueError, match="must match W dimension"):
        SEMProbit(unit_effects="random", y=Y4, X=X4, W=W, units=[0, 1, 2, 2])
    with pytest.raises(NotImplementedError, match="robust=True"):
        SEMProbit(
            unit_effects="random", y=Y4, X=X4, W=W, units=[0, 0, 1, 1], robust=True
        )


def test_formula_mode_validation_and_parsing():
    df = pd.DataFrame(
        {"y": [0, 1, 0, 1], "x": [1.0, 2.0, 3.0, 4.0], "unit": ["a", "a", "b", "b"]}
    )
    W = _W2()
    with pytest.raises(ValueError, match="data is required"):
        SEMProbit(unit_effects="random", formula="y ~ x", W=W, unit_col="unit")
    model = SEMProbit(
        unit_effects="random", formula="y ~ x", data=df, W=W, unit_col="unit"
    )
    assert model._X.shape[0] == 4
    assert model._y.shape[0] == 4
    assert model._unit_names == ["a", "b"]


def test_build_model_requires_predictor_columns():
    model = SEMProbit(
        unit_effects="random", y=Y4, X=np.empty((4, 0)), W=_W2(), units=[0, 0, 1, 1]
    )
    with pytest.raises(ValueError, match="at least one predictor"):
        model._build_pymc_model()


def test_post_fit_accessors_and_summary_labels(monkeypatch):
    model = SEMProbit(unit_effects="random", y=Y4, X=X4, W=_W2(), units=[0, 0, 1, 1])

    with pytest.raises(RuntimeError, match="Model has not been fit yet"):
        model.summary()
    with pytest.raises(RuntimeError, match="Model has not been fit yet"):
        model.random_effects_mean()
    with pytest.raises(RuntimeError, match="Model has not been fit yet"):
        model.fitted_probabilities()

    fake_idata = _idata_for_model(model)
    monkeypatch.setattr(pm, "sample", lambda **_kwargs: fake_idata)
    out = model.fit(draws=2, tune=1, chains=1, progressbar=False)
    assert out is fake_idata
    assert model.pymc_model is not None

    def _fake_summary(_idata, var_names=None, **_kwargs):
        idx = ["beta[Intercept]", "unit_effect[a]", "lam"]
        return pd.DataFrame({"mean": [0.2, 0.1, 0.1]}, index=idx)

    monkeypatch.setattr(az, "summary", _fake_summary)
    smry = model.summary(var_names=["beta", "unit_effect", "lam"])
    assert list(smry.index) == ["Intercept", "unit_effect:a", "lam"]

    c_mean = model.random_effects_mean()
    p_hat = model.fitted_probabilities()
    assert isinstance(c_mean, pd.Series)
    assert c_mean.shape[0] == 2
    assert p_hat.shape[0] == 4
    assert np.all((p_hat >= 0.0) & (p_hat <= 1.0))


def test_probability_impacts_are_the_average_marginal_effects(monkeypatch):
    """``mean(phi(x'beta + c[unit])) * beta_k``, with no indirect part."""
    model = SEMProbit(unit_effects="random", y=Y4, X=X4, W=_W2(), units=[0, 0, 1, 1])
    monkeypatch.setattr(pm, "sample", lambda **_kwargs: _idata_for_model(model))
    model.fit(draws=2, tune=1, chains=1, progressbar=False)
    df = model.impacts(scale="probability")
    eta = np.array([[0.2, 0.21]]).T * X4.sum(1) + np.array([[0.1, 0.11]]).T
    ame = (np.exp(-0.5 * eta**2) / np.sqrt(2 * np.pi)).mean(1) * [0.2, 0.21]
    assert float(df["direct"].iloc[0]) == pytest.approx(ame.mean())
    assert float(df["indirect"].iloc[0]) == 0.0
    assert float(model.impacts()["direct"].iloc[0]) == pytest.approx(0.205)


def test_the_simulator_draws_repeated_units():
    d = dgp.simulate_sem_probit(n_side=3, seed=0, unit_effects="random")
    assert d["units"].tolist() == np.repeat(np.arange(9), 25).tolist()
    assert d["params_true"]["unit_effect_sd"] == 0.5
    d = dgp.simulate_sem_probit(
        n_side=3, seed=0, unit_effects="random", obs_per_unit=4, unit_effect_sd=0.8
    )
    assert d["y"].shape == (36,) and d["params_true"]["unit_effect_sd"] == 0.8
    with pytest.raises(ValueError, match="obs_per_unit needs"):
        dgp.simulate_sem_probit(n_side=3, obs_per_unit=4)
    with pytest.raises(NotImplementedError, match="not fixed"):
        dgp.simulate_sem_probit(n_side=3, unit_effects="fixed")


@pytest.mark.slow
@pytest.mark.recovery
def test_recovery_and_fitted_probabilities(rng):
    lam, beta, sd = 0.35, np.array([0.3, 1.0]), 0.8
    d = dgp.simulate_sem_probit(
        W=sp.csr_matrix(make_line_W(10)),
        lam=lam,
        beta=beta,
        rng=rng,
        unit_effects="random",
        unit_effect_sd=sd,
    )
    model = SEMProbit(
        unit_effects="random",
        y=d["y"],
        X=d["X"],
        W=W_to_graph(make_line_W(10)),
        units=d["units"],
    )
    post = model.fit(**SAMPLE_KWARGS).posterior
    assert abs(float(post["lam"].mean()) - lam) < 0.30
    beta_hat = post["beta"].mean(("chain", "draw")).values
    assert np.all(np.abs(beta_hat - beta) < 0.45)
    assert abs(float(post["unit_effect_sd"].mean()) - sd) < 0.40
    p_hat = model.fitted_probabilities()
    assert np.all((p_hat >= 0.0) & (p_hat <= 1.0))
