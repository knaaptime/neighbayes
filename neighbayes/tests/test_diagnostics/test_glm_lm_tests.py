"""The Bayesian LM tests of the logit and NB families.

The five score tests reproduce ``esda_triad/triad/score.py`` at a shared null
fit (the reference values below were computed by it), the suites run at the
GLM and reduced-form SAR nulls, and a slow simulation checks their null size.
"""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from neighbayes.diagnostics.lmtests import (
    bayesian_glm_lm_error_test,
    bayesian_glm_lm_lag_test,
    bayesian_glm_lm_sarma_test,
    bayesian_glm_lm_wx_test,
    bayesian_glm_moran_working_test,
    bayesian_glm_robust_lm_error_test,
    bayesian_glm_robust_lm_lag_test,
)
from neighbayes.models import Logit, NegBin, SARLogit, SARNegBin
from neighbayes.tests.helpers import W_to_graph, make_rook_W

_NULL_LABELS = {
    "LM-Lag",
    "LM-Error",
    "Robust-LM-Lag",
    "Robust-LM-Error",
    "LM-SARMA",
    "LM-WX",
    "Moran-Working",
}
_FIVE = {
    "lm_full": bayesian_glm_lm_lag_test,
    "lm_err": bayesian_glm_lm_error_test,
    "lm_spill": bayesian_glm_robust_lm_lag_test,
    "rs_err": bayesian_glm_robust_lm_error_test,
    "lm_joint": bayesian_glm_lm_sarma_test,
}


def _at(model, beta, r=None):
    """``model`` with a one-draw posterior at ``(beta, r)``: its statistics are
    the frequentist score tests at that point."""
    post = {"beta": (("chain", "draw", "coefficient"), np.asarray(beta)[None, None])}
    if r is not None:
        post["alpha"] = (("chain", "draw"), np.array([[r]]))
    model._idata = xr.DataTree.from_dict({"posterior": xr.Dataset(post)})
    return model


# ---------------------------------------------------------------------------
# DGPs
# ---------------------------------------------------------------------------


def _grid_X(rng: np.random.Generator, n: int) -> tuple[np.ndarray, np.ndarray]:
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    X = np.column_stack([np.ones(n), x1, x2])
    return X


def _logit_dgp(seed: int, n_side: int, rho: float = 0.0):
    """Generate y from (possibly SAR) logit with linear predictor eta = S(rho) X beta."""
    rng = np.random.default_rng(seed)
    n = n_side * n_side
    W = make_rook_W(n_side)  # dense, row-standardized
    X = _grid_X(rng, n)
    beta = np.array([0.2, 0.8, -0.6])
    Xb = X @ beta
    if rho == 0.0:
        eta = Xb
    else:
        A = np.eye(n) - rho * W
        eta = np.linalg.solve(A, Xb)
    p = 1.0 / (1.0 + np.exp(-eta))
    y = (rng.uniform(size=n) < p).astype(float)
    return y, X, W


def _negbin_dgp(seed: int, n_side: int, rho: float = 0.0, alpha_true: float = 3.0):
    """SAR-NB on the linear predictor of the log-mean."""
    rng = np.random.default_rng(seed)
    n = n_side * n_side
    W = make_rook_W(n_side)
    X = _grid_X(rng, n)
    beta = np.array([0.3, 0.5, -0.4])
    Xb = X @ beta
    if rho == 0.0:
        eta = Xb
    else:
        A = np.eye(n) - rho * W
        eta = np.linalg.solve(A, Xb)
    mu = np.exp(eta)
    y = rng.negative_binomial(alpha_true, alpha_true / (alpha_true + mu)).astype(float)
    return y, X, W


# ---------------------------------------------------------------------------
# Logit
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def logit_null_fit():
    y, X, W = _logit_dgp(seed=0, n_side=12)  # n=144
    m = Logit(y=y, X=X, W=W_to_graph(W), durbin_vars=["x1", "x2"])
    m.fit(draws=300, tune=300, chains=2, progressbar=False, random_seed=11)
    return m


@pytest.fixture(scope="module")
def logit_sar_fit():
    y, X, W = _logit_dgp(seed=1, n_side=15, rho=0.7)
    m = Logit(y=y, X=X, W=W_to_graph(W), durbin_vars=["x1", "x2"])
    m.fit(draws=400, tune=400, chains=2, progressbar=False, random_seed=12)
    return m


def test_glm_lm_error_logit_size(logit_null_fit):
    res = bayesian_glm_lm_error_test(logit_null_fit)
    assert res.df == 1
    # Under H0 the mean should be near 1 (chi^2_1); be generous
    assert res.mean < 6.0
    assert res.bayes_pvalue > 0.01


def test_glm_spillover_logit_power(logit_sar_fit):
    """Reduced-form SAR data: the spillover test, the reduced form's score, fires."""
    res = bayesian_glm_robust_lm_lag_test(logit_sar_fit)
    assert res.df == 1
    assert res.mean > 3.84  # 95% chi^2_1 critical value
    assert res.bayes_pvalue < 0.05


def test_glm_lm_wx_logit_runs(logit_null_fit):
    res = bayesian_glm_lm_wx_test(logit_null_fit)
    assert res.df == 2
    assert res.lm_samples.shape[0] == 600
    assert np.isfinite(res.mean)


def test_logit_spatial_diagnostics_dispatch(logit_null_fit):
    df = logit_null_fit.spatial_diagnostics()
    assert set(df.index) == _NULL_LABELS


# ---------------------------------------------------------------------------
# NegBin
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def negbin_null_fit():
    y, X, W = _negbin_dgp(seed=2, n_side=12)
    m = NegBin(y=y, X=X, W=W_to_graph(W), durbin_vars=["x1", "x2"])
    m.fit(draws=300, tune=300, chains=2, progressbar=False, random_seed=21)
    return m


@pytest.fixture(scope="module")
def negbin_sar_fit():
    y, X, W = _negbin_dgp(seed=3, n_side=15, rho=0.7)
    m = NegBin(y=y, X=X, W=W_to_graph(W), durbin_vars=["x1", "x2"])
    m.fit(draws=400, tune=400, chains=2, progressbar=False, random_seed=22)
    return m


def test_glm_lm_error_negbin_size(negbin_null_fit):
    res = bayesian_glm_lm_error_test(negbin_null_fit)
    assert res.df == 1
    assert res.mean < 6.0


def test_glm_spillover_negbin_power(negbin_sar_fit):
    res = bayesian_glm_robust_lm_lag_test(negbin_sar_fit)
    assert res.df == 1
    assert res.mean > 3.84
    assert res.bayes_pvalue < 0.05


def test_glm_lm_wx_negbin_runs(negbin_null_fit):
    res = bayesian_glm_lm_wx_test(negbin_null_fit)
    assert res.df == 2
    assert np.isfinite(res.mean)


def test_negbin_spatial_diagnostics_dispatch(negbin_null_fit):
    df = negbin_null_fit.spatial_diagnostics()
    assert set(df.index) == _NULL_LABELS


# ---------------------------------------------------------------------------
# Guardrails
# ---------------------------------------------------------------------------


def test_glm_lm_error_requires_W():
    rng = np.random.default_rng(7)
    n = 60
    X = _grid_X(rng, n)
    eta = X @ np.array([0.0, 0.5, -0.3])
    y = (rng.uniform(size=n) < 1.0 / (1.0 + np.exp(-eta))).astype(float)
    m = Logit(y=y, X=X)
    m.fit(draws=100, tune=100, chains=2, progressbar=False, random_seed=3)
    with pytest.raises(ValueError, match="weights matrix W"):
        bayesian_glm_lm_error_test(m)


# ---------------------------------------------------------------------------
# The five score tests reproduce esda_triad at a shared null fit
# ---------------------------------------------------------------------------

# A 10 x 10 rook lattice, X and y drawn below; the ML null fits and statistics
# are ``triad.families.fit_null`` and ``triad.score.score_tests`` on the same
# data (esda_triad 253cb87).
_TRIAD = {
    "logit": {
        "beta": [0.16594254986005025, 0.8443204814897725, -0.3713546900279647],
        "r": None,
        "lm_full": 0.27235510888934594,
        "lm_err": 0.1709250446328548,
        "lm_spill": 0.12631893911071077,
        "rs_err": 0.024888874854219672,
        "lm_joint": 0.29724398374356553,
    },
    "negbin": {
        "beta": [0.6841541670747818, 0.413957789236465, -0.3302213288575717],
        "r": 1.4973354019848801,
        "lm_full": 0.2765557668905371,
        "lm_err": 0.23571250471858557,
        "lm_spill": 3.5093161714778773,
        "rs_err": 3.468472909305926,
        "lm_joint": 3.745028676196463,
    },
}


def _triad_data():
    W = make_rook_W(10)
    rng = np.random.default_rng(2026)
    X = np.column_stack([np.ones(100), rng.normal(size=100), rng.normal(size=100)])
    p = 1 / (1 + np.exp(-(X @ np.array([0.2, 0.8, -0.6]))))
    y_logit = (rng.random(100) < p).astype(float)
    mu = np.exp(X @ np.array([0.5, 0.4, -0.3]))
    y_nb = rng.negative_binomial(2.0, 2.0 / (2.0 + mu)).astype(float)
    return W, X, {"logit": y_logit, "negbin": y_nb}


@pytest.mark.parametrize("family", ["logit", "negbin"])
def test_score_tests_reproduce_triad(family):
    W, X, ys = _triad_data()
    cls = Logit if family == "logit" else NegBin
    ref = _TRIAD[family]
    model = _at(cls(y=ys[family], X=X, W=W_to_graph(W)), ref["beta"], ref["r"])
    for key, test in _FIVE.items():
        np.testing.assert_allclose(test(model).mean, ref[key], rtol=1e-9, err_msg=key)


def test_score_decomposition_identities():
    """LM-Lag + Robust-LM-Error = LM-SARMA = Robust-LM-Lag + LM-Error."""
    W, X, ys = _triad_data()
    ref = _TRIAD["negbin"]
    model = _at(NegBin(y=ys["negbin"], X=X, W=W_to_graph(W)), ref["beta"], ref["r"])
    stat = {key: test(model).mean for key, test in _FIVE.items()}
    np.testing.assert_allclose(stat["lm_full"] + stat["rs_err"], stat["lm_joint"])
    np.testing.assert_allclose(stat["lm_spill"] + stat["lm_err"], stat["lm_joint"])


# ---------------------------------------------------------------------------
# At a reduced-form SAR null
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def sar_logit_fit():
    y, X, W = _logit_dgp(seed=4, n_side=12, rho=0.4)
    m = SARLogit(y=y, X=X, W=W_to_graph(W))
    m.fit(draws=200, tune=200, chains=1, progressbar=False, random_seed=5, n_jobs=1)
    return m


@pytest.fixture(scope="module")
def sar_negbin_fit():
    y, X, W = _negbin_dgp(seed=5, n_side=12, rho=0.4)
    m = SARNegBin(y=y, X=X, W=W_to_graph(W))
    m.fit(draws=200, tune=200, chains=1, progressbar=False, random_seed=6, n_jobs=1)
    return m


@pytest.mark.parametrize("fit", ["sar_logit_fit", "sar_negbin_fit"])
def test_sar_null_suite(request, fit):
    model = request.getfixturevalue(fit)
    df = model.spatial_diagnostics()
    assert set(df.index) == {"LM-Error", "LM-WX"}
    assert np.isfinite(df["statistic"]).all()
    assert bayesian_glm_lm_error_test(model).details["null"] == "reduced-form SAR"


@pytest.mark.parametrize(
    "test",
    [
        bayesian_glm_lm_lag_test,
        bayesian_glm_robust_lm_lag_test,
        bayesian_glm_robust_lm_error_test,
        bayesian_glm_lm_sarma_test,
        bayesian_glm_moran_working_test,
    ],
)
def test_null_level_tests_refuse_a_fitted_lag(sar_logit_fit, test):
    with pytest.raises(ValueError, match="aspatial fit"):
        test(sar_logit_fit)


# ---------------------------------------------------------------------------
# Null size at the ML fit (the esda_triad calibration)
# ---------------------------------------------------------------------------


def _ml_beta(family, y, X, r=None):
    """Newton (logit) or Fisher scoring at known ``r`` (NB) for the null fit."""
    beta = np.zeros(X.shape[1])
    for _ in range(100):
        eta = X @ beta
        if family == "logit":
            p = 1 / (1 + np.exp(-eta))
            e, w = y - p, p * (1 - p)
        else:
            mu = np.exp(eta)
            e, w = r * (y - mu) / (r + mu), r * mu / (r + mu)
        step = np.linalg.solve(X.T @ (w[:, None] * X), X.T @ e)
        beta += step
        if np.abs(step).max() < 1e-10:
            break
    return beta


@pytest.mark.slow
@pytest.mark.parametrize("family", ["logit", "negbin"])
def test_null_size(family):
    """Rejection rates at 5% stay within 3.5 Monte Carlo s.e. of nominal."""
    reps, n_side, r = 300, 20, 2.0
    n = n_side * n_side
    G = W_to_graph(make_rook_W(n_side))
    rejections = dict.fromkeys(_FIVE, 0)
    for rep in range(reps):
        rng = np.random.default_rng(10_000 + rep)
        X = np.column_stack([np.ones(n), rng.normal(size=n), rng.normal(size=n)])
        if family == "logit":
            y = (rng.random(n) < 1 / (1 + np.exp(-X @ [0.2, 0.8, -0.6]))).astype(float)
            model, fit_r = Logit(y=y, X=X, W=G), None
        else:
            mu = np.exp(X @ np.array([0.5, 0.4, -0.3]))
            y = rng.negative_binomial(r, r / (r + mu)).astype(float)
            model, fit_r = NegBin(y=y, X=X, W=G), r
        _at(model, _ml_beta(family, y, X, fit_r), fit_r)
        for key, test in _FIVE.items():
            rejections[key] += test(model).bayes_pvalue < 0.05
    se = np.sqrt(0.05 * 0.95 / reps)
    for key, count in rejections.items():
        assert abs(count / reps - 0.05) < 3.5 * se, (key, count / reps)
