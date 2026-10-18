"""End-to-end tests for the PG-augmented (GLM) Bayesian LM diagnostics."""

from __future__ import annotations

import numpy as np
import pytest

from neighbayes.diagnostics.lmtests import (
    bayesian_glm_lm_error_test,
    bayesian_glm_lm_lag_test,
    bayesian_glm_lm_wx_test,
)
from neighbayes.models import Logit, NegBin
from neighbayes.tests.helpers import W_to_graph, make_rook_W

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


def test_glm_lm_lag_logit_power(logit_sar_fit):
    res = bayesian_glm_lm_lag_test(logit_sar_fit)
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
    assert set(df.index) == {"LM-Lag", "LM-Error", "LM-WX"}


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


def test_glm_lm_lag_negbin_power(negbin_sar_fit):
    res = bayesian_glm_lm_lag_test(negbin_sar_fit)
    assert res.df == 1
    assert res.mean > 3.84
    assert res.bayes_pvalue < 0.05


def test_glm_lm_wx_negbin_runs(negbin_null_fit):
    res = bayesian_glm_lm_wx_test(negbin_null_fit)
    assert res.df == 2
    assert np.isfinite(res.mean)


def test_negbin_spatial_diagnostics_dispatch(negbin_null_fit):
    df = negbin_null_fit.spatial_diagnostics()
    assert set(df.index) == {"LM-Lag", "LM-Error", "LM-WX"}


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
