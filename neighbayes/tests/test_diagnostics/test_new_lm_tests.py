"""Fast unit tests for new Bayesian LM test functions.

Tests bayesian_robust_lm_lag_test, bayesian_robust_lm_error_test,
bayesian_lm_error_sdm_test, bayesian_lm_lag_sdem_test,
bayesian_panel_lm_error_sdm_test and bayesian_panel_lm_lag_sdem_test using
mock models with synthetic posteriors.
"""

from __future__ import annotations

import numpy as np
import pytest

from neighbayes.diagnostics.lmtests import (
    bayesian_lm_error_sdm_test,
    bayesian_lm_lag_sdem_test,
    bayesian_panel_lm_error_sdm_test,
    bayesian_panel_lm_lag_sdem_test,
    bayesian_robust_lm_error_test,
    bayesian_robust_lm_lag_test,
)
from neighbayes.tests.test_diagnostics._lm_mocks import exact_logdet_grad

# ---------------------------------------------------------------------------
# Helpers: mock models with synthetic posteriors
# ---------------------------------------------------------------------------


class _MockParamArray:
    """Mimics xarray DataArray: subscriptable and has .values."""

    def __init__(self, values):
        self._values = np.asarray(values)

    @property
    def values(self):
        return self._values

    def __getitem__(self, key):
        return self._values[key]


class _MockPosterior(dict):
    """Mimics xarray Dataset: subscriptable by param name."""

    pass


class _MockInferenceData:
    def __init__(self, posterior):
        self.posterior = posterior


def _make_ols_mock(n: int = 20, k: int = 2, draws: int = 50):
    """Create a mock OLS model with synthetic posterior draws."""
    rng = np.random.default_rng(42)

    X = rng.standard_normal((n, k))
    y = X @ np.array([1.0, 2.0]) + rng.standard_normal(n) * 0.5

    # Row-standardized W
    W = np.zeros((n, n))
    for i in range(n):
        if i > 0:
            W[i, i - 1] = 1.0
        if i < n - 1:
            W[i, i + 1] = 1.0
    rs = W.sum(axis=1, keepdims=True)
    rs[rs == 0] = 1.0
    W = W / rs

    import scipy.sparse as sp

    W_sp = sp.csr_matrix(W)
    Wy = W @ y
    T_ww = float(np.trace(W.T @ W + W @ W))

    # Synthetic posterior
    beta_draws = rng.standard_normal((draws, k)) * 0.1 + np.array([1.0, 2.0])
    sigma_draws = np.abs(rng.standard_normal(draws)) * 0.5 + 0.5

    posterior = _MockPosterior()
    posterior["beta"] = _MockParamArray(beta_draws[:, None, :])
    posterior["sigma"] = _MockParamArray(sigma_draws[:, None])
    idata = _MockInferenceData(posterior)

    class MockModel:
        pass

    model = MockModel()
    model._y = y
    model._X = X
    model._Wy = Wy
    model._W_sparse = W_sp
    model._T_ww = T_ww
    model._nonintercept_indices = list(range(X.shape[1]))
    model._logdet_grad_numpy_vec_fn = exact_logdet_grad(W_sp)
    model.inference_data = idata

    return model


def _make_sdm_mock(n: int = 20, k: int = 2, draws: int = 50):
    """Create a mock SDM model with synthetic posterior draws."""
    rng = np.random.default_rng(42)

    X = rng.standard_normal((n, k))
    W = np.zeros((n, n))
    for i in range(n):
        if i > 0:
            W[i, i - 1] = 1.0
        if i < n - 1:
            W[i, i + 1] = 1.0
    rs = W.sum(axis=1, keepdims=True)
    rs[rs == 0] = 1.0
    W = W / rs

    WX = W @ X
    Wy = W @ (X @ np.array([1.0, 2.0]) + rng.standard_normal(n) * 0.5)
    y = (
        X @ np.array([1.0, 2.0])
        + WX @ np.array([0.3, -0.2])
        + 0.5 * Wy
        + rng.standard_normal(n) * 0.5
    )

    import scipy.sparse as sp

    W_sp = sp.csr_matrix(W)
    T_ww = float(np.trace(W.T @ W + W @ W))

    beta_draws = rng.standard_normal((draws, 2 * k)) * 0.1 + np.array(
        [1.0, 2.0, 0.3, -0.2]
    )
    rho_draws = rng.standard_normal(draws) * 0.05 + 0.5
    sigma_draws = np.abs(rng.standard_normal(draws)) * 0.5 + 0.5

    posterior = _MockPosterior()
    posterior["beta"] = _MockParamArray(beta_draws[:, None, :])
    posterior["rho"] = _MockParamArray(rho_draws[:, None])
    posterior["sigma"] = _MockParamArray(sigma_draws[:, None])
    idata = _MockInferenceData(posterior)

    class MockModel:
        pass

    model = MockModel()
    model._y = y
    model._X = X
    model._WX = WX
    model._Wy = Wy
    model._W_sparse = W_sp
    model._T_ww = T_ww
    model._nonintercept_indices = list(range(X.shape[1]))
    model._logdet_grad_numpy_vec_fn = exact_logdet_grad(W_sp)
    model.inference_data = idata

    return model


def _make_sdem_mock(n: int = 20, k: int = 2, draws: int = 50):
    """Create a mock SDEM model with synthetic posterior draws."""
    rng = np.random.default_rng(42)

    X = rng.standard_normal((n, k))
    W = np.zeros((n, n))
    for i in range(n):
        if i > 0:
            W[i, i - 1] = 1.0
        if i < n - 1:
            W[i, i + 1] = 1.0
    rs = W.sum(axis=1, keepdims=True)
    rs[rs == 0] = 1.0
    W = W / rs

    WX = W @ X
    y = (
        X @ np.array([1.0, 2.0])
        + WX @ np.array([0.3, -0.2])
        + rng.standard_normal(n) * 0.5
    )

    import scipy.sparse as sp

    W_sp = sp.csr_matrix(W)
    Wy = W @ y
    T_ww = float(np.trace(W.T @ W + W @ W))

    beta_draws = rng.standard_normal((draws, 2 * k)) * 0.1 + np.array(
        [1.0, 2.0, 0.3, -0.2]
    )
    lam_draws = rng.standard_normal(draws) * 0.05 + 0.3
    sigma_draws = np.abs(rng.standard_normal(draws)) * 0.5 + 0.5

    posterior = _MockPosterior()
    posterior["beta"] = _MockParamArray(beta_draws[:, None, :])
    posterior["lam"] = _MockParamArray(lam_draws[:, None])
    posterior["sigma"] = _MockParamArray(sigma_draws[:, None])
    idata = _MockInferenceData(posterior)

    class MockModel:
        pass

    model = MockModel()
    model._y = y
    model._X = X
    model._WX = WX
    model._Wy = Wy
    model._W_sparse = W_sp
    model._W_dense = W
    model._T_ww = T_ww
    model._nonintercept_indices = list(range(X.shape[1]))
    model._logdet_grad_numpy_vec_fn = exact_logdet_grad(W_sp)
    model.inference_data = idata

    return model


def _make_panel_sdm_mock(N: int = 5, T: int = 4, k: int = 2, draws: int = 50):
    """Create a mock panel SDM model with synthetic posterior draws."""
    rng = np.random.default_rng(42)
    n = N * T

    X = rng.standard_normal((n, k))
    W = np.zeros((N, N))
    for i in range(N):
        if i > 0:
            W[i, i - 1] = 1.0
        if i < N - 1:
            W[i, i + 1] = 1.0
    rs = W.sum(axis=1, keepdims=True)
    rs[rs == 0] = 1.0
    W = W / rs

    import scipy.sparse as sp

    W_block = sp.kron(sp.eye(T), W, format="csr")
    W_sp = sp.csr_matrix(W)  # N×N for _panel_spatial_lag
    WX = np.asarray(W_block @ X)
    y = (
        X @ np.array([1.0, 2.0])
        + WX @ np.array([0.3, -0.2])
        + rng.standard_normal(n) * 0.5
    )
    Wy = np.asarray(W_block @ y)
    T_ww = float(np.trace(W.T @ W + W @ W))

    beta_draws = rng.standard_normal((draws, 2 * k)) * 0.1 + np.array(
        [1.0, 2.0, 0.3, -0.2]
    )
    rho_draws = rng.standard_normal(draws) * 0.05 + 0.5
    sigma_draws = np.abs(rng.standard_normal(draws)) * 0.5 + 0.5

    posterior = _MockPosterior()
    posterior["beta"] = _MockParamArray(beta_draws[:, None, :])
    posterior["rho"] = _MockParamArray(rho_draws[:, None])
    posterior["sigma"] = _MockParamArray(sigma_draws[:, None])
    idata = _MockInferenceData(posterior)

    class MockModel:
        pass

    model = MockModel()
    model._y = y
    model._X = X
    model._WX = WX
    model._Wy = Wy
    model._W_sparse = W_sp  # N×N for _panel_spatial_lag
    model._T_ww = T_ww
    model._N = N
    model._T = T
    model._is_panel = True
    model._design = "static"
    model._jacobian_T, model._jacobian_shift, model._n_effective = T, 0.0, N * T
    model._nonintercept_indices = list(range(X.shape[1]))
    model._logdet_grad_numpy_vec_fn = exact_logdet_grad(W_sp)
    model.inference_data = idata

    return model


def _make_panel_sdem_mock(N: int = 5, T: int = 4, k: int = 2, draws: int = 50):
    """Create a mock panel SDEM model with synthetic posterior draws."""
    rng = np.random.default_rng(42)
    n = N * T

    X = rng.standard_normal((n, k))
    W = np.zeros((N, N))
    for i in range(N):
        if i > 0:
            W[i, i - 1] = 1.0
        if i < N - 1:
            W[i, i + 1] = 1.0
    rs = W.sum(axis=1, keepdims=True)
    rs[rs == 0] = 1.0
    W = W / rs

    import scipy.sparse as sp

    W_block = sp.kron(sp.eye(T), W, format="csr")
    W_sp = sp.csr_matrix(W)  # N×N for _panel_spatial_lag
    WX = np.asarray(W_block @ X)
    y = (
        X @ np.array([1.0, 2.0])
        + WX @ np.array([0.3, -0.2])
        + rng.standard_normal(n) * 0.5
    )
    Wy = np.asarray(W_block @ y)
    T_ww = float(np.trace(W.T @ W + W @ W))

    beta_draws = rng.standard_normal((draws, 2 * k)) * 0.1 + np.array(
        [1.0, 2.0, 0.3, -0.2]
    )
    lam_draws = rng.standard_normal(draws) * 0.05 + 0.3
    sigma_draws = np.abs(rng.standard_normal(draws)) * 0.5 + 0.5

    posterior = _MockPosterior()
    posterior["beta"] = _MockParamArray(beta_draws[:, None, :])
    posterior["lam"] = _MockParamArray(lam_draws[:, None])
    posterior["sigma"] = _MockParamArray(sigma_draws[:, None])
    idata = _MockInferenceData(posterior)

    class MockModel:
        pass

    model = MockModel()
    model._y = y
    model._X = X
    model._WX = WX
    model._Wy = Wy
    model._W_sparse = W_sp  # N×N for _panel_spatial_lag
    model._W_dense = np.kron(np.eye(T), W)  # (N*T)×(N*T) for whitening
    model._T_ww = T_ww
    model._N = N
    model._T = T
    model._is_panel = True
    model._design = "static"
    model._jacobian_T, model._jacobian_shift, model._n_effective = T, 0.0, N * T
    model._nonintercept_indices = list(range(X.shape[1]))
    model._logdet_grad_numpy_vec_fn = exact_logdet_grad(W_sp)
    model.inference_data = idata

    return model


class TestRobustLmLag:
    def test_returns_result(self):
        model = _make_ols_mock()
        result = bayesian_robust_lm_lag_test(model)
        assert hasattr(result, "lm_samples")
        assert hasattr(result, "bayes_pvalue")
        assert len(result.lm_samples) > 0
        assert result.df == 1

    def test_lm_samples_nonnegative(self):
        model = _make_ols_mock()
        result = bayesian_robust_lm_lag_test(model)
        assert np.all(result.lm_samples >= -1e-10)  # allow tiny numerical noise


# ---------------------------------------------------------------------------
# bayesian_robust_lm_error_test
# ---------------------------------------------------------------------------


class TestRobustLmError:
    def test_returns_result(self):
        model = _make_ols_mock()
        result = bayesian_robust_lm_error_test(model)
        assert hasattr(result, "lm_samples")
        assert hasattr(result, "bayes_pvalue")
        assert result.df == 1

    def test_lm_samples_nonnegative(self):
        model = _make_ols_mock()
        result = bayesian_robust_lm_error_test(model)
        assert np.all(result.lm_samples >= -1e-10)


# ---------------------------------------------------------------------------
# bayesian_lm_error_sdm_test
# ---------------------------------------------------------------------------


class TestLmErrorSdm:
    def test_returns_result(self):
        model = _make_sdm_mock()
        result = bayesian_lm_error_sdm_test(model)
        assert hasattr(result, "lm_samples")
        assert result.df == 1

    def test_lm_samples_nonnegative(self):
        model = _make_sdm_mock()
        result = bayesian_lm_error_sdm_test(model)
        assert np.all(result.lm_samples >= -1e-10)


# ---------------------------------------------------------------------------
# bayesian_lm_lag_sdem_test
# ---------------------------------------------------------------------------


class TestLmLagSdem:
    def test_returns_result(self):
        model = _make_sdem_mock()
        result = bayesian_lm_lag_sdem_test(model)
        assert hasattr(result, "lm_samples")
        assert result.df == 1

    def test_lm_samples_nonnegative(self):
        model = _make_sdem_mock()
        result = bayesian_lm_lag_sdem_test(model)
        assert np.all(result.lm_samples >= -1e-10)


# ---------------------------------------------------------------------------
# bayesian_panel_lm_error_sdm_test
# ---------------------------------------------------------------------------


class TestPanelLmErrorSdm:
    def test_returns_result(self):
        model = _make_panel_sdm_mock()
        result = bayesian_panel_lm_error_sdm_test(model)
        assert hasattr(result, "lm_samples")
        assert result.df == 1

    def test_lm_samples_nonnegative(self):
        model = _make_panel_sdm_mock()
        result = bayesian_panel_lm_error_sdm_test(model)
        assert np.all(result.lm_samples >= -1e-10)


# ---------------------------------------------------------------------------
# bayesian_panel_lm_lag_sdem_test
# ---------------------------------------------------------------------------


class TestPanelLmLagSdem:
    def test_returns_result(self):
        model = _make_panel_sdem_mock()
        result = bayesian_panel_lm_lag_sdem_test(model)
        assert hasattr(result, "lm_samples")
        assert result.df == 1

    def test_lm_samples_nonnegative(self):
        model = _make_panel_sdem_mock()
        result = bayesian_panel_lm_lag_sdem_test(model)
        assert np.all(result.lm_samples >= -1e-10)
