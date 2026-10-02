from unittest.mock import MagicMock

import arviz as az
import numpy as np
import pandas as pd
import pytest

from neighbayes.diagnostics.lmtests import (
    BayesianLMTestResult,
    bayesian_lm_error_sdm_test,
    bayesian_lm_error_test,
    bayesian_lm_lag_sdem_test,
    bayesian_lm_lag_test,
    bayesian_lm_sdm_joint_test,
    bayesian_lm_slx_error_joint_test,
    bayesian_lm_wx_test,
    bayesian_panel_lm_error_sdm_test,
    bayesian_panel_lm_error_test,
    bayesian_panel_lm_lag_sdem_test,
    # Panel LM tests
    bayesian_panel_lm_lag_test,
    bayesian_panel_lm_sdm_joint_test,
    bayesian_panel_lm_slx_error_joint_test,
    bayesian_panel_lm_wx_test,
    bayesian_panel_robust_lm_error_sdem_test,
    bayesian_panel_robust_lm_error_test,
    bayesian_panel_robust_lm_lag_sdm_test,
    bayesian_panel_robust_lm_lag_test,
    bayesian_panel_robust_lm_wx_test,
    bayesian_robust_lm_error_sdem_test,
    bayesian_robust_lm_lag_sdm_test,
    bayesian_robust_lm_wx_test,
)


def make_sar_sem_data(n=20, seed=42):
    rng = np.random.default_rng(seed)
    X = np.column_stack([np.ones(n), rng.normal(size=n)])
    beta_true = np.array([1.0, 2.0])
    y = X @ beta_true + rng.normal(scale=0.5, size=n)
    # Simple row-normalized spatial weights (ring)
    W_dense = np.eye(n, k=1) + np.eye(n, k=-1)
    W_dense[0, -1] = W_dense[-1, 0] = 1
    W_dense = W_dense / W_dense.sum(axis=1, keepdims=True)
    import scipy.sparse as sp

    W_sparse = sp.csr_matrix(W_dense)
    return y, X, W_dense, W_sparse


def _make_mock_model(y, X, W_sparse, beta_noise=0.1, draws=100):
    """Build a mock model object that exposes the attributes expected by the
    Bayesian LM test functions."""
    n, k = X.shape
    beta_samples = np.tile(np.linalg.lstsq(X, y, rcond=None)[0], (draws, 1))
    rng = np.random.default_rng(42)
    beta_samples += rng.normal(scale=beta_noise, size=beta_samples.shape)

    # Pre-compute Wy from sparse W (as SpatialModel does)
    Wy = np.asarray(W_sparse @ y, dtype=np.float64)

    # Build InferenceData using az.from_dict (works with all ArViZ versions)
    idata = az.from_dict(
        {
            "posterior": {
                "beta": beta_samples[:, None, :],
                "sigma": np.ones(draws)[:, None],
            },
            "observed_data": {"y": y},
        }
    )

    model = MagicMock()
    model._y = y
    model._X = X
    model._WX = np.empty((n, 0), dtype=float)
    model._Wy = Wy
    model._W_sparse = W_sparse
    model._T_ww = float(W_sparse.power(2).sum() + W_sparse.multiply(W_sparse.T).sum())
    model.inference_data = idata
    return model


def test_bayesian_lm_lag_and_error_basic():
    y, X, W_dense, W_sparse = make_sar_sem_data(n=16)
    model = _make_mock_model(y, X, W_sparse, beta_noise=0.1, draws=200)

    # LM-lag (SAR)
    result_lag = bayesian_lm_lag_test(model)
    assert isinstance(result_lag.mean, float)
    assert result_lag.lm_samples.shape[0] == 200
    assert 0.0 <= result_lag.bayes_pvalue <= 1.0

    # LM-error (SEM)
    result_err = bayesian_lm_error_test(model)
    assert isinstance(result_err.mean, float)
    assert result_err.lm_samples.shape[0] == 200
    assert 0.0 <= result_err.bayes_pvalue <= 1.0


def test_bayesian_lm_lag_and_error_extreme():
    y, X, W_dense, W_sparse = make_sar_sem_data(n=10)
    model = _make_mock_model(y, X, W_sparse, beta_noise=0.0, draws=50)

    # All beta draws are the same (perfect fit), so LM should be very small
    result_lag = bayesian_lm_lag_test(model)
    assert np.all(result_lag.lm_samples >= 0)
    assert result_lag.mean < 10

    result_err = bayesian_lm_error_test(model)
    assert np.all(result_err.lm_samples >= 0)
    assert result_err.mean < 10


def test_bayesian_lm_result_df_field():
    """Test that BayesianLMTestResult has a df field with default 1."""
    y, X, W_dense, W_sparse = make_sar_sem_data(n=16)
    model = _make_mock_model(y, X, W_sparse, beta_noise=0.1, draws=50)
    result = bayesian_lm_lag_test(model)
    assert hasattr(result, "df")
    assert result.df == 1


# ---------------------------------------------------------------------------
# Mock model builders for SDM/SDEM variant tests
# ---------------------------------------------------------------------------


def _make_mock_ols_model_with_wx(y, X, WX, W_sparse, beta_noise=0.1, draws=100):
    """Build a mock OLS model with WX columns (for joint tests)."""
    n, k = X.shape
    Z = np.hstack([X, WX])  # OLS uses only X for residuals, but has WX available
    beta_ols = np.linalg.lstsq(X, y, rcond=None)[0]  # OLS on X only
    beta_samples = np.tile(beta_ols, (draws, 1))
    rng = np.random.default_rng(42)
    beta_samples += rng.normal(scale=beta_noise, size=beta_samples.shape)

    Wy = np.asarray(W_sparse @ y, dtype=np.float64)

    idata = az.from_dict(
        {
            "posterior": {
                "beta": beta_samples[:, None, :],
                "sigma": np.ones(draws)[:, None],
            },
            "observed_data": {"y": y},
        }
    )

    model = MagicMock()
    model._y = y
    model._X = X
    model._WX = WX
    model._Wy = Wy
    model._W_sparse = W_sparse
    model._T_ww = float(W_sparse.power(2).sum() + W_sparse.multiply(W_sparse.T).sum())
    model.inference_data = idata
    return model


def _make_mock_sar_model(y, X, WX, W_sparse, beta_noise=0.1, rho_noise=0.05, draws=100):
    """Build a mock SAR model with rho in the posterior."""
    n, k = X.shape
    beta_ols = np.linalg.lstsq(X, y, rcond=None)[0]
    beta_samples = np.tile(beta_ols, (draws, 1))
    rng = np.random.default_rng(42)
    beta_samples += rng.normal(scale=beta_noise, size=beta_samples.shape)
    rho_samples = rng.normal(scale=rho_noise, size=draws)  # near zero under H₀

    Wy = np.asarray(W_sparse @ y, dtype=np.float64)

    idata = az.from_dict(
        {
            "posterior": {
                "beta": beta_samples[:, None, :],
                "rho": rho_samples[:, None],
                "sigma": np.ones(draws)[:, None],
            },
            "observed_data": {"y": y},
        }
    )

    model = MagicMock()
    model._y = y
    model._X = X
    model._WX = WX
    model._Wy = Wy
    model._W_sparse = W_sparse
    model._T_ww = float(W_sparse.power(2).sum() + W_sparse.multiply(W_sparse.T).sum())
    model.inference_data = idata
    return model


def _make_mock_slx_model(y, X, WX, W_sparse, beta_noise=0.1, draws=100):
    """Build a mock SLX model (beta covers [X, WX], no spatial params)."""
    n, k = X.shape
    k_wx = WX.shape[1]
    Z = np.hstack([X, WX])
    beta_slx = np.linalg.lstsq(Z, y, rcond=None)[0]
    beta_samples = np.tile(beta_slx, (draws, 1))
    rng = np.random.default_rng(42)
    beta_samples += rng.normal(scale=beta_noise, size=beta_samples.shape)

    Wy = np.asarray(W_sparse @ y, dtype=np.float64)

    idata = az.from_dict(
        {
            "posterior": {
                "beta": beta_samples[:, None, :],
                "sigma": np.ones(draws)[:, None],
            },
            "observed_data": {"y": y},
        }
    )

    model = MagicMock()
    model._y = y
    model._X = X
    model._WX = WX
    model._Wy = Wy
    model._W_sparse = W_sparse
    model._T_ww = float(W_sparse.power(2).sum() + W_sparse.multiply(W_sparse.T).sum())
    model.inference_data = idata
    return model


def _make_data_with_wx(n=20, k_wx=2, seed=42):
    """Generate data with WX columns for multi-parameter tests."""
    rng = np.random.default_rng(seed)
    X = np.column_stack([np.ones(n), rng.normal(size=n)])
    WX = rng.normal(size=(n, k_wx))
    beta_true = np.array([1.0, 2.0])
    y = X @ beta_true + rng.normal(scale=0.5, size=n)

    W_dense = np.eye(n, k=1) + np.eye(n, k=-1)
    W_dense[0, -1] = W_dense[-1, 0] = 1
    W_dense = W_dense / W_dense.sum(axis=1, keepdims=True)
    import scipy.sparse as sp

    W_sparse = sp.csr_matrix(W_dense)
    return y, X, WX, W_dense, W_sparse


# ---------------------------------------------------------------------------
# Tests for bayesian_lm_wx_test
# ---------------------------------------------------------------------------


class TestBayesianLMWxTest:
    def test_basic_output(self):
        y, X, WX, W_dense, W_sparse = _make_data_with_wx(n=20, k_wx=2)
        model = _make_mock_sar_model(y, X, WX, W_sparse, draws=200)
        result = bayesian_lm_wx_test(model)

        assert isinstance(result, BayesianLMTestResult)
        assert result.lm_samples.shape[0] == 200
        assert result.df == 2  # k_wx = 2
        assert result.test_type == "bayesian_lm_wx"
        assert 0.0 <= result.bayes_pvalue <= 1.0
        assert np.all(result.lm_samples >= 0)

    def test_single_wx_column(self):
        y, X, WX, W_dense, W_sparse = _make_data_with_wx(n=20, k_wx=1)
        model = _make_mock_sar_model(y, X, WX, W_sparse, draws=100)
        result = bayesian_lm_wx_test(model)

        assert result.df == 1
        assert result.lm_samples.shape[0] == 100

    def test_no_wx_raises(self):
        y, X, W_dense, W_sparse = make_sar_sem_data(n=16)
        model = _make_mock_model(y, X, W_sparse, draws=50)
        with pytest.raises(ValueError, match="no WX columns"):
            bayesian_lm_wx_test(model)


# ---------------------------------------------------------------------------
# Tests for bayesian_lm_sdm_joint_test
# ---------------------------------------------------------------------------


class TestBayesianLMSDMJointTest:
    def test_basic_output(self):
        y, X, WX, W_dense, W_sparse = _make_data_with_wx(n=20, k_wx=2)
        model = _make_mock_ols_model_with_wx(y, X, WX, W_sparse, draws=200)
        result = bayesian_lm_sdm_joint_test(model)

        assert isinstance(result, BayesianLMTestResult)
        assert result.lm_samples.shape[0] == 200
        assert result.df == 3  # 1 + k_wx = 1 + 2
        assert result.test_type == "bayesian_lm_sdm_joint"
        assert 0.0 <= result.bayes_pvalue <= 1.0
        assert np.all(result.lm_samples >= 0)

    def test_single_wx_column(self):
        y, X, WX, W_dense, W_sparse = _make_data_with_wx(n=20, k_wx=1)
        model = _make_mock_ols_model_with_wx(y, X, WX, W_sparse, draws=100)
        result = bayesian_lm_sdm_joint_test(model)

        assert result.df == 2  # 1 + 1

    def test_zero_wx_columns(self):
        y, X, W_dense, W_sparse = make_sar_sem_data(n=16)
        model = _make_mock_model(y, X, W_sparse, draws=50)
        result = bayesian_lm_sdm_joint_test(model)

        assert result.df == 1  # 1 + 0
        assert result.test_type == "bayesian_lm_sdm_joint"


# ---------------------------------------------------------------------------
# Tests for bayesian_lm_slx_error_joint_test
# ---------------------------------------------------------------------------


class TestBayesianLMSLXErrorJointTest:
    def test_basic_output(self):
        y, X, WX, W_dense, W_sparse = _make_data_with_wx(n=20, k_wx=2)
        model = _make_mock_ols_model_with_wx(y, X, WX, W_sparse, draws=200)
        result = bayesian_lm_slx_error_joint_test(model)

        assert isinstance(result, BayesianLMTestResult)
        assert result.lm_samples.shape[0] == 200
        assert result.df == 3  # 1 + k_wx = 1 + 2
        assert result.test_type == "bayesian_lm_slx_error_joint"
        assert 0.0 <= result.bayes_pvalue <= 1.0
        assert np.all(result.lm_samples >= 0)

    def test_single_wx_column(self):
        y, X, WX, W_dense, W_sparse = _make_data_with_wx(n=20, k_wx=1)
        model = _make_mock_ols_model_with_wx(y, X, WX, W_sparse, draws=100)
        result = bayesian_lm_slx_error_joint_test(model)

        assert result.df == 2  # 1 + 1


# ---------------------------------------------------------------------------
# Tests for bayesian_robust_lm_lag_sdm_test
# ---------------------------------------------------------------------------


class TestBayesianRobustLMLagSDMTest:
    def test_basic_output(self):
        y, X, WX, W_dense, W_sparse = _make_data_with_wx(n=20, k_wx=2)
        model = _make_mock_slx_model(y, X, WX, W_sparse, draws=200)
        result = bayesian_robust_lm_lag_sdm_test(model)

        assert isinstance(result, BayesianLMTestResult)
        assert result.lm_samples.shape[0] == 200
        assert result.df == 1
        assert result.test_type == "bayesian_robust_lm_lag_sdm"
        assert 0.0 <= result.bayes_pvalue <= 1.0
        assert np.all(result.lm_samples >= 0)

    def test_no_wx_columns(self):
        """Robust test with no WX columns should still work (no adjustment)."""
        y, X, W_dense, W_sparse = make_sar_sem_data(n=16)
        model = _make_mock_model(y, X, W_sparse, draws=50)
        result = bayesian_robust_lm_lag_sdm_test(model)

        assert result.df == 1
        assert result.lm_samples.shape[0] == 50


# ---------------------------------------------------------------------------
# Tests for bayesian_robust_lm_wx_test
# ---------------------------------------------------------------------------


class TestBayesianRobustLMWXTest:
    def test_basic_output(self):
        y, X, WX, W_dense, W_sparse = _make_data_with_wx(n=20, k_wx=2)
        model = _make_mock_sar_model(y, X, WX, W_sparse, draws=200)
        result = bayesian_robust_lm_wx_test(model)

        assert isinstance(result, BayesianLMTestResult)
        assert result.lm_samples.shape[0] == 200
        assert result.df == 2  # k_wx = 2
        assert result.test_type == "bayesian_robust_lm_wx"
        assert 0.0 <= result.bayes_pvalue <= 1.0
        assert np.all(result.lm_samples >= 0)

    def test_no_wx_raises(self):
        y, X, W_dense, W_sparse = make_sar_sem_data(n=16)
        model = _make_mock_model(y, X, W_sparse, draws=50)
        with pytest.raises(ValueError, match="no WX columns"):
            bayesian_robust_lm_wx_test(model)


# ---------------------------------------------------------------------------
# Tests for bayesian_robust_lm_error_sdem_test
# ---------------------------------------------------------------------------


class TestBayesianRobustLMErrorSDEMTest:
    def test_basic_output(self):
        y, X, WX, W_dense, W_sparse = _make_data_with_wx(n=20, k_wx=2)
        model = _make_mock_slx_model(y, X, WX, W_sparse, draws=200)
        result = bayesian_robust_lm_error_sdem_test(model)

        assert isinstance(result, BayesianLMTestResult)
        assert result.lm_samples.shape[0] == 200
        assert result.df == 1
        assert result.test_type == "bayesian_robust_lm_error_sdem"
        assert 0.0 <= result.bayes_pvalue <= 1.0
        assert np.all(result.lm_samples >= 0)

    def test_no_wx_columns(self):
        """Robust test with no WX columns should still work (no adjustment)."""
        y, X, W_dense, W_sparse = make_sar_sem_data(n=16)
        model = _make_mock_model(y, X, W_sparse, draws=50)
        result = bayesian_robust_lm_error_sdem_test(model)

        assert result.df == 1
        assert result.lm_samples.shape[0] == 50


# ---------------------------------------------------------------------------
# SDM/SDEM-aware MANSAR-direction tests
# ---------------------------------------------------------------------------


def _make_mock_sdm_model(
    y, X, WX, W_sparse, draws=200, beta_noise=0.05, rho_noise=0.05
):
    """Build a mock SDM model: beta covers [X, WX] and rho is in posterior."""
    Z = np.hstack([X, WX])
    beta_mean = np.linalg.lstsq(Z, y, rcond=None)[0]
    rng = np.random.default_rng(42)
    beta_samples = np.tile(beta_mean, (draws, 1)) + rng.normal(
        scale=beta_noise, size=(draws, Z.shape[1])
    )
    rho_samples = rng.normal(scale=rho_noise, size=draws)

    Wy = np.asarray(W_sparse @ y, dtype=np.float64)
    idata = az.from_dict(
        {
            "posterior": {
                "beta": beta_samples[:, None, :],
                "rho": rho_samples[:, None],
                "sigma": np.ones(draws)[:, None],
            },
            "observed_data": {"y": y},
        }
    )

    model = MagicMock()
    model._y = y
    model._X = X
    model._WX = WX
    model._Wy = Wy
    model._W_sparse = W_sparse
    model._T_ww = float(W_sparse.power(2).sum() + W_sparse.multiply(W_sparse.T).sum())
    model.inference_data = idata
    return model


def _make_mock_sdem_model(
    y, X, WX, W_sparse, draws=200, beta_noise=0.05, lam_noise=0.05
):
    """Build a mock SDEM model: beta covers [X, WX] and lam is in posterior."""
    Z = np.hstack([X, WX])
    beta_mean = np.linalg.lstsq(Z, y, rcond=None)[0]
    rng = np.random.default_rng(42)
    beta_samples = np.tile(beta_mean, (draws, 1)) + rng.normal(
        scale=beta_noise, size=(draws, Z.shape[1])
    )
    lam_samples = rng.normal(scale=lam_noise, size=draws)

    Wy = np.asarray(W_sparse @ y, dtype=np.float64)
    idata = az.from_dict(
        {
            "posterior": {
                "beta": beta_samples[:, None, :],
                "lam": lam_samples[:, None],
                "sigma": np.ones(draws)[:, None],
            },
            "observed_data": {"y": y},
        }
    )

    model = MagicMock()
    model._y = y
    model._X = X
    model._WX = WX
    model._Wy = Wy
    model._W_sparse = W_sparse
    model._W_dense = W_sparse.toarray()
    model._T_ww = float(W_sparse.power(2).sum() + W_sparse.multiply(W_sparse.T).sum())
    model.inference_data = idata
    return model


class TestBayesianLMErrorSDMTest:
    def test_basic_output(self):
        y, X, WX, W_dense, W_sparse = _make_data_with_wx(n=20, k_wx=2)
        model = _make_mock_sdm_model(y, X, WX, W_sparse, draws=200)
        result = bayesian_lm_error_sdm_test(model)

        assert isinstance(result, BayesianLMTestResult)
        assert result.lm_samples.shape[0] == 200
        assert result.test_type == "bayesian_lm_error_sdm"
        assert 0.0 <= result.bayes_pvalue <= 1.0
        assert np.all(result.lm_samples >= 0)

    def test_sigma_scaling_matches_standard_lm_error(self):
        """At rho=0, k_wx=0 the SDM-error test should reduce to LM-Error.

        Regression test for the sigma^2 vs sigma^4 scaling bug: previously
        V was sigma^2 * T_ww (wrong); now it must be sigma^4 * T_ww.
        """
        y, X, W_dense, W_sparse = make_sar_sem_data(n=20)
        WX = np.empty((X.shape[0], 0), dtype=float)
        model = _make_mock_sdm_model(
            y, X, WX, W_sparse, draws=300, beta_noise=0.0, rho_noise=0.0
        )
        result_sdm = bayesian_lm_error_sdm_test(model)

        # Compare with standard LM-Error using the same OLS-style residuals
        ols_model = _make_mock_model(y, X, W_sparse, beta_noise=0.0, draws=300)
        result_ols = bayesian_lm_error_test(ols_model)

        # With rho=0 and no WX, the two tests should agree to numerical precision
        assert np.isclose(result_sdm.mean, result_ols.mean, rtol=1e-6)


class TestBayesianLMLagSDEMTest:
    def test_basic_output(self):
        y, X, WX, W_dense, W_sparse = _make_data_with_wx(n=20, k_wx=2)
        model = _make_mock_sdem_model(y, X, WX, W_sparse, draws=200)
        result = bayesian_lm_lag_sdem_test(model)

        assert isinstance(result, BayesianLMTestResult)
        assert result.lm_samples.shape[0] == 200
        assert result.test_type == "bayesian_lm_lag_sdem"
        assert 0.0 <= result.bayes_pvalue <= 1.0
        assert np.all(result.lm_samples >= 0)

    def test_variance_uses_sigma4_term(self):
        """Regression test: V must include sigma^4 * T_ww (not sigma^2 * T_ww).

        Construct a model with sigma=2 (so sigma^2=4, sigma^4=16). Verify
        that the LM denominator uses the whitened SDEM-filtered formula
        V = sigma^4 * T_ww + sigma^2 * z_rho' M_{Z_tilde} z_rho,
        where z_rho = A_lambda * W y and Z_tilde = A_lambda * [X, WX].
        """
        y, X, WX, W_dense, W_sparse = _make_data_with_wx(n=16, k_wx=1)
        model = _make_mock_sdem_model(
            y, X, WX, W_sparse, draws=200, beta_noise=0.0, lam_noise=0.0
        )
        # Override sigma to be a fixed value of 2.0
        sigma_val = 2.0
        model.inference_data.posterior["sigma"][:] = sigma_val
        T_ww = model._T_ww
        Wy = model._Wy

        result = bayesian_lm_lag_sdem_test(model)

        # Recompute the score using the same residual logic to back out V
        idata = model.inference_data
        beta_draws = np.asarray(idata.posterior["beta"]).reshape(
            -1, X.shape[1] + WX.shape[1]
        )
        lam_draws = np.asarray(idata.posterior["lam"]).reshape(-1)
        Z = np.hstack([X, WX])
        u = y[None, :] - beta_draws @ Z.T
        Wu = (W_sparse @ u.T).T
        eps = u - lam_draws[:, None] * Wu

        # Whitened quantities (matching the corrected implementation)
        lam_mean = float(np.mean(lam_draws))
        n = W_sparse.shape[0]
        A_lam = np.eye(n) - lam_mean * W_dense
        Z_tilde = A_lam @ Z
        z_rho = A_lam @ Wy
        S = np.dot(eps, z_rho)

        # Expected V using the corrected whitened formula
        from neighbayes.diagnostics.lmtests import _mx_quadratic

        expected_V = sigma_val**4 * T_ww + sigma_val**2 * _mx_quadratic(Z_tilde, z_rho)
        V_implied = float(np.mean(S**2 / (result.lm_samples + 1e-15)))

        # The implied V should match the expected sigma^4 formula, not the
        # old buggy sigma^2 formula
        old_buggy_V = sigma_val**2 * T_ww + float(np.dot(Wy, Wy))
        assert np.isclose(V_implied, expected_V, rtol=1e-3), (
            f"Implied V={V_implied:.3f} should match new sigma^4 formula "
            f"{expected_V:.3f} (old buggy was {old_buggy_V:.3f})"
        )


# ---------------------------------------------------------------------------
# Cross-cutting tests
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Tests that robust tests differ from non-robust tests (cross-term fix)
# ---------------------------------------------------------------------------


class TestRobustVsNonRobust:
    """Verify that robust tests produce different results from non-robust tests.

    After fixing J_{ργ·σ} to be non-zero (instead of zeros), the Neyman
    adjustment should have an effect, making robust tests differ from
    their non-robust counterparts.
    """

    def test_robust_wx_differs_from_nonrobust_wx(self):
        """Robust LM-WX (from SAR) should differ from LM-WX (from SAR)."""
        y, X, WX, W_dense, W_sparse = _make_data_with_wx(n=20, k_wx=2)
        model = _make_mock_sar_model(y, X, WX, W_sparse, draws=200)

        result_wx = bayesian_lm_wx_test(model)
        result_robust_wx = bayesian_robust_lm_wx_test(model)

        # The means should differ because the Neyman adjustment removes
        # the correlation with ρ, changing the test statistic
        assert result_wx.mean != result_robust_wx.mean, (
            f"Robust LM-WX mean ({result_robust_wx.mean:.4f}) should differ "
            f"from LM-WX mean ({result_wx.mean:.4f})"
        )

    def test_robust_lag_sdm_differs_from_lag(self):
        """Robust LM-Lag-SDM (from SLX) should differ from LM-Lag (from OLS).

        Note: These use different null models (SLX vs OLS), so they naturally
        differ. But the Neyman adjustment should also contribute.
        """
        y, X, WX, W_dense, W_sparse = _make_data_with_wx(n=20, k_wx=2)
        ols_model = _make_mock_model(y, X, W_sparse, draws=200)
        slx_model = _make_mock_slx_model(y, X, WX, W_sparse, draws=200)

        result_lag = bayesian_lm_lag_test(ols_model)
        result_robust_lag = bayesian_robust_lm_lag_sdm_test(slx_model)

        # These should differ (different null models + Neyman adjustment)
        assert result_lag.mean != result_robust_lag.mean


# ---------------------------------------------------------------------------
# Panel mock model builders
# ---------------------------------------------------------------------------


def _make_panel_data(N=6, T=3, k_wx=0, seed=42):
    """Generate panel data with N units and T time periods."""
    rng = np.random.default_rng(seed)
    n = N * T
    X = np.column_stack([np.ones(n), rng.normal(size=n)])
    beta_true = np.array([1.0, 2.0])
    y = X @ beta_true + rng.normal(scale=0.5, size=n)

    # Simple row-normalized ring weights for N units
    W_dense = np.eye(N, k=1) + np.eye(N, k=-1)
    W_dense[0, -1] = W_dense[-1, 0] = 1
    W_dense = W_dense / W_dense.sum(axis=1, keepdims=True)
    import scipy.sparse as sp

    W_sparse = sp.csr_matrix(W_dense)

    # WX columns (spatially lagged X, applied period-by-period)
    if k_wx > 0:
        WX_list = []
        for j in range(k_wx):
            wx_j = np.zeros(n)
            for t in range(T):
                start = t * N
                end = (t + 1) * N
                wx_j[start:end] = (
                    W_dense @ X[start:end, 1]
                )  # lag the non-constant column
            WX_list.append(wx_j)
        WX = np.column_stack(WX_list)
    else:
        WX = np.empty((n, 0), dtype=float)

    # Wy (panel spatial lag)
    Wy = np.zeros(n)
    for t in range(T):
        start = t * N
        end = (t + 1) * N
        Wy[start:end] = W_dense @ y[start:end]

    # W eigenvalues
    W_eigs = np.linalg.eigvals(W_dense)

    return y, X, WX, Wy, W_dense, W_sparse, W_eigs, N, T


def _make_mock_panel_ols_model(N=6, T=3, k_wx=0, beta_noise=0.1, draws=100, seed=42):
    """Build a mock OLS panel model for panel LM tests."""
    y, X, WX, Wy, W_dense, W_sparse, W_eigs, N, T = _make_panel_data(N, T, k_wx, seed)

    beta_ols = np.linalg.lstsq(X, y, rcond=None)[0]
    beta_samples = np.tile(beta_ols, (draws, 1))
    rng = np.random.default_rng(42)
    beta_samples += rng.normal(scale=beta_noise, size=beta_samples.shape)

    idata = az.from_dict(
        {
            "posterior": {
                "beta": beta_samples[:, None, :],
                "sigma": np.ones(draws)[:, None],
            },
            "observed_data": {"y": y},
        }
    )

    model = MagicMock()
    model._y = y
    model._X = X
    model._WX = WX
    model._Wy = Wy
    model._W_sparse = W_sparse
    model._W_eigs = W_eigs
    model._N = N
    model._T = T
    model._T_ww = float(W_sparse.power(2).sum() + W_sparse.multiply(W_sparse.T).sum())
    model.inference_data = idata
    return model


def _make_mock_panel_sar_model(
    N=6, T=3, k_wx=2, beta_noise=0.1, rho_noise=0.05, draws=100, seed=42
):
    """Build a mock SAR panel model for panel LM tests."""
    y, X, WX, Wy, W_dense, W_sparse, W_eigs, N, T = _make_panel_data(N, T, k_wx, seed)

    beta_ols = np.linalg.lstsq(X, y, rcond=None)[0]
    beta_samples = np.tile(beta_ols, (draws, 1))
    rng = np.random.default_rng(42)
    beta_samples += rng.normal(scale=beta_noise, size=beta_samples.shape)
    rho_samples = rng.normal(scale=rho_noise, size=draws)

    idata = az.from_dict(
        {
            "posterior": {
                "beta": beta_samples[:, None, :],
                "rho": rho_samples[:, None],
                "sigma": np.ones(draws)[:, None],
            },
            "observed_data": {"y": y},
        }
    )

    model = MagicMock()
    model._y = y
    model._X = X
    model._WX = WX
    model._Wy = Wy
    model._W_sparse = W_sparse
    model._W_eigs = W_eigs
    model._N = N
    model._T = T
    model._T_ww = float(W_sparse.power(2).sum() + W_sparse.multiply(W_sparse.T).sum())
    model.inference_data = idata
    return model


def _make_mock_panel_slx_model(N=6, T=3, k_wx=2, beta_noise=0.1, draws=100, seed=42):
    """Build a mock SLX panel model for panel LM tests."""
    y, X, WX, Wy, W_dense, W_sparse, W_eigs, N, T = _make_panel_data(N, T, k_wx, seed)

    Z = np.hstack([X, WX])
    beta_slx = np.linalg.lstsq(Z, y, rcond=None)[0]
    beta_samples = np.tile(beta_slx, (draws, 1))
    rng = np.random.default_rng(42)
    beta_samples += rng.normal(scale=beta_noise, size=beta_samples.shape)

    idata = az.from_dict(
        {
            "posterior": {
                "beta": beta_samples[:, None, :],
                "sigma": np.ones(draws)[:, None],
            },
            "observed_data": {"y": y},
        }
    )

    model = MagicMock()
    model._y = y
    model._X = X
    model._WX = WX
    model._Wy = Wy
    model._W_sparse = W_sparse
    model._W_eigs = W_eigs
    model._N = N
    model._T = T
    model._T_ww = float(W_sparse.power(2).sum() + W_sparse.multiply(W_sparse.T).sum())
    model.inference_data = idata
    return model


# ---------------------------------------------------------------------------
# Panel LM-Lag and LM-Error tests
# ---------------------------------------------------------------------------


class TestBayesianPanelLMLagTest:
    def test_basic_output(self):
        model = _make_mock_panel_ols_model(draws=200)
        result = bayesian_panel_lm_lag_test(model)

        assert isinstance(result, BayesianLMTestResult)
        assert result.lm_samples.shape[0] == 200
        assert result.df == 1
        assert result.test_type == "bayesian_panel_lm_lag"
        assert 0.0 <= result.bayes_pvalue <= 1.0
        assert np.all(result.lm_samples >= 0)
        assert result.details["N"] == 6
        assert result.details["T"] == 3

    def test_small_panel(self):
        model = _make_mock_panel_ols_model(N=4, T=2, draws=50)
        result = bayesian_panel_lm_lag_test(model)

        assert result.lm_samples.shape[0] == 50
        assert result.df == 1


class TestBayesianPanelLMErrorTest:
    def test_basic_output(self):
        model = _make_mock_panel_ols_model(draws=200)
        result = bayesian_panel_lm_error_test(model)

        assert isinstance(result, BayesianLMTestResult)
        assert result.lm_samples.shape[0] == 200
        assert result.df == 1
        assert result.test_type == "bayesian_panel_lm_error"
        assert 0.0 <= result.bayes_pvalue <= 1.0
        assert np.all(result.lm_samples >= 0)

    def test_small_panel(self):
        model = _make_mock_panel_ols_model(N=4, T=2, draws=50)
        result = bayesian_panel_lm_error_test(model)

        assert result.lm_samples.shape[0] == 50
        assert result.df == 1


# ---------------------------------------------------------------------------
# Panel robust LM-Lag and LM-Error tests
# ---------------------------------------------------------------------------


class TestBayesianPanelRobustLMLagTest:
    def test_basic_output(self):
        model = _make_mock_panel_ols_model(draws=200)
        result = bayesian_panel_robust_lm_lag_test(model)

        assert isinstance(result, BayesianLMTestResult)
        assert result.lm_samples.shape[0] == 200
        assert result.df == 1
        assert result.test_type == "bayesian_panel_robust_lm_lag"
        assert 0.0 <= result.bayes_pvalue <= 1.0
        assert np.all(result.lm_samples >= 0)


class TestBayesianPanelRobustLMErrorTest:
    def test_basic_output(self):
        model = _make_mock_panel_ols_model(draws=200)
        result = bayesian_panel_robust_lm_error_test(model)

        assert isinstance(result, BayesianLMTestResult)
        assert result.lm_samples.shape[0] == 200
        assert result.df == 1
        assert result.test_type == "bayesian_panel_robust_lm_error"
        assert 0.0 <= result.bayes_pvalue <= 1.0
        assert np.all(result.lm_samples >= 0)


# ---------------------------------------------------------------------------
# Panel WX / Joint / Robust SDM/SDEM tests
# ---------------------------------------------------------------------------


class TestBayesianPanelLMWxTest:
    def test_basic_output(self):
        model = _make_mock_panel_sar_model(k_wx=2, draws=200)
        result = bayesian_panel_lm_wx_test(model)

        assert isinstance(result, BayesianLMTestResult)
        assert result.lm_samples.shape[0] == 200
        assert result.df == 2  # k_wx = 2
        assert result.test_type == "bayesian_panel_lm_wx"
        assert 0.0 <= result.bayes_pvalue <= 1.0
        assert np.all(result.lm_samples >= 0)

    def test_no_wx_raises(self):
        model = _make_mock_panel_ols_model(k_wx=0, draws=50)
        with pytest.raises(ValueError, match="no WX columns"):
            bayesian_panel_lm_wx_test(model)


class TestBayesianPanelLMSDMJointTest:
    def test_basic_output(self):
        model = _make_mock_panel_ols_model(k_wx=2, draws=200)
        result = bayesian_panel_lm_sdm_joint_test(model)

        assert isinstance(result, BayesianLMTestResult)
        assert result.lm_samples.shape[0] == 200
        assert result.df == 3  # 1 + k_wx = 1 + 2
        assert result.test_type == "bayesian_panel_lm_sdm_joint"
        assert 0.0 <= result.bayes_pvalue <= 1.0
        assert np.all(result.lm_samples >= 0)

    def test_zero_wx_columns(self):
        model = _make_mock_panel_ols_model(k_wx=0, draws=50)
        result = bayesian_panel_lm_sdm_joint_test(model)

        assert result.df == 1  # 1 + 0


class TestBayesianPanelLMSLXErrorJointTest:
    def test_basic_output(self):
        model = _make_mock_panel_ols_model(k_wx=2, draws=200)
        result = bayesian_panel_lm_slx_error_joint_test(model)

        assert isinstance(result, BayesianLMTestResult)
        assert result.lm_samples.shape[0] == 200
        assert result.df == 3  # 1 + k_wx = 1 + 2
        assert result.test_type == "bayesian_panel_lm_slx_error_joint"
        assert 0.0 <= result.bayes_pvalue <= 1.0
        assert np.all(result.lm_samples >= 0)


class TestBayesianPanelRobustLMLagSDMTest:
    def test_basic_output(self):
        model = _make_mock_panel_slx_model(k_wx=2, draws=200)
        result = bayesian_panel_robust_lm_lag_sdm_test(model)

        assert isinstance(result, BayesianLMTestResult)
        assert result.lm_samples.shape[0] == 200
        assert result.df == 1
        assert result.test_type == "bayesian_panel_robust_lm_lag_sdm"
        assert 0.0 <= result.bayes_pvalue <= 1.0
        assert np.all(result.lm_samples >= 0)

    def test_no_wx_columns(self):
        model = _make_mock_panel_ols_model(k_wx=0, draws=50)
        result = bayesian_panel_robust_lm_lag_sdm_test(model)

        assert result.df == 1


class TestBayesianPanelRobustLMWXTest:
    def test_basic_output(self):
        model = _make_mock_panel_sar_model(k_wx=2, draws=200)
        result = bayesian_panel_robust_lm_wx_test(model)

        assert isinstance(result, BayesianLMTestResult)
        assert result.lm_samples.shape[0] == 200
        assert result.df == 2  # k_wx = 2
        assert result.test_type == "bayesian_panel_robust_lm_wx"
        assert 0.0 <= result.bayes_pvalue <= 1.0
        assert np.all(result.lm_samples >= 0)

    def test_no_wx_raises(self):
        model = _make_mock_panel_ols_model(k_wx=0, draws=50)
        with pytest.raises(ValueError, match="no WX columns"):
            bayesian_panel_robust_lm_wx_test(model)


class TestBayesianPanelRobustLMErrorSDEMTest:
    def test_basic_output(self):
        model = _make_mock_panel_slx_model(k_wx=2, draws=200)
        result = bayesian_panel_robust_lm_error_sdem_test(model)

        assert isinstance(result, BayesianLMTestResult)
        assert result.lm_samples.shape[0] == 200
        assert result.df == 1
        assert result.test_type == "bayesian_panel_robust_lm_error_sdem"
        assert 0.0 <= result.bayes_pvalue <= 1.0
        assert np.all(result.lm_samples >= 0)

    def test_no_wx_columns(self):
        model = _make_mock_panel_ols_model(k_wx=0, draws=50)
        result = bayesian_panel_robust_lm_error_sdem_test(model)

        assert result.df == 1


# ---------------------------------------------------------------------------
# Tests for spatial_diagnostics() method on model classes
# ---------------------------------------------------------------------------


class TestSpatialDiagnosticsMethod:
    """Test the spatial_diagnostics() and spatial_diagnostics_decision(format="model")
    methods on model classes."""

    def _make_model_with_class(
        self, y, X, W_sparse, cls, WX=None, beta_noise=0.1, draws=100, rho_noise=None
    ):
        """Build a mock model that has the right class for spatial_diagnostics."""
        from neighbayes.models import SDEM, SDM

        n, k = X.shape
        beta_ols = np.linalg.lstsq(X, y, rcond=None)[0]
        beta_samples = np.tile(beta_ols, (draws, 1))
        rng = np.random.default_rng(42)
        beta_samples += rng.normal(scale=beta_noise, size=beta_samples.shape)

        Wy = np.asarray(W_sparse @ y, dtype=np.float64)

        # Pad beta to match k + k_wx if WX has columns and class needs it
        k_wx = 0 if WX is None else WX.shape[1]
        if cls in (SDM, SDEM) and k_wx > 0:
            extra = rng.normal(scale=beta_noise, size=(draws, k_wx))
            beta_full = np.hstack([beta_samples, extra])
        else:
            beta_full = beta_samples

        posterior = {"beta": beta_full[:, None, :], "sigma": np.ones(draws)[:, None]}
        if rho_noise is not None:
            rho_samples = rng.normal(scale=rho_noise, size=draws)
            posterior["rho"] = rho_samples[:, None]
        if cls is SDM:
            posterior.setdefault("rho", rng.normal(scale=0.05, size=draws)[:, None])
        if cls is SDEM:
            posterior["lam"] = rng.normal(scale=0.05, size=draws)[:, None]
        # SEM mock needs ``lam`` for the new robust-after-naive tests
        # (``bayesian_robust_lm_lag_sem_test``, ``bayesian_robust_lm_wx_sem_test``).
        from neighbayes.models import SEM as _SEM

        if cls is _SEM:
            posterior["lam"] = rng.normal(scale=0.05, size=draws)[:, None]

        idata = az.from_dict({"posterior": posterior, "observed_data": {"y": y}})

        if WX is None:
            WX = np.empty((n, 0), dtype=float)

        # Create a real instance of the class but override __init__
        # We use a simple object trick: create a bare object and set
        # its class and attributes manually
        obj = object.__new__(cls)
        obj._y = y
        obj._X = X
        obj._WX = WX
        obj._Wy = Wy
        obj._W_sparse = W_sparse
        obj._feature_names = [f"x{i}" for i in range(k)]
        # Eager-fill the lazy cached properties used by ``base.Model._W_dense``
        # and ``base.Model._T_ww`` so the new robust-after-naive tests
        # (which require dense W) work on these mock objects.  We set
        # the attributes directly in __dict__ to shadow the
        # @cached_property descriptors.
        W_dense_arr = np.asarray(W_sparse.todense())
        obj._graph = None
        obj.__dict__["_W_dense"] = W_dense_arr
        obj.__dict__["_T_ww"] = float(
            np.sum(W_dense_arr * W_dense_arr) + np.trace(W_dense_arr @ W_dense_arr)
        )
        obj._idata = idata
        return obj

    def test_ols_spatial_diagnostics_returns_dataframe(self):
        """OLS.spatial_diagnostics() should return a DataFrame with 6 tests."""
        from neighbayes.models import OLS

        y, X, W_dense, W_sparse = make_sar_sem_data(n=16)
        model = self._make_model_with_class(y, X, W_sparse, OLS)

        df = model.spatial_diagnostics()

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 6
        assert set(df.index) == {
            "LM-Lag",
            "LM-Error",
            "LM-SDM-Joint",
            "LM-SLX-Error-Joint",
            "Robust-LM-Lag",
            "Robust-LM-Error",
        }
        assert "statistic" in df.columns
        assert "p_value" in df.columns
        assert "df" in df.columns
        assert "median" in df.columns
        assert "ci_lower" in df.columns
        assert "ci_upper" in df.columns
        # No 'significant' column
        assert "significant" not in df.columns
        assert df.attrs["model_type"] == "OLS"
        assert df.attrs["n_draws"] > 0

    def test_sar_spatial_diagnostics_returns_dataframe(self):
        """SAR.spatial_diagnostics() should return a DataFrame with 4 tests."""
        from neighbayes.models import SAR

        y, X, WX, W_dense, W_sparse = _make_data_with_wx(n=20, k_wx=2)
        model = self._make_model_with_class(y, X, W_sparse, SAR, WX=WX, rho_noise=0.05)

        df = model.spatial_diagnostics()

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 4
        assert set(df.index) == {"LM-Error", "LM-WX", "Robust-LM-WX", "Robust-LM-Error"}

    def test_slx_spatial_diagnostics_returns_dataframe(self):
        """SLX.spatial_diagnostics() should return a DataFrame with 4 tests."""
        from neighbayes.models import SLX

        y, X, WX, W_dense, W_sparse = _make_data_with_wx(n=20, k_wx=2)
        model = self._make_model_with_class(y, X, W_sparse, SLX, WX=WX)

        df = model.spatial_diagnostics()

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 4
        assert set(df.index) == {
            "LM-Lag",
            "LM-Error",
            "Robust-LM-Lag-SDM",
            "Robust-LM-Error-SDEM",
        }

    def test_sdm_spatial_diagnostics_returns_dataframe(self):
        """SDM.spatial_diagnostics() should return a DataFrame with 2 tests."""
        from neighbayes.models import SDM

        y, X, WX, W_dense, W_sparse = _make_data_with_wx(n=20, k_wx=2)
        model = self._make_model_with_class(y, X, W_sparse, SDM, WX=WX)

        df = model.spatial_diagnostics()

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2
        assert set(df.index) == {"LM-Error-SDM", "Robust-LM-Error-SDM"}

    def test_sdem_spatial_diagnostics_returns_dataframe(self):
        """SDEM.spatial_diagnostics() should return a DataFrame with 2 tests."""
        from neighbayes.models import SDEM

        y, X, WX, W_dense, W_sparse = _make_data_with_wx(n=20, k_wx=2)
        model = self._make_model_with_class(y, X, W_sparse, SDEM, WX=WX)

        df = model.spatial_diagnostics()

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2
        assert set(df.index) == {"LM-Lag-SDEM", "Robust-LM-Lag-SDEM"}

    def test_sem_spatial_diagnostics_returns_dataframe(self):
        """SEM.spatial_diagnostics() should return a DataFrame with 4 tests."""
        from neighbayes.models import SEM

        y, X, WX, W_dense, W_sparse = _make_data_with_wx(n=20, k_wx=2)
        model = self._make_model_with_class(y, X, W_sparse, SEM, WX=WX)

        df = model.spatial_diagnostics()

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 4
        assert set(df.index) == {"LM-Lag", "LM-WX", "Robust-LM-Lag", "Robust-LM-WX"}

    def test_spatial_diagnostics_decision_returns_string(self):
        """spatial_diagnostics_decision(format="model") should return a model name string."""
        from neighbayes.models import OLS

        y, X, W_dense, W_sparse = make_sar_sem_data(n=16)
        model = self._make_model_with_class(y, X, W_sparse, OLS)

        decision = model.spatial_diagnostics_decision(format="model")
        assert isinstance(decision, str)
        assert len(decision) > 0

    def test_spatial_diagnostics_decision_ols_no_significance(self):
        """When no tests are significant, OLS decision should return 'OLS'."""
        from neighbayes.models import OLS

        y, X, W_dense, W_sparse = make_sar_sem_data(n=16)
        model = self._make_model_with_class(y, X, W_sparse, OLS)

        # With very small n and random data, p-values are often > 0.05
        decision = model.spatial_diagnostics_decision(alpha=0.001, format="model")
        assert decision == "OLS"

    def test_spatial_diagnostics_requires_fit(self):
        """spatial_diagnostics() should raise if model not fit."""
        from neighbayes.models import OLS

        # Create a bare OLS object without fitting
        obj = object.__new__(OLS)
        obj._idata = None
        obj._W_sparse = None

        with pytest.raises(RuntimeError, match="not been fit"):
            obj.spatial_diagnostics()

    def test_spatial_diagnostics_pvalue_range(self):
        """All p-values should be in [0, 1]."""
        from neighbayes.models import OLS

        y, X, W_dense, W_sparse = make_sar_sem_data(n=16)
        model = self._make_model_with_class(y, X, W_sparse, OLS)

        df = model.spatial_diagnostics()
        assert (df["p_value"] >= 0).all()
        assert (df["p_value"] <= 1).all()

    def test_spatial_diagnostics_raw_results_in_attrs(self):
        """The raw BayesianLMTestResult objects should be in attrs."""
        from neighbayes.models import OLS

        y, X, W_dense, W_sparse = make_sar_sem_data(n=16)
        model = self._make_model_with_class(y, X, W_sparse, OLS)

        df = model.spatial_diagnostics()
        raw = df.attrs.get("_raw_results", {})
        assert len(raw) == 6
        for label, result in raw.items():
            assert isinstance(result, BayesianLMTestResult)

    # ------------------------------------------------------------------
    # Decision-tree branch coverage (regression tests for the
    # SDM/SDEM lookup fixes and the OLS robust-pair upgrade).
    # ------------------------------------------------------------------

    def test_sdm_decision_uses_lm_error_sdm(self, monkeypatch):
        """SDM decision must consult LM-Error-SDM (not the old LM-Error)."""
        from neighbayes.models import SDM

        y, X, WX, W_dense, W_sparse = _make_data_with_wx(n=20, k_wx=2)
        model = self._make_model_with_class(y, X, W_sparse, SDM, WX=WX)

        # Monkeypatch spatial_diagnostics to return a controlled DataFrame.
        # The new SDM tree requires both naive ``LM-Error-SDM`` and the
        # robust-after-naive ``Robust-LM-Error-SDM`` gate to fire before
        # escalating to MANSAR.
        df = pd.DataFrame(
            {"p_value": [0.001, 0.001]},
            index=["LM-Error-SDM", "Robust-LM-Error-SDM"],
        )
        monkeypatch.setattr(model, "spatial_diagnostics", lambda: df)
        assert (
            model.spatial_diagnostics_decision(alpha=0.05, format="model") == "MANSAR"
        )

        df2 = pd.DataFrame(
            {"p_value": [0.9, 0.9]},
            index=["LM-Error-SDM", "Robust-LM-Error-SDM"],
        )
        monkeypatch.setattr(model, "spatial_diagnostics", lambda: df2)
        assert model.spatial_diagnostics_decision(alpha=0.05, format="model") == "SDM"

    def test_sdem_decision_uses_lm_lag_sdem(self, monkeypatch):
        """SDEM decision must consult LM-Lag-SDEM (not the old LM-Lag)."""
        from neighbayes.models import SDEM

        y, X, WX, W_dense, W_sparse = _make_data_with_wx(n=20, k_wx=2)
        model = self._make_model_with_class(y, X, W_sparse, SDEM, WX=WX)

        df = pd.DataFrame(
            {"p_value": [0.001, 0.001]},
            index=["LM-Lag-SDEM", "Robust-LM-Lag-SDEM"],
        )
        monkeypatch.setattr(model, "spatial_diagnostics", lambda: df)
        assert (
            model.spatial_diagnostics_decision(alpha=0.05, format="model") == "MANSAR"
        )

        df2 = pd.DataFrame({"p_value": [0.9]}, index=["LM-Lag-SDEM"])
        monkeypatch.setattr(model, "spatial_diagnostics", lambda: df2)
        assert model.spatial_diagnostics_decision(alpha=0.05, format="model") == "SDEM"

    def test_ols_decision_uses_robust_pair_when_both_naive_significant(
        self, monkeypatch
    ):
        """OLS branch should disambiguate via Robust-LM-Lag/Robust-LM-Error."""
        from neighbayes.models import OLS

        y, X, W_dense, W_sparse = make_sar_sem_data(n=16)
        model = self._make_model_with_class(y, X, W_sparse, OLS)

        def make_df(rlag_p, rerr_p):
            return pd.DataFrame(
                {"p_value": [0.001, 0.001, rlag_p, rerr_p]},
                index=[
                    "LM-Lag",
                    "LM-Error",
                    "Robust-LM-Lag",
                    "Robust-LM-Error",
                ],
            )

        # Robust-Lag fires only -> SAR
        monkeypatch.setattr(model, "spatial_diagnostics", lambda: make_df(0.001, 0.9))
        assert model.spatial_diagnostics_decision(alpha=0.05, format="model") == "SAR"

        # Robust-Error fires only -> SEM
        monkeypatch.setattr(model, "spatial_diagnostics", lambda: make_df(0.9, 0.001))
        assert model.spatial_diagnostics_decision(alpha=0.05, format="model") == "SEM"

        # Both robust fire -> route via robust p-value tie-break.  Equal
        # robust p-values resolve in favour of SAR (lag <= error).  We
        # never escalate directly to SARAR from OLS.
        monkeypatch.setattr(model, "spatial_diagnostics", lambda: make_df(0.001, 0.001))
        assert model.spatial_diagnostics_decision(alpha=0.05, format="model") == "SAR"

        # Both robust fire, error robust strictly smaller -> SEM wins.
        monkeypatch.setattr(model, "spatial_diagnostics", lambda: make_df(0.01, 0.0001))
        assert model.spatial_diagnostics_decision(alpha=0.05, format="model") == "SEM"


# ---------------------------------------------------------------------------
# Bayesian LM tests for spatial flow models
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def _flow_ring_graph():
    """Ring graph on n=6 nodes, row-standardized."""
    from libpysal.graph import Graph

    n = 6
    focal = np.concatenate([np.arange(n), np.arange(n)])
    neighbor = np.concatenate([np.roll(np.arange(n), 1), np.roll(np.arange(n), -1)])
    G = Graph.from_arrays(focal, neighbor, np.ones(2 * n)).transform("r")
    return n, G


@pytest.fixture(scope="module")
def fitted_olsflow(_flow_ring_graph):
    from neighbayes.dgp.flows import generate_flow_data
    from neighbayes.models.flow._flow import OLSFlow

    n, G = _flow_ring_graph
    data = generate_flow_data(
        n=n,
        G=G,
        rho_d=0.0,
        rho_o=0.0,
        rho_w=0.0,
        beta_d=[1.0],
        beta_o=[0.5],
        sigma=0.3,
        seed=0,
    )
    ols = OLSFlow(data["y_vec"], data["X"], data["G"], col_names=data["col_names"])
    ols.fit(draws=150, tune=150, chains=2, progressbar=False, random_seed=0)
    return ols


@pytest.fixture(scope="module")
def fitted_sarflow(_flow_ring_graph):
    from neighbayes.dgp.flows import generate_flow_data
    from neighbayes.models.flow._flow import SARFlow

    n, G = _flow_ring_graph
    data = generate_flow_data(
        n=n,
        G=G,
        rho_d=0.3,
        rho_o=0.1,
        rho_w=0.0,
        beta_d=[1.0],
        beta_o=[0.5],
        sigma=0.3,
        seed=0,
    )
    sar = SARFlow(
        data["y_vec"],
        data["X"],
        data["G"],
        col_names=data["col_names"],
        logdet_method="traces",
        restrict_positive=False,
    )
    sar.fit(draws=150, tune=150, chains=2, progressbar=False, random_seed=0)
    return sar


class TestFlowLMTests:
    def test_marginal_dest(self, fitted_olsflow):
        from neighbayes.diagnostics.lmtests import bayesian_lm_flow_dest_test

        r = bayesian_lm_flow_dest_test(fitted_olsflow)
        assert isinstance(r, BayesianLMTestResult)
        assert r.df == 1
        assert r.test_type == "bayesian_lm_flow_dest"
        assert r.lm_samples.ndim == 1
        assert np.all(r.lm_samples >= 0)
        assert 0.0 <= r.bayes_pvalue <= 1.0

    def test_marginal_orig(self, fitted_olsflow):
        from neighbayes.diagnostics.lmtests import bayesian_lm_flow_orig_test

        r = bayesian_lm_flow_orig_test(fitted_olsflow)
        assert r.df == 1
        assert r.test_type == "bayesian_lm_flow_orig"
        assert 0.0 <= r.bayes_pvalue <= 1.0

    def test_marginal_network(self, fitted_olsflow):
        from neighbayes.diagnostics.lmtests import bayesian_lm_flow_network_test

        r = bayesian_lm_flow_network_test(fitted_olsflow)
        assert r.df == 1
        assert r.test_type == "bayesian_lm_flow_network"
        assert 0.0 <= r.bayes_pvalue <= 1.0

    def test_joint(self, fitted_olsflow):
        from neighbayes.diagnostics.lmtests import bayesian_lm_flow_joint_test

        r = bayesian_lm_flow_joint_test(fitted_olsflow)
        assert r.df == 3
        assert r.test_type == "bayesian_lm_flow_joint"
        assert np.all(r.lm_samples >= 0)
        assert 0.0 <= r.bayes_pvalue <= 1.0

    def test_intra(self, fitted_olsflow):
        from neighbayes.diagnostics.lmtests import bayesian_lm_flow_intra_test

        r = bayesian_lm_flow_intra_test(fitted_olsflow)
        assert r.df >= 1
        assert r.test_type == "bayesian_lm_flow_intra"
        assert 0.0 <= r.bayes_pvalue <= 1.0

    def test_robust_dest(self, fitted_sarflow):
        from neighbayes.diagnostics.lmtests import bayesian_robust_lm_flow_dest_test

        r = bayesian_robust_lm_flow_dest_test(fitted_sarflow)
        assert r.df == 1
        assert r.test_type == "bayesian_robust_lm_flow_dest"
        assert 0.0 <= r.bayes_pvalue <= 1.0

    def test_robust_orig(self, fitted_sarflow):
        from neighbayes.diagnostics.lmtests import bayesian_robust_lm_flow_orig_test

        r = bayesian_robust_lm_flow_orig_test(fitted_sarflow)
        assert r.df == 1
        assert r.test_type == "bayesian_robust_lm_flow_orig"
        assert 0.0 <= r.bayes_pvalue <= 1.0

    def test_robust_network(self, fitted_sarflow):
        from neighbayes.diagnostics.lmtests import bayesian_robust_lm_flow_network_test

        r = bayesian_robust_lm_flow_network_test(fitted_sarflow)
        assert r.df == 1
        assert r.test_type == "bayesian_robust_lm_flow_network"
        assert 0.0 <= r.bayes_pvalue <= 1.0


@pytest.fixture(scope="module")
def fitted_olsflow_panel(_flow_ring_graph):
    from neighbayes.dgp.flows import generate_flow_data
    from neighbayes.models.flow_panel._panel import OLSFlowPanel

    n, G = _flow_ring_graph
    T_periods = 3
    y_stack = []
    X_stack = []
    col_names = None
    for t in range(T_periods):
        d = generate_flow_data(
            n=n,
            G=G,
            rho_d=0.0,
            rho_o=0.0,
            rho_w=0.0,
            beta_d=[1.0],
            beta_o=[0.5],
            sigma=0.3,
            seed=10 + t,
        )
        y_stack.append(d["y_vec"])
        X_stack.append(d["X"])
        col_names = d["col_names"]

    y_panel = np.concatenate(y_stack)
    X_panel = np.vstack(X_stack)

    model = OLSFlowPanel(
        y_panel,
        X_panel,
        G,
        T=T_periods,
        col_names=col_names,
        effects=0,
    )
    model.fit(draws=150, tune=150, chains=2, progressbar=False, random_seed=0)
    return model


class TestFlowPanelLMTests:
    def test_marginal_dest(self, fitted_olsflow_panel):
        from neighbayes.diagnostics.lmtests import bayesian_panel_lm_flow_dest_test

        r = bayesian_panel_lm_flow_dest_test(fitted_olsflow_panel)
        assert r.df == 1
        assert r.test_type == "bayesian_panel_lm_flow_dest"
        assert np.all(r.lm_samples >= 0)
        assert 0.0 <= r.bayes_pvalue <= 1.0
        assert r.details["T"] == 3

    def test_marginal_orig_network(self, fitted_olsflow_panel):
        from neighbayes.diagnostics.lmtests import (
            bayesian_panel_lm_flow_network_test,
            bayesian_panel_lm_flow_orig_test,
        )

        for fn, tt in (
            (bayesian_panel_lm_flow_orig_test, "bayesian_panel_lm_flow_orig"),
            (bayesian_panel_lm_flow_network_test, "bayesian_panel_lm_flow_network"),
        ):
            r = fn(fitted_olsflow_panel)
            assert r.df == 1
            assert r.test_type == tt
            assert 0.0 <= r.bayes_pvalue <= 1.0

    def test_joint(self, fitted_olsflow_panel):
        from neighbayes.diagnostics.lmtests import bayesian_panel_lm_flow_joint_test

        r = bayesian_panel_lm_flow_joint_test(fitted_olsflow_panel)
        assert r.df == 3
        assert r.test_type == "bayesian_panel_lm_flow_joint"
        assert 0.0 <= r.bayes_pvalue <= 1.0

    def test_intra(self, fitted_olsflow_panel):
        from neighbayes.diagnostics.lmtests import bayesian_panel_lm_flow_intra_test

        r = bayesian_panel_lm_flow_intra_test(fitted_olsflow_panel)
        assert r.df >= 1
        assert r.test_type == "bayesian_panel_lm_flow_intra"
        assert 0.0 <= r.bayes_pvalue <= 1.0
