"""Mathematical correctness tests for Bayesian LM diagnostics.

Strategy: build a "degenerate posterior" with a single draw pinned to an
explicit point estimate (OLS / SAR / SLX / SEM coefficients with sigma = 1).
Under this construction the Bayesian-LM formula collapses to the classical
LM formula, so each statistic can be checked against an independent hand
computation to numerical precision (``atol=1e-10``).

This file complements :mod:`neighbayes.tests.test_bayesian_diagnostics`,
which verifies output shapes / df / p-value bounds. Here we verify the
underlying algebra.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import arviz as az
import numpy as np
import pytest
import scipy.sparse as sp

from neighbayes.diagnostics.lmtests import (
    _maybe_subtract_unit_effects,
    _neyman_adjust_scalar,
    _panel_spatial_lag,
    _resolve_X_for_beta,
    bayesian_lm_error_test,
    bayesian_lm_lag_test,
    bayesian_lm_sdm_joint_test,
    bayesian_lm_slx_error_joint_test,
    bayesian_panel_lm_error_sdm_test,
    bayesian_panel_lm_error_test,
    bayesian_panel_lm_lag_sdem_test,
    bayesian_panel_lm_lag_test,
    bayesian_panel_lm_wx_test,
    bayesian_robust_lm_error_sdem_test,
    bayesian_robust_lm_lag_sdm_test,
)
from neighbayes.tests.test_diagnostics._lm_mocks import gaussian_mock, panel_mock

# ---------------------------------------------------------------------------
# Tiny deterministic data + W
# ---------------------------------------------------------------------------


def _ring_W(n: int):
    """Row-normalized ring weights (1 above, 1 below, wrap-around)."""
    W = np.eye(n, k=1) + np.eye(n, k=-1)
    W[0, -1] = W[-1, 0] = 1.0
    W = W / W.sum(axis=1, keepdims=True)
    return W, sp.csr_matrix(W)


def _make_data(n: int = 16, k_wx: int = 2, seed: int = 0):
    rng = np.random.default_rng(seed)
    X = np.column_stack([np.ones(n), rng.normal(size=n)])
    WX = rng.normal(size=(n, k_wx))
    beta_true = np.array([1.0, -0.5])
    y = X @ beta_true + rng.normal(scale=0.5, size=n)
    W_dense, W_sp = _ring_W(n)
    T_ww = float(W_sp.power(2).sum() + W_sp.multiply(W_sp.T).sum())
    return y, X, WX, W_dense, W_sp, T_ww


# ---------------------------------------------------------------------------
# Degenerate-posterior model factories
# ---------------------------------------------------------------------------


def _idata(**posterior):
    """Wrap arrays of shape (draws,) or (draws, k) as ArviZ posterior with
    a single chain (chain=1, draw=N, ...). Avoids ArviZ's "more chains than
    draws" warning that the older (draws, 1) layout triggers."""
    out = {}
    for name, arr in posterior.items():
        arr = np.asarray(arr)
        if arr.ndim == 1:
            out[name] = arr[None, :]  # (chain=1, draw)
        else:
            out[name] = arr[None, :, :]  # (chain=1, draw, k)
    return az.from_dict({"posterior": out})


def _mock_ols(y, X, WX, W_sp, T_ww, beta_hat, sigma_hat=1.0, draws=1):
    Wy = np.asarray(W_sp @ y, dtype=np.float64)
    model = gaussian_mock(X, W_sp)
    model._y = y
    model._X = X
    model._WX = WX
    model._Wy = Wy
    model._W_sparse = W_sp
    model._T_ww = T_ww
    model.inference_data = _idata(
        beta=np.tile(beta_hat, (draws, 1)),
        sigma=np.full(draws, sigma_hat),
    )
    return model


def _mock_sar(y, X, WX, W_sp, T_ww, beta_hat, rho_hat, sigma_hat=1.0, draws=1):
    Wy = np.asarray(W_sp @ y, dtype=np.float64)
    model = gaussian_mock(X, W_sp)
    model._y = y
    model._X = X
    model._WX = WX
    model._Wy = Wy
    model._W_sparse = W_sp
    model._T_ww = T_ww
    model.inference_data = _idata(
        beta=np.tile(beta_hat, (draws, 1)),
        rho=np.full(draws, rho_hat),
        sigma=np.full(draws, sigma_hat),
    )
    return model


def _mock_sem(y, X, WX, W_sp, T_ww, beta_hat, lam_hat, sigma_hat=1.0, draws=1):
    Wy = np.asarray(W_sp @ y, dtype=np.float64)
    model = gaussian_mock(X, W_sp)
    model._y = y
    model._X = X
    model._WX = WX
    model._Wy = Wy
    model._W_sparse = W_sp
    model._T_ww = T_ww
    model.inference_data = _idata(
        beta=np.tile(beta_hat, (draws, 1)),
        lam=np.full(draws, lam_hat),
        sigma=np.full(draws, sigma_hat),
    )
    return model


def _mock_slx(y, X, WX, W_sp, T_ww, beta_hat_full, sigma_hat=1.0, draws=1):
    Wy = np.asarray(W_sp @ y, dtype=np.float64)
    model = gaussian_mock(X, W_sp)
    model._y = y
    model._X = X
    model._WX = WX
    model._Wy = Wy
    model._W_sparse = W_sp
    model._T_ww = T_ww
    model.inference_data = _idata(
        beta=np.tile(beta_hat_full, (draws, 1)),
        sigma=np.full(draws, sigma_hat),
    )
    return model


# ===========================================================================
# Phase 1 — Closed-form / degenerate-posterior cross-section tests
# ===========================================================================


class TestClosedFormLagError:
    """LM-lag and LM-error reduce to (S^2 / V) at the OLS estimate."""

    def test_lm_lag_matches_closed_form(self):
        # Anselin (1996) eq. 13: LM = (e'Wy)^2 / (sigma^4 * T_ww + sigma^2 * ||M_X W X beta||^2)
        y, X, _WX, _Wd, W_sp, T_ww = _make_data(n=20, k_wx=0)
        WX_empty = np.empty((y.size, 0))
        beta_hat = np.linalg.lstsq(X, y, rcond=None)[0]
        sigma_hat = 1.234  # non-unity to catch sigma-scaling bugs
        model = _mock_ols(y, X, WX_empty, W_sp, T_ww, beta_hat, sigma_hat=sigma_hat)

        e = y - X @ beta_hat
        Wy = np.asarray(W_sp @ y)
        WXb = np.asarray(W_sp @ (X @ beta_hat))
        # ||M_X v||^2 = v'v - v'X(X'X)^{-1}X'v
        XtWXb = X.T @ WXb
        proj = float(XtWXb @ np.linalg.solve(X.T @ X, XtWXb))
        mx_quad = float(WXb @ WXb) - proj
        S = float(e @ Wy)
        V = sigma_hat**4 * T_ww + sigma_hat**2 * mx_quad
        expected = S * S / V

        result = bayesian_lm_lag_test(model)
        assert result.lm_samples.shape == (1,)
        assert result.lm_samples[0] == pytest.approx(expected, rel=1e-10, abs=1e-12)

    def test_lm_error_matches_closed_form(self):
        # Anselin (1996) eq. 9: LM = (e'We)^2 / (sigma^4 * T_ww)
        y, X, _WX, _Wd, W_sp, T_ww = _make_data(n=20, k_wx=0)
        WX_empty = np.empty((y.size, 0))
        beta_hat = np.linalg.lstsq(X, y, rcond=None)[0]
        sigma_hat = 0.7  # non-unity to catch sigma-scaling bugs
        model = _mock_ols(y, X, WX_empty, W_sp, T_ww, beta_hat, sigma_hat=sigma_hat)

        e = y - X @ beta_hat
        We = np.asarray(W_sp @ e)
        S = float(e @ We)
        V = sigma_hat**4 * T_ww
        expected = S * S / V

        result = bayesian_lm_error_test(model)
        assert result.lm_samples[0] == pytest.approx(expected, rel=1e-10, abs=1e-12)


class TestClosedFormJoint:
    """Joint SDM and SDEM reduce to g' J^{-1} g at the OLS point estimate."""

    def test_lm_sdm_joint_matches_closed_form(self):
        # Koley-Bera (2024): joint SDM with M_X-projected info matrix.
        y, X, WX, _Wd, W_sp, T_ww = _make_data(n=24, k_wx=2)
        Wy = np.asarray(W_sp @ y)
        beta_hat = np.linalg.lstsq(X, y, rcond=None)[0]
        sigma_hat = 1.3  # non-unity to catch sigma-scaling bugs
        model = _mock_ols(y, X, WX, W_sp, T_ww, beta_hat, sigma_hat=sigma_hat)

        e = y - X @ beta_hat
        WXb = np.asarray(W_sp @ (X @ beta_hat))
        g_rho = float(e @ Wy)
        g_gamma = WX.T @ e
        g = np.concatenate([[g_rho], g_gamma])
        p = 1 + WX.shape[1]

        # M_X-projected raw-score variance
        XtX = X.T @ X
        XtWXb = X.T @ WXb
        XtWX = X.T @ WX
        mx_quad = float(WXb @ WXb) - float(XtWXb @ np.linalg.solve(XtX, XtWXb))
        mx_cross = (WXb @ WX) - XtWXb @ np.linalg.solve(XtX, XtWX)
        mx_gg = (WX.T @ WX) - XtWX.T @ np.linalg.solve(XtX, XtWX)

        V = np.zeros((p, p))
        V[0, 0] = sigma_hat**4 * T_ww + sigma_hat**2 * mx_quad
        V[0, 1:] = sigma_hat**2 * np.asarray(mx_cross).ravel()
        V[1:, 0] = sigma_hat**2 * np.asarray(mx_cross).ravel()
        V[1:, 1:] = sigma_hat**2 * mx_gg
        expected = float(g @ np.linalg.solve(V, g))

        result = bayesian_lm_sdm_joint_test(model)
        assert result.df == p
        assert result.lm_samples[0] == pytest.approx(expected, rel=1e-10, abs=1e-12)

    def test_lm_slx_error_joint_matches_closed_form(self):
        # Koley-Bera (2024) lm_slxerr: block-diagonal V => LM = LM_err + LM_wx
        y, X, WX, _Wd, W_sp, T_ww = _make_data(n=24, k_wx=2)
        beta_hat = np.linalg.lstsq(X, y, rcond=None)[0]
        sigma_hat = 0.85  # non-unity
        model = _mock_ols(y, X, WX, W_sp, T_ww, beta_hat, sigma_hat=sigma_hat)

        e = y - X @ beta_hat
        We = np.asarray(W_sp @ e)
        g_lam = float(e @ We)
        g_gamma = WX.T @ e

        XtWX = X.T @ WX
        mx_gg = (WX.T @ WX) - XtWX.T @ np.linalg.solve(X.T @ X, XtWX)

        p = 1 + WX.shape[1]
        V = np.zeros((p, p))
        V[0, 0] = sigma_hat**4 * T_ww
        V[1:, 1:] = sigma_hat**2 * mx_gg

        lm_lambda = g_lam * g_lam / V[0, 0]
        lm_gamma = float(g_gamma @ np.linalg.solve(V[1:, 1:], g_gamma))
        expected = lm_lambda + lm_gamma

        result = bayesian_lm_slx_error_joint_test(model)
        assert result.df == p
        assert result.lm_samples[0] == pytest.approx(expected, rel=1e-10, abs=1e-12)


class TestClosedFormRobust:
    """At the SLX null the robust pair is Anselin's, with ``M_Z`` for ``M_X``.

    At a degenerate posterior on the SLX estimate the scores of the fitted
    ``(beta, theta)`` vanish, and the lag and error scores' information is
    ``J_rr = T_WW + ||M_Z W Z delta||^2 / sigma^2``, ``J_ll = J_rl = T_WW``.
    """

    def _slx(self, sigma_hat):
        y, X, WX, _Wd, W_sp, T_ww = _make_data(n=24, k_wx=2)
        Z = np.hstack([X, WX])
        delta = np.linalg.lstsq(Z, y, rcond=None)[0]
        model = _mock_slx(y, X, WX, W_sp, T_ww, delta, sigma_hat=sigma_hat)
        e = y - Z @ delta
        s2 = sigma_hat**2
        g_r = float(e @ (W_sp @ y)) / s2
        g_l = float(e @ (W_sp @ e)) / s2
        WZd = np.asarray(W_sp @ (Z @ delta))
        mz = float(WZd @ WZd) - float((Z.T @ WZd) @ np.linalg.solve(Z.T @ Z, Z.T @ WZd))
        return model, g_r, g_l, T_ww + mz / s2, T_ww

    def test_robust_lm_lag_sdm_matches_closed_form(self):
        model, g_r, g_l, J_rr, T = self._slx(sigma_hat=1.4)
        expected = (g_r - g_l) ** 2 / (J_rr - T)
        result = bayesian_robust_lm_lag_sdm_test(model)
        assert result.lm_samples[0] == pytest.approx(expected, rel=1e-10)

    def test_robust_lm_error_sdem_matches_closed_form(self):
        model, g_r, g_l, J_rr, T = self._slx(sigma_hat=0.6)
        expected = (g_l - T / J_rr * g_r) ** 2 / (T * (1.0 - T / J_rr))
        result = bayesian_robust_lm_error_sdem_test(model)
        assert result.lm_samples[0] == pytest.approx(expected, rel=1e-10)


class TestNeymanAdjustScalar:
    def test_known_inputs_match_hand_computation(self):
        g_t = np.array([1.0, 2.0, 3.0])
        g_n = np.array([[0.5, 0.0], [1.0, 0.0], [-0.5, 0.0]])
        J_tt = 4.0
        J_tn = np.array([1.0, 0.0])
        J_nn = np.diag([2.0, 1.0])

        g_star, V_star = _neyman_adjust_scalar(g_t, g_n, J_tt, J_tn, J_nn, label="test")
        # coef = J_nn^-1 @ J_tn = [0.5, 0]; adjustment = g_n @ coef = 0.5 * g_n[:,0]
        expected_g = g_t - 0.5 * g_n[:, 0]
        expected_V = J_tt - 0.5  # J_tn @ coef = 0.5
        np.testing.assert_allclose(g_star, expected_g)
        assert V_star == pytest.approx(expected_V)

    def test_empty_nuisance_returns_input(self):
        g_t = np.array([1.0, 2.0])
        g_n = np.empty((2, 0))
        g_star, V_star = _neyman_adjust_scalar(
            g_t, g_n, 5.0, np.empty(0), np.empty((0, 0)), label="empty"
        )
        np.testing.assert_array_equal(g_star, g_t)
        assert V_star == 5.0


class TestResolveXForBeta:
    def test_returns_X_for_k_dim_beta(self):
        n, k, k_wx = 8, 2, 3
        model = MagicMock()
        model._X = np.zeros((n, k))
        model._WX = np.zeros((n, k_wx))
        beta = np.zeros((5, k))  # 5 draws, k coefficients
        out = _resolve_X_for_beta(model, beta)
        assert out.shape == (n, k)
        assert out is model._X

    def test_returns_hstack_for_2k_dim_beta(self):
        n, k, k_wx = 8, 2, 3
        model = MagicMock()
        model._X = np.arange(n * k, dtype=float).reshape(n, k)
        model._WX = np.arange(n * k_wx, dtype=float).reshape(n, k_wx) + 100
        beta = np.zeros((5, k + k_wx))
        out = _resolve_X_for_beta(model, beta)
        assert out.shape == (n, k + k_wx)
        np.testing.assert_array_equal(out[:, :k], model._X)
        np.testing.assert_array_equal(out[:, k:], model._WX)

    def test_returns_X_when_WX_is_empty(self):
        n, k = 8, 2
        model = MagicMock()
        model._X = np.zeros((n, k))
        model._WX = np.empty((n, 0))
        beta = np.zeros((5, k))
        out = _resolve_X_for_beta(model, beta)
        assert out.shape == (n, k)


class TestMaybeSubtractUnitEffects:
    def test_no_subtraction_without_unit_idx(self):
        model = MagicMock(spec=[])  # no _unit_idx attribute
        idata = _idata(beta=np.zeros((3, 2)))
        resid = np.ones((3, 4))
        out = _maybe_subtract_unit_effects(model, idata, resid)
        np.testing.assert_array_equal(out, resid)

    def test_no_subtraction_without_unit_effects_in_posterior(self):
        model = MagicMock()
        model._unit_idx = np.array([0, 1, 0, 1])
        idata = _idata(beta=np.zeros((3, 2)))  # no unit effects
        resid = np.ones((3, 4))
        out = _maybe_subtract_unit_effects(model, idata, resid)
        np.testing.assert_array_equal(out, resid)

    def test_subtracts_unit_effects_when_present(self):
        model = MagicMock()
        unit_idx = np.array([0, 1, 0, 1])
        model._unit_idx = unit_idx
        alpha = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])  # (3 draws, 2 units)
        idata = _idata(unit_effect=alpha, beta=np.zeros((3, 2)))
        resid = np.zeros((3, 4))
        out = _maybe_subtract_unit_effects(model, idata, resid)
        # alpha[:, unit_idx] gives (3, 4); resid - that == -alpha[:, unit_idx]
        np.testing.assert_allclose(out, -alpha[:, unit_idx])


# ===========================================================================
# Phase 5 — Panel closed-form tests (T-multiplier verification)
# ===========================================================================


def _make_panel_data(N: int = 5, T: int = 4, k_wx: int = 2, seed: int = 0):
    """Stacked-by-time panel data: y, X, WX of length N*T (period-major)."""
    rng = np.random.default_rng(seed)
    n = N * T
    X = np.column_stack([np.ones(n), rng.normal(size=n)])
    WX = rng.normal(size=(n, k_wx))
    beta_true = np.array([0.5, 1.0])
    y = X @ beta_true + rng.normal(scale=0.4, size=n)
    Wn, Wn_sp = _ring_W(N)
    T_ww = float(Wn_sp.power(2).sum() + Wn_sp.multiply(Wn_sp.T).sum())
    return y, X, WX, Wn_sp, T_ww


def _mock_panel_ols(y, X, WX, Wn_sp, T_ww, beta_hat, sigma_hat, N, T, draws=1):
    """Panel OLS mock with `_Wy = (W ⊗ I_T) y` (period-major stacking)."""
    Wy = _panel_spatial_lag(Wn_sp, np.asarray(y), N, T)
    model = panel_mock(X, Wn_sp, N, T)
    model._y = y
    model._X = X
    model._WX = WX
    model._Wy = np.asarray(Wy)
    model._W_sparse = Wn_sp
    model._T_ww = T_ww
    model._N = N
    model._T = T
    model._nonintercept_indices = list(range(X.shape[1]))
    model.inference_data = _idata(
        beta=np.tile(beta_hat, (draws, 1)),
        sigma=np.full(draws, sigma_hat),
    )
    return model


def _mock_panel_sar(y, X, WX, Wn_sp, T_ww, beta_hat, rho_hat, sigma_hat, N, T, draws=1):
    Wy = _panel_spatial_lag(Wn_sp, np.asarray(y), N, T)
    model = panel_mock(X, Wn_sp, N, T)
    model._y = y
    model._X = X
    model._WX = WX
    model._Wy = np.asarray(Wy)
    model._W_sparse = Wn_sp
    model._T_ww = T_ww
    model._N = N
    model._T = T
    model._nonintercept_indices = list(range(X.shape[1]))
    model.inference_data = _idata(
        beta=np.tile(beta_hat, (draws, 1)),
        rho=np.full(draws, rho_hat),
        sigma=np.full(draws, sigma_hat),
    )
    return model


def _mock_panel_sdm(y, X, WX, Wn_sp, T_ww, beta_hat, rho_hat, sigma_hat, N, T, draws=1):
    """Mock SDM panel model: beta covers [X, WX] and rho is in posterior."""
    Wy = _panel_spatial_lag(Wn_sp, np.asarray(y), N, T)
    model = panel_mock(X, Wn_sp, N, T)
    model._y = y
    model._X = X
    model._WX = WX
    model._Wy = np.asarray(Wy)
    model._W_sparse = Wn_sp
    model._W_dense = np.kron(np.eye(T), Wn_sp.toarray())
    model._T_ww = T_ww
    model._N = N
    model._T = T
    model._nonintercept_indices = list(range(X.shape[1]))
    model.inference_data = _idata(
        beta=np.tile(beta_hat, (draws, 1)),
        rho=np.full(draws, rho_hat),
        sigma=np.full(draws, sigma_hat),
    )
    return model


def _mock_panel_sdem(
    y, X, WX, Wn_sp, T_ww, beta_hat, lam_hat, sigma_hat, N, T, draws=1
):
    """Mock SDEM panel model: beta covers [X, WX] and lam is in posterior."""
    Wy = _panel_spatial_lag(Wn_sp, np.asarray(y), N, T)
    model = panel_mock(X, Wn_sp, N, T)
    model._y = y
    model._X = X
    model._WX = WX
    model._Wy = np.asarray(Wy)
    model._W_sparse = Wn_sp
    model._W_dense = np.kron(np.eye(T), Wn_sp.toarray())
    model._T_ww = T_ww
    model._N = N
    model._T = T
    model._nonintercept_indices = list(range(X.shape[1]))
    model.inference_data = _idata(
        beta=np.tile(beta_hat, (draws, 1)),
        lam=np.full(draws, lam_hat),
        sigma=np.full(draws, sigma_hat),
    )
    return model


class TestPanelClosedForm:
    def test_panel_lm_lag_matches_closed_form(self):
        N, T_per = 5, 4
        y, X, WX, Wn_sp, T_ww = _make_panel_data(N=N, T=T_per, k_wx=0)
        WX_empty = np.empty((y.size, 0))
        beta_hat = np.linalg.lstsq(X, y, rcond=None)[0]
        sigma_hat = 0.8
        model = _mock_panel_ols(
            y, X, WX_empty, Wn_sp, T_ww, beta_hat, sigma_hat, N, T_per
        )

        # Hand replicate the implementation:
        e = y - X @ beta_hat
        Wy = _panel_spatial_lag(Wn_sp, np.asarray(y), N, T_per)
        S = float(e @ Wy)
        y_hat = X @ beta_hat
        Wy_hat = _panel_spatial_lag(Wn_sp, y_hat, N, T_per)
        XtX_inv = np.linalg.inv(X.T @ X)
        M_Wy = Wy_hat - X @ (XtX_inv @ (X.T @ Wy_hat))
        WbMWb = float(Wy_hat @ M_Wy)
        # Note the **T** multiplier on T_ww — this is the panel correction
        J_val = WbMWb + T_per * T_ww * sigma_hat**2
        expected = S * S / (sigma_hat**2 * J_val)

        result = bayesian_panel_lm_lag_test(model)
        assert result.df == 1
        assert result.lm_samples[0] == pytest.approx(expected, rel=1e-10, abs=1e-12)

    def test_panel_lm_error_matches_closed_form(self):
        N, T_per = 5, 4
        y, X, WX, Wn_sp, T_ww = _make_panel_data(N=N, T=T_per, k_wx=0)
        WX_empty = np.empty((y.size, 0))
        beta_hat = np.linalg.lstsq(X, y, rcond=None)[0]
        sigma_hat = 0.8
        model = _mock_panel_ols(
            y, X, WX_empty, Wn_sp, T_ww, beta_hat, sigma_hat, N, T_per
        )

        e = y - X @ beta_hat
        We = _panel_spatial_lag(Wn_sp, e, N, T_per)
        S = float(e @ We)
        # V = sigma^4 * T * tr(W'W + W²) — note the T multiplier
        V = sigma_hat**4 * T_per * T_ww
        expected = S * S / V

        result = bayesian_panel_lm_error_test(model)
        assert result.lm_samples[0] == pytest.approx(expected, rel=1e-10, abs=1e-12)

    def test_panel_robust_lm_lag_m_x_projection(self):
        """The statistic ignores the posterior spread of beta (lag variant).

        It is the test at the posterior-mean parameters, so two posteriors with
        the same mean give the same statistic however wide they are.
        """
        N, T_per = 5, 4
        y, X, WX, Wn_sp, T_ww = _make_panel_data(N=N, T=T_per, k_wx=0)
        beta_hat = np.linalg.lstsq(X, y, rcond=None)[0]
        sigma_hat = 0.8

        # Low beta variance — all draws identical
        model_low = _mock_panel_ols(
            y,
            X,
            np.empty((y.size, 0)),
            Wn_sp,
            T_ww,
            beta_hat,
            sigma_hat,
            N,
            T_per,
            draws=50,
        )

        # High beta variance — noisy draws, but same posterior mean
        rng = np.random.default_rng(0)
        beta_noisy = beta_hat + rng.normal(scale=5.0, size=(50, len(beta_hat)))
        # Shift so mean equals beta_hat (preserves J_val)
        beta_noisy = beta_noisy - beta_noisy.mean(axis=0) + beta_hat
        model_high = _mock_panel_ols(
            y,
            X,
            np.empty((y.size, 0)),
            Wn_sp,
            T_ww,
            beta_hat,
            sigma_hat,
            N,
            T_per,
            draws=50,
        )
        model_high.inference_data = _idata(
            beta=beta_noisy,
            sigma=np.full(50, sigma_hat),
        )

        from neighbayes.diagnostics.lmtests import (
            bayesian_panel_robust_lm_lag_test,
        )

        result_low = bayesian_panel_robust_lm_lag_test(model_low)
        result_high = bayesian_panel_robust_lm_lag_test(model_high)
        assert result_high.statistic == pytest.approx(result_low.statistic, rel=1e-10)

    def test_panel_robust_lm_error_m_x_projection(self):
        """The statistic ignores the posterior spread of beta (error variant).

        It is the test at the posterior-mean parameters, so two posteriors with
        the same mean give the same statistic however wide they are.
        """
        N, T_per = 5, 4
        y, X, WX, Wn_sp, T_ww = _make_panel_data(N=N, T=T_per, k_wx=0)
        beta_hat = np.linalg.lstsq(X, y, rcond=None)[0]
        sigma_hat = 0.8

        model_low = _mock_panel_ols(
            y,
            X,
            np.empty((y.size, 0)),
            Wn_sp,
            T_ww,
            beta_hat,
            sigma_hat,
            N,
            T_per,
            draws=50,
        )
        rng = np.random.default_rng(0)
        beta_noisy = beta_hat + rng.normal(scale=5.0, size=(50, len(beta_hat)))
        beta_noisy = beta_noisy - beta_noisy.mean(axis=0) + beta_hat
        model_high = _mock_panel_ols(
            y,
            X,
            np.empty((y.size, 0)),
            Wn_sp,
            T_ww,
            beta_hat,
            sigma_hat,
            N,
            T_per,
            draws=50,
        )
        model_high.inference_data = _idata(
            beta=beta_noisy,
            sigma=np.full(50, sigma_hat),
        )

        from neighbayes.diagnostics.lmtests import (
            bayesian_panel_robust_lm_error_test,
        )

        result_low = bayesian_panel_robust_lm_error_test(model_low)
        result_high = bayesian_panel_robust_lm_error_test(model_high)
        assert result_high.statistic == pytest.approx(result_low.statistic, rel=1e-10)
