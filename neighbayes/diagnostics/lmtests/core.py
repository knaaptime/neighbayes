"""Core helpers for Bayesian LM diagnostic tests.

This module contains the low-level building blocks shared across all
LM test families (cross-sectional, panel, flow):

- Posterior draw extraction
- Residual computation
- Scalar / vector LM finalization
- Matrix algebra utilities (_mx_quadratic, _mx_cross)
"""

import warnings
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
from scipy import stats as sp_stats

from ..._lazy_deps import xr

# ---------------------------------------------------------------------------
# Module constants
# ---------------------------------------------------------------------------

#: Ridge added to the diagonal before inversion in :func:`_safe_inv` / :func:`_safe_solve`.
_INV_RIDGE = 1e-12

#: Condition-number threshold above which the regularized inverse falls back to ``pinv``.
_INV_COND_THRESHOLD = 1e12

# ---------------------------------------------------------------------------
# BayesianLMTestResult dataclass
# ---------------------------------------------------------------------------


@dataclass
class BayesianLMTestResult:
    """Container for Bayesian LM test results.

    Attributes
    ----------
    lm_samples : np.ndarray
        Posterior samples of the LM statistic.
    mean : float
        Posterior mean of the LM statistic.
    median : float
        Posterior median of the LM statistic.
    credible_interval : Tuple[float, float]
        95% credible interval for the LM statistic.
    bayes_pvalue : float
        P-value computed at the posterior mean of the LM statistic,
        ``1 - chi2.cdf(mean, df)``. Under H₀ the LM statistic follows
        a :math:`\\chi^2` distribution (:cite:p:`dogan2021BayesianRobust`, Proposition 1).
    test_type : str
        Label identifying the test (e.g. ``"bayesian_lm_lag"``).
    df : int
        Degrees of freedom for the reference :math:`\\chi^2` distribution.
        Default is 1 for single-parameter tests.
    details : Dict[str, Any]
        Additional metadata (e.g. number of posterior draws).
    """

    lm_samples: np.ndarray
    mean: float
    median: float
    credible_interval: Tuple[float, float]
    bayes_pvalue: float
    test_type: str
    df: int = 1
    details: Dict[str, Any] = None

    def __post_init__(self):
        if self.details is None:
            self.details = {}

    def to_dict(self):
        """Convert the test result to a dictionary."""
        d = {
            "lm_samples": self.lm_samples,
            "mean": self.mean,
            "median": self.median,
            "credible_interval": self.credible_interval,
            "bayes_pvalue": self.bayes_pvalue,
            "test_type": self.test_type,
            "df": self.df,
        }
        if self.details:
            for k in self.details:
                d[k] = self.details[k]
        return d

    def to_series(self):
        """Convert the test result to a pandas Series."""
        import pandas as pd

        return pd.Series(self.to_dict())


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------


def _get_posterior_draws(idata: xr.DataTree, param: str) -> np.ndarray:
    """Extract posterior draws for a parameter from an ArviZ DataTree object.

    Parameters
    ----------
    idata : xarray.DataTree
        DataTree object containing posterior samples.
    param : str
        Name of the parameter to extract.

    Returns
    -------
    np.ndarray
        Posterior draws, reshaped to (draws, ...) where draws is the total
        number of samples.
    """
    arr = idata.posterior[param]
    arr = arr.values if hasattr(arr, "values") else arr
    return arr.reshape(-1, *arr.shape[2:])


def _finalize_lm(
    LM: np.ndarray,
    *,
    test_type: str,
    df: int,
    details: Optional[Dict[str, Any]] = None,
) -> BayesianLMTestResult:
    """Build a :class:`BayesianLMTestResult` from per-draw LM samples.

    Centralises the boilerplate (mean / median / 95% CI / Bayesian p-value
    against ``chi2(df)``) shared by every Bayesian LM test in this module.
    The ``n_draws`` entry is added to ``details`` automatically.
    """
    LM = np.asarray(LM)
    mean = float(np.mean(LM))
    median = float(np.median(LM))
    ci = (float(np.percentile(LM, 2.5)), float(np.percentile(LM, 97.5)))
    bayes_pvalue = float(1 - sp_stats.chi2.cdf(mean, df))
    merged: Dict[str, Any] = {"n_draws": int(LM.shape[0])}
    if details:
        merged.update(details)
    return BayesianLMTestResult(
        lm_samples=LM,
        mean=mean,
        median=median,
        credible_interval=ci,
        bayes_pvalue=bayes_pvalue,
        test_type=test_type,
        df=df,
        details=merged,
    )


def _safe_inv(M: np.ndarray, label: str = "information matrix") -> np.ndarray:
    """Robust matrix inverse for LM-test information / cross-product matrices.

    Adds a tiny ridge (``_INV_RIDGE * I``) to prevent exact singularity, checks
    the condition number of the regularized matrix, and emits a
    ``RuntimeWarning`` plus falls back to ``np.linalg.pinv`` when
    ``cond > _INV_COND_THRESHOLD``.

    Parameters
    ----------
    M : np.ndarray
        Square matrix to invert (typically a Fisher information block or
        :math:`X^\\top X`).
    label : str
        Human-readable label used in the warning message.

    Returns
    -------
    np.ndarray
        Either ``inv(M + ε I)`` (well-conditioned) or ``pinv(M)``
        (ill-conditioned).
    """
    M = np.asarray(M, dtype=np.float64)
    n = M.shape[0]
    M_reg = M + _INV_RIDGE * np.eye(n)
    cond = np.linalg.cond(M_reg)
    if not np.isfinite(cond) or cond > _INV_COND_THRESHOLD:
        warnings.warn(
            f"{label} is ill-conditioned (cond={cond:.2e}); "
            f"ridge {_INV_RIDGE:.0e} insufficient, falling back to pseudo-inverse.",
            RuntimeWarning,
            stacklevel=2,
        )
        return np.linalg.pinv(M)
    return np.linalg.inv(M_reg)


def _safe_solve(
    M: np.ndarray, b: np.ndarray, label: str = "information matrix"
) -> np.ndarray:
    """Solve ``M x = b`` with ridge regularization and ``pinv`` fallback.

    Companion to :func:`_safe_inv` for code paths that only need ``M⁻¹ @ b``
    (avoiding the explicit inverse).  Same ridge magnitude and condition-number
    threshold so results are consistent with :func:`_safe_inv`.

    Parameters
    ----------
    M : np.ndarray
        Square matrix (typically a Fisher information block or ``XᵀX``).
    b : np.ndarray
        Right-hand side (vector or matrix).
    label : str
        Human-readable label used in the warning message.

    Returns
    -------
    np.ndarray
        ``solve(M + ε I, b)`` (well-conditioned) or ``pinv(M) @ b``
        (ill-conditioned).
    """
    M = np.asarray(M, dtype=np.float64)
    n = M.shape[0]
    M_reg = M + _INV_RIDGE * np.eye(n)
    cond = np.linalg.cond(M_reg)
    if not np.isfinite(cond) or cond > _INV_COND_THRESHOLD:
        warnings.warn(
            f"{label} is ill-conditioned (cond={cond:.2e}); "
            f"ridge {_INV_RIDGE:.0e} insufficient, falling back to pseudo-inverse.",
            RuntimeWarning,
            stacklevel=2,
        )
        return np.linalg.pinv(M) @ b
    return np.linalg.solve(M_reg, b)


def _mx_quadratic(X: np.ndarray, v: np.ndarray) -> float:
    r"""Compute :math:`v^\top M_X v = v^\top v - v^\top X (X^\top X)^{-1} X^\top v`.

    Here :math:`M_X = I - X(X^\top X)^{-1} X^\top` is the OLS annihilator
    matrix.  Used in the concentrated information-matrix blocks of the
    Bayesian LM-Lag / LM-WX / SDM-joint tests
    (:cite:p:`anselin1996SimpleDiagnostic`, eq. 13;
    :math:`\beta` has been concentrated out via Schur complement.
    :cite:p:`koley2024UseNot`, Section 3) where the OLS coefficient
    """
    Xv = X.T @ v
    XtX = X.T @ X
    sol, *_ = np.linalg.lstsq(XtX, Xv, rcond=None)
    proj = float(Xv @ sol)
    return float(v @ v) - proj


def _mx_cross(X: np.ndarray, U: np.ndarray, V: np.ndarray) -> np.ndarray:
    r"""Compute :math:`U^\top M_X V = U^\top V - U^\top X (X^\top X)^{-1} X^\top V`.

    Generalises :func:`_mx_quadratic` to arbitrary left/right factors.
    Returns shape inferred from ``U`` and ``V`` (matches ``U.T @ V``):
    scalar / vector / matrix as appropriate.
    """
    XtX = X.T @ X
    XU = X.T @ U  # (k,) or (k, m1)
    XV = X.T @ V  # (k,) or (k, m2)
    UtV = U.T @ V
    sol, *_ = np.linalg.lstsq(XtX, XV, rcond=None)
    return UtV - XU.T @ sol


def _resolve_X_for_beta(model, beta_draws: np.ndarray) -> np.ndarray:
    """Return the design matrix matching the posterior beta dimensions.

    When Gibbs sampling drops the intercept column for FE models,
    ``beta_draws`` has fewer columns than ``model._X``.  This function
    auto-detects that case and returns the appropriate sub-matrix.

    For models with WX coefficients (SDM, SDEM, SLX), the design
    matrix is ``[X, WX]`` (or ``[X[:, ni], WX]`` when the intercept
    is dropped).
    """
    k_beta = beta_draws.shape[1]
    ni = model._nonintercept_indices
    intercept_dropped = len(ni) < model._X.shape[1] and k_beta < model._X.shape[1]

    if intercept_dropped:
        X_base = model._X[:, ni]
    else:
        X_base = model._X

    # Check whether beta covers both X and WX blocks
    if (
        hasattr(model, "_WX")
        and model._WX.shape[1] > 0
        and k_beta == X_base.shape[1] + model._WX.shape[1]
    ):
        return np.hstack([X_base, model._WX])

    # Beta covers only the X block (or X_base already matches)
    if k_beta == X_base.shape[1]:
        return X_base

    # Fallback: try full X with WX
    if (
        hasattr(model, "_WX")
        and model._WX.shape[1] > 0
        and k_beta == model._X.shape[1] + model._WX.shape[1]
    ):
        return np.hstack([model._X, model._WX])

    return model._X


def _maybe_subtract_alpha(model, idata, resid: np.ndarray) -> np.ndarray:
    """Subtract per-unit random-effect ``alpha[unit_idx]`` from ``resid`` if present.

    Used by panel LM tests so that the cross-sectional residual formulas
    transparently work for both pooled and random-effects fits.
    """
    if hasattr(model, "_unit_idx") and "alpha" in idata.posterior:
        alpha_draws = _get_posterior_draws(idata, "alpha")
        resid = resid - alpha_draws[:, model._unit_idx]
    return resid


def _neyman_adjust_scalar(
    g_t: np.ndarray,
    g_n: np.ndarray,
    J_tt: float,
    J_tn: np.ndarray,
    J_nn: np.ndarray,
    *,
    label: str,
):
    """Apply the Neyman-orthogonal score correction for a scalar target parameter.

    Computes ``g_t* = g_t - J_{tn} J_{nn}^{-1} g_n`` and the adjusted variance
    ``V* = J_{tt} - J_{tn} J_{nn}^{-1} J_{nt}``. If the nuisance block is empty
    (``g_n.shape[1] == 0``) the unadjusted score and variance are returned.

    Parameters
    ----------
    g_t : np.ndarray, shape (draws,)
        Score samples for the scalar target parameter.
    g_n : np.ndarray, shape (draws, m)
        Score samples for the nuisance block.
    J_tt : float
        Information for the target parameter.
    J_tn : np.ndarray, shape (m,)
        Cross-information between target and nuisance.
    J_nn : np.ndarray, shape (m, m)
        Information for the nuisance block.
    label : str
        Label passed to :func:`_safe_inv` for diagnostic warnings.

    Returns
    -------
    g_t_star : np.ndarray, shape (draws,)
    V_star : float
    """
    g_n = np.atleast_2d(g_n)
    if g_n.shape[1] == 0:
        return g_t, float(J_tt)
    J_nn_inv = _safe_inv(np.atleast_2d(J_nn), label)
    J_tn_arr = np.atleast_1d(J_tn)
    coef = J_tn_arr @ J_nn_inv  # (m,)
    g_t_star = g_t - g_n @ coef
    V_star = float(J_tt) - float(coef @ J_tn_arr)
    return g_t_star, V_star


def _neyman_adjust_vector(
    g_t: np.ndarray,
    g_n: np.ndarray,
    J_tt: np.ndarray,
    J_tn: np.ndarray,
    J_nn: np.ndarray,
    *,
    label: str,
):
    """Apply the Neyman-orthogonal score correction for a vector target parameter.

    Computes ``g_t* = g_t - g_n @ (J_{nt} J_{nn}^{-1})`` and the adjusted
    variance ``V* = J_{tt} - J_{tn} J_{nn}^{-1} J_{nt}``. If the nuisance
    block is scalar (``J_nn`` is a 0-d or 1×1 array), the inverse is plain
    division. If the nuisance is empty, returns the unadjusted values.

    Parameters
    ----------
    g_t : np.ndarray, shape (draws, p)
        Score samples for the vector target parameter.
    g_n : np.ndarray, shape (draws,) or (draws, m)
        Score samples for the nuisance (scalar or vector).
    J_tt : np.ndarray, shape (p, p)
        Information for the target block.
    J_tn : np.ndarray, shape (p,) or (p, m)
        Cross-information between target and nuisance.
    J_nn : np.ndarray, shape (), (1,) or (m, m)
        Information for the nuisance block.
    label : str
        Label passed to :func:`_safe_inv` for diagnostic warnings.

    Returns
    -------
    g_t_star : np.ndarray, shape (draws, p)
    V_star : np.ndarray, shape (p, p)
    """
    g_n = np.atleast_2d(g_n)
    if g_n.shape[1] == 0:
        return g_t, J_tt

    J_nn_arr = np.atleast_2d(J_nn)
    if J_nn_arr.size == 1:
        # Scalar nuisance: inverse is plain division.
        J_nn_scalar = float(J_nn_arr.ravel()[0])
        coef = np.atleast_1d(J_tn) / (J_nn_scalar + 1e-12)  # (p,)
        g_t_star = g_t - np.outer(g_n[:, 0], coef)  # (draws, p)
        V_star = J_tt - np.outer(coef, np.atleast_1d(J_tn))
    else:
        J_nn_inv = _safe_inv(J_nn_arr, label)
        J_tn_arr = np.atleast_2d(J_tn)
        coef = J_nn_inv @ J_tn_arr.T  # (m, p)
        g_t_star = g_t - g_n @ coef  # (draws, p)
        V_star = J_tt - J_tn_arr.T @ coef  # (p, p) = J_tn @ J_nn_inv @ J_nt
    return g_t_star, V_star


# ---------------------------------------------------------------------------
# Phase-1 helpers — extracted from duplicated boilerplate across ~30 tests
# ---------------------------------------------------------------------------


def _compute_residuals(
    model,
    beta_draws: np.ndarray,
    *,
    use_Z: bool = False,
    rho_draws: np.ndarray | None = None,
    lam_draws: np.ndarray | None = None,
    W_sp=None,
    N: int | None = None,
    T: int | None = None,
) -> np.ndarray:
    """Compute residuals from posterior beta draws.

    Parameters
    ----------
    model : fitted model instance
        Must provide ``_y`` and ``_X``,
        and optionally ``_WX``.
    beta_draws : np.ndarray, shape (draws, k)
        Posterior draws of beta.
    use_Z : bool, default False
        If True, auto-detect whether beta covers ``[X, WX]`` via
        :func:`_resolve_X_for_beta` and use the stacked design.
    rho_draws : np.ndarray, shape (draws,), optional
        If provided, subtract ``rho * Wy`` from the fitted values.
    lam_draws : np.ndarray, shape (draws,), optional
        If provided, apply SDEM whitening: ``eps = u - lam * Wu``.
    W_sp : scipy.sparse matrix, optional
        Required when ``lam_draws`` is provided.
    N, T : int, optional
        Required when ``lam_draws`` is provided (for panel spatial lag).

    Returns
    -------
    np.ndarray, shape (draws, n)
        Residual matrix.
    """
    y = model._y
    idata = model.inference_data

    if use_Z:
        Z = _resolve_X_for_beta(model, beta_draws)
    else:
        # When the intercept was dropped (Gibbs + FE), use the
        # sub-matrix that matches the posterior beta dimensions.
        ni = model._nonintercept_indices
        X_full = model._X
        if len(ni) < X_full.shape[1] and beta_draws.shape[1] < X_full.shape[1]:
            Z = X_full[:, ni]
        else:
            Z = X_full

    fitted = beta_draws @ Z.T  # (draws, n)
    resid = y[None, :] - fitted  # (draws, n)

    if rho_draws is not None:
        Wy = model._Wy
        resid = resid - rho_draws[:, None] * Wy[None, :]

    if lam_draws is not None:
        if W_sp is None or N is None or T is None:
            raise ValueError("W_sp, N, and T are required for SDEM whitening.")
        Wu = _panel_spatial_lag(W_sp, resid, N, T)
        resid = resid - lam_draws[:, None] * Wu

    resid = _maybe_subtract_alpha(model, idata, resid)
    return resid


def _posterior_mean_sigma2(idata: xr.DataTree) -> tuple[np.ndarray, float]:
    """Return sigma draws and posterior-mean sigma².

    Parameters
    ----------
    idata : xarray.DataTree
        DataTree object containing posterior samples.

    Returns
    -------
    sigma_draws : np.ndarray, shape (draws,)
        Posterior draws of sigma.
    sigma2_mean : float
        Posterior mean of sigma².
    """
    sigma_draws = _get_posterior_draws(idata, "sigma")
    sigma2_mean = float(np.mean(sigma_draws**2))
    return sigma_draws, sigma2_mean


def _lm_scalar(
    score: np.ndarray,
    variance: float,
    *,
    test_type: str,
    df: int,
    details: Optional[Dict[str, Any]] = None,
) -> BayesianLMTestResult:
    """Finalize a scalar LM statistic: ``LM = score² / variance``.

    Centralises the ``1e-12`` epsilon guard and the call to
    :func:`_finalize_lm`.
    """
    LM = score**2 / (variance + 1e-12)
    return _finalize_lm(LM, test_type=test_type, df=df, details=details)


def _lm_vector(
    score: np.ndarray,
    V: np.ndarray,
    *,
    test_type: str,
    df: int,
    details: Optional[Dict[str, Any]] = None,
    label: str = "information matrix",
) -> BayesianLMTestResult:
    """Finalize a vector LM statistic: ``LM = g' V⁻¹ g``.

    Centralises the ``_safe_inv`` call, the ``einsum`` pattern, and the
    ``1e-12`` epsilon guard.
    """
    V_inv = _safe_inv(V, label)
    LM = np.einsum("di,ij,dj->d", score, V_inv, score)
    return _finalize_lm(LM, test_type=test_type, df=df, details=details)


# ---------------------------------------------------------------------------
# Panel spatial lag helper (needed by _compute_residuals for SDEM whitening)
# ---------------------------------------------------------------------------


def _panel_spatial_lag(W_sparse, v: np.ndarray, N: int, T: int) -> np.ndarray:
    """Apply panel spatial lag ``I_T ⊗ W`` to a vector or batch of draws.

    Exploits the Kronecker structure ``I_T ⊗ W_n``: all T time-period
    slices are processed in a single sparse BLAS call by reshaping the
    input to ``(T, N)`` (1-D) or ``(draws*T, N)`` (2-D) and applying the
    N×N weight matrix once, then reshaping back.  This avoids the O(T)
    Python-loop overhead of the naive per-period approach and matches the
    Phase-3 ``_batch_sparse_lag`` optimization in the model layer.

    Parameters
    ----------
    W_sparse : scipy.sparse matrix
        N×N spatial weights matrix.
    v : np.ndarray
        Either a 1-D vector of length N*T, or a 2-D array of shape
        ``(draws, N*T)``.
    N : int
        Number of cross-sectional units.
    T : int
        Number of time periods.

    Returns
    -------
    np.ndarray
        Spatially lagged array with the same shape as *v*.
    """
    if v.ndim == 1:
        # Reshape (N*T,) → (T, N), apply W once, reshape back.
        r = v.reshape(T, N)
        Wr = np.asarray(W_sparse @ r.T, dtype=float).T  # (T, N)
        return Wr.reshape(N * T)
    else:
        # v is (draws, N*T) — reshape to (draws*T, N), single sparse matmul.
        draws = v.shape[0]
        r = v.reshape(draws * T, N)
        Wr = np.asarray(W_sparse @ r.T, dtype=float).T  # (draws*T, N)
        return Wr.reshape(draws, N * T)


# ---------------------------------------------------------------------------
# Shared utilities for information-matrix blocks and projections
# ---------------------------------------------------------------------------


def _trace_WtW_WW(W_sparse) -> float:
    r"""Compute :math:`\operatorname{tr}(W^\top W + W^2)` from a sparse W.

    Used by the information-matrix blocks of the Bayesian LM tests
    (Anselin 1996, eq. 13; Koley & Bera 2024, Section 3).

    Parameters
    ----------
    W_sparse : scipy.sparse matrix
        N×N spatial weights matrix.

    Returns
    -------
    float
        :math:`\operatorname{tr}(W^\top W) + \operatorname{tr}(W^2)`.
    """
    # tr(W'W) = ||W||_F^2 = sum(W_ij^2)  [O(nnz)]
    # tr(W^2) = sum_ij W_ij * W_ji = sum(W * W.T)  [O(nnz)]
    return float(W_sparse.power(2).sum() + W_sparse.multiply(W_sparse.T).sum())


def _resolve_lam_name(idata) -> str:
    """Return the posterior variable name for the SEM/SDEM spatial parameter.

    Some models store the spatial-error parameter as ``"lam"`` and others as
    ``"lambda"`` (PyMC reserves ``lambda`` for Python's keyword in some
    contexts).  This helper checks which is present.

    Parameters
    ----------
    idata : xarray.DataTree
        Fitted-model inference data with a ``posterior`` group.

    Returns
    -------
    str
        Either ``"lam"`` or ``"lambda"``.
    """
    return "lam" if "lam" in idata.posterior else "lambda"
