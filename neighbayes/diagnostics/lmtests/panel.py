"""Bayesian LM diagnostic tests — panel variants.

All public test functions return :class:`BayesianLMTestResult`.
"""

import numpy as np
import scipy.sparse as sp

from neighbayes.diagnostics.lmtests.core import (
    BayesianLMTestResult,
    _compute_residuals,
    _finalize_lm,
    _get_posterior_draws,
    _lm_scalar,
    _lm_vector,
    _mx_cross,
    _mx_quadratic,
    _neyman_adjust_scalar,
    _neyman_adjust_vector,
    _panel_spatial_lag,
    _posterior_mean_sigma2,
    _resolve_X_for_beta,
    _safe_inv,
    _trace_WtW_WW,
)
from neighbayes.diagnostics.lmtests.cross_sectional import (
    _info_matrix_blocks_slx_robust,
)


def bayesian_panel_lm_wx_sem_test(
    model,
) -> "BayesianLMTestResult":
    r"""Bayesian panel LM test for WX coefficients in SEM (H₀: γ = 0 | SEM).

    Tests whether spatially lagged covariates (WX) should be added to a
    panel SEM model, i.e., whether the SEM panel model should be extended to an SDEM panel.
    Follows the multi-parameter Bayesian LM test framework of
    :cite:t:`dogan2021BayesianRobust`. The classical WX-in-SEM construction is
    discussed in :cite:t:`koley2024UseNot`; panel-data extensions are covered
    in :cite:t:`elhorst2014SpatialEconometrics` and :cite:t:`anselin2008SpatialPanel`.

    The null model is panel SEM (includes λ but not γ). For each posterior draw
    from the SEM model, residuals are:

    .. math::
        \mathbf{e} = \mathbf{y} - X \beta

    The score vector for the WX coefficients is:

    .. math::
        \mathbf{g}_\gamma = (WX)^\top \mathbf{e}

    The concentration matrix is the information matrix evaluated at the posterior mean:

    .. math::
        J_{\gamma\gamma} = \frac{1}{\bar{\sigma}^2} (WX)^\top (WX)

    Parameters
    ----------
    model : SEMPanelFE or SEMPanelRE
        Fitted SEM panel model instance with ``inference_data`` attribute
        containing posterior draws for ``beta``, ``lambda``, and ``sigma``.

    Returns
    -------
    BayesianLMTestResult
        Dataclass containing LM samples, summary statistics, and metadata.
        The ``df`` field is set to :math:`k_{wx}` (number of WX columns).
    """
    X = model._X
    WX = model._WX
    k_wx = WX.shape[1]
    N = getattr(model, "_N", None)
    T = getattr(model, "_T", None)

    if k_wx == 0:
        raise ValueError(
            "Model has no WX columns. The panel WX test requires at least one "
            "spatially lagged covariate."
        )

    idata = model.inference_data
    beta_draws = _get_posterior_draws(idata, "beta")  # (draws, k)

    resid = _compute_residuals(model, beta_draws)

    # Score: g_gamma = WX' @ e for each draw → (draws, k_wx)
    g_gamma = resid @ WX  # (draws, k_wx)

    # M_X-projected raw-score variance (Koley-Bera 2024).
    _, sigma2_mean = _posterior_mean_sigma2(idata)
    V_gamma_gamma = sigma2_mean * _mx_cross(X, WX, WX)

    return _lm_vector(
        g_gamma,
        V_gamma_gamma,
        test_type="bayesian_panel_lm_wx_sem",
        df=k_wx,
        details={"k_wx": k_wx, "N": N, "T": T},
        label="V_gamma_gamma (panel WX-SEM)",
    )


def _panel_residuals(model, beta_draws: np.ndarray) -> np.ndarray:
    """Compute panel residuals from posterior beta draws.

    Dispatches on FE vs RE model structure:

    - **FE models** (``_y`` and ``_X`` already demeaned):
      ``e = y - X @ beta``

    - **RE models** (raw scale, with ``alpha`` in posterior):
      ``e = y - X @ beta - alpha[unit_idx]``

    Parameters
    ----------
    model : SpatialModel
        Fitted panel model with ``inference_data`` attribute.
    beta_draws : np.ndarray
        Posterior draws of beta, shape ``(draws, k)``.

    Returns
    -------
    np.ndarray
        Residual matrix of shape ``(draws, n)`` where ``n = N*T``.
    """
    return _compute_residuals(model, beta_draws, use_Z=True)


def _panel_info_matrix_blocks(
    X: np.ndarray,
    WX: np.ndarray,
    W_sparse,
    W_eigs: np.ndarray,
    sigma2: float,
    N: int,
    T: int,
    y_hat: np.ndarray | None = None,
    Wy_hat: np.ndarray | None = None,
    T_ww: float | None = None,
) -> dict:
    r"""Raw-score variance blocks for panel LM tests (SDM/SDEM).

    Returns the blocks of the **raw-score variance matrix**
    :math:`V = \sigma^2 J` (i.e. variance of unscaled scores
    :math:`\mathbf{e}^\top W \mathbf{y}`, :math:`(WX)^\top \mathbf{e}`)
    with the appropriate :math:`M_X = I - X(X^\top X)^{-1} X^\top`
    projection on the WX-direction blocks.  Panel-data structure
    contributes the :math:`T` multiplier on the trace term
    (:cite:p:`anselin2008SpatialPanel`, :cite:p:`elhorst2014SpatialEconometrics`):

    .. math::
        V_{\rho\rho}    &= \sigma^4 \, T \cdot \mathrm{tr}(W'W + W^2)
                         + \sigma^2 \, (W\hat{y})^\top M_X (W\hat{y}) \\
        V_{\rho\gamma}  &= \sigma^2 \, (W\hat{y})^\top M_X (WX) \\
        V_{\gamma\gamma}&= \sigma^2 \, (WX)^\top M_X (WX) \\
        V_{\lambda\lambda} &= \sigma^4 \, T \cdot \mathrm{tr}(W'W + W^2) \\
        V_{\rho\lambda} &= \sigma^4 \, T \cdot \mathrm{tr}(W'W + W^2) \\
        V_{\lambda\gamma} &= \mathbf{0} \quad \text{under spherical errors.}

    Dictionary keys are kept as ``J_*`` for backward compatibility with
    earlier API users; the returned values are the raw-score variance
    blocks (``V_*``) in the new convention.

    Parameters
    ----------
    X, WX : np.ndarray
        Design matrix ``(n, k)`` (FE-demeaned for FE models) and
        spatially-lagged covariates ``(n, k_wx)``.
    W_sparse : scipy.sparse matrix
        N×N spatial weights matrix.
    W_eigs : np.ndarray
        Eigenvalues of W (currently unused; kept for signature stability).
    sigma2 : float
        Posterior-mean :math:`\bar{\sigma}^2`.
    N, T : int
        Cross-sectional units and time periods.
    y_hat, Wy_hat : np.ndarray, optional
        Fitted values and panel-spatial-lag of fitted values, shape
        ``(n,)``.  At least one must be supplied for the
        :math:`V_{\rho\rho}` and :math:`V_{\rho\gamma}` projection terms.
    T_ww : float, optional
        Pre-computed :math:`\mathrm{tr}(W'W + W^2)`; recomputed if None.

    Returns
    -------
    dict
        Keys ``J_rho_rho``, ``J_lam_lam``, ``J_rho_lam``,
        ``J_rho_gamma``, ``J_gamma_gamma``, ``T_ww``, ``T_mult``.  All
        ``J_*`` entries are raw-score variance blocks (with proper
        :math:`M_X` projection on WX directions).
    """
    n = X.shape[0]
    k_wx = WX.shape[1]

    if T_ww is None:
        T_ww = _trace_WtW_WW(W_sparse)
    T_mult = T

    # Variance blocks (raw-score scale, panel)
    J_lam_lam = sigma2 * sigma2 * T_mult * T_ww
    J_rho_lam = sigma2 * sigma2 * T_mult * T_ww  # under H0 (Elhorst 2014)

    if y_hat is not None or Wy_hat is not None:
        if Wy_hat is None:
            Wy_hat = _panel_spatial_lag(W_sparse, y_hat, N, T_mult)
        J_rho_rho = sigma2 * sigma2 * T_mult * T_ww + sigma2 * _mx_quadratic(X, Wy_hat)
    else:
        # Fallback (pre-fit point): drop the M_X term.
        J_rho_rho = sigma2 * sigma2 * T_mult * T_ww + sigma2 * float(n)

    # WX-direction blocks: same M_X-projected form as cross-section.
    if k_wx > 0:
        J_gamma_gamma = sigma2 * _mx_cross(X, WX, WX)
    else:
        J_gamma_gamma = np.zeros((0, 0))

    if Wy_hat is not None and k_wx > 0:
        J_rho_gamma = sigma2 * np.asarray(_mx_cross(X, Wy_hat, WX)).ravel()
    else:
        J_rho_gamma = np.zeros(k_wx)

    return {
        "J_rho_rho": J_rho_rho,
        "J_lam_lam": J_lam_lam,
        "J_rho_lam": J_rho_lam,
        "J_rho_gamma": J_rho_gamma,
        "J_gamma_gamma": J_gamma_gamma,
        "T_ww": T_ww,
        "T_mult": T_mult,
    }


# ---------------------------------------------------------------------------
# Panel Bayesian LM tests — lag and error (Anselin et al. 2008, Elhorst 2014)
# ---------------------------------------------------------------------------


def bayesian_panel_lm_lag_test(
    model,
) -> BayesianLMTestResult:
    """Bayesian panel LM test for omitted spatial lag (H₀: ρ = 0).

    Follows :cite:t:`anselin2008SpatialPanel` and the Bayesian framework of :cite:t:`dogan2021BayesianRobust`.
    (2021). Tests whether a spatial lag term should be added to a panel
    regression.

    The null model is a pooled/FE panel OLS. For each posterior draw,
    residuals are:

    .. math::
        \\mathbf{e} = \\mathbf{y} - X \\beta

    (demeaned for FE models; with alpha subtracted for RE models).

    The score for each draw is:

    .. math::
        S = \\mathbf{e}^\\top W_{NT} \\mathbf{y}

    where :math:`W_{NT} = W \\otimes I_T` is the block-diagonal panel
    weights matrix.

    The information matrix (:cite:t:`anselin2008SpatialPanel`) is:

    .. math::
        J = \\frac{1}{\\sigma^2} \\left(
        (W\\hat{y})^\\top M (W\\hat{y}) + T \\cdot \\mathrm{tr}(W'W + W^2)
        \\right)

    where :math:`M = I - X(X^\\top X)^{-1} X^\\top` and
    :math:`\\hat{y} = X\\bar{\\beta}`.

    The LM statistic for each draw is:

    .. math::
        \\mathrm{LM} = \\frac{S^2}{\\sigma^2 \\cdot J}

    which is distributed as :math:`\\chi^2_1` under H₀.

    Parameters
    ----------
    model : SpatialModel
        Fitted panel model (e.g. ``OLSPanelFE``, ``OLSPanelRE``) with
        ``inference_data`` attribute containing posterior draws for
        ``beta`` and ``sigma``.

    Returns
    -------
    BayesianLMTestResult
        Dataclass containing LM samples, summary statistics, and metadata.

    """
    Wy = model._Wy
    W_sp = model._W_sparse
    N = model._N
    T = model._T

    idata = model.inference_data
    beta_draws = _get_posterior_draws(idata, "beta")  # (draws, k)

    # Panel residuals (handles FE vs RE)
    resid = _panel_residuals(model, beta_draws)  # (draws, n)

    # Score: S = e'Wy for each draw
    S = np.dot(resid, Wy)  # (draws,)

    # Use the design matrix that matches the posterior beta block; for
    # SLX/SDM/SDEM panels this is ``Z = [X, WX]`` so the projection
    # absorbs the WX nuisance direction.
    Z = _resolve_X_for_beta(model, beta_draws)

    # Compute information matrix for panel LM-lag (Anselin et al. 2008)
    # J = (Wb'MWb + T*tr(W'W+W²)*σ²) / σ²
    # LM = (e'Wy)² / (σ² * J) = (e'Wy)² / (Wb'MWb + T*tr*σ²)
    beta_mean = np.mean(beta_draws, axis=0)  # (k,)
    _, sigma2_mean = _posterior_mean_sigma2(idata)
    y_hat = Z @ beta_mean  # (n,)

    # Panel spatial lag of y_hat
    Wy_hat = _panel_spatial_lag(W_sp, y_hat, N, T)  # (n,)

    # Annihilator matrix: M = I - Z(Z'Z)^{-1}Z'
    ZtZ_inv = _safe_inv(Z.T @ Z, "X'X (panel LM-lag)")
    M_Wy = Wy_hat - Z @ (ZtZ_inv @ (Z.T @ Wy_hat))
    WbMWb = float(Wy_hat @ M_Wy)

    T_ww = model._T_ww
    J_val = WbMWb + T * T_ww * sigma2_mean

    # Variance evaluated at posterior mean theta* (Dogan et al. 2021,
    # eq. 3.6); only the score varies across draws, consistent with the
    # cross-sectional bayesian_lm_lag_test.
    V = sigma2_mean * J_val
    LM = S**2 / (V + 1e-12)

    return _finalize_lm(
        LM,
        test_type="bayesian_panel_lm_lag",
        df=1,
        details={"N": N, "T": T},
    )


def bayesian_panel_lm_error_test(
    model,
) -> BayesianLMTestResult:
    """Bayesian panel LM test for omitted spatial error (H₀: λ = 0).

    Follows :cite:t:`anselin2008SpatialPanel` and the Bayesian framework of :cite:t:`dogan2021BayesianRobust`.
    (2021). Tests whether a spatial error term should be added to a panel
    regression.

    The null model is a pooled/FE panel OLS. For each posterior draw,
    residuals are computed (demeaned for FE, alpha-adjusted for RE).

    The score for each draw is:

    .. math::
        S = \\mathbf{e}^\\top W_{NT} \\mathbf{e}

    The variance is:

    .. math::
        V = \\sigma^4 \\cdot T \\cdot \\mathrm{tr}(W'W + W^2)

    The LM statistic for each draw is:

    .. math::
        \\mathrm{LM} = \\frac{S^2}{V}

    which is distributed as :math:`\\chi^2_1` under H₀.

    Parameters
    ----------
    model : SpatialModel
        Fitted panel model (e.g. ``OLSPanelFE``, ``OLSPanelRE``) with
        ``inference_data`` attribute containing posterior draws for
        ``beta`` and ``sigma``.

    Returns
    -------
    BayesianLMTestResult
        Dataclass containing LM samples, summary statistics, and metadata.

    """
    W_sp = model._W_sparse
    N = model._N
    T = model._T

    idata = model.inference_data
    beta_draws = _get_posterior_draws(idata, "beta")  # (draws, k)

    # Panel residuals
    resid = _panel_residuals(model, beta_draws)  # (draws, n)

    # Score: S = e'W_nt e for each draw
    # Apply N×N W period-by-period to the N*T stacked residuals
    We_panel = _panel_spatial_lag(W_sp, resid, N, T)  # (draws, n)

    S = np.sum(resid * We_panel, axis=1)  # (draws,)

    # Variance: V = sigma^4 * T * tr(W'W + W²)
    T_ww = model._T_ww
    # Variance evaluated at posterior mean theta* (Dogan et al. 2021,
    # eq. 3.6); consistent with the cross-sectional bayesian_lm_error_test.
    _, sigma2_mean = _posterior_mean_sigma2(idata)
    V = sigma2_mean**2 * T * T_ww  # scalar

    return _lm_scalar(
        S,
        V,
        test_type="bayesian_panel_lm_error",
        df=1,
        details={"N": N, "T": T},
    )


def bayesian_panel_robust_lm_lag_test(
    model,
) -> BayesianLMTestResult:
    r"""Bayesian panel robust LM-Lag test (H₀: ρ = 0, robust to λ).

    Follows :cite:t:`elhorst2014SpatialEconometrics`. Tests the null
    hypothesis that the spatial lag coefficient is zero, robust to the
    local presence of spatial error autocorrelation.

    The null model is a pooled/FE panel OLS.  For each posterior draw,

    .. math::
        \mathrm{LM}_R = \frac{
        \left( \frac{\mathbf{e}^\top W_{NT} \mathbf{y}}{\sigma^2}
        - \frac{\mathbf{e}^\top W_{NT} \mathbf{e}}{\sigma^2} \right)^2
        }{J - T \cdot \mathrm{tr}(W'W + W^2)}

    where :math:`J` is the information matrix from the panel LM-lag test
    and :math:`\mathrm{tr}` denotes :math:`\mathrm{tr}(W'W + W^2)`.

    The score is evaluated at the **M_X-projected residual**
    :math:`\mathbf{e}_\perp = M_X \mathbf{y}` (constant across draws),
    because :math:`\beta` is information-orthogonal to
    :math:`(\rho,\lambda)` under :math:`H_0` and therefore
    :math:`\beta`-posterior variance does not enter the LM reference
    distribution.  This matches the cross-sectional correction
    documented in :func:`bayesian_robust_lm_lag_test`.

    This is distributed as :math:`\chi^2_1` under H₀.

    Parameters
    ----------
    model : SpatialModel
        Fitted panel model (e.g. ``OLSPanelFE``) with ``inference_data``
        attribute containing posterior draws for ``beta`` and ``sigma``.

    Returns
    -------
    BayesianLMTestResult
        Dataclass containing LM samples, summary statistics, and metadata.

    """
    y = model._y
    X = model._X
    Wy = model._Wy
    W_sp = model._W_sparse
    N = model._N
    T = model._T

    idata = model.inference_data
    beta_draws = _get_posterior_draws(idata, "beta")  # (draws, k)
    sigma_draws, sigma2_mean = _posterior_mean_sigma2(idata)

    # Use M_X-projected residual e_perp = M_X y (fixed across draws). The
    # ABFY robust score is evaluated at the OLS estimator, where residuals
    # are X-orthogonal by construction. Replacing e^(d) = y - X beta^(d)
    # with M_X y keeps the score invariant to beta posterior draws -- the
    # information-orthogonality of beta to (rho, lambda) under H_0 means
    # beta-posterior variance does not enter the LM reference distribution.
    XtX_inv = _safe_inv(X.T @ X, "X'X (panel robust LM-lag)")
    e_perp = y - X @ (XtX_inv @ (X.T @ y))  # = M_X y, shape (n,)
    We_perp = _panel_spatial_lag(W_sp, e_perp, N, T)
    S_lag = float(e_perp @ Wy)  # scalar (constant across draws)
    S_err = float(e_perp @ We_perp)  # scalar (constant across draws)

    # Information matrix for lag test
    beta_mean = np.mean(beta_draws, axis=0)
    X_beta = _resolve_X_for_beta(model, beta_draws)
    y_hat = X_beta @ beta_mean
    Wy_hat = _panel_spatial_lag(W_sp, y_hat, N, T)
    M_Wy = Wy_hat - X @ (XtX_inv @ (X.T @ Wy_hat))
    WbMWb = float(Wy_hat @ M_Wy)

    T_ww = model._T_ww
    J_val = WbMWb + T * T_ww * sigma2_mean

    # Robust LM = (S_lag/σ² - S_err/σ²)² / (J - T*tr)
    # where J is in σ² units and tr = tr(W'W+W²)
    sigma2_draws = sigma_draws**2
    robust_score = (S_lag - S_err) / sigma2_draws  # (draws,)
    denom = J_val / sigma2_mean - T * T_ww  # J/σ² - T*tr (scalar)

    LM = robust_score**2 / (abs(denom) + 1e-12)

    return _finalize_lm(
        LM,
        test_type="bayesian_panel_robust_lm_lag",
        df=1,
        details={"N": N, "T": T},
    )


def bayesian_panel_robust_lm_error_test(
    model,
) -> BayesianLMTestResult:
    r"""Bayesian panel robust LM-Error test (H₀: λ = 0, robust to ρ).

    Follows :cite:t:`elhorst2014SpatialEconometrics`. Tests the null
    hypothesis that the spatial error coefficient is zero, robust to the
    local presence of a spatial lag.

    The null model is a pooled/FE panel OLS.  For each posterior draw,

    .. math::
        \mathrm{LM}_R = \frac{
        \left( \frac{\mathbf{e}^\top W_{NT} \mathbf{e}}{\sigma^2}
        - \frac{T \cdot \mathrm{tr}}{J} \cdot
        \frac{\mathbf{e}^\top W_{NT} \mathbf{y}}{\sigma^2} \right)^2
        }{
        T \cdot \mathrm{tr} \cdot \left(1 - \frac{T \cdot \mathrm{tr}}{J}\right)
        }

    where :math:`J` is the information matrix from the panel LM-lag test
    and :math:`\mathrm{tr} = \mathrm{tr}(W'W + W^2)`.

    The score is evaluated at the **M_X-projected residual**
    :math:`\mathbf{e}_\perp = M_X \mathbf{y}` (constant across draws),
    because :math:`\beta` is information-orthogonal to
    :math:`(\rho,\lambda)` under :math:`H_0` and therefore
    :math:`\beta`-posterior variance does not enter the LM reference
    distribution.  This matches the cross-sectional correction
    documented in :func:`bayesian_robust_lm_error_test`.

    This is distributed as :math:`\chi^2_1` under H₀.

    Parameters
    ----------
    model : SpatialModel
        Fitted panel model (e.g. ``OLSPanelFE``) with ``inference_data``
        attribute containing posterior draws for ``beta`` and ``sigma``.

    Returns
    -------
    BayesianLMTestResult
        Dataclass containing LM samples, summary statistics, and metadata.

    """
    y = model._y
    X = model._X
    Wy = model._Wy
    W_sp = model._W_sparse
    N = model._N
    T = model._T

    idata = model.inference_data
    beta_draws = _get_posterior_draws(idata, "beta")  # (draws, k)
    sigma_draws, sigma2_mean = _posterior_mean_sigma2(idata)

    # Use M_X-projected residual (see bayesian_panel_robust_lm_lag_test
    # for the principled justification: beta is information-orthogonal to
    # (rho, lambda) under H_0, so beta posterior variance does not enter
    # the LM reference distribution).
    XtX_inv = _safe_inv(X.T @ X, "X'X (panel robust LM-error)")
    e_perp = y - X @ (XtX_inv @ (X.T @ y))
    We_perp = _panel_spatial_lag(W_sp, e_perp, N, T)
    S_lag = float(e_perp @ Wy)
    S_err = float(e_perp @ We_perp)

    # Information matrix for lag test
    beta_mean = np.mean(beta_draws, axis=0)
    X_beta = _resolve_X_for_beta(model, beta_draws)
    y_hat = X_beta @ beta_mean
    Wy_hat = _panel_spatial_lag(W_sp, y_hat, N, T)
    M_Wy = Wy_hat - X @ (XtX_inv @ (X.T @ Wy_hat))
    WbMWb = float(Wy_hat @ M_Wy)

    T_ww = model._T_ww
    J_val = WbMWb + T * T_ww * sigma2_mean

    # Robust LM = (S_err/σ² - T*tr/J * S_lag/σ²)² / (T*tr*(1 - T*tr/J))
    # where J is in raw units (not divided by σ²)
    # J/σ² is the scaled version
    J_scaled = J_val / sigma2_mean  # J/σ²
    Ttr = T * T_ww

    sigma2_draws = sigma_draws**2
    robust_score = (S_err - (Ttr / J_scaled) * S_lag) / sigma2_draws
    denom = Ttr * (1 - Ttr / J_scaled)

    LM = robust_score**2 / (abs(denom) + 1e-12)

    return _finalize_lm(
        LM,
        test_type="bayesian_panel_robust_lm_error",
        df=1,
        details={"N": N, "T": T},
    )


# ---------------------------------------------------------------------------
# Panel Bayesian LM tests — SDM/SDEM variants (Koley & Bera 2024)
# ---------------------------------------------------------------------------


def bayesian_panel_lm_wx_test(
    model,
) -> BayesianLMTestResult:
    """Bayesian panel LM test for WX coefficients (H₀: γ = 0).

    Tests whether spatially lagged covariates (WX) should be added to a
    SAR panel model, i.e., whether SAR should be extended to SDM.
    Follows the multi-parameter Bayesian LM test framework of
    :cite:t:`dogan2021BayesianRobust` and :cite:t:`koley2024UseNot`.

    The null model is a SAR panel (includes ρ but not γ). For each
    posterior draw from the SAR model, residuals are:

    .. math::
        \\mathbf{e} = \\mathbf{y} - \\rho W \\mathbf{y} - X \\beta

    The score vector for the WX coefficients is:

    .. math::
        \\mathbf{g}_\\gamma = (WX)^\\top \\mathbf{e}

    The concentration matrix is:

    .. math::
        J_{\\gamma\\gamma} = \\frac{1}{\\bar{\\sigma}^2} (WX)^\\top (WX)

    The LM statistic for each draw is:

    .. math::
        \\mathrm{LM} = \\mathbf{g}_\\gamma^\\top J_{\\gamma\\gamma}^{-1}
        \\mathbf{g}_\\gamma

    distributed as :math:`\\chi^2_{k_{wx}}` under H₀.

    Parameters
    ----------
    model : SARPanelFE or SARPanelRE
        Fitted SAR panel model with ``inference_data`` containing
        posterior draws for ``beta``, ``rho``, and ``sigma``.

    Returns
    -------
    BayesianLMTestResult
        Dataclass containing LM samples, summary statistics, and metadata.
        The ``df`` field is set to :math:`k_{wx}`.
    """
    X = model._X
    WX = model._WX
    k_wx = WX.shape[1]
    N = model._N
    T = model._T

    if k_wx == 0:
        raise ValueError(
            "Model has no WX columns. The panel WX test requires at least one "
            "spatially lagged covariate."
        )

    idata = model.inference_data
    beta_draws = _get_posterior_draws(idata, "beta")  # (draws, k)
    rho_draws = _get_posterior_draws(idata, "rho")  # (draws,)

    resid = _compute_residuals(model, beta_draws, rho_draws=rho_draws)

    # Score: g_gamma = WX' @ e for each draw → (draws, k_wx)
    g_gamma = resid @ WX  # (draws, k_wx)

    # M_X-projected raw-score variance (Koley-Bera 2024).
    _, sigma2_mean = _posterior_mean_sigma2(idata)
    V_gamma_gamma = sigma2_mean * _mx_cross(X, WX, WX)

    return _lm_vector(
        g_gamma,
        V_gamma_gamma,
        test_type="bayesian_panel_lm_wx",
        df=k_wx,
        details={"k_wx": k_wx, "N": N, "T": T},
        label="V_gamma_gamma (panel LM-WX)",
    )


def bayesian_panel_lm_sdm_joint_test(
    model,
) -> BayesianLMTestResult:
    """Bayesian panel joint LM test for SDM (H₀: ρ = 0 AND γ = 0).

    Tests the joint null hypothesis that both the spatial lag coefficient
    and the WX coefficients are zero, i.e., whether the OLS panel model
    should be extended to an SDM specification.

    The null model is OLS panel. The joint score vector is:

    .. math::
        \\mathbf{g} = \\begin{pmatrix} \\mathbf{e}^\\top W \\mathbf{y} \\\\
        (WX)^\\top \\mathbf{e} \\end{pmatrix}

    a :math:`(1 + k_{wx}) \\times 1` vector for each draw. The
    concentration matrix uses panel-adjusted J_{ρρ}:

    .. math::
        J = \\begin{pmatrix}
        J_{\\rho\\rho} & J_{\\rho\\gamma} \\\\
        J_{\\gamma\\rho} & J_{\\gamma\\gamma}
        \\end{pmatrix}

    The LM statistic is :math:`\\chi^2_{1 + k_{wx}}` under H₀. The classical
    joint LM construction follows :cite:t:`koley2024UseNot`; panel-data
    extensions are discussed in :cite:t:`anselin2008SpatialPanel` and
    :cite:t:`elhorst2014SpatialEconometrics`. The Bayesian LM statistic is
    computed per posterior draw following :cite:t:`dogan2021BayesianRobust`.

    Parameters
    ----------
    model : SpatialModel
        Fitted OLS panel model with ``inference_data`` containing
        posterior draws for ``beta`` and ``sigma``.

    Returns
    -------
    BayesianLMTestResult
        Dataclass containing LM samples, summary statistics, and metadata.
        The ``df`` field is set to :math:`1 + k_{wx}`.
    """
    X = model._X
    WX = model._WX
    Wy = model._Wy
    W_sp = model._W_sparse
    k_wx = WX.shape[1]
    N = model._N
    T = model._T

    idata = model.inference_data
    beta_draws = _get_posterior_draws(idata, "beta")  # (draws, k)
    _get_posterior_draws(idata, "sigma")  # (draws,)

    # Panel residuals
    resid = _panel_residuals(model, beta_draws)  # (draws, n)

    # Score components
    g_rho = np.dot(resid, Wy)  # (draws,)
    g_gamma = resid @ WX  # (draws, k_wx)

    # Joint score: g = [g_rho, g_gamma']' → (draws, 1+k_wx)
    g = np.column_stack([g_rho, g_gamma])  # (draws, 1+k_wx)

    # Information matrix (panel-adjusted)
    beta_mean = np.mean(beta_draws, axis=0)
    _, sigma2_mean = _posterior_mean_sigma2(idata)
    X_beta = _resolve_X_for_beta(model, beta_draws)
    y_hat = X_beta @ beta_mean

    info = _panel_info_matrix_blocks(
        X,
        WX,
        W_sp,
        model._W_eigs,
        sigma2_mean,
        N,
        T,
        y_hat=y_hat,
        T_ww=model._T_ww,
    )

    p = 1 + k_wx
    J = np.zeros((p, p))
    J[0, 0] = info["J_rho_rho"]
    if k_wx > 0:
        J[0, 1:] = info["J_rho_gamma"]
        J[1:, 0] = info["J_rho_gamma"]
        J[1:, 1:] = info["J_gamma_gamma"]

    return _lm_vector(
        g,
        J,
        test_type="bayesian_panel_lm_sdm_joint",
        df=p,
        details={"k_wx": k_wx, "N": N, "T": T},
        label="J (panel SDM joint)",
    )


def bayesian_panel_lm_slx_error_joint_test(
    model,
) -> BayesianLMTestResult:
    """Bayesian panel joint LM test for SDEM (H₀: λ = 0 AND γ = 0).

    Tests the joint null hypothesis that both the spatial error coefficient
    and the WX coefficients are zero, i.e., whether the OLS panel model
    should be extended to an SDEM specification.

    The null model is OLS panel. The joint score vector is:

    .. math::
        \\mathbf{g} = \\begin{pmatrix} \\mathbf{e}^\\top W \\mathbf{e} \\\\
        (WX)^\\top \\mathbf{e} \\end{pmatrix}

    The concentration matrix has zero off-diagonal blocks under H₀
    (spherical errors):

    .. math::
        J = \\begin{pmatrix}
        T \\cdot \\mathrm{tr}(W'W + W^2) & 0 \\\\
        0 & \\frac{1}{\\bar{\\sigma}^2} (WX)^\\top (WX)
        \\end{pmatrix}

    The LM statistic is :math:`\\chi^2_{1 + k_{wx}}` under H₀. The classical
    joint LM construction follows :cite:t:`koley2024UseNot`; panel-data
    extensions are discussed in :cite:t:`anselin2008SpatialPanel` and
    :cite:t:`elhorst2014SpatialEconometrics`. The Bayesian LM statistic is
    computed per posterior draw following :cite:t:`dogan2021BayesianRobust`.

    Parameters
    ----------
    model : SpatialModel
        Fitted OLS panel model with ``inference_data`` containing
        posterior draws for ``beta`` and ``sigma``.

    Returns
    -------
    BayesianLMTestResult
        Dataclass containing LM samples, summary statistics, and metadata.
        The ``df`` field is set to :math:`1 + k_{wx}`.
    """
    X = model._X
    WX = model._WX
    W_sp = model._W_sparse
    k_wx = WX.shape[1]
    N = model._N
    T = model._T

    idata = model.inference_data
    beta_draws = _get_posterior_draws(idata, "beta")  # (draws, k)

    # Panel residuals
    resid = _panel_residuals(model, beta_draws)  # (draws, n)

    # Score for error: g_lambda = e'W_nt e
    We_panel = _panel_spatial_lag(W_sp, resid, N, T)
    g_lambda = np.sum(resid * We_panel, axis=1)  # (draws,)

    # Score for WX: g_gamma = WX'e
    g_gamma = resid @ WX  # (draws, k_wx)

    # Joint score
    g = np.column_stack([g_lambda, g_gamma])  # (draws, 1+k_wx)

    # Raw-score variance matrix (block-diagonal under H0).
    _, sigma2_mean = _posterior_mean_sigma2(idata)
    T_ww = model._T_ww

    p = 1 + k_wx
    V = np.zeros((p, p))
    V[0, 0] = sigma2_mean * sigma2_mean * T * T_ww  # V_{lambda,lambda}
    if k_wx > 0:
        V[1:, 1:] = sigma2_mean * _mx_cross(X, WX, WX)

    return _lm_vector(
        g,
        V,
        test_type="bayesian_panel_lm_slx_error_joint",
        df=p,
        details={"k_wx": k_wx, "N": N, "T": T},
        label="V (panel SLX-error joint)",
    )


# ---------------------------------------------------------------------------
# Panel robust Bayesian LM tests — SDM/SDEM (Neyman orthogonal score)
# ---------------------------------------------------------------------------


def bayesian_panel_robust_lm_lag_sdm_test(
    model,
) -> BayesianLMTestResult:
    r"""Bayesian panel robust LM-Lag in SDM context (H₀: ρ = 0 | SLX panel).

    Tests :math:`H_0: \rho = 0` using the **SLX panel as the restricted
    null model**.  The SLX OLS normal equations zero the γ-direction
    score :math:`(WX)^\top \mathbf{e}_{slx} = \mathbf{0}` exactly, but
    leave the *other* spatial parameter :math:`\lambda` as a non-trivial
    nuisance.  Without correcting for it, residuals from the SLX null
    carry an unconcentrated component of any true λ and bias
    :math:`g_\rho = \mathbf{e}^\top W_{NT} \mathbf{y}` upward — see the
    SDEM-DGP failure documented in
    ``reference/lm_diagnostics_paper.md`` §5.9.  Following
    :cite:p:`dogan2021BayesianRobust` (Proposition 3) we apply the
    Schur-complement Neyman correction with respect to λ:

    .. math::
        g_\rho^{*\,(d)}      &= g_\rho^{(d)}
            - (J_{\rho\lambda}/J_{\lambda\lambda})\, g_\lambda^{(d)}, \\
        V_{\rho|\lambda}     &= J_{\rho\rho} - J_{\rho\lambda}^2 / J_{\lambda\lambda}, \\
        \mathrm{LM}^{(d)}    &= (g_\rho^{*\,(d)})^2 / V_{\rho|\lambda}
        \;\xrightarrow{d}\; \chi^2_1.

    The information blocks are computed at the posterior means using
    Isserlis' identity for :math:`\mathbf{e} \sim N(0, \sigma^2 M_Z)`
    (with :math:`Z = [X,\;WX]`):

    .. math::
        J_{\rho\rho}       &= \bar{\sigma}^4\, T \cdot T_{WW}
                              + \bar{\sigma}^2 \|M_Z W_{NT} Z\bar{\beta}_{slx}\|^2 \\
        J_{\lambda\lambda} &= \bar{\sigma}^4\, T \cdot T_{WW} \\
        J_{\rho\lambda}     &= \bar{\sigma}^4 \bigl[
            \mathrm{tr}(M_Z W_{NT} M_Z W_{NT}) +
            \mathrm{tr}(M_Z W_{NT} M_Z W_{NT}^\top)\bigr]

    where :math:`T_{WW} = \mathrm{tr}(W^\top W + W^2)` and
    :math:`W_{NT} = I_T \otimes W`.  Trace blocks are computed exactly
    via :math:`Z^\top W_{NT} Z`, :math:`Z^\top W_{NT}^2 Z`, etc., so
    that we never form the dense :math:`(NT, NT)` annihilator.

    Note: this null model differs from the spreg-style algebraic
    subtraction (joint SDM minus LM-WX), which uses an OLS null and
    enforces orthogonality through information-matrix algebra.  The two
    statistics target the same hypothesis but are not numerically equal.

    Parameters
    ----------
    model : SLXPanelFE or SLX-like panel model
        Fitted SLX panel model with ``inference_data`` containing
        posterior draws for ``beta`` and ``sigma`` (``beta`` covers the
        full SLX design ``[X, WX]``).

    Returns
    -------
    BayesianLMTestResult
        Per-draw LM samples, summary statistics and ``df = 1``.
    """
    X = model._X
    WX = model._WX
    Wy = model._Wy
    W_sp = model._W_sparse
    k_wx = WX.shape[1]
    N = model._N
    T = model._T

    idata = model.inference_data
    beta_draws = _get_posterior_draws(idata, "beta")  # (draws, k_total)

    resid = _compute_residuals(model, beta_draws, use_Z=True)

    # Per-draw raw scores
    g_rho = np.dot(resid, Wy)  # (draws,)
    We_panel = _panel_spatial_lag(W_sp, resid, N, T)
    g_lambda = np.sum(resid * We_panel, axis=1)  # (draws,)

    # Information matrix blocks at posterior means using full W_NT.
    beta_mean = np.mean(beta_draws, axis=0)
    _, sigma2_mean = _posterior_mean_sigma2(idata)
    W_NT = sp.kron(sp.identity(T, format="csr"), W_sp, format="csr")
    Z_beta = _resolve_X_for_beta(model, beta_draws)
    blocks = _info_matrix_blocks_slx_robust(
        X=X,
        WX=WX,
        W_sparse=W_NT,
        sigma2=sigma2_mean,
        beta_slx_mean=beta_mean,
        T_ww=T * model._T_ww,
        Z_beta=Z_beta,
    )
    J_rr = blocks["J_rho_rho"]
    J_ll = blocks["J_lam_lam"]
    J_rl = blocks["J_rho_lam"]

    # Schur-complement Neyman correction on λ
    g_rho_star, V_rho_given_lambda = _neyman_adjust_scalar(
        g_rho,
        g_lambda[:, None],
        J_rr,
        J_rl,
        J_ll,
        label="V_rho_given_lambda (panel robust LM-Lag-SDM)",
    )

    return _lm_scalar(
        g_rho_star,
        V_rho_given_lambda,
        test_type="bayesian_panel_robust_lm_lag_sdm",
        df=1,
        details={
            "k_wx": k_wx,
            "N": N,
            "T": T,
            "J_rho_rho": J_rr,
            "J_lam_lam": J_ll,
            "J_rho_lam": J_rl,
        },
    )


def bayesian_panel_robust_lm_wx_test(
    model,
) -> BayesianLMTestResult:
    r"""Bayesian panel robust LM-WX (H₀: γ = 0 | SAR panel, robust to ρ).

    Bayesian extension of the classical robust LM-WX test
    (:cite:p:`koley2024UseNot`) using the Dogǎn, Tas̛pınar & Bera (2021)
    framework (:cite:p:`dogan2021BayesianRobust`, Proposition 3) with
    panel-data adjustments (:cite:p:`anselin2008SpatialPanel`,
    :cite:p:`elhorst2014SpatialEconometrics`).

    The alternative model is SAR panel (includes :math:`\rho` but not
    :math:`\gamma`).  For each posterior draw of
    :math:`(\beta, \rho, \sigma^2)` the SAR residual is
    :math:`\mathbf{e}^{(d)} = \mathbf{y} - \rho^{(d)} W_{NT} \mathbf{y}
    - X \beta^{(d)}`.  The raw scores are

    .. math::
        g_\rho^{(d)} = \mathbf{e}^{(d)\,\top} W_{NT} \mathbf{y},
        \qquad
        \mathbf{g}_\gamma^{(d)} = (WX)^\top \mathbf{e}^{(d)}.

    The Neyman-orthogonal adjusted score uses the canonical Schur
    complement of the raw-score variance matrix:

    .. math::
        \mathbf{g}_\gamma^{*\,(d)} &= \mathbf{g}_\gamma^{(d)}
            - V_{\gamma\rho} V_{\rho\rho}^{-1} g_\rho^{(d)}, \\
        V_{\gamma\gamma\,|\,\rho} &= V_{\gamma\gamma}
            - V_{\gamma\rho} V_{\rho\rho}^{-1} V_{\rho\gamma}.

    The per-draw LM statistic is

    .. math::
        \mathrm{LM}^{(d)} = \mathbf{g}_\gamma^{*\,(d)\,\top}
            V_{\gamma\gamma\,|\,\rho}^{-1} \mathbf{g}_\gamma^{*\,(d)}
        \;\xrightarrow{d}\; \chi^2_{k_{wx}} \quad \text{under } H_0.

    Parameters
    ----------
    model : SARPanelFE or SARPanelRE
        Fitted SAR panel model with ``inference_data`` containing
        posterior draws for ``beta``, ``rho``, ``sigma``.

    Returns
    -------
    BayesianLMTestResult
        Per-draw LM samples, summary statistics and ``df = k_{wx}``.
    """
    X = model._X
    WX = model._WX
    Wy = model._Wy
    W_sp = model._W_sparse
    k_wx = WX.shape[1]
    N = model._N
    T = model._T

    if k_wx == 0:
        raise ValueError(
            "Model has no WX columns. The robust panel LM-WX test requires "
            "at least one spatially lagged covariate."
        )

    idata = model.inference_data
    beta_draws = _get_posterior_draws(idata, "beta")  # (draws, k)
    rho_draws = _get_posterior_draws(idata, "rho")  # (draws,)

    resid = _compute_residuals(model, beta_draws, rho_draws=rho_draws)

    g_rho = np.dot(resid, Wy)  # (draws,)
    g_gamma = resid @ WX  # (draws, k_wx)

    # Raw-score variance blocks at posterior-mean theta*.
    beta_mean = np.mean(beta_draws, axis=0)
    rho_mean = float(np.mean(rho_draws))
    _, sigma2_mean = _posterior_mean_sigma2(idata)
    X_beta = _resolve_X_for_beta(model, beta_draws)
    y_hat = rho_mean * Wy + X_beta @ beta_mean
    Wy_hat = _panel_spatial_lag(W_sp, y_hat, N, T)

    info = _panel_info_matrix_blocks(
        X,
        WX,
        W_sp,
        model._W_eigs,
        sigma2_mean,
        N,
        T,
        Wy_hat=Wy_hat,
        T_ww=model._T_ww,
    )

    V_rho_rho = info["J_rho_rho"]
    V_rho_gamma = info["J_rho_gamma"]  # (k_wx,)
    V_gamma_gamma = info["J_gamma_gamma"]  # (k_wx, k_wx)

    # Neyman: g_gamma* = g_gamma - V_{gamma,rho} V_{rho,rho}^{-1} g_rho
    g_gamma_star, V_gamma_given_rho = _neyman_adjust_vector(
        g_gamma,
        g_rho,
        V_gamma_gamma,
        V_rho_gamma,
        V_rho_rho,
        label="V_gamma_given_rho (panel robust LM-WX)",
    )

    return _lm_vector(
        g_gamma_star,
        V_gamma_given_rho,
        test_type="bayesian_panel_robust_lm_wx",
        df=k_wx,
        details={"k_wx": k_wx, "N": N, "T": T},
        label="V_gamma_given_rho (panel robust LM-WX)",
    )


def bayesian_panel_robust_lm_error_sdem_test(
    model,
) -> BayesianLMTestResult:
    r"""Bayesian panel robust LM-Error in SDEM context (H₀: λ = 0 | SLX panel).

    Tests :math:`H_0: \lambda = 0` using the **SLX panel as the
    restricted null**.  The SLX normal equations zero the γ-direction
    score :math:`(WX)^\top \mathbf{e}_{slx} = \mathbf{0}` exactly, but
    leave the *other* spatial parameter :math:`\rho` as a non-trivial
    nuisance.  Following :cite:p:`dogan2021BayesianRobust` (Proposition
    3) we apply the Schur-complement Neyman correction with respect to
    ρ:

    .. math::
        g_\lambda^{*\,(d)}    &= g_\lambda^{(d)}
            - (J_{\rho\lambda}/J_{\rho\rho})\, g_\rho^{(d)}, \\
        V_{\lambda|\rho}      &= J_{\lambda\lambda} - J_{\rho\lambda}^2 / J_{\rho\rho}, \\
        \mathrm{LM}^{(d)}     &= (g_\lambda^{*\,(d)})^2 / V_{\lambda|\rho}
        \;\xrightarrow{d}\; \chi^2_1.

    Information blocks (:math:`J_{\rho\rho}, J_{\lambda\lambda},
    J_{\rho\lambda}`) are computed at the posterior means via
    :func:`_info_matrix_blocks_slx_robust` using
    :math:`W_{NT} = I_T \otimes W` and :math:`T \cdot T_{WW}`; see
    :func:`bayesian_panel_robust_lm_lag_sdm_test` for the full math
    block.

    Parameters
    ----------
    model : SLXPanelFE or SLX-like panel model
        Fitted SLX panel model with ``inference_data`` containing
        posterior draws for ``beta`` and ``sigma``.

    Returns
    -------
    BayesianLMTestResult
        Per-draw LM samples, summary statistics and ``df = 1``.
    """
    X = model._X
    WX = model._WX
    W_sp = model._W_sparse
    k_wx = WX.shape[1]
    N = model._N
    T = model._T

    idata = model.inference_data
    beta_draws = _get_posterior_draws(idata, "beta")  # (draws, k_total)

    resid = _compute_residuals(model, beta_draws, use_Z=True)

    # Per-draw raw scores
    We_panel = _panel_spatial_lag(W_sp, resid, N, T)
    g_lambda = np.sum(resid * We_panel, axis=1)  # (draws,)
    g_rho = np.dot(resid, model._Wy)  # (draws,)

    # Information matrix blocks at posterior means using full W_NT.
    beta_mean = np.mean(beta_draws, axis=0)
    _, sigma2_mean = _posterior_mean_sigma2(idata)
    W_NT = sp.kron(sp.identity(T, format="csr"), W_sp, format="csr")
    Z_beta = _resolve_X_for_beta(model, beta_draws)
    blocks = _info_matrix_blocks_slx_robust(
        X=X,
        WX=WX,
        W_sparse=W_NT,
        sigma2=sigma2_mean,
        beta_slx_mean=beta_mean,
        T_ww=T * model._T_ww,
        Z_beta=Z_beta,
    )
    J_rr = blocks["J_rho_rho"]
    J_ll = blocks["J_lam_lam"]
    J_rl = blocks["J_rho_lam"]

    # Schur-complement Neyman correction on ρ
    g_lambda_star, V_lambda_given_rho = _neyman_adjust_scalar(
        g_lambda,
        g_rho[:, None],
        J_ll,
        J_rl,
        J_rr,
        label="V_lambda_given_rho (panel robust LM-Error-SDEM)",
    )

    LM = g_lambda_star**2 / (V_lambda_given_rho + 1e-12)

    return _finalize_lm(
        LM,
        test_type="bayesian_panel_robust_lm_error_sdem",
        df=1,
        details={
            "k_wx": k_wx,
            "N": N,
            "T": T,
            "J_rho_rho": J_rr,
            "J_lam_lam": J_ll,
            "J_rho_lam": J_rl,
        },
    )


# =====================================================================
# Bayesian LM diagnostics for spatial flow (origin-destination) models
# =====================================================================
#
# These tests extend the Dogan et al. (2021) Bayesian LM framework to the
# OD-flow filter A = I_N - rho_d W_d - rho_o W_o - rho_w W_w used by the
# SARFlow family.  The score blocks come from the conditional Gaussian
# log-likelihood of an OLSFlow null model; the information matrix uses the
# cached Kronecker-trace block ``model._T_flow_traces`` (see
# :func:`neighbayes.graph.flow_trace_blocks`).  Robust variants pull a
# fitted SARFlow as the unrestricted alternative and apply the
# Neyman-orthogonal score adjustment.
#
# Conventions:
#   - Score for spatial-lag direction i evaluated at OLSFlow draws:
#         g_i^{(g)} = (W_i y)^T e^{(g)} ,    e^{(g)} = y - X beta^{(g)}
#   - Information matrix J = T * T_flow_traces * sigma2_bar + Q,
#     with Q[i,j] = (W_i y)^T (W_j y).
#   - chi^2 reference at the posterior-mean LM, df given per test.

from .flow import _flow_score_info as _flow_panel_score_info


def _flow_panel_marginal_lm(model, key: str, test_type: str) -> BayesianLMTestResult:
    G, J = _flow_panel_score_info(model, restrict_keys=(key,))
    s = G[:, 0]
    v = float(J[0, 0])
    LM = s**2 / (v + 1e-12)
    return _finalize_lm(
        LM,
        test_type=test_type,
        df=1,
        details={"direction": key, "T": int(getattr(model, "_T", 1))},
    )


def bayesian_panel_lm_flow_destination_test(model) -> BayesianLMTestResult:
    r"""Panel analogue of :func:`bayesian_lm_flow_destination_test` operating on a
    fitted :class:`~neighbayes.models.flow_panel.OLSFlowPanelFE`.

    The score is summed over the demeaned panel stack
    (length :math:`n^{2}\cdot T`); the information matrix scales the
    Kronecker trace block by :math:`T` to reflect i.i.d. within-period
    contributions under :math:`H_0`. Spatial-flow gravity construction
    follows :cite:t:`lesage2008SpatialEconometric`; panel-data treatment
    follows :cite:t:`elhorst2014SpatialEconometrics` and
    :cite:t:`anselin2008SpatialPanel`. The Bayesian LM statistic is
    computed per posterior draw following :cite:t:`dogan2021BayesianRobust`."""
    return _flow_panel_marginal_lm(model, "d", "bayesian_panel_lm_flow_destination")


def bayesian_panel_lm_flow_origin_test(model) -> BayesianLMTestResult:
    r"""Panel analogue of :func:`bayesian_lm_flow_origin_test`
    (:math:`H_0\colon \rho_o = 0`). See
    :func:`bayesian_panel_lm_flow_destination_test` for details and references
    (:cite:p:`lesage2008SpatialEconometric`,
    :cite:p:`elhorst2014SpatialEconometrics`,
    :cite:p:`dogan2021BayesianRobust`)."""
    return _flow_panel_marginal_lm(model, "o", "bayesian_panel_lm_flow_origin")


def bayesian_panel_lm_flow_od_test(model) -> BayesianLMTestResult:
    r"""Panel analogue of :func:`bayesian_lm_flow_od_test`
    (:math:`H_0\colon \rho_w = 0`). See
    :func:`bayesian_panel_lm_flow_destination_test` for details and references
    (:cite:p:`lesage2008SpatialEconometric`,
    :cite:p:`elhorst2014SpatialEconometrics`,
    :cite:p:`dogan2021BayesianRobust`)."""
    return _flow_panel_marginal_lm(model, "w", "bayesian_panel_lm_flow_od")


def bayesian_panel_lm_flow_joint_test(model) -> BayesianLMTestResult:
    r"""Panel analogue of :func:`bayesian_lm_flow_joint_test` on
    :class:`OLSFlowPanelFE`. The joint score over
    :math:`(\rho_d, \rho_o, \rho_w)` is :math:`\chi^2_3` under :math:`H_0`.
    See :func:`bayesian_panel_lm_flow_destination_test` for additional details
    and references (:cite:p:`lesage2008SpatialEconometric`,
    :cite:p:`elhorst2014SpatialEconometrics`,
    :cite:p:`dogan2021BayesianRobust`)."""
    G, J = _flow_panel_score_info(model, restrict_keys=("d", "o", "w"))
    J_inv = _safe_inv(J, "J (panel flow joint LM)")
    LM = np.einsum("di,ij,dj->d", G, J_inv, G)
    return _finalize_lm(
        LM,
        test_type="bayesian_panel_lm_flow_joint",
        df=3,
        details={"T": int(getattr(model, "_T", 1))},
    )


def bayesian_panel_lm_flow_intraunit_test(model) -> BayesianLMTestResult:
    r"""Panel analogue of :func:`bayesian_lm_flow_intraunit_test` on
    :class:`OLSFlowPanelFE`. Operates on the demeaned panel design.
    Score :math:`g = X_{\text{intra}}^\top e` and information
    :math:`V = X_{\text{intra}}^\top X_{\text{intra}} / \bar\sigma^2`,
    distributed :math:`\chi^2_{k_{\text{intra}}}` under :math:`H_0`.
    See :func:`bayesian_lm_flow_intraunit_test` for the cross-sectional
    construction (:cite:p:`lesage2008SpatialEconometric`,
    :cite:p:`elhorst2014SpatialEconometrics`,
    :cite:p:`dogan2021BayesianRobust`)."""
    intra_idx = getattr(model, "_intra_idx", None)
    if intra_idx is None or len(intra_idx) == 0:
        raise ValueError(
            "Panel model has no intra columns to test.  Build the design "
            "with flow_design_matrix(..., include_intra=True)."
        )
    intra_idx = np.asarray(intra_idx, dtype=int)

    np.asarray(model._y, dtype=np.float64)
    X = np.asarray(model._X, dtype=np.float64)
    Xi = X[:, intra_idx]
    k_intra = Xi.shape[1]

    idata = model.inference_data
    beta_draws = _get_posterior_draws(idata, "beta")

    resid = _compute_residuals(model, beta_draws)
    G = resid @ Xi

    _, sigma2_mean = _posterior_mean_sigma2(idata)
    V = (Xi.T @ Xi) / sigma2_mean

    return _lm_vector(
        G,
        V,
        test_type="bayesian_panel_lm_flow_intraunit",
        df=k_intra,
        details={"k_intra": k_intra, "T": int(getattr(model, "_T", 1))},
        label="V_intra (panel flow intra LM)",
    )


# ---------------------------------------------------------------------------
# Cross-sectional robust LM tests (OLS-null, Anselin et al. 1996)
# ---------------------------------------------------------------------------


def bayesian_panel_lm_error_sdm_test(
    model,
) -> BayesianLMTestResult:
    r"""Panel LM-Error test from an SDM panel posterior (H₀: λ = 0 | SDM).

    Panel analogue of :func:`bayesian_lm_error_sdm_test`. Residuals are
    computed from the SDM panel mean structure, including
    :math:`\rho\,(I_T \otimes W)\mathbf{y}` and (for RE) the unit-level
    random effect:

    .. math::
        \mathbf{e} = \mathbf{y} - \rho\,(I_T\otimes W)\mathbf{y}
                     - X\beta - WX\gamma - (\iota_T \otimes \alpha).

    The score and variance follow the panel LM-Error construction
    (cf. :func:`bayesian_panel_lm_error_test`), kept on the raw-score
    scale:

    .. math::
        S = \mathbf{e}^\top (I_T \otimes W)\mathbf{e},
        \qquad V = \bar{\sigma}^4 \cdot T \cdot T_{WW},

    with :math:`T_{WW} = \mathrm{tr}(W^\top W + W^2)`. The LM statistic is
    :math:`\chi^2_1` under H₀. Tests whether an SDM panel should be
    extended to a SDARAR panel (SDM with spatial-error autocorrelation);
    this is the panel analogue of the SDM-aware diagnostic discussed in
    :cite:t:`koley2024UseNot`. Panel-data extensions follow
    :cite:t:`anselin2008SpatialPanel` and :cite:t:`elhorst2014SpatialEconometrics`.
    The Bayesian LM statistic is computed per posterior draw following
    :cite:t:`dogan2021BayesianRobust`.
    """
    W_sp = model._W_sparse
    N = model._N
    T = model._T

    idata = model.inference_data
    beta_draws = _get_posterior_draws(idata, "beta")
    rho_draws = _get_posterior_draws(idata, "rho").reshape(-1)

    resid = _compute_residuals(model, beta_draws, use_Z=True, rho_draws=rho_draws)

    We = _panel_spatial_lag(W_sp, resid, N, T)
    S = np.sum(resid * We, axis=1)

    T_ww = model._T_ww
    _, sigma2_mean = _posterior_mean_sigma2(idata)
    # Panel LM-Error raw-score variance: sigma^4 * T * T_ww
    # (cf. bayesian_panel_lm_error_test).
    V = sigma2_mean**2 * T * T_ww

    return _lm_scalar(
        S,
        V,
        test_type="bayesian_panel_lm_error_sdm",
        df=1,
        details={"N": N, "T": T},
    )


def bayesian_panel_lm_lag_sdem_test(
    model,
) -> BayesianLMTestResult:
    r"""Panel LM-Lag test from an SDEM panel posterior (H₀: ρ = 0 | SDEM).

    Panel analogue of :func:`bayesian_lm_lag_sdem_test`. Residuals are the
    spatially-filtered SDEM panel residuals (with RE alpha removed if
    present):

    .. math::
        \mathbf{u} = \mathbf{y} - X\beta - WX\gamma - (\iota_T\otimes\alpha),
        \qquad \boldsymbol{\varepsilon} = \mathbf{u} - \lambda\,(I_T\otimes W)\mathbf{u}.

    The score and variance follow the SDEM-filtered LM-Lag derivation:
    using the whitened lag vector :math:`\tilde z_\rho = \bar A_\lambda
    (I_T\otimes W)\mathbf{y}` with :math:`\bar A_\lambda = I - \bar\lambda
    (I_T\otimes W)` and the whitened design :math:`\tilde Z = \bar A_\lambda
    [X, WX]`,

    .. math::
        S = \boldsymbol{\varepsilon}^\top \tilde z_\rho, \qquad
        V = \bar{\sigma}^4 \, T \, T_{WW}
            + \bar{\sigma}^2 \, \tilde z_\rho^{\top} M_{\tilde Z}\,
              \tilde z_\rho.

    .. note::
       Earlier revisions used an unwhitened :math:`S = \boldsymbol{
       \varepsilon}^\top (I_T\otimes W)\mathbf{y}` paired with :math:`V =
       \bar\sigma^4 T \, T_{WW} + \bar\sigma^2 \|(I_T\otimes W)
       \mathbf{y}\|^2`, which produced empirical size near 1 on SDEM-DGP
       because both the numerator and the denominator omitted the
       :math:`\bar A_\lambda` whitening factor.

    The LM statistic is :math:`\chi^2_1` under H₀. Tests whether an SDEM
    panel should be extended to a SDARAR panel (SDEM with a spatial lag);
    this is the panel analogue of the SDEM-aware diagnostic discussed in
    :cite:t:`koley2024UseNot`. Panel-data extensions follow
    :cite:t:`anselin2008SpatialPanel` and :cite:t:`elhorst2014SpatialEconometrics`.
    The Bayesian LM statistic is computed per posterior draw following
    :cite:t:`dogan2021BayesianRobust`.
    """
    X = model._X
    WX = model._WX
    Wy = model._Wy
    W_sp = model._W_sparse
    N = model._N
    T = model._T

    idata = model.inference_data
    beta_draws = _get_posterior_draws(idata, "beta")
    lam_draws = _get_posterior_draws(idata, "lam").reshape(-1)

    eps = _compute_residuals(
        model,
        beta_draws,
        use_Z=True,
        lam_draws=lam_draws,
        W_sp=W_sp,
        N=N,
        T=T,
    )

    lam_mean = float(np.mean(lam_draws))
    _, sigma2_mean = _posterior_mean_sigma2(idata)
    Z = np.hstack([X, WX])
    # A_λ @ v = v - λ̄((I_T ⊗ W) @ v) — per-period sparse matvec, no dense matrix
    Z_tilde = Z - lam_mean * _panel_spatial_lag(W_sp, Z.T, N, T).T
    z_rho = Wy - lam_mean * _panel_spatial_lag(W_sp, Wy, N, T)  # whitened lag

    S = np.dot(eps, z_rho)

    T_ww = model._T_ww
    # Raw-score variance: sigma^4 * T * T_ww + sigma^2 * z_rho' M_Z_tilde z_rho.
    V = sigma2_mean**2 * T * T_ww + sigma2_mean * _mx_quadratic(Z_tilde, z_rho)

    return _lm_scalar(
        S,
        V,
        test_type="bayesian_panel_lm_lag_sdem",
        df=1,
        details={"N": N, "T": T},
    )


# ---------------------------------------------------------------------------
# Robust-after-naive Bayesian LM tests for SAR / SEM / SDM / SDEM nulls
# (Anselin–Bera–Florax–Yoon precondition; full Schur at posterior mean)
# ---------------------------------------------------------------------------
