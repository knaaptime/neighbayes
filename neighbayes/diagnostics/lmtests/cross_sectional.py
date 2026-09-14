"""Bayesian LM diagnostic tests — cross-sectional variants.

All public test functions return :class:`BayesianLMTestResult`.
"""

import numpy as np
import scipy.sparse as _sp

from neighbayes._ops._backend import _select_sparse_backend, _sparse_factor
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
    _posterior_mean_sigma2,
    _resolve_lam_name,
    _resolve_X_for_beta,
    _safe_inv,
    _trace_WtW_WW,
)


def bayesian_lm_wx_sem_test(
    model,
) -> "BayesianLMTestResult":
    r"""Bayesian LM test for WX coefficients in SEM (H₀: γ = 0 | SEM).

    Tests whether spatially lagged covariates (WX) should be added to a
    SEM model — i.e. whether SEM should be extended to SDEM.  Bayesian
    extension of the classical LM-WX test
    (:cite:p:`koley2024UseNot`) using the Doğan, Taşpınar & Bera (2021)
    framework (:cite:p:`dogan2021BayesianRobust`, Proposition 1).

    The null model is SEM (includes :math:`\lambda` but not
    :math:`\gamma`).  For each posterior draw of
    :math:`(\beta, \lambda, \sigma^2)` the raw score is

    .. math::
        \mathbf{g}_\gamma^{(d)} = (WX)^\top \mathbf{e}^{(d)},
        \qquad \mathbf{e}^{(d)} = \mathbf{y} - X \beta^{(d)}.

    Under :math:`H_0` the variance of the raw score is the same Schur-
    complemented quantity used by spreg's ``lm_wx``
    (:cite:p:`koley2024UseNot`):

    .. math::
        V_{\gamma\gamma} = \bar{\sigma}^2 \, (WX)^\top M_X (WX),

    where :math:`M_X = I - X(X^\top X)^{-1} X^\top` and
    :math:`\bar{\sigma}^2` is the posterior mean of :math:`\sigma^2`.
    The per-draw LM statistic is

    .. math::
        \mathrm{LM}^{(d)} = \mathbf{g}_\gamma^{(d)\,\top}
            V_{\gamma\gamma}^{-1} \mathbf{g}_\gamma^{(d)}
        \;\xrightarrow{d}\; \chi^2_{k_{wx}} \quad \text{under } H_0.

    Parameters
    ----------
    model : SEM
        Fitted SEM model with ``inference_data`` containing posterior
        draws of ``beta``, ``lambda``, ``sigma``.

    Returns
    -------
    BayesianLMTestResult
        Per-draw LM samples, summary statistics, ``df = k_{wx}`` and
        metadata.
    """
    X = model._X
    WX = model._WX
    k_wx = WX.shape[1]

    if k_wx == 0:
        raise ValueError(
            "Model has no WX columns. The WX test requires at least one "
            "spatially lagged covariate. Ensure the model was constructed "
            "with a W matrix and w_vars."
        )

    idata = model.inference_data
    beta_draws = _get_posterior_draws(idata, "beta")  # (draws, k)

    resid = _compute_residuals(model, beta_draws)

    # Per-draw raw score
    g_gamma = resid @ WX  # (draws, k_wx)

    # M_X-projected raw-score variance (Koley-Bera 2024).
    _, sigma2_mean = _posterior_mean_sigma2(idata)
    V_gamma_gamma = sigma2_mean * _mx_cross(X, WX, WX)

    return _lm_vector(
        g_gamma,
        V_gamma_gamma,
        test_type="bayesian_lm_wx_sem",
        df=k_wx,
        details={"k_wx": k_wx},
        label="V_gamma_gamma (LM-WX-SEM)",
    )


def bayesian_lm_lag_test(
    model,
) -> BayesianLMTestResult:
    r"""Bayesian LM test for omitted spatial lag (SAR) model.

    Bayesian extension of the classical LM-Lag test
    (:cite:p:`anselin1996SimpleDiagnostic`, eq. 13) using the Doğan,
    Taşpınar & Bera (2021) quadratic-net-loss framework
    (:cite:p:`dogan2021BayesianRobust`, Proposition 1):

    1. Score :math:`s_\rho(\theta)` evaluated at every posterior draw of
       :math:`(\beta, \sigma^2)` from the OLS null fit.
    2. Concentration matrix :math:`C_{\rho\rho \cdot \beta}(\theta^\star)`
       evaluated at the posterior mean :math:`\theta^\star` (Doğan eq. 3.6).
    3. Posterior averaging of the per-draw quadratic form yields the
       Bayesian LM statistic with asymptotic :math:`\chi^2_1` reference.

    For each posterior draw the raw score is

    .. math::
        S^{(d)} = \mathbf{e}^{(d)\,\top} W \mathbf{y},
        \qquad \mathbf{e}^{(d)} = \mathbf{y} - X \beta^{(d)}.

    Concentrating :math:`\beta` out of the SAR Fisher information
    (:cite:p:`anselin1996SimpleDiagnostic`, eq. 13) gives the variance of
    the raw score under :math:`H_0`:

    .. math::
        V = \bar{\sigma}^4 \, T_{WW}
            + \bar{\sigma}^2 \, \| M_X \, W X \bar{\beta} \|^2,

    where :math:`T_{WW} = \mathrm{tr}(W^\top W + W^2)`,
    :math:`M_X = I - X(X^\top X)^{-1} X^\top` is the OLS annihilator,
    :math:`\bar{\sigma}^2` is the posterior mean of :math:`\sigma^2`, and
    :math:`\bar{\beta}` is the posterior mean of :math:`\beta`.  The
    projected term :math:`\| M_X W X \bar{\beta} \|^2` is the same
    Schur-complement quantity that appears in spreg's ``lmLag`` denominator
    (Anselin 1996 derivation), evaluated at the posterior-mean
    :math:`\beta` rather than the OLS estimate.

    The per-draw LM statistic is

    .. math::
        \mathrm{LM}^{(d)} = \frac{\bigl(S^{(d)}\bigr)^2}{V}
        \;\xrightarrow{d}\; \chi^2_1
        \quad \text{under } H_0,

    and the Bayesian p-value is computed at the posterior-mean LM
    (:cite:p:`dogan2021BayesianRobust`, eq. 3.7).

    Parameters
    ----------
    model : SpatialModel
        Fitted OLS-like model with ``inference_data`` attribute providing
        posterior draws of ``beta`` and ``sigma``, plus the cached
        ``_y``, ``_X``, ``_Wy``, ``_T_ww`` attributes.

    Returns
    -------
    BayesianLMTestResult
        Per-draw LM samples, summary statistics, ``df=1`` and metadata.

    """
    idata = model.inference_data
    beta_draws = _get_posterior_draws(idata, "beta")  # (draws, k)
    Wy = model._Wy
    _, sigma2_mean = _posterior_mean_sigma2(idata)
    T_ww = model._T_ww

    resid = _compute_residuals(model, beta_draws, use_Z=True)
    # Per-draw raw score s_rho^(d) = e^(d)' W y
    S = np.dot(resid, Wy)  # (draws,)

    # Concentration matrix evaluated at theta*: V = sigma^4 * T_ww +
    # sigma^2 * ||M_X (W X beta_bar)||^2  (Anselin 1996, eq. 13).
    X = _resolve_X_for_beta(model, beta_draws)
    beta_mean = np.mean(beta_draws, axis=0)
    Wy_hat = np.asarray(model._W_sparse @ (X @ beta_mean)).ravel()
    proj_norm_sq = _mx_quadratic(X, Wy_hat)
    V = sigma2_mean**2 * T_ww + sigma2_mean * proj_norm_sq

    return _lm_scalar(S, V, test_type="bayesian_lm_lag", df=1)


def bayesian_lm_error_test(
    model,
) -> BayesianLMTestResult:
    r"""Bayesian LM test for omitted spatial error (SEM) model.

    Bayesian extension of the classical LM-Error test
    (:cite:p:`anselin1996SimpleDiagnostic`, eq. 9) using the Doğan,
    Taşpınar & Bera (2021) quadratic-net-loss framework
    (:cite:p:`dogan2021BayesianRobust`, Proposition 1).  The score and
    concentration matrix come from the OLS log-likelihood; the spatial
    error parameter :math:`\lambda` is information-orthogonal to
    :math:`\beta` under :math:`H_0` so no Schur projection is needed.

    For each posterior draw the raw score is

    .. math::
        S^{(d)} = \mathbf{e}^{(d)\,\top} W \mathbf{e}^{(d)},
        \qquad \mathbf{e}^{(d)} = \mathbf{y} - X \beta^{(d)}.

    Under :math:`H_0` with spherical errors, the variance of the raw
    score (negative-Hessian block at :math:`\theta^\star`) is

    .. math::
        V = \bar{\sigma}^4 \, T_{WW},
        \qquad T_{WW} = \mathrm{tr}(W^\top W + W^2),

    where :math:`\bar{\sigma}^2` is the posterior mean of :math:`\sigma^2`.
    The per-draw LM statistic is

    .. math::
        \mathrm{LM}^{(d)} = \frac{\bigl(S^{(d)}\bigr)^2}{V}
        \;\xrightarrow{d}\; \chi^2_1 \quad \text{under } H_0,

    and the Bayesian p-value is computed at the posterior-mean LM
    (:cite:p:`dogan2021BayesianRobust`, eq. 3.7).

    Parameters
    ----------
    model : SpatialModel
        Fitted OLS-like model with ``inference_data`` attribute providing
        posterior draws of ``beta`` and ``sigma``, plus the cached
        ``_y``, ``_X``, ``_W_sparse``, ``_T_ww`` attributes.

    Returns
    -------
    BayesianLMTestResult
        Per-draw LM samples, summary statistics, ``df=1`` and metadata.

    """
    W_sp = model._W_sparse
    idata = model.inference_data
    beta_draws = _get_posterior_draws(idata, "beta")  # (draws, k)
    _, sigma2_mean = _posterior_mean_sigma2(idata)
    T_ww = model._T_ww

    resid = _compute_residuals(model, beta_draws, use_Z=True)
    # We = W @ resid via sparse matmul (avoids dense W)
    We = (W_sp @ resid.T).T  # (draws, n)
    # Per-draw raw score s_lambda^(d) = e^(d)' W e^(d)
    S = np.sum(resid * We, axis=1)  # (draws,)
    # Variance of raw score at theta*: sigma^4 * T_ww (Anselin 1996, eq. 9).
    V = sigma2_mean**2 * T_ww
    return _lm_scalar(S, V, test_type="bayesian_lm_error", df=1)


def bayesian_lm_wx_test(
    model,
) -> BayesianLMTestResult:
    r"""Bayesian LM test for WX coefficients (H₀: γ = 0 | SAR).

    Tests whether spatially lagged covariates (WX) should be added to a
    SAR model — i.e. whether SAR should be extended to SDM.  Bayesian
    extension of the classical LM-WX test
    (:cite:p:`koley2024UseNot`, eq. for ``RS_gamma``) using the Doğan,
    Taşpınar & Bera (2021) framework
    (:cite:p:`dogan2021BayesianRobust`, Proposition 1).

    The null model is SAR (includes :math:`\rho` but not :math:`\gamma`).
    For each posterior draw of :math:`(\beta, \rho, \sigma^2)` the raw
    score is

    .. math::
        \mathbf{g}_\gamma^{(d)} = (WX)^\top \mathbf{e}^{(d)},
        \qquad \mathbf{e}^{(d)} = \mathbf{y} - \rho^{(d)} W\mathbf{y}
                                  - X \beta^{(d)}.

    Concentrating :math:`\beta` out of the SDM information matrix gives
    the variance of the raw score under :math:`H_0`:

    .. math::
        V_{\gamma\gamma} = \bar{\sigma}^2 \,
            (WX)^\top M_X (WX), \qquad M_X = I - X(X^\top X)^{-1} X^\top.

    This is the same Schur-complement quantity used by spreg's ``lm_wx``
    (:cite:p:`koley2024UseNot`), evaluated at the posterior mean
    :math:`\bar{\sigma}^2`.

    The per-draw LM statistic is

    .. math::
        \mathrm{LM}^{(d)} = \mathbf{g}_\gamma^{(d)\,\top}
            V_{\gamma\gamma}^{-1} \mathbf{g}_\gamma^{(d)}
        \;\xrightarrow{d}\; \chi^2_{k_{wx}} \quad \text{under } H_0.

    Parameters
    ----------
    model : SAR
        Fitted SAR model with ``inference_data`` attribute providing
        posterior draws of ``beta``, ``rho``, ``sigma`` and the cached
        ``_y``, ``_X``, ``_WX``, ``_Wy`` attributes.

    Returns
    -------
    BayesianLMTestResult
        Per-draw LM samples, summary statistics, and metadata.
        ``df = k_{wx}`` (the number of WX columns).
    """
    X = model._X
    WX = model._WX
    k_wx = WX.shape[1]

    if k_wx == 0:
        raise ValueError(
            "Model has no WX columns. The WX test requires at least one "
            "spatially lagged covariate. Ensure the model was constructed "
            "with a W matrix and w_vars."
        )

    idata = model.inference_data
    beta_draws = _get_posterior_draws(idata, "beta")  # (draws, k)
    rho_draws = _get_posterior_draws(idata, "rho")  # (draws,)

    resid = _compute_residuals(model, beta_draws, rho_draws=rho_draws)

    # Per-draw raw score g_gamma^(d) = (WX)' e^(d)
    g_gamma = resid @ WX  # (draws, k_wx)

    # Variance of raw score at theta*: sigma^2 * (WX)' M_X (WX)
    # (Koley-Bera 2024 Schur complement; spreg's lm_wx uses this matrix.)
    _, sigma2_mean = _posterior_mean_sigma2(idata)
    V_gamma_gamma = sigma2_mean * _mx_cross(X, WX, WX)  # (k_wx, k_wx)

    return _lm_vector(
        g_gamma,
        V_gamma_gamma,
        test_type="bayesian_lm_wx",
        df=k_wx,
        details={"k_wx": k_wx},
        label="V_gamma_gamma (LM-WX)",
    )


def bayesian_lm_sdm_joint_test(
    model,
) -> BayesianLMTestResult:
    r"""Bayesian joint LM test for SDM (H₀: ρ = 0 AND γ = 0 | OLS).

    Bayesian extension of the joint LM-SDM test
    (:cite:p:`koley2024UseNot`, ``lm_spdurbin``) using the Doğan,
    Taşpınar & Bera (2021) framework
    (:cite:p:`dogan2021BayesianRobust`, Proposition 1).

    The null model is OLS.  For each posterior draw of
    :math:`(\beta, \sigma^2)` the joint raw score is

    .. math::
        \mathbf{g}^{(d)} = \begin{pmatrix}
            \mathbf{e}^{(d)\,\top} W \mathbf{y} \\
            (WX)^\top \mathbf{e}^{(d)}
        \end{pmatrix},
        \qquad \mathbf{e}^{(d)} = \mathbf{y} - X \beta^{(d)}.

    Concentrating :math:`\beta` out of the SDM information matrix
    (:cite:p:`koley2024UseNot`) gives the
    :math:`(1 + k_{wx}) \times (1 + k_{wx})` variance matrix of the raw
    score:

    .. math::
        V = \begin{pmatrix}
            \bar{\sigma}^4\, T_{WW}
              + \bar{\sigma}^2\, \| M_X W X \bar{\beta} \|^2
            & \bar{\sigma}^2\, (W X \bar{\beta})^\top M_X (WX) \\
            \bar{\sigma}^2\, (WX)^\top M_X (W X \bar{\beta})
            & \bar{\sigma}^2\, (WX)^\top M_X (WX)
        \end{pmatrix},

    where :math:`M_X = I - X(X^\top X)^{-1} X^\top` is the OLS annihilator
    and :math:`T_{WW} = \mathrm{tr}(W^\top W + W^2)`.  This matches the
    Schur-complemented information matrix in spreg's ``lm_spdurbin``,
    evaluated at the posterior-mean :math:`(\bar{\beta}, \bar{\sigma}^2)`
    rather than the OLS estimate.

    The per-draw LM statistic is

    .. math::
        \mathrm{LM}^{(d)} = \mathbf{g}^{(d)\,\top} V^{-1} \mathbf{g}^{(d)}
        \;\xrightarrow{d}\; \chi^2_{1 + k_{wx}} \quad \text{under } H_0.

    Parameters
    ----------
    model : SpatialModel
        Fitted OLS-like model with ``inference_data`` attribute providing
        posterior draws of ``beta`` and ``sigma``.

    Returns
    -------
    BayesianLMTestResult
        Per-draw LM samples, summary statistics, ``df = 1 + k_{wx}`` and
        metadata.

    """
    X = model._X
    WX = model._WX
    Wy = model._Wy
    W_sp = model._W_sparse
    k_wx = WX.shape[1]

    idata = model.inference_data
    beta_draws = _get_posterior_draws(idata, "beta")  # (draws, k)

    resid = _compute_residuals(model, beta_draws)

    # Raw score components
    g_rho = np.dot(resid, Wy)  # (draws,)
    g_gamma = resid @ WX  # (draws, k_wx)
    g = np.column_stack([g_rho, g_gamma])  # (draws, 1+k_wx)

    # Concentration matrix at theta* with M_X Schur projection
    # (Koley-Bera 2024 / spreg lm_spdurbin algebra)
    _, sigma2_mean = _posterior_mean_sigma2(idata)
    beta_mean = np.mean(beta_draws, axis=0)
    Wy_hat = np.asarray(W_sp @ (X @ beta_mean)).ravel()
    T_ww = model._T_ww

    p = 1 + k_wx
    V = np.zeros((p, p))
    V[0, 0] = sigma2_mean**2 * T_ww + sigma2_mean * _mx_quadratic(X, Wy_hat)
    if k_wx > 0:
        cross = sigma2_mean * np.asarray(_mx_cross(X, Wy_hat, WX)).ravel()
        V[0, 1:] = cross
        V[1:, 0] = cross
        V[1:, 1:] = sigma2_mean * _mx_cross(X, WX, WX)

    return _lm_vector(
        g,
        V,
        test_type="bayesian_lm_sdm_joint",
        df=p,
        details={"k_wx": k_wx},
        label="V (SDM joint)",
    )


def bayesian_lm_slx_error_joint_test(
    model,
) -> BayesianLMTestResult:
    r"""Bayesian joint LM test for SDEM (H₀: λ = 0 AND γ = 0 | OLS).

    Bayesian extension of the joint LM-SLX-Error test
    (:cite:p:`koley2024UseNot`, ``lm_slxerr``) using the Doğan,
    Taşpınar & Bera (2021) framework
    (:cite:p:`dogan2021BayesianRobust`, Proposition 1).

    The null model is OLS.  For each posterior draw of
    :math:`(\beta, \sigma^2)` the joint raw score is

    .. math::
        \mathbf{g}^{(d)} = \begin{pmatrix}
            \mathbf{e}^{(d)\,\top} W \mathbf{e}^{(d)} \\
            (WX)^\top \mathbf{e}^{(d)}
        \end{pmatrix},
        \qquad \mathbf{e}^{(d)} = \mathbf{y} - X \beta^{(d)}.

    Under :math:`H_0` with spherical errors,
    :math:`\mathrm{Cov}(\mathbf{e}^\top W \mathbf{e},\ (WX)^\top
    \mathbf{e}) = 0` (third moments of normal errors vanish), so the
    information matrix is block-diagonal — matching spreg's
    ``lm_slxerr`` which simply adds ``LM_Error + LM_WX``
    (:cite:p:`koley2024UseNot`).  The variance of the raw score is

    .. math::
        V = \begin{pmatrix}
            \bar{\sigma}^4\, T_{WW} & 0 \\
            0 & \bar{\sigma}^2\, (WX)^\top M_X (WX)
        \end{pmatrix},

    where :math:`M_X = I - X(X^\top X)^{-1} X^\top` and
    :math:`T_{WW} = \mathrm{tr}(W^\top W + W^2)`.  The per-draw LM
    statistic is

    .. math::
        \mathrm{LM}^{(d)} = \mathbf{g}^{(d)\,\top} V^{-1} \mathbf{g}^{(d)}
        \;\xrightarrow{d}\; \chi^2_{1 + k_{wx}} \quad \text{under } H_0.

    Parameters
    ----------
    model : SpatialModel
        Fitted OLS-like model with ``inference_data`` attribute providing
        posterior draws of ``beta`` and ``sigma``.

    Returns
    -------
    BayesianLMTestResult
        Per-draw LM samples, summary statistics, ``df = 1 + k_{wx}`` and
        metadata.

    """
    X = model._X
    WX = model._WX
    W_sp = model._W_sparse
    k_wx = WX.shape[1]

    idata = model.inference_data
    beta_draws = _get_posterior_draws(idata, "beta")  # (draws, k)

    resid = _compute_residuals(model, beta_draws)

    # Raw score components
    We = (W_sp @ resid.T).T  # (draws, n)
    g_lambda = np.sum(resid * We, axis=1)  # (draws,)
    g_gamma = resid @ WX  # (draws, k_wx)
    g = np.column_stack([g_lambda, g_gamma])  # (draws, 1+k_wx)

    _, sigma2_mean = _posterior_mean_sigma2(idata)
    T_ww = model._T_ww

    # Block-diagonal raw-score variance (Koley-Bera 2024).
    p = 1 + k_wx
    V = np.zeros((p, p))
    V[0, 0] = sigma2_mean**2 * T_ww
    if k_wx > 0:
        V[1:, 1:] = sigma2_mean * _mx_cross(X, WX, WX)

    return _lm_vector(
        g,
        V,
        test_type="bayesian_lm_slx_error_joint",
        df=p,
        details={"k_wx": k_wx},
        label="V (SLX-error joint)",
    )


# ---------------------------------------------------------------------------
# Information matrix blocks for Neyman orthogonal score adjustment
# ---------------------------------------------------------------------------


def _info_matrix_blocks_sdm(
    X: np.ndarray,
    WX: np.ndarray,
    W_sparse,
    sigma2: float,
    Wy_hat: np.ndarray | None = None,
    T_ww: float | None = None,
) -> dict:
    r"""Compute raw-score variance blocks for SDM Neyman-orthogonal adjustment.

    Returns the variance blocks of the **raw** scores
    :math:`g_\rho = \mathbf{e}^\top W \mathbf{y}` and
    :math:`\mathbf{g}_\gamma = (WX)^\top \mathbf{e}` evaluated at
    :math:`\theta^\star = (\bar{\beta}, \bar{\sigma}^2)`, with the
    nuisance :math:`\beta` concentrated out via the OLS annihilator
    :math:`M_X = I - X(X^\top X)^{-1} X^\top`
    (:cite:p:`anselin1996SimpleDiagnostic`, eq. 13;
    :cite:p:`koley2024UseNot`, Section 3).

    .. math::
        V_{\rho\rho} &= \bar{\sigma}^4 \, T_{WW}
            + \bar{\sigma}^2 \, \| M_X W X \bar{\beta} \|^2 \\
        V_{\rho\gamma} &= \bar{\sigma}^2 \, (W X \bar{\beta})^\top M_X (WX) \\
        V_{\gamma\gamma} &= \bar{\sigma}^2 \, (WX)^\top M_X (WX)

    where :math:`T_{WW} = \mathrm{tr}(W^\top W + W^2)` and
    :math:`W X \bar{\beta}` are the spatially lagged fitted values under
    :math:`H_0`.  The Neyman-orthogonal adjustment used by
    :func:`bayesian_robust_lm_lag_sdm_test` and
    :func:`bayesian_robust_lm_wx_test`
    (:cite:p:`dogan2021BayesianRobust`, Proposition 3) only depends on
    the **ratios** :math:`V_{\rho\gamma} V_{\gamma\gamma}^{-1}`, which are
    invariant to overall :math:`\sigma^2`-scaling, but the residual
    variance :math:`V_{\rho\rho \cdot \gamma}` must be on the same scale
    as the raw scores — hence the explicit :math:`\sigma^2` factors.

    The dict keys are kept as ``J_rho_rho`` / ``J_rho_gamma`` /
    ``J_gamma_gamma`` for backwards-compatibility.  Their *numerical
    semantics* are the raw-score variance blocks defined above.

    Parameters
    ----------
    X : np.ndarray
        Design matrix of shape ``(n, k)`` including intercept.
    WX : np.ndarray
        Spatially lagged design matrix of shape ``(n, k_wx)``.
    W_sparse : scipy.sparse matrix
        Spatial weights matrix of shape ``(n, n)``.
    sigma2 : float
        Posterior mean of :math:`\sigma^2`.
    Wy_hat : np.ndarray or None, optional
        Spatially lagged fitted values :math:`W X \bar{\beta}` (or
        :math:`W (\rho \mathbf{y} + X\bar{\beta})` for SAR-null contexts).
        If ``None``, the cross-term is set to zero (Neyman adjustment is a
        no-op).
    T_ww : float or None, optional
        Pre-computed :math:`\mathrm{tr}(W^\top W + W^2)`; computed from
        ``W_sparse`` if not supplied.

    Returns
    -------
    dict
        Keys ``J_rho_rho``, ``J_rho_gamma`` (vector of length ``k_wx``),
        ``J_gamma_gamma`` (``k_wx`` x ``k_wx``), ``T_ww``.
    """
    k_wx = WX.shape[1]

    # T_WW = tr(W'W + W²) = ||W||_F^2 + sum(W ⊙ W')
    # Callers should pass the pre-computed model._T_ww to avoid redundant work.
    if T_ww is None:
        import warnings

        warnings.warn(
            "T_ww not provided to _info_matrix_blocks_sdm; computing from W_sparse. "
            "Pass model._T_ww for efficiency.",
            stacklevel=2,
        )
        T_ww = _trace_WtW_WW(W_sparse)

    # V_{γγ} = σ² · (WX)' M_X (WX)  -- M_X-projected, raw-score scale.
    V_gamma_gamma = sigma2 * _mx_cross(X, WX, WX)

    # V_{ρρ} = σ⁴·T_WW + σ²·||M_X W X β̄||²; cross-term = σ²·(W X β̄)'M_X(WX)
    if Wy_hat is not None:
        V_rho_rho = float(sigma2**2 * T_ww + sigma2 * _mx_quadratic(X, Wy_hat))
        V_rho_gamma = sigma2 * np.asarray(_mx_cross(X, Wy_hat, WX)).ravel()
    else:
        V_rho_rho = float(sigma2**2 * T_ww)
        V_rho_gamma = np.zeros(k_wx)

    return {
        "J_rho_rho": V_rho_rho,
        "J_rho_gamma": V_rho_gamma,
        "J_gamma_gamma": V_gamma_gamma,
        "T_ww": T_ww,
    }


def _info_matrix_blocks_sdem(
    X: np.ndarray,
    WX: np.ndarray,
    W_sparse,
    sigma2: float,
    T_ww: float | None = None,
) -> dict:
    r"""Compute raw-score variance blocks for SDEM Neyman-orthogonal adjustment.

    Returns the variance blocks of the **raw** scores
    :math:`g_\lambda = \mathbf{e}^\top W \mathbf{e}` and
    :math:`\mathbf{g}_\gamma = (WX)^\top \mathbf{e}` under
    :math:`H_0: \lambda = 0` with spherical errors
    (:cite:p:`koley2024UseNot`, Section 3):

    .. math::
        V_{\lambda\lambda} &= \bar{\sigma}^4 \, T_{WW} \\
        V_{\lambda\gamma}  &= 0 \quad \text{(odd normal moments vanish)} \\
        V_{\gamma\gamma}   &= \bar{\sigma}^2 \, (WX)^\top M_X (WX)

    where :math:`M_X = I - X(X^\top X)^{-1} X^\top` and
    :math:`T_{WW} = \mathrm{tr}(W^\top W + W^2)`.  The block-diagonal
    structure mirrors spreg's ``lm_slxerr`` decomposition
    (:cite:p:`koley2024UseNot`).  As with
    :func:`_info_matrix_blocks_sdm`, the dict keys retain the historical
    ``J_*`` names but their numerical semantics are raw-score variance.

    Parameters
    ----------
    X : np.ndarray
        Design matrix of shape ``(n, k)`` including intercept.
    WX : np.ndarray
        Spatially lagged design matrix of shape ``(n, k_wx)``.
    W_sparse : scipy.sparse matrix
        Spatial weights matrix of shape ``(n, n)``.
    sigma2 : float
        Posterior mean of :math:`\sigma^2`.
    T_ww : float or None, optional
        Pre-computed :math:`\mathrm{tr}(W^\top W + W^2)`; computed from
        ``W_sparse`` if not supplied.

    Returns
    -------
    dict
        Keys ``J_lam_lam``, ``J_lam_gamma`` (zero vector of length
        ``k_wx``), ``J_gamma_gamma`` (``k_wx`` x ``k_wx``), ``T_ww``.
    """
    if T_ww is None:
        import warnings

        warnings.warn(
            "T_ww not provided to _info_matrix_blocks_sdem; computing from W_sparse. "
            "Pass model._T_ww for efficiency.",
            stacklevel=2,
        )
        T_ww = _trace_WtW_WW(W_sparse)

    k_wx = WX.shape[1]

    V_lam_lam = float(sigma2**2 * T_ww)
    V_lam_gamma = np.zeros(k_wx)
    V_gamma_gamma = sigma2 * _mx_cross(X, WX, WX)

    return {
        "J_lam_lam": V_lam_lam,
        "J_lam_gamma": V_lam_gamma,
        "J_gamma_gamma": V_gamma_gamma,
        "T_ww": T_ww,
    }


def _info_matrix_blocks_slx_robust(
    X: np.ndarray,
    WX: np.ndarray,
    W_sparse,
    sigma2: float,
    beta_slx_mean: np.ndarray,
    T_ww: float | None = None,
    Z_beta: np.ndarray | None = None,
) -> dict:
    r"""Compute the (ρ, λ) raw-score variance blocks at the SLX null.

    Used by :func:`bayesian_robust_lm_lag_sdm_test` and
    :func:`bayesian_robust_lm_error_sdem_test` (and their panel
    counterparts) to apply the Doğan-Taşpınar-Bera Neyman-orthogonal
    correction (:cite:p:`dogan2021BayesianRobust`, Proposition 3) for
    the **other** spatial parameter.  The SLX OLS normal equations zero
    out the γ-direction exactly, so the only non-trivial nuisance under
    H_0: ρ = λ = 0 in the SDM/SDEM neighborhood is the *opposite*
    spatial parameter.  Without this correction, residuals from the SLX
    null carry an unconcentrated component of the true λ (resp. ρ) and
    bias the score upward — see the n = 1600 SDEM-DGP failure documented
    in ``reference/lm_diagnostics_paper.md`` §3.3.

    With ``e ~ N(0, σ² M_Z)`` under H_0 (where ``M_Z = I - Z(Z'Z)⁻¹Z'``
    and ``Z = [X, WX]``), Isserlis' theorem gives the exact raw-score
    variance blocks:

    .. math::
        J_{\rho\rho}   &= \bar{\sigma}^4 T_{WW}
                        + \bar{\sigma}^2 \, \| M_Z\, W Z \bar{\beta}_{slx} \|^2 \\
        J_{\lambda\lambda} &= \bar{\sigma}^4 T_{WW} \\
        J_{\rho\lambda}     &= \bar{\sigma}^4 \,
            \bigl[\,\mathrm{tr}(M_Z W M_Z W)
                 + \mathrm{tr}(M_Z W M_Z W^\top)\,\bigr]

    where :math:`T_{WW} = \mathrm{tr}(W^\top W + W^2)` is on the same
    dimension as ``W_sparse`` (i.e. the *full* :math:`W_{NT}` for panel
    data).  The cross-term ``J_rho_lam`` collapses to
    :math:`\sigma^4 T_{WW}` only when :math:`M_Z = I` (no β,γ to project
    out); the corrections induced by the M_Z projection are computed
    exactly via trace identities on :math:`Z^\top W Z`,
    :math:`Z^\top W^2 Z` and :math:`Z^\top W^\top W Z` rather than by
    forming the dense :math:`(n,n)` annihilator.

    Parameters
    ----------
    X : np.ndarray
        Design matrix of shape ``(n, k)`` including intercept (cross-
        section) or stacked/demeaned design (panel).
    WX : np.ndarray
        Spatially lagged design matrix of shape ``(n, k_wx)``.  For
        panel data this should already use :math:`W_{NT} = I_T \otimes W`.
    W_sparse : scipy.sparse matrix
        Spatial weights matrix of shape ``(n, n)``.  Caller passes the
        full :math:`W_{NT}` for panel data so that all trace identities
        work on the same dimension as ``X``.
    sigma2 : float
        Posterior mean of :math:`\sigma^2`.
    beta_slx_mean : np.ndarray
        Posterior mean of the SLX coefficient vector covering
        :math:`[X, WX]`, shape ``(k + k_wx,)``.
    T_ww : float or None, optional
        Pre-computed :math:`\mathrm{tr}(W^\top W + W^2)` on the same
        dimension as ``W_sparse``.  Computed if not supplied.  For panel
        callers using ``W_NT = I_T ⊗ W`` and a cross-sectional
        ``model._T_ww``, pass ``T * model._T_ww``.
    Z_beta : np.ndarray or None, optional
        Design matrix matching the dimensions of ``beta_slx_mean``.
        When the intercept column was dropped by Gibbs sampling (FE
        models), this should be ``[X[:, ni], WX]`` where ``ni`` are the
        non-intercept column indices.  If None, defaults to
        ``np.hstack([X, WX])`` (the full design including intercept).

    Returns
    -------
    dict
        Keys ``J_rho_rho``, ``J_lam_lam``, ``J_rho_lam``, ``T_ww``.
    """
    Z = np.hstack([X, WX])
    ZtZ = Z.T @ Z
    # Use pseudo-inverse to handle the (rare) case of collinear WX
    # columns; equivalent to np.linalg.inv when ZtZ is full rank.
    ZtZ_inv = np.linalg.pinv(ZtZ)

    # When the intercept was dropped (Gibbs + FE), Z_beta is the
    # sub-matrix matching the posterior beta dimensions.  Use Z for
    # projection calculations (M_Z) and Z_beta for WZ @ beta_mean.
    if Z_beta is None:
        Z_beta = Z

    if T_ww is None:
        T_ww = _trace_WtW_WW(W_sparse)

    # Trace identities for tr(M_Z A M_Z B) without forming dense M_Z:
    #   tr(M_Z A M_Z B) = tr(AB) - tr(P_Z AB) - tr(P_Z BA) + tr(P_Z A P_Z B)
    #   tr(P_Z C) = tr((Z'Z)^{-1} Z' C Z)
    #   tr(P_Z A P_Z B) = tr( (Z'Z)^{-1}Z'AZ · (Z'Z)^{-1}Z'BZ )
    # For A=W, B=W:  AB = W², BA = W², so tr(P_Z AB)=tr(P_Z BA).
    # For A=W, B=W': AB = W·W', BA = W'·W; their P_Z traces differ but
    #               their sum equals tr(P_Z (WW' + W'W)).
    WZ = np.asarray(W_sparse @ Z)  # (n, k_total)
    W2Z = np.asarray(W_sparse @ WZ)  # W²·Z
    WtWZ = np.asarray(W_sparse.T @ WZ)  # W'·W·Z
    # Note: tr(P_Z W·W') = tr((Z'Z)^{-1} Z' W W' Z) = tr((Z'Z)^{-1} (W'Z)' (W'Z))
    WtZ = np.asarray(W_sparse.T @ Z)  # (n, k_total)

    ZtWZ = Z.T @ WZ
    ZtWtZ = ZtWZ.T

    A_ZtWZ = ZtZ_inv @ ZtWZ
    A_ZtWtZ = ZtZ_inv @ ZtWtZ

    # tr(W²) and tr(W'W) on the W_sparse dimension
    tr_W2 = float(W_sparse.multiply(W_sparse.T).sum())
    tr_WtW = float(W_sparse.power(2).sum())
    T_ww = tr_W2 + tr_WtW  # = _trace_WtW_WW(W_sparse)

    # Case A=W, B=W
    tr_PZ_W2 = float(np.trace(ZtZ_inv @ (Z.T @ W2Z)))
    tr_PZ_W_PZ_W = float(np.trace(A_ZtWZ @ A_ZtWZ))
    tr_MZWMZW = tr_W2 - 2.0 * tr_PZ_W2 + tr_PZ_W_PZ_W

    # Case A=W, B=W'.  tr(P_Z W·W') = tr((Z'Z)^{-1}(W'Z)'(W'Z)) and
    # tr(P_Z W'·W) = tr((Z'Z)^{-1}(WZ)'(WZ)).  Their sum is the
    # symmetric "+ - " contribution.
    tr_PZ_WWt = float(np.trace(ZtZ_inv @ (WtZ.T @ WtZ)))
    tr_PZ_WtW = float(np.trace(ZtZ_inv @ (Z.T @ WtWZ)))
    tr_PZ_W_PZ_Wt = float(np.trace(A_ZtWZ @ A_ZtWtZ))
    tr_MZWMZWt = tr_WtW - tr_PZ_WWt - tr_PZ_WtW + tr_PZ_W_PZ_Wt

    # M_Z W Z β̄  (use Z_beta for the beta multiplication when intercept
    # was dropped, but keep full Z for the M_Z projection).
    WZ_beta = np.asarray(W_sparse @ Z_beta)  # (n, k_beta)
    Wybeta = WZ_beta @ beta_slx_mean  # (n,)
    proj = Z @ (ZtZ_inv @ (Z.T @ Wybeta))
    mz_quad = float(np.dot(Wybeta, Wybeta - proj))

    # Variance of e' W e and e' W y under e ~ N(0, sigma^2 M_Z) (SLX OLS
    # residuals satisfy Z' e = 0 exactly).  By Isserlis on M_Z-projected
    # noise, both Var(e' W e) and the "stochastic" part of Var(e' W y)
    # carry the M_Z projection on BOTH W factors and equal
    #     sigma^4 * [tr(M_Z W M_Z W) + tr(M_Z W M_Z W')].
    # The mean-structure quad term sigma^2 * ||M_Z W Z beta||^2 only
    # contributes to V_rr (since y = Z beta + epsilon under H_0).
    # The unprojected sigma^4 * T_ww form previously used here matches
    # neither the M_Z-projected residuals nor the J_rho_lam cross-block
    # and produces a Schur coefficient < 1, leaking g_lambda noise into
    # the corrected score.
    tr_MzWMzW_pair = tr_MZWMZW + tr_MZWMZWt
    J_lam_lam = float(sigma2**2 * tr_MzWMzW_pair)
    J_rho_rho = float(sigma2**2 * tr_MzWMzW_pair + sigma2 * mz_quad)
    J_rho_lam = float(sigma2**2 * tr_MzWMzW_pair)

    return {
        "J_rho_rho": J_rho_rho,
        "J_lam_lam": J_lam_lam,
        "J_rho_lam": J_rho_lam,
        "T_ww": T_ww,
        # Pre-computed σ-free building blocks so callers can construct
        # per-draw J_* with σ²_d in place of the posterior-mean σ² above.
        # By Isserlis on M_Z-projected normal noise:
        #   J_lam_lam(σ²) = σ⁴ · tr_MzWMzW_pair
        #   J_rho_lam(σ²) = σ⁴ · tr_MzWMzW_pair    (== J_lam_lam by construction)
        #   J_rho_rho(σ²) = σ⁴ · tr_MzWMzW_pair + σ² · mz_quad
        # The Schur identity J_rho_lam == J_lam_lam is exact (it is the
        # algebraic content of "γ already absorbed by SLX OLS"), so the
        # rho-direction Schur cancels to g_rho_star = e_perp' W Z β̂.
        "tr_MzWMzW_pair": tr_MzWMzW_pair,
        "mz_quad": mz_quad,
    }


# ---------------------------------------------------------------------------
# Robust Bayesian LM tests (Neyman orthogonal score, Dogan et al. 2021)
# ---------------------------------------------------------------------------


def bayesian_robust_lm_lag_sdm_test(
    model,
) -> BayesianLMTestResult:
    r"""Bayesian robust LM-Lag test in SDM context (H₀: ρ = 0, robust to γ).

    Tests the null hypothesis that the spatial lag coefficient is zero,
    robust to the local presence of WX effects (γ). Uses the Neyman
    orthogonal score adjustment from :cite:t:`dogan2021BayesianRobust`, Proposition 3,
    which is the Bayesian analogue of the robust LM-Lag-SDM test in
    :cite:t:`koley2024UseNot`.

    The alternative model is SDM (the SLX relaxation that adds
    :math:`\rho`); the null model used to draw posteriors is **SLX**, in
    which :math:`\gamma` is a free parameter and has already been
    absorbed into the residuals.  For each posterior draw of
    :math:`(\beta, \gamma, \sigma^2)` from the SLX fit, residuals are

    .. math::
        \mathbf{e} = \mathbf{y} - X\beta - WX\gamma,

    and the raw scores are

    .. math::
        g_\rho     = \mathbf{e}^\top W \mathbf{y}, \qquad
        g_\lambda  = \mathbf{e}^\top W \mathbf{e}.

    The companion score for :math:`\gamma`,
    :math:`\mathbf{g}_\gamma = (WX)^\top \mathbf{e}`, is identically zero
    by the OLS normal equations of the SLX fit, so the γ-direction of the
    Doğan-Taşpınar-Bera Neyman-orthogonal adjustment
    (:cite:p:`dogan2021BayesianRobust`, Proposition 3) collapses to a
    no-op.  However the SLX null leaves :math:`\lambda` *unconcentrated*:
    when the true DGP is SDEM, :math:`g_\rho` is biased upward by
    :math:`\sigma^2 \, \mathrm{tr}\bigl(M_Z W (I-\lambda W)^{-1}
    (I-\lambda W^\top)^{-1}\bigr)`, which destroys the χ² calibration at
    moderate-to-large :math:`n`.  We therefore Schur-correct on
    :math:`\lambda` as a second nuisance, using the raw-score variance
    blocks at the SLX null supplied by
    :func:`_info_matrix_blocks_slx_robust`:

    .. math::
        g_\rho^* &= g_\rho - \frac{J_{\rho\lambda}}{J_{\lambda\lambda}}\, g_\lambda \\
        V_{\rho \cdot \lambda} &= J_{\rho\rho}
            - \frac{J_{\rho\lambda}^2}{J_{\lambda\lambda}}

    with

    .. math::
        J_{\rho\rho}       &= \bar{\sigma}^4 T_{WW}
                            + \bar{\sigma}^2 \, \| M_Z W Z \bar{\beta}_{slx} \|^2 \\
        J_{\lambda\lambda} &= \bar{\sigma}^4 T_{WW} \\
        J_{\rho\lambda}    &= \bar{\sigma}^4 \,
            \bigl[\,\mathrm{tr}(M_Z W M_Z W) + \mathrm{tr}(M_Z W M_Z W^\top)\,\bigr]

    where :math:`Z = [X, WX]` is the SLX design and
    :math:`M_Z = I - Z(Z^\top Z)^{-1} Z^\top` is the SLX OLS annihilator.
    The per-draw robust LM statistic is

    .. math::
        \mathrm{LM}_R^{(d)} = \frac{\bigl(g_\rho^{*\,(d)}\bigr)^2}
                                   {V_{\rho \cdot \lambda}}
        \;\xrightarrow{d}\; \chi^2_1
        \quad \text{under } H_0,

    independent of local misspecification in either :math:`\gamma` or
    :math:`\lambda`.

    Parameters
    ----------
    model : SLX
        Fitted SLX model with ``inference_data`` containing posterior
        draws of ``beta`` (covering the stacked ``[X, WX]`` design) and
        ``sigma``.

    Returns
    -------
    BayesianLMTestResult
        Per-draw LM samples, summary statistics, ``df = 1`` and metadata.
    """
    y = model._y
    X = model._X
    WX = model._WX
    Wy = model._Wy
    W_sp = model._W_sparse
    k_wx = WX.shape[1]

    idata = model.inference_data
    beta_draws = _get_posterior_draws(idata, "beta")  # (draws, k+k_wx)
    sigma_draws = _get_posterior_draws(idata, "sigma")  # (draws,)

    # M_Z-projected residual.  Algebraically e_perp = M_Z y for every
    # posterior draw of β (since M_Z Z = 0 and e^(d) = y - Z β^(d)),
    # so the score numerator is draw-independent by construction.  This
    # is the SLX analogue of the M_X-projection used in
    # `bayesian_robust_lm_error_sar_test` and is what makes the test
    # correctly sized at large n: the per-draw col(Z) noise of β^(d)
    # that previously inflated the score to 100% rejection on SLX-DGP
    # is killed exactly, while genuine posterior uncertainty enters
    # below through σ²_d in the Schur denominator.
    Z = np.hstack([X, WX])
    ZtZ_inv = np.linalg.pinv(Z.T @ Z)
    e_perp = y - Z @ (ZtZ_inv @ (Z.T @ y))  # (n,) — same for all draws
    g_rho = float(e_perp @ Wy)
    We_perp = np.asarray(W_sp @ e_perp)
    g_lambda = float(e_perp @ We_perp)

    # Information-block scaffolding (σ-free pieces) at the SLX null.
    beta_mean = np.mean(beta_draws, axis=0)
    sigma2_mean = float(np.mean(sigma_draws**2))
    info = _info_matrix_blocks_slx_robust(
        X, WX, W_sp, sigma2_mean, beta_mean, T_ww=model._T_ww
    )
    pair = info["tr_MzWMzW_pair"]
    mz_quad = info["mz_quad"]

    # Per-draw J_* with σ²_d.  J_rho_lam == J_lam_lam exactly (Isserlis
    # on M_Z-projected normal noise), so the Schur coefficient is 1
    # and the rho-direction adjustment collapses to a draw-independent
    # numerator g_rho_star = g_rho - g_lambda = e_perp' W Z β̂.  The
    # Schur denominator V_{ρ·λ} = J_rr - J_rl²/J_ll = σ²_d · ‖M_Z W Z β̂‖²
    # carries the full per-draw posterior uncertainty in σ².
    g_rho_star = g_rho - g_lambda
    V_rho_given_lambda = sigma_draws**2 * mz_quad  # (draws,)

    LM = g_rho_star**2 / (V_rho_given_lambda + 1e-12)

    return _finalize_lm(
        LM,
        test_type="bayesian_robust_lm_lag_sdm",
        df=1,
        details={"k_wx": k_wx, "tr_MzWMzW_pair": pair},
    )


def bayesian_robust_lm_wx_test(
    model,
) -> BayesianLMTestResult:
    r"""Bayesian robust LM-WX test (H₀: γ = 0, robust to ρ).

    Bayesian extension of the robust LM-WX test
    (:cite:p:`koley2024UseNot`, ``rlm_wx``) using the Doğan, Taşpınar &
    Bera (2021) Neyman-orthogonal score adjustment
    (:cite:p:`dogan2021BayesianRobust`, Proposition 3).

    The alternative model is SAR (includes :math:`\rho` but not
    :math:`\gamma`).  For each posterior draw of
    :math:`(\beta, \rho, \sigma^2)` from the SAR fit, residuals are
    :math:`\mathbf{e} = \mathbf{y} - \rho W\mathbf{y} - X\beta` and the
    raw scores are

    .. math::
        g_\rho = \mathbf{e}^\top W \mathbf{y}, \qquad
        \mathbf{g}_\gamma = (WX)^\top \mathbf{e}.

    The Neyman-orthogonal adjusted score for :math:`\gamma` is

    .. math::
        \mathbf{g}_\gamma^* = \mathbf{g}_\gamma
            - \frac{V_{\gamma\rho}}{V_{\rho\rho}}\, g_\rho ,

    with raw-score variance blocks supplied by
    :func:`_info_matrix_blocks_sdm`.  By the standard Schur-complement
    identity (:cite:p:`anselin1996SimpleDiagnostic`, Appendix), the
    variance of :math:`\mathbf{g}_\gamma^*` under :math:`H_0` is

    .. math::
        V_{\gamma \cdot \rho} = V_{\gamma\gamma}
            - \frac{V_{\gamma\rho} V_{\rho\gamma}^\top}{V_{\rho\rho}}.

    The robust LM statistic is therefore

    .. math::
        \mathrm{LM}_R^{(d)} = \mathbf{g}_\gamma^{*\,(d)\,\top}
            V_{\gamma \cdot \rho}^{-1}\, \mathbf{g}_\gamma^{*\,(d)}
        \;\xrightarrow{d}\; \chi^2_{k_{wx}} \quad \text{under } H_0,

    independent of local misspecification in :math:`\rho`.

    Parameters
    ----------
    model : SAR
        Fitted SAR model with ``inference_data`` containing posterior
        draws of ``beta``, ``rho``, ``sigma``.

    Returns
    -------
    BayesianLMTestResult
        Per-draw LM samples, summary statistics, ``df = k_{wx}`` and
        metadata.
    """
    X = model._X
    WX = model._WX
    Wy = model._Wy
    W_sp = model._W_sparse
    k_wx = WX.shape[1]

    if k_wx == 0:
        raise ValueError(
            "Model has no WX columns. The robust LM-WX test requires "
            "at least one spatially lagged covariate."
        )

    idata = model.inference_data
    beta_draws = _get_posterior_draws(idata, "beta")  # (draws, k)
    rho_draws = _get_posterior_draws(idata, "rho")  # (draws,)

    resid = _compute_residuals(model, beta_draws, rho_draws=rho_draws)

    # Raw scores
    g_rho = np.dot(resid, Wy)  # (draws,)
    g_gamma = resid @ WX  # (draws, k_wx)

    # Posterior-mean fitted values for the M_X projection in the info blocks
    beta_mean = np.mean(beta_draws, axis=0)
    rho_mean = float(np.mean(rho_draws))
    _, sigma2_mean = _posterior_mean_sigma2(idata)
    y_hat = rho_mean * Wy + X @ beta_mean
    Wy_hat = np.asarray(W_sp @ y_hat).ravel()

    info = _info_matrix_blocks_sdm(
        X, WX, W_sp, sigma2_mean, Wy_hat=Wy_hat, T_ww=model._T_ww
    )
    V_rho_rho = info["J_rho_rho"]
    V_rho_gamma = info["J_rho_gamma"]  # (k_wx,)
    V_gamma_gamma = info["J_gamma_gamma"]  # (k_wx, k_wx)

    # Neyman: g_gamma* = g_gamma - (V_{gamma,rho} / V_{rho,rho}) g_rho
    g_gamma_star, V_gamma_given_rho = _neyman_adjust_vector(
        g_gamma,
        g_rho,
        V_gamma_gamma,
        V_rho_gamma,
        V_rho_rho,
        label="V_gamma_given_rho (robust LM-WX)",
    )

    return _lm_vector(
        g_gamma_star,
        V_gamma_given_rho,
        test_type="bayesian_robust_lm_wx",
        df=k_wx,
        details={"k_wx": k_wx},
        label="V_gamma_given_rho (robust LM-WX)",
    )


def bayesian_robust_lm_error_sdem_test(
    model,
) -> BayesianLMTestResult:
    r"""Bayesian robust LM-Error test in SDEM context (H₀: λ = 0, robust to γ).

    Bayesian extension of the robust LM-Error test in the SDEM context
    (:cite:p:`koley2024UseNot`) using the Doğan, Taşpınar & Bera (2021)
    framework (:cite:p:`dogan2021BayesianRobust`, Proposition 3).

    The alternative model is SDEM (adds :math:`\lambda`); the null model
    is **SLX**, in which :math:`\gamma` is a free parameter and has
    already been absorbed into the residuals.  For each posterior draw of
    :math:`(\beta, \gamma, \sigma^2)` from the SLX fit, residuals are
    :math:`\mathbf{e} = \mathbf{y} - X\beta - WX\gamma` and the raw
    scores are

    .. math::
        g_\lambda = \mathbf{e}^\top W \mathbf{e}, \qquad
        g_\rho    = \mathbf{e}^\top W \mathbf{y}.

    Under :math:`H_0` with spherical errors the cross-block
    :math:`V_{\lambda\gamma} = 0` (odd normal moments vanish,
    :cite:p:`koley2024UseNot`), so the γ-direction of the
    Neyman-orthogonal adjustment is a no-op.  However the SLX null
    leaves :math:`\rho` unconcentrated: when the true DGP is SDM, the
    error score :math:`g_\lambda` is biased upward.  We therefore
    Schur-correct on :math:`\rho` as a second nuisance, using the
    raw-score variance blocks at the SLX null supplied by
    :func:`_info_matrix_blocks_slx_robust`:

    .. math::
        g_\lambda^* &= g_\lambda - \frac{J_{\rho\lambda}}{J_{\rho\rho}}\, g_\rho \\
        V_{\lambda \cdot \rho} &= J_{\lambda\lambda}
            - \frac{J_{\rho\lambda}^2}{J_{\rho\rho}}.

    The per-draw robust LM statistic is

    .. math::
        \mathrm{LM}_R^{(d)} = \frac{\bigl(g_\lambda^{*\,(d)}\bigr)^2}
                                   {V_{\lambda \cdot \rho}}
        \;\xrightarrow{d}\; \chi^2_1
        \quad \text{under } H_0,

    independent of local misspecification in either :math:`\gamma` or
    :math:`\rho`.

    Parameters
    ----------
    model : SLX
        Fitted SLX model with ``inference_data`` containing posterior
        draws of ``beta`` (covering the stacked ``[X, WX]`` design) and
        ``sigma``.

    Returns
    -------
    BayesianLMTestResult
        Per-draw LM samples, summary statistics, ``df = 1`` and metadata.
    """
    y = model._y
    X = model._X
    WX = model._WX
    Wy = model._Wy
    W_sp = model._W_sparse
    k_wx = WX.shape[1]

    idata = model.inference_data
    beta_draws = _get_posterior_draws(idata, "beta")  # (draws, k+k_wx)
    sigma_draws = _get_posterior_draws(idata, "sigma")  # (draws,)

    # M_Z-projected residual: see bayesian_robust_lm_lag_sdm_test for
    # the algebraic rationale.  The projected residual is e_perp = M_Z y
    # for every posterior draw of β, so the score numerators are
    # draw-independent and per-draw posterior uncertainty enters via
    # σ²_d in the Schur denominator below.
    Z = np.hstack([X, WX])
    ZtZ_inv = np.linalg.pinv(Z.T @ Z)
    e_perp = y - Z @ (ZtZ_inv @ (Z.T @ y))  # (n,)
    We_perp = np.asarray(W_sp @ e_perp)
    g_lambda = float(e_perp @ We_perp)
    g_rho = float(e_perp @ Wy)

    beta_mean = np.mean(beta_draws, axis=0)
    sigma2_mean = float(np.mean(sigma_draws**2))
    info = _info_matrix_blocks_slx_robust(
        X, WX, W_sp, sigma2_mean, beta_mean, T_ww=model._T_ww
    )
    pair = info["tr_MzWMzW_pair"]
    mz_quad = info["mz_quad"]

    # Per-draw J_* with σ²_d.  Unlike the lag-direction test (where
    # J_rl/J_ll = 1 exactly), here the Schur target is ρ as nuisance
    # so coef = J_rl / J_rr = σ⁴·pair / (σ⁴·pair + σ²·mz_quad), which
    # depends on σ²_d.  Both numerator and denominator therefore vary
    # per draw via σ²_d.
    sigma2_d = sigma_draws**2  # (draws,)
    J_rr_d = sigma2_d**2 * pair + sigma2_d * mz_quad  # (draws,)
    J_ll_d = sigma2_d**2 * pair  # (draws,)
    J_rl_d = sigma2_d**2 * pair  # (draws,)

    coef_d = J_rl_d / (J_rr_d + 1e-12)
    g_lambda_star = g_lambda - coef_d * g_rho  # (draws,)
    V_lambda_given_rho = J_ll_d - J_rl_d**2 / (J_rr_d + 1e-12)  # (draws,)

    LM = g_lambda_star**2 / (V_lambda_given_rho + 1e-12)

    return _finalize_lm(
        LM,
        test_type="bayesian_robust_lm_error_sdem",
        df=1,
        details={"k_wx": k_wx, "tr_MzWMzW_pair": pair},
    )


# ---------------------------------------------------------------------------
# Panel helpers
# ---------------------------------------------------------------------------


def _ols_lag_information(
    model,
    beta_mean: np.ndarray,
    sigma2_mean: float,
) -> float:
    r"""Anselin (1996) cross-sectional information for ρ in SAR | OLS.

    Computes :math:`J_{\rho\rho} = (W X \hat\beta)^\top M (W X \hat\beta)
    + T_{WW}\,\sigma^2`, where :math:`M = I - X(X^\top X)^{-1}X^\top`
    and :math:`T_{WW} = \mathrm{tr}(W^\top W + W^2)`.
    """
    X = model._X
    W_sp = model._W_sparse
    T_ww = model._T_ww
    y_hat = X @ beta_mean
    Wy_hat = np.asarray(W_sp @ y_hat).ravel()
    XtX_inv = _safe_inv(X.T @ X, "X'X (cross-sectional robust LM-Lag)")
    M_Wy = Wy_hat - X @ (XtX_inv @ (X.T @ Wy_hat))
    WbMWb = float(Wy_hat @ M_Wy)
    return WbMWb + T_ww * sigma2_mean


def bayesian_robust_lm_lag_test(
    model,
) -> BayesianLMTestResult:
    r"""Bayesian robust LM-Lag test (H₀: ρ = 0, robust to local λ).

    Cross-sectional analogue of :func:`bayesian_panel_robust_lm_lag_test`.
    Tests whether the spatial-lag coefficient is zero, robust to the local
    presence of spatial-error autocorrelation. The null model is OLS.

    For each posterior draw,

    .. math::
        \mathrm{LM}_R = \frac{\bigl(g_\rho/\sigma^2 - g_\lambda/\sigma^2\bigr)^2}
                              {J_{\rho\rho}/\sigma^2 - T_{WW}}

    where :math:`g_\rho = \mathbf{e}^\top W\mathbf{y}`,
    :math:`g_\lambda = \mathbf{e}^\top W\mathbf{e}`,
    :math:`J_{\rho\rho} = (WX\hat\beta)^\top M (WX\hat\beta) + T_{WW}\bar\sigma^2`,
    and :math:`T_{WW} = \mathrm{tr}(W^\top W + W^2)`. Distributed as
    :math:`\chi^2_1` under H₀. The robust adjustment follows the
    Anselin–Bera–Florax–Yoon construction
    (:cite:p:`anselin1996SimpleDiagnostic`) derived from the locally-misspecified
    Lagrange-multiplier framework of :cite:t:`bera1993SpecificationTesting`.
    The Bayesian LM statistic is computed per posterior draw following
    :cite:t:`dogan2021BayesianRobust`.

    Parameters
    ----------
    model : SpatialModel
        Fitted OLS-style model with ``inference_data`` containing posterior
        draws for ``beta`` and ``sigma``.

    Returns
    -------
    BayesianLMTestResult
        Dataclass containing LM samples, summary statistics, and metadata.
    """
    y = model._y
    X = model._X
    Wy = model._Wy
    W_sp = model._W_sparse
    T_ww = model._T_ww

    idata = model.inference_data
    beta_draws = _get_posterior_draws(idata, "beta")
    sigma_draws = _get_posterior_draws(idata, "sigma")

    # Use M_X-projected residual e_perp = M_X y (fixed across draws). The
    # ABFY robust score is evaluated at the OLS estimator, where residuals
    # are X-orthogonal by construction. Replacing e^(d) = y - X beta^(d)
    # with M_X y keeps the score invariant to beta posterior draws -- the
    # information-orthogonality of beta to (rho, lambda) under H_0 means
    # beta-posterior variance does not enter the LM reference distribution.
    XtX_inv = _safe_inv(X.T @ X, "X'X (cross-sectional robust LM-Lag)")
    e_perp = y - X @ (XtX_inv @ (X.T @ y))  # = M_X y, shape (n,)
    We_perp = np.asarray(W_sp @ e_perp).ravel()  # = W M_X y
    S_lag = float(e_perp @ Wy)  # scalar (constant across draws)
    S_err = float(e_perp @ We_perp)  # scalar (constant across draws)

    sigma2_draws = sigma_draws**2
    sigma2_mean = float(np.mean(sigma2_draws))

    beta_mean = np.mean(beta_draws, axis=0)
    beta_mean_x = beta_mean[: X.shape[1]]
    J_val = _ols_lag_information(model, beta_mean_x, sigma2_mean)
    denom = J_val / sigma2_mean - T_ww

    robust_score = (S_lag - S_err) / sigma2_draws
    LM = robust_score**2 / (abs(denom) + 1e-12)

    return _finalize_lm(LM, test_type="bayesian_robust_lm_lag", df=1)


def bayesian_robust_lm_error_test(
    model,
) -> BayesianLMTestResult:
    r"""Bayesian robust LM-Error test (H₀: λ = 0, robust to local ρ).

    Cross-sectional analogue of :func:`bayesian_panel_robust_lm_error_test`.
    Tests whether the spatial-error coefficient is zero, robust to the
    local presence of a spatial lag. The null model is OLS.

    For each posterior draw,

    .. math::
        \mathrm{LM}_R = \frac{\bigl(g_\lambda/\sigma^2 -
            (T_{WW}/J^*_{\rho\rho})\,g_\rho/\sigma^2\bigr)^2}
                              {T_{WW}\bigl(1 - T_{WW}/J^*_{\rho\rho}\bigr)}

    where :math:`J^*_{\rho\rho} = J_{\rho\rho}/\sigma^2` and the remaining
    quantities are as in :func:`bayesian_robust_lm_lag_test`. Distributed
    as :math:`\chi^2_1` under H₀ following the Anselin–Bera–Florax–Yoon
    locally-robust construction (:cite:p:`anselin1996SimpleDiagnostic`,
    :cite:p:`bera1993SpecificationTesting`). The Bayesian LM statistic is
    computed per posterior draw following :cite:t:`dogan2021BayesianRobust`.

    Parameters
    ----------
    model : SpatialModel
        Fitted OLS-style model with ``inference_data`` containing posterior
        draws for ``beta`` and ``sigma``.

    Returns
    -------
    BayesianLMTestResult
        Dataclass containing LM samples, summary statistics, and metadata.
    """
    y = model._y
    X = model._X
    Wy = model._Wy
    W_sp = model._W_sparse
    T_ww = model._T_ww

    idata = model.inference_data
    beta_draws = _get_posterior_draws(idata, "beta")
    sigma_draws = _get_posterior_draws(idata, "sigma")

    # Use M_X-projected residual (see bayesian_robust_lm_lag_test for the
    # principled justification: beta is information-orthogonal to (rho,
    # lambda) under H_0, so beta posterior variance does not enter the LM
    # reference distribution).
    XtX_inv = _safe_inv(X.T @ X, "X'X (cross-sectional robust LM-Error)")
    e_perp = y - X @ (XtX_inv @ (X.T @ y))
    We_perp = np.asarray(W_sp @ e_perp).ravel()
    S_lag = float(e_perp @ Wy)
    S_err = float(e_perp @ We_perp)

    sigma2_draws = sigma_draws**2
    sigma2_mean = float(np.mean(sigma2_draws))

    beta_mean = np.mean(beta_draws, axis=0)
    beta_mean_x = beta_mean[: X.shape[1]]
    J_val = _ols_lag_information(model, beta_mean_x, sigma2_mean)
    J_scaled = J_val / sigma2_mean

    robust_score = (S_err - (T_ww / J_scaled) * S_lag) / sigma2_draws
    denom = T_ww * (1.0 - T_ww / J_scaled)
    LM = robust_score**2 / (abs(denom) + 1e-12)

    return _finalize_lm(LM, test_type="bayesian_robust_lm_error", df=1)


# ---------------------------------------------------------------------------
# SDM/SDEM-aware LM tests (correct residuals from the super-model posterior)
# ---------------------------------------------------------------------------


def bayesian_lm_error_sdm_test(
    model,
) -> BayesianLMTestResult:
    r"""Bayesian LM-Error test from an SDM posterior (H₀: λ = 0 | SDM).

    Tests whether an SDM model should be extended to a MANSAR (Manski)
    model that adds a spatially correlated error term. Residuals are
    computed using the SDM mean structure including the spatial lag of
    ``y``:

    .. math::
        \mathbf{e} = \mathbf{y} - \rho W\mathbf{y} - X\beta - WX\theta

    so that the LM-Error score and variance are evaluated at posterior
    draws from the *correct* fitted model. The score and variance follow
    the standard LM-Error formulation
    (:cite:t:`anselin1996SimpleDiagnostic`), kept on the raw-score scale
    consistent with :func:`bayesian_lm_error_test`:

    .. math::
        S = \mathbf{e}^\top W\mathbf{e}, \qquad
        V = \bar\sigma^4\,T_{WW}

    Returns ``LM = S^2 / V`` per draw, distributed as :math:`\chi^2_1`
    under H₀.
    """
    W_sp = model._W_sparse
    T_ww = model._T_ww

    idata = model.inference_data
    beta_draws = _get_posterior_draws(idata, "beta")  # (draws, k+k_wx)
    rho_draws = _get_posterior_draws(idata, "rho").reshape(-1)  # (draws,)

    resid = _compute_residuals(model, beta_draws, use_Z=True, rho_draws=rho_draws)

    We = (W_sp @ resid.T).T  # (draws, n)
    S = np.sum(resid * We, axis=1)  # (draws,)
    _, sigma2_mean = _posterior_mean_sigma2(idata)
    # Raw-score variance: Var(e'We) = sigma^4 * T_ww (cf. bayesian_lm_error_test).
    V = sigma2_mean**2 * T_ww

    return _lm_scalar(S, V, test_type="bayesian_lm_error_sdm", df=1)


def bayesian_lm_lag_sdem_test(
    model,
) -> BayesianLMTestResult:
    r"""Bayesian LM-Lag test from an SDEM posterior (H₀: ρ = 0 | SDEM).

    Tests whether an SDEM model should be extended to MANSAR by adding a
    spatial lag of ``y``. Residuals are the spatially filtered SDEM
    residuals:

    .. math::
        \mathbf{u} = (I - \lambda W)
            \bigl(\mathbf{y} - X\beta - WX\theta\bigr)

    The score and variance follow the SDEM-filtered LM-Lag derivation:
    using the whitened lag vector :math:`\tilde z_\rho = \bar A_\lambda
    W\mathbf{y}` with :math:`\bar A_\lambda = I - \bar\lambda W` and the
    whitened design :math:`\tilde Z = \bar A_\lambda [X, WX]`,

    .. math::
        S = \mathbf{u}^\top \tilde z_\rho, \qquad
        V = \bar\sigma^4 \, T_{WW}
            + \bar\sigma^2 \, \tilde z_\rho^{\top} M_{\tilde Z}\,
              \tilde z_\rho.

    .. note::
       In the SDEM filter context this naive test coincides
       algebraically with :func:`bayesian_robust_lm_lag_sdem_test`: the
       :math:`\gamma`-score vanishes by the SDEM normal equations and
       the filter absorbs :math:`\lambda` at :math:`\bar\lambda`, so the
       Doǧan Neyman-orthogonal Schur adjustment for
       :math:`(\gamma,\lambda)` is a no-op.  Earlier revisions used an
       unwhitened :math:`S = \boldsymbol{\varepsilon}^\top W\mathbf{y}`
       paired with :math:`V = \bar\sigma^4 T_{WW} + \bar\sigma^2 \|W
       \mathbf{y}\|^2`, which produced empirical size near 1 on
       SDEM-DGP because both the numerator and the denominator omitted
       the :math:`\bar A_\lambda` whitening factor.

    Returns ``LM = S^2 / V`` per draw, distributed as :math:`\chi^2_1`
    under H₀.
    """
    X = model._X
    WX = model._WX
    Wy = model._Wy
    W_sp = model._W_sparse
    T_ww = model._T_ww

    idata = model.inference_data
    beta_draws = _get_posterior_draws(idata, "beta")
    lam_name = _resolve_lam_name(idata)
    lam_draws = _get_posterior_draws(idata, lam_name).reshape(-1)

    u = _compute_residuals(
        model,
        beta_draws,
        use_Z=True,
        lam_draws=lam_draws,
        W_sp=W_sp,
        N=W_sp.shape[0],
        T=1,
    )

    lam_mean = float(np.mean(lam_draws))
    _, sigma2_mean = _posterior_mean_sigma2(idata)
    Z = np.hstack([X, WX])
    # A_λ @ v = v - λ̄(W @ v) — sparse matvec, no dense matrix
    Z_tilde = Z - lam_mean * (W_sp @ Z)
    z_rho = Wy - lam_mean * (W_sp @ Wy)

    S = u @ z_rho  # (draws,)
    V = sigma2_mean**2 * T_ww + sigma2_mean * _mx_quadratic(Z_tilde, z_rho)

    return _lm_scalar(S, V, test_type="bayesian_lm_lag_sdem", df=1)


# ---------------------------------------------------------------------------
# Panel analogues of the SDM/SDEM-aware LM tests
# ---------------------------------------------------------------------------


def _sar_null_lambda_info(
    W_sparse,
    X_design: np.ndarray,
    beta_full_mean: np.ndarray,
    rho_mean: float,
    sigma2_mean: float,
    T_ww: float,
) -> dict:
    r"""Raw-score variance blocks at a SAR (or SDM) null point for the
    pair :math:`(g_\lambda = \mathbf{e}^\top W \mathbf{e},\,
    g_\rho = \mathbf{e}^\top W \mathbf{y})`.

    Evaluated at :math:`\theta^\star = (\bar\beta_{\text{full}},
    \bar\rho, \bar\sigma^2)` with :math:`\lambda = 0`.  Let
    :math:`\bar A = I - \bar\rho W` and :math:`G = \bar A^{-1} W`.
    Under spherical errors and using the Magnus / Anselin (1988) Fisher
    information for SARAR concentrated on :math:`\beta`:

    .. math::
        V_{\lambda\lambda} &= \bar\sigma^4 \, T_{WW},\\
        V_{\lambda\rho}    &= \bar\sigma^4 \, \mathrm{tr}(W^\top G + WG),\\
        V_{\rho\rho}       &= \bar\sigma^4 \, \mathrm{tr}(G^\top G + G^2)
                              + \bar\sigma^2 \,\| M_X (G\,X_{\text{design}}
                              \bar\beta_{\text{full}}) \|^2,

    where :math:`M_X = I - X_{\text{design}}(X_{\text{design}}^\top
    X_{\text{design}})^{-1} X_{\text{design}}^\top`.  ``X_design`` is the
    OLS-projection design (``X`` for SAR null, ``[X, WX]`` for SDM null);
    ``beta_full_mean`` must match its column count.

    All five traces :math:`\mathrm{tr}(G)`, :math:`\mathrm{tr}(G^2)`,
    :math:`\mathrm{tr}(G^\top G)`, :math:`\mathrm{tr}(WG)` and
    :math:`\mathrm{tr}(W^\top G)` are **exact** and eigenvalue-free: a
    single sparse factorization of :math:`\bar A` (KLU when
    available, SuperLU otherwise) is reused for chunked column solves
    :math:`G_{:,J} = \bar A^{-1} W_{:,J}`, which yield
    :math:`\mathrm{tr}(G)`, :math:`\|G\|_F^2 = \mathrm{tr}(G^\top G)`,
    :math:`\mathrm{tr}(W^\top G)` and :math:`\mathrm{tr}(WG)` directly; a
    second backsolve on :math:`W G_{:,J}` yields :math:`\mathrm{tr}(G^2)`.
    This avoids both the :math:`O(n^3)` dense solve and the
    eigendecomposition (which is only exact for these transpose traces
    when ``W`` is normal — row-standardized ``W`` generally is not).
    """
    n = W_sparse.shape[0]

    A_csc = _sp.eye(n, format="csc") - rho_mean * W_sparse
    W_csc = W_sparse.tocsc()

    # One sparse factorization, reused for every solve below.  Prefer
    # sparsax's LU (KLU or UMFPACK, whichever measures faster on this pattern)
    # when available, falling back to the configured scikit-sparse / scipy
    # backend.  sparsax's factor + solve-factor pair lets the numeric factor be
    # reused across all chunked column solves; the scikit-sparse / scipy
    # backends do the same via ``factor.solve``.
    from neighbayes._jax_dispatch import _sparsax_available

    if _sparsax_available():
        import jax.numpy as jnp

        from neighbayes._jax_dispatch import ensure_x64
        from neighbayes.samplers._utils._sparsax_lu import sparsax_lu

        ensure_x64()
        _A_coo = A_csc.tocoo()
        _Ai = jnp.asarray(_A_coo.row, dtype=jnp.int32)
        _Aj = jnp.asarray(_A_coo.col, dtype=jnp.int32)
        _Ax = jnp.asarray(_A_coo.data, dtype=jnp.float64)
        _lu = sparsax_lu(_Ai, _Aj, _Ax, n)
        _factor = _lu.factor(_Ai, _Aj, _Ax, n)

        def solve(rhs):
            return np.asarray(
                _lu.solve_factor(_factor, np.asarray(rhs, dtype=np.float64)),
                dtype=np.float64,
            )
    else:
        backend = _select_sparse_backend()
        if backend == "klu":
            factor = _sparse_factor(A_csc, backend)
            solve = factor.solve
        else:
            lu = _sp.linalg.splu(A_csc)
            solve = lu.solve

    # G @ X_design = A⁻¹ (W @ X_design)
    WX_design = np.asarray(W_sparse @ X_design, dtype=np.float64)
    GX = np.asarray(solve(WX_design), dtype=np.float64)  # (n, k)

    # Exact traces via chunked column solves G[:, J] = A⁻¹ W[:, J].
    tr_G = T_GtG = T_WtG = tr_WG = tr_G2 = 0.0
    chunk = max(1, min(512, n))
    for j0 in range(0, n, chunk):
        j1 = min(j0 + chunk, n)
        idx = np.arange(j1 - j0)
        B = W_csc[:, j0:j1].toarray()  # dense (n, c) block of W columns
        Gc = np.asarray(solve(B), dtype=np.float64)  # G[:, j0:j1]
        tr_G += float(Gc[j0 + idx, idx].sum())
        T_GtG += float(np.sum(Gc * Gc))
        T_WtG += float(np.sum(B * Gc))
        WGc = np.asarray(W_sparse @ Gc, dtype=np.float64)
        tr_WG += float(WGc[j0 + idx, idx].sum())
        G2c = np.asarray(solve(WGc), dtype=np.float64)  # G²[:, j0:j1]
        tr_G2 += float(G2c[j0 + idx, idx].sum())

    T_GG = T_GtG + tr_G2
    T_WG = T_WtG + tr_WG

    # tr(M_X G) = tr(G) - tr((X'X)⁻¹ X'GX)
    XtX = X_design.T @ X_design
    XtGX = X_design.T @ GX
    tr_PxG = float(np.trace(np.linalg.solve(XtX, XtGX)))
    tr_MxG = tr_G - tr_PxG

    V_ll = sigma2_mean**2 * T_ww
    V_lr = sigma2_mean**2 * T_WG

    GxBeta = GX @ beta_full_mean  # reuse GX from above
    proj = _mx_quadratic(X_design, GxBeta)
    V_rr = sigma2_mean**2 * T_GG + sigma2_mean * proj

    return {
        "V_ll": V_ll,
        "V_lr": V_lr,
        "V_rr": V_rr,
        "T_GG": T_GG,
        "T_WG": T_WG,
        "tr_G": tr_G,
        "tr_MxG": tr_MxG,
    }


def bayesian_lm_error_from_sar_test(
    model,
) -> BayesianLMTestResult:
    r"""Bayesian LM-Error test from a SAR posterior (H₀: λ = 0 | SAR).

    SAR-aware companion of :func:`bayesian_lm_error_test`. Residuals are
    formed using the SAR mean structure
    :math:`\mathbf{e}^{(d)} = \mathbf{y} - \rho^{(d)} W\mathbf{y} -
    X\beta^{(d)}` so that the LM-Error score is evaluated at the *correct*
    null model (SAR), not at OLS. The score and variance are otherwise
    identical to :func:`bayesian_lm_error_test`:

    .. math::
        S^{(d)} = \mathbf{e}^{(d)\,\top} W \mathbf{e}^{(d)}, \qquad
        V = \bar\sigma^4\, T_{WW},

    so the per-draw statistic is :math:`\mathrm{LM}^{(d)} = (S^{(d)})^2/V`
    and is referenced against :math:`\chi^2_1`.

    This is a precursor diagnostic for the SAR-context Schur-robust
    LM-Error of :func:`bayesian_robust_lm_error_sar_test`; the decision
    tree fires the robust adjustment only when this naive test rejects.
    """
    W_sp = model._W_sparse
    T_ww = model._T_ww

    idata = model.inference_data
    beta_draws = _get_posterior_draws(idata, "beta")  # (draws, k)
    rho_draws = _get_posterior_draws(idata, "rho").reshape(-1)  # (draws,)

    resid = _compute_residuals(model, beta_draws, rho_draws=rho_draws)

    We = (W_sp @ resid.T).T
    S = np.sum(resid * We, axis=1)  # (draws,)

    _, sigma2_mean = _posterior_mean_sigma2(idata)
    V = sigma2_mean**2 * T_ww

    return _lm_scalar(S, V, test_type="bayesian_lm_error_from_sar", df=1)


def bayesian_robust_lm_error_sar_test(
    model,
) -> BayesianLMTestResult:
    r"""Bayesian robust LM-Error test in SAR context (H₀: λ = 0 | SAR).

    Tests whether a SAR fit should be extended to SARAR by adding a
    spatial-error process, robust to the locally-estimated lag parameter
    :math:`\rho`.  Implements the Doǧan–Taşpınar–Bera (2021) Bayesian
    Neyman-orthogonal score adjustment
    (:cite:p:`dogan2021BayesianRobust`, Proposition 3) with the SARAR
    information blocks at the SAR posterior mean
    (:cite:p:`anselin1988SpatialEconometrics`,
    :cite:p:`anselin1996SimpleDiagnostic`).

    For each posterior draw of :math:`(\beta, \rho, \sigma^2)` from the
    SAR fit the residual is
    :math:`\mathbf{e}^{(d)} = \mathbf{y} - \rho^{(d)} W\mathbf{y} -
    X\beta^{(d)}`, and the raw scores are

    .. math::
        g_\lambda^{(d)} = \mathbf{e}^{(d)\,\top} W \mathbf{e}^{(d)},
        \qquad
        g_\rho^{(d)} = \mathbf{e}^{(d)\,\top} W \mathbf{y}.

    With :math:`\bar A = I - \bar\rho W`, :math:`G = \bar A^{-1} W` and
    :math:`T_{B,C} = \mathrm{tr}(B^\top C + BC)`, the variance blocks at
    :math:`\theta^\star` are

    .. math::
        V_{\lambda\lambda} &= \bar\sigma^4 \, T_{WW},\\
        V_{\lambda\rho}    &= \bar\sigma^4 \, T_{W,G},\\
        V_{\rho\rho}       &= \bar\sigma^4 \, T_{G,G}
            + \bar\sigma^2 \, \| M_X (G\,X\bar\beta) \|^2.

    The Neyman-orthogonal adjusted score is

    .. math::
        g_\lambda^{*\,(d)} = g_\lambda^{(d)}
            - \frac{V_{\lambda\rho}}{V_{\rho\rho}}\, g_\rho^{(d)},

    with adjusted variance
    :math:`V_{\lambda\,|\,\rho} = V_{\lambda\lambda} - V_{\lambda\rho}^2 /
    V_{\rho\rho}` and per-draw statistic

    .. math::
        \mathrm{LM}_R^{(d)} = \frac{(g_\lambda^{*\,(d)})^2}
                                   {V_{\lambda\,|\,\rho}}
        \;\xrightarrow{d}\; \chi^2_1
        \quad \text{under } H_0,

    independent of local misspecification in :math:`\rho`.

    Parameters
    ----------
    model : SAR
        Fitted SAR model exposing ``inference_data`` with posterior draws
        of ``beta``, ``rho``, ``sigma`` and the cached ``_y``, ``_X``,
        ``_Wy``, ``_W_sparse``, ``_T_ww`` attributes.

    Returns
    -------
    BayesianLMTestResult
        Per-draw LM samples, summary statistics and ``df = 1``.
    """
    y = model._y
    X = model._X
    Wy = model._Wy
    W_sp = model._W_sparse
    T_ww = model._T_ww

    idata = model.inference_data
    beta_draws = _get_posterior_draws(idata, "beta")  # (draws, k)
    rho_draws = _get_posterior_draws(idata, "rho").reshape(-1)  # (draws,)
    sigma_draws = _get_posterior_draws(idata, "sigma")  # (draws,)

    # Project residual onto M_X to remove beta-direction posterior noise
    # (information-orthogonality of beta to lambda under H_0). The SAR
    # residual e^(d) = y - rho^(d) Wy - X beta^(d) becomes
    # M_X e^(d) = M_X y - rho^(d) M_X Wy, preserving rho-uncertainty
    # propagation while killing the beta-noise component that would
    # otherwise dominate the small Schur-complement denominator.
    XtX_inv = _safe_inv(X.T @ X, "X'X (SAR-null robust LM-Error)")
    Mx_y = y - X @ (XtX_inv @ (X.T @ y))  # = M_X y, shape (n,)
    Mx_Wy = Wy - X @ (XtX_inv @ (X.T @ Wy))  # = M_X Wy, shape (n,)
    # e_perp^(d) = Mx_y - rho^(d) * Mx_Wy
    resid_perp = Mx_y[None, :] - rho_draws[:, None] * Mx_Wy[None, :]
    We_perp = (W_sp @ resid_perp.T).T
    g_lambda = np.sum(resid_perp * We_perp, axis=1)  # (draws,)
    g_rho = np.dot(resid_perp, Wy)  # (draws,)

    beta_mean = np.mean(beta_draws, axis=0)
    rho_mean = float(np.mean(rho_draws))
    sigma_draws, sigma2_mean = _posterior_mean_sigma2(idata)

    info = _sar_null_lambda_info(W_sp, X, beta_mean, rho_mean, sigma2_mean, T_ww)
    V_ll = info["V_ll"]
    V_lr = info["V_lr"]
    V_rr = info["V_rr"]
    tr_MxG = info["tr_MxG"]

    # Center the rho-direction score.  The score uses the M_X-projected
    # residual e_perp = M_X y - rho M_X W y, so under H_0:lambda=0
    #     E[g_rho] = E[(M_X eps)' W y] = sigma^2 tr(M_X G),
    # since W y = G(X beta + eps) and M_X X = 0.  The unprojected
    # centering sigma^2 tr(G) (the SARAR log-likelihood gradient) would
    # mismatch the M_X-projected score and leave an O(n) bias under
    # rho != 0; using tr(M_X G) restores the bias-free score that the
    # Schur correction expects.  At rho = 0, G = W and tr(M_X W) = 0
    # for row-standardized W with intercept, recovering the OLS-null
    # behavior.
    g_rho_centered = g_rho - (sigma_draws**2) * tr_MxG

    g_lambda_star, V_l_given_r = _neyman_adjust_scalar(
        g_lambda,
        g_rho_centered[:, None],
        V_ll,
        V_lr,
        V_rr,
        label="V_l_given_r (robust LM-Error-SAR)",
    )
    LM = g_lambda_star**2 / (abs(V_l_given_r) + 1e-12)

    return _finalize_lm(
        LM,
        test_type="bayesian_robust_lm_error_sar",
        df=1,
    )


def bayesian_robust_lm_error_sdm_test(
    model,
) -> BayesianLMTestResult:
    r"""Bayesian robust LM-Error test in SDM context (H₀: λ = 0 | SDM).

    Tests whether a SDM fit should be extended to MANSAR by adding a
    spatial-error process, robust to the SDM lag parameter
    :math:`\rho`.  The WX coefficients :math:`\gamma` are already in the
    SDM mean structure and are absorbed via the
    :math:`M_Z`-projection with :math:`Z = [X, WX]`; the only nuisance to
    Schur against is :math:`\rho`
    (:cite:p:`dogan2021BayesianRobust`, Proposition 3;
    :cite:p:`anselin1996SimpleDiagnostic`;
    :cite:p:`koley2024UseNot`).

    For each posterior draw of :math:`(\beta, \gamma, \rho, \sigma^2)`
    from the SDM fit the residual is
    :math:`\mathbf{e}^{(d)} = \mathbf{y} - \rho^{(d)} W\mathbf{y}
    - X\beta^{(d)} - WX\gamma^{(d)}`.  Raw scores
    :math:`g_\lambda = \mathbf{e}^\top W \mathbf{e}` and
    :math:`g_\rho = \mathbf{e}^\top W \mathbf{y}` are evaluated per
    draw, and the variance blocks
    :math:`(V_{\lambda\lambda}, V_{\lambda\rho}, V_{\rho\rho})` use the
    SAR-null Magnus identities of
    :func:`_sar_null_lambda_info` with :math:`X_{\text{design}} = Z`.
    The Neyman-orthogonal adjustment and Schur complement are identical
    to :func:`bayesian_robust_lm_error_sar_test`; only the projector
    differs.  The statistic is :math:`\chi^2_1` under :math:`H_0`.

    Parameters
    ----------
    model : SDM
        Fitted SDM model with ``inference_data`` containing posterior
        draws for ``beta`` (covering ``[X, WX]``), ``rho``, ``sigma``.

    Returns
    -------
    BayesianLMTestResult
        Per-draw LM samples, summary statistics and ``df = 1``.
    """
    y = model._y
    X = model._X
    WX = model._WX
    Wy = model._Wy
    W_sp = model._W_sparse
    T_ww = model._T_ww

    idata = model.inference_data
    beta_draws = _get_posterior_draws(idata, "beta")  # (draws, k+k_wx)
    rho_draws = _get_posterior_draws(idata, "rho").reshape(-1)
    sigma_draws = _get_posterior_draws(idata, "sigma")

    Z = np.hstack([X, WX])

    # Project residual onto M_Z = I - Z(Z'Z)^{-1}Z' to remove
    # (beta, gamma)-direction posterior noise.  Without this projection
    # the per-draw residual carries beta- and gamma-uncertainty that
    # dominates the small Schur-complement denominator and makes the
    # statistic blow up under H_0:lambda=0 (size ~ 0.5 at n=1600 in
    # MC).  The projection is exact: under SDM the score is
    # information-orthogonal to (beta, gamma) at theta^*, so projecting
    # them out leaves the rho-uncertainty intact while killing
    # beta/gamma noise.  This is the SDM analogue of the SAR-context
    # M_X projection in `bayesian_robust_lm_error_sar_test`.
    ZtZ_inv = _safe_inv(Z.T @ Z, "Z'Z (SDM-null robust LM-Error)")
    Mz_y = y - Z @ (ZtZ_inv @ (Z.T @ y))  # = M_Z y, shape (n,)
    Mz_Wy = Wy - Z @ (ZtZ_inv @ (Z.T @ Wy))  # = M_Z W y, shape (n,)
    resid_perp = Mz_y[None, :] - rho_draws[:, None] * Mz_Wy[None, :]
    We_perp = (W_sp @ resid_perp.T).T
    g_lambda = np.sum(resid_perp * We_perp, axis=1)
    g_rho = np.dot(resid_perp, Wy)

    beta_mean = np.mean(beta_draws, axis=0)
    rho_mean = float(np.mean(rho_draws))
    sigma2_mean = float(np.mean(sigma_draws**2))

    info = _sar_null_lambda_info(W_sp, Z, beta_mean, rho_mean, sigma2_mean, T_ww)
    V_ll = info["V_ll"]
    V_lr = info["V_lr"]
    V_rr = info["V_rr"]
    tr_MzG = info["tr_MxG"]  # tr(M_Z G) since X_design=Z

    # Center the rho-direction score by E[g_rho | H_0] = sigma^2 tr(M_Z G)
    # to match the M_Z-projected residual.  The unprojected centering
    # sigma^2 tr(G) (the SARAR log-likelihood gradient) would mismatch
    # the M_Z-projected score and leave an O(n) bias under rho != 0;
    # using tr(M_Z G) restores the bias-free score that the Schur
    # correction expects.  Verified by analytic delta-posterior:
    # raw-resid + tr(G) gives size ~ 0.47; M_Z-resid + tr(M_Z G) gives
    # size ~ 0.04 across n in {64, 225, 625}.
    g_rho_centered = g_rho - (sigma_draws**2) * tr_MzG

    g_lambda_star, V_l_given_r = _neyman_adjust_scalar(
        g_lambda,
        g_rho_centered[:, None],
        V_ll,
        V_lr,
        V_rr,
        label="V_l_given_r (robust LM-Error-SDM)",
    )
    LM = g_lambda_star**2 / (abs(V_l_given_r) + 1e-12)

    return _finalize_lm(
        LM,
        test_type="bayesian_robust_lm_error_sdm",
        df=1,
        details={"k_wx": WX.shape[1]},
    )


def _sem_filtered_blocks(
    W_sparse,
    X: np.ndarray,
    Wy: np.ndarray,
    WX: np.ndarray,
    lam_mean: float,
    sigma2_mean: float,
    T_ww: float,
) -> dict:
    r"""Filtered-OLS raw-score variance blocks at a SEM null point.

    Define :math:`\bar A_\lambda = I - \bar\lambda W`,
    :math:`\tilde X = \bar A_\lambda X`, :math:`\tilde z_\rho =
    \bar A_\lambda W\mathbf{y}` and :math:`\tilde Z_\gamma =
    \bar A_\lambda WX`.  Under the spatially-filtered Gaussian model the
    raw-score variance blocks for the additional SARAR / SDEM
    parameters :math:`(\rho, \gamma)`, evaluated with :math:`\beta`
    concentrated out via :math:`M_{\tilde X}`, are

    .. math::
        V_{\rho\rho} &= \bar\sigma^4 \, T_{WW}
                       + \bar\sigma^2 \, \tilde z_\rho^{\top}
                         M_{\tilde X}\, \tilde z_\rho,\\
        V_{\rho\gamma} &= \bar\sigma^2 \, \tilde z_\rho^{\top}
                          M_{\tilde X}\, \tilde Z_\gamma,\\
        V_{\gamma\gamma} &= \bar\sigma^2 \, \tilde Z_\gamma^{\top}
                            M_{\tilde X}\, \tilde Z_\gamma,

    with :math:`T_{WW} = \mathrm{tr}(W^\top W + W^2)`.  The
    :math:`T_{WW}` contribution to :math:`V_{\rho\rho}` arises from the
    SARAR Hessian's Magnus trace term, which is independent of the
    filter at :math:`\rho = 0`.

    All :math:`\bar A_\lambda \mathbf{v}` products are computed as
    :math:`\mathbf{v} - \bar\lambda (W \mathbf{v})` using sparse
    matrix-vector products — :math:`O(\mathrm{nnz})` rather than
    :math:`O(n^2)` dense matmuls.  The filtered vectors
    :math:`\tilde z_\rho` and :math:`\tilde Z_\gamma` are returned in
    the dict so callers need not recompute them.
    """
    # A_λ @ v = v - λ̄(W @ v) — sparse matvec, no dense matrix
    X_tilde = X - lam_mean * (W_sparse @ X)
    z_rho = Wy - lam_mean * (W_sparse @ Wy)
    Z_gamma = WX - lam_mean * (W_sparse @ WX)

    V_rr = sigma2_mean**2 * T_ww + sigma2_mean * _mx_quadratic(X_tilde, z_rho)
    if WX.shape[1] > 0:
        V_gg = sigma2_mean * _mx_cross(X_tilde, Z_gamma, Z_gamma)
        V_rg = sigma2_mean * np.asarray(_mx_cross(X_tilde, z_rho, Z_gamma)).ravel()
    else:
        V_gg = np.zeros((0, 0))
        V_rg = np.zeros(0)

    return {
        "V_rr": V_rr,
        "V_gg": V_gg,
        "V_rg": V_rg,
        "z_rho": z_rho,
        "Z_gamma": Z_gamma,
    }


def bayesian_robust_lm_lag_sem_test(
    model,
) -> BayesianLMTestResult:
    r"""Bayesian robust LM-Lag test in SEM context (H₀: ρ = 0 | SEM).

    Tests whether a SEM fit should be extended to add a spatial lag
    (→ SARAR or SDM/SDEM family), robust to the WX block.  The SEM
    posterior provides :math:`(\beta, \lambda, \sigma^2)`; treating
    :math:`\bar\lambda` as the filtering point, the alternative model
    becomes a filtered OLS with two candidate omitted blocks
    (:math:`\rho` for the lag, :math:`\gamma` for WX).

    For each posterior draw the whitened residual is
    :math:`\mathbf{u}^{(d)} = (I - \lambda^{(d)} W)
    (\mathbf{y} - X\beta^{(d)})`.  In the filter at :math:`\bar\lambda`
    let :math:`\tilde z_\rho = \bar A_\lambda W\mathbf{y}` and
    :math:`\tilde Z_\gamma = \bar A_\lambda WX`.  Raw scores are

    .. math::
        g_\rho^{(d)} = \mathbf{u}^{(d)\,\top} \tilde z_\rho,
        \qquad
        \mathbf{g}_\gamma^{(d)} = \tilde Z_\gamma^{\top} \mathbf{u}^{(d)}.

    Variance blocks come from :func:`_sem_filtered_blocks`.  The
    Neyman-orthogonal score adjusts :math:`g_\rho` for the
    :math:`\gamma` direction:

    .. math::
        g_\rho^{*\,(d)} &= g_\rho^{(d)}
            - V_{\rho\gamma} V_{\gamma\gamma}^{-1}
              \mathbf{g}_\gamma^{(d)},\\
        V_{\rho\,|\,\gamma} &= V_{\rho\rho}
            - V_{\rho\gamma} V_{\gamma\gamma}^{-1} V_{\gamma\rho}.

    The per-draw statistic is

    .. math::
        \mathrm{LM}_R^{(d)} = \frac{(g_\rho^{*\,(d)})^2}
                                   {V_{\rho\,|\,\gamma}}
        \;\xrightarrow{d}\; \chi^2_1
        \quad \text{under } H_0,

    independent of local misspecification in :math:`\gamma`.

    Parameters
    ----------
    model : SEM
        Fitted SEM model exposing ``inference_data`` with posterior
        draws of ``beta``, ``lam`` (or ``lambda``), ``sigma``.

    Returns
    -------
    BayesianLMTestResult
        Per-draw LM samples, summary statistics and ``df = 1``.
    """
    y = model._y
    X = model._X
    WX = model._WX
    Wy = model._Wy
    W_sp = model._W_sparse
    T_ww = model._T_ww

    idata = model.inference_data
    beta_draws = _get_posterior_draws(idata, "beta")
    lam_name = _resolve_lam_name(idata)
    lam_draws = _get_posterior_draws(idata, lam_name).reshape(-1)
    sigma_draws = _get_posterior_draws(idata, "sigma")

    # Per-draw whitened residuals u_d = (I - lam_d W)(y - X beta_d)
    raw = y[None, :] - beta_draws @ X.T  # (draws, n)
    Wraw = (W_sp @ raw.T).T
    u = raw - lam_draws[:, None] * Wraw

    # Filtered designs at posterior-mean lambda
    lam_mean = float(np.mean(lam_draws))
    sigma2_mean = float(np.mean(sigma_draws**2))
    blocks = _sem_filtered_blocks(W_sp, X, Wy, WX, lam_mean, sigma2_mean, T_ww)
    z_rho = blocks["z_rho"]  # (n,)
    Z_gamma = blocks["Z_gamma"]  # (n, k_wx)

    g_rho = u @ z_rho  # (draws,)
    g_gamma = u @ Z_gamma  # (draws, k_wx)

    V_rr = blocks["V_rr"]
    V_gg = blocks["V_gg"]
    V_rg = blocks["V_rg"]
    k_wx = WX.shape[1]

    g_rho_star, V_r_given_g = _neyman_adjust_scalar(
        g_rho,
        g_gamma,
        V_rr,
        V_rg,
        V_gg,
        label="V_r_given_g (robust LM-Lag-SEM)",
    )

    LM = g_rho_star**2 / (abs(V_r_given_g) + 1e-12)

    return _finalize_lm(
        LM,
        test_type="bayesian_robust_lm_lag_sem",
        df=1,
        details={"k_wx": k_wx},
    )


def bayesian_robust_lm_wx_sem_test(
    model,
) -> BayesianLMTestResult:
    r"""Bayesian robust LM-WX test in SEM context (H₀: γ = 0 | SEM).

    Companion to :func:`bayesian_robust_lm_lag_sem_test` with target and
    nuisance swapped: tests whether the SEM fit should be extended to
    SDEM by adding the WX block, robust to a locally-omitted spatial
    lag.

    Setup mirrors :func:`bayesian_robust_lm_lag_sem_test`: per-draw
    whitened residuals :math:`\mathbf{u}^{(d)} = (I - \lambda^{(d)} W)
    (\mathbf{y} - X\beta^{(d)})`, filtered designs at
    :math:`\bar\lambda`, raw scores

    .. math::
        \mathbf{g}_\gamma^{(d)} = \tilde Z_\gamma^{\top} \mathbf{u}^{(d)},
        \qquad
        g_\rho^{(d)} = \mathbf{u}^{(d)\,\top} \tilde z_\rho.

    The Neyman-orthogonal adjustment and Schur complement (with target
    :math:`\gamma`, nuisance :math:`\rho`) give

    .. math::
        \mathbf{g}_\gamma^{*\,(d)} &= \mathbf{g}_\gamma^{(d)}
            - V_{\gamma\rho} V_{\rho\rho}^{-1} g_\rho^{(d)},\\
        V_{\gamma\,|\,\rho} &= V_{\gamma\gamma}
            - V_{\gamma\rho} V_{\rho\rho}^{-1} V_{\rho\gamma}.

    The per-draw statistic is

    .. math::
        \mathrm{LM}_R^{(d)} = \mathbf{g}_\gamma^{*\,(d)\,\top}
            V_{\gamma\,|\,\rho}^{-1} \mathbf{g}_\gamma^{*\,(d)}
        \;\xrightarrow{d}\; \chi^2_{k_{wx}}
        \quad \text{under } H_0,

    independent of local misspecification in :math:`\rho`.

    Parameters
    ----------
    model : SEM
        Fitted SEM model with ``inference_data`` containing posterior
        draws for ``beta``, ``lam`` (or ``lambda``) and ``sigma``.

    Returns
    -------
    BayesianLMTestResult
        Per-draw LM samples, summary statistics and ``df = k_{wx}``.
    """
    y = model._y
    X = model._X
    WX = model._WX
    Wy = model._Wy
    W_sp = model._W_sparse
    T_ww = model._T_ww
    k_wx = WX.shape[1]

    if k_wx == 0:
        raise ValueError(
            "Model has no WX columns. The robust LM-WX test requires at "
            "least one spatially lagged covariate."
        )

    idata = model.inference_data
    beta_draws = _get_posterior_draws(idata, "beta")
    lam_name = _resolve_lam_name(idata)
    lam_draws = _get_posterior_draws(idata, lam_name).reshape(-1)
    sigma_draws = _get_posterior_draws(idata, "sigma")

    raw = y[None, :] - beta_draws @ X.T
    Wraw = (W_sp @ raw.T).T
    u = raw - lam_draws[:, None] * Wraw

    lam_mean = float(np.mean(lam_draws))
    sigma2_mean = float(np.mean(sigma_draws**2))
    blocks = _sem_filtered_blocks(W_sp, X, Wy, WX, lam_mean, sigma2_mean, T_ww)
    z_rho = blocks["z_rho"]
    Z_gamma = blocks["Z_gamma"]

    g_rho = u @ z_rho
    g_gamma = u @ Z_gamma

    V_rr = float(blocks["V_rr"])
    V_gg = blocks["V_gg"]
    V_rg = blocks["V_rg"]

    # Neyman: g_gamma* = g_gamma - V_{gamma,rho} V_{rho,rho}^{-1} g_rho.
    g_gamma_star, V_g_given_r = _neyman_adjust_vector(
        g_gamma,
        g_rho,
        V_gg,
        V_rg,
        V_rr,
        label="V_g_given_r (robust LM-WX-SEM)",
    )

    V_inv = _safe_inv(V_g_given_r, "V_g_given_r (robust LM-WX-SEM)")
    LM = np.einsum("di,ij,dj->d", g_gamma_star, V_inv, g_gamma_star)

    return _finalize_lm(
        LM,
        test_type="bayesian_robust_lm_wx_sem",
        df=k_wx,
        details={"k_wx": k_wx},
    )


def bayesian_robust_lm_lag_sdem_test(
    model,
) -> BayesianLMTestResult:
    r"""Bayesian robust LM-Lag test in SDEM context (H₀: ρ = 0 | SDEM).

    Tests whether a SDEM fit should be extended to MANSAR by adding a
    spatial lag, with the WX block already absorbed in the SDEM mean
    structure and the SDEM error parameter :math:`\lambda` absorbed via a
    posterior-mean filter
    (:cite:p:`dogan2021BayesianRobust`, Proposition 3;
    :cite:p:`koley2024UseNot`).

    For each posterior draw of :math:`(\beta, \gamma, \lambda,
    \sigma^2)` from the SDEM fit the whitened residual is

    .. math::
        \mathbf{u}^{(d)} = (I - \lambda^{(d)} W)
            \bigl(\mathbf{y} - X\beta^{(d)} - WX\gamma^{(d)}\bigr).

    Letting :math:`Z = [X, WX]`, :math:`\tilde Z = \bar A_\lambda Z` and
    :math:`\tilde z_\rho = \bar A_\lambda W\mathbf{y}`, the raw score
    and concentrated variance are

    .. math::
        g_\rho^{(d)} &= \mathbf{u}^{(d)\,\top} \tilde z_\rho,\\
        V_{\rho \cdot \beta,\gamma} &= \bar\sigma^4 \, T_{WW}
            + \bar\sigma^2 \, \tilde z_\rho^{\top} M_{\tilde Z}\,
              \tilde z_\rho.

    Because the SDEM mean structure already contains :math:`WX` (so the
    score for :math:`\gamma` is identically zero from the SDEM normal
    equations) and the filter absorbs :math:`\lambda` at
    :math:`\bar\lambda`, the Doǧan Neyman-orthogonal adjustment is a
    no-op and the statistic reduces to

    .. math::
        \mathrm{LM}_R^{(d)} = \frac{(g_\rho^{(d)})^2}
                                   {V_{\rho \cdot \beta,\gamma}}
        \;\xrightarrow{d}\; \chi^2_1
        \quad \text{under } H_0.

    Parameters
    ----------
    model : SDEM
        Fitted SDEM model with ``inference_data`` containing posterior
        draws for ``beta`` (covering ``[X, WX]``), ``lam`` (or
        ``lambda``), and ``sigma``.

    Returns
    -------
    BayesianLMTestResult
        Per-draw LM samples, summary statistics and ``df = 1``.
    """
    y = model._y
    X = model._X
    WX = model._WX
    Wy = model._Wy
    W_sp = model._W_sparse
    T_ww = model._T_ww

    idata = model.inference_data
    beta_draws = _get_posterior_draws(idata, "beta")  # (draws, k+k_wx)
    lam_name = _resolve_lam_name(idata)
    lam_draws = _get_posterior_draws(idata, lam_name).reshape(-1)
    sigma_draws = _get_posterior_draws(idata, "sigma")

    Z = np.hstack([X, WX])
    raw = y[None, :] - beta_draws @ Z.T
    Wraw = (W_sp @ raw.T).T
    u = raw - lam_draws[:, None] * Wraw

    lam_mean = float(np.mean(lam_draws))
    sigma2_mean = float(np.mean(sigma_draws**2))
    # A_λ @ v = v - λ̄(W @ v) — sparse matvec, no dense matrix
    Z_tilde = Z - lam_mean * (W_sp @ Z)
    z_rho = Wy - lam_mean * (W_sp @ Wy)

    g_rho = u @ z_rho

    V_rr = sigma2_mean**2 * T_ww + sigma2_mean * _mx_quadratic(Z_tilde, z_rho)
    LM = g_rho**2 / (abs(V_rr) + 1e-12)

    return _finalize_lm(
        LM,
        test_type="bayesian_robust_lm_lag_sdem",
        df=1,
        details={"k_wx": WX.shape[1]},
    )
