r"""Bayesian LM diagnostic tests — GLM (PG-augmented) cross-sectional variants.

Mirrors :mod:`neighbayes.diagnostics.lmtests.cross_sectional` but
operates on the Gaussian pseudo-regression induced by Pólya-Gamma
augmentation of logistic / NB-2 likelihoods (Polson, Scott & Windle
2013).  See :mod:`neighbayes.diagnostics.lmtests._glm` for the
augmentation algebra and weight definitions.

All three tests share the same template as the Gaussian SAR family:

1. Per posterior draw :math:`(\beta, \alpha)` compute the working
   response :math:`\tilde z`, working residual :math:`\tilde e =
   \tilde z - X\beta`, and PG weights :math:`\omega_i =
   E[\omega_i \mid \psi_i]`.
2. Form a raw score in the weighted pseudo-regression analogous to
   the Gaussian SAR LM scores.
3. Evaluate the concentrated variance at the posterior mean of the
   weights and pseudo-response.
4. Return :class:`BayesianLMTestResult` via ``_lm_scalar`` /
   ``_lm_vector``.
"""

from __future__ import annotations

import numpy as np

from ._glm import (
    _pg_mean_weight,
    _weighted_mx_cross,
    _weighted_mx_quadratic,
    _weighted_T_ww,
    glm_working_response,
)
from .core import (
    BayesianLMTestResult,
    _get_posterior_draws,
    _lm_scalar,
    _lm_vector,
    _resolve_X_for_beta,
)


def _require_W(model, test_name: str):
    if getattr(model, "_W_sparse", None) is None:
        raise ValueError(
            f"{test_name} requires the model to have been constructed with a "
            "spatial weights matrix W."
        )


def bayesian_glm_lm_error_test(model) -> BayesianLMTestResult:
    r"""Bayesian LM test for omitted spatial error in a GLM (logit / NB).

    PG-augmented analog of :func:`bayesian_lm_error_test`.  Conditional on
    the working weights :math:`\omega`, the augmented model is a weighted
    Gaussian regression of :math:`\tilde z = \kappa/\omega` on :math:`X`
    with covariance :math:`\Omega^{-1}` (Polson, Scott & Windle 2013).
    The SEM-null LM score is

    .. math::
        S^{(d)} = \tilde e^{(d)\top}\, \Omega^{(d)}\, W\, \tilde e^{(d)},
        \qquad \tilde e^{(d)} = \tilde z^{(d)} - X\beta^{(d)},

    and its variance under :math:`H_0` is

    .. math::
        V = \mathrm{tr}(W^2) + \sum_{i,j} \bar\omega_i/\bar\omega_j\, W_{ij}^2,

    evaluated at the posterior-mean weights :math:`\bar\omega`.  This
    reduces to the Anselin (1996) :math:`\sigma^4 T_{WW}` formula when
    :math:`\Omega = \sigma^{-2} I`.

    Returns
    -------
    BayesianLMTestResult
        Per-draw LM samples, summary statistics, ``df = 1``.
    """
    _require_W(model, "bayesian_glm_lm_error_test")
    idata = model.inference_data
    beta_draws = _get_posterior_draws(idata, "beta")
    _, e_tilde, omega = glm_working_response(model, beta_draws, idata)
    W = model._W_sparse

    # Per-draw score: e' Ω W e
    We = (W @ e_tilde.T).T  # (draws, n)
    S = np.einsum("di,di,di->d", e_tilde, omega, We)

    # Variance at posterior-mean weights
    w_bar = omega.mean(axis=0)
    V = _weighted_T_ww(W, w_bar)

    return _lm_scalar(
        S,
        V,
        test_type="bayesian_glm_lm_error",
        df=1,
        details={"model_type": getattr(model, "_model_type", None)},
    )


def bayesian_glm_lm_lag_test(model) -> BayesianLMTestResult:
    r"""Bayesian LM test for an omitted spatial lag of the linear predictor.

    PG-augmented analog of :func:`bayesian_lm_lag_test`.  In the SAR-GLM
    family the linear predictor satisfies :math:`\psi = (I-\rho W)^{-1}
    X\beta`, so :math:`\rho` enters only the *mean* of the augmented
    response :math:`\tilde z = \psi + \varepsilon`; the conditional
    covariance :math:`\mathrm{Cov}(\tilde z\mid\psi,\omega) = \Omega^{-1}`
    is :math:`\rho`-independent.  This is the structural difference from
    Gaussian SAR (where :math:`\mathrm{Var}(y) = \sigma^2 (I-\rho W)^{-1}
    (I-\rho W)^{-\top}` depends on :math:`\rho`) and it drops the
    :math:`T_{WW}` trace term that appears in Anselin (1996, eq. 13).

    The score reduces to the standard GLM score test for adding a single
    regressor :math:`z = WX\beta` to the linear predictor:

    .. math::
        S^{(d)} = (WX\beta^{(d)})^\top\, \Omega^{(d)}\, \tilde e^{(d)},

    with concentrated variance

    .. math::
        V = (WX\bar\beta)^\top \bar\Omega\, M_X^{\bar\Omega}\,
            (WX\bar\beta),
        \qquad M_X^{\bar\Omega} = I - X(X^\top\bar\Omega X)^{-1}
                                  X^\top\bar\Omega.

    Asymptotically :math:`\chi^2_1` under :math:`H_0:\rho=0`.

    Returns
    -------
    BayesianLMTestResult
    """
    _require_W(model, "bayesian_glm_lm_lag_test")
    idata = model.inference_data
    beta_draws = _get_posterior_draws(idata, "beta")
    _, e_tilde, omega = glm_working_response(model, beta_draws, idata)
    W = model._W_sparse
    X = _resolve_X_for_beta(model, beta_draws)

    # Per-draw regressor z^(d) = W X β^(d) (depends on β)
    Xbeta = beta_draws @ X.T  # (draws, n) — equal to ψ^(d)
    z_draw = (W @ Xbeta.T).T  # (draws, n)
    # Per-draw score: (W X β)' Ω ẽ
    S = np.einsum("di,di,di->d", z_draw, omega, e_tilde)

    # Concentrated variance at posterior means
    w_bar = omega.mean(axis=0)
    beta_bar = beta_draws.mean(axis=0)
    z_bar = np.asarray(W @ (X @ beta_bar)).ravel()
    V = _weighted_mx_quadratic(X, z_bar, w_bar)

    return _lm_scalar(
        S,
        V,
        test_type="bayesian_glm_lm_lag",
        df=1,
        details={"model_type": getattr(model, "_model_type", None)},
    )


def bayesian_glm_lm_wx_test(model) -> BayesianLMTestResult:
    r"""Bayesian LM test for omitted WX coefficients in a GLM (H₀: γ = 0).

    PG-augmented analog of :func:`bayesian_lm_wx_test`.  Tests whether
    spatially lagged covariates :math:`WX` should be added to the
    GLM mean.  The score in the weighted pseudo-regression is

    .. math::
        \mathbf{g}_\gamma^{(d)} = (WX)^\top \Omega^{(d)} \tilde e^{(d)},

    with variance under H₀

    .. math::
        V_{\gamma\gamma} = (WX)^\top \bar\Omega\, M_X^{\bar\Omega}\, (WX)
                = (WX)^\top \bar\Omega (WX)
                  - (WX)^\top \bar\Omega X (X^\top \bar\Omega X)^{-1}
                      X^\top \bar\Omega (WX).

    Reduces to the spreg ``lm_wx`` formula when
    :math:`\bar\Omega = \sigma^{-2} I`.

    Returns
    -------
    BayesianLMTestResult
        Per-draw LM samples, summary statistics, ``df = k_{wx}``.
    """
    _require_W(model, "bayesian_glm_lm_wx_test")
    WX = model._WX
    k_wx = WX.shape[1]
    if k_wx == 0:
        raise ValueError(
            "Model has no WX columns. The WX test requires at least one "
            "spatially lagged covariate (set ``durbin_vars`` at construction)."
        )

    idata = model.inference_data
    beta_draws = _get_posterior_draws(idata, "beta")
    _, e_tilde, omega = glm_working_response(model, beta_draws, idata)
    X = model._X

    # Per-draw vector score: g_γ = (WX)' Ω e
    g_gamma = np.einsum("ij,di,di->dj", WX, omega, e_tilde)

    # Variance at posterior-mean weights
    w_bar = omega.mean(axis=0)
    V = _weighted_mx_cross(X, WX, WX, w_bar)

    return _lm_vector(
        g_gamma,
        V,
        test_type="bayesian_glm_lm_wx",
        df=k_wx,
        details={"k_wx": k_wx, "model_type": getattr(model, "_model_type", None)},
        label="V_gamma_gamma (GLM LM-WX)",
    )


# Re-export the conditional-mean PG weight helper for callers that want to
# inspect or override it.
__all__ = [
    "bayesian_glm_lm_error_test",
    "bayesian_glm_lm_lag_test",
    "bayesian_glm_lm_wx_test",
    "_pg_mean_weight",
]
