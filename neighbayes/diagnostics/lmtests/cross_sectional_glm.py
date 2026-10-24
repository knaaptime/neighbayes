r"""Bayesian LM tests for the logit and NB families (cross-section).

The tests follow the Bayesian LM template of the Gaussian suites: posterior
draws from the restricted model, the score of the alternative at each draw,
and the information at the posterior mean of the parameters.  The family
quantities (score residual ``e``, Fisher weight ``omega``, predicted
disturbance ``h``, ``tau = Var(h)``, ``kappa = Cov(e, h)``) are defined in
:mod:`._glm`.

At the GLM null (:class:`~neighbayes.models.Logit`,
:class:`~neighbayes.models.NegBin`) the score of the latent SAR,
:math:`\eta = S(\rho)(X\beta + \varepsilon)` with :math:`\varepsilon` the
family's own disturbance, splits into a spillover and an error term,

.. math::
    s = e^\top W X\beta, \qquad q = e^\top W h,

uncorrelated with each other and with the scores of :math:`\beta` (and the
NB dispersion).  Each has its own information,

.. math::
    I_s = a^\top\Omega a - a^\top\Omega X (X^\top\Omega X)^{-1} X^\top\Omega a,
    \quad a = WX\beta, \qquad
    V_q = \sum_{i \ne j} w_{ij}^2\,\omega_i\tau_j + w_{ij}w_{ji}\,\kappa_i\kappa_j,

and the tests are

==================  ===========================================================  =====
label               statistic                                                    df
==================  ===========================================================  =====
LM-Lag              :math:`(s + q)^2/(I_s + V_q)`, the latent SAR's score test    1
LM-Error            :math:`q^2/V_q`, the latent SEM's score test                  1
Robust-LM-Lag       :math:`s^2/I_s`, the reduced-form SAR's score test            1
Robust-LM-Error     :math:`(I_s q - V_q s)^2 / ((I_s + V_q) I_s V_q)`             1
LM-SARMA            :math:`s^2/I_s + q^2/V_q`                                     2
==================  ===========================================================  =====

In the Gaussian family the five are Anselin's LM-lag, LM-error, the robust
pair and the SARMA test; ``LM-Lag + Robust-LM-Error = LM-SARMA``.

The scores are evaluated at every posterior draw, where the score of
:math:`\beta`, :math:`X^\top e`, is not zero as it is at the restricted
maximum.  A score correlated with it then carries the posterior spread of
:math:`\beta`, which the partialled information leaves out: :math:`s` does,
through the part of :math:`WX\beta` in the span of :math:`X` (the intercept's
:math:`\beta_0 W\mathbf 1`, for a row-standardized :math:`W`).  Each such score
is therefore the Neyman-orthogonal one (Doğan, Taşpınar and Bera 2021),
:math:`s - J_{s\beta}J_{\beta\beta}^{-1}X^\top e` with the information at the
posterior mean, which equals :math:`s` at the maximum and is insensitive to
:math:`\beta` to first order.  :math:`q` needs no adjustment: its cross
information with :math:`\beta` and the NB dispersion is zero.

At a reduced-form SAR null (:class:`~neighbayes.models.SARLogit`,
:class:`~neighbayes.models.SARNegBin`) the restricted linear predictor is
:math:`\eta = S(\rho)X\beta`, and the units are again independent given it.
``LM-Error`` is then the score of a latent error process added to the fitted
model, :math:`q = e^\top W h` at that :math:`\eta`; it is uncorrelated with
every score linear in :math:`e`, including those of :math:`\beta`,
:math:`\rho` and :math:`WX`, so it needs no robust adjustment.  ``LM-WX`` is
the score of :math:`WX` inside the multiplier, partialled against
:math:`\beta` as the Gaussian SAR suite's is.  Partialling :math:`\rho` as
well would leave the information nearly singular: to first order
:math:`S(\rho)X\beta = X\beta + \rho WX\beta`, so :math:`\rho` moves the
reduced form along a :math:`WX` direction, and the reduced-form SDM and SAR
are separated only by the higher-order terms.
"""

from __future__ import annotations

import weakref
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp

from ._glm import (
    _pg_mean_weight,
    _weighted_mx_cross,
    _weighted_mx_quadratic,
    _weighted_T_ww,
    glm_family,
    glm_working_response,
    information_units,
    nb_dispersion_draws,
    score_units,
)
from .core import (
    BayesianLMTestResult,
    _finalize_lm,
    _get_posterior_draws,
    _lm_scalar,
    _lm_vector,
)


def _require_W(model, test_name: str):
    if getattr(model, "_W_sparse", None) is None:
        raise ValueError(
            f"{test_name} requires the model to have been constructed with a "
            "spatial weights matrix W."
        )


def _has_lag(model) -> bool:
    """Whether the restricted model is a reduced-form SAR (its posterior has ρ)."""
    return "rho" in model.inference_data.posterior.data_vars


@dataclass
class _Scores:
    """Per-draw scores and posterior-mean information of one restricted fit.

    The score arrays carry one row per draw and a last row at the
    posterior-mean parameters, at which each statistic is evaluated.
    """

    family: str
    lag: bool
    e: np.ndarray  # (draws, n) score residuals
    q: np.ndarray  # (draws,) error scores e'Wh
    V_q: float
    s: np.ndarray | None = None  # (draws,) spillover scores (GLM null)
    I_s: float | None = None
    eta_bar: np.ndarray | None = None
    omega_bar: np.ndarray | None = None
    SWX: np.ndarray | None = None  # (draws, n, k_wx) S(rho) W X (SAR null)
    S_bar_WX: np.ndarray | None = None
    S_bar_X: np.ndarray | None = None  # S X at the posterior mean
    g_beta: np.ndarray | None = None  # (draws, k) the score of beta per draw


# The scores of one fit serve every test in its suite.
_CACHE: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()


def _error_variance(W, omega, tau, kappa) -> float:
    """``V_q``: the null variance of ``e'Wh`` for independent units."""
    W = sp.csr_matrix(W)
    return float(omega @ (W.multiply(W) @ tau) + kappa @ (W.multiply(W.T) @ kappa))


def _orthogonal(g, g_beta, B, U, omega):
    """The Neyman-orthogonal score ``g - J_{g beta} J_{beta beta}^{-1} g_beta``.

    ``B`` is the derivative of the linear predictor in ``beta`` and ``U`` in
    the tested parameters, both at the posterior mean; the information is
    their ``omega``-weighted cross products.
    """
    J_bb = B.T @ (omega[:, None] * B)
    J_bu = B.T @ (omega[:, None] * U) if U.ndim == 2 else B.T @ (omega * U)
    return g - g_beta @ np.linalg.solve(J_bb, J_bu)


def _scores(model) -> _Scores:
    """Score quantities of the restricted fit, computed once per posterior."""
    idata = model.inference_data
    cached = _CACHE.get(model)
    if cached is not None and cached[0] is idata:
        return cached[1]

    family = glm_family(model)
    y = np.asarray(model._y, dtype=np.float64)
    X = np.asarray(model._X, dtype=np.float64)
    W = sp.csr_matrix(model._W_sparse)
    beta = _get_posterior_draws(idata, "beta")
    r = r_bar = None
    if family == "negbin":
        r_draws = nb_dispersion_draws(model, idata, beta.shape[0])
        r_bar = float(r_draws.mean())
        r = np.r_[r_draws, r_bar][:, None]
    beta_bar = beta.mean(axis=0)
    beta = np.vstack([beta, beta_bar])
    n_draws = beta.shape[0]
    lag = _has_lag(model)

    if lag:
        rho = _get_posterior_draws(idata, "rho").reshape(-1)
        rho_bar = float(rho.mean())
        rho = np.r_[rho, rho_bar]
        WX = np.asarray(model._WX, dtype=np.float64)
        eye = sp.identity(W.shape[0], format="csc")
        eta = np.empty((n_draws, X.shape[0]))
        SX = np.empty((n_draws,) + X.shape)
        SWX = np.empty((n_draws, X.shape[0], WX.shape[1]))
        for d in range(n_draws):
            lu = sp.linalg.splu((eye - rho[d] * W).tocsc())
            SX[d] = lu.solve(X)
            eta[d] = SX[d] @ beta[d]
            if WX.shape[1]:
                SWX[d] = lu.solve(WX)
        lu_bar = sp.linalg.splu((eye - rho_bar * W).tocsc())
        eta_bar = lu_bar.solve(X @ beta_bar)
        S_bar_X = lu_bar.solve(X)
        S_bar_WX = lu_bar.solve(WX) if WX.shape[1] else np.empty((X.shape[0], 0))
    else:
        eta = beta @ X.T
        eta_bar = X @ beta_bar

    e, h = score_units(family, y, eta, r)
    omega, tau, kappa = information_units(family, eta_bar, r_bar)
    q = np.einsum("di,di->d", e, (W @ h.T).T)
    out = _Scores(
        family=family,
        lag=lag,
        e=e,
        q=q,
        V_q=_error_variance(W, omega, tau, kappa),
        eta_bar=eta_bar,
        omega_bar=omega,
    )
    if lag:
        out.SWX, out.S_bar_WX, out.S_bar_X = SWX, S_bar_WX, S_bar_X
        out.g_beta = np.einsum("dij,di->dj", SX, e)
    else:
        out.g_beta = e @ X
        a = (W @ eta.T).T  # W X beta per draw
        out.s = _orthogonal(
            np.einsum("di,di->d", e, a), out.g_beta, X, W @ eta_bar, omega
        )
        out.I_s = _weighted_mx_quadratic(X, W @ eta_bar, omega)
    _CACHE[model] = (idata, out)
    return out


def _null_scores(model, test_name: str) -> _Scores:
    """Scores at the GLM null; the lag tests have no meaning once ρ is fitted."""
    _require_W(model, test_name)
    sc = _scores(model)
    if sc.lag:
        raise ValueError(
            f"{test_name} tests ρ = 0, which {type(model).__name__} has "
            "estimated; run it on the aspatial fit."
        )
    return sc


def _details(model, sc: _Scores, **extra) -> dict:
    return {
        "model_type": getattr(model, "_model_type", None),
        "family": sc.family,
        "null": "reduced-form SAR" if sc.lag else "GLM",
        **extra,
    }


def bayesian_glm_lm_lag_test(model) -> BayesianLMTestResult:
    r"""LM test for the latent SAR at the GLM null.

    :math:`(s + q)^2 / (I_s + V_q)`, the score test of
    :math:`\eta = S(\rho)(X\beta + \varepsilon)` against the logit or NB fit:
    the spillover and the error terms together, in the proportion one
    :math:`\rho` implies (see the module).

    Returns
    -------
    BayesianLMTestResult
        Per-draw LM samples, summary statistics, ``df = 1``.
    """
    sc = _null_scores(model, "bayesian_glm_lm_lag_test")
    score = sc.s + sc.q
    return _lm_scalar(
        score[:-1],
        sc.I_s + sc.V_q,
        test_type="bayesian_glm_lm_lag",
        df=1,
        details=_details(model, sc, s_mean=float(sc.s[-1]), q_mean=float(sc.q[-1])),
        score_bar=score[-1],
    )


def bayesian_glm_lm_error_test(model) -> BayesianLMTestResult:
    r"""LM test for a latent error process.

    :math:`q^2 / V_q` with :math:`q = e^\top W h`: at the GLM null the score
    test of the latent SEM, :math:`\eta = X\beta + S(\lambda)\varepsilon`;
    at a reduced-form SAR null, of a latent error process added to the
    fitted model (see the module).

    Returns
    -------
    BayesianLMTestResult
        Per-draw LM samples, summary statistics, ``df = 1``.
    """
    _require_W(model, "bayesian_glm_lm_error_test")
    sc = _scores(model)
    return _lm_scalar(
        sc.q[:-1],
        sc.V_q,
        test_type="bayesian_glm_lm_error",
        df=1,
        details=_details(model, sc, q_mean=float(sc.q[-1])),
        score_bar=sc.q[-1],
    )


def bayesian_glm_robust_lm_lag_test(model) -> BayesianLMTestResult:
    r"""Robust LM test for the spillover at the GLM null.

    :math:`s^2 / I_s` with :math:`s = e^\top WX\beta`: the score test of the
    reduced-form SAR, :math:`\eta = S(\rho)X\beta`, and the Bera–Yoon robust
    lag test of the latent SAR, :math:`s` being uncorrelated with the error
    score.

    Returns
    -------
    BayesianLMTestResult
        Per-draw LM samples, summary statistics, ``df = 1``.
    """
    sc = _null_scores(model, "bayesian_glm_robust_lm_lag_test")
    return _lm_scalar(
        sc.s[:-1],
        sc.I_s,
        test_type="bayesian_glm_robust_lm_lag",
        df=1,
        details=_details(model, sc, s_mean=float(sc.s[-1])),
        score_bar=sc.s[-1],
    )


def bayesian_glm_robust_lm_error_test(model) -> BayesianLMTestResult:
    r"""Robust LM test for the error term at the GLM null.

    :math:`(I_s q - V_q s)^2 / ((I_s + V_q) I_s V_q)`: the error score purged
    of the spillover that one :math:`\rho` would carry with it, so that it
    detects error dependence the latent SAR does not explain.

    Returns
    -------
    BayesianLMTestResult
        Per-draw LM samples, summary statistics, ``df = 1``.
    """
    sc = _null_scores(model, "bayesian_glm_robust_lm_error_test")
    I_s, V_q = sc.I_s, sc.V_q
    norm = (I_s + V_q) * I_s * V_q + 1e-12
    LM = (I_s * sc.q - V_q * sc.s) ** 2 / norm
    return _finalize_lm(
        LM[:-1],
        test_type="bayesian_glm_robust_lm_error",
        df=1,
        details=_details(model, sc),
        statistic=LM[-1],
    )


def bayesian_glm_lm_sarma_test(model) -> BayesianLMTestResult:
    r"""Joint LM test for spillover and error at the GLM null.

    :math:`s^2/I_s + q^2/V_q`, the two uncorrelated scores together.

    Returns
    -------
    BayesianLMTestResult
        Per-draw LM samples, summary statistics, ``df = 2``.
    """
    sc = _null_scores(model, "bayesian_glm_lm_sarma_test")
    LM = sc.s**2 / (sc.I_s + 1e-12) + sc.q**2 / (sc.V_q + 1e-12)
    return _finalize_lm(
        LM[:-1],
        test_type="bayesian_glm_lm_sarma",
        df=2,
        details=_details(model, sc),
        statistic=LM[-1],
    )


def bayesian_glm_lm_wx_test(model) -> BayesianLMTestResult:
    r"""LM test for omitted :math:`WX` (H₀: θ = 0).

    At the GLM null the score is :math:`(WX)^\top e`; at a reduced-form SAR
    null, :math:`WX` enters inside the multiplier, :math:`\eta =
    S(\rho)(X\beta + WX\theta)`, and the score is :math:`(S(\rho)WX)^\top e`.
    Either is the Neyman-orthogonal score against :math:`\beta`, its
    information partialled likewise (see the module, also on :math:`\rho`),
    with the Fisher weights at the posterior mean.

    Returns
    -------
    BayesianLMTestResult
        Per-draw LM samples, summary statistics, ``df = k_wx``.
    """
    _require_W(model, "bayesian_glm_lm_wx_test")
    WX = np.asarray(model._WX, dtype=np.float64)
    k_wx = WX.shape[1]
    if k_wx == 0:
        raise ValueError(
            "Model has no WX columns. The WX test requires at least one "
            "spatially lagged covariate (set ``durbin_vars`` at construction)."
        )
    sc = _scores(model)
    if sc.lag:
        B, U = sc.S_bar_X, sc.S_bar_WX
        g = np.einsum("dij,di->dj", sc.SWX, sc.e)
    else:
        B, U = np.asarray(model._X, dtype=np.float64), WX
        g = sc.e @ WX
    g = _orthogonal(g, sc.g_beta, B, U, sc.omega_bar)
    V = _weighted_mx_cross(B, U, U, sc.omega_bar)
    return _lm_vector(
        g[:-1],
        V,
        score_bar=g[-1],
        test_type="bayesian_glm_lm_wx",
        df=k_wx,
        details=_details(model, sc, k_wx=k_wx),
        label="V_theta_theta (GLM LM-WX)",
    )


def bayesian_glm_moran_working_test(model) -> BayesianLMTestResult:
    r"""Moran diagnostic on the Pólya–Gamma working residuals at the GLM null.

    With :math:`\tilde e = \tilde z - X\beta` the working residual and
    :math:`\Omega` the PG mean weights (:mod:`._glm`),

    .. math::
        S^{(d)} = \tilde e^{(d)\top}\, \Omega^{(d)}\, W\, \tilde e^{(d)},
        \qquad
        V = \mathrm{tr}(W^2) + \sum_{i,j} \bar\omega_i/\bar\omega_j\, W_{ij}^2,

    the exact null variance of the statistic given the weights.  It pairs the
    score residual with the working residual, where the latent SEM's score
    pairs it with the predicted disturbance, so it is a residual diagnostic,
    not the score test of a fitted model (``LM-Error`` is that).

    Returns
    -------
    BayesianLMTestResult
        Per-draw samples, summary statistics, ``df = 1``.
    """
    _require_W(model, "bayesian_glm_moran_working_test")
    if _has_lag(model):
        raise ValueError(
            "bayesian_glm_moran_working_test uses the GLM's working residuals; "
            f"run it on the aspatial fit, not {type(model).__name__}."
        )
    idata = model.inference_data
    beta_draws = _get_posterior_draws(idata, "beta")
    _, e_tilde, omega = glm_working_response(model, beta_draws, idata)
    W = model._W_sparse
    We = (W @ e_tilde.T).T
    S = np.einsum("di,di,di->d", e_tilde, omega, We)
    V = _weighted_T_ww(W, omega.mean(axis=0))
    return _lm_scalar(
        S,
        V,
        test_type="bayesian_glm_moran_working",
        df=1,
        details={"model_type": getattr(model, "_model_type", None)},
    )


__all__ = [
    "bayesian_glm_lm_lag_test",
    "bayesian_glm_lm_error_test",
    "bayesian_glm_robust_lm_lag_test",
    "bayesian_glm_robust_lm_error_test",
    "bayesian_glm_lm_sarma_test",
    "bayesian_glm_lm_wx_test",
    "bayesian_glm_moran_working_test",
    "_pg_mean_weight",
]
