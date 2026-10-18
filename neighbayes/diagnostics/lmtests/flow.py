"""Bayesian LM diagnostic tests — flow variants.

All public test functions return :class:`BayesianLMTestResult`.
"""

import numpy as np

from neighbayes.diagnostics.lmtests.core import (
    BayesianLMTestResult,
    _compute_residuals,
    _finalize_lm,
    _get_posterior_draws,
    _lm_vector,
    _neyman_adjust_scalar,
    _posterior_mean_sigma2,
    _safe_inv,
)


def _flow_score_info(model, *, restrict_keys=("d", "o", "w"), T: int | None = None):
    """Return per-draw score matrix and the (3,3) information matrix.

    Parameters
    ----------
    model : OLSFlow or OLSFlowPanelFE
        Fitted flow model providing ``inference_data`` (with ``beta`` and
        ``sigma`` posterior draws), the cached spatial lags
        ``_Wd_y``/``_Wo_y``/``_Ww_y``, the cached trace matrix
        ``_T_flow_traces``, and the design matrix ``_X``.
    restrict_keys : tuple of {"d","o","w"}
        Subset of the three spatial-lag directions to keep (in order).
        Mainly for test readability — the joint test uses the full triple.
    T : int, optional
        Number of panel periods.  When ``None`` (default), inferred from
        ``getattr(model, "_T", 1)``.  The information matrix scales the
        Kronecker trace block by ``T`` to reflect i.i.d. within-period
        contributions under :math:`H_0`.

    Returns
    -------
    G : np.ndarray, shape (n_draws, k)
        Score draws, ``G[g, i] = (W_i y)^T e_g``.
    J : np.ndarray, shape (k, k)
        Information matrix
        ``J = T * T_flow_traces[K, K] * sigma2_bar + Q[K, K]``
        with ``Q[i, j] = (W_i y)^T (W_j y)``.
    """
    if T is None:
        T = int(getattr(model, "_T", 1))
    np.asarray(model._y, dtype=np.float64)
    np.asarray(model._X, dtype=np.float64)
    Wy_all = np.column_stack(
        [
            np.asarray(model._Wd_y, dtype=np.float64),
            np.asarray(model._Wo_y, dtype=np.float64),
            np.asarray(model._Ww_y, dtype=np.float64),
        ]
    )  # (N, 3)
    idx = {"d": 0, "o": 1, "w": 2}
    cols = np.array([idx[k] for k in restrict_keys], dtype=int)
    Wy = Wy_all[:, cols]  # (N, k)

    idata = model.inference_data
    beta_draws = _get_posterior_draws(idata, "beta")  # (n_draws, p)
    _, sigma2_mean = _posterior_mean_sigma2(idata)

    resid = _compute_residuals(model, beta_draws)

    G = resid @ Wy  # (n_draws, k)

    Q = Wy.T @ Wy  # (k, k)
    T_blk = model._T_flow_traces[np.ix_(cols, cols)]
    J = T * T_blk * sigma2_mean + Q  # (k, k)

    return G, J


def _flow_marginal_lm(model, key: str, test_type: str) -> BayesianLMTestResult:
    """Helper for the three marginal LM tests (dest / orig / network)."""
    G, J = _flow_score_info(model, restrict_keys=(key,))
    s = G[:, 0]
    v = float(J[0, 0])
    LM = s**2 / (v + 1e-12)
    return _finalize_lm(
        LM,
        test_type=test_type,
        df=1,
        details={"direction": key},
    )


def bayesian_lm_flow_destination_test(model) -> BayesianLMTestResult:
    r"""Bayesian LM test for an omitted destination-side spatial lag.

    Tests :math:`H_0\colon \rho_d = 0` against the SARFlow alternative,
    using a fitted :class:`~neighbayes.models.flow.OLSFlow` as the null
    model.  For each posterior draw :math:`g` the residual is
    :math:`e_g = y - X \beta_g`; the score is
    :math:`s_g = (W_d y)^\top e_g` and the information-matrix denominator
    is

    .. math::

        V = \operatorname{tr}(W_d^\top W_d + W_d^2)\,\bar\sigma^{2}
            + \|W_d y\|^{2},

    with :math:`\bar\sigma^{2}` the posterior mean of :math:`\sigma^{2}`.
    The Bayesian p-value is :math:`1 - F_{\chi^{2}_{1}}(\overline{LM})`.

    The score / information construction follows the gravity-flow
    spatial-econometrics framework of :cite:t:`lesage2008SpatialEconometric`
    and :cite:t:`lesage2009IntroductionSpatial`. The Bayesian LM statistic
    is computed per posterior draw following :cite:t:`dogan2021BayesianRobust`.
    """
    return _flow_marginal_lm(model, "d", "bayesian_lm_flow_destination")


def bayesian_lm_flow_origin_test(model) -> BayesianLMTestResult:
    r"""Bayesian LM test for an omitted origin-side spatial lag
    (:math:`H_0\colon \rho_o = 0`).  See :func:`bayesian_lm_flow_destination_test`
    for details and references (:cite:p:`lesage2008SpatialEconometric`,
    :cite:p:`dogan2021BayesianRobust`)."""
    return _flow_marginal_lm(model, "o", "bayesian_lm_flow_origin")


def bayesian_lm_flow_od_test(model) -> BayesianLMTestResult:
    r"""Bayesian LM test for an omitted network spatial lag
    (:math:`H_0\colon \rho_w = 0`).  See :func:`bayesian_lm_flow_destination_test`
    for details and references (:cite:p:`lesage2008SpatialEconometric`,
    :cite:p:`dogan2021BayesianRobust`)."""
    return _flow_marginal_lm(model, "w", "bayesian_lm_flow_od")


def bayesian_lm_flow_joint_test(model) -> BayesianLMTestResult:
    r"""Joint Bayesian LM test for the SARFlow filter
    (:math:`H_0\colon \rho_d = \rho_o = \rho_w = 0`).

    For each posterior draw :math:`g` of the :class:`OLSFlow` null,
    builds the score vector
    :math:`g_g = ((W_d y)^\top e_g, (W_o y)^\top e_g, (W_w y)^\top e_g)^\top`
    and the information matrix

    .. math::

        J = T_{\text{flow}}\,\bar\sigma^{2} + Q, \qquad
        Q_{ij} = (W_i y)^\top (W_j y),

    where :math:`T_{\text{flow}}` is the cached :math:`3\times 3` Kronecker
    trace matrix from :func:`neighbayes.graph.flow_trace_blocks`.  The
    statistic is :math:`LM_g = g_g^\top J^{-1} g_g`, distributed
    :math:`\chi^{2}_3` under :math:`H_0`. The construction follows the
    gravity-flow spatial-econometrics framework of
    :cite:t:`lesage2008SpatialEconometric` and
    :cite:t:`lesage2009IntroductionSpatial`; the Bayesian LM statistic is
    computed per posterior draw following :cite:t:`dogan2021BayesianRobust`.
    """
    G, J = _flow_score_info(model, restrict_keys=("d", "o", "w"))
    J_inv = _safe_inv(J, "J (flow joint LM)")
    LM = np.einsum("di,ij,dj->d", G, J_inv, G)
    return _finalize_lm(LM, test_type="bayesian_lm_flow_joint", df=3)


# ---------------------------------------------------------------------
# Robust (Neyman-orthogonal) variants — alternative model is a fitted
# SARFlow.  For each posterior draw g of (rho_d, rho_o, rho_w, beta,
# sigma) we build the spatially-filtered residual
#     e_g = y - rho_d^g W_d y - rho_o^g W_o y - rho_w^g W_w y - X beta_g
# and form the full 3-vector score g_g = (W_d y, W_o y, W_w y)^T e_g.
# The information matrix uses the SARFlow posterior mean of sigma^2.
# Marginal robust tests then apply the Neyman adjustment with respect to
# the two nuisance directions.
# ---------------------------------------------------------------------


def _flow_robust_score_info(sarflow_model):
    """Return per-draw 3-vector score and the (3,3) information matrix
    for a fitted :class:`SARFlow`.

    The score is evaluated at the SARFlow posterior draws of
    ``(rho_d, rho_o, rho_w, beta)``; the information matrix uses the
    SARFlow posterior mean of :math:`\\sigma^{2}`.
    """
    y = np.asarray(sarflow_model._y, dtype=np.float64)
    X = np.asarray(sarflow_model._X, dtype=np.float64)
    Wd_y = np.asarray(sarflow_model._Wd_y, dtype=np.float64)
    Wo_y = np.asarray(sarflow_model._Wo_y, dtype=np.float64)
    Ww_y = np.asarray(sarflow_model._Ww_y, dtype=np.float64)
    Wy = np.column_stack([Wd_y, Wo_y, Ww_y])  # (N, 3)

    idata = sarflow_model.inference_data
    beta_draws = _get_posterior_draws(idata, "beta")
    rho_d_draws = _get_posterior_draws(idata, "rho_d").reshape(-1)
    rho_o_draws = _get_posterior_draws(idata, "rho_o").reshape(-1)
    rho_w_draws = _get_posterior_draws(idata, "rho_w").reshape(-1)
    _, sigma2_mean = _posterior_mean_sigma2(idata)

    fitted = beta_draws @ X.T  # (n_draws, N)
    rho_stack = np.column_stack([rho_d_draws, rho_o_draws, rho_w_draws])  # (n,3)
    spatial = rho_stack @ Wy.T  # (n_draws, N)
    resid = y[None, :] - spatial - fitted

    G = resid @ Wy  # (n_draws, 3)

    Q = Wy.T @ Wy
    J = sarflow_model._T_flow_traces * sigma2_mean + Q

    return G, J


def _flow_robust_marginal_lm(
    sarflow_model, target: int, test_type: str
) -> BayesianLMTestResult:
    """Neyman-orthogonal LM for a single direction with two nuisances."""
    G, J = _flow_robust_score_info(sarflow_model)
    nuisance = [i for i in (0, 1, 2) if i != target]
    g_t = G[:, target]
    g_n = G[:, nuisance]  # (n_draws, 2)
    J_tt = float(J[target, target])
    J_tn = J[target, nuisance]  # (2,)
    J_nn = J[np.ix_(nuisance, nuisance)]  # (2, 2)

    # adjusted score: g_t* = g_t - J_{t,n} J_{n,n}^{-1} g_n
    g_star, V_star = _neyman_adjust_scalar(
        g_t,
        g_n,
        J_tt,
        J_tn,
        J_nn,
        label=f"J_nuisance (flow robust {test_type})",
    )
    LM = g_star**2 / (V_star + 1e-12)

    direction = {0: "d", 1: "o", 2: "w"}[target]
    return _finalize_lm(
        LM,
        test_type=test_type,
        df=1,
        details={"direction": direction},
    )


def bayesian_robust_lm_flow_destination_test(sarflow_model) -> BayesianLMTestResult:
    r"""Robust Bayesian LM test for :math:`\rho_d` adjusting for
    :math:`(\rho_o, \rho_w)` nuisance via the Neyman-orthogonal score
    (:cite:t:`bera1993SpecificationTesting`, :cite:t:`anselin1996SimpleDiagnostic`,
    :cite:t:`dogan2021BayesianRobust`).  Requires a fitted
    :class:`~neighbayes.models.flow.SARFlow`."""
    return _flow_robust_marginal_lm(
        sarflow_model, 0, "bayesian_robust_lm_flow_destination"
    )


def bayesian_robust_lm_flow_origin_test(sarflow_model) -> BayesianLMTestResult:
    r"""Robust Bayesian LM test for :math:`\rho_o` adjusting for
    :math:`(\rho_d, \rho_w)` nuisance.  See
    :func:`bayesian_robust_lm_flow_destination_test` for details and references
    (:cite:p:`bera1993SpecificationTesting`,
    :cite:p:`anselin1996SimpleDiagnostic`,
    :cite:p:`dogan2021BayesianRobust`)."""
    return _flow_robust_marginal_lm(sarflow_model, 1, "bayesian_robust_lm_flow_origin")


def bayesian_robust_lm_flow_od_test(sarflow_model) -> BayesianLMTestResult:
    r"""Robust Bayesian LM test for :math:`\rho_w` adjusting for
    :math:`(\rho_d, \rho_o)` nuisance.  See
    :func:`bayesian_robust_lm_flow_destination_test` for details and references
    (:cite:p:`bera1993SpecificationTesting`,
    :cite:p:`anselin1996SimpleDiagnostic`,
    :cite:p:`dogan2021BayesianRobust`)."""
    return _flow_robust_marginal_lm(sarflow_model, 2, "bayesian_robust_lm_flow_od")


# ---------------------------------------------------------------------
# Phase 5 — WX-style test for the intra-block in OLSFlow.
# ---------------------------------------------------------------------


def bayesian_lm_flow_intraunit_test(model) -> BayesianLMTestResult:
    r"""Bayesian LM (WX-style) test for the intra block in an OLSFlow null.

    Tests :math:`H_0\colon \beta_{\text{intra}} = 0` in the gravity model

    .. math::

        y = \alpha + g \gamma + X_d \beta_d + X_o \beta_o
            + X_{\text{intra}} \beta_{\text{intra}} + \varepsilon,

    using a fitted :class:`OLSFlow` whose design already contains the
    intra block (the standard layout from
    :func:`neighbayes.graph.flow_design_matrix`).  Per posterior draw the
    intra-block score is :math:`g_g = X_{\text{intra}}^\top e_g` with
    :math:`e_g = y - X \beta_g` (full-design residual) and the variance
    is :math:`V = X_{\text{intra}}^\top X_{\text{intra}} / \bar\sigma^{2}`,
    matching the SDM-style WX information block of
    :cite:t:`dogan2021BayesianRobust`.  The statistic is
    :math:`LM_g = g_g^\top V^{-1} g_g`, df :math:`= k_{\text{intra}}`.

    Notes
    -----
    The test treats the intra block as the "WX" addition relative to a
    null with the same dest/orig coefficients but no intra columns.  The
    score is evaluated at the OLSFlow draws — this is asymptotically
    equivalent to refitting an "OLSFlow without intra" null when the
    intra block is orthogonal to ``X_dest``/``X_orig`` (typically the
    case for the diagonal-mask layout used by ``flow_design_matrix``).
    """
    intra_idx = getattr(model, "_intra_idx", None)
    if intra_idx is None or len(intra_idx) == 0:
        raise ValueError(
            "Model has no intra columns to test.  Build the design with "
            "flow_design_matrix(..., include_intra=True) before fitting."
        )
    intra_idx = np.asarray(intra_idx, dtype=int)

    np.asarray(model._y, dtype=np.float64)
    X = np.asarray(model._X, dtype=np.float64)
    Xi = X[:, intra_idx]  # (N, k_intra)
    k_intra = Xi.shape[1]

    idata = model.inference_data
    beta_draws = _get_posterior_draws(idata, "beta")  # (n_draws, p)

    resid = _compute_residuals(model, beta_draws)
    G = resid @ Xi  # (n_draws, k_intra)

    _, sigma2_mean = _posterior_mean_sigma2(idata)
    V = (Xi.T @ Xi) / sigma2_mean

    return _lm_vector(
        G,
        V,
        test_type="bayesian_lm_flow_intraunit",
        df=k_intra,
        details={"k_intra": k_intra},
        label="V_intra (flow intra LM)",
    )


# ---------------------------------------------------------------------
# Phase 6 — Panel analogues.  Operate on demeaned (_y, _X) panel stacks
# of length N_flow * T from FlowPanelModel.  The information matrix
# stays additive in T because Q[i,j] = (W_i y)^T (W_j y) is computed on
# the full stacked panel lags (already cached as _Wd_y/_Wo_y/_Ww_y on
# the demeaned panel).
# ---------------------------------------------------------------------
