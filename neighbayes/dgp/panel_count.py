r"""NB2 count panel DGP with unit and period effects.

.. math::

    \eta_t = (I - \rho W)^{-1} X_t \beta + c + \tau_t \mathbf 1, \qquad
    y_{it} \sim \mathrm{NB2}(e^{\eta_{it}}, \alpha),

stacked time-first, matching :class:`~neighbayes.models.SARNegBinPanel`
(``rho = 0`` gives :class:`~neighbayes.models.NegBinPanel`).
"""

from __future__ import annotations

import warnings

import numpy as np

from .panel_fe import _panel_finalize
from .utils import ensure_rng, resolve_weights, spatial_filter_factor


def _simulate_panel_count(
    N: int,
    T: int,
    rho: float,
    beta,
    alpha: float,
    unit_effect_sd: float,
    time_effect_sd: float,
    fe_x_corr: float,
    err_hetero: bool,
    rng,
    seed,
    W,
    gdf,
    n_side,
    contiguity,
) -> dict:
    if not 0.0 <= fe_x_corr < 1.0:
        raise ValueError("fe_x_corr must lie in [0, 1).")
    if err_hetero:
        warnings.warn(
            "err_hetero does not apply to count DGPs and is ignored.", stacklevel=3
        )
    rng = ensure_rng(rng, seed)
    Ws, Wg = resolve_weights(W=W, gdf=gdf, n_side=n_side, contiguity=contiguity)
    if Ws.shape[0] != N:
        raise ValueError("N must match W/gdf unit count.")
    beta = np.asarray([0.5, 0.6] if beta is None else beta, dtype=float)
    k = len(beta) - 1

    # A persistent unit component in the regressors that the unit effects
    # load on: pooled estimates are then biased, so fixed-effects recovery
    # tests have power.
    u = rng.standard_normal((N, max(k, 1)))
    e = rng.standard_normal(N)
    unit_effect = unit_effect_sd * (
        fe_x_corr * u[:, 0] + np.sqrt(1.0 - fe_x_corr**2) * e
    )
    time_effect = np.zeros(T)
    if T > 1:
        time_effect[1:] = time_effect_sd * rng.standard_normal(T - 1)

    solve = spatial_filter_factor(Ws, rho)
    keep = np.sqrt(1.0 - fe_x_corr**2)
    y_list, X_list, mu_list = [], [], []
    for t in range(T):
        Z = fe_x_corr * u[:, :k] + keep * rng.standard_normal((N, k))
        X_t = np.column_stack([np.ones(N), Z])
        eta = solve(X_t @ beta) + unit_effect + time_effect[t]
        mu = np.exp(np.clip(eta, -30.0, 30.0))
        y_t = rng.negative_binomial(alpha, alpha / (alpha + mu))
        y_list.append(y_t.astype(np.int64))
        X_list.append(X_t)
        mu_list.append(mu)
    y, X, idx = _panel_finalize(y_list, X_list, N, T)
    params = {
        "rho": rho,
        "beta": beta,
        "alpha": float(alpha),
        "unit_effect": unit_effect,
        "time_effect": time_effect,
    }
    return {
        "y": y,
        "X": X,
        "mu": np.concatenate(mu_list),
        "unit": idx["unit"],
        "time": idx["time"],
        "W_sparse": Ws,
        "W_graph": Wg,
        "params_true": params,
    }


def simulate_panel_sar_negbin(
    N: int,
    T: int,
    rho: float = 0.4,
    beta: np.ndarray | None = None,
    alpha: float = 2.0,
    unit_effect_sd: float = 0.0,
    time_effect_sd: float = 0.0,
    fe_x_corr: float = 0.0,
    err_hetero: bool = False,
    rng: np.random.Generator | None = None,
    seed: int | None = None,
    W=None,
    gdf=None,
    n_side: int | None = None,
    contiguity: str = "queen",
) -> dict:
    r"""Simulate a spatial-lag NB2 count panel, stacked time-first.

    Parameters
    ----------
    N, T : int
        Units and periods.
    rho : float, default 0.4
        Spatial parameter of the log-mean filter (``0`` for the aspatial model).
    beta : array-like, optional
        Coefficients including the intercept; default ``[0.5, 0.6]``.
    alpha : float, default 2.0
        NB2 dispersion.
    unit_effect_sd, time_effect_sd : float, default 0.0
        Standard deviations of the unit effects ``c_i`` and the period
        effects ``τ_t`` (``τ_0 = 0``), both outside the filter.
    fe_x_corr : float, default 0.0
        Correlation, in ``[0, 1)``, between the unit effects and the persistent
        part of the first regressor.
    err_hetero : bool, default False
        Accepted for API parity; the count variance is set by the mean (and
        ``alpha``), so it is ignored with a warning.
    rng, seed, W, gdf, n_side, contiguity
        As for :func:`~neighbayes.dgp.simulate_panel_sar_fe`.

    Returns
    -------
    dict
        ``y`` (int counts), ``X`` (with intercept), ``mu``, ``unit``, ``time``,
        ``W_sparse``, ``W_graph`` and ``params_true`` (including the effects).
    """
    if alpha <= 0:
        raise ValueError("alpha must be strictly positive.")
    return _simulate_panel_count(
        N, T, rho, beta, alpha, unit_effect_sd, time_effect_sd,
        fe_x_corr, err_hetero, rng, seed, W, gdf, n_side, contiguity,
    )  # fmt: skip


def simulate_panel_sar_zinb(
    N: int,
    T: int,
    rho: float = 0.4,
    lam: float = 0.3,
    beta: np.ndarray | None = None,
    gamma: np.ndarray | None = None,
    alpha: float = 2.0,
    unit_effect_sd: float = 0.0,
    time_effect_sd: float = 0.0,
    fe_x_corr: float = 0.0,
    err_hetero: bool = False,
    rng: np.random.Generator | None = None,
    seed: int | None = None,
    W=None,
    W_sel=None,
    gdf=None,
    n_side: int | None = None,
    contiguity: str = "queen",
) -> dict:
    r"""Simulate a zero-inflated SAR-NB panel with per-period structural zeros.

    The count equation is :func:`simulate_panel_sar_negbin` (including its
    unit- and period-effect knobs); the selection equation is

    .. math::

        \eta^{\mathrm{sel}}_t = (I - \lambda W_{\mathrm{sel}})^{-1} Z_t\gamma,
        \qquad d_{it} \sim \mathrm{Bernoulli}(\mathrm{logit}^{-1}(\eta^{\mathrm{sel}}_{it})),

    with a fresh selection design ``Z_t`` (intercept plus standard-normal
    columns) each period, and ``y = d · y_NB``.  Matches
    :class:`~neighbayes.models.SARZINBPanel` (``rho = lam = 0`` gives
    :class:`~neighbayes.models.ZINBPanel`).

    Parameters
    ----------
    lam : float, default 0.3
        Selection spatial parameter.
    gamma : array-like, optional
        Selection coefficients including the intercept; default ``[0.5, 1.0]``.
    W_sel : Graph or scipy.sparse matrix, optional
        Selection weights; default the count weights.
    N, T, rho, beta, alpha, unit_effect_sd, time_effect_sd, fe_x_corr, \
    err_hetero, rng, seed, W, gdf, n_side, contiguity
        As for :func:`simulate_panel_sar_negbin`.

    Returns
    -------
    dict
        As :func:`simulate_panel_sar_negbin`, with ``y`` zero-inflated, plus
        ``Z``, ``d`` (the activity indicators), ``eta_sel`` and ``W_sel_sparse``;
        ``params_true`` gains ``lam`` and ``gamma``.
    """
    rng = ensure_rng(rng, seed)
    out = _simulate_panel_count(
        N, T, rho, beta, alpha, unit_effect_sd, time_effect_sd,
        fe_x_corr, err_hetero, rng, None, W, gdf, n_side, contiguity,
    )  # fmt: skip
    gamma = np.asarray([0.5, 1.0] if gamma is None else gamma, dtype=float)
    if W_sel is not None:
        Ws_sel, _ = resolve_weights(
            W=W_sel, gdf=gdf, n_side=n_side, contiguity=contiguity
        )
    else:
        Ws_sel = out["W_sparse"]
    solve = spatial_filter_factor(Ws_sel, lam)
    Z_list, eta_list = [], []
    for _ in range(T):
        Z_t = np.column_stack([np.ones(N), rng.standard_normal((N, len(gamma) - 1))])
        Z_list.append(Z_t)
        eta_list.append(solve(Z_t @ gamma))
    Z = np.vstack(Z_list)
    eta_sel = np.concatenate(eta_list)
    d = rng.binomial(1, 1.0 / (1.0 + np.exp(-eta_sel)))
    out["y"] = (out["y"] * d).astype(np.int64)
    out.update(Z=Z, d=d, eta_sel=eta_sel, W_sel_sparse=Ws_sel)
    out["params_true"].update(lam=lam, gamma=gamma)
    return out
