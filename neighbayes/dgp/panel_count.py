r"""NB2 count panel DGP with random unit effects and period effects.

.. math::

    \eta_t = (I - \rho W)^{-1} (X_t \beta + c) + \tau_t \mathbf 1, \qquad
    y_{it} \sim \mathrm{NB2}(e^{\eta_{it}}, \alpha),

with the unit effects ``c`` inside the filter, as random effects enter a
spatial equation,

stacked time-first, matching :class:`~neighbayes.models.SARNegBinPanelRE`
(``rho = 0`` gives :class:`~neighbayes.models.NegBinPanelRE`).
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
        eta = solve(X_t @ beta + unit_effect) + time_effect[t]
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


def _simulate_sar_negbin_panel_re(
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
        Standard deviations of the unit effects ``c_i``, inside the filter,
        and the period effects ``τ_t`` (``τ_0 = 0``), outside it.
    fe_x_corr : float, default 0.0
        Correlation, in ``[0, 1)``, between the unit effects and the persistent
        part of the first regressor.
    err_hetero : bool, default False
        Accepted for API parity; the count variance is set by the mean (and
        ``alpha``), so it is ignored with a warning.
    rng, seed, W, gdf, n_side, contiguity
        As for :func:`~neighbayes.dgp._simulate_sar_panel_fe`.

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


def _simulate_sar_zinb_panel_re(
    N: int,
    T: int,
    rho: float = 0.4,
    rho_zero: float = 0.3,
    beta: np.ndarray | None = None,
    beta_zero: np.ndarray | None = None,
    alpha: float = 2.0,
    unit_effect_sd: float = 0.0,
    time_effect_sd: float = 0.0,
    fe_x_corr: float = 0.0,
    err_hetero: bool = False,
    rng: np.random.Generator | None = None,
    seed: int | None = None,
    W=None,
    W_zero=None,
    gdf=None,
    n_side: int | None = None,
    contiguity: str = "queen",
) -> dict:
    r"""Simulate a zero-inflated SAR-NB panel with per-period structural zeros.

    The count part is :func:`_simulate_sar_negbin_panel_re` (including its
    unit- and period-effect knobs); the zero part is

    .. math::

        \eta_{z,t} = (I - \rho_z W_z)^{-1} X_{z,t}\beta_z,
        \qquad d_{it} \sim \mathrm{Bernoulli}(\mathrm{logit}^{-1}(\eta_{z,it})),

    with a fresh zero-part design ``X_{z,t}`` (intercept plus standard-normal
    columns) each period, and ``y = d · y_NB``.  Matches
    :class:`~neighbayes.models.SARZINB` panels (``rho = rho_zero = 0`` gives
    the aspatial :class:`~neighbayes.models.ZINB`).

    Parameters
    ----------
    rho_zero : float, default 0.3
        Zero-part spatial parameter.
    beta_zero : array-like, optional
        Zero-part coefficients including the intercept; default ``[0.5, 1.0]``.
    W_zero : Graph or scipy.sparse matrix, optional
        Zero-part weights; default the count weights.
    N, T, rho, beta, alpha, unit_effect_sd, time_effect_sd, fe_x_corr, \
    err_hetero, rng, seed, W, gdf, n_side, contiguity
        As for :func:`_simulate_sar_negbin_panel_re`.

    Returns
    -------
    dict
        As :func:`_simulate_sar_negbin_panel_re`, with ``y`` zero-inflated, plus
        ``X_zero``, ``d`` (the activity indicators), ``eta_zero`` and
        ``W_zero_sparse``; ``params_true`` gains ``rho_zero`` and ``beta_zero``.
    """
    rng = ensure_rng(rng, seed)
    out = _simulate_panel_count(
        N, T, rho, beta, alpha, unit_effect_sd, time_effect_sd,
        fe_x_corr, err_hetero, rng, None, W, gdf, n_side, contiguity,
    )  # fmt: skip
    beta_zero = np.asarray([0.5, 1.0] if beta_zero is None else beta_zero, dtype=float)
    if W_zero is not None:
        Ws_sel, _ = resolve_weights(
            W=W_zero, gdf=gdf, n_side=n_side, contiguity=contiguity
        )
    else:
        Ws_sel = out["W_sparse"]
    solve = spatial_filter_factor(Ws_sel, rho_zero)
    Z_list, eta_list = [], []
    for _ in range(T):
        Z_t = np.column_stack(
            [np.ones(N), rng.standard_normal((N, len(beta_zero) - 1))]
        )
        Z_list.append(Z_t)
        eta_list.append(solve(Z_t @ beta_zero))
    X_zero = np.vstack(Z_list)
    eta_sel = np.concatenate(eta_list)
    d = rng.binomial(1, 1.0 / (1.0 + np.exp(-eta_sel)))
    out["y"] = (out["y"] * d).astype(np.int64)
    out.update(X_zero=X_zero, d=d, eta_zero=eta_sel, W_zero_sparse=Ws_sel)
    out["params_true"].update(rho_zero=rho_zero, beta_zero=beta_zero)
    return out
