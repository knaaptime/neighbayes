r"""Hurdle NB DGPs: a reduced-form spatial logit for ``y > 0`` and a truncated NB.

.. math::

    P(y > 0) = \mathrm{logit}^{-1}(\eta^{\mathrm{b}}), \qquad
    y \mid y > 0 \sim \mathrm{NB2}(e^{\eta^{\mathrm{c}}}, \alpha)
    \text{ truncated at } 0,

matching :class:`~neighbayes.models.SARHurdleNB` and the hurdle panels.
"""

from __future__ import annotations

import warnings

import numpy as np

from .cross_sectional import _attach_optional_gdf, _check_rho_stability
from .panel_count import _simulate_panel_count
from .utils import (
    ensure_rng,
    make_design_matrix,
    resolve_weights,
    spatial_filter_factor,
)


def truncated_negative_binomial(rng, mu, alpha: float) -> np.ndarray:
    """Zero-truncated NB2 draws, by resampling the zeros.

    Where ``NB(0)`` is essentially one (``μ → 0``) the truncated NB is
    essentially ``y = 1``, which is returned after 1000 rounds.
    """
    mu = np.asarray(mu, dtype=np.float64)
    p = alpha / (alpha + mu)
    y = rng.negative_binomial(alpha, p).astype(np.int64)
    zero = y == 0
    for _ in range(1000):
        if not zero.any():
            break
        y[zero] = rng.negative_binomial(alpha, p[zero])
        zero = y == 0
    y[zero] = 1
    return y


def _intercept_shift(eta, a0, target):
    """The shift ``s`` with ``mean(logit⁻¹(η + s·a0)) = target``."""
    lo, hi = -50.0, 50.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if np.mean(1.0 / (1.0 + np.exp(-(eta + mid * a0)))) < target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def simulate_sar_hurdle(
    n_side: int | None = None,
    W=None,
    gdf=None,
    rho: float = 0.4,
    lam: float = 0.3,
    beta: np.ndarray | None = None,
    gamma: np.ndarray | None = None,
    alpha: float = 2.0,
    Z: np.ndarray | None = None,
    X: np.ndarray | None = None,
    W_sel=None,
    rng: np.random.Generator | None = None,
    seed: int | None = None,
    contiguity: str = "queen",
    target_pi: float | None = None,
    err_hetero: bool = False,
    create_gdf: bool = False,
    geometry_type: str = "polygon",
) -> dict:
    r"""Simulate a reduced-form SAR hurdle NB cross-section.

    .. math::

        \eta^{\mathrm{b}} = (I - \lambda W_{\mathrm{sel}})^{-1} Z\gamma, \qquad
        \eta^{\mathrm{c}} = (I - \rho W)^{-1} X\beta,

    ``d ~ Bernoulli(logit⁻¹(η^b))`` and ``y = d · tNB(e^{η^c}, α)``.

    Parameters
    ----------
    rho, lam : float
        Count and binary spatial parameters.
    beta, gamma : array-like, optional
        Count and binary coefficients, intercept first; defaults ``[1.0, 0.6]``
        and ``[0.3, 1.0]``.
    alpha : float, default 2.0
        NB2 dispersion of the count half.
    target_pi : float, optional
        Shift the binary intercept (inside the filter) so ``mean P(y > 0)``
        equals ``target_pi``; ``params_true["gamma"]`` reports the shifted γ.
    err_hetero : bool, default False
        Accepted for API parity; the count variance is set by the mean and
        ``alpha``, so it is ignored with a warning.
    n_side, W, gdf, Z, X, W_sel, rng, seed, contiguity, create_gdf, geometry_type
        As for :func:`~neighbayes.dgp.simulate_sar_zinb`.

    Returns
    -------
    dict
        ``y``, ``d``, ``X``, ``Z``, ``eta_bin``, ``eta_cnt``, ``W_sparse``,
        ``W_graph``, ``W_sel_sparse``, ``W_sel_graph``, ``params_true``.
    """
    if alpha <= 0:
        raise ValueError("alpha must be strictly positive.")
    if err_hetero:
        warnings.warn(
            "err_hetero does not apply to count DGPs and is ignored.", stacklevel=2
        )
    rng = ensure_rng(rng, seed)
    Ws, Wg = resolve_weights(W=W, gdf=gdf, n_side=n_side, contiguity=contiguity)
    n = Ws.shape[0]
    if W_sel is not None:
        W_sel_s, W_sel_g = resolve_weights(
            W=W_sel, gdf=gdf, n_side=n_side, contiguity=contiguity
        )
    else:
        W_sel_s, W_sel_g = Ws, Wg
    beta = np.asarray([1.0, 0.6] if beta is None else beta, dtype=float)
    gamma = np.asarray([0.3, 1.0] if gamma is None else gamma, dtype=float)
    X = (
        make_design_matrix(rng, n, k=max(len(beta) - 1, 0), add_intercept=True)
        if X is None
        else np.asarray(X, dtype=np.float64)
    )
    Z = (
        make_design_matrix(rng, n, k=max(len(gamma) - 1, 0), add_intercept=True)
        if Z is None
        else np.asarray(Z, dtype=np.float64)
    )
    _check_rho_stability(rho, name="rho")
    _check_rho_stability(lam, name="lam")

    solve_b = spatial_filter_factor(W_sel_s, lam)
    eta_bin = solve_b(Z @ gamma)
    if target_pi is not None:
        if not 0.0 < float(target_pi) < 1.0:
            raise ValueError(f"target_pi must lie in (0, 1); got {target_pi!r}")
        a0 = np.asarray(solve_b(np.ascontiguousarray(Z[:, 0])), dtype=float)
        s = _intercept_shift(eta_bin, a0, float(target_pi))
        eta_bin = eta_bin + s * a0
        gamma = gamma.copy()
        gamma[0] += s
    d = rng.binomial(1, 1.0 / (1.0 + np.exp(-eta_bin)))
    eta_cnt = spatial_filter_factor(Ws, rho)(X @ beta)
    y = np.zeros(n, dtype=np.int64)
    pos = d == 1
    if pos.any():
        y[pos] = truncated_negative_binomial(
            rng, np.exp(np.clip(eta_cnt[pos], -30.0, 30.0)), alpha
        )
    out = {
        "y": y.astype(np.float64),
        "d": d.astype(np.float64),
        "X": X,
        "Z": Z,
        "eta_bin": eta_bin,
        "eta_cnt": eta_cnt,
        "W_sparse": Ws,
        "W_graph": Wg,
        "W_sel_sparse": W_sel_s,
        "W_sel_graph": W_sel_g,
        "params_true": {
            "rho": rho,
            "lam": lam,
            "beta": beta,
            "gamma": gamma,
            "alpha": float(alpha),
        },
    }
    return _attach_optional_gdf(
        out, source_gdf=gdf, create_gdf=create_gdf, geometry_type=geometry_type
    )


def simulate_panel_sar_hurdle(
    N: int,
    T: int,
    rho: float = 0.4,
    lam: float = 0.3,
    beta: np.ndarray | None = None,
    gamma: np.ndarray | None = None,
    alpha: float = 2.0,
    unit_effect_sd: float = 0.0,
    time_effect_sd: float = 0.0,
    sel_unit_effect_sd: float = 0.0,
    sel_time_effect_sd: float = 0.0,
    effect_corr: float = 0.0,
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
    r"""Simulate a SAR hurdle NB panel with unit and period effects in both halves.

    The count half is :func:`~neighbayes.dgp.simulate_panel_sar_negbin`'s
    log-mean (its ``unit_effect_sd``, ``time_effect_sd`` and ``fe_x_corr``),
    truncated at zero; the binary half is

    .. math::

        \eta^{\mathrm{b}}_t = (I - \lambda W_{\mathrm{sel}})^{-1} Z_t\gamma
        + c^{\mathrm{b}} + \tau^{\mathrm{b}}_t,

    with a fresh ``Z_t`` (intercept plus standard-normal columns) each period,
    ``d_{it} ~ Bernoulli(logit⁻¹(η^b_{it}))`` and ``y = d · tNB``.  Matches
    :class:`~neighbayes.models.SARHurdleNBPanel` (``rho = lam = 0`` gives
    :class:`~neighbayes.models.HurdleNBPanel`).

    Parameters
    ----------
    lam : float, default 0.3
        Binary spatial parameter.
    gamma : array-like, optional
        Binary coefficients including the intercept; default ``[0.5, 1.0]``.
    sel_unit_effect_sd, sel_time_effect_sd : float, default 0.0
        Standard deviations of the binary unit and period effects
        (``τ^b_0 = 0``).
    effect_corr : float, default 0.0
        Correlation, in ``(−1, 1)``, between each unit's binary and count
        effects (the halves are linked when it is non-zero; the models
        assume it is zero).
    N, T, rho, beta, alpha, unit_effect_sd, time_effect_sd, fe_x_corr, \
    err_hetero, rng, seed, W, gdf, n_side, contiguity
        As for :func:`~neighbayes.dgp.simulate_panel_sar_negbin`.

    Returns
    -------
    dict
        ``y``, ``X``, ``Z``, ``d``, ``mu``, ``eta_bin``, ``unit``, ``time``,
        ``W_sparse``, ``W_graph``, ``W_sel_sparse``, ``params_true`` (with
        both halves' effects).
    """
    if alpha <= 0:
        raise ValueError("alpha must be strictly positive.")
    if not -1.0 < effect_corr < 1.0:
        raise ValueError("effect_corr must lie in (-1, 1).")
    rng = ensure_rng(rng, seed)
    out = _simulate_panel_count(
        N, T, rho, beta, alpha, unit_effect_sd, time_effect_sd,
        fe_x_corr, err_hetero, rng, None, W, gdf, n_side, contiguity,
    )  # fmt: skip
    gamma = np.asarray([0.5, 1.0] if gamma is None else gamma, dtype=float)
    Ws_sel = (
        resolve_weights(W=W_sel, gdf=gdf, n_side=n_side, contiguity=contiguity)[0]
        if W_sel is not None
        else out["W_sparse"]
    )
    c_cnt = np.asarray(out["params_true"]["unit_effect"], dtype=float)
    z_cnt = c_cnt / c_cnt.std() if c_cnt.std() > 0 else rng.standard_normal(N)
    sel_unit = sel_unit_effect_sd * (
        effect_corr * z_cnt + np.sqrt(1.0 - effect_corr**2) * rng.standard_normal(N)
    )
    sel_time = np.zeros(T)
    if T > 1:
        sel_time[1:] = sel_time_effect_sd * rng.standard_normal(T - 1)
    solve = spatial_filter_factor(Ws_sel, lam)
    Z_list, eta_list = [], []
    for t in range(T):
        Z_t = np.column_stack([np.ones(N), rng.standard_normal((N, len(gamma) - 1))])
        Z_list.append(Z_t)
        eta_list.append(solve(Z_t @ gamma) + sel_unit + sel_time[t])
    Z = np.vstack(Z_list)
    eta_bin = np.concatenate(eta_list)
    d = rng.binomial(1, 1.0 / (1.0 + np.exp(-eta_bin)))
    y = np.zeros(N * T, dtype=np.int64)
    pos = d == 1
    if pos.any():
        y[pos] = truncated_negative_binomial(rng, out["mu"][pos], alpha)
    out.update(y=y, Z=Z, d=d, eta_bin=eta_bin, W_sel_sparse=Ws_sel)
    out["params_true"].update(
        lam=lam, gamma=gamma, sel_unit_effect=sel_unit, sel_time_effect=sel_time
    )
    return out
