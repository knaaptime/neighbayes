r"""Hurdle NB DGPs: a reduced-form spatial logit for ``y > 0`` and a truncated NB.

.. math::

    P(y > 0) = \mathrm{logit}^{-1}(\eta^{\mathrm{b}}), \qquad
    y \mid y > 0 \sim \mathrm{NB2}(e^{\eta^{\mathrm{c}}}, \alpha)
    \text{ truncated at } 0,

matching :class:`~neighbayes.models.SARHurdleNegBin` and the hurdle panels.
"""

from __future__ import annotations

import warnings

import numpy as np

from ._design import aspatial, count, simulator
from .cross_sectional import _attach_optional_gdf, _check_rho_stability
from .panel_count import _simulate_panel_count
from .utils import (
    ensure_rng,
    make_design_matrix,
    resolve_weights,
    spatial_filter_factor,
)
from .zinb import _latent_parts


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


def truncated_poisson(rng, eta) -> np.ndarray:
    """Zero-truncated ``Poisson(e^eta)`` draws, exactly: the first event's time
    given one by time 1, then a Poisson count over the rest of the interval."""
    lam = np.maximum(np.exp(np.clip(eta, -30.0, 30.0)), 1e-300)
    t = -np.log1p(rng.random(lam.shape) * np.expm1(-lam)) / lam
    return 1 + rng.poisson(lam * np.clip(1.0 - t, 0.0, 1.0))


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


@simulator(count)
def simulate_sar_hurdle_negbin(
    n_side: int | None = None,
    W=None,
    gdf=None,
    rho: float = 0.4,
    rho_zero: float = 0.3,
    beta: np.ndarray | None = None,
    beta_zero: np.ndarray | None = None,
    alpha: float = 2.0,
    X_zero: np.ndarray | None = None,
    X: np.ndarray | None = None,
    W_zero=None,
    rng: np.random.Generator | None = None,
    seed: int | None = None,
    contiguity: str = "queen",
    target_pi: float | None = None,
    err_hetero: bool = False,
    create_gdf: bool = False,
    geometry_type: str = "polygon",
    latent: bool = False,
) -> dict:
    r"""Simulate a reduced-form SAR hurdle NB cross-section.

    .. math::

        \eta_z = (I - \rho_z W_z)^{-1} X_z\beta_z, \qquad
        \eta^{\mathrm{c}} = (I - \rho W)^{-1} X\beta,

    ``d ~ Bernoulli(logit⁻¹(η_z))`` and ``y = d · tNB(e^{η^c}, α)``.

    With ``latent=True`` it draws the latent hurdle ``SARHurdleNegBin(latent=True)``
    fits, each part's own disturbance propagated: ``d = 1(u > 0)``,
    ``u = (I − ρ_z W_z)⁻¹(X_z β_z + ε_z)`` with ``ε_z`` logistic, and
    ``y | d = 1`` Poisson(``e^η``) truncated at zero, ``η = (I − ρW)⁻¹(Xβ + ε)``
    with ``ε = log γ``, ``γ ~ Gamma(α, α)``; ``eta_zero`` and ``eta_cnt`` are
    then the drawn ``u`` and ``η``.

    Parameters
    ----------
    rho, rho_zero : float
        Count- and zero-part spatial parameters.
    beta, beta_zero : array-like, optional
        Count- and zero-part coefficients, intercept first; defaults
        ``[1.0, 0.6]`` and ``[0.3, 1.0]``.
    alpha : float, default 2.0
        NB2 dispersion of the count part.
    target_pi : float, optional
        Shift the zero part's intercept (inside the filter) so
        ``mean P(y > 0)`` equals ``target_pi``; ``params_true["beta_zero"]``
        reports the shifted coefficients.
    err_hetero : bool, default False
        Accepted for API parity; the count variance is set by the mean and
        ``alpha``, so it is ignored with a warning.
    n_side, W, gdf, X_zero, X, W_zero, rng, seed, contiguity, create_gdf, geometry_type
        As for :func:`~neighbayes.dgp.simulate_sar_zinb`.

    Returns
    -------
    dict
        ``y``, ``d``, ``X``, ``X_zero``, ``eta_zero``, ``eta_cnt``, ``W_sparse``,
        ``W_graph``, ``W_zero_sparse``, ``W_zero_graph``, ``params_true``.
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
    if W_zero is not None:
        W_zero_s, W_zero_g = resolve_weights(
            W=W_zero, gdf=gdf, n_side=n_side, contiguity=contiguity
        )
    else:
        W_zero_s, W_zero_g = Ws, Wg
    beta = np.asarray([1.0, 0.6] if beta is None else beta, dtype=float)
    beta_zero = np.asarray([0.3, 1.0] if beta_zero is None else beta_zero, dtype=float)
    X = (
        make_design_matrix(rng, n, k=max(len(beta) - 1, 0), add_intercept=True)
        if X is None
        else np.asarray(X, dtype=np.float64)
    )
    X_zero = (
        make_design_matrix(rng, n, k=max(len(beta_zero) - 1, 0), add_intercept=True)
        if X_zero is None
        else np.asarray(X_zero, dtype=np.float64)
    )
    _check_rho_stability(rho, name="rho")
    _check_rho_stability(rho_zero, name="rho_zero")

    if latent:
        eta_bin, eta_cnt, beta_zero = _latent_parts(
            rng, Ws, W_zero_s, X, X_zero, beta, beta_zero, rho, rho_zero, alpha,
            target_pi,
        )  # fmt: skip
        d = (eta_bin > 0).astype(np.int64)
        y = np.where(d == 1, truncated_poisson(rng, eta_cnt), 0)
        return _hurdle_output(
            y, d, X, X_zero, eta_bin, eta_cnt, Ws, Wg, W_zero_s, W_zero_g,
            rho, rho_zero, beta, beta_zero, alpha, gdf, create_gdf, geometry_type,
        )  # fmt: skip

    solve_b = spatial_filter_factor(W_zero_s, rho_zero)
    eta_bin = solve_b(X_zero @ beta_zero)
    if target_pi is not None:
        if not 0.0 < float(target_pi) < 1.0:
            raise ValueError(f"target_pi must lie in (0, 1); got {target_pi!r}")
        a0 = np.asarray(solve_b(np.ascontiguousarray(X_zero[:, 0])), dtype=float)
        s = _intercept_shift(eta_bin, a0, float(target_pi))
        eta_bin = eta_bin + s * a0
        beta_zero = beta_zero.copy()
        beta_zero[0] += s
    d = rng.binomial(1, 1.0 / (1.0 + np.exp(-eta_bin)))
    eta_cnt = spatial_filter_factor(Ws, rho)(X @ beta)
    y = np.zeros(n, dtype=np.int64)
    pos = d == 1
    if pos.any():
        y[pos] = truncated_negative_binomial(
            rng, np.exp(np.clip(eta_cnt[pos], -30.0, 30.0)), alpha
        )
    return _hurdle_output(
        y, d, X, X_zero, eta_bin, eta_cnt, Ws, Wg, W_zero_s, W_zero_g,
        rho, rho_zero, beta, beta_zero, alpha, gdf, create_gdf, geometry_type,
    )  # fmt: skip


def _hurdle_output(
    y, d, X, X_zero, eta_bin, eta_cnt, Ws, Wg, W_zero_s, W_zero_g,
    rho, rho_zero, beta, beta_zero, alpha, gdf, create_gdf, geometry_type,
) -> dict:  # fmt: skip
    out = {
        "y": y.astype(np.float64),
        "d": d.astype(np.float64),
        "X": X,
        "X_zero": X_zero,
        "eta_zero": eta_bin,
        "eta_cnt": eta_cnt,
        "W_sparse": Ws,
        "W_graph": Wg,
        "W_zero_sparse": W_zero_s,
        "W_zero_graph": W_zero_g,
        "params_true": {
            "rho": rho,
            "rho_zero": rho_zero,
            "beta": beta,
            "beta_zero": beta_zero,
            "alpha": float(alpha),
        },
    }
    return _attach_optional_gdf(
        out, source_gdf=gdf, create_gdf=create_gdf, geometry_type=geometry_type
    )


def _simulate_sar_hurdle_negbin_panel_re(
    N: int,
    T: int,
    rho: float = 0.4,
    rho_zero: float = 0.3,
    beta: np.ndarray | None = None,
    beta_zero: np.ndarray | None = None,
    alpha: float = 2.0,
    unit_effect_sd: float = 0.0,
    time_effect_sd: float = 0.0,
    unit_effect_sd_zero: float = 0.0,
    time_effect_sd_zero: float = 0.0,
    effect_corr: float = 0.0,
    fe_x_corr: float = 0.0,
    err_hetero: bool = False,
    rng: np.random.Generator | None = None,
    seed: int | None = None,
    W=None,
    W_zero=None,
    gdf=None,
    n_side: int | None = None,
    contiguity: str = "queen",
    latent: bool = False,
) -> dict:
    r"""Simulate a SAR hurdle NB panel with unit and period effects in both parts.

    The count part is :func:`~neighbayes.dgp._simulate_sar_negbin_panel_re`'s
    log-mean (its ``unit_effect_sd``, ``time_effect_sd`` and ``fe_x_corr``),
    truncated at zero; the zero part is

    .. math::

        \eta_{z,t} = (I - \rho_z W_z)^{-1} (X_{z,t}\beta_z + c_z) + \tau_{z,t},

    with a fresh ``X_{z,t}`` (intercept plus standard-normal columns) each
    period, ``d_{it} ~ Bernoulli(logit⁻¹(η_{z,it}))`` and ``y = d · tNB``.
    Matches :class:`~neighbayes.models.SARHurdleNegBin` panels
    (``rho = rho_zero = 0`` gives the aspatial :class:`~neighbayes.models.HurdleNegBin`).
    With ``latent=True`` it draws the latent panel,
    ``SARHurdleNegBin(latent=True)``'s: each part's disturbance and period
    effect propagate with its unit effects, a count is positive where the
    utility :math:`(I - \rho_z W_z)^{-1}(X_{z,t}\beta_z + c_z + \tau_{z,t} +
    \varepsilon_{z,t})` is, logistic :math:`\varepsilon_z`, and then
    ``Poisson(e^η)`` truncated at zero (``eta_zero`` holds the utilities).

    Parameters
    ----------
    rho_zero : float, default 0.3
        Zero-part spatial parameter.
    beta_zero : array-like, optional
        Zero-part coefficients including the intercept; default ``[0.5, 1.0]``.
    unit_effect_sd_zero, time_effect_sd_zero : float, default 0.0
        Standard deviations of the zero part's unit and period effects
        (``τ_{z,0} = 0``).
    effect_corr : float, default 0.0
        Correlation, in ``(−1, 1)``, between each unit's zero- and count-part
        effects (the parts are linked when it is non-zero; the models assume
        it is zero).
    N, T, rho, beta, alpha, unit_effect_sd, time_effect_sd, fe_x_corr, \
    err_hetero, rng, seed, W, gdf, n_side, contiguity
        As for :func:`~neighbayes.dgp._simulate_sar_negbin_panel_re`.

    Returns
    -------
    dict
        ``y``, ``X``, ``X_zero``, ``d``, ``mu``, ``eta_zero``, ``unit``, ``time``,
        ``W_sparse``, ``W_graph``, ``W_zero_sparse``, ``params_true`` (with
        both parts' effects).
    """
    if alpha <= 0:
        raise ValueError("alpha must be strictly positive.")
    if not -1.0 < effect_corr < 1.0:
        raise ValueError("effect_corr must lie in (-1, 1).")
    rng = ensure_rng(rng, seed)
    out = _simulate_panel_count(
        N, T, rho, beta, alpha, unit_effect_sd, time_effect_sd,
        fe_x_corr, err_hetero, rng, None, W, gdf, n_side, contiguity, latent,
    )  # fmt: skip
    beta_zero = np.asarray([0.5, 1.0] if beta_zero is None else beta_zero, dtype=float)
    Ws_sel = (
        resolve_weights(W=W_zero, gdf=gdf, n_side=n_side, contiguity=contiguity)[0]
        if W_zero is not None
        else out["W_sparse"]
    )
    c_cnt = np.asarray(out["params_true"]["unit_effect"], dtype=float)
    z_cnt = c_cnt / c_cnt.std() if c_cnt.std() > 0 else rng.standard_normal(N)
    sel_unit = unit_effect_sd_zero * (
        effect_corr * z_cnt + np.sqrt(1.0 - effect_corr**2) * rng.standard_normal(N)
    )
    sel_time = np.zeros(T)
    if T > 1:
        sel_time[1:] = time_effect_sd_zero * rng.standard_normal(T - 1)
    solve = spatial_filter_factor(Ws_sel, rho_zero)
    Z_list, eta_list = [], []
    for t in range(T):
        Z_t = np.column_stack(
            [np.ones(N), rng.standard_normal((N, len(beta_zero) - 1))]
        )
        Z_list.append(Z_t)
        if latent:
            eps = rng.logistic(size=N)
            eta_list.append(solve(Z_t @ beta_zero + sel_unit + sel_time[t] + eps))
        else:
            eta_list.append(solve(Z_t @ beta_zero + sel_unit) + sel_time[t])
    X_zero = np.vstack(Z_list)
    eta_bin = np.concatenate(eta_list)
    if latent:
        d = (eta_bin > 0).astype(np.int64)
    else:
        d = rng.binomial(1, 1.0 / (1.0 + np.exp(-eta_bin)))
    y = np.zeros(N * T, dtype=np.int64)
    pos = d == 1
    if pos.any() and latent:
        y[pos] = truncated_poisson(rng, np.log(out["mu"][pos]))
    elif pos.any():
        y[pos] = truncated_negative_binomial(rng, out["mu"][pos], alpha)
    out.update(y=y, X_zero=X_zero, d=d, eta_zero=eta_bin, W_zero_sparse=Ws_sel)
    out["params_true"].update(
        rho_zero=rho_zero,
        beta_zero=beta_zero,
        unit_effect_zero=sel_unit,
        time_effect_zero=sel_time,
    )
    return out


@aspatial(simulate_sar_hurdle_negbin, {"rho": 0.0, "rho_zero": 0.0}, panel_only=True)
def simulate_hurdle_negbin():
    """Simulate an aspatial hurdle NB panel: :func:`simulate_sar_hurdle_negbin` at zero dependence.

    Panels only, as for :class:`~neighbayes.models.HurdleNegBin`: pass ``T`` and
    ``N``.
    """
