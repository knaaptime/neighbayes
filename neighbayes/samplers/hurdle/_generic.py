r"""Reduced-form hurdle NB Gibbs for any filter pair, with effects in both halves.

Model, stacked time-first over ``T ≥ 1`` periods of ``N`` units or
origin–destination pairs:

.. math::

    \eta^{\mathrm{b}} &= (I_T \otimes A_b(\lambda))^{-1} Z\gamma
        + D_\tau\tau_b + D_g c_b, \qquad
    P(y > 0) = \mathrm{logit}^{-1}(\eta^{\mathrm{b}}), \\
    \eta^{\mathrm{c}} &= (I_T \otimes A_c(\rho))^{-1} X\beta
        + D_\tau\tau_c + D_g c_c, \qquad
    y \mid y > 0 \sim \mathrm{NB2}(e^{\eta^{\mathrm{c}}}, \alpha)
    \text{ truncated at } 0 .

Either filter is one of :mod:`neighbayes.samplers.count_panel._filters`
(``NoFilter`` for an aspatial half); each half is a
:class:`~neighbayes.samplers._utils._glm_equation.MaterializedEquation`.  The
halves share no parameters, so the likelihood factorizes and each is sampled
as its own model:

1. Binary, every row: ``ω ~ PG(1, η_b)``, working response ``(d − ½)/ω``;
   each λ with ``(γ, τ_b, c_b)`` integrated out, then those jointly.
2. Count, positive rows: missed zeros ``m``, ``ω ~ PG(y + α(1+m), η_c − log α)``
   (:mod:`._truncated`); each ρ with ``(β, τ_c, c_c)`` integrated out, then
   those jointly.
3. ``(level, log α)`` jointly by a factor slice on the truncated likelihood
   (:class:`._truncated.LevelAlphaMove`), or α alone when there is no level
   carrier; nothing when ``alpha_fixed`` holds it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .._utils._glm_equation import MaterializedEquation, init_from_counts
from ..count_panel._core import CountPanelFE, _GroupStore


@dataclass
class HurdlePriors:
    """Priors for the generic hurdle kernel.

    ``gamma_*`` cover the binary coefficients ``(γ, τ_b)``; ``theta_*`` the
    count coefficients ``(β, τ_c)``.
    """

    gamma_mu: np.ndarray
    gamma_sigma: np.ndarray
    theta_mu: np.ndarray
    theta_sigma: np.ndarray
    alpha_sigma: float = 2.5
    alpha_nu: float = 3.0
    alpha_fixed: Optional[float] = None


def _init_binary(d, Z, fe: CountPanelFE, rng):
    """Least-squares start on the logits of ``d`` shrunk to ``(¼, ¾)``."""
    target = np.where(d, np.log(3.0), -np.log(3.0))
    D = Z if fe.D_tau is None else np.hstack([Z, fe.D_tau])
    theta = np.linalg.lstsq(D, target, rcond=None)[0] if D.shape[1] else np.zeros(0)
    theta = theta + rng.normal(0.0, 0.1, size=theta.size)
    c = None
    if fe.groups is not None:
        g = fe.groups.row_group
        resid = target - D @ theta
        cnt = np.bincount(g, minlength=fe.groups.n_groups)
        c = np.bincount(g, weights=resid, minlength=fe.groups.n_groups) / np.maximum(
            cnt, 1
        )
    return theta, c


def run_chain(
    y: np.ndarray,
    X: np.ndarray,
    Z: np.ndarray,
    filt_cnt,
    filt_bin,
    fe_cnt: CountPanelFE,
    fe_bin: CountPanelFE,
    priors: HurdlePriors,
    draws: int,
    tune: int,
    *,
    thin: int = 1,
    rng: Optional[np.random.Generator] = None,
    chain_id: int = 0,
    progress_manager: object | None = None,
    store_log_lik: bool = False,
    store_group_draws: bool = True,
) -> dict[str, np.ndarray]:
    """One chain of the generic hurdle sampler.

    Returns
    -------
    dict
        Binary: ``gamma``, ``sel_time_effect``, the binary filter parameters
        (``lam``, ``lam_d``, …) and ``sel_group_effect`` (or its
        ``_mean``/``_sd``).  Count: ``beta``, ``time_effect``, the count
        filter parameters, ``alpha`` and ``group_effect``.  Plus ``log_lik``.
    """
    from .._utils._polyagamma import sample_polyagamma
    from ..zinb._generic import sel_name
    from ._truncated import (
        LevelAlphaMove,
        bernoulli_logit_loglik,
        hurdle_loglik_pointwise,
        sample_alpha_truncated,
        truncated_nb_eta_loglik,
        truncated_working_data,
    )

    rng = np.random.default_rng() if rng is None else rng
    y = np.asarray(y, dtype=np.float64)
    X = np.ascontiguousarray(X, dtype=np.float64)
    Z = np.ascontiguousarray(Z, dtype=np.float64)
    n_rows, k = X.shape
    p = Z.shape[1]
    pos = y > 0
    if not pos.any():
        raise ValueError("A hurdle model needs at least one positive count.")
    d = pos.astype(np.float64)
    y_pos = y[pos]
    gmu = np.asarray(priors.gamma_mu, dtype=np.float64)
    mu0 = np.asarray(priors.theta_mu, dtype=np.float64)
    if gmu.size != p + fe_bin.n_tau or mu0.size != k + fe_cnt.n_tau:
        raise ValueError("priors do not match the binary or count design")

    # --- initial state ---
    lam = filt_bin.init(rng)
    rho = filt_cnt.init(rng)
    theta_b, c_b = _init_binary(pos, Z, fe_bin, rng)
    rows = pos if pos.sum() > k + fe_cnt.n_tau else np.ones(n_rows, bool)
    theta_c, c_c = init_from_counts(y, X, fe_cnt, rows, rng)
    alpha = 1.0 if priors.alpha_fixed is None else float(priors.alpha_fixed)

    binary = MaterializedEquation(
        Z,
        filt_bin,
        gmu,
        1.0 / np.asarray(priors.gamma_sigma, dtype=np.float64) ** 2,
        fe_bin,
        theta=theta_b,
        c=c_b,
        params=lam,
        rng=rng,
    )
    count = MaterializedEquation(
        X,
        filt_cnt,
        mu0,
        1.0 / np.asarray(priors.theta_sigma, dtype=np.float64) ** 2,
        fe_cnt,
        theta=theta_c,
        c=c_c,
        params=rho,
        rng=rng,
    )
    mover = (
        LevelAlphaMove(tune, priors.alpha_sigma, priors.alpha_nu)
        if priors.alpha_fixed is None
        else None
    )

    n_keep = draws // thin if thin > 0 else draws
    out: dict[str, np.ndarray] = {
        "gamma": np.empty((n_keep, p)),
        "sel_time_effect": np.empty((n_keep, fe_bin.n_tau)),
        "beta": np.empty((n_keep, k)),
        "time_effect": np.empty((n_keep, fe_cnt.n_tau)),
        "alpha": np.empty(n_keep),
        "log_lik": np.empty((n_keep, n_rows)) if store_log_lik else None,
    }
    for name in filt_bin.names:
        out[sel_name(name)] = np.empty(n_keep)
    for name in filt_cnt.names:
        out[name] = np.empty(n_keep)
    learned = {
        key: fe.groups is not None and fe.groups.sigma_scale is not None
        for key, fe in (("sel_group_sd", fe_bin), ("group_sd", fe_cnt))
    }
    for key, on in learned.items():
        if on:
            out[key] = np.empty(n_keep)
    store_b = (
        _GroupStore(n_keep, fe_bin.groups.n_groups, store_group_draws)
        if fe_bin.groups is not None
        else None
    )
    store_c = (
        _GroupStore(n_keep, fe_cnt.groups.n_groups, store_group_draws)
        if fe_cnt.groups is not None
        else None
    )

    for it in range(tune + draws):
        tuning = it < tune

        # --- 1. binary half, every row ---
        om_b = sample_polyagamma(np.ones(n_rows), binary.eta, rng=rng)
        binary.update(None, (d - 0.5) / om_b, om_b, rng, tuning)
        binary.exact_sd_move(lambda e: bernoulli_logit_loglik(d, e), rng)

        # --- 2. count half, positive rows (truncation augmented) ---
        om_c, z_c = truncated_working_data(y_pos, count.eta[pos], alpha, rng)
        count.update(pos, z_c, om_c, rng, tuning)
        count.exact_sd_move(
            lambda e: truncated_nb_eta_loglik(y_pos, e[pos], alpha), rng
        )

        # --- 3. dispersion, jointly with the level when one is carried ---
        if mover is not None:
            level = count.level_direction()
            if level is None:
                alpha = sample_alpha_truncated(
                    alpha,
                    y_pos,
                    count.eta[pos],
                    alpha_sigma=priors.alpha_sigma,
                    alpha_nu=priors.alpha_nu,
                    rng=rng,
                )
            else:
                v, log_prior, apply = level
                t, alpha = mover.step(
                    it, y_pos, count.eta[pos], v[pos], log_prior, alpha, rng
                )
                apply(t)

        if not tuning and (it - tune) % thin == 0:
            j = (it - tune) // thin
            if j < n_keep:
                out["gamma"][j] = binary.theta[:p]
                out["sel_time_effect"][j] = binary.theta[p:]
                out["beta"][j] = count.theta[:k]
                out["time_effect"][j] = count.theta[k:]
                out["alpha"][j] = alpha
                for name in filt_bin.names:
                    out[sel_name(name)][j] = binary.params[name]
                for name in filt_cnt.names:
                    out[name][j] = count.params[name]
                if learned["sel_group_sd"]:
                    out["sel_group_sd"][j] = binary.group_sd
                if learned["group_sd"]:
                    out["group_sd"][j] = count.group_sd
                if store_b is not None:
                    store_b.add(j, binary.c)
                if store_c is not None:
                    store_c.add(j, count.c)
                if store_log_lik:
                    out["log_lik"][j] = hurdle_loglik_pointwise(
                        y, binary.eta, count.eta, alpha
                    )
        if progress_manager is not None:
            progress_manager.update(chain_id, it, tuning=tuning)

    if store_b is not None:
        out.update({"sel_" + key: val for key, val in store_b.result().items()})
    if store_c is not None:
        out.update(store_c.result())
    return out
