r"""Reduced-form ZINB Gibbs for any filter pair, with count-equation fixed effects.

Model, stacked time-first over ``T ≥ 1`` periods of ``N`` units or
origin–destination pairs:

.. math::

    \eta^{\mathrm{sel}} &= (I_T \otimes A_s(\lambda))^{-1} Z\gamma, \qquad
    z \sim \mathrm{Bernoulli}(\mathrm{logit}^{-1}(\eta^{\mathrm{sel}})), \\
    \eta^{\mathrm{cnt}} &= (I_T \otimes A_c(\rho))^{-1} X\beta + D_\tau\tau + D_g c,
    \qquad y \mid z = 1 \sim \mathrm{NB2}(e^{\eta^{\mathrm{cnt}}}, \alpha),
    \quad y \mid z = 0 = 0.

Structural zeros are per period.  Either filter is one of
:mod:`neighbayes.samplers.count_panel._filters` (``NoFilter`` for an aspatial
equation); group effects ``c`` and period effects ``τ`` enter the count
equation only, exactly as in :mod:`neighbayes.samplers.count_panel`.

**Sweep** (partially collapsed Gibbs):

1. Selection: ``ω_s ~ PG(1, η_sel)``; each λ by slice sampling with γ
   integrated out (working response ``(z − ½)/ω_s``); then γ.
2. Zero allocation: ``z`` for the zero counts
   (:func:`~neighbayes.samplers.zinb._core._sample_z`).
3. Count, on the ``z = 1`` rows only — the structural zeros are not in the
   count likelihood, so they are dropped rather than given vanishing weight:
   ``ω_c ~ PG(y + α, η − log α)``; each ρ with ``(β, τ, c)`` integrated out;
   then ``(β, τ, c)`` jointly; then α (unless held by ``alpha_fixed``).

``U = A⁻¹X`` is materialized; the separable flow model at scale has a
structured sweep of its own (:mod:`._flow_structured`).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .._utils._count_loglik import zinb_count_eta_loglik
from .._utils._glm_equation import MaterializedEquation, init_from_counts
from ..count_panel._core import CountPanelFE, _GroupStore


@dataclass
class ZINBPriors:
    """Priors for the generic ZINB kernel.

    ``theta_*`` cover the count coefficients ``(β, τ)``; ``gamma_*`` the
    selection coefficients.
    """

    gamma_mu: np.ndarray
    gamma_sigma: np.ndarray
    theta_mu: np.ndarray
    theta_sigma: np.ndarray
    alpha_sigma: float = 2.5
    alpha_nu: float = 3.0
    alpha_fixed: Optional[float] = None


def sel_name(name: str) -> str:
    """The selection-equation name of a filter parameter (``rho_d`` → ``lam_d``)."""
    return name.replace("rho", "lam", 1)


def run_chain(
    y: np.ndarray,
    X: np.ndarray,
    Z: np.ndarray,
    filt_cnt,
    filt_sel,
    fe: CountPanelFE,
    priors: ZINBPriors,
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
    """One chain of the generic ZINB sampler.

    Returns
    -------
    dict
        ``gamma``, the selection parameters (``lam``, ``lam_d``, …), ``beta``,
        ``time_effect``, the count parameters (``rho``, ``rho_d``, …),
        ``alpha``, the group effects and ``log_lik``.
    """
    from .._utils._polyagamma import sample_polyagamma
    from ..negbin._core import GibbsState as _AlphaState
    from ..negbin._core import _sample_alpha
    from ..negbin_reduced._core import ReducedGibbsPriors, _sample_omega
    from ._core import _sample_z, _zinb_loglik_pointwise

    rng = np.random.default_rng() if rng is None else rng
    y = np.asarray(y, dtype=np.float64)
    X = np.ascontiguousarray(X, dtype=np.float64)
    Z = np.ascontiguousarray(Z, dtype=np.float64)
    n_rows, k = X.shape
    p = Z.shape[1]
    n_tau = fe.n_tau
    gmu = np.asarray(priors.gamma_mu, dtype=np.float64)
    mu0 = np.asarray(priors.theta_mu, dtype=np.float64)
    if gmu.size != p or mu0.size != k + n_tau:
        raise ValueError("priors do not match the selection or count design")
    alpha_priors = ReducedGibbsPriors(
        alpha_sigma=priors.alpha_sigma,
        alpha_nu=priors.alpha_nu,
        alpha_fixed=priors.alpha_fixed,
    )

    # --- initial state ---
    positive = y > 0
    lam = filt_sel.init(rng)
    rho = filt_cnt.init(rng)
    share = float(np.clip(positive.mean() * 1.5, 0.05, 0.95))
    gamma = np.zeros(p)
    const = np.flatnonzero(np.all(Z == Z[:1], axis=0) & (Z[0] != 0))
    if const.size:
        gamma[const[0]] = np.log(share / (1.0 - share)) / Z[0, const[0]]
    gamma += rng.normal(0.0, 0.1, size=p)
    # Count start from the positive counts only: the zeros are partly structural.
    rows = positive if positive.sum() > k + n_tau else np.ones(n_rows, bool)
    theta, c = init_from_counts(y, X, fe, rows, rng)
    alpha = 1.0 if priors.alpha_fixed is None else float(priors.alpha_fixed)
    z = np.where(positive, 1, rng.binomial(1, 0.5, size=n_rows)).astype(np.int8)

    sel = MaterializedEquation(
        Z,
        filt_sel,
        gmu,
        1.0 / np.asarray(priors.gamma_sigma, dtype=np.float64) ** 2,
        theta=gamma,
        params=lam,
        rng=rng,
    )
    cnt = MaterializedEquation(
        X,
        filt_cnt,
        mu0,
        1.0 / np.asarray(priors.theta_sigma, dtype=np.float64) ** 2,
        fe,
        theta=theta,
        c=c,
        params=rho,
        rng=rng,
    )

    n_keep = draws // thin if thin > 0 else draws
    out: dict[str, np.ndarray] = {
        "gamma": np.empty((n_keep, p)),
        "beta": np.empty((n_keep, k)),
        "time_effect": np.empty((n_keep, n_tau)),
        "alpha": np.empty(n_keep),
        "log_lik": np.empty((n_keep, n_rows)) if store_log_lik else None,
    }
    for name in filt_sel.names:
        out[sel_name(name)] = np.empty(n_keep)
    for name in filt_cnt.names:
        out[name] = np.empty(n_keep)
    gstore = (
        _GroupStore(n_keep, fe.groups.n_groups, store_group_draws)
        if fe.groups is not None
        else None
    )

    for it in range(tune + draws):
        tuning = it < tune

        # --- 1. selection: PG-logit on the latent allocation z ---
        om_s = sample_polyagamma(np.ones(n_rows), sel.eta, rng=rng)
        sel.update(None, (z - 0.5) / om_s, om_s, rng, tuning)

        # --- 2. zero allocation ---
        z = _sample_z(y, cnt.eta, sel.eta, alpha, rng=rng)
        act = z == 1

        # --- 3. count, active rows only ---
        if np.any(act):
            ya = y[act]
            om_c = _sample_omega(ya, alpha, cnt.eta[act] - np.log(alpha), rng=rng)
            zc = 0.5 * (ya - alpha) / om_c + np.log(alpha)
            cnt.update(act, zc, om_c, rng, tuning)
            cnt.exact_sd_move(
                lambda e: zinb_count_eta_loglik(y, sel.eta, e, alpha), rng
            )
            st = _AlphaState(
                eta=cnt.eta[act],
                beta=cnt.theta,
                sigma2=1.0,
                rho=0.0,
                alpha=alpha,
                omega=om_c,
            )
            alpha = _sample_alpha(st, ya, alpha_priors, rng=rng)

        if not tuning and (it - tune) % thin == 0:
            j = (it - tune) // thin
            if j < n_keep:
                out["gamma"][j] = sel.theta
                out["beta"][j] = cnt.theta[:k]
                out["time_effect"][j] = cnt.theta[k:]
                out["alpha"][j] = alpha
                for name in filt_sel.names:
                    out[sel_name(name)][j] = sel.params[name]
                for name in filt_cnt.names:
                    out[name][j] = cnt.params[name]
                if gstore is not None:
                    gstore.add(j, cnt.c)
                if store_log_lik:
                    out["log_lik"][j] = _zinb_loglik_pointwise(
                        y, sel.eta, cnt.eta, alpha
                    )
        if progress_manager is not None:
            progress_manager.update(chain_id, it, tuning=tuning)

    if gstore is not None:
        out.update(gstore.result())
    return out
