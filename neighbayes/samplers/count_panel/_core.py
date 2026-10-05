r"""Reduced-form NB2 Gibbs for count panels with group and period fixed effects.

Model, stacked time-first over ``T`` periods of ``N`` groups (spatial units or
origin–destination pairs):

.. math::

    \eta = (I_T \otimes A(\rho))^{-1} X\beta + D_\tau \tau + D_g c,
    \qquad y \sim \mathrm{NB2}(e^\eta, \alpha).

Group effects sit outside the filter.  ``A⁻¹`` commutes with the group dummies
``D_g = 1_T ⊗ I_N`` (``(I_T⊗A⁻¹)D_g = D_g A⁻¹``), so effects inside the filter
are the same model reparameterized as ``c = A⁻¹μ``; outside, the reduced form
never needs ``A⁻¹D_g`` and the prior on ``c`` is free of ρ.  Period effects
``τ`` join the coefficient block ``θ = (β, τ)``.

**Sweep** (partially collapsed Gibbs):

1. Pólya–Gamma ``ω ~ PG(y + α, η − log α)``, leaving a Gaussian working model
   ``z = Uθ + D_g c + e``.
2. Each ρ by slice sampling with ``θ`` and ``c`` integrated out — ``c``
   through the ω-weighted within transform of
   :mod:`~neighbayes.samplers._utils._group_effects`.
3. ``(θ, c) | ρ`` jointly: ``θ`` from the Schur Gram, then ``c | θ``; then
   the effects' sd when it is learned (``GroupEffects.sigma_scale``), which
   makes the group effects partially pooled — centred ``σ | c`` then
   non-centred ``σ | c̃`` (interweaving, so σ mixes at small T).
4. ``α`` by slice sampling on ``log α``, unless ``alpha_fixed`` holds it.

``filter`` is one of :mod:`._filters`; :class:`~._filters.NoFilter` gives the
aspatial model.  ``U = A⁻¹X`` is materialized, so this kernel suits place-based
panels and flow panels of moderate ``n``; the separable NB flow model at scale
has its own structured kernel.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Optional

import numpy as np

from .._utils._count_loglik import nb_eta_loglik
from .._utils._group_effects import (
    GroupEffects,
    GroupProjector,
    collapsed_log_density,
    draw_collapsed,
    exact_noncentered_sd,
    interweave_group_sd,
)
from .._utils._slice import (
    SliceWidthState,
    slice_sample_1d_adaptive,
    update_slice_width,
)


@dataclass
class CountPanelPriors:
    """Coefficient and dispersion priors.

    ``theta_mu``/``theta_sigma`` cover ``θ = (β, τ)`` in that order.
    ``alpha_fixed`` holds the dispersion at a value instead of sampling it.
    """

    theta_mu: np.ndarray
    theta_sigma: np.ndarray
    alpha_sigma: float = 2.5
    alpha_nu: float = 3.0
    alpha_fixed: Optional[float] = None


@dataclass
class CountPanelFE:
    """Fixed-effect layout of the rows.

    Parameters
    ----------
    groups : GroupEffects or None
        Group effects (one per unit or pair), integrated out each sweep.
    D_tau : ndarray, shape (N·T, n_tau), or None
        Period dummies, outside the filter.
    """

    groups: Optional[GroupEffects] = None
    D_tau: Optional[np.ndarray] = None

    @property
    def n_tau(self) -> int:
        return 0 if self.D_tau is None else self.D_tau.shape[1]


@dataclass
class _GroupStore:
    """Per-draw group effects, or a running mean and variance (Welford)."""

    n_keep: int
    n_groups: int
    keep_draws: bool
    draws: Optional[np.ndarray] = None
    count: int = 0
    mean: Optional[np.ndarray] = None
    m2: Optional[np.ndarray] = None

    def __post_init__(self):
        if self.keep_draws:
            self.draws = np.empty((self.n_keep, self.n_groups))
        else:
            self.mean = np.zeros(self.n_groups)
            self.m2 = np.zeros(self.n_groups)

    def add(self, j: int, c: np.ndarray) -> None:
        if self.keep_draws:
            self.draws[j] = c
            return
        self.count += 1
        delta = c - self.mean
        self.mean += delta / self.count
        self.m2 += delta * (c - self.mean)

    def result(self) -> dict:
        if self.keep_draws:
            return {"group_effect": self.draws}
        var = self.m2 / max(self.count - 1, 1)
        return {"group_effect_mean": self.mean, "group_effect_sd": np.sqrt(var)}


def init_theta(y, X, fe: CountPanelFE, rng, jitter: float = 0.1):
    """Least-squares start on ``log(y + ½)``; group effects from the residual."""
    target = np.log(np.asarray(y, dtype=np.float64) + 0.5)
    Z = X if fe.D_tau is None else np.hstack([X, fe.D_tau])
    theta = np.linalg.lstsq(Z, target, rcond=None)[0] if Z.shape[1] else np.zeros(0)
    theta = theta + rng.normal(0.0, jitter, size=theta.size)
    c = None
    if fe.groups is not None:
        g = fe.groups.row_group
        resid = target - Z @ theta
        cnt = np.bincount(g, minlength=fe.groups.n_groups)
        c = np.bincount(g, weights=resid, minlength=fe.groups.n_groups) / np.maximum(
            cnt, 1
        )
    return theta, c


def run_chain(
    y: np.ndarray,
    X: np.ndarray,
    filt,
    fe: CountPanelFE,
    priors: CountPanelPriors,
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
    """One chain of the NB count panel sampler.

    Parameters
    ----------
    y : ndarray, shape (N·T,)
        Counts, stacked time-first.
    X : ndarray, shape (N·T, k)
        Design (absorbed columns already dropped).
    filt
        A filter from :mod:`._filters`.
    fe : CountPanelFE
    priors : CountPanelPriors
    store_group_draws : bool, default True
        Keep every draw of the group effects; otherwise their posterior mean
        and sd only.

    Returns
    -------
    dict
        ``beta``, ``time_effect``, each ρ, ``alpha``, the group effects and
        ``log_lik``.
    """
    from ..negbin._core import GibbsState as _AlphaState
    from ..negbin._core import _nb_loglik_pointwise, _sample_alpha
    from ..negbin_reduced._core import ReducedGibbsPriors, _sample_omega

    rng = np.random.default_rng() if rng is None else rng
    y = np.asarray(y, dtype=np.float64)
    X = np.ascontiguousarray(X, dtype=np.float64)
    k = X.shape[1]
    n_tau = fe.n_tau
    mu0 = np.asarray(priors.theta_mu, dtype=np.float64)
    prec0 = 1.0 / np.asarray(priors.theta_sigma, dtype=np.float64) ** 2
    if mu0.size != k + n_tau or prec0.size != k + n_tau:
        raise ValueError("theta priors must cover the design columns and periods")
    alpha_priors = ReducedGibbsPriors(
        alpha_sigma=priors.alpha_sigma,
        alpha_nu=priors.alpha_nu,
        alpha_fixed=priors.alpha_fixed,
    )

    def full_design(rho):
        U = filt.solve(rho, X)
        return U if fe.D_tau is None else np.hstack([U, fe.D_tau])

    rho = filt.init(rng)
    theta, c = init_theta(y, X, fe, rng)
    alpha = 1.0 if priors.alpha_fixed is None else float(priors.alpha_fixed)
    width = {name: SliceWidthState() for name in filt.names}
    g_idx = fe.groups.row_group if fe.groups is not None else None

    def eta_of(U, th, cc):
        e = U @ th
        return e if cc is None else e + cc[g_idx]

    n_keep = draws // thin if thin > 0 else draws
    out: dict[str, np.ndarray] = {
        "beta": np.empty((n_keep, k)),
        "time_effect": np.empty((n_keep, n_tau)),
        "alpha": np.empty(n_keep),
        "log_lik": np.empty((n_keep, y.size)) if store_log_lik else None,
    }
    for name in filt.names:
        out[name] = np.empty(n_keep)
    learn_sd = fe.groups is not None and fe.groups.sigma_scale is not None
    if learn_sd:
        out["group_sd"] = np.empty(n_keep)
    gstore = (
        _GroupStore(n_keep, fe.groups.n_groups, store_group_draws)
        if fe.groups is not None
        else None
    )

    U = full_design(rho)
    eta = eta_of(U, theta, c)

    for it in range(tune + draws):
        # --- 1. ω and the Gaussian working response ---
        omega = _sample_omega(y, alpha, eta - np.log(alpha), rng=rng)
        z = 0.5 * (y - alpha) / omega + np.log(alpha)
        proj = GroupProjector(fe.groups, omega) if fe.groups is not None else None

        # --- 2. each ρ with θ and c integrated out ---
        for name in filt.names:
            lo, hi = filt.bounds(name)

            def log_density(v, _name=name):
                cand = dict(rho)
                cand[_name] = float(v)
                if not filt.admissible(cand):
                    return -np.inf
                try:
                    Uc = full_design(cand)
                except (RuntimeError, ValueError, np.linalg.LinAlgError):
                    return -np.inf
                return collapsed_log_density(Uc, z, omega, mu0, prec0, proj)

            new, _, sl, sr = slice_sample_1d_adaptive(
                log_density,
                rho[name],
                lower=lo,
                upper=hi,
                width_state=width[name],
                rng=rng,
            )
            if it < tune:
                update_slice_width(width[name], sl, sr)
            rho[name] = float(new)

        # --- 3. (θ, c) | ρ jointly ---
        U = full_design(rho)
        theta, c_new = draw_collapsed(U, z, omega, mu0, prec0, proj, rng)
        if c_new is not None:
            c = c_new
            if learn_sd:
                groups, c = interweave_group_sd(
                    fe.groups, c, z - U @ theta, omega, g_idx, rng
                )
                # Non-centred again on the exact NB likelihood (no PG layer).
                ct = (c - groups.mu) / groups.sigma
                rest, a = U @ theta + groups.mu, ct[g_idx]
                sd = exact_noncentered_sd(
                    groups.sigma,
                    groups.sigma_scale,
                    lambda s: nb_eta_loglik(y, rest + s * a, alpha),
                    rng,
                )
                c = groups.mu + sd * ct
                fe = replace(fe, groups=replace(groups, sigma=sd))
        eta = eta_of(U, theta, c)

        # --- 4. dispersion (held when alpha_fixed) ---
        st = _AlphaState(
            eta=eta, beta=theta, sigma2=1.0, rho=0.0, alpha=alpha, omega=omega
        )
        alpha = _sample_alpha(st, y, alpha_priors, rng=rng)

        if it >= tune and (it - tune) % thin == 0:
            j = (it - tune) // thin
            if j < n_keep:
                out["beta"][j] = theta[:k]
                out["time_effect"][j] = theta[k:]
                out["alpha"][j] = alpha
                for name in filt.names:
                    out[name][j] = rho[name]
                if learn_sd:
                    out["group_sd"][j] = fe.groups.sigma
                if gstore is not None:
                    gstore.add(j, c)
                if store_log_lik:
                    out["log_lik"][j] = _nb_loglik_pointwise(y, eta, alpha)
        if progress_manager is not None:
            progress_manager.update(chain_id, it, tuning=it < tune)

    if gstore is not None:
        out.update(gstore.result())
    return out
