r"""Reduced-form NB2 Gibbs for count panels with random group effects and period effects.

Model, stacked time-first over ``T`` periods of ``N`` groups (spatial units or
origin–destination pairs):

.. math::

    \eta = (I_T \otimes A(\rho))^{-1} (X\beta + D_g c) + D_\tau \tau,
    \qquad c \sim N(\mu, \sigma^2 I), \qquad y \sim \mathrm{NB2}(e^\eta, \alpha).

The group effects enter inside the filter, with the covariates, as persistent
unobservables propagate in the space–time steady state.  ``A⁻¹`` commutes
with the group dummies ``D_g = 1_T ⊗ I_N``, so the working model carries
``c̃ = A⁻¹c``, under the prior ``N(μA⁻¹1, σ²(AᵀA)⁻¹)``.  That is a different
model from independent effects outside the filter: the two coincide only for
unpooled effects.  Period effects ``τ``, outside the filter, join the
coefficient block ``θ = (β, τ)``.

**Sweep** (partially collapsed Gibbs), one
:class:`~neighbayes.samplers._utils._glm_equation.MaterializedEquation`:

1. Pólya–Gamma ``ω ~ PG(y + α, η − log α)``, leaving a Gaussian working model
   ``z = Uθ + D_g c̃ + e``.
2. Each ρ by slice sampling with ``θ`` and ``c̃`` integrated out — ``c̃``
   through the ω-weighted within transform of
   :mod:`~neighbayes.samplers._utils._group_effects`, whose prior precision
   ``AᵀA/σ²`` moves with ρ.
3. ``(θ, c̃) | ρ`` jointly: ``θ`` from the Schur Gram, then ``c̃ | θ``; then
   the effects' sd — centred ``σ | c`` then non-centred ``σ | c/σ``
   (interweaving, so σ mixes at small T), the latter again on the exact NB
   likelihood.
4. ``α`` by slice sampling on ``log α``, unless ``alpha_fixed`` holds it.

``filter`` is one of :mod:`._filters`; :class:`~._filters.NoFilter` gives the
aspatial model.  ``U = A⁻¹X`` is materialized, so this kernel suits place-based
panels and flow panels of moderate ``n``; the separable NB flow model at scale
has its own structured kernel.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .._utils._count_loglik import nb_eta_loglik
from .._utils._group_effects import GroupEffects


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
        Random group effects (one per unit or pair), inside the filter;
        integrated out of each ρ update.
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
    from .._utils._glm_equation import MaterializedEquation, init_from_counts
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

    rho = filt.init(rng)
    theta, c = init_from_counts(y, X, fe, np.ones(y.size, bool), rng)
    eq = MaterializedEquation(
        X, filt, mu0, prec0, fe, theta=theta, c=c, params=rho, rng=rng
    )
    alpha = 1.0 if priors.alpha_fixed is None else float(priors.alpha_fixed)

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

    for it in range(tune + draws):
        tuning = it < tune
        # --- 1. ω and the Gaussian working response ---
        omega = _sample_omega(y, alpha, eq.eta - np.log(alpha), rng=rng)
        z = 0.5 * (y - alpha) / omega + np.log(alpha)

        # --- 2–3. each ρ, then (θ, c̃) and the effects' sd ---
        eq.update(None, z, omega, rng, tuning)
        # Non-centred again on the exact NB likelihood (no PG layer).
        eq.exact_sd_move(lambda e: nb_eta_loglik(y, e, alpha), rng)

        # --- 4. dispersion (held when alpha_fixed) ---
        st = _AlphaState(eta=eq.eta, beta=eq.theta, rho=0.0, alpha=alpha, omega=omega)
        alpha = _sample_alpha(st, y, alpha_priors, rng=rng)

        if not tuning and (it - tune) % thin == 0:
            j = (it - tune) // thin
            if j < n_keep:
                out["beta"][j] = eq.theta[:k]
                out["time_effect"][j] = eq.theta[k:]
                out["alpha"][j] = alpha
                for name in filt.names:
                    out[name][j] = eq.params[name]
                if learn_sd:
                    out["group_sd"][j] = eq.group_sd
                if gstore is not None:
                    gstore.add(j, eq.effects())
                if store_log_lik:
                    out["log_lik"][j] = _nb_loglik_pointwise(y, eq.eta, alpha)
        if progress_manager is not None:
            progress_manager.update(chain_id, it, tuning=tuning)

    if gstore is not None:
        out.update(gstore.result())
    return out
