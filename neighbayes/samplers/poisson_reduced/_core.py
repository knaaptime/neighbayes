r"""Reduced-form auxiliary-mixture Gibbs sampler for spatial-lag Poisson flows.

Targets the reduced-form spatial-lag Poisson flow model

.. math::

    y_{ij} \sim \operatorname{Poisson}(\mu_{ij}), \qquad
    \log \boldsymbol{\mu} = A(\boldsymbol{\rho})^{-1} X \beta

where :math:`A = I_N - \rho_d W_d - \rho_o W_o - \rho_w W_w` (unrestricted) or
:math:`A = L_d \otimes L_o` with :math:`L_k = I_n - \rho_k W` and
:math:`\rho_w = -\rho_d \rho_o` (separable) — the same mean propagator the
NB flow sampler uses, so the ρ geometry and the impact decomposition carry over
unchanged.

Three blocks per sweep, against the NB sampler's four:

1. **augmentation** — Frühwirth-Schnatter & Wagner inter-arrival times plus
   mixture indicators (replaces the Pólya–Gamma block).
2. **β** — conjugate Gaussian given :math:`\tilde X = A^{-1} X`.
3. **ρ** — 1-D adaptive slice on each spatial parameter, β marginalized.

There is no dispersion block: Poisson has no free variance parameter, so the
NB sampler's ``log α`` slice — its least vectorisable step — simply disappears.

Why not Pólya–Gamma
-------------------
Poisson admits no exact PG representation, and the obvious workaround (an NB
with α fixed large) degenerates precisely in the limit it is meant to
approximate: the PG conditional precision
:math:`E[\omega] = (\alpha/2\psi)\tanh(\psi/2)` diverges while the marginal
Fisher information stays at :math:`\mu`, so the Gibbs autocorrelation tends to
1.  Measured on this package's NB flow sampler against Poisson data, ESS falls
18× between α=10 and α=10⁴ and rhat reaches 1.80.  The augmentation here is
exact and its working precision converges to :math:`\mu`.

The working data contract
-------------------------
Like the PG samplers, each block downstream of augmentation sees only a working
response ``s`` and a working precision ``omega``.  The one structural
difference is that the augmented design is **ragged**: every observation
contributes one inter-arrival row, and each strictly positive count contributes
a second last-arrival row, so the working design is ``U[design.rows]`` with
``N + N_pos`` rows.  ``A⁻¹`` is still applied once to the ``N``-row ``X``, so
the Krylov machinery is untouched.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import scipy.sparse.linalg as spla
from scipy.linalg import solve_triangular

from ...models.priors import FlowReducedGibbsPriors
from .._utils._slice import slice_sample_1d_adaptive, update_slice_width
from ..negbin_reduced._flow import (
    FlowReducedGibbsCache,
    _assemble_A_unrestricted,
    _build_kron_krylov_basis,
    _eval_U_from_basis,
    _solve_A_separable,
    _solve_A_unrestricted,
    build_unrestricted_krylov_basis,
    flow_system_is_invertible,
)
from ._augment import AugmentedDesign, build_augmented_index, draw_augmentation

__all__ = [
    "FlowPoissonGibbsState",
    "run_chain_unrestricted",
    "run_chain_separable",
]


@dataclass
class FlowPoissonGibbsState:
    """Mutable state for one chain of the reduced-form flow Poisson sampler.

    Parameters
    ----------
    beta : ndarray, shape (k,)
        Regression coefficients.
    rho_d, rho_o : float
        Destination and origin spatial autoregressive parameters.
    rho_w : float or None
        Cross spatial parameter (``None`` for the separable model, where
        ``rho_w = -rho_d * rho_o`` is deterministic).
    s : ndarray, shape (N + N_pos,)
        Working response from the current augmentation.
    omega : ndarray, shape (N + N_pos,)
        Working precision from the current augmentation.
    """

    beta: np.ndarray
    rho_d: float
    rho_o: float
    rho_w: Optional[float]
    s: np.ndarray
    omega: np.ndarray


def _prior_arrays(priors: FlowReducedGibbsPriors, k: int):
    """Unpack scalar-or-vector β prior hyperparameters into arrays."""
    beta_sigma = priors.beta_sigma
    if np.isscalar(beta_sigma):
        V0_inv_diag = np.full(k, 1.0 / (float(beta_sigma) ** 2))
    else:
        V0_inv_diag = 1.0 / (np.asarray(beta_sigma, dtype=np.float64) ** 2)
    beta_mu = priors.beta_mu
    if np.isscalar(beta_mu):
        mu0 = np.full(k, float(beta_mu))
    else:
        mu0 = np.asarray(beta_mu, dtype=np.float64)
    return V0_inv_diag, mu0


def _solve_U(rho_d, rho_o, rho_w, X, cache, basis=None, rho_val=None):
    """Compute ``U = A(ρ)⁻¹X``, reusing the Krylov basis where admissible.

    The shift-invert series is valid for both the separable and unrestricted
    systems — each is affine in the ρ being varied — but only inside the
    basis's own convergence radius.  ``safe_dmax`` comes from a root test on
    the series coefficients, and the configured ``krylov_dmax`` is clamped to
    it: beyond that radius the series diverges and would silently return
    nonsense, which is exactly the failure mode that made the separable
    sampler look statistically broken.
    """
    if (
        basis is not None
        and getattr(basis, "degree", 0) > 0
        and cache.T == 1
        and rho_val is not None
    ):
        drho = rho_val - basis.rho_basis
        if abs(drho) <= min(cache.krylov_dmax, getattr(basis, "safe_dmax", np.inf)):
            return _eval_U_from_basis(basis, drho)
    if cache.separable:
        return _solve_A_separable(rho_d, rho_o, X, cache.W_csc, cache.n, T=cache.T)
    A = _assemble_A_unrestricted(
        rho_d, rho_o, rho_w, cache.Wd, cache.Wo, cache.Ww, cache.Nf
    )
    return _solve_A_unrestricted(A, X, T=cache.T)


def _sample_beta(
    U: np.ndarray,
    design: AugmentedDesign,
    s: np.ndarray,
    omega: np.ndarray,
    priors: FlowReducedGibbsPriors,
    *,
    rng: np.random.Generator,
    beta_current: np.ndarray,
) -> np.ndarray:
    r"""Conjugate Gaussian draw for β on the augmented design.

    .. math::

        \Sigma_\beta^{-1} = \tilde X_a^\top \Omega \tilde X_a + V_0^{-1}, \qquad
        m_\beta = \Sigma_\beta (\tilde X_a^\top \Omega s + V_0^{-1}\mu_0)

    where :math:`\tilde X_a = U[\text{rows}]` is the ragged augmented design.
    """
    k = U.shape[1]
    V0_inv_diag, mu0 = _prior_arrays(priors, k)
    Ua = U[design.rows]
    Uw = Ua * omega[:, None]

    P = Ua.T @ Uw
    P.flat[:: k + 1] += V0_inv_diag
    b = Uw.T @ s + V0_inv_diag * mu0

    try:
        L = np.linalg.cholesky(P)
    except np.linalg.LinAlgError:
        P.flat[:: k + 1] += 1e-10
        try:
            L = np.linalg.cholesky(P)
        except np.linalg.LinAlgError:
            return beta_current
    mean = solve_triangular(L.T, solve_triangular(L, b, lower=True), lower=False)
    return mean + solve_triangular(L.T, rng.normal(size=k), lower=False)


def _rho_log_density_marginal(
    rho_val: float,
    rho_name: str,
    state: FlowPoissonGibbsState,
    cache: FlowReducedGibbsCache,
    X: np.ndarray,
    design: AugmentedDesign,
    priors: FlowReducedGibbsPriors,
    *,
    basis: Optional[object] = None,
) -> float:
    r"""β-marginalized conditional log-density for one ρ parameter.

    Identical in form to the NB sampler's — the working data is Gaussian either
    way — but evaluated on the ragged augmented design.
    """
    if rho_val <= cache.rho_lower or rho_val >= cache.rho_upper:
        return -np.inf

    if rho_name == "rho_d":
        rho_d, rho_o = rho_val, state.rho_o
        rho_w = state.rho_w if state.rho_w is not None else -rho_d * rho_o
    elif rho_name == "rho_o":
        rho_d, rho_o = state.rho_d, rho_val
        rho_w = state.rho_w if state.rho_w is not None else -rho_d * rho_o
    elif rho_name == "rho_w":
        rho_d, rho_o, rho_w = state.rho_d, state.rho_o, rho_val
    else:
        raise ValueError(f"Unknown rho_name: {rho_name}")

    if not cache.separable:
        # Exact spectral admissibility (see flow_system_is_invertible): the
        # old |rho_d|+|rho_o|+|rho_w| < 1 rule is sufficient but not necessary
        # and truncated the posterior short of the MLE.
        if not flow_system_is_invertible(
            rho_d, rho_o, rho_w, cache.W_eig_min, cache.W_eig_max
        ):
            return -np.inf
        if cache.positive and (rho_d < 0.0 or rho_o < 0.0 or rho_w < 0.0):
            return -np.inf

    k = X.shape[1]
    V0_inv_diag, mu0 = _prior_arrays(priors, k)

    try:
        U = _solve_U(rho_d, rho_o, rho_w, X, cache, basis=basis, rho_val=rho_val)
    except (RuntimeError, ValueError, spla.ArpackNoConvergence):
        return -np.inf

    Ua = U[design.rows]
    omega, s = state.omega, state.s
    r = s - Ua @ mu0

    Uw = Ua * omega[:, None]
    M = Ua.T @ Uw
    M.flat[:: k + 1] += V0_inv_diag
    v = Uw.T @ r

    try:
        L = np.linalg.cholesky(M)
    except np.linalg.LinAlgError:
        M.flat[:: k + 1] += 1e-10
        try:
            L = np.linalg.cholesky(M)
        except np.linalg.LinAlgError:
            return -np.inf

    w = solve_triangular(L, v, lower=True)
    result = -float(np.sum(np.log(np.diag(L)))) - 0.5 * (
        float(np.dot(r, omega * r)) - float(w @ w)
    )
    return result if np.isfinite(result) else -np.inf


def _sample_rho_k(
    rho_name: str,
    state: FlowPoissonGibbsState,
    cache: FlowReducedGibbsCache,
    X: np.ndarray,
    design: AugmentedDesign,
    priors: FlowReducedGibbsPriors,
    *,
    rng: np.random.Generator,
    sweep_idx: int,
    tune: int,
    basis: Optional[object] = None,
) -> float:
    """1-D adaptive slice on one ρ parameter with β marginalized."""
    rho_lower, rho_upper = cache.rho_lower, cache.rho_upper
    if cache.positive and not cache.separable:
        rho_lower = max(rho_lower, 0.0)

    if rho_name == "rho_d":
        rho_current, width_state = state.rho_d, cache.rho_d_slice_width_state
    elif rho_name == "rho_o":
        rho_current, width_state = state.rho_o, cache.rho_o_slice_width_state
    elif rho_name == "rho_w":
        rho_current, width_state = state.rho_w, cache.rho_w_slice_width_state
    else:
        raise ValueError(f"Unknown rho_name: {rho_name}")

    def log_density(rho_val: float) -> float:
        return _rho_log_density_marginal(
            rho_val, rho_name, state, cache, X, design, priors, basis=basis
        )

    rho_new, _, steps_left, steps_right = slice_sample_1d_adaptive(
        log_density=log_density,
        x0=rho_current,
        lower=rho_lower,
        upper=rho_upper,
        width_state=width_state,
        rng=rng,
        log_density_x0=log_density(rho_current),
    )
    if sweep_idx < tune:
        update_slice_width(width_state, steps_left, steps_right)
    return rho_new


# ---------------------------------------------------------------------------
# Chain runners
# ---------------------------------------------------------------------------


def _init_state(
    X: np.ndarray,
    y: np.ndarray,
    design: AugmentedDesign,
    separable: bool,
    rng: np.random.Generator,
) -> FlowPoissonGibbsState:
    """Least-squares start on ``log(max(y, 0.5))``, ρ at zero."""
    y_off = np.log(np.maximum(y, 0.5))
    try:
        beta0 = np.linalg.lstsq(X, y_off, rcond=None)[0]
    except np.linalg.LinAlgError:
        beta0 = np.zeros(X.shape[1])
    s0, om0 = draw_augmentation(y, X @ beta0, design, rng=rng)
    return FlowPoissonGibbsState(
        beta=beta0,
        rho_d=0.0,
        rho_o=0.0,
        rho_w=None if separable else 0.0,
        s=s0,
        omega=om0,
    )


def _run_chain(
    y: np.ndarray,
    X: np.ndarray,
    cache: FlowReducedGibbsCache,
    priors: FlowReducedGibbsPriors,
    draws: int,
    tune: int,
    *,
    thin: int = 1,
    rng: Optional[np.random.Generator] = None,
    init: Optional[FlowPoissonGibbsState] = None,
    store_log_lik: bool = True,
) -> dict[str, np.ndarray]:
    """Run one chain; ``cache.separable`` selects the ρ parameterization.

    ``store_log_lik=False`` skips the pointwise log-likelihood (one value per
    draw and flow) and returns ``None`` under ``"log_lik"``.
    """
    rng = np.random.default_rng() if rng is None else rng
    y = np.asarray(y, dtype=np.float64)
    X = np.asarray(X, dtype=np.float64)
    design = build_augmented_index(y)
    separable = cache.separable

    state = init if init is not None else _init_state(X, y, design, separable, rng)

    n_keep = draws // thin
    k = X.shape[1]
    out = {
        "beta": np.empty((n_keep, k)),
        "rho_d": np.empty(n_keep),
        "rho_o": np.empty(n_keep),
        "rho_w": np.empty(n_keep),
        "log_lik": np.empty((n_keep, y.size)) if store_log_lik else None,
    }

    rho_names = ("rho_d", "rho_o") if separable else ("rho_d", "rho_o", "rho_w")
    idx = 0
    for sweep in range(tune + draws):
        rho_w_eff = -state.rho_d * state.rho_o if state.rho_w is None else state.rho_w
        # The Krylov basis is *direction-specific*: a basis built for rho_d
        # cannot evaluate rho_o candidates.  Sharing one across both makes the
        # rho_o conditional a copy of rho_d's, and the two draws come out
        # numerically identical.  Build one per direction.
        # The Krylov basis is direction-specific — one basis cannot serve both
        # rho_d and rho_o (sharing one makes the two draws come out
        # numerically identical).  Build one per direction; correctness of the
        # underlying recurrence is pinned by
        # tests/test_samplers/test_kron_krylov_matvec.py.
        bases: dict[str, object] = {}
        if cache.krylov_degree > 0 and cache.T == 1:
            if separable:
                for _dir in ("rho_d", "rho_o"):
                    try:
                        bases[_dir] = _build_kron_krylov_basis(
                            state.rho_d,
                            state.rho_o,
                            X,
                            cache.W_csc,
                            cache.n,
                            _dir,
                            cache.krylov_degree,
                        )
                    except (RuntimeError, ValueError):
                        bases[_dir] = None
            else:
                # 3 basis builds per sweep instead of ~23 refactorizations.
                for _dir in ("rho_d", "rho_o", "rho_w"):
                    try:
                        bases[_dir] = build_unrestricted_krylov_basis(
                            state.rho_d,
                            state.rho_o,
                            rho_w_eff,
                            X,
                            cache.Wd,
                            cache.Wo,
                            cache.Ww,
                            _dir,
                            cache.krylov_degree,
                        )
                    except (RuntimeError, ValueError):
                        bases[_dir] = None

        U = _solve_U(state.rho_d, state.rho_o, rho_w_eff, X, cache)
        eta = U @ state.beta

        # Block 1 — auxiliary mixture augmentation.
        state.s, state.omega = draw_augmentation(y, eta, design, rng=rng)

        # Block 2 — adaptive slice on each ρ, β marginalized.
        #
        # ρ must be drawn *before* β, matching the NB flow sampler's
        # (ω, ρ, β) cycle.  Drawing β first leaves it fitted at the old ρ, so
        # the next sweep would form eta = A(ρ_new)⁻¹X · β(ρ_old) — a mismatch
        # that corrupts the augmentation and destroys mixing (measured: ESS 3,
        # rhat 2.1).  Redrawing β at the new ρ keeps the pair consistent.
        for name in rho_names:
            val = _sample_rho_k(
                name,
                state,
                cache,
                X,
                design,
                priors,
                rng=rng,
                sweep_idx=sweep,
                tune=tune,
                basis=bases.get(name),
            )
            setattr(state, name, val)

        # Block 3 — conjugate Gaussian β at the *updated* ρ.
        rho_w_new = -state.rho_d * state.rho_o if state.rho_w is None else state.rho_w
        U = _solve_U(state.rho_d, state.rho_o, rho_w_new, X, cache)
        state.beta = _sample_beta(
            U,
            design,
            state.s,
            state.omega,
            priors,
            rng=rng,
            beta_current=state.beta,
        )

        if sweep >= tune and (sweep - tune) % thin == 0 and idx < n_keep:
            out["beta"][idx] = state.beta
            out["rho_d"][idx] = state.rho_d
            out["rho_o"][idx] = state.rho_o
            out["rho_w"][idx] = rho_w_new
            if store_log_lik:
                eta_f = U @ state.beta
                out["log_lik"][idx] = _poisson_loglik_pointwise(y, eta_f)
            idx += 1

    return out


def _poisson_loglik_pointwise(y: np.ndarray, eta: np.ndarray) -> np.ndarray:
    """Pointwise Poisson log-likelihood at the given linear predictor."""
    from scipy.special import gammaln

    eta = np.clip(eta, -30.0, 30.0)
    return y * eta - np.exp(eta) - gammaln(y + 1.0)


def run_chain_unrestricted(
    y, X, cache, priors, draws, tune, *, thin=1, rng=None, init=None, store_log_lik=True
) -> dict[str, np.ndarray]:
    """Run one chain of the unrestricted (3-ρ) flow Poisson sampler."""
    if cache.separable:
        raise ValueError("cache.separable is True; use run_chain_separable")
    return _run_chain(
        y,
        X,
        cache,
        priors,
        draws,
        tune,
        thin=thin,
        rng=rng,
        init=init,
        store_log_lik=store_log_lik,
    )


def run_chain_separable(
    y, X, cache, priors, draws, tune, *, thin=1, rng=None, init=None, store_log_lik=True
) -> dict[str, np.ndarray]:
    """Run one chain of the separable (2-ρ Kronecker) flow Poisson sampler."""
    if not cache.separable:
        raise ValueError("cache.separable is False; use run_chain_unrestricted")
    return _run_chain(
        y,
        X,
        cache,
        priors,
        draws,
        tune,
        thin=thin,
        rng=rng,
        init=init,
        store_log_lik=store_log_lik,
    )
