r"""Structured separable NB flow sweep with pair and period fixed effects.

Extends :mod:`._flow_structured` (read it first) to

.. math::

    \eta_t = L_o^{-1} B_t L_d^{-\top} + \tau_t \mathbf 1\mathbf 1^\top + C,

with ``C`` the ``n × n`` array of pair effects (outside the filter: ``A⁻¹``
commutes with the pair dummies, so this is the inside-filter model with
``C = A⁻¹μ``) and ``τ_t`` period effects, also outside — for the last
``n_tau`` periods (``T − 1`` beside pair effects, ``τ_0 = 0``; all ``T``
without them).

Given the PG weights ``Ω_t`` the pair effects ``C ~ N(m, s²)`` integrate out
exactly through the ω-weighted within transform
(:mod:`~neighbayes.samplers._utils._group_effects`): every inner product
``Σ_t ⟨Ω_t x_t, y_t⟩`` loses ``Σ S(x) S(y) / D`` with ``S(x) = Σ_t Ω_t ⊙ x_t``
and ``D = Σ_t Ω_t + 1/s²``, all ``n × n`` arrays.

**Sweep** (partially collapsed Gibbs):

1. ``ω ~ PG(y + α, η − log α)``.
2. ``ρ_d`` then ``ρ_o`` by slice sampling with the rank-one block ``β_c``, the
   period effects ``τ`` and the pair effects ``C`` integrated out (``β_full``
   held).  A period column is ``Ω_t`` itself, so its corrections against other
   periods, ``Σ Ω_s Ω_t / D``, are free of ρ and computed once per sweep.
3. ``(β, τ) | ρ, ω`` from the Schur Gram, then ``C | β, τ`` (independent per
   pair).
4. ``α`` by slice sampling on ``log α``.

Nothing ``N × N`` is formed; the extra memory is one ``n × n`` array per
rank-one column, and one per full-rank column in step 3.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import scipy.sparse as sp
from scipy.linalg import cho_factor, cho_solve, solve_triangular

from .._utils._count_loglik import nb_eta_loglik
from .._utils._slice import (
    SliceWidthState,
    slice_sample_1d_adaptive,
    update_slice_width,
)
from .._utils._spatial_normal import FACTORIZATION_ERRORS
from ..negbin._core import GibbsState as _StructuralState
from ..negbin._core import _nb_loglik_pointwise, _sample_alpha
from ._core import ReducedGibbsPriors, _sample_omega
from ._flow_structured import (
    FlowDesignStructure,
    _FilterSolver,
    _prior_arrays,
    classify_flow_design,
)


def _cheap_arrays(A, B, struct, t):
    """Period ``t``'s rank-one columns as ``(k_c, n, n)`` outer products."""
    ui, vi = struct.u_idx[t], struct.v_idx[t]
    return A[:, ui].T[:, :, None] * B[:, vi].T[:, None, :]


def _gram_block(Oms, Rs, cols_of, prec, prec_tau, D, OmOm_tau):
    r"""Collapsed moments ``(M, v, quad)`` over columns ``[x-cols, τ]``.

    ``cols_of(t)`` gives period ``t``'s x-columns as ``(k, n, n)``; ``Rs[t]`` the
    centered residual.  The period columns cover the last ``len(prec_tau)``
    periods.  ``D`` is the pair-effect precision (``None`` without pair
    effects) and ``OmOm_tau = Σ_pairs Ω_s Ω_t / D`` over those periods.
    """
    T = len(Oms)
    k = len(prec)
    n_tau = len(prec_tau)
    t0 = T - n_tau
    K = k + n_tau
    M = np.zeros((K, K))
    v = np.zeros(K)
    quad = 0.0
    S = None
    SR = None
    if D is not None:
        S = np.zeros((k,) + Oms[0].shape) if k else None
        SR = np.zeros(Oms[0].shape)
    for t in range(T):
        Om, R = Oms[t], Rs[t]
        OmR = Om * R
        quad += float(np.sum(OmR * R))
        C = cols_of(t) if k else None
        if k:
            OmC = Om[None] * C
            M[:k, :k] += np.tensordot(OmC, C, axes=([1, 2], [1, 2]))
            v[:k] += np.tensordot(OmC, R, axes=2)
        if n_tau and t >= t0:
            j = k + t - t0
            M[j, j] += float(Om.sum())
            v[j] += float(OmR.sum())
            if k:
                cross = OmC.sum(axis=(1, 2))
                M[:k, j] += cross
                M[j, :k] += cross
        if D is not None:
            SR += OmR
            if k:
                S += OmC
    if D is not None:
        quad -= float(np.sum(SR * SR / D))
        SRd = SR / D
        if k:
            Sd = S / D[None]
            M[:k, :k] -= np.tensordot(Sd, S, axes=([1, 2], [1, 2]))
            v[:k] -= np.tensordot(S, SRd, axes=2)
        if n_tau:
            M[k:, k:] -= OmOm_tau
            v[k:] -= np.array([float(np.sum(Oms[t] * SRd)) for t in range(t0, T)])
            if k:
                cross = np.stack(
                    [np.tensordot(Sd, Oms[t], axes=2) for t in range(t0, T)], axis=1
                )
                M[:k, k:] -= cross
                M[k:, :k] -= cross.T
    M[np.diag_indices(K)] += np.concatenate([prec, prec_tau])
    return M, v, quad


class StructuredEquation:
    r"""One separable flow equation, sampled from Gaussian working data.

    ``η_t = L_o⁻¹ B_t L_d⁻ᵀ + τ_t + C`` with ``B_t = Σ_j β_j X_{t,j}`` on the
    ``n × n`` flow array.  :meth:`update` takes the working precisions
    ``Ω_t`` and responses ``z_t`` of one Gibbs sweep — Pólya–Gamma for an NB
    count or a logit selection — and draws ``ρ_d`` and ``ρ_o`` (rank-one
    columns, ``τ`` and ``C`` integrated out), then ``(β, τ)`` jointly, then
    ``C``.  Cells with ``Ω = 0`` drop out of every moment exactly, which is how
    the ZINB count equation leaves out its structural zeros.

    Parameters
    ----------
    X : ndarray, shape (n²·T, k)
        Time-first stacked design.
    beta_mu, beta_sigma
        Normal prior on β.
    init_beta, init_rho
        Starting β and ``{"rho_d": ., "rho_o": .}``.
    pair_effects : (mu, sigma) or (mu, sigma, scale) or None
        Prior of the pair effects ``C ~ N(mu, sigma²)``; ``None`` omits them.
        With a third element the sd is learned — ``sigma`` is its start and
        ``scale`` the scale of its half-t(3) prior — and the pair effects are
        partially pooled.
    period_effects : (mu, sigma, n_tau) or None
        Prior of the period effects on the last ``n_tau`` periods.
    C_init : ndarray, shape (n, n), optional
        Starting pair effects.
    """

    def __init__(
        self,
        X,
        W_csc,
        n: int,
        T: int,
        beta_mu,
        beta_sigma,
        *,
        init_beta,
        init_rho: dict,
        pair_effects: Optional[tuple[float, float]] = None,
        period_effects: Optional[tuple[float, float, int]] = None,
        rho_lower: float = -0.999,
        rho_upper: float = 0.999,
        C_init: Optional[np.ndarray] = None,
        struct: Optional[FlowDesignStructure] = None,
    ):
        from types import SimpleNamespace

        self.n, self.T = n, T
        self.struct = classify_flow_design(X, n, T) if struct is None else struct
        st = self.struct
        self.k = st.k
        self.cc, self.fc = st.cheap_cols, st.full_cols
        self.k_c = len(self.cc)
        self.mu0, self.prec0 = _prior_arrays(
            SimpleNamespace(beta_mu=beta_mu, beta_sigma=beta_sigma), self.k
        )
        self.mu_c, self.prec_c = self.mu0[self.cc], self.prec0[self.cc]
        self.mu_tau, s_tau, self.n_tau = (
            period_effects if period_effects is not None else (0, 1, 0)
        )
        self.t0 = T - self.n_tau
        self.prec_tau = np.full(self.n_tau, 1.0 / float(s_tau) ** 2)
        self.has_pairs = pair_effects is not None
        self.m_pair, self.s_pair = (
            (float(pair_effects[0]), float(pair_effects[1]))
            if self.has_pairs
            else (0.0, 1.0)
        )
        self.pair_sd_scale = (
            float(pair_effects[2])
            if self.has_pairs and len(pair_effects) > 2 and pair_effects[2] is not None
            else None
        )
        self.rho_lower, self.rho_upper = rho_lower, rho_upper
        self.solve_o = _FilterSolver(W_csc)
        self.solve_d = _FilterSolver(W_csc)
        self.width = {"rho_d": SliceWidthState(), "rho_o": SliceWidthState()}
        self.beta = np.asarray(init_beta, dtype=np.float64).copy()
        self.rho = {
            "rho_d": float(init_rho["rho_d"]),
            "rho_o": float(init_rho["rho_o"]),
        }
        self.tau = np.full(self.n_tau, float(self.mu_tau))
        self.C = np.zeros((n, n)) if C_init is None or not self.has_pairs else C_init
        self.full_keys = [tuple(int(i) for i in st.f_idx[t]) for t in range(T)]
        self.distinct_keys = sorted(set(self.full_keys))
        self.order = np.concatenate([self.cc, self.fc]).astype(int)
        A, Bv, Ufull = self._transformed(self.rho["rho_d"], self.rho["rho_o"])
        self.eta = self._eta_from(A, Bv, Ufull, self.beta) + self.C

    # --- helpers -------------------------------------------------------

    def _full_sums(self, b_full):
        st, n = self.struct, self.n
        out = {}
        for key in self.distinct_keys:
            acc = np.zeros((n, n))
            for bj, i in zip(b_full, key):
                acc += bj * st.F[i]
            out[key] = acc
        return out

    def _tau_t(self, t):
        return self.tau[t - self.t0] if (self.n_tau and t >= self.t0) else 0.0

    def _eta_from(self, A, Bv, Ufull, b):
        """``A⁻¹Xβ + τ`` per period, without ``C``."""
        st, n = self.struct, self.n
        out = np.empty((self.T, n, n))
        for t in range(self.T):
            e = np.zeros((n, n))
            for bj, i in zip(b[self.fc], st.f_idx[t]):
                e += bj * Ufull[i]
            if self.k_c:
                ui, vi = st.u_idx[t], st.v_idx[t]
                e += (A[:, ui] * b[self.cc]) @ Bv[:, vi].T
            out[t] = e + self._tau_t(t)
        return out

    def _transformed(self, rd, ro):
        st = self.struct
        Lo, Ld = self.solve_o.at(ro), self.solve_d.at(rd)
        A, Bv = Lo.solve(st.U), Ld.solve(st.V)
        Ufull = [Ld.solve(Lo.solve(Fi).T).T for Fi in st.F]
        return A, Bv, Ufull

    def exact_sd_move(self, loglik, rng) -> None:
        """Non-centred pair-effect sd step on the exact likelihood.

        ``loglik(eta)`` gives the data log-likelihood at a ``(T, n, n)``
        predictor; the pair effects are rescaled ``m + σ C̃`` with ``C̃`` held.
        No-op unless the pair sd is learned.
        """
        from .._utils._group_effects import exact_noncentered_sd

        if self.pair_sd_scale is None:
            return
        ct = (self.C - self.m_pair) / self.s_pair
        rest = self.eta - self.C[None] + self.m_pair
        sd = exact_noncentered_sd(
            self.s_pair, self.pair_sd_scale, lambda s: loglik(rest + s * ct[None]), rng
        )
        self.s_pair = sd
        self.C = self.m_pair + sd * ct
        self.eta = rest + sd * ct[None]

    def level_direction(self, col: Optional[int] = None, value: float = 1.0):
        """``(v, log_prior, apply)`` for a shift ``η → η + t·v`` of the level.

        The level is carried by the constant design column ``col`` when there
        is one (``value`` its entries; filtered column ``value ·
        (L_o⁻¹1)(L_d⁻¹1)ᵀ``), else by period effects covering every period,
        else by the pair effects (all shifted by ``t``).  The intercept comes
        first: shifting pooled pair effects would move their mean off the
        prior's, which is what their sd is estimated from.  ``None`` when none
        of these is present.
        """
        n = self.n
        if col is not None:
            j = int(col)
            mu, prec = self.mu0[j], self.prec0[j]

            def log_prior(t):
                return -0.5 * prec * (self.beta[j] + t - mu) ** 2

            def shift(t):
                self.beta = self.beta.copy()
                self.beta[j] += t

            ones = np.ones(n)
            a = self.solve_o.at(self.rho["rho_o"]).solve(ones)
            b = self.solve_d.at(self.rho["rho_d"]).solve(ones)
            v = value * np.outer(a, b)
        elif self.n_tau == self.T and self.T > 0:
            mu, prec = self.mu_tau, self.prec_tau

            def log_prior(t):
                return -0.5 * float(np.sum(prec * (self.tau + t - mu) ** 2))

            def shift(t):
                self.tau = self.tau + t

            v = np.ones((n, n))
        elif self.has_pairs:
            m, s = self.m_pair, self.s_pair

            def log_prior(t):
                return -0.5 * float(np.sum((self.C + t - m) ** 2)) / s**2

            def shift(t):
                self.C = self.C + t

            v = np.ones((n, n))
        else:
            return None

        def apply(t):
            shift(t)
            self.eta = self.eta + t * v

        return v, log_prior, apply

    # --- one Gibbs update ------------------------------------------------

    def update(self, Oms: np.ndarray, z: np.ndarray, rng, tuning: bool) -> None:
        """Draw ``ρ``, then ``(β, τ)``, then ``C`` given ``(T, n, n)`` working data."""
        st, n, T = self.struct, self.n, self.T
        k_c, n_tau, t0 = self.k_c, self.n_tau, self.t0
        zc = z - self.m_pair
        D = Oms.sum(axis=0) + 1.0 / self.s_pair**2 if self.has_pairs else None
        OmOm_tau = None
        if self.has_pairs and n_tau:
            OmD = Oms[t0:].reshape(n_tau, -1) / np.sqrt(D).ravel()[None, :]
            OmOm_tau = OmD @ OmD.T
            del OmD
        B_full = self._full_sums(self.beta[self.fc])
        rho = self.rho

        # ρ_d, ρ_o with β_c, τ and C integrated out
        for name in ("rho_d", "rho_o"):
            if name == "rho_d":
                Lo = self.solve_o.at(rho["rho_o"])
                A_fix = Lo.solve(st.U)
                P = {key: Lo.solve(Bf) for key, Bf in B_full.items()}

                def parts(rv, _P=P, _A=A_fix):
                    Ld = self.solve_d.at(rv)
                    return (
                        {key: Ld.solve(Pk.T).T for key, Pk in _P.items()},
                        _A,
                        Ld.solve(st.V),
                    )

            else:
                Ld = self.solve_d.at(rho["rho_d"])
                B_fix = Ld.solve(st.V)
                P = {key: Ld.solve(Bf.T).T for key, Bf in B_full.items()}

                def parts(rv, _P=P, _B=B_fix):
                    Lo = self.solve_o.at(rv)
                    return (
                        {key: Lo.solve(Pk) for key, Pk in _P.items()},
                        Lo.solve(st.U),
                        _B,
                    )

            def log_density(rv, _parts=parts):
                if not (self.rho_lower < rv < self.rho_upper):
                    return -np.inf
                try:
                    eta_f, Av, Bv_ = _parts(rv)
                except FACTORIZATION_ERRORS:
                    return -np.inf  # ρ at the edge of its support
                Rs = []
                for t in range(T):
                    R = zc[t] - eta_f[self.full_keys[t]]
                    if k_c:
                        ui, vi = st.u_idx[t], st.v_idx[t]
                        R = R - (Av[:, ui] * self.mu_c) @ Bv_[:, vi].T
                    if n_tau and t >= t0:
                        R = R - self.mu_tau
                    Rs.append(R)
                M, v, quad = _gram_block(
                    Oms,
                    Rs,
                    lambda t: _cheap_arrays(Av, Bv_, st, t),
                    self.prec_c if k_c else np.zeros(0),
                    self.prec_tau,
                    D,
                    OmOm_tau,
                )
                if M.size == 0:
                    return -0.5 * quad
                try:
                    L = np.linalg.cholesky(M)
                except np.linalg.LinAlgError:
                    return -np.inf
                w = solve_triangular(L, v, lower=True)
                val = -np.sum(np.log(np.diag(L))) - 0.5 * (quad - float(w @ w))
                return val if np.isfinite(val) else -np.inf

            new, _, sl, sr = slice_sample_1d_adaptive(
                log_density,
                rho[name],
                lower=self.rho_lower,
                upper=self.rho_upper,
                width_state=self.width[name],
                rng=rng,
            )
            if tuning:
                update_slice_width(self.width[name], sl, sr)
            rho[name] = float(new)

        # (β, τ) jointly, then C
        A, Bv, Ufull = self._transformed(rho["rho_d"], rho["rho_o"])
        fc = self.fc

        def cols_of(t, _A=A, _Bv=Bv, _U=Ufull):
            parts_ = []
            if k_c:
                parts_.append(_cheap_arrays(_A, _Bv, st, t))
            if len(fc):
                parts_.append(np.stack([_U[i] for i in st.f_idx[t]]))
            return np.concatenate(parts_) if parts_ else np.zeros((0, n, n))

        order = self.order
        M, h, _ = _gram_block(
            Oms, list(zc), cols_of, self.prec0[order], self.prec_tau, D, OmOm_tau
        )
        h = h + np.concatenate(
            [self.prec0[order] * self.mu0[order], self.prec_tau * self.mu_tau]
        )
        Lg, lower = cho_factor(M, lower=True)
        mean = cho_solve((Lg, lower), h)
        theta = mean + solve_triangular(
            Lg, rng.standard_normal(mean.size), lower=True, trans="T"
        )
        self.beta = np.empty(self.k)
        self.beta[order] = theta[: self.k]
        self.tau = theta[self.k :]
        eta = self._eta_from(A, Bv, Ufull, self.beta)
        del Ufull
        if self.has_pairs:
            S = np.sum(Oms * (zc - eta), axis=0)
            self.C = self.m_pair + S / D + rng.standard_normal((n, n)) / np.sqrt(D)
            if self.pair_sd_scale is not None:
                from .._utils._group_effects import (
                    noncentered_effect_sd,
                    sample_effect_sd,
                )

                # Centred σ | C, then non-centred σ | C̃ (interweaving).
                dev2 = float(np.sum((self.C - self.m_pair) ** 2))
                self.s_pair = sample_effect_sd(
                    self.s_pair, dev2, n * n, self.pair_sd_scale, rng
                )
                ct = (self.C - self.m_pair) / self.s_pair
                OmC = Oms * ct[None]
                P = float(np.sum(OmC * ct[None]))
                h = float(np.sum(OmC * (zc - eta)))
                self.s_pair = noncentered_effect_sd(
                    self.s_pair, P, h, self.pair_sd_scale, rng
                )
                self.C = self.m_pair + self.s_pair * ct
            eta = eta + self.C
        self.eta = eta


def run_chain_separable_structured_fe(
    y: np.ndarray,
    X: np.ndarray,
    W_csc: sp.csc_matrix,
    n: int,
    priors,
    init,
    draws: int,
    tune: int,
    *,
    T: int,
    pair_effects: Optional[tuple[float, float]] = None,
    period_effects: Optional[tuple[float, float, int]] = None,
    rho_lower: float = -0.999,
    rho_upper: float = 0.999,
    thin: int = 1,
    rng: Optional[np.random.Generator] = None,
    chain_id: int = 0,
    progress_manager: object | None = None,
    store_log_lik: bool = False,
    store_group_draws: bool = True,
    struct: Optional[FlowDesignStructure] = None,
) -> dict[str, np.ndarray]:
    """One chain of the structured separable NB flow sampler with fixed effects.

    Parameters
    ----------
    pair_effects : (mu, sigma) or None
        Normal prior of the pair effects; ``None`` omits them.
    period_effects : (mu, sigma, n_tau) or None
        Normal prior of the period effects, which cover the last ``n_tau``
        periods (``T − 1`` or ``T``); ``None`` omits them.

    Returns
    -------
    dict
        ``rho_d, rho_o, rho_w, beta, alpha, time_effect`` and the pair effects
        (``group_effect`` draws, or ``group_effect_mean``/``_sd``), ``log_lik``.
    """
    from ..count_panel._core import _GroupStore

    rng = np.random.default_rng() if rng is None else rng
    y = np.asarray(y, dtype=np.float64)
    core_priors = ReducedGibbsPriors(
        alpha_sigma=priors.alpha_sigma,
        alpha_nu=priors.alpha_nu,
        alpha_fixed=getattr(priors, "alpha_fixed", None),
    )
    C_init = None
    if pair_effects is not None:
        Y = y.reshape(T, n, n)
        C_init = np.log(Y.mean(axis=0) + 0.5) - np.log(Y.mean() + 0.5) + pair_effects[0]
    eq = StructuredEquation(
        X,
        W_csc,
        n,
        T,
        priors.beta_mu,
        priors.beta_sigma,
        init_beta=init.beta,
        init_rho={"rho_d": init.rho_d, "rho_o": init.rho_o},
        pair_effects=pair_effects,
        period_effects=period_effects,
        rho_lower=rho_lower,
        rho_upper=rho_upper,
        C_init=C_init,
        struct=struct,
    )
    alpha = float(init.alpha)

    n_keep = draws // thin if thin > 0 else draws
    out = {
        "rho_d": np.empty(n_keep),
        "rho_o": np.empty(n_keep),
        "rho_w": np.empty(n_keep),
        "beta": np.empty((n_keep, eq.k)),
        "alpha": np.empty(n_keep),
        "time_effect": np.empty((n_keep, eq.n_tau)),
        "log_lik": np.empty((n_keep, y.size)) if store_log_lik else None,
    }
    if eq.pair_sd_scale is not None:
        out["group_sd"] = np.empty(n_keep)
    gstore = _GroupStore(n_keep, n * n, store_group_draws) if eq.has_pairs else None

    y3 = y.reshape(T, n, n)
    for it in range(tune + draws):
        omega = _sample_omega(y, alpha, eq.eta.ravel() - np.log(alpha), rng=rng)
        z = (0.5 * (y - alpha) / omega + np.log(alpha)).reshape(T, n, n)
        eq.update(omega.reshape(T, n, n), z, rng, tuning=it < tune)
        eq.exact_sd_move(lambda e: nb_eta_loglik(y3, e, alpha), rng)
        st = _StructuralState(
            eta=eq.eta.ravel(),
            beta=eq.beta,
            sigma2=1.0,
            rho=eq.rho["rho_d"],
            alpha=alpha,
            omega=omega,
        )
        alpha = _sample_alpha(st, y, core_priors, rng=rng)

        if it >= tune and (it - tune) % thin == 0:
            j = (it - tune) // thin
            if j < n_keep:
                out["rho_d"][j] = eq.rho["rho_d"]
                out["rho_o"][j] = eq.rho["rho_o"]
                out["rho_w"][j] = -eq.rho["rho_d"] * eq.rho["rho_o"]
                out["beta"][j] = eq.beta
                out["alpha"][j] = alpha
                out["time_effect"][j] = eq.tau
                if eq.pair_sd_scale is not None:
                    out["group_sd"][j] = eq.s_pair
                if gstore is not None:
                    gstore.add(j, eq.C.ravel().copy())
                if store_log_lik:
                    out["log_lik"][j] = _nb_loglik_pointwise(y, eq.eta.ravel(), alpha)
        if progress_manager is not None:
            progress_manager.update(chain_id, it, tuning=it < tune)

    if gstore is not None:
        out.update(gstore.result())
    return out
