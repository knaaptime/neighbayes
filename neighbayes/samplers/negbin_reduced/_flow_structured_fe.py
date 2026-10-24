r"""Structured separable NB flow sweep with pair and period effects.

Extends :mod:`._flow_structured` (read it first) to

.. math::

    \eta_t = L_o^{-1} (B_t + C) L_d^{-\top} + \tau_t \mathbf 1\mathbf 1^\top,
    \qquad C_{od} \sim N(\mu, \sigma^2),

with ``C`` the ``n × n`` array of random pair effects, inside the filter with
the covariates, and ``τ_t`` period effects outside it — for the last
``n_tau`` periods (``T − 1`` beside pair effects, ``τ_0 = 0``; all ``T``
without them).  The sweep carries the working effects ``C̃ = A⁻¹C``
(``A = L_o ⊗ L_d``, so ``A⁻¹C = L_o⁻¹ C L_d⁻ᵀ``), whose prior
``N(μA⁻¹1, σ²(AᵀA)⁻¹)`` depends on ρ.  Its ``n² × n²`` precision cannot be
factored at scale, so — unlike the generic kernel of
:mod:`neighbayes.samplers.count_panel`, which integrates the effects out — the
effects are held while ρ moves and drawn by conjugate gradients.

**Sweep** (partially collapsed Gibbs):

1. ``ω ~ PG(y + α, η − log α)``.
2. ``ρ_d`` then ``ρ_o`` by slice sampling with the rank-one block ``β_c`` and
   the period effects ``τ`` integrated out (``β_full`` held), twice: centred,
   with ``C̃`` held and its prior ``n log|L_o||L_d| − ‖AC̃ − μ‖²/2σ²`` in the
   density; then non-centred, with ``C`` held and ``C̃ = A(ρ)⁻¹C`` moving.
   Interweaving the two keeps ρ mixing whether the data pin the effects or
   the prior does.
3. ``(β, τ) | ρ, C̃`` from the Schur Gram, then ``C̃ | β, τ`` from
   ``N(P⁻¹b, P⁻¹)``, ``P = Σ_t Ω_t + AᵀA/σ²``, by Jacobi-preconditioned
   conjugate gradients on a perturbed right-hand side (exact to the solver's
   tolerance), warm-started at the last draw.
4. The time-invariant columns ``J`` (intercept, distance, intraunit
   indicators) trade off exactly with ``C``: ``β_J + t``, ``C − Z_J t`` leaves
   η unchanged, so ``t`` is drawn from the two priors alone.  Without this
   move the effects pin those coefficients and they barely mix.
5. The pair-effect sd: centred ``σ | C``, then non-centred ``σ | C/σ``.
6. ``α`` by slice sampling on ``log α``.

Nothing ``N × N`` is formed; the extra memory is a few ``n × n`` arrays per
column and per CG iterate.
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

#: Relative residual at which the pair-effect CG solve stops: far below the
#: Monte Carlo error of any draw, so the draw is exact for practical purposes.
_CG_TOL = 1e-10
#: A cap that only a pathological precision reaches; CG warns if it does.
_CG_MAXITER = 20_000


def _cheap_arrays(A, B, struct, t):
    """Period ``t``'s rank-one columns as ``(k_c, n, n)`` outer products."""
    ui, vi = struct.u_idx[t], struct.v_idx[t]
    return A[:, ui].T[:, :, None] * B[:, vi].T[:, None, :]


def _gram_block(Oms, Rs, cols_of, prec, prec_tau):
    r"""Moments ``(M, v, quad)`` over columns ``[x-cols, τ]``.

    ``cols_of(t)`` gives period ``t``'s x-columns as ``(k, n, n)``; ``Rs[t]`` the
    centered residual.  The period columns cover the last ``len(prec_tau)``
    periods; ``prec`` and ``prec_tau`` are the prior precisions added to ``M``.
    """
    T = len(Oms)
    k = len(prec)
    n_tau = len(prec_tau)
    t0 = T - n_tau
    K = k + n_tau
    M = np.zeros((K, K))
    v = np.zeros(K)
    quad = 0.0
    for t in range(T):
        Om, R = Oms[t], Rs[t]
        OmR = Om * R
        quad += float(np.sum(OmR * R))
        if k:
            C = cols_of(t)
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
    M[np.diag_indices(K)] += np.concatenate([prec, prec_tau])
    return M, v, quad


def _normalize_cell_mask(cell_mask, T: int, n: int):
    """Validate a cell mask and flatten it to ``(T·n·n,)`` bool, or ``None``.

    Accepts a bool array shaped ``(T, n, n)``, ``(T·n·n,)`` or ``(n, n)``
    (broadcast over periods); ``True`` marks a live cell.  The flat order is
    C-contiguous time-first, matching the stacked response.
    """
    if cell_mask is None:
        return None
    m = np.asarray(cell_mask)
    if m.dtype != bool:
        m = m.astype(bool)
    N = T * n * n
    if m.shape == (T, n, n):
        return m.reshape(N)
    if m.shape == (N,):
        return m
    if m.shape == (n, n):
        return np.broadcast_to(m, (T, n, n)).reshape(N).copy()
    raise ValueError(
        f"cell_mask must be bool of shape {(T, n, n)}, {(N,)} or {(n, n)}; "
        f"got {m.shape}"
    )


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
        self.W = sp.csr_matrix(W_csc)
        self.Wt = self.W.T.tocsr()
        self._w_diag = self.W.diagonal()
        self._w_colsq = np.asarray(self.W.multiply(self.W).sum(axis=0)).ravel()
        self.width = {"rho_d": SliceWidthState(), "rho_o": SliceWidthState()}
        self.width_nc = {"rho_d": SliceWidthState(), "rho_o": SliceWidthState()}
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
        self._ti = self._time_invariant()
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

    # --- the filter on the pair effects ----------------------------------

    def _apply_A(self, X, rd, ro, transpose=False):
        """``A X = L_o X L_dᵀ`` (``Aᵀ X = L_oᵀ X L_d`` with ``transpose``) for
        an ``n × n`` ``X``, from the sparse ``W``."""
        W = self.Wt if transpose else self.W
        Y = X - ro * (W @ X)
        return Y - rd * (W @ Y.T).T

    def _solve_A(self, X):
        """``A⁻¹X = L_o⁻¹ X L_d⁻ᵀ`` at the current ρ."""
        Lo = self.solve_o.at(self.rho["rho_o"])
        Ld = self.solve_d.at(self.rho["rho_d"])
        return Ld.solve(Lo.solve(X).T).T

    def _ones_filtered(self):
        """``A⁻¹1 = (L_o⁻¹1)(L_d⁻¹1)ᵀ`` at the current ρ."""
        ones = np.ones(self.n)
        a = self.solve_o.at(self.rho["rho_o"]).solve(ones)
        b = self.solve_d.at(self.rho["rho_d"]).solve(ones)
        return np.outer(a, b)

    def _prior_mean(self):
        """``μA⁻¹1``, the working effects' prior mean, at the current ρ."""
        return self.m_pair * self._ones_filtered() if self.m_pair else 0.0

    def effects(self) -> np.ndarray:
        """The pair effects ``C = A C̃``, ``n × n``."""
        return self._apply_A(self.C, self.rho["rho_d"], self.rho["rho_o"])

    def _gram_diag(self, r):
        """``diag(LᵀL)`` for ``L = I − rW``."""
        return 1.0 - 2.0 * r * self._w_diag + r * r * self._w_colsq

    def _time_invariant(self):
        """The design columns constant over periods: ``(J, Z, filtered)``.

        ``J`` indexes them in the design, ``Z`` holds them as ``(m, n, n)``
        pair arrays, and ``filtered(A, Bv, Ufull)`` gives ``A⁻¹Z`` from the
        current transformed columns.
        """
        st = self.struct
        J, Z, recipes = [], [], []
        for j in range(self.k_c):
            u, v = st.u_idx[:, j], st.v_idx[:, j]
            if np.all(u == u[0]) and np.all(v == v[0]):
                J.append(int(self.cc[j]))
                Z.append(np.outer(st.U[:, u[0]], st.V[:, v[0]]))
                recipes.append(("c", int(u[0]), int(v[0])))
        for j in range(len(self.fc)):
            f = st.f_idx[:, j]
            if np.all(f == f[0]):
                J.append(int(self.fc[j]))
                Z.append(np.asarray(st.F[f[0]], dtype=np.float64))
                recipes.append(("f", int(f[0])))

        def filtered(A, Bv, Ufull):
            return np.stack(
                [
                    np.outer(A[:, r[1]], Bv[:, r[2]]) if r[0] == "c" else Ufull[r[1]]
                    for r in recipes
                ]
            )

        Z = np.stack(Z) if Z else np.zeros((0, self.n, self.n))
        return np.asarray(J, dtype=int), Z, filtered

    def exact_sd_move(self, loglik, rng) -> None:
        """Non-centred pair-effect sd step on the exact likelihood.

        ``loglik(eta)`` gives the data log-likelihood at a ``(T, n, n)``
        predictor; the pair effects are rescaled ``m + σ C̃`` with ``C̃`` held.
        No-op unless the pair sd is learned.
        """
        from .._utils._group_effects import exact_noncentered_sd

        if self.pair_sd_scale is None:
            return
        mean = self._prior_mean()
        ct = (self.C - mean) / self.s_pair
        rest = self.eta - (self.C - mean)[None]
        sd = exact_noncentered_sd(
            self.s_pair, self.pair_sd_scale, lambda s: loglik(rest + s * ct[None]), rng
        )
        self.s_pair = sd
        self.C = mean + sd * ct
        self.eta = rest + sd * ct[None]

    def level_direction(self, col: Optional[int] = None, value: float = 1.0):
        """``(v, log_prior, apply)`` for a shift ``η → η + t·v`` of the level.

        The level is carried by the constant design column ``col`` when there
        is one (``value`` its entries; filtered column ``value ·
        (L_o⁻¹1)(L_d⁻¹1)ᵀ``), else by period effects covering every period,
        else by the pair effects (all shifted by ``t``, which moves ``C̃`` by
        ``tA⁻¹1``).  The intercept comes first: shifting pooled pair effects
        would move their mean off the prior's, which is what their sd is
        estimated from.  ``None`` when none of these is present.
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
            dev = self.effects() - m
            v = self._ones_filtered()

            def log_prior(t):
                return -0.5 * float(np.sum((dev + t) ** 2)) / s**2

            def shift(t):
                self.C = self.C + t * v
        else:
            return None

        def apply(t):
            shift(t)
            self.eta = self.eta + t * v

        return v, log_prior, apply

    # --- one Gibbs update ------------------------------------------------

    def _parts(self, name, B_full):
        """``rv ↦ (η_full, A, Bv)`` at ``name = rv``, the other ρ held."""
        st, rho = self.struct, self.rho
        if name == "rho_d":
            Lo = self.solve_o.at(rho["rho_o"])
            A_fix = Lo.solve(st.U)
            P = {key: Lo.solve(Bf) for key, Bf in B_full.items()}

            def parts(rv):
                Ld = self.solve_d.at(rv)
                eta_f = {key: Ld.solve(Pk.T).T for key, Pk in P.items()}
                return eta_f, A_fix, Ld.solve(st.V)

        else:
            Ld = self.solve_d.at(rho["rho_d"])
            B_fix = Ld.solve(st.V)
            P = {key: Ld.solve(Bf.T).T for key, Bf in B_full.items()}

            def parts(rv):
                Lo = self.solve_o.at(rv)
                eta_f = {key: Lo.solve(Pk) for key, Pk in P.items()}
                return eta_f, Lo.solve(st.U), B_fix

        return parts

    def _collapsed(self, Oms, Zr, eta_f, Av, Bv_):
        """``log p`` of the working data with ``β_c`` and ``τ`` integrated out.

        ``Zr`` is the ``(T, n, n)`` working response less the pair effects;
        ``β_full`` is held.
        """
        st, T, k_c, n_tau, t0 = self.struct, self.T, self.k_c, self.n_tau, self.t0
        Rs = []
        for t in range(T):
            R = Zr[t] - eta_f[self.full_keys[t]]
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

    def _slice_rho(self, name, log_density, width, rng, tuning):
        new, _, sl, sr = slice_sample_1d_adaptive(
            log_density,
            self.rho[name],
            lower=self.rho_lower,
            upper=self.rho_upper,
            width_state=width,
            rng=rng,
        )
        if tuning:
            update_slice_width(width, sl, sr)
        self.rho[name] = float(new)

    def _draw_C(self, Oms, R, rng):
        """``C̃ | β, τ, ρ ~ N(P⁻¹b, P⁻¹)``, ``P = Σ_t Ω_t + AᵀA/σ²``.

        Perturbation–optimization: solving ``P x = b + √D ξ₁ + Aᵀξ₂/σ`` gives an
        exact draw; the solve is Jacobi-preconditioned CG from the last draw.
        ``R`` is the ``(T, n, n)`` working residual without the pair effects.
        """
        from scipy.sparse.linalg import LinearOperator

        from .._utils._spatial_normal import cg_solve

        n, s = self.n, self.s_pair
        rd, ro = self.rho["rho_d"], self.rho["rho_o"]
        D = Oms.sum(axis=0)
        b = np.sum(Oms * R, axis=0)
        if self.m_pair:
            b = b + self._apply_A(np.full((n, n), self.m_pair), rd, ro, True) / s**2
        b = b + np.sqrt(D) * rng.standard_normal((n, n))
        b = b + self._apply_A(rng.standard_normal((n, n)), rd, ro, True) / s

        def matvec(x):
            X = x.reshape(n, n)
            AX = self._apply_A(X, rd, ro)
            return (D * X + self._apply_A(AX, rd, ro, True) / s**2).ravel()

        diag = D + np.outer(self._gram_diag(ro), self._gram_diag(rd)) / s**2
        x = cg_solve(
            LinearOperator((n * n, n * n), matvec=matvec),
            b.ravel(),
            tol=_CG_TOL,
            maxiter=_CG_MAXITER,
            diagonal=diag.ravel(),
            x0=self.C.ravel(),
        )
        return x.reshape(n, n)

    def _block_move(self, A, Bv, Ufull, rng):
        """Draw ``t`` for ``β_J + t``, ``C − Z_J t`` (η unchanged) from the priors."""
        J, Z, filtered = self._ti
        if not J.size:
            return
        s2 = self.s_pair**2
        prec = self.prec0[J]
        Q = np.diag(prec) + np.tensordot(Z, Z, axes=([1, 2], [1, 2])) / s2
        dev = self.effects() - self.m_pair
        h = -prec * (self.beta[J] - self.mu0[J]) + np.tensordot(Z, dev, axes=2) / s2
        L = np.linalg.cholesky(Q)
        step = cho_solve((L, True), h) + solve_triangular(
            L, rng.standard_normal(J.size), lower=True, trans="T"
        )
        self.beta = self.beta.copy()
        self.beta[J] += step
        self.C = self.C - np.tensordot(step, filtered(A, Bv, Ufull), axes=1)

    # --- one Gibbs update ------------------------------------------------

    def update(self, Oms: np.ndarray, z: np.ndarray, rng, tuning: bool) -> None:
        """Draw ``ρ``, then ``(β, τ)``, then the pair effects, from ``(T, n, n)``
        working data."""
        st, n = self.struct, self.n
        k_c, fc = self.k_c, self.fc
        B_full = self._full_sums(self.beta[fc])

        # Centred: C̃ held, its ρ-dependent prior in the density.
        for name in ("rho_d", "rho_o"):
            parts = self._parts(name, B_full)

            def log_density(rv, _name=name, _parts=parts):
                if not (self.rho_lower < rv < self.rho_upper):
                    return -np.inf
                try:
                    eta_f, Av, Bv_ = _parts(rv)
                except FACTORIZATION_ERRORS:
                    return -np.inf  # ρ at the edge of its support
                val = self._collapsed(Oms, z - self.C, eta_f, Av, Bv_)
                if self.has_pairs and np.isfinite(val):
                    rd, ro = (
                        (rv, self.rho["rho_o"])
                        if _name == "rho_d"
                        else (self.rho["rho_d"], rv)
                    )
                    logdet = self.solve_o.at(ro).logdet() + self.solve_d.at(rd).logdet()
                    dev = self._apply_A(self.C, rd, ro) - self.m_pair
                    val += n * logdet - 0.5 * float(np.sum(dev**2)) / self.s_pair**2
                return val

            self._slice_rho(name, log_density, self.width[name], rng, tuning)

        # Non-centred: the effects C held, C̃ = A(ρ)⁻¹C moving with ρ.
        if self.has_pairs:
            C_in = self.effects()
            for name in ("rho_d", "rho_o"):
                parts = self._parts(name, B_full)
                if name == "rho_d":
                    fixed = self.solve_o.at(self.rho["rho_o"]).solve(C_in)

                    def moving(rv, _f=fixed):
                        return self.solve_d.at(rv).solve(_f.T).T

                else:
                    fixed = self.solve_d.at(self.rho["rho_d"]).solve(C_in.T).T

                    def moving(rv, _f=fixed):
                        return self.solve_o.at(rv).solve(_f)

                def log_density(rv, _parts=parts, _moving=moving):
                    if not (self.rho_lower < rv < self.rho_upper):
                        return -np.inf
                    try:
                        eta_f, Av, Bv_ = _parts(rv)
                        Ct = _moving(rv)
                    except FACTORIZATION_ERRORS:
                        return -np.inf
                    return self._collapsed(Oms, z - Ct, eta_f, Av, Bv_)

                self._slice_rho(name, log_density, self.width_nc[name], rng, tuning)
            self.C = self._solve_A(C_in)

        # (β, τ) jointly, then the pair effects
        rho = self.rho
        A, Bv, Ufull = self._transformed(rho["rho_d"], rho["rho_o"])

        def cols_of(t, _A=A, _Bv=Bv, _U=Ufull):
            parts_ = []
            if k_c:
                parts_.append(_cheap_arrays(_A, _Bv, st, t))
            if len(fc):
                parts_.append(np.stack([_U[i] for i in st.f_idx[t]]))
            return np.concatenate(parts_) if parts_ else np.zeros((0, n, n))

        order = self.order
        M, h, _ = _gram_block(
            Oms, list(z - self.C), cols_of, self.prec0[order], self.prec_tau
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
        if self.has_pairs:
            R = z - self._eta_from(A, Bv, Ufull, self.beta)
            self.C = self._draw_C(Oms, R, rng)
            self._block_move(A, Bv, Ufull, rng)
        eta = self._eta_from(A, Bv, Ufull, self.beta)
        del Ufull
        if self.has_pairs:
            if self.pair_sd_scale is not None:
                from .._utils._group_effects import (
                    noncentered_effect_sd,
                    sample_effect_sd,
                )

                # Centred σ | C, then non-centred σ | C/σ (interweaving).
                dev2 = float(np.sum((self.effects() - self.m_pair) ** 2))
                self.s_pair = sample_effect_sd(
                    self.s_pair, dev2, n * n, self.pair_sd_scale, rng
                )
                mean_c = self._prior_mean()
                ct = (self.C - mean_c) / self.s_pair
                OmC = Oms * ct[None]
                P = float(np.sum(OmC * ct[None]))
                h = float(np.sum(OmC * (z - eta - mean_c)))
                self.s_pair = noncentered_effect_sd(
                    self.s_pair, P, h, self.pair_sd_scale, rng
                )
                self.C = mean_c + self.s_pair * ct
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
    cell_mask: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """One chain of the structured separable NB flow sampler with fixed effects.

    Parameters
    ----------
    pair_effects : (mu, sigma) or None
        Normal prior of the pair effects; ``None`` omits them.
    period_effects : (mu, sigma, n_tau) or None
        Normal prior of the period effects, which cover the last ``n_tau``
        periods (``T − 1`` or ``T``); ``None`` omits them.
    cell_mask : bool array, optional
        Live cells (``True`` = live) shaped ``(T, n, n)``, ``(T·n·n,)`` or
        ``(n, n)``; ``None`` keeps every cell.  Masked cells are drawn-then-
        discarded: their Pólya–Gamma variates are drawn (so the random stream
        is mask-invariant), then zeroed.  A pair whose every period is masked
        carries no likelihood; its effect is informed by the prior and, through
        the filter, by its neighbours'.  The α block tabulates live counts
        only; with ``store_log_lik`` the stored pointwise values still cover
        every cell.

    Returns
    -------
    dict
        ``rho_d, rho_o, rho_w, beta, alpha, time_effect`` and the pair effects
        (``group_effect`` draws, or ``group_effect_mean``/``_sd``), ``log_lik``.
    """
    from ..count_panel._core import _GroupStore

    rng = np.random.default_rng() if rng is None else rng
    y = np.asarray(y, dtype=np.float64)
    cell_mask = _normalize_cell_mask(cell_mask, T, n)
    y_tab = (
        np.unique(y[cell_mask], return_counts=True) if cell_mask is not None else None
    )
    core_priors = ReducedGibbsPriors(
        alpha_sigma=priors.alpha_sigma,
        alpha_nu=priors.alpha_nu,
        alpha_fixed=getattr(priors, "alpha_fixed", None),
    )
    C_init = None
    if pair_effects is not None:
        Y = y.reshape(T, n, n)
        if cell_mask is not None:
            # Mask-aware start: no masked count may reach a live cell's η.
            m3 = cell_mask.reshape(T, n, n)
            Yl = np.where(m3, Y, 0.0)
            pair_mean = Yl.sum(axis=0) / np.maximum(m3.sum(axis=0), 1)
            grand = Yl.sum() / m3.sum()
            C_init = np.log(pair_mean + 0.5) - np.log(grand + 0.5) + pair_effects[0]
        else:
            C_init = (
                np.log(Y.mean(axis=0) + 0.5) - np.log(Y.mean() + 0.5) + pair_effects[0]
            )
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
    mask3 = cell_mask.reshape(T, n, n) if cell_mask is not None else None
    for it in range(tune + draws):
        # Masked cells draw with neutral inputs (h = α): the C PG sampler's
        # random consumption is input-dependent, so this keeps the stream —
        # and therefore every live draw — identical whatever masked y holds.
        y_pg = np.where(cell_mask, y, 0.0) if cell_mask is not None else y
        omega = _sample_omega(y_pg, alpha, eq.eta.ravel() - np.log(alpha), rng=rng)
        if cell_mask is not None:
            omega = np.where(cell_mask, omega, 0.0)
            z = 0.5 * (y - alpha) / np.where(cell_mask, omega, 1.0) + np.log(alpha)
            z = np.where(cell_mask, z, 0.0)
        else:
            z = 0.5 * (y - alpha) / omega + np.log(alpha)
        eq.update(
            omega.reshape(T, n, n),
            z.reshape(T, n, n),
            rng,
            tuning=it < tune,
        )
        eq.exact_sd_move(lambda e: nb_eta_loglik(y3, e, alpha, mask3), rng)
        st = _StructuralState(
            eta=eq.eta.ravel(),
            beta=eq.beta,
            rho=eq.rho["rho_d"],
            alpha=alpha,
            omega=omega,
        )
        alpha = _sample_alpha(
            st, y, core_priors, rng=rng, cell_mask=cell_mask, y_tab=y_tab
        )

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
                    gstore.add(j, eq.effects().ravel())
                if store_log_lik:
                    out["log_lik"][j] = _nb_loglik_pointwise(y, eq.eta.ravel(), alpha)
        if progress_manager is not None:
            progress_manager.update(chain_id, it, tuning=it < tune)

    if gstore is not None:
        out.update(gstore.result())
    return out
