r"""Structured PG-Gibbs for the separable NB flow model (cross-section and panel).

Model: ``y_od ~ NB2(μ_od, α)``, ``log μ = η = A⁻¹Xβ`` with the separable filter
``A = L_o ⊗ L_d``, ``L_k = I_n − ρ_k W``.  On the ``n × n`` flow array (rows =
origins) ``A⁻¹`` acts as ``Y ↦ L_o⁻¹ Y L_d⁻ᵀ``, so every solve is a pair of
``n × n`` factorizations with ``n`` right-hand sides; nothing ``N × N`` exists.

**Design structure.**  Flow design columns are of two kinds on the flow array:

* *rank one*: the intercept (``1·1ᵀ``), destination attributes (``1·xᵀ``) and
  origin attributes (``x·1ᵀ``).  ``A⁻¹`` keeps them rank one,
  ``(L_o⁻¹u)(L_d⁻¹v)ᵀ``, so each costs two ``n``-vector solves.
* *full*: log-distance, the intra dummy, ``intra × x`` and anything else.  Each
  costs a two-sided ``n × n`` solve.

Columns are classified from the data (constant rows or constant columns), not
from their names.

**Sweep** (partially collapsed Gibbs, van Dyk & Park 2008):

1. ``ω ~ PG(y + α, η − log α)``.
2. ``ρ_d`` then ``ρ_o`` by slice sampling ``p(ρ | ω, α, β_full, y)`` with the
   rank-one block ``β_c`` integrated out under its Normal prior.  That block holds
   the intercept, whose correlation with ρ (the mean level scales like
   ``β₀ / ((1−ρ_o)(1−ρ_d))``) is what conditioning on β would otherwise freeze.
   Per candidate: one ``n × n`` solve for the full-rank part of ``η`` and ``O(N)``
   rank-one algebra.
3. ``β = (β_c, β_full) | ρ, ω, α`` jointly — which also redraws ``β_c`` straight
   after the ρ draws that integrated it out.
4. ``α`` by slice sampling on ``log α``.

Given ``ω`` the PG likelihood is Gaussian in ``η`` with working response
``z = κ/ω + log α``, ``κ = (y − α)/2``, precision ``ω``.

**Panels.**  ``T`` periods stacked time-first share ``W``, ``ρ``, ``β`` and ``α``,
so ``A`` is block-diagonal and each period's ``η_t = L_o⁻¹ B_t L_d⁻ᵀ`` uses the
same two factorizations.  A column is rank one only if it is rank one in every
period; vectors and full-rank arrays are de-duplicated across periods, so a
time-invariant column (log-distance, the intra dummy) is stored and solved once.
The ρ density and the β moments accumulate over periods.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
from scipy.linalg import cho_factor, cho_solve, solve_triangular

from .._utils._slice import (
    SliceWidthState,
    slice_sample_1d_adaptive,
    update_slice_width,
)
from ..negbin._core import GibbsState as _StructuralState
from ..negbin._core import _nb_loglik_pointwise, _sample_alpha
from ._core import ReducedGibbsPriors, _sample_omega

# ---------------------------------------------------------------------------
# Design classification
# ---------------------------------------------------------------------------


@dataclass
class FlowDesignStructure:
    """Rank-one and full-rank columns of a (panel) flow design.

    Period ``t``'s rank-one column ``j`` is ``U[:, u_idx[t, j]] V[:, v_idx[t, j]]ᵀ``
    and its full-rank column ``j`` is ``F[f_idx[t, j]]`` (an ``n × n`` array).
    ``U``, ``V`` and ``F`` are de-duplicated across columns and periods.
    """

    n: int
    T: int
    cheap_cols: np.ndarray
    full_cols: np.ndarray
    U: np.ndarray  # (n, n_u) origin-side vectors
    V: np.ndarray  # (n, n_v) destination-side vectors
    u_idx: np.ndarray  # (T, k_c)
    v_idx: np.ndarray  # (T, k_c)
    F: list  # distinct full-rank (n, n) arrays
    f_idx: np.ndarray  # (T, k_e)

    @property
    def k(self) -> int:
        return len(self.cheap_cols) + len(self.full_cols)


class _Dedup:
    """Index distinct arrays by content."""

    def __init__(self):
        self.items: list[np.ndarray] = []
        self._index: dict[bytes, int] = {}

    def add(self, a: np.ndarray) -> int:
        key = np.ascontiguousarray(a).tobytes()
        if key not in self._index:
            self._index[key] = len(self.items)
            self.items.append(np.ascontiguousarray(a))
        return self._index[key]


def _rank_one(Y: np.ndarray):
    """``(u, v)`` with ``Y = u vᵀ`` for constant-row or constant-column ``Y``."""
    if np.all(Y == Y[0:1, :]):
        return np.ones(Y.shape[0]), Y[0].copy()
    if np.all(Y == Y[:, 0:1]):
        return Y[:, 0].copy(), np.ones(Y.shape[1])
    return None


def classify_flow_design(X: np.ndarray, n: int, T: int = 1) -> FlowDesignStructure:
    """Split a time-first stacked flow design into rank-one and full-rank parts."""
    X = np.asarray(X, dtype=np.float64)
    N = n * n
    if X.shape[0] != N * T:
        raise ValueError(f"X has {X.shape[0]} rows, expected n²·T = {N * T}.")
    blocks = X.reshape(T, n, n, X.shape[1])  # [t, o, d, j]
    cheap, full = [], []
    for j in range(X.shape[1]):
        factors = [_rank_one(blocks[t, :, :, j]) for t in range(T)]
        (cheap if all(f is not None for f in factors) else full).append((j, factors))
    Ud, Vd, Fd = _Dedup(), _Dedup(), _Dedup()
    u_idx = np.zeros((T, len(cheap)), dtype=int)
    v_idx = np.zeros((T, len(cheap)), dtype=int)
    for c, (_, factors) in enumerate(cheap):
        for t, (u, v) in enumerate(factors):
            u_idx[t, c] = Ud.add(u)
            v_idx[t, c] = Vd.add(v)
    f_idx = np.zeros((T, len(full)), dtype=int)
    for c, (j, _) in enumerate(full):
        for t in range(T):
            f_idx[t, c] = Fd.add(blocks[t, :, :, j])
    stack = lambda items: np.column_stack(items) if items else np.zeros((n, 0))  # noqa: E731
    return FlowDesignStructure(
        n=n,
        T=T,
        cheap_cols=np.asarray([j for j, _ in cheap], dtype=int),
        full_cols=np.asarray([j for j, _ in full], dtype=int),
        U=stack(Ud.items),
        V=stack(Vd.items),
        u_idx=u_idx,
        v_idx=v_idx,
        F=Fd.items,
        f_idx=f_idx,
    )


# ---------------------------------------------------------------------------
# I − ρW solvers
# ---------------------------------------------------------------------------


class _FilterSolver:
    """``(I − ρW)⁻¹`` with the package's SuiteSparse routing, refactored per ρ."""

    def __init__(self, W_csc: sp.csc_matrix):
        from .._utils._spatial_normal import CholmodFactor
        from ._core import _make_cholmod_pattern, make_sar_solver

        n = W_csc.shape[0]
        W_sym, WtW, pattern = _make_cholmod_pattern(W_csc, n)
        self._solver = make_sar_solver(CholmodFactor(pattern), W_csc, W_sym, WtW, n)
        self.rho: float | None = None

    def at(self, rho: float) -> "_FilterSolver":
        if rho != self.rho:
            self._solver.factorize(float(rho))
            self.rho = float(rho)
        return self

    def solve(self, rhs: np.ndarray) -> np.ndarray:
        return np.asarray(self._solver.solve(np.asarray(rhs, dtype=np.float64)))


# ---------------------------------------------------------------------------
# Per-period algebra
# ---------------------------------------------------------------------------


def _cheap_gram_and_cross(Om, Rm, A, B, u_idx, v_idx):
    """``U_cᵀΩU_c`` and ``U_cᵀΩr`` for one period's rank-one columns ``a_j b_jᵀ``.

    ``A`` / ``B`` hold the transformed origin / destination vectors
    (``L_o⁻¹U``, ``L_d⁻¹V``); column ``j`` is ``A[:, u_idx[j]] B[:, v_idx[j]]ᵀ``.
    Everything is a pass over the ``n × n`` weight array ``Om``.
    """
    k_c = len(u_idx)
    vs = sorted(set(int(v) for v in v_idx))
    pairs = [(p, q) for i, p in enumerate(vs) for q in vs[i:]]
    pair_of = {pq: i for i, pq in enumerate(pairs)}
    Bprod = np.column_stack([B[:, p] * B[:, q] for p, q in pairs])
    OmB = Om @ Bprod  # (n, n_pairs)
    M = np.empty((k_c, k_c))
    for i in range(k_c):
        for j in range(i, k_c):
            p, q = sorted((int(v_idx[i]), int(v_idx[j])))
            a2 = A[:, u_idx[i]] * A[:, u_idx[j]]
            M[i, j] = M[j, i] = a2 @ OmB[:, pair_of[(p, q)]]
    ORB = (Om * Rm) @ B[:, vs]
    col = {v: c for c, v in enumerate(vs)}
    v = np.array([A[:, u_idx[j]] @ ORB[:, col[int(v_idx[j])]] for j in range(k_c)])
    return M, v


def _low_rank(A, B, u_idx, v_idx, coef):
    """``Σ_j coef_j a_j b_jᵀ`` as an ``n × n`` array."""
    return (A[:, u_idx] * coef) @ B[:, v_idx].T


def _rho_log_density(Oms, Rbases, A, B, struct, mu_c, prec_c):
    """``log p(ρ | ω, α, β_full, y)`` up to a constant, rank-one block integrated out.

    ``Oms[t]`` and ``Rbases[t] = z_t − η_full,t`` are period ``t``'s ``n × n``
    arrays.  With ``r_t = Rbase_t − U_c,t μ_c`` and
    ``M = Λ_c + Σ_t U_c,tᵀΩ_tU_c,t``, ``v = Σ_t U_c,tᵀΩ_t r_t``:
    ``−½ log|M| − ½ (Σ_t r_tᵀΩ_t r_t − vᵀM⁻¹v)``.
    """
    k_c = len(struct.cheap_cols)
    M = np.diag(np.asarray(prec_c, dtype=np.float64)) if k_c else None
    v = np.zeros(k_c)
    quad = 0.0
    for t, (Om, Rb) in enumerate(zip(Oms, Rbases)):
        if k_c:
            ui, vi = struct.u_idx[t], struct.v_idx[t]
            R = Rb - _low_rank(A, B, ui, vi, mu_c)
            Mc, vc = _cheap_gram_and_cross(Om, R, A, B, ui, vi)
            M += Mc
            v += vc
        else:
            R = Rb
        quad += float(np.sum(Om * R * R))
    if not k_c:
        return -0.5 * quad
    try:
        L = np.linalg.cholesky(M)
    except np.linalg.LinAlgError:
        return -np.inf
    w = solve_triangular(L, v, lower=True)
    return -np.sum(np.log(np.diag(L))) - 0.5 * (quad - float(w @ w))


def _beta_gram(Oms, zs, A, Bv, Ufull, struct):
    """``G = Σ_t U_tᵀΩ_tU_t`` and ``h = Σ_t U_tᵀΩ_t z_t`` from structured parts.

    ``Ufull[i]`` is the transformed ``L_o⁻¹ F_i L_d⁻ᵀ`` of distinct full-rank
    array ``i``; ``Oms``/``zs`` are per-period ``n × n`` arrays.
    """
    cc, fc = struct.cheap_cols, struct.full_cols
    G = np.zeros((struct.k, struct.k))
    h = np.zeros(struct.k)
    for t, (Om, z) in enumerate(zip(Oms, zs)):
        ui, vi = struct.u_idx[t], struct.v_idx[t]
        if len(cc):
            Mc, vc = _cheap_gram_and_cross(Om, z, A, Bv, ui, vi)
            G[np.ix_(cc, cc)] += Mc
            h[cc] += vc
        OmZ = Om * z
        Ut = [Ufull[i] for i in struct.f_idx[t]]
        for a_i, (ci, Ui) in enumerate(zip(fc, Ut)):
            OU = Om * Ui
            h[ci] += float(np.sum(OmZ * Ui))
            for cj, Uj in zip(fc[a_i:], Ut[a_i:]):
                val = float(np.sum(OU * Uj))
                G[ci, cj] += val
                if cj != ci:
                    G[cj, ci] += val
            if len(cc):
                OUB = OU @ Bv
                cross = np.array([A[:, ui[j]] @ OUB[:, vi[j]] for j in range(len(cc))])
                G[ci, cc] += cross
                G[cc, ci] += cross
    return G, h


# ---------------------------------------------------------------------------
# Chain
# ---------------------------------------------------------------------------


def _prior_arrays(priors, k):
    mu = np.broadcast_to(np.asarray(priors.beta_mu, dtype=np.float64), (k,)).copy()
    sd = np.broadcast_to(np.asarray(priors.beta_sigma, dtype=np.float64), (k,))
    return mu, 1.0 / sd**2


def run_chain_separable_structured(
    y: np.ndarray,
    X: np.ndarray | None,
    W_csc: sp.csc_matrix,
    n: int,
    priors,
    init,
    draws: int,
    tune: int,
    *,
    T: int = 1,
    rho_lower: float = -0.999,
    rho_upper: float = 0.999,
    thin: int = 1,
    rng: np.random.Generator | None = None,
    chain_id: int = 0,
    progress_manager: object | None = None,
    store_log_lik: bool = False,
    struct: FlowDesignStructure | None = None,
) -> dict[str, np.ndarray]:
    """One chain of the structured separable NB flow sampler.

    ``y`` and ``X`` are stacked time-first over ``T`` periods of ``n²`` flows
    (``T = 1`` for the cross-section).  Pass ``struct`` (from
    :func:`classify_flow_design`) to share the classification across chains; ``X``
    is then unused.  Returns ``rho_d, rho_o, rho_w, beta, alpha, log_lik``.
    """
    rng = np.random.default_rng() if rng is None else rng
    y = np.asarray(y, dtype=np.float64)
    struct = classify_flow_design(X, n, T) if struct is None else struct
    T = struct.T
    k = struct.k
    cc, fc = struct.cheap_cols, struct.full_cols
    mu0, prec0 = _prior_arrays(priors, k)
    mu_c, prec_c = mu0[cc], prec0[cc]
    core_priors = ReducedGibbsPriors(
        beta_mu=priors.beta_mu,
        beta_sigma=priors.beta_sigma,
        alpha_sigma=priors.alpha_sigma,
        alpha_nu=priors.alpha_nu,
        alpha_fixed=getattr(priors, "alpha_fixed", None),
        rho_lower=rho_lower,
        rho_upper=rho_upper,
    )
    solve_o = _FilterSolver(W_csc)
    solve_d = _FilterSolver(W_csc)
    width = {"rho_d": SliceWidthState(), "rho_o": SliceWidthState()}

    beta = np.asarray(init.beta, dtype=np.float64).copy()
    rho = {"rho_d": float(init.rho_d), "rho_o": float(init.rho_o)}
    alpha = float(init.alpha)
    # Periods whose full-rank columns coincide share B_full (and its solves).
    full_keys = [tuple(int(i) for i in struct.f_idx[t]) for t in range(T)]
    distinct_keys = sorted(set(full_keys))

    def full_sums(b_full):
        """``{key: Σ_j β_j F_j}`` for each distinct period pattern of full columns."""
        out = {}
        for key in distinct_keys:
            acc = np.zeros((n, n))
            for bj, i in zip(b_full, key):
                acc += bj * struct.F[i]
            out[key] = acc
        return out

    def eta_periods(b):
        """``η_t = L_o⁻¹ B_t L_d⁻ᵀ`` at the current factors, per period."""
        Lo, Ld = solve_o.at(rho["rho_o"]), solve_d.at(rho["rho_d"])
        A, Bv = Lo.solve(struct.U), Ld.solve(struct.V)
        full = {key: Ld.solve(Lo.solve(Bf).T).T for key, Bf in full_sums(b[fc]).items()}
        return np.stack(
            [
                full[full_keys[t]]
                + (
                    _low_rank(A, Bv, struct.u_idx[t], struct.v_idx[t], b[cc])
                    if len(cc)
                    else 0.0
                )
                for t in range(T)
            ]
        )

    n_keep = draws // thin if thin > 0 else draws
    out = {
        "rho_d": np.empty(n_keep),
        "rho_o": np.empty(n_keep),
        "rho_w": np.empty(n_keep),
        "beta": np.empty((n_keep, k)),
        "alpha": np.empty(n_keep),
        "log_lik": np.empty((n_keep, y.size)) if store_log_lik else None,
    }
    eta = eta_periods(beta)  # (T, n, n)

    for it in range(tune + draws):
        # --- ω, working response (all periods) ---
        omega = _sample_omega(y, alpha, eta.ravel() - np.log(alpha), rng=rng)
        Oms = omega.reshape(T, n, n)
        zs = (0.5 * (y - alpha) / omega + np.log(alpha)).reshape(T, n, n)
        B_full = full_sums(beta[fc])

        # --- ρ_d, ρ_o with the rank-one block integrated out ---
        for name in ("rho_d", "rho_o"):
            if name == "rho_d":
                Lo = solve_o.at(rho["rho_o"])
                A_fix = Lo.solve(struct.U)  # origin side fixed
                P = {key: Lo.solve(Bf) for key, Bf in B_full.items()}

                def parts(rv, _P=P, _A=A_fix):
                    Ld = solve_d.at(rv)
                    eta_f = {key: Ld.solve(Pk.T).T for key, Pk in _P.items()}
                    return eta_f, _A, Ld.solve(struct.V)

            else:
                Ld = solve_d.at(rho["rho_d"])
                B_fix = Ld.solve(struct.V)  # destination side fixed
                P = {key: Ld.solve(Bf.T).T for key, Bf in B_full.items()}

                def parts(rv, _P=P, _B=B_fix):
                    Lo = solve_o.at(rv)
                    eta_f = {key: Lo.solve(Pk) for key, Pk in _P.items()}
                    return eta_f, Lo.solve(struct.U), _B

            def log_density(rv, _parts=parts):
                if not (rho_lower < rv < rho_upper):
                    return -np.inf
                try:
                    eta_f, Av, Bv_ = _parts(rv)
                except Exception:
                    return -np.inf
                Rbases = [zs[t] - eta_f[full_keys[t]] for t in range(T)]
                val = _rho_log_density(Oms, Rbases, Av, Bv_, struct, mu_c, prec_c)
                return val if np.isfinite(val) else -np.inf

            new, _, sl, sr = slice_sample_1d_adaptive(
                log_density,
                rho[name],
                lower=rho_lower,
                upper=rho_upper,
                width_state=width[name],
                rng=rng,
            )
            if it < tune:
                update_slice_width(width[name], sl, sr)
            rho[name] = float(new)

        # --- β = (β_c, β_full) jointly given ρ, ω ---
        Lo, Ld = solve_o.at(rho["rho_o"]), solve_d.at(rho["rho_d"])
        A, Bv = Lo.solve(struct.U), Ld.solve(struct.V)
        Ufull = [Ld.solve(Lo.solve(Fi).T).T for Fi in struct.F]
        G, h = _beta_gram(Oms, zs, A, Bv, Ufull, struct)
        G[np.diag_indices_from(G)] += prec0
        Lg, lower = cho_factor(G, lower=True)
        mean = cho_solve((Lg, lower), h + prec0 * mu0)
        beta = mean + solve_triangular(
            Lg, rng.standard_normal(k), lower=True, trans="T"
        )

        # --- η at the new state (reusing the transformed arrays) ---
        eta = np.stack(
            [
                sum(
                    (bj * Ufull[i] for bj, i in zip(beta[fc], struct.f_idx[t])),
                    np.zeros((n, n)),
                )
                + (
                    _low_rank(A, Bv, struct.u_idx[t], struct.v_idx[t], beta[cc])
                    if len(cc)
                    else 0.0
                )
                for t in range(T)
            ]
        )
        del Ufull

        # --- α ---
        st = _StructuralState(
            eta=eta.ravel(),
            beta=beta,
            sigma2=1.0,
            rho=rho["rho_d"],
            alpha=alpha,
            omega=omega,
        )
        alpha = _sample_alpha(st, y, core_priors, rng=rng)

        if it >= tune and (it - tune) % thin == 0:
            j = (it - tune) // thin
            if j < n_keep:
                out["rho_d"][j] = rho["rho_d"]
                out["rho_o"][j] = rho["rho_o"]
                out["rho_w"][j] = -rho["rho_d"] * rho["rho_o"]
                out["beta"][j] = beta
                out["alpha"][j] = alpha
                if store_log_lik:
                    out["log_lik"][j] = _nb_loglik_pointwise(y, eta.ravel(), alpha)
        if progress_manager is not None:
            progress_manager.update(chain_id, it, tuning=it < tune)

    return out
