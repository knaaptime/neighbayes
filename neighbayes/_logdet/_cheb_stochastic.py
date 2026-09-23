"""Stochastic Chebyshev expansion of log|I - ρW|.

Extends the Barry-Pace idea (:cite:t:`barry1999MonteCarlo`) by replacing the
Taylor series ``log(1-ρx) = -Σ ρ^k x^k / k`` with a **Chebyshev expansion** in
the operator ``W̃ = (2W - (λ_max+λ_min)I) / (λ_max-λ_min)``, following
:cite:t:`han2015LargescaleLogdeterminant`.

The Taylor truncation error is ``O(|ρλ_max|^p / (p(1-|ρ|)))`` — algebraic decay
that degrades as ``ρ → 1``.  The Chebyshev truncation error is ``O(ν^{-p})``
(Bernstein ellipse geometric decay) — **uniform across the entire ρ interval**.

Same computational structure as Barry-Pace:

* **Precompute**: ``p`` batched sparse matvecs via three-term recurrence
  ``v_{j+1} = 2W̃v_j - v_{j-1}``, ``k`` probes batched.  No reorthogonalization.
* **Per-ρ eval**: ``O(p)`` Clenshaw-like evaluation:
  ``(c₀(ρ)/2)·n + Σ c_j(ρ)·μ_j``.

**Exact low-order moments** (``n_exact``): the first ``d`` moments are replaced
by their exact values, computed from the power traces ``tr(W^k)`` in ``O(nnz)``
sparse work and no probes at all.  This is Hutchinson control-variate variance
reduction, and it places the estimator on a continuum with the literature rather
than beside it:

* :cite:t:`pace2004` is *exact traces, order 2*;
* this module at ``n_exact=1`` (the historical setting) is *exact through order
  one, probes above*;
* ``n_exact=d`` is *exact through order d, probes above*.

Pace-LeSage's low-order exact traces are therefore not a competing estimator but
a **component** of this one.  The variance reduction is large.  Holding the
order fixed at 30 on a rook lattice at ``n = 90,000``, moving from ``d=1`` to
``d=6`` cuts the RMSE 91x at ``ρ = 0.8``, 28x at ``ρ = 0.9`` and 4.2x at
``ρ = 0.99``; the sparse products cost 11-16% of the precompute at degree 4.
Because Hutchinson variance scales as ``1/K``, a 20x RMSE cut is worth ~400x the
probes.

End to end — this depth *plus* the order rule below, which it requires — the
maximum error over ``[-0.99, 0.99]`` falls by a median 3.2x at 50 probes and
4.3x at 200 on symmetric lattices, and by a median 23.7x on directed k-NN
matrices, for a median 2.6-2.9x setup.

The power traces need at most two sparse-sparse products (``tr(W³) = ⟨W², Wᵀ⟩``,
``tr(W⁴) = ‖W²‖_F²``, ``tr(W⁵) = ⟨W², W³ᵀ⟩``, ``tr(W⁶) = ‖W³‖_F²``), so ``d ≤ 2``
is free, ``d ≤ 4`` costs one product and ``d ≤ 6`` two.  Fill-in makes those
products expensive on high-degree graphs, so :func:`_resolve_exact_depth`
degrades ``d`` to 2 above ``max_degree`` mean neighbors.

**Control variates only reduce variance**, so they are wasted — and near
``ρ = 1`` actively harmful — when Chebyshev *truncation* is the binding error
instead.  At ``order=15`` and ``ρ = 0.99``, raising ``d`` from 1 to 6 makes a
rook lattice *worse* (RMSE 59.7 → 64.6), because exact moments re-weighted by a
badly truncated series do not help.  :func:`cheb_stochastic_order` therefore
picks an order whose rigorous truncation bound sits below the probe-noise floor,
and the precompute degrades ``n_exact`` to 1 with a warning when an explicitly
pinned ``order`` is too low for the requested interval.

**Deflation** (optional): When ``n_deflate > 0`` *and* ``W`` is symmetrizable
(undirected graph), the top-``n_deflate`` **eigenpairs** (by magnitude) of the
D-symmetrized, rescaled operator ``W̃_sym = D^{1/2} W̃ D^{-1/2}`` are captured
exactly via ``eigsh`` (applied matrix-free, never materialized) and removed
from the residual; stochastic Chebyshev then runs only on the deflated
residual, whose Frobenius norm — and hence Hutchinson variance — is smaller.
Because Chebyshev traces are similarity-invariant, ``tr(T_j(W̃_sym)) =
tr(T_j(W̃))``, and an eigenpair split decomposes the trace exactly
(``tr(T_j(W̃)) = Σᵢ T_j(λᵢ) − r·T_j(0) + tr(T_j(W̃_res))``) — unlike the
non-invariant singular-value split it replaced.  Directed W has an asymmetric
sparsity pattern and is not symmetrizable, so deflation is skipped with a
warning.  Spatial spectra are often flat enough that deflation barely helps,
so it is **off by default** (``n_deflate=0``).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field, replace

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from ._slq import _recover_symmetrizing_diagonal

#: Default number of exact low-order moments.  ``d = 4`` needs a single
#: sparse-sparse product (``W²``) and captures most of the available variance
#: reduction; ``d = 6`` needs a second product and roughly doubles the setup on
#: degree-8+ graphs, so it is opt-in.
DEFAULT_N_EXACT = 4

#: Hard ceiling on ``n_exact``.  Past ``tr(W⁶)`` a third sparse-sparse product
#: is required and the fill-in stops being predictable.
MAX_N_EXACT = 6

#: Mean degree (``nnz / n``) above which the sparse-sparse products behind
#: ``n_exact > 2`` are judged too expensive and the depth degrades to 2 — which
#: needs no product at all.  At degree 16 the ``W²`` product already costs ~41%
#: of a 30-order precompute, against ~11% at degree 4.
DEFAULT_MAX_DEGREE = 16.0

#: Truncation is required to sit this far below the estimated probe-noise floor
#: before it is considered non-binding (see :func:`cheb_stochastic_order`).
_ORDER_SAFETY = 0.1


@dataclass(frozen=True)
class ChebStochasticPrecompute:
    """Precomputed stochastic Chebyshev moments for log|I - ρW|.

    Attributes
    ----------
    moments : np.ndarray, shape (order + 1,)
        Stochastic estimates of ``tr(T_j(W̃))`` for ``j = 0, .., order``,
        where ``W̃ = (2W - (λ_max+λ_min)I) / (λ_max-λ_min)``.
    lam_min : float
        Lower spectral bound used for rescaling.
    lam_max : float
        Upper spectral bound used for rescaling.
    order : int
        Chebyshev polynomial degree (number of moments minus one).
    n : int
        Matrix dimension.
    n_exact : int
        Number of leading moments that were replaced by exact values.  Recorded
        for diagnostics; the evaluators do not branch on it.
    """

    moments: np.ndarray
    lam_min: float
    lam_max: float
    order: int
    n: int
    n_exact: int = 0


# ---------------------------------------------------------------------------
# Spectral bounds estimation
# ---------------------------------------------------------------------------


def _estimate_spectral_bounds(
    W: sp.csr_matrix,
    n_iters: int = 10,
    rng: np.random.Generator | None = None,
) -> tuple[float, float]:
    """Estimate [λ_min, λ_max] of W.

    For row-standardized W, λ_max = 1 (Perron) and |λ| ≤ ‖W‖_∞ = 1
    (Gershgorin), so the conservative bracket [-1, 1] is always valid.
    We use power iteration to tighten λ_max, and Gershgorin for λ_min.

    Looseness in the bracket costs convergence rate (more Chebyshev terms
    needed) but never correctness — the Chebyshev expansion still converges
    on the larger interval, just slower.

    Parameters
    ----------
    W : sp.csr_matrix
        Spatial weights matrix.
    n_iters : int, default 10
        Power iteration steps for λ_max refinement.
    rng : np.random.Generator, optional
    """
    n = W.shape[0]

    # Power iteration for λ_max
    if rng is None:
        rng = np.random.default_rng()
    v = rng.standard_normal(n)
    v /= np.linalg.norm(v)
    for _ in range(n_iters):
        v = W @ v
        norm = np.linalg.norm(v)
        if norm < 1e-300:
            break
        v /= norm
    lam_max = float(np.real(v @ (W @ v)))

    # For row-standardized W, λ_max = 1 (Perron).  Be slightly conservative.
    lam_max = max(lam_max, 1.0)

    # Gershgorin bound: |λ| ≤ ‖W‖_∞ = max row sum = 1 for row-standardized
    # Conservative: lam_min = -lam_max
    lam_min = -lam_max

    return lam_min, lam_max


# ---------------------------------------------------------------------------
# Exact low-order moments (Hutchinson control variates)
# ---------------------------------------------------------------------------


def _power_traces(W_sp: sp.csr_matrix, kmax: int) -> np.ndarray:
    """``tr(W^k)`` for ``k = 0, .., kmax`` using at most two sparse products.

    Uses ``tr(AB) = Σ_ij A_ij B_ji = ⟨A, Bᵀ⟩`` to read each trace off a
    Frobenius inner product rather than a matrix power's diagonal, so only
    ``W²`` (for ``kmax ≥ 3``) and ``W³`` (for ``kmax ≥ 5``) are ever formed.
    Valid for directed ``W`` as well — ``tr(W^k)`` is basis-free.
    """
    n = W_sp.shape[0]
    tr = np.zeros(kmax + 1, dtype=np.float64)
    tr[0] = float(n)
    if kmax >= 1:
        tr[1] = float(W_sp.diagonal().sum())
    if kmax >= 2:
        Wt = W_sp.T.tocsr()
        tr[2] = float(W_sp.multiply(Wt).sum())
    if kmax >= 3:
        W2 = (W_sp @ W_sp).tocsr()
        tr[3] = float(W2.multiply(Wt).sum())
    if kmax >= 4:
        tr[4] = float(W2.multiply(W2.T.tocsr()).sum())
    if kmax >= 5:
        W3 = (W2 @ W_sp).tocsr()
        tr[5] = float(W2.multiply(W3.T.tocsr()).sum())
    if kmax >= 6:
        tr[6] = float(W3.multiply(W3.T.tocsr()).sum())
    return tr


def _cheb_affine_power_coeffs(j: int, a: float, b: float) -> np.ndarray:
    """Power-basis coefficients of ``T_j(a·x + b)``.

    The rescaled operator is ``W̃ = a·W + b·I``, so the exact moment
    ``μ_j = tr(T_j(W̃))`` is a linear combination of the power traces
    ``tr(W^k)`` with these coefficients.  Evaluated by Horner on polynomials.
    """
    tj = np.polynomial.chebyshev.cheb2poly(np.eye(j + 1)[j])
    out = np.zeros(1, dtype=np.float64)
    for c in tj[::-1]:
        out = np.polynomial.polynomial.polymul(out, [b, a])
        out[0] += c
    return out


def _exact_cheb_moments(
    W_sp: sp.csr_matrix, depth: int, lam_min: float, lam_max: float
) -> np.ndarray:
    """Exact ``μ_j = tr(T_j(W̃))`` for ``j = 0, .., depth``, probe-free."""
    spread = lam_max - lam_min
    a = 2.0 / spread
    b = -(lam_max + lam_min) / spread
    tr = _power_traces(W_sp, depth)
    return np.array(
        [
            float(np.dot(_cheb_affine_power_coeffs(j, a, b), tr[: j + 1]))
            for j in range(depth + 1)
        ]
    )


def _resolve_exact_depth(
    W_sp: sp.csr_matrix, n_exact: int | None, max_degree: float
) -> int:
    """Effective ``n_exact`` after the mean-degree guard.

    ``n_exact=-1`` (or ``None``) auto-selects :data:`DEFAULT_N_EXACT`.  Depths
    above 2 need sparse-sparse products whose fill-in grows with degree, so on
    graphs denser than ``max_degree`` the depth degrades to 2 — still a large
    variance win, and free.
    """
    depth = DEFAULT_N_EXACT if n_exact is None or n_exact < 0 else int(n_exact)
    depth = max(0, min(depth, MAX_N_EXACT))
    if depth <= 2:
        return depth
    n = max(W_sp.shape[0], 1)
    degree = W_sp.nnz / n
    if degree > max_degree:
        warnings.warn(
            f"Mean degree {degree:.1f} exceeds max_degree={max_degree:.1f}; the "
            f"sparse products behind n_exact={depth} would dominate the "
            "precompute. Falling back to n_exact=2 (no sparse-sparse product). "
            "Raise max_degree to override.",
            stacklevel=3,
        )
        return 2
    return depth


# ---------------------------------------------------------------------------
# Order selection against the probe-noise floor
# ---------------------------------------------------------------------------

_POLE_MARGIN = 1e-6
"""Relative margin holding ρ off the stability boundary ``1/λ``."""


def _clamp_off_pole(rho: float, lam_min: float, lam_max: float) -> float:
    """Pull ``rho`` just inside the stability interval ``(1/λ_min, 1/λ_max)``.

    ``log|1 − ρλ|`` has a logarithmic pole at ``ρ = 1/λ``, and the
    Clenshaw-Curtis nodes of :func:`_log_cheb_coeffs` include both spectral
    endpoints, so ρ exactly on the boundary makes the very first node diverge.
    Row-standardized ``W`` puts that boundary at ρ = 1 — which is also the
    default ``rho_max`` — so the untreated endpoint is the *common* case, not a
    rare one.  Clamping costs nothing statistically: ρ = 1/λ_max is a unit root,
    outside the stationary region the expansion is meant to cover.  Clamping the
    argument keeps the exact coefficients of a neighbouring ρ, rather than
    fabricating a finite value at a genuine pole the way flooring the log would.

    This is the stochastic-Chebyshev counterpart of
    :func:`.._chol_cheb._clamp_interval`, which does the same job for the
    Cholesky/LU interpolants.
    """
    hi = (1.0 - _POLE_MARGIN) / lam_max if lam_max > 0.0 else np.inf
    lo = (1.0 - _POLE_MARGIN) / lam_min if lam_min < 0.0 else -np.inf
    return float(min(max(float(rho), lo), hi))


def _probe_noise_rtol(rho: float, n: int, n_probes: int) -> float:
    """Target truncation, relative to ``n``, that sits under the probe noise.

    The Hutchinson standard error of ``Σ_j c_j μ_j`` is
    ``sqrt(2/K)·‖log(I − ρW̃)‖_F``, and ``‖·‖_F ≤ sqrt(n)·max|log(1 − ρλ)|``
    over the spectrum.  Dividing by the ``O(n)`` scale of the log-determinant
    gives a relative noise floor of ``sqrt(2/(K·n))·max|log(1 − ρλ)|``; the
    truncation target is :data:`_ORDER_SAFETY` times that.  No fitted
    constants — the only judgement is the safety factor.
    """
    r = min(abs(float(rho)), 0.999999)
    worst_log = max(abs(np.log1p(-r)), abs(np.log1p(r)), 1e-12)
    noise_rel = np.sqrt(2.0 / (max(n_probes, 1) * max(n, 1))) * worst_log
    return float(_ORDER_SAFETY * noise_rel)


def cheb_stochastic_order(
    rho_min: float,
    rho_max: float,
    n: int,
    n_probes: int = 50,
    lam_min: float = -1.0,
    lam_max: float = 1.0,
    floor: int = 15,
    cap: int = 120,
    probe_order: int = 256,
) -> int:
    """Smallest order whose truncation bound sits below the probe-noise floor.

    Uses the *rigorous* tail bound rather than a fitted error model: because
    ``|T_j(x)| ≤ 1`` on ``[-1, 1]``, every moment obeys ``|μ_j| ≤ n``, so

        ``|J − J_p| = |Σ_{j>p} c_j μ_j| ≤ n · Σ_{j>p} |c_j|``

    and the order follows from the scalar coefficients alone — an ``O(p²)``
    computation independent of ``n`` and of the probes.  The bound is
    worst-case (it assumes every moment saturates and aligns in sign), so the
    selected order is conservative.

    The coefficients are largest at the interval endpoint furthest from zero,
    which is where the bound is evaluated.  For directed ``W`` the spectrum is
    complex and ``|μ_j| ≤ n`` no longer holds, so the result is a heuristic
    there — consistent with the module's general directed-``W`` caveat.
    """
    r = max(abs(float(rho_min)), abs(float(rho_max)))
    if r <= 0.0:
        return int(floor)
    # Both the target and the coefficients must refer to the same ρ, so clamp
    # once here rather than leaving each to its own guard.  Without this the
    # default interval (rho_max = 1, λ_max = 1) lands exactly on the pole, the
    # tail bound comes back inf, no order clears it, and the selection below
    # silently falls through to ``cap`` — safe, but not a selection.
    r = _clamp_off_pole(r, lam_min, lam_max)
    target = _probe_noise_rtol(r, n, n_probes)
    coeffs = np.abs(_log_cheb_coeffs(r, lam_min, lam_max, probe_order))
    tail = np.cumsum(coeffs[::-1])[::-1]
    hits = np.nonzero(tail <= target)[0]
    order = int(hits[0]) if hits.size else int(cap)
    return int(np.clip(order, floor, cap))


# ---------------------------------------------------------------------------
# Stochastic Chebyshev moments via three-term recurrence
# ---------------------------------------------------------------------------


def _chebyshev_moments(
    matvec,
    n: int,
    order: int,
    n_probes: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Estimate ``tr(T_j(A))`` for ``j = 0, .., order`` via Hutchinson probes.

    ``matvec`` applies the operator ``A`` (spectrum in [-1, 1]) to an
    ``(n, n_probes)`` block; it may be a bare sparse matmul or a matrix-free
    closure (e.g. the deflated residual).  Uses the three-term recurrence::

        v_0 = ω,  v_1 = Aω,  v_{j+1} = 2Av_j - v_{j-1}

    with moment estimate ``μ̂_j = (n/k) Σ_l ω_l^T v_j^{(l)} / ‖ω_l‖²``.

    Cost: ``order`` batched matvecs — identical to Barry-Pace, no
    reorthogonalization.

    Parameters
    ----------
    matvec : callable
        ``(n, n_probes) -> (n, n_probes)`` application of the operator.
    n : int
        Matrix dimension.
    order : int
        Maximum Chebyshev degree.
    n_probes : int
        Number of Hutchinson probe vectors.
    rng : np.random.Generator

    Returns
    -------
    np.ndarray, shape (order + 1,)
        Moment estimates ``μ̂_0, ..., μ̂_order``.  ``μ̂_0 = n`` is exact; the
        caller may override ``μ̂_1 = tr(A)`` with its exact value.
    """
    U = rng.standard_normal((n, n_probes))
    Q = _probe_moments(matvec, U, order)

    # μ_0 = tr(T_0(A)) = tr(I) = n (exact); μ̂_j = (n / k) Σ_l Q[j, l].
    moments = np.zeros(order + 1, dtype=np.float64)
    moments[0] = float(n)
    for j in range(1, order + 1):
        moments[j] = n * np.mean(Q[j])
    return moments


def _probe_moments(matvec, U: np.ndarray, order: int) -> np.ndarray:
    """Each probe's ``ωᵀ T_j(A) ω / ‖ω‖²`` for ``j = 0, .., order``.

    Returns ``Q`` of shape ``(order + 1, k)`` for the ``k`` columns of ``U``;
    the Hutchinson moment estimate is ``n`` times the mean of row ``j``.  The
    recurrence computes every probe's value anyway, so keeping them is free,
    and it is what lets a probe pool grow and measure its own spread
    (:class:`ChebStochasticPool`).
    """
    utu = np.einsum("ij,ij->j", U, U)
    Q = np.empty((order + 1, U.shape[1]), dtype=np.float64)
    Q[0] = 1.0
    v_prev = U
    v_curr = matvec(U)
    Q[1] = np.einsum("ij,ij->j", U, v_curr) / utu
    for j in range(1, order):
        v_next = 2.0 * matvec(v_curr) - v_prev
        Q[j + 1] = np.einsum("ij,ij->j", U, v_next) / utu
        v_prev, v_curr = v_curr, v_next
    return Q


def _cheb_recurrence(x: np.ndarray, order: int) -> np.ndarray:
    """Chebyshev polynomials ``T_j(x)`` for ``j = 0, .., order``.

    Parameters
    ----------
    x : np.ndarray, shape (r,)
        Evaluation points (assumed in [-1, 1]).
    order : int

    Returns
    -------
    np.ndarray, shape (order + 1, r)
        ``T[j, i] = T_j(x_i)``.
    """
    T = np.empty((order + 1, x.shape[0]), dtype=np.float64)
    T[0] = 1.0
    if order >= 1:
        T[1] = x
    for j in range(1, order):
        T[j + 1] = 2.0 * x * T[j] - T[j - 1]
    return T


def _deflated_moments(
    W_sp: sp.csr_matrix,
    D: np.ndarray,
    lam_min: float,
    lam_max: float,
    r: int,
    order: int,
    n_probes: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Eigen-deflated Chebyshev moments for a symmetrizable ``W``.

    Captures the top-``r`` eigenpairs (by magnitude) of the D-symmetrized,
    rescaled operator ``W̃_sym`` exactly and estimates the remaining moments
    on the deflated residual.  Both operators are applied matrix-free — the
    ``n × n`` low-rank correction is never materialized.

    Combination (deflated directions sit at eigenvalue 0 in the residual)::

        tr(T_j(W̃)) = Σᵢ T_j(λᵢ) − r·T_j(0) + tr(T_j(W̃_res))

    with ``T_j(0) = cos(jπ/2)``.  ``μ₀`` and ``μ₁`` are overridden with their
    exact values by the caller.
    """
    n = W_sp.shape[0]
    spread = lam_max - lam_min
    sqrt_D = np.sqrt(D)
    inv_sqrt_D = 1.0 / sqrt_D
    scale = 2.0 / spread
    shift = (lam_max + lam_min) / spread

    def wtilde_sym(B: np.ndarray) -> np.ndarray:
        """Apply ``W̃_sym = scale·(D^{1/2} W D^{-1/2}) − shift·I`` to a block."""
        two_d = B if B.ndim == 2 else B[:, None]
        WB = sqrt_D[:, None] * (W_sp @ (inv_sqrt_D[:, None] * two_d))
        out = scale * WB - shift * two_d
        return out if B.ndim == 2 else out[:, 0]

    # Top-r eigenpairs by magnitude of the rescaled symmetric operator
    # (which="LM" captures both ends of the [-1, 1] spectrum).
    op = spla.LinearOperator(
        (n, n),
        matvec=wtilde_sym,
        rmatvec=wtilde_sym,
        matmat=wtilde_sym,
        dtype=np.float64,
    )
    try:
        lam_eig, U_eig = spla.eigsh(op, k=r, which="LM")
    except spla.ArpackNoConvergence as exc:  # pragma: no cover - defensive
        lam_eig, U_eig = exc.eigenvalues, exc.eigenvectors
        r = lam_eig.shape[0]
        warnings.warn(
            "eigsh did not converge during deflation; proceeding with the "
            f"{r} eigenpairs that did converge.",
            stacklevel=2,
        )
    if r == 0:  # pragma: no cover - defensive
        U_eig = np.zeros((n, 0), dtype=np.float64)
        lam_eig = np.zeros(0, dtype=np.float64)

    def residual_matvec(B: np.ndarray) -> np.ndarray:
        """W̃_res @ B = W̃_sym @ B − U (λ ∘ (Uᵀ B))."""
        return wtilde_sym(B) - U_eig @ (lam_eig[:, None] * (U_eig.T @ B))

    res_moments = _chebyshev_moments(residual_matvec, n, order, n_probes, rng)

    # Exact contribution of the r captured eigenpairs: Σᵢ T_j(λᵢ).
    eig_moments = _cheb_recurrence(np.clip(lam_eig, -1.0, 1.0), order).sum(axis=1)

    # Deflated directions contribute T_j(0) = cos(jπ/2) = [1, 0, -1, 0, ...].
    Tj0 = np.zeros(order + 1, dtype=np.float64)
    Tj0[0::4] = 1.0
    Tj0[2::4] = -1.0

    return eig_moments - r * Tj0 + res_moments


# ---------------------------------------------------------------------------
# Chebyshev coefficients of log|a - b·x| on [-1, 1]
# ---------------------------------------------------------------------------


def _log_cheb_coeffs(
    rho: float,
    lam_min: float,
    lam_max: float,
    order: int,
) -> np.ndarray:
    """Compute Chebyshev coefficients ``c_j(ρ)`` of ``log|a - b·x|`` on [-1, 1].

    Here ``a = 1 - ρ(λ_max+λ_min)/2`` and ``b = ρ(λ_max-λ_min)/2``, so that::

        log|I - ρW| = log|a - b·W̃|

    where ``W̃ = (2W - (λ_max+λ_min)I) / (λ_max-λ_min)`` has spectrum in [-1, 1].

    Coefficients are computed via Clenshaw-Curtis quadrature (DCT-I), which is
    O(p²) — cheap for ``p ≤ 50``.

    Parameters
    ----------
    rho : float
        Spatial autoregressive parameter.  Clamped into the pole-free interval
        ``(1/λ_min, 1/λ_max)`` by :func:`_clamp_off_pole` before use.
    lam_min, lam_max : float
        Spectral bounds of W.
    order : int
        Chebyshev polynomial degree.

    Returns
    -------
    np.ndarray, shape (order + 1,)
        Chebyshev coefficients ``c_0, c_1, ..., c_order``.
    """
    # The nodes below include x = ±1, i.e. λ = λ_max and λ = λ_min, so ρ on the
    # stability boundary would evaluate log(0).  Clamp inside it first.
    rho = _clamp_off_pole(rho, lam_min, lam_max)

    a = 1.0 - rho * (lam_max + lam_min) / 2.0
    b = rho * (lam_max - lam_min) / 2.0

    if abs(b) < 1e-300:
        # ρ ≈ 0: log|a| = log(1) = 0
        return np.zeros(order + 1, dtype=np.float64)

    # Clenshaw-Curtis nodes: x_j = cos(πj/order), j = 0, ..., order
    k = np.arange(order + 1)
    x_nodes = np.cos(np.pi * k / order)
    f_vals = np.log(np.abs(a - b * x_nodes))

    # DCT-I: c_j = (2/order) * Σ w_k * f_k * cos(j*π*k/order)
    # with w_0 = w_order = 0.5, w_k = 1 otherwise
    w = np.ones(order + 1, dtype=np.float64)
    w[0] = 0.5
    w[-1] = 0.5

    coeffs = np.zeros(order + 1, dtype=np.float64)
    for j in range(order + 1):
        coeffs[j] = (2.0 / order) * np.sum(w * f_vals * np.cos(j * np.pi * k / order))
    coeffs[0] /= 2.0  # c_0/2 convention: store c₀/2 so eval uses sum c_j μ_j directly

    return coeffs


# ---------------------------------------------------------------------------
# Public API: precompute + eval
# ---------------------------------------------------------------------------


def cheb_stochastic_logdet_precompute(
    W,
    order: int | None = 15,
    n_probes: int = 50,
    n_deflate: int = 0,
    lam_min: float | None = None,
    lam_max: float | None = None,
    rng: np.random.Generator | None = None,
    n_exact: int | None = -1,
    rho_min: float | None = None,
    rho_max: float | None = None,
    max_degree: float = DEFAULT_MAX_DEGREE,
) -> ChebStochasticPrecompute:
    """Precompute stochastic Chebyshev moments for ``log|I - ρW|``.

    Parameters
    ----------
    W : array-like or scipy.sparse matrix
        Spatial weights matrix (dense or sparse).
    order : int or None, default 15
        Chebyshev polynomial degree.  Truncation converges geometrically
        (Bernstein ellipse), so 15 terms suffice for ~0.3% accuracy at ρ=0.9
        — far fewer than Barry-Pace's 20-30 Taylor terms.  Pass ``None`` to
        select the order from ``[rho_min, rho_max]`` via
        :func:`cheb_stochastic_order`, which sizes truncation against the
        probe-noise floor; that is what the library's own factories do.
    n_exact : int or None, default -1
        Number of leading moments to replace with exact, probe-free values
        (Hutchinson control variates).  ``-1`` auto-selects
        :data:`DEFAULT_N_EXACT`; ``0``/``1`` reproduce the historical
        behavior.  Depths above 2 need sparse-sparse products and are
        degraded to 2 on graphs denser than ``max_degree``.  A depth above 1
        is also degraded to 1 when ``order`` is explicitly pinned too low for
        ``[rho_min, rho_max]``, since control variates cannot help when
        truncation is the binding error.
    rho_min, rho_max : float, optional
        The ρ interval the surrogate will be evaluated on.  Used only to size
        ``order`` and to validate it against ``n_exact``; the moments
        themselves remain exactly ρ-independent.  Both are required when
        ``order is None``.  When they are omitted the order/``n_exact``
        consistency check is skipped — there is no interval to check against —
        so callers pinning a low ``order`` for use near ρ = 1 should pass them.
    max_degree : float, default 16.0
        Mean-degree ceiling for the ``n_exact > 2`` sparse products.
    n_probes : int, default 50
        Number of Hutchinson probes for moment estimation.  Since
        ``‖T_j(W̃)‖₂ ≤ 1`` uniformly in *j*, variance is bounded and 50
        probes match Barry-Pace's 100-probe accuracy at half the cost.
    n_deflate : int, default 0
        Number of top eigenpairs (by magnitude) to deflate exactly.  When
        ``n_deflate > 0`` *and* ``W`` is symmetrizable (undirected graph),
        the top-``n_deflate`` eigenpairs of the D-symmetrized, rescaled
        operator are captured with no stochastic noise (matrix-free
        ``eigsh``) and stochastic Chebyshev is applied only to the deflated
        residual.  Directed W is skipped with a warning.  Auto-selected as
        ``min(5, n // 100)`` when set to ``-1``.  Off by default: on the flat
        spectra typical of spatial ``W`` the variance reduction is usually
        marginal.
    lam_min, lam_max : float, optional
        Spectral bounds of W.  If not provided, estimated via power iteration.
    rng : np.random.Generator, optional
        Probe-vector RNG.  Defaults to a *seeded* generator so the
        precomputed moments (and thus the logdet approximation) are
        reproducible run-to-run; pass your own Generator to randomize.
    """
    if rng is None:
        rng = np.random.default_rng(0)

    if sp.issparse(W) or hasattr(W, "format"):
        W_sp = sp.csr_matrix(W, dtype=np.float64)
    else:
        W_sp = sp.csr_matrix(np.asarray(W, dtype=np.float64))

    n = W_sp.shape[0]

    # Auto-select deflation count
    if n_deflate == -1:
        n_deflate = min(5, n // 100)

    # Spectral bounds
    if lam_min is None or lam_max is None:
        est_min, est_max = _estimate_spectral_bounds(W_sp, rng=rng)
        lam_min = est_min if lam_min is None else lam_min
        lam_max = est_max if lam_max is None else lam_max

    # Order: explicit, or sized against the probe-noise floor on [ρ_min, ρ_max].
    bounds_given = rho_min is not None and rho_max is not None
    lo = -1.0 if rho_min is None else float(rho_min)
    hi = 1.0 if rho_max is None else float(rho_max)
    if order is None:
        if not bounds_given:
            raise ValueError(
                "order=None sizes the order from the ρ interval, so rho_min and "
                "rho_max must both be given. Pass them, or pin an explicit order."
            )
        order = cheb_stochastic_order(
            lo, hi, n, n_probes=n_probes, lam_min=lam_min, lam_max=lam_max
        )
    order = int(order)

    # Rescale W → W̃ with spectrum in [-1, 1]
    # W̃ = (2W - (λ_max+λ_min)I) / (λ_max-λ_min)
    spread = lam_max - lam_min
    if spread < 1e-300:
        # Degenerate case: W is a scalar multiple of I (shouldn't happen for spatial W)
        W_tilde = sp.csr_matrix((n, n), dtype=np.float64)
    else:
        W_tilde = (2.0 / spread) * W_sp - ((lam_max + lam_min) / spread) * sp.eye(
            n, format="csr"
        )
        W_tilde = W_tilde.tocsr()

    # Exact μ₁ = tr(W̃) (variance reduction; overrides the stochastic estimate).
    mu1_exact = float(W_tilde.diagonal().sum())

    # Deflation (symmetrizable W only): capture the top-r eigenpairs exactly
    # and run stochastic Chebyshev on the lower-variance deflated residual.
    r = min(n_deflate, n - 2) if n_deflate > 0 else 0
    D = _recover_symmetrizing_diagonal(W_sp) if r >= 1 and spread >= 1e-300 else None
    if n_deflate > 0 and r >= 1 and D is None:
        warnings.warn(
            "Eigen-deflation requires a symmetrizable (undirected) W, but the "
            "sparsity pattern of W is asymmetric (directed graph). Falling back "
            "to plain stochastic Chebyshev; n_deflate is ignored.",
            stacklevel=2,
        )

    if D is not None:
        moments = _deflated_moments(W_sp, D, lam_min, lam_max, r, order, n_probes, rng)
    else:
        # No deflation: standard stochastic Chebyshev on W̃.
        moments = _chebyshev_moments(lambda B: W_tilde @ B, n, order, n_probes, rng)

    # Exact low-order overrides (control variates): μ₀ = n, μ₁ = tr(W̃), and —
    # when n_exact > 1 — μ₂..μ_d from the power traces.  Depth is capped by the
    # order actually in use, since exact moments cannot repair truncation.
    depth = _resolve_exact_depth(W_sp, n_exact, max_degree)
    if depth > 1 and bounds_given:
        needed = cheb_stochastic_order(
            lo, hi, n, n_probes=n_probes, lam_min=lam_min, lam_max=lam_max
        )
        if order < needed:
            warnings.warn(
                f"order={order} is below the {needed} required for "
                f"rho in [{lo:.3g}, {hi:.3g}] at n_probes={n_probes}: Chebyshev "
                "truncation, not probe variance, is the binding error, so exact "
                "moments cannot help and may hurt. Falling back to n_exact=1; "
                "pass order=None to size the order automatically.",
                stacklevel=2,
            )
            depth = 1
    depth = min(depth, order)

    moments[0] = float(n)
    if depth >= 1:
        moments[1] = mu1_exact
    if depth > 1:
        moments[: depth + 1] = _exact_cheb_moments(W_sp, depth, lam_min, lam_max)

    return ChebStochasticPrecompute(
        moments=moments,
        lam_min=lam_min,
        lam_max=lam_max,
        order=order,
        n=n,
        n_exact=depth,
    )


def cheb_stochastic_logdet_eval(pre: ChebStochasticPrecompute, rho: float) -> float:
    """Evaluate ``log|I - ρW|`` from precomputed stochastic Chebyshev moments.

    Computes Chebyshev coefficients ``c_j(ρ)`` on-the-fly (O(p²) via
    Clenshaw-Curtis), then evaluates::

        log|I - ρW| ≈ (c₀/2)·n + Σ_{j=1}^p c_j·μ_j

    Parameters
    ----------
    pre : ChebStochasticPrecompute
        Precomputed moments from :func:`cheb_stochastic_logdet_precompute`.
    rho : float
        Spatial autoregressive parameter.
    """
    coeffs = _log_cheb_coeffs(rho, pre.lam_min, pre.lam_max, pre.order)
    # log|I - ρW| = Σ c_j · μ_j  (c₀ already includes /2 convention from _log_cheb_coeffs)
    val = 0.0
    for j in range(pre.order + 1):
        val += coeffs[j] * pre.moments[j]
    return float(val)


def cheb_stochastic_logdet_eval_vec(
    pre: ChebStochasticPrecompute, rho_arr: np.ndarray
) -> np.ndarray:
    """Vectorized evaluation over an array of ρ values.

    Precomputes Chebyshev coefficients for each ρ, then evaluates via
    matrix-vector product (coeffs @ moments).
    """
    rho_arr = np.asarray(rho_arr, dtype=np.float64)
    n_rho = len(rho_arr)

    # Build coefficient matrix: (n_rho, order+1)
    all_coeffs = np.zeros((n_rho, pre.order + 1), dtype=np.float64)
    for i in range(n_rho):
        all_coeffs[i] = _log_cheb_coeffs(
            rho_arr[i], pre.lam_min, pre.lam_max, pre.order
        )

    # logdet_i = Σ c_j · μ_j  (c₀ already includes /2 convention)
    return all_coeffs @ pre.moments


# ---------------------------------------------------------------------------
# Probe pools: the probe count sized by the estimator's own spread
# ---------------------------------------------------------------------------

#: Probes a pool starts with: the fixed estimator's count, so sizing a pool only
#: ever adds probes.  A smaller start saves setup time alone, since per-draw cost
#: does not depend on the count, and the saving is small (0.8 s against 3.2 s for
#: 50 at n = 490,000).  It also leaves the spread estimated from too few probes:
#: starting from 12, 17% of validation runs at ρ = 0.9 exceeded 0.045 posterior
#: sd, against none starting from 50.
DEFAULT_PROBE_MIN = 50

#: Largest pool.  It also sizes the Chebyshev order, so truncation stays under
#: the probe noise at every size the pool can grow to.
DEFAULT_PROBE_MAX = 200

#: Target for the expected |bias| of the posterior mean of ρ, in posterior
#: standard deviations: 1/√500, the Monte Carlo error of a posterior mean with
#: an effective sample size of 500.
DEFAULT_PROBE_BAR = 0.045

#: Safety divisor on the bar.  The spread is estimated from the probes in hand,
#: and the pool stops on the first estimate under the target, so the realized
#: bias runs somewhat above the prediction.
DEFAULT_PROBE_Z = 2.0

#: E|N(0, 1)|, which turns the standard deviation of the bias into its mean size.
_ABS_NORMAL = float(np.sqrt(2.0 / np.pi))


@dataclass(frozen=True)
class ChebStochasticPool:
    """Per-probe stochastic Chebyshev moments that can grow.

    The same estimator as :func:`cheb_stochastic_logdet_precompute`, with each
    probe's moments kept rather than averaged away.  Probe ``l`` is drawn from
    its own stream, ``default_rng([seed, l])``, so a pool of ``s`` probes is the
    same however it grew, and growing one runs the recurrence on new probes only.

    Attributes
    ----------
    probe_moments : np.ndarray, shape (order + 1, n_probes)
        Each probe's ``ωᵀ T_j(W̃) ω / ‖ω‖²``.
    exact_moments : np.ndarray, shape (n_exact + 1,)
        Probe-free ``μ_0 .. μ_d`` that replace the estimates (control variates).
    lam_min, lam_max, order, n, n_exact
        As in :class:`ChebStochasticPrecompute`.
    seed : int
        Root of the per-probe streams.
    W_tilde : scipy.sparse.csr_matrix
        The rescaled operator, kept so the pool can grow.
    """

    probe_moments: np.ndarray
    exact_moments: np.ndarray
    lam_min: float
    lam_max: float
    order: int
    n: int
    n_exact: int
    seed: int
    W_tilde: sp.csr_matrix = field(repr=False, compare=False)

    @property
    def n_probes(self) -> int:
        """Number of probes in the pool."""
        return int(self.probe_moments.shape[1])


def _pool_probes(seed: int, start: int, stop: int, n: int) -> np.ndarray:
    """Probes ``start .. stop - 1`` of a pool, as columns of an ``(n, k)`` block."""
    U = np.empty((n, stop - start), dtype=np.float64)
    for col, idx in enumerate(range(start, stop)):
        U[:, col] = np.random.default_rng([seed, idx]).standard_normal(n)
    return U


def cheb_stochastic_pool(
    W,
    n_probes: int = DEFAULT_PROBE_MIN,
    *,
    max_probes: int = DEFAULT_PROBE_MAX,
    rho_min: float | None = None,
    rho_max: float | None = None,
    n_exact: int | None = -1,
    max_degree: float = DEFAULT_MAX_DEGREE,
    lam_min: float | None = None,
    lam_max: float | None = None,
    seed: int = 0,
) -> ChebStochasticPool:
    """Start a probe pool for ``log|I - ρW|`` with ``n_probes`` probes.

    Spectral bounds and exact moments are those of
    :func:`cheb_stochastic_logdet_precompute`.  The order is sized against the
    noise of ``max_probes`` probes, so truncation stays under the probe noise at
    every size the pool can grow to.  Deflation is not offered: it changes the
    operator the probes see.
    """
    if sp.issparse(W) or hasattr(W, "format"):
        W_sp = sp.csr_matrix(W, dtype=np.float64)
    else:
        W_sp = sp.csr_matrix(np.asarray(W, dtype=np.float64))
    n = W_sp.shape[0]
    if lam_min is None or lam_max is None:
        est_min, est_max = _estimate_spectral_bounds(
            W_sp, rng=np.random.default_rng(seed)
        )
        lam_min = est_min if lam_min is None else lam_min
        lam_max = est_max if lam_max is None else lam_max
    lo = -1.0 if rho_min is None else float(rho_min)
    hi = 1.0 if rho_max is None else float(rho_max)
    order = cheb_stochastic_order(
        lo, hi, n, n_probes=max_probes, lam_min=lam_min, lam_max=lam_max
    )
    spread = lam_max - lam_min
    W_tilde = (2.0 / spread) * W_sp - ((lam_max + lam_min) / spread) * sp.eye(
        n, format="csr"
    )
    W_tilde = W_tilde.tocsr()

    depth = min(_resolve_exact_depth(W_sp, n_exact, max_degree), order)
    if depth > 1:
        exact = _exact_cheb_moments(W_sp, depth, lam_min, lam_max)
    else:
        exact = np.array([float(n), float(W_tilde.diagonal().sum())])[: depth + 1]

    U = _pool_probes(seed, 0, int(n_probes), n)
    Q = _probe_moments(lambda B: W_tilde @ B, U, order)
    return ChebStochasticPool(
        probe_moments=Q,
        exact_moments=exact,
        lam_min=float(lam_min),
        lam_max=float(lam_max),
        order=int(order),
        n=int(n),
        n_exact=int(depth),
        seed=int(seed),
        W_tilde=W_tilde,
    )


def grow_pool(pool: ChebStochasticPool, n_probes: int) -> ChebStochasticPool:
    """The pool with probes appended up to ``n_probes`` (itself if it has them)."""
    have = pool.n_probes
    if int(n_probes) <= have:
        return pool
    U = _pool_probes(pool.seed, have, int(n_probes), pool.n)
    Q = _probe_moments(lambda B: pool.W_tilde @ B, U, pool.order)
    return replace(pool, probe_moments=np.concatenate([pool.probe_moments, Q], axis=1))


def pool_precompute(pool: ChebStochasticPool) -> ChebStochasticPrecompute:
    """The estimator over every probe in the pool, for the ordinary evaluators."""
    moments = pool.n * pool.probe_moments.mean(axis=1)
    moments[0] = float(pool.n)
    moments[: pool.n_exact + 1] = pool.exact_moments
    return ChebStochasticPrecompute(
        moments=moments,
        lam_min=pool.lam_min,
        lam_max=pool.lam_max,
        order=pool.order,
        n=pool.n,
        n_exact=pool.n_exact,
    )


def _log_cheb_coeffs_vec(
    rho: np.ndarray, lam_min: float, lam_max: float, order: int
) -> np.ndarray:
    """:func:`_log_cheb_coeffs` for an array of ρ, as an ``(R, order + 1)`` array."""
    hi = (1.0 - _POLE_MARGIN) / lam_max if lam_max > 0.0 else np.inf
    lo = (1.0 - _POLE_MARGIN) / lam_min if lam_min < 0.0 else -np.inf
    r = np.clip(np.asarray(rho, dtype=np.float64), lo, hi)
    a = 1.0 - r * (lam_max + lam_min) / 2.0
    b = r * (lam_max - lam_min) / 2.0
    k = np.arange(order + 1)
    x_nodes = np.cos(np.pi * k / order)
    with np.errstate(divide="ignore"):
        f = np.log(np.abs(a[:, None] - b[:, None] * x_nodes[None, :]))
    w = np.ones(order + 1, dtype=np.float64)
    w[0] = w[-1] = 0.5
    basis = np.cos(np.pi * np.outer(k, k) / order)
    coeffs = (2.0 / order) * (f * w) @ basis.T
    coeffs[:, 0] /= 2.0
    coeffs[np.abs(b) < 1e-300] = 0.0
    return coeffs


def pool_probe_bias(
    pool: ChebStochasticPool, rho_draws, *, T: int = 1, max_draws: int = 4000
) -> np.ndarray | None:
    """Each probe's first-order shift of the posterior mean of ρ, in posterior sd.

    A log-density error ``e(ρ)`` moves the posterior mean by ``Cov(ρ, e(ρ))`` to
    first order.  Posterior draws of ρ estimate that covariance directly, with
    equal weights, since the draws already carry the posterior; weighting them
    by the density again would sample its square and understate the spread by
    ``1 − 1/√2``.  Probe ``l``'s error is its own estimate of the
    log-determinant; the exact moments shift every probe equally and drop out.
    The bias of an ``s``-probe estimate then has standard deviation
    ``std(q)/√s``.  Returns ``None`` when the draws have no spread.
    """
    rho = np.asarray(rho_draws, dtype=np.float64).ravel()
    rho = rho[np.isfinite(rho)]
    if rho.size > max_draws:
        rho = rho[np.linspace(0, rho.size - 1, max_draws).astype(int)]
    sd = float(rho.std()) if rho.size > 1 else 0.0
    if not sd > 0.0:
        return None
    d = pool.n_exact + 1
    C = _log_cheb_coeffs_vec(rho, pool.lam_min, pool.lam_max, pool.order)
    P = (float(T) * pool.n) * (C[:, d:] @ pool.probe_moments[d:])
    return ((rho - rho.mean()) @ (P - P.mean(axis=0))) / (rho.size * sd)


def probes_needed(
    q: np.ndarray,
    bar: float = DEFAULT_PROBE_BAR,
    z: float = DEFAULT_PROBE_Z,
) -> int:
    """Probes at which the expected |bias| of the posterior mean falls to ``bar / z``.

    ``E|bias| = √(2/π) · std(q) / √s``, solved for ``s``.  The error is taken
    against the exact log-determinant, so no finite-pool correction applies.
    """
    spread = float(np.std(q, ddof=1)) if len(q) > 1 else 0.0
    return int(np.ceil((_ABS_NORMAL * z * spread / bar) ** 2))


@dataclass(frozen=True)
class ProbeCheck:
    """How a probe pool was sized against the posterior of ρ.

    Attributes
    ----------
    n_probes : int
        Probes in the pool after sizing; the estimator uses all of them.
    spread : float
        ``std(q)``: standard deviation of one probe's shift of the posterior
        mean, in posterior sd.
    bias : float
        Expected |bias| of the posterior mean at ``n_probes``, in posterior sd.
    bar, z : float
        The target, met when ``bias <= bar / z``.
    capped : bool
        Whether the pool stopped at its maximum short of the target.
    """

    n_probes: int
    spread: float
    bias: float
    bar: float
    z: float
    capped: bool


def size_pool(
    pool: ChebStochasticPool,
    rho_draws,
    *,
    bar: float = DEFAULT_PROBE_BAR,
    z: float = DEFAULT_PROBE_Z,
    max_probes: int = DEFAULT_PROBE_MAX,
    T: int = 1,
) -> tuple[ChebStochasticPool, ProbeCheck | None]:
    """Grow ``pool`` until its own spread says it holds enough probes.

    Each round re-estimates the spread from every probe in hand, so the count
    settles as the estimate firms up; the pool only grows, so the loop ends at
    the target or at ``max_probes``.  Returns the pool and the check, or the
    pool unchanged and ``None`` when the draws cannot price the probes.
    """
    while True:
        q = pool_probe_bias(pool, rho_draws, T=T)
        if q is None:
            return pool, None
        need = probes_needed(q, bar, z)
        if min(need, int(max_probes)) <= pool.n_probes:
            break
        pool = grow_pool(pool, min(need, int(max_probes)))
    spread = float(np.std(q, ddof=1))
    s = pool.n_probes
    check = ProbeCheck(
        n_probes=s,
        spread=spread,
        bias=_ABS_NORMAL * spread / np.sqrt(s),
        bar=float(bar),
        z=float(z),
        capped=bool(need > s),
    )
    return pool, check
