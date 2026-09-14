r"""Pólya–Gamma Gibbs sampler for the *reduced-form* SAR-NB model.

Target posterior
----------------
.. math::

    y_i \sim \mathrm{NegBin}(\mu_i, \alpha), \quad
    \mu_i = \exp(\eta_i), \quad
    \eta = (I - \rho W)^{-1} X \beta.

Parameters are :math:`(\beta, \rho, \alpha)`.  Unlike
:mod:`neighbayes.samplers.negbin`, the latent field :math:`\eta` is a
*deterministic* function of :math:`(\beta, \rho)` — there is no
:math:`\sigma`-noise term and no n-dimensional augmentation step.

Pólya–Gamma augmentation
------------------------
With auxiliary variables :math:`\omega_i \sim \mathrm{PG}(y_i + \alpha,
\psi_i)` where :math:`\psi_i = \eta_i - \log\alpha`, the NB
log-likelihood becomes quadratic in :math:`\psi` with working response
:math:`\kappa_i = (y_i - \alpha)/2`.  Writing
:math:`\tilde X = (I - \rho W)^{-1} X`, the conditional posterior of
:math:`\beta` is the conjugate Gaussian

.. math::

    \beta \mid \omega, \rho, \alpha, y \;\sim\; N(m_\beta, \Sigma_\beta), \\
    \Sigma_\beta^{-1} = \tilde X^\top \Omega \tilde X + V_0^{-1}, \\
    m_\beta = \Sigma_\beta \bigl(\tilde X^\top (\kappa + \omega \log\alpha)
                                  + V_0^{-1} \mu_0\bigr).

Sweep
-----
Four blocks per iteration:

1. **ω | β, ρ, α, y** — vectorized PG draw at :math:`\psi`.
2. **β | ω, ρ, α, y** — conjugate normal via the construction above.
   Requires building :math:`\tilde X = A_\rho^{-1} X` (one sparse LU
   factorization of :math:`A_\rho = I - \rho W` plus k triangular
   solves), then a :math:`k \times k` Cholesky.
3. **ρ | ω, α, y** — 1-D adaptive slice sampler on the
   **β-marginalized** conditional density.  With working response
   :math:`s_i = (y_i - \alpha) / (2\omega_i) + \log\alpha` and
   :math:`U(\rho) = A_\rho^{-1} X`, integrating out :math:`\beta\sim
   N(b_0, V_0)` gives :math:`s\mid\rho,\omega,\alpha \sim
   N(U b_0,\,\Omega^{-1} + U V_0 U^\top)`.  Via the matrix-determinant
   lemma / Woodbury identity with :math:`M(\rho) = V_0^{-1} + U^\top
   \Omega U` and :math:`r = s - U b_0`,

   .. math::

       \log p(\rho \mid \cdot)
         = -\tfrac{1}{2} \log |M(\rho)|
           - \tfrac{1}{2}\bigl(r^\top \Omega r
               - (U^\top \Omega r)^\top M^{-1} (U^\top \Omega r)\bigr)
           + \log p_0(\rho),

   up to terms independent of :math:`\rho`.  Marginalizing β inside
   the ρ update breaks the β–ρ posterior correlation that would
   otherwise dominate single-site ρ mixing.  No :math:`\log|A_\rho|`
   Jacobian appears (η is not being integrated out).
4. **α | y, η** — 1-D slice on :math:`\log\alpha` of the NB
   log-likelihood + Half-Student-t prior.  Reuses
   :func:`neighbayes.samplers.negbin._core._sample_alpha`.

Per-sweep cost is dominated by the :math:`\rho` slice: candidates
within the Krylov radius are evaluated via a cheap Horner polynomial,
while candidates outside the radius use CG iterative solves
(:math:`O(K \cdot \mathrm{nnz})` per candidate, where :math:`K \approx
\sqrt{\kappa}`).  For :math:`n < 2500`, the Krylov basis build uses
CHOLMOD factorization (fast at small n); for :math:`n \geq 2500`,
it uses CG iterative solves (avoids the :math:`O(\mathrm{nnz}^{1.5})`
factorization cost).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple, Optional

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from ...models.priors import ReducedGibbsPriors
from .._utils._polyagamma import sample_polyagamma
from .._utils._slice import (
    SliceWidthState,
    slice_sample_1d,
    slice_sample_1d_adaptive,
    update_slice_width,
)
from .._utils._spatial_normal import CholmodFactor, _series_radius
from ..negbin._core import _nb_loglik_pointwise, _sample_alpha

# ---------------------------------------------------------------------------
# Sparse solve for A_ρ = I − ρW
# ---------------------------------------------------------------------------

# When CHOLMOD is available we solve (I − ρW) x = b via the normal
# equations  A^T A x = A^T b  where  A^T A = I − ρ(W+W^T) + ρ² W^T W
# is SPD.  This avoids scipy SuperLU (``splu``), which can deadlock on macOS
# when Apple Accelerate BLAS is called concurrently from multiple
# processes.  CHOLMOD is also faster because it reuses the symbolic
# analysis across all ρ values.


class _CholmodNormalEqSolver:
    """Solve ``(I − ρW) x = b`` via CHOLMOD on the normal equations.

    The normal-equation matrix

    .. math::

        A_\\rho^T A_\\rho = I - \\rho (W + W^T) + \\rho^2 W^T W

    is symmetric positive definite for any non-singular ``A_ρ``.
    CHOLMOD factorizes it and we recover ``x = A_ρ⁻¹ b`` from
    ``A_ρ^T A_ρ x = A_ρ^T b``.

    The ``cholmod_factor`` holds the symbolic analysis (computed once
    from a pattern matrix) so that only the numeric factorization is
    needed for each new ρ.

    Parameters
    ----------
    cholmod_factor : CholmodFactor
        Pre-built CHOLMOD factor with the sparsity pattern of
        ``A^T A`` (any valid ρ gives the same pattern).
    W_csc : csc_matrix
        Spatial weights in CSC format.
    W_sym : csc_matrix
        ``W + W^T`` in CSC format (precomputed).
    WtW : csc_matrix
        ``W^T @ W`` in CSC format (precomputed).
    n : int
        Dimension of the spatial weights matrix.
    """

    def __init__(
        self,
        cholmod_factor: CholmodFactor,
        W_csc: sp.csc_matrix,
        W_sym: sp.csc_matrix,
        WtW: sp.csc_matrix,
        n: int,
    ) -> None:
        self._cholmod = cholmod_factor
        self._W_csc = W_csc
        self._W_sym = W_sym
        self._WtW = WtW
        self._n = n
        self._rho: float | None = None

    def factorize(self, rho: float) -> None:
        """Build and factorize ``A^T A`` at the given ρ."""
        AtA = sp.eye(self._n, format="csc") - rho * self._W_sym + rho**2 * self._WtW
        self._cholmod.factorize(AtA)
        self._rho = rho

    def solve(self, rhs: np.ndarray) -> np.ndarray:
        """Solve ``(I − ρW) x = rhs`` via the normal equations.

        Works for both vector and matrix right-hand sides.
        """
        # A^T b  where A = I − ρW
        Atb = rhs - self._W_csc.T @ (self._rho * rhs)
        # CHOLMOD solve: A^T A x = A^T b
        return self._cholmod.solve(Atb)


class _DSymCholSolver:
    r"""Solve :math:`(I-\rho W)x = b` by Cholesky on the D-symmetrized twin.

    A row-standardized ``W = D^{-1}A`` with symmetric adjacency ``A`` is not
    itself symmetric — each row carries its own degree — but it is *similar*
    to a symmetric matrix:

    .. math::

        S = D^{1/2} W D^{-1/2} = D^{-1/2} A D^{-1/2},

    with the same eigenvalues as ``W``.  Hence
    :math:`M(\rho) = D^{1/2}(I-\rho W)D^{-1/2} = I - \rho S` is symmetric,
    and positive definite wherever ``ρ`` is inside the stability range.

    This is the best of the three routes when it applies.  ``M`` carries
    ``W``'s own sparsity, so its Cholesky is far cheaper than one on
    :math:`A^\top A` (two-hop fill) and cheaper than an LU of ``A``; it also
    avoids squaring the condition number, and :math:`\log\det A = \log\det M`
    comes free from the same factorization.

    Solves map back through the similarity:
    :math:`x = D^{-1/2} M^{-1} D^{1/2} b`.

    Parameters
    ----------
    W_csc : csc_matrix
        Row-standardized weights, known to be D-symmetrizable.
    d : ndarray, shape (n,)
        Symmetrizing diagonal from
        :func:`neighbayes._logdet._slq._recover_symmetrizing_diagonal`.
    n : int
        Matrix dimension.
    """

    def __init__(self, W_csc: sp.csc_matrix, d: np.ndarray, n: int) -> None:
        sq = np.sqrt(np.asarray(d, dtype=np.float64))
        inv_sq = 1.0 / sq
        S = (sp.diags(sq) @ sp.csc_matrix(W_csc) @ sp.diags(inv_sq)).tocsc()
        # Symmetrize explicitly: the similarity makes S symmetric in exact
        # arithmetic, but CHOLMOD reads a single triangle, so any residual
        # round-off asymmetry would silently change the matrix being factored.
        self._S = (0.5 * (S + S.T)).tocsc()
        self._sq = sq
        self._inv_sq = inv_sq
        self._n = n
        # I + 0.5·S has the pattern of I − ρS for every ρ and is SPD
        # (eigenvalues 1 + 0.5λ ∈ [0.5, 1.5] for λ(S) ∈ [−1, 1]).
        self._chol = CholmodFactor((sp.eye(n, format="csc") + 0.5 * self._S).tocsc())
        self._rho: float | None = None

    def factorize(self, rho: float) -> None:
        """Factorize ``M = I − ρS``."""
        M = (sp.eye(self._n, format="csc") - float(rho) * self._S).tocsc()
        self._chol.factorize(M)
        self._rho = float(rho)

    def solve(self, rhs: np.ndarray) -> np.ndarray:
        """Solve ``(I − ρW) x = rhs`` (vector or matrix RHS)."""
        arr = np.asarray(rhs, dtype=np.float64)
        scale = self._sq[:, None] if arr.ndim > 1 else self._sq
        unscale = self._inv_sq[:, None] if arr.ndim > 1 else self._inv_sq
        return self._chol.solve(arr * scale) * unscale

    def logdet(self) -> float:
        """``log det(I − ρW)``, equal to ``log det M`` by similarity."""
        return self._chol.logdet()


def make_sar_solver(
    cholmod_factor: CholmodFactor,
    W_csc: sp.csc_matrix,
    W_sym: sp.csc_matrix,
    WtW: sp.csc_matrix,
    n: int,
    force: str | None = None,
    fill_threshold: float = 1.5,
):
    r"""Return the right :math:`(I-\rho W)` solver for this ``W``.

    Both returned types expose ``factorize(rho)`` / ``solve(rhs)``.

    Three routes, preferred in this order:

    1. **D-symmetrized Cholesky** (:class:`_DSymCholSolver`) whenever ``W`` is
       D-symmetrizable — i.e. ``W = D^{-1}A`` for a symmetric adjacency ``A``,
       which covers every row-standardized undirected graph.  ``I − ρW`` is
       then similar to the symmetric ``I − ρS``, carrying ``W``'s own
       sparsity.  Cheapest of the three, and exact.
    2. **KLU on ``A``** (:class:`KluSarSolver`) for genuinely directed ``W``
       (flow matrices, asymmetric k-nearest-neighbor graphs), where no
       symmetrizing diagonal exists.
    3. **CHOLMOD normal equations** (:class:`_CholmodNormalEqSolver`) as the
       fallback when neither applies — correct for any non-singular ``A``,
       but it pays :math:`W^\top W`'s two-hop fill and squares the condition
       number.

    Raw symmetry is deliberately *not* the test.  Row-standardizing divides
    each row by its own degree, so a symmetric adjacency generally yields an
    asymmetric ``W``; keying on ``W == Wᵀ`` would reject the very cases route
    1 handles best.  Measured on row-standardized weights (6 RHS, one
    factorization each):

    ==========  ====  ========  =======  ========
    graph          n   AᵀA         KLU     D-sym
    ==========  ====  ========  =======  ========
    ring        6000    3.2 ms   3.9 ms
    queen       2500   31.9 ms  13.3 ms
    knn (k=6)   6000  277.6 ms  188.0 ms
    ==========  ====  ========  =======  ========

    Parameters
    ----------
    cholmod_factor : CholmodFactor
        Pre-built factor carrying the ``AᵀA`` symbolic analysis.
    W_csc, W_sym, WtW : csc_matrix
        ``W``, ``W + Wᵀ`` and ``WᵀW``.
    n : int
        Matrix dimension.
    force : {"cholmod", "klu", "dsym"}, optional
        Override the automatic routing (mainly for tests and benchmarks).
    fill_threshold : float, default 1.5
        Only consulted on the fallback path, when ``W`` is not
        D-symmetrizable and KLU is unavailable.

    Returns
    -------
    _DSymCholSolver, KluSarSolver or _CholmodNormalEqSolver
    """
    from .._utils._sparsax_utils import KluSarSolver

    def _cholmod():
        return _CholmodNormalEqSolver(
            cholmod_factor=cholmod_factor,
            W_csc=W_csc,
            W_sym=W_sym,
            WtW=WtW,
            n=n,
        )

    if force == "cholmod":
        return _cholmod()
    if force == "klu":
        return KluSarSolver(W_csc, n)
    if force == "dsym":
        return _DSymCholSolver(W_csc, _symmetrizing_diagonal(W_csc), n)

    d = _symmetrizing_diagonal(W_csc)
    if d is not None:
        return _DSymCholSolver(W_csc, d, n)

    from ..._jax_dispatch import _sparsax_available

    if _sparsax_available():
        return KluSarSolver(W_csc, n)
    return _cholmod()


def _symmetrizing_diagonal(W_csc: sp.csc_matrix) -> np.ndarray | None:
    """Return a positive ``D`` with ``D^{1/2}WD^{-1/2}`` symmetric, else ``None``.

    Thin guard around
    :func:`neighbayes._logdet._slq._recover_symmetrizing_diagonal`: that
    routine returns ``None`` for directed graphs and can return a
    sign-inconsistent ``D`` for weights that are not of the form ``D⁻¹A``, so
    we additionally require ``D > 0`` and verify the resulting similarity is
    symmetric before trusting it.
    """
    from ..._logdet._slq import _recover_symmetrizing_diagonal

    try:
        d = _recover_symmetrizing_diagonal(sp.csr_matrix(W_csc))
    except Exception:
        return None
    if d is None or not np.all(np.isfinite(d)) or np.any(d <= 0.0):
        return None
    sq = np.sqrt(d)
    S = (sp.diags(sq) @ sp.csc_matrix(W_csc) @ sp.diags(1.0 / sq)).tocsr()
    diff = (S - S.T).tocoo()
    scale = max(np.abs(S.data).max(), 1.0) if S.nnz else 1.0
    if diff.nnz and np.max(np.abs(diff.data)) > 1e-9 * scale:
        return None
    return d


def _factor_A(rho: float, W_csc: sp.csc_matrix, n: int):
    """Factorize :math:`A_\\rho = I - \\rho W` via the backend sparse solver.

    Returns a factor object whose ``.solve(rhs)`` method handles single
    and multiple right-hand sides.  Uses KLU via
    :func:`neighbayes._ops._backend._sparse_factor` when available,
    falling back to scipy SuperLU.

    .. deprecated::
        Used only as a fallback when CHOLMOD is not available.
        The CHOLMOD normal-equations path (``_CholmodNormalEqSolver``)
        is preferred to avoid scipy SuperLU deadlocks on macOS.
    """
    from ..._ops._backend import _select_sparse_backend, _sparse_factor

    A = (sp.eye(n, format="csc") - rho * W_csc).tocsc()
    backend = _select_sparse_backend()
    return _sparse_factor(A, backend)


def _make_cholmod_pattern(
    W_csc: sp.csc_matrix,
    n: int,
) -> tuple[sp.csc_matrix, sp.csc_matrix, sp.csc_matrix]:
    """Precompute the CHOLMOD pattern matrix and sparse building blocks.

    Returns
    -------
    W_sym : csc_matrix
        ``W + W^T`` in CSC format.
    WtW : csc_matrix
        ``W^T @ W`` in CSC format.
    pattern : csc_matrix
        Sparsity pattern for ``A^T A = I − ρ(W+W^T) + ρ² W^T W``
        that covers all valid ρ.  Built as
        ``I + 0.5*(W+W^T) + 0.25*W^T@W`` so that every possible
        fill-in position is present.
    """
    W_sym = (W_csc + W_csc.T).tocsc()
    WtW = (W_csc.T @ W_csc).tocsc()
    pattern = (sp.eye(n, format="csc") + 0.5 * W_sym + 0.25 * WtW).tocsc()
    return W_sym, WtW, pattern


def _make_solver(
    rho: float,
    W_csc: sp.csc_matrix,
    n: int,
    cholmod_solver: _CholmodNormalEqSolver | None = None,
) -> _CholmodNormalEqSolver | object:
    """Return a solver for ``(I − ρW) x = b``.

    When ``cholmod_solver`` is provided (CHOLMOD available), factorizes
    the normal-equation matrix ``A^T A`` and returns the solver.
    Otherwise falls back to ``splu`` (scipy SuperLU).

    Both return types expose a ``.solve(rhs)`` method.
    """
    if cholmod_solver is not None:
        cholmod_solver.factorize(rho)
        return cholmod_solver
    return _factor_A(rho, W_csc, n)


# ---------------------------------------------------------------------------
# Shift-invert Krylov basis for fast ρ-slice evaluation
# ---------------------------------------------------------------------------

# Default Krylov degree and maximum |Δρ| for polynomial approximation.
# Beyond ``krylov_dmax`` (or the basis' own convergence radius) both backends
# fall back to a direct solve per slice candidate, so a wide dmax — with enough
# degree to keep the Horner approximation accurate — keeps those solves rare.
_KRYLOV_DEGREE_DEFAULT = 12
_KRYLOV_DMAX_DEFAULT = 0.4

# CG is now only a *fallback* for when no direct solver could be built, not a
# large-n strategy.
#
# It used to take over above n = 2500, back when the only direct option was
# CHOLMOD on the normal equations ``AᵀA`` — which pays ``WᵀW``'s two-hop fill
# and squares the condition number, so it did lose to CG as n grew.  Routing
# now picks a factorization matched to the weights (``make_sar_solver``):
# Cholesky on the D-symmetrized ``I − ρS``, which carries ``W``'s own
# sparsity, or KLU for genuinely directed ``W``.  Measured on queen
# contiguity, one degree-12 basis build (13 multi-RHS solves), CG vs the
# routed factorization:
#
#   n =  1600   9.3 ms vs  1.7 ms   (5.55x)
#   n =  2500  11.6 ms vs  2.7 ms   (4.25x)   <- the old threshold
#   n =  4900  17.1 ms vs  6.1 ms   (2.81x)
#   n = 10000  30.8 ms vs 14.6 ms   (2.11x)
#   n = 16900  42.5 ms vs 30.9 ms   (1.38x)
#
# The factorization wins everywhere measured, agreeing with CG to 4e-07 (CG's
# own iterative tolerance), and the margin was still positive at the largest
# size tried — so there is no crossover left to switch at.
_CG_THRESHOLD = None

# Safety factor on the Neumann convergence radius (see krylov_safe_radius).
# The series error behaves like r^(degree+1) for r = |Δρ|·ϱ(A_c⁻¹W); at the
# default degree 12, r = 0.6 gives ~1e-3 while r = 0.8 gives ~6e-2.
_KRYLOV_RADIUS_SAFETY = 0.6


def krylov_safe_radius(
    rho_c: float,
    W_eig_min: float = -1.0,
    W_eig_max: float = 1.0,
    dmax: float = _KRYLOV_DMAX_DEFAULT,
    safety: float = _KRYLOV_RADIUS_SAFETY,
) -> float:
    r"""Largest ``|Δρ|`` the Neumann series can be trusted over at ``ρ_c``.

    :math:`U(\rho_c+\Delta\rho)` is expanded as
    :math:`\sum_j \Delta\rho^j (A_c^{-1}W)^j A_c^{-1} X`, which converges only
    while :math:`|\Delta\rho|\,\varrho(A_c^{-1}W) < 1`.  The eigenvalues of
    :math:`A_c^{-1}W` are :math:`\lambda/(1-\rho_c\lambda)`, so the spectral
    radius is attained at one end of ``W``'s spectrum.

    A *fixed* ``dmax`` is therefore unsafe: for row-standardized ``W``
    (:math:`\lambda_{\max}=1`) the radius of convergence is
    :math:`1-\rho_c`, so the default ``dmax = 0.4`` already diverges once
    ``ρ_c > 0.6`` — squarely inside the range spatial models care about.
    Measured relative error of the degree-12 series on a ring lattice:

    =======  ========  ========  ========
    ρ_c      Δρ = 0.2  Δρ = 0.3  Δρ = 0.4
    =======  ========  ========  ========
    0.30     4.5e-07   7.9e-06   3.6e-04
    0.50     3.5e-06   7.3e-04   3.7e-02
    0.70     3.2e-03   diverges  diverges
    =======  ========  ========  ========

    Returns ``min(dmax, safety / ϱ)``, so callers keep their configured
    radius wherever it is genuinely safe and tighten only where it is not.
    """
    denom_max = 1.0 - rho_c * W_eig_max
    denom_min = 1.0 - rho_c * W_eig_min
    radius = 0.0
    for lam, denom in ((W_eig_max, denom_max), (W_eig_min, denom_min)):
        if denom > 0.0:
            radius = max(radius, abs(lam) / denom)
    if radius <= 0.0:
        return float(dmax)
    return float(min(dmax, safety / radius))


class ReducedKrylovBasis(NamedTuple):
    """Precomputed shift-invert Krylov basis for fast ρ-slice evaluation.

    At a center point :math:`\\rho_c`, we factorize
    :math:`A_c = I - \\rho_c W` once and build the basis

    .. math::

        V_0 = A_c^{-1} X, \\quad
        V_{j+1} = A_c^{-1} (W V_j), \\quad j = 0, \\dots, m-1.

    For any nearby :math:`\\rho = \\rho_c + \\Delta\\rho` the
    β-marginalized slice density only needs

    .. math::

        U(\\rho) \\approx \\sum_{j=0}^{m} (\\Delta\\rho)^j V_j,

    which is a cheap ``einsum`` instead of a fresh factorization.
    The approximation error decays geometrically in :math:`m` as
    :math:`O((\\Delta\\rho \\|A_c^{-1} W\\|)^{m+1})`.

    Attributes
    ----------
    rho_basis : float
        Center point :math:`\\rho_c` at which the system was factored.
    solver : _CholmodNormalEqSolver or spla.SuperLU or None
        The factored solver for :math:`A_c = I - \\rho_c W`.
        ``None`` when the CG path was used (no factorization).
    V_stack : ndarray, shape (m+1, n, k)
        Krylov basis vectors stacked along axis 0.
    degree : int
        Krylov degree :math:`m` (number of correction terms beyond
        :math:`V_0`).
    """

    rho_basis: float
    solver: _CholmodNormalEqSolver | spla.SuperLU | None
    V_stack: np.ndarray
    degree: int
    # Largest |Δρ| the series is trustworthy over at this center; see
    # krylov_safe_radius.  Consumers clamp their configured dmax to this.
    safe_dmax: float = _KRYLOV_DMAX_DEFAULT


def _build_krylov_basis(
    rho_c: float,
    X: np.ndarray,
    W_csc: sp.csc_matrix,
    n: int,
    degree: int = _KRYLOV_DEGREE_DEFAULT,
    cholmod_solver: _CholmodNormalEqSolver | None = None,
) -> ReducedKrylovBasis:
    """Build a shift-invert Krylov basis at :math:`\\rho_c`.

    One factorization plus ``(degree + 1)`` multi-RHS solves, against whatever
    :func:`make_sar_solver` routed to — Cholesky on the D-symmetrized
    ``I - rho*S`` for row-standardized undirected ``W``, KLU for directed
    ``W``.  ``_make_solver`` falls back to a plain sparse LU when no routed
    solver was supplied, so there is always a factorization available.

    There is no iterative path.  CG used to take over above n = 2500, back
    when the only direct option was CHOLMOD on the normal equations ``AtA``
    (two-hop fill from ``WtW``, squared condition number).  Against the routed
    factorization it loses everywhere measured — one degree-12 build on queen
    contiguity: 5.55x at n=1600, 4.25x at n=2500, 2.81x at n=4900, 2.11x at
    n=10000, 1.38x at n=16900, agreeing to 4e-07 — with the margin still
    positive at the largest size tried.
    """
    m = degree
    V_stack = np.empty((m + 1, n, X.shape[1]), dtype=np.float64)
    solver = _make_solver(rho_c, W_csc, n, cholmod_solver=cholmod_solver)
    V_stack[0] = solver.solve(X)  # (n, k)
    for j in range(m):
        Wv = W_csc @ V_stack[j]  # (n, k)
        V_stack[j + 1] = solver.solve(Wv)

    return ReducedKrylovBasis(
        rho_basis=rho_c,
        solver=solver,
        V_stack=V_stack,
        degree=m,
        # Radius from the coefficients themselves (root test), not from W's
        # spectrum -- so no eigenvalue bounds are needed anywhere.
        safe_dmax=_series_radius(V_stack),
    )


def _eval_U_from_basis(
    basis: ReducedKrylovBasis,
    drho: float,
) -> np.ndarray:
    """Evaluate :math:`U(\\rho_c + \\Delta\\rho) \\approx \\sum (\\Delta\\rho)^j V_j`.

    Uses Horner's method for numerical stability.
    """
    # Horner: V_0 + drho*(V_1 + drho*(V_2 + ... + drho*V_m))
    result = basis.V_stack[basis.degree].copy()
    for j in range(basis.degree - 1, -1, -1):
        result = basis.V_stack[j] + drho * result
    return result


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class ReducedGibbsState:
    """Mutable state for one chain of the reduced-form SAR-NB Gibbs sampler.

    Parameters
    ----------
    beta : ndarray, shape (k,)
        Regression coefficients.
    rho : float
        Spatial autoregressive parameter.
    alpha : float
        NB dispersion (NB2 parameterization; ``Var(y) = mu + mu^2 / alpha``).
    omega : ndarray, shape (n,)
        Pólya–Gamma auxiliary variables.
    """

    beta: np.ndarray
    rho: float
    alpha: float
    omega: np.ndarray


class ReducedGibbsCache(NamedTuple):
    """Constants reused across sweeps.

    Attributes
    ----------
    W_sparse : scipy.sparse.csr_matrix
        Row-standardized spatial weights (csr for fast matvec).
    W_csc : scipy.sparse.csc_matrix
        Same matrix in csc format (preferred for ``splu`` fallback).
    rho_lower, rho_upper : float
        Support bounds for the ρ slice sampler.
    rho_adaptive_width : bool
        Whether to tune the ρ slice-sampler width during warmup.
    rho_slice_width_state : SliceWidthState
        Mutable width state for the adaptive ρ slice sampler.
    krylov_degree : int
        Krylov basis degree :math:`m` for the shift-invert polynomial
        approximation of :math:`(I - \\rho W)^{-1} X`.  Default 8.
        Set to 0 to disable Krylov acceleration (use exact solve per
        candidate, as in the legacy path).
    krylov_dmax : float
        Maximum :math:`|\\Delta\\rho|` for which the Krylov basis is
        used.  When a slice candidate falls outside this radius around
        the basis center, a fresh factorization is performed for
        that single candidate.  Default 0.15.
    cholmod_pattern : csc_matrix or None
        When CHOLMOD is available, a sparse matrix with the sparsity
        pattern for ``A^T A = I − ρ(W+W^T) + ρ² W^T W``.  The
        ``CholmodFactor`` is created from this pattern **in the worker
        process**, avoiding CHOLMOD/BLAS calls in the parent that
        accumulate state and cause deadlocks after many ``fit()`` calls.
        ``None`` when CHOLMOD is not installed (falls back to ``splu``).
    W_sym : csc_matrix or None
        ``W + W^T`` in CSC format (precomputed for CHOLMOD path).
    WtW : csc_matrix or None
        ``W^T @ W`` in CSC format (precomputed for CHOLMOD path).
    W_eig_max : float
        Maximum absolute eigenvalue of W.  Used to compute eigenvalue
        bounds for the CG iterative solver:
        ``lam_min(A_rho) = 1 - rho * W_eig_max``.
        Default 1.0 (correct for row-standardized W).
    W_eig_min : float
        Minimum (real) eigenvalue of W.  Used to compute eigenvalue
        bounds for the CG iterative solver:
        ``lam_max(A_rho) = 1 - rho * W_eig_min``.
        Default -1.0 (correct for row-standardized W).
    n_rho_omega_cycles : int
        Number of (ω, ρ, β) Gibbs cycles per sweep.  At high ρ with
        large β₀, the ρ conditional mode shifts by ~2 posterior
        stds when ω is redrawn.  A single ω→ρ update leaves the
        chain lagging behind the mode, giving ESS ≈ 6.  Interleaving
        multiple ω→ρ→β cycles allows ρ to track the conditional
        mode, dramatically improving ESS.  Each cycle is a valid
        Gibbs update.  Default 1 (single cycle, original behavior).
        Set to 3–10 for data with high ρ and large β₀.
    krylov_reuse : bool
        When ``True`` (default), the Krylov basis built at the
        previous sweep's ρ is reused when |Δρ| <
        ``krylov_reuse_threshold``, skipping the CHOLMOD factorization
        + ``(degree + 1)`` triangular solves that account for 27–47%
        of per-sweep time.  Measured reuse rates are 95–100%
        post-warmup, giving 1.7–5.3× end-to-end speedup.  When
        ``False``, the basis is rebuilt every sweep (legacy
        behavior).
    krylov_reuse_threshold : float
        Maximum |Δρ| for which the previous sweep's Krylov basis is
        reused.  Must be ≤ ``krylov_dmax`` to stay within the
        polynomial's accuracy radius.  Default 0.15.
    """

    W_sparse: sp.csr_matrix
    W_csc: sp.csc_matrix
    rho_lower: float
    rho_upper: float
    rho_adaptive_width: bool = True
    rho_slice_width_state: Optional[SliceWidthState] = None
    krylov_degree: int = _KRYLOV_DEGREE_DEFAULT
    krylov_dmax: float = _KRYLOV_DMAX_DEFAULT
    cholmod_pattern: Optional[sp.csc_matrix] = None
    W_sym: Optional[sp.csc_matrix] = None
    WtW: Optional[sp.csc_matrix] = None
    W_eig_max: float = 1.0
    W_eig_min: float = -1.0
    n_rho_omega_cycles: int = 1
    krylov_reuse: bool = True
    krylov_reuse_threshold: float = 0.15


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


# ``_factor_A`` is defined above (alongside the CHOLMOD solver).


# ---------------------------------------------------------------------------
# Gibbs block samplers
# ---------------------------------------------------------------------------


def _sample_omega(
    y: np.ndarray,
    alpha: float,
    psi: np.ndarray,
    *,
    rng: np.random.Generator,
) -> np.ndarray:
    """Block 1: draw :math:`\\omega \\sim \\mathrm{PG}(y + \\alpha, \\psi)`.

    Parameters
    ----------
    y : ndarray, shape (n,)
        Integer responses.
    alpha : float
        NB dispersion parameter.
    psi : ndarray, shape (n,)
        Current tilting :math:`\\psi = \\eta - \\log\\alpha`.

    Notes
    -----
    The shape parameter ``h = y + alpha`` is clamped to ``1e-3`` to avoid
    the ``polyagamma`` ``"alternate"`` method's rejection of values
    below :math:`\\sim 10^{-3}`.  The hybrid sampler (``method=None``)
    automatically selects the saddle approximation for large ``h``,
    which is O(1) per draw regardless of ``h`` magnitude.
    """
    h = np.maximum(y + alpha, 1e-3)
    # Clamp psi to prevent the PG rejection sampler from
    # hanging on extreme |z| values.  For |z| > 20 the tilting is
    # saturated (tanh(z/2) ≈ 1), so clipping has negligible effect.
    psi_clamped = np.clip(psi, -20.0, 20.0)
    return sample_polyagamma(h, psi_clamped, rng=rng)


def _sample_beta(
    beta_current: np.ndarray,
    Xtilde: np.ndarray,
    omega: np.ndarray,
    y: np.ndarray,
    alpha: float,
    priors: ReducedGibbsPriors,
    *,
    rng: np.random.Generator,
    rho: float = 0.0,
    intercept_col: int = 0,
) -> np.ndarray:
    r"""Block 2: conjugate Gaussian draw for :math:`\beta`.

    Given :math:`\tilde X = A_\rho^{-1} X` and PG weights :math:`\omega`,
    the posterior is

    .. math::

        \beta \mid \cdot \sim N(m_\beta, \Sigma_\beta), \\
        \Sigma_\beta^{-1} = \tilde X^\top \Omega \tilde X + V_0^{-1}, \\
        m_\beta = \Sigma_\beta \bigl(\tilde X^\top (\kappa + \omega \log\alpha)
                                      + V_0^{-1} \mu_0\bigr),

    where :math:`\kappa = (y - \alpha)/2`.

    **Intercept reparameterization.**  For row-standardized :math:`W`,
    the intercept column of :math:`\tilde X` equals :math:`\mathbf{1}/(1-\rho)`,
    creating strong :math:`\rho`–:math:`\beta_0` posterior correlation at
    high :math:`\rho`.  We reparameterize :math:`\delta_0 = \beta_0/(1-\rho)`
    so that the intercept enters :math:`\eta` directly as
    :math:`\delta_0 \cdot \mathbf{1}`, breaking the correlation.  The draw
    is in :math:`\delta`-space; we transform back via
    :math:`\beta_0 = \delta_0 (1-\rho)` before returning.

    Parameters
    ----------
    beta_current : ndarray, shape (k,)
        Current draw (used only as a fallback if the linear solve fails).
    Xtilde : ndarray, shape (n, k)
        :math:`A_\rho^{-1} X` at the current :math:`\rho`.
    omega : ndarray, shape (n,)
        PG weights.
    y : ndarray, shape (n,)
        Integer responses.
    alpha : float
        NB dispersion.
    priors : ReducedGibbsPriors
    rng : numpy.random.Generator
    rho : float, default 0.0
        Current spatial autoregressive parameter (used for intercept
        reparameterization).
    intercept_col : int, default 0
        Column index of the intercept in :math:`X`.  Set to ``-1`` to
        disable the reparameterization.

    Returns
    -------
    beta_new : ndarray, shape (k,)
    """
    k = Xtilde.shape[1]
    kappa = 0.5 * (y - alpha)
    log_alpha = np.log(alpha)

    # Prior precision and mean
    beta_mu = priors.beta_mu
    beta_sigma = priors.beta_sigma
    if np.isscalar(beta_sigma):
        V0_inv_diag = np.full(k, 1.0 / (float(beta_sigma) ** 2))
    else:
        V0_inv_diag = 1.0 / (np.asarray(beta_sigma, dtype=np.float64) ** 2)
    if np.isscalar(beta_mu):
        mu0 = np.full(k, float(beta_mu))
    else:
        mu0 = np.asarray(beta_mu, dtype=np.float64)

    # --- Intercept reparameterization: δ₀ = β₀/(1−ρ) ---
    # Replace the intercept column of Xtilde (which is 1/(1−ρ) · 1)
    # with 1 · 1, so we sample δ₀ instead of β₀.
    # Prior on δ₀: N(μ₀/(1−ρ), σ₀²/(1−ρ)²) — precision scales by (1−ρ)².
    # For diffuse priors (σ₀ ≫ 1) this is negligible, but we apply it
    # correctly for correctness.
    reparam = intercept_col >= 0 and abs(rho) > 1e-8
    if reparam:
        scale = 1.0 - rho
        Xtilde_rp = Xtilde.copy()
        Xtilde_rp[:, intercept_col] = 1.0  # replace 1/(1−ρ)·1 with 1·1
        # Adjust prior for δ₀ = β₀/(1−ρ)
        V0_inv_diag_rp = V0_inv_diag.copy()
        V0_inv_diag_rp[intercept_col] = V0_inv_diag[intercept_col] * scale * scale
        mu0_rp = mu0.copy()
        mu0_rp[intercept_col] = mu0[intercept_col] / scale
    else:
        Xtilde_rp = Xtilde
        V0_inv_diag_rp = V0_inv_diag
        mu0_rp = mu0

    V0_inv_mu0 = V0_inv_diag_rp * mu0_rp

    # Σ_β^{-1} = X̃ᵀ Ω X̃ + V₀⁻¹
    # rhs = X̃ᵀ (κ + ω log α) + V₀⁻¹ μ₀
    Xt_omega = Xtilde_rp * omega[:, None]  # (n, k)
    Sigma_beta_inv = Xt_omega.T @ Xtilde_rp
    # Add prior precision on the diagonal
    Sigma_beta_inv.flat[:: k + 1] += V0_inv_diag_rp

    rhs = Xtilde_rp.T @ (kappa + omega * log_alpha) + V0_inv_mu0

    # Posterior draw via Cholesky:
    #   Σ_β⁻¹ = L Lᵀ
    #   m_β   = Σ_β rhs       (solve L Lᵀ m = rhs)
    #   sample = m_β + L⁻ᵀ z  with z ~ N(0, I), since Cov(L⁻ᵀz) = (L Lᵀ)⁻¹ = Σ_β.
    from scipy.linalg import solve_triangular

    try:
        L = np.linalg.cholesky(Sigma_beta_inv)
    except np.linalg.LinAlgError:
        # First attempt failed — add a small ridge and retry.
        Sigma_beta_inv.flat[:: k + 1] += 1e-10
        try:
            L = np.linalg.cholesky(Sigma_beta_inv)
        except np.linalg.LinAlgError:
            # Posterior precision is numerically singular (can happen when
            # omega collapses to near-zero for extreme alpha).  Reuse the
            # previous beta draw rather than crashing the chain.
            return beta_current

    w = solve_triangular(L, rhs, lower=True)
    m_beta = solve_triangular(L.T, w, lower=False)
    z = rng.standard_normal(k)
    delta = solve_triangular(L.T, z, lower=False)
    result = m_beta + delta

    # Transform δ₀ back to β₀ = δ₀ · (1−ρ)
    if reparam:
        result[intercept_col] *= scale

    return result


def _prior_precision_and_mean(
    priors: ReducedGibbsPriors, k: int
) -> tuple[np.ndarray, np.ndarray, float]:
    """Return ``(V0_inv_diag, mu0, log_det_V0)`` for the β prior.

    ``log_det_V0`` is currently unused by the ρ slice (constant in
    :math:`\\rho`) but returned for completeness.
    """
    beta_sigma = priors.beta_sigma
    if np.isscalar(beta_sigma):
        V0_inv_diag = np.full(k, 1.0 / (float(beta_sigma) ** 2))
        log_det_V0 = 2.0 * k * np.log(float(beta_sigma))
    else:
        sigma = np.asarray(beta_sigma, dtype=np.float64)
        V0_inv_diag = 1.0 / (sigma**2)
        log_det_V0 = 2.0 * float(np.sum(np.log(sigma)))
    beta_mu = priors.beta_mu
    if np.isscalar(beta_mu):
        mu0 = np.full(k, float(beta_mu))
    else:
        mu0 = np.asarray(beta_mu, dtype=np.float64)
    return V0_inv_diag, mu0, log_det_V0


def _rho_log_density_marginal(
    rho: float,
    omega: np.ndarray,
    y: np.ndarray,
    X: np.ndarray,
    W_csc: sp.csc_matrix,
    n: int,
    alpha: float,
    V0_inv_diag: np.ndarray,
    mu0: np.ndarray,
    rho_lower: float,
    rho_upper: float,
    *,
    basis: Optional[ReducedKrylovBasis] = None,
    krylov_dmax: float = _KRYLOV_DMAX_DEFAULT,
    cholmod_solver: Optional[_CholmodNormalEqSolver] = None,
    intercept_col: int = 0,
) -> float:
    r"""β-marginalized conditional log-density for the ρ slice.

    When a :class:`ReducedKrylovBasis` is provided and
    :math:`|\rho - \rho_c| \leq \texttt{krylov_dmax}`, the expensive
    solve is replaced by a cheap Horner evaluation of the
    shift-invert Krylov polynomial.  Otherwise a CG iterative
    solve is used (no factorization needed), with eigenvalue bounds
    derived from ``W_eig_max`` and ``W_eig_min``.

    **Intercept reparameterization.**  The intercept column of
    :math:`U = (I - \rho W)^{-1} X` is replaced with :math:`\mathbf{1}`,
    and the prior is adjusted for :math:`\delta_0 = \beta_0/(1-\rho)`.
    This breaks the :math:`\rho`–:math:`\beta_0` posterior correlation
    that causes ESS collapse at high :math:`\rho`.  Set
    ``intercept_col=-1`` to disable.
    """
    if rho <= rho_lower or rho >= rho_upper:
        return -np.inf

    # --- Compute U = (I - ρW)^{-1} X ---
    use_basis = (
        basis is not None
        and basis.degree > 0
        and abs(rho - basis.rho_basis) <= min(krylov_dmax, basis.safe_dmax)
    )
    if use_basis:
        drho = rho - basis.rho_basis
        U = _eval_U_from_basis(basis, drho)
    else:
        try:
            # Outside the Krylov radius: factor A_rho at this candidate.
            # This used to be a Chebyshev/CG solve parameterized by W's
            # spectral bounds; the routed factorization is faster and needs
            # no bounds, which is what let the eigendecomposition go.
            U = _make_solver(rho, W_csc, n, cholmod_solver=cholmod_solver).solve(X)
        except (RuntimeError, ValueError):
            return -np.inf

    # --- Intercept reparameterization: δ₀ = β₀/(1−ρ) ---
    reparam = intercept_col >= 0 and abs(rho) > 1e-8
    if reparam:
        scale = 1.0 - rho
        U = U.copy()
        U[:, intercept_col] = 1.0  # replace 1/(1−ρ)·1 with 1·1
        V0_inv_diag_rp = V0_inv_diag.copy()
        V0_inv_diag_rp[intercept_col] = V0_inv_diag[intercept_col] * scale * scale
        mu0_rp = mu0.copy()
        mu0_rp[intercept_col] = mu0[intercept_col] / scale
    else:
        V0_inv_diag_rp = V0_inv_diag
        mu0_rp = mu0

    # --- Working response and residual ---
    log_alpha = np.log(alpha)
    kappa = 0.5 * (y - alpha)
    s = kappa / omega + log_alpha
    r = s - U @ mu0_rp

    # M = V0^{-1} + U^T Ω U  (k x k)
    Uw = U * omega[:, None]
    M = U.T @ Uw
    k = M.shape[0]
    M.flat[:: k + 1] += V0_inv_diag_rp

    v = Uw.T @ r  # = U^T Ω r

    try:
        L = np.linalg.cholesky(M)
    except np.linalg.LinAlgError:
        M.flat[:: k + 1] += 1e-10
        try:
            L = np.linalg.cholesky(M)
        except np.linalg.LinAlgError:
            return -np.inf

    from scipy.linalg import solve_triangular

    w = solve_triangular(L, v, lower=True)
    quad_pen = float(w @ w)
    rOr = float(np.dot(r, omega * r))
    log_det_M = 2.0 * float(np.sum(np.log(np.diag(L))))

    result = -0.5 * log_det_M - 0.5 * (rOr - quad_pen)
    # Jacobian correction for the intercept reparameterization:
    # β₀ = δ₀·(1−ρ), so |∂β/∂δ| = (1−ρ) and log|det J| = log(1−ρ).
    if reparam:
        result += np.log(scale)
    # Guard against nan from numerical overflow in the Krylov path
    # or near-singular matrices.  Returning -inf causes the slice
    # sampler to reject the candidate and shrink the interval.
    if not np.isfinite(result):
        return -np.inf
    return result


def _sample_rho(
    state: ReducedGibbsState,
    cache: ReducedGibbsCache,
    y: np.ndarray,
    X: np.ndarray,
    priors: ReducedGibbsPriors,
    *,
    rng: np.random.Generator,
    sweep_idx: int,
    tune: int,
    basis: Optional[ReducedKrylovBasis] = None,
    cholmod_solver: Optional[_CholmodNormalEqSolver] = None,
    intercept_col: int = 0,
) -> tuple[float, float]:
    """Block 3: 1-D adaptive slice on :math:`\\rho` with β marginalized.

    When ``basis`` is provided (``krylov_degree > 0``), the slice density
    is evaluated via the shift-invert Krylov polynomial instead of a
    fresh factorization per candidate.  The basis is built once per sweep
    at the current ρ and reused for all candidates within ``krylov_dmax``.
    """
    n, k = X.shape
    rho_lower = cache.rho_lower
    rho_upper = cache.rho_upper
    V0_inv_diag, mu0, _ = _prior_precision_and_mean(priors, k)

    def log_density(rho: float) -> float:
        return _rho_log_density_marginal(
            rho=rho,
            omega=state.omega,
            y=y,
            X=X,
            W_csc=cache.W_csc,
            n=n,
            alpha=state.alpha,
            V0_inv_diag=V0_inv_diag,
            mu0=mu0,
            rho_lower=rho_lower,
            rho_upper=rho_upper,
            basis=basis,
            krylov_dmax=cache.krylov_dmax,
            cholmod_solver=cholmod_solver,
            intercept_col=intercept_col,
        )

    if cache.rho_adaptive_width and cache.rho_slice_width_state is not None:
        width_state = cache.rho_slice_width_state
        log_dens_x0 = log_density(state.rho)
        rho_new, log_density_new, steps_left, steps_right = slice_sample_1d_adaptive(
            log_density=log_density,
            x0=state.rho,
            lower=rho_lower,
            upper=rho_upper,
            width_state=width_state,
            rng=rng,
            log_density_x0=log_dens_x0,
        )
        if sweep_idx < tune:
            update_slice_width(width_state, steps_left, steps_right)
    else:
        rho_new, log_density_new = slice_sample_1d(
            log_density=log_density,
            x0=state.rho,
            lower=rho_lower,
            upper=rho_upper,
            w=0.2,
            rng=rng,
        )
    return rho_new, log_density_new


# ---------------------------------------------------------------------------
# Chain runner
# ---------------------------------------------------------------------------


def run_chain(
    y: np.ndarray,
    X: np.ndarray,
    W_sparse: sp.csr_matrix,
    priors: ReducedGibbsPriors,
    cache: ReducedGibbsCache,
    init: ReducedGibbsState,
    draws: int,
    tune: int,
    thin: int = 1,
    rng: np.random.Generator | None = None,
    chain_id: int = 0,
    progress_manager: object | None = None,
) -> dict[str, np.ndarray]:
    """Run one chain of the reduced-form SAR-NB PG-Gibbs sampler.

    Parameters
    ----------
    y : ndarray, shape (n,)
        Integer responses.
    X : ndarray, shape (n, k)
        Design matrix (intercept column expected if desired).
    W_sparse : scipy.sparse.csr_matrix, shape (n, n)
        Row-standardized spatial weights matrix.
    priors : ReducedGibbsPriors
        Prior hyperparameters.
    cache : ReducedGibbsCache
        Precomputed constants for the sweep (sparsity formats, slice
        width state, …).
    init : ReducedGibbsState
        Initial state.
    draws, tune : int
        Post-warmup draws and warmup sweeps respectively.
    thin : int, default 1
        Keep every ``thin``-th post-warmup draw.
    rng : numpy.random.Generator, optional
        Per-chain random state.
    chain_id : int, default 0
        Index used by ``progress_manager``.
    progress_manager : object, optional
        ``run_chains``-style progress callback.

    Returns
    -------
    dict[str, np.ndarray]
        Posterior samples with keys ``rho``, ``beta``, ``alpha``,
        ``log_lik`` (each indexed by post-warmup draw).
    """
    if rng is None:
        rng = np.random.default_rng()

    n, k = X.shape
    total_iters = tune + draws
    n_keep = draws // thin if thin > 0 else draws

    rho_samples = np.empty(n_keep, dtype=np.float64)
    beta_samples = np.empty((n_keep, k), dtype=np.float64)
    alpha_samples = np.empty(n_keep, dtype=np.float64)
    log_lik_samples = np.empty((n_keep, n), dtype=np.float64)

    state = ReducedGibbsState(
        beta=np.asarray(init.beta, dtype=np.float64).copy(),
        rho=float(init.rho),
        alpha=float(init.alpha),
        omega=np.asarray(init.omega, dtype=np.float64).copy(),
    )

    X = np.ascontiguousarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    from ..negbin._core import GibbsState as _StructuralState

    # Whether to use Krylov acceleration (degree > 0) or exact solve per
    # candidate (degree == 0, legacy path).
    use_krylov = cache.krylov_degree > 0
    krylov_degree = cache.krylov_degree

    # Detect intercept column for reparameterization: find the first
    # column of X that is all ones.  This breaks the ρ–β₀ correlation
    # that causes ESS collapse at high ρ.  Set to -1 if none found.
    intercept_col = -1
    for _j in range(k):
        if np.all(X[:, _j] == 1.0):
            intercept_col = _j
            break

    # Build the CHOLMOD normal-equations solver once per chain.
    # When CHOLMOD is available, this replaces all ``splu`` (scipy SuperLU)
    # calls with CHOLMOD on the SPD matrix A^T A, avoiding Apple
    # Accelerate BLAS deadlocks on macOS under concurrent access.
    # The CholmodFactor is created from the pattern matrix **here in
    # the worker**, not in the parent process, to avoid accumulating
    # CHOLMOD/BLAS state across many fit() calls.
    cholmod_solver: _CholmodNormalEqSolver | None = None
    if (
        cache.cholmod_pattern is not None
        and cache.W_sym is not None
        and cache.WtW is not None
    ):
        cholmod_factor = CholmodFactor(cache.cholmod_pattern)
        cholmod_solver = make_sar_solver(
            cholmod_factor=cholmod_factor,
            W_csc=cache.W_csc,
            W_sym=cache.W_sym,
            WtW=cache.WtW,
            n=n,
        )

    # Per-chain Krylov basis cache for reuse across sweeps.
    _prev_basis = None
    _prev_rho = None

    for i in range(total_iters):
        # --- Build Krylov basis at current ρ (or factorize for legacy) ---
        if use_krylov:
            # Basis reuse: skip the factorization + (degree+1) solves when
            # ρ hasn't moved far since the last rebuild.  The Krylov basis
            # at ρ_c is valid (to Horner accuracy) for any ρ within
            # krylov_dmax of ρ_c, so a |Δρ| < threshold reuse is exact
            # within the same tolerance the slice sampler already relies on.
            if (
                cache.krylov_reuse
                and _prev_basis is not None
                and abs(state.rho - _prev_rho) < cache.krylov_reuse_threshold
            ):
                basis = _prev_basis
            else:
                try:
                    basis = _build_krylov_basis(
                        state.rho,
                        X,
                        cache.W_csc,
                        n,
                        degree=krylov_degree,
                        cholmod_solver=cholmod_solver,
                    )
                except (RuntimeError, ValueError):
                    # CHOLMOD factorization failed (e.g. A^T A not SPD for
                    # extreme ρ).  Fall back to ρ = 0 (identity transform).
                    state.rho = 0.0
                    basis = _build_krylov_basis(
                        0.0,
                        X,
                        cache.W_csc,
                        n,
                        degree=krylov_degree,
                        cholmod_solver=cholmod_solver,
                    )
                _prev_basis = basis
                _prev_rho = state.rho

            # η for the ω block: η = U(ρ_c) @ β.  When the basis was reused,
            # ρ ≠ ρ_basis so evaluate U(ρ) via the Horner polynomial.
            if abs(state.rho - basis.rho_basis) < 1e-12:
                eta = basis.V_stack[0] @ state.beta
            else:
                _drho_eta = state.rho - basis.rho_basis
                eta = _eval_U_from_basis(basis, _drho_eta) @ state.beta
        else:
            try:
                solver = _make_solver(
                    state.rho, cache.W_csc, n, cholmod_solver=cholmod_solver
                )
            except (RuntimeError, ValueError):
                state.rho = 0.0
                solver = _make_solver(
                    0.0, cache.W_csc, n, cholmod_solver=cholmod_solver
                )
            eta = solver.solve(X @ state.beta)
            basis = None

        psi = eta - np.log(state.alpha)

        # --- Block 1+2: (ω, ρ, β) cycles ---
        # At high ρ with large β₀, the ρ conditional is extremely
        # peaked (std ≈ 0.001) and its mode shifts by ~2 stds when ω
        # is redrawn.  A single ω→ρ update leaves the chain lagging
        # behind the conditional mode, giving ESS ≈ 6.  Interleaving
        # multiple ω→ρ→β cycles per sweep breaks the ω–ρ dependence
        # and allows ρ to move further.  Each cycle is a valid
        # Gibbs update.
        #
        # Structure: draw ω before the loop (using the η computed
        # above), then for each cycle do (ρ, β, ω).  After the β
        # draw, η = X̃@β is a cheap matvec — no iterative solve
        # needed.  The last cycle's X̃ and η are reused for the
        # α draw and log-likelihood, avoiding a redundant solve.
        _n_cycles = cache.n_rho_omega_cycles
        state.omega = _sample_omega(y, state.alpha, psi, rng=rng)
        Xtilde = None  # will be set by the last cycle

        for _cycle in range(_n_cycles):
            # --- ρ | ω, α, y (β marginalized) ---
            state.rho, _ = _sample_rho(
                state=state,
                cache=cache,
                y=y,
                X=X,
                priors=priors,
                rng=rng,
                sweep_idx=i,
                tune=tune,
                basis=basis,
                cholmod_solver=cholmod_solver,
                intercept_col=intercept_col,
            )

            # --- β | ρ, ω, α, y ---
            # Compute X̃ = (I − ρW)⁻¹X at the new ρ.
            # Use Krylov eval when the basis is available and ρ is
            # within dmax — much cheaper than an iterative solve.
            _lam_at_max = 1.0 - state.rho * cache.W_eig_max
            _lam_at_min = 1.0 - state.rho * cache.W_eig_min
            _lam_min = min(_lam_at_max, _lam_at_min)
            _lam_max = max(_lam_at_max, _lam_at_min)
            if _lam_min <= 0:
                # ρ is too extreme — fall back to ρ = 0
                state.rho = 0.0
                _lam_min = 1.0
                _lam_max = 1.0
                Xtilde = X.copy()
            elif basis is not None:
                drho = state.rho - basis.rho_basis
                if abs(drho) <= min(cache.krylov_dmax, basis.safe_dmax):
                    Xtilde = _eval_U_from_basis(basis, drho)
                else:
                    Xtilde = _make_solver(
                        state.rho, cache.W_csc, n, cholmod_solver=cholmod_solver
                    ).solve(X)
            else:
                Xtilde = _make_solver(
                    state.rho, cache.W_csc, n, cholmod_solver=cholmod_solver
                ).solve(X)

            state.beta = _sample_beta(
                beta_current=state.beta,
                Xtilde=Xtilde,
                omega=state.omega,
                y=y,
                alpha=state.alpha,
                priors=priors,
                rng=rng,
                rho=state.rho,
                intercept_col=intercept_col,
            )

            # η = X̃@β — cheap matvec, no solve needed.
            eta = Xtilde @ state.beta

            # Draw ω for the next cycle (or for the α draw if
            # this is the last cycle).  Placed after β so that
            # the next ω uses the correct η.
            if _cycle < _n_cycles - 1:
                psi = eta - np.log(state.alpha)
                state.omega = _sample_omega(y, state.alpha, psi, rng=rng)

        # --- Block 3 is done: β was drawn in the last cycle ---
        # Xtilde and eta are from the last cycle's β draw.

        # --- Block 4: α | y, η_new ---
        alpha_state = _StructuralState(
            eta=eta,
            beta=state.beta,
            sigma2=1.0,  # unused by _sample_alpha
            rho=state.rho,
            alpha=state.alpha,
            omega=state.omega,
        )
        state.alpha = _sample_alpha(alpha_state, y, priors, rng=rng)

        # --- Store post-warmup draw ---
        if i >= tune and (i - tune) % thin == 0:
            idx = (i - tune) // thin
            if idx < n_keep:
                rho_samples[idx] = state.rho
                beta_samples[idx] = state.beta
                alpha_samples[idx] = state.alpha
                log_lik_samples[idx] = _nb_loglik_pointwise(y, eta, state.alpha)

        if progress_manager is not None:
            progress_manager.update(chain_id, i, tuning=i < tune)

    return {
        "rho": rho_samples,
        "beta": beta_samples,
        "alpha": alpha_samples,
        "log_lik": log_lik_samples,
    }
