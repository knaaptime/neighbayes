r"""Resolvent–Kronecker gradient of the unrestricted flow log-determinant.

For the unrestricted 3-parameter flow model the system matrix is

.. math::

    W_F = \rho_d\,(I_n \otimes W) + \rho_o\,(W \otimes I_n) + \rho_w\,(W \otimes W)

an :math:`N \times N` operator with :math:`N = n^2`.  Sampling the flow
:math:`\rho`'s with a gradient-based sampler (NUTS/MALA) needs

.. math::

    g_k \;=\; \frac{\partial}{\partial \rho_k}\,\log|I_N - W_F|
        \;=\; -\,\operatorname{tr}\!\big(W_k (I_N - W_F)^{-1}\big),
    \qquad W_k \in \{I\otimes W,\; W\otimes I,\; W\otimes W\}.

Unlike the log-determinant *value* — which the ``"traces"`` method estimates from
spectral moments and which is catastrophically noise-amplified at scale for large
directed ``W`` (the multinomial coefficients blow up Hutchinson error) — each
gradient component is a **single** resolvent trace.  Its Hutchinson relative error
scales like :math:`1/\sqrt{N P}` (``P`` probes), so it *improves* with the flow
sample size ``N``.  Everything here is **matvec-only** (no eigendecomposition,
directed-``W`` friendly): the Kronecker structure lets

.. math::

    W_F\,\operatorname{vec}(X)
      = \operatorname{vec}\!\big(\rho_d\,WX + \rho_o\,XW^\top + \rho_w\,WXW^\top\big)

be applied in :math:`O(n\cdot\mathrm{nnz})`, and the resolvent solve
:math:`(I_N - W_F)^{-\top} z` is done with GMRES on that operator.

This is the scalable, eigenvalue-free path for the unrestricted flow logdet; see
``resolvent_paper`` and the plan notes for the validation (relative error falling
as ``N`` grows).
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from neighbayes._jax_dispatch import ensure_x64

__all__ = [
    "FlowKron",
    "FlowKronJax",
    "flow_logdet_grad",
    "flow_logdet_grad_exact",
    "flow_logdet_value",
    "flow_logdet_value_and_grad",
    "flow_logdet_grad_jax",
    "flow_logdet_value_and_grad_jax",
]


def _as_csr(W) -> sp.csr_matrix:
    if sp.issparse(W):
        return W.tocsr().astype(np.float64)
    return sp.csr_matrix(np.asarray(W, dtype=np.float64))


class FlowKron:
    """Kronecker matvecs for the flow system built from an ``n x n`` weights ``W``.

    Holds ``W`` (CSR) and ``Wᵀ`` (CSR) once and applies the flow operators to a
    length-``N=n²`` vector by reshaping to ``n x n`` and using sparse–dense
    products.  All operators are :math:`O(n\\cdot\\mathrm{nnz}(W))`.
    """

    def __init__(self, W):
        self.W = _as_csr(W)
        self.Wt = self.W.T.tocsr()
        self.n = int(self.W.shape[0])
        self.N = self.n * self.n

    # -- W_F and its transpose --------------------------------------------
    # NumPy's ``reshape`` is row-major, for which ``(A ⊗ B) vec(X) = vec(A X Bᵀ)``.
    # Hence I⊗W ↦ X Wᵀ, W⊗I ↦ W X, W⊗W ↦ W X Wᵀ.
    def matvec_WF(self, x, rho_d, rho_o, rho_w):
        r"""``W_F x`` = vec(ρ_d X Wᵀ + ρ_o W X + ρ_w W X Wᵀ)."""
        X = x.reshape(self.n, self.n)
        WX = self.W @ X
        Y = rho_d * (X @ self.Wt) + rho_o * WX + rho_w * (WX @ self.Wt)
        return Y.ravel()

    def matvec_WF_T(self, x, rho_d, rho_o, rho_w):
        r"""``W_Fᵀ x`` = vec(ρ_d X W + ρ_o Wᵀ X + ρ_w Wᵀ X W)."""
        X = x.reshape(self.n, self.n)
        WtX = self.Wt @ X
        Y = rho_d * (X @ self.W) + rho_o * WtX + rho_w * (WtX @ self.W)
        return Y.ravel()

    # -- individual W_k (for the trace contractions) ----------------------
    def matvec_Wd(self, x):
        """``(I ⊗ W) x`` = vec(X Wᵀ)."""
        return (x.reshape(self.n, self.n) @ self.Wt).ravel()

    def matvec_Wo(self, x):
        """``(W ⊗ I) x`` = vec(W X)."""
        return (self.W @ x.reshape(self.n, self.n)).ravel()

    def matvec_Ww(self, x):
        """``(W ⊗ W) x`` = vec(W X Wᵀ)."""
        return (self.W @ (x.reshape(self.n, self.n) @ self.Wt)).ravel()

    def resolvent_T_operator(self, rho_d, rho_o, rho_w):
        """``LinearOperator`` for ``(I_N − W_Fᵀ)`` at the given ρ."""
        return spla.LinearOperator(
            (self.N, self.N),
            matvec=lambda x: x - self.matvec_WF_T(x, rho_d, rho_o, rho_w),
            dtype=np.float64,
        )

    def resolvent_T_sparse(self, rho_d, rho_o, rho_w):
        """Sparse CSC matrix for ``(I_N − W_Fᵀ)`` at the given ρ.

        ``W_Fᵀ = ρ_d(I⊗Wᵀ) + ρ_o(Wᵀ⊗I) + ρ_w(Wᵀ⊗Wᵀ)`` which, using
        ``kron(A, B)`` on CSR matrices, is a sparse N×N matrix with
        ``O(n²·nnz(W))`` nonzeros.  Used for KLU factorization when
        the dense N×N system is too large but the sparse pattern is
        manageable.
        """
        I_n = sp.eye(self.n, format="csr")
        WF_T = (
            rho_d * sp.kron(I_n, self.Wt, format="csr")
            + rho_o * sp.kron(self.Wt, I_n, format="csr")
            + rho_w * sp.kron(self.Wt, self.Wt, format="csr")
        )
        return (sp.eye(self.N, format="csr") - WF_T).tocsc()

    def _resolvent_T_pattern(self):
        """Cached COO sparsity pattern + value decomposition for ``(I_N − W_Fᵀ)``.

        The pattern of ``I_N − W_Fᵀ`` is **fixed** regardless of ρ — only the
        numerical values change.  This decomposes the matrix values into:

        * ``rows``, ``cols`` — COO indices (int32, for sparsax)
        * ``const_vals`` — values from the identity block (always 1.0)
        * ``coef_d``, ``coef_o``, ``coef_w`` — per-entry coefficients so that
          ``Ax = const_vals - ρ_d·coef_d - ρ_o·coef_o - ρ_w·coef_w``

        Computed once and cached so that sparsax's symbolic analysis
        (``analyze``) can be reused across all ρ evaluations.
        """
        if hasattr(self, "_pattern_cache"):
            return self._pattern_cache
        I_n = sp.eye(self.n, format="csr")
        Wd_kron = sp.kron(I_n, self.Wt, format="coo")  # I⊗Wᵀ
        Wo_kron = sp.kron(self.Wt, I_n, format="coo")  # Wᵀ⊗I
        Ww_kron = sp.kron(self.Wt, self.Wt, format="coo")  # Wᵀ⊗Wᵀ
        I_coo = sp.eye(self.N, format="coo")

        # Build combined COO with all entries, then sum duplicates
        all_rows = np.concatenate([I_coo.row, Wd_kron.row, Wo_kron.row, Ww_kron.row])
        all_cols = np.concatenate([I_coo.col, Wd_kron.col, Wo_kron.col, Ww_kron.col])
        shape = (self.N, self.N)

        def _sum_dups(data):
            m = sp.coo_matrix((data, (all_rows, all_cols)), shape=shape)
            m.sum_duplicates()
            return m

        const_mat = _sum_dups(
            np.concatenate(
                [
                    np.ones(I_coo.nnz),
                    np.zeros(Wd_kron.nnz),
                    np.zeros(Wo_kron.nnz),
                    np.zeros(Ww_kron.nnz),
                ]
            )
        )
        d_mat = _sum_dups(
            np.concatenate(
                [
                    np.zeros(I_coo.nnz),
                    Wd_kron.data,
                    np.zeros(Wo_kron.nnz),
                    np.zeros(Ww_kron.nnz),
                ]
            )
        )
        o_mat = _sum_dups(
            np.concatenate(
                [
                    np.zeros(I_coo.nnz),
                    np.zeros(Wd_kron.nnz),
                    Wo_kron.data,
                    np.zeros(Ww_kron.nnz),
                ]
            )
        )
        w_mat = _sum_dups(
            np.concatenate(
                [
                    np.zeros(I_coo.nnz),
                    np.zeros(Wd_kron.nnz),
                    np.zeros(Wo_kron.nnz),
                    Ww_kron.data,
                ]
            )
        )

        rows = np.asarray(const_mat.row, dtype=np.int32)
        cols = np.asarray(const_mat.col, dtype=np.int32)
        const_vals = np.asarray(const_mat.data, dtype=np.float64)
        coef_d = np.asarray(d_mat.data, dtype=np.float64)
        coef_o = np.asarray(o_mat.data, dtype=np.float64)
        coef_w = np.asarray(w_mat.data, dtype=np.float64)
        self._pattern_cache = (rows, cols, const_vals, coef_d, coef_o, coef_w)
        return self._pattern_cache


def flow_logdet_grad(
    W,
    rho_d: float,
    rho_o: float,
    rho_w: float,
    *,
    n_probes: int = 32,
    rng=None,
    probes: np.ndarray | None = None,
    tol: float = 1e-8,
    maxiter: int = 1000,
    return_probes: bool = False,
):
    r"""Stochastic estimate of ``∇_ρ log|I_N − W_F|`` via Kronecker Krylov solves.

    Returns ``g = (g_d, g_o, g_w)`` where
    ``g_k = −tr(W_k (I_N − W_F)^{-1})``.  For each Rademacher probe ``z`` it solves
    the single system ``x̃ = (I_N − W_Fᵀ)^{-1} z`` (GMRES; directed-``W`` safe) and
    accumulates ``x̃ᵀ (W_k z)`` for all three ``k`` — so one solve per probe yields
    the whole gradient.

    Parameters
    ----------
    W : array or sparse
        The ``n x n`` (row-standardized, possibly directed) weights matrix.
    rho_d, rho_o, rho_w : float
        Flow spatial parameters.
    n_probes : int
        Number of Hutchinson probes (ignored if ``probes`` is given).
    rng : numpy Generator, optional
        Randomness source for fresh probes.
    probes : ndarray (N, P), optional
        **Frozen** probe matrix reused across ρ (columns are ±1 vectors).  Giving
        the same ``probes`` at every call yields a smooth, consistent gradient
        field suitable for MALA / pseudo-marginal correction.
    tol, maxiter : float, int
        GMRES ``rtol`` and iteration cap.
    return_probes : bool
        Also return the probe matrix used (handy for freezing on the first call).

    Returns
    -------
    grad : ndarray (3,)
        ``(g_d, g_o, g_w)``.
    probes : ndarray (N, P), optional
        Only if ``return_probes``.
    """
    kron = W if isinstance(W, FlowKron) else FlowKron(W)
    N = kron.N

    if probes is None:
        rng = np.random.default_rng() if rng is None else rng
        probes = rng.choice([-1.0, 1.0], size=(N, int(n_probes))).astype(np.float64)
    else:
        probes = np.asarray(probes, dtype=np.float64)
        if probes.ndim == 1:
            probes = probes[:, None]
        if probes.shape[0] != N:
            raise ValueError(f"probes must have {N} rows; got {probes.shape[0]}.")

    op = kron.resolvent_T_operator(rho_d, rho_o, rho_w)
    P = probes.shape[1]
    acc = np.zeros(3, dtype=np.float64)

    # Solver priority for the numpy path:
    #   1. sksparse KLU (factorize once, solve P vectors sequentially)
    #   2. sparsax (batched solve, but requires JAX array conversion overhead)
    #   3. GMRES (iterative fallback)
    # The JAX-native path (_make_flow_kron_jax) uses sparsax directly without
    # numpy conversion; here we prefer sksparse for pure-numpy performance.
    from neighbayes._ops._backend import _select_sparse_backend, _sparse_factor

    backend = _select_sparse_backend()
    if backend == "klu":
        A_csc = kron.resolvent_T_sparse(rho_d, rho_o, rho_w)
        factor = _sparse_factor(A_csc, backend)
        for p in range(P):
            z = probes[:, p]
            xt = np.asarray(factor.solve(z), dtype=np.float64)
            acc[0] += xt @ kron.matvec_Wd(z)
            acc[1] += xt @ kron.matvec_Wo(z)
            acc[2] += xt @ kron.matvec_Ww(z)
    else:
        from neighbayes._jax_dispatch import _sparsax_available

        if _sparsax_available():
            import jax.numpy as jnp

            from neighbayes.samplers._utils._sparsax_lu import sparsax_lu

            ensure_x64()
            rows, cols, const_vals, coef_d, coef_o, coef_w = kron._resolvent_T_pattern()
            Ai = jnp.asarray(np.asarray(rows, dtype=np.int32))
            Aj = jnp.asarray(np.asarray(cols, dtype=np.int32))
            Ax = jnp.asarray(
                const_vals - rho_d * coef_d - rho_o * coef_o - rho_w * coef_w,
                dtype=jnp.float64,
            )
            lu_solve = sparsax_lu(Ai, Aj, Ax, kron.N).solve
            Xt = np.asarray(
                lu_solve(Ai, Aj, Ax, jnp.asarray(probes, dtype=jnp.float64)),
                dtype=np.float64,
            )
            for p in range(P):
                z = probes[:, p]
                xt = Xt[:, p]
                acc[0] += xt @ kron.matvec_Wd(z)
                acc[1] += xt @ kron.matvec_Wo(z)
                acc[2] += xt @ kron.matvec_Ww(z)
        else:
            for p in range(P):
                z = probes[:, p]
                xt, _info = spla.lgmres(op, z, rtol=tol, atol=0.0, maxiter=maxiter)
                acc[0] += xt @ kron.matvec_Wd(z)
                acc[1] += xt @ kron.matvec_Wo(z)
                acc[2] += xt @ kron.matvec_Ww(z)
    grad = -acc / P

    if return_probes:
        return grad, probes
    return grad


def flow_logdet_value(
    W,
    rho_d: float,
    rho_o: float,
    rho_w: float,
    *,
    n_probes: int = 32,
    n_quad: int = 8,
    rng=None,
    probes: np.ndarray | None = None,
    tol: float = 1e-8,
    maxiter: int = 1000,
):
    r"""Estimate ``log|I_N − W_F(ρ)|`` by integrating the gradient along ``0→ρ``.

    Since ``W_F(tρ) = t\,W_F(ρ)`` is linear in ``ρ``,

    .. math::

        \log|I_N - W_F(\rho)|
          = \int_0^1 \frac{d}{dt}\log|I_N - W_F(t\rho)|\,dt
          = \int_0^1 \sum_k \rho_k\,g_k(t\rho)\,dt ,

    a 1-D integral of the (scalable, resolvent-estimated) gradient.  With
    ``n_quad`` Gauss–Legendre nodes this is exact given exact gradients, and
    inherits the gradient's :math:`1/\sqrt{N P}` accuracy otherwise — so, unlike
    the moment-based ``"traces"`` value, it *improves* with the flow sample size.

    A single **frozen** probe matrix is shared across all quadrature nodes
    (generated once here if not supplied) so the value is a smooth, consistent
    function of ``ρ`` — suitable for a Metropolis correction.
    """
    kron = W if isinstance(W, FlowKron) else FlowKron(W)
    rho = np.array([rho_d, rho_o, rho_w], dtype=np.float64)

    if probes is None:
        rng = np.random.default_rng() if rng is None else rng
        probes = rng.choice([-1.0, 1.0], size=(kron.N, int(n_probes))).astype(
            np.float64
        )

    nodes, weights = np.polynomial.legendre.leggauss(int(n_quad))
    nodes = 0.5 * (nodes + 1.0)  # map [-1,1] -> [0,1]
    weights = 0.5 * weights

    val = 0.0
    for t, wq in zip(nodes, weights):
        g = flow_logdet_grad(
            kron,
            t * rho_d,
            t * rho_o,
            t * rho_w,
            probes=probes,
            tol=tol,
            maxiter=maxiter,
        )
        val += wq * float(rho @ g)
    return val


def flow_logdet_value_and_grad(
    W,
    rho_d: float,
    rho_o: float,
    rho_w: float,
    *,
    n_probes: int = 32,
    n_quad: int = 8,
    rng=None,
    probes: np.ndarray | None = None,
    tol: float = 1e-8,
    maxiter: int = 1000,
):
    """Return ``(value, grad)`` = ``(log|I_N−W_F|, ∇_ρ log|I_N−W_F|)``.

    The gradient is evaluated at ``ρ`` (the ``t=1`` endpoint) and the value by
    ray integration; both share the same frozen probes for consistency.
    """
    kron = W if isinstance(W, FlowKron) else FlowKron(W)
    if probes is None:
        rng = np.random.default_rng() if rng is None else rng
        probes = rng.choice([-1.0, 1.0], size=(kron.N, int(n_probes))).astype(
            np.float64
        )
    grad = flow_logdet_grad(
        kron, rho_d, rho_o, rho_w, probes=probes, tol=tol, maxiter=maxiter
    )
    value = flow_logdet_value(
        kron,
        rho_d,
        rho_o,
        rho_w,
        n_quad=n_quad,
        probes=probes,
        tol=tol,
        maxiter=maxiter,
    )
    return value, grad


def flow_logdet_grad_exact(W, rho_d: float, rho_o: float, rho_w: float) -> np.ndarray:
    r"""Exact ``∇_ρ log|I_N − W_F|`` from the spectrum of ``W`` (reference; small ``n``).

    Uses the shared Kronecker eigenbasis: ``μ_ij = ρ_o λ_i + ρ_d λ_j + ρ_w λ_i λ_j``
    and ``g_d = −Σ_ij λ_j/(1−μ_ij)`` etc.  Only for testing / small ``n`` — forms
    the ``n`` eigenvalues of ``W`` (``O(n³)``), never the ``N×N`` matrix.
    """
    Wd = _as_csr(W).toarray()
    lam = np.linalg.eigvals(Wd)
    li = lam[:, None]
    lj = lam[None, :]
    mu = rho_o * li + rho_d * lj + rho_w * (li * lj)
    inv = 1.0 / (1.0 - mu)
    g_d = -np.sum(lj * inv).real
    g_o = -np.sum(li * inv).real
    g_w = -np.sum(li * lj * inv).real
    return np.array([g_d, g_o, g_w], dtype=np.float64)


# ---------------------------------------------------------------------------
# JAX-native resolvent gradient (sparsax + equinox, fully jittable)
# ---------------------------------------------------------------------------

import jax.numpy as jnp  # noqa: E402 — lazy import for JAX convenience funcs


def _make_flow_kron_jax(kron: FlowKron, probes: np.ndarray, n_quad: int = 8):
    r"""Build a JAX-native, JIT-compilable flow logdet value+grad closure.

    Precomputes the fixed sparsax symbolic analysis and frozen probes as JAX
    arrays, then returns an ``eqx.filter_jit``-decorated function

    ``fn(rho_d, rho_o, rho_w) -> (value, grad3)``

    where the value is computed by ray integration (Gauss-Legendre quadrature
    of the gradient along ``0 → ρ``) and the gradient is the Hutchinson
    resolvent trace — all in JAX, with sparsax for the batched sparse solve.

    The Kronecker matvecs (``W_k @ x``) use sparse ``BCOO`` copies of ``W`` and
    ``Wᵀ`` on the ``n × n`` reshape (``O(n·nnz)`` per matvec) — ``W`` is never
    densified — while remaining fully jittable and vmappable.

    Parameters
    ----------
    kron : FlowKron
        The numpy-side Kronecker operator (provides ``W``, ``Wt``, ``n``,
        ``N``, and the COO pattern decomposition).
    probes : ndarray (N, P)
        Frozen Rademacher probe matrix.
    n_quad : int
        Number of Gauss-Legendre quadrature nodes for the ray integration.

    Returns
    -------
    callable
        ``(rho_d, rho_o, rho_w) -> (value: float, grad: jnp.ndarray(3,))``
        — JIT-compiled via ``eqx.filter_jit``.
    """
    import jax
    import jax.numpy as jnp

    ensure_x64()

    from neighbayes._jax_dispatch import _sparsax_available

    if not _sparsax_available():
        raise ImportError(
            "sparsax is required for the JAX-native flow resolvent path. "
            "Install with: pip install sparsax"
        )

    from neighbayes.samplers._utils._sparsax_lu import sparsax_lu

    n = kron.n

    # --- Precompute static data as JAX arrays ---
    # Sparse W and Wt as BCOO for Kronecker matvecs in JAX (O(n·nnz) per matvec)
    from jax.experimental.sparse import BCOO

    W_coo = kron.W.tocoo()
    Wt_coo = kron.Wt.tocoo()
    W_bcoo = BCOO.from_scipy_sparse(W_coo)
    Wt_bcoo = BCOO.from_scipy_sparse(Wt_coo)

    # Fixed sparsity pattern for all ρ; sparsax caches the fill-reducing
    # analysis internally keyed on (Ai, Aj), reused across solves.
    rows, cols, const_vals, coef_d, coef_o, coef_w = kron._resolvent_T_pattern()
    Ai = jnp.asarray(np.asarray(rows, dtype=np.int32))
    Aj = jnp.asarray(np.asarray(cols, dtype=np.int32))
    _const_vals = jnp.asarray(const_vals, dtype=jnp.float64)
    _coef_d = jnp.asarray(coef_d, dtype=jnp.float64)
    _coef_o = jnp.asarray(coef_o, dtype=jnp.float64)
    _coef_w = jnp.asarray(coef_w, dtype=jnp.float64)
    # KLU or UMFPACK, whichever is faster on this pattern, probed at ρ = 0.2 in
    # each direction (inside the stable region).
    lu_solve = sparsax_lu(
        Ai, Aj, _const_vals - 0.2 * (_coef_d + _coef_o + _coef_w), kron.N
    ).solve

    # Frozen probes as JAX array
    probes_jax = jnp.asarray(probes, dtype=jnp.float64)  # (N, P)

    # Gauss-Legendre quadrature nodes/weights on [0, 1]
    nodes_np, weights_np = np.polynomial.legendre.leggauss(n_quad)
    nodes_np = 0.5 * (nodes_np + 1.0)
    weights_np = 0.5 * weights_np
    quad_nodes = jnp.asarray(nodes_np, dtype=jnp.float64)
    quad_weights = jnp.asarray(weights_np, dtype=jnp.float64)

    # --- JAX sparse matvec helpers ---
    def _matvec_Wd(x):
        """(I⊗W) x = vec(X Wᵀ) where x = vec(X), X is (n, n) row-major."""
        X = x.reshape(n, n)
        return (X @ Wt_bcoo).ravel()

    def _matvec_Wo(x):
        """(W⊗I) x = vec(W X)."""
        X = x.reshape(n, n)
        return (W_bcoo @ X).ravel()

    def _matvec_Ww(x):
        """(W⊗W) x = vec(W X Wᵀ)."""
        X = x.reshape(n, n)
        return (W_bcoo @ (X @ Wt_bcoo)).ravel()

    # --- Core gradient at a single (rd, ro, rw) ---
    def _grad_at(rd, ro, rw):
        """Hutchinson resolvent gradient at (rd, ro, rw)."""
        Ax = _const_vals - rd * _coef_d - ro * _coef_o - rw * _coef_w
        # Batched solve: all P probes at once → (N, P)
        Xt = lu_solve(Ai, Aj, Ax, probes_jax)

        # Accumulate x̃ᵀ (W_k z) for each probe, each k
        # Using vmap over probes for the contractions
        def _single_probe_contrib(z, xt):
            return jnp.array(
                [
                    xt @ _matvec_Wd(z),
                    xt @ _matvec_Wo(z),
                    xt @ _matvec_Ww(z),
                ]
            )

        # (P, 3)
        contribs = jax.vmap(_single_probe_contrib)(probes_jax.T, Xt.T)
        grad = -jnp.mean(contribs, axis=0)
        return grad

    # --- Value via ray integration ---
    def _value_at(rd, ro, rw):
        """log|I - W_F(ρ)| via ray integration of the gradient."""
        rho = jnp.array([rd, ro, rw])

        def _integrand(t):
            g = _grad_at(t * rd, t * ro, t * rw)
            return jnp.dot(rho, g)

        # Gauss-Legendre quadrature
        vals = jax.vmap(_integrand)(quad_nodes)
        return jnp.dot(quad_weights, vals)

    # --- Combined value + grad ---
    def _value_and_grad(rd, ro, rw):
        grad = _grad_at(rd, ro, rw)
        value = _value_at(rd, ro, rw)
        return value, grad

    # JIT-compile with equinox (static data is filtered out)
    try:
        import equinox as eqx

        _value_and_grad_jitted = eqx.filter_jit(_value_and_grad)
    except ImportError:
        # Fallback: plain jax.jit (all static data is captured in closure)
        _value_and_grad_jitted = jax.jit(_value_and_grad)

    return _value_and_grad_jitted


class FlowKronJax:
    """JAX-native flow logdet value+grad with sparsax solves and JIT compilation.

    Wraps :func:`_make_flow_kron_jax` to provide a clean interface matching
    the numpy :class:`FlowKron` but returning JAX arrays.

    Parameters
    ----------
    W : array or sparse
        The ``n x n`` weights matrix.
    n_probes : int
        Number of frozen Rademacher probes.
    n_quad : int
        Number of Gauss-Legendre quadrature nodes for the value.
    seed : int
        Random seed for probe generation.
    """

    def __init__(self, W, n_probes: int = 48, n_quad: int = 8, seed: int = 0):
        self.kron = FlowKron(W)
        self.n = self.kron.n
        self.N = self.kron.N
        rng = np.random.default_rng(seed)
        probes = rng.choice([-1.0, 1.0], size=(self.N, int(n_probes))).astype(
            np.float64
        )
        self._fn = _make_flow_kron_jax(self.kron, probes, n_quad=n_quad)

    def value_and_grad(self, rho_d, rho_o, rho_w):
        """Return ``(value, grad)`` as JAX arrays."""
        import jax.numpy as jnp

        return self._fn(jnp.float64(rho_d), jnp.float64(rho_o), jnp.float64(rho_w))

    def value_and_grad_numpy(self, rho_d, rho_o, rho_w):
        """Return ``(value, grad)`` as numpy arrays (convenience)."""
        v, g = self.value_and_grad(rho_d, rho_o, rho_w)
        return float(v), np.asarray(g)


def flow_logdet_grad_jax(
    W,
    rho_d: float,
    rho_o: float,
    rho_w: float,
    *,
    probes: np.ndarray | None = None,
    n_probes: int = 48,
    seed: int = 0,
) -> np.ndarray:
    """JAX-native stochastic gradient of ``log|I_N − W_F(ρ)|`` (sparsax, jittable).

    Convenience wrapper around :class:`FlowKronJax` for one-off gradient
    evaluation.  For repeated calls (e.g. inside a sampler), construct a
    :class:`FlowKronJax` once and call ``value_and_grad_numpy`` to reuse the
    JIT-compiled function and cached symbolic analysis.

    Parameters
    ----------
    W : array or sparse
    rho_d, rho_o, rho_w : float
    probes : ndarray (N, P), optional
        Frozen probe matrix.  Generated if not provided.
    n_probes : int
    seed : int

    Returns
    -------
    np.ndarray, shape (3,)
        Gradient ``(g_d, g_o, g_w)``.
    """
    kron = FlowKron(W)
    if probes is None:
        rng = np.random.default_rng(seed)
        probes = rng.choice([-1.0, 1.0], size=(kron.N, int(n_probes))).astype(
            np.float64
        )
    fn = _make_flow_kron_jax(kron, probes, n_quad=2)
    _, grad = fn(jnp.float64(rho_d), jnp.float64(rho_o), jnp.float64(rho_w))
    return np.asarray(grad)


def flow_logdet_value_and_grad_jax(
    W,
    rho_d: float,
    rho_o: float,
    rho_w: float,
    *,
    probes: np.ndarray | None = None,
    n_probes: int = 48,
    n_quad: int = 8,
    seed: int = 0,
) -> tuple[float, np.ndarray]:
    """JAX-native ``log|I_N − W_F|`` and ``∇_ρ log|I_N − W_F|`` (sparsax, jittable).

    Convenience wrapper around :class:`FlowKronJax`.

    Returns
    -------
    value : float
    grad : np.ndarray, shape (3,)
    """
    est = FlowKronJax(W, n_probes=n_probes, n_quad=n_quad, seed=seed)
    if probes is not None:
        # Rebuild with provided probes
        est._fn = _make_flow_kron_jax(est.kron, probes, n_quad=n_quad)
    return est.value_and_grad_numpy(rho_d, rho_o, rho_w)
