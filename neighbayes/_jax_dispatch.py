"""JAX dispatch registrations for the custom Ops in :mod:`neighbayes.ops`.

This module enables JAX-backed NUTS samplers (``"blackjax"``, ``"numpyro"``)
for models that depend on :class:`~neighbayes.ops.SparseFlowSolveOp`,
:class:`~neighbayes.ops.SparseFlowSolveMatrixOp`,
:class:`~neighbayes.ops.SparseSARSolveOp`,
:class:`~neighbayes.ops.KroneckerFlowSolveOp`, and
:class:`~neighbayes.ops.KroneckerFlowSolveMatrixOp`.

The Kronecker Ops are translated into pure-JAX dense LU solves
(:math:`n\\times n`, jittable, vmappable).  The general sparse Ops are wrapped
in :func:`jax.pure_callback` because JAX has no CPU sparse direct solver;
their gradients are routed through the corresponding ``_*VJPOp`` whose JAX
dispatch is also a callback.  PyTensor inserts the VJP node into the symbolic
graph (via ``L_op``) *before* JAX transpilation, so JAX never has to
differentiate across the callback.

Availability is probed via :func:`importlib.util.find_spec`; the registration
function is a no-op when JAX or PyTensor's JAX dispatch module is missing,
so importing this module is always safe.
"""

from __future__ import annotations

import importlib.util
import os
import warnings
from functools import lru_cache


def ensure_x64() -> None:
    """Enable JAX float64 mode (idempotent).

    Every JAX entry point in the package requires ``jax_enable_x64``;
    call this instead of scattering ``jax.config.update`` at each site.
    """
    import jax

    jax.config.update("jax_enable_x64", True)


@lru_cache(maxsize=1)
def _eqx_available() -> bool:
    """Return ``True`` when optional ``equinox`` is importable."""
    return importlib.util.find_spec("equinox") is not None


@lru_cache(maxsize=1)
def _jax_available() -> bool:
    """Return ``True`` if JAX and PyTensor's JAX dispatch are importable."""
    return (
        importlib.util.find_spec("jax") is not None
        and importlib.util.find_spec("pytensor.link.jax.dispatch") is not None
    )


@lru_cache(maxsize=1)
def _sparsax_available() -> bool:
    """Return ``True`` when optional ``sparsax`` is importable.

    ``sparsax`` exposes CHOLMOD sparse SPD Cholesky as JIT-compatible
    JAX primitives (``solve``, ``logdet``, ``update_solve``) with custom
    VJP gradients and vmap batching.  It is CPU-only and requires
    ``jax_enable_x64=True``.

    A real import is attempted (not just ``find_spec``) so that broken
    installs — e.g. a stale editable install whose source tree has moved —
    count as unavailable instead of crashing at op-registration time.
    """
    if importlib.util.find_spec("sparsax") is None:
        return False
    try:
        importlib.import_module("sparsax")
    except Exception:
        return False
    return True


@lru_cache(maxsize=1)
def _warn_jax_auto_fallback_once(missing: str, target: str) -> None:
    """Emit a one-time advisory warning for JAX sparse backend auto-fallbacks."""
    install_hint = ""
    if missing == "sparsax":
        install_hint = " Install 'sparsax' to enable the JAX-native SuiteSparse (CHOLMOD/KLU) path."
    warnings.warn(
        "NEIGHBAYES_JAX_SPARSE_BACKEND=auto selected fallback backend "
        f"'{target}' because optional dependency '{missing}' is not installed. "
        f"Estimation is likely faster when the optional sparse backend is installed.{install_hint}",
        RuntimeWarning,
        stacklevel=3,
    )


@lru_cache(maxsize=1)
def _select_jax_sparse_backend() -> str:
    """Resolve JAX sparse backend from env vars with robust fallback.

    Environment
    -----------
    NEIGHBAYES_JAX_SPARSE_BACKEND : {"auto", "callback", "sparsax"}
        Default ``auto``. ``auto`` prefers ``sparsax`` (JAX-native SuiteSparse:
        CHOLMOD for SPD, KLU for asymmetric) when available, else ``callback``.
    NEIGHBAYES_JAX_SPARSE_STRICT : {"0", "1", "false", "true"}
        If truthy, missing requested optional backends raise ImportError.
    """
    requested = os.environ.get("NEIGHBAYES_JAX_SPARSE_BACKEND", "auto").strip().lower()
    strict = os.environ.get("NEIGHBAYES_JAX_SPARSE_STRICT", "0").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }

    if requested in {"", "auto"}:
        if _sparsax_available():
            return "sparsax"
        # JAX path fallback chain:
        #   1) sparsax (JAX-native SuiteSparse: CHOLMOD/KLU)
        #   2) callback + scipy SuperLU
        # The callback solver selection is handled in ops._select_sparse_backend.
        _warn_jax_auto_fallback_once("sparsax", "callback+scipy")
        return "callback"

    if requested in {"callback", "scipy", "pure_callback"}:
        return "callback"

    if requested == "sparsax":
        if _sparsax_available():
            return "sparsax"
        msg = (
            "NEIGHBAYES_JAX_SPARSE_BACKEND=sparsax requested, but optional "
            "dependency 'sparsax' is not installed. Falling back to callback backend."
        )
        if strict:
            raise ImportError(msg)
        warnings.warn(msg, RuntimeWarning)
        return "callback"

    msg = (
        f"Unknown NEIGHBAYES_JAX_SPARSE_BACKEND='{requested}'. "
        "Valid values are: auto, callback, sparsax. Falling back to auto."
    )
    if strict:
        raise ValueError(msg)
    warnings.warn(msg, RuntimeWarning)
    return "sparsax" if _sparsax_available() else "callback"


def _strict_env() -> bool:
    return os.environ.get("NEIGHBAYES_JAX_SPARSE_STRICT", "0").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


@lru_cache(maxsize=1)
def _sparsax_jax_enabled() -> bool:
    """Return ``True`` unless the sparsax JAX Gibbs path is opted *out*.

    Environment
    -----------
    NEIGHBAYES_JAX_SPARSAX : {"0", "1", "false", "true", ...}
        Default **enabled**.  When ``sparsax`` is installed (checked
        separately via :func:`_sparsax_available`), the JAX Gibbs samplers use
        its native SuiteSparse solvers (CHOLMOD for SPD, KLU for asymmetric)
        instead of the dense ``jnp.linalg.cholesky`` stopgap.  Set to a falsy
        value (``0``/``false``/``no``/``off``) to force the dense path — useful
        for benchmarking, debugging, or GPU experiments (sparsax is CPU-only).
    """
    return os.environ.get("NEIGHBAYES_JAX_SPARSAX", "1").strip().lower() not in {
        "0",
        "false",
        "no",
        "off",
    }


# Threshold above which the auto solver switches from eigen to the sparse
# (sparsax) path.  Eigen uses O(N^2) memory (two N×N complex128 matrices)
# and O(N^3) eigendecomposition time, which becomes prohibitive for very
# large N; the sparse path is O(nnz).
#
# Default is 0 (eigen path disabled). The eigen path materializes three
# N×N complex128 matrices (eigenvalues, eigenvectors, inverse eigenvectors)
# plus a dense N×N float64 W for the gradient — totalling ~24N² bytes.
# For n=2000 this is ~96 MB of GPU constants that XLA must trace through,
# causing multi-minute JIT compilation. The callback path (scipy sparse LU
# via host callback) avoids all of this and is faster for n > ~500.
#
# Set NEIGHBAYES_JAX_SAR_EIGEN_N_MAX to a positive value to re-enable
# the eigen path for small problems where it may be slightly faster per-step.
_JAX_SAR_EIGEN_N_MAX = int(os.environ.get("NEIGHBAYES_JAX_SAR_EIGEN_N_MAX", "0"))


def _resolve_auto_sar_solver(n: int) -> str:
    """Resolve ``"auto"`` to a concrete solver based on problem size *n*.

    Selection order:

    1. ``eigen`` when *n* is at or below ``NEIGHBAYES_JAX_SAR_EIGEN_N_MAX``
       (default 0, i.e. opt-in only). The eigen path materializes three
       N×N complex128 matrices plus a dense N×N float64 W and triggers
       multi-minute XLA compile times for n > ~500, so we keep it gated.
    2. ``sparsax`` when installed (default; disable via
       ``NEIGHBAYES_JAX_SPARSAX=0``).  Sparse SuiteSparse solve via JAX FFI —
       CHOLMOD Cholesky for D-symmetrizable W, KLU (asymmetric LU) for directed
       W — with its own VJP.  CPU-only.
    3. ``jax_gmres`` as the final fallback (matrix-free iterative solve).
    """
    if n <= _JAX_SAR_EIGEN_N_MAX:
        return "eigen"
    if _sparsax_jax_enabled() and _sparsax_available():
        return "sparsax"
    return "jax_gmres"


@lru_cache(maxsize=1)
def _select_jax_sar_solver() -> str:
    """Resolve the JAX SAR solver from env vars.

    Returns one of ``"auto"``, ``"eigen"``, ``"callback"``, ``"sparsax"``,
    ``"jax_gmres"``.

    ``"auto"`` is resolved to a concrete solver at Op registration time
    by :func:`_resolve_auto_sar_solver` based on the problem size *n*.

    Environment
    -----------
    NEIGHBAYES_JAX_SAR_SOLVER : {"auto", "eigen", "callback", "sparsax", "jax_gmres"}
        Default ``auto``. ``auto`` selects ``eigen`` when
        N ≤ ``NEIGHBAYES_JAX_SAR_EIGEN_N_MAX`` (default 0, i.e. opt-in),
        otherwise ``sparsax`` when installed (default; disable via
        ``NEIGHBAYES_JAX_SPARSAX=0``), else ``jax_gmres``.
    NEIGHBAYES_JAX_SAR_EIGEN_N_MAX : int, default 0
        Maximum N for which ``auto`` selects the eigen path. Default
        0 disables eigen in ``auto`` because the dense materialization
        triggers multi-minute XLA compile times for n > ~500.
    NEIGHBAYES_JAX_SPARSE_STRICT : truthy
        If set, missing requested optional dependencies raise ImportError
        instead of falling back.
    """
    requested = os.environ.get("NEIGHBAYES_JAX_SAR_SOLVER", "auto").strip().lower()
    strict = _strict_env()

    if requested in {"", "auto"}:
        return "auto"

    if requested == "eigen":
        return "eigen"

    if requested in {"callback", "scipy", "pure_callback"}:
        return "callback"

    if requested == "sparsax":
        if _sparsax_available():
            return "sparsax"
        msg = (
            "NEIGHBAYES_JAX_SAR_SOLVER=sparsax requested, but optional "
            "dependency 'sparsax' is not installed. Falling back to callback."
        )
        if strict:
            raise ImportError(msg)
        warnings.warn(msg, RuntimeWarning)
        return "callback"

    if requested in {"jax_gmres", "gmres", "jaxgmres"}:
        return "jax_gmres"

    msg = (
        f"Unknown NEIGHBAYES_JAX_SAR_SOLVER='{requested}'. "
        "Valid values are: auto, eigen, callback, sparsax, jax_gmres. "
        "Falling back to auto."
    )
    if strict:
        raise ValueError(msg)
    warnings.warn(msg, RuntimeWarning)
    return "auto"


@lru_cache(maxsize=1)
def register_jax_dispatch() -> bool:
    """Register JAX dispatches for all Ops in :mod:`neighbayes.ops`.

    Idempotent (cached). Returns ``True`` if registration ran, ``False`` if
    JAX is not available.
    """
    if not _jax_available():
        return False

    import jax
    import jax.numpy as jnp
    import jax.scipy.linalg as jsla
    import jax.scipy.sparse.linalg as jssl
    import numpy as np
    import scipy.sparse as sp
    from pytensor.link.jax.dispatch import jax_funcify

    sar_solver = _select_jax_sar_solver()

    from ._ops import (
        KroneckerFlowSolveMatrixOp,
        KroneckerFlowSolveOp,
        SparseFlowSolveMatrixOp,
        SparseFlowSolveOp,
        SparseSARSolveOp,
        _KroneckerFlowVJPMatrixOp,
        _KroneckerFlowVJPOp,
        _SparseFlowVJPMatrixOp,
        _SparseFlowVJPOp,
        _SparseSARVJPOp,
    )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _dense(W):
        """scipy CSR -> dense float64 numpy array (closed over once)."""
        return np.asarray(W.toarray(), dtype=np.float64)

    def _reshape_F(arr, shape):
        """Equivalent to ``arr.reshape(shape, order='F')`` for 2D targets.

        For a 1D ``arr`` of length ``n*n`` reshaped to ``(n, n)`` Fortran-order,
        this is ``arr.reshape((n, n)).T``.
        """
        # Only used here for 1D -> (n, n)
        n = shape[0]
        return arr.reshape((n, n)).T

    def _ravel_F_2d(M):
        """Equivalent to ``M.ravel(order='F')`` for a 2D array."""
        return M.T.reshape(-1)

    def _kron_sparsax_ctx(op):
        """sparsax LU context for the separable-Kronecker regional solves.

        ``Ld = I − ρ_d W`` and ``Lo = I − ρ_o W`` are sparse (``W`` is sparse),
        so we solve them with sparsax's asymmetric sparse LU (KLU or UMFPACK,
        whichever :func:`~neighbayes.samplers._utils._sparsax_lu.sparsax_lu`
        measures faster) instead of forming a dense ``n×n`` and calling
        ``jsla.solve``.  Both share the ``I ∪ W`` pattern, whose fill-reducing
        analysis sparsax computes once and caches (content-addressed), reusing
        it across forward (``solve`` → ``lu_solve(Ai, Aj, ·)``) and adjoint
        (``tsolve`` → ``lu_solve(Aj, Ai, ·)``, the transpose via swapped COO
        indices) right-hand sides.  ``W`` is never densified.
        """
        from jax.experimental import sparse as jsparse

        from .samplers._utils._sparsax_lu import sparsax_lu

        n = op._n
        eye = sp.eye(n, format="coo", dtype=np.float64)
        Wc = op._W.tocoo()
        _rows = np.concatenate([eye.row, Wc.row])
        _cols = np.concatenate([eye.col, Wc.col])

        def _slot(parts):
            c = sp.coo_matrix((np.concatenate(parts), (_rows, _cols)), shape=(n, n))
            c.sum_duplicates()
            return c

        eye_c = _slot([np.ones(eye.nnz), np.zeros(Wc.nnz)])
        w_c = _slot([np.zeros(eye.nnz), Wc.data])
        Ai = jnp.asarray(np.asarray(eye_c.row, dtype=np.int32))
        Aj = jnp.asarray(np.asarray(eye_c.col, dtype=np.int32))
        eye_vals = jnp.asarray(eye_c.data, dtype=jnp.float64)
        w_vals = jnp.asarray(w_c.data, dtype=jnp.float64)
        W_bcoo = jsparse.BCOO.from_scipy_sparse(op._W.tocsr())

        lu_solve = sparsax_lu(Ai, Aj, eye_vals - 0.5 * w_vals, n).solve

        def solve(Ax, rhs):  # L x = rhs
            return lu_solve(Ai, Aj, Ax, rhs)

        def tsolve(Ax, rhs):  # Lᵀ x = rhs (transpose via swapped COO indices)
            return lu_solve(Aj, Ai, Ax, rhs)

        return n, eye_vals, w_vals, solve, tsolve, W_bcoo

    # ------------------------------------------------------------------
    # Kronecker forward — pure JAX
    # ------------------------------------------------------------------

    @jax_funcify.register(KroneckerFlowSolveOp)
    def _funcify_kron_solve(op, **kwargs):
        if _sparsax_available():
            n, eye_vals, w_vals, solve, _tsolve, _W = _kron_sparsax_ctx(op)

            def kron_solve(rho_d, rho_o, b):
                Hb = _reshape_F(b, (n, n))  # (n, n)
                Hp = solve(eye_vals - rho_d * w_vals, Hb)  # Ld Hp = Hb
                Z = solve(eye_vals - rho_o * w_vals, Hp.T)  # Lo Z = Hp^T
                return Z.reshape(-1)

            return kron_solve

        # Dense fallback (sparsax not installed).
        W_d = jnp.asarray(_dense(op._W))
        n = op._n
        I = jnp.eye(n, dtype=jnp.float64)

        def kron_solve(rho_d, rho_o, b):
            Ld = I - rho_d * W_d
            Lo = I - rho_o * W_d
            Hb = _reshape_F(b, (n, n))  # (n, n)
            Hp = jsla.solve(Ld, Hb)  # Ld H' = Hb
            Z = jsla.solve(Lo, Hp.T)  # Lo Z = Hp^T
            # perform: Z.T.ravel(order='F') == Z.ravel()
            return Z.reshape(-1)

        return kron_solve

    # ------------------------------------------------------------------
    # Kronecker VJP — pure JAX
    # ------------------------------------------------------------------

    @jax_funcify.register(_KroneckerFlowVJPOp)
    def _funcify_kron_vjp(op, **kwargs):
        if _sparsax_available():
            n, eye_vals, w_vals, _solve, tsolve, W_bcoo = _kron_sparsax_ctx(op)

            def kron_vjp(rho_d, rho_o, eta, g):
                H_eta = _reshape_F(eta, (n, n))  # (n, n)
                Hg = _reshape_F(g, (n, n))  # (n, n)

                # Adjoint: (Lo^T ⊗ Ld^T) v = g  =>  Ld^T H_v Lo = Hg
                P = tsolve(eye_vals - rho_d * w_vals, Hg)  # Ld^T P = Hg
                Q = tsolve(eye_vals - rho_o * w_vals, P.T)  # Lo^T Q = P^T (Q=H_v^T)
                H_v = Q.T  # (n, n)

                W_H = W_bcoo @ H_eta  # W @ H_eta
                Ld_H = H_eta - rho_d * W_H  # Ld @ H_eta = H_eta - ρ_d W H_eta
                # W_H @ W^T = (W @ W_H^T)^T ; Ld_H @ W^T = (W @ Ld_H^T)^T
                WH_Wt = (W_bcoo @ W_H.T).T  # W_H @ W^T
                LdH_Wt = (W_bcoo @ Ld_H.T).T  # Ld_H @ W^T
                # W_H @ Lo^T = W_H - ρ_o (W_H @ W^T)
                grad_rd = jnp.sum(H_v * (W_H - rho_o * WH_Wt))
                grad_ro = jnp.sum(H_v * LdH_Wt)  # Ld_H @ W_d^T
                grad_b = _ravel_F_2d(H_v)
                return grad_rd, grad_ro, grad_b

            return kron_vjp

        # Dense fallback (sparsax not installed).
        W_d = jnp.asarray(_dense(op._W))
        n = op._n
        I = jnp.eye(n, dtype=jnp.float64)

        def kron_vjp(rho_d, rho_o, eta, g):
            Ld = I - rho_d * W_d
            Lo = I - rho_o * W_d

            H_eta = _reshape_F(eta, (n, n))  # (n, n)
            Hg = _reshape_F(g, (n, n))  # (n, n)

            # Adjoint: (Lo^T ⊗ Ld^T) v = g  =>  Ld^T H_v Lo = Hg
            P = jsla.solve(Ld.T, Hg)  # Ld^T P = Hg
            Q = jsla.solve(Lo.T, P.T)  # Lo^T Q = P^T  (Q = H_v^T)
            H_v = Q.T  # (n, n)

            W_H = W_d @ H_eta  # (n, n)
            Ld_H = Ld @ H_eta  # (n, n)
            grad_rd = jnp.sum(H_v * (W_H @ Lo.T))
            grad_ro = jnp.sum(H_v * (Ld_H @ W_d.T))
            grad_b = _ravel_F_2d(H_v)
            return grad_rd, grad_ro, grad_b

        return kron_vjp

    # ------------------------------------------------------------------
    # Kronecker matrix forward / VJP — vmap over the single-vector path
    # ------------------------------------------------------------------

    @jax_funcify.register(KroneckerFlowSolveMatrixOp)
    def _funcify_kron_solve_matrix(op, **kwargs):
        if _sparsax_available():
            n, eye_vals, w_vals, solve, _tsolve, _W = _kron_sparsax_ctx(op)

            def kron_solve_mat(rho_d, rho_o, B):
                # ρ is shared across columns; sparsax caches the factor for
                # each distinct Ax, so the per-column vmap reuses it.
                Ax_d = eye_vals - rho_d * w_vals
                Ax_o = eye_vals - rho_o * w_vals

                def _one(b):
                    Hb = _reshape_F(b, (n, n))
                    Hp = solve(Ax_d, Hb)
                    Z = solve(Ax_o, Hp.T)
                    return Z.reshape(-1)

                return jax.vmap(_one, in_axes=1, out_axes=1)(B)

            return kron_solve_mat

        # Dense fallback (sparsax not installed).
        W_d = jnp.asarray(_dense(op._W))
        n = op._n
        I = jnp.eye(n, dtype=jnp.float64)

        def _solve_one(rho_d, rho_o, b):
            Ld = I - rho_d * W_d
            Lo = I - rho_o * W_d
            Hb = _reshape_F(b, (n, n))
            Hp = jsla.solve(Ld, Hb)
            Z = jsla.solve(Lo, Hp.T)
            return Z.reshape(-1)

        def kron_solve_mat(rho_d, rho_o, B):
            # vmap over the column (T) axis: B is (N, T) -> output (T, N) -> (N, T)
            solver = jax.vmap(_solve_one, in_axes=(None, None, 1), out_axes=1)
            return solver(rho_d, rho_o, B)

        return kron_solve_mat

    @jax_funcify.register(_KroneckerFlowVJPMatrixOp)
    def _funcify_kron_vjp_matrix(op, **kwargs):
        if _sparsax_available():
            n, eye_vals, w_vals, _solve, tsolve, W_bcoo = _kron_sparsax_ctx(op)

            def kron_vjp_mat(rho_d, rho_o, H_eta, G):
                Ax_d = eye_vals - rho_d * w_vals
                Ax_o = eye_vals - rho_o * w_vals

                def _one(eta_col, g_col):
                    H_e = _reshape_F(eta_col, (n, n))
                    Hg = _reshape_F(g_col, (n, n))
                    P = tsolve(Ax_d, Hg)
                    Q = tsolve(Ax_o, P.T)
                    H_v = Q.T
                    W_H = W_bcoo @ H_e
                    Ld_H = H_e - rho_d * W_H
                    WH_Wt = (W_bcoo @ W_H.T).T  # W_H @ W^T
                    LdH_Wt = (W_bcoo @ Ld_H.T).T  # Ld_H @ W^T
                    grad_rd = jnp.sum(H_v * (W_H - rho_o * WH_Wt))
                    grad_ro = jnp.sum(H_v * LdH_Wt)
                    return grad_rd, grad_ro, _ravel_F_2d(H_v)

                vjper = jax.vmap(_one, in_axes=(1, 1), out_axes=(0, 0, 1))
                grad_rd_per_t, grad_ro_per_t, grad_B = vjper(H_eta, G)
                return jnp.sum(grad_rd_per_t), jnp.sum(grad_ro_per_t), grad_B

            return kron_vjp_mat

        # Dense fallback (sparsax not installed).
        W_d = jnp.asarray(_dense(op._W))
        n = op._n
        I = jnp.eye(n, dtype=jnp.float64)

        def _vjp_one(rho_d, rho_o, eta_col, g_col):
            Ld = I - rho_d * W_d
            Lo = I - rho_o * W_d
            H_eta = _reshape_F(eta_col, (n, n))
            Hg = _reshape_F(g_col, (n, n))
            P = jsla.solve(Ld.T, Hg)
            Q = jsla.solve(Lo.T, P.T)
            H_v = Q.T
            W_H = W_d @ H_eta
            Ld_H = Ld @ H_eta
            grad_rd = jnp.sum(H_v * (W_H @ Lo.T))
            grad_ro = jnp.sum(H_v * (Ld_H @ W_d.T))
            grad_b = _ravel_F_2d(H_v)
            return grad_rd, grad_ro, grad_b

        def kron_vjp_mat(rho_d, rho_o, H_eta, G):
            # vmap over column axis; sum scalar grads, stack vector grad
            vjper = jax.vmap(_vjp_one, in_axes=(None, None, 1, 1), out_axes=(0, 0, 1))
            grad_rd_per_t, grad_ro_per_t, grad_B = vjper(rho_d, rho_o, H_eta, G)
            return jnp.sum(grad_rd_per_t), jnp.sum(grad_ro_per_t), grad_B

        return kron_vjp_mat

    # ------------------------------------------------------------------
    # Sparse Ops — wrap scipy splu via jax.pure_callback
    # ------------------------------------------------------------------
    #
    # JAX has no CPU sparse direct solver. We use a host callback that runs
    # the existing perform() logic. Two distinct gradient paths must work:
    #
    # 1. PyTensor's symbolic L_op path inserts the VJP node into the graph
    #    BEFORE JAX transpilation.  Each VJP node has its own callback
    #    dispatch, so JAX never differentiates across the callback here.
    # 2. PyMC's JAX samplers (blackjax, numpyro) compile only the forward
    #    log-density and then call ``jax.grad`` on it.  ``jax.grad`` traces
    #    through ``pure_callback`` and raises ``Pure callbacks do not
    #    support JVP``.  To make this path work we wrap the forward solve
    #    in ``jax.custom_vjp`` — the bwd rule calls the existing analytic
    #    adjoint via another ``pure_callback``.

    def _make_solve_with_custom_vjp(forward_op, vjp_op, *, matrix: bool):
        """Build a ``custom_vjp``-decorated solver that reuses the Op callbacks."""

        def _host_solve(rd, ro, rw, rhs):
            outputs = [[None]]
            forward_op.perform(
                None,
                [np.asarray(rd), np.asarray(ro), np.asarray(rw), np.asarray(rhs)],
                outputs,
            )
            return outputs[0][0]

        def _host_vjp(rd, ro, rw, sol, g):
            outputs = [[None], [None], [None], [None]]
            vjp_op.perform(
                None,
                [
                    np.asarray(rd),
                    np.asarray(ro),
                    np.asarray(rw),
                    np.asarray(sol),
                    np.asarray(g),
                ],
                outputs,
            )
            return (outputs[0][0], outputs[1][0], outputs[2][0], outputs[3][0])

        @jax.custom_vjp
        def solve(rho_d, rho_o, rho_w, rhs):
            return jax.pure_callback(
                _host_solve,
                jax.ShapeDtypeStruct(rhs.shape, jnp.float64),
                rho_d,
                rho_o,
                rho_w,
                rhs,
                vmap_method="sequential",
            )

        def solve_fwd(rho_d, rho_o, rho_w, rhs):
            sol = solve(rho_d, rho_o, rho_w, rhs)
            return sol, (rho_d, rho_o, rho_w, sol)

        def solve_bwd(residuals, g):
            rho_d, rho_o, rho_w, sol = residuals
            scalar = jax.ShapeDtypeStruct((), jnp.float64)
            shapes = (
                scalar,
                scalar,
                scalar,
                jax.ShapeDtypeStruct(sol.shape, jnp.float64),
            )
            grad_rd, grad_ro, grad_rw, grad_rhs = jax.pure_callback(
                _host_vjp,
                shapes,
                rho_d,
                rho_o,
                rho_w,
                sol,
                g,
                vmap_method="sequential",
            )
            return grad_rd, grad_ro, grad_rw, grad_rhs

        solve.defvjp(solve_fwd, solve_bwd)
        return solve, _host_vjp

    @jax_funcify.register(SparseFlowSolveOp)
    def _funcify_sparse_solve(op, **kwargs):
        solve, _ = _make_solve_with_custom_vjp(op, op._vjp_op, matrix=False)

        def sparse_solve(rho_d, rho_o, rho_w, b):
            return solve(rho_d, rho_o, rho_w, b)

        return sparse_solve

    @jax_funcify.register(_SparseFlowVJPOp)
    def _funcify_sparse_vjp(op, **kwargs):
        # Used by PyTensor's symbolic L_op path. Pure callback is fine
        # here because PyTensor never differentiates through this node
        # (it IS the gradient).
        def _host_vjp(rd, ro, rw, eta, g):
            outputs = [[None], [None], [None], [None]]
            op.perform(
                None,
                [
                    np.asarray(rd),
                    np.asarray(ro),
                    np.asarray(rw),
                    np.asarray(eta),
                    np.asarray(g),
                ],
                outputs,
            )
            return (outputs[0][0], outputs[1][0], outputs[2][0], outputs[3][0])

        def sparse_vjp(rho_d, rho_o, rho_w, eta, g):
            scalar = jax.ShapeDtypeStruct((), jnp.float64)
            shapes = (
                scalar,
                scalar,
                scalar,
                jax.ShapeDtypeStruct(eta.shape, jnp.float64),
            )
            return jax.pure_callback(
                _host_vjp,
                shapes,
                rho_d,
                rho_o,
                rho_w,
                eta,
                g,
                vmap_method="sequential",
            )

        return sparse_vjp

    @jax_funcify.register(SparseFlowSolveMatrixOp)
    def _funcify_sparse_solve_matrix(op, **kwargs):
        solve, _ = _make_solve_with_custom_vjp(op, op._vjp_op, matrix=True)

        def sparse_solve_mat(rho_d, rho_o, rho_w, B):
            return solve(rho_d, rho_o, rho_w, B)

        return sparse_solve_mat

    @jax_funcify.register(_SparseFlowVJPMatrixOp)
    def _funcify_sparse_vjp_matrix(op, **kwargs):
        def _host_vjp(rd, ro, rw, H, G):
            outputs = [[None], [None], [None], [None]]
            op.perform(
                None,
                [
                    np.asarray(rd),
                    np.asarray(ro),
                    np.asarray(rw),
                    np.asarray(H),
                    np.asarray(G),
                ],
                outputs,
            )
            return (outputs[0][0], outputs[1][0], outputs[2][0], outputs[3][0])

        def sparse_vjp_mat(rho_d, rho_o, rho_w, H, G):
            scalar = jax.ShapeDtypeStruct((), jnp.float64)
            shapes = (
                scalar,
                scalar,
                scalar,
                jax.ShapeDtypeStruct(H.shape, jnp.float64),
            )
            return jax.pure_callback(
                _host_vjp,
                shapes,
                rho_d,
                rho_o,
                rho_w,
                H,
                G,
                vmap_method="sequential",
            )

        return sparse_vjp_mat

    # ------------------------------------------------------------------
    # Cross-sectional SAR sparse Op — wrap scipy splu via jax.pure_callback
    # ------------------------------------------------------------------

    def _make_sar_solve_with_custom_vjp(forward_op, vjp_op):
        """Build a custom_vjp wrapper for SparseSARSolveOp."""

        def _host_solve(rho, rhs):
            outputs = [[None]]
            forward_op.perform(
                None,
                [np.asarray(rho), np.asarray(rhs)],
                outputs,
            )
            return outputs[0][0]

        def _host_vjp(rho, sol, g):
            outputs = [[None], [None]]
            vjp_op.perform(
                None,
                [np.asarray(rho), np.asarray(sol), np.asarray(g)],
                outputs,
            )
            return (outputs[0][0], outputs[1][0])

        @jax.custom_vjp
        def solve(rho, rhs):
            return jax.pure_callback(
                _host_solve,
                jax.ShapeDtypeStruct(rhs.shape, jnp.float64),
                rho,
                rhs,
                vmap_method="sequential",
            )

        def solve_fwd(rho, rhs):
            sol = solve(rho, rhs)
            return sol, (rho, sol)

        def solve_bwd(residuals, g):
            rho, sol = residuals
            scalar = jax.ShapeDtypeStruct((), jnp.float64)
            shapes = (
                scalar,
                jax.ShapeDtypeStruct(sol.shape, jnp.float64),
            )
            grad_rho, grad_rhs = jax.pure_callback(
                _host_vjp,
                shapes,
                rho,
                sol,
                g,
                vmap_method="sequential",
            )
            return grad_rho, grad_rhs

        solve.defvjp(solve_fwd, solve_bwd)
        return solve

    # ------------------------------------------------------------------
    # Cross-sectional SAR sparse Op — Lineax matrix-free iterative solve
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Cross-sectional SAR sparse Op — JAX native GMRES with BCOO
    # ------------------------------------------------------------------

    def _build_jax_gmres_sar_paths(op):
        """Return ``(forward_fn, vjp_fn)`` for the JAX-native GMRES path.

        Uses :class:`jax.experimental.sparse.BCOO` for the weight matrix
        and :func:`jax.scipy.sparse.linalg.gmres` with a diagonal
        (Jacobi) preconditioner.  This path is JIT-compilable, vmappable,
        and differentiable — no host callbacks required.

        The diagonal preconditioner is nearly optimal for 2-D lattice
        spatial weights (bounded degree ≈ 4) and keeps GMRES iteration
        counts low for typical ρ ∈ [0.3, 0.7].

        Parameters
        ----------
        op : SparseSARSolveOp
            The Op being dispatched; ``op._W`` and ``op._n`` are used.

        Returns
        -------
        forward : callable
            ``forward(rho, b) -> eta``.
        vjp : callable
            ``vjp(rho, eta, g) -> (grad_rho, grad_b)``.
        """
        from jax.experimental import sparse as jsparse

        W_bcoo = jsparse.BCOO.from_scipy_sparse(op._W)
        W_T_bcoo = jsparse.BCOO.from_scipy_sparse(op._W.transpose().tocsr())

        # Diagonal preconditioner: M^{-1} = diag(1 / (1 - rho * diag(W)))
        # For row-standardized W, diag(W) is the self-loop weight.
        W_diag_np = np.asarray(op._W.diagonal(), dtype=np.float64)
        W_diag_j = jnp.asarray(W_diag_np, dtype=jnp.float64)

        # GMRES settings — tuned for spatial SAR systems on lattice graphs.
        _GMRES_TOL = float(os.environ.get("NEIGHBAYES_JAX_GMRES_TOL", "1e-8"))
        _GMRES_MAXITER = int(os.environ.get("NEIGHBAYES_JAX_GMRES_MAXITER", "100"))
        _GMRES_RESTART = int(os.environ.get("NEIGHBAYES_JAX_GMRES_RESTART", "20"))

        def _solve(W_matvec, rho, b):
            # Build diagonal preconditioner for this rho
            diag_inv = 1.0 / (1.0 - rho * W_diag_j)

            def matvec(x):
                return x - rho * W_matvec(x)

            def precond(x):
                return diag_inv * x

            x, info = jssl.gmres(
                matvec,
                b,
                tol=_GMRES_TOL,
                atol=0.0,
                maxiter=_GMRES_MAXITER,
                restart=_GMRES_RESTART,
                M=precond,
            )
            # info == 0 means converged; non-zero means maxiter reached.
            # Return the iterate regardless — NUTS will reject non-finite
            # log-prob if the solve is poor.
            return x

        def forward(rho, b):
            return _solve(lambda x: W_bcoo @ x, rho, b)

        def vjp(rho, eta, g):
            # Adjoint system: (I - rho W^T) v = g
            v = _solve(lambda x: W_T_bcoo @ x, rho, g)
            grad_rho = jnp.vdot(v, W_bcoo @ eta)
            return grad_rho, v

        return forward, vjp

    # ------------------------------------------------------------------
    # Cross-sectional SAR sparse Op — eigendecomposition path (default)
    # ------------------------------------------------------------------

    def _build_eigen_sar_paths(op):
        """Return ``(forward_fn, vjp_fn)`` for the eigen SAR path.

        Precomputes the eigendecomposition of W once, then solves
        ``(I - rho W)^{-1} b = V @ diag(1/(1 - rho*lambda)) @ V^{-1} @ b``
        using pure dense JAX operations.  This avoids sparse LU factorization
        entirely and is robust to near-singular system matrices that cause
        ``klu_factor`` to segfault or raise ``INVALID_ARGUMENT``.

        Row-standardized spatial weight matrices are generally non-symmetric,
        so the eigendecomposition uses complex arithmetic.  The final result
        is real-valued (the imaginary parts cancel), and JAX's autodiff
        correctly propagates gradients through the complex→real conversion.

        The gradient w.r.t. ``rho`` is ``v^T W eta`` where
        ``v = (I - rho W^T)^{-1} g``.

        If the Op was constructed with a shared ``eigendecomposition``
        cache (from the model's ``_W_eigendecomposition`` property), it
        is reused here to avoid a redundant O(n³) decomposition.
        """

        # Consume shared eigendecomposition cache if available.
        if op._eigendecomposition is not None:
            eigs_np, V_np, Vinv_np = op._eigendecomposition
        else:
            W_dense = np.asarray(op._W.toarray(), dtype=np.float64)
            eigs_np, V_np = np.linalg.eig(W_dense)
            Vinv_np = np.linalg.inv(V_np)
            # Sort eigenvalues by real part (descending) for numerical stability.
            idx = np.argsort(eigs_np.real)[::-1]
            eigs_np = eigs_np[idx]
            V_np = V_np[:, idx]
            Vinv_np = Vinv_np[idx, :]

        # Use complex128 to handle non-symmetric W correctly.
        # Row-standardized W can have complex eigenvalues/eigenvectors.
        eigs_j = jnp.asarray(eigs_np.astype(np.complex128))
        V_j = jnp.asarray(V_np.astype(np.complex128))
        Vinv_j = jnp.asarray(Vinv_np.astype(np.complex128))
        # Materialize dense W for the gradient (W @ eta).
        # This is O(n²) — dominated by the O(n³) eigendecomposition
        # that we either reuse from cache or compute above.
        W_dense_for_grad = np.asarray(op._W.toarray(), dtype=np.float64)
        W_j = jnp.asarray(W_dense_for_grad, dtype=jnp.float64)

        def forward(rho, b):
            inv_eigs = 1.0 / (1.0 - rho * eigs_j)
            return (V_j @ (inv_eigs * (Vinv_j @ b.astype(jnp.complex128)))).real

        def vjp(rho, eta, g):
            # Adjoint: v = (I - rho W^T)^{-1} g
            # (I - rho W^T)^{-1} = V^{-T} @ diag(1/(1-rho*lambda)) @ V^T g
            # where V^{-T} = conj(Vinv) for the eigendecomposition W = V diag(lam) V^{-1}
            inv_eigs = 1.0 / (1.0 - rho * eigs_j)
            g_c = g.astype(jnp.complex128)
            v = (jnp.conj(Vinv_j).T @ (inv_eigs * (jnp.conj(V_j).T @ g_c))).real
            grad_rho = jnp.vdot(v, W_j @ eta)
            return grad_rho, v

        return forward, vjp

    @jax_funcify.register(SparseSARSolveOp)
    def _funcify_sparse_sar_solve(op, **kwargs):
        # Resolve "auto" to a concrete solver based on problem size.
        resolved = (
            _resolve_auto_sar_solver(op._n) if sar_solver == "auto" else sar_solver
        )

        if resolved == "eigen":
            forward, _ = _build_eigen_sar_paths(op)

            def sparse_sar_solve(rho, b):
                return forward(rho, b)

            return sparse_sar_solve

        if resolved == "jax_gmres":
            forward, _ = _build_jax_gmres_sar_paths(op)

            def sparse_sar_solve(rho, b):
                return forward(rho, b)

            return sparse_sar_solve

        if resolved == "sparsax":
            import sparsax as _chj

            n = op._n
            try:
                from ._logdet._chol_cheb import _d_symmetrize

                # D-symmetrize W: raises ValueError if not symmetrizable.
                W_sym_sp = _d_symmetrize(op._W)  # csc_matrix, symmetric
            except ValueError:
                # Directed / non-symmetrizable W → sparsax LU (KLU or UMFPACK).
                # Fixed COO pattern for A(ρ) = I − ρW over the I ∪ W union; W is
                # never densified (O(nnz)).  sparsax caches the analysis.
                eye_coo = sp.eye(n, format="coo", dtype=np.float64)
                W_coo = op._W.tocoo()
                _rows = np.concatenate([eye_coo.row, W_coo.row])
                _cols = np.concatenate([eye_coo.col, W_coo.col])

                def _aligned(parts):
                    c = sp.coo_matrix(
                        (np.concatenate(parts), (_rows, _cols)), shape=(n, n)
                    )
                    c.sum_duplicates()
                    return c

                eye_c = _aligned([np.ones(eye_coo.nnz), np.zeros(W_coo.nnz)])
                w_c = _aligned([np.zeros(eye_coo.nnz), W_coo.data])
                Ai = jnp.asarray(np.asarray(eye_c.row, dtype=np.int32))
                Aj = jnp.asarray(np.asarray(eye_c.col, dtype=np.int32))
                const_vals = jnp.asarray(eye_c.data, dtype=jnp.float64)
                w_vals = jnp.asarray(w_c.data, dtype=jnp.float64)

                from .samplers._utils._sparsax_lu import sparsax_lu

                lu_solve = sparsax_lu(Ai, Aj, const_vals - 0.5 * w_vals, n).solve

                def sparse_sar_solve(rho, b):
                    return lu_solve(Ai, Aj, const_vals - rho * w_vals, b)

                return sparse_sar_solve

            # Symmetrizable W → sparsax SPD Cholesky.
            # Build COO pattern for I − ρW_sym (upper triangle + diagonal).
            W_sym_coo = W_sym_sp.tocoo()
            mask_upper = W_sym_coo.row <= W_sym_coo.col
            upper_rows = W_sym_coo.row[mask_upper]
            upper_cols = W_sym_coo.col[mask_upper]
            upper_vals = W_sym_coo.data[mask_upper]

            # Add missing diagonal entries (for I, since W_sym has zero
            # diagonal for graphs without self-loops).
            existing_diag = set(zip(upper_rows.tolist(), upper_cols.tolist()))
            diag_rows = []
            diag_cols = []
            for i in range(n):
                if (i, i) not in existing_diag:
                    diag_rows.append(i)
                    diag_cols.append(i)

            _Ai = jnp.asarray(
                np.concatenate(
                    [
                        upper_rows.astype(np.int32),
                        np.array(diag_rows, dtype=np.int32),
                    ]
                )
            )
            _Aj = jnp.asarray(
                np.concatenate(
                    [
                        upper_cols.astype(np.int32),
                        np.array(diag_cols, dtype=np.int32),
                    ]
                )
            )
            # W_sym values at pattern positions (0 for added diagonal)
            _w_vals = jnp.asarray(
                np.concatenate(
                    [
                        upper_vals.astype(np.float64),
                        np.zeros(len(diag_rows), dtype=np.float64),
                    ]
                )
            )
            # Diagonal indices
            _diag_idx = np.full(n, -1, dtype=np.int32)
            all_rows = np.concatenate([upper_rows, np.array(diag_rows)])
            all_cols = np.concatenate([upper_cols, np.array(diag_cols)])
            for k_idx in range(len(all_rows)):
                if all_rows[k_idx] == all_cols[k_idx]:
                    _diag_idx[all_rows[k_idx]] = k_idx
            _diag_idx = jnp.asarray(_diag_idx)

            def sparse_sar_solve(rho, b):
                # Ax = I − ρW_sym at pattern positions
                Ax = -rho * _w_vals
                diag_vals = jnp.zeros_like(_w_vals)
                diag_vals = diag_vals.at[_diag_idx].set(1.0)
                Ax = Ax + diag_vals
                return _chj.solve(_Ai, _Aj, Ax, b)

            return sparse_sar_solve

        solve = _make_sar_solve_with_custom_vjp(op, op._vjp_op)

        def sparse_sar_solve(rho, b):
            return solve(rho, b)

        return sparse_sar_solve

    @jax_funcify.register(_SparseSARVJPOp)
    def _funcify_sparse_sar_vjp(op, **kwargs):
        # Resolve "auto" to a concrete solver based on problem size.
        resolved = (
            _resolve_auto_sar_solver(op._n) if sar_solver == "auto" else sar_solver
        )

        if resolved == "eigen":
            _, vjp = _build_eigen_sar_paths(op)

            def sparse_sar_vjp(rho, eta, g):
                return vjp(rho, eta, g)

            return sparse_sar_vjp

        if resolved == "jax_gmres":
            _, vjp = _build_jax_gmres_sar_paths(op)

            def sparse_sar_vjp(rho, eta, g):
                return vjp(rho, eta, g)

            return sparse_sar_vjp

        # Used by PyTensor's symbolic L_op path. This node is itself the
        # gradient, so pure_callback is sufficient.
        def _host_vjp(rho, eta, g):
            outputs = [[None], [None]]
            op.perform(
                None,
                [np.asarray(rho), np.asarray(eta), np.asarray(g)],
                outputs,
            )
            return (outputs[0][0], outputs[1][0])

        def sparse_sar_vjp(rho, eta, g):
            scalar = jax.ShapeDtypeStruct((), jnp.float64)
            shapes = (
                scalar,
                jax.ShapeDtypeStruct(eta.shape, jnp.float64),
            )
            return jax.pure_callback(
                _host_vjp,
                shapes,
                rho,
                eta,
                g,
                vmap_method="sequential",
            )

        return sparse_sar_vjp

    return True
