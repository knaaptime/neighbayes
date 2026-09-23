"""JAX-native log-determinant evaluation.

Mirrors the PyTensor symbolic functions but uses ``jax.numpy`` so the
returned callables are compatible with ``jax.jit`` and ``jax.grad``.

Supports ``"eigenvalue"``, ``"chebyshev"``, ``"cheb_stochastic"``,
``"cheb_cholesky"``, ``"aaa"``, and ``"slq"``.  The stochastic Chebyshev
method precomputes moments in numpy and evaluates via JAX-native Clenshaw.
SLQ consumes the numpy :func:`slq_logdet_precompute` quadrature rules
(sparse batched D-symmetrized Lanczos, canonical ``n·v₁²`` weights; complex
bilinear ``γ`` for the directed-``W`` Arnoldi fallback) and evaluates the
ρ-dependent quadrature in JAX — differentiable and JIT-compatible, with no
dense ``W`` materialization.  ``cheb_cholesky`` precomputes Chebyshev
coefficients via sparse Cholesky in numpy and evaluates via JAX-native
Clenshaw.  ``aaa`` precomputes support points and barycentric weights via
sparse LU in numpy and evaluates via JAX-native barycentric formula.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from ._chebyshev import chebyshev
from ._config import resolve_logdet_method
from ._slq import slq_logdet_precompute


def jax_logdet_chebyshev_traced(rho, coeffs, rmin, rmax):
    """Clenshaw evaluation with ``coeffs``/``rmin``/``rmax`` as *traced* values.

    :func:`jax_logdet_chebyshev` bakes the coefficients in as compile-time
    constants and unrolls the recurrence in Python, so changing them — or
    changing their length — forces a full retrace.  That is fine when the
    interpolant is fixed for the life of a chain, and it is the faster option
    when it is: the unrolled form has no loop overhead.

    It is not fine for a warmup-adaptive refit, which replaces the interpolant
    partway through a run.  On a Gibbs chain the retrace costs on the order of a
    second — an order of magnitude more than the refit's own factorizations —
    which would make the refit a net loss on this backend.  This variant keeps
    everything traced and drives the recurrence with ``lax.fori_loop`` over a
    fixed-capacity array, so a refit changes array *values* only and the
    compiled step is reused unchanged.

    Zero-padding is exact, not an approximation: Clenshaw's ``b`` terms are
    initialized to zero and a run of zero coefficients leaves them at zero, so
    padding a degree-``m`` series out to capacity ``M`` reproduces the same
    ``b_m = b_{m+1} = 0`` the unpadded recurrence starts from.  Pass a capacity
    at least as large as any order the refit may select.
    """
    import jax
    import jax.numpy as jnp

    c = jnp.asarray(coeffs, dtype=jnp.float64)
    m = c.shape[0]
    x = (2.0 * rho - rmax - rmin) / (rmax - rmin)

    if m == 0:
        return jnp.zeros_like(rho)
    if m == 1:
        return jnp.broadcast_to(c[0], jnp.shape(x)) * jnp.ones_like(x)

    def body(i, carry):
        b_curr, b_next = carry
        k = m - 1 - i  # descends m-2 .. 1
        return (2.0 * x * b_curr - b_next + c[k], b_curr)

    b_curr = jnp.broadcast_to(c[m - 1], jnp.shape(x))
    b_next = jnp.zeros_like(x)
    b_curr, b_next = jax.lax.fori_loop(1, m - 1, body, (b_curr, b_next))

    return c[0] + x * b_curr - b_next


def jax_logdet_chebyshev(
    rho,
    coeffs: np.ndarray,
    rmin: float = -1.0,
    rmax: float = 1.0,
):
    """Evaluate Chebyshev approximation of log|I - ρW| in JAX via Clenshaw."""
    import jax.numpy as jnp

    m = len(coeffs)
    if m == 0:
        return jnp.zeros_like(rho)

    x = (2.0 * rho - rmax - rmin) / (rmax - rmin)

    if m == 1:
        return jnp.full_like(rho, coeffs[0])

    c = jnp.asarray(coeffs, dtype=jnp.float64)
    b_next = jnp.zeros_like(x)
    b_curr = jnp.broadcast_to(c[m - 1], jnp.shape(x))

    for k in range(m - 2, 0, -1):
        b_new = 2.0 * x * b_curr - b_next + c[k]
        b_next = b_curr
        b_curr = b_new

    return c[0] + x * b_curr - b_next


def make_logdet_jax_param_fn(method: str, T: int = 1):
    """Return ``fn(rho, params) -> logdet`` for a refittable method.

    The counterpart to :func:`make_logdet_jax_fn` for samplers that swap the
    interpolant mid-run.  Where that factory closes over the precomputed
    coefficients — making them compile-time constants, so any change forces a
    retrace — this one takes them as an argument, so a caller can carry them in
    the sampler state and substitute a refit without recompiling the step.

    ``params`` comes from :meth:`~._refit.LogdetRefitter.jax_params` and is
    zero-padded to a fixed capacity, so its shapes never change either.

    Parameters
    ----------
    method : str
        ``"cheb_cholesky"`` (Chebyshev coefficients), or ``"aaa"`` /
        ``"chol_aaa"`` (barycentric support points, values and weights).  The
        two AAA variants differ only in the factorizer used to obtain the
        support values at precompute time — LU versus Cholesky — and share one
        evaluation form, so they share this parameterization.
    T : int, default 1
        Panel replication factor applied to the result.

    Returns
    -------
    callable
        ``(rho, params) -> jax.numpy.ndarray``, differentiable and
        JIT-compatible.
    """
    T = int(T)

    if method in ("cheb_cholesky", "lu_cheb"):

        def _cheb(rho, params):
            coeffs, rmin, rmax = params
            val = jax_logdet_chebyshev_traced(rho, coeffs, rmin, rmax)
            return val if T == 1 else T * val

        return _cheb

    if method in ("aaa", "chol_aaa"):

        def _aaa(rho, params):
            import jax.numpy as jnp

            z_j, f_j, w_j = params
            # Padded entries carry w_j = 0 at a z_j far outside the interval,
            # so they contribute nothing and never divide by zero.
            diff = rho - z_j
            val = jnp.sum(w_j * f_j / diff) / jnp.sum(w_j / diff)
            return val if T == 1 else T * val

        return _aaa

    raise ValueError(
        f"make_logdet_jax_param_fn does not support method {method!r}; "
        "only 'cheb_cholesky', 'lu_cheb', 'aaa' and 'chol_aaa' carry a refittable "
        "parameterization."
    )


def make_logdet_jax_fn(
    W,
    method: str | None = None,
    rho_min: float = -1.0,
    rho_max: float = 1.0,
    T: int = 1,
):
    """Return a JAX-native ``(rho) -> log|I - ρW|`` callable.

    Supports ``"eigenvalue"``, ``"chebyshev"``, ``"cheb_stochastic"``,
    ``"cheb_cholesky"``, ``"aaa"``, ``"cholmod"``, and ``"slq"``.
    ``"cholmod"`` uses ``sparsax`` for exact sparse CHOLMOD logdet
    (requires the ``sparsax`` package; CPU-only).
    """
    T = int(T)

    eigs = None
    if sp.issparse(W):
        W_sparse = W.tocsr().astype(np.float64)
        n = W_sparse.shape[0]
    else:
        W_arr = np.asarray(W, dtype=np.float64)
        if W_arr.ndim == 1:
            eigs = W_arr
            n = len(eigs)
        else:
            n = W_arr.shape[0]
            W_sparse = sp.csr_matrix(W_arr)

    method = resolve_logdet_method(
        method, n=n, W=W_sparse if "W_sparse" in dir() else W_arr
    )

    if method == "eigenvalue":
        if eigs is None:
            eigs = np.linalg.eigvals(W_sparse.toarray())
        _eigs = np.asarray(eigs, dtype=np.complex128)

        def _jax_eigenvalue(rho):
            import jax.numpy as jnp

            eigs_jax = jnp.asarray(_eigs)
            result = jnp.sum(jnp.log(jnp.abs(1.0 - rho * eigs_jax)))
            return result if T == 1 else T * result

        return _jax_eigenvalue

    if method == "chebyshev":
        out = chebyshev(
            W_sparse if eigs is None else None,
            order=20,
            rmin=rho_min,
            rmax=rho_max,
            eigs=eigs,
        )
        coeffs = out["coeffs"].astype(np.float64)
        rmin_cb = float(out["rmin"])
        rmax_cb = float(out["rmax"])

        def _jax_chebyshev(rho):
            val = jax_logdet_chebyshev(rho, coeffs, rmin=rmin_cb, rmax=rmax_cb)
            return val if T == 1 else T * val

        return _jax_chebyshev

    if method == "cheb_stochastic":
        # Precompute stochastic moments in numpy, then evaluate at Chebyshev
        # nodes in ρ-space and fit a Chebyshev-in-ρ polynomial for JAX
        # Clenshaw evaluation (differentiable, JIT-compatible).  The series runs
        # to ~117 terms on the full prior interval; the looped recurrence
        # compiles about 2 s faster than the unrolled one in a Gibbs step at
        # n = 10,000, and draws cost the same (measured 2026-09-19).
        from ._factories import _cheb_stochastic_coeffs

        coeffs, rmin_cb, rmax_cb = _cheb_stochastic_coeffs(W_sparse, rho_min, rho_max)

        def _jax_cheb_stochastic(rho):
            val = jax_logdet_chebyshev_traced(rho, coeffs, rmin_cb, rmax_cb)
            return val if T == 1 else T * val

        return _jax_cheb_stochastic

    if method == "slq":
        if eigs is not None:
            raise ValueError(
                "SLQ requires the weight matrix W, not a 1-D eigenvalue array."
            )
        # Consume the numpy sparse SLQ rules: batched D-symmetrized Lanczos
        # (real nodes θ and real n·v₁² weights) for undirected W, or Arnoldi
        # (complex Ritz values θ and complex bilinear weights γ) for directed
        # W.  Evaluate the ρ-dependent quadrature in JAX via the complex log;
        # the Lanczos case is the zero-imaginary special case of the same
        # formula.  No dense W is materialized — the precompute is matvec-only.
        pre = slq_logdet_precompute(W_sparse)
        nodes = np.asarray(pre.nodes)
        weights = np.asarray(pre.weights)
        n_probes = pre.n_probes

        # The exact-moment control variate is part of the value the numpy
        # evaluator returns, so it has to be carried into the traced graph too;
        # otherwise the JAX and numpy paths compute different functions.
        cv = (
            np.zeros(0)
            if pre.cv_coeffs is None
            else np.asarray(pre.cv_coeffs, dtype=np.float64)
        )

        nodes_real = np.ascontiguousarray(nodes.real.astype(np.float64))
        nodes_imag = np.ascontiguousarray(nodes.imag.astype(np.float64))
        w_real = np.ascontiguousarray(weights.real.astype(np.float64))
        w_imag = np.ascontiguousarray(weights.imag.astype(np.float64))

        def _jax_slq(rho):
            import jax.numpy as jnp

            nr = jnp.asarray(nodes_real)
            ni = jnp.asarray(nodes_imag)
            wr = jnp.asarray(w_real)
            wi = jnp.asarray(w_imag)
            # 1 - ρθ = (1 - ρ·Re θ) + i(-ρ·Im θ)
            # log(1 - ρθ) = 0.5·log|1-ρθ|² + i·atan2(Im, Re)
            re = 1.0 - rho * nr
            im = -rho * ni
            log_re = 0.5 * jnp.log(jnp.maximum(re**2 + im**2, 1e-300))
            log_im = jnp.arctan2(im, re)
            # Re(Σ γ·log(1-ρθ)) = Σ [Re(γ)·log_re - Im(γ)·log_im]; the second
            # term vanishes for real (Lanczos) weights but keeps the cross term
            # that a magnitude-only log would drop for the complex Arnoldi case.
            val = jnp.sum(wr * log_re - wi * log_im) / n_probes
            if cv.size:
                j = jnp.arange(1, cv.size + 1, dtype=jnp.float64)
                val = val + jnp.sum(rho**j / j * jnp.asarray(cv))
            return val if T == 1 else T * val

        return _jax_slq

    if method in ("cheb_cholesky", "lu_cheb"):
        from ._factories import _cheb_precompute_for

        # Precompute Chebyshev coefficients in numpy — via sparse Cholesky for
        # ``cheb_cholesky``, sparse LU for ``lu_cheb`` — then evaluate via
        # JAX-native Clenshaw (differentiable, JIT-compatible).
        pre = _cheb_precompute_for(method)(
            W_sparse, order=None, rho_min=rho_min, rho_max=rho_max
        )
        coeffs = pre.coeffs.astype(np.float64)
        rmin_cb = float(pre.rho_min)
        rmax_cb = float(pre.rho_max)

        def _jax_cheb_chol(rho):
            val = jax_logdet_chebyshev(rho, coeffs, rmin=rmin_cb, rmax=rmax_cb)
            return val if T == 1 else T * val

        return _jax_cheb_chol

    if method == "aaa":
        from ._aaa import aaa_logdet_precompute

        # Precompute support points and barycentric weights via sparse LU
        # in numpy, then evaluate via JAX-native barycentric formula
        # (differentiable, JIT-compatible).
        pre = aaa_logdet_precompute(W_sparse, rho_min=rho_min, rho_max=rho_max)
        sp_z = pre.support_points.astype(np.float64)
        sp_f = pre.support_values.astype(np.float64)
        w = pre.weights.astype(np.float64)

        # Same barycentric formula as the refittable form; the only difference
        # is that the support arrays are fixed here, so bind them and reuse it.
        _bary = make_logdet_jax_param_fn("aaa", T=T)

        def _jax_aaa(rho):
            import jax.numpy as jnp

            return _bary(rho, (jnp.asarray(sp_z), jnp.asarray(sp_f), jnp.asarray(w)))

        return _jax_aaa

    if method == "chol_aaa":
        from ._aaa import chol_aaa_logdet_precompute

        # Same as "aaa" but precompute via sparse Cholesky of the D-symmetrized
        # system (~2× cheaper than KLU for symmetrizable W).
        pre = chol_aaa_logdet_precompute(W_sparse, rho_min=rho_min, rho_max=rho_max)
        sp_z = pre.support_points.astype(np.float64)
        sp_f = pre.support_values.astype(np.float64)
        w = pre.weights.astype(np.float64)

        _bary = make_logdet_jax_param_fn("aaa", T=T)

        def _jax_chol_aaa(rho):
            import jax.numpy as jnp

            return _bary(rho, (jnp.asarray(sp_z), jnp.asarray(sp_f), jnp.asarray(w)))

        return _jax_chol_aaa

    if method == "grid_spline":
        from ._grid_spline import (
            grid_spline_breaks_coeffs,
            grid_spline_eval_jax,
            grid_spline_logdet_precompute,
        )

        # Incumbent practice: precompute the exact LU grid and fit the cubic
        # spline in numpy, then evaluate its piecewise-cubic form JAX-natively
        # (differentiable, JIT-compatible), mirroring the "aaa" split above.
        pre = grid_spline_logdet_precompute(W_sparse, rho_min=rho_min, rho_max=rho_max)
        breaks_np, coeffs_np = grid_spline_breaks_coeffs(pre)

        def _jax_grid_spline(rho):
            import jax.numpy as jnp

            return grid_spline_eval_jax(
                rho, jnp.asarray(breaks_np), jnp.asarray(coeffs_np), T=T
            )

        return _jax_grid_spline

    if method == "cholmod":
        # JAX-native exact logdet via sparsax sparse CHOLMOD.
        # Requires W to be D-symmetrizable (row-standardized undirected
        # graph): W = D⁻¹A with symmetric A → W_sym = D^{1/2} W D^{-1/2}
        # is symmetric with the same eigenvalues, so I−ρW_sym is SPD
        # for |ρ| < 1 and sparsax.logdet applies directly.
        # If W is not D-symmetrizable (directed graph), this raises
        # ValueError — use logdet_method="aaa" for such matrices.
        from .._jax_dispatch import _sparsax_available

        if not _sparsax_available():
            raise ImportError(
                "logdet method 'cholmod' requires the 'sparsax' package. "
                "Install it with: pip install sparsax"
            )

        from ._chol_cheb import _d_symmetrize

        # D-symmetrize: raises ValueError if W is not symmetrizable.
        W_sym_sp = _d_symmetrize(W_sparse)  # csc_matrix, symmetric

        # Build the COO pattern for I − ρW_sym.
        # sparsax reads only Ai <= Aj entries (upper triangle),
        # so we include the upper triangle of W_sym plus all diagonal
        # entries (for the I in I − ρW_sym, since W_sym has zero diagonal
        # for graphs without self-loops).
        W_sym_coo = W_sym_sp.tocoo()
        mask_upper = W_sym_coo.row <= W_sym_coo.col
        upper_rows = W_sym_coo.row[mask_upper]
        upper_cols = W_sym_coo.col[mask_upper]
        upper_vals = W_sym_coo.data[mask_upper]

        # Add diagonal entries that are missing from W_sym's pattern
        existing_diag = set(zip(upper_rows.tolist(), upper_cols.tolist()))
        diag_rows = []
        diag_cols = []
        for i in range(n):
            if (i, i) not in existing_diag:
                diag_rows.append(i)
                diag_cols.append(i)

        _Ai_direct = np.concatenate(
            [
                upper_rows.astype(np.int32),
                np.array(diag_rows, dtype=np.int32),
            ]
        )
        _Aj_direct = np.concatenate(
            [
                upper_cols.astype(np.int32),
                np.array(diag_cols, dtype=np.int32),
            ]
        )
        # W_sym values at these positions (0 for added diagonal entries)
        _W_sym_vals = np.concatenate(
            [
                upper_vals.astype(np.float64),
                np.zeros(len(diag_rows), dtype=np.float64),
            ]
        )
        _nnz_direct = len(_Ai_direct)
        _n_static = n

        # Diagonal indices for I − ρW_sym
        _diag_idx_direct = np.full(n, -1, dtype=np.int32)
        for k_idx in range(_nnz_direct):
            if _Ai_direct[k_idx] == _Aj_direct[k_idx]:
                _diag_idx_direct[_Ai_direct[k_idx]] = k_idx

        def _jax_cholmod(rho):
            import jax.numpy as jnp
            import sparsax

            Ai = jnp.asarray(_Ai_direct, dtype=jnp.int32)
            Aj = jnp.asarray(_Aj_direct, dtype=jnp.int32)
            W_vals = jnp.asarray(_W_sym_vals, dtype=jnp.float64)
            diag_idx = jnp.asarray(_diag_idx_direct, dtype=jnp.int32)

            # Ax = I − ρW_sym at pattern positions
            Ax = -rho * W_vals
            diag_vals = jnp.zeros(_nnz_direct, dtype=jnp.float64)
            diag_vals = diag_vals.at[diag_idx].set(1.0)
            Ax = Ax + diag_vals

            val = sparsax.logdet(Ai, Aj, Ax, _n_static)
            return val if T == 1 else T * val

        return _jax_cholmod

    raise ValueError(
        f"Method '{method}' has no JAX implementation. "
        "Use 'eigenvalue', 'chebyshev', 'cheb_stochastic', "
        "'cheb_cholesky', 'aaa', 'chol_aaa', 'cholmod', or 'slq'."
    )
