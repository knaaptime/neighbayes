"""Shared helpers for sparsax integration in JAX Gibbs samplers.

Provides the COO sparsity-pattern precomputation and value-assembly
utilities needed to use :mod:`sparsax` (JAX-native sparse CHOLMOD)
inside JIT-compiled Gibbs steps.

The precision matrix

.. math::

    P = I + \\mathrm{diag}(\\omega) - \\rho (W + W^T) + \\rho^2 W^T W

is symmetric positive definite for any valid ``ρ`` and ``ω ≥ 0``.
Its sparsity pattern is **fixed** (independent of ``ρ`` and ``ω``),
so we precompute the COO indices ``(Ai, Aj)`` once on the host and
assemble only the values ``Ax(ρ, ω)`` inside the JIT boundary.

This mirrors the NumPy-CHOLMOD pattern in
:func:`neighbayes.samplers.negbin_reduced._core._make_cholmod_pattern`
but returns int32 COO arrays suitable for ``sparsax``.
"""

from __future__ import annotations

import warnings

import numpy as np
import scipy.sparse as sp


def precompute_sparsax_pattern(
    W_csc: sp.csc_matrix,
    n: int,
) -> dict:
    """Precompute the fixed COO sparsity pattern for the precision matrix.

    The pattern covers all fill-in positions of
    ``P = I + diag(ω) − ρ(W+Wᵀ) + ρ²WᵀW`` for any valid ``ρ`` and ``ω ≥ 0``.
    Built as ``I + 0.5*(W+Wᵀ) + 0.25*WᵀW`` so every possible nonzero is present.

    Parameters
    ----------
    W_csc : scipy.sparse.csc_matrix
        The **raw** (row-standardized) spatial weights matrix ``W`` in CSC
        format — *not* ``W+Wᵀ``.  This function derives ``W+Wᵀ`` and ``WᵀW``
        from it internally; passing an already-symmetrized matrix would double
        the symmetric part and corrupt ``WᵀW``.
    n : int
        Matrix dimension.

    Returns
    -------
    dict with keys:
        ``Ai`` : np.ndarray, shape (nnz,), dtype int32 — COO row indices.
        ``Aj`` : np.ndarray, shape (nnz,), dtype int32 — COO column indices.
        ``W_sym_vals`` : np.ndarray, shape (nnz,), dtype float64 —
            Values of ``W + Wᵀ`` at the pattern positions (0 where pattern
            has entries but W+Wᵀ does not).
        ``WtW_vals`` : np.ndarray, shape (nnz,), dtype float64 —
            Values of ``WᵀW`` at the pattern positions.
        ``is_diag`` : np.ndarray, shape (nnz,), dtype bool —
            Boolean mask for diagonal entries (``Ai == Aj``).
        ``diag_idx`` : np.ndarray, shape (n,), dtype int32 —
            Indices into the pattern arrays where the diagonal entries live.
            Used to scatter ``1 + ω`` into ``Ax``.
        ``n`` : int — Matrix dimension.
    """
    W_sym = (W_csc + W_csc.T).tocsc()
    WtW = (W_csc.T @ W_csc).tocsc()
    pattern = (sp.eye(n, format="csc") + 0.5 * W_sym + 0.25 * WtW).tocoo()

    # sparsax reads only the upper triangle, so carrying the lower one would
    # be ~2x the COO entries for no effect.  Restrict the pattern up front.
    keep = pattern.row <= pattern.col
    Ai = pattern.row[keep].astype(np.int32)
    Aj = pattern.col[keep].astype(np.int32)
    nnz = Ai.size

    # Align W_sym / WtW values onto the pattern by matching linearized
    # (row, col) keys.  A Python dict over nnz entries would cost hundreds of
    # MB and seconds at realistic n; searchsorted is O(nnz log nnz) and stays
    # in numpy.
    pattern_keys = Ai.astype(np.int64) * n + Aj.astype(np.int64)
    key_order = np.argsort(pattern_keys, kind="stable")
    sorted_keys = pattern_keys[key_order]

    def _align_upper(M):
        M_coo = M.tocoo()
        vals = np.zeros(nnz, dtype=np.float64)
        if M_coo.nnz == 0:
            return vals
        upper = M_coo.row <= M_coo.col
        mk = M_coo.row[upper].astype(np.int64) * n + M_coo.col[upper].astype(np.int64)
        pos = np.searchsorted(sorted_keys, mk)
        np.clip(pos, 0, max(nnz - 1, 0), out=pos)
        found = sorted_keys[pos] == mk
        vals[key_order[pos[found]]] = M_coo.data[upper][found]
        return vals

    W_sym_vals = _align_upper(W_sym)
    WtW_vals = _align_upper(WtW)

    is_diag = Ai == Aj
    # For each diagonal position i, find its index in the pattern.
    diag_idx = np.full(n, -1, dtype=np.int32)
    diag_pos = np.flatnonzero(is_diag)
    diag_idx[Ai[diag_pos]] = diag_pos.astype(np.int32)

    return {
        "Ai": Ai,
        "Aj": Aj,
        "W_sym_vals": W_sym_vals,
        "WtW_vals": WtW_vals,
        "is_diag": is_diag,
        "diag_idx": diag_idx,
        "n": n,
    }


def make_sparsax_ops(Ai, Aj, n: int):
    """Return ``(eta_sample, solve_logdet)`` factor-once closures over a fixed pattern.

    Both do **one** numeric factorization per call (matching numpy's
    ``CholmodFactor`` reuse), using sparsax 0.4's factor-once primitives when
    available and falling back to the 0.3 idiom otherwise:

    - ``eta_sample(Ax, mean_term, z) -> N(P⁻¹ mean_term, P⁻¹)`` draw — 0.4:
      :func:`sparsax.sample_gaussian` (one factorization); 0.3: mean solve +
      ``MODE_LT`` + ``MODE_PT`` (three solves ≈ three factorizations under vmap).
    - ``solve_logdet(Ax, b) -> (P⁻¹ b, log|P|)`` — 0.4:
      :func:`sparsax.factor_solve` with ``want_logdet=True`` (one factorization,
      no working-copy); 0.3: :func:`sparsax.update_solve` with a zero update
      column and ``return_logdet=True``.

    ``Ai``/``Aj`` are the fixed COO indices (int32); ``b`` may be ``(n,)`` or
    ``(n, n_rhs)``.
    """
    import jax.numpy as jnp
    import sparsax as _chj

    Ai = jnp.asarray(Ai, dtype=jnp.int32)
    Aj = jnp.asarray(Aj, dtype=jnp.int32)

    if hasattr(_chj, "sample_gaussian"):  # sparsax >= 0.4
        _MODE_A = getattr(_chj, "MODE_A", 0)

        def eta_sample(Ax, mean_term, z):
            eta, _mean = _chj.sample_gaussian(Ai, Aj, Ax, mean_term, z)
            return eta

        def solve_logdet(Ax, b):
            sols, ld = _chj.factor_solve(Ai, Aj, Ax, [(b, _MODE_A)], want_logdet=True)
            return sols[0], ld

    else:  # sparsax 0.3 fallback
        _Czero = jnp.zeros((n, 1), dtype=jnp.float64)
        _MODE_LT, _MODE_PT = _chj.MODE_LT, _chj.MODE_PT

        def eta_sample(Ax, mean_term, z):
            m = _chj.solve(Ai, Aj, Ax, mean_term)
            w = _chj.solve(Ai, Aj, Ax, z, mode=_MODE_LT)
            w = _chj.solve(Ai, Aj, Ax, w, mode=_MODE_PT)
            return m + w

        def solve_logdet(Ax, b):
            x, ld = _chj.update_solve(Ai, Aj, Ax, _Czero, b, return_logdet=True)
            return x, ld

    return eta_sample, solve_logdet


def resolve_pg_jax_backend(backend, *, W_sparse, W_sym, WtW, n, logdet_bounds):
    """Resolve the PG-Gibbs backend method and its JAX precomputes.

    Shared by the SAR-logit / SEM-logit / structural SAR-NB Gibbs fits, which
    previously each carried this ~40-line block verbatim.

    Parameters
    ----------
    backend : {"jax", "numpy"}
        Resolved execution backend.
    W_sparse, W_sym, WtW : scipy.sparse matrices
        Raw row-standardized ``W``, ``W + Wᵀ`` and ``WᵀW``.
    n : int
        Number of observations.
    logdet_bounds : LogdetBounds
        The model's resolved logdet bounds (method, rho_min, rho_max).

    Returns
    -------
    method : str
        One of ``"cholmod"`` (numpy), ``"jax_dense"``, ``"cholmod_jax"`` —
        used for all three of the cache's solve/logdet_P/sample methods.
    jax_parts : dict
        ``W_sym_dense``, ``WtW_dense``, ``logdet_jax``, ``sparsax_pattern``
        (all ``None`` on the numpy path).
    """
    jax_parts = {
        "W_sym_dense": None,
        "WtW_dense": None,
        "logdet_jax": None,
        "sparsax_pattern": None,
    }
    if backend != "jax":
        return "cholmod", jax_parts

    from ..._jax_dispatch import (
        _sparsax_available,
        _sparsax_jax_enabled,
        ensure_x64,
    )

    method = (
        "cholmod_jax"
        if _sparsax_jax_enabled() and _sparsax_available()
        else "jax_dense"
    )

    import jax.numpy as jnp

    ensure_x64()

    # Only the dense-Cholesky fallback needs the dense (W+Wᵀ) and WᵀW; the
    # cholmod_jax path assembles P from the sparse COO pattern and does its
    # matvecs via BCOO, so we never densify W there.
    if method == "jax_dense":
        jax_parts["W_sym_dense"] = jnp.asarray(W_sym.toarray(), dtype=jnp.float64)
        jax_parts["WtW_dense"] = jnp.asarray(WtW.toarray(), dtype=jnp.float64)

    from ..._logdet import make_logdet_jax_fn

    jax_parts["logdet_jax"] = make_logdet_jax_fn(
        W_sparse,
        method=logdet_bounds.method,
        rho_min=logdet_bounds.rho_min,
        rho_max=logdet_bounds.rho_max,
    )

    if method == "cholmod_jax":
        # Pass the raw (row-standardized) W; the helper derives W+Wᵀ and WᵀW
        # internally.  Passing W_sym here would double the symmetric part and
        # corrupt WᵀW.
        jax_parts["sparsax_pattern"] = precompute_sparsax_pattern(W_sparse.tocsc(), n)

    return method, jax_parts


# ---------------------------------------------------------------------------
# NumPy-side cached-pattern sparse solve (host loops, no JIT)
# ---------------------------------------------------------------------------


class CachedSparseSolver:
    r"""Sparse direct solver that reuses one symbolic analysis across calls.

    Many posterior-loop hot paths solve

    .. math::

        A(\theta)\, x = b, \qquad A(\theta) = I - \sum_k \theta_k\, W_k,

    repeatedly for many values of ``θ`` (posterior draws, ρ-grid search,
    posterior-predictive replications) with a **fixed** sparsity pattern —
    only the numeric values rescale.  sparsax's sparse LU caches the
    fill-reducing symbolic analysis keyed on the ``(Ai, Aj)`` COO indices,
    so calls sharing a pattern pay the symbolic cost once and each later
    call is just a numeric refactor + triangular solves.

    This helper precomputes the merged COO pattern
    ``(Ai, Aj, const_vals, w_vals_list)`` once and assembles
    ``Ax = const_vals + Σ_k θ_k · w_vals_list[k]`` per call, dispatching to
    sparsax when available and falling back to a per-call scipy ``splu``
    when it is not.  It is the host-side analogue of
    :func:`precompute_sparsax_pattern` / :func:`make_sparsax_ops` for the
    JAX Gibbs path.

    Parameters
    ----------
    weight_matrices : list of scipy.sparse matrices
        The :math:`W_k` (any sparse format).  All must share the same
        shape.  The identity :math:`I` is added internally as a constant
        coefficient (its pattern is merged with that of the :math:`W_k`).
    n : int
        Matrix dimension (``weight_matrices[0].shape[0]``).

    Attributes
    ----------
    Ai, Aj : np.ndarray of int32
        Merged COO row/column indices.
    const_vals : np.ndarray of float64
        Values of the identity at the pattern positions.
    w_vals_list : list of np.ndarray of float64
        Values of each :math:`W_k` at the pattern positions.

    Notes
    -----
    sparsax availability is resolved once at construction time via
    :func:`neighbayes._jax_dispatch._sparsax_available`; no JAX import is
    required on the fallback path.  The merged pattern is built so every
    nonzero of any :math:`W_k` plus the diagonal is present.

    Examples
    --------
    Single-ρ SAR system, many posterior draws::

        solver = CachedSparseSolver([W_sparse], n)
        for rho in rho_draws:
            x = solver.solve([rho], rhs)   # one cached symbolic analysis

    Three-ρ flow system::

        solver = CachedSparseSolver([Wd, Wo, Ww], N)
        for rd, ro, rw in zip(rd_draws, ro_draws, rw_draws):
            x = solver.solve([rd, ro, rw], rhs)
    """

    def __init__(self, weight_matrices, n):
        self.n = int(n)
        mats = [sp.csc_matrix(m) for m in weight_matrices]
        shapes = {m.shape[0] for m in mats}
        if len(shapes) != 1 or shapes.pop() != self.n:
            raise ValueError(
                "All weight matrices must be square with shape (n, n) matching n."
            )
        # Merge the I and every W_k patterns into one COO layout so a single
        # (Ai, Aj) tuple drives every solve; duplicate (i, j) entries are
        # summed, matching how Ax = const + Σ θ_k W_k is evaluated.
        I_coo = sp.eye(self.n, format="coo")
        rows = [I_coo.row]
        cols = [I_coo.col]
        data = [np.ones(I_coo.nnz, dtype=np.float64)]
        for Wk in mats:
            Wk_coo = Wk.tocoo()
            rows.append(Wk_coo.row)
            cols.append(Wk_coo.col)
            data.append(Wk_coo.data.astype(np.float64, copy=False))
        rows = np.concatenate(rows)
        cols = np.concatenate(cols)
        data = np.concatenate(data)
        merged = sp.coo_matrix((data, (rows, cols)), shape=(self.n, self.n))
        merged.eliminate_zeros()
        merged.sum_duplicates()
        # Split merged values back into const (I) and per-W contributions.
        self.Ai = merged.row.astype(np.int32)
        self.Aj = merged.col.astype(np.int32)

        # Align each matrix's values onto the merged pattern by matching
        # linearized (row, col) keys.  Sparse fancy-indexing would either
        # densify the n x n matrix or build an nnz-sized Python dict; a
        # searchsorted over the sorted key array is O(nnz log nnz) and
        # stays sparse.
        pattern_keys = self.Ai.astype(np.int64) * self.n + self.Aj.astype(np.int64)
        key_order = np.argsort(pattern_keys, kind="stable")
        sorted_keys = pattern_keys[key_order]

        def _align(M):
            """Values of sparse ``M`` at the merged pattern positions."""
            M_coo = M.tocoo()
            M_coo.sum_duplicates()
            vals = np.zeros(pattern_keys.size, dtype=np.float64)
            if M_coo.nnz == 0:
                return vals
            mk = M_coo.row.astype(np.int64) * self.n + M_coo.col.astype(np.int64)
            pos = np.searchsorted(sorted_keys, mk)
            # Entries dropped by eliminate_zeros() have no slot; mask them
            # out rather than scattering into a neighboring position.
            np.clip(pos, 0, sorted_keys.size - 1, out=pos)
            found = sorted_keys[pos] == mk
            np.add.at(
                vals,
                key_order[pos[found]],
                M_coo.data.astype(np.float64, copy=False)[found],
            )
            return vals

        self.const_vals = _align(I_coo)
        self.w_vals_list = [_align(Wk) for Wk in mats]

        from ..._jax_dispatch import _sparsax_available

        self._use_sparsax = _sparsax_available()
        self._Ai_jax = None
        self._Aj_jax = None
        self._const_jax = None
        self._w_jax_list = None
        self._has_lu_factor = False
        self._lu = None
        if self._use_sparsax:
            import jax.numpy as jnp
            import sparsax as sparsax_mod

            from ..._jax_dispatch import ensure_x64
            from ._sparsax_lu import sparsax_lu

            ensure_x64()
            self._Ai_jax = jnp.asarray(self.Ai, dtype=jnp.int32)
            self._Aj_jax = jnp.asarray(self.Aj, dtype=jnp.int32)
            self._const_jax = jnp.asarray(self.const_vals, dtype=jnp.float64)
            self._w_jax_list = [
                jnp.asarray(v, dtype=jnp.float64) for v in self.w_vals_list
            ]
            self._has_lu_factor = hasattr(sparsax_mod, "lu_factor") and hasattr(
                sparsax_mod, "lu_solve_factor"
            )
            # KLU or UMFPACK, whichever is faster on this pattern, probed at
            # A = I - Σ_k (0.5 / K) W_k, inside the stable region.
            probe_coeffs = [-0.5 / max(len(mats), 1)] * len(mats)
            self._lu = sparsax_lu(
                self._Ai_jax, self._Aj_jax, self._assemble_Ax(probe_coeffs), self.n
            )
        # Last (coeffs -> LU token) pair, so back-to-back solves at the same
        # θ (e.g. several RHS blocks per posterior draw) skip the refactor.
        self._last_coeffs = None
        self._last_token = None
        self._splu = None  # scipy fallback factor, held by factorize()

    def _assemble_Ax(self, coeffs):
        ax = self._const_jax
        for c, wv in zip(coeffs, self._w_jax_list):
            ax = ax + float(c) * wv
        return ax

    def _token(self, coeffs):
        """Return an LU token for ``A(θ)``, reusing the last one when θ repeats."""
        key = tuple(float(c) for c in coeffs)
        if self._last_token is not None and key == self._last_coeffs:
            return self._last_token
        token = self._lu.factor(
            self._Ai_jax, self._Aj_jax, self._assemble_Ax(coeffs), self.n
        )
        self._last_coeffs = key
        self._last_token = token
        return token

    def solve(self, coeffs, rhs):
        """Solve :math:`A(\\theta) x = b` for vector RHS.

        Parameters
        ----------
        coeffs : sequence of float
            Coefficients :math:`\\theta_k` for each weight matrix, in the
            order passed to the constructor.  ``A = I + Σ θ_k W_k``; for the
            usual ``I - ρ W`` form pass ``[-ρ]`` (or equivalently ``solve``
            assembles ``const + Σ θ_k W_k``, so ``[-ρ]`` gives ``I - ρ W``).
        rhs : ndarray, shape (n,) or (n, k)
            Right-hand side(s).

        Returns
        -------
        x : ndarray, shape matching ``rhs``
        """
        rhs_np = np.asarray(rhs, dtype=np.float64)
        single = rhs_np.ndim == 1
        if single:
            rhs_np = rhs_np[:, None]
        if self._use_sparsax:
            import jax.numpy as jnp

            b = jnp.asarray(rhs_np)
            if self._has_lu_factor:
                # One numeric factorization, then every RHS column solved
                # against the held token.
                out = np.asarray(
                    self._lu.solve_factor(self._token(coeffs), b), dtype=np.float64
                )
            else:
                # Older sparsax: lu_solve still takes a 2-D RHS and caches
                # the symbolic analysis on (Ai, Aj).
                out = np.asarray(
                    self._lu.solve(
                        self._Ai_jax, self._Aj_jax, self._assemble_Ax(coeffs), b
                    ),
                    dtype=np.float64,
                )
        else:
            # Fallback: assemble scipy sparse A and factorize once per call,
            # then solve all columns against that factor.
            A_csc = sp.csc_matrix(
                (self._numpy_Ax(coeffs), (self.Ai, self.Aj)),
                shape=(self.n, self.n),
            )
            out = np.asarray(sp.linalg.splu(A_csc).solve(rhs_np), dtype=np.float64)
        return out[:, 0] if single else out

    def _numpy_Ax(self, coeffs):
        ax = self.const_vals.copy()
        for c, wv in zip(coeffs, self.w_vals_list):
            ax = ax + float(c) * wv
        return ax

    # -- Explicit factor / solve / logdet, for callers that reuse one θ -----

    def factorize(self, coeffs):
        """Factor ``A(θ)`` and hold the factor for later solves.

        Separating this from :meth:`solve` lets a caller pay one numeric
        factorization and then issue many solves and read ``log|A|`` off the
        same factor — the pattern the ρ-slice samplers need.
        """
        if self._use_sparsax and self._has_lu_factor:
            self._token(coeffs)
        else:
            A_csc = sp.csc_matrix(
                (self._numpy_Ax(coeffs), (self.Ai, self.Aj)),
                shape=(self.n, self.n),
            )
            self._splu = sp.linalg.splu(A_csc)
            self._last_coeffs = tuple(float(c) for c in coeffs)
        return self

    def solve_factored(self, rhs):
        """Solve against the factor held by the last :meth:`factorize` call."""
        rhs_np = np.asarray(rhs, dtype=np.float64)
        single = rhs_np.ndim == 1
        if single:
            rhs_np = rhs_np[:, None]
        if self._use_sparsax and self._has_lu_factor:
            import jax.numpy as jnp

            if self._last_token is None:
                raise RuntimeError("factorize() must be called before solve_factored()")
            out = np.asarray(
                self._lu.solve_factor(self._last_token, jnp.asarray(rhs_np)),
                dtype=np.float64,
            )
        else:
            if getattr(self, "_splu", None) is None:
                raise RuntimeError("factorize() must be called before solve_factored()")
            out = np.asarray(self._splu.solve(rhs_np), dtype=np.float64)
        return out[:, 0] if single else out

    def logdet(self):
        r"""``log|det A(θ)|`` from the held factor.

        For :math:`A = I - \rho W` with ``ρ`` inside the stability range the
        determinant is positive, so this is :math:`\log\det(I-\rho W)` — the
        SAR Jacobian term, free from a factorization the sampler already
        needed.  Valid for asymmetric ``W``: the LU carries the determinant
        just as Cholesky does for the symmetric case.
        """
        if self._use_sparsax and self._has_lu_factor:
            if self._last_token is None:
                raise RuntimeError("factorize() must be called before logdet()")
            return float(self._lu.logdet_factor(self._last_token))
        if getattr(self, "_splu", None) is None:
            raise RuntimeError("factorize() must be called before logdet()")
        return float(np.sum(np.log(np.abs(self._splu.U.diagonal()))))


class KluSarSolver:
    r"""Solve :math:`(I-\rho W)x = b` by KLU on ``A`` directly.

    Drop-in replacement for the CHOLMOD normal-equation solver used by the
    reduced-form samplers, exposing the same ``factorize(rho)`` /
    ``solve(rhs)`` pair.  The normal-equation route factorizes

    .. math::

        A^\top A = I - \rho(W + W^\top) + \rho^2 W^\top W,

    which is only worth its cost when ``W`` is symmetric: for asymmetric
    ``W`` it squares the condition number and :math:`W^\top W` carries
    several times the nonzeros of ``W`` (two-hop fill-in), so the Cholesky
    is both slower and less accurate than an LU of ``A`` itself.  A sparse LU
    (KLU or UMFPACK, whichever measures faster) factorizes the unsymmetric
    ``A`` once per ρ and additionally hands back :math:`\log\det A` for free
    via :meth:`logdet`.

    Parameters
    ----------
    W : scipy.sparse matrix, shape (n, n)
        Spatial weights.  May be asymmetric.
    n : int
        Matrix dimension.
    """

    def __init__(self, W, n):
        self._inner = CachedSparseSolver([W], n)
        self._n = int(n)
        self._rho: float | None = None

    def factorize(self, rho: float) -> None:
        """Factor ``A = I − ρW`` at the given ρ."""
        self._rho = float(rho)
        self._inner.factorize([-float(rho)])

    def solve(self, rhs: np.ndarray) -> np.ndarray:
        """Solve ``(I − ρW) x = rhs`` (vector or matrix RHS)."""
        if self._rho is None:
            raise RuntimeError("factorize(rho) must be called before solve()")
        return self._inner.solve_factored(rhs)

    def logdet(self) -> float:
        """``log det(I − ρW)`` from the held LU factor."""
        if self._rho is None:
            raise RuntimeError("factorize(rho) must be called before logdet()")
        return self._inner.logdet()


def profile_loglik_rho_grid(
    y,
    X,
    W_sparse,
    *,
    rho_min: float = 0.05,
    rho_max: float = 0.95,
    rho_step: float = 0.05,
):
    r"""Profile-log-likelihood ρ-grid search with cached sparse solves.

    For each candidate ρ on ``[rho_min, rho_max]`` step ``rho_step``, solves
    :math:`\tilde X = (I - \rho W)^{-1} X` and computes the Gaussian
    profile log-likelihood

    .. math::

        \ell_p(\rho) = -\tfrac{n}{2}\log\hat\sigma^2(\rho) - \tfrac{n}{2},
        \quad \hat\beta(\rho) = (\tilde X^\top \tilde X)^{-1} \tilde X^\top y,
        \quad \hat\sigma^2(\rho) = \tfrac{1}{n}\|y - \tilde X \hat\beta\|^2.

    The sparsity pattern of :math:`I - \rho W` is independent of ρ, so a
    single :class:`CachedSparseSolver` is built once and reused across the
    whole grid (sparsax caches the fill-reducing symbolic analysis; scipy
    fallback still benefits from the precomputed pattern assembly).

    Parameters
    ----------
    y, X : ndarray, shapes (n,) and (n, k)
        Response and design matrix.
    W_sparse : scipy.sparse matrix, shape (n, n)
        Row-standardized spatial weights.
    rho_min, rho_max, rho_step : float
        Grid definition.

    Returns
    -------
    best_rho : float
        Grid argmax of the profile log-likelihood.
    best_beta : ndarray, shape (k,)
        Least-squares β at ``best_rho``.
    best_ll : float
        Profile log-likelihood at ``best_rho``.  ``-np.inf`` when every solve
        failed (e.g. ρ outside the valid range for ``W``).
    """
    y = np.asarray(y, dtype=np.float64)
    X = np.asarray(X, dtype=np.float64)
    n, k = X.shape
    solver = CachedSparseSolver([W_sparse], n)
    # Inclusive of rho_max (up to floating-point slack): np.arange would drop
    # the endpoint, silently truncating the default grid at 0.90.
    n_steps = int(np.floor((rho_max - rho_min) / rho_step + 1e-9)) + 1
    grid = rho_min + rho_step * np.arange(max(n_steps, 1))
    best_rho, best_beta, best_ll = 0.0, np.zeros(k), -np.inf
    failures: list[tuple[float, str]] = []
    for rho_g in grid:
        try:
            Xtilde = solver.solve([-float(rho_g)], X)
        except (RuntimeError, ValueError, np.linalg.LinAlgError) as exc:
            # A singular / indefinite A_ρ means this ρ is outside the valid
            # range for W; skip it but keep a record so an all-failed grid
            # can say why rather than returning a bare -inf.
            failures.append((float(rho_g), str(exc)))
            continue
        beta_g = np.linalg.lstsq(Xtilde, y, rcond=None)[0]
        eta_g = Xtilde @ beta_g
        sig2_g = float(np.mean((y - eta_g) ** 2))
        if sig2_g > 1e-10:
            ll_g = -0.5 * n * np.log(sig2_g) - 0.5 * n
            if ll_g > best_ll:
                best_ll, best_rho, best_beta = ll_g, float(rho_g), beta_g.copy()
    if not np.isfinite(best_ll) and failures:
        warnings.warn(
            f"profile_loglik_rho_grid: every ρ in [{rho_min}, {rho_max}] failed to "
            f"solve ({len(failures)} candidates); first error at ρ={failures[0][0]}: "
            f"{failures[0][1]}",
            RuntimeWarning,
            stacklevel=2,
        )
    return best_rho, best_beta, best_ll


def cached_sar_solve(W_sparse, n, rho, rhs):
    """Solve :math:`(I - \\rho W) x = b` with one cached symbolic analysis.

    Thin convenience wrapper over :class:`CachedSparseSolver` for the common
    one-off case: build the solver, solve once, return the result.  Callers
    doing many solves should construct a :class:`CachedSparseSolver` once
    and call ``solve([-ρ], rhs)`` per draw to amortise the pattern build.
    """
    return CachedSparseSolver([W_sparse], n).solve([-float(rho)], rhs)
