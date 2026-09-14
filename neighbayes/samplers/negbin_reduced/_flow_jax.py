r"""JAX/sparsax sparse solve primitives for the unrestricted flow NB Gibbs sampler.

The unrestricted origin–destination flow model has system matrix

.. math::

    A(\rho_d, \rho_o, \rho_w) = I - \rho_d W_d - \rho_o W_o - \rho_w W_w

on the ``N = n^2`` flow lattice.  ``A`` is **directed** (non-symmetric,
non-D-symmetrizable), so no Cholesky applies; and it is far too large to
densify (``N \times N`` with ``N = n^2``).  The numpy chain factorizes the
sparse ``A`` on the host every time a ``\rho`` moves (see
``_flow._solve_A_unrestricted``).  This module provides the JAX-native
equivalent: a single ``sparsax`` symbolic analysis reused across the whole
run, with per-``\rho`` numeric refactor-and-solve that is JIT-compatible and
autodiff-capable — the enabling piece for a GPU-friendly flow backend.

The crucial invariant is that **the sparsity pattern of ``A`` is constant**
across ``\rho`` (it is the structural union of ``I, W_d, W_o, W_w``).  We
build that shared pattern once and carry four value vectors aligned to it, so
each solve only rescales values and calls sparsax's LU solve — the
symbolic factorization (AMD ordering + elimination tree) is never redone.

Keeping this alongside the numpy host path is intentional: sparsax shines on
GPU, while host KLU remains competitive on CPU.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from ._core import _KRYLOV_DEGREE_DEFAULT, _KRYLOV_DMAX_DEFAULT


def build_flow_pattern(
    Wd: sp.spmatrix,
    Wo: sp.spmatrix,
    Ww: sp.spmatrix,
    N: int,
) -> dict:
    """Build the shared COO pattern of ``I - ρ_d W_d - ρ_o W_o - ρ_w W_w``.

    Returns ``Ai, Aj`` (int32 COO coordinates of the structural union) plus
    four float64 value vectors aligned to that pattern — ``eye_vals`` (1 on
    the diagonal), ``wd_vals``, ``wo_vals``, ``ww_vals`` — such that

    ``Ax(ρ) = eye_vals - ρ_d·wd_vals - ρ_o·wo_vals - ρ_w·ww_vals``

    is exactly ``A(ρ)`` on that pattern.  All four vectors share identical
    ``(Ai, Aj)`` ordering because they are assembled over the same
    concatenated coordinate list and ``sum_duplicates`` sorts deterministically.
    """
    eye = sp.eye(N, format="coo")
    Wd, Wo, Ww = Wd.tocoo(), Wo.tocoo(), Ww.tocoo()
    rows = np.concatenate([eye.row, Wd.row, Wo.row, Ww.row])
    cols = np.concatenate([eye.col, Wd.col, Wo.col, Ww.col])
    nnz = (eye.nnz, Wd.nnz, Wo.nnz, Ww.nnz)

    def _slot(parts: list[np.ndarray]) -> sp.coo_matrix:
        c = sp.coo_matrix((np.concatenate(parts), (rows, cols)), shape=(N, N))
        c.sum_duplicates()
        return c

    z = [np.zeros(m) for m in nnz]
    eye_c = _slot([np.ones(nnz[0]), z[1], z[2], z[3]])
    wd_c = _slot([z[0], Wd.data, z[2], z[3]])
    wo_c = _slot([z[0], z[1], Wo.data, z[3]])
    ww_c = _slot([z[0], z[1], z[2], Ww.data])

    # All four coo's share identical (row, col) after sum_duplicates.
    return {
        "Ai": np.asarray(eye_c.row, dtype=np.int32),
        "Aj": np.asarray(eye_c.col, dtype=np.int32),
        "eye_vals": np.asarray(eye_c.data, dtype=np.float64),
        "wd_vals": np.asarray(wd_c.data, dtype=np.float64),
        "wo_vals": np.asarray(wo_c.data, dtype=np.float64),
        "ww_vals": np.asarray(ww_c.data, dtype=np.float64),
        "N": int(N),
    }


def build_sar_pattern(W: sp.spmatrix, n: int) -> dict:
    """Build the shared COO pattern of ``I - ρW`` (single-ρ reduced-form SAR).

    The single-ρ analogue of :func:`build_flow_pattern`: returns ``Ai, Aj``
    (int32 COO of the structural union of ``I`` and ``W``) plus aligned value
    vectors ``eye_vals`` (1 on the diagonal) and ``w_vals``, so that
    ``Ax(ρ) = eye_vals - ρ·w_vals`` is exactly ``I - ρW`` on that pattern.
    Never densifies ``W``.
    """
    eye = sp.eye(n, format="coo")
    Wc = W.tocoo()
    rows = np.concatenate([eye.row, Wc.row])
    cols = np.concatenate([eye.col, Wc.col])

    def _slot(parts: list[np.ndarray]) -> sp.coo_matrix:
        c = sp.coo_matrix((np.concatenate(parts), (rows, cols)), shape=(n, n))
        c.sum_duplicates()
        return c

    eye_c = _slot([np.ones(eye.nnz), np.zeros(Wc.nnz)])
    w_c = _slot([np.zeros(eye.nnz), Wc.data])
    return {
        "Ai": np.asarray(eye_c.row, dtype=np.int32),
        "Aj": np.asarray(eye_c.col, dtype=np.int32),
        "eye_vals": np.asarray(eye_c.data, dtype=np.float64),
        "w_vals": np.asarray(w_c.data, dtype=np.float64),
        "N": int(n),
    }


def _max_steps(acc, new):
    """Elementwise maximum of two ``(left, right)`` step-out count pairs."""
    import jax.numpy as jnp

    return (jnp.maximum(acc[0], new[0]), jnp.maximum(acc[1], new[1]))


def make_flow_solve(pattern: dict):
    """Build a JIT-compiled ``solve(ρ_d, ρ_o, ρ_w, rhs) -> A(ρ)⁻¹ rhs``.

    Uses sparsax's sparse LU (KLU or UMFPACK, whichever
    :func:`.._utils._sparsax_lu.sparsax_lu` measures faster): the fill-reducing
    analysis is cached by the shared pattern, so each call only rebuilds the
    value vector ``Ax(ρ)``.  ``rhs`` may be a vector ``(N,)`` or matrix
    ``(N, k)`` (batched solve — used for ``X̃ = A⁻¹X``).
    """
    import jax
    import jax.numpy as jnp

    from ..._jax_dispatch import ensure_x64
    from .._utils._sparsax_lu import sparsax_lu

    ensure_x64()

    Ai = jnp.asarray(pattern["Ai"], jnp.int32)
    Aj = jnp.asarray(pattern["Aj"], jnp.int32)
    eye_vals = jnp.asarray(pattern["eye_vals"])
    wd_vals = jnp.asarray(pattern["wd_vals"])
    wo_vals = jnp.asarray(pattern["wo_vals"])
    ww_vals = jnp.asarray(pattern["ww_vals"])
    # Route on ρ = 0.2 in each direction, inside the stable region.
    lu_solve = sparsax_lu(
        Ai, Aj, eye_vals - 0.2 * (wd_vals + wo_vals + ww_vals), pattern["N"]
    ).solve

    @jax.jit
    def solve(rho_d, rho_o, rho_w, rhs):
        Ax = eye_vals - rho_d * wd_vals - rho_o * wo_vals - rho_w * ww_vals
        return lu_solve(Ai, Aj, Ax, rhs)

    return solve


def build_flow_ctx(Wd, Wo, Ww, N) -> dict:
    """Sparse solve context for the unrestricted flow (W never densified).

    Bundles the shared COO pattern (:func:`build_flow_pattern`) and BCOO copies
    of the three lag matrices for sparse matvecs.  The fill-reducing symbolic
    analysis is cached inside sparsax, keyed on the (constant) pattern.
    """
    from jax.experimental import sparse as jsparse

    ctx = build_flow_pattern(Wd.tocsr(), Wo.tocsr(), Ww.tocsr(), N)
    ctx["Wd_bcoo"] = jsparse.BCOO.from_scipy_sparse(Wd.tocsr())
    ctx["Wo_bcoo"] = jsparse.BCOO.from_scipy_sparse(Wo.tocsr())
    ctx["Ww_bcoo"] = jsparse.BCOO.from_scipy_sparse(Ww.tocsr())
    return ctx


def _make_flow_solvers(ctx):
    """Build sparse-LU solve closures for ``A(ρ_d,ρ_o,ρ_w) = I−ρ_dWd−ρ_oWo−ρ_wWw``.

    Returns ``(solve, matvec)`` where ``solve(ρ_d,ρ_o,ρ_w,rhs)`` →
    ``A(ρ)⁻¹ rhs`` via sparsax's sparse LU and ``matvec`` is a dict
    ``{"d","o","w"}`` of sparse (BCOO) lag matvecs.

    The LU is KLU or UMFPACK, whichever :func:`.._utils._sparsax_lu.sparsax_lu`
    measures faster on this pattern.  Both reuse their numeric factorization
    via a content-addressed cache: the m+1 solves of a Krylov basis at a fixed
    (ρ_d,ρ_o,ρ_w) pay one factorization and m cheap solves per chain; see
    ``set_sparsax_lu_cache_size``.
    """
    import jax.numpy as jnp

    from .._utils._sparsax_lu import sparsax_lu

    Ai = jnp.asarray(ctx["Ai"], jnp.int32)
    Aj = jnp.asarray(ctx["Aj"], jnp.int32)
    eye_vals = jnp.asarray(ctx["eye_vals"])
    wd_vals = jnp.asarray(ctx["wd_vals"])
    wo_vals = jnp.asarray(ctx["wo_vals"])
    ww_vals = jnp.asarray(ctx["ww_vals"])
    Wd_bcoo, Wo_bcoo, Ww_bcoo = ctx["Wd_bcoo"], ctx["Wo_bcoo"], ctx["Ww_bcoo"]
    # Route on ρ = 0.2 in each direction, inside the stable region.
    lu_solve = sparsax_lu(
        Ai, Aj, eye_vals - 0.2 * (wd_vals + wo_vals + ww_vals), ctx["N"]
    ).solve

    def solve(rho_d, rho_o, rho_w, rhs):
        Ax = eye_vals - rho_d * wd_vals - rho_o * wo_vals - rho_w * ww_vals
        return lu_solve(Ai, Aj, Ax, rhs)

    matvec = {
        "d": lambda v: Wd_bcoo @ v,
        "o": lambda v: Wo_bcoo @ v,
        "w": lambda v: Ww_bcoo @ v,
    }
    return solve, matvec


def _make_flow_gibbs_step(
    y_jax,
    X_jax,
    ctx,
    n,
    k,
    priors,
    *,
    krylov_degree,
    krylov_dmax,
    positive,
    n_cycles,
    krylov_reuse=True,
):
    """Build a JIT-compiled unrestricted-flow NB Gibbs step (ω → 3×ρ → β → α).

    Reuses the reduced-form cross-section blocks: sampling one ρ_k (holding the
    other two fixed) is ``(A_0 − Δρ_k W_k)⁻¹X``, structurally identical to the
    single-ρ SAR slice, so the same shift-invert Krylov basis + slice sampler
    apply with ``W_k`` as the direction and the current ``A_0`` as the base.
    The joint stability wall ``|ρ_d|+|ρ_o|+|ρ_w| < ρ_upper`` is enforced through
    the per-ρ_k slice bounds.  W is never densified.
    """
    import jax
    import jax.numpy as jnp
    from jax.scipy.linalg import cho_solve, solve_triangular

    from ..._jax_dispatch import ensure_x64
    from .._utils._jax_utils import make_pg_draw
    from ._jax import (
        _build_krylov_basis_jax,
        _sample_alpha_jax_reduced,
        _slice_sample_rho_jax,
    )

    _draw_pg = make_pg_draw()

    ensure_x64()
    solve, matvec = _make_flow_solvers(ctx)

    beta_sigma = priors.beta_sigma
    if np.isscalar(beta_sigma):
        V0_inv_diag = jnp.full(k, 1.0 / (float(beta_sigma) ** 2))
    else:
        V0_inv_diag = 1.0 / jnp.asarray(beta_sigma, dtype=jnp.float64) ** 2
    beta_mu = priors.beta_mu
    mu0 = (
        jnp.full(k, float(beta_mu))
        if np.isscalar(beta_mu)
        else jnp.asarray(beta_mu, dtype=jnp.float64)
    )
    rho_lo = jnp.float64(priors.rho_lower)
    rho_hi = jnp.float64(priors.rho_upper)
    B = jnp.float64(priors.rho_upper)  # joint stability wall bound
    alpha_sigma = jnp.float64(priors.alpha_sigma)
    alpha_nu = jnp.float64(priors.alpha_nu)
    dmax = jnp.float64(krylov_dmax)
    _reuse_threshold = jnp.float64(0.15) if krylov_reuse else jnp.float64(0.0)
    _pos = bool(positive)

    def _wall_bounds(other_abs_sum):
        room = B - other_abs_sum
        lo = jnp.maximum(rho_lo, -room)
        hi = jnp.minimum(rho_hi, room)
        if _pos:
            lo = jnp.maximum(lo, 0.0)
        return lo, hi

    def _draw_omega(y, alpha, eta, key):
        h = jnp.maximum(y + alpha, 1e-3)
        z = jnp.clip(eta - jnp.log(alpha), -20.0, 20.0)
        return _draw_pg(h, z, key)

    def _draw_beta(Xtilde, omega, alpha, key):
        kappa = 0.5 * (y_jax - alpha)
        log_alpha = jnp.log(alpha)
        Xt_omega = Xtilde * omega[:, None]
        Sig_inv = Xt_omega.T @ Xtilde + jnp.diag(V0_inv_diag) + 1e-10 * jnp.eye(k)
        rhs = Xtilde.T @ (kappa + omega * log_alpha) + V0_inv_diag * mu0
        L = jnp.linalg.cholesky(Sig_inv)
        m = cho_solve((L, True), rhs)
        z = jax.random.normal(key, shape=(k,), dtype=jnp.float64)
        return m + solve_triangular(L.T, z, lower=False)

    def _slice_one(
        rho_k,
        rd,
        ro,
        rw,
        wkey,
        other_abs,
        omega,
        alpha,
        slice_width,
        key,
        V_stack_prev,
        rd_basis,
        ro_basis,
        rw_basis,
    ):
        """One ρ_k slice with a W_k-direction basis at the current A_0.

        Candidates inside the Krylov radius are evaluated from the basis; those
        outside it take a direct sparse solve of ``A(ρ)⁻¹X`` with the candidate
        in ρ_k's slot, as in the NumPy sampler.  Rejecting them instead would
        confine each slice to a window around the basis center, a
        state-dependent truncation that does not leave the conditional
        invariant.

        Basis reuse: when all three ρ's are within ``_reuse_threshold`` of
        the basis center, the previous sweep's basis is reused.
        """

        def _rebuild(_):
            V = _build_krylov_basis_jax(
                lambda rhs: solve(rd, ro, rw, rhs),
                X_jax,
                matvec[wkey],
                n,
                k,
                krylov_degree,
            )
            return V, rd, ro, rw

        def _reuse(_):
            return V_stack_prev, rd_basis, ro_basis, rw_basis

        can_reuse = (
            (jnp.abs(rd - rd_basis) < _reuse_threshold)
            & (jnp.abs(ro - ro_basis) < _reuse_threshold)
            & (jnp.abs(rw - rw_basis) < _reuse_threshold)
        )
        V_stack, rd_b, ro_b, rw_b = jax.lax.cond(
            can_reuse,
            _reuse,
            _rebuild,
            operand=None,
        )

        # Horner origin must be the center V_stack was *built* at, not the
        # current ρ_k: on the reuse branch they differ by up to
        # ``_reuse_threshold``, and measuring Δρ from ρ_k would evaluate
        # U at ρ_basis + (ρ − ρ_k) instead of at ρ.  ``wkey`` is a static
        # Python string, so this selects at trace time.
        rho_basis_k = {"d": rd_b, "o": ro_b, "w": rw_b}[wkey]

        def _with_candidate(v):
            """(ρ_d, ρ_o, ρ_w) with the candidate ``v`` in ρ_k's slot."""
            return {"d": (v, ro, rw), "o": (rd, v, rw), "w": (rd, ro, v)}[wkey]

        lo, hi = _wall_bounds(other_abs)

        rho_new, steps_left, steps_right = _slice_sample_rho_jax(
            rho_current=rho_k,
            V_stack=V_stack,
            rho_basis=rho_basis_k,
            omega=omega,
            y_jax=y_jax,
            alpha=alpha,
            V0_inv_diag=V0_inv_diag,
            mu0=mu0,
            intercept_col=-1,
            rho_lower=lo,
            rho_upper=hi,
            krylov_dmax=dmax,
            slice_width=slice_width,
            key=key,
            X_jax=X_jax,
            solve_at=lambda v, rhs: solve(*_with_candidate(v), rhs),
            return_steps=True,
        )
        return rho_new, (steps_left, steps_right), V_stack, rd_b, ro_b, rw_b

    @jax.jit
    def gibbs_step(state, key, slice_widths):
        """One sweep; ``slice_widths`` holds the ρ_d, ρ_o and ρ_w slice widths.

        Returns ``(new_state, η, steps)``, where ``steps`` holds each ρ slice's
        ``(left, right)`` step-out counts (the maximum over the sweep's cycles)
        for warmup width adaptation.
        """
        w_d, w_o, w_w = slice_widths
        beta = state["beta"]
        rd, ro, rw = state["rho_d"], state["rho_o"], state["rho_w"]
        alpha = state["alpha"]

        # Basis caches from previous sweep (for reuse)
        Vd_prev = state["V_stack_d"]
        Vo_prev = state["V_stack_o"]
        Vw_prev = state["V_stack_w"]
        rd_b_prev = state["rd_basis"]
        ro_b_prev = state["ro_basis"]
        rw_b_prev = state["rw_basis"]

        eta = solve(rd, ro, rw, X_jax @ beta)
        key, kpg = jax.random.split(key)
        omega = _draw_omega(y_jax, alpha, eta, kpg)

        no_steps = (jnp.float64(0.0), jnp.float64(0.0))
        steps_d = steps_o = steps_w = no_steps
        for cyc in range(n_cycles):
            key, kd, ko, kw, kb = jax.random.split(key, 5)
            rd, s_d, Vd, rd_b, ro_b_d, rw_b_d = _slice_one(
                rd,
                rd,
                ro,
                rw,
                "d",
                jnp.abs(ro) + jnp.abs(rw),
                omega,
                alpha,
                w_d,
                kd,
                V_stack_prev=Vd_prev,
                rd_basis=rd_b_prev,
                ro_basis=ro_b_prev,
                rw_basis=rw_b_prev,
            )
            steps_d = _max_steps(steps_d, s_d)
            ro, s_o, Vo, rd_b_o, ro_b, rw_b_o = _slice_one(
                ro,
                rd,
                ro,
                rw,
                "o",
                jnp.abs(rd) + jnp.abs(rw),
                omega,
                alpha,
                w_o,
                ko,
                V_stack_prev=Vo_prev,
                rd_basis=rd_b_prev,
                ro_basis=ro_b_prev,
                rw_basis=rw_b_prev,
            )
            steps_o = _max_steps(steps_o, s_o)
            rw, s_w, Vw, rd_b_w, ro_b_w, rw_b = _slice_one(
                rw,
                rd,
                ro,
                rw,
                "w",
                jnp.abs(rd) + jnp.abs(ro),
                omega,
                alpha,
                w_w,
                kw,
                V_stack_prev=Vw_prev,
                rd_basis=rd_b_prev,
                ro_basis=ro_b_prev,
                rw_basis=rw_b_prev,
            )
            steps_w = _max_steps(steps_w, s_w)

            # β step needs X̃ = A(ρ_new)⁻¹X at the just-updated (ρ_d,ρ_o,ρ_w);
            # all three moved, so no single basis covers it — one direct solve.
            Xtilde = solve(rd, ro, rw, X_jax)
            beta = _draw_beta(Xtilde, omega, alpha, kb)
            eta = Xtilde @ beta
            if cyc < n_cycles - 1:
                key, kpg2 = jax.random.split(key)
                omega = _draw_omega(y_jax, alpha, eta, kpg2)

        key, ka = jax.random.split(key)
        alpha = _sample_alpha_jax_reduced(eta, y_jax, alpha, alpha_sigma, alpha_nu, ka)

        # Return the fitted latent η so the runner forms the pointwise NB
        # log-likelihood on-device (reusing the sweep's solve) instead of a
        # post-hoc per-draw host-solve loop.
        return (
            {
                "beta": beta,
                "rho_d": rd,
                "rho_o": ro,
                "rho_w": rw,
                "alpha": alpha,
                "omega": omega,
                "V_stack_d": Vd,
                "V_stack_o": Vo,
                "V_stack_w": Vw,
                "rd_basis": rd_b,
                "ro_basis": ro_b,
                "rw_basis": rw_b,
            },
            eta,
            (steps_d, steps_o, steps_w),
        )

    return gibbs_step


def run_chains_jax_flow(
    y,
    X,
    Wd,
    Wo,
    Ww,
    priors,
    inits,
    draws,
    tune,
    *,
    thin=1,
    krylov_degree=_KRYLOV_DEGREE_DEFAULT,
    krylov_dmax=_KRYLOV_DMAX_DEFAULT,
    positive=False,
    n_cycles=1,
    jax_seeds=None,
    progressbar=False,
    slice_width=0.4,
    krylov_reuse=True,
):
    """Run the unrestricted flow NB Gibbs sampler on the JAX backend.

    Chains run in parallel threads (see
    :func:`.._utils._jax_utils.run_chains_chunked`).  The non-symmetric LU solve
    goes through sparsax (KLU or UMFPACK, whichever is faster on the pattern)
    with numeric factor reuse.  ``W`` is never densified.  Each ρ slice starts
    at ``slice_width`` and adapts its own width during warmup.

    Returns one dict per chain with keys ``rho_d``, ``rho_o``, ``rho_w``,
    ``beta``, ``alpha``, ``log_lik``.
    """
    import jax
    import jax.numpy as jnp
    from scipy.special import gammaln

    from ..._jax_dispatch import ensure_x64

    ensure_x64()
    N, k = X.shape
    y_jax = jnp.asarray(y, dtype=jnp.float64)
    X_jax = jnp.asarray(X, dtype=jnp.float64)
    ctx = build_flow_ctx(Wd, Wo, Ww, N)

    gibbs_step = _make_flow_gibbs_step(
        y_jax,
        X_jax,
        ctx,
        N,
        k,
        priors,
        krylov_degree=krylov_degree,
        krylov_dmax=krylov_dmax,
        positive=positive,
        n_cycles=n_cycles,
        krylov_reuse=krylov_reuse,
    )

    chains = len(inits)
    if jax_seeds is None:
        jax_seeds = list(range(chains))

    # sparsax's LU factor cache must hold each chain's distinct factors live
    # across the sweep's several solves (η, the 3 directional bases, X̃).
    from .._utils._sparsax_lu import set_sparsax_lu_cache_size

    set_sparsax_lu_cache_size(max(32, 8 * chains))

    _V_init = jnp.zeros((krylov_degree + 1, N, k), dtype=jnp.float64)
    states = [
        {
            "beta": jnp.asarray(i.beta, dtype=jnp.float64),
            "rho_d": jnp.float64(float(i.rho_d)),
            "rho_o": jnp.float64(float(i.rho_o)),
            "rho_w": jnp.float64(float(i.rho_w if i.rho_w is not None else 0.0)),
            "alpha": jnp.float64(float(i.alpha)),
            "omega": jnp.asarray(i.omega, dtype=jnp.float64),
            "V_stack_d": _V_init,
            "V_stack_o": _V_init,
            "V_stack_w": _V_init,
            "rd_basis": jnp.float64(0.0),
            "ro_basis": jnp.float64(0.0),
            "rw_basis": jnp.float64(0.0),
            "slice_widths": (jnp.float64(slice_width),) * 3,
        }
        for i in inits
    ]
    warm_keys = [jax.random.PRNGKey(int(s)) for s in jax_seeds]
    draw_keys = [jax.random.fold_in(jax.random.PRNGKey(int(s)), 1) for s in jax_seeds]

    from .._utils._jax_slice import adapt_slice_width
    from .._utils._jax_utils import run_chains_chunked

    def _sweep(st, key, tuning):
        widths = st["slice_widths"]
        core = {name: v for name, v in st.items() if name != "slice_widths"}
        core, eta, steps = gibbs_step(core, key, widths)
        widths = tuple(
            adapt_slice_width(w, left, right, tuning)
            for w, (left, right) in zip(widths, steps)
        )
        trace = (core["rho_d"], core["rho_o"], core["rho_w"])
        trace += (core["beta"], core["alpha"], eta)
        return dict(core, slice_widths=widths), trace

    _, (rd_all, ro_all, rw_all, beta_all, alpha_all, eta_all) = run_chains_chunked(
        _sweep, states, warm_keys, draw_keys, tune=tune, draws=draws
    )
    sl = slice(None, None, thin) if thin > 1 else slice(None)
    rd_all = rd_all[:, sl]
    ro_all = ro_all[:, sl]
    rw_all = rw_all[:, sl]
    beta_all = beta_all[:, sl]
    alpha_all = alpha_all[:, sl]
    eta_all = eta_all[:, sl]  # (chains, n_keep, N)

    # Pointwise NB log-likelihood from the fitted η collected during sampling —
    # no post-hoc per-draw solves.
    y_np = np.asarray(y, dtype=np.float64)
    results = []
    for c in range(chains):
        alpha_s = alpha_all[c]
        mu = np.exp(np.clip(eta_all[c], -30.0, 30.0))  # (n_keep, N)
        a = alpha_s[:, None]
        log_lik = (
            gammaln(y_np + a)
            - gammaln(a)
            - gammaln(y_np + 1.0)
            + y_np * np.log(np.maximum(mu / (mu + a), 1e-300))
            + a * np.log(np.maximum(a / (mu + a), 1e-300))
        )
        results.append(
            {
                "rho_d": rd_all[c],
                "rho_o": ro_all[c],
                "rho_w": rw_all[c],
                "beta": beta_all[c],
                "alpha": alpha_s,
                "log_lik": log_lik,
            }
        )
    return results


# ---------------------------------------------------------------------------
# Separable flow NB Gibbs step (2-ρ Kronecker, JAX)
# ---------------------------------------------------------------------------


def _build_sar_solver_jax(W_csc, n):
    """Build a sparsax-based n×n solver for ``L(ρ) = I − ρW``.

    Returns ``solve(rho, rhs)`` where ``rhs`` is ``(n,)`` or ``(n, m)``.
    The symbolic analysis is cached by sparsax keyed on the constant
    COO pattern, so only the numeric factorization is redone per ρ.  The LU is
    KLU or UMFPACK, whichever :func:`.._utils._sparsax_lu.sparsax_lu` measures
    faster on the pattern.
    """
    import jax.numpy as jnp

    from .._utils._sparsax_lu import sparsax_lu

    pat = build_sar_pattern(W_csc.tocsr(), n)
    Ai = jnp.asarray(pat["Ai"], jnp.int32)
    Aj = jnp.asarray(pat["Aj"], jnp.int32)
    eye_vals = jnp.asarray(pat["eye_vals"])
    w_vals = jnp.asarray(pat["w_vals"])
    lu_solve = sparsax_lu(Ai, Aj, eye_vals - 0.5 * w_vals, n).solve

    def solve(rho, rhs):
        Ax = eye_vals - rho * w_vals
        return lu_solve(Ai, Aj, Ax, rhs)

    return solve


def _kron_solve_jax(solve_Ld, solve_Lo, B, n):
    """JAX Kronecker solve: ``(L_o ⊗ L_d) X = B`` via two n×n solves.

    Mirrors :func:`neighbayes._ops._kron_solve.kron_solve_matrix` but
    with JAX arrays and sparsax solves.  ``B`` is ``(N, k)`` where
    ``N = n²``.

    Uses the vec-permutation identity:
    ``(L_o ⊗ L_d) vec(H) = vec(L_d H L_o^T)``.

    Every reshape here is Fortran-order and every axis permutation matters.
    This is a verbatim port of the NumPy reference and is pinned against a
    dense ``inv(L_o ⊗ L_d)``.  Two earlier deviations from it — a C-order
    ``reshape(n, k, n)`` and a missing ``transpose(1, 0, 2)`` before the
    final flatten — made this return ``vec(Hᵀ)`` rather than ``vec(H)``,
    silently fitting a permuted system.
    """
    k = B.shape[1]
    # R = B reshaped as (n, n*k) column-major (Fortran order)
    R = B.reshape(n, n * k, order="F")
    # Step 1: L_d^{-1} R  →  (n, n*k)
    Hp = solve_Ld(R)
    # (n, n, k) Fortran → (k, n, n) → (k*n, n) → transpose to the (n, k*n) RHS.
    Hp3 = Hp.reshape(n, n, k, order="F")
    RHS2 = Hp3.transpose(2, 0, 1).reshape(k * n, n).T  # (n, k*n)
    # Step 2: solve L_o Z = RHS2.
    Z_h = solve_Lo(RHS2)  # (n, k*n)
    # Back to (n, n, k) Fortran, then swap the two n axes before flattening.
    Z3 = Z_h.reshape(n, n, k, order="F")
    return Z3.transpose(1, 0, 2).reshape(n * n, k, order="F")


# Axis 0 of the (n, n, k) Fortran view is the *destination* index and axis 1
# the *origin* index, so with ``(B ⊗ A) vec(H) = vec(A H Bᵀ)`` the left factor
# acts on axis 0 and the right factor on axis 1.
#
# These two operators have been wrong twice.  They are pinned end-to-end
# against a dense perturbed solve by ``test_kron_krylov_series_matches_dense``
# — an earlier version applied ``Wᵀ`` where ``W`` belongs and put ``L_d`` on
# the wrong axis, computing ``(L_oᵀ ⊗ W)`` and ``(L_d ⊗ Wᵀ)``.  A wrong matvec
# does not raise; it silently distorts the ρ conditional.  **Do not adjust
# these by reasoning — change them only against that test.**


def kron_matvec_d_jax(W_bcoo, rho_o, v, n, k):
    """``(L_o ⊗ W) v = vec(W H L_oᵀ)`` where ``v = vec(H)``, ``L_o = I − ρ_o W``."""
    N = n * n
    v3 = v.reshape(n, n, k, order="F")
    # W on axis 0 (destination): W H
    Wv3 = (W_bcoo @ v3.reshape(n, -1, order="F")).reshape(n, n, k, order="F")
    # L_oᵀ on axis 1 (origin): right-multiplication by L_oᵀ is, on the
    # axis-swapped view, left-multiplication by L_o = I − ρ_o W.
    T = Wv3.transpose(1, 0, 2).reshape(n, -1, order="F")
    out = T - rho_o * (W_bcoo @ T)
    return out.reshape(n, n, k, order="F").transpose(1, 0, 2).reshape(N, k, order="F")


def kron_matvec_o_jax(W_bcoo, rho_d, v, n, k):
    """``(W ⊗ L_d) v = vec(L_d H Wᵀ)`` where ``v = vec(H)``, ``L_d = I − ρ_d W``."""
    N = n * n
    v3 = v.reshape(n, n, k, order="F")
    # L_d on axis 0 (destination): L_d H
    flat = v3.reshape(n, -1, order="F")
    Ldv3 = (flat - rho_d * (W_bcoo @ flat)).reshape(n, n, k, order="F")
    # Wᵀ on axis 1 (origin): right-multiplication by Wᵀ is, on the
    # axis-swapped view, left-multiplication by W.
    T = Ldv3.transpose(1, 0, 2).reshape(n, -1, order="F")
    out = W_bcoo @ T
    return out.reshape(n, n, k, order="F").transpose(1, 0, 2).reshape(N, k, order="F")


def _make_flow_sep_gibbs_step(
    y_jax,
    X_jax,
    W_csc,
    n,
    N,
    k,
    priors,
    *,
    krylov_degree,
    krylov_dmax,
    krylov_reuse=True,
    n_cycles=1,
):
    """Build a JIT-compiled separable-flow NB Gibbs step (ω → ρ_d → ρ_o → β → α).

    The separable model factors ``A = L_o ⊗ L_d`` where
    ``L_k = I_n − ρ_k W`` (both n×n).  Each ρ_k slice uses a
    **Kronecker-aware Krylov basis**: the basis is built on the N×N
    Kronecker system via two n×n solves per basis vector, and the
    matvec ``(L_o ⊗ W) v`` (for ρ_d) or ``(W ⊗ L_d) v`` (for ρ_o)
    is computed via reshapes + n×n sparse matvecs.

    Basis reuse: each ρ_k's Krylov basis is carried in the state dict
    and reused via ``jax.lax.cond`` when ``|Δρ_k| < threshold``.
    """
    import jax
    import jax.numpy as jnp
    from jax.experimental import sparse as jsparse
    from jax.scipy.linalg import cho_solve, solve_triangular

    from .._utils._jax_utils import make_pg_draw
    from ._jax import (
        _eval_U_from_basis_jax,
        _sample_alpha_jax_reduced,
        _slice_sample_rho_jax,
    )

    _draw_pg = make_pg_draw()

    # n×n sparsax solvers for L_d and L_o
    solve_Ld = _build_sar_solver_jax(W_csc, n)
    solve_Lo = _build_sar_solver_jax(W_csc, n)

    # BCOO W for the Kronecker matvecs (Wᵀ is not needed: both directions
    # reduce to left-multiplication by W on an axis-swapped view).
    W_bcoo = jsparse.BCOO.from_scipy_sparse(W_csc.tocsr())

    # Priors
    beta_sigma = priors.beta_sigma
    if np.isscalar(beta_sigma):
        V0_inv_diag = jnp.full(k, 1.0 / (float(beta_sigma) ** 2))
    else:
        V0_inv_diag = 1.0 / jnp.asarray(beta_sigma, dtype=jnp.float64) ** 2
    beta_mu = priors.beta_mu
    mu0 = (
        jnp.full(k, float(beta_mu))
        if np.isscalar(beta_mu)
        else jnp.asarray(beta_mu, dtype=jnp.float64)
    )
    rho_lo = jnp.float64(priors.rho_lower)
    rho_hi = jnp.float64(priors.rho_upper)
    alpha_sigma = jnp.float64(priors.alpha_sigma)
    alpha_nu = jnp.float64(priors.alpha_nu)
    dmax = jnp.float64(krylov_dmax)
    _reuse_threshold = jnp.float64(0.15) if krylov_reuse else jnp.float64(0.0)
    _deg = int(krylov_degree)
    _V_init = jnp.zeros((_deg + 1, N, k), dtype=jnp.float64)

    def _draw_omega(y, alpha, eta, key):
        h = jnp.maximum(y + alpha, 1e-3)
        z = jnp.clip(eta - jnp.log(alpha), -20.0, 20.0)
        return _draw_pg(h, z, key)

    def _draw_beta(Xtilde, omega, alpha, key):
        kappa = 0.5 * (y_jax - alpha)
        log_alpha = jnp.log(alpha)
        Xt_omega = Xtilde * omega[:, None]
        Sig_inv = Xt_omega.T @ Xtilde + jnp.diag(V0_inv_diag) + 1e-10 * jnp.eye(k)
        rhs = Xtilde.T @ (kappa + omega * log_alpha) + V0_inv_diag * mu0
        L = jnp.linalg.cholesky(Sig_inv)
        m = cho_solve((L, True), rhs)
        z = jax.random.normal(key, shape=(k,), dtype=jnp.float64)
        return m + solve_triangular(L.T, z, lower=False)

    def _kron_solve(rd, ro, B):
        return _kron_solve_jax(
            lambda rhs: solve_Ld(rd, rhs),
            lambda rhs: solve_Lo(ro, rhs),
            B,
            n,
        )

    def _kron_krylov_basis(rd, ro, direction):
        """Build a Kronecker-aware Krylov basis at (rd, ro).

        For direction='rho_d': matvec is (L_o ⊗ W) v = vec(W H L_o^T).
        For direction='rho_o': matvec is (W ⊗ L_d) v = vec(L_d H W^T).
        """
        m = _deg

        def _kron_solve_rhs(rhs):
            return _kron_solve(rd, ro, rhs)

        # V_0 = A_c^{-1} X
        V0 = _kron_solve_rhs(X_jax)

        def _matvec_d(v):
            return kron_matvec_d_jax(W_bcoo, ro, v, n, k)

        def _matvec_o(v):
            return kron_matvec_o_jax(W_bcoo, rd, v, n, k)

        matvec_fn = _matvec_d if direction == "rho_d" else _matvec_o

        # Build basis: V_{j+1} = A_c^{-1} (matvec V_j)
        # Use lax.scan for the loop
        def _scan_body(carry, _):
            V_j = carry
            Wv = matvec_fn(V_j)
            V_next = _kron_solve_rhs(Wv)
            return V_next, V_next

        _, V_rest = jax.lax.scan(_scan_body, V0, jnp.arange(m))
        V_stack = jnp.concatenate([V0[None], V_rest], axis=0)

        rho_basis_val = rd if direction == "rho_d" else ro
        return V_stack, rho_basis_val

    def _eval_kron_krylov(V_stack, drho):
        """Horner evaluation: U ≈ Σ (Δρ)^j V_j."""
        return _eval_U_from_basis_jax(V_stack, drho)

    def _slice_rho_k(
        rho_k,
        direction,
        rd,
        ro,
        omega,
        alpha,
        V_stack_prev,
        rho_basis_prev,
        slice_width,
        key,
    ):
        """One ρ_k slice with Kronecker Krylov basis + reuse.

        Candidates outside the Krylov radius take a direct Kronecker solve with
        the candidate in ρ_k's slot, as in the NumPy sampler, rather than being
        rejected (which would truncate the slice to a state-dependent window).
        """

        def _solve_at(v, rhs):
            if direction == "rho_d":
                return _kron_solve(v, ro, rhs)
            return _kron_solve(rd, v, rhs)

        def _rebuild(_):
            V, rb = _kron_krylov_basis(rd, ro, direction)
            return V, rb

        def _reuse(_):
            return V_stack_prev, rho_basis_prev

        V_stack, rho_basis = jax.lax.cond(
            jnp.abs(rho_k - rho_basis_prev) < _reuse_threshold,
            _reuse,
            _rebuild,
            operand=None,
        )

        rho_new, steps_left, steps_right = _slice_sample_rho_jax(
            rho_current=rho_k,
            V_stack=V_stack,
            rho_basis=rho_basis,
            omega=omega,
            y_jax=y_jax,
            alpha=alpha,
            V0_inv_diag=V0_inv_diag,
            mu0=mu0,
            intercept_col=-1,
            rho_lower=rho_lo,
            rho_upper=rho_hi,
            krylov_dmax=dmax,
            slice_width=slice_width,
            key=key,
            X_jax=X_jax,
            solve_at=_solve_at,
            return_steps=True,
        )
        return rho_new, (steps_left, steps_right), V_stack, rho_basis

    @jax.jit
    def gibbs_step(state, key, slice_widths):
        """One sweep; ``slice_widths`` holds the ρ_d and ρ_o slice widths.

        Returns ``(new_state, η, steps)``, where ``steps`` holds each ρ slice's
        ``(left, right)`` step-out counts (the maximum over the sweep's cycles)
        for warmup width adaptation.
        """
        w_d, w_o = slice_widths
        beta = state["beta"]
        rd, ro = state["rho_d"], state["rho_o"]
        alpha = state["alpha"]
        Vd_prev = state["V_stack_d"]
        Vo_prev = state["V_stack_o"]
        rd_basis_prev = state["rho_basis_d"]
        ro_basis_prev = state["rho_basis_o"]

        # η = A⁻¹ Xβ via Kronecker solve
        eta = _kron_solve(rd, ro, X_jax @ beta.reshape(-1, 1)).ravel()

        key, kpg = jax.random.split(key)
        omega = _draw_omega(y_jax, alpha, eta, kpg)

        no_steps = (jnp.float64(0.0), jnp.float64(0.0))
        steps_d = steps_o = no_steps
        for cyc in range(n_cycles):
            key, kd, ko, kb = jax.random.split(key, 4)

            rd, s_d, Vd, rd_basis = _slice_rho_k(
                rd,
                "rho_d",
                rd,
                ro,
                omega,
                alpha,
                Vd_prev,
                rd_basis_prev,
                w_d,
                kd,
            )
            steps_d = _max_steps(steps_d, s_d)
            Vd_prev = Vd
            rd_basis_prev = rd_basis

            ro, s_o, Vo, ro_basis = _slice_rho_k(
                ro,
                "rho_o",
                rd,
                ro,
                omega,
                alpha,
                Vo_prev,
                ro_basis_prev,
                w_o,
                ko,
            )
            steps_o = _max_steps(steps_o, s_o)
            Vo_prev = Vo
            ro_basis_prev = ro_basis

            # β step: X̃ = (L_o ⊗ L_d)⁻¹ X via Kronecker solve
            Xtilde = _kron_solve(rd, ro, X_jax)
            beta = _draw_beta(Xtilde, omega, alpha, kb)
            eta = Xtilde @ beta
            if cyc < n_cycles - 1:
                key, kpg2 = jax.random.split(key)
                omega = _draw_omega(y_jax, alpha, eta, kpg2)

        key, ka = jax.random.split(key)
        alpha = _sample_alpha_jax_reduced(eta, y_jax, alpha, alpha_sigma, alpha_nu, ka)

        return (
            {
                "beta": beta,
                "rho_d": rd,
                "rho_o": ro,
                "alpha": alpha,
                "omega": omega,
                "V_stack_d": Vd_prev,
                "rho_basis_d": rd_basis_prev,
                "V_stack_o": Vo_prev,
                "rho_basis_o": ro_basis_prev,
            },
            eta,
            (steps_d, steps_o),
        )

    return gibbs_step


def run_chains_jax_flow_separable(
    y,
    X,
    W_csc,
    n,
    priors,
    inits,
    draws,
    tune,
    *,
    thin=1,
    krylov_degree=_KRYLOV_DEGREE_DEFAULT,
    krylov_dmax=_KRYLOV_DMAX_DEFAULT,
    krylov_reuse=True,
    n_cycles=1,
    jax_seeds=None,
    progressbar=False,
    slice_width=0.4,
):
    """Run the separable flow NB Gibbs sampler on the JAX backend.

    The separable Kronecker model (``ρ_w = -ρ_d·ρ_o``) factors the ``N×N``
    system into two ``n×n`` solves, each using a sparsax LU factorization
    on the regional weights pattern.  Each ρ_k slice uses a Krylov basis
    on the n×n system with cross-sweep reuse via ``jax.lax.cond``.

    Returns one dict per chain with keys ``rho_d``, ``rho_o``, ``rho_w``,
    ``beta``, ``alpha``, ``log_lik``.
    """
    import jax
    import jax.numpy as jnp
    from scipy.special import gammaln

    from ..._jax_dispatch import ensure_x64

    ensure_x64()
    N, k = X.shape
    y_jax = jnp.asarray(y, dtype=jnp.float64)
    X_jax = jnp.asarray(X, dtype=jnp.float64)

    gibbs_step = _make_flow_sep_gibbs_step(
        y_jax,
        X_jax,
        W_csc,
        n,
        N,
        k,
        priors,
        krylov_degree=krylov_degree,
        krylov_dmax=krylov_dmax,
        krylov_reuse=krylov_reuse,
        n_cycles=n_cycles,
    )

    chains = len(inits)
    if jax_seeds is None:
        jax_seeds = list(range(chains))

    from .._utils._sparsax_lu import set_sparsax_lu_cache_size

    set_sparsax_lu_cache_size(max(32, 8 * chains))

    _V_init = jnp.zeros((krylov_degree + 1, n * n, k), dtype=jnp.float64)

    states = [
        {
            "beta": jnp.asarray(i.beta, dtype=jnp.float64),
            "rho_d": jnp.float64(float(i.rho_d)),
            "rho_o": jnp.float64(float(i.rho_o)),
            "alpha": jnp.float64(float(i.alpha)),
            "omega": jnp.asarray(i.omega, dtype=jnp.float64),
            "V_stack_d": _V_init,
            "rho_basis_d": jnp.float64(0.0),
            "V_stack_o": _V_init,
            "rho_basis_o": jnp.float64(0.0),
            "slice_widths": (jnp.float64(slice_width),) * 2,
        }
        for i in inits
    ]
    warm_keys = [jax.random.PRNGKey(int(s)) for s in jax_seeds]
    draw_keys = [jax.random.fold_in(jax.random.PRNGKey(int(s)), 1) for s in jax_seeds]

    from .._utils._jax_slice import adapt_slice_width
    from .._utils._jax_utils import run_chains_chunked

    def _sweep(st, key, tuning):
        widths = st["slice_widths"]
        core = {name: v for name, v in st.items() if name != "slice_widths"}
        core, eta, steps = gibbs_step(core, key, widths)
        widths = tuple(
            adapt_slice_width(w, left, right, tuning)
            for w, (left, right) in zip(widths, steps)
        )
        trace = (core["rho_d"], core["rho_o"], core["beta"], core["alpha"], eta)
        return dict(core, slice_widths=widths), trace

    _, (rd_all, ro_all, beta_all, alpha_all, eta_all) = run_chains_chunked(
        _sweep, states, warm_keys, draw_keys, tune=tune, draws=draws
    )
    sl = slice(None, None, thin) if thin > 1 else slice(None)
    rd_all = rd_all[:, sl]
    ro_all = ro_all[:, sl]
    beta_all = beta_all[:, sl]
    alpha_all = alpha_all[:, sl]
    eta_all = eta_all[:, sl]
    rw_all = -rd_all * ro_all  # separable constraint

    y_np = np.asarray(y, dtype=np.float64)
    results = []
    for c in range(chains):
        alpha_s = alpha_all[c]
        mu = np.exp(np.clip(eta_all[c], -30.0, 30.0))
        a = alpha_s[:, None]
        log_lik = (
            gammaln(y_np + a)
            - gammaln(a)
            - gammaln(y_np + 1.0)
            + y_np * np.log(np.maximum(mu / (mu + a), 1e-300))
            + a * np.log(np.maximum(a / (mu + a), 1e-300))
        )
        results.append(
            {
                "rho_d": rd_all[c],
                "rho_o": ro_all[c],
                "rho_w": rw_all[c],
                "beta": beta_all[c],
                "alpha": alpha_s,
                "log_lik": log_lik,
            }
        )
    return results
