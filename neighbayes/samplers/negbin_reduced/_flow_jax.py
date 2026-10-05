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


def _flow_data(y, X, ctx, priors, krylov_dmax):
    """Host side of the unrestricted flow sweep: ``(lu_solve, data)``.

    ``lu_solve`` is sparsax's KLU or UMFPACK solve, as
    :func:`.._utils._sparsax_lu.sparsax_lu` routes the pattern (a
    module-level function, so the sweep keeps a stable identity); ``data`` is
    every data-dependent array and scalar, which the sweep takes as an
    argument.  The LU reuses its numeric factorization via a content-addressed
    cache: the m+1 solves of a Krylov basis at a fixed (ρ_d,ρ_o,ρ_w) pay one
    factorization and m cheap solves per chain; see ``set_sparsax_lu_cache_size``.
    """
    import jax.numpy as jnp

    from .._utils._sparsax_lu import sparsax_lu

    k = X.shape[1]
    arrays = {
        "Ai": jnp.asarray(ctx["Ai"], jnp.int32),
        "Aj": jnp.asarray(ctx["Aj"], jnp.int32),
        "eye": jnp.asarray(ctx["eye_vals"]),
        "wd": jnp.asarray(ctx["wd_vals"]),
        "wo": jnp.asarray(ctx["wo_vals"]),
        "ww": jnp.asarray(ctx["ww_vals"]),
    }
    # Route on ρ = 0.2 in each direction, inside the stable region.
    lu_solve = sparsax_lu(arrays["Ai"], arrays["Aj"], ctx["N"]).solve
    beta_sigma = np.broadcast_to(np.asarray(priors.beta_sigma, dtype=np.float64), (k,))
    beta_mu = np.broadcast_to(np.asarray(priors.beta_mu, dtype=np.float64), (k,))
    alpha_fixed = getattr(priors, "alpha_fixed", None)
    data = {
        "y": jnp.asarray(y, dtype=jnp.float64),
        "X": jnp.asarray(X, dtype=jnp.float64),
        **arrays,
        "Wd": ctx["Wd_bcoo"],
        "Wo": ctx["Wo_bcoo"],
        "Ww": ctx["Ww_bcoo"],
        "V0_inv_diag": jnp.asarray(1.0 / beta_sigma**2),
        "mu0": jnp.asarray(beta_mu),
        "rho_lo": jnp.float64(priors.rho_lower),
        "rho_hi": jnp.float64(priors.rho_upper),
        "alpha_sigma": jnp.float64(priors.alpha_sigma),
        "alpha_nu": jnp.float64(priors.alpha_nu),
        "alpha_fixed": jnp.float64(alpha_fixed if alpha_fixed is not None else 1.0),
        "dmax": jnp.float64(krylov_dmax),
    }
    return lu_solve, data


def _flow_sweep(
    n, k, lu_solve, krylov_degree, positive, n_cycles, has_alpha_fixed, store_log_lik
):
    """The unrestricted-flow NB sweep (ω → 3×ρ → β → α) for one structure.

    ``sweep(state, key, tuning, data) -> (state, trace)``; the trace is
    ``(ρ_d, ρ_o, ρ_w, β, α[, η])``.  Each ρ slice adapts its width in warmup.

    Reuses the reduced-form cross-section blocks: sampling one ρ_k (holding the
    other two fixed) is ``(A_0 − Δρ_k W_k)⁻¹X``, structurally identical to the
    single-ρ SAR slice, so the same shift-invert Krylov basis + slice sampler
    apply with ``W_k`` as the direction and the current ``A_0`` as the base.
    The joint stability wall ``|ρ_d|+|ρ_o|+|ρ_w| < ρ_upper`` is enforced through
    the per-ρ_k slice bounds.  W is never densified.  The sweep closes over
    the structure only, so it compiles once per structure.
    """
    import jax
    import jax.numpy as jnp
    from jax.scipy.linalg import cho_solve, solve_triangular

    from .._utils._jax_slice import adapt_slice_width
    from .._utils._jax_utils import make_pg_draw
    from ._jax import (
        _build_krylov_basis_jax,
        _sample_alpha_jax_reduced,
        _slice_sample_rho_jax,
    )

    _draw_pg = make_pg_draw()
    _pos = bool(positive)

    def _solver(data):
        def solve(rho_d, rho_o, rho_w, rhs):
            Ax = (
                data["eye"]
                - rho_d * data["wd"]
                - rho_o * data["wo"]
                - rho_w * data["ww"]
            )
            return lu_solve(data["Ai"], data["Aj"], Ax, rhs)

        return solve

    def _wall_bounds(other_abs_sum, data):
        # ρ_upper is also the joint stability wall's bound.
        room = data["rho_hi"] - other_abs_sum
        lo = jnp.maximum(data["rho_lo"], -room)
        hi = jnp.minimum(data["rho_hi"], room)
        if _pos:
            lo = jnp.maximum(lo, 0.0)
        return lo, hi

    def _draw_omega(y, alpha, eta, key):
        h = jnp.maximum(y + alpha, 1e-3)
        z = jnp.clip(eta - jnp.log(alpha), -20.0, 20.0)
        return _draw_pg(h, z, key)

    def _draw_beta(Xtilde, omega, alpha, key, data):
        V0_inv_diag, mu0 = data["V0_inv_diag"], data["mu0"]
        kappa = 0.5 * (data["y"] - alpha)
        log_alpha = jnp.log(alpha)
        Xt_omega = Xtilde * omega[:, None]
        Sig_inv = Xt_omega.T @ Xtilde + jnp.diag(V0_inv_diag) + 1e-10 * jnp.eye(k)
        rhs = Xtilde.T @ (kappa + omega * log_alpha) + V0_inv_diag * mu0
        L = jnp.linalg.cholesky(Sig_inv)
        m = cho_solve((L, True), rhs)
        z = jax.random.normal(key, shape=(k,), dtype=jnp.float64)
        return m + solve_triangular(L.T, z, lower=False)

    def _slice_one(rho_k, rd, ro, rw, wkey, other_abs, omega, alpha, width, key, data):
        """One ρ_k slice with a W_k-direction basis at the current A_0.

        Candidates inside the Krylov radius are evaluated from the basis; those
        outside it take a direct sparse solve of ``A(ρ)⁻¹X`` with the candidate
        in ρ_k's slot, as in the NumPy sampler.  Rejecting them instead would
        confine each slice to a window around the basis center, a
        state-dependent truncation that does not leave the conditional
        invariant.

        The basis is built at the current (ρ_d, ρ_o, ρ_w) every call: it
        depends on all three, and reusing one evaluated the density at stale
        values of the other two, which biased the posterior.
        """
        solve = _solver(data)
        W_k = data["W" + wkey]
        V_stack = _build_krylov_basis_jax(
            lambda rhs: solve(rd, ro, rw, rhs),
            data["X"],
            lambda v: W_k @ v,
            n,
            k,
            krylov_degree,
        )

        def _with_candidate(v):
            """(ρ_d, ρ_o, ρ_w) with the candidate ``v`` in ρ_k's slot."""
            return {"d": (v, ro, rw), "o": (rd, v, rw), "w": (rd, ro, v)}[wkey]

        lo, hi = _wall_bounds(other_abs, data)
        rho_new, steps_left, steps_right = _slice_sample_rho_jax(
            rho_current=rho_k,
            V_stack=V_stack,
            rho_basis=rho_k,
            omega=omega,
            y_jax=data["y"],
            alpha=alpha,
            V0_inv_diag=data["V0_inv_diag"],
            mu0=data["mu0"],
            intercept_col=-1,
            rho_lower=lo,
            rho_upper=hi,
            krylov_dmax=data["dmax"],
            slice_width=width,
            key=key,
            X_jax=data["X"],
            solve_at=lambda v, rhs: solve(*_with_candidate(v), rhs),
            return_steps=True,
        )
        return rho_new, (steps_left, steps_right)

    def sweep(st, key, tuning, data):
        y, X = data["y"], data["X"]
        solve = _solver(data)
        w_d, w_o, w_w = st["slice_widths"]
        beta = st["beta"]
        rd, ro, rw = st["rho_d"], st["rho_o"], st["rho_w"]
        alpha = st["alpha"]

        eta = solve(rd, ro, rw, X @ beta)
        key, kpg = jax.random.split(key)
        omega = _draw_omega(y, alpha, eta, kpg)

        no_steps = (jnp.float64(0.0), jnp.float64(0.0))
        steps_d = steps_o = steps_w = no_steps
        for cyc in range(n_cycles):
            key, kd, ko, kw, kb = jax.random.split(key, 5)
            rd, s_d = _slice_one(
                rd, rd, ro, rw, "d", jnp.abs(ro) + jnp.abs(rw), omega, alpha, w_d,
                kd, data,
            )  # fmt: skip
            steps_d = _max_steps(steps_d, s_d)
            ro, s_o = _slice_one(
                ro, rd, ro, rw, "o", jnp.abs(rd) + jnp.abs(rw), omega, alpha, w_o,
                ko, data,
            )  # fmt: skip
            steps_o = _max_steps(steps_o, s_o)
            rw, s_w = _slice_one(
                rw, rd, ro, rw, "w", jnp.abs(rd) + jnp.abs(ro), omega, alpha, w_w,
                kw, data,
            )  # fmt: skip
            steps_w = _max_steps(steps_w, s_w)

            # β step needs X̃ = A(ρ_new)⁻¹X at the just-updated (ρ_d,ρ_o,ρ_w);
            # all three moved, so no single basis covers it — one direct solve.
            Xtilde = solve(rd, ro, rw, X)
            beta = _draw_beta(Xtilde, omega, alpha, kb, data)
            eta = Xtilde @ beta
            if cyc < n_cycles - 1:
                key, kpg2 = jax.random.split(key)
                omega = _draw_omega(y, alpha, eta, kpg2)

        key, ka = jax.random.split(key)
        if not has_alpha_fixed:
            alpha = _sample_alpha_jax_reduced(
                eta, y, alpha, data["alpha_sigma"], data["alpha_nu"], ka
            )
        else:
            alpha = data["alpha_fixed"]

        widths = tuple(
            adapt_slice_width(w, left, right, tuning)
            for w, (left, right) in zip(
                (w_d, w_o, w_w), (steps_d, steps_o, steps_w), strict=True
            )
        )
        new = {
            "beta": beta,
            "rho_d": rd,
            "rho_o": ro,
            "rho_w": rw,
            "alpha": alpha,
            "omega": omega,
            "slice_widths": widths,
        }
        trace = (rd, ro, rw, beta, alpha)
        # The fitted latent η, traced only so the runner forms the pointwise
        # NB log-likelihood from it rather than re-solving per draw.
        if store_log_lik:
            trace += (eta,)
        return new, trace

    return sweep


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
    store_log_lik=True,
):
    """Run the unrestricted flow NB Gibbs sampler on the JAX backend.

    Chains run in parallel threads (see
    :func:`.._utils._jax_utils.run_chains_chunked`).  The non-symmetric LU solve
    goes through sparsax (KLU or UMFPACK, routed by the pattern)
    with numeric factor reuse.  ``W`` is never densified.  Each ρ slice starts
    at ``slice_width`` and adapts its own width during warmup.

    Returns one dict per chain with keys ``rho_d``, ``rho_o``, ``rho_w``,
    ``beta``, ``alpha``, ``log_lik``.
    """
    import jax
    import jax.numpy as jnp
    from scipy.special import gammaln

    from ..._jax_dispatch import ensure_x64
    from .._utils._jax_utils import cached_sweep, run_chains_chunked
    from .._utils._sparsax_lu import set_sparsax_lu_cache_size

    ensure_x64()
    N, k = X.shape
    ctx = build_flow_ctx(Wd, Wo, Ww, N)
    lu_solve, data = _flow_data(y, X, ctx, priors, krylov_dmax)
    static = (
        N, k, lu_solve, int(krylov_degree), bool(positive), int(n_cycles),
        getattr(priors, "alpha_fixed", None) is not None, bool(store_log_lik),
    )  # fmt: skip
    sweep = cached_sweep(("nb_flow", *static), lambda: _flow_sweep(*static))

    chains = len(inits)
    if jax_seeds is None:
        jax_seeds = list(range(chains))

    # sparsax's LU factor cache must hold each chain's distinct factors live
    # across the sweep's several solves (η, the 3 directional bases, X̃).
    set_sparsax_lu_cache_size(max(32, 8 * chains))

    states = [
        {
            "beta": jnp.asarray(i.beta, dtype=jnp.float64),
            "rho_d": jnp.float64(float(i.rho_d)),
            "rho_o": jnp.float64(float(i.rho_o)),
            "rho_w": jnp.float64(float(i.rho_w if i.rho_w is not None else 0.0)),
            "alpha": jnp.float64(float(i.alpha)),
            "omega": jnp.asarray(i.omega, dtype=jnp.float64),
            "slice_widths": (jnp.float64(slice_width),) * 3,
        }
        for i in inits
    ]
    warm_keys = [jax.random.PRNGKey(int(s)) for s in jax_seeds]
    draw_keys = [jax.random.fold_in(jax.random.PRNGKey(int(s)), 1) for s in jax_seeds]
    _, traces = run_chains_chunked(
        sweep, states, warm_keys, draw_keys, tune=tune, draws=draws, consts=data
    )
    rd_all, ro_all, rw_all, beta_all, alpha_all = traces[:5]
    eta_all = traces[5] if store_log_lik else None
    sl = slice(None, None, thin) if thin > 1 else slice(None)
    rd_all = rd_all[:, sl]
    ro_all = ro_all[:, sl]
    rw_all = rw_all[:, sl]
    beta_all = beta_all[:, sl]
    alpha_all = alpha_all[:, sl]
    if store_log_lik:
        eta_all = eta_all[:, sl]  # (chains, n_keep, N)

    # Pointwise NB log-likelihood from the fitted η collected during sampling —
    # no post-hoc per-draw solves.
    y_np = np.asarray(y, dtype=np.float64)
    results = []
    for c in range(chains):
        alpha_s = alpha_all[c]
        log_lik = None
        if store_log_lik:
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
# n × n filter solver (used by the structured separable sampler)
# ---------------------------------------------------------------------------


def _sar_solver_parts(W_csc, n):
    """Host side of the ``I − ρW`` solve: ``(lu_solve, arrays)``.

    ``lu_solve`` is sparsax's KLU or UMFPACK solve, as
    :func:`.._utils._sparsax_lu.sparsax_lu` routes the pattern — a
    module-level function, so a sweep closing over it keeps a stable identity.
    The arrays reach the sweep as data (:func:`_bind_sar_solver`).
    """
    import jax.numpy as jnp

    from .._utils._sparsax_lu import sparsax_lu

    pat = build_sar_pattern(W_csc.tocsr(), n)
    arrays = {
        "Ai": jnp.asarray(pat["Ai"], jnp.int32),
        "Aj": jnp.asarray(pat["Aj"], jnp.int32),
        "eye": jnp.asarray(pat["eye_vals"]),
        "w": jnp.asarray(pat["w_vals"]),
    }
    lu_solve = sparsax_lu(arrays["Ai"], arrays["Aj"], n).solve
    return lu_solve, arrays


def _bind_sar_solver(lu_solve, arrays):
    """``solve(rho, rhs)``: ``(I − ρW)⁻¹ rhs`` over the arrays of :func:`_sar_solver_parts`."""

    def solve(rho, rhs):
        return lu_solve(
            arrays["Ai"], arrays["Aj"], arrays["eye"] - rho * arrays["w"], rhs
        )

    return solve
