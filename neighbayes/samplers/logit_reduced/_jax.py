r"""JAX reduced-form SAR-logit Pólya–Gamma Gibbs sampler.

Reuses the reduced-form SAR-NB machinery (``..negbin_reduced._jax``): the sparse
``(I−ρW)⁻¹`` solve (sparsax LU, never densified), the shift-invert Krylov
basis, the thread-parallel chain runner.  Only the Pólya–Gamma augmentation
differs — Bernoulli (h = 1, κ = y − ½, working response κ/ω, no α) instead of
Negative-Binomial.
"""

from __future__ import annotations

import numpy as np

from ..negbin_reduced._core import _KRYLOV_DEGREE_DEFAULT, _KRYLOV_DMAX_DEFAULT


def _rho_log_density_logit(
    rho_val,
    V_stack,
    rho_basis,
    omega,
    y_jax,
    V0_inv_diag,
    mu0,
    intercept_col,
    krylov_dmax,
    X_jax=None,
    solve_at=None,
):
    """β-marginalized log-density of ρ for the reduced-form logit.

    Identical Gaussian core to the reduced-NB density, with the Bernoulli
    working response s = κ/ω (κ = y − ½) — no ``log α`` offset — and no
    ``log|I−ρW|`` term (it cancels when β is marginalized out).

    Evaluates U(ρ) via the Krylov basis when |Δρ| is within the *safe* radius
    (the minimum of ``krylov_dmax`` and the Neumann series' actual convergence
    radius estimated from the Krylov coefficients).  When a direct sparsax solve
    is available (``solve_at``), candidates outside the safe radius use that
    instead — matching the NumPy path's conditional fallback and allowing the
    slice sampler to traverse the full ρ support.  Without a fallback,
    out-of-radius candidates are rejected (−inf).
    """
    import jax
    import jax.numpy as jnp
    from jax.scipy.linalg import solve_triangular

    from ..negbin_reduced._jax import _eval_U_from_basis_jax, _series_radius_jax

    k = V_stack.shape[2]

    # Reject candidates where I − ρW is numerically singular (KLU would fail).
    _singular = jnp.abs(rho_val) > 0.995

    # Clamp the configured dmax to the Krylov basis' actual convergence radius.
    safe_dmax = jnp.minimum(krylov_dmax, _series_radius_jax(V_stack))
    drho = rho_val - rho_basis
    use_basis = jnp.abs(drho) <= safe_dmax

    U_krylov = _eval_U_from_basis_jax(V_stack, drho)

    # Direct sparse solve fallback (sparsax; correct for any ρ)
    has_fallback = (X_jax is not None) and (solve_at is not None)
    if has_fallback:
        # Use lax.cond (not jnp.where) so the direct solve is only evaluated
        # when actually needed.  Clamp ρ away from the singular boundary to
        # avoid KLU failures on near-singular I − ρW.
        _rho_clamped = jnp.clip(rho_val, -0.995, 0.995)
        U = jax.lax.cond(
            use_basis,
            lambda _: U_krylov,
            lambda _: solve_at(_rho_clamped, X_jax),
            operand=None,
        )
    else:
        U = U_krylov

    reparam = (intercept_col >= 0) & (jnp.abs(rho_val) > 1e-8)
    scale = 1.0 - rho_val
    U_rp = jnp.where(reparam, U.at[:, intercept_col].set(1.0), U)
    V0_inv_diag_rp = jnp.where(
        reparam,
        V0_inv_diag.at[intercept_col].set(V0_inv_diag[intercept_col] * scale * scale),
        V0_inv_diag,
    )
    mu0_rp = jnp.where(
        reparam, mu0.at[intercept_col].set(mu0[intercept_col] / scale), mu0
    )

    # Bernoulli working response: κ = y − ½, s = κ/ω (no log-α offset).
    kappa = y_jax - 0.5
    s = kappa / omega
    r = s - U_rp @ mu0_rp

    Uw = U_rp * omega[:, None]
    M = U_rp.T @ Uw + jnp.diag(V0_inv_diag_rp)
    v = Uw.T @ r
    L_M = jnp.linalg.cholesky(M + 1e-10 * jnp.eye(k))
    w = solve_triangular(L_M, v, lower=True)
    quad_pen = w @ w
    rOr = jnp.dot(r, omega * r)
    log_det_M = 2.0 * jnp.sum(jnp.log(jnp.diag(L_M)))

    result = -0.5 * log_det_M - 0.5 * (rOr - quad_pen)
    result = jnp.where(reparam, result + jnp.log(scale), result)

    # Reject candidates where I − ρW is numerically singular (KLU would fail).
    result = jnp.where(_singular, -jnp.inf, result)

    # Reject out-of-radius candidates only when no direct-solve fallback exists.
    if not has_fallback:
        result = jnp.where(use_basis, result, -jnp.inf)
    return result


def _make_reduced_logit_gibbs_step(
    y_jax,
    X_jax,
    sparse_ctx,
    n,
    k,
    priors,
    *,
    intercept_col,
    krylov_degree,
    krylov_dmax,
    krylov_reuse=True,
):
    """Build a JIT-compiled reduced-form SAR-logit Gibbs step (ω → ρ → β)."""
    import jax
    import jax.numpy as jnp
    from jax.scipy.linalg import cho_solve, solve_triangular

    from ..._jax_dispatch import ensure_x64
    from .._utils._jax_slice import jax_slice_sample_1d
    from .._utils._jax_utils import make_pg_draw
    from ..negbin_reduced._jax import (
        _build_krylov_basis_jax,
        _eval_U_from_basis_jax,
        _make_sparse_solvers,
        _series_radius_jax,
    )

    # Pólya-Gamma: pgjax (on-device, exact) or numpy C extension fallback.
    _draw_pg = make_pg_draw()

    ensure_x64()
    _solve, _matvec_W = _make_sparse_solvers(sparse_ctx)

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
    dmax = jnp.float64(krylov_dmax)
    _reuse_threshold = jnp.float64(0.15) if krylov_reuse else jnp.float64(0.0)
    _V_init = jnp.zeros((krylov_degree + 1, n, k), dtype=jnp.float64)
    _ic = int(intercept_col)
    _deg = int(krylov_degree)

    @jax.jit
    def gibbs_step(state, key, slice_width):
        """One sweep; returns ``(new_state, η, (steps_left, steps_right))``.

        The step-out counts come from the ρ slice and drive the runner's warmup
        width adaptation.
        """
        beta = state["beta"]
        rho = state["rho"]
        key_rho, key_beta, key_pg = jax.random.split(key, 3)

        # ── Krylov basis: reuse or rebuild ──
        V_stack_prev = state.get("V_stack", jnp.zeros_like(_V_init))
        rho_basis_prev = state.get("rho_basis", jnp.float64(0.0))
        _drho_check = rho - rho_basis_prev

        # Clamp ρ away from the singular boundary before building the Krylov
        # basis — sparsax's LU fails on near-singular I − ρW (ρ ≈ ±1).
        _rho_safe = jnp.clip(rho, -0.995, 0.995)

        def _rebuild_basis(_):
            V = _build_krylov_basis_jax(
                lambda rhs: _solve(_rho_safe, rhs), X_jax, _matvec_W, n, k, _deg
            )
            _eta = V[0] @ beta
            return V, _rho_safe, _eta

        def _reuse_basis(_):
            _U = _eval_U_from_basis_jax(V_stack_prev, _drho_check)
            _eta = _U @ beta
            return V_stack_prev, rho_basis_prev, _eta

        V_stack, rho_basis, eta = jax.lax.cond(
            jnp.abs(_drho_check) < _reuse_threshold,
            _reuse_basis,
            _rebuild_basis,
            operand=None,
        )

        # ── Block 0: ω ~ PG(1, η) ── (Bernoulli augmentation)
        z = jnp.clip(eta, -20.0, 20.0)
        h = jnp.ones_like(z)  # per-device (n,)
        omega = _draw_pg(h, z, key_pg)

        # ── Block 1: ρ — slice with Krylov basis + direct-solve fallback ──
        def _dens(rv):
            return _rho_log_density_logit(
                rv,
                V_stack,
                rho_basis,
                omega,
                y_jax,
                V0_inv_diag,
                mu0,
                _ic,
                dmax,
                X_jax=X_jax,
                solve_at=lambda rho_val, rhs: _solve(rho_val, rhs),
            )

        rho_new, _, steps_left, steps_right = jax_slice_sample_1d(
            _dens, rho, rho_lo, rho_hi, key=key_rho, w=slice_width, return_steps=True
        )

        # ── Block 2: β | ρ, ω, y — conjugate normal ──
        # Evaluate X̃ = (I−ρ_new W)⁻¹X: Krylov basis when within safe radius,
        # direct sparsax solve otherwise.  Clamp ρ to avoid KLU failures.
        drho_new = rho_new - rho_basis
        _safe_dmax_beta = jnp.minimum(dmax, _series_radius_jax(V_stack))
        _use_krylov_beta = jnp.abs(drho_new) <= _safe_dmax_beta
        _rho_new_clamped = jnp.clip(rho_new, -0.995, 0.995)
        Xtilde = jax.lax.cond(
            _use_krylov_beta,
            lambda _: _eval_U_from_basis_jax(V_stack, drho_new),
            lambda _: _solve(_rho_new_clamped, X_jax),
            operand=None,
        )

        reparam_beta = (_ic >= 0) & (jnp.abs(rho_new) > 1e-8)
        scale_beta = 1.0 - rho_new
        Xtilde_rp = jnp.where(reparam_beta, Xtilde.at[:, _ic].set(1.0), Xtilde)
        V0_inv_diag_rp = jnp.where(
            reparam_beta,
            V0_inv_diag.at[_ic].set(V0_inv_diag[_ic] * scale_beta * scale_beta),
            V0_inv_diag,
        )
        mu0_rp = jnp.where(reparam_beta, mu0.at[_ic].set(mu0[_ic] / scale_beta), mu0)

        kappa = y_jax - 0.5
        Xt_omega = Xtilde_rp * omega[:, None]
        Sigma_beta_inv = Xt_omega.T @ Xtilde_rp + jnp.diag(V0_inv_diag_rp)
        rhs = Xtilde_rp.T @ kappa + V0_inv_diag_rp * mu0_rp

        L_beta = jnp.linalg.cholesky(Sigma_beta_inv + 1e-10 * jnp.eye(k))
        m_beta = cho_solve((L_beta, True), rhs)
        z_beta = jax.random.normal(key_beta, shape=(k,), dtype=jnp.float64)
        beta_draw = m_beta + solve_triangular(L_beta.T, z_beta, lower=False)
        beta_new = jnp.where(
            reparam_beta, beta_draw.at[_ic].set(beta_draw[_ic] * scale_beta), beta_draw
        )

        eta_new = Xtilde @ beta_new
        new_state = {
            "beta": beta_new,
            "rho": rho_new,
            "omega": omega,
            "V_stack": V_stack,
            "rho_basis": rho_basis,
        }
        # η feeds the on-device Bernoulli log-likelihood.
        return new_state, eta_new, (steps_left, steps_right)

    return gibbs_step


def run_chains_jax_reduced_logit(
    y,
    X,
    W_sparse,
    priors,
    inits,
    draws,
    tune,
    *,
    thin=1,
    intercept_col=0,
    krylov_degree=_KRYLOV_DEGREE_DEFAULT,
    krylov_dmax=_KRYLOV_DMAX_DEFAULT,
    slice_width=0.4,
    jax_seeds=None,
    progressbar=False,
    krylov_reuse=True,
    store_log_lik=True,
):
    """Run the reduced-form SAR-logit PG-Gibbs sampler (chains in parallel threads).

    Returns one dict per chain with keys ``rho``, ``beta``, ``log_lik``.
    ``slice_width`` is the initial ρ slice width; each chain adapts it during
    warmup and holds it for the draws.
    """
    import jax
    import jax.numpy as jnp

    from ..._jax_dispatch import ensure_x64
    from .._utils._jax_utils import run_chains_chunked
    from ..negbin_reduced._jax import _build_sparse_ctx

    ensure_x64()
    chains = len(inits)
    n, k = X.shape
    y_jax = jnp.asarray(y, dtype=jnp.float64)
    X_jax = jnp.asarray(X, dtype=jnp.float64)
    sparse_ctx = _build_sparse_ctx(W_sparse, n)

    from .._utils._sparsax_lu import set_sparsax_lu_cache_size

    set_sparsax_lu_cache_size(max(32, 6 * chains))

    if jax_seeds is None:
        jax_seeds = list(range(chains))

    gibbs_step = _make_reduced_logit_gibbs_step(
        y_jax=y_jax,
        X_jax=X_jax,
        sparse_ctx=sparse_ctx,
        n=n,
        k=k,
        priors=priors,
        intercept_col=intercept_col,
        krylov_degree=krylov_degree,
        krylov_dmax=krylov_dmax,
        krylov_reuse=krylov_reuse,
    )

    _V_init = jnp.zeros((krylov_degree + 1, n, k), dtype=jnp.float64)
    states = [
        {
            "beta": jnp.asarray(i.beta, dtype=jnp.float64),
            "rho": jnp.float64(float(i.rho)),
            "omega": jnp.asarray(i.omega, dtype=jnp.float64),
            "V_stack": _V_init,
            "rho_basis": jnp.float64(0.0),
            "slice_width": jnp.float64(slice_width),
        }
        for i in inits
    ]
    warm_keys = [jax.random.PRNGKey(int(s)) for s in jax_seeds]
    draw_keys = [jax.random.fold_in(jax.random.PRNGKey(int(s)), 1) for s in jax_seeds]

    from .._utils._jax_slice import adapt_slice_width

    def _sweep(st, key, tuning):
        width = st["slice_width"]
        core = {name: value for name, value in st.items() if name != "slice_width"}
        core, eta, (steps_left, steps_right) = gibbs_step(core, key, width)
        width = adapt_slice_width(width, steps_left, steps_right, tuning)
        trace = (core["rho"], core["beta"])
        if store_log_lik:  # η is traced only to build the log-likelihood
            trace += (eta,)
        return dict(core, slice_width=width), trace

    _, traces = run_chains_chunked(
        _sweep, states, warm_keys, draw_keys, tune=tune, draws=draws
    )
    rho_all, beta_all = traces[:2]
    eta_all = traces[2] if store_log_lik else None

    # Pointwise Bernoulli-logit log-likelihood from the fitted η (no post-hoc solves).
    from ._core import _logit_loglik_pointwise

    sl = slice(None, None, thin) if thin > 1 else slice(None)
    y_np = np.asarray(y, dtype=np.float64)
    results = []
    for c in range(chains):
        log_lik = None
        if store_log_lik:
            eta_c = eta_all[c, sl]  # (n_keep, n)
            log_lik = _logit_loglik_pointwise(y_np, eta_c)
        results.append(
            {
                "rho": rho_all[c, sl],
                "beta": beta_all[c, sl],
                "log_lik": log_lik,
            }
        )
    return results
