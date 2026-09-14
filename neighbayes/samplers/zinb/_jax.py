r"""JAX reduced-form Zero-Inflated SAR Negative-Binomial Pólya-Gamma Gibbs.

Composes the two reduced-form jax samplers already built:

* **selection** — reduced-form SAR-**logit** on the *latent* activation ``z``
  (``η_sel = (I − λ W_sel)⁻¹ Z γ``), reusing ``logit_reduced``'s Krylov-only
  ρ-slice density; the working response is ``κ_sel = z − ½`` and ``z`` is redrawn
  every sweep;
* **count** — reduced-form SAR-**NB** on the counts (``η_cnt = (I − ρ W_cnt)⁻¹ X β``),
  reusing ``negbin_reduced``'s Krylov-only density, **z-masked** (structural
  zeros contribute nothing: ``ω_cnt = ε`` and the working ``y`` is set to ``α``
  so ``κ = 0`` there);

linked by the latent indicator ``z``.  Both equations use sparsax-LU solves
(never densified), the on-device Pólya-Gamma draw (pgjax), and run the chains in
parallel threads.
"""

from __future__ import annotations

import numpy as np

from ..negbin_reduced._core import _KRYLOV_DEGREE_DEFAULT, _KRYLOV_DMAX_DEFAULT


def _make_zinb_gibbs_step(
    y_jax,
    d_jax,
    Z_jax,
    X_jax,
    sel_ctx,
    cnt_ctx,
    n,
    p,
    k,
    priors,
    *,
    krylov_degree,
    krylov_dmax,
):
    """Build a JIT-compiled reduced-form ZINB Gibbs step (9 blocks)."""
    import jax
    import jax.numpy as jnp

    from ..._jax_dispatch import ensure_x64
    from .._utils._jax_slice import jax_slice_sample_1d
    from ..logit_reduced._jax import _rho_log_density_logit
    from ..negbin_reduced._jax import (
        _build_krylov_basis_jax,
        _eval_U_from_basis_jax,
        _make_sparse_solvers,
        _rho_log_density_marginal_jax,
        _series_radius_jax,
    )

    ensure_x64()

    from .._utils._jax_utils import make_pg_draw

    _draw_pg = make_pg_draw()

    _solve_sel, _matvec_Wsel = _make_sparse_solvers(sel_ctx)
    _solve_cnt, _matvec_Wcnt = _make_sparse_solvers(cnt_ctx)

    def _prior_vec(mu, sigma, dim):
        v0 = (
            jnp.full(dim, 1.0 / float(sigma) ** 2)
            if np.isscalar(sigma)
            else 1.0 / jnp.asarray(sigma, dtype=jnp.float64) ** 2
        )
        m0 = (
            jnp.full(dim, float(mu))
            if np.isscalar(mu)
            else jnp.asarray(mu, dtype=jnp.float64)
        )
        return v0, m0

    V0g, mu0g = _prior_vec(priors.gamma_mu, priors.gamma_sigma, p)
    V0b, mu0b = _prior_vec(priors.beta_mu, priors.beta_sigma, k)
    lam_lo, lam_hi = jnp.float64(priors.lam_lower), jnp.float64(priors.lam_upper)
    rho_lo, rho_hi = jnp.float64(priors.rho_lower), jnp.float64(priors.rho_upper)
    a_sigma, a_nu = jnp.float64(priors.alpha_sigma), jnp.float64(priors.alpha_nu)
    dmax = jnp.float64(krylov_dmax)
    _deg = int(krylov_degree)
    _IC = -1  # reparam disabled (target unchanged); simpler/robust

    from .._utils._jax_utils import conjugate_normal as _conjugate_normal

    @jax.jit
    def gibbs_step(state, key, slice_widths):
        """One sweep; ``slice_widths`` is ``(λ width, ρ width)``.

        Returns ``(new_state, trace, steps)``, where ``steps`` holds the λ and ρ
        slices' ``(left, right)`` step-out counts for warmup width adaptation.
        """
        w_lam, w_rho = slice_widths
        gamma = state["gamma"]
        lam = state["lam"]
        beta = state["beta"]
        rho = state["rho"]
        alpha = state["alpha"]
        z = state["z"]  # (n,) float 0/1 from the previous sweep
        (kβ, kγ, kλ, kρ, kα, kzs, kzc, kpg_s, kpg_c) = jax.random.split(key, 9)

        # ─────────── SELECTION (reduced-form logit, response z) ───────────
        eta_sel = _solve_sel(lam, Z_jax @ gamma)
        omega_sel = _draw_pg(jnp.ones(n), jnp.clip(eta_sel, -20.0, 20.0), kpg_s)

        V_sel = _build_krylov_basis_jax(
            lambda rhs: _solve_sel(lam, rhs), Z_jax, _matvec_Wsel, n, p, _deg
        )
        lam_new, _, lam_left, lam_right = jax_slice_sample_1d(
            # ``solve_at`` is what lets the slice leave the Krylov radius.
            # Without it every out-of-radius candidate evaluates to −inf and
            # λ is confined to a window around its current value each sweep —
            # not a valid slice, and visibly biased at small n.  The standalone
            # reduced-logit and reduced-NB samplers both pass it.
            lambda lv: _rho_log_density_logit(
                lv,
                V_sel,
                lam,
                omega_sel,
                z,
                V0g,
                mu0g,
                _IC,
                dmax,
                X_jax=Z_jax,
                solve_at=lambda v, rhs: _solve_sel(v, rhs),
            ),
            lam,
            lam_lo,
            lam_hi,
            key=kλ,
            w=w_lam,
            return_steps=True,
        )
        # Z̃ = (I−λ_new W_sel)⁻¹Z: Krylov basis inside the safe radius, direct
        # sparsax solve outside it (the slice can now land there).
        _dlam = lam_new - lam
        Ztilde = jax.lax.cond(
            jnp.abs(_dlam) <= jnp.minimum(dmax, _series_radius_jax(V_sel)),
            lambda _: _eval_U_from_basis_jax(V_sel, _dlam),
            lambda _: _solve_sel(jnp.clip(lam_new, -0.995, 0.995), Z_jax),
            operand=None,
        )
        gamma = _conjugate_normal(Ztilde, omega_sel, z - 0.5, V0g, mu0g, kγ, p)
        lam = lam_new
        eta_sel = Ztilde @ gamma

        # ─────────────────── ZERO ALLOCATION (z draw) ────────────────────
        eta_cnt = _solve_cnt(rho, X_jax @ beta)
        pi = jax.nn.sigmoid(eta_sel)
        mu_cnt = jnp.exp(jnp.clip(eta_cnt, -30.0, 30.0))
        p_nb0 = jnp.power(alpha / (mu_cnt + alpha), alpha)
        p_z1_if0 = pi * p_nb0 / (pi * p_nb0 + (1.0 - pi) + 1e-300)
        prob = jnp.where(y_jax > 0, 1.0, p_z1_if0)
        z = (jax.random.uniform(kzc, shape=(n,), dtype=jnp.float64) < prob).astype(
            jnp.float64
        )
        z1 = z > 0.5

        # ─────────── COUNT (reduced-form NB, z-masked) ───────────
        h_cnt = jnp.where(z1, jnp.maximum(y_jax + alpha, 1e-3), 1.0)
        omega_cnt = _draw_pg(
            h_cnt, jnp.clip(eta_cnt - jnp.log(alpha), -20.0, 20.0), kpg_c
        )
        omega_cnt = jnp.where(z1, omega_cnt, 1e-300)  # mask structural zeros
        y_for = jnp.where(z1, y_jax, alpha)  # κ = 0.5(y−α) = 0 where z=0

        V_cnt = _build_krylov_basis_jax(
            lambda rhs: _solve_cnt(rho, rhs), X_jax, _matvec_Wcnt, n, k, _deg
        )
        rho_new, _, rho_left, rho_right = jax_slice_sample_1d(
            lambda rv: _rho_log_density_marginal_jax(
                rv,
                V_cnt,
                rho,
                omega_cnt,
                y_for,
                alpha,
                V0b,
                mu0b,
                _IC,
                dmax,
                X_jax=X_jax,
                solve_at=lambda v, rhs: _solve_cnt(v, rhs),
            ),
            rho,
            rho_lo,
            rho_hi,
            key=kρ,
            w=w_rho,
            return_steps=True,
        )
        _drho = rho_new - rho
        Xtilde = jax.lax.cond(
            jnp.abs(_drho) <= jnp.minimum(dmax, _series_radius_jax(V_cnt)),
            lambda _: _eval_U_from_basis_jax(V_cnt, _drho),
            lambda _: _solve_cnt(jnp.clip(rho_new, -0.995, 0.995), X_jax),
            operand=None,
        )
        log_alpha = jnp.log(alpha)
        working_cnt = 0.5 * (y_for - alpha) + omega_cnt * log_alpha
        beta = _conjugate_normal(Xtilde, omega_cnt, working_cnt, V0b, mu0b, kβ, k)
        rho = rho_new
        eta_cnt = Xtilde @ beta

        # α | y, η_cnt, z  — slice on log α, z-masked NB log-likelihood.
        from jax.scipy.special import gammaln as _gammaln

        def _alpha_logdens(log_a):
            a = jnp.exp(log_a)
            mu = jnp.exp(jnp.clip(eta_cnt, -30.0, 30.0))
            ll = (
                _gammaln(y_jax + a)
                - _gammaln(a)
                + y_jax * jnp.log(jnp.maximum(mu / (mu + a), 1e-300))
                + a * jnp.log(jnp.maximum(a / (mu + a), 1e-300))
            )
            total = jnp.sum(z * ll)  # only z=1 obs contribute
            log_prior = (
                -0.5 * (a_nu + 1.0) * jnp.log1p((a * a) / (a_nu * a_sigma * a_sigma))
            )
            return log_a + total + log_prior

        log_a_new, _ = jax_slice_sample_1d(
            _alpha_logdens,
            jnp.log(alpha),
            jnp.float64(-4.0),
            jnp.float64(4.0),
            key=kα,
            w=jnp.float64(1.0),
        )
        alpha = jnp.exp(log_a_new)

        new_state = {
            "gamma": gamma,
            "lam": lam,
            "beta": beta,
            "rho": rho,
            "alpha": alpha,
            "z": z,
        }
        trace = (lam, gamma, rho, beta, alpha, eta_sel, eta_cnt)
        return new_state, trace, ((lam_left, lam_right), (rho_left, rho_right))

    return gibbs_step


def run_chains_jax_zinb(
    y,
    d,
    Z,
    X,
    W_sel_sparse,
    W_cnt_sparse,
    priors,
    inits,
    draws,
    tune,
    *,
    thin=1,
    krylov_degree=_KRYLOV_DEGREE_DEFAULT,
    krylov_dmax=_KRYLOV_DMAX_DEFAULT,
    slice_width=0.4,
    jax_seeds=None,
    progressbar=True,
):
    """Run the reduced-form ZINB PG-Gibbs sampler (chains in parallel threads).

    Returns one dict per chain with keys ``lam``, ``gamma``, ``rho``, ``beta``,
    ``alpha``, ``log_lik``, ``pi_mean``.  ``log_lik`` is the **marginal**
    ZINB log-pmf of ``y`` with the latent allocation integrated out.

    ``slice_width`` is the initial width of the λ and ρ slices; each chain
    adapts both during warmup and holds them for the draws.

    ``d`` (``= 1(y > 0)``) is unused — the selection equation is fit against
    the latent allocation ``z`` — and is kept only for signature parity with
    :func:`neighbayes.samplers.zinb._core.run_zinb_chain`.
    """
    import jax
    import jax.numpy as jnp

    from ..._jax_dispatch import ensure_x64
    from .._utils._progress import GibbsProgressBarManager
    from ..negbin_reduced._jax import _build_sparse_ctx
    from ._core import _zinb_loglik_pointwise

    ensure_x64()
    chains = len(inits)
    n, k = X.shape
    p = Z.shape[1]
    y_jax = jnp.asarray(y, dtype=jnp.float64)
    d_jax = jnp.asarray(d, dtype=jnp.float64)
    Z_jax = jnp.asarray(Z, dtype=jnp.float64)
    X_jax = jnp.asarray(X, dtype=jnp.float64)
    sel_ctx = _build_sparse_ctx(W_sel_sparse, n)
    cnt_ctx = _build_sparse_ctx(W_cnt_sparse, n)

    from .._utils._sparsax_lu import set_sparsax_lu_cache_size

    # two patterns (W_sel, W_cnt) x chains x a few distinct rho/lam per sweep
    set_sparsax_lu_cache_size(max(64, 12 * chains))

    if jax_seeds is None:
        jax_seeds = list(range(chains))

    gibbs_step = _make_zinb_gibbs_step(
        y_jax,
        d_jax,
        Z_jax,
        X_jax,
        sel_ctx,
        cnt_ctx,
        n,
        p,
        k,
        priors,
        krylov_degree=krylov_degree,
        krylov_dmax=krylov_dmax,
    )

    state0 = {
        "gamma": jnp.asarray(np.stack([i.gamma for i in inits]), dtype=jnp.float64),
        "lam": jnp.asarray([float(i.lam) for i in inits], dtype=jnp.float64),
        "beta": jnp.asarray(np.stack([i.beta for i in inits]), dtype=jnp.float64),
        "rho": jnp.asarray([float(i.rho) for i in inits], dtype=jnp.float64),
        "alpha": jnp.asarray([float(i.alpha) for i in inits], dtype=jnp.float64),
        "z": jnp.asarray(np.stack([np.asarray(i.z, dtype=np.float64) for i in inits])),
        "slice_widths": (jnp.full(chains, slice_width, dtype=jnp.float64),) * 2,
    }
    warm_keys = jnp.stack([jax.random.PRNGKey(int(s)) for s in jax_seeds])
    draw_keys = jnp.stack(
        [jax.random.fold_in(jax.random.PRNGKey(int(s)), 1) for s in jax_seeds]
    )

    # Chains run in parallel threads, in compiled chunks, and the progress bar
    # advances between chunks (see run_chains_chunked).
    from .._utils._jax_slice import adapt_slice_width
    from .._utils._jax_utils import run_chains_chunked

    def _sweep(st, key, tuning):
        widths = st["slice_widths"]
        core = {name: v for name, v in st.items() if name != "slice_widths"}
        core, trace, steps = gibbs_step(core, key, widths)
        widths = tuple(
            adapt_slice_width(w, left, right, tuning)
            for w, (left, right) in zip(widths, steps)
        )
        return dict(core, slice_widths=widths), trace

    with GibbsProgressBarManager(
        chains=chains,
        draws=draws,
        tune=tune,
        progressbar=progressbar,
        model_type="zinb_sar",
    ) as pm:
        if pm is not None:
            for c in range(chains):
                pm.start_chain(c)

        def _progress(i, tuning):
            if pm is not None:
                for c in range(chains):
                    pm.update(c, i, tuning=tuning)

        _, traces = run_chains_chunked(
            _sweep,
            [
                jax.tree_util.tree_map(lambda a, c=c: a[c], state0)
                for c in range(chains)
            ],
            list(warm_keys),
            list(draw_keys),
            tune=tune,
            draws=draws,
            on_chunk=_progress,
        )

    lam_all, gamma_all, rho_all, beta_all, alpha_all, etasel_all, etacnt_all = traces

    sl = slice(None, None, thin) if thin > 1 else slice(None)
    y_np = np.asarray(y, dtype=np.float64)
    results = []
    for c in range(chains):
        eta_sel = etasel_all[c, sl]
        eta_cnt = etacnt_all[c, sl]
        a = alpha_all[c, sl][:, None]
        # Marginal ZINB log-pmf of the observed count, latent z integrated
        # out — the same helper the NumPy backend stores.
        log_lik = _zinb_loglik_pointwise(y_np, eta_sel, eta_cnt, a)
        results.append(
            {
                "lam": lam_all[c, sl],
                "gamma": gamma_all[c, sl],
                "rho": rho_all[c, sl],
                "beta": beta_all[c, sl],
                "alpha": alpha_all[c, sl],
                "log_lik": log_lik,
                "pi_mean": (1.0 / (1.0 + np.exp(-eta_sel))).mean(axis=1),
            }
        )
    return results
