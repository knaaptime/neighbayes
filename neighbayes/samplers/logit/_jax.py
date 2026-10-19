r"""JAX ports of the SAR- and SEM-logit Gibbs sweeps with random effects.

The sweep of :mod:`._core` — ω → r (η and β integrated out) → σ (η and β
integrated out) → β → η → ridge move → scale move, for SAR (``r = ρ``) and SEM
(``r = λ``) alike — compiled into one program per structure and run
thread-per-chain through :func:`.._utils._jax_utils.run_chains_chunked`.

- **PG sampling**: exact Pólya–Gamma draws via :func:`make_pg_draw` (pgjax's
  Devroye sampler when installed; NumPy C extension fallback).
- **Precisions**: ``P = AᵀA/σ² + Ω`` is assembled as ``P̃ = σ²P``, the σ = 1
  assembly with ``σ²ω`` on the diagonal, and factored by sparsax (sparse
  Cholesky) or, without it, densely.  Optional Krylov reuse factors ``P̃`` once
  per r slice.

References
----------
Polson, N. G., Scott, J. G., & Windle, J. (2013). Bayesian inference
for logistic models using Pólya–Gamma latent variables.
*Journal of the American Statistical Association*, 108(504), 1339–1349.
"""

from __future__ import annotations

from functools import partial

import numpy as np

from ..._jax_dispatch import ensure_x64
from .._utils._jax_utils import (
    build_w_bcoo as _build_w_bcoo,
)
from .._utils._jax_utils import (
    make_pg_draw,
)
from .._utils._spatial_normal import (
    _sparsax_factor_ops_available,
    build_precision_krylov_basis_jax,
    eval_precision_logdet_from_basis_jax,
    eval_precision_solve_from_basis_jax,
)
from ._core import _ATTENUATION, JAXLogitGibbsState, JAXSEMLogitGibbsState

# Pólya-Gamma for the ω ~ PG(1, η) block.  pgjax's exact Devroye sampler is
# on-device (no host round-trip) and exact for any h; falls back to the numpy
# C extension via pure_callback when pgjax is unavailable.  h = 1 (Bernoulli).
_draw_pg1 = make_pg_draw()


def _sd_log_prior_jax(jnp, s, nu_sd, sd_scale):
    """See :func:`._core._sd_log_prior`."""
    return -0.5 * (nu_sd + 1.0) * jnp.log1p(s * s / (nu_sd * sd_scale**2))


def _slice_log_sd_jax(jnp, jax_slice_sample_1d, log_density, sigma, sd_scale, w,
                      key):  # fmt: skip
    """One slice draw of ``log σ`` on ``[log A − 10, log A + 5]``, returned as σ.

    See :func:`.._utils._group_effects._slice_log_sd`.
    """
    la = jnp.log(sd_scale)
    lo, hi = la - 10.0, la + 5.0
    ls, _ = jax_slice_sample_1d(
        log_density, jnp.clip(jnp.log(sigma), lo, hi), lo, hi, key=key, w=w
    )
    return jnp.exp(ls)


def _ridge_move_jax(jnp, jax_slice_sample_1d, y, eta, m, beta, sigma, beta_mu,
                    beta_sigma2, nu_sd, sd_scale, key):  # fmt: skip
    """σ with ``ν/σ`` and ``β/g(σ)`` held; see :func:`._core._ridge_move`."""
    k = beta.shape[0]
    g0 = jnp.sqrt(1.0 + _ATTENUATION * sigma**2)
    d = eta - m
    bt = beta / g0

    def at(s):
        g = jnp.sqrt(1.0 + _ATTENUATION * s * s)
        return (g / g0) * m + (s / sigma) * d, g * bt, g

    def log_density(ls):
        s = jnp.exp(ls)
        e, b, g = at(s)
        val = (
            jnp.sum(y * e - jnp.logaddexp(0.0, e))
            - 0.5 * jnp.sum((b - beta_mu) ** 2 / beta_sigma2)
            + k * jnp.log(g)
            + _sd_log_prior_jax(jnp, s, nu_sd, sd_scale)
            + ls
        )
        return jnp.where(jnp.isfinite(val), val, -jnp.inf)

    s = _slice_log_sd_jax(
        jnp, jax_slice_sample_1d, log_density, sigma, sd_scale, 0.3, key
    )
    e, b, _ = at(s)
    return e, b, s


def _scale_move_jax(jnp, jax_slice_sample_1d, y, eta, beta, sigma, beta_mu,
                    beta_sigma2, nu_sd, sd_scale, key):  # fmt: skip
    """``(η, β, σ) → eᵗ(η, β, σ)``; see :func:`._core._scale_move`."""
    k = beta.shape[0]

    def log_density(t):
        e = jnp.exp(t)
        eta_t = e * eta
        val = (
            jnp.sum(y * eta_t - jnp.logaddexp(0.0, eta_t))
            - 0.5 * jnp.sum((e * beta - beta_mu) ** 2 / beta_sigma2)
            + _sd_log_prior_jax(jnp, e * sigma, nu_sd, sd_scale)
            + (k + 1) * t
        )
        return jnp.where(jnp.isfinite(val), val, -jnp.inf)

    t, _ = jax_slice_sample_1d(log_density, 0.0, -30.0, 30.0, key=key, w=0.2)
    e = jnp.exp(t)
    return e * eta, e * beta, e * sigma


def _make_step(
    n,
    k,
    *,
    sem,
    logdet_kind,
    logdet_closure=None,
    use_sparsax=True,
    krylov_degree: int = 0,
    krylov_dmax: float = 0.4,
):
    """The SAR- or (``sem``) SEM-logit Gibbs step, closing over static settings.

    ``gibbs_step(state, key, data, slice_width, return_steps=False)``, with
    ``data`` from :func:`_logit_data`: one compiled program serves every fit
    of the same structure.  The sweep is that of :mod:`._core`, on the
    design's ``u(r) = AᵀD`` and ``DᵀD(r)`` (``data["U"]``, ``data["G"]``, the
    coefficients of their polynomials in r).  With ``P̃ = σ²P``,
    ``sol = P̃⁻¹[κ | u]``: ``Σ_β⁻¹ = (DᵀD − uᵀP̃⁻¹u)/σ² + V₀⁻¹``,
    ``r_β = uᵀP̃⁻¹κ + V₀⁻¹b₀``, ``κᵀP⁻¹κ = σ²κᵀP̃⁻¹κ``, and
    ``−n log σ − ½log|P| = −½log|P̃|``.
    """
    import equinox as eqx
    import jax
    import jax.numpy as jnp
    from jax.scipy.linalg import cho_solve, solve_triangular

    from ..._logdet._jax import eval_logdet_params
    from .._utils._jax_slice import jax_slice_sample_1d
    from .._utils._sparsax_utils import make_sparsax_ops

    state_cls = JAXSEMLogitGibbsState if sem else JAXLogitGibbsState
    spatial = "lam" if sem else "rho"
    use_krylov = use_sparsax and krylov_degree > 0 and _sparsax_factor_ops_available()

    @eqx.filter_jit
    def gibbs_step(state, key, data, slice_width, return_steps=False):
        """One sweep; with ``return_steps`` also the r slice's step-out counts."""
        eta, r, sigma = state.eta, getattr(state, spatial), state.sigma
        X, kappa, U, G = data["X"], data["kappa"], data["U"], data["G"]
        y = kappa + 0.5
        beta_mu, beta_sigma2 = data["beta_mu"], data["beta_sigma2"]
        V0_inv = 1.0 / beta_sigma2
        lower, upper = data["lower"], data["upper"]
        nu_sd, sd_scale = data["sd_nu"], data["sd_scale"]

        def u_of(rv):
            return jnp.tensordot(rv ** jnp.arange(U.shape[0]), U, axes=1)

        def DtD_of(rv):
            return jnp.tensordot(rv ** jnp.arange(G.shape[0]), G, axes=1)

        if use_sparsax:
            sp_ = data["sparsax"]
            Ai, Aj = sp_["Ai"], sp_["Aj"]
            W_sym_vals, WtW_vals = sp_["W_sym_vals"], sp_["WtW_vals"]
            diag_idx = sp_["diag_idx"]
            nnz = Ai.shape[0]
            # Factor-once closures: the η draw and each density evaluation do
            # exactly ONE numeric factorization (sample_gaussian / factor_solve).
            eta_sample, solve_logdet = make_sparsax_ops(Ai, Aj, n)

            def assemble(om_t, rv):
                """COO values of ``I + diag(om_t) − r(W+Wᵀ) + r²WᵀW``."""
                Ax = -rv * W_sym_vals + rv**2 * WtW_vals
                return Ax + jnp.zeros(nnz, dtype=jnp.float64).at[diag_idx].set(
                    1.0 + om_t
                )

            def solve_P(om_t, rv, rhs):
                return solve_logdet(assemble(om_t, rv), rhs)

            def sample_P(om_t, rv, rhs, z):
                return eta_sample(assemble(om_t, rv), rhs, z)

        else:
            W_sym, WtW = data["W_sym_dense"], data["WtW_dense"]

            def chol_P(om_t, rv):
                P = jnp.diag(1.0 + om_t) - rv * W_sym + rv**2 * WtW
                return jnp.linalg.cholesky(P)

            def solve_P(om_t, rv, rhs):
                L = chol_P(om_t, rv)
                return cho_solve((L, True), rhs), 2.0 * jnp.sum(jnp.log(jnp.diag(L)))

            def sample_P(om_t, rv, rhs, z):
                L = chol_P(om_t, rv)
                return cho_solve((L, True), rhs) + solve_triangular(L.T, z, lower=False)

        def logdet_A(rv):
            prm = logdet_closure if logdet_kind == "closure" else data["logdet"]
            return eval_logdet_params(logdet_kind, prm, rv)

        keys = jax.random.split(key, 8)

        # ── 1. ω ~ PG(1, η) — on-device (pgjax Devroye, else exp) ──
        omega = _draw_pg1(jnp.ones(n), eta, keys[0])

        def collapsed(rv, s, basis=None):
            """``log p(r, σ | ω)`` with η and β integrated out, and β's moments."""
            s2 = s * s
            u = u_of(rv)
            sol, log_det = solve_P(s2 * omega, rv, jnp.column_stack([kappa, u]))
            if basis is not None:
                V_stack, ld_coefs, safe_dmax = basis
                dr = rv - r
                within = jnp.abs(dr) <= safe_dmax
                sol_all = eval_precision_solve_from_basis_jax(V_stack, dr)
                M_kry = sum(
                    rv**j * sol_all[:, 1 + j * k : 1 + (j + 1) * k]
                    for j in range(U.shape[0])
                )
                ld_kry = eval_precision_logdet_from_basis_jax(ld_coefs, dr)
                # Both operands are evaluated; the direct solve covers
                # candidates outside the Krylov radius.
                sol = jnp.where(within, jnp.column_stack([sol_all[:, 0], M_kry]), sol)
                log_det = jnp.where(within, ld_kry, log_det)
            z_t, M_t = sol[:, 0], sol[:, 1:]
            Sig_inv = (DtD_of(rv) - u.T @ M_t) / s2 + jnp.diag(V0_inv)
            L_sig = jnp.linalg.cholesky(Sig_inv + 1e-10 * jnp.eye(k))
            rhs_b = u.T @ z_t + V0_inv * beta_mu
            m_b = cho_solve((L_sig, True), rhs_b)
            val = (
                logdet_A(rv)
                - 0.5 * log_det
                - jnp.sum(jnp.log(jnp.diag(L_sig)))
                + 0.5 * s2 * (kappa @ z_t)
                + 0.5 * (rhs_b @ m_b)
            )
            val = jnp.where((rv >= lower) & (rv <= upper), val, -jnp.inf)
            return jnp.where(jnp.isfinite(val), val, -jnp.inf), L_sig, m_b

        # ── 2. r | ω, σ, then σ | ω, r — η and β integrated out ──
        basis = None
        if use_krylov:
            # Factor P̃(r_c) once at the slice centre and evaluate candidates
            # within krylov_dmax by a Horner recurrence on the shift-invert basis.
            seeds = jnp.column_stack([kappa, *[U[j] for j in range(U.shape[0])]])
            (_token, V_stack, ld_coefs, safe_dmax) = build_precision_krylov_basis_jax(
                Ai, Aj, assemble(sigma**2 * omega, r), W_sym_vals - 2.0 * r * WtW_vals,
                WtW_vals, seeds, n=n, degree=krylov_degree, dmax=krylov_dmax,
            )  # fmt: skip
            basis = (V_stack, ld_coefs, safe_dmax)

        r_new, _, steps_left, steps_right = jax_slice_sample_1d(
            lambda rv: collapsed(rv, sigma, basis)[0],
            r,
            lower,
            upper,
            key=keys[1],
            w=slice_width,
            return_steps=True,
        )

        def sigma_density(ls):
            s = jnp.exp(ls)
            return (
                collapsed(r_new, s)[0] + _sd_log_prior_jax(jnp, s, nu_sd, sd_scale) + ls
            )

        sigma_new = _slice_log_sd_jax(
            jnp, jax_slice_sample_1d, sigma_density, sigma, sd_scale, 0.5, keys[2]
        )

        # ── 3. β | ω, r, σ — from the collapsed moments ──
        _, L_sig, m_b = collapsed(r_new, sigma_new)
        z_beta = jax.random.normal(keys[3], shape=(k,), dtype=jnp.float64)
        beta_new = m_b + solve_triangular(L_sig.T, z_beta, lower=False)

        # ── 4. η | ω, r, β, σ ~ N(P⁻¹rhs, P⁻¹) = σ·N(P̃⁻¹(σ rhs), P̃⁻¹) ──
        s2 = sigma_new**2
        u_beta = u_of(r_new) @ beta_new
        z_eta = jax.random.normal(keys[4], shape=(n,), dtype=jnp.float64)
        eta_new = sigma_new * sample_P(
            s2 * omega, r_new, sigma_new * (u_beta / s2 + kappa), z_eta
        )

        # ── 5. the ridge move, on η's β part m = A⁻¹Dβ ──
        if sem:
            m = X @ beta_new
        else:
            # m = A⁻¹Xβ from the normal equations AᵀA m = AᵀXβ = u β.
            m, _ = solve_P(jnp.zeros(n), r_new, u_beta)
        eta_new, beta_new, sigma_new = _ridge_move_jax(
            jnp, jax_slice_sample_1d, y, eta_new, m, beta_new, sigma_new, beta_mu,
            beta_sigma2, nu_sd, sd_scale, keys[5],
        )  # fmt: skip

        # ── 6. the scale move ──
        eta_new, beta_new, sigma_new = _scale_move_jax(
            jnp, jax_slice_sample_1d, y, eta_new, beta_new, sigma_new, beta_mu,
            beta_sigma2, nu_sd, sd_scale, keys[6],
        )  # fmt: skip

        new_state = state_cls(
            eta=eta_new,
            beta=beta_new,
            omega=omega,
            sigma=sigma_new,
            **{spatial: r_new},
        )
        if return_steps:
            return new_state, jnp.float64(1.0), (steps_left, steps_right)
        return new_state, jnp.float64(1.0)  # slice always accepts

    return gibbs_step


def _logit_data(
    y, X, W_bcoo, Wt_bcoo, priors, logdet_params, *, spatial="rho",
    W_sym_dense=None, WtW_dense=None, sparsax_pattern=None,
):  # fmt: skip
    """The logit steps' ``data``: every data-dependent array and scalar.

    ``U`` and ``G`` hold the coefficients of ``u(r) = AᵀD`` and ``DᵀD(r)``,
    polynomials in r (see :class:`._core._Design`): ``D = X`` for SAR
    (``spatial="rho"``), ``D = AX`` for SEM.
    """
    import jax.numpy as jnp

    k = X.shape[1]
    X_jax = jnp.asarray(X, dtype=jnp.float64)
    WX, WtX = W_bcoo @ X_jax, Wt_bcoo @ X_jax
    XtX = X_jax.T @ X_jax
    if spatial == "lam":
        WsX, WtWX = WX + WtX, Wt_bcoo @ WX
        U = jnp.stack([X_jax, -WsX, WtWX])
        G = jnp.stack([XtX, -(X_jax.T @ WsX), X_jax.T @ WtWX])
    else:
        U, G = jnp.stack([X_jax, -WtX]), XtX[None]
    data = {
        "X": X_jax,
        "U": U,
        "G": G,
        "kappa": jnp.asarray(y, dtype=jnp.float64) - 0.5,
        "beta_mu": jnp.broadcast_to(
            jnp.asarray(priors.beta_mu, dtype=jnp.float64), (k,)
        ),
        "beta_sigma2": jnp.broadcast_to(
            jnp.asarray(priors.beta_sigma, dtype=jnp.float64) ** 2, (k,)
        ),
        "lower": jnp.float64(getattr(priors, f"{spatial}_lower")),
        "upper": jnp.float64(getattr(priors, f"{spatial}_upper")),
        "sd_nu": jnp.float64(priors.sd_nu),
        "sd_scale": jnp.float64(priors.sd_scale),
        "logdet": None if logdet_params[0] == "closure" else logdet_params[1],
    }
    if sparsax_pattern is not None:
        data["sparsax"] = {
            "Ai": jnp.asarray(sparsax_pattern["Ai"], dtype=jnp.int32),
            "Aj": jnp.asarray(sparsax_pattern["Aj"], dtype=jnp.int32),
            "W_sym_vals": jnp.asarray(sparsax_pattern["W_sym_vals"], dtype=jnp.float64),
            "WtW_vals": jnp.asarray(sparsax_pattern["WtW_vals"], dtype=jnp.float64),
            "diag_idx": jnp.asarray(sparsax_pattern["diag_idx"], dtype=jnp.int32),
        }
    else:
        data["W_sym_dense"] = jnp.asarray(W_sym_dense, dtype=jnp.float64)
        data["WtW_dense"] = jnp.asarray(WtW_dense, dtype=jnp.float64)
    return data


# ===========================================================================
# Multi-chain runners
# ===========================================================================
#
# Chains run in parallel threads, each an ordinary ``jax.jit`` program sharing
# one compiled Gibbs step (see ``run_chains_chunked``).  ``jax.pmap`` and
# ``jax.vmap`` are both slower here; ``run_chains_in_threads`` records why.


def _logit_loglik_pointwise_jax_op(y_jax, eta):
    """Pointwise logit log-likelihood as a pure-JAX op (vmap-safe).

    The one copy of this formula that must stay JAX-native: it runs inside
    the jitted, vmapped Gibbs step.  Keep it equal to
    :func:`..logit._core._logit_loglik_pointwise`.
    """
    import jax.numpy as jnp

    return y_jax * eta - jnp.logaddexp(0.0, eta)


def _stack_chain_inits(inits, state_cls, scalar_field):
    """Convert a list of per-chain states into one vmap-able pytree.

    ``scalar_field`` names the scalar parameter (``"rho"`` or ``"lam"``)
    so the helper works for both SAR and SEM state classes.
    """
    import jax.numpy as jnp

    jax_inits = [
        state_cls(
            eta=jnp.asarray(init.eta, dtype=jnp.float64),
            beta=jnp.asarray(init.beta, dtype=jnp.float64),
            omega=jnp.asarray(init.omega, dtype=jnp.float64),
            sigma=jnp.float64(init.sigma),
            **{scalar_field: jnp.float64(getattr(init, scalar_field))},
        )
        for init in inits
    ]
    import jax

    return jax.tree.map(lambda *a: jnp.stack(a), *jax_inits)


def _cached_logit_sweep(
    name, factory, spatial, n, k, logdet_params, use_sparsax, krylov_degree,
    krylov_dmax, store_log_lik,
):  # fmt: skip
    """The chunked sweep for a logit step, one per structure (see cached_sweep)."""
    from .._utils._jax_slice import adapt_slice_width
    from .._utils._jax_utils import cached_sweep

    kind = logdet_params[0]
    closure = logdet_params[1] if kind == "closure" else None
    static = (n, k, kind, closure, use_sparsax, krylov_degree, krylov_dmax)

    def _build():
        gibbs_step = factory(
            n, k, logdet_kind=kind, logdet_closure=closure, use_sparsax=use_sparsax,
            krylov_degree=krylov_degree, krylov_dmax=krylov_dmax,
        )  # fmt: skip

        def _sweep(carry, key, tuning, data):
            # The slice starts at width 0.2, as in the NumPy sampler, and
            # adapts during warmup.
            state, width = carry
            state, _, (steps_left, steps_right) = gibbs_step(
                state, key, data, width, return_steps=True
            )
            width = adapt_slice_width(width, steps_left, steps_right, tuning)
            trace = (
                getattr(state, spatial),
                state.beta,
                state.sigma,
                state.eta @ state.eta,
            )
            if store_log_lik:
                trace += (
                    _logit_loglik_pointwise_jax_op(data["kappa"] + 0.5, state.eta),
                )
            return (state, width), trace

        return _sweep

    return cached_sweep((name, *static, bool(store_log_lik)), _build)


def run_chains_jax_vectorized(
    y: np.ndarray,
    X: np.ndarray,
    W_sparse,
    W_sym_dense,
    WtW_dense,
    logdet_jax,
    priors,
    inits: list,
    draws: int,
    tune: int,
    thin: int = 1,
    jax_seeds: list[int] | None = None,
    progressbar: bool = True,
    sparsax_pattern=None,
    krylov_degree: int = 0,
    krylov_dmax: float = 0.4,
    store_log_lik: bool = True,
    logdet_params=None,
) -> list[dict]:
    """Run multiple SAR-logit Gibbs chains in parallel.

    ``logdet_params`` (from :func:`neighbayes._logdet._jax.logdet_jax_params`)
    lets the compiled program be reused by later fits of the same structure;
    with only ``logdet_jax`` (a closure) it is compiled per fit.

    Chains run in parallel threads, each an ordinary ``jax.jit`` program, and
    the Gibbs step is compiled once (see
    :func:`.._utils._jax_utils.run_chains_chunked`).

    Parameters
    ----------
    y, X, W_sparse :
        Binary outcome, design and sparse W.
    W_sym_dense, WtW_dense :
        Dense ``W + Wᵀ`` and ``WᵀW`` for the dense path (no sparsax).
    logdet_jax :
        ``log|I − ρW|`` as a JAX closure, used when ``logdet_params`` is not
        given.
    priors : LogitREGibbsPriors
    inits : list of LogitGibbsState
        Per-chain initial states; their number is the number of chains.
    draws, tune, thin :
        Kept draws per chain, warmup sweeps and the thinning interval.
    jax_seeds : list of int, optional
        The first seeds the master PRNG key, from which every chain's keys
        are split.  Defaults to ``range(chains)``.
    progressbar : bool, default True
        Show per-chain progress bars, updated once per chunk.

    Returns
    -------
    list of dict
        One dict per chain with keys ``rho``, ``beta``, ``sigma``,
        ``eta_norm``, ``log_lik`` and ``mh_accept_rate``.  The latent field
        is not kept.
    """
    import jax
    import jax.numpy as jnp

    ensure_x64()

    chains = len(inits)
    n, k = X.shape

    W_bcoo, Wt_bcoo = _build_w_bcoo(W_sparse)
    logdet_params = logdet_params or ("closure", logdet_jax)
    data = _logit_data(
        y, X, W_bcoo, Wt_bcoo, priors, logdet_params,
        W_sym_dense=W_sym_dense, WtW_dense=WtW_dense, sparsax_pattern=sparsax_pattern,
    )  # fmt: skip

    init_states = _stack_chain_inits(inits, JAXLogitGibbsState, "rho")

    if jax_seeds is None:
        jax_seeds = list(range(chains))
    master_key = jax.random.PRNGKey(int(jax_seeds[0]))
    warmup_keys = jax.random.split(master_key, chains)

    from .._utils._jax_utils import run_chains_chunked
    from .._utils._progress import GibbsProgressBarManager

    draw_keys = jax.random.split(jax.random.fold_in(master_key, 1), chains)

    _sweep = _cached_logit_sweep(
        "sar_logit_re", partial(_make_step, sem=False), "rho", n, k, logdet_params,
        sparsax_pattern is not None, krylov_degree, krylov_dmax, store_log_lik,
    )  # fmt: skip

    with GibbsProgressBarManager(
        chains=chains,
        draws=draws,
        tune=tune,
        progressbar=progressbar,
        model_type="sar_logit",
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
                (jax.tree.map(lambda a, c=c: a[c], init_states), jnp.float64(0.2))
                for c in range(chains)
            ],
            list(warmup_keys),
            list(draw_keys),
            tune=tune,
            draws=draws,
            on_chunk=_progress,
            consts=data,
        )

    # Thin and pack as per-chain dicts
    rhos, betas, sigmas, eta_norms = traces[:4]
    log_liks = traces[4] if store_log_lik else None
    thin_slice = slice(None, None, thin) if thin > 1 else slice(None)
    results = []
    for c in range(chains):
        results.append(
            {
                "rho": rhos[c, thin_slice].copy(),
                "beta": betas[c, thin_slice].copy(),
                "sigma": sigmas[c, thin_slice].copy(),
                "eta_norm": eta_norms[c, thin_slice].copy(),
                "log_lik": log_liks[c, thin_slice].copy() if store_log_lik else None,
                "mh_accept_rate": 1.0,
            }
        )
    return results


def run_chains_jax_sem_vectorized(
    y: np.ndarray,
    X: np.ndarray,
    W_sparse,
    W_sym_dense,
    WtW_dense,
    logdet_jax,
    priors,
    inits: list,
    draws: int,
    tune: int,
    thin: int = 1,
    jax_seeds: list[int] | None = None,
    progressbar: bool = True,
    sparsax_pattern=None,
    krylov_degree: int = 0,
    krylov_dmax: float = 0.4,
    store_log_lik: bool = True,
    logdet_params=None,
) -> list[dict]:
    """Run multiple SEM-logit Gibbs chains in parallel.

    See :func:`run_chains_jax_vectorized` for the SAR-logit analogue
    and shared design rationale.  Returns per-chain dicts keyed on
    ``lam`` rather than ``rho``.
    """
    import jax
    import jax.numpy as jnp

    ensure_x64()

    chains = len(inits)
    n, k = X.shape

    W_bcoo, Wt_bcoo = _build_w_bcoo(W_sparse)
    logdet_params = logdet_params or ("closure", logdet_jax)
    data = _logit_data(
        y, X, W_bcoo, Wt_bcoo, priors, logdet_params, spatial="lam",
        W_sym_dense=W_sym_dense, WtW_dense=WtW_dense, sparsax_pattern=sparsax_pattern,
    )  # fmt: skip

    init_states = _stack_chain_inits(inits, JAXSEMLogitGibbsState, "lam")

    if jax_seeds is None:
        jax_seeds = list(range(chains))
    master_key = jax.random.PRNGKey(int(jax_seeds[0]))
    warmup_keys = jax.random.split(master_key, chains)

    from .._utils._jax_utils import run_chains_chunked
    from .._utils._progress import GibbsProgressBarManager

    draw_keys = jax.random.split(jax.random.fold_in(master_key, 1), chains)

    _sweep = _cached_logit_sweep(
        "sem_logit", partial(_make_step, sem=True), "lam", n, k, logdet_params,
        sparsax_pattern is not None, krylov_degree, krylov_dmax, store_log_lik,
    )  # fmt: skip

    with GibbsProgressBarManager(
        chains=chains,
        draws=draws,
        tune=tune,
        progressbar=progressbar,
        model_type="sem_logit",
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
                (jax.tree.map(lambda a, c=c: a[c], init_states), jnp.float64(0.2))
                for c in range(chains)
            ],
            list(warmup_keys),
            list(draw_keys),
            tune=tune,
            draws=draws,
            on_chunk=_progress,
            consts=data,
        )

    lams, betas, sigmas, eta_norms = traces[:4]
    log_liks = traces[4] if store_log_lik else None
    thin_slice = slice(None, None, thin) if thin > 1 else slice(None)
    results = []
    for c in range(chains):
        results.append(
            {
                "lam": lams[c, thin_slice].copy(),
                "beta": betas[c, thin_slice].copy(),
                "sigma": sigmas[c, thin_slice].copy(),
                "eta_norm": eta_norms[c, thin_slice].copy(),
                "log_lik": log_liks[c, thin_slice].copy() if store_log_lik else None,
                "mh_accept_rate": 1.0,
            }
        )
    return results
