r"""JAX port of the structured separable NB flow sampler (cross-section and panel).

Same sweep as :mod:`._flow_structured` — ω → ρ_d → ρ_o (rank-one block ``β_c``
integrated out) → β jointly → α — with every block compiled into one XLA program
per chunk of sweeps and the chains run thread-per-chain through
:func:`.._utils._jax_utils.run_chains_chunked`.

The host-side bookkeeping of the NumPy sampler (which design vectors each period
uses, which destination-vector pairs a rank-one Gram needs, which periods share
full-rank columns) becomes static index arrays built once here, so the traced
sweep is pure array algebra:

* ``A = L_o⁻¹U`` and ``B = L_d⁻¹V`` hold the transformed rank-one factors.
* Periods whose full-rank columns coincide share one *key*; ``B_key`` is
  ``Σ_j β_j F_j`` for that key, so a time-invariant design costs one two-sided
  solve per candidate regardless of ``T``.
* Each ρ candidate is one sparse LU solve whose right-hand side stacks every
  key's ``n × n`` array and the rank-one factors side by side.
* Per-period terms run under ``lax.map``, so temporaries stay ``O(n²)`` rather
  than ``O(T n²)``.
"""

from __future__ import annotations

import numpy as np

from ._flow_jax import _build_sar_solver_jax
from ._flow_structured import FlowDesignStructure, classify_flow_design


def _cheap_pair_slots(struct: FlowDesignStructure):
    """Per-period destination-vector pairs for the rank-one Gram.

    Entry ``(i, j)`` (``i ≤ j``) of period ``t``'s rank-one Gram is
    ``Σ_o a_i a_j (Ω (b_p ∘ b_q))_o`` with ``(p, q)`` the destination vectors of
    columns ``i`` and ``j``.  Many column pairs share ``(p, q)`` (the intercept
    and every origin attribute have ``v = 1``), so each period's distinct pairs
    are listed once, padded to a common length, and ``slot[t, m]`` maps upper-
    triangle entry ``m`` to its pair.
    """
    k_c = len(struct.cheap_cols)
    I, J = np.triu_indices(k_c)
    per_t = []
    for t in range(struct.T):
        vi = struct.v_idx[t]
        pairs = [tuple(sorted((int(vi[i]), int(vi[j])))) for i, j in zip(I, J)]
        distinct = sorted(set(pairs))
        per_t.append((distinct, [distinct.index(pq) for pq in pairs]))
    p_max = max(len(d) for d, _ in per_t)
    P = np.zeros((struct.T, p_max), dtype=np.int32)
    Q = np.zeros((struct.T, p_max), dtype=np.int32)
    slot = np.zeros((struct.T, len(I)), dtype=np.int32)
    for t, (distinct, s) in enumerate(per_t):
        P[t, : len(distinct)] = [p for p, _ in distinct]
        Q[t, : len(distinct)] = [q for _, q in distinct]
        slot[t] = s
    return I.astype(np.int32), J.astype(np.int32), P, Q, slot


def make_structured_sweep(y, W_csc, struct: FlowDesignStructure, priors):
    """Build ``sweep(state, key, tuning, data) -> (state, trace)`` for one chain.

    Returns ``(sweep, initial_eta, data)``; ``data`` holds the data-sized arrays
    both functions take as an argument.

    ``state`` holds ``beta, rho_d, rho_o, alpha``, the current ``eta`` (``T × n ×
    n``) and the two ρ slice widths; ``trace`` is ``(rho_d, rho_o, beta, alpha,
    eta)``.
    """
    import jax
    import jax.numpy as jnp
    from jax.scipy.linalg import solve_triangular
    from jax.scipy.special import gammaln

    from .._utils._jax_slice import adapt_slice_width, jax_slice_sample_1d
    from .._utils._jax_utils import make_pg_draw

    n, T, k = struct.n, struct.T, struct.k
    cc = np.asarray(struct.cheap_cols, dtype=np.int32)
    fc = np.asarray(struct.full_cols, dtype=np.int32)
    k_c, k_e = len(cc), len(fc)

    # --- priors ---
    mu0 = np.broadcast_to(np.asarray(priors.beta_mu, dtype=np.float64), (k,)).copy()
    sd0 = np.broadcast_to(np.asarray(priors.beta_sigma, dtype=np.float64), (k,))
    prec0 = 1.0 / sd0**2
    mu0_j, prec0_j = jnp.asarray(mu0), jnp.asarray(prec0)
    mu_c, prec_c = jnp.asarray(mu0[cc]), jnp.asarray(prec0[cc])
    rho_lo, rho_hi = float(priors.rho_lower), float(priors.rho_upper)
    alpha_sigma, alpha_nu = float(priors.alpha_sigma), float(priors.alpha_nu)

    # --- data and static structure ---
    # Arrays of data size enter the compiled sweep as arguments (``data``), not
    # closure constants: XLA embeds and constant-folds closed-over arrays, which
    # cost gigabytes at compile time on large panels.
    y_np = np.asarray(y, dtype=np.float64)
    y_vals, y_counts = np.unique(y_np, return_counts=True)
    y_vals, y_counts = jnp.asarray(y_vals), jnp.asarray(y_counts, dtype=jnp.float64)
    n_obs = float(y_np.size)

    ui = jnp.asarray(struct.u_idx, dtype=jnp.int32)  # (T, k_c)
    vi = jnp.asarray(struct.v_idx, dtype=jnp.int32)
    tI, tJ, pP, pQ, slot = _cheap_pair_slots(struct) if k_c else (None,) * 5
    if k_c:
        tI, tJ = jnp.asarray(tI), jnp.asarray(tJ)
        pP, pQ, slot = jnp.asarray(pP), jnp.asarray(pQ), jnp.asarray(slot)

    data = {
        "y": jnp.asarray(y_np),
        "F": jnp.asarray(np.stack(struct.F)) if k_e else jnp.zeros((0, n, n)),
        "U": jnp.asarray(struct.U),  # (n, n_u)
        "V": jnp.asarray(struct.V),  # (n, n_v)
    }
    n_F = data["F"].shape[0]
    f_idx = jnp.asarray(struct.f_idx, dtype=jnp.int32)  # (T, k_e)
    keys = sorted({tuple(int(i) for i in struct.f_idx[t]) for t in range(T)})
    n_keys = len(keys)
    key_of_t = jnp.asarray(
        [keys.index(tuple(int(i) for i in struct.f_idx[t])) for t in range(T)],
        dtype=jnp.int32,
    )
    key_idx = jnp.asarray(np.asarray(keys, dtype=np.int32).reshape(n_keys, k_e))

    solve = _build_sar_solver_jax(W_csc, n)
    draw_pg = make_pg_draw()
    period_idx = jnp.arange(T)

    # --- helpers ---
    def key_arrays(C, arrays):
        """``Σ_i C[key, i] arrays[i]`` for each key: ``(n_keys, n, n)``."""
        return jnp.tensordot(C, arrays, axes=1)

    def key_coefs(b_full):
        """``(n_keys, n_F)`` coefficient matrix of ``B_key = Σ_j β_j F_{key_j}``."""
        rows = jnp.repeat(jnp.arange(n_keys), k_e)
        return (
            jnp.zeros((n_keys, n_F))
            .at[rows, key_idx.ravel()]
            .add(jnp.tile(b_full, n_keys))
        )

    def solve_left(rho, stack, extra):
        """``L⁻¹ S_i`` for every ``n × n`` slice of ``stack`` plus ``L⁻¹ extra``."""
        m = stack.shape[0]
        rhs = jnp.concatenate(
            [stack.transpose(1, 0, 2).reshape(n, m * n), extra], axis=1
        )
        out = solve(rho, rhs)
        return out[:, : m * n].reshape(n, m, n).transpose(1, 0, 2), out[:, m * n :]

    def solve_right(rho, stack, extra):
        """``S_i L⁻ᵀ`` for every slice of ``stack`` plus ``L⁻¹ extra``."""
        m = stack.shape[0]
        rhs = jnp.concatenate(
            [stack.transpose(2, 0, 1).reshape(n, m * n), extra], axis=1
        )
        out = solve(rho, rhs)
        return out[:, : m * n].reshape(n, m, n).transpose(1, 2, 0), out[:, m * n :]

    def cheap_terms(Om, R, Ac, Bc, B, t):
        """Period ``t``'s rank-one Gram ``U_cᵀΩU_c`` and cross ``U_cᵀΩR``."""
        Bprod = B[:, pP[t]] * B[:, pQ[t]]
        OmB = Om @ Bprod  # (n, p_max)
        a2 = Ac[:, tI] * Ac[:, tJ]
        vals = jnp.sum(a2 * OmB[:, slot[t]], axis=0)
        M = jnp.zeros((k_c, k_c)).at[tI, tJ].set(vals).at[tJ, tI].set(vals)
        v = jnp.sum(Ac * ((Om * R) @ Bc), axis=0)
        return M, v

    def rho_log_density(Om_all, Z_all, eta_f, A, B):
        """``log p(ρ | ω, α, β_full, y)`` with ``β_c`` integrated out."""

        def period(t):
            R = Z_all[t] - eta_f[key_of_t[t]]
            if k_c:
                Ac, Bc = A[:, ui[t]], B[:, vi[t]]
                R = R - (Ac * mu_c) @ Bc.T
                M, v = cheap_terms(Om_all[t], R, Ac, Bc, B, t)
            else:
                M, v = jnp.zeros((0, 0)), jnp.zeros(0)
            return M, v, jnp.sum(Om_all[t] * R * R)

        M, v, quad = jax.lax.map(period, period_idx)
        quad = quad.sum()
        if not k_c:
            return -0.5 * quad
        L = jnp.linalg.cholesky(jnp.diag(prec_c) + M.sum(0))
        w = solve_triangular(L, v.sum(0), lower=True)
        val = -jnp.sum(jnp.log(jnp.diag(L))) - 0.5 * (quad - w @ w)
        return jnp.where(jnp.isfinite(val), val, -jnp.inf)

    def beta_gram(Om_all, Z_all, Ufull, A, B):
        """``G = Σ_t U_tᵀΩ_tU_t`` and ``h = Σ_t U_tᵀΩ_t z_t``."""

        def period(t):
            Om, Z = Om_all[t], Z_all[t]
            G = jnp.zeros((k, k))
            h = jnp.zeros(k)
            if k_c:
                Ac, Bc = A[:, ui[t]], B[:, vi[t]]
                M, v = cheap_terms(Om, Z, Ac, Bc, B, t)
                G = G.at[cc[:, None], cc[None, :]].add(M)
                h = h.at[cc].add(v)
            if k_e:
                Uf = Ufull[f_idx[t]]  # (k_e, n, n)
                OU = Om[None] * Uf
                G = G.at[fc[:, None], fc[None, :]].add(
                    jnp.einsum("iod,jod->ij", OU, Uf)
                )
                h = h.at[fc].add(jnp.einsum("iod,od->i", OU, Z))
                if k_c:
                    cross = jnp.einsum("iod,dj,oj->ij", OU, Bc, Ac)
                    G = G.at[fc[:, None], cc[None, :]].add(cross)
                    G = G.at[cc[:, None], fc[None, :]].add(cross.T)
            return G, h

        G, h = jax.lax.map(period, period_idx)
        return G.sum(0), h.sum(0)

    def eta_of(beta, Ufull, A, B):
        eta_key = key_arrays(key_coefs(beta[fc]), Ufull)

        def period(t):
            e = eta_key[key_of_t[t]]
            if k_c:
                e = e + (A[:, ui[t]] * beta[cc]) @ B[:, vi[t]].T
            return e

        return jax.lax.map(period, period_idx)

    def alpha_log_density(log_a, y_j, y_dot_eta, mu):
        a = jnp.exp(log_a)
        log_mu_a = jnp.log(mu + a)
        ll = (
            y_counts @ gammaln(y_vals + a)
            - n_obs * gammaln(a)
            + y_dot_eta
            - y_j @ log_mu_a
            - a * log_mu_a.sum()
            + n_obs * a * log_a
        )
        prior = -0.5 * (alpha_nu + 1.0) * jnp.log1p(a * a / (alpha_nu * alpha_sigma**2))
        return log_a + ll + prior

    def sweep(state, key, tuning, data):
        y_j, F, U, V = data["y"], data["F"], data["U"], data["V"]
        beta, rd, ro, alpha = (
            state["beta"],
            state["rho_d"],
            state["rho_o"],
            state["alpha"],
        )
        w_d, w_o = state["slice_widths"]
        k_pg, k_d, k_o, k_b, k_a = jax.random.split(key, 5)

        # ω and the working response, all periods
        eta_flat = state["eta"].ravel()
        omega = draw_pg(
            jnp.maximum(y_j + alpha, 1e-3),
            jnp.clip(eta_flat - jnp.log(alpha), -30.0, 30.0),
            k_pg,
        )
        Om_all = omega.reshape(T, n, n)
        Z_all = (0.5 * (y_j - alpha) / omega + jnp.log(alpha)).reshape(T, n, n)
        B_keys = key_arrays(key_coefs(beta[fc]), F)

        # ρ_d: origin side fixed at ρ_o
        P, A_fix = solve_left(ro, B_keys, U)

        def ld_d(rv):
            eta_f, Bv = solve_right(rv, P, V)
            return rho_log_density(Om_all, Z_all, eta_f, A_fix, Bv)

        rd, _, sl_d, sr_d = jax_slice_sample_1d(
            ld_d, rd, rho_lo, rho_hi, key=k_d, w=w_d, return_steps=True
        )

        # ρ_o: destination side fixed at the new ρ_d
        P, B_fix = solve_right(rd, B_keys, V)

        def ld_o(rv):
            eta_f, Av = solve_left(rv, P, U)
            return rho_log_density(Om_all, Z_all, eta_f, Av, B_fix)

        ro, _, sl_o, sr_o = jax_slice_sample_1d(
            ld_o, ro, rho_lo, rho_hi, key=k_o, w=w_o, return_steps=True
        )

        # β = (β_c, β_full) jointly
        half, A = solve_left(ro, F, U)
        Ufull, B = solve_right(rd, half, V)
        G, h = beta_gram(Om_all, Z_all, Ufull, A, B)
        Lg = jnp.linalg.cholesky(G + jnp.diag(prec0_j))
        mean = jax.scipy.linalg.cho_solve((Lg, True), h + prec0_j * mu0_j)
        beta = mean + solve_triangular(
            Lg.T, jax.random.normal(k_b, (k,), dtype=jnp.float64), lower=False
        )
        eta = eta_of(beta, Ufull, A, B)

        # α on log α
        eta_flat = eta.ravel()
        y_dot_eta, mu = y_j @ eta_flat, jnp.exp(eta_flat)
        log_alpha, _ = jax_slice_sample_1d(
            lambda la: alpha_log_density(la, y_j, y_dot_eta, mu),
            jnp.log(alpha),
            -10.0,
            10.0,
            key=k_a,
            w=0.5,
        )
        alpha = jnp.exp(log_alpha)

        widths = (
            adapt_slice_width(w_d, sl_d, sr_d, tuning),
            adapt_slice_width(w_o, sl_o, sr_o, tuning),
        )
        new = {
            "beta": beta,
            "rho_d": rd,
            "rho_o": ro,
            "alpha": alpha,
            "eta": eta,
            "slice_widths": widths,
        }
        return new, (rd, ro, beta, alpha, eta)

    def initial_eta(beta, rd, ro, data):
        half, A = solve_left(ro, data["F"], data["U"])
        Ufull, B = solve_right(rd, half, data["V"])
        return eta_of(beta, Ufull, A, B)

    # The blocks, for tests that pin them against the NumPy sampler's algebra.
    sweep.kernels = {
        "rho_log_density": rho_log_density,
        "beta_gram": beta_gram,
        "solve_left": solve_left,
        "solve_right": solve_right,
    }
    return sweep, jax.jit(initial_eta), data


def run_chains_jax_flow_structured(
    y,
    W_csc,
    n,
    priors,
    inits,
    draws,
    tune,
    *,
    X=None,
    T=1,
    struct: FlowDesignStructure | None = None,
    thin=1,
    jax_seeds=None,
    slice_width=0.4,
    store_log_lik=False,
):
    """Run the structured separable NB flow sampler on the JAX backend.

    ``y`` (and ``X``, unless ``struct`` is given) are stacked time-first over
    ``T`` periods of ``n²`` flows.  Returns one dict per chain with keys
    ``rho_d, rho_o, rho_w, beta, alpha, log_lik``.
    """
    import jax
    import jax.numpy as jnp
    from scipy.special import gammaln

    from ..._jax_dispatch import ensure_x64
    from .._utils._jax_utils import run_chains_chunked
    from .._utils._sparsax_lu import set_sparsax_lu_cache_size

    ensure_x64()
    struct = classify_flow_design(X, n, T) if struct is None else struct
    chains = len(inits)
    if jax_seeds is None:
        jax_seeds = list(range(chains))
    set_sparsax_lu_cache_size(max(32, 8 * chains))

    sweep, initial_eta, data = make_structured_sweep(y, W_csc, struct, priors)
    if store_log_lik:
        trace_sweep = sweep
    else:

        def trace_sweep(state, key, tuning, data):  # η is traced only for log_lik
            new, trace = sweep(state, key, tuning, data)
            return new, trace[:4]

    states = []
    for i in inits:
        beta = jnp.asarray(i.beta, dtype=jnp.float64)
        rd, ro = jnp.float64(float(i.rho_d)), jnp.float64(float(i.rho_o))
        states.append(
            {
                "beta": beta,
                "rho_d": rd,
                "rho_o": ro,
                "alpha": jnp.float64(float(i.alpha)),
                "eta": initial_eta(beta, rd, ro, data),
                "slice_widths": (jnp.float64(slice_width),) * 2,
            }
        )
    warm_keys = [jax.random.PRNGKey(int(s)) for s in jax_seeds]
    draw_keys = [jax.random.fold_in(jax.random.PRNGKey(int(s)), 1) for s in jax_seeds]
    _, traces = run_chains_chunked(
        trace_sweep, states, warm_keys, draw_keys, tune=tune, draws=draws, consts=data
    )

    sl = slice(None, None, thin) if thin > 1 else slice(None)
    rd_all, ro_all, beta_all, alpha_all = (a[:, sl] for a in traces[:4])
    y_np = np.asarray(y, dtype=np.float64)
    results = []
    for c in range(chains):
        log_lik = None
        if store_log_lik:
            eta = traces[4][c, sl].reshape(len(alpha_all[c]), -1)
            mu = np.exp(np.clip(eta, -30.0, 30.0))
            a = alpha_all[c][:, None]
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
                "rho_w": -rd_all[c] * ro_all[c],
                "beta": beta_all[c],
                "alpha": alpha_all[c],
                "log_lik": log_lik,
            }
        )
    return results
