r"""JAX port of the structured separable NB flow sweep with fixed effects.

Same sweep as :mod:`._flow_structured_fe` (read it first) — ω → ρ_d → ρ_o
(``β_c``, period and pair effects integrated out) → ``(β, τ)`` jointly → pair
effects ``C`` → α — compiled into one XLA program per chunk of sweeps and run
thread-per-chain through :func:`.._utils._jax_utils.run_chains_chunked`, as the
pooled port :mod:`._flow_structured_jax` is.

Per-period terms run under ``lax.scan``/``lax.map`` so temporaries stay
``O(k n²)``; the ω-weighted group sums ``S(x) = Σ_t Ω_t ⊙ x_t`` ride the scan
carry.  When per-draw pair effects would not fit in memory the chain keeps
their running mean and variance in its state instead (Welford, draws only).
"""

from __future__ import annotations

import numpy as np

from ._flow_structured import FlowDesignStructure, classify_flow_design


def _equation_parts(
    W_csc, struct: FlowDesignStructure, beta_mu, beta_sigma, *, rho_bounds,
    pair_effects=None, period_effects=None, prefix: str = "",
):  # fmt: skip
    """Host side of one structured equation: ``(static, data)``.

    ``static`` is what the compiled kernels close over — sizes, the design's
    column split, the effect switches, the routed LU function and the key
    prefix — and keys their cache; ``data`` (keys prefixed) is every
    data-dependent array and scalar, which the kernels take as an argument.
    """
    import jax.numpy as jnp

    from ._flow_jax import _sar_solver_parts

    n, T, k = struct.n, struct.T, struct.k
    cc = np.asarray(struct.cheap_cols, dtype=np.int32)
    fc = np.asarray(struct.full_cols, dtype=np.int32)
    k_e = len(fc)
    mu0 = np.broadcast_to(np.asarray(beta_mu, dtype=np.float64), (k,)).copy()
    sd0 = np.broadcast_to(np.asarray(beta_sigma, dtype=np.float64), (k,))
    has_pairs = pair_effects is not None
    m_pair, s_pair = (
        (float(pair_effects[0]), float(pair_effects[1])) if has_pairs else (0.0, 1.0)
    )
    # A third element learns the pair-effect sd (half-t(3, scale) prior); the
    # sd then rides in the equation state as ``s_pair``.
    sd_scale = (
        float(pair_effects[2])
        if has_pairs and len(pair_effects) > 2 and pair_effects[2] is not None
        else None
    )
    mu_tau, s_tau, n_tau = period_effects if period_effects is not None else (0, 1, 0)
    lu_solve, sar = _sar_solver_parts(W_csc, n)
    raw = {
        "F": jnp.asarray(np.stack(struct.F)) if k_e else jnp.zeros((0, n, n)),
        "U": jnp.asarray(struct.U),
        "V": jnp.asarray(struct.V),
        "mu0": jnp.asarray(mu0),
        "prec0": jnp.asarray(1.0 / sd0**2),
        "rho_lo": jnp.float64(rho_bounds[0]),
        "rho_hi": jnp.float64(rho_bounds[1]),
        "m_pair": jnp.float64(m_pair),
        "s_pair": jnp.float64(s_pair),
        "sd_scale": jnp.float64(sd_scale if sd_scale is not None else 1.0),
        "mu_tau": jnp.float64(mu_tau),
        "s_tau": jnp.float64(s_tau),
        "ui": jnp.asarray(struct.u_idx, dtype=jnp.int32),
        "vi": jnp.asarray(struct.v_idx, dtype=jnp.int32),
        "f_idx": jnp.asarray(struct.f_idx, dtype=jnp.int32),
        "sar": sar,
    }
    data = {prefix + name: value for name, value in raw.items()}
    static = (
        n, T, k, tuple(int(c) for c in cc), tuple(int(c) for c in fc), has_pairs,
        sd_scale is not None, int(n_tau), lu_solve, prefix,
    )  # fmt: skip
    return static, data


def _equation_fns(n, T, k, cc_t, fc_t, has_pairs, learn_sd, n_tau, lu_solve, prefix):
    """One structured equation's kernels for one structure, each taking ``data``."""
    import jax
    import jax.numpy as jnp
    from jax.scipy.linalg import solve_triangular

    from .._utils._jax_slice import adapt_slice_width, jax_slice_sample_1d
    from ._flow_jax import _bind_sar_solver

    cc = np.asarray(cc_t, dtype=np.int32)
    fc = np.asarray(fc_t, dtype=np.int32)
    k_c, k_e = len(cc), len(fc)
    order = np.concatenate([cc, fc]).astype(np.int32)
    t0 = T - n_tau
    kF, kU, kV = prefix + "F", prefix + "U", prefix + "V"

    def bind(data):
        def pre(name):
            return data[prefix + name]

        mu0, prec0 = pre("mu0"), pre("prec0")
        mu_c, prec_c = mu0[cc], prec0[cc]
        mu_o, prec_o = mu0[order], prec0[order]
        rho_lo, rho_hi = pre("rho_lo"), pre("rho_hi")
        m_pair, s_pair, sd_scale = pre("m_pair"), pre("s_pair"), pre("sd_scale")
        mu_tau, s_tau = pre("mu_tau"), pre("s_tau")
        prec_tau = jnp.ones(n_tau) / s_tau**2
        in_tau = jnp.asarray(np.arange(T) >= t0)
        ui, vi, f_idx = pre("ui"), pre("vi"), pre("f_idx")
        solve = _bind_sar_solver(lu_solve, pre("sar"))
        periods = jnp.arange(T)

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

        def cheap_arrays(A, B, t):
            return A[:, ui[t]].T[:, :, None] * B[:, vi[t]].T[:, None, :]

        def moments(cols_of, resid_of, kx, prec_x, Om_all, D, OmOm_tau):
            r"""Collapsed ``(M, v, quad)`` over ``[x-cols, τ]`` (see ``_gram_block``)."""
            K = kx + n_tau

            def step(carry, t):
                Mx, vx, quad, SR, S = carry
                Om, R = Om_all[t], resid_of(t)
                OmR = Om * R
                quad = quad + jnp.sum(OmR * R)
                if kx:
                    C = cols_of(t)
                    OmC = Om[None] * C
                    Mx = Mx + jnp.tensordot(OmC, C, axes=([1, 2], [1, 2]))
                    vx = vx + jnp.tensordot(OmC, R, axes=2)
                    cross = OmC.sum(axis=(1, 2))
                    if has_pairs:
                        S = S + OmC
                else:
                    cross = jnp.zeros(0)
                if has_pairs:
                    SR = SR + OmR
                return (Mx, vx, quad, SR, S), (cross, Om.sum(), OmR.sum())

            init = (
                jnp.zeros((kx, kx)),
                jnp.zeros(kx),
                jnp.float64(0.0),
                jnp.zeros((n, n)),
                jnp.zeros((kx, n, n)) if (has_pairs and kx) else jnp.zeros((0, n, n)),
            )
            (Mx, vx, quad, SR, S), (cross, om_sum, omr_sum) = jax.lax.scan(
                step, init, periods
            )
            M = jnp.zeros((K, K)).at[:kx, :kx].set(Mx)
            v = jnp.zeros(K).at[:kx].set(vx)
            if n_tau:
                tau_ix = jnp.arange(kx, K)
                M = M.at[tau_ix, tau_ix].add(om_sum[t0:])
                v = v.at[kx:].add(omr_sum[t0:])
                if kx:
                    M = M.at[:kx, kx:].add(cross[t0:].T).at[kx:, :kx].add(cross[t0:])
            if has_pairs:
                SRd = SR / D
                quad = quad - jnp.sum(SR * SRd)
                if kx:
                    Sd = S / D[None]
                    M = M.at[:kx, :kx].add(-jnp.tensordot(Sd, S, axes=([1, 2], [1, 2])))
                    v = v.at[:kx].add(-jnp.tensordot(S, SRd, axes=2))
                if n_tau:
                    Om_tau = Om_all[t0:]
                    M = M.at[kx:, kx:].add(-OmOm_tau)
                    v = v.at[kx:].add(-jnp.tensordot(Om_tau, SRd, axes=2))
                    if kx:
                        cr = jax.lax.map(
                            lambda Om: jnp.tensordot(Sd, Om, axes=2), Om_tau
                        )  # (n_tau, kx)
                        M = M.at[:kx, kx:].add(-cr.T).at[kx:, :kx].add(-cr)
            M = M + jnp.diag(jnp.concatenate([prec_x, prec_tau]))
            return M, v, quad

        def collapsed_density(M, v, quad):
            if M.shape[0] == 0:
                return -0.5 * quad
            L = jnp.linalg.cholesky(M)
            w = solve_triangular(L, v, lower=True)
            val = -jnp.sum(jnp.log(jnp.diag(L))) - 0.5 * (quad - w @ w)
            return jnp.where(jnp.isfinite(val), val, -jnp.inf)

        def key_sum(b_full, F):
            """``Σ_j β_j F_j`` per period, ``(T, n, n)``."""
            return jnp.tensordot(b_full, F[f_idx], axes=([0], [1])) if k_e else None

        def eta_of(beta, tau, C, Ufull, A, B):
            def period(t):
                e = jnp.zeros((n, n))
                if k_e:
                    e = e + jnp.tensordot(beta[fc], Ufull[f_idx[t]], axes=1)
                if k_c:
                    e = e + (A[:, ui[t]] * beta[cc]) @ B[:, vi[t]].T
                if n_tau:
                    e = e + jnp.where(in_tau[t], tau[jnp.maximum(t - t0, 0)], 0.0)
                return e + C

            return jax.lax.map(period, periods)

        def update(eq, Om_all, Z_all, key, tuning):
            F, U, V = data[kF], data[kU], data[kV]
            beta, rd, ro = eq["beta"], eq["rho_d"], eq["rho_o"]
            w_d, w_o = eq["slice_widths"]
            if learn_sd:
                k_d, k_o, k_b, k_c_, k_s, k_s2 = jax.random.split(key, 6)
            else:
                k_d, k_o, k_b, k_c_ = jax.random.split(key, 4)
            s_cur = eq["s_pair"] if learn_sd else s_pair

            Zc = Z_all - m_pair
            D = Om_all.sum(axis=0) + 1.0 / s_cur**2 if has_pairs else None
            OmOm_tau = None
            if has_pairs and n_tau:
                OmD = Om_all[t0:].reshape(n_tau, -1) / jnp.sqrt(D).ravel()[None, :]
                OmOm_tau = OmD @ OmD.T
            B_full = key_sum(beta[fc], F)
            tau_shift = jnp.where(in_tau, mu_tau, 0.0)

            def rho_density(eta_f, Av, Bv):
                def resid(t):
                    R = Zc[t] - tau_shift[t]
                    if k_e:
                        R = R - eta_f[t]
                    if k_c:
                        R = R - (Av[:, ui[t]] * mu_c) @ Bv[:, vi[t]].T
                    return R

                M, v, quad = moments(
                    lambda t: cheap_arrays(Av, Bv, t),
                    resid,
                    k_c,
                    prec_c,
                    Om_all,
                    D,
                    OmOm_tau,
                )
                return collapsed_density(M, v, quad)

            empty = jnp.zeros((0, n, n))
            P, A_fix = solve_left(ro, B_full if k_e else empty, U)

            def ld_d(rv):
                eta_f, Bv = solve_right(rv, P, V)
                return rho_density(eta_f, A_fix, Bv)

            rd, _, sl_d, sr_d = jax_slice_sample_1d(
                ld_d, rd, rho_lo, rho_hi, key=k_d, w=w_d, return_steps=True
            )
            P, B_fix = solve_right(rd, B_full if k_e else empty, V)

            def ld_o(rv):
                eta_f, Av = solve_left(rv, P, U)
                return rho_density(eta_f, Av, B_fix)

            ro, _, sl_o, sr_o = jax_slice_sample_1d(
                ld_o, ro, rho_lo, rho_hi, key=k_o, w=w_o, return_steps=True
            )

            half, A = solve_left(ro, F, U)
            Ufull, B = solve_right(rd, half, V)

            def cols(t):
                parts = []
                if k_c:
                    parts.append(cheap_arrays(A, B, t))
                if k_e:
                    parts.append(Ufull[f_idx[t]])
                return jnp.concatenate(parts) if parts else jnp.zeros((0, n, n))

            M, h, _ = moments(cols, lambda t: Zc[t], k, prec_o, Om_all, D, OmOm_tau)
            h = h + jnp.concatenate([prec_o * mu_o, prec_tau * mu_tau])
            Lg = jnp.linalg.cholesky(M)
            mean = jax.scipy.linalg.cho_solve((Lg, True), h)
            theta = mean + solve_triangular(
                Lg.T, jax.random.normal(k_b, mean.shape, dtype=jnp.float64), lower=False
            )
            beta = jnp.zeros(k).at[order].set(theta[:k])
            tau = theta[k:]
            eta = eta_of(beta, tau, jnp.zeros((n, n)), Ufull, A, B)
            C = eq["C"]
            if has_pairs:
                S = jnp.sum(Om_all * (Zc - eta), axis=0)
                C = (
                    m_pair
                    + S / D
                    + jax.random.normal(k_c_, (n, n), dtype=jnp.float64) / jnp.sqrt(D)
                )
                eta = eta + C
            out = {
                "beta": beta,
                "tau": tau,
                "C": C,
                "rho_d": rd,
                "rho_o": ro,
                "eta": eta,
                "slice_widths": (
                    adapt_slice_width(w_d, sl_d, sr_d, tuning),
                    adapt_slice_width(w_o, sl_o, sr_o, tuning),
                ),
            }
            if learn_sd:
                # Centred σ | C, then non-centred σ | C̃ (interweaving).
                dev2 = jnp.sum((C - m_pair) ** 2)
                nu, g = 3.0, float(n * n)

                def log_prior(sd):
                    return -0.5 * (nu + 1.0) * jnp.log1p(sd * sd / (nu * sd_scale**2))

                def sd_density(ls):
                    return (
                        -g * ls
                        - 0.5 * dev2 / jnp.exp(2.0 * ls)
                        + log_prior(jnp.exp(ls))
                        + ls
                    )

                ls, _ = jax_slice_sample_1d(
                    sd_density, jnp.log(s_cur), -10.0, 5.0, key=k_s, w=0.5
                )
                ct = (C - m_pair) / jnp.exp(ls)
                eta_rest = eta - C
                OmC = Om_all * ct[None]
                P = jnp.sum(OmC * ct[None])
                hh = jnp.sum(OmC * (Zc - eta_rest))

                def nc_density(ls2):
                    sd = jnp.exp(ls2)
                    return -0.5 * P * sd * sd + hh * sd + log_prior(sd) + ls2

                ls2, _ = jax_slice_sample_1d(
                    nc_density, ls, -10.0, 5.0, key=k_s2, w=0.5
                )
                sd = jnp.exp(ls2)
                C = m_pair + sd * ct
                out["C"] = C
                out["eta"] = eta_rest + C
                out["s_pair"] = sd
            return out

        def initial_eta(beta, tau, C, rd, ro):
            half, A = solve_left(ro, data[kF], data[kU])
            Ufull, B = solve_right(rd, half, data[kV])
            return eta_of(beta, tau, C, Ufull, A, B)

        def exact_sd_move(eq, loglik, key):
            """Non-centred pair-effect sd on the exact likelihood ``loglik(eta)``.

            ``C̃ = (C − m)/σ`` held; returns ``eq`` unchanged unless the sd is
            learned.  See :func:`.._utils._group_effects.exact_noncentered_sd`.
            """
            if not learn_sd:
                return eq
            s = eq["s_pair"]
            ct = (eq["C"] - m_pair) / s
            rest = eq["eta"] - eq["C"][None] + m_pair
            nu = 3.0

            def density(ls):
                sd = jnp.exp(ls)
                val = (
                    loglik(rest + sd * ct[None])
                    - 0.5 * (nu + 1.0) * jnp.log1p(sd * sd / (nu * sd_scale**2))
                    + ls
                )
                return jnp.where(jnp.isfinite(val), val, -jnp.inf)

            ls, _ = jax_slice_sample_1d(density, jnp.log(s), -10.0, 5.0, key=key, w=0.3)
            sd = jnp.exp(ls)
            out = dict(eq)
            out.update(s_pair=sd, C=m_pair + sd * ct, eta=rest + sd * ct[None])
            return out

        def level_vector(rd, ro):
            """The filtered constant column ``(L_o⁻¹1)(L_d⁻¹1)ᵀ``, ``(n, n)``."""
            ones = jnp.ones((n, 1))
            return jnp.outer(solve(ro, ones)[:, 0], solve(rd, ones)[:, 0])

        return {
            "update": update,
            "initial_eta": initial_eta,
            "exact_sd_move": exact_sd_move,
            "level_vector": level_vector,
            "moments": moments,
        }

    def update(eq, Om_all, Z_all, key, tuning, data):
        return bind(data)["update"](eq, Om_all, Z_all, key, tuning)

    def initial_eta(beta, tau, C, rd, ro, data):
        return bind(data)["initial_eta"](beta, tau, C, rd, ro)

    def exact_sd_move(eq, loglik, key, data):
        return bind(data)["exact_sd_move"](eq, loglik, key)

    def level_vector(rd, ro, data):
        return bind(data)["level_vector"](rd, ro)

    from types import SimpleNamespace

    return SimpleNamespace(
        update=update,
        initial_eta=initial_eta,
        initial_eta_jit=jax.jit(initial_eta),
        exact_sd_move=exact_sd_move,
        level_vector=level_vector,
        bind=bind,
    )


def make_structured_equation(
    W_csc,
    struct: FlowDesignStructure,
    beta_mu,
    beta_sigma,
    *,
    rho_bounds,
    pair_effects=None,
    period_effects=None,
    prefix: str = "",
):
    """One separable flow equation for the compiled sweeps.

    The JAX twin of
    :class:`~neighbayes.samplers.negbin_reduced._flow_structured_fe.StructuredEquation`.
    Returns a namespace with

    * ``data`` — the equation's data-dependent arrays and scalars (each key
      prefixed by ``prefix``), to merge into the sweep's ``consts``;
    * ``update(eq, Om_all, Z_all, key, tuning, data) -> eq`` — draw ρ_d, ρ_o,
      then ``(β, τ)``, then ``C`` from ``(T, n, n)`` working data, where ``eq``
      holds ``beta, tau, C, rho_d, rho_o, eta, slice_widths``;
    * ``initial_eta(beta, tau, C, rd, ro, data)`` (compiled),
      ``exact_sd_move(eq, loglik, key, data)`` and
      ``level_vector(rd, ro, data)``;
    * ``moments`` (bound to this data, for tests), ``static`` (the cache key of
      the kernels) and the host-side prior values.

    The kernels close over the equation's structure only and are shared by
    every equation of the same structure, so a sweep built on them compiles
    once per structure.
    """
    from .._utils._jax_utils import cached_sweep

    static, data = _equation_parts(
        W_csc, struct, beta_mu, beta_sigma, rho_bounds=rho_bounds,
        pair_effects=pair_effects, period_effects=period_effects, prefix=prefix,
    )  # fmt: skip
    fns = cached_sweep(("flow_equation", *static), lambda: _equation_fns(*static))
    k = struct.k
    has_pairs = pair_effects is not None
    m_pair, s_pair = (
        (float(pair_effects[0]), float(pair_effects[1])) if has_pairs else (0.0, 1.0)
    )
    mu_tau, s_tau, n_tau = period_effects if period_effects is not None else (0, 1, 0)
    mu0 = np.broadcast_to(np.asarray(beta_mu, dtype=np.float64), (k,)).copy()
    sd0 = np.broadcast_to(np.asarray(beta_sigma, dtype=np.float64), (k,))

    from types import SimpleNamespace

    return SimpleNamespace(
        data=data,
        update=fns.update,
        initial_eta=fns.initial_eta_jit,
        level_vector=fns.level_vector,
        exact_sd_move=fns.exact_sd_move,
        moments=fns.bind(data)["moments"],
        static=static,
        k=k,
        n_tau=n_tau,
        mu_tau=float(mu_tau),
        s_tau=float(s_tau),
        m_pair=float(m_pair),
        s_pair=float(s_pair),
        learn_sd=static[6],
        mu0=mu0,
        prec0=1.0 / sd0**2,
        has_pairs=has_pairs,
    )


def welford_update(state: dict, C, tuning) -> dict:
    """Running mean and M2 of the pair effects over the post-warmup sweeps."""
    import jax.numpy as jnp

    cnt = state["c_count"] + jnp.where(tuning, 0.0, 1.0)
    delta = C - state["c_mean"]
    c_mean = jnp.where(tuning, state["c_mean"], state["c_mean"] + delta / cnt)
    c_m2 = jnp.where(tuning, state["c_m2"], state["c_m2"] + delta * (C - c_mean))
    return {"c_count": cnt, "c_mean": c_mean, "c_m2": c_m2}


def _fe_sweep(eq_static, has_alpha_fixed, keep_group_draws, store_eta, n_obs):
    """The structured NB flow FE sweep for one structure, taking ``data``."""
    import jax
    import jax.numpy as jnp
    from jax.scipy.special import gammaln

    from .._utils._jax_slice import jax_slice_sample_1d
    from .._utils._jax_utils import cached_sweep, make_pg_draw

    eq = cached_sweep(("flow_equation", *eq_static), lambda: _equation_fns(*eq_static))
    n, T = eq_static[0], eq_static[1]
    has_pairs, learn_sd = eq_static[5], eq_static[6]
    draw_pg = make_pg_draw()

    def alpha_log_density(log_a, y_j, y_dot_eta, mu, data):
        y_vals, y_counts = data["y_vals"], data["y_counts"]
        alpha_sigma, alpha_nu = data["alpha_sigma"], data["alpha_nu"]
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
        y_j = data["y"]
        alpha = state["alpha"]
        k_pg, k_eq, k_a, k_sd = jax.random.split(key, 4)
        omega = draw_pg(
            jnp.maximum(y_j + alpha, 1e-3),
            jnp.clip(state["eta"].ravel() - jnp.log(alpha), -30.0, 30.0),
            k_pg,
        )
        Om_all = omega.reshape(T, n, n)
        Z_all = (0.5 * (y_j - alpha) / omega + jnp.log(alpha)).reshape(T, n, n)
        eq_keys = ("beta", "tau", "C", "rho_d", "rho_o", "eta", "slice_widths")
        eq_state = {
            key_: state[key_] for key_ in eq_keys + (("s_pair",) if learn_sd else ())
        }
        new = eq.update(eq_state, Om_all, Z_all, k_eq, tuning, data)
        if learn_sd:
            y3 = y_j.reshape(T, n, n)
            la_ = jnp.log(alpha)
            new = eq.exact_sd_move(
                new,
                lambda e: jnp.sum(y3 * e - (y3 + alpha) * jnp.logaddexp(e, la_)),
                k_sd,
                data,
            )

        eta_flat = new["eta"].ravel()
        y_dot_eta, mu = y_j @ eta_flat, jnp.exp(eta_flat)
        if not has_alpha_fixed:
            log_alpha, _ = jax_slice_sample_1d(
                lambda la: alpha_log_density(la, y_j, y_dot_eta, mu, data),
                jnp.log(alpha),
                -10.0,
                10.0,
                key=k_a,
                w=0.5,
            )
            alpha = jnp.exp(log_alpha)
        else:
            alpha = data["alpha_fixed"]
        new["alpha"] = alpha
        if has_pairs and not keep_group_draws:
            new.update(welford_update(state, new["C"], tuning))
        trace = (new["rho_d"], new["rho_o"], new["beta"], new["tau"], alpha)
        if learn_sd:
            trace = trace + (new["s_pair"],)
        if has_pairs and keep_group_draws:
            trace = trace + (new["C"],)
        if store_eta:
            trace = trace + (new["eta"],)
        return new, trace

    return sweep


def make_structured_fe_sweep(
    y,
    W_csc,
    struct: FlowDesignStructure,
    priors,
    *,
    pair_effects=None,
    period_effects=None,
    keep_group_draws: bool = True,
    store_eta: bool = False,
):
    """Build ``sweep(state, key, tuning, data) -> (state, trace)`` for one chain.

    ``pair_effects`` is ``(mu, sigma)`` or ``None``; ``period_effects`` is
    ``(mu, sigma, n_tau)`` or ``None``, the effects covering the last ``n_tau``
    periods; a third element of ``pair_effects`` learns their sd.  Returns
    ``(sweep, initial_eta, data)``.  ``trace`` is ``(rho_d, rho_o, beta, tau,
    alpha[, s_pair][, C][, eta])``.  The compiled program is shared by every
    model of the same structure; ``sweep.kernels`` holds the moment block
    bound to this model's data, for tests.
    """
    sweep, initial_eta, data, eq = _fe_parts(
        y, W_csc, struct, priors, pair_effects=pair_effects,
        period_effects=period_effects, keep_group_draws=keep_group_draws,
        store_eta=store_eta,
    )  # fmt: skip

    def wrapped(state, key, tuning, data_):
        return sweep(state, key, tuning, data_)

    wrapped.kernels = {"moments": eq.moments}
    return wrapped, initial_eta, data


def _fe_parts(
    y, W_csc, struct, priors, *, pair_effects, period_effects, keep_group_draws,
    store_eta,
):  # fmt: skip
    """``(cached sweep, initial_eta, data, equation)`` for the FE flow sampler."""
    import jax.numpy as jnp

    from .._utils._jax_utils import cached_sweep, padded_value_counts

    eq = make_structured_equation(
        W_csc,
        struct,
        priors.beta_mu,
        priors.beta_sigma,
        rho_bounds=(priors.rho_lower, priors.rho_upper),
        pair_effects=pair_effects,
        period_effects=period_effects,
    )
    alpha_fixed = getattr(priors, "alpha_fixed", None)
    y_np = np.asarray(y, dtype=np.float64)
    y_vals, y_counts = padded_value_counts(y_np)
    # Data-sized arrays and data-dependent scalars enter as arguments.
    data = {
        "y": jnp.asarray(y_np),
        "y_vals": jnp.asarray(y_vals),
        "y_counts": jnp.asarray(y_counts),
        "alpha_sigma": jnp.float64(priors.alpha_sigma),
        "alpha_nu": jnp.float64(priors.alpha_nu),
        "alpha_fixed": jnp.float64(alpha_fixed if alpha_fixed is not None else 1.0),
        **eq.data,
    }
    static = (
        eq.static, alpha_fixed is not None, bool(keep_group_draws), bool(store_eta),
        float(y_np.size),
    )  # fmt: skip
    sweep = cached_sweep(("nb_flow_fe", *static), lambda: _fe_sweep(*static))
    return sweep, eq.initial_eta, data, eq


def run_chains_jax_flow_structured_fe(
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
    pair_effects=None,
    period_effects=None,
    keep_group_draws: bool = True,
    jax_seeds=None,
    slice_width=0.4,
    store_log_lik=False,
):
    """Run the structured separable NB flow sampler with fixed effects on JAX.

    Returns one dict per chain with the keys of
    :func:`._flow_structured_fe.run_chain_separable_structured_fe`.
    """
    import jax
    import jax.numpy as jnp
    from scipy.special import gammaln

    from ..._jax_dispatch import ensure_x64
    from .._utils._jax_utils import run_chains_chunked
    from .._utils._sparsax_lu import set_sparsax_lu_cache_size

    ensure_x64()
    struct = classify_flow_design(X, n, T) if struct is None else struct
    T = struct.T
    chains = len(inits)
    if jax_seeds is None:
        jax_seeds = list(range(chains))
    set_sparsax_lu_cache_size(max(32, 8 * chains))
    has_pairs = pair_effects is not None
    learn_sd = has_pairs and len(pair_effects) > 2 and pair_effects[2] is not None
    mu_tau, _, n_tau = period_effects if period_effects is not None else (0, 1, 0)

    sweep, initial_eta, data, _ = _fe_parts(
        y,
        W_csc,
        struct,
        priors,
        pair_effects=pair_effects,
        period_effects=period_effects,
        keep_group_draws=keep_group_draws,
        store_eta=store_log_lik,
    )
    y_np = np.asarray(y, dtype=np.float64)
    Y = y_np.reshape(T, n, n)
    states = []
    for i in inits:
        beta = jnp.asarray(i.beta, dtype=jnp.float64)
        tau = jnp.full(n_tau, float(mu_tau))
        C0 = np.zeros((n, n))
        if has_pairs:
            C0 = np.log(Y.mean(axis=0) + 0.5) - np.log(Y.mean() + 0.5) + pair_effects[0]
        C0 = jnp.asarray(C0)
        rd, ro = jnp.float64(float(i.rho_d)), jnp.float64(float(i.rho_o))
        st = {
            "beta": beta,
            "tau": tau,
            "C": C0,
            "rho_d": rd,
            "rho_o": ro,
            "alpha": jnp.float64(float(i.alpha)),
            "eta": initial_eta(beta, tau, C0, rd, ro, data),
            "slice_widths": (jnp.float64(slice_width),) * 2,
        }
        if learn_sd:
            st["s_pair"] = jnp.float64(float(pair_effects[1]))
        if has_pairs and not keep_group_draws:
            st.update(
                c_count=jnp.float64(0.0),
                c_mean=jnp.zeros((n, n)),
                c_m2=jnp.zeros((n, n)),
            )
        states.append(st)
    warm_keys = [jax.random.PRNGKey(int(s)) for s in jax_seeds]
    draw_keys = [jax.random.fold_in(jax.random.PRNGKey(int(s)), 1) for s in jax_seeds]
    final, traces = run_chains_chunked(
        sweep, states, warm_keys, draw_keys, tune=tune, draws=draws, consts=data
    )

    rd_all, ro_all, beta_all, tau_all, alpha_all = traces[:5]
    nxt = 5
    sd_all = None
    if learn_sd:
        sd_all = traces[nxt]
        nxt += 1
    C_all = None
    if has_pairs and keep_group_draws:
        C_all = traces[nxt]
        nxt += 1
    eta_all = traces[nxt] if store_log_lik else None
    results = []
    for c in range(chains):
        out = {
            "rho_d": rd_all[c],
            "rho_o": ro_all[c],
            "rho_w": -rd_all[c] * ro_all[c],
            "beta": beta_all[c],
            "time_effect": tau_all[c],
            "alpha": alpha_all[c],
            "log_lik": None,
        }
        if sd_all is not None:
            out["group_sd"] = sd_all[c]
        if has_pairs and keep_group_draws:
            out["group_effect"] = C_all[c].reshape(draws, n * n)
        elif has_pairs:
            cnt = float(final[c]["c_count"])
            out["group_effect_mean"] = np.asarray(final[c]["c_mean"]).ravel()
            out["group_effect_sd"] = np.sqrt(
                np.asarray(final[c]["c_m2"]).ravel() / max(cnt - 1.0, 1.0)
            )
        if store_log_lik:
            eta = eta_all[c].reshape(draws, -1)
            mu = np.exp(np.clip(eta, -30.0, 30.0))
            a = alpha_all[c][:, None]
            out["log_lik"] = (
                gammaln(y_np + a)
                - gammaln(a)
                - gammaln(y_np + 1.0)
                + y_np * np.log(np.maximum(mu / (mu + a), 1e-300))
                + a * np.log(np.maximum(a / (mu + a), 1e-300))
            )
        results.append(out)
    return results
