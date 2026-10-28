r"""JAX port of the structured separable NB flow sweep with pair and period effects.

Same sweep as :mod:`._flow_structured_fe` (read it first) — ω → ρ_d → ρ_o,
centred then non-centred (``β`` and the period effects integrated out, the
pair effects held) → ``(β, τ)`` jointly → the pair effects by conjugate
gradients → the time-invariant block move → their sd → α — compiled into one
XLA program per chunk of sweeps and run thread-per-chain through
:func:`.._utils._jax_utils.run_chains_chunked`, as the pooled port
:mod:`._flow_structured_jax` is.

Per-period terms run under ``lax.scan``/``lax.map`` so temporaries stay
``O(k n²)``.  ``A X = L_o X L_dᵀ`` uses sparse ``W`` (BCOO) and the effects'
``log|L_o||L_d|`` the traced log-determinant of
:func:`neighbayes._logdet._jax.logdet_jax_params`, so no compiled program
closes over a particular ``W``.  When per-draw pair effects would not fit in
memory the chain keeps their running mean and variance in its state instead
(Welford, draws only).
"""

from __future__ import annotations

import numpy as np

from ._flow_structured import FlowDesignStructure, classify_flow_design

#: Relative residual of the pair-effect CG solve (see ``_flow_structured_fe``).
_CG_TOL = 1e-10
_CG_MAXITER = 20_000
#: A run warns when a CG solve stopped above this relative residual.
_CG_WARN = 1e-8


def pair_state(has_pairs: bool) -> dict:
    """The initial state a pair-effect equation carries besides its draws: the
    largest relative CG residual so far, which a compiled sweep cannot warn on."""
    import jax.numpy as jnp

    return {"cg_resid": jnp.float64(0.0)} if has_pairs else {}


def warn_unconverged_cg(residuals) -> None:
    """Warn on the host when any chain's pair-effect CG stopped short."""
    import warnings

    worst = max((float(r) for r in residuals), default=0.0)
    if worst > _CG_WARN:
        warnings.warn(
            f"The pair-effect conjugate-gradient solve stopped at a relative "
            f"residual of {worst:.1e} (tolerance {_CG_TOL:.0e}, at most "
            f"{_CG_MAXITER} iterations); those draws of the pair effects are "
            "approximate.",
            RuntimeWarning,
            stacklevel=3,
        )


def _time_invariant_columns(struct: FlowDesignStructure):
    """``(J, recipes, Z)``: the design columns constant over periods.

    A recipe is ``("c", u, v)`` for a rank-one column ``U_u V_vᵀ`` or
    ``("f", i)`` for the full-rank ``F_i``; ``Z`` stacks them as pair arrays.
    """
    J, recipes, Z = [], [], []
    for j, col in enumerate(struct.cheap_cols):
        u, v = struct.u_idx[:, j], struct.v_idx[:, j]
        if np.all(u == u[0]) and np.all(v == v[0]):
            J.append(int(col))
            recipes.append(("c", int(u[0]), int(v[0])))
            Z.append(np.outer(struct.U[:, u[0]], struct.V[:, v[0]]))
    for j, col in enumerate(struct.full_cols):
        f = struct.f_idx[:, j]
        if np.all(f == f[0]):
            J.append(int(col))
            recipes.append(("f", int(f[0])))
            Z.append(np.asarray(struct.F[f[0]], dtype=np.float64))
    Z = np.stack(Z) if Z else np.zeros((0, struct.n, struct.n))
    return tuple(J), tuple(recipes), Z


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
    ld_kind, ti_J, ti_recipes = None, (), ()
    pair_raw = {}
    if has_pairs:
        import scipy.sparse as sp

        from ..._logdet._jax import logdet_jax_params
        from .._utils._jax_utils import build_w_bcoo

        W = sp.csr_matrix(W_csc, dtype=np.float64)
        ld_kind, ld_params = logdet_jax_params(W, None, *rho_bounds)
        W_bcoo, Wt_bcoo = build_w_bcoo(W)
        ti_J, ti_recipes, Z_ti = _time_invariant_columns(struct)
        pair_raw = {
            "W": W_bcoo,
            "Wt": Wt_bcoo,
            "w_diag": jnp.asarray(W.diagonal()),
            "w_colsq": jnp.asarray(np.asarray(W.multiply(W).sum(axis=0)).ravel()),
            "logdet": ld_params,
            "Z_ti": jnp.asarray(Z_ti),
        }
    raw = {
        **pair_raw,
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
        sd_scale is not None, int(n_tau), lu_solve, prefix, ld_kind, ti_J, ti_recipes,
    )  # fmt: skip
    return static, data


def _equation_fns(
    n, T, k, cc_t, fc_t, has_pairs, learn_sd, n_tau, lu_solve, prefix, ld_kind,
    ti_J, ti_recipes,
):  # fmt: skip
    """One structured equation's kernels for one structure, each taking ``data``."""
    import jax
    import jax.numpy as jnp
    from jax.scipy.linalg import solve_triangular

    from ..._logdet._jax import eval_logdet_params
    from .._utils._jax_slice import adapt_slice_width, jax_slice_sample_1d
    from ._flow_jax import _bind_sar_solver

    cc = np.asarray(cc_t, dtype=np.int32)
    fc = np.asarray(fc_t, dtype=np.int32)
    k_c, k_e = len(cc), len(fc)
    order = np.concatenate([cc, fc]).astype(np.int32)
    t0 = T - n_tau
    kF, kU, kV = prefix + "F", prefix + "U", prefix + "V"
    kW, kWt = prefix + "W", prefix + "Wt"

    def bind(data):
        def pre(name):
            return data[prefix + name]

        mu0, prec0 = pre("mu0"), pre("prec0")
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

        def moments(cols_of, resid_of, kx, prec_x, Om_all, mu_x=None):
            r"""``(M, v, quad)`` over ``[x-cols, τ]`` (see ``_gram_block``).

            ``mu_x`` is taken off each period's residual along its columns.
            """
            K = kx + n_tau

            def step(carry, t):
                Mx, vx, quad = carry
                Om, R = Om_all[t], resid_of(t)
                if kx:
                    C = cols_of(t)
                    if mu_x is not None:
                        R = R - jnp.tensordot(mu_x, C, axes=1)
                OmR = Om * R
                quad = quad + jnp.sum(OmR * R)
                if kx:
                    OmC = Om[None] * C
                    Mx = Mx + jnp.tensordot(OmC, C, axes=([1, 2], [1, 2]))
                    vx = vx + jnp.tensordot(OmC, R, axes=2)
                    cross = OmC.sum(axis=(1, 2))
                else:
                    cross = jnp.zeros(0)
                return (Mx, vx, quad), (cross, Om.sum(), OmR.sum())

            init = (jnp.zeros((kx, kx)), jnp.zeros(kx), jnp.float64(0.0))
            (Mx, vx, quad), (cross, om_sum, omr_sum) = jax.lax.scan(step, init, periods)
            M = jnp.zeros((K, K)).at[:kx, :kx].set(Mx)
            v = jnp.zeros(K).at[:kx].set(vx)
            if n_tau:
                tau_ix = jnp.arange(kx, K)
                M = M.at[tau_ix, tau_ix].add(om_sum[t0:])
                v = v.at[kx:].add(omr_sum[t0:])
                if kx:
                    M = M.at[:kx, kx:].add(cross[t0:].T).at[kx:, :kx].add(cross[t0:])
            M = M + jnp.diag(jnp.concatenate([prec_x, prec_tau]))
            return M, v, quad

        def collapsed_density(M, v, quad):
            if M.shape[0] == 0:
                return -0.5 * quad
            L = jnp.linalg.cholesky(M)
            w = solve_triangular(L, v, lower=True)
            val = -jnp.sum(jnp.log(jnp.diag(L))) - 0.5 * (quad - w @ w)
            return jnp.where(jnp.isfinite(val), val, -jnp.inf)

        def columns(A, B, Ufull):
            """``t ↦`` period ``t``'s transformed columns, ``(k, n, n)`` in ``order``."""

            def cols(t):
                parts = []
                if k_c:
                    parts.append(cheap_arrays(A, B, t))
                if k_e:
                    parts.append(Ufull[f_idx[t]])
                return jnp.concatenate(parts) if parts else jnp.zeros((0, n, n))

            return cols

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

        def apply_A(X, rd_, ro_, transpose=False):
            """``A X = L_o X L_dᵀ`` (``Aᵀ X = L_oᵀ X L_d``) for an ``n × n`` X."""
            W_, Wt_ = (data[kWt], data[kW]) if transpose else (data[kW], data[kWt])
            Y = X - ro_ * (W_ @ X)
            return Y - rd_ * (Y @ Wt_)

        def gram_diag(r):
            """``diag(LᵀL)`` for ``L = I − rW``."""
            return 1.0 - 2.0 * r * pre("w_diag") + r * r * pre("w_colsq")

        def logdet(r):
            return eval_logdet_params(ld_kind, pre("logdet"), r)

        def level_vector(rd_, ro_):
            """The filtered constant column ``A⁻¹1 = (L_o⁻¹1)(L_d⁻¹1)ᵀ``."""
            ones = jnp.ones((n, 1))
            return jnp.outer(solve(ro_, ones)[:, 0], solve(rd_, ones)[:, 0])

        def effects(eq):
            """The pair effects ``C = A C̃`` from the working ``C̃``."""
            return apply_A(eq["C"], eq["rho_d"], eq["rho_o"])

        def update(eq, Om_all, Z_all, key, tuning):
            F, U, V = data[kF], data[kU], data[kV]
            rd, ro = eq["rho_d"], eq["rho_o"]
            w_d, w_o, w_dn, w_on = eq["slice_widths"]
            keys = jax.random.split(key, 10)
            s_cur = eq["s_pair"] if learn_sd else s_pair
            C = eq["C"]
            tau_shift = jnp.where(in_tau, mu_tau, 0.0)
            F_all = F if k_e else jnp.zeros((0, n, n))

            def rho_density(Zr, Uf, Av, Bv):
                """The working data's density, ``β`` and ``τ`` integrated out."""
                M, v, quad = moments(
                    columns(Av, Bv, Uf),
                    lambda t: Zr[t] - tau_shift[t],
                    k,
                    prec_o,
                    Om_all,
                    mu_x=mu_o,
                )
                return collapsed_density(M, v, quad)

            def prior_term(rd_, ro_):
                """``log p(C̃ | ρ, σ)`` up to a constant: ``n log|L_o||L_d|`` less
                the deviance of the effects ``A C̃``."""
                if not has_pairs:
                    return 0.0
                dev = apply_A(C, rd_, ro_) - m_pair
                return (
                    n * (logdet(ro_) + logdet(rd_)) - 0.5 * jnp.sum(dev**2) / s_cur**2
                )

            # Centred: C̃ held, its ρ-dependent prior in the density.
            Zc = Z_all - C[None]
            P, A_fix = solve_left(ro, F_all, U)

            def ld_d(rv):
                Uf, Bv = solve_right(rv, P, V)
                return rho_density(Zc, Uf, A_fix, Bv) + prior_term(rv, ro)

            rd, _, sl_d, sr_d = jax_slice_sample_1d(
                ld_d, rd, rho_lo, rho_hi, key=keys[0], w=w_d, return_steps=True
            )
            P, B_fix = solve_right(rd, F_all, V)

            def ld_o(rv):
                Uf, Av = solve_left(rv, P, U)
                return rho_density(Zc, Uf, Av, B_fix) + prior_term(rd, rv)

            ro, _, sl_o, sr_o = jax_slice_sample_1d(
                ld_o, ro, rho_lo, rho_hi, key=keys[1], w=w_o, return_steps=True
            )
            widths_nc = (w_dn, w_on)
            if has_pairs:
                # Non-centred: the effects held, C̃ = A(ρ)⁻¹C moving with ρ.
                C_in = apply_A(C, rd, ro)
                P, A_fix = solve_left(ro, F_all, U)
                fixed = solve(ro, C_in)

                def ld_dn(rv):
                    Uf, Bv = solve_right(rv, P, V)
                    Ct = solve(rv, fixed.T).T
                    return rho_density(Z_all - Ct[None], Uf, A_fix, Bv)

                rd, _, sl_dn, sr_dn = jax_slice_sample_1d(
                    ld_dn, rd, rho_lo, rho_hi, key=keys[2], w=w_dn, return_steps=True
                )
                P, B_fix = solve_right(rd, F_all, V)
                fixed = solve(rd, C_in.T).T

                def ld_on(rv):
                    Uf, Av = solve_left(rv, P, U)
                    Ct = solve(rv, fixed)
                    return rho_density(Z_all - Ct[None], Uf, Av, B_fix)

                ro, _, sl_on, sr_on = jax_slice_sample_1d(
                    ld_on, ro, rho_lo, rho_hi, key=keys[3], w=w_on, return_steps=True
                )
                C = solve(rd, solve(ro, C_in).T).T
                widths_nc = (
                    adapt_slice_width(w_dn, sl_dn, sr_dn, tuning),
                    adapt_slice_width(w_on, sl_on, sr_on, tuning),
                )

            # (β, τ) jointly, then the pair effects
            half, A = solve_left(ro, F, U)
            Ufull, B = solve_right(rd, half, V)
            cols = columns(A, B, Ufull)

            Zc = Z_all - C[None]
            M, h, _ = moments(cols, lambda t: Zc[t], k, prec_o, Om_all)
            h = h + jnp.concatenate([prec_o * mu_o, prec_tau * mu_tau])
            Lg = jnp.linalg.cholesky(M)
            mean = jax.scipy.linalg.cho_solve((Lg, True), h)
            theta = mean + solve_triangular(
                Lg.T, jax.random.normal(keys[4], mean.shape, dtype=jnp.float64),
                lower=False,
            )  # fmt: skip
            beta = jnp.zeros(k).at[order].set(theta[:k])
            tau = theta[k:]
            zeros = jnp.zeros((n, n))
            if has_pairs:
                # C̃ | β, τ, ρ: CG on P = Σ_t Ω_t + AᵀA/σ² with a perturbed
                # right-hand side (exact to the tolerance), from the last draw.
                R = Z_all - eta_of(beta, tau, zeros, Ufull, A, B)
                D = Om_all.sum(axis=0)
                b = (
                    jnp.sum(Om_all * R, axis=0)
                    + apply_A(jnp.full((n, n), m_pair), rd, ro, True) / s_cur**2
                    + jnp.sqrt(D) * jax.random.normal(keys[5], (n, n), dtype=jnp.float64)
                    + apply_A(
                        jax.random.normal(keys[6], (n, n), dtype=jnp.float64), rd, ro, True
                    ) / s_cur
                )  # fmt: skip
                diag = D + jnp.outer(gram_diag(ro), gram_diag(rd)) / s_cur**2

                def P_mv(X):
                    return D * X + apply_A(apply_A(X, rd, ro), rd, ro, True) / s_cur**2

                C, _ = jax.scipy.sparse.linalg.cg(
                    P_mv, b, x0=C, tol=_CG_TOL, maxiter=_CG_MAXITER,
                    M=lambda r: r / diag,
                )  # fmt: skip
                cg_resid = jnp.maximum(
                    eq["cg_resid"],
                    jnp.linalg.norm(P_mv(C) - b) / jnp.linalg.norm(b),
                )
                if ti_recipes:
                    # β_J + t, C − Z_J t leaves η unchanged: t from the priors.
                    Zti = pre("Z_ti")
                    filtered = jnp.stack(
                        [
                            jnp.outer(A[:, r[1]], B[:, r[2]])
                            if r[0] == "c"
                            else Ufull[r[1]]
                            for r in ti_recipes
                        ]
                    )
                    J = np.asarray(ti_J, dtype=np.int32)
                    prec_J = prec0[J]
                    Q = (
                        jnp.diag(prec_J)
                        + jnp.tensordot(Zti, Zti, axes=([1, 2], [1, 2])) / s_cur**2
                    )
                    hJ = (
                        -prec_J * (beta[J] - mu0[J])
                        + jnp.tensordot(Zti, apply_A(C, rd, ro) - m_pair, axes=2)
                        / s_cur**2
                    )
                    LQ = jnp.linalg.cholesky(Q)
                    step = jax.scipy.linalg.cho_solve((LQ, True), hJ) + solve_triangular(
                        LQ.T, jax.random.normal(keys[7], (len(J),), dtype=jnp.float64),
                        lower=False,
                    )  # fmt: skip
                    beta = beta.at[J].add(step)
                    C = C - jnp.tensordot(step, filtered, axes=1)
            eta = eta_of(beta, tau, C, Ufull, A, B)
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
                    *widths_nc,
                ),
            }
            if has_pairs:
                out["cg_resid"] = cg_resid
            if learn_sd:
                # Centred σ | C, then non-centred σ | C/σ (interweaving).
                dev2 = jnp.sum((apply_A(C, rd, ro) - m_pair) ** 2)
                g = float(n * n)
                nu = 3.0

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
                    sd_density, jnp.log(s_cur), -10.0, 5.0, key=keys[8], w=0.5
                )
                mean_c = m_pair * level_vector(rd, ro)
                ct = (C - mean_c) / jnp.exp(ls)
                eta_rest = eta - C
                OmC = Om_all * ct[None]
                P = jnp.sum(OmC * ct[None])
                hh = jnp.sum(OmC * (Z_all - eta_rest - mean_c))

                def nc_density(ls2):
                    sd = jnp.exp(ls2)
                    return -0.5 * P * sd * sd + hh * sd + log_prior(sd) + ls2

                ls2, _ = jax_slice_sample_1d(
                    nc_density, ls, -10.0, 5.0, key=keys[9], w=0.5
                )
                sd = jnp.exp(ls2)
                C = mean_c + sd * ct
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

            ``(C̃ − μA⁻¹1)/σ`` held; returns ``eq`` unchanged unless the sd is
            learned.  See :func:`.._utils._group_effects.exact_noncentered_sd`.
            """
            if not learn_sd:
                return eq
            s = eq["s_pair"]
            mean_c = m_pair * level_vector(eq["rho_d"], eq["rho_o"])
            ct = (eq["C"] - mean_c) / s
            rest = eq["eta"] - (eq["C"] - mean_c)[None]
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
            out.update(s_pair=sd, C=mean_c + sd * ct, eta=rest + sd * ct[None])
            return out

        return {
            "update": update,
            "initial_eta": initial_eta,
            "exact_sd_move": exact_sd_move,
            "level_vector": level_vector,
            "effects": effects,
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

    def effects(eq, data):
        return bind(data)["effects"](eq)

    from types import SimpleNamespace

    return SimpleNamespace(
        update=update,
        initial_eta=initial_eta,
        initial_eta_jit=jax.jit(initial_eta),
        exact_sd_move=exact_sd_move,
        level_vector=level_vector,
        effects=effects,
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
      then ``(β, τ)``, then the working pair effects ``C`` (``A⁻¹`` of the
      effects) from ``(T, n, n)`` working data, where ``eq`` holds ``beta, tau,
      C, rho_d, rho_o, eta, slice_widths`` (four widths: centred and
      non-centred ρ_d, ρ_o);
    * ``initial_eta(beta, tau, C, rd, ro, data)`` (compiled),
      ``exact_sd_move(eq, loglik, key, data)``, ``level_vector(rd, ro, data)``
      and ``effects(eq, data)``, the pair effects ``A C``;
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
        effects=fns.effects,
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


def _fe_sweep(eq_static, has_alpha_fixed, keep_group_draws, store_eta, n_obs, has_mask):
    """The structured NB flow FE sweep for one structure, taking ``data``.

    Under a cell mask (``has_mask``) the masked cells are drawn-then-discarded:
    their Pólya–Gamma variates are drawn (the random stream is
    mask-invariant), then zeroed, and every α term sums live cells only
    (``n_obs`` is the live count; ``y_vals``/``y_counts`` tabulate live y).
    """
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
        if has_mask:
            m = data["mask_f"]
            ll = (
                y_counts @ gammaln(y_vals + a)
                - n_obs * gammaln(a)
                + y_dot_eta
                - jnp.sum(m * y_j * log_mu_a)
                - a * jnp.sum(m * log_mu_a)
                + n_obs * a * log_a
            )
        else:
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
        # Masked cells draw with neutral inputs (h = α) so the stream — and
        # every live draw — is identical whatever masked y holds.
        h_in = (
            jnp.where(data["mask_f"] > 0.5, y_j, 0.0) + alpha
            if has_mask
            else y_j + alpha
        )
        omega = draw_pg(
            jnp.maximum(h_in, 1e-3),
            jnp.clip(state["eta"].ravel() - jnp.log(alpha), -30.0, 30.0),
            k_pg,
        )
        if has_mask:
            live = data["mask_f"] > 0.5
            omega = jnp.where(live, omega, 0.0)
            z_flat = 0.5 * (y_j - alpha) / jnp.where(live, omega, 1.0) + jnp.log(alpha)
            z_flat = jnp.where(live, z_flat, 0.0)
        else:
            z_flat = 0.5 * (y_j - alpha) / omega + jnp.log(alpha)
        Om_all = omega.reshape(T, n, n)
        Z_all = z_flat.reshape(T, n, n)
        eq_keys = ("beta", "tau", "C", "rho_d", "rho_o", "eta", "slice_widths")
        eq_keys = eq_keys + (("s_pair",) if learn_sd else ())
        eq_keys = eq_keys + (("cg_resid",) if has_pairs else ())
        eq_state = {key_: state[key_] for key_ in eq_keys}
        new = eq.update(eq_state, Om_all, Z_all, k_eq, tuning, data)
        if learn_sd:
            y3 = y_j.reshape(T, n, n)
            la_ = jnp.log(alpha)
            if has_mask:
                mask3 = data["mask_f"].reshape(T, n, n)
                new = eq.exact_sd_move(
                    new,
                    lambda e: jnp.sum(
                        mask3 * (y3 * e - (y3 + alpha) * jnp.logaddexp(e, la_))
                    ),
                    k_sd,
                    data,
                )
            else:
                new = eq.exact_sd_move(
                    new,
                    lambda e: jnp.sum(y3 * e - (y3 + alpha) * jnp.logaddexp(e, la_)),
                    k_sd,
                    data,
                )

        eta_flat = new["eta"].ravel()
        if has_mask:
            m = data["mask_f"]
            y_dot_eta, mu = jnp.sum(m * y_j * eta_flat), jnp.exp(eta_flat)
        else:
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
        if has_pairs:
            effects = eq.effects(new, data)
        if has_pairs and not keep_group_draws:
            new.update(welford_update(state, effects, tuning))
        trace = (new["rho_d"], new["rho_o"], new["beta"], new["tau"], alpha)
        if learn_sd:
            trace = trace + (new["s_pair"],)
        if has_pairs and keep_group_draws:
            trace = trace + (effects,)
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
    store_eta, cell_mask=None,
):  # fmt: skip
    """``(cached sweep, initial_eta, data, equation)`` for the FE flow sampler."""
    import jax.numpy as jnp

    from .._utils._jax_utils import cached_sweep, padded_value_counts
    from ._flow_structured_fe import _normalize_cell_mask

    y_np = np.asarray(y, dtype=np.float64)
    cell_mask = _normalize_cell_mask(cell_mask, struct.T, struct.n)
    has_mask = cell_mask is not None
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
    y_live = y_np[cell_mask] if has_mask else y_np
    y_vals, y_counts = padded_value_counts(y_live)
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
    if has_mask:
        data["mask_f"] = jnp.asarray(cell_mask.astype(np.float64))
    static = (
        eq.static, alpha_fixed is not None, bool(keep_group_draws), bool(store_eta),
        float(y_live.size), has_mask,
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
    cell_mask=None,
    progressbar=False,
):
    """Run the structured separable NB flow sampler with fixed effects on JAX.

    Returns one dict per chain with the keys of
    :func:`._flow_structured_fe.run_chain_separable_structured_fe`.
    ``cell_mask`` follows that function's contract (live cells ``True``, drawn-
    then-discarded elsewhere).
    """
    import jax
    import jax.numpy as jnp
    from scipy.special import gammaln

    from ..._jax_dispatch import ensure_x64
    from .._utils._jax_utils import run_chains_chunked
    from .._utils._sparsax_lu import set_sparsax_lu_cache_size
    from ._flow_structured_fe import _normalize_cell_mask

    ensure_x64()
    struct = classify_flow_design(X, n, T) if struct is None else struct
    T = struct.T
    cell_mask = _normalize_cell_mask(cell_mask, T, n)
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
        cell_mask=cell_mask,
    )
    y_np = np.asarray(y, dtype=np.float64)
    Y = y_np.reshape(T, n, n)
    states = []
    for i in inits:
        beta = jnp.asarray(i.beta, dtype=jnp.float64)
        tau = jnp.full(n_tau, float(mu_tau))
        C0 = np.zeros((n, n))
        if has_pairs:
            if cell_mask is None:
                C0 = (
                    np.log(Y.mean(axis=0) + 0.5)
                    - np.log(Y.mean() + 0.5)
                    + pair_effects[0]
                )
            else:
                # Mask-aware start: no masked count may reach a live cell's η.
                m3 = cell_mask.reshape(T, n, n)
                Yl = np.where(m3, Y, 0.0)
                pair_mean = Yl.sum(axis=0) / np.maximum(m3.sum(axis=0), 1)
                grand = Yl.sum() / m3.sum()
                C0 = np.log(pair_mean + 0.5) - np.log(grand + 0.5) + pair_effects[0]
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
            "slice_widths": (jnp.float64(slice_width),) * 4,
        }
        if learn_sd:
            st["s_pair"] = jnp.float64(float(pair_effects[1]))
        st.update(pair_state(has_pairs))
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
        sweep,
        states,
        warm_keys,
        draw_keys,
        tune=tune,
        draws=draws,
        progressbar=progressbar,
        model_type="nb_flow_panel",
        consts=data,
    )
    if has_pairs:
        warn_unconverged_cg(f["cg_resid"] for f in final)

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
