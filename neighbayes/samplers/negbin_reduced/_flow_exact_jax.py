r"""JAX port of the exact separable NB flow sampler (:mod:`._flow_exact`).

The same sweep — ``θ`` along the Fisher information's eigenvectors, the levels,
``(a, b)`` along their warmup covariance's with ``θ`` sheared along its
regression on them, then ``α`` — compiled into one XLA program per chunk of
sweeps, the chains run thread-per-chain through
:func:`.._utils._jax_utils.run_chains_chunked`.  Periods whose full-rank columns
coincide share one *key* and its two-sided solve, as in the Pólya–Gamma
sampler this one replaced.  The warmup schedule, the live cells and their
count are data, so fits of different lengths and masks share one compiled
program.  Each level set is drawn at once, with a fixed number of Newton
steps to each mode.
"""

from __future__ import annotations

import numpy as np

from ._flow_exact import (
    _AB_MIN_DRAWS,
    _AB_REFRESH,
    _AB_START,
    _AB_WIDTH,
    _FISHER_EVERY,
    _FISHER_FIRST,
    _LEVEL_DF,
    _LEVEL_MAX_STEP,
    _LEVEL_NEWTON,
    _THETA_WIDTH,
    FlowLevels,
    live_cells,
)
from ._flow_structured import FlowDesignStructure, classify_flow_design


def _exact_parts(
    y,
    W_csc,
    struct: FlowDesignStructure,
    priors,
    tune,
    levels=None,
    cell_mask=None,
    keep_pair_draws=True,
):
    """Host side of the sweep: ``(static, data)``.

    ``static`` is what the compiled sweep closes over and keys its cache on;
    ``data`` every data-dependent array, passed as an argument (an array of
    data size closed over is embedded and constant-folded by XLA).  ``levels``
    and ``cell_mask`` as :class:`._flow_exact.ExactFlowChain`; without
    ``keep_pair_draws`` the sweep keeps the pair levels' running moments in
    its state instead of tracing them.
    """
    import jax.numpy as jnp

    from .._utils._jax_utils import padded_value_counts
    from ._flow_jax import _sar_solver_parts

    n, T, k = struct.n, struct.T, struct.k
    k_e = len(struct.full_cols)
    mu0 = np.broadcast_to(np.asarray(priors.beta_mu, dtype=np.float64), (k,)).copy()
    sd0 = np.broadcast_to(np.asarray(priors.beta_sigma, dtype=np.float64), (k,))
    alpha_fixed = getattr(priors, "alpha_fixed", None)
    y_np = np.asarray(y, dtype=np.float64)
    levels = levels if levels is not None else FlowLevels(())
    live = live_cells(y_np, T, n, levels, cell_mask)
    y_vals, y_counts = padded_value_counts(y_np[live.ravel() > 0])
    keys = sorted({tuple(int(i) for i in struct.f_idx[t]) for t in range(T)})
    lu_solve, sar = _sar_solver_parts(W_csc, n)
    sets = levels.sets
    data = {
        "y": jnp.asarray(y_np),
        "live": jnp.asarray(live),
        "n_obs": jnp.float64(live.sum()),
        "lev_mu": jnp.asarray([st.mu for st in sets], dtype=jnp.float64),
        "lev_scale": jnp.asarray(
            [st.scale if np.isfinite(st.scale) else 1.0 for st in sets],
            dtype=jnp.float64,
        ),
        "F": jnp.asarray(np.stack(struct.F)) if k_e else jnp.zeros((0, n, n)),
        "U": jnp.asarray(struct.U),
        "V": jnp.asarray(struct.V),
        "mu0": jnp.asarray(mu0),
        "prec0": jnp.asarray(1.0 / sd0**2),
        "ab_lo": jnp.float64(-np.log1p(-priors.rho_lower)),
        "ab_hi": jnp.float64(-np.log1p(-priors.rho_upper)),
        "alpha_sigma": jnp.float64(priors.alpha_sigma),
        "alpha_nu": jnp.float64(priors.alpha_nu),
        "alpha_fixed": jnp.float64(alpha_fixed if alpha_fixed is not None else 1.0),
        "y_vals": jnp.asarray(y_vals),
        "y_counts": jnp.asarray(y_counts),
        "ui": jnp.asarray(struct.u_idx, dtype=jnp.int32),  # (T, k_c)
        "vi": jnp.asarray(struct.v_idx, dtype=jnp.int32),
        "f_idx": jnp.asarray(struct.f_idx, dtype=jnp.int32),  # (T, k_e)
        "key_of_t": jnp.asarray(
            [keys.index(tuple(int(i) for i in struct.f_idx[t])) for t in range(T)],
            dtype=jnp.int32,
        ),
        "key_idx": jnp.asarray(
            np.asarray(keys, dtype=np.int32).reshape(len(keys), k_e)
        ),
        "tune": jnp.int32(tune),
        "sar": sar,
    }
    spec = tuple(
        (st.kind, st.random, st.flat, levels.baseline(i)) for i, st in enumerate(sets)
    )
    static = (
        n,
        T,
        k,
        tuple(int(c) for c in struct.cheap_cols),
        tuple(int(c) for c in struct.full_cols),
        len(keys),
        lu_solve,
        alpha_fixed is not None,
        spec,
        bool(keep_pair_draws),
    )
    return static, data


def _exact_fns(
    n, T, k, cc_t, fc_t, n_keys, lu_solve, has_alpha_fixed, spec, keep_pair=True
):
    """``sweep(state, key, tuning, data) -> (state, trace)`` for one structure."""
    import jax
    import jax.numpy as jnp
    from jax.scipy.special import gammaln

    from .._utils._jax_slice import jax_slice_along, jax_slice_sample_1d
    from ._flow_exact import _LEVEL_AXES
    from ._flow_jax import _bind_sar_solver
    from ._flow_structured_fe_jax import welford_update

    cc = np.asarray(cc_t, dtype=np.int32)
    fc = np.asarray(fc_t, dtype=np.int32)
    k_c, k_e = len(cc), len(fc)
    period_idx = np.arange(T)
    axes = [_LEVEL_AXES[kind] for kind, *_ in spec]
    ndims = [2 if kind == "pair" else 1 for kind, *_ in spec]

    def grow(v, ax):
        return jnp.expand_dims(v, ax)

    def bind(data):
        y = data["y"]
        F, U, V = data["F"], data["U"], data["V"]
        mu0, prec0 = data["mu0"], data["prec0"]
        ui, vi, f_idx = data["ui"], data["vi"], data["f_idx"]
        key_of_t, key_idx = data["key_of_t"], data["key_idx"]
        M = data["live"]
        Mf = M.ravel()
        Y = y.reshape(T, n, n)
        n_obs = data["n_obs"]
        solve = _bind_sar_solver(lu_solve, data["sar"])

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

        def filtered(a, b, stack):
            """``(L_o⁻¹ S_i L_d⁻ᵀ, L_o⁻¹U, L_d⁻¹V)`` at ``(a, b)``."""
            half, A = solve_left(-jnp.expm1(-b), stack, U)
            full, B = solve_right(-jnp.expm1(-a), half, V)
            return full, A, B

        def columns(a, b):
            """``U = A⁻¹X/m`` at ``(a, b)``, one coefficient a slice: ``(k, T, n, n)``."""
            Ufull, A, B = filtered(a, b, F)
            cols = jnp.zeros((k, T, n, n))
            if k_c:
                cheap = jnp.einsum("otj,dtj->jtod", A[:, ui], B[:, vi])
                cols = cols.at[cc].set(cheap)
            if k_e:
                cols = cols.at[fc].set(Ufull[f_idx].transpose(1, 0, 2, 3))
            return cols * jnp.exp(-(a + b))

        def filtered_at(a, b, theta):
            """``A⁻¹Xθ/m`` at ``(a, b)``, ``(T, n, n)``."""
            rows = jnp.repeat(jnp.arange(n_keys), k_e)
            C = (
                jnp.zeros((n_keys, F.shape[0]))
                .at[rows, key_idx.ravel()]
                .add(jnp.tile(theta[fc], n_keys))
            )
            full, A, B = filtered(a, b, jnp.tensordot(C, F, axes=1))

            def period(t):
                e = full[key_of_t[t]]
                if k_c:
                    e = e + (A[:, ui[t]] * theta[cc]) @ B[:, vi[t]].T
                return e

            return jax.lax.map(period, period_idx) * jnp.exp(-(a + b))

        def nb_ll(eta, alpha):
            return jnp.sum(
                Mf * (y * eta - (y + alpha) * jnp.logaddexp(eta, jnp.log(alpha)))
            )

        def log_prior(theta, s):
            return -0.5 * jnp.sum(prec0 * (theta * jnp.exp(-s) - mu0) ** 2)

        def alpha_log_density(log_a, y_dot_eta, mu):
            a = jnp.exp(log_a)
            log_mu_a = jnp.log(mu + a)
            ll = (
                data["y_counts"] @ gammaln(data["y_vals"] + a)
                - n_obs * gammaln(a)
                + y_dot_eta
                - (Mf * y) @ log_mu_a
                - a * (Mf @ log_mu_a)
                + n_obs * a * log_a
            )
            nu, sig = data["alpha_nu"], data["alpha_sigma"]
            return log_a + ll - 0.5 * (nu + 1.0) * jnp.log1p(a * a / (nu * sig**2))

        # --- the levels ------------------------------------------------------
        def live_mean(Fx, ax):
            return jnp.sum(Fx * M, axis=ax) / jnp.maximum(jnp.sum(M, axis=ax), 1.0)

        def level_sum(lev):
            out = jnp.zeros((T, n, n))
            for v, ax in zip(lev, axes):
                out = out + grow(v, ax)
            return out

        def level_parts(Fx):
            """The parts of ``Fx`` each set absorbs (see ``_flow_exact``)."""
            parts, rest = [], Fx
            for ax in axes:
                part = live_mean(rest, ax)
                parts.append(part)
                rest = rest - grow(part, ax)
            for i, (_, _, _, baseline) in enumerate(spec):
                if baseline:
                    lead = parts[i].shape[: parts[i].ndim - ndims[i]]
                    c = parts[i].reshape(lead + (-1,))[..., 0]
                    parts[i] = parts[i] - jnp.expand_dims(c, tuple(range(-ndims[i], 0)))
                    parts[0] = parts[0] + jnp.expand_dims(c, tuple(range(-ndims[0], 0)))
            return parts

        def precisions(tau):
            return [
                1.0 / tau[i] ** 2
                if random
                else (0.0 if flat else 1.0 / data["lev_scale"][i] ** 2)
                for i, (_, random, flat, _) in enumerate(spec)
            ]

        def level_log_prior(lev, tau):
            out = 0.0
            for i, (v, prec) in enumerate(zip(lev, precisions(tau))):
                if not spec[i][2]:
                    out = out - 0.5 * prec * jnp.sum((v - data["lev_mu"][i]) ** 2)
            return out

        def level_draw(base, alpha, current, mu, prec, ax, key):
            """Every level of a set at once (see ``_flow_exact._level_draw``)."""
            la = jnp.log(alpha)
            yM, aM = Y * M, (Y + alpha) * M
            sy = yM.sum(axis=ax)
            live = M.sum(axis=ax) > 0

            def logp(v):
                e = base + grow(v, ax)
                ll = (yM * e - aM * jnp.logaddexp(e, la)).sum(axis=ax)
                return ll - 0.5 * prec * (v - mu) ** 2

            def newton(_, x):
                p = jax.nn.sigmoid(base + grow(x, ax) - la)
                grad = sy - (aM * p).sum(axis=ax) - prec * (x - mu)
                hess = (aM * p * (1.0 - p)).sum(axis=ax) + prec
                step = jnp.clip(
                    grad / jnp.where(live, hess, 1.0), -_LEVEL_MAX_STEP, _LEVEL_MAX_STEP
                )
                return x + jnp.where(live, step, 0.0)

            lse = jax.scipy.special.logsumexp(jnp.where(M > 0, base, -jnp.inf), axis=ax)
            start = jnp.where(live, jnp.log(sy + 0.5) - jnp.where(live, lse, 0.0), mu)
            x = jax.lax.fori_loop(0, _LEVEL_NEWTON, newton, start)
            p = jax.nn.sigmoid(base + grow(x, ax) - la)
            info = (aM * p * (1.0 - p)).sum(axis=ax) + prec
            sd = 1.0 / jnp.sqrt(jnp.where(live, info, 1.0))

            def logq(v):
                z = (v - x) / sd
                return -0.5 * (_LEVEL_DF + 1.0) * jnp.log1p(z * z / _LEVEL_DF)

            k_t, k_u = jax.random.split(key)
            prop = x + sd * jax.random.t(k_t, _LEVEL_DF, x.shape)
            ratio = logp(prop) - logp(current) + logq(current) - logq(prop)
            accept = live & (jnp.log(jax.random.uniform(k_u, x.shape)) < ratio)
            return jnp.where(accept, prop, current)

        def sd_draw(sigma, x, mu, scale, key):
            """The sd of ``x ~ N(mu, sigma²)`` under half-t(3, scale), on ``log sigma``."""
            ss, G = jnp.sum((x - mu) ** 2), x.size

            def logp(ls):
                s2 = jnp.exp(2.0 * ls)
                return (
                    -G * ls
                    - 0.5 * ss / s2
                    - 2.0 * jnp.log1p(s2 / (3.0 * scale**2))
                    + ls
                )

            ls, _ = jax_slice_sample_1d(
                logp, jnp.log(sigma), -10.0, 10.0, key=key, w=0.5
            )
            return jnp.exp(ls)

        def level_moves(eta, lev, tau, alpha, key):
            """The levels, the shifts between random sets and their sds."""
            E = eta.reshape(T, n, n)
            lev, tau = list(lev), list(tau)
            precs = precisions(tau)
            keys = jax.random.split(key, 3 * len(spec) + 1)
            for i, (_, _, _, baseline) in enumerate(spec):
                mu = data["lev_mu"][i]
                base = E - grow(lev[i], axes[i])
                new = level_draw(base, alpha, lev[i], mu, precs[i], axes[i], keys[i])
                if baseline:
                    new = new.reshape(-1).at[0].set(mu).reshape(new.shape)
                lev[i] = new
                E = base + grow(new, axes[i])
            for i in range(len(spec) - 1):
                if not (spec[i][1] and spec[i + 1][1]):
                    continue
                # c in (l_i + c, l_j − c), η unchanged: a Gaussian conditional.
                pi, pj = precs[i], precs[i + 1]
                prec = lev[i].size * pi + lev[i + 1].size * pj
                mean = (
                    pi * jnp.sum(data["lev_mu"][i] - lev[i])
                    + pj * jnp.sum(lev[i + 1] - data["lev_mu"][i + 1])
                ) / prec
                c = mean + jax.random.normal(keys[len(spec) + i]) / jnp.sqrt(prec)
                lev[i], lev[i + 1] = lev[i] + c, lev[i + 1] - c
            for i, (_, random, _, _) in enumerate(spec):
                if random:
                    tau[i] = sd_draw(
                        tau[i],
                        lev[i],
                        data["lev_mu"][i],
                        data["lev_scale"][i],
                        keys[2 * len(spec) + i],
                    )
            return E.ravel(), tuple(lev), tuple(tau)

        def sweep(state, key, tuning):
            theta, a, b, alpha, it = (
                state["theta"],
                state["a"],
                state["b"],
                state["alpha"],
                state["it"],
            )
            lev, tau = state["lev"], state["tau"]
            k_th, k_thp, k_ab, k_abp, k_al, k_lev = jax.random.split(key, 6)
            s = a + b

            # 1. θ | a, b, α along the Fisher information's eigenvectors, the
            # levels following the parts of the moved column they absorb.
            cols = columns(a, b)
            eta = (jnp.tensordot(theta, cols, axes=1) + level_sum(lev)).ravel()
            parts = level_parts(cols)
            for part, ax in zip(parts, axes):
                cols = cols - grow(part, ax)
            cols = cols.reshape(k, -1)
            parts = [part.reshape(k, -1) for part in parts]

            def fisher(_):
                w = alpha * jax.nn.sigmoid(eta - jnp.log(alpha)) * Mf
                H = (cols * w) @ cols.T + jnp.diag(prec0 * jnp.exp(-2.0 * s))
                for part, prec in zip(parts, precisions(tau)):
                    H = H + prec * (part @ part.T)
                lam, V_ = jnp.linalg.eigh(H)
                return V_, 2.0 / jnp.sqrt(jnp.maximum(lam, 1e-12))

            refresh = tuning & ((it < _FISHER_FIRST) | (it % _FISHER_EVERY == 0))
            dirs, widths = jax.lax.cond(
                refresh, fisher, lambda _: (state["th_dirs"], state["th_w"]), None
            )
            UD = dirs.T @ cols
            LD = [-(dirs.T @ part) for part in parts]
            order = jax.random.permutation(k_thp, k)
            shapes = [v.shape for v in lev]

            def th_step(i, carry):
                th, e, f, lv = carry
                j = order[i]
                d, ud = dirs[:, j], UD[j]
                ld = [D[j].reshape(sh) for D, sh in zip(LD, shapes)]

                def fx(t, c):
                    moved = [v + t * dv for v, dv in zip(lv, ld)]
                    val = (
                        nb_ll(e + t * ud, alpha)
                        + log_prior(th + t * d, s)
                        + level_log_prior(moved, tau)
                    )
                    return val, None, c

                t, f, _, _ = jax_slice_along(
                    fx, f, None, None, widths[j], jax.random.fold_in(k_th, i)
                )
                return (
                    th + t * d,
                    e + t * ud,
                    f,
                    tuple(v + t * dv for v, dv in zip(lv, ld)),
                )

            f = nb_ll(eta, alpha) + log_prior(theta, s) + level_log_prior(lev, tau)
            theta, eta, f, lev = jax.lax.fori_loop(
                0, k, th_step, (theta, eta, f, tuple(lev))
            )
            if spec:
                eta, lev, tau = level_moves(eta, lev, tau, alpha, k_lev)

            # 2. (a, b) with θ sheared along G, the Jacobian e^{-(k+1)(a+b)},
            # each level's total held.
            k1 = k + 1.0
            here = jnp.concatenate([jnp.stack([a, b]), theta])
            F0 = eta.reshape(T, n, n) - level_sum(lev)
            hold = [v + part for v, part in zip(lev, level_parts(F0))]

            def g(z, carry):
                inside = jnp.all((z[:2] > data["ab_lo"]) & (z[:2] < data["ab_hi"]))
                z = jnp.where(inside, z, here)
                Fz = filtered_at(z[0], z[1], z[2:])
                lz = tuple(h - part for h, part in zip(hold, level_parts(Fz)))
                e = (Fz + level_sum(lz)).ravel()
                s_ = z[0] + z[1]
                val = nb_ll(e, alpha) + log_prior(z[2:], s_) - k1 * s_
                val = val + level_log_prior(lz, tau)
                ok = inside & jnp.isfinite(val)
                return jnp.where(ok, val, -jnp.inf), (e, lz), carry

            ab_dirs, ab_w, shear = state["ab_dirs"], state["ab_w"], state["ab_shear"]
            order2 = jax.random.permutation(k_abp, 2)

            def ab_step(i, carry):
                z, gz, aux = carry
                d = ab_dirs[:, order2[i]]
                v = jnp.concatenate([d, shear @ d])
                t, gz, aux, _ = jax_slice_along(
                    lambda t, c: g(z + t * v, c),
                    gz,
                    aux,
                    None,
                    ab_w[order2[i]],
                    jax.random.fold_in(k_ab, i),
                )
                return z + t * v, gz, aux

            g0 = nb_ll(eta, alpha) + log_prior(theta, s) - k1 * s
            g0 = g0 + level_log_prior(lev, tau)
            z, _, (eta, lev) = jax.lax.fori_loop(
                0, 2, ab_step, (here, g0, (eta, tuple(lev)))
            )
            a, b, theta = z[0], z[1], z[2:]

            # The (a, b, θ) covariance (Welford) and block 2's directions, in
            # warmup.
            tune = data["tune"]
            acc = tuning & (it >= _AB_START * tune // 5)
            cnt, mean, M2 = state["w_cnt"], state["w_mean"], state["w_M2"]
            cnt = cnt + acc.astype(jnp.int32)
            delta = z - mean
            mean = jnp.where(acc, mean + delta / jnp.maximum(cnt, 1), mean)
            M2 = jnp.where(acc, M2 + jnp.outer(delta, z - mean), M2)
            at = jnp.zeros((), bool)
            for q in _AB_REFRESH:
                at = at | (it == q * tune // 5)

            def ab_directions(_):
                C = M2 / (cnt - 1)
                C_ab = C[:2, :2] + 1e-10 * jnp.eye(2)
                lam, V_ = jnp.linalg.eigh(C_ab)
                G = jnp.linalg.solve(C_ab, C[:2, 2:]).T
                return V_, 2.0 * jnp.sqrt(jnp.maximum(lam, 1e-12)), G

            ab_dirs, ab_w, shear = jax.lax.cond(
                acc & at & (cnt > _AB_MIN_DRAWS),
                ab_directions,
                lambda _: (ab_dirs, ab_w, shear),
                None,
            )

            # 3. α | η, over the live cells.
            if has_alpha_fixed:
                alpha = data["alpha_fixed"]
            else:
                y_dot_eta, mu = (Mf * y) @ eta, jnp.exp(eta)
                log_alpha, _ = jax_slice_sample_1d(
                    lambda la: alpha_log_density(la, y_dot_eta, mu),
                    jnp.log(alpha),
                    -10.0,
                    10.0,
                    key=k_al,
                    w=0.5,
                )
                alpha = jnp.exp(log_alpha)

            new = {
                "theta": theta,
                "a": a,
                "b": b,
                "alpha": alpha,
                "it": it + 1,
                "th_dirs": dirs,
                "th_w": widths,
                "ab_dirs": ab_dirs,
                "ab_w": ab_w,
                "ab_shear": shear,
                "w_cnt": cnt,
                "w_mean": mean,
                "w_M2": M2,
                "lev": tuple(lev),
                "tau": tuple(tau),
            }
            rho_d, rho_o = -jnp.expm1(-a), -jnp.expm1(-b)
            trace = (rho_d, rho_o, theta * jnp.exp(-(a + b)), alpha)
            for i, (kind, random, _, _) in enumerate(spec):
                if kind == "pair" and not keep_pair:
                    new.update(welford_update(state, lev[i], tuning))
                else:
                    unit_live = M.sum(axis=axes[i]) > 0
                    trace = trace + (jnp.where(unit_live, lev[i], jnp.nan),)
                if random:
                    trace = trace + (tau[i],)
            return new, trace + (eta,)

        return sweep

    def sweep(state, key, tuning, data):
        return bind(data)(state, key, tuning)

    return sweep


def trace_names(
    levels: FlowLevels | None, keep_pair_draws: bool = True
) -> tuple[str, ...]:
    """The names of a sweep's trace after ``rho_d, rho_o, beta, alpha``."""
    names = ()
    for st in levels.sets if levels is not None else ():
        if keep_pair_draws or st.kind != "pair":
            names += (f"{st.kind}_effect",)
        if st.random:
            names += (f"{st.kind}_effect_sd",)
    return names


def start_state(
    struct: FlowDesignStructure,
    init,
    levels: FlowLevels | None,
    keep_pair_draws: bool = True,
):
    """A chain's starting state from ``init`` (``beta``, ``rho_d``, ``rho_o``, ``alpha``)."""
    import jax.numpy as jnp

    n, T, k = struct.n, struct.T, struct.k
    shapes = {"origin": (n,), "destination": (n,), "pair": (n, n), "period": (T,)}
    sets = levels.sets if levels is not None else ()
    a0, b0 = -np.log1p(-float(init.rho_d)), -np.log1p(-float(init.rho_o))
    summary = (
        {"c_count": jnp.float64(0.0), "c_mean": jnp.zeros((n, n)),
         "c_m2": jnp.zeros((n, n))}
        if not keep_pair_draws and any(st.kind == "pair" for st in sets)
        else {}
    )  # fmt: skip
    return summary | {
        "theta": jnp.asarray(np.asarray(init.beta, float) * np.exp(a0 + b0)),
        "a": jnp.float64(a0),
        "b": jnp.float64(b0),
        "alpha": jnp.float64(float(init.alpha)),
        "it": jnp.int32(0),
        "th_dirs": jnp.eye(k),
        "th_w": jnp.full(k, _THETA_WIDTH),
        "ab_dirs": jnp.eye(2),
        "ab_w": jnp.full(2, _AB_WIDTH),
        "ab_shear": jnp.zeros((k, 2)),
        "w_cnt": jnp.int32(0),
        "w_mean": jnp.zeros(k + 2),
        "w_M2": jnp.zeros((k + 2, k + 2)),
        "lev": tuple(jnp.full(shapes[st.kind], st.mu) for st in sets),
        "tau": tuple(jnp.float64(st.scale if st.random else 1.0) for st in sets),
    }


def run_chains_jax_flow_exact(
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
    store_log_lik=False,
    progressbar=False,
    levels: FlowLevels | None = None,
    cell_mask=None,
    keep_pair_draws=True,
):
    """Run the exact separable NB flow sampler on the JAX backend.

    ``y`` (and ``X``, unless ``struct`` is given) are stacked time-first over
    ``T`` periods of ``n²`` flows.  Returns one dict per chain with keys
    ``rho_d, rho_o, rho_w, beta, alpha, log_lik`` and each level set's draws
    (:func:`trace_names`).  Without ``keep_pair_draws`` the pair levels come
    back as ``pair_effect_summary``, their posterior ``{"mean", "sd"}`` over
    every post-warmup sweep.
    """
    import jax
    from scipy.special import gammaln

    from ..._jax_dispatch import ensure_x64
    from .._utils._jax_utils import cached_sweep, run_chains_chunked
    from .._utils._sparsax_lu import set_sparsax_lu_cache_size

    ensure_x64()
    struct = classify_flow_design(X, n, T) if struct is None else struct
    chains = len(inits)
    if jax_seeds is None:
        jax_seeds = list(range(chains))
    set_sparsax_lu_cache_size(max(32, 8 * chains))
    static, data = _exact_parts(
        y, W_csc, struct, priors, tune, levels, cell_mask, keep_pair_draws
    )

    def build():
        core = _exact_fns(*static)
        if store_log_lik:
            return core

        def without_eta(state, key, tuning, data):  # η is traced for log_lik only
            new, trace = core(state, key, tuning, data)
            return new, trace[:-1]

        return without_eta

    sweep = cached_sweep(("nb_flow_exact", *static, store_log_lik), build)
    states = [start_state(struct, i, levels, keep_pair_draws) for i in inits]
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
        model_type="nb_flow",
        consts=data,
    )

    sl = slice(None, None, thin) if thin > 1 else slice(None)
    rd_all, ro_all, beta_all, alpha_all = (np.asarray(t)[:, sl] for t in traces[:4])
    names = trace_names(levels, keep_pair_draws)
    level_draws = {
        name: np.asarray(t)[:, sl] for name, t in zip(names, traces[4 : 4 + len(names)])
    }
    y_np = np.asarray(y, dtype=np.float64)
    live = np.asarray(data["live"]).ravel() > 0
    results = []
    for c in range(chains):
        log_lik = None
        if store_log_lik:
            eta = np.asarray(traces[-1])[c, sl].reshape(len(alpha_all[c]), -1)
            mu = np.exp(np.clip(eta, -30.0, 30.0))
            a = alpha_all[c][:, None]
            log_lik = np.where(
                live,
                gammaln(y_np + a)
                - gammaln(a)
                - gammaln(y_np + 1.0)
                + y_np * np.log(np.maximum(mu / (mu + a), 1e-300))
                + a * np.log(np.maximum(a / (mu + a), 1e-300)),
                0.0,
            )
        results.append(
            {
                "rho_d": rd_all[c],
                "rho_o": ro_all[c],
                "rho_w": -rd_all[c] * ro_all[c],
                "beta": beta_all[c],
                "alpha": alpha_all[c],
                "log_lik": log_lik,
                **{name: v[c] for name, v in level_draws.items()},
            }
        )
        if "c_mean" in final[c]:
            dead = np.asarray(data["live"]).sum(axis=0) == 0
            m2 = np.asarray(final[c]["c_m2"]) / max(float(final[c]["c_count"]) - 1, 1)
            results[-1]["pair_effect_summary"] = {
                "mean": np.where(dead, np.nan, np.asarray(final[c]["c_mean"])),
                "sd": np.where(dead, np.nan, np.sqrt(m2)),
            }
    return results
