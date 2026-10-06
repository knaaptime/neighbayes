r"""JAX port of the structured separable hurdle flow sweep.

The same sweep as :mod:`._flow_structured` — binary half on every cell, count
half on the positive cells (truncation augmented), then the count level and
``log α`` jointly — compiled into one XLA program per chunk of sweeps and run
thread-per-chain through
:func:`~neighbayes.samplers._utils._jax_utils.run_chains_chunked`.  Each half
is a
:func:`~neighbayes.samplers.negbin_reduced._flow_structured_fe_jax.make_structured_equation`.

The joint level/α slice runs along the eigenvectors of the running
covariance of ``(level, log α)``, accumulated over the warmup sweeps and
frozen afterwards.
"""

from __future__ import annotations

import numpy as np

from ..negbin_reduced._flow_structured import classify_flow_design
from ..negbin_reduced._flow_structured_fe_jax import make_structured_equation


def _const_col(M):
    const = np.flatnonzero(np.all(M == M[:1], axis=0) & (M[0] != 0))
    return (int(const[0]), float(M[0, const[0]])) if const.size else (None, 1.0)


def make_hurdle_flow_sweep(
    y,
    X,
    Z,
    W_csc,
    W_sel_csc,
    n: int,
    T: int,
    priors,
    *,
    pair_effects=None,
    period_effects=None,
    sel_pair_effects=None,
    sel_period_effects=None,
    keep_group_draws: bool = True,
    store_eta: bool = False,
):
    """Build ``sweep(state, key, tuning, data) -> (state, trace)`` for one chain.

    ``trace`` is ``(lam_d, lam_o, gamma, sel_tau, rho_d, rho_o, beta, tau,
    alpha[, C_b, C_c][, eta_b, eta_c])``.  Returns ``(sweep, binary, count,
    data)``.  The sweep closes over the structure only and is shared by every
    model of the same structure, so it compiles once per structure.
    """
    import jax.numpy as jnp

    from .._utils._jax_utils import cached_sweep

    binary = make_structured_equation(
        W_sel_csc,
        classify_flow_design(Z, n, T),
        priors.gamma_mu,
        priors.gamma_sigma,
        rho_bounds=priors.lam_bounds,
        pair_effects=sel_pair_effects,
        period_effects=sel_period_effects,
        prefix="b_",
    )
    count = make_structured_equation(
        W_csc,
        classify_flow_design(X, n, T),
        priors.beta_mu,
        priors.beta_sigma,
        rho_bounds=priors.rho_bounds,
        pair_effects=pair_effects,
        period_effects=period_effects,
        prefix="c_",
    )
    alpha_fixed = getattr(priors, "alpha_fixed", None)
    level_col, level_val = _const_col(np.asarray(X, dtype=np.float64))
    # The intercept first: shifting pooled pair effects would move their mean
    # off the prior's, which is what their sd is estimated from.
    if level_col is not None:
        carrier = "beta"
    elif count.n_tau == T:
        carrier = "tau"
    elif count.has_pairs:
        carrier = "C"
    else:
        carrier = None
    y_np = np.asarray(y, dtype=np.float64)
    data = {
        "y": jnp.asarray(y_np),
        "pos": jnp.asarray(y_np > 0),
        "a_sigma": jnp.float64(priors.alpha_sigma),
        "a_nu": jnp.float64(priors.alpha_nu),
        "level_val": jnp.float64(level_val),
        **binary.data,
        **count.data,
    }
    static = (
        binary.static, count.static, alpha_fixed is not None, carrier, level_col,
        bool(keep_group_draws), bool(store_eta),
    )  # fmt: skip
    sweep = cached_sweep(("hurdle_flow", *static), lambda: _hurdle_sweep(*static))
    return sweep, binary, count, data


def _hurdle_sweep(
    b_static, c_static, has_alpha_fixed, carrier, level_col, keep_group_draws,
    store_eta,
):  # fmt: skip
    """The structured hurdle flow sweep for one structure, taking ``data``."""
    import jax
    import jax.numpy as jnp

    from .._utils._jax_slice import jax_slice_sample_1d
    from .._utils._jax_utils import cached_sweep, make_pg_draw
    from ..negbin_reduced._flow_structured_fe_jax import _equation_fns
    from ._truncated import make_truncated_jax

    binary = cached_sweep(
        ("flow_equation", *b_static), lambda: _equation_fns(*b_static)
    )
    count = cached_sweep(("flow_equation", *c_static), lambda: _equation_fns(*c_static))
    n, T = c_static[0], c_static[1]
    b_pairs, b_learn = b_static[5], b_static[6]
    c_pairs, c_learn = c_static[5], c_static[6]
    working_data, alpha_update_1d, _ = make_truncated_jax()
    shape = (T, n, n)
    draw_pg = make_pg_draw()
    move = not has_alpha_fixed and carrier is not None
    one_d = not has_alpha_fixed and carrier is None

    def carrier_log_prior(eq, t, data):
        if carrier == "C":
            s = eq["s_pair"] if c_learn else data["c_s_pair"]
            return -0.5 * jnp.sum((eq["C"] + t - data["c_m_pair"]) ** 2) / s**2
        if carrier == "tau":
            return (
                -0.5
                * jnp.sum((eq["tau"] + t - data["c_mu_tau"]) ** 2)
                / data["c_s_tau"] ** 2
            )
        j = level_col
        return -0.5 * data["c_prec0"][j] * (eq["beta"][j] + t - data["c_mu0"][j]) ** 2

    def apply_shift(eq, t, v):
        eq = dict(eq)
        if carrier == "C":
            eq["C"] = eq["C"] + t
        elif carrier == "tau":
            eq["tau"] = eq["tau"] + t
        else:
            eq["beta"] = eq["beta"].at[level_col].add(t)
        eq["eta"] = eq["eta"] + t * v[None]
        return eq

    def tnb_loglik(y_j, pos, eta, a):
        from jax.scipy.special import gammaln

        la = jnp.log(a)
        log_mu_a = jnp.logaddexp(eta, la)
        log_p0 = jnp.minimum(a * (la - log_mu_a), -1e-300)
        log1m = jnp.where(
            log_p0 > -0.6931471805599453,
            jnp.log(-jnp.expm1(log_p0)),
            jnp.log1p(-jnp.exp(log_p0)),
        )
        ll = gammaln(y_j + a) - gammaln(a) + y_j * (eta - log_mu_a) + log_p0 - log1m
        return jnp.sum(jnp.where(pos, ll, 0.0))

    def joint_log_density(th, eq, v, data):
        y_j, pos = data["y"], data["pos"]
        a_sigma, a_nu = data["a_sigma"], data["a_nu"]
        t, la = th[0], th[1]
        a = jnp.exp(la)
        eta = eq["eta"].ravel() + t * jnp.broadcast_to(v[None], shape).ravel()
        prior = -0.5 * (a_nu + 1.0) * jnp.log1p(a * a / (a_nu * a_sigma**2))
        val = tnb_loglik(y_j, pos, eta, a) + carrier_log_prior(eq, t, data) + la + prior
        inside = (la > -10.0) & (la < 10.0)
        return jnp.where(inside & jnp.isfinite(val), val, -jnp.inf)

    def welford(state, prefix, C, tuning):
        cnt = state[prefix + "count"] + jnp.where(tuning, 0.0, 1.0)
        delta = C - state[prefix + "mean"]
        mean = jnp.where(
            tuning, state[prefix + "mean"], state[prefix + "mean"] + delta / cnt
        )
        m2 = jnp.where(
            tuning, state[prefix + "m2"], state[prefix + "m2"] + delta * (C - mean)
        )
        return {prefix + "count": cnt, prefix + "mean": mean, prefix + "m2": m2}

    def sweep(state, key, tuning, data):
        y_j, pos = data["y"], data["pos"]
        d = pos.astype(jnp.float64)
        alpha = state["alpha"]
        k_pb, k_b, k_tc, k_c, k_lv, k_sb, k_sc = jax.random.split(key, 7)

        # binary: PG-logit on every cell
        om_b = draw_pg(
            jnp.ones_like(y_j), jnp.clip(state["bin"]["eta"].ravel(), -30.0, 30.0), k_pb
        )
        bin_new = binary.update(
            state["bin"],
            om_b.reshape(shape),
            ((d - 0.5) / om_b).reshape(shape),
            k_b,
            tuning,
            data,
        )
        d3 = d.reshape(shape)
        bin_new = binary.exact_sd_move(
            bin_new, lambda e: jnp.sum(d3 * e - jnp.logaddexp(0.0, e)), k_sb, data
        )

        # count: positive cells, truncation augmented; Ω = 0 elsewhere
        om_c, z_c = working_data(
            y_j, state["cnt"]["eta"].ravel(), pos, alpha, k_tc, draw_pg
        )
        cnt_new = count.update(
            state["cnt"], om_c.reshape(shape), z_c.reshape(shape), k_c, tuning, data
        )
        y3, pos3 = y_j.reshape(shape), pos.reshape(shape)
        cnt_new = count.exact_sd_move(
            cnt_new, lambda e: tnb_loglik(y3, pos3, e, alpha), k_sc, data
        )

        new = dict(state)
        new["bin"] = bin_new
        if move:
            v = (
                jnp.ones((n, n))
                if carrier in ("C", "tau")
                else data["level_val"]
                * count.level_vector(cnt_new["rho_d"], cnt_new["rho_o"], data)
            )
            dirs, widths = state["lv_dirs"], state["lv_widths"]
            th = jnp.array([0.0, jnp.log(alpha)])
            keys = jax.random.split(k_lv, 2)
            for j in range(2):
                dj = dirs[:, j]
                s, _ = jax_slice_sample_1d(
                    lambda s, _th=th, _d=dj: joint_log_density(
                        _th + s * _d, cnt_new, v, data
                    ),
                    0.0,
                    -1e3,
                    1e3,
                    key=keys[j],
                    w=widths[j],
                )
                th = th + s * dj
            cnt_new = apply_shift(cnt_new, th[0], v)
            alpha = jnp.exp(th[1])
            # running (level, log α) covariance over warmup → slice directions
            vb = jnp.broadcast_to(v[None], shape).ravel()
            level = jnp.sum(jnp.where(pos, cnt_new["eta"].ravel(), 0.0)) / jnp.sum(
                jnp.where(pos, vb, 0.0)
            )
            x = jnp.array([level, th[1]])
            n_lv = state["lv_n"] + jnp.where(tuning, 1.0, 0.0)
            dx = x - state["lv_mean"]
            mean = jnp.where(tuning, state["lv_mean"] + dx / n_lv, state["lv_mean"])
            m2 = jnp.where(
                tuning, state["lv_m2"] + jnp.outer(dx, x - mean), state["lv_m2"]
            )
            cov = m2 / jnp.maximum(n_lv - 1.0, 1.0) + 1e-10 * jnp.eye(2)
            ev, vecs = jnp.linalg.eigh(cov)
            refresh = tuning & (n_lv > 20.0)
            new.update(
                lv_n=n_lv,
                lv_mean=mean,
                lv_m2=m2,
                lv_dirs=jnp.where(refresh, vecs, dirs),
                lv_widths=jnp.where(
                    refresh, 2.0 * jnp.sqrt(jnp.maximum(ev, 1e-12)), widths
                ),
            )
        elif one_d:
            alpha = alpha_update_1d(
                alpha, y_j, cnt_new["eta"].ravel(), pos, k_lv, data["a_sigma"],
                data["a_nu"],
            )  # fmt: skip
        new["cnt"] = cnt_new
        new["alpha"] = alpha

        if b_pairs and not keep_group_draws:
            new.update(welford(state, "b_", bin_new["C"], tuning))
        if c_pairs and not keep_group_draws:
            new.update(welford(state, "c_", cnt_new["C"], tuning))
        trace = (
            bin_new["rho_d"],
            bin_new["rho_o"],
            bin_new["beta"],
            bin_new["tau"],
            cnt_new["rho_d"],
            cnt_new["rho_o"],
            cnt_new["beta"],
            cnt_new["tau"],
            alpha,
        )
        for learn, new_ in ((b_learn, bin_new), (c_learn, cnt_new)):
            if learn:
                trace = trace + (new_["s_pair"],)
        if keep_group_draws:
            if b_pairs:
                trace = trace + (bin_new["C"],)
            if c_pairs:
                trace = trace + (cnt_new["C"],)
        if store_eta:
            trace = trace + (bin_new["eta"], cnt_new["eta"])
        return new, trace

    return sweep


def run_chains_jax_hurdle_flow_structured(
    y,
    X,
    Z,
    W_csc,
    W_sel_csc,
    n: int,
    T: int,
    priors,
    draws: int,
    tune: int,
    *,
    pair_effects=None,
    period_effects=None,
    sel_pair_effects=None,
    sel_period_effects=None,
    keep_group_draws: bool = True,
    jax_seeds=None,
    slice_width: float = 0.4,
    store_log_lik: bool = False,
):
    """Run the structured separable hurdle flow sampler on JAX, one dict per chain.

    The keys are those of
    :func:`~neighbayes.samplers.hurdle._flow_structured.run_chain_hurdle_flow_structured`.
    """
    import jax
    import jax.numpy as jnp

    from ..._jax_dispatch import ensure_x64
    from .._utils._jax_utils import run_chains_chunked
    from .._utils._sparsax_lu import set_sparsax_lu_cache_size
    from ._truncated import hurdle_loglik_pointwise

    ensure_x64()
    chains = len(jax_seeds)
    set_sparsax_lu_cache_size(max(32, 16 * chains))
    sweep, binary, count, data = make_hurdle_flow_sweep(
        y, X, Z, W_csc, W_sel_csc, n, T, priors,
        pair_effects=pair_effects, period_effects=period_effects,
        sel_pair_effects=sel_pair_effects, sel_period_effects=sel_period_effects,
        keep_group_draws=keep_group_draws, store_eta=store_log_lik,
    )  # fmt: skip
    init_b, init_c = binary.initial_eta, count.initial_eta
    y_np = np.asarray(y, dtype=np.float64)
    X, Z = np.asarray(X, dtype=np.float64), np.asarray(Z, dtype=np.float64)
    pos = y_np > 0
    shape = (T, n, n)
    target = np.where(pos, np.log(3.0), -np.log(3.0))
    g0 = np.linalg.lstsq(Z, target, rcond=None)[0]
    rows = pos if pos.sum() > X.shape[1] else np.ones(y_np.size, bool)
    b0 = np.linalg.lstsq(X[rows], np.log(y_np[rows] + 0.5), rcond=None)[0]
    Cb0 = (target - Z @ g0).reshape(shape).mean(axis=0) if binary.has_pairs else None
    Cc0 = np.zeros((n, n))
    if count.has_pairs:
        Y, P = y_np.reshape(shape), pos.reshape(shape)
        cnt = P.sum(axis=0)
        mean_pos = np.where(cnt > 0, (Y * P).sum(axis=0) / np.maximum(cnt, 1), 1.0)
        Cc0 = np.log(mean_pos) - np.log(y_np[pos].mean())
    w0 = (jnp.float64(slice_width),) * 2
    alpha0 = 1.0 if priors.alpha_fixed is None else float(priors.alpha_fixed)

    states = []
    for s in jax_seeds:
        rng = np.random.default_rng(int(s))
        eqs = {}
        for name, eq, b_start, C_start, init, pe in (
            ("bin", binary, g0, Cb0, init_b, sel_pair_effects),
            ("cnt", count, b0, Cc0, init_c, pair_effects),
        ):
            st = {
                "beta": jnp.asarray(b_start + rng.normal(0.0, 0.1, size=b_start.size)),
                "tau": jnp.full(eq.n_tau, eq.mu_tau),
                "C": jnp.asarray(C_start if C_start is not None else np.zeros((n, n))),
                "rho_d": jnp.float64(rng.uniform(-0.1, 0.1)),
                "rho_o": jnp.float64(rng.uniform(-0.1, 0.1)),
                "slice_widths": w0,
            }
            if eq.learn_sd:
                st["s_pair"] = jnp.float64(float(pe[1]))
            st["eta"] = init(
                st["beta"], st["tau"], st["C"], st["rho_d"], st["rho_o"], data
            )
            eqs[name] = st
        state = {"bin": eqs["bin"], "cnt": eqs["cnt"], "alpha": jnp.float64(alpha0)}
        state.update(
            lv_n=jnp.float64(0.0),
            lv_mean=jnp.zeros(2),
            lv_m2=jnp.zeros((2, 2)),
            lv_dirs=jnp.eye(2),
            lv_widths=jnp.full(2, 0.5),
        )
        for prefix, eq in (("b_", binary), ("c_", count)):
            if eq.has_pairs and not keep_group_draws:
                state.update(
                    {
                        prefix + "count": jnp.float64(0.0),
                        prefix + "mean": jnp.zeros((n, n)),
                        prefix + "m2": jnp.zeros((n, n)),
                    }
                )
        states.append(state)

    warm_keys = [jax.random.PRNGKey(int(s)) for s in jax_seeds]
    draw_keys = [jax.random.fold_in(jax.random.PRNGKey(int(s)), 1) for s in jax_seeds]
    final, traces = run_chains_chunked(
        sweep, states, warm_keys, draw_keys, tune=tune, draws=draws, consts=data
    )
    ld, lo, gam, stau, rd, ro, beta, tau, alpha = traces[:9]
    nxt = 9
    sds = {}
    for key, eq in (("sel_group_sd", binary), ("group_sd", count)):
        if eq.learn_sd:
            sds[key], nxt = traces[nxt], nxt + 1
    Cb_all = Cc_all = None
    if keep_group_draws:
        if binary.has_pairs:
            Cb_all, nxt = traces[nxt], nxt + 1
        if count.has_pairs:
            Cc_all, nxt = traces[nxt], nxt + 1

    def summary(fin, prefix):
        cnt_ = float(fin[prefix + "count"])
        return (
            np.asarray(fin[prefix + "mean"]).ravel(),
            np.sqrt(np.asarray(fin[prefix + "m2"]).ravel() / max(cnt_ - 1.0, 1.0)),
        )

    results = []
    for c in range(chains):
        out = {
            "lam_d": ld[c],
            "lam_o": lo[c],
            "lam_w": -ld[c] * lo[c],
            "gamma": gam[c],
            "sel_time_effect": stau[c],
            "rho_d": rd[c],
            "rho_o": ro[c],
            "rho_w": -rd[c] * ro[c],
            "beta": beta[c],
            "time_effect": tau[c],
            "alpha": alpha[c],
            "log_lik": None,
        }
        out.update({key: val[c] for key, val in sds.items()})
        for pre, eq, all_ in (("sel_", binary, Cb_all), ("", count, Cc_all)):
            if not eq.has_pairs:
                continue
            if all_ is not None:
                out[pre + "group_effect"] = all_[c].reshape(draws, n * n)
            else:
                mean, sd = summary(final[c], "b_" if pre else "c_")
                out[pre + "group_effect_mean"] = mean
                out[pre + "group_effect_sd"] = sd
        if store_log_lik:
            eb = traces[nxt][c].reshape(draws, -1)
            ec = traces[nxt + 1][c].reshape(draws, -1)
            out["log_lik"] = hurdle_loglik_pointwise(
                y_np[None, :], eb, ec, alpha[c][:, None]
            )
        results.append(out)
    return results
