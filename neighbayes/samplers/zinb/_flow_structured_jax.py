r"""JAX port of the structured separable ZINB flow sweep.

The same sweep as :mod:`._flow_structured` — selection equation, zero
allocation, count equation on the active cells, α — compiled into one XLA
program per chunk of sweeps and run thread-per-chain through
:func:`~neighbayes.samplers._utils._jax_utils.run_chains_chunked`.  Each
equation is a
:func:`~neighbayes.samplers.negbin_reduced._flow_structured_fe_jax.make_structured_equation`.
"""

from __future__ import annotations

import numpy as np

from ..negbin_reduced._flow_structured import classify_flow_design
from ..negbin_reduced._flow_structured_fe_jax import (
    make_structured_equation,
    welford_update,
)

_EQ_KEYS = ("beta", "tau", "C", "rho_d", "rho_o", "eta", "slice_widths")


def make_zinb_flow_sweep(
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
    keep_group_draws: bool = True,
    store_eta: bool = False,
):
    """Build ``sweep(state, key, tuning, data) -> (state, trace)`` for one chain.

    ``trace`` is ``(lam_d, lam_o, gamma, rho_d, rho_o, beta, tau, alpha[, C]
    [, eta_sel, eta_cnt])``.  Returns ``(sweep, sel, cnt, data)`` with the two
    equation namespaces.  The sweep closes over the structure only and is
    shared by every model of the same structure, so it compiles once per
    structure.
    """
    import jax.numpy as jnp

    from .._utils._jax_utils import cached_sweep

    sel = make_structured_equation(
        W_sel_csc,
        classify_flow_design(Z, n, T),
        priors.gamma_mu,
        priors.gamma_sigma,
        rho_bounds=priors.lam_bounds,
        prefix="s_",
    )
    cnt = make_structured_equation(
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
    data = {
        "y": jnp.asarray(np.asarray(y, dtype=np.float64)),
        "alpha_sigma": jnp.float64(priors.alpha_sigma),
        "alpha_nu": jnp.float64(priors.alpha_nu),
        "alpha_fixed": jnp.float64(alpha_fixed if alpha_fixed is not None else 1.0),
        **sel.data,
        **cnt.data,
    }
    static = (
        sel.static, cnt.static, alpha_fixed is not None, bool(keep_group_draws),
        bool(store_eta),
    )  # fmt: skip
    sweep = cached_sweep(("zinb_flow", *static), lambda: _zinb_sweep(*static))
    return sweep, sel, cnt, data


def _zinb_sweep(s_static, c_static, has_alpha_fixed, keep_group_draws, store_eta):
    """The structured ZINB flow sweep for one structure, taking ``data``."""
    import jax
    import jax.numpy as jnp
    from jax.scipy.special import gammaln

    from .._utils._jax_slice import jax_slice_sample_1d
    from .._utils._jax_utils import cached_sweep, make_pg_draw
    from ..negbin_reduced._flow_structured_fe_jax import _equation_fns

    sel = cached_sweep(("flow_equation", *s_static), lambda: _equation_fns(*s_static))
    cnt = cached_sweep(("flow_equation", *c_static), lambda: _equation_fns(*c_static))
    n, T = c_static[0], c_static[1]
    c_pairs, c_learn = c_static[5], c_static[6]
    shape = (T, n, n)
    draw_pg = make_pg_draw()

    def alpha_log_density(log_a, y_j, act, mu, data):
        alpha_sigma, alpha_nu = data["alpha_sigma"], data["alpha_nu"]
        a = jnp.exp(log_a)
        ll = (
            gammaln(y_j + a)
            - gammaln(a)
            + y_j * jnp.log(jnp.maximum(mu / (mu + a), 1e-300))
            + a * jnp.log(jnp.maximum(a / (mu + a), 1e-300))
        )
        prior = -0.5 * (alpha_nu + 1.0) * jnp.log1p(a * a / (alpha_nu * alpha_sigma**2))
        return log_a + jnp.sum(act * ll) + prior

    def sweep(state, key, tuning, data):
        y_j = data["y"]
        alpha, z = state["alpha"], state["z"]
        k_ps, k_s, k_z, k_pc, k_c, k_a, k_sd = jax.random.split(key, 7)

        # selection: PG-logit on the latent allocation
        eta_s = state["sel"]["eta"].ravel()
        om_s = draw_pg(jnp.ones_like(y_j), jnp.clip(eta_s, -30.0, 30.0), k_ps)
        sel_new = sel.update(
            state["sel"],
            om_s.reshape(shape),
            ((z - 0.5) / om_s).reshape(shape),
            k_s,
            tuning,
            data,
        )

        # zero allocation
        eta_s = sel_new["eta"].ravel()
        eta_c = state["cnt"]["eta"].ravel()
        mu = jnp.exp(jnp.clip(eta_c, -30.0, 30.0))
        pi = jax.nn.sigmoid(eta_s)
        nb0 = jnp.exp(jnp.clip(alpha * jnp.log(alpha / (mu + alpha)), -700.0, 0.0))
        num = pi * nb0
        p_z1 = jnp.where(num + (1.0 - pi) > 0, num / (num + (1.0 - pi)), 0.0)
        z = jnp.where(y_j > 0, 1.0, jax.random.bernoulli(k_z, p_z1).astype(jnp.float64))

        # count: active cells only (Ω = 0 on the structural zeros)
        om_raw = draw_pg(
            jnp.maximum(y_j + alpha, 1e-3),
            jnp.clip(eta_c - jnp.log(alpha), -30.0, 30.0),
            k_pc,
        )
        om_c = om_raw * z
        safe = jnp.where(z > 0, om_raw, 1.0)
        zc = jnp.where(z > 0, 0.5 * (y_j - alpha) / safe + jnp.log(alpha), 0.0)
        cnt_new = cnt.update(
            state["cnt"], om_c.reshape(shape), zc.reshape(shape), k_c, tuning, data
        )
        if c_learn:
            y3 = y_j.reshape(shape)
            es = sel_new["eta"]
            log_pi, log_1m = -jnp.logaddexp(0.0, -es), -jnp.logaddexp(0.0, es)
            la_ = jnp.log(alpha)

            def zinb_ll(e):
                lma = jnp.logaddexp(e, la_)
                lp0 = alpha * (la_ - lma)
                ll = jnp.where(
                    y3 > 0,
                    y3 * (e - lma) + lp0,
                    jnp.logaddexp(log_1m, log_pi + lp0),
                )
                return jnp.sum(ll)

            cnt_new = cnt.exact_sd_move(cnt_new, zinb_ll, k_sd, data)

        eta_c = cnt_new["eta"].ravel()
        mu = jnp.exp(jnp.clip(eta_c, -30.0, 30.0))
        if not has_alpha_fixed:
            log_alpha, _ = jax_slice_sample_1d(
                lambda la: alpha_log_density(la, y_j, z, mu, data),
                jnp.log(alpha),
                -10.0,
                10.0,
                key=k_a,
                w=0.5,
            )
            alpha = jnp.exp(log_alpha)
        else:
            alpha = data["alpha_fixed"]

        new = {"sel": sel_new, "cnt": cnt_new, "alpha": alpha, "z": z}
        if c_pairs and not keep_group_draws:
            new.update(welford_update(state, cnt_new["C"], tuning))
        trace = (
            sel_new["rho_d"],
            sel_new["rho_o"],
            sel_new["beta"],
            cnt_new["rho_d"],
            cnt_new["rho_o"],
            cnt_new["beta"],
            cnt_new["tau"],
            alpha,
        )
        if c_learn:
            trace = trace + (cnt_new["s_pair"],)
        if c_pairs and keep_group_draws:
            trace = trace + (cnt_new["C"],)
        if store_eta:
            trace = trace + (sel_new["eta"], cnt_new["eta"])
        return new, trace

    return sweep


def run_chains_jax_zinb_flow_structured(
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
    keep_group_draws: bool = True,
    jax_seeds=None,
    slice_width: float = 0.4,
    store_log_lik: bool = False,
):
    """Run the structured separable ZINB flow sampler on JAX, one dict per chain.

    The keys are those of
    :func:`~neighbayes.samplers.zinb._flow_structured.run_chain_zinb_flow_structured`.
    """
    import jax
    import jax.numpy as jnp

    from ..._jax_dispatch import ensure_x64
    from .._utils._jax_utils import run_chains_chunked
    from .._utils._sparsax_lu import set_sparsax_lu_cache_size
    from ._core import _zinb_loglik_pointwise

    ensure_x64()
    chains = len(jax_seeds)
    set_sparsax_lu_cache_size(max(32, 16 * chains))
    sweep, sel, cnt, data = make_zinb_flow_sweep(
        y,
        X,
        Z,
        W_csc,
        W_sel_csc,
        n,
        T,
        priors,
        pair_effects=pair_effects,
        period_effects=period_effects,
        keep_group_draws=keep_group_draws,
        store_eta=store_log_lik,
    )
    init_sel, init_cnt = sel.initial_eta, cnt.initial_eta
    y_np = np.asarray(y, dtype=np.float64)
    X, Z = np.asarray(X, dtype=np.float64), np.asarray(Z, dtype=np.float64)
    positive = y_np > 0
    share = float(np.clip(positive.mean() * 1.5, 0.05, 0.95))
    rows = positive if positive.sum() > X.shape[1] else np.ones(y_np.size, bool)
    b0 = np.linalg.lstsq(X[rows], np.log(y_np[rows] + 0.5), rcond=None)[0]
    const = np.flatnonzero(np.all(Z == Z[:1], axis=0) & (Z[0] != 0))
    C0 = np.zeros((n, n))
    if cnt.has_pairs:
        Y = y_np.reshape(T, n, n)
        C0 = np.log(Y.mean(axis=0) + 0.5) - np.log(Y.mean() + 0.5)
    w0 = (jnp.float64(slice_width),) * 2

    states = []
    for s in jax_seeds:
        rng = np.random.default_rng(int(s))
        g0 = rng.normal(0.0, 0.1, size=Z.shape[1])
        if const.size:
            g0[const[0]] += np.log(share / (1.0 - share)) / Z[0, const[0]]
        beta0 = b0 + rng.normal(0.0, 0.1, size=b0.size)
        eq_s = {
            "beta": jnp.asarray(g0),
            "tau": jnp.zeros(0),
            "C": jnp.zeros((n, n)),
            "rho_d": jnp.float64(rng.uniform(-0.1, 0.1)),
            "rho_o": jnp.float64(rng.uniform(-0.1, 0.1)),
            "slice_widths": w0,
        }
        eq_s["eta"] = init_sel(
            eq_s["beta"], eq_s["tau"], eq_s["C"], eq_s["rho_d"], eq_s["rho_o"], data
        )
        eq_c = {
            "beta": jnp.asarray(beta0),
            "tau": jnp.full(cnt.n_tau, cnt.mu_tau),
            "C": jnp.asarray(C0),
            "rho_d": jnp.float64(rng.uniform(-0.1, 0.1)),
            "rho_o": jnp.float64(rng.uniform(-0.1, 0.1)),
            "slice_widths": w0,
        }
        if cnt.learn_sd:
            eq_c["s_pair"] = jnp.float64(float(pair_effects[1]))
        eq_c["eta"] = init_cnt(
            eq_c["beta"], eq_c["tau"], eq_c["C"], eq_c["rho_d"], eq_c["rho_o"], data
        )
        z0 = np.where(positive, 1.0, rng.binomial(1, 0.5, size=y_np.size))
        alpha0 = 1.0 if priors.alpha_fixed is None else float(priors.alpha_fixed)
        st = {
            "sel": eq_s,
            "cnt": eq_c,
            "alpha": jnp.float64(alpha0),
            "z": jnp.asarray(z0, dtype=jnp.float64),
        }
        if cnt.has_pairs and not keep_group_draws:
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
    ld, lo, gam, rd, ro, beta, tau, alpha = traces[:8]
    nxt = 8
    sd_all = None
    if cnt.learn_sd:
        sd_all, nxt = traces[nxt], nxt + 1
    C_all = None
    if cnt.has_pairs and keep_group_draws:
        C_all = traces[nxt]
        nxt += 1
    results = []
    for c in range(chains):
        out = {
            "lam_d": ld[c],
            "lam_o": lo[c],
            "lam_w": -ld[c] * lo[c],
            "gamma": gam[c],
            "rho_d": rd[c],
            "rho_o": ro[c],
            "rho_w": -rd[c] * ro[c],
            "beta": beta[c],
            "time_effect": tau[c],
            "alpha": alpha[c],
            "log_lik": None,
        }
        if sd_all is not None:
            out["group_sd"] = sd_all[c]
        if C_all is not None:
            out["group_effect"] = C_all[c].reshape(draws, n * n)
        elif cnt.has_pairs:
            cnt_ = float(final[c]["c_count"])
            out["group_effect_mean"] = np.asarray(final[c]["c_mean"]).ravel()
            out["group_effect_sd"] = np.sqrt(
                np.asarray(final[c]["c_m2"]).ravel() / max(cnt_ - 1.0, 1.0)
            )
        if store_log_lik:
            es = traces[nxt][c].reshape(draws, -1)
            ec = traces[nxt + 1][c].reshape(draws, -1)
            out["log_lik"] = _zinb_loglik_pointwise(
                y_np[None, :], es, ec, alpha[c][:, None]
            )
        results.append(out)
    return results
