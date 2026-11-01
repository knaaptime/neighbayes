r"""JAX port of the exact separable NB flow sampler (:mod:`._flow_exact`).

The same sweep — ``θ`` along the Fisher information's eigenvectors, ``(a, b)``
along their warmup covariance's with ``θ`` sheared along its regression on
them, then ``α`` — compiled into one XLA program
per chunk of sweeps, the chains run thread-per-chain through
:func:`.._utils._jax_utils.run_chains_chunked`.  Periods whose full-rank columns
coincide share one *key* and its two-sided solve, as in the Pólya–Gamma
sampler this one replaced.  The warmup schedule reads ``tune`` from the data,
so fits of different lengths share one compiled program.
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
    _THETA_WIDTH,
)
from ._flow_structured import FlowDesignStructure, classify_flow_design


def _exact_parts(y, W_csc, struct: FlowDesignStructure, priors, tune):
    """Host side of the sweep: ``(static, data)``.

    ``static`` is what the compiled sweep closes over and keys its cache on;
    ``data`` every data-dependent array, passed as an argument (an array of
    data size closed over is embedded and constant-folded by XLA).
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
    y_vals, y_counts = padded_value_counts(y_np)
    keys = sorted({tuple(int(i) for i in struct.f_idx[t]) for t in range(T)})
    lu_solve, sar = _sar_solver_parts(W_csc, n)
    data = {
        "y": jnp.asarray(y_np),
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
    static = (
        n,
        T,
        k,
        tuple(int(c) for c in struct.cheap_cols),
        tuple(int(c) for c in struct.full_cols),
        len(keys),
        float(y_np.size),
        lu_solve,
        alpha_fixed is not None,
    )
    return static, data


def _exact_fns(n, T, k, cc_t, fc_t, n_keys, n_obs, lu_solve, has_alpha_fixed):
    """``sweep(state, key, tuning, data) -> (state, trace)`` for one structure."""
    import jax
    import jax.numpy as jnp
    from jax.scipy.special import gammaln

    from .._utils._jax_slice import jax_slice_along, jax_slice_sample_1d
    from ._flow_jax import _bind_sar_solver

    cc = np.asarray(cc_t, dtype=np.int32)
    fc = np.asarray(fc_t, dtype=np.int32)
    k_c, k_e = len(cc), len(fc)
    period_idx = np.arange(T)

    def bind(data):
        y = data["y"]
        F, U, V = data["F"], data["U"], data["V"]
        mu0, prec0 = data["mu0"], data["prec0"]
        ui, vi, f_idx = data["ui"], data["vi"], data["f_idx"]
        key_of_t, key_idx = data["key_of_t"], data["key_idx"]
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
            """``U = A⁻¹X/m`` at ``(a, b)``, one row per coefficient: ``(k, T n²)``."""
            Ufull, A, B = filtered(a, b, F)
            cols = jnp.zeros((k, T, n, n))
            if k_c:
                cheap = jnp.einsum("otj,dtj->jtod", A[:, ui], B[:, vi])
                cols = cols.at[cc].set(cheap)
            if k_e:
                cols = cols.at[fc].set(Ufull[f_idx].transpose(1, 0, 2, 3))
            return cols.reshape(k, -1) * jnp.exp(-(a + b))

        def eta_at(a, b, theta):
            """``η = A⁻¹Xθ/m`` at ``(a, b)``, flat over the periods."""
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

            return jax.lax.map(period, period_idx).ravel() * jnp.exp(-(a + b))

        def nb_ll(eta, alpha):
            return jnp.sum(y * eta - (y + alpha) * jnp.logaddexp(eta, jnp.log(alpha)))

        def log_prior(theta, s):
            return -0.5 * jnp.sum(prec0 * (theta * jnp.exp(-s) - mu0) ** 2)

        def alpha_log_density(log_a, y_dot_eta, mu):
            a = jnp.exp(log_a)
            log_mu_a = jnp.log(mu + a)
            ll = (
                data["y_counts"] @ gammaln(data["y_vals"] + a)
                - n_obs * gammaln(a)
                + y_dot_eta
                - y @ log_mu_a
                - a * log_mu_a.sum()
                + n_obs * a * log_a
            )
            nu, sig = data["alpha_nu"], data["alpha_sigma"]
            return log_a + ll - 0.5 * (nu + 1.0) * jnp.log1p(a * a / (nu * sig**2))

        def sweep(state, key, tuning):
            theta, a, b, alpha, it = (
                state["theta"],
                state["a"],
                state["b"],
                state["alpha"],
                state["it"],
            )
            k_th, k_thp, k_ab, k_abp, k_al = jax.random.split(key, 5)
            s = a + b

            # 1. θ | a, b, α along the Fisher information's eigenvectors.
            cols = columns(a, b)
            eta = theta @ cols

            def fisher(_):
                w = alpha * jax.nn.sigmoid(eta - jnp.log(alpha))
                H = (cols * w) @ cols.T + jnp.diag(prec0 * jnp.exp(-2.0 * s))
                lam, V_ = jnp.linalg.eigh(H)
                return V_, 2.0 / jnp.sqrt(jnp.maximum(lam, 1e-12))

            refresh = tuning & ((it < _FISHER_FIRST) | (it % _FISHER_EVERY == 0))
            dirs, widths = jax.lax.cond(
                refresh, fisher, lambda _: (state["th_dirs"], state["th_w"]), None
            )
            UD = dirs.T @ cols
            order = jax.random.permutation(k_thp, k)

            def th_step(i, carry):
                th, e, f = carry
                j = order[i]
                d, ud = dirs[:, j], UD[j]
                t, f, _, _ = jax_slice_along(
                    lambda t, c: (
                        nb_ll(e + t * ud, alpha) + log_prior(th + t * d, s),
                        None,
                        c,
                    ),
                    f,
                    None,
                    None,
                    widths[j],
                    jax.random.fold_in(k_th, i),
                )
                return th + t * d, e + t * ud, f

            f = nb_ll(eta, alpha) + log_prior(theta, s)
            theta, eta, f = jax.lax.fori_loop(0, k, th_step, (theta, eta, f))

            # 2. (a, b) with θ sheared along G, the Jacobian e^{-(k+1)(a+b)}.
            k1 = k + 1.0
            here = jnp.concatenate([jnp.stack([a, b]), theta])

            def g(z, carry):
                inside = jnp.all((z[:2] > data["ab_lo"]) & (z[:2] < data["ab_hi"]))
                z = jnp.where(inside, z, here)
                e = eta_at(z[0], z[1], z[2:])
                s_ = z[0] + z[1]
                val = nb_ll(e, alpha) + log_prior(z[2:], s_) - k1 * s_
                return jnp.where(inside & jnp.isfinite(val), val, -jnp.inf), e, carry

            ab_dirs, ab_w, shear = state["ab_dirs"], state["ab_w"], state["ab_shear"]
            order2 = jax.random.permutation(k_abp, 2)

            def ab_step(i, carry):
                z, gz, e = carry
                d = ab_dirs[:, order2[i]]
                v = jnp.concatenate([d, shear @ d])
                t, gz, e, _ = jax_slice_along(
                    lambda t, c: g(z + t * v, c),
                    gz,
                    e,
                    None,
                    ab_w[order2[i]],
                    jax.random.fold_in(k_ab, i),
                )
                return z + t * v, gz, e

            z, _, eta = jax.lax.fori_loop(0, 2, ab_step, (here, f - k1 * s, eta))
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

            # 3. α | η.
            if has_alpha_fixed:
                alpha = data["alpha_fixed"]
            else:
                y_dot_eta, mu = y @ eta, jnp.exp(eta)
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
            }
            rho_d, rho_o = -jnp.expm1(-a), -jnp.expm1(-b)
            return new, (rho_d, rho_o, theta * jnp.exp(-(a + b)), alpha, eta)

        return sweep

    def sweep(state, key, tuning, data):
        return bind(data)(state, key, tuning)

    return sweep


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
):
    """Run the exact separable NB flow sampler on the JAX backend.

    ``y`` (and ``X``, unless ``struct`` is given) are stacked time-first over
    ``T`` periods of ``n²`` flows.  Returns one dict per chain with keys
    ``rho_d, rho_o, rho_w, beta, alpha, log_lik``.
    """
    import jax
    import jax.numpy as jnp
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
    static, data = _exact_parts(y, W_csc, struct, priors, tune)

    def build():
        core = _exact_fns(*static)
        if store_log_lik:
            return core

        def without_eta(state, key, tuning, data):  # η is traced for log_lik only
            new, trace = core(state, key, tuning, data)
            return new, trace[:4]

        return without_eta

    sweep = cached_sweep(("nb_flow_exact", *static, store_log_lik), build)
    k = struct.k
    states = []
    for i in inits:
        a0, b0 = -np.log1p(-float(i.rho_d)), -np.log1p(-float(i.rho_o))
        states.append(
            {
                "theta": jnp.asarray(np.asarray(i.beta, float) * np.exp(a0 + b0)),
                "a": jnp.float64(a0),
                "b": jnp.float64(b0),
                "alpha": jnp.float64(float(i.alpha)),
                "it": jnp.int32(0),
                "th_dirs": jnp.eye(k),
                "th_w": jnp.full(k, _THETA_WIDTH),
                "ab_dirs": jnp.eye(2),
                "ab_w": jnp.full(2, _AB_WIDTH),
                "ab_shear": jnp.zeros((k, 2)),
                "w_cnt": jnp.int32(0),
                "w_mean": jnp.zeros(k + 2),
                "w_M2": jnp.zeros((k + 2, k + 2)),
            }
        )
    warm_keys = [jax.random.PRNGKey(int(s)) for s in jax_seeds]
    draw_keys = [jax.random.fold_in(jax.random.PRNGKey(int(s)), 1) for s in jax_seeds]
    _, traces = run_chains_chunked(
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
    y_np = np.asarray(y, dtype=np.float64)
    results = []
    for c in range(chains):
        log_lik = None
        if store_log_lik:
            eta = np.asarray(traces[4])[c, sl].reshape(len(alpha_all[c]), -1)
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
