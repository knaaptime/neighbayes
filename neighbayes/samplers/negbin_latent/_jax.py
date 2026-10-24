r"""JAX backend of the latent SAR and SEM negative binomial sampler (:mod:`._core`).

The same sweep, with the log-means updated by colour class rather than one at a
time.  The conditional of :math:`\eta_i` touches the disturbances in the rows
of column ``i`` of :math:`A = I - \rho W`; log-means whose columns share no row
are conditionally independent, so the latent logit's colouring
(:func:`..logit_latent._jax.colour_classes`) splits them into classes, each
drawn at once and the classes in turn.  A slice per lane would run as long as
the slowest lane of its class, so a class is drawn instead by independence
Metropolis–Hastings from a *t* at each conditional's mode (:func:`_eta_mh`),
in a fixed number of passes.  The filter solves are BiCGSTAB from the nearby
solution at hand, falling back on the sparse LU should one stall; products with
:math:`W` are gathers from its padded rows, which on CPU beat scatter-adds.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from .._utils._jax_utils import cached_sweep, run_chains_chunked
from ..logit_latent._jax import colour_classes
from ._core import (
    _BETA_DF,
    _PAR_ACCEPT,
    _SOLVE_MAXITER,
    _SOLVE_RTOL,
    filter_columns,
    initial_state,
)


def _column_table(W):
    """Column ``i`` of ``A``'s pattern as row ``i`` of padded tables.

    Returns ``(rows, ones, w)`` of shape ``(n, m)``, ``m`` the largest column
    count; padding has ``rows = 0`` and zero values, so it contributes a
    constant to every conditional and nothing to an update.
    """
    indptr, rows, ones, wvals = filter_columns(W)
    n = W.shape[0]
    counts = np.diff(indptr)
    col = np.repeat(np.arange(n), counts)
    pos = np.arange(rows.size) - np.repeat(indptr[:-1], counts)
    shape = (n, int(counts.max()))
    R = np.zeros(shape, np.int32)
    O, V = np.zeros(shape), np.zeros(shape)
    R[col, pos], O[col, pos], V[col, pos] = rows, ones, wvals
    return R, O, V


def _row_table(W):
    """``W``'s rows as padded ``(cols, vals)`` tables, for products by gather."""
    W = sp.csr_matrix(W)
    n = W.shape[0]
    counts = np.diff(W.indptr)
    row = np.repeat(np.arange(n), counts)
    pos = np.arange(W.indices.size) - np.repeat(W.indptr[:-1], counts)
    shape = (n, int(counts.max()))
    C, V = np.zeros(shape, np.int32), np.zeros(shape)
    C[row, pos], V[row, pos] = W.indices, W.data
    return C, V


def _class_table(colour):
    """The members of each colour class, padded with the out-of-range index ``n``."""
    n = colour.size
    sizes = np.bincount(colour)
    order = np.argsort(colour, kind="stable")
    start = np.r_[0, np.cumsum(sizes)[:-1]]
    pos = np.arange(n) - np.repeat(start, sizes)
    table = np.full((sizes.size, int(sizes.max())), n, np.int32)
    table[colour[order], pos] = order
    return table


def _latent_data(y, X, W, priors, ld_params, sparse_arrays):
    """The read-only arrays every chain's sweep takes (``consts``)."""
    import jax.numpy as jnp

    W = sp.csr_matrix(W, dtype=np.float64)
    k = X.shape[1]
    WX = np.asarray(W @ X)
    beta_sigma = np.broadcast_to(np.asarray(priors.beta_sigma, float), (k,))
    R, O, V = _column_table(W)
    Wc, Wv = _row_table(W)
    return {
        "y": jnp.asarray(y, dtype=jnp.float64),
        "X": jnp.asarray(X, dtype=jnp.float64),
        "WX": jnp.asarray(WX),
        "XtXiXt": jnp.asarray(np.linalg.solve(X.T @ X, X.T)),
        "G0": jnp.asarray(X.T @ X),
        "G1": jnp.asarray(X.T @ WX + WX.T @ X),
        "G2": jnp.asarray(WX.T @ WX),
        "Wcols": jnp.asarray(Wc),
        "Wvals": jnp.asarray(Wv),
        "colrows": jnp.asarray(R),
        "colones": jnp.asarray(O),
        "colw": jnp.asarray(V),
        "classes": jnp.asarray(_class_table(colour_classes(W))),
        "b0": jnp.asarray(np.broadcast_to(np.asarray(priors.beta_mu, float), (k,))),
        "B0inv": jnp.asarray(np.diag(1.0 / beta_sigma**2)),
        "lo": jnp.float64(priors.rho_lower),
        "hi": jnp.float64(priors.rho_upper),
        "alpha_nu": jnp.float64(priors.alpha_nu),
        "alpha_sigma": jnp.float64(priors.alpha_sigma),
        "logdet": ld_params,
        "sparse": sparse_arrays,
    }


#: Newton steps to the mode of a log-mean's conditional, and the largest step.
_ETA_NEWTON = 3
_ETA_NEWTON_MAX_STEP = 5.0


def _eta_mh(x0, yi, e, Av, alpha, key, active):
    """Independence MH for the log-means of one colour class.

    ``e`` are the disturbances in the rows of each lane's column of ``A`` and
    ``Av`` that column's values.  The proposal is a *t* with 2 degrees of
    freedom, drawn by inversion, at the conditional mode with the curvature
    there as its scale; its tails are heavier than the conditional's on both
    sides.  Newton starts from the precision-weighted mean of the terms' own
    modes (the Poisson's at ``log(y + 1/2)``, each disturbance's at zero),
    which, like the steps, involves the other log-means only, so the proposal
    does not depend on the current value.  Lanes outside ``active`` keep it.
    """
    import jax
    import jax.numpy as jnp

    rest = e - Av * x0[:, None]

    def f(t):
        ee = rest + Av * t[:, None]
        return yi * t - jnp.exp(t) + alpha * jnp.sum(ee - jnp.exp(ee), axis=1)

    def grad_hess(t):
        ee = jnp.exp(rest + Av * t[:, None])
        et = jnp.exp(t)
        return (
            yi - et + alpha * jnp.sum(Av * (1.0 - ee), axis=1),
            et + alpha * jnp.sum(Av * Av * ee, axis=1),
        )

    yp = yi + 0.5
    t = (yp * jnp.log(yp) - alpha * jnp.sum(Av * rest, axis=1)) / (
        yp + alpha * jnp.sum(Av * Av, axis=1)
    )
    for _ in range(_ETA_NEWTON):
        g, h = grad_hess(t)
        t = t + jnp.clip(g / h, -_ETA_NEWTON_MAX_STEP, _ETA_NEWTON_MAX_STEP)
    sd = 1.0 / jnp.sqrt(grad_hess(t)[1])

    def logq(x):
        z = (x - t) / sd
        return -1.5 * jnp.log1p(0.5 * z * z)

    k_prop, k_acc = jax.random.split(key)
    u = jax.random.uniform(k_prop, x0.shape, minval=1e-12, maxval=1.0 - 1e-12)
    prop = t + sd * (2.0 * u - 1.0) / jnp.sqrt(2.0 * u * (1.0 - u))
    log_ratio = f(prop) - f(x0) + logq(x0) - logq(prop)
    accept = active & (jnp.log(jax.random.uniform(k_acc, x0.shape)) < log_ratio)
    return jnp.where(accept, prop, x0)


def _make_sweep(n, k, process, n_colours, alpha_fixed, row_std, logdet_eval, lu_solve):
    import jax
    import jax.numpy as jnp
    from jax.scipy.sparse.linalg import bicgstab
    from jax.scipy.special import digamma, gammaln, polygamma

    from .._utils._jax_slice import adapt_slice_width, jax_slice_sample_1d

    sar = process == "sar"

    def gamma_moments(a):
        return digamma(a) - jnp.log(a), jnp.sqrt(polygamma(1, a))

    def sweep(st, key, tuning, d):
        X, WX, y, b0, B0inv = d["X"], d["WX"], d["y"], d["b0"], d["B0inv"]
        nu, sig = d["alpha_nu"], d["alpha_sigma"]
        sparse = d["sparse"]

        def W(v):
            return jnp.sum(d["Wvals"] * v[d["Wcols"]], axis=1)

        def solve(r, b, x0):
            def A(v):
                return v - r * W(v)

            x, _ = bicgstab(A, b, x0=x0, tol=_SOLVE_RTOL, maxiter=_SOLVE_MAXITER)
            ok = jnp.linalg.norm(b - A(x)) <= 100.0 * _SOLVE_RTOL * jnp.linalg.norm(b)
            return jax.lax.cond(
                ok,
                lambda: x,
                lambda: lu_solve(
                    sparse["Ai"], sparse["Aj"], sparse["eye"] - r * sparse["w"], b
                ),
            )

        def poisson(eta):
            return y @ eta - jnp.sum(jnp.exp(eta))

        def gamma_stat(e):
            return jnp.sum(e - jnp.exp(e))

        def alpha_density(s, la):
            a = jnp.exp(la)
            return (
                n * (a * la - gammaln(a))
                + a * s
                - 0.5 * (nu + 1.0) * jnp.log1p(a * a / (nu * sig * sig))
                + la
            )

        par, beta, alpha, eta = st["par"], st["beta"], st["alpha"], st["eta"]
        keys = jax.random.split(key, 8)

        # 1. The log-means, by colour class.
        if sar:
            eps = eta - par * W(eta) - X @ beta
        else:
            r0 = eta - X @ beta
            eps = r0 - par * W(r0)
        aval = d["colones"] - par * d["colw"]

        def colour_step(c, carry):
            eta, eps = carry
            idx = d["classes"][c]
            active = idx < n

            def take(a):
                # The padding index ``n`` is out of range: clip, then mask.
                return jnp.take(a, idx, axis=0, mode="clip")

            x0, yi = take(eta), take(y)
            R, Av = take(d["colrows"]), take(aval)
            e = eps[R]

            x1 = _eta_mh(x0, yi, e, Av, alpha, jax.random.fold_in(keys[0], c), active)
            delta = jnp.where(active, x1 - x0, 0.0)
            eps = eps.at[R].add(Av * delta[:, None])
            return eta.at[idx].set(x1, mode="drop"), eps

        eta, eps = jax.lax.fori_loop(0, n_colours, colour_step, (eta, eps))

        # 2. The spatial parameter, given eta.
        Weta = W(eta)
        if sar:
            g = d["XtXiXt"] @ Weta
            u = beta + par * g
            e0 = eta - X @ u
            dv = Weta - X @ g

            def dens(r):
                b = u - r * g - b0
                e = e0 - r * dv
                return (
                    logdet_eval(r, d["logdet"])
                    + alpha * gamma_stat(e)
                    - 0.5 * b @ B0inv @ b
                )

            def beta_at(r):
                return u - r * g
        else:
            m = gamma_moments(alpha)[0]
            h0 = X.T @ (eta - m)
            h1 = X.T @ Weta + WX.T @ (eta - m)
            h2 = WX.T @ Weta

            def bhat(lam):
                return jnp.linalg.solve(
                    d["G0"] - lam * d["G1"] + lam * lam * d["G2"],
                    h0 - lam * h1 + lam * lam * h2,
                )

            u = beta - bhat(par)

            def dens(lam):
                b = u + bhat(lam)
                e = eta - lam * Weta - (X - lam * WX) @ b
                return (
                    logdet_eval(lam, d["logdet"])
                    + alpha * gamma_stat(e)
                    - 0.5 * (b - b0) @ B0inv @ (b - b0)
                )

            def beta_at(lam):
                return u + bhat(lam)

        width = st["slice_width"]
        par, _, left, right = jax_slice_sample_1d(
            dens, par, d["lo"], d["hi"], key=keys[1], w=width, return_steps=True
        )
        width = adapt_slice_width(width, left, right, tuning)
        beta = beta_at(par)
        c = eta - par * Weta
        Z = X if sar else X - par * WX

        # 3. beta: independence MH from a t at the Laplace approximation.
        def beta_logp(x):
            r = x - b0
            return alpha * gamma_stat(c - Z @ x) - 0.5 * r @ B0inv @ r

        m, s = gamma_moments(alpha)
        b = jnp.linalg.solve(Z.T @ Z / s**2 + B0inv, Z.T @ (c - m) / s**2 + B0inv @ b0)

        def newton(_, carry):
            b, lp = carry
            ee = jnp.exp(c - Z @ b)
            grad = alpha * (Z.T @ (ee - 1.0)) - B0inv @ (b - b0)
            H = alpha * (Z.T * ee) @ Z + B0inv
            stp = jnp.linalg.solve(H, grad)

            def halve(h):
                t = 0.5 * h[0]
                cand = b + t * stp
                return t, cand, beta_logp(cand)

            t, cand, lc = jax.lax.while_loop(
                lambda h: (h[2] < lp) & (h[0] >= 1e-6),
                halve,
                (1.0, b + stp, beta_logp(b + stp)),
            )
            return cand, lc

        def unconverged(st):
            b, lp, i, lp_old = st
            return (i < 20) & (jnp.abs(lp - lp_old) > 1e-10 * (1.0 + jnp.abs(lp)))

        def newton_step(st):
            b, lp, i, _ = st
            return (*newton(i, (b, lp)), i + 1, lp)

        lp = beta_logp(b)
        mode = jax.lax.while_loop(unconverged, newton_step, (b, lp, 0, lp + 1.0))[0]
        H = alpha * (Z.T * jnp.exp(c - Z @ mode)) @ Z + B0inv
        L = jnp.linalg.cholesky(H)

        def logq(x):
            zz = L.T @ (x - mode)
            return -0.5 * (_BETA_DF + k) * jnp.log1p(zz @ zz / _BETA_DF)

        k_n, k_c, k_u = jax.random.split(keys[2], 3)
        scale = jnp.sqrt(2.0 * jax.random.gamma(k_c, _BETA_DF / 2.0) / _BETA_DF)
        prop = (
            mode
            + jax.scipy.linalg.solve_triangular(
                L.T, jax.random.normal(k_n, (k,)), lower=False
            )
            / scale
        )
        log_ratio = beta_logp(prop) - beta_logp(beta) + logq(beta) - logq(prop)
        beta = jnp.where(jnp.log(jax.random.uniform(k_u)) < log_ratio, prop, beta)
        eps = c - Z @ beta

        sxb, s1 = st["sxb"], st["s1"]
        if not alpha_fixed:
            # 4. alpha, given the disturbances.
            stat = gamma_stat(eps)
            la, _ = jax_slice_sample_1d(
                lambda la: alpha_density(stat, la),
                jnp.log(alpha),
                -10.0,
                10.0,
                key=keys[3],
                w=0.5,
            )
            alpha = jnp.exp(la)

            # 5. alpha with the standardized disturbances held.
            ones = jnp.ones(n)
            s1 = ones / (1.0 - par) if row_std else solve(par, ones, s1)
            sxb = solve(par, X @ beta, sxb) if sar else X @ beta
            m, s = gamma_moments(alpha)
            z = (eps - m) / s
            dev = eta - sxb

            def move(la):
                mm, ss = gamma_moments(jnp.exp(la))
                bb = ss / s
                return mm + ss * z, sxb + (mm - bb * m) * s1 + bb * dev, ss

            def inter_density(la):
                e, et, ss = move(la)
                return poisson(et) + alpha_density(gamma_stat(e), la) + n * jnp.log(ss)

            la, _ = jax_slice_sample_1d(
                inter_density, jnp.log(alpha), -10.0, 10.0, key=keys[4], w=0.5
            )
            alpha = jnp.exp(la)
            eta = move(la)[1]

        # 6. The spatial parameter with the disturbances held.
        if sar:
            xi = eta - par * W(eta)
            cur, base = eta, jnp.zeros(n)
        else:
            cur, base = eta - X @ beta, X @ beta
            xi = cur - par * W(cur)
        step = st["step"]
        prop = par + step * jax.random.normal(keys[5])
        inside = (prop > d["lo"]) & (prop < d["hi"])
        new = solve(jnp.where(inside, prop, par), xi, cur)
        log_ratio = poisson(base + new) - poisson(base + cur)
        accepted = inside & (jnp.log(jax.random.uniform(keys[6])) < log_ratio)
        par = jnp.where(accepted, prop, par)
        eta = jnp.where(accepted, base + new, eta)
        it = st["it"]
        step = jnp.where(
            tuning,
            step * jnp.exp((accepted - _PAR_ACCEPT) / jnp.sqrt(it + 1.0)),
            step,
        )

        out = {
            "par": par,
            "beta": beta,
            "alpha": alpha,
            "eta": eta,
            "sxb": sxb,
            "s1": s1,
            "step": step,
            "slice_width": width,
            "it": it + 1.0,
        }
        return out, (par, beta, alpha)

    return sweep


def run_chains_jax_latent_negbin(
    *,
    y,
    X,
    W,
    process,
    priors,
    logdet_kind,
    logdet_params,
    draws,
    tune,
    thin=1,
    seeds,
    slice_width=0.4,
):
    """Run the latent-NB chains in parallel threads; one dict per chain."""
    import jax
    import jax.numpy as jnp

    from ..._jax_dispatch import ensure_x64
    from ..._logdet._jax import logdet_param_evaluator
    from ..negbin_reduced._jax import _build_sparse_ctx, _sparse_solve_parts

    ensure_x64()
    n, k = X.shape
    W = sp.csr_matrix(W, dtype=np.float64)
    if logdet_kind == "closure":
        fn = logdet_params

        def logdet_eval(r, _params):
            return fn(r)

        ld_params, ld_key = jnp.float64(0.0), ("closure", id(fn))
    else:
        logdet_eval = logdet_param_evaluator(logdet_kind)
        ld_params, ld_key = logdet_params, (logdet_kind,)
    lu_solve, sparse_arrays = _sparse_solve_parts(_build_sparse_ctx(W, n))
    sparse_arrays = {key: sparse_arrays[key] for key in ("Ai", "Aj", "eye", "w")}
    data = _latent_data(y, X, W, priors, ld_params, sparse_arrays)
    n_colours = int(data["classes"].shape[0])
    alpha_fixed = priors.alpha_fixed is not None
    row_std = bool(np.allclose(np.asarray(W.sum(axis=1)).ravel(), 1.0))

    static = (n, k, process, n_colours, alpha_fixed, row_std, lu_solve, *ld_key)
    sweep = cached_sweep(
        ("negbin_latent", *static),
        lambda: _make_sweep(
            n, k, process, n_colours, alpha_fixed, row_std, logdet_eval, lu_solve
        ),
    )
    eta0, beta0, alpha0 = initial_state(np.asarray(y, dtype=np.float64), X)
    if alpha_fixed:
        alpha0 = float(priors.alpha_fixed)
    states = [
        {
            "par": jnp.float64(0.0),
            "beta": jnp.asarray(beta0),
            "alpha": jnp.float64(alpha0),
            "eta": jnp.asarray(eta0),
            "sxb": jnp.asarray(X @ beta0),
            "s1": jnp.ones(n),
            "step": jnp.float64(0.1),
            "slice_width": jnp.float64(slice_width),
            "it": jnp.float64(0.0),
        }
        for _ in seeds
    ]
    warm = [jax.random.PRNGKey(int(s)) for s in seeds]
    draw = [jax.random.fold_in(jax.random.PRNGKey(int(s)), 1) for s in seeds]
    _, (par_all, beta_all, alpha_all) = run_chains_chunked(
        sweep, states, warm, draw, tune=tune, draws=draws, consts=data
    )
    name = "rho" if process == "sar" else "lam"
    sl = slice(None, None, thin) if thin > 1 else slice(None)
    return [
        {name: par_all[c, sl], "beta": beta_all[c, sl], "alpha": alpha_all[c, sl]}
        for c in range(len(seeds))
    ]
