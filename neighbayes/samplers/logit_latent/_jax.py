r"""JAX backend of the latent SAR and SEM logit sampler (:mod:`._core`).

The same exact sweep, with the latent utilities updated by colour rather than
one at a time.  Utilities ``i`` and ``j`` interact only through
:math:`Q_{ij} = [A^\top P A]_{ij}`, which is zero unless some row of
:math:`A = I - \rho W` touches both, so a colouring of the pattern of
:math:`(I + |W|)^\top (I + |W|)` splits them into classes that are
conditionally independent: a class is drawn at once, exactly, from

.. math::
    y^*_i \mid \cdot \sim N\!\left(y^*_i - g_i / Q_{ii},\; 1 / Q_{ii}\right)
    \text{ on the side of } 0 \text{ that } y_i \text{ fixes},
    \qquad g = A^\top P \varepsilon,

and the classes are swept in turn.  A class reads :math:`\varepsilon` on its
members' columns of :math:`A` only and its draws move it there only, so a
sweep is one pass over the nonzeros of :math:`A` whatever the number of
colours.  The Kolmogorov scales come from
:func:`augax.ks_sample`, on the device.  A zero-inflated count's zero (``sign``
0, ``mixed``) has its side of zero drawn first, the positive side weighted by
``exp(logq)``, as in :func:`._core._orthant_sweep`.
"""

from __future__ import annotations

import numba
import numpy as np
import scipy.sparse as sp

from .._utils._jax_utils import cached_sweep, probe_midpoint, run_chains_chunked
from ..negbin_latent._core import filter_columns


@numba.njit(cache=True)
def _greedy_colour(indptr, indices, n):
    """Greedy colouring of a symmetric CSR pattern, highest degree first."""
    degree = indptr[1:] - indptr[:-1]
    order = np.argsort(-degree)
    colour = -np.ones(n, dtype=np.int64)
    used = np.zeros(n + 1, dtype=np.int64) - 1
    for v in order:
        for p in range(indptr[v], indptr[v + 1]):
            c = colour[indices[p]]
            if c >= 0:
                used[c] = v
        c = 0
        while used[c] == v:
            c += 1
        colour[v] = c
    return colour


def colour_classes(W) -> np.ndarray:
    """Colour of each utility: classes are conditionally independent given the rest."""
    W = sp.csr_matrix(W)
    n = W.shape[0]
    B = (sp.identity(n, format="csr") + (W != 0).astype(np.float64)).astype(bool)
    C = (B.T @ B).tocsr()
    C.setdiag(False)
    C.eliminate_zeros()
    return _greedy_colour(C.indptr.astype(np.int64), C.indices.astype(np.int64), n)


def class_table(colour):
    """The members of each colour class, padded with the out-of-range index ``n``."""
    n = colour.size
    sizes = np.bincount(colour)
    order = np.argsort(colour, kind="stable")
    start = np.r_[0, np.cumsum(sizes)[:-1]]
    pos = np.arange(n) - np.repeat(start, sizes)
    table = np.full((sizes.size, int(sizes.max())), n, np.int32)
    table[colour[order], pos] = order
    return table


def column_table(W):
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


def _latent_data(y, X, W, priors, ld_params, sign=None, unit_effects=None):
    """The read-only arrays every chain's sweep takes (``consts``).

    ``sign`` is each utility's side of zero, that of ``y`` by default;
    ``unit_effects`` adds a SAR's random unit effects.
    """
    import jax.numpy as jnp

    W = sp.csr_matrix(W)
    coo = W.tocoo()
    k = X.shape[1]
    beta_sigma = np.broadcast_to(np.asarray(priors.beta_sigma, float), (k,))
    B0inv = np.diag(1.0 / beta_sigma**2)
    b0 = np.broadcast_to(np.asarray(priors.beta_mu, float), (k,))
    if sign is None:
        sign = np.where(np.asarray(y) > 0, 1.0, -1.0)
    R, O, V = column_table(W)
    fx = unit_effects
    effects = (
        {
            "group": jnp.asarray(fx.group, dtype=jnp.int32),
            "c_mu": jnp.float64(fx.mu),
            "c_scale": jnp.float64(fx.sd_scale),
        }
        if fx is not None
        else {}
    )
    return {
        **effects,
        "sign": jnp.asarray(sign, dtype=jnp.float64),
        "X": jnp.asarray(X, dtype=jnp.float64),
        "WX": jnp.asarray(W @ X, dtype=jnp.float64),
        "rows": jnp.asarray(coo.row, dtype=jnp.int32),
        "cols": jnp.asarray(coo.col, dtype=jnp.int32),
        "vals": jnp.asarray(coo.data, dtype=jnp.float64),
        "colrows": jnp.asarray(R),
        "colones": jnp.asarray(O),
        "colw": jnp.asarray(V),
        "classes": jnp.asarray(class_table(colour_classes(W))),
        "b0": jnp.asarray(b0),
        "B0inv": jnp.asarray(B0inv),
        "B0inv_b0": jnp.asarray(B0inv @ b0),
        "lo": jnp.float64(priors.rho_lower),
        "hi": jnp.float64(priors.rho_upper),
        "logdet": ld_params,
        # The scale move's draw is a gamma when the prior means are zero.
        "scale_gamma": jnp.asarray(
            not np.any(b0) and (fx is None or float(fx.mu) == 0.0)
        ),
    }


#: Above this standardized bound ``Phi(-a)`` nears underflow, and the draw falls
#: back on rejection.
_TN_INVERT_MAX = 30.0


def _tn_invert(a, u):
    """``z ~ N(0, 1) | z > a`` by inversion of the uniform ``u``.

    Exact to rounding: the tail ``Phi(-a)`` is computed directly, never as
    ``1 - Phi(a)``, so the inversion keeps its precision for large ``a``.
    """
    import jax.numpy as jnp
    from jax.scipy.special import ndtr, ndtri

    tail = ndtr(-a)
    return jnp.where(a > 0, -ndtri(u * tail), ndtri(ndtr(a) + u * tail))


def _tn_beyond(key, a, active):
    """``z ~ N(0, 1) | z > a`` where ``active``, by exact rejection (Robert 1995)."""
    import jax
    import jax.numpy as jnp

    naive = a < 0.45
    alpha = 0.5 * (a + jnp.sqrt(a * a + 4.0))

    def cond(st):
        return jnp.any(~st[2])

    def body(st):
        key, z, done = st
        key, k1, k2, k3 = jax.random.split(key, 4)
        zn = jax.random.normal(k1, a.shape)
        ze = a + jax.random.exponential(k2, a.shape) / alpha
        u = jax.random.uniform(k3, a.shape)
        prop = jnp.where(naive, zn, ze)
        ok = jnp.where(naive, zn > a, u <= jnp.exp(-0.5 * (ze - alpha) ** 2))
        z = jnp.where(ok & ~done, prop, z)
        return key, z, done | ok

    _, z, _ = jax.lax.while_loop(cond, body, (key, jnp.zeros_like(a), ~active))
    return z


def _make_sweep(
    n, k, process, n_colours, logdet_eval, store_latent, mixed=False, n_groups=0
):
    """One chain's sweep; ``mixed`` draws the unknown sides (``sign`` 0) too.

    A mixed sweep reads ``logq`` from the consts and keeps each utility's
    probability of the negative side in the state's ``"pneg"``.  ``n_groups``
    random unit effects (a SAR's) are the state's ``"c"`` and ``"sigma_c"``.
    """
    import augax
    import jax
    import jax.numpy as jnp
    from jax.scipy.special import log_ndtr

    from .._utils._jax_slice import adapt_slice_width, jax_slice_sample_1d

    sar = process == "sar"
    effects = n_groups > 0

    def sweep(st, key, tuning, d):
        rows, cols, vals = d["rows"], d["cols"], d["vals"]

        # Scatter-adds: on CPU they beat sorted segment sums here.
        def W(v):
            return jnp.zeros(n).at[rows].add(vals * v[cols])

        par, beta, ystar, m = st["par"], st["beta"], st["ystar"], st["m"]
        X, WX = d["X"], d["WX"]
        offset = st["c"][d["group"]] if effects else 0.0
        Xb = X @ beta + offset
        p = 1.0 / (m * m)
        if sar:
            eps = ystar - par * W(ystar) - Xb
        else:
            r0 = ystar - Xb
            eps = r0 - par * W(r0)
        aval = d["colones"] - par * d["colw"]

        key, k_y, k_tail, k_m, k_par, k_beta = jax.random.split(key, 6)
        # Each utility is drawn once per sweep, so one uniform each suffices.
        u_all = jax.random.uniform(k_y, (n,))
        if mixed:
            u_side = jax.random.uniform(jax.random.fold_in(k_y, 1), (n,))

        def colour_step(c, carry):
            # A class's members read and move eps only on their columns of
            # A, which no other member's touch.
            ys, eps, pneg = carry
            idx = d["classes"][c]
            active = idx < n

            def take(a):
                # The padding index ``n`` is out of range: clip, then mask.
                return jnp.take(a, idx, axis=0, mode="clip")

            R, Av = take(d["colrows"]), take(aval)
            pAv = p[R] * Av
            qdiag = jnp.sum(pAv * Av, axis=1)
            x0 = take(ys)
            mean = x0 - jnp.sum(pAv * eps[R], axis=1) / qdiag
            sd = 1.0 / jnp.sqrt(qdiag)
            side = take(d["sign"])
            if mixed:
                log_odds = log_ndtr(mean / sd) + take(d["logq"]) - log_ndtr(-mean / sd)
                p_neg = jax.nn.sigmoid(-log_odds)
                unknown = side == 0
                pneg = pneg.at[idx].set(
                    jnp.where(unknown, p_neg, take(pneg)), mode="drop"
                )
                side = jnp.where(
                    unknown, jnp.where(take(u_side) < p_neg, -1.0, 1.0), side
                )
            # Standardized bound on the side of 0 that y fixes.
            a = -side * mean / sd
            far = active & (a > _TN_INVERT_MAX)
            z = jnp.where(
                far, 0.0, _tn_invert(jnp.minimum(a, _TN_INVERT_MAX), take(u_all))
            )
            z = jax.lax.cond(
                jnp.any(far),
                lambda zz: jnp.where(
                    far, _tn_beyond(jax.random.fold_in(k_tail, c), a, far), zz
                ),
                lambda zz: zz,
                z,
            )
            x1 = mean + side * sd * z
            delta = jnp.where(active, x1 - x0, 0.0)
            eps = eps.at[R].add(Av * delta[:, None])
            return ys.at[idx].set(x1, mode="drop"), eps, pneg

        ystar, eps, pneg = jax.lax.fori_loop(
            0, n_colours, colour_step, (ystar, eps, st.get("pneg", jnp.zeros(n)))
        )
        m = augax.ks_sample(eps, k_m)
        p = 1.0 / (m * m)
        z = W(ystar)
        # The SAR's response, less its unit effects.
        yt = ystar - offset
        py, pz = p * yt, p * z
        params = d["logdet"]

        if sar:
            Binv = X.T @ (p[:, None] * X) + d["B0inv"]
            L = jnp.linalg.cholesky(Binv)

            def chol_solve(v):
                return jax.scipy.linalg.cho_solve((L, True), v)

            c0 = X.T @ py + d["B0inv_b0"]
            c1 = X.T @ pz
            Bc0, Bc1 = chol_solve(c0), chol_solve(c1)
            q00, q01, q11 = c0 @ Bc0, c0 @ Bc1, c1 @ Bc1
            yPy, yPz, zPz = yt @ py, yt @ pz, z @ pz

            def dens(r):
                return (
                    logdet_eval(r, params)
                    - 0.5 * (yPy - 2.0 * r * yPz + r * r * zPz)
                    + 0.5 * (q00 - 2.0 * r * q01 + r * r * q11)
                )
        else:
            G0 = X.T @ (p[:, None] * X)
            G1 = X.T @ (p[:, None] * WX)
            G1 = G1 + G1.T
            G2 = WX.T @ (p[:, None] * WX)
            h0, h1, h2 = X.T @ py, X.T @ pz + WX.T @ py, WX.T @ pz
            s0, s1, s2 = ystar @ py, 2.0 * (ystar @ pz), z @ pz

            def blocks(lam):
                Binv = G0 - lam * G1 + lam * lam * G2 + d["B0inv"]
                return jnp.linalg.cholesky(Binv), h0 - lam * h1 + lam * lam * h2 + d[
                    "B0inv_b0"
                ]

            def dens(lam):
                Lc, mm = blocks(lam)
                w = jax.scipy.linalg.solve_triangular(Lc, mm, lower=True)
                return (
                    logdet_eval(lam, params)
                    - 0.5 * (s0 - lam * s1 + lam * lam * s2)
                    + 0.5 * (w @ w)
                    - jnp.sum(jnp.log(jnp.diag(Lc)))
                )

        width = st["slice_width"]
        par, _, left, right = jax_slice_sample_1d(
            dens, par, d["lo"], d["hi"], key=k_par, w=width, return_steps=True
        )
        width = adapt_slice_width(width, left, right, tuning)
        if sar:
            mean = Bc0 - par * Bc1
        else:
            L, mm = blocks(par)
            mean = jax.scipy.linalg.cho_solve((L, True), mm)
        beta = mean + jax.scipy.linalg.solve_triangular(
            L.T, jax.random.normal(k_beta, (k,)), lower=False
        )
        new = {"par": par, "slice_width": width, "m": m}
        if mixed:
            new["pneg"] = pneg
        offset = 0.0
        if effects:
            # The unit effects given the scales, then their sd.
            grp, mu, sig = d["group"], d["c_mu"], st["sigma_c"]
            e = ystar - par * z - X @ beta
            prec = jax.ops.segment_sum(p, grp, num_segments=n_groups) + 1.0 / sig**2
            mean = jax.ops.segment_sum(p * e, grp, num_segments=n_groups) + mu / sig**2
            c = mean / prec + jax.random.normal(
                jax.random.fold_in(k_beta, 1), (n_groups,)
            ) / jnp.sqrt(prec)
            ss, scale = jnp.sum((c - mu) ** 2), d["c_scale"]

            def sd_density(ls):
                s2 = jnp.exp(2.0 * ls)
                return (
                    -n_groups * ls
                    - 0.5 * ss / s2
                    - 2.0 * jnp.log1p(s2 / (3.0 * scale**2))
                    + ls
                )

            ls, _ = jax_slice_sample_1d(
                sd_density,
                jnp.log(sig),
                -10.0,
                10.0,
                key=jax.random.fold_in(k_beta, 2),
                w=0.5,
            )
            new["sigma_c"] = jnp.exp(ls)
            offset = c[grp]

        # The scale move (see :mod:`._core`): (y*, beta, c) -> g (y*, beta, c).
        if sar:
            e = ystar - par * z - X @ beta - offset
        else:
            e = (ystar - X @ beta) - par * (z - WX @ beta)
        epe = e @ (p * e)
        dim = n + k + n_groups
        b0, B0inv, k_g = d["b0"], d["B0inv"], jax.random.fold_in(k_beta, 3)

        def gamma_draw():
            Q = epe + beta @ B0inv @ beta
            if effects:
                Q = Q + c @ c / new["sigma_c"] ** 2
            return jnp.sqrt(2.0 * jax.random.gamma(k_g, 0.5 * dim) / Q)

        def slice_draw():
            def logp(lg):
                g = jnp.exp(lg)
                r = g * beta - b0
                v = dim * lg - 0.5 * g * g * epe - 0.5 * r @ B0inv @ r
                if effects:
                    v = (
                        v
                        - 0.5 * jnp.sum((g * c - d["c_mu"]) ** 2) / new["sigma_c"] ** 2
                    )
                return v

            lg, _ = jax_slice_sample_1d(
                logp, 0.0, -10.0, 10.0, key=k_g, w=2.0 / jnp.sqrt(dim)
            )
            return jnp.exp(lg)

        g = jax.lax.cond(d["scale_gamma"], gamma_draw, slice_draw)
        ystar, beta = ystar * g, beta * g
        new["ystar"], new["beta"] = ystar, beta
        draws = (par, beta)
        if effects:
            new["c"] = c * g
            draws = draws + (new["c"], new["sigma_c"])
        if store_latent:
            draws = draws + (ystar,)
        return new, draws

    return sweep


def run_chains_jax_latent_logit(
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
    store_latent=False,
    progressbar=False,
    probes=None,
):
    """Run the latent-logit chains in parallel threads; one dict per chain.

    ``store_latent`` adds the utilities at every kept draw, ``"latent"``.
    ``probes``, a :class:`~neighbayes._logdet._probe_check.WarmupProbes` whose
    initial parameters are ``logdet_params``, sizes the probe pool halfway
    through warmup.
    """
    import jax
    import jax.numpy as jnp

    from ..._jax_dispatch import ensure_x64
    from ..._logdet._jax import logdet_param_evaluator

    ensure_x64()
    n, k = X.shape
    if logdet_kind == "closure":
        fn = logdet_params

        def logdet_eval(r, _params):
            return fn(r)

        ld_params, ld_key = jnp.float64(0.0), ("closure", id(fn))
    else:
        logdet_eval = logdet_param_evaluator(logdet_kind)
        ld_params, ld_key = logdet_params, (logdet_kind,)
    data = _latent_data(y, X, W, priors, ld_params)
    n_colours = int(data["classes"].shape[0])

    static = (n, k, process, n_colours, store_latent, *ld_key)
    sweep = cached_sweep(
        ("logit_latent", *static),
        lambda: _make_sweep(n, k, process, n_colours, logdet_eval, store_latent),
    )
    sign = np.asarray(data["sign"])
    m0 = np.sqrt(2.0 * np.pi) * np.log(2.0)
    states = [
        {
            "par": jnp.float64(0.0),
            "beta": jnp.asarray(data["b0"]),
            "ystar": jnp.asarray(0.5 * sign),
            "m": jnp.full(n, m0),
            "slice_width": jnp.float64(slice_width),
        }
        for _ in seeds
    ]
    warm = [jax.random.PRNGKey(int(s)) for s in seeds]
    draw = [jax.random.fold_in(jax.random.PRNGKey(int(s)), 1) for s in seeds]
    _, kept = run_chains_chunked(
        sweep,
        states,
        warm,
        draw,
        tune=tune,
        draws=draws,
        progressbar=progressbar,
        model_type="latent_logit",
        on_warmup_midpoint=None
        if probes is None
        else probe_midpoint([(probes, lambda t: t[0], lambda c: c["logdet"])]),
        consts=data,
    )
    name = "rho" if process == "sar" else "lam"
    sl = slice(None, None, thin) if thin > 1 else slice(None)
    keys = (name, "beta", "latent")
    return [
        {key: arr[c, sl] for key, arr in zip(keys, kept)} for c in range(len(seeds))
    ]
