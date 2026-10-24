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

and the classes are swept in turn.  The Kolmogorov scales come from
:func:`augax.ks_sample`, on the device.
"""

from __future__ import annotations

import numba
import numpy as np
import scipy.sparse as sp

from .._utils._jax_utils import cached_sweep, run_chains_chunked


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


def _latent_data(y, X, W, colour, priors, ld_params):
    """The read-only arrays every chain's sweep takes (``consts``)."""
    import jax.numpy as jnp

    W = sp.csr_matrix(W)
    coo = W.tocoo()
    k = X.shape[1]
    beta_sigma = np.broadcast_to(np.asarray(priors.beta_sigma, float), (k,))
    B0inv = np.diag(1.0 / beta_sigma**2)
    b0 = np.broadcast_to(np.asarray(priors.beta_mu, float), (k,))
    diag = W.diagonal()
    return {
        "sign": jnp.asarray(np.where(np.asarray(y) > 0, 1.0, -1.0)),
        "X": jnp.asarray(X, dtype=jnp.float64),
        "WX": jnp.asarray(W @ X, dtype=jnp.float64),
        "rows": jnp.asarray(coo.row, dtype=jnp.int32),
        "cols": jnp.asarray(coo.col, dtype=jnp.int32),
        "vals": jnp.asarray(coo.data, dtype=jnp.float64),
        "Wdiag": jnp.asarray(diag),
        "colour": jnp.asarray(colour, dtype=jnp.int32),
        "b0": jnp.asarray(b0),
        "B0inv": jnp.asarray(B0inv),
        "B0inv_b0": jnp.asarray(B0inv @ b0),
        "lo": jnp.float64(priors.rho_lower),
        "hi": jnp.float64(priors.rho_upper),
        "logdet": ld_params,
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


def _make_sweep(n, k, process, n_colours, logdet_eval, store_latent):
    import augax
    import jax
    import jax.numpy as jnp

    from .._utils._jax_slice import adapt_slice_width, jax_slice_sample_1d

    sar = process == "sar"

    def sweep(st, key, tuning, d):
        rows, cols, vals = d["rows"], d["cols"], d["vals"]

        # Scatter-adds: on CPU they beat sorted segment sums here.
        def W(v):
            return jnp.zeros(n).at[rows].add(vals * v[cols])

        def Wt(v):
            return jnp.zeros(n).at[cols].add(vals * v[rows])

        par, beta, ystar, m = st["par"], st["beta"], st["ystar"], st["m"]
        X, WX = d["X"], d["WX"]
        Xb = X @ beta
        p = 1.0 / (m * m)
        qdiag = (
            p
            - 2.0 * par * d["Wdiag"] * p
            + par * par * jnp.zeros(n).at[cols].add(vals * vals * p[rows])
        )

        def eps_of(ys):
            return ys - par * W(ys) - Xb if sar else (ys - Xb) - par * W(ys - Xb)

        key, k_y, k_tail, k_m, k_par, k_beta = jax.random.split(key, 6)
        # Each utility is drawn once per sweep, so one uniform each suffices.
        u_all = jax.random.uniform(k_y, (n,))

        def colour_step(c, ys):
            pe = p * eps_of(ys)
            g = pe - par * Wt(pe)
            mean = ys - g / qdiag
            sd = 1.0 / jnp.sqrt(qdiag)
            active = d["colour"] == c
            # Standardized bound on the side of 0 that y fixes.
            a = -d["sign"] * mean / sd
            far = active & (a > _TN_INVERT_MAX)
            z = jnp.where(far, 0.0, _tn_invert(jnp.minimum(a, _TN_INVERT_MAX), u_all))
            z = jax.lax.cond(
                jnp.any(far),
                lambda zz: jnp.where(
                    far, _tn_beyond(jax.random.fold_in(k_tail, c), a, far), zz
                ),
                lambda zz: zz,
                z,
            )
            return jnp.where(active, mean + d["sign"] * sd * z, ys)

        ystar = jax.lax.fori_loop(0, n_colours, colour_step, ystar)
        m = augax.ks_sample(eps_of(ystar), k_m)
        p = 1.0 / (m * m)
        z = W(ystar)
        py, pz = p * ystar, p * z
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
            yPy, yPz, zPz = ystar @ py, ystar @ pz, z @ pz

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
        new = {"par": par, "beta": beta, "ystar": ystar, "m": m, "slice_width": width}
        return new, (par, beta, ystar) if store_latent else (par, beta)

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
):
    """Run the latent-logit chains in parallel threads; one dict per chain.

    ``store_latent`` adds the utilities at every kept draw, ``"latent"``.
    """
    import jax
    import jax.numpy as jnp

    from ..._jax_dispatch import ensure_x64
    from ..._logdet._jax import logdet_param_evaluator

    ensure_x64()
    n, k = X.shape
    colour = colour_classes(W)
    n_colours = int(colour.max()) + 1
    if logdet_kind == "closure":
        fn = logdet_params

        def logdet_eval(r, _params):
            return fn(r)

        ld_params, ld_key = jnp.float64(0.0), ("closure", id(fn))
    else:
        logdet_eval = logdet_param_evaluator(logdet_kind)
        ld_params, ld_key = logdet_params, (logdet_kind,)
    data = _latent_data(y, X, W, colour, priors, ld_params)

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
        sweep, states, warm, draw, tune=tune, draws=draws, consts=data
    )
    name = "rho" if process == "sar" else "lam"
    sl = slice(None, None, thin) if thin > 1 else slice(None)
    keys = (name, "beta", "latent")
    return [
        {key: arr[c, sl] for key, arr in zip(keys, kept)} for c in range(len(seeds))
    ]
