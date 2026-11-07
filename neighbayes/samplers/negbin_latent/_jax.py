r"""JAX backend of the latent SAR and SEM negative binomial sampler (:mod:`._core`).

The same sweep, with the log-means updated by colour class rather than one at a
time.  The conditional of :math:`\eta_i` touches the disturbances in the rows
of column ``i`` of :math:`A = I - \rho W`; log-means whose columns share no row
are conditionally independent, so the latent logit's colouring
(:func:`..binary_latent._jax.colour_classes`) splits them into classes, each
drawn at once and the classes in turn.  A slice per lane would run as long as
the slowest lane of its class, so a class is drawn instead by independence
Metropolis–Hastings from a two-component *t* mixture at each conditional's
mode (:func:`_eta_mh`), in a fixed number of passes.  The filter solves are
BiCGSTAB from the nearby solution at hand, falling back on the sparse LU should
one stall; products with :math:`W` are gathers from its padded rows, which on
CPU beat scatter-adds.

Step 8's solves on the zero counts' sites run on the whole stack, the filter
masked to those sites and the identity elsewhere, and its
:math:`\log|A_{ZZ}|` is a Chebyshev series of fixed order
(:func:`_zero_logdet_series`), so that the compiled sweep does not depend on
which counts are zero.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from .._utils._jax_utils import cached_sweep, probe_midpoint, run_chains_chunked
from .._utils._latent_loglik import POISSON, TRUNCATED, UNOBSERVED
from ..binary_latent._jax import class_table, colour_classes, column_table
from ._core import (
    _BETA_DF,
    _JOINT_FD_STEP,
    _JOINT_MAX_SHRINKS,
    _JOINT_MAX_STEPS,
    _JOINT_WIDTH,
    _PAR_ACCEPT,
    _SOLVE_MAXITER,
    _SOLVE_RTOL,
    initial_state,
    joint_axes,
    zero_set_logdet,
)

#: Order of the Chebyshev series of step 8's ``log|I - rho W_ZZ|``.
_ZERO_LOGDET_ORDER = 128


def _zero_logdet_series(fn, lo, hi):
    """``fn``, step 8's ``log|I - rho W_ZZ|``, as Chebyshev coefficients on ``[lo, hi]``."""
    import jax.numpy as jnp

    from ..._logdet._chebyshev import chebyshev_coeffs_dct1, chebyshev_gauss_nodes

    nodes, _ = chebyshev_gauss_nodes(_ZERO_LOGDET_ORDER, lo, hi)
    coeffs = chebyshev_coeffs_dct1(np.array([fn(r) for r in nodes]))
    return jnp.asarray(coeffs), jnp.float64(lo), jnp.float64(hi)


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


def _latent_data(
    y,
    X,
    W,
    priors,
    ld_params,
    sparse_arrays,
    T=1,
    unit_effects=None,
    kind=None,
    level_shift=0.0,
    zero_logdet=None,
):
    """The read-only arrays every chain's sweep takes (``consts``).

    ``kind`` is what each count observes, :data:`POISSON` by default;
    ``level_shift`` and ``zero_logdet`` as :class:`._core.LatentNegBinChain`.
    """
    import jax.numpy as jnp

    from ..negbin_reduced._core import _symmetrizing_diagonal
    from ._core import _lag

    W = sp.csr_matrix(W, dtype=np.float64)
    nT, k = X.shape
    n = nT // T
    WX = _lag(W, X.reshape(T, n, k)).reshape(nT, k)
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
    beta_sigma = np.broadcast_to(np.asarray(priors.beta_sigma, float), (k,))
    R, O, V = column_table(W)
    Wc, Wv = _row_table(W)
    # A W similar to a symmetric matrix by D^{1/2} gets that matrix's values
    # on its own pattern, for the filter solves (see ``_make_sweep``).
    dsym = _symmetrizing_diagonal(sp.csc_matrix(W))
    if dsym is not None:
        sq = np.sqrt(dsym)
        sym = {"sqd": jnp.asarray(sq), "Wsym": jnp.asarray(sq[:, None] * Wv / sq[Wc])}
    else:
        sym = {}
    zero = np.asarray(y).reshape(T, n) == 0
    lo, hi = float(priors.rho_lower), float(priors.rho_upper)
    if zero_logdet is None:
        zero_logdet = zero_set_logdet(W, zero, None, lo, hi)
    _, level = joint_axes(X, priors.alpha_fixed is not None)
    return {
        **effects,
        **sym,
        "zmask": jnp.asarray(zero, dtype=jnp.float64),
        "level": jnp.asarray(level),
        "zero_logdet": _zero_logdet_series(zero_logdet, lo, hi),
        "y": jnp.asarray(y, dtype=jnp.float64).reshape(T, n),
        "kind": jnp.asarray(
            np.full(nT, POISSON) if kind is None else kind, dtype=jnp.int32
        ).reshape(T, n),
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
        "classes": jnp.asarray(class_table(colour_classes(W))),
        "b0": jnp.asarray(np.broadcast_to(np.asarray(priors.beta_mu, float), (k,))),
        "B0inv": jnp.asarray(np.diag(1.0 / beta_sigma**2)),
        "lo": jnp.float64(priors.rho_lower),
        "hi": jnp.float64(priors.rho_upper),
        "alpha_nu": jnp.float64(priors.alpha_nu),
        "alpha_sigma": jnp.float64(priors.alpha_sigma),
        "logdet": ld_params,
        "level_shift": jnp.float64(level_shift),
        "sparse": sparse_arrays,
    }


#: Newton steps to the mode of a log-mean's conditional, and the largest step.
_ETA_NEWTON = 3
_ETA_NEWTON_MAX_STEP = 5.0
#: Floor of the slope whose inverse scales the wide proposal component.
_ETA_MIN_SLOPE = 1e-3


def _obs_terms(t, y, kind):
    """``log p(y | eta = t) + log y!``, its score and negative Hessian, by kind.

    The truncated count's score is ``y - g``, ``g = λ / (1 - e^{-λ})`` its
    mean, and its negative Hessian ``λ g'(λ)``, by series at small ``λ``.
    """
    import jax.numpy as jnp

    et = jnp.exp(t)
    small = et < 1e-2
    q = -jnp.expm1(-et)
    qs = jnp.where(small, 1.0, q)
    g = jnp.where(small, 1.0 + et / 2.0 + et * et / 12.0, et / qs)
    dg = jnp.where(
        small, et * (0.5 + et / 6.0), et * (qs - et * jnp.exp(-et)) / (qs * qs)
    )
    far = t < -30.0
    log_pos = jnp.where(far, t - 0.5 * et, jnp.log(jnp.where(far, 1.0, q)))
    trunc, on = kind == TRUNCATED, kind != UNOBSERVED
    value = y * t - et - jnp.where(trunc, log_pos, 0.0)
    score = y - jnp.where(trunc, g, et)
    info = jnp.where(trunc, dg, et)
    return (
        jnp.where(on, value, 0.0),
        jnp.where(on, score, 0.0),
        jnp.where(on, info, 0.0),
    )


def _eta_mh(x0, yi, ki, e, Av, alpha, key, active):
    """Independence MH for the log-means of one colour class.

    ``e`` are the disturbances in the rows of each lane's column of ``A`` and
    ``Av`` that column's values.  The proposal is an equal mixture of two *t*'s
    with 2 degrees of freedom at the conditional mode, drawn by inversion: one
    scaled by the curvature there, one by ``1 / s``, ``s = y + alpha sum(Av)``
    the conditional's slope where every disturbance sits in its left tail.  At
    small ``alpha`` a zero count's conditional is nearly flat for tens of units
    below its mode, beyond the curvature's reach, and the second component
    covers that range.  Newton starts from the precision-weighted mean of the
    terms' own modes (the Poisson's at ``log(y + 1/2)``, each disturbance's at
    zero), which, like the steps, involves the other log-means only, so the
    proposal does not depend on the current value.  Lanes outside ``active``
    keep it.
    """
    import jax
    import jax.numpy as jnp

    rest = e - Av * x0[:, None]

    def f(t):
        ee = rest + Av * t[:, None]
        return _obs_terms(t, yi, ki)[0] + alpha * jnp.sum(ee - jnp.exp(ee), axis=1)

    def grad_hess(t):
        ee = jnp.exp(rest + Av * t[:, None])
        _, score, info = _obs_terms(t, yi, ki)
        return (
            score + alpha * jnp.sum(Av * (1.0 - ee), axis=1),
            info + alpha * jnp.sum(Av * Av * ee, axis=1),
        )

    # An unobserved count's term carries no weight.
    yp = yi + 0.5
    wp = jnp.where(ki != UNOBSERVED, yp, 0.0)
    t = (wp * jnp.log(yp) - alpha * jnp.sum(Av * rest, axis=1)) / (
        wp + alpha * jnp.sum(Av * Av, axis=1)
    )
    for _ in range(_ETA_NEWTON):
        g, h = grad_hess(t)
        t = t + jnp.clip(g / h, -_ETA_NEWTON_MAX_STEP, _ETA_NEWTON_MAX_STEP)
    sd = 1.0 / jnp.sqrt(grad_hess(t)[1])
    slope = jnp.where(ki != UNOBSERVED, yi, 0.0) + alpha * jnp.sum(Av, axis=1)
    wide = jnp.maximum(sd, 1.0 / jnp.maximum(slope, _ETA_MIN_SLOPE))

    def logq(x):
        def log_t(s):
            z = (x - t) / s
            return -1.5 * jnp.log1p(0.5 * z * z) - jnp.log(s)

        return jnp.logaddexp(log_t(sd), log_t(wide))

    k_pick, k_prop, k_acc = jax.random.split(key, 3)
    scale = jnp.where(jax.random.uniform(k_pick, x0.shape) < 0.5, sd, wide)
    u = jax.random.uniform(k_prop, x0.shape, minval=1e-12, maxval=1.0 - 1e-12)
    prop = t + scale * (2.0 * u - 1.0) / jnp.sqrt(2.0 * u * (1.0 - u))
    log_ratio = f(prop) - f(x0) + logq(x0) - logq(prop)
    accept = active & (jnp.log(jax.random.uniform(k_acc, x0.shape)) < log_ratio)
    return jnp.where(accept, prop, x0)


def _make_sweep(
    n,
    k,
    process,
    n_colours,
    alpha_fixed,
    row_std,
    logdet_eval,
    lu_solve,
    store_latent,
    T=1,
    n_groups=0,
    axes=(0, 1, 2),
):
    import jax
    import jax.numpy as jnp
    from jax.scipy.sparse.linalg import bicgstab, cg
    from jax.scipy.special import digamma, gammaln, polygamma

    from ..._logdet._jax import jax_logdet_chebyshev_traced
    from .._utils._jax_slice import (
        adapt_slice_width,
        jax_slice_along,
        jax_slice_sample_1d,
    )

    sar = process == "sar"
    n_obs = n * T
    effects = n_groups > 0

    # Step 8's Hessian stencil on its axes: entry (i, j) from four points.
    dim = len(axes)
    free_alpha = 1 in axes
    h = _JOINT_FD_STEP[list(axes)]
    eye = np.eye(dim)
    stencil = [
        (i, j, a, b)
        for i in range(dim)
        for j in range(i, dim)
        for a, b in ((1, 1), (1, -1), (-1, 1), (-1, -1))
    ]
    fd_points = np.array(
        [a * h[i] * eye[i] + b * h[j] * eye[j] for i, j, a, b in stencil]
    )
    fd_weight = np.array([-a * b / (4 * h[i] * h[j]) for i, j, a, b in stencil])
    fd_i = np.array([s[0] for s in stencil])
    fd_j = np.array([s[1] for s in stencil])

    def gamma_moments(a):
        return digamma(a) - jnp.log(a), jnp.sqrt(polygamma(1, a))

    def sweep(st, key, tuning, d):
        X, WX, y, b0, B0inv = d["X"], d["WX"], d["y"], d["b0"], d["B0inv"]
        nu, sig = d["alpha_nu"], d["alpha_sigma"]
        sparse = d["sparse"]

        def W(v):
            """``W`` applied to each period's row of ``v`` (``(T, n)``)."""
            return jnp.sum(d["Wvals"] * v[..., d["Wcols"]], axis=-1)

        def solve(r, b, x0, mask=None):
            """``A(r)^{-1}`` applied to each period of ``b`` (``(T, n)``).

            A ``W`` with a symmetric ``D^{1/2} W D^{-1/2}`` is solved by CG on
            ``D^{1/2} A D^{-1/2}``, positive definite inside the stationary
            interval: a quarter cheaper than BiCGSTAB per solve, which tells
            as ``|rho|`` and the iteration count grow.  Any other ``W`` by
            BiCGSTAB.  A solve that misses its tolerance falls back
            on the sparse LU.  ``mask``, 0/1 like ``b``, keeps ``W``'s rows
            and columns at its ones, the identity elsewhere (step 8).
            """

            def lag(v, vals):
                if mask is None:
                    return jnp.sum(vals * v[..., d["Wcols"]], axis=-1)
                return mask * jnp.sum(vals * (mask * v)[..., d["Wcols"]], axis=-1)

            if "sqd" in d:
                sq = d["sqd"]

                def A(v):
                    return v - r * lag(v, d["Wsym"])

                rhs = sq * b
                v, _ = cg(A, rhs, x0=sq * x0, tol=_SOLVE_RTOL, maxiter=_SOLVE_MAXITER)
                x = v / sq
            else:

                def A(v):
                    return v - r * lag(v, d["Wvals"])

                rhs = b
                v, _ = bicgstab(A, b, x0=x0, tol=_SOLVE_RTOL, maxiter=_SOLVE_MAXITER)
                x = v
            ok = jnp.linalg.norm(rhs - A(v)) <= 100.0 * _SOLVE_RTOL * jnp.linalg.norm(
                rhs
            )
            Ai, Aj = sparse["Ai"], sparse["Aj"]
            if mask is None:
                vals = sparse["eye"] - r * sparse["w"]
                return jax.lax.cond(
                    ok,
                    lambda: x,
                    lambda: jax.vmap(lambda bb: lu_solve(Ai, Aj, vals, bb))(b),
                )
            vals = sparse["eye"] - r * sparse["w"] * mask[:, Ai] * mask[:, Aj]
            return jax.lax.cond(
                ok,
                lambda: x,
                lambda: jax.vmap(lambda vv, bb: lu_solve(Ai, Aj, vv, bb))(vals, b),
            )

        def poisson(eta):
            """The counts' log-likelihood at ``eta``, each by its kind."""
            return jnp.sum(_obs_terms(eta, y, d["kind"])[0])

        def gamma_stat(e):
            return jnp.sum(e - jnp.exp(e))

        def alpha_density(s, la):
            a = jnp.exp(la)
            return (
                n_obs * (a * la - gammaln(a))
                + a * s
                - 0.5 * (nu + 1.0) * jnp.log1p(a * a / (nu * sig * sig))
                + la
            )

        par, beta, alpha, eta = st["par"], st["beta"], st["alpha"], st["eta"]
        keys = jax.random.split(key, 10)

        def stack(v):
            return v.reshape(T, n)

        def unit():
            """The unit effects on the stack, ``(T, n)``."""
            return stack(st_c[d["group"]]) if effects else 0.0

        st_c = st["c"] if effects else None

        # 1. The log-means, by colour class, every period at once.
        Xb = stack(X @ beta)
        if sar:
            eps = eta - par * W(eta) - Xb - unit()
        else:
            r0 = eta - Xb
            eps = r0 - par * W(r0)
        aval = d["colones"] - par * d["colw"]

        def colour_step(c, carry):
            eta, eps = carry
            idx = d["classes"][c]
            active = jnp.broadcast_to(idx < n, (T, idx.size))

            def take(a):
                # The padding index ``n`` is out of range: clip, then mask.
                return jnp.take(a, idx, axis=-1 if a.ndim == 2 else 0, mode="clip")

            x0, yi, ki = take(eta), take(y), take(d["kind"])
            R, Av = (
                jnp.take(d["colrows"], idx, axis=0, mode="clip"),
                jnp.take(aval, idx, axis=0, mode="clip"),
            )
            e = eps[:, R]
            Avt = jnp.broadcast_to(Av, e.shape)
            x1 = _eta_mh(
                x0.reshape(-1),
                yi.reshape(-1),
                ki.reshape(-1),
                e.reshape(-1, R.shape[1]),
                Avt.reshape(-1, R.shape[1]),
                alpha,
                jax.random.fold_in(keys[0], c),
                active.reshape(-1),
            ).reshape(T, -1)
            delta = jnp.where(active, x1 - x0, 0.0)
            eps = eps.at[:, R].add(Av * delta[:, :, None])
            return eta.at[:, idx].set(x1, mode="drop"), eps

        eta, eps = jax.lax.fori_loop(0, n_colours, colour_step, (eta, eps))

        # 2. The spatial parameter, given eta.
        Weta = W(eta)
        if sar:
            g = d["XtXiXt"] @ Weta.ravel()
            u = beta + par * g
            e0 = (eta - unit()).ravel() - X @ u
            dv = Weta.ravel() - X @ g

            def dens(r):
                b = u - r * g - b0
                e = e0 - r * dv
                return (
                    T * logdet_eval(r, d["logdet"])
                    - d["level_shift"] * jnp.log1p(-r)
                    + alpha * gamma_stat(e)
                    - 0.5 * b @ B0inv @ b
                )

            def beta_at(r):
                return u - r * g
        else:
            m = gamma_moments(alpha)[0]
            ev, We = eta.ravel(), Weta.ravel()
            h0 = X.T @ (ev - m)
            h1 = X.T @ We + WX.T @ (ev - m)
            h2 = WX.T @ We

            def bhat(lam):
                return jnp.linalg.solve(
                    d["G0"] - lam * d["G1"] + lam * lam * d["G2"],
                    h0 - lam * h1 + lam * lam * h2,
                )

            u = beta - bhat(par)

            def dens(lam):
                b = u + bhat(lam)
                e = ev - lam * We - (X - lam * WX) @ b
                return (
                    T * logdet_eval(lam, d["logdet"])
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
        c = (eta - par * Weta - unit()).ravel()
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

        sigma_c = st["sigma_c"] if effects else None
        if effects:
            # 7. The unit effects: independence MH from a t at each mode.
            grp, mu_c = d["group"], d["c_mu"]
            base = eps + st_c[grp]
            cnt = jax.ops.segment_sum(jnp.ones_like(base), grp, num_segments=n_groups)
            Ssum = jax.ops.segment_sum(jnp.exp(base), grp, num_segments=n_groups)
            Bsum = jax.ops.segment_sum(base, grp, num_segments=n_groups)
            prec = 1.0 / sigma_c**2

            def c_logp(x):
                return (
                    alpha * (Bsum - cnt * x - jnp.exp(-x) * Ssum)
                    - 0.5 * prec * (x - mu_c) ** 2
                )

            def c_newton(_, x):
                grad = alpha * (jnp.exp(-x) * Ssum - cnt) - prec * (x - mu_c)
                hess = alpha * jnp.exp(-x) * Ssum + prec
                return x + jnp.clip(grad / hess, -2.0, 2.0)

            c_mode = jax.lax.fori_loop(0, 30, c_newton, st_c)
            c_sd = 1.0 / jnp.sqrt(alpha * jnp.exp(-c_mode) * Ssum + prec)

            def c_logq(x):
                z = (x - c_mode) / c_sd
                return -0.5 * (_BETA_DF + 1.0) * jnp.log1p(z * z / _BETA_DF)

            k_t, k_a = jax.random.split(keys[7])
            c_prop = c_mode + c_sd * jax.random.t(k_t, _BETA_DF, (n_groups,))
            c_ratio = c_logp(c_prop) - c_logp(st_c) + c_logq(st_c) - c_logq(c_prop)
            st_c = jnp.where(
                jnp.log(jax.random.uniform(k_a, (n_groups,))) < c_ratio, c_prop, st_c
            )
            ss = jnp.sum((st_c - mu_c) ** 2)
            scale = d["c_scale"]

            def sd_density(ls):
                s2 = jnp.exp(2.0 * ls)
                return (
                    -n_groups * ls
                    - 0.5 * ss / s2
                    - 2.0 * jnp.log1p(s2 / (3.0 * scale**2))
                    + ls
                )

            ls, _ = jax_slice_sample_1d(
                sd_density, jnp.log(sigma_c), -10.0, 10.0, key=keys[8], w=0.5
            )
            sigma_c = jnp.exp(ls)
            eps = base - st_c[grp]

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
            ones = jnp.ones((T, n))
            s1 = ones / (1.0 - par) if row_std else solve(par, ones, s1)
            Xb = stack(X @ beta)
            sxb = solve(par, Xb + unit(), sxb) if sar else Xb
            m, s = gamma_moments(alpha)
            z = stack((eps - m) / s)
            dev = eta - sxb

            def move(la):
                mm, ss = gamma_moments(jnp.exp(la))
                bb = ss / s
                return mm + ss * z, sxb + (mm - bb * m) * s1 + bb * dev, ss

            def inter_density(la):
                e, et, ss = move(la)
                return (
                    poisson(et) + alpha_density(gamma_stat(e), la) + n_obs * jnp.log(ss)
                )

            la, _ = jax_slice_sample_1d(
                inter_density, jnp.log(alpha), -10.0, 10.0, key=keys[4], w=0.5
            )
            alpha = jnp.exp(la)
            eta = move(la)[1]

        # 6. The spatial parameter with the disturbances held.
        Xb = stack(X @ beta)
        if sar:
            xi = eta - par * W(eta)
            cur, base = eta, jnp.zeros((T, n))
        else:
            cur, base = eta - Xb, Xb
            xi = cur - par * W(cur)
        step = st["step"]
        prop = par + step * jax.random.normal(keys[5])
        inside = (prop > d["lo"]) & (prop < d["hi"])
        at = jnp.where(inside, prop, par)
        new = solve(at, xi, cur)
        log_ratio = (
            poisson(base + new)
            - poisson(base + cur)
            + d["level_shift"] * (jnp.log1p(-par) - jnp.log1p(-at))
        )
        accepted = inside & (jnp.log(jax.random.uniform(keys[6])) < log_ratio)
        par = jnp.where(accepted, prop, par)
        eta = jnp.where(accepted, base + new, eta)
        it = st["it"]
        step = jnp.where(
            tuning,
            step * jnp.exp((accepted - _PAR_ACCEPT) / jnp.sqrt(it + 1.0)),
            step,
        )

        # 8. The spatial parameter, alpha and the level jointly, the log-means
        # held at the positive counts and the standardized disturbances at the
        # zeros (see ._core); the zeros' solve is masked to their sites.
        zm, lv = d["zmask"], d["level"]
        n_zero = jnp.sum(zm)

        def offset(xb):
            return xb + unit() if sar else jnp.zeros_like(xb)

        xb_now = stack(X @ beta)
        u_now = eta if sar else eta - xb_now
        m_now, s_now = gamma_moments(alpha)
        z_held = zm * (u_now - par * W(u_now) - offset(xb_now) - m_now) / s_now
        eta_held, par_now, alpha_now, beta_now = eta, par, alpha, beta

        def coords(x):
            full = [par_now, jnp.log(alpha_now), 0.0]
            for i, ax in enumerate(axes):
                full[ax] = x[i]
            return full

        def joint_eval(x, start):
            """``(log density, eta, zero-site solution)`` at ``x``; ``-inf`` outside."""
            r, la, shift = coords(x)
            inside = (r > d["lo"]) & (r < d["hi"])
            if free_alpha:
                inside = inside & (la > -10.0) & (la < 10.0)
            r = jnp.where(inside, r, par_now)
            la = jnp.where(inside, la, jnp.log(alpha_now))
            b = beta_now + shift * lv
            xb = stack(X @ b)
            o = offset(xb)
            mm, ss = gamma_moments(jnp.exp(la))
            up = (1.0 - zm) * (eta_held if sar else eta_held - xb)
            ez = mm + ss * z_held
            uz = solve(r, zm * (o + ez + r * W(up)), start, mask=zm)
            u = up + zm * uz
            et = u if sar else u + xb
            stat = gamma_stat(jnp.where(zm > 0, ez, u - r * W(u) - o))
            rb = b - b0
            val = (
                jnp.sum(zm * _obs_terms(et, y, d["kind"])[0])
                + T * logdet_eval(r, d["logdet"])
                - d["level_shift"] * jnp.log1p(-r)
                - jax_logdet_chebyshev_traced(r, *d["zero_logdet"])
                + n_zero * jnp.log(ss)
                + (alpha_density(stat, la) if free_alpha else alpha_now * stat)
                - 0.5 * rb @ B0inv @ rb
            )
            return jnp.where(inside, val, -jnp.inf), et, uz

        point = jnp.stack([[par, jnp.log(alpha), 0.0][ax] for ax in axes])
        dirs, widths = st["joint_dirs"], st["joint_widths"]

        def estimate(start):
            """The Hessian's eigenvectors and twice the sd along each, if it is
            positive definite (see ``._core``)."""

            pts, wts = jnp.asarray(fd_points), jnp.asarray(fd_weight)
            rows, cols = jnp.asarray(fd_i), jnp.asarray(fd_j)

            def body(m, carry):
                acc, start = carry
                f, _, start = joint_eval(point + pts[m], start)
                return acc.at[rows[m], cols[m]].add(wts[m] * f), start

            acc, start = jax.lax.fori_loop(
                0, len(stencil), body, (jnp.zeros((dim, dim)), start)
            )
            H = acc + jnp.triu(acc, 1).T
            finite = jnp.all(jnp.isfinite(H))
            lam, V = jnp.linalg.eigh(jnp.where(finite, H, jnp.eye(dim)))
            ok = finite & (jnp.min(lam) > 0)
            new_widths = 2.0 / jnp.sqrt(jnp.where(ok, lam, 1.0))
            return jnp.where(ok, V, dirs), jnp.where(ok, new_widths, widths), start

        i_it = it.astype(jnp.int32)
        refresh = jnp.logical_and(tuning, (i_it & (i_it + 1)) == 0)
        dirs, widths, start = jax.lax.cond(
            refresh, estimate, lambda s: (dirs, widths, s), st["joint_start"]
        )
        f_now, eta_now, start = joint_eval(point, start)
        k_perm, k_dirs = jax.random.split(keys[9])
        order = jax.random.permutation(k_perm, dim)

        def direction(j, carry):
            """One slice along a direction, from the point at hand."""
            x, f, et, start = carry
            dv = dirs[:, order[j]]
            t, f, et, start = jax_slice_along(
                lambda t, s: joint_eval(x + t * dv, s),
                f,
                et,
                start,
                widths[order[j]],
                jax.random.fold_in(k_dirs, j),
                _JOINT_MAX_STEPS,
                _JOINT_MAX_SHRINKS,
            )
            return x + t * dv, f, et, start

        x, _, eta, start = jax.lax.fori_loop(
            0, dim, direction, (point, f_now, eta_now, start)
        )
        par, la, shift = coords(x)
        if free_alpha:
            alpha = jnp.exp(la)
        beta = beta + shift * lv

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
            "joint_dirs": dirs,
            "joint_widths": widths,
            "joint_start": start,
        }
        draws = (par, beta, alpha)
        if effects:
            out["c"], out["sigma_c"] = st_c, sigma_c
            draws = draws + (st_c, sigma_c)
        if store_latent:
            draws = draws + (eta.ravel(),)
        return out, draws

    return sweep


def start_state(y, X, T, priors, slice_width, unit_effects, axes):
    """A chain's starting state, as :class:`._core.LatentNegBinChain` starts."""
    import jax.numpy as jnp

    nT = X.shape[0]
    n = nT // T
    eta0, beta0, alpha0 = initial_state(np.asarray(y, dtype=np.float64), X)
    if priors.alpha_fixed is not None:
        alpha0 = float(priors.alpha_fixed)
    start = {
        "par": jnp.float64(0.0),
        "beta": jnp.asarray(beta0),
        "alpha": jnp.float64(alpha0),
        "eta": jnp.asarray(eta0).reshape(T, n),
        "sxb": jnp.asarray(X @ beta0).reshape(T, n),
        "s1": jnp.ones((T, n)),
        "step": jnp.float64(0.1),
        "slice_width": jnp.float64(slice_width),
        "it": jnp.float64(0.0),
        "joint_dirs": jnp.eye(len(axes)),
        "joint_widths": jnp.asarray(_JOINT_WIDTH[list(axes)]),
        "joint_start": jnp.zeros((T, n)),
    }
    if unit_effects is not None:
        start["c"] = jnp.zeros(unit_effects.n_groups)
        start["sigma_c"] = jnp.float64(unit_effects.sd_scale)
    return start


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
    store_latent=False,
    T=1,
    unit_effects=None,
    level_shift=0.0,
    zero_logdet=None,
    progressbar=False,
    probes=None,
):
    """Run the latent-NB chains in parallel threads; one dict per chain.

    ``T``, ``unit_effects``, ``level_shift`` and ``zero_logdet`` as
    :class:`._core.LatentNegBinChain`; ``store_latent``
    adds the log-means at every kept draw, ``"latent"``.  ``probes``, a
    :class:`~neighbayes._logdet._probe_check.WarmupProbes` whose initial
    parameters are ``logdet_params``, sizes the probe pool halfway through
    warmup.
    """
    import jax
    import jax.numpy as jnp

    from ..._jax_dispatch import ensure_x64
    from ..._logdet._jax import logdet_param_evaluator
    from ..negbin_reduced._jax import _build_sparse_ctx, _sparse_solve_parts

    ensure_x64()
    nT, k = X.shape
    n = nT // T
    if unit_effects is not None and process != "sar":
        raise NotImplementedError("Random unit effects are implemented for the SAR.")
    n_groups = 0 if unit_effects is None else int(unit_effects.n_groups)
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
    data = _latent_data(
        y,
        X,
        W,
        priors,
        ld_params,
        sparse_arrays,
        T,
        unit_effects,
        level_shift=level_shift,
        zero_logdet=zero_logdet,
    )
    n_colours = int(data["classes"].shape[0])
    alpha_fixed = priors.alpha_fixed is not None
    row_std = bool(np.allclose(np.asarray(W.sum(axis=1)).ravel(), 1.0))
    axes, _ = joint_axes(X, alpha_fixed)

    static = (n, k, process, n_colours, alpha_fixed, row_std, lu_solve, store_latent)
    sweep = cached_sweep(
        ("negbin_latent", *static, T, n_groups, axes, *ld_key),
        lambda: _make_sweep(
            *static[:6], logdet_eval, lu_solve, store_latent, T, n_groups, axes
        ),
    )
    start = start_state(y, X, T, priors, slice_width, unit_effects, axes)
    states = [dict(start) for _ in seeds]
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
        model_type="latent_negbin",
        on_warmup_midpoint=None
        if probes is None
        else probe_midpoint([(probes, lambda t: t[0], lambda c: c["logdet"])]),
        consts=data,
    )
    name = "rho" if process == "sar" else "lam"
    sl = slice(None, None, thin) if thin > 1 else slice(None)
    keys = (name, "beta", "alpha")
    if n_groups:
        keys = keys + ("unit_effect", "unit_effect_sd")
    keys = keys + ("latent",)
    return [
        {key: arr[c, sl] for key, arr in zip(keys, kept)} for c in range(len(seeds))
    ]
