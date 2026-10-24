"""JAX backend of the latent hurdle and zero-inflated NB samplers (:mod:`._core`).

One sweep is the latent logit's sweep on the zero part
(:mod:`..logit_latent._jax`) and the latent NB's on the count part
(:mod:`..negbin_latent._jax`), each with its own consts.  The zero-inflated
sweep passes the count part's ``logq = -e^eta`` to the zero part, which draws
the zero counts' sides of zero, and the active units to the count part as its
observed counts; it accumulates each zero's probability of being structural
after warmup.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from .._utils._jax_utils import cached_sweep, run_chains_chunked
from .._utils._latent_loglik import POISSON, TRUNCATED, UNOBSERVED


def _logdet_parts(logdet):
    """``(evaluator, params, cache key)`` from ``logdet_jax_params``' ``(kind, params)``."""
    import jax.numpy as jnp

    from ..._logdet._jax import logdet_param_evaluator

    kind, params = logdet
    if kind == "closure":
        fn = params

        def evaluate(r, _params):
            return fn(r)

        return evaluate, jnp.float64(0.0), ("closure", id(fn))
    return logdet_param_evaluator(kind), params, (kind,)


def run_chains_jax_latent_two_part(
    *,
    model,
    y,
    X,
    Z,
    W,
    W_zero,
    priors,
    priors_zero,
    logdet,
    logdet_zero,
    draws,
    tune,
    thin=1,
    seeds,
    slice_width=0.4,
    store_latent=False,
):
    """Run the latent ``model`` (``"hurdle"`` or ``"zinb"``) chains; one dict per chain.

    ``logdet`` and ``logdet_zero`` are the parts' ``logdet_jax_params``.  The
    keys are those of :func:`._core.run_chain`.
    """
    import jax
    import jax.numpy as jnp

    from ..._jax_dispatch import ensure_x64
    from ..logit_latent import _jax as logit_jax
    from ..negbin_latent import _jax as nb_jax
    from ..negbin_latent._core import initial_state
    from ..negbin_reduced._jax import _build_sparse_ctx, _sparse_solve_parts

    ensure_x64()
    y = np.asarray(y, dtype=np.float64)
    n, k = X.shape
    p = Z.shape[1]
    zinb = model == "zinb"
    pos = y > 0

    # The zero part: the latent logit, its unknown sides drawn for a ZINB.
    W_zero = sp.csr_matrix(W_zero, dtype=np.float64)
    eval_z, params_z, key_z = _logdet_parts(logdet_zero)
    sign = np.where(pos, 1.0, 0.0 if zinb else -1.0)
    data_z = logit_jax._latent_data(
        pos.astype(np.float64), Z, W_zero, priors_zero, params_z, sign=sign
    )
    n_colours_z = int(data_z["classes"].shape[0])
    if zinb:
        data_z["logq"] = jnp.zeros(n)
    static_z = (n, p, "sar", n_colours_z, store_latent, zinb, *key_z)
    sweep_z = cached_sweep(
        ("logit_latent", *static_z),
        lambda: logit_jax._make_sweep(
            n, p, "sar", n_colours_z, eval_z, store_latent, mixed=zinb
        ),
    )

    # The count part: the latent NB, observing the positive counts (hurdle)
    # or, set each sweep, the active units' (ZINB).
    W = sp.csr_matrix(W, dtype=np.float64)
    eval_c, params_c, key_c = _logdet_parts(logdet)
    lu_solve, sparse_arrays = _sparse_solve_parts(_build_sparse_ctx(W, n))
    sparse_arrays = {key: sparse_arrays[key] for key in ("Ai", "Aj", "eye", "w")}
    kind = np.where(pos, TRUNCATED, UNOBSERVED) if not zinb else None
    data_c = nb_jax._latent_data(y, X, W, priors, params_c, sparse_arrays, kind=kind)
    n_colours_c = int(data_c["classes"].shape[0])
    alpha_fixed = priors.alpha_fixed is not None
    row_std = bool(np.allclose(np.asarray(W.sum(axis=1)).ravel(), 1.0))
    static_c = (n, k, "sar", n_colours_c, alpha_fixed, row_std, lu_solve, store_latent)
    sweep_c = cached_sweep(
        ("negbin_latent", *static_c, 1, 0, *key_c),
        lambda: nb_jax._make_sweep(*static_c[:6], eval_c, lu_solve, store_latent, 1, 0),
    )

    zeros = jnp.asarray(~pos)

    def build():
        def sweep(st, key, tuning, d):
            k_z, k_c = jax.random.split(key)
            dz, dc = d["zero"], d["count"]
            if zinb:
                # q_i = P(y_i = 0 | active) = exp(-e^eta_i).
                dz = {**dz, "logq": -jnp.exp(st["count"]["eta"].ravel())}
            st_z, draws_z = sweep_z(st["zero"], k_z, tuning, dz)
            new = {"zero": st_z}
            if zinb:
                active = st_z["ystar"] > 0
                dc = {
                    **dc,
                    "kind": jnp.where(active, POISSON, UNOBSERVED)
                    .astype(jnp.int32)
                    .reshape(1, n),
                }
                keep = jnp.where(tuning, 0.0, 1.0)
                pz = jnp.where(d["zeros"], st_z["pneg"], 0.0)
                new["structural"] = st["structural"] + keep * pz
                new["structural_sq"] = st["structural_sq"] + keep * pz * pz
                new["n_kept"] = st["n_kept"] + keep
            st_c, draws_c = sweep_c(st["count"], k_c, tuning, dc)
            new["count"] = st_c
            return new, tuple(draws_z) + tuple(draws_c)

        return sweep

    sweep = cached_sweep(("two_part_latent", model, *static_z, *static_c), build)

    eta0, beta0, alpha0 = initial_state(y, X)
    if alpha_fixed:
        alpha0 = float(priors.alpha_fixed)
    m0 = np.sqrt(2.0 * np.pi) * np.log(2.0)
    start_z = {
        "par": jnp.float64(0.0),
        "beta": jnp.asarray(data_z["b0"]),
        "ystar": jnp.asarray(0.5 * np.where(sign != 0, sign, 1.0)),
        "m": jnp.full(n, m0),
        "slice_width": jnp.float64(slice_width),
    }
    if zinb:
        start_z["pneg"] = jnp.zeros(n)
    start_c = {
        "par": jnp.float64(0.0),
        "beta": jnp.asarray(beta0),
        "alpha": jnp.float64(alpha0),
        "eta": jnp.asarray(eta0).reshape(1, n),
        "sxb": jnp.asarray(X @ beta0).reshape(1, n),
        "s1": jnp.ones((1, n)),
        "step": jnp.float64(0.1),
        "slice_width": jnp.float64(slice_width),
        "it": jnp.float64(0.0),
    }
    start = {"zero": start_z, "count": start_c}
    if zinb:
        start.update(
            structural=jnp.zeros(n), structural_sq=jnp.zeros(n), n_kept=jnp.float64(0)
        )
    consts = {"zero": data_z, "count": data_c, "zeros": zeros}
    states = [dict(start) for _ in seeds]
    warm = [jax.random.PRNGKey(int(s)) for s in seeds]
    draw = [jax.random.fold_in(jax.random.PRNGKey(int(s)), 1) for s in seeds]
    final, kept = run_chains_chunked(
        sweep, states, warm, draw, tune=tune, draws=draws, consts=consts
    )
    sl = slice(None, None, thin) if thin > 1 else slice(None)
    keys = ("rho_zero", "beta_zero") + (("latent_zero",) if store_latent else ())
    keys = keys + ("rho", "beta", "alpha") + (("latent",) if store_latent else ())
    out = []
    for c in range(len(seeds)):
        r = {key: np.asarray(arr[c, sl]) for key, arr in zip(keys, kept)}
        if zinb:
            st = final[c]
            k_ = max(float(st["n_kept"]), 1.0)
            mean = np.asarray(st["structural"])[~pos] / k_
            second = np.asarray(st["structural_sq"])[~pos] / k_
            r["structural_prob"] = mean
            r["structural_prob_sd"] = np.sqrt(np.maximum(second - mean**2, 0.0))
        out.append(r)
    return out
