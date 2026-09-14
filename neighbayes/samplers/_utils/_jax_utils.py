"""Shared JAX utility helpers for Gibbs samplers.

Centralises small boilerplate — JAX/equinox availability checks, BCOO
construction, Pólya-Gamma draw factories, and the conjugate-normal β
draw — that was previously copy-pasted across six sampler sub-packages.

Not part of the public API.
"""

from __future__ import annotations

import importlib.util


def check_jax_available(*, require_equinox: bool = False) -> None:
    """Raise ``ImportError`` if JAX (and optionally equinox) is not installed.

    Parameters
    ----------
    require_equinox : bool, default False
        If ``True``, also check for the ``equinox`` package.
    """
    if importlib.util.find_spec("jax") is None:
        raise ImportError(
            "JAX is required for the JAX Gibbs sampler. Install with: pip install jax"
        )
    if require_equinox and importlib.util.find_spec("equinox") is None:
        raise ImportError(
            "equinox is required for the JAX Gibbs sampler. "
            "Install with: pip install equinox"
        )


def build_w_bcoo(W_sparse):
    """Build ``(W, Wᵀ)`` as JAX BCOO sparse matrices — never densify W.

    Parameters
    ----------
    W_sparse : scipy.sparse.spmatrix
        Row-standardised spatial weights matrix.

    Returns
    -------
    tuple[jax.experimental.sparse.BCOO, jax.experimental.sparse.BCOO]
        ``(W_bcoo, Wt_bcoo)`` — W and its transpose as JAX BCOO matrices.
    """
    from jax.experimental import sparse as jsparse

    W_bcoo = jsparse.BCOO.from_scipy_sparse(W_sparse.tocsr())
    Wt_bcoo = jsparse.BCOO.from_scipy_sparse(W_sparse.T.tocsr())
    return W_bcoo, Wt_bcoo


def run_chains_in_threads(fn, per_chain_args):
    """Call ``fn(*args)`` once per chain, concurrently, and wait for every result.

    Each chain is an ordinary ``jax.jit`` program driven from its own Python
    thread, which blocks on the result.  XLA releases the GIL while a program
    runs, so the chains execute in parallel on separate cores.

    This is how the Gibbs samplers parallelize chains on CPU, in place of
    ``jax.pmap`` and ``jax.vmap``:

    - ``jax.pmap`` is now implemented on ``shard_map``
      (``jax_pmap_shmap_merge``), and on CPU that path runs a Gibbs sweep ~8x
      slower than the same program under ``jit`` — reduced-form SAR-NB at
      n=625: 7.5 ms against 0.9 ms per sweep, even for a single chain.
    - ``jax.vmap`` lowers every ``lax.cond`` whose predicate differs across
      chains to ``select``, so the sparse solves in untaken branches (Krylov
      rebuilds, direct-solve fallbacks) run on every sweep.
    - Dispatching the chains asynchronously from one thread runs them one
      after another.

    Parameters
    ----------
    fn : callable
        Usually a ``jax.jit``-compiled function.  Compile it before calling this
        when that is cheap to arrange: threads that reach an uncompiled function
        together each compile their own copy.
    per_chain_args : list of tuple
        Positional arguments for each chain.

    Returns
    -------
    list
        ``fn``'s result for each chain, in order, with every array computed.
    """
    import jax

    if len(per_chain_args) == 1:
        return [jax.block_until_ready(fn(*per_chain_args[0]))]

    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=len(per_chain_args)) as pool:
        return list(
            pool.map(lambda args: jax.block_until_ready(fn(*args)), per_chain_args)
        )


def run_chains_chunked(
    sweep, states, warm_keys, draw_keys, *, tune, draws, on_chunk=None
):
    """Run Gibbs chains in parallel, in compiled chunks of sweeps.

    Every JAX Gibbs runner drives its chains through here.  A chunk is one
    compiled fixed-length ``lax.scan`` whose first ``n_active`` sweeps run and
    whose remaining iterations hold the state, so warmup, draws and a short
    final chunk all reuse a single program.  It is compiled once, on the
    calling thread, before the chains fan out to
    :func:`run_chains_in_threads`.  Between chunks the caller can report
    progress.

    Parameters
    ----------
    sweep : callable
        ``sweep(state, key, tuning) -> (state, trace)``: one Gibbs sweep of one
        chain.  ``tuning`` is a traced boolean, true during warmup, so a sweep
        can adapt tuning parameters it carries in ``state``.  ``trace`` is a
        pytree of arrays, recorded for post-warmup sweeps.
    states : list of pytree
        Initial state of each chain.
    warm_keys, draw_keys : sequence of PRNG keys
        Each chain's key for the warmup phase and for the draw phase.
    tune, draws : int
        Warmup and post-warmup sweeps per chain.
    on_chunk : callable, optional
        ``on_chunk(sweep_index, tuning)``, called after each chunk with the
        0-based index of the last completed sweep (warmup sweeps come first).

    Returns
    -------
    states : list of pytree
        Final state of each chain.
    traces : pytree of numpy.ndarray
        ``trace`` stacked with leading axes ``(chains, draws)``.
    """
    import jax
    import jax.numpy as jnp
    import numpy as np

    states = list(states)
    chains = len(states)
    chunk_len = max(50, max(tune, draws) // 10)
    trace_avals = jax.eval_shape(sweep, states[0], warm_keys[0], True)[1]

    def _chunk(state, key, n_active, tuning):
        def body(carry, i):
            st, kk = carry

            def _run(_):
                kk_next, sk = jax.random.split(kk)
                st_next, trace = sweep(st, sk, tuning)
                return (st_next, kk_next), trace

            def _hold(_):
                return (st, kk), jax.tree_util.tree_map(
                    lambda a: jnp.zeros(a.shape, a.dtype), trace_avals
                )

            return jax.lax.cond(i < n_active, _run, _hold, None)

        return jax.lax.scan(body, (state, key), jnp.arange(chunk_len))

    chunk = jax.jit(_chunk)
    # A zero-sweep call compiles here; threads reaching an uncompiled function
    # together would each compile their own copy.
    jax.block_until_ready(chunk(states[0], warm_keys[0], 0, True))

    def _advance(keys, n_active, tuning):
        out = run_chains_in_threads(
            chunk, [(states[c], keys[c], n_active, tuning) for c in range(chains)]
        )
        for c in range(chains):
            states[c] = out[c][0][0]
        return [o[0][1] for o in out], [o[1] for o in out]

    keys = list(warm_keys)
    done = 0
    while done < tune:
        n_active = min(chunk_len, tune - done)
        keys, _ = _advance(keys, n_active, True)
        done += n_active
        if on_chunk is not None:
            on_chunk(done - 1, True)

    keys = list(draw_keys)
    done = 0
    parts = []  # per chunk, per chain: the trace pytree cut to n_active sweeps
    while done < draws:
        n_active = min(chunk_len, draws - done)
        keys, chunk_traces = _advance(keys, n_active, False)
        parts.append(
            [
                jax.tree_util.tree_map(lambda a: np.asarray(a)[:n_active], t)
                for t in chunk_traces
            ]
        )
        done += n_active
        if on_chunk is not None:
            on_chunk(tune + done - 1, False)

    if not parts:
        return states, jax.tree_util.tree_map(
            lambda a: np.zeros((chains, 0) + a.shape, a.dtype), trace_avals
        )
    per_chain = [
        jax.tree_util.tree_map(lambda *xs: np.concatenate(xs), *[p[c] for p in parts])
        for c in range(chains)
    ]
    return states, jax.tree_util.tree_map(lambda *xs: np.stack(xs), *per_chain)


def make_pg_draw():
    """Return a JAX-compatible Pólya-Gamma draw function.

    Prefers ``pgjax.pg_sample`` (exact Devroye sampler, on-device, no
    host round-trip) when installed — it dominates every alternative on
    speed, works inside ``jax.lax.scan``, and is exact for any ``h``
    (integer or non-integer).  Falls back to the ``polyagamma`` C extension
    via ``jax.pure_callback``; this is slower (host round-trip per call)
    but still correct for any ``h``.

    Returns
    -------
    callable
        ``draw_pg(h, z, key) -> jnp.ndarray`` — vectorised PG draw.
    """
    try:
        import pgjax

        def _draw_pg(h, z, key):
            return pgjax.pg_sample(h, z, key)

        return _draw_pg
    except ImportError:
        pass

    # Fallback: numpy polyagamma via pure_callback (slower but correct).
    # Numpy polyagamma beats the old on-device "exp" approximation at every
    # size tested, so there is no reason to keep the truncated-series path.
    def _draw_pg(h, z, key):
        import jax
        import jax.numpy as jnp
        import numpy as np

        from ._polyagamma import sample_polyagamma

        h_j = jnp.asarray(h, dtype=jnp.float64)
        z_j = jnp.asarray(z, dtype=jnp.float64)
        scalar_input = h_j.ndim == 0
        if scalar_input:
            h_j = h_j[None]
            z_j = z_j[None]

        result_shape = jnp.empty_like(h_j)
        key, cb_key = jax.random.split(key)
        cb_seed = jax.random.key_data(cb_key)[0].astype(jnp.int64) % (2**31)

        def _callback(h_np, z_np, seed_np):
            rng = np.random.default_rng(int(seed_np))
            return sample_polyagamma(
                np.asarray(h_np, dtype=np.float64),
                np.asarray(z_np, dtype=np.float64),
                rng=rng,
            )

        result = jax.pure_callback(_callback, result_shape, h_j, z_j, cb_seed)
        result = jnp.maximum(result, 1e-6)
        if scalar_input:
            result = result[0]
        return result

    return _draw_pg


def conjugate_normal(Ut, omega, working, V0, mu0, key, dim):
    """Draw β ~ N(Σ (Uᵀ working + V₀⁻¹μ₀), Σ), Σ⁻¹ = UᵀΩU + V₀⁻¹.

    Shared conjugate-normal posterior draw used by all JAX Gibbs samplers.

    Parameters
    ----------
    Ut : jnp.ndarray, shape (n, dim)
        Design matrix (pre-computed, possibly transformed).
    omega : jnp.ndarray, shape (n,)
        Pólya-Gamma precision weights.
    working : jnp.ndarray, shape (n,)
        Working response (z = κ/ω or similar).
    V0 : jnp.ndarray, shape (dim,)
        Prior precision vector (diagonal of V₀⁻¹).
    mu0 : jnp.ndarray, shape (dim,)
        Prior mean vector.
    key : jax.Array
        PRNG key.
    dim : int
        Dimension of β (passed explicitly to avoid re-deriving).

    Returns
    -------
    jnp.ndarray, shape (dim,)
        Posterior draw of β.
    """
    import jax
    import jax.numpy as jnp
    from jax.scipy.linalg import cho_solve, solve_triangular

    Uw = Ut * omega[:, None]
    Sig_inv = Uw.T @ Ut + jnp.diag(V0) + 1e-10 * jnp.eye(dim)
    rhs = Ut.T @ working + V0 * mu0
    L = jnp.linalg.cholesky(Sig_inv)
    m = cho_solve((L, True), rhs)
    zc = jax.random.normal(key, shape=(dim,), dtype=jnp.float64)
    return m + solve_triangular(L.T, zc, lower=False)
