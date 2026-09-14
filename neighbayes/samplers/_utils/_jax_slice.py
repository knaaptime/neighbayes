"""JAX-accelerated slice sampler for 1-D distributions.

Implements Neal (2003) slice sampling as a pure JAX function, enabling
JIT compilation and eliminating Python dispatch overhead when the
log-density is also a JAX function.

The key advantage over the Python slice sampler is that the entire
stepping-out and shrinkage loop runs inside a single XLA kernel,
with no Python→JAX dispatch between log-density evaluations.

References
----------
Neal, R. M. (2003). Slice sampling. *Annals of Statistics*, 31(3), 705–767.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


def jax_slice_sample_1d(
    log_density,
    x0,
    lower,
    upper,
    *,
    key,
    w=1.0,
    max_steps_out=50,
    max_shrink_iters=200,
    return_steps=False,
):
    """Draw one sample from a univariate distribution via slice sampling.

    JAX-accelerated version of ``slice_sample_1d`` that runs the entire
    stepping-out and shrinkage loop inside a single JIT-compiled function.

    Parameters
    ----------
    log_density : callable
        Log-density function that takes a JAX scalar and returns a JAX
        scalar. Must be JIT-compatible (no Python side effects).
    x0 : float or jax.numpy scalar
        Current state (starting point).
    lower, upper : float or jax.numpy scalar
        Support bounds for the variable.
    key : jax.random.PRNGKey
        JAX random key for reproducibility.
    w : float, default 1.0
        Initial step-out width for interval expansion.
    max_steps_out : int, default 50
        Maximum number of stepping-out iterations per side.
    max_shrink_iters : int, default 200
        Maximum number of shrinkage iterations (safety limit).
    return_steps : bool, default False
        Also return the left and right stepping-out counts, which drive warmup
        width adaptation (:func:`adapt_slice_width`).

    Returns
    -------
    x_new : jax.numpy scalar
        New sample point.
    log_density_new : jax.numpy scalar
        Log-density evaluated at ``x_new``.
    steps_left, steps_right : jax.numpy scalar
        Stepping-out counts; returned only when ``return_steps``.
    """
    # Split key for all random draws needed
    key, key_u, key_Lu, key_Ru = jax.random.split(key, 4)

    # Evaluate log-density at current point
    log_y0 = log_density(x0)

    # Draw vertical level: log(u) where u ~ Uniform(0, f(x0))
    log_u = log_y0 + jnp.log(jax.random.uniform(key_u))

    # --- Stepping out ---
    # Initialise interval [L, R] around x0
    u_L = jax.random.uniform(key_Lu)
    L = jnp.maximum(x0 - u_L * w, lower)
    R = jnp.minimum(L + w, upper)

    # Step out left: expand L until log_density(L) <= log_u or L hits lower bound
    def _step_left_cond(carry):
        L, ld_L, i = carry
        return (ld_L > log_u) & (i < max_steps_out) & (L > lower)

    def _step_left_body(carry):
        L, ld_L, i = carry
        L_new = jnp.maximum(L - w, lower)
        ld_L_new = log_density(L_new)
        return L_new, ld_L_new, i + 1

    L, _, steps_left = jax.lax.while_loop(
        _step_left_cond,
        _step_left_body,
        (L, log_density(L), jnp.array(0)),
    )

    # Step out right: expand R until log_density(R) <= log_u or R hits upper bound
    def _step_right_cond(carry):
        R, ld_R, i = carry
        return (ld_R > log_u) & (i < max_steps_out) & (R < upper)

    def _step_right_body(carry):
        R, ld_R, i = carry
        R_new = jnp.minimum(R + w, upper)
        ld_R_new = log_density(R_new)
        return R_new, ld_R_new, i + 1

    R, _, steps_right = jax.lax.while_loop(
        _step_right_cond,
        _step_right_body,
        (R, log_density(R), jnp.array(0)),
    )

    # --- Shrinkage ---
    # Sample from [L, R] and shrink until we find a point above log_u.
    # Start with ld_init = -inf so the loop always executes at least once.
    # Use fold_in for deterministic key splitting per iteration.
    def _shrink_cond(carry):
        L, R, x_new, ld_new, i = carry
        accepted = ld_new > log_u
        collapsed = (R - L) < 1e-15
        return (~accepted) & (~collapsed) & (i < max_shrink_iters)

    def _shrink_body(carry):
        L, R, x_new, ld_new, i = carry
        # Generate proposal using fold_in for deterministic key splitting
        key_i = jax.random.fold_in(key, i)
        x_prop = L + jax.random.uniform(key_i) * (R - L)
        ld_prop = log_density(x_prop)

        # Accept if above slice
        accepted = ld_prop > log_u

        # Shrink interval
        L_new = jnp.where(x_prop < x0, x_prop, L)
        R_new = jnp.where(x_prop >= x0, x_prop, R)

        # Update if accepted
        x_out = jnp.where(accepted, x_prop, x_new)
        ld_out = jnp.where(accepted, ld_prop, ld_new)

        return L_new, R_new, x_out, ld_out, i + 1

    # Initial values for shrinkage: start with -inf log-density so the
    # loop always executes at least once (proposing a new point from [L,R]).
    x_init = x0
    ld_init = jnp.float64(-jnp.inf)

    # Run shrinkage loop
    _, _, x_new, ld_new, _ = jax.lax.while_loop(
        _shrink_cond,
        _shrink_body,
        (L, R, x_init, ld_init, jnp.array(0)),
    )

    if return_steps:
        return x_new, ld_new, steps_left, steps_right
    return x_new, ld_new


def jax_slice_sample_1d_adaptive(
    log_density,
    x0,
    lower,
    upper,
    *,
    key,
    w=1.0,
    L_prev=None,
    R_prev=None,
    max_steps_out=50,
    max_shrink_iters=200,
):
    """Draw one sample via slice sampling with persistent interval reuse.

    JAX-accelerated adaptive slice sampler that mirrors the NumPy
    ``slice_sample_1d_adaptive`` logic but remains fully JIT-compilable.

    The key enhancement over ``jax_slice_sample_1d`` is **persistent
    interval reuse**: if the previous draw's final bracket ``[L, R]``
    still brackets the slice at the new ``x0``, we skip stepping-out
    entirely.  This is the primary source of ESS improvement — when
    the posterior changes slowly (typical post-burn-in), almost all
    draws require zero step-out evaluations.

    Width adaptation (tuning ``w``) is handled outside this function
    in Python, between warmup and sampling phases — same pattern as
    MALA step-size adaptation.

    When called from a JIT-compiled context (e.g., inside
    ``jax.lax.scan``), ``L_prev`` and ``R_prev`` should always be
    JAX arrays (use ``lower``/``upper`` as sentinel values to signal
    "no persistent interval").  The function checks whether ``x0``
    lies inside ``[L_prev, R_prev]`` to decide whether to attempt
    reuse.

    Parameters
    ----------
    log_density : callable
        Log-density function (JAX scalar → JAX scalar). Must be
        JIT-compatible.
    x0 : float or jax.numpy scalar
        Current state (starting point).
    lower, upper : float or jax.numpy scalar
        Support bounds for the variable.
    key : jax.random.PRNGKey
        JAX random key for reproducibility.
    w : float, default 1.0
        Initial step-out width for interval expansion.
    L_prev, R_prev : float or jax.numpy scalar or None
        Persistent interval from the previous draw.  If ``None``,
        fresh stepping-out is used (Python-level branch, resolved at
        trace time).  When called from JIT, pass JAX arrays and use
        ``lower``/``upper`` as sentinel values — the function will
        check whether ``x0`` lies inside ``[L_prev, R_prev]``.
    max_steps_out : int, default 50
        Maximum number of stepping-out iterations per side.
    max_shrink_iters : int, default 200
        Maximum number of shrinkage iterations (safety limit).

    Returns
    -------
    x_new : jax.numpy scalar
        New sample point.
    log_density_new : jax.numpy scalar
        Log-density evaluated at ``x_new``.
    L_final : jax.numpy scalar
        Left boundary of the final interval (for persistent reuse).
    R_final : jax.numpy scalar
        Right boundary of the final interval (for persistent reuse).
    steps_out : jax.numpy scalar
        Total number of step-out evaluations (for width adaptation).
        Returns -1 if the persistent interval was reused (no fresh
        stepping-out), matching the NumPy convention.
    """
    # Split key for all random draws needed
    key, key_u, key_Lu, key_Ru = jax.random.split(key, 4)

    # Evaluate log-density at current point
    log_y0 = log_density(x0)

    # Draw vertical level: log(u) where u ~ Uniform(0, f(x0))
    log_u = log_y0 + jnp.log(jax.random.uniform(key_u))

    # --- Determine initial interval ---
    # Convert None to sentinel values (support bounds).  When L_prev
    # and R_prev span the entire support, x0 is always inside, but
    # the endpoints will be above log_u so fresh stepping-out is
    # triggered.  This avoids jax.lax.cond with None branches.
    if L_prev is None:
        L_prev = lower
    if R_prev is None:
        R_prev = upper
    L_prev = jnp.asarray(L_prev, dtype=jnp.float64)
    R_prev = jnp.asarray(R_prev, dtype=jnp.float64)

    # Check whether x0 lies inside the persistent interval.
    # If x0 is outside, the interval is stale and we must step out fresh.
    x0_inside = (x0 > L_prev) & (x0 < R_prev)

    def _fresh_stepping_out(args):
        """Standard stepping-out from x0."""
        _key_Lu, _key_Ru, _w, _log_u, _x0, _lower, _upper = args
        u_L = jax.random.uniform(_key_Lu)
        L = jnp.maximum(_x0 - u_L * _w, _lower)
        R = jnp.minimum(L + _w, _upper)

        # Step out left
        def _step_left_cond(carry):
            L_c, ld_L, i = carry
            return (ld_L > _log_u) & (i < max_steps_out) & (L_c > _lower)

        def _step_left_body(carry):
            L_c, ld_L, i = carry
            L_new = jnp.maximum(L_c - _w, _lower)
            ld_L_new = log_density(L_new)
            return L_new, ld_L_new, i + 1

        L, _, steps_left = jax.lax.while_loop(
            _step_left_cond,
            _step_left_body,
            (L, log_density(L), jnp.array(0)),
        )

        # Step out right
        def _step_right_cond(carry):
            R_c, ld_R, i = carry
            return (ld_R > _log_u) & (i < max_steps_out) & (R_c < _upper)

        def _step_right_body(carry):
            R_c, ld_R, i = carry
            R_new = jnp.minimum(R_c + _w, _upper)
            ld_R_new = log_density(R_new)
            return R_new, ld_R_new, i + 1

        R, _, steps_right = jax.lax.while_loop(
            _step_right_cond,
            _step_right_body,
            (R, log_density(R), jnp.array(0)),
        )

        return L, R, steps_left + steps_right

    def _try_reuse(args):
        """Try to reuse the persistent interval, fall back to fresh."""
        _key_Lu, _key_Ru, _w, _log_u, _x0, _lower, _upper = args

        # Clamp persistent interval to support bounds
        L = jnp.maximum(L_prev, _lower)
        R = jnp.minimum(R_prev, _upper)

        # Check if both endpoints are below the slice level
        ld_L = log_density(L)
        ld_R = log_density(R)
        left_ok = (L <= _lower) | (ld_L < _log_u)
        right_ok = (R >= _upper) | (ld_R < _log_u)
        interval_ok = x0_inside & left_ok & right_ok

        # If interval is OK, use it directly (no stepping-out)
        # If not, fall back to fresh stepping-out
        L_fresh, R_fresh, steps_fresh = _fresh_stepping_out(args)

        L = jnp.where(interval_ok, L, L_fresh)
        R = jnp.where(interval_ok, R, R_fresh)
        # -1 signals "persistent interval reused" (no fresh stepping-out)
        steps = jnp.where(interval_ok, jnp.array(-1), steps_fresh)

        return L, R, steps

    # Always try persistent interval reuse first.
    # If x0 is outside [L_prev, R_prev] or the interval doesn't
    # bracket the slice, _try_reuse falls back to fresh stepping-out.
    L, R, steps_out = _try_reuse(
        (key_Lu, key_Ru, w, log_u, x0, lower, upper),
    )

    # --- Shrinkage ---
    def _shrink_cond(carry):
        L, R, x_new, ld_new, i = carry
        accepted = ld_new > log_u
        collapsed = (R - L) < 1e-15
        return (~accepted) & (~collapsed) & (i < max_shrink_iters)

    def _shrink_body(carry):
        L, R, x_new, ld_new, i = carry
        key_i = jax.random.fold_in(key, i)
        x_prop = L + jax.random.uniform(key_i) * (R - L)
        ld_prop = log_density(x_prop)

        accepted = ld_prop > log_u

        L_new = jnp.where(x_prop < x0, x_prop, L)
        R_new = jnp.where(x_prop >= x0, x_prop, R)

        x_out = jnp.where(accepted, x_prop, x_new)
        ld_out = jnp.where(accepted, ld_prop, ld_new)

        return L_new, R_new, x_out, ld_out, i + 1

    x_init = x0
    ld_init = jnp.float64(-jnp.inf)

    L_final, R_final, x_new, ld_new, _ = jax.lax.while_loop(
        _shrink_cond,
        _shrink_body,
        (L, R, x_init, ld_init, jnp.array(0)),
    )

    return x_new, ld_new, L_final, R_final, steps_out


def adapt_slice_width(width, steps_left, steps_right, tuning=True):
    """Tune a slice width from one draw's stepping-out counts (JAX-traceable).

    The NumPy samplers' rule (:func:`._slice.update_slice_width`), with its
    constants taken from :class:`._slice.SliceWidthState`: widen when either
    side needed more than ``target_steps`` step-outs, narrow when neither side
    needed any, and clamp to ``[w_min, w_max]``.  ``tuning``, a traced boolean,
    gates the update, so a sweep can call this unconditionally and hold the
    width fixed after warmup.

    Every step-out and every rejected shrinkage proposal costs a log-density
    evaluation.  A width far wider than the posterior spends them shrinking; a
    far narrower one spends them stepping out.
    """
    from ._slice import SliceWidthState

    rule = SliceWidthState()
    grow = jnp.maximum(steps_left, steps_right) > rule.target_steps
    shrink = (steps_left == 0) & (steps_right == 0)
    tuned = jnp.where(
        grow,
        width * rule.expand_factor,
        jnp.where(shrink, width * rule.shrink_factor, width),
    )
    return jnp.where(tuning, jnp.clip(tuned, rule.w_min, rule.w_max), width)
