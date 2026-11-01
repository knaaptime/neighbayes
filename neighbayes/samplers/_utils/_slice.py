"""Univariate slice sampler with stepping-out and shrinkage.

Implements Neal (2003) slice sampling for 1-D distributions with bounded
or unbounded support. Returns both the new sample and the log-density
value at that sample, allowing callers to cache expensive evaluations
(e.g., log-determinant in the ρ update).

Adaptive width variant
----------------------
`slice_sample_1d_adaptive` tracks how many step-out steps were needed
and tunes the initial width ``w`` so that the interval rarely needs
more than 1–2 expansions.  This mimics the scale-factor adaptation in
NumPyro's ESS (Ensemble Slice Sampling) and is much cheaper than
mode-finding.

References
----------
Neal, R. M. (2003). Slice sampling. *Annals of Statistics*, 31(3), 705–767.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

# ---------------------------------------------------------------------------
# Adaptive width state
# ---------------------------------------------------------------------------


@dataclass
class SliceWidthState:
    """Mutable state for adaptive slice-sampler width tuning.

    Parameters
    ----------
    w : float
        Current step-out width.
    w_min : float
        Lower bound for ``w``.
    w_max : float
        Upper bound for ``w``.
    expand_factor : float
        Multiplicative factor when too many step-outs are observed.
    shrink_factor : float
        Multiplicative factor when no step-outs are observed.
    target_steps : int
        Desired maximum number of step-out steps per side.
    L : float or None
        Left boundary of the persistent interval from the previous draw.
    R : float or None
        Right boundary of the persistent interval from the previous draw.
    """

    w: float = 1.0
    w_min: float = 1e-6
    w_max: float = 1e3
    expand_factor: float = 1.10
    shrink_factor: float = 0.95
    target_steps: int = 2
    L: float | None = None
    R: float | None = None


def slice_sample_1d(
    log_density: Callable[[float], float],
    x0: float,
    lower: float,
    upper: float,
    *,
    w: float = 1.0,
    max_steps_out: int = 50,
    max_shrink_iters: int = 500,
    rng: np.random.Generator | None = None,
) -> tuple[float, float]:
    """Draw one sample from a univariate distribution via slice sampling.

    Uses Neal's stepping-out procedure to find the slice interval, then
    shrinks to find the new point. The log-density may be called multiple
    times per draw (typically 5–15 calls).

    Parameters
    ----------
    log_density : callable
        Log-density function (up to a normalising constant). Must accept
        a single float and return a float. May be called multiple times.
    x0 : float
        Current state (starting point).
    lower, upper : float
        Support bounds for the variable. Use ``-np.inf`` / ``np.inf``
        for unbounded support.
    w : float, default=1.0
        Initial step-out width for interval expansion.
    max_steps_out : int, default=50
        Maximum number of stepping-out iterations to prevent infinite
        loops on very flat densities.    max_shrink_iters : int, default 500
        Maximum number of shrinkage iterations.  Prevents infinite
        loops if the log-density returns nan (treated as -inf) at
        every point in the interval.    rng : numpy.random.Generator, optional
        Random state. If None, a fresh generator is created.

    Returns
    -------
    x_new : float
        New sample point.
    log_density_new : float
        Log-density evaluated at ``x_new``. Callers can cache this to
        avoid redundant evaluations when the log-density is expensive
        (e.g., log-determinant in the ρ update).

    Raises
    ------
    ValueError
        If ``x0`` is not in ``[lower, upper]`` or if ``lower >= upper``.
    """
    if rng is None:
        rng = np.random.default_rng()

    if lower >= upper:
        raise ValueError(f"lower ({lower}) must be less than upper ({upper}).")
    if x0 < lower or x0 > upper:
        raise ValueError(f"x0 ({x0}) must be in [lower, upper] = [{lower}, {upper}].")

    # Evaluate log-density at current point
    log_y0 = log_density(x0)
    # Guard against nan at the current point — treat as very negative
    # so the slice level is drawn at -inf and any finite candidate is
    # accepted, allowing the sampler to escape a nan region.
    if not np.isfinite(log_y0):
        log_y0 = -1e300

    # Draw vertical level: log(u) where u ~ Uniform(0, f(x0))
    log_u = log_y0 + np.log(rng.uniform())

    # --- Stepping out ---
    # Initialise interval [L, R] around x0
    u_rand = rng.uniform()
    L = x0 - u_rand * w
    R = L + w

    # Clamp to support bounds
    L = max(L, lower)
    R = min(R, upper)

    # Step out left
    steps_out_left = 0
    while L > lower and log_density(L) > log_u and steps_out_left < max_steps_out:
        L -= w
        L = max(L, lower)
        steps_out_left += 1

    # Step out right
    steps_out_right = 0
    while R < upper and log_density(R) > log_u and steps_out_right < max_steps_out:
        R += w
        R = min(R, upper)
        steps_out_right += 1

    # --- Shrinkage ---
    # Sample from [L, R] and shrink until we find a point above log_u
    for _shrink_iter in range(max_shrink_iters):
        x_new = L + rng.uniform() * (R - L)
        log_density_new = log_density(x_new)
        # Treat nan as -inf: reject and shrink the interval.
        # Without this guard, nan comparisons (nan > log_u → False,
        # nan < x0 → False) cause an infinite loop because neither
        # L nor R is updated.
        if np.isnan(log_density_new):
            log_density_new = -np.inf
        if log_density_new > log_u:
            # Accept: point is above the slice
            return x_new, log_density_new

        # Shrink the interval
        if x_new < x0:
            L = x_new
        else:
            R = x_new

        # Safety: if interval has collapsed, return x0
        if R - L < 1e-15:
            return x0, log_y0

    # Exhausted shrinkage budget — return x0 as a fallback
    return x0, log_y0


def slice_sample_1d_adaptive(
    log_density: Callable[[float], float],
    x0: float,
    lower: float,
    upper: float,
    *,
    width_state: SliceWidthState,
    max_steps_out: int = 50,
    max_shrink_iters: int = 500,
    rng: np.random.Generator | None = None,
    log_density_x0: float | None = None,
) -> tuple[float, float, int, int]:
    r"""Draw one sample with adaptive width and persistent interval tuning.

    Implements Neal (2003) slice sampling with two enhancements:

    1. **Adaptive width** — tracks step-out counts and tunes ``w`` so
       that the interval rarely needs more than ``target_steps``
       expansions per side.
    2. **Persistent interval** — stores the final bracket ``[L, R]``
       from draw :math:`t` and reuses it to initialize the interval for
       draw :math:`t+1`.  When the posterior changes slowly (typical
       post-burn-in), this eliminates almost all stepping-out.

    The vertical slice level is drawn as

    .. math::

        u \sim \mathrm{Uniform}(0, f(x_0)),
        \qquad \log u = \log f(x_0) + \log(\mathrm{Uniform}(0,1)).

    Parameters
    ----------
    log_density : callable
        Log-density function (up to a normalising constant).
    x0 : float
        Current state (starting point).
    lower, upper : float
        Support bounds for the variable.
    width_state : SliceWidthState
        Mutable state holding ``w`` and the persistent ``[L, R]``.
    max_steps_out : int, default=50
        Maximum number of stepping-out iterations.    max_shrink_iters : int, default 500
        Maximum number of shrinkage iterations.  Prevents infinite
        loops if the log-density returns nan (treated as -inf) at
        every point in the interval.    rng : numpy.random.Generator, optional
        Random state.
    log_density_x0 : float, optional
        Pre-computed log-density at ``x0``. If provided, avoids one
        redundant call to ``log_density(x0)``.

    Returns
    -------
    x_new : float
        New sample point.
    log_density_new : float
        Log-density evaluated at ``x_new``.
    steps_out_left : int
        Number of left step-out steps taken.
    steps_out_right : int
        Number of right step-out steps taken.
    """
    if rng is None:
        rng = np.random.default_rng()

    if lower >= upper:
        raise ValueError(f"lower ({lower}) must be less than upper ({upper}).")
    if x0 < lower or x0 > upper:
        raise ValueError(f"x0 ({x0}) must be in [lower, upper] = [{lower}, {upper}].")

    w = width_state.w

    # Evaluate log-density at current point (or use cached value)
    log_y0 = log_density_x0 if log_density_x0 is not None else log_density(x0)
    # Guard against nan at the current point — treat as very negative
    # so the slice level is drawn at -inf and any finite candidate is
    # accepted, allowing the sampler to escape a nan region.
    if not np.isfinite(log_y0):
        log_y0 = -1e300

    # Draw vertical level: log(u) where u ~ Uniform(0, f(x0))
    log_u = log_y0 + np.log(rng.uniform())

    # --- Stepping out (with persistent interval) ---
    # If we have a persistent interval from the previous draw and x0
    # lies inside it, try to reuse it.  This avoids stepping-out when
    # the posterior changes slowly (typical post-burn-in).
    persistent_L = width_state.L
    persistent_R = width_state.R
    has_persistent = (
        persistent_L is not None
        and persistent_R is not None
        and persistent_L < x0 < persistent_R
    )

    if has_persistent:
        L = max(persistent_L, lower)
        R = min(persistent_R, upper)
        # Verify the interval still brackets the slice.  If both
        # endpoints are below log_u, no stepping-out is needed.
        left_ok = L <= lower or log_density(L) < log_u
        right_ok = R >= upper or log_density(R) < log_u
        if left_ok and right_ok:
            # Interval was reused — signal to skip width update.
            # Returning -1 tells update_slice_width that no fresh
            # stepping-out occurred, so the width should not be shrunk.
            steps_out_left = -1
            steps_out_right = -1
        else:
            # Fall back to fresh stepping-out from x0
            u_rand = rng.uniform()
            L = x0 - u_rand * w
            R = L + w
            L = max(L, lower)
            R = min(R, upper)
            steps_out_left = 0
            while (
                L > lower and log_density(L) > log_u and steps_out_left < max_steps_out
            ):
                L -= w
                L = max(L, lower)
                steps_out_left += 1
            steps_out_right = 0
            while (
                R < upper and log_density(R) > log_u and steps_out_right < max_steps_out
            ):
                R += w
                R = min(R, upper)
                steps_out_right += 1
    else:
        # Standard stepping-out from x0
        u_rand = rng.uniform()
        L = x0 - u_rand * w
        R = L + w

        # Clamp to support bounds
        L = max(L, lower)
        R = min(R, upper)

        # Step out left
        steps_out_left = 0
        while L > lower and log_density(L) > log_u and steps_out_left < max_steps_out:
            L -= w
            L = max(L, lower)
            steps_out_left += 1

        # Step out right
        steps_out_right = 0
        while R < upper and log_density(R) > log_u and steps_out_right < max_steps_out:
            R += w
            R = min(R, upper)
            steps_out_right += 1

    # --- Shrinkage ---
    for _shrink_iter in range(max_shrink_iters):
        x_new = L + rng.uniform() * (R - L)
        log_density_new = log_density(x_new)

        # Treat nan as -inf: reject and shrink the interval.
        # Without this guard, nan comparisons (nan > log_u → False,
        # nan < x0 → False) cause an infinite loop because neither
        # L nor R is updated.
        if np.isnan(log_density_new):
            log_density_new = -np.inf

        if log_density_new > log_u:
            # Store the final interval for the next draw
            width_state.L = L
            width_state.R = R
            return x_new, log_density_new, steps_out_left, steps_out_right

        if x_new < x0:
            L = x_new
        else:
            R = x_new

        if R - L < 1e-15:
            width_state.L = L
            width_state.R = R
            return x0, log_y0, steps_out_left, steps_out_right

    # Exhausted shrinkage budget — return x0 as a fallback
    width_state.L = L
    width_state.R = R
    return x0, log_y0, steps_out_left, steps_out_right


def update_slice_width(
    width_state: SliceWidthState,
    steps_out_left: int,
    steps_out_right: int,
) -> None:
    """Update ``width_state.w`` based on observed step-out counts.

    Logic (mimics NumPyro ESS ``tune_mu``):

    * If either side needed **more** than ``target_steps`` expansions,
      the width is too small → multiply by ``expand_factor``.
    * If **both** sides needed **zero** expansions, the width is too
      large → multiply by ``shrink_factor``.
    * Otherwise leave ``w`` unchanged.

    The updated width is clamped to ``[w_min, w_max]``.

    Parameters
    ----------
    width_state : SliceWidthState
        Mutable state object; ``w`` is updated in-place.
    steps_out_left : int
        Number of left step-out steps from the last draw.
    steps_out_right : int
        Number of right step-out steps from the last draw.
    """
    # Skip width update when the persistent interval was reused
    # (signaled by steps_out == -1).  Reusing the interval means the
    # width was already adequate — shrinking would be incorrect.
    if steps_out_left < 0 or steps_out_right < 0:
        return

    max_steps = max(steps_out_left, steps_out_right)

    if max_steps > width_state.target_steps:
        width_state.w *= width_state.expand_factor
    elif steps_out_left == 0 and steps_out_right == 0:
        width_state.w *= width_state.shrink_factor

    # Clamp
    width_state.w = max(width_state.w_min, min(width_state.w_max, width_state.w))


def slice_along(
    f: Callable[[float], tuple[float, object]],
    f0: float,
    aux0: object,
    w: float,
    rng: np.random.Generator,
    max_steps: int = 16,
    max_shrinks: int = 200,
) -> tuple[float, float, object]:
    """One slice on ``t ↦ f(t)`` from ``t = 0``, for a move along a fixed line.

    ``f(t)`` returns ``(log density, aux)``, ``aux`` anything the caller wants
    back from the accepted point; ``f0`` and ``aux0`` are its values at 0.  The
    stepping-out budget ``max_steps`` is shared between the sides at random
    (Neal 2003, Fig. 3), so the slice stays reversible when the budget binds.
    Returns ``(t, log density, aux)`` at the new point, or at 0 when the
    shrinkage budget runs out.
    """
    level = f0 - rng.standard_exponential()
    lo = -w * rng.random()
    hi = lo + w
    left = int(max_steps * rng.random())
    right = max_steps - 1 - left
    while left > 0 and f(lo)[0] > level:
        lo -= w
        left -= 1
    while right > 0 and f(hi)[0] > level:
        hi += w
        right -= 1
    for _ in range(max_shrinks):
        t = lo + rng.random() * (hi - lo)
        ft, aux = f(t)
        if ft > level:
            return t, ft, aux
        if t < 0:
            lo = t
        else:
            hi = t
    return 0.0, f0, aux0
