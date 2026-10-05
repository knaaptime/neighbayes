"""Vectorized Pólya–Gamma draw wrapper.

Thin wrapper around the ``polyagamma`` package that centralises the call
site and provides a consistent interface. Uses the library's hybrid
sampler (``method=None``) which automatically selects the fastest
algorithm based on ``h`` values: Devroye for small integer h, saddle
approximation for large h, and alternate for small non-integer h.

**Known bias.**  polyagamma 2.0.2, the latest release, is biased wherever its
hybrid sampler leaves Devroye's exact method for ``h < 8`` (measured with 2M
draws per cell against the exact moments):

* the alternate method, used for non-integer ``h`` and for integer ``h > 1``
  at ``z > 1``, overstates the mean by 0.4–1% and the variance by 1–5% at
  ``|z| < 4``; its bounding kernel carries ``log(sqrt(π/2))`` where
  ``log(π/2)`` belongs.  Below ``h = 1`` it understates the mean by 0.4–0.8%;
* the saddlepoint approximation, used for ``4 < h < 8`` at ``z ≤ 4``,
  understates the variance by 1–2%.

Negative-binomial augmentation draws at ``h = y + α``, which usually falls in
that range, so every NumPy Gibbs sampler for counts is exposed; logit draws
at ``h = 1`` are exact.  A fix has been submitted upstream.  Until a release
carries it, :func:`sample_polyagamma` warns when a draw falls in the biased
range, and the JAX backend, whose sampler (pgjax) is exact, is the remedy.

If ``polyagamma`` is unavailable, an ``ImportError`` is raised with a
helpful message.
"""

from __future__ import annotations

import functools
import warnings

import numpy as np

#: The last polyagamma release known to carry the bias.  Raise it if a later
#: release ships without the fix.
_LAST_BIASED_RELEASE = (2, 0, 2)

BIAS_MESSAGE = (
    "Pólya–Gamma draws on the NumPy backend come from the polyagamma package, "
    "whose sampler is biased for shape h < 8 (mean by up to 1%, variance by up "
    "to 5%); negative-binomial, ZINB and hurdle samplers draw at h = y + α. Use "
    "gibbs_backend='jax' with pgjax installed, whose sampler is exact."
)


@functools.lru_cache(maxsize=1)
def _installed_release_is_biased() -> bool:
    """Whether the installed polyagamma is a release known to carry the bias."""
    from importlib.metadata import PackageNotFoundError, version

    try:
        installed = version("polyagamma")
    except PackageNotFoundError:
        return True
    release = []
    for part in installed.split("."):
        digits = "".join(ch for ch in part if ch.isdigit())
        if not digits:
            break
        release.append(int(digits))
    return tuple(release) <= _LAST_BIASED_RELEASE


def draws_are_biased(h, z) -> bool:
    """Whether polyagamma's hybrid sampler draws any ``PG(h, z)`` with bias.

    The hybrid sampler is exact (Devroye) at ``h = 1`` and at integer ``h``
    with ``z ≤ 1``; below ``h = 8`` it otherwise uses the biased alternate or
    saddlepoint methods, whose error fades by ``|z| = 4``.  Always ``False``
    once the installed release is newer than :data:`_LAST_BIASED_RELEASE`.
    """
    if not _installed_release_is_biased():
        return False
    h = np.asarray(h, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    exact = (h == 1.0) | ((h == np.floor(h)) & (z <= 1.0))
    return bool(np.any((h < 8.0) & ~exact & (np.abs(z) < 4.0)))


def sample_polyagamma(
    h: np.ndarray,
    z: np.ndarray,
    *,
    rng: np.random.Generator | None = None,
    warn: bool = True,
) -> np.ndarray:
    """Draw vectorized Pólya–Gamma samples.

    For each element i, draws ω_i ~ PG(h_i, z_i) where PG is the
    Pólya–Gamma distribution (Polson, Scott & Windle, 2013).

    Parameters
    ----------
    h : ndarray of shape (n,)
        Shape parameters. For NB augmentation: h_i = y_i + alpha.
        Must be positive.
    z : ndarray of shape (n,)
        Tilting parameters. For NB augmentation: z_i = eta_i.
    rng : numpy.random.Generator, optional
        Random state. If None, a fresh generator is created.
    warn : bool, default True
        Warn (``RuntimeWarning``) when any draw falls where the polyagamma
        release is biased (see the module notes).  Python shows a given
        warning once per location, so a sampler warns once.  Pass ``False``
        for draws whose bias cannot matter, such as starting values.

    Returns
    -------
    omega : ndarray of shape (n,)
        PG(h, z) draws. All elements are positive.

    Raises
    ------
    ImportError
        If the ``polyagamma`` package is not installed.
    ValueError
        If ``h`` and ``z`` have different shapes or if any h <= 0.
    """
    try:
        from polyagamma import random_polyagamma as _pg_draw
    except ImportError:
        raise ImportError(
            "The 'polyagamma' package is required for Pólya–Gamma sampling. "
            "Install it with: pip install polyagamma"
        )

    if rng is None:
        rng = np.random.default_rng()

    h = np.asarray(h, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)

    if h.shape != z.shape:
        raise ValueError(
            f"h and z must have the same shape, got {h.shape} and {z.shape}."
        )
    if np.any(h <= 0):
        raise ValueError("All h values must be positive.")

    # Use the library's hybrid method (method=None) which automatically
    # selects the fastest algorithm based on h values:
    #   - Devroye for small integer h (e.g. logit with h=1)
    #   - Saddle approximation for large h (h > ~20)
    #   - Alternate for small non-integer h
    # The saddle method is O(1) per draw regardless of h magnitude,
    # making it essential for NB-PG augmentation where h_i = y_i + alpha
    # can be very large (e.g. 10^6 for high-mu NB observations).
    # Previously, we forced method="alternate" for non-integer h, but
    # the alternate method's cost scales with h, making it catastrophically
    # slow for large h values.
    if warn and draws_are_biased(h, z):
        warnings.warn(BIAS_MESSAGE, RuntimeWarning, stacklevel=2)
    omega = _pg_draw(h=h, z=z, method=None, random_state=rng)
    return np.asarray(omega, dtype=np.float64)
