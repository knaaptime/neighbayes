"""Runtime configuration helpers for :mod:`neighbayes`.

This module exposes opt-in helpers that tune global runtime behavior
(compile caches, etc.) for the JAX and PyTensor toolchains used by the
package. Nothing in this module is invoked at import time — users must
call the helpers explicitly.
"""

from __future__ import annotations

from pathlib import Path

__all__ = ["checkpoints", "enable_compile_cache"]


def enable_compile_cache(path: str | Path | None = None) -> Path:
    """Enable persistent on-disk compile caches for JAX and PyTensor.

    JAX-backed NUTS samplers (``"numpyro"``, ``"blackjax"``) and the
    PyTensor-compiled PyMC sampler can both reuse compiled artefacts
    across processes when given a stable cache directory. This helper
    sets:

    * ``jax.config.update("jax_compilation_cache_dir", <path>/"jax")``
    * ``pytensor.config.compiledir = <path>/"pytensor"``

    Both subdirectories are created if they do not already exist. The
    helper is idempotent and safe to call from notebooks or scripts at
    startup.

    Parameters
    ----------
    path :
        Root directory for the caches. When omitted, defaults to the
        platform-appropriate user cache directory
        (``platformdirs.user_cache_dir("neighbayes")``).

    Returns
    -------
    pathlib.Path
        The resolved root cache path.

    Notes
    -----
    This is an opt-in helper. The package does not enable caching
    automatically because tests and CI generally prefer ephemeral
    directories.
    """
    if path is None:
        from platformdirs import user_cache_dir

        root = Path(user_cache_dir("neighbayes")) / "compile_cache"
    else:
        root = Path(path)
    jax_dir = root / "jax"
    pytensor_dir = root / "pytensor"
    jax_dir.mkdir(parents=True, exist_ok=True)
    pytensor_dir.mkdir(parents=True, exist_ok=True)

    import jax
    import pytensor

    jax.config.update("jax_compilation_cache_dir", str(jax_dir))
    pytensor.config.compiledir = str(pytensor_dir)
    return root


def checkpoints(path: str | Path, *, every: float = 300.0):
    """Save long JAX Gibbs fits to disk as they run, and resume them after a crash.

    A context manager.  Every JAX Gibbs run started inside it is saved under
    ``path`` at the end of a 128-sweep chunk, at most once every ``every``
    seconds, in a directory named by a fingerprint of everything that
    determines its draws: the run's lengths, the sampler's structure, every
    chain's starting state and random keys, and the data.  Starting the same
    fit again, with the same data, options and ``random_seed``, inside the same
    ``path`` resumes it from its last save, and the draws are those an
    uninterrupted run would have returned.  A fit whose inputs differ starts
    afresh in its own directory, and so does any fit without a fixed
    ``random_seed``.  A run that finishes removes its directory.

    Parameters
    ----------
    path : str or Path
        Directory for the checkpoints; created if missing.
    every : float, default 300
        Least number of seconds between saves of one run.  Each save writes
        every chain's state, so at large ``n`` it costs a few seconds.

    Examples
    --------
    >>> from neighbayes.config import checkpoints
    >>> with checkpoints("runs/la-flows"):  # doctest: +SKIP
    ...     idata = model.fit(backend="jax", random_seed=7)

    Notes
    -----
    With :func:`enable_compile_cache` a resumed fit also skips compiling its
    sampler.  The NumPy samplers are not checkpointed.  Checkpoints are
    pickles; load only those you wrote.
    """
    from .samplers._utils._checkpoint import checkpoints as _checkpoints

    return _checkpoints(path, every=every)
