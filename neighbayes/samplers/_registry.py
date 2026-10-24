"""Registry mapping ``(likelihood, structure)`` → a Gibbs runner.

A plain module-level dict — no plugins, no entry points (3 structures × ~7
likelihoods is well under the point where anything fancier pays off).

Contract
--------
Each model family registers **one** :class:`GibbsEntry` whose ``run`` callable
takes the model plus the resolved sampling controls and returns a finished
:class:`xarray.DataTree`::

    def run(model, *, draws, tune, chains, random_seed, thin, n_jobs,
            progressbar, backend, **family_opts) -> xr.DataTree

Returning a finished ``DataTree`` (rather than raw chain results) is a
deliberate low-risk choice: class-based families keep ``Gaussian*Gibbs.fit`` and
function-based families keep ``run_chains`` + ``gibbs_to_inference_data``; the
registry unifies only the *dispatch* layer, not the sampler internals.

Registration strategy
----------------------
Each runner module calls :func:`register` at import time.  A model cannot reach
``fit(sampler="gibbs")`` without its class being imported, and each model module
imports its (cheap — JAX stays lazy inside function bodies) runner module, so the
entry is always registered by the time :func:`resolve` runs.  NUTS needs no
entry; a missing entry simply means "no Gibbs sampler for this model".
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    import xarray as xr

# A runner returns a finished DataTree.
GibbsRunner = Callable[..., "xr.DataTree"]


@dataclass(frozen=True)
class GibbsEntry:
    """One family's Gibbs sampler, as seen by the base ``fit()``.

    Attributes
    ----------
    run
        Callable ``(model, *, draws, tune, chains, ...) -> xr.DataTree``.
    backends
        Execution backends this family supports, a subset of
        ``{"jax", "numpy"}``.  ``"auto"`` resolves to JAX when the family
        supports it and JAX is installed, else NumPy.
    options
        Names of family-specific ``fit`` keyword arguments this runner accepts
        (e.g. ``{"init_jitter", "slice_width", "krylov_degree"}``).  The base
        ``fit()`` pops exactly these from ``**sample_kwargs`` and rejects any
        leftover when ``sampler="gibbs"``.
    supports_robust
        Whether the Gibbs sampler supports a robust (Student-t) likelihood.
        When ``False`` the base ``fit()`` raises a clear ``NotImplementedError``
        for ``robust=True`` models instead of each runner re-checking.
    skips_log_likelihood
        Whether ``run`` accepts ``log_likelihood=`` and skips computing the
        pointwise log-likelihood when it is ``False``.  Every family stores it
        only on request (:func:`run_entry`); for the others the group is
        computed and then dropped.
    """

    run: GibbsRunner
    backends: frozenset[str]
    options: frozenset[str] = field(default_factory=frozenset)
    supports_robust: bool = False
    skips_log_likelihood: bool = False


_REGISTRY: dict[tuple[str, str], GibbsEntry] = {}


def register(
    likelihood: str,
    structure: str,
    *,
    run: GibbsRunner,
    backends,
    options=(),
    supports_robust: bool = False,
    skips_log_likelihood: bool = False,
) -> GibbsEntry:
    """Register (once) the Gibbs runner for ``(likelihood, structure)``."""
    key = (likelihood, structure)
    if key in _REGISTRY:
        raise ValueError(f"duplicate Gibbs registry entry for {key!r}")
    backends = frozenset(backends)
    if not backends:
        raise ValueError(f"registry entry {key!r} must support at least one backend")
    if not backends <= {"jax", "numpy"}:
        raise ValueError(
            f"registry entry {key!r} backends must be a subset of "
            f"{{'jax', 'numpy'}}, got {sorted(backends)}"
        )
    entry = GibbsEntry(
        run=run,
        backends=backends,
        options=frozenset(options),
        supports_robust=supports_robust,
        skips_log_likelihood=bool(skips_log_likelihood),
    )
    _REGISTRY[key] = entry
    return entry


def resolve(likelihood: str, structure: str) -> GibbsEntry | None:
    """Return the registered entry for ``(likelihood, structure)`` or ``None``."""
    return _REGISTRY.get((likelihood, structure))


def resolve_backend(requested: str, entry: GibbsEntry, *, jax_ok: bool) -> str:
    """Map a Gibbs ``backend`` ∈ {auto, jax, numpy} to a concrete supported one.

    Parameters
    ----------
    requested
        The user's ``backend`` value.
    entry
        The resolved registry entry (its ``backends`` bound the choice).
    jax_ok
        Result of the single :func:`neighbayes._backends.jax_available` probe.

    Raises
    ------
    ValueError
        For an invalid value, or an explicit backend the family does not support.
    ImportError
        For an explicit ``"jax"`` request when JAX is not installed.

    Notes
    -----
    Resolving to ``"jax"`` enables JAX's float64 mode here, before the model or
    the runner builds any array: an array built while it is off is silently
    float32, and a model that caches one would carry it into every later fit.
    """
    return resolve_backend_for(requested, entry.backends, jax_ok=jax_ok)


def resolve_backend_for(requested: str, backends, *, jax_ok: bool) -> str:
    """:func:`resolve_backend` for a sampler outside the registry.

    ``backends`` is the subset of ``{"jax", "numpy"}`` the sampler supports for
    this model.  The flow models use it, since which of their configurations
    have a JAX kernel depends on the model's own options.
    """
    backends = frozenset(backends)
    valid = {"auto", "jax", "numpy"}
    if requested not in valid:
        raise ValueError(
            f"backend for sampler='gibbs' must be one of {sorted(valid)}, "
            f"got {requested!r}"
        )
    if requested == "auto":
        # JAX when the family supports it and it is importable, else NumPy.
        if "jax" in backends and jax_ok:
            return _jax_backend()
        return "numpy" if "numpy" in backends else next(iter(backends))
    if requested not in backends:
        raise ValueError(
            f"backend={requested!r} is not supported for this model "
            f"(supported: {sorted(backends)})"
        )
    if requested == "jax" and not jax_ok:
        raise ImportError("backend='jax' requires JAX. Install with: pip install jax")
    return _jax_backend() if requested == "jax" else requested


def _jax_backend() -> str:
    """``"jax"``, with JAX's float64 mode on (see :func:`resolve_backend`)."""
    from .._jax_dispatch import ensure_x64

    ensure_x64()
    return "jax"


def pop_options(sample_kwargs: dict, entry: GibbsEntry) -> dict:
    """Pop the entry's declared ``options`` from ``sample_kwargs`` (in place).

    Any remaining key is an unsupported Gibbs option and raises ``TypeError`` —
    the strict replacement for the old silent ``**kwargs`` swallow.
    """
    opts = {k: sample_kwargs.pop(k) for k in list(sample_kwargs) if k in entry.options}
    if sample_kwargs:
        raise TypeError(
            "unsupported keyword argument(s) for sampler='gibbs': "
            f"{sorted(sample_kwargs)}"
        )
    return opts


def run_entry(entry: GibbsEntry, model, *, log_likelihood: bool, **kwargs):
    """Run ``entry`` and store the pointwise log-likelihood only when requested.

    Matches PyMC, where ``idata_kwargs={"log_likelihood": True}`` opts in: the
    array holds one value per draw, chain, and observation (16 GB at
    n = 250,000 with 4 × 2,000 draws), and only LOO/WAIC read it.
    """
    if entry.skips_log_likelihood:
        kwargs["log_likelihood"] = bool(log_likelihood)
    idata = entry.run(model, **kwargs)
    if not log_likelihood and "log_likelihood" in idata.children:
        del idata["log_likelihood"]
    return idata
