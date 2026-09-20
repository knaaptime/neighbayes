"""Registry mapping ``(likelihood, structure)`` → a Gibbs runner.

A plain module-level dict — no plugins, no entry points (3 structures × ~7
likelihoods is well under the point where anything fancier pays off).

Contract
--------
Each model family registers **one** :class:`GibbsEntry` whose ``run`` callable
takes the model plus the resolved sampling controls and returns a finished
:class:`arviz.InferenceData`::

    def run(model, *, draws, tune, chains, random_seed, thin, n_jobs,
            progressbar, backend, **family_opts) -> az.InferenceData

Returning a finished ``InferenceData`` (rather than raw chain results) is a
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
    import arviz as az

# A runner returns a finished InferenceData.
GibbsRunner = Callable[..., "az.InferenceData"]


@dataclass(frozen=True)
class GibbsEntry:
    """One family's Gibbs sampler, as seen by the base ``fit()``.

    Attributes
    ----------
    run
        Callable ``(model, *, draws, tune, chains, ...) -> az.InferenceData``.
    backends
        Execution backends this family supports, a subset of
        ``{"jax", "numpy"}``.  ``"auto"`` resolves against this set; every
        family supports ``"numpy"`` (JAX is the optional accelerated path).
    auto_backend
        Which backend ``gibbs_backend="auto"`` prefers.  ``"jax"`` (default)
        selects JAX when installed and supported, else NumPy — the JAX Gibbs
        path uses the sparse sparsax solve (it never densifies ``W``; a dense
        fallback runs only when sparsax is unavailable or on GPU).  ``"numpy"``
        pins ``auto`` to NumPy regardless of JAX availability — for families with
        no JAX backend, or where the CHOLMOD path is intentionally the default.
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
    auto_backend: str = "jax"
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
    auto_backend: str | None = None,
    options=(),
    supports_robust: bool = False,
    skips_log_likelihood: bool = False,
) -> GibbsEntry:
    """Register (once) the Gibbs runner for ``(likelihood, structure)``.

    ``auto_backend`` defaults to ``"jax"`` when the family supports JAX, else
    ``"numpy"``.  Pass ``auto_backend="numpy"`` explicitly for a JAX-capable
    family whose ``auto`` should still prefer NumPy (e.g. ZINB, whose JAX and
    NumPy backends fit different model specs).
    """
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
    if auto_backend is None:
        auto_backend = "jax" if "jax" in backends else "numpy"
    if auto_backend not in backends:
        raise ValueError(
            f"registry entry {key!r} auto_backend={auto_backend!r} must be one of "
            f"its backends {sorted(backends)}"
        )
    entry = GibbsEntry(
        run=run,
        backends=backends,
        auto_backend=auto_backend,
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
    """Map ``gibbs_backend`` ∈ {auto, jax, numpy} to a concrete supported backend.

    Parameters
    ----------
    requested
        The user's ``gibbs_backend`` value.
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
    """
    valid = {"auto", "jax", "numpy"}
    if requested not in valid:
        raise ValueError(
            f"gibbs_backend must be one of {sorted(valid)}, got {requested!r}"
        )
    if requested == "auto":
        # Prefer the entry's declared auto backend; only escalate to JAX when the
        # family prefers it *and* JAX is importable, else fall back to NumPy.
        if entry.auto_backend == "jax" and "jax" in entry.backends and jax_ok:
            return "jax"
        return "numpy" if "numpy" in entry.backends else next(iter(entry.backends))
    if requested not in entry.backends:
        raise ValueError(
            f"gibbs_backend={requested!r} is not supported for this model "
            f"(supported: {sorted(entry.backends)})"
        )
    if requested == "jax" and not jax_ok:
        raise ImportError(
            "gibbs_backend='jax' requires JAX. Install with: pip install jax"
        )
    return requested


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
    if not log_likelihood and "log_likelihood" in idata.groups():
        del idata["log_likelihood"]
    return idata
