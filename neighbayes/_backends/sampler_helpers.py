"""Helpers for tweaking ``pm.sample`` keyword arguments per backend.

PyMC supports several NUTS samplers via the ``nuts_sampler`` keyword:

* ``"pymc"``    — built-in C/PyTensor implementation (always available; default).
* ``"blackjax"`` — JAX-backed sampler.
* ``"numpyro"`` — also JAX-backed; uses NumPyro's NUTS.
* ``"nutpie"``  — Rust-backed.

This module provides three small helpers used by ``fit()`` methods:

* :func:`enforce_c_backend` — for models whose custom :class:`pytensor.graph.op.Op`
  has no JAX dispatch (e.g. NB sparse-flow models that wrap
  :class:`scipy.sparse.linalg.splu`), downgrade a JAX-backed request to
  ``"pymc"`` with a one-time ``UserWarning``.
* :func:`prepare_idata_kwargs` — strip ``log_likelihood=True`` for JAX backends
  on potential-only models where PyMC's JAX path would crash.
* :func:`prepare_compile_kwargs` — auto-inject ``compile_kwargs={"mode": "NUMBA"}``
  when the resolved sampler is ``"pymc"`` and ``numba`` is importable.
"""

from __future__ import annotations

import importlib.util
import warnings
from functools import lru_cache


@lru_cache(maxsize=None)
def _has_module(name: str) -> bool:
    """Return ``True`` if ``name`` is importable without importing it."""
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


#: What can execute each sampler, its default first.
GIBBS_BACKENDS = ("auto", "jax", "numpy")
NUTS_BACKENDS = ("pymc", "nutpie", "numpyro", "blackjax")


def select_backend(sampler: str, backend: str | None, sample_kwargs: dict) -> str:
    """The executor that ``fit(sampler=..., backend=...)`` names.

    A Gibbs sampler runs on ``"auto"`` (the default: JAX when it is installed
    and the model has a JAX kernel, else NumPy), ``"jax"`` or ``"numpy"``;
    NUTS on ``"pymc"`` (the default), ``"nutpie"``, ``"numpyro"`` or
    ``"blackjax"``.  ``None`` takes the sampler's default.
    """
    retired = {"gibbs_backend", "nuts_sampler"} & sample_kwargs.keys()
    if retired:
        raise TypeError(f"{min(retired)}= is replaced by backend=.")
    if sampler not in ("gibbs", "nuts"):
        raise ValueError(f"sampler must be 'gibbs', 'nuts', or None, got {sampler!r}")
    valid = GIBBS_BACKENDS if sampler == "gibbs" else NUTS_BACKENDS
    if backend is None:
        return valid[0]
    if backend not in valid:
        raise ValueError(
            f"backend for sampler={sampler!r} must be one of {valid}, got {backend!r}."
        )
    return backend


def jax_available() -> bool:
    """Return ``True`` when optional ``jax`` is importable.

    Single probe used by every ``fit`` path that resolves a ``"jax"``/
    ``"jax_dense"`` request against JAX availability, so the check lives in
    exactly one place (results are cached via :func:`_has_module`).
    """
    return _has_module("jax")


@lru_cache(maxsize=1)
def _jax_dispatches_available() -> bool:
    """Return ``True`` if both JAX and PyTensor's JAX dispatch are present.

    When this is true, the custom Ops in :mod:`neighbayes._ops` register their
    own ``jax_funcify`` implementations on import, so models that previously
    required the C backend can sample under ``"blackjax"`` or ``"numpyro"``.
    """
    return _has_module("jax") and _has_module("pytensor.link.jax.dispatch")


def use_jax_likelihood(nuts_sampler: str) -> bool:
    """Return ``True`` when the resolved sampler is JAX-backed.

    Models that define their likelihood via :func:`pymc.Potential` cannot be
    captured by PyMC's JAX log-likelihood path (it iterates ``observed_RVs``
    only).  When this helper returns ``True``, ``_build_pymc_model`` should
    use a :class:`pymc.CustomDist` with an observed RV instead, so PyMC can
    capture ``log_likelihood`` natively.
    """
    return nuts_sampler in ("blackjax", "numpyro")


def enforce_c_backend(
    nuts_sampler: str,
    *,
    requires_c_backend: bool,
    model_name: str,
) -> str:
    """Downgrade a JAX-backed NUTS request to ``"pymc"`` when JAX dispatch is missing.

    Parameters
    ----------
    nuts_sampler :
        The user-requested ``nuts_sampler`` value (already resolved from
        ``sample_kwargs``).
    requires_c_backend :
        If ``True``, the calling model relies on a custom :class:`pytensor.graph.op.Op`
        that has no JAX dispatch.  Any non-``"pymc"`` request is downgraded to
        ``"pymc"`` with a one-time ``UserWarning``.
    model_name :
        Class name used in the warning message.

    Returns
    -------
    str
        Either the original ``nuts_sampler`` value or ``"pymc"`` if a downgrade
        was forced.
    """
    if not requires_c_backend:
        return nuts_sampler
    if nuts_sampler == "pymc":
        return nuts_sampler
    if _jax_dispatches_available():
        return nuts_sampler
    _warn_once(
        ("c_backend", nuts_sampler, model_name),
        f"backend={nuts_sampler!r} requested but {model_name} uses a custom "
        "PyTensor Op without a JAX dispatch; falling back to "
        "PyMC's default NUTS sampler.",
    )
    return "pymc"


def prepare_idata_kwargs(
    idata_kwargs: dict | None,
    model,
    nuts_sampler: str,
) -> dict:
    """Normalize ``idata_kwargs`` and handle JAX ``log_likelihood`` policy.

    PyMC's JAX sampling path (``pm.sampling.jax._get_log_likelihood``) iterates
    ``model.observed_RVs``; when a model defines its likelihood purely with
    ``pm.Potential`` and no observed RV, that list is empty and the helper
    raises ``TypeError: 'NoneType' object is not iterable``.

    To make model-comparison diagnostics (BIC/Bayes factors/LOO/WAIC) work out
    of the box, JAX samplers default to requesting pointwise log-likelihood
    unless the caller explicitly sets ``log_likelihood=False``. For potential-
    only models, the request is then stripped as a defensive fallback to avoid
    the PyMC JAX crash described above.

    Most spatial-error models (SEM, SDEM, all panel SEM/SDEM variants,
    SEMTobitPanel) now expose a dual-path ``_build_pymc_model``: when the
    selected sampler is ``"blackjax"`` or ``"numpyro"`` they register the
    likelihood via ``pm.CustomDist("obs", ..., observed=y)`` so PyMC captures
    ``log_likelihood["obs"]`` natively. Flow models (SEMFlow, SEMFlowPanelFE,
    SEMFlowSeparablePanelFE) achieve the same result via the mean-form
    ``pm.Normal/StudentT("obs", mu=..., observed=y)`` pattern combined with a
    Jacobian ``pm.Potential`` that is folded back into ``log_likelihood["obs"]``
    after sampling.

    For these migrated models ``observed_RVs`` is non-empty under the JAX
    backend, so this helper is a no-op. The strip remains as a defensive
    fallback for any not-yet-migrated potential-only model that would
    otherwise crash the JAX sampler (those models recompute the log-likelihood
    manually in their own ``fit`` override).
    """
    idata_kwargs = dict(idata_kwargs or {})
    if not idata_kwargs.get("log_likelihood"):
        return idata_kwargs
    if nuts_sampler not in ("blackjax", "numpyro"):
        return idata_kwargs
    if getattr(model, "observed_RVs", None):
        return idata_kwargs
    idata_kwargs.pop("log_likelihood", None)
    return idata_kwargs


def prepare_compile_kwargs(
    sample_kwargs: dict | None,
    nuts_sampler: str,
) -> dict:
    """Inject ``compile_kwargs={"mode": "NUMBA"}`` for the PyMC sampler.

    Numba is a soft dependency; when it is importable the C/PyTensor
    backend used by ``nuts_sampler="pymc"`` is materially faster under
    the NUMBA mode.  This helper sets that compile mode by default while
    leaving JAX-backed (``"blackjax"``, ``"numpyro"``) and Rust-backed
    (``"nutpie"``) samplers untouched — they ignore ``compile_kwargs``.

    For JAX-backed samplers (``"blackjax"``, ``"numpyro"``), this helper
    also defaults ``nuts_sampler_kwargs["chain_method"] = "vectorized"``
    so that multiple chains run in a single XLA-compiled batched call
    rather than serially.  Caller overrides (including an explicit
    ``None`` or any other ``chain_method`` value) are preserved.

    Behaviour:

    * ``"pymc"`` sampler:
        - ``"compile_kwargs"`` already present → unchanged.
        - ``numba`` importable → adds ``compile_kwargs={"mode": "NUMBA"}``.
        - ``numba`` missing → unchanged + one-time ``UserWarning``.
    * ``"blackjax"`` / ``"numpyro"`` sampler:
        - ``nuts_sampler_kwargs["chain_method"]`` already present → unchanged.
        - Otherwise → inserts ``"chain_method": "vectorized"``.
    * Any other sampler → returns ``sample_kwargs`` unchanged.

    Parameters
    ----------
    sample_kwargs :
        The keyword-argument dict eventually splatted into ``pm.sample``.
        ``None`` is treated as an empty dict.
    nuts_sampler :
        The resolved sampler name.

    Returns
    -------
    dict
        A new dict that may have ``compile_kwargs`` and/or
        ``nuts_sampler_kwargs`` added; never mutates the input.
    """
    out = dict(sample_kwargs or {})
    if nuts_sampler in ("blackjax", "numpyro"):
        nsk = dict(out.get("nuts_sampler_kwargs") or {})
        if "chain_method" not in nsk:
            nsk["chain_method"] = "vectorized"
            out["nuts_sampler_kwargs"] = nsk
        return out
    if nuts_sampler != "pymc":
        return out
    if "compile_kwargs" in out:
        return out
    if not _has_module("numba"):
        _warn_once(
            ("numba_missing",),
            "numba is not installed; the PyMC NUTS sampler will use PyTensor's "
            "default C compile mode.  Install 'numba' to enable the faster "
            "NUMBA backend.",
        )
        return out
    out["compile_kwargs"] = {"mode": "NUMBA"}
    return out


@lru_cache(maxsize=None)
def _warn_once(key: tuple, message: str) -> None:
    del key
    warnings.warn(message, UserWarning, stacklevel=3)
