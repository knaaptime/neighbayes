"""Unit tests for the Gibbs sampler registry primitives (Phase 5b.0).

These exercise the registry contract in isolation — no model wiring — so the
dispatch semantics (backend resolution, strict option handling, duplicate
guards) are pinned before any family is migrated onto it.
"""

from __future__ import annotations

import pytest

from neighbayes.samplers import _registry as R


@pytest.fixture
def fresh_registry(monkeypatch):
    """Give each test an isolated empty registry dict."""
    monkeypatch.setattr(R, "_REGISTRY", {})
    return R


def _dummy_run(model, **kw):  # pragma: no cover - never actually called
    return "idata"


def test_register_and_resolve(fresh_registry):
    entry = fresh_registry.register(
        "gaussian", "cross_section", run=_dummy_run, backends={"jax", "numpy"}
    )
    assert fresh_registry.resolve("gaussian", "cross_section") is entry
    assert fresh_registry.resolve("gaussian", "panel_fe") is None


def test_register_rejects_duplicate(fresh_registry):
    fresh_registry.register("g", "cs", run=_dummy_run, backends={"numpy"})
    with pytest.raises(ValueError, match="duplicate"):
        fresh_registry.register("g", "cs", run=_dummy_run, backends={"numpy"})


def test_register_validates_backends(fresh_registry):
    with pytest.raises(ValueError, match="at least one backend"):
        fresh_registry.register("g", "cs", run=_dummy_run, backends=set())
    with pytest.raises(ValueError, match="subset"):
        fresh_registry.register("g", "cs", run=_dummy_run, backends={"cuda"})


def test_resolve_backend_auto_prefers_jax_when_available(fresh_registry):
    entry = fresh_registry.register(
        "g", "cs", run=_dummy_run, backends={"jax", "numpy"}
    )
    assert R.resolve_backend("auto", entry, jax_ok=True) == "jax"
    assert R.resolve_backend("auto", entry, jax_ok=False) == "numpy"


def test_resolve_backend_auto_numpy_only_family(fresh_registry):
    entry = fresh_registry.register("zinb", "cs", run=_dummy_run, backends={"numpy"})
    # Even with JAX available, a numpy-only family resolves to numpy.
    assert R.resolve_backend("auto", entry, jax_ok=True) == "numpy"


def test_resolve_backend_auto_is_numpy_for_a_numpy_only_family(fresh_registry):
    entry = fresh_registry.register("re", "panel", run=_dummy_run, backends={"numpy"})
    assert R.resolve_backend("auto", entry, jax_ok=True) == "numpy"


def test_resolve_backend_explicit_unsupported_raises(fresh_registry):
    entry = fresh_registry.register("zinb", "cs", run=_dummy_run, backends={"numpy"})
    with pytest.raises(ValueError, match="not supported"):
        R.resolve_backend("jax", entry, jax_ok=True)


def test_resolve_backend_explicit_jax_without_jax_raises(fresh_registry):
    entry = fresh_registry.register(
        "g", "cs", run=_dummy_run, backends={"jax", "numpy"}
    )
    with pytest.raises(ImportError, match="requires JAX"):
        R.resolve_backend("jax", entry, jax_ok=False)


def test_resolve_backend_invalid_value(fresh_registry):
    entry = fresh_registry.register("g", "cs", run=_dummy_run, backends={"numpy"})
    with pytest.raises(ValueError, match="backend for sampler='gibbs' must be one of"):
        R.resolve_backend("factorize", entry, jax_ok=True)


def test_pop_options_extracts_declared_and_rejects_unknown(fresh_registry):
    entry = fresh_registry.register(
        "g",
        "cs",
        run=_dummy_run,
        backends={"numpy"},
        options={"slice_width", "init_jitter"},
    )
    sk = {"slice_width": 0.2, "init_jitter": 0.1}
    opts = R.pop_options(sk, entry)
    assert opts == {"slice_width": 0.2, "init_jitter": 0.1}
    assert sk == {}  # all consumed


def test_pop_options_rejects_leftover(fresh_registry):
    entry = fresh_registry.register(
        "g", "cs", run=_dummy_run, backends={"numpy"}, options={"slice_width"}
    )
    sk = {"slice_width": 0.2, "target_accept": 0.9}
    with pytest.raises(TypeError, match="unsupported keyword"):
        R.pop_options(sk, entry)
