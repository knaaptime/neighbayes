"""Shared pytest fixtures for parameter recovery tests.

Helper functions and constants live in tests/helpers.py so they can be
imported from test files directly without conftest injection.

All tests are marked slow because they run MCMC. Run them with::

    pytest -m slow
    pytest -m recovery   # parameter recovery tests only
    pytest               # skips slow by default (see pyproject.toml)
"""

from __future__ import annotations

import sys

import numpy as np
import pytest
from libpysal.graph import Graph

from .helpers import PANEL_N, W_to_graph, make_line_W, make_rook_W

SIDE = 15  # 225 cross-sectional units


# ---------------------------------------------------------------------------
# Pytest collection hooks
# ---------------------------------------------------------------------------


def pytest_collection_modifyitems(config, items):
    """Auto-skip ``requires_jax``-marked tests on Windows.

    JAX (jaxlib) and the JAX-backed samplers (``backend="jax"``,
    ``numpyro``, ``blackjax``) are not reliably installable on Windows CI.
    Tests that activate those backends are tagged with
    ``@pytest.mark.requires_jax`` and skipped here so that cross-platform CI
    matrices don't need per-job marker exclusions.
    """
    if not sys.platform.startswith("win"):
        return
    skip_marker = pytest.mark.skip(
        reason="JAX backends (numpyro/blackjax) are not supported on Windows CI"
    )
    for item in items:
        if "requires_jax" in item.keywords:
            item.add_marker(skip_marker)


# ---------------------------------------------------------------------------
# Pytest fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(0)


@pytest.fixture(scope="session")
def W_dense() -> np.ndarray:
    return make_rook_W(SIDE)


@pytest.fixture(scope="session")
def W_graph(W_dense: np.ndarray) -> Graph:
    return W_to_graph(W_dense)


@pytest.fixture(scope="session")
def W_panel_dense() -> np.ndarray:
    return make_line_W(PANEL_N)


@pytest.fixture(scope="session")
def W_panel_graph(W_panel_dense: np.ndarray) -> Graph:
    return W_to_graph(W_panel_dense)
