"""Fixtures for the LeSage-toolbox validation tests (pytest marker ``octave``).

These tests are deselected by default (``addopts`` skips ``octave``); the
bespoke Actions workflow runs ``pytest -m octave`` after installing Octave
and fetching the toolbox at its pinned URL.
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp

from neighbayes.dgp import simulate_sar, simulate_sem
from neighbayes.tests.helpers import make_rook_W
from neighbayes.tests.test_validation import _octave

# Dataset construction
RHO_TRUE = 0.6
LAM_TRUE = 0.5
BETA_TRUE = np.array([1.0, 1.5])
SIGMA_TRUE = 1.0

# Toolbox chain lengths; neighbayes fits the same length per chain x 4 chains
TB_DRAWS = 3000
TB_OMIT = 1000


@pytest.fixture(scope="session")
def octave():
    """Skip unless an Octave executable is available."""
    if _octave.octave_binary() is None:
        pytest.skip("Octave not found (PATH or NEIGHBAYES_OCTAVE)")


@pytest.fixture(scope="session")
def toolbox():
    """Skip unless the LeSage toolbox is available."""
    path = _octave.toolbox_dir()
    if path is None:
        pytest.skip(
            "LeSage toolbox not found (NEIGHBAYES_TOOLBOX_DIR or "
            "../reference/toolbox_may2021)"
        )
    return path


def _rook_dataset(simulator, spatial_true, seed):
    """One dataset, one row-standardized W, shared verbatim by both sides."""
    W_std = make_rook_W(9)  # n = 81
    data = simulator(
        W=sp.csr_matrix(W_std),
        **spatial_true,
        beta=BETA_TRUE,
        sigma=SIGMA_TRUE,
        seed=seed,
    )
    return {
        "y": np.asarray(data["y"]).ravel(),
        "X": np.asarray(data["X"]),
        "W": W_std,
        "params_true": data["params_true"],
    }


@pytest.fixture(scope="module")
def sar_dataset():
    return _rook_dataset(simulate_sar, {"rho": RHO_TRUE}, seed=20261018)


@pytest.fixture(scope="module")
def sem_dataset():
    return _rook_dataset(simulate_sem, {"lam": LAM_TRUE}, seed=20261019)


@pytest.fixture
def octave_run(tmp_path, octave, toolbox):
    """Write driver text (and optionally the dataset), run Octave in tmp_path."""

    def run(script_text: str, dataset: dict | None = None, prefix: str | None = None):
        if dataset is not None:
            _octave.write_ascii(dataset, tmp_path)
        _octave.write_shims(tmp_path)
        script = tmp_path / "driver.m"
        script.write_text(script_text)
        stdout = _octave.run_octave(script, tmp_path)
        draws = None
        if prefix is not None:
            draws = _octave.load_draws(tmp_path, prefix)
        return stdout, draws

    return run
