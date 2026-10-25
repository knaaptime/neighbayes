"""Tests verifying that ``_logdet_bounds`` is wired through model classes."""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from neighbayes._logdet import LogdetBounds
from neighbayes.models.cross_section.sar import SAR
from neighbayes.models.cross_section.sem import SEM


def _toy_inputs(n: int = 25, seed: int = 0):
    rng = np.random.default_rng(seed)
    coords = rng.uniform(size=(n, 2))
    d = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1)
    np.fill_diagonal(d, np.inf)
    W = (1.0 / d) * (d < np.quantile(d, 0.15))
    rs = W.sum(axis=1, keepdims=True)
    rs[rs == 0] = 1.0
    W = W / rs
    X = rng.standard_normal((n, 2))
    y = rng.standard_normal(n)
    return y, X, sp.csr_matrix(W)


def test_sar_logdet_bounds_default():
    y, X, W = _toy_inputs()
    m = SAR(y=y, X=X, W=W, logdet_method="eigenvalue")
    b = m._logdet_bounds
    assert isinstance(b, LogdetBounds)
    assert b.method == "eigenvalue"
    assert b.rho_min == -1.0
    assert b.rho_max == 1.0


def test_sem_logdet_bounds_default():
    y, X, W = _toy_inputs()
    m = SEM(y=y, X=X, W=W, logdet_method="eigenvalue")
    b = m._logdet_bounds
    assert isinstance(b, LogdetBounds)
    assert b.rho_min == -1.0
    assert b.rho_max == 1.0


def test_sar_logdet_bounds_from_priors():
    y, X, W = _toy_inputs()
    m = SAR(
        y=y,
        X=X,
        W=W,
        logdet_method="eigenvalue",
        priors={"rho_lower": -0.5, "rho_upper": 0.8},
    )
    b = m._logdet_bounds
    assert b.rho_min == -0.5
    assert b.rho_max == 0.8
    assert b.source == "prior"


def test_eigenvalue_guard_large_n():
    """Forcing eigenvalue on a large W raises a clear ValueError."""
    import pytest

    from neighbayes._logdet._config import _eigen_hard_max_n

    hard_limit = _eigen_hard_max_n()
    n = hard_limit + 50  # above the hard guard
    y, X, W = _toy_inputs(n=n)
    m = SAR(y=y, X=X, W=W, logdet_method="eigenvalue")
    # Accessing _W_eigs should raise, not silently densify.
    with pytest.raises(ValueError, match="NEIGHBAYES_LOGDET_EIGEN_HARD_MAX_N"):
        _ = m._W_eigs
    # Same for the full eigendecomposition.
    with pytest.raises(ValueError, match="NEIGHBAYES_LOGDET_EIGEN_HARD_MAX_N"):
        _ = m._W_eigendecomposition
