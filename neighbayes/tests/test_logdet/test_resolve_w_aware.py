"""Regression tests: logdet auto-selection must see W everywhere it is resolved.

A directed (pattern-asymmetric) W must resolve to ``"lu_aaa"`` — not the
symmetric-only ``"chol_cheb"`` — in every resolution site: the model's
``_resolved_logdet_method`` and the method recorded on ``_logdet_bounds``,
for both cross-sectional and panel models.
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp


def _directed_w(n: int, seed: int = 7) -> sp.csr_matrix:
    """Row-standardized W with an asymmetric (directed) pattern: links to i+1, i+2."""
    rows, cols = [], []
    for i in range(n):
        rows += [i, i]
        cols += [(i + 1) % n, (i + 2) % n]
    A = sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))
    assert (abs(A - A.T) != 0).nnz > 0  # genuinely asymmetric pattern
    d = np.asarray(A.sum(axis=1)).ravel()
    return sp.csr_matrix(sp.diags(1.0 / d) @ A)


def test_cross_section_directed_resolves_aaa():
    from neighbayes.models import SAR

    n = 600  # above the eigenvalue cutoff (500), inside the cheb/aaa range
    rng = np.random.default_rng(0)
    W = _directed_w(n)
    X = np.column_stack([np.ones(n), rng.normal(size=n)])
    y = rng.normal(size=n)

    model = SAR(y=y, X=X, W=W)
    assert model._resolved_logdet_method == "lu_aaa"
    assert model._logdet_bounds.method == "lu_aaa"


def test_panel_directed_resolves_aaa():
    from neighbayes.models import SAR

    N, T = 600, 3
    rng = np.random.default_rng(0)
    W = _directed_w(N)
    X = np.column_stack([np.ones(N * T), rng.normal(size=N * T)])
    y = rng.normal(size=N * T)

    model = SAR(y=y, X=X, W=W, N=N, T=T)
    assert model._resolved_logdet_method == "lu_aaa"
    assert model._logdet_bounds.method == "lu_aaa"
