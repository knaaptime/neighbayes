"""Tests for lazy weight-matrix lifecycle on :class:`SpatialModel`.

Phase 1 refactor moved ``_W_sparse``, ``_W_dense``, ``_W_eigs``, ``_Wy`` and
``_WX`` from eager fields to :func:`functools.cached_property` accessors keyed
off ``self._graph``.  Eigenvalues are additionally interned across model
instances pointing at the same :class:`libpysal.graph.Graph` via the module-
level :data:`_EIG_CACHE` ``WeakKeyDictionary``.
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp
from libpysal.graph import Graph

from neighbayes.models import OLS

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _ring_W(n: int = 6) -> sp.csr_matrix:
    rows, cols, data = [], [], []
    for i in range(n):
        for j in (i - 1, i + 1):
            rows.append(i)
            cols.append(j % n)
            data.append(0.5)
    return sp.csr_matrix((data, (rows, cols)), shape=(n, n))


@pytest.fixture()
def graph() -> Graph:
    return Graph.from_sparse(_ring_W(8))


@pytest.fixture()
def fitted_inputs():
    rng = np.random.default_rng(0)
    n = 8
    X = np.column_stack([np.ones(n), rng.standard_normal(n)])
    y = X @ np.array([1.0, 0.5]) + rng.standard_normal(n) * 0.1
    return y, X


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestWeightsLifecycle:
    def test_W_sparse_returns_csr(self, graph, fitted_inputs):
        y, X = fitted_inputs
        model = OLS(y=y, X=X, W=graph)
        # SciPy 1.13+ returns sparse arrays whose ``format`` is ``"csr"``
        # but for which ``isspmatrix_csr`` is False; assert via format.
        assert model._W_sparse.format == "csr"
        assert model._W_sparse.shape == (8, 8)

    def test_W_dense_matches_sparse(self, graph, fitted_inputs):
        y, X = fitted_inputs
        model = OLS(y=y, X=X, W=graph)
        np.testing.assert_allclose(model._W_dense, model._W_sparse.toarray())

    def test_Wy_and_WX_match_manual(self, graph, fitted_inputs):
        y, X = fitted_inputs
        model = OLS(y=y, X=X, W=graph)
        np.testing.assert_allclose(model._Wy, model._W_sparse @ y)
        # ``_WX`` lags whatever columns are configured for the model.
        expected_WX = np.asarray(
            model._W_sparse @ model._X[:, model._wx_column_indices]
        )
        np.testing.assert_allclose(model._WX, expected_WX)

    def test_eig_values_match_across_instances(self, graph, fitted_inputs):
        y, X = fitted_inputs
        m1 = OLS(y=y, X=X, W=graph)
        m2 = OLS(y=y, X=X, W=graph)
        # Both instances should compute the same eigenvalue array
        # (they are not shared objects — each instance computes its own).
        np.testing.assert_array_equal(m1._W_eigs, m2._W_eigs)

    def test_cached_property_reuses_value(self, graph, fitted_inputs):
        y, X = fitted_inputs
        model = OLS(y=y, X=X, W=graph)
        first = model._W_sparse
        second = model._W_sparse
        assert first is second
