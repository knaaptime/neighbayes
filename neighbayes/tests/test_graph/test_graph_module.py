"""Tests for neighbayes.graph — Kronecker-product weight utilities."""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp
from libpysal.graph import Graph

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_ring_graph(n: int) -> Graph:
    """Build a simple ring-contiguity Graph for n units (row-standardized)."""
    focal = np.concatenate([np.arange(n), np.arange(n)])
    neighbor = np.concatenate([np.roll(np.arange(n), 1), np.roll(np.arange(n), -1)])
    weight = np.ones(len(focal), dtype=float)
    G = Graph.from_arrays(focal, neighbor, weight)
    return G.transform("r")


# ---------------------------------------------------------------------------
# _graph_to_csr
# ---------------------------------------------------------------------------


class TestValidateGraph:
    def test_returns_csr(self):
        from neighbayes.graph import _graph_to_csr

        G = _make_ring_graph(5)
        W = _graph_to_csr(G)
        assert sp.issparse(W)
        assert W.format == "csr"
        assert W.shape == (5, 5)

    def test_rejects_non_graph(self):
        from neighbayes.graph import _graph_to_csr

        with pytest.raises(TypeError, match="libpysal.graph.Graph"):
            _graph_to_csr(np.eye(4))

    def test_warns_if_not_row_standardized(self):
        from neighbayes.models._base._shared import resolve_W

        # Build a non-row-standardized graph by using "b" transformation
        G = _make_ring_graph(5).transform("b")
        with pytest.warns(UserWarning, match="row-standardized"):
            resolve_W(G, n=5)


# ---------------------------------------------------------------------------
# destination_weights / origin_weights / od_weights
# ---------------------------------------------------------------------------


class TestWeightMatrixShapes:
    @pytest.mark.parametrize("n", [4, 7])
    def test_destination_shape(self, n):
        from neighbayes.graph import destination_weights

        G = _make_ring_graph(n)
        Wd = destination_weights(G)
        assert Wd.shape == (n * n, n * n)
        assert sp.issparse(Wd)

    @pytest.mark.parametrize("n", [4, 7])
    def test_origin_shape(self, n):
        from neighbayes.graph import origin_weights

        G = _make_ring_graph(n)
        Wo = origin_weights(G)
        assert Wo.shape == (n * n, n * n)

    @pytest.mark.parametrize("n", [4, 7])
    def test_network_shape(self, n):
        from neighbayes.graph import od_weights

        G = _make_ring_graph(n)
        Ww = od_weights(G)
        assert Ww.shape == (n * n, n * n)

    def test_destination_block_diagonal(self):
        """W_d = I_n ⊗ W should be block-diagonal with n copies of W."""
        from neighbayes.graph import _graph_to_csr, destination_weights

        n = 4
        G = _make_ring_graph(n)
        W = _graph_to_csr(G)
        Wd = destination_weights(G).toarray()
        W_arr = W.toarray()

        # Check each diagonal block
        for i in range(n):
            block = Wd[i * n : (i + 1) * n, i * n : (i + 1) * n]
            np.testing.assert_allclose(block, W_arr, atol=1e-12)

    def test_network_equals_kron_W_W(self):
        from neighbayes.graph import _graph_to_csr, od_weights

        n = 4
        G = _make_ring_graph(n)
        W = _graph_to_csr(G)
        Ww = od_weights(G).toarray()
        expected = sp.kron(W, W, format="csr").toarray()
        np.testing.assert_allclose(Ww, expected, atol=1e-12)


class TestWeightMatrixKronecker:
    def test_network_equals_dest_at_origin(self):
        """W_w = W_d @ W_o / (row normalization) for row-stochastic W."""
        from neighbayes.graph import (
            destination_weights,
            od_weights,
            origin_weights,
        )

        n = 5
        G = _make_ring_graph(n)
        Wd = destination_weights(G)
        Wo = origin_weights(G)
        Ww = od_weights(G)
        # For row-stochastic W: (I ⊗ W)(W ⊗ I) = W ⊗ W
        product = Wd @ Wo
        np.testing.assert_allclose(Ww.toarray(), product.toarray(), atol=1e-12)

    def test_row_sums_of_dest(self):
        """Wd inherits row-stochastic property of W in each block."""
        from neighbayes.graph import destination_weights

        n = 5
        G = _make_ring_graph(n)
        Wd = destination_weights(G)
        row_sums = np.asarray(Wd.sum(axis=1)).ravel()
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-12)

    def test_row_sums_of_origin(self):
        from neighbayes.graph import origin_weights

        n = 5
        G = _make_ring_graph(n)
        Wo = origin_weights(G)
        row_sums = np.asarray(Wo.sum(axis=1)).ravel()
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-12)


class TestFlowWeightMatrices:
    def test_returns_three_keys(self):
        from neighbayes.graph import flow_weight_matrices

        G = _make_ring_graph(5)
        wms = flow_weight_matrices(G)
        assert set(wms.keys()) == {"destination", "origin", "od"}

    def test_consistency_with_individual_functions(self):
        from neighbayes.graph import (
            destination_weights,
            flow_weight_matrices,
            od_weights,
            origin_weights,
        )

        n = 5
        G = _make_ring_graph(n)
        wms = flow_weight_matrices(G)
        np.testing.assert_allclose(
            wms["destination"].toarray(), destination_weights(G).toarray(), atol=1e-12
        )
        np.testing.assert_allclose(
            wms["origin"].toarray(), origin_weights(G).toarray(), atol=1e-12
        )
        np.testing.assert_allclose(
            wms["od"].toarray(), od_weights(G).toarray(), atol=1e-12
        )


# ---------------------------------------------------------------------------
# flow_design_matrix / FlowDesignMatrix
# ---------------------------------------------------------------------------


class TestFlowDesignMatrix:
    def setup_method(self):
        self.n = 5
        self.k = 2
        self.N = self.n * self.n
        rng = np.random.default_rng(42)
        self.X = rng.standard_normal((self.n, self.k))

    def test_shapes(self):
        from neighbayes.graph import flow_design_matrix

        dm = flow_design_matrix(self.X)
        assert dm.X_dest.shape == (self.N, self.k)
        assert dm.X_orig.shape == (self.N, self.k)
        assert dm.X_intra.shape == (self.N, self.k)
        assert dm.intra_indicator.shape == (self.N,)
        assert dm.combined.shape[0] == self.N

    def test_combined_columns_without_dist(self):
        from neighbayes.graph import flow_design_matrix

        dm = flow_design_matrix(self.X)
        # Expect: intercept(1) + ia(1) + dest(k) + orig(k) + intra(k)
        expected_cols = 1 + 1 + self.k + self.k + self.k
        assert dm.combined.shape[1] == expected_cols

    def test_combined_columns_with_dist(self):
        from neighbayes.graph import flow_design_matrix

        dist = np.abs(np.random.default_rng(0).standard_normal((self.n, self.n)))
        dm = flow_design_matrix(self.X, dist=dist, log_distance=False)
        expected_cols = 1 + 1 + self.k + self.k + self.k + 1
        assert dm.combined.shape[1] == expected_cols

    def test_dist_vec_populated(self):
        from neighbayes.graph import flow_design_matrix

        dist = np.arange(self.n * self.n, dtype=float).reshape(self.n, self.n)
        dm = flow_design_matrix(self.X, dist=dist, log_distance=False)
        assert dm.dist_vec is not None
        np.testing.assert_allclose(dm.dist_vec, dist.ravel(), atol=1e-12)

    def test_dist_vec_none_by_default(self):
        from neighbayes.graph import flow_design_matrix

        dm = flow_design_matrix(self.X)
        assert dm.dist_vec is None

    def test_intercept_column_all_ones(self):
        from neighbayes.graph import flow_design_matrix

        dm = flow_design_matrix(self.X)
        np.testing.assert_allclose(dm.combined[:, 0], 1.0, atol=1e-12)

    def test_intra_indicator_is_binary(self):
        from neighbayes.graph import flow_design_matrix

        dm = flow_design_matrix(self.X)
        assert set(np.unique(dm.combined[:, 1]).tolist()).issubset({0.0, 1.0})

    def test_intra_indicator_count(self):
        """Number of intra-zonal cells should be n."""
        from neighbayes.graph import flow_design_matrix

        dm = flow_design_matrix(self.X)
        assert int(dm.combined[:, 1].sum()) == self.n

    def test_feature_names_length(self):
        from neighbayes.graph import flow_design_matrix

        dm = flow_design_matrix(self.X, col_names=["pop", "income"])
        expected = 1 + 1 + self.k + self.k + self.k
        assert len(dm.feature_names) == expected

    def test_dest_block_kron_structure(self):
        """X_dest[i*n+j] should equal X[j, :] for all i, j."""
        from neighbayes.graph import flow_design_matrix

        n = self.n
        dm = flow_design_matrix(self.X)
        for i in range(n):
            for j in range(n):
                np.testing.assert_allclose(dm.X_dest[i * n + j], self.X[j], atol=1e-12)

    def test_orig_block_kron_structure(self):
        """X_orig[i*n+j] should equal X[i, :] for all i, j."""
        from neighbayes.graph import flow_design_matrix

        n = self.n
        dm = flow_design_matrix(self.X)
        for i in range(n):
            for j in range(n):
                np.testing.assert_allclose(dm.X_orig[i * n + j], self.X[i], atol=1e-12)

    def test_intra_block_zero_off_diagonal(self):
        """X_intra should be zero for non-intra-zonal O-D pairs."""
        from neighbayes.graph import flow_design_matrix

        n = self.n
        dm = flow_design_matrix(self.X)
        for i in range(n):
            for j in range(n):
                if i != j:
                    np.testing.assert_allclose(dm.X_intra[i * n + j], 0.0, atol=1e-12)

    def test_wrong_dist_shape_raises(self):
        from neighbayes.graph import flow_design_matrix

        with pytest.raises(ValueError, match="dist must have shape"):
            flow_design_matrix(self.X, dist=np.ones((self.n + 1, self.n + 1)))
