"""Tests for asymmetric k_d ≠ k_o support in flow design matrices, DGP, and models."""

from __future__ import annotations

import numpy as np
import pytest
from libpysal.graph import Graph


def _make_ring_graph(n: int) -> Graph:
    """Build a simple ring-contiguity Graph for n units (row-standardized)."""
    focal = np.concatenate([np.arange(n), np.arange(n)])
    neighbor = np.concatenate([np.roll(np.arange(n), 1), np.roll(np.arange(n), -1)])
    weight = np.ones(len(focal), dtype=float)
    G = Graph.from_arrays(focal, neighbor, weight)
    return G.transform("r")


# ---------------------------------------------------------------------------
# FlowDesignMatrix asymmetric tests
# ---------------------------------------------------------------------------


class TestFlowDesignMatrixAsymmetric:
    """Tests for flow_design_matrix_asymmetric and k_d ≠ k_o support."""

    def test_asymmetric_shapes(self):
        """flow_design_matrix_asymmetric produces correct shapes when k_d ≠ k_o."""
        from neighbayes.graph import flow_design_matrix_asymmetric

        n = 4
        k_d, k_o = 2, 1
        rng = np.random.default_rng(42)
        Xd = rng.standard_normal((n, k_d))
        Xo = rng.standard_normal((n, k_o))
        N = n * n

        dm = flow_design_matrix_asymmetric(Xd, Xo)
        assert dm.X_dest.shape == (N, k_d)
        assert dm.X_orig.shape == (N, k_o)
        assert dm.X_intra.shape == (N, k_d)  # intra uses dest columns
        assert dm.k_d == k_d
        assert dm.k_o == k_o
        assert dm.k == k_d  # backward compat alias

    def test_asymmetric_combined_columns(self):
        """Combined design matrix has correct number of columns when k_d ≠ k_o."""
        from neighbayes.graph import flow_design_matrix_asymmetric

        n = 4
        k_d, k_o = 3, 1
        rng = np.random.default_rng(42)
        Xd = rng.standard_normal((n, k_d))
        Xo = rng.standard_normal((n, k_o))

        dm = flow_design_matrix_asymmetric(Xd, Xo)
        # intercept(1) + ia(1) + dest(k_d) + orig(k_o) + intra(k_d)
        expected_cols = 1 + 1 + k_d + k_o + k_d
        assert dm.combined.shape[1] == expected_cols

    def test_asymmetric_combined_with_dist(self):
        """Combined design matrix with distance column when k_d ≠ k_o."""
        from neighbayes.graph import flow_design_matrix_asymmetric

        n = 4
        k_d, k_o = 2, 1
        rng = np.random.default_rng(42)
        Xd = rng.standard_normal((n, k_d))
        Xo = rng.standard_normal((n, k_o))
        dist = np.abs(rng.standard_normal((n, n)))

        dm = flow_design_matrix_asymmetric(Xd, Xo, dist=dist)
        expected_cols = 1 + 1 + k_d + k_o + k_d + 1  # +1 for dist
        assert dm.combined.shape[1] == expected_cols

    def test_asymmetric_feature_names(self):
        """Feature names use y-prefix for origin columns when k_d ≠ k_o."""
        from neighbayes.graph import flow_design_matrix_asymmetric

        n = 4
        k_d, k_o = 2, 1
        rng = np.random.default_rng(42)
        Xd = rng.standard_normal((n, k_d))
        Xo = rng.standard_normal((n, k_o))

        dm = flow_design_matrix_asymmetric(Xd, Xo)
        assert dm.feature_names[0] == "intercept"
        assert dm.feature_names[1] == "intra_indicator"
        assert dm.feature_names[2] == "dest_x0"
        assert dm.feature_names[3] == "dest_x1"
        assert dm.feature_names[4] == "orig_y0"
        assert dm.feature_names[5] == "intra_x0"
        assert dm.feature_names[6] == "intra_x1"

    def test_asymmetric_dest_block_kron(self):
        """X_dest[i*n+j] should equal Xd[j, :] for all i, j."""
        from neighbayes.graph import flow_design_matrix_asymmetric

        n = 4
        k_d, k_o = 2, 1
        rng = np.random.default_rng(42)
        Xd = rng.standard_normal((n, k_d))
        Xo = rng.standard_normal((n, k_o))

        dm = flow_design_matrix_asymmetric(Xd, Xo)
        for i in range(n):
            for j in range(n):
                np.testing.assert_allclose(dm.X_dest[i * n + j], Xd[j], atol=1e-12)

    def test_asymmetric_orig_block_kron(self):
        """X_orig[i*n+j] should equal Xo[i, :] for all i, j."""
        from neighbayes.graph import flow_design_matrix_asymmetric

        n = 4
        k_d, k_o = 2, 1
        rng = np.random.default_rng(42)
        Xd = rng.standard_normal((n, k_d))
        Xo = rng.standard_normal((n, k_o))

        dm = flow_design_matrix_asymmetric(Xd, Xo)
        for i in range(n):
            for j in range(n):
                np.testing.assert_allclose(dm.X_orig[i * n + j], Xo[i], atol=1e-12)

    def test_asymmetric_intra_block(self):
        """X_intra should be zero for off-diagonal O-D pairs."""
        from neighbayes.graph import flow_design_matrix_asymmetric

        n = 4
        k_d, k_o = 2, 1
        rng = np.random.default_rng(42)
        Xd = rng.standard_normal((n, k_d))
        Xo = rng.standard_normal((n, k_o))

        dm = flow_design_matrix_asymmetric(Xd, Xo)
        for i in range(n):
            for j in range(n):
                if i != j:
                    np.testing.assert_allclose(dm.X_intra[i * n + j], 0.0, atol=1e-12)

    def test_with_orig_symmetric_delegates(self):
        """flow_design_matrix_with_orig delegates to flow_design_matrix when k_d == k_o."""
        from neighbayes.graph import flow_design_matrix_with_orig

        n = 4
        k = 2
        rng = np.random.default_rng(42)
        Xd = rng.standard_normal((n, k))
        Xo = rng.standard_normal((n, k))

        dm = flow_design_matrix_with_orig(Xd, Xo)
        assert dm.k_d == k
        assert dm.k_o == k
        assert dm.k == k

    def test_with_orig_asymmetric_delegates(self):
        """flow_design_matrix_with_orig delegates to asymmetric when k_d ≠ k_o."""
        from neighbayes.graph import flow_design_matrix_with_orig

        n = 4
        k_d, k_o = 2, 1
        rng = np.random.default_rng(42)
        Xd = rng.standard_normal((n, k_d))
        Xo = rng.standard_normal((n, k_o))

        dm = flow_design_matrix_with_orig(Xd, Xo)
        assert dm.k_d == k_d
        assert dm.k_o == k_o
        assert dm.X_orig.shape == (n * n, k_o)

    def test_symmetric_flow_design_matrix_k_fields(self):
        """flow_design_matrix sets k_d == k_o == k for symmetric case."""
        from neighbayes.graph import flow_design_matrix

        n = 4
        k = 2
        rng = np.random.default_rng(42)
        X = rng.standard_normal((n, k))

        dm = flow_design_matrix(X)
        assert dm.k_d == k
        assert dm.k_o == k
        assert dm.k == k


# ---------------------------------------------------------------------------
# DGP asymmetric tests
# ---------------------------------------------------------------------------


class TestDGPAsymmetric:
    """Tests for DGP functions with k_d ≠ k_o."""

    def test_generate_flow_data_asymmetric(self):
        """simulate_sar_flow works with beta_d and beta_o of different lengths."""
        from neighbayes.dgp.flows import simulate_sar_flow

        n = 5
        data = simulate_sar_flow(
            n=n,
            rho_d=0.15,
            rho_o=0.1,
            rho_w=0.05,
            beta_d=[1.0, -0.5],
            beta_o=[0.5],
            sigma=1.0,
            seed=42,
        )
        assert data["design"].k_d == 2
        assert data["design"].k_o == 1
        assert data["beta_d"].shape == (2,)
        assert data["beta_o"].shape == (1,)
        assert data["X_regional_d"].shape == (n, 2)
        assert data["X_regional_o"].shape == (n, 1)
        # Combined design: intercept(1) + ia(1) + dest(2) + orig(1) + intra(2) + dist(1)
        assert data["X"].shape[1] == 8

    def test_generate_flow_data_asymmetric_beta_layout(self):
        """Beta layout in asymmetric DGP places coefficients correctly."""
        from neighbayes.dgp.flows import simulate_sar_flow

        n = 5
        beta_d = [1.0, -0.5]
        beta_o = [0.5]
        data = simulate_sar_flow(
            n=n,
            rho_d=0.15,
            rho_o=0.1,
            rho_w=0.05,
            beta_d=beta_d,
            beta_o=beta_o,
            sigma=1.0,
            seed=42,
        )
        # Feature names should reflect k_d=2, k_o=1
        names = data["col_names"]
        assert "dest_x0" in names
        assert "dest_x1" in names
        assert "orig_y0" in names

    def test_generate_negbin_flow_data_asymmetric(self):
        """simulate_sar_negbin_flow works with k_d ≠ k_o."""
        from neighbayes.dgp.flows import simulate_sar_negbin_flow

        data = simulate_sar_negbin_flow(
            n=5,
            k_d=2,
            k_o=1,
            beta_d=[1.0, -0.5],
            beta_o=[0.5],
            seed=42,
        )
        assert data["Xd"].shape[1] == 2
        assert data["Xo"].shape[1] == 1
        assert data["design"].k_d == 2
        assert data["design"].k_o == 1

    def test_generate_panel_flow_data_asymmetric(self):
        """simulate_sar_flow_panel_fe works with k_d ≠ k_o."""
        from neighbayes.dgp.flows import simulate_sar_flow_panel_fe

        data = simulate_sar_flow_panel_fe(
            n=5,
            T=3,
            beta_d=[1.0, -0.5],
            beta_o=[0.5],
            sigma=1.0,
            seed=42,
        )
        assert data["beta_d"].shape == (2,)
        assert data["beta_o"].shape == (1,)

    def test_generate_flow_data_symmetric_backward_compat(self):
        """Symmetric case (k_d == k_o) still works as before."""
        from neighbayes.dgp.flows import simulate_sar_flow

        data = simulate_sar_flow(
            n=5,
            rho_d=0.15,
            rho_o=0.1,
            rho_w=0.05,
            beta_d=[1.0],
            beta_o=[0.5],
            sigma=1.0,
            seed=42,
        )
        assert data["design"].k_d == 1
        assert data["design"].k_o == 1
        assert data["design"].k == 1


# ---------------------------------------------------------------------------
# Model class asymmetric tests
# ---------------------------------------------------------------------------


class TestModelAsymmetric:
    """Tests for flow model classes with k_d ≠ k_o."""

    def test_model_infers_kd_ko_from_colnames(self):
        """FlowModel infers k_d and k_o from dest_*/orig_* column names."""
        from neighbayes.dgp.flows import simulate_sar_flow
        from neighbayes.models.flow._flow import SARFlow

        data = simulate_sar_flow(
            n=5,
            rho_d=0.15,
            rho_o=0.1,
            rho_w=0.05,
            beta_d=[1.0, -0.5],
            beta_o=[0.5],
            sigma=1.0,
            seed=42,
        )
        model = SARFlow(
            data["y_vec"],
            data["X"],
            data["G"],
            col_names=data["col_names"],
        )
        assert model._k_d == 2
        assert model._k_o == 1
        assert model._k == 2  # backward compat alias

    def test_effects_asymmetric_shape(self):
        """Spatial effects have correct shapes when k_d ≠ k_o."""
        from neighbayes.dgp.flows import simulate_sar_flow
        from neighbayes.models.flow._flow import OLSFlow

        data = simulate_sar_flow(
            n=5,
            rho_d=0.0,
            rho_o=0.0,
            rho_w=0.0,
            beta_d=[1.0, -0.5],
            beta_o=[0.5],
            sigma=0.1,
            seed=42,
        )
        model = OLSFlow(
            np.log(data["y_vec"]),
            data["X"],
            data["G"],
            col_names=data["col_names"],
        )
        model.fit(draws=30, tune=30, chains=1, progressbar=False, random_seed=0)

        posterior = model._compute_spatial_effects_posterior()
        # dest effects: k_d=2, orig effects: k_o=1, combined: k_d+k_o=3
        assert posterior["dest_total"].shape[1] == 2
        assert posterior["orig_total"].shape[1] == 1
        assert posterior["total"].shape[1] == 3  # concatenated when k_d != k_o

    def test_effects_symmetric_shape(self):
        """Spatial effects have correct shapes when k_d == k_o (symmetric)."""
        from neighbayes.dgp.flows import simulate_sar_flow
        from neighbayes.models.flow._flow import OLSFlow

        data = simulate_sar_flow(
            n=5,
            rho_d=0.0,
            rho_o=0.0,
            rho_w=0.0,
            beta_d=[1.0],
            beta_o=[0.5],
            sigma=0.1,
            seed=42,
        )
        model = OLSFlow(
            np.log(data["y_vec"]),
            data["X"],
            data["G"],
            col_names=data["col_names"],
        )
        model.fit(draws=30, tune=30, chains=1, progressbar=False, random_seed=0)

        posterior = model._compute_spatial_effects_posterior()
        # dest effects: k_d=1, orig effects: k_o=1, combined: k_d=1 (summed)
        assert posterior["dest_total"].shape[1] == 1
        assert posterior["orig_total"].shape[1] == 1
        assert posterior["total"].shape[1] == 1  # summed when k_d == k_o
