"""Tests for distance as a default regressor in flow DGPs."""

from __future__ import annotations

import numpy as np
import pytest
from libpysal.graph import Graph

from neighbayes.dgp import simulate_sar_flow, simulate_sar_negbin_flow
from neighbayes.dgp.utils import (
    pairwise_distance_matrix,
    synth_point_geodataframe,
)

N = 5
G = simulate_sar_flow(n=N, seed=0)["G"]
BETA_D = [1.0, -0.5]
BETA_O = [0.5, 0.3]


@pytest.fixture
def all_dgp_outputs():
    """Outputs from all 8 flow DGPs with default args."""
    return {
        "flow": simulate_sar_flow(N, G, 0.2, 0.2, 0.05, BETA_D, BETA_O, seed=0),
        "negbin": simulate_sar_negbin_flow(n=N, k=2, seed=0),
        "panel": simulate_sar_flow(
            n=N,
            T=2,
            G=G,
            rho_d=0.2,
            rho_o=0.1,
            rho_w=0.05,
            beta_d=BETA_D,
            beta_o=BETA_O,
            seed=0,
            pair_effects="fixed",
        ),
        "panel_negbin": simulate_sar_negbin_flow(n=N, T=2, G=G, seed=0),
        "flow_sep": simulate_sar_flow(
            n=N,
            G=G,
            rho_d=0.3,
            rho_o=0.2,
            beta_d=BETA_D,
            beta_o=BETA_O,
            seed=0,
            separable=True,
        ),
        "negbin_sep": simulate_sar_negbin_flow(n=N, k=2, seed=0, separable=True),
        "panel_sep": simulate_sar_flow(
            n=N,
            T=2,
            G=G,
            rho_d=0.3,
            rho_o=0.2,
            beta_d=BETA_D,
            beta_o=BETA_O,
            seed=0,
            separable=True,
            pair_effects="fixed",
        ),
        "panel_negbin_sep": simulate_sar_negbin_flow(
            n=N, T=2, G=G, seed=0, separable=True
        ),
    }


class TestSynthGeodataframe:
    def test_returns_n_rows(self):
        gdf = synth_point_geodataframe(7)
        assert len(gdf) == 7
        assert hasattr(gdf, "geometry")

    def test_pairwise_distance_zero_diagonal_symmetric(self):
        gdf = synth_point_geodataframe(6)
        d = pairwise_distance_matrix(gdf)
        assert d.shape == (6, 6)
        assert np.allclose(np.diag(d), 0.0)
        assert np.allclose(d, d.T)
        assert (d[~np.eye(6, dtype=bool)] > 0).all()


class TestDistanceColumnPresent:
    @pytest.mark.parametrize(
        "key",
        [
            "flow",
            "negbin",
            "panel",
            "panel_negbin",
            "flow_sep",
            "negbin_sep",
            "panel_sep",
            "panel_negbin_sep",
        ],
    )
    def test_log_distance_column(self, all_dgp_outputs, key):
        out = all_dgp_outputs[key]
        assert "log_distance" in out["col_names"]
        assert out["col_names"][-1] == "log_distance"

    @pytest.mark.parametrize(
        "key",
        [
            "flow",
            "negbin",
            "panel",
            "panel_negbin",
            "flow_sep",
            "negbin_sep",
            "panel_sep",
            "panel_negbin_sep",
        ],
    )
    def test_dist_and_gdf_returned(self, all_dgp_outputs, key):
        out = all_dgp_outputs[key]
        n_actual = out["G"].n_nodes
        assert out["gamma_dist"] == -0.5
        assert out["dist"].shape == (n_actual, n_actual)
        assert np.allclose(out["dist"], out["dist"].T)
        assert np.allclose(np.diag(out["dist"]), 0.0)
        assert len(out["gdf"]) == n_actual

    def test_design_last_column_is_log1p_dist(self, all_dgp_outputs):
        out = all_dgp_outputs["flow"]
        expected = np.log1p(out["dist"]).ravel()
        np.testing.assert_allclose(out["X"][:, -1], expected, atol=1e-12)


class TestDistanceFromUserGdf:
    def test_user_point_gdf(self):
        import geopandas as gpd
        from scipy.spatial.distance import cdist
        from shapely.geometry import Point

        rng = np.random.default_rng(0)
        coords = rng.uniform(0, 10, size=(N, 2))
        gdf = gpd.GeoDataFrame(
            {"id": np.arange(N)},
            geometry=[Point(x, y) for x, y in coords],
        )
        out = simulate_sar_flow(N, G, 0.1, 0.1, 0.05, BETA_D, BETA_O, gdf=gdf, seed=0)
        np.testing.assert_allclose(out["dist"], cdist(coords, coords), atol=1e-12)

    def test_polygon_gdf_uses_centroids(self):
        import geopandas as gpd
        from scipy.spatial.distance import cdist
        from shapely.geometry import box

        # 5 unit squares stacked in a row → centroids at (i+0.5, 0.5)
        polys = [box(i, 0, i + 1, 1) for i in range(N)]
        gdf = gpd.GeoDataFrame({"id": np.arange(N)}, geometry=polys)
        cents = np.column_stack([gdf.geometry.centroid.x, gdf.geometry.centroid.y])
        out = simulate_sar_flow(N, G, 0.1, 0.1, 0.05, BETA_D, BETA_O, gdf=gdf, seed=0)
        np.testing.assert_allclose(out["dist"], cdist(cents, cents), atol=1e-12)


class TestGammaDistOverride:
    def test_zero_disables_distance_effect(self):
        out_neg = simulate_sar_flow(
            N, G, 0.2, 0.2, 0.05, BETA_D, BETA_O, gamma_dist=-0.5, seed=0
        )
        out_zero = simulate_sar_flow(
            N, G, 0.2, 0.2, 0.05, BETA_D, BETA_O, gamma_dist=0.0, seed=0
        )
        # Identical X (same seed, same Xreg), but different y because of
        # the distance-decay term in the linear predictor.
        np.testing.assert_allclose(out_neg["X"], out_zero["X"], atol=1e-12)
        assert not np.allclose(out_neg["y_vec"], out_zero["y_vec"])
        assert out_neg["gamma_dist"] == -0.5
        assert out_zero["gamma_dist"] == 0.0


class TestAutoBuildG:
    """Auto-construction of the spatial graph G when not supplied."""

    def test_auto_build_G_when_none(self):
        # Neither n, G nor gdf supplied → falls back to default_n=25.
        out = simulate_sar_flow(seed=0)
        assert isinstance(out["G"], Graph)
        assert out["G"].n_nodes == 25
        assert len(out["gdf"]) == 25
        # KNN graph: each row of W has roughly knn_k positive entries
        # (after row standardization).
        nnz_per_row = np.diff(out["W"].indptr)
        assert nnz_per_row.min() >= 1

    def test_n_only_auto_build(self):
        # Only n supplied → builds gdf and KNN graph for that n.
        out = simulate_sar_negbin_flow(n=16, seed=0)
        assert out["G"].n_nodes == 16
        assert len(out["gdf"]) == 16
        assert out["y_mat"].shape == (16, 16)

    def test_user_gdf_drives_G(self):
        # gdf supplied without G → KNN graph derived from gdf.
        gdf = synth_point_geodataframe(9)
        out = simulate_sar_flow(gdf=gdf, beta_d=BETA_D, beta_o=BETA_O, seed=0, knn_k=3)
        assert out["G"].n_nodes == 9
        # The returned gdf is the same object the user passed in.
        assert out["gdf"] is gdf
        # Distance matrix matches the user's gdf.
        np.testing.assert_allclose(
            out["dist"], pairwise_distance_matrix(gdf), atol=1e-12
        )

    def test_explicit_G_synthesizes_gdf_for_distance(self):
        # G supplied without gdf → synthetic gdf used to populate distance.
        out = simulate_sar_flow(G=G, beta_d=BETA_D, beta_o=BETA_O, seed=0)
        assert out["G"].n_nodes == G.n_nodes
        assert out["gdf"] is not None
        assert len(out["gdf"]) == G.n_nodes
        assert out["dist"].shape == (G.n_nodes, G.n_nodes)

    def test_n_inferred_from_G(self):
        # When G is supplied, n is inferred (and disagreeing n raises).
        out = simulate_sar_negbin_flow(T=2, G=G, seed=0)
        assert out["G"].n_nodes == G.n_nodes
        assert out["y"].shape == (2 * G.n_nodes * G.n_nodes,)
        with pytest.raises(ValueError, match="disagrees"):
            simulate_sar_negbin_flow(n=G.n_nodes + 1, T=2, G=G, seed=0)

    def test_gdf_size_mismatch_raises(self):
        gdf = synth_point_geodataframe(7)
        with pytest.raises(ValueError, match="rows"):
            simulate_sar_flow(G=G, gdf=gdf, beta_d=BETA_D, beta_o=BETA_O, seed=0)
