"""API-level tests for the public neighbayes.dgp module."""

from __future__ import annotations

import inspect

import numpy as np
import pytest
import scipy.sparse as sp

from neighbayes import dgp
from neighbayes.tests.helpers import W_to_graph, make_rook_W


def test_dgp_exports_have_gdf_argument() -> None:
    """All exported simulator functions should accept a ``gdf`` argument."""
    funcs = [getattr(dgp, name) for name in dgp.__all__]
    for fn in funcs:
        sig = inspect.signature(fn)
        assert "gdf" in sig.parameters or any(
            p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
        ), f"{fn.__name__} must accept gdf directly or through **kwargs"


def test_cross_sectional_generators_run_with_sparse_W() -> None:
    """Cross-sectional DGPs should run and return consistent output keys."""
    rng = np.random.default_rng(123)
    W = sp.csr_matrix(make_rook_W(4))

    out_sar = dgp.simulate_sar(W=W, rng=rng)
    out_sem = dgp.simulate_sem(W=W, rng=rng)
    out_slx = dgp.simulate_slx(W=W, rng=rng)
    out_sdm = dgp.simulate_sdm(W=W, rng=rng)
    out_sdem = dgp.simulate_sdem(W=W, rng=rng)

    for out in (out_sar, out_sem, out_slx, out_sdm, out_sdem):
        assert set(("y", "X", "W_sparse", "W_graph", "params_true")).issubset(out)
        assert out["y"].shape[0] == out["X"].shape[0]


def test_cross_sectional_generators_run_with_n_only_as_square_grid() -> None:
    """Supplying only n should create an n x n grid with n^2 observations."""
    rng = np.random.default_rng(123)
    n_side = 4
    expected_nobs = n_side * n_side

    out_sar = dgp.simulate_sar(n_side=n_side, rng=rng)
    out_sem = dgp.simulate_sem(n_side=n_side, rng=rng)
    out_slx = dgp.simulate_slx(n_side=n_side, rng=rng)
    out_sdm = dgp.simulate_sdm(n_side=n_side, rng=rng)
    out_sdem = dgp.simulate_sdem(n_side=n_side, rng=rng)

    for out in (out_sar, out_sem, out_slx, out_sdm, out_sdem):
        assert out["W_sparse"].shape == (expected_nobs, expected_nobs)
        assert out["y"].shape[0] == expected_nobs
        assert out["X"].shape[0] == expected_nobs


def test_cross_sectional_create_gdf_with_point_geometry() -> None:
    """create_gdf=True with geometry_type='point' should return point geometry."""
    gpd = pytest.importorskip("geopandas")

    rng = np.random.default_rng(123)
    n_side = 4
    expected_nobs = n_side * n_side

    outs = [
        dgp.simulate_sar(
            n_side=n_side, rng=rng, create_gdf=True, geometry_type="point"
        ),
        dgp.simulate_sem(
            n_side=n_side, rng=rng, create_gdf=True, geometry_type="point"
        ),
        dgp.simulate_slx(
            n_side=n_side, rng=rng, create_gdf=True, geometry_type="point"
        ),
        dgp.simulate_sdm(
            n_side=n_side, rng=rng, create_gdf=True, geometry_type="point"
        ),
        dgp.simulate_sdem(
            n_side=n_side, rng=rng, create_gdf=True, geometry_type="point"
        ),
    ]

    for gdf_out in outs:
        assert isinstance(gdf_out, gpd.GeoDataFrame)
        assert len(gdf_out) == expected_nobs
        assert "y" in gdf_out.columns
        assert "X_0" in gdf_out.columns
        assert gdf_out.geom_type.eq("Point").all()


def test_cross_sectional_create_gdf_with_polygon_geometry() -> None:
    """create_gdf=True with geometry_type='polygon' should return polygon geometry."""
    gpd = pytest.importorskip("geopandas")

    gdf_out = dgp.simulate_sar(
        n_side=4, seed=123, create_gdf=True, geometry_type="polygon"
    )
    assert isinstance(gdf_out, gpd.GeoDataFrame)
    assert gdf_out.geom_type.eq("Polygon").all()


def test_cross_sectional_create_gdf_rejects_invalid_geometry_type() -> None:
    """Unsupported geometry types should raise a clear ValueError."""
    with pytest.raises(ValueError, match="geometry_type"):
        dgp.simulate_sar(n_side=4, seed=123, create_gdf=True, geometry_type="triangle")


def test_cross_sectional_create_gdf_default_is_backward_compatible() -> None:
    """Without create_gdf, the return value should be a plain dict."""
    out = dgp.simulate_sar(n_side=4, seed=123)
    assert isinstance(out, dict)


def test_panel_generators_run_with_sparse_W() -> None:
    """Panel DGPs should return time-first stacked outputs of length ``N*T``."""
    rng = np.random.default_rng(123)
    N, T = 9, 5  # make_rook_W(3) produces a 3x3 grid -> 9 units
    W = sp.csr_matrix(make_rook_W(3))

    panel_outs = [
        dgp.simulate_panel_ols_fe(N=N, T=T, W=W, rng=rng),
        dgp.simulate_panel_sar_fe(N=N, T=T, W=W, rng=rng),
        dgp.simulate_panel_sem_fe(N=N, T=T, W=W, rng=rng),
        dgp.simulate_panel_sdm_fe(N=N, T=T, W=W, rng=rng),
        dgp.simulate_panel_sdem_fe(N=N, T=T, W=W, rng=rng),
        dgp.simulate_panel_ols_re(N=N, T=T, W=W, rng=rng),
        dgp.simulate_panel_sar_re(N=N, T=T, W=W, rng=rng),
        dgp.simulate_panel_sem_re(N=N, T=T, W=W, rng=rng),
        dgp.simulate_panel_dlm_fe(N=N, T=T, W=W, rng=rng),
        dgp.simulate_panel_sdmr_fe(N=N, T=T, W=W, rng=rng),
        dgp.simulate_panel_sdmu_fe(N=N, T=T, W=W, rng=rng),
        dgp.simulate_panel_sar_tobit_fe(N=N, T=T, W=W, rng=rng),
        dgp.simulate_panel_sem_tobit_fe(N=N, T=T, W=W, rng=rng),
        dgp.simulate_panel_slx_fe(N=N, T=T, W=W, rng=rng),
    ]

    for out in panel_outs:
        assert out["y"].shape[0] == N * T
        assert out["X"].shape[0] == N * T
        assert out["unit"].shape[0] == N * T
        assert out["time"].shape[0] == N * T


def test_nonlinear_generators_run_with_sparse_W() -> None:
    """Nonlinear DGPs should run and return expected keys and shapes."""
    rng = np.random.default_rng(123)
    W = sp.csr_matrix(make_rook_W(3))

    out_sar_tobit = dgp.simulate_sar_tobit(W=W, rng=rng)
    out_sem_tobit = dgp.simulate_sem_tobit(W=W, rng=rng)
    out_sdm_tobit = dgp.simulate_sdm_tobit(W=W, rng=rng)
    out_probit = dgp.simulate_spatial_probit(W=W, rng=rng)

    for out in (out_sar_tobit, out_sem_tobit, out_sdm_tobit):
        assert "y_latent" in out and "censored_mask" in out
        assert out["y"].shape == out["y_latent"].shape
        assert out["censored_mask"].shape == out["y"].shape

    assert out_probit["y"].shape[0] == out_probit["X"].shape[0]
    assert out_probit["region_ids"].shape[0] == out_probit["y"].shape[0]


def test_gdf_input_path_for_representative_generators() -> None:
    """Representative simulators should accept GeoDataFrame input.

    This test uses a small regular grid and checks one cross-sectional, one
    panel, and one nonlinear generator.
    """
    gpd = pytest.importorskip("geopandas")
    from shapely.geometry import box

    rng = np.random.default_rng(123)
    polys = [box(j, i, j + 1, i + 1) for i in range(3) for j in range(3)]
    gdf = gpd.GeoDataFrame({"id": range(9)}, geometry=polys)

    out_sar = dgp.simulate_sar(gdf=gdf, rng=rng)
    out_panel = dgp.simulate_panel_sar_fe(N=9, T=3, gdf=gdf, rng=rng)
    out_probit = dgp.simulate_spatial_probit(gdf=gdf, rng=rng)

    assert isinstance(out_sar, gpd.GeoDataFrame)
    assert len(out_sar) == 9
    assert "y" in out_sar.columns
    assert isinstance(out_panel, tuple)
    _, panel_long = out_panel
    assert panel_long["y"].shape[0] == 27
    assert out_probit["region_ids"].max() == 8


def test_cross_sectional_create_gdf_reuses_input_geometry() -> None:
    """When gdf is provided, create_gdf should attach columns onto that geometry."""
    gpd = pytest.importorskip("geopandas")
    from shapely.geometry import box

    polys = [box(j, i, j + 1, i + 1) for i in range(3) for j in range(3)]
    gdf = gpd.GeoDataFrame({"id": range(9)}, geometry=polys)

    gdf_out = dgp.simulate_sar(
        gdf=gdf, seed=123, create_gdf=True, geometry_type="point"
    )
    assert isinstance(gdf_out, gpd.GeoDataFrame)
    assert len(gdf_out) == len(gdf)
    assert gdf_out.geom_type.eq("Polygon").all()
    assert "y" in gdf_out.columns
    assert "X_0" in gdf_out.columns


def test_both_gdf_and_graph_inputs_are_supported() -> None:
    """Providing both gdf and an associated Graph should be supported."""
    gpd = pytest.importorskip("geopandas")
    from shapely.geometry import box

    rng = np.random.default_rng(123)
    polys = [box(j, i, j + 1, i + 1) for i in range(3) for j in range(3)]
    gdf = gpd.GeoDataFrame({"id": range(9)}, geometry=polys)

    W_dense = make_rook_W(3)
    W_graph = W_to_graph(W_dense)

    out = dgp.simulate_sar(W=W_graph, gdf=gdf, rng=rng)
    assert isinstance(out, gpd.GeoDataFrame)
    assert len(out) == 9
    assert "y" in out.columns


def test_both_gdf_and_graph_inputs_must_match_dimensions() -> None:
    """Passing incompatible W and gdf should raise a clear error."""
    gpd = pytest.importorskip("geopandas")
    from shapely.geometry import box

    polys = [box(j, i, j + 1, i + 1) for i in range(3) for j in range(3)]
    gdf = gpd.GeoDataFrame({"id": range(9)}, geometry=polys)

    W_dense = make_rook_W(2)
    W_graph = W_to_graph(W_dense)

    with pytest.raises(ValueError, match="same number of spatial units"):
        dgp.simulate_sar(W=W_graph, gdf=gdf, seed=123)


def test_n_must_match_explicit_weights_dimensions() -> None:
    """When W is provided, n_side**2 must match the number of units."""
    W = sp.csr_matrix(make_rook_W(3))
    with pytest.raises(ValueError, match="n_side=4 implies 16 units"):
        dgp.simulate_sar(W=W, n_side=4, seed=123)


def test_panel_create_gdf_returns_tuple() -> None:
    """create_gdf=True returns (N-row GeoDataFrame, N*T-row DataFrame)."""
    gpd = pytest.importorskip("geopandas")
    N, T = 9, 3
    W = sp.csr_matrix(make_rook_W(3))
    result = dgp.simulate_panel_sar_fe(N=N, T=T, W=W, seed=42, create_gdf=True)
    assert isinstance(result, tuple)
    unit_gdf, long_df = result
    assert isinstance(unit_gdf, gpd.GeoDataFrame)
    assert len(unit_gdf) == N
    assert len(long_df) == N * T
    assert set(long_df.columns) >= {"unit", "time", "y", "X_0", "X_1"}


def test_panel_create_gdf_reuses_input_geometry() -> None:
    """When gdf is passed, the unit GeoDataFrame reuses that geometry."""
    gpd = pytest.importorskip("geopandas")
    from shapely.geometry import box

    N, T = 9, 3
    polys = [box(j, i, j + 1, i + 1) for i in range(3) for j in range(3)]
    input_gdf = gpd.GeoDataFrame({"extra_col": range(N)}, geometry=polys)

    unit_gdf, long_df = dgp.simulate_panel_sar_fe(N=N, T=T, gdf=input_gdf, seed=42)
    assert isinstance(unit_gdf, gpd.GeoDataFrame)
    assert len(unit_gdf) == N
    # geometry is reused; extra non-geometry columns are NOT carried over
    assert unit_gdf.geometry.geom_type.eq("Polygon").all()
    assert "extra_col" not in unit_gdf.columns
    assert len(long_df) == N * T


def test_panel_wide_returns_single_geodataframe() -> None:
    """wide=True returns a single N-row GeoDataFrame with pivoted columns."""
    gpd = pytest.importorskip("geopandas")
    N, T = 9, 3
    W = sp.csr_matrix(make_rook_W(3))
    result = dgp.simulate_panel_sar_fe(N=N, T=T, W=W, seed=42, wide=True)
    assert isinstance(result, gpd.GeoDataFrame)
    assert len(result) == N
    assert "y_t0" in result.columns
    assert f"y_t{T - 1}" in result.columns
    assert "X_0_t0" in result.columns


def test_panel_default_backward_compatible() -> None:
    """Panel functions still return a plain dict when create_gdf=False and gdf=None."""
    W = sp.csr_matrix(make_rook_W(3))
    out = dgp.simulate_panel_sar_fe(N=9, T=3, W=W, seed=42)
    assert isinstance(out, dict)
    assert "y" in out and "W_graph" in out


def test_panel_tobit_create_gdf() -> None:
    """Panel Tobit wrappers respect the create_gdf parameter."""
    gpd = pytest.importorskip("geopandas")
    N, T = 9, 3
    W = sp.csr_matrix(make_rook_W(3))
    result = dgp.simulate_panel_sar_tobit_fe(
        N=N, T=T, W=W, seed=42, create_gdf=True, censoring=0.0
    )
    assert isinstance(result, tuple)
    unit_gdf, long_df = result
    assert isinstance(unit_gdf, gpd.GeoDataFrame)
    assert len(unit_gdf) == N
    assert len(long_df) == N * T


# --- Heteroskedasticity tests ---


def test_dgp_exports_have_err_hetero_argument() -> None:
    """All exported simulator functions should accept an ``err_hetero`` argument."""
    funcs = [getattr(dgp, name) for name in dgp.__all__]
    for fn in funcs:
        sig = inspect.signature(fn)
        assert "err_hetero" in sig.parameters or any(
            p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
        ), f"{fn.__name__} must accept err_hetero directly or through **kwargs"


def test_cross_sectional_err_hetero_produces_different_variances() -> None:
    """err_hetero=True should produce heteroskedastic errors in cross-sectional DGPs.

    We verify this by checking that the _hetero_scale helper produces
    observation-specific standard deviations that vary with X norms.
    """
    from neighbayes.dgp.utils import _hetero_scale

    rng = np.random.default_rng(42)
    # Create a design matrix with varying row norms.
    X = np.column_stack([np.ones(20), rng.standard_normal((20, 3))])
    scales = _hetero_scale(X, sigma=1.0)
    # Scales should vary across observations.
    assert scales.max() / scales.min() > 1.3, (
        f"_hetero_scale should produce varying scales, got max/min = {scales.max() / scales.min():.2f}"
    )
    # Scales should be >= sigma (since sqrt(1 + ||x||^2) >= 1).
    assert (scales >= 1.0).all(), "All scales should be >= sigma"

    # Also verify that simulate_ols with err_hetero=True produces different
    # results than err_hetero=False (same seed should give different y).
    out_homo = dgp.simulate_ols(n_side=4, sigma=1.0, err_hetero=False, seed=123)
    out_hetero = dgp.simulate_ols(n_side=4, sigma=1.0, err_hetero=True, seed=123)
    # The y values should differ because the error scaling is different.
    assert not np.allclose(out_homo["y"], out_hetero["y"]), (
        "err_hetero=True should produce different y values than err_hetero=False"
    )


def test_cross_sectional_err_hetero_false_is_homoskedastic() -> None:
    """err_hetero=False (default) should produce homoskedastic errors."""
    # Verify that with err_hetero=False, the error scale is constant sigma.
    from neighbayes.dgp.utils import _hetero_scale

    rng = np.random.default_rng(42)
    X = np.column_stack([np.ones(20), rng.standard_normal((20, 3))])
    # When err_hetero=False, the scale is just sigma (scalar), not _hetero_scale.
    # Verify _hetero_scale produces non-constant scales for comparison.
    scales = _hetero_scale(X, sigma=1.0)
    assert not np.allclose(scales, 1.0), (
        "_hetero_scale should produce non-constant scales for non-trivial X"
    )

    # Verify backward compatibility: default err_hetero=False gives same
    # results as before the parameter was added.
    out1 = dgp.simulate_ols(n_side=4, sigma=1.0, seed=123)
    out2 = dgp.simulate_ols(n_side=4, sigma=1.0, seed=123, err_hetero=False)
    assert np.allclose(out1["y"], out2["y"]), (
        "err_hetero=False should produce identical results to the default"
    )


def test_panel_fe_err_hetero_runs_and_preserves_shape() -> None:
    """Panel FE simulators with err_hetero=True should return same shapes."""
    rng = np.random.default_rng(42)
    N, T = 9, 3
    W = sp.csr_matrix(make_rook_W(3))

    for sim_fn in [
        dgp.simulate_panel_ols_fe,
        dgp.simulate_panel_sar_fe,
        dgp.simulate_panel_sem_fe,
        dgp.simulate_panel_sdm_fe,
        dgp.simulate_panel_sdem_fe,
    ]:
        out_homo = sim_fn(N=N, T=T, W=W, rng=rng, err_hetero=False)
        out_hetero = sim_fn(N=N, T=T, W=W, rng=rng, err_hetero=True)
        assert out_homo["y"].shape == out_hetero["y"].shape
        assert out_homo["X"].shape == out_hetero["X"].shape
        assert out_homo["params_true"].keys() == out_hetero["params_true"].keys()


def test_panel_re_err_hetero_forwarded() -> None:
    """Panel RE wrappers should forward err_hetero to underlying FE simulators."""
    rng = np.random.default_rng(42)
    N, T = 9, 3
    W = sp.csr_matrix(make_rook_W(3))

    for sim_fn in [
        dgp.simulate_panel_ols_re,
        dgp.simulate_panel_sar_re,
        dgp.simulate_panel_sem_re,
    ]:
        out = sim_fn(N=N, T=T, W=W, rng=rng, err_hetero=True)
        assert out["y"].shape == (N * T,)


def test_dynamic_panel_err_hetero_runs_and_preserves_shape() -> None:
    """Dynamic panel simulators with err_hetero=True should return same shapes."""
    rng = np.random.default_rng(42)
    N, T = 9, 3
    W = sp.csr_matrix(make_rook_W(3))

    for sim_fn in [
        dgp.simulate_panel_dlm_fe,
        dgp.simulate_panel_sdmr_fe,
        dgp.simulate_panel_sdmu_fe,
        dgp.simulate_panel_sar_dynamic_fe,
        dgp.simulate_panel_sem_dynamic_fe,
        dgp.simulate_panel_sdem_dynamic_fe,
        dgp.simulate_panel_slx_dynamic_fe,
    ]:
        out_homo = sim_fn(N=N, T=T, W=W, rng=rng, err_hetero=False)
        out_hetero = sim_fn(N=N, T=T, W=W, rng=rng, err_hetero=True)
        assert out_homo["y"].shape == out_hetero["y"].shape
        assert out_homo["X"].shape == out_hetero["X"].shape


def test_tobit_err_hetero_forwarded() -> None:
    """Tobit wrappers should forward err_hetero to underlying simulators."""
    rng = np.random.default_rng(42)
    W = sp.csr_matrix(make_rook_W(3))

    # Cross-sectional tobit
    for sim_fn in [
        dgp.simulate_sar_tobit,
        dgp.simulate_sem_tobit,
        dgp.simulate_sdm_tobit,
    ]:
        out = sim_fn(W=W, rng=rng, err_hetero=True)
        assert "y_latent" in out
        assert "censored_mask" in out

    # Panel tobit
    N, T = 9, 3
    for sim_fn in [dgp.simulate_panel_sar_tobit_fe, dgp.simulate_panel_sem_tobit_fe]:
        out = sim_fn(N=N, T=T, W=W, rng=rng, err_hetero=True)
        assert "y_latent" in out
        assert "censored_mask" in out


def test_spatial_probit_err_hetero_runs() -> None:
    """Spatial probit with err_hetero=True should run and return same keys."""
    rng = np.random.default_rng(42)
    W = sp.csr_matrix(make_rook_W(3))

    out_homo = dgp.simulate_spatial_probit(W=W, rng=rng, err_hetero=False)
    out_hetero = dgp.simulate_spatial_probit(W=W, rng=rng, err_hetero=True)
    assert out_homo.keys() == out_hetero.keys()
    assert out_homo["y"].shape == out_hetero["y"].shape
