"""Unit tests for :mod:`neighbayes.diagnostics.spatial_cv`."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("sklearn")
from libpysal.graph import Graph

from neighbayes.dgp import simulate_sar
from neighbayes.diagnostics import SpatialCVResult, spatial_kfold
from neighbayes.models import OLS, SAR

FIT_KW = dict(draws=20, tune=20, chains=1, random_seed=0, progressbar=False)


@pytest.fixture(scope="module")
def sar_grid():
    """6x6 SAR-generated GeoDataFrame plus its rook contiguity Graph."""
    gdf = simulate_sar(
        n_side=6,
        rho=0.5,
        beta=np.array([1.0, 2.0]),
        sigma=1.0,
        seed=0,
        create_gdf=True,
        geometry_type="polygon",
    )
    W = Graph.build_contiguity(gdf, rook=True).transform("r")
    return gdf, W


def test_spatial_kfold_iid_with_predefined_split(sar_grid):
    """OLS path: PredefinedSplit folds, iid closed-form predictive."""
    from sklearn.model_selection import PredefinedSplit

    gdf, W = sar_grid
    model = OLS(formula="y ~ X_1", data=gdf, W=W)
    n = len(gdf)
    fold_ids = np.arange(n) % 3
    res = spatial_kfold(model, PredefinedSplit(fold_ids), **FIT_KW)
    assert isinstance(res, SpatialCVResult)
    assert res.method == "PredefinedSplit"
    assert res.n_folds == 3
    np.testing.assert_array_equal(res.fold_ids, fold_ids)
    assert res.elpd_per_fold.shape == (3,)
    assert int(res.n_per_fold.sum()) == n
    assert np.isfinite(res.elpd)
    assert np.isfinite(res.se) and res.se >= 0.0


def test_spatial_kfold_spatial_lag_with_group_kfold(sar_grid):
    """SAR path: spatial blocks as groups, spatial-precision predictive."""
    from sklearn.model_selection import GroupKFold

    gdf, W = sar_grid
    model = SAR(formula="y ~ X_1", data=gdf, W=W, logdet_method="eigenvalue")
    # 6x6 lattice in row-major order: three 2-row bands.
    groups = np.arange(len(gdf)) // 12
    res = spatial_kfold(model, GroupKFold(n_splits=3), groups=groups, **FIT_KW)
    assert res.method == "GroupKFold"
    assert res.n_folds == 3
    assert np.isfinite(res.elpd)
    assert int(res.n_per_fold.sum()) == len(gdf)


def test_spatial_kfold_with_geovalidate_splitter(sar_grid):
    """Integration: a geovalidate splitter takes the geometry as ``X``."""
    geovalidate = pytest.importorskip("geovalidate")
    gdf, W = sar_grid
    model = SAR(formula="y ~ X_1", data=gdf, W=W, logdet_method="eigenvalue")
    splitter = geovalidate.HilbertKFold(n_splits=3)
    res = spatial_kfold(model, splitter, geometry=gdf.geometry, **FIT_KW)
    assert res.method == "HilbertKFold"
    assert res.n_folds == 3
    assert int(res.n_per_fold.sum()) == len(gdf)
    assert np.isfinite(res.elpd)


def test_spatial_kfold_requires_a_splitter(sar_grid):
    gdf, W = sar_grid
    model = OLS(formula="y ~ X_1", data=gdf, W=W)
    with pytest.raises(TypeError):
        spatial_kfold(model, **FIT_KW)
    with pytest.raises(TypeError, match="split"):
        spatial_kfold(model, np.arange(len(gdf)) % 3, **FIT_KW)


def test_spatial_kfold_rejects_robust(sar_grid):
    from sklearn.model_selection import PredefinedSplit

    gdf, W = sar_grid
    model = SAR(formula="y ~ X_1", data=gdf, W=W, robust=True)
    with pytest.raises(NotImplementedError, match="robust"):
        spatial_kfold(model, PredefinedSplit(np.arange(len(gdf)) % 2), **FIT_KW)


def test_spatial_kfold_requires_at_least_two_folds(sar_grid):
    from sklearn.model_selection import PredefinedSplit

    gdf, W = sar_grid
    model = OLS(formula="y ~ X_1", data=gdf, W=W)
    with pytest.raises(ValueError, match="at least 2 folds"):
        spatial_kfold(model, PredefinedSplit(np.zeros(len(gdf), dtype=int)), **FIT_KW)


class _ModuloSplitter:
    """Minimal duck-typed splitter: only ``split(X)``, no sklearn base class."""

    def __init__(self, n_splits: int):
        self.n_splits = n_splits

    def split(self, X):
        n = len(X)
        idx = np.arange(n)
        for f in range(self.n_splits):
            test = idx[idx % self.n_splits == f]
            train = idx[idx % self.n_splits != f]
            yield train, test


def test_spatial_kfold_accepts_duck_typed_splitter(sar_grid):
    gdf, W = sar_grid
    model = OLS(formula="y ~ X_1", data=gdf, W=W)
    res = spatial_kfold(
        model, _ModuloSplitter(n_splits=3), geometry=gdf.geometry, **FIT_KW
    )
    assert res.method == "_ModuloSplitter"
    assert res.n_folds == 3
    assert int(res.n_per_fold.sum()) == len(gdf)
    assert np.isfinite(res.elpd)
    assert np.isfinite(res.se) and res.se >= 0.0


def test_spatial_kfold_refit_keeps_w_vars():
    """The per-fold refit lags the same covariates as the model (``w_vars``)."""
    from sklearn.model_selection import PredefinedSplit

    from neighbayes.models import SDM

    gdf = simulate_sar(
        n_side=6,
        rho=0.5,
        beta=np.array([1.0, 2.0, -1.0]),
        sigma=1.0,
        seed=0,
        create_gdf=True,
        geometry_type="polygon",
    )
    W = Graph.build_contiguity(gdf, rook=True).transform("r")
    model = SDM(formula="y ~ X_1 + X_2", data=gdf, W=W, w_vars=["X_1"])
    res = spatial_kfold(model, PredefinedSplit(np.arange(len(gdf)) % 2), **FIT_KW)
    assert np.isfinite(res.elpd)
