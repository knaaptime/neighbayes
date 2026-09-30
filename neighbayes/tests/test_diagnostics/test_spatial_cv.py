"""Unit tests for :mod:`neighbayes.diagnostics.spatial_cv`."""

from __future__ import annotations

import numpy as np
import pytest
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


def test_spatial_kfold_iid_with_explicit_fold_ids(sar_grid):
    """OLS path: explicit fold_ids, iid closed-form predictive."""
    gdf, W = sar_grid
    model = OLS(formula="y ~ X_1", data=gdf, W=W)
    n = len(gdf)
    fold_ids = np.arange(n) % 3
    res = spatial_kfold(model, fold_ids=fold_ids, **FIT_KW)
    assert isinstance(res, SpatialCVResult)
    assert res.method == "explicit"
    assert res.n_folds == 3
    assert res.elpd_per_fold.shape == (3,)
    assert int(res.n_per_fold.sum()) == n
    assert np.isfinite(res.elpd)
    assert np.isfinite(res.se) and res.se >= 0.0


def test_spatial_kfold_spatial_lag_with_geometry(sar_grid):
    """SAR path: KMeans fold construction + spatial-precision predictive."""
    gdf, W = sar_grid
    model = SAR(formula="y ~ X_1", data=gdf, W=W, logdet_method="eigenvalue")
    res = spatial_kfold(model, geometry=gdf.geometry, n_blocks=3, **FIT_KW)
    assert res.method == "kmeans"
    assert res.n_folds == 3
    assert np.isfinite(res.elpd)
    assert int(res.n_per_fold.sum()) == len(gdf)


def test_spatial_kfold_requires_fold_ids_or_geometry(sar_grid):
    gdf, W = sar_grid
    model = OLS(formula="y ~ X_1", data=gdf, W=W)
    with pytest.raises(ValueError, match="fold_ids, or geometry"):
        spatial_kfold(model, **FIT_KW)


def test_spatial_kfold_validates_fold_ids_length(sar_grid):
    gdf, W = sar_grid
    model = OLS(formula="y ~ X_1", data=gdf, W=W)
    with pytest.raises(ValueError, match="fold_ids has length"):
        spatial_kfold(model, fold_ids=np.zeros(len(gdf) - 1, dtype=int), **FIT_KW)


def test_spatial_kfold_requires_at_least_two_folds(sar_grid):
    gdf, W = sar_grid
    model = OLS(formula="y ~ X_1", data=gdf, W=W)
    with pytest.raises(ValueError, match="at least 2 folds"):
        spatial_kfold(model, fold_ids=np.zeros(len(gdf), dtype=int), **FIT_KW)


class _ModuloSplitter:
    """Minimal sklearn-style splitter for testing the splitter= path.

    Stands in for any geovalidate splitter (HilbertKFold, LeaveClusterOut,
    etc.) which all expose the same ``split(X) -> (train, test)`` protocol.
    """

    def __init__(self, n_splits: int):
        self.n_splits = n_splits

    def split(self, X):
        n = len(X)
        idx = np.arange(n)
        for f in range(self.n_splits):
            test = idx[idx % self.n_splits == f]
            train = idx[idx % self.n_splits != f]
            yield train, test


def test_spatial_kfold_accepts_sklearn_splitter(sar_grid):
    """splitter= path: any sklearn-compatible splitter (incl. geovalidate)."""
    gdf, W = sar_grid
    model = OLS(formula="y ~ X_1", data=gdf, W=W)
    splitter = _ModuloSplitter(n_splits=3)
    res = spatial_kfold(model, splitter=splitter, geometry=gdf.geometry, **FIT_KW)
    assert res.method == "_ModuloSplitter"
    assert res.n_folds == 3
    assert int(res.n_per_fold.sum()) == len(gdf)
    assert np.isfinite(res.elpd)
    assert np.isfinite(res.se) and res.se >= 0.0


def test_spatial_kfold_rejects_splitter_and_fold_ids(sar_grid):
    gdf, W = sar_grid
    model = OLS(formula="y ~ X_1", data=gdf, W=W)
    with pytest.raises(ValueError, match="splitter or fold_ids"):
        spatial_kfold(
            model,
            splitter=_ModuloSplitter(n_splits=2),
            fold_ids=np.arange(len(gdf)) % 2,
            **FIT_KW,
        )
