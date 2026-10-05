"""Shared utilities for DGP simulation in neighbayes."""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
import pandas as pd
import scipy.sparse as sp
from libpysal.graph import Graph


def ensure_rng(
    rng: np.random.Generator | None = None, seed: int | None = None
) -> np.random.Generator:
    """Return a reproducible NumPy random generator.

    Parameters
    ----------
    rng : np.random.Generator, optional
        Existing generator. If provided it takes precedence over ``seed``.
    seed : int, optional
        Seed for constructing a new generator when ``rng`` is not supplied.

    Returns
    -------
    np.random.Generator
        Random number generator used by simulation functions.
    """
    if rng is not None:
        return rng
    return np.random.default_rng(seed)


def _row_standardize_sparse(W: sp.spmatrix) -> sp.csr_matrix:
    """Row-standardize a sparse weights matrix, leaving zero-sum rows untouched.

    Rows whose sum is exactly ``0`` (isolates) stay rows of zeros rather than
    becoming ``NaN``; the result is row-stochastic on the non-isolated rows only.
    """
    W = sp.csr_matrix(W, dtype=np.float64)
    rs = np.asarray(W.sum(axis=1)).ravel()
    rs[rs == 0.0] = 1.0
    return sp.csr_matrix(sp.diags(1.0 / rs) @ W)


def rook_grid_weights(n_side: int) -> tuple[sp.csr_matrix, Graph]:
    """Build row-standardized rook-contiguity weights on an ``n_side x n_side`` grid.

    Parameters
    ----------
    n_side : int
        Number of rows and columns in the square grid.

    Returns
    -------
    tuple[scipy.sparse.csr_matrix, Graph]
        Sparse and Graph forms of the same row-standardized weights.
    """
    n_side = int(n_side)
    if n_side <= 0:
        raise ValueError(
            "n_side must be a positive integer when generating a default grid."
        )
    if n_side == 1:
        raise ValueError(
            "n_side=1 is degenerate: a 1×1 grid has no rook neighbors. Use n_side >= 2."
        )

    idx = np.arange(n_side * n_side).reshape(n_side, n_side)
    # Horizontal and vertical neighbor pairs, each in both directions.
    pairs = np.concatenate(
        [
            np.column_stack([idx[:, :-1].ravel(), idx[:, 1:].ravel()]),
            np.column_stack([idx[:-1, :].ravel(), idx[1:, :].ravel()]),
        ]
    )
    focal = np.concatenate([pairs[:, 0], pairs[:, 1]])
    neighbor = np.concatenate([pairs[:, 1], pairs[:, 0]])
    order = np.lexsort((neighbor, focal))

    g = Graph.from_arrays(
        focal[order],
        neighbor[order],
        np.ones(len(focal), dtype=float),
    ).transform("r")
    return g.sparse.tocsr().astype(np.float64), g


def weights_from_geodataframe(
    gdf: Any,
    contiguity: str = "queen",
    k: int = 4,
    distance_threshold: float | None = None,
) -> Graph:
    """Build a row-standardized Graph from GeoDataFrame geometry.

    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        Input geodataframe with geometry column.
    contiguity : {"queen", "rook", "knn", "distance"}, default="queen"
        Rule to build neighbor structure.
    k : int, default=4
        Number of neighbors for ``contiguity='knn'``.
    distance_threshold : float, optional
        Distance cutoff for ``contiguity='distance'``.

    Returns
    -------
    Graph
        Row-standardized graph built from geometry.
    """
    if gdf is None:
        raise ValueError("gdf must be provided when building weights from geometry.")
    if not hasattr(gdf, "geometry"):
        raise TypeError(
            "gdf must be a GeoDataFrame-like object with a geometry column."
        )

    mode = contiguity.lower()
    if mode == "queen":
        g = Graph.build_contiguity(gdf, rook=False)
    elif mode == "rook":
        g = Graph.build_contiguity(gdf, rook=True)
    elif mode == "knn":
        g = Graph.build_knn(gdf, k=int(k))
    elif mode == "distance":
        if distance_threshold is None:
            raise ValueError(
                "distance_threshold must be supplied when contiguity='distance'."
            )
        g = Graph.build_distance_band(
            gdf, threshold=float(distance_threshold), binary=True
        )
    else:
        raise ValueError(
            "contiguity must be one of {'queen', 'rook', 'knn', 'distance'}."
        )

    return g.transform("r")


def resolve_weights(
    W: Graph | sp.spmatrix | None = None,
    gdf: Any | None = None,
    n_side: int | None = None,
    contiguity: str = "queen",
    k: int = 4,
    distance_threshold: float | None = None,
) -> tuple[sp.csr_matrix, Graph]:
    """Resolve user-supplied spatial structure to a sparse matrix and a Graph.

    ``W`` is never densified: every simulator works from the sparse form.

    Parameters
    ----------
    W : Graph or scipy.sparse matrix, optional
        Explicit spatial structure. If provided together with ``gdf``, the
        graph is used and checked for dimensional compatibility with the
        GeoDataFrame.
    gdf : geopandas.GeoDataFrame, optional
        Used only when ``W`` is not supplied.
    n_side : int, optional
        If provided without ``W`` and ``gdf``, generate a rook-contiguity
        ``n_side x n_side`` grid (``n_side**2`` units).  When ``W`` or
        ``gdf`` is supplied, ``n_side**2`` must equal the number of units.
    contiguity : str, default="queen"
        GeoDataFrame neighbor construction mode.
    k : int, default=4
        KNN neighbor count when ``contiguity='knn'``.
    distance_threshold : float, optional
        Distance threshold when ``contiguity='distance'``.

    Returns
    -------
    tuple[scipy.sparse.csr_matrix, Graph]
        ``(W_sparse, W_graph)``, both row-standardized.
    """
    if W is not None:
        if isinstance(W, Graph):
            g = W.transform("r")
            Ws = g.sparse.tocsr().astype(np.float64)
        elif sp.issparse(W):
            Ws = _row_standardize_sparse(W)
            g = Graph.from_sparse(Ws.tocoo())
        else:
            raise TypeError(
                "W must be a libpysal.graph.Graph or a scipy sparse matrix, "
                f"got {type(W).__name__}."
            )
        if gdf is not None and Ws.shape[0] != len(gdf):
            raise ValueError(
                "W and gdf must describe the same number of spatial units."
            )
        _check_n_side(n_side, Ws.shape[0])
        return Ws, g

    if gdf is None:
        if n_side is None:
            raise ValueError("Provide either W, gdf, or n_side.")
        return rook_grid_weights(int(n_side))

    g = weights_from_geodataframe(
        gdf, contiguity=contiguity, k=k, distance_threshold=distance_threshold
    )
    _check_n_side(n_side, g.n_nodes)
    return g.sparse.tocsr().astype(np.float64), g


def _check_n_side(n_side: int | None, n_units: int) -> None:
    """Raise if ``n_side`` was given and ``n_side**2`` disagrees with ``n_units``."""
    if n_side is not None and int(n_side) ** 2 != n_units:
        raise ValueError(
            f"n_side={n_side} implies {int(n_side) ** 2} units, but W/gdf has "
            f"{n_units}."
        )


def spatial_filter_factor(W: sp.spmatrix, coef: float) -> Callable:
    """Factor ``I - coef * W`` with SuiteSparse; returns a ``solve(rhs)`` callable.

    Routing follows the samplers' :func:`make_sar_solver`: CHOLMOD on the
    D-symmetrized twin for undirected (D-symmetrizable) W, KLU for directed W
    when sparsax is installed, else CHOLMOD on the normal equations.
    ``rhs`` may be a vector or an ``(n, m)`` matrix, which CHOLMOD solves with
    blocked kernels, so panel and flow simulators factor once and solve many
    right-hand sides together.
    """
    from ..samplers._utils._spatial_normal import CholmodFactor
    from ..samplers.negbin_reduced._core import (
        _make_cholmod_pattern,
        make_sar_solver,
    )

    W_csc = sp.csc_matrix(W, dtype=np.float64)
    n = W_csc.shape[0]
    W_sym, WtW, pattern = _make_cholmod_pattern(W_csc, n)
    solver = make_sar_solver(CholmodFactor(pattern), W_csc, W_sym, WtW, n)
    coef = float(coef)
    solver.factorize(coef)

    def solve(rhs):
        x = np.asarray(solver.solve(rhs))
        # CHOLMOD does not always raise on a singular I − coef·W; it can return
        # a finite but meaningless answer.  Check the residual instead.
        resid = x - coef * (W_csc @ x) - rhs
        scale = max(float(np.max(np.abs(rhs))), 1e-300)
        if not np.all(np.isfinite(x)) or float(np.max(np.abs(resid))) > 1e-6 * scale:
            raise ValueError(
                f"I - {coef:g}*W is singular or too ill-conditioned to solve."
            )
        return x

    return solve


def _hetero_scale(X: np.ndarray, sigma: float) -> np.ndarray:
    """Compute observation-specific standard deviations for heteroskedastic errors.

    When ``err_hetero=True`` in a DGP simulator, each observation's error
    standard deviation is scaled by the norm of its regressor row so that
    units with larger covariate values receive noisier shocks.

    Parameters
    ----------
    X : np.ndarray
        Design matrix of shape ``(n, k)`` (including intercept column if
        applicable).
    sigma : float
        Base innovation standard deviation.

    Returns
    -------
    np.ndarray
        Array of shape ``(n,)`` with element-wise standard deviations
        ``sigma * sqrt(1 + ||x_i||^2)``.
    """
    return sigma * np.sqrt(1.0 + np.sum(X**2, axis=1))


def make_design_matrix(
    rng: np.random.Generator, n: int, k: int = 1, add_intercept: bool = True
) -> np.ndarray:
    """Generate synthetic design matrix.

    Parameters
    ----------
    rng : np.random.Generator
        Random generator.
    n : int
        Number of observations.
    k : int, default=1
        Number of non-intercept regressors.
    add_intercept : bool, default=True
        Whether to prepend a constant column.

    Returns
    -------
    np.ndarray
        Design matrix of shape ``(n, k + int(add_intercept))``.
    """
    Z = rng.standard_normal((n, k))
    if add_intercept:
        return np.column_stack([np.ones(n), Z])
    return Z


def panel_index(N: int, T: int) -> pd.DataFrame:
    """Create time-first panel index DataFrame.

    Parameters
    ----------
    N : int
        Number of units.
    T : int
        Number of periods.

    Returns
    -------
    pd.DataFrame
        Columns ``unit`` and ``time`` matching stacked time-first ordering.
    """
    units = np.tile(np.arange(N), T)
    times = np.repeat(np.arange(T), N)
    return pd.DataFrame({"unit": units, "time": times})


def make_output_geodataframe(
    y: np.ndarray,
    X: np.ndarray,
    gdf: Any | None = None,
    geometry_type: str = "polygon",
) -> Any:
    """Create a GeoDataFrame carrying simulated ``y`` and ``X`` columns.

    Parameters
    ----------
    y : np.ndarray
        Simulated dependent variable of shape ``(n_obs,)``.
    X : np.ndarray
        Simulated design matrix of shape ``(n_obs, k)``.
    gdf : geopandas.GeoDataFrame, optional
        Existing geometry source. If provided, its geometry is reused.
    geometry_type : {"point", "polygon"}, default="polygon"
        Geometry to generate when ``gdf`` is not provided.

    Returns
    -------
    geopandas.GeoDataFrame
        GeoDataFrame with columns ``y``, ``X_0``, ``X_1``, ... and geometry.
    """
    try:
        import geopandas as gpd
        from shapely.geometry import Point, box
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "create_gdf=True requires optional dependencies geopandas and shapely."
        ) from exc

    y = np.asarray(y, dtype=float)
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError("X must be a 2D array when building output GeoDataFrame.")
    if y.ndim != 1:
        raise ValueError("y must be a 1D array when building output GeoDataFrame.")
    if y.shape[0] != X.shape[0]:
        raise ValueError("y and X must have the same number of observations.")

    n_obs = y.shape[0]
    out = {"y": y}
    for j in range(X.shape[1]):
        out[f"X_{j}"] = X[:, j]

    if gdf is not None:
        if len(gdf) != n_obs:
            raise ValueError(
                "Provided gdf must have the same number of rows as y and X."
            )
        out_gdf = gdf.copy()
        for col, values in out.items():
            out_gdf[col] = values
        return out_gdf

    mode = str(geometry_type).lower()
    if mode not in {"point", "polygon"}:
        raise ValueError("geometry_type must be one of {'point', 'polygon'}.")

    n_cols = int(np.ceil(np.sqrt(n_obs)))
    int(np.ceil(n_obs / n_cols))

    geoms = []
    for idx in range(n_obs):
        r, c = divmod(idx, n_cols)
        if mode == "point":
            geoms.append(Point(c + 0.5, r + 0.5))
        else:
            geoms.append(box(c, r, c + 1.0, r + 1.0))

    return gpd.GeoDataFrame(out, geometry=geoms)


def make_panel_output_geodataframe(
    y: np.ndarray,
    X: np.ndarray,
    unit: np.ndarray,
    time: np.ndarray,
    N: int,
    T: int,
    *,
    gdf: Any | None = None,
    geometry_type: str = "polygon",
    wide: bool = False,
) -> Any:
    """Create panel GeoDataFrame output from simulated arrays.

    Parameters
    ----------
    y : np.ndarray
        Stacked dependent variable of shape ``(N*T,)``.
    X : np.ndarray
        Stacked design matrix of shape ``(N*T, k)``.
    unit : np.ndarray
        Unit index of shape ``(N*T,)``.
    time : np.ndarray
        Time index of shape ``(N*T,)``.
    N : int
        Number of spatial units.
    T : int
        Number of time periods.
    gdf : geopandas.GeoDataFrame, optional
        N-row geometry source. If provided its geometry is reused; any
        non-geometry columns are dropped before merging.
    geometry_type : {"point", "polygon"}, default="polygon"
        Geometry type to generate when ``gdf`` is not provided.
    wide : bool, default=False
        If True return a single N-row wide GeoDataFrame with columns
        ``y_t0``, ``y_t1``, ..., ``X_0_t0``, ``X_0_t1``, ...
        If False return ``(unit_gdf, long_panel_df)`` where ``unit_gdf``
        carries geometry only and ``long_panel_df`` carries ``unit``,
        ``time``, ``y``, ``X_0``, ...

    Returns
    -------
    GeoDataFrame or tuple[GeoDataFrame, DataFrame]
        Wide GeoDataFrame when ``wide=True``; otherwise
        ``(unit_gdf, long_panel_df)`` 2-tuple.
    """
    try:
        import geopandas as gpd
        from shapely.geometry import Point, box
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "create_gdf=True requires optional dependencies geopandas and shapely."
        ) from exc

    y = np.asarray(y, dtype=float)
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError("X must be a 2D array.")
    if y.ndim != 1:
        raise ValueError("y must be a 1D array.")
    if y.shape[0] != N * T:
        raise ValueError(f"y length {y.shape[0]} does not match N*T={N * T}.")

    # Build N-row unit GeoDataFrame (geometry only).
    if gdf is not None:
        if len(gdf) != N:
            raise ValueError("Provided gdf must have N rows.")
        unit_gdf = gdf[[gdf.geometry.name]].copy()
    else:
        mode = str(geometry_type).lower()
        if mode not in {"point", "polygon"}:
            raise ValueError("geometry_type must be one of {'point', 'polygon'}.")
        n_cols = int(np.ceil(np.sqrt(N)))
        geoms = []
        for idx_i in range(N):
            r, c = divmod(idx_i, n_cols)
            if mode == "point":
                geoms.append(Point(c + 0.5, r + 0.5))
            else:
                geoms.append(box(c, r, c + 1.0, r + 1.0))
        unit_gdf = gpd.GeoDataFrame({"unit_id": np.arange(N)}, geometry=geoms)
        unit_gdf = unit_gdf[[unit_gdf.geometry.name]]

    unit_gdf = unit_gdf.reset_index(drop=True)

    # Build long panel DataFrame.
    long_data: dict[str, Any] = {"unit": unit, "time": time, "y": y}
    for j in range(X.shape[1]):
        long_data[f"X_{j}"] = X[:, j]
    long_df = pd.DataFrame(long_data)

    if not wide:
        return unit_gdf, long_df

    # Pivot to wide format: one row per unit.
    x_cols = [c for c in long_df.columns if c.startswith("X_")]
    value_cols = ["y"] + x_cols
    wide_df = long_df.pivot(index="unit", columns="time", values=value_cols)
    wide_df.columns = [f"{col}_t{t}" for col, t in wide_df.columns]
    wide_df = wide_df.reset_index(drop=True)

    combined = pd.concat([unit_gdf.reset_index(drop=True), wide_df], axis=1)
    return gpd.GeoDataFrame(combined, geometry=unit_gdf.geometry.name)


def synth_point_geodataframe(n: int) -> Any:
    """Synthesize a point GeoDataFrame on a √n × √n grid.

    Used by flow DGPs when the user does not supply a ``gdf`` so that a
    consistent set of centroid coordinates is available for distance
    computation.

    Parameters
    ----------
    n : int
        Number of points (rows) to generate.

    Returns
    -------
    geopandas.GeoDataFrame
        n-row GeoDataFrame whose ``geometry`` column contains
        ``Point(c + 0.5, r + 0.5)`` cells laid out row-major on a
        ``ceil(sqrt(n)) × ceil(sqrt(n))`` grid.
    """
    try:
        import geopandas as gpd
        from shapely.geometry import Point
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "synth_point_geodataframe requires optional dependencies "
            "geopandas and shapely."
        ) from exc

    n = int(n)
    if n <= 0:
        raise ValueError("n must be a positive integer.")

    n_cols = int(np.ceil(np.sqrt(n)))
    geoms = []
    for idx in range(n):
        r, c = divmod(idx, n_cols)
        geoms.append(Point(c + 0.5, r + 0.5))

    return gpd.GeoDataFrame({"unit_id": np.arange(n)}, geometry=geoms)


def pairwise_distance_matrix(gdf: Any) -> np.ndarray:
    """Compute the dense ``n × n`` pairwise Euclidean distance matrix from a
    GeoDataFrame.

    Polygons are downcast to centroids first; points are used directly.
    The resulting matrix has zero diagonal by construction and is symmetric.

    This is the dense-matrix complement to
    :meth:`libpysal.graph.Graph.build_distance_band`, which returns a
    thresholded sparse graph; flow regressions need the full pairwise
    distances to populate an O-D design column.

    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        Geometry source.  Polygons are reduced to centroids automatically.

    Returns
    -------
    np.ndarray
        ``(n, n)`` Euclidean distance matrix in the same units / CRS as
        ``gdf.geometry``.
    """
    from scipy.spatial.distance import cdist

    if gdf is None or not hasattr(gdf, "geometry"):
        raise TypeError("pairwise_distance_matrix requires a GeoDataFrame-like object.")

    geom = gdf.geometry
    # geopandas' .centroid is a no-op for points and yields polygon centroids
    # otherwise; explicitly access .x / .y to obtain coordinates.
    cents = geom.centroid
    coords = np.column_stack([cents.x.to_numpy(), cents.y.to_numpy()])
    return cdist(coords, coords)


def _resolve_flow_geometry(
    n: int | None = None,
    G: Graph | None = None,
    gdf: Any | None = None,
    knn_k: int = 4,
    default_n: int = 25,
) -> tuple[int, Graph, Any]:
    """Resolve ``(n, G, gdf)`` for flow DGPs from any combination of inputs.

    The flow DGPs need a row-standardized spatial graph *G* (size ``n``)
    and a point GeoDataFrame *gdf* (length ``n``) to derive a pairwise
    distance matrix.  Either, both, or neither of *G* and *gdf* may be
    supplied.  When *G* is missing it is constructed from *gdf* using
    KNN contiguity (``k = knn_k``) and row-standardized.  When *gdf* is
    missing a synthetic point grid is generated via
    :func:`synth_point_geodataframe`.

    Parameters
    ----------
    n : int, optional
        Desired number of spatial units.  Ignored when *G* or *gdf* is
        provided (the size is taken from those objects).  When neither
        is supplied, falls back to *default_n*.
    G : libpysal.graph.Graph, optional
        Pre-built spatial graph.  If supplied it is row-standardized on
        the way out (idempotent for already row-standard graphs).
    gdf : geopandas.GeoDataFrame, optional
        Point or polygon geometry.  Used both to build *G* (when
        missing) and to compute distances downstream.
    knn_k : int, default 4
        Number of nearest neighbors used when building *G* from *gdf*.
    default_n : int, default 25
        Number of units used when neither *G* nor *gdf* is provided.

    Returns
    -------
    n_actual : int
        Resolved number of spatial units.
    G : libpysal.graph.Graph
        Row-standardized graph on *n_actual* units.
    gdf : geopandas.GeoDataFrame
        Geometry source on *n_actual* units (synthesized when not
        supplied).

    Raises
    ------
    ValueError
        If both *G* and *gdf* are supplied but their sizes disagree, or
        if *n* is supplied alongside *G* / *gdf* with an inconsistent
        size.
    """
    if G is not None and gdf is not None:
        if len(gdf) != G.n_nodes:
            raise ValueError(f"gdf has {len(gdf)} rows but G has {G.n_nodes} nodes.")
        n_actual = G.n_nodes
    elif G is not None:
        n_actual = G.n_nodes
        gdf = synth_point_geodataframe(n_actual)
    elif gdf is not None:
        n_actual = len(gdf)
        k_eff = max(1, min(int(knn_k), n_actual - 1))
        G = Graph.build_knn(gdf, k=k_eff).transform("r")
    else:
        n_actual = int(n) if n is not None else int(default_n)
        gdf = synth_point_geodataframe(n_actual)
        k_eff = max(1, min(int(knn_k), n_actual - 1))
        G = Graph.build_knn(gdf, k=k_eff).transform("r")

    if n is not None and int(n) != n_actual:
        raise ValueError(f"n={n} disagrees with resolved geometry size {n_actual}.")

    # Ensure G is row-standardized
    G = G.transform("r")
    return n_actual, G, gdf


def _left_censor(
    y_latent: np.ndarray, censoring: float
) -> tuple[np.ndarray, np.ndarray]:
    """Left-censor ``y_latent`` at ``censoring``.

    Returns the observed vector (values ``<= censoring`` clamped to the
    threshold) and the boolean mask of censored observations.  Shared by the
    cross-sectional and panel Tobit DGPs.
    """
    mask = y_latent <= censoring
    y_obs = y_latent.copy()
    y_obs[mask] = censoring
    return y_obs, mask


def _maybe_geodataframe(
    *,
    y: np.ndarray,
    X: np.ndarray,
    idx: dict,
    N: int,
    T: int,
    Ws: sp.csr_matrix,
    Wg,
    params_true: dict,
    create_gdf: bool,
    gdf,
    geometry_type: str,
    wide: bool,
) -> dict:
    """Assemble a panel DGP output dict, optionally as a (Geo)DataFrame.

    Returns the plain long-format dict unless ``create_gdf``, an input ``gdf``,
    or ``wide`` is requested, in which case the output is materialized via
    :func:`make_panel_output_geodataframe`.  Shared by the panel FE and dynamic
    DGP families.
    """
    out = {
        "y": y,
        "X": X,
        "unit": idx["unit"],
        "time": idx["time"],
        "W_sparse": Ws,
        "W_graph": Wg,
        "params_true": params_true,
    }
    if create_gdf or gdf is not None or wide:
        return make_panel_output_geodataframe(
            y,
            X,
            idx["unit"],
            idx["time"],
            N,
            T,
            gdf=gdf,
            geometry_type=geometry_type,
            wide=wide,
        )
    return out
