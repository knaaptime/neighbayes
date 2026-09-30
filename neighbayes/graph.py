"""Kronecker-product spatial weight utilities for flow (O-D) models.

Provides helpers to build destination, origin, and network weight matrices
from a single n×n :class:`libpysal.graph.Graph` using Kronecker products,
following :cite:t:`lesage2008SpatialEconometric`.

The N×N flow weight matrices are:

.. math::

    W_d = I_n \\otimes W \\quad (\\text{destination effects})

    W_o = W \\otimes I_n \\quad (\\text{origin effects})

    W_w = W \\otimes W \\quad (\\text{network / origin-destination effects})

where :math:`N = n^2` is the total number of O-D pairs and :math:`n` is the
number of spatial units.

Examples
--------
>>> import libpysal
>>> G = libpysal.graph.Graph.build_contiguity(gdf).transform("r")
>>> W_d, W_o, W_w = flow_weight_matrices(G).values()
>>> dm = flow_design_matrix(X, dist=dist_matrix)
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
from libpysal.graph import Graph


def sparse_trace_WtW_plus_WW(W: sp.spmatrix) -> float:
    r"""Return :math:`\operatorname{tr}(W^\top W) + \operatorname{tr}(W^2)`.

    Computed via sparse element-wise operations
    (:math:`\|W\|_F^2 + \sum_{ij} W_{ij} W_{ji}`), which is :math:`O(\mathrm{nnz})`
    rather than :math:`O(n^2)`.  This quantity appears in many spatial
    Lagrange-multiplier information-matrix formulas.
    """
    return float(W.power(2).sum() + W.multiply(W.T).sum())


def flow_trace_blocks(W: sp.spmatrix) -> np.ndarray:
    r"""Return the symmetric :math:`3 \times 3` Kronecker trace matrix
    :math:`T_{ij} = \operatorname{tr}(W_i^\top W_j) + \operatorname{tr}(W_i W_j)`,
    indexed by the flow weight matrices :math:`(W_d, W_o, W_w)` with
    :math:`W_d = I_n \otimes W`, :math:`W_o = W \otimes I_n`,
    :math:`W_w = W \otimes W`.

    Used by Bayesian LM tests on flow models as the noise-free part of the
    information matrix block for the spatial-lag scores.  All entries are
    derived in :math:`O(\mathrm{nnz})` from three base traces of the
    :math:`n \times n` graph weight matrix
    (:math:`\operatorname{tr}(W)`, :math:`\operatorname{tr}(W^2)`,
    :math:`\operatorname{tr}(W^\top W)`) using
    :math:`\operatorname{tr}(A \otimes B) = \operatorname{tr}(A)\operatorname{tr}(B)`.

    Parameters
    ----------
    W : scipy.sparse.spmatrix
        Row-standardized :math:`n \times n` weight matrix on the base graph.

    Returns
    -------
    numpy.ndarray
        Dense ``(3, 3)`` array with rows/cols ordered as
        ``(destination, origin, network)``.
    """
    W = W.tocsr()
    n = W.shape[0]
    tr_W = float(W.diagonal().sum())
    tr_W2 = float(W.multiply(W.T).sum())  # tr(W^2) = sum_ij W_ij W_ji
    tr_WtW = float(W.power(2).sum())  # tr(W'W) = ||W||_F^2

    s = tr_WtW + tr_W2  # tr(W'W) + tr(W^2)
    T = np.empty((3, 3), dtype=np.float64)
    T[0, 0] = n * s
    T[1, 1] = n * s
    T[2, 2] = tr_WtW * tr_WtW + tr_W2 * tr_W2
    T[0, 1] = T[1, 0] = 2.0 * tr_W * tr_W
    T[0, 2] = T[2, 0] = tr_W * s
    T[1, 2] = T[2, 1] = tr_W * s
    return T


def _graph_to_csr(G: Graph) -> sp.csr_matrix:
    """Extract the CSR sparse matrix from a :class:`libpysal.graph.Graph`.

    Raises ``TypeError`` if *G* is not a Graph.
    """
    if not isinstance(G, Graph):
        raise TypeError(
            f"G must be a libpysal.graph.Graph, got {type(G).__name__}. "
            "Convert a legacy libpysal.weights.W with Graph.from_W(w)."
        )
    return G.sparse.tocsr().astype(np.float64)


def _weights_to_csr(W) -> sp.csr_matrix:
    """Coerce spatial weights to a CSR matrix, accepting a Graph or a matrix.

    Flow models take their ``n×n`` regional weights as ``W`` (mirroring the
    other spatial models); this helper accepts a :class:`libpysal.graph.Graph`
    (row-standardized) or a scipy-sparse / dense array and never densifies a
    sparse input.
    """
    if isinstance(W, Graph):
        return _graph_to_csr(W)
    if sp.issparse(W):
        return W.tocsr().astype(np.float64)
    arr = np.asarray(W, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        raise TypeError(
            "W must be a libpysal.graph.Graph or a square (n×n) weights matrix; "
            f"got array with shape {getattr(arr, 'shape', None)}."
        )
    return sp.csr_matrix(arr)


def destination_weights(G: Graph) -> sp.csr_matrix:
    """Build the N×N destination weight matrix :math:`W_d = I_n \\otimes W`.

    A non-zero entry at position :math:`(i \\cdot n + j,\\; i \\cdot n + k)`
    indicates that destination *k* is a spatial neighbor of destination *j*
    (for flows originating from *i*).

    Parameters
    ----------
    G :
        Row-standardized :class:`libpysal.graph.Graph` of shape *n×n*.

    Returns
    -------
    scipy.sparse.csr_matrix
        :math:`N \\times N` destination weight matrix (:math:`N = n^2`).
    """
    W = _weights_to_csr(G)
    n = W.shape[0]
    return sp.kron(sp.eye(n, format="csr"), W, format="csr")


def origin_weights(G: Graph) -> sp.csr_matrix:
    """Build the N×N origin weight matrix :math:`W_o = W \\otimes I_n`.

    A non-zero entry at position :math:`(i \\cdot n + j,\\; k \\cdot n + j)`
    indicates that origin *k* is a spatial neighbor of origin *i*
    (for flows terminating at *j*).

    Parameters
    ----------
    G :
        Row-standardized :class:`libpysal.graph.Graph` of shape *n×n*.

    Returns
    -------
    scipy.sparse.csr_matrix
        :math:`N \\times N` origin weight matrix (:math:`N = n^2`).
    """
    W = _weights_to_csr(G)
    n = W.shape[0]
    return sp.kron(W, sp.eye(n, format="csr"), format="csr")


def network_weights(G: Graph) -> sp.csr_matrix:
    """Build the N×N network weight matrix :math:`W_w = W \\otimes W`.

    Captures joint origin-destination neighborhood structure: a non-zero
    entry at :math:`(i \\cdot n + j,\\; k \\cdot n + l)` indicates that
    origin *k* neighbors *i* AND destination *l* neighbors *j*.

    Parameters
    ----------
    G :
        Row-standardized :class:`libpysal.graph.Graph` of shape *n×n*.

    Returns
    -------
    scipy.sparse.csr_matrix
        :math:`N \\times N` network weight matrix (:math:`N = n^2`).
    """
    W = _weights_to_csr(G)
    return sp.kron(W, W, format="csr")


def flow_weight_matrices(G: Graph) -> dict[str, sp.csr_matrix]:
    """Build all three N×N flow weight matrices from a single Graph.

    Parameters
    ----------
    G :
        Row-standardized :class:`libpysal.graph.Graph` of shape *n×n*.

    Returns
    -------
    dict[str, scipy.sparse.csr_matrix]
        Dictionary with keys ``"destination"``, ``"origin"``, ``"network"``.
    """
    W = _weights_to_csr(G)
    n = W.shape[0]
    eye = sp.eye(n, format="csr")
    return {
        "destination": sp.kron(eye, W, format="csr"),
        "origin": sp.kron(W, eye, format="csr"),
        "network": sp.kron(W, W, format="csr"),
    }


def flow_lags(G, V, T: int = 1) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(W_d V, W_o V, W_w V)`` without forming the ``N × N`` Kronecker matrices.

    With ``N = n²`` pairs, ``W_d = I⊗W``, ``W_o = W⊗I`` and ``W_w = W⊗W``.  On
    the flow array ``Y`` (origins × destinations, the row-major order of the
    flow vec) these are ``YWᵀ``, ``WY`` and ``WYWᵀ``: a few ``n × n`` sparse
    products instead of matrices with up to ``nnz(W)²`` entries.

    Parameters
    ----------
    G : Graph or sparse matrix
        The ``n × n`` spatial weights.
    V : ndarray, shape ``(T·N,)`` or ``(T·N, k)``
        Flow vector(s); for panels, periods stacked time-first.
    T : int, default 1
        Number of stacked periods (each lagged separately).

    Returns
    -------
    tuple of ndarray
        Destination, origin and network lags, each shaped like ``V``.
    """
    W = _weights_to_csr(G)
    n = W.shape[0]
    V = np.asarray(V, dtype=np.float64)
    shape = V.shape
    k = int(np.prod(shape[1:])) if V.ndim > 1 else 1
    # (T, o, d, k) → origins first, all other axes flattened into columns.
    Y = V.reshape(T, n, n, k).transpose(1, 0, 2, 3).reshape(n, -1)  # [o, (t,d,c)]
    WY = (W @ Y).reshape(n, T, n, k)  # W acting on origins

    def _right(Z):  # Z[o, t, d, c] → Σ_d' W[d, d'] Z[o, t, d', c]
        Zd = Z.transpose(2, 0, 1, 3).reshape(n, -1)
        return (W @ Zd).reshape(n, n, T, k).transpose(1, 2, 0, 3)

    Y4 = Y.reshape(n, T, n, k)
    lag_d = _right(Y4)
    lag_o = WY
    lag_w = _right(WY)

    def _back(Z):
        return Z.transpose(1, 0, 2, 3).reshape(shape)

    return _back(lag_d), _back(lag_o), _back(lag_w)


@dataclass
class FlowDesignMatrix:
    """Combined design matrix for an O-D flow regression.

    Attributes
    ----------
    X_dest : np.ndarray, shape (N, k_d)
        Destination-side characteristics: :math:`\\iota_n \\otimes X_d`.
    X_orig : np.ndarray, shape (N, k_o)
        Origin-side characteristics: :math:`X_o \\otimes \\iota_n`.
    X_intra : np.ndarray, shape (N, k_d)
        Intra-zonal characteristics (non-zero only on the diagonal of the
        flow matrix): rows of *X_d* selected by ``vec(I_n)``.
        Uses destination-side columns because
        :func:`flow_design_matrix` constructs
        ``X_intra = intra_indicator * X_dest``.
    intra_indicator : np.ndarray, shape (N,)
        Binary indicator for diagonal (intra-zonal) O-D pairs.
    dist_vec : np.ndarray or None, shape (N,)
        Vectorized distance/cost matrix ``vec(dist)`` if provided.
    combined : np.ndarray, shape (N, p)
        Full design matrix ready for regression.  Column order:
        intercept | intra_indicator | X_dest (k_d cols) | X_orig (k_o cols) |
        X_intra (k_d cols) [| dist (1 col if provided)].
    feature_names : list[str]
        Column labels for *combined* aligned with ``beta`` in the flow model.
    n : int
        Number of spatial units (*n*).  Flow count is :math:`N = n^2`.
    k_d : int
        Number of destination-side attribute columns.
    k_o : int
        Number of origin-side attribute columns.
    k : int
        Alias for ``k_d`` for backward compatibility.  Equal to ``k_d``.
    """

    X_dest: np.ndarray
    X_orig: np.ndarray
    X_intra: np.ndarray
    intra_indicator: np.ndarray
    dist_vec: np.ndarray | None
    combined: np.ndarray
    feature_names: list
    n: int
    k_d: int
    k_o: int
    k: int


def flow_design_matrix(
    X: np.ndarray,
    col_names: list[str] | None = None,
    dist: np.ndarray | None = None,
    log_distance: bool = True,
) -> FlowDesignMatrix:
    """Build a flow regression design matrix from regional attribute data.

    Constructs the standard LeSage-Fischer O-D design matrix with separate
    destination, origin, and intra-zonal blocks, plus an optional distance
    column, following :cite:t:`lesage2008SpatialEconometric` (Section 4.2).

    Parameters
    ----------
    X : np.ndarray, shape (n, k)
        Array of *k* regional attributes for *n* spatial units.
        Should **not** include an intercept column.
    col_names : list[str], optional
        Names for the *k* columns of *X*.  Defaults to ``["x0", "x1", ...]``.
    dist : np.ndarray, shape (n, n), optional
        Distance or cost matrix.  If provided, ``vec(dist)`` is appended as
        the last column of *combined*.
    log_distance : bool, default True
        If True and ``dist`` is provided, the appended column is
        ``log(1 + dist).ravel()`` and is named ``"log_distance"``.  If
        False, the raw distance vector is appended and named ``"dist"``.
        Using ``log(1 + d)`` matches the gravity-model convention while
        keeping the diagonal at zero.

    Returns
    -------
    FlowDesignMatrix
        Dataclass with all sub-matrices and a combined design matrix.

    Notes
    -----
    The full beta vector is structured as:

    .. math::

        \\beta = [\\alpha,\\; \\alpha_i,\\; \\beta_d^1 \\ldots \\beta_d^k,\\;
                 \\beta_o^1 \\ldots \\beta_o^k,\\; \\beta_i^1 \\ldots \\beta_i^k
                 [,\\; \\gamma]]

    matching ``f2_sarfm.m`` from the LeSage spatial flows toolbox.

    The leading **intercept** column is always added because flow models
    are typically estimated on log-flow outcomes whose grand mean is
    informative; omitting it would force the destination/origin/intra
    blocks to absorb the global level and complicates effects
    decomposition.  Users wishing to suppress the intercept should drop
    the first column of ``combined`` *and* the corresponding row/column
    of any prior covariance.

    Examples
    --------
    Build a flow design from a 3-region attribute matrix with population
    and income:

    >>> import numpy as np
    >>> X = np.array([[100.0, 50.0],     # region 0: pop=100, inc=50
    ...               [200.0, 75.0],     # region 1: pop=200, inc=75
    ...               [150.0, 60.0]])    # region 2: pop=150, inc=60
    >>> design = flow_design_matrix(X, col_names=["pop", "inc"])
    >>> design.combined.shape  # 3*3 = 9 OD pairs, 1+1+2+2+2 = 8 cols
    (9, 8)
    >>> design.feature_names[:4]
    ['intercept', 'intra_indicator', 'dest_pop', 'dest_inc']

    Optionally append a vectorized distance matrix as a final column:

    >>> dist = np.array([[0.0, 10.0, 5.0],
    ...                  [10.0, 0.0, 8.0],
    ...                  [5.0, 8.0, 0.0]])
    >>> design_d = flow_design_matrix(X, col_names=["pop", "inc"], dist=dist)
    >>> design_d.feature_names[-1]
    'dist'
    """
    X_arr = np.asarray(X, dtype=np.float64)
    if X_arr.ndim == 1:
        X_arr = X_arr[:, None]
    n, k = X_arr.shape

    # Warn on zero-variance columns: they are perfectly collinear with
    # the intercept and would inflate the design's condition number.
    if k > 0:
        col_var = X_arr.var(axis=0)
        zero_var = np.where(col_var == 0.0)[0]
        if zero_var.size > 0:
            warnings.warn(
                f"flow_design_matrix received zero-variance column(s) at "
                f"indices {zero_var.tolist()}; these are collinear with the "
                "intercept block and will inflate the condition number of "
                "`combined`.",
                stacklevel=2,
            )

    N = n * n
    ones_n = np.ones((n, 1), dtype=np.float64)

    # Intra-zonal selector: vec(I_n), length N
    intra_indicator = np.eye(n, dtype=np.float64).ravel()  # 1 on diagonal cells

    # Destination characteristics: kron(ones_n, X) — each row repeats its destination X
    X_dest = np.kron(ones_n, X_arr)  # (N, k)

    # Origin characteristics: kron(X, ones_n) — each row repeats its origin X
    X_orig = np.kron(X_arr, ones_n)  # (N, k)

    # Intra-zonal characteristics: non-zero only on diagonal cells
    X_intra = intra_indicator[:, None] * X_dest  # (N, k); zeros off-diagonal

    # Intercept column
    intercept = np.ones((N, 1), dtype=np.float64)

    # Assemble combined design matrix: [1, ia, X_dest, X_orig, X_intra, (dist)]
    parts = [intercept, intra_indicator[:, None], X_dest, X_orig, X_intra]
    names = ["intercept", "intra_indicator"]

    if col_names is None:
        col_names = [f"x{i}" for i in range(k)]
    names += [f"dest_{c}" for c in col_names]
    names += [f"orig_{c}" for c in col_names]
    names += [f"intra_{c}" for c in col_names]

    dist_vec: np.ndarray | None = None
    if dist is not None:
        dist_arr = np.asarray(dist, dtype=np.float64)
        if dist_arr.shape != (n, n):
            raise ValueError(
                f"dist must have shape ({n}, {n}) to match X, got {dist_arr.shape}."
            )
        if log_distance:
            dist_vec = np.log1p(dist_arr).ravel()
            dist_name = "log_distance"
        else:
            dist_vec = dist_arr.ravel()
            dist_name = "dist"
        parts.append(dist_vec[:, None])
        names.append(dist_name)

    combined = np.concatenate(parts, axis=1)

    return FlowDesignMatrix(
        X_dest=X_dest,
        X_orig=X_orig,
        X_intra=X_intra,
        intra_indicator=intra_indicator,
        dist_vec=dist_vec,
        combined=combined,
        feature_names=names,
        n=n,
        k_d=k,
        k_o=k,
        k=k,
    )


def flow_design_matrix_asymmetric(
    Xd: np.ndarray,
    Xo: np.ndarray,
    col_names_d: list[str] | None = None,
    col_names_o: list[str] | None = None,
    dist: np.ndarray | None = None,
    log_distance: bool = True,
) -> FlowDesignMatrix:
    """Build a flow design matrix with different numbers of dest/origin variables.

    Unlike :func:`flow_design_matrix` (which uses a single ``(n, k)`` matrix
    for both destination and origin blocks), this function accepts separate
    attribute matrices *Xd* of shape ``(n, k_d)`` and *Xo* of shape
    ``(n, k_o)`` where ``k_d`` and ``k_o`` may differ.

    The design matrix column layout is:

    ``intercept | intra_indicator | X_dest (k_d cols) | X_orig (k_o cols) | X_intra (k_d cols) [| dist]``

    The intra-zonal block uses ``k_d`` columns (destination-side variables),
    matching the LeSage convention where ``X_intra = intra_indicator * X_dest``.

    Parameters
    ----------
    Xd : np.ndarray, shape (n, k_d)
        Destination-side regional attribute matrix (no intercept).
    Xo : np.ndarray, shape (n, k_o)
        Origin-side regional attribute matrix (no intercept).
        ``k_o`` may differ from ``k_d``.
    col_names_d : list[str], optional
        Names for the *k_d* destination columns.  Defaults to
        ``["x0", "x1", ...]``.
    col_names_o : list[str], optional
        Names for the *k_o* origin columns.  Defaults to
        ``["y0", "y1", ...]`` when ``k_o != k_d``, or the same names as
        *col_names_d* when ``k_o == k_d``.
    dist : np.ndarray, shape (n, n), optional
        Distance / cost matrix appended as the last predictor.
    log_distance : bool, default True
        If True and ``dist`` is provided, the appended column is
        ``log(1 + dist).ravel()`` and is named ``"log_distance"``.

    Returns
    -------
    FlowDesignMatrix
        Dataclass with ``k_d`` and ``k_o`` set independently.

    Examples
    --------
    >>> import numpy as np
    >>> n = 3
    >>> Xd = np.ones((n, 2))   # 2 destination variables
    >>> Xo = 2 * np.ones((n, 1))  # 1 origin variable
    >>> dm = flow_design_matrix_asymmetric(Xd, Xo)
    >>> dm.combined.shape  # 9 OD pairs, 1+1+2+1+2 = 7 cols
    (9, 7)
    >>> dm.k_d, dm.k_o
    (2, 1)
    >>> dm.feature_names
    ['intercept', 'intra_indicator', 'dest_x0', 'dest_x1', 'orig_y0', 'intra_x0', 'intra_x1']
    """
    Xd_arr = np.asarray(Xd, dtype=np.float64)
    Xo_arr = np.asarray(Xo, dtype=np.float64)
    if Xd_arr.ndim == 1:
        Xd_arr = Xd_arr[:, None]
    if Xo_arr.ndim == 1:
        Xo_arr = Xo_arr[:, None]

    n = Xd_arr.shape[0]
    k_d = Xd_arr.shape[1]
    k_o = Xo_arr.shape[1]

    if Xo_arr.shape[0] != n:
        raise ValueError(
            f"Xd and Xo must have the same number of rows, "
            f"got {Xd_arr.shape[0]} and {Xo_arr.shape[0]}."
        )

    N = n * n
    ones_n = np.ones((n, 1), dtype=np.float64)

    # Intra-zonal selector: vec(I_n), length N
    intra_indicator = np.eye(n, dtype=np.float64).ravel()

    # Destination characteristics: kron(ones_n, Xd)
    X_dest = np.kron(ones_n, Xd_arr)  # (N, k_d)

    # Origin characteristics: kron(Xo, ones_n)
    X_orig = np.kron(Xo_arr, ones_n)  # (N, k_o)

    # Intra-zonal characteristics: non-zero only on diagonal cells (uses Xd)
    X_intra = intra_indicator[:, None] * X_dest  # (N, k_d)

    # Intercept column
    intercept = np.ones((N, 1), dtype=np.float64)

    # Column names
    if col_names_d is None:
        col_names_d = [f"x{i}" for i in range(k_d)]
    if col_names_o is None:
        if k_o == k_d and k_o <= len(col_names_d):
            col_names_o = list(col_names_d[:k_o])
        else:
            col_names_o = [f"y{i}" for i in range(k_o)]

    # Assemble combined design matrix: [1, ia, X_dest, X_orig, X_intra, (dist)]
    parts = [intercept, intra_indicator[:, None], X_dest, X_orig, X_intra]
    names = ["intercept", "intra_indicator"]
    names += [f"dest_{c}" for c in col_names_d]
    names += [f"orig_{c}" for c in col_names_o]
    names += [f"intra_{c}" for c in col_names_d]

    dist_vec: np.ndarray | None = None
    if dist is not None:
        dist_arr = np.asarray(dist, dtype=np.float64)
        if dist_arr.shape != (n, n):
            raise ValueError(
                f"dist must have shape ({n}, {n}) to match X, got {dist_arr.shape}."
            )
        if log_distance:
            dist_vec = np.log1p(dist_arr).ravel()
            dist_name = "log_distance"
        else:
            dist_vec = dist_arr.ravel()
            dist_name = "dist"
        parts.append(dist_vec[:, None])
        names.append(dist_name)

    combined = np.concatenate(parts, axis=1)

    return FlowDesignMatrix(
        X_dest=X_dest,
        X_orig=X_orig,
        X_intra=X_intra,
        intra_indicator=intra_indicator,
        dist_vec=dist_vec,
        combined=combined,
        feature_names=names,
        n=n,
        k_d=k_d,
        k_o=k_o,
        k=k_d,
    )


def flow_design_matrix_with_orig(
    Xd: np.ndarray,
    Xo: np.ndarray,
    col_names: list[str] | None = None,
    dist: np.ndarray | None = None,
    log_distance: bool = True,
) -> FlowDesignMatrix:
    """Build a flow design matrix with separate destination and origin data.

    This is a convenience wrapper around :func:`flow_design_matrix` that
    handles the common pattern of building a design matrix from destination
    attributes *Xd* and then splicing in separate origin attributes *Xo*.

    When ``k_d == k_o`` (same number of destination and origin columns),
    this delegates to :func:`flow_design_matrix` and splices in the origin
    block.  When ``k_d != k_o``, it delegates to
    :func:`flow_design_matrix_asymmetric`.

    Parameters
    ----------
    Xd : np.ndarray, shape (n, k_d)
        Destination-side regional attribute matrix (no intercept).
    Xo : np.ndarray, shape (n, k_o)
        Origin-side regional attribute matrix (no intercept).
        ``k_o`` may differ from ``k_d``.
    col_names : list[str], optional
        Names for the destination columns.  Defaults to
        ``["x0", "x1", ...]``.  When ``k_d != k_o``, origin columns
        default to ``["y0", "y1", ...]``.
    dist : np.ndarray, shape (n, n), optional
        Distance / cost matrix appended as the last predictor.
    log_distance : bool, default True
        If True and ``dist`` is provided, the appended column is
        ``log(1 + dist).ravel()`` and is named ``"log_distance"``.

    Returns
    -------
    FlowDesignMatrix
        Dataclass with ``X_orig`` and ``combined`` updated to reflect *Xo*.

    Raises
    ------
    ValueError
        If *Xd* and *Xo* have incompatible row counts.

    Examples
    --------
    >>> import numpy as np
    >>> n, k = 4, 2
    >>> Xd = np.ones((n, k))
    >>> Xo = 2 * np.ones((n, k))
    >>> dm = flow_design_matrix_with_orig(Xd, Xo)
    >>> dm.X_orig.shape
    (16, 2)
    >>> np.allclose(dm.X_orig[:, 0], 2.0)
    True
    """
    Xd_arr = np.asarray(Xd, dtype=np.float64)
    Xo_arr = np.asarray(Xo, dtype=np.float64)
    if Xd_arr.ndim == 1:
        Xd_arr = Xd_arr[:, None]
    if Xo_arr.ndim == 1:
        Xo_arr = Xo_arr[:, None]

    k_d = Xd_arr.shape[1]
    k_o = Xo_arr.shape[1]
    n = Xd_arr.shape[0]

    if k_d == k_o:
        # Symmetric case: delegate to flow_design_matrix and splice
        design = flow_design_matrix(
            Xd_arr, col_names=col_names, dist=dist, log_distance=log_distance
        )

        # Override the X_orig block with actual Xo: kron(Xo, ones_n)
        ones_n = np.ones((n, 1), dtype=np.float64)
        X_orig_actual = np.kron(Xo_arr, ones_n)  # (N, k_o)

        # Splice into combined: intercept | ia | Xd_block | Xo_block | Xi_block [| dist]
        combined = design.combined.copy()
        combined[:, 2 + k_d : 2 + k_d + k_o] = X_orig_actual

        return FlowDesignMatrix(
            X_dest=design.X_dest,
            X_orig=X_orig_actual,
            X_intra=design.X_intra,
            intra_indicator=design.intra_indicator,
            dist_vec=design.dist_vec,
            combined=combined,
            feature_names=design.feature_names,
            n=design.n,
            k_d=k_d,
            k_o=k_o,
            k=k_d,
        )
    else:
        # Asymmetric case: k_d != k_o
        return flow_design_matrix_asymmetric(
            Xd_arr,
            Xo_arr,
            col_names_d=col_names,
            col_names_o=None,
            dist=dist,
            log_distance=log_distance,
        )
