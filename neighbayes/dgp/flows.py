"""Data-generating process for Bayesian spatial flow (O-D) models.

Simulates origin-destination flow data from the SAR flow model:

.. math::

    y = (I_N - \\rho_d W_d - \\rho_o W_o - \\rho_w W_w)^{-1}
        (X\\beta + \\varepsilon), \\quad \\varepsilon \\sim \\mathcal{N}(0, \\sigma^2 I_N)
"""

from __future__ import annotations

import warnings
from typing import Optional, Union

import numpy as np
import scipy.sparse.linalg as spla
from libpysal.graph import Graph

from ..graph import (
    _graph_to_csr,
    flow_design_matrix,
    flow_design_matrix_asymmetric,
    flow_design_matrix_with_orig,
)
from .utils import (
    _resolve_flow_geometry,
    pairwise_distance_matrix,
    spatial_filter_factor,
)

# ---------------------------------------------------------------------------
# Shared building blocks
#
# The twelve public generators share the flow-filter assembly, the singular-
# matrix error handling, the beta-vector layout, and a couple of drawing
# blocks.  Everything else (parameter inference rules, design-matrix choice,
# noise placement, observation family, return schema — and, critically, the
# per-generator rng draw order) is generator-specific and stays explicit in
# each function.  None of these helpers consumes rng draws except
# ``_hetero_flow_eps``, which performs the exact draw the callers previously
# made inline.
# ---------------------------------------------------------------------------


class _FlowFilter:
    """The N×N flow filter ``I_N − r_d W_d − r_o W_o − r_w W_w``, never materialized.

    With ``N = n²`` pairs, ``W_d = I⊗W``, ``W_o = W⊗I`` and ``W_w = W⊗W``.  On
    the flow array ``Y = v.reshape(n, n)`` (origins × destinations, the
    row-major order of those Kronecker products) the three terms are ``YWᵀ``,
    ``WY`` and ``WYWᵀ``, so the operator costs a few n-sized sparse products
    and ``W⊗W`` — ``nnz(W)²`` entries — is never built.

    Solves are exact for the separable filter ``r_w = −r_d·r_o``, which
    factors as ``(I − r_o W) ⊗ (I − r_d W)``: two n×n SuiteSparse factors
    (see :func:`spatial_filter_factor`), with
    ``Y = B_o⁻¹ R B_d⁻ᵀ``.  Otherwise GMRES on the matrix-free operator,
    preconditioned by that separable solve.
    """

    def __init__(
        self,
        G: Graph,
        r_d: float,
        r_o: float,
        r_w: float,
        letter: str = "A",
        prefix: str = "rho",
    ):
        self.W = _graph_to_csr(G)
        self.n = self.W.shape[0]
        self.WT = self.W.T.tocsr()
        self.r_d, self.r_o, self.r_w = float(r_d), float(r_o), float(r_w)
        self.letter, self.prefix = letter, prefix
        self.separable = abs(self.r_w + self.r_d * self.r_o) <= 1e-12 * max(
            1.0, abs(self.r_w)
        )
        try:
            self._solve_o = spatial_filter_factor(self.W, self.r_o)
            self._solve_d = spatial_filter_factor(self.W, self.r_d)
        except Exception as exc:
            raise ValueError(_singular_flow_message(letter, prefix)) from exc

    def _right_W(self, Y: np.ndarray) -> np.ndarray:
        """``Y Wᵀ`` for dense Y, as ``(W Yᵀ)ᵀ`` so W stays on the sparse side."""
        return (self.W @ Y.T).T

    def matvec(self, v: np.ndarray) -> np.ndarray:
        Y = np.asarray(v, dtype=np.float64).reshape(self.n, self.n)
        WY = self.W @ Y
        out = Y - self.r_d * self._right_W(Y) - self.r_o * WY
        if self.r_w:
            out = out - self.r_w * self._right_W(WY)
        return out.ravel()

    def _separable_solve(self, v: np.ndarray) -> np.ndarray:
        R = np.asarray(v, dtype=np.float64).reshape(self.n, self.n)
        Z = self._solve_o(R)  # B_o⁻¹ R
        return np.ascontiguousarray(self._solve_d(Z.T).T).ravel()  # Z B_d⁻ᵀ

    def solve(self, rhs: np.ndarray) -> np.ndarray:
        rhs = np.asarray(rhs, dtype=np.float64)
        if self.separable:
            return self._separable_solve(rhs)
        N = self.n * self.n
        A = spla.LinearOperator((N, N), matvec=self.matvec, dtype=np.float64)
        M = spla.LinearOperator((N, N), matvec=self._separable_solve, dtype=np.float64)
        x, info = spla.gmres(
            A,
            rhs,
            x0=self._separable_solve(rhs),
            M=M,
            rtol=1e-10,
            atol=0.0,
            restart=10,
            maxiter=100,
        )
        if info != 0:
            raise ValueError(_singular_flow_message(self.letter, self.prefix))
        return x


def _flow_system(
    G: Graph, r_d: float, r_o: float, r_w: float, letter: str = "A", prefix: str = "rho"
) -> _FlowFilter:
    """The flow filter ``I_N - r_d Wd - r_o Wo - r_w Ww`` as a matrix-free operator."""
    return _FlowFilter(G, r_d, r_o, r_w, letter=letter, prefix=prefix)


def _singular_flow_message(letter: str, prefix: str) -> str:
    return (
        f"{letter} = I_N - {prefix}_d*Wd - {prefix}_o*Wo - {prefix}_w*Ww is singular. "
        f"Check that {prefix}_d + {prefix}_o + {prefix}_w < 1 for row-stochastic W."
    )


def _spsolve_flow(
    A: _FlowFilter, rhs, letter: str = "A", prefix: str = "rho"
) -> np.ndarray:
    """``A^{-1} rhs`` for a flow filter."""
    return A.solve(rhs)


def _factorized_flow(A: _FlowFilter, letter: str = "A", prefix: str = "rho"):
    """A reusable ``solve(rhs)`` for a flow filter (panel generators)."""
    return A.solve


def _warn_flow_stability(
    r_d: float, r_o: float, r_w: float, prefix: str, filter_desc: str
) -> None:
    """Warn when the sufficient stability bound ``|r_d|+|r_o|+|r_w| < 1`` fails.

    For row-standardized W_d/W_o/W_w each has spectral radius 1, so the bound
    is sufficient (and necessary for guaranteed invertibility across all valid
    weight inputs); a tighter eigenvalue check would require an O(N^3) solve
    on N = n^2.
    """
    r_sum = abs(r_d) + abs(r_o) + abs(r_w)
    if r_sum >= 1.0:
        warnings.warn(
            f"|{prefix}_d|+|{prefix}_o|+|{prefix}_w| = {r_sum:g} >= 1; "
            f"{filter_desc} may be singular or "
            "numerically unstable. Reduce parameters to satisfy the "
            f"sufficient stability bound |{prefix}_d|+|{prefix}_o|+|{prefix}_w| < 1.",
            stacklevel=3,
        )


def _flow_beta_full(
    p: int,
    beta_d_arr: np.ndarray,
    beta_o_arr: np.ndarray,
    gamma_dist: float,
    intercept: float = 0.0,
) -> np.ndarray:
    """Full coefficient vector on the O-D design layout.

    Layout: ``[intercept, intra_indicator=0, beta_d..., beta_o...,
    intra_x=0..., gamma_dist]`` — the intra coefficients stay 0 in every DGP
    and the trailing log-distance coefficient carries the distance decay.
    """
    k_d, k_o = len(beta_d_arr), len(beta_o_arr)
    beta_full = np.zeros(p, dtype=np.float64)
    beta_full[0] = intercept
    beta_full[2 : 2 + k_d] = beta_d_arr
    beta_full[2 + k_d : 2 + k_d + k_o] = beta_o_arr
    beta_full[-1] = gamma_dist
    return beta_full


def _resolve_nb_flow_betas(
    beta_d, beta_o, k: int, k_d: int | None, k_o: int | None
) -> tuple[np.ndarray, np.ndarray, int, int]:
    """NB-generator coefficient/column-count inference (scalar broadcast)."""
    if beta_d is not None:
        beta_d_arr = np.asarray(beta_d, dtype=float).ravel()
        k_d_val = k_d if k_d is not None else len(beta_d_arr)
    else:
        k_d_val = k_d if k_d is not None else k
        beta_d_arr = np.ones(k_d_val, dtype=float)

    if beta_o is not None:
        beta_o_arr = np.asarray(beta_o, dtype=float).ravel()
        k_o_val = k_o if k_o is not None else len(beta_o_arr)
    else:
        k_o_val = k_o if k_o is not None else k
        beta_o_arr = np.ones(k_o_val, dtype=float)

    return beta_d_arr, beta_o_arr, k_d_val, k_o_val


def _hetero_flow_eps(
    rng: np.random.Generator,
    sigma: float,
    X_d_arr: np.ndarray,
    X_o_arr: np.ndarray,
    n: int,
) -> np.ndarray:
    """Heteroskedastic innovations: sd ``sigma*sqrt(1+||x_dest||²+||x_orig||²)``."""
    N = n * n
    dest_idx = np.repeat(np.arange(n), n)  # row = destination unit
    orig_idx = np.tile(np.arange(n), n)  # col = origin unit
    scale_vec = sigma * np.sqrt(
        1.0
        + np.sum(X_d_arr[dest_idx] ** 2, axis=1)
        + np.sum(X_o_arr[orig_idx] ** 2, axis=1)
    )
    return rng.standard_normal(N) * scale_vec


def generate_flow_data(
    n: Optional[int] = None,
    G: Optional[Graph] = None,
    rho_d: float = 0.3,
    rho_o: float = 0.2,
    rho_w: float = 0.1,
    beta_d: Union[np.ndarray, list, None] = None,
    beta_o: Union[np.ndarray, list, None] = None,
    sigma: float = 1.0,
    X: Optional[np.ndarray] = None,
    col_names: Optional[list] = None,
    dist: Optional[np.ndarray] = None,
    gamma_dist: float = -0.5,
    alpha: float = 0.0,
    seed: Optional[int] = None,
    gdf=None,
    err_hetero: bool = False,
    knn_k: int = 4,
    distribution: str = "lognormal",
) -> dict:
    """Simulate flow data from a SAR flow model.

    Generates :math:`N = n^2` flow observations.  The latent
    SAR-filtered process is

    .. math::

        \\eta = A^{-1}(X\\beta + \\varepsilon), \\quad
        A = I_N - \\rho_d W_d - \\rho_o W_o - \\rho_w W_w, \\quad
        \\varepsilon \\sim \\mathcal{N}(0, \\sigma^2 I_N)

    and the observed flows are either

    .. math::

        y = \\exp(\\eta) \\quad \\text{(default, } \\texttt{distribution=\"lognormal\"})

    so that :math:`y > 0` and :math:`\\mathbb{E}[y] = \\exp(\\eta + \\sigma^2/2)`,
    or :math:`y = \\eta` when ``distribution=\"normal\"`` (legacy
    Gaussian-on-y behavior).

    To recover the SAR parameters with the existing
    :class:`~neighbayes.models.flow.SARFlow` /
    :class:`~neighbayes.models.flow.SARFlowSeparable`, fit on
    ``np.log(y_vec)`` (which by construction equals ``eta_vec``).

    Parameters
    ----------
    n : int
        Number of spatial units.  Must match the size of *G*.
    G : libpysal.graph.Graph
        Row-standardized spatial graph on *n* units.
    rho_d : float
        Destination spatial autoregressive parameter.
    rho_o : float
        Origin spatial autoregressive parameter.
    rho_w : float
        Network (origin-destination) spatial autoregressive parameter.
    beta_d : array-like, shape (k_d,)
        Destination-side regression coefficients.
    beta_o : array-like, shape (k_o,)
        Origin-side regression coefficients.  When ``k_o != k_d``,
        separate destination and origin attribute matrices are generated
        or required.
    sigma : float, default 1.0
        Standard deviation of the error term.
    X : np.ndarray, shape (n, k) or (n, k_d + k_o), optional
        Regional attribute matrix.  If None, draws X_d and X_o separately
        from N(0, 1).  If a single matrix is provided with ``k_d == k_o``,
        it is used for both destination and origin blocks.  If it has
        ``k_d + k_o`` columns, the first ``k_d`` are used as destination
        attributes and the remaining ``k_o`` as origin attributes.
    col_names : list[str], optional
        Names for the *k* columns of *X*.
    dist : np.ndarray, shape (n, n), optional
        Distance / cost matrix.  If ``None`` (default), one is computed
        automatically from *gdf* (or from a synthetic point grid when
        *gdf* is also ``None``) and entered as ``log(1 + d)`` in the
        design matrix.  Pass an array explicitly to override.
    gamma_dist : float, default -0.5
        True coefficient on the (log-) distance column in the DGP.
        Defaults to ``-0.5`` to mimic gravity-model distance decay; set
        to ``0.0`` to neutralize the effect.
    alpha : float, default 0.0
        Intercept term added uniformly to all latent flow cells.  Under
        ``distribution=\"lognormal\"`` (default) this multiplies the
        observed flows by ``exp(alpha)``; under
        ``distribution=\"normal\"`` it is an additive shift on ``y``.
    seed : int, optional
        Random seed for reproducibility.
    gdf : geopandas.GeoDataFrame, optional
        Geometry source used to derive distance.  If ``None`` and *dist*
        is also ``None``, a synthetic point grid is built via
        :func:`~neighbayes.dgp.utils.synth_point_geodataframe`.
    err_hetero : bool, default False
        If True, generate heteroskedastic innovations: each flow cell
        :math:`(i,j)` has standard deviation
        :math:`\\sigma \\sqrt{1 + \\|x_i\\|^2 + \\|x_j\\|^2}` where
        :math:`x_i`, :math:`x_j` are the destination and origin attribute
        vectors for that cell.
    knn_k : int, default 4
        Number of nearest neighbors used when synthesising a default
        graph from a synthetic point grid (see
        :func:`~neighbayes.dgp.utils._resolve_flow_geometry`).
    distribution : {\"lognormal\", \"normal\"}, default \"lognormal\"
        Observation-scale family.  ``\"lognormal\"`` returns
        ``y = exp(eta)`` (strictly positive flows, the default).
        ``\"normal\"`` returns ``y = eta`` (legacy Gaussian-on-y
        behavior).  In both cases ``\"eta_vec\"``/``\"eta_mat\"`` is
        also exposed in the return dict.

    Returns
    -------
    dict
        Dictionary with keys:

        - ``\"y_vec\"`` (N,): vectorized flows on the observation scale.
        - ``\"y_mat\"`` (n, n): flow matrix form.
        - ``\"eta_vec\"`` (N,): latent SAR-filtered linear predictor
          (equals ``log(y_vec)`` when ``distribution=\"lognormal\"``).
        - ``\"eta_mat\"`` (n, n): ``eta_vec`` reshaped.
        - ``\"distribution\"`` str: the value of the *distribution* arg.
        - ``\"X\"`` (N, p): full O-D design matrix (for model fitting).
        - ``\"X_regional\"`` (n, k_d): destination-side regional attribute matrix.
        - ``\"X_regional_d\"`` (n, k_d): destination-side regional attribute matrix.
        - ``\"X_regional_o\"`` (n, k_o): origin-side regional attribute matrix.
        - ``\"design\"`` :class:`~neighbayes.graph.FlowDesignMatrix`: full design.
        - ``\"W\"`` scipy.sparse.csr_matrix: n×n weight matrix.
        - ``\"G\"`` libpysal.graph.Graph: spatial graph.
        - ``\"rho_d\"``, ``\"rho_o\"``, ``\"rho_w\"``, ``\"sigma\"``: true parameters.
        - ``\"beta_d\"``, ``\"beta_o\"``: true coefficient vectors.

    Raises
    ------
    ValueError
        If the A matrix is singular (invalid parameter combination).
    """
    rng = np.random.default_rng(seed)

    if distribution not in ("lognormal", "normal"):
        raise ValueError(
            f"distribution must be 'lognormal' or 'normal', got {distribution!r}."
        )

    n, G, gdf = _resolve_flow_geometry(n=n, G=G, gdf=gdf, knn_k=knn_k)
    W = _graph_to_csr(G)
    N = n * n

    if beta_d is None:
        beta_d = [1.0]
    if beta_o is None:
        beta_o = [1.0]
    beta_d_arr = np.asarray(beta_d, dtype=np.float64).ravel()
    beta_o_arr = np.asarray(beta_o, dtype=np.float64).ravel()
    k_d = len(beta_d_arr)
    k_o = len(beta_o_arr)

    # Generate or validate regional attributes
    if X is None:
        X_d_arr = rng.standard_normal((n, k_d))
        X_o_arr = rng.standard_normal((n, k_o))
    else:
        X_arr = np.asarray(X, dtype=np.float64)
        if X_arr.ndim == 1:
            X_arr = X_arr[:, None]
        if X_arr.shape[1] == k_d + k_o:
            # Split into destination and origin columns
            X_d_arr = X_arr[:, :k_d]
            X_o_arr = X_arr[:, k_d:]
        elif X_arr.shape[1] == k_d and k_d == k_o:
            # Same matrix for both (legacy symmetric case)
            X_d_arr = X_arr
            X_o_arr = X_arr
        else:
            raise ValueError(
                f"X must have {k_d + k_o} columns (split) or "
                f"{k_d} columns (symmetric), got {X_arr.shape[1]}."
            )

    # Resolve distance
    if dist is None:
        dist = pairwise_distance_matrix(gdf)
    dist = np.asarray(dist, dtype=np.float64)
    if dist.shape != (n, n):
        raise ValueError(f"dist must have shape ({n}, {n}), got {dist.shape}.")

    # Build design matrix
    if k_d == k_o:
        design = flow_design_matrix(
            X_d_arr, col_names=col_names, dist=dist, log_distance=True
        )
    else:
        design = flow_design_matrix_asymmetric(
            X_d_arr,
            X_o_arr,
            col_names_d=col_names,
            col_names_o=None,
            dist=dist,
            log_distance=True,
        )

    A = _flow_system(G, rho_d, rho_o, rho_w)
    _warn_flow_stability(
        rho_d,
        rho_o,
        rho_w,
        prefix="rho",
        filter_desc="the flow filter (I_N - rho_d W_d - rho_o W_o - rho_w W_w)",
    )

    beta_full = _flow_beta_full(
        design.combined.shape[1], beta_d_arr, beta_o_arr, gamma_dist, intercept=alpha
    )

    Xbeta = design.combined @ beta_full  # (N,)

    # Solve y = A^{-1}(Xbeta + epsilon)
    if err_hetero:
        eps = _hetero_flow_eps(rng, sigma, X_d_arr, X_o_arr, n)
    else:
        eps = rng.normal(scale=sigma, size=N)
    eta_vec = _spsolve_flow(A, Xbeta + eps)

    if distribution == "lognormal":
        y_vec = np.exp(eta_vec)
    else:
        y_vec = eta_vec

    y_mat = y_vec.reshape(n, n)
    eta_mat = eta_vec.reshape(n, n)

    return {
        "y_vec": y_vec,
        "y_mat": y_mat,
        "eta_vec": eta_vec,
        "eta_mat": eta_mat,
        "distribution": distribution,
        "X": design.combined,
        "X_regional": X_d_arr,
        "X_regional_d": X_d_arr,
        "X_regional_o": X_o_arr,
        "col_names": design.feature_names,
        "design": design,
        "W": W,
        "G": G,
        "gdf": gdf,
        "dist": dist,
        "rho_d": rho_d,
        "rho_o": rho_o,
        "rho_w": rho_w,
        "sigma": sigma,
        "beta_d": beta_d_arr,
        "beta_o": beta_o_arr,
        "gamma_dist": gamma_dist,
    }


def generate_negbin_flow_data(
    n: int | None = None,
    k: int = 2,
    k_d: int | None = None,
    k_o: int | None = None,
    rho_d: float = 0.3,
    rho_o: float = 0.2,
    rho_w: float = 0.1,
    beta_d: float | list[float] | None = None,
    beta_o: float | list[float] | None = None,
    alpha: float = 2.0,
    gamma_dist: float = -0.5,
    seed: int = 42,
    G: Graph | None = None,
    err_hetero: bool = False,
    gdf: object = None,
    knn_k: int = 4,
) -> dict:
    r"""Generate synthetic O-D flow counts from an NB2 SAR flow DGP.

    The data-generating process follows:

    .. math::

        \eta = A(\rho_d, \rho_o, \rho_w)^{-1} X\beta, \qquad
        y_{ij} \sim \operatorname{NegBin}(\mu=\exp(\eta_{ij}),\, \alpha)

    where :math:`\operatorname{Var}(y)=\mu+\mu^2/\alpha` and the system
    matrix is

    .. math::

        A = I_N - \rho_d (I_n \otimes W) - \rho_o (W \otimes I_n)
              - \rho_w (W \otimes W), \quad N = n^2

    and :math:`W` is the row-standardized spatial weight matrix.

    Parameters
    ----------
    n : int, default 10
        Approximate number of spatial units.  When neither *G* nor *gdf*
        is provided, a rook-contiguity grid with
        round(sqrt(n)) units per side is created, yielding
        approximately *n* units.  Total number of flows is
        N = n_actual^2.  When *G* is provided, *n* must
        match the number of units in *G*.
    k : int, default 2
        Number of destination/origin attribute columns when ``k_d`` and
        ``k_o`` are not specified (excluding intercepts added internally).
        Ignored when ``k_d`` and/or ``k_o`` are provided, or when
        ``beta_d``/``beta_o`` are lists whose length determines ``k_d``/``k_o``.
    k_d : int or None, default None
        Number of destination-side attribute columns.  Overrides ``k`` for
        the destination side when provided.
    k_o : int or None, default None
        Number of origin-side attribute columns.  Overrides ``k`` for
        the origin side when provided.
    rho_d : float, default 0.3
        Destination autocorrelation parameter.
    rho_o : float, default 0.2
        Origin autocorrelation parameter.
    rho_w : float, default 0.1
        Network autocorrelation parameter.
    beta_d : float or list of float or None, default None
        Destination-side coefficients for the *k* attributes.  A scalar
        broadcasts to all columns.  Defaults to ``1.0`` for all columns.
    beta_o : float or list of float or None, default None
        Origin-side coefficients.  Defaults to ``1.0`` for all columns.
    alpha : float, default 2.0
        NB2 dispersion parameter.  Must be strictly positive.
        Larger values approach Poisson; smaller values increase overdispersion.
    gamma_dist : float, default -0.5
        Coefficient on the log-distance regressor.
    seed : int, default 42
        Seed for :class:`numpy.random.default_rng`.
    G : libpysal.graph.Graph or None, default None
        Row-standardized spatial graph on *n* units.  If ``None``, a
        rook-contiguity graph on a regular grid is constructed automatically
        via :func:`~neighbayes.dgp.utils.resolve_weights`.
    err_hetero : bool, default False
        Accepted for API parity with other DGP functions; ignored for the
        NB model (the variance is determined by the mean and alpha).
    gdf : GeoDataFrame or None, default None
        Accepted for API parity; ignored (use *G* instead).

    Returns
    -------
    dict with keys:
        ``y_vec`` : np.ndarray, shape (N,), dtype int64
            Flattened count observations.
        ``y_mat`` : np.ndarray, shape (n, n), dtype int64
            Count observations reshaped as an O×D matrix.
        ``eta_vec`` : np.ndarray, shape (N,)
            Log-mean (spatially filtered linear predictor).
        ``lambda_vec`` : np.ndarray, shape (N,)
            NB means (:math:`\exp(\eta_{ij})`).
        ``Xd`` : np.ndarray, shape (n, k_d)
            Destination-side regional attribute matrix.
        ``Xo`` : np.ndarray, shape (n, k_o)
            Origin-side regional attribute matrix.
        ``X`` : np.ndarray, shape (N, p)
            Full O-D design matrix (for model fitting).
        ``design`` : FlowDesignMatrix
            Full O-D design matrix (for downstream inspection).
        ``W`` : np.ndarray, shape (n, n)
            Dense row-standardized weight matrix.
        ``G`` : libpysal.graph.Graph
            Spatial graph.
        ``rho_d``, ``rho_o``, ``rho_w``
            True autocorrelation parameters.
        ``beta_d``, ``beta_o``
            True coefficient vectors.
        ``alpha`` : float
            True NB2 dispersion parameter.

    Raises
    ------
    ValueError
        If ``alpha <= 0`` or the system matrix :math:`A` is singular.
    """
    if alpha <= 0:
        raise ValueError("alpha must be strictly positive.")

    from .utils import _resolve_flow_geometry

    rng = np.random.default_rng(seed)

    # --- Resolve spatial weights & geometry ---
    n, G, gdf = _resolve_flow_geometry(n=n, G=G, gdf=gdf, knn_k=knn_k)
    W = _graph_to_csr(G)

    # --- Coefficient vectors ---
    beta_d_arr, beta_o_arr, k_d_val, k_o_val = _resolve_nb_flow_betas(
        beta_d, beta_o, k, k_d, k_o
    )

    # --- Regional attributes ---
    Xd_raw = rng.standard_normal((n, k_d_val))
    Xo_raw = rng.standard_normal((n, k_o_val))

    # --- Distance matrix ---
    dist = pairwise_distance_matrix(gdf)

    # --- Build design matrix with separate Xd and Xo blocks ---
    design = flow_design_matrix_with_orig(
        Xd_raw,
        Xo_raw,
        col_names=[f"x{i}" for i in range(k_d_val)],
        dist=dist,
        log_distance=True,
    )

    beta_full = _flow_beta_full(
        design.combined.shape[1], beta_d_arr, beta_o_arr, gamma_dist
    )

    Xbeta = design.combined @ beta_full  # (N,)

    A = _flow_system(G, rho_d, rho_o, rho_w)

    # --- Solve A eta = Xbeta  (reduced form: no sigma, no noise) ---
    eta_vec = _spsolve_flow(A, Xbeta)

    lambda_vec = np.exp(eta_vec)
    p_nb = alpha / (alpha + lambda_vec)
    y_vec = rng.negative_binomial(alpha, p_nb).astype(np.int64)
    y_mat = y_vec.reshape(n, n)

    return {
        "y_vec": y_vec,
        "y_mat": y_mat,
        "eta_vec": eta_vec,
        "lambda_vec": lambda_vec,
        "Xd": Xd_raw,
        "Xo": Xo_raw,
        "X": design.combined,
        "col_names": design.feature_names,
        "design": design,
        "W": W,
        "G": G,
        "gdf": gdf,
        "dist": dist,
        "rho_d": rho_d,
        "rho_o": rho_o,
        "rho_w": rho_w,
        "beta_d": beta_d_arr,
        "beta_o": beta_o_arr,
        "gamma_dist": gamma_dist,
        "alpha": float(alpha),
    }


def generate_negbin_flow_data_separable(
    n: int | None = None,
    rho_d: float = 0.3,
    rho_o: float = 0.2,
    **kwargs,
) -> dict:
    """NB2 flow DGP with separability constraint ``rho_w = -rho_d * rho_o``."""
    rho_w = -rho_d * rho_o
    return generate_negbin_flow_data(
        n=n,
        rho_d=rho_d,
        rho_o=rho_o,
        rho_w=rho_w,
        **kwargs,
    )


def generate_panel_flow_data(
    n: Optional[int] = None,
    T: int = 5,
    G: Optional[Graph] = None,
    rho_d: float = 0.3,
    rho_o: float = 0.2,
    rho_w: float = 0.1,
    beta_d: Union[np.ndarray, list, None] = None,
    beta_o: Union[np.ndarray, list, None] = None,
    sigma: float = 1.0,
    sigma_alpha: float = 0.5,
    gamma_dist: float = -0.5,
    seed: Optional[int] = None,
    k: Optional[int] = None,
    err_hetero: bool = False,
    gdf: object = None,
    knn_k: int = 4,
    distribution: str = "lognormal",
) -> dict:
    r"""Simulate panel flow data from a SAR flow model with unit effects.

    For each period :math:`t = 1, \dots, T`, generates :math:`N = n^2`
    latent flow observations from:

    .. math::

        \eta_t = A^{-1}(X_t \beta + \alpha + \varepsilon_t), \quad
        A = I_N - \rho_d W_d - \rho_o W_o - \rho_w W_w, \quad
        \varepsilon_t \sim \mathcal{N}(0, \sigma^2 I_N)

    where :math:`\alpha \sim \mathcal{N}(0, \sigma_\alpha^2 I_N)` are
    O-D-pair random effects drawn once and held fixed across periods.
    The observed flows are :math:`y_t = \exp(\eta_t)` under the default
    ``distribution="lognormal"`` (strictly positive flows), or
    :math:`y_t = \eta_t` under ``distribution="normal"`` (legacy
    Gaussian-on-y behavior).

    Observations are stacked in **time-first** order so that the
    observation at index :math:`t \cdot n^2 + k` is O-D pair :math:`k`
    at time :math:`t`.

    To recover the SAR parameters with the existing
    :class:`~neighbayes.models.flow_panel.SARFlowPanel` /
    :class:`~neighbayes.models.flow_panel.SARFlowSeparablePanel`,
    fit on ``np.log(y)``.

    Parameters
    ----------
    n : int
        Number of spatial units.  Must match the size of *G*.
    T : int
        Number of time periods.
    G : libpysal.graph.Graph
        Row-standardized spatial graph on *n* units.
    rho_d : float
        Destination spatial autoregressive parameter.
    rho_o : float
        Origin spatial autoregressive parameter.
    rho_w : float
        Network (origin-destination) spatial autoregressive parameter.
    beta_d : array-like, shape (k_d,)
        Destination-side regression coefficients.
    beta_o : array-like, shape (k_o,)
        Origin-side regression coefficients.  When ``k_o != k_d``,
        separate destination and origin attribute matrices are generated.
    sigma : float, default 1.0
        Standard deviation of the idiosyncratic error term.
    sigma_alpha : float, default 0.5
        Standard deviation of the O-D-pair random effect.
        Set to 0 for pooled (no unit effects).
    seed : int, optional
        Random seed for reproducibility.
    k : int, optional
        Number of regional attribute columns.  If ``None``, inferred
        from the length of *beta_d*.
    err_hetero : bool, default False
        Accepted for API parity with other DGP functions; currently
        ignored (homoskedastic errors are always generated).
    gdf : object, optional
        Accepted for API parity with other DGP functions; not used
        (pass *G* directly instead).
    knn_k : int, default 4
        Number of nearest neighbors used when synthesising a default
        graph from a synthetic point grid.
    distribution : {"lognormal", "normal"}, default "lognormal"
        Observation-scale family.  ``"lognormal"`` returns
        ``y = exp(eta)`` (strictly positive flows, the default);
        ``"normal"`` returns ``y = eta`` (legacy Gaussian-on-y).

    Returns
    -------
    dict
        Dictionary with keys:

        - ``"y"`` (n²T,): time-first stacked flow vector on the
          observation scale.
        - ``"eta"`` (n²T,): latent SAR-filtered linear predictor
          (equals ``log(y)`` under ``distribution="lognormal"``).
        - ``"distribution"`` str: the value of the *distribution* arg.
        - ``"X"`` (n²T, p): time-first stacked O-D design matrix.
        - ``"col_names"`` list[str]: feature names.
        - ``"G"`` libpysal.graph.Graph: spatial graph.
        - ``"rho_d"``, ``"rho_o"``, ``"rho_w"``, ``"sigma"``,
          ``"sigma_alpha"``: true parameters.
        - ``"beta_d"``, ``"beta_o"``: true coefficient vectors.
        - ``"params_true"`` dict: nested dict of all true parameters
          (including ``"distribution"``).

    Raises
    ------
    ValueError
        If the A matrix is singular (invalid parameter combination).
    """
    rng = np.random.default_rng(seed)

    if distribution not in ("lognormal", "normal"):
        raise ValueError(
            f"distribution must be 'lognormal' or 'normal', got {distribution!r}."
        )

    n, G, gdf = _resolve_flow_geometry(n=n, G=G, gdf=gdf, knn_k=knn_k)
    _graph_to_csr(G)
    N = n * n  # O-D pairs per period

    if beta_d is None:
        beta_d = [1.0]
    if beta_o is None:
        beta_o = [1.0]
    beta_d_arr = np.asarray(beta_d, dtype=np.float64).ravel()
    beta_o_arr = np.asarray(beta_o, dtype=np.float64).ravel()
    k_d_val = len(beta_d_arr)
    k_o_val = len(beta_o_arr)

    # Assemble and factorize A once (same A for every period)
    A = _flow_system(G, rho_d, rho_o, rho_w)
    solve_A = _factorized_flow(A)

    # Draw O-D-pair random effects once
    alpha = rng.normal(0.0, sigma_alpha, N) if sigma_alpha > 0 else np.zeros(N)

    # Distance matrix (time-invariant)
    dist = pairwise_distance_matrix(gdf)

    y_list, X_list, eta_list = [], [], []
    col_names = None

    for _ in range(T):
        # Generate fresh regional attributes each period
        X_d_reg = rng.standard_normal((n, k_d_val))
        X_o_reg = rng.standard_normal((n, k_o_val))
        if k_d_val == k_o_val:
            design = flow_design_matrix(X_d_reg, dist=dist, log_distance=True)
        else:
            design = flow_design_matrix_asymmetric(
                X_d_reg, X_o_reg, dist=dist, log_distance=True
            )
        if col_names is None:
            col_names = design.feature_names

        beta_full = _flow_beta_full(
            design.combined.shape[1], beta_d_arr, beta_o_arr, gamma_dist
        )

        Xbeta = design.combined @ beta_full  # (N,)
        eps = rng.normal(scale=sigma, size=N)
        rhs = Xbeta + alpha + eps

        eta_t = solve_A(rhs)
        if distribution == "lognormal":
            y_t = np.exp(eta_t)
        else:
            y_t = eta_t
        eta_list.append(eta_t)
        y_list.append(y_t)
        X_list.append(design.combined)

    y = np.concatenate(y_list)  # (N*T,)
    eta = np.concatenate(eta_list)  # (N*T,)
    X = np.vstack(X_list)  # (N*T, p)

    return {
        "y": y,
        "eta": eta,
        "distribution": distribution,
        "X": X,
        "col_names": col_names,
        "G": G,
        "gdf": gdf,
        "dist": dist,
        "rho_d": rho_d,
        "rho_o": rho_o,
        "rho_w": rho_w,
        "sigma": sigma,
        "sigma_alpha": sigma_alpha,
        "beta_d": beta_d_arr,
        "beta_o": beta_o_arr,
        "gamma_dist": gamma_dist,
        "params_true": {
            "rho_d": rho_d,
            "rho_o": rho_o,
            "rho_w": rho_w,
            "sigma": sigma,
            "sigma_alpha": sigma_alpha,
            "beta_d": beta_d_arr,
            "beta_o": beta_o_arr,
            "gamma_dist": gamma_dist,
            "distribution": distribution,
        },
    }


def generate_panel_negbin_flow_data(
    n: int | None = None,
    T: int = 5,
    G: Graph | None = None,
    rho_d: float = 0.3,
    rho_o: float = 0.2,
    rho_w: float = 0.1,
    beta_d: float | list[float] | None = None,
    beta_o: float | list[float] | None = None,
    alpha: float = 2.0,
    gamma_dist: float = -0.5,
    seed: int = 42,
    k: int = 2,
    k_d: int | None = None,
    k_o: int | None = None,
    err_hetero: bool = False,
    gdf: object = None,
    knn_k: int = 4,
) -> dict:
    r"""Generate panel NB2 flow counts from a spatial autoregressive DGP.

    For each period :math:`t = 1, \dots, T`, generates :math:`N = n^2`
    flow counts from:

    .. math::

        \eta_t = A^{-1} X_t \beta, \quad
        y_{ij,t} \sim \operatorname{NegBin}(\mu=\exp(\eta_{ij,t}),\, \alpha)

    where the system matrix is

    .. math::

        A = I_N - \rho_d (I_n \otimes W) - \rho_o (W \otimes I_n)
              - \rho_w (W \otimes W), \quad N = n^2

    Observations are stacked in **time-first** order.  There is no
    :math:`\sigma` parameter (the NB variance is determined by the mean
    and alpha).

    Parameters
    ----------
    n : int
        Number of spatial units.  Must match the size of *G*.
    T : int, default 5
        Number of time periods.
    G : libpysal.graph.Graph
        Row-standardized spatial graph on *n* units.
    rho_d : float, default 0.3
        Destination autocorrelation parameter.
    rho_o : float, default 0.2
        Origin autocorrelation parameter.
    rho_w : float, default 0.1
        Network autocorrelation parameter.
    beta_d : float or list of float or None, default None
        Destination-side coefficients for the *k* attributes.  A scalar
        broadcasts to all columns.  Defaults to ``1.0`` for all columns.
    beta_o : float or list of float or None, default None
        Origin-side coefficients.  Defaults to ``1.0`` for all columns.
    alpha : float, default 2.0
        NB2 dispersion parameter.  Must be strictly positive.
    gamma_dist : float, default -0.5
        Coefficient on the log-distance regressor.
    seed : int, default 42
        Seed for :class:`numpy.random.default_rng`.
    k : int, default 2
        Number of destination/origin attribute columns.
    err_hetero : bool, default False
        Accepted for API parity with other DGP functions; ignored for
        the NB model (the variance is determined by the mean and alpha).
    gdf : object, optional
        Accepted for API parity with other DGP functions; not used
        (pass *G* directly instead).

    Returns
    -------
    dict
        Dictionary with keys:

        - ``"y"`` (n²T,): time-first stacked count vector (int64).
        - ``"lambda"`` (n²T,): time-first stacked NB means.
        - ``"X"`` (n²T, p): time-first stacked O-D design matrix.
        - ``"col_names"`` list[str]: feature names.
        - ``"G"`` libpysal.graph.Graph: spatial graph.
        - ``"rho_d"``, ``"rho_o"``, ``"rho_w"``: true parameters.
        - ``"beta_d"``, ``"beta_o"``: true coefficient vectors.
        - ``"alpha"``: true NB2 dispersion parameter.
        - ``"params_true"`` dict: nested dict of all true parameters.

    Raises
    ------
    ValueError
        If ``alpha <= 0`` or the A matrix is singular.
    """
    if alpha <= 0:
        raise ValueError("alpha must be strictly positive.")

    from .utils import _resolve_flow_geometry

    rng = np.random.default_rng(seed)

    n, G, gdf = _resolve_flow_geometry(n=n, G=G, gdf=gdf, knn_k=knn_k)
    _graph_to_csr(G)

    beta_d_arr, beta_o_arr, k_d_val, k_o_val = _resolve_nb_flow_betas(
        beta_d, beta_o, k, k_d, k_o
    )

    # Assemble and factorize A once (same A for every period)
    A = _flow_system(G, rho_d, rho_o, rho_w)
    solve_A = _factorized_flow(A)

    y_list, X_list, lambda_list = [], [], []
    col_names = None

    # Distance matrix (time-invariant)
    dist = pairwise_distance_matrix(gdf)

    for _ in range(T):
        # Generate fresh regional attributes each period
        Xd_raw = rng.standard_normal((n, k_d_val))
        Xo_raw = rng.standard_normal((n, k_o_val))

        design = flow_design_matrix_with_orig(
            Xd_raw,
            Xo_raw,
            col_names=[f"x{i}" for i in range(k_d_val)],
            dist=dist,
            log_distance=True,
        )
        if col_names is None:
            col_names = design.feature_names

        beta_full = _flow_beta_full(
            design.combined.shape[1], beta_d_arr, beta_o_arr, gamma_dist
        )

        Xbeta = design.combined @ beta_full  # (N,)

        # Solve A eta = Xbeta  (reduced form: no sigma, no noise)
        eta_t = solve_A(Xbeta)
        lambda_t = np.exp(eta_t)
        p_nb = alpha / (alpha + lambda_t)
        y_t = rng.negative_binomial(alpha, p_nb).astype(np.int64)

        y_list.append(y_t)
        lambda_list.append(lambda_t)
        X_list.append(design.combined)

    y = np.concatenate(y_list)  # (N*T,)
    lambda_vec = np.concatenate(lambda_list)  # (N*T,)
    X = np.vstack(X_list)  # (N*T, p)

    return {
        "y": y,
        "lambda": lambda_vec,
        "X": X,
        "col_names": col_names,
        "G": G,
        "gdf": gdf,
        "dist": dist,
        "rho_d": rho_d,
        "rho_o": rho_o,
        "rho_w": rho_w,
        "beta_d": beta_d_arr,
        "beta_o": beta_o_arr,
        "gamma_dist": gamma_dist,
        "alpha": float(alpha),
        "params_true": {
            "rho_d": rho_d,
            "rho_o": rho_o,
            "rho_w": rho_w,
            "beta_d": beta_d_arr,
            "beta_o": beta_o_arr,
            "gamma_dist": gamma_dist,
            "alpha": float(alpha),
        },
    }


def generate_flow_data_separable(
    n: Optional[int] = None,
    G: Optional[Graph] = None,
    rho_d: float = 0.3,
    rho_o: float = 0.2,
    beta_d: Union[np.ndarray, list, None] = None,
    beta_o: Union[np.ndarray, list, None] = None,
    **kwargs,
) -> dict:
    """Simulate flow data from a *separable* SAR flow model.

    Identical to :func:`generate_flow_data` except the network parameter is
    derived from the separability constraint
    :math:`\\rho_w = -\\rho_d \\rho_o`, so it is **not** a free argument.
    Use this function to generate training data for
    :class:`~neighbayes.models.flow.SARFlowSeparable`.

    Parameters
    ----------
    n : int
        Number of spatial units.
    G : libpysal.graph.Graph
        Row-standardized spatial graph on *n* units.
    rho_d : float
        Destination spatial autoregressive parameter.
    rho_o : float
        Origin spatial autoregressive parameter.
    beta_d : array-like, shape (k,)
        Destination-side regression coefficients.
    beta_o : array-like, shape (k,)
        Origin-side regression coefficients.
    **kwargs
        Forwarded to :func:`generate_flow_data` (e.g. ``sigma``, ``seed``,
        ``X``, ``col_names``).

    Returns
    -------
    dict
        Same as :func:`generate_flow_data`.  The ``"rho_w"`` entry will equal
        ``-rho_d * rho_o``.
    """
    rho_w = -rho_d * rho_o
    return generate_flow_data(
        n=n,
        G=G,
        rho_d=rho_d,
        rho_o=rho_o,
        rho_w=rho_w,
        beta_d=beta_d,
        beta_o=beta_o,
        **kwargs,
    )


def generate_panel_flow_data_separable(
    n: Optional[int] = None,
    T: int = 5,
    G: Optional[Graph] = None,
    rho_d: float = 0.3,
    rho_o: float = 0.2,
    beta_d: Union[np.ndarray, list, None] = None,
    beta_o: Union[np.ndarray, list, None] = None,
    **kwargs,
) -> dict:
    """Simulate panel flow data from a *separable* SAR flow model.

    Identical to :func:`generate_panel_flow_data` except the network parameter
    is derived as :math:`\\rho_w = -\\rho_d \\rho_o`.  Use this function to
    generate training data for
    :class:`~neighbayes.models.flow_panel.SARFlowSeparablePanel`.

    Parameters
    ----------
    n : int
        Number of spatial units.
    T : int
        Number of time periods.
    G : libpysal.graph.Graph
        Row-standardized spatial graph on *n* units.
    rho_d : float
        Destination spatial autoregressive parameter.
    rho_o : float
        Origin spatial autoregressive parameter.
    beta_d : array-like, shape (k,)
        Destination-side regression coefficients.
    beta_o : array-like, shape (k,)
        Origin-side regression coefficients.
    **kwargs
        Forwarded to :func:`generate_panel_flow_data` (e.g. ``sigma``,
        ``sigma_alpha``, ``seed``).

    Returns
    -------
    dict
        Same as :func:`generate_panel_flow_data`.  The ``"rho_w"`` entry will
        equal ``-rho_d * rho_o``.
    """
    rho_w = -rho_d * rho_o
    return generate_panel_flow_data(
        n=n,
        T=T,
        G=G,
        rho_d=rho_d,
        rho_o=rho_o,
        rho_w=rho_w,
        beta_d=beta_d,
        beta_o=beta_o,
        **kwargs,
    )


def generate_panel_negbin_flow_data_separable(
    n: int | None = None,
    T: int = 5,
    G: Graph | None = None,
    rho_d: float = 0.3,
    rho_o: float = 0.2,
    **kwargs,
) -> dict:
    """Panel NB2 flow DGP with separability constraint ``rho_w=-rho_d*rho_o``."""
    rho_w = -rho_d * rho_o
    return generate_panel_negbin_flow_data(
        n=n,
        T=T,
        G=G,
        rho_d=rho_d,
        rho_o=rho_o,
        rho_w=rho_w,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Spatial-error (SEM) flow DGPs
# ---------------------------------------------------------------------------


def generate_sem_flow_data(
    n: Optional[int] = None,
    G: Optional[Graph] = None,
    lam_d: float = 0.3,
    lam_o: float = 0.2,
    lam_w: float = 0.1,
    beta_d: Union[np.ndarray, list, None] = None,
    beta_o: Union[np.ndarray, list, None] = None,
    sigma: float = 1.0,
    X: Optional[np.ndarray] = None,
    col_names: Optional[list] = None,
    dist: Optional[np.ndarray] = None,
    gamma_dist: float = -0.5,
    alpha: float = 0.0,
    seed: Optional[int] = None,
    gdf=None,
    err_hetero: bool = False,
    knn_k: int = 4,
    distribution: str = "lognormal",
) -> dict:
    r"""Simulate flow data from a *spatial-error* (SEM) flow model.

    Spatial-error analogue of :func:`generate_flow_data`.  The latent
    additive linear predictor is

    .. math::

        \eta = X\beta + B^{-1} \varepsilon, \quad
        B = I_N - \lambda_d W_d - \lambda_o W_o - \lambda_w W_w, \quad
        \varepsilon \sim \mathcal{N}(0, \sigma^2 I_N)

    with observed flows :math:`y = \exp(\eta)` (default ``"lognormal"``)
    or :math:`y = \eta` (``"normal"``).  Use to generate training data for
    :class:`~neighbayes.models.flow.SEMFlow`.

    Parameters and return dict mirror :func:`generate_flow_data`.
    """
    rng = np.random.default_rng(seed)

    if distribution not in ("lognormal", "normal"):
        raise ValueError(
            f"distribution must be 'lognormal' or 'normal', got {distribution!r}."
        )

    n, G, gdf = _resolve_flow_geometry(n=n, G=G, gdf=gdf, knn_k=knn_k)
    W = _graph_to_csr(G)
    N = n * n

    if beta_d is None:
        beta_d = [1.0]
    if beta_o is None:
        beta_o = [1.0]
    beta_d_arr = np.asarray(beta_d, dtype=np.float64).ravel()
    beta_o_arr = np.asarray(beta_o, dtype=np.float64).ravel()
    k_d = len(beta_d_arr)
    k_o = len(beta_o_arr)

    if X is None:
        X_d_arr = rng.standard_normal((n, k_d))
        X_o_arr = rng.standard_normal((n, k_o))
    else:
        X_arr = np.asarray(X, dtype=np.float64)
        if X_arr.ndim == 1:
            X_arr = X_arr[:, None]
        if X_arr.shape[1] == k_d + k_o:
            X_d_arr = X_arr[:, :k_d]
            X_o_arr = X_arr[:, k_d:]
        elif X_arr.shape[1] == k_d and k_d == k_o:
            X_d_arr = X_arr
            X_o_arr = X_arr
        else:
            raise ValueError(
                f"X must have {k_d + k_o} columns (split) or "
                f"{k_d} columns (symmetric), got {X_arr.shape[1]}."
            )

    if dist is None:
        dist = pairwise_distance_matrix(gdf)
    dist = np.asarray(dist, dtype=np.float64)
    if dist.shape != (n, n):
        raise ValueError(f"dist must have shape ({n}, {n}), got {dist.shape}.")

    if k_d == k_o:
        design = flow_design_matrix(
            X_d_arr, col_names=col_names, dist=dist, log_distance=True
        )
    else:
        design = flow_design_matrix_asymmetric(
            X_d_arr,
            X_o_arr,
            col_names_d=col_names,
            col_names_o=None,
            dist=dist,
            log_distance=True,
        )

    B = _flow_system(G, lam_d, lam_o, lam_w, letter="B", prefix="lam")
    _warn_flow_stability(
        lam_d,
        lam_o,
        lam_w,
        prefix="lam",
        filter_desc="the SEM flow filter B = (I_N - lam_d W_d - lam_o W_o - lam_w W_w)",
    )

    beta_full = _flow_beta_full(
        design.combined.shape[1], beta_d_arr, beta_o_arr, gamma_dist, intercept=alpha
    )

    Xbeta = design.combined @ beta_full

    if err_hetero:
        eps = _hetero_flow_eps(rng, sigma, X_d_arr, X_o_arr, n)
    else:
        eps = rng.normal(scale=sigma, size=N)

    u = _spsolve_flow(B, eps, letter="B", prefix="lam")

    eta_vec = Xbeta + u

    if distribution == "lognormal":
        y_vec = np.exp(eta_vec)
    else:
        y_vec = eta_vec

    return {
        "y_vec": y_vec,
        "y_mat": y_vec.reshape(n, n),
        "eta_vec": eta_vec,
        "eta_mat": eta_vec.reshape(n, n),
        "distribution": distribution,
        "X": design.combined,
        "X_regional": X_d_arr,
        "X_regional_d": X_d_arr,
        "X_regional_o": X_o_arr,
        "col_names": design.feature_names,
        "design": design,
        "W": W,
        "G": G,
        "gdf": gdf,
        "dist": dist,
        "lam_d": lam_d,
        "lam_o": lam_o,
        "lam_w": lam_w,
        "sigma": sigma,
        "beta_d": beta_d_arr,
        "beta_o": beta_o_arr,
        "gamma_dist": gamma_dist,
    }


def generate_sem_flow_data_separable(
    n: Optional[int] = None,
    G: Optional[Graph] = None,
    lam_d: float = 0.3,
    lam_o: float = 0.2,
    beta_d: Union[np.ndarray, list, None] = None,
    beta_o: Union[np.ndarray, list, None] = None,
    **kwargs,
) -> dict:
    r"""Simulate SEM flow data with the separability constraint :math:`\lambda_w = -\lambda_d \lambda_o`.

    Spatial-error analogue of :func:`generate_flow_data_separable`.
    """
    lam_w = -lam_d * lam_o
    return generate_sem_flow_data(
        n=n,
        G=G,
        lam_d=lam_d,
        lam_o=lam_o,
        lam_w=lam_w,
        beta_d=beta_d,
        beta_o=beta_o,
        **kwargs,
    )


def generate_panel_sem_flow_data(
    n: Optional[int] = None,
    T: int = 5,
    G: Optional[Graph] = None,
    lam_d: float = 0.3,
    lam_o: float = 0.2,
    lam_w: float = 0.1,
    beta_d: Union[np.ndarray, list, None] = None,
    beta_o: Union[np.ndarray, list, None] = None,
    sigma: float = 1.0,
    sigma_alpha: float = 0.5,
    gamma_dist: float = -0.5,
    seed: Optional[int] = None,
    k: Optional[int] = None,
    err_hetero: bool = False,
    gdf: object = None,
    knn_k: int = 4,
    distribution: str = "lognormal",
) -> dict:
    r"""Simulate panel SEM flow data with O-D-pair random effects.

    Per-period DGP:

    .. math::

        \eta_t = X_t\beta + \alpha + B^{-1}\varepsilon_t, \quad
        \varepsilon_t \sim \mathcal{N}(0, \sigma^2 I_N).

    Panel analogue of :func:`generate_panel_flow_data` for SEM.  Use to
    generate training data for
    :class:`~neighbayes.models.flow_panel.SEMFlowPanel`.
    """
    rng = np.random.default_rng(seed)

    if distribution not in ("lognormal", "normal"):
        raise ValueError(
            f"distribution must be 'lognormal' or 'normal', got {distribution!r}."
        )

    n, G, gdf = _resolve_flow_geometry(n=n, G=G, gdf=gdf, knn_k=knn_k)
    _graph_to_csr(G)
    N = n * n

    if beta_d is None:
        beta_d = [1.0]
    if beta_o is None:
        beta_o = [1.0]
    beta_d_arr = np.asarray(beta_d, dtype=np.float64).ravel()
    beta_o_arr = np.asarray(beta_o, dtype=np.float64).ravel()
    k_d_val = len(beta_d_arr)
    k_o_val = len(beta_o_arr)

    B = _flow_system(G, lam_d, lam_o, lam_w, letter="B", prefix="lam")
    solve_B = _factorized_flow(B, letter="B", prefix="lam")

    alpha = rng.normal(0.0, sigma_alpha, N) if sigma_alpha > 0 else np.zeros(N)
    dist = pairwise_distance_matrix(gdf)

    y_list, X_list, eta_list = [], [], []
    col_names = None
    for _ in range(T):
        X_d_reg = rng.standard_normal((n, k_d_val))
        X_o_reg = rng.standard_normal((n, k_o_val))
        if k_d_val == k_o_val:
            design = flow_design_matrix(X_d_reg, dist=dist, log_distance=True)
        else:
            design = flow_design_matrix_asymmetric(
                X_d_reg, X_o_reg, dist=dist, log_distance=True
            )
        if col_names is None:
            col_names = design.feature_names

        beta_full = _flow_beta_full(
            design.combined.shape[1], beta_d_arr, beta_o_arr, gamma_dist
        )

        Xbeta = design.combined @ beta_full
        eps = rng.normal(scale=sigma, size=N)
        u = solve_B(eps)
        eta_t = Xbeta + alpha + u
        y_t = np.exp(eta_t) if distribution == "lognormal" else eta_t
        eta_list.append(eta_t)
        y_list.append(y_t)
        X_list.append(design.combined)

    y = np.concatenate(y_list)
    eta = np.concatenate(eta_list)
    X = np.vstack(X_list)

    return {
        "y": y,
        "eta": eta,
        "distribution": distribution,
        "X": X,
        "col_names": col_names,
        "G": G,
        "gdf": gdf,
        "dist": dist,
        "lam_d": lam_d,
        "lam_o": lam_o,
        "lam_w": lam_w,
        "sigma": sigma,
        "sigma_alpha": sigma_alpha,
        "beta_d": beta_d_arr,
        "beta_o": beta_o_arr,
        "gamma_dist": gamma_dist,
        "params_true": {
            "lam_d": lam_d,
            "lam_o": lam_o,
            "lam_w": lam_w,
            "sigma": sigma,
            "sigma_alpha": sigma_alpha,
            "beta_d": beta_d_arr,
            "beta_o": beta_o_arr,
            "gamma_dist": gamma_dist,
            "distribution": distribution,
        },
    }


def generate_panel_sem_flow_data_separable(
    n: Optional[int] = None,
    T: int = 5,
    G: Optional[Graph] = None,
    lam_d: float = 0.3,
    lam_o: float = 0.2,
    beta_d: Union[np.ndarray, list, None] = None,
    beta_o: Union[np.ndarray, list, None] = None,
    **kwargs,
) -> dict:
    r"""Panel SEM flow DGP with separability constraint :math:`\lambda_w = -\lambda_d \lambda_o`."""
    lam_w = -lam_d * lam_o
    return generate_panel_sem_flow_data(
        n=n,
        T=T,
        G=G,
        lam_d=lam_d,
        lam_o=lam_o,
        lam_w=lam_w,
        beta_d=beta_d,
        beta_o=beta_o,
        **kwargs,
    )


def _poissonize_flow(out: dict, mean_count: float, seed: int) -> dict:
    r"""Redraw a flow dataset's response as Poisson at a target mean count.

    The reduced-form flow models use a mean propagator, :math:`\log \mu =
    A(\rho)^{-1} X\beta`, and for row-standardised weights :math:`A^{-1}`
    preserves constants — so shifting ``eta`` by a constant is an *exact*
    reparameterisation of the intercept, not an approximation.  That is what
    lets the target mean be set without disturbing the spatial parameters.
    """
    eta = out["eta_vec"]
    eta = eta + (np.log(mean_count) - np.log(np.exp(eta).mean()))
    rng = np.random.default_rng(seed)
    lam = np.exp(eta)
    y = rng.poisson(lam).astype(np.int64)
    n = int(np.sqrt(y.size))
    out = dict(out)
    out["eta_vec"] = eta
    out["lambda_vec"] = lam
    out["y_vec"] = y
    out["y_mat"] = y.reshape(n, n)
    out.pop("alpha", None)
    return out


def generate_poisson_flow_data(
    n: int | None = None,
    rho_d: float = 0.3,
    rho_o: float = 0.2,
    rho_w: float = 0.1,
    mean_count: float = 8.0,
    signal: float = 0.3,
    seed: int = 42,
    **kwargs,
) -> dict:
    r"""Generate synthetic O-D flow counts from a Poisson SAR flow DGP.

    .. math::

        y_{ij} \sim \operatorname{Poisson}(\mu_{ij}), \qquad
        \log \boldsymbol{\mu} = A(\rho_d, \rho_o, \rho_w)^{-1} X\beta

    Parameters
    ----------
    n : int, optional
        Approximate number of spatial units (``N = n^2`` flows).
    rho_d, rho_o, rho_w : float
        Destination, origin and cross spatial parameters.
    mean_count : float, default 8.0
        Target mean of the simulated counts.
    signal : float, default 0.3
        Magnitude of the regression coefficients, which sets the spread of
        ``eta``.  The default gives ``sd(eta) ~ 1`` and roughly 5-10% zeros —
        realistic gravity data.  Raising it past ~0.5 produces the very
        heavy-tailed, mostly-zero designs on which *no* flow sampler in this
        package recovers (``sd(eta) ~ 3``, 60-70% zeros); those are a bad
        test bed, not a hard one.
    seed : int, default 42
        Random seed.
    **kwargs
        Forwarded to :func:`generate_negbin_flow_data`.

    Returns
    -------
    dict
        As :func:`generate_negbin_flow_data`, without ``alpha``.
    """
    kwargs.setdefault("beta_d", [signal, signal])
    kwargs.setdefault("beta_o", [signal, signal])
    kwargs.setdefault("gamma_dist", -signal)
    out = generate_negbin_flow_data(
        n=n, rho_d=rho_d, rho_o=rho_o, rho_w=rho_w, alpha=2.0, seed=seed, **kwargs
    )
    return _poissonize_flow(out, mean_count, seed)


def generate_poisson_flow_data_separable(
    n: int | None = None,
    rho_d: float = 0.3,
    rho_o: float = 0.2,
    mean_count: float = 8.0,
    signal: float = 0.3,
    seed: int = 42,
    **kwargs,
) -> dict:
    r"""Separable Poisson flow DGP (:math:`\rho_w = -\rho_d \rho_o`).

    See :func:`generate_poisson_flow_data` for the parameters.
    """
    kwargs.setdefault("beta_d", [signal, signal])
    kwargs.setdefault("beta_o", [signal, signal])
    kwargs.setdefault("gamma_dist", -signal)
    out = generate_negbin_flow_data_separable(
        n=n, rho_d=rho_d, rho_o=rho_o, alpha=2.0, seed=seed, **kwargs
    )
    return _poissonize_flow(out, mean_count, seed)
