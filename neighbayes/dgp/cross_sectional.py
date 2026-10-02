"""Cross-sectional linear spatial DGP functions."""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import scipy.sparse as sp

from .utils import (
    _hetero_scale,
    ensure_rng,
    make_design_matrix,
    make_output_geodataframe,
    resolve_weights,
    spatial_filter_factor,
)


def _check_rho_stability(rho: float, name: str = "rho") -> None:
    """Warn when ``|rho|`` exceeds the spectral stability bound of W.

    The DGP map :math:`y = (I - \\rho W)^{-1} u` is well-defined iff
    :math:`(I - \\rho W)` is invertible, which requires
    :math:`|\\rho| < 1 / \\max_i |\\omega_i|` where :math:`\\omega_i`
    are the (real-part) eigenvalues of :math:`W`.  For row-standardized
    W this bound is ``1``.  We emit a UserWarning rather than raising
    so that callers running deliberate boundary tests can proceed.

    The DGP always uses row-standardized W, so the spectral radius is
    exactly 1.0 and we simply check ``|rho| < 1`` — no O(N³)
    eigenvalue computation needed.
    """
    if abs(rho) >= 1.0:
        warnings.warn(
            f"{name}={rho:g} is outside the stability domain "
            f"|{name}| < 1 (row-standardized W); the simulated draw "
            "may be numerically singular or unbounded.",
            stacklevel=3,
        )


def _attach_optional_gdf(
    out: dict,
    *,
    source_gdf: Any | None,
    create_gdf: bool,
    geometry_type: str,
):
    if not create_gdf and source_gdf is None:
        return out
    return make_output_geodataframe(
        y=out["y"],
        X=out["X"],
        gdf=source_gdf,
        geometry_type=geometry_type,
    )


def _simulate_sdm_core(
    *,
    nobs: int,
    Ws: sp.csr_matrix | None,
    X: np.ndarray,
    beta1: np.ndarray,
    beta2: np.ndarray,
    rho: float,
    sigma: float,
    err_hetero: bool,
    rng: np.random.Generator,
) -> np.ndarray:
    """SDM kernel: ``y = (I - rho W)^-1 (X beta1 + WX beta2 + eps)``.

    Allows degenerate cases that nest the simpler models:

    * ``len(beta2) == 0`` skips the WX term (collapses SDM -> SAR / OLS).
    * ``rho == 0`` skips the spatial solve (collapses SDM -> SLX / OLS).
    * ``Ws is None`` is permitted only when both restrictions above hold,
      enabling the OLS path with no weights matrix.
    """
    has_wx = len(beta2) > 0
    if has_wx:
        if Ws is None:
            raise ValueError("W must be supplied when beta2 is non-empty.")
        Wx = Ws @ X[:, 1:]
        if Wx.shape[1] != len(beta2):
            raise ValueError(
                "len(beta2) must match number of non-intercept regressors."
            )
        wx_beta = Wx @ beta2
    else:
        wx_beta = 0.0

    eps = (_hetero_scale(X, sigma) if err_hetero else sigma) * rng.standard_normal(nobs)
    rhs = X @ beta1 + wx_beta + eps
    if Ws is None or rho == 0.0:
        return rhs
    _check_rho_stability(rho, name="rho")
    return spatial_filter_factor(Ws, rho)(rhs)


def _simulate_sdem_core(
    *,
    nobs: int,
    Ws: sp.csr_matrix | None,
    X: np.ndarray,
    beta1: np.ndarray,
    beta2: np.ndarray,
    lam: float,
    sigma: float,
    err_hetero: bool,
    rng: np.random.Generator,
) -> np.ndarray:
    """SDEM kernel: ``y = X beta1 + WX beta2 + (I - lam W)^-1 eps``.

    Same nesting semantics as :func:`_simulate_sdm_core`: ``beta2=[]``
    drops the WX term, ``lam=0`` drops the error solve, and ``Ws=None``
    is allowed only when both restrictions hold.
    """
    has_wx = len(beta2) > 0
    if has_wx:
        if Ws is None:
            raise ValueError("W must be supplied when beta2 is non-empty.")
        Wx = Ws @ X[:, 1:]
        if Wx.shape[1] != len(beta2):
            raise ValueError(
                "len(beta2) must match number of non-intercept regressors."
            )
        wx_beta = Wx @ beta2
    else:
        wx_beta = 0.0

    eps = (_hetero_scale(X, sigma) if err_hetero else sigma) * rng.standard_normal(nobs)
    if Ws is None or lam == 0.0:
        u = eps
    else:
        _check_rho_stability(lam, name="lam")
        u = spatial_filter_factor(Ws, lam)(eps)
    return X @ beta1 + wx_beta + u


def simulate_sar(
    n_side: int | None = None,
    W=None,
    gdf=None,
    rho: float = 0.5,
    beta: np.ndarray | None = None,
    sigma: float = 1.0,
    err_hetero: bool = False,
    rng: np.random.Generator | None = None,
    seed: int | None = None,
    contiguity: str = "queen",
    create_gdf: bool = False,
    geometry_type: str = "polygon",
) -> dict:
    """Simulate data from SAR DGP ``y = (I-rho W)^(-1)(X beta + eps)``.

    Parameters
    ----------
    n_side : int, optional
        Side length of the square rook grid used when neither ``W`` nor
        ``gdf`` is supplied; the grid has ``n_side**2`` observations.  When
        ``W`` or ``gdf`` is provided, ``n_side**2`` (if given) must match
        the number of units.
    W : Graph or scipy.sparse matrix, optional
        Spatial weights. If supplied, takes precedence over ``gdf``.
    gdf : geopandas.GeoDataFrame, optional
        Spatial units source used when ``W`` is not provided.
    rho : float, default=0.5
        Spatial autoregressive coefficient.
    beta : np.ndarray, optional
        Regression coefficients including intercept.
    sigma : float, default=1.0
        Innovation standard deviation.
    err_hetero : bool, default=False
        If True, generate heteroskedastic innovations with
        observation-specific standard deviations
        :math:`\\sigma_i = \\sigma \\sqrt{1 + \\|x_i\\|^2}`.
    rng : np.random.Generator, optional
        Random generator.
    seed : int, optional
        Seed used when ``rng`` is not supplied.
    contiguity : str, default="queen"
        Neighbor rule for ``gdf`` mode.
    create_gdf : bool, default=False
        If True, include a ``gdf`` key in the returned dict with ``y`` and
        ``X_*`` columns attached to geometry.
    geometry_type : {"point", "polygon"}, default="polygon"
        Geometry type to generate when ``create_gdf=True`` and ``gdf`` is not
        provided.

    Returns
    -------
    dict
        Keys: ``y``, ``X``, ``W_sparse``, ``W_graph``, ``params_true``.

    Notes
    -----
    Equivalent to ``simulate_sdm`` with ``beta2=[]`` (no WX terms); see
    :func:`simulate_sdm` for the unified Spatial Durbin form.
    """
    rng = ensure_rng(rng, seed)
    Ws, Wg = resolve_weights(W=W, gdf=gdf, n_side=n_side, contiguity=contiguity)
    nobs = Ws.shape[0]

    if beta is None:
        beta = np.array([1.0, 2.0], dtype=float)
    beta = np.asarray(beta, dtype=float)

    X = make_design_matrix(rng, nobs, k=max(len(beta) - 1, 0), add_intercept=True)
    y = _simulate_sdm_core(
        nobs=nobs,
        Ws=Ws,
        X=X,
        beta1=beta,
        beta2=np.empty(0, dtype=float),
        rho=rho,
        sigma=sigma,
        err_hetero=err_hetero,
        rng=rng,
    )
    out = {
        "y": y,
        "X": X,
        "W_sparse": Ws,
        "W_graph": Wg,
        "params_true": {"rho": rho, "beta": beta, "sigma": sigma},
    }
    return _attach_optional_gdf(
        out,
        source_gdf=gdf,
        create_gdf=create_gdf,
        geometry_type=geometry_type,
    )


def simulate_sar_negbin(
    n_side: int | None = None,
    W=None,
    gdf=None,
    rho: float = 0.5,
    beta: np.ndarray | None = None,
    alpha: float = 2.0,
    sigma2: float = 0.0,
    err_hetero: bool = False,
    rng: np.random.Generator | None = None,
    seed: int | None = None,
    contiguity: str = "queen",
    create_gdf: bool = False,
    geometry_type: str = "polygon",
) -> dict:
    r"""Simulate data from a SAR-NB2 DGP.

    The latent log-mean follows the SAR structural form:

    .. math::

        \eta = \rho W \eta + X\beta + \nu, \quad \nu \sim N(0, \sigma^2 I)

    which in reduced form is:

    .. math::

        \eta = (I - \rho W)^{-1}(X\beta + \nu), \quad \mu = \exp(\eta)

    When ``sigma2 = 0`` (the default), the DGP reduces to the
    deterministic reduced form :math:`\eta = (I - \rho W)^{-1} X\beta`,
    matching the :class:`SARNegBin` model specification.
    When ``sigma2 > 0``, the structural-form noise is included,
    matching the :class:`SARNegBinStructural` model specification.

    Counts are sampled as NB2:

    .. math::

        y_i \sim \mathrm{NegBin}(\mu_i, \alpha),
        \;\;\mathrm{Var}(y_i)=\mu_i+\mu_i^2/\alpha.

    Parameters
    ----------
    n_side : int, optional
        Side length of the square rook grid used when neither ``W`` nor
        ``gdf`` is supplied; the grid has ``n_side**2`` observations.  When
        ``W`` or ``gdf`` is provided, ``n_side**2`` (if given) must match
        the number of units.
    W : Graph or array-like, optional
        Spatial weights.
    gdf : GeoDataFrame, optional
        Geodataframe used to construct weights.
    rho : float, default 0.5
        Spatial autoregressive parameter.
    beta : ndarray, optional
        Regression coefficients (including intercept). Defaults to
        ``[1.0, 0.6]``.
    alpha : float, default 2.0
        NB2 dispersion parameter. Must be strictly positive.
    sigma2 : float, default 0.0
        Structural-form residual variance. When 0, the DGP is
        deterministic (no noise in the latent field). When > 0,
        Gaussian noise is added to the structural form.
    err_hetero : bool, default False
        Heteroskedastic errors (not yet implemented; ignored with a
        warning).
    rng : numpy.random.Generator, optional
        Random number generator.
    seed : int, optional
        Random seed (used only if rng is None).
    contiguity : str, default "queen"
        Neighbor rule used when W is built from ``gdf``.
    create_gdf : bool, default False
        Whether to attach a GeoDataFrame to the output.
    geometry_type : str, default "polygon"
        Type of geometry for the GeoDataFrame.

    Returns
    -------
    dict
        Dictionary with keys ``y``, ``X``, ``mu``, ``W_sparse``,
        ``W_graph``, and ``params_true``. When ``sigma2 > 0``,
        ``params_true`` also includes ``sigma2``.
    """

    if alpha <= 0:
        raise ValueError("alpha must be strictly positive.")
    if sigma2 < 0:
        raise ValueError("sigma2 must be non-negative.")
    if err_hetero:
        warnings.warn(
            "err_hetero is not implemented for simulate_sar_negbin and is ignored.",
            stacklevel=2,
        )

    rng = ensure_rng(rng, seed)
    Ws, Wg = resolve_weights(W=W, gdf=gdf, n_side=n_side, contiguity=contiguity)
    nobs = Ws.shape[0]

    if beta is None:
        beta = np.array([1.0, 0.6], dtype=float)
    beta = np.asarray(beta, dtype=float)

    X = make_design_matrix(rng, nobs, k=max(len(beta) - 1, 0), add_intercept=True)
    _check_rho_stability(rho, name="rho")

    solve = spatial_filter_factor(Ws, rho)

    # Structural form: (I - rho*W) eta = X beta + nu
    # When sigma2 > 0, add Gaussian noise nu ~ N(0, sigma2 I)
    if sigma2 > 0:
        nu = rng.normal(0, np.sqrt(sigma2), size=nobs)
        eta = solve(X @ beta + nu)
    else:
        eta = solve(X @ beta)

    mu = np.exp(np.clip(eta, -30.0, 30.0))

    p = alpha / (alpha + mu)
    y = rng.negative_binomial(alpha, p).astype(np.float64)

    params_true = {"rho": rho, "beta": beta, "alpha": alpha}
    if sigma2 > 0:
        params_true["sigma2"] = sigma2

    out = {
        "y": y,
        "X": X,
        "mu": mu,
        "W_sparse": Ws,
        "W_graph": Wg,
        "params_true": params_true,
    }
    return _attach_optional_gdf(
        out,
        source_gdf=gdf,
        create_gdf=create_gdf,
        geometry_type=geometry_type,
    )


def simulate_ols(
    n_side: int | None = None,
    W=None,
    gdf=None,
    beta: np.ndarray | None = None,
    sigma: float = 1.0,
    err_hetero: bool = False,
    rng: np.random.Generator | None = None,
    seed: int | None = None,
    contiguity: str = "queen",
    create_gdf: bool = False,
    geometry_type: str = "polygon",
) -> dict:
    """Simulate data from a non-spatial OLS DGP ``y = X beta + eps``.

    Generates a random design matrix with an intercept and ``len(beta) - 1``
    continuous regressors, and draws the response from a homoskedastic
    Normal error model. No spatial weights matrix is required or produced;
    this function is the natural complement to the spatial DGPs for use as
    a non-spatial baseline.

    Parameters
    ----------
    n_side : int, optional
        Side length of the square rook grid used when neither ``W`` nor
        ``gdf`` is supplied; the grid has ``n_side**2`` observations.  When
        ``W`` or ``gdf`` is provided, ``n_side**2`` (if given) must match
        the number of units.
    W : Graph or scipy.sparse matrix, optional
        Spatial weights input used only to infer the number of observations.
        Not used in the OLS data-generating mechanism.
    gdf : geopandas.GeoDataFrame, optional
        Spatial units source used only to infer the number of observations
        when ``W`` is not provided.
    beta : array-like, optional
        Coefficient vector including intercept. Defaults to
        ``[1.0, 2.0]`` (intercept = 1, one regressor with slope = 2).
    sigma : float, default=1.0
        Innovation standard deviation :math:`\\sigma`.
    err_hetero : bool, default=False
        If True, generate heteroskedastic innovations with
        observation-specific standard deviations
        :math:`\\sigma_i = \\sigma \\sqrt{1 + \\|x_i\\|^2}`.
    rng : numpy.random.Generator, optional
        Random generator instance for reproducibility.
    seed : int, optional
        Integer seed used when ``rng`` is not supplied.
    contiguity : str, default="queen"
        Neighbor rule used when inferring the size from ``gdf``.
    create_gdf : bool, default=False
        If ``True``, attaches a GeoDataFrame with ``y`` and ``X_*`` columns
        to geometry generated on an ``n_side x n_side`` grid.
    geometry_type : {"point", "polygon"}, default="polygon"
        Geometry type to generate when ``create_gdf=True``.

    Returns
    -------
    dict
        Keys:

        - ``y`` : np.ndarray of shape ``(n,)`` — response variable.
        - ``X`` : np.ndarray of shape ``(n, k)`` — design matrix with
          intercept in the first column.
        - ``params_true`` : dict with ``beta`` and ``sigma``.
        - ``gdf`` : GeoDataFrame (only present when ``create_gdf=True``).

    Notes
    -----
    Equivalent to ``simulate_sdm`` with ``rho=0`` and ``beta2=[]``; the
    spatial weights matrix is ignored even when supplied. See
    :func:`simulate_sdm` for the unified Spatial Durbin form.
    """

    rng = ensure_rng(rng, seed)

    if n_side is None and W is None and gdf is None:
        raise ValueError("Provide one of n_side, W, or gdf.")

    if W is not None or gdf is not None:
        Ws, _ = resolve_weights(W=W, gdf=gdf, n_side=n_side, contiguity=contiguity)
        nobs = Ws.shape[0]
    else:
        nobs = int(n_side) ** 2

    if beta is None:
        beta = np.array([1.0, 2.0], dtype=float)
    beta = np.asarray(beta, dtype=float)

    X = make_design_matrix(rng, nobs, k=max(len(beta) - 1, 0), add_intercept=True)
    y = _simulate_sdm_core(
        nobs=nobs,
        Ws=None,
        X=X,
        beta1=beta,
        beta2=np.empty(0, dtype=float),
        rho=0.0,
        sigma=sigma,
        err_hetero=err_hetero,
        rng=rng,
    )

    out: dict = {
        "y": y,
        "X": X,
        "params_true": {"beta": beta, "sigma": sigma},
    }
    return _attach_optional_gdf(
        out,
        source_gdf=gdf,
        create_gdf=create_gdf,
        geometry_type=geometry_type,
    )


def simulate_sem(
    n_side: int | None = None,
    W=None,
    gdf=None,
    lam: float = 0.5,
    beta: np.ndarray | None = None,
    sigma: float = 1.0,
    err_hetero: bool = False,
    rng: np.random.Generator | None = None,
    seed: int | None = None,
    contiguity: str = "queen",
    create_gdf: bool = False,
    geometry_type: str = "polygon",
) -> dict:
    """Simulate data from SEM DGP ``y = X beta + (I-lam W)^(-1) eps``.

    Parameters are analogous to :func:`simulate_sar` with ``lam`` replacing
    ``rho``.

    Returns
    -------
    dict
        Keys: ``y``, ``X``, ``W_sparse``, ``W_graph``, ``params_true``.

    Notes
    -----
    Equivalent to ``simulate_sdem`` with ``beta2=[]`` (no WX terms); see
    :func:`simulate_sdem` for the unified Spatial Durbin Error form.
    """
    rng = ensure_rng(rng, seed)
    Ws, Wg = resolve_weights(W=W, gdf=gdf, n_side=n_side, contiguity=contiguity)
    nobs = Ws.shape[0]

    if beta is None:
        beta = np.array([1.0, 2.0], dtype=float)
    beta = np.asarray(beta, dtype=float)

    X = make_design_matrix(rng, nobs, k=max(len(beta) - 1, 0), add_intercept=True)
    y = _simulate_sdem_core(
        nobs=nobs,
        Ws=Ws,
        X=X,
        beta1=beta,
        beta2=np.empty(0, dtype=float),
        lam=lam,
        sigma=sigma,
        err_hetero=err_hetero,
        rng=rng,
    )
    out = {
        "y": y,
        "X": X,
        "W_sparse": Ws,
        "W_graph": Wg,
        "params_true": {"lam": lam, "beta": beta, "sigma": sigma},
    }
    return _attach_optional_gdf(
        out,
        source_gdf=gdf,
        create_gdf=create_gdf,
        geometry_type=geometry_type,
    )


def simulate_slx(
    n_side: int | None = None,
    W=None,
    gdf=None,
    beta1: np.ndarray | None = None,
    beta2: np.ndarray | None = None,
    sigma: float = 1.0,
    err_hetero: bool = False,
    rng: np.random.Generator | None = None,
    seed: int | None = None,
    contiguity: str = "queen",
    create_gdf: bool = False,
    geometry_type: str = "polygon",
) -> dict:
    """Simulate data from SLX DGP ``y = X beta1 + W X_no_intercept beta2 + eps``.

    Returns
    -------
    dict
        Keys: ``y``, ``X``, ``W_sparse``, ``W_graph``, ``params_true``.
    """
    rng = ensure_rng(rng, seed)
    Ws, Wg = resolve_weights(W=W, gdf=gdf, n_side=n_side, contiguity=contiguity)
    nobs = Ws.shape[0]

    if beta1 is None:
        beta1 = np.array([1.0, 2.0], dtype=float)
    beta1 = np.asarray(beta1, dtype=float)
    if beta2 is None:
        beta2 = np.array([0.8], dtype=float)
    beta2 = np.asarray(beta2, dtype=float)

    X = make_design_matrix(rng, nobs, k=max(len(beta1) - 1, 0), add_intercept=True)
    y = _simulate_sdm_core(
        nobs=nobs,
        Ws=Ws,
        X=X,
        beta1=beta1,
        beta2=beta2,
        rho=0.0,
        sigma=sigma,
        err_hetero=err_hetero,
        rng=rng,
    )
    out = {
        "y": y,
        "X": X,
        "W_sparse": Ws,
        "W_graph": Wg,
        "params_true": {"beta1": beta1, "beta2": beta2, "sigma": sigma},
    }
    return _attach_optional_gdf(
        out,
        source_gdf=gdf,
        create_gdf=create_gdf,
        geometry_type=geometry_type,
    )


def simulate_sdm(
    n_side: int | None = None,
    W=None,
    gdf=None,
    rho: float = 0.4,
    beta1: np.ndarray | None = None,
    beta2: np.ndarray | None = None,
    sigma: float = 1.0,
    err_hetero: bool = False,
    rng: np.random.Generator | None = None,
    seed: int | None = None,
    contiguity: str = "queen",
    create_gdf: bool = False,
    geometry_type: str = "polygon",
) -> dict:
    """Simulate data from SDM DGP ``y = (I-rho W)^(-1)(Xb1 + WXb2 + eps)``.

    Returns
    -------
    dict
        Keys: ``y``, ``X``, ``W_sparse``, ``W_graph``, ``params_true``.
    """
    rng = ensure_rng(rng, seed)
    Ws, Wg = resolve_weights(W=W, gdf=gdf, n_side=n_side, contiguity=contiguity)
    nobs = Ws.shape[0]

    if beta1 is None:
        beta1 = np.array([1.0, 2.0], dtype=float)
    beta1 = np.asarray(beta1, dtype=float)
    if beta2 is None:
        beta2 = np.array([0.8], dtype=float)
    beta2 = np.asarray(beta2, dtype=float)

    X = make_design_matrix(rng, nobs, k=max(len(beta1) - 1, 0), add_intercept=True)
    y = _simulate_sdm_core(
        nobs=nobs,
        Ws=Ws,
        X=X,
        beta1=beta1,
        beta2=beta2,
        rho=rho,
        sigma=sigma,
        err_hetero=err_hetero,
        rng=rng,
    )
    out = {
        "y": y,
        "X": X,
        "W_sparse": Ws,
        "W_graph": Wg,
        "params_true": {"rho": rho, "beta1": beta1, "beta2": beta2, "sigma": sigma},
    }
    return _attach_optional_gdf(
        out,
        source_gdf=gdf,
        create_gdf=create_gdf,
        geometry_type=geometry_type,
    )


def simulate_sdem(
    n_side: int | None = None,
    W=None,
    gdf=None,
    lam: float = 0.4,
    beta1: np.ndarray | None = None,
    beta2: np.ndarray | None = None,
    sigma: float = 1.0,
    err_hetero: bool = False,
    rng: np.random.Generator | None = None,
    seed: int | None = None,
    contiguity: str = "queen",
    create_gdf: bool = False,
    geometry_type: str = "polygon",
) -> dict:
    """Simulate data from SDEM DGP ``y = Xb1 + WXb2 + (I-lam W)^(-1)eps``.

    Returns
    -------
    dict
        Keys: ``y``, ``X``, ``W_sparse``, ``W_graph``, ``params_true``.
    """
    rng = ensure_rng(rng, seed)
    Ws, Wg = resolve_weights(W=W, gdf=gdf, n_side=n_side, contiguity=contiguity)
    nobs = Ws.shape[0]

    if beta1 is None:
        beta1 = np.array([1.0, 2.0], dtype=float)
    beta1 = np.asarray(beta1, dtype=float)
    if beta2 is None:
        beta2 = np.array([0.8], dtype=float)
    beta2 = np.asarray(beta2, dtype=float)

    X = make_design_matrix(rng, nobs, k=max(len(beta1) - 1, 0), add_intercept=True)
    y = _simulate_sdem_core(
        nobs=nobs,
        Ws=Ws,
        X=X,
        beta1=beta1,
        beta2=beta2,
        lam=lam,
        sigma=sigma,
        err_hetero=err_hetero,
        rng=rng,
    )
    out = {
        "y": y,
        "X": X,
        "W_sparse": Ws,
        "W_graph": Wg,
        "params_true": {"lam": lam, "beta1": beta1, "beta2": beta2, "sigma": sigma},
    }
    return _attach_optional_gdf(
        out,
        source_gdf=gdf,
        create_gdf=create_gdf,
        geometry_type=geometry_type,
    )
