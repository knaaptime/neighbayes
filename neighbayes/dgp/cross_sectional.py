"""Cross-sectional linear spatial DGP functions."""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import scipy.sparse as sp

from ._design import aspatial, count, gaussian, simulator
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


@simulator(gaussian)
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


@simulator(count)
def simulate_sar_negbin(
    n_side: int | None = None,
    W=None,
    gdf=None,
    rho: float = 0.5,
    beta: np.ndarray | None = None,
    alpha: float = 2.0,
    err_hetero: bool = False,
    rng: np.random.Generator | None = None,
    seed: int | None = None,
    contiguity: str = "queen",
    create_gdf: bool = False,
    geometry_type: str = "polygon",
    latent: bool = False,
) -> dict:
    r"""Simulate data from the reduced-form SAR-NB2 that :class:`SARNegBin` fits.

    .. math::

        \eta = (I - \rho W)^{-1} X\beta, \quad \mu = \exp(\eta)

    With ``T`` (and ``N``) it draws the :class:`SARNegBin` panel, whose random
    unit effects (``unit_effects="random"``) enter inside the filter.  With
    ``latent=True`` it draws the latent SAR that ``SARNegBin(latent=True)``
    fits, whose filter propagates the negative binomial's gamma disturbance:

    .. math::

        \eta = (I - \rho W)^{-1}(X\beta + \varepsilon), \quad
        \varepsilon_i = \log\gamma_i,\; \gamma_i \sim \mathrm{Gamma}(\alpha,
        \alpha), \quad y_i \sim \mathrm{Poisson}(e^{\eta_i}).

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
    err_hetero : bool, default False
        Scale the latent SAR's disturbances by the norm of each row of ``X``.
        The reduced form has no disturbance, so it ignores the flag with a
        warning.
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
    latent : bool, default False
        Draw the latent SAR instead of the reduced form; ``eta_true`` is then
        the log-mean.

    Returns
    -------
    dict
        Dictionary with keys ``y``, ``X``, ``mu``, ``W_sparse``,
        ``W_graph``, and ``params_true`` (and ``eta_true`` when latent).
    """

    if alpha <= 0:
        raise ValueError("alpha must be strictly positive.")
    if latent:
        return _simulate_latent_negbin(
            "sar", n_side, W, gdf, rho, beta, None, alpha, err_hetero, rng, seed,
            contiguity, create_gdf, geometry_type,
        )  # fmt: skip
    if err_hetero:
        warnings.warn(
            "err_hetero is ignored by the reduced form, which has no disturbance.",
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

    eta = solve(X @ beta)

    mu = np.exp(np.clip(eta, -30.0, 30.0))

    p = alpha / (alpha + mu)
    y = rng.negative_binomial(alpha, p).astype(np.float64)

    params_true = {"rho": rho, "beta": beta, "alpha": alpha}

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


def _simulate_latent_negbin(
    process, n_side, W, gdf, par, beta, theta, alpha, err_hetero, rng, seed,
    contiguity, create_gdf, geometry_type,
) -> dict:  # fmt: skip
    """The latent SAR, SEM or SDEM negative binomial: one gamma disturbance each."""
    if alpha <= 0:
        raise ValueError("alpha must be strictly positive.")
    rng = ensure_rng(rng, seed)
    Ws, Wg = resolve_weights(W=W, gdf=gdf, n_side=n_side, contiguity=contiguity)
    nobs = Ws.shape[0]
    beta = np.array([1.0, 0.6]) if beta is None else np.asarray(beta, dtype=float)
    X = make_design_matrix(rng, nobs, k=max(len(beta) - 1, 0), add_intercept=True)
    name = "rho" if process == "sar" else "lam"
    _check_rho_stability(par, name=name)
    eps = np.log(rng.gamma(alpha, 1.0 / alpha, size=nobs))
    if err_hetero:
        eps = eps * _hetero_scale(X, 1.0)
    solve = spatial_filter_factor(Ws, par)
    params_true = {name: par, "beta": beta, "alpha": alpha}
    if process == "sar":
        eta = solve(X @ beta + eps)
    else:
        mean = X @ beta
        if theta is not None:
            mean = mean + np.asarray(Ws @ X[:, 1:]) @ theta
            params_true["theta"] = theta
        eta = mean + solve(eps)
    mu = np.exp(np.clip(eta, -30.0, 30.0))
    out = {
        "y": rng.poisson(mu).astype(np.float64),
        "X": X,
        "mu": mu,
        "eta_true": eta,
        "W_sparse": Ws,
        "W_graph": Wg,
        "params_true": params_true,
    }
    return _attach_optional_gdf(
        out, source_gdf=gdf, create_gdf=create_gdf, geometry_type=geometry_type
    )


@simulator(count)
def simulate_sem_negbin(
    n_side: int | None = None,
    W=None,
    gdf=None,
    lam: float = 0.5,
    beta: np.ndarray | None = None,
    alpha: float = 2.0,
    err_hetero: bool = False,
    rng: np.random.Generator | None = None,
    seed: int | None = None,
    contiguity: str = "queen",
    create_gdf: bool = False,
    geometry_type: str = "polygon",
) -> dict:
    r"""Simulate the latent SEM negative binomial that :class:`SEMNegBin` fits.

    .. math::

        \eta = X\beta + (I - \lambda W)^{-1}\varepsilon, \quad
        \varepsilon_i = \log\gamma_i,\; \gamma_i \sim \mathrm{Gamma}(\alpha,
        \alpha), \quad y_i \sim \mathrm{Poisson}(e^{\eta_i})

    so that ``lam = 0`` draws the NB2 of :class:`NegBin`.  Parameters as for
    :func:`simulate_sar_negbin` with ``latent=True``, the spatial parameter
    being ``lam``; ``eta_true`` is the log-mean.
    """
    return _simulate_latent_negbin(
        "sem", n_side, W, gdf, lam, beta, None, alpha, err_hetero, rng, seed,
        contiguity, create_gdf, geometry_type,
    )  # fmt: skip


@simulator(count)
def simulate_sdem_negbin(
    n_side: int | None = None,
    W=None,
    gdf=None,
    lam: float = 0.5,
    beta: np.ndarray | None = None,
    theta: np.ndarray | None = None,
    alpha: float = 2.0,
    err_hetero: bool = False,
    rng: np.random.Generator | None = None,
    seed: int | None = None,
    contiguity: str = "queen",
    create_gdf: bool = False,
    geometry_type: str = "polygon",
) -> dict:
    r"""Simulate the latent SDEM negative binomial that :class:`SDEMNegBin` fits.

    .. math::

        \eta = X\beta + WX\theta + (I - \lambda W)^{-1}\varepsilon

    with the disturbances of :func:`simulate_sem_negbin`; ``theta`` (default
    0.5 for each covariate) multiplies the lags of the non-intercept columns.
    """
    beta_ = np.array([1.0, 0.6]) if beta is None else np.asarray(beta, dtype=float)
    theta = np.full(len(beta_) - 1, 0.5) if theta is None else np.asarray(theta, float)
    return _simulate_latent_negbin(
        "sem", n_side, W, gdf, lam, beta_, theta, alpha, err_hetero, rng, seed,
        contiguity, create_gdf, geometry_type,
    )  # fmt: skip


@simulator(gaussian)
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


@simulator(gaussian)
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


@simulator(gaussian)
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


@simulator(gaussian)
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


@simulator(gaussian)
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


@aspatial(simulate_sar_negbin, {"rho": 0.0}, xs_random=False)
def simulate_negbin():
    """Simulate an aspatial NB2 regression: :func:`simulate_sar_negbin` with ``rho = 0``."""
