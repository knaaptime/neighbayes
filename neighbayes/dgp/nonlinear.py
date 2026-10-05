"""Nonlinear and censored DGP functions.

Simulates data from spatial Tobit and probit models, including left-censored
SAR, SEM, and SDM Tobit variants, as well as spatial probit with regional
random effects.
"""

from __future__ import annotations

import numpy as np
from scipy.special import erf

from .cross_sectional import (
    _attach_optional_gdf,
    simulate_sar,
    simulate_sdm,
    simulate_sem,
)
from .utils import (
    _hetero_scale,
    _left_censor,
    ensure_rng,
    make_design_matrix,
    resolve_weights,
    spatial_filter_factor,
)


def simulate_sar_tobit(
    *, censoring: float = 0.0, err_hetero: bool = False, **kwargs
) -> dict:
    """Simulate left-censored SAR Tobit data.

    Parameters
    ----------
    censoring : float, default=0.0
        Left-censoring threshold ``c`` where observed ``y = max(c, y*)``.
    err_hetero : bool, default=False
        If True, generate heteroskedastic innovations. Forwarded to
        :func:`simulate_sar`.
    **kwargs
        Forwarded to :func:`simulate_sar`.

    Returns
    -------
    dict
        Includes ``y`` (observed), ``y_latent``, and ``censored_mask``.
    """
    out = simulate_sar(err_hetero=err_hetero, **kwargs)
    y_obs, mask = _left_censor(out["y"], censoring)
    out["y_latent"] = out["y"]
    out["y"] = y_obs
    out["censored_mask"] = mask
    out["params_true"]["censoring"] = censoring
    return out


def simulate_sem_tobit(
    *, censoring: float = 0.0, err_hetero: bool = False, **kwargs
) -> dict:
    """Simulate left-censored SEM Tobit data.

    Parameters
    ----------
    censoring : float, default=0.0
        Left-censoring threshold ``c``.
    err_hetero : bool, default=False
        If True, generate heteroskedastic innovations. Forwarded to
        :func:`simulate_sem`.
    **kwargs
        Forwarded to :func:`simulate_sem`.

    Returns
    -------
    dict
        Includes ``y`` (observed), ``y_latent``, and ``censored_mask``.
    """
    out = simulate_sem(err_hetero=err_hetero, **kwargs)
    y_obs, mask = _left_censor(out["y"], censoring)
    out["y_latent"] = out["y"]
    out["y"] = y_obs
    out["censored_mask"] = mask
    out["params_true"]["censoring"] = censoring
    return out


def simulate_sdm_tobit(
    *, censoring: float = 0.0, err_hetero: bool = False, **kwargs
) -> dict:
    """Simulate left-censored SDM Tobit data.

    Parameters
    ----------
    censoring : float, default=0.0
        Left-censoring threshold ``c``.
    err_hetero : bool, default=False
        If True, generate heteroskedastic innovations. Forwarded to
        :func:`simulate_sdm`.
    **kwargs
        Forwarded to :func:`simulate_sdm`.

    Returns
    -------
    dict
        Includes ``y`` (observed), ``y_latent``, and ``censored_mask``.
    """
    out = simulate_sdm(err_hetero=err_hetero, **kwargs)
    y_obs, mask = _left_censor(out["y"], censoring)
    out["y_latent"] = out["y"]
    out["y"] = y_obs
    out["censored_mask"] = mask
    out["params_true"]["censoring"] = censoring
    return out


def simulate_spatial_probit(
    W=None,
    gdf=None,
    n_side: int | None = None,
    rho: float = 0.35,
    beta: np.ndarray | None = None,
    sigma_a: float = 0.8,
    n_per_region: int = 25,
    err_hetero: bool = False,
    rng: np.random.Generator | None = None,
    seed: int | None = None,
    contiguity: str = "queen",
) -> dict:
    """Simulate SARProbit-style binary outcome data.

    DGP
    ---
    ``a = (I-rho W)^(-1) sigma_a z`` (region effects),
    ``eta = X beta + a[region]``,
    ``y ~ Bernoulli(Phi(eta))``.

    Parameters
    ----------
    W, gdf
        Spatial unit structure. If ``W`` is provided it takes precedence;
        otherwise ``gdf`` is used with ``contiguity``.
    n_side : int, optional
        Side length of the square rook grid used when neither ``W`` nor
        ``gdf`` is supplied (``n_side**2`` units).
    rho : float, default=0.35
        Spatial dependence in regional effects.
    beta : np.ndarray, optional
        Coefficients including intercept. Defaults to ``[0.3, 1.0]``.
    sigma_a : float, default=0.8
        Regional effect innovation scale.
    n_per_region : int, default=25
        Number of observations per region.
    err_hetero : bool, default=False
        If True, generate heteroskedastic region effects with
        region-specific standard deviations
        :math:`\\sigma_{a,j} = \\sigma_a \\sqrt{1 + \\|\\bar{x}_j\\|^2}`
        where :math:`\\bar{x}_j` is the mean regressor vector for region
        ``j``.
    rng, seed
        Random state controls.
    contiguity : str, default="queen"
        GeoDataFrame neighbor rule when ``W`` is omitted.

    Returns
    -------
    dict
        Keys: ``y``, ``X``, ``region_ids``, ``W_sparse``, ``W_graph``,
        ``params_true``.
    """
    rng = ensure_rng(rng, seed)
    W_sparse, Wg = resolve_weights(W=W, gdf=gdf, n_side=n_side, contiguity=contiguity)
    m = W_sparse.shape[0]

    if beta is None:
        beta = np.array([0.3, 1.0], dtype=float)
    beta = np.asarray(beta, dtype=float)

    nobs = int(m * n_per_region)
    region_ids = np.repeat(np.arange(m), n_per_region)

    if err_hetero:
        # Generate X first so we can compute region-level means for
        # heteroskedastic scaling.  This changes the RNG draw order
        # relative to the homoskedastic path.
        X = make_design_matrix(rng, nobs, k=max(len(beta) - 1, 0), add_intercept=True)
        X_region_mean = X.reshape(m, n_per_region, -1).mean(axis=1)
        a_scale = _hetero_scale(X_region_mean, sigma_a)
    else:
        # Preserve the original RNG draw order: a is drawn before X.
        a_scale = sigma_a

    a = spatial_filter_factor(W_sparse, rho)(a_scale * rng.standard_normal(m))

    if not err_hetero:
        X = make_design_matrix(rng, nobs, k=max(len(beta) - 1, 0), add_intercept=True)

    eta = X @ beta + a[region_ids]
    p = 0.5 * (1.0 + erf(eta / np.sqrt(2.0)))
    y = rng.binomial(1, p).astype(float)

    return {
        "y": y,
        "X": X,
        "region_ids": region_ids,
        "W_sparse": W_sparse,
        "W_graph": Wg,
        "params_true": {
            "rho": rho,
            "beta": beta,
            "sigma_a": sigma_a,
            "n_per_region": n_per_region,
        },
    }


def simulate_sar_logit(
    W=None,
    gdf=None,
    n_side: int | None = None,
    rho: float = 0.35,
    beta: np.ndarray | None = None,
    rng: np.random.Generator | None = None,
    seed: int | None = None,
    contiguity: str = "queen",
    create_gdf: bool = False,
    geometry_type: str = "polygon",
    structural: bool = True,
    **kwargs,
) -> dict:
    """Simulate SAR-logit binary outcome data.

    DGP
    ---
    ``eta = (I - rho W)^{-1} (X beta + nu)``, ``nu ~ N(0, I)``,
    ``y ~ Bernoulli(logit^{-1}(eta))`` — the latent-field model
    :class:`~neighbayes.models.SARLogitStructural` fits.  With
    ``structural=False`` the noise is omitted, ``eta = (I - rho W)^{-1} X beta``,
    which is the reduced-form :class:`~neighbayes.models.SARLogit`.

    Parameters
    ----------
    W, gdf
        Spatial unit structure. If ``W`` is provided it takes precedence;
        otherwise ``gdf`` is used with ``contiguity``.
    n_side : int, optional
        Side length of the square rook grid used when neither ``W`` nor
        ``gdf`` is supplied (``n_side**2`` units).
    rho : float, default=0.35
        Spatial autoregressive parameter.
    beta : np.ndarray, optional
        Coefficients including intercept. Defaults to ``[0.3, 1.0]``.
    rng, seed
        Random state controls.
    contiguity : str, default="queen"
        GeoDataFrame neighbor rule when ``W`` is omitted.
    create_gdf : bool, default=False
        If ``True`` (or a source ``gdf`` is supplied), return a
        GeoDataFrame carrying ``y`` and ``X_*`` columns instead of the
        dict, matching the other cross-sectional simulators.
    geometry_type : {"point", "polygon"}, default="polygon"
        Geometry to generate when ``create_gdf=True`` and no source
        ``gdf`` is provided.
    structural : bool, default True
        Include the latent ``nu`` (``SARLogitStructural``); ``False`` gives the
        reduced form (``SARLogit``).

    Returns
    -------
    dict or geopandas.GeoDataFrame
        Dict with keys ``y``, ``X``, ``W_sparse``, ``W_graph``,
        ``eta_true``, ``params_true``.  When ``create_gdf=True`` (or a
        source ``gdf`` is supplied) a GeoDataFrame with ``y`` and
        ``X_*`` columns is returned instead.
    """
    rng = ensure_rng(rng, seed)
    W_sparse, Wg = resolve_weights(W=W, gdf=gdf, n_side=n_side, contiguity=contiguity)
    n_obs = W_sparse.shape[0]

    if beta is None:
        beta = np.array([0.3, 1.0], dtype=float)
    beta = np.asarray(beta, dtype=float)

    X = make_design_matrix(rng, n_obs, k=max(len(beta) - 1, 0), add_intercept=True)

    # Latent field eta = (I - rho W)^{-1} (X beta + nu); nu = 0 in the reduced form.
    nu = rng.standard_normal(n_obs) if structural else 0.0
    Xbeta = X @ beta
    eta = spatial_filter_factor(W_sparse, rho)(Xbeta + nu)

    # Binary response: y ~ Bernoulli(logit^{-1}(eta))
    probs = 1.0 / (1.0 + np.exp(-eta))
    y = rng.binomial(1, probs).astype(float)

    out = {
        "y": y,
        "X": X,
        "W_sparse": W_sparse,
        "W_graph": Wg,
        "eta_true": eta,
        "params_true": {
            "rho": rho,
            "beta": beta,
        },
    }
    return _attach_optional_gdf(
        out,
        source_gdf=gdf,
        create_gdf=create_gdf,
        geometry_type=geometry_type,
    )


def simulate_sem_logit(
    W=None,
    gdf=None,
    n_side: int | None = None,
    lam: float = 0.35,
    beta: np.ndarray | None = None,
    rng: np.random.Generator | None = None,
    seed: int | None = None,
    contiguity: str = "queen",
    create_gdf: bool = False,
    geometry_type: str = "polygon",
    **kwargs,
) -> dict:
    """Simulate SEM-logit binary outcome data.

    DGP
    ---
    ``eta = X beta + (I - lam W)^{-1} nu``, ``nu ~ N(0, I)``,
    ``y ~ Bernoulli(logit^{-1}(eta))``.

    Parameters
    ----------
    W, gdf
        Spatial unit structure. If ``W`` is provided it takes precedence;
        otherwise ``gdf`` is used with ``contiguity``.
    n_side : int, optional
        Side length of the square rook grid used when neither ``W`` nor
        ``gdf`` is supplied (``n_side**2`` units).
    lam : float, default 0.35
        Spatial error parameter.
    beta : np.ndarray, optional
        Coefficients including intercept. Defaults to ``[0.3, 1.0]``.
    rng, seed
        Random state controls.
    contiguity : str, default "queen"
        GeoDataFrame neighbor rule when ``W`` is omitted.
    create_gdf : bool, default=False
        If ``True`` (or a source ``gdf`` is supplied), return a
        GeoDataFrame carrying ``y`` and ``X_*`` columns instead of the
        dict, matching the other cross-sectional simulators.
    geometry_type : {"point", "polygon"}, default="polygon"
        Geometry to generate when ``create_gdf=True`` and no source
        ``gdf`` is provided.

    Returns
    -------
    dict or geopandas.GeoDataFrame
        Dict with keys ``y``, ``X``, ``W_sparse``, ``W_graph``,
        ``eta_true``, ``params_true``.  When ``create_gdf=True`` (or a
        source ``gdf`` is supplied) a GeoDataFrame with ``y`` and
        ``X_*`` columns is returned instead.
    """
    rng = ensure_rng(rng, seed)
    W_sparse, Wg = resolve_weights(W=W, gdf=gdf, n_side=n_side, contiguity=contiguity)
    n_obs = W_sparse.shape[0]

    if beta is None:
        beta = np.array([0.3, 1.0], dtype=float)
    beta = np.asarray(beta, dtype=float)

    X = make_design_matrix(rng, n_obs, k=max(len(beta) - 1, 0), add_intercept=True)

    # Generate latent field: eta = X beta + (I - lam W)^{-1} nu
    nu = rng.standard_normal(n_obs)
    Xbeta = X @ beta
    eta = Xbeta + spatial_filter_factor(W_sparse, lam)(nu)

    # Binary response: y ~ Bernoulli(logit^{-1}(eta))
    probs = 1.0 / (1.0 + np.exp(-eta))
    y = rng.binomial(1, probs).astype(float)

    out = {
        "y": y,
        "X": X,
        "W_sparse": W_sparse,
        "W_graph": Wg,
        "eta_true": eta,
        "params_true": {
            "lam": lam,
            "beta": beta,
        },
    }
    return _attach_optional_gdf(
        out,
        source_gdf=gdf,
        create_gdf=create_gdf,
        geometry_type=geometry_type,
    )
