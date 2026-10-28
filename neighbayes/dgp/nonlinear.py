"""Nonlinear and censored DGP functions.

Simulates data from spatial Tobit, logit and probit models: the left-censored
SAR, SEM, and SDM Tobit; the SAR (latent or reduced-form), SEM and SDEM logit
and probit; and the SEM probit with random unit effects.
"""

from __future__ import annotations

import numpy as np
from scipy.special import erf, ndtr

from ._design import aspatial, binary, sem_probit, simulator, tobit
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


@simulator(tobit)
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


@simulator(tobit)
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


@simulator(tobit)
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


def _disturbances(link: str, rng, n: int) -> np.ndarray:
    """The link's own disturbances: standard logistic or standard normal."""
    if link == "logit":
        return rng.logistic(size=n)
    return rng.standard_normal(n)


def _bernoulli(link: str, rng, eta: np.ndarray) -> np.ndarray:
    """``y ~ Bernoulli(F(eta))``, ``F`` the link's inverse."""
    p = 1.0 / (1.0 + np.exp(-eta)) if link == "logit" else ndtr(eta)
    return rng.binomial(1, p).astype(float)


def _sar_binary(
    link, *, W, gdf, n_side, rho, beta, err_hetero, rng, seed, contiguity,
    create_gdf, geometry_type, latent,
) -> dict:  # fmt: skip
    """The SAR of a binary link: latent, or the reduced form."""
    if err_hetero and not latent:
        raise ValueError(
            "err_hetero scales the latent SAR's disturbances; the reduced form "
            "has none (pass latent=True)."
        )
    rng = ensure_rng(rng, seed)
    W_sparse, Wg = resolve_weights(W=W, gdf=gdf, n_side=n_side, contiguity=contiguity)
    n_obs = W_sparse.shape[0]

    if beta is None:
        beta = np.array([0.3, 1.0], dtype=float)
    beta = np.asarray(beta, dtype=float)

    X = make_design_matrix(rng, n_obs, k=max(len(beta) - 1, 0), add_intercept=True)

    solve = spatial_filter_factor(W_sparse, rho)
    if latent:
        # The latent SAR: the multiplier propagates the link's disturbance.
        eps = _disturbances(link, rng, n_obs)
        if err_hetero:
            eps = eps * _hetero_scale(X, 1.0)
        eta = solve(X @ beta + eps)
        y = (eta > 0).astype(float)
    else:
        # The reduced form: eta = (I - rho W)^{-1} X beta, y ~ Bernoulli.
        eta = solve(X @ beta)
        y = _bernoulli(link, rng, eta)

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


def _sem_binary(
    link, *, W, gdf, n_side, lam, beta, err_hetero, rng, seed, contiguity,
    create_gdf, geometry_type, theta=None, durbin=False,
) -> dict:  # fmt: skip
    """The latent SEM of a binary link, or with ``durbin`` the latent SDEM."""
    rng = ensure_rng(rng, seed)
    W_sparse, Wg = resolve_weights(W=W, gdf=gdf, n_side=n_side, contiguity=contiguity)
    n_obs = W_sparse.shape[0]
    if beta is None:
        beta = np.array([0.3, 1.0], dtype=float)
    beta = np.asarray(beta, dtype=float)
    if durbin:
        theta = (
            np.full(len(beta) - 1, 0.5) if theta is None else np.asarray(theta, float)
        )
    X = make_design_matrix(rng, n_obs, k=max(len(beta) - 1, 0), add_intercept=True)
    eps = _disturbances(link, rng, n_obs)
    if err_hetero:
        eps = eps * _hetero_scale(X, 1.0)
    params = {"lam": lam, "beta": beta}
    if durbin:
        WX = np.asarray(W_sparse @ X[:, 1:])
        eta = X @ beta + WX @ theta + spatial_filter_factor(W_sparse, lam)(eps)
        params["theta"] = theta
    else:
        eta = X @ beta + spatial_filter_factor(W_sparse, lam)(eps)
    out = {
        "y": (eta > 0).astype(float),
        "X": X,
        "W_sparse": W_sparse,
        "W_graph": Wg,
        "eta_true": eta,
        "params_true": params,
    }
    return _attach_optional_gdf(
        out,
        source_gdf=gdf,
        create_gdf=create_gdf,
        geometry_type=geometry_type,
    )


@simulator(binary)
def simulate_sar_logit(
    W=None,
    gdf=None,
    n_side: int | None = None,
    rho: float = 0.35,
    beta: np.ndarray | None = None,
    err_hetero: bool = False,
    rng: np.random.Generator | None = None,
    seed: int | None = None,
    contiguity: str = "queen",
    create_gdf: bool = False,
    geometry_type: str = "polygon",
    latent: bool = False,
) -> dict:
    """Simulate SAR-logit binary outcome data.

    DGP
    ---
    ``eta = (I - rho W)^{-1} X beta``, ``y ~ Bernoulli(logit^{-1}(eta))``: the
    reduced form :class:`~neighbayes.models.SARLogit` fits.  With
    ``latent=True``, the latent SAR ``SARLogit(latent=True)`` fits:
    ``y* = (I - rho W)^{-1} (X beta + eps)``, ``eps`` iid standard logistic,
    ``y = 1(y* > 0)``.

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
    err_hetero : bool, default False
        Scale each latent SAR disturbance by ``sqrt(1 + ||x_i||^2)``, a
        heteroskedastic departure from the model (``latent=True`` only).
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
    latent : bool, default=False
        Draw the latent SAR instead of the reduced form; ``eta_true`` is then
        the latent utility ``y*``.

    Returns
    -------
    dict or geopandas.GeoDataFrame
        Dict with keys ``y``, ``X``, ``W_sparse``, ``W_graph``,
        ``eta_true``, ``params_true``.  When ``create_gdf=True`` (or a
        source ``gdf`` is supplied) a GeoDataFrame with ``y`` and
        ``X_*`` columns is returned instead.
    """
    return _sar_binary(
        "logit", W=W, gdf=gdf, n_side=n_side, rho=rho, beta=beta,
        err_hetero=err_hetero, rng=rng,
        seed=seed, contiguity=contiguity, create_gdf=create_gdf,
        geometry_type=geometry_type, latent=latent,
    )  # fmt: skip


@simulator(binary)
def simulate_sem_logit(
    W=None,
    gdf=None,
    n_side: int | None = None,
    lam: float = 0.35,
    beta: np.ndarray | None = None,
    err_hetero: bool = False,
    rng: np.random.Generator | None = None,
    seed: int | None = None,
    contiguity: str = "queen",
    create_gdf: bool = False,
    geometry_type: str = "polygon",
) -> dict:
    """Simulate the latent SEM logit that :class:`~neighbayes.models.SEMLogit` fits.

    DGP
    ---
    ``y* = X beta + (I - lam W)^{-1} eps``, ``eps`` iid standard logistic,
    ``y = 1(y* > 0)``.

    Parameters
    ----------
    W, gdf, n_side, contiguity
        Spatial unit structure, as for :func:`simulate_sar_logit`.
    lam : float, default=0.35
        Spatial error parameter.
    beta : np.ndarray, optional
        Coefficients including intercept. Defaults to ``[0.3, 1.0]``.
    err_hetero : bool, default False
        Scale each logistic disturbance by ``sqrt(1 + ||x_i||^2)``, a
        heteroskedastic departure from the model.
    rng, seed
        Random state controls.
    create_gdf, geometry_type
        As for :func:`simulate_sar_logit`.

    Returns
    -------
    dict or geopandas.GeoDataFrame
        Dict with keys ``y``, ``X``, ``W_sparse``, ``W_graph``, ``eta_true``
        (the latent utility ``y*``) and ``params_true``.
    """
    return _sem_binary(
        "logit", W=W, gdf=gdf, n_side=n_side, lam=lam, beta=beta,
        err_hetero=err_hetero, rng=rng, seed=seed, contiguity=contiguity,
        create_gdf=create_gdf, geometry_type=geometry_type,
    )  # fmt: skip


@simulator(binary)
def simulate_sdem_logit(
    W=None,
    gdf=None,
    n_side: int | None = None,
    lam: float = 0.35,
    beta: np.ndarray | None = None,
    theta: np.ndarray | None = None,
    err_hetero: bool = False,
    rng: np.random.Generator | None = None,
    seed: int | None = None,
    contiguity: str = "queen",
    create_gdf: bool = False,
    geometry_type: str = "polygon",
) -> dict:
    """Simulate the latent SDEM logit that :class:`~neighbayes.models.SDEMLogit` fits.

    DGP
    ---
    ``y* = X beta + WX theta + (I - lam W)^{-1} eps``, ``eps`` iid standard
    logistic, ``y = 1(y* > 0)``; ``WX`` lags every covariate but the intercept.

    Parameters
    ----------
    W, gdf, n_side, contiguity, lam, err_hetero, rng, seed, create_gdf, geometry_type
        As for :func:`simulate_sem_logit`.
    beta : np.ndarray, optional
        Coefficients including intercept. Defaults to ``[0.3, 1.0]``.
    theta : np.ndarray, optional
        Coefficients of the lagged covariates. Defaults to ``0.5`` each.

    Returns
    -------
    dict or geopandas.GeoDataFrame
        Dict with keys ``y``, ``X``, ``W_sparse``, ``W_graph``, ``eta_true``
        (the latent utility ``y*``) and ``params_true``.
    """
    return _sem_binary(
        "logit", W=W, gdf=gdf, n_side=n_side, lam=lam, beta=beta,
        err_hetero=err_hetero, rng=rng, seed=seed, contiguity=contiguity,
        create_gdf=create_gdf, geometry_type=geometry_type, theta=theta,
        durbin=True,
    )  # fmt: skip


@simulator(binary)
def simulate_sar_probit(
    W=None,
    gdf=None,
    n_side: int | None = None,
    rho: float = 0.35,
    beta: np.ndarray | None = None,
    err_hetero: bool = False,
    rng: np.random.Generator | None = None,
    seed: int | None = None,
    contiguity: str = "queen",
    create_gdf: bool = False,
    geometry_type: str = "polygon",
    latent: bool = True,
) -> dict:
    """Simulate the SAR probit that :class:`~neighbayes.models.SARProbit` fits.

    DGP
    ---
    ``y* = (I - rho W)^{-1} (X beta + eps)``, ``eps`` iid standard normal,
    ``y = 1(y* > 0)``: the latent SAR, the unmarked SAR probit.  With
    ``latent=False``, the reduced form ``SARProbit(latent=False)`` fits:
    ``eta = (I - rho W)^{-1} X beta``, ``y ~ Bernoulli(Phi(eta))``.

    Parameters
    ----------
    W, gdf, n_side, contiguity, rho, beta, err_hetero, rng, seed, create_gdf, geometry_type
        As for :func:`simulate_sar_logit`.
    latent : bool, default=True
        ``False`` draws the reduced form; ``eta_true`` is the latent utility
        ``y*`` for the latent SAR and the linear predictor for the reduced
        form.

    Returns
    -------
    dict or geopandas.GeoDataFrame
        As for :func:`simulate_sar_logit`.
    """
    return _sar_binary(
        "probit", W=W, gdf=gdf, n_side=n_side, rho=rho, beta=beta,
        err_hetero=err_hetero, rng=rng,
        seed=seed, contiguity=contiguity, create_gdf=create_gdf,
        geometry_type=geometry_type, latent=latent,
    )  # fmt: skip


@simulator(sem_probit)
def simulate_sem_probit(
    W=None,
    gdf=None,
    n_side: int | None = None,
    lam: float = 0.35,
    beta: np.ndarray | None = None,
    err_hetero: bool = False,
    rng: np.random.Generator | None = None,
    seed: int | None = None,
    contiguity: str = "queen",
    create_gdf: bool = False,
    geometry_type: str = "polygon",
    obs_per_unit: int | None = None,
) -> dict:
    """Simulate the SEM probit that :class:`~neighbayes.models.SEMProbit` fits.

    DGP
    ---
    ``y* = X beta + (I - lam W)^{-1} eps``, ``eps`` iid standard normal,
    ``y = 1(y* > 0)``: the latent SEM.  With ``unit_effects="random"`` each
    unit is observed ``obs_per_unit`` times (default 25) and its effect
    follows the error process, ``c = (I - lam W)^{-1} unit_effect_sd v``,
    ``y ~ Bernoulli(Phi(X beta + c[unit]))``.

    Parameters
    ----------
    W, gdf, n_side, contiguity, lam, beta, rng, seed, create_gdf, geometry_type
        As for :func:`simulate_sem_logit`; with unit effects ``W`` is over the
        units and no GeoDataFrame is returned.
    err_hetero : bool, default False
        Scale each normal disturbance by ``sqrt(1 + ||x_i||^2)``, a
        heteroskedastic departure from the model; with unit effects, each
        unit's effect by the same function of its mean regressors.
    obs_per_unit : int, optional
        The observations of each unit (``unit_effects="random"`` only).

    Returns
    -------
    dict or geopandas.GeoDataFrame
        As for :func:`simulate_sem_logit`; with unit effects a dict that adds
        ``units``, each observation's unit, as ``SEMProbit`` takes it.
    """
    return _sem_binary(
        "probit", W=W, gdf=gdf, n_side=n_side, lam=lam, beta=beta,
        err_hetero=err_hetero, rng=rng, seed=seed, contiguity=contiguity,
        create_gdf=create_gdf, geometry_type=geometry_type,
    )  # fmt: skip


@simulator(binary)
def simulate_sdem_probit(
    W=None,
    gdf=None,
    n_side: int | None = None,
    lam: float = 0.35,
    beta: np.ndarray | None = None,
    theta: np.ndarray | None = None,
    err_hetero: bool = False,
    rng: np.random.Generator | None = None,
    seed: int | None = None,
    contiguity: str = "queen",
    create_gdf: bool = False,
    geometry_type: str = "polygon",
) -> dict:
    """Simulate the latent SDEM probit that :class:`~neighbayes.models.SDEMProbit` fits.

    DGP
    ---
    ``y* = X beta + WX theta + (I - lam W)^{-1} eps``, ``eps`` iid standard
    normal, ``y = 1(y* > 0)``; ``WX`` lags every covariate but the intercept.

    Parameters
    ----------
    W, gdf, n_side, contiguity, lam, beta, theta, err_hetero, rng, seed, create_gdf, geometry_type
        As for :func:`simulate_sdem_logit`.

    Returns
    -------
    dict or geopandas.GeoDataFrame
        As for :func:`simulate_sdem_logit`.
    """
    return _sem_binary(
        "probit", W=W, gdf=gdf, n_side=n_side, lam=lam, beta=beta,
        err_hetero=err_hetero, rng=rng, seed=seed, contiguity=contiguity,
        create_gdf=create_gdf, geometry_type=geometry_type, theta=theta,
        durbin=True,
    )  # fmt: skip


def _simulate_sem_probit_re(
    W=None,
    gdf=None,
    n_side: int | None = None,
    lam: float = 0.35,
    beta: np.ndarray | None = None,
    unit_effect_sd: float = 0.5,
    obs_per_unit: int = 25,
    err_hetero: bool = False,
    rng: np.random.Generator | None = None,
    seed: int | None = None,
    contiguity: str = "queen",
) -> dict:
    r"""The SEM probit with random unit effects (:func:`simulate_sem_probit`).

    DGP
    ---
    ``c = (I - lam W)^{-1} unit_effect_sd v``, ``v ~ N(0, I)`` (unit
    effects), ``eta = X beta + c[unit]``, ``y ~ Bernoulli(Phi(eta))``, with
    ``obs_per_unit`` observations of each unit.  ``err_hetero`` scales each
    unit's effect by :math:`\sqrt{1 + \|\bar{x}_j\|^2}`, :math:`\bar{x}_j`
    the unit's mean regressor vector.
    """
    rng = ensure_rng(rng, seed)
    W_sparse, Wg = resolve_weights(W=W, gdf=gdf, n_side=n_side, contiguity=contiguity)
    m = W_sparse.shape[0]

    if beta is None:
        beta = np.array([0.3, 1.0], dtype=float)
    beta = np.asarray(beta, dtype=float)

    nobs = int(m * obs_per_unit)
    unit_ids = np.repeat(np.arange(m), obs_per_unit)

    if err_hetero:
        # Generate X first so we can compute unit-level means for
        # heteroskedastic scaling.  This changes the RNG draw order
        # relative to the homoskedastic path.
        X = make_design_matrix(rng, nobs, k=max(len(beta) - 1, 0), add_intercept=True)
        X_unit_mean = X.reshape(m, obs_per_unit, -1).mean(axis=1)
        c_scale = _hetero_scale(X_unit_mean, unit_effect_sd)
    else:
        # Preserve the original RNG draw order: c is drawn before X.
        c_scale = unit_effect_sd

    c = spatial_filter_factor(W_sparse, lam)(c_scale * rng.standard_normal(m))

    if not err_hetero:
        X = make_design_matrix(rng, nobs, k=max(len(beta) - 1, 0), add_intercept=True)

    eta = X @ beta + c[unit_ids]
    p = 0.5 * (1.0 + erf(eta / np.sqrt(2.0)))
    y = rng.binomial(1, p).astype(float)

    return {
        "y": y,
        "X": X,
        "units": unit_ids,
        "W_sparse": W_sparse,
        "W_graph": Wg,
        "params_true": {
            "lam": lam,
            "beta": beta,
            "unit_effect_sd": unit_effect_sd,
            "obs_per_unit": obs_per_unit,
        },
    }


@aspatial(simulate_sar_logit, {"rho": 0.0})
def simulate_logit():
    """Simulate an aspatial logit: :func:`simulate_sar_logit` with ``rho = 0``."""


@aspatial(simulate_sar_probit, {"rho": 0.0})
def simulate_probit():
    """Simulate an aspatial probit: :func:`simulate_sar_probit` with ``rho = 0``."""
