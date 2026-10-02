"""Panel fixed-effects style DGP functions.

The :func:`simulate_panel_sdm_fe` and :func:`simulate_panel_sdem_fe`
generators are the most general lag-on-y and lag-on-error panel DGPs.
The simpler models in this module (``sar``, ``sem``, ``slx``, ``ols``)
are thin wrappers that call the corresponding core kernel with
parameter restrictions:

==================  ==========  ==========
Wrapper             :math:`\\rho`     :math:`\\gamma` (WX)
==================  ==========  ==========
``ols_fe``          0           0
``sar_fe``          free        0
``slx_fe``          0           free
``sdm_fe``          free        free
==================  ==========  ==========

The corresponding error-side restrictions hold for ``sem_fe`` /
``sdem_fe`` (with :math:`\\lambda` replacing :math:`\\rho`).
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from .utils import (
    _hetero_scale,
    _maybe_geodataframe,
    ensure_rng,
    make_design_matrix,
    panel_index,
    resolve_weights,
    spatial_filter_factor,
)


def _panel_finalize(
    y_list: list[np.ndarray], X_list: list[np.ndarray], N: int, T: int
) -> tuple[np.ndarray, np.ndarray, dict]:
    y = np.concatenate(y_list)
    X = np.vstack(X_list)
    idx_df = panel_index(N, T)
    return y, X, {"unit": idx_df["unit"].to_numpy(), "time": idx_df["time"].to_numpy()}


def _simulate_panel_sdm_fe_core(
    *,
    N: int,
    T: int,
    Ws: sp.csr_matrix,
    beta1: np.ndarray,
    beta2: np.ndarray,
    rho: float,
    sigma: float,
    sigma_alpha: float,
    err_hetero: bool,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Panel SDM-FE kernel.

    ``y_t = (I - rho W)^-1 (X_t beta1 + W X_t[:,1:] beta2 + alpha + eps_t)``.

    Allows ``rho=0`` (skips the spatial solve) and ``len(beta2)==0``
    (skips the WX term), nesting the SAR/SLX/OLS panel DGPs.
    """
    has_wx = len(beta2) > 0
    use_solve = rho != 0.0
    solve = spatial_filter_factor(Ws, rho) if use_solve else None

    alpha = rng.normal(0.0, sigma_alpha, N)
    y_list, X_list = [], []
    for _ in range(T):
        Xt = make_design_matrix(rng, N, k=max(len(beta1) - 1, 0), add_intercept=True)
        if has_wx:
            Wx = Ws @ Xt[:, 1:]
            if Wx.shape[1] != len(beta2):
                raise ValueError(
                    "len(beta2) must match number of non-intercept regressors."
                )
            wx_beta = Wx @ beta2
        else:
            wx_beta = 0.0
        eps = (_hetero_scale(Xt, sigma) if err_hetero else sigma) * rng.standard_normal(
            N
        )
        rhs = Xt @ beta1 + wx_beta + alpha + eps
        yt = solve(rhs) if use_solve else rhs
        y_list.append(yt)
        X_list.append(Xt)
    return _panel_finalize(y_list, X_list, N, T)


def _simulate_panel_sdem_fe_core(
    *,
    N: int,
    T: int,
    Ws: sp.csr_matrix,
    beta1: np.ndarray,
    beta2: np.ndarray,
    lam: float,
    sigma: float,
    sigma_alpha: float,
    err_hetero: bool,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Panel SDEM-FE kernel.

    ``u_t = (I - lam W)^-1 eps_t`` and
    ``y_t = X_t beta1 + W X_t[:,1:] beta2 + alpha + u_t``.

    Allows ``lam=0`` and ``len(beta2)==0`` to nest SEM/SLX/OLS panel DGPs.
    """
    has_wx = len(beta2) > 0
    use_solve = lam != 0.0
    solve = spatial_filter_factor(Ws, lam) if use_solve else None

    alpha = rng.normal(0.0, sigma_alpha, N)
    y_list, X_list = [], []
    for _ in range(T):
        Xt = make_design_matrix(rng, N, k=max(len(beta1) - 1, 0), add_intercept=True)
        if has_wx:
            Wx = Ws @ Xt[:, 1:]
            if Wx.shape[1] != len(beta2):
                raise ValueError(
                    "len(beta2) must match number of non-intercept regressors."
                )
            wx_beta = Wx @ beta2
        else:
            wx_beta = 0.0
        eps = (_hetero_scale(Xt, sigma) if err_hetero else sigma) * rng.standard_normal(
            N
        )
        u = solve(eps) if use_solve else eps
        yt = Xt @ beta1 + wx_beta + alpha + u
        y_list.append(yt)
        X_list.append(Xt)
    return _panel_finalize(y_list, X_list, N, T)


def simulate_panel_ols_fe(
    N: int,
    T: int,
    beta: np.ndarray | None = None,
    sigma: float = 1.0,
    sigma_alpha: float = 0.5,
    err_hetero: bool = False,
    rng: np.random.Generator | None = None,
    seed: int | None = None,
    W=None,
    gdf=None,
    contiguity: str = "queen",
    create_gdf: bool = False,
    geometry_type: str = "polygon",
    wide: bool = False,
) -> dict:
    """Simulate pooled data compatible with OLSPanelFE model assumptions.

    Equivalent to :func:`simulate_panel_sdm_fe` with ``rho=0`` and ``beta2=[]``;
    the spatial weights matrix is required for shape validation but does not
    appear in the data equation.

    Returns
    -------
    dict
        Keys: ``y``, ``X``, ``unit``, ``time``, ``W_sparse``, ``W_graph``,
        ``params_true`` with ``{beta, sigma, sigma_alpha}``.
    """
    rng = ensure_rng(rng, seed)
    Ws, Wg = resolve_weights(W=W, gdf=gdf, contiguity=contiguity)
    if Ws.shape[0] != N:
        raise ValueError("N must match W/gdf unit count.")

    if beta is None:
        beta = np.array([1.0, 2.0], dtype=float)
    beta = np.asarray(beta, dtype=float)

    y, X, idx = _simulate_panel_sdm_fe_core(
        N=N,
        T=T,
        Ws=Ws,
        beta1=beta,
        beta2=np.empty(0, dtype=float),
        rho=0.0,
        sigma=sigma,
        sigma_alpha=sigma_alpha,
        err_hetero=err_hetero,
        rng=rng,
    )
    return _maybe_geodataframe(
        y=y,
        X=X,
        idx=idx,
        N=N,
        T=T,
        Ws=Ws,
        Wg=Wg,
        params_true={"beta": beta, "sigma": sigma, "sigma_alpha": sigma_alpha},
        create_gdf=create_gdf,
        gdf=gdf,
        geometry_type=geometry_type,
        wide=wide,
    )


def simulate_panel_sar_fe(
    N: int,
    T: int,
    rho: float = 0.4,
    beta: np.ndarray | None = None,
    sigma: float = 1.0,
    sigma_alpha: float = 0.5,
    err_hetero: bool = False,
    rng: np.random.Generator | None = None,
    seed: int | None = None,
    W=None,
    gdf=None,
    n_side: int | None = None,
    contiguity: str = "queen",
    create_gdf: bool = False,
    geometry_type: str = "polygon",
    wide: bool = False,
) -> dict:
    """Simulate SAR panel data in time-first stacking order.

    DGP: ``y_t = (I - rho W)^-1 (X_t beta + alpha + eps_t)``.

    Equivalent to :func:`simulate_panel_sdm_fe` with ``beta2=[]``.
    """
    rng = ensure_rng(rng, seed)
    Ws, Wg = resolve_weights(W=W, gdf=gdf, n_side=n_side, contiguity=contiguity)
    if Ws.shape[0] != N:
        raise ValueError("N must match W/gdf unit count.")

    if beta is None:
        beta = np.array([1.0, 2.0], dtype=float)
    beta = np.asarray(beta, dtype=float)

    y, X, idx = _simulate_panel_sdm_fe_core(
        N=N,
        T=T,
        Ws=Ws,
        beta1=beta,
        beta2=np.empty(0, dtype=float),
        rho=rho,
        sigma=sigma,
        sigma_alpha=sigma_alpha,
        err_hetero=err_hetero,
        rng=rng,
    )
    return _maybe_geodataframe(
        y=y,
        X=X,
        idx=idx,
        N=N,
        T=T,
        Ws=Ws,
        Wg=Wg,
        params_true={
            "rho": rho,
            "beta": beta,
            "sigma": sigma,
            "sigma_alpha": sigma_alpha,
        },
        create_gdf=create_gdf,
        gdf=gdf,
        geometry_type=geometry_type,
        wide=wide,
    )


def simulate_panel_sem_fe(
    N: int,
    T: int,
    lam: float = 0.4,
    beta: np.ndarray | None = None,
    sigma: float = 1.0,
    sigma_alpha: float = 0.5,
    err_hetero: bool = False,
    rng: np.random.Generator | None = None,
    seed: int | None = None,
    W=None,
    gdf=None,
    n_side: int | None = None,
    contiguity: str = "queen",
    create_gdf: bool = False,
    geometry_type: str = "polygon",
    wide: bool = False,
) -> dict:
    """Simulate SEM panel data in time-first stacking order.

    DGP: ``u_t = (I - lam W)^-1 eps_t`` and ``y_t = X_t beta + alpha + u_t``.

    Equivalent to :func:`simulate_panel_sdem_fe` with ``beta2=[]``.
    """
    rng = ensure_rng(rng, seed)
    Ws, Wg = resolve_weights(W=W, gdf=gdf, n_side=n_side, contiguity=contiguity)
    if Ws.shape[0] != N:
        raise ValueError("N must match W/gdf unit count.")

    if beta is None:
        beta = np.array([1.0, 2.0], dtype=float)
    beta = np.asarray(beta, dtype=float)

    y, X, idx = _simulate_panel_sdem_fe_core(
        N=N,
        T=T,
        Ws=Ws,
        beta1=beta,
        beta2=np.empty(0, dtype=float),
        lam=lam,
        sigma=sigma,
        sigma_alpha=sigma_alpha,
        err_hetero=err_hetero,
        rng=rng,
    )
    return _maybe_geodataframe(
        y=y,
        X=X,
        idx=idx,
        N=N,
        T=T,
        Ws=Ws,
        Wg=Wg,
        params_true={
            "lam": lam,
            "beta": beta,
            "sigma": sigma,
            "sigma_alpha": sigma_alpha,
        },
        create_gdf=create_gdf,
        gdf=gdf,
        geometry_type=geometry_type,
        wide=wide,
    )


def simulate_panel_sdm_fe(
    N: int,
    T: int,
    rho: float = 0.4,
    beta1: np.ndarray | None = None,
    beta2: np.ndarray | None = None,
    sigma: float = 1.0,
    sigma_alpha: float = 0.5,
    err_hetero: bool = False,
    rng: np.random.Generator | None = None,
    seed: int | None = None,
    W=None,
    gdf=None,
    contiguity: str = "queen",
    create_gdf: bool = False,
    geometry_type: str = "polygon",
    wide: bool = False,
) -> dict:
    """Simulate SDM panel FE data.

    DGP: ``y_t = (I - rho W)^-1 (X_t beta1 + W X_t[:,1:] beta2 + alpha + eps_t)``.

    This is the general lag-on-y panel-FE DGP; SAR/SLX/OLS panel-FE are
    nested by setting ``beta2=[]`` and/or ``rho=0``.
    """
    rng = ensure_rng(rng, seed)
    Ws, Wg = resolve_weights(W=W, gdf=gdf, contiguity=contiguity)
    if Ws.shape[0] != N:
        raise ValueError("N must match W/gdf unit count.")

    if beta1 is None:
        beta1 = np.array([1.0, 2.0], dtype=float)
    beta1 = np.asarray(beta1, dtype=float)
    if beta2 is None:
        beta2 = np.array([0.8], dtype=float)
    beta2 = np.asarray(beta2, dtype=float)

    y, X, idx = _simulate_panel_sdm_fe_core(
        N=N,
        T=T,
        Ws=Ws,
        beta1=beta1,
        beta2=beta2,
        rho=rho,
        sigma=sigma,
        sigma_alpha=sigma_alpha,
        err_hetero=err_hetero,
        rng=rng,
    )
    return _maybe_geodataframe(
        y=y,
        X=X,
        idx=idx,
        N=N,
        T=T,
        Ws=Ws,
        Wg=Wg,
        params_true={
            "rho": rho,
            "beta1": beta1,
            "beta2": beta2,
            "sigma": sigma,
            "sigma_alpha": sigma_alpha,
        },
        create_gdf=create_gdf,
        gdf=gdf,
        geometry_type=geometry_type,
        wide=wide,
    )


def simulate_panel_sdem_fe(
    N: int,
    T: int,
    lam: float = 0.4,
    beta1: np.ndarray | None = None,
    beta2: np.ndarray | None = None,
    sigma: float = 1.0,
    sigma_alpha: float = 0.5,
    err_hetero: bool = False,
    rng: np.random.Generator | None = None,
    seed: int | None = None,
    W=None,
    gdf=None,
    contiguity: str = "queen",
    create_gdf: bool = False,
    geometry_type: str = "polygon",
    wide: bool = False,
) -> dict:
    """Simulate SDEM panel FE data.

    DGP: ``u_t = (I - lam W)^-1 eps_t`` and
    ``y_t = X_t beta1 + W X_t[:,1:] beta2 + alpha + u_t``.

    This is the general lag-on-error panel-FE DGP; SEM/SLX/OLS panel-FE
    are nested by setting ``beta2=[]`` and/or ``lam=0``.
    """
    rng = ensure_rng(rng, seed)
    Ws, Wg = resolve_weights(W=W, gdf=gdf, contiguity=contiguity)
    if Ws.shape[0] != N:
        raise ValueError("N must match W/gdf unit count.")

    if beta1 is None:
        beta1 = np.array([1.0, 2.0], dtype=float)
    beta1 = np.asarray(beta1, dtype=float)
    if beta2 is None:
        beta2 = np.array([0.8], dtype=float)
    beta2 = np.asarray(beta2, dtype=float)

    y, X, idx = _simulate_panel_sdem_fe_core(
        N=N,
        T=T,
        Ws=Ws,
        beta1=beta1,
        beta2=beta2,
        lam=lam,
        sigma=sigma,
        sigma_alpha=sigma_alpha,
        err_hetero=err_hetero,
        rng=rng,
    )
    return _maybe_geodataframe(
        y=y,
        X=X,
        idx=idx,
        N=N,
        T=T,
        Ws=Ws,
        Wg=Wg,
        params_true={
            "lam": lam,
            "beta1": beta1,
            "beta2": beta2,
            "sigma": sigma,
            "sigma_alpha": sigma_alpha,
        },
        create_gdf=create_gdf,
        gdf=gdf,
        geometry_type=geometry_type,
        wide=wide,
    )


def simulate_panel_slx_fe(
    N: int,
    T: int,
    beta1: np.ndarray | None = None,
    beta2: np.ndarray | None = None,
    sigma: float = 1.0,
    sigma_alpha: float = 0.5,
    err_hetero: bool = False,
    rng: np.random.Generator | None = None,
    seed: int | None = None,
    W=None,
    gdf=None,
    contiguity: str = "queen",
    create_gdf: bool = False,
    geometry_type: str = "polygon",
    wide: bool = False,
) -> dict:
    """Simulate SLX panel FE data.

    DGP: ``y_t = X_t beta1 + W X_t[:,1:] beta2 + alpha + eps_t``.

    Equivalent to :func:`simulate_panel_sdm_fe` with ``rho=0``.
    """
    rng = ensure_rng(rng, seed)
    Ws, Wg = resolve_weights(W=W, gdf=gdf, contiguity=contiguity)
    if Ws.shape[0] != N:
        raise ValueError("N must match W/gdf unit count.")

    if beta1 is None:
        beta1 = np.array([1.0, 2.0], dtype=float)
    beta1 = np.asarray(beta1, dtype=float)
    if beta2 is None:
        beta2 = np.array([0.8], dtype=float)
    beta2 = np.asarray(beta2, dtype=float)

    y, X, idx = _simulate_panel_sdm_fe_core(
        N=N,
        T=T,
        Ws=Ws,
        beta1=beta1,
        beta2=beta2,
        rho=0.0,
        sigma=sigma,
        sigma_alpha=sigma_alpha,
        err_hetero=err_hetero,
        rng=rng,
    )
    return _maybe_geodataframe(
        y=y,
        X=X,
        idx=idx,
        N=N,
        T=T,
        Ws=Ws,
        Wg=Wg,
        params_true={
            "beta1": beta1,
            "beta2": beta2,
            "sigma": sigma,
            "sigma_alpha": sigma_alpha,
        },
        create_gdf=create_gdf,
        gdf=gdf,
        geometry_type=geometry_type,
        wide=wide,
    )
