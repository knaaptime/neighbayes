"""Dynamic panel DGP functions.

The :func:`simulate_panel_sdmu_fe` (Spatial Durbin, Unrestricted) generator
is the most general lag-on-y dynamic panel DGP in this module: it allows
free coefficients on ``W y_t``, ``y_{t-1}``, and ``W y_{t-1}``. Other
lag-on-y wrappers nest under it via parameter restrictions:

==========================================  ============  ==============  ==========
Wrapper                                     :math:`\\rho`     :math:`\\phi`        :math:`\\theta`
==========================================  ============  ==============  ==========
``simulate_panel_dlm_fe``                   0             free            0
``simulate_panel_slx_dynamic_fe``           0             free            0
``simulate_panel_sar_dynamic_fe``           free          free            0
``simulate_panel_sdmr_fe``                  free          free            :math:`-\\rho\\phi`
``simulate_panel_sdmu_fe``                  free          free            free
==========================================  ============  ==============  ==========

For the lag-on-error branch, :func:`simulate_panel_sdem_dynamic_fe` is the
core; :func:`simulate_panel_sem_dynamic_fe` calls it with the same
algebra (the two functions share the underlying DGP today).

Note: the existing implementations of ``simulate_panel_slx_dynamic_fe``
and ``simulate_panel_sdem_dynamic_fe`` do not add an explicit ``W X_t
theta`` term to the data equation; this preserves their pre-refactor
behavior. See the module-level discussion in the neighbayes docs for
the exact form.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from .panel_fe import _panel_finalize
from .utils import (
    _hetero_scale,
    _maybe_geodataframe,
    ensure_rng,
    make_design_matrix,
    resolve_weights,
    spatial_filter_factor,
)


def _simulate_panel_sdmu_fe_core(
    *,
    N: int,
    T: int,
    Ws: sp.csr_matrix,
    beta: np.ndarray,
    rho: float,
    phi: float,
    theta: float,
    sigma: float,
    sigma_alpha: float,
    err_hetero: bool,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Dynamic SDM-U panel FE kernel.

    ``y_t = (I - rho W)^-1 (phi y_{t-1} + theta W y_{t-1} + X_t beta + alpha + eps_t)``.

    Allows ``rho=0`` (skips spatial solve) and arbitrary ``phi`` / ``theta``
    so this single kernel reproduces SDMU, SDMR (``theta = -rho*phi``),
    dynamic SAR (``theta=0``), DLM and dynamic SLX (``rho=0, theta=0``).
    """
    use_solve = rho != 0.0
    solve = spatial_filter_factor(Ws, rho) if use_solve else None

    alpha = rng.normal(0.0, sigma_alpha, N)
    y_prev = rng.normal(scale=sigma, size=N)

    y_list, X_list = [], []
    for _ in range(T):
        Xt = make_design_matrix(rng, N, k=max(len(beta) - 1, 0), add_intercept=True)
        eps = (_hetero_scale(Xt, sigma) if err_hetero else sigma) * rng.standard_normal(
            N
        )
        rhs = phi * y_prev + Xt @ beta + alpha + eps
        if theta != 0.0:
            rhs = rhs + theta * (Ws @ y_prev)
        yt = solve(rhs) if use_solve else rhs
        y_list.append(yt)
        X_list.append(Xt)
        y_prev = yt
    return _panel_finalize(y_list, X_list, N, T)


def _simulate_panel_sdem_dynamic_fe_core(
    *,
    N: int,
    T: int,
    Ws: sp.csr_matrix,
    beta: np.ndarray,
    lam: float,
    phi: float,
    sigma: float,
    sigma_alpha: float,
    err_hetero: bool,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Dynamic lag-on-error panel FE kernel.

    ``y_t = phi y_{t-1} + X_t beta + alpha + u_t,  u_t = (I - lam W)^-1 eps_t``.

    Both :func:`simulate_panel_sem_dynamic_fe` and
    :func:`simulate_panel_sdem_dynamic_fe` route through this kernel.
    """
    use_solve = lam != 0.0
    solve = spatial_filter_factor(Ws, lam) if use_solve else None

    alpha = rng.normal(0.0, sigma_alpha, N)
    y_prev = rng.normal(scale=sigma, size=N)

    y_list, X_list = [], []
    for _ in range(T):
        Xt = make_design_matrix(rng, N, k=max(len(beta) - 1, 0), add_intercept=True)
        eps = (_hetero_scale(Xt, sigma) if err_hetero else sigma) * rng.standard_normal(
            N
        )
        u = solve(eps) if use_solve else eps
        yt = phi * y_prev + Xt @ beta + alpha + u
        y_list.append(yt)
        X_list.append(Xt)
        y_prev = yt
    return _panel_finalize(y_list, X_list, N, T)


def _coerce_beta(beta: np.ndarray | None) -> np.ndarray:
    if beta is None:
        beta = np.array([1.0, 2.0], dtype=float)
    return np.asarray(beta, dtype=float)


def _resolve_panel_weights(W, gdf, contiguity, N):
    Ws, Wg = resolve_weights(W=W, gdf=gdf, contiguity=contiguity)
    if Ws.shape[0] != N:
        raise ValueError("N must match W/gdf unit count.")
    return Ws, Wg


def simulate_panel_dlm_fe(
    N: int,
    T: int,
    phi: float = 0.4,
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
    """Simulate dynamic non-spatial panel FE data.

    DGP: ``y_t = phi y_{t-1} + X_t beta + alpha + eps_t``.

    Equivalent to :func:`simulate_panel_sdmu_fe` with ``rho=0`` and ``theta=0``.
    """
    rng = ensure_rng(rng, seed)
    Ws, Wg = _resolve_panel_weights(W, gdf, contiguity, N)
    beta = _coerce_beta(beta)
    y, X, idx = _simulate_panel_sdmu_fe_core(
        N=N,
        T=T,
        Ws=Ws,
        beta=beta,
        rho=0.0,
        phi=phi,
        theta=0.0,
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
            "phi": phi,
            "beta": beta,
            "sigma": sigma,
            "sigma_alpha": sigma_alpha,
        },
        create_gdf=create_gdf,
        gdf=gdf,
        geometry_type=geometry_type,
        wide=wide,
    )


def simulate_panel_sdmr_fe(
    N: int,
    T: int,
    rho: float = 0.3,
    phi: float = 0.4,
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
    """Simulate dynamic restricted SDM panel FE data.

    DGP:
    ``y_t = rho W y_t + phi y_{t-1} - rho phi W y_{t-1} + X_t beta + alpha + eps_t``.

    Equivalent to :func:`simulate_panel_sdmu_fe` with ``theta = -rho * phi``.
    """
    rng = ensure_rng(rng, seed)
    Ws, Wg = _resolve_panel_weights(W, gdf, contiguity, N)
    beta = _coerce_beta(beta)
    y, X, idx = _simulate_panel_sdmu_fe_core(
        N=N,
        T=T,
        Ws=Ws,
        beta=beta,
        rho=rho,
        phi=phi,
        theta=-rho * phi,
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
            "phi": phi,
            "beta": beta,
            "sigma": sigma,
            "sigma_alpha": sigma_alpha,
        },
        create_gdf=create_gdf,
        gdf=gdf,
        geometry_type=geometry_type,
        wide=wide,
    )


def simulate_panel_sdmu_fe(
    N: int,
    T: int,
    rho: float = 0.3,
    phi: float = 0.4,
    theta: float = -0.1,
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
    """Simulate dynamic unrestricted SDM panel FE data.

    DGP:
    ``y_t = rho W y_t + phi y_{t-1} + theta W y_{t-1} + X_t beta + alpha + eps_t``.

    This is the most general lag-on-y dynamic panel-FE DGP; the other
    dynamic generators in this module (``dlm_fe``, ``sar_dynamic_fe``,
    ``slx_dynamic_fe``, ``sdmr_fe``) are wrappers around this kernel.
    """
    rng = ensure_rng(rng, seed)
    Ws, Wg = _resolve_panel_weights(W, gdf, contiguity, N)
    beta = _coerce_beta(beta)
    y, X, idx = _simulate_panel_sdmu_fe_core(
        N=N,
        T=T,
        Ws=Ws,
        beta=beta,
        rho=rho,
        phi=phi,
        theta=theta,
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
            "phi": phi,
            "theta": theta,
            "beta": beta,
            "sigma": sigma,
            "sigma_alpha": sigma_alpha,
        },
        create_gdf=create_gdf,
        gdf=gdf,
        geometry_type=geometry_type,
        wide=wide,
    )


def simulate_panel_sar_dynamic_fe(
    N: int,
    T: int,
    rho: float = 0.3,
    phi: float = 0.4,
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
    """Simulate dynamic SAR panel FE data.

    DGP: ``y_t = rho W y_t + phi y_{t-1} + X_t beta + alpha + eps_t``.

    Equivalent to :func:`simulate_panel_sdmu_fe` with ``theta=0``.
    """
    rng = ensure_rng(rng, seed)
    Ws, Wg = _resolve_panel_weights(W, gdf, contiguity, N)
    beta = _coerce_beta(beta)
    y, X, idx = _simulate_panel_sdmu_fe_core(
        N=N,
        T=T,
        Ws=Ws,
        beta=beta,
        rho=rho,
        phi=phi,
        theta=0.0,
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
            "phi": phi,
            "beta": beta,
            "sigma": sigma,
            "sigma_alpha": sigma_alpha,
        },
        create_gdf=create_gdf,
        gdf=gdf,
        geometry_type=geometry_type,
        wide=wide,
    )


def simulate_panel_sem_dynamic_fe(
    N: int,
    T: int,
    lam: float = 0.3,
    phi: float = 0.4,
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
    """Simulate dynamic SEM panel FE data.

    DGP: ``y_t = phi y_{t-1} + X_t beta + alpha + u_t,  u_t = lam W u_t + eps_t``.

    Equivalent to :func:`simulate_panel_sdem_dynamic_fe` (the two share the
    same algebra in the current implementation).
    """
    rng = ensure_rng(rng, seed)
    Ws, Wg = _resolve_panel_weights(W, gdf, contiguity, N)
    beta = _coerce_beta(beta)
    y, X, idx = _simulate_panel_sdem_dynamic_fe_core(
        N=N,
        T=T,
        Ws=Ws,
        beta=beta,
        lam=lam,
        phi=phi,
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
            "phi": phi,
            "beta": beta,
            "sigma": sigma,
            "sigma_alpha": sigma_alpha,
        },
        create_gdf=create_gdf,
        gdf=gdf,
        geometry_type=geometry_type,
        wide=wide,
    )


def simulate_panel_sdem_dynamic_fe(
    N: int,
    T: int,
    lam: float = 0.3,
    phi: float = 0.4,
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
    """Simulate dynamic SDEM panel FE data.

    DGP: ``y_t = phi y_{t-1} + X_t beta + alpha + u_t,  u_t = lam W u_t + eps_t``.

    This is the lag-on-error dynamic core that :func:`simulate_panel_sem_dynamic_fe`
    also dispatches to.
    """
    rng = ensure_rng(rng, seed)
    Ws, Wg = _resolve_panel_weights(W, gdf, contiguity, N)
    beta = _coerce_beta(beta)
    y, X, idx = _simulate_panel_sdem_dynamic_fe_core(
        N=N,
        T=T,
        Ws=Ws,
        beta=beta,
        lam=lam,
        phi=phi,
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
            "phi": phi,
            "beta": beta,
            "sigma": sigma,
            "sigma_alpha": sigma_alpha,
        },
        create_gdf=create_gdf,
        gdf=gdf,
        geometry_type=geometry_type,
        wide=wide,
    )


def simulate_panel_slx_dynamic_fe(
    N: int,
    T: int,
    phi: float = 0.4,
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
    """Simulate dynamic SLX panel FE data.

    DGP: ``y_t = phi y_{t-1} + X_t beta + alpha + eps_t``.

    Equivalent to :func:`simulate_panel_sdmu_fe` with ``rho=0`` and ``theta=0``;
    in the current implementation no explicit ``W X_t`` term is added (this
    matches the pre-refactor behavior).
    """
    rng = ensure_rng(rng, seed)
    Ws, Wg = _resolve_panel_weights(W, gdf, contiguity, N)
    beta = _coerce_beta(beta)
    y, X, idx = _simulate_panel_sdmu_fe_core(
        N=N,
        T=T,
        Ws=Ws,
        beta=beta,
        rho=0.0,
        phi=phi,
        theta=0.0,
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
            "phi": phi,
            "beta": beta,
            "sigma": sigma,
            "sigma_alpha": sigma_alpha,
        },
        create_gdf=create_gdf,
        gdf=gdf,
        geometry_type=geometry_type,
        wide=wide,
    )
