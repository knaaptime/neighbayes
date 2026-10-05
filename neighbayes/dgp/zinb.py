"""Zero-inflated spatial DGP functions."""

from __future__ import annotations

import numpy as np

from .cross_sectional import (
    _attach_optional_gdf,
    _check_rho_stability,
)
from .utils import (
    ensure_rng,
    make_design_matrix,
    resolve_weights,
    spatial_filter_factor,
)


def simulate_sar_zinb(
    n_side: int | None = None,
    W=None,
    gdf=None,
    rho: float = 0.5,
    lam: float = 0.3,
    beta: np.ndarray | None = None,
    gamma: np.ndarray | None = None,
    alpha: float = 2.0,
    Z: np.ndarray | None = None,
    X: np.ndarray | None = None,
    W_sel=None,
    rng: np.random.Generator | None = None,
    seed: int | None = None,
    contiguity: str = "queen",
    target_pi: float | None = None,
    err_hetero: bool = False,
    create_gdf: bool = False,
    geometry_type: str = "polygon",
) -> dict:
    r"""Simulate data from a zero-inflated SAR-NB DGP.

    Two-equation model:

    **Selection** (corridor activity — SAR-logit):

    .. math::

        d_i \sim \text{Bernoulli}(\text{logit}^{-1}(\eta^{\text{sel}}_i)),
        \quad \eta^{\text{sel}} = (I - \lambda W_{\text{sel}})^{-1} Z\gamma,

    the reduced-form selection :class:`~neighbayes.models.SARZINB` fits.  (Until
    2026-10-03 a latent ``ν ~ N(0, I)`` was added inside the filter, a random
    effect the model does not have, which attenuated the γ estimates.)

    **Count** (flow volume — reduced-form SAR-NB):

    .. math::

        y_i \mid d_i = 1 \sim \text{NegBin}(\exp(\eta^{\text{cnt}}_i), \alpha),
        \quad \eta^{\text{cnt}} = (I - \rho W_{\text{cnt}})^{-1} X\beta

    **Observation:**

    .. math::

        y_i = 0 \text{ if } d_i = 0, \quad
        y_i \sim \text{NegBin}(\cdot) \text{ if } d_i = 1

    Parameters
    ----------
    n_side : int, optional
        Side length of the square rook grid used when neither ``W`` nor
        ``gdf`` is supplied (``n_side**2`` observations).
    W : Graph or scipy.sparse matrix, optional
        Spatial weights for the count equation. Also used for the
        selection equation when ``W_sel`` is not provided.
    gdf : GeoDataFrame, optional
        Geodataframe used to construct weights.
    rho : float, default 0.5
        Spatial autoregressive parameter for the count equation.
    lam : float, default 0.3
        Spatial autoregressive parameter for the selection equation.
    beta : ndarray, optional
        Count equation coefficients (including intercept). Defaults to
        ``[1.0, 0.6]``.
    gamma : ndarray, optional
        Selection equation coefficients (including intercept). Defaults
        to ``[0.3, 1.0]``.
    alpha : float, default 2.0
        NB2 dispersion parameter. Must be strictly positive.
    Z : ndarray, optional
        Selection covariate matrix of shape ``(nobs, p)``. If not
        provided, a random design matrix is generated from ``gamma``.
    X : ndarray, optional
        Count covariate matrix of shape ``(nobs, k)``. If not provided,
        a random design matrix is generated from ``beta``.
    W_sel : Graph or scipy.sparse matrix, optional
        Spatial weights for the selection equation. If ``None``, uses
        ``W`` (same weights for both equations).
    rng : numpy.random.Generator, optional
        Random number generator.
    seed : int, optional
        Random seed (used only if rng is None).
    contiguity : str, default "queen"
        Neighbor rule used when W is built from ``gdf``.
    target_pi : float, optional
        If given, an intercept shift is solved so that the marginal
        corridor activation probability ``mean(pi) == target_pi``.
        The shift is added to ``gamma[0]`` (inside the filter, so the
        reported ``gamma`` generates ``eta_sel``).
    err_hetero : bool, default False
        Not implemented for ZINB models. If True, a warning is issued
        and the parameter is ignored (homoskedastic errors are used).
    create_gdf : bool, default False
        Whether to attach a GeoDataFrame to the output.
    geometry_type : str, default "polygon"
        Type of geometry for the GeoDataFrame.

    Returns
    -------
    dict
        Keys: ``y``, ``d``, ``X``, ``Z``, ``eta_cnt``, ``eta_sel``,
        ``W_sparse``, ``W_graph``, ``W_sel_sparse``, ``W_sel_graph``,
        ``params_true``.
    """
    if alpha <= 0:
        raise ValueError("alpha must be strictly positive.")
    if err_hetero:
        import warnings

        warnings.warn(
            "err_hetero is not implemented for simulate_sar_zinb and is ignored.",
            stacklevel=2,
        )

    rng = ensure_rng(rng, seed)
    Ws, Wg = resolve_weights(W=W, gdf=gdf, n_side=n_side, contiguity=contiguity)
    nobs = Ws.shape[0]

    # Resolve selection weights
    if W_sel is not None:
        W_sel_s, W_sel_g = resolve_weights(
            W=W_sel, gdf=gdf, n_side=n_side, contiguity=contiguity
        )
    else:
        W_sel_s, W_sel_g = Ws, Wg

    if beta is None:
        beta = np.array([1.0, 0.6], dtype=float)
    beta = np.asarray(beta, dtype=float)

    if gamma is None:
        gamma = np.array([0.3, 1.0], dtype=float)
    gamma = np.asarray(gamma, dtype=float)

    # Generate design matrices if not provided
    if X is None:
        X = make_design_matrix(rng, nobs, k=max(len(beta) - 1, 0), add_intercept=True)
    else:
        X = np.asarray(X, dtype=np.float64)

    if Z is None:
        Z = make_design_matrix(rng, nobs, k=max(len(gamma) - 1, 0), add_intercept=True)
    else:
        Z = np.asarray(Z, dtype=np.float64)

    _check_rho_stability(rho, name="rho")
    _check_rho_stability(lam, name="lam")

    # --- Selection equation: SAR-logit ---
    # Reduced form, as the model: eta_sel = (I - lam * W_sel)^{-1} Z gamma.
    solve_sel = spatial_filter_factor(W_sel_s, lam)
    eta_sel = solve_sel(Z @ gamma)

    # Apply target_pi shift if requested.  The shift moves gamma[0] inside the
    # filter, so eta_sel moves by c·(I − λW_sel)⁻¹Z[:, 0], not by c.
    if target_pi is not None:
        if not 0.0 < float(target_pi) < 1.0:
            raise ValueError(f"target_pi must lie in (0, 1); got {target_pi!r}")
        a0 = np.asarray(solve_sel(Z[:, 0]), dtype=float)

        def _mean_pi(c: float) -> float:
            return float(np.mean(1.0 / (1.0 + np.exp(-(eta_sel + c * a0)))))

        lo, hi = -50.0, 50.0
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            if _mean_pi(mid) < target_pi:
                lo = mid
            else:
                hi = mid
        c = 0.5 * (lo + hi)
        eta_sel = eta_sel + c * a0
        gamma = gamma.copy()
        gamma[0] = gamma[0] + c

    # Binary corridor activity
    pi = 1.0 / (1.0 + np.exp(-eta_sel))
    d = rng.binomial(1, pi).astype(np.float64)

    # --- Count equation: reduced-form SAR-NB ---
    # eta_cnt = (I - rho * W)^{-1} X @ beta
    eta_cnt = spatial_filter_factor(Ws, rho)(X @ beta)

    # --- ZINB observation ---
    mu = np.exp(np.clip(eta_cnt, -30.0, 30.0))
    p_nb = alpha / (alpha + mu)
    y_nb = rng.negative_binomial(alpha, p_nb).astype(np.float64)

    # Combine: y = 0 where d=0, y = NB draw where d=1
    y = np.where(d == 1, y_nb, 0.0)

    params_true = {
        "rho": rho,
        "lam": lam,
        "beta": beta,
        "gamma": gamma,
        "alpha": alpha,
    }

    out = {
        "y": y,
        "d": d,
        "X": X,
        "Z": Z,
        "eta_cnt": eta_cnt,
        "eta_sel": eta_sel,
        "W_sparse": Ws,
        "W_graph": Wg,
        "W_sel_sparse": W_sel_s,
        "W_sel_graph": W_sel_g,
        "params_true": params_true,
    }
    return _attach_optional_gdf(
        out,
        source_gdf=gdf,
        create_gdf=create_gdf,
        geometry_type=geometry_type,
    )
