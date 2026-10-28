"""Zero-inflated spatial DGP functions."""

from __future__ import annotations

import numpy as np

from ._design import aspatial, count, simulator
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


@simulator(count)
def simulate_sar_zinb(
    n_side: int | None = None,
    W=None,
    gdf=None,
    rho: float = 0.5,
    rho_zero: float = 0.3,
    beta: np.ndarray | None = None,
    beta_zero: np.ndarray | None = None,
    alpha: float = 2.0,
    X_zero: np.ndarray | None = None,
    X: np.ndarray | None = None,
    W_zero=None,
    rng: np.random.Generator | None = None,
    seed: int | None = None,
    contiguity: str = "queen",
    target_pi: float | None = None,
    err_hetero: bool = False,
    create_gdf: bool = False,
    geometry_type: str = "polygon",
    latent: bool = False,
) -> dict:
    r"""Simulate data from a zero-inflated SAR-NB DGP.

    Two-part model:

    **Zero part** (unit activity — reduced-form SAR logit):

    .. math::

        d_i \sim \text{Bernoulli}(\text{logit}^{-1}(\eta_{z,i})),
        \quad \eta_z = (I - \rho_z W_z)^{-1} X_z\beta_z,

    the reduced-form zero part :class:`~neighbayes.models.SARZINB` fits.  (Until
    2026-10-03 a latent ``ν ~ N(0, I)`` was added inside the filter, a random
    effect the model does not have, which attenuated the ``β_z`` estimates.)

    **Count part** (volume — reduced-form SAR-NB):

    .. math::

        y_i \mid d_i = 1 \sim \text{NegBin}(\exp(\eta^{\text{cnt}}_i), \alpha),
        \quad \eta^{\text{cnt}} = (I - \rho W_{\text{cnt}})^{-1} X\beta

    **Observation:**

    .. math::

        y_i = 0 \text{ if } d_i = 0, \quad
        y_i \sim \text{NegBin}(\cdot) \text{ if } d_i = 1

    With ``latent=True`` it draws the latent model ``SARZINB(latent=True)``
    fits, each part's own disturbance propagated: :math:`d_i = \mathbf 1(u_i >
    0)`, :math:`u = (I - \rho_z W_z)^{-1}(X_z\beta_z + \varepsilon_z)` with
    :math:`\varepsilon_z` logistic, and :math:`y_i \mid d_i = 1 \sim
    \mathrm{Poisson}(e^{\eta_i})`, :math:`\eta = (I - \rho W)^{-1}(X\beta +
    \varepsilon)` with :math:`\varepsilon_i = \log\gamma_i`,
    :math:`\gamma_i \sim \mathrm{Gamma}(\alpha, \alpha)`; ``eta_zero`` and
    ``eta_cnt`` are then the drawn ``u`` and ``eta``.

    Parameters
    ----------
    n_side : int, optional
        Side length of the square rook grid used when neither ``W`` nor
        ``gdf`` is supplied (``n_side**2`` observations).
    W : Graph or scipy.sparse matrix, optional
        Spatial weights for the count part. Also used for the zero part
        when ``W_zero`` is not provided.
    gdf : GeoDataFrame, optional
        Geodataframe used to construct weights.
    rho : float, default 0.5
        Spatial autoregressive parameter for the count part.
    rho_zero : float, default 0.3
        Spatial autoregressive parameter for the zero part.
    beta : ndarray, optional
        Count-part coefficients (including intercept). Defaults to
        ``[1.0, 0.6]``.
    beta_zero : ndarray, optional
        Zero-part coefficients (including intercept). Defaults to
        ``[0.3, 1.0]``.
    alpha : float, default 2.0
        NB2 dispersion parameter. Must be strictly positive.
    X_zero : ndarray, optional
        Zero-part covariate matrix of shape ``(nobs, p)``. If not provided,
        a random design matrix is generated from ``beta_zero``.
    X : ndarray, optional
        Count covariate matrix of shape ``(nobs, k)``. If not provided,
        a random design matrix is generated from ``beta``.
    W_zero : Graph or scipy.sparse matrix, optional
        Spatial weights for the zero part. If ``None``, uses ``W`` (the same
        weights in both parts).
    rng : numpy.random.Generator, optional
        Random number generator.
    seed : int, optional
        Random seed (used only if rng is None).
    contiguity : str, default "queen"
        Neighbor rule used when W is built from ``gdf``.
    target_pi : float, optional
        If given, an intercept shift is solved so that the marginal
        activation probability ``mean(pi) == target_pi``.
        The shift is added to ``beta_zero[0]`` (inside the filter, so the
        reported ``beta_zero`` generates ``eta_zero``).
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
        Keys: ``y``, ``d``, ``X``, ``X_zero``, ``eta_cnt``, ``eta_zero``,
        ``W_sparse``, ``W_graph``, ``W_zero_sparse``, ``W_zero_graph``,
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
    if W_zero is not None:
        W_zero_s, W_zero_g = resolve_weights(
            W=W_zero, gdf=gdf, n_side=n_side, contiguity=contiguity
        )
    else:
        W_zero_s, W_zero_g = Ws, Wg

    if beta is None:
        beta = np.array([1.0, 0.6], dtype=float)
    beta = np.asarray(beta, dtype=float)

    if beta_zero is None:
        beta_zero = np.array([0.3, 1.0], dtype=float)
    beta_zero = np.asarray(beta_zero, dtype=float)

    # Generate design matrices if not provided
    if X is None:
        X = make_design_matrix(rng, nobs, k=max(len(beta) - 1, 0), add_intercept=True)
    else:
        X = np.asarray(X, dtype=np.float64)

    if X_zero is None:
        X_zero = make_design_matrix(
            rng, nobs, k=max(len(beta_zero) - 1, 0), add_intercept=True
        )
    else:
        X_zero = np.asarray(X_zero, dtype=np.float64)

    _check_rho_stability(rho, name="rho")
    _check_rho_stability(rho_zero, name="rho_zero")

    if latent:
        eta_sel, eta_cnt, beta_zero = _latent_parts(
            rng, Ws, W_zero_s, X, X_zero, beta, beta_zero, rho, rho_zero, alpha,
            target_pi,
        )  # fmt: skip
        d = (eta_sel > 0).astype(np.float64)
        mu = np.exp(np.clip(eta_cnt, -30.0, 30.0))
        y = np.where(d == 1, rng.poisson(mu), 0).astype(np.float64)
        return _zinb_output(
            y, d, X, X_zero, eta_cnt, eta_sel, Ws, Wg, W_zero_s, W_zero_g,
            rho, rho_zero, beta, beta_zero, alpha, gdf, create_gdf, geometry_type,
        )  # fmt: skip

    # --- Selection equation: SAR-logit ---
    # Reduced form, as the model: eta_sel = (I - lam * W_zero)^{-1} Z gamma.
    solve_sel = spatial_filter_factor(W_zero_s, rho_zero)
    eta_sel = solve_sel(X_zero @ beta_zero)

    # Apply target_pi shift if requested.  The shift moves gamma[0] inside the
    # filter, so eta_sel moves by c·(I − λW_zero)⁻¹Z[:, 0], not by c.
    if target_pi is not None:
        if not 0.0 < float(target_pi) < 1.0:
            raise ValueError(f"target_pi must lie in (0, 1); got {target_pi!r}")
        a0 = np.asarray(solve_sel(X_zero[:, 0]), dtype=float)

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
        beta_zero = beta_zero.copy()
        beta_zero[0] = beta_zero[0] + c

    # Binary unit activity
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
    return _zinb_output(
        y, d, X, X_zero, eta_cnt, eta_sel, Ws, Wg, W_zero_s, W_zero_g,
        rho, rho_zero, beta, beta_zero, alpha, gdf, create_gdf, geometry_type,
    )  # fmt: skip


def _zinb_output(
    y, d, X, X_zero, eta_cnt, eta_sel, Ws, Wg, W_zero_s, W_zero_g,
    rho, rho_zero, beta, beta_zero, alpha, gdf, create_gdf, geometry_type,
) -> dict:  # fmt: skip
    params_true = {
        "rho": rho,
        "rho_zero": rho_zero,
        "beta": beta,
        "beta_zero": beta_zero,
        "alpha": alpha,
    }

    out = {
        "y": y,
        "d": d,
        "X": X,
        "X_zero": X_zero,
        "eta_cnt": eta_cnt,
        "eta_zero": eta_sel,
        "W_sparse": Ws,
        "W_graph": Wg,
        "W_zero_sparse": W_zero_s,
        "W_zero_graph": W_zero_g,
        "params_true": params_true,
    }
    return _attach_optional_gdf(
        out,
        source_gdf=gdf,
        create_gdf=create_gdf,
        geometry_type=geometry_type,
    )


def _latent_parts(
    rng, Ws, W_zero_s, X, X_zero, beta, beta_zero, rho, rho_zero, alpha, target_pi
):
    """The latent two-part utilities and log-means, each disturbance propagated.

    ``u = (I - rho_zero W_zero)^{-1}(X_zero beta_zero + eps_z)``, ``eps_z``
    logistic, and ``eta = (I - rho W)^{-1}(X beta + eps)``, ``eps = log g``,
    ``g ~ Gamma(alpha, alpha)``.  ``target_pi`` shifts the zero part's intercept
    so the mean of ``P(u > 0)`` (over 256 disturbance draws) is the target.
    Returns ``(u, eta, beta_zero)``.
    """
    n = X.shape[0]
    solve_z = spatial_filter_factor(W_zero_s, rho_zero)
    beta_zero = np.array(beta_zero, dtype=float)
    if target_pi is not None:
        if not 0.0 < float(target_pi) < 1.0:
            raise ValueError(f"target_pi must lie in (0, 1); got {target_pi!r}")
        m = solve_z(X_zero @ beta_zero)
        a0 = np.asarray(solve_z(np.ascontiguousarray(X_zero[:, 0])), dtype=float)
        noise = np.asarray(solve_z(rng.logistic(size=(n, 256))))
        lo, hi = -50.0, 50.0
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            if np.mean(m[:, None] + mid * a0[:, None] + noise > 0) < target_pi:
                lo = mid
            else:
                hi = mid
        beta_zero[0] += 0.5 * (lo + hi)
    u = solve_z(X_zero @ beta_zero + rng.logistic(size=n))
    # log g by log Gamma(alpha + 1) + log(U) / alpha, which does not underflow.
    eps = np.log(rng.gamma(alpha + 1.0, 1.0 / alpha, size=n))
    eps += np.log(rng.random(n)) / alpha
    eta = spatial_filter_factor(Ws, rho)(X @ beta + eps)
    return u, eta, beta_zero


@aspatial(simulate_sar_zinb, {"rho": 0.0, "rho_zero": 0.0}, panel_only=True)
def simulate_zinb():
    """Simulate an aspatial ZINB panel: :func:`simulate_sar_zinb` with ``rho = rho_zero = 0``.

    Panels only, as for :class:`~neighbayes.models.ZINB`: pass ``T`` and ``N``.
    """
