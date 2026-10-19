"""Panel helpers: the effects codes, the within transformation, dense ``W``."""

from __future__ import annotations

from typing import Union

import numpy as np
import scipy.sparse as sp
from libpysal.graph import Graph

# ---------------------------------------------------------------------------
# Effects specification helpers
# ---------------------------------------------------------------------------

_EFFECTS_MAP: dict[str, int] = {
    "pooled": 0,
    "unit": 1,
    "time": 2,
    "two_way": 3,
}

_EFFECTS_NAMES: dict[int, str] = {v: k for k, v in _EFFECTS_MAP.items()}


def _resolve_effects(effects: Union[str, int]) -> int:
    """Map ``effects`` argument to an integer FE mode (0–3).

    Accepts integers (0–3) or strings ("pooled", "unit", "time", "two_way").
    """
    if isinstance(effects, str):
        key = effects.strip().lower()
        if key not in _EFFECTS_MAP:
            valid = ", ".join(sorted(_EFFECTS_MAP))
            raise ValueError(
                f"effects={effects!r} is not recognized. "
                f"Valid strings: {valid}; valid ints: 0–3."
            )
        return _EFFECTS_MAP[key]
    if isinstance(effects, int) and 0 <= effects <= 3:
        return effects
    raise ValueError(
        f"effects={effects!r} is not recognized. "
        "Use an int 0–3 or one of: 'pooled', 'unit', 'time', 'two_way'."
    )


def _demean_panel(y: np.ndarray, X: np.ndarray, N: int, T: int, effects: int):
    """Apply panel demeaning transformation.

    Implements the within-transformation for two-way fixed-effects panel
    models prior to the spatial filter.  In the SAR-FE setting we model

    .. math::

        y_{it} = \\rho \\sum_j W_{ij} y_{jt} + X_{it}\\beta + \\mu_i
                 + \\alpha_t + \\varepsilon_{it},

    and concentrate out the fixed effects by demeaning *both* sides of
    the equation before the spatial lag is applied.  Because :math:`W`
    operates only across units (within a period), the within-period
    demeaning commutes with :math:`W` (i.e. ``W (M_T y) = M_T (W y)``)
    so the order of "demean then apply :math:`W`" or "apply :math:`W`
    then demean" yields the same likelihood — a fact exploited in
    Lee & Yu (2010) and Elhorst (2014, ch. 3).  This is why
    :class:`~neighbayes.models._base.SpatialModel` builds ``Wy`` from the
    *demeaned* ``y`` returned here.  Demeaning costs degrees of freedom,
    so the likelihood of the demeaned data follows Lee & Yu's (2010)
    transformation approach: a unit effect leaves :math:`N(T-1)`
    effective observations and a Jacobian of
    :math:`(T-1)\\,\\log|I_N - \\rho W|`; a time effect with a
    row-standardized :math:`W` subtracts :math:`\\log(1-\\rho)` per
    retained period (see ``SpatialModel._jacobian_T``).

    References
    ----------
    Lee, L.-F. & Yu, J. (2010). Estimation of spatial autoregressive
    panel data models with fixed effects.  *Journal of Econometrics*,
    154(2), 165–185.

    Elhorst, J.P. (2014). *Spatial Econometrics: From Cross-Sectional
    Data to Spatial Panels*. Springer.

    Parameters
    ----------
    y : np.ndarray
        Stacked dependent variable of shape ``(N*T,)``.
    X : np.ndarray
        Stacked regressor matrix of shape ``(N*T, k)``.
    N : int
        Number of cross-sectional units.
    T : int
        Number of time periods.
    effects : int
        Fixed-effects mode: 0 pooled, 1 unit FE, 2 time FE, 3 two-way FE.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Demeaned ``(y, X)`` arrays in stacked format.
    """
    y2 = y.reshape(T, N)
    X3 = X.reshape(T, N, X.shape[1])

    if effects in (1, 3) and T < 2:
        raise ValueError(
            f"Unit fixed effects (effects={effects}) require T >= 2 to identify "
            "within-unit variation, but T=" + str(T) + " was supplied. "
            "Use effects=0 (pooled) or effects=2 (time FE) when T=1."
        )

    if effects == 0:
        y_with = y2
        X_with = X3
    elif effects == 1:
        y_with = y2 - y2.mean(axis=0, keepdims=True)
        X_with = X3 - X3.mean(axis=0, keepdims=True)
    elif effects == 2:
        y_with = y2 - y2.mean(axis=1, keepdims=True)
        X_with = X3 - X3.mean(axis=1, keepdims=True)
    elif effects == 3:
        y_i = y2.mean(axis=0, keepdims=True)
        y_t = y2.mean(axis=1, keepdims=True)
        y_g = y2.mean()
        y_with = y2 - y_i - y_t + y_g

        X_i = X3.mean(axis=0, keepdims=True)
        X_t = X3.mean(axis=1, keepdims=True)
        X_g = X3.mean(axis=(0, 1), keepdims=True)
        X_with = X3 - X_i - X_t + X_g
    else:
        raise ValueError("effects must be one of {0,1,2,3}")

    return y_with.reshape(-1), X_with.reshape(-1, X.shape[1])


def _absorbed_columns(X_within: np.ndarray, X_raw: np.ndarray) -> np.ndarray:
    """Columns the fixed effects absorb: zero after the within transform.

    ``X_within`` is ``X_raw`` after :func:`_demean_panel`.  A column fixed
    within the demeaning groups (an intercept, a time-invariant attribute under
    unit effects) is not identified alongside the effects.
    """
    scale = np.maximum(np.abs(X_raw).max(axis=0), 1.0)
    return np.abs(X_within).max(axis=0) <= 1e-10 * scale


def _as_dense_W(W: Union[Graph, sp.spmatrix, np.ndarray], N: int, T: int) -> np.ndarray:
    """Convert graph/sparse/array weights into dense panel-compatible matrix.

    Parameters
    ----------
    W : Graph, scipy.sparse, or np.ndarray
        Either an ``N x N`` cross-sectional matrix or an ``(N*T) x (N*T)``
        block-diagonal panel matrix. Public APIs accept only Graph or sparse
        inputs; ndarray is supported here for internal use.
    N : int
        Number of units.
    T : int
        Number of periods.

    Returns
    -------
    np.ndarray
        Dense panel weights matrix.
    """
    if isinstance(W, Graph):
        Wn = W.sparse.toarray().astype(float)
    elif sp.issparse(W):
        Wn = W.toarray().astype(float)
    else:
        Wn = np.asarray(W, dtype=float)

    if Wn.shape == (N, N):
        return np.kron(np.eye(T), Wn)
    if Wn.shape == (N * T, N * T):
        return Wn

    raise ValueError(
        f"W has shape {Wn.shape}; expected (N,N)=({N},{N}) or (N*T,N*T)=({N * T},{N * T})."
    )
