r"""Bayesian LM diagnostic tests — flow variants.

The flow tests are the score tests of the spatial-lag flow model

.. math::
    y = \rho_d W_d y + \rho_o W_o y + \rho_w W_w y + X\beta + \varepsilon,
    \qquad \varepsilon \sim N(0, \sigma^2 I),

at a fitted :class:`~neighbayes.models.OLSFlow`, cross-section or panel, on the
construction of :mod:`._gaussian`: the score of the tested directions at every
posterior draw, Neyman-orthogonal to :math:`(\beta, \sigma^2)`, the efficient
information at the posterior mean, and the statistic at the posterior-mean
parameters.  At :math:`\rho = 0` the scores are

.. math::
    g_{\rho_k} = -\mathrm{tr}\,W_k + (W_k y)^\top e/\sigma^2, \qquad
    J_{\rho_i\rho_j} = \mathrm{tr}(W_i^\top W_j + W_iW_j)
        + (W_iX\beta)^\top W_jX\beta/\sigma^2,

and with :math:`W_d = I \otimes W`, :math:`W_o = W \otimes I` and
:math:`W_w = W \otimes W` every trace factors, :math:`\mathrm{tr}(A \otimes B) =
\mathrm{tr}A\,\mathrm{tr}B`, so nothing of the size of the :math:`n^2` pairs is
formed.  A robust test removes the other two directions (Bera and Yoon 1993).
A panel uses the Lee and Yu (2010) transformed likelihood its demeaned data
give: traces are counted by the Jacobian multiplier, and time effects remove the
direction :math:`\mathbf 1`, which every :math:`W_k` of a row-standardized
:math:`W` maps to itself.

All public test functions return :class:`BayesianLMTestResult`.
"""

import weakref

import numpy as np
import scipy.sparse as sp

from neighbayes.diagnostics.lmtests._gaussian import efficient_lm
from neighbayes.diagnostics.lmtests.core import (
    BayesianLMTestResult,
    _finalize_lm,
    _get_posterior_draws,
)

#: The flow directions, in the order of the scores.
_DIRECTIONS = ("d", "o", "w")
#: Each direction's Kronecker factors, (origin, destination): I or W.
_FACTORS = {"d": ("I", "W"), "o": ("W", "I"), "w": ("W", "W")}

_CACHE: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()


def _flow_traces(W, time_fe: bool) -> tuple[np.ndarray, np.ndarray]:
    r"""Per-period ``tr(J W_k)`` and ``tr(J W_i' J W_j) + tr(J W_i J W_j)``.

    Without time effects :math:`J = I`; with them :math:`J` removes the mean
    across pairs, and :math:`\mathrm{tr}(JAJB) = \mathrm{tr}(AB) - (\mathbf
    1^\top AB\mathbf 1 + \mathbf 1^\top BA\mathbf 1)/N + (\mathbf 1^\top
    A\mathbf 1)(\mathbf 1^\top B\mathbf 1)/N^2`.
    """
    from neighbayes.graph import flow_lags

    W = sp.csr_matrix(W)
    n = W.shape[0]
    t1 = float(W.diagonal().sum())
    t2 = float(W.multiply(W.T).sum())
    t3 = float(W.multiply(W).sum())
    prod = {("I", "I"): n, ("I", "W"): t1, ("W", "I"): t1, ("W", "W"): t2}
    tprod = {("I", "I"): n, ("I", "W"): t1, ("W", "I"): t1, ("W", "W"): t3}
    single = {"I": n, "W": t1}
    tr = np.array(
        [single[_FACTORS[k][0]] * single[_FACTORS[k][1]] for k in _DIRECTIONS]
    )
    TS = np.empty((3, 3))
    TT = np.empty((3, 3))
    for i, a in enumerate(_DIRECTIONS):
        for j, b in enumerate(_DIRECTIONS):
            (ao, ad), (bo, bd) = _FACTORS[a], _FACTORS[b]
            TS[i, j] = prod[ao, bo] * prod[ad, bd]
            TT[i, j] = tprod[ao, bo] * tprod[ad, bd]
    if time_fe:
        N = n * n
        one = np.ones(N)
        r1 = np.column_stack(flow_lags(W, one))  # W_k 1
        c1 = np.column_stack(flow_lags(W.T.tocsr(), one))  # W_k' 1
        s = r1.sum(axis=0)
        tr = tr - s / N
        # tr(J W_i J W_j): a = W_i, b = W_j;  tr(J W_i' J W_j): a = W_i'.
        TS = TS - (c1.T @ r1 + r1.T @ c1) / N + np.outer(s, s) / N**2
        TT = TT - (r1.T @ r1 + c1.T @ c1) / N + np.outer(s, s) / N**2
    return tr, TT + TS


def _flow_scores(model):
    """Scores (a last row at the posterior mean), information and free indices.

    The parameters are ``beta``, then ``rho_d``, ``rho_o``, ``rho_w`` and
    ``sigma2``; the null estimates ``beta`` and ``sigma2``.
    """
    idata = model.inference_data
    cached = _CACHE.get(model)
    if cached is not None and cached[0] is idata:
        return cached[1]
    from neighbayes.graph import flow_lags

    post = idata.posterior
    spatial = [v for v in post if str(v).startswith(("rho", "lam"))]
    if spatial or "sigma" not in post:
        raise ValueError(
            "The flow LM tests take a Gaussian aspatial null (OLSFlow); "
            f"{type(model).__name__} "
            + (
                f"estimates {', '.join(map(str, spatial))}."
                if spatial
                else "is not Gaussian."
            )
        )
    y = np.asarray(model._y, dtype=np.float64)
    X = np.asarray(model._X, dtype=np.float64)
    W = sp.csr_matrix(model._W_sparse, dtype=np.float64)
    T = int(getattr(model, "_T", 1))
    m = float(model._jacobian_T)
    # Time effects drop log(1 - rho) from the Jacobian only for a row-standardized
    # W, where tr(J W_k) = tr(W_k) - 1 accounts for it.
    time_fe = float(getattr(model, "_flow_jacobian_shift", 0.0)) != 0.0
    n_eff = float(model._n_effective)
    N = y.size // T

    def lag(v):
        out = np.stack(flow_lags(W, v, T=T))  # (3, ...) like v
        if time_fe:
            r = out.reshape(3, T, N, -1)
            out = (r - r.mean(axis=2, keepdims=True)).reshape(out.shape)
        return out

    beta = _get_posterior_draws(idata, "beta")
    if beta.shape[1] != X.shape[1]:
        raise NotImplementedError(
            f"The flow LM tests need beta over the design; {type(model).__name__}'s "
            f"posterior has {beta.shape[1]} coefficients for {X.shape[1]} columns."
        )
    sigma2 = _get_posterior_draws(idata, "sigma").reshape(-1) ** 2
    beta = np.vstack([beta, beta.mean(axis=0)])
    sigma2 = np.r_[sigma2, sigma2.mean()]
    tr, T_blk = _flow_traces(W, time_fe)

    e = y[None, :] - beta @ X.T
    Wy = lag(y)  # (3, n)
    g = np.column_stack(
        [
            e @ X / sigma2[:, None],
            -m * tr[None, :] + (e @ Wy.T) / sigma2[:, None],
            -n_eff / (2.0 * sigma2) + np.sum(e * e, axis=1) / (2.0 * sigma2**2),
        ]
    )
    k = X.shape[1]
    s2 = sigma2[-1]
    WXb = lag(X @ beta[-1])  # (3, n)
    J = np.zeros((k + 4, k + 4))
    J[:k, :k] = X.T @ X / s2
    J[:k, k : k + 3] = X.T @ WXb.T / s2
    J[k : k + 3, :k] = J[:k, k : k + 3].T
    J[k : k + 3, k : k + 3] = m * T_blk + WXb @ WXb.T / s2
    J[k : k + 3, k + 3] = J[k + 3, k : k + 3] = m * tr / s2
    J[k + 3, k + 3] = n_eff / (2.0 * s2**2)
    out = (g, J, list(range(k)) + [k + 3], k)
    _CACHE[model] = (idata, out)
    return out


def _flow_test(
    model, tested, robust=(), *, test_type: str, details=None
) -> BayesianLMTestResult:
    g, J, free, k = _flow_scores(model)
    T = [k + _DIRECTIONS.index(d) for d in tested]
    R = [k + _DIRECTIONS.index(d) for d in robust]
    LM, statistic = efficient_lm(g, J, free, T, R, test_name=test_type)
    return _finalize_lm(
        LM, test_type=test_type, df=len(T), details=details, statistic=statistic
    )


def bayesian_lm_flow_destination_test(model) -> BayesianLMTestResult:
    r"""LM test for an omitted destination lag at an OLSFlow fit (H₀: ρ_d = 0).

    The score of :math:`\rho_d`, orthogonal to :math:`(\beta, \sigma^2)`
    (:cite:t:`lesage2008SpatialEconometric`; :cite:t:`dogan2021BayesianRobust`).

    Returns
    -------
    BayesianLMTestResult
        Per-draw LM samples, summary statistics and ``df = 1``.
    """
    return _flow_test(
        model, ("d",), test_type="bayesian_lm_flow_destination",
        details={"direction": "d"},
    )  # fmt: skip


def bayesian_lm_flow_origin_test(model) -> BayesianLMTestResult:
    r"""LM test for an omitted origin lag at an OLSFlow fit (H₀: ρ_o = 0).

    See :func:`bayesian_lm_flow_destination_test`.
    """
    return _flow_test(
        model, ("o",), test_type="bayesian_lm_flow_origin", details={"direction": "o"}
    )


def bayesian_lm_flow_od_test(model) -> BayesianLMTestResult:
    r"""LM test for an omitted network lag at an OLSFlow fit (H₀: ρ_w = 0).

    See :func:`bayesian_lm_flow_destination_test`.
    """
    return _flow_test(
        model, ("w",), test_type="bayesian_lm_flow_od", details={"direction": "w"}
    )


def bayesian_lm_flow_joint_test(model) -> BayesianLMTestResult:
    r"""Joint LM test at an OLSFlow fit (H₀: ρ_d = ρ_o = ρ_w = 0).

    The three lag scores together, ``df = 3``.
    """
    return _flow_test(model, _DIRECTIONS, test_type="bayesian_lm_flow_joint")


def bayesian_robust_lm_flow_destination_test(model) -> BayesianLMTestResult:
    r"""Robust LM test for the destination lag at an OLSFlow fit.

    H₀: :math:`\rho_d = 0`, robust to local origin and network lags
    (:cite:t:`bera1993SpecificationTesting`, :cite:t:`anselin1996SimpleDiagnostic`,
    :cite:t:`dogan2021BayesianRobust`).
    """
    return _flow_test(
        model, ("d",), ("o", "w"), test_type="bayesian_robust_lm_flow_destination",
        details={"direction": "d"},
    )  # fmt: skip


def bayesian_robust_lm_flow_origin_test(model) -> BayesianLMTestResult:
    r"""Robust LM test for the origin lag at an OLSFlow fit.

    H₀: :math:`\rho_o = 0`, robust to local destination and network lags; see
    :func:`bayesian_robust_lm_flow_destination_test`.
    """
    return _flow_test(
        model, ("o",), ("d", "w"), test_type="bayesian_robust_lm_flow_origin",
        details={"direction": "o"},
    )  # fmt: skip


def bayesian_robust_lm_flow_od_test(model) -> BayesianLMTestResult:
    r"""Robust LM test for the network lag at an OLSFlow fit.

    H₀: :math:`\rho_w = 0`, robust to local destination and origin lags; see
    :func:`bayesian_robust_lm_flow_destination_test`.
    """
    return _flow_test(
        model, ("w",), ("d", "o"), test_type="bayesian_robust_lm_flow_od",
        details={"direction": "w"},
    )  # fmt: skip
