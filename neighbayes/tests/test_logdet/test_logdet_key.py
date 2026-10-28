"""The public log-determinant axes: ``logdet_method`` × ``logdet_decomp``.

The models take the two public names and translate them, once, to the internal
key the factories dispatch on (:func:`~neighbayes._logdet.logdet_key`).  These
tests pin that translation, its refusals, and the one key it added: ``exact`` on
``chol``, an exact CHOLMOD factorization at every ρ.
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp

from neighbayes._logdet import (
    FLOW_LOGDET_METHODS,
    LOGDET_DECOMPS,
    LOGDET_METHODS,
    logdet_key,
    logdet_label,
)


def _rook(side: int) -> sp.csr_matrix:
    path = sp.diags([np.ones(side - 1), np.ones(side - 1)], [-1, 1])
    eye = sp.eye(side)
    A = sp.csr_matrix(sp.kron(path, eye) + sp.kron(eye, path))
    deg = np.asarray(A.sum(axis=1)).ravel()
    return sp.csr_matrix(sp.diags(1.0 / deg) @ A)


def _knn(n: int, k: int = 3) -> sp.csr_matrix:
    """Row-standardized k-nearest neighbors: a directed graph."""
    pts = np.random.default_rng(0).random((n, 2))
    d = ((pts[:, None, :] - pts[None, :, :]) ** 2).sum(-1)
    np.fill_diagonal(d, np.inf)
    cols = np.argsort(d, axis=1)[:, :k].ravel()
    rows = np.repeat(np.arange(n), k)
    return sp.csr_matrix((np.full(n * k, 1.0 / k), (rows, cols)), shape=(n, n))


@pytest.fixture(scope="module")
def W():
    return _rook(6)


PAIRS = [
    ("cheb", "chol", "chol_cheb"),
    ("cheb", "lu", "lu_cheb"),
    ("aaa", "chol", "chol_aaa"),
    ("aaa", "lu", "lu_aaa"),
    ("spline", "lu", "lu_spline"),
]


@pytest.mark.parametrize("method, decomp, key", PAIRS)
def test_each_pair_names_its_key(W, method, decomp, key):
    assert logdet_key(method, decomp, W=W) == key
    assert logdet_label(key) == f"{method} on {decomp}"


@pytest.mark.parametrize(
    "method, key", [("eigenvalue", "eigenvalue"), ("stochastic", "stoch_cheb")]
)
@pytest.mark.parametrize("decomp", LOGDET_DECOMPS)
def test_methods_that_factor_nothing_ignore_the_decomposition(W, method, key, decomp):
    assert logdet_key(method, decomp, W=W) == key
    assert logdet_label(key) == method


def test_both_auto_leaves_the_choice_to_the_problem(W):
    assert logdet_key(None, "auto", W=W) is None


@pytest.mark.parametrize("decomp, key", [("chol", "chol_aaa"), ("lu", "lu_aaa")])
def test_a_decomposition_alone_picks_aaa(W, decomp, key):
    assert logdet_key(None, decomp, W=W) == key


def test_auto_decomposition_follows_the_graph(W):
    """``chol`` where the method has it and W is undirected; ``lu`` otherwise."""
    assert logdet_key("aaa", "auto", W=W) == "chol_aaa"
    assert logdet_key("cheb", "auto", W=W) == "chol_cheb"
    assert logdet_key("cheb", "auto", W=_knn(40)) == "lu_cheb"
    assert logdet_key("aaa", "auto", W=_knn(40)) == "lu_aaa"
    assert logdet_key("spline", "auto", W=W) == "lu_spline"


def test_unimplemented_pairs_raise(W):
    with pytest.raises(NotImplementedError, match="implemented pairs"):
        logdet_key("spline", "chol", W=W)


@pytest.mark.parametrize("method", ["exact", "chebyshev", "slq", "chol_cheb", "traces"])
def test_internal_and_retired_names_are_refused(W, method):
    with pytest.raises(ValueError, match="logdet_method must be one of"):
        logdet_key(method, W=W)


@pytest.mark.parametrize("decomp", ["eig", "qr"])
def test_unknown_decomposition_is_refused(W, decomp):
    with pytest.raises(ValueError, match="logdet_decomp must be one of"):
        logdet_key("aaa", decomp, W=W)


@pytest.mark.parametrize("method", FLOW_LOGDET_METHODS)
def test_flow_methods_are_refused_outside_the_flows(W, method):
    assert method in LOGDET_METHODS
    with pytest.raises(ValueError, match="unrestricted flow"):
        logdet_key(method, W=W)


# ---------------------------------------------------------------------------
# Model constructors
# ---------------------------------------------------------------------------


def _sar_data(side=6, seed=0):
    W = _rook(side)
    n = W.shape[0]
    rng = np.random.default_rng(seed)
    X = np.column_stack([np.ones(n), rng.normal(size=n)])
    y = sp.linalg.spsolve(
        sp.csc_matrix(sp.eye(n) - 0.4 * W), X @ [1.0, 1.0] + rng.normal(size=n)
    )
    return y, X, W


def test_model_resolves_the_pair_at_construction():
    from neighbayes.models import SAR

    y, X, W = _sar_data()
    model = SAR(y=y, X=X, W=W, logdet_method="aaa", logdet_decomp="chol")
    assert model._logdet_bounds.method == "chol_aaa"


@pytest.mark.parametrize(
    "kw",
    [
        {"logdet_refit": False},
        {"logdet_aaa_check": False},
        {"logdet_probe_check": False},
    ],
)
def test_retired_switches_are_refused(kw):
    from neighbayes.models import SAR

    y, X, W = _sar_data()
    with pytest.raises(TypeError):
        SAR(y=y, X=X, W=W, **kw)


def test_unrestricted_flow_refuses_a_single_w_method():
    from neighbayes.dgp import simulate_sar_flow
    from neighbayes.models import SARFlow

    out = simulate_sar_flow(
        n=4, rho_d=0.2, rho_o=0.1, rho_w=-0.02, beta_d=[1.0], beta_o=[0.5], seed=1
    )
    with pytest.raises(ValueError, match="trace_series"):
        SARFlow(
            out["y_vec"],
            out["X"],
            out["G"],
            col_names=out["col_names"],
            logdet_method="aaa",
        )
