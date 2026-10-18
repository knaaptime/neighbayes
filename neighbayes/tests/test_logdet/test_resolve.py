"""Tests for the LogDetMethod enum and resolve_logdet_method validator."""

import pytest

from neighbayes._logdet import (
    VALID_LOGDET_METHODS,
    LogDetMethod,
    resolve_logdet_method,
)

CANONICAL_METHODS = {
    "eigenvalue",
    "chebyshev",
    "cheb_stochastic",
    "cheb_cholesky",
    "lu_cheb",
    "aaa",
    "chol_aaa",
    "grid_spline",
    "cholmod",
}


def test_enum_members_match_canonical_names():
    assert {m.value for m in LogDetMethod} == CANONICAL_METHODS


def test_valid_logdet_methods_constant():
    assert VALID_LOGDET_METHODS == CANONICAL_METHODS


def test_enum_str_equality():
    assert LogDetMethod.EIGENVALUE == "eigenvalue"
    assert LogDetMethod.CHEBYSHEV == "chebyshev"


@pytest.mark.parametrize("name", sorted(CANONICAL_METHODS))
def test_resolve_accepts_canonical_names(name):
    assert resolve_logdet_method(name, n=100) == name


def test_resolve_none_auto_selects():
    assert resolve_logdet_method(None, n=100) == "eigenvalue"
    assert resolve_logdet_method(None, n=1000) == "chol_aaa"
    assert resolve_logdet_method(None, n=10000) == "chol_aaa"
    assert resolve_logdet_method(None, n=50000) == "chol_aaa"
    assert resolve_logdet_method(None, n=200000) == "cheb_stochastic"


def test_resolve_none_auto_selects_nonsymmetric_W():
    """Non-symmetric W in medium range should auto-select aaa."""
    import numpy as np
    import scipy.sparse as sp

    # Build a non-symmetric sparse W (directed graph)
    n = 1000
    np.random.seed(42)
    rows = np.array([0, 0, 1, 1, 2, 3, 3, 4])
    cols = np.array([1, 2, 0, 3, 4, 1, 4, 2])
    vals = np.ones(len(rows))
    W_nonsym = sp.csr_matrix((vals, (rows, cols)), shape=(n, n))

    assert resolve_logdet_method(None, n=n, W=W_nonsym) == "aaa"

    # Symmetric W should select chol_aaa (Cholesky-based AAA)
    W_sym = sp.csr_matrix((vals, (rows, cols)), shape=(n, n))
    W_sym = W_sym + W_sym.T
    assert resolve_logdet_method(None, n=n, W=W_sym) == "chol_aaa"

    # No W provided defaults to chol_aaa (symmetric assumption)
    assert resolve_logdet_method(None, n=n) == "chol_aaa"


def test_resolve_unknown_method_raises():
    with pytest.raises(ValueError, match="Unknown logdet method"):
        resolve_logdet_method("bogus", n=100)


# ---------------------------------------------------------------------------
# Fill-in guard tests
# ---------------------------------------------------------------------------


def _knn_graph(n: int, k: int, seed: int = 42):
    """Build a symmetric k-NN graph with high fill-in."""
    import numpy as np
    import scipy.sparse as sp

    rng = np.random.default_rng(seed)
    pts = rng.standard_normal((n, 2))
    d = np.sqrt(((pts[:, None, :] - pts[None, :, :]) ** 2).sum(axis=2))
    W = np.zeros((n, n))
    for i in range(n):
        nearest = np.argsort(d[i])[: k + 1]
        for j in nearest:
            if j != i:
                W[i, j] = 1.0
    W = np.maximum(W, W.T)
    return sp.csr_matrix(W)


def test_fillin_guard_routes_dense_to_stochastic():
    """Dense W at medium n should auto-select cheb_stochastic with a warning."""
    import numpy as np
    import scipy.sparse as sp

    n = 2000
    W_dense = np.ones((n, n)) - np.eye(n)  # fully dense, symmetric
    with pytest.warns(UserWarning, match="fill-in ratio"):
        result = resolve_logdet_method(None, n=n, W=W_dense)
    assert result == "cheb_stochastic"


def test_fillin_guard_routes_high_degree_sparse_to_stochastic():
    """High-degree sparse W (KNN) at medium n should route to stochastic."""
    n = 1000
    W = _knn_graph(n, k=50)
    with pytest.warns(UserWarning, match="fill-in ratio"):
        result = resolve_logdet_method(None, n=n, W=W)
    assert result == "cheb_stochastic"


def test_fillin_guard_preserves_low_degree_sparse():
    """Low-degree sparse W (rook-like) should still route to chol_aaa."""
    import numpy as np
    import scipy.sparse as sp

    # Small sparse symmetric graph with low degree — ratio well below 20
    rows = np.array([0, 1, 1, 2, 2, 3, 3, 0])
    cols = np.array([1, 0, 2, 1, 3, 2, 0, 3])
    vals = np.ones(len(rows))
    W = sp.csr_matrix((vals, (rows, cols)), shape=(4, 4))
    # n=4 is below eigen_cutoff (500), so this hits eigenvalue, not the guard.
    # Use n in the medium range with a sparse W.
    n = 1000
    W_sparse = sp.csr_matrix((vals, (rows, cols)), shape=(n, n))
    W_sparse = W_sparse + W_sparse.T
    result = resolve_logdet_method(None, n=n, W=W_sparse)
    assert result == "chol_aaa"


def test_fillin_guard_no_W_unchanged():
    """No W provided should still default to chol_aaa (no fill-in check)."""
    result = resolve_logdet_method(None, n=10000)
    assert result == "chol_aaa"


def test_fillin_guard_explicit_method_respected():
    """Explicit logdet_method should bypass the fill-in guard entirely."""
    import numpy as np

    n = 2000
    W_dense = np.ones((n, n)) - np.eye(n)
    # No warning should be emitted when method is explicit
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = resolve_logdet_method("chol_aaa", n=n, W=W_dense)
    assert result == "chol_aaa"


def test_fillin_guard_env_var_override():
    """NEIGHBAYES_LOGDET_MAX_FILLIN_RATIO=999 should disable the guard."""
    import numpy as np

    n = 2000
    W_dense = np.ones((n, n)) - np.eye(n)
    # Set env var to effectively disable the guard
    import os

    old = os.environ.get("NEIGHBAYES_LOGDET_MAX_FILLIN_RATIO")
    os.environ["NEIGHBAYES_LOGDET_MAX_FILLIN_RATIO"] = "99999"
    try:
        result = resolve_logdet_method(None, n=n, W=W_dense)
        assert result == "chol_aaa"
    finally:
        if old is None:
            os.environ.pop("NEIGHBAYES_LOGDET_MAX_FILLIN_RATIO", None)
        else:
            os.environ["NEIGHBAYES_LOGDET_MAX_FILLIN_RATIO"] = old


def test_grid_spline_matches_exact_and_across_backends():
    """The incumbent baseline is wired into every factory and agrees across them."""
    import numpy as np
    import scipy.sparse as sp

    from neighbayes._logdet import (
        make_logdet_grad_numpy_fn,
        make_logdet_numpy_fn,
        make_logdet_numpy_vec_fn,
    )

    side = 20
    n = side * side
    rows, cols = [], []
    for i in range(side):
        for j in range(side):
            for di, dj in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                ni, nj = i + di, j + dj
                if 0 <= ni < side and 0 <= nj < side:
                    rows.append(i * side + j)
                    cols.append(ni * side + nj)
    A = sp.coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n)).tocsr()
    deg = np.asarray(A.sum(axis=1)).ravel()
    W = sp.csr_matrix(sp.diags(1.0 / deg) @ A)

    f = make_logdet_numpy_fn(W, None, "grid_spline", -0.99, 0.99)
    fv = make_logdet_numpy_vec_fn(W, None, "grid_spline", -0.99, 0.99)
    fg = make_logdet_grad_numpy_fn(W, None, "grid_spline", -0.99, 0.99)

    rhos = np.linspace(-0.9, 0.9, 11)
    eigs = np.linalg.eigvals(W.toarray())
    exact = np.array([np.sum(np.log(np.abs(1.0 - r * eigs))) for r in rhos])
    approx = np.array([f(r) for r in rhos])

    # Interpolation error only; the spline is far looser than AAA but not wrong.
    assert np.abs(approx - exact).max() < 1e-3
    # Scalar and vectorised factories must agree exactly.
    np.testing.assert_allclose(approx, np.asarray(fv(rhos)), rtol=0, atol=0)
    # Gradient is the analytic spline derivative.
    fd = (f(0.5 + 1e-6) - f(0.5 - 1e-6)) / 2e-6
    assert abs(fg(0.5) - fd) < 1e-6
