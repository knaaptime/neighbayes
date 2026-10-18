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
from neighbayes._logdet._exact_chol import ExactCholLogdet


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
    ("exact", "eig", "eigenvalue"),
    ("exact", "chol", "cholmod"),
    ("cheb", "eig", "chebyshev"),
    ("cheb", "chol", "cheb_cholesky"),
    ("cheb", "lu", "lu_cheb"),
    ("aaa", "chol", "chol_aaa"),
    ("aaa", "lu", "aaa"),
    ("spline", "lu", "grid_spline"),
]


@pytest.mark.parametrize("method, decomp, key", PAIRS)
def test_each_pair_names_its_key(W, method, decomp, key):
    assert logdet_key(method, decomp, n=W.shape[0], W=W) == key
    assert logdet_label(key) == f"{method} on {decomp}"


@pytest.mark.parametrize("decomp", LOGDET_DECOMPS)
def test_stochastic_ignores_the_decomposition(W, decomp):
    assert logdet_key("stochastic", decomp, n=W.shape[0], W=W) == "cheb_stochastic"
    assert logdet_label("cheb_stochastic") == "stochastic"


def test_both_auto_leaves_the_choice_to_the_problem(W):
    assert logdet_key(None, "auto", n=W.shape[0], W=W) is None


@pytest.mark.parametrize(
    "decomp, key", [("eig", "eigenvalue"), ("chol", "chol_aaa"), ("lu", "aaa")]
)
def test_a_decomposition_alone_picks_its_method(W, decomp, key):
    assert logdet_key(None, decomp, n=W.shape[0], W=W) == key


def test_auto_decomposition_follows_n_and_direction(W):
    """``eig`` while the eigendecomposition is affordable; past it, the graph decides."""
    assert logdet_key("aaa", "auto", n=W.shape[0], W=W) == "chol_aaa"  # AAA has no eig
    assert logdet_key("cheb", "auto", n=W.shape[0], W=W) == "chebyshev"
    assert logdet_key("cheb", "auto", n=10_000, W=W) == "cheb_cholesky"
    assert logdet_key("cheb", "auto", n=10_000, W=_knn(40)) == "lu_cheb"


def test_auto_decomposition_skips_what_the_method_lacks(W):
    assert logdet_key("spline", "auto", n=W.shape[0], W=W) == "grid_spline"
    assert logdet_key("aaa", "auto", n=W.shape[0], W=_knn(40)) == "aaa"


def test_auto_decomposition_respects_the_eigen_cutoff(W, monkeypatch):
    monkeypatch.setenv("NEIGHBAYES_LOGDET_EIGEN_MAX_N", "10")
    assert logdet_key("exact", "auto", n=W.shape[0], W=W) == "cholmod"


def test_auto_decomposition_raises_when_none_fits():
    """``exact`` past the eigen cutoff on a directed graph has nothing to factor with."""
    with pytest.raises(NotImplementedError, match="no decomposition for this W"):
        logdet_key("exact", "auto", n=10_000, W=_knn(40))


@pytest.mark.parametrize(
    "method, decomp",
    [("spline", "chol"), ("spline", "eig"), ("exact", "lu"), ("aaa", "eig")],
)
def test_unimplemented_pairs_raise(W, method, decomp):
    with pytest.raises(NotImplementedError, match="implemented pairs"):
        logdet_key(method, decomp, n=W.shape[0], W=W)


@pytest.mark.parametrize("method", ["eigenvalue", "slq", "cheb_cholesky", "traces"])
def test_internal_and_retired_names_are_refused(W, method):
    with pytest.raises(ValueError, match="logdet_method must be one of"):
        logdet_key(method, n=W.shape[0], W=W)


def test_unknown_decomposition_is_refused(W):
    with pytest.raises(ValueError, match="logdet_decomp must be one of"):
        logdet_key("exact", "qr", n=W.shape[0], W=W)


@pytest.mark.parametrize("method", FLOW_LOGDET_METHODS)
def test_flow_methods_are_refused_outside_the_flows(W, method):
    assert method in LOGDET_METHODS
    with pytest.raises(ValueError, match="unrestricted flow"):
        logdet_key(method, n=W.shape[0], W=W)


# ---------------------------------------------------------------------------
# exact on chol
# ---------------------------------------------------------------------------


class TestExactChol:
    def test_value_matches_the_eigenvalues(self, W):
        lam = np.linalg.eigvals(W.toarray())
        rho = np.array([-0.9, -0.3, 0.0, 0.4, 0.95])
        exact = np.array([np.sum(np.log(np.abs(1 - r * lam))) for r in rho])
        np.testing.assert_allclose(ExactCholLogdet(W).value(rho), exact, atol=1e-10)

    def test_gradient_matches_the_resolvent_trace(self, W):
        Wd = W.toarray()
        n = Wd.shape[0]
        ev = ExactCholLogdet(W)
        for r in (-0.6, 0.2, 0.9):
            g = -np.trace(Wd @ np.linalg.inv(np.eye(n) - r * Wd))
            assert ev.grad(r)[0] == pytest.approx(g, rel=1e-7)

    def test_value_is_minus_infinity_at_the_stability_edge(self, W):
        v = ExactCholLogdet(W).value(np.array([-1.0, 1.0, 1.2]))
        assert np.all(np.isneginf(v))

    def test_directed_graph_is_refused(self):
        with pytest.raises(ValueError, match="logdet_decomp='lu'"):
            ExactCholLogdet(_knn(30))

    def test_jax_matches_numpy(self, W):
        jnp = pytest.importorskip("jax.numpy")
        pytest.importorskip("sparsax")
        from neighbayes._logdet import make_logdet_jax_fn

        fn = make_logdet_jax_fn(W, method="cholmod")
        ev = ExactCholLogdet(W)
        for r in (-0.5, 0.0, 0.7):
            assert float(fn(jnp.float64(r))) == pytest.approx(ev.value(r)[0], abs=1e-9)
        assert np.isneginf(float(fn(jnp.float64(1.0))))


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
    model = SAR(y=y, X=X, W=W, logdet_method="exact", logdet_decomp="chol")
    assert model._logdet_bounds.method == "cholmod"


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


def test_exact_chol_gibbs_agrees_with_exact_eig():
    """The CHOLMOD Jacobian and the eigenvalue Jacobian give one posterior."""
    from neighbayes.models import SAR

    y, X, W = _sar_data(side=10, seed=3)
    rho = {}
    for decomp in ("eig", "chol"):
        idata = SAR(y=y, X=X, W=W, logdet_method="exact", logdet_decomp=decomp).fit(
            draws=400,
            tune=300,
            chains=1,
            random_seed=5,
            progressbar=False,
            backend="numpy",
        )
        rho[decomp] = np.asarray(idata.posterior["rho"]).ravel()
    assert rho["chol"].mean() == pytest.approx(rho["eig"].mean(), abs=0.03)
