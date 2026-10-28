"""Separable-flow logdet factories delegate to the general single-parameter methods.

For the separable constraint ``rho_w = -rho_d * rho_o`` the flow system logdet
factors exactly as ``n*log|I - rho_d W| + n*log|I - rho_o W|``.  The factories in
:mod:`neighbayes._logdet._flow` delegate each single-parameter half to the general
factories, so every method (``eigenvalue``/``chebyshev``/``chol_cheb``/``aaa``/
``stoch_cheb``) and ``None``-auto is available — with ``aaa`` the auto choice
for directed (non-symmetric) flow ``W``.
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp

from neighbayes._logdet._config import resolve_logdet_method
from neighbayes._logdet._flow import (
    make_flow_separable_logdet,
    make_flow_separable_logdet_numpy,
)


def _row_std(A: np.ndarray) -> sp.csr_matrix:
    return sp.csr_matrix(A / A.sum(axis=1, keepdims=True))


@pytest.fixture(scope="module")
def _weights():
    rng = np.random.default_rng(0)
    n = 40
    As = rng.uniform(size=(n, n))
    As = (As + As.T) / 2.0
    np.fill_diagonal(As, 0.0)
    Ad = rng.uniform(size=(n, n))
    np.fill_diagonal(Ad, 0.0)
    return n, _row_std(As), _row_std(Ad)


def _eig_reference(W: sp.csr_matrix, n: int, rd, ro) -> np.ndarray:
    return make_flow_separable_logdet_numpy(W, n, method="eigenvalue")(rd, ro)


def test_numpy_aaa_matches_eigenvalue(_weights):
    """AAA (the directed workhorse) matches the eigenvalue reference tightly."""
    n, Wsym, Wdir = _weights
    rng = np.random.default_rng(1)
    rd = rng.uniform(0.05, 0.9, 9)
    ro = rng.uniform(0.05, 0.9, 9)
    for W in (Wsym, Wdir):
        val = make_flow_separable_logdet_numpy(W, n, method="lu_aaa")(rd, ro)
        ref = _eig_reference(W, n, rd, ro)
        np.testing.assert_allclose(val, ref, atol=1e-6, rtol=1e-6)


def test_numpy_cheb_cholesky_symmetric(_weights):
    """chol_cheb matches the reference for symmetric W (to its ~1e-3 accuracy)."""
    n, Wsym, _ = _weights
    rng = np.random.default_rng(2)
    rd = rng.uniform(0.05, 0.9, 9)
    ro = rng.uniform(0.05, 0.9, 9)
    val = make_flow_separable_logdet_numpy(Wsym, n, method="chol_cheb")(rd, ro)
    ref = _eig_reference(Wsym, n, rd, ro)
    np.testing.assert_allclose(val, ref, atol=5e-2, rtol=1e-2)


def test_numpy_cheb_cholesky_rejects_directed(_weights):
    """chol_cheb is symmetric-only; directed flow W must raise (steer to aaa)."""
    n, _, Wdir = _weights
    with pytest.raises(ValueError, match="aaa|symmetr|D-symmetr"):
        make_flow_separable_logdet_numpy(Wdir, n, method="chol_cheb")(
            np.array([0.3]), np.array([0.2])
        )


def test_numpy_none_auto_matches_eigenvalue(_weights):
    """None auto-resolves; at n=40 it lands on eigenvalue and is exact."""
    n, Wsym, Wdir = _weights
    rd = np.array([0.4, 0.1])
    ro = np.array([0.3, 0.6])
    for W in (Wsym, Wdir):
        assert resolve_logdet_method(None, n=n, W=W) == "eigenvalue"
        val = make_flow_separable_logdet_numpy(W, n, method=None)(rd, ro)
        ref = _eig_reference(W, n, rd, ro)
        np.testing.assert_allclose(val, ref, atol=1e-12)


def test_separable_factorization_identity(_weights):
    """The factory equals n*(L(rho_d) + L(rho_o)) with L the single-param logdet."""
    n, Wsym, _ = _weights
    from neighbayes._logdet._factories import make_logdet_numpy_vec_fn

    single = make_logdet_numpy_vec_fn(Wsym, eigs=None, method="lu_aaa")
    rd = np.array([0.4, 0.55])
    ro = np.array([0.2, 0.7])
    val = make_flow_separable_logdet_numpy(Wsym, n, method="lu_aaa")(rd, ro)
    expected = n * (single(rd) + single(ro))
    np.testing.assert_allclose(val, expected, atol=1e-12)


def test_pytensor_factory_builds_graph(_weights):
    """The pytensor factory returns a scalar graph (the NUTS Jacobian).

    We only build the symbolic graph (cheap) — not compile it.  The
    differentiability and numeric correctness of the underlying single-parameter
    ``make_logdet_fn`` are covered in the general logdet test suite, and the
    delegation math (``n*(L(rho_d)+L(rho_o))``) by the numpy tests above.  Full
    pytensor compilation of the AAA barycentric graph is prohibitively slow, so
    it is intentionally not exercised here.
    """
    import pytensor.tensor as pt

    n, Wsym, _ = _weights
    fn = make_flow_separable_logdet(Wsym, n, method="chol_cheb")
    expr = fn(pt.scalar("rd"), pt.scalar("ro"))
    assert expr.ndim == 0  # scalar Jacobian expression
