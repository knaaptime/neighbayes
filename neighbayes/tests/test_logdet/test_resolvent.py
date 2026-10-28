"""Tests for the resolvent-trace / logdet-gradient core (``_resolvent.py``).

Each ``logdet_grad_*`` returns g(ρ) = d/dρ log|I − ρW| for a representation an
existing precompute produces.  The core guarantees, verified here:

1. **derivative correctness** — matches a central finite-difference of the
   corresponding *value* surrogate (backend-independent, ~1e-6);
2. **autodiff parity** — matches ``jax.grad`` of the corresponding jax closure
   (~1e-10), so the numpy path and the autodiff paths compute the same object;
3. **exactness** (eigenvalue) — matches dense-eigenvalue truth and a
   finite-difference of the exact logdet;
4. **backend agnosticism** — ``xp=numpy`` and ``xp=jax.numpy`` agree.
"""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest
import scipy.sparse as sp

from neighbayes._logdet import (
    aaa_logdet_eval,
    aaa_logdet_precompute,
    make_logdet_grad_numpy_fn,
    make_logdet_numpy_fn,
)
from neighbayes._logdet._chol_cheb import (
    chol_cheb_logdet_precompute,
)
from neighbayes._logdet._clenshaw import clenshaw_scalar
from neighbayes._logdet._resolvent import (
    clenshaw_deriv_x,
    logdet_grad_aaa,
    logdet_grad_chebyshev,
    logdet_grad_eigenvalue,
)
from neighbayes.tests.helpers import eigen_cheb_coeffs

_HAS_JAX = importlib.util.find_spec("jax") is not None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _rook(side: int) -> np.ndarray:
    """Row-standardized rook-contiguity W (symmetric sparsity → Lanczos)."""
    n = side * side
    A = np.zeros((n, n), dtype=float)
    for r in range(side):
        for c in range(side):
            i = r * side + c
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                rr, cc = r + dr, c + dc
                if 0 <= rr < side and 0 <= cc < side:
                    A[i, rr * side + cc] = 1.0
    return A / A.sum(axis=1, keepdims=True)


def _knn(n: int, k: int) -> np.ndarray:
    """Row-standardized kNN W (asymmetric sparsity → Arnoldi / complex eigs)."""
    rng = np.random.default_rng(0)
    pts = rng.random((n, 2))
    A = np.zeros((n, n), dtype=float)
    for i in range(n):
        d = np.sum((pts - pts[i]) ** 2, axis=1)
        d[i] = np.inf
        for j in np.argsort(d)[:k]:
            A[i, j] = 1.0
    return A / A.sum(axis=1, keepdims=True)


def _fd(value_fn, rho, h=1e-6):
    """Central finite-difference of a scalar value function."""
    return (value_fn(rho + h) - value_fn(rho - h)) / (2 * h)


def _exact_logdet(rho, W):
    n = W.shape[0]
    return float(np.linalg.slogdet(np.eye(n) - rho * W)[1])


# ---------------------------------------------------------------------------
# clenshaw_deriv_x — pure math
# ---------------------------------------------------------------------------


class TestClenshawDerivX:
    def test_matches_numpy_chebyshev_derivative(self):
        """d/dx Σ c_j T_j(x) matches numpy.polynomial.chebyshev."""
        rng = np.random.default_rng(1)
        coeffs = rng.standard_normal(12)
        deriv_coeffs = np.polynomial.chebyshev.chebder(coeffs)
        for x in [-0.9, -0.3, 0.0, 0.25, 0.8]:
            got = float(clenshaw_deriv_x(coeffs, np.float64(x)))
            want = float(np.polynomial.chebyshev.chebval(x, deriv_coeffs))
            np.testing.assert_allclose(got, want, rtol=1e-11, atol=1e-11)

    def test_constant_and_empty(self):
        assert float(clenshaw_deriv_x(np.array([3.0]), np.float64(0.4))) == 0.0
        assert float(clenshaw_deriv_x(np.array([]), np.float64(0.4))) == 0.0

    def test_linear(self):
        # c0 + c1 T_1(x) = c0 + c1 x  ⇒  derivative c1
        got = float(clenshaw_deriv_x(np.array([2.0, 1.5]), np.float64(0.3)))
        np.testing.assert_allclose(got, 1.5, rtol=1e-12)


# ---------------------------------------------------------------------------
# eigenvalue — exact
# ---------------------------------------------------------------------------


class TestEigenvalueGrad:
    def test_matches_exact_logdet_fd_symmetric(self):
        W = _rook(5)
        eigs = np.linalg.eigvals(W)
        for rho in [-0.4, 0.0, 0.3, 0.6, 0.85]:
            g = float(logdet_grad_eigenvalue(rho, eigs))
            fd = _fd(lambda r: _exact_logdet(r, W), rho)
            np.testing.assert_allclose(g, fd, rtol=1e-5, atol=1e-6)

    def test_matches_exact_logdet_fd_directed(self):
        W = _knn(16, k=3)
        eigs = np.linalg.eigvals(W)  # complex
        for rho in [-0.3, 0.0, 0.3, 0.6]:
            g = float(logdet_grad_eigenvalue(rho, eigs))
            fd = _fd(lambda r: _exact_logdet(r, W), rho)
            np.testing.assert_allclose(g, fd, rtol=1e-5, atol=1e-6)


# ---------------------------------------------------------------------------
# chebyshev-family — matches FD of the Clenshaw value form
# ---------------------------------------------------------------------------


class TestChebyshevGrad:
    def test_chebyshev_matches_value_fd(self):
        W = _rook(5)
        out = eigen_cheb_coeffs(W, order=20, rmin=-0.9, rmax=0.9)
        coeffs, rmin, rmax = out["coeffs"], out["rmin"], out["rmax"]
        for rho in [-0.5, 0.0, 0.4, 0.7]:
            g = float(logdet_grad_chebyshev(rho, coeffs, rmin, rmax))
            fd = _fd(lambda r: clenshaw_scalar(coeffs, r, rmin, rmax), rho)
            np.testing.assert_allclose(g, fd, rtol=1e-5, atol=1e-7)

    def test_cheb_cholesky_matches_value_fd(self):
        W = _rook(5)
        pre = chol_cheb_logdet_precompute(
            sp.csr_matrix(W), order=None, rho_min=0.1, rho_max=0.8
        )
        coeffs, rmin, rmax = pre.coeffs, pre.rho_min, pre.rho_max
        for rho in [0.2, 0.4, 0.6, 0.75]:
            g = float(logdet_grad_chebyshev(rho, coeffs, rmin, rmax))
            fd = _fd(lambda r: clenshaw_scalar(coeffs, r, rmin, rmax), rho)
            np.testing.assert_allclose(g, fd, rtol=1e-5, atol=1e-7)

    def test_cheb_cholesky_grad_near_exact(self):
        """chol-cheb is exact, so its gradient tracks the true logdet gradient."""
        W = _rook(5)
        eigs = np.linalg.eigvals(W)
        pre = chol_cheb_logdet_precompute(
            sp.csr_matrix(W), order=None, rho_min=0.1, rho_max=0.8
        )
        for rho in [0.2, 0.5, 0.7]:
            g = float(logdet_grad_chebyshev(rho, pre.coeffs, pre.rho_min, pre.rho_max))
            g_exact = float(logdet_grad_eigenvalue(rho, eigs))
            np.testing.assert_allclose(g, g_exact, rtol=1e-4, atol=1e-5)


# ---------------------------------------------------------------------------
# AAA — directed W, matches FD of the barycentric value form
# ---------------------------------------------------------------------------


class TestAAAGrad:
    def test_aaa_matches_value_fd(self):
        W = _knn(20, k=4)
        pre = aaa_logdet_precompute(sp.csr_matrix(W), rho_min=0.1, rho_max=0.8)
        for rho in [0.2, 0.4, 0.6, 0.75]:
            g = float(
                logdet_grad_aaa(
                    rho, pre.support_points, pre.support_values, pre.weights
                )
            )
            fd = _fd(lambda r: aaa_logdet_eval(pre, r), rho)
            np.testing.assert_allclose(g, fd, rtol=1e-4, atol=1e-6)

    def test_aaa_grad_near_exact(self):
        """AAA is exact for directed W, so its gradient tracks the true one."""
        W = _knn(20, k=4)
        eigs = np.linalg.eigvals(W)
        pre = aaa_logdet_precompute(sp.csr_matrix(W), rho_min=0.1, rho_max=0.8)
        for rho in [0.2, 0.5, 0.7]:
            g = float(
                logdet_grad_aaa(
                    rho, pre.support_points, pre.support_values, pre.weights
                )
            )
            g_exact = float(logdet_grad_eigenvalue(rho, eigs))
            np.testing.assert_allclose(g, g_exact, rtol=1e-4, atol=1e-4)


# ---------------------------------------------------------------------------
# make_logdet_grad_numpy_fn — matched pair with make_logdet_numpy_fn
# ---------------------------------------------------------------------------


class TestNumpyGradFactory:
    """The gradient factory must be the exact derivative of the value factory.

    For gradient-based samplers the (logp, grad) pair must satisfy
    ``grad == d(logp)/dρ`` or leapfrog/MALA/NUTS break — this pins every
    method's gradient factory to a finite-difference of its own value factory.
    """

    def _check(self, W_sparse, eigs, method, rho_min, rho_max, rhos, tol=1e-5):
        val = make_logdet_numpy_fn(
            W_sparse, eigs, method, rho_min=rho_min, rho_max=rho_max
        )
        grad = make_logdet_grad_numpy_fn(
            W_sparse, eigs, method, rho_min=rho_min, rho_max=rho_max
        )
        for rho in rhos:
            g = grad(rho)
            fd = _fd(val, rho)
            np.testing.assert_allclose(g, fd, rtol=tol, atol=1e-6)

    def test_eigenvalue(self):
        W = _rook(5)
        self._check(
            sp.csr_matrix(W),
            np.linalg.eigvals(W),
            "eigenvalue",
            -1.0,
            1.0,
            [-0.4, 0.0, 0.3, 0.6],
        )

    def test_cheb_cholesky(self):
        W = _rook(5)
        self._check(
            sp.csr_matrix(W), None, "chol_cheb", 0.1, 0.8, [0.2, 0.4, 0.6, 0.75]
        )

    def test_cheb_stochastic(self):
        W = _rook(6)
        self._check(sp.csr_matrix(W), None, "stoch_cheb", -0.9, 0.9, [-0.4, 0.0, 0.4])

    def test_aaa(self):
        W = _knn(20, k=4)
        self._check(sp.csr_matrix(W), None, "lu_aaa", 0.1, 0.8, [0.2, 0.4, 0.6, 0.75])

    def test_panel_multiplier(self):
        """T scales the gradient (gradient of T·logdet)."""
        W = _rook(5)
        eigs = np.linalg.eigvals(W)
        g1 = make_logdet_grad_numpy_fn(sp.csr_matrix(W), eigs, "eigenvalue", T=1)
        g3 = make_logdet_grad_numpy_fn(sp.csr_matrix(W), eigs, "eigenvalue", T=3)
        for rho in [0.2, 0.5]:
            np.testing.assert_allclose(g3(rho), 3.0 * g1(rho), rtol=1e-12)


# ---------------------------------------------------------------------------
# Autodiff parity + backend agnosticism (jax-gated)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _HAS_JAX, reason="JAX not installed")
class TestAutodiffParity:
    def _setup_jax(self):
        import jax

        jax.config.update("jax_enable_x64", True)
        return jax

    def test_eigenvalue_matches_jax_grad(self):
        jax = self._setup_jax()
        import jax.numpy as jnp

        from neighbayes._logdet import make_logdet_jax_fn

        W = _rook(5)
        eigs = np.linalg.eigvals(W)
        fn = make_logdet_jax_fn(W, method="eigenvalue")
        for rho in [-0.4, 0.0, 0.3, 0.6]:
            g_core = float(logdet_grad_eigenvalue(rho, eigs))
            g_ad = float(jax.grad(fn)(jnp.float64(rho)))
            np.testing.assert_allclose(g_core, g_ad, rtol=1e-9, atol=1e-10)

    def test_cheb_cholesky_matches_jax_grad(self):
        jax = self._setup_jax()
        import jax.numpy as jnp

        from neighbayes._logdet import make_logdet_jax_fn

        W = _rook(5)
        pre = chol_cheb_logdet_precompute(
            sp.csr_matrix(W), order=None, rho_min=0.1, rho_max=0.8
        )
        fn = make_logdet_jax_fn(W, method="chol_cheb", rho_min=0.1, rho_max=0.8)
        for rho in [0.2, 0.5, 0.7]:
            g_core = float(
                logdet_grad_chebyshev(rho, pre.coeffs, pre.rho_min, pre.rho_max)
            )
            g_ad = float(jax.grad(fn)(jnp.float64(rho)))
            np.testing.assert_allclose(g_core, g_ad, rtol=1e-9, atol=1e-10)

    def test_aaa_matches_jax_grad(self):
        jax = self._setup_jax()
        import jax.numpy as jnp

        from neighbayes._logdet import make_logdet_jax_fn

        W = _knn(20, k=4)
        pre = aaa_logdet_precompute(sp.csr_matrix(W), rho_min=0.1, rho_max=0.8)
        fn = make_logdet_jax_fn(
            sp.csr_matrix(W), method="lu_aaa", rho_min=0.1, rho_max=0.8
        )
        for rho in [0.2, 0.5, 0.7]:
            g_core = float(
                logdet_grad_aaa(
                    rho, pre.support_points, pre.support_values, pre.weights
                )
            )
            g_ad = float(jax.grad(fn)(jnp.float64(rho)))
            np.testing.assert_allclose(g_core, g_ad, rtol=1e-9, atol=1e-10)

    def test_xp_agnostic_chebyshev_and_aaa(self):
        """numpy and jax.numpy backends produce identical gradients."""
        self._setup_jax()
        import jax.numpy as jnp

        W = _rook(5)
        out = eigen_cheb_coeffs(W, order=20, rmin=-0.9, rmax=0.9)
        coeffs, rmin, rmax = out["coeffs"], out["rmin"], out["rmax"]
        pre = aaa_logdet_precompute(
            sp.csr_matrix(_knn(20, k=4)), rho_min=0.1, rho_max=0.8
        )
        for rho in [0.2, 0.5, 0.7]:
            g_np = float(logdet_grad_chebyshev(rho, coeffs, rmin, rmax))
            g_jx = float(
                logdet_grad_chebyshev(jnp.float64(rho), coeffs, rmin, rmax, xp=jnp)
            )
            np.testing.assert_allclose(g_np, g_jx, rtol=1e-10, atol=1e-11)

            a_np = float(
                logdet_grad_aaa(
                    rho, pre.support_points, pre.support_values, pre.weights
                )
            )
            a_jx = float(
                logdet_grad_aaa(
                    jnp.float64(rho),
                    pre.support_points,
                    pre.support_values,
                    pre.weights,
                    xp=jnp,
                )
            )
            np.testing.assert_allclose(a_np, a_jx, rtol=1e-10, atol=1e-11)
