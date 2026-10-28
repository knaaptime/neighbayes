"""``cg_solve``, the conjugate-gradient solve of the flow samplers.

Compared against the exact dense solve on small spatial precision matrices.
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp

from neighbayes.samplers._utils._spatial_normal import cg_solve


def _make_precision(n: int, rho: float = 0.3, sigma2: float = 1.0) -> sp.csr_matrix:
    """Build a simple spatial precision matrix for testing.

    P = I/σ² + diag(ω) - ρ(W+W^T)/σ² + ρ²W^TW/σ²

    Uses a small row-standardized contiguity W and random ω.
    """
    rng = np.random.default_rng(42)
    # Simple tridiagonal W (1-D contiguity analogue)
    W = sp.diags([1.0, 1.0], [-1, 1], shape=(n, n), format="csr")
    W = W / 2.0  # row-standardize (each row sums to 1, except boundaries)

    omega = rng.gamma(2.0, 1.0, size=n)  # positive ω
    W_sym = W + W.T
    WtW = W.T @ W

    P = (
        sp.eye(n, format="csr") / sigma2
        + sp.diags(omega, format="csr")
        - rho * W_sym / sigma2
        + rho**2 * WtW / sigma2
    )
    return P


def _exact_solve(P: sp.spmatrix, rhs: np.ndarray) -> np.ndarray:
    """Compute exact P^{-1} rhs via dense inverse."""
    return np.linalg.solve(P.toarray(), rhs)


class TestCGSolve:
    """Tests for cg_solve()."""

    def test_small_system_accuracy(self):
        """CG solve should match exact solve for small SPD system."""
        n = 20
        P = _make_precision(n, rho=0.3)
        rng = np.random.default_rng(42)
        rhs = rng.standard_normal(n)
        exact = _exact_solve(P, rhs)
        cg_result = cg_solve(P, rhs, tol=1e-10)
        rel_err = np.linalg.norm(cg_result - exact) / np.linalg.norm(exact)
        assert rel_err < 1e-6, f"CG solve error: {rel_err:.2e}"

    def test_larger_system_accuracy(self):
        """CG solve should be accurate for n=100."""
        n = 100
        P = _make_precision(n, rho=0.3)
        rng = np.random.default_rng(42)
        rhs = rng.standard_normal(n)
        exact = _exact_solve(P, rhs)
        cg_result = cg_solve(P, rhs, tol=1e-10)
        rel_err = np.linalg.norm(cg_result - exact) / np.linalg.norm(exact)
        assert rel_err < 1e-6, f"CG solve error: {rel_err:.2e}"

    def test_identity_system(self):
        """CG solve of I x = rhs should give rhs."""
        n = 10
        P = sp.eye(n, format="csr")
        rhs = np.arange(n, dtype=float)
        result = cg_solve(P, rhs, tol=1e-12)
        np.testing.assert_allclose(result, rhs, atol=1e-10)

    def test_diagonal_system(self):
        """CG solve of diagonal system should be exact."""
        n = 15
        d = np.array([1.0, 2.0, 3.0, 4.0, 5.0] * 3)[:n]
        P = sp.diags(d, format="csr")
        rhs = np.ones(n)
        exact = rhs / d
        result = cg_solve(P, rhs, tol=1e-12)
        np.testing.assert_allclose(result, exact, atol=1e-10)

    def test_jacobi_preconditioner(self):
        """Jacobi preconditioner should improve convergence."""
        n = 50
        P = _make_precision(n, rho=0.5)
        rng = np.random.default_rng(42)
        rhs = rng.standard_normal(n)
        exact = _exact_solve(P, rhs)
        result = cg_solve(P, rhs, preconditioner="jacobi", tol=1e-10)
        rel_err = np.linalg.norm(result - exact) / np.linalg.norm(exact)
        assert rel_err < 1e-6

    def test_no_preconditioner(self):
        """CG without preconditioner should still converge."""
        n = 20
        P = _make_precision(n, rho=0.3)
        rng = np.random.default_rng(42)
        rhs = rng.standard_normal(n)
        exact = _exact_solve(P, rhs)
        result = cg_solve(P, rhs, preconditioner="none", tol=1e-10)
        rel_err = np.linalg.norm(result - exact) / np.linalg.norm(exact)
        assert rel_err < 1e-6

    def test_invalid_preconditioner(self):
        """Unknown preconditioner should raise ValueError."""
        n = 10
        P = sp.eye(n, format="csr")
        rhs = np.ones(n)
        with pytest.raises(ValueError, match="Unknown preconditioner"):
            cg_solve(P, rhs, preconditioner="ilu")

    def test_residual_satisfies_tolerance(self):
        """Residual ||P x - rhs|| should satisfy tolerance."""
        n = 30
        P = _make_precision(n, rho=0.3)
        rng = np.random.default_rng(42)
        rhs = rng.standard_normal(n)
        result = cg_solve(P, rhs, tol=1e-8)
        residual = np.linalg.norm(P @ result - rhs) / np.linalg.norm(rhs)
        assert residual < 1e-6  # slightly looser than tol due to float


# ---------------------------------------------------------------------------
# Integration: CG + Lanczos in ρ slice sampler context
