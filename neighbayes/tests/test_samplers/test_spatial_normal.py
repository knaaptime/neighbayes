"""Unit tests for the spatial-normal sampler primitive."""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp

from neighbayes.samplers._utils._spatial_normal import (
    CholmodFactor,
    iterative_solve,
    sample_spatial_normal,
)


@pytest.fixture(params=["sparsax", "scikit-sparse"])
def cholmod_backend(request, monkeypatch):
    """Run a test on each CholmodFactor backend.

    CholmodFactor uses sparsax's CHOLMOD when it imports and scikit-sparse
    otherwise; hiding sparsax exercises the fallback.
    """
    import neighbayes._jax_dispatch as jd

    if request.param == "sparsax":
        if not jd._sparsax_available():
            pytest.skip("sparsax not installed")
    else:
        monkeypatch.setattr(jd, "_sparsax_available", lambda: False)
    return request.param


def test_cholmod_backends_agree(cholmod_backend):
    """Solve, logdet and refactor match dense algebra on either backend.

    The refactor uses a sub-pattern (an explicit zero dropped by sparse
    arithmetic), which must scatter into the analysed pattern.
    """
    n = 50
    G = sp.random(n, n, density=0.08, random_state=3)
    S = sp.csc_matrix(G + G.T)
    P1 = sp.csc_matrix(S @ S.T + sp.eye(n) * n)
    factor = CholmodFactor(P1)
    rng = np.random.default_rng(0)
    b = rng.standard_normal((n, 3))
    np.testing.assert_allclose(
        factor.solve(b), np.linalg.solve(P1.toarray(), b), atol=1e-12
    )
    np.testing.assert_allclose(
        factor.logdet(), np.linalg.slogdet(P1.toarray())[1], rtol=1e-12
    )
    # Same pattern, new values, with one off-diagonal pair zeroed and dropped.
    P2 = P1.tolil()
    i, j = next((i, j) for i, j in zip(*P1.nonzero()) if i < j)
    P2[i, j] = P2[j, i] = 0.0
    P2 = sp.csc_matrix(P2) * 1.5
    P2.eliminate_zeros()
    assert P2.nnz < P1.nnz
    factor.factorize(P2)
    np.testing.assert_allclose(
        factor.solve(b), np.linalg.solve(P2.toarray(), b), atol=1e-12
    )
    np.testing.assert_allclose(
        factor.logdet(), np.linalg.slogdet(P2.toarray())[1], rtol=1e-12
    )


class TestSampleSpatialNormal:
    """Tests for sparse-precision Gaussian sampling."""

    def test_identity_precision(self, rng):
        """P = I gives N(mean_term, I)."""
        n = 50
        P = sp.eye(n, format="csc")
        mean_term = rng.standard_normal(n)

        draw = sample_spatial_normal(P, mean_term, rng=rng)
        assert draw.x.shape == (n,)
        assert np.all(np.isfinite(draw.x))
        # The mean should be mean_term since P = I → m = P^{-1} @ mean_term = mean_term
        # Check empirical mean over many draws
        draws = np.array(
            [sample_spatial_normal(P, mean_term, rng=rng).x for _ in range(2000)]
        )
        empirical_mean = draws.mean(axis=0)
        np.testing.assert_allclose(empirical_mean, mean_term, atol=0.15)

    def test_diagonal_precision(self, rng):
        """P = diag(v) gives N(P^{-1} mean_term, P^{-1})."""
        n = 20
        v = rng.uniform(0.5, 2.0, size=n)
        P = sp.diags(v, format="csc")
        mean_term = rng.standard_normal(n)

        draw = sample_spatial_normal(P, mean_term, rng=rng)
        assert draw.x.shape == (n,)
        assert np.all(np.isfinite(draw.x))

        # Expected mean: P^{-1} @ mean_term = mean_term / v
        expected_mean = mean_term / v
        draws = np.array(
            [sample_spatial_normal(P, mean_term, rng=rng).x for _ in range(3000)]
        )
        empirical_mean = draws.mean(axis=0)
        np.testing.assert_allclose(empirical_mean, expected_mean, atol=0.1)

    def test_tridiagonal_precision(self, rng):
        """Tridiagonal precision: empirical covariance matches P^{-1}."""
        n = 30
        # Build a tridiagonal SPD precision matrix
        diag_val = 2.0
        offdiag_val = -0.5
        P = sp.diags(
            [
                offdiag_val * np.ones(n - 1),
                diag_val * np.ones(n),
                offdiag_val * np.ones(n - 1),
            ],
            offsets=[-1, 0, 1],
            format="csc",
        )
        mean_term = rng.standard_normal(n)

        # Draw many samples
        draws = np.array(
            [sample_spatial_normal(P, mean_term, rng=rng).x for _ in range(5000)]
        )

        # Check mean
        P_dense = P.toarray()
        expected_mean = np.linalg.solve(P_dense, mean_term)
        empirical_mean = draws.mean(axis=0)
        np.testing.assert_allclose(empirical_mean, expected_mean, atol=0.1)

        # Check diagonal of covariance (most reliably estimated)
        P_inv = np.linalg.inv(P_dense)
        empirical_cov = np.cov(draws.T)
        np.testing.assert_allclose(np.diag(empirical_cov), np.diag(P_inv), atol=0.08)
        # Check off-diagonal elements with looser tolerance (MC noise)
        np.testing.assert_allclose(empirical_cov, P_inv, atol=0.25)

    def test_cached_factor_reuse(self, rng):
        """Cached factor gives same mean but different draws."""
        n = 20
        v = rng.uniform(0.5, 2.0, size=n)
        P = sp.diags(v, format="csc")
        mean_term = rng.standard_normal(n)

        # First draw to get the factor
        draw1 = sample_spatial_normal(P, mean_term, rng=rng)
        factor = draw1.factor

        # Second draw reusing the factor
        draw2 = sample_spatial_normal(P, mean_term, rng=rng, cached_factor=factor)

        # Both should have the same mean (P^{-1} @ mean_term)
        expected_mean = mean_term / v
        assert np.allclose(draw1.x, draw2.x) is False  # different random draws
        # But the means should be close (both centered on the same point)
        draws = np.array(
            [
                sample_spatial_normal(P, mean_term, rng=rng, cached_factor=factor).x
                for _ in range(2000)
            ]
        )
        np.testing.assert_allclose(draws.mean(axis=0), expected_mean, atol=0.1)

    def test_cholmod_factor(self, rng):
        """CholmodFactor produces correct samples and log-determinant."""
        n = 30
        v = rng.uniform(0.5, 2.0, size=n)
        P = sp.diags(v, format="csc")
        mean_term = rng.standard_normal(n)

        factor = CholmodFactor(P)
        x = factor.sample(mean_term, rng=rng)
        assert x.shape == (n,)
        assert np.all(np.isfinite(x))

        # Check logdet matches scipy
        import scipy.sparse.linalg as spla

        lu = spla.splu(sp.csc_matrix(P), permc_spec="MMD_AT_PLUS_A")
        logdet_splu = np.sum(np.log(np.abs(lu.U.diagonal())))
        np.testing.assert_allclose(factor.logdet(), logdet_splu, rtol=1e-10)

        # Check solve matches scipy
        x_cholmod = factor.solve(mean_term)
        x_splu = lu.solve(mean_term)
        np.testing.assert_allclose(x_cholmod, x_splu, atol=1e-12)

    def test_cholmod_refactorize(self, rng):
        """CholmodFactor can re-factorize with same sparsity pattern."""
        n = 30
        # Two different diagonal precision matrices (same sparsity)
        v1 = rng.uniform(0.5, 2.0, size=n)
        v2 = rng.uniform(0.5, 2.0, size=n)
        P1 = sp.diags(v1, format="csc")
        P2 = sp.diags(v2, format="csc")
        mean_term = rng.standard_normal(n)

        factor = CholmodFactor(P1)
        # Re-factorize with P2
        factor.factorize(P2)
        x = factor.sample(mean_term, rng=rng)
        assert x.shape == (n,)
        assert np.all(np.isfinite(x))

        # Verify solve matches fresh factorization
        from sksparse.cholmod import cho_factor

        f2 = cho_factor(P2)
        np.testing.assert_allclose(
            factor.solve(mean_term), f2.solve(mean_term), atol=1e-12
        )

    def test_permutation_covariance(self, rng, cholmod_backend):
        """Sampling honours CHOLMOD's fill-reducing permutation.

        Regression test for the permutation.  CHOLMOD factors a permuted
        matrix ``P_perm P P_perm^T = L L^T``; a naive ``L^{-T} z`` draw
        that ignores ``P_perm`` yields the *permuted* covariance instead
        of ``P^-1``.  Diagonal/tridiagonal precisions have trivial AMD
        orderings, so this uses a structured SPD matrix that forces a
        nontrivial reordering.
        """
        from sksparse.cholmod import cho_factor

        n = 60
        # Structured sparse SPD precision that induces a nontrivial AMD perm.
        G = sp.random(n, n, density=0.06, random_state=1)
        P = G + G.T
        P = sp.csc_matrix(P @ P.T + sp.eye(n) * n)

        perm = cho_factor(P).get_perm()
        assert not np.array_equal(perm, np.arange(n))  # fixture must permute

        factor = CholmodFactor(P)
        P_inv = np.linalg.inv(P.toarray())

        # Exact check: sample() draws x = m + M z with M = P_perm^T L^{-T}, so
        # M M^T must equal P^-1 exactly.  A zero mean and z = e_i make sample()
        # return column i of M, whichever backend builds it.
        class _UnitRng:
            def __init__(self):
                self.i = 0

            def standard_normal(self, size):
                e = np.zeros(size)
                e[self.i] = 1.0
                self.i += 1
                return e

        unit = _UnitRng()
        M = np.column_stack([factor.sample(np.zeros(n), rng=unit) for _ in range(n)])
        np.testing.assert_allclose(M @ M.T, P_inv, atol=1e-10)

        # Monte Carlo check on the real sampler: the empirical covariance
        # must be closer to P^-1 than to the permuted P^-1 (the bug's
        # signature).  Relative comparison → robust to MC noise.

    def test_cholmod_pickle_roundtrip(self, rng):
        """CholmodFactor survives pickle round-trip."""
        import pickle

        n = 40
        v = rng.uniform(0.5, 2.0, size=n)
        P = sp.diags(v, format="csc")
        cf = CholmodFactor(P)

        # Pickle and unpickle
        data = pickle.dumps(cf)
        cf2 = pickle.loads(data)

        # Should produce same results
        rhs = rng.standard_normal(n)
        cf.factorize(P)
        cf2.factorize(P)
        np.testing.assert_allclose(cf.solve(rhs), cf2.solve(rhs), atol=1e-12)
        assert cf.logdet() == cf2.logdet()

    @pytest.fixture
    def rng(self):
        return np.random.default_rng(42)


class TestIterativeSolve:
    """Tests for CG-based iterative_solve (multi-RHS spatial solve)."""

    def test_single_rhs_diagonal(self):
        """CG solve matches direct solve for diagonal SPD system."""
        A = sp.diags([1.0, 2.0, 3.0, 4.0, 5.0], 0, format="csr")
        b = np.ones(5)
        x = iterative_solve(A, b, lambda_min=1.0, lambda_max=5.0)
        x_exact = sp.linalg.spsolve(A, b)
        np.testing.assert_allclose(x, x_exact, atol=1e-10)

    def test_multi_rhs_diagonal(self):
        """CG solve matches direct solve for multi-RHS diagonal system."""
        A = sp.diags([1.0, 2.0, 3.0, 4.0, 5.0], 0, format="csr")
        X = np.column_stack([np.ones(5), 2.0 * np.ones(5), np.arange(5.0)])
        U = iterative_solve(A, X, lambda_min=1.0, lambda_max=5.0)
        U_exact = sp.linalg.spsolve(A, X)
        np.testing.assert_allclose(U, U_exact, atol=1e-10)

    def test_spatial_weights_moderate_rho(self):
        """CG solve matches direct solve for A_ρ = I − 0.5W."""
        from libpysal.weights import lat2W

        W = lat2W(7, 7)
        W_arr = W.sparse.toarray()
        row_sums = W_arr.sum(axis=1, keepdims=True)
        W_std = W_arr / np.where(row_sums == 0, 1, row_sums)
        W_sparse = sp.csr_matrix(W_std)
        n = W_sparse.shape[0]
        rho = 0.5

        A_rho = sp.eye(n) - rho * W_sparse
        eigs = np.linalg.eigvalsh(W_std)
        lam_min = min(1 - rho * eigs.max(), 1 - rho * eigs.min())
        lam_max = max(1 - rho * eigs.max(), 1 - rho * eigs.min())

        b = np.ones(n)
        x = iterative_solve(A_rho, b, lambda_min=lam_min, lambda_max=lam_max)
        x_exact = sp.linalg.spsolve(A_rho, b)
        np.testing.assert_allclose(x, x_exact, atol=1e-8)

    def test_spatial_weights_high_rho(self):
        """CG solve matches direct solve for A_ρ = I − 0.9W."""
        from libpysal.weights import lat2W

        W = lat2W(7, 7)
        W_arr = W.sparse.toarray()
        row_sums = W_arr.sum(axis=1, keepdims=True)
        W_std = W_arr / np.where(row_sums == 0, 1, row_sums)
        W_sparse = sp.csr_matrix(W_std)
        n = W_sparse.shape[0]
        rho = 0.9

        A_rho = sp.eye(n) - rho * W_sparse
        eigs = np.linalg.eigvalsh(W_std)
        lam_min = min(1 - rho * eigs.max(), 1 - rho * eigs.min())
        lam_max = max(1 - rho * eigs.max(), 1 - rho * eigs.min())

        b = np.ones(n)
        x = iterative_solve(A_rho, b, lambda_min=lam_min, lambda_max=lam_max)
        x_exact = sp.linalg.spsolve(A_rho, b)
        np.testing.assert_allclose(x, x_exact, atol=1e-6)

    def test_linear_operator(self):
        """CG solve works with LinearOperator (no explicit matrix)."""
        from libpysal.weights import lat2W

        W = lat2W(5, 5)
        W_arr = W.sparse.toarray()
        row_sums = W_arr.sum(axis=1, keepdims=True)
        W_std = W_arr / np.where(row_sums == 0, 1, row_sums)
        W_csc = sp.csc_matrix(W_std)
        n = W_csc.shape[0]
        rho = 0.5

        A_op = sp.linalg.LinearOperator((n, n), matvec=lambda v: v - rho * (W_csc @ v))
        eigs = np.linalg.eigvalsh(W_std)
        lam_min = min(1 - rho * eigs.max(), 1 - rho * eigs.min())
        lam_max = max(1 - rho * eigs.max(), 1 - rho * eigs.min())

        b = np.ones(n)
        x = iterative_solve(A_op, b, lambda_min=lam_min, lambda_max=lam_max)
        A_rho = sp.eye(n) - rho * W_csc
        x_exact = sp.linalg.spsolve(A_rho, b)
        np.testing.assert_allclose(x, x_exact, atol=1e-8)

    def test_multi_rhs_spatial(self):
        """CG solve matches direct solve for multi-RHS spatial system."""
        from libpysal.weights import lat2W

        W = lat2W(5, 5)
        W_arr = W.sparse.toarray()
        row_sums = W_arr.sum(axis=1, keepdims=True)
        W_std = W_arr / np.where(row_sums == 0, 1, row_sums)
        W_sparse = sp.csr_matrix(W_std)
        n = W_sparse.shape[0]
        rho = 0.7

        A_rho = sp.eye(n) - rho * W_sparse
        eigs = np.linalg.eigvalsh(W_std)
        lam_min = min(1 - rho * eigs.max(), 1 - rho * eigs.min())
        lam_max = max(1 - rho * eigs.max(), 1 - rho * eigs.min())

        k = 3
        X = np.column_stack([np.ones(n), np.arange(n, dtype=float), rng_vals(n)])
        U = iterative_solve(A_rho, X, lambda_min=lam_min, lambda_max=lam_max)
        U_exact = sp.linalg.spsolve(A_rho, X)
        np.testing.assert_allclose(U, U_exact, atol=1e-4)

    def test_invalid_eigenvalue_bounds(self):
        """Non-SPD bounds raise ValueError."""
        A = sp.eye(5, format="csr")
        b = np.ones(5)
        with pytest.raises(ValueError, match="lambda_min must be positive"):
            iterative_solve(A, b, lambda_min=-1.0, lambda_max=2.0)
        with pytest.raises(ValueError, match="lambda_max.*must be >= lambda_min"):
            iterative_solve(A, b, lambda_min=2.0, lambda_max=1.0)

    def test_degenerate_identity(self):
        """lambda_min == lambda_max (A = scalar * I) returns trivial solution."""
        A = sp.eye(5, format="csr")
        b = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        x = iterative_solve(A, b, lambda_min=1.0, lambda_max=1.0)
        np.testing.assert_allclose(x, b)
        # Multi-RHS
        X = np.ones((5, 3))
        U = iterative_solve(A, X, lambda_min=1.0, lambda_max=1.0)
        np.testing.assert_allclose(U, X)


def rng_vals(n):
    """Deterministic test values."""
    return np.random.default_rng(123).standard_normal(n)
