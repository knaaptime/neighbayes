"""Tests of :class:`CholmodFactor`, the sparse Cholesky of the samplers."""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp

from neighbayes.samplers._utils._spatial_normal import CholmodFactor


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


class TestCholmodFactor:
    """Factorization, sampling, logdet and pickling of :class:`CholmodFactor`."""

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
