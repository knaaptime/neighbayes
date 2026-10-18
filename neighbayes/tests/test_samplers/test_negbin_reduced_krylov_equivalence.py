"""Equivalence test: Krylov-accelerated vs exact-LU reduced NB sampler.

Both paths must produce statistically indistinguishable posteriors
when run with the same seed on a small problem.
"""

import numpy as np
import pytest

# Skip if scipy.sparse.linalg.splu is unavailable (shouldn't happen)
pytest.importorskip("scipy.sparse.linalg")


@pytest.fixture
def small_nb_problem():
    """Build a tiny SAR-NB problem (n=49, k=3) for fast iteration."""
    from libpysal.weights import lat2W

    W = lat2W(7, 7)
    W.transform = "r"
    W_sparse = W.sparse
    n = W_sparse.shape[0]
    rng = np.random.default_rng(42)

    X = np.column_stack([np.ones(n), rng.standard_normal(n), rng.standard_normal(n)])
    beta_true = np.array([0.5, -0.3, 0.2])
    rho_true = 0.4
    alpha_true = 2.0

    from scipy import sparse as sp
    from scipy.sparse.linalg import splu

    A_sp = sp.eye(n) - rho_true * W_sparse
    eta = splu(A_sp.tocsc()).solve(X @ beta_true)
    mu = np.exp(eta)
    # NegBin draw: y ~ NegBin(mu, alpha)
    p = alpha_true / (alpha_true + mu)
    y = rng.negative_binomial(alpha_true, p)

    return {
        "y": y.astype(np.float64),
        "X": X,
        "W_sparse": W_sparse,
        "rho_true": rho_true,
        "beta_true": beta_true,
        "alpha_true": alpha_true,
    }


def test_krylov_vs_exact_same_posterior(small_nb_problem):
    """Krylov (degree=8) and exact-LU (degree=0) produce same posterior."""
    from neighbayes.models.cross_section.sar_negbin import SARNegBin

    p = small_nb_problem

    # --- Exact-LU path (krylov_degree=0) ---
    model_exact = SARNegBin(
        y=p["y"],
        X=p["X"],
        W=p["W_sparse"],
    )
    idata_exact = model_exact.fit(
        draws=200,
        tune=200,
        chains=1,
        random_seed=123,
        progressbar=False,
        backend="numpy",  # krylov_degree=0 exact-LU is a NumPy-path concept
        krylov_degree=0,
    )

    # --- Krylov path (degree=8) ---
    model_krylov = SARNegBin(
        y=p["y"],
        X=p["X"],
        W=p["W_sparse"],
    )
    idata_krylov = model_krylov.fit(
        draws=200,
        tune=200,
        chains=1,
        random_seed=123,
        progressbar=False,
        backend="numpy",  # compare NumPy exact-LU vs NumPy Krylov
        krylov_degree=8,
        krylov_dmax=0.15,
    )

    # Same seed → same posterior draws (up to floating-point differences
    # from the Krylov approximation).  Assert means agree within MC error.
    for param in ["rho", "alpha"]:
        mean_exact = float(idata_exact.posterior[param].mean())
        mean_krylov = float(idata_krylov.posterior[param].mean())
        std_exact = float(idata_exact.posterior[param].std())
        # Allow 2× MC standard error tolerance
        atol = max(2 * std_exact / np.sqrt(200), 0.05)
        assert abs(mean_exact - mean_krylov) < atol, (
            f"{param}: exact={mean_exact:.4f}, krylov={mean_krylov:.4f}, "
            f"diff={abs(mean_exact - mean_krylov):.4f} > atol={atol:.4f}"
        )

    # Beta: check each coefficient
    beta_exact = idata_exact.posterior["beta"].mean(dim=("chain", "draw")).values
    beta_krylov = idata_krylov.posterior["beta"].mean(dim=("chain", "draw")).values
    beta_std = idata_exact.posterior["beta"].std(dim=("chain", "draw")).values
    for j in range(len(beta_exact)):
        atol_j = max(2 * beta_std[j] / np.sqrt(200), 0.05)
        assert abs(beta_exact[j] - beta_krylov[j]) < atol_j, (
            f"beta[{j}]: exact={beta_exact[j]:.4f}, krylov={beta_krylov[j]:.4f}, "
            f"diff={abs(beta_exact[j] - beta_krylov[j]):.4f} > atol={atol_j:.4f}"
        )


def test_krylov_basis_accuracy():
    """Unit test: Krylov basis reconstruction error < 1e-6 at degree=8."""
    from scipy.sparse.linalg import splu

    from neighbayes.samplers._utils._spatial_normal import CholmodFactor
    from neighbayes.samplers.negbin_reduced._core import (
        ReducedKrylovBasis,
        _build_krylov_basis,
        _CholmodNormalEqSolver,
        _eval_U_from_basis,
        _make_cholmod_pattern,
    )

    rng = np.random.default_rng(0)
    n, k = 200, 3

    # Random sparse W (row-standardized)
    from scipy import sparse

    W_dense = rng.random((n, n)) * (rng.random((n, n)) < 0.05)
    row_sums = W_dense.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    W_dense = W_dense / row_sums
    W_csc = sparse.csc_matrix(W_dense)
    X = rng.standard_normal((n, k))

    # Use CHOLMOD for exact factorization (CG has ~1e-4 absolute error
    # which swamps the Krylov polynomial approximation error we want
    # to test).
    W_sym, WtW, pattern = _make_cholmod_pattern(W_csc, n)
    cholmod_factor = CholmodFactor(pattern)
    cholmod_solver = _CholmodNormalEqSolver(cholmod_factor, W_csc, W_sym, WtW, n)

    rho_c = 0.3
    basis = _build_krylov_basis(
        rho_c, X, W_csc, n, degree=8, cholmod_solver=cholmod_solver
    )

    # Test at several Δρ values within krylov_dmax
    for drho in [0.0, 0.05, 0.1, -0.1, 0.15]:
        rho = rho_c + drho
        U_krylov = _eval_U_from_basis(basis, drho)
        A = (sparse.eye(n, format="csc") - rho * W_csc).tocsc()
        U_exact = splu(A).solve(X)
        max_err = np.max(np.abs(U_krylov - U_exact))
        assert max_err < 1e-6, f"drho={drho}: max_err={max_err:.2e} > 1e-6"
