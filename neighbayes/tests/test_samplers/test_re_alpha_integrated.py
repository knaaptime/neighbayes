"""The RE Gibbs blocks that integrate α out must match brute-force algebra.

The SEM-RE α precision ``P(λ) = DᵀAᵀAD/σ² + I/σ_α²`` is assembled from sparse
λ-independent pieces on one fixed CHOLMOD pattern, and both RE spatial
densities integrate α out in closed form.  These tests pin each against the
explicit dense computation it replaces, on balanced and unbalanced panels.
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp

from neighbayes.samplers.panel._re_core import (
    _sar_re_rho_quadratic,
    _sem_re_alpha_structure,
    _sem_re_lam_log_density,
)

LAM_GRID = [-0.5, -0.2, 0.0, 0.3, 0.8]


def _row_std_ring(n: int) -> sp.csr_matrix:
    row = np.concatenate([np.arange(n), np.arange(n)])
    col = np.concatenate([(np.arange(n) - 1) % n, (np.arange(n) + 1) % n])
    W = sp.csr_matrix((np.ones(2 * n), (row, col)), shape=(n, n))
    d = np.asarray(W.sum(axis=1)).ravel()
    return (sp.diags(1.0 / d) @ W).tocsr()


def _panel(N: int, T: int, drop: int = 0):
    """Time-major panel; ``drop`` removes the last ``drop`` obs (unbalanced)."""
    W_nt = sp.block_diag([_row_std_ring(N)] * T, format="csr")
    unit_idx = np.tile(np.arange(N), T)
    if drop:
        keep = np.arange(N * T - drop)
        W_nt = W_nt[keep][:, keep]
        unit_idx = unit_idx[keep]
    return W_nt.tocsr(), unit_idx


def _D(unit_idx, N):
    n = len(unit_idx)
    return sp.csr_matrix((np.ones(n), (np.arange(n), unit_idx)), shape=(n, N))


def _logdet_fn(W_nt):
    n = W_nt.shape[0]
    Wd = W_nt.toarray()
    return lambda lam: np.linalg.slogdet(np.eye(n) - lam * Wd)[1]


@pytest.mark.parametrize("N,T,drop", [(6, 4, 0), (10, 3, 0), (7, 4, 3)])
def test_precision_matches_explicit(N, T, drop):
    W_nt, unit_idx = _panel(N, T, drop)
    struct = _sem_re_alpha_structure(W_nt, unit_idx, N)
    D = _D(unit_idx, N).toarray()
    n = len(unit_idx)
    sigma2, sigma_alpha2 = 0.7, 1.9
    for lam in LAM_GRID:
        B = (np.eye(n) - lam * W_nt.toarray()) @ D
        explicit = B.T @ B / sigma2 + np.eye(N) / sigma_alpha2
        got = struct.precision(lam, sigma2, sigma_alpha2)
        # Same structure every time, so CHOLMOD's symbolic analysis holds.
        np.testing.assert_array_equal(got.indices, struct.pattern.indices)
        np.testing.assert_allclose(got.toarray(), explicit, atol=1e-12, rtol=0)


def _marginal_loglik(resid, cov):
    _, logdet = np.linalg.slogdet(cov)
    return -0.5 * logdet - 0.5 * resid @ np.linalg.solve(cov, resid)


@pytest.mark.parametrize("drop", [0, 3])
def test_sem_re_density_matches_brute_force(drop):
    N, T = 7, 4
    W_nt, unit_idx = _panel(N, T, drop)
    n = len(unit_idx)
    rng = np.random.default_rng(0)
    X = np.column_stack([np.ones(n), rng.standard_normal(n)])
    y = rng.standard_normal(n) + unit_idx * 0.1
    beta = np.array([0.2, -0.4])
    sigma2, sigma_alpha2 = 0.8, 1.3
    struct = _sem_re_alpha_structure(W_nt, unit_idx, N)
    factor = struct.new_factor()
    logdet_fn = _logdet_fn(W_nt)
    D = _D(unit_idx, N).toarray()

    got, want = [], []
    for lam in LAM_GRID:
        got.append(
            _sem_re_lam_log_density(
                lam,
                beta,
                sigma2,
                sigma_alpha2,
                y,
                X,
                W_nt,
                logdet_fn,
                N,
                unit_idx,
                struct,
                factor,
            )
        )
        # y = Xβ + Dα + A⁻¹ε  ⇒  A(y − Xβ) ~ N(0, σ²I + σ_α² ADDᵀAᵀ), Jacobian |A|.
        A = np.eye(n) - lam * W_nt.toarray()
        cov = sigma2 * np.eye(n) + sigma_alpha2 * A @ D @ D.T @ A.T
        want.append(logdet_fn(lam) + _marginal_loglik(A @ (y - X @ beta), cov))
    got, want = np.array(got), np.array(want)
    np.testing.assert_allclose(got - got[0], want - want[0], atol=1e-8)


@pytest.mark.parametrize("drop", [0, 3])
def test_sar_re_density_matches_brute_force(drop):
    N, T = 7, 4
    W_nt, unit_idx = _panel(N, T, drop)
    n = len(unit_idx)
    rng = np.random.default_rng(1)
    X = np.column_stack([np.ones(n), rng.standard_normal(n)])
    y = rng.standard_normal(n) + unit_idx * 0.1
    Wy = W_nt @ y
    beta = np.array([0.2, -0.4])
    sigma2, sigma_alpha2 = 0.8, 1.3
    counts = np.bincount(unit_idx, minlength=N).astype(float)
    logdet_fn = _logdet_fn(W_nt)
    D = _D(unit_idx, N).toarray()

    q0, q1, q2 = _sar_re_rho_quadratic(
        beta, sigma2, sigma_alpha2, y, Wy, X, N, unit_idx, counts
    )
    cov = sigma2 * np.eye(n) + sigma_alpha2 * D @ D.T
    got, want = [], []
    for rho in LAM_GRID:
        got.append(logdet_fn(rho) + q0 + q1 * rho + q2 * rho * rho)
        want.append(logdet_fn(rho) + _marginal_loglik(y - X @ beta - rho * Wy, cov))
    got, want = np.array(got), np.array(want)
    np.testing.assert_allclose(got - got[0], want - want[0], atol=1e-8)
