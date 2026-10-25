"""Resolvent-Kronecker gradient of the unrestricted flow log-determinant.

The gradient ``g_k = -tr(W_k (I_N - W_F)^{-1})`` is estimated matvec-only via
Hutchinson probes + Kronecker Krylov solves (eigenvalue-free, directed-W safe).
These tests pin the Kronecker conventions against a dense construction and check
the stochastic gradient against the exact eigenvalue reference.
"""

from __future__ import annotations

import numpy as np
import pytest

from neighbayes._logdet._flow_resolvent import (
    FlowKron,
    flow_logdet_grad,
    flow_logdet_value,
)
from neighbayes.tests.helpers import flow_logdet_grad_exact


def _directed_row_std(n, seed, density=0.06):
    rng = np.random.default_rng(seed)
    A = (rng.uniform(size=(n, n)) < density).astype(float)
    np.fill_diagonal(A, 0.0)
    A[A.sum(1) == 0, 0] = 1.0  # avoid empty rows
    return A / A.sum(1, keepdims=True)


def test_kronecker_matvecs_match_dense():
    """W_F, W_Fᵀ and each W_k match the explicit Kronecker matrices (row-major vec)."""
    rng = np.random.default_rng(1)
    n = 5
    W = rng.uniform(size=(n, n))
    Ide = np.eye(n)
    Wd, Wo, Ww = np.kron(Ide, W), np.kron(W, Ide), np.kron(W, W)
    rd, ro, rw = 0.4, -0.3, 0.15
    WF = rd * Wd + ro * Wo + rw * Ww

    k = FlowKron(W)
    x = rng.uniform(size=n * n)
    np.testing.assert_allclose(k.matvec_WF(x, rd, ro, rw), WF @ x, atol=1e-12)
    np.testing.assert_allclose(k.matvec_WF_T(x, rd, ro, rw), WF.T @ x, atol=1e-12)
    np.testing.assert_allclose(k.matvec_Wd(x), Wd @ x, atol=1e-12)
    np.testing.assert_allclose(k.matvec_Wo(x), Wo @ x, atol=1e-12)
    np.testing.assert_allclose(k.matvec_Ww(x), Ww @ x, atol=1e-12)


def test_exact_gradient_matches_finite_difference():
    """flow_logdet_grad_exact matches a finite difference of the exact logdet."""
    W = _directed_row_std(30, seed=2)
    lam = np.linalg.eigvals(W)

    def logdet(rd, ro, rw):
        mu = ro * lam[:, None] + rd * lam[None, :] + rw * np.outer(lam, lam)
        return np.sum(np.log(np.abs(1.0 - mu)))

    rd, ro, rw = 0.35, 0.25, -0.08
    g = flow_logdet_grad_exact(W, rd, ro, rw)
    h = 1e-6
    fd = np.array(
        [
            (logdet(rd + h, ro, rw) - logdet(rd - h, ro, rw)) / (2 * h),
            (logdet(rd, ro + h, rw) - logdet(rd, ro - h, rw)) / (2 * h),
            (logdet(rd, ro, rw + h) - logdet(rd, ro, rw - h)) / (2 * h),
        ]
    )
    np.testing.assert_allclose(g, fd, rtol=1e-5, atol=1e-4)


def test_stochastic_gradient_matches_exact_directed():
    """The Hutchinson/Krylov gradient recovers the dominant exact components."""
    n = 60
    W = _directed_row_std(n, seed=3)
    rd, ro, rw = 0.4, 0.3, -0.1
    ge = flow_logdet_grad_exact(W, rd, ro, rw)

    kron = FlowKron(W)
    ests = np.array(
        [
            flow_logdet_grad(
                kron, rd, ro, rw, n_probes=64, rng=np.random.default_rng(s)
            )
            for s in range(4)
        ]
    ).mean(0)

    # Dominant components (g_d, g_o) are O(n^2); require them within 15% and
    # correctly signed.  g_w is O(1) here (near zero) so its sign is noise.
    rel = np.abs(ests[:2] - ge[:2]) / np.abs(ge[:2])
    assert np.all(rel < 0.15), (
        f"rel-err {rel} exceeds tolerance (exact {ge}, est {ests})"
    )
    assert np.all(np.sign(ests[:2]) == np.sign(ge[:2]))


def test_value_from_gradient_integration_identity():
    """log|I_N−W_F| = ∫₀¹ ρ·g(tρ) dt — exact given exact gradients (quadrature check)."""
    W = _directed_row_std(40, seed=5)
    lam = np.linalg.eigvals(W)

    def value_exact(rd, ro, rw):
        mu = ro * lam[:, None] + rd * lam[None, :] + rw * np.outer(lam, lam)
        return np.sum(np.log(np.abs(1.0 - mu)))

    nodes, weights = np.polynomial.legendre.leggauss(8)
    nodes = 0.5 * (nodes + 1.0)
    weights = 0.5 * weights
    for rd, ro, rw in [(0.4, 0.3, -0.1), (0.6, 0.2, 0.1), (0.5, 0.5, -0.2)]:
        rho = np.array([rd, ro, rw])
        integrated = sum(
            wq * (rho @ flow_logdet_grad_exact(W, *(t * rho)))
            for t, wq in zip(nodes, weights)
        )
        np.testing.assert_allclose(integrated, value_exact(rd, ro, rw), rtol=1e-4)


def test_stochastic_value_runs_and_is_finite():
    """flow_logdet_value runs end-to-end and returns a finite scalar.

    Its *accuracy* is N-dependent (inherits the gradient's 1/sqrt(N P) law and so
    improves with the flow sample size); the exactness of the underlying
    quadrature is pinned by ``test_value_from_gradient_integration_identity``.  At
    the tiny N used here it is noise-dominated, so only finiteness is asserted.
    """
    W = _directed_row_std(50, seed=6)
    v = flow_logdet_value(
        W, 0.4, 0.3, -0.1, n_probes=48, n_quad=8, rng=np.random.default_rng(0)
    )
    assert np.isfinite(v)


def test_frozen_probes_are_reproducible():
    """Reusing a frozen probe matrix gives a deterministic (consistent) gradient."""
    W = _directed_row_std(40, seed=4)
    kron = FlowKron(W)
    rng = np.random.default_rng(0)
    probes = rng.choice([-1.0, 1.0], size=(kron.N, 12)).astype(float)
    g1, returned = flow_logdet_grad(
        kron, 0.4, 0.3, -0.1, probes=probes, return_probes=True
    )
    g2 = flow_logdet_grad(kron, 0.4, 0.3, -0.1, probes=probes)
    np.testing.assert_array_equal(g1, g2)
    np.testing.assert_array_equal(returned, probes)
