"""Regression: the JAX Gibbs logdet for a panel must not double-count T.

Bug (fixed): the panel FE Gibbs path passes the ``NT×NT`` block-diagonal lag
matrix (``I_T ⊗ W``) as the sampler's ``W_sparse``.  ``log|I_NT − ρW_NT|``
already equals ``T·log|I_N − ρW|``; ``_build_logdet_jax`` then multiplied by
``T=self.T`` again, yielding ``T²·log`` — an over-strong Jacobian that biased ρ
toward 0 (worsening with T).  This pins the JAX-built logdet to ``T·log|I_N−ρW|``.
"""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest
import scipy.sparse as sp

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("jax") is None
    or importlib.util.find_spec("equinox") is None,
    reason="JAX + equinox required",
)

import jax  # noqa: E402

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

from neighbayes.samplers.gaussian import GaussianGibbsPriors  # noqa: E402
from neighbayes.samplers.gaussian._estimation import GaussianSARGibbs  # noqa: E402


def _rook_W(side: int) -> np.ndarray:
    n = side * side
    A = np.zeros((n, n))
    for r in range(side):
        for c in range(side):
            i = r * side + c
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                rr, cc = r + dr, c + dc
                if 0 <= rr < side and 0 <= cc < side:
                    A[i, rr * side + cc] = 1.0
    return A / A.sum(axis=1, keepdims=True)


def _make_gibbs(W_sparse, N, T, method, W_eigs=None):
    n = N * T
    rng = np.random.default_rng(0)
    X = np.column_stack([np.ones(n), rng.standard_normal(n)])
    y = rng.standard_normal(n)
    priors = GaussianGibbsPriors(
        beta_mu=0.0,
        beta_sigma=10.0,
        sigma2_alpha=2.0,
        sigma2_beta=1.0,
        rho_lower=-1.0,
        rho_upper=1.0,
    )
    return GaussianSARGibbs(
        y=y,
        X=X,
        W_sparse=W_sparse,
        Wy=None,
        priors=priors,
        logdet_fn=lambda r: 0.0,
        logdet_vec_fn=lambda r: np.zeros_like(r),
        feature_names=["x0", "x1"],
        model_type="sar",
        W_eigs=W_eigs,
        logdet_method=method,
        T=T,
    )


def _T_exact(rho, eigs, T):
    return T * float(np.sum(np.log(np.abs(1.0 - rho * eigs))))


@pytest.mark.parametrize("method", ["chol_cheb", "eigenvalue"])
def test_panel_jax_logdet_is_T_times_exact_not_T_squared(method):
    """Panel (T>1) JAX logdet must equal T·log|I_N−ρW|, not T²·log."""
    side, T = 4, 6  # N=16
    N = side * side
    W = _rook_W(side)
    eigs = np.linalg.eigvals(W).real
    W_NT = sp.kron(sp.eye(T), sp.csr_matrix(W), format="csr")  # I_T ⊗ W

    W_eigs = eigs if method == "eigenvalue" else None
    est = _make_gibbs(W_NT, N, T, method, W_eigs=W_eigs)
    ldf = est._build_logdet_jax()

    for rho in (0.2, 0.5, 0.8):
        got = float(ldf(jnp.float64(rho)))
        want = _T_exact(rho, eigs, T)
        # Would be T× too large under the bug.
        np.testing.assert_allclose(got, want, rtol=1e-4, atol=1e-4)


def test_cross_section_logdet_unaffected():
    """T=1 path (W is already N×N) is a no-op for the block extraction."""
    side = 4
    N = side * side
    W = _rook_W(side)
    eigs = np.linalg.eigvals(W).real
    est = _make_gibbs(sp.csr_matrix(W), N, 1, "chol_cheb")
    ldf = est._build_logdet_jax()
    for rho in (0.2, 0.5, 0.8):
        np.testing.assert_allclose(
            float(ldf(jnp.float64(rho))),
            _T_exact(rho, eigs, 1),
            rtol=1e-4,
            atol=1e-4,
        )
