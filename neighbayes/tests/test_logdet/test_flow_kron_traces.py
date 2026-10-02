"""Exact unrestricted flow log-determinant from n × n trace moments.

``FlowKronTraceLogdet`` evaluates ``log|I − ρ_d(I⊗W) − ρ_o(W⊗I) − ρ_w(W⊗W)|`` and
its three gradients from exact ``n × n`` traces, never forming the ``n² × n²``
system.  These tests pin it against dense ``slogdet`` and against an exact
eigenvalue double sum, for both bases and up to the sampler's stability wall.
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp

from neighbayes._logdet._chol_cheb import _d_symmetrize
from neighbayes._logdet._flow_kron_traces import FlowKronTraceLogdet
from neighbayes.dgp.utils import rook_grid_weights

PARAMS = [
    (0.0, 0.0, 0.0),
    (0.4, 0.3, 0.1),
    (0.4, 0.3, -0.12),
    (-0.3, 0.5, 0.15),
    (0.6, 0.3, 0.09),
]


def _dense(W, rd, ro, rw):
    W = sp.csr_matrix(W)
    n = W.shape[0]
    eye = sp.eye(n)
    Ms = (sp.kron(eye, W), sp.kron(W, eye), sp.kron(W, W))
    A = (sp.eye(n * n) - rd * Ms[0] - ro * Ms[1] - rw * Ms[2]).toarray()
    Ainv = np.linalg.inv(A)
    grad = np.array([-np.sum(M.toarray().T * Ainv) for M in Ms])
    return np.linalg.slogdet(A)[1], grad


def _graphs():
    n = 20
    A = sp.random(n, n, density=0.12, random_state=4, format="csr")
    A = A + sp.diags(np.ones(n - 1), 1) + sp.diags(np.ones(n - 1), -1)
    A.data[:] = 1.0
    U = ((A + A.T) > 0).astype(float)
    return {
        "undirected": sp.diags(1 / np.asarray(U.sum(1)).ravel()) @ U,
        "directed": sp.diags(1 / np.asarray(A.sum(1)).ravel()) @ A,
        "raw": U * (0.95 / np.abs(np.linalg.eigvalsh(U.toarray())).max()),
    }


@pytest.mark.parametrize("params", PARAMS)
@pytest.mark.parametrize("kind", ["undirected", "directed", "raw"])
def test_matches_dense(kind, params):
    W = _graphs()[kind]
    ld = FlowKronTraceLogdet(W)
    assert ld.basis == ("taylor" if kind == "directed" else "chebyshev")
    value, grad = ld(*params)
    v0, g0 = _dense(W, *params)
    np.testing.assert_allclose(value, v0, atol=1e-11)
    np.testing.assert_allclose(grad, g0, atol=1e-11)


@pytest.mark.parametrize(
    "params",
    [(0.5, 0.45, 0.04), (0.2, 0.2, -0.59), (-0.45, -0.45, 0.089), (0.6, 0.3, 0.0989)],
)
def test_near_the_wall_matches_eigen_double_sum(params):
    """Chebyshev basis stays exact up to |ρ| sums of 0.999 on a bipartite grid."""
    W, _ = rook_grid_weights(20)  # spectrum reaches both −1 and 1
    lam = np.linalg.eigvalsh(_d_symmetrize(W).toarray())
    li, lj = lam[:, None], lam[None, :]
    rd, ro, rw = params
    g = 1 - ro * li - rd * lj - rw * li * lj
    v0 = np.log(g).sum()
    g0 = np.array([(-lj / g).sum(), (-li / g).sum(), (-(li * lj) / g).sum()])
    value, grad = FlowKronTraceLogdet(W)(*params)
    np.testing.assert_allclose(value, v0, rtol=1e-12)
    np.testing.assert_allclose(grad, g0, rtol=1e-11)


def test_unstable_parameters_raise():
    W, _ = rook_grid_weights(5)
    with pytest.raises(ValueError, match="stability"):
        FlowKronTraceLogdet(W)(0.6, 0.5, 0.0)
