"""The matrix-free flow filter must equal the materialized N² × N² system.

``_FlowFilter`` applies ``I − r_d(I⊗W) − r_o(W⊗I) − r_w(W⊗W)`` through n×n
products and solves it by a Kronecker factorization (separable) or
preconditioned GMRES (unrestricted), so the flow DGPs never build ``W⊗W``.
These tests pin both the operator and the solve against the explicit Kronecker
matrices on small graphs, for undirected (D-symmetrizable) and directed W.
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from libpysal.graph import Graph

from neighbayes.dgp.flows import _FlowFilter
from neighbayes.dgp.utils import synth_point_geodataframe
from neighbayes.tests.helpers import W_to_graph, make_rook_W

PARAMS = [
    (0.4, 0.3, -0.12),  # separable: r_w = −r_d·r_o
    (0.4, 0.3, 0.1),
    (0.5, 0.2, 0.0),
    (-0.3, 0.4, 0.2),
]


def _graphs():
    yield "rook", W_to_graph(make_rook_W(5))
    yield "knn", Graph.build_knn(synth_point_geodataframe(25), k=4).transform("r")


def _materialized(G, r_d, r_o, r_w):
    W = G.sparse.tocsr().astype(float)
    n = W.shape[0]
    eye = sp.eye(n, format="csr")
    return (
        sp.eye(n * n)
        - r_d * sp.kron(eye, W)
        - r_o * sp.kron(W, eye)
        - r_w * sp.kron(W, W)
    ).tocsc()


@pytest.mark.parametrize("params", PARAMS)
@pytest.mark.parametrize("kind", ["rook", "knn"])
def test_operator_and_solve_match_kronecker(kind, params):
    G = dict(_graphs())[kind]
    F = _FlowFilter(G, *params)
    A = _materialized(G, *params)
    v = np.random.default_rng(0).standard_normal(A.shape[0])
    np.testing.assert_allclose(F.matvec(v), A @ v, atol=1e-13)
    np.testing.assert_allclose(F.solve(v), spla.spsolve(A, v), atol=1e-9)
    assert F.separable == (params[2] == -params[0] * params[1])


def test_singular_filter_raises():
    G = W_to_graph(make_rook_W(4))
    F = _FlowFilter(G, 1.0, 0.0, 0.0)  # I − (I⊗W) is singular
    with pytest.raises(ValueError, match="singular"):
        F.solve(np.ones(G.n_nodes**2))
