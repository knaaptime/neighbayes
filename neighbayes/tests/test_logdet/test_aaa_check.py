"""The warmup AAA node check: the posterior region, adequacy against the reach, refinement."""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp

from neighbayes._logdet._refit import (
    AAA_CHECK_PAD_SD,
    AAA_NODE_STEP,
    AAA_PILOT_NODES,
    LogdetRefitter,
    aaa_check_region,
)


def _lattice(side: int, queen: bool = False) -> sp.csr_matrix:
    """Row-standardized rook (or queen) contiguity on a side × side lattice."""
    path = sp.diags([np.ones(side - 1), np.ones(side - 1)], [-1, 1])
    eye = sp.eye(side)
    if queen:
        A = sp.kron(path + eye, path + eye) - sp.eye(side * side)
    else:
        A = sp.kron(path, eye) + sp.kron(eye, path)
    A = sp.csr_matrix(A)
    deg = np.asarray(A.sum(axis=1)).ravel()
    return sp.csr_matrix(sp.diags(1.0 / deg) @ A)


@pytest.fixture(scope="module")
def rook():
    return LogdetRefitter(_lattice(30), "chol_aaa")


class TestRegion:
    def test_mean_plus_minus_pad(self):
        draws = np.random.default_rng(0).normal(0.5, 0.01, 400)
        lo, hi = aaa_check_region(draws, -0.99, 0.99)
        sd = draws.std(ddof=1)
        assert lo == pytest.approx(draws.mean() - AAA_CHECK_PAD_SD * sd)
        assert hi == pytest.approx(draws.mean() + AAA_CHECK_PAD_SD * sd)

    def test_clipped_to_the_interval(self):
        draws = np.random.default_rng(1).normal(0.985, 0.01, 400)
        assert aaa_check_region(draws, -0.99, 0.99)[1] == 0.99

    def test_too_few_draws(self):
        assert aaa_check_region(np.full(5, 0.5), -0.99, 0.99) is None


class TestAdequacy:
    def test_pilot_passes_away_from_the_singularity(self, rook):
        pre = rook.aaa_fit(-0.99, 0.99, AAA_PILOT_NODES)
        ok, (right, left) = rook.aaa_adequate(pre, (0.45, 0.55))
        assert ok
        assert 0.0 < right < 0.1 and 0.0 < left < 0.1

    def test_pilot_fails_near_the_singularity_and_refinement_passes(self, rook):
        region = (0.95, 0.96)
        pre = rook.aaa_fit(-0.99, 0.99, AAA_PILOT_NODES)
        assert not rook.aaa_adequate(pre, region)[0]
        pre, nodes, check = rook.aaa_refine(
            pre, AAA_PILOT_NODES, region, -0.99, 0.99, 64
        )
        assert check.passed
        assert (
            nodes > AAA_PILOT_NODES and (nodes - AAA_PILOT_NODES) % AAA_NODE_STEP == 0
        )
        assert check.nodes == nodes and check.support_points == len(pre.support_points)
        assert rook.last_pre is pre and rook.last_nodes == nodes

    def test_refinement_stops_at_the_cap(self, rook):
        region = (0.989, 0.99)
        pre = rook.aaa_fit(-0.99, 0.99, AAA_PILOT_NODES)
        _, nodes, check = rook.aaa_refine(
            pre, AAA_PILOT_NODES, region, -0.99, 0.99, AAA_PILOT_NODES + 4
        )
        assert nodes <= AAA_PILOT_NODES + 4
        assert not check.passed

    def test_negative_side_uses_the_actual_singularity(self):
        # Queen contiguity is not bipartite: its smallest eigenvalue lies well
        # above -1, so its negative singularity 1/λ_min lies well below -1 and a
        # posterior near ρ = -0.9 needs no extra nodes.
        queen = LogdetRefitter(_lattice(30, queen=True), "chol_aaa")
        assert queen.left_singularity().real < -1.2
        pre = queen.aaa_fit(-0.99, 0.99, AAA_PILOT_NODES)
        assert queen.aaa_adequate(pre, (-0.91, -0.89))[0]

    def test_directed_weights(self):
        rng = np.random.default_rng(3)
        n, k = 900, 6
        pts = rng.uniform(size=(n, 2))
        d = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=-1)
        np.fill_diagonal(d, np.inf)
        idx = np.argpartition(d, k, axis=1)[:, :k]
        A = sp.csr_matrix(
            (np.ones(n * k), (np.repeat(np.arange(n), k), idx.ravel())), shape=(n, n)
        )
        W = sp.csr_matrix(sp.diags(1.0 / np.asarray(A.sum(axis=1)).ravel()) @ A)
        knn = LogdetRefitter(W, "aaa")
        pre = knn.aaa_fit(-0.99, 0.99, AAA_PILOT_NODES)
        assert knn.aaa_adequate(pre, (0.4, 0.6))[0]
        assert knn.left_singularity().real <= -1.0
