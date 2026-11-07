"""The separable flow design's classification into rank-one and full-rank columns.

The separable flow samplers never form ``U = A⁻¹X`` (``n² × k``) by ``N × N``
solves: rank-one design columns stay rank one and full-rank columns are
``n × n`` two-sided solves.  That rests on the classification reproducing every
column from its factors, which these tests pin.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from neighbayes.graph import flow_design_matrix_with_orig
from neighbayes.samplers.negbin_reduced._flow_structured import classify_flow_design
from neighbayes.tests.helpers import make_rook_W

N_SIDE = 3  # n = 9 regions, N = 81 flows


def _design(T=1, seed=0):
    """A time-first flow design: attributes vary by period, log-distance does not."""
    rng = np.random.default_rng(seed)
    n = sp.csr_matrix(make_rook_W(N_SIDE)).shape[0]
    xy = np.column_stack(np.divmod(np.arange(n), N_SIDE)).astype(float)
    dist = np.sqrt(((xy[:, None] - xy[None, :]) ** 2).sum(-1))
    blocks = [
        flow_design_matrix_with_orig(
            rng.standard_normal((n, 2)),
            rng.standard_normal((n, 2)),
            col_names=["a", "b"],
            dist=dist,
            log_distance=True,
        ).combined
        for _ in range(T)
    ]
    return n, np.vstack([np.asarray(b, dtype=float) for b in blocks])


def test_design_classification():
    n, X = _design()
    s = classify_flow_design(X, n)
    assert len(s.cheap_cols) + len(s.full_cols) == X.shape[1]
    # intercept + 2 destination + 2 origin attributes are rank one
    assert len(s.cheap_cols) == 5
    # every rank-one column is reproduced by its factors
    for j, c in enumerate(s.cheap_cols):
        Y = np.outer(s.U[:, s.u_idx[0, j]], s.V[:, s.v_idx[0, j]])
        np.testing.assert_array_equal(Y.ravel(), X[:, c])


def test_panel_classification_dedups_time_invariant_columns():
    T = 3
    n, X = _design(T)
    s = classify_flow_design(X, n, T)
    assert s.T == T
    # log-distance and the intra dummy repeat every period: stored once each
    assert len(s.F) < T * len(s.full_cols)
    N = n * n
    for t in range(T):
        for j, c in enumerate(s.cheap_cols):
            Y = np.outer(s.U[:, s.u_idx[t, j]], s.V[:, s.v_idx[t, j]])
            np.testing.assert_array_equal(Y.ravel(), X[t * N : (t + 1) * N, c])
        for j, c in enumerate(s.full_cols):
            np.testing.assert_array_equal(
                s.F[s.f_idx[t, j]].ravel(), X[t * N : (t + 1) * N, c]
            )
