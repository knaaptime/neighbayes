"""Resolvent-based direct effects for the nonlinear spatial models.

The log-odds / log-mean / count-equation direct-effect trace
``(1/n) tr((I − ρW)⁻¹)`` is the logdet gradient in disguise, so the
SAR-GLMs (logit, neg-bin, structural neg-bin, ZINB) share the same
resolvent route as the Gaussian SAR and must never trigger the O(n³)
eigendecomposition on a row-standardized, surrogate-logdet model.
"""

from __future__ import annotations

import arviz as az
import numpy as np
import pytest
import scipy.sparse as sp
from libpysal.graph import Graph

from neighbayes.diagnostics.spatial_effects import _chunked_eig_means
from neighbayes.models.cross_section.sar_logit import SARLogit
from neighbayes.models.cross_section.sar_negbin import SARNegBin
from neighbayes.models.cross_section.sar_negbin_structural import SARNegBinStructural
from neighbayes.models.cross_section.sar_zinb import SARZINB


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


def _graph(W: np.ndarray) -> Graph:
    return Graph.from_sparse(sp.csr_matrix(W))


@pytest.fixture(scope="module")
def _design():
    W = _rook_W(8)  # n=64
    n = W.shape[0]
    rng = np.random.default_rng(0)
    X = np.column_stack([np.ones(n), np.linspace(-1, 1, n), rng.standard_normal(n)])
    return W, n, X, rng


@pytest.mark.parametrize(
    "model_cls, y_kind",
    [
        (SARLogit, "binary"),
        (SARNegBin, "count"),
        (SARNegBinStructural, "count"),
    ],
)
def test_direct_effect_trace_matches_eigen_reference(model_cls, y_kind, _design):
    """``_batch_mean_diag`` (resolvent) equals the eigenvalue trace form."""
    W, n, X, rng = _design
    y = (
        (rng.uniform(size=n) < 0.5).astype(float)
        if y_kind == "binary"
        else rng.poisson(2.0, size=n).astype(float)
    )
    model = model_cls(y=y, X=X, W=_graph(W), logdet_method="cheb_cholesky")
    rho = rng.uniform(0.05, 0.9, size=1500)

    md_res = model._batch_mean_diag(rho)
    md_eig = _chunked_eig_means(rho, model._W_eigs)
    np.testing.assert_allclose(md_res, md_eig, atol=1e-4, rtol=1e-4)


def test_zinb_selection_direct_effect_same_W(_design):
    """ZINB with shared weights reuses the count-equation resolvent helper."""
    W, n, X, rng = _design
    y = rng.poisson(2.0, size=n).astype(float)
    model = SARZINB(y=y, X=X, W=_graph(W), logdet_method="cheb_cholesky")
    assert model._same_W is True

    lam = rng.uniform(0.05, 0.9, size=1500)
    md_res = model._sel_batch_mean_diag(lam)
    md_eig = _chunked_eig_means(lam, model._W_eigs)
    np.testing.assert_allclose(md_res, md_eig, atol=1e-4, rtol=1e-4)


def test_zinb_selection_direct_effect_distinct_W(_design):
    """A distinct selection W gets its own resolvent evaluator."""
    W, n, X, rng = _design
    y = rng.poisson(2.0, size=n).astype(float)
    W_sel = _rook_W(8)  # same structure, distinct object → _same_W False
    model = SARZINB(
        y=y, X=X, W=_graph(W), W_sel=_graph(W_sel), logdet_method="cheb_cholesky"
    )
    assert model._same_W is False

    lam = rng.uniform(0.05, 0.9, size=1500)
    sel_eigs = np.linalg.eigvals(model._W_sel_sparse.toarray().astype(np.float64))
    md_res = model._sel_batch_mean_diag(lam)
    md_eig = _chunked_eig_means(lam, sel_eigs)
    np.testing.assert_allclose(md_res, md_eig, atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize(
    "model_cls, y_kind", [(SARLogit, "binary"), (SARNegBin, "count")]
)
def test_linear_scale_impacts_use_no_eigendecomposition(model_cls, y_kind, _design):
    """A row-standardized chol-cheb SAR-GLM computes impacts without any eig call."""
    W, n, X, rng = _design
    y = (
        (rng.uniform(size=n) < 0.5).astype(float)
        if y_kind == "binary"
        else rng.poisson(2.0, size=n).astype(float)
    )
    model = model_cls(y=y, X=X, W=_graph(W), logdet_method="cheb_cholesky")

    G = 200
    beta_draws = rng.standard_normal((1, G, X.shape[1]))
    rho_draws = rng.uniform(0.0, 0.9, size=(1, G))
    model._idata = az.from_dict({"posterior": {"beta": beta_draws, "rho": rho_draws}})

    orig_eig, orig_eigvals = np.linalg.eig, np.linalg.eigvals

    def _boom(*a, **k):
        raise AssertionError("eigendecomposition was triggered")

    np.linalg.eig = _boom
    np.linalg.eigvals = _boom
    try:
        result = model.spatial_effects()
    finally:
        np.linalg.eig, np.linalg.eigvals = orig_eig, orig_eigvals

    assert "direct" in result.columns
    assert np.all(np.isfinite(result["direct"].to_numpy()))
