"""Tests for Bayesian spatial effects (impact measures) inference.

Verifies that posterior-draw-level computation of direct, indirect, and total
effects matches the existing posterior-mean computation, and that the
DataFrame returned by spatial_effects() is correctly constructed.
"""

from __future__ import annotations

import warnings

import arviz as az
import numpy as np
import pandas as pd
import pytest

from neighbayes.diagnostics.spatial_effects import (
    _build_effects_dataframe,
    _compute_bayesian_pvalue,
    _compute_ci,
)
from neighbayes.models import SAR, SDEM, SDM, SEM, SLX

# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _make_W(n: int = 5) -> np.ndarray:
    """Create a simple row-standardized rook weights matrix."""
    W = np.zeros((n, n))
    for i in range(n):
        if i > 0:
            W[i, i - 1] = 1
        if i < n - 1:
            W[i, i + 1] = 1
    # Row-standardize
    row_sums = W.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    W = W / row_sums
    return W


def _W_to_graph(W: np.ndarray):
    """Convert a numpy weights matrix to a libpysal Graph."""
    from libpysal.graph import Graph
    from scipy.sparse import csr_matrix

    return Graph.from_sparse(csr_matrix(W))


def _W_to_graph_raw(W: np.ndarray):
    """Convert a numpy weights matrix to a Graph without setting transform metadata."""
    from libpysal.graph import Graph

    rows, cols = np.nonzero(W)
    vals = W[rows, cols]
    return Graph.from_arrays(rows, cols, vals.astype(float))


from neighbayes.tests.helpers import (
    set_posterior_means as _set_posterior_means,  # noqa: E402
)


def _set_posterior_draws(
    model, beta_draws: np.ndarray, rho_draws: np.ndarray | None = None
) -> None:
    """Inject posterior draws into a model for testing.

    Parameters
    ----------
    beta_draws : np.ndarray, shape (chains, draws, k) or (draws, k)
    rho_draws : np.ndarray, shape (chains, draws) or (draws,), optional
    """
    if beta_draws.ndim == 2:
        # Add chain dimension
        beta_draws = beta_draws[np.newaxis, :, :]
    posterior: dict[str, np.ndarray] = {
        "beta": beta_draws,
    }
    if rho_draws is not None:
        if rho_draws.ndim == 1:
            rho_draws = rho_draws[np.newaxis, :]
        posterior["rho"] = rho_draws
    model._idata = az.from_dict({"posterior": posterior})


# ------------------------------------------------------------------
# Tests for helper functions
# ------------------------------------------------------------------


class TestComputeBayesianPvalue:
    def test_all_positive(self):
        samples = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        pval = _compute_bayesian_pvalue(samples)
        assert pval.shape == (1,)
        # All positive => P(effect < 0) = 0 => p = 2 * 0 = 0
        assert pval[0] == pytest.approx(0.0)

    def test_all_negative(self):
        samples = np.array([-1.0, -2.0, -3.0, -4.0, -5.0])
        pval = _compute_bayesian_pvalue(samples)
        # All negative => P(effect > 0) = 0 => p = 0
        assert pval[0] == pytest.approx(0.0)

    def test_symmetric_around_zero(self):
        samples = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
        pval = _compute_bayesian_pvalue(samples)
        # P(>0) = 2/5, P(<0) = 2/5 => p = 2 * 2/5 = 4/5
        assert pval[0] == pytest.approx(0.8)

    def test_multidimensional(self):
        samples = np.array([[1.0, -1.0], [2.0, -2.0], [3.0, 0.5]])
        pval = _compute_bayesian_pvalue(samples)
        assert pval.shape == (2,)
        assert pval[0] == pytest.approx(0.0)  # all positive
        # Second column: P(>0) = 1/3, P(<0) = 2/3 => p = 2 * 1/3
        assert pval[1] == pytest.approx(2.0 / 3.0)


class TestComputeCI:
    def test_1d(self):
        samples = np.arange(100, dtype=float)
        ci = _compute_ci(samples)
        assert len(ci) == 1
        # 2.5th percentile of [0..99] is 2.475, 97.5th is 96.525
        assert ci[0][0] == pytest.approx(2.475, abs=0.01)
        assert ci[0][1] == pytest.approx(96.525, abs=0.01)

    def test_2d(self):
        samples = np.column_stack(
            [np.arange(100, dtype=float), np.arange(100, 200, dtype=float)]
        )
        ci = _compute_ci(samples)
        assert len(ci) == 2


# ------------------------------------------------------------------
# Tests for SAR spatial effects posterior
# ------------------------------------------------------------------


class TestSARSpatialEffectsPosterior:
    def test_accepts_numeric_row_standardized_graph_without_transform(self):
        n = 5
        W = _make_W(n)
        x1 = np.linspace(-1.0, 1.0, n)
        X = np.column_stack([np.ones(n), x1])
        y = np.zeros(n)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model = SAR(y=y, X=X, W=_W_to_graph_raw(W))

        assert model._is_row_std is True
        assert getattr(model._W_sparse, "shape", None) == (n, n)
        assert len(caught) == 0

        np.random.seed(42)
        G = 50
        beta_draws = np.random.randn(G, 2)
        rho_draws = np.random.uniform(-0.5, 0.5, G)
        _set_posterior_draws(model, beta_draws, rho_draws)

        result = model.spatial_effects()
        assert isinstance(result, pd.DataFrame)
        assert np.all(np.isfinite(result[["direct", "indirect", "total"]].to_numpy()))

    def test_posterior_mean_matches_existing(self):
        """Posterior mean of draw-level effects should match existing point estimate."""
        n = 5
        W = _make_W(n)
        x1 = np.linspace(-1.0, 1.0, n)
        X = np.column_stack([np.ones(n), x1])
        y = np.zeros(n)
        model = SAR(y=y, X=X, W=_W_to_graph(W))

        # Set up posterior draws
        np.random.seed(42)
        G = 100
        beta_draws = np.random.randn(G, 2)
        rho_draws = np.random.uniform(-0.5, 0.5, G)
        _set_posterior_draws(model, beta_draws, rho_draws)

        # Get posterior effects
        direct_samples, indirect_samples, total_samples = (
            model._compute_spatial_effects_posterior()
        )

        # Verify shapes (intercept excluded from effects)
        assert direct_samples.shape == (G, 1)
        assert indirect_samples.shape == (G, 1)
        assert total_samples.shape == (G, 1)

        # Verify that indirect = total - direct
        np.testing.assert_allclose(indirect_samples, total_samples - direct_samples)

    def test_spatial_effects_result(self):
        """Test that spatial_effects() returns a DataFrame."""
        n = 5
        W = _make_W(n)
        x1 = np.linspace(-1.0, 1.0, n)
        X = np.column_stack([np.ones(n), x1])
        y = np.zeros(n)
        model = SAR(y=y, X=X, W=_W_to_graph(W))

        np.random.seed(42)
        G = 100
        beta_draws = np.random.randn(G, 2)
        rho_draws = np.random.uniform(-0.5, 0.5, G)
        _set_posterior_draws(model, beta_draws, rho_draws)

        result = model.spatial_effects()
        assert isinstance(result, pd.DataFrame)
        assert result.attrs["model_type"] == "SAR"
        assert result.attrs["n_draws"] == G
        # SAR excludes intercept from effects
        assert len(result.index) == 1
        # Check expected columns
        expected_cols = {
            "direct",
            "direct_ci_lower",
            "direct_ci_upper",
            "direct_pvalue",
            "indirect",
            "indirect_ci_lower",
            "indirect_ci_upper",
            "indirect_pvalue",
            "total",
            "total_ci_lower",
            "total_ci_upper",
            "total_pvalue",
        }
        assert set(result.columns) == expected_cols
        # P-values should be in [0, 1]
        assert np.all(result["direct_pvalue"].values >= 0)
        assert np.all(result["direct_pvalue"].values <= 1)
        assert np.all(result["indirect_pvalue"].values >= 0)
        assert np.all(result["indirect_pvalue"].values <= 1)

    def test_return_posterior_samples(self):
        """Test that spatial_effects(return_posterior_samples=True) returns DataFrame and dict."""
        n = 5
        W = _make_W(n)
        x1 = np.linspace(-1.0, 1.0, n)
        X = np.column_stack([np.ones(n), x1])
        y = np.zeros(n)
        model = SAR(y=y, X=X, W=_W_to_graph(W))

        np.random.seed(42)
        G = 100
        beta_draws = np.random.randn(G, 2)
        rho_draws = np.random.uniform(-0.5, 0.5, G)
        _set_posterior_draws(model, beta_draws, rho_draws)

        df, samples = model.spatial_effects(return_posterior_samples=True)
        assert isinstance(df, pd.DataFrame)
        assert isinstance(samples, dict)
        assert "direct" in samples
        assert "indirect" in samples
        assert "total" in samples
        assert samples["direct"].shape == (G, 1)
        assert samples["indirect"].shape == (G, 1)
        assert samples["total"].shape == (G, 1)
        # Verify posterior means match DataFrame values
        np.testing.assert_allclose(
            df["direct"].values, np.mean(samples["direct"], axis=0)
        )


# ------------------------------------------------------------------
# Tests for SDM spatial effects posterior
# ------------------------------------------------------------------


class TestSDMSpatialEffectsPosterior:
    def test_posterior_mean_matches_existing(self):
        n = 5
        W = _make_W(n)
        x1 = np.linspace(-1.0, 1.0, n)
        X = np.column_stack([np.ones(n), x1])
        y = np.zeros(n)
        model = SDM(y=y, X=X, W=_W_to_graph(W))

        np.random.seed(42)
        G = 100
        # SDM beta = [intercept, x1, W*x1]
        beta_draws = np.random.randn(G, 3)
        rho_draws = np.random.uniform(-0.5, 0.5, G)
        _set_posterior_draws(model, beta_draws, rho_draws)

        direct_samples, indirect_samples, total_samples = (
            model._compute_spatial_effects_posterior()
        )

        # SDM has k_wx = 1 (only x1 is lagged)
        assert direct_samples.shape == (G, 1)
        assert indirect_samples.shape == (G, 1)
        assert total_samples.shape == (G, 1)

        np.testing.assert_allclose(indirect_samples, total_samples - direct_samples)


# ------------------------------------------------------------------
# Resolvent-based direct effects (no eigendecomposition)
# ------------------------------------------------------------------


def _rook_grid_W(side: int) -> np.ndarray:
    """Row-standardized rook-contiguity W on a side×side lattice."""
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


class TestResolventDirectEffects:
    """The direct-effect traces come from the logdet gradient, not eigenvalues."""

    def test_mean_diag_matches_eigenvalue_reference(self):
        """tr(S)/n and tr(SW)/n via the resolvent match the eigenvalue form."""
        from neighbayes.diagnostics.spatial_effects import _chunked_eig_means

        W = _rook_grid_W(8)  # n=64
        n = W.shape[0]
        X = np.column_stack([np.ones(n), np.linspace(-1, 1, n)])
        y = np.zeros(n)
        rho = np.random.default_rng(0).uniform(0.05, 0.9, size=2000)

        # Force the surrogate (not eigenvalue) logdet method.
        model = SAR(y=y, X=X, W=_W_to_graph(W), logdet_method="cheb_cholesky")
        eigs = model._W_eigs  # reference eigendecomposition

        md_res = model._batch_mean_diag(rho)
        md_eig = _chunked_eig_means(rho, eigs)
        np.testing.assert_allclose(md_res, md_eig, atol=1e-4, rtol=1e-4)

        mdw_res = model._batch_mean_diag_MW(rho)
        mdw_eig = _chunked_eig_means(rho, eigs, weights=eigs)
        np.testing.assert_allclose(mdw_res, mdw_eig, atol=1e-4, rtol=1e-4)

    def test_eigenvalue_identity_is_exact(self):
        """When the method IS eigenvalue, tr(S)/n = 1 − (ρ/n)g holds exactly."""
        from neighbayes.diagnostics.spatial_effects import _chunked_eig_means

        W = _rook_grid_W(6)  # n=36 → eigenvalue method
        n = W.shape[0]
        X = np.column_stack([np.ones(n), np.linspace(-1, 1, n)])
        model = SAR(y=np.zeros(n), X=X, W=_W_to_graph(W), logdet_method="eigenvalue")
        rho = np.linspace(-0.9, 0.9, 50)
        np.testing.assert_allclose(
            model._batch_mean_diag(rho),
            _chunked_eig_means(rho, model._W_eigs),
            atol=1e-12,
        )

    def test_spatial_effects_uses_no_eigendecomposition(self):
        """A row-standardized chol-cheb SAR computes impacts without any eig call."""
        W = _rook_grid_W(8)
        n = W.shape[0]
        X = np.column_stack([np.ones(n), np.linspace(-1, 1, n)])
        model = SAR(y=np.zeros(n), X=X, W=_W_to_graph(W), logdet_method="cheb_cholesky")
        G = 200
        beta_draws = np.random.default_rng(1).standard_normal((G, 2))
        rho_draws = np.random.default_rng(2).uniform(0.0, 0.9, G)
        _set_posterior_draws(model, beta_draws, rho_draws)

        # Poison the dense eigendecomposition — impacts must not touch it.
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


# ------------------------------------------------------------------
# Tests for SLX spatial effects posterior
# ------------------------------------------------------------------


class TestSLXSpatialEffectsPosterior:
    def test_posterior_mean_matches_existing(self):
        n = 5
        W = _make_W(n)
        x1 = np.linspace(-1.0, 1.0, n)
        X = np.column_stack([np.ones(n), x1])
        y = np.zeros(n)
        model = SLX(y=y, X=X, W=_W_to_graph(W))

        np.random.seed(42)
        G = 100
        beta_draws = np.random.randn(G, 3)  # [intercept, x1, W*x1]
        _set_posterior_draws(model, beta_draws)

        direct_samples, indirect_samples, total_samples = (
            model._compute_spatial_effects_posterior()
        )

        assert direct_samples.shape == (G, 1)
        np.testing.assert_allclose(indirect_samples, total_samples - direct_samples)


# ------------------------------------------------------------------
# Tests for SDEM spatial effects posterior
# ------------------------------------------------------------------


class TestSDEMSpatialEffectsPosterior:
    def test_posterior_mean_matches_existing(self):
        n = 5
        W = _make_W(n)
        x1 = np.linspace(-1.0, 1.0, n)
        X = np.column_stack([np.ones(n), x1])
        y = np.zeros(n)
        model = SDEM(y=y, X=X, W=_W_to_graph(W))

        np.random.seed(42)
        G = 100
        beta_draws = np.random.randn(G, 3)  # [intercept, x1, W*x1]
        _set_posterior_draws(model, beta_draws)

        direct_samples, indirect_samples, total_samples = (
            model._compute_spatial_effects_posterior()
        )

        assert direct_samples.shape == (G, 1)
        np.testing.assert_allclose(indirect_samples, total_samples - direct_samples)


# ------------------------------------------------------------------
# Tests for SEM spatial effects posterior
# ------------------------------------------------------------------


class TestSEMSpatialEffectsPosterior:
    def test_posterior_mean_matches_existing(self):
        n = 5
        W = _make_W(n)
        x1 = np.linspace(-1.0, 1.0, n)
        X = np.column_stack([np.ones(n), x1])
        y = np.zeros(n)
        model = SEM(y=y, X=X, W=_W_to_graph(W))

        np.random.seed(42)
        G = 100
        beta_draws = np.random.randn(G, 2)
        _set_posterior_draws(model, beta_draws)

        direct_samples, indirect_samples, total_samples = (
            model._compute_spatial_effects_posterior()
        )

        # SEM excludes intercept from effects
        assert direct_samples.shape == (G, 1)
        # SEM: indirect = 0, total = direct
        np.testing.assert_allclose(indirect_samples, 0.0)
        np.testing.assert_allclose(total_samples, direct_samples)


# ------------------------------------------------------------------
# Tests for _build_effects_dataframe
# ------------------------------------------------------------------


class TestBuildEffectsDataFrame:
    def test_construction(self):
        G = 50
        k = 3
        rng = np.random.default_rng(42)
        direct_samples = rng.normal(size=(G, k))
        indirect_samples = rng.normal(size=(G, k))
        total_samples = direct_samples + indirect_samples
        feature_names = ["x1", "x2", "x3"]

        df = _build_effects_dataframe(
            direct_samples=direct_samples,
            indirect_samples=indirect_samples,
            total_samples=total_samples,
            feature_names=feature_names,
            model_type="SAR",
        )

        assert isinstance(df, pd.DataFrame)
        assert df.attrs["model_type"] == "SAR"
        assert df.attrs["n_draws"] == G
        assert list(df.index) == feature_names
        assert df.index.name == "variable"
        # Check all expected columns
        expected_cols = {
            "direct",
            "direct_ci_lower",
            "direct_ci_upper",
            "direct_pvalue",
            "indirect",
            "indirect_ci_lower",
            "indirect_ci_upper",
            "indirect_pvalue",
            "total",
            "total_ci_lower",
            "total_ci_upper",
            "total_pvalue",
        }
        assert set(df.columns) == expected_cols
        # Check posterior means match
        np.testing.assert_allclose(df["direct"].values, np.mean(direct_samples, axis=0))
        np.testing.assert_allclose(
            df["indirect"].values, np.mean(indirect_samples, axis=0)
        )
        np.testing.assert_allclose(df["total"].values, np.mean(total_samples, axis=0))
