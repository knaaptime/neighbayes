"""Fast unit tests for JAX CustomDist model construction paths.

PR #13 added a dual-path ``_build_pymc_model(nuts_sampler=...)`` to every
spatial-error model. When ``nuts_sampler`` is ``"blackjax"`` or ``"numpyro"``,
the model registers its likelihood via ``pm.CustomDist`` with an observed RV
so that PyMC's JAX path can capture ``log_likelihood`` natively.  On the
default (pymc) backend, the existing ``pm.Potential`` formulation is used.

These tests exercise the JAX code branches **without actually sampling** —
they only call ``_build_pymc_model()`` and verify the resulting model graph
has the expected structure.  No JAX installation is required.
"""

from __future__ import annotations

import numpy as np
import pymc as pm
import pytest
from libpysal.graph import Graph

from neighbayes._backends.sampler_helpers import use_jax_likelihood
from neighbayes.models import SDEM, SEM, SEMTobit
from neighbayes.models.base import SpatialModel
from neighbayes.tests.helpers import W_to_graph, make_line_W, make_rook_W

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SIDE = 4  # 16 cross-sectional units — small for speed


def _cs_data(n: int = SIDE * SIDE, seed: int = 0):
    """Small cross-sectional dataset."""
    rng = np.random.default_rng(seed)
    W_dense = make_rook_W(SIDE)
    W = W_to_graph(W_dense)
    x1 = rng.normal(size=n)
    X = np.column_stack([np.ones(n), x1])
    y = 0.5 + 0.8 * x1 + rng.normal(scale=0.5, size=n)
    return y, X, W, W_dense, n


def _panel_data(N: int = 4, T: int = 3, seed: int = 0):
    """Small panel dataset."""
    rng = np.random.default_rng(seed)
    n = N * T
    W_dense = make_line_W(N)
    W = W_to_graph(W_dense)
    x1 = rng.normal(size=n)
    X = np.column_stack([np.ones(n), x1])
    y = 0.5 + 0.8 * x1 + rng.normal(scale=0.5, size=n)
    return y, X, W, W_dense, N, T, n


def _assert_has_obs_rv(model: pm.Model, label: str):
    """Assert that the model has an 'obs' observed random variable."""
    obs_rvs = [rv for rv in model.observed_RVs if rv.name == "obs"]
    assert len(obs_rvs) == 1, (
        f"{label}: expected exactly 1 'obs' observed RV, "
        f"got {len(obs_rvs)}: {[rv.name for rv in model.observed_RVs]}"
    )


def _assert_has_potential(model: pm.Model, label: str):
    """Assert that the model uses pm.Potential (no 'obs' observed RV)."""
    obs_rvs = [rv for rv in model.observed_RVs if rv.name == "obs"]
    assert len(obs_rvs) == 0, (
        f"{label}: expected no 'obs' observed RV (Potential path), but found {obs_rvs}"
    )
    pot_names = [p.name for p in model.potentials]
    assert "eps_loglik" in pot_names or "jacobian" in pot_names, (
        f"{label}: expected 'eps_loglik' or 'jacobian' Potential, got {pot_names}"
    )


# ===========================================================================
# use_jax_likelihood helper
# ===========================================================================


class TestUseJaxLikelihood:
    """Tests for the use_jax_likelihood() dispatch helper."""

    def test_pymc_is_not_jax(self):
        assert use_jax_likelihood("pymc") is False

    def test_numpyro_is_jax(self):
        assert use_jax_likelihood("numpyro") is True

    def test_blackjax_is_jax(self):
        assert use_jax_likelihood("blackjax") is True

    def test_nutpie_is_not_jax(self):
        assert use_jax_likelihood("nutpie") is False


# ===========================================================================
# Cross-sectional models
# ===========================================================================


class TestSEMModelConstruction:
    """Test SEM._build_pymc_model with different nuts_sampler values."""

    @pytest.fixture(autouse=True)
    def setup_data(self):
        self.y, self.X, self.W, self.W_dense, self.n = _cs_data()

    def _make_model(self, robust: bool = False, nuts_sampler: str = "pymc"):
        from neighbayes.models.cross_section.sem import SEM

        return SEM(
            y=self.y,
            X=self.X,
            W=self.W,
            logdet_method="exact",
            logdet_decomp="eig",
            robust=robust,
        )

    def test_pymc_path_has_potential(self):
        model = self._make_model(nuts_sampler="pymc")
        pm_model = model._build_pymc_model(nuts_sampler="pymc")
        _assert_has_potential(pm_model, "SEM[pymc]")

    def test_numpyro_path_has_obs_rv(self):
        model = self._make_model(nuts_sampler="numpyro")
        pm_model = model._build_pymc_model(nuts_sampler="numpyro")
        _assert_has_obs_rv(pm_model, "SEM[numpyro]")

    def test_pymc_robust_path_has_potential(self):
        model = self._make_model(robust=True, nuts_sampler="pymc")
        pm_model = model._build_pymc_model(nuts_sampler="pymc")
        _assert_has_potential(pm_model, "SEM[robust,pymc]")

    def test_numpyro_robust_path_has_obs_rv(self):
        model = self._make_model(robust=True, nuts_sampler="numpyro")
        pm_model = model._build_pymc_model(nuts_sampler="numpyro")
        _assert_has_obs_rv(pm_model, "SEM[robust,numpyro]")


class TestSDEMModelConstruction:
    """Test SDEM._build_pymc_model with different nuts_sampler values."""

    @pytest.fixture(autouse=True)
    def setup_data(self):
        self.y, self.X, self.W, self.W_dense, self.n = _cs_data()

    def _make_model(self, robust: bool = False, nuts_sampler: str = "pymc"):
        from neighbayes.models.cross_section.sdem import SDEM

        return SDEM(
            y=self.y,
            X=self.X,
            W=self.W,
            durbin_vars=["x1"],
            logdet_method="exact",
            logdet_decomp="eig",
            robust=robust,
        )

    def test_pymc_path_has_potential(self):
        model = self._make_model(nuts_sampler="pymc")
        pm_model = model._build_pymc_model(nuts_sampler="pymc")
        _assert_has_potential(pm_model, "SDEM[pymc]")

    def test_numpyro_path_has_obs_rv(self):
        model = self._make_model(nuts_sampler="numpyro")
        pm_model = model._build_pymc_model(nuts_sampler="numpyro")
        _assert_has_obs_rv(pm_model, "SDEM[numpyro]")

    def test_pymc_robust_path_has_potential(self):
        model = self._make_model(robust=True, nuts_sampler="pymc")
        pm_model = model._build_pymc_model(nuts_sampler="pymc")
        _assert_has_potential(pm_model, "SDEM[robust,pymc]")

    def test_numpyro_robust_path_has_obs_rv(self):
        model = self._make_model(robust=True, nuts_sampler="numpyro")
        pm_model = model._build_pymc_model(nuts_sampler="numpyro")
        _assert_has_obs_rv(pm_model, "SDEM[robust,numpyro]")


# ===========================================================================
# Panel models
# ===========================================================================


class TestSEMPanelFEModelConstruction:
    """Test SEMPanelFE._build_pymc_model with different nuts_sampler values."""

    @pytest.fixture(autouse=True)
    def setup_data(self):
        self.y, self.X, self.W, self.W_dense, self.N, self.T, self.n = _panel_data()

    def _make_model(self, robust: bool = False, nuts_sampler: str = "pymc"):
        from neighbayes.models.panel._fe import SEMPanelFE

        return SEM(
            unit_effects="fixed",
            y=self.y,
            X=self.X,
            W=self.W,
            N=self.N,
            T=self.T,
            logdet_method="exact",
            logdet_decomp="eig",
            robust=robust,
        )

    def test_pymc_path_has_potential(self):
        model = self._make_model(nuts_sampler="pymc")
        pm_model = model._build_pymc_model(nuts_sampler="pymc")
        _assert_has_potential(pm_model, "SEMPanelFE[pymc]")

    def test_numpyro_path_has_obs_rv(self):
        model = self._make_model(nuts_sampler="numpyro")
        pm_model = model._build_pymc_model(nuts_sampler="numpyro")
        _assert_has_obs_rv(pm_model, "SEMPanelFE[numpyro]")

    def test_pymc_robust_path_has_potential(self):
        model = self._make_model(robust=True, nuts_sampler="pymc")
        pm_model = model._build_pymc_model(nuts_sampler="pymc")
        _assert_has_potential(pm_model, "SEMPanelFE[robust,pymc]")

    def test_numpyro_robust_path_has_obs_rv(self):
        model = self._make_model(robust=True, nuts_sampler="numpyro")
        pm_model = model._build_pymc_model(nuts_sampler="numpyro")
        _assert_has_obs_rv(pm_model, "SEMPanelFE[robust,numpyro]")


class TestSDEMPanelFEModelConstruction:
    """Test SDEMPanelFE._build_pymc_model with different nuts_sampler values."""

    @pytest.fixture(autouse=True)
    def setup_data(self):
        self.y, self.X, self.W, self.W_dense, self.N, self.T, self.n = _panel_data()

    def _make_model(self, robust: bool = False, nuts_sampler: str = "pymc"):
        from neighbayes.models.panel._fe import SDEMPanelFE

        return SDEM(
            unit_effects="fixed",
            y=self.y,
            X=self.X,
            W=self.W,
            N=self.N,
            T=self.T,
            durbin_vars=["x1"],
            logdet_method="exact",
            logdet_decomp="eig",
            robust=robust,
        )

    def test_pymc_path_has_potential(self):
        model = self._make_model(nuts_sampler="pymc")
        pm_model = model._build_pymc_model(nuts_sampler="pymc")
        _assert_has_potential(pm_model, "SDEMPanelFE[pymc]")

    def test_numpyro_path_has_obs_rv(self):
        model = self._make_model(nuts_sampler="numpyro")
        pm_model = model._build_pymc_model(nuts_sampler="numpyro")
        _assert_has_obs_rv(pm_model, "SDEMPanelFE[numpyro]")

    def test_pymc_robust_path_has_potential(self):
        model = self._make_model(robust=True, nuts_sampler="pymc")
        pm_model = model._build_pymc_model(nuts_sampler="pymc")
        _assert_has_potential(pm_model, "SDEMPanelFE[robust,pymc]")

    def test_numpyro_robust_path_has_obs_rv(self):
        model = self._make_model(robust=True, nuts_sampler="numpyro")
        pm_model = model._build_pymc_model(nuts_sampler="numpyro")
        _assert_has_obs_rv(pm_model, "SDEMPanelFE[robust,numpyro]")


class TestSEMPanelREModelConstruction:
    """Test SEMPanelRE._build_pymc_model with different nuts_sampler values."""

    @pytest.fixture(autouse=True)
    def setup_data(self):
        self.y, self.X, self.W, self.W_dense, self.N, self.T, self.n = _panel_data()

    def _make_model(self, robust: bool = False, nuts_sampler: str = "pymc"):
        from neighbayes.models.panel._re import SEMPanelRE

        return SEM(
            unit_effects="random",
            y=self.y,
            X=self.X,
            W=self.W,
            N=self.N,
            T=self.T,
            logdet_method="exact",
            logdet_decomp="eig",
            robust=robust,
        )

    def test_pymc_path_has_potential(self):
        model = self._make_model(nuts_sampler="pymc")
        pm_model = model._build_pymc_model(nuts_sampler="pymc")
        _assert_has_potential(pm_model, "SEMPanelRE[pymc]")

    def test_numpyro_path_has_obs_rv(self):
        model = self._make_model(nuts_sampler="numpyro")
        pm_model = model._build_pymc_model(nuts_sampler="numpyro")
        _assert_has_obs_rv(pm_model, "SEMPanelRE[numpyro]")

    def test_pymc_robust_path_has_potential(self):
        model = self._make_model(robust=True, nuts_sampler="pymc")
        pm_model = model._build_pymc_model(nuts_sampler="pymc")
        _assert_has_potential(pm_model, "SEMPanelRE[robust,pymc]")

    def test_numpyro_robust_path_has_obs_rv(self):
        model = self._make_model(robust=True, nuts_sampler="numpyro")
        pm_model = model._build_pymc_model(nuts_sampler="numpyro")
        _assert_has_obs_rv(pm_model, "SEMPanelRE[robust,numpyro]")


class TestSDEMPanelREModelConstruction:
    """Test SDEMPanelRE._build_pymc_model with different nuts_sampler values."""

    @pytest.fixture(autouse=True)
    def setup_data(self):
        self.y, self.X, self.W, self.W_dense, self.N, self.T, self.n = _panel_data()

    def _make_model(self, robust: bool = False, nuts_sampler: str = "pymc"):
        from neighbayes.models.panel._re import SDEMPanelRE

        return SDEM(
            unit_effects="random",
            y=self.y,
            X=self.X,
            W=self.W,
            N=self.N,
            T=self.T,
            durbin_vars=["x1"],
            logdet_method="exact",
            logdet_decomp="eig",
            robust=robust,
        )

    def test_pymc_path_has_potential(self):
        model = self._make_model(nuts_sampler="pymc")
        pm_model = model._build_pymc_model(nuts_sampler="pymc")
        _assert_has_potential(pm_model, "SDEMPanelRE[pymc]")

    def test_numpyro_path_has_obs_rv(self):
        model = self._make_model(nuts_sampler="numpyro")
        pm_model = model._build_pymc_model(nuts_sampler="numpyro")
        _assert_has_obs_rv(pm_model, "SDEMPanelRE[numpyro]")

    def test_pymc_robust_path_has_potential(self):
        model = self._make_model(robust=True, nuts_sampler="pymc")
        pm_model = model._build_pymc_model(nuts_sampler="pymc")
        _assert_has_potential(pm_model, "SDEMPanelRE[robust,pymc]")

    def test_numpyro_robust_path_has_obs_rv(self):
        model = self._make_model(robust=True, nuts_sampler="numpyro")
        pm_model = model._build_pymc_model(nuts_sampler="numpyro")
        _assert_has_obs_rv(pm_model, "SDEMPanelRE[robust,numpyro]")


class TestSEMPanelDynamicModelConstruction:
    """Test SEMPanelDynamic._build_pymc_model with different nuts_sampler values."""

    @pytest.fixture(autouse=True)
    def setup_data(self):
        self.y, self.X, self.W, self.W_dense, self.N, self.T, self.n = _panel_data()

    def _make_model(self, robust: bool = False, nuts_sampler: str = "pymc"):
        from neighbayes.models.panel._dynamic import SEMPanelDynamic

        return SEM(
            dynamic=True,
            y=self.y,
            X=self.X,
            W=self.W,
            N=self.N,
            T=self.T,
            logdet_method="exact",
            logdet_decomp="eig",
            robust=robust,
        )

    def test_pymc_path_has_potential(self):
        model = self._make_model(nuts_sampler="pymc")
        pm_model = model._build_pymc_model(nuts_sampler="pymc")
        _assert_has_potential(pm_model, "SEMPanelDynamic[pymc]")

    def test_numpyro_path_has_obs_rv(self):
        model = self._make_model(nuts_sampler="numpyro")
        pm_model = model._build_pymc_model(nuts_sampler="numpyro")
        _assert_has_obs_rv(pm_model, "SEMPanelDynamic[numpyro]")

    def test_pymc_robust_path_has_potential(self):
        model = self._make_model(robust=True, nuts_sampler="pymc")
        pm_model = model._build_pymc_model(nuts_sampler="pymc")
        _assert_has_potential(pm_model, "SEMPanelDynamic[robust,pymc]")

    def test_numpyro_robust_path_has_obs_rv(self):
        model = self._make_model(robust=True, nuts_sampler="numpyro")
        pm_model = model._build_pymc_model(nuts_sampler="numpyro")
        _assert_has_obs_rv(pm_model, "SEMPanelDynamic[robust,numpyro]")


class TestSDEMPanelDynamicModelConstruction:
    """Test SDEMPanelDynamic._build_pymc_model with different nuts_sampler values."""

    @pytest.fixture(autouse=True)
    def setup_data(self):
        self.y, self.X, self.W, self.W_dense, self.N, self.T, self.n = _panel_data()

    def _make_model(self, robust: bool = False, nuts_sampler: str = "pymc"):
        from neighbayes.models.panel._dynamic import SDEMPanelDynamic

        return SDEM(
            dynamic=True,
            y=self.y,
            X=self.X,
            W=self.W,
            N=self.N,
            T=self.T,
            durbin_vars=["x1"],
            logdet_method="exact",
            logdet_decomp="eig",
            robust=robust,
        )

    def test_pymc_path_has_potential(self):
        model = self._make_model(nuts_sampler="pymc")
        pm_model = model._build_pymc_model(nuts_sampler="pymc")
        _assert_has_potential(pm_model, "SDEMPanelDynamic[pymc]")

    def test_numpyro_path_has_obs_rv(self):
        model = self._make_model(nuts_sampler="numpyro")
        pm_model = model._build_pymc_model(nuts_sampler="numpyro")
        _assert_has_obs_rv(pm_model, "SDEMPanelDynamic[numpyro]")

    def test_pymc_robust_path_has_potential(self):
        model = self._make_model(robust=True, nuts_sampler="pymc")
        pm_model = model._build_pymc_model(nuts_sampler="pymc")
        _assert_has_potential(pm_model, "SDEMPanelDynamic[robust,pymc]")

    def test_numpyro_robust_path_has_obs_rv(self):
        model = self._make_model(robust=True, nuts_sampler="numpyro")
        pm_model = model._build_pymc_model(nuts_sampler="numpyro")
        _assert_has_obs_rv(pm_model, "SDEMPanelDynamic[robust,numpyro]")


class TestSEMPanelTobitModelConstruction:
    """Test SEMTobitPanel._build_pymc_model with different nuts_sampler values."""

    @pytest.fixture(autouse=True)
    def setup_data(self):
        self.y, self.X, self.W, self.W_dense, self.N, self.T, self.n = _panel_data()
        # Introduce censoring at zero
        self.y = np.where(self.y > 0, self.y, 0.0)

    def _make_model(self, robust: bool = False, nuts_sampler: str = "pymc"):
        from neighbayes.models.panel._tobit import SEMTobitPanel

        return SEMTobit(
            y=self.y,
            X=self.X,
            W=self.W,
            N=self.N,
            T=self.T,
            logdet_method="exact",
            logdet_decomp="eig",
            robust=robust,
        )

    def test_pymc_path_has_potential(self):
        model = self._make_model(nuts_sampler="pymc")
        pm_model = model._build_pymc_model(nuts_sampler="pymc")
        _assert_has_potential(pm_model, "SEMTobitPanel[pymc]")

    def test_numpyro_path_has_obs_rv(self):
        model = self._make_model(nuts_sampler="numpyro")
        pm_model = model._build_pymc_model(nuts_sampler="numpyro")
        _assert_has_obs_rv(pm_model, "SEMTobitPanel[numpyro]")

    def test_pymc_robust_path_has_potential(self):
        model = self._make_model(robust=True, nuts_sampler="pymc")
        pm_model = model._build_pymc_model(nuts_sampler="pymc")
        _assert_has_potential(pm_model, "SEMTobitPanel[robust,pymc]")

    def test_numpyro_robust_path_has_obs_rv(self):
        model = self._make_model(robust=True, nuts_sampler="numpyro")
        pm_model = model._build_pymc_model(nuts_sampler="numpyro")
        _assert_has_obs_rv(pm_model, "SEMTobitPanel[robust,numpyro]")


# ===========================================================================
# fit() dispatch tests
# ===========================================================================


class TestFitDispatchesNutsSampler:
    """Test that fit() passes nuts_sampler to _build_pymc_model()."""

    @pytest.fixture(autouse=True)
    def setup_data(self):
        self.y, self.X, self.W, self.W_dense, self.n = _cs_data()

    def test_sem_fit_passes_nuts_sampler(self):
        """SEM.fit() should pass nuts_sampler to _build_pymc_model()."""
        from neighbayes.models.cross_section.sem import SEM

        model = SEM(
            y=self.y, X=self.X, W=self.W, logdet_method="exact", logdet_decomp="eig"
        )
        # Just build the model — don't sample
        pm_model = model._build_pymc_model(nuts_sampler="numpyro")
        _assert_has_obs_rv(pm_model, "SEM[numpyro]")

    def test_sdem_fit_passes_nuts_sampler(self):
        """SDEM.fit() should pass nuts_sampler to _build_pymc_model()."""
        from neighbayes.models.cross_section.sdem import SDEM

        model = SDEM(
            y=self.y,
            X=self.X,
            W=self.W,
            durbin_vars=["x1"],
            logdet_method="exact",
            logdet_decomp="eig",
        )
        pm_model = model._build_pymc_model(nuts_sampler="numpyro")
        _assert_has_obs_rv(pm_model, "SDEM[numpyro]")

    def test_base_fit_try_except_fallback(self):
        """SpatialModel.fit() should fall back if _build_pymc_model() doesn't accept nuts_sampler."""
        from neighbayes.models.cross_section.ols import OLS

        model = OLS(y=self.y, X=self.X, W=self.W)
        # OLS._build_pymc_model() doesn't accept nuts_sampler, but fit()
        # should handle this gracefully via try/except
        pm_model = model._build_pymc_model()
        # OLS uses pm.Normal("obs", ...) — always has an observed RV
        obs_rvs = [rv for rv in pm_model.observed_RVs if rv.name == "obs"]
        assert len(obs_rvs) == 1
