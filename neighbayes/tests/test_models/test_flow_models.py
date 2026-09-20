"""Tests for neighbayes.models.flow and supporting logdet flow functions.

Smoke tests use small n=5 grids and minimal draws (100, chains=1) to keep
CI runtime reasonable.  Recovery tests use n=8.
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp

from neighbayes.tests.helpers import SAMPLE_KWARGS

# ---------------------------------------------------------------------------
# logdet flow helper functions
# ---------------------------------------------------------------------------


class TestBarryPaceTraces:
    def test_shape(self):
        from neighbayes._logdet._chebyshev import _barry_pace_traces

        n = 10
        W = sp.random(n, n, density=0.3, format="csr", random_state=0, dtype=np.float64)
        # Row-normalize
        row_sums = np.asarray(W.sum(axis=1)).ravel()
        row_sums[row_sums == 0] = 1
        W = sp.diags(1.0 / row_sums) @ W
        rng = np.random.default_rng(0)
        out = _barry_pace_traces(W, order=5, iter=10, rng=rng)
        assert out.shape == (5, 10)

    def test_exact_tr_W_override(self):
        """tr(W) = 0 for zero-diagonal W must be set in row 0."""
        from neighbayes._logdet._chebyshev import _barry_pace_traces

        n = 8
        W = sp.random(n, n, density=0.4, format="csr", random_state=1, dtype=np.float64)
        W.setdiag(0)
        W.eliminate_zeros()
        row_sums = np.asarray(W.sum(axis=1)).ravel()
        row_sums[row_sums == 0] = 1
        W = sp.diags(1.0 / row_sums) @ W
        rng = np.random.default_rng(2)
        out = _barry_pace_traces(W, order=4, iter=20, rng=rng)
        # All rows 0 should be the exact tr(W) = 0
        np.testing.assert_allclose(out[0, :], 0.0, atol=1e-12)


class TestFlowModelConstruction:
    def setup_method(self):
        from neighbayes.dgp.flows import generate_flow_data

        self.n = 5
        self.N = self.n * self.n
        out = generate_flow_data(
            n=self.n,
            rho_d=0.0,
            rho_o=0.0,
            rho_w=0.0,
            beta_d=[1.0, -0.5],
            beta_o=[0.5, 0.3],
            sigma=1.0,
            seed=10,
        )
        self.G = out["G"]
        self.X_regional = out["X_regional"]
        self.design = out["design"]
        self.X = out["X"]
        self.col_names = out["col_names"]
        rng = np.random.default_rng(10)
        self.y_vec = rng.standard_normal(self.N)
        self.y_mat = self.y_vec.reshape(self.n, self.n)

    def test_sar_flow_builds_from_vec(self):
        from neighbayes.models.flow import SARFlow

        model = SARFlow(
            self.y_vec,
            self.X,
            self.G,
            col_names=self.col_names,
        )
        assert model._n == self.n
        assert model._N == self.N

    def test_sar_flow_builds_from_mat(self):
        from neighbayes.models.flow import SARFlow

        model = SARFlow(
            self.y_mat,
            self.X,
            self.G,
            col_names=self.col_names,
        )
        np.testing.assert_allclose(model._y_vec, self.y_vec, atol=1e-12)

    def test_sar_flow_separable_builds(self):
        from neighbayes.models.flow import SARFlowSeparable

        model = SARFlowSeparable(self.y_vec, self.X, self.G, col_names=self.col_names)
        assert model._n == self.n

    def test_wrong_y_length_raises(self):
        from neighbayes.models.flow import SARFlow

        with pytest.raises(ValueError, match="N="):
            SARFlow(np.zeros(self.N + 1), self.X, self.G)

    def test_asymmetric_beta_shapes_work_in_dgp(self):
        """generate_flow_data should support beta_d and beta_o of different lengths."""
        from neighbayes.dgp.flows import generate_flow_data

        data = generate_flow_data(
            n=self.n,
            rho_d=0.2,
            rho_o=0.2,
            rho_w=0.1,
            beta_d=[1.0],
            beta_o=[1.0, 2.0],
            sigma=1.0,
            seed=0,
        )
        # Design matrix should have k_d=1 dest cols + k_o=2 orig cols + k_d=1 intra cols
        assert data["design"].k_d == 1
        assert data["design"].k_o == 2
        assert data["beta_d"].shape == (1,)
        assert data["beta_o"].shape == (2,)

    def test_pymc_model_raises_not_implemented(self):
        """SARFlow samples via the resolvent sampler; the PyMC path was removed."""
        from neighbayes.models.flow import SARFlow

        model = SARFlow(
            self.y_vec,
            self.X,
            self.G,
            col_names=self.col_names,
        )
        with pytest.raises(NotImplementedError, match="resolvent"):
            model._build_pymc_model()

    def test_pymc_model_separable_builds_without_error(self):
        from neighbayes.models.flow import SARFlowSeparable

        model = SARFlowSeparable(self.y_vec, self.X, self.G, col_names=self.col_names)
        pm_model = model._build_pymc_model()
        assert pm_model is not None


# ---------------------------------------------------------------------------
# DGP
# ---------------------------------------------------------------------------


class TestGenerateFlowData:
    def setup_method(self):
        self.n = 5

    def test_output_keys(self):
        from neighbayes.dgp.flows import generate_flow_data

        out = generate_flow_data(
            n=self.n,
            rho_d=0.2,
            rho_o=0.2,
            rho_w=0.1,
            beta_d=[1.0, -0.5],
            beta_o=[0.5, 0.3],
            sigma=1.0,
            seed=0,
        )
        expected = {
            "y_vec",
            "y_mat",
            "X",
            "X_regional",
            "col_names",
            "design",
            "W",
            "G",
            "rho_d",
            "rho_o",
            "rho_w",
            "sigma",
            "beta_d",
            "beta_o",
        }
        assert expected.issubset(set(out.keys()))

    def test_y_shapes(self):
        from neighbayes.dgp.flows import generate_flow_data

        out = generate_flow_data(
            n=self.n,
            rho_d=0.2,
            rho_o=0.2,
            rho_w=0.1,
            beta_d=[1.0],
            beta_o=[0.5],
            sigma=1.0,
            seed=1,
        )
        assert out["y_vec"].shape == (self.n * self.n,)
        assert out["y_mat"].shape == (self.n, self.n)

    def test_y_vec_mat_consistency(self):
        from neighbayes.dgp.flows import generate_flow_data

        out = generate_flow_data(
            n=self.n,
            rho_d=0.2,
            rho_o=0.2,
            rho_w=0.1,
            beta_d=[1.0],
            beta_o=[0.5],
            sigma=1.0,
            seed=2,
        )
        np.testing.assert_allclose(out["y_vec"], out["y_mat"].ravel(), atol=1e-12)

    def test_reproducibility(self):
        from neighbayes.dgp.flows import generate_flow_data

        out1 = generate_flow_data(
            n=self.n,
            rho_d=0.2,
            rho_o=0.2,
            rho_w=0.1,
            beta_d=[1.0],
            beta_o=[0.5],
            sigma=1.0,
            seed=99,
        )
        out2 = generate_flow_data(
            n=self.n,
            rho_d=0.2,
            rho_o=0.2,
            rho_w=0.1,
            beta_d=[1.0],
            beta_o=[0.5],
            sigma=1.0,
            seed=99,
        )
        np.testing.assert_allclose(out1["y_vec"], out2["y_vec"], atol=1e-12)

    def test_different_seeds_differ(self):
        from neighbayes.dgp.flows import generate_flow_data

        out1 = generate_flow_data(
            n=self.n,
            rho_d=0.2,
            rho_o=0.2,
            rho_w=0.1,
            beta_d=[1.0],
            beta_o=[0.5],
            sigma=1.0,
            seed=1,
        )
        out2 = generate_flow_data(
            n=self.n,
            rho_d=0.2,
            rho_o=0.2,
            rho_w=0.1,
            beta_d=[1.0],
            beta_o=[0.5],
            sigma=1.0,
            seed=2,
        )
        assert not np.allclose(out1["y_vec"], out2["y_vec"])


class TestFlowDistributionDefault:
    """``distribution`` argument: default lognormal, opt-in normal."""

    n = 5
    kwargs = dict(
        rho_d=0.2, rho_o=0.2, rho_w=0.1, beta_d=[1.0], beta_o=[0.5], sigma=1.0, seed=11
    )

    def test_default_is_lognormal_positive(self):
        from neighbayes.dgp.flows import generate_flow_data

        out = generate_flow_data(n=self.n, **self.kwargs)
        assert out["distribution"] == "lognormal"
        assert (out["y_vec"] > 0).all()
        np.testing.assert_allclose(np.log(out["y_vec"]), out["eta_vec"], atol=1e-10)
        np.testing.assert_allclose(out["eta_vec"], out["eta_mat"].ravel(), atol=1e-12)

    def test_normal_optin_matches_eta(self):
        from neighbayes.dgp.flows import generate_flow_data

        out = generate_flow_data(n=self.n, distribution="normal", **self.kwargs)
        assert out["distribution"] == "normal"
        np.testing.assert_array_equal(out["y_vec"], out["eta_vec"])

    def test_invalid_distribution_raises(self):
        from neighbayes.dgp.flows import generate_flow_data

        with pytest.raises(ValueError, match="distribution"):
            generate_flow_data(n=self.n, distribution="poisson", **self.kwargs)

    def test_panel_default_is_lognormal_positive(self):
        from neighbayes.dgp.flows import generate_panel_flow_data

        out = generate_panel_flow_data(
            n=4,
            T=2,
            rho_d=0.2,
            rho_o=0.1,
            rho_w=0.05,
            beta_d=[1.0],
            beta_o=[0.5],
            sigma=0.5,
            sigma_alpha=0.0,
            seed=3,
        )
        assert out["distribution"] == "lognormal"
        assert out["params_true"]["distribution"] == "lognormal"
        assert (out["y"] > 0).all()
        np.testing.assert_allclose(np.log(out["y"]), out["eta"], atol=1e-10)


# ---------------------------------------------------------------------------
# Smoke fit tests (minimal draws — just check posterior shapes)
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestSARFlowFitSmoke:
    """Minimal sampling smoke tests.  Marked slow; skipped in fast CI runs."""

    def setup_method(self):
        from neighbayes.dgp.flows import generate_flow_data

        self.n = 5
        out = generate_flow_data(
            n=self.n,
            rho_d=0.25,
            rho_o=0.25,
            rho_w=0.10,
            beta_d=[1.0, -0.5],
            beta_o=[0.5, 0.3],
            sigma=1.0,
            seed=42,
        )
        self.G = out["G"]
        self.y = out["y_vec"]
        self.X = out["X"]  # (N, p) full O-D design matrix
        self.col_names = out["col_names"]

    def test_sar_flow_fit_posterior_keys(self):
        from neighbayes.models.flow import SARFlow

        model = SARFlow(
            self.y,
            self.X,
            self.G,
            col_names=self.col_names,
            restrict_positive=True,
        )
        idata = model.fit(draws=50, tune=50, chains=1, progressbar=False, random_seed=0)
        posterior = idata.posterior
        assert "rho_d" in posterior
        assert "rho_o" in posterior
        assert "rho_w" in posterior
        assert "beta" in posterior
        assert "sigma" in posterior

    def test_sar_flow_separable_fit_posterior_keys(self):
        from neighbayes.models.flow import SARFlowSeparable

        model = SARFlowSeparable(self.y, self.X, self.G, col_names=self.col_names)
        idata = model.fit(draws=50, tune=50, chains=1, progressbar=False, random_seed=0)
        posterior = idata.posterior
        assert "rho_d" in posterior
        assert "rho_o" in posterior
        assert "rho_w" in posterior

    def test_sar_flow_rho_d_o_w_in_unit_interval(self):
        """With Dirichlet prior, rho_d, rho_o, rho_w should stay in [0, 1]."""
        from neighbayes.models.flow import SARFlow

        model = SARFlow(
            self.y,
            self.X,
            self.G,
            col_names=self.col_names,
            restrict_positive=True,
        )
        idata = model.fit(draws=50, tune=50, chains=1, progressbar=False, random_seed=0)
        rho_d = idata.posterior["rho_d"].values.ravel()
        rho_o = idata.posterior["rho_o"].values.ravel()
        rho_w = idata.posterior["rho_w"].values.ravel()
        assert (rho_d >= 0).all() and (rho_d <= 1).all()
        assert (rho_o >= 0).all() and (rho_o <= 1).all()
        assert (rho_w >= 0).all() and (rho_w <= 1).all()

    def test_sar_flow_inference_data_property(self):
        from neighbayes.models.flow import SARFlow

        model = SARFlow(
            self.y,
            self.X,
            self.G,
            col_names=self.col_names,
        )
        assert model.inference_data is None  # before fit
        model.fit(draws=30, tune=30, chains=1, progressbar=False, random_seed=0)
        assert model.inference_data is not None


# ---------------------------------------------------------------------------
# Effects accounting identity (small, fast)
# ---------------------------------------------------------------------------


class TestEffectsAccountingIdentity:
    """total = origin + destination + intra + network (per draw, per variable)."""

    def test_effects_sum_to_total(self):
        from neighbayes.dgp.flows import generate_flow_data
        from neighbayes.models.flow import SARFlowSeparable

        n = 4
        out = generate_flow_data(
            n=n,
            rho_d=0.3,
            rho_o=0.2,
            rho_w=-0.06,
            beta_d=[1.0],
            beta_o=[0.5],
            sigma=0.5,
            seed=7,
        )
        G = out["G"]
        model = SARFlowSeparable(out["y_vec"], out["X"], G, col_names=out["col_names"])
        # Use minimal posterior (mock-style: just run prior predictive)
        model.fit(draws=20, tune=20, chains=1, progressbar=False, random_seed=0)
        effects = model._compute_spatial_effects_posterior(draws=5)

        total = effects["total"]
        check = (
            effects["origin"]
            + effects["destination"]
            + effects["intra"]
            + effects["network"]
        )
        np.testing.assert_allclose(total, check, atol=1e-8)


# ---------------------------------------------------------------------------
# SparseFlowSolveOp unit tests
# ---------------------------------------------------------------------------


class TestSparseFlowSolveOp:
    """Forward pass matches scipy.sparse.linalg.spsolve; gradients are finite."""

    @pytest.fixture(autouse=True)
    def _build_op(self):
        from neighbayes._ops import SparseFlowSolveOp
        from neighbayes.dgp.flows import generate_flow_data

        n = 4
        G = generate_flow_data(n=n, seed=0)["G"]
        W = G.sparse.astype(np.float64).tocsr()
        # Row-standardize
        row_sums = np.asarray(W.sum(axis=1)).ravel()
        W = sp.diags(1.0 / row_sums) @ W

        I_n = sp.eye(n, format="csr")
        I_N = sp.eye(n * n, format="csr")
        Wd = sp.kron(I_n, W, format="csr")
        Wo = sp.kron(W, I_n, format="csr")
        Ww = sp.kron(W, W, format="csr")

        self.n = n
        self.Wd = Wd
        self.Wo = Wo
        self.Ww = Ww
        self.I_N = I_N
        self.op = SparseFlowSolveOp(Wd, Wo, Ww)

    def test_forward_matches_scipy(self):
        """Op.perform output equals spsolve(A, b)."""
        import pytensor
        import pytensor.tensor as pt

        rho_d = pt.dscalar("rho_d")
        rho_o = pt.dscalar("rho_o")
        rho_w = pt.dscalar("rho_w")
        b = pt.dvector("b")
        eta = self.op(rho_d, rho_o, rho_w, b)
        fn = pytensor.function([rho_d, rho_o, rho_w, b], eta)

        rd, ro, rw = 0.2, 0.15, 0.05
        b_val = np.arange(1, self.n**2 + 1, dtype=np.float64)
        A = self.I_N - rd * self.Wd - ro * self.Wo - rw * self.Ww
        expected = sp.linalg.spsolve(A, b_val)

        result = fn(rd, ro, rw, b_val)
        np.testing.assert_allclose(result, expected, rtol=1e-10)

    def test_gradient_rho_d_finite(self):
        """Gradient w.r.t. rho_d is finite and non-zero."""
        import pytensor
        import pytensor.tensor as pt

        rho_d = pt.dscalar("rho_d")
        rho_o = pt.dscalar("rho_o")
        rho_w = pt.dscalar("rho_w")
        b = pt.dvector("b")
        eta = self.op(rho_d, rho_o, rho_w, b)
        loss = eta.sum()
        g_rd = pt.grad(loss, rho_d)
        fn = pytensor.function([rho_d, rho_o, rho_w, b], g_rd)

        b_val = np.ones(self.n**2, dtype=np.float64)
        result = fn(0.2, 0.15, 0.05, b_val)
        assert np.isfinite(result)
        assert result != 0.0

    def test_gradient_b_matches_adjoint(self):
        """Gradient w.r.t. b equals (A^T)^{-1} ones."""
        import pytensor
        import pytensor.tensor as pt

        rho_d = pt.dscalar("rho_d")
        rho_o = pt.dscalar("rho_o")
        rho_w = pt.dscalar("rho_w")
        b = pt.dvector("b")
        eta = self.op(rho_d, rho_o, rho_w, b)
        loss = eta.sum()
        g_b = pt.grad(loss, b)
        fn = pytensor.function([rho_d, rho_o, rho_w, b], g_b)

        rd, ro, rw = 0.2, 0.15, 0.05
        b_val = np.ones(self.n**2, dtype=np.float64)
        # grad_b = (A^T)^{-1} g,  g = ones (because sum)
        A = self.I_N - rd * self.Wd - ro * self.Wo - rw * self.Ww
        expected = sp.linalg.spsolve(A.T.tocsr(), b_val)
        result = fn(rd, ro, rw, b_val)
        np.testing.assert_allclose(result, expected, rtol=1e-8)

    def test_distinct_op_instances_have_different_ids(self):
        """Two separate SparseFlowSolveOp instances get distinct _op_id values."""
        from neighbayes._ops import SparseFlowSolveOp

        op2 = SparseFlowSolveOp(self.Wd, self.Wo, self.Ww)
        assert self.op._op_id != op2._op_id


# ---------------------------------------------------------------------------
# Parameter recovery tests (slow — deselected by default)
# ---------------------------------------------------------------------------

# Flow-specific dimensions and true parameter values
FLOW_N = 8  # 64 O-D pairs — enough for reliable identification
RHO_D_TRUE = 0.25
RHO_O_TRUE = 0.25
RHO_W_TRUE = 0.10
BETA_D_TRUE = np.array([1.0, -0.5])
BETA_O_TRUE = np.array([0.5, 0.3])
SIGMA_TRUE = 1.0

# Separable-model true values — asymmetric so rho_d ≠ rho_o (breaks swap
# symmetry); rho_w = -0.4*0.3 = -0.12 provides clear identification signal
RHO_D_SEP_TRUE = 0.40
RHO_O_SEP_TRUE = 0.30
# Larger grid for separable: the bilinear rho_w=-rho_d*rho_o term creates a
# harder posterior surface — more obs improves identification reliably
FLOW_N_SEP = 12  # 144 O-D pairs (vs 64 for unrestricted model)

# Tolerances — wider than standard panel models because 3 spatial
# parameters are harder to identify simultaneously
ABS_TOL_RHO = 0.20
ABS_TOL_RHO_SEP = 0.25  # separable model: slightly wider
ABS_TOL_BETA = 0.35
ABS_TOL_BETA_SEP = 0.40
ABS_TOL_SIGMA = 0.35


# ---------------------------------------------------------------------------
# Smoke tests: SARFlowSeparable with alternative logdet methods
# ---------------------------------------------------------------------------


class TestSARFlowSeparableLogdetMethods:
    """SARFlowSeparable fits without error for each logdet method."""

    @pytest.fixture(params=["eigenvalue", "chebyshev"], scope="class")
    def fitted_model(self, request):
        from neighbayes.dgp.flows import generate_flow_data
        from neighbayes.models.flow import SARFlowSeparable

        out = generate_flow_data(
            n=5,
            rho_d=0.3,
            rho_o=0.2,
            rho_w=-0.06,
            beta_d=np.array([0.5, -0.3]),
            beta_o=np.array([0.4, 0.2]),
            sigma=1.0,
            seed=0,
        )
        G = out["G"]
        model = SARFlowSeparable(
            out["y_vec"],
            out["X"],
            G,
            col_names=out["col_names"],
            logdet_method=request.param,
        )
        idata = model.fit(**{**SAMPLE_KWARGS, "draws": 50, "tune": 50, "chains": 1})
        return idata, request.param

    def test_posterior_has_rho_d(self, fitted_model):
        idata, _ = fitted_model
        assert "rho_d" in idata.posterior

    def test_posterior_has_rho_o(self, fitted_model):
        idata, _ = fitted_model
        assert "rho_o" in idata.posterior

    def test_rho_w_is_deterministic(self, fitted_model):
        idata, _ = fitted_model
        assert "rho_w" in idata.posterior

    def test_beta_shape(self, fitted_model):
        idata, _ = fitted_model
        assert idata.posterior["beta"].shape[-1] > 0

    def test_invalid_method_raises(self):
        import pytest

        from neighbayes.dgp.flows import generate_flow_data
        from neighbayes.models.flow import SARFlowSeparable

        out = generate_flow_data(
            n=4,
            rho_d=0.2,
            rho_o=0.1,
            rho_w=-0.02,
            beta_d=np.array([1.0]),
            beta_o=np.array([0.5]),
            sigma=1.0,
            seed=1,
        )
        G = out["G"]
        with pytest.raises(ValueError, match="logdet_method"):
            SARFlowSeparable(
                out["y_vec"],
                out["X"],
                G,
                col_names=out["col_names"],
                logdet_method="traces",
            )


@pytest.mark.slow
class TestSARFlowRecovery:
    """SARFlow posterior means should be close to true DGP values."""

    @pytest.fixture(scope="class")
    def fitted_sar_flow(self):
        from neighbayes.dgp.flows import generate_flow_data
        from neighbayes.models.flow import SARFlow

        out = generate_flow_data(
            n=FLOW_N,
            rho_d=RHO_D_TRUE,
            rho_o=RHO_O_TRUE,
            rho_w=RHO_W_TRUE,
            beta_d=BETA_D_TRUE,
            beta_o=BETA_O_TRUE,
            sigma=SIGMA_TRUE,
            seed=42,
        )
        G = out["G"]
        # Default DGP is lognormal; SARFlow has Gaussian likelihood, so fit on
        # the latent scale.  np.log(y_vec) == eta_vec by construction.
        model = SARFlow(
            np.log(out["y_vec"]),
            out["X"],
            G,
            col_names=out["col_names"],
            restrict_positive=True,
        )
        idata = model.fit(**SAMPLE_KWARGS)
        return idata

    def test_sar_flow_recovers_rho_d(self, fitted_sar_flow):
        rho_hat = float(fitted_sar_flow.posterior["rho_d"].mean())
        assert abs(rho_hat - RHO_D_TRUE) < ABS_TOL_RHO, (
            f"SARFlow rho_d: expected ≈{RHO_D_TRUE}, got {rho_hat:.3f}"
        )

    def test_sar_flow_recovers_rho_o(self, fitted_sar_flow):
        rho_hat = float(fitted_sar_flow.posterior["rho_o"].mean())
        assert abs(rho_hat - RHO_O_TRUE) < ABS_TOL_RHO, (
            f"SARFlow rho_o: expected ≈{RHO_O_TRUE}, got {rho_hat:.3f}"
        )

    def test_sar_flow_recovers_rho_w(self, fitted_sar_flow):
        rho_hat = float(fitted_sar_flow.posterior["rho_w"].mean())
        assert abs(rho_hat - RHO_W_TRUE) < ABS_TOL_RHO, (
            f"SARFlow rho_w: expected ≈{RHO_W_TRUE}, got {rho_hat:.3f}"
        )

    def test_sar_flow_recovers_sigma(self, fitted_sar_flow):
        sigma_hat = float(fitted_sar_flow.posterior["sigma"].mean())
        assert abs(sigma_hat - SIGMA_TRUE) < ABS_TOL_SIGMA, (
            f"SARFlow sigma: expected ≈{SIGMA_TRUE}, got {sigma_hat:.3f}"
        )


@pytest.mark.slow
class TestSARFlowSeparableRecovery:
    """SARFlowSeparable posterior means should be close to true DGP values."""

    @pytest.fixture(scope="class")
    def fitted_separable(self):
        from neighbayes.dgp.flows import generate_flow_data_separable
        from neighbayes.models.flow import SARFlowSeparable

        out = generate_flow_data_separable(
            n=FLOW_N_SEP,
            rho_d=RHO_D_SEP_TRUE,
            rho_o=RHO_O_SEP_TRUE,
            beta_d=BETA_D_TRUE,
            beta_o=BETA_O_TRUE,
            sigma=SIGMA_TRUE,
            seed=42,
        )
        G = out["G"]
        # Default DGP is lognormal; fit on the latent (log) scale.
        model = SARFlowSeparable(
            np.log(out["y_vec"]),
            out["X"],
            G,
            col_names=out["col_names"],
        )
        idata = model.fit(**SAMPLE_KWARGS)
        return idata

    def test_separable_recovers_rho_d(self, fitted_separable):
        rho_hat = float(fitted_separable.posterior["rho_d"].mean())
        assert abs(rho_hat - RHO_D_SEP_TRUE) < ABS_TOL_RHO_SEP, (
            f"SARFlowSeparable rho_d: expected ≈{RHO_D_SEP_TRUE}, got {rho_hat:.3f}"
        )

    def test_separable_recovers_rho_o(self, fitted_separable):
        rho_hat = float(fitted_separable.posterior["rho_o"].mean())
        assert abs(rho_hat - RHO_O_SEP_TRUE) < ABS_TOL_RHO_SEP, (
            f"SARFlowSeparable rho_o: expected ≈{RHO_O_SEP_TRUE}, got {rho_hat:.3f}"
        )


# ---------------------------------------------------------------------------
# kron_solve_vec / kron_solve_matrix utilities
# ---------------------------------------------------------------------------


class TestKronSolveUtilities:
    """Verify kron_solve_vec and kron_solve_matrix match scipy reference."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        from neighbayes._ops import kron_solve_matrix, kron_solve_vec

        self.kron_solve_vec = kron_solve_vec
        self.kron_solve_matrix = kron_solve_matrix

        rng = np.random.default_rng(42)
        self.n = 6
        N = self.n**2
        W_dense = rng.random((self.n, self.n))
        W_dense /= W_dense.sum(axis=1, keepdims=True)
        self.W = sp.csr_matrix(W_dense)
        I_n = sp.eye(self.n, format="csr", dtype=np.float64)
        rd, ro = 0.3, 0.4
        self.Ld = (I_n - rd * self.W).tocsr()
        self.Lo = (I_n - ro * self.W).tocsr()
        # Full N×N system matrix for reference
        Wd = sp.kron(sp.eye(self.n), self.W, format="csr")
        Wo = sp.kron(self.W, sp.eye(self.n), format="csr")
        Ww = sp.kron(self.W, self.W, format="csr")
        self.A = sp.eye(N, format="csr") - rd * Wd - ro * Wo + rd * ro * Ww
        self.b = rng.random(N)
        self.B = rng.random((N, 4))

    def test_kron_solve_vec_matches_scipy(self):
        eta_ref = sp.linalg.spsolve(self.A, self.b)
        eta_kron = self.kron_solve_vec(self.Lo, self.Ld, self.b, self.n)
        np.testing.assert_allclose(eta_kron, eta_ref, atol=1e-10)

    def test_kron_solve_matrix_matches_scipy(self):
        H_ref = sp.linalg.spsolve(self.A, self.B)
        H_kron = self.kron_solve_matrix(self.Lo, self.Ld, self.B, self.n)
        np.testing.assert_allclose(H_kron, H_ref, atol=1e-10)

    def test_kron_solve_matrix_single_column_matches_vec(self):
        b = self.b
        h_vec = self.kron_solve_vec(self.Lo, self.Ld, b, self.n)
        h_mat = self.kron_solve_matrix(self.Lo, self.Ld, b[:, np.newaxis], self.n)
        np.testing.assert_allclose(h_mat[:, 0], h_vec, atol=1e-12)


# ---------------------------------------------------------------------------
# Public spatial_effects + posterior_predictive (Phase 3.3)
# ---------------------------------------------------------------------------


class TestFlowSpatialEffectsAndPPC:
    @pytest.fixture(scope="class")
    def fitted(self):
        from neighbayes.dgp.flows import generate_flow_data
        from neighbayes.models.flow import SARFlow

        n = 5
        data = generate_flow_data(
            n=n,
            rho_d=0.2,
            rho_o=0.15,
            rho_w=0.1,
            beta_d=[1.0, -0.5],
            beta_o=[0.5, 0.2],
            sigma=1.0,
            seed=11,
        )
        G = data["G"]
        model = SARFlow(
            data["y_vec"],
            data["X"],
            data["G"],
            col_names=data["col_names"],
        )
        model.fit(draws=40, tune=40, chains=1, progressbar=False, random_seed=0)
        return model

    def test_spatial_effects_returns_dataframe(self, fitted):
        df = fitted.spatial_effects()
        assert df.index.names == ["predictor", "side", "effect"]
        assert {"mean", "ci_lower", "ci_upper", "bayes_pvalue"} <= set(df.columns)
        # Five effect labels per predictor.
        effects = set(df.index.get_level_values("effect"))
        assert effects == {"origin", "destination", "intra", "network", "total"}

    def test_spatial_effects_return_samples(self, fitted):
        df, samples = fitted.spatial_effects(return_posterior_samples=True)
        for key in ["origin", "destination", "intra", "network", "total"]:
            arr = samples[key]
            assert arr.ndim == 2
            assert arr.shape[1] == fitted._k

    def test_posterior_predictive_shape(self, fitted):
        y_rep = fitted.posterior_predictive(n_draws=10, random_seed=3)
        assert y_rep.shape == (10, fitted._N)
        assert np.all(np.isfinite(y_rep))

    def test_posterior_predictive_reproducible(self, fitted):
        a = fitted.posterior_predictive(n_draws=5, random_seed=7)
        b = fitted.posterior_predictive(n_draws=5, random_seed=7)
        np.testing.assert_allclose(a, b)


# ---------------------------------------------------------------------------
# LeSage-decomposition correctness tests for the new spatial_effects math
# ---------------------------------------------------------------------------


class TestFlowEffectsLeSageDecomposition:
    """Verify spatial_effects matches the LeSage (2008) calc_effects.m reference."""

    def test_helper_zero_rho_closed_form(self):
        """With A = I, effects collapse to closed-form expressions."""
        from neighbayes.models.flow import (
            _build_flow_effect_masks,
            _compute_flow_effects_lesage,
        )

        n, k = 4, 2
        dmask, omask, imask = _build_flow_effect_masks(n)
        beta_d = np.array([2.0, -1.0])
        beta_o = np.array([0.5, 3.0])
        res = _compute_flow_effects_lesage(
            lambda rhs: rhs,
            dmask,
            omask,
            imask,
            beta_d,
            beta_o,
            n,
            k,
        )
        for p in range(k):
            bd, bo = beta_d[p], beta_o[p]
            assert np.isclose(res["total"][p], bd + bo)
            assert np.isclose(res["intra"][p], (bd + bo) / n)
            assert np.isclose(res["origin"][p], (n - 1) / n * bo)
            assert np.isclose(res["destination"][p], (n - 1) / n * bd)
            assert np.isclose(res["network"][p], 0.0)

    def test_helper_total_identity(self):
        """total == origin + destination + intra + network for arbitrary A."""
        import scipy.sparse as sp_local

        from neighbayes.models.flow import (
            _build_flow_effect_masks,
            _compute_flow_effects_lesage,
        )

        n, k = 4, 2
        N = n * n
        dmask, omask, imask = _build_flow_effect_masks(n)
        # Random invertible A.
        rng = np.random.default_rng(0)
        M = rng.standard_normal((N, N))
        A = np.eye(N) + 0.05 * M
        lu = sp_local.linalg.splu(sp_local.csc_matrix(A))

        def solve(rhs):
            return lu.solve(rhs)

        beta_d = np.array([2.0, -1.0])
        beta_o = np.array([0.5, 3.0])
        res = _compute_flow_effects_lesage(
            solve,
            dmask,
            omask,
            imask,
            beta_d,
            beta_o,
            n,
            k,
        )
        for p in range(k):
            s = (
                res["origin"][p]
                + res["destination"][p]
                + res["intra"][p]
                + res["network"][p]
            )
            assert np.isclose(res["total"][p], s)

    def test_calc_effects_reference_match(self):
        """End-to-end sanity: spatial_effects matches a hand-rolled LeSage solve at posterior means."""
        import scipy.sparse as sp_local

        from neighbayes.dgp.flows import generate_flow_data
        from neighbayes.models.flow import SARFlow, _build_flow_effect_masks

        n = 4
        data = generate_flow_data(
            n=n,
            rho_d=0.2,
            rho_o=0.15,
            rho_w=0.1,
            beta_d=[1.0],
            beta_o=[0.5],
            sigma=1.0,
            seed=3,
        )
        G = data["G"]
        model = SARFlow(
            data["y_vec"],
            data["X"],
            data["G"],
            col_names=data["col_names"],
        )
        model.fit(draws=20, tune=20, chains=1, progressbar=False, random_seed=0)

        post = model.spatial_effects(return_posterior_samples=True)
        df, samples = post
        # Reference computation: rebuild for one draw and compare.
        rho_d = float(model.inference_data.posterior["rho_d"].values.reshape(-1)[0])
        rho_o = float(model.inference_data.posterior["rho_o"].values.reshape(-1)[0])
        rho_w = float(model.inference_data.posterior["rho_w"].values.reshape(-1)[0])
        beta = model.inference_data.posterior["beta"].values.reshape(
            -1, len(model._feature_names)
        )[0]
        k = model._k
        beta_d = beta[2 : 2 + k]
        beta_o = beta[2 + k : 2 + 2 * k]
        # Posterior β_intra (DGP value is 0, but the fitted draw is generally nonzero).
        beta_intra = beta[2 + 2 * k : 2 + 3 * k]

        N = n * n
        dmask, omask, imask = _build_flow_effect_masks(n)
        Wd = sp_local.kron(sp_local.eye(n), model._W_sparse, format="csr")
        Wo = sp_local.kron(model._W_sparse, sp_local.eye(n), format="csr")
        Ww = sp_local.kron(model._W_sparse, model._W_sparse, format="csr")
        A = (sp_local.eye(N) - rho_d * Wd - rho_o * Wo - rho_w * Ww).tocsc()
        lu = sp_local.linalg.splu(A)

        for p in range(k):
            shock = np.zeros((N, n))
            shock[dmask] = beta_d[p]
            shock[omask] = beta_o[p]
            # X_intra = intra_indicator * X_dest, so shocking X_dest also shocks
            # the intra cell by β_intra (Thomas-Agnan & LeSage 2014, §83.5).
            shock[imask] = beta_d[p] + beta_o[p] + beta_intra[p]
            T_resp = lu.solve(shock)
            ref_total = T_resp.sum() / N
            ref_intra = T_resp[imask].sum() / N
            ref_origin = T_resp[omask].sum() / N
            ref_dest = T_resp[dmask].sum() / N
            ref_network = ref_total - ref_origin - ref_dest - ref_intra

            assert np.isclose(samples["total"][0, p], ref_total)
            assert np.isclose(samples["intra"][0, p], ref_intra)
            assert np.isclose(samples["origin"][0, p], ref_origin)
            assert np.isclose(samples["destination"][0, p], ref_dest)
            assert np.isclose(samples["network"][0, p], ref_network)

    def test_separable_matches_unrestricted(self):
        """SARFlowSeparable effects ≈ SARFlow effects when rho_w = -rho_d * rho_o."""
        import scipy.sparse as sp_local

        from neighbayes.dgp.flows import generate_flow_data
        from neighbayes.models.flow import (
            SARFlow,
            SARFlowSeparable,
            _build_flow_effect_masks,
            _compute_flow_effects_lesage,
        )

        n = 4
        data = generate_flow_data(
            n=n,
            rho_d=0.2,
            rho_o=0.15,
            rho_w=-0.2 * 0.15,
            beta_d=[1.0],
            beta_o=[0.5],
            sigma=1.0,
            seed=4,
        )
        G = data["G"]
        # Compute reference effects directly with both code paths at the same parameter point.
        from neighbayes._ops import kron_solve_matrix

        rd, ro = 0.2, 0.15
        rw = -rd * ro
        dmask, omask, imask = _build_flow_effect_masks(n)
        N = n * n
        Wd = sp_local.kron(sp_local.eye(n), G.sparse.tocsr(), format="csr")
        Wo = sp_local.kron(G.sparse.tocsr(), sp_local.eye(n), format="csr")
        Ww = sp_local.kron(G.sparse.tocsr(), G.sparse.tocsr(), format="csr")
        A = (sp_local.eye(N) - rd * Wd - ro * Wo - rw * Ww).tocsc()
        lu = sp_local.linalg.splu(A)

        I_n = sp_local.eye(n, format="csr")
        Ld = (I_n - rd * G.sparse.tocsr()).tocsr()
        Lo = (I_n - ro * G.sparse.tocsr()).tocsr()

        beta_d = np.array([1.0])
        beta_o = np.array([0.5])

        res_general = _compute_flow_effects_lesage(
            lambda r: lu.solve(r),
            dmask,
            omask,
            imask,
            beta_d,
            beta_o,
            n,
            1,
        )
        res_kron = _compute_flow_effects_lesage(
            lambda r: kron_solve_matrix(Lo, Ld, r, n),
            dmask,
            omask,
            imask,
            beta_d,
            beta_o,
            n,
            1,
        )
        for key in res_general:
            np.testing.assert_allclose(res_general[key], res_kron[key], atol=1e-10)


class TestFlowPanelSpatialEffectsAndPPC:
    """Smoke + correctness tests for the new FlowPanelModel public API."""

    @pytest.fixture(scope="class")
    def fitted_panel(self):
        from neighbayes.dgp.flows import generate_flow_data
        from neighbayes.graph import flow_design_matrix
        from neighbayes.models.flow_panel import SARFlowPanel

        n = 4
        T = 3
        G = generate_flow_data(n=n, seed=0)["G"]
        rng = np.random.default_rng(5)
        y_list, X_list = [], []
        col_names = None
        for _ in range(T):
            Xr = rng.standard_normal((n, 1))
            d = flow_design_matrix(Xr)
            if col_names is None:
                col_names = d.feature_names
            X_list.append(d.combined)
            y_list.append(rng.standard_normal(n * n))
        y = np.concatenate(y_list)
        X = np.vstack(X_list)
        model = SARFlowPanel(
            y=y,
            W=G,
            X=X,
            T=T,
            col_names=col_names,
        )
        model.fit(draws=30, tune=30, chains=1, progressbar=False, random_seed=0)
        return model

    def test_panel_spatial_effects_dataframe(self, fitted_panel):
        df = fitted_panel.spatial_effects()
        assert df.index.names == ["predictor", "side", "effect"]
        effects = set(df.index.get_level_values("effect"))
        assert effects == {"origin", "destination", "intra", "network", "total"}

    def test_panel_total_identity(self, fitted_panel):
        df, samples = fitted_panel.spatial_effects(return_posterior_samples=True)
        s = (
            samples["origin"]
            + samples["destination"]
            + samples["intra"]
            + samples["network"]
        )
        np.testing.assert_allclose(samples["total"], s, atol=1e-10)

    def test_panel_posterior_predictive(self, fitted_panel):
        y_rep = fitted_panel.posterior_predictive(n_draws=5, random_seed=11)
        assert y_rep.shape == (5, fitted_panel._N_flow * fitted_panel._T)
        assert np.all(np.isfinite(y_rep))


# ---------------------------------------------------------------------------
# Phase 4: Thomas-Agnan & LeSage (2014) enhancements
# ---------------------------------------------------------------------------


class TestFlowEffectsAsymmetricAndIntra:
    """Tests for the new asymmetric Xo/Xd shocks and beta_intra contribution."""

    def test_helper_intra_shock_includes_beta_intra(self):
        """β_intra adds (β_intra)/n to the intra effect on the dest side under A=I."""
        from neighbayes.models.flow import (
            _build_flow_effect_masks,
            _compute_flow_effects_lesage,
        )

        n, k = 4, 1
        dmask, omask, imask = _build_flow_effect_masks(n)
        beta_d = np.array([1.0])
        beta_o = np.array([0.5])
        beta_intra = np.array([2.0])

        res_with = _compute_flow_effects_lesage(
            lambda rhs: rhs,
            dmask,
            omask,
            imask,
            beta_d,
            beta_o,
            n,
            k,
            beta_intra=beta_intra,
        )
        res_without = _compute_flow_effects_lesage(
            lambda rhs: rhs,
            dmask,
            omask,
            imask,
            beta_d,
            beta_o,
            n,
            k,
            beta_intra=None,
        )
        # Intra increases by β_intra/n (single perturbed region averaged).
        assert np.isclose(
            res_with["intra"][0] - res_without["intra"][0],
            beta_intra[0] / n,
        )
        # Total increases by β_intra/n as well.
        assert np.isclose(
            res_with["total"][0] - res_without["total"][0],
            beta_intra[0] / n,
        )
        # Network unchanged (intra is part of the "owner" region's column).
        assert np.isclose(res_with["network"][0], res_without["network"][0])

    def test_helper_per_side_keys_decompose_combined(self):
        """combined_eff == dest_eff + orig_eff for every effect type."""
        from neighbayes.models.flow import (
            _EFFECT_KEYS,
            _build_flow_effect_masks,
            _compute_flow_effects_lesage,
        )

        n, k = 5, 2
        dmask, omask, imask = _build_flow_effect_masks(n)
        rng = np.random.default_rng(7)
        beta_d = rng.standard_normal(k)
        beta_o = rng.standard_normal(k)
        beta_intra = rng.standard_normal(k)

        res = _compute_flow_effects_lesage(
            lambda rhs: rhs,
            dmask,
            omask,
            imask,
            beta_d,
            beta_o,
            n,
            k,
            beta_intra=beta_intra,
        )
        for eff in _EFFECT_KEYS:
            np.testing.assert_allclose(
                res[eff], res[f"dest_{eff}"] + res[f"orig_{eff}"], atol=1e-12
            )

    def test_asymmetric_design_separate_mode(self):
        """Asymmetric Xo!=Xd auto-detects to 'separate' mode and reports both sides."""
        from neighbayes.dgp.flows import generate_flow_data
        from neighbayes.graph import flow_design_matrix_with_orig
        from neighbayes.models.flow import SARFlow

        n = 4
        rng = np.random.default_rng(0)
        Xd = rng.standard_normal((n, 1))
        Xo = rng.standard_normal((n, 1))  # different from Xd

        data = generate_flow_data(
            n=n,
            rho_d=0.1,
            rho_o=0.1,
            rho_w=0.05,
            beta_d=[1.0],
            beta_o=[0.5],
            sigma=1.0,
            seed=1,
        )
        G = data["G"]
        # Override design with asymmetric Xo!=Xd
        design = flow_design_matrix_with_orig(Xd, Xo, col_names=["x"])
        model = SARFlow(
            data["y_vec"],
            design.combined,
            G,
            col_names=design.feature_names,
        )
        assert model._symmetric_xo_xd is False
        model.fit(draws=20, tune=20, chains=1, progressbar=False, random_seed=0)
        df = model.spatial_effects()  # auto -> separate
        sides = set(df.index.get_level_values("side"))
        assert sides == {"dest", "orig"}

    def test_combined_mode_matches_dest_plus_orig(self):
        """mode='combined' equals the sum of per-side rows from mode='separate'."""
        from neighbayes.dgp.flows import generate_flow_data
        from neighbayes.models.flow import SARFlow

        n = 4
        data = generate_flow_data(
            n=n,
            rho_d=0.15,
            rho_o=0.1,
            rho_w=0.05,
            beta_d=[1.0],
            beta_o=[0.5],
            sigma=1.0,
            seed=2,
        )
        model = SARFlow(
            data["y_vec"],
            data["X"],
            data["G"],
            col_names=data["col_names"],
        )
        model.fit(draws=20, tune=20, chains=1, progressbar=False, random_seed=0)
        df_combined = model.spatial_effects(mode="combined")
        df_separate = model.spatial_effects(mode="separate")

        for eff in ("origin", "destination", "intra", "network", "total"):
            cmean = df_combined.xs(eff, level="effect")["mean"].iloc[0]
            d = df_separate.xs(("dest", eff), level=("side", "effect"))["mean"].iloc[0]
            o = df_separate.xs(("orig", eff), level=("side", "effect"))["mean"].iloc[0]
            assert np.isclose(cmean, d + o, atol=1e-10)


class TestOLSFlowEffects:
    """Tests for the new non-spatial OLSFlow gravity baseline."""

    def test_olsflow_fits_and_effects_table_83_1(self):
        """OLSFlow analytic effects: TE = βd + βo, NE = 0, IE = (βd+βo+β_intra)/n."""
        from neighbayes.dgp.flows import generate_flow_data
        from neighbayes.models.flow import OLSFlow

        n = 5
        # ρ = 0 in the DGP so the OLS model is correctly specified.
        data = generate_flow_data(
            n=n,
            rho_d=0.0,
            rho_o=0.0,
            rho_w=0.0,
            beta_d=[1.0],
            beta_o=[0.5],
            sigma=0.1,
            seed=0,
        )
        # DGP defaults to lognormal; OLS fit on the latent (log) scale.
        model = OLSFlow(
            np.log(data["y_vec"]), data["X"], data["G"], col_names=data["col_names"]
        )
        model.fit(draws=40, tune=40, chains=1, progressbar=False, random_seed=0)
        df = model.spatial_effects(mode="combined")

        # NE = 0 exactly (closed form).
        assert np.isclose(df.xs("network", level="effect")["mean"].iloc[0], 0.0)
        # TE close to βd + βo (sampling-noise tolerance).
        te = df.xs("total", level="effect")["mean"].iloc[0]
        assert np.isclose(te, 1.0 + 0.5, atol=0.2)
        # IE = (TE + β_intra)/n with β_intra ≈ 0 in the DGP.
        ie = df.xs("intra", level="effect")["mean"].iloc[0]
        assert np.isclose(ie, te / n, atol=0.1)

    def test_olsflow_posterior_predictive_shape(self):
        """OLSFlow posterior_predictive returns (n_draws, N) finite values."""
        from neighbayes.dgp.flows import generate_flow_data
        from neighbayes.models.flow import OLSFlow

        n = 4
        data = generate_flow_data(
            n=n,
            rho_d=0.0,
            rho_o=0.0,
            rho_w=0.0,
            beta_d=[1.0],
            beta_o=[0.5],
            sigma=0.5,
            seed=1,
        )
        model = OLSFlow(
            data["y_vec"], data["X"], data["G"], col_names=data["col_names"]
        )
        model.fit(draws=20, tune=20, chains=1, progressbar=False, random_seed=0)
        y_rep = model.posterior_predictive(n_draws=5, random_seed=3)
        assert y_rep.shape == (5, n * n)
        assert np.all(np.isfinite(y_rep))


# ---------------------------------------------------------------------------
# Pointwise log-likelihood (with Jacobian correction)
# ---------------------------------------------------------------------------


class TestFlowLogLikelihood:
    """Verify ``log_likelihood`` group is attached and usable for model
    comparison via ``az.loo`` / ``az.waic`` / ``az.compare``."""

    def setup_method(self):
        from neighbayes.dgp.flows import (
            generate_flow_data,
            generate_negbin_flow_data,
        )

        self.n = 5
        self.gauss = generate_flow_data(
            n=self.n,
            rho_d=0.20,
            rho_o=0.20,
            rho_w=0.05,
            beta_d=[1.0, -0.5],
            beta_o=[0.5, 0.3],
            sigma=1.0,
            seed=0,
        )
        self.G = self.gauss["G"]
        self.poi = generate_negbin_flow_data(n=self.n, k=2, seed=0)

    def _check_loo(self, idata):
        import arviz as az

        assert hasattr(idata, "log_likelihood")
        assert "obs" in idata.log_likelihood.data_vars
        ll = idata.log_likelihood["obs"].values
        assert ll.shape[2] == self.n * self.n  # N obs
        assert np.isfinite(ll).all()
        loo = az.loo(idata)
        assert np.isfinite(loo.elpd_loo)

    def test_sar_flow_loglik(self):
        from neighbayes.models.flow import SARFlow

        m = SARFlow(
            self.gauss["y_vec"],
            self.gauss["X"],
            self.G,
            col_names=self.gauss["col_names"],
            restrict_positive=True,
        )
        idata = m.fit(
            draws=30,
            tune=30,
            chains=1,
            progressbar=False,
            random_seed=0,
            idata_kwargs={"log_likelihood": True},
        )
        self._check_loo(idata)

    def test_sar_flow_separable_loglik(self):
        from neighbayes.models.flow import SARFlowSeparable

        m = SARFlowSeparable(
            self.gauss["y_vec"],
            self.gauss["X"],
            self.G,
            col_names=self.gauss["col_names"],
        )
        idata = m.fit(
            draws=30,
            tune=30,
            chains=1,
            progressbar=False,
            random_seed=0,
            idata_kwargs={"log_likelihood": True},
        )
        self._check_loo(idata)

    def test_ols_flow_loglik(self):
        from neighbayes.models.flow import OLSFlow

        m = OLSFlow(
            self.gauss["y_vec"],
            self.gauss["X"],
            self.G,
            col_names=self.gauss["col_names"],
        )
        idata = m.fit(
            draws=30,
            tune=30,
            chains=1,
            progressbar=False,
            random_seed=0,
            idata_kwargs={"log_likelihood": True},
        )
        self._check_loo(idata)

    def test_compare_flow_models(self):
        """az.compare runs across SARFlow/SARFlowSeparable/OLSFlow."""
        import arviz as az

        from neighbayes.models.flow import OLSFlow, SARFlow, SARFlowSeparable

        m_sar = SARFlow(
            self.gauss["y_vec"],
            self.gauss["X"],
            self.G,
            col_names=self.gauss["col_names"],
            restrict_positive=True,
        )
        m_sep = SARFlowSeparable(
            self.gauss["y_vec"],
            self.gauss["X"],
            self.G,
            col_names=self.gauss["col_names"],
        )
        m_ols = OLSFlow(
            self.gauss["y_vec"],
            self.gauss["X"],
            self.G,
            col_names=self.gauss["col_names"],
        )
        kw = dict(
            draws=40,
            tune=40,
            chains=1,
            progressbar=False,
            random_seed=0,
            idata_kwargs={"log_likelihood": True},
        )
        idata_sar = m_sar.fit(**kw)
        idata_sep = m_sep.fit(**kw)
        idata_ols = m_ols.fit(**kw)
        comp = az.compare({"sar": idata_sar, "sep": idata_sep, "ols": idata_ols})
        assert "rank" in comp.columns
        assert len(comp) == 3
