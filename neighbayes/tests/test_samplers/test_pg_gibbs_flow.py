"""Tests for Kronecker matvec primitives and flow Gibbs samplers.

Smoke tests use small n=5 grids. Recovery tests use n=8.
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from neighbayes.samplers.panel._kronecker import (
    kron_At_matvec,
    kron_eigenvalue_bounds,
    kron_logdet_A,
    kron_matvec,
    kron_P_matvec,
)

# ---------------------------------------------------------------------------
# Kronecker matvec primitives
# ---------------------------------------------------------------------------


class TestKronMatvec:
    """Tests for kron_matvec against brute-force Kronecker product."""

    def setup_method(self):
        rng = np.random.default_rng(42)
        self.n = 4
        self.N = self.n * self.n
        self.Ld = rng.standard_normal((self.n, self.n))
        self.Lo = rng.standard_normal((self.n, self.n))
        self.v = rng.standard_normal(self.N)

    def test_kron_matvec_matches_brute_force(self):
        """kron_matvec(v, Ld, Lo) == (Lo ⊗ Ld) @ v."""
        Lo_kron_Ld = np.kron(self.Lo, self.Ld)
        expected = Lo_kron_Ld @ self.v
        result = kron_matvec(self.v, self.Ld, self.Lo)
        np.testing.assert_allclose(result, expected, atol=1e-10)

    def test_kron_At_matvec_matches_brute_force(self):
        """kron_At_matvec(v, Ld.T, Lo.T) == (Lo^T ⊗ Ld^T) @ v."""
        LoT_kron_LdT = np.kron(self.Lo.T, self.Ld.T)
        expected = LoT_kron_LdT @ self.v
        result = kron_At_matvec(self.v, self.Ld.T, self.Lo.T)
        np.testing.assert_allclose(result, expected, atol=1e-10)

    def test_kron_P_matvec_matches_brute_force(self):
        """kron_P_matvec matches (Ld^T Ld ⊗ Lo^T Lo)/σ² + diag(ω)."""
        rng = np.random.default_rng(123)
        sigma2 = 2.0
        omega = rng.uniform(0.5, 2.0, size=self.N)

        LdtLd = self.Ld.T @ self.Ld
        LotLo = self.Lo.T @ self.Lo
        kron_part = np.kron(LotLo, LdtLd) / sigma2
        P_dense = kron_part + np.diag(omega)

        v = rng.standard_normal(self.N)
        expected = P_dense @ v
        result = kron_P_matvec(v, LdtLd, LotLo, omega, sigma2)
        np.testing.assert_allclose(result, expected, atol=1e-10)

    def test_kron_matvec_with_identity(self):
        """kron_matvec with Lo=I gives Ld ⊗ I applied to v."""
        I_n = np.eye(self.n)
        result = kron_matvec(self.v, self.Ld, I_n)
        expected = np.kron(I_n, self.Ld) @ self.v
        np.testing.assert_allclose(result, expected, atol=1e-10)


class TestKronLogdetA:
    """Tests for kron_logdet_A."""

    def test_zero_rho_gives_zero(self):
        """log|A| = 0 when both rho_d and rho_o are zero."""
        n = 5

        def logdet_fn(rho):
            return n * np.log(abs(1 - rho))  # trivial W with single eigenvalue

        result = kron_logdet_A(0.0, 0.0, n, logdet_fn)
        np.testing.assert_allclose(result, 0.0, atol=1e-12)

    def test_matches_eigenvalue(self):
        """kron_logdet_A matches n*log|I-rho_d*W| + n*log|I-rho_o*W|."""
        n = 5
        rng = np.random.default_rng(42)
        W = sp.random(n, n, density=0.3, format="csr", random_state=0, dtype=np.float64)
        row_sums = np.asarray(W.sum(axis=1)).ravel()
        row_sums[row_sums == 0] = 1
        W = sp.diags(1.0 / row_sums) @ W
        eigs = np.linalg.eigvals(W.toarray().astype(np.float64)).real

        def logdet_fn(rho):
            return float(np.sum(np.log(np.maximum(1 - rho * eigs, 1e-300))))

        rho_d, rho_o = 0.3, 0.2
        expected = n * logdet_fn(rho_d) + n * logdet_fn(rho_o)
        result = kron_logdet_A(rho_d, rho_o, n, logdet_fn)
        np.testing.assert_allclose(result, expected, atol=1e-10)


class TestKronEigenvalueBounds:
    """Tests for kron_eigenvalue_bounds."""

    def test_bounds_contain_true_eigenvalues(self):
        """Gershgorin bounds should contain all true eigenvalues of P."""
        n = 5
        rng = np.random.default_rng(42)
        W = sp.random(n, n, density=0.3, format="csr", random_state=0, dtype=np.float64)
        row_sums = np.asarray(W.sum(axis=1)).ravel()
        row_sums[row_sums == 0] = 1
        W = sp.diags(1.0 / row_sums) @ W
        W_dense = W.toarray().astype(np.float64)

        rho_d, rho_o = 0.3, 0.2
        omega_val, sigma2 = 1.5, 2.0

        Ld = np.eye(n) - rho_d * W_dense
        Lo = np.eye(n) - rho_o * W_dense
        LdtLd = Ld.T @ Ld
        LotLo = Lo.T @ Lo
        omega = np.full(n * n, omega_val)

        lam_min, lam_max = kron_eigenvalue_bounds(LdtLd, LotLo, omega, sigma2)

        # Build full P matrix using Kronecker product (separable structure)
        N = n * n
        P_full = np.kron(LdtLd, LotLo) / sigma2 + omega_val * np.eye(N)
        true_eigs = np.linalg.eigvalsh(P_full)
        assert lam_min <= true_eigs.min() + 1e-10
        assert lam_max >= true_eigs.max() - 1e-10

    def test_zero_rho_bounds(self):
        """With zero rho, LdtLd = LotLo = I, so P = I/sigma2 + omega*I."""
        n = 4
        LdtLd = np.eye(n)
        LotLo = np.eye(n)
        omega_val, sigma2 = 1.0, 1.0
        omega = np.full(n * n, omega_val)
        lam_min, lam_max = kron_eigenvalue_bounds(LdtLd, LotLo, omega, sigma2)
        # P = I/sigma2 + omega*I = (1 + omega)*I when rho=0
        expected = 1.0 / sigma2 + omega_val
        assert lam_min <= expected + 1e-10
        assert lam_max >= expected - 1e-10


# ---------------------------------------------------------------------------
# Model-level Gibbs integration tests
# ---------------------------------------------------------------------------


def _make_flow_data(n=5, k=2, seed=42):
    """Generate small NB flow dataset for testing."""
    from neighbayes.dgp import generate_negbin_flow_data

    return generate_negbin_flow_data(
        n=n, k=k, rho_d=0.1, rho_o=0.1, rho_w=0.0, alpha=5.0, seed=seed
    )


class TestNegativeBinomialFlowGibbs:
    """Tests for NegBinFlow.fit(sampler='gibbs')."""

    def test_gibbs_returns_inference_data(self):
        """Gibbs sampler returns valid InferenceData."""
        from neighbayes.models.flow._flow import NegBinFlow

        data = _make_flow_data()
        model = NegBinFlow(data["y_vec"], data["X"], data["G"])
        idata = model.fit(
            draws=20,
            tune=20,
            chains=2,
            sampler="gibbs",
            random_seed=42,
            progressbar=False,
        )
        assert hasattr(idata, "posterior")
        assert "beta" in idata.posterior
        assert "alpha" in idata.posterior

    def test_gibbs_shapes(self):
        """Gibbs posterior has correct shapes."""
        from neighbayes.models.flow._flow import NegBinFlow

        data = _make_flow_data()
        model = NegBinFlow(data["y_vec"], data["X"], data["G"])
        draws, chains = 30, 2
        idata = model.fit(
            draws=draws,
            tune=20,
            chains=chains,
            sampler="gibbs",
            random_seed=42,
            progressbar=False,
        )
        assert idata.posterior["beta"].shape == (
            chains,
            draws,
            len(model._feature_names),
        )
        assert idata.posterior["alpha"].shape == (chains, draws)

    def test_gibbs_alpha_positive(self):
        """NB dispersion alpha should be positive."""
        from neighbayes.models.flow._flow import NegBinFlow

        data = _make_flow_data()
        model = NegBinFlow(data["y_vec"], data["X"], data["G"])
        idata = model.fit(
            draws=30,
            tune=30,
            chains=2,
            sampler="gibbs",
            random_seed=42,
            progressbar=False,
        )
        assert (idata.posterior["alpha"].values > 0).all()

    def test_nuts_still_works(self):
        """NUTS (default sampler) still works after adding Gibbs."""
        from neighbayes.models.flow._flow import NegBinFlow

        data = _make_flow_data()
        model = NegBinFlow(data["y_vec"], data["X"], data["G"])
        idata = model.fit(
            draws=20,
            tune=20,
            chains=2,
            random_seed=42,
            progressbar=False,
        )
        assert "beta" in idata.posterior
        assert "alpha" in idata.posterior


class TestNegativeBinomialSARFlowSeparableGibbs:
    """Tests for SARNegBinFlowSeparable.fit(sampler='gibbs')."""

    def test_gibbs_returns_inference_data(self):
        """Gibbs sampler returns valid InferenceData with spatial params."""
        from neighbayes.models.flow._flow import SARNegBinFlowSeparable

        data = _make_flow_data()
        model = SARNegBinFlowSeparable(
            data["y_vec"],
            data["X"],
            data["G"],
            logdet_method="eigenvalue",
        )
        idata = model.fit(
            draws=20,
            tune=20,
            chains=2,
            sampler="gibbs",
            random_seed=42,
            progressbar=False,
        )
        assert "beta" in idata.posterior
        assert "alpha" in idata.posterior
        assert "rho_d" in idata.posterior
        assert "rho_o" in idata.posterior
        assert "rho_w" in idata.posterior

    def test_gibbs_rho_w_deterministic(self):
        """rho_w = -rho_d * rho_o for separable model."""
        from neighbayes.models.flow._flow import SARNegBinFlowSeparable

        data = _make_flow_data()
        model = SARNegBinFlowSeparable(
            data["y_vec"],
            data["X"],
            data["G"],
            logdet_method="eigenvalue",
        )
        idata = model.fit(
            draws=30,
            tune=30,
            chains=2,
            sampler="gibbs",
            random_seed=42,
            progressbar=False,
        )
        rho_d = idata.posterior["rho_d"].values
        rho_o = idata.posterior["rho_o"].values
        rho_w = idata.posterior["rho_w"].values
        np.testing.assert_allclose(rho_w, -rho_d * rho_o, atol=1e-10)

    def test_gibbs_shapes(self):
        """Gibbs posterior has correct shapes for separable model."""
        from neighbayes.models.flow._flow import SARNegBinFlowSeparable

        data = _make_flow_data()
        model = SARNegBinFlowSeparable(
            data["y_vec"],
            data["X"],
            data["G"],
            logdet_method="eigenvalue",
        )
        draws, chains = 30, 2
        idata = model.fit(
            draws=draws,
            tune=20,
            chains=chains,
            sampler="gibbs",
            random_seed=42,
            progressbar=False,
        )
        k = len(model._feature_names)
        assert idata.posterior["beta"].shape == (chains, draws, k)
        assert idata.posterior["rho_d"].shape == (chains, draws)
        assert idata.posterior["rho_o"].shape == (chains, draws)
        assert idata.posterior["alpha"].shape == (chains, draws)


class TestNegativeBinomialSARFlowGibbs:
    """Tests for SARNegBinFlow.fit(sampler='gibbs')."""

    def test_gibbs_returns_inference_data(self):
        """Gibbs sampler returns valid InferenceData with 3 spatial params."""
        from neighbayes.models.flow._flow import SARNegBinFlow

        data = _make_flow_data()
        model = SARNegBinFlow(data["y_vec"], data["X"], data["G"])
        idata = model.fit(
            draws=20,
            tune=20,
            chains=2,
            sampler="gibbs",
            random_seed=42,
            progressbar=False,
        )
        assert "beta" in idata.posterior
        assert "alpha" in idata.posterior
        assert "rho_d" in idata.posterior
        assert "rho_o" in idata.posterior
        assert "rho_w" in idata.posterior

    def test_gibbs_jax_backend_returns_inference_data(self):
        """gibbs_backend='jax' (sparsax-sparse chain) returns valid InferenceData."""
        import importlib.util

        if importlib.util.find_spec("sparsax") is None:
            pytest.skip("sparsax not installed")
        from neighbayes.models.flow._flow import SARNegBinFlow

        data = _make_flow_data()
        model = SARNegBinFlow(data["y_vec"], data["X"], data["G"])
        idata = model.fit(
            draws=20,
            tune=20,
            chains=2,
            sampler="gibbs",
            gibbs_backend="jax",
            random_seed=42,
            progressbar=False,
            idata_kwargs={"log_likelihood": True},
        )
        for v in ("beta", "alpha", "rho_d", "rho_o", "rho_w"):
            assert v in idata.posterior
            assert np.isfinite(idata.posterior[v].to_numpy()).all()
        assert "log_likelihood" in idata.groups()

    @pytest.mark.requires_jax
    def test_separable_jax_backend_works(self):
        """The separable Kronecker model now supports the JAX backend."""
        import importlib.util

        if importlib.util.find_spec("sparsax") is None:
            pytest.skip("sparsax not installed")
        from neighbayes.models.flow._flow import SARNegBinFlowSeparable

        data = _make_flow_data()
        model = SARNegBinFlowSeparable(data["y_vec"], data["X"], data["G"])
        idata = model.fit(
            draws=10,
            tune=10,
            chains=1,
            sampler="gibbs",
            gibbs_backend="jax",
            progressbar=False,
        )
        assert "rho_d" in idata.posterior
        assert "rho_o" in idata.posterior

    def test_gibbs_shapes(self):
        """Gibbs posterior has correct shapes for unrestricted model."""
        from neighbayes.models.flow._flow import SARNegBinFlow

        data = _make_flow_data()
        model = SARNegBinFlow(data["y_vec"], data["X"], data["G"])
        draws, chains = 30, 2
        idata = model.fit(
            draws=draws,
            tune=20,
            chains=chains,
            sampler="gibbs",
            random_seed=42,
            progressbar=False,
        )
        k = len(model._feature_names)
        assert idata.posterior["beta"].shape == (chains, draws, k)
        assert idata.posterior["rho_d"].shape == (chains, draws)
        assert idata.posterior["rho_o"].shape == (chains, draws)
        assert idata.posterior["rho_w"].shape == (chains, draws)
        assert idata.posterior["alpha"].shape == (chains, draws)

    def test_gibbs_rho_in_bounds(self):
        """Spatial parameters should be within prior bounds."""
        from neighbayes.models.flow._flow import SARNegBinFlow

        data = _make_flow_data()
        model = SARNegBinFlow(data["y_vec"], data["X"], data["G"])
        idata = model.fit(
            draws=30,
            tune=30,
            chains=2,
            sampler="gibbs",
            random_seed=42,
            progressbar=False,
        )
        for param in ["rho_d", "rho_o", "rho_w"]:
            vals = idata.posterior[param].values
            assert vals.min() >= -1.0 - 1e-6, f"{param} below lower bound"
            assert vals.max() <= 1.0 + 1e-6, f"{param} above upper bound"
