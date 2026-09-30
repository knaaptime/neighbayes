"""Fast build/method tests for SARNegBin NUTS path.

These exercise the reduced-form NUTS model (no σ, no z) that is now
built into SARNegBin via _build_pymc_model().
"""

from __future__ import annotations

import arviz as az
import numpy as np
import pymc as pm
import pytest

from neighbayes import dgp
from neighbayes.models import SARNegBin
from neighbayes.tests.helpers import W_to_graph, _as_sparse, make_line_W


def _idata(vars_dict: dict[str, np.ndarray]) -> az.InferenceData:
    payload = {k: np.asarray(v)[None, ...] for k, v in vars_dict.items()}
    return az.from_dict(posterior=payload)


def _count_data(seed: int = 101):
    rng = np.random.default_rng(seed)
    n = 10
    x1 = rng.normal(size=n)
    X = np.column_stack([np.ones(n), x1])
    eta = 0.3 + 0.6 * x1
    mu = np.exp(eta)
    y = rng.poisson(mu).astype(float)
    W = W_to_graph(make_line_W(n))
    return y, X, W


def test_sar_negbin_build_pymc_model():
    """Reduced-form NUTS model has rho, beta, alpha — no jacobian, sigma or z."""
    y, X, W = _count_data()
    model = SARNegBin(y=y, X=X, W=W)
    pymc_model = model._build_pymc_model()

    assert isinstance(pymc_model, pm.Model)
    assert "rho" in pymc_model.named_vars
    assert "alpha" in pymc_model.named_vars
    assert "beta" in pymc_model.named_vars
    # No log|I - rhoW| Jacobian: y is not linearly transformed in the
    # reduced-form count model, so there is no change of variables to
    # correct for. Including it biases rho toward zero.
    assert "jacobian" not in pymc_model.named_vars
    # Reduced form must NOT have sigma, sigma2, or z
    assert "sigma" not in pymc_model.named_vars
    assert "sigma2" not in pymc_model.named_vars
    assert "z" not in pymc_model.named_vars


def test_sar_negbin_rejects_noninteger_or_negative_y():
    _, X, W = _count_data(seed=102)

    y_bad = np.array([0.0, 1.2, 2.0, 1.0])
    X_bad = np.column_stack([np.ones(4), np.arange(4)])
    with pytest.raises(ValueError, match="integer-valued"):
        SARNegBin(y=y_bad, X=X_bad, W=W_to_graph(make_line_W(4)))

    y_neg = np.array([0.0, 1.0, -1.0, 2.0])
    with pytest.raises(ValueError, match="non-negative"):
        SARNegBin(y=y_neg, X=X_bad, W=W_to_graph(make_line_W(4)))


def test_sar_negbin_fitted_values_and_effects_with_mock_posterior():
    y, X, W = _count_data(seed=103)
    n = X.shape[0]
    model = SARNegBin(y=y, X=X, W=W)

    # Reduced form: only β, ρ, α — no σ, no z.
    model._idata = _idata(
        {
            "beta": np.stack([np.array([0.2, 0.7]), np.array([0.21, 0.71])]),
            "rho": np.array([0.15, 0.16]),
            "alpha": np.array([2.0, 2.1]),
        }
    )

    fitted = model.fitted_values()
    effects = model.spatial_effects()
    count_effects, samples = model.spatial_effects(
        scale="count", return_posterior_samples=True
    )

    assert fitted.shape == y.shape
    assert np.all(np.isfinite(fitted))
    assert np.all(fitted > 0)
    assert "direct" in effects.columns
    assert np.all(np.isfinite(effects["direct"].values))
    assert "direct" in count_effects.columns
    assert np.all(np.isfinite(count_effects["direct"].values))
    assert count_effects.attrs["scale"] == "count"
    assert samples["direct"].shape[1] == 1
    assert not np.allclose(effects["direct"].values, count_effects["direct"].values)


def test_sar_negbin_count_effects_sparse_matches_eigen():
    """Sparse Hutchinson path agrees with eigen path within Monte-Carlo tol.

    Also exercises the cached-LU code path: each draw must factorize A
    exactly once and reuse the factor across all 20 Hutchinson probes plus
    the eta and row-sum solves. Regression test for the prior bug that
    re-factorized A on every probe and returned only a 2-tuple.
    """
    y, X, W = _count_data(seed=103)
    n = X.shape[0]
    model = SARNegBin(y=y, X=X, W=W)
    model._idata = _idata(
        {
            "beta": np.stack([np.array([0.2, 0.7]), np.array([0.21, 0.71])]),
            "rho": np.array([0.15, 0.16]),
            "alpha": np.array([2.0, 2.1]),
        }
    )

    df_eigen, samples_eigen = model.spatial_effects(
        scale="count", return_posterior_samples=True, method="eigen"
    )
    df_sparse, samples_sparse = model.spatial_effects(
        scale="count", return_posterior_samples=True, method="sparse"
    )

    for key in ("direct", "indirect", "total"):
        assert samples_sparse[key].shape == samples_eigen[key].shape
        assert np.all(np.isfinite(samples_sparse[key]))
        # Hutchinson with 20 probes on small n is loose; eigen is exact.
        # Direct/indirect/total bases scale with mean(mu * diag) and
        # mean(mu * row_sum); for small W the structure dominates and
        # estimates land within ~25% relative error.
        np.testing.assert_allclose(
            samples_sparse[key], samples_eigen[key], rtol=0.35, atol=5e-3
        )

    assert df_sparse.attrs["scale"] == "count"
    assert list(df_sparse.columns) == list(df_eigen.columns)


def test_sar_negbin_count_effects_sparse_batched_matches_columnwise():
    """Batched (n, 22) matrix solve must equal per-column solves bit-exact.

    Regression test for the Phase-E refactor: the cached LU is reused
    across a stacked RHS containing ``[Xβ, ones?, Z₁, …, Z_K]``. Slicing
    bugs in column extraction would silently swap eta with row sums or
    misalign Hutchinson probes. Re-solve each RHS column independently
    and verify direct/indirect/total samples agree to machine precision.
    """
    import scipy.sparse as sp

    y, X, W = _count_data(seed=131)
    n = X.shape[0]
    model = SARNegBin(y=y, X=X, W=W)
    model._idata = _idata(
        {
            "beta": np.stack([np.array([0.25, 0.55]), np.array([0.24, 0.57])]),
            "rho": np.array([0.18, 0.22]),
            "alpha": np.array([1.8, 2.2]),
        }
    )

    _, samples_batched = model.spatial_effects(
        scale="count", return_posterior_samples=True, method="sparse"
    )

    # Re-implement the per-draw arithmetic with one solve per RHS column.
    rng = np.random.default_rng(42)
    n = X.shape[0]
    Z = rng.choice(np.array([-1.0, 1.0], dtype=np.float64), size=(n, 20))
    ones = np.ones(n, dtype=np.float64)
    I_n = sp.eye(n, format="csr", dtype=np.float64)
    rho_draws = np.array([0.18, 0.22])
    beta_draws = np.stack([np.array([0.25, 0.55]), np.array([0.24, 0.57])])
    ni = model._nonintercept_indices

    direct = np.empty((2, len(ni)))
    total = np.empty((2, len(ni)))
    for g, (rho, beta) in enumerate(zip(rho_draws, beta_draws, strict=False)):
        A = (I_n - float(rho) * model._W_sparse).tocsc()
        solver = sp.linalg.splu(A)
        eta = solver.solve(X @ beta)
        AinvZ = np.column_stack([solver.solve(Z[:, k]) for k in range(20)])
        mu = np.exp(np.clip(eta, -50.0, 50.0))
        diag_est = np.mean(Z * AinvZ, axis=1)
        if model._is_row_std:
            row_sums = np.full(n, 1.0 / (1.0 - float(rho)))
        else:
            row_sums = solver.solve(ones)
        direct[g] = float(np.mean(mu * diag_est)) * beta[ni]
        total[g] = float(np.mean(mu * row_sums)) * beta[ni]

    np.testing.assert_allclose(
        samples_batched["direct"], direct, atol=1e-10, rtol=1e-10
    )
    np.testing.assert_allclose(samples_batched["total"], total, atol=1e-10, rtol=1e-10)
    np.testing.assert_allclose(
        samples_batched["indirect"], total - direct, atol=1e-10, rtol=1e-10
    )


def test_sar_negbin_spatial_effects_rejects_unknown_method():
    y, X, W = _count_data(seed=103)
    n = X.shape[0]
    model = SARNegBin(y=y, X=X, W=W)
    model._idata = _idata(
        {
            "beta": np.stack([np.array([0.2, 0.7]), np.array([0.21, 0.71])]),
            "rho": np.array([0.15, 0.16]),
            "alpha": np.array([2.0, 2.1]),
        }
    )
    with pytest.raises(ValueError, match="method must be one of"):
        model.spatial_effects(scale="count", method="bogus")


def test_sar_negbin_spatial_effects_rejects_unknown_scale():
    y, X, W = _count_data(seed=104)
    n = X.shape[0]
    model = SARNegBin(y=y, X=X, W=W)
    model._idata = _idata(
        {
            "beta": np.stack([np.array([0.2, 0.7]), np.array([0.21, 0.71])]),
            "rho": np.array([0.15, 0.16]),
            "alpha": np.array([2.0, 2.1]),
        }
    )

    with pytest.raises(ValueError, match="scale must be either 'logmean' or 'count'"):
        model.spatial_effects(scale="response")


def test_simulate_sar_negbin_output_contract():
    W = W_to_graph(make_line_W(8))
    out = dgp.simulate_sar_negbin(W=_as_sparse(W), rho=0.25, alpha=1.5, seed=42)

    assert {"y", "X", "mu", "W_sparse", "W_graph", "params_true"}.issubset(out)
    y = out["y"]
    assert y.ndim == 1
    assert np.all(y >= 0)
    assert np.allclose(y, np.round(y))
    assert out["params_true"]["alpha"] == 1.5


@pytest.mark.requires_jax
def test_sar_negbin_jax_logp_grad(monkeypatch):
    """End-to-end smoke: SAR-NB logp + grad compile under JAX (sparsax SAR)."""
    pytest.importorskip("jax")
    pytest.importorskip("sparsax")

    monkeypatch.setenv("NEIGHBAYES_JAX_SAR_SOLVER", "sparsax")
    monkeypatch.setenv("NEIGHBAYES_JAX_SPARSE_STRICT", "1")

    from neighbayes._jax_dispatch import (
        _select_jax_sar_solver,
        _select_jax_sparse_backend,
        register_jax_dispatch,
    )

    _select_jax_sparse_backend.cache_clear()
    _select_jax_sar_solver.cache_clear()
    register_jax_dispatch.cache_clear()
    register_jax_dispatch()

    try:
        y, X, W = _count_data(seed=104)
        model = SARNegBin(y=y, X=X, W=W)
        pm_model = model._build_pymc_model()

        with pm_model:
            ip = pm_model.initial_point()
            ip["beta"] = np.array([0.3, 0.6])
            ip["rho_interval__"] = np.array(0.4)
            # Reduced form: no sigma2_log__ or z
            logp_fn = pm_model.compile_logp(mode="JAX")
            dlogp_fn = pm_model.compile_dlogp(mode="JAX")
            lp = logp_fn(ip)
            grads = dlogp_fn(ip)

        assert np.isfinite(lp)
        assert np.all(np.isfinite(np.asarray(grads)))
    finally:
        _select_jax_sparse_backend.cache_clear()
        _select_jax_sar_solver.cache_clear()
        register_jax_dispatch.cache_clear()


def test_sar_negbin_fit_rejects_invalid_sampler():
    """fit() raises ValueError for invalid sampler string."""
    y, X, W = _count_data(seed=105)
    model = SARNegBin(y=y, X=X, W=W)
    with pytest.raises(ValueError, match="sampler must be"):
        model.fit(draws=10, tune=10, sampler="hmc")


def test_sar_negbin_fit_nuts_routes_to_nuts_path(monkeypatch):
    """fit(sampler='nuts') routes through the base NUTS path (_fit_nuts)."""
    from neighbayes.models.base import SpatialModel

    y, X, W = _count_data(seed=106)
    model = SARNegBin(y=y, X=X, W=W)

    called_with = {}

    def _fake_fit_nuts(self, **kwargs):
        called_with.update(kwargs)
        self._idata = az.from_dict(posterior={"rho": np.array([[0.1]])})
        return self._idata, False

    monkeypatch.setattr(SpatialModel, "_fit_nuts", _fake_fit_nuts)
    model.fit(draws=50, tune=25, chains=1, random_seed=42, sampler="nuts")

    assert called_with["draws"] == 50
    assert called_with["tune"] == 25
    assert called_with["chains"] == 1
    assert called_with["random_seed"] == 42
