"""Fast build/method tests for SARNegBin NUTS path.

These exercise the reduced-form NUTS model (no σ, no z) that is now
built into SARNegBin via _build_pymc_model().
"""

from __future__ import annotations

import arviz as az
import numpy as np
import pymc as pm
import pytest
import xarray as xr

from neighbayes import dgp
from neighbayes.models import SARNegBin
from neighbayes.tests.helpers import W_to_graph, _as_sparse, make_line_W


def _idata(vars_dict: dict[str, np.ndarray]) -> xr.DataTree:
    payload = {k: np.asarray(v)[None, ...] for k, v in vars_dict.items()}
    return az.from_dict({"posterior": payload})


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
    model = SARNegBin(y=y, X=X, W=W, latent=False)
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
        SARNegBin(y=y_bad, X=X_bad, W=W_to_graph(make_line_W(4)), latent=False)

    y_neg = np.array([0.0, 1.0, -1.0, 2.0])
    with pytest.raises(ValueError, match="non-negative"):
        SARNegBin(y=y_neg, X=X_bad, W=W_to_graph(make_line_W(4)), latent=False)


def test_sar_negbin_fitted_values_and_effects_with_mock_posterior():
    y, X, W = _count_data(seed=103)
    n = X.shape[0]
    model = SARNegBin(y=y, X=X, W=W, latent=False)

    # Reduced form: only β, ρ, α — no σ, no z.
    model._idata = _idata(
        {
            "beta": np.stack([np.array([0.2, 0.7]), np.array([0.21, 0.71])]),
            "rho": np.array([0.15, 0.16]),
            "alpha": np.array([2.0, 2.1]),
        }
    )

    fitted = model.fitted_values()
    effects = model.impacts()
    count_effects, samples = model.impacts(scale="count", return_posterior_samples=True)

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


def test_sar_negbin_count_effects_match_the_dense_multiplier():
    """Count-scale effects, interpolated across draws, against explicit inverses."""
    y, X, W = _count_data(seed=103)
    model = SARNegBin(y=y, X=X, W=W, latent=False)
    beta = np.stack([np.array([0.2, 0.7]), np.array([0.21, 0.71]), [0.25, 0.55]])
    rho = np.array([0.15, 0.62, -0.3])
    model._idata = _idata(
        {"beta": beta, "rho": rho, "alpha": np.array([2.0, 2.1, 1.8])}
    )

    df, samples = model.impacts(scale="count", return_posterior_samples=True)

    Wd = model._W_sparse.toarray()
    ni = model._nonintercept_indices
    direct, total = [], []
    for r, b in zip(rho, beta):
        S = np.linalg.inv(np.eye(Wd.shape[0]) - r * Wd)
        mu = np.exp(S @ X @ b)
        direct.append(np.mean(mu * np.diag(S)) * b[ni])
        total.append(np.mean(mu * S.sum(axis=1)) * b[ni])
    direct, total = np.array(direct), np.array(total)
    np.testing.assert_allclose(samples["direct"], direct, rtol=1e-8)
    np.testing.assert_allclose(samples["total"], total, rtol=1e-8)
    np.testing.assert_allclose(samples["indirect"], total - direct, rtol=1e-7)
    assert df.attrs["scale"] == "count"


def test_sar_negbin_impacts_rejects_unknown_scale():
    y, X, W = _count_data(seed=104)
    n = X.shape[0]
    model = SARNegBin(y=y, X=X, W=W, latent=False)
    model._idata = _idata(
        {
            "beta": np.stack([np.array([0.2, 0.7]), np.array([0.21, 0.71])]),
            "rho": np.array([0.15, 0.16]),
            "alpha": np.array([2.0, 2.1]),
        }
    )

    with pytest.raises(ValueError, match="scale must be either 'logmean' or 'count'"):
        model.impacts(scale="response")


def test_simulate_sar_negbin_output_contract():
    W = W_to_graph(make_line_W(8))
    out = dgp.simulate_sar_negbin(
        W=_as_sparse(W), rho=0.25, alpha=1.5, seed=42, latent=False
    )

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
        model = SARNegBin(y=y, X=X, W=W, latent=False)
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
    model = SARNegBin(y=y, X=X, W=W, latent=False)
    with pytest.raises(ValueError, match="sampler must be"):
        model.fit(draws=10, tune=10, sampler="hmc")


def test_sar_negbin_fit_nuts_routes_to_nuts_path(monkeypatch):
    """fit(sampler='nuts') routes through the base NUTS path (_fit_nuts)."""
    from neighbayes.models.base import SpatialModel

    y, X, W = _count_data(seed=106)
    model = SARNegBin(y=y, X=X, W=W, latent=False)

    called_with = {}

    def _fake_fit_nuts(self, **kwargs):
        called_with.update(kwargs)
        self._idata = az.from_dict({"posterior": {"rho": np.array([[0.1]])}})
        return self._idata, False

    monkeypatch.setattr(SpatialModel, "_fit_nuts", _fake_fit_nuts)
    model.fit(draws=50, tune=25, chains=1, random_seed=42, sampler="nuts")

    assert called_with["draws"] == 50
    assert called_with["tune"] == 25
    assert called_with["chains"] == 1
    assert called_with["random_seed"] == 42
