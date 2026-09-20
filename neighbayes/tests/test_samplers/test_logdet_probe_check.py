"""Gibbs samplers with the stochastic Chebyshev probe count set at the warmup midpoint.

The probe check starts warmup on a small probe pool, prices the bias of the
posterior mean of ρ from the pool's own spread on the pooled warmup draws, and
grows the pool before the first retained draw.  These tests fit both backends
and compare against the exact (eigenvalue) log-determinant.
"""

from __future__ import annotations

import sys

import numpy as np
import pytest
import scipy.sparse as sp

from neighbayes._logdet._cheb_stochastic import DEFAULT_PROBE_MIN
from neighbayes._logdet._probe_check import FIXED_PROBES, WarmupProbes
from neighbayes.models.cross_section import SAR, SEM
from neighbayes.tests.helpers import make_rook_W

BACKENDS = ["numpy"] if sys.platform.startswith("win") else ["numpy", "jax"]


def _rook(side):
    return sp.csr_matrix(make_rook_W(side))


def _sar_data(side, rho, seed):
    from scipy.sparse.linalg import spsolve

    W = _rook(side)
    n = W.shape[0]
    rng = np.random.default_rng(seed)
    X = np.column_stack([np.ones(n), rng.normal(size=n), rng.normal(size=n)])
    A = (sp.eye(n, format="csc") - rho * sp.csc_matrix(W)).tocsc()
    y = spsolve(A, X @ np.array([1.0, 2.0, -1.0]) + rng.normal(size=n))
    return y, X, W


@pytest.fixture(scope="module")
def data():
    return _sar_data(24, 0.9, 20260919)


def _fit(model_cls, data, backend, **kw):
    y, X, W = data
    model = model_cls(y=y, X=X, W=W, **kw)
    idata = model.fit(
        draws=1500,
        tune=1000,
        chains=2,
        random_seed=11,
        progressbar=False,
        gibbs_backend=backend,
    )
    name = "rho" if model_cls is SAR else "lam"
    return idata, np.asarray(idata.posterior[name]).ravel()


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("model_cls", [SAR, SEM])
class TestProbeCheckFit:
    def test_agrees_with_the_exact_logdet(self, data, backend, model_cls):
        import arviz as az

        exact_idata, exact = _fit(model_cls, data, backend, logdet_method="eigenvalue")
        idata, draws = _fit(
            model_cls,
            data,
            backend,
            logdet_method="cheb_stochastic",
            logdet_probe_check=True,
        )
        name = "rho" if model_cls is SAR else "lam"
        ess_e = float(az.ess(exact_idata, var_names=[name])[name])
        ess_p = float(az.ess(idata, var_names=[name])[name])
        mcse = np.sqrt(exact.var(ddof=1) / ess_e + draws.var(ddof=1) / ess_p)
        assert abs(draws.mean() - exact.mean()) < 4.0 * mcse
        assert draws.std(ddof=1) == pytest.approx(exact.std(ddof=1), rel=0.15)

    def test_probe_count_is_recorded(self, data, backend, model_cls):
        idata, _ = _fit(
            model_cls,
            data,
            backend,
            logdet_method="cheb_stochastic",
            logdet_probe_check=True,
        )
        a = idata.attrs
        assert a["logdet_probes"] >= DEFAULT_PROBE_MIN
        assert a["logdet_probe_bar"] == pytest.approx(0.045)
        if not a["logdet_probe_capped"]:
            assert a["logdet_probe_bias"] <= a["logdet_probe_bar"] / a["logdet_probe_z"]


@pytest.mark.parametrize("backend", BACKENDS)
def test_on_by_default(data, backend):
    idata, _ = _fit(SAR, data, backend, logdet_method="cheb_stochastic")
    assert idata.attrs["logdet_probes"] >= DEFAULT_PROBE_MIN


@pytest.mark.parametrize("backend", BACKENDS)
def test_off_records_nothing(data, backend):
    idata, _ = _fit(
        SAR, data, backend, logdet_method="cheb_stochastic", logdet_probe_check=False
    )
    assert "logdet_probes" not in idata.attrs


@pytest.mark.parametrize("backend", BACKENDS)
def test_other_methods_ignore_the_flag(data, backend):
    idata, _ = _fit(
        SAR, data, backend, logdet_method="cheb_cholesky", logdet_probe_check=True
    )
    assert "logdet_probes" not in idata.attrs


class TestWarmupProbes:
    @pytest.fixture(scope="class")
    def W(self):
        return _rook(20)

    @staticmethod
    def _draws(mu=0.9, sd=0.01):
        return np.random.default_rng(3).normal(mu, sd, 2000)

    def test_warmup_starts_small_and_without_a_midpoint_uses_the_fixed_count(self, W):
        wp = WarmupProbes(W, min_probes=12)
        wp.initial(tune=1000)
        assert wp.pool.n_probes == 12
        wp.initial(tune=1)
        assert wp.pool.n_probes == FIXED_PROBES and not wp.adapts(1)
        default = WarmupProbes(W)
        default.initial(tune=1000)
        assert default.pool.n_probes == DEFAULT_PROBE_MIN == FIXED_PROBES

    def test_loose_target_keeps_the_installed_evaluators(self, W):
        wp = WarmupProbes(W, bar=10.0)
        wp.initial(tune=1000)
        assert wp.adapt(self._draws()) is None
        assert wp.probe_check.n_probes == DEFAULT_PROBE_MIN

    def test_growth_keeps_the_jax_parameter_shape(self, W):
        pytest.importorskip("jax")
        wp = WarmupProbes(W, bar=0.02, min_probes=12)
        start = wp.initial(tune=1000, jax=True)
        adapted = wp.adapt(self._draws(), jax=True)
        assert adapted is not None and wp.pool.n_probes > 12
        assert np.shape(adapted.params[0]) == np.shape(start.params[0])
        assert not np.allclose(adapted.params[0], start.params[0])
        assert float(wp.param_fn()(0.9, adapted.params)) == pytest.approx(
            adapted.scalar_fn(0.9), rel=1e-10
        )

    def test_panel_evaluators_carry_T(self, W):
        per_period = WarmupProbes(W).initial(tune=1000)
        panel = WarmupProbes.for_sampler(sp.block_diag([W] * 3, format="csr"), T=3)
        stacked = panel.initial(tune=1000)
        assert stacked.scalar_fn(0.5) == pytest.approx(3 * per_period.scalar_fn(0.5))

    def test_cap_warns(self, W):
        wp = WarmupProbes(W, bar=1e-6, min_probes=12, max_probes=16)
        wp.initial(tune=1000)
        with pytest.warns(RuntimeWarning, match="cap"):
            wp.adapt(self._draws())
        assert wp.probe_check.capped and wp.pool.n_probes == 16
