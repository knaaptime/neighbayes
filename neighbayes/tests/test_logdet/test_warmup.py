"""The warmup log-determinant component shared by the Gibbs samplers."""

from __future__ import annotations

import sys

import numpy as np
import pytest
import scipy.sparse as sp

from neighbayes._logdet._refit import AAA_PILOT_NODES
from neighbayes._logdet._warmup import WarmupJacobian, sampler_builds_evaluators


def _rook(side: int) -> sp.csr_matrix:
    path = sp.diags([np.ones(side - 1), np.ones(side - 1)], [-1, 1])
    eye = sp.eye(side)
    A = sp.csr_matrix(sp.kron(path, eye) + sp.kron(eye, path))
    deg = np.asarray(A.sum(axis=1)).ravel()
    return sp.csr_matrix(sp.diags(1.0 / deg) @ A)


def _draws(mean: float, sd: float = 0.01, n: int = 400) -> np.ndarray:
    return np.random.default_rng(0).normal(mean, sd, n)


class _Idata:
    def __init__(self):
        self.attrs: dict = {}


@pytest.fixture(scope="module")
def W():
    return _rook(30)


def test_sampler_builds_evaluators():
    assert sampler_builds_evaluators("chol_aaa", True, refit=False, aaa_check=True)
    assert sampler_builds_evaluators("cheb_cholesky", True, refit=True, aaa_check=False)
    assert not sampler_builds_evaluators("cheb_cholesky", True, refit=False, aaa_check=True)
    assert not sampler_builds_evaluators("eigenvalue", True, refit=True, aaa_check=True)
    assert not sampler_builds_evaluators("chol_aaa", False, refit=True, aaa_check=True)


def test_check_keeps_the_pilot_away_from_the_singularity(W):
    wj = WarmupJacobian(W, "chol_aaa", refit=False)
    wj.initial(tune=100)
    assert wj.adapt(_draws(0.5)) is None
    assert wj.check.nodes == AAA_PILOT_NODES and wj.check.passed


def test_check_adds_nodes_near_the_singularity(W):
    wj = WarmupJacobian(W, "chol_aaa", refit=False)
    wj.initial(tune=100)
    ev = wj.adapt(_draws(0.955, sd=0.002))
    assert ev is not None
    assert wj.check.passed and wj.check.nodes > AAA_PILOT_NODES
    assert (ev.rho_min, ev.rho_max) == wj.prior


def test_refit_narrows_the_support(W):
    wj = WarmupJacobian(W, "cheb_cholesky", aaa_check=False)
    wj.initial(tune=100)
    ev = wj.adapt(_draws(0.5))
    assert ev is not None and wj.window is not None
    assert wj.prior[0] < ev.rho_min < 0.5 < ev.rho_max < wj.prior[1]
    assert wj.check is None


def test_refit_then_check_runs_on_the_window(W):
    wj = WarmupJacobian(W, "chol_aaa")
    wj.initial(tune=100)
    ev = wj.adapt(_draws(0.5))
    assert ev is not None and wj.window is not None
    assert wj.check.passed
    assert wj.check.region[0] >= ev.rho_min and wj.check.region[1] <= ev.rho_max


def test_no_midpoint_means_the_static_fit(W):
    wj = WarmupJacobian(W, "chol_aaa")
    assert not wj.adapts(1)
    ev = wj.initial(tune=1)
    # The full stability region reaches past |ρ| = 0.9, so the static count is 18.
    assert wj.check.nodes == 18 and wj.check.region is None
    assert np.isfinite(ev.scalar_fn(0.3))


def test_unsupported_method_is_inactive(W):
    wj = WarmupJacobian(W, "eigenvalue")
    assert not wj.active and not wj.adapts(1000)


def test_record_writes_attrs_and_warns_at_a_truncated_edge(W):
    wj = WarmupJacobian(W, "chol_aaa", refit_pad_sd=0.05)
    wj.initial(tune=100)
    wj.adapt(_draws(0.5))
    idata = _Idata()
    with pytest.warns(RuntimeWarning, match="truncated"):
        wj.record(idata, _draws(0.5, sd=0.05, n=2000))
    assert "logdet_refit_window" in idata.attrs
    assert idata.attrs["logdet_aaa_nodes"] >= AAA_PILOT_NODES


def test_panel_uses_the_per_period_block(W):
    T = 3
    W_nt = sp.csr_matrix(sp.block_diag([W] * T))
    wj = WarmupJacobian.for_sampler(W_nt, "chol_aaa", T=T, refit=False)
    ev = wj.initial(tune=0)
    n = W.shape[0]
    exact = T * np.linalg.slogdet(np.eye(n) - 0.4 * W.toarray())[1]
    assert ev.scalar_fn(0.4) == pytest.approx(exact, rel=1e-4)


@pytest.mark.skipif(sys.platform.startswith("win"), reason="jaxlib unavailable on Windows")
def test_jax_params_keep_their_shape_across_the_adaptation(W):
    wj = WarmupJacobian(W, "chol_aaa", refit=False)
    first = wj.initial(tune=100, jax=True)
    later = wj.adapt(_draws(0.955, sd=0.002), jax=True)
    assert [np.shape(a) for a in first.params] == [np.shape(a) for a in later.params]
    f = wj.param_fn()
    assert float(f(0.4, later.params)) == pytest.approx(later.scalar_fn(0.4), rel=1e-9)


def test_pickles_without_the_factorization(W):
    """Chain closures reach worker processes by pickling; a live factor cannot go."""
    import pickle

    for method in ("chol_aaa", "cheb_cholesky"):
        wj = WarmupJacobian(W, method)
        wj.initial(tune=100)
        assert wj._fitter._context is not None
        clone = pickle.loads(pickle.dumps(wj))
        assert clone._fitter._context is None
        assert clone.prior == wj.prior and clone.method == method
