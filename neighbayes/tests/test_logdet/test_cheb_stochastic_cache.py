"""One stochastic Chebyshev precompute per fit, and the looped JAX evaluator."""

from __future__ import annotations

import sys

import numpy as np
import pytest
import scipy.sparse as sp

import neighbayes._logdet._factories as fac
from neighbayes._logdet import clear_logdet_fn_cache, make_logdet_jax_fn
from neighbayes.tests.helpers import make_rook_W

BACKENDS = ["numpy"] if sys.platform.startswith("win") else ["numpy", "jax"]


@pytest.fixture
def W():
    return sp.csr_matrix(make_rook_W(20))


@pytest.fixture
def count_precomputes(monkeypatch):
    clear_logdet_fn_cache()
    calls = []
    orig = fac.cheb_stochastic_logdet_precompute

    def counted(*a, **k):
        calls.append(1)
        return orig(*a, **k)

    monkeypatch.setattr(fac, "cheb_stochastic_logdet_precompute", counted)
    yield calls
    clear_logdet_fn_cache()


def test_coefficients_are_computed_once_per_key(W, count_precomputes):
    first = fac._cheb_stochastic_coeffs(W, -0.99, 0.99)
    again = fac._cheb_stochastic_coeffs(W.copy(), -0.99, 0.99)
    assert again is first and len(count_precomputes) == 1
    fac._cheb_stochastic_coeffs(W, 0.0, 0.95)
    assert len(count_precomputes) == 2
    with pytest.raises(ValueError):
        first[0][0] = 0.0
    clear_logdet_fn_cache()
    fac._cheb_stochastic_coeffs(W, -0.99, 0.99)
    assert len(count_precomputes) == 3


@pytest.mark.parametrize("backend", BACKENDS)
def test_a_fit_builds_one_precompute(W, backend, count_precomputes):
    from scipy.sparse.linalg import spsolve

    from neighbayes.models.cross_section import SAR

    n = W.shape[0]
    rng = np.random.default_rng(0)
    X = np.column_stack([np.ones(n), rng.normal(size=n)])
    A = (sp.eye(n, format="csc") - 0.5 * sp.csc_matrix(W)).tocsc()
    y = spsolve(A, X @ np.array([1.0, 2.0]) + rng.normal(size=n))
    SAR(y=y, X=X, W=W, logdet_method="cheb_stochastic", logdet_probe_check=False).fit(
        draws=50,
        tune=50,
        chains=2,
        random_seed=1,
        progressbar=False,
        gibbs_backend=backend,
    )
    assert len(count_precomputes) == 1


def test_looped_jax_evaluator_matches_the_unrolled_one(W):
    jax = pytest.importorskip("jax")
    from neighbayes._logdet._jax import jax_logdet_chebyshev

    fn = make_logdet_jax_fn(W, method="cheb_stochastic", rho_min=-0.99, rho_max=0.99)
    coeffs, lo, hi = fac._cheb_stochastic_coeffs(W, -0.99, 0.99)

    def unrolled(r):
        return jax_logdet_chebyshev(r, coeffs, rmin=lo, rmax=hi)

    for r in (-0.5, 0.3, 0.9, 0.98):
        assert float(fn(r)) == pytest.approx(float(unrolled(r)), rel=1e-12, abs=1e-9)
        assert float(jax.grad(fn)(r)) == pytest.approx(
            float(jax.grad(unrolled)(r)), rel=1e-10, abs=1e-8
        )
