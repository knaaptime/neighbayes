"""End-to-end tests for the warmup AAA node check on the Gaussian Gibbs samplers.

The check sets the number of exact factorizations behind an AAA log-determinant
from where the posterior lies: warmup starts on a 14-node fit, and at its midpoint
nodes are added only if the posterior lies near a singularity of the Jacobian.
Like the refit, it is an adaptation frozen before the first retained draw, so what
has to be established is that it does not move the answer.
"""

from __future__ import annotations

import sys

import numpy as np
import pytest
import scipy.sparse as sp

from neighbayes.models.cross_section import SAR

# jaxlib is not available on Windows, so only run the NumPy backend there.
BACKENDS = ["numpy"] if sys.platform.startswith("win") else ["numpy", "jax"]
PRIORS = {"rho_lower": -0.99, "rho_upper": 0.99}


def _rook(side: int) -> sp.csr_matrix:
    path = sp.diags([np.ones(side - 1), np.ones(side - 1)], [-1, 1])
    eye = sp.eye(side)
    A = sp.csr_matrix(sp.kron(path, eye) + sp.kron(eye, path))
    deg = np.asarray(A.sum(axis=1)).ravel()
    return sp.csr_matrix(sp.diags(1.0 / deg) @ A)


def _data(rho: float, side: int = 24):
    from scipy.sparse.linalg import spsolve

    W = _rook(side)
    n = W.shape[0]
    rng = np.random.default_rng(20260727)
    X = np.column_stack([np.ones(n), rng.normal(size=n), rng.normal(size=n)])
    eps = rng.normal(size=n)
    A = sp.eye(n, format="csc") - rho * sp.csc_matrix(W)
    y = spsolve(A, X @ np.array([1.0, 2.0, -1.0]) + eps)
    return y, X, W


@pytest.fixture(scope="module")
def mid():
    return _data(0.5)


@pytest.fixture(scope="module")
def high():
    return _data(0.95)


def _fit(data, backend, method="chol_aaa", **kw):
    y, X, W = data
    model = SAR(y=y, X=X, W=W, priors=PRIORS, logdet_method=method, **kw)
    return model.fit(
        draws=600,
        tune=600,
        chains=2,
        random_seed=7,
        progressbar=False,
        gibbs_backend=backend,
    )


def _rho(idata) -> np.ndarray:
    return np.asarray(idata.posterior["rho"]).ravel()


@pytest.mark.parametrize("backend", BACKENDS)
class TestAAANodeCheck:
    def test_on_by_default_and_keeps_the_pilot_away_from_the_singularity(self, mid, backend):
        idata = _fit(mid, backend)
        assert idata.attrs["logdet_aaa_nodes"] == 14
        assert idata.attrs["logdet_aaa_check_passed"] == 1
        lo, hi = idata.attrs["logdet_aaa_region"]
        assert lo < _rho(idata).mean() < hi

    def test_adds_nodes_near_the_singularity(self, high, backend):
        idata = _fit(high, backend)
        assert idata.attrs["logdet_aaa_nodes"] > 14
        assert idata.attrs["logdet_aaa_check_passed"] == 1

    def test_agrees_with_the_exact_jacobian(self, high, backend):
        """Any shift from the eigenvalue Jacobian must be inside Monte Carlo error."""
        import arviz as az

        aaa = _fit(high, backend)
        exact = _fit(high, backend, method="eigenvalue")
        ra, re = _rho(aaa), _rho(exact)
        ess_a = float(az.ess(aaa, var_names=["rho"])["rho"])
        ess_e = float(az.ess(exact, var_names=["rho"])["rho"])
        mcse = np.sqrt(ra.var(ddof=1) / ess_a + re.var(ddof=1) / ess_e)
        assert abs(ra.mean() - re.mean()) < 4.0 * mcse

    def test_off_records_nothing(self, mid, backend):
        idata = _fit(mid, backend, logdet_aaa_check=False)
        assert "logdet_aaa_nodes" not in idata.attrs
        assert np.isfinite(_rho(idata).mean())


def test_non_aaa_methods_are_untouched(mid):
    idata = _fit(mid, "numpy", method="cheb_cholesky")
    assert "logdet_aaa_nodes" not in idata.attrs


def test_panel_fe_runs_refit_and_check():
    """Gaussian FE panels share the sampler, so both adaptations reach them."""
    from neighbayes.models.panel._fe import SARPanelFE

    N, T = 144, 4
    W = _rook(12)
    rng = np.random.default_rng(3)
    X = rng.normal(size=(N * T, 1))
    from scipy.sparse.linalg import spsolve

    A = sp.eye(N, format="csc") - 0.5 * sp.csc_matrix(W)
    y = np.concatenate(
        [spsolve(A, 2.0 * X[t * N:(t + 1) * N, 0] + rng.normal(size=N)) for t in range(T)]
    )
    model = SARPanelFE(y=y, X=X, W=W, N=N, T=T, effects=1, logdet_method="chol_aaa")
    idata = model.fit(
        sampler="gibbs",
        draws=300,
        tune=300,
        chains=2,
        random_seed=7,
        n_jobs=1,
        progressbar=False,
    )
    assert "logdet_refit_window" in idata.attrs
    assert idata.attrs["logdet_aaa_nodes"] >= 14
