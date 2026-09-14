"""End-to-end tests for ``logdet_refit`` on the Gaussian Gibbs samplers.

Rebuilding the Jacobian interpolant partway through warmup changes the
transition kernel, so the thing that has to be established is that it does not
change the *answer*.  It is an adaptation in the same sense as step-size or
slice-width tuning: allowed during warmup, frozen before the first retained
draw.

These run on both backends.  The JAX path is the one with a sharp edge — its
Gibbs step is compiled once and reused, so the interpolant is carried as traced
arrays of fixed capacity rather than closure constants; otherwise the swap would
retrace the step and cost far more than the refit saves.
"""

from __future__ import annotations

import sys

import numpy as np
import pytest
import scipy.sparse as sp

from neighbayes.models.cross_section import SAR, SEM

# jaxlib is not available on Windows, so only run the NumPy backend there.
BACKENDS = ["numpy"] if sys.platform.startswith("win") else ["numpy", "jax"]


def _rook(side: int) -> sp.csr_matrix:
    n = side * side
    rows, cols = [], []
    for i in range(side):
        for j in range(side):
            for di, dj in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                ni, nj = i + di, j + dj
                if 0 <= ni < side and 0 <= nj < side:
                    rows.append(i * side + j)
                    cols.append(ni * side + nj)
    A = sp.coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n)).tocsr()
    deg = np.asarray(A.sum(axis=1)).ravel()
    return sp.csr_matrix(sp.diags(1.0 / deg) @ A)


@pytest.fixture(scope="module")
def data():
    from scipy.sparse.linalg import spsolve

    W = _rook(24)
    n = W.shape[0]
    rng = np.random.default_rng(20260727)
    X = np.column_stack([np.ones(n), rng.normal(size=n), rng.normal(size=n)])
    beta = np.array([1.0, 2.0, -1.0])
    eps = rng.normal(size=n)
    A = (sp.eye(n, format="csc") - 0.6 * sp.csc_matrix(W)).tocsc()
    y = spsolve(A, X @ beta + eps)
    return y, X, W


PRIORS = {"rho_lower": 0.0, "rho_upper": 0.95}


def _fit(data, backend, **kw):
    y, X, W = data
    model = SAR(y=y, X=X, W=W, priors=PRIORS, logdet_method="cheb_cholesky", **kw)
    idata = model.fit(
        draws=1500,
        tune=1000,
        chains=4,
        random_seed=7,
        progressbar=False,
        gibbs_backend=backend,
    )
    return idata, np.asarray(idata.posterior["rho"]).ravel()


@pytest.mark.parametrize("backend", BACKENDS)
class TestRefitDoesNotMoveThePosterior:
    def test_agrees_with_the_unrefitted_fit(self, data, backend):
        """Any shift must be inside Monte Carlo error."""
        import arviz as az

        base_idata, base = _fit(data, backend, logdet_refit=False)
        ref_idata, ref = _fit(data, backend, logdet_refit=True)

        ess_b = float(az.ess(base_idata, var_names=["rho"])["rho"])
        ess_r = float(az.ess(ref_idata, var_names=["rho"])["rho"])
        mcse = np.sqrt(base.var(ddof=1) / ess_b + ref.var(ddof=1) / ess_r)

        assert abs(ref.mean() - base.mean()) < 4.0 * mcse
        assert ref.std(ddof=1) == pytest.approx(base.std(ddof=1), rel=0.10)

    def test_window_contains_the_unrefitted_posterior(self, data, backend):
        """A window that clipped real mass would bias the answer."""
        _, base = _fit(data, backend, logdet_refit=False)
        idata, _ = _fit(data, backend, logdet_refit=True)
        lo, hi = idata.attrs["logdet_refit_window"]
        q_lo, q_hi = np.quantile(base, [0.0005, 0.9995])
        assert lo < q_lo and hi > q_hi

    def test_refit_is_recorded(self, data, backend):
        idata, _ = _fit(data, backend, logdet_refit=True)
        assert "logdet_refit_window" in idata.attrs
        assert idata.attrs["logdet_refit_order"] > 0
        assert idata.attrs["logdet_refit_pad_sd"] == 10.0

    def test_window_is_narrower_than_the_prior(self, data, backend):
        """A refit that happened at all must have cleared the narrowing guard.

        How much narrower depends on the problem: this ``n = 576`` design has a
        posterior sd near 0.03, so a ±10-sd window is only ~1.6× inside the
        prior.  The gain grows with ``n`` — at ``n = 2,500`` the same design
        narrows by ~3.5× — so the contract to assert is the guard, not a number.
        """
        from neighbayes._logdet._refit import MIN_NARROWING

        idata, _ = _fit(data, backend, logdet_refit=True)
        lo, hi = idata.attrs["logdet_refit_window"]
        prior_width = PRIORS["rho_upper"] - PRIORS["rho_lower"]
        assert prior_width / (hi - lo) >= MIN_NARROWING

    def test_on_by_default(self, data, backend):
        idata, _ = _fit(data, backend)
        assert "logdet_refit_window" in idata.attrs

    def test_off_records_nothing(self, data, backend):
        idata, _ = _fit(data, backend, logdet_refit=False)
        assert "logdet_refit_window" not in idata.attrs


@pytest.mark.parametrize("backend", BACKENDS)
def test_too_tight_a_window_warns(data, backend):
    """A window the chain runs into must be reported, not silently truncated."""
    with pytest.warns(RuntimeWarning, match="truncated"):
        _fit(data, backend, logdet_refit=True, logdet_refit_pad_sd=0.05)


@pytest.mark.parametrize("backend", BACKENDS)
def test_unsupported_method_falls_back_quietly(data, backend, caplog):
    """``eigenvalue`` is already exact; asking to refit it is a no-op, not an error."""
    import logging

    y, X, W = data
    model = SAR(
        y=y, X=X, W=W, priors=PRIORS, logdet_method="eigenvalue", logdet_refit=True
    )
    with caplog.at_level(logging.INFO, logger="neighbayes"):
        idata = model.fit(
            draws=300,
            tune=200,
            chains=2,
            random_seed=7,
            progressbar=False,
            gibbs_backend=backend,
        )
    assert "logdet_refit_window" not in idata.attrs
    assert np.isfinite(np.asarray(idata.posterior["rho"]).mean())


def test_sem_refits_on_lambda(data):
    """The refit must key off whichever parameter the family calls spatial."""
    y, X, W = data
    model = SEM(
        y=y,
        X=X,
        W=W,
        priors={"lam_lower": 0.0, "lam_upper": 0.95},
        logdet_method="cheb_cholesky",
        logdet_refit=True,
    )
    idata = model.fit(
        draws=800,
        tune=600,
        chains=2,
        random_seed=7,
        progressbar=False,
        gibbs_backend="numpy",
    )
    lo, hi = idata.attrs["logdet_refit_window"]
    lam = np.asarray(idata.posterior["lam"]).ravel()
    assert lo < lam.min() and hi > lam.max()
