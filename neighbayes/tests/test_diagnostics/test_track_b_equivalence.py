"""Equivalence pins for the Phase 3 Track B diagnostics vectorizations.

Each optimization here is a behavior-preserving rewrite: the vectorized /
FFT / trace-identity form must reproduce the naive reference it replaces to
floating-point precision, so the estimators (bridge-sampling log-ML, robust
LM statistics) are provably unchanged.

Covered:

* ``_run_iterative_scheme`` — the per-sample logsumexp Python loops are
  replaced by broadcast ``np.logaddexp`` calls.
* ``_compute_ess`` — the dense O(n^2) ``np.correlate`` autocovariance is
  replaced by an FFT (O(n log n)).
* ``_sar_null_lambda_info`` — the O(n^3) ``G @ G`` / ``W_dense @ G`` products
  taken only for their trace are replaced by the O(n^2) identity
  ``tr(BC) = sum(B * C.T)``.
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp
from scipy.special import logsumexp

from neighbayes.diagnostics.bayesfactor import (
    _compute_ess,
    _run_iterative_scheme,
)
from neighbayes.diagnostics.lmtests.cross_sectional import _sar_null_lambda_info


def _reference_iterative_scheme(
    q11, q12, q21, q22, r0, tol, maxiter, criterion, neff, use_neff
):
    """Pre-vectorization reference: per-sample ``scipy.special.logsumexp`` loops."""
    N1 = len(q11)
    N2 = len(q21)
    l1 = q11 - q12
    l2 = q21 - q22
    lstar = np.median(l1)
    n_eff = neff if (use_neff and neff is not None) else N1
    s1 = n_eff / (n_eff + N2)
    s2 = N2 / (n_eff + N2)
    r = r0
    logml = np.log(r) + lstar
    criterion_val = 1.0 + tol
    i = 0
    while i < maxiter and criterion_val > tol:
        rold = r
        logmlold = logml
        log_s1 = np.log(s1)
        log_s2_r = np.log(s2) + np.log(r) if r > 0 else -np.inf
        l2_shifted = l2 - lstar
        l1_shifted = l1 - lstar
        log_num = np.array(
            [
                l2_shifted[j] - logsumexp(np.array([log_s1 + l2_shifted[j], log_s2_r]))
                for j in range(N2)
            ]
        )
        log_den = np.array(
            [
                l1_shifted[j] - logsumexp(np.array([log_s1 + l1_shifted[j], log_s2_r]))
                for j in range(N1)
            ]
        )
        num_vals = np.exp(log_num)
        den_vals = np.exp(log_den)
        r = np.mean(num_vals) / np.mean(den_vals)
        logml = np.log(r) + lstar
        if criterion == "r":
            criterion_val = abs((r - rold) / r) if r != 0 else abs(r)
        else:
            criterion_val = (
                abs((logml - logmlold) / logml) if logml != 0 else abs(logml)
            )
        i += 1
    return logml


def _make_bridge_inputs(seed=0, n=4000):
    rng = np.random.default_rng(seed)
    post = rng.multivariate_normal(np.zeros(2), np.eye(2), size=n)
    prop = rng.multivariate_normal(np.zeros(2), np.eye(2), size=n)
    from scipy.stats import multivariate_normal

    q11 = -0.5 * np.sum(post**2, axis=1)
    q12 = multivariate_normal.logpdf(post, mean=np.zeros(2), cov=np.eye(2))
    q21 = -0.5 * np.sum(prop**2, axis=1)
    q22 = multivariate_normal.logpdf(prop, mean=np.zeros(2), cov=np.eye(2))
    return q11, q12, q21, q22


class TestIterativeSchemeVectorization:
    @pytest.mark.parametrize("use_neff,neff", [(False, None), (True, 3123.4)])
    def test_matches_logsumexp_reference(self, use_neff, neff):
        q11, q12, q21, q22 = _make_bridge_inputs()
        kw = dict(
            r0=1.0, tol=1e-10, maxiter=1000, criterion="r", neff=neff, use_neff=use_neff
        )
        ref = _reference_iterative_scheme(q11, q12, q21, q22, **kw)
        got = _run_iterative_scheme(q11=q11, q12=q12, q21=q21, q22=q22, **kw)["logml"]
        np.testing.assert_allclose(got, ref, rtol=0, atol=1e-10)


class TestComputeEssFFT:
    def _reference_ess(self, samples):
        n_params = samples.shape[1]
        ess_vals = []
        for p in range(n_params):
            x = samples[:, p]
            n = len(x)
            xc = x - np.mean(x)
            var_x = np.var(x, ddof=0)
            if var_x == 0:
                ess_vals.append(n)
                continue
            max_lag = min(n // 2, 1000)
            acf = np.correlate(xc, xc, mode="full")[n - 1 :]
            acf = acf / (var_x * n)
            tau = 1.0
            for lag in range(1, max_lag):
                if acf[lag] < 0:
                    break
                tau += 2 * acf[lag]
            ess_vals.append(max(1.0, n / tau))
        return float(np.median(ess_vals))

    def test_matches_correlate_reference(self):
        rng = np.random.default_rng(7)
        # Autocorrelated series (AR(1)) across a few parameters.
        n, k = 3000, 4
        samples = np.zeros((n, k))
        for j in range(k):
            e = rng.normal(size=n)
            x = np.zeros(n)
            for t in range(1, n):
                x[t] = 0.6 * x[t - 1] + e[t]
            samples[:, j] = x
        np.testing.assert_allclose(
            _compute_ess(samples), self._reference_ess(samples), rtol=1e-8
        )


class TestSarNullTraceIdentity:
    def _reference_info(self, W_sparse, W_dense, X_design, beta, rho, sigma2, T_ww):
        """Old dense form: explicit G @ G and W_dense @ G matmuls."""
        n = W_sparse.shape[0]
        A = np.eye(n) - rho * W_dense
        G = np.linalg.solve(A, W_dense)
        T_GG = float(np.sum(G * G) + np.trace(G @ G))
        T_WG = float(np.sum(W_dense * G) + np.trace(W_dense @ G))
        return T_GG, T_WG

    def test_trace_identity_matches_dense(self):
        rng = np.random.default_rng(3)
        # n > 512 so the chunked column-solve path crosses a chunk boundary.
        n, k = 600, 3
        # Row-standardized ring lattice -> asymmetric weighted W.
        row = np.concatenate([np.arange(n), np.arange(n)])
        col = np.concatenate([(np.arange(n) - 1) % n, (np.arange(n) + 1) % n])
        W = sp.csr_matrix((rng.uniform(0.5, 1.5, 2 * n), (row, col)), shape=(n, n))
        d = np.asarray(W.sum(axis=1)).ravel()
        W = (sp.diags(1.0 / d) @ W).tocsr()
        W_dense = W.toarray()
        X = rng.normal(size=(n, k))
        beta = rng.normal(size=k)
        rho, sigma2, T_ww = (
            0.4,
            1.3,
            float((W_dense.T @ W_dense).trace() + (W_dense @ W_dense).trace()),
        )

        info = _sar_null_lambda_info(W, X, beta, rho, sigma2, T_ww)
        ref_GG, ref_WG = self._reference_info(W, W_dense, X, beta, rho, sigma2, T_ww)
        np.testing.assert_allclose(info["T_GG"], ref_GG, rtol=0, atol=1e-9)
        np.testing.assert_allclose(info["T_WG"], ref_WG, rtol=0, atol=1e-9)
