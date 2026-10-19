"""Tests for the logit-with-random-effects Gibbs sampler (NumPy path).

Covers the PG draw and the pointwise log-likelihood; the doubly collapsed
density of ``(r, σ)`` and the conditional moments against dense linear algebra,
for SAR and SEM; the invariants of the ridge and scale moves; chain output;
and SEM parameter recovery.
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp
from scipy.stats import multivariate_normal

from neighbayes.samplers._utils._polyagamma import sample_polyagamma
from neighbayes.samplers._utils._slice import SliceWidthState
from neighbayes.samplers._utils._spatial_normal import CholmodFactor
from neighbayes.samplers.logit._core import (
    LogitGibbsCache,
    LogitGibbsState,
    LogitREGibbsPriors,
    SEMLogitGibbsCache,
    SEMLogitGibbsPriors,
    SEMLogitGibbsState,
    _attenuation,
    _collapsed_parts,
    _logit_loglik_pointwise,
    _ridge_move,
    _sample_eta,
    _sample_omega,
    _sample_r_sigma_beta,
    _sar_design,
    _scale_move,
    _sem_design,
    run_chain,
    run_chain_sem,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ring(n):
    W = np.zeros((n, n))
    for i in range(n):
        W[i, (i + 1) % n] = W[i, (i - 1) % n] = 0.5
    return W


def _problem(n=20, k=3, rho=0.3, sigma=0.8, sem=False, seed=42):
    rng = np.random.default_rng(seed)
    X = np.column_stack([np.ones(n), rng.standard_normal((n, k - 1))])
    beta = np.array([0.5, -1.0, 0.8])[:k]
    W_dense = _ring(n)
    W = sp.csr_matrix(W_dense)
    A = sp.eye(n, format="csc") - rho * sp.csc_matrix(W)
    nu = sigma * rng.standard_normal(n)
    if sem:
        eta = X @ beta + sp.linalg.spsolve(A, nu)
    else:
        eta = sp.linalg.spsolve(A, X @ beta + nu)
    y = rng.binomial(1, 1.0 / (1.0 + np.exp(-eta))).astype(float)
    return y, X, W, W_dense, beta, eta


def _logdet_fn(W_dense):
    ev = np.linalg.eigvals(W_dense)
    return lambda r: float(np.sum(np.log(np.abs(1.0 - r * ev))))


def _cache(W, X, sem=False):
    n = W.shape[0]
    W_sym, WtW = W + W.T, W.T @ W
    factor = CholmodFactor(sp.eye(n, format="csr") + 0.5 * W_sym + 0.25 * WtW)
    common = dict(
        W_sparse=W,
        XtX=X.T @ X,
        logdet_fn=_logdet_fn(W.toarray()),
        cholmod_factor=factor,
        W_sym=W_sym,
        WtW=WtW,
    )
    if sem:
        return SEMLogitGibbsCache(
            lam_lower=-0.999,
            lam_upper=0.999,
            lam_slice_width_state=SliceWidthState(w=0.2),
            **common,
        )
    return LogitGibbsCache(
        rho_lower=-0.999,
        rho_upper=0.999,
        WtX=np.asarray(W.T @ X),
        rho_slice_width_state=SliceWidthState(w=0.2),
        **common,
    )


PRIORS = LogitREGibbsPriors(beta_mu=0.0, beta_sigma=2.0)
SEM_PRIORS = SEMLogitGibbsPriors(beta_mu=0.0, beta_sigma=2.0)


def _init(y, X, rng, sem=False, sigma=1.0):
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    eta = X @ beta
    omega = sample_polyagamma(np.ones(len(y)), eta, rng=rng)
    if sem:
        return SEMLogitGibbsState(eta=eta, beta=beta, lam=0.0, omega=omega, sigma=sigma)
    return LogitGibbsState(eta=eta, beta=beta, rho=0.0, omega=omega, sigma=sigma)


# ---------------------------------------------------------------------------
# PG draw and log-likelihood
# ---------------------------------------------------------------------------


class TestSampleOmega:
    def test_shape_and_positive(self):
        omega = _sample_omega(
            np.array([-5.0, -1.0, 0.0, 1.0, 5.0]), rng=np.random.default_rng(1)
        )
        assert omega.shape == (5,) and np.all(omega > 0)

    def test_mean_at_zero_tilting(self):
        omega = _sample_omega(np.zeros(10000), rng=np.random.default_rng(2))
        assert abs(np.mean(omega) - 0.25) < 0.02  # E[PG(1, 0)] = 1/4


class TestLogitLoglik:
    def test_values(self):
        ll = _logit_loglik_pointwise(
            np.array([0.0, 1.0, 1.0, 0.0]), np.array([0.0, 0.0, 40.0, -40.0])
        )
        np.testing.assert_allclose(ll, [-np.log(2), -np.log(2), 0.0, 0.0], atol=1e-12)


# ---------------------------------------------------------------------------
# The collapsed algebra against dense linear algebra (σ ≠ 1)
# ---------------------------------------------------------------------------
#
# Given ω, the PG working likelihood is that of z = κ/ω ~ N(η, Ω⁻¹), so every
# collapsed density must differ from a dense Gaussian marginal of z by a
# constant.


def _working(n, seed=3):
    rng = np.random.default_rng(seed)
    omega = rng.uniform(0.1, 0.3, n)
    y = rng.binomial(1, 0.5, n).astype(float)
    return y, omega, (y - 0.5) / omega


def _filter(Wd, r, sem):
    """``S = A⁻¹`` and the design's map of X into η, ``SX`` (SAR) or ``X``."""
    S = np.linalg.inv(np.eye(Wd.shape[0]) - r * Wd)
    return S, (lambda X: X) if sem else (lambda X: S @ X)


SEM_CASES = pytest.mark.parametrize("sem", [False, True], ids=["sar", "sem"])


@SEM_CASES
def test_collapsed_density_matches_dense_marginal(sem):
    y, X, W, Wd, _, _ = _problem(sem=sem)
    n, k = X.shape
    cache = _cache(W, X, sem=sem)
    design = (_sem_design if sem else _sar_design)(X, cache)
    priors = SEM_PRIORS if sem else PRIORS
    yw, omega, z = _working(n)
    gaps = []
    for r, sigma in ((-0.5, 0.7), (0.0, 1.0), (0.3, 0.2), (0.7, 2.5), (0.9, 1.3)):
        S, into = _filter(Wd, r, sem)
        cov = np.diag(1 / omega) + sigma**2 * S @ S.T + 4.0 * into(X) @ into(X).T
        exact = multivariate_normal(np.zeros(n), cov).logpdf(z)
        ours, _, _ = _collapsed_parts(r, sigma, omega, yw - 0.5, design, cache, priors)
        gaps.append(exact - ours)
    # The gap is the PG working likelihood's normalizer, free of (r, σ).
    np.testing.assert_allclose(gaps, gaps[0], atol=1e-8)


@SEM_CASES
def test_beta_moments_integrate_eta_out(sem):
    y, X, W, Wd, _, _ = _problem(sem=sem)
    n, k = X.shape
    cache = _cache(W, X, sem=sem)
    design = (_sem_design if sem else _sar_design)(X, cache)
    b0, sd = np.array([0.2, -0.1, 0.3]), np.sqrt(2.0)
    cls = SEMLogitGibbsPriors if sem else LogitREGibbsPriors
    priors = cls(beta_mu=b0, beta_sigma=sd)
    yw, omega, z = _working(n, seed=4)
    r, sigma = 0.4, 1.3
    _, Lb, rhs_b = _collapsed_parts(r, sigma, omega, yw - 0.5, design, cache, priors)
    S, into = _filter(Wd, r, sem)
    Ci = np.linalg.inv(np.diag(1 / omega) + sigma**2 * S @ S.T)
    U = into(X)
    prec = U.T @ Ci @ U + np.eye(k) / sd**2
    mean = np.linalg.solve(prec, U.T @ Ci @ z + b0 / sd**2)
    np.testing.assert_allclose(Lb @ Lb.T, prec, rtol=1e-8)
    np.testing.assert_allclose(np.linalg.solve(Lb @ Lb.T, rhs_b), mean, atol=1e-8)


@SEM_CASES
def test_eta_conditional_mean(sem):
    y, X, W, Wd, beta, _ = _problem(sem=sem)
    n = X.shape[0]
    cache = _cache(W, X, sem=sem)
    design = (_sem_design if sem else _sar_design)(X, cache)
    yw, omega, _ = _working(n, seed=5)
    r, sigma = 0.5, 0.6
    A = np.eye(n) - r * Wd
    D = A @ X if sem else X
    P = A.T @ A / sigma**2 + np.diag(omega)
    mean = np.linalg.solve(P, A.T @ D @ beta / sigma**2 + (yw - 0.5))
    rng = np.random.default_rng(0)
    draws = np.array(
        [
            _sample_eta(r, sigma, beta, omega, yw, design, cache, rng=rng)
            for _ in range(4000)
        ]
    )
    np.testing.assert_allclose(
        draws.mean(0), mean, atol=4 * np.sqrt(np.diag(np.linalg.inv(P)) / 4000).max()
    )


# ---------------------------------------------------------------------------
# The moves on the exact likelihood
# ---------------------------------------------------------------------------


def test_ridge_move_holds_standardized_effects_and_attenuated_beta():
    y, X, W, _, beta, eta = _problem(sem=True)
    m = X @ beta
    e, b, s = _ridge_move(eta, m, beta, 0.8, y, SEM_PRIORS, np.random.default_rng(0))
    assert s != 0.8
    np.testing.assert_allclose(b / _attenuation(s), beta / _attenuation(0.8))
    np.testing.assert_allclose((e - X @ b) / s, (eta - m) / 0.8, atol=1e-12)


def test_scale_move_holds_the_ratios():
    y, X, _, _, beta, eta = _problem()
    e, b, s = _scale_move(eta, beta, 0.8, y, PRIORS, np.random.default_rng(0))
    c = s / 0.8
    assert c != 1.0
    np.testing.assert_allclose(e, c * eta)
    np.testing.assert_allclose(b, c * beta)


# ---------------------------------------------------------------------------
# Blocks and chains
# ---------------------------------------------------------------------------


@SEM_CASES
def test_collapsed_block_stays_in_bounds(sem):
    y, X, W, _, _, eta = _problem(sem=sem)
    cache = _cache(W, X, sem=sem)
    r, sigma, b = _sample_r_sigma_beta(
        0.3, 0.8, np.ones(len(y)), y, X,
        (_sem_design if sem else _sar_design)(X, cache), cache,
        SEM_PRIORS if sem else PRIORS, (-0.999, 0.999), SliceWidthState(w=0.2),
        rng=np.random.default_rng(0), tuning=True,
    )  # fmt: skip
    assert -0.999 <= r <= 0.999 and sigma > 0
    assert b.shape == (3,) and np.all(np.isfinite(b))


@pytest.mark.parametrize("sem", [False, True])
def test_chain_output(sem):
    y, X, W, _, _, _ = _problem(sem=sem)
    n, k = X.shape
    rng = np.random.default_rng(0)
    run = run_chain_sem if sem else run_chain
    out = run(
        y=y, X=X, W_sparse=W, priors=SEM_PRIORS if sem else PRIORS,
        cache=_cache(W, X, sem=sem), init=_init(y, X, rng, sem=sem),
        draws=30, tune=30, rng=rng, return_eta=True,
    )  # fmt: skip
    spatial = out["lam" if sem else "rho"]
    assert spatial.shape == (30,) and np.all(np.abs(spatial) < 1.0)
    assert out["beta"].shape == (30, k)
    assert out["sigma"].shape == (30,) and np.all(out["sigma"] > 0)
    assert out["log_lik"].shape == (30, n) and out["eta"].shape == (30, n)


# ---------------------------------------------------------------------------
# Recovery
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_sem_logit_recovery():
    """The truth lies in the posterior's bulk for what the data identify.

    That is λ and the attenuated ``β/g(σ)``, not β and σ apart.  The effects
    are strong (σ = 2): at the simulator's default σ = 0.5 they barely move
    binary outcomes and λ's posterior is close to its prior.
    """
    from neighbayes.dgp import simulate_sem_logit

    sd, lam, beta = 2.0, 0.7, np.array([0.3, 1.0])
    out = simulate_sem_logit(
        n_side=12, lam=lam, beta=beta, seed=42, unit_effects="random",
        unit_effect_sd=sd,
    )  # fmt: skip
    y, X, W = out["y"], out["X"], out["W_sparse"]
    rng = np.random.default_rng(0)
    res = run_chain_sem(
        y=y, X=X, W_sparse=W, priors=SEM_PRIORS, cache=_cache(W, X, sem=True),
        init=_init(y, X, rng, sem=True), draws=2000, tune=1000, rng=rng,
    )  # fmt: skip
    assert abs(res["lam"].mean() - lam) < 2 * res["lam"].std()
    att = res["beta"] / _attenuation(res["sigma"])[:, None]
    assert np.all(np.abs(att.mean(0) - beta / _attenuation(sd)) < 2 * att.std(0))
