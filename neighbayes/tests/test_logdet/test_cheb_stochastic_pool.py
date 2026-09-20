"""Tests for probe pools: the stochastic Chebyshev probe count sized by its own spread."""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp

from neighbayes._logdet._cheb_stochastic import (
    ProbeCheck,
    _log_cheb_coeffs,
    _log_cheb_coeffs_vec,
    _pool_probes,
    cheb_stochastic_logdet_eval,
    cheb_stochastic_logdet_precompute,
    cheb_stochastic_pool,
    grow_pool,
    pool_precompute,
    pool_probe_bias,
    probes_needed,
    size_pool,
)


def _ring_W(n=400, k=3, seed=0):
    """Row-standardized W of an undirected weighted ring."""
    rng = np.random.default_rng(seed)
    rows, cols, vals = [], [], []
    for i in range(n):
        for d in range(1, k + 1):
            for j in (i - d, i + d):
                rows.append(i)
                cols.append(j % n)
                vals.append(rng.uniform(0.5, 2.0))
    A = sp.csr_matrix((vals, (rows, cols)), shape=(n, n))
    A = A.maximum(A.T)
    deg = np.asarray(A.sum(axis=1)).ravel()
    return (sp.diags(1.0 / deg) @ A).tocsr()


@pytest.fixture(scope="module")
def ring():
    W = _ring_W()
    eigs = np.linalg.eigvals(W.toarray())
    return W, eigs


class _FixedProbes:
    """Stands in for a Generator so the shipped precompute sees given probes."""

    def __init__(self, U):
        self.U = U

    def standard_normal(self, shape):
        assert shape == self.U.shape
        return self.U.copy()


class TestPool:
    def test_same_estimator_as_shipped_on_same_probes(self, ring):
        W, _ = ring
        pool = cheb_stochastic_pool(W, 20, rho_min=-0.99, rho_max=0.99, seed=3)
        U = _pool_probes(3, 0, 20, W.shape[0])
        shipped = cheb_stochastic_logdet_precompute(
            W,
            order=pool.order,
            n_probes=20,
            lam_min=pool.lam_min,
            lam_max=pool.lam_max,
            rho_min=-0.99,
            rho_max=0.99,
            rng=_FixedProbes(U),
        )
        np.testing.assert_allclose(
            pool_precompute(pool).moments, shipped.moments, rtol=1e-13, atol=1e-10
        )
        assert pool_precompute(pool).n_exact == shipped.n_exact

    def test_growth_path_does_not_change_the_pool(self, ring):
        W, _ = ring
        direct = cheb_stochastic_pool(W, 60, seed=1)
        stepped = grow_pool(grow_pool(cheb_stochastic_pool(W, 12, seed=1), 25), 60)
        assert stepped.n_probes == 60
        np.testing.assert_array_equal(stepped.probe_moments, direct.probe_moments)

    def test_grow_to_fewer_probes_is_a_no_op(self, ring):
        W, _ = ring
        pool = cheb_stochastic_pool(W, 20, seed=0)
        assert grow_pool(pool, 12) is pool
        assert grow_pool(pool, 20) is pool

    def test_order_is_sized_for_the_largest_pool(self, ring):
        W, _ = ring
        small = cheb_stochastic_pool(W, 12, max_probes=12, rho_min=-0.99, rho_max=0.99)
        large = cheb_stochastic_pool(W, 12, max_probes=400, rho_min=-0.99, rho_max=0.99)
        assert large.order >= small.order

    def test_more_probes_are_more_accurate(self, ring):
        W, eigs = ring
        exact = float(np.sum(np.log(np.abs(1.0 - 0.9 * eigs))))
        err = {12: [], 150: []}
        for seed in range(10):
            pool = cheb_stochastic_pool(W, 12, rho_min=-0.99, rho_max=0.99, seed=seed)
            for s in err:
                pre = pool_precompute(grow_pool(pool, s))
                err[s].append(cheb_stochastic_logdet_eval(pre, 0.9) - exact)
        rmse = {s: np.sqrt(np.mean(np.square(e))) for s, e in err.items()}
        assert rmse[150] < rmse[12]


def test_vectorized_coefficients_match_scalar():
    rhos = np.array([-1.0, -0.99, -0.3, 0.0, 0.5, 0.9, 0.999, 1.0])
    vec = _log_cheb_coeffs_vec(rhos, -1.0, 1.0, 40)
    for i, r in enumerate(rhos):
        np.testing.assert_allclose(
            vec[i], _log_cheb_coeffs(float(r), -1.0, 1.0, 40), rtol=1e-11, atol=1e-12
        )


class TestProbeBias:
    MU, SD = 0.85, 0.01

    def _draws(self, size=4000):
        return np.random.default_rng(7).normal(self.MU, self.SD, size)

    def test_predicted_spread_matches_realized_bias(self, ring):
        """std(q)/√s prices the posterior-mean bias against the exact logdet."""
        from neighbayes._logdet._cheb_stochastic import cheb_stochastic_logdet_eval_vec

        W, eigs = ring
        grid = np.linspace(self.MU - 6 * self.SD, self.MU + 6 * self.SD, 241)
        lp0 = -0.5 * ((grid - self.MU) / self.SD) ** 2
        exact = np.sum(np.log(np.abs(1.0 - grid[:, None] * eigs[None, :])), axis=1).real
        w0 = np.exp(lp0 - lp0.max())
        w0 /= w0.sum()
        mean0 = w0 @ grid
        sd0 = np.sqrt(w0 @ (grid - mean0) ** 2)
        s, realized, predicted = 20, [], []
        for seed in range(40):
            pool = cheb_stochastic_pool(W, s, rho_min=-0.99, rho_max=0.99, seed=seed)
            err = cheb_stochastic_logdet_eval_vec(pool_precompute(pool), grid) - exact
            lp = lp0 + err
            w = np.exp(lp - lp.max())
            w /= w.sum()
            realized.append((w @ grid - mean0) / sd0)
            q = pool_probe_bias(pool, self._draws())
            predicted.append(np.std(q, ddof=1) / np.sqrt(s))
        ratio = np.std(realized, ddof=1) / np.mean(predicted)
        assert 0.7 < ratio < 1.4, ratio

    def test_panel_bias_scales_with_T(self, ring):
        W, _ = ring
        pool = cheb_stochastic_pool(W, 12, seed=0)
        q1 = pool_probe_bias(pool, self._draws(), T=1)
        q3 = pool_probe_bias(pool, self._draws(), T=3)
        np.testing.assert_allclose(q3, 3.0 * q1, rtol=1e-12)

    def test_draws_without_spread_cannot_price_probes(self, ring):
        W, _ = ring
        pool = cheb_stochastic_pool(W, 12, seed=0)
        assert pool_probe_bias(pool, np.full(50, 0.5)) is None
        sized, check = size_pool(pool, np.full(50, 0.5))
        assert sized is pool and check is None


class TestSizePool:
    def _draws(self):
        return np.random.default_rng(7).normal(0.9, 0.01, 4000)

    def test_grows_until_the_target_is_met(self, ring):
        W, _ = ring
        pool = cheb_stochastic_pool(W, 12, rho_min=-0.99, rho_max=0.99, seed=0)
        q = pool_probe_bias(pool, self._draws())
        bar = 2.0 * np.sqrt(2 / np.pi) * np.std(q, ddof=1) / np.sqrt(60)
        sized, check = size_pool(pool, self._draws(), bar=bar, z=2.0)
        assert isinstance(check, ProbeCheck)
        assert sized.n_probes > 12 and sized.n_probes == check.n_probes
        assert check.bias <= 1.3 * bar / 2.0
        assert not check.capped
        np.testing.assert_array_equal(sized.probe_moments[:, :12], pool.probe_moments)

    def test_loose_target_keeps_the_pool(self, ring):
        W, _ = ring
        pool = cheb_stochastic_pool(W, 12, seed=0)
        sized, check = size_pool(pool, self._draws(), bar=10.0)
        assert sized is pool and check.n_probes == 12

    def test_cap_is_reported(self, ring):
        W, _ = ring
        pool = cheb_stochastic_pool(W, 12, seed=0)
        sized, check = size_pool(pool, self._draws(), bar=1e-6, max_probes=16)
        assert sized.n_probes == 16 and check.capped

    def test_probes_needed_inverts_the_expected_bias(self):
        q = np.array([-1.0, 1.0, -1.0, 1.0])
        spread = np.std(q, ddof=1)
        need = probes_needed(q, bar=0.1, z=2.0)
        assert np.sqrt(2 / np.pi) * spread / np.sqrt(need) <= 0.05
        assert np.sqrt(2 / np.pi) * spread / np.sqrt(need - 1) > 0.05
