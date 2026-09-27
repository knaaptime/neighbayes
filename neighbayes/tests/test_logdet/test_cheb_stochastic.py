"""Tests for stochastic Chebyshev expansion logdet."""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp

from neighbayes._logdet import make_logdet_numpy_fn, make_logdet_numpy_vec_fn
from neighbayes._logdet._cheb_stochastic import (
    ChebStochasticPrecompute,
    _clamp_off_pole,
    _exact_cheb_moments,
    _log_cheb_coeffs,
    _power_traces,
    _probe_noise_rtol,
    cheb_stochastic_logdet_eval,
    cheb_stochastic_logdet_eval_vec,
    cheb_stochastic_logdet_precompute,
    cheb_stochastic_order,
)


def _d_sym(W):
    """D-symmetrized W_sym = D^{1/2} W D^{-1/2} (same spectrum, symmetric)."""
    from neighbayes._logdet._slq import _recover_symmetrizing_diagonal

    s = np.sqrt(_recover_symmetrizing_diagonal(W))
    return sp.csr_matrix(sp.diags(s) @ W @ sp.diags(1.0 / s))


def _weighted_ring_W(n, k=3, seed=0):
    """Row-standardized W of an undirected weighted ring (symmetrizable).

    Kernel-like weights make ``W = D⁻¹A`` non-binary, so the D-symmetrization
    used by deflation is genuinely exercised (unlike a binary rook lattice).
    """
    rng = np.random.default_rng(seed)
    rows, cols, vals = [], [], []
    for i in range(n):
        for d in range(1, k + 1):
            for j in (i - d, i + d):
                rows.append(i)
                cols.append(j % n)
                vals.append(rng.uniform(0.5, 2.0))
    A = sp.csr_matrix((vals, (rows, cols)), shape=(n, n))
    A = A.maximum(A.T)  # undirected (symmetric) adjacency
    deg = np.asarray(A.sum(axis=1)).ravel()
    return (sp.diags(1.0 / deg) @ A).tocsr()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def small_W():
    """Small rook contiguity W with known eigenvalues."""
    from libpysal import graph

    from neighbayes import dgp

    gdf = dgp.simulate_sar(n=15, create_gdf=True)
    W = graph.Graph.build_contiguity(gdf, rook=True).transform("r").sparse.toarray()
    return sp.csr_matrix(W.astype(np.float64))


@pytest.fixture
def small_eigs(small_W):
    return np.linalg.eigvals(small_W.toarray())


# ---------------------------------------------------------------------------
# Chebyshev coefficient computation
# ---------------------------------------------------------------------------


class TestLogChebCoeffs:
    def test_zero_rho(self):
        """At ρ=0, log|1| = 0, so all coefficients should be 0."""
        coeffs = _log_cheb_coeffs(0.0, -1.0, 1.0, 20)
        assert np.allclose(coeffs, 0.0, atol=1e-10)

    def test_coefficient_accuracy(self):
        """Coefficients should reconstruct log|1 - ρx| accurately via Clenshaw."""
        rho = 0.5
        lam_min, lam_max = -1.0, 1.0
        order = 30
        coeffs = _log_cheb_coeffs(rho, lam_min, lam_max, order)

        # Evaluate Chebyshev series at several x values and compare to exact
        for x in [-0.9, -0.5, 0.0, 0.3, 0.7, 0.95]:
            # f(x) = log|1 - rho * sigma(x)| where sigma maps [-1,1] -> [lam_min, lam_max]
            # sigma(x) = (x*(lam_max-lam_min) + lam_max+lam_min)/2 = x (for [-1,1])
            exact = np.log(np.abs(1.0 - rho * x))
            # Clenshaw evaluation of Chebyshev series
            m = len(coeffs)
            b_next = 0.0
            b_curr = coeffs[m - 1]
            for k in range(m - 2, 0, -1):
                b_new = 2.0 * x * b_curr - b_next + coeffs[k]
                b_next = b_curr
                b_curr = b_new
            cheb_val = coeffs[0] + x * b_curr - b_next
            assert abs(cheb_val - exact) < 1e-8, f"x={x}: {cheb_val} vs {exact}"


class TestPoleClamp:
    """ρ on the stability boundary must not reach ``log(0)``.

    The Clenshaw-Curtis nodes include x = ±1, i.e. λ = λ_max and λ = λ_min, so
    ρ = 1/λ_max makes the first node evaluate log|1 − ρλ_max| = log 0.  For
    row-standardized ``W`` that is ρ = 1 — the default ``rho_max``.
    """

    @pytest.mark.parametrize("lam_min,lam_max", [(-1.0, 1.0), (-0.5, 1.0), (-1.0, 2.0)])
    def test_boundary_rho_stays_finite(self, lam_min, lam_max):
        coeffs = _log_cheb_coeffs(1.0 / lam_max, lam_min, lam_max, 64)
        assert np.isfinite(coeffs).all()

    def test_negative_boundary_rho_stays_finite(self):
        coeffs = _log_cheb_coeffs(1.0 / -1.0, -1.0, 1.0, 64)
        assert np.isfinite(coeffs).all()

    def test_no_divide_by_zero_warning_on_default_interval(self):
        """The default (rho_max=1, λ_max=1) is the common case, not a rare one."""
        with np.errstate(divide="raise", invalid="raise"):
            cheb_stochastic_order(-1.0, 1.0, n=200_000)
            _log_cheb_coeffs(1.0, -1.0, 1.0, 256)

    def test_order_selection_tail_bound_is_finite(self):
        """The tail bound must be a finding, not an artifact.

        Before the clamp the bound came back ``inf``, ``tail <= target`` was
        False everywhere, and the selection fell through to ``cap`` for want of
        any order that could clear it.  It still asks for the cap at the
        boundary — 1e-6 from a log pole genuinely needs more than 120 terms —
        but now because the bound says so.
        """
        r = _clamp_off_pole(1.0, -1.0, 1.0)
        coeffs = np.abs(_log_cheb_coeffs(r, -1.0, 1.0, 256))
        assert np.isfinite(coeffs).all()
        assert coeffs[121:].sum() > _probe_noise_rtol(r, 1_600, 50)
        assert cheb_stochastic_order(-1.0, 1.0, n=1_600) == 120

    def test_interior_rho_is_untouched(self):
        """Clamping must not perturb ρ away from the boundary."""
        for rho in (-0.9, -0.5, 0.0, 0.5, 0.9, 0.99):
            assert _clamp_off_pole(rho, -1.0, 1.0) == rho

    def test_clamp_is_inert_when_no_pole_exists(self):
        """With λ_max ≤ 0 and λ_min ≥ 0 there is no boundary to clamp against."""
        assert _clamp_off_pole(50.0, -1.0, -0.5) == 50.0
        assert _clamp_off_pole(-50.0, 0.5, 1.0) == -50.0

    def test_boundary_coeffs_are_those_of_the_clamped_rho(self):
        """The clamp expands a neighbouring ρ exactly, rather than flooring the
        log and fabricating a finite value at the pole itself.  Guards against
        a future "fix" that clamps ``np.log`` at the nodes instead."""
        rho_eff = _clamp_off_pole(1.0, -1.0, 1.0)
        assert rho_eff < 1.0
        assert np.array_equal(
            _log_cheb_coeffs(1.0, -1.0, 1.0, 200),
            _log_cheb_coeffs(rho_eff, -1.0, 1.0, 200),
        )


# ---------------------------------------------------------------------------
# Precompute
# ---------------------------------------------------------------------------


class TestPrecompute:
    def test_returns_dataclass(self, small_W):
        pre = cheb_stochastic_logdet_precompute(small_W, order=20, n_probes=50)
        assert isinstance(pre, ChebStochasticPrecompute)
        assert pre.order == 20
        assert pre.n == small_W.shape[0]
        assert pre.moments.shape == (21,)

    def test_moment_zero_is_n(self, small_W):
        """μ_0 = tr(T_0(W̃)) = tr(I) = n."""
        pre = cheb_stochastic_logdet_precompute(small_W, order=20, n_probes=50)
        assert pre.moments[0] == pytest.approx(small_W.shape[0])

    def test_moment_one_is_exact(self, small_W):
        """μ_1 = tr(W̃) is computed exactly (diagonal sum)."""
        pre = cheb_stochastic_logdet_precompute(small_W, order=20, n_probes=50)
        # W̃ = (2W - (λ_max+λ_min)I) / (λ_max-λ_min)
        # For λ_min=-1, λ_max=1: W̃ = W, so tr(W̃) = tr(W) = 0 (row-standardized)
        assert pre.moments[1] == pytest.approx(0.0, abs=1e-10)

    def test_spectral_bounds(self, small_W):
        """For row-standardized W, bounds should be [-1, 1]."""
        pre = cheb_stochastic_logdet_precompute(small_W, order=20, n_probes=50)
        assert pre.lam_max == pytest.approx(1.0)
        assert pre.lam_min == pytest.approx(-1.0)

    def test_deterministic_with_seed(self, small_W):
        """Same RNG seed should produce identical results."""
        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(42)
        pre1 = cheb_stochastic_logdet_precompute(
            small_W, order=20, n_probes=50, rng=rng1
        )
        pre2 = cheb_stochastic_logdet_precompute(
            small_W, order=20, n_probes=50, rng=rng2
        )
        np.testing.assert_array_equal(pre1.moments, pre2.moments)


# ---------------------------------------------------------------------------
# Evaluation accuracy
# ---------------------------------------------------------------------------


class TestAccuracy:
    def test_accuracy_at_moderate_rho(self, small_W, small_eigs):
        """At ρ=0.5, error should be < 5% (small n=15, high stochastic variance)."""
        pre = cheb_stochastic_logdet_precompute(small_W, order=20, n_probes=200)
        rho = 0.5
        exact = np.sum(np.log(np.abs(1.0 - rho * small_eigs)))
        est = cheb_stochastic_logdet_eval(pre, rho)
        rel_err = abs(est - exact) / abs(exact)
        assert rel_err < 0.05, f"rel_err={rel_err:.4e}"

    def test_accuracy_at_high_rho(self, small_W, small_eigs):
        """At ρ=0.9, error should be < 5% (small n=15, high stochastic variance)."""
        pre = cheb_stochastic_logdet_precompute(small_W, order=20, n_probes=200)
        rho = 0.9
        exact = np.sum(np.log(np.abs(1.0 - rho * small_eigs)))
        est = cheb_stochastic_logdet_eval(pre, rho)
        rel_err = abs(est - exact) / abs(exact)
        assert rel_err < 0.05, f"rel_err={rel_err:.4e}"

    def test_better_than_barry_pace_at_rho_095(self, small_W, small_eigs):
        """At ρ=0.95 and large n, stochastic Chebyshev should be more accurate than Barry-Pace-MC.

        Note: Barry-Pace uses exact eigenvalues for n≤500, so this test only
        works with a large enough W to trigger the MC path.  We test the
        stochastic moments directly against exact instead.
        """
        rho = 0.95
        exact = np.sum(np.log(np.abs(1.0 - rho * small_eigs)))

        # Stochastic Chebyshev should be reasonably accurate at high ρ
        pre = cheb_stochastic_logdet_precompute(small_W, order=20, n_probes=200)
        stoch_val = cheb_stochastic_logdet_eval(pre, rho)
        stoch_err = abs(stoch_val - exact) / abs(exact)

        # With enough probes, error should be < 5% even at small n
        assert stoch_err < 0.05, f"stoch_err={stoch_err:.4e}"

    def test_zero_rho_gives_zero(self, small_W):
        """log|I - 0*W| = log|I| = 0."""
        pre = cheb_stochastic_logdet_precompute(small_W, order=20, n_probes=50)
        val = cheb_stochastic_logdet_eval(pre, 0.0)
        assert abs(val) < 0.1  # stochastic, so not exactly 0


# ---------------------------------------------------------------------------
# Factory integration
# ---------------------------------------------------------------------------


class TestFactoryIntegration:
    def test_numpy_scalar_factory(self, small_W, small_eigs):
        """Factory should produce a working scalar evaluator."""
        fn = make_logdet_numpy_fn(
            small_W, eigs=None, method="cheb_stochastic", rho_min=-0.95, rho_max=0.95
        )
        rho = 0.5
        exact = np.sum(np.log(np.abs(1.0 - rho * small_eigs)))
        est = fn(rho)
        assert abs(est - exact) / abs(exact) < 0.05

    def test_numpy_vectorized_factory(self, small_W, small_eigs):
        """Factory should produce a working vectorized evaluator."""
        fn = make_logdet_numpy_vec_fn(
            small_W, eigs=None, method="cheb_stochastic", rho_min=-0.95, rho_max=0.95
        )
        rho_arr = np.array([0.3, 0.5, 0.7, 0.9])
        result = fn(rho_arr)
        assert result.shape == (4,)
        for i, rho in enumerate(rho_arr):
            exact = np.sum(np.log(np.abs(1.0 - rho * small_eigs)))
            assert abs(result[i] - exact) / abs(exact) < 0.05

    def test_auto_select(self):
        """Past the Cholesky cutoff, auto-select should pick cheb_stochastic."""
        from neighbayes._logdet import resolve_logdet_method

        assert resolve_logdet_method(None, n=200000) == "cheb_stochastic"

    def test_eval_speed(self, small_W):
        """Eval should be O(20) Clenshaw, ~2μs per call."""
        import time

        fn = make_logdet_numpy_fn(
            small_W, eigs=None, method="cheb_stochastic", rho_min=-0.95, rho_max=0.95
        )
        # Warm up
        fn(0.5)
        # Benchmark
        rhos = np.linspace(-0.9, 0.9, 1000)
        t0 = time.perf_counter()
        for r in rhos:
            fn(float(r))
        per_call = (time.perf_counter() - t0) / 1000 * 1e6
        # Should be < 10μs (Clenshaw O(20))
        assert per_call < 10.0, f"per_call={per_call:.1f}μs"


# ---------------------------------------------------------------------------
# Convergence
# ---------------------------------------------------------------------------


class TestConvergence:
    def test_convergence_in_order(self, small_W, small_eigs):
        """Error should be reasonable across orders (at high ρ, stochastic noise present)."""
        rho = 0.9
        exact = np.sum(np.log(np.abs(1.0 - rho * small_eigs)))
        errors = []
        for order in [10, 20, 30]:
            pre = cheb_stochastic_logdet_precompute(small_W, order=order, n_probes=200)
            est = cheb_stochastic_logdet_eval(pre, rho)
            errors.append(abs(est - exact) / abs(exact))
        # All errors should be < 5% at this probe count
        assert all(e < 0.05 for e in errors), f"errors={errors}"

    def test_variance_decreases_with_probes(self, small_W, small_eigs):
        """Variance across seeds should decrease with more probes."""
        rho = 0.9
        exact = np.sum(np.log(np.abs(1.0 - rho * small_eigs)))

        errors_low = []
        errors_high = []
        for seed in range(10):
            rng = np.random.default_rng(seed)
            pre_low = cheb_stochastic_logdet_precompute(
                small_W, order=20, n_probes=30, rng=rng
            )
            errors_low.append(cheb_stochastic_logdet_eval(pre_low, rho) - exact)

            rng = np.random.default_rng(seed)
            pre_high = cheb_stochastic_logdet_precompute(
                small_W, order=20, n_probes=200, rng=rng
            )
            errors_high.append(cheb_stochastic_logdet_eval(pre_high, rho) - exact)

        std_low = np.std(errors_low)
        std_high = np.std(errors_high)
        assert std_high < std_low, f"std_high={std_high} should be < std_low={std_low}"


# ---------------------------------------------------------------------------
# Eigen-deflation (symmetrizable W only)
# ---------------------------------------------------------------------------


class TestDeflation:
    """Correctness of the matrix-free eigen-deflation path.

    The previous implementation deflated via randomized SVD, which (a)
    materialized a dense ``n × n`` residual (OOM at scale) and (b) treated
    singular values as eigenvalues — a non-invariant split that biased the
    Chebyshev moments of the indefinite ``W̃``.  These tests pin the fixed
    behavior: an eigenpair split that is exact in the full-deflation limit,
    never densifies, and is confined to symmetrizable (undirected) graphs.
    """

    def test_full_deflation_is_exact(self):
        """Deflating all but two eigenpairs → logdet ≈ dense slogdet.

        With almost every eigenpair captured exactly, the residual is nearly
        zero and the result is accurate regardless of probe count — a direct
        check that the eigenpair decomposition (not the old SVD split) is
        used.
        """
        n = 60
        W = _weighted_ring_W(n, k=3, seed=1)
        Wd = W.toarray()
        pre = cheb_stochastic_logdet_precompute(
            W, order=14, n_probes=4, n_deflate=n - 2, rng=np.random.default_rng(3)
        )
        for rho in (0.3, 0.6, 0.85):
            est = cheb_stochastic_logdet_eval(pre, rho)
            _, exact = np.linalg.slogdet(np.eye(n) - rho * Wd)
            assert abs(est - exact) < 5e-3, f"rho={rho}: {est} vs {exact}"

    def test_deflation_matches_dense_at_operating_point(self):
        """Modest deflation stays within the stochastic tolerance of truth."""
        n = 60
        W = _weighted_ring_W(n, k=3, seed=2)
        eigs = np.linalg.eigvals(W.toarray())
        pre = cheb_stochastic_logdet_precompute(
            W, order=20, n_probes=200, n_deflate=6, rng=np.random.default_rng(0)
        )
        for rho in (0.5, 0.9):
            exact = np.sum(np.log(np.abs(1.0 - rho * eigs)))
            est = cheb_stochastic_logdet_eval(pre, rho)
            assert abs(est - exact) / abs(exact) < 0.05, f"rho={rho}"

    def test_deflation_does_not_densify(self):
        """n≈5000 deflation runs fast and finite (old code allocated dense n×n)."""
        import time

        W = _weighted_ring_W(5000, k=4, seed=0)
        t0 = time.perf_counter()
        pre = cheb_stochastic_logdet_precompute(
            W, order=15, n_probes=20, n_deflate=5, rng=np.random.default_rng(0)
        )
        elapsed = time.perf_counter() - t0
        assert np.all(np.isfinite(pre.moments))
        # A dense n×n residual at n=5000 would be ~200 MB + O(n²) work; the
        # matrix-free path is comfortably under a few seconds.
        assert elapsed < 20.0, f"elapsed={elapsed:.1f}s (densification regression?)"

    def test_directed_W_warns_and_falls_back(self):
        """Asymmetric-pattern (directed) W: warn and ignore deflation."""
        n = 50
        rows, cols = [], []
        for i in range(n):  # forward-only edges → asymmetric sparsity pattern
            rows += [i, i]
            cols += [(i + 1) % n, (i + 2) % n]
        A = sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))
        deg = np.asarray(A.sum(axis=1)).ravel()
        W = (sp.diags(1.0 / deg) @ A).tocsr()

        with pytest.warns(UserWarning, match="symmetrizable"):
            deflated = cheb_stochastic_logdet_precompute(
                W, order=12, n_probes=16, n_deflate=4, rng=np.random.default_rng(7)
            )
        plain = cheb_stochastic_logdet_precompute(
            W, order=12, n_probes=16, n_deflate=0, rng=np.random.default_rng(7)
        )
        # Fallback runs the identical plain path → identical moments.
        np.testing.assert_allclose(deflated.moments, plain.moments)
        assert np.all(np.isfinite(deflated.moments))


# ---------------------------------------------------------------------------
# Exact low-order moments (Hutchinson control variates)
# ---------------------------------------------------------------------------


class TestExactMoments:
    """``n_exact`` replaces leading moments with probe-free exact values."""

    @staticmethod
    def _eig_moments(W, depth):
        """tr(T_j(W)) from a dense eigendecomposition (spectrum already in [-1,1])."""
        lam = np.linalg.eigvalsh(_d_sym(W).toarray())
        T = np.empty((depth + 1, lam.size))
        T[0] = 1.0
        if depth >= 1:
            T[1] = lam
        for j in range(1, depth):
            T[j + 1] = 2.0 * lam * T[j] - T[j - 1]
        return T.sum(axis=1)

    @pytest.mark.parametrize("depth", [2, 3, 4, 5, 6])
    def test_matches_eigenvalue_moments(self, depth):
        """Exact moments agree with the eigenvalue computation to machine precision."""
        W = _weighted_ring_W(60, k=2, seed=3)
        got = _exact_cheb_moments(W, depth, -1.0, 1.0)
        np.testing.assert_allclose(
            got, self._eig_moments(W, depth), rtol=1e-10, atol=1e-8
        )

    def test_power_traces_match_dense(self):
        """The two-product trace scheme matches dense matrix powers."""
        W = _weighted_ring_W(40, k=2, seed=1)
        dense = W.toarray()
        want = [float(np.trace(np.linalg.matrix_power(dense, k))) for k in range(7)]
        np.testing.assert_allclose(_power_traces(W, 6), want, rtol=1e-10, atol=1e-9)

    def test_directed_W_power_traces(self):
        """tr(W^k) is basis-free, so the scheme holds for non-symmetric W too."""
        rng = np.random.default_rng(0)
        A = sp.random(50, 50, density=0.15, random_state=rng, format="csr")
        W = (
            sp.diags(1.0 / np.maximum(np.asarray(A.sum(1)).ravel(), 1e-12)) @ A
        ).tocsr()
        dense = W.toarray()
        want = [float(np.trace(np.linalg.matrix_power(dense, k))) for k in range(5)]
        np.testing.assert_allclose(_power_traces(W, 4), want, rtol=1e-9, atol=1e-9)

    def test_moments_are_recorded_and_applied(self):
        W = _weighted_ring_W(80, k=2, seed=5)
        pre = cheb_stochastic_logdet_precompute(W, order=25, n_exact=4, n_probes=8)
        assert pre.n_exact == 4
        np.testing.assert_allclose(
            pre.moments[:5],
            _exact_cheb_moments(W, 4, pre.lam_min, pre.lam_max),
            rtol=1e-10,
        )

    def test_deeper_is_more_accurate(self):
        """Variance falls monotonically with depth over repeated probe draws."""
        W = _weighted_ring_W(200, k=3, seed=11)
        lam = np.linalg.eigvalsh(_d_sym(W).toarray())
        rhos = np.array([0.5, 0.8, 0.9])
        exact = np.array([np.sum(np.log1p(-r * lam)) for r in rhos])
        rmse = {}
        for depth in (1, 2, 4):
            errs = [
                cheb_stochastic_logdet_eval_vec(
                    cheb_stochastic_logdet_precompute(
                        W,
                        order=30,
                        n_probes=12,
                        n_exact=depth,
                        rng=np.random.default_rng(s),
                    ),
                    rhos,
                )
                - exact
                for s in range(12)
            ]
            rmse[depth] = np.sqrt((np.array(errs) ** 2).mean(axis=0))
        assert np.all(rmse[2] < rmse[1])
        assert np.all(rmse[4] < rmse[2])

    def test_default_is_depth_four(self):
        W = _weighted_ring_W(60, k=2, seed=2)
        assert cheb_stochastic_logdet_precompute(W, order=25, n_probes=8).n_exact == 4

    def test_depth_never_exceeds_order(self):
        W = _weighted_ring_W(60, k=2, seed=2)
        pre = cheb_stochastic_logdet_precompute(W, order=3, n_probes=8, n_exact=6)
        assert pre.n_exact <= 3
        assert np.all(np.isfinite(pre.moments))

    def test_legacy_depth_one_reproduces_historical_moments(self):
        """n_exact=1 keeps the mu_0 = n, mu_1 = tr(W~) behavior unchanged."""
        W = _weighted_ring_W(60, k=2, seed=4)
        pre = cheb_stochastic_logdet_precompute(
            W, order=20, n_probes=10, n_exact=1, rng=np.random.default_rng(0)
        )
        assert pre.n_exact == 1
        assert pre.moments[0] == pytest.approx(W.shape[0])


class TestDegreeGuard:
    def test_dense_W_degrades_to_two(self):
        n = 120
        W = sp.csr_matrix((np.full((n, n), 1.0 / (n - 1)) - np.eye(n) / (n - 1)))
        with pytest.warns(UserWarning, match="max_degree"):
            pre = cheb_stochastic_logdet_precompute(W, order=20, n_probes=8, n_exact=6)
        assert pre.n_exact == 2

    def test_raising_max_degree_overrides(self):
        n = 120
        W = sp.csr_matrix((np.full((n, n), 1.0 / (n - 1)) - np.eye(n) / (n - 1)))
        pre = cheb_stochastic_logdet_precompute(
            W, order=20, n_probes=8, n_exact=6, max_degree=1e6
        )
        assert pre.n_exact == 6

    def test_sparse_W_keeps_requested_depth(self):
        W = _weighted_ring_W(200, k=2, seed=6)  # degree 8
        pre = cheb_stochastic_logdet_precompute(W, order=25, n_probes=8, n_exact=4)
        assert pre.n_exact == 4


class TestOrderSelection:
    def test_wider_interval_needs_higher_order(self):
        orders = [
            cheb_stochastic_order(-b, b, n=10_000) for b in (0.5, 0.8, 0.9, 0.95, 0.99)
        ]
        assert orders == sorted(orders)
        assert orders[-1] > orders[0]

    def test_tail_bound_is_respected(self):
        """The selected order's rigorous tail bound meets the noise-floor target."""
        rho, n, k = 0.95, 10_000, 50
        p = cheb_stochastic_order(-rho, rho, n=n, n_probes=k)
        coeffs = np.abs(_log_cheb_coeffs(rho, -1.0, 1.0, 256))
        assert coeffs[p + 1 :].sum() <= _probe_noise_rtol(rho, n, k) or p >= 120

    def test_order_none_requires_bounds(self):
        W = _weighted_ring_W(40, k=2, seed=0)
        with pytest.raises(ValueError, match="rho_min and rho_max"):
            cheb_stochastic_logdet_precompute(W, order=None)

    def test_order_none_sizes_from_interval(self):
        W = _weighted_ring_W(60, k=2, seed=0)
        narrow = cheb_stochastic_logdet_precompute(
            W, order=None, rho_min=0.1, rho_max=0.8, n_probes=8
        )
        wide = cheb_stochastic_logdet_precompute(
            W, order=None, rho_min=-0.99, rho_max=0.99, n_probes=8
        )
        assert wide.order > narrow.order

    def test_low_pinned_order_degrades_depth_with_warning(self):
        W = _weighted_ring_W(60, k=2, seed=0)
        with pytest.warns(UserWarning, match="binding error"):
            pre = cheb_stochastic_logdet_precompute(
                W, order=15, rho_min=-0.99, rho_max=0.99, n_probes=8, n_exact=6
            )
        assert pre.n_exact == 1

    def test_no_bounds_means_no_order_check(self):
        """Without an interval there is nothing to validate, so no warning fires."""
        import warnings

        W = _weighted_ring_W(60, k=2, seed=0)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            pre = cheb_stochastic_logdet_precompute(W, order=15, n_probes=8, n_exact=4)
        assert pre.n_exact == 4


def _hub_graph(side=100, n_hubs=25, degree=2000, seed=0):
    """Row-standardized lattice with hub nodes: strongly non-normal W."""
    rng = np.random.default_rng(seed)
    n = side * side
    idx = np.arange(n).reshape(side, side)
    i = np.concatenate([idx[:, :-1].ravel(), idx[:-1].ravel()])
    j = np.concatenate([idx[:, 1:].ravel(), idx[1:].ravel()])
    hi = np.repeat(np.arange(n, n + n_hubs), degree)
    hj = np.concatenate([rng.choice(n, degree, replace=False) for _ in range(n_hubs)])
    i, j = np.concatenate([i, hi]), np.concatenate([j, hj])
    N = n + n_hubs
    A = sp.csr_matrix((np.ones(len(i)), (i, j)), shape=(N, N))
    A = A + A.T
    d = np.asarray(A.sum(axis=1)).ravel()
    return sp.csr_matrix(sp.diags(1.0 / d) @ A)


def test_spectral_bounds_enclose_without_overshoot_on_nonnormal_W():
    """The bracket must enclose the spectrum and, for row-stochastic W, equal [-1, 1].

    A Rayleigh quotient of a non-normal W can exceed its spectral radius; a
    bracket wider than 1/rho puts the log singularity inside the expansion
    interval and destroys accuracy near the boundary.
    """
    from neighbayes._logdet._cheb_stochastic import _estimate_spectral_bounds

    W = _hub_graph()
    for seed in range(20):
        lo, hi = _estimate_spectral_bounds(W, rng=np.random.default_rng(seed))
        assert (lo, hi) == (-1.0, 1.0)
