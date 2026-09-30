"""Tests for Cholesky-Chebyshev log-determinant."""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp

from neighbayes._logdet import make_logdet_numpy_fn, make_logdet_numpy_vec_fn
from neighbayes._logdet._chebyshev import cheb_order_for_tolerance
from neighbayes._logdet._chol_cheb import (
    CholChebPrecompute,
    _d_symmetrize,
    chol_cheb_logdet_eval,
    chol_cheb_logdet_eval_vec,
    chol_cheb_logdet_precompute,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def small_W():
    """Small rook contiguity W with known eigenvalues."""
    from libpysal import graph

    from neighbayes import dgp

    gdf = dgp.simulate_sar(n_side=20, create_gdf=True)
    W = graph.Graph.build_contiguity(gdf, rook=True).transform("r").sparse.toarray()
    return sp.csr_matrix(W.astype(np.float64))


@pytest.fixture
def small_eigs(small_W):
    return np.linalg.eigvals(small_W.toarray())


# ---------------------------------------------------------------------------
# D-symmetrization
# ---------------------------------------------------------------------------


class TestDSymmetrize:
    def test_symmetric(self, small_W):
        """W_sym should be symmetric."""
        W_sym = _d_symmetrize(small_W)
        dense = W_sym.toarray()
        assert np.allclose(dense, dense.T, atol=1e-12)

    def test_preserves_eigenvalues(self, small_W):
        """W_sym should have the same eigenvalues as W."""
        W_sym = _d_symmetrize(small_W)
        eigs_W = np.sort(np.linalg.eigvals(small_W.toarray()).real)
        eigs_sym = np.sort(np.linalg.eigvals(W_sym.toarray()).real)
        assert np.allclose(eigs_W, eigs_sym, atol=1e-10)

    def test_spectrum_in_unit_circle(self, small_W):
        """Row-standardized W should have eigenvalues in [-1, 1]."""
        eigs = np.linalg.eigvals(small_W.toarray()).real
        assert np.all(np.abs(eigs) <= 1.0 + 1e-10)


# ---------------------------------------------------------------------------
# Precompute
# ---------------------------------------------------------------------------


class TestPrecompute:
    def test_returns_precompute(self, small_W):
        pre = chol_cheb_logdet_precompute(small_W, order=10)
        assert isinstance(pre, CholChebPrecompute)
        assert pre.order == 10
        assert pre.n == small_W.shape[0]
        assert pre.coeffs.shape == (10,)

    def test_custom_interval(self, small_W):
        pre = chol_cheb_logdet_precompute(small_W, order=8, rho_min=0.2, rho_max=0.7)
        assert pre.rho_min == 0.2
        assert pre.rho_max == 0.7

    def test_accepts_dense(self, small_W):
        """Should accept dense input as well as sparse."""
        pre_dense = chol_cheb_logdet_precompute(small_W.toarray(), order=8)
        pre_sparse = chol_cheb_logdet_precompute(small_W, order=8)
        assert np.allclose(pre_dense.coeffs, pre_sparse.coeffs, atol=1e-8)


# ---------------------------------------------------------------------------
# Adaptive order selection
# ---------------------------------------------------------------------------


class TestAdaptiveOrder:
    """Order selection is driven by the Bernstein-ellipse convergence rate.

    The property that matters is that the order tracks the interval's *distance
    to the ρ = ±1 singularities*, not its width.  The width-keyed lookup table
    this replaced could not tell those apart, so it returned the same 15 nodes
    for the applied default ``[0.1, 0.8]`` and for a post-warmup window an
    order of magnitude narrower.
    """

    def test_order_rises_as_interval_approaches_singularities(self):
        """Wider intervals, closer to ±1, must cost more nodes."""
        orders = [
            cheb_order_for_tolerance(0.55, 0.65, 10_000),
            cheb_order_for_tolerance(0.1, 0.8, 10_000),
            cheb_order_for_tolerance(-0.5, 0.95, 10_000),
            cheb_order_for_tolerance(-0.95, 0.95, 10_000),
            cheb_order_for_tolerance(-0.99, 0.99, 10_000),
        ]
        assert orders == sorted(orders)
        assert len(set(orders)) == len(orders), "each interval should differ"

    def test_narrow_interval_is_much_cheaper_than_the_applied_default(self):
        """A post-warmup-width window costs a fraction of ``[0.1, 0.8]``.

        This is the whole point of the rule: the old table returned 15 for both.
        """
        assert cheb_order_for_tolerance(
            0.55, 0.65, 10_000
        ) * 2 < cheb_order_for_tolerance(0.1, 0.8, 10_000)

    def test_order_is_nearly_independent_of_n(self):
        """The default target is relative to ``|J| ~ O(n)``, so ``m`` barely moves."""
        orders = {
            cheb_order_for_tolerance(0.1, 0.8, n) for n in (20, 400, 10_000, 60_000)
        }
        assert max(orders) - min(orders) <= 1

    def test_auto_order_used_when_none(self, small_W):
        """When order=None, the precompute should auto-select based on interval."""
        wide = chol_cheb_logdet_precompute(
            small_W, order=None, rho_min=-0.95, rho_max=0.95
        )
        narrow = chol_cheb_logdet_precompute(
            small_W, order=None, rho_min=0.1, rho_max=0.8
        )
        assert wide.order == cheb_order_for_tolerance(-0.95, 0.95, small_W.shape[0])
        assert narrow.order == cheb_order_for_tolerance(0.1, 0.8, small_W.shape[0])
        assert narrow.order < wide.order

    def test_interval_clamped(self, small_W):
        """Interval [-1, 1] should be clamped to [-0.99, 0.99]."""
        pre = chol_cheb_logdet_precompute(
            small_W, order=None, rho_min=-1.0, rho_max=1.0
        )
        assert pre.rho_min == -0.99
        assert pre.rho_max == 0.99
        assert pre.order == cheb_order_for_tolerance(-0.99, 0.99, small_W.shape[0])

    def test_explicit_order_overrides_auto(self, small_W):
        """Explicit order should override auto-selection."""
        pre = chol_cheb_logdet_precompute(small_W, order=8, rho_min=-0.95, rho_max=0.95)
        assert pre.order == 8

    def test_tol_tightens_the_order(self, small_W):
        """A tighter absolute target must buy more nodes."""
        loose = chol_cheb_logdet_precompute(small_W, rho_min=0.1, rho_max=0.8, tol=1e-4)
        tight = chol_cheb_logdet_precompute(
            small_W, rho_min=0.1, rho_max=0.8, tol=1e-12
        )
        assert tight.order > loose.order

    def test_selected_order_meets_its_target(self, small_W, small_eigs):
        """The rule's contract: the auto order actually delivers ``tol``."""
        for lo, hi, tol in [(0.1, 0.8, 1e-8), (0.55, 0.65, 1e-8), (-0.5, 0.9, 1e-6)]:
            pre = chol_cheb_logdet_precompute(small_W, rho_min=lo, rho_max=hi, tol=tol)
            grid = np.linspace(lo, hi, 41)
            exact = np.array(
                [np.sum(np.log(np.abs(1.0 - r * small_eigs))) for r in grid]
            )
            approx = np.array([chol_cheb_logdet_eval(pre, float(r)) for r in grid])
            err = np.abs(approx - exact).max()
            # The fitted model's worst observed under-prediction was 4x.
            assert err < 10 * tol, f"[{lo}, {hi}] tol={tol}: err={err:.2e}"


# ---------------------------------------------------------------------------
# Accuracy
# ---------------------------------------------------------------------------


class TestAccuracy:
    def test_exact_at_nodes(self, small_W, small_eigs):
        """At Chebyshev nodes, the polynomial should match the exact logdet."""
        order = 12
        pre = chol_cheb_logdet_precompute(
            small_W, order=order, rho_min=0.1, rho_max=0.8
        )

        # Recompute the nodes
        k = np.arange(1, order + 1)
        nodes_cos = np.cos((2 * k - 1) * np.pi / (2 * order))
        rho_nodes = 0.5 * (0.8 - 0.1) * nodes_cos + 0.5 * (0.8 + 0.1)

        for rho in rho_nodes:
            exact = np.sum(np.log(np.abs(1.0 - rho * small_eigs)))
            approx = chol_cheb_logdet_eval(pre, float(rho))
            assert abs(approx - exact) < 1e-8, f"rho={rho}: {approx} vs {exact}"

    def test_near_exact_between_nodes(self, small_W, small_eigs):
        """Between nodes, Chebyshev interpolation should be very accurate."""
        pre = chol_cheb_logdet_precompute(small_W, order=15, rho_min=0.1, rho_max=0.8)

        for rho in [0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75]:
            exact = np.sum(np.log(np.abs(1.0 - rho * small_eigs)))
            approx = chol_cheb_logdet_eval(pre, rho)
            # Chebyshev interpolation error should be tiny for smooth functions
            assert abs(approx - exact) < 1e-6, f"rho={rho}: {approx} vs {exact}"

    def test_zero_rho(self, small_W):
        """At ρ=0, log|I| = 0.  Chebyshev extrapolation to endpoint should be close."""
        pre = chol_cheb_logdet_precompute(small_W, order=15, rho_min=0.01, rho_max=0.8)
        val = chol_cheb_logdet_eval(pre, 0.0)
        assert abs(val) < 1e-3

    def test_monotone(self, small_W):
        """log|det(I - ρW)| should be monotonically decreasing for ρ > 0
        (as ρ→1, det→0 so logdet→-∞)."""
        pre = chol_cheb_logdet_precompute(small_W, order=15, rho_min=0.1, rho_max=0.8)
        rhos = np.linspace(0.15, 0.75, 20)
        vals = [chol_cheb_logdet_eval(pre, r) for r in rhos]
        diffs = np.diff(vals)
        assert np.all(diffs < 0), f"Not monotone decreasing: diffs={diffs}"

    def test_full_theoretical_range(self, small_W, small_eigs):
        """Adaptive order should give good accuracy across [-0.95, 0.95]."""
        pre = chol_cheb_logdet_precompute(
            small_W, order=None, rho_min=-0.95, rho_max=0.95
        )
        for rho in [-0.9, -0.5, -0.1, 0.0, 0.1, 0.3, 0.5, 0.7, 0.9]:
            exact = np.sum(np.log(np.abs(1.0 - rho * small_eigs)))
            approx = chol_cheb_logdet_eval(pre, rho)
            assert abs(approx - exact) < 1e-4, f"rho={rho}: {approx} vs {exact}"

    def test_negative_rho(self, small_W, small_eigs):
        """Negative ρ should work — I - ρW is still SPD for |ρ| < 1."""
        pre = chol_cheb_logdet_precompute(small_W, order=20, rho_min=-0.8, rho_max=-0.1)
        for rho in [-0.7, -0.5, -0.3, -0.15]:
            exact = np.sum(np.log(np.abs(1.0 - rho * small_eigs)))
            approx = chol_cheb_logdet_eval(pre, rho)
            assert abs(approx - exact) < 1e-6, f"rho={rho}: {approx} vs {exact}"


# ---------------------------------------------------------------------------
# Vectorized evaluation
# ---------------------------------------------------------------------------


class TestVectorized:
    def test_vec_matches_scalar(self, small_W):
        """Vectorized eval should match scalar eval for each element."""
        pre = chol_cheb_logdet_precompute(small_W, order=12, rho_min=0.1, rho_max=0.8)
        rhos = np.linspace(0.15, 0.75, 50)
        vec_vals = chol_cheb_logdet_eval_vec(pre, rhos)
        scalar_vals = np.array([chol_cheb_logdet_eval(pre, r) for r in rhos])
        assert np.allclose(vec_vals, scalar_vals, atol=1e-12)

    def test_vec_shape(self, small_W):
        pre = chol_cheb_logdet_precompute(small_W, order=10)
        rhos = np.linspace(0.1, 0.8, 30)
        vals = chol_cheb_logdet_eval_vec(pre, rhos)
        assert vals.shape == (30,)

    def test_vec_empty(self, small_W):
        pre = chol_cheb_logdet_precompute(small_W, order=10)
        vals = chol_cheb_logdet_eval_vec(pre, np.array([]))
        assert vals.shape == (0,)


# ---------------------------------------------------------------------------
# Factory integration
# ---------------------------------------------------------------------------


class TestFactory:
    def test_scalar_factory(self, small_W, small_eigs):
        """make_logdet_numpy_fn with cheb_cholesky should match exact logdet."""
        fn = make_logdet_numpy_fn(
            small_W, small_eigs, method="cheb_cholesky", rho_min=0.1, rho_max=0.8
        )
        for rho in [0.2, 0.3, 0.5, 0.7]:
            exact = np.sum(np.log(np.abs(1.0 - rho * small_eigs)))
            approx = fn(rho)
            assert abs(approx - exact) < 1e-6, f"rho={rho}: {approx} vs {exact}"

    def test_vec_factory(self, small_W, small_eigs):
        """make_logdet_numpy_vec_fn with cheb_cholesky should match exact logdet."""
        fn = make_logdet_numpy_vec_fn(
            small_W, small_eigs, method="cheb_cholesky", rho_min=0.1, rho_max=0.8
        )
        rhos = np.linspace(0.15, 0.75, 40)
        vals = fn(rhos)
        exact = np.array([np.sum(np.log(np.abs(1.0 - r * small_eigs))) for r in rhos])
        assert np.allclose(vals, exact, atol=1e-6)

    def test_factory_T(self, small_W, small_eigs):
        """T multiplier should scale the output."""
        fn = make_logdet_numpy_fn(
            small_W, small_eigs, method="cheb_cholesky", rho_min=0.1, rho_max=0.8, T=3
        )
        rho = 0.5
        exact = 3.0 * np.sum(np.log(np.abs(1.0 - rho * small_eigs)))
        assert abs(fn(rho) - exact) < 1e-6

    def test_auto_select_midrange(self):
        """Auto-select should pick chol_aaa for n in (500, 60000] when W is symmetric."""
        from neighbayes._logdet import resolve_logdet_method

        # Without W, the default falls to chol_aaa (symmetric assumption).
        assert resolve_logdet_method(None, n=501) == "chol_aaa"
        assert resolve_logdet_method(None, n=1000) == "chol_aaa"
        assert resolve_logdet_method(None, n=10000) == "chol_aaa"
        assert resolve_logdet_method(None, n=20000) == "chol_aaa"
        assert resolve_logdet_method(None, n=60000) == "chol_aaa"
        assert resolve_logdet_method(None, n=200000) == "cheb_stochastic"
        # cheb_cholesky remains available as an explicit opt-in.
        assert resolve_logdet_method("cheb_cholesky", n=10000) == "cheb_cholesky"


# ---------------------------------------------------------------------------
# Reusable factorization context
# ---------------------------------------------------------------------------


class TestCholChebContext:
    """A context reuses D-symmetrization and CHOLMOD's symbolic analysis.

    This is what makes a warmup-adaptive refit affordable: fitting a second
    interpolant on a narrower interval must cost only its numeric
    factorizations, not another full setup.
    """

    def test_matches_one_shot_precompute_exactly(self, small_W):
        """Coefficients from a context must equal the one-shot wrapper's."""
        from neighbayes._logdet._chol_cheb import CholChebContext

        ctx = CholChebContext(small_W)
        for lo, hi, order in [(0.1, 0.8, 15), (-0.5, 0.9, 20), (0.55, 0.65, 6)]:
            via_ctx = ctx.coeffs_on(lo, hi, order=order)
            fresh = chol_cheb_logdet_precompute(
                small_W, order=order, rho_min=lo, rho_max=hi
            )
            assert np.array_equal(via_ctx.coeffs, fresh.coeffs)
            assert (via_ctx.rho_min, via_ctx.rho_max) == (fresh.rho_min, fresh.rho_max)

    def test_refits_are_independent(self, small_W, small_eigs):
        """Refitting must not corrupt the interval fitted before it."""
        from neighbayes._logdet._chol_cheb import CholChebContext

        ctx = CholChebContext(small_W)
        wide = ctx.coeffs_on(0.1, 0.8, order=15)
        ctx.coeffs_on(0.55, 0.65, order=6)  # refit on a narrower window
        wide_again = ctx.coeffs_on(0.1, 0.8, order=15)
        assert np.array_equal(wide.coeffs, wide_again.coeffs)

        for rho in (0.2, 0.5, 0.75):
            exact = np.sum(np.log(np.abs(1.0 - rho * small_eigs)))
            assert abs(chol_cheb_logdet_eval(wide, rho) - exact) < 1e-6

    def test_narrow_refit_is_far_more_accurate(self, small_W, small_eigs):
        """The payoff: over the refit window, the narrow fit is orders better."""
        from neighbayes._logdet._chol_cheb import CholChebContext

        lo, hi = 0.55, 0.65
        ctx = CholChebContext(small_W)
        wide = ctx.coeffs_on(0.1, 0.8)
        narrow = ctx.coeffs_on(lo, hi, tol=1e-12)

        grid = np.linspace(lo, hi, 41)
        exact = np.array([np.sum(np.log(np.abs(1.0 - r * small_eigs))) for r in grid])
        err_wide = np.abs(
            np.array([chol_cheb_logdet_eval(wide, float(r)) for r in grid]) - exact
        ).max()
        err_narrow = np.abs(
            np.array([chol_cheb_logdet_eval(narrow, float(r)) for r in grid]) - exact
        ).max()
        assert err_narrow < err_wide / 100.0

    def test_rejects_directed_W(self):
        """A directed graph has no symmetrizing diagonal — fail loudly."""
        from neighbayes._logdet._chol_cheb import CholChebContext

        W = sp.csr_matrix(np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]]))
        with pytest.raises(ValueError, match="D-symmetrizable"):
            CholChebContext(W)

    def test_rejects_inverted_interval(self, small_W):
        from neighbayes._logdet._chol_cheb import CholChebContext

        with pytest.raises(ValueError, match="Invalid rho interval"):
            CholChebContext(small_W).coeffs_on(0.8, 0.2)


class TestLUCheb:
    """The LU-Chebyshev cell: same interpolant, different factorizer.

    ``lu_cheb`` completes the factorizer x interpolant grid.  Its value is that
    it isolates the two choices -- comparing it against ``cheb_cholesky`` varies
    only the factorizer, and against ``aaa`` only the interpolant -- and that it
    is the only way to put a Chebyshev interpolant on a directed ``W``.
    """

    @staticmethod
    def _rook(side):
        import numpy as np
        import scipy.sparse as sp

        n = side * side
        rows, cols = [], []
        for i in range(side):
            for j in range(side):
                k = i * side + j
                for di, dj in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    a, b = i + di, j + dj
                    if 0 <= a < side and 0 <= b < side:
                        rows.append(k)
                        cols.append(a * side + b)
        A = sp.csr_matrix(
            (np.ones(len(rows)), (rows, cols)), shape=(n, n), dtype=np.float64
        )
        deg = np.asarray(A.sum(axis=1)).ravel()
        return sp.diags(1.0 / deg) @ A

    def test_matches_cholesky_on_symmetrizable_W(self):
        """Same nodes, same order, different factorizer -> same interpolant.

        This is the factorizer/interpolant independence claim in its sharpest
        form: the two agree to floating-point noise, not merely to tolerance.
        """
        import numpy as np

        from neighbayes._logdet._chol_cheb import (
            chol_cheb_logdet_eval,
            chol_cheb_logdet_precompute,
            lu_cheb_logdet_precompute,
        )

        W = self._rook(20)
        lo, hi = -0.9, 0.9
        pc = chol_cheb_logdet_precompute(W, rho_min=lo, rho_max=hi)
        pl = lu_cheb_logdet_precompute(W, rho_min=lo, rho_max=hi)

        assert pl.order == pc.order
        assert pl.n == pc.n
        np.testing.assert_allclose(pl.coeffs, pc.coeffs, rtol=1e-9, atol=1e-9)

        for r in np.linspace(lo, hi, 11):
            assert chol_cheb_logdet_eval(pl, float(r)) == pytest.approx(
                chol_cheb_logdet_eval(pc, float(r)), rel=1e-9, abs=1e-9
            )

    def test_works_on_directed_W_where_cholesky_cannot(self):
        """Directed ``W`` has no symmetrizing diagonal; LU does not care."""
        import numpy as np
        import scipy.sparse as sp

        from neighbayes._logdet._chol_cheb import (
            CholChebContext,
            chol_cheb_logdet_eval,
            lu_cheb_logdet_precompute,
        )

        rng = np.random.default_rng(0)
        n, k = 60, 4
        rows, cols = [], []
        for i in range(n):
            for j in rng.choice([x for x in range(n) if x != i], size=k, replace=False):
                rows.append(i)
                cols.append(int(j))
        A = sp.csr_matrix(
            (np.ones(len(rows)), (rows, cols)), shape=(n, n), dtype=np.float64
        )
        W = sp.diags(1.0 / np.asarray(A.sum(axis=1)).ravel()) @ A

        with pytest.raises(ValueError):
            CholChebContext(W)

        pre = lu_cheb_logdet_precompute(W, rho_min=-0.9, rho_max=0.9)
        dense = W.toarray()
        for r in np.linspace(-0.9, 0.9, 9):
            exact = float(np.linalg.slogdet(np.eye(n) - r * dense)[1])
            assert chol_cheb_logdet_eval(pre, float(r)) == pytest.approx(
                exact, abs=1e-6
            )

    def test_is_refittable(self):
        from neighbayes._logdet._refit import REFITTABLE_METHODS

        assert "lu_cheb" in REFITTABLE_METHODS

    def test_jax_param_fn_shares_the_chebyshev_parameterization(self):
        from neighbayes._logdet._jax import make_logdet_jax_param_fn

        assert make_logdet_jax_param_fn("lu_cheb") is not None
