"""Tests for the warmup-adaptive Jacobian refit.

The refit rebuilds the log-determinant interpolant on the ρ range a warmed-up
sampler actually explores.  Two things have to hold for that to be sound:

1. the refit must not move the posterior — it is an adaptation, not a change of
   model, so any shift must be within Monte Carlo error; and
2. it narrows the interpolant's domain, and a Chebyshev series diverges outside
   its interval, so the sampler's support must follow it and the truncation must
   be reported rather than silent.

The tests below cover both, plus the mechanics that make the refit worth doing:
a window built from draws, fixed-shape JAX parameters so the swap costs no
recompilation, and the guards that decline to refit when it would not pay.
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp

from neighbayes._logdet._refit import (
    REFITTABLE_METHODS,
    LogdetRefitter,
    RefitWindow,
    boundary_warning,
    refit_window,
)


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


def _knn(n: int = 400, k: int = 4, seed: int = 0) -> sp.csr_matrix:
    """Directed graph — routed to AAA, where Cholesky is unavailable."""
    rng = np.random.default_rng(seed)
    rows = np.repeat(np.arange(n), k)
    cols = np.concatenate(
        [
            rng.choice(np.delete(np.arange(n), i), size=k, replace=False)
            for i in range(n)
        ]
    )
    A = sp.coo_matrix((np.ones(n * k), (rows, cols)), shape=(n, n)).tocsr()
    A.data[:] = 1.0
    deg = np.asarray(A.sum(axis=1)).ravel()
    deg[deg == 0] = 1.0
    return sp.csr_matrix(sp.diags(1.0 / deg) @ A)


@pytest.fixture(scope="module")
def W():
    return _rook(30)


# ---------------------------------------------------------------------------
# Window selection
# ---------------------------------------------------------------------------


class TestRefitWindow:
    def test_spans_draws_plus_padding(self):
        rng = np.random.default_rng(0)
        draws = 0.6 + 0.01 * rng.standard_normal(500)
        lo, hi = refit_window(draws, 0.0, 0.95, pad_sd=10.0)
        assert lo < draws.min() and hi > draws.max()
        # 10 sd of padding on each side, sd ≈ 0.01
        assert 0.08 < draws.min() - lo < 0.12
        assert 0.08 < hi - draws.max() < 0.12

    def test_never_exceeds_the_prior(self):
        rng = np.random.default_rng(0)
        draws = 0.6 + 0.05 * rng.standard_normal(500)  # padding would run past 0.95
        lo, hi = refit_window(draws, 0.3, 0.95, pad_sd=10.0)
        assert lo >= 0.3 and hi <= 0.95

    def test_clamped_away_from_the_singularities(self):
        rng = np.random.default_rng(0)
        draws = 0.9 + 0.05 * rng.standard_normal(500)
        lo, hi = refit_window(draws, -1.0, 1.0, pad_sd=10.0)
        assert lo >= -0.99 and hi <= 0.99

    @pytest.mark.parametrize(
        "draws",
        [
            np.array([]),
            np.full(500, 0.6),  # stuck chain: no spread to build a window from
            np.array([0.6] * 5),  # too few draws
            np.array([np.nan] * 100),
        ],
        ids=["empty", "degenerate", "too-few", "all-nan"],
    )
    def test_declines_on_unusable_draws(self, draws):
        """Better to keep the prior interval than to refit onto nothing."""
        assert refit_window(draws, 0.0, 0.95) is None


class TestBoundaryWarning:
    def _win(self, lo, hi, prior=(0.0, 0.95)):
        return RefitWindow(
            rho_min=lo,
            rho_max=hi,
            order=8,
            n_warmup_draws=500,
            pad_sd=10.0,
            prior_min=prior[0],
            prior_max=prior[1],
        )

    def test_silent_when_draws_sit_well_inside(self):
        rng = np.random.default_rng(0)
        draws = 0.6 + 0.01 * rng.standard_normal(2000)
        assert boundary_warning(draws, self._win(0.4, 0.8)) is None

    def test_fires_when_draws_reach_a_truncated_edge(self):
        rng = np.random.default_rng(0)
        draws = 0.6 + 0.01 * rng.standard_normal(2000)
        msg = boundary_warning(draws, self._win(0.4, 0.62))
        assert msg is not None and "upper edge" in msg
        assert "logdet_adapt" in msg  # tells the user what to change

    def test_silent_at_an_edge_that_is_the_prior_bound(self):
        """The prior always truncated there; the refit changed nothing."""
        rng = np.random.default_rng(0)
        draws = 0.94 + 0.005 * rng.standard_normal(2000)
        assert boundary_warning(draws, self._win(0.8, 0.95)) is None


# ---------------------------------------------------------------------------
# Refitter
# ---------------------------------------------------------------------------


class TestRefitter:
    def test_only_interval_owning_methods_are_supported(self, W):
        for method in ("cheb_cholesky", "aaa"):
            assert LogdetRefitter(W, method).supported
            assert method in REFITTABLE_METHODS
        for method in ("eigenvalue", "cheb_stochastic", "chebyshev", "cholmod"):
            assert not LogdetRefitter(W, method).supported

    def test_construction_does_no_work(self, W):
        """A run that never reaches the refit point must not pay for one."""
        r = LogdetRefitter(W, "cheb_cholesky")
        assert r._context is None

    def test_declines_when_the_window_is_not_narrower(self, W):
        r = LogdetRefitter(W, "cheb_cholesky")
        assert not r.worth_refitting(0.0, 0.94, 0.0, 0.95)
        assert r.worth_refitting(0.5, 0.7, 0.0, 0.95)

    def test_refit_is_exact_over_its_window(self, W):
        """The point of the refit: near-roundoff accuracy where ρ actually lives."""
        from sksparse.cholmod import cho_factor

        from neighbayes._logdet import make_logdet_numpy_fn
        from neighbayes._logdet._chol_cheb import _d_symmetrize

        lo, hi = 0.55, 0.68
        n = W.shape[0]
        W_sym = _d_symmetrize(sp.csr_matrix(W))
        eye = sp.eye(n, format="csc")

        def exact(r):
            return float(cho_factor(sp.csc_matrix(eye - r * W_sym)).logdet())

        grid = np.linspace(lo, hi, 25)
        truth = np.array([exact(r) for r in grid])

        base_fn = make_logdet_numpy_fn(W, None, "cheb_cholesky", 0.0, 0.95)
        base = np.array([base_fn(float(r)) for r in grid]) - truth

        refit_fn, _, info = LogdetRefitter(W, "cheb_cholesky").refit(lo, hi, 0.0, 0.95)
        got = np.array([refit_fn(float(r)) for r in grid]) - truth

        # The tilt — the variation of the error across the window — is what
        # moves a posterior; a constant offset cancels.
        tilt_base = base.max() - base.min()
        tilt_refit = got.max() - got.min()
        assert tilt_refit < tilt_base / 100.0
        assert info.rho_min == pytest.approx(lo) and info.rho_max == pytest.approx(hi)

    def test_scalar_and_vector_evaluators_agree(self, W):
        fn, vec_fn, _ = LogdetRefitter(W, "cheb_cholesky").refit(0.5, 0.7, 0.0, 0.95)
        grid = np.linspace(0.51, 0.69, 17)
        np.testing.assert_allclose(
            vec_fn(grid), [fn(float(r)) for r in grid], rtol=0, atol=1e-12
        )

    def test_T_scales_the_panel_jacobian(self, W):
        one, _, _ = LogdetRefitter(W, "cheb_cholesky", T=1).refit(0.5, 0.7, 0.0, 0.95)
        three, _, _ = LogdetRefitter(W, "cheb_cholesky", T=3).refit(0.5, 0.7, 0.0, 0.95)
        assert three(0.6) == pytest.approx(3.0 * one(0.6))

    def test_aaa_refit_on_directed_weights(self):
        W_dir = _knn()
        fn, vec_fn, info = LogdetRefitter(W_dir, "aaa").refit(0.5, 0.7, 0.0, 0.95)
        grid = np.linspace(0.51, 0.69, 13)
        np.testing.assert_allclose(
            vec_fn(grid), [fn(float(r)) for r in grid], atol=1e-9
        )
        assert info.order >= 2

    def test_context_is_reused_across_refits(self, W):
        """The symbolic analysis is paid once, however many windows are fitted."""
        r = LogdetRefitter(W, "cheb_cholesky")
        r.refit(0.4, 0.8, 0.0, 0.95)
        ctx = r._context
        r.refit(0.55, 0.65, 0.0, 0.95)
        assert r._context is ctx


# ---------------------------------------------------------------------------
# JAX parameterization
# ---------------------------------------------------------------------------

jax = pytest.importorskip("jax")


class TestJaxParams:
    """Params must be fixed-shape so a refit does not invalidate the compiled step."""

    @pytest.mark.parametrize("method", ["cheb_cholesky", "aaa"])
    def test_shapes_are_identical_across_intervals(self, W, method):
        Wm = W if method == "cheb_cholesky" else _knn()
        r = LogdetRefitter(Wm, method)
        cap = r.capacity(0.0, 0.95)
        shapes = []
        for lo, hi in [(0.0, 0.95), (0.4, 0.8), (0.58, 0.62)]:
            params, _ = r.jax_params(lo, hi, cap)
            shapes.append(jax.tree.map(lambda a: (a.shape, a.dtype), params))
        assert shapes[0] == shapes[1] == shapes[2]

    def test_params_are_float64(self, W):
        """A float32/float64 mismatch at the swap would force the retrace."""
        params, _ = LogdetRefitter(W, "cheb_cholesky").jax_params(
            0.4, 0.8, LogdetRefitter(W, "cheb_cholesky").capacity(0.0, 0.95)
        )
        for leaf in jax.tree.leaves(params):
            assert leaf.dtype == np.float64

    @pytest.mark.parametrize("method", ["cheb_cholesky", "aaa"])
    def test_padded_params_evaluate_exactly(self, W, method):
        """Zero padding must contribute nothing to either representation.

        Padded out from one fit, so the comparison isolates the padding — the
        capacity also caps the order, so refitting at a larger capacity would
        change the interpolant as well.
        """
        import jax.numpy as jnp

        from neighbayes._logdet._jax import make_logdet_jax_param_fn

        Wm = W if method == "cheb_cholesky" else _knn()
        r = LogdetRefitter(Wm, method)
        fn = make_logdet_jax_param_fn(method)
        lo, hi = 0.5, 0.7

        tight, _ = r.jax_params(lo, hi, r.capacity(lo, hi))

        def _extend(leaf, fill):
            return jnp.concatenate([leaf, jnp.full((40,), fill, leaf.dtype)])

        if method == "cheb_cholesky":
            coeffs, rmin, rmax = tight
            roomy = (_extend(coeffs, 0.0), rmin, rmax)
        else:
            z, f, w = tight
            roomy = (
                _extend(z, LogdetRefitter._AAA_PAD_Z),
                _extend(f, 0.0),
                _extend(w, 0.0),
            )

        for rho in (0.52, 0.6, 0.68):
            assert float(fn(rho, tight)) == pytest.approx(
                float(fn(rho, roomy)), abs=1e-10
            )

    def test_matches_the_numpy_evaluator(self, W):
        from neighbayes._logdet._jax import make_logdet_jax_param_fn

        r = LogdetRefitter(W, "cheb_cholesky")
        cap = r.capacity(0.0, 0.95)
        params, _ = r.jax_params(0.5, 0.7, cap)
        fn_np, _, _ = r.refit(0.5, 0.7, 0.0, 0.95)
        fn_jax = make_logdet_jax_param_fn("cheb_cholesky")
        for rho in (0.52, 0.6, 0.68):
            assert float(fn_jax(rho, params)) == pytest.approx(fn_np(rho), abs=1e-10)

    def test_capacity_bounds_every_window(self, W):
        """Any sub-interval of the prior must fit; otherwise the refit would raise."""
        r = LogdetRefitter(W, "cheb_cholesky")
        cap = r.capacity(-0.99, 0.99)
        for lo, hi in [(-0.99, 0.99), (-0.5, 0.9), (0.1, 0.8), (0.59, 0.61)]:
            params, info = r.jax_params(lo, hi, cap)
            assert info.order <= cap

    def test_capacity_caps_the_order_rather_than_overflowing(self, W):
        """The Chebyshev order is clamped to capacity, never allowed to exceed it.

        The clamp is what keeps the padded evaluator's loop — whose length is a
        compile-time constant — from growing past the order the un-refitted run
        would have used.  It is safe because the window is a sub-interval of the
        prior, so at equal order the refit is strictly the more accurate fit.
        """
        r = LogdetRefitter(W, "cheb_cholesky")
        for cap in (4, 8, 20):
            _, info = r.jax_params(-0.99, 0.99, cap)
            assert info.order <= cap

    def test_gradient_flows(self, W):
        """The refit path must stay differentiable for gradient-based samplers."""
        from neighbayes._logdet._jax import make_logdet_jax_param_fn

        r = LogdetRefitter(W, "cheb_cholesky")
        params, _ = r.jax_params(0.5, 0.7, r.capacity(0.0, 0.95))
        fn = make_logdet_jax_param_fn("cheb_cholesky")
        g = jax.grad(lambda rho: fn(rho, params))(0.6)
        assert np.isfinite(float(g))
        # dJ/dρ = -tr(W(I-ρW)^{-1}) < 0 for ρ > 0 on a row-standardized W
        assert float(g) < 0.0
