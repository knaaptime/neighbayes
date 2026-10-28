"""Tests for the auxiliary-mixture Poisson augmentation.

The mixture tables in :mod:`neighbayes.samplers.poisson_reduced._mixture` were
fitted numerically rather than transcribed, so their accuracy is a measurable
property.  These tests re-derive the exact density independently and pin the
error bounds the module's docstring quotes.
"""

import numpy as np
import pytest
from scipy.special import digamma, gammaln, polygamma

from neighbayes.samplers.poisson_reduced._augment import (
    build_augmented_index,
    draw_augmentation,
)
from neighbayes.samplers.poisson_reduced._mixture import (
    MAX_TABULATED_SHAPE,
    mixture_for_shape,
    mixture_for_unit_shape,
)


def _exact_logpdf(x, nu):
    """Density of -log(G), G ~ Gamma(nu, 1), derived from scratch."""
    return -nu * x - np.exp(-x) - gammaln(nu)


def _kl(nu, ngrid=4001):
    """KL(exact || mixture) on a grid spanning the exact density."""
    mean, sd = -digamma(nu), np.sqrt(polygamma(1, nu))
    x = np.linspace(mean - 9 * sd, mean + 11 * sd, ngrid)
    dx = x[1] - x[0]
    p = np.exp(_exact_logpdf(x, nu))
    p /= p.sum() * dx
    w, m, v = mixture_for_shape(nu)
    q = (
        (w / np.sqrt(2 * np.pi * v))[None, :]
        * np.exp(-0.5 * (x[:, None] - m[None, :]) ** 2 / v[None, :])
    ).sum(axis=1)
    return float(np.sum(p * (np.log(p + 1e-300) - np.log(q + 1e-300))) * dx)


class TestMixtureTables:
    @pytest.mark.parametrize("nu", range(1, 31))
    def test_integer_shapes_accurate(self, nu):
        assert _kl(nu) < 5e-6

    @pytest.mark.parametrize(
        "nu", [31, 37, 44, 60, 95, 150, 240, 380, 600, 950, 1500, 1999]
    )
    def test_interpolated_shapes_accurate(self, nu):
        assert _kl(nu) < 5e-6

    @pytest.mark.parametrize("nu", [2001, 3000, 10000, 100000])
    def test_large_shapes_use_single_normal(self, nu):
        w, m, v = mixture_for_shape(nu)
        assert w.shape == (1,)
        assert m[0] == pytest.approx(-digamma(nu))
        assert v[0] == pytest.approx(polygamma(1, nu))
        assert _kl(nu) < 5e-5

    @pytest.mark.parametrize("nu", [1, 7, 15, 30, 55, 500, 1999])
    def test_moments_match_exact(self, nu):
        """Mixture mean/variance must match -psi(nu) and psi'(nu)."""
        w, m, v = mixture_for_shape(nu)
        mean = float((w * m).sum())
        var = float((w * (v + m**2)).sum() - mean**2)
        assert mean == pytest.approx(-digamma(nu), abs=1e-4)
        assert var == pytest.approx(polygamma(1, nu), rel=1e-4)

    @pytest.mark.parametrize("nu", list(range(1, 31)) + [55, 300, 1500])
    def test_no_degenerate_components(self, nu):
        """Zero weights or zero variances would produce log(0) downstream."""
        w, m, v = mixture_for_shape(nu)
        assert np.all(w > 0)
        assert np.all(v > 0)
        assert float(w.sum()) == pytest.approx(1.0)
        assert np.all(np.isfinite(m))

    def test_unit_shape_is_gumbel(self):
        """-log Exp(1) is exactly a standard Gumbel: mean gamma, var pi^2/6."""
        w, m, v = mixture_for_unit_shape()
        mean = float((w * m).sum())
        var = float((w * (v + m**2)).sum() - mean**2)
        assert mean == pytest.approx(np.euler_gamma, abs=1e-4)
        assert var == pytest.approx(np.pi**2 / 6, rel=1e-4)

    def test_rejects_nonpositive_shape(self):
        with pytest.raises(ValueError):
            mixture_for_shape(0.0)
        with pytest.raises(ValueError):
            mixture_for_shape(-2.0)

    def test_single_normal_would_be_inadequate_at_the_table_edge(self):
        """Guards the cutoff: a lone normal at nu=30 is orders worse."""
        nu = 30
        mean, sd = -digamma(nu), np.sqrt(polygamma(1, nu))
        x = np.linspace(mean - 9 * sd, mean + 11 * sd, 4001)
        dx = x[1] - x[0]
        p = np.exp(_exact_logpdf(x, nu))
        p /= p.sum() * dx
        q = np.exp(-0.5 * ((x - mean) / sd) ** 2) / (sd * np.sqrt(2 * np.pi))
        kl_normal = float(np.sum(p * (np.log(p + 1e-300) - np.log(q + 1e-300))) * dx)
        assert kl_normal > 1e-3
        assert _kl(nu) < kl_normal / 100
        assert MAX_TABULATED_SHAPE > nu


class TestAugmentation:
    def test_shapes_and_row_index(self):
        rng = np.random.default_rng(0)
        y = np.array([0.0, 3.0, 0.0, 7.0, 1.0])
        design = build_augmented_index(y)
        assert design.N == 5
        assert design.n_aug == 5 + 3
        np.testing.assert_array_equal(design.pos, [1, 3, 4])
        np.testing.assert_array_equal(design.rows, [0, 1, 2, 3, 4, 1, 3, 4])
        s, om = draw_augmentation(y, np.zeros(5), design, rng=rng)
        assert s.shape == (8,) and om.shape == (8,)
        assert np.all(np.isfinite(s)) and np.all(om > 0)

    def test_all_zero_counts_gives_one_row_each(self):
        rng = np.random.default_rng(1)
        y = np.zeros(6)
        design = build_augmented_index(y)
        assert design.n_aug == 6
        s, om = draw_augmentation(y, np.zeros(6), design, rng=rng)
        assert s.shape == (6,) and np.all(np.isfinite(s))

    def test_working_precision_converges_to_poisson_information(self):
        """Working precision must approach mu from above, never diverge from it.

        This is the property Polya-Gamma fails in the Poisson limit.  There the
        conditional precision E[omega] = (alpha/2*psi)*tanh(psi/2) outruns the
        Fisher information without bound (measured ratio 0.3 -> 87.5 as alpha
        goes 10 -> 10^4), which is what destroys mixing.  Here the ratio falls
        monotonically toward 1, so the augmentation gets *more* efficient as
        counts grow -- the regime trade flows actually occupy.
        """
        rng = np.random.default_rng(2)
        ratios = []
        for mu in (2.0, 20.0, 200.0, 5000.0):
            y = rng.poisson(mu, size=20000).astype(float)
            eta = np.full(y.size, np.log(mu))
            design = build_augmented_index(y)
            _, om = draw_augmentation(y, eta, design, rng=rng)
            ratios.append(float(om.sum() / y.size / mu))

        # Always above 1 (conditioning on the augmentation adds information)
        # but bounded, and shrinking toward 1 as mu grows.
        assert all(r > 0.95 for r in ratios)
        assert ratios[0] < 6.0
        assert ratios == sorted(ratios, reverse=True)
        assert ratios[-1] < 1.1

    def test_recovers_poisson_glm_coefficients(self):
        """Augmentation + conjugate beta draw must match the Poisson MLE."""
        rng = np.random.default_rng(0)
        n, k = 2500, 3
        X = np.column_stack([np.ones(n), rng.normal(size=(n, k - 1))])
        beta_true = np.array([1.1, 0.45, -0.3])
        y = rng.poisson(np.exp(X @ beta_true)).astype(float)

        design = build_augmented_index(y)
        Xa = X[design.rows]
        V0inv = np.eye(k) / 1e6
        beta = np.zeros(k)
        keep = []
        for it in range(900):
            s, om = draw_augmentation(y, X @ beta, design, rng=rng)
            P = Xa.T @ (Xa * om[:, None]) + V0inv
            L = np.linalg.cholesky(P)
            beta = np.linalg.solve(P, Xa.T @ (om * s)) + np.linalg.solve(
                L.T, rng.normal(size=k)
            )
            if it >= 300:
                keep.append(beta.copy())
        post = np.array(keep).mean(axis=0)
        # Within 4 Monte-Carlo/asymptotic standard errors of the truth.
        se = 1.0 / np.sqrt((np.exp(X @ beta_true)[:, None] * X**2).sum(axis=0))
        assert np.all(np.abs(post - beta_true) < 4 * se)


@pytest.mark.parametrize("count", [1, 10, 60, 150, 400])
def test_augmented_gibbs_matches_exact_posterior(count):
    """One observation, ``η ~ N(0, 3²)``: Gibbs on the augmentation vs quadrature.

    Pins the last-arrival draw.  Given ``y`` arrivals in ``[0, 1]`` the last
    one is ``Beta(y, 1)``; drawing it from ``Gamma(y, λ)`` truncated to
    ``(0, 1)`` instead put the posterior mean 4.6 sd off at ``y = 400`` and
    tripled its spread, while every mixture-table test still passed.
    """
    rng = np.random.default_rng(count)
    y = np.array([float(count)])
    design = build_augmented_index(y)
    eta, draws = np.log(count + 0.5), []
    for it in range(6000):
        s, om = draw_augmentation(y, np.array([eta]), design, rng=rng)
        prec = om.sum() + 1.0 / 9.0
        eta = (om * s).sum() / prec + rng.standard_normal() / np.sqrt(prec)
        if it >= 500:
            draws.append(eta)
    draws = np.asarray(draws)

    grid = np.linspace(-15.0, 10.0, 40001)
    lp = count * grid - np.exp(grid) - 0.5 * grid**2 / 9.0
    w = np.exp(lp - lp.max())
    w /= w.sum()
    mean = float((w * grid).sum())
    sd = float(np.sqrt((w * (grid - mean) ** 2).sum()))
    assert abs(draws.mean() - mean) < 0.15 * sd
    assert abs(draws.std() / sd - 1.0) < 0.1
