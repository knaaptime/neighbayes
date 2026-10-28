"""Contract pins for the shared numpy Clenshaw helper (``_logdet._clenshaw``).

``chol_cheb_logdet_eval_vec`` and the logdet factories both evaluate a
Chebyshev-in-ρ series through these helpers, so they must reproduce
``numpy.polynomial.chebyshev.chebval`` on the mapped domain exactly, and the
scalar and vector forms must agree.
"""

from __future__ import annotations

import numpy as np
from numpy.polynomial import chebyshev as _cheb

from neighbayes._logdet._clenshaw import clenshaw_scalar, clenshaw_vec

RMIN, RMAX = -0.9, 0.95


def _reference(coeffs, r):
    x = (2.0 * r - RMAX - RMIN) / (RMAX - RMIN)
    return _cheb.chebval(x, coeffs)


def test_scalar_matches_chebval():
    rng = np.random.default_rng(0)
    coeffs = rng.normal(size=9)
    for r in np.linspace(RMIN, RMAX, 13):
        got = clenshaw_scalar(coeffs, r, RMIN, RMAX)
        np.testing.assert_allclose(got, _reference(coeffs, r), rtol=0, atol=1e-12)


def test_vec_matches_scalar():
    rng = np.random.default_rng(1)
    coeffs = rng.normal(size=12)
    rr = np.linspace(RMIN, RMAX, 25)
    vec = clenshaw_vec(coeffs, rr, RMIN, RMAX)
    scal = np.array([clenshaw_scalar(coeffs, r, RMIN, RMAX) for r in rr])
    np.testing.assert_allclose(vec, scal, rtol=0, atol=0)


def test_T_scales_linearly():
    rng = np.random.default_rng(2)
    coeffs = rng.normal(size=6)
    base = clenshaw_scalar(coeffs, 0.3, RMIN, RMAX, 1)
    np.testing.assert_allclose(clenshaw_scalar(coeffs, 0.3, RMIN, RMAX, 4), 4 * base)
    vbase = clenshaw_vec(coeffs, np.array([0.1, 0.3]), RMIN, RMAX, 1)
    np.testing.assert_allclose(
        clenshaw_vec(coeffs, np.array([0.1, 0.3]), RMIN, RMAX, 4), 4 * vbase
    )


def test_degenerate_lengths():
    # empty and single-coefficient series
    assert clenshaw_scalar([], 0.2, RMIN, RMAX) == 0.0
    assert clenshaw_scalar([2.5], 0.2, RMIN, RMAX) == 2.5
    np.testing.assert_array_equal(
        clenshaw_vec([], np.array([0.1, 0.2]), RMIN, RMAX), np.zeros(2)
    )
    np.testing.assert_array_equal(
        clenshaw_vec([2.5], np.array([0.1, 0.2]), RMIN, RMAX), np.full(2, 2.5)
    )
