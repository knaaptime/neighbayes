"""Tests for legacy-style log-determinant helper implementations."""

from __future__ import annotations

import numpy as np
import pytest

import neighbayes._logdet._factories as fac
from neighbayes._logdet import get_cached_logdet_fn, make_logdet_fn
from neighbayes.tests.helpers import eigen_cheb_coeffs


def _toy_w() -> np.ndarray:
    # Small row-stochastic matrix with spectral radius <= 1.
    return np.array(
        [
            [0.0, 1.0, 0.0],
            [0.5, 0.0, 0.5],
            [0.0, 1.0, 0.0],
        ],
        dtype=float,
    )


def test_make_logdet_fn_accepts_new_method_names() -> None:
    W = _toy_w()

    for method in ("eigenvalue", "eigenvalue", "eigenvalue", "eigenvalue"):
        fn = make_logdet_fn(W, method=method, rho_min=1e-5, rho_max=0.5)
        assert callable(fn)


def test_chebyshev_accuracy_against_exact() -> None:
    W = _toy_w()
    I = np.eye(W.shape[0])
    out = eigen_cheb_coeffs(W, order=20, rmin=-0.5, rmax=0.5)
    coeffs = out["coeffs"]
    rmin, rmax = out["rmin"], out["rmax"]

    # Evaluate Chebyshev approximation at several rho values
    import pytensor
    import pytensor.tensor as pt

    from neighbayes._logdet import logdet_chebyshev

    rho_sym = pt.scalar("rho")
    expr = logdet_chebyshev(rho_sym, coeffs, rmin=rmin, rmax=rmax)
    fn = pytensor.function([rho_sym], expr)

    test_rhos = np.linspace(-0.4, 0.4, 9)
    for rho in test_rhos:
        approx = float(fn(rho))
        exact = np.linalg.slogdet(I - rho * W)[1]
        # Chebyshev with order=20 should be very accurate for this small matrix
        assert abs(approx - exact) < 0.05, f"rho={rho}: approx={approx}, exact={exact}"


def test_get_cached_logdet_fn_reuses_callable_for_same_key() -> None:
    W = _toy_w()
    fac._LOGDET_FN_CACHE.clear()

    fn1 = get_cached_logdet_fn(W, method="eigenvalue", rho_min=-0.5, rho_max=0.5)
    fn2 = get_cached_logdet_fn(W, method="eigenvalue", rho_min=-0.5, rho_max=0.5)

    assert fn1 is fn2


def test_get_cached_logdet_fn_separates_different_bounds() -> None:
    W = _toy_w()
    fac._LOGDET_FN_CACHE.clear()

    fn1 = get_cached_logdet_fn(W, method="eigenvalue", rho_min=-0.5, rho_max=0.5)
    fn2 = get_cached_logdet_fn(W, method="eigenvalue", rho_min=-0.3, rho_max=0.3)

    assert fn1 is not fn2


# ---------------------------------------------------------------------------
# logdet_mc_poly_pytensor
# ---------------------------------------------------------------------------


class TestMakeFlowSeparableLogdet:
    """make_flow_separable_logdet returns fn(rho_d, rho_o) = n*f(rho_d) + n*f(rho_o)."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        import pytensor
        import pytensor.tensor as pt
        import scipy.sparse as sp

        self.pytensor = pytensor
        self.pt = pt

        rng = np.random.default_rng(7)
        n = 8
        W_dense = rng.random((n, n))
        W_dense /= W_dense.sum(axis=1, keepdims=True)
        self.W = W_dense
        self.W_sp = sp.csr_matrix(W_dense)
        self.n = n
        eigs = np.linalg.eigvals(W_dense)
        self.rho_d, self.rho_o = 0.3, 0.25

        # Reference: exact eigenvalue-based answer
        rho_d_t = pt.dscalar("rd")
        rho_o_t = pt.dscalar("ro")
        from neighbayes._logdet import logdet_eigenvalue

        ref_expr = n * logdet_eigenvalue(rho_d_t, eigs) + n * logdet_eigenvalue(
            rho_o_t, eigs
        )
        self.ref_fn = pytensor.function([rho_d_t, rho_o_t], ref_expr)
        self.ref_val = self.ref_fn(self.rho_d, self.rho_o)

    def _compile(self, fn):
        rho_d_t = self.pt.dscalar("rd")
        rho_o_t = self.pt.dscalar("ro")
        return self.pytensor.function([rho_d_t, rho_o_t], fn(rho_d_t, rho_o_t))

    def test_eigenvalue_method(self):
        from neighbayes._logdet import make_flow_separable_logdet

        fn = make_flow_separable_logdet(self.W_sp, self.n, method="eigenvalue")
        compiled = self._compile(fn)
        val = float(compiled(self.rho_d, self.rho_o))
        assert abs(val - self.ref_val) < 1e-8

    def test_invalid_method_raises(self):
        from neighbayes._logdet import make_flow_separable_logdet

        # The separable factory now delegates to the general single-parameter
        # factory, which rejects methods it does not support (e.g. "traces",
        # which is only for the unrestricted 3-parameter flow) and unknown names.
        with pytest.raises(ValueError, match="[Uu]nsupported|[Uu]nknown"):
            make_flow_separable_logdet(self.W_sp, self.n, method="traces")
        with pytest.raises(ValueError, match="[Uu]nsupported|[Uu]nknown"):
            make_flow_separable_logdet(self.W_sp, self.n, method="bogus")
