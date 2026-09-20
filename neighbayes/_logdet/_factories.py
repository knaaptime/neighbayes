"""Factory functions that build logdet evaluators.

Three backends:

* **PyTensor** (``make_logdet_fn``) — symbolic expression for ``pm.Potential``.
* **NumPy scalar** (``make_logdet_numpy_fn``) — ``(rho: float) -> float``.
* **NumPy vectorized** (``make_logdet_numpy_vec_fn``) — ``(rho_arr) -> np.ndarray``.
* **JAX** (``make_logdet_jax_fn`` in ``_jax.py``) — JIT-compatible.

All accept ``method="eigenvalue"`` or ``method="chebyshev"`` (or ``None``
for auto-select).  ``T`` multiplies the result for panel models.
"""

from __future__ import annotations

import hashlib
from collections import OrderedDict

import numpy as np
import scipy.sparse as sp

from ._cheb_stochastic import (
    cheb_stochastic_logdet_eval,
    cheb_stochastic_logdet_precompute,
)
from ._chebyshev import chebyshev, chebyshev_coeffs_dct1, chebyshev_gauss_nodes
from ._clenshaw import clenshaw_scalar as _clenshaw_scalar
from ._clenshaw import clenshaw_vec as _clenshaw_vec
from ._config import (
    _LOGDET_FN_CACHE,
    _LOGDET_FN_CACHE_MAXSIZE,
    resolve_logdet_method,
)
from ._pytensor import logdet_chebyshev, logdet_eigenvalue
from ._slq import (
    slq_logdet_precompute,
    slq_to_chebyshev_coeffs,
)

# ---------------------------------------------------------------------------
# Stochastic-Chebyshev coefficient helper
# ---------------------------------------------------------------------------


def _cheb_precompute_for(method: str):
    """Return the Chebyshev precompute matching ``method``'s factorizer.

    ``cheb_cholesky`` and ``lu_cheb`` differ only in how the exact node values
    are obtained — sparse Cholesky versus sparse LU — and produce the same
    :class:`~._chol_cheb.CholChebPrecompute`, so every downstream evaluation,
    gradient and refit path is shared.
    """
    from ._chol_cheb import chol_cheb_logdet_precompute, lu_cheb_logdet_precompute

    return (
        lu_cheb_logdet_precompute
        if method == "lu_cheb"
        else chol_cheb_logdet_precompute
    )


#: Chebyshev-in-ρ coefficients of the stochastic surrogate, keyed by W, interval
#: and probe count.  A fit asks for them once per evaluator it builds (scalar,
#: vectorized, JAX, gradient); the precompute is seeded, so the key determines
#: them and one precompute serves every evaluator.
_CHEB_STOCHASTIC_COEFFS: OrderedDict[tuple, tuple] = OrderedDict()
_CHEB_STOCHASTIC_COEFFS_MAXSIZE = 16


def _cheb_stochastic_coeffs(W_sparse, rho_min, rho_max, n_probes: int = 50):
    """Stochastic-Chebyshev logdet → Chebyshev-in-ρ coefficients.

    Precomputes stochastic moments, evaluates the logdet at Chebyshev nodes in
    ``[rho_min, rho_max]``, then fits a Chebyshev-in-ρ polynomial via DCT-I so
    it can be evaluated with an O(m) Clenshaw recurrence.

    The ρ interval is forwarded so the matrix-argument order is sized against
    the probe-noise floor rather than pinned at the module default, which is
    what lets the exact low-order moments (``n_exact``) pay off near ρ = 1.

    The refit degree comes from the same interval rule the deterministic
    Chebyshev interpolant uses (:func:`~._chebyshev.cheb_order_for_tolerance`),
    not a fixed 20.  A fixed 20 was harmless while the estimator itself carried
    an error of tens; once the exact moments cut that error several-fold, the
    refit's own interpolation error became the binding one, inflating the max
    error by up to 2x at ``n = 59,536`` over ``[-0.99, 0.99]``.  Sizing it from
    the interval keeps the refit off the critical path at every width — and on
    the narrow post-warmup intervals of :mod:`._refit` it asks for *fewer* nodes
    than 20, so per-ρ evaluation there gets cheaper, not dearer.  The nodes cost
    one O(p²) scalar evaluation each against a precompute measured in hundreds
    of milliseconds, so the extra width is free at setup.
    """
    key = (
        _logdet_w_signature(W_sparse),
        float(rho_min),
        float(rho_max),
        int(n_probes),
    )
    hit = _CHEB_STOCHASTIC_COEFFS.get(key)
    if hit is not None:
        _CHEB_STOCHASTIC_COEFFS.move_to_end(key)
        return hit
    pre = cheb_stochastic_logdet_precompute(
        W_sparse, order=None, rho_min=rho_min, rho_max=rho_max, n_probes=n_probes
    )
    out = _cheb_stochastic_coeffs_from(pre, rho_min, rho_max)
    out[0].flags.writeable = False
    _CHEB_STOCHASTIC_COEFFS[key] = out
    if len(_CHEB_STOCHASTIC_COEFFS) > _CHEB_STOCHASTIC_COEFFS_MAXSIZE:
        _CHEB_STOCHASTIC_COEFFS.popitem(last=False)
    return out


def _cheb_stochastic_coeffs_from(pre, rho_min, rho_max):
    """Chebyshev-in-ρ coefficients of an existing stochastic Chebyshev precompute.

    The degree depends on the interval and ``n`` only, never on the probe count,
    so a probe pool that grows keeps the same coefficient shape.
    """
    from ._chebyshev import cheb_order_for_tolerance

    # An explicit cap keeps NEIGHBAYES_LOGDET_NODE_CAP out of this path: that
    # knob matches *factorization* budgets across interpolants, and the refit
    # performs none — its nodes are scalar evaluations of an existing surrogate.
    degree = cheb_order_for_tolerance(rho_min, rho_max, pre.n, floor=8, cap=200)
    rho_nodes, _ = chebyshev_gauss_nodes(degree, rho_min, rho_max)
    logdet_vals = np.array(
        [cheb_stochastic_logdet_eval(pre, float(r)) for r in rho_nodes]
    )
    return chebyshev_coeffs_dct1(logdet_vals), float(rho_min), float(rho_max)


# ---------------------------------------------------------------------------
# NumPy scalar factory
# ---------------------------------------------------------------------------


def make_logdet_numpy_fn(
    W_sparse,
    eigs: np.ndarray | None,
    method: str | None,
    rho_min: float = -1.0,
    rho_max: float = 1.0,
    T: int = 1,
):
    """Return a pure-numpy ``(rho: float) -> float`` logdet evaluator."""
    T = int(T)
    n = eigs.shape[0] if eigs is not None else int(W_sparse.shape[0])
    method = resolve_logdet_method(method, n=n, W=W_sparse)

    if method == "eigenvalue":
        if eigs is None:
            eigs = np.linalg.eigvals(np.asarray(W_sparse.toarray(), dtype=np.float64))
        _eigs = np.asarray(eigs, dtype=np.complex128)
        if T == 1:
            return lambda r: float(np.sum(np.log(np.abs(1.0 - r * _eigs))))
        return lambda r: T * float(np.sum(np.log(np.abs(1.0 - r * _eigs))))

    if method == "chebyshev":
        out = chebyshev(W_sparse, order=20, rmin=rho_min, rmax=rho_max, eigs=eigs)
        coeffs = out["coeffs"]
        rmin_cb, rmax_cb = out["rmin"], out["rmax"]
        return lambda r: _clenshaw_scalar(coeffs, r, rmin_cb, rmax_cb, T)

    if method == "cheb_stochastic":
        coeffs, rmin_cb, rmax_cb = _cheb_stochastic_coeffs(W_sparse, rho_min, rho_max)
        return lambda r: _clenshaw_scalar(coeffs, r, rmin_cb, rmax_cb, T)

    if method in ("cheb_cholesky", "lu_cheb"):
        # Cholesky-Chebyshev: exact logdet via sparse Cholesky at Chebyshev nodes.
        # No stochastic noise, no O(n³) eigendecomposition.  SPD for |ρ| < 1.
        pre = _cheb_precompute_for(method)(
            W_sparse, order=None, rho_min=rho_min, rho_max=rho_max
        )
        coeffs = pre.coeffs
        rmin_cb, rmax_cb = pre.rho_min, pre.rho_max
        return lambda r: _clenshaw_scalar(coeffs, r, rmin_cb, rmax_cb, T)

    if method == "aaa":
        from ._aaa import aaa_logdet_eval, aaa_logdet_precompute

        # AAA rational approximation: exact logdet via sparse LU at
        # adaptively-selected support points, then barycentric evaluation.
        # For non-symmetric W where Cholesky is unavailable.
        pre = aaa_logdet_precompute(W_sparse, rho_min=rho_min, rho_max=rho_max)

        def _aaa_numpy(r):
            val = aaa_logdet_eval(pre, float(r))
            return val if T == 1 else T * val

        return _aaa_numpy

    if method == "chol_aaa":
        from ._aaa import aaa_logdet_eval, chol_aaa_logdet_precompute

        # Cholesky-based AAA: same rational approximant as "aaa" but exact
        # values come from sparse Cholesky of the D-symmetrized system,
        # which is ~2× cheaper than KLU for symmetrizable W.
        pre = chol_aaa_logdet_precompute(W_sparse, rho_min=rho_min, rho_max=rho_max)

        def _chol_aaa_numpy(r):
            val = aaa_logdet_eval(pre, float(r))
            return val if T == 1 else T * val

        return _chol_aaa_numpy

    if method == "grid_spline":
        from ._grid_spline import grid_spline_logdet_eval, grid_spline_logdet_precompute

        # Incumbent practice: exact LU on an equispaced grid + cubic spline.
        # Node count fixed by convention, not derived from the interval.
        pre = grid_spline_logdet_precompute(W_sparse, rho_min=rho_min, rho_max=rho_max)

        def _grid_spline_numpy(r):
            val = grid_spline_logdet_eval(pre, float(r))
            return val if T == 1 else T * val

        return _grid_spline_numpy

    if method == "slq":
        # SLQ precompute → Chebyshev coefficients → Clenshaw evaluation
        pre = slq_logdet_precompute(W_sparse)
        cheb = slq_to_chebyshev_coeffs(
            pre, W=W_sparse, order=20, rho_min=rho_min, rho_max=rho_max
        )
        coeffs = cheb["coeffs"]
        rmin_cb, rmax_cb = cheb["rmin"], cheb["rmax"]
        return lambda r: _clenshaw_scalar(coeffs, r, rmin_cb, rmax_cb, T)

    raise ValueError(f"Unsupported logdet method: {method!r}")


# ---------------------------------------------------------------------------
# NumPy gradient factory (logdet gradient = resolvent trace)
# ---------------------------------------------------------------------------


def make_logdet_grad_numpy_fn(
    W_sparse,
    eigs: np.ndarray | None,
    method: str | None,
    rho_min: float = -1.0,
    rho_max: float = 1.0,
    T: int = 1,
):
    """Return a pure-numpy ``(rho: float) -> float`` **logdet-gradient** evaluator.

    Returns ``g(ρ) = d/dρ log|I − ρW| = −tr(W(I − ρW)⁻¹)`` — the analytic
    derivative of the matching :func:`make_logdet_numpy_fn` value form, so the
    two compose into a consistent ``(logp, grad)`` pair for gradient-based
    samplers (nutpie-custom, MALA, pseudo-marginal).  The negated value,
    ``−g(ρ)``, is the resolvent trace used by spatial impacts.

    Every branch mirrors :func:`make_logdet_numpy_fn`'s representation choice
    (in particular ``slq`` shares the Chebyshev-converted coefficients its value
    form uses) so ``grad == d(value)/dρ`` holds exactly for each method.
    """
    from ._resolvent import (
        logdet_grad_aaa,
        logdet_grad_chebyshev,
        logdet_grad_eigenvalue,
    )

    T = int(T)
    n = eigs.shape[0] if eigs is not None else int(W_sparse.shape[0])
    method = resolve_logdet_method(method, n=n, W=W_sparse)

    def _scale(g):
        return float(g) if T == 1 else T * float(g)

    if method == "eigenvalue":
        if eigs is None:
            eigs = np.linalg.eigvals(np.asarray(W_sparse.toarray(), dtype=np.float64))
        _eigs = np.asarray(eigs, dtype=np.complex128)
        return lambda r: _scale(logdet_grad_eigenvalue(float(r), _eigs))

    if method == "chebyshev":
        out = chebyshev(W_sparse, order=20, rmin=rho_min, rmax=rho_max, eigs=eigs)
        coeffs, rmin_cb, rmax_cb = out["coeffs"], out["rmin"], out["rmax"]
        return lambda r: _scale(
            logdet_grad_chebyshev(float(r), coeffs, rmin_cb, rmax_cb)
        )

    if method == "cheb_stochastic":
        coeffs, rmin_cb, rmax_cb = _cheb_stochastic_coeffs(W_sparse, rho_min, rho_max)
        return lambda r: _scale(
            logdet_grad_chebyshev(float(r), coeffs, rmin_cb, rmax_cb)
        )

    if method in ("cheb_cholesky", "lu_cheb"):
        pre = _cheb_precompute_for(method)(
            W_sparse, order=None, rho_min=rho_min, rho_max=rho_max
        )
        coeffs, rmin_cb, rmax_cb = pre.coeffs, pre.rho_min, pre.rho_max
        return lambda r: _scale(
            logdet_grad_chebyshev(float(r), coeffs, rmin_cb, rmax_cb)
        )

    if method == "aaa":
        from ._aaa import aaa_logdet_precompute

        pre = aaa_logdet_precompute(W_sparse, rho_min=rho_min, rho_max=rho_max)
        sp_z = pre.support_points
        sp_f = pre.support_values
        w = pre.weights
        return lambda r: _scale(logdet_grad_aaa(float(r), sp_z, sp_f, w))

    if method == "chol_aaa":
        from ._aaa import chol_aaa_logdet_precompute

        pre = chol_aaa_logdet_precompute(W_sparse, rho_min=rho_min, rho_max=rho_max)
        sp_z = pre.support_points
        sp_f = pre.support_values
        w = pre.weights
        return lambda r: _scale(logdet_grad_aaa(float(r), sp_z, sp_f, w))

    if method == "grid_spline":
        from ._grid_spline import grid_spline_logdet_precompute

        pre = grid_spline_logdet_precompute(W_sparse, rho_min=rho_min, rho_max=rho_max)
        deriv = pre.spline.derivative()
        return lambda r: _scale(
            float(deriv(min(max(float(r), pre.rho_min), pre.rho_max)))
        )

    if method == "slq":
        # Match make_logdet_numpy_fn: its slq value form is the Chebyshev-
        # converted series, so the consistent gradient differentiates that.
        pre = slq_logdet_precompute(W_sparse)
        cheb = slq_to_chebyshev_coeffs(
            pre, W=W_sparse, order=20, rho_min=rho_min, rho_max=rho_max
        )
        coeffs, rmin_cb, rmax_cb = cheb["coeffs"], cheb["rmin"], cheb["rmax"]
        return lambda r: _scale(
            logdet_grad_chebyshev(float(r), coeffs, rmin_cb, rmax_cb)
        )

    raise ValueError(f"Unsupported logdet method: {method!r}")


def make_logdet_grad_numpy_vec_fn(
    W_sparse,
    eigs: np.ndarray | None,
    method: str | None,
    rho_min: float = -1.0,
    rho_max: float = 1.0,
    T: int = 1,
):
    """Vectorized ``(rho_arr) -> np.ndarray`` **logdet-gradient** evaluator.

    Batched form of :func:`make_logdet_grad_numpy_fn`: returns ``g(ρ)`` for a
    whole array of ρ draws at once.  Used by the spatial-impacts path, where the
    per-draw direct-effect trace quantities are exactly ``g(ρ)`` (see
    :meth:`neighbayes.models.base.SpatialModel._batch_mean_diag`), so impacts
    never need the O(n³) eigendecomposition when a fast logdet method exists.
    """
    from ._resolvent import logdet_grad_chebyshev

    T = int(T)
    n = eigs.shape[0] if eigs is not None else int(W_sparse.shape[0])
    method = resolve_logdet_method(method, n=n, W=W_sparse)

    def _scale(g):
        return g if T == 1 else T * g

    if method == "eigenvalue":
        if eigs is None:
            eigs = np.linalg.eigvals(np.asarray(W_sparse.toarray(), dtype=np.float64))
        _eigs = np.asarray(eigs, dtype=np.complex128)

        def _vec_eig_grad(rho_arr, chunk: int = 1024):
            rho_arr = np.asarray(rho_arr, dtype=np.float64)
            out = np.empty(rho_arr.shape[0], dtype=np.float64)
            for i in range(0, rho_arr.shape[0], chunk):
                r = rho_arr[i : i + chunk, None]
                block = _eigs[None, :] / (1.0 - r * _eigs[None, :])
                out[i : i + chunk] = -block.sum(axis=1).real
            return _scale(out)

        return _vec_eig_grad

    # Chebyshev-family (chebyshev / cheb_cholesky / cheb_stochastic / slq) share
    # the Clenshaw-derivative, which vectorizes naturally over ρ arrays.
    if method in ("chebyshev", "cheb_stochastic", "cheb_cholesky", "lu_cheb", "slq"):
        if method == "chebyshev":
            out = chebyshev(W_sparse, order=20, rmin=rho_min, rmax=rho_max, eigs=eigs)
            coeffs, rmin_cb, rmax_cb = out["coeffs"], out["rmin"], out["rmax"]
        elif method == "cheb_stochastic":
            coeffs, rmin_cb, rmax_cb = _cheb_stochastic_coeffs(
                W_sparse, rho_min, rho_max
            )
        elif method in ("cheb_cholesky", "lu_cheb"):
            pre = _cheb_precompute_for(method)(
                W_sparse, order=None, rho_min=rho_min, rho_max=rho_max
            )
            coeffs, rmin_cb, rmax_cb = pre.coeffs, pre.rho_min, pre.rho_max
        else:  # slq
            pre = slq_logdet_precompute(W_sparse)
            cheb = slq_to_chebyshev_coeffs(
                pre, W=W_sparse, order=20, rho_min=rho_min, rho_max=rho_max
            )
            coeffs, rmin_cb, rmax_cb = cheb["coeffs"], cheb["rmin"], cheb["rmax"]

        def _vec_cheb_grad(rho_arr):
            rho_arr = np.asarray(rho_arr, dtype=np.float64)
            return _scale(
                np.asarray(logdet_grad_chebyshev(rho_arr, coeffs, rmin_cb, rmax_cb))
            )

        return _vec_cheb_grad

    if method == "aaa":
        from ._aaa import aaa_logdet_precompute

        pre = aaa_logdet_precompute(W_sparse, rho_min=rho_min, rho_max=rho_max)
        z = pre.support_points.astype(np.float64)
        f = pre.support_values.astype(np.float64)
        w = pre.weights.astype(np.float64)

        def _vec_aaa_grad(rho_arr):
            # Barycentric derivative (N'D - N D')/D², broadcast over ρ.
            rho_arr = np.asarray(rho_arr, dtype=np.float64)
            diff = rho_arr[:, None] - z[None, :]  # (G, m)
            inv = w[None, :] / diff
            inv2 = w[None, :] / diff**2
            n_val = (inv * f[None, :]).sum(axis=1)
            d_val = inv.sum(axis=1)
            dn = -(inv2 * f[None, :]).sum(axis=1)
            dd = -inv2.sum(axis=1)
            return _scale((dn * d_val - n_val * dd) / d_val**2)

        return _vec_aaa_grad

    if method == "chol_aaa":
        from ._aaa import chol_aaa_logdet_precompute

        pre = chol_aaa_logdet_precompute(W_sparse, rho_min=rho_min, rho_max=rho_max)
        z = pre.support_points.astype(np.float64)
        f = pre.support_values.astype(np.float64)
        w = pre.weights.astype(np.float64)

        def _vec_chol_aaa_grad(rho_arr):
            rho_arr = np.asarray(rho_arr, dtype=np.float64)
            diff = rho_arr[:, None] - z[None, :]
            inv = w[None, :] / diff
            inv2 = w[None, :] / diff**2
            n_val = (inv * f[None, :]).sum(axis=1)
            d_val = inv.sum(axis=1)
            dn = -(inv2 * f[None, :]).sum(axis=1)
            dd = -inv2.sum(axis=1)
            return _scale((dn * d_val - n_val * dd) / d_val**2)

        return _vec_chol_aaa_grad

    if method == "grid_spline":
        from ._grid_spline import grid_spline_logdet_precompute

        pre = grid_spline_logdet_precompute(W_sparse, rho_min=rho_min, rho_max=rho_max)
        deriv = pre.spline.derivative()

        def _vec_grid_spline_grad(rho_arr):
            r = np.clip(np.asarray(rho_arr, dtype=np.float64), pre.rho_min, pre.rho_max)
            return _scale(np.asarray(deriv(r), dtype=np.float64))

        return _vec_grid_spline_grad

    raise ValueError(f"Unsupported logdet method: {method!r}")


# ---------------------------------------------------------------------------
# NumPy vectorized factory
# ---------------------------------------------------------------------------


def make_logdet_numpy_vec_fn(
    W_sparse,
    eigs: np.ndarray | None,
    method: str | None,
    rho_min: float = -1.0,
    rho_max: float = 1.0,
    T: int = 1,
):
    """Return a vectorized numpy ``(rho_arr: np.ndarray) -> np.ndarray`` logdet evaluator."""
    T = int(T)
    n = eigs.shape[0] if eigs is not None else int(W_sparse.shape[0])
    method = resolve_logdet_method(method, n=n, W=W_sparse)

    if method == "eigenvalue":
        if eigs is None:
            eigs = np.linalg.eigvals(np.asarray(W_sparse.toarray(), dtype=np.float64))
        _eigs = np.asarray(eigs, dtype=np.complex128)

        def _vec_eigenvalue(rho_arr: np.ndarray) -> np.ndarray:
            rho_arr = np.asarray(rho_arr, dtype=np.float64)
            val = np.sum(
                np.log(np.abs(1.0 - rho_arr[:, None] * _eigs[None, :])), axis=1
            )
            return val if T == 1 else T * val

        return _vec_eigenvalue

    if method == "chebyshev":
        out = chebyshev(W_sparse, order=20, rmin=rho_min, rmax=rho_max, eigs=eigs)
        coeffs = out["coeffs"].astype(np.float64)
        rmin_cb, rmax_cb = float(out["rmin"]), float(out["rmax"])
        return lambda rho_arr: _clenshaw_vec(coeffs, rho_arr, rmin_cb, rmax_cb, T)

    if method == "cheb_stochastic":
        coeffs, rmin_cb, rmax_cb = _cheb_stochastic_coeffs(W_sparse, rho_min, rho_max)
        return lambda rho_arr: _clenshaw_vec(coeffs, rho_arr, rmin_cb, rmax_cb, T)

    if method in ("cheb_cholesky", "lu_cheb"):
        from ._chol_cheb import chol_cheb_logdet_eval_vec

        # Cholesky-Chebyshev: exact logdet via sparse Cholesky at Chebyshev nodes.
        pre = _cheb_precompute_for(method)(
            W_sparse, order=None, rho_min=rho_min, rho_max=rho_max
        )

        def _vec_cheb_chol(rho_arr: np.ndarray) -> np.ndarray:
            vals = chol_cheb_logdet_eval_vec(pre, np.asarray(rho_arr, dtype=np.float64))
            return vals if T == 1 else T * vals

        return _vec_cheb_chol

    if method == "aaa":
        from ._aaa import aaa_logdet_eval_vec, aaa_logdet_precompute

        pre = aaa_logdet_precompute(W_sparse, rho_min=rho_min, rho_max=rho_max)

        def _vec_aaa(rho_arr: np.ndarray) -> np.ndarray:
            vals = aaa_logdet_eval_vec(pre, np.asarray(rho_arr, dtype=np.float64))
            return vals if T == 1 else T * vals

        return _vec_aaa

    if method == "chol_aaa":
        from ._aaa import aaa_logdet_eval_vec, chol_aaa_logdet_precompute

        pre = chol_aaa_logdet_precompute(W_sparse, rho_min=rho_min, rho_max=rho_max)

        def _vec_chol_aaa(rho_arr: np.ndarray) -> np.ndarray:
            vals = aaa_logdet_eval_vec(pre, np.asarray(rho_arr, dtype=np.float64))
            return vals if T == 1 else T * vals

        return _vec_chol_aaa

    if method == "grid_spline":
        from ._grid_spline import grid_spline_logdet_precompute

        pre = grid_spline_logdet_precompute(W_sparse, rho_min=rho_min, rho_max=rho_max)

        def _vec_grid_spline(rho_arr: np.ndarray) -> np.ndarray:
            r = np.clip(np.asarray(rho_arr, dtype=np.float64), pre.rho_min, pre.rho_max)
            vals = np.asarray(pre.spline(r), dtype=np.float64)
            return vals if T == 1 else T * vals

        return _vec_grid_spline

    if method == "slq":
        # SLQ precompute → Chebyshev coefficients → vectorized Clenshaw
        pre = slq_logdet_precompute(W_sparse)
        cheb = slq_to_chebyshev_coeffs(
            pre, W=W_sparse, order=20, rho_min=rho_min, rho_max=rho_max
        )
        coeffs = cheb["coeffs"].astype(np.float64)
        rmin_cb, rmax_cb = float(cheb["rmin"]), float(cheb["rmax"])
        return lambda rho_arr: _clenshaw_vec(coeffs, rho_arr, rmin_cb, rmax_cb, T)

    raise ValueError(f"Unsupported logdet method: {method!r}")


# ---------------------------------------------------------------------------
# PyTensor factory
# ---------------------------------------------------------------------------


def make_logdet_fn(
    W,
    method: str | None = None,
    rho_min: float = -1.0,
    rho_max: float = 1.0,
    T: int = 1,
):
    """Return a function ``(rho) -> pytensor`` log|I - ρW| expression.

    Accepts a dense matrix, sparse matrix, or 1-D eigenvalue array.
    """
    T = int(T)

    if sp.issparse(W):
        W_sparse = W.tocsr().astype(np.float64)
        method = resolve_logdet_method(method, n=W_sparse.shape[0], W=W_sparse)
        if method == "cheb_stochastic":
            # Stochastic Chebyshev → Clenshaw-like eval (PyTensor-differentiable)
            # Build Cheb coeffs at Chebyshev nodes in ρ-space, then use logdet_chebyshev
            # for the differentiable Clenshaw evaluation.
            coeffs_np, rmin_cb, rmax_cb = _cheb_stochastic_coeffs(
                W_sparse, rho_min, rho_max
            )

            def _cheb_stoch_sparse(rho):
                val = logdet_chebyshev(rho, coeffs_np, rmin=rmin_cb, rmax=rmax_cb)
                return val if T == 1 else T * val

            return _cheb_stoch_sparse
        if method in ("cheb_cholesky", "lu_cheb"):
            # Cholesky-Chebyshev: exact logdet via sparse Cholesky at Chebyshev nodes.
            pre = _cheb_precompute_for(method)(
                W_sparse, order=None, rho_min=rho_min, rho_max=rho_max
            )
            coeffs_np = pre.coeffs.astype(np.float64)
            rmin_cb, rmax_cb = pre.rho_min, pre.rho_max

            def _cheb_chol_sparse(rho):
                val = logdet_chebyshev(rho, coeffs_np, rmin=rmin_cb, rmax=rmax_cb)
                return val if T == 1 else T * val

            return _cheb_chol_sparse
        if method == "aaa":
            from ._aaa import aaa_logdet_precompute

            # AAA rational approximation for non-symmetric W.
            # PyTensor-differentiable via barycentric evaluation on
            # precomputed support points and weights.
            pre = aaa_logdet_precompute(W_sparse, rho_min=rho_min, rho_max=rho_max)
            sp_z = pre.support_points.astype(np.float64)
            sp_f = pre.support_values.astype(np.float64)
            w = pre.weights.astype(np.float64)

            def _aaa_sparse(rho):
                # Barycentric formula using PyTensor operations
                import pytensor.tensor as pt

                diff = rho - sp_z  # (m,)
                n_val = pt.sum(w * sp_f / diff)
                d_val = pt.sum(w / diff)
                val = n_val / d_val
                return val if T == 1 else T * val

            return _aaa_sparse
        if method == "chol_aaa":
            from ._aaa import chol_aaa_logdet_precompute

            # Cholesky-based AAA: same barycentric evaluation as "aaa"
            # but support values come from sparse Cholesky (~2× cheaper
            # than KLU for symmetrizable W).
            pre = chol_aaa_logdet_precompute(W_sparse, rho_min=rho_min, rho_max=rho_max)
            sp_z = pre.support_points.astype(np.float64)
            sp_f = pre.support_values.astype(np.float64)
            w = pre.weights.astype(np.float64)

            def _chol_aaa_sparse(rho):
                import pytensor.tensor as pt

                diff = rho - sp_z
                n_val = pt.sum(w * sp_f / diff)
                d_val = pt.sum(w / diff)
                val = n_val / d_val
                return val if T == 1 else T * val

            return _chol_aaa_sparse
        if method == "slq":
            # SLQ precompute → Chebyshev coefficients → differentiable Clenshaw
            pre = slq_logdet_precompute(W_sparse)
            cheb = slq_to_chebyshev_coeffs(
                pre, W=W_sparse, order=20, rho_min=rho_min, rho_max=rho_max
            )
            coeffs_np = cheb["coeffs"]
            rmin_cb, rmax_cb = cheb["rmin"], cheb["rmax"]

            def _slq_sparse(rho):
                val = logdet_chebyshev(rho, coeffs_np, rmin=rmin_cb, rmax=rmax_cb)
                return val if T == 1 else T * val

            return _slq_sparse
        if method == "chebyshev":
            out = chebyshev(W_sparse, order=20, rmin=rho_min, rmax=rho_max)
            coeffs_np = out["coeffs"]
            rmin_cb, rmax_cb = out["rmin"], out["rmax"]

            def _cheb_sparse(rho):
                val = logdet_chebyshev(rho, coeffs_np, rmin=rmin_cb, rmax=rmax_cb)
                return val if T == 1 else T * val

            return _cheb_sparse
        # eigenvalue path: materialize dense
        W = np.asarray(W_sparse.toarray(), dtype=np.float64)
    else:
        # A 1-D eigenvalue array keeps its dtype: row-standardized directed W
        # has a complex spectrum, and both downstream consumers
        # (``logdet_eigenvalue``, ``chebyshev``) need the imaginary parts.
        # Narrowing to float64 here silently drops them and biases log|I-rhoW|.
        W = np.asarray(W)
        if W.ndim != 1:
            W = W.astype(np.float64)

    if W.ndim == 1:
        # 1-D eigenvalue array
        eigs = W
        method = resolve_logdet_method(method, n=eigs.shape[0])
        if method == "eigenvalue":
            if T == 1:
                return lambda rho: logdet_eigenvalue(rho, eigs)
            return lambda rho: T * logdet_eigenvalue(rho, eigs)
        if method == "chebyshev":
            out = chebyshev(None, order=20, rmin=rho_min, rmax=rho_max, eigs=eigs)
            coeffs_np = out["coeffs"]
            rmin_cb, rmax_cb = out["rmin"], out["rmax"]

            def _cheb_eig(rho):
                val = logdet_chebyshev(rho, coeffs_np, rmin=rmin_cb, rmax=rmax_cb)
                return val if T == 1 else T * val

            return _cheb_eig
        raise ValueError(f"Unsupported logdet method for eigenvalue input: {method!r}")

    # 2-D dense matrix
    W_dense = W
    method = resolve_logdet_method(method, n=W_dense.shape[0], W=W_dense)
    if method == "cheb_stochastic":
        # Stochastic Chebyshev → Clenshaw (PyTensor-differentiable via ρ-space coeffs)
        coeffs_np, rmin_cb, rmax_cb = _cheb_stochastic_coeffs(W_dense, rho_min, rho_max)

        def _cheb_stoch_dense(rho):
            val = logdet_chebyshev(rho, coeffs_np, rmin=rmin_cb, rmax=rmax_cb)
            return val if T == 1 else T * val

        return _cheb_stoch_dense
    if method in ("cheb_cholesky", "lu_cheb"):
        # Cholesky-Chebyshev: exact logdet via sparse Cholesky at Chebyshev nodes.
        pre = _cheb_precompute_for(method)(
            sp.csr_matrix(W_dense), order=None, rho_min=rho_min, rho_max=rho_max
        )
        coeffs_np = pre.coeffs.astype(np.float64)
        rmin_cb, rmax_cb = pre.rho_min, pre.rho_max

        def _cheb_chol_dense(rho):
            val = logdet_chebyshev(rho, coeffs_np, rmin=rmin_cb, rmax=rmax_cb)
            return val if T == 1 else T * val

        return _cheb_chol_dense
    if method == "aaa":
        from ._aaa import aaa_logdet_precompute

        pre = aaa_logdet_precompute(
            sp.csc_matrix(W_dense), rho_min=rho_min, rho_max=rho_max
        )
        sp_z = pre.support_points.astype(np.float64)
        sp_f = pre.support_values.astype(np.float64)
        w = pre.weights.astype(np.float64)

        def _aaa_dense(rho):
            import pytensor.tensor as pt

            diff = rho - sp_z
            n_val = pt.sum(w * sp_f / diff)
            d_val = pt.sum(w / diff)
            val = n_val / d_val
            return val if T == 1 else T * val

        return _aaa_dense
    if method == "chol_aaa":
        from ._aaa import chol_aaa_logdet_precompute

        pre = chol_aaa_logdet_precompute(
            sp.csc_matrix(W_dense), rho_min=rho_min, rho_max=rho_max
        )
        sp_z = pre.support_points.astype(np.float64)
        sp_f = pre.support_values.astype(np.float64)
        w = pre.weights.astype(np.float64)

        def _chol_aaa_dense(rho):
            import pytensor.tensor as pt

            diff = rho - sp_z
            n_val = pt.sum(w * sp_f / diff)
            d_val = pt.sum(w / diff)
            val = n_val / d_val
            return val if T == 1 else T * val

        return _chol_aaa_dense
    if method == "slq":
        # SLQ precompute → Chebyshev coefficients → differentiable Clenshaw
        pre = slq_logdet_precompute(W_dense)
        cheb = slq_to_chebyshev_coeffs(
            pre, W=sp.csr_matrix(W_dense), order=20, rho_min=rho_min, rho_max=rho_max
        )
        coeffs_np = cheb["coeffs"]
        rmin_cb, rmax_cb = cheb["rmin"], cheb["rmax"]

        def _slq_dense(rho):
            val = logdet_chebyshev(rho, coeffs_np, rmin=rmin_cb, rmax=rmax_cb)
            return val if T == 1 else T * val

        return _slq_dense
    if method == "eigenvalue":
        eigs = np.linalg.eigvals(W_dense)
        if T == 1:
            return lambda rho: logdet_eigenvalue(rho, eigs)
        return lambda rho: T * logdet_eigenvalue(rho, eigs)
    if method == "chebyshev":
        out = chebyshev(W_dense, order=20, rmin=rho_min, rmax=rho_max)
        coeffs_np = out["coeffs"]
        rmin_cb, rmax_cb = out["rmin"], out["rmax"]

        def _cheb_dense(rho):
            val = logdet_chebyshev(rho, coeffs_np, rmin=rmin_cb, rmax=rmax_cb)
            return val if T == 1 else T * val

        return _cheb_dense
    raise ValueError(f"Unsupported logdet method: {method!r}")


# ---------------------------------------------------------------------------
# Caching
# ---------------------------------------------------------------------------


def _hash_array(arr: np.ndarray) -> str:
    arr_c = np.ascontiguousarray(arr)
    h = hashlib.blake2b(digest_size=16)
    h.update(str(arr_c.dtype).encode("ascii"))
    h.update(np.asarray(arr_c.shape, dtype=np.int64).tobytes())
    h.update(arr_c.view(np.uint8))
    return h.hexdigest()


def _logdet_w_signature(W) -> tuple:
    if sp.issparse(W):
        W_csr = W.tocsr().astype(np.float64)
        h = hashlib.blake2b(digest_size=16)
        h.update(np.asarray(W_csr.shape, dtype=np.int64).tobytes())
        h.update(np.asarray([W_csr.nnz], dtype=np.int64).tobytes())
        h.update(np.ascontiguousarray(W_csr.indptr).view(np.uint8))
        h.update(np.ascontiguousarray(W_csr.indices).view(np.uint8))
        h.update(np.ascontiguousarray(W_csr.data).view(np.uint8))
        return ("sparse", W_csr.shape, int(W_csr.nnz), h.hexdigest())
    W_arr = np.asarray(W, dtype=np.float64)
    if W_arr.ndim == 1:
        return ("eigs", W_arr.shape, _hash_array(W_arr))
    if W_arr.ndim == 2:
        return ("dense", W_arr.shape, _hash_array(W_arr))
    raise ValueError(f"Unsupported W with ndim={W_arr.ndim}")


def clear_logdet_fn_cache() -> None:
    """Clear the shared caches of logdet callables and stochastic Chebyshev coefficients."""
    _LOGDET_FN_CACHE.clear()
    _CHEB_STOCHASTIC_COEFFS.clear()


def get_cached_logdet_fn(
    W,
    method: str | None = None,
    rho_min: float = -1.0,
    rho_max: float = 1.0,
    T: int = 1,
):
    """Return a shared cached ``make_logdet_fn`` callable."""
    T = int(T)
    if sp.issparse(W):
        n_w = int(W.shape[0])
    else:
        n_w = int(np.asarray(W).shape[0])
    resolved_method = resolve_logdet_method(method, n=n_w, W=W)

    # NOTE: the key carries no rng/seed component by design — the stochastic
    # precomputes (SLQ, cheb_stochastic) default to a fixed rng(0), so the
    # cached callable is deterministic for a given (W, method, bounds, T).
    key = (
        _logdet_w_signature(W),
        resolved_method,
        float(rho_min),
        float(rho_max),
        T,
    )
    fn = _LOGDET_FN_CACHE.get(key)
    if fn is not None:
        _LOGDET_FN_CACHE.move_to_end(key)
        return fn

    fn = make_logdet_fn(
        W, method=resolved_method, rho_min=rho_min, rho_max=rho_max, T=T
    )
    _LOGDET_FN_CACHE[key] = fn
    if len(_LOGDET_FN_CACHE) > _LOGDET_FN_CACHE_MAXSIZE:
        _LOGDET_FN_CACHE.popitem(last=False)
    return fn
