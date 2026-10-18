"""Resolvent-trace / logdet-gradient core (backend-parametrized).

Every entry point returns the **logdet gradient**

    g(ρ) = d/dρ log|I − ρW| = −tr(W (I − ρW)⁻¹)                       (Jacobi)

as a closed form of a representation that some existing precompute already
produces (eigenvalues, Chebyshev coefficients, or AAA support points and
weights).  Because each formula is the *analytic derivative of the
corresponding value surrogate*, it agrees with ``jax.grad`` / ``pytensor.grad``
of that surrogate to machine precision — the numpy path in
:func:`neighbayes._logdet.make_logdet_grad_numpy_fn` and the autodiff paths
therefore compute the same object, from the same coefficients, with no per-ρ
solves.

One object, three consumers (see the module plan):

* **gradient-based samplers** (NUTS/MALA on any backend) need ``g(ρ)``;
* **spatial impacts** need the resolvent trace ``tr(W(I−ρW)⁻¹) = −g(ρ)`` and,
  in mean form, ``mean(λ/(1−ρλ)) = −g(ρ)/n`` — the same quantity
  :func:`neighbayes.diagnostics.impacts._chunked_eig_means` builds from
  dense eigenvalues today, available here at any ``n`` without an
  eigendecomposition;
* the value log-density adds ``g(ρ)`` as its only extra ρ-term over OLS.

Backend parametrization: pass ``xp=numpy`` (default) or ``xp=jax.numpy``.  The
functions use only elementwise arithmetic, ``sum``, ``real``, ``zeros_like`` —
the shared subset of both array libraries — so a single implementation backs
every target.  This is a leaf module (numpy-only imports) to avoid import
cycles; the ``xp`` argument is what makes it library-agnostic.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "logdet_grad_eigenvalue",
    "logdet_grad_chebyshev",
    "logdet_grad_aaa",
    "clenshaw_deriv_x",
]


def logdet_grad_eigenvalue(rho, eigs, *, xp=np):
    """``g(ρ) = −Σ Re(λᵢ / (1 − ρλᵢ))`` from the eigenvalues of ``W``.

    Exact.  ``eigs`` may be complex (directed ``W``); only the real part of the
    resolvent trace survives because ``log|I − ρW|`` is real.
    """
    lam = xp.asarray(eigs)
    res = lam / (1.0 - rho * lam)
    return -xp.sum(xp.real(res))


def clenshaw_deriv_x(coeffs, x, *, xp=np):
    """``d/dx Σ_j coeffs[j] T_j(x)`` via a differentiated Clenshaw recurrence.

    Carries ``b_k`` and ``β_k = db_k/dx`` through the same recurrence used by
    :func:`neighbayes._logdet._clenshaw.clenshaw_scalar`, so the result is the
    exact derivative of that evaluation (matches autodiff of the value form).
    """
    m = len(coeffs)
    z = xp.zeros_like(x)
    if m <= 1:
        return z
    b_next = z  # b_{m}   = 0
    b_curr = z + coeffs[m - 1]  # b_{m-1} = c_{m-1}
    bd_next = z  # β_{m}   = 0
    bd_curr = z  # β_{m-1} = 0
    for k in range(m - 2, 0, -1):
        b_new = 2.0 * x * b_curr - b_next + coeffs[k]
        bd_new = 2.0 * b_curr + 2.0 * x * bd_curr - bd_next
        b_next, b_curr = b_curr, b_new
        bd_next, bd_curr = bd_curr, bd_new
    # After the loop: b_curr=b_1, b_next=b_2, bd_curr=β_1, bd_next=β_2.
    # f(x) = c_0 + x·b_1 − b_2  ⇒  f'(x) = b_1 + x·β_1 − β_2.
    return b_curr + x * bd_curr - bd_next


def logdet_grad_chebyshev(rho, coeffs, rmin=-1.0, rmax=1.0, *, xp=np):
    """``g(ρ)`` for a Chebyshev-in-ρ logdet surrogate ``Σ_j coeffs[j] T_j(x(ρ))``.

    ``x(ρ) = (2ρ − rmax − rmin)/(rmax − rmin)`` so ``dx/dρ = 2/(rmax − rmin)``.
    Serves the ``chebyshev``, ``cheb_cholesky``, ``lu_cheb`` and
    ``cheb_stochastic`` keys (all share this coefficient representation).
    """
    x = (2.0 * rho - rmax - rmin) / (rmax - rmin)
    dxdrho = 2.0 / (rmax - rmin)
    return dxdrho * clenshaw_deriv_x(coeffs, x, xp=xp)


def logdet_grad_aaa(rho, support_points, support_values, weights, *, xp=np):
    """``g(ρ)`` for the AAA barycentric rational ``L(ρ) = N(ρ)/D(ρ)``.

    ``N = Σ wⱼfⱼ/(ρ−zⱼ)``, ``D = Σ wⱼ/(ρ−zⱼ)``; the analytic derivative is
    ``(N'D − ND')/D²`` with ``N' = −Σ wⱼfⱼ/(ρ−zⱼ)²`` and
    ``D' = −Σ wⱼ/(ρ−zⱼ)²`` — exactly the derivative of the barycentric value
    form used by the AAA evaluators.
    """
    z = xp.asarray(support_points)
    f = xp.asarray(support_values)
    w = xp.asarray(weights)
    diff = rho - z
    inv = w / diff
    inv2 = w / diff**2
    n_val = xp.sum(inv * f)
    d_val = xp.sum(inv)
    dn = -xp.sum(inv2 * f)
    dd = -xp.sum(inv2)
    return (dn * d_val - n_val * dd) / d_val**2
