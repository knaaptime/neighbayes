r"""Exact unrestricted flow log-determinant from ``n × n`` trace moments.

The unrestricted flow system is ``A = I_N − ρ_d(I⊗W) − ρ_o(W⊗I) − ρ_w(W⊗W)``,
``N = n²``.  The three Kronecker terms commute, so for any polynomial basis
``{φ_p}``

.. math::

    \log|A| = \sum_{i,j} f(\lambda_i, \lambda_j) \approx
        \sum_{p,q} a_{pq}\,\operatorname{tr}\varphi_p(W)\,\operatorname{tr}\varphi_q(W),
    \qquad f(x, y) = \log(1 - \rho_o x - \rho_d y - \rho_w x y),

where ``a_pq`` are the coefficients of ``f`` in that basis and ``tr φ_p(W)`` are
**n × n** traces.  The traces depend on ``W`` alone, so they are computed once,
exactly, and every evaluation afterwards costs a 2-D transform of size
``(P+1)²`` — independent of ``n``, with nothing ``N``-sized anywhere.  The three
gradients ``∂ log|A| / ∂ρ_k`` are the same contraction applied to ``∂f/∂ρ_k``.

This differs from the old ``"traces"`` method, which estimated moments of the
``N × N`` operator stochastically and amplified that noise through the
multinomial coefficients.  Here the Kronecker structure factors every moment into
a product of exact ``n × n`` traces, so there is no noise to amplify.

Two bases:

* ``"chebyshev"`` (undirected W, real spectrum).  With ``w`` a bound on the
  spectral radius, ``τ_p = tr T_p(W/w)`` and ``a_pq`` from a 2-D DCT of ``f`` on
  a Chebyshev grid.  Convergence is geometric and survives the stability wall:
  roughly ``exp(−P·√(2(1−r)))`` for ``r = |ρ_d|+|ρ_o|+|ρ_w|``.  This is the
  stochastic-Chebyshev idea (:mod:`._cheb_stochastic`) with exact moments.
* ``"taylor"`` (any W, including directed with a complex spectrum inside the
  disc of radius ``w``).  ``t_p = tr W^p`` and ``a_pq`` from the power series of
  ``log(1 − u)``; the error decays only like ``r^P``, so it is slow near the wall.

The moments are computed with a column-block recursion, ``O(P · nnz(W) · n)``
work and ``O(n · block)`` memory; they are extended on demand.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from scipy.fft import dctn

__all__ = ["FlowKronTraceLogdet"]

#: bytes allowed for one column block of the trace recursion
_BLOCK_BYTES = 256 * 1024 * 1024


class FlowKronTraceLogdet:
    """``log|A(ρ)|`` and its gradient for the unrestricted flow filter, exactly.

    Parameters
    ----------
    W : sparse matrix or array
        The ``n × n`` spatial weights.
    basis : {"auto", "chebyshev", "taylor"}, default "auto"
        ``"auto"`` picks Chebyshev when ``W`` is D-symmetrizable (undirected, so
        its spectrum is real) and Taylor otherwise.
    tol : float, default 1e-13
        Relative size of the neglected coefficient tail.
    max_order : int, default 4096
        Largest polynomial order used; beyond it an evaluation raises.
    init_order : int, default 64
        Moments computed up front (extended on demand).

    Calling the instance with ``(rho_d, rho_o, rho_w)`` returns
    ``(value, grad)`` with ``grad`` ordered ``(d, o, w)``, the contract of
    :class:`~neighbayes.samplers.gaussian._flow_resolvent.FlowResolventTarget`.
    """

    def __init__(
        self,
        W,
        basis: str = "auto",
        tol: float = 1e-13,
        max_order: int = 4096,
        init_order: int = 64,
    ):
        self.W = sp.csr_matrix(W, dtype=np.float64)
        self.n = self.W.shape[0]
        self.w = float(abs(self.W).sum(axis=1).max()) if self.W.nnz else 1.0
        if basis == "auto":
            from ._config import _is_symmetric_W

            basis = "chebyshev" if _is_symmetric_W(self.W) else "taylor"
        if basis not in ("chebyshev", "taylor"):
            raise ValueError(
                f"basis must be 'auto', 'chebyshev' or 'taylor', got {basis!r}"
            )
        self.basis = basis
        # Spectrum of W as λ = c + h·ξ, ξ ∈ [−1, 1] (Chebyshev) or |ξ| ≤ 1 (Taylor).
        if basis == "chebyshev":
            self.c, self.h = self._real_spectral_interval()
        else:
            self.c, self.h = 0.0, self.w
        self.tol = float(tol)
        self.max_order = int(max_order)
        self._moments = np.empty(0)
        self._order = 16  # order of the last converged evaluation
        self._ensure_moments(int(init_order) + 1)

    def _real_spectral_interval(self) -> tuple[float, float]:
        """Centre and half-width of an interval bracketing W's real spectrum.

        W is D-symmetrizable, so it is similar to a symmetric S.  Lanczos Ritz
        values sit inside the spectrum, so each end is pushed outward by its
        residual norm (a rigorous bound for symmetric S) plus a small margin.
        """
        import scipy.sparse.linalg as spla

        from ._chol_cheb import _d_symmetrize

        S = sp.csr_matrix(_d_symmetrize(self.W))
        n = S.shape[0]
        if n <= 64:
            ev = np.linalg.eigvalsh(S.toarray())
            lo, hi = float(ev[0]), float(ev[-1])
            pad = 1e-9 * max(1.0, abs(lo), abs(hi))
            lo, hi = lo - pad, hi + pad
        else:
            ends = []
            for which in ("SA", "LA"):
                val, vec = spla.eigsh(S, k=1, which=which, tol=1e-10)
                resid = float(np.linalg.norm(S @ vec[:, 0] - val[0] * vec[:, 0]))
                ends.append((float(val[0]), resid))
            (lo, r_lo), (hi, r_hi) = ends
            margin = 1e-8 * max(1.0, abs(lo), abs(hi))
            lo, hi = lo - r_lo - margin, hi + r_hi + margin
        # Never wider than the max-row-sum bound, which is also rigorous.
        lo, hi = max(lo, -self.w), min(hi, self.w)
        return 0.5 * (lo + hi), 0.5 * (hi - lo)

    # -- moments -----------------------------------------------------------
    def _ensure_moments(self, P: int) -> None:
        """Exact ``tr φ_p(W)`` for ``p = 0..P`` (recomputed at a larger size)."""
        if P < self._moments.size:
            return
        P = max(P, 2 * max(self._moments.size - 1, 1))
        n = self.n
        # ξ-operator (W − cI)/h: spectrum inside [−1, 1] (or the unit disc).
        A = (self.W - self.c * sp.eye(n, format="csr")) / self.h
        m = np.zeros(P + 1)
        m[0] = n
        block = max(1, min(n, _BLOCK_BYTES // (16 * n)))
        cheb = self.basis == "chebyshev"
        for a in range(0, n, block):
            b = min(n, a + block)
            rows = np.arange(a, b)
            cols = np.arange(b - a)
            Z_prev = np.zeros((n, b - a))
            Z_prev[rows, cols] = 1.0
            Z = A @ Z_prev  # φ_1 = T_1 = x for both bases (scaled)
            m[1] += Z[rows, cols].sum()
            for p in range(2, P + 1):
                if cheb:
                    Z, Z_prev = 2.0 * (A @ Z) - Z_prev, Z
                else:
                    Z = A @ Z
                m[p] += Z[rows, cols].sum()
        self._moments = m

    # -- coefficients ------------------------------------------------------
    def _cheb_coeffs(self, rd, ro, rw, P):
        """2-D Chebyshev coefficients of ``f`` and ``∂f/∂ρ_{d,o,w}`` (order P)."""
        M = P + 1
        x = np.cos(np.pi * (np.arange(M) + 0.5) / M)
        X, Y = np.meshgrid(
            x, x, indexing="ij"
        )  # X ↔ origin eigenvalue, Y ↔ destination
        lx, ly = self.c + self.h * X, self.c + self.h * Y
        g = 1.0 - ro * lx - rd * ly - rw * lx * ly
        if np.any(g <= 0.0):
            raise ValueError("flow parameters outside the stability region of W")
        vals = np.stack([np.log(g), -ly / g, -lx / g, -(lx * ly) / g])
        c = dctn(vals, type=2, axes=(1, 2)) / (M * M)
        c[:, 0, :] *= 0.5
        c[:, :, 0] *= 0.5
        return c  # (4, M, M): value, d, o, w

    def _taylor_coeffs(self, rd, ro, rw, P):
        """Power-series coefficients of ``f`` and its partials in (x/w, y/w)."""
        w = self.w
        a_o, a_d, a_w = ro * w, rd * w, rw * w * w  # coefficients on scaled variables
        K = P + 1
        C = np.zeros((K + 1, K + 1))  # of 1/(1 − a_o x − a_d y − a_w xy)
        C[0, 0] = 1.0
        idx = np.arange(K + 1)
        for k in range(1, 2 * K + 1):
            p = idx[(idx <= k) & (k - idx <= K)]
            q = k - p
            v = np.zeros(p.size)
            m = p > 0
            v[m] += a_o * C[p[m] - 1, q[m]]
            m = q > 0
            v[m] += a_d * C[p[m], q[m] - 1]
            m = (p > 0) & (q > 0)
            v[m] += a_w * C[p[m] - 1, q[m] - 1]
            C[p, q] = v
        D = np.zeros(
            (K, K)
        )  # of log(1 − …):  p·D_pq = −(a_o C_{p−1,q} + a_w C_{p−1,q−1})
        Cq1 = np.zeros((P, K))
        Cq1[:, 1:] = C[:P, :P]
        D[1:, :] = -(a_o * C[:P, :K] + a_w * Cq1) / np.arange(1, K)[:, None]
        D[0, 1:] = -(a_d * C[0, :P]) / np.arange(1, K)
        # ∂f/∂ρ_o = −λ_x/g → coefficient (p, q) is −w·C_{p−1,q}; similarly for d, w.
        Gd = np.zeros((K, K))
        Go = np.zeros((K, K))
        Gw = np.zeros((K, K))
        Go[1:, :] = -w * C[:P, :K]
        Gd[:, 1:] = -w * C[:K, :P]
        Gw[1:, 1:] = -w * w * C[:P, :P]
        return np.stack([D, Gd, Go, Gw])

    # -- evaluation --------------------------------------------------------
    def _tail(self, c, P):
        """Largest coefficient in the last two rows/columns, relative to the largest.

        Normalized over all four expansions (value and gradients): at ρ = 0 the
        value's expansion is identically zero while the gradients' are not.
        """
        head = np.abs(c).max()
        k = max(0, P - 1)
        edge = max(np.abs(c[:, k:, :]).max(), np.abs(c[:, :, k:]).max())
        return edge / max(head, 1e-300)

    def __call__(self, rho_d, rho_o, rho_w):
        rd, ro, rw = float(rho_d), float(rho_o), float(rho_w)
        if self.basis == "taylor":
            r = (abs(rd) + abs(ro)) * self.w + abs(rw) * self.w**2
            if r >= 1.0:
                raise ValueError(
                    "flow parameters outside the Taylor convergence region"
                )
            P = (
                1
                if r == 0.0
                else int(np.ceil(np.log(self.tol * (1.0 - r)) / np.log(r)))
            )
            if P > self.max_order:
                raise ValueError(
                    f"Taylor flow log-determinant needs order {P} > max_order at "
                    f"|rho| sum {r:.4f}; use the Chebyshev basis (undirected W) or "
                    "the resolvent estimator."
                )
            coeffs = self._taylor_coeffs(rd, ro, rw, P)
        else:
            P = self._order
            while True:
                coeffs = self._cheb_coeffs(rd, ro, rw, P)
                if self._tail(coeffs, P) < self.tol:
                    break
                if P >= self.max_order:
                    raise ValueError(
                        f"Chebyshev flow log-determinant did not converge by order "
                        f"{self.max_order}; the parameters are too close to the "
                        "stability wall."
                    )
                P = min(2 * P, self.max_order)
            # Shrink the starting order when this one had room to spare.
            self._order = (
                P if self._tail(coeffs, P // 2) >= self.tol else max(16, P // 2)
            )
        self._ensure_moments(P + 2)
        # Taylor gradients use moments one order higher (∂f/∂ρ shifts the index).
        m = self._moments[: coeffs.shape[1]]
        out = np.einsum("kpq,p,q->k", coeffs, m, m)
        return float(out[0]), out[1:4].copy()
