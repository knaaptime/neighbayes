"""The JAX/sparsax precision basis must match its NumPy twin.

This path is gated on ``krylov_degree > 0``, which no default enables, so it
had never executed under test.  These tests drive it directly and pin it to
the NumPy implementation, which is checked against exact CHOLMOD elsewhere.
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp

from neighbayes.samplers._utils._sparsax_utils import precompute_sparsax_pattern
from neighbayes.samplers._utils._spatial_normal import (
    CholmodFactor,
    _sparsax_factor_ops_available,
    build_precision_krylov_basis,
    build_precision_krylov_basis_jax,
    eval_precision_logdet_from_basis,
    eval_precision_logdet_from_basis_jax,
    eval_precision_solve_from_basis,
    eval_precision_solve_from_basis_jax,
)

pytestmark = pytest.mark.skipif(
    not _sparsax_factor_ops_available(),
    reason="sparsax factor-reuse primitives unavailable",
)

DMAX = 0.4
DEGREE = 12


def _queen(side):
    n = side * side
    rows, cols = [], []
    for i in range(side):
        for j in range(side):
            for di in (-1, 0, 1):
                for dj in (-1, 0, 1):
                    if di or dj:
                        a, b = i + di, j + dj
                        if 0 <= a < side and 0 <= b < side:
                            rows.append(i * side + j)
                            cols.append(a * side + b)
    W = sp.coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n)).tocsr()
    return (sp.diags(1.0 / np.asarray(W.sum(axis=1)).ravel()) @ W).tocsc()


def _both_bases(side=16, rho_c=0.3, omega_val=0.5, seed=0, numpy=True):
    """Build the JAX and NumPy bases over the same P(ρ), plus exact pieces.

    ``numpy=False`` builds only the JAX basis (the NumPy one also factors
    through sparsax when it is installed).
    """
    import jax
    import jax.numpy as jnp

    jax.config.update("jax_enable_x64", True)

    W = _queen(side)
    n = W.shape[0]
    omega = np.ones(n) * omega_val
    rng = np.random.default_rng(seed)
    rhs = rng.standard_normal((n, 3))

    pat = precompute_sparsax_pattern(W, n)
    Ai = jnp.asarray(pat["Ai"])
    Aj = jnp.asarray(pat["Aj"])
    W_sym_vals = jnp.asarray(pat["W_sym_vals"])
    WtW_vals = jnp.asarray(pat["WtW_vals"])

    Ax_c = jnp.zeros(Ai.shape[0], dtype=jnp.float64)
    Ax_c = Ax_c.at[jnp.asarray(pat["diag_idx"])].add(1.0 + jnp.asarray(omega))
    Ax_c = Ax_c - rho_c * W_sym_vals + rho_c**2 * WtW_vals
    G_vals = W_sym_vals - 2.0 * rho_c * WtW_vals

    jax_basis = build_precision_krylov_basis_jax(
        Ai,
        Aj,
        Ax_c,
        G_vals,
        WtW_vals,
        jnp.asarray(rhs),
        n=n,
        degree=DEGREE,
        dmax=DMAX,
    )

    if not numpy:
        return jax_basis, None, None
    W_sym = (W + W.T).tocsc()
    WtW = (W.T @ W).tocsc()
    base = (sp.eye(n, format="csc") + sp.diags(omega)).tocsc()
    P0 = (sp.eye(n, format="csr") + 0.5 * W_sym + 0.25 * WtW).tocsc()
    np_basis = build_precision_krylov_basis(
        rho_c,
        base,
        W_sym,
        WtW,
        rhs,
        degree=DEGREE,
        cholmod_factor=CholmodFactor(P0),
        dmax=DMAX,
        rng=rng,
    )
    return jax_basis, np_basis, (base, W_sym, WtW, P0, rhs, rho_c)


def _exact(rho, base, W_sym, WtW, P0, rhs):
    P = (base - rho * W_sym + rho**2 * WtW).tocsc()
    factor = CholmodFactor(P0)
    factor.factorize(P)
    return factor.solve(rhs), factor.logdet()


class TestJaxMatchesNumpy:
    def test_solve_matches_numpy_and_exact(self):
        (_, V, _, _), np_basis, parts = _both_bases()
        base, W_sym, WtW, P0, rhs, rho_c = parts
        for drho in (0.05, 0.2, DMAX):
            exact, _ = _exact(rho_c + drho, base, W_sym, WtW, P0, rhs)
            jax_sol = np.asarray(eval_precision_solve_from_basis_jax(V, drho))
            np_sol = eval_precision_solve_from_basis(np_basis, drho)
            denom = np.linalg.norm(exact)
            assert np.linalg.norm(jax_sol - exact) / denom < 1e-4
            # The two implementations must agree far more tightly than
            # either agrees with the truncated truth.
            assert np.linalg.norm(jax_sol - np_sol) / denom < 1e-10

    def test_symmetric_matvec_uses_both_triangles(self):
        """Regression: the COO pattern stores one triangle only.

        Scattering just the stored entries drops every mirrored
        off-diagonal contribution, which left the solve ~1e-1 wrong while
        the logdet still looked fine.
        """
        (_, V, _, _), _, parts = _both_bases()
        base, W_sym, WtW, P0, rhs, rho_c = parts
        exact, _ = _exact(rho_c + 0.05, base, W_sym, WtW, P0, rhs)
        sol = np.asarray(eval_precision_solve_from_basis_jax(V, 0.05))
        rel = np.linalg.norm(sol - exact) / np.linalg.norm(exact)
        assert rel < 1e-8, f"half-matvec regression: rel err {rel:.2e}"

    def test_logdet_matches_exact_in_nats(self):
        (_, _, coefs, _), _, parts = _both_bases()
        base, W_sym, WtW, P0, rhs, rho_c = parts
        for drho in (0.05, 0.2, DMAX):
            _, exact = _exact(rho_c + drho, base, W_sym, WtW, P0, rhs)
            got = float(eval_precision_logdet_from_basis_jax(coefs, drho))
            assert abs(got - exact) < 1.0, f"drho={drho}: {abs(got - exact):.3f} nats"

    def test_safe_dmax_agrees_with_numpy(self):
        (_, _, _, safe), np_basis, _ = _both_bases()
        assert float(safe) == pytest.approx(np_basis.safe_dmax, rel=1e-6)

    def test_build_refactors_only_for_logdet_nodes(self):
        """The recurrence must reuse the held factor, not refactor per step.

        Only the exact logdet nodes are allowed to factor again.

        ``omega_val`` is unique to this test so sparsax's content-addressed
        cache cannot serve a factor built by another test and deflate the
        count.
        """
        import time

        import jax
        import sparsax

        def settled_count():
            # The counter is process-wide, and work an earlier test dispatched
            # asynchronously may still be landing: wait until it stops moving.
            count = sparsax.factorization_count()
            while True:
                time.sleep(0.05)
                now = sparsax.factorization_count()
                if now == count:
                    return count
                count = now

        before = settled_count()
        # JAX dispatches asynchronously: wait for every factorization to run
        # before reading the counter.
        jax.block_until_ready(_both_bases(omega_val=0.3717, seed=3, numpy=False)[0])
        # 1 for P_c + 4 Chebyshev logdet nodes; the degree-12 recurrence and
        # the seed solve all run against the held factor.
        assert sparsax.factorization_count() - before == 1 + 4


class TestSafeRadiusUnderSmallOmega:
    """P = diag(ω) + AᵀA/σ² → singular as ω → 0, shrinking the radius."""

    def test_radius_shrinks_and_keeps_error_bounded(self):
        worst = 0.0
        for omega_val, rho_c in ((0.02, 0.9), (0.05, 0.7), (0.5, 0.9)):
            (_, V, _, safe), _, parts = _both_bases(
                rho_c=rho_c, omega_val=omega_val, seed=1
            )
            base, W_sym, WtW, P0, rhs, _ = parts
            d = float(safe)
            exact, _ = _exact(rho_c + d, base, W_sym, WtW, P0, rhs)
            sol = np.asarray(eval_precision_solve_from_basis_jax(V, d))
            worst = max(worst, np.linalg.norm(sol - exact) / np.linalg.norm(exact))
        assert worst < 1e-2, f"worst relative error at the clamped radius: {worst:.2e}"
