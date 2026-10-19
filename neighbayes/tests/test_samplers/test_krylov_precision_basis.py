"""Accuracy tests for the Krylov precision basis (structural Gibbs).

Validates that the shift-invert Krylov basis for P(ρ) = base − ρG1 + ρ²G2
reproduces the direct CHOLMOD solve and logdet across the radius it reports
as safe.

There is deliberately no timing benchmark here.  A per-candidate
microbenchmark makes the basis look like a large win, but only by assuming
more ρ candidates per slice step than the sampler actually draws: break-even
is ~10 candidates and a real slice step uses ~8.  End to end the structural
path measures ~0.78x, because ``jnp.where`` in the JAX log-density evaluates
the direct solve on every candidate regardless (measured on the structural
NB sampler, since removed).  That is why ``krylov_degree`` defaults to 0,
and why a benchmark asserting a speedup here would be encoding a claim the
sampler does not deliver.
"""

from __future__ import annotations

import time

import numpy as np
import scipy.sparse as sp

from neighbayes.samplers._utils._spatial_normal import (
    CholmodFactor,
    KrylovPrecisionBasis,
    build_precision_krylov_basis,
    eval_precision_logdet_from_basis,
    eval_precision_solve_from_basis,
    lanczos_logdet,
)

# The radius the samplers actually configure (``cache.krylov_dmax``).  Tests
# assert accuracy out to here, not just near the center.
DMAX = 0.4


def _make_precision_components(n, rho_c, omega, seed=0):
    """Build base, G1, G2 and a CHOLMOD factor for a ring-lattice W."""
    rng = np.random.default_rng(seed)
    W_dense = np.zeros((n, n))
    for i in range(n):
        W_dense[i, (i + 1) % n] = 1.0
        W_dense[i, (i - 1) % n] = 1.0
    row_sums = W_dense.sum(axis=1, keepdims=True)
    W_dense = W_dense / row_sums
    W = sp.csr_matrix(W_dense)
    W_sym = (W + W.T).tocsr()
    WtW = (W.T @ W).tocsr()
    base = (sp.eye(n, format="csr") + sp.diags(omega, format="csr")).tocsr()
    P0 = sp.eye(n, format="csr") + 0.5 * W_sym + 0.25 * WtW
    cholmod_factor = CholmodFactor(P0)
    return base, W_sym, WtW, cholmod_factor


def _direct_solve_logdet(rho, base, W_sym, WtW, rhs, cholmod_factor):
    """Direct CHOLMOD path: factor P(ρ), solve, read logdet."""
    P = (base - rho * W_sym + rho**2 * WtW).tocsc()
    cholmod_factor.factorize(P)
    sol = cholmod_factor.solve(rhs)
    logdet = cholmod_factor.logdet()
    return sol, logdet


class TestKrylovPrecisionBasisEquivalence:
    """Numerical equivalence within the Krylov radius."""

    def test_solve_matches_direct_within_radius(self):
        n, rho_c, degree = 200, 0.3, 12
        omega = np.ones(n) * 0.5
        base, W_sym, WtW, cholmod_factor = _make_precision_components(n, rho_c, omega)
        rng = np.random.default_rng(1)
        rhs = rng.standard_normal((n, 4))

        basis = build_precision_krylov_basis(
            rho_c,
            base,
            W_sym,
            WtW,
            rhs,
            degree=degree,
            cholmod_factor=cholmod_factor,
            dmax=DMAX,
            rng=rng,
        )
        # Exercise the *whole* configured radius, not just its inner edge —
        # dmax is what the samplers actually evaluate over.
        for drho in [0.001, 0.01, 0.05, 0.1, 0.2, 0.3, DMAX]:
            rho = rho_c + drho
            sol_kry = eval_precision_solve_from_basis(basis, drho)
            sol_dir, _ = _direct_solve_logdet(
                rho, base, W_sym, WtW, rhs, cholmod_factor
            )
            rel_err = np.linalg.norm(sol_kry - sol_dir) / np.linalg.norm(sol_dir)
            assert rel_err < 1e-4, f"drho={drho}: rel_err={rel_err:.2e} > 1e-4"

    def test_logdet_matches_direct_within_radius(self):
        """Accuracy is asserted in *absolute nats*, deliberately.

        ``log|P|`` grows like O(n), so a relative-error bound hides an
        arbitrarily large absolute error as n grows — and it is the absolute
        error that lands in the slice sampler's log-density.
        """
        n, rho_c, degree = 200, 0.3, 12
        omega = np.ones(n) * 0.5
        base, W_sym, WtW, cholmod_factor = _make_precision_components(n, rho_c, omega)
        rng = np.random.default_rng(2)
        rhs = rng.standard_normal((n, 2))

        basis = build_precision_krylov_basis(
            rho_c,
            base,
            W_sym,
            WtW,
            rhs,
            degree=degree,
            cholmod_factor=cholmod_factor,
            dmax=DMAX,
            rng=rng,
        )
        for drho in [0.001, 0.01, 0.05, 0.1, 0.2, 0.3, DMAX]:
            rho = rho_c + drho
            logdet_kry = eval_precision_logdet_from_basis(basis, drho)
            _, logdet_dir = _direct_solve_logdet(
                rho, base, W_sym, WtW, rhs, cholmod_factor
            )
            abs_err = abs(logdet_kry - logdet_dir)
            assert abs_err < 1.0, f"drho={drho}: logdet off by {abs_err:.3f} nats"

    def test_logdet_is_deterministic_in_rho(self):
        """Repeated evaluation at one ρ must return the identical value.

        The slice sampler's shrinkage loop assumes a fixed density surface;
        re-drawing Hutchinson probes per candidate would break that
        invariant even though each draw is individually unbiased.
        """
        n, rho_c = 200, 0.3
        omega = np.ones(n) * 0.5
        base, W_sym, WtW, cholmod_factor = _make_precision_components(n, rho_c, omega)
        rng = np.random.default_rng(11)
        basis = build_precision_krylov_basis(
            rho_c,
            base,
            W_sym,
            WtW,
            rng.standard_normal((n, 2)),
            degree=12,
            cholmod_factor=cholmod_factor,
            dmax=DMAX,
            rng=rng,
        )
        vals = [eval_precision_logdet_from_basis(basis, 0.2) for _ in range(25)]
        assert max(vals) == min(vals), f"logdet varies by {max(vals) - min(vals):.3e}"

    def test_quadratic_term_is_not_dropped(self):
        """The Δρ²G₂ term is part of the exact re-centering, not a remainder.

        Linearizing ``P(ρ_c+Δρ) = P_c − Δρ·G + Δρ²·G₂`` to ``P_c − Δρ·G``
        leaves a model error no Krylov degree can remove, so raising the
        degree must keep driving the error down.
        """
        n, rho_c = 200, 0.3
        omega = np.ones(n) * 0.5
        base, W_sym, WtW, cholmod_factor = _make_precision_components(n, rho_c, omega)
        rng = np.random.default_rng(12)
        rhs = rng.standard_normal((n, 2))
        sol_dir, _ = _direct_solve_logdet(
            rho_c + 0.3, base, W_sym, WtW, rhs, cholmod_factor
        )
        errs = []
        for degree in (4, 8, 16):
            basis = build_precision_krylov_basis(
                rho_c,
                base,
                W_sym,
                WtW,
                rhs,
                degree=degree,
                cholmod_factor=cholmod_factor,
                dmax=DMAX,
                rng=rng,
            )
            sol = eval_precision_solve_from_basis(basis, 0.3)
            errs.append(np.linalg.norm(sol - sol_dir) / np.linalg.norm(sol_dir))
        assert errs[0] > errs[1] > errs[2], f"error did not fall with degree: {errs}"
        assert errs[-1] < 1e-6, f"degree-16 error {errs[-1]:.2e} suggests a fixed bias"

    def test_basis_carries_solver_and_matvec(self):
        n, rho_c, degree = 100, 0.3, 8
        omega = np.ones(n) * 0.5
        base, W_sym, WtW, cholmod_factor = _make_precision_components(n, rho_c, omega)
        rhs = np.ones((n, 2))
        basis = build_precision_krylov_basis(
            rho_c,
            base,
            W_sym,
            WtW,
            rhs,
            degree=degree,
            cholmod_factor=cholmod_factor,
        )
        assert callable(basis.G_matvec)
        assert callable(basis.solve_at_c)
        # The cached solve_at_c should reproduce V_stack[0]
        v0_check = basis.solve_at_c(rhs)
        assert np.allclose(v0_check, basis.V_stack[0])


class TestJaxPathUnaffected:
    """The Krylov basis is NumPy-only; the JAX path must still work."""

    def test_jax_import_still_works(self):
        # Smoke test: importing the module should not pull JAX into the
        # Krylov path, and the existing JAX log-density core should still
        # be importable.
        from neighbayes.samplers._utils._spatial_normal import (
            _jax_log_density_core,
        )

        assert callable(_jax_log_density_core)
