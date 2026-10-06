"""Regression tests: a reused Krylov basis must keep its own Horner origin.

The reduced-form samplers evaluate ``U(ρ) = (I − ρW)⁻¹X`` as a Taylor series
about the ρ the basis was *built* at::

    U(ρ_basis + Δρ) ≈ Σ_j Δρ^j V_j

When a basis is reused across sweeps, the current ρ has drifted away from
``ρ_basis`` (up to the reuse threshold).  Measuring ``Δρ`` from the current ρ
instead of from ``ρ_basis`` silently evaluates U at the wrong point — the
error is first-order in the drift, so it dwarfs the truncation error the
series was designed around.
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp

from neighbayes.samplers.negbin_reduced._core import (
    _build_krylov_basis,
    _eval_U_from_basis,
)

REUSE_THRESHOLD = 0.15  # matches krylov_reuse_threshold


def _ring_W(n):
    W = sp.lil_matrix((n, n))
    for i in range(n):
        W[i, (i + 1) % n] = 1.0
        W[i, (i - 1) % n] = 1.0
    W = sp.csr_matrix(W)
    return (sp.diags(1.0 / np.asarray(W.sum(axis=1)).ravel()) @ W).tocsc()


def _exact_U(rho, W, X):
    n = X.shape[0]
    return sp.linalg.spsolve((sp.eye(n) - rho * W).tocsc(), X)


@pytest.mark.parametrize("rho_c", [0.2, 0.5])
@pytest.mark.parametrize("drift", [0.05, 0.10, REUSE_THRESHOLD])
def test_offset_must_be_measured_from_basis_center(rho_c, drift):
    """Δρ from ρ_basis is accurate; Δρ from the drifted current ρ is not."""
    n = 300
    W = _ring_W(n)
    rng = np.random.default_rng(0)
    X = rng.standard_normal((n, 3))

    basis = _build_krylov_basis(rho_c, X, W, n, degree=12)
    assert basis.rho_basis == rho_c

    rho_current = rho_c + drift  # where the chain drifted to
    rho_eval = rho_current + 0.05  # a slice candidate
    exact = _exact_U(rho_eval, W, X)

    correct = _eval_U_from_basis(basis, rho_eval - basis.rho_basis)
    wrong = _eval_U_from_basis(basis, rho_eval - rho_current)

    err_correct = np.linalg.norm(correct - exact) / np.linalg.norm(exact)
    err_wrong = np.linalg.norm(wrong - exact) / np.linalg.norm(exact)

    assert err_correct < 1e-5, f"basis-centered offset inaccurate: {err_correct:.2e}"
    assert err_wrong > 100 * err_correct, (
        "measuring the offset from the drifted current ρ should be far worse; "
        f"got wrong={err_wrong:.2e} vs correct={err_correct:.2e}"
    )


def test_flow_jax_slice_builds_its_basis_at_the_current_rho():
    """The JAX flow sweep carries no basis between slices.

    A ρ_k basis depends on the other two ρ's, which move every slice; reusing
    one evaluated the density at stale values of them and biased the
    posterior.  Each slice therefore builds its basis at the state it slices
    from, so the Horner origin is the current ρ_k.  Assert on the source that
    no basis or basis center rides in the sweep state.
    """
    import inspect

    from neighbayes.samplers.negbin_reduced import _flow_jax

    src = inspect.getsource(_flow_jax._flow_sweep)
    for stale in ("V_stack_prev", "rd_basis", "ro_basis", "rw_basis"):
        assert stale not in src, f"the flow sweep carries {stale} between slices"
    assert "rho_basis=rho_k," in src


class TestSafeRadius:
    """The Neumann radius shrinks with ρ_c; a fixed dmax is not safe."""

    def test_radius_tightens_as_rho_grows(self):
        from neighbayes.samplers.negbin_reduced._core import krylov_safe_radius

        radii = [krylov_safe_radius(r) for r in (0.0, 0.3, 0.6, 0.9)]
        assert radii == sorted(radii, reverse=True), radii
        assert radii[0] == pytest.approx(0.4)  # configured dmax kept where safe
        assert radii[-1] < 0.1  # heavily tightened near the stability edge

    def test_basis_radius_tracks_rho(self):
        """The basis reports a radius that tightens as ρ_c nears stability."""
        n = 300
        W = _ring_W(n)
        X = np.random.default_rng(2).standard_normal((n, 3))
        radii = [
            _build_krylov_basis(r, X, W, n, degree=12).safe_dmax
            for r in (0.1, 0.5, 0.9)
        ]
        assert radii == sorted(radii, reverse=True), radii

    @pytest.mark.parametrize("rho_c", [0.3, 0.6, 0.8, 0.9])
    def test_series_accurate_at_the_clamped_radius(self, rho_c):
        """Evaluating out to ``basis.safe_dmax`` must stay accurate for every ρ_c.

        With the unclamped 0.4 default this diverges outright for ρ_c ≥ 0.6,
        since the radius of convergence is 1 − ρ_c.
        """
        n = 300
        W = _ring_W(n)
        X = np.random.default_rng(1).standard_normal((n, 3))
        basis = _build_krylov_basis(rho_c, X, W, n, degree=12)

        # ``safe_dmax`` is the raw convergence radius; consumers evaluate out
        # to ``min(krylov_dmax, safe_dmax)``, so test at that effective edge.
        effective = min(0.4, basis.safe_dmax)
        rho = rho_c + effective
        exact = _exact_U(rho, W, X)
        approx = _eval_U_from_basis(basis, effective)
        rel = np.linalg.norm(approx - exact) / np.linalg.norm(exact)
        assert rel < 1e-2, f"rho_c={rho_c}: rel err {rel:.2e} at clamped radius"
