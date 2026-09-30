"""Structured separable NB flow sampler: its conditionals equal brute-force algebra.

The structured sampler never forms ``U = A⁻¹X`` (``n² × k``): rank-one design
columns stay rank one and full-rank columns are ``n × n`` two-sided solves.
These tests build ``U`` densely from ``A = L_o ⊗ L_d`` on a small grid and check
that the ρ log-density (rank-one block integrated out) and the β-conditional
moments computed from structured parts match it exactly.
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp

from neighbayes.graph import flow_design_matrix_with_orig
from neighbayes.samplers.negbin_reduced._flow_structured import (
    _beta_gram,
    _FilterSolver,
    _rho_log_density,
    classify_flow_design,
)
from neighbayes.tests.helpers import make_rook_W

N_SIDE = 3  # n = 9 regions, N = 81 flows


def _setup(seed=0):
    rng = np.random.default_rng(seed)
    W = sp.csr_matrix(make_rook_W(N_SIDE))
    n = W.shape[0]
    xy = np.column_stack(np.divmod(np.arange(n), N_SIDE)).astype(float)
    dist = np.sqrt(((xy[:, None] - xy[None, :]) ** 2).sum(-1))
    design = flow_design_matrix_with_orig(
        rng.standard_normal((n, 2)),
        rng.standard_normal((n, 2)),
        col_names=["a", "b"],
        dist=dist,
        log_distance=True,
    )
    X = np.asarray(design.combined, dtype=float)
    omega = rng.uniform(0.2, 2.0, n * n)
    z = rng.standard_normal(n * n)
    return rng, W, n, X, omega, z


def _dense_U(W, n, rd, ro, X):
    eye = np.eye(n)
    A = np.kron(eye - ro * W.toarray(), eye - rd * W.toarray())
    return np.linalg.solve(A, X)


def test_design_classification():
    _, _, n, X, _, _ = _setup()
    s = classify_flow_design(X, n)
    assert len(s.cheap_cols) + len(s.full_cols) == X.shape[1]
    # intercept + 2 destination + 2 origin attributes are rank one
    assert len(s.cheap_cols) == 5
    # every rank-one column is reproduced by its factors
    for j, c in enumerate(s.cheap_cols):
        Y = np.outer(s.U[:, s.u_idx[0, j]], s.V[:, s.v_idx[0, j]])
        np.testing.assert_array_equal(Y.ravel(), X[:, c])


@pytest.mark.parametrize("rd, ro", [(0.3, 0.2), (-0.4, 0.5), (0.0, 0.0)])
def test_rho_density_matches_dense(rd, ro):
    rng, W, n, X, omega, z = _setup(1)
    s = classify_flow_design(X, n)
    cc, fc = s.cheap_cols, s.full_cols
    beta_full = rng.standard_normal(len(fc))
    mu_c = rng.standard_normal(len(cc))
    prec_c = rng.uniform(0.5, 2.0, len(cc))

    # Structured: L_o⁻¹ B_full L_d⁻ᵀ and transformed rank-one factors.
    Lo, Ld = (
        _FilterSolver(sp.csc_matrix(W)).at(ro),
        _FilterSolver(sp.csc_matrix(W)).at(rd),
    )
    B_full = sum(b * s.F[i] for b, i in zip(beta_full, s.f_idx[0]))
    eta_full = Ld.solve(Lo.solve(B_full).T).T
    Om, Z = omega.reshape(n, n), z.reshape(n, n)
    got = _rho_log_density(
        [Om], [Z - eta_full], Lo.solve(s.U), Ld.solve(s.V), s, mu_c, prec_c
    )

    # Dense: β_c ~ N(μ_c, diag(1/prec_c)) integrated out of N(z | Uβ, Ω⁻¹).
    U = _dense_U(W, n, rd, ro, X)
    Uc = U[:, cc]
    r = z - U[:, fc] @ beta_full - Uc @ mu_c
    M = Uc.T @ (omega[:, None] * Uc) + np.diag(prec_c)
    v = Uc.T @ (omega * r)
    want = -0.5 * np.linalg.slogdet(M)[1] - 0.5 * (
        r @ (omega * r) - v @ np.linalg.solve(M, v)
    )
    np.testing.assert_allclose(got, want, rtol=1e-10, atol=1e-10)


def test_beta_gram_matches_dense():
    _, W, n, X, omega, z = _setup(2)
    s = classify_flow_design(X, n)
    rd, ro = 0.35, -0.2
    Lo, Ld = (
        _FilterSolver(sp.csc_matrix(W)).at(ro),
        _FilterSolver(sp.csc_matrix(W)).at(rd),
    )
    Ufull = [Ld.solve(Lo.solve(F).T).T for F in s.F]
    G, h = _beta_gram(
        [omega.reshape(n, n)], [z.reshape(n, n)], Lo.solve(s.U), Ld.solve(s.V), Ufull, s
    )
    U = _dense_U(W, n, rd, ro, X)
    np.testing.assert_allclose(G, U.T @ (omega[:, None] * U), rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(h, U.T @ (omega * z), rtol=1e-10, atol=1e-10)


def _panel_setup(T=3, seed=4):
    """Time-first panel: attributes vary by period, log-distance does not."""
    rng = np.random.default_rng(seed)
    W = sp.csr_matrix(make_rook_W(N_SIDE))
    n = W.shape[0]
    xy = np.column_stack(np.divmod(np.arange(n), N_SIDE)).astype(float)
    dist = np.sqrt(((xy[:, None] - xy[None, :]) ** 2).sum(-1))
    blocks = [
        flow_design_matrix_with_orig(
            rng.standard_normal((n, 2)),
            rng.standard_normal((n, 2)),
            col_names=["a", "b"],
            dist=dist,
            log_distance=True,
        ).combined
        for _ in range(T)
    ]
    X = np.vstack([np.asarray(b, dtype=float) for b in blocks])
    omega = rng.uniform(0.2, 2.0, T * n * n)
    z = rng.standard_normal(T * n * n)
    return rng, W, n, X, omega, z


def _dense_panel_U(W, n, T, rd, ro, X):
    N = n * n
    eye = np.eye(n)
    A = np.kron(eye - ro * W.toarray(), eye - rd * W.toarray())
    return np.vstack([np.linalg.solve(A, X[t * N : (t + 1) * N]) for t in range(T)])


def test_panel_classification_dedups_time_invariant_columns():
    T = 3
    _, _, n, X, _, _ = _panel_setup(T)
    s = classify_flow_design(X, n, T)
    assert s.T == T
    # log-distance and the intra dummy repeat every period: stored once each
    assert len(s.F) < T * len(s.full_cols)
    N = n * n
    for t in range(T):
        for j, c in enumerate(s.cheap_cols):
            Y = np.outer(s.U[:, s.u_idx[t, j]], s.V[:, s.v_idx[t, j]])
            np.testing.assert_array_equal(Y.ravel(), X[t * N : (t + 1) * N, c])
        for j, c in enumerate(s.full_cols):
            np.testing.assert_array_equal(
                s.F[s.f_idx[t, j]].ravel(), X[t * N : (t + 1) * N, c]
            )


def test_panel_rho_density_and_beta_gram_match_dense():
    T = 3
    rng, W, n, X, omega, z = _panel_setup(T)
    s = classify_flow_design(X, n, T)
    cc, fc = s.cheap_cols, s.full_cols
    rd, ro = 0.3, -0.25
    beta_full = rng.standard_normal(len(fc))
    mu_c = rng.standard_normal(len(cc))
    prec_c = rng.uniform(0.5, 2.0, len(cc))
    Lo = _FilterSolver(sp.csc_matrix(W)).at(ro)
    Ld = _FilterSolver(sp.csc_matrix(W)).at(rd)
    A, B = Lo.solve(s.U), Ld.solve(s.V)
    Ufull = [Ld.solve(Lo.solve(F).T).T for F in s.F]
    Oms, Zs = omega.reshape(T, n, n), z.reshape(T, n, n)
    eta_full = [
        sum(b * Ufull[i] for b, i in zip(beta_full, s.f_idx[t])) for t in range(T)
    ]
    got = _rho_log_density(
        list(Oms), [Zs[t] - eta_full[t] for t in range(T)], A, B, s, mu_c, prec_c
    )

    U = _dense_panel_U(W, n, T, rd, ro, X)
    Uc = U[:, cc]
    r = z - U[:, fc] @ beta_full - Uc @ mu_c
    M = Uc.T @ (omega[:, None] * Uc) + np.diag(prec_c)
    v = Uc.T @ (omega * r)
    want = -0.5 * np.linalg.slogdet(M)[1] - 0.5 * (
        r @ (omega * r) - v @ np.linalg.solve(M, v)
    )
    np.testing.assert_allclose(got, want, rtol=1e-10, atol=1e-10)

    G, h = _beta_gram(list(Oms), list(Zs), A, B, Ufull, s)
    np.testing.assert_allclose(G, U.T @ (omega[:, None] * U), rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(h, U.T @ (omega * z), rtol=1e-10, atol=1e-10)


@pytest.mark.requires_jax
@pytest.mark.parametrize("T", [1, 3])
def test_jax_kernels_match_dense(T):
    """The JAX sweep's ρ density and β moments equal the dense algebra."""
    import jax.numpy as jnp

    from neighbayes._jax_dispatch import ensure_x64
    from neighbayes.models.priors import FlowReducedGibbsPriors
    from neighbayes.samplers.negbin_reduced._flow_structured_jax import (
        make_structured_sweep,
    )

    ensure_x64()
    rng, W, n, X, omega, z = _panel_setup(T, seed=7)
    s = classify_flow_design(X, n, T)
    cc, fc = s.cheap_cols, s.full_cols
    k = s.k
    mu0 = rng.standard_normal(k)
    sd0 = rng.uniform(0.7, 1.5, k)
    priors = FlowReducedGibbsPriors(
        beta_mu=mu0,
        beta_sigma=sd0,
        alpha_sigma=2.5,
        alpha_nu=3.0,
        rho_lower=-0.999,
        rho_upper=0.999,
    )
    sweep, _, _ = make_structured_sweep(
        np.zeros(X.shape[0]), sp.csc_matrix(W), s, priors
    )
    kern = sweep.kernels
    rd, ro = 0.3, -0.25
    beta_full = rng.standard_normal(len(fc))
    Oms, Zs = jnp.asarray(omega.reshape(T, n, n)), jnp.asarray(z.reshape(T, n, n))

    # ρ_d candidate, as the sweep builds it: B_key → L_o⁻¹ B_key → · L_d⁻ᵀ
    keys = sorted({tuple(int(i) for i in s.f_idx[t]) for t in range(T)})
    B_keys = jnp.asarray(
        np.stack([sum(b * s.F[i] for b, i in zip(beta_full, key)) for key in keys])
    )
    P, A = kern["solve_left"](ro, B_keys, jnp.asarray(s.U))
    eta_f, B = kern["solve_right"](rd, P, jnp.asarray(s.V))
    got = float(kern["rho_log_density"](Oms, Zs, eta_f, A, B))

    U = _dense_panel_U(W, n, T, rd, ro, X)
    Uc = U[:, cc]
    r = z - U[:, fc] @ beta_full - Uc @ mu0[cc]
    M = Uc.T @ (omega[:, None] * Uc) + np.diag(1 / sd0[cc] ** 2)
    v = Uc.T @ (omega * r)
    want = -0.5 * np.linalg.slogdet(M)[1] - 0.5 * (
        r @ (omega * r) - v @ np.linalg.solve(M, v)
    )
    np.testing.assert_allclose(got, want, rtol=1e-10, atol=1e-10)

    half, A = kern["solve_left"](ro, jnp.asarray(np.stack(s.F)), jnp.asarray(s.U))
    Ufull, B = kern["solve_right"](rd, half, jnp.asarray(s.V))
    G, h = kern["beta_gram"](Oms, Zs, Ufull, A, B)
    np.testing.assert_allclose(G, U.T @ (omega[:, None] * U), rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(h, U.T @ (omega * z), rtol=1e-10, atol=1e-10)
