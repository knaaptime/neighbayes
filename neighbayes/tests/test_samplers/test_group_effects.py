"""The group-effect projector against explicit one-hot dummies."""

import numpy as np
import pytest
from scipy.stats import multivariate_normal

from neighbayes.samplers._utils._group_effects import GroupEffects, GroupProjector


def _problem(seed, n_rows=60, n_groups=9, k=3, ragged=True):
    rng = np.random.default_rng(seed)
    if ragged:
        # Unequal group sizes, every group present.
        g = np.concatenate(
            [np.arange(n_groups), rng.integers(0, n_groups, n_rows - n_groups)]
        )
        rng.shuffle(g)
    else:
        g = np.tile(np.arange(n_groups), n_rows // n_groups)
    spec = GroupEffects(g, n_groups, mu=0.7, sigma=1.3)
    omega = rng.gamma(2.0, 0.5, size=g.size)
    U = rng.normal(size=(g.size, k))
    z = rng.normal(size=g.size)
    D = np.zeros((g.size, n_groups))
    D[np.arange(g.size), g] = 1.0
    return rng, spec, omega, U, z, D


def _dense_schur(spec, omega, D):
    """Marginal precision of the rows with ``c`` integrated out."""
    Om = np.diag(omega)
    P = D.T @ Om @ D + np.eye(D.shape[1]) / spec.sigma**2
    return Om - Om @ D @ np.linalg.solve(P, D.T @ Om)


@pytest.mark.parametrize("ragged", [True, False])
def test_gram_cross_quad_match_dense_schur(ragged):
    _, spec, omega, U, z, D = _problem(0, ragged=ragged)
    proj = GroupProjector(spec, omega)
    Om_t = _dense_schur(spec, omega, D)
    r = z - spec.mu
    np.testing.assert_allclose(proj.gram(U), U.T @ Om_t @ U, atol=1e-10)
    np.testing.assert_allclose(proj.cross(U, r), U.T @ Om_t @ r, atol=1e-10)
    np.testing.assert_allclose(proj.quad(r), r @ Om_t @ r, atol=1e-10)


def test_collapsed_density_matches_dense_marginal_up_to_constant():
    """β and c integrated out: the projector density tracks the exact marginal.

    Several designs ``U`` stand in for ``A(ρ)⁻¹X`` at different ρ; the gap to
    the dense marginal must not depend on ``U``.
    """
    rng, spec, omega, _, z, D = _problem(1)
    k = 3
    mu_b = rng.normal(size=k)
    prec_b = 1.0 / rng.uniform(0.5, 2.0, size=k) ** 2
    proj = GroupProjector(spec, omega)
    gaps = []
    for _ in range(4):
        U = rng.normal(size=(z.size, k))
        cov = (
            np.diag(1.0 / omega)
            + U @ np.diag(1.0 / prec_b) @ U.T
            + spec.sigma**2 * D @ D.T
        )
        exact = multivariate_normal(U @ mu_b + spec.mu, cov).logpdf(z)

        r = z - spec.mu - U @ mu_b
        M = np.diag(prec_b) + proj.gram(U)
        v = proj.cross(U, r)
        L = np.linalg.cholesky(M)
        w = np.linalg.solve(L, v)
        ours = -np.sum(np.log(np.diag(L))) - 0.5 * (proj.quad(r) - w @ w)
        gaps.append(exact - ours)
    np.testing.assert_allclose(gaps, gaps[0], atol=1e-9)


def test_two_step_draw_has_joint_posterior_mean_and_cov():
    """β from the Schur Gram, then c | β, reproduces the joint (β, c) posterior."""
    rng, spec, omega, U, z, D = _problem(2)
    k = U.shape[1]
    prec_b = np.full(k, 0.25)
    Om = np.diag(omega)
    Z = np.hstack([U, D])
    prior_prec = np.concatenate([prec_b, np.full(D.shape[1], 1 / spec.sigma**2)])
    prior_mean = np.concatenate([np.zeros(k), np.full(D.shape[1], spec.mu)])
    P = Z.T @ Om @ Z + np.diag(prior_prec)
    joint_cov = np.linalg.inv(P)
    joint_mean = joint_cov @ (Z.T @ Om @ z + prior_prec * prior_mean)

    proj = GroupProjector(spec, omega)
    G = proj.gram(U) + np.diag(prec_b)
    h = proj.cross(U, z - spec.mu)
    beta_mean = np.linalg.solve(G, h)
    np.testing.assert_allclose(beta_mean, joint_mean[:k], atol=1e-10)
    np.testing.assert_allclose(np.linalg.inv(G), joint_cov[:k, :k], atol=1e-10)
    np.testing.assert_allclose(proj.mean(z - U @ beta_mean), joint_mean[k:], atol=1e-10)

    # Monte Carlo: marginal covariance of c through both steps.
    Lg = np.linalg.cholesky(np.linalg.inv(G))
    cs = np.array(
        [
            proj.draw(z - U @ (beta_mean + Lg @ rng.standard_normal(k)), rng)
            for _ in range(20000)
        ]
    )
    np.testing.assert_allclose(
        np.cov(cs.T), joint_cov[k:, k:], atol=0.05 * np.abs(joint_cov[k:, k:]).max()
    )


def test_spec_validation():
    with pytest.raises(ValueError, match="n_groups"):
        GroupEffects(np.array([0, 3]), 3, 0.0, 1.0)
    with pytest.raises(ValueError, match="sigma"):
        GroupEffects(np.array([0, 1]), 2, 0.0, 0.0)
    with pytest.raises(ValueError, match="integer"):
        GroupEffects(np.array([0.0, 1.0]), 2, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Structured flow sweep: the n × n moment code against materialized columns
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("pairs", [True, False])
@pytest.mark.parametrize("n_tau_full", [True, False])
def test_structured_gram_block_matches_projector(pairs, n_tau_full):
    """``_gram_block`` on ``n × n`` arrays equals the row-level collapsed moments."""
    from neighbayes.samplers._utils._group_effects import _weighted_moments
    from neighbayes.samplers.negbin_reduced._flow_structured_fe import _gram_block

    rng = np.random.default_rng(7)
    n, T, k = 4, 3, 2
    N = n * n
    Oms = rng.gamma(2.0, 0.5, size=(T, n, n))
    Rs = rng.normal(size=(T, n, n))
    cols = rng.normal(size=(T, k, n, n))
    prec = np.array([0.3, 0.7])
    n_tau = T if n_tau_full else T - 1
    t0 = T - n_tau
    prec_tau = np.full(n_tau, 0.2)
    s = 1.4
    D = Oms.sum(axis=0) + 1.0 / s**2 if pairs else None
    OmOm = None
    if pairs:
        flat = Oms[t0:].reshape(n_tau, -1)
        OmOm = (flat / D.ravel()) @ flat.T
    M, v, quad = _gram_block(
        list(Oms), list(Rs), lambda t: cols[t], prec, prec_tau, D, OmOm
    )

    # Materialize: rows time-first, columns [x-cols, period dummies].
    U = np.zeros((T * N, k + n_tau))
    for t in range(T):
        U[t * N : (t + 1) * N, :k] = cols[t].reshape(k, N).T
        if t >= t0:
            U[t * N : (t + 1) * N, k + t - t0] = 1.0
    proj = None
    if pairs:
        spec = GroupEffects(np.tile(np.arange(N), T), N, mu=0.0, sigma=s)
        proj = GroupProjector(spec, Oms.ravel())
    M_ref, v_ref, quad_ref = _weighted_moments(U, Rs.ravel(), Oms.ravel(), proj)
    M_ref[np.diag_indices_from(M_ref)] += np.concatenate([prec, prec_tau])
    np.testing.assert_allclose(M, M_ref, atol=1e-10)
    np.testing.assert_allclose(v, v_ref, atol=1e-10)
    np.testing.assert_allclose(quad, quad_ref, atol=1e-10)


@pytest.mark.requires_jax
@pytest.mark.parametrize("pairs", [True, False])
@pytest.mark.parametrize("n_tau_full", [True, False])
def test_structured_jax_moments_match_numpy(pairs, n_tau_full):
    """The JAX moment block against the NumPy ``_gram_block``."""
    from types import SimpleNamespace

    import jax.numpy as jnp
    import scipy.sparse as sp

    from neighbayes._jax_dispatch import ensure_x64
    from neighbayes.samplers.negbin_reduced._flow_structured import (
        classify_flow_design,
    )
    from neighbayes.samplers.negbin_reduced._flow_structured_fe import _gram_block
    from neighbayes.samplers.negbin_reduced._flow_structured_fe_jax import (
        make_structured_fe_sweep,
    )

    ensure_x64()
    rng = np.random.default_rng(3)
    n, T, k = 4, 3, 2
    X = rng.normal(size=(T * n * n, k))  # full-rank columns only
    struct = classify_flow_design(X, n, T)
    W = sp.random(n, n, density=0.5, random_state=0, format="csc")
    n_tau = T if n_tau_full else T - 1
    priors = SimpleNamespace(
        beta_mu=np.zeros(k),
        beta_sigma=np.ones(k),
        alpha_sigma=2.5,
        alpha_nu=3.0,
        rho_lower=-0.9,
        rho_upper=0.9,
    )
    s = 1.3
    sweep, _, _ = make_structured_fe_sweep(
        np.ones(T * n * n),
        W,
        struct,
        priors,
        pair_effects=(0.0, s) if pairs else None,
        period_effects=(0.0, 2.0, n_tau),
    )
    Oms = rng.gamma(2.0, 0.5, size=(T, n, n))
    Rs = rng.normal(size=(T, n, n))
    cols = rng.normal(size=(T, k, n, n))
    prec = np.array([0.4, 0.9])
    D = Oms.sum(axis=0) + 1.0 / s**2 if pairs else None
    OmOm = None
    if pairs:
        flat = Oms[T - n_tau :].reshape(n_tau, -1)
        OmOm = (flat / D.ravel()) @ flat.T
    ref = _gram_block(
        list(Oms), list(Rs), lambda t: cols[t], prec, np.full(n_tau, 0.25), D, OmOm
    )
    got = sweep.kernels["moments"](
        lambda t: jnp.asarray(cols)[t],
        lambda t: jnp.asarray(Rs)[t],
        k,
        jnp.asarray(prec),
        jnp.asarray(Oms),
        None if D is None else jnp.asarray(D),
        None if OmOm is None else jnp.asarray(OmOm),
    )
    for a, b in zip(got, ref):
        np.testing.assert_allclose(np.asarray(a), b, atol=1e-10)


def test_masked_cells_drop_out_of_structured_moments():
    """``Ω = 0`` cells (the ZINB count's structural zeros) equal a row subset."""
    from neighbayes.samplers._utils._group_effects import _weighted_moments
    from neighbayes.samplers.negbin_reduced._flow_structured_fe import _gram_block

    rng = np.random.default_rng(11)
    n, T, k = 4, 3, 2
    N = n * n
    Oms = rng.gamma(2.0, 0.5, size=(T, n, n))
    mask = rng.random((T, n, n)) < 0.6
    Oms = Oms * mask
    Rs = rng.normal(size=(T, n, n))
    cols = rng.normal(size=(T, k, n, n))
    prec, prec_tau, s = np.array([0.3, 0.7]), np.full(T - 1, 0.2), 1.4
    D = Oms.sum(axis=0) + 1.0 / s**2
    flat = Oms[1:].reshape(T - 1, -1)
    OmOm = (flat / D.ravel()) @ flat.T
    M, v, quad = _gram_block(
        list(Oms), list(Rs), lambda t: cols[t], prec, prec_tau, D, OmOm
    )

    keep = mask.ravel()
    U = np.zeros((T * N, k + T - 1))
    for t in range(T):
        U[t * N : (t + 1) * N, :k] = cols[t].reshape(k, N).T
        if t >= 1:
            U[t * N : (t + 1) * N, k + t - 1] = 1.0
    spec = GroupEffects(np.tile(np.arange(N), T)[keep], N, mu=0.0, sigma=s)
    proj = GroupProjector(spec, Oms.ravel()[keep])
    M_ref, v_ref, quad_ref = _weighted_moments(
        U[keep], Rs.ravel()[keep], Oms.ravel()[keep], proj
    )
    M_ref[np.diag_indices_from(M_ref)] += np.concatenate([prec, prec_tau])
    np.testing.assert_allclose(M, M_ref, atol=1e-10)
    np.testing.assert_allclose(v, v_ref, atol=1e-10)
    np.testing.assert_allclose(quad, quad_ref, atol=1e-10)
