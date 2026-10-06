"""Correctness of the resolvent-gradient flow sampler's ρ-conditional target.

The novel coupling is the ρ log-posterior and its gradient (exact O(N) data term +
resolvent log-determinant).  We validate the gradient against a finite difference of
the target value using the *exact* log-determinant — deterministic and fast, with no
GMRES or MCMC loop.  (End-to-end posterior recovery is a GPU benchmark.)
"""

from __future__ import annotations

import numpy as np

from neighbayes._logdet._flow_resolvent import flow_logdet_grad_exact
from neighbayes.samplers.gaussian._flow_resolvent import (
    FlowResolventTarget,
    SEMFlowResolventTarget,
)


def _directed_sem_data(n, seed):
    """SEM flow: y = Xβ + u, A(λ) u = ε."""
    rng = np.random.default_rng(seed)
    A = (rng.uniform(size=(n, n)) < 0.2).astype(float)
    np.fill_diagonal(A, 0.0)
    A[A.sum(1) == 0, 0] = 1.0
    W = A / A.sum(1, keepdims=True)
    N = n * n
    Ide = np.eye(n)
    WF = 0.3 * np.kron(Ide, W) + 0.2 * np.kron(W, Ide) - 0.05 * np.kron(W, W)
    X = np.column_stack([np.ones(N), rng.standard_normal(N)])
    beta = np.array([1.0, -0.5])
    u = np.linalg.solve(np.eye(N) - WF, 0.4 * rng.standard_normal(N))
    y = X @ beta + u
    return W, y, X


def _directed_flow_data(n, seed):
    rng = np.random.default_rng(seed)
    A = (rng.uniform(size=(n, n)) < 0.2).astype(float)
    np.fill_diagonal(A, 0.0)
    A[A.sum(1) == 0, 0] = 1.0
    W = A / A.sum(1, keepdims=True)
    N = n * n
    Ide = np.eye(n)
    WF = 0.3 * np.kron(Ide, W) + 0.2 * np.kron(W, Ide) - 0.05 * np.kron(W, W)
    X = np.column_stack([np.ones(N), rng.standard_normal(N)])
    beta = np.array([1.0, -0.5])
    y = np.linalg.solve(np.eye(N) - WF, X @ beta + 0.4 * rng.standard_normal(N))
    return W, y, X


def _exact_value_and_grad(W):
    lam = np.linalg.eigvals(W)
    li, lj = lam[:, None], lam[None, :]

    def _vg(rd, ro, rw):
        mu = ro * li + rd * lj + rw * (li * lj)
        val = float(np.sum(np.log(np.abs(1.0 - mu))))
        return val, flow_logdet_grad_exact(W, rd, ro, rw)

    return _vg


def test_rho_gradient_matches_finite_difference():
    """logpost_and_grad's gradient equals a finite difference of its value (exact ld)."""
    W, y, X = _directed_flow_data(n=15, seed=0)
    tgt = FlowResolventTarget(W, y, X, logdet_value_and_grad=_exact_value_and_grad(W))

    beta = np.array([0.8, -0.4])
    sigma2 = 0.3
    h = 1e-6
    for rho in ([0.3, 0.2, -0.05], [0.5, 0.1, 0.1], [-0.2, 0.4, -0.1]):
        rho = np.array(rho)
        _, grad = tgt.logpost_and_grad(rho, beta, sigma2)
        fd = np.zeros(3)
        for k in range(3):
            rp, rm = rho.copy(), rho.copy()
            rp[k] += h
            rm[k] -= h
            vp, _ = tgt.logpost_and_grad(rp, beta, sigma2)
            vm, _ = tgt.logpost_and_grad(rm, beta, sigma2)
            fd[k] = (vp - vm) / (2 * h)
        np.testing.assert_allclose(grad, fd, rtol=1e-4, atol=1e-3)


def test_Ay_is_linear_in_rho():
    """A(ρ)y = y - Lρ uses the precomputed lags and is exactly linear in ρ."""
    W, y, X = _directed_flow_data(n=12, seed=1)
    tgt = FlowResolventTarget(W, y, X, logdet_value_and_grad=_exact_value_and_grad(W))
    r1 = tgt.Ay(np.array([0.3, 0.2, -0.05]))
    r2 = tgt.Ay(np.array([0.1, 0.0, 0.2]))
    mid = tgt.Ay(np.array([0.2, 0.1, 0.075]))
    np.testing.assert_allclose(mid, 0.5 * (r1 + r2), atol=1e-12)


def test_beta_sigma2_gibbs_runs():
    """The β | σ², σ² | β draw returns finite values of the right shape."""
    W, y, X = _directed_flow_data(n=12, seed=2)
    tgt = FlowResolventTarget(W, y, X, logdet_value_and_grad=_exact_value_and_grad(W))
    beta, sigma2 = tgt.draw_beta_sigma2(
        np.array([0.3, 0.2, -0.05]), np.random.default_rng(0), 1.0
    )
    assert beta.shape == (2,)
    assert np.isfinite(beta).all() and np.isfinite(sigma2) and sigma2 > 0


def test_sample_flow_resolvent_returns_inferencedata():
    """The model-facing entry point returns InferenceData and recovers dominant ρ."""
    from neighbayes.samplers.gaussian._flow_resolvent import sample_flow_resolvent

    W, y, X = _directed_flow_data(n=15, seed=7)
    idata = sample_flow_resolvent(
        W,
        y,
        X,
        draws=300,
        tune=300,
        chains=2,
        step_size=6e-4,
        coord_names=["const", "x1"],
        logdet_value_and_grad=_exact_value_and_grad(W),
        random_seed=1,
    )
    for v in ("rho_d", "rho_o", "rho_w", "sigma", "beta"):
        assert v in idata.posterior.data_vars
    assert idata.posterior["rho_d"].shape == (2, 300)
    assert idata.posterior["beta"].shape == (2, 300, 2)
    # Dominant components recover (true ρ_d=0.3, ρ_o=0.2).
    assert 0.15 < float(idata.posterior["rho_d"].mean()) < 0.45
    assert 0.05 < float(idata.posterior["rho_o"].mean()) < 0.35


def _directed_panel_sar_data(n, T, seed):
    """Panel SAR flow: for each period t, A y_t = X_t β + ε_t (stacked period-major)."""
    rng = np.random.default_rng(seed)
    A = (rng.uniform(size=(n, n)) < 0.2).astype(float)
    np.fill_diagonal(A, 0.0)
    A[A.sum(1) == 0, 0] = 1.0
    W = A / A.sum(1, keepdims=True)
    Nf = n * n
    Ide = np.eye(n)
    WF = 0.3 * np.kron(Ide, W) + 0.2 * np.kron(W, Ide) - 0.05 * np.kron(W, W)
    Ainv = np.linalg.inv(np.eye(Nf) - WF)
    beta = np.array([1.0, -0.5])
    ys, Xs = [], []
    for _ in range(T):
        Xt = np.column_stack([np.ones(Nf), rng.standard_normal(Nf)])
        yt = Ainv @ (Xt @ beta + 0.4 * rng.standard_normal(Nf))
        ys.append(yt)
        Xs.append(Xt)
    return W, np.concatenate(ys), np.vstack(Xs)


def test_panel_rho_gradient_matches_finite_difference():
    """Panel (T>1) ρ-gradient matches a finite difference; log-det scaled by T."""
    T = 3
    W, y, X = _directed_panel_sar_data(n=12, T=T, seed=0)
    tgt = FlowResolventTarget(
        W, y, X, T=T, logdet_value_and_grad=_exact_value_and_grad(W)
    )
    assert tgt.T == T and tgt.Ntot == y.shape[0]
    beta = np.array([0.8, -0.4])
    sigma2 = 0.3
    h = 1e-6
    for rho in ([0.3, 0.2, -0.05], [0.5, 0.1, 0.1]):
        rho = np.array(rho)
        _, grad = tgt.logpost_and_grad(rho, beta, sigma2)
        fd = np.zeros(3)
        for k in range(3):
            rp, rm = rho.copy(), rho.copy()
            rp[k] += h
            rm[k] -= h
            vp, _ = tgt.logpost_and_grad(rp, beta, sigma2)
            vm, _ = tgt.logpost_and_grad(rm, beta, sigma2)
            fd[k] = (vp - vm) / (2 * h)
        np.testing.assert_allclose(grad, fd, rtol=1e-4, atol=1e-3)


def test_panel_sample_recovers_dominant_rho():
    """A short panel run (T>1, exact ld) recovers the dominant ρ components."""
    from neighbayes.samplers.gaussian._flow_resolvent import sample_flow_resolvent

    T = 3
    W, y, X = _directed_panel_sar_data(n=12, T=T, seed=4)
    idata = sample_flow_resolvent(
        W,
        y,
        X,
        T=T,
        draws=300,
        tune=300,
        chains=2,
        step_size=5e-4,
        coord_names=["const", "x1"],
        logdet_value_and_grad=_exact_value_and_grad(W),
        random_seed=1,
    )
    assert idata.posterior["rho_d"].shape == (2, 300)
    assert 0.15 < float(idata.posterior["rho_d"].mean()) < 0.45
    assert 0.05 < float(idata.posterior["rho_o"].mean()) < 0.35


def test_sem_rho_gradient_matches_finite_difference():
    """SEM target's λ-gradient equals a finite difference of its value (exact ld)."""
    W, y, X = _directed_sem_data(n=15, seed=0)
    tgt = SEMFlowResolventTarget(
        W, y, X, logdet_value_and_grad=_exact_value_and_grad(W)
    )
    beta = np.array([0.8, -0.4])
    sigma2 = 0.3
    h = 1e-6
    for rho in ([0.3, 0.2, -0.05], [0.5, 0.1, 0.1], [-0.2, 0.4, -0.1]):
        rho = np.array(rho)
        _, grad = tgt.logpost_and_grad(rho, beta, sigma2)
        fd = np.zeros(3)
        for k in range(3):
            rp, rm = rho.copy(), rho.copy()
            rp[k] += h
            rm[k] -= h
            vp, _ = tgt.logpost_and_grad(rp, beta, sigma2)
            vm, _ = tgt.logpost_and_grad(rm, beta, sigma2)
            fd[k] = (vp - vm) / (2 * h)
        np.testing.assert_allclose(grad, fd, rtol=1e-4, atol=1e-3)


def test_sample_sem_flow_resolvent_returns_inferencedata():
    """The SEM entry point returns InferenceData (lam_*) and recovers dominant λ."""
    from neighbayes.samplers.gaussian._flow_resolvent import sample_sem_flow_resolvent

    W, y, X = _directed_sem_data(n=15, seed=7)
    idata = sample_sem_flow_resolvent(
        W,
        y,
        X,
        draws=300,
        tune=300,
        chains=2,
        step_size=6e-4,
        coord_names=["const", "x1"],
        logdet_value_and_grad=_exact_value_and_grad(W),
        random_seed=1,
    )
    for v in ("lam_d", "lam_o", "lam_w", "sigma", "beta"):
        assert v in idata.posterior.data_vars
    assert idata.posterior["lam_d"].shape == (2, 300)
    assert 0.1 < float(idata.posterior["lam_d"].mean()) < 0.5


def test_inferencedata_carries_jacobian_and_loglik():
    """The resolvent InferenceData attaches per-draw log|A| and a pointwise
    log_likelihood whose per-draw sum equals Gaussian density + T·log|A|."""
    import arviz as az

    from neighbayes.samplers.gaussian._flow_resolvent import (
        FlowResolventTarget,
        sample_flow_resolvent,
    )

    W, y, X = _directed_flow_data(n=15, seed=7)
    vg = _exact_value_and_grad(W)
    idata = sample_flow_resolvent(
        W,
        y,
        X,
        draws=200,
        tune=200,
        chains=2,
        step_size=6e-4,
        coord_names=["const", "x1"],
        logdet_value_and_grad=vg,
        random_seed=1,
    )
    assert "log_likelihood" in idata.children and "sample_stats" in idata.children
    assert idata.sample_stats["log_abs_det"].shape == (2, 200)
    assert idata.log_likelihood["obs"].shape == (2, 200, 15 * 15)
    # az.loo consumes the pointwise log-likelihood without error.
    az.loo(idata)

    # The Jacobian is genuinely inside the pointwise ll: Σ_obs ll == gaussian + log|A|.
    tgt = FlowResolventTarget(W, y, X, logdet_value_and_grad=vg)
    post = idata.posterior
    rho = np.array(
        [
            float(post["rho_d"][0, 0]),
            float(post["rho_o"][0, 0]),
            float(post["rho_w"][0, 0]),
        ]
    )
    beta = post["beta"][0, 0].values
    s2 = float(post["sigma"][0, 0]) ** 2
    r = tgt.residual(rho, beta)
    gauss = float(
        (-0.5 * np.log(2 * np.pi) - 0.5 * np.log(s2) - 0.5 * (r * r) / s2).sum()
    )
    ld_val, _ = vg(rho[0], rho[1], rho[2])
    ll_sum = float(idata.log_likelihood["obs"][0, 0].sum())
    np.testing.assert_allclose(ll_sum, gauss + ld_val, atol=1e-6)
    np.testing.assert_allclose(
        float(idata.sample_stats["log_abs_det"][0, 0]), ld_val, atol=1e-8
    )


def test_panel_jacobian_scales_with_T_in_loglik():
    """For the panel (T>1) the joint per-draw ll includes T·log|A|, and the stored
    log_abs_det equals T·log|A|."""
    from neighbayes.samplers.gaussian._flow_resolvent import (
        FlowResolventTarget,
        sample_flow_resolvent,
    )

    T = 3
    W, y, X = _directed_panel_sar_data(n=12, T=T, seed=4)
    vg = _exact_value_and_grad(W)
    idata = sample_flow_resolvent(
        W,
        y,
        X,
        T=T,
        draws=150,
        tune=150,
        chains=2,
        step_size=5e-4,
        coord_names=["const", "x1"],
        logdet_value_and_grad=vg,
        random_seed=1,
    )
    assert idata.log_likelihood["obs"].shape == (2, 150, T * 12 * 12)
    tgt = FlowResolventTarget(W, y, X, T=T, logdet_value_and_grad=vg)
    post = idata.posterior
    rho = np.array(
        [
            float(post["rho_d"][0, 0]),
            float(post["rho_o"][0, 0]),
            float(post["rho_w"][0, 0]),
        ]
    )
    beta = post["beta"][0, 0].values
    s2 = float(post["sigma"][0, 0]) ** 2
    r = tgt.residual(rho, beta)
    gauss = float(
        (-0.5 * np.log(2 * np.pi) - 0.5 * np.log(s2) - 0.5 * (r * r) / s2).sum()
    )
    ld_val, _ = vg(rho[0], rho[1], rho[2])
    ll_sum = float(idata.log_likelihood["obs"][0, 0].sum())
    np.testing.assert_allclose(ll_sum, gauss + T * ld_val, atol=1e-6)
    np.testing.assert_allclose(
        float(idata.sample_stats["log_abs_det"][0, 0]), T * ld_val, atol=1e-8
    )


def test_compute_log_likelihood_false_omits_group_keeps_jacobian():
    """``compute_log_likelihood=False`` skips the pointwise group but still records
    the cheap per-draw Jacobian log|A| in sample_stats."""
    from neighbayes.samplers.gaussian._flow_resolvent import sample_flow_resolvent

    W, y, X = _directed_flow_data(n=12, seed=1)
    idata = sample_flow_resolvent(
        W,
        y,
        X,
        draws=50,
        tune=50,
        chains=2,
        step_size=6e-4,
        logdet_value_and_grad=_exact_value_and_grad(W),
        random_seed=1,
        compute_log_likelihood=False,
    )
    assert "log_likelihood" not in idata.children
    assert "log_abs_det" in idata.sample_stats.data_vars


def test_attach_flow_log_abs_det_is_diagnostic_not_loglik():
    """The count-model helper writes per-draw log|A| to sample_stats and does NOT
    create/alter a log_likelihood group (the discrete count LOO must stay clean).
    At small n the exact eigenvalue path is used, so the value matches the reference
    exactly and scales exactly with T."""
    import arviz as az

    from neighbayes.samplers.gaussian._flow_resolvent import attach_flow_log_abs_det

    W, y, X = _directed_flow_data(n=14, seed=2)  # n << exact_max_n → exact path
    nchain, ndraw = 2, 4
    rd, ro, rw = 0.3, 0.2, -0.05
    idata = az.from_dict(
        {
            "posterior": {
                "rho_d": rd * np.ones((nchain, ndraw)),
                "rho_o": ro * np.ones((nchain, ndraw)),
                "rho_w": rw * np.ones((nchain, ndraw)),
                "beta": np.random.default_rng(0).standard_normal((nchain, ndraw, 2)),
            }
        },
        coords={"coefficient": ["const", "x1"]},
        dims={"beta": ["coefficient"]},
    )
    attach_flow_log_abs_det(idata, W, T=1)
    assert "sample_stats" in idata.children
    assert "log_likelihood" not in idata.children  # never pollutes the count LOO
    lad1 = idata.sample_stats["log_abs_det"].values
    assert lad1.shape == (nchain, ndraw)

    lam = np.linalg.eigvals(W)
    li, lj = lam[:, None], lam[None, :]
    exact = float(np.sum(np.log(np.abs(1.0 - (ro * li + rd * lj + rw * (li * lj))))))
    np.testing.assert_allclose(lad1, exact, atol=1e-9)  # exact path is exact

    # T only rescales the (same) log|A| — an exact ×3.
    attach_flow_log_abs_det(idata, W, T=3)
    np.testing.assert_allclose(
        idata.sample_stats["log_abs_det"].values, 3.0 * exact, atol=1e-9
    )


def test_attach_flow_log_abs_det_skips_aspatial_posterior():
    """A posterior without the spatial-filter parameters (aspatial NegBinFlow) is
    skipped gracefully rather than raising."""
    import arviz as az

    from neighbayes.samplers.gaussian._flow_resolvent import attach_flow_log_abs_det

    W, y, X = _directed_flow_data(n=10, seed=0)
    idata = az.from_dict(
        {"posterior": {"alpha": np.ones((2, 3)), "beta": np.zeros((2, 3, 2))}},
        coords={"coefficient": ["const", "x1"]},
        dims={"beta": ["coefficient"]},
    )
    attach_flow_log_abs_det(idata, W, T=1)  # must not raise
    assert "log_abs_det" not in (
        idata.sample_stats.data_vars if "sample_stats" in idata.children else {}
    )


def test_separable_count_flow_log_abs_det_uses_closed_form(monkeypatch):
    """Separable count flow classes take log|A| from n f(rho_d) + n f(rho_o),
    never from the resolvent route, which factors the N x N system (KLU ran out
    of memory at N = 9.68M).  The value equals the exact double sum with
    rho_w = -rho_d rho_o, and T rescales it."""
    import arviz as az

    import neighbayes.samplers.gaussian._flow_resolvent as fr
    from neighbayes.models.flow import SARNegBinFlowSeparable

    W, _, X = _directed_flow_data(n=12, seed=4)
    y = np.random.default_rng(4).poisson(3.0, size=X.shape[0])
    model = SARNegBinFlowSeparable(y, X, W, logdet_method="eigenvalue")

    def _no_resolvent(*args, **kwargs):
        raise AssertionError("separable class reached the resolvent route")

    monkeypatch.setattr(fr, "attach_flow_log_abs_det", _no_resolvent)

    nchain, ndraw = 2, 3
    rd = np.array([[0.3, 0.1, -0.2], [0.25, 0.0, 0.4]])
    ro = np.array([[0.2, -0.3, 0.15], [0.05, 0.35, -0.1]])
    idata = az.from_dict(
        {"posterior": {"rho_d": rd, "rho_o": ro, "rho_w": -rd * ro}}
    )
    model._attach_flow_log_abs_det(idata)
    lad = idata.sample_stats["log_abs_det"].values
    assert lad.shape == (nchain, ndraw)
    assert "log_likelihood" not in idata.children

    lam = np.linalg.eigvals(W)
    li, lj = lam[:, None], lam[None, :]
    exact = np.array(
        [
            np.sum(np.log(np.abs(1.0 - (b * li + a * lj - a * b * (li * lj)))))
            for a, b in zip(rd.ravel(), ro.ravel())
        ]
    ).reshape(nchain, ndraw)
    np.testing.assert_allclose(lad, exact, rtol=1e-8, atol=1e-8)

    model._T = 3
    model._attach_flow_log_abs_det(idata)
    np.testing.assert_allclose(
        idata.sample_stats["log_abs_det"].values, 3.0 * exact, rtol=1e-8, atol=1e-8
    )


def test_short_chain_with_exact_logdet_recovers_signs():
    """A short MALA-within-Gibbs run (exact ld, no GMRES) moves ρ toward the truth."""
    from neighbayes.samplers.gaussian._flow_resolvent import run_flow_resolvent_gibbs

    W, y, X = _directed_flow_data(n=15, seed=3)
    tgt = FlowResolventTarget(W, y, X, logdet_value_and_grad=_exact_value_and_grad(W))
    post = run_flow_resolvent_gibbs(tgt, draws=400, tune=400, step_size=6e-4, seed=1)
    # True ρ = (0.3, 0.2, -0.05): dominant components positive, small and finite.
    assert np.isfinite(post["rho_d"]).all()
    assert post["rho_d"].mean() > 0.0
    assert post["rho_o"].mean() > 0.0
    assert abs(post["rho_d"].mean()) < 0.9
