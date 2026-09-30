"""Parameterized backend recovery tests for Gibbs-sampler models.

Every model that supports ``gibbs_backend`` ∈ {numpy, jax} is exercised
**on both backends** via ``pytest.parametrize``, ensuring the JAX and
NumPy paths produce statistically equivalent posteriors that recover the
true DGP parameters.  Grid sizes, true parameters, tolerances, and
sampling settings match the existing per-model recovery test files
exactly — these are **not** loose.

The bug this catches: the JAX Krylov path previously diverged near the
ρ boundary (|ρ| ≈ 1) because it used a fixed ``krylov_dmax`` that
exceeded the Neumann series' convergence radius, producing spurious
density barriers that trapped the sampler — yielding posteriors that
disagreed with the NumPy path.

Run with::

    pytest -m recovery -v                        # recovery tests only
    pytest test_gibbs_backend_recovery.py -v     # this file only
"""

from __future__ import annotations

import numpy as np
import pytest

from neighbayes import dgp
from neighbayes.tests.helpers import (
    W_to_graph,
    _as_sparse,
    make_rook_W,
    make_sar_logit_data,
    make_sar_logit_structural_data,
    make_sem_logit_data,
)

pytestmark = [pytest.mark.slow, pytest.mark.recovery]

BACKENDS = ["numpy", "jax"]


# ---------------------------------------------------------------------------
# Logit models: 30×30 grid (n=900), matching test_spatial_logit_recovery.py
# NB/ZINB models: 32×32 grid (n=1024), matching test_sar_negbin_latent.py
# ---------------------------------------------------------------------------
SIDE_LOGIT = 30
SIDE_NB = 32

# True parameters — logit (matching test_spatial_logit_recovery.py)
RHO_LOGIT = 0.4
LAM_LOGIT = 0.4
BETA_LOGIT = np.array([0.3, 1.0])

# True parameters — NB (matching test_sar_negbin_latent.py)
RHO_NB = 0.4
BETA_NB = np.array([0.5, 0.8])
ALPHA_NB = 2.0
SIGMA2_NB = 0.5

# True parameters — ZINB (matching test_sar_zinb_recovery.py)
RHO_ZINB = 0.4
LAM_ZINB = 0.3
BETA_ZINB = np.array([1.0, 0.6])
GAMMA_ZINB = np.array([0.3, 1.0])
ALPHA_ZINB = 2.0

# Sampling
DRAWS = 1000
TUNE = 1000
CHAINS = 2

# Tolerances — matching the per-model recovery tests exactly
TOL_SPATIAL_LOGIT = 0.15
TOL_BETA_LOGIT = 0.25
TOL_BETA_INTERCEPT_SEM = 0.50

TOL_SPATIAL_NB = 0.20
TOL_BETA_NB = 0.40
TOL_ALPHA_NB = 1.5

TOL_SPATIAL_ZINB = 0.20
TOL_LAM_ZINB = 0.25
TOL_BETA_ZINB = 0.40
TOL_GAMMA_ZINB = 0.40
TOL_ALPHA_ZINB = 1.5


def _fit_kwargs(backend):
    """Common fit kwargs, with n_jobs only for the NumPy backend."""
    kw = dict(
        draws=DRAWS,
        tune=TUNE,
        chains=CHAINS,
        random_seed=42,
        progressbar=False,
    )
    if backend == "numpy":
        kw["n_jobs"] = 1
    return kw


# ======================================================================
# SARLogit (reduced-form)
# ======================================================================


@pytest.fixture(scope="module")
def sar_logit_data():
    """Reduced-form SAR-logit data (noise-free DGP) on a 30×30 rook grid."""
    rng = np.random.default_rng(42)
    W_dense = make_rook_W(SIDE_LOGIT)
    y, X = make_sar_logit_data(rng, W_dense, rho=RHO_LOGIT, beta=BETA_LOGIT)
    return y, X, W_to_graph(W_dense)


@pytest.mark.parametrize("backend", BACKENDS)
def test_sar_logit_backend_recovery(sar_logit_data, backend):
    """SARLogit recovers ρ and β on both backends."""
    from neighbayes.models import SARLogit

    y, X, W = sar_logit_data
    idata = SARLogit(y=y, X=X, W=W).fit(gibbs_backend=backend, **_fit_kwargs(backend))

    rho_hat = float(idata.posterior["rho"].mean())
    beta_hat = idata.posterior["beta"].mean(("chain", "draw")).values

    assert abs(rho_hat - RHO_LOGIT) < TOL_SPATIAL_LOGIT, (
        f"SARLogit[{backend}] rho: {rho_hat:.3f} vs {RHO_LOGIT}"
    )
    for j, (bh, bt) in enumerate(zip(beta_hat, BETA_LOGIT)):
        assert abs(bh - bt) < TOL_BETA_LOGIT, (
            f"SARLogit[{backend}] beta[{j}]: {bh:.3f} vs {bt}"
        )


# ======================================================================
# SARLogitStructural (latent-field)
# ======================================================================


@pytest.fixture(scope="module")
def sar_logit_struct_data():
    """Structural-form SAR-logit data (with noise) on a 30×30 rook grid."""
    rng = np.random.default_rng(42)
    W_dense = make_rook_W(SIDE_LOGIT)
    y, X = make_sar_logit_structural_data(rng, W_dense, rho=RHO_LOGIT, beta=BETA_LOGIT)
    return y, X, W_to_graph(W_dense)


@pytest.mark.parametrize("backend", BACKENDS)
def test_sar_logit_struct_backend_recovery(sar_logit_struct_data, backend):
    """SARLogitStructural recovers ρ and β on both backends."""
    from neighbayes.models import SARLogitStructural

    y, X, W = sar_logit_struct_data
    idata = SARLogitStructural(y=y, X=X, W=W).fit(
        gibbs_backend=backend, **_fit_kwargs(backend)
    )

    rho_hat = float(idata.posterior["rho"].mean())
    beta_hat = idata.posterior["beta"].mean(("chain", "draw")).values

    assert abs(rho_hat - RHO_LOGIT) < TOL_SPATIAL_LOGIT, (
        f"SARLogitStructural[{backend}] rho: {rho_hat:.3f} vs {RHO_LOGIT}"
    )
    for j, (bh, bt) in enumerate(zip(beta_hat, BETA_LOGIT)):
        assert abs(bh - bt) < TOL_BETA_LOGIT, (
            f"SARLogitStructural[{backend}] beta[{j}]: {bh:.3f} vs {bt}"
        )


# ======================================================================
# SEMLogit (spatial error)
# ======================================================================


@pytest.fixture(scope="module")
def sem_logit_data():
    """SEM-logit data on a 30×30 rook grid."""
    rng = np.random.default_rng(42)
    W_dense = make_rook_W(SIDE_LOGIT)
    y, X = make_sem_logit_data(rng, W_dense, lam=LAM_LOGIT, beta=BETA_LOGIT)
    return y, X, W_to_graph(W_dense)


@pytest.mark.parametrize("backend", BACKENDS)
def test_sem_logit_backend_recovery(sem_logit_data, backend):
    """SEMLogit recovers λ and β on both backends."""
    from neighbayes.models import SEMLogit

    y, X, W = sem_logit_data
    idata = SEMLogit(y=y, X=X, W=W).fit(gibbs_backend=backend, **_fit_kwargs(backend))

    lam_hat = float(idata.posterior["lam"].mean())
    beta_hat = idata.posterior["beta"].mean(("chain", "draw")).values

    assert abs(lam_hat - LAM_LOGIT) < TOL_SPATIAL_LOGIT, (
        f"SEMLogit[{backend}] lam: {lam_hat:.3f} vs {LAM_LOGIT}"
    )
    for j, (bh, bt) in enumerate(zip(beta_hat, BETA_LOGIT)):
        tol = TOL_BETA_INTERCEPT_SEM if j == 0 else TOL_BETA_LOGIT
        assert abs(bh - bt) < tol, f"SEMLogit[{backend}] beta[{j}]: {bh:.3f} vs {bt}"


# ======================================================================
# SARNegBin (reduced-form, sigma2=0)
# ======================================================================


@pytest.fixture(scope="module")
def sar_negbin_data():
    """Reduced-form SAR-NB data (deterministic latent, sigma2=0) on a 32×32 rook grid."""
    rng = np.random.default_rng(42)
    W = W_to_graph(make_rook_W(SIDE_NB))
    return dgp.simulate_sar_negbin(
        W=_as_sparse(W),
        rho=RHO_NB,
        beta=BETA_NB,
        alpha=ALPHA_NB,
        sigma2=0.0,
        rng=rng,
    )


@pytest.mark.parametrize("backend", BACKENDS)
def test_sar_negbin_backend_recovery(sar_negbin_data, backend):
    """SARNegBin recovers ρ, β, and α on both backends."""
    from neighbayes.models import SARNegBin

    d = sar_negbin_data
    W = W_to_graph(make_rook_W(SIDE_NB))
    idata = SARNegBin(y=d["y"], X=d["X"], W=W).fit(
        gibbs_backend=backend, **_fit_kwargs(backend)
    )

    rho_hat = float(idata.posterior["rho"].mean())
    beta_hat = idata.posterior["beta"].mean(("chain", "draw")).values
    alpha_hat = float(idata.posterior["alpha"].mean())

    assert abs(rho_hat - RHO_NB) < TOL_SPATIAL_NB, (
        f"SARNegBin[{backend}] rho: {rho_hat:.3f} vs {RHO_NB}"
    )
    for j, (bh, bt) in enumerate(zip(beta_hat, BETA_NB)):
        assert abs(bh - bt) < TOL_BETA_NB, (
            f"SARNegBin[{backend}] beta[{j}]: {bh:.3f} vs {bt}"
        )
    assert abs(alpha_hat - ALPHA_NB) < TOL_ALPHA_NB, (
        f"SARNegBin[{backend}] alpha: {alpha_hat:.3f} vs {ALPHA_NB}"
    )


# ======================================================================
# SARNegBinStructural (latent-field, sigma2>0)
# ======================================================================


@pytest.fixture(scope="module")
def sar_negbin_struct_data():
    """Structural-form SAR-NB data (with latent noise, sigma2=0.5) on a 32×32 rook grid."""
    rng = np.random.default_rng(42)
    W = W_to_graph(make_rook_W(SIDE_NB))
    return dgp.simulate_sar_negbin(
        W=_as_sparse(W),
        rho=RHO_NB,
        beta=BETA_NB,
        alpha=ALPHA_NB,
        sigma2=SIGMA2_NB,
        rng=rng,
    )


@pytest.mark.parametrize("backend", BACKENDS)
def test_sar_negbin_struct_backend_recovery(sar_negbin_struct_data, backend):
    """SARNegBinStructural recovers ρ, β, α on both backends."""
    from neighbayes.models import SARNegBinStructural

    d = sar_negbin_struct_data
    W = W_to_graph(make_rook_W(SIDE_NB))
    idata = SARNegBinStructural(y=d["y"], X=d["X"], W=W).fit(
        gibbs_backend=backend, **_fit_kwargs(backend)
    )

    rho_hat = float(idata.posterior["rho"].mean())
    beta_hat = idata.posterior["beta"].mean(("chain", "draw")).values
    alpha_hat = float(idata.posterior["alpha"].mean())

    assert abs(rho_hat - RHO_NB) < TOL_SPATIAL_NB, (
        f"SARNegBinStructural[{backend}] rho: {rho_hat:.3f} vs {RHO_NB}"
    )
    for j, (bh, bt) in enumerate(zip(beta_hat, BETA_NB)):
        assert abs(bh - bt) < TOL_BETA_NB, (
            f"SARNegBinStructural[{backend}] beta[{j}]: {bh:.3f} vs {bt}"
        )
    assert abs(alpha_hat - ALPHA_NB) < TOL_ALPHA_NB, (
        f"SARNegBinStructural[{backend}] alpha: {alpha_hat:.3f} vs {ALPHA_NB}"
    )


# ======================================================================
# SARZINB (zero-inflated NB)
# ======================================================================


@pytest.fixture(scope="module")
def sar_zinb_data():
    """ZINB data on a 32×32 rook grid."""
    rng = np.random.default_rng(42)
    W = W_to_graph(make_rook_W(SIDE_NB))
    return dgp.simulate_sar_zinb(
        W=_as_sparse(W),
        rho=RHO_ZINB,
        lam=LAM_ZINB,
        beta=BETA_ZINB,
        gamma=GAMMA_ZINB,
        alpha=ALPHA_ZINB,
        rng=rng,
    )


@pytest.mark.parametrize("backend", BACKENDS)
def test_sar_zinb_backend_recovery(sar_zinb_data, backend):
    """SARZINB recovers ρ, λ, β, γ, α on both backends."""
    from neighbayes.models import SARZINB

    d = sar_zinb_data
    W = W_to_graph(make_rook_W(SIDE_NB))
    idata = SARZINB(y=d["y"], X=d["X"], Z=d["Z"], W=W).fit(
        gibbs_backend=backend, **_fit_kwargs(backend)
    )

    rho_hat = float(idata.posterior["rho"].mean())
    lam_hat = float(idata.posterior["lam"].mean())
    beta_hat = idata.posterior["beta"].mean(("chain", "draw")).values
    gamma_hat = idata.posterior["gamma"].mean(("chain", "draw")).values
    alpha_hat = float(idata.posterior["alpha"].mean())

    assert abs(rho_hat - RHO_ZINB) < TOL_SPATIAL_ZINB, (
        f"SARZINB[{backend}] rho: {rho_hat:.3f} vs {RHO_ZINB}"
    )
    assert abs(lam_hat - LAM_ZINB) < TOL_LAM_ZINB, (
        f"SARZINB[{backend}] lam: {lam_hat:.3f} vs {LAM_ZINB}"
    )
    for j, (bh, bt) in enumerate(zip(beta_hat, BETA_ZINB)):
        assert abs(bh - bt) < TOL_BETA_ZINB, (
            f"SARZINB[{backend}] beta[{j}]: {bh:.3f} vs {bt}"
        )
    for j, (gh, gt) in enumerate(zip(gamma_hat, GAMMA_ZINB)):
        assert abs(gh - gt) < TOL_GAMMA_ZINB, (
            f"SARZINB[{backend}] gamma[{j}]: {gh:.3f} vs {gt}"
        )
    assert abs(alpha_hat - ALPHA_ZINB) < TOL_ALPHA_ZINB, (
        f"SARZINB[{backend}] alpha: {alpha_hat:.3f} vs {ALPHA_ZINB}"
    )


# ======================================================================
# Cross-backend mixing quality (catches the original Krylov divergence bug)
# ======================================================================


@pytest.mark.parametrize("backend", BACKENDS)
def test_sar_logit_backend_mixing(sar_logit_data, backend):
    """SARLogit ρ has good ESS and R̂ on both backends (not stuck at boundary)."""
    import arviz as az

    from neighbayes.models import SARLogit

    y, X, W = sar_logit_data
    idata = SARLogit(y=y, X=X, W=W).fit(gibbs_backend=backend, **_fit_kwargs(backend))

    summ = az.summary(idata, var_names=["rho"])
    ess = float(summ.loc["rho", "ess_bulk"])
    rhat = float(summ.loc["rho", "r_hat"])

    # The JAX bug caused ESS ≈ 7 and R̂ ≈ 1.5 (stuck at ρ ≈ −0.9).
    assert ess > 100, f"SARLogit[{backend}] ESS={ess:.0f} too low (sampler stuck?)"
    assert rhat < 1.1, f"SARLogit[{backend}] R̂={rhat:.3f} too high (chains not mixed)"


# ======================================================================
# Cross-backend *agreement* — the two backends against each other
# ======================================================================
#
# Tolerances below are calibrated, not guessed.  Measured inter-backend
# difference in posterior means over these same fixtures (2 chains x 1000
# draws, n = 900/1024):
#
#     SARLogit             rho 0.0006   beta 0.0060
#     SARLogitStructural   rho 0.0007   beta 0.0098
#     SEMLogit             lam 0.0161   beta 0.0064
#     SARNegBin            rho 0.0020   beta 0.0029   alpha 0.0054
#     SARNegBinStructural  rho 0.0013   beta 0.0018   alpha 0.0038
#
# The tolerances sit ~3-30x above that noise and one to two orders of
# magnitude below the bugs this class of test exists to catch.
#
# The recovery tests above check each backend against truth with tolerances
# wide enough to absorb NB dispersion attenuation (TOL_ALPHA_ZINB = 1.5).
# That is the right tolerance for a recovery check and the wrong one for
# catching a backend that samples a subtly different target: two ZINB bugs
# lived under it, both found 2026-09-03.
#
#   1. The NumPy count sweep drew β before ρ and then evaluated the α block
#      at an η^cnt from before *both* moved.  The ρ step marginalizes β out,
#      so it leaves the joint invariant only when β is redrawn immediately
#      after (van Dyk & Park 2008); the stale η compounded it.  α came out
#      at 1.49 against JAX's 1.75 on the same data (n=1024).
#   2. The JAX ZINB step called both slice densities without ``solve_at``,
#      so every candidate outside the Krylov radius evaluated to −inf and
#      λ was confined to a window around its current value each sweep.  At
#      n=324, λ = 0.65 against NumPy's 0.44, with ESS 155.
#
# Both are invisible to a backend-vs-truth check and obvious in a
# backend-vs-backend one.  n=324 is deliberate: it is where both bugs bit
# hardest, and small enough to fit two full fits in the suite.


@pytest.fixture(scope="module")
def sar_zinb_small_data():
    """ZINB data on an 18×18 queen grid (n=324).

    The seed and geometry are not arbitrary: on this draw the λ posterior
    sits near 0.44, far enough from its starting value that the slice must
    step outside the Krylov radius to reach it — which is what made bug 2
    visible.  On a rook grid at seed 42 the same posterior lands at λ ≈
    −0.22, the truncation never binds, and both bugs hide.  Any change to
    this fixture should be re-checked against
    ``test_sar_zinb_backends_agree`` with the fixes reverted.
    """
    return dgp.simulate_sar_zinb(
        n_side=18,
        rho=RHO_ZINB,
        lam=LAM_ZINB,
        beta=BETA_ZINB,
        gamma=GAMMA_ZINB,
        alpha=ALPHA_ZINB,
        seed=3,
    )


def test_sar_zinb_backends_agree(sar_zinb_small_data):
    """Both ZINB backends must sample the *same* posterior, not merely a
    posterior near the truth.

    Tolerances are set well inside the two historical bugs (Δλ = 0.22,
    Δα = 0.60, Δγ₀ = 0.11) and well outside the Monte-Carlo noise between
    two 2×1000-draw fits (observed ≈ 0.02 on every parameter).
    """
    from neighbayes.models import SARZINB

    d = sar_zinb_small_data
    W = d["W_graph"]

    fits = {}
    for backend in BACKENDS:
        idata = SARZINB(y=d["y"], X=d["X"], Z=d["Z"], W=W).fit(
            gibbs_backend=backend, **_fit_kwargs(backend)
        )
        fits[backend] = idata.posterior

    def _mean(post, name):
        v = post[name]
        return np.atleast_1d(v.mean(("chain", "draw")).values)

    for name, tol in (
        ("rho", 0.08),
        ("lam", 0.08),
        ("alpha", 0.25),
        ("beta", 0.10),
        ("gamma", 0.08),
    ):
        a, b = _mean(fits["numpy"], name), _mean(fits["jax"], name)
        assert np.allclose(a, b, atol=tol), (
            f"ZINB backends disagree on {name}: numpy={np.round(a, 3)} "
            f"jax={np.round(b, 3)} (atol={tol})"
        )


def _assert_backends_agree(posteriors, tols, label):
    """Assert two backends' posterior means match, parameter by parameter."""
    a_post, b_post = posteriors["numpy"], posteriors["jax"]
    for name, tol in tols.items():
        a = np.atleast_1d(a_post[name].mean(("chain", "draw")).values)
        b = np.atleast_1d(b_post[name].mean(("chain", "draw")).values)
        assert np.allclose(a, b, atol=tol), (
            f"{label} backends disagree on {name}: "
            f"numpy={np.round(a, 4)} jax={np.round(b, 4)} (atol={tol})"
        )


# (model name, fixture, builder, tolerances)
_AGREEMENT_CASES = [
    ("SARLogit", "sar_logit_data", {"rho": 0.05, "beta": 0.05}),
    ("SARLogitStructural", "sar_logit_struct_data", {"rho": 0.05, "beta": 0.05}),
    ("SEMLogit", "sem_logit_data", {"lam": 0.06, "beta": 0.05}),
    ("SARNegBin", "sar_negbin_data", {"rho": 0.05, "beta": 0.05, "alpha": 0.15}),
    (
        "SARNegBinStructural",
        "sar_negbin_struct_data",
        {"rho": 0.05, "beta": 0.05, "alpha": 0.15},
    ),
]


@pytest.mark.parametrize("model_name,fixture_name,tols", _AGREEMENT_CASES)
def test_backends_agree(request, model_name, fixture_name, tols):
    """Every dual-backend family must sample the *same* posterior.

    The recovery tests above compare each backend to truth, at tolerances
    wide enough to absorb real statistical attenuation.  Only a
    backend-vs-backend comparison can be tight enough to catch a backend
    that has drifted onto a different target.
    """
    import neighbayes.models as models

    data = request.getfixturevalue(fixture_name)
    cls = getattr(models, model_name)

    if isinstance(data, tuple):  # logit fixtures: (y, X, W)
        y, X, W = data
        kwargs = dict(y=y, X=X, W=W)
    else:  # NB fixtures: dgp dict on the NB grid
        W = W_to_graph(make_rook_W(SIDE_NB))
        kwargs = dict(y=data["y"], X=data["X"], W=W)

    posteriors = {
        b: cls(**kwargs).fit(gibbs_backend=b, **_fit_kwargs(b)).posterior
        for b in BACKENDS
    }
    _assert_backends_agree(posteriors, tols, model_name)


# ----------------------------------------------------------------------
# Separable NB flow
# ----------------------------------------------------------------------
#
# This one is not decoration.  Until 2026-09-03 the JAX flow backend was
# simply wrong: ``_kron_solve_jax`` returned ``vec(Hᵀ)`` and both Kronecker
# matvecs applied a stray transpose, so the sampler fit a permuted system
# and reported rho_d = -0.21 against a true +0.35 with R-hat = 1.00 —
# converged, and wrong.  The operators are pinned directly in
# ``test_kron_krylov_matvec.py``; this test is the end-to-end backstop, and
# the only recovery coverage the flow JAX path has.


@pytest.fixture(scope="module")
def nb_flow_sep_data():
    """Separable NB flow counts on a 20x20 origin-destination grid (N=400)."""
    from neighbayes.dgp.flows import generate_negbin_flow_data_separable

    return generate_negbin_flow_data_separable(
        n=20, rho_d=0.35, rho_o=0.25, alpha=2.0, seed=7
    )


def test_nb_flow_separable_backends_agree(nb_flow_sep_data):
    """Both separable-flow backends must sample the same posterior."""
    from neighbayes.models.flow import SARNegBinFlowSeparable

    d = nb_flow_sep_data
    posteriors = {}
    for backend in BACKENDS:
        model = SARNegBinFlowSeparable(
            d["y_vec"], d["X"], d["G"], col_names=d["col_names"]
        )
        posteriors[backend] = model.fit(
            draws=DRAWS,
            tune=TUNE // 2,
            chains=CHAINS,
            random_seed=11,
            gibbs_backend=backend,
            progressbar=False,
        ).posterior

    _assert_backends_agree(
        posteriors,
        {"rho_d": 0.08, "rho_o": 0.08, "alpha": 0.30},
        "SARNegBinFlowSeparable",
    )


def test_nb_flow_separable_jax_recovers_and_mixes(nb_flow_sep_data):
    """The JAX flow backend must recover rho and actually mix.

    The pre-fix sampler produced ESS in the single digits with sign-flipped
    rho; both halves of that failure are pinned here.
    """
    import arviz as az

    from neighbayes.models.flow import SARNegBinFlowSeparable

    d = nb_flow_sep_data
    idata = SARNegBinFlowSeparable(
        d["y_vec"], d["X"], d["G"], col_names=d["col_names"]
    ).fit(
        draws=DRAWS,
        tune=TUNE // 2,
        chains=CHAINS,
        random_seed=11,
        gibbs_backend="jax",
        progressbar=False,
    )

    for name, truth in (("rho_d", 0.35), ("rho_o", 0.25)):
        hat = float(idata.posterior[name].mean())
        assert abs(hat - truth) < 0.20, f"flow[jax] {name}: {hat:.3f} vs {truth}"
        ess = float(az.ess(idata, var_names=[name])[name])
        assert ess > 100, f"flow[jax] {name} ESS={ess:.0f} too low (sampler stuck?)"
