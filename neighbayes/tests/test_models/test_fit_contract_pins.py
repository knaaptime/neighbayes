"""Phase 5b migration contract pins: posterior var-names per model × sampler.

These lock the *structural* ``fit()`` contract — the exact set of posterior
variables each model produces under each sampler, and that Gibbs attaches a
``log_likelihood`` group when asked (``idata_kwargs={"log_likelihood": True}``)
and none by default, as PyMC does — on the **current** code, so the registry / one-``fit()``
migration must reproduce it exactly.  They assert structure, not draws, so they
are insensitive to RNG-order changes (the roadmap's rule).

Families are added to the tables as each migration wave begins; the existing
recovery/model suites remain the correctness (parameter-recovery) pin.
"""

from __future__ import annotations

import numpy as np
import pytest

from neighbayes.models import (
    OLS,
    SAR,
    SARZINB,
    SDEM,
    SDM,
    SEM,
    SLX,
    SARLogit,
    SARLogitStructural,
    SARNegBin,
    SARNegBinStructural,
    SEMLogit,
)
from neighbayes.models.panel._fe import (
    OLSPanelFE,
    SARPanelFE,
    SDEMPanelFE,
    SDMPanelFE,
    SEMPanelFE,
    SLXPanelFE,
)
from neighbayes.models.panel._re import (
    OLSPanelRE,
    SARPanelRE,
    SDEMPanelRE,
    SEMPanelRE,
)
from neighbayes.tests.helpers import (
    PANEL_N,
    PANEL_T,
    W_to_graph,
    make_line_W,
    make_panel_ols_data,
    make_panel_sar_data,
    make_panel_sdem_fe_data,
    make_panel_sdm_fe_data,
    make_panel_sem_data,
    make_rook_W,
    make_sar_data,
    make_sdem_data,
    make_sdm_data,
    make_sem_data,
    make_slx_data,
)

_W_DENSE = make_rook_W(6)  # n = 36


def _graph():
    return W_to_graph(_W_DENSE)


#: Gibbs contract fits request the pointwise log-likelihood, so the opt-in is
#: pinned for every family; ``test_gibbs_log_likelihood_is_opt_in`` pins the
#: default.
_LL = {"log_likelihood": True}


def _fit_varnames(model, sampler):
    kw = {"idata_kwargs": _LL} if sampler == "gibbs" else {}
    idata = model.fit(
        sampler=sampler,
        draws=6,
        tune=6,
        chains=1,
        progressbar=False,
        random_seed=1,
        **kw,
    )
    return set(idata.posterior.data_vars), ("log_likelihood" in idata.groups())


# name -> (ctor, data_fn, expected_varnames, samplers, gibbs_has_ll)
GAUSSIAN_XS: dict[str, tuple] = {
    "SAR": (SAR, make_sar_data, {"beta", "rho", "sigma", "sigma2"}, ("gibbs", "nuts")),
    "SEM": (SEM, make_sem_data, {"beta", "lam", "sigma", "sigma2"}, ("gibbs", "nuts")),
    "SDM": (SDM, make_sdm_data, {"beta", "rho", "sigma", "sigma2"}, ("gibbs", "nuts")),
    "SDEM": (
        SDEM,
        make_sdem_data,
        {"beta", "lam", "sigma", "sigma2"},
        ("gibbs", "nuts"),
    ),
    "OLS": (OLS, make_sar_data, {"beta", "sigma", "sigma2"}, ("nuts",)),
    "SLX": (SLX, make_slx_data, {"beta", "sigma", "sigma2"}, ("nuts",)),
}


def _gaussian_cases():
    for name, (ctor, data_fn, expected, samplers) in GAUSSIAN_XS.items():
        for sampler in samplers:
            yield pytest.param(
                name, ctor, data_fn, expected, sampler, id=f"{name}-{sampler}"
            )


@pytest.mark.parametrize("name,ctor,data_fn,expected,sampler", list(_gaussian_cases()))
def test_gaussian_xs_fit_contract(name, ctor, data_fn, expected, sampler):
    rng = np.random.default_rng(0)
    y, X = data_fn(rng, _W_DENSE)
    model = ctor(y=y, X=X, W=_graph())
    varnames, has_ll = _fit_varnames(model, sampler)
    assert varnames == expected, f"{name} [{sampler}]: {sorted(varnames)}"
    if sampler == "gibbs":
        assert has_ll, f"{name} gibbs should attach a log_likelihood group on request"


# ---------------------------------------------------------------------------
# Gaussian panel fixed-effects family
# ---------------------------------------------------------------------------

_W_PANEL = make_line_W(PANEL_N)


def _panel_graph():
    return W_to_graph(_W_PANEL)


# name -> (ctor, data_fn, expected_varnames, samplers)
GAUSSIAN_PANEL_FE: dict[str, tuple] = {
    "SAR": (
        SARPanelFE,
        make_panel_sar_data,
        {"beta", "rho", "sigma", "sigma2"},
        ("gibbs", "nuts"),
    ),
    "SEM": (
        SEMPanelFE,
        make_panel_sem_data,
        {"beta", "lam", "sigma", "sigma2"},
        ("gibbs", "nuts"),
    ),
    "SDM": (
        SDMPanelFE,
        make_panel_sdm_fe_data,
        {"beta", "rho", "sigma", "sigma2"},
        ("gibbs", "nuts"),
    ),
    "SDEM": (
        SDEMPanelFE,
        make_panel_sdem_fe_data,
        {"beta", "lam", "sigma", "sigma2"},
        ("gibbs", "nuts"),
    ),
    "OLS": (OLSPanelFE, make_panel_ols_data, {"beta", "sigma", "sigma2"}, ("nuts",)),
    # SLX has no closed-form panel DGP helper; the SDM generator supplies WX
    # signal and SLX ignores the ρ lag.
    "SLX": (SLXPanelFE, make_panel_sdm_fe_data, {"beta", "sigma", "sigma2"}, ("nuts",)),
}


def _panel_fe_cases():
    for name, (ctor, data_fn, expected, samplers) in GAUSSIAN_PANEL_FE.items():
        for sampler in samplers:
            yield pytest.param(
                name, ctor, data_fn, expected, sampler, id=f"{name}-{sampler}"
            )


@pytest.mark.parametrize("name,ctor,data_fn,expected,sampler", list(_panel_fe_cases()))
def test_gaussian_panel_fe_fit_contract(name, ctor, data_fn, expected, sampler):
    y, X, _ = data_fn(np.random.default_rng(1), _W_PANEL, PANEL_N, PANEL_T)
    model = ctor(y=y, X=X, W=_panel_graph(), N=PANEL_N, T=PANEL_T, effects=1)
    kwargs = dict(
        sampler=sampler, draws=6, tune=6, chains=1, progressbar=False, random_seed=1
    )
    if sampler == "gibbs":
        kwargs["n_jobs"] = 1
        kwargs["idata_kwargs"] = _LL
    idata = model.fit(**kwargs)
    varnames = set(idata.posterior.data_vars)
    has_ll = "log_likelihood" in idata.groups()
    assert varnames == expected, f"{name} [{sampler}]: {sorted(varnames)}"
    if sampler == "gibbs":
        assert has_ll, f"{name} gibbs should attach a log_likelihood group on request"


# ---------------------------------------------------------------------------
# Gaussian panel random-effects family
#
# RE posteriors carry the unit effect ``alpha`` and its scale ``sigma_alpha``.
# The 5-block RE Gibbs sampler does not emit the deterministic ``sigma2`` that
# the NUTS build derives, so the contract is pinned *per sampler*.
# ---------------------------------------------------------------------------

_RE_BASE = {"alpha", "beta", "sigma", "sigma_alpha"}

# name -> (ctor, data_fn, {sampler: expected_varnames})
GAUSSIAN_PANEL_RE: dict[str, tuple] = {
    "OLS": (OLSPanelRE, make_panel_ols_data, {"nuts": _RE_BASE | {"sigma2"}}),
    "SAR": (
        SARPanelRE,
        make_panel_sar_data,
        {"gibbs": _RE_BASE | {"rho"}, "nuts": _RE_BASE | {"rho", "sigma2"}},
    ),
    "SEM": (
        SEMPanelRE,
        make_panel_sem_data,
        {"gibbs": _RE_BASE | {"lam"}, "nuts": _RE_BASE | {"lam", "sigma2"}},
    ),
    "SDEM": (
        SDEMPanelRE,
        make_panel_sdem_fe_data,
        {"nuts": _RE_BASE | {"lam", "sigma2"}},
    ),
}


def _panel_re_cases():
    for name, (ctor, data_fn, expected_by_sampler) in GAUSSIAN_PANEL_RE.items():
        for sampler, expected in expected_by_sampler.items():
            yield pytest.param(
                name, ctor, data_fn, expected, sampler, id=f"{name}-{sampler}"
            )


@pytest.mark.parametrize("name,ctor,data_fn,expected,sampler", list(_panel_re_cases()))
def test_gaussian_panel_re_fit_contract(name, ctor, data_fn, expected, sampler):
    y, X, _ = data_fn(np.random.default_rng(1), _W_PANEL, PANEL_N, PANEL_T)
    model = ctor(y=y, X=X, W=_panel_graph(), N=PANEL_N, T=PANEL_T)
    kwargs = dict(
        sampler=sampler, draws=6, tune=6, chains=1, progressbar=False, random_seed=1
    )
    if sampler == "gibbs":
        kwargs["n_jobs"] = 1
        kwargs["idata_kwargs"] = _LL
    idata = model.fit(**kwargs)
    varnames = set(idata.posterior.data_vars)
    has_ll = "log_likelihood" in idata.groups()
    assert varnames == expected, f"{name} [{sampler}]: {sorted(varnames)}"
    if sampler == "gibbs":
        assert has_ll, f"{name} gibbs should attach a log_likelihood group on request"


# ---------------------------------------------------------------------------
# Binary (Pólya-Gamma logit) cross-section family
#
# SARLogit/SEMLogit are Gibbs-only (no NUTS build).  The logit link fixes
# σ² = 1, so the posterior carries only the spatial parameter and β.
# ---------------------------------------------------------------------------

_N_BINARY = 24


def _binary_xy(seed: int = 3):
    rng = np.random.default_rng(seed)
    x1 = rng.standard_normal(_N_BINARY)
    X = np.column_stack([np.ones(_N_BINARY), x1])
    probs = 1.0 / (1.0 + np.exp(-(0.3 + 0.6 * x1)))
    y = rng.binomial(1, probs).astype(float)
    return y, X


def _binary_graph():
    return W_to_graph(make_line_W(_N_BINARY))


# name -> (ctor, expected_varnames)
# The canonical SARLogit is reduced-form (auto→jax); SARLogitStructural is the
# structural latent-field model; SEMLogit is the structural SEM.  All three
# expose only the spatial parameter and β.
BINARY_XS: dict[str, tuple] = {
    "SARLogit": (SARLogit, {"beta", "rho"}),
    "SARLogitStructural": (SARLogitStructural, {"beta", "rho"}),
    "SEMLogit": (SEMLogit, {"beta", "lam"}),
}


@pytest.mark.parametrize("name", list(BINARY_XS))
def test_binary_xs_fit_contract(name):
    ctor, expected = BINARY_XS[name]
    y, X = _binary_xy()
    model = ctor(y=y, X=X, W=_binary_graph())
    idata = model.fit(
        sampler="gibbs",
        draws=6,
        tune=6,
        chains=1,
        n_jobs=1,
        progressbar=False,
        random_seed=1,
        idata_kwargs=_LL,
    )
    varnames = set(idata.posterior.data_vars)
    assert varnames == expected, f"{name} [gibbs]: {sorted(varnames)}"
    assert "log_likelihood" in idata.groups(), (
        f"{name} gibbs should attach log_lik on request"
    )


def test_sarlogit_numpy_backend_fit_contract():
    """The reduced SARLogit's opt-in NumPy backend matches the JAX contract."""
    y, X = _binary_xy()
    model = SARLogit(y=y, X=X, W=_binary_graph())
    idata = model.fit(
        sampler="gibbs",
        gibbs_backend="numpy",
        draws=6,
        tune=6,
        chains=1,
        n_jobs=1,
        progressbar=False,
        random_seed=1,
        idata_kwargs=_LL,
    )
    assert set(idata.posterior.data_vars) == {"beta", "rho"}
    assert "log_likelihood" in idata.groups()


# ---------------------------------------------------------------------------
# Count (Pólya-Gamma NegBin / ZINB) cross-section family
#
# Gibbs-registered count models.  ``auto`` resolves to NumPy/CHOLMOD for all
# of them (the jax_dense path is opt-in via gibbs_backend="jax").  SARNegBin
# also supports NUTS (reduced-form PyMC model with a Deterministic ``mu``).
# ---------------------------------------------------------------------------

_N_COUNT = 40


def _count_xy(seed: int = 5, zero_inflate: bool = False):
    rng = np.random.default_rng(seed)
    x1 = rng.standard_normal(_N_COUNT)
    X = np.column_stack([np.ones(_N_COUNT), x1])
    y = rng.poisson(np.exp(0.3 + 0.2 * x1)).astype(float)
    if zero_inflate:
        y = y * (rng.uniform(size=_N_COUNT) > 0.3)
    return y, X


def _count_graph():
    return W_to_graph(make_line_W(_N_COUNT))


# name -> (ctor, zero_inflate, {sampler: expected_varnames})
COUNT_XS: dict[str, tuple] = {
    "SARNegBin": (
        SARNegBin,
        False,
        {"gibbs": {"beta", "rho", "alpha"}, "nuts": {"beta", "rho", "alpha", "mu"}},
    ),
    "SARNegBinStructural": (
        SARNegBinStructural,
        False,
        {"gibbs": {"beta", "rho", "alpha", "sigma"}},
    ),
    "SARZINB": (
        SARZINB,
        True,
        {"gibbs": {"beta", "rho", "alpha", "gamma", "lam"}},
    ),
}


def _count_cases():
    for name, (ctor, zi, expected_by_sampler) in COUNT_XS.items():
        for sampler, expected in expected_by_sampler.items():
            yield pytest.param(
                name, ctor, zi, expected, sampler, id=f"{name}-{sampler}"
            )


@pytest.mark.parametrize("name,ctor,zi,expected,sampler", list(_count_cases()))
@pytest.mark.filterwarnings("ignore:SAR Negative Binomial:UserWarning")
@pytest.mark.filterwarnings("ignore:Zero-inflated NB:UserWarning")
def test_count_xs_fit_contract(name, ctor, zi, expected, sampler):
    y, X = _count_xy(zero_inflate=zi)
    model = ctor(y=y, X=X, W=_count_graph())
    kwargs = dict(
        sampler=sampler, draws=6, tune=6, chains=1, progressbar=False, random_seed=1
    )
    if sampler == "gibbs":
        kwargs["n_jobs"] = 1
        kwargs["idata_kwargs"] = _LL
    idata = model.fit(**kwargs)
    varnames = set(idata.posterior.data_vars)
    assert varnames == expected, f"{name} [{sampler}]: {sorted(varnames)}"
    if sampler == "gibbs":
        assert "log_likelihood" in idata.groups(), (
            f"{name} gibbs should attach log_lik on request"
        )


@pytest.mark.requires_jax
@pytest.mark.filterwarnings("ignore:Zero-inflated NB:UserWarning")
def test_zinb_jax_backend_fit_contract():
    """SARZINB's opt-in JAX backend yields the same posterior contract.

    The JAX path is a *reduced-form* selection variant (vs the structural
    NumPy default), but exposes the identical parameter set and attaches a
    pointwise log-likelihood, so downstream tooling is backend-agnostic.
    """
    y, X = _count_xy(zero_inflate=True)
    model = SARZINB(y=y, X=X, W=_count_graph())
    idata = model.fit(
        sampler="gibbs",
        gibbs_backend="jax",
        draws=6,
        tune=6,
        chains=1,
        n_jobs=1,
        progressbar=False,
        random_seed=1,
        idata_kwargs=_LL,
    )
    varnames = set(idata.posterior.data_vars)
    assert varnames == {"beta", "rho", "alpha", "gamma", "lam"}, sorted(varnames)
    assert "log_likelihood" in idata.groups()


# ---------------------------------------------------------------------------
# Default: no pointwise log-likelihood unless requested
# ---------------------------------------------------------------------------


def _default_cases():
    rng = np.random.default_rng(0)
    y, X = make_sar_data(rng, _W_DENSE)
    yield "SAR", lambda: SAR(y=y, X=X, W=_graph()), {}
    yp, Xp, _ = make_panel_sar_data(
        np.random.default_rng(1), _W_PANEL, PANEL_N, PANEL_T
    )
    yield (
        "SARPanelFE",
        lambda: SARPanelFE(
            y=yp, X=Xp, W=_panel_graph(), N=PANEL_N, T=PANEL_T, effects=1
        ),
        {"n_jobs": 1},
    )
    yield (
        "SARPanelRE",
        lambda: SARPanelRE(y=yp, X=Xp, W=_panel_graph(), N=PANEL_N, T=PANEL_T),
        {"n_jobs": 1},
    )
    yb, Xb = _binary_xy()
    yield "SARLogit", lambda: SARLogit(y=yb, X=Xb, W=_binary_graph()), {"n_jobs": 1}
    yc, Xc = _count_xy(zero_inflate=False)
    yield "SARNegBin", lambda: SARNegBin(y=yc, X=Xc, W=_count_graph()), {"n_jobs": 1}


@pytest.mark.parametrize(
    "name,build,extra", [pytest.param(*c, id=c[0]) for c in _default_cases()]
)
@pytest.mark.filterwarnings("ignore:SAR Negative Binomial:UserWarning")
def test_gibbs_log_likelihood_is_opt_in(name, build, extra):
    idata = build().fit(
        sampler="gibbs",
        draws=6,
        tune=6,
        chains=1,
        progressbar=False,
        random_seed=1,
        **extra,
    )
    assert "log_likelihood" not in idata.groups(), name
