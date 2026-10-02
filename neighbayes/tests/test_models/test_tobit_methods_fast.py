"""Fast tests for cross-sectional Tobit model fitted/effects methods."""

from __future__ import annotations

import arviz as az
import numpy as np
import xarray as xr

from neighbayes.models import SARTobit, SDMTobit, SEMTobit
from neighbayes.tests.helpers import W_to_graph, make_line_W


def _idata(vars_dict: dict[str, np.ndarray]) -> xr.DataTree:
    payload = {k: np.asarray(v)[None, ...] for k, v in vars_dict.items()}
    return az.from_dict({"posterior": payload})


def _cs_data(seed: int = 40):
    rng = np.random.default_rng(seed)
    n = 8
    x1 = rng.normal(size=n)
    X = np.column_stack([np.ones(n), x1])
    y = 0.2 + 1.0 * x1 + rng.normal(scale=0.3, size=n)
    y[[1, 4]] = 0.0  # censored observations
    W = W_to_graph(make_line_W(n))
    return y, X, W


def test_sar_sem_tobit_fitted_values_and_effects_run_with_mock_posteriors():
    y, X, W = _cs_data()

    sar = SARTobit(y=y, X=X, W=W)
    sem = SEMTobit(y=y, X=X, W=W)

    beta = np.array([0.3, 0.9])
    yc_sar = np.vstack(
        [
            np.linspace(0.05, 0.15, sar._censored_idx.size),
            np.linspace(0.06, 0.16, sar._censored_idx.size),
        ]
    )
    yc_sem = np.vstack(
        [
            np.linspace(0.05, 0.15, sem._censored_idx.size),
            np.linspace(0.06, 0.16, sem._censored_idx.size),
        ]
    )

    sar._idata = _idata(
        {
            "beta": np.stack([beta, beta + 1e-3]),
            "rho": np.array([0.2, 0.201]),
            "y_cens_gap": yc_sar,
        }
    )
    sem._idata = _idata(
        {
            "beta": np.stack([beta, beta + 1e-3]),
            "lam": np.array([0.1, 0.101]),
            "y_cens_gap": yc_sem,
        }
    )

    for m in [sar, sem]:
        fitted = m.fitted_values()
        effects = m.spatial_effects()
        assert fitted.shape == y.shape
        assert np.all(np.isfinite(fitted))
        assert np.all(np.isfinite(effects["direct"].values))

    sem_eff = sem.spatial_effects()
    assert np.allclose(sem_eff["indirect"].values, 0.0)


def test_sdm_tobit_fitted_values_and_effects_run_with_mock_posterior():
    y, X, W = _cs_data(seed=41)
    sdm = SDMTobit(y=y, X=X, W=W)

    # k=2, kw=1 when intercept is excluded from WX terms
    beta = np.array([0.25, 0.8, 0.15])
    yc = np.vstack(
        [
            np.linspace(0.05, 0.15, sdm._censored_idx.size),
            np.linspace(0.06, 0.16, sdm._censored_idx.size),
        ]
    )

    sdm._idata = _idata(
        {
            "beta": np.stack([beta, beta + 1e-3]),
            "rho": np.array([0.18, 0.181]),
            "y_cens_gap": yc,
        }
    )

    fitted = sdm.fitted_values()
    effects = sdm.spatial_effects()

    assert fitted.shape == y.shape
    assert np.all(np.isfinite(fitted))
    assert np.all(np.isfinite(effects["direct"].values))
    # SDM reports effects for all covariates (including intercept)
    assert len(effects.index) >= 1


# ---------------------------------------------------------------------------
# Regression tests for the H1/H2/H3 fixes
# ---------------------------------------------------------------------------


def test_tobit_censoring_mask_uses_exact_threshold():
    """H3: censoring mask must not include a ``+ 1e-12`` slack."""
    y, X, W = _cs_data()
    # Inject a value just above the censoring threshold; it must NOT be
    # classified as censored.
    y_above = y.copy()
    y_above[2] = 1e-12  # strictly > censoring=0
    sar = SARTobit(y=y_above, X=X, W=W, censoring=0.0)
    assert not sar._censored_mask[2]
    # Exact equality at threshold is censored.
    y_eq = y.copy()
    y_eq[3] = 0.0
    sar2 = SARTobit(y=y_eq, X=X, W=W, censoring=0.0)
    assert sar2._censored_mask[3]


def test_tobit_fitted_mean_respects_censoring_floor():
    """H1: ``fitted_values()[censored]`` must equal the censoring point."""
    y, X, W = _cs_data()
    sar = SARTobit(y=y, X=X, W=W, censoring=0.0)
    sdm = SDMTobit(y=y, X=X, W=W, censoring=0.0)
    sem = SEMTobit(y=y, X=X, W=W, censoring=0.0)

    beta_sar = np.array([0.3, 0.9])
    beta_sdm = np.array([0.25, 0.8, 0.15])
    yc_sar = np.vstack(
        [
            np.linspace(0.05, 0.15, sar._censored_idx.size),
            np.linspace(0.06, 0.16, sar._censored_idx.size),
        ]
    )
    yc_sdm = np.vstack(
        [
            np.linspace(0.05, 0.15, sdm._censored_idx.size),
            np.linspace(0.06, 0.16, sdm._censored_idx.size),
        ]
    )
    yc_sem = np.vstack(
        [
            np.linspace(0.05, 0.15, sem._censored_idx.size),
            np.linspace(0.06, 0.16, sem._censored_idx.size),
        ]
    )

    sar._idata = _idata(
        {
            "beta": np.stack([beta_sar, beta_sar + 1e-3]),
            "rho": np.array([0.2, 0.201]),
            "y_cens_gap": yc_sar,
        }
    )
    sdm._idata = _idata(
        {
            "beta": np.stack([beta_sdm, beta_sdm + 1e-3]),
            "rho": np.array([0.18, 0.181]),
            "y_cens_gap": yc_sdm,
        }
    )
    sem._idata = _idata(
        {
            "beta": np.stack([beta_sar, beta_sar + 1e-3]),
            "lam": np.array([0.1, 0.101]),
            "y_cens_gap": yc_sem,
        }
    )

    for m in (sar, sdm, sem):
        fitted = m.fitted_values()
        cens = m._censored_idx
        # H1: censored fitted values are at or above the censoring point.
        assert np.all(fitted[cens] >= m.censoring - 1e-12)


def test_tobit_fitted_mean_does_not_depend_on_y_cens_gap():
    """H2: the structural fitted mean must be independent of the latent
    censored-gap posterior — changing ``y_cens_gap`` cannot move the
    reported fitted mean (it is a function of (rho, beta) only).
    """
    y, X, W = _cs_data()
    sar = SARTobit(y=y, X=X, W=W, censoring=0.0)

    beta = np.array([0.3, 0.9])
    rho = np.array([0.2, 0.201])
    yc_a = np.vstack(
        [
            np.full(sar._censored_idx.size, 0.05),
            np.full(sar._censored_idx.size, 0.06),
        ]
    )
    yc_b = yc_a + 100.0  # wildly different gap posterior

    sar._idata = _idata(
        {
            "beta": np.stack([beta, beta + 1e-3]),
            "rho": rho,
            "y_cens_gap": yc_a,
        }
    )
    fitted_a = sar.fitted_values().copy()
    sar._idata = _idata(
        {
            "beta": np.stack([beta, beta + 1e-3]),
            "rho": rho,
            "y_cens_gap": yc_b,
        }
    )
    fitted_b = sar.fitted_values().copy()
    np.testing.assert_allclose(fitted_a, fitted_b)
