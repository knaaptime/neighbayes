"""Interweaving of σ_α in the random-effects panel Gibbs sampler.

The centred σ_α | α update alone mixes at ~1% of draws when the unit effects
are small against σ/√T (the usual large-N, small-T spatial panel); the
non-centred σ_α | α̃ step fixes that.  These tests pin the posterior against
NUTS in that regime and guard the mixing.
"""

from __future__ import annotations

import numpy as np
import pytest

from neighbayes import dgp
from neighbayes.models import SARPanelRE, SEMPanelRE


def _weak_effects(sim, seed):
    return sim(N=400, T=3, n_side=20, sigma=1.0, sigma_alpha=0.3, seed=seed)


def test_sigma_alpha_mixes_with_weak_effects():
    import arviz as az

    d = _weak_effects(dgp.simulate_panel_sar_re, 3)
    m = SARPanelRE(y=d["y"], X=d["X"], W=d["W_sparse"], N=400, T=3)
    post = m.fit(
        sampler="gibbs", draws=1000, tune=500, chains=2, progressbar=False,
        random_seed=1,
    ).posterior  # fmt: skip
    x = post["sigma_alpha"].values
    # Centred-only sampling gave ~1% of draws here.
    assert float(az.ess(x)) > 0.05 * x.size


@pytest.mark.slow
@pytest.mark.parametrize(
    "cls, sim, spatial",
    [
        (SARPanelRE, dgp.simulate_panel_sar_re, "rho"),
        (SEMPanelRE, dgp.simulate_panel_sem_re, "lam"),
    ],
)
def test_re_gibbs_matches_nuts_with_weak_effects(cls, sim, spatial):
    d = _weak_effects(sim, 7)
    m = cls(y=d["y"], X=d["X"], W=d["W_sparse"], N=400, T=3)
    g = m.fit(
        sampler="gibbs", draws=4000, tune=1000, chains=4, progressbar=False,
        random_seed=1,
    ).posterior  # fmt: skip
    n = m.fit(
        sampler="nuts", draws=2000, tune=1500, chains=4, progressbar=False,
        random_seed=2, target_accept=0.95,
    ).posterior  # fmt: skip
    for var in (spatial, "sigma", "sigma_alpha", "beta"):
        a = g[var].values.reshape(-1, *g[var].shape[2:])
        b = n[var].values.reshape(-1, *n[var].shape[2:])
        assert np.all(np.abs(a.mean(0) - b.mean(0)) < 0.2 * b.std(0)), var
        np.testing.assert_allclose(a.std(0), b.std(0), rtol=0.15, err_msg=var)
