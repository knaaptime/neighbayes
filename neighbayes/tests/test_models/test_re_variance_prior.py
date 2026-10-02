"""The random-effect scale prior: half-t on both backends, non-centered NUTS.

``sigma_alpha`` gets a half-t prior (half-Cauchy by default) scaled to
``sd(y)``.  NUTS places it directly; the Gibbs sampler draws it through the
inverse-gamma mixture of Huang & Wand (2013).  These tests pin the mixture to
the half-t, and the two backends to one posterior.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from neighbayes.models import SARPanelRE
from neighbayes.models.priors import REGibbsPriors
from neighbayes.samplers.panel._re_core import _sample_sigma_alpha2
from neighbayes.tests.helpers import (
    PANEL_N,
    PANEL_T,
    W_to_graph,
    make_line_W,
    make_panel_sar_data,
)


@pytest.mark.parametrize("nu, scale", [(1.0, 2.0), (3.0, 0.5), (7.0, 10.0)])
def test_gibbs_mixture_targets_the_half_t_prior(nu, scale):
    """With no unit effects the σ_α² conditional is the prior itself.

    Alternating the two conjugate draws then samples σ_α ~ half-t_ν(0, A), so
    its quantiles must match the half-t's.
    """
    rng = np.random.default_rng(0)
    priors = REGibbsPriors(sigma_alpha_nu=nu, sigma_alpha_scale=scale)
    no_effects = np.zeros(0)
    s2, draws = scale**2, np.empty(200_000)
    for i in range(draws.size):
        s2, _ = _sample_sigma_alpha2(no_effects, s2, priors, rng)
        draws[i] = np.sqrt(s2)
    probs = np.array([0.25, 0.5, 0.75, 0.9])
    # The half-t is the |t| distribution: its p-quantile is t's (1+p)/2.
    want = scale * stats.t.ppf((1 + probs) / 2, df=nu)
    np.testing.assert_allclose(np.quantile(draws, probs), want, rtol=0.03)


def test_old_half_normal_key_is_rejected():
    """``sigma_alpha_sigma`` named a half-normal scale; it must not be reused."""
    W = make_line_W(PANEL_N)
    y, X, _ = make_panel_sar_data(
        np.random.default_rng(0), W, PANEL_N, PANEL_T, rho=0.3
    )
    with pytest.raises(TypeError, match="sigma_alpha_sigma"):
        SARPanelRE(
            y=y,
            X=X,
            W=W_to_graph(W),
            N=PANEL_N,
            T=PANEL_T,
            priors={"sigma_alpha_sigma": 10.0},
        )


@pytest.mark.slow
def test_nuts_and_gibbs_share_the_posterior():
    """Non-centered NUTS and conjugate Gibbs agree on the SAR-RE posterior."""
    import arviz as az

    N, T = 30, 6
    W = make_line_W(N)
    y, X, _ = make_panel_sar_data(
        np.random.default_rng(4), W, N, T, rho=0.4, sigma_alpha=0.5
    )
    kw = dict(draws=3000, tune=1500, chains=4, random_seed=7, progressbar=False)
    fits = {
        sampler: SARPanelRE(y=y, X=X, W=W_to_graph(W), N=N, T=T).fit(
            sampler=sampler, **kw
        )
        for sampler in ("nuts", "gibbs")
    }
    nuts, gibbs = fits["nuts"], fits["gibbs"]
    assert int(nuts.sample_stats["diverging"].sum()) == 0
    for name in ("sigma_alpha", "rho", "beta"):
        a = nuts.posterior[name].values.reshape(4 * 3000, -1)
        b = gibbs.posterior[name].values.reshape(a.shape)
        ess = min(
            float(np.min(np.asarray(az.ess(fit.posterior[name]))))
            for fit in (nuts, gibbs)
        )
        # Five combined Monte Carlo standard errors on the mean.
        se = np.sqrt(a.var(0) / ess + b.var(0) / ess)
        np.testing.assert_array_less(np.abs(a.mean(0) - b.mean(0)), 5 * se)
        np.testing.assert_allclose(a.std(0), b.std(0), rtol=0.1)
