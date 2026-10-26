r"""Static panel posteriors against the toolbox's ``sar_panel_FE_g``.

Unit fixed effects, homoscedastic errors, one row-standardized ``W`` and one
panel exported verbatim.  The toolbox demeans and then counts the ``NT``
demeaned observations as independent: :math:`\sigma^2` gets ``NT`` degrees of
freedom and the Jacobian is :math:`T \log|I - \rho W|`, the direct approach,
which understates :math:`\sigma^2` by :math:`(T - 1)/T` (Lee & Yu 2010).
neighbayes applies Lee & Yu's correction: :math:`N(T - 1)` observations and
:math:`(T - 1)\log|I - \rho W|`.

So the comparison runs twice.  With the correction switched off the two
samplers target one posterior and must agree in every mean and sd.  With it on,
the default, both the Jacobian and the variance's count shrink by about
:math:`(T - 1)/T`, which tempers the marginal of :math:`\rho` without moving
its mode: :math:`\rho` and :math:`\beta` keep their location, and
:math:`\sigma` sits above the toolbox's by the square root of the ratio of the
two counts, :math:`(NT - k)/(N(T - 1) - k)` with :math:`\beta` integrated out.
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp

from neighbayes.models import SAR
from neighbayes.tests.test_validation import _mcse, _octave
from neighbayes.tests.test_validation import conftest as setup

pytestmark = pytest.mark.octave

TB_POST = setup.TB_DRAWS - setup.TB_OMIT

NB_TUNE = 1000
NB_DRAWS = setup.TB_DRAWS
NB_CHAINS = 4

SEED = 20261026


class _DirectSAR(SAR):
    """The SAR panel without the Lee & Yu dimensions: the toolbox's estimator."""

    _lee_yu = False


def _fit_neighbayes(model_cls, dataset):
    """The package's Gibbs on the exported panel, under the toolbox's priors.

    As for the cross-section: ``N(0, 1e12)`` on beta and the Jeffreys prior on
    sigma^2 as IG(1e-6, 1e-6); rho is uniform on (-1, 1) on both sides.
    """
    model = model_cls(
        unit_effects="fixed",
        y=dataset["y"],
        X=dataset["X"],
        W=sp.csr_matrix(dataset["W"]),
        N=dataset["N"],
        T=dataset["T"],
        priors={
            "beta_mu": 0.0,
            "beta_sigma": 1e12,
            "sigma2_alpha": 1e-6,
            "sigma2_beta": 1e-6,
        },
    )
    return model.fit(
        sampler="gibbs",
        tune=NB_TUNE,
        draws=NB_DRAWS,
        chains=NB_CHAINS,
        n_jobs=1,
        progressbar=False,
        random_seed=SEED,
    ).posterior


def test_sar_panel_fe_g_posterior_matches_neighbayes(
    panel_dataset, octave_run, toolbox, tmp_path
):
    text = _octave.panel_driver(
        toolbox,
        tmp_path,
        T=panel_dataset["T"],
        ndraw=setup.TB_DRAWS,
        nomit=setup.TB_OMIT,
        seed=13,
        prefix="tb",
    )
    stdout, draws = octave_run(text, dataset=panel_dataset, prefix="tb")
    assert "METH sarsfe_g" in stdout
    k = panel_dataset["X"].shape[1]
    assert draws["rho"].shape == (TB_POST,)
    assert draws["beta"].shape == (TB_POST, k)
    tb_sigma = np.sqrt(draws["sige"])  # the toolbox's sige is the variance

    # the direct approach: one posterior, every summary agrees
    nb = _fit_neighbayes(_DirectSAR, panel_dataset)
    for summary in (_mcse.assert_mean_matches, _mcse.assert_sd_matches):
        summary("rho", nb["rho"].values, draws["rho"])
        summary("sigma", nb["sigma"].values, tb_sigma)
        for j in range(k):
            summary(f"beta[{j}]", nb["beta"][:, :, j].values, draws["beta"][:, j])
    assert abs(float(nb["rho"].mean()) - setup.RHO_TRUE) < 0.15

    # the default, Lee & Yu: rho and beta keep their location, sigma moves
    # by the ratio of the observation counts
    ly = _fit_neighbayes(SAR, panel_dataset)
    _mcse.assert_mean_matches("rho (Lee & Yu)", ly["rho"].values, draws["rho"])
    for j in range(k):
        _mcse.assert_mean_matches(
            f"beta[{j}] (Lee & Yu)", ly["beta"][:, :, j].values, draws["beta"][:, j]
        )
    N, T = panel_dataset["N"], panel_dataset["T"]
    expected = np.sqrt((N * T - k) / (N * (T - 1) - k))
    ratio = float(ly["sigma"].mean()) / float(tb_sigma.mean())
    assert abs(ratio / expected - 1.0) < 0.005, (ratio, expected)
