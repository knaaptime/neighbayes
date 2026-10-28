r"""SAR probit posteriors against the toolbox's ``sarp_g``.

The plan's highest-value target.  The toolbox draws the latent utilities one
at a time by truncated-normal Gibbs and :math:`\rho` on a grid; neighbayes
draws the utilities by colour class on JAX and :math:`\rho` by slice.  Same
posterior, code paths that share nothing.

``sarp_g`` as shipped does not sample that posterior.  On this dataset its
:math:`\beta` posterior is 11 to 13% wider than the model's and its means 5 to
7 Monte-Carlo errors away (40,000 draws), where neighbayes agrees with a plain
full-conditional Gibbs sampler written for the purpose to within one.  Three
defects, each repaired and measured: the sweep keeps the deviations
:math:`z = y^* - \mu` while :math:`\mu` changes, so every pass starts from
shifted utilities (the width); :math:`\beta` is drawn before :math:`\rho` is
redrawn with :math:`\beta` integrated out, and the utilities then see the
stale :math:`\beta`; and the :math:`\rho` conditional integrates out an error
variance the probit fixes at one (together, a further shift in :math:`\rho`
and in its correlation with the intercept).

So the comparison runs against ``sarp_g_repaired``: the served ``sarp_g.m``
with the three repaired and nothing else changed
(:func:`._octave.write_sarp_g_repaired`).  A second test keeps the shipped
sampler's deviation on record, which is what justifies the repair.
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp

from neighbayes.models import SARProbit
from neighbayes.tests.test_validation import _mcse, _octave

pytestmark = pytest.mark.octave

# The toolbox's single-site sweep mixes slowly in beta (ESS about 4% of the
# draws), so its chain is long.
TB_DRAWS = 22_000
TB_OMIT = 2_000
TB_POST = TB_DRAWS - TB_OMIT

NB_TUNE = 1500
NB_DRAWS = 5000
NB_CHAINS = 4

SEED = 20261026


@pytest.fixture(scope="module")
def neighbayes_posterior(probit_dataset):
    """The latent SAR probit, under the toolbox's ``N(0, 1e12)`` on beta.

    rho is uniform on (-1, 1) on the toolbox's side, (-0.999, 0.999) here.
    """
    model = SARProbit(
        y=probit_dataset["y"],
        X=probit_dataset["X"],
        W=sp.csr_matrix(probit_dataset["W"]),
        priors={"beta_mu": 0.0, "beta_sigma": 1e6},
    )
    return model.fit(
        tune=NB_TUNE,
        draws=NB_DRAWS,
        chains=NB_CHAINS,
        progressbar=False,
        random_seed=SEED,
    ).posterior


def _run(routine, probit_dataset, octave_run, toolbox, tmp_path, seed):
    text = _octave.probit_driver(
        routine,
        toolbox,
        tmp_path,
        ndraw=TB_DRAWS,
        nomit=TB_OMIT,
        seed=seed,
        prefix="tb",
    )
    stdout, draws = octave_run(
        text, dataset=probit_dataset, prefix="tb", keys=("rho", "beta")
    )
    assert "METH sarp_g" in stdout
    assert draws["rho"].shape == (TB_POST,)
    assert draws["beta"].shape == (TB_POST, probit_dataset["X"].shape[1])
    return draws


def test_repaired_sarp_g_posterior_matches_neighbayes(
    probit_dataset, neighbayes_posterior, octave_run, toolbox, tmp_path
):
    routine = _octave.write_sarp_g_repaired(toolbox, tmp_path)
    draws = _run(routine, probit_dataset, octave_run, toolbox, tmp_path, seed=14)
    nb = neighbayes_posterior
    for summary in (_mcse.assert_mean_matches, _mcse.assert_sd_matches):
        summary("rho", nb["rho"].values, draws["rho"])
        for j in range(draws["beta"].shape[1]):
            summary(f"beta[{j}]", nb["beta"][:, :, j].values, draws["beta"][:, j])
    # the repaired sweep stores beta with the rho it was drawn under
    _mcse.assert_corr_matches(
        "corr(beta[0], rho)",
        (nb["beta"][:, :, 0].values, nb["rho"].values),
        (draws["beta"][:, 0], draws["rho"]),
    )


def test_shipped_sarp_g_posterior_deviates(
    probit_dataset, neighbayes_posterior, octave_run, toolbox, tmp_path
):
    """Each beta's mean or sd, as shipped, lies over 3 Monte-Carlo errors out."""
    draws = _run("sarp_g", probit_dataset, octave_run, toolbox, tmp_path, seed=14)
    nb = neighbayes_posterior
    for j in range(draws["beta"].shape[1]):
        nb_j, tb_j = nb["beta"][:, :, j].values, draws["beta"][:, j]
        shift = abs(float(np.mean(tb_j) - np.mean(nb_j))) / float(
            np.hypot(_mcse.mcse_mean(nb_j), _mcse.mcse_mean(tb_j))
        )
        widening = float(np.std(tb_j) - np.std(nb_j)) / float(
            np.hypot(_mcse.mcse_sd(nb_j), _mcse.mcse_sd(tb_j))
        )
        assert max(shift, widening) > 3.0, (
            f"beta[{j}]: shipped sarp_g mean {np.mean(tb_j):.4f} sd "
            f"{np.std(tb_j):.4f} vs the model's {np.mean(nb_j):.4f} sd "
            f"{np.std(nb_j):.4f} (shift {shift:.1f}, widening {widening:.1f} MCSE)"
        )
