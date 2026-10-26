"""Gaussian cross-section posteriors against the toolbox's ``sar_g``/``sem_g``.

Plan gate 2: the reference runs.  One dataset, one row-standardized ``W``,
built in Python and exported as matrix files that the ``.m`` side reads
verbatim, so construction-order and standardization drift cannot enter as a
disagreement.  Both sides carry diffuse priors with the same uniform support
on the spatial parameter, so every tolerance comes from the two chains'
Monte-Carlo error (:mod:`._mcse`, joined in quadrature), never from the
reference's print precision.

The Gaussian cross-section is the plan's first comparison target: two Gibbs
implementations with entirely different log-determinant machinery agreeing on
one posterior.

The toolbox stores each iteration's beta beside the *next* iteration's rho:
``sar_g`` draws beta given the current rho, then a new rho from its marginal
posterior, and saves the pair.  The marginals are right, but the stored pairs
are independent (the intercept-rho posterior correlation, about -0.9 here,
reads as 0), and so are the pairs the toolbox's own impacts are computed from.
Shifting rho back one iteration restores the joint, which the SAR test
compares.
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp

from neighbayes.models import SAR, SEM
from neighbayes.tests.test_validation import _mcse, _octave
from neighbayes.tests.test_validation import conftest as setup

pytestmark = pytest.mark.octave

# toolbox post-warmup draws (struct drops the omitted burn-in)
TB_POST = setup.TB_DRAWS - setup.TB_OMIT

# neighbayes chain layout (match the toolbox's post-warmup length per chain)
NB_TUNE = 1000
NB_DRAWS = setup.TB_DRAWS
NB_CHAINS = 4

SEED = 20261018


def _fit_neighbayes(model_cls, dataset):
    """The package's blocked Gibbs on the exported dataset.

    Priors are matched to the toolbox's defaults so both sides sample the
    same posterior and the comparison isolates the samplers: the toolbox's
    diffuse ``N(0, 1e12)`` on beta and, with ``prior.novi = 1``, an improper
    Jeffreys prior on sigma^2 (``1/sige ~ Gamma(0, 0)``, i.e. IG(0, 0)),
    approximated by an IG(1e-6, 1e-6) that differs from it only past
    sigma^2 = 1e6.  The rho priors already coincide: uniform on (-1, 1).
    """
    model = model_cls(
        y=dataset["y"],
        X=dataset["X"],
        W=sp.csr_matrix(dataset["W"]),
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
    )


def _assert_truth_sanity(dataset, idata, spatial_name, abs_tol):
    """Recovery sanity band on the exported dataset's own true parameters."""
    params = dataset["params_true"]
    for name, true in (
        (spatial_name, params[spatial_name]),
        *[(f"beta[{j}]", b) for j, b in enumerate(params["beta"])],
        ("sigma", params["sigma"]),
    ):
        if name.startswith("beta"):
            hat = float(idata.posterior["beta"][:, :, int(name[5:-1])].mean())
        elif name == "sigma":
            hat = float(idata.posterior["sigma"].mean())
        else:
            hat = float(idata.posterior[spatial_name].mean())
        assert abs(hat - true) < abs_tol, f"{name}: expected ≈{true}, got {hat:.3f}"


def _compare(dataset, idata, draws, spatial_name):
    """Run the full summary comparison between the two sides."""
    nb = idata.posterior
    # the toolbox's `sige` is the error VARIANCE (LeSage: e = N(0, sige*V),
    # 1/sige ~ Gamma); neighbayes's `sigma` is the std deviation
    tb_sigma = np.sqrt(draws["sige"])
    # posterior means, tolerance from each side's Monte-Carlo error
    _mcse.assert_mean_matches(spatial_name, nb[spatial_name].values, draws["rho"])
    _mcse.assert_mean_matches("sigma", nb["sigma"].values, tb_sigma)
    for j in range(nb["beta"].shape[-1]):
        _mcse.assert_mean_matches(
            f"beta[{j}]", nb["beta"][:, :, j].values, draws["beta"][:, j]
        )
    # posterior shapes agree
    _mcse.assert_sd_matches(spatial_name, nb[spatial_name].values, draws["rho"])
    _mcse.assert_sd_matches("sigma", nb["sigma"].values, tb_sigma)
    for j in range(nb["beta"].shape[-1]):
        _mcse.assert_sd_matches(
            f"beta[{j}]", nb["beta"][:, :, j].values, draws["beta"][:, j]
        )
    _assert_truth_sanity(dataset, idata, spatial_name, abs_tol=0.25)


def test_sar_g_posterior_matches_neighbayes(sar_dataset, octave_run, toolbox, tmp_path):
    idata = _fit_neighbayes(SAR, sar_dataset)
    text = _octave.gauss_driver(
        "sar_g",
        toolbox,
        tmp_path,
        ndraw=setup.TB_DRAWS,
        nomit=setup.TB_OMIT,
        seed=11,
        prefix="tb",
    )
    stdout, draws = octave_run(text, dataset=sar_dataset, prefix="tb")
    assert "METH sar_g" in stdout
    assert draws["rho"].shape == (TB_POST,)
    assert draws["beta"].shape == (TB_POST, sar_dataset["X"].shape[1])
    _compare(sar_dataset, idata, draws, "rho")

    # the joint: the intercept-rho correlation, with the toolbox's rho shifted
    # back to the iteration its beta was drawn under
    nb = idata.posterior
    tb_intercept, tb_rho = draws["beta"][:, 0], draws["rho"]
    _mcse.assert_corr_matches(
        "corr(beta[0], rho)",
        (nb["beta"][:, :, 0].values, nb["rho"].values),
        (tb_intercept[1:], tb_rho[:-1]),
    )
    # as stored, the toolbox's pairs carry no correlation at all
    r_stored = np.corrcoef(tb_intercept, tb_rho)[0, 1]
    assert abs(r_stored) < 4.0 / np.sqrt(_mcse.ess(tb_rho)), r_stored


def test_sem_g_posterior_matches_neighbayes(sem_dataset, octave_run, toolbox, tmp_path):
    idata = _fit_neighbayes(SEM, sem_dataset)
    text = _octave.gauss_driver(
        "sem_g",
        toolbox,
        tmp_path,
        ndraw=setup.TB_DRAWS,
        nomit=setup.TB_OMIT,
        seed=12,
        prefix="tb",
    )
    stdout, draws = octave_run(text, dataset=sem_dataset, prefix="tb")
    assert "METH sem_g" in stdout
    assert draws["rho"].shape == (TB_POST,)
    _compare(sem_dataset, idata, draws, "lam")
