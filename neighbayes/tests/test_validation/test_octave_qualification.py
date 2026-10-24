"""Runner qualification for the LeSage toolbox under Octave (plan gate 1).

The toolbox ships its own documented worked example: ``data/anselin.txt``
gives the call pattern (Columbus crime rates on a constant, income and house
values) and ``data/wmat.txt`` the row-standardized 49-observation weights,
loaded as ``W = sparse(wmat(:,1), wmat(:,2), wmat(:,3))``.  This test runs
that example verbatim under Octave and checks the pieces the reference runs
depend on: the toolbox's maximum-likelihood ``sar`` and its Bayesian ``sar_g``
agree on the posterior location, and ``prt`` produces its documented printed
table ("Posterior Estimates").  Octave's random streams do not reproduce
MATLAB's, so the comparison happens at the level of posterior summaries with
Monte-Carlo error bands — never trace-level.

A runtime that cannot reproduce the toolbox's own published example cannot be
trusted to produce reference numbers.
"""

from __future__ import annotations

import re

import numpy as np
import pytest

from neighbayes.tests.test_validation import _mcse, _octave

pytestmark = pytest.mark.octave

QUAL_SEED = 3


def test_octave_reproduces_the_documented_columbus_example(
    octave_run, toolbox, tmp_path
):
    text = _octave.qualification_driver(
        toolbox, tmp_path, ndraw=3000, nomit=1000, seed=QUAL_SEED
    )
    stdout, draws = octave_run(text, prefix="tbq")

    # the documented data load worked and the driver reached the end
    assert "NOBS 49" in stdout
    assert "QUAL_OK" in stdout

    # the documented printed table is in the run's output
    assert "Posterior Estimates" in stdout

    # the toolbox's own Bayesian draws, parsed back (the struct's draws
    # exclude the omitted burn-in: ndraw - nomit)
    rho = draws["rho"]
    assert rho.shape == (2000,)
    assert -1.0 < rho.mean() < 1.0

    # the toolbox's maximum-likelihood estimate on the same documented call
    mle_rho = float(re.search(r"MLE_RHO (-?[0-9.]+)", stdout).group(1))
    mle_se = float(re.search(r"MLE_RHO_SE (-?[0-9.]+)", stdout).group(1))

    # diffuse priors: the Bayesian posterior mean agrees with its own ML
    # estimate inside Monte-Carlo error (or three ML standard errors)
    mcse = _mcse.mcse_mean(rho)
    tol = max(4.0 * mcse, 3.0 * abs(mle_se))
    assert abs(rho.mean() - mle_rho) <= tol, (
        f"sar_g posterior mean {rho.mean():.4f} vs toolbox MLE {mle_rho:.4f} "
        f"(tol {tol:.4f})"
    )
