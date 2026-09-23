"""Tests for the auxiliary-mixture Poisson flow models."""

import numpy as np
import pytest

from neighbayes.dgp.flows import (
    generate_poisson_flow_data,
    generate_poisson_flow_data_separable,
)
from neighbayes.models.flow import SARPoissonFlow, SARPoissonFlowSeparable


@pytest.fixture(scope="module")
def sep_data():
    return generate_poisson_flow_data_separable(n=36, rho_d=0.35, rho_o=0.25, seed=3)


class TestConstruction:
    def test_rejects_non_integer(self, sep_data):
        y = sep_data["y_vec"].astype(float) + 0.5
        with pytest.raises(ValueError, match="integer"):
            SARPoissonFlowSeparable(y, sep_data["X"], sep_data["G"])

    def test_rejects_negative(self, sep_data):
        y = sep_data["y_vec"].copy()
        y[0] = -1
        with pytest.raises(ValueError, match="non-negative"):
            SARPoissonFlowSeparable(y, sep_data["X"], sep_data["G"])

    def test_accepts_float_valued_integers(self, sep_data):
        y = sep_data["y_vec"].astype(float)
        m = SARPoissonFlowSeparable(y, sep_data["X"], sep_data["G"])
        assert m._y_int_vec.dtype == np.int64

    def test_rejects_nuts(self, sep_data):
        m = SARPoissonFlowSeparable(sep_data["y_vec"], sep_data["X"], sep_data["G"])
        with pytest.raises(NotImplementedError, match="gibbs"):
            m.fit(sampler="nuts")


class TestSeparableRecovery:
    """The separable 2-rho model is the recommended Poisson flow model."""

    def test_recovers_rho_and_mixes(self, sep_data):
        m = SARPoissonFlowSeparable(
            sep_data["y_vec"],
            sep_data["X"],
            sep_data["G"],
            col_names=sep_data["col_names"],
        )
        idata = m.fit(
            draws=1200, tune=600, chains=2, random_seed=11, progressbar=False, n_jobs=1
        )
        import arviz as az

        post = idata.posterior
        ess, rhat = az.ess(idata, method="bulk"), az.rhat(idata)

        assert float(post["rho_d"].mean()) == pytest.approx(0.35, abs=0.10)
        assert float(post["rho_o"].mean()) == pytest.approx(0.25, abs=0.12)
        # Well-mixing: the whole point of the auxiliary-mixture scheme.
        for p in ("rho_d", "rho_o"):
            assert float(ess[p]) > 100, f"{p} ESS too low"
            assert float(rhat[p]) < 1.05, f"{p} rhat too high"

    def test_rho_w_is_deterministic(self, sep_data):
        """Separable model pins rho_w = -rho_d * rho_o."""
        m = SARPoissonFlowSeparable(
            sep_data["y_vec"],
            sep_data["X"],
            sep_data["G"],
            col_names=sep_data["col_names"],
        )
        idata = m.fit(
            draws=200, tune=150, chains=1, random_seed=4, progressbar=False, n_jobs=1
        )
        p = idata.posterior
        np.testing.assert_allclose(
            p["rho_w"].values, -p["rho_d"].values * p["rho_o"].values, atol=1e-10
        )

    def test_idata_surface(self, sep_data):
        m = SARPoissonFlowSeparable(
            sep_data["y_vec"],
            sep_data["X"],
            sep_data["G"],
            col_names=sep_data["col_names"],
        )
        idata = m.fit(
            draws=100,
            tune=80,
            chains=2,
            random_seed=2,
            progressbar=False,
            n_jobs=1,
            idata_kwargs={"log_likelihood": True},
        )
        assert set(idata.posterior.data_vars) == {"beta", "rho_d", "rho_o", "rho_w"}
        # No dispersion parameter: Poisson has no free variance.
        assert "alpha" not in idata.posterior
        assert "log_likelihood" in idata
        assert idata.posterior["beta"].shape[:2] == (2, 100)
        assert list(idata.posterior.coefficient.values) == list(sep_data["col_names"])


class TestUnrestricted:
    """The 3-rho variant is exposed but documented as poorly identified."""

    def test_runs_and_returns_three_rhos(self):
        d = generate_poisson_flow_data(n=12, rho_d=0.3, rho_o=0.2, rho_w=0.1, seed=5)
        m = SARPoissonFlow(d["y_vec"], d["X"], d["G"], col_names=d["col_names"])
        idata = m.fit(
            draws=80, tune=60, chains=1, random_seed=3, progressbar=False, n_jobs=1
        )
        for p in ("rho_d", "rho_o", "rho_w"):
            assert p in idata.posterior
            assert np.all(np.isfinite(idata.posterior[p].values))


class TestPosteriorPredictive:
    """Replicates must be counts on the observation scale, not Gaussian draws.

    Both Poisson classes previously inherited the Gaussian ``SARFlow``
    implementation, which silently returned replicates on the wrong scale.
    """

    def test_separable_replicates_are_counts(self, sep_data):
        m = SARPoissonFlowSeparable(
            sep_data["y_vec"],
            sep_data["X"],
            sep_data["G"],
            col_names=sep_data["col_names"],
        )
        m.fit(
            draws=200, tune=150, chains=2, random_seed=11, progressbar=False, n_jobs=1
        )
        y_rep = m.posterior_predictive(random_seed=0)

        assert y_rep.shape == (400, sep_data["y_vec"].size)
        assert np.all(y_rep >= 0)
        np.testing.assert_array_equal(y_rep, np.round(y_rep))

    def test_separable_replicates_match_observed_dispersion(self, sep_data):
        """Poisson-drawn data: replicated var/mean must track the observed one."""
        y = sep_data["y_vec"]
        m = SARPoissonFlowSeparable(
            y, sep_data["X"], sep_data["G"], col_names=sep_data["col_names"]
        )
        m.fit(
            draws=300, tune=200, chains=2, random_seed=11, progressbar=False, n_jobs=1
        )
        y_rep = m.posterior_predictive(random_seed=0)

        obs = y.var() / y.mean()
        rep = float(np.mean(y_rep.var(axis=1) / y_rep.mean(axis=1)))
        assert rep == pytest.approx(obs, rel=0.35)

    def test_unrestricted_replicates_are_counts(self):
        d = generate_poisson_flow_data(n=10, rho_d=0.3, rho_o=0.2, rho_w=0.1, seed=5)
        m = SARPoissonFlow(d["y_vec"], d["X"], d["G"], col_names=d["col_names"])
        m.fit(draws=60, tune=40, chains=1, random_seed=3, progressbar=False, n_jobs=1)
        y_rep = m.posterior_predictive(random_seed=0)

        assert y_rep.shape == (60, d["y_vec"].size)
        assert np.all(y_rep >= 0)
        np.testing.assert_array_equal(y_rep, np.round(y_rep))

    def test_n_draws_truncates(self, sep_data):
        m = SARPoissonFlowSeparable(
            sep_data["y_vec"],
            sep_data["X"],
            sep_data["G"],
            col_names=sep_data["col_names"],
        )
        m.fit(draws=100, tune=80, chains=2, random_seed=2, progressbar=False, n_jobs=1)
        assert m.posterior_predictive(n_draws=25, random_seed=0).shape[0] == 25
