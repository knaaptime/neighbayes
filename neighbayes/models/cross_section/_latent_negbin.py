r"""The latent SAR, SEM and SDEM negative binomial.

Shared by the unmarked :class:`SARNegBin` (:math:`(I - \rho W)\eta = X\beta +
\varepsilon`), :class:`SEMNegBin` (:math:`(I - \lambda W)(\eta - X\beta) =
\varepsilon`) and :class:`SDEMNegBin`, with :math:`y_i \sim
\mathrm{Poisson}(e^{\eta_i})` and :math:`\varepsilon_i = \log\gamma_i`,
:math:`\gamma_i \sim \mathrm{Gamma}(\alpha, \alpha)`: the negative binomial's
own disturbance, so that no dependence is :class:`NegBin`.  The sampler is
:mod:`neighbayes.samplers.negbin_latent`.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import scipy.sparse as sp

# Importing the sampler registers its Gibbs entry.
from ...samplers._utils._latent_loglik import log_gamma_noise
from ...samplers.negbin_latent._core import LatentNegBinChain, zero_set_logdet
from .._base._nb import nb_alpha_fixed
from ..priors import ReducedGibbsPriors
from ._latent import LatentMixin


class LatentNegBinMixin(LatentMixin):
    """The latent negative binomial's priors, samplers and fitted counts."""

    _latent_extra = ("alpha",)
    _latent_no_nuts = (
        "its likelihood integrates the propagated gamma disturbances over all "
        "n locations at once, which no NUTS model evaluates."
    )
    _latent_chain_class = LatentNegBinChain

    @property
    def _latent_chain_kw(self) -> dict:
        """The joint move's log-determinant on the zero counts' sites."""
        b = self._logdet_bounds
        zero = np.asarray(self._y).reshape(self._T if self._is_panel else 1, -1) == 0
        return {
            "zero_logdet": zero_set_logdet(
                self._W_sparse, zero, b.method, b.rho_min, b.rho_max
            )
        }

    @staticmethod
    def _latent_chains_jax(**kwargs):
        from ...samplers.negbin_latent._jax import run_chains_jax_latent_negbin

        return run_chains_jax_latent_negbin(**kwargs)

    def _latent_priors(self, X, names) -> ReducedGibbsPriors:
        bounds = self._logdet_bounds
        beta_mu, beta_sigma = self._resolved_beta_prior(
            design=X, feature_names=names, link="log"
        )
        return ReducedGibbsPriors(
            beta_mu=beta_mu,
            beta_sigma=beta_sigma,
            alpha_sigma=self.priors.get("alpha_sigma", 2.5),
            alpha_nu=self.priors.get("alpha_nu", 3.0),
            alpha_fixed=nb_alpha_fixed(self.priors),
            rho_lower=float(bounds.rho_min),
            rho_upper=float(bounds.rho_max),
        )

    @property
    def _latent_observed(self) -> np.ndarray:
        return self._y_int

    _index_scale = "logmean"
    _outcome_scale = "count"
    _latent_family = 0  # integrated_loglik's NEGBIN

    def impacts(self, return_posterior_samples: bool = False, scale: str = "logmean"):
        r"""Direct, indirect and total impacts on the log-mean or the count scale.

        ``scale="count"`` gives the effects on the expected count
        :math:`\mathbb{E}\,y_i = \mathbb{E}\,e^{\eta_i}`, which averages over the
        propagated gamma heterogeneity (estimated by Monte Carlo per posterior
        draw); ``"logmean"`` the effects on the latent log-mean.
        """
        return self._latent_impacts(scale, return_posterior_samples)

    def _latent_noise(self, rng, size, params) -> np.ndarray:
        """Log-gamma disturbances at one posterior draw's ``alpha``."""
        return log_gamma_noise(rng, float(params["alpha"]), size)

    @staticmethod
    def _latent_outcome_weight(m, U, E, d, params, exact) -> np.ndarray:
        r"""``E exp(m_i + u_i)``, the expected count.

        With an exact multiplier diagonal ``d`` the unit's own factor is taken in
        closed form, :math:`\mathbb{E}\,\gamma_i^{d_i} = \Gamma(\alpha + d_i) /
        (\Gamma(\alpha)\,\alpha^{d_i})`, and only the rest averaged, which
        halves the Monte Carlo spread; with an estimated one the whole average
        is taken, which needs no diagonal and stays unbiased.
        """
        from scipy.special import gammaln

        if exact:
            a = float(params["alpha"])
            own = gammaln(a + d) - gammaln(a) - d * np.log(a)
            U = U - d[:, None] * E
        else:
            own = 0.0
        top = U.max(axis=1, keepdims=True)
        log_mean = top[:, 0] + np.log(np.mean(np.exp(U - top), axis=1))
        return np.exp(np.clip(m + own + log_mean, -50.0, 50.0))

    def _latent_disturbances(self, rng, size):
        return log_gamma_noise(rng, float(self._posterior_mean("alpha")), size)

    def _fitted_mean_from_posterior_latent(
        self, n_sims: int = 1000, random_seed: Optional[int] = 0
    ) -> np.ndarray:
        r"""Expected counts :math:`E[e^{\eta_i}]` at the posterior means, by simulation.

        The multiplier mixes the gamma disturbances, so the expected count is
        not :math:`e^{E\eta_i}`; ``n_sims`` disturbance vectors are drawn and
        the mean of :math:`e^{\eta}` returned.
        """
        return np.exp(self._latent_index_draws(n_sims, random_seed)).mean(axis=1)

    def _posterior_predictive_latent(
        self,
        n_draws: Optional[int] = None,
        random_seed: Optional[int] = None,
    ) -> np.ndarray:
        """Posterior-predictive counts, ``(n_draws, n_obs)``: one fresh simulation per draw.

        Each posterior draw's disturbances are drawn afresh and propagated
        through its own filter, and the counts are Poisson given the log-means.
        """
        self._require_fit()
        post = self._idata.posterior
        par = post[self._latent_param].values.reshape(-1)
        alpha = post["alpha"].values.reshape(-1)
        systematic = self._latent_systematic(post)
        total = par.size if n_draws is None else min(int(n_draws), par.size)
        rng = np.random.default_rng(random_seed)
        n = systematic.shape[1]
        eye = sp.identity(n, format="csc")
        W = self._latent_W
        out = np.empty((total, n))
        for g in range(total):
            eps = self._latent_noise(rng, n, {"alpha": alpha[g]})
            lu = sp.linalg.splu((eye - par[g] * W).tocsc())
            if self._latent_process == "sar":
                eta = lu.solve(systematic[g] + eps)
            else:
                eta = systematic[g] + lu.solve(eps)
            out[g] = rng.poisson(np.exp(np.clip(eta, -30.0, 30.0)))
        return out
