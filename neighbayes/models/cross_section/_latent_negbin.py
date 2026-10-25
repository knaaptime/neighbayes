r"""The latent SAR, SEM and SDEM negative binomial.

Shared by ``SARNegBin(latent=True)`` (:math:`(I - \rho W)\eta = X\beta +
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

# Importing the sampler registers its Gibbs entry.
from ...samplers.negbin_latent._core import LatentNegBinChain
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
        alpha = float(params["alpha"])
        return np.log(rng.gamma(alpha, 1.0 / alpha, size=size))

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
        alpha = float(self._posterior_mean("alpha"))
        return np.log(rng.gamma(alpha, 1.0 / alpha, size=size))

    def _fitted_mean_from_posterior_latent(
        self, n_sims: int = 1000, random_seed: Optional[int] = 0
    ) -> np.ndarray:
        r"""Expected counts :math:`E[e^{\eta_i}]` at the posterior means, by simulation.

        The multiplier mixes the gamma disturbances, so the expected count is
        not :math:`e^{E\eta_i}`; ``n_sims`` disturbance vectors are drawn and
        the mean of :math:`e^{\eta}` returned.
        """
        return np.exp(self._latent_index_draws(n_sims, random_seed)).mean(axis=1)
