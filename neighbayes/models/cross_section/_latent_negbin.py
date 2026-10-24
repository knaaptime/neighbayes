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

from ...samplers.negbin_latent import run_chain  # registers the Gibbs entry
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
    _latent_chain = staticmethod(run_chain)

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
