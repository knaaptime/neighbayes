r"""The latent SAR, SEM and SDEM logit.

Shared by ``SARLogit(latent=True)`` (:math:`(I - \rho W) y^* = X\beta +
\varepsilon`), :class:`SEMLogit` (:math:`(I - \lambda W)(y^* - X\beta) =
\varepsilon`) and :class:`SDEMLogit`, with :math:`y = \mathbf 1\{y^* > 0\}` and
the disturbance the logit's own, standard logistic.  The sampler is
:mod:`neighbayes.samplers.logit_latent`.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from ...samplers.logit_latent import run_chain  # registers the Gibbs entry
from ..priors import LogitGibbsPriors
from ._latent import LatentMixin


class LatentLogitMixin(LatentMixin):
    """The latent logit's priors, samplers and fitted probabilities."""

    _latent_no_nuts = (
        "its likelihood is an orthant probability of correlated utilities, "
        "which no NUTS model evaluates."
    )
    _latent_chain = staticmethod(run_chain)

    @staticmethod
    def _latent_chains_jax(**kwargs):
        from ...samplers.logit_latent._jax import run_chains_jax_latent_logit

        return run_chains_jax_latent_logit(**kwargs)

    def _latent_priors(self, X, names) -> LogitGibbsPriors:
        bounds = self._logdet_bounds
        beta_mu, beta_sigma = self._resolved_beta_prior(
            design=X, feature_names=names, link="logit"
        )
        return LogitGibbsPriors(
            beta_mu=beta_mu,
            beta_sigma=beta_sigma,
            rho_lower=float(bounds.rho_min),
            rho_upper=float(bounds.rho_max),
        )

    @staticmethod
    def _latent_disturbances(rng, size):
        return rng.logistic(size=size)

    def _fitted_probabilities_latent(
        self, n_sims: int = 1000, random_seed: Optional[int] = 0
    ) -> np.ndarray:
        r"""``P(y_i = 1)`` at the posterior-mean parameters, by simulation.

        The utilities' marginals are not logistic (the multiplier mixes the
        disturbances), so ``n_sims`` disturbance vectors are drawn and the
        share of positive utilities is returned.
        """
        return (self._latent_index_draws(n_sims, random_seed) > 0).mean(axis=1)

    def _fitted_mean_from_posterior_latent(self) -> np.ndarray:
        """The fitted probabilities (:meth:`_fitted_probabilities_latent`)."""
        return self._fitted_probabilities_latent()
