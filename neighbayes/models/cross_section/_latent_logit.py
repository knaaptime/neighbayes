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

    _index_scale = "logodds"
    _outcome_scale = "probability"
    _latent_family = 1  # integrated_loglik's LOGIT

    def impacts(self, return_posterior_samples: bool = False, scale: str = "logodds"):
        r"""Direct, indirect and total impacts on the log-odds or probability scale.

        ``scale="probability"`` gives the effects on :math:`P(y_i = 1)`, whose
        derivative carries the density of the propagated logistic disturbance
        at the threshold (estimated per posterior draw by Monte Carlo,
        conditioning on all but the unit's own disturbance); ``"logodds"`` the
        effects on the latent index.
        """
        return self._latent_impacts(scale, return_posterior_samples)

    @staticmethod
    def _latent_noise(rng, size, params) -> np.ndarray:
        return rng.logistic(size=size)

    @staticmethod
    def _latent_outcome_weight(m, U, E, d, params, exact) -> np.ndarray:
        """The density of ``u_i`` at ``-m_i``, Rao-Blackwellized over ``ε_i``.

        ``u_i = d_i ε_i + v_i``; given ``v_i`` the density is the logistic's,
        scaled by ``d_i``, at ``(-m_i - v_i) / d_i``.
        """
        t = np.abs((-m[:, None] - (U - d[:, None] * E)) / d[:, None])
        e = np.exp(-t)
        return np.mean(e / (1.0 + e) ** 2, axis=1) / d

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
