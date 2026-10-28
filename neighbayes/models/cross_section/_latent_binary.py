r"""The latent SAR, SEM and SDEM logit and probit.

Shared by ``SARLogit(latent=True)`` and :class:`SARProbit` (:math:`(I - \rho
W) y^* = X\beta + \varepsilon`), :class:`SEMLogit` and :class:`SEMProbit`
(:math:`(I - \lambda W)(y^* - X\beta) = \varepsilon`) and the SDEMs, with
:math:`y = \mathbf 1\{y^* > 0\}` and the disturbance the link's own: standard
logistic for the logit, standard normal for the probit.  The sampler is
:mod:`neighbayes.samplers.binary_latent`.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

# Importing the sampler registers its Gibbs entry.
from ...samplers.binary_latent._core import LatentBinaryChain
from ..priors import BinaryGibbsPriors
from ._latent import LatentMixin


class LatentBinaryMixin(LatentMixin):
    """A latent binary model's priors, samplers and fitted probabilities.

    The link's mixin sets ``_link``, the scale of the latent index, the
    disturbance and the family code of the integrated log-likelihood.
    """

    _link: str
    _latent_chain_class = LatentBinaryChain
    _outcome_scale = "probability"

    @property
    def _latent_no_nuts(self) -> str:
        return (
            "its likelihood is an orthant probability of correlated "
            f"{self._latent_disturbance_name} utilities, which no NUTS model "
            "evaluates."
        )

    @property
    def _latent_chain_kw(self) -> dict:
        return {"link": self._link}

    @staticmethod
    def _latent_chains_jax(**kwargs):
        from ...samplers.binary_latent._jax import run_chains_jax_latent_binary

        return run_chains_jax_latent_binary(**kwargs)

    def _latent_priors(self, X, names) -> BinaryGibbsPriors:
        bounds = self._logdet_bounds
        beta_mu, beta_sigma = self._resolved_beta_prior(
            design=X, feature_names=names, link=self._link
        )
        return BinaryGibbsPriors(
            beta_mu=beta_mu,
            beta_sigma=beta_sigma,
            rho_lower=float(bounds.rho_min),
            rho_upper=float(bounds.rho_max),
        )

    @classmethod
    def _latent_disturbances(cls, rng, size):
        return cls._latent_noise(rng, size, {})

    def _fitted_probabilities_latent(
        self, n_sims: int = 1000, random_seed: Optional[int] = 0
    ) -> np.ndarray:
        r"""``P(y_i = 1)`` at the posterior-mean parameters, by simulation.

        The utilities' marginals are not the link's (the multiplier mixes the
        disturbances), so ``n_sims`` disturbance vectors are drawn and the
        share of positive utilities is returned.
        """
        return (self._latent_index_draws(n_sims, random_seed) > 0).mean(axis=1)

    def _fitted_mean_from_posterior_latent(self) -> np.ndarray:
        """The fitted probabilities (:meth:`_fitted_probabilities_latent`)."""
        return self._fitted_probabilities_latent()


class LatentLogitMixin(LatentBinaryMixin):
    """The latent logit: a standard logistic disturbance."""

    _link = "logit"
    _latent_disturbance_name = "logistic"
    _index_scale = "logodds"
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


class LatentProbitMixin(LatentBinaryMixin):
    """The latent probit: a standard normal disturbance."""

    _link = "probit"
    _latent_disturbance_name = "normal"
    _index_scale = "index"
    _latent_family = 2  # integrated_loglik's PROBIT

    def impacts(self, return_posterior_samples: bool = False, scale: str = "index"):
        r"""Direct, indirect and total impacts on the latent index or probability scale.

        ``scale="probability"`` gives the effects on :math:`P(y_i = 1)`, whose
        derivative carries the density of the propagated normal disturbance at
        the threshold (estimated per posterior draw by Monte Carlo,
        conditioning on all but the unit's own disturbance); ``"index"`` the
        effects on the latent index :math:`y^*`, in units of the disturbance's
        standard deviation.
        """
        return self._latent_impacts(scale, return_posterior_samples)

    @staticmethod
    def _latent_noise(rng, size, params) -> np.ndarray:
        return rng.standard_normal(size=size)

    @staticmethod
    def _latent_outcome_weight(m, U, E, d, params, exact) -> np.ndarray:
        """The density of ``u_i`` at ``-m_i``, Rao-Blackwellized over ``ε_i``.

        ``u_i = d_i ε_i + v_i``; given ``v_i`` the density is the normal's,
        scaled by ``d_i``, at ``(-m_i - v_i) / d_i``.
        """
        t = (-m[:, None] - (U - d[:, None] * E)) / d[:, None]
        return np.mean(np.exp(-0.5 * t * t), axis=1) / (np.sqrt(2.0 * np.pi) * d)
