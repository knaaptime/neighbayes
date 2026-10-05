r"""Posterior summaries of the zero-inflated NB models.

The two-equation machinery (impacts, activation probabilities, draw helpers)
is :class:`~neighbayes.models._mixins._two_part.TwoPartMixin`; this adds what
is specific to zero inflation: the split of each zero into structural and
sampling, the fitted mean :math:`E[y] = \pi\mu`, and prediction with the
latent allocation.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from ._two_part import _SUMMARY_DRAWS, TwoPartMixin, _expit, _nb_zero_prob


class ZINBMixin(TwoPartMixin):
    """Zero attribution and prediction for the ZINB models."""

    def zero_attribution(self, draws: Optional[int] = _SUMMARY_DRAWS) -> dict:
        """Posterior split of each observed zero into structural and sampling.

        For ``y = 0``, ``P(structural | y = 0) = (1−π) / ((1−π) + π·NB(0))``.
        Returned as posterior means and sds over ``draws`` draws (all when
        ``None``).

        Returns
        -------
        dict
            ``structural_prob``, ``structural_prob_sd``, ``sampling_prob``
            (``1 − structural_prob``) and ``zero_indices``.
        """
        self._require_fit()
        zero_idx = np.flatnonzero(np.asarray(self._y) == 0)
        alpha = self._flat_draw("alpha")
        idx = self._draw_indices(draws)
        vals = np.empty((len(idx), zero_idx.size))
        for j, g in enumerate(idx):
            eta_sel, eta_cnt = self._part_etas(int(g))
            pi = _expit(eta_sel[zero_idx])
            nb0 = _nb_zero_prob(eta_cnt[zero_idx], float(alpha[g]))
            structural = 1.0 - pi
            denom = structural + pi * nb0
            vals[j] = np.where(
                denom > 0, structural / np.where(denom > 0, denom, 1), 1.0
            )
        mean = vals.mean(axis=0)
        return {
            "structural_prob": mean,
            "structural_prob_sd": vals.std(axis=0),
            "sampling_prob": 1.0 - mean,
            "zero_indices": zero_idx,
        }

    def _fitted_mean_from_posterior(self) -> np.ndarray:
        """Posterior mean of ``E[y] = π · exp(η_cnt)``, averaged over draws."""
        return self._mean_over_draws(
            lambda es, ec, a: _expit(es) * np.exp(np.clip(ec, -30.0, 30.0))
        )

    def posterior_predictive(
        self,
        n_draws: Optional[int] = None,
        random_seed: Optional[int] = None,
    ) -> np.ndarray:
        """Posterior-predictive counts: ``d ~ Bern(π)``, ``y = d · NB(μ, α)``.

        Returns an array of shape ``(n_draws, n_obs)``.
        """
        self._require_fit()
        alpha = self._flat_draw("alpha")
        total = alpha.shape[0] if n_draws is None else min(int(n_draws), alpha.shape[0])
        rng = np.random.default_rng(random_seed)
        out = None
        for g in range(total):
            eta_sel, eta_cnt = self._part_etas(g)
            a = float(alpha[g])
            mu = np.exp(np.clip(eta_cnt, -30.0, 30.0))
            active = rng.random(mu.size) < _expit(eta_sel)
            y = rng.negative_binomial(a, a / (a + mu)) * active
            if out is None:
                out = np.empty((total, mu.size))
            out[g] = y
        return out
