r"""Posterior summaries of the hurdle NB models.

A hurdle's binary half gives :math:`\pi = P(y > 0)` and its count half a
zero-truncated NB2, so

.. math::

    E[y \mid y > 0] = \frac{\mu}{1 - p_0}, \qquad
    E[y] = \pi \, \frac{\mu}{1 - p_0}, \qquad
    p_0 = \Big(\frac{\alpha}{\mu + \alpha}\Big)^{\alpha}.

The impacts and probabilities are those of
:class:`~neighbayes.models._mixins._two_part.TwoPartMixin`; the binary
equation is called ``"selection"`` there (alias ``"hurdle"``).
"""

from __future__ import annotations

import warnings
from typing import Optional

import numpy as np

from ._two_part import _SUMMARY_DRAWS, TwoPartMixin, _expit, _nb_zero_prob


def _truncated_mean(eta_cnt: np.ndarray, alpha: float) -> np.ndarray:
    mu = np.exp(np.clip(eta_cnt, -30.0, 30.0))
    p0 = _nb_zero_prob(eta_cnt, alpha)
    # μ / (1 − p₀) → 1 as μ → 0 (the truncated NB concentrates on y = 1).
    return np.where(p0 < 1.0, mu / np.maximum(1.0 - p0, 1e-300), 1.0)


def _draw_truncated_nb(rng, mu: np.ndarray, alpha: float) -> np.ndarray:
    """Zero-truncated NB2 draws by resampling the zeros."""
    p = alpha / (alpha + mu)
    y = rng.negative_binomial(alpha, p)
    zero = y == 0
    for _ in range(1000):
        if not zero.any():
            break
        y[zero] = rng.negative_binomial(alpha, p[zero])
        zero = y == 0
    # Left only where p₀ ≈ 1, where the truncated NB is essentially y = 1.
    y[zero] = 1
    return y


class HurdleMixin(TwoPartMixin):
    """Priors, fitted means and prediction for the hurdle models.

    Hosts provide ``_y`` (the counts), ``priors`` and the two designs.
    """

    # ------------------------------------------------------------------
    # Priors (shared by the Gibbs and NUTS paths)
    # ------------------------------------------------------------------

    @property
    def _positive_mean(self) -> float:
        y = np.asarray(self._y, dtype=np.float64)
        pos = y[y > 0]
        return float(pos.mean()) if pos.size else 1.0

    @property
    def _positive_share(self) -> float:
        return float(np.mean(np.asarray(self._y) > 0))

    def _hurdle_beta_prior(self, X, names) -> tuple[np.ndarray, np.ndarray]:
        """Count half: Gelman et al. (2008) on the log scale, at ``log mean(y | y > 0)``."""
        from .._base._shared import gelman_default_beta_prior

        k = X.shape[1]
        mu, sd = gelman_default_beta_prior(
            np.full(X.shape[0], self._positive_mean), X, list(names), link="log"
        )
        mu = np.broadcast_to(np.asarray(self.priors.get("beta_mu", mu), float), (k,))
        sd = np.broadcast_to(np.asarray(self.priors.get("beta_sigma", sd), float), (k,))
        return mu.copy(), sd.copy()

    def _hurdle_gamma_prior(self, Z, names) -> tuple[np.ndarray, np.ndarray]:
        """Binary half: Gelman et al. (2008) on the logit scale, at ``logit(mean d)``."""
        from .._base._shared import gelman_default_beta_prior

        p = Z.shape[1]
        d = (np.asarray(self._y) > 0).astype(np.float64)
        mu, sd = gelman_default_beta_prior(d, Z, list(names), link="logit")
        mu = np.broadcast_to(np.asarray(self.priors.get("gamma_mu", mu), float), (p,))
        sd = np.broadcast_to(
            np.asarray(self.priors.get("gamma_sigma", sd), float), (p,)
        )
        return mu.copy(), sd.copy()

    def _hurdle_alpha(self) -> tuple[float, float, Optional[float]]:
        """``(alpha_sigma, alpha_nu, alpha_fixed)`` of the truncated count."""
        from .._base._nb import nb_alpha_fixed

        return (
            float(self.priors.get("alpha_sigma", 2.5)),
            float(self.priors.get("alpha_nu", 3.0)),
            nb_alpha_fixed(self.priors),
        )

    # ------------------------------------------------------------------
    # Effects in both halves (panel hosts with CountPanelFEMixin)
    # ------------------------------------------------------------------

    def _binary_effects_design(self, n_groups: int, T: int) -> None:
        """The binary design under the effects (and Mundlak means); warns on
        groups that never switch between zero and positive counts.

        Keeps ``_sel_keep`` (raw columns kept, ``None`` when all are),
        ``_sel_design_feature_names`` and ``_sel_mundlak_names``.
        """
        from ._count_panel import effects_design

        names = list(self._sel_feature_names)
        self._sel_design_feature_names = list(names)
        Z_new, names_new, keep, mundlak_names, slopes = effects_design(
            self._Z, names, n_groups, T, self._count_effects, self._mundlak
        )
        self._Z, self._sel_feature_names = Z_new, names_new
        self._sel_keep, self._sel_mundlak_names = keep, mundlak_names
        if slopes:
            warnings.warn(
                f"{slopes} vary only over time and are absorbed by the period "
                "effects in the binary half; their coefficients are not "
                "identified.",
                UserWarning,
                stacklevel=4,
            )
        if self._fe_groups:
            share = np.bincount(
                self._row_group,
                weights=(np.asarray(self._y) > 0).astype(float),
                minlength=n_groups,
            ) / float(T)
            n_fixed = int(np.sum((share == 0.0) | (share == 1.0)))
            if n_fixed:
                warnings.warn(
                    f"{n_fixed} of {n_groups} groups never switch between zero and "
                    "positive counts; their binary effects are informed only by "
                    "the pooled effect distribution.",
                    UserWarning,
                    stacklevel=4,
                )

    def _count_fe_priors(self) -> dict:
        """Count-half effects, centred on the positive counts (``log mean(y | y > 0)``)."""
        level = float(np.log(max(self._positive_mean, 1e-3)))
        return self._effect_priors(level, self._X, "")

    def _sel_fe_priors(self) -> dict:
        """Binary-half effects, centred at ``logit(mean(y > 0))``; keys ``sel_*``."""
        share = float(np.clip(self._positive_share, 1e-3, 1.0 - 1e-3))
        return self._effect_priors(
            float(np.log(share / (1.0 - share))), self._Z, "sel_"
        )

    def _sel_fe_spec(self):
        """The binary half's :class:`~neighbayes.samplers.count_panel.CountPanelFE`."""
        from ...samplers.count_panel import CountPanelFE

        groups = None
        if self._fe_groups:
            groups = self._group_spec(
                self._row_group, self._n_groups, self._sel_fe_priors()
            )
        return CountPanelFE(groups=groups, D_tau=self._D_tau)

    @property
    def _sel_nonintercept_indices(self) -> list[int]:
        """Non-constant binary columns, without the Mundlak group means."""
        skip = set(getattr(self, "_sel_mundlak_names", ()))
        out = []
        for j, name in enumerate(self._sel_feature_names):
            col = self._Z[:, j]
            if name in skip or name.lower() == "intercept" or np.allclose(col, col[0]):
                continue
            out.append(j)
        return out

    def _effects_at(self, g: int, prefix: str) -> np.ndarray:
        """``D_g c + D_τ τ`` of one half at draw ``g`` (summary mean if not stored)."""
        post = self._idata.posterior
        out = np.zeros(self._n_groups * self._T_count)
        if self._fe_groups:
            if prefix + "group_effect" in post.data_vars:
                c = self._flat_draw(prefix + "group_effect")[g]
            else:
                c = self._idata[prefix + "group_effect_summary"]["mean"].values
            out += np.asarray(c)[self._row_group]
        if self._fe_periods:
            out += self._D_tau @ self._flat_draw(prefix + "time_effect")[g]
        return out

    # ------------------------------------------------------------------
    # Posterior expectations and prediction
    # ------------------------------------------------------------------

    def conditional_mean(self, draws: Optional[int] = _SUMMARY_DRAWS) -> np.ndarray:
        """Posterior mean of ``E[y | y > 0] = μ / (1 − p₀)`` for every cell."""
        return self._mean_over_draws(lambda es, ec, a: _truncated_mean(ec, a), draws)

    def _fitted_mean_from_posterior(self) -> np.ndarray:
        """Posterior mean of ``E[y] = π · μ / (1 − p₀)``, averaged over draws."""
        return self._mean_over_draws(
            lambda es, ec, a: _expit(es) * _truncated_mean(ec, a)
        )

    def posterior_predictive(
        self,
        n_draws: Optional[int] = None,
        random_seed: Optional[int] = None,
    ) -> np.ndarray:
        """Posterior-predictive counts: ``d ~ Bern(π)``, ``y | d = 1 ~ tNB(μ, α)``.

        Returns an array of shape ``(n_draws, n_obs)``.
        """
        self._require_fit()
        alpha = self._flat_draw("alpha")
        total = alpha.shape[0] if n_draws is None else min(int(n_draws), alpha.shape[0])
        rng = np.random.default_rng(random_seed)
        out = None
        for g in range(total):
            eta_sel, eta_cnt = self._part_etas(g)
            mu = np.exp(np.clip(eta_cnt, -30.0, 30.0))
            positive = rng.random(mu.size) < _expit(eta_sel)
            y = np.zeros(mu.size)
            if positive.any():
                y[positive] = _draw_truncated_nb(rng, mu[positive], float(alpha[g]))
            if out is None:
                out = np.empty((total, mu.size))
            out[g] = y
        return out
