r"""Posterior summaries shared by the two-part count models (ZINB and hurdle).

A two-part model has two linear predictors: a binary log-odds
:math:`\eta^{\mathrm{sel}}` with :math:`\pi = \mathrm{logit}^{-1}(\eta^{\mathrm{sel}})`
— whether a unit is active (ZINB) or has any count (hurdle) — and the count
log-mean :math:`\eta^{\mathrm{cnt}}`.  Each model supplies them for one
posterior draw through :meth:`TwoPartMixin._part_etas`; everything here is
built on that.

Probabilities and fitted means are posterior expectations, averaged over
draws — not plug-ins at the posterior means, which for these nonlinear
functions are not the same thing.
"""

from __future__ import annotations

from functools import cached_property
from typing import Optional

import numpy as np

#: Draws averaged by the posterior summaries unless the caller asks otherwise.
_SUMMARY_DRAWS = 200

#: Accepted names of the binary equation in :meth:`TwoPartMixin.spatial_effects`.
_BINARY_NAMES = ("selection", "hurdle")


def _expit(x: np.ndarray) -> np.ndarray:
    return 0.5 * (1.0 + np.tanh(0.5 * x))


def _nb_zero_prob(eta_cnt: np.ndarray, alpha: float) -> np.ndarray:
    """``NB(0; exp(η), α) = (α / (μ + α))^α``, computed on the log scale."""
    mu = np.exp(np.clip(eta_cnt, -30.0, 30.0))
    return np.exp(np.clip(alpha * np.log(alpha / (mu + alpha)), -700.0, 0.0))


class TwoPartMixin:
    """Two-equation impacts and binary-equation probabilities.

    Hosts provide ``_Z``, ``_sel_feature_names``, ``_W_sel_sparse``,
    ``_same_W``, ``_is_sel_row_std``, ``_y`` and :meth:`_part_etas`.
    """

    # ------------------------------------------------------------------
    # Per-draw linear predictors
    # ------------------------------------------------------------------

    def _part_etas(self, g: int) -> tuple[np.ndarray, np.ndarray]:
        """``(η_sel, η_cnt)`` at posterior draw ``g`` (flat chain × draw index)."""
        raise NotImplementedError

    def _flat_draw(self, name: str) -> np.ndarray:
        """A posterior variable flattened over (chain, draw), leading axis."""
        v = self._idata.posterior[name].values
        return v.reshape((-1,) + v.shape[2:])

    def _draw_indices(self, draws: Optional[int]) -> np.ndarray:
        total = self._flat_draw("alpha").shape[0]
        if draws is None or draws >= total:
            return np.arange(total)
        return np.unique(np.linspace(0, total - 1, int(draws)).round().astype(int))

    # ------------------------------------------------------------------
    # Selection-equation helpers (impacts of Z)
    # ------------------------------------------------------------------

    @cached_property
    def _sel_logdet_grad_numpy_vec_fn(self):
        """Vectorized ``(λ_arr) -> g(λ)`` logdet-gradient evaluator for W_sel."""
        from ..._logdet import make_logdet_grad_numpy_vec_fn

        return make_logdet_grad_numpy_vec_fn(
            self._W_sel_sparse,
            eigs=None,
            method=None,
            rho_min=float(self.priors.get("lam_lower", self._logdet_bounds.rho_min)),
            rho_max=float(self.priors.get("lam_upper", self._logdet_bounds.rho_max)),
        )

    def _sel_batch_mean_diag(self, lam_draws: np.ndarray) -> np.ndarray:
        """``(1/n) tr((I − λ W_sel)⁻¹)`` per draw, via the resolvent identity."""
        if self._same_W:
            return self._batch_mean_diag(lam_draws)
        lam_draws = np.asarray(lam_draws, dtype=np.float64)
        n = int(self._W_sel_sparse.shape[0])
        g = np.asarray(self._sel_logdet_grad_numpy_vec_fn(lam_draws), dtype=np.float64)
        return 1.0 - (lam_draws / n) * g

    @cached_property
    def _sel_nonintercept_indices(self) -> list[int]:
        """Indices of non-constant columns in Z (selection covariates)."""
        indices: list[int] = []
        for j, name in enumerate(self._sel_feature_names):
            column = self._Z[:, j]
            if not (name.lower() == "intercept" or np.allclose(column, column[0])):
                indices.append(j)
        return indices

    @cached_property
    def _sel_nonintercept_feature_names(self) -> list[str]:
        return [self._sel_feature_names[i] for i in self._sel_nonintercept_indices]

    # ------------------------------------------------------------------
    # Impacts
    # ------------------------------------------------------------------

    def _compute_spatial_effects_posterior(
        self, equation: str = "count"
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Posterior impacts per draw for the count or the binary equation."""
        from ...diagnostics.lmtests import _get_posterior_draws

        idata = self.inference_data
        post = idata.posterior
        if equation == "count":
            beta_draws = _get_posterior_draws(idata, "beta")
            ni = self._nonintercept_indices
            if "rho" not in post.data_vars:  # aspatial count equation
                direct = beta_draws[:, ni]
                return direct, np.zeros_like(direct), direct.copy()
            rho_draws = _get_posterior_draws(idata, "rho")
            mean_diag = self._batch_mean_diag(rho_draws)
            mean_row_sum = self._batch_mean_row_sum(rho_draws)
            direct = mean_diag[:, None] * beta_draws[:, ni]
            total = mean_row_sum[:, None] * beta_draws[:, ni]
        elif equation in _BINARY_NAMES:
            gamma_draws = _get_posterior_draws(idata, "gamma")
            if "lam" not in post.data_vars:  # aspatial binary equation
                direct = gamma_draws[:, self._sel_nonintercept_indices]
                return direct, np.zeros_like(direct), direct.copy()
            lam_draws = _get_posterior_draws(idata, "lam")
            mean_diag = self._sel_batch_mean_diag(lam_draws)
            if self._is_sel_row_std:
                mean_row_sum = 1.0 / (1.0 - lam_draws)
            else:
                # 1'S1 is a column-weighted bilinear form, not a trace, so the
                # non-row-standardized total keeps the eigen decomposition.
                from ...diagnostics.spatial_effects import _chunked_eig_means

                n = self._W_sel_sparse.shape[0]
                W_dense = self._W_sel_sparse.toarray().astype(np.float64)
                eigvals, V = np.linalg.eig(W_dense)
                c = np.linalg.solve(V, np.ones(n))
                mean_row_sum = _chunked_eig_means(
                    lam_draws, eigvals, weights=V.sum(axis=0) * c
                )
            ni = self._sel_nonintercept_indices
            direct = mean_diag[:, None] * gamma_draws[:, ni]
            total = mean_row_sum[:, None] * gamma_draws[:, ni]
        else:
            raise ValueError(
                f"equation must be 'count' or 'selection', got '{equation}'"
            )
        return direct, total - direct, total

    def spatial_effects(
        self,
        equation: str = "count",
        return_posterior_samples: bool = False,
    ):
        """Direct, indirect and total impacts for one equation.

        Parameters
        ----------
        equation : {"count", "selection"}, default "count"
            ``"count"``: impacts of X on the count log-mean, through
            ``(I − ρW)⁻¹Xβ``.  ``"selection"`` (alias ``"hurdle"``): impacts of
            Z on the log-odds of the binary equation, through
            ``(I − λW_sel)⁻¹Zγ``.
        return_posterior_samples : bool, default False
            Also return the per-draw impacts.
        """
        from ...diagnostics.spatial_effects import _build_effects_dataframe

        self._require_fit()
        direct, indirect, total = self._compute_spatial_effects_posterior(
            equation=equation
        )
        k = direct.shape[1]
        if equation in _BINARY_NAMES:
            names = self._sel_nonintercept_feature_names
            feature_names = (
                list(names) if len(names) == k else list(self._sel_feature_names[:k])
            )
        elif len(self._nonintercept_feature_names) == k:
            feature_names = list(self._nonintercept_feature_names)
        else:
            feature_names = list(self._feature_names[:k])
        df = _build_effects_dataframe(
            direct_samples=direct,
            indirect_samples=indirect,
            total_samples=total,
            feature_names=feature_names,
            model_type=f"{type(self).__name__} ({equation})",
        )
        if return_posterior_samples:
            return df, {"direct": direct, "indirect": indirect, "total": total}
        return df

    # ------------------------------------------------------------------
    # Posterior expectations
    # ------------------------------------------------------------------

    def corridor_probabilities(
        self, draws: Optional[int] = _SUMMARY_DRAWS
    ) -> np.ndarray:
        """Posterior mean of each binary probability ``π = logit⁻¹(η_sel)``.

        The activation probability for ZINB, ``P(y > 0)`` for a hurdle.
        Averaged over ``draws`` posterior draws (all when ``None``).
        """
        self._require_fit()
        idx = self._draw_indices(draws)
        acc = None
        for g in idx:
            pi = _expit(self._part_etas(int(g))[0])
            acc = pi if acc is None else acc + pi
        return acc / len(idx)

    def _mean_over_draws(self, fn, draws: Optional[int] = _SUMMARY_DRAWS):
        """Posterior mean of ``fn(η_sel, η_cnt, α)`` over ``draws`` draws."""
        self._require_fit()
        alpha = self._flat_draw("alpha")
        idx = self._draw_indices(draws)
        acc = None
        for g in idx:
            eta_sel, eta_cnt = self._part_etas(int(g))
            v = fn(eta_sel, eta_cnt, float(alpha[g]))
            acc = v if acc is None else acc + v
        return acc / len(idx)
