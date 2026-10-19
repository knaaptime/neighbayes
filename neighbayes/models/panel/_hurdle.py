r"""Hurdle NB2 panels with unit and period effects in both parts.

.. math::

    \eta_{z,t} &= (I - \rho_z W_z)^{-1} (X_{z,t}\beta_z + c_z)
        + \tau_{z,t}, \qquad
    P(y_{it} > 0) = \mathrm{logit}^{-1}(\eta_{z,it}), \\
    \eta^{\mathrm{c}}_t &= (I - \rho W)^{-1} (X_t\beta + c) + \tau_t, \qquad
    y_{it} \mid y_{it} > 0 \sim \mathrm{NB2}(e^{\eta^{\mathrm{c}}_{it}}, \alpha)
    \text{ truncated at } 0,

stacked time-first over ``T`` periods of ``N`` units.  ``unit_effects="random"``
and ``time_effects=True`` set unit and period effects on **both** parts; each part's random unit effects enter
inside its own filter.  In the zero part a unit effect absorbs
each unit's baseline chance of a positive count, so ``β_z`` is identified from
within-unit switches between zero and positive periods (units that never
switch carry no information on ``β_z`` and their effects are bounded by the
prior); in the count part it absorbs each unit's usual volume.  The parts
share no parameters (linking them through correlated unit effects is a planned
extension), so the posterior factorizes.  ``ρ_z``, ``β_z`` are the posterior's
``rho_zero``, ``beta_zero``.

The panel design of :class:`~neighbayes.models.SARHurdleNegBin` (spatial lag
in both parts), and :class:`HurdleNegBin`, the aspatial baseline, which is
implemented for panels only.  Both default to the Pólya–Gamma Gibbs sampler of
:mod:`neighbayes.samplers.hurdle._generic`; ``sampler="nuts"`` fits the same
model in PyMC.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pytensor.tensor as pt
import scipy.sparse as sp

from ..._lazy_deps import pm, xr
from ...samplers import hurdle as _hurdle  # noqa: F401  (registers the runners)
from .._base._nb import nb_alpha_rv
from .._base._panel import _EFFECTS_NAMES
from .._mixins._count_panel import CountPanelFEMixin
from .._mixins._designs import CountDesigns
from .._mixins._hurdle import HurdleMixin
from .._mixins._zero_part import ZeroPartMixin
from ..base import SpatialModel
from ..priors import PanelHurdlePriors


class _HurdlePanel(CountPanelFEMixin):
    """The hurdle panel design (``"panel"``) of SARHurdleNegBin and HurdleNegBin.

    The host builds the zero part (:class:`~.._mixins._zero_part.ZeroPartMixin`)
    before :meth:`_init_panel`, which puts both parts' designs through the
    effects.
    """

    def _init_panel(self, effects: int, mundlak: bool) -> None:
        N, T = self._N, self._T
        if self._W_sparse.shape[0] != N:
            raise ValueError(
                f"{type(self).__name__} needs the N × N weights, got "
                f"{self._W_sparse.shape}."
            )
        self._y_int_vec = self._y_int
        self._init_count_fe(effects, N, T, mundlak)
        self._binary_effects_design(N, T)
        self.model = self._count_effects
        self.effects = _EFFECTS_NAMES[self._count_effects]
        self._wx_column_indices = self._spatial_lag_column_indices(
            self._X, self._feature_names
        )
        self._wx_feature_names = [
            self._feature_names[i] for i in self._wx_column_indices
        ]
        self._WX = (
            self._sparse_panel_lag(self._X[:, self._wx_column_indices])
            if self._wx_column_indices
            else np.empty((self._X.shape[0], 0))
        )

    def _model_coords_panel(self, extra: Optional[dict] = None) -> dict:
        coords = super()._model_coords(extra)
        coords.update(self._fe_coords())
        coords["coefficient_zero"] = list(self._sel_feature_names)
        return coords

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def _fit_gibbs_panel(
        self,
        draws: int = 2000,
        tune: int = 1000,
        chains: int = 4,
        random_seed: Optional[int] = None,
        thin: int = 1,
        n_jobs: int = -1,
        progressbar: bool = True,
        store_effects: Optional[bool] = None,
        log_likelihood: bool = False,
    ) -> xr.DataTree:
        """Sample via the generic hurdle kernel (registry ``("hurdle", "panel")``)."""
        from ...samplers._utils._seeds import spawn_chain_seeds
        from ...samplers.gaussian._chain_runner import run_chains
        from ...samplers.hurdle._generic import HurdlePriors, run_chain
        from ...samplers.zinb._generic import sel_name

        filt_cnt, filt_bin = self._filters()
        fe_cnt, fe_bin = self._count_fe_spec(), self._sel_fe_spec()
        fp, sp_ = self._count_fe_priors(), self._sel_fe_priors()
        beta_mu, beta_sigma = self._hurdle_beta_prior(self._X, self._feature_names)
        gamma_mu, gamma_sigma = self._hurdle_gamma_prior(
            self._Z, self._sel_feature_names
        )
        a_sigma, a_nu, a_fixed = self._hurdle_alpha()
        n_tau = fe_cnt.n_tau
        priors = HurdlePriors(
            gamma_mu=np.concatenate([gamma_mu, np.full(n_tau, sp_["time_effect_mu"])]),
            gamma_sigma=np.concatenate(
                [gamma_sigma, np.full(n_tau, sp_["time_effect_sigma"])]
            ),
            theta_mu=np.concatenate([beta_mu, np.full(n_tau, fp["time_effect_mu"])]),
            theta_sigma=np.concatenate(
                [beta_sigma, np.full(n_tau, fp["time_effect_sigma"])]
            ),
            alpha_sigma=a_sigma,
            alpha_nu=a_nu,
            alpha_fixed=a_fixed,
        )
        keep_groups = self._store_group_draws(
            store_effects, chains, draws // max(thin, 1)
        )
        y = self._y.astype(np.float64)
        X = np.asarray(self._X, dtype=np.float64)
        Z = np.asarray(self._Z, dtype=np.float64)

        def _chain_fn(chain_id, seed, progress_manager=None, chain_id_kw=0):
            return run_chain(
                y,
                X,
                Z,
                filt_cnt,
                filt_bin,
                fe_cnt,
                fe_bin,
                priors,
                draws,
                tune,
                thin=thin,
                rng=np.random.default_rng(seed),
                chain_id=chain_id,
                progress_manager=progress_manager,
                store_log_lik=log_likelihood,
                store_group_draws=keep_groups,
            )

        seeds = (
            spawn_chain_seeds(random_seed, chains) if random_seed is not None else None
        )
        results = run_chains(
            chain_fn=_chain_fn,
            n_chains=chains,
            seeds=seeds,
            n_jobs=n_jobs,
            progressbar=progressbar,
            parallel=n_jobs != 1,
            draws=draws,
            tune=tune,
            model_type=type(self).__name__,
            timeout=None,
        )
        scalars = list(filt_cnt.names) + [sel_name(n) for n in filt_bin.names]
        return self._assemble_count_idata(
            results,
            scalars,
            keep_groups,
            log_likelihood,
            extra={"beta_zero": ("coefficient_zero", self._sel_feature_names)},
            sel_effects=True,
        )

    def _build_pymc_model_panel(self) -> pm.Model:
        """The same reduced-form hurdle and priors as the Gibbs path, in PyMC."""
        beta_mu, beta_sigma = self._hurdle_beta_prior(self._X, self._feature_names)
        gamma_mu, gamma_sigma = self._hurdle_gamma_prior(
            self._Z, self._sel_feature_names
        )
        a_sigma, a_nu, a_fixed = self._hurdle_alpha()
        X_t = pt.as_tensor_variable(self._X.astype(np.float64))
        Z_t = pt.as_tensor_variable(self._Z.astype(np.float64))
        with pm.Model(coords=self._model_coords()) as model:
            beta = pm.Normal("beta", mu=beta_mu, sigma=beta_sigma, dims="coefficient")
            gamma = pm.Normal(
                "beta_zero", mu=gamma_mu, sigma=gamma_sigma, dims="coefficient_zero"
            )
            inner_c, outer_c = self._pymc_fe_terms()
            inner_b, outer_b = self._pymc_fe_terms(True, self._sel_fe_priors())
            eta_c = pt.dot(X_t, beta) + inner_c
            eta_b = pt.dot(Z_t, gamma) + inner_b
            if self._count_spatial:
                from ..._ops import SparseSARSolveOp

                rho = pm.Uniform("rho", *self._bounds())
                lam = pm.Uniform("rho_zero", *self._bounds(zero=True))
                eta_c = SparseSARSolveOp(self._W_sparse_NT)(rho, eta_c)
                W_zero_NT = sp.csr_matrix(
                    sp.kron(sp.eye(self._T, format="csr"), self._W_zero_sparse)
                )
                eta_b = SparseSARSolveOp(W_zero_NT)(lam, eta_b)
            eta_c = eta_c + outer_c
            eta_b = eta_b + outer_b
            alpha = nb_alpha_rv(a_fixed, a_nu, a_sigma)
            pm.HurdleNegativeBinomial(
                "obs",
                psi=pm.math.sigmoid(eta_b),
                mu=pt.exp(eta_c),
                alpha=alpha,
                observed=self._y_int_vec,
            )
        return model

    # ------------------------------------------------------------------
    # Per-draw linear predictors
    # ------------------------------------------------------------------

    def _part_etas_panel(self, g: int) -> tuple[np.ndarray, np.ndarray]:
        """``(η_b, η_c)`` at draw ``g``.

        With the unit effects stored only as a summary
        (``store_effects=False``), their posterior mean stands in.
        """
        inner_b, outer_b = self._effects_at(g, True)
        inner_c, outer_c = self._effects_at(g, False)
        eta_b = self._Z @ self._flat_draw("beta_zero")[g] + inner_b
        eta_c = self._X @ self._flat_draw("beta")[g] + inner_c
        if self._count_spatial:
            eta_b = self._solve_periods(
                self._W_zero_sparse, float(self._flat_draw("rho_zero")[g]), eta_b
            )
            eta_c = self._solve_periods(
                self._W_sparse, float(self._flat_draw("rho")[g]), eta_c
            )
        return eta_b + outer_b, eta_c + outer_c


class HurdleNegBin(
    CountDesigns, _HurdlePanel, ZeroPartMixin, HurdleMixin, SpatialModel
):
    """Hurdle negative binomial panel without spatial dependence.

    Implemented for panels only: pass ``time_col`` (formula mode) or ``T``
    (matrix mode).  The arguments are
    :class:`~neighbayes.models.SARHurdleNegBin`'s; ``W`` (``N × N``) is
    required, though the aspatial model does not use it.
    """

    _DESIGNS = {"panel": {}}
    _zero_part = True
    _needs_positive = True
    _count_spatial = False
    _priors_cls = PanelHurdlePriors
    _gibbs_key: tuple[str, str] | None = ("hurdle", "panel")
