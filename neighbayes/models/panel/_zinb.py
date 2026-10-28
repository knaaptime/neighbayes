r"""Zero-inflated NB2 panels with count-equation effects.

.. math::

    \eta_{z,t} &= (I - \rho_z W_z)^{-1} X_{z,t} \beta_z, \qquad
    z_{it} \sim \mathrm{Bernoulli}(\mathrm{logit}^{-1}(\eta_{z,it})), \\
    \eta^{\mathrm{cnt}}_t &= (I - \rho W)^{-1} (X_t \beta + c) + \tau_t, \qquad
    y_{it} \mid z_{it} = 1 \sim \mathrm{NB2}(e^{\eta^{\mathrm{cnt}}_{it}}, \alpha),
    \quad y_{it} \mid z_{it} = 0 = 0,

stacked time-first over ``T`` periods of ``N`` units.  Structural zeros are per
period; ``X_zero`` may vary over time (add period dummies to it for period
effects in the zero part).  ``unit_effects="random"`` and ``time_effects=True``
set random unit effects (inside the count filter) and period effects on the
**count** part only, as in the :class:`~neighbayes.models.SARNegBin` panel; the
zero part has none (a unit effect there would compete with the zero
process for the same information).  ``ρ_z``, ``β_z`` are the posterior's
``rho_zero``, ``beta_zero``.

The panel design of :class:`~neighbayes.models.SARZINB` (spatial lag in both
parts), and :class:`ZINB`, the aspatial baseline, which is implemented for
panels only.  Both default to the Pólya–Gamma Gibbs sampler of
:mod:`neighbayes.samplers.zinb._generic`; ``sampler="nuts"`` fits the same
model in PyMC.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pytensor.tensor as pt
import scipy.sparse as sp

from ..._lazy_deps import pm, xr
from ...samplers._registry import register
from .._base._nb import nb_alpha_fixed, nb_alpha_rv
from .._base._panel import _EFFECTS_NAMES
from .._mixins._count_panel import CountPanelFEMixin
from .._mixins._designs import CountDesigns
from .._mixins._zero_part import ZeroPartMixin
from .._mixins._zinb import ZINBMixin
from ..base import SpatialModel
from ..priors import PanelZINBPriors


class _ZINBPanel(CountPanelFEMixin):
    """The ZINB panel design (``"panel"``) of SARZINB and ZINB.

    The host builds the zero part (:class:`~.._mixins._zero_part.ZeroPartMixin`)
    before :meth:`_init_panel`, so a default ``Z`` keeps the columns the count
    part's effects absorb.
    """

    def _init_panel(self, effects: int, mundlak: bool) -> None:
        if self._W_sparse.shape[0] != self._N:
            raise ValueError(
                f"{type(self).__name__} needs the N × N weights, got "
                f"{self._W_sparse.shape}."
            )
        self._y_int_vec = self._y_int
        self._init_count_fe(effects, self._N, self._T, mundlak)
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
    # Priors
    # ------------------------------------------------------------------

    def _gamma_prior(self) -> tuple[np.ndarray, np.ndarray]:
        from .._base._shared import gelman_default_beta_prior

        p = self._Z.shape[1]
        g_mu, g_sd = gelman_default_beta_prior(
            np.full(self._Z.shape[0], 0.5),
            self._Z,
            list(self._sel_feature_names),
            link="logit",
        )
        mu = np.broadcast_to(
            np.asarray(self.priors.get("beta_mu_zero", g_mu), dtype=float), (p,)
        )
        sd = np.broadcast_to(
            np.asarray(self.priors.get("beta_sigma_zero", g_sd), dtype=float), (p,)
        )
        return mu.copy(), sd.copy()

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
        """Sample via the generic ZINB Gibbs kernel (registry ``("zinb", "panel")``)."""
        from ...samplers._utils._seeds import spawn_chain_seeds
        from ...samplers.gaussian._chain_runner import run_chains
        from ...samplers.zinb._generic import ZINBPriors, run_chain, sel_name

        filt_cnt, filt_sel = self._filters()
        fe = self._count_fe_spec()
        fp = self._count_fe_priors()
        beta_mu, beta_sigma = self._resolved_beta_prior(link="log")
        gamma_mu, gamma_sigma = self._gamma_prior()
        n_tau = fe.n_tau
        priors = ZINBPriors(
            gamma_mu=gamma_mu,
            gamma_sigma=gamma_sigma,
            theta_mu=np.concatenate([beta_mu, np.full(n_tau, fp["time_effect_mu"])]),
            theta_sigma=np.concatenate(
                [beta_sigma, np.full(n_tau, fp["time_effect_sigma"])]
            ),
            alpha_sigma=float(self.priors.get("alpha_sigma", 2.5)),
            alpha_nu=float(self.priors.get("alpha_nu", 3.0)),
            alpha_fixed=nb_alpha_fixed(self.priors),
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
                filt_sel,
                fe,
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
        scalars = list(filt_cnt.names) + [sel_name(n) for n in filt_sel.names]
        return self._assemble_count_idata(
            results,
            scalars,
            keep_groups,
            log_likelihood,
            extra={"beta_zero": ("coefficient_zero", self._sel_feature_names)},
        )

    def _build_pymc_model_panel(self) -> pm.Model:
        """The same reduced-form ZINB and priors as the Gibbs path, in PyMC."""
        beta_mu, beta_sigma = self._resolved_beta_prior(link="log")
        gamma_mu, gamma_sigma = self._gamma_prior()
        X_t = pt.as_tensor_variable(self._X.astype(np.float64))
        Z_t = pt.as_tensor_variable(self._Z.astype(np.float64))
        with pm.Model(coords=self._model_coords()) as model:
            beta = pm.Normal("beta", mu=beta_mu, sigma=beta_sigma, dims="coefficient")
            gamma = pm.Normal(
                "beta_zero", mu=gamma_mu, sigma=gamma_sigma, dims="coefficient_zero"
            )
            inner, outer = self._pymc_fe_terms()
            eta_cnt = pt.dot(X_t, beta) + inner
            eta_sel = pt.dot(Z_t, gamma)
            if self._count_spatial:
                from ..._ops import SparseSARSolveOp

                rho = pm.Uniform("rho", *self._bounds())
                lam = pm.Uniform("rho_zero", *self._bounds(zero=True))
                eta_cnt = SparseSARSolveOp(self._W_sparse_NT)(rho, eta_cnt)
                W_zero_NT = sp.csr_matrix(
                    sp.kron(sp.eye(self._T, format="csr"), self._W_zero_sparse)
                )
                eta_sel = SparseSARSolveOp(W_zero_NT)(lam, eta_sel)
            eta_cnt = eta_cnt + outer
            alpha = nb_alpha_rv(
                nb_alpha_fixed(self.priors),
                float(self.priors.get("alpha_nu", 3.0)),
                float(self.priors.get("alpha_sigma", 2.5)),
            )
            pm.ZeroInflatedNegativeBinomial(
                "obs",
                psi=pm.math.sigmoid(eta_sel),
                mu=pt.exp(eta_cnt),
                alpha=alpha,
                observed=self._y_int_vec,
            )
        return model

    # ------------------------------------------------------------------
    # Per-draw linear predictors
    # ------------------------------------------------------------------

    def _part_etas_panel(self, g: int) -> tuple[np.ndarray, np.ndarray]:
        """``(η_sel, η_cnt)`` at draw ``g``.

        With the unit effects stored only as a summary
        (``store_effects=False``), their posterior mean stands in.
        """
        post = self._idata.posterior
        gamma = self._flat_draw("beta_zero")[g]
        beta = self._flat_draw("beta")[g]
        eta_sel = self._Z @ gamma
        eta_cnt = self._X @ beta
        if self._fe_groups:
            # The unit effects enter inside the count filter.
            if self._effect_name() in post.data_vars:
                c = self._flat_draw(self._effect_name())[g]
            else:
                c = self._idata[self._effect_name("_summary")]["mean"].values
            eta_cnt = eta_cnt + np.asarray(c)[self._row_group]
        if self._count_spatial:
            eta_sel = self._solve_periods(
                self._W_zero_sparse, float(self._flat_draw("rho_zero")[g]), eta_sel
            )
            eta_cnt = self._solve_periods(
                self._W_sparse, float(self._flat_draw("rho")[g]), eta_cnt
            )
        if self._fe_periods:
            eta_cnt = eta_cnt + self._D_tau @ self._flat_draw("time_effect")[g]
        return eta_sel, eta_cnt


def _run_zinb_panel(
    model,
    *,
    draws,
    tune,
    chains,
    random_seed,
    thin,
    n_jobs,
    progressbar,
    backend,
    store_effects=None,
    log_likelihood=False,
):
    """Registry runner for the ZINB panels: a thin adapter over ``_fit_gibbs``."""
    return model._fit_gibbs(
        draws=draws,
        tune=tune,
        chains=chains,
        random_seed=random_seed,
        thin=thin,
        n_jobs=n_jobs,
        progressbar=progressbar,
        store_effects=store_effects,
        log_likelihood=log_likelihood,
    )


register(
    "zinb",
    "panel",
    run=_run_zinb_panel,
    backends={"numpy"},
    options={"store_effects"},
    skips_log_likelihood=True,
)


class ZINB(CountDesigns, _ZINBPanel, ZeroPartMixin, ZINBMixin, SpatialModel):
    """Zero-inflated negative binomial panel without spatial dependence.

    Implemented for panels only: pass ``time_col`` (formula mode) or ``T``
    (matrix mode).  The arguments are :class:`~neighbayes.models.SARZINB`'s;
    ``W`` (``N × N``) is required, though the aspatial model does not use it.
    """

    _DESIGNS = {"panel": {}}
    _zero_part = True
    _count_spatial = False
    _priors_cls = PanelZINBPriors
    _gibbs_key: tuple[str, str] | None = ("zinb", "panel")
