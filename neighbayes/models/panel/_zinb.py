r"""Zero-inflated NB2 panels with count-equation effects.

.. math::

    \eta^{\mathrm{sel}}_t &= (I - \lambda W_{\mathrm{sel}})^{-1} Z_t \gamma, \qquad
    z_{it} \sim \mathrm{Bernoulli}(\mathrm{logit}^{-1}(\eta^{\mathrm{sel}}_{it})), \\
    \eta^{\mathrm{cnt}}_t &= (I - \rho W)^{-1} X_t \beta + c + \tau_t, \qquad
    y_{it} \mid z_{it} = 1 \sim \mathrm{NB2}(e^{\eta^{\mathrm{cnt}}_{it}}, \alpha),
    \quad y_{it} \mid z_{it} = 0 = 0,

stacked time-first over ``T`` periods of ``N`` units.  Structural zeros are per
period; ``Z`` may vary over time (add period dummies to it for period effects
in the selection).  ``effects`` sets unit (pooled) and period effects on
the **count** equation only, as in :class:`~neighbayes.models.SARNegBinPanel`; the
selection equation has none (a unit effect there would compete with the zero
process for the same information).

* :class:`SARZINBPanel` — spatial lag in both equations.
* :class:`ZINBPanel` — the aspatial baseline.

Both default to the Pólya–Gamma Gibbs sampler of
:mod:`neighbayes.samplers.zinb._generic`; ``sampler="nuts"`` fits the same
model in PyMC.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pytensor.tensor as pt
import scipy.sparse as sp

from ..._lazy_deps import pm, xr
from ...samplers._registry import register
from .._base._nb import nb_alpha_fixed, nb_alpha_rv
from .._base._shared import _parse_W
from .._mixins._count_panel import CountPanelFEMixin, require_counts
from .._mixins._zinb import ZINBMixin
from ..cross_section.sar_zinb import _selection_design
from ..panel_base import _EFFECTS_NAMES, SpatialPanelModel
from ..priors import PanelZINBPriors

_PARAMS_DOC = """
    Parameters
    ----------
    formula : str, optional
        Count-equation formula, e.g. ``"y ~ x1 + x2"``. Requires ``data``,
        ``unit_col`` and ``time_col``.
    data : pandas.DataFrame, optional
        Long-format panel data (formula mode).
    y : array-like, optional
        Stacked counts of shape ``(N*T,)``, time-first (matrix mode).
    X : array-like or pandas.DataFrame, optional
        Stacked count design (matrix mode).
    Z : array-like or pandas.DataFrame, optional
        Stacked selection design, time-first. Default: ``X``.
    sel_formula : str, optional
        Selection right-hand side, e.g. ``"~ z1 + z2"`` (formula mode).
    W : libpysal.graph.Graph or scipy.sparse matrix
        ``N × N`` count-equation weights, row-standardized.
    W_sel : libpysal.graph.Graph or scipy.sparse matrix, optional
        ``N × N`` selection weights. Default: ``W``.
    unit_col, time_col : str, optional
        Unit and period columns (formula mode).
    N, T : int, optional
        Units and periods (matrix mode).
    effects : int or str, default 0
        Count-equation effects: ``0``/``"pooled"``, ``1``/``"unit"``,
        ``2``/``"time"``, ``3``/``"two_way"``.
    priors : dict, optional
        See :class:`~neighbayes.models.priors.PanelZINBPriors` (β, ρ, α incl.
        ``alpha_fixed``, unit/period effects, γ, λ bounds).
"""


class _ZINBPanelBase(ZINBMixin, CountPanelFEMixin, SpatialPanelModel):
    """Shared construction, sampling and prediction for the ZINB panels."""

    _priors_cls = PanelZINBPriors
    _likelihood: str = "count"
    _jacobian_param: str | None = None
    _gibbs_key: tuple[str, str] | None = ("zinb", "panel")

    def __init__(
        self,
        *args,
        Z=None,
        sel_formula: Optional[str] = None,
        W_sel=None,
        effects: Any = 0,
        mundlak: bool = False,
        **kwargs,
    ):
        kwargs.pop("model", None)
        kwargs.pop("robust", None)
        super().__init__(*args, effects=0, **kwargs)
        N, T = self._N, self._T
        if self._W_sparse.shape[0] != N:
            raise ValueError(
                f"{type(self).__name__} needs the N × N weights, got "
                f"{self._W_sparse.shape}."
            )
        y_arr = require_counts(self._y_raw, type(self).__name__)
        self._y_int_vec = y_arr.reshape(-1).astype(np.int64)
        self._y = self._y_int_vec.astype(np.float64)

        # Selection design, before the count effects drop absorbed X columns.
        data = kwargs.get("data")
        if sel_formula is not None and data is not None:
            data = data.sort_values(
                [kwargs["time_col"], kwargs["unit_col"]]
            ).reset_index(drop=True)
        self._Z, self._sel_feature_names = _selection_design(
            Z, sel_formula, data, N * T
        )
        if self._Z is None:
            self._Z = np.asarray(self._X, dtype=np.float64).copy()
            self._sel_feature_names = list(self._feature_names)
        if W_sel is not None:
            self._W_sel_sparse, self._is_sel_row_std = _parse_W(W_sel, N)
            self._same_W = False
        else:
            self._W_sel_sparse = self._W_sparse
            self._is_sel_row_std = self._is_row_std
            self._same_W = True

        self._init_count_fe(effects, N, T, mundlak)
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

    def _model_coords(self, extra: Optional[dict] = None) -> dict:
        coords = super()._model_coords(extra)
        coords.update(self._fe_coords())
        coords["sel_coefficient"] = list(self._sel_feature_names)
        return coords

    # ------------------------------------------------------------------
    # Priors and filters
    # ------------------------------------------------------------------

    def _bounds(self, prefix: str) -> tuple[float, float]:
        b = self._logdet_bounds
        lo = float(self.priors.get(f"{prefix}_lower", b.rho_min))
        hi = float(self.priors.get(f"{prefix}_upper", b.rho_max))
        return max(lo, b.rho_min), min(hi, b.rho_max)

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
            np.asarray(self.priors.get("gamma_mu", g_mu), dtype=float), (p,)
        )
        sd = np.broadcast_to(
            np.asarray(self.priors.get("gamma_sigma", g_sd), dtype=float), (p,)
        )
        return mu.copy(), sd.copy()

    def _filters(self):
        from ...samplers.count_panel._filters import NoFilter, SARFilter

        if not self._count_spatial:
            return NoFilter(), NoFilter()
        return (
            SARFilter(self._W_sparse, self._T, *self._bounds("rho")),
            SARFilter(self._W_sel_sparse, self._T, *self._bounds("lam")),
        )

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def _fit_gibbs(
        self,
        draws: int = 2000,
        tune: int = 1000,
        chains: int = 4,
        random_seed: Optional[int] = None,
        thin: int = 1,
        n_jobs: int = -1,
        progressbar: bool = True,
        store_group_effects: Optional[bool] = None,
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
            store_group_effects, chains, draws // max(thin, 1)
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
            extra={"gamma": ("sel_coefficient", self._sel_feature_names)},
        )

    def _build_pymc_model(self) -> pm.Model:
        """The same reduced-form ZINB and priors as the Gibbs path, in PyMC."""
        beta_mu, beta_sigma = self._resolved_beta_prior(link="log")
        gamma_mu, gamma_sigma = self._gamma_prior()
        X_t = pt.as_tensor_variable(self._X.astype(np.float64))
        Z_t = pt.as_tensor_variable(self._Z.astype(np.float64))
        with pm.Model(coords=self._model_coords()) as model:
            beta = pm.Normal("beta", mu=beta_mu, sigma=beta_sigma, dims="coefficient")
            gamma = pm.Normal(
                "gamma", mu=gamma_mu, sigma=gamma_sigma, dims="sel_coefficient"
            )
            eta_cnt = pt.dot(X_t, beta)
            eta_sel = pt.dot(Z_t, gamma)
            if self._count_spatial:
                from ..._ops import SparseSARSolveOp

                rho = pm.Uniform("rho", *self._bounds("rho"))
                lam = pm.Uniform("lam", *self._bounds("lam"))
                eta_cnt = SparseSARSolveOp(self._W_sparse_NT)(rho, eta_cnt)
                W_sel_NT = sp.csr_matrix(
                    sp.kron(sp.eye(self._T, format="csr"), self._W_sel_sparse)
                )
                eta_sel = SparseSARSolveOp(W_sel_NT)(lam, eta_sel)
            eta_cnt = eta_cnt + self._pymc_fe_offset()
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

    def _solve_periods(self, W, coef: float, b: np.ndarray) -> np.ndarray:
        N, T = self._N, self._T
        A = (sp.eye(N, format="csc") - coef * W).tocsc()
        out = sp.linalg.splu(A).solve(np.ascontiguousarray(b.reshape(T, N).T))
        return out.T.reshape(-1)

    def _part_etas(self, g: int) -> tuple[np.ndarray, np.ndarray]:
        """``(η_sel, η_cnt)`` at draw ``g``.

        With the unit effects stored only as a summary
        (``store_group_effects=False``), their posterior mean stands in.
        """
        post = self._idata.posterior
        gamma = self._flat_draw("gamma")[g]
        beta = self._flat_draw("beta")[g]
        eta_sel = self._Z @ gamma
        eta_cnt = self._X @ beta
        if self._count_spatial:
            eta_sel = self._solve_periods(
                self._W_sel_sparse, float(self._flat_draw("lam")[g]), eta_sel
            )
            eta_cnt = self._solve_periods(
                self._W_sparse, float(self._flat_draw("rho")[g]), eta_cnt
            )
        if self._fe_groups:
            if "group_effect" in post.data_vars:
                c = self._flat_draw("group_effect")[g]
            else:
                c = self._idata["group_effect_summary"]["mean"].values
            eta_cnt = eta_cnt + np.asarray(c)[self._row_group]
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
    store_group_effects=None,
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
        store_group_effects=store_group_effects,
        log_likelihood=log_likelihood,
    )


register(
    "zinb",
    "panel",
    run=_run_zinb_panel,
    backends={"numpy"},
    options={"store_group_effects"},
    skips_log_likelihood=True,
)


class SARZINBPanel(_ZINBPanelBase):
    __doc__ = (
        r"""Zero-inflated SAR-NB panel: spatial lag in both equations.

    .. math::

        \eta^{\mathrm{sel}}_t = (I - \lambda W_{\mathrm{sel}})^{-1} Z_t\gamma,
        \quad \eta^{\mathrm{cnt}}_t = (I - \rho W)^{-1} X_t\beta + c + \tau_t
    """
        + _PARAMS_DOC
    )
    _count_spatial = True


class ZINBPanel(_ZINBPanelBase):
    __doc__ = (
        r"""Aspatial zero-inflated NB panel with count-equation effects."""
        + _PARAMS_DOC
    )
    _count_spatial = False
