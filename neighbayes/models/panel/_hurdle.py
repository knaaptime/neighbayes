r"""Hurdle NB2 panels with unit and period effects in both halves.

.. math::

    \eta^{\mathrm{b}}_t &= (I - \lambda W_{\mathrm{sel}})^{-1} Z_t\gamma
        + c^{\mathrm{b}} + \tau^{\mathrm{b}}_t, \qquad
    P(y_{it} > 0) = \mathrm{logit}^{-1}(\eta^{\mathrm{b}}_{it}), \\
    \eta^{\mathrm{c}}_t &= (I - \rho W)^{-1} X_t\beta + c + \tau_t, \qquad
    y_{it} \mid y_{it} > 0 \sim \mathrm{NB2}(e^{\eta^{\mathrm{c}}_{it}}, \alpha)
    \text{ truncated at } 0,

stacked time-first over ``T`` periods of ``N`` units.  ``effects`` sets unit
and period effects on **both** halves.  In the binary half a unit effect
absorbs each unit's baseline chance of a positive count, so γ is identified
from within-unit switches between zero and positive periods (units that never
switch carry no information on γ and their effects are bounded by the prior);
in the count half it absorbs each unit's usual volume.  The halves share no
parameters (linking them through correlated unit effects is a planned
extension), so the posterior factorizes.

* :class:`SARHurdleNBPanel` — spatial lag in both halves.
* :class:`HurdleNBPanel` — the aspatial baseline.

Both default to the Pólya–Gamma Gibbs sampler of
:mod:`neighbayes.samplers.hurdle._generic`; ``sampler="nuts"`` fits the same
model in PyMC.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pytensor.tensor as pt
import scipy.sparse as sp

from ..._lazy_deps import pm, xr
from ...samplers import hurdle as _hurdle  # noqa: F401  (registers the runners)
from .._base._nb import nb_alpha_rv
from .._base._shared import _parse_W
from .._mixins._count_panel import CountPanelFEMixin, require_counts
from .._mixins._hurdle import HurdleMixin
from ..cross_section.sar_zinb import _selection_design
from ..panel_base import _EFFECTS_NAMES, SpatialPanelModel
from ..priors import PanelHurdlePriors

_PARAMS_DOC = """
    Parameters
    ----------
    formula : str, optional
        Count-half formula, e.g. ``"y ~ x1 + x2"``. Requires ``data``,
        ``unit_col`` and ``time_col``.
    data : pandas.DataFrame, optional
        Long-format panel data (formula mode).
    y : array-like, optional
        Stacked counts of shape ``(N*T,)``, time-first (matrix mode).
    X : array-like or pandas.DataFrame, optional
        Stacked count design (matrix mode).
    Z : array-like or pandas.DataFrame, optional
        Stacked binary design, time-first. Default: ``X``.
    sel_formula : str, optional
        Binary right-hand side, e.g. ``"~ z1 + z2"`` (formula mode).
    W : libpysal.graph.Graph or scipy.sparse matrix
        ``N × N`` count-half weights, row-standardized.
    W_sel : libpysal.graph.Graph or scipy.sparse matrix, optional
        ``N × N`` binary-half weights. Default: ``W``.
    unit_col, time_col : str, optional
        Unit and period columns (formula mode).
    N, T : int, optional
        Units and periods (matrix mode).
    effects : int or str, default 0
        Effects in both halves: ``0``/``"pooled"``, ``1``/``"unit"``,
        ``2``/``"time"``, ``3``/``"two_way"``.
    priors : dict, optional
        See :class:`~neighbayes.models.priors.PanelHurdlePriors` (β, ρ, α incl.
        ``alpha_fixed``, the count effects; γ, λ bounds, the binary effects).
"""


class _HurdlePanelBase(HurdleMixin, CountPanelFEMixin, SpatialPanelModel):
    """Shared construction, sampling and prediction for the hurdle panels."""

    _priors_cls = PanelHurdlePriors
    _likelihood: str = "count"
    _jacobian_param: str | None = None
    _gibbs_key: tuple[str, str] | None = ("hurdle", "panel")

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
        if not np.any(self._y > 0):
            raise ValueError(
                f"{type(self).__name__} needs at least one positive count."
            )

        # Binary design, before the effects drop absorbed columns.
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
            extra={"gamma": ("sel_coefficient", self._sel_feature_names)},
            sel_effects=True,
        )

    def _build_pymc_model(self) -> pm.Model:
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
                "gamma", mu=gamma_mu, sigma=gamma_sigma, dims="sel_coefficient"
            )
            eta_c = pt.dot(X_t, beta)
            eta_b = pt.dot(Z_t, gamma)
            if self._count_spatial:
                from ..._ops import SparseSARSolveOp

                rho = pm.Uniform("rho", *self._bounds("rho"))
                lam = pm.Uniform("lam", *self._bounds("lam"))
                eta_c = SparseSARSolveOp(self._W_sparse_NT)(rho, eta_c)
                W_sel_NT = sp.csr_matrix(
                    sp.kron(sp.eye(self._T, format="csr"), self._W_sel_sparse)
                )
                eta_b = SparseSARSolveOp(W_sel_NT)(lam, eta_b)
            eta_c = eta_c + self._pymc_fe_offset()
            eta_b = eta_b + self._pymc_fe_offset("sel_", self._sel_fe_priors())
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

    def _solve_periods(self, W, coef: float, b: np.ndarray) -> np.ndarray:
        N, T = self._N, self._T
        A = (sp.eye(N, format="csc") - coef * W).tocsc()
        out = sp.linalg.splu(A).solve(np.ascontiguousarray(b.reshape(T, N).T))
        return out.T.reshape(-1)

    def _part_etas(self, g: int) -> tuple[np.ndarray, np.ndarray]:
        """``(η_b, η_c)`` at draw ``g``.

        With the unit effects stored only as a summary
        (``store_group_effects=False``), their posterior mean stands in.
        """
        eta_b = self._Z @ self._flat_draw("gamma")[g]
        eta_c = self._X @ self._flat_draw("beta")[g]
        if self._count_spatial:
            eta_b = self._solve_periods(
                self._W_sel_sparse, float(self._flat_draw("lam")[g]), eta_b
            )
            eta_c = self._solve_periods(
                self._W_sparse, float(self._flat_draw("rho")[g]), eta_c
            )
        return eta_b + self._effects_at(g, "sel_"), eta_c + self._effects_at(g, "")


class SARHurdleNBPanel(_HurdlePanelBase):
    __doc__ = (
        r"""Reduced-form SAR hurdle NB panel: spatial lag in both halves.

    .. math::

        \eta^{\mathrm{b}}_t = (I - \lambda W_{\mathrm{sel}})^{-1} Z_t\gamma
        + c^{\mathrm{b}} + \tau^{\mathrm{b}}_t,
        \quad \eta^{\mathrm{c}}_t = (I - \rho W)^{-1} X_t\beta + c + \tau_t
    """
        + _PARAMS_DOC
    )
    _count_spatial = True


class HurdleNBPanel(_HurdlePanelBase):
    __doc__ = r"""Aspatial hurdle NB panel with effects in both halves.""" + _PARAMS_DOC
    _count_spatial = False
