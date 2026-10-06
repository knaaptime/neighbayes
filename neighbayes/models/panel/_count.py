r"""NB2 count panels with pooled unit effects and period effects.

.. math::

    \eta_t = (I_N - \rho W)^{-1} X_t \beta + c + \tau_t \mathbf 1,
    \qquad y_{it} \sim \mathrm{NB2}(e^{\eta_{it}}, \alpha),

stacked time-first over ``T`` periods of ``N`` units.  The spatial filter acts
on the log-mean only — ``y`` is not transformed, so there is no
:math:`\log|I - \rho W|` Jacobian (see
:class:`~neighbayes.models.SARNegBin`).

* :class:`SARNegBinPanel` — spatial lag on the log-mean, sampled by
  reduced-form Pólya–Gamma Gibbs with β, the period effects and the unit
  effects integrated out of every ρ update.
* :class:`NegBinPanel` — the aspatial baseline.

Both default to Gibbs, as the Gaussian panels do; ``sampler="nuts"`` runs the
PyMC model instead.

``effects``: ``0``/``"pooled"``, ``1``/``"unit"``, ``2``/``"time"``,
``3``/``"two_way"``.  The Gaussian panels' within transform does not carry to a
log link, so the effects are parameters, held as a unit index rather than
dummy columns (:mod:`~neighbayes.models._mixins._count_panel`).  Under unit
effects ρ and β are identified only by within-unit variation of ``X`` over
time, and the dispersion α and the period effects carry an incidental-parameter
bias of order ``1/T`` (Allison & Waterman 2002).

``priors={"alpha_fixed": a}`` holds α at ``a``; a large value (10-20 times the
typical mean count) gives an essentially Poisson model that keeps the exact
Pólya–Gamma sampler (:mod:`~neighbayes.models._base._nb`).
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pytensor.tensor as pt
import scipy.sparse as sp

from ..._lazy_deps import pm, xr
from ...samplers._registry import register
from .._base._nb import nb_alpha_fixed, nb_alpha_rv
from .._mixins._count_panel import CountPanelFEMixin, require_counts
from ..panel_base import _EFFECTS_NAMES, SpatialPanelModel
from ..priors import PanelCountPriors

_PARAMS_DOC = """
    Parameters
    ----------
    formula : str, optional
        Wilkinson-style formula, e.g. ``"y ~ x1 + x2"``. Requires ``data``,
        ``unit_col`` and ``time_col``.
    data : pandas.DataFrame, optional
        Long-format panel data when using formula mode.
    y : array-like, optional
        Stacked counts of shape ``(N*T,)``, time-first. Required in matrix mode.
    X : array-like or pandas.DataFrame, optional
        Stacked design matrix. Required in matrix mode.
    W : libpysal.graph.Graph or scipy.sparse matrix
        ``N × N`` spatial weights, row-standardized.
    unit_col, time_col : str, optional
        Unit and period columns in ``data`` (formula mode).
    N, T : int, optional
        Units and periods (matrix mode).
    effects : int or str, default 0
        ``0``/``"pooled"``, ``1``/``"unit"``, ``2``/``"time"``,
        ``3``/``"two_way"``.
    priors : dict, optional
        Overrides; see :class:`~neighbayes.models.priors.PanelCountPriors`
        (``beta_mu``, ``beta_sigma``, ``rho_lower``, ``rho_upper``,
        ``alpha_sigma``, ``alpha_nu``, ``alpha_fixed``, ``group_effect_mu``,
        ``group_effect_sd_scale``, ``time_effect_mu``, ``time_effect_sigma``).
    mundlak : bool, default False
        Add the unit means of the time-varying columns to the design (inside
        the filter), absorbing a correlation between the unit effects and
        ``X``.  Needs unit effects.
    logdet_method : str, optional
        Only used for the log-mean impacts (``tr((I − ρW)⁻¹)``).
"""


class _CountPanelBase(CountPanelFEMixin, SpatialPanelModel):
    """Shared construction, sampling and prediction for the NB count panels."""

    _priors_cls = PanelCountPriors
    _likelihood: str = "count"
    _jacobian_param: str | None = None

    _gibbs_key: tuple[str, str] | None = ("count", "panel")

    def __init__(self, *args, effects: Any = 0, mundlak: bool = False, **kwargs):
        kwargs.pop("model", None)
        kwargs.pop("robust", None)
        super().__init__(*args, effects=0, **kwargs)
        if self._W_sparse.shape[0] != self._N:
            raise ValueError(
                f"{type(self).__name__} needs the N × N weights, got "
                f"{self._W_sparse.shape}."
            )
        y_arr = require_counts(self._y_raw, type(self).__name__)
        self._y_int_vec = y_arr.reshape(-1).astype(np.int64)
        self._y = self._y_int_vec.astype(np.float64)
        self._init_count_fe(effects, self._N, self._T, mundlak)
        self.model = self._count_effects
        self.effects = _EFFECTS_NAMES[self._count_effects]
        # Re-derive the lag bookkeeping on the design that survived the effects.
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
        return coords

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def _filter(self):
        from ...samplers.count_panel._filters import NoFilter, SARFilter

        if not self._count_spatial:
            return NoFilter()
        b = self._logdet_bounds
        return SARFilter(self._W_sparse, self._T, b.rho_min, b.rho_max)

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
        """Sample via reduced-form count Gibbs (registry entry ``("count", "panel")``).

        Reached through :meth:`SpatialPanelModel.fit` with ``sampler="gibbs"``,
        the default (``sampler="nuts"`` runs :meth:`_build_pymc_model`).  ``store_group_effects``
        (a ``fit`` keyword) keeps every draw of the unit effects — by default
        when they fit in about 500 MB — and otherwise their posterior mean and
        sd in ``idata["group_effect_summary"]``.
        """
        beta_mu, beta_sigma = self._resolved_beta_prior(link="log")
        return self._run_count_panel_gibbs(
            self._filter(),
            beta_mu=beta_mu,
            beta_sigma=beta_sigma,
            alpha_sigma=float(self.priors.get("alpha_sigma", 2.5)),
            alpha_nu=float(self.priors.get("alpha_nu", 3.0)),
            alpha_fixed=nb_alpha_fixed(self.priors),
            draws=draws,
            tune=tune,
            chains=chains,
            random_seed=random_seed,
            thin=thin,
            progressbar=progressbar,
            n_jobs=n_jobs,
            log_likelihood=log_likelihood,
            store_group_effects=store_group_effects,
            model_type=type(self).__name__,
        )

    def _build_pymc_model(self) -> pm.Model:
        """Reduced-form PyMC model; no ``log|I − ρW|`` term (``y`` is not transformed)."""
        beta_mu, beta_sigma = self._resolved_beta_prior(link="log")
        X_t = pt.as_tensor_variable(self._X.astype(np.float64))
        with pm.Model(coords=self._model_coords()) as model:
            beta = pm.Normal("beta", mu=beta_mu, sigma=beta_sigma, dims="coefficient")
            eta = pt.dot(X_t, beta)
            if self._count_spatial:
                from ..._ops import SparseSARSolveOp

                b = self._logdet_bounds
                rho = pm.Uniform("rho", lower=b.rho_min, upper=b.rho_max)
                eta = SparseSARSolveOp(self._W_sparse_NT)(rho, eta)
            eta = eta + self._pymc_fe_offset()
            alpha = nb_alpha_rv(
                nb_alpha_fixed(self.priors),
                float(self.priors.get("alpha_nu", 3.0)),
                float(self.priors.get("alpha_sigma", 2.5)),
            )
            pm.NegativeBinomial(
                "obs", mu=pt.exp(eta), alpha=alpha, observed=self._y_int_vec
            )
        return model

    # ------------------------------------------------------------------
    # Posterior summaries
    # ------------------------------------------------------------------

    def _eta_reduced(self, rho: Optional[float], beta: np.ndarray) -> np.ndarray:
        """``(I_T ⊗ (I − ρW))⁻¹ Xβ`` for one draw, time-first stacked."""
        Xb = self._X @ beta
        if not self._count_spatial:
            return Xb
        N, T = self._N, self._T
        A = (sp.eye(N, format="csc") - rho * self._W_sparse).tocsc()
        out = sp.linalg.splu(A).solve(np.ascontiguousarray(Xb.reshape(T, N).T))
        return out.T.reshape(-1)

    def posterior_predictive(
        self,
        n_draws: Optional[int] = None,
        random_seed: Optional[int] = None,
    ) -> np.ndarray:
        """Posterior-predictive counts, shape ``(n_draws, N·T)``, time-first."""
        self._require_fit()
        post = self._idata.posterior
        beta = post["beta"].values.reshape(-1, len(self._feature_names))
        total = beta.shape[0] if n_draws is None else min(int(n_draws), beta.shape[0])
        rho = post["rho"].values.reshape(-1) if self._count_spatial else None
        alpha = post["alpha"].values.reshape(-1)
        offsets = self._fe_offset_draws(total)
        rng = np.random.default_rng(random_seed)
        out = np.empty((total, self._N * self._T))
        for g in range(total):
            eta = self._eta_reduced(None if rho is None else float(rho[g]), beta[g])
            if offsets is not None:
                eta = eta + offsets[g]
            out[g] = self._count_draw(rng, eta, float(alpha[g]))
        return out

    def _fitted_mean_from_posterior(self) -> np.ndarray:
        """Expected counts ``exp(η)`` at the posterior means."""
        self._require_fit()
        post = self._idata.posterior
        rho = float(self._posterior_mean("rho")) if self._count_spatial else None
        eta = self._eta_reduced(rho, self._posterior_mean("beta"))
        if self._fe_groups:
            if "group_effect" in post.data_vars:
                c = self._posterior_mean("group_effect")
            else:
                c = self._idata["group_effect_summary"]["mean"].values
            eta = eta + np.asarray(c)[self._row_group]
        if self._fe_periods:
            eta = eta + self._D_tau @ self._posterior_mean("time_effect")
        return np.exp(eta)

    def _compute_spatial_effects_posterior(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Impacts on the log-mean scale for each draw."""
        if self._count_spatial:
            return self._sar_effects()
        return self._sem_effects()


def _run_count_panel(
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
    """Registry runner for the NB count panels: a thin adapter over ``_fit_gibbs``.

    NumPy-only, so ``backend`` is always ``"numpy"``.
    """
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
    "count",
    "panel",
    run=_run_count_panel,
    backends={"numpy"},
    options={"store_group_effects"},
    skips_log_likelihood=True,
)


class SARNegBinPanel(_CountPanelBase):
    __doc__ = (
        r"""Spatial-lag NB2 count panel with pooled unit effects and period effects.

    .. math::

        y_{it} \sim \mathrm{NB2}(e^{\eta_{it}}, \alpha), \quad
        \eta_t = (I - \rho W)^{-1} X_t \beta + c + \tau_t

    Sampled by reduced-form Pólya–Gamma Gibbs (``sampler="nuts"`` for PyMC).
    """
        + _PARAMS_DOC
    )
    _count_spatial = True


class NegBinPanel(_CountPanelBase):
    __doc__ = (
        r"""Aspatial NB2 count panel with pooled unit effects and period effects.

    Pólya–Gamma Gibbs with the unit effects integrated out (NUTS cannot carry
    one parameter per unit at scale); ``sampler="nuts"`` for PyMC.
    """
        + _PARAMS_DOC
    )
    _count_spatial = False
