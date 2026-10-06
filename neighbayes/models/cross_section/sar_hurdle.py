r"""Reduced-form SAR hurdle negative binomial (cross-section).

.. math::

    P(y_i > 0) &= \mathrm{logit}^{-1}(\eta^{\mathrm{b}}_i), \qquad
    \eta^{\mathrm{b}} = (I - \lambda W_{\mathrm{sel}})^{-1} Z\gamma, \\
    y_i \mid y_i > 0 &\sim \mathrm{NB2}(e^{\eta^{\mathrm{c}}_i}, \alpha)
    \text{ truncated at } 0, \qquad
    \eta^{\mathrm{c}} = (I - \rho W)^{-1} X\beta .

A hurdle separates *whether* a count is positive from *how large* it is when
it is.  Unlike a zero-inflated model it has no latent allocation: the binary
half is a reduced-form spatial logit on the observed ``d = 1(y > 0)`` and the
count half a zero-truncated reduced-form spatial NB on the positive counts, so
both halves are identified from observed data even when the counts are sparse.
Both halves are mean-propagating (the spatial lag acts on the linear predictor,
with no latent noise field), so no Jacobian enters either.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import scipy.sparse as sp

from ..._lazy_deps import xr
from ...samplers import hurdle as _hurdle  # noqa: F401  (registers the runners)
from .._base._nb import nb_alpha_rv
from .._base._shared import _parse_W
from .._mixins._hurdle import HurdleMixin
from ..base import SpatialModel
from ..priors import SARHurdlePriors
from .sar_zinb import _selection_design


class SARHurdleNB(HurdleMixin, SpatialModel):
    """Bayesian reduced-form SAR hurdle NB2 with a Pólya–Gamma Gibbs sampler.

    Parameters
    ----------
    formula : str, optional
        Count-half formula, e.g. ``"y ~ x1 + x2"``. Requires ``data``.
    data : pandas.DataFrame or geopandas.GeoDataFrame, optional
        Data source for formula mode.
    y : array-like, optional
        Non-negative integer counts of shape ``(n,)`` (matrix mode).
    X : array-like, optional
        Count-half design of shape ``(n, k)`` (matrix mode).
    Z : array-like or pandas.DataFrame, optional
        Binary-half design of shape ``(n, p)``; a DataFrame's column names
        label γ.  Default: ``X``.
    sel_formula : str, optional
        Binary-half right-hand side, e.g. ``"~ z1 + z2"`` (formula mode).
    W : libpysal.graph.Graph or scipy.sparse matrix
        Count-half weights, ``n × n``.
    W_sel : libpysal.graph.Graph or scipy.sparse matrix, optional
        Binary-half weights. Default: ``W``.
    priors : dict, optional
        ``beta_mu``/``beta_sigma`` (default Gelman et al. 2008 on the log
        scale, the intercept at ``log mean(y | y > 0)``), ``gamma_mu``/
        ``gamma_sigma`` (logit scale, the intercept at ``logit(mean(y > 0))``),
        ``rho_lower``/``rho_upper``, ``lam_lower``/``lam_upper``,
        ``alpha_sigma``/``alpha_nu`` and ``alpha_fixed``.  See
        :class:`~neighbayes.models.priors.SARHurdlePriors`.
    logdet_method : str, optional
        Accepted for API symmetry; neither half has a Jacobian.

    Notes
    -----
    ``fit()`` runs the Gibbs sampler of :mod:`neighbayes.samplers.hurdle._generic`:
    a Pólya–Gamma logit for the binary half; for the count half a Pólya–Gamma
    NB with the truncation augmented exactly (geometric counts of the zeros it
    hid), plus a joint slice on the count level and ``log α``, which otherwise
    trade off along a ridge when the positive counts are mostly ones.
    ``sampler="nuts"`` fits the same model and priors in PyMC
    (``HurdleNegativeBinomial``).

    The binary equation is called ``"selection"`` in
    :meth:`spatial_effects` (alias ``"hurdle"``), with parameters ``gamma`` and
    ``lam``; :meth:`corridor_probabilities` gives ``P(y > 0)``,
    :meth:`conditional_mean` ``E[y | y > 0]``.
    """

    _spatial_params: tuple[str, ...] = ("rho", "lam")
    _lag_terms: tuple[str, ...] = ()
    _jacobian_param: str | None = None
    _gibbs_class: str | None = None
    _model_type: str = "hurdle_sar"
    _likelihood: str = "count"
    _gibbs_key: tuple[str, str] | None = ("hurdle", "cross_section")
    _priors_cls = SARHurdlePriors

    def __init__(
        self,
        formula=None,
        data=None,
        y=None,
        X=None,
        Z=None,
        W=None,
        sel_formula=None,
        W_sel=None,
        priors=None,
        logdet_method=None,
        robust=False,
        **kwargs,
    ):
        if robust:
            raise NotImplementedError("robust=True is not supported for SARHurdleNB.")
        super().__init__(
            formula=formula,
            data=data,
            y=y,
            X=X,
            W=W,
            priors=priors,
            logdet_method=logdet_method,
            robust=False,
            **kwargs,
        )
        if np.any(self._y < 0) or not np.allclose(self._y, np.round(self._y)):
            raise ValueError("y must be non-negative integer counts for SARHurdleNB.")
        if not np.any(self._y > 0):
            raise ValueError("SARHurdleNB needs at least one positive count.")
        self._y_int = np.round(self._y).astype(np.int64)
        self._Z, self._sel_feature_names = _selection_design(
            Z, sel_formula, data, len(self._y)
        )
        if self._Z is None:
            self._Z = self._X.copy()
            self._sel_feature_names = list(self._feature_names)
        if W_sel is not None:
            self._W_sel_sparse, self._is_sel_row_std = _parse_W(W_sel, len(self._y))
            self._same_W = False
        else:
            self._W_sel_sparse = self._W_sparse
            self._is_sel_row_std = self._is_row_std
            self._same_W = True

    def _bounds(self, prefix: str) -> tuple[float, float]:
        b = self._logdet_bounds
        lo = float(self.priors.get(f"{prefix}_lower", b.rho_min))
        hi = float(self.priors.get(f"{prefix}_upper", b.rho_max))
        return max(lo, b.rho_min), min(hi, b.rho_max)

    def _model_coords(self, extra: Optional[dict] = None) -> dict:
        coords = super()._model_coords(extra)
        coords["sel_coefficient"] = list(self._sel_feature_names)
        return coords

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def _fit_gibbs(
        self,
        *,
        draws: int = 2000,
        tune: int = 1000,
        chains: int = 4,
        random_seed: Optional[int] = None,
        thin: int = 1,
        n_jobs: int = -1,
        progressbar: bool = True,
        log_likelihood: bool = False,
    ) -> xr.DataTree:
        """Sample via the generic hurdle kernel (registry ``("hurdle", "cross_section")``)."""
        from ...samplers._utils._idata import gibbs_to_inference_data
        from ...samplers._utils._seeds import spawn_chain_seeds
        from ...samplers.count_panel import CountPanelFE
        from ...samplers.count_panel._filters import SARFilter
        from ...samplers.gaussian._chain_runner import run_chains
        from ...samplers.hurdle._generic import HurdlePriors, run_chain

        beta_mu, beta_sigma = self._hurdle_beta_prior(self._X, self._feature_names)
        gamma_mu, gamma_sigma = self._hurdle_gamma_prior(
            self._Z, self._sel_feature_names
        )
        a_sigma, a_nu, a_fixed = self._hurdle_alpha()
        priors = HurdlePriors(
            gamma_mu=gamma_mu,
            gamma_sigma=gamma_sigma,
            theta_mu=beta_mu,
            theta_sigma=beta_sigma,
            alpha_sigma=a_sigma,
            alpha_nu=a_nu,
            alpha_fixed=a_fixed,
        )
        filt_cnt = SARFilter(self._W_sparse, 1, *self._bounds("rho"))
        filt_bin = SARFilter(self._W_sel_sparse, 1, *self._bounds("lam"))
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
                CountPanelFE(),
                CountPanelFE(),
                priors,
                draws,
                tune,
                thin=thin,
                rng=np.random.default_rng(seed),
                chain_id=chain_id,
                progress_manager=progress_manager,
                store_log_lik=log_likelihood,
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
        stack = lambda key: np.stack([r[key] for r in results], axis=0)  # noqa: E731
        post = {name: stack(name) for name in ("lam", "gamma", "rho", "beta", "alpha")}
        idata = gibbs_to_inference_data(
            posterior_samples=post,
            log_likelihood={"obs": stack("log_lik")} if log_likelihood else None,
            observed_data={"obs": self._y_int},
            coords={
                "sel_coefficient": list(self._sel_feature_names),
                "coefficient": list(self._feature_names),
                "obs_id": list(range(len(self._y))),
            },
            dims={
                "gamma": ["sel_coefficient"],
                "beta": ["coefficient"],
                "obs": ["obs_id"],
            },
        )
        self._idata = idata
        return idata

    def _build_pymc_model(self):
        r"""The same reduced-form hurdle and priors as the Gibbs path, in PyMC.

        ``y ~ HurdleNB(ψ = π, μ = exp(η_c), α)``, where PyMC's ``ψ`` is
        ``P(y > 0)``.  No ``log|I − ρW|`` terms: neither half transforms ``y``.
        """
        import pytensor.tensor as pt

        from ..._lazy_deps import pm
        from ..._ops import SparseSARSolveOp

        beta_mu, beta_sigma = self._hurdle_beta_prior(self._X, self._feature_names)
        gamma_mu, gamma_sigma = self._hurdle_gamma_prior(
            self._Z, self._sel_feature_names
        )
        a_sigma, a_nu, a_fixed = self._hurdle_alpha()
        with pm.Model(coords=self._model_coords()) as model:
            lam = pm.Uniform("lam", *self._bounds("lam"))
            gamma = pm.Normal(
                "gamma", mu=gamma_mu, sigma=gamma_sigma, dims="sel_coefficient"
            )
            rho = pm.Uniform("rho", *self._bounds("rho"))
            beta = pm.Normal("beta", mu=beta_mu, sigma=beta_sigma, dims="coefficient")
            alpha = nb_alpha_rv(a_fixed, a_nu, a_sigma)
            eta_b = SparseSARSolveOp(self._W_sel_sparse)(
                lam, pt.dot(pt.as_tensor_variable(self._Z), gamma)
            )
            eta_c = SparseSARSolveOp(self._W_sparse)(
                rho, pt.dot(pt.as_tensor_variable(self._X), beta)
            )
            pm.HurdleNegativeBinomial(
                "obs",
                psi=pm.math.sigmoid(eta_b),
                mu=pt.exp(eta_c),
                alpha=alpha,
                observed=self._y_int,
            )
        return model

    def _part_etas(self, g: int) -> tuple[np.ndarray, np.ndarray]:
        """``(η_b, η_c)`` at posterior draw ``g``."""
        n = self._X.shape[0]
        lam = float(self._flat_draw("lam")[g])
        rho = float(self._flat_draw("rho")[g])
        eye = sp.eye(n, format="csc", dtype=np.float64)
        eta_b = sp.linalg.splu((eye - lam * self._W_sel_sparse).tocsc()).solve(
            self._Z @ self._flat_draw("gamma")[g]
        )
        eta_c = sp.linalg.splu((eye - rho * self._W_sparse).tocsc()).solve(
            self._X @ self._flat_draw("beta")[g]
        )
        return eta_b, eta_c
