r"""Shared methods for cross-sectional and panel flow models.

Centralises the verbatim-duplicate methods that were previously copy-pasted
between :class:`~neighbayes.models.flow.FlowModel` and
:class:`~neighbayes.models.flow_panel.FlowPanelModel`:

* ``fit`` — NUTS sampling dispatch
* ``_posterior_var_names`` — filter free_RVs / deterministics
* ``spatial_diagnostics_decision`` / ``_get_decision_spec`` — decision tree
* ``_attach_complete_log_likelihood`` — Jacobian-corrected log-likelihood
* ``_attach_flow_log_abs_det`` — diagnostic log-determinant in sample_stats
* ``_assemble_A`` / ``_A_solver`` / ``_solve_A`` — sparse flow filter solves
* ``_compute_jacobian_log_det`` — default ``None`` (subclasses override)
* ``_fitted_mean_from_posterior`` — default ``NotImplementedError``

The only cross-section vs panel difference is:

1. **System size**: cross-section uses ``self._N`` (= n²), panel uses
   ``self._N_flow`` (= n²).  The ``_flow_system_size`` property unifies these.
2. **Decision-tree spec lookup**: cross-section uses ``get_flow_spec``,
   panel uses ``get_panel_flow_spec``.  The ``_flow_spec_fn`` property
   returns the appropriate callable.
"""

from __future__ import annotations

from functools import cached_property
from typing import Any, Optional

import numpy as np
import scipy.sparse as sp

from ..._backends.sampler_helpers import (
    enforce_c_backend,
    jax_available,
    prepare_compile_kwargs,
    prepare_idata_kwargs,
)
from ..._lazy_deps import pm, xr


class FlowSharedMethods:
    """Mixin with shared methods for flow and flow-panel models.

    Both :class:`FlowModel` and :class:`FlowPanelModel` inherit from this
    mixin.  Subclasses are expected to set ``self._W_sparse`` (the ``n × n``
    weights; the ``N × N`` flow weights are derived lazily) and either
    ``self._N`` (cross-section)
    or ``self._N_flow`` (panel) before calling any mixin method.
    """

    def _resolve_gibbs_backend(self, requested: str, *, jax: bool) -> str:
        """``gibbs_backend`` resolved for this flow configuration.

        ``jax`` says whether the configuration has a JAX Gibbs kernel.
        ``"auto"`` takes it when JAX is installed and NumPy otherwise; an
        explicit ``"jax"`` without one raises.
        """
        from ...samplers._registry import resolve_backend_for

        backends = {"jax", "numpy"} if jax else {"numpy"}
        return resolve_backend_for(requested, backends, jax_ok=jax_available())

    # Subclasses override these as needed.
    _panel_diagnostics: bool = False

    # ------------------------------------------------------------------
    # Properties that abstract cross-section vs panel differences
    # ------------------------------------------------------------------

    @cached_property
    def _flow_kron_weights(self) -> dict:
        """The ``N × N`` flow weights ``I⊗W``, ``W⊗I``, ``W⊗W``, built on first use.

        ``W⊗W`` holds ``nnz(W)²`` entries, so the constructors never build it:
        lags use :func:`~neighbayes.graph.flow_lags` and effects use W-only
        moments.  Only the samplers and PyMC ops that factor the ``N × N``
        system ask for these.
        """
        from ...graph import flow_weight_matrices

        return flow_weight_matrices(self._W_sparse)

    @property
    def _Wd(self) -> sp.csr_matrix:
        return self._flow_kron_weights["destination"]

    @property
    def _Wo(self) -> sp.csr_matrix:
        return self._flow_kron_weights["origin"]

    @property
    def _Ww(self) -> sp.csr_matrix:
        return self._flow_kron_weights["network"]

    @cached_property
    def _flow_effect_moments(self):
        """W-only moments for the exact LeSage effects breakdown.

        Built on first use, only when spatial effects are computed; see
        :class:`~neighbayes.models.flow._flow._FlowEffectMoments`.
        """
        from ..flow._flow import _FlowEffectMoments

        return _FlowEffectMoments(self._W_sparse)

    def _flow_effects_for_draws(self, rho_d, rho_o, rho_w, beta_d, beta_o, beta_intra):
        """LeSage effects for every posterior draw, with no n²-sized arrays."""
        from ..flow._flow import _compute_flow_effects, _flow_effect_sums

        sums = _flow_effect_sums(self._flow_effect_moments, rho_d, rho_o, rho_w)
        return _compute_flow_effects(sums, beta_d, beta_o, beta_intra)

    @property
    def _flow_system_size(self) -> int:
        """Return the flow system dimension n².

        Cross-section models store this as ``self._N``; panel models
        store it as ``self._N_flow``.
        """
        return getattr(self, "_N_flow", getattr(self, "_N", 0))

    @property
    def _flow_spec_fn(self):
        """Return the decision-tree spec lookup function for this model."""
        from ...diagnostics import _decision_trees as _dt

        if self._panel_diagnostics:
            return _dt.get_panel_flow_spec
        return _dt.get_flow_spec

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def _posterior_var_names(
        self,
        model: "pm.Model",
        *,
        store_lambda: bool,
    ) -> list[str]:
        names = [rv.name for rv in model.free_RVs]
        names.extend(
            var.name
            for var in model.deterministics
            if store_lambda or var.name != "lambda"
        )
        return list(dict.fromkeys(name for name in names if name is not None))

    def fit(
        self,
        draws: int = 2000,
        tune: int = 1000,
        chains: int = 4,
        random_seed: Optional[int] = None,
        store_lambda: bool = False,
        idata_kwargs: Optional[dict] = None,
        progressbar: bool = True,
        **sample_kwargs,
    ) -> "xr.DataTree":
        """Draw samples from the posterior via PyMC NUTS.

        Parameters
        ----------
        draws : int, default 2000
            Number of posterior samples per chain (after tuning).
        tune : int, default 1000
            Number of tuning (warm-up) steps per chain.
        chains : int, default 4
            Number of parallel chains.
        random_seed : int, optional
            Seed for reproducibility.
        store_lambda : bool, default False
            If True, include the high-dimensional fitted mean ``lambda`` in the
            stored posterior. Leaving this False reduces memory and conversion
            overhead for NB flow models.
        idata_kwargs : dict, optional
            Forwarded to ``pm.sample``.  ``{"log_likelihood": True}`` stores
            the pointwise log-likelihood that ``az.loo`` /
            ``az.compare`` need; for SAR flow variants the captured Gaussian
            log-likelihood is post-processed to add the Jacobian contribution
            from ``log|I_N - rho_d W_d - rho_o W_o - rho_w W_w|``.  Off by
            default, as in PyMC: it holds one value per draw, chain, and flow.
        progressbar : bool, default True
            Show progress bar during sampling.
        **sample_kwargs
            Additional keyword arguments forwarded to ``pm.sample``.
            Pass ``target_accept=0.95`` to adjust the NUTS acceptance rate.

        Returns
        -------
        xarray.DataTree
        """
        idata_kwargs = dict(idata_kwargs) if idata_kwargs else {}
        compute_log_likelihood = bool(idata_kwargs.get("log_likelihood", False))
        nuts_sampler = sample_kwargs.pop("nuts_sampler", "pymc")
        target_accept = sample_kwargs.pop("target_accept", 0.9)
        nuts_sampler = enforce_c_backend(
            nuts_sampler,
            requires_c_backend=getattr(self, "_requires_c_backend", False),
            model_name=type(self).__name__,
        )

        model = self._build_pymc_model()
        self._pymc_model = model
        if "var_names" not in sample_kwargs and not store_lambda:
            sample_kwargs["var_names"] = self._posterior_var_names(
                model,
                store_lambda=False,
            )
        idata_kwargs = prepare_idata_kwargs(idata_kwargs, model, nuts_sampler)
        sample_kwargs = prepare_compile_kwargs(sample_kwargs, nuts_sampler)
        with model:
            self._idata = pm.sample(
                draws=draws,
                tune=tune,
                chains=chains,
                target_accept=target_accept,
                random_seed=random_seed,
                idata_kwargs=idata_kwargs,
                progressbar=progressbar,
                nuts_sampler=nuts_sampler,
                **sample_kwargs,
            )
        if compute_log_likelihood:
            self._attach_complete_log_likelihood(self._idata)
        return self._idata

    def spatial_diagnostics_decision(
        self, alpha: float = 0.05, format: str = "graphviz"
    ) -> Any:
        """Return a model-selection decision from Bayesian LM test results.

        Walks the flow decision tree using Bayesian p-values from
        :meth:`spatial_diagnostics` and recommends either the OLS flow
        baseline (no spatial dependence detected) or the SAR flow model
        (at least one direction is significant).

        Parameters
        ----------
        alpha : float, default 0.05
            Significance level for the Bayesian p-values.
        format : {"graphviz", "ascii", "model"}, default "graphviz"
            Output format.  ``"model"`` returns the recommended model name
            string.  ``"ascii"`` returns an indented box-drawing tree.
            ``"graphviz"`` returns a :class:`graphviz.Digraph` (with ASCII
            fallback if graphviz is not installed).

        Returns
        -------
        str or graphviz.Digraph
        """
        from ...diagnostics import _decision_trees as _dt

        diag = self.spatial_diagnostics()
        model_type = self.__class__.__name__

        def _sig(test_name: str) -> bool:
            if test_name not in diag.index:
                return False
            pval = diag.loc[test_name, "p_value"]
            return not np.isnan(pval) and pval < alpha

        spec = self._flow_spec_fn(model_type)
        decision, path = _dt.evaluate(spec, sig_lookup=_sig)

        p_values: dict[str, float] = {}
        for test_name in diag.index:
            pv = diag.loc[test_name, "p_value"]
            if not np.isnan(pv):
                p_values[str(test_name)] = float(pv)

        return _dt.render(
            spec,
            path,
            decision,
            p_values=p_values,
            alpha=alpha,
            fmt=format,
            title=f"{model_type} decision tree (alpha={alpha})",
        )

    def _get_decision_spec(self, model_type: str):
        """Return the flow decision-tree spec for this model type."""
        return self._flow_spec_fn(model_type)

    # ------------------------------------------------------------------
    # Pointwise log-likelihood (with Jacobian correction for SAR variants)
    # ------------------------------------------------------------------

    def _flow_gaussian_priors(self) -> dict:
        """Resolved priors for the Gaussian flow models, shared by NUTS and Gibbs.

        ``beta`` gets the Gelman et al. (2008) default on the sampled design
        (scaled to ``sd(y)`` and each column's sd) and ``σ²`` gets
        ``IG(2, Var y)``, the same defaults as the cross-section and panel
        Gaussian models, so both mean the same thing in any units of ``y``.
        """
        if "sigma_sigma" in self.priors:
            raise ValueError(
                "'sigma_sigma' set the old HalfNormal prior on sigma; the Gaussian "
                "flow models now place IG(sigma2_alpha, sigma2_beta) on sigma**2 "
                "(default IG(2, Var y))."
            )
        k = self._X.shape[1]
        names = list(self._feature_names) or [f"x{j}" for j in range(k)]
        mu, sd = self._gelman_default_beta_prior(self._X, names)
        p = self.priors
        return {
            "beta_mu": np.broadcast_to(
                np.asarray(p.get("beta_mu", mu), dtype=np.float64), (k,)
            ).copy(),
            "beta_sigma": np.broadcast_to(
                np.asarray(p.get("beta_sigma", sd), dtype=np.float64), (k,)
            ).copy(),
            "sigma2_alpha": float(p.get("sigma2_alpha", 2.0)),
            "sigma2_beta": float(p.get("sigma2_beta", np.var(self._y))),
        }

    def _flow_count_priors(self) -> dict:
        """Resolved priors for the count flow models, shared by NUTS and Gibbs.

        ``beta`` gets the Gelman et al. (2008) default on the log scale: the
        intercept is centred on ``log(mean(y))`` with scale 2.5 and each slope
        has scale ``2.5 / sd(x_j)``.  The NB2 dispersion ``alpha`` gets a
        half-t(``alpha_nu``, ``alpha_sigma``), default half-t(3, 2.5), unless
        ``alpha_fixed`` holds it (see :mod:`neighbayes.models._base._nb`).
        """
        from .._base._nb import nb_alpha_fixed

        k = self._X.shape[1]
        names = list(self._feature_names) or [f"x{j}" for j in range(k)]
        mu, sd = self._resolved_beta_prior(self._X, names, link="log")
        return {
            "beta_mu": mu,
            "beta_sigma": sd,
            "alpha_sigma": float(self.priors.get("alpha_sigma", 2.5)),
            "alpha_nu": float(self.priors.get("alpha_nu", 3.0)),
            "alpha_fixed": nb_alpha_fixed(self.priors),
        }

    def _flow_sigma(self, pv: dict):
        """``σ² ~ IG`` with ``σ`` recorded as a deterministic (inside a model)."""
        import pytensor.tensor as pt

        sigma2 = pm.InverseGamma(
            "sigma2", alpha=pv["sigma2_alpha"], beta=pv["sigma2_beta"]
        )
        return pm.Deterministic("sigma", pt.sqrt(sigma2))

    def _compute_jacobian_log_det(self, posterior) -> Optional[np.ndarray]:
        """Per-draw log-determinant of the flow filter matrix.

        Returns ``None`` (the default) when no Jacobian correction is
        required — for OLS / NB flow baselines (``A = I_N``) and the
        NB SAR variants (the ``pm.NegativeBinomial("obs", ...)`` log-likelihood
        already captured by PyMC is the appropriate pointwise density on
        observed counts).

        Subclasses with a Gaussian observation model and a
        ``pm.Potential("jacobian", ...)`` term must override this to return
        an array of shape ``(n_draws,)`` with the per-draw log-determinant.
        """
        return None

    def _attach_complete_log_likelihood(self, idata) -> None:
        """Add Jacobian contribution to the pointwise log-likelihood.

        ``pm.sample(idata_kwargs={"log_likelihood": True})`` only captures
        observed-RV log densities, so the ``pm.Potential("jacobian", ...)``
        contribution from ``log|I_N - rho_d W_d - rho_o W_o - rho_w W_w|``
        is added post-hoc to the stored log-likelihood so that
        ``az.loo`` / ``az.compare`` operate on the full
        joint log-likelihood.
        """
        if idata is None or not hasattr(idata, "log_likelihood"):
            return
        if "obs" not in idata.log_likelihood.data_vars:
            return

        jacobian_draws = self._compute_jacobian_log_det(idata.posterior)
        if jacobian_draws is None:
            return

        import xarray as xr

        ll_da = idata.log_likelihood["obs"]
        n_chains = ll_da.sizes["chain"]
        n_draws_per_chain = ll_da.sizes["draw"]
        n_obs = int(np.prod(ll_da.shape[2:]))

        ll_array = ll_da.values.reshape(n_chains * n_draws_per_chain, n_obs)
        jacobian_draws = np.asarray(jacobian_draws, dtype=np.float64).reshape(-1)
        if jacobian_draws.shape[0] != ll_array.shape[0]:
            raise RuntimeError(
                "Posterior draw count does not match log-likelihood shape: "
                f"{jacobian_draws.shape[0]} vs {ll_array.shape[0]}."
            )

        ll_array = ll_array + jacobian_draws[:, None] / n_obs
        ll_array = ll_array.reshape(n_chains, n_draws_per_chain, n_obs)

        new_da = xr.DataArray(ll_array, dims=("chain", "draw", "obs_dim"), name="obs")
        idata["log_likelihood"] = xr.Dataset({"obs": new_da})

    def _attach_flow_log_abs_det(self, idata, *, n_probes: int = 16, n_quad: int = 6):
        """Record per-draw ``T·log|A(ρ)|`` in ``idata.sample_stats`` as a diagnostic.

        Used by the count (NB) flow models: the discrete likelihood carries no
        ``|A|`` change-of-variables term (so it must not enter the LOO
        ``log_likelihood``), but the spatial-filter log-determinant is still
        exposed for inspection — computed with the scalable resolvent value
        estimator and scaled by the panel length ``T``.
        """
        from ...samplers.gaussian._flow_resolvent import attach_flow_log_abs_det

        attach_flow_log_abs_det(
            idata,
            self._W_sparse,
            T=int(getattr(self, "_T", 1)),
            n_probes=n_probes,
            n_quad=n_quad,
        )
        return idata

    # ------------------------------------------------------------------
    # Internal helpers — sparse flow filter A = I - ρ_d W_d - ρ_o W_o - ρ_w W_w
    # ------------------------------------------------------------------

    def _assemble_A(
        self,
        rho_d: float,
        rho_o: float,
        rho_w: float,
    ) -> sp.csr_matrix:
        """Assemble ``A = I - ρ_d W_d - ρ_o W_o - ρ_w W_w`` (sparse n²×n²)."""
        I_N = sp.eye(self._flow_system_size, format="csr", dtype=np.float64)
        return I_N - rho_d * self._Wd - rho_o * self._Wo - rho_w * self._Ww

    @property
    def _A_solver(self):
        """Lazily-built :class:`CachedSparseSolver` over ``[Wd, Wo, Ww]``.

        ``A = I - ρ_d W_d - ρ_o W_o - ρ_w W_w`` has a fixed sparsity pattern
        — only the three ρ values rescale per draw.  sparsax (when installed)
        caches the fill-reducing symbolic analysis keyed on the merged COO
        pattern, so repeated solves across posterior draws / posterior
        predictive / LeSage effects pay the symbolic cost once.  Built once
        per model instance and reused across methods.
        """
        cached = getattr(self, "_cached_A_solver", None)
        if cached is None:
            from ...samplers._utils._sparsax_utils import CachedSparseSolver

            cached = CachedSparseSolver(
                [self._Wd, self._Wo, self._Ww], self._flow_system_size
            )
            self._cached_A_solver = cached
        return cached

    def _solve_A(self, rho_d, rho_o, rho_w, rhs):
        """Solve ``A(ρ_d, ρ_o, ρ_w) x = rhs`` using the cached symbolic analysis."""
        return self._A_solver.solve([-rho_d, -rho_o, -rho_w], rhs)

    def _fitted_mean_from_posterior(self) -> np.ndarray:
        """Posterior-mean fitted values on transformed scale.

        Flow models override this in subclasses when fitted values are needed.
        The base implementation raises ``NotImplementedError``.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement fitted_values()."
        )
