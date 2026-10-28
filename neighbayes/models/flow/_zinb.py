r"""The zero-inflated negative binomial SAR flow model (separable).

.. math::

    \eta_{z,t} = (L_{z,o} \otimes L_{z,d})^{-1} X_{z,t}\beta_z, \qquad
    \eta^{\mathrm{cnt}}_t = (L_o^{\rho} \otimes L_d^{\rho})^{-1} X_t\beta
        + \tau_t + C,

a structural zero with probability ``1 - logit⁻¹(η_z)`` and otherwise an NB2
count with log-mean ``η^cnt``; see :class:`SARZINBFlow`.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pytensor.tensor as pt
import scipy.sparse as sp

from ..._backends.sampler_helpers import select_backend
from ..._lazy_deps import pm, xr
from ..._ops import kron_solve_matrix
from .._base._nb import nb_alpha_rv
from .._mixins._designs import FlowCountDesigns
from .._mixins._zinb import ZINBMixin
from ._count import _FlowCount
from ._flow import FlowModel


class SARZINBFlow(FlowCountDesigns, ZINBMixin, _FlowCount, FlowModel):
    r"""Zero-inflated negative binomial SAR flow model (separable).

    .. math::

        \eta_{z,t} &= (L_{z,o} \otimes L_{z,d})^{-1} X_{z,t}\beta_z,
        \qquad z \sim \mathrm{Bernoulli}(\mathrm{logit}^{-1}(\eta_z)), \\
        \eta^{\mathrm{cnt}}_t &= (L_o^{\rho} \otimes L_d^{\rho})^{-1} X_t\beta
        + \tau_t + C, \qquad y \mid z = 1 \sim \mathrm{NB2}(e^{\eta^{\mathrm{cnt}}}, \alpha),

    with :math:`L_{z,k} = I - \rho_{z,k} W_z`,
    :math:`L_k^{\rho} = I - \rho_k W` and per-cell structural zeros
    (``y = 0`` when ``z = 0``), for each of ``T`` periods (``T=1`` the
    cross-section).  ``pair_effects="random"`` and ``time_effects=True`` set pair
    and period effects on the **count** part only.  Implemented as the separable
    flow only: pass ``separable=True``.  ``ρ_{z,d}``, ``ρ_{z,o}``, ``β_z`` are the posterior's
    ``rho_d_zero``, ``rho_o_zero``, ``beta_zero``.

    Sampling: ``fit()`` runs the structured ``n × n`` Pólya–Gamma sweep
    (:mod:`neighbayes.samplers.zinb._flow_structured`), on NumPy or
    ``backend="jax"``; ``sampler="nuts"`` fits the same model in PyMC.

    On sparse flows (active pairs averaging about one count) the zero part is
    weakly identified in a cross-section: only the shape of the count
    distribution separates structural from sampling zeros, and NB with a
    smaller α explains the zeros nearly as well.  Check ``rho_d_zero`` and
    ``rho_o_zero`` against their priors and the zero part's R-hat, or use a
    panel with pair effects, or :class:`SARHurdleNegBinFlow`.

    Parameters
    ----------
    y, X, W, T, pair_effects, time_effects, col_names, k, priors, shared_od_covariates
        As :class:`SARNegBinFlow` with ``separable=True``.
    X_zero : array-like or pandas.DataFrame, optional
        Zero-part design, time-first stacked ``(n²·T, p)``.  Default: ``X``
        (then the zero part's impacts are available).
    W_zero : libpysal.graph.Graph or scipy.sparse matrix, optional
        ``n × n`` zero-part weights.  Default: ``W``.
    priors : dict, optional
        Count keys as :class:`SARNegBinFlow` (incl.
        ``alpha_fixed``) plus ``beta_mu_zero``, ``beta_sigma_zero``,
        ``rho_lower_zero`` and ``rho_upper_zero`` for the zero part.
    """

    _DESIGNS = {"separable": {"_flow_filter": "separable"}}
    _zero_part = True
    _effects_equation = "count"

    def _model_coords(self, extra: Optional[dict] = None) -> dict:
        coords = super()._model_coords(extra)
        coords["coefficient_zero"] = list(self._sel_feature_names)
        return coords

    # ------------------------------------------------------------------
    # Priors
    # ------------------------------------------------------------------

    def _zinb_flow_priors(self):
        from types import SimpleNamespace

        from .._base._shared import gelman_default_beta_prior

        pv = self._flow_count_priors()
        p = self._Z.shape[1]
        g_mu, g_sd = gelman_default_beta_prior(
            np.full(self._Z.shape[0], 0.5),
            self._Z,
            list(self._sel_feature_names),
            link="logit",
        )
        gamma_mu = np.broadcast_to(
            np.asarray(self.priors.get("beta_mu_zero", g_mu), dtype=float), (p,)
        ).copy()
        gamma_sigma = np.broadcast_to(
            np.asarray(self.priors.get("beta_sigma_zero", g_sd), dtype=float), (p,)
        ).copy()
        return SimpleNamespace(
            beta_mu=pv["beta_mu"],
            beta_sigma=pv["beta_sigma"],
            gamma_mu=gamma_mu,
            gamma_sigma=gamma_sigma,
            alpha_sigma=pv["alpha_sigma"],
            alpha_nu=pv["alpha_nu"],
            alpha_fixed=pv["alpha_fixed"],
            rho_bounds=(
                float(self.priors.get("rho_lower", -0.999)),
                float(self.priors.get("rho_upper", 0.999)),
            ),
            lam_bounds=(
                float(self.priors.get("rho_lower_zero", -0.999)),
                float(self.priors.get("rho_upper_zero", 0.999)),
            ),
        )

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def fit(
        self,
        draws: int = 2000,
        tune: int = 1000,
        chains: int = 4,
        random_seed: Optional[int] = None,
        *,
        sampler: str = "gibbs",
        backend: str | None = None,
        progressbar: bool = True,
        n_jobs: int = -1,
        idata_kwargs: Optional[dict] = None,
        store_effects: Optional[bool] = None,
        **sample_kwargs,
    ) -> xr.DataTree:
        """Sample the posterior: structured Gibbs (default) or ``sampler="nuts"``.

        ``backend="auto"`` (default) takes JAX when it is installed, else
        NumPy; ``"numpy"`` or ``"jax"`` pins one, and NUTS takes ``"pymc"``
        (its default), ``"nutpie"``, ``"numpyro"`` or ``"blackjax"``.
        ``store_effects`` keeps every draw of the pair effects (default:
        when they fit in about 500 MB).  ``idata_kwargs={"log_likelihood":
        True}`` stores the marginal ZINB pointwise log-likelihood.
        """
        backend = select_backend(sampler, backend, sample_kwargs)
        if sampler == "nuts":
            return FlowModel.fit(
                self,
                draws=draws,
                tune=tune,
                chains=chains,
                random_seed=random_seed,
                progressbar=progressbar,
                idata_kwargs=idata_kwargs,
                backend=backend,
                **sample_kwargs,
            )
        if sample_kwargs:
            raise TypeError(f"Unexpected keyword arguments: {sorted(sample_kwargs)}")
        return self._fit_gibbs_zinb(
            backend=self._resolve_gibbs_backend(backend, jax=True),
            draws=draws,
            tune=tune,
            chains=chains,
            random_seed=random_seed,
            progressbar=progressbar,
            n_jobs=n_jobs,
            log_likelihood=bool((idata_kwargs or {}).get("log_likelihood", False)),
            store_effects=store_effects,
        )

    def _zinb_fe_args(self):
        fp = self._count_fe_priors()
        pair = self._pair_args(fp) if self._fe_groups else None
        periods = (
            (fp["time_effect_mu"], fp["time_effect_sigma"], self._n_tau)
            if self._fe_periods
            else None
        )
        return pair, periods

    def _fit_gibbs_zinb(
        self,
        *,
        backend: str,
        draws: int,
        tune: int,
        chains: int,
        random_seed: Optional[int],
        progressbar: bool,
        n_jobs: int,
        log_likelihood: bool,
        store_effects: Optional[bool],
    ) -> xr.DataTree:
        from ...samplers._utils._seeds import seed_sequence_to_int, spawn_chain_seeds
        from ...samplers.gaussian._chain_runner import run_chains
        from ...samplers.zinb._flow_structured import run_chain_zinb_flow_structured

        pr = self._zinb_flow_priors()
        pair, periods = self._zinb_fe_args()
        keep_groups = self._store_group_draws(store_effects, chains, draws)
        y = self._y_int_vec.astype(np.float64)
        X = np.asarray(self._X, dtype=np.float64)
        Z = np.asarray(self._Z, dtype=np.float64)
        W_csc, W_zero_csc = self._W_sparse.tocsc(), self._W_zero_sparse.tocsc()
        scalars = ["rho_d", "rho_o", "rho_w", "rho_d_zero", "rho_o_zero", "rho_w_zero"]
        extra = {"beta_zero": ("coefficient_zero", self._sel_feature_names)}

        if backend == "jax":
            from ...samplers.zinb._flow_structured_jax import (
                run_chains_jax_zinb_flow_structured,
            )

            int_seeds = [
                seed_sequence_to_int(s) for s in spawn_chain_seeds(random_seed, chains)
            ]
            results = run_chains_jax_zinb_flow_structured(
                y,
                X,
                Z,
                W_csc,
                W_zero_csc,
                self._n,
                self._T,
                pr,
                draws,
                tune,
                pair_effects=pair,
                period_effects=periods,
                keep_group_draws=keep_groups,
                jax_seeds=int_seeds,
                store_log_lik=log_likelihood,
                progressbar=progressbar,
            )
            return self._assemble_count_idata(
                results, scalars, keep_groups, log_likelihood, extra=extra
            )

        def _chain_fn(chain_id, seed, progress_manager=None, chain_id_kw=0):
            return run_chain_zinb_flow_structured(
                y,
                X,
                Z,
                W_csc,
                W_zero_csc,
                self._n,
                self._T,
                pr,
                draws,
                tune,
                pair_effects=pair,
                period_effects=periods,
                rho_bounds=pr.rho_bounds,
                lam_bounds=pr.lam_bounds,
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
        return self._assemble_count_idata(
            results, scalars, keep_groups, log_likelihood, extra=extra
        )

    def _build_pymc_model(self) -> pm.Model:
        """The same ZINB and priors as the Gibbs path, in PyMC."""
        from ..._ops import KroneckerFlowSolveMatrixOp

        pr = self._zinb_flow_priors()
        n, N, T = self._n, self._N, self._T
        X_t = pt.as_tensor_variable(self._X.astype(np.float64))
        Z_t = pt.as_tensor_variable(self._Z.astype(np.float64))
        with pm.Model(coords=self._model_coords()) as model:
            rho_d = pm.Uniform("rho_d", *pr.rho_bounds)
            rho_o = pm.Uniform("rho_o", *pr.rho_bounds)
            pm.Deterministic("rho_w", -rho_d * rho_o)
            lam_d = pm.Uniform("rho_d_zero", *pr.lam_bounds)
            lam_o = pm.Uniform("rho_o_zero", *pr.lam_bounds)
            pm.Deterministic("rho_w_zero", -lam_d * lam_o)
            beta = pm.Normal(
                "beta", mu=pr.beta_mu, sigma=pr.beta_sigma, dims="coefficient"
            )
            gamma = pm.Normal(
                "beta_zero",
                mu=pr.gamma_mu,
                sigma=pr.gamma_sigma,
                dims="coefficient_zero",
            )
            inner, outer = self._pymc_fe_terms()
            Xb = pt.reshape(pt.dot(X_t, beta) + inner, (T, N)).T
            Zg = pt.reshape(pt.dot(Z_t, gamma), (T, N)).T
            eta_cnt = pt.reshape(
                KroneckerFlowSolveMatrixOp(self._W_sparse, n)(rho_d, rho_o, Xb).T,
                (N * T,),
            )
            eta_sel = pt.reshape(
                KroneckerFlowSolveMatrixOp(self._W_zero_sparse, n)(lam_d, lam_o, Zg).T,
                (N * T,),
            )
            eta_cnt = eta_cnt + outer
            alpha = nb_alpha_rv(pr.alpha_fixed, pr.alpha_nu, pr.alpha_sigma)
            pm.ZeroInflatedNegativeBinomial(
                "obs",
                psi=pm.math.sigmoid(eta_sel),
                mu=pt.exp(eta_cnt),
                alpha=alpha,
                observed=self._y_int_vec,
            )
        return model

    # ------------------------------------------------------------------
    # Per-draw predictors and impacts
    # ------------------------------------------------------------------

    def _kron_eta(self, W, rd: float, ro: float, b: np.ndarray) -> np.ndarray:
        n, N, T = self._n, self._N, self._T
        I_n = sp.eye(n, format="csr", dtype=np.float64)
        Ld = (I_n - rd * W).tocsr()
        Lo = (I_n - ro * W).tocsr()
        return kron_solve_matrix(Lo, Ld, b.reshape(T, N).T, n).T.reshape(-1)

    def _part_etas(self, g: int) -> tuple[np.ndarray, np.ndarray]:
        """``(η_sel, η_cnt)`` at draw ``g`` (pair-effect posterior mean if summarized)."""
        f = self._flat_draw
        post = self._idata.posterior
        inner = 0.0
        if self._fe_groups:
            # The pair effects enter inside the count filter.
            if self._effect_name() in post.data_vars:
                c = f(self._effect_name())[g]
            else:
                c = self._idata[self._effect_name("_summary")]["mean"].values
            inner = np.asarray(c)[self._row_group]
        eta_sel = self._kron_eta(
            self._W_zero_sparse,
            float(f("rho_d_zero")[g]),
            float(f("rho_o_zero")[g]),
            self._Z @ f("beta_zero")[g],
        )
        eta_cnt = self._kron_eta(
            self._W_sparse,
            float(f("rho_d")[g]),
            float(f("rho_o")[g]),
            self._X @ f("beta")[g] + inner,
        )
        if self._fe_periods:
            eta_cnt = eta_cnt + self._D_tau @ f("time_effect")[g]
        return eta_sel, eta_cnt

    def _compute_impacts_posterior(self, draws: Optional[int] = None) -> dict:
        if self._idata is None:
            raise RuntimeError("Model has not been fit yet. Call fit() first.")
        post = self._idata.posterior
        if self._effects_equation == "zero":
            coef = post["beta_zero"].values.reshape(-1, len(self._sel_feature_names))
            return self._compute_flow_effects_kron(
                post["rho_d_zero"].values.reshape(-1),
                post["rho_o_zero"].values.reshape(-1),
                coef,
                draws=draws,
            )
        return self._compute_flow_effects_kron(
            post["rho_d"].values.reshape(-1),
            post["rho_o"].values.reshape(-1),
            self._beta_layout(post),
            draws=draws,
        )

    def impacts(
        self,
        equation: str = "count",
        draws: Optional[int] = None,
        return_posterior_samples: bool = False,
        ci: float = 0.95,
        mode: str = "auto",
    ):
        """Origin, destination, intraunit and network impacts for one part.

        ``equation="count"`` (default) on the count part's log-mean through
        ``ρ``; ``"zero"`` on the activation log-odds through ``rho_zero`` and
        ``beta_zero``, available when the zero part's design is the flow design
        (``X_zero=None``).
        Other arguments as :meth:`FlowModel.impacts`.
        """
        if equation not in ("count", "zero"):
            raise ValueError(f"equation must be 'count' or 'zero', got {equation!r}")
        if equation == "zero" and not self._sel_is_X:
            raise NotImplementedError(
                "Zero-part flow impacts need the flow design layout; they are "
                "available when the model is built with X_zero=None (X_zero = X)."
            )
        self._effects_equation = equation
        try:
            return FlowModel.impacts(
                self,
                draws=draws,
                return_posterior_samples=return_posterior_samples,
                ci=ci,
                mode=mode,
            )
        finally:
            self._effects_equation = "count"
