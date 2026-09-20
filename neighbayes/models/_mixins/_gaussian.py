"""Mixin providing a unified ``_build_pymc_model`` for Gaussian spatial models.

The mixin is driven by declarative attributes already present on the model
class (``_jacobian_param``, ``_has_wx_in_beta``, ``_spatial_params``).
Subclasses only need to set those attributes and inherit from both
``SpatialModel`` (or ``SpatialPanelModel``) and ``GaussianLikelihoodMixin``;
the mixin supplies the full ``_build_pymc_model`` implementation.

Three likelihood branches are dispatched based on ``_jacobian_param``:

* ``None``  (OLS, SLX):  ``mu = Z @ β``,  ``pm.Normal/StudentT``, no potential.
* ``"rho"`` (SAR, SDM):  ``mu = ρ·Wy + Z @ β``, as ``pm.Normal/StudentT`` plus
  ``pm.Potential("jacobian", logdet(ρ))``, or — when a pointwise
  log-likelihood is requested — as a ``pm.CustomDist`` with the Jacobian
  folded in.
* ``"lam"`` (SEM, SDEM):  spatially-filtered residual via ``pm.CustomDist``
  (JAX path, or when a pointwise log-likelihood is requested) or
  ``pm.Potential`` (default path), plus Jacobian.

Because ``pm.Potential`` terms are invisible to PyMC's log-likelihood
machinery (it iterates ``observed_RVs``), the Jacobian must live inside an
observed RV for the captured group to be complete.  Folding it in costs
nothing — ``logdet`` is evaluated once per logp call either way — but the
Potential form remains the default because it is the benchmarked-fast graph
for the C/Numba backend.

The ``_spatial_lag(X)`` hook abstracts the W@X product so the same mixin
works for both cross-section (``self._W_sparse @ X``) and panel
(``self._sparse_panel_lag(X)``) models.
"""

from __future__ import annotations

import numpy as np

from ..._backends.sampler_helpers import use_jax_likelihood
from ..._lazy_deps import pm
from ...samplers._registry import register


class GaussianLikelihoodMixin:
    """Mixin that provides ``_build_pymc_model`` for Gaussian spatial models.

    Expects the host class to also inherit from
    :class:`~neighbayes.models.base.SpatialModel` (or
    :class:`~neighbayes.models.panel_base.SpatialPanelModel`) and to define
    the following declarative attributes:

    * ``_jacobian_param``: ``None``, ``"rho"``, or ``"lam"``
    * ``_has_wx_in_beta``: ``bool``
    * ``_spatial_params``: ``tuple[str, ...]``
    * ``robust``: ``bool``
    * ``priors``: dict-like with prior hyper-parameters

    And to provide these methods (inherited from ``SpatialModel``):

    * ``_gelman_default_beta_prior(design, names)`` → ``(mu, sigma)``
    * ``_model_coords()`` → dict of coordinate arrays
    * ``_nu`` → fixed Student-t degrees of freedom (robust models)
    * ``_logdet_pytensor_fn``: pytensor log-determinant callable

    The ``_spatial_lag(X)`` method defaults to ``self._W_sparse @ X``
    (cross-section).  Panel models override it to use
    ``self._sparse_panel_lag(X)``.
    """

    # ------------------------------------------------------------------
    # Spatial lag hook
    # ------------------------------------------------------------------
    # ``_spatial_lag(X)`` is defined on ``SpatialModel`` (cross-section:
    # ``self._W_sparse @ X``) and overridden on ``SpatialPanelModel`` (panel:
    # ``self._sparse_panel_lag(X)`` which applies W ⊗ I_T).  The mixin calls
    # ``self._spatial_lag(X)`` so the correct implementation is dispatched
    # via MRO regardless of which base class the model inherits from.

    # ------------------------------------------------------------------
    # Design matrix
    # ------------------------------------------------------------------

    def _design_matrix(self) -> np.ndarray:
        """Return the effective design matrix for the Gaussian likelihood.

        For models with ``_has_wx_in_beta=True`` (SLX, SDM, SDEM), this
        stacks ``[X, WX]``.  Otherwise, returns ``X`` alone.
        """
        if self._has_wx_in_beta:
            return np.hstack([self._X, self._WX])
        return self._X

    def _design_names(self) -> list[str]:
        """Return feature names aligned with the design matrix columns."""
        if self._has_wx_in_beta:
            return list(self._feature_names) + [
                f"W*{name}" for name in self._wx_feature_names
            ]
        return list(self._feature_names)

    # ------------------------------------------------------------------
    # Prior extraction
    # ------------------------------------------------------------------

    def _gaussian_priors(self, Z: np.ndarray, names: list[str]):
        """Extract prior hyper-parameters with Gelman defaults.

        Returns
        -------
        dict
            Keys: ``beta_mu``, ``beta_sigma``, ``sigma2_alpha``, ``sigma2_beta``,
            plus any spatial-param priors (``rho_lower``, ``rho_upper``,
            ``lam_lower``, ``lam_upper``).
        """
        default_beta_mu, default_beta_sigma = self._gelman_default_beta_prior(Z, names)
        return {
            "beta_mu": self.priors.get("beta_mu", default_beta_mu),
            "beta_sigma": self.priors.get("beta_sigma", default_beta_sigma),
            "sigma2_alpha": self.priors.get("sigma2_alpha", 2.0),
            "sigma2_beta": self.priors.get("sigma2_beta", float(np.var(self._y))),
            "rho_lower": self.priors.get("rho_lower", -1.0),
            "rho_upper": self.priors.get("rho_upper", 1.0),
            "lam_lower": self.priors.get("lam_lower", -1.0),
            "lam_upper": self.priors.get("lam_upper", 1.0),
        }

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def _build_pymc_model(
        self,
        *,
        compute_log_likelihood: bool = False,
        nuts_sampler: str = "pymc",
    ) -> pm.Model:
        """Build the PyMC model for a Gaussian spatial model.

        Dispatches to the appropriate branch based on ``_jacobian_param``:

        * ``None``  → :meth:`_build_pymc_model_no_jacobian`
        * ``"rho"`` → :meth:`_build_pymc_model_rho`
        * ``"lam"`` → :meth:`_build_pymc_model_lam`

        Parameters
        ----------
        compute_log_likelihood : bool, default False
            When ``True``, SAR/SDM/SEM/SDEM register the likelihood as an
            observed ``pm.CustomDist`` with the Jacobian folded in, so PyMC
            captures a complete pointwise log-likelihood natively.  Ignored by
            OLS/SLX, which have no Jacobian and capture it natively anyway.
        nuts_sampler : str, default ``"pymc"``
            Passed through to SEM/SDEM models to select JAX vs Potential path.
            Ignored by OLS/SLX/SAR/SDM.
        """
        jacobian_param = self._jacobian_param
        if jacobian_param is None:
            # No Jacobian: the observed RV carries the whole likelihood.
            self._native_log_likelihood = True
            return self._build_pymc_model_no_jacobian()
        elif jacobian_param == "rho":
            return self._build_pymc_model_rho(
                compute_log_likelihood=compute_log_likelihood,
            )
        elif jacobian_param == "lam":
            return self._build_pymc_model_lam(
                nuts_sampler=nuts_sampler,
                compute_log_likelihood=compute_log_likelihood,
            )
        else:
            raise ValueError(
                f"Unknown _jacobian_param={jacobian_param!r}; "
                f"expected None, 'rho', or 'lam'."
            )

    # ------------------------------------------------------------------
    # WX validation
    # ------------------------------------------------------------------

    def _validate_wx_columns(self) -> None:
        """Raise ValueError if the model requires WX columns but has none.

        Models with ``_has_wx_in_beta=True`` (SLX, SDM, SDEM) require at
        least one WX column.  This check provides a clear error message
        rather than a cryptic shape mismatch later.
        """
        if self._has_wx_in_beta and not self._wx_column_indices:
            model_name = type(self).__name__
            if self._jacobian_param == "rho":
                alt = "SAR"
            elif self._jacobian_param == "lam":
                alt = "SEM"
            else:
                alt = "OLS"
            raise ValueError(
                f"{model_name} requires at least one WX column. "
                f"Pass `w_vars=[...]` to choose which regressors receive "
                f"a spatial lag, or fit an {alt} model instead."
            )

    # ------------------------------------------------------------------
    # Branch 1: No Jacobian (OLS, SLX)
    # ------------------------------------------------------------------

    def _build_pymc_model_no_jacobian(self) -> pm.Model:
        """Build PyMC model for OLS or SLX (no spatial autoregressive term)."""
        import pytensor.tensor as pt

        self._validate_wx_columns()
        Z = self._design_matrix()
        names = self._design_names()
        priors = self._gaussian_priors(Z, names)

        with pm.Model(coords=self._model_coords()) as model:
            beta = pm.Normal(
                "beta",
                mu=priors["beta_mu"],
                sigma=priors["beta_sigma"],
                dims="coefficient",
            )
            sigma2 = pm.InverseGamma(
                "sigma2",
                alpha=priors["sigma2_alpha"],
                beta=priors["sigma2_beta"],
            )
            sigma = pm.Deterministic("sigma", pt.sqrt(sigma2))
            mu = pt.dot(Z, beta)
            if self.robust:
                nu = self._nu
                pm.StudentT("obs", nu=nu, mu=mu, sigma=sigma, observed=self._y)
            else:
                pm.Normal("obs", mu=mu, sigma=sigma, observed=self._y)

        return model

    # ------------------------------------------------------------------
    # Branch 2: ρ Jacobian (SAR, SDM)
    # ------------------------------------------------------------------

    def _build_pymc_model_rho(
        self,
        *,
        compute_log_likelihood: bool = False,
    ) -> pm.Model:
        """Build PyMC model for SAR or SDM (spatial lag with ρ Jacobian).

        Two formulations of an identical total log-density:

        * ``compute_log_likelihood=False`` (default) — observed
          ``pm.Normal``/``pm.StudentT`` plus a separate
          ``pm.Potential("jacobian", logdet(ρ))``.  PyMC's log-likelihood
          machinery walks ``observed_RVs`` only, so the Potential is invisible
          to it and any captured group would be missing ``log|I - ρW|/n``.
        * ``compute_log_likelihood=True`` — the Jacobian is folded into an
          observed ``pm.CustomDist`` (spread uniformly across the ``n``
          observations, the same convention as the λ branch), so PyMC captures
          a *complete* pointwise log-likelihood natively and no post-hoc
          reconstruction is needed.

        Both forms evaluate ``logdet(ρ)`` exactly once per logp call, so the
        Jacobian is moved rather than duplicated and the posterior is
        unchanged.  The Potential form stays the default because it is the
        benchmarked-fast graph for the C/Numba backend.
        """
        import pytensor.tensor as pt

        self._validate_wx_columns()
        Z = self._design_matrix()
        names = self._design_names()
        priors = self._gaussian_priors(Z, names)

        logdet_fn = self._logdet_pytensor_fn
        n_obs = int(self._y.shape[0])
        native_ll = bool(compute_log_likelihood)
        self._native_log_likelihood = native_ll

        with pm.Model(coords=self._model_coords()) as model:
            rho = pm.Uniform(
                "rho",
                lower=priors["rho_lower"],
                upper=priors["rho_upper"],
            )
            beta = pm.Normal(
                "beta",
                mu=priors["beta_mu"],
                sigma=priors["beta_sigma"],
                dims="coefficient",
            )
            sigma2 = pm.InverseGamma(
                "sigma2",
                alpha=priors["sigma2_alpha"],
                beta=priors["sigma2_beta"],
            )
            sigma = pm.Deterministic("sigma", pt.sqrt(sigma2))

            if native_ll:
                # Constants folded into the logp closure so the graph matches
                # the Potential form term for term.
                Wy_const = pt.as_tensor_variable(self._Wy)
                Z_const = pt.as_tensor_variable(Z)
                inv_n = 1.0 / n_obs

                if self.robust:
                    nu = self._nu

                    def _logp(value, rho_, beta_, sigma_, nu_):
                        mu_ = rho_ * Wy_const + pt.dot(Z_const, beta_)
                        log_dens = pm.logp(
                            pm.StudentT.dist(nu=nu_, mu=mu_, sigma=sigma_),
                            value,
                        )
                        return log_dens + logdet_fn(rho_) * inv_n

                    pm.CustomDist(
                        "obs",
                        rho,
                        beta,
                        sigma,
                        nu,
                        logp=_logp,
                        observed=self._y,
                    )
                else:

                    def _logp(value, rho_, beta_, sigma_):
                        mu_ = rho_ * Wy_const + pt.dot(Z_const, beta_)
                        log_dens = pm.logp(
                            pm.Normal.dist(mu=mu_, sigma=sigma_),
                            value,
                        )
                        return log_dens + logdet_fn(rho_) * inv_n

                    pm.CustomDist(
                        "obs",
                        rho,
                        beta,
                        sigma,
                        logp=_logp,
                        observed=self._y,
                    )
            else:
                # mu = rho * Wy + Z @ beta
                mu = rho * self._Wy + pt.dot(Z, beta)
                if self.robust:
                    nu = self._nu
                    pm.StudentT("obs", nu=nu, mu=mu, sigma=sigma, observed=self._y)
                else:
                    pm.Normal("obs", mu=mu, sigma=sigma, observed=self._y)

                # Jacobian: log|I - rho*W|
                pm.Potential("jacobian", logdet_fn(rho))

        return model

    # ------------------------------------------------------------------
    # Branch 3: λ Jacobian (SEM, SDEM)
    # ------------------------------------------------------------------

    def _build_pymc_model_lam(
        self,
        *,
        nuts_sampler: str = "pymc",
        compute_log_likelihood: bool = False,
    ) -> pm.Model:
        """Build PyMC model for SEM or SDEM (spatial error with λ Jacobian).

        The ``pm.CustomDist`` form is used when the JAX samplers require an
        observed RV (their log-likelihood path iterates ``observed_RVs`` and
        crashes on an empty list) *or* when a pointwise log-likelihood is
        requested — in the latter case it lets PyMC capture a complete group
        natively instead of reconstructing it afterwards.  Otherwise the
        benchmarked ``pm.Potential`` formulation is used.
        """
        import pytensor.tensor as pt

        self._validate_wx_columns()
        Z = self._design_matrix()
        names = self._design_names()
        priors = self._gaussian_priors(Z, names)

        logdet_fn = self._logdet_pytensor_fn

        # Precompute W @ Z so the spatial filter can be expressed as
        #   eps = (y - lam*Wy) - (Z - lam*WZ) @ beta
        # avoiding any sparse matvec inside the NUTS gradient loop.
        # Uses _spatial_lag hook so panel models use _sparse_panel_lag.
        cache_attr = "_WZ_cache"
        if not hasattr(self, cache_attr) or getattr(self, cache_attr) is None:
            setattr(
                self, cache_attr, np.asarray(self._spatial_lag(Z), dtype=np.float64)
            )
        WZ = getattr(self, cache_attr)

        n_obs = int(self._y.shape[0])
        jax_logp = use_jax_likelihood(nuts_sampler) or bool(compute_log_likelihood)
        self._native_log_likelihood = jax_logp

        with pm.Model(coords=self._model_coords()) as model:
            lam = pm.Uniform(
                "lam",
                lower=priors["lam_lower"],
                upper=priors["lam_upper"],
            )
            beta = pm.Normal(
                "beta",
                mu=priors["beta_mu"],
                sigma=priors["beta_sigma"],
                dims="coefficient",
            )
            sigma2 = pm.InverseGamma(
                "sigma2",
                alpha=priors["sigma2_alpha"],
                beta=priors["sigma2_beta"],
            )
            sigma = pm.Deterministic("sigma", pt.sqrt(sigma2))

            if jax_logp:
                # JAX path: register an observed RV via pm.CustomDist so PyMC
                # can capture ``log_likelihood`` natively.
                Wy_const = pt.as_tensor_variable(self._Wy)
                Z_const = pt.as_tensor_variable(Z)
                WZ_const = pt.as_tensor_variable(WZ)
                inv_n = 1.0 / n_obs

                if self.robust:
                    nu = self._nu

                    def _logp(value, lam_, beta_, sigma_, nu_):
                        y_star = value - lam_ * Wy_const
                        Z_star = Z_const - lam_ * WZ_const
                        eps = y_star - pt.dot(Z_star, beta_)
                        log_dens = pm.logp(
                            pm.StudentT.dist(nu=nu_, mu=0.0, sigma=sigma_),
                            eps,
                        )
                        return log_dens + logdet_fn(lam_) * inv_n

                    pm.CustomDist(
                        "obs",
                        lam,
                        beta,
                        sigma,
                        nu,
                        logp=_logp,
                        observed=self._y,
                    )
                else:

                    def _logp(value, lam_, beta_, sigma_):
                        y_star = value - lam_ * Wy_const
                        Z_star = Z_const - lam_ * WZ_const
                        eps = y_star - pt.dot(Z_star, beta_)
                        log_dens = pm.logp(
                            pm.Normal.dist(mu=0.0, sigma=sigma_),
                            eps,
                        )
                        return log_dens + logdet_fn(lam_) * inv_n

                    pm.CustomDist(
                        "obs",
                        lam,
                        beta,
                        sigma,
                        logp=_logp,
                        observed=self._y,
                    )
            else:
                # Default (C / Numba) path: benchmarked pm.Potential formulation.
                y_star = self._y - lam * self._Wy
                Z_star = Z - lam * WZ
                eps = y_star - pt.dot(Z_star, beta)
                if self.robust:
                    nu = self._nu
                    logp_eps = pm.logp(
                        pm.StudentT.dist(nu=nu, mu=0.0, sigma=sigma),
                        eps,
                    )
                else:
                    logp_eps = pm.logp(
                        pm.Normal.dist(mu=0.0, sigma=sigma),
                        eps,
                    )
                pm.Potential("eps_loglik", logp_eps.sum())
                pm.Potential("jacobian", logdet_fn(lam))

        return model


# ---------------------------------------------------------------------------
# Gibbs registry entry
# ---------------------------------------------------------------------------


def _run_gaussian_gibbs(
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
    slice_width=None,
    chain_method=None,
    log_likelihood=False,
):
    """Registry runner for Gaussian Gibbs (cross-section and panel FE).

    Thin adapter over ``model._fit_gibbs``; ``backend`` (``"jax"`` or
    ``"numpy"``) maps directly onto the sampler's ``gibbs_method``.  The
    cross-section vs panel FE distinction is handled entirely by MRO dispatch
    of ``_fit_gibbs`` (``SpatialModel`` vs ``SpatialPanelModel``), so the same
    runner serves both registry keys.
    """
    return model._fit_gibbs(
        draws=draws,
        tune=tune,
        chains=chains,
        random_seed=random_seed,
        thin=thin,
        n_jobs=n_jobs,
        progressbar=progressbar,
        gibbs_method=backend,
        slice_width=slice_width,
        chain_method=chain_method,
        log_likelihood=log_likelihood,
    )


for _structure in ("cross_section", "panel_fe"):
    register(
        "gaussian",
        _structure,
        run=_run_gaussian_gibbs,
        backends={"jax", "numpy"},
        options={"slice_width", "chain_method"},
        skips_log_likelihood=True,
    )
