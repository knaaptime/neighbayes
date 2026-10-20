r"""The latent SAR and SEM logit: sampling and summaries.

Shared by ``SARLogit(latent=True)`` (:math:`(I - \rho W) y^* = X\beta +
\varepsilon`) and :class:`SEMLogit` (:math:`(I - \lambda W)(y^* - X\beta) =
\varepsilon`), with :math:`y = \mathbf 1\{y^* > 0\}` and the disturbance the
logit's own, standard logistic.  The sampler is
:mod:`neighbayes.samplers.logit_latent`.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import scipy.sparse as sp

from ..._lazy_deps import xr
from ...samplers._utils._idata import gibbs_to_inference_data
from ...samplers.gaussian._chain_runner import run_chains
from ...samplers.logit_latent import run_chain  # registers the Gibbs entry
from ..priors import LogitGibbsPriors


class LatentLogitMixin:
    """Gibbs fit and fitted probabilities of a latent logit.

    The host sets ``_latent_process`` to ``"sar"`` or ``"sem"``.
    """

    _latent_process: str = "sar"

    @property
    def _latent_param(self) -> str:
        return "rho" if self._latent_process == "sar" else "lam"

    @property
    def _latent_design(self) -> tuple[np.ndarray, list[str]]:
        """The design and coefficient names: ``[X, WX]`` for a Durbin model."""
        X = np.ascontiguousarray(self._X, dtype=np.float64)
        if getattr(self, "_has_wx_in_beta", False):
            return np.hstack([X, np.asarray(self._WX, dtype=np.float64)]), list(
                self._beta_names()
            )
        return X, list(self._feature_names)

    def _fit_gibbs_latent(
        self,
        *,
        draws: int = 2000,
        tune: int = 1000,
        chains: int = 4,
        random_seed: Optional[int] = None,
        thin: int = 1,
        n_jobs: int = -1,
        progressbar: bool = True,
        backend: str = "numpy",
        slice_width: float = 0.4,
    ) -> xr.DataTree:
        """Sample by the exact latent-utility Gibbs sweep (registry ``binary_latent``).

        ``backend="numpy"`` updates the utilities one at a time (a compiled
        loop); ``"jax"`` updates them by colour class, on the device
        (:mod:`neighbayes.samplers.logit_latent._jax`).  ``slice_width`` is the
        initial width of the spatial parameter's slice, adapted in warmup on
        the JAX backend.  The pointwise log-likelihood is an orthant
        probability with no closed form, so none is stored.
        """
        y = np.asarray(self._y, dtype=np.float64)
        X, names = self._latent_design
        bounds = self._logdet_bounds
        beta_mu, beta_sigma = self._resolved_beta_prior(
            design=X, feature_names=names, link="logit"
        )
        priors = LogitGibbsPriors(
            beta_mu=beta_mu,
            beta_sigma=beta_sigma,
            rho_lower=float(bounds.rho_min),
            rho_upper=float(bounds.rho_max),
        )
        W = self._W_sparse.tocsr()
        process = self._latent_process
        rng = np.random.default_rng(random_seed)
        seeds = [int(s) for s in rng.integers(0, 2**31, size=chains)]
        if backend == "jax":
            return self._latent_idata(
                self._fit_latent_jax(
                    y, X, W, priors, draws, tune, thin, seeds, slice_width
                )
            )
        logdet = self._logdet_numpy_fn

        def _run_one_chain(chain_id, seed, progress_manager=None, chain_id_kw=None):
            return run_chain(
                y=y,
                X=X,
                W=W,
                process=process,
                priors=priors,
                logdet_fn=logdet,
                draws=draws,
                tune=tune,
                thin=thin,
                rng=np.random.default_rng(seed),
                chain_id=chain_id_kw if chain_id_kw is not None else chain_id,
                progress_manager=progress_manager,
                slice_width=slice_width,
            )

        results = run_chains(
            chain_fn=_run_one_chain,
            n_chains=chains,
            seeds=seeds,
            n_jobs=n_jobs,
            progressbar=progressbar,
            parallel=(n_jobs != 1),
            draws=draws,
            tune=tune,
            model_type=type(self).__name__,
            timeout=None,
        )
        return self._latent_idata(results)

    def _fit_latent_jax(self, y, X, W, priors, draws, tune, thin, seeds, slice_width):
        """The chains on the JAX backend, the log-determinant passed as data."""
        from ..._logdet._jax import logdet_jax_params
        from ...samplers.logit_latent._jax import run_chains_jax_latent_logit

        bounds = self._logdet_bounds
        eigs = self._logdet_eigs if bounds.method == "eigenvalue" else None
        kind, params = logdet_jax_params(
            W, bounds.method, bounds.rho_min, bounds.rho_max, eigs=eigs
        )
        return run_chains_jax_latent_logit(
            y=y,
            X=X,
            W=W,
            process=self._latent_process,
            priors=priors,
            logdet_kind=kind,
            logdet_params=params,
            draws=draws,
            tune=tune,
            thin=thin,
            seeds=seeds,
            slice_width=slice_width,
        )

    def _latent_idata(self, results) -> xr.DataTree:
        name = self._latent_param
        idata = gibbs_to_inference_data(
            posterior_samples={
                name: np.stack([r[name] for r in results], axis=0),
                "beta": np.stack([r["beta"] for r in results], axis=0),
            },
            observed_data={"obs": np.asarray(self._y, dtype=np.float64)},
            coords={"coefficient": self._latent_design[1]},
            dims={"beta": ["coefficient"]},
        )
        self._idata = idata
        return idata

    def _build_pymc_model_latent(self):
        """Not supported: the latent logit's likelihood has no closed form."""
        raise NotImplementedError(
            f"{type(self).__name__}'s latent logit is fitted by Gibbs sampling: "
            "its likelihood is an orthant probability of correlated utilities, "
            "which no NUTS model evaluates."
        )

    def _fitted_probabilities_latent(
        self, n_sims: int = 1000, random_seed: Optional[int] = 0
    ) -> np.ndarray:
        r"""``P(y_i = 1)`` at the posterior-mean parameters, by simulation.

        The utilities' marginals are not logistic (the multiplier mixes the
        disturbances), so ``n_sims`` disturbance vectors are drawn and the
        share of positive utilities is returned.
        """
        self._require_fit()
        par = float(self._posterior_mean(self._latent_param))
        Xb = self._latent_design[0] @ self._posterior_mean("beta")
        n = Xb.shape[0]
        eps = np.random.default_rng(random_seed).logistic(size=(n, n_sims))
        lu = sp.linalg.splu(
            (sp.identity(n, format="csc") - par * self._W_sparse).tocsc()
        )
        if self._latent_process == "sar":
            ystar = lu.solve(Xb[:, None] + eps)
        else:
            ystar = Xb[:, None] + lu.solve(eps)
        return (ystar > 0).mean(axis=1)

    def _fitted_mean_from_posterior_latent(self) -> np.ndarray:
        """The fitted probabilities (:meth:`_fitted_probabilities_latent`)."""
        return self._fitted_probabilities_latent()
