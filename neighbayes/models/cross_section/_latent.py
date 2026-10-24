r"""The latent SAR, SEM and SDEM families: sampling and the latent index.

A latent model propagates the family's own disturbance through the spatial
filter, :math:`A\eta = X\beta + \varepsilon` (SAR) or :math:`A(\eta - X\beta)
= \varepsilon` (SEM), with :math:`A = I - \rho W`: the logit's logistic
(:mod:`._latent_logit`) and the negative binomial's log-gamma
(:mod:`._latent_negbin`).  The family supplies its Gibbs priors, samplers,
disturbance draws and summaries; :class:`LatentMixin` runs the chains, builds
the posterior and simulates the index.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import scipy.sparse as sp

from ..._lazy_deps import xr
from ...samplers._utils._idata import gibbs_to_inference_data
from ...samplers.gaussian._chain_runner import run_chains


class LatentMixin:
    """Gibbs fit and latent-index simulation of a latent model.

    The host sets ``_latent_process`` to ``"sar"`` or ``"sem"``; the family
    mixin implements :meth:`_latent_priors`, :meth:`_latent_chain`,
    :meth:`_latent_chains_jax` and :meth:`_latent_disturbances`, and names its
    parameters beyond the spatial one and ``beta`` in ``_latent_extra``.
    """

    _latent_process: str = "sar"
    _latent_extra: tuple[str, ...] = ()
    #: Why the family's latent model has no NUTS path.
    _latent_no_nuts: str = ""

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
        """Sample by the family's latent Gibbs sweep.

        ``backend="numpy"`` updates the latent index one coordinate at a time
        (a compiled loop); ``"jax"`` updates it by colour class, on the device.
        ``slice_width`` is the initial width of the spatial parameter's slice,
        adapted in warmup.  The pointwise log-likelihood integrates the
        propagated disturbances and has no closed form, so none is stored.
        """
        y = np.asarray(self._y, dtype=np.float64)
        X, names = self._latent_design
        priors = self._latent_priors(X, names)
        W = self._W_sparse.tocsr()
        rng = np.random.default_rng(random_seed)
        seeds = [int(s) for s in rng.integers(0, 2**31, size=chains)]
        if backend == "jax":
            from ..._logdet._jax import logdet_jax_params

            bounds = self._logdet_bounds
            eigs = self._logdet_eigs if bounds.method == "eigenvalue" else None
            kind, params = logdet_jax_params(
                W, bounds.method, bounds.rho_min, bounds.rho_max, eigs=eigs
            )
            return self._latent_idata(
                self._latent_chains_jax(
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
            )
        logdet = self._logdet_numpy_fn
        process = self._latent_process
        chain = self._latent_chain

        def _run_one_chain(chain_id, seed, progress_manager=None, chain_id_kw=None):
            return chain(
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

    def _latent_idata(self, results) -> xr.DataTree:
        names = (self._latent_param, "beta", *self._latent_extra)
        idata = gibbs_to_inference_data(
            posterior_samples={
                name: np.stack([r[name] for r in results], axis=0) for name in names
            },
            observed_data={"obs": self._latent_observed},
            coords={"coefficient": self._latent_design[1]},
            dims={"beta": ["coefficient"]},
        )
        self._idata = idata
        return idata

    @property
    def _latent_observed(self) -> np.ndarray:
        return np.asarray(self._y, dtype=np.float64)

    def _build_pymc_model_latent(self):
        """Not supported: the latent likelihood has no closed form."""
        raise NotImplementedError(
            f"{type(self).__name__}'s latent model is fitted by Gibbs sampling: "
            + self._latent_no_nuts
        )

    def _latent_index_draws(
        self, n_sims: int, random_seed: Optional[int]
    ) -> np.ndarray:
        """``n_sims`` draws of the latent index at the posterior means, ``(n, n_sims)``."""
        self._require_fit()
        par = float(self._posterior_mean(self._latent_param))
        Xb = self._latent_design[0] @ self._posterior_mean("beta")
        n = Xb.shape[0]
        eps = self._latent_disturbances(np.random.default_rng(random_seed), (n, n_sims))
        lu = sp.linalg.splu(
            (sp.identity(n, format="csc") - par * self._W_sparse).tocsc()
        )
        if self._latent_process == "sar":
            return lu.solve(Xb[:, None] + eps)
        return Xb[:, None] + lu.solve(eps)
