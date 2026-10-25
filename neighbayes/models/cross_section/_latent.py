r"""The latent SAR, SEM and SDEM families: sampling and the latent index.

A latent model propagates the family's own disturbance through the spatial
filter, :math:`A\eta = X\beta + \varepsilon` (SAR) or :math:`A(\eta - X\beta)
= \varepsilon` (SEM), with :math:`A = I - \rho W`: the logit's logistic
(:mod:`._latent_logit`) and the negative binomial's log-gamma
(:mod:`._latent_negbin`).  The family supplies its Gibbs priors, samplers,
disturbance draws and summaries; :class:`LatentMixin` runs the chains, builds
the posterior and simulates the index.

Impacts on the outcome scale (expected counts, probabilities) integrate over
the propagated disturbances.  With :math:`u = M\varepsilon` the propagated
disturbance (:math:`M = A^{-1}` for SAR, :math:`(I - \lambda W)^{-1}` for SEM)
and :math:`m` the mean index, the outcome mean is
:math:`\mathbb{E}\,g(m_i + u_i)`, and the derivative with respect to
:math:`x_{jk}` is :math:`w_i\,\partial m_i/\partial x_{jk}` with a weight
:math:`w_i` the family gives: :math:`\mathbb{E}\,e^{m_i + u_i}` for the negative
binomial and the density of :math:`u_i` at :math:`-m_i` for the logit.
:meth:`LatentMixin._latent_outcome_effects` estimates :math:`w` by Monte Carlo
over fresh disturbances, one batched sparse solve per posterior draw.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import scipy.sparse as sp

from ..._lazy_deps import xr
from ...samplers._utils._idata import gibbs_to_inference_data
from ...samplers._utils._sweeps import probe_sweep_midpoint, run_sweeps_with_midpoint


def _chain_par(chain) -> float:
    """A latent chain's spatial parameter."""
    return chain.par


def _set_chain_logdet(chain, fn) -> None:
    chain.logdet_fn = fn


class LatentMixin:
    """Gibbs fit and latent-index simulation of a latent model.

    The host sets ``_latent_process`` to ``"sar"`` or ``"sem"``; the family
    mixin sets ``_latent_chain_class``, its NumPy chain object, implements
    :meth:`_latent_priors`, :meth:`_latent_chains_jax` and
    :meth:`_latent_disturbances`, and names its parameters beyond the spatial
    one and ``beta`` in ``_latent_extra``.
    """

    _latent_process: str = "sar"
    _latent_extra: tuple[str, ...] = ()
    #: Why the family's latent model has no NUTS path.
    _latent_no_nuts: str = ""
    #: The scales impacts are reported on: the latent index and the outcome.
    _index_scale: str
    _outcome_scale: str
    #: Largest ``n`` whose multiplier diagonal comes from the eigendecomposition.
    _LATENT_EFFECTS_EIGEN_MAX_N: int = 2000

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

    @property
    def _latent_periods(self) -> bool:
        """Whether the latent panel has period effects (columns of the β block)."""
        return self._is_panel and getattr(self, "_fe_periods", False)

    @property
    def _latent_units(self) -> bool:
        """Whether the latent panel has random unit effects."""
        return self._is_panel and getattr(self, "_fe_groups", False)

    @property
    def _latent_W(self) -> sp.csr_matrix:
        """The filter's weights on the stack: ``I_T ⊗ W`` for a panel."""
        W = self._W_sparse.tocsr()
        if self._is_panel:
            return sp.kron(sp.identity(self._T, format="csr"), W, format="csr")
        return W

    def _latent_block(self):
        """The sampler's coefficient block: the design, then the period dummies.

        Returns ``(Z, priors)``; the period effects take their own prior.
        """
        from dataclasses import replace

        X, names = self._latent_design
        priors = self._latent_priors(X, names)
        if not self._latent_periods:
            return X, priors
        fp = self._count_fe_priors()
        k, n_tau = X.shape[1], self._D_tau.shape[1]
        mu = np.r_[
            np.broadcast_to(priors.beta_mu, (k,)), np.full(n_tau, fp["time_effect_mu"])
        ]
        sd = np.r_[
            np.broadcast_to(priors.beta_sigma, (k,)),
            np.full(n_tau, fp["time_effect_sigma"]),
        ]
        return np.hstack([X, self._D_tau]), replace(priors, beta_mu=mu, beta_sigma=sd)

    @property
    def _latent_logdet_fn(self):
        """``log|I - ρW|`` of one period; the sampler counts it ``T`` times."""
        if not self._is_panel:
            return self._logdet_numpy_fn
        from ..._logdet import make_logdet_numpy_fn

        b = self._logdet_bounds
        return make_logdet_numpy_fn(
            self._W_sparse,
            self._logdet_eigs,
            method=b.method,
            rho_min=b.rho_min,
            rho_max=b.rho_max,
            T=1,
        )

    def _latent_probes(self):
        """The probe check of a stochastic log-determinant, or ``None`` for any other.

        A :class:`~neighbayes._logdet._probe_check.WarmupProbes` on one period's
        W, sized against the panel's ``T``-period Jacobian.
        """
        b = self._logdet_bounds
        if b.method != "cheb_stochastic" or not self.logdet_adapt:
            return None
        from ..._logdet._probe_check import WarmupProbes

        return WarmupProbes(
            self._W_sparse.tocsr(),
            T=self._latent_panel_kwargs().get("T", 1),
            rho_min=b.rho_min,
            rho_max=b.rho_max,
        )

    def _latent_panel_kwargs(self) -> dict:
        """``T`` and the unit-effect spec the samplers take for a panel."""
        if not self._is_panel:
            return {}
        from ...samplers.negbin_latent._core import UnitEffects

        fx = None
        if self._latent_units:
            fp = self._count_fe_priors()
            fx = UnitEffects(
                group=np.asarray(self._row_group),
                n_groups=int(self._n_groups),
                mu=fp["effect_mu"],
                sd_scale=fp["effect_sd_scale"],
            )
        return {"T": int(self._T), "unit_effects": fx}

    def _latent_systematic(self, post) -> np.ndarray:
        """``(G, n T)``: each draw's systematic part, ``Xβ`` plus the panel's effects."""
        X, _ = self._latent_design
        G = post["beta"].values.reshape(-1, X.shape[1]).shape[0]
        out = post["beta"].values.reshape(G, -1) @ X.T
        if self._latent_periods:
            tau = post["time_effect"].values.reshape(G, -1)
            out = out + tau @ self._D_tau.T
        if self._latent_units:
            c = post["unit_effect"].values.reshape(G, -1)
            out = out + c[:, self._row_group]
        return out

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
        log_likelihood: bool = False,
    ) -> xr.DataTree:
        """Sample by the family's latent Gibbs sweep.

        ``backend="numpy"`` updates the latent index one coordinate at a time
        (a compiled loop); ``"jax"`` updates it by colour class, on the device.
        ``slice_width`` is the initial width of the spatial parameter's slice,
        adapted in warmup.  ``log_likelihood`` stores each observation's
        likelihood with its latent value integrated out,
        :math:`p(y_i \\mid \\theta, \\eta_{-i})` (see
        :mod:`neighbayes.samplers._utils._latent_loglik`), for ``az.loo``.
        """
        y = np.asarray(self._y, dtype=np.float64)
        X, priors = self._latent_block()
        panel_kw = self._latent_panel_kwargs()
        W = self._W_sparse.tocsr()
        rng = np.random.default_rng(random_seed)
        seeds = [int(s) for s in rng.integers(0, 2**31, size=chains)]
        probes = self._latent_probes()
        self._fitted_probes = probes
        if backend == "jax":
            if probes is None:
                from ..._logdet._jax import logdet_jax_params

                bounds = self._logdet_bounds
                eigs = self._logdet_eigs if bounds.method == "eigenvalue" else None
                kind, params = logdet_jax_params(
                    W, bounds.method, bounds.rho_min, bounds.rho_max, eigs=eigs
                )
            else:
                kind, params = "cheb", probes.initial(tune, jax=True).params
            results = self._latent_chains_jax(
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
                store_latent=log_likelihood,
                **panel_kw,
                progressbar=progressbar,
                probes=probes,
            )
        else:
            if probes is None:
                logdet = self._latent_logdet_fn
            else:
                probes.initial(tune)
                logdet = probes.period_fn()
            cls, process = self._latent_chain_class, self._latent_process

            def make_chain(chain_rng):
                return cls(
                    y=y,
                    X=X,
                    W=W,
                    process=process,
                    priors=priors,
                    logdet_fn=logdet,
                    rng=chain_rng,
                    slice_width=slice_width,
                    **panel_kw,
                )

            results = run_sweeps_with_midpoint(
                make_chain,
                seeds=seeds,
                draws=draws,
                tune=tune,
                thin=thin,
                midpoint=None
                if probes is None
                else probe_sweep_midpoint([(probes, _chain_par, _set_chain_logdet)]),
                store_latent=log_likelihood,
                n_jobs=n_jobs,
                progressbar=progressbar,
                model_type=type(self).__name__,
            )
        idata = self._latent_idata(results, log_likelihood)
        if probes is not None:
            probes.record(idata, idata.posterior[self._latent_param].values)
        return idata

    def _latent_idata(self, results, log_likelihood: bool = False) -> xr.DataTree:
        def stack(name):
            return np.stack([np.asarray(r[name]) for r in results], axis=0)

        names = (self._latent_param, *self._latent_extra)
        post = {name: stack(name) for name in names}
        _, coef = self._latent_design
        block = stack("beta")
        post["beta"] = block[..., : len(coef)]
        coords = {"coefficient": coef}
        dims = {"beta": ["coefficient"]}
        if self._latent_periods:
            post["time_effect"] = block[..., len(coef) :]
            coords["time"] = self._period_labels
            dims["time_effect"] = ["time"]
        if self._latent_units:
            index = self._effect_index
            post[self._effect_name()] = stack("unit_effect")
            post[self._effect_name("_sd")] = stack("unit_effect_sd")
            coords[index] = list(range(self._n_groups))
            dims[self._effect_name()] = [index]
        idata = gibbs_to_inference_data(
            posterior_samples=post,
            log_likelihood=(
                {"obs": self._latent_log_likelihood(results)}
                if log_likelihood
                else None
            ),
            observed_data={"obs": self._latent_observed},
            coords=coords,
            dims=dims,
        )
        self._idata = idata
        return idata

    def _latent_log_likelihood(self, results) -> np.ndarray:
        """``(chains, draws, n)``: each observation's latent value integrated out."""
        from scipy.special import gammaln

        from ...samplers._utils._latent_loglik import POISSON, integrated_loglik
        from ...samplers.negbin_latent._core import filter_columns

        X, _ = self._latent_block()
        W = self._latent_W
        indptr, rows, ones, wvals = filter_columns(W)
        y = np.asarray(self._latent_observed, dtype=np.float64)
        out = []
        for r in results:
            latent = np.ascontiguousarray(r["latent"], dtype=np.float64)
            par = np.asarray(r[self._latent_param], dtype=np.float64)
            m = np.asarray(r["beta"], dtype=np.float64) @ X.T
            if self._latent_units:
                m = m + np.asarray(r["unit_effect"])[:, self._row_group]
            # e = A η - c: the disturbances, c = m (SAR) or A m (SEM).
            dev = latent - m if self._latent_process == "sem" else latent
            e = dev - par[:, None] * np.asarray(W @ dev.T).T
            if self._latent_process == "sar":
                e -= m
            alpha = np.asarray(r.get("alpha", np.ones_like(par)), dtype=np.float64)
            out.append(
                integrated_loglik(
                    latent,
                    np.ascontiguousarray(e),
                    y,
                    indptr,
                    rows,
                    ones,
                    wvals,
                    par,
                    alpha,
                    self._latent_family,
                    gammaln(y + 1.0),
                    np.full(y.size, POISSON, dtype=np.int8),
                )  # fmt: skip
            )
        return np.stack(out)

    @property
    def _latent_observed(self) -> np.ndarray:
        return np.asarray(self._y, dtype=np.float64)

    def _build_pymc_model_latent(self):
        """Not supported: the latent likelihood has no closed form."""
        raise NotImplementedError(
            f"{type(self).__name__}'s latent model is fitted by Gibbs sampling: "
            + self._latent_no_nuts
        )

    def _latent_impacts(self, scale: str, return_posterior_samples: bool):
        """Impacts on the latent index (``_index_scale``) or the outcome scale."""
        if scale == self._index_scale:
            self._require_fit()
            return self._effects_frame(
                *self._compute_impacts_posterior(),
                return_posterior_samples=return_posterior_samples,
                scale=scale,
            )
        if scale != self._outcome_scale:
            raise ValueError(
                f"scale must be {self._index_scale!r} or {self._outcome_scale!r}."
            )
        self._require_fit()
        return self._effects_frame(
            *self._latent_outcome_effects(),
            return_posterior_samples=return_posterior_samples,
            scale=scale,
        )

    def _latent_outcome_effects(
        self, n_mc: int = 128, n_probes: int = 64, seed: int = 0
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Posterior draws of the outcome-scale direct, indirect and total effects.

        See :func:`latent_outcome_effects`; the multiplier's diagonal is exact
        from the eigendecomposition up to ``_LATENT_EFFECTS_EIGEN_MAX_N`` units
        and a Hutchinson estimate (``n_probes`` probes) beyond.
        """
        post = self.inference_data.posterior
        par = post[self._latent_param].values.reshape(-1)
        X, _ = self._latent_design
        sar = self._latent_process == "sar"
        durbin = None
        if getattr(self, "_has_wx_in_beta", False):
            W = self._latent_W
            durbin = (
                self._X.shape[1],
                list(self._beta_wx_column_indices),
                np.asarray(W.sum(axis=1)).ravel(),
                W.diagonal(),
            )
        eigen = None
        if self._W_sparse.shape[0] <= self._LATENT_EFFECTS_EIGEN_MAX_N:
            eigs, V, Vinv = self._W_eigendecomposition
            eigen = (eigs, V * Vinv.T)
        direct, total = latent_outcome_effects(
            par=par,
            beta=post["beta"].values.reshape(par.size, X.shape[1]),
            offset=self._latent_systematic(post),
            extra={k: post[k].values.reshape(-1) for k in self._latent_extra},
            W=self._latent_W,
            sar=sar,
            ni=None if durbin else list(self._beta_nonintercept_indices),
            noise=self._latent_noise,
            weight=self._latent_outcome_weight,
            eigen=eigen,
            row_sums=self._multiplier_row_sums if sar and self._is_row_std else None,
            durbin=durbin,
            periods=self._T if self._is_panel else 1,
            n_mc=n_mc,
            n_probes=n_probes,
            seed=seed,
        )
        return direct, total - direct, total

    def _latent_index_draws(
        self, n_sims: int, random_seed: Optional[int]
    ) -> np.ndarray:
        """``n_sims`` draws of the latent index at the posterior means, ``(n, n_sims)``."""
        self._require_fit()
        par = float(self._posterior_mean(self._latent_param))
        Xb = self._latent_systematic(self._idata.posterior).mean(axis=0)
        n = Xb.shape[0]
        eps = self._latent_disturbances(np.random.default_rng(random_seed), (n, n_sims))
        lu = sp.linalg.splu(
            (sp.identity(n, format="csc") - par * self._latent_W).tocsc()
        )
        if self._latent_process == "sar":
            return lu.solve(Xb[:, None] + eps)
        return Xb[:, None] + lu.solve(eps)


def latent_outcome_effects(
    *,
    par: np.ndarray,
    beta: np.ndarray,
    offset: np.ndarray,
    extra: dict,
    W: sp.csr_matrix,
    sar: bool,
    ni,
    noise,
    weight,
    eigen=None,
    row_sums=None,
    durbin=None,
    periods: int = 1,
    n_mc: int = 128,
    n_probes: int = 64,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    r"""Posterior draws ``(direct, total)`` of a latent model's outcome-scale effects.

    For each posterior draw, ``n_mc`` fresh disturbances ``noise(rng, size,
    params)`` give the weights :math:`w = ` ``weight(m, U, E, d, params,
    exact)`` (see the module); the effects are then the SAR's average diagonal
    and row sum of :math:`\operatorname{diag}(w)\,S`, the SEM's :math:`\bar
    w\beta`, and the SDEM's :math:`\bar w\beta` plus :math:`\overline{w\,W
    \mathbf 1}\,\theta` indirect.

    ``par`` is ``(G,)``, ``beta`` ``(G, k)``, ``offset`` ``(G, n)`` each draw's
    systematic part, ``extra`` the family's other parameters by name, ``(G,)``
    each, and ``W`` the filter's weights on the stack.  ``ni`` are the effect
    columns of ``beta``; ``durbin`` is ``(k, wx_idx, w_rows, w_diag)`` for the
    SDEM instead.  ``eigen`` is ``(eigs, V * Vinv.T)`` of the per-period
    weights, for an exact multiplier diagonal (tiled over ``periods``), else
    ``n_probes`` Hutchinson probes estimate it.  ``row_sums(r)`` gives a
    row-standardized SAR's multiplier row sums, else they are solved for.
    """
    from ..._ops import _make_cached_sparse_solver

    G = par.size
    n = offset.shape[1]
    I_n = sp.identity(n, format="csc")
    rng = np.random.default_rng(seed)
    if eigen is not None:
        eigs, VVinv = eigen
    else:
        Z = rng.choice(np.array([-1.0, 1.0]), size=(n, n_probes))
    if durbin is not None:
        k, wx_idx, w_rows, w_diag = durbin

    n_eff = len(wx_idx) if durbin is not None else len(ni)
    direct = np.empty((G, n_eff))
    total = np.empty((G, n_eff))
    for g in range(G):
        r = float(par[g])
        A = (I_n - r * W).tocsc()
        solver = _make_cached_sparse_solver(A) or sp.linalg.splu(A)
        params = {k_: v[g] for k_, v in extra.items()}
        E = noise(rng, (n, n_mc), params)
        Xb = offset[g]
        rhs = [E]
        if sar:
            rhs.append(Xb[:, None])
            if row_sums is None:
                rhs.append(np.ones((n, 1)))
        if eigen is None:
            rhs.append(Z)
        sol = np.asarray(solver.solve(np.hstack(rhs)), dtype=np.float64)
        U = sol[:, :n_mc]
        if eigen is not None:
            d = np.tile((VVinv @ (1.0 / (1.0 - r * eigs))).real, periods)
        else:
            d = np.mean(Z * sol[:, -n_probes:], axis=1)
        m = sol[:, n_mc] if sar else Xb
        w = weight(m, U, E, d, params, exact=eigen is not None)
        if sar:
            rows = (
                np.tile(row_sums(r), periods)
                if row_sums is not None
                else sol[:, n_mc + 1]
            )
            direct[g] = np.mean(w * d) * beta[g, ni]
            total[g] = np.mean(w * rows) * beta[g, ni]
        elif durbin is not None:
            b1, b2 = beta[g, :k][wx_idx], beta[g, k:]
            direct[g] = np.mean(w) * b1 + np.mean(w * w_diag) * b2
            total[g] = np.mean(w) * b1 + np.mean(w * w_rows) * b2
        else:
            direct[g] = total[g] = np.mean(w) * beta[g, ni]
    return direct, total
