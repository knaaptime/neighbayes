r"""The latent SAR, SEM and SDEM families: sampling and the latent index.

A latent model propagates the family's own disturbance through the spatial
filter, :math:`A\eta = X\beta + \varepsilon` (SAR) or :math:`A(\eta - X\beta)
= \varepsilon` (SEM), with :math:`A = I - \rho W`: the logit's logistic
and the probit's normal (:mod:`._latent_binary`) and the negative binomial's
log-gamma (:mod:`._latent_negbin`).  The family supplies its Gibbs priors, samplers,
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
over fresh disturbances, with the draws' sparse solves batched; the negative
binomial's has a closed form (:func:`._latent_negbin.latent_count_effects`).
"""

from __future__ import annotations

import warnings
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

import numpy as np
import scipy.sparse as sp

from ..._lazy_deps import xr
from ...samplers._utils._idata import gibbs_to_inference_data
from ...samplers._utils._sparsax_utils import (
    _BATCH_BYTES,
    CachedSparseSolver,
    ResolventInterpolant,
)
from ...samplers._utils._sweeps import probe_sweep_midpoint, run_sweeps_with_midpoint

#: Disturbance vectors per posterior draw.  The effects average over the units,
#: so their Monte Carlo error falls with ``n`` as the posterior sd does.  For the
#: bounded weights simulated here 32 leave it at 1-3% of the posterior sd (logit
#: and probit, ``n`` = 2500), which widens an interval by far less than 1%.
_OUTCOME_SIMS = 32
#: Posterior draws solved in one batched call.
_OUTCOME_CHUNK = 64


def _chain_par(chain) -> float:
    """A latent chain's spatial parameter."""
    return chain.par


def _set_chain_logdet(chain, fn) -> None:
    chain.logdet_fn = fn


class LatentMixin:
    """Gibbs fit and latent-index simulation of a latent model.

    The host sets ``_latent_process`` to ``"sar"`` or ``"sem"``; the family
    mixin sets ``_latent_chain_class``, its NumPy chain object, and
    ``_latent_chain_kw``, the keywords both backends' chains take beyond the
    shared ones, implements :meth:`_latent_priors`, :meth:`_latent_chains_jax`
    and, if it simulates its fitted values, :meth:`_latent_disturbances`, and
    names its parameters beyond the spatial one and ``beta`` in ``_latent_extra``.
    """

    _latent_process: str = "sar"
    _latent_extra: tuple[str, ...] = ()
    _latent_chain_kw: dict = {}
    #: Why the family's latent model has no NUTS path.
    _latent_no_nuts: str = ""
    #: The scales impacts are reported on: the latent index and the outcome.
    _index_scale: str
    _outcome_scale: str

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
    def _latent_level_shift(self) -> float:
        """Lee and Yu's (2010) time-effects term: ``T``, or 0 without period effects.

        With period effects inside the filter and a row-standardized ``W``, the
        samplers add ``-T log(1 - ρ)`` to ρ's target, which makes each period's
        effect flat on its log-mean level rather than on itself (see
        :mod:`neighbayes.samplers.negbin_latent._core`), as the Gaussian panels
        do (``SpatialModel._jacobian_shift``).
        """
        if not self._latent_periods or self._latent_process != "sar":
            return 0.0
        return float(self._T) if self._W_row_standardized() else 0.0

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
        if b.method != "stoch_cheb" or not self.logdet_adapt:
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
        return {
            "T": int(self._T),
            "unit_effects": fx,
            "level_shift": self._latent_level_shift,
        }

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
                **self._latent_chain_kw,
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
            chain_kw = self._latent_chain_kw

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
                    **chain_kw,
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
            + self._reduced_form_hint
        )

    @property
    def _reduced_form_hint(self) -> str:
        """Where the class also has a reduced form, how to fit it by NUTS."""
        if self._design != "latent":
            return ""
        return " Pass latent=False for the reduced form, which NUTS fits."

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

    def _latent_effect_args(self) -> dict:
        """The posterior and design arguments of the outcome-scale effects."""
        post = self.inference_data.posterior
        par = post[self._latent_param].values.reshape(-1)
        X, _ = self._latent_design
        W = self._W_sparse.tocsr()
        periods = self._T if self._is_panel else 1
        durbin = None
        if getattr(self, "_has_wx_in_beta", False):
            durbin = (
                self._X.shape[1],
                list(self._beta_wx_column_indices),
                np.tile(np.asarray(W.sum(axis=1)).ravel(), periods),
                np.tile(W.diagonal(), periods),
            )
        return {
            "par": par,
            "beta": post["beta"].values.reshape(par.size, X.shape[1]),
            "offset": self._latent_systematic(post),
            "W": W,
            "sar": self._latent_process == "sar",
            "ni": None if durbin else list(self._beta_nonintercept_indices),
            "durbin": durbin,
            "periods": periods,
        }

    def _latent_outcome_effects(
        self, n_mc: int = _OUTCOME_SIMS, seed: int = 0
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Posterior draws of the outcome-scale direct, indirect and total effects.

        See :func:`latent_outcome_effects`.
        """
        post = self.inference_data.posterior
        direct, total = latent_outcome_effects(
            **self._latent_effect_args(),
            extra={k: post[k].values.reshape(-1) for k in self._latent_extra},
            noise=self._latent_noise,
            weight=self._latent_outcome_weight,
            n_mc=n_mc,
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
        N = self._W_sparse.shape[0]
        T = n // N
        solver = CachedSparseSolver([self._W_sparse], N)
        rhs = Xb[:, None] + eps if self._latent_process == "sar" else eps
        x = stacked_columns(solver.solve([-par], period_columns(rhs, N, T)), N, T)
        return x if self._latent_process == "sar" else Xb[:, None] + x


def period_columns(B: np.ndarray, N: int, T: int) -> np.ndarray:
    """A time-first stack ``(N T, M)`` as columns of one period's system, ``(N, T M)``."""
    return B if T == 1 else B.reshape(T, N, -1).transpose(1, 0, 2).reshape(N, -1)


def stacked_columns(X: np.ndarray, N: int, T: int) -> np.ndarray:
    """The inverse of :func:`period_columns`."""
    return X if T == 1 else X.reshape(N, T, -1).transpose(1, 0, 2).reshape(N * T, -1)


def weighted_effects(w, d, rows, beta, ni, durbin) -> tuple[np.ndarray, np.ndarray]:
    r"""``(direct, total)`` of draws ``beta`` ``(B, k)`` from their weights ``w`` ``(B, n)``.

    The SAR's average diagonal and row sum of :math:`\operatorname{diag}(w)\,S`
    (``d`` and ``rows``), the SEM's :math:`\bar w\beta` (``rows`` is ``None``),
    and the SDEM's :math:`\bar w\beta` plus :math:`\overline{w\,W \mathbf
    1}\,\theta` indirect (``durbin``).
    """
    mean = np.mean(w, axis=1, keepdims=True)
    if durbin is not None:
        k, wx_idx, w_rows, w_diag = durbin
        own = mean * beta[:, :k][:, wx_idx]
        lagged = beta[:, k:]
        return (
            own + np.mean(w * w_diag, axis=1, keepdims=True) * lagged,
            own + np.mean(w * w_rows, axis=1, keepdims=True) * lagged,
        )
    b = beta[:, ni]
    if rows is None:
        return mean * b, mean * b
    return (
        np.mean(w * d, axis=1, keepdims=True) * b,
        np.mean(w * rows, axis=1, keepdims=True) * b,
    )


def defined_effects(direct, total) -> tuple[np.ndarray, np.ndarray]:
    """``(direct, total)`` less the draws with no expected count, with a warning.

    A negative multiplier entry at or below ``-alpha`` makes a unit's expected
    count infinite (a negative ``rho`` at small ``alpha``), though its log-mean
    is finite.  The effects of the rest of the posterior are still defined.
    """
    keep = np.isfinite(direct).all(axis=1) & np.isfinite(total).all(axis=1)
    if keep.all():
        return direct, total
    if not keep.any():
        raise ValueError(
            "No posterior draw has an expected count: a multiplier entry at or "
            "below -alpha makes E[gamma^s] infinite (a negative rho at small "
            "alpha).  The effects on the log-mean are still defined."
        )
    warnings.warn(
        f"{keep.size - keep.sum()} of {keep.size} posterior draws have no expected "
        "count (a multiplier entry at or below -alpha, from a negative rho at small "
        "alpha) and are left out of the count-scale effects.",
        UserWarning,
        stacklevel=3,
    )
    return direct[keep], total[keep]


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
    durbin=None,
    periods: int = 1,
    n_mc: int = _OUTCOME_SIMS,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    r"""Posterior draws ``(direct, total)`` of a latent model's outcome-scale effects.

    For each posterior draw, ``n_mc`` fresh disturbances ``noise(rng, size,
    params)`` give the weights :math:`w = ` ``weight(m, U, E, d, params)`` (see
    the module), which :func:`weighted_effects` turns into effects.  A weight
    should be bounded: Monte Carlo cannot see the rare draws that dominate the
    mean of an unbounded one at small dispersion, whatever ``n_mc``
    (:func:`._latent_negbin.latent_count_effects`).

    ``par`` is ``(G,)``, ``beta`` ``(G, k)``, ``offset`` ``(G, n)`` each draw's
    systematic part, ``extra`` the family's other parameters by name, ``(G,)``
    each, and ``W`` one period's weights, the stack holding ``periods`` of them.
    ``ni`` are the effect columns of ``beta``; ``durbin`` is ``(k, wx_idx, w_rows,
    w_diag)`` on the stack for the SDEM instead.  The multiplier's diagonal and
    row sums are exact, interpolated across the draws
    (:class:`~neighbayes.samplers._utils._sparsax_utils.ResolventInterpolant`),
    and the disturbances' solves batched.  Each draw has its own random stream
    and its simulations run on threads (the generators and ufuncs release the
    GIL), so ``seed`` fixes the result whatever the chunking or thread count.
    """
    G = par.size
    n = offset.shape[1]
    N = W.shape[0]
    F = ResolventInterpolant(W, par, diag=True, row_sums=sar)
    L = F.weights(par)
    solver = CachedSparseSolver([W], N)
    cols = n_mc + int(sar)
    chunk = int(np.clip(_BATCH_BYTES // (8 * n * cols), 1, _OUTCOME_CHUNK))

    n_eff = len(durbin[1]) if durbin is not None else len(ni)
    direct = np.empty((G, n_eff))
    total = np.empty((G, n_eff))
    with ThreadPoolExecutor() as pool:
        for start in range(0, G, chunk):
            idx = np.arange(start, min(start + chunk, G))
            params = [{k_: v[g] for k_, v in extra.items()} for g in idx]

            def simulate(i):
                rng = np.random.default_rng([seed, int(idx[i])])
                e = noise(rng, (n, n_mc), params[i])
                rhs = np.column_stack([e, offset[idx[i]]]) if sar else e
                return e, period_columns(rhs, N, periods)

            E, rhs = zip(*pool.map(simulate, range(idx.size)))
            sol = solver.solve_batch(-par[idx, None], np.stack(rhs), pad_to=chunk)
            d = np.tile(L[idx] @ F.values["diag"], periods)

            def weigh(i):
                x = stacked_columns(sol[i], N, periods)
                m = x[:, n_mc] if sar else offset[idx[i]]
                return weight(m, x[:, :n_mc], E[i], d[i], params[i])

            w = np.stack(list(pool.map(weigh, range(idx.size))))
            rows = np.tile(L[idx] @ F.values["row_sums"], periods) if sar else None
            direct[idx], total[idx] = weighted_effects(
                w, d, rows, beta[idx], ni, durbin
            )
    return direct, total
