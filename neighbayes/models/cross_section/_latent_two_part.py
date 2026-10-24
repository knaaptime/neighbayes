r"""The latent SAR hurdle and zero-inflated negative binomial.

``SARHurdleNegBin(latent=True)`` and ``SARZINB(latent=True)`` pair a latent SAR
logit zero part with a latent SAR negative binomial count part,

.. math::
    (I - \rho_z W_z)u = Z\gamma + \varepsilon_z, \qquad
    (I - \rho W)\eta = X\beta + \varepsilon,

each part's own disturbance (logistic, log-gamma) propagated by its filter.
The hurdle's :math:`u_i > 0` is a positive count, which is then
:math:`\mathrm{Poisson}(e^{\eta_i})` truncated at zero; the zero-inflated
model's is an active unit, whose count is :math:`\mathrm{Poisson}(e^{\eta_i})`.
The sampler is :mod:`neighbayes.samplers.two_part_latent`.

The link-scale impacts are the reduced form's formulas, since
:math:`\partial E u / \partial z' = S_z\gamma` and :math:`\partial E\eta /
\partial x' = S\beta`.  Probabilities, means and predictions are posterior
expectations: each posterior draw's are averaged over fresh disturbances, and
the draws averaged.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import scipy.sparse as sp

from ..._lazy_deps import xr
from ...samplers import two_part_latent as _two_part_latent  # noqa: F401
from ...samplers._utils._idata import gibbs_to_inference_data
from .._mixins._two_part import _SUMMARY_DRAWS, PART_SCALES

#: Fresh disturbance vectors per posterior draw in the simulated summaries.
_SUMMARY_SIMS = 64


def _log_gamma_noise(rng, alpha: float, size) -> np.ndarray:
    """``log g``, ``g ~ Gamma(alpha, alpha)``, without underflow at small ``alpha``."""
    return (
        np.log(rng.gamma(alpha + 1.0, 1.0 / alpha, size=size))
        + np.log(rng.random(size)) / alpha
    )


def _truncated_mean(lam: np.ndarray) -> np.ndarray:
    """``lam / (1 - e^{-lam})``, the mean of a Poisson count truncated at zero."""
    q = -np.expm1(-lam)
    return np.where(lam < 1e-8, 1.0 + 0.5 * lam, lam / np.where(q > 0, q, 1.0))


def _truncated_slope(lam: np.ndarray) -> np.ndarray:
    """``d/d log(lam)`` of the truncated mean: ``lam g'(lam)``, by series at small ``lam``."""
    q = -np.expm1(-lam)
    qs = np.where(lam < 1e-2, 1.0, q)
    closed = lam * (qs - lam * np.exp(-lam)) / (qs * qs)
    return np.where(lam < 1e-2, lam * (0.5 + lam / 6.0), closed)


def _draw_truncated_poisson(rng, lam: np.ndarray) -> np.ndarray:
    """Zero-truncated Poisson draws, exactly: the first event's time given one by
    time 1, then a Poisson count over the rest of the interval."""
    lam = np.maximum(lam, 1e-300)
    t = -np.log1p(rng.random(lam.shape) * np.expm1(-lam)) / lam
    return 1 + rng.poisson(lam * np.clip(1.0 - t, 0.0, 1.0))


class LatentTwoPartMixin:
    """Gibbs fit, log-likelihood and simulated summaries of the latent two-part models.

    The host sets ``_two_part`` (``"hurdle"`` or ``"zinb"``) and implements
    :meth:`_latent_two_part_priors`.
    """

    _two_part: str
    #: Largest ``n`` whose multiplier diagonal comes from the eigendecomposition.
    _LATENT_EFFECTS_EIGEN_MAX_N: int = 2000

    def _latent_two_part_priors(self):
        """``(count priors, zero priors)`` for the sampler."""
        raise NotImplementedError

    def _build_pymc_model_latent(self):
        raise NotImplementedError(
            f"{type(self).__name__}'s latent model is fitted by Gibbs sampling: its "
            "likelihood integrates both parts' propagated disturbances over all n "
            "locations at once, which no NUTS model evaluates."
        )

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def _latent_logdets(self, backend: str):
        """The count and zero parts' log-determinants for ``backend``."""
        from ..._logdet import make_logdet_numpy_fn

        b = self._logdet_bounds
        parts = [(self._W_sparse, self._logdet_eigs)]
        parts.append(parts[0] if self._same_W else (self._W_zero_sparse, None))
        if backend == "jax":
            from ..._logdet._jax import logdet_jax_params

            return [
                logdet_jax_params(
                    W.tocsr(),
                    b.method,
                    b.rho_min,
                    b.rho_max,
                    eigs=eigs if b.method == "eigenvalue" else None,
                )
                for W, eigs in parts
            ]
        return [
            make_logdet_numpy_fn(
                W.tocsr(),
                eigs,
                method=b.method,
                rho_min=b.rho_min,
                rho_max=b.rho_max,
                T=1,
            )
            for W, eigs in parts
        ]

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
        """Sample both parts by the joint Gibbs sweep.

        ``log_likelihood`` stores each observation's likelihood with both
        parts' latent values integrated out, for ``az.loo``.
        """
        priors, priors_zero = self._latent_two_part_priors()
        y = np.asarray(self._y, dtype=np.float64)
        X = np.ascontiguousarray(self._X, dtype=np.float64)
        Z = np.ascontiguousarray(self._Z, dtype=np.float64)
        W = self._W_sparse.tocsr()
        W_zero = self._W_zero_sparse.tocsr()
        rng = np.random.default_rng(random_seed)
        seeds = [int(s) for s in rng.integers(0, 2**31, size=chains)]
        logdet, logdet_zero = self._latent_logdets(backend)
        common = dict(
            model=self._two_part,
            y=y,
            X=X,
            Z=Z,
            W=W,
            W_zero=W_zero,
            priors=priors,
            priors_zero=priors_zero,
            draws=draws,
            tune=tune,
            thin=thin,
            slice_width=slice_width,
            store_latent=log_likelihood,
        )
        if backend == "jax":
            from ...samplers.two_part_latent._jax import run_chains_jax_latent_two_part

            results = run_chains_jax_latent_two_part(
                **common, logdet=logdet, logdet_zero=logdet_zero, seeds=seeds
            )
            return self._latent_two_part_idata(results, log_likelihood)

        from ...samplers.gaussian._chain_runner import run_chains
        from ...samplers.two_part_latent._core import run_chain

        def _run_one_chain(chain_id, seed, progress_manager=None, chain_id_kw=None):
            return run_chain(
                **common,
                logdet_fn=logdet,
                logdet_zero_fn=logdet_zero,
                rng=np.random.default_rng(seed),
                chain_id=chain_id_kw if chain_id_kw is not None else chain_id,
                progress_manager=progress_manager,
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
        return self._latent_two_part_idata(results, log_likelihood)

    def _latent_two_part_idata(self, results, log_likelihood: bool) -> xr.DataTree:
        def stack(name):
            return np.stack([np.asarray(r[name]) for r in results], axis=0)

        post = {
            name: stack(name)
            for name in ("rho_zero", "beta_zero", "rho", "beta", "alpha")
        }
        if self._two_part == "zinb":
            mean = np.mean([r["structural_prob"] for r in results], axis=0)
            second = np.mean(
                [
                    r["structural_prob_sd"] ** 2 + r["structural_prob"] ** 2
                    for r in results
                ],
                axis=0,
            )
            self._latent_structural = (mean, np.sqrt(np.maximum(second - mean**2, 0)))
        idata = gibbs_to_inference_data(
            posterior_samples=post,
            log_likelihood=(
                {"obs": self._latent_two_part_loglik(results)}
                if log_likelihood
                else None
            ),
            observed_data={"obs": self._y_int},
            coords={
                "coefficient_zero": list(self._sel_feature_names),
                "coefficient": list(self._feature_names),
                "obs_id": list(range(len(self._y))),
            },
            dims={
                "beta_zero": ["coefficient_zero"],
                "beta": ["coefficient"],
                "obs": ["obs_id"],
            },
        )
        self._idata = idata
        return idata

    def _latent_two_part_loglik(self, results) -> np.ndarray:
        r"""``(chains, draws, n)``: each observation's likelihood, latent values integrated out.

        With the parts' conditionals :math:`P_1 = P(u_i > 0 \mid u_{-i})` and
        :math:`C = \int p(y_i \mid \eta_i)\, p(\eta_i \mid \eta_{-i})\,
        d\eta_i`, the hurdle's is :math:`P_1 C` for a positive count and
        :math:`1 - P_1` for a zero, the zero-inflated model's :math:`P_1 C`
        and :math:`(1 - P_1) + P_1 C`.
        """
        from scipy.special import gammaln

        from ...samplers._utils._latent_loglik import (
            LOGIT,
            NEGBIN,
            POISSON,
            TRUNCATED,
            UNOBSERVED,
            integrated_loglik,
        )
        from ...samplers.negbin_latent._core import filter_columns

        y = np.asarray(self._y, dtype=np.float64)
        n = y.size
        pos = y > 0
        X = np.asarray(self._X, dtype=np.float64)
        Z = np.asarray(self._Z, dtype=np.float64)
        W = self._W_sparse.tocsr()
        W_zero = self._W_zero_sparse.tocsr()
        cols, cols_zero = filter_columns(W), filter_columns(W_zero)
        hurdle = self._two_part == "hurdle"
        kind = (
            np.where(pos, TRUNCATED, UNOBSERVED) if hurdle else np.full(n, POISSON)
        ).astype(np.int8)
        lgy = gammaln(y + 1.0)
        none = np.zeros(n)

        def disturbances(latent, par, mean, Wp):
            return np.ascontiguousarray(
                latent - par[:, None] * np.asarray(Wp @ latent.T).T - mean
            )

        out = []
        for r in results:
            u = np.ascontiguousarray(r["latent_zero"], dtype=np.float64)
            rz = np.asarray(r["rho_zero"], dtype=np.float64)
            ez = disturbances(u, rz, np.asarray(r["beta_zero"]) @ Z.T, W_zero)

            def zero_side(side):
                return integrated_loglik(
                    u, ez, side, *cols_zero, rz, np.ones_like(rz), LOGIT, none, kind
                )

            eta = np.ascontiguousarray(r["latent"], dtype=np.float64)
            rc = np.asarray(r["rho"], dtype=np.float64)
            e = disturbances(eta, rc, np.asarray(r["beta"]) @ X.T, W)
            alpha = np.asarray(r["alpha"], dtype=np.float64)
            count = integrated_loglik(eta, e, y, *cols, rc, alpha, NEGBIN, lgy, kind)
            if hurdle:
                out.append(zero_side(pos.astype(np.float64)) + count)
            else:
                p1, p0 = zero_side(np.ones(n)), zero_side(none)
                out.append(np.where(pos, p1 + count, np.logaddexp(p0, p1 + count)))
        return np.stack(out)

    # ------------------------------------------------------------------
    # Impacts
    # ------------------------------------------------------------------

    def _two_part_effects(self, equation: str, scale: Optional[str]):
        if self._design != "latent":
            return super()._two_part_effects(equation, scale)
        link, outcome = PART_SCALES[equation]
        if scale in (None, link):
            return self._compute_impacts_posterior(equation=equation)
        if scale != outcome:
            raise ValueError(
                f"scale must be {link!r} or {outcome!r} for equation {equation!r}."
            )
        direct, total = self._latent_part_effects(equation)
        return direct, total - direct, total

    def _latent_part_effects(self, equation: str, n_mc: int = 128, n_probes: int = 64):
        r"""Posterior draws ``(direct, total)`` of one part's outcome-scale effects.

        The zero part's outcome is :math:`P(u_i > 0)`, the count part's the
        expected count of an active unit, :math:`E e^{\eta_i}` (ZINB), or of a
        positive count, :math:`E\,g(e^{\eta_i})`, :math:`g` the truncated
        Poisson's mean (hurdle); see
        :func:`~neighbayes.models.cross_section._latent.latent_outcome_effects`.
        """
        from ._latent import latent_outcome_effects
        from ._latent_logit import LatentLogitMixin
        from ._latent_negbin import LatentNegBinMixin

        post = self.inference_data.posterior
        zero = equation == "zero"
        name = "_zero" if zero else ""
        par = post[f"rho{name}"].values.reshape(-1)
        X = self._Z if zero else self._X
        beta = post[f"beta{name}"].values.reshape(par.size, X.shape[1])
        W = (self._W_zero_sparse if zero else self._W_sparse).tocsr()
        if zero:
            extra = {}

            def noise(rng, size, params):
                return rng.logistic(size=size)

            weight = LatentLogitMixin._latent_outcome_weight
        else:
            extra = {"alpha": post["alpha"].values.reshape(-1)}

            def noise(rng, size, params):
                return _log_gamma_noise(rng, float(params["alpha"]), size)

            if self._two_part == "zinb":
                weight = LatentNegBinMixin._latent_outcome_weight
            else:

                def weight(m, U, E, d, params, exact):
                    lam = np.exp(np.clip(m[:, None] + U, -30, 30))
                    return np.mean(_truncated_slope(lam), axis=1)

        own_W = not zero or self._same_W
        eigen = None
        if W.shape[0] <= self._LATENT_EFFECTS_EIGEN_MAX_N:
            if own_W:
                eigs, V, Vinv = self._W_eigendecomposition
            else:
                eigs, V = np.linalg.eig(W.toarray())
                Vinv = np.linalg.inv(V)
            eigen = (eigs, V * Vinv.T)
        row_std = self._is_sel_row_std if zero else self._is_row_std
        return latent_outcome_effects(
            par=par,
            beta=beta,
            offset=beta @ X.T,
            extra=extra,
            W=W,
            sar=True,
            ni=list(
                self._sel_nonintercept_indices if zero else self._nonintercept_indices
            ),
            noise=noise,
            weight=weight,
            eigen=eigen,
            row_sums=self._multiplier_row_sums if own_W and row_std else None,
            n_mc=n_mc,
            n_probes=n_probes,
        )

    # ------------------------------------------------------------------
    # Simulated summaries
    # ------------------------------------------------------------------

    def _latent_part_sims(
        self, g: int, n_sims: int, rng
    ) -> tuple[np.ndarray, np.ndarray]:
        """Fresh utilities and log-means at posterior draw ``g``, each ``(n, n_sims)``."""
        n = self._X.shape[0]
        eye = sp.identity(n, format="csc")

        def solve(W, par, rhs):
            return sp.linalg.splu((eye - par * W).tocsc()).solve(rhs)

        draw = {
            name: self._flat_draw(name)[g]
            for name in ("rho_zero", "beta_zero", "rho", "beta", "alpha")
        }
        U = solve(
            self._W_zero_sparse,
            float(draw["rho_zero"]),
            (self._Z @ draw["beta_zero"])[:, None] + rng.logistic(size=(n, n_sims)),
        )
        H = solve(
            self._W_sparse,
            float(draw["rho"]),
            (self._X @ draw["beta"])[:, None]
            + _log_gamma_noise(rng, float(draw["alpha"]), (n, n_sims)),
        )
        return U, H

    def _latent_average(self, fn, draws: Optional[int], seed: int = 0) -> np.ndarray:
        """Posterior mean over ``draws`` draws of ``fn(U, H)`` (simulated parts)."""
        self._require_fit()
        rng = np.random.default_rng(seed)
        idx = self._draw_indices(draws)
        total = 0.0
        for g in idx:
            total = total + fn(*self._latent_part_sims(int(g), _SUMMARY_SIMS, rng))
        return total / len(idx)

    def activation_probabilities(
        self, draws: Optional[int] = _SUMMARY_DRAWS
    ) -> np.ndarray:
        """Posterior mean of each ``P(u > 0)``: an active unit (ZINB), a positive count (hurdle).

        The latent model's averages the propagated disturbances by simulation.
        """
        if self._design != "latent":
            return super().activation_probabilities(draws)
        return self._latent_average(lambda U, H: (U > 0).mean(axis=1), draws)

    def _fitted_mean_from_posterior_latent(self) -> np.ndarray:
        """Posterior mean of ``E[y]``, the propagated disturbances simulated."""
        if self._two_part == "hurdle":
            return self._latent_average(
                lambda U, H: (
                    (U > 0).mean(axis=1)
                    * _truncated_mean(np.exp(np.clip(H, -30, 30))).mean(axis=1)
                ),
                _SUMMARY_DRAWS,
            )
        return self._latent_average(
            lambda U, H: (
                (U > 0).mean(axis=1) * np.exp(np.clip(H, -30, 30)).mean(axis=1)
            ),
            _SUMMARY_DRAWS,
        )

    def _posterior_predictive_latent(
        self,
        n_draws: Optional[int] = None,
        random_seed: Optional[int] = None,
    ) -> np.ndarray:
        """Posterior-predictive counts, ``(n_draws, n_obs)``: one fresh simulation per draw."""
        self._require_fit()
        total = self._flat_draw("alpha").shape[0]
        total = total if n_draws is None else min(int(n_draws), total)
        rng = np.random.default_rng(random_seed)
        out = np.empty((total, self._X.shape[0]))
        for g in range(total):
            U, H = self._latent_part_sims(g, 1, rng)
            lam = np.exp(np.clip(H[:, 0], -30.0, 30.0))
            on = U[:, 0] > 0
            y = np.zeros(lam.size)
            if self._two_part == "hurdle":
                y[on] = _draw_truncated_poisson(rng, lam[on])
            else:
                y[on] = rng.poisson(lam[on])
            out[g] = y
        return out


class LatentHurdleMixin(LatentTwoPartMixin):
    """``SARHurdleNegBin(latent=True)``: priors and the positive count's mean."""

    _two_part = "hurdle"

    def _latent_two_part_priors(self):
        from ..priors import LogitGibbsPriors, ReducedGibbsPriors

        beta_mu, beta_sigma = self._hurdle_beta_prior(self._X, self._feature_names)
        gamma_mu, gamma_sigma = self._hurdle_gamma_prior(
            self._Z, self._sel_feature_names
        )
        a_sigma, a_nu, a_fixed = self._hurdle_alpha()
        lo, hi = self._bounds()
        lo_z, hi_z = self._bounds(zero=True)
        return (
            ReducedGibbsPriors(
                beta_mu=beta_mu,
                beta_sigma=beta_sigma,
                alpha_sigma=a_sigma,
                alpha_nu=a_nu,
                alpha_fixed=a_fixed,
                rho_lower=lo,
                rho_upper=hi,
            ),
            LogitGibbsPriors(
                beta_mu=gamma_mu,
                beta_sigma=gamma_sigma,
                rho_lower=lo_z,
                rho_upper=hi_z,
            ),
        )

    def positive_mean(self, draws: Optional[int] = _SUMMARY_DRAWS) -> np.ndarray:
        """Posterior mean of ``E[y | y > 0]`` for every unit.

        The latent model's is the truncated Poisson's mean averaged over the
        propagated gamma disturbances, by simulation.
        """
        if self._design != "latent":
            return super().positive_mean(draws)
        return self._latent_average(
            lambda U, H: _truncated_mean(np.exp(np.clip(H, -30, 30))).mean(axis=1),
            draws,
        )


class LatentZINBMixin(LatentTwoPartMixin):
    """``SARZINB(latent=True)``: priors and the zeros' attribution."""

    _two_part = "zinb"

    def _latent_two_part_priors(self):
        from ..priors import LogitGibbsPriors, ReducedGibbsPriors

        p = self._zinb_priors()
        return (
            ReducedGibbsPriors(
                beta_mu=p.beta_mu,
                beta_sigma=p.beta_sigma,
                alpha_sigma=p.alpha_sigma,
                alpha_nu=p.alpha_nu,
                alpha_fixed=p.alpha_fixed,
                rho_lower=p.rho_lower,
                rho_upper=p.rho_upper,
            ),
            LogitGibbsPriors(
                beta_mu=p.gamma_mu,
                beta_sigma=p.gamma_sigma,
                rho_lower=p.lam_lower,
                rho_upper=p.lam_upper,
            ),
        )

    def zero_attribution(self, draws: Optional[int] = _SUMMARY_DRAWS) -> dict:
        """Posterior split of each observed zero into structural and sampling.

        The latent model's is the sampler's: the average, over the sweeps after
        warmup, of each zero's conditional probability that its unit is
        inactive (``draws`` does not apply).
        """
        if self._design != "latent":
            return super().zero_attribution(draws)
        self._require_fit()
        mean, sd = self._latent_structural
        return {
            "structural_prob": mean,
            "structural_prob_sd": sd,
            "sampling_prob": 1.0 - mean,
            "zero_indices": np.flatnonzero(np.asarray(self._y) == 0),
        }
