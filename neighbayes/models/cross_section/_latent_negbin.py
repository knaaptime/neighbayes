r"""The latent SAR, SEM and SDEM negative binomial.

Shared by the unmarked :class:`SARNegBin` (:math:`(I - \rho W)\eta = X\beta +
\varepsilon`), :class:`SEMNegBin` (:math:`(I - \lambda W)(\eta - X\beta) =
\varepsilon`) and :class:`SDEMNegBin`, with :math:`y_i \sim
\mathrm{Poisson}(e^{\eta_i})` and :math:`\varepsilon_i = \log\gamma_i`,
:math:`\gamma_i \sim \mathrm{Gamma}(\alpha, \alpha)`: the negative binomial's
own disturbance, so that no dependence is :class:`NegBin`.  The sampler is
:mod:`neighbayes.samplers.negbin_latent`.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from scipy.special import digamma, gammaln, polygamma

# Importing the sampler registers its Gibbs entry.
from ...samplers._utils._latent_loglik import log_gamma_noise
from ...samplers._utils._sparsax_utils import (
    _NODE_BATCH,
    CachedSparseSolver,
    ResolventInterpolant,
    _lagrange_weights,
    _lobatto,
    fold_resolvent_columns,
)
from ...samplers.negbin_latent._core import LatentNegBinChain, zero_set_logdet
from .._base._nb import nb_alpha_fixed
from ..priors import ReducedGibbsPriors
from ._latent import (
    _OUTCOME_CHUNK,
    LatentMixin,
    defined_effects,
    period_columns,
    stacked_columns,
    weighted_effects,
)

#: Keeps an expected count finite; no floor, since small ``alpha`` and large ``rho``
#: put a true log-mean tens of units below ``m``.
_MAX_LOG_COUNT = 700.0
#: Entries within this fraction of ``alpha`` take the moment's series.  Its first
#: neglected term, ``(0.002)^5 / 5``, must stay far below the table's tolerance:
#: entries cross this line as ``rho`` moves, and a jump the interpolant can see
#: keeps its nodes doubling.
_SERIES_RADIUS = 0.002


class _UndefinedMoment(ArithmeticError):
    """A node of the (rho, alpha) table has no moment."""


def log_gamma_moment(s: np.ndarray, alpha: float) -> np.ndarray:
    r"""``log E γ^s``, ``γ ~ Gamma(α, α)``: :math:`\log\Gamma(\alpha + s) - \log\Gamma(\alpha) - s\log\alpha`.

    Infinite for :math:`s \le -\alpha`, which a negative multiplier entry reaches
    at small :math:`\alpha`: the model then has no expected count there.
    """
    out = gammaln(alpha + s) - gammaln(alpha) - s * np.log(alpha)
    return np.where(alpha + s > 0.0, out, np.inf)


def log_moment_sums(C: np.ndarray, alpha) -> np.ndarray:
    r"""``(len(alpha), n)``: :math:`\sum_j \log E\gamma^{C_{ij}}` per row at each ``alpha``.

    Most entries of a sparse graph's multiplier are far inside the radius
    :math:`\alpha` of the moment's series, whose terms there fall as
    :math:`(s/\alpha)^r / r`.  They share four power sums, computed once for all
    ``alpha``; the few large ones go through :func:`log_gamma_moment`.
    """
    alpha = np.atleast_1d(np.asarray(alpha, dtype=np.float64))
    tiny = np.abs(C) < _SERIES_RADIUS * alpha.min()
    small = np.where(tiny, C, 0.0)
    squared = small * small
    powers = np.stack(
        [
            small.sum(axis=1),
            squared.sum(axis=1),
            (squared * small).sum(axis=1),
            (squared * squared).sum(axis=1),
        ]
    )
    coef = np.stack(
        [
            digamma(alpha) - np.log(alpha),
            polygamma(1, alpha) / 2.0,
            polygamma(2, alpha) / 6.0,
            polygamma(3, alpha) / 24.0,
        ],
        axis=1,
    )
    out = coef @ powers
    rows, cols = np.nonzero(~tiny)
    large = C[rows, cols]
    for k, a in enumerate(alpha):
        out[k] += np.bincount(
            rows, weights=log_gamma_moment(large, a), minlength=C.shape[0]
        )
    return out


def expected_count(log_mean: np.ndarray) -> np.ndarray:
    """``exp(log_mean)`` capped against overflow, with an undefined mean left infinite."""
    return np.where(
        np.isposinf(log_mean), np.inf, np.exp(np.minimum(log_mean, _MAX_LOG_COUNT))
    )


class _AlphaGrid:
    r"""Chebyshev nodes in :math:`\log\alpha` over the draws, as many as ``log_gamma_moment`` needs.

    They are chosen on the moment itself, over entries ``s`` up to the
    multiplier's largest, :math:`1/(1 - |\rho|)`.  A held ``alpha`` has one node.
    """

    def __init__(self, alpha: np.ndarray, rho: np.ndarray, tol: float = 1e-10):
        u = np.log(alpha)
        self._mid = 0.5 * (u.max() + u.min())
        self._half = 0.5 * (u.max() - u.min())
        if self._half < 1e-12:
            self.alpha = np.array([float(alpha[0])])
            return
        s = np.geomspace(1e-3, 1.0 / (1.0 - min(float(np.abs(rho).max()), 0.999)), 12)

        def table(x):
            return (
                gammaln(np.exp(self._mid + self._half * x)[:, None] + s)
                - gammaln(np.exp(self._mid + self._half * x)[:, None])
                - s * (self._mid + self._half * x)[:, None]
            )

        K = 5
        while K < 129:
            fresh = _lobatto(2 * K - 1)[1::2]
            gap = _lagrange_weights(_lobatto(K), fresh) @ table(_lobatto(K)) - table(
                fresh
            )
            if np.max(np.abs(gap)) <= tol * np.max(np.abs(table(_lobatto(K)))):
                break
            K = 2 * K - 1
        self._nodes = _lobatto(K)
        self.alpha = np.exp(self._mid + self._half * self._nodes)

    def weights(self, alpha: np.ndarray) -> np.ndarray:
        """``(len(alpha), K)``: the weights that combine node values at each ``alpha``."""
        if self.alpha.size == 1:
            return np.ones((alpha.size, 1))
        return _lagrange_weights(self._nodes, (np.log(alpha) - self._mid) / self._half)


def log_moment_table(W, rho: np.ndarray, alpha: np.ndarray, **carry):
    r"""``(F, at)``: :math:`\sum_j \log E\gamma^{S_{ij}(\rho)}` across the draws.

    ``F`` is the :class:`ResolventInterpolant` over the draws of ``rho``, carrying
    ``carry`` too.  ``at(i)`` gives the sums, ``(len(i), n)``, at draws ``i``.  If
    any node of the table has no moment, which need not be a draw (the corner of
    the most negative ``rho`` and the smallest ``alpha``), each draw is summed
    directly.
    """
    grid = _AlphaGrid(alpha, rho)

    def fold(C, _):
        out = log_moment_sums(C, grid.alpha)
        if not np.isfinite(out).all():
            raise _UndefinedMoment
        return out

    try:
        F = ResolventInterpolant(W, rho, column_fold=fold, tol=1e-9, **carry)
    except _UndefinedMoment:
        F = ResolventInterpolant(W, rho, **carry)
        solver = CachedSparseSolver([W], W.shape[0])

        def direct(i):
            parts = []
            for a in range(0, len(i), _NODE_BATCH):
                b = i[a : a + _NODE_BATCH]
                parts.append(
                    fold_resolvent_columns(
                        solver,
                        rho[b],
                        lambda C, k, b=b: log_moment_sums(C, alpha[b[k]])[0],
                    )
                )
            return np.concatenate(parts)

        return F, direct
    table = F.values["fold"]
    Lr, La = F.weights(rho), grid.weights(alpha)

    def at(i):
        return np.sum(np.tensordot(Lr[i], table, axes=1) * La[i][:, :, None], axis=1)

    return F, at


def latent_count_effects(
    *, par, alpha, beta, offset, W, sar, ni, durbin=None, periods: int = 1
) -> tuple[np.ndarray, np.ndarray]:
    r"""Posterior draws ``(direct, total)`` of the expected count's effects, in closed form.

    With :math:`u_i = \sum_j S_{ij}\varepsilon_j` a sum of independent
    :math:`\log\gamma_j`, the weight of :func:`._latent.weighted_effects` is

    .. math::

        w_i = \mathbb{E}\,e^{m_i + u_i}
            = \exp\Bigl(m_i + \sum_j \log\mathbb{E}\,\gamma^{S_{ij}}\Bigr),
        \qquad
        \log\mathbb{E}\,\gamma^s
            = \log\Gamma(\alpha + s) - \log\Gamma(\alpha) - s\log\alpha.

    Simulating this mean fails at small :math:`\alpha`, whatever the number of
    draws: :math:`\log\gamma` then has a mean near :math:`-17` (at
    :math:`\alpha = 0.05`) and rare draws carry the expectation, so 128 of them
    read 16-29% low at :math:`\rho = 0.7` and lose the effect entirely at 0.9.
    The sum needs every entry of :math:`S(\rho)`, but depends on the parameters
    only through :math:`(\rho, \alpha)`, so it is tabulated on a few of those and
    interpolated (:func:`log_moment_table`); only :math:`m_i` needs a solve per
    draw.  Arguments as :func:`._latent.latent_outcome_effects`.
    """
    G = par.size
    N = W.shape[0]
    F, log_moments = log_moment_table(W, par, alpha, diag=sar, row_sums=sar)
    L = F.weights(par)
    solver = CachedSparseSolver([W], N)
    chunk = _OUTCOME_CHUNK
    n_eff = len(durbin[1]) if durbin is not None else len(ni)
    direct = np.empty((G, n_eff))
    total = np.empty((G, n_eff))
    for start in range(0, G, chunk):
        idx = np.arange(start, min(start + chunk, G))
        m = offset[idx]
        d = rows = None
        if sar:
            rhs = m.reshape(idx.size, periods, N).transpose(0, 2, 1)
            sol = solver.solve_batch(-par[idx, None], rhs, pad_to=chunk)
            m = sol.transpose(0, 2, 1).reshape(idx.size, -1)
            d = np.tile(L[idx] @ F.values["diag"], periods)
            rows = np.tile(L[idx] @ F.values["row_sums"], periods)
        w = expected_count(m + np.tile(log_moments(idx), periods))
        with np.errstate(invalid="ignore"):
            direct[idx], total[idx] = weighted_effects(
                w, d, rows, beta[idx], ni, durbin
            )
    return defined_effects(direct, total)


class LatentNegBinMixin(LatentMixin):
    """The latent negative binomial's priors, samplers and fitted counts."""

    _latent_extra = ("alpha",)
    _latent_no_nuts = (
        "its likelihood integrates the propagated gamma disturbances over all "
        "n locations at once, which no NUTS model evaluates."
    )
    _latent_chain_class = LatentNegBinChain

    @property
    def _latent_chain_kw(self) -> dict:
        """The joint move's log-determinant on the zero counts' sites."""
        b = self._logdet_bounds
        zero = np.asarray(self._y).reshape(self._T if self._is_panel else 1, -1) == 0
        return {
            "zero_logdet": zero_set_logdet(
                self._W_sparse, zero, b.method, b.rho_min, b.rho_max
            )
        }

    @staticmethod
    def _latent_chains_jax(**kwargs):
        from ...samplers.negbin_latent._jax import run_chains_jax_latent_negbin

        return run_chains_jax_latent_negbin(**kwargs)

    def _latent_priors(self, X, names) -> ReducedGibbsPriors:
        bounds = self._logdet_bounds
        beta_mu, beta_sigma = self._resolved_beta_prior(
            design=X, feature_names=names, link="log"
        )
        return ReducedGibbsPriors(
            beta_mu=beta_mu,
            beta_sigma=beta_sigma,
            alpha_sigma=self.priors.get("alpha_sigma", 2.5),
            alpha_nu=self.priors.get("alpha_nu", 3.0),
            alpha_fixed=nb_alpha_fixed(self.priors),
            rho_lower=float(bounds.rho_min),
            rho_upper=float(bounds.rho_max),
        )

    @property
    def _latent_observed(self) -> np.ndarray:
        return self._y_int

    _index_scale = "logmean"
    _outcome_scale = "count"
    _latent_family = 0  # integrated_loglik's NEGBIN

    def impacts(self, return_posterior_samples: bool = False, scale: str = "logmean"):
        r"""Direct, indirect and total impacts on the log-mean or the count scale.

        ``scale="count"`` gives the effects on the expected count
        :math:`\mathbb{E}\,y_i = \mathbb{E}\,e^{\eta_i}`, which averages over the
        propagated gamma heterogeneity (estimated by Monte Carlo per posterior
        draw); ``"logmean"`` the effects on the latent log-mean.
        """
        return self._latent_impacts(scale, return_posterior_samples)

    def _latent_noise(self, rng, size, params) -> np.ndarray:
        """Log-gamma disturbances at one posterior draw's ``alpha``."""
        return log_gamma_noise(rng, float(params["alpha"]), size)

    def _latent_outcome_effects(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Posterior draws of the count-scale effects, in closed form.

        See :func:`latent_count_effects`.
        """
        post = self.inference_data.posterior
        direct, total = latent_count_effects(
            **self._latent_effect_args(), alpha=post["alpha"].values.reshape(-1)
        )
        return direct, total - direct, total

    def _fitted_mean_from_posterior_latent(self) -> np.ndarray:
        r"""Expected counts :math:`E[e^{\eta_i}]` at the posterior means, in closed form.

        The multiplier mixes the gamma disturbances, so the expected count is
        not :math:`e^{E\eta_i}`; see :func:`latent_count_effects`.
        """
        self._require_fit()
        par = float(self._posterior_mean(self._latent_param))
        alpha = float(self._posterior_mean("alpha"))
        Xb = self._latent_systematic(self._idata.posterior).mean(axis=0)
        N = self._W_sparse.shape[0]
        T = Xb.size // N
        solver = CachedSparseSolver([self._W_sparse], N)
        log_moments = fold_resolvent_columns(
            solver, [par], lambda C, _: log_moment_sums(C, alpha)[0]
        )[0]
        m = Xb
        if self._latent_process == "sar":
            x = solver.solve([-par], period_columns(Xb[:, None], N, T))
            m = stacked_columns(x, N, T).ravel()
        return expected_count(m + np.tile(log_moments, T))

    def _posterior_predictive_latent(
        self,
        n_draws: Optional[int] = None,
        random_seed: Optional[int] = None,
    ) -> np.ndarray:
        """Posterior-predictive counts, ``(n_draws, n_obs)``: one fresh simulation per draw.

        Each posterior draw's disturbances are drawn afresh and propagated
        through its own filter, and the counts are Poisson given the log-means.
        """
        self._require_fit()
        post = self._idata.posterior
        par = post[self._latent_param].values.reshape(-1)
        alpha = post["alpha"].values.reshape(-1)
        systematic = self._latent_systematic(post)
        total = par.size if n_draws is None else min(int(n_draws), par.size)
        rng = np.random.default_rng(random_seed)
        n = systematic.shape[1]
        N = self._W_sparse.shape[0]
        T = n // N
        solver = CachedSparseSolver([self._W_sparse], N)
        sar = self._latent_process == "sar"
        chunk = 64
        out = np.empty((total, n))
        for start in range(0, total, chunk):
            idx = np.arange(start, min(start + chunk, total))
            eps = [self._latent_noise(rng, n, {"alpha": alpha[g]}) for g in idx]
            rhs = np.stack(
                [
                    period_columns((systematic[g] + e if sar else e)[:, None], N, T)
                    for e, g in zip(eps, idx)
                ]
            )
            sol = solver.solve_batch(-par[idx, None], rhs, pad_to=chunk)
            for i, g in enumerate(idx):
                x = stacked_columns(sol[i], N, T).ravel()
                eta = x if sar else systematic[g] + x
                out[g] = rng.poisson(np.exp(np.clip(eta, -30.0, 30.0)))
        return out
