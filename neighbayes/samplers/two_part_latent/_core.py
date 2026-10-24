r"""Gibbs samplers for the latent SAR hurdle and zero-inflated negative binomial.

Both models pair a latent SAR logit, the zero part, with a latent SAR negative
binomial, the count part:

.. math::
    A_z u = Z\gamma + \varepsilon_z, \qquad
    A\eta = X\beta + \varepsilon,

with :math:`\varepsilon_{z,i}` standard logistic and :math:`\varepsilon_i =
\log\gamma_i`, :math:`\gamma_i \sim \mathrm{Gamma}(\alpha, \alpha)`, each part's
disturbance propagated by its own filter.  They differ in what the zero part
decides.

The **hurdle**'s zero part is whether a count is positive, :math:`y_i > 0 \iff
u_i > 0`, and a positive count is :math:`y_i \mid \eta_i \sim
\mathrm{Poisson}(e^{\eta_i})` truncated at zero.  Its parts share no unknown,
so a sweep is one sweep of each sampler
(:class:`~neighbayes.samplers.logit_latent._core.LatentLogitChain` on
:math:`d = \mathbf 1(y > 0)`,
:class:`~neighbayes.samplers.negbin_latent._core.LatentNegBinChain` observing
the positive counts only).

The **zero-inflated** model's zero part is whether a unit is active,
:math:`d_i = \mathbf 1(u_i > 0)`: an inactive unit counts zero, an active one
:math:`\mathrm{Poisson}(e^{\eta_i})`.  A zero count leaves :math:`d_i` unknown,
so its utility's side of zero is drawn with it, given :math:`\eta_i`, the
positive side weighted by :math:`q_i = P(y_i = 0 \mid d_i = 1) =
e^{-e^{\eta_i}}`; the count part then observes the active units only.  Each
step draws from its full conditional, so the sweep is exact.  The posterior
probability that a zero is structural is the average, over the sweeps after
warmup, of the conditional probability of the negative side (Rao–Blackwell).
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import scipy.sparse as sp

from ...models.priors import LogitGibbsPriors, ReducedGibbsPriors
from .._utils._sweeps import run_sweeps
from ..logit_latent._core import LatentLogitChain
from ..negbin_latent._core import (
    POISSON,
    TRUNCATED,
    UNOBSERVED,
    LatentNegBinChain,
)


class _TwoPartChain:
    """A zero part and a count part; ``draw`` names the zero part's ``*_zero``."""

    zero: LatentLogitChain
    count: LatentNegBinChain

    def draw(self, store_latent: bool) -> dict:
        z = self.zero.draw(store_latent)
        out = {"rho_zero": z["rho"], "beta_zero": z["beta"]}
        out.update(self.count.draw(store_latent))
        if store_latent:
            out["latent_zero"] = z["latent"]
        return out

    def summary(self) -> dict:
        """What the chain accumulates beyond its draws."""
        return {}


class LatentHurdleChain(_TwoPartChain):
    """The latent hurdle: one sweep of each part (see the module)."""

    def __init__(
        self,
        *,
        y: np.ndarray,
        X: np.ndarray,
        Z: np.ndarray,
        W: sp.csr_matrix,
        W_zero: sp.csr_matrix,
        priors: ReducedGibbsPriors,
        priors_zero: LogitGibbsPriors,
        logdet_fn: Callable[[float], float],
        logdet_zero_fn: Callable[[float], float],
        rng: np.random.Generator,
        slice_width: float = 0.4,
    ):
        positive = np.asarray(y) > 0
        self.zero = LatentLogitChain(
            y=positive.astype(np.float64),
            X=Z,
            W=W_zero,
            process="sar",
            priors=priors_zero,
            logdet_fn=logdet_zero_fn,
            rng=rng,
            slice_width=slice_width,
        )
        self.count = LatentNegBinChain(
            y=y,
            X=X,
            W=W,
            process="sar",
            priors=priors,
            logdet_fn=logdet_fn,
            rng=rng,
            slice_width=slice_width,
            kind=np.where(positive, TRUNCATED, UNOBSERVED),
        )

    def sweep(self, tuning: bool) -> None:
        self.zero.sweep(tuning)
        self.count.sweep(tuning)


class LatentZINBChain(_TwoPartChain):
    """The latent zero-inflated NB: activation, then counts (see the module)."""

    def __init__(
        self,
        *,
        y: np.ndarray,
        X: np.ndarray,
        Z: np.ndarray,
        W: sp.csr_matrix,
        W_zero: sp.csr_matrix,
        priors: ReducedGibbsPriors,
        priors_zero: LogitGibbsPriors,
        logdet_fn: Callable[[float], float],
        logdet_zero_fn: Callable[[float], float],
        rng: np.random.Generator,
        slice_width: float = 0.4,
    ):
        y = np.asarray(y, dtype=np.float64)
        self.zeros = y == 0
        self.zero = LatentLogitChain(
            y=y,
            X=Z,
            W=W_zero,
            process="sar",
            priors=priors_zero,
            logdet_fn=logdet_zero_fn,
            rng=rng,
            slice_width=slice_width,
            sign=np.where(self.zeros, 0, 1),
        )
        self.count = LatentNegBinChain(
            y=y,
            X=X,
            W=W,
            process="sar",
            priors=priors,
            logdet_fn=logdet_fn,
            rng=rng,
            slice_width=slice_width,
        )
        self.structural = np.zeros(y.size)
        self.structural_sq = np.zeros(y.size)
        self.n_kept = 0

    def sweep(self, tuning: bool) -> None:
        count = self.count
        # q_i = P(y_i = 0 | active) = exp(-e^eta_i).
        self.zero.logq = -np.exp(count.eta.reshape(-1))
        self.zero.sweep(tuning)
        active = self.zero.ystar > 0
        count.kind = np.where(active, POISSON, UNOBSERVED).astype(np.int8)
        count.kind = count.kind.reshape(count.T, count.n)
        count.sweep(tuning)
        if not tuning:
            p = np.where(self.zeros, self.zero.pneg, 0.0)
            self.structural += p
            self.structural_sq += p * p
            self.n_kept += 1

    def summary(self) -> dict:
        """Each zero count's posterior probability of being structural, and its sd."""
        k = max(self.n_kept, 1)
        mean = self.structural[self.zeros] / k
        second = self.structural_sq[self.zeros] / k
        return {
            "structural_prob": mean,
            "structural_prob_sd": np.sqrt(np.maximum(second - mean**2, 0)),
        }


def run_chain(
    *,
    model: str,
    y: np.ndarray,
    X: np.ndarray,
    Z: np.ndarray,
    W: sp.csr_matrix,
    W_zero: sp.csr_matrix,
    priors: ReducedGibbsPriors,
    priors_zero: LogitGibbsPriors,
    logdet_fn: Callable[[float], float],
    logdet_zero_fn: Callable[[float], float],
    draws: int,
    tune: int,
    thin: int = 1,
    rng: np.random.Generator,
    chain_id: int = 0,
    progress_manager=None,
    slice_width: float = 0.4,
    store_latent: bool = False,
) -> dict:
    """One chain of the latent ``model`` (``"hurdle"`` or ``"zinb"``).

    Returns ``rho_zero``, ``beta_zero``, ``rho``, ``beta`` and ``alpha``;
    ``store_latent`` adds the utilities (``"latent_zero"``) and log-means
    (``"latent"``) at every kept draw.  The ZINB adds ``structural_prob`` and
    ``structural_prob_sd`` over the zero counts.
    """
    cls = {"hurdle": LatentHurdleChain, "zinb": LatentZINBChain}[model]
    chain = cls(
        y=y,
        X=X,
        Z=Z,
        W=W,
        W_zero=W_zero,
        priors=priors,
        priors_zero=priors_zero,
        logdet_fn=logdet_fn,
        logdet_zero_fn=logdet_zero_fn,
        rng=rng,
        slice_width=slice_width,
    )
    out = run_sweeps(
        chain,
        draws=draws,
        tune=tune,
        thin=thin,
        chain_id=chain_id,
        progress_manager=progress_manager,
        store_latent=store_latent,
    )
    out.update(chain.summary())
    return out
