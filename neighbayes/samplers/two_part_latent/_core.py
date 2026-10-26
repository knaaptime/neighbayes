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
(:class:`~neighbayes.samplers.binary_latent._core.LatentBinaryChain` on
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

A panel stacks ``T`` periods time-first: the count part is the latent NB's
panel (random unit effects inside its filter, period effects as columns of
``X``), the zero part the latent logit on :math:`I_T \otimes W_z` with
:math:`T \log|A_z|`, and the hurdle's zero part takes its own unit effects
(period effects as columns of ``Z``).
"""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np
import scipy.sparse as sp

from ...models.priors import BinaryGibbsPriors, ReducedGibbsPriors
from ..binary_latent._core import LatentBinaryChain
from ..negbin_latent._core import (
    POISSON,
    TRUNCATED,
    UNOBSERVED,
    LatentNegBinChain,
    UnitEffects,
)


def _zero_part(
    y, Z, W_zero, priors_zero, logdet_zero_fn, rng, slice_width, T, sign, effects
) -> LatentBinaryChain:
    """The zero part's latent logit, on ``I_T ⊗ W_z`` with ``T log|A_z|``."""
    if T > 1:
        W_zero = sp.kron(sp.identity(T, format="csr"), W_zero, format="csr")
        per_period = logdet_zero_fn

        def logdet_zero_fn(r):
            return T * per_period(r)

    return LatentBinaryChain(
        y=y,
        X=Z,
        W=W_zero,
        process="sar",
        priors=priors_zero,
        logdet_fn=logdet_zero_fn,
        rng=rng,
        slice_width=slice_width,
        sign=sign,
        unit_effects=effects,
    )


class _TwoPartChain:
    """A zero part and a count part; ``draw`` names the zero part's ``*_zero``."""

    zero: LatentBinaryChain
    count: LatentNegBinChain

    def draw(self, store_latent: bool) -> dict:
        out = {f"{k}_zero": v for k, v in self.zero.draw(store_latent).items()}
        out.update(self.count.draw(store_latent))
        return out

    def summary(self) -> dict:
        """What the chain accumulates beyond its draws."""
        return {}

    def set_logdet(self, fn: Callable[[float], float], *, zero: bool) -> None:
        """Install one part's log-determinant of one period."""
        if not zero:
            self.count.logdet_fn = fn
            return
        T = self.count.T
        self.zero.logdet_fn = fn if T == 1 else (lambda r: T * fn(r))


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
        priors_zero: BinaryGibbsPriors,
        logdet_fn: Callable[[float], float],
        logdet_zero_fn: Callable[[float], float],
        rng: np.random.Generator,
        slice_width: float = 0.4,
        T: int = 1,
        unit_effects: Optional[UnitEffects] = None,
        unit_effects_zero: Optional[UnitEffects] = None,
    ):
        positive = np.asarray(y) > 0
        self.zero = _zero_part(
            positive.astype(np.float64),
            Z,
            W_zero,
            priors_zero,
            logdet_zero_fn,
            rng,
            slice_width,
            T,
            None,
            unit_effects_zero,
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
            T=T,
            unit_effects=unit_effects,
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
        priors_zero: BinaryGibbsPriors,
        logdet_fn: Callable[[float], float],
        logdet_zero_fn: Callable[[float], float],
        rng: np.random.Generator,
        slice_width: float = 0.4,
        T: int = 1,
        unit_effects: Optional[UnitEffects] = None,
    ):
        y = np.asarray(y, dtype=np.float64)
        self.zeros = y == 0
        self.zero = _zero_part(
            y,
            Z,
            W_zero,
            priors_zero,
            logdet_zero_fn,
            rng,
            slice_width,
            T,
            np.where(self.zeros, 0, 1),
            None,
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
            T=T,
            unit_effects=unit_effects,
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


def make_chain(
    *,
    model: str,
    y: np.ndarray,
    X: np.ndarray,
    Z: np.ndarray,
    W: sp.csr_matrix,
    W_zero: sp.csr_matrix,
    priors: ReducedGibbsPriors,
    priors_zero: BinaryGibbsPriors,
    logdet_fn: Callable[[float], float],
    logdet_zero_fn: Callable[[float], float],
    rng: np.random.Generator,
    slice_width: float = 0.4,
    T: int = 1,
    unit_effects: Optional[UnitEffects] = None,
    unit_effects_zero: Optional[UnitEffects] = None,
) -> _TwoPartChain:
    """The chain object of the latent ``model`` (``"hurdle"`` or ``"zinb"``).

    ``W`` and ``W_zero`` are a period's weights, ``logdet_fn`` and
    ``logdet_zero_fn`` a period's log-determinants.  Under :func:`~neighbayes.samplers._utils._sweeps.run_sweeps` its draws are
    ``rho_zero``, ``beta_zero``, ``rho``, ``beta`` and ``alpha``;
    ``store_latent`` adds the utilities (``latent_zero``) and log-means
    (``latent``).  A panel's unit effects add ``unit_effect`` and
    ``unit_effect_sd`` (the hurdle's zero part's ``*_zero``).  Its
    ``summary()`` holds the ZINB's ``structural_prob`` and
    ``structural_prob_sd`` over the zero counts.
    """
    panel = dict(T=T, unit_effects=unit_effects)
    if model == "hurdle":
        panel["unit_effects_zero"] = unit_effects_zero
    elif unit_effects_zero is not None:
        raise ValueError("The zero-inflated model's zero part takes no unit effects.")
    cls = {"hurdle": LatentHurdleChain, "zinb": LatentZINBChain}[model]
    return cls(
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
        **panel,
    )
