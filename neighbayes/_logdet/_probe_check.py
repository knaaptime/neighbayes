"""The stochastic Chebyshev probe count, set halfway through Gibbs warmup.

A fixed probe count is set before anyone knows where the posterior lies.  In the
interior of the stationary interval 50 probes are more than enough; near its
boundary a large lattice needs a hundred or more.  Warmup shows where the
posterior lies, and the probes' own spread prices the bias each count leaves
there (:func:`._cheb_stochastic.size_pool`), so the pool starts at the fixed
count and grows where that count falls short.

:class:`WarmupProbes` offers a Gibbs sampler the steps of
:class:`._warmup.WarmupJacobian`: the evaluators warmup starts on, one
adaptation at the warmup midpoint on draws pooled across chains, frozen before
the first retained draw, and a record on the InferenceData.  Adapting only
appends probes, and the Chebyshev-in-ρ series has the same length at any probe
count, so a compiled JAX step is reused without a retrace.
"""

from __future__ import annotations

import logging
import warnings

from ._cheb_stochastic import (
    DEFAULT_PROBE_BAR,
    DEFAULT_PROBE_MAX,
    DEFAULT_PROBE_MIN,
    DEFAULT_PROBE_Z,
    ProbeCheck,
    cheb_stochastic_pool,
    pool_precompute,
    size_pool,
)
from ._clenshaw import clenshaw_scalar, clenshaw_vec
from ._warmup import Evaluators

_log = logging.getLogger(__name__)

#: Probes used when warmup has no midpoint to size the pool at: the count the
#: fixed stochastic Chebyshev estimator uses.
FIXED_PROBES = 50


class WarmupProbes:
    """Stochastic Chebyshev evaluators for Gibbs warmup, with the probe count set at its midpoint.

    Parameters
    ----------
    W : scipy.sparse matrix
        The per-period ``N × N`` weights.
    T : int, default 1
        Panel replication factor applied to every evaluator.
    rho_min, rho_max : float
        Prior support of the spatial parameter; the Chebyshev-in-ρ series is fit
        on it.
    bar, z : float
        Target expected |bias| of the posterior mean of ρ, ``bar / z`` posterior sd.
    min_probes, max_probes : int
        The pool warmup starts with, and the most it may grow to.
    seed : int, default 0
        Root of the probe streams, fixed so a fit is reproducible for a given W.

    Attributes
    ----------
    probe_check : ProbeCheck or None
        How the pool was sized, once :meth:`adapt` has run.
    window, check : None
        The refit window and AAA node check of :class:`._warmup.WarmupJacobian`;
        this adaptation has neither, and leaves the support unchanged.
    """

    method = "cheb_stochastic"
    window = None
    check = None

    def __init__(
        self,
        W,
        *,
        T: int = 1,
        rho_min: float = -0.99,
        rho_max: float = 0.99,
        bar: float = DEFAULT_PROBE_BAR,
        z: float = DEFAULT_PROBE_Z,
        min_probes: int = DEFAULT_PROBE_MIN,
        max_probes: int = DEFAULT_PROBE_MAX,
        seed: int = 0,
    ):
        self.W = W
        self.T = int(T)
        self.rho_min, self.rho_max = float(rho_min), float(rho_max)
        self.bar, self.z = float(bar), float(z)
        self.min_probes, self.max_probes = int(min_probes), int(max_probes)
        self.seed = int(seed)
        self.pool = None
        self.probe_check: ProbeCheck | None = None

    @classmethod
    def for_sampler(cls, W_sparse, *, T: int = 1, **kwargs):
        """Build from a sampler's lag matrix; a panel's is ``I_T ⊗ W``, so take one block."""
        T = int(T)
        n_units = W_sparse.shape[0] // T
        W = W_sparse[:n_units, :n_units] if T > 1 else W_sparse
        return cls(W, T=T, **kwargs)

    def adapts(self, tune: int) -> bool:
        """Whether a warmup of ``tune`` iterations has a midpoint to size the pool at."""
        return int(tune) >= 2

    def param_fn(self):
        """The JAX evaluator ``(rho, params) -> log|I - rho W|`` for :attr:`Evaluators.params`."""
        from ._jax import jax_logdet_chebyshev_traced

        T = self.T

        def _fn(rho, params):
            coeffs, rmin, rmax = params
            val = jax_logdet_chebyshev_traced(rho, coeffs, rmin, rmax)
            return val if T == 1 else T * val

        return _fn

    def initial(self, tune: int, *, jax: bool = False) -> Evaluators:
        """The evaluators warmup starts on, from ``min_probes`` probes.

        Without a midpoint to size the pool at, the pool is never smaller than the
        fixed estimator's.
        """
        start = self.min_probes
        if not self.adapts(tune):
            start = max(start, FIXED_PROBES)
        self.probe_check = None
        self.pool = cheb_stochastic_pool(
            self.W,
            start,
            max_probes=max(self.max_probes, start),
            rho_min=self.rho_min,
            rho_max=self.rho_max,
            seed=self.seed,
        )
        return self._evaluators(jax)

    def adapt(self, warmup_draws, *, jax: bool = False) -> Evaluators | None:
        """Size the pool on the warmup posterior; ``None`` keeps the installed evaluators."""
        pool, check = size_pool(
            self.pool,
            warmup_draws,
            bar=self.bar,
            z=self.z,
            max_probes=self.max_probes,
            T=self.T,
        )
        grew = pool is not self.pool
        self.pool, self.probe_check = pool, check
        if check is not None:
            _log.info(
                f"logdet probe check: {check.n_probes} probes, expected |bias| "
                f"{check.bias:.4f} sd against a target of {check.bar / check.z:.4f}"
            )
            if check.capped:
                warnings.warn(
                    f"The stochastic log-determinant reached its {check.n_probes}-probe "
                    f"cap with an expected bias of {check.bias:.3f} posterior sd in the "
                    f"posterior mean of the spatial parameter, above the target of "
                    f"{check.bar / check.z:.3f}.",
                    RuntimeWarning,
                    stacklevel=4,
                )
        return self._evaluators(jax) if grew else None

    def record(self, idata, retained_draws) -> None:
        """Attach the probe count and, when it was sized, how, to ``idata``."""
        if self.pool is not None:
            idata.attrs["logdet_probes"] = int(self.pool.n_probes)
        chk = self.probe_check
        if chk is None:
            return
        idata.attrs["logdet_probe_bias"] = float(chk.bias)
        idata.attrs["logdet_probe_spread"] = float(chk.spread)
        idata.attrs["logdet_probe_bar"] = float(chk.bar)
        idata.attrs["logdet_probe_z"] = float(chk.z)
        idata.attrs["logdet_probe_capped"] = int(chk.capped)

    def _evaluators(self, jax: bool) -> Evaluators:
        from ._factories import _cheb_stochastic_coeffs_from

        coeffs, lo, hi = _cheb_stochastic_coeffs_from(
            pool_precompute(self.pool), self.rho_min, self.rho_max
        )
        T = self.T
        params = None
        if jax:
            from .._jax_dispatch import ensure_x64

            ensure_x64()
            import jax.numpy as jnp

            params = (jnp.asarray(coeffs), jnp.float64(lo), jnp.float64(hi))
        return Evaluators(
            scalar_fn=lambda r: clenshaw_scalar(coeffs, r, lo, hi, T),
            vec_fn=lambda a: clenshaw_vec(coeffs, a, lo, hi, T),
            rho_min=lo,
            rho_max=hi,
            params=params,
        )
