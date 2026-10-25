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
the first retained draw, and a record on the DataTree.  Adapting only
appends probes, and the Chebyshev-in-ρ series has the same length at any probe
count, so a compiled JAX step is reused without a retrace.
"""

from __future__ import annotations

import logging
import warnings

import numpy as np

from ._cheb_stochastic import (
    DEFAULT_PROBE_MIN,
    DEFAULT_PROBE_Z,
    ProbeCheck,
    cheb_stochastic_pool,
    pool_precompute,
    probe_cap,
    run_bar,
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
    bar : float or None, default None
        Target expected |bias| of the posterior mean of ρ, in posterior sd.
        ``None`` derives it from the run: ``1/√ESS``, the Monte Carlo error of
        the posterior mean the run will produce (:func:`._cheb_stochastic.run_bar`).
    z : float
        Safety divisor: the target is ``bar / z``.
    min_probes : int
        The pool warmup starts with.
    max_probes : int or None, default None
        The most the pool may grow to; ``None`` caps it by the lattice size
        (:func:`._cheb_stochastic.probe_cap`).
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
        bar: float | None = None,
        z: float = DEFAULT_PROBE_Z,
        min_probes: int = DEFAULT_PROBE_MIN,
        max_probes: int | None = None,
        seed: int = 0,
    ):
        self.W = W
        self.T = int(T)
        self.rho_min, self.rho_max = float(rho_min), float(rho_max)
        self.bar = None if bar is None else float(bar)
        self.z = float(z)
        self.min_probes = int(min_probes)
        self.max_probes = (
            probe_cap(W.shape[0]) if max_probes is None else int(max_probes)
        )
        self.seed = int(seed)
        self.pool = None
        self.probe_check: ProbeCheck | None = None
        self._coeffs = None  # (pool, its coefficients)

    @classmethod
    def for_sampler(
        cls, W_sparse, *, T: int = 1, jacobian_T: int | None = None, **kwargs
    ):
        """Build from a sampler's lag matrix; a panel's is ``I_T ⊗ W``, so take one block.

        ``T`` slices the block; ``jacobian_T`` (default ``T``) multiplies the
        Jacobian, ``T - 1`` for a unit fixed-effects panel (Lee & Yu 2010).
        """
        T = int(T)
        n_units = W_sparse.shape[0] // T
        W = W_sparse[:n_units, :n_units] if T > 1 else W_sparse
        return cls(W, T=T if jacobian_T is None else int(jacobian_T), **kwargs)

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

    def adapt(
        self, warmup_draws, *, draws: int, chains: int, jax: bool = False
    ) -> Evaluators | None:
        """Size the pool on the warmup posterior; ``None`` keeps the installed evaluators.

        ``warmup_draws`` is ``(chains, iters)``, the settled warmup draws of ρ
        with the chains apart, or ``(m, chains, iters)`` for ``m`` spatial
        parameters on this W, such as a two-part model's two; the pool is sized
        for each in turn, and only grows.  ``draws`` and ``chains`` are the
        run's, which set each bar unless one was given.  The check recorded is
        the one nearest its target.
        """
        sets = np.asarray(warmup_draws, dtype=np.float64)
        sets = sets.reshape((-1,) + np.atleast_2d(sets).shape[-2:])
        pool, checks = self.pool, []
        for by_chain in sets:
            bar = self.bar if self.bar is not None else run_bar(by_chain, draws, chains)
            pool, c = size_pool(
                pool,
                by_chain.ravel(),
                bar=bar,
                z=self.z,
                max_probes=self.max_probes,
                T=self.T,
            )
            if c is not None:
                checks.append(c)
        check = max(checks, key=lambda c: c.bias * c.z / c.bar, default=None)
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

    def record(self, idata, retained_draws, *, suffix: str = "") -> None:
        """Attach the probe count and, when it was sized, how, to ``idata``.

        ``suffix`` ends each name, for a model with a second W.
        """
        if self.pool is not None:
            idata.attrs[f"logdet_probes{suffix}"] = int(self.pool.n_probes)
        chk = self.probe_check
        if chk is None:
            return
        idata.attrs[f"logdet_probe_bias{suffix}"] = float(chk.bias)
        idata.attrs[f"logdet_probe_spread{suffix}"] = float(chk.spread)
        idata.attrs[f"logdet_probe_bar{suffix}"] = float(chk.bar)
        idata.attrs[f"logdet_probe_z{suffix}"] = float(chk.z)
        idata.attrs[f"logdet_probe_capped{suffix}"] = int(chk.capped)

    def coefficients(self):
        """``(coeffs, lo, hi)``: the current pool's Chebyshev series in ρ, of one period."""
        from ._factories import _cheb_stochastic_coeffs_from

        if self._coeffs is None or self._coeffs[0] is not self.pool:
            self._coeffs = (
                self.pool,
                _cheb_stochastic_coeffs_from(
                    pool_precompute(self.pool), self.rho_min, self.rho_max
                ),
            )
        return self._coeffs[1]

    def period_fn(self):
        """``log|I - ρW|`` of one period from the current pool.

        For samplers that apply the panel's ``T`` themselves; the pool is still
        sized against the ``T``-period Jacobian.
        """
        coeffs, lo, hi = self.coefficients()
        return lambda r: clenshaw_scalar(coeffs, r, lo, hi, 1)

    def grad_vec_fn(self):
        """Vectorized ``g(ρ) = d/dρ log|I - ρW|`` of one period from the current pool.

        The impacts' direct-effect trace uses it.  The pool was sized to keep
        the log-determinant's slope error from moving the posterior of ρ, which
        is the error this gradient carries into the trace.
        """
        from ._resolvent import logdet_grad_chebyshev

        coeffs, lo, hi = self.coefficients()
        return lambda a: np.asarray(
            logdet_grad_chebyshev(np.asarray(a, dtype=np.float64), coeffs, lo, hi)
        )

    def _evaluators(self, jax: bool) -> Evaluators:
        coeffs, lo, hi = self.coefficients()
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
