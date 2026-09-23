"""The log-determinant a Gibbs sampler warms up on, and its adaptation halfway through warmup.

Warmup reveals where the posterior of the spatial parameter lies, and two
adaptations of a log-determinant interpolant use it:

* the refit (:mod:`._refit`) rebuilds the interpolant on the range the chains
  found, which needs fewer nodes and is more accurate there;
* the AAA node check sets the number of factorizations behind an AAA fit from how
  close the posterior lies to a singularity of the Jacobian.

Both change the transition kernel, so both run once, halfway through warmup, on
draws pooled across chains, and are frozen before the first retained draw, the
same discipline as step-size or slice-width adaptation.  :class:`WarmupJacobian`
holds them for any Gibbs sampler, which

1. installs :meth:`WarmupJacobian.initial`, the evaluators warmup starts on;
2. runs the first half of warmup on every chain;
3. passes the settled draws of all chains to :meth:`WarmupJacobian.adapt` and
   resumes every chain under what it returns, if anything;
4. calls :meth:`WarmupJacobian.record` on the assembled InferenceData.

A model asks :func:`sampler_builds_evaluators` whether to hand the sampler
evaluators of its own, so the model and the sampler never disagree about who
builds them.
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from ._refit import (
    AAA_METHODS,
    AAA_PILOT_NODES,
    DEFAULT_PAD_SD,
    REFITTABLE_METHODS,
    AAACheck,
    LogdetRefitter,
    RefitWindow,
    aaa_check_region,
    boundary_warning,
)

_log = logging.getLogger(__name__)

_NO_REACH = (float("nan"), float("nan"))


def sampler_builds_evaluators(
    method: str | None,
    has_W: bool,
    refit: bool,
    aaa_check: bool,
    probe_check: bool = False,
) -> bool:
    """Whether a Gibbs sampler with these settings builds its own log-determinant evaluators.

    When it does, evaluators a model built for it would never be evaluated.  The
    probe check (:mod:`._probe_check`) builds its own for ``cheb_stochastic``.
    """
    if not has_W or method is None:
        return False
    return bool(
        (refit and method in REFITTABLE_METHODS)
        or (aaa_check and method in AAA_METHODS)
        or (probe_check and method == "cheb_stochastic")
    )


@dataclass(frozen=True)
class Evaluators:
    """Log-determinant evaluators from one fit, and the interval they are valid on.

    ``params`` holds the same fit as fixed-shape JAX arrays, for a compiled step that
    carries the interpolant as traced state; it is ``None`` unless requested.
    """

    scalar_fn: Callable[[float], float]
    vec_fn: Callable[[np.ndarray], np.ndarray]
    rho_min: float
    rho_max: float
    params: Any = None


class WarmupJacobian:
    """Log-determinant evaluators for Gibbs warmup, and their adaptation at its midpoint.

    Parameters
    ----------
    W : scipy.sparse matrix
        The per-period ``N × N`` weights.  :meth:`for_sampler` starts instead from a
        sampler's lag matrix, which for a panel is block-diagonal.
    method : str
        Resolved log-determinant method.
    T : int, default 1
        Panel replication factor applied to every evaluator.
    rho_min, rho_max : float
        Prior support of the spatial parameter, clamped away from ``±1``.
    refit : bool, default True
        Rebuild the interpolant on the warmup range.  Applies to the refittable
        methods only.
    refit_pad_sd : float
        Padding of the refit window, in warmup standard deviations.
    aaa_check : bool, default True
        Set an AAA fit's node count from where the warmup posterior lies.  Applies
        to the AAA methods only.

    Attributes
    ----------
    window : RefitWindow or None
        The refit performed, if any.
    check : AAACheck or None
        The outcome of the AAA node check, or the static count used when there was
        no warmup midpoint to check at.
    """

    def __init__(
        self,
        W,
        method: str,
        *,
        T: int = 1,
        rho_min: float = -0.99,
        rho_max: float = 0.99,
        refit: bool = True,
        refit_pad_sd: float = DEFAULT_PAD_SD,
        aaa_check: bool = True,
    ):
        from ._chol_cheb import _clamp_interval

        self.method = str(method)
        self.T = int(T)
        lo, hi = _clamp_interval(rho_min, rho_max)
        self.prior = (float(lo), float(hi))
        fitter = LogdetRefitter(W, self.method, T=self.T)
        self._fitter = fitter if fitter.supported else None
        if refit and self._fitter is None:
            _log.info(
                f"logdet_refit requested but method {self.method!r} does not support "
                "it (no reusable factorization or no ρ interval); continuing with the "
                "prior interval."
            )
        self.refit = bool(refit) and self._fitter is not None
        self.aaa_check = (
            bool(aaa_check) and self._fitter is not None and self.method in AAA_METHODS
        )
        self.refit_pad_sd = float(refit_pad_sd)
        self.window: RefitWindow | None = None
        self.check: AAACheck | None = None
        self._interval = self.prior

    @classmethod
    def for_sampler(cls, W_sparse, logdet_method: str | None, *, T: int = 1, **kwargs):
        """Build from a sampler's lag matrix, resolving the method as the sampler does.

        A panel sampler receives the ``NT × NT`` block-diagonal lag matrix
        ``I_T ⊗ W``, whose determinant already carries the ``T`` replication.  The
        interpolant is built from the per-period block with ``T`` reapplied; built
        from the full matrix, it would count ``T`` twice.
        """
        from ._config import resolve_logdet_method

        T = int(T)
        n_units = W_sparse.shape[0] // T
        W = W_sparse[:n_units, :n_units] if T > 1 else W_sparse
        method = resolve_logdet_method(logdet_method, n=n_units, W=W)
        return cls(W, method, T=T, **kwargs)

    @property
    def supported(self) -> bool:
        """Whether the method has a reusable factorization to fit evaluators from."""
        return self._fitter is not None

    @property
    def active(self) -> bool:
        """Whether either adaptation applies to this method."""
        return self.refit or self.aaa_check

    def adapts(self, tune: int) -> bool:
        """Whether a warmup of ``tune`` iterations has a midpoint to adapt at."""
        return self.active and int(tune) >= 2

    @property
    def capacity(self) -> int:
        """Parameter capacity shared by every fit of the run; see :meth:`LogdetRefitter.capacity`."""
        return self._fitter.capacity(*self.prior)

    def param_fn(self):
        """The JAX evaluator ``(rho, params) -> log|I - rho W|`` for :attr:`Evaluators.params`."""
        from ._jax import make_logdet_jax_param_fn

        return make_logdet_jax_param_fn(self.method, T=self.T)

    def initial(self, tune: int, *, jax: bool = False) -> Evaluators:
        """The evaluators warmup starts on, fitted on the prior support.

        With a warmup midpoint ahead, the fit is deliberately cheap: the AAA pilot
        when the node check will run, the loose scouting fit when the refit will
        replace a Chebyshev interpolant and scouting is cheaper, and otherwise the
        prior interval's full fit.  Without one (``tune < 2``) nothing adapts, so
        the fit is the prior interval's full fit, and for an AAA method the node
        count used is recorded as the check's outcome.
        """
        if self._fitter is None:
            raise RuntimeError(
                f"Method {self.method!r} has no refittable fit to build evaluators from."
            )
        fitter, cap = self._fitter, self.capacity
        lo, hi = self.prior
        self.window, self.check, self._interval = None, None, self.prior
        if not self.adapts(tune):
            pre, _, _ = fitter._fit(lo, hi, cap=cap)
            if self.aaa_check:
                self.check = AAACheck(
                    nodes=int(fitter.last_nodes),
                    support_points=len(pre.support_points),
                    region=None,
                    reach=_NO_REACH,
                    passed=False,
                )
            evaluators = self._evaluators(pre, cap, jax)
            fitter.release()
            return evaluators
        if self.aaa_check:
            pre = fitter.aaa_fit(lo, hi, AAA_PILOT_NODES)
        elif self.refit and fitter.scout_order(lo, hi) < cap:
            pre, order, _ = fitter._fit(lo, hi, tol=fitter.scout_tol)
            _log.info(
                f"logdet_refit: warmup on a {order}-node scouting interpolant "
                f"(against {cap} for the un-refitted run)"
            )
        else:
            pre, _, _ = fitter._fit(lo, hi, cap=cap)
        return self._evaluators(pre, cap, jax)

    def adapt(self, warmup_draws, *, jax: bool = False) -> Evaluators | None:
        """Adapt the installed fit to the warmup posterior; ``None`` keeps it.

        ``warmup_draws`` are the settled draws of the spatial parameter pooled across
        every chain.  The refit runs first, when it applies and its window is
        materially narrower than the interval in use; the returned interval is then
        the sampler's new support.  The AAA node check runs next, on whichever fit
        is installed.  Each adaptation runs once per fit, so the factorization
        context is released afterwards.
        """
        fitter, cap = self._fitter, self.capacity
        lo_p, hi_p = self.prior
        new = None
        if self.refit:
            window = fitter.plan(
                warmup_draws, lo_p, hi_p, *self._interval, pad_sd=self.refit_pad_sd
            )
            if window is not None:
                pre, order, err_est = fitter._fit(window[0], window[1], cap=cap)
                self.window = fitter._window(
                    pre,
                    order,
                    err_est,
                    lo_p,
                    hi_p,
                    (int(np.size(warmup_draws)), self.refit_pad_sd),
                )
                self._interval = (float(pre.rho_min), float(pre.rho_max))
                new = pre
                _log.info(f"logdet_refit: rebuilt Jacobian on {self.window}")
        if self.aaa_check:
            pre, changed = self._node_check(warmup_draws)
            if changed:
                new = pre
        fitter.release()
        return None if new is None else self._evaluators(new, cap, jax)

    def record(self, idata, retained_draws) -> None:
        """Attach the node check and the refit window to ``idata``.

        The refit window is the sampler's support, so retained draws that reach an
        edge the refit introduced mean the window was set too tight; that raises a
        warning.
        """
        chk = self.check
        if chk is not None:
            idata.attrs["logdet_aaa_nodes"] = int(chk.nodes)
            idata.attrs["logdet_aaa_support_points"] = int(chk.support_points)
            if chk.region is not None:
                idata.attrs["logdet_aaa_region"] = [
                    float(chk.region[0]),
                    float(chk.region[1]),
                ]
                idata.attrs["logdet_aaa_reach"] = [
                    float(chk.reach[0]),
                    float(chk.reach[1]),
                ]
                idata.attrs["logdet_aaa_check_passed"] = int(chk.passed)
        info = self.window
        if info is None:
            return
        idata.attrs["logdet_refit_window"] = [info.rho_min, info.rho_max]
        idata.attrs["logdet_refit_order"] = info.order
        idata.attrs["logdet_refit_pad_sd"] = info.pad_sd
        idata.attrs["logdet_refit_err_est"] = info.err_est
        msg = boundary_warning(
            np.asarray(retained_draws, dtype=np.float64).ravel(), info
        )
        if msg is not None:
            warnings.warn(msg, RuntimeWarning, stacklevel=3)

    # ------------------------------------------------------------------

    def _node_check(self, warmup_draws):
        """Check the installed AAA fit against the warmup posterior, adding nodes if needed.

        Returns ``(precompute, changed)``.  When the draws cannot locate the
        posterior, the fit takes the interval's static count.  A fit that reaches the
        node cap without passing raises a warning, since the posterior then lies
        nearer a singularity than any fit the cap allows resolves.
        """
        from ._aaa import _adaptive_n_coarse

        fitter = self._fitter
        start = fitter.last_pre
        pre, nodes = start, int(fitter.last_nodes)
        lo, hi = float(pre.rho_min), float(pre.rho_max)
        region = aaa_check_region(warmup_draws, lo, hi)
        if region is None:
            target = _adaptive_n_coarse(lo, hi)
            if target > nodes:
                pre, nodes = fitter.aaa_fit(lo, hi, target), target
            self.check = AAACheck(
                nodes=nodes,
                support_points=len(pre.support_points),
                region=None,
                reach=_NO_REACH,
                passed=False,
            )
        else:
            pre, nodes, self.check = fitter.aaa_refine(
                pre, nodes, region, lo, hi, 2 * fitter.capacity(lo, hi)
            )
            if not self.check.passed:
                warnings.warn(
                    f"The warmup posterior of the spatial parameter, "
                    f"[{region[0]:.4f}, {region[1]:.4f}], lies nearer a singularity of "
                    f"the Jacobian than an AAA fit of {nodes} nodes resolves; the "
                    "log-determinant may bias the posterior there.",
                    RuntimeWarning,
                    stacklevel=4,
                )
        _log.info(
            f"logdet AAA check: {nodes} nodes, {len(pre.support_points)} support points"
        )
        return pre, pre is not start

    def _evaluators(self, pre, cap: int, jax: bool) -> Evaluators:
        scalar_fn, vec_fn = self._fitter._numpy_fns(pre)
        params = self._fitter.params_from(pre, cap) if jax else None
        return Evaluators(
            scalar_fn=scalar_fn,
            vec_fn=vec_fn,
            rho_min=float(pre.rho_min),
            rho_max=float(pre.rho_max),
            params=params,
        )
