"""Rebuild the Jacobian interpolant on the range a warmed-up sampler explores.

The interval an interpolant of ``log|I - ρW|`` is built on is normally taken
from the prior — often the whole stability region ``(-1, 1)``.  A sampler that
has finished warmup knows far more than that: the posterior of ρ is routinely one
to two orders of magnitude narrower than any prior an analyst would write down.

Two things follow, and the second is the one that matters.

**Nodes.**  The Chebyshev order needed is set by the interval's distance to the
``ρ = ±1`` singularities (:func:`~._chebyshev.cheb_order_for_tolerance`), so a
post-warmup window costs a fraction of the nodes.  On a rook lattice at
n = 10,000 the default ``[-0.99, 0.99]`` needs 117 nodes and ``[0.1, 0.8]``
needs 17, against 6 for a ``[0.55, 0.65]`` window.

**Accuracy.**  Refitting at a tight tolerance drives the error over the region
the posterior actually occupies down to the factorization's own roundoff floor.
That matters more than the node count, because what biases a posterior is not
the interpolant's maximum error but its *tilt* — the variation of the error
across the posterior's support.  An error that is constant in ρ adds a constant
to the log posterior and cancels exactly.  On the same lattice, refitting from
``[0.1, 0.8]`` at 17 nodes onto a ±10-sd window at 13 nodes takes the tilt from
~1e-5 to ~4e-12: fewer factorizations *and* six orders of magnitude less of the
quantity that moves inference.

Both depend on the refit being cheap, which is what
:class:`~._chol_cheb.CholChebContext` and :class:`~._aaa.AAAContext` provide:
the symmetrization and the symbolic factorization are paid once, so a second fit
costs only its numeric factorizations.

Validity
--------
Refitting narrows the interpolant's domain, and a Chebyshev series diverges
violently outside its interval, so the window becomes the sampler's support.
That is a real change to the target and this module treats it as one:

* the window is padded by a generous multiple of the warmup posterior's standard
  deviation (10 by default, at which the truncated tail is ~1e-23 under
  normality) and never widened beyond the prior;
* the refit must happen *during* warmup and be frozen before sampling begins —
  the same discipline as step-size, mass-matrix, or slice-width adaptation, all
  of which change the transition kernel and all of which stop before the draws
  that are kept;
* :func:`boundary_warning` reports whether the retained draws ever approached an
  edge that was actually truncated, which is the diagnostic that the window was
  set too tight.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

_log = logging.getLogger(__name__)

#: Methods that own a reusable factorization context and a ρ interval.
#: ``eigenvalue`` is excluded because it is already exact and interval-free;
#: the stochastic estimators are excluded because their error is dominated by
#: probe noise rather than by the interval, so narrowing it buys nothing.
REFITTABLE_METHODS = frozenset({"cheb_cholesky", "lu_cheb", "aaa", "chol_aaa"})

#: Default padding, in warmup posterior standard deviations.
DEFAULT_PAD_SD = 10.0

#: Default absolute error target for the refit.  Tight on purpose: on a narrow
#: window this costs no more nodes than the prior-interval fit and buys the
#: roundoff floor.
DEFAULT_REFIT_TOL = 1e-12

#: Absolute error target for the *scouting* interpolant — the one that carries
#: the chains through the first half of warmup, before the window is known.
#:
#: Deliberately loose, and this is where the refit stops being cost-neutral and
#: starts paying.  Scouting draws are discarded, so the scouting interpolant
#: only has to steer the chains into the right neighborhood; it does not have
#: to be the interpolant anyone reports from.  A target of one log-unit is
#: looser than every stochastic estimator this library offers, and those move a
#: posterior mean by under a tenth of a standard deviation even when they are
#: used for the *whole* run rather than discarded half of warmup.
#:
#: The saving is largest exactly where the un-refitted cost is worst.  On the
#: full stability region the order rule asks for 117 nodes at n = 10,000; a
#: scouting fit needs 37 and the refit that replaces it needs 13, so the run
#: pays 50 factorizations rather than 117.  On an interval already narrow enough
#: to be cheap there is correspondingly less to win, which is the right shape
#: for the trade.
DEFAULT_SCOUT_TOL = 1.0

#: Skip the refit unless the window is at least this much narrower than the
#: interval already in use — below that the accuracy gain does not repay the
#: extra factorizations.
MIN_NARROWING = 1.25

#: Narrowest window that will be accepted, in absolute ρ units.  A chain that
#: has stuck reports a standard deviation of order the floating-point residue
#: rather than exactly zero, which would otherwise pass the ``sd > 0`` check and
#: collapse the window — and hence the sampler's support — onto a point.
MIN_WINDOW_WIDTH = 1e-6

#: The AAA methods, whose node count the warmup check sets.
AAA_METHODS = frozenset({"aaa", "chol_aaa"})

#: Nodes of the AAA fit warmup starts on, before the posterior is located (7
#: support points).  In the calibration sweeps this count kept the bias an
#: estimate of ρ carries under 1e-4 for every true ``|ρ| ≤ 0.9``.
AAA_PILOT_NODES = 14

#: Nodes added each time the check fails: two more support points.
AAA_NODE_STEP = 4

#: A fit passes when the posterior region lies at least this many reaches from the
#: nearest singularity (see :func:`~._aaa.aaa_reach`).  No fit in the calibration
#: sweeps with at least :data:`AAA_MIN_SUPPORT` support points and no spurious pole
#: carried a bias of 1e-4 beyond this ratio.
AAA_REACH_FACTOR = 4.0

#: Fewest support points a fit may pass with.
AAA_MIN_SUPPORT = 7

#: A side of the posterior region lying at least this far from its singularity
#: passes with :data:`AAA_MIN_SUPPORT` support points, whatever the reach.  In the
#: calibration sweeps 14 nodes kept the bias under 1e-4 wherever the estimate lay at
#: least 0.1 from the nearest singularity, on every matrix.  The reach test is a
#: sufficient condition calibrated near the singularities; far from a singularity it
#: would demand poles the fit does not need.
AAA_FAR_DISTANCE = 0.1

#: Half-width of the posterior region the check protects, in warmup standard
#: deviations around the warmup mean.
AAA_CHECK_PAD_SD = 4.0


@dataclass(frozen=True)
class RefitWindow:
    """The interval a refit was performed on, and how it was chosen."""

    rho_min: float
    rho_max: float
    order: int
    n_warmup_draws: int
    pad_sd: float
    prior_min: float
    prior_max: float
    err_est: float = float("nan")

    @property
    def truncates_below(self) -> bool:
        """Whether the lower edge cuts into the prior rather than resting on it."""
        return self.rho_min > self.prior_min + 1e-12

    @property
    def truncates_above(self) -> bool:
        return self.rho_max < self.prior_max - 1e-12

    def __str__(self) -> str:  # pragma: no cover - logging convenience
        return (
            f"[{self.rho_min:.4f}, {self.rho_max:.4f}] "
            f"({self.order} nodes, ±{self.pad_sd:g} sd)"
        )


@dataclass(frozen=True)
class AAACheck:
    """The outcome of the warmup AAA node check."""

    nodes: int
    support_points: int
    region: tuple[float, float] | None
    reach: tuple[float, float]
    passed: bool


def refit_window(
    rho_draws,
    prior_min: float,
    prior_max: float,
    pad_sd: float = DEFAULT_PAD_SD,
) -> tuple[float, float] | None:
    """Choose a ρ interval from warmup draws, or ``None`` if they are unusable.

    The window spans the observed range padded by ``pad_sd`` warmup standard
    deviations on each side, intersected with the prior support and clamped away
    from the ``±1`` singularities.  Padding off the observed *range* rather than
    the mean keeps the window honest when warmup has not yet settled — a chain
    still drifting produces a wide range and therefore a wide window.

    Returns ``None`` when there are too few draws to estimate a spread, or when
    the draws are degenerate (a stuck chain, whose padded window would be
    narrower than :data:`MIN_WINDOW_WIDTH`), so the caller keeps the interval it
    has rather than refitting onto a window built from nothing.
    """
    draws = np.asarray(rho_draws, dtype=np.float64).ravel()
    draws = draws[np.isfinite(draws)]
    if draws.size < 20:
        return None

    sd = float(draws.std(ddof=1))
    if not np.isfinite(sd) or sd <= 0.0:
        return None

    pad = float(pad_sd) * sd
    lo = max(float(draws.min()) - pad, float(prior_min), -0.99)
    hi = min(float(draws.max()) + pad, float(prior_max), 0.99)
    if hi - lo < MIN_WINDOW_WIDTH:
        return None
    return lo, hi


def aaa_check_region(
    rho_draws,
    rho_min: float,
    rho_max: float,
    pad_sd: float = AAA_CHECK_PAD_SD,
) -> tuple[float, float] | None:
    """The ρ region the AAA node check protects, or ``None`` if the draws are unusable.

    The warmup mean plus and minus ``pad_sd`` warmup standard deviations,
    intersected with the interpolant's interval.  A chain still drifting has a
    wide spread and so a wide region, which errs toward more nodes.
    """
    draws = np.asarray(rho_draws, dtype=np.float64).ravel()
    draws = draws[np.isfinite(draws)]
    if draws.size < 20:
        return None
    sd = float(draws.std(ddof=1))
    if not np.isfinite(sd):
        return None
    mean = float(draws.mean())
    lo = max(mean - pad_sd * sd, float(rho_min))
    hi = min(mean + pad_sd * sd, float(rho_max))
    if hi < lo:
        return None
    return lo, hi


def boundary_warning(
    rho_draws,
    window: RefitWindow,
    n_sd: float = 1.0,
) -> str | None:
    """Report whether retained draws approached a *truncated* window edge.

    An edge that coincides with the prior's own bound is not a truncation and is
    not reported — the sampler was always going to stop there.  Only edges the
    refit introduced can distort the posterior, and only if the chain reaches
    them.

    Returns a message, or ``None`` when the window comfortably contained the
    draws.
    """
    draws = np.asarray(rho_draws, dtype=np.float64).ravel()
    draws = draws[np.isfinite(draws)]
    if draws.size == 0:
        return None
    sd = float(draws.std(ddof=1)) if draws.size > 1 else 0.0
    margin = n_sd * sd

    hits = []
    if window.truncates_below and draws.min() <= window.rho_min + margin:
        hits.append(f"lower edge {window.rho_min:.4f} (min draw {draws.min():.4f})")
    if window.truncates_above and draws.max() >= window.rho_max - margin:
        hits.append(f"upper edge {window.rho_max:.4f} (max draw {draws.max():.4f})")
    if not hits:
        return None

    return (
        "Post-warmup draws of the spatial parameter reached the "
        + " and ".join(hits)
        + " of the warmup-refit window, so the reported posterior is truncated "
        "there. Increase `logdet_refit_pad_sd`, lengthen warmup, or disable the "
        "refit with `logdet_refit=False`."
    )


class LogdetRefitter:
    """Holds a reusable factorization context and refits the interpolant on it.

    Parameters
    ----------
    W_sparse : scipy.sparse matrix
        The spatial weights the interpolant is built from.
    method : str
        Resolved log-determinant method.  Only those in
        :data:`REFITTABLE_METHODS` are supported; for anything else
        :attr:`supported` is ``False`` and the caller should not refit.
    T : int, default 1
        Panel replication factor applied to the returned evaluators, matching
        :func:`~._factories.make_logdet_numpy_fn`.
    tol : float, default 1e-12
        Absolute error target for the refit.  Applies to ``cheb_cholesky``,
        where it selects the Chebyshev order.  It is **not** used by ``aaa``:
        AAA picks its support points greedily to its own internal convergence
        tolerance, so its accuracy is not a dial the caller sets here — in
        practice it lands near 1e-8, comfortably past what the refit is for.

    Notes
    -----
    Construction is lazy: no factorization happens until :meth:`refit` is first
    called, so building a refitter for a run that never reaches the refit point
    costs nothing.
    """

    def __init__(
        self,
        W_sparse,
        method: str,
        *,
        T: int = 1,
        tol: float = DEFAULT_REFIT_TOL,
        scout_tol: float = DEFAULT_SCOUT_TOL,
    ):
        self.method = str(method)
        self.W_sparse = W_sparse
        self.n = int(W_sparse.shape[0]) if W_sparse is not None else 0
        self.T = int(T)
        self.tol = float(tol)
        self.scout_tol = float(scout_tol)
        self._context = None
        # The most recent AAA fit and its node count, which the warmup node
        # check starts from, and the left singularity once it has been located.
        self.last_pre = None
        self.last_nodes = 0
        self._sigma_left = None

    def __getstate__(self):
        """Pickle without the factorization context.

        The context holds a live CHOLMOD or KLU factor, which cannot be pickled.
        It is a cache that the next fit rebuilds, so a copy sent to a worker
        process leaves it behind.
        """
        state = self.__dict__.copy()
        state["_context"] = None
        return state

    @property
    def supported(self) -> bool:
        return self.method in REFITTABLE_METHODS and self.W_sparse is not None

    def release(self) -> None:
        """Drop the cached factorization.

        The context holds a live CHOLMOD factor whose ``L`` carries the full
        Cholesky fill-in — the dominant memory of the setup, hundreds of MB at
        n = 60,000 — plus the symmetrized matrix.  A sampler refits once and
        then runs for the rest of the chain, so holding them for that whole time
        buys nothing.
        """
        self._context = None

    def _build_context(self):
        if self._context is None:
            if self.method in ("cheb_cholesky", "lu_cheb"):
                from ._chol_cheb import CholChebContext

                self._context = CholChebContext(self.W_sparse)
            elif self.method == "lu_cheb":
                from ._chol_cheb import LUChebContext

                self._context = LUChebContext(self.W_sparse)
            elif self.method == "aaa":
                from ._aaa import AAAContext

                self._context = AAAContext(self.W_sparse)
            elif self.method == "chol_aaa":
                from ._aaa import CholAAAContext

                self._context = CholAAAContext(self.W_sparse)
            else:
                raise ValueError(f"Method {self.method!r} does not support refitting.")
        return self._context

    # ------------------------------------------------------------------
    # JAX: parameters carried as traced arrays of fixed shape
    # ------------------------------------------------------------------
    #
    # A JAX Gibbs chain compiles its step once and reuses it.  If the
    # interpolant is baked into that step as a constant — which is what
    # ``make_logdet_jax_fn`` does — swapping it at the refit point changes the
    # jit cache key and forces a full retrace, measured at ~1.1 s on a Gaussian
    # SAR chain.  That is more than an order of magnitude above the refit's own
    # factorization cost, so on this backend the naive refit is a net loss.
    #
    # The fix is to carry the interpolant as *traced arrays of fixed shape*:
    # zero-pad to a capacity large enough for any order the refit may select,
    # and the compiled step is reused unchanged because only values differ.
    # Padding is exact for both representations — see
    # :func:`~._jax.jax_logdet_chebyshev_traced` for Chebyshev, and for AAA the
    # padded support points sit far outside the ρ interval with zero weight, so
    # they contribute nothing to either barycentric sum.

    #: Where padded AAA support points are parked: far outside any ρ interval,
    #: so ``rho - z_j`` is never zero and the zero weight makes the term vanish.
    _AAA_PAD_Z = 1.0e6

    def capacity(self, prior_min: float, prior_max: float) -> int:
        """Array capacity for the parameters, and the ceiling on the refit order.

        This is deliberately the order the *un-refitted* fit would use on the
        prior interval, not the order :data:`DEFAULT_REFIT_TOL` would ask for
        there.  It has to be a ceiling rather than a generous upper bound
        because the padded arrays are what the compiled evaluator loops over:
        the loop length must be a compile-time constant to stay
        reverse-differentiable, so it runs at ``capacity`` on every call
        whatever the fitted order is.  Sizing capacity off the refit tolerance
        would leave the sampler evaluating a 200-term recurrence for a 13-term
        interpolant — cost the refit is supposed to remove.

        Clamping the refit to this ceiling costs nothing that matters.  The
        refit window is always a sub-interval of the prior support, so at equal
        order it is strictly more accurate than the fit it replaces; the clamp
        only bites when the window is barely narrower, exactly where there was
        little to gain.
        """
        if self.method in ("cheb_cholesky", "lu_cheb"):
            from ._chebyshev import cheb_order_for_tolerance

            n = int(self.W_sparse.shape[0])
            return int(cheb_order_for_tolerance(prior_min, prior_max, n))
        # AAA selects m ≤ n_coarse // 2 support points, and ``_adaptive_n_coarse``
        # returns at most 18 nodes; 32 leaves room for a fit that adds nodes.
        return 32

    def _fit(
        self,
        rho_min: float,
        rho_max: float,
        cap: int | None = None,
        tol: float | None = None,
        n_coarse: int | None = None,
    ):
        """Fit on ``[rho_min, rho_max]``; the one place a context is invoked.

        Returns ``(precompute, order, err_est)``.  ``cap``, when given, is a
        hard ceiling on the order — see :meth:`capacity`.  ``tol`` overrides the
        instance target, which is how the scouting fit gets its looser one.
        ``n_coarse`` sets the number of factorizations of an AAA fit; by default
        it is the interval's static count.
        """
        ctx = self._build_context()
        if self.method in ("cheb_cholesky", "lu_cheb"):
            from ._chebyshev import cheb_order_for_tolerance

            order = cheb_order_for_tolerance(
                rho_min, rho_max, self.n, tol=self.tol if tol is None else tol
            )
            if cap is not None:
                order = min(order, int(cap))
            pre = ctx.coeffs_on(rho_min, rho_max, order=order)
            return pre, pre.order, pre.err_est
        from ._aaa import _adaptive_n_coarse

        nodes = (
            int(n_coarse)
            if n_coarse is not None
            else _adaptive_n_coarse(rho_min, rho_max)
        )
        pre = ctx.fit_on(rho_min=rho_min, rho_max=rho_max, n_coarse=nodes)
        self.last_pre, self.last_nodes = pre, nodes
        return pre, len(pre.support_points), float("nan")

    def scout_order(self, prior_min: float, prior_max: float) -> int:
        """Nodes the scouting interpolant will use on the prior interval."""
        if self.method not in ("cheb_cholesky", "lu_cheb"):
            # AAA's coarse grid is already small and its size is not a
            # tolerance dial, so there is no coarse variant to offer.
            return self.capacity(prior_min, prior_max)
        from ._chebyshev import cheb_order_for_tolerance

        return int(
            cheb_order_for_tolerance(prior_min, prior_max, self.n, tol=self.scout_tol)
        )

    def scout_fit(self, prior_min: float, prior_max: float):
        """Build the loose interpolant that carries warmup up to the refit.

        Returns ``(scalar_fn, vec_fn, order)``.  Costs
        :meth:`scout_order` factorizations against the
        :meth:`capacity` the un-refitted run would have paid, and that
        difference is the whole of the refit's speed argument — see
        :data:`DEFAULT_SCOUT_TOL`.
        """
        pre, order, _ = self._fit(prior_min, prior_max, tol=self.scout_tol)
        scalar_fn, vec_fn = self._numpy_fns(pre)
        return scalar_fn, vec_fn, order

    def _numpy_fns(self, pre):
        """NumPy ``(scalar, vectorized)`` evaluators for an already-fitted precompute."""
        T = self.T
        if self.method in ("cheb_cholesky", "lu_cheb"):
            # ``clenshaw_*`` already apply the panel factor, so no wrapper is
            # needed here — the same route ``make_logdet_numpy_fn`` takes.
            from ._clenshaw import clenshaw_scalar, clenshaw_vec

            c, lo_c, hi_c = pre.coeffs, pre.rho_min, pre.rho_max
            return (
                lambda r: clenshaw_scalar(c, float(r), lo_c, hi_c, T),
                lambda a: clenshaw_vec(
                    c, np.asarray(a, dtype=np.float64), lo_c, hi_c, T
                ),
            )

        from ._aaa import aaa_logdet_eval, aaa_logdet_eval_vec

        def _scalar(r: float) -> float:
            val = aaa_logdet_eval(pre, float(r))
            return val if T == 1 else T * val

        def _vec(rho_arr) -> np.ndarray:
            vals = aaa_logdet_eval_vec(pre, np.asarray(rho_arr, dtype=np.float64))
            return vals if T == 1 else T * vals

        return _scalar, _vec

    def _window(self, pre, order, err_est, prior_min, prior_max, meta) -> RefitWindow:
        return RefitWindow(
            rho_min=float(pre.rho_min),
            rho_max=float(pre.rho_max),
            order=int(order),
            n_warmup_draws=int(meta[0]),
            pad_sd=float(meta[1]),
            prior_min=float(prior_min),
            prior_max=float(prior_max),
            err_est=float(err_est),
        )

    # ------------------------------------------------------------------
    # AAA node check
    # ------------------------------------------------------------------

    def aaa_fit(self, rho_min: float, rho_max: float, n_coarse: int):
        """Fit AAA on ``[rho_min, rho_max]`` from ``n_coarse`` factorizations."""
        pre, _, _ = self._fit(rho_min, rho_max, n_coarse=n_coarse)
        return pre

    def left_singularity(self) -> complex:
        """The singularity ``1/λ`` of the Jacobian nearest ``ρ = -1``.

        Found by shift-invert Lanczos near ``λ = -1`` (Arnoldi for directed
        ``W``), at the cost of one sparse factorization.  It is ``-1`` for a
        bipartite graph or one with an isolated pair of neighbors, and farther
        out otherwise.  Falls back to ``-1``, the nearest a singularity can lie,
        if the eigensolver fails.
        """
        if self._sigma_left is None:
            import scipy.sparse as sp
            from scipy.sparse.linalg import eigs, eigsh

            ctx = self._build_context()
            try:
                if self.method == "chol_aaa":
                    lam = eigsh(
                        sp.csc_matrix(ctx.W_sym),
                        k=3,
                        sigma=-1.001,
                        which="LM",
                        return_eigenvectors=False,
                    )
                else:
                    lam = eigs(
                        sp.csc_matrix(ctx.W_sp),
                        k=6,
                        sigma=-1.001,
                        which="LM",
                        return_eigenvectors=False,
                    )
                sing = 1.0 / np.asarray(lam, dtype=complex)
                self._sigma_left = complex(sing[np.argmin(np.abs(sing + 1.0))])
            except Exception:  # noqa: BLE001 - any eigensolver failure falls back
                _log.info(
                    "logdet AAA check: eigensolver failed; using -1 as the left singularity."
                )
                self._sigma_left = complex(-1.0)
        return self._sigma_left

    def aaa_adequate(self, pre, region) -> tuple[bool, tuple[float, float]]:
        """Whether an AAA fit resolves the Jacobian over ``region``, and its reach.

        The fit passes when it keeps at least :data:`AAA_MIN_SUPPORT` support
        points and, on each side, ``region`` lies at least
        :data:`AAA_FAR_DISTANCE` from the singularity there or at least
        :data:`AAA_REACH_FACTOR` reaches from it.  Row-standardized ``W`` has
        spectral radius 1, so the right singularity is ``ρ = 1`` and every other
        one lies on or outside the unit circle.  The left side is tested first
        against ``-1``, the nearest a singularity there can lie, and the
        eigensolver of :meth:`left_singularity` runs only when that test fails.
        A side nearer than :data:`AAA_FAR_DISTANCE` with no pole fails.
        """
        from ._aaa import aaa_reach

        lo, hi = float(region[0]), float(region[1])
        right, left = aaa_reach(pre, -1.0)
        if len(pre.support_points) < AAA_MIN_SUPPORT:
            return False, (right, left)

        def _side(distance: float, reach: float) -> bool:
            if distance >= AAA_FAR_DISTANCE:
                return True
            return bool(np.isfinite(reach) and distance >= AAA_REACH_FACTOR * reach)

        ok_right = _side(1.0 - hi, right)
        ok_left = _side(lo + 1.0, left)
        if ok_right and not ok_left:
            s_left = self.left_singularity()
            _, left = aaa_reach(pre, s_left)
            ok_left = _side(abs(lo - s_left), left)
        return bool(ok_right and ok_left), (right, left)

    def aaa_refine(
        self, pre, n_coarse: int, region, rho_min: float, rho_max: float, max_nodes: int
    ):
        """Add nodes to an AAA fit until :meth:`aaa_adequate` passes or ``max_nodes`` binds.

        Returns ``(precompute, n_coarse, AAACheck)``.  Each rebuild factorizes at
        every node of the larger grid, since the Chebyshev grids do not nest.
        """
        ok, reach = self.aaa_adequate(pre, region)
        while not ok and n_coarse + AAA_NODE_STEP <= max_nodes:
            n_coarse += AAA_NODE_STEP
            pre = self.aaa_fit(rho_min, rho_max, n_coarse)
            ok, reach = self.aaa_adequate(pre, region)
        check = AAACheck(
            nodes=int(n_coarse),
            support_points=len(pre.support_points),
            region=(float(region[0]), float(region[1])),
            reach=(float(reach[0]), float(reach[1])),
            passed=bool(ok),
        )
        return pre, n_coarse, check

    def params_from(self, pre, capacity: int):
        """A fitted interpolant as fixed-shape JAX arrays, zero-padded to ``capacity``.

        The pytree has the same structure and shapes for every fit of a run, so
        substituting one into a compiled step triggers no retrace.
        """
        from .._jax_dispatch import ensure_x64

        ensure_x64()
        import jax.numpy as jnp

        cap = int(capacity)
        if self.method in ("cheb_cholesky", "lu_cheb"):
            order = int(pre.order)
            if order > cap:
                raise ValueError(
                    f"Refit needs {order} terms but the parameter capacity is {cap}."
                )
            coeffs = np.zeros(cap, dtype=np.float64)
            coeffs[:order] = pre.coeffs
            return (
                jnp.asarray(coeffs),
                jnp.float64(pre.rho_min),
                jnp.float64(pre.rho_max),
            )
        order = len(pre.support_points)
        if order > cap:
            raise ValueError(
                f"The AAA fit keeps {order} support points but the parameter capacity is {cap}."
            )
        z = np.full(cap, self._AAA_PAD_Z, dtype=np.float64)
        f = np.zeros(cap, dtype=np.float64)
        w = np.zeros(cap, dtype=np.float64)
        z[:order] = pre.support_points
        f[:order] = pre.support_values
        w[:order] = pre.weights
        return (jnp.asarray(z), jnp.asarray(f), jnp.asarray(w))

    def jax_params(
        self,
        rho_min: float,
        rho_max: float,
        capacity: int,
        *,
        prior_min: float | None = None,
        prior_max: float | None = None,
        n_warmup_draws: int = 0,
        pad_sd: float = float("nan"),
        with_numpy_fns: bool = False,
        tol: float | None = None,
        n_coarse: int | None = None,
    ):
        """Interpolant parameters as fixed-shape JAX arrays, plus the window.

        The returned pytree has the same structure and shapes for every
        interval, so substituting it into a compiled step triggers no retrace.
        The :class:`RefitWindow` comes back alongside so the caller does not
        have to fit twice to describe what it just built.

        ``tol`` overrides the instance target, which is how the scouting fit of
        :data:`DEFAULT_SCOUT_TOL` is built through this same path.

        Set ``with_numpy_fns`` to also get the scalar and vectorized NumPy
        evaluators for the *same* fit — the JAX Gibbs path needs them for the
        post-chain pointwise log-likelihood, and refitting to obtain them would
        double the factorization count.  The return is then
        ``(params, window, scalar_fn, vec_fn)`` instead of ``(params, window)``.
        """
        # x64 must be on *before* the arrays are made: this is often the first
        # JAX call in a run, and float32 params would later be replaced by
        # float64 ones at the refit, changing the traced dtype and forcing the
        # retrace this whole parameterization exists to avoid.
        from .._jax_dispatch import ensure_x64

        ensure_x64()

        cap = int(capacity)
        pre, order, err_est = self._fit(
            rho_min, rho_max, cap=cap, tol=tol, n_coarse=n_coarse
        )
        if order > cap:
            raise ValueError(
                f"Refit needs {order} terms but the parameter capacity is {cap}."
            )

        params = self.params_from(pre, cap)

        prior = (
            rho_min if prior_min is None else prior_min,
            rho_max if prior_max is None else prior_max,
        )
        window = self._window(
            pre, order, err_est, prior[0], prior[1], (n_warmup_draws, pad_sd)
        )
        if with_numpy_fns:
            scalar_fn, vec_fn = self._numpy_fns(pre)
            return params, window, scalar_fn, vec_fn
        return params, window

    def worth_refitting(
        self,
        rho_min: float,
        rho_max: float,
        current_min: float,
        current_max: float,
    ) -> bool:
        """Whether the window is enough narrower than the current interval."""
        if not self.supported:
            return False
        current_width = float(current_max) - float(current_min)
        new_width = float(rho_max) - float(rho_min)
        if new_width <= 0.0:
            return False
        return current_width / new_width >= MIN_NARROWING

    def plan(
        self,
        warmup_rho,
        prior_min: float,
        prior_max: float,
        current_min: float,
        current_max: float,
        pad_sd: float = DEFAULT_PAD_SD,
    ) -> tuple[float, float] | None:
        """Decide whether and where to refit, logging the reason when not.

        Both backends route their decision through here so the acceptance rule
        and its diagnostics cannot drift apart between them.
        """
        window = refit_window(warmup_rho, prior_min, prior_max, pad_sd=pad_sd)
        if window is None:
            _log.info("logdet_refit: warmup draws unusable; keeping prior interval.")
            return None
        lo, hi = window
        if not self.worth_refitting(lo, hi, current_min, current_max):
            _log.info(
                f"logdet_refit: warmup window [{lo:.4f}, {hi:.4f}] is not "
                "materially narrower than the current interval; not refitting."
            )
            return None
        return lo, hi

    def refit(
        self,
        rho_min: float,
        rho_max: float,
        prior_min: float,
        prior_max: float,
        *,
        capacity: int | None = None,
        n_warmup_draws: int = 0,
        pad_sd: float = float("nan"),
    ):
        """Fit on ``[rho_min, rho_max]`` and return evaluators plus the window.

        ``n_warmup_draws`` and ``pad_sd`` are recorded on the returned
        :class:`RefitWindow` for reporting; they do not affect the fit.

        Returns
        -------
        (logdet_fn, logdet_vec_fn, RefitWindow)
            Callables with the same signatures as
            :func:`~._factories.make_logdet_numpy_fn` and
            :func:`~._factories.make_logdet_numpy_vec_fn`.
        """
        pre, order, err_est = self._fit(rho_min, rho_max, cap=capacity)
        _scalar, _vec = self._numpy_fns(pre)
        window = self._window(
            pre, order, err_est, prior_min, prior_max, (n_warmup_draws, pad_sd)
        )
        return _scalar, _vec, window
