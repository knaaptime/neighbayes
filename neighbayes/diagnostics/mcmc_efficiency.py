"""MCMC sampling-efficiency diagnostics for Bayesian spatial models.

This module implements the post-fit checks recommended by
:cite:t:`wolf2018StochasticEfficiency` for Bayesian spatial econometric
MCMC.  Their empirical study showed that the spatial dependence
parameter (``rho`` or ``lam``) tends to mix poorly: chains that look
visually converged in their point estimates can still produce posterior
credible intervals that are 10–12 % narrower than their long-run width
because the sampler has not had time to explore the tails of the
posterior.

The remedy advocated by Wolf et al. is to **report and stop on
effective sample size (ESS)**, not on the nominal post-warmup draw
count.  This module provides a single function,
:func:`spatial_mcmc_diagnostic`, that wraps the standard
:mod:`arviz` primitives (``ess_bulk``, ``ess_tail``, ``rhat``,
``mcse_mean``) with spatial-econometrics-aware thresholds and surfaces
warnings when those thresholds are violated.

Notes
-----
The ESS estimator used by ArviZ matches the autocorrelation-based
definition in :cite:t:`gelman2014BayesianData`, eq. (11.8), which is
the formula adopted by Wolf et al. (2018), eq. (1).

References
----------
.. bibliography::
   :filter: docname in docnames
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import ClassVar, Optional

import numpy as np

from .._lazy_deps import az


@dataclass
class SpatialMCMCReport:
    """Summary of MCMC sampling-efficiency checks for a spatial model.

    Attributes
    ----------
    parameters : list of str
        Parameter names actually checked (subset of those present in the
        posterior, in the order they were inspected).
    ess_bulk : dict[str, float]
        Bulk effective sample size per parameter (ArviZ ``ess_bulk``).
    ess_tail : dict[str, float]
        Tail effective sample size per parameter (ArviZ ``ess_tail``).
    r_hat : dict[str, float]
        Gelman–Rubin :math:`\\hat{R}` per parameter.
    mcse_mean : dict[str, float]
        Monte-Carlo standard error of the posterior mean per parameter.
    nominal_size : int
        Total post-warmup draws aggregated across chains.
    yield_pct : dict[str, float]
        ESS-bulk / nominal × 100, the *yield* statistic from Wolf et al.
    hpdi_drift_pct : dict[str, float]
        Relative change in 95 % HPDI width between the last third and
        the full post-warmup chain, in percent.  Large drift indicates
        the chain has not yet visited the tails enough times for the
        interval estimate to stabilize.
    warnings_triggered : list of str
        Human-readable description of each adequacy check that failed.
    adequate : bool
        ``True`` when no warnings were triggered for the spatial
        dependence parameter(s).
    adequate_by_param : dict[str, bool]
        Per-parameter adequacy flag.  ``True`` for a parameter when
        none of the adequacy checks (ESS, yield, HPDI drift, R-hat)
        failed for it.  Only populated for the auto-detected spatial
        dependence parameters; ``extra_params`` are not gated.
    """

    parameters: list[str]
    ess_bulk: dict[str, float]
    ess_tail: dict[str, float]
    r_hat: dict[str, float]
    mcse_mean: dict[str, float]
    nominal_size: int
    yield_pct: dict[str, float]
    hpdi_drift_pct: dict[str, float]
    warnings_triggered: list[str] = field(default_factory=list)
    adequate: bool = True
    adequate_by_param: dict[str, bool] = field(default_factory=dict)

    #: Per-parameter columns produced by :meth:`to_frame`, in display order.
    _FRAME_COLUMNS: ClassVar[tuple[str, ...]] = (
        "ess_bulk",
        "ess_tail",
        "r_hat",
        "mcse_mean",
        "yield_pct",
        "hpdi_drift_pct",
        "adequate",
    )

    def to_frame(self):
        """Return per-parameter diagnostics as a ``pandas.DataFrame``.

        One row per spatial dependence parameter (``self.parameters``),
        with columns for each per-parameter metric plus an
        ``adequate`` column derived from ``adequate_by_param``.  Global
        scalars (``nominal_size``) and ``warnings_triggered`` are not
        included; access them on the dataclass directly.

        Returns
        -------
        pandas.DataFrame
            Index name ``"parameter"``; columns ordered per
            :attr:`_FRAME_COLUMNS`.
        """
        import pandas as pd

        col_data = {
            "ess_bulk": self.ess_bulk,
            "ess_tail": self.ess_tail,
            "r_hat": self.r_hat,
            "mcse_mean": self.mcse_mean,
            "yield_pct": self.yield_pct,
            "hpdi_drift_pct": self.hpdi_drift_pct,
            "adequate": self.adequate_by_param,
        }
        df = pd.DataFrame(col_data, index=pd.Index(self.parameters, name="parameter"))
        return df[list(self._FRAME_COLUMNS)]


#: Names treated as "the spatial dependence parameter" for the purposes
#: of triggering warnings.  Covers cross-sectional / panel models
#: (``rho``, ``lam``), origin-destination flow models (``rho_d``,
#: ``rho_o``, ``rho_w``), and dynamic-panel temporal-AR coefficients
#: (``phi``).  These are the parameters Wolf et al. (2018) identify as
#: the MCMC bottleneck.
_SPATIAL_PARAM_NAMES: tuple[str, ...] = (
    "rho",
    "lam",
    "rho_d",
    "rho_o",
    "rho_w",
    "phi",
)


def _pick_spatial_params(posterior) -> list[str]:
    """Return spatial-dependence parameters present in ``posterior``.

    Auto-detects the standard names used across ``neighbayes`` model
    families: ``rho``/``lam`` (cross-sectional and panel SAR/SEM/SDM),
    ``rho_d``/``rho_o``/``rho_w`` (flow models), and ``phi`` (dynamic-
    panel temporal-AR coefficient).
    """
    return [name for name in _SPATIAL_PARAM_NAMES if name in posterior.data_vars]


def _flatten_chain(da) -> np.ndarray:
    """Flatten an ArviZ DataArray over (chain, draw) for a scalar variable."""
    arr = np.asarray(da.values)
    if arr.ndim == 1:
        return arr
    return arr.reshape(arr.shape[0] * arr.shape[1], *arr.shape[2:])


def _hpdi_width(samples: np.ndarray, hdi_prob: float = 0.95) -> float:
    """Return the width of an equal-tailed HDI for a 1-D sample."""
    lo, hi = np.asarray(az.hdi(samples, prob=hdi_prob))
    return float(hi - lo)


def spatial_mcmc_diagnostic(
    model,
    *,
    min_ess: float = 1000.0,
    min_yield_pct: float = 5.0,
    max_hpdi_drift_pct: float = 5.0,
    max_r_hat: float = 1.01,
    extra_params: Optional[list[str]] = None,
    emit_warnings: bool = True,
) -> SpatialMCMCReport:
    """Run MCMC adequacy checks on a fitted Bayesian spatial model.

    Implements the practical recommendations of
    :cite:t:`wolf2018StochasticEfficiency`: report effective sample
    size and yield for the spatial dependence parameter, and check that
    the posterior credible-interval width has stabilized.  When any
    threshold is violated, a :class:`UserWarning` is emitted (unless
    ``emit_warnings=False``) describing the failure and pointing to
    the actionable fix (longer chains or thinning).

    Parameters
    ----------
    model : neighbayes model
        Fitted spatial model exposing ``inference_data`` with a
        ``posterior`` group containing ``rho`` and/or ``lam``.
    min_ess : float, optional
        Minimum acceptable bulk ESS for monitored parameters.  The
        default of ``1000`` matches the target effective sample size
        used in the Wolf et al. (2018) illustrations.
    min_yield_pct : float, optional
        Minimum acceptable yield (ESS / nominal × 100).  The default
        of ``5 %`` is the lower end of the "well-behaved" range
        reported in Wolf et al. (2018), Table 1.
    max_hpdi_drift_pct : float, optional
        Maximum tolerated relative change in the 95 % HPDI width
        between the last third and the full post-warmup chain.  Large
        drift indicates the chain has not yet stabilized and
        terminating now would understate uncertainty (Wolf et al.,
        2018, §"Impacts of small effective size and trace
        autocorrelation").
    max_r_hat : float, optional
        Maximum tolerated :math:`\\hat{R}`.  Default ``1.01`` follows
        the modern ArviZ recommendation.
    extra_params : list of str, optional
        Additional posterior variables to include in the report (for
        example, ``["beta", "sigma"]``).  These are reported but do
        *not* count toward the ``adequate`` flag, which tracks only
        the spatial dependence parameter(s).
    emit_warnings : bool, optional
        If ``True`` (default), each failed check emits a
        :class:`UserWarning`.  Set to ``False`` to inspect the report
        silently.

    Returns
    -------
    SpatialMCMCReport
        Dataclass with per-parameter ESS, yield, HPDI-drift, and
        :math:`\\hat{R}` statistics, plus a list of warnings triggered
        and an ``adequate`` flag.

    Raises
    ------
    RuntimeError
        If ``model`` has not been fit yet (``inference_data`` is
        ``None``).
    ValueError
        If the posterior contains neither ``rho`` nor ``lam`` and no
        ``extra_params`` were requested.

    Notes
    -----
    The diagnostic deliberately monitors only ``rho`` / ``lam`` by
    default.  Wolf et al. (2018) demonstrate that :math:`\\beta`
    typically mixes well in spatial-lag specifications; the spatial
    dependence parameter is the bottleneck.  Spatial impact summaries
    (direct / indirect / total) inherit ``rho``'s autocorrelation
    through :math:`(I-\\rho W)^{-1}`, so checking ``rho`` is a
    sufficient screen for them as well.

    The HPDI-drift check is a lightweight alternative to running
    multiple long chains and comparing their long-run widths: a
    converged sampler should produce a stable interval width across
    the second half of the chain.  When drift exceeds
    ``max_hpdi_drift_pct``, the warning text suggests doubling the
    number of post-warmup draws.

    Examples
    --------
    >>> from neighbayes.models import SAR
    >>> from neighbayes.diagnostics import spatial_mcmc_diagnostic
    >>> model = SAR(y, X, W=W).fit(draws=2000, tune=1000)  # doctest: +SKIP
    >>> report = spatial_mcmc_diagnostic(model)            # doctest: +SKIP
    >>> report.adequate                                    # doctest: +SKIP
    True

    References
    ----------
    .. bibliography::
       :filter: docname in docnames
    """
    idata = getattr(model, "inference_data", None)
    if idata is None or "posterior" not in idata.children:
        raise RuntimeError(
            "Model has not been fit yet; `inference_data` is unavailable."
        )

    posterior = idata.posterior
    spatial_params = _pick_spatial_params(posterior)
    extras = list(extra_params or [])
    monitored = spatial_params + [p for p in extras if p in posterior.data_vars]

    if not monitored:
        raise ValueError(
            "Posterior contains neither 'rho' nor 'lam', and no "
            "matching `extra_params` were supplied."
        )

    # Nominal post-warmup size = chains * draws (multidimensional vars
    # get the same nominal size; dimensions beyond chain/draw are
    # treated separately below).
    n_chain = int(posterior.sizes.get("chain", 1))
    n_draw = int(posterior.sizes.get("draw", 0))
    nominal_size = n_chain * n_draw

    ess_bulk: dict[str, float] = {}
    ess_tail: dict[str, float] = {}
    r_hat: dict[str, float] = {}
    mcse_mean: dict[str, float] = {}
    yield_pct: dict[str, float] = {}
    hpdi_drift_pct: dict[str, float] = {}
    warnings_triggered: list[str] = []

    for name in monitored:
        # On a DataArray the ArviZ diagnostics return a DataArray over the
        # parameter's own dims; for vector-valued params (e.g. beta with k
        # coords) we collapse to the worst case.
        da = posterior[name]
        ess_b = float(np.nanmin(np.asarray(az.ess(da, method="bulk"))))
        ess_t = float(np.nanmin(np.asarray(az.ess(da, method="tail"))))
        rhat_v = float(np.nanmax(np.asarray(az.rhat(da))))
        mcse_v = float(np.nanmax(np.asarray(az.mcse(da, method="mean"))))

        ess_bulk[name] = ess_b
        ess_tail[name] = ess_t
        r_hat[name] = rhat_v
        mcse_mean[name] = mcse_v
        yield_pct[name] = (ess_b / nominal_size * 100.0) if nominal_size else 0.0

        # HPDI drift: only meaningful for scalar parameters; for
        # vector-valued draws we report the worst per-coordinate drift.
        flat = _flatten_chain(posterior[name])
        if flat.ndim == 1:
            full_w = _hpdi_width(flat)
            tail_w = _hpdi_width(flat[-(len(flat) // 3) :])
            drift = abs(tail_w - full_w) / max(full_w, 1e-12) * 100.0
        else:
            drifts = []
            flat2 = flat.reshape(flat.shape[0], -1)
            for j in range(flat2.shape[1]):
                col = flat2[:, j]
                full_w = _hpdi_width(col)
                tail_w = _hpdi_width(col[-(len(col) // 3) :])
                drifts.append(abs(tail_w - full_w) / max(full_w, 1e-12) * 100.0)
            drift = float(max(drifts))
        hpdi_drift_pct[name] = float(drift)

    adequate = True
    adequate_by_param: dict[str, bool] = {name: True for name in spatial_params}
    suggestion = (
        "Consider doubling `draws` and/or `tune`, or re-running with "
        "more chains. See Wolf, Anselin & Arribas-Bel (2018), Geographical "
        "Analysis 50:97-119."
    )
    for name in spatial_params:
        if ess_bulk[name] < min_ess:
            msg = (
                f"Bulk ESS for spatial parameter '{name}' is "
                f"{ess_bulk[name]:.0f} (< target {min_ess:g}). "
                "Posterior credible intervals may be unreliable. " + suggestion
            )
            warnings_triggered.append(msg)
            adequate = False
            adequate_by_param[name] = False
        if yield_pct[name] < min_yield_pct:
            msg = (
                f"Sampler yield for '{name}' is "
                f"{yield_pct[name]:.2f}% (< {min_yield_pct:g}%); the chain "
                "is heavily autocorrelated. " + suggestion
            )
            warnings_triggered.append(msg)
            adequate = False
            adequate_by_param[name] = False
        if hpdi_drift_pct[name] > max_hpdi_drift_pct:
            msg = (
                f"95% HPDI width for '{name}' drifts by "
                f"{hpdi_drift_pct[name]:.1f}% between the last third "
                f"and the full chain (> {max_hpdi_drift_pct:g}%); the "
                "credible interval has not yet stabilized. " + suggestion
            )
            warnings_triggered.append(msg)
            adequate = False
            adequate_by_param[name] = False
        if not np.isnan(r_hat[name]) and r_hat[name] > max_r_hat:
            msg = (
                f"R-hat for '{name}' is {r_hat[name]:.3f} "
                f"(> {max_r_hat:g}); chains have not converged."
            )
            warnings_triggered.append(msg)
            adequate = False
            adequate_by_param[name] = False

    if emit_warnings:
        for msg in warnings_triggered:
            warnings.warn(msg, UserWarning, stacklevel=2)

    return SpatialMCMCReport(
        parameters=monitored,
        ess_bulk=ess_bulk,
        ess_tail=ess_tail,
        r_hat=r_hat,
        mcse_mean=mcse_mean,
        nominal_size=nominal_size,
        yield_pct=yield_pct,
        hpdi_drift_pct=hpdi_drift_pct,
        warnings_triggered=warnings_triggered,
        adequate=adequate,
        adequate_by_param=adequate_by_param,
    )
