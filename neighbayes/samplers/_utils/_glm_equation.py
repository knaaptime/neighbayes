r"""One reduced-form equation with a materialized filtered design.

.. math::

    \eta = (I_T \otimes A(\rho))^{-1} X\theta_x + D_\tau\tau + D_g c,

``θ = (θ_x, τ)``, for any filter of
:mod:`neighbayes.samplers.count_panel._filters` (``NoFilter`` for an aspatial
equation).  :meth:`MaterializedEquation.update` takes one sweep's Gaussian
working data ``(z, ω)`` on a subset of the rows — Pólya–Gamma for a logit or an
NB count — and draws each filter parameter with ``θ`` and the group effects
``c`` integrated out, then ``(θ, c)`` jointly; ``η`` is then recomputed on
every row.  Rows outside the subset do not enter the likelihood (a ZINB count
on its active rows, a hurdle count on its positive rows); a group with no
working rows draws its effect from the prior.

The two-part samplers compose two of these, as the structured flow sweeps
compose two :class:`~neighbayes.samplers.negbin_reduced._flow_structured_fe.StructuredEquation`.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from ._group_effects import (
    GroupEffects,
    GroupProjector,
    collapsed_log_density,
    draw_collapsed,
)
from ._slice import SliceWidthState, slice_sample_1d_adaptive, update_slice_width


def slice_filter_params(filt, params, density, widths, rng, tuning):
    """Slice each of ``filt``'s parameters in turn, updating ``params`` in place."""
    for name in filt.names:
        lo, hi = filt.bounds(name)

        def log_density(v, _name=name):
            cand = dict(params)
            cand[_name] = float(v)
            if not filt.admissible(cand):
                return -np.inf
            try:
                return density(cand)
            except (RuntimeError, ValueError, np.linalg.LinAlgError):
                return -np.inf

        new, _, sl, sr = slice_sample_1d_adaptive(
            log_density,
            params[name],
            lower=lo,
            upper=hi,
            width_state=widths[name],
            rng=rng,
        )
        if tuning:
            update_slice_width(widths[name], sl, sr)
        params[name] = float(new)


def init_from_counts(y, X, fe, rows: np.ndarray, rng, jitter: float = 0.1):
    """Least-squares start on ``log(y + ½)`` over ``rows``; group effects from the residual."""
    target = np.log(np.asarray(y, dtype=np.float64) + 0.5)
    D = X if fe.D_tau is None else np.hstack([X, fe.D_tau])
    theta = (
        np.linalg.lstsq(D[rows], target[rows], rcond=None)[0]
        if D.shape[1]
        else np.zeros(0)
    )
    theta = theta + rng.normal(0.0, jitter, size=theta.size)
    c = None
    if fe.groups is not None:
        g = fe.groups.row_group[rows]
        resid = (target - D @ theta)[rows]
        cnt = np.bincount(g, minlength=fe.groups.n_groups)
        c = np.bincount(g, weights=resid, minlength=fe.groups.n_groups) / np.maximum(
            cnt, 1
        )
    return theta, c


class MaterializedEquation:
    """One equation ``η = (I_T ⊗ A)⁻¹Xθ_x + D_τ τ + D_g c`` with ``U`` materialized.

    Parameters
    ----------
    X : ndarray, shape (N·T, k)
        Time-first stacked design (absorbed columns already dropped).
    filt
        A filter from :mod:`neighbayes.samplers.count_panel._filters`.
    mu0, prec0 : ndarray, shape (k + n_tau,)
        Normal prior on ``θ = (θ_x, τ)``.
    fe : CountPanelFE or None
        Group effects and period dummies; ``None`` for neither.
    theta, c : ndarray
        Starting ``θ`` and group effects.
    params : dict, optional
        Starting filter parameters; drawn by ``filt.init`` when omitted.
    rng : numpy.random.Generator
    """

    def __init__(
        self,
        X: np.ndarray,
        filt,
        mu0,
        prec0,
        fe=None,
        *,
        theta,
        c: Optional[np.ndarray] = None,
        params: Optional[dict] = None,
        rng: np.random.Generator,
    ):
        from ..count_panel._core import CountPanelFE

        self.X = np.ascontiguousarray(X, dtype=np.float64)
        self.filt = filt
        self.fe = CountPanelFE() if fe is None else fe
        self.mu0 = np.asarray(mu0, dtype=np.float64)
        self.prec0 = np.asarray(prec0, dtype=np.float64)
        if self.mu0.size != self.X.shape[1] + self.fe.n_tau:
            raise ValueError("priors must cover the design columns and periods")
        self.params = filt.init(rng) if params is None else dict(params)
        self.theta = np.asarray(theta, dtype=np.float64).copy()
        self.c = None if c is None else np.asarray(c, dtype=np.float64).copy()
        self.widths = {name: SliceWidthState() for name in filt.names}
        groups = self.fe.groups
        self._g_idx = groups.row_group if groups is not None else None
        self.U = self.design(self.params)
        self.eta = self._eta_of(self.U)

    @property
    def k(self) -> int:
        return self.X.shape[1]

    def design(self, params: dict) -> np.ndarray:
        """``[(I_T ⊗ A)⁻¹X, D_τ]`` at ``params``."""
        U = self.filt.solve(params, self.X)
        return U if self.fe.D_tau is None else np.hstack([U, self.fe.D_tau])

    def _eta_of(self, U: np.ndarray) -> np.ndarray:
        e = U @ self.theta
        return e if self.c is None else e + self.c[self._g_idx]

    def _projector(self, rows, omega):
        groups = self.fe.groups
        if groups is None:
            return None
        g = groups.row_group if rows is None else groups.row_group[rows]
        spec = GroupEffects(g, groups.n_groups, groups.mu, groups.sigma)
        return GroupProjector(spec, omega)

    def exact_sd_move(self, loglik, rng) -> None:
        """Non-centred sd step on the exact likelihood (see
        :func:`~._group_effects.exact_noncentered_sd`).

        ``loglik(eta)`` gives the data log-likelihood at a full predictor
        ``eta``; the group part is rescaled ``μ + σ c̃`` with ``c̃`` held.
        No-op unless the group sd is learned.
        """
        from dataclasses import replace

        from ._group_effects import exact_noncentered_sd

        groups = self.fe.groups
        if groups is None or groups.sigma_scale is None or self.c is None:
            return
        g = groups.row_group
        ct = (self.c - groups.mu) / groups.sigma
        a = ct[g]
        rest = self.eta - self.c[g] + groups.mu
        sd = exact_noncentered_sd(
            groups.sigma, groups.sigma_scale, lambda s: loglik(rest + s * a), rng
        )
        self.fe = replace(self.fe, groups=replace(groups, sigma=sd))
        self.c = groups.mu + sd * ct
        self.eta = rest + sd * a

    def level_direction(self):
        """``(v, log_prior, apply)`` for a shift ``η → η + t·v`` of the level.

        The level is carried by the intercept column when there is one (``v``
        its filtered column), else by period effects covering every row, else
        by the group effects (all shifted by ``t``).  The intercept comes first:
        shifting pooled group effects would move their mean off the prior's,
        which is exactly what their sd is estimated from.  ``log_prior(t)`` is
        the carrier's log prior at the shift; ``apply(t)`` makes it.  ``None``
        when nothing carries a level.
        """
        k = self.k
        const = np.flatnonzero(np.all(self.X == self.X[:1], axis=0) & (self.X[0] != 0))
        if const.size:
            j = int(const[0])
            mu, prec = self.mu0[j], self.prec0[j]

            def log_prior(t):
                return -0.5 * prec * (self.theta[j] + t - mu) ** 2

            def shift(t):
                self.theta = self.theta.copy()
                self.theta[j] += t

            v = self.U[:, j].copy()
        elif self.fe.D_tau is not None and np.all(self.fe.D_tau.sum(axis=1) == 1.0):
            mu, prec = self.mu0[k:], self.prec0[k:]

            def log_prior(t):
                return -0.5 * float(np.sum(prec * (self.theta[k:] + t - mu) ** 2))

            def shift(t):
                self.theta = self.theta.copy()
                self.theta[k:] += t

            v = np.ones(self.eta.size)
        elif self.c is not None:
            groups = self.fe.groups

            def log_prior(t):
                return (
                    -0.5
                    * float(np.sum((self.c + t - groups.mu) ** 2))
                    / groups.sigma**2
                )

            def shift(t):
                self.c = self.c + t

            v = np.ones(self.eta.size)
        else:
            return None

        def apply(t):
            shift(t)
            self.eta = self.eta + t * v

        return v, log_prior, apply

    def update(
        self,
        rows: Optional[np.ndarray],
        z: np.ndarray,
        omega: np.ndarray,
        rng: np.random.Generator,
        tuning: bool,
    ) -> None:
        """Draw the filter parameters, then ``(θ, c)``, from working data on ``rows``.

        ``rows`` is a boolean mask or index array (``None`` for every row);
        ``z`` and ``omega`` are given on those rows only.
        """
        proj = self._projector(rows, omega)
        take = (lambda U: U) if rows is None else (lambda U: U[rows])
        slice_filter_params(
            self.filt,
            self.params,
            lambda cand: collapsed_log_density(
                take(self.design(cand)), z, omega, self.mu0, self.prec0, proj
            ),
            self.widths,
            rng,
            tuning,
        )
        U = self.design(self.params)
        theta, c_new = draw_collapsed(
            take(U), z, omega, self.mu0, self.prec0, proj, rng
        )
        self.theta = theta
        if c_new is not None:
            self.c = c_new
            groups = self.fe.groups
            if groups.sigma_scale is not None:
                from dataclasses import replace

                from ._group_effects import interweave_group_sd

                g = groups.row_group if rows is None else groups.row_group[rows]
                new_groups, self.c = interweave_group_sd(
                    groups, self.c, z - take(U) @ theta, omega, g, rng
                )
                self.fe = replace(self.fe, groups=new_groups)
        self.U = U
        self.eta = self._eta_of(U)

    @property
    def group_sd(self) -> Optional[float]:
        groups = self.fe.groups
        return None if groups is None else float(groups.sigma)
