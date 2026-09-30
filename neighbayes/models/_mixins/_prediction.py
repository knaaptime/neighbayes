"""Posterior-predictive prediction for the Gaussian cross-section models.

Adds :meth:`~GaussianPredictionMixin.predict` (new units) and
:meth:`~GaussianPredictionMixin.predict_in_sample` (leave-one-out
conditionals) to OLS, SLX, SAR, SDM, SEM and SDEM.  Both average the exact
conditional Gaussian of :mod:`neighbayes._prediction` over posterior draws,
so the predictive carries parameter uncertainty rather than plugging in point
estimates.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pandas as pd
import scipy.sparse as sp

from ..._lazy_deps import az

_KIND = {None: "iid", "rho": "lag", "lam": "error"}


class GaussianPredictionMixin:
    """Prediction methods for models with a Gaussian joint likelihood.

    Relies on the ``SpatialModel`` attributes ``_jacobian_param`` and
    ``_has_wx_in_beta`` and on the fitted posterior variables ``beta``,
    ``sigma`` and, for spatial models, ``rho`` or ``lam``.
    """

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _prediction_kind(self) -> str:
        if getattr(self, "robust", False):
            raise NotImplementedError(
                "Prediction uses the Gaussian conditional of y_O given y_S, "
                "which does not hold for Student-t errors (robust=True)."
            )
        self._require_fit()
        return _KIND[self._jacobian_param]

    def _new_design(self, X_new: Any) -> np.ndarray:
        """Covariates for new units, in the column order of the fitted ``X``."""
        k = self._X.shape[1]
        if isinstance(X_new, pd.DataFrame):
            if self._model_spec is not None:
                X_mm = self._model_spec.get_model_matrix(X_new)
                missing = [c for c in self._feature_names if c not in X_mm.columns]
                if missing:
                    raise ValueError(f"X_new does not produce columns {missing}.")
                return X_mm[self._feature_names].to_numpy(dtype=np.float64)
            missing = [c for c in self._feature_names if c not in X_new.columns]
            if missing:
                raise ValueError(f"X_new is missing columns {missing}.")
            return X_new[self._feature_names].to_numpy(dtype=np.float64)
        if self._model_spec is not None:
            raise TypeError(
                "This model was built from a formula; pass X_new as a DataFrame "
                "with the variables the formula uses."
            )
        X_arr = np.asarray(X_new, dtype=np.float64)
        if X_arr.ndim != 2 or X_arr.shape[1] != k:
            raise ValueError(f"X_new must have shape (n_new, {k}), got {X_arr.shape}.")
        return X_arr

    def _draws(self, kind: str, thin: int):
        """Posterior draws ``(beta, sigma, theta)`` of shape ``(C, D, ...)``."""
        if thin < 1:
            raise ValueError(f"thin must be a positive integer, got {thin}.")
        post = self._idata.posterior
        beta = post["beta"].transpose("chain", "draw", ...).to_numpy()[:, ::thin]
        sigma = post["sigma"].transpose("chain", "draw").to_numpy()[:, ::thin]
        if kind == "iid":
            theta = np.zeros_like(sigma)
        else:
            name = self._jacobian_param
            theta = post[name].transpose("chain", "draw").to_numpy()[:, ::thin]
        return beta, sigma, theta

    @staticmethod
    def _mean_solver(kind: str, W: sp.csr_matrix, n: int):
        """``mu(theta, Z beta)`` — the marginal mean at one draw."""
        if kind != "lag":
            return lambda theta, Zb: Zb
        from ...samplers._utils._sparsax_utils import CachedSparseSolver

        solver = CachedSparseSolver([W], n)
        return lambda theta, Zb: solver.solve([-theta], Zb)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def predict(
        self,
        X_new: Any,
        W: Any,
        oos: Optional[np.ndarray] = None,
        *,
        thin: int = 1,
        random_seed: Optional[int] = None,
    ) -> az.InferenceData:
        """Posterior predictive distribution of ``y`` at new units.

        For each posterior draw the outcome at the new units :math:`O`,
        given the fitted outcomes :math:`y_S`, is exactly

        .. math::

            y_O \\mid y_S \\sim N\\bigl(\\mu_O - \\Lambda_{OO}^{-1}
            \\Lambda_{OS}(y_S - \\mu_S),\\; \\Lambda_{OO}^{-1}\\bigr),
            \\qquad \\Lambda = (I - \\theta W)^{\\top}(I - \\theta W)/\\sigma^2,

        the best predictor (BP) of :cite:t:`goulard2017PredictionsSpatial`
        (eq. 9) at that draw.  :math:`\\theta` is :math:`\\rho` for SAR and
        SDM, :math:`\\lambda` for SEM and SDEM, and zero for OLS and SLX,
        whose predictive is :math:`N(\\mu_O, \\sigma^2 I)`.  The marginal mean
        :math:`\\mu` is :math:`(I - \\rho W)^{-1} Z\\beta` for the lag models
        and :math:`Z\\beta` otherwise, computed on the union graph.

        Parameters
        ----------
        X_new : DataFrame or array of shape (n_new, k)
            Covariates of the new units.  For a model built from a formula,
            a DataFrame with the variables the formula uses; the formula's
            transforms are replayed on it.  Otherwise, columns in the order
            of the fitted ``X`` (or a DataFrame with its column names).
        W : libpysal.graph.Graph or scipy.sparse matrix
            Weights over the union of fitted and new units,
            ``n_fit + n_new`` square, standardized the way the fitted ``W``
            was.  For SLX, SDM and SDEM the lagged covariates of *every* unit
            are recomputed on this graph, since a fitted unit with a new
            neighbor gets a new ``WX`` row.
        oos : array of int, optional
            Positions of the new units in ``W``, in the row order of
            ``X_new``; the fitted units fill the remaining positions in
            their fitted order.  Defaults to the last ``n_new`` positions.
        thin : int, default 1
            Use every ``thin``-th draw of each chain.
        random_seed : int, optional
            Seed for the predictive draws.

        Returns
        -------
        arviz.InferenceData
            A ``predictions`` group with dimensions ``(chain, draw, obs_new)``
            holding ``y`` (posterior predictive draws), ``y_bp`` (the
            conditional mean at each draw; its average over draws is the
            posterior predictive mean) and ``y_tc`` (the trend predictor
            :math:`\\mu_O`, which ignores the observed neighbors), and a
            ``predictions_constant_data`` group with ``X_new``.  Comparing
            ``y_bp`` with ``y_tc`` shows what the spatial correction adds.

        Notes
        -----
        The parameters were estimated on the fitted units' ``W``, and the
        prediction uses the union ``W``: the feasible practice of
        :cite:t:`goulard2017PredictionsSpatial` (their eq. 8).
        """
        from ..._prediction import GaussianConditional
        from .._base._shared import resolve_W

        kind = self._prediction_kind()
        Xo = self._new_design(X_new)
        n_fit, n_new = self._X.shape[0], Xo.shape[0]
        n = n_fit + n_new
        W_u, _ = resolve_W(W, n)
        if oos is None:
            oos = np.arange(n_fit, n)
        oos = np.asarray(oos, dtype=np.int64).ravel()
        if oos.size != n_new:
            raise ValueError(
                f"oos has {oos.size} positions but X_new has {n_new} rows."
            )
        cond = GaussianConditional(None if kind == "iid" else W_u, oos, n)

        X_u = np.empty((n, self._X.shape[1]), dtype=np.float64)
        X_u[cond.obs_idx] = self._X
        X_u[oos] = Xo
        Z = X_u
        if self._has_wx_in_beta and self._wx_column_indices:
            Z = np.hstack([X_u, W_u @ X_u[:, self._wx_column_indices]])
        y_u = np.zeros(n, dtype=np.float64)
        y_u[cond.obs_idx] = self._y

        beta, sigma, theta = self._draws(kind, thin)
        if beta.shape[-1] != Z.shape[1]:
            raise ValueError(
                f"Posterior beta has {beta.shape[-1]} coefficients but the design "
                f"has {Z.shape[1]} columns."
            )
        mean_fn = self._mean_solver(kind, W_u, n)
        rng = np.random.default_rng(random_seed)
        C, D = sigma.shape
        out = {k: np.empty((C, D, n_new)) for k in ("y", "y_bp", "y_tc")}
        for c in range(C):
            for d in range(D):
                mu = mean_fn(theta[c, d], Z @ beta[c, d])
                cond.update(theta[c, d], sigma[c, d])
                m = cond.mean(mu, y_u)
                out["y_tc"][c, d] = mu[oos]
                out["y_bp"][c, d] = m
                out["y"][c, d] = cond.sample(m, rng)

        coords = {"obs_new": _row_labels(X_new, oos)}
        dims = {k: ["obs_new"] for k in out}
        return az.from_dict(
            predictions=out,
            predictions_constant_data={"X_new": Xo},
            coords={**coords, "coefficient": list(self._feature_names)},
            dims={**dims, "X_new": ["obs_new", "coefficient"]},
        )

    def predict_in_sample(
        self,
        *,
        thin: int = 1,
        random_seed: Optional[int] = None,
    ) -> az.InferenceData:
        """Leave-one-out conditional predictive of each fitted unit.

        For each unit :math:`i` and posterior draw, the distribution of
        :math:`y_i` given every other fitted outcome is

        .. math::

            y_i \\mid y_{-i} \\sim N\\bigl(y_i - (\\Lambda r)_i/\\Lambda_{ii},
            \\; 1/\\Lambda_{ii}\\bigr), \\qquad r = y - \\mu,

        the in-sample BP of :cite:t:`goulard2017PredictionsSpatial`
        (eq. 14).  All units together cost one sparse matrix-vector product
        per draw.

        The draws condition on the posterior from the *full* data, which
        includes :math:`y_i`, so this is a smoother, not cross-validation;
        use :func:`neighbayes.diagnostics.spatial_kfold` to score held-out
        predictive accuracy.

        Parameters
        ----------
        thin : int, default 1
            Use every ``thin``-th draw of each chain.
        random_seed : int, optional
            Seed for the predictive draws.

        Returns
        -------
        arviz.InferenceData
            A ``predictions`` group with dimensions ``(chain, draw, obs)``
            holding ``y`` (draws), ``y_bp`` (the leave-one-out conditional
            mean) and ``y_tc`` (the marginal mean :math:`\\mu`).  Each
            unit's draws follow its own conditional; they are marginally
            correct per unit but not a joint draw across units.
        """
        kind = self._prediction_kind()
        n = self._X.shape[0]
        W = self._W_sparse
        Z = self._X
        if self._has_wx_in_beta and self._wx_column_indices:
            Z = np.hstack([self._X, self._WX])
        y = np.asarray(self._y, dtype=np.float64)
        if kind == "iid":
            diag_w = diag_wtw = np.zeros(n)
        else:
            diag_w = W.diagonal()
            diag_wtw = np.asarray(W.multiply(W).sum(axis=0)).ravel()  # (WᵀW)_ii

        beta, sigma, theta = self._draws(kind, thin)
        mean_fn = self._mean_solver(kind, W, n)
        rng = np.random.default_rng(random_seed)
        C, D = sigma.shape
        out = {k: np.empty((C, D, n)) for k in ("y", "y_bp", "y_tc")}
        for c in range(C):
            for d in range(D):
                th = float(theta[c, d])
                mu = mean_fn(th, Z @ beta[c, d])
                r = y - mu
                if kind == "iid":
                    Lr, Ldiag = r, np.ones(n)
                else:
                    Wr = W @ r
                    Lr = r - th * (Wr + W.T @ r) + th * th * (W.T @ Wr)
                    Ldiag = 1.0 - 2.0 * th * diag_w + th * th * diag_wtw
                # Both carry the same 1/σ², which cancels in the mean.
                m = y - Lr / Ldiag
                sd = sigma[c, d] / np.sqrt(Ldiag)
                out["y_tc"][c, d] = mu
                out["y_bp"][c, d] = m
                out["y"][c, d] = m + sd * rng.standard_normal(n)

        return az.from_dict(
            predictions=out,
            coords={"obs": np.arange(n)},
            dims={k: ["obs"] for k in out},
        )


def _row_labels(X_new: Any, oos: np.ndarray) -> np.ndarray:
    """Coordinate labels for new units: the DataFrame index, else positions."""
    if isinstance(X_new, pd.DataFrame):
        return X_new.index.to_numpy()
    return oos
