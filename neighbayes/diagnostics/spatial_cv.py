"""Spatial block cross-validation for fitted Bayesian spatial models.

Implements the refit-based spatial k-fold predictive evaluation of
:cite:t:`roberts2017CrossvalidationStrategies` for the models in
:mod:`neighbayes.models`.

The estimator avoids the well-known failures of PSIS-LOO on spatially
dependent data (importance ratios assume the per-observation likelihoods
factorize across observations, which spatial models violate by
construction).  Each fold refits the model on the training subset and
evaluates ``log p(y_test | y_train, theta)`` under the *full-data*
joint Gaussian induced by the model:

.. math::

    \\log p(y_{\\text{test}} \\mid y_{\\text{train}}, \\theta) =
    \\tfrac{1}{2} \\log |\\Lambda_{tt}|
    - \\tfrac{n_{\\text{test}}}{2} \\log(2\\pi)
    - \\tfrac{1}{2} z_{\\text{test}}^{\\top} \\Lambda_{tt}^{-1} z_{\\text{test}},

where :math:`\\Lambda` is the full :math:`n\\times n` precision matrix at
draw :math:`\\theta`, :math:`\\Lambda_{tt}` is its test-block, and
:math:`z = \\Lambda(y - \\mu)` with :math:`\\mu` the implied marginal
mean (:math:`A^{-1}X\\beta` for SAR/SDM, :math:`X\\beta` for SEM/SDEM,
:math:`X\\beta` for OLS/SLX).  Per-fold elpd is obtained by
``logsumexp`` over posterior draws.
"""

from __future__ import annotations

import contextlib
import logging
import os
import warnings
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.special import logsumexp

__all__ = ["SpatialCVResult", "spatial_kfold"]


@contextlib.contextmanager
def _silence_fit():
    """Suppress stdout/stderr and PyMC logger chatter from a model fit."""
    pymc_logger = logging.getLogger("pymc")
    prev_level = pymc_logger.level
    pymc_logger.setLevel(logging.ERROR)
    devnull = open(os.devnull, "w")
    try:
        with (
            contextlib.redirect_stdout(devnull),
            contextlib.redirect_stderr(devnull),
            warnings.catch_warnings(),
        ):
            warnings.simplefilter("ignore")
            yield
    finally:
        devnull.close()
        pymc_logger.setLevel(prev_level)


@dataclass
class SpatialCVResult:
    """Result of :func:`spatial_kfold`.

    Attributes
    ----------
    elpd : float
        Expected log pointwise predictive density summed over folds.
    se : float
        Standard error of ``elpd``, estimated as
        ``sqrt(n * var(per_obs_elpd))`` where ``per_obs_elpd`` spreads each
        fold's elpd uniformly over its observations.
    elpd_per_fold : np.ndarray
        Per-fold elpd of shape ``(n_folds,)``.
    n_per_fold : np.ndarray
        Number of observations in each fold, shape ``(n_folds,)``.
    fold_ids : np.ndarray
        Fold in which each observation was held out, shape ``(n,)``; ``-1``
        for observations never held out.  When a splitter tests an
        observation more than once, the last fold wins.
    n_folds : int
        Number of folds actually used.
    method : str
        Class name of the splitter that produced the folds.
    """

    elpd: float
    se: float
    elpd_per_fold: np.ndarray
    n_per_fold: np.ndarray
    fold_ids: np.ndarray
    n_folds: int
    method: str


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _model_kind(model: Any) -> str:
    """Classify a fitted model as ``"lag"``, ``"error"`` or ``"iid"``."""
    name = type(model).__name__.upper()
    if name in ("SAR", "SDM"):
        return "lag"
    if name in ("SEM", "SDEM"):
        return "error"
    if name in ("OLS", "SLX"):
        return "iid"
    raise NotImplementedError(
        f"spatial_kfold is not implemented for model class {type(model).__name__}."
    )


_WX_MODELS = {"SLX", "SDM", "SDEM"}


def _full_design(model: Any) -> np.ndarray:
    """Effective design matrix matching the model's ``beta`` posterior.

    SLX / SDM / SDEM concatenate ``[X, WX]`` into ``beta``; the other
    model families fit ``beta`` to ``X`` alone even though ``_WX`` is
    cached on the model.
    """
    uses_wx = type(model).__name__.upper() in _WX_MODELS
    if not uses_wx or model._WX.shape[1] == 0:
        return np.asarray(model._X, dtype=np.float64)
    return np.hstack([np.asarray(model._X), np.asarray(model._WX)]).astype(np.float64)


def _stack_draws(idata, name: str) -> np.ndarray:
    """Stack chain × draw → sample for posterior variable ``name``."""
    arr = idata.posterior[name]
    return arr.stack(sample=("chain", "draw")).transpose("sample", ...).values


def _refit_on_train(
    model: Any,
    train_idx: np.ndarray,
    fit_kwargs: dict,
) -> Any:
    """Construct a fresh model instance on the training subset and fit it."""
    W_train: Optional[sp.spmatrix]
    if model._W_sparse is not None:
        W_sub = model._W_sparse[train_idx, :][:, train_idx].tocsr()
        # Subsetting a globally row-standardized W breaks row-normalization
        # (some neighbors fall outside train_idx, so row sums < 1). Re-
        # standardize rows so the training W matches the original convention.
        row_sums = np.asarray(W_sub.sum(axis=1)).ravel()
        inv = np.zeros_like(row_sums)
        nz = row_sums > 0
        inv[nz] = 1.0 / row_sums[nz]
        W_train = sp.diags(inv) @ W_sub
        W_train = W_train.tocsr()
    else:
        W_train = None
    # Named columns, so the refit lags the same covariates (``w_vars``).
    X_train = pd.DataFrame(model._X[train_idx, :], columns=model._feature_names)
    new = model.__class__(
        y=model._y[train_idx],
        X=X_train,
        W=W_train,
        priors=model.priors_obj,
        logdet_method=model.logdet_method,
        robust=model.robust,
        w_vars=model._wx_feature_names if W_train is not None else None,
        logdet_refit=model.logdet_refit,
        logdet_refit_pad_sd=model.logdet_refit_pad_sd,
        logdet_aaa_check=model.logdet_aaa_check,
        logdet_probe_check=model.logdet_probe_check,
    )
    new.fit(**fit_kwargs)
    return new


def _fold_elpd(
    refit: Any,
    *,
    y_full: np.ndarray,
    design_full: np.ndarray,
    W_full: Optional[sp.csr_matrix],
    test_idx: np.ndarray,
    kind: str,
) -> float:
    """log E_posterior[ p(y_test | y_train, theta) ] from refit's draws."""
    idata = refit.inference_data
    beta = _stack_draws(idata, "beta")  # (G, k_design)
    sigma = _stack_draws(idata, "sigma").reshape(-1)  # (G,)
    if beta.shape[1] != design_full.shape[1]:
        raise ValueError(
            "Posterior beta dimension does not match the full-data design "
            f"matrix ({beta.shape[1]} vs {design_full.shape[1]}). This can "
            "happen if a column became constant on the training subset."
        )

    G = beta.shape[0]
    n = y_full.shape[0]

    if kind != "iid" and W_full is None:
        raise ValueError(f"Model kind {kind!r} requires W_full but it is None.")

    from .._prediction import GaussianConditional

    cond = GaussianConditional(None if kind == "iid" else W_full, test_idx, n)
    if kind == "iid":
        spatial = np.zeros(G)
    else:
        spatial = _stack_draws(idata, "rho" if kind == "lag" else "lam").reshape(-1)

    # Cached symbolic analysis: A = I - rho W shares one sparsity pattern
    # across all G draws (only rho rescales the values).
    cached_solver = None
    if kind == "lag":
        from ..samplers._utils._sparsax_utils import CachedSparseSolver

        cached_solver = CachedSparseSolver([W_full], n)

    y_test = y_full[test_idx]
    log_p = np.empty(G, dtype=np.float64)
    for g in range(G):
        theta = float(spatial[g])
        Xb = design_full @ beta[g]
        mu = cached_solver.solve([-theta], Xb) if kind == "lag" else Xb
        cond.update(theta, float(sigma[g]))
        log_p[g] = cond.logpdf(y_test, cond.mean(mu, y_full))

    return float(logsumexp(log_p) - np.log(G))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def spatial_kfold(
    model: Any,
    splitter: Any,
    *,
    geometry: Optional[Any] = None,
    groups: Optional[Any] = None,
    draws: int = 400,
    tune: int = 400,
    chains: int = 2,
    random_seed: int = 0,
    progressbar: bool = True,
    verbose: bool = False,
    **fit_kwargs: Any,
) -> SpatialCVResult:
    """Spatial block cross-validation for a fitted Bayesian spatial model.

    Refits the model on each training fold and evaluates the conditional
    Gaussian predictive density of the held-out fold under the full-data
    joint implied by the model.

    Parameters
    ----------
    model : SpatialModel
        A model from :mod:`neighbayes.models`.  The model must have been
        constructed (its ``_X``, ``_y`` and ``_W_sparse`` will be used for
        prediction); it does **not** need to be fit, since fold-specific
        refits are performed internally.
    splitter : cross-validation splitter
        Any object whose ``split`` method yields ``(train_idx, test_idx)``
        pairs, called as ``splitter.split(geometry)`` or, when ``groups`` is
        given, ``splitter.split(geometry, groups=groups)``: the scikit-learn
        ``BaseCrossValidator`` protocol.  Spatial splitters from
        `geovalidate <https://github.com/ljwolf/geovalidate/>`_ (e.g.
        ``HilbertKFold``, ``CellStratifiedKFold``, ``BallKFold``) take the
        geometry as ``X``; scikit-learn splitters such as ``GroupKFold`` or
        ``PredefinedSplit`` express blocks built by other means.
    geometry : geopandas.GeoSeries, optional
        Passed to ``splitter.split`` as ``X``.  Required by geometry-aware
        splitters; when omitted, a placeholder of ``n`` rows is passed.
    groups : array-like, optional
        Group label per observation, forwarded to ``splitter.split``
        (e.g. for ``GroupKFold``).
    draws, tune, chains, random_seed
        Forwarded to :meth:`SpatialModel.fit` for each per-fold refit.
        Defaults are deliberately modest to keep CV affordable.
    progressbar : bool, default True
        If True, display a fold-level progress bar (via ``tqdm``)
        showing CV progress.  Independent of any per-chain progress bar
        inside :meth:`SpatialModel.fit`, which is always disabled.
    verbose : bool, default False
        If True, allow per-fold ``fit`` calls to print their usual
        sampler / compile messages to stdout/stderr.  When False (the
        default) those messages — along with PyMC's ``INFO`` logger
        output and warnings — are suppressed so only the fold-level
        progress bar is visible.
    **fit_kwargs
        Extra keyword arguments forwarded to :meth:`SpatialModel.fit`.

    Returns
    -------
    SpatialCVResult

    Notes
    -----
    When ``splitter`` produces folds whose test sets do not form a
    disjoint partition of the data (e.g. ``LeaveBallOut`` with an
    exclusion buffer, or any splitter where some observations are tested
    multiple times or not at all), the per-observation accounting used
    to estimate ``se`` is undefined.  In that case ``se`` is set to
    ``nan``; ``elpd_per_fold`` and ``elpd`` remain valid.

    Notes
    -----
    Computation is :math:`O(K \\cdot G \\cdot \\text{nnz}(W))` per fold
    plus the cost of refitting; spatial folds are typically a handful
    (5\u201310).  For ``OLS``/``SLX`` the predictive
    collapses to the standard independent Gaussian and the per-fold
    cost is :math:`O(G \\cdot n_{\\text{test}} \\cdot k)`.
    """
    n = int(model._y.shape[0])
    kind = _model_kind(model)
    if getattr(model, "robust", False):
        raise NotImplementedError(
            "spatial_kfold scores the Gaussian conditional density, which does "
            "not hold for Student-t errors (robust=True)."
        )
    if not callable(getattr(splitter, "split", None)):
        raise TypeError(
            "splitter must expose a split(X) method yielding (train_idx, test_idx)."
        )
    split_X = geometry if geometry is not None else np.zeros((n, 1))
    split_kw = {} if groups is None else {"groups": groups}
    folds = [
        (np.asarray(tr, dtype=np.int64), np.asarray(te, dtype=np.int64))
        for tr, te in splitter.split(split_X, **split_kw)
    ]
    method = type(splitter).__name__
    fold_ids_out = np.full(n, -1, dtype=np.int64)
    for f, (_, te) in enumerate(folds):
        fold_ids_out[te] = f  # last-writer-wins for overlapping splitters

    n_folds = len(folds)
    if n_folds < 2:
        raise ValueError(f"spatial_kfold requires at least 2 folds (got {n_folds}).")

    y_full = np.asarray(model._y, dtype=np.float64)
    design_full = _full_design(model)
    W_full = model._W_sparse
    if W_full is not None and not sp.isspmatrix_csr(W_full):
        W_full = W_full.tocsr()

    base_fit_kwargs = dict(
        draws=draws,
        tune=tune,
        chains=chains,
        random_seed=random_seed,
        progressbar=False,  # per-chain bars are noisy across folds
    )
    base_fit_kwargs.update(fit_kwargs)

    elpd_per_fold = np.empty(n_folds, dtype=np.float64)
    n_per_fold = np.empty(n_folds, dtype=np.int64)
    per_obs_elpd = np.full(n, np.nan, dtype=np.float64)
    test_counts = np.zeros(n, dtype=np.int64)

    if progressbar:
        try:
            from tqdm.auto import tqdm

            fold_iter = tqdm(
                enumerate(folds),
                total=n_folds,
                desc="spatial CV",
                unit="fold",
            )
        except ImportError:
            fold_iter = enumerate(folds)
    else:
        fold_iter = enumerate(folds)

    fit_ctx = contextlib.nullcontext() if verbose else _silence_fit()

    with fit_ctx:
        for f, (train_idx, test_idx) in fold_iter:
            if test_idx.size == 0 or train_idx.size == 0:
                raise ValueError(
                    f"Fold {f} produces an empty test or training partition."
                )
            refit = _refit_on_train(model, train_idx, base_fit_kwargs)
            fold_elpd = _fold_elpd(
                refit,
                y_full=y_full,
                design_full=design_full,
                W_full=W_full,
                test_idx=test_idx,
                kind=kind,
            )
            elpd_per_fold[f] = fold_elpd
            n_per_fold[f] = test_idx.size
            per_obs_elpd[test_idx] = fold_elpd / test_idx.size
            test_counts[test_idx] += 1

    elpd_total = float(elpd_per_fold.sum())
    is_partition = bool(np.all(test_counts == 1))
    if is_partition and n > 1:
        se = float(np.sqrt(n * np.var(per_obs_elpd, ddof=1)))
    elif n_folds > 1:
        # Non-partition splitter: fall back to fold-level SE estimate.
        se = float(np.std(elpd_per_fold, ddof=1) * np.sqrt(n_folds))
    else:
        se = 0.0

    return SpatialCVResult(
        elpd=elpd_total,
        se=se,
        elpd_per_fold=elpd_per_fold,
        n_per_fold=n_per_fold,
        fold_ids=fold_ids_out,
        n_folds=n_folds,
        method=method,
    )
