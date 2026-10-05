r"""Spatial multilevel model: nested levels, a graph and a process at each.

Units (level 0) nest in groups (level 1), which nest in larger groups, up to
level ``L``.  Each level has its own graph, covariates, innovation sd and
process, and each level's effect enters the equation of the level below:

.. math::

    \theta_\ell = \rho_\ell W_\ell \theta_\ell + X_\ell \beta_\ell
                  + \Delta_\ell \theta_{\ell+1} + \varepsilon_\ell,
    \qquad \varepsilon_\ell \sim N(0, \sigma_\ell^2 I),

with :math:`\theta_0 \equiv y`, :math:`\Delta_\ell` mapping each level-ℓ row
to its parent, and an error process
(:math:`\theta_\ell = X_\ell\beta_\ell + \Delta_\ell\theta_{\ell+1} + v_\ell`,
:math:`v_\ell = \lambda_\ell W_\ell v_\ell + \varepsilon_\ell`) or none in place
of the lag at any level.  Every graph spans all groups at its level, so
dependence crosses parent boundaries.  With two levels the model nests the
HSAR of Dong & Harris (2015) and the two-level models of Lacombe & McIntyre
(2016) and Wolf et al.; with more, a shift at an upper level reaches the
units through every filter below it.
"""

from __future__ import annotations

import warnings
from dataclasses import KW_ONLY, dataclass
from typing import Any

import numpy as np
import pandas as pd
import pytensor.tensor as pt
import scipy.sparse as sp
from libpysal.graph import Graph

from ..._lazy_deps import pm, xr
from ...samplers._registry import register
from ...samplers.multilevel import (
    PROCESSES,
    LevelSpec,
    MultilevelGibbsPriors,
    MultilevelStructure,
    run_multilevel_chain,
)
from .._base._shared import _parse_W, gelman_default_beta_prior
from ..base import SpatialModel
from ..priors import MultilevelPriors

_PARAM = {"lag": "rho", "error": "lam", "none": None}


@dataclass
class Level:
    """One level's equation for :class:`SpatialMultilevel`.

    Parameters
    ----------
    formula : str, optional
        At the units, ``"y ~ x1 + x2"``.  Above them, the level's covariates,
        ``"~ z1 + z2"``; the intercept is dropped (the units carry the only
        one).  ``None`` or ``"~ 1"`` means no covariates.
    data : DataFrame, optional
        The level's rows.  At the units, the data the formula reads.  Above,
        one row per group, matched to the groups by ``key`` (or the index).
    y, X : array-like, optional
        Matrix mode in place of ``formula``/``data`` (``y`` at the units only).
    W : libpysal.graph.Graph or scipy.sparse matrix
        The level's graph over **all** its groups, regardless of parent.
        Optional when ``process="none"`` and ``durbin=False``.
    process : {"lag", "error", "none"}, default "lag"
        The level's spatial process.
    durbin : bool, default False
        Add the spatial lags of the level's covariates, ``W_ℓ X_ℓ``.
    key : str, optional
        The id column shared with the level below: each lower row's value
        names its group here.  Looked up in the level below's data, then in
        the units' data (strict nesting is checked).
    groups : array-like, optional
        Matrix mode in place of ``key``: one group label per row of the
        level below.
    name : str, optional
        A label for summaries; defaults to ``key``.

    Notes
    -----
    The groups are ordered by the Graph's ids when ``W`` is a Graph (the
    level's data are reindexed to them); otherwise by the rows of ``data``,
    or by the sorted labels when there are no data.  A sparse ``W`` must
    follow that order.
    """

    formula: str | None = None
    data: pd.DataFrame | None = None
    _: KW_ONLY
    y: Any = None
    X: Any = None
    W: Any = None
    process: str = "lag"
    durbin: bool = False
    key: str | None = None
    groups: Any = None
    name: str | None = None


@dataclass
class _Resolved:
    """A level after linking and design construction."""

    name: str
    process: str
    durbin: bool
    W: sp.csr_matrix | None
    X: np.ndarray  # design, Durbin columns included
    names: list[str]  # its column names
    ids: list  # group labels in order
    parent: np.ndarray | None = None  # index into the next level's groups
    n_cov: int = 0  # covariate columns before the Durbin lags
    durbin_cols: list[int] | None = None  # covariate column each lag repeats


def _covariate_frame(level: Level, name: str, ids: pd.Index) -> pd.DataFrame | None:
    """The level's data with one row per group, in ``ids`` order."""
    if level.data is None:
        return None
    df = level.data
    if level.key is not None and level.key in df.columns:
        df = df.set_index(level.key)
    if not df.index.is_unique:
        raise ValueError(f"Level {name!r}: data has repeated ids.")
    missing = ids.difference(df.index)
    if len(missing):
        raise ValueError(
            f"Level {name!r}: no data rows for groups {list(missing[:5])}."
        )
    return df.loc[ids]


def _upper_design(level: Level, name: str, frame: pd.DataFrame | None, J: int):
    """``(X, names)`` for an upper level, without an intercept."""
    if level.formula is not None:
        rhs = level.formula.split("~", 1)[-1].strip()
        if rhs in ("", "1", "0", "-1", "1 - 1"):
            return np.empty((J, 0)), []
        if frame is None:
            raise ValueError(f"Level {name!r}: a formula needs data.")
        from formulaic import model_matrix

        mm = model_matrix(rhs, frame)
        X, names = np.asarray(mm, dtype=np.float64), list(mm.columns)
    elif level.X is not None:
        X = np.asarray(level.X, dtype=np.float64)
        if X.ndim == 1:
            X = X[:, None]
        names = (
            [str(c) for c in level.X.columns]
            if isinstance(level.X, pd.DataFrame)
            else [f"x{j}" for j in range(X.shape[1])]
        )
        if X.shape[0] != J:
            raise ValueError(f"Level {name!r}: X has {X.shape[0]} rows for {J} groups.")
    else:
        return np.empty((J, 0)), []
    keep = [
        j
        for j in range(X.shape[1])
        if names[j].lower() != "intercept" and not np.allclose(X[:, j], X[0, j])
    ]
    if len(keep) < X.shape[1] and level.formula is None:
        warnings.warn(
            f"Level {name!r}: dropped constant column(s); the units carry the "
            "only intercept.",
            stacklevel=4,
        )
    return X[:, keep], [names[j] for j in keep]


def _mean_rowsums(W: sp.csr_matrix, coef: np.ndarray, solve) -> tuple:
    """Per draw, the mean row sums of ``S = (I − ρW)⁻¹`` and of ``SW``.

    Closed form when ``W`` is row-standardized: connected rows of both sum to
    ``1/(1 − ρ)``; an isolate's row of ``S`` is ``e_i`` (sum 1) and of ``SW``
    zero.  Otherwise one solve per draw.
    """
    rows = np.asarray(W.sum(axis=1)).ravel()
    iso = rows == 0
    J, m = len(rows), int(iso.sum())
    if np.allclose(rows[~iso], 1.0):
        conn = (J - m) / (1.0 - coef)
        return (conn + m) / J, conn / J
    rs = np.array([solve(c, np.ones(J)).mean() for c in coef])
    rs_w = np.array([solve(c, rows).mean() for c in coef])
    return rs, rs_w


def _labels_from_units(unit_labels, unit_to_lower, lower_J, name):
    """Each lower group's label, read off its units; checks strict nesting."""
    codes, uniques = pd.factorize(pd.Series(unit_labels), sort=False)
    first = np.full(lower_J, -1)
    first[unit_to_lower[::-1]] = codes[::-1]
    if np.any(first < 0):
        raise ValueError(
            f"Level {name!r}: a group of the level below has no units, so its "
            "parent can't be read from the units' data. Put the key in the "
            "level below's data."
        )
    if np.any(codes != first[unit_to_lower]):
        raise ValueError(
            f"Level {name!r}: the levels are not nested — units of one group "
            "carry different keys."
        )
    return np.asarray(uniques)[first]


class SpatialMultilevel(SpatialModel):
    r"""Bayesian spatial multilevel model with a graph and a process at each level.

    .. math::

        \theta_\ell = \rho_\ell W_\ell \theta_\ell + X_\ell \beta_\ell
                      + \Delta_\ell \theta_{\ell+1} + \varepsilon_\ell,
        \qquad \ell = 0, \dots, L,\quad \theta_0 \equiv y,

    with a lag, an error process
    (:math:`v_\ell = \lambda_\ell W_\ell v_\ell + \varepsilon_\ell`) or none
    at each level and :math:`\varepsilon_\ell \sim N(0, \sigma_\ell^2 I)`.
    Each level's effect enters the equation of the level below, as in a
    hierarchical linear model, so a shift at level ℓ reaches the units through
    every filter beneath it.  Each graph spans all groups at its level,
    whatever their parent.

    Parameters
    ----------
    *levels : Level
        The units first, then each level above in turn.  A level's position
        is its index ℓ in the notation and in the posterior names.
    priors : dict or MultilevelPriors, optional
        See :class:`~neighbayes.models.priors.MultilevelPriors`.
    logdet_method : str, optional
        Log-determinant method at the units; upper levels choose their own.

    Notes
    -----
    Posterior variables follow the levels: ``rho_ℓ`` (or ``lam_ℓ`` for an
    error process), ``beta_ℓ``, ``sigma_ℓ``, and the effects ``theta_ℓ`` for
    ``ℓ ≥ 1`` (coordinate ``group_ℓ``).  NUTS also records ``theta_ℓ_raw``,
    the standardized innovations, and ``sigma2_0``.

    The Gibbs sampler (the default) draws all coefficients and effects in one
    sparse Gaussian block and integrates them out of every ρ and upper-level
    σ (``fit(parametrization="collapsed")``).  ``"centred"``, ``"noncentred"``
    and ``"interweave"`` update the upper levels' ρ and σ given the effects,
    given the standardized innovations, or both in turn — the schemes that
    carry over to non-Gaussian likelihoods.  ``fit(store_theta=False)`` skips
    storing the effects.

    Examples
    --------
    >>> from neighbayes.models import SpatialMultilevel, Level
    >>> m = SpatialMultilevel(
    ...     Level("score ~ frl", data=schools, W=W_schools),
    ...     Level("~ income", data=districts, W=W_districts, key="district"),
    ...     Level(data=states, W=W_states, key="state", process="error"),
    ... )  # doctest: +SKIP
    """

    _priors_cls = MultilevelPriors
    _likelihood: str = "gaussian_multilevel"
    _gibbs_key: tuple[str, str] | None = ("gaussian", "multilevel")
    _model_type = "multilevel"

    def __init__(
        self,
        *levels: Level,
        priors: dict | MultilevelPriors | None = None,
        logdet_method: str | None = None,
    ):
        if len(levels) < 2:
            raise ValueError(
                "SpatialMultilevel needs the units and at least one level above them."
            )
        for lv in levels:
            if not isinstance(lv, Level):
                raise TypeError(
                    f"Levels must be Level objects, got {type(lv).__name__}."
                )
            if lv.process not in PROCESSES:
                raise ValueError(
                    f"process must be one of {PROCESSES}, got {lv.process!r}."
                )
            if lv.W is None and (lv.process != "none" or lv.durbin):
                raise ValueError(
                    "A level with a lag, an error process or Durbin terms needs W."
                )
        unit = levels[0]
        super().__init__(
            formula=unit.formula,
            data=unit.data,
            y=unit.y,
            X=unit.X,
            W=unit.W,
            priors=priors,
            logdet_method=logdet_method,
        )
        self._y = np.asarray(self._y, dtype=np.float64).reshape(-1)
        self._jacobian_param = _PARAM[unit.process]
        self._resolve_levels(levels)

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def _resolve_levels(self, levels: tuple[Level, ...]) -> None:
        unit = levels[0]
        n = len(self._y)
        names0 = list(self._feature_names)
        X0 = self._X
        durbin0: list[int] = []
        if unit.durbin:
            durbin0 = list(self._wx_column_indices)
            X0 = np.hstack([X0, self._WX])
            names0 = names0 + [f"W*{self._feature_names[j]}" for j in durbin0]
        res = [
            _Resolved(
                name=unit.name or "units",
                process=unit.process,
                durbin=unit.durbin,
                W=self._W_sparse,
                X=X0,
                names=names0,
                ids=list(range(n)),
                n_cov=self._X.shape[1],
                durbin_cols=durbin0,
            )
        ]
        unit_data = unit.data
        lower_data = unit.data
        lower_J = n
        unit_to_lower = np.arange(n)
        for ell in range(1, len(levels)):
            lv = levels[ell]
            name = lv.name or lv.key or f"level{ell}"
            # --- links from the level below ---
            if lv.groups is not None:
                labels = np.asarray(lv.groups)
                if labels.shape != (lower_J,):
                    raise ValueError(
                        f"Level {name!r}: groups has {labels.size} labels for "
                        f"{lower_J} rows below."
                    )
            elif lv.key is not None:
                if lower_data is not None and lv.key in lower_data.columns:
                    labels = lower_data[lv.key].to_numpy()
                elif unit_data is not None and lv.key in unit_data.columns:
                    labels = _labels_from_units(
                        unit_data[lv.key].to_numpy(), unit_to_lower, lower_J, name
                    )
                else:
                    raise ValueError(
                        f"Level {name!r}: key {lv.key!r} is in neither the level "
                        "below's data nor the units' data."
                    )
            else:
                raise ValueError(
                    f"Level {name!r}: pass key= (a shared id column) or groups=."
                )
            # --- order of this level's groups ---
            if isinstance(lv.W, Graph):
                ids = pd.Index(lv.W.unique_ids)
            elif lv.data is not None:
                df = lv.data
                ids = pd.Index(
                    df[lv.key]
                    if (lv.key is not None and lv.key in df.columns)
                    else df.index
                )
            else:
                ids = pd.Index(pd.unique(labels)).sort_values()
            if not ids.is_unique:
                raise ValueError(f"Level {name!r}: group ids are not unique.")
            parent = ids.get_indexer(labels)
            if np.any(parent < 0):
                bad = pd.unique(labels[parent < 0])[:5]
                raise ValueError(f"Level {name!r}: no group for labels {list(bad)}.")
            J = len(ids)
            res[-1].parent = parent
            # --- graph and covariates ---
            W = None
            if lv.W is not None:
                W, _ = _parse_W(lv.W, J)
            frame = _covariate_frame(lv, name, ids)
            X, names = _upper_design(lv, name, frame, J)
            n_cov = X.shape[1]
            durbin_cols: list[int] = []
            if lv.durbin and n_cov:
                X = np.hstack([X, np.asarray(W @ X)])
                durbin_cols = list(range(n_cov))
                names = names + [f"W*{c}" for c in names[:n_cov]]
            res.append(
                _Resolved(
                    name=name,
                    process=lv.process,
                    durbin=lv.durbin,
                    W=W,
                    X=X,
                    names=names,
                    ids=list(ids),
                    n_cov=n_cov,
                    durbin_cols=durbin_cols,
                )
            )
            lower_data = frame if frame is not None else None
            lower_J = J
            unit_to_lower = parent[unit_to_lower]
        self._levels = res
        self._unit_ancestor = [np.arange(n)]
        anc = np.arange(n)
        for ell in range(len(res) - 1):
            anc = res[ell].parent[anc]
            self._unit_ancestor.append(anc)

    @property
    def L(self) -> int:
        """Number of levels above the units."""
        return len(self._levels) - 1

    @property
    def level_names(self) -> list[str]:
        """Each level's label, the units first."""
        return [r.name for r in self._levels]

    def _level_index(self, level: int | str) -> int:
        if isinstance(level, str):
            try:
                return self.level_names.index(level)
            except ValueError:
                raise ValueError(
                    f"Unknown level {level!r}; levels are {self.level_names}."
                ) from None
        if not 0 <= int(level) <= self.L:
            raise ValueError(f"level must be in 0..{self.L}, got {level}.")
        return int(level)

    def __repr__(self) -> str:
        parts = []
        for ell, r in enumerate(self._levels):
            parts.append(f"{ell}:{r.name}(J={len(r.ids)}, {r.process})")
        return f"SpatialMultilevel({', '.join(parts)})"

    # ------------------------------------------------------------------
    # Per-level logdets and priors
    # ------------------------------------------------------------------

    def _level_logdet(self, ell: int):
        """``(bounds, numpy fn, pytensor fn, grad vec fn)`` for level ℓ (cached)."""
        cache = self.__dict__.setdefault("_logdet_cache", {})
        if ell in cache:
            return cache[ell]
        if ell == 0:
            out = (
                self._logdet_bounds,
                self._logdet_numpy_fn,
                self._logdet_pytensor_fn,
                self._logdet_grad_numpy_vec_fn,
            )
        else:
            from ..._logdet import resolve_logdet_bounds
            from ..._logdet._factories import (
                make_logdet_fn,
                make_logdet_grad_numpy_vec_fn,
                make_logdet_numpy_fn,
            )

            W = self._levels[ell].W
            b = resolve_logdet_bounds(None, n=W.shape[0], priors=self.priors, W=W)
            kw = dict(method=b.method, rho_min=b.rho_min, rho_max=b.rho_max)
            out = (
                b,
                make_logdet_numpy_fn(W, None, **kw),
                make_logdet_fn(W, **kw),
                make_logdet_grad_numpy_vec_fn(W, None, **kw),
            )
        cache[ell] = out
        return out

    def _level_logdet_jax(self, ell: int):
        """``(kind, params)`` of level ℓ's log-determinant for the JAX sweep (cached)."""
        cache = self.__dict__.setdefault("_logdet_jax_cache", {})
        if ell not in cache:
            from ..._logdet._jax import logdet_jax_params

            b, *_ = self._level_logdet(ell)
            eigs = self._logdet_eigs if ell == 0 else None
            cache[ell] = logdet_jax_params(
                self._levels[ell].W, b.method, b.rho_min, b.rho_max, eigs=eigs
            )
        return cache[ell]

    def _beta_prior(self, ell: int) -> tuple[np.ndarray, np.ndarray]:
        r = self._levels[ell]
        if ell == 0:
            return self._resolved_beta_prior(r.X, r.names)
        k = r.X.shape[1]
        if k == 0:
            return np.empty(0), np.empty(0)
        mu, sd = gelman_default_beta_prior(self._y, r.X, r.names)
        for attr, arr in (("level_beta_mu", mu), ("level_beta_sigma", sd)):
            over = getattr(self.priors_obj, attr) or {}
            val = over.get(ell, over.get(r.name))
            if val is not None:
                arr[:] = np.broadcast_to(np.asarray(val, dtype=np.float64), (k,))
        return mu, sd

    def _within_group_variance(self) -> float:
        """Pooled variance of ``y`` within level-1 groups: the units' share of Var(y).

        The data-informed scale of the σ₀² prior.  ``Var(y)``, the single-level
        choice, also counts every upper level's variance, and an
        Inv-Γ(a, b) prior shifts the posterior mean of σ₀² by about ``2b/n``:
        with the upper levels holding most of Var(y) that was 3–4 posterior sd.
        Falls back to ``Var(y)`` when there are no groups or no group has two
        units.
        """
        y = self._y
        parent = self._levels[0].parent if len(self._levels) > 1 else None
        if parent is None:
            return float(np.var(y))
        counts = np.bincount(parent)
        means = np.bincount(parent, weights=y) / np.maximum(counts, 1)
        resid = y - means[parent]
        dof = y.size - np.count_nonzero(counts)
        return float(resid @ resid / dof) if dof > 0 else float(np.var(y))

    def _variance_priors(self) -> dict[str, float]:
        p = self.priors_obj
        sd_y = float(np.std(self._y)) or 1.0
        return {
            "sigma2_alpha": float(p.sigma2_alpha),
            "sigma2_beta": float(
                p.sigma2_beta
                if p.sigma2_beta is not None
                else self._within_group_variance()
            ),
            "sigma_nu": float(p.sigma_nu),
            "sigma_scale": float(p.sigma_scale if p.sigma_scale is not None else sd_y),
        }

    def _level_specs(self) -> list[LevelSpec]:
        specs = []
        for ell, r in enumerate(self._levels):
            mu, sd = self._beta_prior(ell)
            lo, hi, logdet, method = -1.0, 1.0, None, None
            if r.process != "none":
                b, logdet, _, _ = self._level_logdet(ell)
                lo, hi, method = b.rho_min, b.rho_max, b.method
            specs.append(
                LevelSpec(
                    X=r.X,
                    W=r.W,
                    process=r.process,
                    parent=r.parent,
                    beta_mu=mu,
                    beta_sigma=sd,
                    rho_lower=lo,
                    rho_upper=hi,
                    logdet=logdet,
                    logdet_method=method,
                )
            )
        return specs

    def _model_coords(self, extra: dict | None = None) -> dict:
        coords: dict[str, list] = {}
        for ell, r in enumerate(self._levels):
            if r.X.shape[1]:
                coords[f"coefficient_{ell}"] = list(r.names)
            if ell >= 1:
                coords[f"group_{ell}"] = list(r.ids)
        if extra:
            coords.update(extra)
        return coords

    def _param_names(self) -> list[str]:
        """Scalar and coefficient variables, level by level (no effects)."""
        out = []
        for ell, r in enumerate(self._levels):
            if r.process != "none":
                out.append(f"{_PARAM[r.process]}_{ell}")
            if r.X.shape[1]:
                out.append(f"beta_{ell}")
            out.append(f"sigma_{ell}")
        return out

    # ------------------------------------------------------------------
    # Gibbs
    # ------------------------------------------------------------------

    def _fit_gibbs(
        self,
        draws: int = 2000,
        tune: int = 1000,
        chains: int = 4,
        random_seed: int | None = None,
        thin: int = 1,
        n_jobs: int = -1,
        progressbar: bool = True,
        log_likelihood: bool = False,
        parametrization: str = "collapsed",
        store_theta: bool = True,
        backend: str = "numpy",
    ) -> xr.DataTree:
        from ...samplers._utils._seeds import spawn_chain_seeds
        from ...samplers.gaussian._chain_runner import run_chains

        specs = self._level_specs()
        if backend == "jax":
            for ell, spec in enumerate(specs):
                if spec.spatial:
                    spec.logdet_jax = self._level_logdet_jax(ell)
        st = MultilevelStructure(self._y, specs)
        priors = MultilevelGibbsPriors(**self._variance_priors())
        seeds = spawn_chain_seeds(random_seed, chains)
        if backend == "jax":
            from ...samplers.multilevel._jax import run_multilevel_jax

            results = run_multilevel_jax(
                st,
                priors,
                draws,
                tune,
                chains,
                seeds,
                thin=thin,
                parametrization=parametrization,
                store_theta=store_theta,
                store_log_lik=log_likelihood,
                progressbar=progressbar,
            )
            return self._assemble(results, log_likelihood, store_theta)

        def _run_one_chain(chain_id, seed, progress_manager=None, chain_id_kw=None):
            return run_multilevel_chain(
                st,
                priors,
                draws,
                tune,
                thin=thin,
                rng=np.random.default_rng(seed),
                parametrization=parametrization,
                store_theta=store_theta,
                store_log_lik=log_likelihood,
                chain_id=chain_id_kw if chain_id_kw is not None else chain_id,
                progress_manager=progress_manager,
            )

        results = run_chains(
            chain_fn=_run_one_chain,
            n_chains=chains,
            seeds=seeds,
            n_jobs=n_jobs,
            progressbar=progressbar,
            parallel=n_jobs != 1,
            draws=draws,
            tune=tune,
            model_type="multilevel",
        )
        return self._assemble(results, log_likelihood, store_theta)

    def _assemble(self, results, log_likelihood, store_theta) -> xr.DataTree:
        """Per-chain trace dicts → the posterior, named by level."""
        from ...samplers._utils._idata import gibbs_to_inference_data

        post: dict[str, np.ndarray] = {}
        dims: dict[str, list[str]] = {}
        for ell, r in enumerate(self._levels):
            if r.process != "none":
                post[f"{_PARAM[r.process]}_{ell}"] = np.stack(
                    [c["rho"][:, ell] for c in results]
                )
            if r.X.shape[1]:
                post[f"beta_{ell}"] = np.stack([c[f"beta_{ell}"] for c in results])
                dims[f"beta_{ell}"] = [f"coefficient_{ell}"]
            post[f"sigma_{ell}"] = np.stack([c["sigma"][:, ell] for c in results])
            if ell >= 1 and store_theta:
                post[f"theta_{ell}"] = np.stack([c[f"theta_{ell}"] for c in results])
                dims[f"theta_{ell}"] = [f"group_{ell}"]
        ll = (
            {"obs": np.stack([c["log_lik"] for c in results])}
            if log_likelihood
            else None
        )
        self._idata = gibbs_to_inference_data(
            posterior_samples=post,
            log_likelihood=ll,
            observed_data={"obs": self._y},
            coords=self._model_coords(),
            dims=dims,
        )
        return self._idata

    # ------------------------------------------------------------------
    # NUTS
    # ------------------------------------------------------------------

    def _build_pymc_model(self) -> pm.Model:
        """Non-centred PyMC model with the same priors as the Gibbs sampler."""
        from pytensor import sparse as pts

        from ..._ops import SparseSARSolveOp

        vp = self._variance_priors()
        L = self.L
        with pm.Model(coords=self._model_coords()) as model:
            theta: dict[int, Any] = {}
            for ell in range(L, 0, -1):
                r = self._levels[ell]
                J = len(r.ids)
                mean = pt.zeros(J)
                if r.X.shape[1]:
                    mu, sd = self._beta_prior(ell)
                    beta = pm.Normal(
                        f"beta_{ell}", mu=mu, sigma=sd, dims=f"coefficient_{ell}"
                    )
                    mean = mean + pt.dot(r.X, beta)
                if ell < L:
                    mean = mean + theta[ell + 1][r.parent]
                sigma = pm.HalfStudentT(
                    f"sigma_{ell}", nu=vp["sigma_nu"], sigma=vp["sigma_scale"]
                )
                raw = pm.Normal(f"theta_{ell}_raw", 0.0, 1.0, dims=f"group_{ell}")
                if r.process == "none":
                    th = mean + sigma * raw
                else:
                    b, *_ = self._level_logdet(ell)
                    coef = pm.Uniform(
                        f"{_PARAM[r.process]}_{ell}", lower=b.rho_min, upper=b.rho_max
                    )
                    op = SparseSARSolveOp(r.W)
                    if r.process == "lag":
                        th = op(coef, mean + sigma * raw)
                    else:
                        th = mean + sigma * op(coef, raw)
                theta[ell] = pm.Deterministic(f"theta_{ell}", th, dims=f"group_{ell}")

            r0 = self._levels[0]
            mu0, sd0 = self._beta_prior(0)
            mean0 = theta[1][r0.parent]
            if r0.X.shape[1]:
                beta0 = pm.Normal("beta_0", mu=mu0, sigma=sd0, dims="coefficient_0")
                mean0 = mean0 + pt.dot(r0.X, beta0)
            sigma2_0 = pm.InverseGamma(
                "sigma2_0", alpha=vp["sigma2_alpha"], beta=vp["sigma2_beta"]
            )
            sigma0 = pm.Deterministic("sigma_0", pt.sqrt(sigma2_0))
            if r0.process == "none":
                pm.Normal("obs", mu=mean0, sigma=sigma0, observed=self._y)
                return model
            b0, _, logdet0, _ = self._level_logdet(0)
            coef0 = pm.Uniform(
                f"{_PARAM[r0.process]}_0", lower=b0.rho_min, upper=b0.rho_max
            )
            if r0.process == "lag":
                pm.Normal(
                    "obs", mu=coef0 * self._Wy + mean0, sigma=sigma0, observed=self._y
                )
            else:
                u = self._y - mean0
                eps = (
                    u
                    - coef0
                    * pts.structured_dot(self._W_pt_sparse, u[:, None]).flatten()
                )
                pm.Potential(
                    "obs_loglik", pm.logp(pm.Normal.dist(0.0, sigma0), eps).sum()
                )
            pm.Potential("jacobian", logdet0(coef0))
        return model

    # ------------------------------------------------------------------
    # Posterior helpers
    # ------------------------------------------------------------------

    def _draws(self, var: str) -> np.ndarray:
        """Posterior draws flattened over chains: ``(G, ...)``."""
        x = self._idata.posterior[var].values
        return x.reshape(-1, *x.shape[2:])

    def _coef_draws(self, ell: int) -> np.ndarray:
        r = self._levels[ell]
        if r.process == "none":
            G = self._draws(f"sigma_{ell}").shape[0]
            return np.zeros(G)
        return self._draws(f"{_PARAM[r.process]}_{ell}")

    def _require_theta(self) -> None:
        self._require_fit()
        if "theta_1" not in self._idata.posterior:
            raise RuntimeError(
                "The effects were not stored; refit without store_theta=False."
            )

    def summary(self, var_names: list | None = None, **kwargs) -> pd.DataFrame:
        """Posterior summary of every level's parameters (effects excluded).

        Pass ``var_names`` to choose, e.g. ``["theta_1"]`` for the effects.
        """
        return super().summary(
            var_names=var_names if var_names is not None else self._param_names(),
            **kwargs,
        )

    def _filter_solve(self, ell: int, coef: float, rhs: np.ndarray) -> np.ndarray:
        """``S_ℓ rhs``: ``(I − ρW)⁻¹ rhs`` for a lag, ``rhs`` otherwise."""
        r = self._levels[ell]
        if r.process != "lag":
            return rhs
        from scipy.sparse.linalg import splu

        A = (sp.eye(r.W.shape[0], format="csc") - coef * r.W).tocsc()
        return splu(A).solve(np.asarray(rhs, dtype=np.float64))

    def _fitted_mean_from_posterior(self) -> np.ndarray:
        """``E[y]`` at the posterior means of the parameters and the effects."""
        self._require_theta()
        r0 = self._levels[0]
        mean = self._draws("theta_1").mean(0)[r0.parent]
        if r0.X.shape[1]:
            mean = mean + r0.X @ self._draws("beta_0").mean(0)
        return self._filter_solve(0, float(self._coef_draws(0).mean()), mean)

    def posterior_predictive(
        self, max_draws: int | None = None, random_seed: int | None = None
    ) -> np.ndarray:
        """Replicated outcomes ``(G, n)``, one per draw, given that draw's effects."""
        self._require_theta()
        rng = np.random.default_rng(random_seed)
        r0 = self._levels[0]
        idx = self._draw_subset(max_draws, rng)
        theta1 = self._draws("theta_1")[idx]
        beta0 = self._draws("beta_0")[idx] if r0.X.shape[1] else None
        sig = self._draws("sigma_0")[idx]
        coef = self._coef_draws(0)[idx]
        n = len(self._y)
        out = np.empty((len(idx), n))
        for g in range(len(idx)):
            mean = theta1[g][r0.parent]
            if beta0 is not None:
                mean = mean + r0.X @ beta0[g]
            eps = sig[g] * rng.standard_normal(n)
            if r0.process == "lag":
                out[g] = self._filter_solve(0, coef[g], mean + eps)
            elif r0.process == "error":
                from scipy.sparse.linalg import splu

                A = (sp.eye(n, format="csc") - coef[g] * r0.W).tocsc()
                out[g] = mean + splu(A).solve(eps)
            else:
                out[g] = mean + eps
        return out

    def _draw_subset(self, max_draws: int | None, rng=None) -> np.ndarray:
        G = self._draws("sigma_0").shape[0]
        if max_draws is None or max_draws >= G:
            return np.arange(G)
        return np.linspace(0, G - 1, int(max_draws)).round().astype(int)

    # ------------------------------------------------------------------
    # Effects
    # ------------------------------------------------------------------

    def _effect_columns(self, ell: int) -> tuple[list[str], np.ndarray, np.ndarray]:
        """``(names, β index, δ index or −1)`` for each reported covariate."""
        r = self._levels[ell]
        if ell == 0:
            cov = list(self._nonintercept_indices)
            lag_of = {c: r.n_cov + i for i, c in enumerate(r.durbin_cols or [])}
        else:
            cov = list(range(r.n_cov))
            lag_of = {c: r.n_cov + i for i, c in enumerate(r.durbin_cols or [])}
        names = [r.names[c] for c in cov]
        b_idx = np.asarray(cov, dtype=int)
        d_idx = np.asarray([lag_of.get(c, -1) for c in cov], dtype=int)
        return names, b_idx, d_idx

    def _within_traces(self, ell: int, coef: np.ndarray):
        """Per draw: mean diag of ``S`` and ``SW``, mean row sums of ``S`` and ``SW``."""
        r = self._levels[ell]
        J = r.W.shape[0] if r.W is not None else len(r.ids)
        G = len(coef)
        if r.W is None:
            one = np.ones(G)
            return one, np.zeros(G), one, np.zeros(G)
        w_diag = float(r.W.diagonal().mean())
        w_rows = np.asarray(r.W.sum(axis=1)).ravel()
        if r.process != "lag":
            one = np.ones(G)
            return one, np.full(G, w_diag), one, np.full(G, w_rows.mean())
        _, _, _, grad = self._level_logdet(ell)
        g = np.asarray(grad(coef), dtype=np.float64)
        md = 1.0 - (coef / J) * g
        md_w = -g / J
        rs, rs_w = _mean_rowsums(r.W, coef, lambda c, b: self._filter_solve(ell, c, b))
        return md, md_w, rs, rs_w

    def _compute_spatial_effects_posterior(self):
        """Effects of the units' covariates through the units' filter."""
        return self._effects_at(0, on="units")[1:]

    def _effects_at(
        self,
        ell: int,
        on: str = "units",
        max_draws: int | None = None,
        exact_max: int = 500,
        n_probes: int = 64,
        random_seed: int | None = None,
    ):
        """``(names, direct, indirect, total)``, each ``(G, k)``."""
        self._require_fit()
        names, b_idx, d_idx = self._effect_columns(ell)
        rng = np.random.default_rng(random_seed)
        idx = self._draw_subset(max_draws, rng)
        beta = self._draws(f"beta_{ell}")[idx] if self._levels[ell].X.shape[1] else None
        if beta is None or not len(b_idx):
            G = len(idx)
            empty = np.empty((G, 0))
            return names, empty, empty, empty
        B = beta[:, b_idx]
        Dl = np.where(d_idx[None, :] >= 0, beta[:, np.maximum(d_idx, 0)], 0.0)
        coef = self._coef_draws(ell)[idx]

        if ell == 0 or on == "level":
            md, md_w, rs, rs_w = self._within_traces(ell, coef)
            direct = md[:, None] * B + md_w[:, None] * Dl
            total = rs[:, None] * B + rs_w[:, None] * Dl
            return names, direct, total - direct, total
        if on != "units":
            raise ValueError(f"on must be 'units' or 'level', got {on!r}.")

        # Composed: M = S_0 Δ_0 S_1 ⋯ Δ_{ℓ−1} S_ℓ, applied to blocks of columns.
        r = self._levels[ell]
        J = len(r.ids)
        n = len(self._y)
        anc = self._unit_ancestor[ell]
        exact = J <= exact_max
        has_d = bool(np.any(d_idx >= 0))
        W_l = r.W
        w_rows = np.asarray(W_l.sum(axis=1)).ravel() if has_d else None
        coefs = [self._coef_draws(m)[idx] for m in range(ell + 1)]

        from scipy.sparse.linalg import splu

        def factors(g):
            """One LU per lag level for draw ``g`` (``None`` elsewhere)."""
            lus = []
            for m in range(ell + 1):
                rm = self._levels[m]
                if rm.process == "lag":
                    A = sp.eye(rm.W.shape[0], format="csc") - coefs[m][g] * rm.W
                    lus.append(splu(A.tocsc()))
                else:
                    lus.append(None)
            return lus

        def through(lus, V):
            V = np.asarray(V, dtype=np.float64)
            for m in range(ell, -1, -1):
                if lus[m] is not None:
                    V = lus[m].solve(V)
                if m > 0:
                    V = V[self._levels[m - 1].parent]
            return V

        def own_trace(lus, cols_fn):
            """``(1/n) Σ_i (M E)[i, anc(i)]`` over E = I (exact) or probes."""
            if exact:
                acc = 0.0
                for lo in range(0, J, 64):
                    hi = min(J, lo + 64)
                    E = np.zeros((J, hi - lo))
                    E[np.arange(lo, hi), np.arange(hi - lo)] = 1.0
                    V = through(lus, cols_fn(E))
                    sel = (anc >= lo) & (anc < hi)
                    acc += V[np.flatnonzero(sel), anc[sel] - lo].sum()
                return acc / n
            V = through(lus, cols_fn(Zp))
            return float(np.sum(Zp[anc] * V)) / (n * Zp.shape[1])

        Zp = None if exact else rng.choice([-1.0, 1.0], size=(J, n_probes))
        G = len(idx)
        tr_b, tr_d, tot_b, tot_d = (np.zeros(G) for _ in range(4))
        for g in range(G):
            lus = factors(g)
            tr_b[g] = own_trace(lus, lambda E: E)
            tot_b[g] = through(lus, np.ones(J)).mean()
            if has_d:
                tr_d[g] = own_trace(lus, lambda E: W_l @ E)
                tot_d[g] = through(lus, w_rows).mean()
        direct = tr_b[:, None] * B + tr_d[:, None] * Dl
        total = tot_b[:, None] * B + tot_d[:, None] * Dl
        return names, direct, total - direct, total

    def spatial_effects(
        self,
        level: int | str = 0,
        on: str = "units",
        return_posterior_samples: bool = False,
        max_draws: int | None = None,
        exact_max: int = 500,
        n_probes: int = 64,
        random_seed: int | None = None,
    ):
        r"""Direct, indirect and total effects of one level's covariates.

        Parameters
        ----------
        level : int or str, default 0
            The level whose covariates shift, by number or name.
        on : {"units", "level"}, default "units"
            ``"units"``: the effect on the outcome, through the composed
            multiplier :math:`S_0\Delta_0 S_1 \cdots \Delta_{\ell-1} S_\ell`
            (:math:`S_m = (I-\rho_m W_m)^{-1}` for a lag, ``I`` otherwise).
            *Direct* is the mean effect on a unit of a shift in its own
            group; *total*, of a shift in every group; *indirect*, the
            difference — what arrives from other groups through the filters.
            ``"level"``: the effect on the level's own effects
            :math:`\theta_\ell`, through :math:`S_\ell` alone.  The two agree
            at ``level=0``.
        return_posterior_samples : bool, default False
            Also return the per-draw arrays.
        max_draws : int, optional
            Thin to this many draws; each draw factors every filter below.
        exact_max, n_probes : int
            The own-group trace is exact when the level has at most
            ``exact_max`` groups, else estimated with ``n_probes`` Rademacher
            probes.
        random_seed : int, optional
            For the probes.
        """
        from ...diagnostics.spatial_effects import _build_effects_dataframe

        ell = self._level_index(level)
        names, direct, indirect, total = self._effects_at(
            ell, on, max_draws, exact_max, n_probes, random_seed
        )
        df = _build_effects_dataframe(
            direct_samples=direct,
            indirect_samples=indirect,
            total_samples=total,
            feature_names=names,
            model_type=f"SpatialMultilevel[{self._levels[ell].name}]",
        )
        if return_posterior_samples:
            return df, {"direct": direct, "indirect": indirect, "total": total}
        return df


# ---------------------------------------------------------------------------
# Gibbs registry entry
# ---------------------------------------------------------------------------


def _run_multilevel(
    model,
    *,
    draws,
    tune,
    chains,
    random_seed,
    thin,
    n_jobs,
    progressbar,
    backend,
    log_likelihood=False,
    parametrization="collapsed",
    store_theta=True,
):
    """Registry runner: NumPy (CHOLMOD) or JAX (compiled sweeps, sparsax)."""
    return model._fit_gibbs(
        draws=draws,
        tune=tune,
        chains=chains,
        random_seed=random_seed,
        thin=thin,
        n_jobs=n_jobs,
        progressbar=progressbar,
        log_likelihood=log_likelihood,
        parametrization=parametrization,
        store_theta=store_theta,
        backend=backend,
    )


register(
    "gaussian",
    "multilevel",
    run=_run_multilevel,
    backends={"numpy", "jax"},
    options={"parametrization", "store_theta"},
    skips_log_likelihood=True,
)
