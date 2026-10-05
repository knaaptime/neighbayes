r"""Unit (or pair) and period effects for the count panels, without dummies.

The within transform that gives the Gaussian panels their fixed effects
(Frisch–Waugh–Lovell) does not carry to a log link, so the count panels keep the
raw counts and carry the effects as parameters:

* **Group effects** ``c`` (one per spatial unit, or per origin–destination pair
  in a flow panel) sit outside the spatial filter.  ``(I_T ⊗ A⁻¹)`` commutes
  with the group dummies ``1_T ⊗ I``, so effects inside the filter are the same
  model with ``c = A⁻¹μ``.  They are **partially pooled**: ``c_g ~ N(μ, σ²)``
  with the sd σ learned (half-t(3, ``group_effect_sd_scale``) prior), so a
  group seen in few periods is shrunk toward the common mean in proportion to
  how little its data say.  The samplers integrate ``c`` out of every ρ update,
  draw it jointly with β, then draw σ (see
  :mod:`~neighbayes.samplers._utils._group_effects`); it is stored as a group
  index, never as columns of ``X``.
* **Period effects** ``τ``, also outside the filter, join the coefficient
  block under a fixed prior (there are too few to learn their sd): ``T − 1`` of
  them (first period the baseline) beside group effects, all ``T`` without
  (the intercept is then absorbed).

``effects``: ``0`` pooled, ``1`` group effects, ``2`` period effects, ``3`` both.

Group effects absorb nothing: the intercept and time-invariant columns stay in
the design, identified by between-group variation and anchoring the level (a
fixed-prior effect that absorbed the intercept left the filter's amplification
of the level unconstrained, and ρ drifted to a degenerate mode near 1).  Period
effects absorb the columns that vary only over time.  With ``mundlak=True`` the
group means of the time-varying columns are added to the design *inside* the
filter (Mundlak 1978; the right place in a reduced form, where the effects
correlate with the filtered design), which absorbs a correlation between the
effects and ``X``; their coefficients are not reported as effects.
"""

from __future__ import annotations

import warnings
from typing import Optional

import numpy as np

from ..._lazy_deps import xr

#: Per-draw group effects are kept when they fit in this many bytes.
_GROUP_DRAWS_MAX_BYTES = 500 * 1024**2


def _resolve_count_effects(effects) -> int:
    from ..panel_base import _resolve_effects

    if isinstance(effects, (np.integer,)):
        effects = int(effects)
    return _resolve_effects(effects)


def require_counts(y, cls_name: str) -> np.ndarray:
    """Validate and coerce a response to non-negative integer counts."""
    y_arr = np.asarray(y)
    if not np.issubdtype(y_arr.dtype, np.integer):
        y_rounded = np.round(y_arr).astype(np.int64)
        if not np.allclose(y_arr, y_rounded):
            raise ValueError(
                f"{cls_name} requires integer-valued observations; got dtype "
                f"{y_arr.dtype} with non-integer values."
            )
        y_arr = y_rounded
    if np.any(y_arr < 0):
        raise ValueError(f"{cls_name} requires non-negative integer observations.")
    return y_arr


def _panel_column_kinds(M: np.ndarray, n_groups: int, T: int):
    """``(period_only, time_varying)`` masks of a time-first stacked design."""
    A = np.asarray(M, dtype=np.float64).reshape(T, n_groups, -1)
    tol = 1e-10 * np.maximum(np.abs(A).max(axis=(0, 1)), 1.0)
    period_only = np.all(np.abs(A - A[:, :1]) <= tol, axis=(0, 1))
    time_varying = ~np.all(np.abs(A - A[:1]) <= tol, axis=(0, 1))
    return period_only, time_varying


def effects_design(M, names, n_groups: int, T: int, effects: int, mundlak: bool):
    """A design under group (pooled) and period effects.

    Period effects absorb the columns constant across groups within every
    period; beside group effects (``T − 1`` period effects) the first constant
    column stays as the baseline level.  Group effects absorb nothing.  With
    ``mundlak`` and group effects, the group means of the time-varying columns
    are appended.

    Returns
    -------
    M_new, names_new, keep, mundlak_names, absorbed_slopes
        ``keep`` indexes the raw columns kept (``None`` when all are), and
        ``absorbed_slopes`` names the absorbed non-constant columns.
    """
    M = np.asarray(M, dtype=np.float64)
    names = list(names)
    groups = effects in (1, 3)
    periods = effects in (2, 3) and T > 1
    period_only, time_varying = _panel_column_kinds(M, n_groups, T)
    absorbed = np.zeros(M.shape[1], dtype=bool)
    if periods:
        absorbed = period_only.copy()
        if groups:
            const = np.flatnonzero(
                period_only & np.all(M == M[:1], axis=0) & (M[0] != 0)
            )
            if const.size:
                absorbed[const[0]] = False
    keep = np.flatnonzero(~absorbed)
    slopes = [
        names[j] for j in np.flatnonzero(absorbed) if not np.allclose(M[:, j], M[0, j])
    ]
    M_new, names_new = M[:, keep], [names[j] for j in keep]
    mundlak_names: list[str] = []
    if mundlak and groups:
        tv = [j for j in keep if time_varying[j]]
        if tv:
            means = M.reshape(T, n_groups, -1)[:, :, tv].mean(axis=0)
            M_new = np.hstack([M_new, np.tile(means, (T, 1))])
            mundlak_names = [f"{names[j]}_mean" for j in tv]
            names_new = names_new + mundlak_names
    return M_new, names_new, (keep if absorbed.any() else None), mundlak_names, slopes


def has_level_column(M: np.ndarray) -> bool:
    """Whether a design has a constant non-zero column (an intercept)."""
    M = np.asarray(M)
    return bool(M.shape[1]) and bool(np.any(np.all(M == M[:1], axis=0) & (M[0] != 0)))


def pop_mundlak(kwargs: dict) -> bool:
    return bool(kwargs.pop("mundlak", False))


class CountPanelFEMixin:
    """Fixed-effect bookkeeping, priors and Gibbs driver for count panels.

    The host class builds its pooled base first (``effects=0``, so ``_X`` and
    ``_y`` are the raw design and counts) and then calls
    :meth:`_init_count_fe`.
    """

    _count_spatial: bool = True
    # Count likelihoods carry no Jacobian; the Lee & Yu adjustments are Gaussian.
    _lee_yu: bool = False

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def _init_count_fe(
        self, effects: int, n_groups: int, T: int, mundlak: bool = False
    ) -> None:
        effects = _resolve_count_effects(effects)
        if effects in (1, 3) and T < 2:
            raise ValueError(
                f"Group effects (effects={effects}) need T >= 2; got T={T}."
            )
        if mundlak and effects not in (1, 3):
            raise ValueError("mundlak=True needs group effects (effects=1 or 3).")
        self._count_effects = effects
        self._n_groups = int(n_groups)
        self._T_count = int(T)
        self._mundlak = bool(mundlak)

        names = list(self._feature_names)
        self._design_feature_names = list(names)
        X_new, names_new, keep, mundlak_names, slopes = effects_design(
            self._X, names, n_groups, T, effects, mundlak
        )
        self._X, self._feature_names = X_new, names_new
        self._beta_keep: Optional[np.ndarray] = keep
        self._mundlak_names = mundlak_names
        if slopes:
            warnings.warn(
                f"{slopes} vary only over time and are absorbed by the period "
                "effects; their coefficients are not identified.",
                UserWarning,
                stacklevel=3,
            )
        if self._count_spatial and self._X.shape[1] == 0:
            raise ValueError(
                "No covariate is left after the period effects, so the spatial "
                "parameters are not identified: the count model filters only the "
                "mean."
            )

        self._fe_groups = effects in (1, 3)
        self._fe_periods = effects in (2, 3) and T > 1
        self._row_group = (
            np.tile(np.arange(n_groups, dtype=np.intp), T) if self._fe_groups else None
        )
        # Beside group effects the first period is the baseline (the intercept
        # carries the level); without them every period gets an effect, the
        # period effects having absorbed the intercept.
        self._n_tau = 0
        self._D_tau = None
        if self._fe_periods:
            self._n_tau = T - 1 if self._fe_groups else T
            t0 = T - self._n_tau
            D = np.zeros((n_groups * T, self._n_tau))
            for t in range(t0, T):
                D[t * n_groups : (t + 1) * n_groups, t - t0] = 1.0
            self._D_tau = D

        if self._fe_groups:
            totals = np.bincount(
                self._row_group,
                weights=np.asarray(self._count_y(), dtype=np.float64),
                minlength=n_groups,
            )
            n_zero = int(np.sum(totals == 0))
            if n_zero:
                warnings.warn(
                    f"{n_zero} of {n_groups} groups have zero counts in every "
                    "period; their effects are informed only by the pooled "
                    "effect distribution.",
                    UserWarning,
                    stacklevel=3,
                )

    @property
    def _nonintercept_indices(self) -> list[int]:
        """Non-constant design columns, without the Mundlak group means."""
        skip = set(getattr(self, "_mundlak_names", ()))
        out = []
        for j, name in enumerate(self._feature_names):
            col = self._X[:, j]
            if name in skip or name.lower() == "intercept" or np.allclose(col, col[0]):
                continue
            out.append(j)
        return out

    def _count_y(self) -> np.ndarray:
        return self._y_int_vec

    @property
    def _period_labels(self) -> list[str]:
        return [f"t{t}" for t in range(self._T_count - self._n_tau, self._T_count)]

    def _fe_coords(self) -> dict:
        coords = {}
        if self._fe_periods:
            coords["time"] = self._period_labels
        if self._fe_groups:
            coords["group"] = list(range(self._n_groups))
        return coords

    def _count_fe_priors(self) -> dict:
        """Group and period effect priors on the log scale.

        ``group_effect ~ N(μ, σ²)``, σ learned under a half-t(3,
        ``group_effect_sd_scale``) prior (default scale 1); ``μ`` defaults to 0
        when the design keeps an intercept (it carries the level), else
        ``log mean(y)``.  ``time_effect ~ N(0, 2.5²)`` when it measures a shift
        from the baseline period (group effects present), ``N(log mean(y),
        2.5²)`` when it is each period's level.  Override with
        ``group_effect_mu``, ``group_effect_sd_scale``, ``time_effect_mu`` and
        ``time_effect_sigma``.
        """
        ybar = float(np.mean(self._count_y()))
        return self._effect_priors(float(np.log(max(ybar, 1e-3))), self._X, "")

    def _effect_priors(self, level: float, design, prefix: str) -> dict:
        p = self.priors
        group_mu = 0.0 if has_level_column(design) else level
        return {
            "group_effect_mu": float(p.get(prefix + "group_effect_mu", group_mu)),
            "group_effect_sd_scale": float(
                p.get(prefix + "group_effect_sd_scale", 1.0)
            ),
            "time_effect_mu": float(
                p.get(prefix + "time_effect_mu", 0.0 if self._fe_groups else level)
            ),
            "time_effect_sigma": float(p.get(prefix + "time_effect_sigma", 2.5)),
        }

    @staticmethod
    def _group_spec(row_group, n_groups: int, fp: dict):
        """Pooled group effects: the sd starts at its prior scale and is learned."""
        from ...samplers._utils._group_effects import GroupEffects

        scale = fp["group_effect_sd_scale"]
        return GroupEffects(
            row_group, n_groups, fp["group_effect_mu"], scale, sigma_scale=scale
        )

    @staticmethod
    def _pair_args(fp: dict):
        """``(mu, sd start, sd scale)`` for the structured flow equations."""
        scale = fp["group_effect_sd_scale"]
        return (fp["group_effect_mu"], scale, scale)

    def _count_fe_spec(self):
        from ...samplers.count_panel import CountPanelFE

        groups = None
        if self._fe_groups:
            groups = self._group_spec(
                self._row_group, self._n_groups, self._count_fe_priors()
            )
        return CountPanelFE(groups=groups, D_tau=self._D_tau)

    def _store_group_draws(
        self, store: Optional[bool], chains: int, draws: int
    ) -> bool:
        if not self._fe_groups:
            return False
        if store is not None:
            return bool(store)
        nbytes = 8 * self._n_groups * chains * draws
        if nbytes > _GROUP_DRAWS_MAX_BYTES:
            warnings.warn(
                f"Per-draw group effects would take {nbytes / 1024**2:.0f} MB; "
                "storing their posterior mean and sd only "
                "(pass store_group_effects=True to keep every draw).",
                UserWarning,
                stacklevel=3,
            )
            return False
        return True

    # ------------------------------------------------------------------
    # Gibbs
    # ------------------------------------------------------------------

    def _run_count_panel_gibbs(
        self,
        filt,
        *,
        beta_mu: np.ndarray,
        beta_sigma: np.ndarray,
        alpha_sigma: float = 2.5,
        alpha_nu: float = 3.0,
        alpha_fixed: Optional[float] = None,
        draws: int,
        tune: int,
        chains: int,
        random_seed: Optional[int],
        progressbar: bool,
        n_jobs: int,
        log_likelihood: bool,
        store_group_effects: Optional[bool],
        model_type: str,
        separable_flow: bool = False,
        thin: int = 1,
    ) -> xr.DataTree:
        """Run :func:`~neighbayes.samplers.count_panel.run_chain` over chains."""
        from ...samplers._utils._seeds import spawn_chain_seeds
        from ...samplers.count_panel import CountPanelPriors, run_chain
        from ...samplers.gaussian._chain_runner import run_chains

        fe = self._count_fe_spec()
        fp = self._count_fe_priors()
        n_tau = fe.n_tau
        priors = CountPanelPriors(
            theta_mu=np.concatenate(
                [np.asarray(beta_mu, float), np.full(n_tau, fp["time_effect_mu"])]
            ),
            theta_sigma=np.concatenate(
                [np.asarray(beta_sigma, float), np.full(n_tau, fp["time_effect_sigma"])]
            ),
            alpha_sigma=alpha_sigma,
            alpha_nu=alpha_nu,
            alpha_fixed=alpha_fixed,
        )
        keep_groups = self._store_group_draws(
            store_group_effects, chains, draws // max(thin, 1)
        )
        y = self._count_y().astype(np.float64)
        X = np.asarray(self._X, dtype=np.float64)

        def _chain_fn(chain_id, seed, progress_manager=None, chain_id_kw=0):
            return run_chain(
                y,
                X,
                filt,
                fe,
                priors,
                draws,
                tune,
                thin=thin,
                rng=np.random.default_rng(seed),
                chain_id=chain_id,
                progress_manager=progress_manager,
                store_log_lik=log_likelihood,
                store_group_draws=keep_groups,
            )

        seeds = (
            spawn_chain_seeds(random_seed, chains) if random_seed is not None else None
        )
        results = run_chains(
            chain_fn=_chain_fn,
            n_chains=chains,
            seeds=seeds,
            n_jobs=n_jobs,
            progressbar=progressbar,
            parallel=n_jobs != 1,
            draws=draws,
            tune=tune,
            model_type=model_type,
            timeout=None,
        )
        names = list(filt.names)
        if separable_flow:
            for r in results:
                r["rho_w"] = -r["rho_d"] * r["rho_o"]
            names.append("rho_w")
        return self._assemble_count_idata(results, names, keep_groups, log_likelihood)

    def _assemble_count_idata(
        self,
        results: list[dict],
        rho_names: list[str],
        keep_groups: bool,
        log_likelihood: bool,
        extra: Optional[dict] = None,
        sel_effects: bool = False,
    ) -> xr.DataTree:
        """Stack chain output into a DataTree (shared with the structured kernel).

        ``extra`` maps further vector variables to ``(dim, labels)`` (the
        selection coefficients); scalar extras go in ``rho_names``.  With
        ``sel_effects`` the binary half's period and group effects
        (``sel_time_effect``, ``sel_group_effect``) are stored as well, on the
        count half's layout (the hurdle panels).
        """
        from ...samplers._utils._idata import gibbs_to_inference_data

        stack = lambda key: np.stack([r[key] for r in results], axis=0)  # noqa: E731
        post = {name: stack(name) for name in rho_names}
        post["beta"] = stack("beta")
        dims = {"beta": ["coefficient"]}
        coords = {"coefficient": list(self._feature_names)}
        for name, (dim, labels) in (extra or {}).items():
            post[name] = stack(name)
            dims[name] = [dim]
            coords[dim] = list(labels)
        post["alpha"] = stack("alpha")
        for key in ("group_sd", "sel_group_sd"):
            if key in results[0]:
                post[key] = stack(key)
        prefixes = ("", "sel_") if sel_effects else ("",)
        for pre in prefixes:
            if self._fe_periods:
                post[pre + "time_effect"] = stack(pre + "time_effect")
                dims[pre + "time_effect"] = ["time"]
                coords["time"] = self._period_labels
            if self._fe_groups and keep_groups:
                post[pre + "group_effect"] = stack(pre + "group_effect")
                dims[pre + "group_effect"] = ["group"]
                coords["group"] = list(range(self._n_groups))
        ll = {"obs": stack("log_lik")} if log_likelihood else None
        idata = gibbs_to_inference_data(
            posterior_samples=post,
            log_likelihood=ll,
            observed_data={"obs": self._count_y()},
            coords=coords,
            dims=dims,
        )
        if self._fe_groups and not keep_groups:
            for pre in prefixes:
                # Pool the chains' running moments (equal draw counts per chain).
                means = np.stack([r[pre + "group_effect_mean"] for r in results])
                sds = np.stack([r[pre + "group_effect_sd"] for r in results])
                mean = means.mean(axis=0)
                var = (sds**2).mean(axis=0) + means.var(axis=0)
                idata[pre + "group_effect_summary"] = xr.Dataset(
                    {
                        "mean": (("group",), mean),
                        "sd": (("group",), np.sqrt(var)),
                    },
                    coords={"group": np.arange(self._n_groups)},
                )
        self._idata = idata
        return idata

    # ------------------------------------------------------------------
    # PyMC terms and posterior helpers
    # ------------------------------------------------------------------

    def _pymc_fe_offset(self, prefix: str = "", fe_priors: Optional[dict] = None):
        """Inside a ``pm.Model``: the fixed-effect part of the log-mean, or 0.

        ``prefix`` and ``fe_priors`` (keys as :meth:`_count_fe_priors`) name and
        price a second set of effects (the hurdle's binary half, ``"sel_"``).
        """
        import pytensor.tensor as pt

        from ..._lazy_deps import pm

        fp = self._count_fe_priors() if fe_priors is None else fe_priors
        offset = 0.0
        if self._fe_groups:
            # Pooled effects, non-centred: c = μ + σ·z keeps NUTS off the funnel.
            sd = pm.HalfStudentT(
                prefix + "group_sd", nu=3.0, sigma=fp["group_effect_sd_scale"]
            )
            raw = pm.Normal(prefix + "group_effect_raw", 0.0, 1.0, dims="group")
            c = pm.Deterministic(
                prefix + "group_effect", fp["group_effect_mu"] + sd * raw, dims="group"
            )
            offset = offset + c[self._row_group]
        if self._fe_periods:
            tau = pm.Normal(
                prefix + "time_effect",
                mu=fp["time_effect_mu"],
                sigma=fp["time_effect_sigma"],
                dims="time",
            )
            offset = offset + pt.dot(pt.as_tensor_variable(self._D_tau), tau)
        return offset

    def _fe_offset_draws(self, n_draws: int) -> Optional[np.ndarray]:
        """Per-draw ``D_g c + D_τ τ``, shape ``(n_draws, N·T)``, or ``None``."""
        if not (self._fe_groups or self._fe_periods):
            return None
        post = self._idata.posterior
        out = np.zeros((n_draws, self._n_groups * self._T_count))
        if self._fe_groups:
            if "group_effect" not in post.data_vars:
                raise RuntimeError(
                    "posterior_predictive needs per-draw group effects, which "
                    "this fit stored only as a summary; refit with "
                    "store_group_effects=True."
                )
            c = post["group_effect"].values.reshape(-1, self._n_groups)[:n_draws]
            out += c[:, self._row_group]
        if self._fe_periods:
            tau = post["time_effect"].values.reshape(-1, self._n_tau)[:n_draws]
            out += tau @ self._D_tau.T
        return out

    def _count_draw(self, rng, eta: np.ndarray, alpha: Optional[float]) -> np.ndarray:
        lam = np.exp(np.clip(eta, -50.0, 50.0))
        if alpha is None:
            raise ValueError("alpha is required for NegBin posterior_predictive")
        return rng.negative_binomial(alpha, alpha / (alpha + lam)).astype(np.float64)
