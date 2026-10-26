r"""Latent SEM probit, and the SEM probit with random unit effects.

.. math::

    y_i = \mathbf 1\{y^*_i > 0\}, \qquad
    y^* = X\beta + (I - \lambda W)^{-1}\varepsilon, \qquad
    \varepsilon_i \sim N(0, 1)

The error process carries the probit's own disturbance, so :math:`\lambda = 0`
is the probit, and the GLM suite's ``LM-Error`` is this model's score test.  A
SEM whose error process carried no disturbance would be the aspatial probit,
so the unmarked SEM probit is the latent one (ADR 0003).

With ``unit_effects="random"`` each unit (a row of ``W``) is observed more
than once, and the error process carries a normal random effect per unit in
place of the probit's disturbance (Smith and LeSage 2004; LeSage and Pace
2009, §10.6):

.. math::

    y_j = \mathbf 1\{x_j'\beta + c_{u(j)} + \varepsilon_j > 0\}, \qquad
    c = \lambda W c + v, \quad v \sim N(0, \sigma_c^2 I).
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
import pytensor.tensor as pt
from libpysal.graph import Graph

from ..._backends.sampler_helpers import (
    prepare_compile_kwargs,
    prepare_idata_kwargs,
    select_backend,
)
from ..._lazy_deps import pm, xr
from .._design import _is_panel, _pop_design
from .._mixins._designs import BinaryDesigns, _by_design, _constructor_signature
from ..base import SpatialModel
from ..priors import SEMProbitPriors, SEMProbitREPriors, priors_as_dict, resolve_priors
from ._latent_binary import LatentProbitMixin


class SEMProbit(BinaryDesigns, LatentProbitMixin, SpatialModel):
    r"""Bayesian latent SEM probit, or the SEM probit with random unit effects.

    .. math::

        y_i = \mathbb 1\{y^*_i > 0\}, \qquad
        (I - \lambda W)(y^* - X\beta) = \varepsilon, \qquad
        \varepsilon_i \sim N(0, 1).

    The latent SEM is sampled by an exact Gibbs sweep: the latent utilities
    (Albert & Chib 1993), :math:`\lambda` with :math:`\beta` integrated out,
    then :math:`\beta`, on JAX by default (``backend="numpy"`` for the compiled
    NumPy sweep).  There is no NUTS path, the likelihood being an orthant
    probability.

    Parameters
    ----------
    formula : str, optional
        Wilkinson-style formula, e.g. ``"y ~ x1 + x2"``.  Requires ``data``.
    data : pandas.DataFrame or geopandas.GeoDataFrame, optional
        Data source for formula mode.
    y : array-like, optional
        Binary dependent variable (0/1) (matrix mode).
    X : array-like or pandas.DataFrame, optional
        Design matrix (matrix mode).
    W : libpysal.graph.Graph or scipy.sparse matrix
        Spatial weights over the units.  Without unit effects every
        observation is a unit, ``(n, n)``; with them, ``(m, m)`` over the
        ``m`` units the observations repeat, and it should be row-standardized
        (a :class:`UserWarning` is raised if not).
    unit_effects : {None, "random"}, default None
        ``"random"``: the SEM probit with random unit effects, whose error
        process carries one normal effect per unit,

        .. math::
            y_j = \mathbb 1\{x_j'\beta + c_{u(j)} + \varepsilon_j > 0\},
            \qquad c = \lambda W c + v, \quad v \sim N(0, \sigma_c^2 I),

        fitted by NUTS.  Needs ``unit_col`` (formula mode) or ``units``
        (matrix mode).
    unit_col : str, optional
        The column of ``data`` naming each observation's unit (formula mode).
    units : array-like, optional
        Each observation's unit (matrix mode), in any coding; the units in
        order of first appearance are the rows of ``W``.
    priors : dict, SEMProbitPriors or SEMProbitREPriors, optional
        ``lam_lower``, ``lam_upper``: the Uniform prior on :math:`\lambda`
        (default -0.999, 0.999; -0.95, 0.95 with unit effects); ``beta_mu``,
        ``beta_sigma`` (default Gelman et al. 2008 on the probit scale);
        with unit effects, ``unit_effect_sd_sigma`` (default 2.0), the
        half-normal scale of :math:`\sigma_c`.
    logdet_method, logdet_decomp
        As for :class:`~neighbayes.models.SEM` (the latent SEM only).

    Notes
    -----
    The latent SEM's impacts are :math:`\beta` on the latent index (the SEM
    has no spillovers), or on :math:`P(y_i = 1)` with
    ``scale="probability"``; :meth:`fitted_probabilities` simulates the
    utilities, whose marginal variances the multiplier makes unequal.  With
    unit effects the probability-scale impacts are the average marginal
    effects :math:`\phi(x_j'\beta + c_{u(j)})\beta_k`, again with no indirect
    part, and the model does not sample the heteroskedastic ``v_i``/``r``
    hierarchy of LeSage's ``semip_g``.
    """

    _priors_cls = SEMProbitPriors
    _spatial_params: tuple[str, ...] = ("lam",)
    _jacobian_param: str | None = "lam"
    _model_type: str = "sem_probit"
    _gibbs_key: tuple[str, str] | None = ("binary_latent", "cross_section")
    _latent_process = "sem"
    _DESIGNS = {
        "static": {},
        "random": {"_priors_cls": SEMProbitREPriors, "_gibbs_key": None},
    }

    def __init__(
        self,
        formula: Optional[str] = None,
        data: Optional[pd.DataFrame] = None,
        y=None,
        X=None,
        W=None,
        *,
        unit_effects: Optional[str] = None,
        unit_col: Optional[str] = None,
        units=None,
        **kwargs,
    ):
        design, rest = self._resolve_design(
            dict(kwargs, unit_effects=unit_effects, unit_col=unit_col, units=units)
        )
        if design == "random":
            self._init_unit_effects(formula, data, y, X, W, **rest)
        else:
            super().__init__(
                formula, data, y, X, W, unit_effects=unit_effects, **kwargs
            )

    @classmethod
    def _resolve_design(cls, kw: dict) -> tuple[str, dict]:
        """The latent SEM, or ``"random"``: random unit effects over repeated observations."""
        name = cls.__name__
        unit_col, units = kw.pop("unit_col", None), kw.pop("units", None)
        if kw.get("unit_effects") != "random":
            if unit_col is not None or units is not None:
                raise ValueError(
                    "unit_col and units index random unit effects; pass "
                    "unit_effects='random'."
                )
            return super()._resolve_design(kw)
        _, time, dynamic = _pop_design(kw)
        if time or dynamic or _is_panel(kw) or kw.pop("latent", None):
            raise NotImplementedError(
                f"{name} with random unit effects is a cross-section with repeated "
                "observations of each unit; it takes no time, dynamics or latent=."
            )
        if unit_col is None and units is None:
            raise ValueError(
                "Random unit effects need unit_col (formula mode) or units "
                "(matrix mode)."
            )
        return "random", dict(kw, unit_col=unit_col, units=units)

    # The one cross-section design is the latent SEM.
    _fit_gibbs_static = LatentProbitMixin._fit_gibbs_latent
    _build_pymc_model_static = LatentProbitMixin._build_pymc_model_latent
    _fitted_probabilities_static = LatentProbitMixin._fitted_probabilities_latent
    _fitted_mean_from_posterior_static = (
        LatentProbitMixin._fitted_mean_from_posterior_latent
    )

    fit = _by_design("fit")
    fit.__doc__ = SpatialModel.fit.__doc__
    fit.__wrapped__ = SpatialModel.fit

    def _compute_impacts_posterior(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Direct effects ``beta`` on the latent index; no indirect effects."""
        return self._sem_effects()

    def impacts(self, return_posterior_samples: bool = False, scale: str = "index"):
        r"""Direct, indirect and total impacts on the latent index or probability scale.

        ``scale="probability"`` gives the effects on :math:`P(y_i = 1)`: for
        the latent SEM the derivative carries the density of the propagated
        normal disturbance at the threshold (Monte Carlo per posterior draw,
        conditioning on all but the unit's own disturbance), with unit
        effects the average marginal effect :math:`\phi(x_j'\beta +
        c_{u(j)})\beta_k`.  ``"index"`` gives :math:`\beta` on the latent
        index.  Neither has an indirect part.
        """
        if self._design == "random" and scale == "probability":
            self._require_fit()
            return self._effects_frame(
                *self._unit_effects_ame(),
                return_posterior_samples=return_posterior_samples,
                scale=scale,
            )
        return self._latent_impacts(scale, return_posterior_samples)

    # ------------------------------------------------------------------
    # The SEM probit with random unit effects
    # ------------------------------------------------------------------

    def _init_unit_effects(
        self, formula, data, y, X, W, *, unit_col, units, priors=None, robust=False,
        **unknown,
    ):  # fmt: skip
        """Construct the random-unit-effects design (see the class)."""
        name = type(self).__name__
        if unknown:
            raise TypeError(f"{name} got unexpected arguments {sorted(unknown)}.")
        if robust:
            raise NotImplementedError(f"robust=True is not supported for {name}.")
        if W is None:
            raise ValueError("W is required.")
        self._set_design("random")
        self.priors_obj = resolve_priors(priors, self._priors_cls)
        self.priors = priors_as_dict(self.priors_obj)
        self.robust = False
        self._idata: Optional[xr.DataTree] = None
        self._pymc_model: Optional[pm.Model] = None

        self._W_dense = self._as_dense_unit_W(W)
        self._m = self._W_dense.shape[0]
        # W is over the units, not the observations: the observation-level W
        # machinery (spatial_diagnostics, ...) does not apply, and _require_W
        # raises cleanly.
        self._W_sparse = None

        if formula is not None:
            if data is None:
                raise ValueError("data is required when using formula mode.")
            if unit_col is None:
                raise ValueError("unit_col is required when using formula mode.")
            self._y, self._X, self._feature_names = self._parse_formula(formula, data)
            unit_ids = data[unit_col]
        elif y is not None and X is not None:
            if units is None:
                raise ValueError("units is required in matrix mode.")
            self._y, self._X, self._feature_names = self._parse_matrices(y, X)
            unit_ids = np.asarray(units)
            if unit_ids.shape[0] != self._X.shape[0]:
                raise ValueError("units must have one entry per observation.")
        else:
            raise ValueError(
                "Provide either (formula, data, unit_col) or (y, X, units)."
            )
        codes, uniques = pd.factorize(unit_ids, sort=False)
        self._unit_codes = codes.astype(int)
        self._unit_names = [str(v) for v in uniques.tolist()]

        # Shared parsers keep y as passed; flatten possible (n, 1) columns.
        self._y = np.asarray(self._y, dtype=np.float64).reshape(-1)
        if self._X.shape[0] != self._y.shape[0]:
            raise ValueError("X and y must have the same number of observations.")
        if not np.isin(self._y, [0.0, 1.0]).all():
            raise ValueError("y must be binary with values in {0, 1}.")
        if len(self._unit_names) != self._m:
            raise ValueError(
                f"Number of observed units ({len(self._unit_names)}) must match "
                f"W dimension ({self._m})."
            )

    @staticmethod
    def _as_dense_unit_W(W) -> np.ndarray:
        import scipy.sparse as sp

        from .._base import _check_row_standardization

        if isinstance(W, Graph):
            W_csr = W.sparse.tocsr().astype(float)
        elif sp.issparse(W):
            W_csr = W.tocsr().astype(float)
        elif hasattr(W, "sparse") and hasattr(W, "transform"):
            raise TypeError(
                "W appears to be a legacy libpysal.weights.W object. "
                "Convert it to a libpysal.graph.Graph first: "
                "Graph.from_W(w), or pass w.sparse (the scipy sparse matrix) directly."
            )
        else:
            raise TypeError(
                f"W must be a libpysal.graph.Graph or a scipy sparse matrix, "
                f"got {type(W).__name__}."
            )
        if W_csr.ndim != 2 or W_csr.shape[0] != W_csr.shape[1]:
            raise ValueError("W must be a square unit-level matrix.")
        _check_row_standardization(W_csr, stacklevel=4)
        return W_csr.toarray()

    def _model_coords(self) -> dict[str, list[str]]:
        coords = super()._model_coords()
        if self._design == "random":
            coords["unit"] = list(self._unit_names)
        return coords

    def _build_pymc_model_random(self) -> pm.Model:
        if self._X.shape[1] == 0:
            raise ValueError("X must contain at least one predictor.")
        lam_lower = self.priors.get("lam_lower", -0.95)
        lam_upper = self.priors.get("lam_upper", 0.95)
        beta_mu, beta_sigma = self._resolved_beta_prior(link="probit")
        sd_sigma = self.priors.get("unit_effect_sd_sigma", 2.0)

        W_pt = pt.as_tensor_variable(self._W_dense)
        I_pt = pt.eye(self._m)

        with pm.Model(coords=self._model_coords()) as model:
            lam = pm.Uniform("lam", lower=lam_lower, upper=lam_upper)
            beta = pm.Normal("beta", mu=beta_mu, sigma=beta_sigma, dims="coefficient")
            sd = pm.HalfNormal("unit_effect_sd", sigma=sd_sigma)

            c_raw = pm.Normal("unit_effect_raw", mu=0.0, sigma=1.0, dims="unit")
            c = pm.Deterministic(
                "unit_effect",
                pt.linalg.solve(I_pt - lam * W_pt, sd * c_raw),
                dims="unit",
            )

            eta = pt.dot(self._X, beta) + c[self._unit_codes]
            p = pm.Deterministic("p", 0.5 * (1.0 + pt.erf(eta / np.sqrt(2.0))))
            pm.Bernoulli("obs", p=p, observed=self._y)

        return model

    def _fit_random(
        self,
        draws: int = 2000,
        tune: int = 1000,
        chains: int = 4,
        random_seed: Optional[int] = None,
        progressbar: bool = True,
        sampler: str | None = None,
        backend: str | None = None,
        **sample_kwargs,
    ) -> xr.DataTree:
        """Draw samples from the posterior with NUTS."""
        if sampler not in (None, "nuts"):
            raise NotImplementedError(
                "The SEM probit with random unit effects has no Gibbs sampler; "
                "use sampler='nuts'."
            )
        nuts_sampler = select_backend("nuts", backend, sample_kwargs)
        target_accept = sample_kwargs.pop("target_accept", 0.9)
        model = self._build_pymc_model_random()
        self._pymc_model = model
        if "idata_kwargs" in sample_kwargs:
            sample_kwargs["idata_kwargs"] = prepare_idata_kwargs(
                sample_kwargs["idata_kwargs"], model, nuts_sampler
            )
        sample_kwargs = prepare_compile_kwargs(sample_kwargs, nuts_sampler)
        with model:
            self._idata = pm.sample(
                draws=draws,
                tune=tune,
                chains=chains,
                target_accept=target_accept,
                random_seed=random_seed,
                progressbar=progressbar,
                nuts_sampler=nuts_sampler,
                **sample_kwargs,
            )
        return self._idata

    @staticmethod
    def _rename_summary_index(summary_df: pd.DataFrame) -> pd.DataFrame:
        # The shared version strips beta[...]; also relabel the unit effects
        # unit_effect[...] -> unit_effect:...
        out = SpatialModel._rename_summary_index(summary_df)
        pre = "unit_effect["
        out.index = [
            f"unit_effect:{label[len(pre) : -1]}"
            if label.startswith(pre) and label.endswith("]")
            else label
            for label in out.index.astype(str)
        ]
        return out

    def random_effects_mean(self) -> pd.Series:
        """The posterior mean unit effects (``unit_effects="random"``)."""
        if self._design != "random":
            raise ValueError("random_effects_mean needs unit_effects='random'.")
        self._require_fit()
        c = self._idata.posterior["unit_effect"].mean(("chain", "draw")).to_numpy()
        return pd.Series(c, index=self._unit_names, name="unit_effect_mean")

    def _fitted_probabilities_random(self) -> np.ndarray:
        """The posterior mean of each observation's ``P(y = 1)``."""
        self._require_fit()
        p = self._idata.posterior["p"].mean(("chain", "draw")).to_numpy()
        return np.asarray(p, dtype=float)

    _fitted_mean_from_posterior_random = _fitted_probabilities_random

    def _unit_effects_ame(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        r"""Average marginal effects :math:`\overline{\phi(x_j'\beta + c_{u(j)})}\,\beta_k`."""
        post = self._idata.posterior
        beta = post["beta"].stack(samples=("chain", "draw")).values.T
        c = post["unit_effect"].stack(samples=("chain", "draw")).values.T
        ni = self._nonintercept_indices
        eta = beta @ self._X.T + c[:, self._unit_codes]
        density = np.exp(-0.5 * eta * eta) / np.sqrt(2.0 * np.pi)
        direct = density.mean(axis=1)[:, None] * beta[:, ni]
        return direct, np.zeros_like(direct), direct.copy()


SEMProbit.__signature__ = _constructor_signature(
    BinaryDesigns.__init__, extra=("units",)
)
