"""Spatial probit model with spatially dependent regional effects.

Implements a Bayesian binary-response model analogous to legacy ``semip_g``:

.. math::
    y_{ij} = 1[z_{ij} > 0],\\quad z = X\\beta + \\Delta a + \\varepsilon,

where regional effects follow

.. math::
    a = \\rho W a + u,\\quad u \\sim \\mathcal{N}(0, \\sigma_a^2 I).

The probit link is used directly via ``P(y=1) = Phi(X\\beta + \\Delta a)``.
"""

from __future__ import annotations

from typing import Any, Optional, Union

import numpy as np
import pandas as pd
import pytensor.tensor as pt
from libpysal.graph import Graph

from ..._backends.sampler_helpers import prepare_compile_kwargs, prepare_idata_kwargs
from ..._lazy_deps import pm, xr
from .._base._shared import SharedSpatialMethods
from ..priors import SARProbitPriors, priors_as_dict, resolve_priors


class SARProbit(SharedSpatialMethods):
    """Bayesian spatial probit with regional random effects.

    A binary-response model in which the latent utility includes a
    spatially autoregressive regional random effect. Each observation
    :math:`i` belongs to one of :math:`m` regions; region-level effects
    :math:`a` follow a SAR process on the region-level weights matrix
    :math:`W`, and observation-level disturbances are standard Normal
    (probit link).

    .. math::

        y_i = \\mathbb{1}[z_i > 0],\\quad
        z_i = x_i'\\beta + a_{r(i)} + \\varepsilon_i,\\quad
        \\varepsilon_i \\sim \\mathcal{N}(0, 1),

    .. math::

        a = \\rho W a + u,\\quad u \\sim \\mathcal{N}(0, \\sigma_a^2 I_m),

    so that :math:`a \\sim \\mathcal{N}(0, \\sigma_a^2 (I_m - \\rho W)^{-1}
    (I_m - \\rho W)^{-T})`. The marginal choice probability is
    :math:`P(y_i = 1 \\mid \\beta, a) = \\Phi(x_i'\\beta + a_{r(i)})`.

    Parameters
    ----------
    formula : str, optional
        Formula for the binary response model, e.g. ``"y ~ x1 + x2"``.
        Requires ``data`` and ``region_col``.
    data : pandas.DataFrame, optional
        Data source used with ``formula`` mode.
    y : array-like, optional
        Binary dependent variable (0/1), required in matrix mode.
    X : array-like or pandas.DataFrame, optional
        Covariate matrix, required in matrix mode.
    W : libpysal.graph.Graph or scipy.sparse matrix
        Region-level ``m x m`` spatial weights matrix. Accepts a
        :class:`libpysal.graph.Graph` (the modern libpysal graph API) or any
        :class:`scipy.sparse` matrix.  The legacy :class:`libpysal.weights.W`
        object is **not** accepted directly; pass ``w.sparse`` or convert with
        ``libpysal.graph.Graph.from_W(w)``.
        W should be row-standardized; a :class:`UserWarning` is raised if not.
    region_col : str, optional
        Region identifier column in ``data`` (formula mode).
    region_ids : array-like, optional
        Region identifier per observation (matrix mode).
    mobs : array-like, optional
        Region observation counts ``(m,)`` in sorted region order
        (matrix mode alternative to ``region_ids``).
    priors : dict, optional
        Override default priors. Supported keys:

        - ``rho_lower`` (float, default -0.95): Lower bound of the
          Uniform prior on :math:`\\rho`.
        - ``rho_upper`` (float, default 0.95): Upper bound of the
          Uniform prior on :math:`\\rho`.
        - ``beta_mu`` (float, default 0.0): Normal prior mean for
          :math:`\\beta`.
        - ``beta_sigma`` (float, default 1e6): Normal prior std for
          :math:`\\beta`.
        - ``sigma_a_sigma`` (float, default 10.0): HalfNormal scale
          for the regional random-effect std :math:`\\sigma_a`.

    robust : bool, default False
        Not supported. The probit link uses a Normal CDF; a Student-t
        analogue is not implemented. Setting ``robust=True`` raises.

    Notes
    -----
    This class follows the core ``semip_g`` structure (binary response with
    spatially dependent regional effects). It uses a standard probit link with
    unit observation-level variance and does not currently sample the ``v_i``/
    ``r`` heteroskedastic hierarchy from legacy ``semip_g``.

    **Robust regression**

    ``robust=True`` is not supported for SARProbit. The probit link
    function uses a Normal CDF; a robust version would require a Student-t
    CDF link, which is not yet implemented. Use ``robust=True`` with
    Gaussian models (OLS, SAR, SEM, etc.) instead.
    """

    def __init__(
        self,
        formula: Optional[str] = None,
        data: Optional[pd.DataFrame] = None,
        y: Optional[Union[np.ndarray, pd.Series]] = None,
        X: Optional[Union[np.ndarray, pd.DataFrame]] = None,
        W: Optional[Union[Graph, np.ndarray]] = None,
        region_col: Optional[str] = None,
        region_ids: Optional[Union[np.ndarray, pd.Series]] = None,
        mobs: Optional[Union[np.ndarray, list[int]]] = None,
        priors: Optional[dict] = None,
        robust: bool = False,
    ):
        if W is None:
            raise ValueError("W is required.")

        # Resolve typed priors.
        self.priors_obj = resolve_priors(priors, SARProbitPriors)
        self.priors = priors_as_dict(self.priors_obj)
        self.robust = robust
        self._idata: Optional[xr.DataTree] = None
        self._pymc_model: Optional[pm.Model] = None

        self._W_dense = self._as_dense_region_W(W)
        self._m = self._W_dense.shape[0]
        # The weights matrix is region-level (m x m), not observation-level;
        # inherited observation-level W machinery (spatial_diagnostics, ...)
        # is not applicable and _require_W raises cleanly.
        self._W_sparse = None

        if formula is not None:
            if data is None:
                raise ValueError("data is required when using formula mode.")
            if region_col is None:
                raise ValueError("region_col is required when using formula mode.")
            self._y, self._X, self._feature_names = self._parse_formula(formula, data)
            region_series = data[region_col]
            # pd.factorize returns (codes, uniques)
            codes, uniques = pd.factorize(region_series, sort=False)
            self._region_codes = codes.astype(int)
            self._region_names = [str(v) for v in uniques.tolist()]
        elif y is not None and X is not None:
            self._y, self._X, self._feature_names = self._parse_matrices(y, X)
            self._region_codes, self._region_names = self._parse_regions(
                nobs=self._X.shape[0],
                region_ids=region_ids,
                mobs=mobs,
            )
        else:
            raise ValueError(
                "Provide either (formula, data, region_col) or (y, X, region_ids/mobs)."
            )

        # Shared parsers keep y as passed; flatten possible (n, 1) columns.
        self._y = np.asarray(self._y, dtype=np.float64).reshape(-1)

        if self._X.shape[0] != self._y.shape[0]:
            raise ValueError("X and y must have the same number of observations.")

        if not np.isin(self._y, [0.0, 1.0]).all():
            raise ValueError("y must be binary with values in {0, 1}.")

        if len(np.unique(self._region_codes)) != self._m:
            raise ValueError(
                f"Number of observed regions ({len(np.unique(self._region_codes))}) "
                f"must match W dimension ({self._m})."
            )

    @staticmethod
    def _as_dense_region_W(W: Union[Graph, Any, np.ndarray]) -> np.ndarray:
        import scipy.sparse as sp

        from .._base import _check_row_standardization

        if isinstance(W, Graph):
            W_csr = W.sparse.tocsr().astype(float)
        elif sp.issparse(W):
            W_csr = W.tocsr().astype(float)
        elif hasattr(W, "sparse") and hasattr(W, "transform"):
            raise TypeError(
                "W appears to be a legacy libpysal.weights.W object. "
                "Convert it to a libpysysal.graph.Graph first: "
                "Graph.from_W(w), or pass w.sparse (the scipy sparse matrix) directly."
            )
        else:
            raise TypeError(
                f"W must be a libpysal.graph.Graph or a scipy sparse matrix, "
                f"got {type(W).__name__}."
            )
        if W_csr.ndim != 2 or W_csr.shape[0] != W_csr.shape[1]:
            raise ValueError("W must be a square region-level matrix.")
        _check_row_standardization(W_csr, stacklevel=3)
        return W_csr.toarray()

    @staticmethod
    def _parse_regions(
        nobs: int,
        region_ids: Optional[Union[np.ndarray, pd.Series]],
        mobs: Optional[Union[np.ndarray, list[int]]],
    ) -> tuple[np.ndarray, list[str]]:
        if region_ids is not None:
            ids = np.asarray(region_ids)
            if ids.shape[0] != nobs:
                raise ValueError("region_ids must have one entry per observation.")
            codes, uniques = pd.factorize(ids, sort=False)
            return codes.astype(int), [str(v) for v in uniques.tolist()]

        if mobs is not None:
            counts = np.asarray(mobs, dtype=int).reshape(-1)
            if counts.sum() != nobs:
                raise ValueError("sum(mobs) must equal number of observations.")
            codes = np.repeat(np.arange(len(counts), dtype=int), counts)
            names = [f"region_{i}" for i in range(len(counts))]
            return codes, names

        raise ValueError("Provide either region_ids or mobs in matrix mode.")

    def _model_coords(self) -> dict[str, list[str]]:
        coords = super()._model_coords()
        coords["region"] = list(self._region_names)
        return coords

    def _build_pymc_model(self) -> pm.Model:
        k = self._X.shape[1]
        if k == 0:
            raise ValueError("X must contain at least one predictor.")

        rho_lower = self.priors.get("rho_lower", -0.95)
        rho_upper = self.priors.get("rho_upper", 0.95)
        beta_mu = self.priors.get("beta_mu", 0.0)
        beta_sigma = self.priors.get("beta_sigma", 10.0)
        sigma_a_sigma = self.priors.get("sigma_a_sigma", 2.0)

        if self.robust:
            raise NotImplementedError(
                "Robust (Student-t) error distribution is not supported for "
                "SARProbit. The probit link function uses a Normal CDF; "
                "a robust version would require a t-link (Student-t CDF) which "
                "is not yet implemented. Use robust=True with Gaussian models "
                "(OLS, SAR, SEM, etc.) instead."
            )

        W_pt = pt.as_tensor_variable(self._W_dense)
        I_pt = pt.eye(self._m)

        with pm.Model(coords=self._model_coords()) as model:
            rho = pm.Uniform("rho", lower=rho_lower, upper=rho_upper)
            beta = pm.Normal("beta", mu=beta_mu, sigma=beta_sigma, dims="coefficient")
            sigma_a = pm.HalfNormal("sigma_a", sigma=sigma_a_sigma)

            a_raw = pm.Normal("a_raw", mu=0.0, sigma=1.0, dims="region")
            a = pm.Deterministic(
                "a",
                pt.linalg.solve(I_pt - rho * W_pt, sigma_a * a_raw),
                dims="region",
            )

            eta = pt.dot(self._X, beta) + a[self._region_codes]
            p = pm.Deterministic("p", 0.5 * (1.0 + pt.erf(eta / np.sqrt(2.0))))
            pm.Bernoulli("obs", p=p, observed=self._y)

        return model

    def fit(
        self,
        draws: int = 2000,
        tune: int = 1000,
        chains: int = 4,
        random_seed: Optional[int] = None,
        progressbar: bool = True,
        **sample_kwargs,
    ) -> xr.DataTree:
        """Draw samples from the posterior."""
        nuts_sampler = sample_kwargs.pop("nuts_sampler", "pymc")
        target_accept = sample_kwargs.pop("target_accept", 0.9)
        model = self._build_pymc_model()
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

    @property
    def pymc_model(self) -> Optional[pm.Model]:
        """Return the PyMC model object built for the most recent fit.

        Returns
        -------
        pymc.Model or None
            The model object used by :meth:`fit`, or ``None`` if the instance
            has not been fit yet.
        """
        return self._pymc_model

    @staticmethod
    def _rename_summary_index(summary_df: pd.DataFrame) -> pd.DataFrame:
        # Shared version strips beta[...]; additionally relabel the regional
        # effects a[...] -> a:...
        out = SharedSpatialMethods._rename_summary_index(summary_df)
        out.index = [
            f"a:{label[2:-1]}"
            if label.startswith("a[") and label.endswith("]")
            else label
            for label in out.index.astype(str)
        ]
        return out

    def random_effects_mean(self) -> pd.Series:
        """Return posterior mean regional effects."""
        self._require_fit()
        amean = self._idata.posterior["a"].mean(("chain", "draw")).to_numpy()
        return pd.Series(amean, index=self._region_names, name="a_mean")

    def fitted_probabilities(self) -> np.ndarray:
        """Return posterior mean fitted probabilities for observed data."""
        self._require_fit()
        p = self._idata.posterior["p"].mean(("chain", "draw")).to_numpy()
        return np.asarray(p, dtype=float)

    def spatial_effects(
        self,
        return_posterior_samples: bool = False,
    ) -> Union[pd.DataFrame, tuple[pd.DataFrame, dict[str, np.ndarray]]]:
        r"""Compute average marginal effects (AME) for the spatial probit model.

        For the spatial probit with SAR regional random effects

        .. math::

            z_i = x_i'\beta + a_{r(i)} + \varepsilon_i,
            \quad a = \rho W a + u,

        the marginal effect of covariate :math:`k` on the choice
        probability is

        .. math::

            \frac{\partial P(y_i=1)}{\partial x_{ik}}
            = \phi(x_i'\beta + a_{r(i)}) \, \beta_k,

        where :math:`\phi(\cdot)` is the standard-normal PDF.

        Because the spatial autoregression enters only through the
        unobserved regional effects :math:`a` (not through a spatial
        multiplier on :math:`x`), there is **no indirect effect** of
        :math:`x_j` on :math:`y_i` for :math:`i \neq j`.  The
        ``indirect`` column is therefore zero and ``total`` equals
        ``direct``.

        Parameters
        ----------
        return_posterior_samples : bool, default False
            If ``True``, also return a dict of posterior draws for
            each effect type.

        Returns
        -------
        pandas.DataFrame
            One row per non-intercept covariate with columns
            ``direct``, ``direct_ci_lower``, ``direct_ci_upper``,
            ``direct_pvalue``, ``indirect_*``, ``total_*``.
        dict, optional
            Only returned when ``return_posterior_samples=True``.
            Keys: ``direct_samples``, ``indirect_samples``,
            ``total_samples``.
        """
        import scipy.stats as st

        self._require_fit()
        post = self._idata.posterior
        beta_draws = post["beta"].stack(samples=("chain", "draw")).values.T
        a_draws = post["a"].stack(samples=("chain", "draw")).values.T
        n_draws = beta_draws.shape[0]

        # Identify non-intercept columns
        ni = [
            i
            for i, name in enumerate(self._feature_names)
            if name not in ("Intercept", "1", "intercept")
        ]
        if not ni:
            ni = list(range(len(self._feature_names)))
        names = [self._feature_names[i] for i in ni]

        n_eff = len(ni)

        direct_samples = np.empty((n_draws, n_eff), dtype=np.float64)

        for g in range(n_draws):
            beta_g = beta_draws[g]
            a_g = a_draws[g]
            # Linear predictor for each observation (using region-level a)
            z_g = self._X @ beta_g + a_g[self._region_codes]
            # Standard-normal PDF evaluated at z_g
            pdf_g = st.norm.pdf(z_g)
            # Average marginal effect per covariate
            direct_samples[g] = np.mean(pdf_g[:, None] * self._X[:, ni], axis=0)

        indirect_samples = np.zeros_like(direct_samples)
        total_samples = direct_samples.copy()

        def _summarize(samples: np.ndarray) -> dict[str, np.ndarray]:
            return {
                "mean": np.mean(samples, axis=0),
                "ci_lower": np.percentile(samples, 2.5, axis=0),
                "ci_upper": np.percentile(samples, 97.5, axis=0),
                "pvalue": 2.0
                * np.minimum(
                    np.mean(samples > 0, axis=0),
                    np.mean(samples < 0, axis=0),
                ),
            }

        d = _summarize(direct_samples)
        i = _summarize(indirect_samples)
        t = _summarize(total_samples)

        df = pd.DataFrame(
            {
                "direct": d["mean"],
                "direct_ci_lower": d["ci_lower"],
                "direct_ci_upper": d["ci_upper"],
                "direct_pvalue": d["pvalue"],
                "indirect": i["mean"],
                "indirect_ci_lower": i["ci_lower"],
                "indirect_ci_upper": i["ci_upper"],
                "indirect_pvalue": i["pvalue"],
                "total": t["mean"],
                "total_ci_lower": t["ci_lower"],
                "total_ci_upper": t["ci_upper"],
                "total_pvalue": t["pvalue"],
            },
            index=names,
        )
        df.attrs["model_type"] = self.__class__.__name__
        df.attrs["n_draws"] = n_draws

        if return_posterior_samples:
            samples_dict = {
                "direct_samples": direct_samples,
                "indirect_samples": indirect_samples,
                "total_samples": total_samples,
            }
            return df, samples_dict
        return df
