"""Spatial Tobit model classes.

Implements left-censored (default at 0) Bayesian spatial Tobit variants:

- ``SARTobit``: spatial autoregressive Tobit
- ``SEMTobit``: spatial error Tobit
- ``SDMTobit``: spatial Durbin Tobit

All classes use latent-data augmentation for censored observations.
"""

from __future__ import annotations

import numpy as np
import pytensor.tensor as pt
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from pytensor import sparse as pts

from ..._backends.sampler_helpers import use_jax_likelihood
from ..._lazy_deps import pm, xr
from .._base._shared import _tobit_pointwise_loglik, _write_log_likelihood_to_idata
from .._design import _is_panel, _pop_design
from ..base import SpatialModel
from ..priors import SARTobitPriors, SDMTobitPriors, SEMTobitPriors


def _batched_sar_mean(W_sp, rho_f, rhs, n):
    r"""Batched latent mean ``mu[i] = (I - rho_f[i] W)^{-1} rhs[i]``.

    Host-side reconstruction (one solve per posterior draw) used by the Tobit
    log-likelihood rebuilders.  Prefers sparsax's cached LU — the fixed
    ``(I - rho W)`` sparsity pattern is analyzed once and reused across draws
    (only the values rescale with ``rho``) — and falls back to a per-draw scipy
    sparse solve when sparsax is unavailable.

    Parameters
    ----------
    W_sp : scipy.sparse matrix, shape (n, n)
        Spatial weights.
    rho_f : ndarray, shape (s,)
        Flattened spatial-parameter draws.
    rhs : ndarray, shape (s, n)
        Per-draw right-hand sides (e.g. ``Xβ`` or ``[X, WX]β``).
    n : int
        Number of observations.

    Returns
    -------
    mu : ndarray, shape (s, n)
    """
    import scipy.sparse as _sp

    s = rhs.shape[0]
    mu = np.empty((s, n), dtype=np.float64)

    from ..._jax_dispatch import _sparsax_available

    if _sparsax_available():
        import jax.numpy as jnp

        from ..._jax_dispatch import ensure_x64
        from ...samplers._utils._sparsax_lu import sparsax_lu

        ensure_x64()
        # Fixed COO pattern for (I - ρW): merge the I and W patterns so a single
        # (Ai, Aj) with separate const/W value vectors drives every draw.
        I_coo = _sp.eye(n, format="coo")
        W_coo = W_sp.tocoo()
        rows = np.concatenate([I_coo.row, W_coo.row])
        cols = np.concatenate([I_coo.col, W_coo.col])
        const_coo = _sp.coo_matrix(
            (np.concatenate([np.ones(I_coo.nnz), np.zeros(W_coo.nnz)]), (rows, cols)),
            shape=(n, n),
        )
        const_coo.sum_duplicates()
        w_coo = _sp.coo_matrix(
            (np.concatenate([np.zeros(I_coo.nnz), W_coo.data]), (rows, cols)),
            shape=(n, n),
        )
        w_coo.sum_duplicates()
        Ai = jnp.asarray(const_coo.row, dtype=jnp.int32)
        Aj = jnp.asarray(const_coo.col, dtype=jnp.int32)
        const_vals = jnp.asarray(const_coo.data, dtype=jnp.float64)
        w_vals = jnp.asarray(w_coo.data, dtype=jnp.float64)
        lu_solve = sparsax_lu(Ai, Aj, n).solve
        for i in range(s):
            Ax = const_vals - float(rho_f[i]) * w_vals
            mu[i] = np.asarray(
                lu_solve(Ai, Aj, Ax, jnp.asarray(rhs[i])), dtype=np.float64
            )
        return mu

    from ..._ops._backend import _solve_sparse_vector

    I_sp = _sp.eye(n, format="csc")
    for i in range(s):
        A_csc = (I_sp - rho_f[i] * W_sp).tocsc()
        mu[i] = _solve_sparse_vector(A_csc, rhs[i])
    return mu


class _SpatialTobitBase(SpatialModel):
    """Shared helpers for spatial Tobit models.

    A Tobit panel is pooled: a within transformation would mix censored and
    uncensored values inside the likelihood on the observed scale, so the
    panel takes no unit or time effects, and no dynamics.
    """

    #: Whether the class has a panel (``SDMTobit`` does not).
    _has_panel = True

    def __init__(
        self,
        *args,
        censoring: float = 0.0,
        unit_effects=None,
        time_effects: bool = False,
        dynamic=False,
        **kwargs,
    ):
        kwargs = self._resolve_design(
            dict(
                kwargs,
                unit_effects=unit_effects,
                time_effects=time_effects,
                dynamic=dynamic,
            )
        )[1]
        self.censoring = float(censoring)
        super().__init__(*args, **kwargs)
        self._censored_mask = self._y <= self.censoring
        self._censored_idx = np.where(self._censored_mask)[0]

    @classmethod
    def _resolve_design(cls, kw: dict) -> tuple[str, dict]:
        """``"static"`` — a cross-section or a pooled panel — and the base's arguments."""
        name = cls.__name__
        unit, time, dynamic = _pop_design(kw)
        if unit is not None or time or dynamic:
            raise NotImplementedError(f"{name} takes no effects and no dynamics.")
        if _is_panel(kw) and not cls._has_panel:
            raise NotImplementedError(f"{name} is not implemented with a panel.")
        return "static", kw

    def _tobit_beta_prior(self):
        """Gelman et al. (2008) ``beta`` prior, scaled to the observed ``y``."""
        return self._resolved_beta_prior(
            self._X, list(self._model_coords()["coefficient"])
        )

    def _tobit_sigma(self):
        """``σ² ~ IG(sigma2_alpha, sigma2_beta)`` (default ``IG(2, Var y)``); σ recorded."""
        sigma2 = pm.InverseGamma(
            "sigma2",
            alpha=float(self.priors.get("sigma2_alpha", 2.0)),
            beta=float(self.priors.get("sigma2_beta", np.var(self._y))),
        )
        return pm.Deterministic("sigma", pt.sqrt(sigma2))

    @staticmethod
    def _censored_gap(n_cens: int):
        """Gap ``c − y*`` of each censored latent value, with a flat prior on (0, ∞).

        The regression density on ``y*`` is the only distribution the latent
        values get, so integrating them out leaves the censored likelihood
        ``P(y* ≤ c)`` exactly.  A proper prior on the gap would be a second,
        spurious density on the same data.
        """
        return pm.HalfFlat("y_cens_gap", shape=n_cens)

    def _latent_y_tensor(self) -> pt.TensorVariable:
        """Create latent y* tensor where censored values are sampled."""
        y_lat = pt.as_tensor_variable(self._y.astype(np.float64))
        n_cens = int(self._censored_idx.size)
        if n_cens > 0:
            y_cens_gap = self._censored_gap(n_cens)
            y_cens = self.censoring - y_cens_gap
            y_lat = pt.set_subtensor(y_lat[self._censored_idx], y_cens)
        return y_lat

    def _posterior_latent_y_mean(self) -> np.ndarray:
        """Posterior mean of latent y* on the observed index set."""
        y_lat = self._y.copy().astype(float)
        if self._censored_idx.size > 0 and "y_cens_gap" in self._idata.posterior:
            gap_hat = (
                self._idata.posterior["y_cens_gap"].mean(("chain", "draw")).to_numpy()
            )
            y_lat[self._censored_idx] = self.censoring - np.asarray(
                gap_hat, dtype=float
            )
        return y_lat

    def _compute_impacts_posterior(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Posterior samples of direct, indirect, and total effects.

        Effects are reported on the latent (uncensored) scale, so they share
        the linear SAR/SEM/SDM impact structure of the corresponding Gaussian
        model.  The diagonal traces ride the resolvent identities
        ``tr(S)/n = 1 - (rho/n)*g`` and ``tr(SW)/n = -g/n`` and need no
        eigendecomposition.
        """
        from ...diagnostics.lmtests import _get_posterior_draws

        idata = self.inference_data

        if isinstance(self, SARTobit):
            rho_draws = _get_posterior_draws(idata, "rho")
            beta_draws = _get_posterior_draws(idata, "beta")
            mean_diag = self._batch_mean_diag(rho_draws)
            mean_row_sum = self._batch_mean_row_sum(rho_draws)
            ni = self._nonintercept_indices
            direct_samples = mean_diag[:, None] * beta_draws[:, ni]
            total_samples = mean_row_sum[:, None] * beta_draws[:, ni]
            indirect_samples = total_samples - direct_samples

        elif isinstance(self, SEMTobit):
            beta_draws = _get_posterior_draws(idata, "beta")
            ni = self._nonintercept_indices
            direct_samples = beta_draws[:, ni].copy()
            indirect_samples = np.zeros_like(direct_samples)
            total_samples = direct_samples.copy()

        elif isinstance(self, SDMTobit):
            rho_draws = _get_posterior_draws(idata, "rho")
            beta_draws = _get_posterior_draws(idata, "beta")
            k = self._X.shape[1]
            kw = self._WX.shape[1]
            beta1_draws = beta_draws[:, :k]
            beta2_draws = beta_draws[:, k : k + kw]
            mean_diag_M = self._batch_mean_diag(rho_draws)
            mean_diag_MW = self._batch_mean_diag_MW(rho_draws)
            mean_row_sum_M = self._batch_mean_row_sum(rho_draws)
            mean_row_sum_MW = self._batch_mean_row_sum_MW(rho_draws)
            wx_idx = self._wx_column_indices
            direct_samples = (
                mean_diag_M[:, None] * beta1_draws[:, wx_idx]
                + mean_diag_MW[:, None] * beta2_draws
            )
            total_samples = (
                mean_row_sum_M[:, None] * beta1_draws[:, wx_idx]
                + mean_row_sum_MW[:, None] * beta2_draws
            )
            indirect_samples = total_samples - direct_samples

        else:
            raise NotImplementedError(
                f"Impacts not implemented for {type(self).__name__}."
            )

        return direct_samples, indirect_samples, total_samples


class SARTobit(_SpatialTobitBase):
    """Bayesian spatial autoregressive Tobit model.

    .. math::
        y^* = \\rho W y^* + X\\beta + \\varepsilon,\\quad \\varepsilon \\sim N(0,\\sigma^2 I),

    with observed outcome

    .. math::
        y = \\max(c, y^*)

    where ``c`` is the left-censoring point (default ``0``). Censored
    observations contribute their CDF to the likelihood; uncensored
    observations contribute the density of :math:`y^*` evaluated at
    :math:`y`.

    Parameters
    ----------
    formula : str, optional
        Wilkinson-style formula, e.g. ``"y ~ x1 + x2"``. Requires
        ``data``.
    data : pandas.DataFrame or geopandas.GeoDataFrame, optional
        Data source for formula mode.
    y : array-like, optional
        Observed (censored) response of shape ``(n,)``. Required in
        matrix mode.
    X : array-like or pandas.DataFrame, optional
        Design matrix. Required in matrix mode.
    W : libpysal.graph.Graph or scipy.sparse matrix
        Spatial weights of shape ``(n, n)``; see :class:`SAR` for
        accepted formats.
    censoring : float, default 0.0
        Left-censoring threshold ``c``. Observations with
        ``y <= censoring`` are treated as censored and the latent
        :math:`y^*` is sampled.
    priors : dict, optional
        Override default priors. Supported keys:

        - ``rho_lower`` (float, default -1.0): Lower bound of Uniform
          prior on :math:`\\rho`.
        - ``rho_upper`` (float, default 1.0): Upper bound of Uniform
          prior on :math:`\\rho`.
        - ``beta_mu``, ``beta_sigma`` (array, default Gelman 2008): Normal
          prior on :math:`\\beta`, scaled to ``sd(y)`` and each column's sd.
        - ``sigma2_alpha``, ``sigma2_beta`` (float, default 2 and ``Var(y)``):
          InverseGamma prior on :math:`\\sigma^2`.
        - ``nu`` (float, default 4.0): Fixed Student-t degrees of
          freedom (only used when ``robust=True``).

    logdet_method : {"exact", "cheb", "aaa", "spline", "stochastic"}, optional
        How ``log|I - rho*W|`` is approximated over rho: evaluated exactly at
        every rho, interpolated (Chebyshev, AAA rational, or cubic spline)
        through exact values, or expanded stochastically with no
        factorization.  ``None`` (default) selects from the size and symmetry
        of ``W``: ``"exact"`` for ``n <= 500``; ``"aaa"`` up to
        ``n = 60000`` (``NEIGHBAYES_LOGDET_FACTOR_MAX_N``) unless the
        factorization's fill-in would dominate; ``"stochastic"`` beyond.
    logdet_decomp : {"auto", "eig", "chol", "lu"}, default "auto"
        What supplies the exact values: a dense eigendecomposition, a sparse
        Cholesky (undirected ``W``; scikit-sparse CHOLMOD) or a sparse LU.
        ``"auto"`` takes the first of ``"eig"`` (``n <= 500``,
        ``NEIGHBAYES_LOGDET_EIGEN_MAX_N``), ``"chol"`` (undirected ``W``) and
        ``"lu"`` that the method is implemented on.  Ignored by
        ``"stochastic"``.  A pair with no implementation (``"spline"`` on
        ``"chol"``, ``"exact"`` on ``"lu"``) raises.
    robust : bool, default False
        If True, replace the Normal innovation with Student-t. See
        *Robust regression* below.

    time_col, unit_col : str, optional
        Formula-mode panel: the period and unit columns of ``data``.  Given
        time, the model is a pooled panel, stacked by period, then unit.  A
        Tobit panel takes no unit or time effects: a within transformation
        would mix censored and uncensored values in the likelihood.
    N, T : int, optional
        Matrix-mode panel: the numbers of units and periods; ``T=1`` is the
        cross-section.

    Notes
    -----
    **Robust regression**

    When ``robust=True``, the error distribution is changed from Normal
    to Student-t. For uncensored observations the density becomes:

    .. math::

        f(y^*_i \\mid \\mu_i, \\sigma, \\nu) =
        \\frac{1}{\\sigma} \\, t_\\nu\\!\\left(\\frac{y^*_i - \\mu_i}{\\sigma}\\right)

    and for censored observations the probability becomes:

    .. math::

        P(y^*_i \\le c) = T_\\nu\\!\\left(\\frac{c - \\mu_i}{\\sigma}\\right)

    where :math:`T_\\nu` is the Student-t CDF with :math:`\\nu` degrees of
    freedom, and :math:`\\nu` is a **fixed** hyperparameter set by
    ``priors={"nu": value}`` (default 4, LeSage's ``rval``).
    """

    _priors_cls = SARTobitPriors

    def _build_pymc_model(self) -> pm.Model:
        rho_lower = self.priors.get("rho_lower", -1.0)
        rho_upper = self.priors.get("rho_upper", 1.0)
        beta_mu, beta_sigma = self._tobit_beta_prior()
        logdet_fn = self._logdet_pytensor_fn
        W_pt = self._W_pt_sparse
        with pm.Model(coords=self._model_coords()) as model:
            rho = pm.Uniform("rho", lower=rho_lower, upper=rho_upper)
            beta = pm.Normal("beta", mu=beta_mu, sigma=beta_sigma, dims="coefficient")
            sigma = self._tobit_sigma()
            y_lat = self._latent_y_tensor()
            resid = (
                y_lat
                - rho * pts.structured_dot(W_pt, y_lat[:, None]).flatten()
                - pt.dot(self._X, beta)
            )
            if self.robust:
                nu = self._nu
                logp_resid = pm.logp(
                    pm.StudentT.dist(nu=nu, mu=0.0, sigma=sigma), resid
                ).sum()
            else:
                logp_resid = pm.logp(pm.Normal.dist(mu=0.0, sigma=sigma), resid).sum()
            pm.Potential("resid_loglik", logp_resid)
            pm.Potential("jacobian", logdet_fn(rho))
        return model

    def _fitted_mean_from_posterior(self) -> np.ndarray:
        """Reported fitted mean: ``max(c, E[y* | X, params])``.

        The latent mean for SARTobit is
        :math:`E[y^* \\mid X] = (I - \\rho W)^{-1} X \\beta`, evaluated at the
        posterior mean of :math:`(\\rho, \\beta)`. Censored observations are
        reported at the censoring point ``c`` (consistent with the
        observation rule ``y = max(c, y*)``).
        """
        rho = float(self._posterior_mean("rho"))
        beta = self._posterior_mean("beta")
        if self._is_panel:
            # A panel reports the mean given the latent outcome: ρWy* + Xβ.
            y_lat = self._posterior_latent_y_mean()
            return rho * self._sparse_panel_lag(y_lat) + self._X @ beta
        n = self._y.shape[0]
        A = sp.eye(n, format="csc") - rho * sp.csc_matrix(self._W_sparse)
        mean = spla.spsolve(A, self._X @ beta)
        return np.maximum(self.censoring, mean)

    def _postprocess_idata(self, idata: xr.DataTree) -> xr.DataTree:
        """Attach the complete pointwise Tobit log-likelihood for IC metrics.

        The SAR Tobit model expresses both the residual log-likelihood and the
        Jacobian via ``pm.Potential``, so PyMC captures no pointwise
        ``log_likelihood``.  We rebuild it here from the Tobit censoring
        formula (Normal or Student-t density for uncensored observations, the
        matching left-tail log-CDF for censored ones) with the latent
        mean :math:`\\mu = (I - \\rho W)^{-1} X\\beta`; a panel's uses the
        mean given the latent outcome, :math:`\\rho W y^* + X\\beta`, and its
        Jacobian applies once per period.
        """
        if "log_likelihood" in idata.children and "obs" in idata.log_likelihood:
            return idata
        if self._is_panel:
            return self._postprocess_idata_panel(idata)
        rho = idata.posterior["rho"].values
        beta = idata.posterior["beta"].values
        sigma = idata.posterior["sigma"].values
        c, d = rho.shape
        s = c * d
        n = self._y.shape[0]
        rho_f = rho.reshape(s)
        beta_f = beta.reshape(s, beta.shape[-1])
        sigma_f = sigma.reshape(s)
        W_sp = self._W_sparse
        n = self._y.shape[0]
        Xb = beta_f @ self._X.T
        mu = _batched_sar_mean(W_sp, rho_f, Xb, n)
        nu_f = np.full(s, self._nu) if self.robust else None
        ll = _tobit_pointwise_loglik(
            self._y, mu, sigma_f, self._censored_mask, self.censoring, nu_f
        )
        jac = self._logdet_numpy_vec_fn(rho_f)
        ll = (ll + jac[:, None] / n).reshape(c, d, n)
        _write_log_likelihood_to_idata(idata, ll)
        return idata

    def _postprocess_idata_panel(self, idata: xr.DataTree) -> xr.DataTree:
        """The panel log-likelihood given the latent outcome (see above)."""
        rho = idata.posterior["rho"].values
        beta = idata.posterior["beta"].values
        sigma = idata.posterior["sigma"].values
        c, d = rho.shape
        s = c * d
        n = self._y.shape[0]
        rho_f = rho.reshape(s)
        beta_f = beta.reshape(s, beta.shape[-1])
        sigma_f = sigma.reshape(s)
        y_lat = self._posterior_latent_y_mean()
        Wy_lat = self._sparse_panel_lag(y_lat)
        mu = rho_f[:, None] * Wy_lat[None, :] + beta_f @ self._X.T
        nu_f = np.full(s, self._nu) if self.robust else None
        ll = _tobit_pointwise_loglik(
            self._y, mu, sigma_f, self._censored_mask, self.censoring, nu_f
        )
        # ``_logdet_numpy_vec_fn`` is built with ``T=self._T`` and already
        # returns T·log|I_N - ρW_N|; scaling by T again squared the Jacobian.
        jac = self._logdet_numpy_vec_fn(rho_f)
        ll = (ll + jac[:, None] / n).reshape(c, d, n)
        _write_log_likelihood_to_idata(idata, ll)
        return idata


class SEMTobit(_SpatialTobitBase):
    """Bayesian spatial error Tobit model.

    .. math::
        y^* = X\\beta + u,\\quad u = \\lambda W u + \\varepsilon,
        \\quad \\varepsilon \\sim N(0,\\sigma^2 I)

    with observed outcome ``y = max(c, y*)``. Censored observations
    contribute their CDF; uncensored observations contribute the
    spatially-filtered density of :math:`y^*`.

    Parameters
    ----------
    formula : str, optional
        Wilkinson-style formula. Requires ``data``.
    data : pandas.DataFrame or geopandas.GeoDataFrame, optional
        Data source for formula mode.
    y : array-like, optional
        Observed (censored) response. Required in matrix mode.
    X : array-like or pandas.DataFrame, optional
        Design matrix. Required in matrix mode.
    W : libpysal.graph.Graph or scipy.sparse matrix
        Spatial weights of shape ``(n, n)``; see :class:`SAR` for
        accepted formats.
    censoring : float, default 0.0
        Left-censoring threshold ``c``.
    priors : dict, optional
        Override default priors. Supported keys:

        - ``lam_lower`` (float, default -1.0): Lower bound of Uniform
          prior on :math:`\\lambda`.
        - ``lam_upper`` (float, default 1.0): Upper bound of Uniform
          prior on :math:`\\lambda`.
        - ``beta_mu``, ``beta_sigma`` (array, default Gelman 2008): Normal
          prior on :math:`\\beta`, scaled to ``sd(y)`` and each column's sd.
        - ``sigma2_alpha``, ``sigma2_beta`` (float, default 2 and ``Var(y)``):
          InverseGamma prior on :math:`\\sigma^2`.
        - ``nu`` (float, default 4.0): Fixed Student-t degrees of
          freedom (only used when ``robust=True``).

    logdet_method : {"exact", "cheb", "aaa", "spline", "stochastic"}, optional
        How ``log|I - rho*W|`` is approximated over rho: evaluated exactly at
        every rho, interpolated (Chebyshev, AAA rational, or cubic spline)
        through exact values, or expanded stochastically with no
        factorization.  ``None`` (default) selects from the size and symmetry
        of ``W``: ``"exact"`` for ``n <= 500``; ``"aaa"`` up to
        ``n = 60000`` (``NEIGHBAYES_LOGDET_FACTOR_MAX_N``) unless the
        factorization's fill-in would dominate; ``"stochastic"`` beyond.
    logdet_decomp : {"auto", "eig", "chol", "lu"}, default "auto"
        What supplies the exact values: a dense eigendecomposition, a sparse
        Cholesky (undirected ``W``; scikit-sparse CHOLMOD) or a sparse LU.
        ``"auto"`` takes the first of ``"eig"`` (``n <= 500``,
        ``NEIGHBAYES_LOGDET_EIGEN_MAX_N``), ``"chol"`` (undirected ``W``) and
        ``"lu"`` that the method is implemented on.  Ignored by
        ``"stochastic"``.  A pair with no implementation (``"spline"`` on
        ``"chol"``, ``"exact"`` on ``"lu"``) raises.
    robust : bool, default False
        If True, replace the Normal innovation with Student-t.

    time_col, unit_col : str, optional
        Formula-mode panel: the period and unit columns of ``data``.  Given
        time, the model is a pooled panel, stacked by period, then unit.  A
        Tobit panel takes no unit or time effects: a within transformation
        would mix censored and uncensored values in the likelihood.
    N, T : int, optional
        Matrix-mode panel: the numbers of units and periods; ``T=1`` is the
        cross-section.

    Notes
    -----
    **Robust regression**

    When ``robust=True``, the spatially-filtered error distribution is
    changed from Normal to Student-t.  For uncensored observations:

    .. math::

        f(y^*_i \\mid \\mu_i, \\sigma, \\nu) =
        \\frac{1}{\\sigma} \\, t_\\nu\\!\\left(\\frac{y^*_i - \\mu_i}{\\sigma}\\right)

    and for censored observations:

    .. math::

        P(y^*_i \\le c) = T_\\nu\\!\\left(\\frac{c - \\mu_i}{\\sigma}\\right)

    where :math:`T_\\nu` is the Student-t CDF and :math:`\\nu` is a **fixed**
    hyperparameter set by ``priors={"nu": value}`` (default 4).
    """

    _priors_cls = SEMTobitPriors

    def _build_pymc_model(self, nuts_sampler: str = "pymc") -> pm.Model:
        lam_lower = self.priors.get("lam_lower", -1.0)
        lam_upper = self.priors.get("lam_upper", 1.0)
        beta_mu, beta_sigma = self._tobit_beta_prior()
        logdet_fn = self._logdet_pytensor_fn
        W_pt = self._W_pt_sparse
        n_obs = int(self._y.shape[0])
        inv_n = 1.0 / n_obs
        jax_logp = use_jax_likelihood(nuts_sampler)
        n_cens = int(self._censored_idx.size)
        with pm.Model(coords=self._model_coords()) as model:
            lam = pm.Uniform("lam", lower=lam_lower, upper=lam_upper)
            beta = pm.Normal("beta", mu=beta_mu, sigma=beta_sigma, dims="coefficient")
            sigma = self._tobit_sigma()
            if jax_logp:
                if n_cens > 0:
                    y_cens_gap = self._censored_gap(n_cens)
                else:
                    y_cens_gap = None
                X_const = pt.as_tensor_variable(self._X)
                cens_idx_const = (
                    pt.as_tensor_variable(self._censored_idx) if n_cens > 0 else None
                )
                censoring_const = self.censoring

                def _eps(value, lam_, beta_, gap_):
                    y_lat = value
                    if gap_ is not None:
                        y_cens = censoring_const - gap_
                        y_lat = pt.set_subtensor(y_lat[cens_idx_const], y_cens)
                    resid = y_lat - pt.dot(X_const, beta_)
                    return (
                        resid
                        - lam_ * pts.structured_dot(W_pt, resid[:, None]).flatten()
                    )

                if self.robust:
                    nu = self._nu
                    if n_cens > 0:

                        def sempanel_tobit_logp(value, lam_, beta_, sigma_, gap_, nu_):
                            eps = _eps(value, lam_, beta_, gap_)
                            log_dens = pm.logp(
                                pm.StudentT.dist(nu=nu_, mu=0.0, sigma=sigma_), eps
                            )
                            return log_dens + logdet_fn(lam_) * inv_n

                        pm.CustomDist(
                            "obs",
                            lam,
                            beta,
                            sigma,
                            y_cens_gap,
                            nu,
                            logp=sempanel_tobit_logp,
                            observed=self._y,
                        )
                    else:

                        def sempanel_tobit_logp(value, lam_, beta_, sigma_, nu_):
                            eps = _eps(value, lam_, beta_, None)
                            log_dens = pm.logp(
                                pm.StudentT.dist(nu=nu_, mu=0.0, sigma=sigma_), eps
                            )
                            return log_dens + logdet_fn(lam_) * inv_n

                        pm.CustomDist(
                            "obs",
                            lam,
                            beta,
                            sigma,
                            nu,
                            logp=sempanel_tobit_logp,
                            observed=self._y,
                        )
                elif n_cens > 0:

                    def sempanel_tobit_logp(value, lam_, beta_, sigma_, gap_):
                        eps = _eps(value, lam_, beta_, gap_)
                        log_dens = pm.logp(pm.Normal.dist(mu=0.0, sigma=sigma_), eps)
                        return log_dens + logdet_fn(lam_) * inv_n

                    pm.CustomDist(
                        "obs",
                        lam,
                        beta,
                        sigma,
                        y_cens_gap,
                        logp=sempanel_tobit_logp,
                        observed=self._y,
                    )
                else:

                    def sempanel_tobit_logp(value, lam_, beta_, sigma_):
                        eps = _eps(value, lam_, beta_, None)
                        log_dens = pm.logp(pm.Normal.dist(mu=0.0, sigma=sigma_), eps)
                        return log_dens + logdet_fn(lam_) * inv_n

                    pm.CustomDist(
                        "obs",
                        lam,
                        beta,
                        sigma,
                        logp=sempanel_tobit_logp,
                        observed=self._y,
                    )
            else:
                y_lat = self._latent_y_tensor()
                resid = y_lat - pt.dot(self._X, beta)
                eps = resid - lam * pts.structured_dot(W_pt, resid[:, None]).flatten()
                if self.robust:
                    nu = self._nu
                    logp_eps = pm.logp(
                        pm.StudentT.dist(nu=nu, mu=0.0, sigma=sigma), eps
                    ).sum()
                else:
                    logp_eps = pm.logp(pm.Normal.dist(mu=0.0, sigma=sigma), eps).sum()
                pm.Potential("eps_loglik", logp_eps)
                pm.Potential("jacobian", logdet_fn(lam))
        return model

    def _fitted_mean_from_posterior(self) -> np.ndarray:
        """Reported fitted mean: ``max(c, E[y* | X, params])``.

        For SEMTobit the latent mean is
        :math:`E[y^* \\mid X] = X\\beta` (the spatial filter operates on the
        error term and integrates out). Censored entries are reported at
        the censoring point; a panel reports :math:`X\\beta` unclipped.
        """
        beta = self._posterior_mean("beta")
        if self._is_panel:
            return self._X @ beta
        return np.maximum(self.censoring, self._X @ beta)

    def _postprocess_idata(self, idata: xr.DataTree) -> xr.DataTree:
        """Attach the complete pointwise Tobit log-likelihood for IC metrics.

        The SEM Tobit model expresses both the error log-likelihood and the
        Jacobian via ``pm.Potential``, so PyMC captures no pointwise
        ``log_likelihood`` (on JAX backends the ``obs`` CustomDist already
        carries it, and we return early).  We rebuild it here from the Tobit
        censoring formula with latent mean :math:`\\mu = X\\beta` (the
        spatial filter operates on the error term and is absorbed into the
        Jacobian, once per period in a panel).
        """
        if "log_likelihood" in idata.children and "obs" in idata.log_likelihood:
            return idata
        lam = idata.posterior["lam"].values
        beta = idata.posterior["beta"].values
        sigma = idata.posterior["sigma"].values
        c, d = lam.shape
        s = c * d
        n = self._y.shape[0]
        lam_f = lam.reshape(s)
        beta_f = beta.reshape(s, beta.shape[-1])
        sigma_f = sigma.reshape(s)
        mu = beta_f @ self._X.T
        nu_f = np.full(s, self._nu) if self.robust else None
        ll = _tobit_pointwise_loglik(
            self._y, mu, sigma_f, self._censored_mask, self.censoring, nu_f
        )
        jac = self._logdet_numpy_vec_fn(lam_f)
        ll = (ll + jac[:, None] / n).reshape(c, d, n)
        _write_log_likelihood_to_idata(idata, ll)
        return idata


class SDMTobit(_SpatialTobitBase):
    """Bayesian spatial Durbin Tobit model.

    .. math::
        y^* = \\rho Wy^* + X\\beta + WX\\theta + \\varepsilon,
        \\quad \\varepsilon \\sim N(0,\\sigma^2 I)

    with observed outcome ``y = max(c, y*)``. The sampled coefficient
    vector stacks the local and lagged-regressor blocks as
    :math:`[\\beta, \\theta]`.

    Parameters
    ----------
    formula : str, optional
        Wilkinson-style formula. Requires ``data``.
    data : pandas.DataFrame or geopandas.GeoDataFrame, optional
        Data source for formula mode.
    y : array-like, optional
        Observed (censored) response. Required in matrix mode.
    X : array-like or pandas.DataFrame, optional
        Design matrix. Required in matrix mode.
    W : libpysal.graph.Graph or scipy.sparse matrix
        Spatial weights of shape ``(n, n)``.
    censoring : float, default 0.0
        Left-censoring threshold ``c``.
    priors : dict, optional
        Override default priors. Supported keys:

        - ``rho_lower`` (float, default -1.0): Lower bound of Uniform
          prior on :math:`\\rho`.
        - ``rho_upper`` (float, default 1.0): Upper bound of Uniform
          prior on :math:`\\rho`.
        - ``beta_mu``, ``beta_sigma`` (array, default Gelman 2008): Normal
          prior on :math:`[\\beta, \\theta]`, scaled to ``sd(y)`` and each column's sd.
        - ``sigma2_alpha``, ``sigma2_beta`` (float, default 2 and ``Var(y)``):
          InverseGamma prior on :math:`\\sigma^2`.
        - ``nu`` (float, default 4.0): Fixed Student-t degrees of
          freedom (only used when ``robust=True``).

    logdet_method : {"exact", "cheb", "aaa", "spline", "stochastic"}, optional
        How ``log|I - rho*W|`` is approximated over rho: evaluated exactly at
        every rho, interpolated (Chebyshev, AAA rational, or cubic spline)
        through exact values, or expanded stochastically with no
        factorization.  ``None`` (default) selects from the size and symmetry
        of ``W``: ``"exact"`` for ``n <= 500``; ``"aaa"`` up to
        ``n = 60000`` (``NEIGHBAYES_LOGDET_FACTOR_MAX_N``) unless the
        factorization's fill-in would dominate; ``"stochastic"`` beyond.
    logdet_decomp : {"auto", "eig", "chol", "lu"}, default "auto"
        What supplies the exact values: a dense eigendecomposition, a sparse
        Cholesky (undirected ``W``; scikit-sparse CHOLMOD) or a sparse LU.
        ``"auto"`` takes the first of ``"eig"`` (``n <= 500``,
        ``NEIGHBAYES_LOGDET_EIGEN_MAX_N``), ``"chol"`` (undirected ``W``) and
        ``"lu"`` that the method is implemented on.  Ignored by
        ``"stochastic"``.  A pair with no implementation (``"spline"`` on
        ``"chol"``, ``"exact"`` on ``"lu"``) raises.
    robust : bool, default False
        If True, replace the Normal innovation with Student-t.
    durbin_vars : list of str, optional
        Names of X columns to spatially lag. By default all
        non-constant columns are lagged.

    Notes
    -----
    **Robust regression**

    When ``robust=True``, the error distribution is changed from Normal
    to Student-t.  For uncensored observations the density becomes:

    .. math::

        f(y^*_i \\mid \\mu_i, \\sigma, \\nu) =
        \\frac{1}{\\sigma} \\, t_\\nu\\!\\left(\\frac{y^*_i - \\mu_i}{\\sigma}\\right)

    and for censored observations:

    .. math::

        P(y^*_i \\le c) = T_\\nu\\!\\left(\\frac{c - \\mu_i}{\\sigma}\\right)

    where :math:`T_\\nu` is the Student-t CDF and :math:`\\nu` is a **fixed**
    hyperparameter set by ``priors={"nu": value}`` (default 4).
    """

    _priors_cls = SDMTobitPriors
    _has_panel = False
    _has_wx_in_beta: bool = True

    def _build_pymc_model(self) -> pm.Model:
        Z = np.hstack([self._X, self._WX])
        rho_lower = self.priors.get("rho_lower", -1.0)
        rho_upper = self.priors.get("rho_upper", 1.0)
        beta_mu, beta_sigma = self._resolved_beta_prior(
            Z, list(self._model_coords()["coefficient"])
        )
        logdet_fn = self._logdet_pytensor_fn
        W_pt = self._W_pt_sparse
        with pm.Model(coords=self._model_coords()) as model:
            rho = pm.Uniform("rho", lower=rho_lower, upper=rho_upper)
            beta = pm.Normal("beta", mu=beta_mu, sigma=beta_sigma, dims="coefficient")
            sigma = self._tobit_sigma()
            y_lat = self._latent_y_tensor()
            resid = (
                y_lat
                - rho * pts.structured_dot(W_pt, y_lat[:, None]).flatten()
                - pt.dot(Z, beta)
            )
            if self.robust:
                nu = self._nu
                logp_resid = pm.logp(
                    pm.StudentT.dist(nu=nu, mu=0.0, sigma=sigma), resid
                ).sum()
            else:
                logp_resid = pm.logp(pm.Normal.dist(mu=0.0, sigma=sigma), resid).sum()
            pm.Potential("resid_loglik", logp_resid)
            pm.Potential("jacobian", logdet_fn(rho))
        return model

    def _fitted_mean_from_posterior(self) -> np.ndarray:
        """Reported fitted mean: ``max(c, E[y* | X, params])``.

        The latent mean for SDMTobit is
        :math:`E[y^* \\mid X] = (I - \\rho W)^{-1} (X\\beta + WX\\theta)`,
        evaluated at posterior means; censored entries are reported at
        the censoring point.
        """
        rho = float(self._posterior_mean("rho"))
        beta = self._posterior_mean("beta")
        Z = np.hstack([self._X, self._WX])
        n = self._y.shape[0]
        A = sp.eye(n, format="csc") - rho * sp.csc_matrix(self._W_sparse)
        mean = spla.spsolve(A, Z @ beta)
        return np.maximum(self.censoring, mean)

    def _postprocess_idata(self, idata: xr.DataTree) -> xr.DataTree:
        """Attach the complete pointwise Tobit log-likelihood for IC metrics.

        The SDM Tobit model expresses both the residual log-likelihood and the
        Jacobian via ``pm.Potential``, so PyMC captures no pointwise
        ``log_likelihood``.  We rebuild it here from the Tobit censoring
        formula with latent mean
        :math:`\\mu = (I - \\rho W)^{-1} (X\\beta + WX\\theta)`, where the
        stacked design ``Z = [X, WX]`` carries the Durbin lags.
        """
        if "log_likelihood" in idata.children and "obs" in idata.log_likelihood:
            return idata
        rho = idata.posterior["rho"].values
        beta = idata.posterior["beta"].values
        sigma = idata.posterior["sigma"].values
        c, d = rho.shape
        s = c * d
        n = self._y.shape[0]
        Z = np.hstack([self._X, self._WX])
        rho_f = rho.reshape(s)
        beta_f = beta.reshape(s, beta.shape[-1])
        sigma_f = sigma.reshape(s)
        W_sp = self._W_sparse
        Zb = beta_f @ Z.T
        mu = _batched_sar_mean(W_sp, rho_f, Zb, n)
        nu_f = np.full(s, self._nu) if self.robust else None
        ll = _tobit_pointwise_loglik(
            self._y, mu, sigma_f, self._censored_mask, self.censoring, nu_f
        )
        jac = self._logdet_numpy_vec_fn(rho_f)
        ll = (ll + jac[:, None] / n).reshape(c, d, n)
        _write_log_likelihood_to_idata(idata, ll)
        return idata
