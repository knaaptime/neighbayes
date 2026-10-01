"""Shared helper functions for Bayesian spatial regression models.

These utilities are used by multiple base classes (SpatialModel,
SpatialPanelModel, FlowModel, FlowPanelModel) and are collected here
to avoid circular imports and code duplication.
"""

from __future__ import annotations

import inspect
import warnings
from functools import cached_property, partial
from typing import Any, Optional, Union

import numpy as np
import pandas as pd
import scipy.sparse as sp
from formulaic import model_matrix
from libpysal.graph import Graph

from ..._backends.sampler_helpers import (
    prepare_compile_kwargs,
    prepare_idata_kwargs,
    use_jax_likelihood,
)
from ..._lazy_deps import az, pm, xr
from ..._logdet import (
    make_logdet_fn,
    make_logdet_grad_numpy_vec_fn,
    make_logdet_numpy_fn,
    make_logdet_numpy_vec_fn,
)


def gelman_default_beta_prior(
    y: np.ndarray,
    design: np.ndarray,
    feature_names: list[str],
    scale: float = 2.5,
) -> tuple[np.ndarray, np.ndarray]:
    r"""Weakly-informative default prior on regression coefficients.

    Follows Gelman, Jakulin, Pittau & Su (2008) by setting per-column
    prior scales from ``sd(y)`` and ``sd(x_j)``.  For each column ``j``
    of ``design``:

    * **Intercept-like** (named ``"intercept"`` or numerically constant):
      ``mu_j = mean(y)``, ``sigma_j = scale * sd(y)``.
    * **Slope**:
      ``mu_j = 0``, ``sigma_j = scale * sd(y) / sd(x_j)``.

    Parameters
    ----------
    y : ndarray, shape (n,)
        Response vector.
    design : ndarray, shape (n, p)
        Effective design matrix used by ``beta`` in the model
        (i.e. ``X`` for SAR/SEM/OLS; ``[X, WX]`` for SDM/SDEM/SLX).
    feature_names : list[str]
        Column labels aligned with ``design``.  Used to detect
        intercept-like columns named ``"intercept"``.
    scale : float, default 2.5
        Multiplier on the standardized prior scale.

    Returns
    -------
    beta_mu : ndarray, shape (p,)
    beta_sigma : ndarray, shape (p,)

    References
    ----------
    Gelman, A., Jakulin, A., Pittau, M. G., & Su, Y.-S. (2008).
    *A weakly informative default prior distribution for logistic and
    other regression models.* Annals of Applied Statistics, 2(4),
    1360-1383.
    """
    sd_y = float(np.std(y))
    if sd_y <= 0.0:
        sd_y = 1.0
    mean_y = float(np.mean(y))
    p = design.shape[1]
    beta_mu = np.zeros(p, dtype=np.float64)
    beta_sigma = np.empty(p, dtype=np.float64)
    for j in range(p):
        col = design[:, j]
        name = feature_names[j] if j < len(feature_names) else ""
        is_named_intercept = name.lower() == "intercept"
        is_constant = np.allclose(col, col[0])
        if is_named_intercept or is_constant:
            beta_mu[j] = mean_y
            beta_sigma[j] = scale * sd_y
        else:
            sd_col = float(np.std(col))
            beta_sigma[j] = scale * sd_y / sd_col if sd_col > 0.0 else scale * sd_y
    return beta_mu, beta_sigma


_ROW_STD_ATOL = 1e-6


def _row_sums_csr(W_csr: sp.csr_matrix) -> np.ndarray:
    """Row sums of ``W`` as a flat float array."""
    return np.asarray(W_csr.sum(axis=1)).ravel()


def _isolate_mask_csr(W_csr: sp.csr_matrix) -> np.ndarray:
    """Boolean mask of *isolates* — units whose row sums to zero.

    Isolates are units with no neighbour inside the bandwidth, a normal outcome
    of distance-band contiguity: row-standardizing leaves their row at 0 because
    there is nothing to divide by.  They are valid weights, but they do break
    ``W1 = 1``, so the spatial-multiplier closed forms carry an explicit isolate
    term rather than assuming row-stochasticity.  See
    :meth:`SpatialEffectsMixin._batch_mean_row_sum`.
    """
    return np.abs(_row_sums_csr(W_csr)) <= _ROW_STD_ATOL


def _is_row_standardized_csr(W_csr: sp.csr_matrix) -> bool:
    """Return True when every non-isolate row sum is numerically close to one.

    Isolates are accepted rather than treated as a standardization failure:
    a distance-band ``W`` with unreachable units is correctly standardized, and
    rejecting it would push a common, valid case onto the eigenvalue path this
    library exists to avoid.  The closed forms stay available — they just take
    an isolate correction (see :func:`_isolate_mask_csr`).

    An all-zero ``W`` has no standardized row to judge and returns False.
    """
    row_sums = _row_sums_csr(W_csr)
    connected = ~_isolate_mask_csr(W_csr)
    return bool(
        connected.any() and np.allclose(row_sums[connected], 1.0, atol=_ROW_STD_ATOL)
    )


def _check_row_standardization(W_csr: sp.csr_matrix, *, stacklevel: int = 4) -> bool:
    """Return the row-standardization flag, warning when ``W`` looks unstandardized.

    Read off the matrix rather than a :class:`~libpysal.graph.Graph`'s
    ``transformation`` tag.  The tag records only that ``transform('r')`` was
    called, not what came out; the matrix is the one reliable witness, and
    checking it costs a single ``nnz``-length reduction.
    """
    if _is_row_standardized_csr(W_csr):
        return True
    warnings.warn(
        "W does not appear to be row-standardized (row sums ∉ {0, 1}). "
        "Most spatial models assume W is row-standardized; results may be "
        "unreliable otherwise. For a scipy sparse matrix normalize rows "
        "manually (divide each row by its sum). To use a libpysal.graph.Graph "
        "set its transformation attribute: "
        "graph = graph.transform('r'). "
        "Rows summing to 0 are isolates — those are valid and not the problem.",
        UserWarning,
        stacklevel=stacklevel,
    )
    return False


def resolve_W(
    W: Union[Graph, sp.spmatrix],
    n: int,
    T: int = 1,
) -> tuple[sp.csr_matrix, bool]:
    """Validate and normalize a spatial weights argument to CSR.

    Unified W parser for cross-section (T=1) and panel (T>1) models.
    Accepts a :class:`libpysal.graph.Graph` or any :class:`scipy.sparse`
    matrix.

    Parameters
    ----------
    W :
        Either a :class:`libpysal.graph.Graph` or any :class:`scipy.sparse`
        matrix.
    n :
        Expected number of cross-sectional units.
    T :
        Number of time periods.  When ``T=1`` (default), W must be ``n×n``.
        When ``T>1``, W may be ``n×n`` (broadcast over time) or
        ``(n*T)×(n*T)`` (full block-diagonal panel matrix).

    Returns
    -------
    W_csr : scipy.sparse.csr_matrix
        Row-compressed version of W.
    row_std : bool
        Whether W appears to be row-standardized.

    Raises
    ------
    TypeError
        If *W* is not a Graph or scipy sparse matrix.
    ValueError
        If *W* is not square or its size does not match *n* (or *n*T*).

    Warns
    -----
    UserWarning
        If *W* does not appear to be row-standardized.
    """
    if isinstance(W, Graph):
        W_csr = W.sparse.tocsr().astype(np.float64)
    elif sp.issparse(W):
        W_csr = W.tocsr().astype(np.float64)
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
        raise ValueError(f"W must be a square matrix, got shape {W_csr.shape}.")

    if T > 1:
        # Panel mode: accept n×n or (n*T)×(n*T)
        if W_csr.shape[0] == n:
            pass  # n×n — will be Kronecker-expanded by caller
        elif W_csr.shape[0] == n * T:
            pass  # full block-diagonal panel matrix
        else:
            raise ValueError(
                f"W has shape {W_csr.shape} but data has N={n} units (T={T} periods). "
                f"W must be ({n},{n}) or ({n * T},{n * T})."
            )
    else:
        # Cross-section mode: must be n×n
        if W_csr.shape[0] != n:
            raise ValueError(
                f"W has shape {W_csr.shape} but data has {n} observations. "
                "W must be an n\u00d7n matrix."
            )

    return W_csr, _check_row_standardization(W_csr)


def _parse_W(
    W: Union[Graph, sp.spmatrix],
    n: int,
) -> tuple[sp.csr_matrix, bool]:
    """Backward-compatible alias for :func:`resolve_W` with ``T=1``."""
    return resolve_W(W, n, T=1)


def _pointwise_gaussian_loglik(
    eps: np.ndarray,
    sigma_draws: np.ndarray,
    nu_draws: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Compute pointwise Gaussian or Student-t log-likelihood.

    Parameters
    ----------
    eps : np.ndarray
        Residual matrix of shape ``(n_draws, n_obs)``.
    sigma_draws : np.ndarray
        Posterior scale draws of shape ``(n_draws,)``.
    nu_draws : np.ndarray, optional
        Student-t degrees-of-freedom draws of shape ``(n_draws,)``.
        When ``None``, computes Gaussian log-likelihood.

    Returns
    -------
    np.ndarray
        Pointwise log-likelihood with shape ``(n_draws, n_obs)``.
    """
    eps = np.asarray(eps, dtype=np.float64)
    sigma = np.asarray(sigma_draws, dtype=np.float64).reshape(-1)
    if eps.ndim != 2:
        raise ValueError(f"eps must be 2D (n_draws, n_obs), got shape {eps.shape}.")
    if sigma.shape[0] != eps.shape[0]:
        raise ValueError(
            "sigma_draws length must equal eps first dimension; "
            f"got {sigma.shape[0]} and {eps.shape[0]}."
        )

    sigma = np.maximum(sigma, np.finfo(np.float64).tiny)
    sigma_2d = sigma[:, None]

    if nu_draws is None:
        return (
            -0.5 * (eps / sigma_2d) ** 2 - np.log(sigma_2d) - 0.5 * np.log(2.0 * np.pi)
        )

    nu = np.asarray(nu_draws, dtype=np.float64).reshape(-1)
    if nu.shape[0] != eps.shape[0]:
        raise ValueError(
            "nu_draws length must equal eps first dimension; "
            f"got {nu.shape[0]} and {eps.shape[0]}."
        )
    nu = np.maximum(nu, np.finfo(np.float64).tiny)
    from scipy import stats

    return stats.t.logpdf(eps, df=nu[:, None], loc=0.0, scale=sigma_2d)


def _write_log_likelihood_to_idata(
    idata: xr.DataTree,
    ll_array: np.ndarray,
) -> None:
    """Write a complete pointwise log-likelihood array to DataTree.

    Parameters
    ----------
    idata : xr.DataTree
        Target inference data object to mutate in place.
    ll_array : np.ndarray
        Array with shape ``(chain, draw, obs)``.
    """
    import xarray as xr

    ll = np.asarray(ll_array, dtype=np.float64)
    if ll.ndim != 3:
        raise ValueError(
            f"ll_array must be 3D (chain, draw, obs), got shape {ll.shape}."
        )

    ll_da = xr.DataArray(ll, dims=("chain", "draw", "obs_dim"), name="obs")
    idata["log_likelihood"] = xr.Dataset({"obs": ll_da})


def _tobit_pointwise_loglik(
    y: np.ndarray,
    mu: np.ndarray,
    sigma_f: np.ndarray,
    censored: np.ndarray,
    censoring: float,
    nu_f: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Pointwise Tobit log-likelihood (density part, no Jacobian).

    Uncensored observations contribute the log-density of the (Normal, or
    Student-t when ``nu_f`` is given) innovation evaluated at ``y``; censored
    observations contribute the corresponding left-tail log-CDF at the
    censoring point ``c``.  The caller adds the ``log|I - ρW|`` Jacobian.

    Parameters
    ----------
    y : np.ndarray
        Observed (censored) response, shape ``(n,)``.
    mu : np.ndarray
        Latent mean per flattened posterior draw, shape ``(s, n)``.
    sigma_f : np.ndarray
        Innovation scale per draw, shape ``(s,)``.
    censored : np.ndarray
        Boolean mask of censored observations, shape ``(n,)``.
    censoring : float
        Left-censoring threshold ``c``.
    nu_f : np.ndarray, optional
        Student-t degrees of freedom per draw, shape ``(s,)``.  ``None``
        (default) selects the Gaussian innovation.

    Returns
    -------
    np.ndarray
        Pointwise log-likelihood, shape ``(s, n)``.
    """
    s, n = mu.shape
    ll = np.empty((s, n), dtype=np.float64)
    uncens = ~censored
    sig = sigma_f[:, None]
    resid = (y[uncens][None, :] - mu[:, uncens]) / sig
    if nu_f is not None:
        from scipy.special import gammaln
        from scipy.stats import t as t_dist

        nu = nu_f[:, None]
        ll[:, uncens] = (
            gammaln((nu + 1) / 2)
            - gammaln(nu / 2)
            - 0.5 * np.log(nu * np.pi)
            - np.log(sig)
            - (nu + 1) / 2 * np.log1p(resid**2 / nu)
        )
        ll[:, censored] = t_dist.logcdf((censoring - mu[:, censored]) / sig, df=nu)
    else:
        from scipy.stats import norm

        ll[:, uncens] = -0.5 * (resid**2 + np.log(2.0 * np.pi) + 2.0 * np.log(sig))
        ll[:, censored] = norm.logcdf((censoring - mu[:, censored]) / sig)
    return ll


# Tie-break predicates used by the LM decision trees.  Each key maps to a
# ``(test_a, test_b)`` pair evaluated as ``p_a <= p_b``; panel trees use the
# same pairs with a ``panel_`` key prefix and ``Panel-`` test-name prefix.
#  - lag_pval_le_error_pval: OLS tree, both naive tests fire.
#  - robust_lag_pval_le_error_pval: OLS tree, both naive AND both robust
#    tests fire — route to the dominant single-channel model based on the
#    smaller robust p-value.  We never escalate directly to SARAR from OLS;
#    the user must fit SAR (or SEM) and re-run diagnostics from there.
#  - lag_sdm_pval_le_error_sdem_pval: SLX tree, both omitted-channel tests
#    fire — the channel with the smaller p-value (larger statistic) wins.
#  - lag_sem_pval_le_wx_sem_pval: SEM tree, both Schur-purged score tests
#    survive — lag direction → SARAR, WX direction → SDEM.  (Cross-section
#    only; the panel specs never reference this predicate.)
_DECISION_PREDICATE_PAIRS: dict[str, tuple[str, str]] = {
    "lag_pval_le_error_pval": ("LM-Lag", "LM-Error"),
    "robust_lag_pval_le_error_pval": ("Robust-LM-Lag", "Robust-LM-Error"),
    "lag_sdm_pval_le_error_sdem_pval": ("Robust-LM-Lag-SDM", "Robust-LM-Error-SDEM"),
    "lag_sem_pval_le_wx_sem_pval": ("Robust-LM-Lag", "Robust-LM-WX"),
}


class SharedSpatialMethods:
    """Behaviour-preserving mixin of methods shared verbatim by the two base
    classes (:class:`SpatialModel` and :class:`SpatialPanelModel`).

    Phase 5c converges the ~39 duplicated methods into this single home before
    the two hierarchies are collapsed.  Only methods whose bodies are *already
    identical* across the two classes live here; anything that differs (N vs NT
    sizing, cross-section vs panel Jacobians, ...) stays on the subclasses and
    overrides the mixin via normal MRO.  All methods rely only on attributes
    both classes provide (``self._X``, ``self._feature_names``, ``self.priors``,
    ``self._idata``).

    Structure-dependent operations (spatial lag, logdet ``W`` operand) are
    delegated to ``self._structure`` (a
    :class:`neighbayes.models._base._structure.SpatialStructure`), which the two
    base ``__init__`` methods assign.  The class-level ``None`` default keeps the
    attribute resolvable for classes that do not use it (e.g. flow models, which
    shadow ``_spatial_lag`` and never touch the single-``W`` logdet path).
    """

    _structure = None

    # Panel decision trees use "Panel-"-prefixed test names, "panel_"-prefixed
    # predicate keys, and the panel spec registry; SpatialPanelModel flips this.
    _panel_diagnostics = False

    @property
    def _nonintercept_indices(self) -> list[int]:
        """Return indices of non-constant (non-intercept) columns in X.

        This is used to exclude the intercept from impact measures, since
        the intercept has no meaningful spatial effect interpretation.

        Returns
        -------
        list[int]
            Column indices of X that are not constant/intercept columns.
        """
        indices: list[int] = []
        for j, name in enumerate(self._feature_names):
            column = self._X[:, j]
            is_named_intercept = name.lower() == "intercept"
            is_constant = np.allclose(column, column[0])
            if not (is_named_intercept or is_constant):
                indices.append(j)
        return indices

    @property
    def _nonintercept_feature_names(self) -> list[str]:
        """Return feature names for non-intercept columns.

        Returns
        -------
        list[str]
            Feature names excluding intercept/constant columns.
        """
        return [self._feature_names[i] for i in self._nonintercept_indices]

    @property
    def _beta_nonintercept_indices(self) -> list[int]:
        """Indices into the *posterior beta* for non-intercept columns.

        When the intercept is present in beta (NUTS or cross-section models),
        these are the same as ``_nonintercept_indices``.
        When the intercept was dropped (Gibbs + FE), the indices are
        shifted to account for the missing column.

        Returns
        -------
        list[int]
        """
        if not getattr(self, "_intercept_dropped", False):
            return list(self._nonintercept_indices)
        ni = self._nonintercept_indices
        n_const = ni[0] if ni else 0
        return [i - n_const for i in ni]

    @property
    def _beta_wx_column_indices(self) -> list[int]:
        """Indices into the *posterior beta1* (first k cols) for WX columns.

        When the intercept is present, same as ``_wx_column_indices``.
        When dropped (Gibbs + FE), shifted to account for the missing column.

        Returns
        -------
        list[int]
        """
        if not getattr(self, "_intercept_dropped", False):
            return list(getattr(self, "_wx_column_indices", []))
        ni = self._nonintercept_indices
        n_const = ni[0] if ni else 0
        return [i - n_const for i in getattr(self, "_wx_column_indices", [])]

    @property
    def _nu(self) -> float:
        """Student-t degrees of freedom for robust models (a fixed constant).

        Following LeSage (2009), :math:`\\nu` is a fixed hyperparameter (his
        ``rval``, default 4) rather than a sampled quantity.  Fixing it is
        what lets the NUTS and Gibbs paths target *identical* posteriors: the
        Gibbs sampler reaches the Student-t likelihood through the scale
        mixture :math:`\\varepsilon \\sim N(0, \\sigma^2 V)`,
        :math:`V = \\mathrm{diag}(v_i)`, with :math:`r/v_i \\sim \\chi^2(r)`,
        whose conditional is only conjugate for known :math:`r`.

        Values around 4 give genuinely fat tails (outliers are downweighted);
        large values (≳ 30) approach the Normal.  A lower bound of 2 is
        enforced so the variance exists.
        """
        nu = float(self.priors.get("nu", 4.0))
        if nu <= 2.0:
            raise ValueError(
                f"priors['nu'] must be > 2 so the Student-t variance exists; got {nu}."
            )
        return nu

    def _beta_names(self) -> list[str]:
        """Return coefficient labels used for posterior summaries.

        For models with ``_has_wx_in_beta=True`` (SLX, SDM, SDEM), appends
        ``W*{name}`` labels for the spatially-lagged covariates.

        Returns
        -------
        list[str]
            Coefficient labels aligned with the ``beta`` parameter.
        """
        names = list(self._feature_names)
        if getattr(self, "_has_wx_in_beta", False) and getattr(
            self, "_wx_feature_names", None
        ):
            names += [f"W*{name}" for name in self._wx_feature_names]
        return names

    def _model_coords(self, extra: Optional[dict] = None) -> dict[str, list[str]]:
        """Return PyMC coordinate labels for named dimensions.

        Parameters
        ----------
        extra : dict, optional
            Additional coordinates to merge into the result.

        Returns
        -------
        dict[str, list[str]]
            Coordinates passed to :class:`pymc.Model`.
        """
        coords: dict[str, list[str]] = {"coefficient": self._beta_names()}
        if extra:
            coords.update(extra)
        return coords

    # ------------------------------------------------------------------
    # Shared spatial-effects helpers
    # ------------------------------------------------------------------
    # These centralise the repeated direct/indirect/total effects formulas
    # used by SAR, SEM, SLX/SDEM, and SDM models (cross-section and panel).
    # Each returns ``(direct, indirect, total)`` arrays of shape ``(G, k)``.

    def _sar_effects(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """SAR impacts: ``Direct = mean_diag(S)·β``, ``Total = mean_rowsum(S)·β``.

        Uses the resolvent trace identities — no O(n³) eigendecomposition.
        """
        from ...diagnostics.lmtests import _get_posterior_draws

        idata = self.inference_data
        rho_draws = _get_posterior_draws(idata, "rho")
        beta_draws = _get_posterior_draws(idata, "beta")

        mean_diag = self._batch_mean_diag(rho_draws)
        mean_row_sum = self._batch_mean_row_sum(rho_draws)

        ni = self._beta_nonintercept_indices
        direct = mean_diag[:, None] * beta_draws[:, ni]
        total = mean_row_sum[:, None] * beta_draws[:, ni]
        indirect = total - direct
        return direct, indirect, total

    def _sem_effects(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """SEM impacts: ``Direct = β``, ``Indirect = 0``, ``Total = β``.

        The spatial-error filter does not affect partial derivatives of
        :math:`y` w.r.t. :math:`X`.
        """
        from ...diagnostics.lmtests import _get_posterior_draws

        idata = self.inference_data
        beta_draws = _get_posterior_draws(idata, "beta")

        ni = self._beta_nonintercept_indices
        direct = beta_draws[:, ni].copy()
        indirect = np.zeros_like(direct)
        total = direct.copy()
        return direct, indirect, total

    def _slx_effects(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """SLX/SDEM impacts: linear in β₁ and β₂ via W diagonal/row-sum means.

        ``Direct = β₁ + mean_diag(W)·β₂``, ``Total = β₁ + mean_rowsum(W)·β₂``.
        """
        from ...diagnostics.lmtests import _get_posterior_draws

        idata = self.inference_data
        beta_draws = _get_posterior_draws(idata, "beta")

        if getattr(self, "_intercept_dropped", False):
            k = len(self._beta_nonintercept_indices)
        else:
            k = self._X.shape[1]
        kw = self._WX.shape[1]

        beta1 = beta_draws[:, :k]
        beta2 = beta_draws[:, k : k + kw]

        mean_diag_w = float(self._W_sparse.diagonal().mean())
        mean_row_sum_w = float(self._W_sparse.sum() / self._W_sparse.shape[0])

        wx_idx = self._beta_wx_column_indices
        direct = beta1[:, wx_idx] + mean_diag_w * beta2
        total = beta1[:, wx_idx] + mean_row_sum_w * beta2
        indirect = total - direct
        return direct, indirect, total

    def _sdm_effects(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """SDM impacts: multiplier acts on both X and WX.

        ``Direct = mean_diag(M)·β₁ + mean_diag(MW)·β₂``,
        ``Total = mean_rowsum(M)·β₁ + mean_rowsum(MW)·β₂``.
        """
        from ...diagnostics.lmtests import _get_posterior_draws

        idata = self.inference_data
        rho_draws = _get_posterior_draws(idata, "rho")
        beta_draws = _get_posterior_draws(idata, "beta")

        if getattr(self, "_intercept_dropped", False):
            k = len(self._beta_nonintercept_indices)
        else:
            k = self._X.shape[1]
        kw = self._WX.shape[1]

        beta1 = beta_draws[:, :k]
        beta2 = beta_draws[:, k : k + kw]

        mean_diag_M = self._batch_mean_diag(rho_draws)
        mean_diag_MW = self._batch_mean_diag_MW(rho_draws)
        mean_row_sum_M = self._batch_mean_row_sum(rho_draws)
        mean_row_sum_MW = self._batch_mean_row_sum_MW(rho_draws)

        wx_idx = self._beta_wx_column_indices
        direct = mean_diag_M[:, None] * beta1[:, wx_idx] + mean_diag_MW[:, None] * beta2
        total = (
            mean_row_sum_M[:, None] * beta1[:, wx_idx]
            + mean_row_sum_MW[:, None] * beta2
        )
        indirect = total - direct
        return direct, indirect, total

    # ------------------------------------------------------------------
    # Default fitted values at posterior mean
    # ------------------------------------------------------------------

    def _fitted_mean_from_posterior(self) -> np.ndarray:
        """Compute fitted values at posterior mean parameters.

        Dispatches on ``_jacobian_param`` and ``_has_wx_in_beta``:

        * **SAR/SDM** (``_jacobian_param == "rho"``): ``ρ·Wy + Z·β``
        * **SEM/SDEM** (``_jacobian_param == "lam"`` or ``None``): ``Z·β``
        * Models with ``_has_wx_in_beta``: ``Z = [X, WX]``, else ``Z = X``
        * Panel FE with ``_intercept_dropped``: subset X columns

        Subclasses with random effects (panel RE) or dynamic terms
        override this method.
        """
        beta = self._posterior_mean("beta")

        if self._has_wx_in_beta:
            if getattr(self, "_intercept_dropped", False):
                ni = self._beta_nonintercept_indices
                Z = np.hstack([self._X[:, ni], self._WX])
            else:
                Z = np.hstack([self._X, self._WX])
        else:
            if getattr(self, "_intercept_dropped", False):
                ni = self._beta_nonintercept_indices
                Z = self._X[:, ni]
            else:
                Z = self._X

        fitted = Z @ beta

        if self._jacobian_param == "rho":
            rho = float(self._posterior_mean("rho"))
            fitted = rho * self._Wy + fitted

        return fitted

    @staticmethod
    def _rename_summary_index(summary_df: pd.DataFrame) -> pd.DataFrame:
        """Strip the ``beta[...]`` wrapper from coefficient row labels.

        Parameters
        ----------
        summary_df : pandas.DataFrame
            ArviZ summary output.

        Returns
        -------
        pandas.DataFrame
            Summary with human-readable coefficient row labels.
        """
        renamed = []
        for label in summary_df.index.astype(str):
            if label.startswith("beta[") and label.endswith("]"):
                renamed.append(label[5:-1])
            else:
                renamed.append(label)
        out = summary_df.copy()
        out.index = renamed
        return out

    @property
    def inference_data(self) -> Optional[xr.DataTree]:
        """Return the ArviZ DataTree from the most recent fit.

        Returns
        -------
        xarray.DataTree or None
            The inference data object, or ``None`` if the model has not
            been fit yet.
        """
        return self._idata

    def _require_fit(self):
        if self._idata is None:
            raise RuntimeError("Model has not been fit yet. Call .fit() first.")

    def _posterior_mean(self, var: str) -> np.ndarray:
        return self._idata.posterior[var].mean(("chain", "draw")).to_numpy()

    def _postprocess_idata(self, idata: xr.DataTree) -> xr.DataTree:
        """Hook to augment ``idata`` after sampling, before it is returned.

        The default is a no-op.  Subclasses whose likelihood is expressed via
        ``pm.Potential`` (so PyMC captures no pointwise ``log_likelihood``),
        e.g. the (panel) Tobit families, override this to attach a complete
        Jacobian-corrected pointwise log-likelihood for information criteria.
        """
        return idata

    # ------------------------------------------------------------------
    # Shared NUTS sampling path
    # ------------------------------------------------------------------

    def _fit_nuts(
        self,
        *,
        draws: int,
        tune: int,
        chains: int,
        target_accept: float = 0.9,
        random_seed: Optional[int],
        progressbar: bool,
        nuts_sampler: str = "pymc",
        idata_kwargs: dict[str, Any] | None = None,
        compute_log_likelihood: bool = False,
        sample_kwargs: dict[str, Any] | None = None,
    ) -> tuple["xr.DataTree", bool]:
        """Shared NUTS sampling path used by model-specific ``fit`` methods.

        Inspects ``_build_pymc_model`` for optional ``compute_log_likelihood``
        and ``nuts_sampler`` parameters, builds the model, runs ``pm.sample``,
        and returns ``(idata, compute_log_likelihood)`` where the boolean
        reflects the post-policy value after :func:`prepare_idata_kwargs`.

        Returns
        -------
        tuple[xarray.DataTree, bool]
        """
        sample_kwargs = dict(sample_kwargs or {})
        idata_kwargs = dict(idata_kwargs or {})

        build_kwargs: dict[str, Any] = {}
        build_sig = inspect.signature(self._build_pymc_model)
        if "compute_log_likelihood" in build_sig.parameters:
            build_kwargs["compute_log_likelihood"] = compute_log_likelihood
        if "nuts_sampler" in build_sig.parameters:
            build_kwargs["nuts_sampler"] = nuts_sampler

        model = self._build_pymc_model(**build_kwargs)
        self._pymc_model = model

        idata_kwargs = prepare_idata_kwargs(idata_kwargs, model, nuts_sampler)
        compute_log_likelihood = bool(idata_kwargs.get("log_likelihood", False))
        sample_kwargs = prepare_compile_kwargs(sample_kwargs, nuts_sampler)

        with model:
            self._idata = pm.sample(
                draws=draws,
                tune=tune,
                chains=chains,
                target_accept=target_accept,
                random_seed=random_seed,
                idata_kwargs=idata_kwargs,
                nuts_sampler=nuts_sampler,
                progressbar=progressbar,
                **sample_kwargs,
            )

        return self._idata, compute_log_likelihood

    # ------------------------------------------------------------------
    # Shared Gaussian log-likelihood reconstruction
    # ------------------------------------------------------------------

    def _reconstruct_gaussian_log_likelihood(
        self,
        *,
        spatial_param: str,
        nuts_sampler: str,
        spatial_lag_fn=None,
        alpha_component: Optional[np.ndarray] = None,
    ) -> None:
        """Rebuild complete pointwise log-likelihood for Gaussian spatial models.

        Shared core for cross-sectional and panel models.  Dispatches by
        spatial term type (``rho`` vs ``lam``), computes per-draw residuals,
        applies the Jacobian correction, and writes back to ``idata``.

        Parameters
        ----------
        spatial_param : str
            Either ``"rho"`` (SAR/SDM) or ``"lam"`` (SEM/SDEM).
        nuts_sampler : str
            NUTS sampler name — JAX backends already capture the complete
            log-likelihood for SEM, so reconstruction is skipped.
        spatial_lag_fn : callable, optional
            Function ``f(resid) -> W_resid`` for the SEM whitening step.
            If ``None``, uses ``self._spatial_lag`` (which delegates to the
            model's ``_structure``).
        alpha_component : np.ndarray, optional
            Random-effects contribution for panel RE models, shape
            ``(n_draws, n_obs)``.  Added to the mean for ``rho`` models and
            subtracted from residuals for ``lam`` models.
        """
        if not hasattr(self, "_idata"):
            return

        if spatial_param not in {"rho", "lam"}:
            return

        if getattr(self, "_native_log_likelihood", False):
            return

        if spatial_param == "lam" and use_jax_likelihood(nuts_sampler):
            return

        idata = self._idata
        n = int(self._y.shape[0])
        Z = np.hstack([self._X, self._WX]) if self._has_wx_in_beta else self._X

        spatial_draws = idata.posterior[spatial_param].values.reshape(-1)
        beta_draws = idata.posterior["beta"].values.reshape(-1, Z.shape[1])
        sigma_draws = idata.posterior["sigma"].values.reshape(-1)
        nu_draws = np.full(spatial_draws.shape[0], self._nu) if self.robust else None

        if spatial_param == "rho":
            mu = spatial_draws[:, None] * self._Wy[None, :] + (beta_draws @ Z.T)
            if alpha_component is not None:
                mu = mu + alpha_component
            eps = self._y[None, :] - mu
        else:
            resid = self._y[None, :] - (beta_draws @ Z.T)
            if alpha_component is not None:
                resid = resid - alpha_component
            if spatial_lag_fn is not None:
                W_resid = spatial_lag_fn(resid)
            else:
                # Cross-section default: (W @ resid.T).T for batch (draws, n).
                W_resid = np.asarray(self._W_sparse @ resid.T, dtype=np.float64).T
            eps = resid - spatial_draws[:, None] * W_resid

        ll_data = _pointwise_gaussian_loglik(eps, sigma_draws, nu_draws)
        jacobian = self._logdet_numpy_vec_fn(spatial_draws)
        ll_total = ll_data + jacobian[:, None] / n

        n_chains = idata.posterior.sizes["chain"]
        n_draws_per_chain = idata.posterior.sizes["draw"]
        _write_log_likelihood_to_idata(
            idata,
            ll_total.reshape(n_chains, n_draws_per_chain, n),
        )

    @staticmethod
    def _spatial_lag_column_indices(
        X: np.ndarray, feature_names: list[str]
    ) -> list[int]:
        """Return indices of regressors that should receive spatial lags.

        Constant columns are treated as intercept-like and excluded, which
        avoids adding redundant ``W * intercept`` terms to SLX/Durbin models.

        Parameters
        ----------
        X : np.ndarray
            Design matrix (raw, before any panel fixed-effects transformation).
        feature_names : list[str]
            Column labels aligned with ``X``.

        Returns
        -------
        list[int]
            Column indices eligible for spatial lags.
        """
        indices: list[int] = []
        for j, name in enumerate(feature_names):
            column = X[:, j]
            is_named_intercept = name.lower() == "intercept"
            is_constant = np.allclose(column, column[0])
            if not (is_named_intercept or is_constant):
                indices.append(j)
        return indices

    def summary(self, var_names: Optional[list] = None, **kwargs) -> pd.DataFrame:
        """Return posterior summary table.

        Parameters
        ----------
        var_names : list, optional
            Variable names to include in the summary.
        **kwargs
            Additional arguments passed to :func:`arviz.summary`.

        Returns
        -------
        pandas.DataFrame
            Posterior summary statistics.
        """
        self._require_fit()
        summary_df = az.summary(self._idata, var_names=var_names, **kwargs)
        return self._rename_summary_index(summary_df)

    def fitted_values(self) -> np.ndarray:
        """Return fitted values at posterior mean parameters.

        Returns
        -------
        np.ndarray
            Posterior-mean fitted values (on the model's native scale;
            fixed-effects-transformed for panel models).
        """
        self._require_fit()
        return self._fitted_mean_from_posterior()

    def residuals(self) -> np.ndarray:
        """Return residuals ``y - fitted_values``.

        Returns
        -------
        np.ndarray
            Residual vector ``y - fitted_values`` on the same scale as
            :meth:`fitted_values`.
        """
        self._require_fit()
        return self._y - self.fitted_values()

    def _spatial_lag(self, X: np.ndarray) -> np.ndarray:
        """Compute the spatial lag of ``X`` for the spatial filter.

        Delegates to the model's :attr:`_structure`: ``W @ X`` for
        cross-section, the Kronecker ``W ⊗ I_T`` lag for panel.
        """
        return self._structure.spatial_lag(X)

    def _gelman_default_beta_prior(
        self,
        design: np.ndarray,
        feature_names: list[str],
        scale: float = 2.5,
    ) -> tuple[np.ndarray, np.ndarray]:
        r"""Weakly-informative default prior on regression coefficients.

        Thin wrapper around :func:`gelman_default_beta_prior` that uses
        ``self._y`` as the response.  See that function for details.
        """
        return gelman_default_beta_prior(self._y, design, feature_names, scale=scale)

    @cached_property
    def _W_eigs(self) -> np.ndarray | None:
        """Eigenvalues of the N×N spatial weights matrix (complex), lazy.

        For large ``n`` this is O(n³), so it is only computed when needed
        (e.g. by the eigenvalue logdet method); Chebyshev / trace / sparse-grid
        methods never trigger it.  Cross-section models return ``None`` when no
        ``W`` was supplied (panel models always have a ``W``).

        Raises
        ------
        ValueError
            If ``n`` exceeds ``NEIGHBAYES_LOGDET_EIGEN_MAX_N`` (default 500).
            The eigenvalue logdet method densifies ``W`` and runs an O(n³)
            eigendecomposition, which is prohibitively expensive for large
            graphs.  Use ``logdet_method="cheb_cholesky"`` (symmetric ``W``)
            or ``logdet_method="aaa"`` (non-symmetric ``W``) instead.
        """
        if self._W_sparse is None:
            return None
        from ..._logdet._config import _eigen_hard_max_n

        n = self._W_sparse.shape[0]
        if n > _eigen_hard_max_n():
            raise ValueError(
                f"Eigenvalue logdet/effect computation requires densifying W "
                f"(n={n} > NEIGHBAYES_LOGDET_EIGEN_HARD_MAX_N={_eigen_hard_max_n()}). "
                f"This is O(n³) and O(n²) memory. Use "
                f'logdet_method="cheb_cholesky" (symmetric W) or '
                f'logdet_method="aaa" (non-symmetric W) instead, or raise '
                f"NEIGHBAYES_LOGDET_EIGEN_HARD_MAX_N."
            )
        return np.linalg.eigvals(self._W_sparse.toarray().astype(np.float64))

    @cached_property
    def _W_spectral_bounds(self) -> tuple[float, float]:
        r"""``(λ_max, λ_min)`` bounds on ``W``'s spectrum, without densifying.

        The Gibbs samplers need these only to bracket ``A_ρ = I - ρW``: the
        SPD guard ``min(1-ρλ_max, 1-ρλ_min) > 0``, the Chebyshev interval for
        the iterative fallback, and the Krylov convergence radius. All three
        are correct with *bounds* — none needs the actual spectrum.

        Taking them from :attr:`_W_eigs` costs a dense O(n³) eigendecomposition
        that dominated the whole fit (89% of a 12.8 s run at n=4900) and hard
        failed past ``NEIGHBAYES_LOGDET_EIGEN_HARD_MAX_N``. Avoiding it is the
        point of the sparse factorizations.

        For row-standardized ``W`` the bounds are not merely cheap but
        *exact*: ``W1 = 1`` makes 1 an eigenvalue, and row-stochasticity puts
        every eigenvalue in the closed unit disc, so ``λ_max = 1``. Otherwise
        fall back to the Gershgorin/``∞``-norm bound ``|λ| ≤ max_i Σ_j |W_ij|``,
        which is O(nnz).
        """
        if self._W_sparse is None:
            return 1.0, -1.0
        if getattr(self, "_is_row_std", False):
            return 1.0, -1.0
        radius = float(abs(self._W_sparse).sum(axis=1).max())
        radius = max(radius, 1e-12)
        return radius, -radius

    @cached_property
    def _T_ww(self) -> float:
        """Sparse trace ``tr(WᵀW + WW)`` used by LM diagnostics (lazy)."""
        from ...graph import sparse_trace_WtW_plus_WW

        return sparse_trace_WtW_plus_WW(self._W_sparse)

    @property
    def _logdet_W_operand(self):
        """Non-eigenvalue ``W`` operand handed to the logdet factory.

        Delegates to the model's :attr:`_structure` (dense ``W`` for
        cross-section, sparse ``W`` for panel).
        """
        return self._structure.logdet_W_operand()

    @cached_property
    def _logdet_eigs(self) -> np.ndarray | None:
        """Eigenvalues when the resolved method needs them, else ``None``.

        The single place that decides whether the O(n³) eigendecomposition is
        required.  Every logdet evaluator reads this instead of re-deriving it
        from the method name, so the decision cannot drift between them.
        """
        self._require_W()
        if self._logdet_bounds.method != "eigenvalue":
            return None
        return self._W_eigs

    @cached_property
    def _W_for_logdet(self):
        """Argument passed to ``make_logdet_fn`` — eigenvalues or the
        structure-specific ``W`` operand (:attr:`_logdet_W_operand`).

        Computed lazily so that init never forces an eigendecomposition for
        chebyshev / sparse-grid methods.
        """
        eigs = self._logdet_eigs
        return self._logdet_W_operand if eigs is None else eigs

    def spatial_effects(
        self, return_posterior_samples: bool = False
    ) -> pd.DataFrame | tuple[pd.DataFrame, dict[str, np.ndarray]]:
        r"""Compute Bayesian inference for direct, indirect, and total impacts.

        Computes impact measures for each posterior draw, then summarizes
        the posterior distribution with means, 95% credible intervals, and
        Bayesian p-values.  This is the fully Bayesian analog of the
        simulation-based approach in :cite:t:`lesage2009IntroductionSpatial`
        and the asymptotic variance formulas in
        :cite:t:`arbia2020TestingImpact`.

        Models without a spatial lag on y do not exhibit global
        feedback propagation through :math:`(I-\\rho W)^{-1}`. However,
        models with spatially lagged covariates (SLX, SDEM) can still
        have non-zero neighbor spillovers captured in the indirect term.

        Parameters
        ----------
        return_posterior_samples : bool, optional
            If ``True``, return a ``(DataFrame, dict)`` tuple where the
            dict contains the full posterior draws under keys
            ``"direct"``, ``"indirect"``, and ``"total"``.  Default
            ``False``.

        Returns
        -------
        pd.DataFrame or tuple of (pd.DataFrame, dict)
            If *return_posterior_samples* is ``False`` (default), returns
            a DataFrame indexed by feature names with columns for posterior
            means, credible-interval bounds, and Bayesian p-values.

            If *return_posterior_samples* is ``True``, returns
            ``(DataFrame, dict)`` where the dict has keys
            ``"direct"``, ``"indirect"``, ``"total"``, each mapping
            to a ``(G, k)`` array of posterior draws.
        """
        from ...diagnostics.spatial_effects import _build_effects_dataframe

        self._require_fit()
        direct_samples, indirect_samples, total_samples = (
            self._compute_spatial_effects_posterior()
        )

        # Determine feature names based on the shape of the posterior samples.
        # Models with WX terms (SDM, SLX, SDEM) report effects only for
        # lagged covariates (k_wx columns), while models without WX terms
        # (SAR, SEM) report effects for non-intercept covariates.
        k_effects = direct_samples.shape[1]
        if (
            hasattr(self, "_wx_feature_names")
            and len(self._wx_feature_names) == k_effects
        ):
            feature_names = list(self._wx_feature_names)
        elif (
            hasattr(self, "_nonintercept_feature_names")
            and len(self._nonintercept_feature_names) == k_effects
        ):
            feature_names = list(self._nonintercept_feature_names)
        else:
            feature_names = list(self._feature_names[:k_effects])

        # Determine model type label
        model_type = self.__class__.__name__

        df = _build_effects_dataframe(
            direct_samples=direct_samples,
            indirect_samples=indirect_samples,
            total_samples=total_samples,
            feature_names=feature_names,
            model_type=model_type,
        )

        if return_posterior_samples:
            posterior_samples = {
                "direct": direct_samples,
                "indirect": indirect_samples,
                "total": total_samples,
            }
            return df, posterior_samples
        return df

    # ------------------------------------------------------------------
    # Input parsing / guard helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_formula(formula: str, data: pd.DataFrame):
        """Parse formula/data inputs into ``y`` and ``X`` arrays.

        See :meth:`_parse_formula_with_spec`, which also returns the RHS
        model spec.
        """
        return SharedSpatialMethods._parse_formula_with_spec(formula, data)[:3]

    @staticmethod
    def _parse_formula_with_spec(formula: str, data: pd.DataFrame):
        """Parse formula/data inputs into ``y`` and ``X`` arrays.

        Parameters
        ----------
        formula : str
            Wilkinson-style formula containing ``lhs ~ rhs``.
        data : pandas.DataFrame
            Tabular data source referenced by the formula.

        Returns
        -------
        tuple[np.ndarray, np.ndarray, list[str], formulaic.ModelSpec]
            Dependent variable, design matrix, feature names, and the RHS
            model spec, which rebuilds the design for new data with the same
            transforms and categorical levels.
        """
        lhs_name, rhs = formula.split("~", 1)
        lhs_name = lhs_name.strip()
        rhs = rhs.strip()

        # Build RHS model matrix (handles intercept, interactions, transforms)
        X_mm = model_matrix(rhs, data)
        feature_names = list(X_mm.columns)
        X_arr = np.asarray(X_mm, dtype=np.float64)

        y_arr = np.asarray(data[lhs_name], dtype=np.float64)
        return y_arr, X_arr, feature_names, X_mm.model_spec

    @staticmethod
    def _parse_matrices(y, X):
        """Parse matrix-mode inputs and infer feature names.

        Parameters
        ----------
        y : array-like
            Dependent variable.
        X : array-like or pandas.DataFrame
            Predictor matrix.

        Returns
        -------
        tuple[np.ndarray, np.ndarray, list[str]]
            Dependent variable, design matrix, and feature names.
        """
        y_arr = np.asarray(y, dtype=np.float64)
        if isinstance(X, pd.DataFrame):
            feature_names = list(X.columns)
            X_arr = X.to_numpy(dtype=np.float64)
        else:
            X_arr = np.asarray(X, dtype=np.float64)
            feature_names = [f"x{i}" for i in range(X_arr.shape[1])]
        return y_arr, X_arr, feature_names

    def _require_W(self):
        """Raise if no spatial weights matrix was supplied."""
        if self._W_sparse is None:
            raise ValueError(
                "This method requires a spatial weights matrix W. "
                "Pass W when constructing the model."
            )

    # ------------------------------------------------------------------
    # Lazy logdet evaluators (cross-section is the T=1 case)
    # ------------------------------------------------------------------

    # Likelihood dimensions.  A cross-section (and a pooled panel) counts
    # every observation and applies the Jacobian T times; fixed-effects
    # panels override these (Lee & Yu 2010; see ``SpatialPanelModel``).

    @property
    def _jacobian_T(self) -> int:
        """Multiplier on the per-period Jacobian ``log|I - ρW|``."""
        return getattr(self, "_T", 1)

    @property
    def _jacobian_shift(self) -> float:
        """Coefficient ``m`` of the ``-m·log(1 - ρ)`` time-effects Jacobian term."""
        return 0.0

    @property
    def _n_effective(self) -> int:
        """Independent observations the Gaussian likelihood counts."""
        return int(np.asarray(self._y).shape[0])

    @cached_property
    def _logdet_numpy_fn(self):
        """Pure-numpy ``(rho) -> float`` logdet evaluator (lazy)."""
        self._require_W()
        # The time-effects term -m·log(1 - ρ) is not included: the Gibbs
        # samplers that consume these add it themselves (``_jacobian_shift``).
        return make_logdet_numpy_fn(
            self._W_sparse,
            self._logdet_eigs,
            method=self._logdet_bounds.method,
            rho_min=self._logdet_bounds.rho_min,
            rho_max=self._logdet_bounds.rho_max,
            T=self._jacobian_T,
        )

    @cached_property
    def _logdet_numpy_vec_fn(self):
        """Vectorized pure-numpy logdet evaluator (lazy)."""
        self._require_W()
        return make_logdet_numpy_vec_fn(
            self._W_sparse,
            self._logdet_eigs,
            method=self._logdet_bounds.method,
            rho_min=self._logdet_bounds.rho_min,
            rho_max=self._logdet_bounds.rho_max,
            T=self._jacobian_T,
        )

    @cached_property
    def _logdet_grad_numpy_vec_fn(self):
        """Vectorized ``(rho_arr) -> g(ρ)`` gradient of the **N×N** logdet (lazy).

        Built with ``T=1`` even for panels — the direct-effect trace is a
        per-period property of the N×N spatial multiplier, independent of
        the number of periods.  Uses the resolved logdet method and rho
        bounds, so it rides the fast surrogate (chol-cheb / AAA / …) and
        only touches the eigenvalue path when that is genuinely the
        resolved method (tiny n).
        """
        self._require_W()
        return make_logdet_grad_numpy_vec_fn(
            self._W_sparse,
            self._logdet_eigs,
            method=self._logdet_bounds.method,
            rho_min=self._logdet_bounds.rho_min,
            rho_max=self._logdet_bounds.rho_max,
            T=1,
        )

    @cached_property
    def _logdet_pytensor_fn(self):
        """PyTensor logdet evaluator used inside ``_build_pymc_model`` (lazy).

        The full Jacobian NUTS adds, including any time-effects term.
        """
        self._require_W()
        fn = make_logdet_fn(
            self._W_for_logdet,
            method=self._logdet_bounds.method,
            rho_min=self._logdet_bounds.rho_min,
            rho_max=self._logdet_bounds.rho_max,
            T=self._jacobian_T,
        )
        m = self._jacobian_shift
        if not m:
            return fn
        import pytensor.tensor as pt

        return lambda rho: fn(rho) - m * pt.log1p(-rho)

    # ------------------------------------------------------------------
    # Eigendecomposition-backed spatial-effect helpers
    # ------------------------------------------------------------------

    @cached_property
    def _W_eigendecomposition(self):
        """Full eigendecomposition W = V diag(λ) V⁻¹ with complex128 arithmetic.

        Always decomposes the per-period (cross-sectional) N×N ``W`` — for
        panels the spatial multiplier is defined per period, never the
        (N·T)×(N·T) Kronecker product.  Returns a 3-tuple
        ``(eigs, V, Vinv)`` of complex128 arrays, or ``None`` when no
        spatial weights matrix was supplied.

        Eigenvalues are sorted by real part (descending) for numerical
        stability.  Row-standardized W is generally non-symmetric, so V
        and Vinv are complex; taking ``.real`` prematurely drops imaginary
        parts and produces wrong results for spatial effects.

        Raises
        ------
        ValueError
            If ``n`` exceeds ``NEIGHBAYES_LOGDET_EIGEN_MAX_N`` (default 500).
            See :attr:`_W_eigs` for details and alternatives.
        """
        if self._W_sparse is None:
            return None
        from ..._logdet._config import _eigen_hard_max_n

        n = self._W_sparse.shape[0]
        if n > _eigen_hard_max_n():
            raise ValueError(
                f"Eigenvalue logdet/effect computation requires densifying W "
                f"(n={n} > NEIGHBAYES_LOGDET_EIGEN_HARD_MAX_N={_eigen_hard_max_n()}). "
                f"This is O(n³) and O(n²) memory. Use "
                f'logdet_method="cheb_cholesky" (symmetric W) or '
                f'logdet_method="aaa" (non-symmetric W) instead, or raise '
                f"NEIGHBAYES_LOGDET_EIGEN_HARD_MAX_N."
            )
        W_dense = np.asarray(self._W_sparse.toarray(), dtype=np.float64)
        eigs, V = np.linalg.eig(W_dense)
        Vinv = np.linalg.inv(V)
        idx = np.argsort(eigs.real)[::-1]
        eigs_sorted = eigs[idx].astype(np.complex128)
        V_sorted = V[:, idx].astype(np.complex128)
        Vinv_sorted = Vinv[idx, :].astype(np.complex128)
        return (eigs_sorted, V_sorted, Vinv_sorted)

    @cached_property
    def _eig_inv_ones(self) -> Optional[np.ndarray]:
        """V⁻¹ @ 1ₙ (complex128), cached once.

        Used by :meth:`_batch_mean_row_sum` and
        :meth:`_batch_mean_row_sum_MW` for the eigenvalue-based
        computation of mean row sums.
        """
        decomp = self._W_eigendecomposition
        if decomp is None:
            return None
        return decomp[2] @ np.ones(decomp[0].shape[0], dtype=np.complex128)

    def _batch_mean_diag(self, rho_draws: np.ndarray) -> np.ndarray:
        """``(1/n) tr((I - ρW)⁻¹)`` per draw — the SAR/SDM direct-effect trace.

        Uses the resolvent identity ``tr(S)/n = 1 − (ρ/n)·g(ρ)`` with
        ``g(ρ) = d/dρ log|I − ρW| = −tr(W(I−ρW)⁻¹)``, so it rides the model's
        fast logdet surrogate and never triggers the O(n³) eigendecomposition
        (unless the resolved method is already ``eigenvalue``).
        """
        rho_draws = np.asarray(rho_draws, dtype=np.float64)
        n = int(self._W_sparse.shape[0])
        g = np.asarray(self._logdet_grad_numpy_vec_fn(rho_draws), dtype=np.float64)
        return 1.0 - (rho_draws / n) * g

    def _batch_mean_diag_MW(self, rho_draws: np.ndarray) -> np.ndarray:
        """``(1/n) tr((I - ρW)⁻¹ W)`` per draw — the SDM cross direct-effect trace.

        Equals ``−g(ρ)/n`` for the same ``g``, again with no eigendecomposition.
        """
        rho_draws = np.asarray(rho_draws, dtype=np.float64)
        n = int(self._W_sparse.shape[0])
        g = np.asarray(self._logdet_grad_numpy_vec_fn(rho_draws), dtype=np.float64)
        return -g / n

    @cached_property
    def _isolate_mask(self) -> np.ndarray:
        """Boolean mask of units with no neighbours — see :func:`_isolate_mask_csr`.

        Isolates are valid weights but break ``W1 = 1``, so the closed forms
        below carry an explicit isolate term.  Detecting them is one
        ``nnz``-length reduction; the alternative — routing every isolate-bearing
        ``W`` to :attr:`_W_eigendecomposition` — is O(n³) and would defeat the
        sparse factorizations this library is built on.
        """
        if self._W_sparse is None:
            return np.zeros(0, dtype=bool)
        return _isolate_mask_csr(self._W_sparse)

    @cached_property
    def _n_isolates(self) -> int:
        """Number of units with no neighbours."""
        return int(self._isolate_mask.sum())

    def _multiplier_row_sums(self, rho: float) -> np.ndarray:
        """Per-unit row sums of ``(I - ρW)⁻¹`` for row-standardized ``W``.

        ``1/(1 − ρ)`` on connected units, and exactly 1 on isolates, whose row
        of the multiplier is just ``e_i``.  O(n) and factorization-free — the
        alternative is carrying an extra ``ones`` column through the sparse
        solve on every draw.
        """
        n = int(self._W_sparse.shape[0])
        out = np.full(n, 1.0 / (1.0 - float(rho)), dtype=np.float64)
        if self._n_isolates:
            out[self._isolate_mask] = 1.0
        return out

    def _batch_mean_row_sum(self, rho_draws: np.ndarray) -> np.ndarray:
        """Compute mean row sum of (I - rho*W)^{-1} for each posterior draw.

        Always uses the N×N cross-sectional weights matrix (never the
        (N·T)×(N·T) Kronecker product), because spatial effects are defined
        in terms of the cross-sectional spatial multiplier.

        For row-standardized W this is ``1/(1 - rho)`` on the connected units.
        An isolate's row of ``(I - rho*W)^{-1}`` is just ``e_i``, so its row sum
        is exactly 1 — giving ``((n - m)/(1 - rho) + m)/n`` for ``m`` isolates.
        Still O(1) per draw: isolates cost a count, not a decomposition.  The
        ratio ``(n - m)/n`` is invariant to the Kronecker form, so this is
        correct whether ``_W_sparse`` is the N×N or the (N·T)×(N·T) matrix.

        For non-row-standardized W the eigenvalue decomposition is used:
        ``mean_row_sum = (1/n) * ones' V diag(1/(1-rho*omega)) V^{-1} ones``,
        where the vector ``c = V^{-1} ones`` is pre-computed once.

        Parameters
        ----------
        rho_draws : np.ndarray, shape (G,)
            Spatial autoregressive parameter draws.

        Returns
        -------
        np.ndarray, shape (G,)
            Mean row sum for each draw.
        """
        if self._is_row_std:
            m = self._n_isolates
            if m == 0:
                return 1.0 / (1.0 - rho_draws)
            n = int(self._W_sparse.shape[0])
            return ((n - m) / (1.0 - rho_draws) + m) / n

        # Eigenvalue-based computation using shared eigendecomposition cache.
        decomp = self._W_eigendecomposition
        if decomp is None:
            raise ValueError("No spatial weights matrix available.")
        c = self._eig_inv_ones  # complex128, (n,)
        eigs = decomp[0]  # complex128, (n,)
        V_col_sums = decomp[1].sum(axis=0)  # complex128, (n,)
        from ...diagnostics.spatial_effects import _chunked_eig_means

        return _chunked_eig_means(
            rho_draws,
            eigs,
            weights=V_col_sums * c,
        )

    def _batch_mean_row_sum_MW(self, rho_draws: np.ndarray) -> np.ndarray:
        """Compute mean row sum of (I - rho*W)^{-1} W for each posterior draw.

        Always uses the N×N cross-sectional weights matrix (never the
        (N·T)×(N·T) Kronecker product), because spatial effects are defined
        in terms of the cross-sectional spatial multiplier.

        For row-standardized W this equals ``1/(1 - rho)`` (same as
        ``_batch_mean_row_sum``) because row sums of M@W = row sums of M
        when W is row-standardized.

        Isolates differ here.  Since ``M W = (M - I)/rho``, a connected row of
        ``M@W`` still sums to ``1/(1 - rho)`` but an isolate row sums to
        ``(1 - 1)/rho = 0`` — so the mean is ``(n - m)/(n(1 - rho))``, *not* the
        same as :meth:`_batch_mean_row_sum`.  Still O(1) per draw.

        For non-row-standardized W the eigenvalue decomposition is used:
        ``mean_row_sum_MW = (1/n) * ones' V diag(omega/(1-rho*omega)) V^{-1} ones``.

        Parameters
        ----------
        rho_draws : np.ndarray, shape (G,)
            Spatial autoregressive parameter draws.

        Returns
        -------
        np.ndarray, shape (G,)
            Mean row sum of M@W for each draw.
        """
        if self._is_row_std:
            m = self._n_isolates
            if m == 0:
                return 1.0 / (1.0 - rho_draws)
            n = int(self._W_sparse.shape[0])
            return (n - m) / (n * (1.0 - rho_draws))

        # Eigenvalue-based computation using shared eigendecomposition cache.
        decomp = self._W_eigendecomposition
        if decomp is None:
            raise ValueError("No spatial weights matrix available.")
        c = self._eig_inv_ones  # complex128, (n,)
        eigs = decomp[0]  # complex128, (n,)
        V_col_sums = decomp[1].sum(axis=0)  # complex128, (n,)
        from ...diagnostics.spatial_effects import _chunked_eig_means

        return _chunked_eig_means(
            rho_draws,
            eigs,
            weights=eigs * V_col_sums * c,
        )

    # ------------------------------------------------------------------
    # LM diagnostics
    # ------------------------------------------------------------------

    @staticmethod
    def _run_lm_diagnostics(model, tests: list[tuple]) -> pd.DataFrame:
        """Execute a registry of LM tests and return a tidy DataFrame.

        Shared helper used by :meth:`spatial_diagnostics` on every model
        family (cross-section, panel, flow).
        """
        from ...diagnostics.lmtests import BayesianLMTestResult

        rows: dict[str, dict] = {}
        raw_results: dict[str, BayesianLMTestResult] = {}

        for test_fn, label in tests:
            try:
                result = test_fn(model)
                rows[label] = {
                    "statistic": result.mean,
                    "median": result.median,
                    "df": result.df,
                    "p_value": result.bayes_pvalue,
                    "ci_lower": result.credible_interval[0],
                    "ci_upper": result.credible_interval[1],
                }
                raw_results[label] = result
            except (ValueError, np.linalg.LinAlgError) as exc:
                rows[label] = {
                    "statistic": np.nan,
                    "median": np.nan,
                    "df": np.nan,
                    "p_value": np.nan,
                    "ci_lower": np.nan,
                    "ci_upper": np.nan,
                    "error": str(exc),
                }

        df = pd.DataFrame.from_dict(rows, orient="index")
        df.index.name = "test"

        idata = model._idata
        n_draws = int(idata.posterior.sizes.get("draw", 0))
        n_chains = int(idata.posterior.sizes.get("chain", 1))
        df.attrs["model_type"] = model.__class__.__name__
        df.attrs["n_draws"] = n_draws * n_chains
        df.attrs["_raw_results"] = raw_results

        return df

    def spatial_diagnostics(self) -> pd.DataFrame:
        """Run Bayesian LM specification tests and return a summary table.

        Looks up the diagnostic suite registered for this model class
        and calls each test function on this fitted model, collecting the
        results into a tidy DataFrame.  The set of tests depends on the
        model type — for example, an OLS model runs LM-Lag, LM-Error,
        LM-SDM-Joint, and LM-SLX-Error-Joint, while an SAR model runs
        LM-Error, LM-WX, and Robust-LM-WX.  Panel models run the
        ``Panel-``-prefixed analogues (e.g. Panel-LM-Lag).

        Requires the model to have been fit (``.fit()`` called) and a
        spatial weights matrix ``W`` to have been supplied at construction
        time.

        Returns
        -------
        pandas.DataFrame
            DataFrame indexed by test name with columns:

            ==============  =====================================================
            Column          Description
            ==============  =====================================================
            statistic       Posterior mean of the LM statistic
            median          Posterior median of the LM statistic
            df              Degrees of freedom for the :math:`\\chi^2` reference
            p_value         Bayesian p-value: ``1 - chi2.cdf(mean, df)``
            ci_lower        Lower bound of 95% credible interval (2.5%)
            ci_upper        Upper bound of 95% credible interval (97.5%)
            ==============  =====================================================

            The DataFrame has ``attrs["model_type"]`` (class name) and
            ``attrs["n_draws"]`` (total posterior draws) metadata.

        Raises
        ------
        RuntimeError
            If the model has not been fit yet.
        ValueError
            If no spatial weights matrix ``W`` was supplied.

        See Also
        --------
        spatial_diagnostics_decision : Model-selection decision based on
            the test results.
        spatial_effects : Posterior inference for direct/indirect/total
            impacts.

        Examples
        --------
        >>> ols = OLS(formula="price ~ income + crime", data=df, W=w)
        >>> ols.fit()
        >>> ols.spatial_diagnostics()
                         statistic  median  df  p_value  ci_lower  ci_upper
        LM-Lag                3.21    2.98   1    0.073      0.12      8.54
        LM-Error              5.67    5.34   1    0.017      0.34     12.10
        LM-SDM-Joint          7.89    7.12   4    0.096      1.23     18.32
        LM-SLX-Error-Joint    6.45    5.98   4    0.168      0.89     15.67
        """
        self._require_fit()
        self._require_W()

        from ...diagnostics.lmtests.registry import get_diagnostic_suite

        suite = get_diagnostic_suite(self)
        if suite is None:
            raise ValueError(
                f"No diagnostic suite registered for {type(self).__name__}. "
                f"Register one in neighbayes.diagnostics.lmtests.registry."
            )
        return self._run_lm_diagnostics(self, suite.tests)

    def spatial_diagnostics_decision(
        self, alpha: float = 0.05, format: str = "graphviz"
    ) -> Any:
        """Return a model-selection decision from Bayesian LM test results.

        Implements the decision tree from :cite:t:`koley2024UseNot`
        (the Bayesian analogue of the classical ``stge_kb`` procedure
        in :cite:t:`anselin1996SimpleDiagnostic`).  Panel models use the
        ``Panel-``-prefixed test analogues and the panel decision specs,
        following :cite:t:`elhorst2014SpatialEconometrics`.  The decision
        logic depends on the current model type and the pattern of
        significant tests:

        **From OLS** (6-test decision tree):

        1. If only LM-Lag is significant → SAR.
        2. If only LM-Error is significant → SEM.
        3. If both are significant → use the Anselin–Florax / Koley–Bera
           robust pair: Robust-LM-Lag → SAR, Robust-LM-Error → SEM,
           both → SARAR. If neither robust test is significant, fall
           back to the lower raw p-value.
        4. If neither naive test is significant → OLS.

        **From SAR** (3-test decision tree):

        - LM-Error significant → SARAR; LM-WX significant → SDM;
          Robust-LM-WX significant → SDM.

        **From SEM** (2-test decision tree):

        - LM-Lag significant → SARAR; LM-WX significant → SDEM.

        **From SLX** (4-test decision tree):

        - Robust-LM-Lag-SDM significant → SDM;
          Robust-LM-Error-SDEM significant → SDEM;
          both → MANSAR; neither → SLX.

        **From SDM**: LM-Error-SDM significant → MANSAR; else SDM.

        **From SDEM**: LM-Lag-SDEM significant → MANSAR; else SDEM.

        Parameters
        ----------
        alpha : float, default 0.05
            Significance level for the Bayesian p-values.
        format : {"graphviz", "ascii", "model"}, default "graphviz"
            Output format. ``"model"`` returns the recommended-model name
            string. ``"ascii"`` returns an indented box-drawing rendering
            of the full decision tree with the chosen path highlighted.
            ``"graphviz"`` returns a :class:`graphviz.Digraph` object that
            renders inline in Jupyter; if the optional ``graphviz`` package
            is not installed a :class:`UserWarning` is issued and the
            ASCII rendering is returned instead.

        Returns
        -------
        str or graphviz.Digraph
            Recommended model name when ``format="model"``, an ASCII tree
            string when ``format="ascii"``, or a ``graphviz.Digraph`` when
            ``format="graphviz"`` (with ASCII fallback on missing dep).

        See Also
        --------
        spatial_diagnostics : Compute the Bayesian LM test statistics.

        References
        ----------
        :cite:t:`koley2024UseNot`, :cite:t:`anselin1996SimpleDiagnostic`,
        :cite:t:`elhorst2014SpatialEconometrics`
        """
        from ...diagnostics import _decision_trees as _dt

        diag = self.spatial_diagnostics()
        model_type = self.__class__.__name__

        def _sig(test_name: str) -> bool:
            if test_name not in diag.index:
                return False
            pval = diag.loc[test_name, "p_value"]
            return not np.isnan(pval) and pval < alpha

        def _p_le(test_a: str, test_b: str) -> bool:
            return diag.loc[test_a, "p_value"] <= diag.loc[test_b, "p_value"]

        panel = self._panel_diagnostics
        key_prefix = "panel_" if panel else ""
        test_prefix = "Panel-" if panel else ""
        predicate_lookup = {
            key_prefix + key: partial(_p_le, test_prefix + a, test_prefix + b)
            for key, (a, b) in _DECISION_PREDICATE_PAIRS.items()
        }

        spec = _dt.get_panel_spec(model_type) if panel else _dt.get_spec(model_type)
        decision, path = _dt.evaluate(
            spec,
            sig_lookup=_sig,
            predicate_lookup=predicate_lookup,
        )

        # Build p-value lookup for renderers (only test rows present).
        p_values: dict[str, float] = {}
        for test_name in diag.index:
            pv = diag.loc[test_name, "p_value"]
            if not np.isnan(pv):
                p_values[str(test_name)] = float(pv)

        return _dt.render(
            spec,
            path,
            decision,
            p_values=p_values,
            alpha=alpha,
            fmt=format,
            title=f"{model_type} decision tree (alpha={alpha})",
        )
