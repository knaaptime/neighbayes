r"""Reduced-form SAR-logit with Pólya–Gamma Gibbs sampler.

.. math::

    y_i \sim \mathrm{Bernoulli}(\mathrm{logit}^{-1}(\eta_i)), \quad
    \eta = (I - \rho W)^{-1} X\beta

This is the canonical spatial binary model: the spatial lag enters the
*linear predictor* as a deterministic mean-propagator (there is **no**
latent noise field, so σ does not appear).  The ``|I − ρW|`` Jacobian
cancels when β is marginalized out, making the ρ conditional linear and
Krylov-accelerable.  The Pólya–Gamma augmentation yields fully conjugate
Gibbs updates for β and ρ (via a collapsed slice sampler).

Both backends fit the same model: ``backend="jax"`` (the default via
``"auto"``) runs the chains in parallel threads, one ``jax.jit`` program each;
``"numpy"`` uses the CHOLMOD factorization path.  There is no SAR logit with
random effects: in a cross-section their sd and ρ are barely identified from
binary outcomes.

``latent=True`` selects the latent SAR, which propagates the logit's own
disturbance, ``y = 1{(I - ρW)⁻¹(Xβ + ε) > 0}`` with ε standard logistic
(:mod:`._latent_binary`).

Use this model when:
- The response is binary (0/1).
- You need spatial autocorrelation in the log-odds.
- NUTS is slow or unreliable for the spatial parameter ρ.

References
----------
Polson, N. G., Scott, J. G., & Windle, J. (2013). Bayesian inference
for logistic models using Pólya–Gamma latent variables.
*Journal of the American Statistical Association*, 108(504), 1339–1349.
"""

from __future__ import annotations

from .._mixins._designs import BinaryDesigns
from ..base import SpatialModel
from ..priors import SARLogitPriors
from ._latent_binary import LatentLogitMixin
from ._reduced_binary import ReducedBinaryMixin


class SARLogit(BinaryDesigns, ReducedBinaryMixin, LatentLogitMixin, SpatialModel):
    """Bayesian SAR logit: the reduced form, or the latent SAR with ``latent=True``.

    Parameters
    ----------
    formula : str, optional
        Wilkinson-style formula, e.g. ``"y ~ x1 + x2"``. Requires
        ``data``. An intercept is included by default; suppress with
        ``"y ~ x - 1"``.
    data : pandas.DataFrame or geopandas.GeoDataFrame, optional
        Data source for formula mode.
    y : array-like, optional
        Binary dependent variable of shape ``(n,)``. Required in matrix
        mode. Must contain only 0 and 1.
    X : array-like or pandas.DataFrame, optional
        Design matrix. Required in matrix mode.
    W : libpysal.graph.Graph or scipy.sparse matrix
        Spatial weights of shape ``(n, n)``.
    priors : dict or SARLogitPriors, optional
        Override default priors. Supported keys:

        - ``rho_lower`` (float, default -0.999): Lower bound of the
          Uniform prior on :math:`\\rho`.
        - ``rho_upper`` (float, default 0.999): Upper bound of the
          Uniform prior on :math:`\\rho`.
        - ``beta_mu``, ``beta_sigma`` (float or array, default Gelman et al.
          2008): Normal prior on :math:`\\beta`, on the logit scale (intercept at ``logit(mean(y))``, scale 2.5; slopes ``2.5 / sd(x_j)``).

    logdet_method : {"eigenvalue", "cheb", "aaa", "spline", "stochastic"}, optional
        How ``log|I - rho*W|`` is computed over rho: summed over the
        eigenvalues of ``W``; interpolated (Chebyshev, AAA rational, or cubic
        spline) through values from a sparse factorization, exact to the
        interpolant's tolerance; or expanded stochastically with no
        factorization, the one method with sampling error.  ``None`` (default)
        selects from the size and symmetry of ``W``: ``"eigenvalue"`` for
        ``n <= 500``; ``"aaa"`` up to ``n = 60000``
        (``NEIGHBAYES_LOGDET_FACTOR_MAX_N``) unless the factorization's fill-in
        would dominate; ``"stochastic"`` beyond.
    logdet_decomp : {"auto", "chol", "lu"}, default "auto"
        The factorization an interpolating method uses: sparse Cholesky
        (undirected ``W``; scikit-sparse CHOLMOD) or sparse LU (KLU or UMFPACK,
        chosen by the sparsity pattern).  ``"auto"`` takes ``"chol"`` where the
        method has it and ``W`` is undirected, else ``"lu"``.  Ignored by
        ``"eigenvalue"`` and ``"stochastic"``; ``"spline"`` on ``"chol"``
        raises.
    robust : bool, default False
        Not supported. Raises ``NotImplementedError`` if True.
    latent : bool, default False
        Fit the latent SAR instead of the reduced form:

        .. math::
            y_i = \\mathbb 1\\{y^*_i > 0\\}, \\qquad
            (I - \\rho W) y^* = X\\beta + \\varepsilon, \\qquad
            \\varepsilon_i \\sim \\mathrm{Logistic}(0, 1),

        whose ρ = 0 member is the logit.  It is sampled by an exact Gibbs sweep
        over the latent utilities and the Kolmogorov scales of the logistic
        (Albert & Chib 1993; Holmes & Held 2006), which needs ``augax``: on
        JAX by default, the utilities by colour class, or with
        ``backend="numpy"`` one at a time.  Its impacts are on the latent
        index, ``scale="logodds"``.

    Notes
    -----
    The reduced form parameterizes the log-odds as the deterministic
    mean-propagator ``eta = (I - rho * W)^{-1} X @ beta`` (no latent noise
    field), and augments the logistic likelihood with Pólya–Gamma auxiliary
    variables to obtain fully conjugate Gibbs updates for β and a
    β-marginalized collapsed slice update for ρ (the ``|I − ρW|`` Jacobian
    cancels, so ρ is Krylov-accelerable).

    The sampler bypasses PyMC's NUTS entirely. It produces an
    ``xarray.DataTree`` object compatible with all downstream
    diagnostics (``spatial_diagnostics()``, ``impacts()``,
    ``summary()``).

    With the Gibbs sampler, ``fit()`` does **not** accept ``target_accept``,
    a NUTS-only argument; passing it raises ``TypeError``.

    The PG shape parameter is
    always h = 1 (one trial per observation).
    """

    _spatial_params: tuple[str, ...] = ("rho",)
    _lag_terms: tuple[str, ...] = ("Wy",)
    _jacobian_param: str | None = "rho"
    _gibbs_class: str | None = None  # Gibbs-only, no NUTS
    _model_type: str = "sar_logit"
    _gibbs_key: tuple[str, str] | None = ("binary", "cross_section")
    _priors_cls = SARLogitPriors
    _DESIGNS = {
        "static": {},
        "latent": {"_gibbs_key": ("binary_latent", "cross_section")},
    }
    _latent_process = "sar"

    def impacts(
        self,
        return_posterior_samples: bool = False,
        scale: str = "logodds",
        method: str = "auto",
    ):
        r"""Compute Bayesian inference for direct, indirect, and total impacts.

        Parameters
        ----------
        return_posterior_samples : bool, optional
            If ``True``, also return the posterior draws for each effect type.
        scale : {"logodds", "probability"}, default "logodds"
            Scale on which impacts are reported.

            ``"logodds"`` returns impacts on the linear-predictor (log-odds)
            scale :math:`\eta = (I - \rho W)^{-1} X\beta`. This is the
            cheap default; effects are linear in :math:`\beta` and do not
            require evaluating the link.

            ``"probability"`` returns response-scale impacts

            .. math::

                \partial p / \partial x_r =
                \operatorname{diag}(p(1-p))\,(I - \rho W)^{-1}\beta_r,

            which match the LeSage-Pace (2009) and PG-paper formulas. This
            is more expensive because it requires the diagonal of the
            spatial multiplier and the fitted probabilities for each draw.
        method : {"auto", "eigen", "sparse"}, default "auto"
            Only used when ``scale="probability"``. ``"eigen"`` uses the
            shared eigendecomposition cache (O(n²) per draw); ``"sparse"``
            uses one sparse LU per draw plus a Hutchinson diagonal
            estimator (O(nnz) per draw); ``"auto"`` picks sparse when
            :math:`n` exceeds :attr:`_PROBABILITY_EFFECTS_EIGEN_MAX_N`
            (default 2000).
        """
        return self._sar_binary_impacts(scale, return_posterior_samples, method)
