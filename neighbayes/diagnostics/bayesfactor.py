"""Bayes factor comparison of Bayesian spatial models.

Provides functions for estimating marginal likelihoods (via bridge sampling
or BIC approximation) and computing pairwise Bayes factors between
competing models, following the approach of :cite:t:`meng1996SimulatingRatios`,
:cite:t:`gronau2020bridgesampling`, and :cite:t:`wagenmakers2007PracticalSolution`.

The bridge sampling implementation follows the R ``bridgesampling`` package
(:cite:t:`gronau2020bridgesampling`) and supports:

- Automatic compilation of the log-posterior from fitted model objects
  via ``compile_log_posterior()``.
- Effective sample size (ESS) weighting in the iterative scheme.
- Two-phase convergence with geometric-mean restart.
- Monte Carlo standard error (MCSE) following :cite:t:`micaletto2025MCSE`.

Functions
---------
compile_log_posterior
    Compile a PyMC model's log-posterior into a callable for bridge sampling.
bic_to_bf
    Convert BIC values to Bayes factors.
bayes_factor_compare_models
    Compute all pairwise Bayes factors for a set of models.
post_prob
    Compute posterior model probabilities from marginal likelihoods.
"""

from __future__ import annotations

import warnings
from typing import Callable, Optional, Union

import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal

from ._native_log_posterior import native_log_posterior

_BAYES_FACTOR_METHODS = {}


def _register_bayes_factor_method(name):
    """Register a marginal-likelihood estimation method by name."""

    def decorator(fn):
        _BAYES_FACTOR_METHODS[name] = fn
        return fn

    return decorator


# ---------------------------------------------------------------------------
# Utility: compile log-posterior from a PyMC model
# ---------------------------------------------------------------------------


def compile_log_posterior(pymc_model) -> tuple[Callable, list[str], dict, Callable]:
    """Compile a PyMC model's log-posterior into a callable for bridge sampling.

    This is the Python equivalent of the R ``bridgesampling`` package's
    automatic ``log_prob`` extraction from Stan models.  The compiled
    function evaluates the **full** unnormalized log-posterior
    :math:`\\log p(y \\mid \\theta) p(\\theta)` at any parameter vector,
    including all ``pm.Potential`` terms (e.g., Jacobians like
    :math:`\\log|I - \\rho W|`).

    Parameters
    ----------
    pymc_model : pymc.Model
        A compiled PyMC model object (e.g., ``model._pymc_model`` after
        calling ``fit()``).

    Returns
    -------
    log_posterior_fn : callable
        A function ``f(theta_flat) -> float`` that evaluates the log
        unnormalized posterior at a flat parameter vector **in the
        unconstrained (transformed) space**.
    param_names : list of str
        Names of the free parameters in the **unconstrained** space
        (e.g., ``"sigma_log__"`` instead of ``"sigma"``), matching the
        order in ``theta_flat``.
    param_info : dict
        Dict with keys ``"shapes"`` and ``"sizes"`` mapping parameter
        names to their shapes and total scalar sizes.
    constrained_to_unconstrained : callable
        A function ``f(constrained_samples_dict) -> ndarray`` that
        converts a dict of constrained posterior samples (as stored in
        ``idata.posterior``, keyed by original parameter names like
        ``"sigma"``) into a flat array in the unconstrained space, in
        the same order as ``param_names``.  This is needed because
        PyMC's ``compile_logp()`` operates in the unconstrained space,
        but posterior samples are stored in the constrained space.

    Examples
    --------
    After fitting a model::

        model = SAR(y=y, X=X, W=W)
        model.fit(draws=2000, idata_kwargs={"log_likelihood": True})

        logp_fn, names, info, to_unconstrained = compile_log_posterior(model.pymc_model)
        # Convert constrained posterior samples to unconstrained space
        theta = to_unconstrained(model.inference_data.posterior)
        print(logp_fn(theta[0]))  # evaluate at first sample
    """
    logp_fn = pymc_model.compile_logp()
    free_vars = pymc_model.free_RVs

    # PyMC compiles logp in the *transformed* space (e.g. sigma -> sigma_log__).
    # model.value_vars gives the actual parameter names the compiled function
    # expects (in the transformed/unconstrained space).
    value_vars = pymc_model.value_vars
    input_names = [v.name for v in value_vars]

    # Build mapping from constrained (free_RV) names to unconstrained (value_var) names
    # and record the transform object for each parameter.
    constrained_to_unconstrained_name = {}
    transform_types = {}  # constrained_name -> transform object or None
    for fv in free_vars:
        vv = pymc_model.rvs_to_values[fv]
        t = pymc_model.rvs_to_transforms.get(fv)
        constrained_to_unconstrained_name[fv.name] = vv.name
        transform_types[fv.name] = t

    # Build shape/size info using the model's initial point, which reliably
    # resolves shapes for all parameters including those created with dims=
    # (e.g., beta with dims="coefficient").  This is more robust than
    # fv.eval().shape, which can silently fail for dims-based parameters
    # and fall back to shape=() (treating vectors as scalars).
    initial_pt = pymc_model.initial_point()
    param_shapes = {}
    param_sizes = {}
    for vv in value_vars:
        name = vv.name
        if name in initial_pt:
            shape = tuple(np.asarray(initial_pt[name]).shape)
        else:
            # Fallback: try to find the corresponding free_RV
            shape = ()
            for fv in free_vars:
                if name == fv.name or name.startswith(fv.name + "_"):
                    try:
                        shape = tuple(fv.eval().shape)
                    except Exception:
                        shape = ()
                    break
        param_shapes[name] = shape
        param_sizes[name] = int(np.prod(shape)) if shape else 1

    def log_posterior(theta_flat: np.ndarray) -> float:
        """Evaluate log p(y|θ)p(θ) at a flat parameter vector.

        ``theta_flat`` must be in the *transformed* (unconstrained) space,
        matching the order of ``input_names``.
        """
        theta_dict = {}
        offset = 0
        for name in input_names:
            size = param_sizes[name]
            raw = theta_flat[offset : offset + size]
            offset += size
            if param_shapes[name]:
                theta_dict[name] = raw.reshape(param_shapes[name])
            else:
                # Scalar parameter — must pass a Python float, not a 0-d or 1-d array
                theta_dict[name] = float(raw[0]) if raw.ndim else float(raw)
        return float(logp_fn(theta_dict))

    def constrained_to_unconstrained(posterior) -> np.ndarray:
        """Convert constrained posterior samples to unconstrained space.

        Parameters
        ----------
        posterior : xarray.Dataset or dict-like
            Posterior samples in the constrained space, as stored in
            ``idata.posterior``.  Must contain all free parameter names
            (e.g., ``"sigma"``, ``"rho"``, ``"beta"``).

        Returns
        -------
        ndarray, shape (n_draws, n_params)
            Flattened posterior samples in the unconstrained (transformed)
            space, in the same order as ``input_names``.
        """
        n_chains = posterior.sizes["chain"]
        n_draws = posterior.sizes["draw"]
        n_total = n_chains * n_draws

        blocks = []
        for fv in free_vars:
            constrained_name = fv.name
            transform = transform_types[constrained_name]

            # Get constrained samples from posterior
            arr = posterior[constrained_name].values  # (chain, draw, ...)
            if arr.ndim == 2:
                arr = arr.reshape(n_total, 1)
            else:
                arr = arr.reshape(n_total, -1)

            # Apply transform: constrained -> unconstrained.
            # Delegate to the transform object's own ``forward()`` — the same
            # method PyMC applies when building initial points — passing the
            # RV's distribution inputs, exactly PyMC's calling convention.
            # ``None`` means the variable was never transformed (identity).
            if transform is not None:
                arr = np.asarray(transform.forward(arr, *fv.owner.inputs).eval())

            blocks.append(arr)

        return np.hstack(blocks)

    param_info = {"shapes": param_shapes, "sizes": param_sizes}
    return log_posterior, input_names, param_info, constrained_to_unconstrained


# ---------------------------------------------------------------------------
# Utility: numerically stable helpers
# ---------------------------------------------------------------------------


def _compute_ess(samples: np.ndarray) -> float:
    """Compute median effective sample size across parameters.

    Parameters
    ----------
    samples : ndarray, shape (n_draws, n_params)
        Flattened posterior samples.

    Returns
    -------
    float
        Median ESS across all parameters.
    """
    n_params = samples.shape[1]
    ess_vals = []
    for p in range(n_params):
        x = samples[:, p]
        n = len(x)
        x_centered = x - np.mean(x)
        var_x = np.var(x, ddof=0)
        if var_x == 0:
            ess_vals.append(n)
            continue
        max_lag = min(n // 2, 1000)
        # Biased autocovariance via FFT (O(n log n)) instead of the dense
        # O(n^2) ``np.correlate``.  Zero-padding to >= 2n-1 makes the
        # circular FFT product yield the *linear* autocorrelation, so the
        # result matches ``np.correlate(x_c, x_c, "full")[n-1:]`` to
        # floating-point precision — the ESS estimate is unchanged.
        nfft = int(2 ** np.ceil(np.log2(2 * n - 1)))
        f = np.fft.rfft(x_centered, n=nfft)
        acf = np.fft.irfft(f * np.conj(f), n=nfft)[:max_lag]
        acf = acf / (var_x * n)
        tau = 1.0
        for lag in range(1, max_lag):
            if acf[lag] < 0:
                break
            tau += 2 * acf[lag]
        ess_vals.append(max(1.0, n / tau))
    return float(np.median(ess_vals))


def _nearest_pos_def(A: np.ndarray) -> np.ndarray:
    """Find the nearest positive-definite matrix.

    Uses eigenvalue clamping — similar to R's ``Matrix::nearPD()`` used
    by the bridgesampling package.

    Parameters
    ----------
    A : ndarray, shape (n, n)
        Symmetric matrix.

    Returns
    -------
    ndarray
        Nearest positive-definite matrix.
    """
    from numpy.linalg import eigh

    B = (A + A.T) / 2
    eigvals, eigvecs = eigh(B)
    # Floor eigenvalues so the result is *strictly* positive-definite under
    # scipy's check.  ``scipy.stats._multivariate._PSD`` treats eigenvalues
    # below ``max(M.shape) * eps * max(|eigvals|)`` as zero; we floor an
    # order of magnitude above that threshold so the Cholesky path in
    # ``multivariate_normal.logpdf`` succeeds reliably.
    max_eig = float(eigvals.max())
    scipy_threshold = B.shape[0] * np.finfo(B.dtype).eps * max(max_eig, 1.0)
    floor = max(1e3 * scipy_threshold, 1e-12)
    eigvals = np.maximum(eigvals, floor)
    return eigvecs @ np.diag(eigvals) @ eigvecs.T


# ---------------------------------------------------------------------------
# Iterative bridge sampling scheme
# ---------------------------------------------------------------------------


def _run_iterative_scheme(
    q11: np.ndarray,
    q12: np.ndarray,
    q21: np.ndarray,
    q22: np.ndarray,
    r0: float = 1.0,
    tol: float = 1e-10,
    maxiter: int = 1000,
    criterion: str = "r",
    neff: Optional[float] = None,
    use_neff: bool = True,
) -> dict:
    """Run the iterative bridge sampling scheme (:cite:p:`meng1996SimulatingRatios`, eq. 4.1).

    Uses the "optimal" bridge function with numerically stable log-sum-exp
    arithmetic.  Follows the R ``bridgesampling`` package's
    ``.run.iterative.scheme`` function.

    Parameters
    ----------
    q11 : ndarray, shape (N1,)
        Log unnormalized posterior evaluated at posterior samples (second half).
    q12 : ndarray, shape (N1,)
        Log proposal density evaluated at posterior samples (second half).
    q21 : ndarray, shape (N2,)
        Log unnormalized posterior evaluated at proposal samples.
    q22 : ndarray, shape (N2,)
        Log proposal density evaluated at proposal samples.
    r0 : float, default 1.0
        Starting value for the ratio r.
    tol : float
        Convergence tolerance.
    maxiter : int
        Maximum iterations.
    criterion : str
        Convergence criterion: ``"r"`` for relative change in r,
        ``"logml"`` for relative change in logml.
    neff : float or None
        Effective sample size.  If None, uses N1.
    use_neff : bool
        Whether to use ESS in the bridge function weights.

    Returns
    -------
    dict with keys: logml, niter, r_vals, mcse_logml, converged
    """
    N1 = len(q11)
    N2 = len(q21)

    # l1 = log(unnormalized posterior / proposal) for posterior samples
    # l2 = log(unnormalized posterior / proposal) for proposal samples
    l1 = q11 - q12
    l2 = q21 - q22

    # Stabilise: subtract median before exponentiating (like R's lstar)
    lstar = np.median(l1)

    # Effective sample size for weighting
    if use_neff and neff is not None:
        n_eff = neff
    else:
        n_eff = N1

    s1 = n_eff / (n_eff + N2)
    s2 = N2 / (n_eff + N2)

    r = r0
    r_vals = [r]
    logml = np.log(r) + lstar
    logml_vals = [logml]
    criterion_val = 1.0 + tol
    i = 0

    while i < maxiter and criterion_val > tol:
        rold = r
        logmlold = logml

        log_s1 = np.log(s1)
        log_s2_r = np.log(s2) + np.log(r) if r > 0 else -np.inf

        l2_shifted = l2 - lstar
        l1_shifted = l1 - lstar

        # Optimal bridge function, vectorized.  logsumexp over a
        # two-element array is exactly ``np.logaddexp``; broadcasting the
        # scalar ``log_s2_r`` collapses the per-sample Python loops (run
        # every one of up to ``maxiter`` iterations) into a single
        # vectorized call each — identical values, O(N) numpy instead of
        # O(N) Python-level logsumexp allocations.
        log_num = l2_shifted - np.logaddexp(log_s1 + l2_shifted, log_s2_r)
        log_den = l1_shifted - np.logaddexp(log_s1 + l1_shifted, log_s2_r)

        # Check for infinities
        if np.any(np.isinf(log_num)) or np.any(np.isinf(log_den)):
            warnings.warn(
                "Infinite value in iterative scheme, returning NA. "
                "Try rerunning with more samples.",
                stacklevel=3,
            )
            return dict(
                logml=np.nan, niter=i, r_vals=r_vals, mcse_logml=np.nan, converged=False
            )

        num_vals = np.exp(log_num)
        den_vals = np.exp(log_den)

        mean_num = np.mean(num_vals)
        mean_den = np.mean(den_vals)

        if mean_den == 0:
            warnings.warn(
                "Denominator mean is zero in iterative scheme, returning NA. "
                "Try rerunning with more samples.",
                stacklevel=3,
            )
            return dict(
                logml=np.nan, niter=i, r_vals=r_vals, mcse_logml=np.nan, converged=False
            )

        r = mean_num / mean_den
        logml = np.log(r) + lstar

        r_vals.append(r)
        logml_vals.append(logml)

        # Convergence criterion
        if criterion == "r":
            criterion_val = abs((r - rold) / r) if r != 0 else abs(r)
        else:  # "logml"
            criterion_val = (
                abs((logml - logmlold) / logml) if logml != 0 else abs(logml)
            )

        i += 1

    # Compute MCSE (Micaletto & Vehtari, 2025)
    var_num = np.var(num_vals)
    var_den = np.var(den_vals)

    if use_neff and neff is not None:
        var_den_adj = var_den * N1 / neff
    else:
        var_den_adj = var_den

    var_r = (mean_num**2 / mean_den**2) * (
        var_num / mean_num**2 + var_den_adj / mean_den**2
    )
    var_r = var_r / N2

    if r > 0 and var_r / r**2 > -1:
        var_logml = np.log(1 + var_r / r**2)
    else:
        var_logml = np.nan

    mcse_logml = np.sqrt(var_logml) if not np.isnan(var_logml) else np.nan

    converged = criterion_val <= tol

    return dict(
        logml=logml, niter=i, r_vals=r_vals, mcse_logml=mcse_logml, converged=converged
    )


@_register_bayes_factor_method("bridge")
def _bridge_logml(
    idata,
    log_posterior,
    constrained_to_unconstrained=None,
    return_diagnostics=False,
    maxiter=1000,
    tol1=1e-10,
    tol2=1e-4,
    random_state=None,
    use_neff=True,
    repetitions=1,
):
    """Estimate log marginal likelihood via bridge sampling (:cite:p:`meng1996SimulatingRatios`).

    Follows the R ``bridgesampling`` package (:cite:t:`gronau2020bridgesampling`) with the "normal" proposal method.

    Parameters
    ----------
    idata : arviz.InferenceData
        Posterior samples.
    log_posterior : callable
        A function ``f(theta_flat) -> float`` that evaluates the
        **unnormalized** log-posterior :math:`\\log p(y \\mid \\theta)
        p(\\theta)` at a flat parameter vector (in the transformed/
        unconstrained space).  This is required for accurate bridge
        sampling and matches the R package's ``log_posterior`` argument.

        Can be obtained via :func:`compile_log_posterior`::

            logp_fn, names, info, to_unconstrained = compile_log_posterior(model.pymc_model)
    constrained_to_unconstrained : callable or None
        A function ``f(posterior_dataset) -> ndarray`` that converts
        constrained posterior samples (as stored in ``idata.posterior``)
        to unconstrained (transformed) space.  If None, posterior samples
        are used as-is (assumed already in unconstrained space).

        This is **required** when using PyMC models with constrained
        parameters (e.g., ``HalfNormal`` for sigma, ``Uniform`` for rho),
        because ``log_posterior`` operates in the unconstrained space but
        posterior samples are stored in the constrained space.

        Obtain from :func:`compile_log_posterior`::

            _, _, _, to_unconstrained = compile_log_posterior(model.pymc_model)
    return_diagnostics : bool, default False
        If True, return a dict with diagnostics.
    maxiter : int, default 1000
        Maximum number of bridge iterations per phase.
    tol1 : float, default 1e-10
        Convergence tolerance for the first phase (criterion = "r").
    tol2 : float, default 1e-4
        Convergence tolerance for the second phase (criterion = "logml"),
        used if the first phase does not converge.
    random_state : int or None
        Random seed for reproducibility.
    use_neff : bool, default True
        If True, use effective sample size (ESS) instead of nominal sample
        size in the bridge function weights, matching the R package's
        ``use_neff`` argument.
    repetitions : int, default 1
        Number of times to repeat the bridge sampling procedure with
        different proposal draws.  If > 1, returns the median logml.

    Returns
    -------
    logml : float
        Estimated log marginal likelihood.  If ``repetitions > 1``, the
        median across repetitions.
    diagnostics : dict
        Diagnostics if ``return_diagnostics=True``.
    """
    rng = np.random.default_rng(random_state)

    # --- 1. Extract posterior samples (n_draws x n_params) ---
    posterior = idata.posterior

    if constrained_to_unconstrained is not None:
        # Convert constrained posterior samples to unconstrained space
        # (e.g., sigma -> sigma_log__, rho -> rho_interval__)
        samples = constrained_to_unconstrained(posterior)
    else:
        # Fallback: use posterior samples as-is (assumed already unconstrained)
        n_chains = posterior.sizes["chain"]
        n_draws = posterior.sizes["draw"]
        n_posterior = n_chains * n_draws
        sample_blocks = []
        for var in posterior.data_vars:
            arr = posterior[var].values
            if arr.ndim >= 3:
                sample_blocks.append(arr.reshape(n_posterior, -1))
            elif arr.ndim == 2:
                sample_blocks.append(arr.reshape(n_posterior, 1))
            else:
                sample_blocks.append(arr.reshape(-1, 1))
        samples = np.hstack(sample_blocks)
    n_samples, n_dim = samples.shape
    if n_samples < 2 or n_dim < 1 or samples.size == 0:
        var_names_used = list(posterior.data_vars)
        raise ValueError(
            f"Bridge sampling failed: Not enough usable posterior variables or samples.\n"
            f"Variables used: {var_names_used}\n"
            f"Shape: {samples.shape}.\n"
            "Check your model output and ensure all sampled parameters have "
            "matching, non-degenerate shapes."
        )

    # --- 2. Split samples: first half for fitting proposal, second for iteration ---
    N1 = n_samples // 2
    N2 = n_samples - N1
    samples_4_fit = samples[:N1]
    samples_4_iter = samples[N1:]

    # --- 3. Fit proposal (multivariate normal) ---
    mu = np.mean(samples_4_fit, axis=0)
    cov = np.cov(samples_4_fit, rowvar=False)
    if n_dim == 1:
        cov = np.atleast_2d(cov)
    cov = _nearest_pos_def(cov)

    # --- 4. Compute ESS for the iterative samples ---
    if use_neff:
        try:
            neff = _compute_ess(samples_4_iter)
        except Exception:
            warnings.warn(
                "ESS computation failed; using nominal sample size.",
                stacklevel=2,
            )
            neff = None
    else:
        neff = None

    # --- 5. Evaluate log densities for posterior samples ---
    q11 = np.array([log_posterior(samples_4_iter[j]) for j in range(N1)])

    # q12: log proposal density at posterior samples (second half).
    # ``cov`` has already been floored to strict positive-definiteness by
    # ``_nearest_pos_def`` above; ``allow_singular=True`` is a belt-and-braces
    # guard against scipy's PSD check rejecting matrices whose smallest
    # eigenvalue lands at its rejection threshold (scipy always uses SVD
    # internally regardless of this flag, so there is no performance cost).
    q12 = multivariate_normal.logpdf(
        samples_4_iter, mean=mu, cov=cov, allow_singular=True
    )

    # --- 6. Run bridge sampling with repetitions ---
    logml_reps = []
    niter_reps = []
    mcse_reps = []
    converged_reps = []

    for rep in range(repetitions):
        # Draw proposal samples
        prop_samples = rng.multivariate_normal(mu, cov, size=N2)

        # q22: log proposal density at proposal samples
        q22 = multivariate_normal.logpdf(
            prop_samples, mean=mu, cov=cov, allow_singular=True
        )

        # q21: log unnormalized posterior at proposal samples
        q21 = np.array([log_posterior(prop_samples[j]) for j in range(N2)])

        # Handle -Inf / NA in log-posterior evaluations
        n_inf_q11 = np.sum(np.isinf(q11))
        n_inf_q21 = np.sum(np.isinf(q21))
        if n_inf_q11 > 0:
            warnings.warn(
                f"{n_inf_q11} of {len(q11)} log_posterior evaluations on "
                "posterior draws produced -Inf/Inf.",
                stacklevel=2,
            )
        if n_inf_q21 > 0:
            warnings.warn(
                f"{n_inf_q21} of {len(q21)} log_posterior evaluations on "
                "proposal draws produced -Inf/Inf.",
                stacklevel=2,
            )
        q11_safe = np.where(np.isinf(q11), -1e300, q11)
        q21_safe = np.where(np.isinf(q21), -1e300, q21)

        # Phase 1: criterion = "r"
        result = _run_iterative_scheme(
            q11=q11_safe,
            q12=q12,
            q21=q21_safe,
            q22=q22,
            r0=1.0,
            tol=tol1,
            maxiter=maxiter,
            criterion="r",
            neff=neff,
            use_neff=use_neff,
        )

        # Phase 2: if not converged, restart with geometric mean
        if np.isnan(result["logml"]) and len(result["r_vals"]) >= 2:
            warnings.warn(
                "logml could not be estimated within maxiter; rerunning "
                "with adjusted starting value. Estimate may be more variable.",
                stacklevel=2,
            )
            lr = len(result["r_vals"])
            r0_2 = np.sqrt(result["r_vals"][lr - 1] * result["r_vals"][lr - 2])
            result2 = _run_iterative_scheme(
                q11=q11_safe,
                q12=q12,
                q21=q21_safe,
                q22=q22,
                r0=r0_2,
                tol=tol2,
                maxiter=maxiter,
                criterion="logml",
                neff=neff,
                use_neff=use_neff,
            )
            result2["niter"] = maxiter + result2["niter"]
            result = result2

        logml_reps.append(result["logml"])
        niter_reps.append(result["niter"])
        mcse_reps.append(result.get("mcse_logml", np.nan))
        converged_reps.append(result.get("converged", False))

    # --- 7. Aggregate across repetitions ---
    if repetitions == 1:
        logml = logml_reps[0]
        niter = niter_reps[0]
        mcse = mcse_reps[0]
        converged = converged_reps[0]
    else:
        logml = float(np.median(logml_reps))
        niter = int(np.max(niter_reps))
        mcse = float(np.median(mcse_reps))
        converged = all(converged_reps)

    diagnostics = dict(
        logml=logml,
        iterations=niter,
        mcse_logml=mcse,
        N1=N1,
        N2=N2,
        neff=neff,
        tol1=tol1,
        tol2=tol2,
        converged=converged,
        method="bridge",
        repetitions=repetitions,
    )
    if repetitions > 1:
        diagnostics["logml_reps"] = logml_reps
        diagnostics["logml_range"] = (min(logml_reps), max(logml_reps))

    if return_diagnostics:
        return diagnostics
    return logml


@_register_bayes_factor_method("bic")
def _bic_logml(idata, return_diagnostics=False, model=None):
    """Estimate log marginal likelihood via the BIC approximation.

    Following bayestestR's ``bic_to_bf()`` (:cite:p:`wagenmakers2007PracticalSolution`):

    .. math::
        BF_{10} = \\exp((BIC_0 - BIC_1) / 2)

    which is equivalent to:

    .. math::
        \\log(ML) \\approx -BIC / 2

    since :math:`BIC = k \\log(n) - 2 \\hat{\\ell}_{\\max}`, so
    :math:`\\log(ML) \\approx -BIC/2 + (k \\log(n))/2` and the
    :math:`k \\log(n)/2` terms cancel when computing BF ratios.

    .. note::
       The maximized log-likelihood :math:`\\hat{\\ell}_{\\max}` is
       approximated by the maximum of the posterior log-likelihoods
       (``np.max``), not the posterior mean.  Using the mean would
       underestimate the log-likelihood and distort model rankings
       because the gap between mean and max varies across models.

    Parameters
    ----------
    idata : arviz.InferenceData
        Must include ``log_likelihood`` group.
    return_diagnostics : bool, default False
        If True, return a dict with diagnostics.
    model : object, optional
        A fitted model object with a ``_y`` attribute (the dependent
        variable array).  Used as a fallback to determine the number
        of observations when ``observed_data`` and ``sample_stats``
        groups are absent from the InferenceData.

    Returns
    -------
    logml : float
        Estimated log marginal likelihood (via BIC approximation).
    diagnostics : dict
        Diagnostics if return_diagnostics=True.
    """
    if not hasattr(idata, "log_likelihood"):
        raise ValueError(
            "InferenceData must have a log_likelihood group for BIC approximation. "
            "It is stored only on request: refit with "
            "fit(..., idata_kwargs={'log_likelihood': True})."
        )

    log_like_group = idata.log_likelihood
    # Sum log-likelihood over observations for each draw
    per_draw_ll = [v.values.sum(axis=-1) for v in log_like_group.data_vars.values()]
    log_like_total = np.sum(per_draw_ll, axis=0)
    if log_like_total.ndim > 1:
        log_like_total = log_like_total.reshape(-1)
    # ML log-likelihood: maximum of posterior log-likelihoods.
    # This approximates the MLE (the true BIC uses the maximized
    # log-likelihood, not the posterior mean).  Using the mean
    # underestimates the log-likelihood and distorts model rankings
    # because the gap between mean and max varies across models.
    loglik_ml = np.max(log_like_total)

    # Number of parameters from posterior
    posterior = idata.posterior
    n_params = sum(
        np.prod(v.values.shape[2:]) if v.values.ndim >= 3 else 1
        for v in posterior.data_vars.values()
    )

    # Number of observations — try multiple sources in order of reliability
    n_obs = None
    if hasattr(idata, "sample_stats") and hasattr(idata.sample_stats, "n_data_points"):
        n_obs = int(idata.sample_stats.n_data_points.values)
    elif hasattr(idata, "observed_data"):
        # Infer from observed_data
        obs_vars = list(idata.observed_data.data_vars)
        n_obs = int(np.prod(idata.observed_data[obs_vars[0]].shape))
    elif hasattr(idata, "log_likelihood"):
        # Infer from log_likelihood group: the last dimension is the
        # observation dimension (shape = chain, draw, obs)
        ll_vars = list(idata.log_likelihood.data_vars)
        if ll_vars:
            n_obs = int(idata.log_likelihood[ll_vars[0]].shape[-1])
    if n_obs is None and model is not None and hasattr(model, "_y"):
        # Fallback: use the model's dependent variable length
        n_obs = len(model._y)
    if n_obs is None:
        raise ValueError(
            "Cannot determine number of observations. Provide an InferenceData "
            "with observed_data, sample_stats.n_data_points, or log_likelihood "
            "groups, or pass a fitted model object with a _y attribute."
        )

    # BIC = k * log(n) - 2 * loglik_ML
    bic = n_params * np.log(n_obs) - 2 * loglik_ml
    # log(ML) ≈ -BIC/2  (up to a constant that cancels in BF ratios)
    logml = -bic / 2

    if return_diagnostics:
        diagnostics = dict(
            logml=logml,
            bic=bic,
            loglik_ml=loglik_ml,
            n_params=n_params,
            n_obs=n_obs,
            method="bic",
        )
        return diagnostics
    return logml


def bic_to_bf(bic_values, denominator=None, log=False):
    """Convert BIC values to Bayes factors via the BIC-approximation method.

    Following bayestestR (:cite:p:`wagenmakers2007PracticalSolution`):

    .. math::
        BF_{10} = \\exp((BIC_0 - BIC_1) / 2)

    Parameters
    ----------
    bic_values : array-like
        Vector of BIC values, one per model.
    denominator : float or None
        The BIC value of the denominator (reference) model. If None, uses
        the first model as the denominator.
    log : bool, default False
        If True, return log(BF) instead of BF.

    Returns
    -------
    numpy.ndarray
        Bayes factors (or log Bayes factors if ``log=True``) relative to the
        denominator model.

    Examples
    --------
    >>> bic1, bic2, bic3 = 100, 95, 110
    >>> bic_to_bf([bic1, bic2, bic3], denominator=bic1)  # BF against model 1
    array([1.        , 12.18249396,  0.082085  ])
    """
    bic_values = np.asarray(bic_values, dtype=float)
    if denominator is None:
        denominator = bic_values[0]
    delta = (denominator - bic_values) / 2
    if log:
        return delta
    return np.exp(delta)


def bayes_factor_compare_models(
    models,
    model_labels=None,
    method: str = "bridge",
    prior_note: str = None,
    return_diagnostics: bool = False,
    log: bool = False,
    **kwargs,
) -> Union[pd.DataFrame, tuple[pd.DataFrame, dict]]:
    """Compute all pairwise Bayes factors for a set of Bayesian models.

    Following bayestestR's ``bayesfactor_models()`` (:cite:p:`makowski2019bayestestR`),
    this function estimates marginal likelihoods for
    each model and then computes Bayes factors as ratios of marginal
    likelihoods:

    .. math::
        BF_{ij} = ML_i / ML_j = \\exp(\\log ML_i - \\log ML_j)

    The transitivity property of Bayes factors allows all pairwise comparisons
    from a single set of marginal likelihood estimates:

    .. math::
        BF_{AB} = BF_{AC} / BF_{BC} = ML_A / ML_B

    Parameters
    ----------
    models : list or dict
        Fitted model objects or InferenceData objects to compare.

        - **Fitted model objects** (recommended): Each object must have
          ``inference_data`` and ``pymc_model`` attributes (e.g., a
          fitted :class:`SAR`, :class:`SEM`, etc.).  For bridge sampling,
          the log-posterior is automatically compiled from the PyMC model.
        - **Dict of {str: model_object}**: Keys are used as model labels
          (unless ``model_labels`` is also provided), matching the
          convention of :func:`arviz.compare`.
        - **List of InferenceData**: For ``method='bic'``, InferenceData
          objects can be passed directly.  For ``method='bridge'``,
          fitted model objects are required so the log-posterior can be
          compiled automatically.

    model_labels : list of str, optional
        Labels for each model (for DataFrame index/columns). If None,
        models are labeled by index (or by dict keys when ``models`` is
        a dict).
    method : str, default 'bridge'
        Marginal likelihood estimation method. Supported:

        - ``'bridge'``: Bridge sampling (:cite:p:`meng1996SimulatingRatios`). Uses the
          iterative scheme with a multivariate normal proposal, ESS
          weighting, two-phase convergence, and MCSE diagnostics —
          following the R ``bridgesampling`` package (:cite:p:`gronau2020bridgesampling`).
          **Requires fitted model objects** so the log-posterior can be
          compiled automatically.
        - ``'bic'``: BIC approximation (:cite:p:`wagenmakers2007PracticalSolution`).
          Computes :math:`\\log(ML) \\approx -BIC/2`.  Works with either
          fitted model objects or InferenceData.

    prior_note : str, optional
        Optional string describing the priors used (for reporting).
    return_diagnostics : bool, default False
        If True, also return a dict of diagnostics for each model.
    log : bool, default False
        If True, return **log** Bayes factors (i.e. ``logml_i - logml_j``)
        instead of Bayes factors.  Useful when the marginal-likelihood
        differences are large enough to overflow ``exp`` (\\|Δlogml\\| > ~709).
    **kwargs
        Additional keyword arguments forwarded to the marginal-likelihood
        estimator.  For ``method='bridge'``, the following are accepted:

        - ``maxiter`` (int, default 1000): Maximum bridge iterations per
          phase.
        - ``tol1`` (float, default 1e-10): Convergence tolerance for
          Phase 1 (criterion = ``"r"``).
        - ``tol2`` (float, default 1e-4): Convergence tolerance for
          Phase 2 (criterion = ``"logml"``), used if Phase 1 does not
          converge.
        - ``use_neff`` (bool, default True): Use effective sample size
          (ESS) instead of nominal sample size in the bridge-function
          weights, matching the R package's ``use_neff`` argument.
        - ``repetitions`` (int, default 1): Number of times to repeat
          bridge sampling with different proposal draws.  If > 1, the
          median logml across repetitions is returned.
        - ``random_state`` (int or None): Random seed for reproducibility.

    Returns
    -------
    bayes_factors : pandas.DataFrame
        DataFrame of Bayes factors (BF[i, j] = BF for model i vs model j;
        >1 favors i over j). The diagonal is 1 (each model compared to itself).
    diagnostics : dict
        If ``return_diagnostics=True``, a dict of diagnostics for each
        model (keyed by model label) is returned as a second element.
        Each value is a dict containing:

        - ``logml``: Estimated log marginal likelihood.
        - ``iterations``: Number of bridge iterations.
        - ``mcse_logml``: Monte Carlo standard error of logml
          (:cite:p:`micaletto2025MCSE`).
        - ``converged``: Whether the iterative scheme converged.
        - ``neff``: Effective sample size used for weighting (if
          ``use_neff=True``).
        - ``N1``, ``N2``: Number of posterior and proposal samples.
        - ``method``: ``"bridge"`` or ``"bic"``.
        - ``repetitions``: Number of repetitions (if > 1, also includes
          ``logml_reps`` with per-repetition estimates).

    Notes
    -----
    **Bridge sampling algorithm.**  The ``'bridge'`` method implements the
    iterative bridge sampling estimator of :cite:t:`meng1996SimulatingRatios`, eq. 4.1, with
    the "optimal" bridge function.  The procedure is:

    1. Split posterior samples in half: first half fits a multivariate
       normal proposal, second half drives the iterative scheme.
    2. Evaluate the log unnormalized posterior at both posterior and
       proposal samples (via ``log_posterior``).
    3. Run the iterative scheme in two phases:

       - **Phase 1**: criterion = ``"r"`` (relative change in the ratio),
         tolerance ``tol1``.
       - **Phase 2** (if Phase 1 fails): restart with the geometric mean
         of the last two ratios, criterion = ``"logml"``, tolerance ``tol2``.

    4. Compute MCSE following Micaletto & Vehtari (2025).

    **ESS weighting.**  By default (``use_neff=True``), the bridge function
    uses :math:`s_1 = n_{\\mathrm{eff}} / (n_{\\mathrm{eff}} + N_2)` instead of
    :math:`s_1 = N_1 / (N_1 + N_2)`, matching the R ``bridgesampling``
    package.  This down-weights autocorrelated samples and generally
    improves accuracy.

    **Interpreting Bayes factors.**  The Bayes factor quantifies the
    relative evidence for two competing models given the observed data.

    - BF > 1 favors the row model; BF < 1 favors the column model.
    - Conventional thresholds (Jeffreys, 1961; Kass & Raftery, 1995):
      1–3 (anecdotal), 3–10 (moderate), 10–30 (strong), 30–100 (very strong),
      >100 (extreme).

    **Sample size.**  For bridge sampling, at least 40,000 posterior
    samples are recommended for precise estimates (:cite:p:`gronau2020bridgesampling`).
    A warning is emitted when fewer samples are detected.

    **BIC vs. bridge sampling.**  The ``'bic'`` method approximates
    :math:`\\log(ML) \\approx \\hat{\\ell}_{\\max} - \\frac{k}{2}\\log(n)`,
    which assumes unit-information priors (priors containing as much
    information as a single observation).  When the actual priors are
    wider — as is typical for spatial models — bridge sampling will
    penalize complex models more heavily than BIC.  This is not a bug:
    it reflects the fact that wide priors on unnecessary parameters
    reduce the marginal likelihood (Bayesian Occam's razor).  For
    models with ``Normal(0, 100)`` priors on WX coefficients, the
    bridge-sampling penalty per coefficient can be 5–10× larger than
    the BIC penalty.  When the two methods disagree, bridge sampling
    is generally more trustworthy because it accounts for the actual
    prior specification.

    **Posterior model probabilities.**  After computing Bayes factors,
    posterior model probabilities can be obtained via :func:`post_prob`::

        from neighbayes.diagnostics import post_prob
        probs = post_prob(logml_list, model_names=model_labels)

    Examples
    --------
    Compare fitted spatial models using bridge sampling (recommended)::

        from neighbayes.diagnostics import bayes_factor_compare_models

        # Pass fitted model objects directly — log-posterior is compiled
        # automatically from each model's PyMC model.
        bf = bayes_factor_compare_models(
            {"SAR": sar, "SEM": sem, "SDM": sdm},
            method="bridge",
        )

    Quick comparison using the BIC approximation (no log-posterior needed)::

        bf = bayes_factor_compare_models(
            {"SAR": sar, "SEM": sem},
            method="bic",
        )

    With diagnostics and repetitions::

        bf, diag = bayes_factor_compare_models(
            {"SAR": sar, "SEM": sem},
            method="bridge",
            return_diagnostics=True,
            repetitions=3,
            random_seed=42,
        )

    """
    # Backward-compatibility: older call sites may pass ``metric="bic"``.
    # If both are supplied, they must agree.
    metric = kwargs.pop("metric", None)
    if metric is not None:
        if method != "bridge" and metric != method:
            raise ValueError(
                "Received both `method` and `metric` with different values: "
                f"method={method!r}, metric={metric!r}. Use only one."
            )
        method = metric

    # --- Unpack models into idata_list, log_posterior_list, and labels ---
    if isinstance(models, dict):
        items = list(models.items())
        if model_labels is None:
            model_labels = [k for k, v in items]
        model_objects = [v for k, v in items]
    elif isinstance(models, (list, tuple)):
        model_objects = list(models)
        if model_labels is None:
            model_labels = [f"model_{i}" for i in range(len(model_objects))]
    else:
        raise TypeError(
            f"models must be a list, tuple, or dict, got {type(models).__name__}"
        )

    if len(model_labels) != len(model_objects):
        raise ValueError("model_labels must match length of models")

    # Resolve each entry: either a fitted model object or InferenceData
    idata_list = []
    log_posterior_list = []
    constrained_to_unconstrained_list = []
    for i, obj in enumerate(model_objects):
        # Check if this is a fitted model object (has inference_data and
        # pymc_model).  Probe the *class*, not the instance: ``pymc_model`` is a
        # lazily-building property, so ``hasattr(obj, "pymc_model")`` would
        # construct and compile a PyTensor graph just to answer "does this
        # attribute exist?" — defeating the native log-posterior path below for
        # every Gibbs-fitted model.  Looking the property up on the type returns
        # the descriptor without invoking its getter.
        _has_pymc_model = hasattr(type(obj), "pymc_model") or "pymc_model" in getattr(
            obj, "__dict__", {}
        )
        if hasattr(obj, "inference_data") and _has_pymc_model:
            idata = obj.inference_data
            if idata is None:
                raise ValueError(
                    f"Model at index {i} ('{model_labels[i]}') has no "
                    "inference_data. Call .fit() before comparing models."
                )
            idata_list.append(idata)
            if method == "bridge":
                # Prefer the closed-form density when the model family has one.
                # A Gibbs fit has no PyTensor graph, so the PyMC route below
                # would build and compile one purely as a density oracle — work
                # the native path skips entirely.  ``None`` means "unsupported
                # family", not "failed", so fall through rather than raise.
                native = native_log_posterior(obj)
                if native is not None:
                    logp_fn, _, _, to_unconstrained = native
                else:
                    pymc_model = obj.pymc_model
                    if pymc_model is None:
                        raise ValueError(
                            f"Model at index {i} ('{model_labels[i]}') has no "
                            "pymc_model. Call .fit() before comparing models."
                        )
                    logp_fn, _, _, to_unconstrained = compile_log_posterior(pymc_model)
                log_posterior_list.append(logp_fn)
                constrained_to_unconstrained_list.append(to_unconstrained)
            else:
                log_posterior_list.append(None)
                constrained_to_unconstrained_list.append(None)
        elif hasattr(obj, "posterior"):
            # Bare InferenceData object
            idata_list.append(obj)
            if method == "bridge":
                raise ValueError(
                    f"Entry at index {i} ('{model_labels[i]}') is an "
                    "InferenceData object, but bridge sampling requires a "
                    "fitted model object with a pymc_model attribute so the "
                    "log-posterior can be compiled automatically.  Pass the "
                    "fitted model object (e.g., sar, sem) instead of its "
                    "inference_data."
                )
            log_posterior_list.append(None)
            constrained_to_unconstrained_list.append(None)
        else:
            raise TypeError(
                f"Entry at index {i} ('{model_labels[i]}') must be a fitted "
                "model object (with .inference_data and .pymc_model attributes) "
                f"or an InferenceData object, got {type(obj).__name__}"
            )

    # Warn about sample size for bridge sampling (Gronau et al., 2017)
    if method == "bridge":
        for label, idata in zip(model_labels, idata_list):
            total = idata.posterior.sizes["chain"] * idata.posterior.sizes["draw"]
            if total < 40_000:
                warnings.warn(
                    f"Bridge sampling with {total} posterior samples for "
                    f"'{label}' may yield imprecise marginal-likelihood "
                    "estimates. A conservative rule of thumb is 40,000+ "
                    "samples (Gronau, Singmann, & Wagenmakers, 2017).",
                    stacklevel=2,
                )
                break

    if method not in _BAYES_FACTOR_METHODS:
        raise ValueError(
            f"Unknown method: {method}. Available: {list(_BAYES_FACTOR_METHODS.keys())}"
        )

    logml_fn = _BAYES_FACTOR_METHODS[method]
    logmls = []
    diagnostics = {}
    for i, (label, idata, log_post) in enumerate(
        zip(model_labels, idata_list, log_posterior_list)
    ):
        # Build kwargs for the method
        call_kwargs = dict(kwargs)
        if method == "bridge" and log_post is not None:
            call_kwargs["log_posterior"] = log_post
            call_kwargs["constrained_to_unconstrained"] = (
                constrained_to_unconstrained_list[i]
            )
        elif method == "bic":
            # Strip bridge-specific kwargs that _bic_logml doesn't accept
            for _key in (
                "log_posterior",
                "constrained_to_unconstrained",
                "maxiter",
                "tol1",
                "tol2",
                "use_neff",
                "repetitions",
                "random_state",
            ):
                call_kwargs.pop(_key, None)
            # Pass the model object so _bic_logml can use len(model._y)
            # as a fallback for n_obs
            model_obj = model_objects[i]
            if hasattr(model_obj, "_y"):
                call_kwargs["model"] = model_obj

        try:
            if return_diagnostics:
                diag = logml_fn(idata, return_diagnostics=True, **call_kwargs)
                if isinstance(diag, dict) and "logml" in diag:
                    logmls.append(diag["logml"])
                    diagnostics[label] = diag
                else:
                    logmls.append(diag)
                    diagnostics[label] = None
            else:
                logmls.append(logml_fn(idata, **call_kwargs))
        except TypeError:
            # Bridge sampling requires ``log_posterior``; retrying without
            # kwargs would mask the original error and fail with a less useful
            # message. Re-raise to preserve context.
            if method == "bridge":
                raise
            # Fallback for methods that don't support extra kwargs
            if return_diagnostics:
                result = logml_fn(idata, return_diagnostics=True)
                if isinstance(result, dict) and "logml" in result:
                    logmls.append(result["logml"])
                    diagnostics[label] = result
                else:
                    logmls.append(result)
                    diagnostics[label] = None
            else:
                logmls.append(logml_fn(idata))
                diagnostics[label] = None

    n = len(logmls)
    log_bf_mat = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                log_bf_mat[i, j] = logmls[i] - logmls[j]

    if log:
        df = pd.DataFrame(log_bf_mat, index=model_labels, columns=model_labels)
    else:
        with np.errstate(over="ignore"):
            bf_mat = np.exp(log_bf_mat)
        if np.any(np.isinf(bf_mat)):
            warnings.warn(
                "Some Bayes factors overflowed to inf because the log "
                "marginal-likelihood differences exceed ~709. Pass "
                "``log=True`` to return log Bayes factors instead.",
                stacklevel=2,
            )
        df = pd.DataFrame(bf_mat, index=model_labels, columns=model_labels)
    if prior_note:
        warnings.warn(f"Bayes factors computed with priors: {prior_note}", stacklevel=2)
    if return_diagnostics:
        return df, diagnostics
    return df


# ---------------------------------------------------------------------------
# Posterior model probabilities
# ---------------------------------------------------------------------------


def post_prob(
    logml_list,
    model_names=None,
    prior_prob=None,
) -> pd.Series:
    """Compute posterior model probabilities from marginal likelihoods.

    Following the R ``bridgesampling`` package's ``post_prob()`` function.

    Parameters
    ----------
    logml_list : array-like
        Log marginal likelihoods, one per model.
    model_names : list of str, optional
        Labels for each model.  If None, models are labeled by index.
    prior_prob : array-like or None
        Prior model probabilities.  If None, uniform priors are used
        (all models equally likely a priori).

    Returns
    -------
    pandas.Series
        Posterior model probabilities (sum to 1), indexed by model names.

    Examples
    --------
    >>> post_prob([-20.8, -18.0, -19.0], model_names=["H0", "H1", "H2"])
    H0    0.05...
    H1    0.72...
    H2    0.22...
    dtype: float64
    """
    logmls = np.asarray(logml_list, dtype=float)
    n = len(logmls)

    if model_names is None:
        model_names = [f"model_{i}" for i in range(n)]

    if prior_prob is None:
        prior_prob = np.ones(n) / n
    else:
        prior_prob = np.asarray(prior_prob, dtype=float)
        if len(prior_prob) != n:
            raise ValueError("prior_prob must match length of logml_list")
        if np.any(prior_prob < 0):
            raise ValueError("prior_prob must be non-negative")
        prior_prob = prior_prob / prior_prob.sum()

    # Numerically stable computation: subtract max before exponentiating
    log_unnorm = logmls + np.log(prior_prob)
    log_unnorm_shifted = log_unnorm - np.max(log_unnorm)
    probs = np.exp(log_unnorm_shifted)
    probs = probs / probs.sum()

    return pd.Series(probs, index=model_names, name="post_prob")


# ---------------------------------------------------------------------------
# Object-oriented wrapper
# ---------------------------------------------------------------------------


class ModelComparison:
    """Compare a collection of fitted Bayesian models.

    Object-oriented wrapper around :func:`bayes_factor_compare_models`,
    :func:`post_prob`, and :func:`arviz.compare`.  Marginal likelihoods are
    computed once on first access and cached, so repeated calls to
    :meth:`bayes_factors`, :meth:`posterior_probabilities`, etc. reuse the
    same estimates.

    Parameters
    ----------
    models : list, tuple, or dict
        Fitted model objects (each with ``inference_data`` and, for bridge
        sampling, ``pymc_model``).  A ``dict`` is interpreted as
        ``{label: model}``; otherwise pass ``labels`` explicitly.
    labels : list of str, optional
        Display labels.  Required when ``models`` is a list/tuple and no
        ``__name__``-style label is available; auto-generated when omitted.
    method : {"bridge", "bic"}, default "bridge"
        Marginal-likelihood estimator.  See
        :func:`bayes_factor_compare_models`.
    **kwargs
        Forwarded to :func:`bayes_factor_compare_models` (e.g.
        ``repetitions``, ``maxiter`` for bridge sampling).

    Examples
    --------
    >>> cmp = ModelComparison({"OLS": ols, "SAR": sar, "SDM": sdm})
    >>> cmp.log_marginal_likelihoods
    OLS   -123.4
    SAR   -120.1
    SDM   -119.8
    Name: log_ml, dtype: float64
    >>> cmp.bayes_factors        # pairwise BF matrix
    >>> cmp.posterior_probabilities()
    >>> cmp.loo_comparison()     # arviz.compare on LOO
    """

    def __init__(
        self,
        models,
        labels: Optional[list[str]] = None,
        method: str = "bridge",
        **kwargs,
    ):
        if isinstance(models, dict):
            if labels is not None:
                raise ValueError(
                    "Pass `labels` either via dict keys or the `labels` "
                    "argument, not both."
                )
            self._labels = list(models.keys())
            self._models = list(models.values())
        elif isinstance(models, (list, tuple)):
            self._models = list(models)
            if labels is None:
                self._labels = [f"model_{i}" for i in range(len(self._models))]
            else:
                if len(labels) != len(self._models):
                    raise ValueError(
                        "labels must match length of models "
                        f"({len(labels)} vs {len(self._models)})."
                    )
                self._labels = list(labels)
        else:
            raise TypeError(
                f"models must be a list, tuple, or dict; got {type(models).__name__}."
            )

        self.method = method
        self._kwargs = dict(kwargs)
        self._bf_df: Optional[pd.DataFrame] = None
        self._diagnostics: Optional[dict] = None

    # -- core: compute once, cache -----------------------------------------

    def _compute(self) -> None:
        if self._bf_df is not None:
            return
        bf_df, diag = bayes_factor_compare_models(
            self._models,
            model_labels=self._labels,
            method=self.method,
            return_diagnostics=True,
            **self._kwargs,
        )
        self._bf_df = bf_df
        self._diagnostics = diag

    # -- properties --------------------------------------------------------

    @property
    def labels(self) -> list[str]:
        """Model labels in the order supplied."""
        return list(self._labels)

    @property
    def log_marginal_likelihoods(self) -> pd.Series:
        """Estimated :math:`\\log p(y \\mid M_k)` for each model."""
        self._compute()
        # bayes_factor_compare_models returns a square BF matrix; we
        # recover logmls from the first row of log(BF) plus an arbitrary
        # zero baseline by reading them out of the diagnostics dict when
        # available, otherwise from the BF matrix directly.
        if self._diagnostics and all(
            isinstance(self._diagnostics.get(lbl), dict)
            and "logml" in self._diagnostics[lbl]
            for lbl in self._labels
        ):
            logmls = [self._diagnostics[lbl]["logml"] for lbl in self._labels]
        else:
            # Reconstruct relative log-ML from the BF matrix.  Anchor on
            # the first model: log(BF[i, 0]) = logml[i] - logml[0].
            log_bf = np.log(self._bf_df.to_numpy())
            logmls = log_bf[:, 0]
        return pd.Series(logmls, index=self._labels, name="log_ml")

    @property
    def bayes_factors(self) -> pd.DataFrame:
        """Pairwise Bayes factors :math:`BF_{ij} = ML_i / ML_j`."""
        self._compute()
        return self._bf_df.copy()

    @property
    def diagnostics(self) -> Optional[dict]:
        """Per-model diagnostics from the marginal-likelihood estimator."""
        self._compute()
        return self._diagnostics

    # -- methods -----------------------------------------------------------

    def posterior_probabilities(
        self, prior_prob: Optional[np.ndarray] = None
    ) -> pd.Series:
        """Posterior model probabilities given the estimated marginal
        likelihoods.

        Parameters
        ----------
        prior_prob : array-like, optional
            Prior model probabilities.  Defaults to uniform.

        Returns
        -------
        pandas.Series
            Posterior probabilities, summing to 1, indexed by label.
        """
        logmls = self.log_marginal_likelihoods.to_numpy()
        return post_prob(logmls, model_names=self._labels, prior_prob=prior_prob)

    def loo_comparison(self, **kwargs) -> pd.DataFrame:
        """Compare models by LOO via :func:`arviz.compare`.

        Each model must have a ``log_likelihood`` group in its
        ``inference_data``.  Extra keyword arguments are forwarded to
        :func:`arviz.compare`.

        Returns
        -------
        pandas.DataFrame
            Output of ``arviz.compare`` (one row per model).
        """
        import arviz as az

        idata_dict = {}
        for lbl, m in zip(self._labels, self._models):
            idata = getattr(m, "inference_data", m)
            if idata is None:
                raise ValueError(
                    f"Model '{lbl}' has no inference_data; call .fit() first."
                )
            idata_dict[lbl] = idata
        return az.compare(idata_dict, **kwargs)

    # -- repr --------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"ModelComparison(n_models={len(self._labels)}, "
            f"method='{self.method}', labels={self._labels!r})"
        )
