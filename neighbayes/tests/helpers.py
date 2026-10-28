"""Synthetic data generators and constants shared across the test suite.

Import from here rather than from conftest.py to avoid sys.path issues.
"""

from __future__ import annotations

import importlib.util

import arviz as az
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
import xarray as xr
from libpysal.graph import Graph

from neighbayes import dgp
from neighbayes._logdet._chebyshev import chebyshev_coeffs_dct1

# These helpers draw unit effects into dynamic panels (sigma_alpha), which the
# public simulators refuse, so they call the private dynamic simulators.
from neighbayes.dgp.panel_dynamic import (
    _simulate_ols_panel_dynamic,
    _simulate_sar_panel_dynamic,
    _simulate_sdem_panel_dynamic,
    _simulate_sdm_panel_dynamic,
    _simulate_sdm_panel_dynamic_restricted,
    _simulate_sem_panel_dynamic,
    _simulate_slx_panel_dynamic,
)

# ---------------------------------------------------------------------------
# Optional-dependency skip helpers
# ---------------------------------------------------------------------------

requires_nutpie = pytest.mark.skipif(
    importlib.util.find_spec("nutpie") is None,
    reason="nutpie is not installed",
)

# ---------------------------------------------------------------------------
# Sampling settings
# ---------------------------------------------------------------------------
# Cheap default: enough for parameter recovery within MC tolerance on the
# small synthetic problems used in the suite. Tests that are weakly identified
# (e.g. spatial probit with binary data, censored Tobit with heavy censoring)
# can opt into ``SAMPLE_KWARGS_HEAVY``.

SAMPLE_KWARGS: dict = dict(
    tune=500, draws=500, chains=2, random_seed=42, progressbar=False
)

SAMPLE_KWARGS_HEAVY: dict = dict(
    tune=1000, draws=1500, chains=4, random_seed=42, progressbar=False
)

# Panel dimensions
PANEL_N = 10  # cross-sectional units (larger N for reliable recovery)
PANEL_T = 10  # time periods


# ---------------------------------------------------------------------------
# Spatial weight helpers
# ---------------------------------------------------------------------------


def _as_sparse(W):
    """Pass a Graph or sparse matrix through; wrap a dense test fixture as CSR.

    The DGPs accept only a Graph or a sparse matrix, since W is never
    densified.  Test fixtures build small dense matrices for clarity.
    """
    if isinstance(W, Graph) or sp.issparse(W):
        return W
    return sp.csr_matrix(np.asarray(W, dtype=float))


def make_rook_W(side: int) -> np.ndarray:
    """Row-standardized rook-contiguity weights on a ``side x side`` grid."""
    n = side * side
    W = np.zeros((n, n))
    for r in range(side):
        for c in range(side):
            i = r * side + c
            if r > 0:
                W[i, (r - 1) * side + c] = 1
            if r < side - 1:
                W[i, (r + 1) * side + c] = 1
            if c > 0:
                W[i, r * side + (c - 1)] = 1
            if c < side - 1:
                W[i, r * side + (c + 1)] = 1
    row_sums = W.sum(axis=1, keepdims=True)
    return W / np.where(row_sums == 0, 1, row_sums)


def make_line_W(n: int) -> np.ndarray:
    """Row-standardized line-lattice weights for ``n`` units.

    Unit ``i`` is connected to immediate neighbors ``i-1`` and ``i+1``.
    """
    W = np.zeros((n, n))
    for i in range(n):
        if i > 0:
            W[i, i - 1] = 1.0
        if i < n - 1:
            W[i, i + 1] = 1.0
    row_sums = W.sum(axis=1, keepdims=True)
    return W / np.where(row_sums == 0, 1, row_sums)


def W_to_graph(W_dense: np.ndarray) -> Graph:
    """Convert a dense weight matrix to a libpysal Graph."""
    n = W_dense.shape[0]
    focal, neighbor, weight = [], [], []
    for i in range(n):
        for j in range(n):
            if W_dense[i, j] != 0:
                focal.append(i)
                neighbor.append(j)
                weight.append(W_dense[i, j])
    return Graph.from_arrays(
        np.array(focal),
        np.array(neighbor),
        np.array(weight, dtype=float),
    ).transform("r")


# ---------------------------------------------------------------------------
# Cross-sectional data generators
# ---------------------------------------------------------------------------


def make_sar_data(
    rng: np.random.Generator,
    W: np.ndarray,
    rho: float = 0.5,
    beta: np.ndarray | None = None,
    sigma: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate SAR data: y = (I - rho*W)^{-1}(X@beta + eps)."""
    out = dgp.simulate_sar(W=_as_sparse(W), rho=rho, beta=beta, sigma=sigma, rng=rng)
    return out["y"], out["X"]


def make_sem_data(
    rng: np.random.Generator,
    W: np.ndarray,
    lam: float = 0.5,
    beta: np.ndarray | None = None,
    sigma: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate SEM data: u = (I - lam*W)^{-1}*eps; y = X@beta + u."""
    out = dgp.simulate_sem(W=_as_sparse(W), lam=lam, beta=beta, sigma=sigma, rng=rng)
    return out["y"], out["X"]


def make_slx_data(
    rng: np.random.Generator,
    W: np.ndarray,
    beta1: np.ndarray | None = None,
    beta2: np.ndarray | None = None,
    sigma: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate SLX data: y = X@beta1 + W@X_noint@beta2 + eps."""
    out = dgp.simulate_slx(
        W=_as_sparse(W), beta1=beta1, beta2=beta2, sigma=sigma, rng=rng
    )
    return out["y"], out["X"]


def make_sdm_data(
    rng: np.random.Generator,
    W: np.ndarray,
    rho: float = 0.4,
    beta1: np.ndarray | None = None,
    beta2: np.ndarray | None = None,
    sigma: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate SDM data: y = (I-rho*W)^{-1}(X@beta1 + WX_noint@beta2 + eps)."""
    out = dgp.simulate_sdm(
        W=_as_sparse(W),
        rho=rho,
        beta1=beta1,
        beta2=beta2,
        sigma=sigma,
        rng=rng,
    )
    return out["y"], out["X"]


def make_sdem_data(
    rng: np.random.Generator,
    W: np.ndarray,
    lam: float = 0.4,
    beta1: np.ndarray | None = None,
    beta2: np.ndarray | None = None,
    sigma: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate SDEM data: y = X@beta1 + WX_noint@beta2 + (I-lam*W)^{-1}eps."""
    out = dgp.simulate_sdem(
        W=_as_sparse(W),
        lam=lam,
        beta1=beta1,
        beta2=beta2,
        sigma=sigma,
        rng=rng,
    )
    return out["y"], out["X"]


# ---------------------------------------------------------------------------
# Panel data generators  (time-first stacking: obs t*N+i → unit i)
# ---------------------------------------------------------------------------


def make_panel_ols_data(
    rng: np.random.Generator,
    W: np.ndarray,
    N: int,
    T: int,
    beta: np.ndarray | None = None,
    sigma: float = 1.0,
    sigma_alpha: float = 0.5,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Generate panel OLS data with unit random effects."""
    out = dgp.simulate_ols(
        N=N,
        T=T,
        beta=beta,
        sigma=sigma,
        W=_as_sparse(W),
        rng=rng,
        unit_effects="fixed",
        unit_effect_sd=sigma_alpha,
    )
    y, X = out["y"], out["X"]
    units = out["unit"]
    times = out["time"]
    df = pd.DataFrame({"y": y, "x1": X[:, 1], "unit": units, "time": times})
    return y, X, df


def make_panel_sar_data(
    rng: np.random.Generator,
    W: np.ndarray,
    N: int,
    T: int,
    rho: float = 0.4,
    beta: np.ndarray | None = None,
    sigma: float = 1.0,
    sigma_alpha: float = 0.5,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Generate SAR panel data with unit random effects."""
    out = dgp.simulate_sar(
        N=N,
        T=T,
        rho=rho,
        beta=beta,
        sigma=sigma,
        W=_as_sparse(W),
        rng=rng,
        unit_effects="fixed",
        unit_effect_sd=sigma_alpha,
    )
    y, X = out["y"], out["X"]
    units = out["unit"]
    times = out["time"]
    df = pd.DataFrame({"y": y, "x1": X[:, 1], "unit": units, "time": times})
    return y, X, df


def make_panel_sem_data(
    rng: np.random.Generator,
    W: np.ndarray,
    N: int,
    T: int,
    lam: float = 0.4,
    beta: np.ndarray | None = None,
    sigma: float = 1.0,
    sigma_alpha: float = 0.5,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Generate SEM panel data with unit random effects."""
    out = dgp.simulate_sem(
        N=N,
        T=T,
        lam=lam,
        beta=beta,
        sigma=sigma,
        W=_as_sparse(W),
        rng=rng,
        unit_effects="fixed",
        unit_effect_sd=sigma_alpha,
    )
    y, X = out["y"], out["X"]
    units = out["unit"]
    times = out["time"]
    df = pd.DataFrame({"y": y, "x1": X[:, 1], "unit": units, "time": times})
    return y, X, df


def make_panel_dlm_data(
    rng: np.random.Generator,
    W: np.ndarray,
    N: int,
    T: int,
    phi: float = 0.4,
    beta: np.ndarray | None = None,
    sigma: float = 1.0,
    sigma_alpha: float = 0.5,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Generate dynamic non-spatial panel data with unit effects."""
    out = _simulate_ols_panel_dynamic(
        N=N,
        T=T,
        phi=phi,
        beta=beta,
        sigma=sigma,
        sigma_alpha=sigma_alpha,
        W=_as_sparse(W),
        rng=rng,
    )
    y, X = out["y"], out["X"]
    units = out["unit"]
    times = out["time"]
    df = pd.DataFrame({"y": y, "x1": X[:, 1], "unit": units, "time": times})
    return y, X, df


def make_panel_sdmr_data(
    rng: np.random.Generator,
    W: np.ndarray,
    N: int,
    T: int,
    rho: float = 0.3,
    phi: float = 0.4,
    beta: np.ndarray | None = None,
    sigma: float = 1.0,
    sigma_alpha: float = 0.5,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Generate dynamic restricted SDM panel data with unit effects."""
    out = _simulate_sdm_panel_dynamic_restricted(
        N=N,
        T=T,
        rho=rho,
        phi=phi,
        beta=beta,
        sigma=sigma,
        sigma_alpha=sigma_alpha,
        W=_as_sparse(W),
        rng=rng,
    )
    y, X = out["y"], out["X"]
    units = out["unit"]
    times = out["time"]
    df = pd.DataFrame({"y": y, "x1": X[:, 1], "unit": units, "time": times})
    return y, X, df


def make_panel_sdmu_data(
    rng: np.random.Generator,
    W: np.ndarray,
    N: int,
    T: int,
    rho: float = 0.3,
    phi: float = 0.4,
    theta: float = -0.1,
    beta: np.ndarray | None = None,
    sigma: float = 1.0,
    sigma_alpha: float = 0.5,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Generate dynamic unrestricted SDM panel data with unit effects."""
    out = _simulate_sdm_panel_dynamic(
        N=N,
        T=T,
        rho=rho,
        phi=phi,
        theta=theta,
        beta=beta,
        sigma=sigma,
        sigma_alpha=sigma_alpha,
        W=_as_sparse(W),
        rng=rng,
    )
    y, X = out["y"], out["X"]
    units = out["unit"]
    times = out["time"]
    df = pd.DataFrame({"y": y, "x1": X[:, 1], "unit": units, "time": times})
    return y, X, df


# ---------------------------------------------------------------------------
# Tobit data generators
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Spatial logit data generators
# ---------------------------------------------------------------------------


def make_sar_logit_data(
    rng: np.random.Generator,
    W: np.ndarray,
    rho: float = 0.35,
    beta: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate reduced-form SAR-logit binary data (noise-free).

    The reduced-form model is deterministic:
    ``eta = (I - rho W)^{-1} X beta`` (no latent noise field),
    ``y ~ Bernoulli(logit^{-1}(eta))``.

    """
    import scipy.sparse as sp

    W_sparse = sp.csr_matrix(W)
    n_obs = W.shape[0]
    if beta is None:
        beta = np.array([0.3, 1.0], dtype=float)
    beta = np.asarray(beta, dtype=float)
    X = np.column_stack([np.ones(n_obs), rng.standard_normal((n_obs, len(beta) - 1))])
    eta = sp.linalg.spsolve(sp.eye(n_obs, format="csr") - rho * W_sparse, X @ beta)
    probs = 1.0 / (1.0 + np.exp(-eta))
    y = (rng.uniform(size=n_obs) < probs).astype(float)
    return y, X


def make_sar_tobit_data(
    rng: np.random.Generator,
    W: np.ndarray,
    rho: float = 0.4,
    beta: np.ndarray | None = None,
    sigma: float = 0.8,
    censoring: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate left-censored SAR Tobit data using the dgp module."""
    out = dgp.simulate_sar_tobit(
        W=_as_sparse(W),
        rho=rho,
        beta=beta,
        sigma=sigma,
        censoring=censoring,
        rng=rng,
    )
    return out["y"], out["X"]


def make_sem_tobit_data(
    rng: np.random.Generator,
    W: np.ndarray,
    lam: float = 0.4,
    beta: np.ndarray | None = None,
    sigma: float = 0.8,
    censoring: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate left-censored SEM Tobit data using the dgp module."""
    out = dgp.simulate_sem_tobit(
        W=_as_sparse(W),
        lam=lam,
        beta=beta,
        sigma=sigma,
        censoring=censoring,
        rng=rng,
    )
    return out["y"], out["X"]


def make_sdm_tobit_data(
    rng: np.random.Generator,
    W: np.ndarray,
    rho: float = 0.4,
    beta1: np.ndarray | None = None,
    beta2: np.ndarray | None = None,
    sigma: float = 0.8,
    censoring: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate left-censored SDM Tobit data using the dgp module."""
    out = dgp.simulate_sdm_tobit(
        W=_as_sparse(W),
        rho=rho,
        beta1=beta1,
        beta2=beta2,
        sigma=sigma,
        censoring=censoring,
        rng=rng,
    )
    return out["y"], out["X"]


def make_panel_sar_tobit_data(
    rng: np.random.Generator,
    W: np.ndarray,
    N: int,
    T: int,
    rho: float = 0.35,
    beta: np.ndarray | None = None,
    sigma: float = 0.8,
    censoring: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate left-censored panel SAR FE Tobit data using the dgp module."""
    out = dgp.simulate_sar_tobit(
        N=N,
        T=T,
        W=_as_sparse(W),
        rho=rho,
        beta=beta,
        sigma=sigma,
        censoring=censoring,
        rng=rng,
    )
    return out["y"], out["X"]


def make_panel_sem_tobit_data(
    rng: np.random.Generator,
    W: np.ndarray,
    N: int,
    T: int,
    lam: float = 0.35,
    beta: np.ndarray | None = None,
    sigma: float = 0.8,
    censoring: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate left-censored panel SEM FE Tobit data using the dgp module."""
    out = dgp.simulate_sem_tobit(
        N=N,
        T=T,
        W=_as_sparse(W),
        lam=lam,
        beta=beta,
        sigma=sigma,
        censoring=censoring,
        rng=rng,
    )
    return out["y"], out["X"]


# ---------------------------------------------------------------------------
# Dynamic DE (direct-estimation) panel data generators
# ---------------------------------------------------------------------------


def make_panel_sar_dynamic_data(
    rng: np.random.Generator,
    W: np.ndarray,
    N: int,
    T: int,
    rho: float = 0.3,
    phi: float = 0.4,
    beta: np.ndarray | None = None,
    sigma: float = 1.0,
    sigma_alpha: float = 0.5,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Generate dynamic SAR panel FE data."""
    out = _simulate_sar_panel_dynamic(
        N=N,
        T=T,
        W=_as_sparse(W),
        rho=rho,
        phi=phi,
        beta=beta,
        sigma=sigma,
        sigma_alpha=sigma_alpha,
        rng=rng,
    )
    y, X = out["y"], out["X"]
    units, times = out["unit"], out["time"]
    df = pd.DataFrame({"y": y, "x1": X[:, 1], "unit": units, "time": times})
    return y, X, df


def make_panel_sem_dynamic_data(
    rng: np.random.Generator,
    W: np.ndarray,
    N: int,
    T: int,
    lam: float = 0.3,
    phi: float = 0.4,
    beta: np.ndarray | None = None,
    sigma: float = 1.0,
    sigma_alpha: float = 0.5,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Generate dynamic SEM panel FE data."""
    out = _simulate_sem_panel_dynamic(
        N=N,
        T=T,
        W=_as_sparse(W),
        lam=lam,
        phi=phi,
        beta=beta,
        sigma=sigma,
        sigma_alpha=sigma_alpha,
        rng=rng,
    )
    y, X = out["y"], out["X"]
    units, times = out["unit"], out["time"]
    df = pd.DataFrame({"y": y, "x1": X[:, 1], "unit": units, "time": times})
    return y, X, df


def make_panel_sdem_dynamic_data(
    rng: np.random.Generator,
    W: np.ndarray,
    N: int,
    T: int,
    lam: float = 0.3,
    phi: float = 0.4,
    beta: np.ndarray | None = None,
    sigma: float = 1.0,
    sigma_alpha: float = 0.5,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Generate dynamic SDEM panel FE data."""
    out = _simulate_sdem_panel_dynamic(
        N=N,
        T=T,
        W=_as_sparse(W),
        lam=lam,
        phi=phi,
        beta=beta,
        sigma=sigma,
        sigma_alpha=sigma_alpha,
        rng=rng,
    )
    y, X = out["y"], out["X"]
    units, times = out["unit"], out["time"]
    df = pd.DataFrame({"y": y, "x1": X[:, 1], "unit": units, "time": times})
    return y, X, df


def make_panel_slx_dynamic_data(
    rng: np.random.Generator,
    W: np.ndarray,
    N: int,
    T: int,
    phi: float = 0.4,
    beta: np.ndarray | None = None,
    sigma: float = 1.0,
    sigma_alpha: float = 0.5,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Generate dynamic SLX panel FE data."""
    out = _simulate_slx_panel_dynamic(
        N=N,
        T=T,
        W=_as_sparse(W),
        phi=phi,
        beta=beta,
        sigma=sigma,
        sigma_alpha=sigma_alpha,
        rng=rng,
    )
    y, X = out["y"], out["X"]
    units, times = out["unit"], out["time"]
    df = pd.DataFrame({"y": y, "x1": X[:, 1], "unit": units, "time": times})
    return y, X, df


# ---------------------------------------------------------------------------
# Static SDM / SDEM panel data generators (for FE recovery tests)
# ---------------------------------------------------------------------------


def make_panel_sdm_fe_data(
    rng: np.random.Generator,
    W: np.ndarray,
    N: int,
    T: int,
    rho: float = 0.4,
    beta1: np.ndarray | None = None,
    beta2: np.ndarray | None = None,
    sigma: float = 1.0,
    sigma_alpha: float = 0.5,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Generate SDM panel FE data with WX terms."""
    out = dgp.simulate_sdm(
        N=N,
        T=T,
        W=_as_sparse(W),
        rho=rho,
        beta1=beta1,
        beta2=beta2,
        sigma=sigma,
        rng=rng,
        unit_effects="fixed",
        unit_effect_sd=sigma_alpha,
    )
    y, X = out["y"], out["X"]
    units, times = out["unit"], out["time"]
    df = pd.DataFrame({"y": y, "x1": X[:, 1], "unit": units, "time": times})
    return y, X, df


def make_panel_sdem_fe_data(
    rng: np.random.Generator,
    W: np.ndarray,
    N: int,
    T: int,
    lam: float = 0.4,
    beta1: np.ndarray | None = None,
    beta2: np.ndarray | None = None,
    sigma: float = 1.0,
    sigma_alpha: float = 0.5,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Generate SDEM panel FE data with WX terms."""
    out = dgp.simulate_sdem(
        N=N,
        T=T,
        W=_as_sparse(W),
        lam=lam,
        beta1=beta1,
        beta2=beta2,
        sigma=sigma,
        rng=rng,
        unit_effects="fixed",
        unit_effect_sd=sigma_alpha,
    )
    y, X = out["y"], out["X"]
    units, times = out["unit"], out["time"]
    df = pd.DataFrame({"y": y, "x1": X[:, 1], "unit": units, "time": times})
    return y, X, df


def set_posterior_means(model, beta: np.ndarray, rho: float | None = None) -> None:
    """Inject posterior means into a model for testing without MCMC."""
    posterior: dict[str, np.ndarray] = {
        "beta": np.array([[beta]], dtype=float),
    }
    if rho is not None:
        posterior["rho"] = np.array([[rho]], dtype=float)
    model._idata = az.from_dict({"posterior": posterior})


def make_idata(samples_by_var: dict[str, np.ndarray]) -> xr.DataTree:
    """Build an InferenceData from a dict of arrays.

    Each array is treated as posterior draws. Arrays with one dimension are
    expanded with a leading chain axis; otherwise they are passed through.
    """
    posterior: dict[str, np.ndarray] = {}
    for k, v in samples_by_var.items():
        arr = np.asarray(v)
        if arr.ndim == 1:
            arr = arr[None, ...]
        posterior[k] = arr
    return az.from_dict({"posterior": posterior})


def effects_kwargs(
    code: int, index: str = "unit_effects", treatment: str = "fixed"
) -> dict:
    """Design keywords for a legacy effects code: 0 none, 1 unit or pair, 2 time, 3 both."""
    return {
        index: treatment if code in (1, 3) else None,
        "time_effects": code in (2, 3),
    }


def flow_form(cls, **design):
    """A constructor of the flow family ``cls`` with ``design`` keywords fixed.

    It takes a legacy ``effects`` code (0 none, 1 pair, 2 time, 3 both), the
    pair effects being the family's treatment.
    """

    def make(*args, effects: int = 0, **kw):
        if effects:
            kw.update(effects_kwargs(effects, "pair_effects", cls._pair_effects))
        return cls(*args, **design, **kw)

    make.__name__ = cls.__name__ + ("(separable)" if design.get("separable") else "")
    make.family = cls
    return make


# ---------------------------------------------------------------------------
# Log-determinant oracles: exact references the package itself no longer needs
# ---------------------------------------------------------------------------


def eigen_cheb_coeffs(
    W,
    order: int = 20,
    rmin: float = -1.0,
    rmax: float = 1.0,
    eigs: np.ndarray | None = None,
) -> dict:
    """Chebyshev coefficients in ρ of log|I - ρW| from W's eigenvalues (a test oracle).

    Near-minimax polynomial approximation over ``[rmin, rmax]``.

    Parameters
    ----------
    W : array-like
        Spatial weights matrix (dense or sparse).
    order : int, default 20
        Polynomial degree.  15–30 is usually sufficient.
    rmin : float, default -1.0
        Lower bound of the rho interval.
    rmax : float, default 1.0
        Upper bound of the rho interval.
    eigs : np.ndarray, optional
        Pre-computed eigenvalues of W (skips the O(n³) decomposition).

    Returns
    -------
    dict
        ``{"coeffs", "rmin", "rmax", "order", "method"}`` with ``method``
        ``"eigenvalue"``.
    """
    if order <= 0:
        raise ValueError("order must be positive.")
    if rmax <= rmin:
        raise ValueError("rmax must be greater than rmin.")

    if eigs is not None:
        eigs_arr = np.asarray(eigs, dtype=np.complex128)
    else:
        W_dense = W.toarray() if sp.issparse(W) else np.asarray(W, dtype=np.float64)
        eigs_arr = np.linalg.eigvals(W_dense)

    # Chebyshev nodes on [-1, 1], mapped to [rmin, rmax]
    k = np.arange(1, order + 1)
    nodes_cos = np.cos((2 * k - 1) * np.pi / (2 * order))
    rho_nodes = 0.5 * (rmax - rmin) * nodes_cos + 0.5 * (rmax + rmin)

    logdet_at_nodes = np.sum(
        np.log(np.abs(1.0 - rho_nodes[:, None] * eigs_arr[None, :])), axis=1
    ).real

    # Chebyshev coefficients via DCT-I
    coeffs = chebyshev_coeffs_dct1(logdet_at_nodes)

    result = {
        "coeffs": coeffs,
        "rmin": rmin,
        "rmax": rmax,
        "order": order,
        "method": "eigenvalue",
    }

    return result


def flow_logdet_grad_exact(W, rho_d: float, rho_o: float, rho_w: float) -> np.ndarray:
    r"""Exact ``∇_ρ log|I_N − W_F|`` from the spectrum of ``W`` (reference; small ``n``).

    Uses the shared Kronecker eigenbasis: ``μ_ij = ρ_o λ_i + ρ_d λ_j + ρ_w λ_i λ_j``
    and ``g_d = −Σ_ij λ_j/(1−μ_ij)`` etc.  Only for testing / small ``n`` — forms
    the ``n`` eigenvalues of ``W`` (``O(n³)``), never the ``N×N`` matrix.
    """
    Wd = sp.csr_matrix(W).toarray()
    lam = np.linalg.eigvals(Wd)
    li = lam[:, None]
    lj = lam[None, :]
    mu = rho_o * li + rho_d * lj + rho_w * (li * lj)
    inv = 1.0 / (1.0 - mu)
    g_d = -np.sum(lj * inv).real
    g_o = -np.sum(li * inv).real
    g_w = -np.sum(li * lj * inv).real
    return np.array([g_d, g_o, g_w], dtype=np.float64)
