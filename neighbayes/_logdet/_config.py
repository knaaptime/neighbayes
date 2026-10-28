"""Log-determinant configuration: method names, resolution, and bounds.

A model names its log-determinant with ``logdet_method``:

* ``"eigenvalue"`` — the sum over W's eigenvalues, from one dense
  eigendecomposition;
* ``"cheb"`` — Chebyshev interpolation of factorized values at Chebyshev nodes;
* ``"aaa"`` — AAA rational interpolation of factorized values at adaptive points;
* ``"spline"`` — cubic spline through factorized values on a grid;
* ``"stochastic"`` — stochastic Chebyshev expansion (Han et al. 2015), no
  factorization, the one method with sampling error;
* ``"trace_series"``, ``"resolvent"`` — the unrestricted flow models' trace
  series and stochastic resolvent estimator.

The interpolating methods factor ``I − ρW``, and ``logdet_decomp`` picks how:
``"chol"`` (sparse Cholesky, for an undirected, D-symmetrizable ``W``) or
``"lu"`` (sparse LU, KLU or UMFPACK by pattern); ``"auto"`` takes ``chol`` where
the method has it and ``W`` allows it, else ``lu``.  The other methods ignore it.

With both left to auto the method is selected from the problem: ``eigenvalue``
for n ≤ ``NEIGHBAYES_LOGDET_EIGEN_MAX_N`` (default 500); ``aaa`` on ``chol``
(undirected ``W``) or ``lu`` (directed) for n ≤
``NEIGHBAYES_LOGDET_FACTOR_MAX_N`` (default 60000) unless factorization
fill-in would dominate; ``stochastic`` otherwise.

Internally each pair resolves to one key (:func:`logdet_key`), on which the
factories and samplers dispatch.
"""

from __future__ import annotations

import os
import warnings
from collections import OrderedDict
from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping

# ---------------------------------------------------------------------------
# Cache constants
# ---------------------------------------------------------------------------

_LOGDET_FN_CACHE_MAXSIZE = 64
_LOGDET_FN_CACHE: OrderedDict[tuple, Any] = OrderedDict()

# ---------------------------------------------------------------------------
# Enum and type alias
# ---------------------------------------------------------------------------


class LogDetMethod(str, Enum):
    """Internal log-determinant keys (see :func:`logdet_key` for the public names)."""

    EIGENVALUE = "eigenvalue"
    STOCH_CHEB = "stoch_cheb"
    CHOL_CHEB = "chol_cheb"
    LU_CHEB = "lu_cheb"
    LU_AAA = "lu_aaa"
    CHOL_AAA = "chol_aaa"
    LU_SPLINE = "lu_spline"


VALID_LOGDET_METHODS: frozenset[str] = frozenset(m.value for m in LogDetMethod)

#: The public names: the method, and for the interpolating ones, the factorization.
LOGDET_METHODS = (
    "eigenvalue",
    "cheb",
    "aaa",
    "spline",
    "stochastic",
    "trace_series",
    "resolvent",
)
LOGDET_DECOMPS = ("auto", "chol", "lu")

#: ``(logdet_method, logdet_decomp)`` → internal key.
_KEYS = {
    ("cheb", "chol"): "chol_cheb",
    ("cheb", "lu"): "lu_cheb",
    ("aaa", "chol"): "chol_aaa",
    ("aaa", "lu"): "lu_aaa",
    ("spline", "lu"): "lu_spline",
}
#: Methods that factor nothing (``logdet_decomp`` is ignored).
_UNDECOMPOSED = {"eigenvalue": "eigenvalue", "stochastic": "stoch_cheb"}
#: The unrestricted flow models' own methods.
FLOW_LOGDET_METHODS = ("trace_series", "resolvent")
_PUBLIC = {key: pair for pair, key in _KEYS.items()} | {
    key: (method, None) for method, key in _UNDECOMPOSED.items()
}


def logdet_key(method: str | None, decomp: str = "auto", *, W=None) -> str | None:
    """The internal key for ``logdet_method`` × ``logdet_decomp``.

    ``None`` when both are left to auto (the key is then selected from the
    problem by :func:`resolve_logdet_method`).  An interpolating method given
    with ``decomp="auto"`` takes ``chol`` where it has it and ``W`` is
    undirected, else ``lu``; a decomposition given with ``method=None`` takes
    ``aaa``.

    Raises
    ------
    ValueError
        For an unknown name, or a flow-only method.
    NotImplementedError
        For a pair with no implementation (e.g. ``spline`` on ``chol``).
    """
    if decomp not in LOGDET_DECOMPS:
        raise ValueError(
            f"logdet_decomp must be one of {LOGDET_DECOMPS}, got {decomp!r}."
        )
    if method is None:
        if decomp == "auto":
            return None
        method = "aaa"
    if method not in LOGDET_METHODS:
        raise ValueError(
            f"logdet_method must be one of {LOGDET_METHODS} or None, got {method!r}."
        )
    if method in FLOW_LOGDET_METHODS:
        raise ValueError(
            f"logdet_method={method!r} is for the unrestricted flow models only."
        )
    if method in _UNDECOMPOSED:
        return _UNDECOMPOSED[method]
    if decomp == "auto":
        # Every interpolating method has lu.
        decomp = "chol" if (method, "chol") in _KEYS and _is_symmetric_W(W) else "lu"
    key = _KEYS.get((method, decomp))
    if key is None:
        pairs = ", ".join(f"{m}/{d}" for m, d in _KEYS)
        raise NotImplementedError(
            f"logdet_method={method!r} on logdet_decomp={decomp!r} is not "
            f"implemented; implemented pairs: {pairs}."
        )
    return key


def logdet_label(key: str) -> str:
    """The public spelling of an internal key, for messages (``"aaa on chol"``)."""
    method, decomp = _PUBLIC.get(key, (key, None))
    return method if decomp is None else f"{method} on {decomp}"


# ---------------------------------------------------------------------------
# Dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LogdetBounds:
    """Resolved logdet method and rho interval."""

    method: str
    rho_min: float
    rho_max: float
    source: str


# ---------------------------------------------------------------------------
# Resolution functions
# ---------------------------------------------------------------------------


def resolve_logdet_method(
    method: str | None,
    *,
    n: int,
    W=None,
) -> str:
    """Validate ``method`` and auto-select when ``None``.

    Parameters
    ----------
    method
        One of the valid method names, or ``None`` for auto-selection.
    n
        Spatial dimension; used for auto-selection.
    W
        Optional spatial weights matrix.  When ``method`` is ``None``
        and ``n`` is in the medium range, the auto-selector checks
        whether ``W`` is symmetric (undirected graph) to choose between
        ``"chol_cheb"`` (symmetric) and ``"lu_aaa"`` (non-symmetric).
        If ``W`` is not provided, defaults to ``"chol_cheb"``.

    Returns
    -------
    str
        Canonical method name.
    """
    if method is None:
        return _auto_logdet_method(int(n), W=W)
    if method not in VALID_LOGDET_METHODS:
        valid = ", ".join(sorted(VALID_LOGDET_METHODS))
        raise ValueError(f"Unknown logdet method: {method!r}. Valid options: {valid}.")
    return method


def _is_symmetric_W(W) -> bool:
    """Check whether ``W`` describes an undirected graph → the Cholesky logdet.

    The discriminator is **D-symmetrizability**, not literal matrix symmetry:
    a *row-standardized* undirected graph ``W = D⁻¹A`` (``A`` symmetric) is not
    literally symmetric, yet ``chol_cheb`` handles it via D-symmetrization.
    Only genuinely directed graphs (asymmetric adjacency, e.g. KNN / travel time
    / migration) fall through to the LU-based ``aaa`` path.

    Uses ``libpysal.graph.Graph.asymmetry(intrinsic=False)`` (a topology check)
    when ``W`` is a Graph.  For a sparse/dense matrix — which is what the models
    actually pass (``self._W_sparse``) — a literally symmetric matrix returns
    ``True`` immediately; otherwise D-symmetrizability is tested with the same
    ``_d_symmetrize`` routine ``chol_cheb`` relies on, so this predicate
    agrees exactly with whether the Cholesky path is applicable.
    """
    import numpy as np
    import scipy.sparse as sp

    if W is None:
        return True  # default: assume symmetric

    # libpysal Graph: use built-in topology asymmetry check (intrinsic=False
    # ignores weight values, so row-standardization does not read as directed).
    if hasattr(W, "asymmetry"):
        return W.asymmetry(intrinsic=False).empty

    if sp.issparse(W):
        # Sparse difference stays sparse — never densify (n=20k dense is ~3.2GB).
        Wc = W.tocsr()
        diff = (Wc - Wc.T).tocoo()
        if diff.nnz == 0 or bool(np.all(np.abs(diff.data) <= 1e-10)):
            return True
        # Not literally symmetric: may still be a D-symmetrizable (row-
        # standardized undirected) graph, which chol_cheb handles.  Test
        # with the actual symmetrization so routing == applicability.
        try:
            from ._chol_cheb import _d_symmetrize

            _d_symmetrize(Wc)
            return True
        except ValueError:  # no symmetrizing diagonal: directed W
            return False
    else:
        W_arr = np.asarray(W)
        if W_arr.ndim != 2:
            return True  # 1-D eigenvalue array — not applicable
        if np.allclose(W_arr, W_arr.T, atol=1e-10):
            return True
        try:
            from ._chol_cheb import _d_symmetrize

            _d_symmetrize(sp.csr_matrix(W_arr))
            return True
        except ValueError:  # no symmetrizing diagonal: directed W
            return False


def _auto_logdet_method(n: int, W=None) -> str:
    """Auto-select based on matrix dimension n and W symmetry.

    - ``eigenvalue`` for n ≤ eigen_cutoff (default 500): exact O(n³) eigendecomposition.
    - ``chol_cheb`` for n ≤ cheb_cutoff (default 60000) when W is symmetric:
      exact logdet via sparse Cholesky at Chebyshev nodes with symbolic reuse.
      Measured setup (2D rook, ρ ∈ [0.1, 0.8], 2026-07): ~96ms at n=10k, ~583ms
      at n=40k, ~1.18s at n=60k.  Accuracy: 4.6e-7 (n=10k) to 2.6e-6 (n=60k).
      Eval: ~1.7μs/ρ via Clenshaw.
    - ``aaa`` for n ≤ cheb_cutoff when W is non-symmetric (directed graph):
      exact logdet via sparse LU (KLU with symbolic reuse) at adaptively-selected
      AAA support points.  Rational approximation converges exponentially near
      singularities.  Uses an adaptive coarse grid of 8–30 LU factorizations,
      sized from the interval's Bernstein-ellipse rate, selecting ~7 support
      points.  Measured setup ~157ms at n=10k; eval ~5μs/ρ; error 1e-8 to 5e-8.
    - ``stoch_cheb`` for n > cheb_cutoff: stochastic Chebyshev expansion.
      Lower setup cost (~62ms at n=10k, ~328ms at n=60k) but carries stochastic
      error 0.7 to 3.5 with 200 probes.  Eval: ~57μs/ρ.  Use when factorization
      fill-in makes exact setup too expensive.

    **Fill-in guard**: when ``W`` is provided and ``n ≤ cheb_cutoff``, the
    auto-selector estimates the fill-in ratio ``nnz(W²) / nnz(W)`` (in
    ``O(nnz)`` via :func:`~._fillin.estimate_fillin_ratio`).  If this exceeds
    ``NEIGHBAYES_LOGDET_MAX_FILLIN_RATIO`` (default 20), the exact
    factorization path is skipped in favour of ``stoch_cheb`` and a
    warning is issued.  This catches dense or hub-dominated graphs (e.g.
    KNN-50, fully dense W) at medium ``n`` where the ``n`` cutoff alone
    would not route to stochastic.

    The ``cheb_cutoff`` default of 60,000 is where the benchmark ends, not where
    the exact path stops paying.  It was raised from 20,000 after vectorising the
    symmetrizing-diagonal recovery roughly halved Cholesky setup: at n = 60,000
    the exact path now costs ~850ms more than the stochastic one for six orders
    of magnitude less error, which is negligible against any chain that runs for
    seconds.  Raise it further via ``NEIGHBAYES_LOGDET_FACTOR_MAX_N`` if Cholesky
    fill-in on your graph stays affordable past that.
    """
    eigen_cutoff = _eigen_cutoff()
    cheb_cutoff_raw = os.getenv("NEIGHBAYES_LOGDET_FACTOR_MAX_N", "60000")
    try:
        cheb_cutoff = max(eigen_cutoff + 1, int(cheb_cutoff_raw))
    except ValueError:
        cheb_cutoff = 60000
    if n <= eigen_cutoff:
        return "eigenvalue"
    if n <= cheb_cutoff:
        # Fill-in guard: if W² is much denser than W, Cholesky/LU fill-in
        # will dominate setup cost.  Route to stochastic instead.
        if W is not None:
            import scipy.sparse as sp

            from ._fillin import estimate_fillin_ratio

            W_csr = sp.csr_matrix(W) if not sp.issparse(W) else W.tocsr()
            ratio = estimate_fillin_ratio(W_csr)
            max_ratio = _max_fillin_ratio()
            if ratio > max_ratio:
                fallback = "chol_aaa" if _is_symmetric_W(W) else "lu_aaa"
                warnings.warn(
                    f"Estimated fill-in ratio nnz(W²)/nnz(W) ≈ {ratio:.0f}× "
                    f"exceeds max_fillin_ratio={max_ratio:.0f}×. "
                    f"Sparse factorization fill-in would dominate setup cost; "
                    f"auto-selecting the stochastic method instead of "
                    f"{logdet_label(fallback)}. Set logdet_method explicitly to "
                    f"override.",
                    stacklevel=3,
                )
                return "stoch_cheb"
        # Check W symmetry: chol_aaa for symmetric (undirected graph),
        # aaa for non-symmetric (directed graph: KNN, travel time, migration).
        # chol_aaa combines CHOLMOD's cheaper factorization with AAA's
        # root-exponential convergence — the best of both.
        if _is_symmetric_W(W):
            return "chol_aaa"
        else:
            return "lu_aaa"
    # Stochastic Chebyshev (Han et al. 2015): geometric convergence via
    # Bernstein ellipse, avoids O(n³) eigendecomposition.
    return "stoch_cheb"


def _eigen_cutoff() -> int:
    """Return the eigendecomposition size cutoff (``NEIGHBAYES_LOGDET_EIGEN_MAX_N``).

    Single source of truth for the maximum ``n`` at which the O(n³)
    eigendecomposition is considered affordable.  Read from the environment
    variable ``NEIGHBAYES_LOGDET_EIGEN_MAX_N`` (default 500).  Used by
    :func:`_auto_logdet_method` for auto-selection.
    """
    raw = os.getenv("NEIGHBAYES_LOGDET_EIGEN_MAX_N", "500")
    try:
        return max(1, int(raw))
    except ValueError:
        return 500


#: Hard limit for an explicit ``logdet_decomp="eig"`` on large W.
#: The auto-select cutoff (:func:`_eigen_cutoff`, default 500) is a performance
#: guideline — slightly exceeding it is fine.  This hard guard prevents
#: genuinely dangerous O(n²) memory allocations (e.g. a dense 20k×20k matrix
#: is ~3 GB).  Override with ``NEIGHBAYES_LOGDET_EIGEN_HARD_MAX_N``.
_EIGEN_HARD_MAX_N_DEFAULT = 5000


def _eigen_hard_max_n() -> int:
    """Return the hard size limit for explicit eigenvalue eigendecomposition.

    The auto-select cutoff (``NEIGHBAYES_LOGDET_EIGEN_MAX_N``, default 500) is
    a performance guideline — tests and users may legitimately force
    ``logdet_decomp="eig"`` on moderately larger W.  This hard guard
    only fires when the allocation would be genuinely dangerous (O(n²) memory
    in the multi-GB range).  Read from ``NEIGHBAYES_LOGDET_EIGEN_HARD_MAX_N``
    (default 5000; a 5000×5000 dense matrix is ~200 MB).
    """
    raw = os.getenv(
        "NEIGHBAYES_LOGDET_EIGEN_HARD_MAX_N", str(_EIGEN_HARD_MAX_N_DEFAULT)
    )
    try:
        return max(1, int(raw))
    except ValueError:
        return _EIGEN_HARD_MAX_N_DEFAULT


#: Default fill-in ratio (estimated ``nnz(W²) / nnz(W)``) above which the
#: auto-selector routes to ``stoch_cheb`` instead of exact factorization.
#: The Cholesky factor has at least as many nonzeros as ``W²`` in its early
#: rows, so this ratio directly predicts factorization cost.  The 2D rook
#: lattice (degree ~8) gives ratio ~8; KNN-50 gives ~50; dense W gives ~n.
#: Default 20.0 sits between regular lattices and dense/high-degree graphs.
DEFAULT_MAX_FILLIN_RATIO = 20.0


def _max_fillin_ratio() -> float:
    """Return the fill-in ratio threshold for auto-selection override.

    Read from ``NEIGHBAYES_LOGDET_MAX_FILLIN_RATIO`` (default 20.0).
    Set very high (e.g. 99999) to effectively disable the guard.
    """
    raw = os.getenv("NEIGHBAYES_LOGDET_MAX_FILLIN_RATIO", str(DEFAULT_MAX_FILLIN_RATIO))
    try:
        return max(1.0, float(raw))
    except ValueError:
        return DEFAULT_MAX_FILLIN_RATIO


def resolve_logdet_bounds(
    method: str | None,
    *,
    n: int,
    priors: Mapping[str, Any] | None = None,
    rho_min: float | None = None,
    rho_max: float | None = None,
    W=None,
) -> LogdetBounds:
    """Resolve rho bounds from explicit overrides, priors, or defaults.

    For row-standardized W the stability interval is approximately (-1, 1).

    ``W`` (when supplied) participates in auto-selection so that the
    method recorded here agrees with every other resolution site —
    without it a directed graph would be auto-routed to the
    symmetric-only ``chol_cheb``.
    """
    resolved_method = resolve_logdet_method(method, n=int(n), W=W)
    source = "default"

    if rho_min is not None or rho_max is not None:
        if rho_min is None or rho_max is None:
            raise ValueError("Both rho_min and rho_max must be provided together.")
        lo = float(rho_min)
        hi = float(rho_max)
        source = "override"
    else:
        p = priors or {}
        lo_prior = None
        hi_prior = None
        for lk, hk in (("rho_lower", "rho_upper"), ("lam_lower", "lam_upper")):
            if lk in p and hk in p:
                lo_prior = float(p[lk])
                hi_prior = float(p[hk])
                break

        if lo_prior is not None and hi_prior is not None:
            lo = lo_prior
            hi = hi_prior
            source = "prior"
        else:
            lo = -1.0
            hi = 1.0

    if hi <= lo:
        raise ValueError(f"Invalid rho interval: rho_min={lo}, rho_max={hi}.")

    return LogdetBounds(
        method=resolved_method,
        rho_min=float(lo),
        rho_max=float(hi),
        source=source,
    )
