"""The flow (origin-destination) model base and the count flow families.

:class:`FlowModel` carries the flow structure every family shares: an
:math:`n \\times n` flow matrix per period (or its vectorized form), periods
stacked time-first, and three Kronecker-product weight matrices

.. math::

    W_d = I_n \\otimes W, \\quad W_o = W \\otimes I_n, \\quad W_w = W \\otimes W,

following :cite:t:`lesage2008SpatialEconometric`, with :math:`N = n^2`
pairs.  The filter mixins :class:`_Unrestricted`, :class:`_Separable` and
:class:`_Aspatial` give a family its spatial filter.  The Gaussian families
are in :mod:`._gaussian`; this module holds the negative binomial and Poisson
cross-sections.
"""

from __future__ import annotations

import warnings
from abc import abstractmethod
from typing import Any, Optional, Union

import numpy as np
import pandas as pd
import pytensor.tensor as pt
import scipy.sparse as sp

from ..._backends.sampler_helpers import select_backend
from ..._lazy_deps import pm, xr
from ..._logdet import (
    FLOW_LOGDET_METHODS,
    make_flow_separable_logdet,
    make_flow_separable_logdet_numpy,
)
from ..._ops import kron_solve_vec
from ...graph import _weights_to_csr, flow_lags
from .._base._nb import nb_alpha_rv
from .._base._panel import _absorbed_columns, _demean_panel
from .._design import flow
from .._mixins._flow_shared import FlowSharedMethods
from ..base import SpatialModel

_EFFECT_KEYS = ("origin", "destination", "intraunit", "network", "total")


class _FlowEffectMoments:
    r"""W-only moments behind the exact LeSage flow-effects decomposition.

    With ``Y`` the ``n × n`` flow array (origins × destinations, row-major as
    in the flow vec), the filter acts as
    ``L(Y) = ρ_o WY + ρ_d YWᵀ + ρ_w WYWᵀ``.  Left and right multiplication
    commute, so

    .. math::

        A^{-1}(Y) = \sum_{p,q} C_{pq}\, W^p Y (W^\top)^q, \qquad
        \sum_{p,q} C_{pq} a^p b^q = \frac{1}{1 - \rho_o a - \rho_d b - \rho_w ab}.

    Applied to the region-``j`` shocks (column ``j``, row ``j`` and cell
    ``(j, j)``) and summed over ``j``, every masked response sum the effects
    need reduces to inner products of three vector sequences that depend on
    ``W`` alone: row sums ``R_p = W^p 1``, column sums ``c_p = (Wᵀ)^p 1`` and
    diagonals ``d_p = diag(W^p)``.  They are computed once per model and
    extended on demand; no ``n² × n²`` operator or ``(n², n)`` array is formed.
    ``diag(W^p)`` is taken in column blocks, so memory stays ``n × block``.
    """

    #: bytes allowed for one column block of the diag(W^p) recursion
    _BLOCK_BYTES = 256 * 1024 * 1024

    def __init__(self, W):
        self.W = sp.csr_matrix(W, dtype=np.float64)
        self.n = self.W.shape[0]
        self.w_inf = float(abs(self.W).sum(axis=1).max()) if self.W.nnz else 0.0
        self.P = -1

    def ensure(self, P: int) -> None:
        """Compute moments through order ``P`` (recomputing from scratch)."""
        if P <= self.P:
            return
        n, W = self.n, self.W
        WT = W.T.tocsr()
        R = np.empty((P + 1, n))
        C = np.empty((P + 1, n))
        D = np.empty((P + 1, n))
        R[0] = C[0] = D[0] = 1.0
        for p in range(1, P + 1):
            R[p] = W @ R[p - 1]
            C[p] = WT @ C[p - 1]
        block = max(1, min(n, self._BLOCK_BYTES // (8 * n)))
        for a in range(0, n, block):
            b = min(n, a + block)
            cols = np.arange(a, b)
            Z = np.zeros((n, b - a))
            Z[cols, np.arange(b - a)] = 1.0
            for p in range(1, P + 1):
                Z = W @ Z
                D[p, a:b] = Z[cols, np.arange(b - a)]
        self.P = P
        m = R.sum(axis=1)  # 1ᵀW^p1
        t = D.sum(axis=1)  # tr(W^p)
        # (shock, statistic) → (P+1)×(P+1) moment matrix M, so that the sum
        # for that shock is Σ_pq C_pq M_pq.  Statistics: total, intra,
        # row j (origin = j, intra included), column j (destination = j).
        self.mats = np.stack(
            [
                # destination shock 1·e_jᵀ: entry [o, d] = R_p[o] (W^q)_{dj}
                np.outer(m, m),
                R @ D.T,
                R @ C.T,
                np.outer(m, t),
                # origin shock e_j·1ᵀ: entry [o, d] = (W^p)_{oj} R_q[d]
                np.outer(m, m),
                D @ R.T,
                np.outer(t, m),
                C @ R.T,
                # intra shock e_j·e_jᵀ: entry [o, d] = (W^p)_{oj} (W^q)_{dj}
                C @ C.T,
                D @ D.T,
                D @ C.T,
                C @ D.T,
            ]
        )

    def order_for(self, rd, ro, rw, tol: float = 1e-12) -> int:
        """Series order bounding the neglected tail below ``tol`` (relative)."""
        w = self.w_inf
        rd, ro, rw = (np.asarray(x, dtype=np.float64) for x in (rd, ro, rw))
        # A draw on the separable surface ρ_w = −ρ_d ρ_o has coefficients
        # C_pq = ρ_o^p ρ_d^q, so its series converges whenever each ρ alone is
        # inside the disc; the general bound would reject valid separable draws.
        separable = np.abs(rw + rd * ro) <= 1e-12 * np.maximum(np.abs(rd * ro), 1e-300)
        rd, ro, rw = np.abs(rd), np.abs(ro), np.abs(rw)
        r_all = np.where(separable, np.maximum(rd, ro) * w, (ro + rd) * w + rw * w * w)
        r = float(np.max(r_all, initial=0.0))
        if r >= 1.0:
            raise ValueError(
                "Flow effects need |rho_d| + |rho_o| + |rho_w| < 1 (or, on the "
                "separable surface, |rho_d|, |rho_o| < 1) in the max-row-sum norm "
                f"of W; a posterior draw reaches {r:.4f}."
            )
        if r == 0.0:
            return 1
        return int(np.clip(np.ceil(np.log(tol * (1.0 - r)) / np.log(r)), 1, 20_000))


def _flow_effect_sums(moments: _FlowEffectMoments, rd, ro, rw) -> np.ndarray:
    """Masked response sums for every draw, shape ``(G, 3, 4)``.

    Axis 1 is the shock (destination ``D+I``, origin ``O+I``, intra ``I``);
    axis 2 is (total, intra, origin, destination) as in LeSage's
    ``calc_effects``, each divided by ``N = n²``.  Exact up to the series tail.
    """
    rd, ro, rw = (np.atleast_1d(np.asarray(x, dtype=np.float64)) for x in (rd, ro, rw))
    P = moments.order_for(rd, ro, rw)
    moments.ensure(P)
    M = moments.mats[:, : P + 1, : P + 1]  # (12, P+1, P+1)
    G = rd.shape[0]
    acc = np.zeros((M.shape[0], G))
    # C on anti-diagonals k = p + q, indexed by p: C_k[p] = C_{p, k−p}.
    prev2 = np.zeros((P + 1, G))
    prev1 = np.zeros((P + 1, G))
    prev1[0] = 1.0  # k = 0
    acc += M[:, 0, 0][:, None] * prev1[0][None, :]
    p_all = np.arange(P + 1)
    for k in range(1, 2 * P + 1):
        cur = np.zeros((P + 1, G))
        cur[1:] += ro * prev1[:-1] + rw * prev2[:-1]  # from (p−1, q) and (p−1, q−1)
        cur += rd * prev1  # from (p, q−1)
        valid = (p_all <= k) & (k - p_all <= P)
        cur[~valid] = 0.0
        pv = p_all[valid]
        acc += M[:, pv, k - pv] @ cur[pv]
        prev2, prev1 = prev1, cur
    acc /= float(moments.n) ** 2
    r = acc.reshape(3, 4, G)  # (shock, [total, intra, row j, column j], draw)
    tot, intra, row, col = r[:, 0], r[:, 1], r[:, 2], r[:, 3]
    out = np.stack([tot, intra, row - intra, col - intra], axis=1)  # (3, 4, G)
    return np.transpose(out, (2, 0, 1))


def _compute_flow_effects(
    sums: np.ndarray,
    beta_d: np.ndarray,
    beta_o: np.ndarray,
    beta_intra: Optional[np.ndarray] = None,
) -> dict[str, np.ndarray]:
    """LeSage / Thomas-Agnan scalar effects for every draw.

    Implements the decomposition of Thomas-Agnan & LeSage (2014, §83.5).  The
    destination shock for predictor p places ``β_d`` on every flow whose
    destination is region ``j`` and ``β_d + β_intra`` on the intra flow
    ``(j, j)`` (``X_intra`` is built from ``X_dest``); the origin shock places
    ``β_o`` on every flow whose origin is ``j``.  Both are linear in β, so the
    response sums are ``β_d·S_{D+I} + β_intra·S_I`` and ``β_o·S_{O+I}`` with the
    unit-shock sums from :func:`_flow_effect_sums`.

    Parameters
    ----------
    sums : ndarray, shape (G, 3, 4)
        Output of :func:`_flow_effect_sums`.
    beta_d, beta_o : ndarray, shape (G, k_d) and (G, k_o)
    beta_intra : ndarray, shape (G, k_d), optional
        Coefficients on the ``intra_*`` block; ``None`` means zero.

    Returns
    -------
    dict
        Per-side keys ``"dest_<eff>"`` (G, k_d) and ``"orig_<eff>"`` (G, k_o)
        and combined keys ``<eff>``: the sum of the two sides when
        ``k_d == k_o`` (same variables), else their concatenation.
    """
    beta_d = np.atleast_2d(beta_d)
    beta_o = np.atleast_2d(beta_o)
    k_d, k_o = beta_d.shape[1], beta_o.shape[1]
    S_DI, S_OI, S_I = sums[:, 0, :], sums[:, 1, :], sums[:, 2, :]  # (G, 4)
    dest = beta_d[:, :, None] * S_DI[:, None, :]  # (G, k_d, 4)
    if beta_intra is not None:
        dest = dest + np.atleast_2d(beta_intra)[:, :, None] * S_I[:, None, :]
    orig = beta_o[:, :, None] * S_OI[:, None, :]

    out: dict[str, np.ndarray] = {}
    for side, arr in (("dest", dest), ("orig", orig)):
        total, intra, origin, destination = (arr[..., i] for i in range(4))
        out[f"{side}_total"] = total
        out[f"{side}_intraunit"] = intra
        out[f"{side}_origin"] = origin
        out[f"{side}_destination"] = destination
        out[f"{side}_network"] = total - origin - destination - intra
    for eff in _EFFECT_KEYS:
        if k_d == k_o:
            out[eff] = out[f"dest_{eff}"] + out[f"orig_{eff}"]
        else:
            out[eff] = np.concatenate([out[f"dest_{eff}"], out[f"orig_{eff}"]], axis=1)
    return out


class FlowModel(FlowSharedMethods, SpatialModel):
    """Abstract base class for Bayesian spatial flow regression models.

    Unlike :class:`~neighbayes.models.base.SpatialModel`, this class works
    with an :math:`N = n^2` vectorized response per period and three
    Kronecker-product weight matrices constructed from a single ``n x n``
    graph.  The API mirrors :class:`~neighbayes.models.base.SpatialModel`
    (``fit``, ``summary``, ``inference_data``) but the internals are tailored
    to the flow structure.  A panel stacks ``T`` periods time-first; ``T=1``
    is the cross-section.

    The model accepts a full O-D design matrix *X* of shape ``(N T, p)``,
    typically produced by :func:`~neighbayes.graph.flow_design_matrix` or
    :func:`~neighbayes.graph.flow_design_matrix_with_orig`.

    Parameters
    ----------
    y : array-like
        The flows: shape ``(n, n)`` or ``(N,)`` for a cross-section, and
        ``(T, n, n)``, ``(T, N)`` or ``(N T,)`` for a panel.
    X : np.ndarray or pandas.DataFrame, shape (N T, p)
        Full origin-destination design matrix, periods stacked time-first.
        If a DataFrame, column names are inferred automatically.
    W : libpysal.graph.Graph or scipy.sparse / dense (n×n) matrix
        Row-standardized regional weights on *n* units (Graph or matrix).
        Validated by :func:`~neighbayes.graph._graph_to_csr`.
    T : int, default 1
        Number of periods; ``1`` is the cross-section.
    col_names : list[str], optional
        Column labels for *X*.  If *X* is a DataFrame, column names are
        inferred automatically.  Defaults to ``["x0", "x1", ...]``.
    k : int, optional
        Number of regional attribute columns in the design matrix (i.e.,
        the number of destination/origin variable pairs).  When the design
        matrix follows the standard LeSage layout
        ``[intercept, intra_indicator, dest_*, orig_*, intra_*, (dist)]``,
        *k* can be inferred from the column names.  Provide *k* explicitly
        if column names do not follow the ``dest_*``/``orig_*`` convention.
    priors : dict, optional
        Override default priors.  Supported keys vary by subclass.
    logdet_method : str, optional
        The flow log-determinant; ``None`` (default) auto-selects.  The
        unrestricted classes take ``"trace_series"`` or ``"resolvent"``; the
        separable classes factor it into ``n x n`` terms and take the
        single-``W`` methods with ``logdet_decomp``; the aspatial classes have
        none.
    nonnegative_dependence : bool, default True
        If True, use a ``pm.Dirichlet`` prior that restricts :math:`\\rho_d,
        \\rho_o, \\rho_w \\geq 0` with :math:`\\rho_d + \\rho_o + \\rho_w \\leq 1`.
        This is NUTS-safe and appropriate for most flow applications.
        If False, use three independent ``pm.Uniform(-1, 1)`` priors with a
        differentiable quadratic-wall stability potential.
    robust : bool, default False
        Student-t observation errors, on the classes whose NUTS model has them.
    shared_od_covariates : bool, optional
        If ``None`` (default), the destination and origin design blocks are
        compared and symmetry is auto-detected.  Set explicitly to override
        the heuristic — for example, when using
        :func:`~neighbayes.graph.flow_design_matrix_with_orig` with distinct
        attributes for the origin and destination sides.  Controls the default
        behavior of :meth:`impacts` when ``mode="auto"``.
    effects : int, default 0
        The panel's fixed effects: ``0`` none, ``1`` pair, ``2`` time, ``3``
        both.  A cross-section takes none.
    """

    #: The flow filter: ``"unrestricted"`` (three free ρ, whose log-determinant
    #: is the trace series or the resolvent estimator), ``"separable"``
    #: (``ρ_w = -ρ_d ρ_o``, so ``A = L_o ⊗ L_d``) or ``None`` (``A = I``).
    _flow_filter: Optional[str] = None
    #: Prefix of the spatial parameters' posterior names (``rho_d``, ``lam_d``);
    #: ``None`` for an aspatial family.
    _spatial_prefix: Optional[str] = "rho"
    #: Whether the class's NUTS model has Student-t errors (``robust=True``).
    _robust_nuts = False

    def __init__(
        self,
        y: Union[np.ndarray, pd.Series],
        X: Union[np.ndarray, pd.DataFrame],
        W,
        *,
        T: int = 1,
        col_names: Optional[list] = None,
        k: Optional[int] = None,
        priors: Optional[dict] = None,
        logdet_method: Optional[str] = None,
        nonnegative_dependence: bool = True,
        robust: bool = False,
        shared_od_covariates: Optional[bool] = None,
        effects: int = 0,
        logdet_decomp: str = "auto",
    ):
        self.priors = priors or {}
        self.logdet_method = logdet_method
        self.logdet_decomp = logdet_decomp
        self.nonnegative_dependence = nonnegative_dependence
        self.robust = bool(robust)
        if self.robust and not self._robust_nuts:
            raise NotImplementedError(
                f"robust=True is not supported for {type(self).__name__}."
            )
        self.effects = int(effects)
        if self.effects not in (0, 1, 2, 3):
            raise ValueError("effects must be one of {0,1,2,3}.")
        self._is_row_std = True  # W is assumed row-standardized
        self._idata: Optional[xr.DataTree] = None
        self._pymc_model: Optional[pm.Model] = None

        # Validate and extract the n×n weight matrix (Graph or matrix accepted).
        self._W_sparse: sp.csr_matrix = _weights_to_csr(W)
        self._n: int = self._W_sparse.shape[0]
        self._N: int = self._n * self._n

        self._T: int = int(T)
        if self._T <= 0:
            raise ValueError(f"T must be positive, got {T}.")
        if self._T == 1 and self.effects:
            raise ValueError("Fixed effects need a panel (T > 1).")
        self._is_panel = self._T > 1

        self._y_raw = self._flow_vector(y)

        # Validate X and build design matrix
        if isinstance(X, pd.DataFrame):
            if col_names is None:
                col_names = list(X.columns)
            X_arr = X.to_numpy(dtype=np.float64)
        else:
            X_arr = np.asarray(X, dtype=np.float64)
        if X_arr.ndim == 1:
            X_arr = X_arr[:, None]
        if X_arr.shape[0] != self._N * self._T:
            raise ValueError(
                f"X must have n^2*T={self._N * self._T} rows, got {X_arr.shape[0]}."
            )
        self._X_raw = X_arr

        if col_names is not None:
            self._feature_names: list[str] = list(col_names)
        elif X_arr.shape[1] == 0:
            self._feature_names = []
        else:
            self._feature_names = [f"x{i}" for i in range(X_arr.shape[1])]

        # Infer k_d and k_o (number of regional attribute columns) for effects computation.
        # Standard LeSage layout: [intercept, intra_indicator, dest_*, orig_*, intra_*, (dist)]
        if k is not None:
            self._k: int = int(k)
            self._k_d: int = int(k)
            self._k_o: int = int(k)
        else:
            self._k_d = sum(n.startswith("dest_") for n in self._feature_names)
            self._k_o = sum(n.startswith("orig_") for n in self._feature_names)
            # 0 when there are no destination columns: no effects breakdown.
            self._k = self._k_d

        # Locate β_intra slice (Thomas-Agnan & LeSage 2014, §83.4): coefficients
        # on the `intra_*` block contribute to the intraregional shock.  When
        # the design lacks these columns the intra contribution is zero.
        if self._k_d > 0:
            intra_cols = [
                i
                for i, name in enumerate(self._feature_names)
                if name.startswith("intra_")
            ]
            self._intra_idx: Optional[np.ndarray] = (
                np.asarray(intra_cols, dtype=np.int64) if intra_cols else None
            )
        else:
            self._intra_idx = None

        # Detect whether the destination and origin design blocks are
        # identical (the symmetric Xo = Xd case), on the raw design.  When
        # asymmetric the Thomas-Agnan & LeSage (2014, §83.5.2) shortcut of
        # summing dest and orig effects is not appropriate, and
        # `impacts(mode="auto")` falls back to reporting both sides separately.
        if (
            shared_od_covariates is None
            and self._k_d > 0
            and self._k_d == self._k_o
            and X_arr.shape[1] >= 2 + self._k_d + self._k_o
        ):
            dest_block = X_arr[:, 2 : 2 + self._k_d]
            orig_block = X_arr[:, 2 + self._k_d : 2 + self._k_d + self._k_o]
            self._shared_od_covariates: bool = bool(
                np.array_equal(dest_block, orig_block)
            )
        else:
            self._shared_od_covariates = (
                bool(shared_od_covariates)
                if shared_od_covariates is not None
                else (self._k_d == self._k_o)
            )

        # Demean the panel over its N origin-destination pairs.
        self._y, self._X = _demean_panel(
            self._y_raw, self._X_raw, self._N, self._T, self.effects
        )

        # Columns fixed within the demeaning groups (intercept, intra indicator
        # and log distance under pair effects; any time-invariant attribute)
        # are zero after demeaning: absorbed by the fixed effects, not
        # identified.  Drop them from the sampled design, as SpatialModel
        # does; effects read beta in the full layout via _beta_layout, with NaN
        # for absorbed columns.
        self._design_feature_names = list(self._feature_names)
        self._beta_keep: Optional[np.ndarray] = None
        if self.effects != 0:
            absorbed = _absorbed_columns(self._X, X_arr)
            if absorbed.any():
                keep = np.flatnonzero(~absorbed)
                self._beta_keep = keep
                self._X = self._X[:, keep]
                self._feature_names = [self._feature_names[j] for j in keep]
                slopes = [
                    name
                    for name, a in zip(self._design_feature_names, absorbed)
                    if a and name.startswith(("dest_", "orig_", "intra_x"))
                ]
                if slopes:
                    warnings.warn(
                        f"{slopes} do not vary within the fixed-effect groups and "
                        "are absorbed; their effects are not identified (NaN).",
                        UserWarning,
                        stacklevel=2,
                    )

        # Pre-compute logdet data for separable constraint: log|Lo⊗Ld| = n*f(ρ_d) + n*f(ρ_o).
        # ``_W_eigs`` is populated only by the eigenvalue logdet path.
        self._W_eigs: Optional[np.ndarray] = None
        self._separable_logdet_fn = None
        self._separable_logdet_numpy_fn = None
        if self._flow_filter == "unrestricted":
            if self.logdet_method not in (None, "none", *FLOW_LOGDET_METHODS):
                raise ValueError(
                    f"{type(self).__name__} takes logdet_method 'trace_series', "
                    f"'resolvent' or None, got {self.logdet_method!r}."
                )
        elif self._flow_filter == "separable" and self.logdet_method != "none":
            from ..._logdet._config import logdet_key, resolve_logdet_method

            key = logdet_key(
                self.logdet_method, self.logdet_decomp, n=self._n, W=self._W_sparse
            )
            self._separable_logdet_fn = make_flow_separable_logdet(
                self._W_sparse,
                self._n,
                method=key,
            )
            self._separable_logdet_numpy_fn = make_flow_separable_logdet_numpy(
                self._W_sparse,
                self._n,
                method=key,
            )
            # Populate ``_W_eigs`` only when the resolved key is eigenvalue
            # (auto-selection may resolve None to eigenvalue for small n).
            if resolve_logdet_method(key, n=self._n, W=self._W_sparse) == "eigenvalue":
                self._W_eigs = np.linalg.eigvals(
                    self._W_sparse.toarray().astype(np.float64)
                ).real

        # Spatial lags of the (demeaned) flows, period by period.
        # Matrix-free: W⊗W alone would hold nnz(W)² entries.
        self._Wd_y, self._Wo_y, self._Ww_y = flow_lags(
            self._W_sparse, self._y, T=self._T
        )

        # The unrestricted 3-parameter flow log-determinant is handled by the
        # resolvent-Kronecker gradient sampler; the old "traces" value method was
        # removed (it amplifies stochastic-moment noise for large directed W).
        self._traces = None

    def _flow_vector(self, y) -> np.ndarray:
        """The flows as one time-first vector of length ``N T``."""
        y_arr = np.asarray(y, dtype=np.float64)
        n, N, T = self._n, self._N, self._T
        shapes = {(T, n, n), (T, N), (N * T,)} | ({(n, n)} if T == 1 else set())
        if y_arr.shape not in shapes:
            expected = (
                "(n, n) or (n^2,)" if T == 1 else "(T, n, n), (T, n^2) or (n^2*T,)"
            )
            raise ValueError(
                f"y must have shape {expected} with n={n}, T={T}; got {y_arr.shape}."
            )
        return y_arr.reshape(-1)

    def __repr__(self) -> str:
        panel = f", T={self._T}, effects={self.effects}" if self._is_panel else ""
        separable = ", separable=True" if self._flow_filter == "separable" else ""
        return (
            f"{type(self).__name__}(n={self._n}, N={self._N}{panel}{separable}, "
            f"k={self._X.shape[1]}, features={self._feature_names})"
        )

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    def _build_pymc_model(self) -> pm.Model:
        """Construct and return the PyMC model.  Implemented by subclasses."""

    @abstractmethod
    def _compute_impacts_posterior(
        self, draws: Optional[int] = None
    ) -> dict[str, np.ndarray]:
        """Compute posterior impacts.  Implemented by subclasses."""

    # ------------------------------------------------------------------
    # Public API (fit, spatial_diagnostics_decision, etc.) inherited from
    # FlowSharedMethods — see .._mixins._flow_shared
    # ------------------------------------------------------------------

    @property
    def _W_eigs_complex(self) -> Optional[np.ndarray]:
        """Complex eigenvalues of W, or None when not pre-computed.

        Mirrors :attr:`SpatialModel._W_eigs` so that downstream samplers
        (e.g. the latent NB flow Gibbs) can request eigenvalues uniformly.
        """
        return self._W_eigs

    # ------------------------------------------------------------------
    # Fixed-effects likelihood (Lee & Yu 2010; see SpatialModel)
    # ------------------------------------------------------------------
    # The panel units are the N origin-destination pairs.  With time effects
    # and a row-standardized W, each of W_d, W_o, W_w has unit row sums, so
    # the flow filter removes the eigenvalue 1 - ρ_d - ρ_o - ρ_w
    # ((1 - ρ_d)(1 - ρ_o) when separable).

    @property
    def _fe_unit(self) -> bool:
        return self._lee_yu and self.effects in (1, 3)

    @property
    def _fe_time(self) -> bool:
        return self._lee_yu and self.effects in (2, 3)

    @property
    def _flow_jacobian_shift(self) -> float:
        if not self._fe_time or not self._W_row_standardized():
            return 0.0
        return float(self._jacobian_T)

    @property
    def _flow_fe_dims(self) -> dict:
        """Likelihood dimensions for the flow resolvent samplers."""
        return {
            "jacobian_T": self._jacobian_T,
            "n_eff": self._n_effective,
            "jacobian_shift": self._flow_jacobian_shift,
        }

    def _flow_jacobian(self, logdet, one_minus_rowsum, lib):
        """``jacobian_T·log|A|``, less the time-effects term (``lib``: np or pt)."""
        val = self._jacobian_T * logdet
        m = self._flow_jacobian_shift
        if m:
            val = val - m * lib.log(one_minus_rowsum)
        return val

    def _fe_dof_sigma(self, sigma) -> None:
        """Count ``_n_effective`` observations in a Normal over all rows (σ prior)."""
        surplus = int(np.asarray(self._y).shape[0]) - int(self._n_effective)
        if surplus:
            pm.Potential("fe_dof", surplus * pt.log(sigma))

    def _beta_layout(self, posterior) -> np.ndarray:
        """Posterior ``beta`` draws in the full design layout.

        The effects code indexes the design by position (intercept, intra
        indicator, dest, orig, intra blocks).  Columns absorbed by the fixed
        effects were not sampled and come back as NaN.
        """
        beta = posterior["beta"].values.reshape(-1, len(self._feature_names))
        n_mundlak = len(getattr(self, "_mundlak_names", ()))
        if n_mundlak:
            beta = beta[:, : beta.shape[1] - n_mundlak]
        if getattr(self, "_beta_keep", None) is None:
            return beta
        full = np.full((beta.shape[0], len(self._design_feature_names)), np.nan)
        full[:, self._beta_keep] = beta
        return full

    # ------------------------------------------------------------------
    # Public diagnostics
    # ------------------------------------------------------------------

    def impacts(
        self,
        draws: Optional[int] = None,
        return_posterior_samples: bool = False,
        ci: float = 0.95,
        mode: str = "auto",
    ) -> "pd.DataFrame | tuple[pd.DataFrame, dict[str, np.ndarray]]":
        """Summarize posterior origin/destination/intra/network/total effects.

        Wraps :meth:`_compute_impacts_posterior` to produce a tidy
        DataFrame indexed by predictor with posterior means, credible-interval
        bounds, and Bayesian p-values for each effect type (origin,
        destination, intra, network, total).  Following Thomas-Agnan & LeSage
        (2014, §83.5.2), when destination and origin design blocks differ the
        decomposition is reported separately for shocks applied to each side.

        Parameters
        ----------
        draws : int, optional
            Maximum number of posterior draws to use.  Defaults to all.
        return_posterior_samples : bool, default False
            If True, also return the underlying posterior-draw arrays.
        ci : float, default 0.95
            Credible-interval coverage.
        mode : {"auto", "combined", "separate"}, default "auto"
            Controls whether destination- and origin-side effects are summed
            or reported separately.  ``"auto"`` collapses to combined when
            the destination and origin design blocks are identical
            (``self._shared_od_covariates``) and reports both sides otherwise.
            ``"combined"`` always sums; ``"separate"`` always reports both.

        Returns
        -------
        pandas.DataFrame, or (DataFrame, dict)
            Long-format summary indexed by ``(predictor, side, effect)`` where
            ``side`` is one of ``"combined"``, ``"dest"``, ``"orig"``.
        """
        from ...diagnostics.impacts import _compute_bayesian_pvalue

        if self._idata is None:
            raise RuntimeError("Model has not been fit yet.  Call fit() first.")
        if self._k == 0:
            raise RuntimeError(
                "Cannot compute impacts: no `dest_*` columns detected "
                "in the design matrix.  Pass `k=` explicitly when constructing "
                "the model."
            )
        if mode not in {"auto", "combined", "separate"}:
            raise ValueError(
                f"mode must be 'auto', 'combined', or 'separate'; got {mode!r}."
            )

        posterior = self._compute_impacts_posterior(draws=draws)

        if mode == "auto":
            effective_mode = "combined" if self._shared_od_covariates else "separate"
        else:
            effective_mode = mode

        if effective_mode == "combined":
            display = [("combined", eff) for eff in _EFFECT_KEYS]
        else:
            display = [(side, eff) for side in ("dest", "orig") for eff in _EFFECT_KEYS]

        # Predictor names: dest_* and orig_* labels stripped of the prefix, from
        # the full design (absorbed columns included).
        dest_feature_names = [
            name[len("dest_") :]
            for name in self._design_feature_names
            if name.startswith("dest_")
        ][: self._k_d]
        if len(dest_feature_names) != self._k_d:
            dest_feature_names = [f"x{i}" for i in range(self._k_d)]

        orig_feature_names = [
            name[len("orig_") :]
            for name in self._design_feature_names
            if name.startswith("orig_")
        ][: self._k_o]
        if len(orig_feature_names) != self._k_o:
            orig_feature_names = [f"y{i}" for i in range(self._k_o)]

        # For combined mode: when k_d == k_o, combined effects are the sum
        # of dest and orig (same variables), so use dest names.
        # When k_d != k_o, combined effects are concatenated (different variables).
        if self._k_d == self._k_o:
            feature_names = dest_feature_names
        else:
            feature_names = dest_feature_names + orig_feature_names

        alpha = (1.0 - ci) / 2.0
        rows = []
        for side, effect_name in display:
            key = effect_name if side == "combined" else f"{side}_{effect_name}"
            samples = posterior[key]
            means = samples.mean(axis=0)
            lower = np.quantile(samples, alpha, axis=0)
            upper = np.quantile(samples, 1.0 - alpha, axis=0)
            pvals = _compute_bayesian_pvalue(samples)
            if side == "combined":
                fnames = feature_names
            elif side == "dest":
                fnames = dest_feature_names
            else:
                fnames = orig_feature_names
            for j, fname in enumerate(fnames):
                rows.append(
                    {
                        "predictor": fname,
                        "side": side,
                        "effect": effect_name,
                        "mean": float(means[j]),
                        "ci_lower": float(lower[j]),
                        "ci_upper": float(upper[j]),
                        "bayes_pvalue": float(pvals[j]),
                    }
                )

        df = pd.DataFrame(rows).set_index(["predictor", "side", "effect"])
        if return_posterior_samples:
            return df, posterior
        return df

    def _simulate_y_rep(
        self,
        rho_d: float,
        rho_o: float,
        rho_w: float,
        beta: np.ndarray,
        sigma: Optional[float],
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Draw a single posterior-predictive replicate of every period.

        Default Gaussian implementation: :math:`y_{rep,t} = A^{-1}(X_t \\beta
        + \\sigma\\varepsilon_t)` for each period ``t``, with one sparse
        :math:`LU` factorization of :math:`A` reused across periods.
        Subclasses (OLS, SEM and count variants) override this method.
        """
        N, T = self._N, self._T
        rhs = (self._X @ beta).reshape(T, N).T  # (N, T)
        if sigma is not None:
            rhs = rhs + rng.normal(scale=float(sigma), size=(N, T))
        # Cached symbolic analysis: A = I - ρ_d W_d - ρ_o W_o - ρ_w W_w
        # shares its sparsity pattern across draws, so sparsax reuses one
        # fill-reducing analysis; scipy fallback still avoids re-symbolic work.
        out = self._solve_A(rho_d, rho_o, rho_w, rhs)  # (N, T)
        return out.T.reshape(-1)  # back to time-first stacked vector

    def posterior_predictive(
        self,
        n_draws: Optional[int] = None,
        random_seed: Optional[int] = None,
    ) -> np.ndarray:
        """Draw posterior-predictive samples ``y_rep``.

        For each (subsampled) posterior draw, simulates every period's flows
        from the implied data-generating process (:meth:`_simulate_y_rep`).

        Parameters
        ----------
        n_draws : int, optional
            Number of posterior draws to use.  Defaults to all.
        random_seed : int, optional
            Seed for the posterior-predictive sampler.

        Returns
        -------
        np.ndarray
            Array of shape ``(n_draws, N * T)`` with posterior-predictive
            flows in time-first stacked order.
        """
        if self._idata is None:
            raise RuntimeError("Model has not been fit yet.  Call fit() first.")

        post = self._idata.posterior
        beta_draws = post["beta"].values.reshape(-1, len(self._feature_names))
        sigma_draws = (
            post["sigma"].values.reshape(-1) if "sigma" in post.data_vars else None
        )
        total = beta_draws.shape[0]
        if self._spatial_prefix is None:
            spatial = np.zeros((total, 3))
        else:
            spatial = np.column_stack(
                [
                    post[f"{self._spatial_prefix}_{k}"].values.reshape(-1)
                    for k in ("d", "o", "w")
                ]
            )
        if n_draws is not None:
            total = min(int(n_draws), total)

        rng = np.random.default_rng(random_seed)
        out = np.empty((total, self._N * self._T), dtype=np.float64)
        for g in range(total):
            sigma_g = float(sigma_draws[g]) if sigma_draws is not None else None
            out[g] = self._simulate_y_rep(*spatial[g], beta_draws[g], sigma_g, rng)
        return out

    # ------------------------------------------------------------------
    # Internal effects helpers
    # ------------------------------------------------------------------

    def _compute_flow_effects_from_draws(
        self,
        rho_d_draws: np.ndarray,
        rho_o_draws: np.ndarray,
        rho_w_draws: np.ndarray,
        beta_draws: np.ndarray,
        draws: Optional[int] = None,
    ) -> dict[str, np.ndarray]:
        """LeSage flow effects from posterior draws of ``ρ`` and ``β`` (full layout).

        Effects are those of one period's ``n^2 x n^2`` system, which is
        time-invariant under static panel parameters; see
        :func:`~neighbayes.models.flow._compute_flow_effects` for the
        decomposition, computed from W-only moments with no ``n^2``-sized
        arrays.
        """
        k_d = self._k_d
        k_o = self._k_o
        dest_start = 2
        orig_start = 2 + k_d
        intra_start = 2 + k_d + k_o
        has_intra = (
            self._intra_idx is not None and beta_draws.shape[1] >= intra_start + k_d
        )
        if draws is not None:
            keep = min(draws, len(rho_d_draws))
            rho_d_draws = rho_d_draws[:keep]
            rho_o_draws = rho_o_draws[:keep]
            rho_w_draws = rho_w_draws[:keep]
            beta_draws = beta_draws[:keep]
        return self._flow_effects_for_draws(
            rho_d_draws,
            rho_o_draws,
            rho_w_draws,
            beta_draws[:, dest_start : dest_start + k_d],
            beta_draws[:, orig_start : orig_start + k_o],
            beta_draws[:, intra_start : intra_start + k_d] if has_intra else None,
        )

    def _compute_flow_effects_kron(
        self,
        rho_d_draws: np.ndarray,
        rho_o_draws: np.ndarray,
        beta_draws: np.ndarray,
        draws: Optional[int] = None,
    ) -> dict[str, np.ndarray]:
        """:meth:`_compute_flow_effects_from_draws` under ``ρ_w = −ρ_d ρ_o``."""
        return self._compute_flow_effects_from_draws(
            rho_d_draws,
            rho_o_draws,
            -rho_d_draws * rho_o_draws,
            beta_draws,
            draws=draws,
        )

    def _spatial_draws(self, posterior, keys: str = "dow") -> list[np.ndarray]:
        """The flattened draws of the spatial parameters ``keys`` (``d``, ``o``, ``w``)."""
        return [
            np.asarray(
                posterior[f"{self._spatial_prefix}_{k}"].values.reshape(-1),
                dtype=np.float64,
            )
            for k in keys
        ]

    def _require_idata(self):
        if self._idata is None:
            raise RuntimeError("Model has not been fit yet.  Call fit() first.")
        return self._idata.posterior

    def _effects_unrestricted(self, draws: Optional[int] = None) -> dict:
        """LeSage effects under the unrestricted filter."""
        post = self._require_idata()
        return self._compute_flow_effects_from_draws(
            *self._spatial_draws(post), self._beta_layout(post), draws=draws
        )

    def _effects_separable(self, draws: Optional[int] = None) -> dict:
        """LeSage effects under the separable filter."""
        post = self._require_idata()
        return self._compute_flow_effects_kron(
            *self._spatial_draws(post, "do"), self._beta_layout(post), draws=draws
        )

    def _effects_aspatial(self, draws: Optional[int] = None) -> dict:
        """Closed-form effects of a mean ``X β`` (no ``X``-mediated spillovers)."""
        post = self._require_idata()
        return _compute_ols_flow_effects(
            self._idata,
            beta_draws=self._beta_layout(post),
            n=self._n,
            k_d=self._k_d,
            k_o=self._k_o,
            feature_names=self._design_feature_names,
            intra_idx=self._intra_idx,
            draws=draws,
        )

    def _trace_logdet(self):
        """Exact ``log|A(ρ)|`` value-and-gradient on the ``n²`` flow system."""
        if getattr(self, "_trace_logdet_obj", None) is None:
            from ..._logdet._flow_kron_traces import FlowKronTraceLogdet

            self._trace_logdet_obj = FlowKronTraceLogdet(self._W_sparse)
        return self._trace_logdet_obj

    def _jacobian_unrestricted(self, posterior) -> np.ndarray:
        """Per-draw Jacobian term of the unrestricted filter (exact trace series)."""
        ld = self._trace_logdet()
        d, o, w = self._spatial_draws(posterior)
        logdet = np.array([ld(*r)[0] for r in zip(d, o, w)])
        return self._flow_jacobian(logdet, 1 - d - o - w, np)

    def _jacobian_separable(self, posterior) -> np.ndarray:
        """Per-draw Jacobian term of the separable filter, ``n log|L_d| + n log|L_o|``."""
        if self._separable_logdet_numpy_fn is None:
            raise RuntimeError(
                "Missing separable numeric logdet evaluator. "
                "Initialize with a separable logdet_method (None/auto, "
                "eigenvalue, chebyshev, cheb_cholesky, aaa, or cheb_stochastic)."
            )
        d, o = self._spatial_draws(posterior, "do")
        return self._flow_jacobian(
            self._separable_logdet_numpy_fn(d, o), (1 - d) * (1 - o), np
        )


class _Unrestricted:
    """The unrestricted filter ``A = I - ρ_d W_d - ρ_o W_o - ρ_w W_w``."""

    _flow_filter = "unrestricted"

    def _compute_impacts_posterior(self, draws: Optional[int] = None) -> dict:
        return self._effects_unrestricted(draws)


class _Separable:
    """The separable filter ``A = L_o ⊗ L_d`` (``ρ_w = -ρ_d ρ_o``)."""

    _flow_filter = "separable"

    def _compute_impacts_posterior(self, draws: Optional[int] = None) -> dict:
        return self._effects_separable(draws)

    def _compute_jacobian_log_det(self, posterior) -> np.ndarray:
        return self._jacobian_separable(posterior)


class _Aspatial:
    """No filter, ``A = I``: the gravity baselines."""

    _flow_filter = None
    _spatial_prefix = None

    def _compute_impacts_posterior(self, draws: Optional[int] = None) -> dict:
        return self._effects_aspatial(draws)

    def _simulate_y_rep(self, rho_d, rho_o, rho_w, beta, sigma, rng) -> np.ndarray:
        """``y_rep = X β + σ ε`` (every period)."""
        Xb = self._X @ beta
        if sigma is None:
            return Xb
        return Xb + rng.normal(scale=float(sigma), size=Xb.shape[0])


def _compute_ols_flow_effects(
    idata: "xr.DataTree",
    *,
    n: int,
    k_d: int,
    k_o: int,
    feature_names: list[str],
    intra_idx: Optional[np.ndarray],
    draws: Optional[int],
    beta_draws: Optional[np.ndarray] = None,
) -> dict[str, np.ndarray]:
    """Closed-form Thomas-Agnan & LeSage (2014, Table 83.1) effects.

    The effects of :class:`OLSFlow` and :class:`SEMFlow`, whose mean is
    :math:`\\mathbb{E}[y] = X\\beta` (no :math:`X`-mediated spillovers).
    """
    if beta_draws is None:
        beta_draws = idata.posterior["beta"].values.reshape(-1, len(feature_names))

    dest_start = 2
    orig_start = 2 + k_d
    intra_start = 2 + k_d + k_o
    has_intra = intra_idx is not None and beta_draws.shape[1] >= intra_start + k_d

    n_draws_total = beta_draws.shape[0]
    if draws is not None:
        n_draws_total = min(draws, n_draws_total)
        beta_draws = beta_draws[:n_draws_total]

    bd = beta_draws[:, dest_start : dest_start + k_d]
    bo = beta_draws[:, orig_start : orig_start + k_o]
    bi = (
        beta_draws[:, intra_start : intra_start + k_d]
        if has_intra
        else np.zeros((n_draws_total, k_d), dtype=np.float64)
    )

    zeros_d = np.zeros_like(bd)
    zeros_o = np.zeros_like(bo)
    out: dict[str, np.ndarray] = {}
    out["dest_total"] = bd + bi / n
    out["dest_destination"] = bd * (n - 1) / n
    out["dest_intraunit"] = (bd + bi) / n
    out["dest_origin"] = zeros_d.copy()
    out["dest_network"] = zeros_d.copy()

    out["orig_total"] = bo
    out["orig_origin"] = bo * (n - 1) / n
    out["orig_intraunit"] = bo / n
    out["orig_destination"] = zeros_o.copy()
    out["orig_network"] = zeros_o.copy()

    if k_d == k_o:
        for eff in _EFFECT_KEYS:
            out[eff] = out[f"dest_{eff}"] + out[f"orig_{eff}"]
    else:
        for eff in _EFFECT_KEYS:
            out[eff] = np.concatenate([out[f"dest_{eff}"], out[f"orig_{eff}"]], axis=1)

    return out


# ---------------------------------------------------------------------------
# Negative binomial SAR and aspatial flows
# ---------------------------------------------------------------------------


class _NegBinFlowMixin:
    """Shared ``fit`` dispatch for Negative-Binomial flow models.

    The NB flow classes share a single ``fit`` wrapper — the reduced-form
    Pólya–Gamma Gibbs sampler by default, or PyMC NUTS on the exact count
    likelihood via ``sampler="nuts"`` (much slower).  Only the
    per-class :meth:`_fit_gibbs` (unrestricted 3-ρ vs. separable 2-ρ
    reduced-form sampler) differs, so it stays on each subclass.  The mixin is
    listed first in the bases so its ``fit`` wins and ``super().fit`` resolves
    to the Gaussian :class:`FlowModel.fit` NUTS path.
    """

    #: Whether this model has a JAX Gibbs kernel (see ``backend``).
    _gibbs_jax = True

    def fit(
        self,
        draws: int = 2000,
        tune: int = 1000,
        chains: int = 4,
        random_seed: Optional[int] = None,
        sampler: str = "gibbs",
        backend: str | None = None,
        store_lambda: bool = False,
        idata_kwargs: Optional[dict] = None,
        progressbar: bool = True,
        attach_log_abs_det: bool = True,
        n_jobs: int = -1,
        **sample_kwargs,
    ) -> xr.DataTree:
        """Draw samples from the posterior.

        Parameters
        ----------
        draws : int, default 2000
            Number of posterior samples per chain (after tuning).
        tune : int, default 1000
            Number of tuning (warm-up) steps per chain.
        chains : int, default 4
            Number of parallel chains.
        random_seed : int, optional
            Seed for reproducibility.
        sampler : {"gibbs", "nuts"}, default "gibbs"
            Sampling method: ``"gibbs"`` (default) for the reduced-form
            Pólya–Gamma Gibbs sampler, or ``"nuts"`` for PyMC NUTS on the
            exact count likelihood (much slower).
        backend : str, optional
            What executes the sampler.  For Gibbs, ``"auto"`` (the default),
            ``"jax"`` or ``"numpy"``: ``"jax"`` compiles the sweep with JAX and
            sparsax sparse LU solves, chains on threads; ``"numpy"`` uses the
            host CHOLMOD/KLU path; ``"auto"`` takes JAX when it is installed
            and the model has a JAX kernel (the aspatial ``NegBinFlow`` does
            not), else NumPy.  For NUTS, ``"pymc"`` (the default),
            ``"nutpie"``, ``"numpyro"`` or ``"blackjax"``.
        store_lambda : bool, default False
            If True, include the high-dimensional fitted mean ``lambda`` in the
            stored posterior (NUTS only).
        idata_kwargs : dict, optional
            ``{"log_likelihood": True}`` stores the pointwise log-likelihood
            (one value per draw, chain, and flow) for ``az.loo``,
            on either sampler.  Off by default, as in PyMC.  For NUTS the dict
            is also forwarded to ``pm.sample``.
        progressbar : bool, default True
            Show progress bar during sampling.
        attach_log_abs_det : bool, default True
            If True, record the per-draw spatial-filter Jacobian ``log|A(ρ)|`` in
            ``idata.sample_stats["log_abs_det"]`` (a diagnostic — it is *not* folded
            into the count model's ``log_likelihood``).  Computed with the resolvent
            value estimator; set ``False`` to skip its per-draw cost at very large
            ``N``.
        **sample_kwargs
            Additional keyword arguments forwarded to ``pm.sample`` (NUTS
            only).  Pass ``target_accept=0.95`` to adjust the NUTS acceptance
            rate.

        Returns
        -------
        xarray.DataTree
        """
        backend = select_backend(sampler, backend, sample_kwargs)
        if sampler == "gibbs":
            gibbs_backend = self._resolve_gibbs_backend(backend, jax=self._gibbs_jax)
            idata = self._fit_gibbs(
                draws=draws,
                tune=tune,
                chains=chains,
                random_seed=random_seed,
                progressbar=progressbar,
                n_jobs=n_jobs,
                gibbs_backend=gibbs_backend,
                sample_kwargs=sample_kwargs,
                log_likelihood=bool((idata_kwargs or {}).get("log_likelihood", False)),
            )
        else:
            # Call FlowModel.fit explicitly: SARFlow's own ``fit`` is the
            # Gaussian resolvent sampler, which must never see count data.
            idata = FlowModel.fit(
                self,
                draws=draws,
                tune=tune,
                chains=chains,
                random_seed=random_seed,
                store_lambda=store_lambda,
                idata_kwargs=idata_kwargs,
                progressbar=progressbar,
                backend=backend,
                **sample_kwargs,
            )
        if attach_log_abs_det:
            self._attach_flow_log_abs_det(idata)
        return idata


class SARNegBinFlow(_NegBinFlowMixin, _Unrestricted, FlowModel):
    r"""Bayesian SAR flow model with NB2 observation noise.

    This class extends :class:`SARFlow` with a Negative Binomial likelihood:

    .. math::

        y_{ij} \sim \operatorname{NegBin}(\mu_{ij}, \alpha),

    where ``alpha`` is an overdispersion parameter sampled from a
    HalfNormal prior.
    """

    _dispatch = classmethod(flow)

    def __init__(self, y, X, W, **kwargs):
        y_arr = np.asarray(y)
        if not np.issubdtype(y_arr.dtype, np.integer):
            y_rounded = np.round(y_arr).astype(np.int64)
            if not np.allclose(y_arr, y_rounded):
                raise ValueError(
                    "SARNegBinFlow requires integer-valued "
                    f"observations; got dtype {y_arr.dtype} with non-integer "
                    "values."
                )
            y_arr = y_rounded
        if np.any(y_arr < 0):
            raise ValueError(
                "SARNegBinFlow requires non-negative integer observations."
            )
        super().__init__(y_arr.astype(np.float64), X, W, **kwargs)
        self._y_int_vec: np.ndarray = y_arr.ravel().astype(np.int64)

    def _compute_jacobian_log_det(self, posterior) -> Optional[np.ndarray]:
        return None

    def _build_pymc_model(self) -> pm.Model:
        from ..._ops import SparseFlowSolveOp

        pv = self._flow_count_priors()
        beta_mu, beta_sigma = pv["beta_mu"], pv["beta_sigma"]
        alpha_sigma, alpha_nu = pv["alpha_sigma"], pv["alpha_nu"]

        X_t = pt.as_tensor_variable(self._X.astype(np.float64))

        with pm.Model(coords=self._model_coords()) as model:
            if self.nonnegative_dependence:
                rho_simplex = pm.Dirichlet("rho_simplex", a=np.ones(4))
                rho_d = pm.Deterministic("rho_d", rho_simplex[0])
                rho_o = pm.Deterministic("rho_o", rho_simplex[1])
                rho_w = pm.Deterministic("rho_w", rho_simplex[2])
            else:
                rho_lower = self.priors.get("rho_lower", -1.0)
                rho_upper = self.priors.get("rho_upper", 1.0)
                rho_d = pm.Uniform("rho_d", lower=rho_lower, upper=rho_upper)
                rho_o = pm.Uniform("rho_o", lower=rho_lower, upper=rho_upper)
                rho_w = pm.Uniform("rho_w", lower=rho_lower, upper=rho_upper)
                slack = 1.0 - rho_d - rho_o - rho_w
                pm.Potential(
                    "stability",
                    pt.switch(slack > 0.0, 0.0, -1e6 * slack**2),
                )

            beta = pm.Normal("beta", mu=beta_mu, sigma=beta_sigma, dims="coefficient")
            alpha = nb_alpha_rv(pv["alpha_fixed"], alpha_nu, alpha_sigma)

            Xb = pt.dot(X_t, beta)
            solve_op = SparseFlowSolveOp(self._Wd, self._Wo, self._Ww)
            eta = solve_op(rho_d, rho_o, rho_w, Xb)
            lam = pm.Deterministic("lambda", pt.exp(eta))

            pm.NegativeBinomial("obs", mu=lam, alpha=alpha, observed=self._y_int_vec)

            # No |A| change-of-variables Jacobian for the count likelihood: the
            # NB mean is η = A⁻¹Xβ and y is modelled directly, so the spatial
            # log-determinant does not enter (adding it biases β).

        return model

    def posterior_predictive(
        self,
        n_draws: Optional[int] = None,
        random_seed: Optional[int] = None,
    ) -> np.ndarray:
        """Draw posterior-predictive flow counts for NB SAR flow model."""
        if self._idata is None:
            raise RuntimeError("Model has not been fit yet.  Call fit() first.")

        post = self._idata.posterior
        beta_draws = post["beta"].values.reshape(-1, len(self._feature_names))
        rho_d_draws = post["rho_d"].values.reshape(-1)
        rho_o_draws = post["rho_o"].values.reshape(-1)
        rho_w_draws = post["rho_w"].values.reshape(-1)
        alpha_draws = post["alpha"].values.reshape(-1)

        total = beta_draws.shape[0]
        if n_draws is not None:
            total = min(int(n_draws), total)
            beta_draws = beta_draws[:total]
            rho_d_draws = rho_d_draws[:total]
            rho_o_draws = rho_o_draws[:total]
            rho_w_draws = rho_w_draws[:total]
            alpha_draws = alpha_draws[:total]

        rng = np.random.default_rng(random_seed)
        out = np.empty((total, self._N), dtype=np.float64)
        for g in range(total):
            eta = self._solve_A(
                rho_d_draws[g], rho_o_draws[g], rho_w_draws[g], self._X @ beta_draws[g]
            )
            lam = np.exp(np.clip(eta, -50.0, 50.0))
            alpha = float(alpha_draws[g])
            p = alpha / (alpha + lam)
            out[g] = rng.negative_binomial(alpha, p).astype(np.float64)
        return out

    def _fit_gibbs(
        self,
        draws: int = 2000,
        tune: int = 1000,
        chains: int = 4,
        random_seed: Optional[int] = None,
        progressbar: bool = True,
        n_jobs: int = -1,
        gibbs_backend: str = "numpy",
        sample_kwargs: dict[str, Any] | None = None,
        log_likelihood: bool = False,
    ) -> xr.DataTree:
        """Sample posterior via reduced-form PG-Gibbs (unrestricted 3-ρ)."""
        from ._nb_gibbs import run_negbin_flow_gibbs

        return run_negbin_flow_gibbs(
            self,
            separable=False,
            gibbs_backend=gibbs_backend,
            model_type="nb_sar_flow",
            omega_size=self._N,
            draws=draws,
            tune=tune,
            chains=chains,
            random_seed=random_seed,
            progressbar=progressbar,
            n_jobs=n_jobs,
            log_likelihood=log_likelihood,
        )


class SARNegBinFlowSeparable(_NegBinFlowMixin, _Separable, FlowModel):
    """Separable SAR flow model with NB2 observation noise."""

    def __init__(self, y, X, W, **kwargs):
        y_arr = np.asarray(y)
        if not np.issubdtype(y_arr.dtype, np.integer):
            y_rounded = np.round(y_arr).astype(np.int64)
            if not np.allclose(y_arr, y_rounded):
                raise ValueError(
                    "SARNegBinFlowSeparable requires integer-valued "
                    f"observations; got dtype {y_arr.dtype} with non-integer "
                    "values."
                )
            y_arr = y_rounded
        if np.any(y_arr < 0):
            raise ValueError(
                "SARNegBinFlowSeparable requires non-negative integer observations."
            )
        super().__init__(y_arr.astype(np.float64), X, W, **kwargs)
        self._y_int_vec: np.ndarray = y_arr.ravel().astype(np.int64)

    def _compute_jacobian_log_det(self, posterior) -> Optional[np.ndarray]:
        return None

    def _build_pymc_model(self) -> pm.Model:
        from ..._ops import KroneckerFlowSolveOp

        pv = self._flow_count_priors()
        beta_mu, beta_sigma = pv["beta_mu"], pv["beta_sigma"]
        alpha_sigma, alpha_nu = pv["alpha_sigma"], pv["alpha_nu"]
        rho_lower = self.priors.get("rho_lower", -0.999)
        rho_upper = self.priors.get("rho_upper", 0.999)

        n = self._n
        if self._separable_logdet_fn is None:
            raise RuntimeError(
                "SARNegBinFlowSeparable requires precomputed logdet data; "
                "initialize with a separable logdet_method (None/auto, "
                "eigenvalue, chebyshev, cheb_cholesky, aaa, or cheb_stochastic)"
            )
        X_t = pt.as_tensor_variable(self._X.astype(np.float64))

        with pm.Model(coords=self._model_coords()) as model:
            rho_d = pm.Uniform("rho_d", lower=rho_lower, upper=rho_upper)
            rho_o = pm.Uniform("rho_o", lower=rho_lower, upper=rho_upper)
            pm.Deterministic("rho_w", -rho_d * rho_o)

            beta = pm.Normal("beta", mu=beta_mu, sigma=beta_sigma, dims="coefficient")
            alpha = nb_alpha_rv(pv["alpha_fixed"], alpha_nu, alpha_sigma)

            Xb = pt.dot(X_t, beta)
            solve_op = KroneckerFlowSolveOp(self._W_sparse, n)
            eta = solve_op(rho_d, rho_o, Xb)
            lam = pm.Deterministic("lambda", pt.exp(eta))

            pm.NegativeBinomial("obs", mu=lam, alpha=alpha, observed=self._y_int_vec)

            # No |A| change-of-variables Jacobian for the count likelihood:
            # the NB mean is η = A⁻¹Xβ and y is modelled directly, so the
            # spatial filter enters only through the mean.  (The Gaussian
            # separable model keeps the Jacobian; copying it here biases
            # ρ toward the negative-logdet region.)

        return model

    def posterior_predictive(
        self,
        n_draws: Optional[int] = None,
        random_seed: Optional[int] = None,
    ) -> np.ndarray:
        """Draw posterior-predictive flow counts for separable NB SAR flow."""
        if self._idata is None:
            raise RuntimeError("Model has not been fit yet.  Call fit() first.")

        post = self._idata.posterior
        beta_draws = post["beta"].values.reshape(-1, len(self._feature_names))
        rho_d_draws = post["rho_d"].values.reshape(-1)
        rho_o_draws = post["rho_o"].values.reshape(-1)
        alpha_draws = post["alpha"].values.reshape(-1)

        total = beta_draws.shape[0]
        if n_draws is not None:
            total = min(int(n_draws), total)
            beta_draws = beta_draws[:total]
            rho_d_draws = rho_d_draws[:total]
            rho_o_draws = rho_o_draws[:total]
            alpha_draws = alpha_draws[:total]

        rng = np.random.default_rng(random_seed)
        out = np.empty((total, self._N), dtype=np.float64)
        n = self._n
        I_n = sp.eye(n, format="csr", dtype=np.float64)
        for g in range(total):
            Ld = I_n - float(rho_d_draws[g]) * self._W_sparse
            Lo = I_n - float(rho_o_draws[g]) * self._W_sparse
            eta = kron_solve_vec(Lo, Ld, self._X @ beta_draws[g], n)
            lam = np.exp(np.clip(eta, -50.0, 50.0))
            alpha = float(alpha_draws[g])
            p = alpha / (alpha + lam)
            out[g] = rng.negative_binomial(alpha, p).astype(np.float64)
        return out

    def _fit_gibbs(
        self,
        draws: int = 2000,
        tune: int = 1000,
        chains: int = 4,
        random_seed: Optional[int] = None,
        progressbar: bool = True,
        n_jobs: int = -1,
        gibbs_backend: str = "numpy",
        sample_kwargs: dict[str, Any] | None = None,
        log_likelihood: bool = False,
    ) -> xr.DataTree:
        """Sample posterior via reduced-form PG-Gibbs (separable 2-ρ)."""
        from ._nb_gibbs import run_negbin_flow_gibbs

        return run_negbin_flow_gibbs(
            self,
            separable=True,
            gibbs_backend=gibbs_backend,
            model_type="nb_sar_flow_sep",
            omega_size=self._N,
            draws=draws,
            tune=tune,
            chains=chains,
            random_seed=random_seed,
            progressbar=progressbar,
            n_jobs=n_jobs,
            log_likelihood=log_likelihood,
        )


class NegBinFlow(_NegBinFlowMixin, _Aspatial, FlowModel):
    """Aspatial OD-flow Negative Binomial gravity baseline."""

    _dispatch = classmethod(flow)

    _gibbs_jax = False  # NumPy Gibbs only

    def __init__(self, y, X, W, **kwargs):
        y_arr = np.asarray(y)
        if not np.issubdtype(y_arr.dtype, np.integer):
            y_rounded = np.round(y_arr).astype(np.int64)
            if not np.allclose(y_arr, y_rounded):
                raise ValueError(
                    "NegBinFlow requires integer-valued "
                    f"observations; got dtype {y_arr.dtype} with non-integer "
                    "values."
                )
            y_arr = y_rounded
        if np.any(y_arr < 0):
            raise ValueError("NegBinFlow requires non-negative integer observations.")
        super().__init__(y_arr.astype(np.float64), X, W, **kwargs)
        self._y_int_vec: np.ndarray = y_arr.ravel().astype(np.int64)

    def _compute_jacobian_log_det(self, posterior) -> Optional[np.ndarray]:
        return None

    def _build_pymc_model(self) -> pm.Model:
        pv = self._flow_count_priors()
        beta_mu, beta_sigma = pv["beta_mu"], pv["beta_sigma"]
        alpha_sigma, alpha_nu = pv["alpha_sigma"], pv["alpha_nu"]

        X_t = pt.as_tensor_variable(self._X.astype(np.float64))

        with pm.Model(coords=self._model_coords()) as model:
            beta = pm.Normal("beta", mu=beta_mu, sigma=beta_sigma, dims="coefficient")
            alpha = nb_alpha_rv(pv["alpha_fixed"], alpha_nu, alpha_sigma)
            eta = pt.dot(X_t, beta)
            lam = pm.Deterministic("lambda", pt.exp(eta))
            pm.NegativeBinomial("obs", mu=lam, alpha=alpha, observed=self._y_int_vec)

        return model

    def posterior_predictive(
        self,
        n_draws: Optional[int] = None,
        random_seed: Optional[int] = None,
    ) -> np.ndarray:
        """Draw posterior-predictive flow counts for NB gravity baseline."""
        if self._idata is None:
            raise RuntimeError("Model has not been fit yet.  Call fit() first.")

        post = self._idata.posterior
        beta_draws = post["beta"].values.reshape(-1, len(self._feature_names))
        alpha_draws = post["alpha"].values.reshape(-1)

        total = beta_draws.shape[0]
        if n_draws is not None:
            total = min(int(n_draws), total)
            beta_draws = beta_draws[:total]
            alpha_draws = alpha_draws[:total]

        rng = np.random.default_rng(random_seed)
        out = np.empty((total, self._N), dtype=np.float64)
        for g in range(total):
            eta = self._X @ beta_draws[g]
            lam = np.exp(np.clip(eta, -50.0, 50.0))
            alpha = float(alpha_draws[g])
            p = alpha / (alpha + lam)
            out[g] = rng.negative_binomial(alpha, p).astype(np.float64)
        return out

    def _fit_gibbs(
        self,
        draws: int = 2000,
        tune: int = 1000,
        chains: int = 4,
        random_seed: Optional[int] = None,
        progressbar: bool = True,
        n_jobs: int = -1,
        gibbs_backend: str = "numpy",
        sample_kwargs: dict[str, Any] | None = None,
        log_likelihood: bool = False,
    ) -> xr.DataTree:
        """Sample posterior via aspatial PG-Gibbs (no spatial parameters).

        Three blocks per sweep: ω (Pólya–Gamma), β (conjugate normal),
        α (slice on log(α)).
        """
        from ...samplers._utils._idata import gibbs_to_inference_data
        from ...samplers.gaussian._chain_runner import run_chains
        from ...samplers.negbin._core import GibbsState
        from ...samplers.negbin_reduced._core import (
            ReducedGibbsPriors,
            _nb_loglik_pointwise,
            _sample_alpha,
            _sample_beta,
            _sample_omega,
        )

        X = self._X
        y = self._y_int_vec.astype(np.float64)
        N, k = X.shape

        # --- Build priors ---
        pv = self._flow_count_priors()
        priors = ReducedGibbsPriors(
            beta_mu=pv["beta_mu"],
            beta_sigma=pv["beta_sigma"],
            alpha_sigma=pv["alpha_sigma"],
            alpha_nu=pv["alpha_nu"],
            alpha_fixed=pv["alpha_fixed"],
            rho_lower=-0.999,
            rho_upper=0.999,
        )

        # --- Aspatial chain runner ---
        def _run_chain_aspatial(
            y: np.ndarray,
            X: np.ndarray,
            priors: ReducedGibbsPriors,
            beta0: np.ndarray,
            alpha0: float,
            draws: int,
            tune: int,
            rng: np.random.Generator | None = None,
            chain_id: int = 0,
            progress_manager: object | None = None,
        ) -> dict[str, np.ndarray]:
            if rng is None:
                rng = np.random.default_rng()
            total = tune + draws
            n_keep = draws
            beta_samples = np.empty((n_keep, k), dtype=np.float64)
            alpha_samples = np.empty(n_keep, dtype=np.float64)
            log_lik_samples = (
                np.empty((n_keep, N), dtype=np.float64) if log_likelihood else None
            )

            beta = beta0.copy()
            alpha = alpha0
            omega = np.ones(N, dtype=np.float64) * 0.5

            for i in range(total):
                eta = X @ beta
                psi = eta - np.log(alpha)
                omega = _sample_omega(y, alpha, psi, rng=rng)
                # X̃ = X (no spatial solve)
                beta = _sample_beta(
                    beta_current=beta,
                    Xtilde=X,
                    omega=omega,
                    y=y,
                    alpha=alpha,
                    priors=priors,
                    rng=rng,
                    rho=0.0,
                    intercept_col=-1,
                )
                eta = X @ beta
                state = GibbsState(
                    eta=eta,
                    beta=beta,
                    rho=0.0,
                    alpha=alpha,
                    omega=omega,
                )
                alpha = _sample_alpha(state, y, priors, rng=rng)

                if i >= tune:
                    idx = i - tune
                    if idx < n_keep:
                        beta_samples[idx] = beta
                        alpha_samples[idx] = alpha
                        if log_likelihood:
                            log_lik_samples[idx] = _nb_loglik_pointwise(y, eta, alpha)

                if progress_manager is not None:
                    progress_manager.update(chain_id, i, tuning=i < tune)

            return {
                "beta": beta_samples,
                "alpha": alpha_samples,
                "log_lik": log_lik_samples,
            }

        # --- Chain function ---
        def _chain_fn(chain_id, seed, progress_manager=None, chain_id_kw=0):
            rng = np.random.default_rng(seed)
            beta0 = rng.normal(0.0, 0.1, size=k)
            alpha0 = 1.0
            return _run_chain_aspatial(
                y=y,
                X=X,
                priors=priors,
                beta0=beta0,
                alpha0=alpha0,
                draws=draws,
                tune=tune,
                rng=rng,
                chain_id=chain_id,
                progress_manager=progress_manager,
            )

        # --- Run chains ---
        from ...samplers._utils._seeds import spawn_chain_seeds

        np_seeds = (
            spawn_chain_seeds(random_seed, chains) if random_seed is not None else None
        )
        chain_results = run_chains(
            chain_fn=_chain_fn,
            n_chains=chains,
            seeds=np_seeds,
            n_jobs=n_jobs,
            progressbar=progressbar,
            parallel=n_jobs != 1,
            draws=draws,
            tune=tune,
            model_type="nb_flow",
        )

        # --- Assemble DataTree ---
        posterior_samples = {
            "beta": np.stack([c["beta"] for c in chain_results], axis=0),
            "alpha": np.stack([c["alpha"] for c in chain_results], axis=0),
        }
        ll = None
        if log_likelihood:
            ll = {"obs": np.stack([c["log_lik"] for c in chain_results], axis=0)}
        coords = {"coefficient": list(self._feature_names)}
        dims = {"beta": ["coefficient"]}

        self._idata = gibbs_to_inference_data(
            posterior_samples=posterior_samples,
            log_likelihood=ll,
            observed_data={"obs": self._y_int_vec},
            coords=coords,
            dims=dims,
        )
        return self._idata


# ---------------------------------------------------------------------------
# Poisson SAR flows
# ---------------------------------------------------------------------------


#: Pearson dispersion above which a Poisson flow fit warns.
_POISSON_DISPERSION_WARN = 2.0


class _PoissonFlowMixin:
    """Shared ``fit`` dispatch for Poisson flow models.

    Mirrors :class:`_NegBinFlowMixin`, but dispatches to the auxiliary-mixture
    Gibbs sampler (Frühwirth-Schnatter & Wagner 2006) rather than Pólya–Gamma.

    **Use it for counts that are close to Poisson.**  The augmentation draws
    are exact, but the finite normal mixture standing in for the
    ``−log Gamma`` error has quadratic log-tails where the true density's are
    linear.  When counts sit far from their fitted means — overdispersion the
    model does not capture — their working residuals land deep in those tails
    and pull the fit toward the large counts.  Against an exact intercept-only
    posterior: no bias on Poisson data, about one posterior sd at Pearson
    dispersion ≈ 4, seven at ≈ 13.  ``fit`` reports the dispersion in
    ``idata.attrs["pearson_dispersion"]`` and warns above
    ``_POISSON_DISPERSION_WARN``.  For overdispersed data use the NB flow
    models; Pólya–Gamma is exact there.  ``priors={"alpha_fixed": a}`` on an NB
    model with ``a`` 10-20 times the typical mean is an essentially Poisson
    model on that exact sampler (mixing degrades only for ``a`` ≳ 1000).

    Gibbs-only: unlike the NB classes there is no ``sampler="nuts"`` path yet.
    """

    def fit(
        self,
        draws: int = 2000,
        tune: int = 1000,
        chains: int = 4,
        random_seed: Optional[int] = None,
        sampler: str = "gibbs",
        backend: str | None = None,
        progressbar: bool = True,
        n_jobs: int = -1,
        idata_kwargs: Optional[dict] = None,
        **sample_kwargs,
    ) -> xr.DataTree:
        """Draw samples from the posterior.

        Parameters
        ----------
        draws, tune, chains : int
            Post-warmup draws, warmup sweeps, and number of chains.
        random_seed : int, optional
            Seed for reproducibility.
        sampler : {"gibbs"}, default "gibbs"
            Only the auxiliary-mixture Gibbs sampler is implemented.
        backend : {"auto", "numpy"}, optional
            The sampler runs on NumPy only.
        progressbar : bool, default True
            Show a progress bar.
        n_jobs : int, default -1
            Chain-level parallelism.
        idata_kwargs : dict, optional
            ``{"log_likelihood": True}`` stores the pointwise log-likelihood
            (one value per draw, chain, and flow) for ``az.loo``.
            Off by default, as in PyMC.
        """
        if sampler != "gibbs":
            raise NotImplementedError(
                f"{type(self).__name__} supports sampler='gibbs' only; "
                f"got {sampler!r}.  There is no NUTS path for the "
                "auxiliary-mixture Poisson flow models yet."
            )
        self._resolve_gibbs_backend(
            select_backend(sampler, backend, sample_kwargs), jax=False
        )
        if sample_kwargs:
            raise TypeError(f"Unexpected keyword arguments: {sorted(sample_kwargs)}")
        idata = self._fit_gibbs(
            draws=draws,
            tune=tune,
            chains=chains,
            random_seed=random_seed,
            progressbar=progressbar,
            n_jobs=n_jobs,
            log_likelihood=bool((idata_kwargs or {}).get("log_likelihood", False)),
        )
        self._check_dispersion(idata)
        return idata

    def _build_pymc_model(self) -> pm.Model:
        raise NotImplementedError(
            f"{type(self).__name__} builds no PyMC graph: it samples by the "
            "auxiliary-mixture Gibbs sampler only."
        )

    def _poisson_eta(self, rho_d, rho_o, rho_w, beta) -> np.ndarray:
        """``A(ρ)⁻¹Xβ`` for one parameter value."""
        raise NotImplementedError

    def _check_dispersion(self, idata) -> float:
        """Pearson dispersion at the posterior mean; warn when well above 1."""
        post = idata.posterior
        mean = {k: float(post[k].mean()) for k in ("rho_d", "rho_o", "rho_w")}
        beta = post["beta"].mean(("chain", "draw")).values
        mu = np.exp(np.clip(self._poisson_eta(**mean, beta=beta), -50.0, 50.0))
        y = self._y_int_vec.astype(np.float64)
        phi = float(np.sum((y - mu) ** 2 / mu) / max(y.size - beta.size, 1))
        idata.attrs["pearson_dispersion"] = phi
        if phi > _POISSON_DISPERSION_WARN:
            warnings.warn(
                f"Pearson dispersion {phi:.2f}: the counts are overdispersed "
                "relative to this Poisson fit, and the auxiliary-mixture sampler "
                "is biased toward large counts under overdispersion (about one "
                "posterior sd at dispersion 4).  Use the negative binomial flow "
                "model instead.",
                UserWarning,
                stacklevel=3,
            )
        return phi


def _require_counts(y, cls_name: str) -> np.ndarray:
    """Validate and coerce a response vector to non-negative integer counts."""
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


class SARPoissonFlow(_PoissonFlowMixin, _Unrestricted, FlowModel):
    """Unrestricted 3-ρ SAR flow model with Poisson observation noise.

    .. warning::

       The unrestricted 3-ρ parameterization mixes poorly at moderate ``n`` for
       *both* likelihoods — ρ_d, ρ_o and ρ_w trade off along a ridge that
       one-at-a-time slice updates cannot traverse.  Measured at n=36
       (N=1296): ESS ≈ 4 of 2000 draws with rhat ≈ 1.4 for this sampler *and*
       for :class:`SARNegBinFlow` on the same data.  Prefer
       :class:`SARPoissonFlowSeparable` unless ρ_w is genuinely of interest.
    """

    _dispatch = classmethod(flow)

    def __init__(self, y, X, W, **kwargs):
        y_arr = _require_counts(y, "SARPoissonFlow")
        super().__init__(y_arr.astype(np.float64), X, W, **kwargs)
        self._y_int_vec: np.ndarray = y_arr.ravel().astype(np.int64)

    def _compute_jacobian_log_det(self, posterior) -> Optional[np.ndarray]:
        return None

    def posterior_predictive(
        self,
        n_draws: Optional[int] = None,
        random_seed: Optional[int] = None,
    ) -> np.ndarray:
        """Draw posterior-predictive flow counts for Poisson SAR flow model."""
        if self._idata is None:
            raise RuntimeError("Model has not been fit yet.  Call fit() first.")

        post = self._idata.posterior
        beta_draws = post["beta"].values.reshape(-1, len(self._feature_names))
        rho_d_draws = post["rho_d"].values.reshape(-1)
        rho_o_draws = post["rho_o"].values.reshape(-1)
        rho_w_draws = post["rho_w"].values.reshape(-1)

        total = beta_draws.shape[0]
        if n_draws is not None:
            total = min(int(n_draws), total)
            beta_draws = beta_draws[:total]
            rho_d_draws = rho_d_draws[:total]
            rho_o_draws = rho_o_draws[:total]
            rho_w_draws = rho_w_draws[:total]

        rng = np.random.default_rng(random_seed)
        out = np.empty((total, self._N), dtype=np.float64)
        for g in range(total):
            eta = self._poisson_eta(
                rho_d_draws[g], rho_o_draws[g], rho_w_draws[g], beta_draws[g]
            )
            lam = np.exp(np.clip(eta, -50.0, 50.0))
            out[g] = rng.poisson(lam).astype(np.float64)
        return out

    def _poisson_eta(self, rho_d, rho_o, rho_w, beta) -> np.ndarray:
        return self._solve_A(rho_d, rho_o, rho_w, self._X @ beta)

    def _fit_gibbs(
        self,
        draws: int = 2000,
        tune: int = 1000,
        chains: int = 4,
        random_seed: Optional[int] = None,
        progressbar: bool = True,
        n_jobs: int = -1,
        log_likelihood: bool = False,
    ) -> xr.DataTree:
        """Sample via reduced-form auxiliary-mixture Gibbs (unrestricted 3-ρ)."""
        from ._poisson_gibbs import run_poisson_flow_gibbs

        return run_poisson_flow_gibbs(
            self,
            separable=False,
            model_type="pois_sar_flow",
            draws=draws,
            tune=tune,
            chains=chains,
            random_seed=random_seed,
            progressbar=progressbar,
            n_jobs=n_jobs,
            log_likelihood=log_likelihood,
        )


class SARPoissonFlowSeparable(_PoissonFlowMixin, _Separable, FlowModel):
    """Separable SAR flow model with Poisson observation noise.

    ``rho_w = -rho_d * rho_o`` is deterministic, which removes the ρ ridge that
    makes the unrestricted variant intractable.  This is the recommended
    Poisson flow model, for counts close to Poisson: under overdispersion the
    auxiliary-mixture sampler is biased (see the warning ``fit`` raises, and
    use :class:`SARNegBinFlowSeparable` instead).
    """

    def __init__(self, y, X, W, **kwargs):
        y_arr = _require_counts(y, "SARPoissonFlowSeparable")
        super().__init__(y_arr.astype(np.float64), X, W, **kwargs)
        self._y_int_vec: np.ndarray = y_arr.ravel().astype(np.int64)

    def _compute_jacobian_log_det(self, posterior) -> Optional[np.ndarray]:
        return None

    def posterior_predictive(
        self,
        n_draws: Optional[int] = None,
        random_seed: Optional[int] = None,
    ) -> np.ndarray:
        """Draw posterior-predictive flow counts for separable Poisson SAR flow."""
        if self._idata is None:
            raise RuntimeError("Model has not been fit yet.  Call fit() first.")

        post = self._idata.posterior
        beta_draws = post["beta"].values.reshape(-1, len(self._feature_names))
        rho_d_draws = post["rho_d"].values.reshape(-1)
        rho_o_draws = post["rho_o"].values.reshape(-1)

        total = beta_draws.shape[0]
        if n_draws is not None:
            total = min(int(n_draws), total)
            beta_draws = beta_draws[:total]
            rho_d_draws = rho_d_draws[:total]
            rho_o_draws = rho_o_draws[:total]

        rng = np.random.default_rng(random_seed)
        out = np.empty((total, self._N), dtype=np.float64)
        for g in range(total):
            eta = self._poisson_eta(rho_d_draws[g], rho_o_draws[g], None, beta_draws[g])
            lam = np.exp(np.clip(eta, -50.0, 50.0))
            out[g] = rng.poisson(lam).astype(np.float64)
        return out

    def _poisson_eta(self, rho_d, rho_o, rho_w, beta) -> np.ndarray:
        """``(L_o ⊗ L_d)⁻¹Xβ``; ``rho_w`` is implied (``−ρ_d ρ_o``) and ignored."""
        n = self._n
        I_n = sp.eye(n, format="csr", dtype=np.float64)
        Ld = I_n - float(rho_d) * self._W_sparse
        Lo = I_n - float(rho_o) * self._W_sparse
        return kron_solve_vec(Lo, Ld, self._X @ beta, n)

    def _fit_gibbs(
        self,
        draws: int = 2000,
        tune: int = 1000,
        chains: int = 4,
        random_seed: Optional[int] = None,
        progressbar: bool = True,
        n_jobs: int = -1,
        log_likelihood: bool = False,
    ) -> xr.DataTree:
        """Sample via reduced-form auxiliary-mixture Gibbs (separable 2-ρ)."""
        from ._poisson_gibbs import run_poisson_flow_gibbs

        return run_poisson_flow_gibbs(
            self,
            separable=True,
            model_type="pois_sar_flow_sep",
            draws=draws,
            tune=tune,
            chains=chains,
            random_seed=random_seed,
            progressbar=progressbar,
            n_jobs=n_jobs,
            log_likelihood=log_likelihood,
        )
