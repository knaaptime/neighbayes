"""Bayesian spatial flow (origin-destination) models.

Implements fully Bayesian SAR-type flow models following
:cite:t:`lesage2008SpatialEconometric`.  The observed variable is an
:math:`n \\times n` flow matrix (or its vectorized form), and the weight
structure uses three Kronecker-product matrices:

.. math::

    W_d = I_n \\otimes W, \\quad W_o = W \\otimes I_n, \\quad W_w = W \\otimes W

so that the model is:

.. math::

    y = \\rho_d W_d y + \\rho_o W_o y + \\rho_w W_w y + X\\beta + \\varepsilon,
    \\quad \\varepsilon \\sim \\mathcal{N}(0, \\sigma^2 I_N)

where :math:`N = n^2`.

Two variants are provided:

* :class:`SARFlow` — three free ρ parameters with a Dirichlet stability
  constraint (default) or a quadratic-wall potential when competitive effects
  are needed (``restrict_positive=False``).
* :class:`SARFlowSeparable` — constrained :math:`\\rho_w = -\\rho_d \\rho_o`,
  enabling exact eigenvalue-based log-determinant.
"""

from __future__ import annotations

from abc import abstractmethod
from typing import Any, Optional, Union

import numpy as np
import pandas as pd
import pytensor.tensor as pt
import scipy.sparse as sp

from ..._lazy_deps import az, pm
from ..._logdet import (
    make_flow_separable_logdet,
    make_flow_separable_logdet_numpy,
)
from ..._ops import kron_solve_vec
from ...graph import _weights_to_csr, flow_lags, flow_trace_blocks
from .._mixins._flow_shared import FlowSharedMethods
from ..base import SpatialModel

_EFFECT_KEYS = ("origin", "destination", "intra", "network", "total")


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
        r = float(
            np.max((np.abs(ro) + np.abs(rd)) * w + np.abs(rw) * w * w, initial=0.0)
        )
        if r >= 1.0:
            raise ValueError(
                "Flow effects need |rho_d| + |rho_o| + |rho_w| < 1 (in the "
                f"max-row-sum norm of W); a posterior draw reaches {r:.4f}."
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
        out[f"{side}_intra"] = intra
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
    with an :math:`N = n^2` vectorized response and three Kronecker-product
    weight matrices constructed from a single n×n graph.  The API mirrors
    :class:`~neighbayes.models.base.SpatialModel` (``fit``, ``summary``,
    ``inference_data``) but the internals are tailored to the flow structure.

    The model accepts a full O-D design matrix *X* of shape ``(n², p)``,
    typically produced by :func:`~neighbayes.graph.flow_design_matrix` or
    :func:`~neighbayes.graph.flow_design_matrix_with_orig`.

    Parameters
    ----------
    y : array-like, shape (n, n) or (N,)
        Observed O-D flow matrix (or its vec-form).  Must be a square
        matrix or a flat vector of length :math:`N = n^2`.
    W : libpysal.graph.Graph or scipy.sparse / dense (n×n) matrix
        Row-standardized regional weights on *n* units (Graph or matrix).  Validated by
        :func:`~neighbayes.graph._graph_to_csr`.
    X : np.ndarray or pandas.DataFrame, shape (N, p)
        Full origin-destination design matrix with :math:`N = n^2` rows.
        This is typically produced by
        :func:`~neighbayes.graph.flow_design_matrix` or
        :func:`~neighbayes.graph.flow_design_matrix_with_orig`.
        If a DataFrame, column names are inferred automatically.
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
        How to compute :math:`\\log|I_N - \\rho_d W_d - \\rho_o W_o - \\rho_w W_w|`.
        ``None`` (default) auto-selects.  Concrete subclasses override the
        default with their recommended method: the unrestricted SAR/SEM flow
        classes use the resolvent-gradient sampler (``"resolvent"``), the
        separable classes accept any single-``W`` method (``"eigenvalue"``,
        ``"chebyshev"``, ``"cheb_cholesky"``, ``"aaa"``,
        ``"cheb_stochastic"``) via the exact Kronecker factorization, and
        aspatial classes skip the log-determinant entirely.
    restrict_positive : bool, default True
        If True, use a ``pm.Dirichlet`` prior that restricts :math:`\\rho_d,
        \\rho_o, \\rho_w \\geq 0` with :math:`\\rho_d + \\rho_o + \\rho_w \\leq 1`.
        This is NUTS-safe and appropriate for most flow applications.
        If False, use three independent ``pm.Uniform(-1, 1)`` priors with a
        differentiable quadratic-wall stability potential.
    symmetric_xo_xd : bool, optional
        If ``None`` (default), the destination and origin design blocks are
        compared and symmetry is auto-detected.  Set explicitly to override
        the heuristic — for example, when using
        :func:`~neighbayes.graph.flow_design_matrix_with_orig` with distinct
        attributes for the origin and destination sides.  Controls the default
        behavior of :meth:`spatial_effects` when ``mode="auto"``.
    """

    def __init__(
        self,
        y: Union[np.ndarray, pd.Series],
        X: Union[np.ndarray, pd.DataFrame],
        W,
        col_names: Optional[list] = None,
        k: Optional[int] = None,
        priors: Optional[dict] = None,
        logdet_method: Optional[str] = None,
        restrict_positive: bool = True,
        symmetric_xo_xd: Optional[bool] = None,
    ):
        self.priors = priors or {}
        self.logdet_method = logdet_method
        self.restrict_positive = restrict_positive
        self.robust = False
        self._is_row_std = True  # W is assumed row-standardized
        self._idata: Optional[az.InferenceData] = None
        self._pymc_model: Optional[pm.Model] = None

        # Validate and extract the n×n weight matrix (Graph or matrix accepted).
        self._W_sparse: sp.csr_matrix = _weights_to_csr(W)
        self._n: int = self._W_sparse.shape[0]
        self._N: int = self._n * self._n

        # Validate y
        y_arr = np.asarray(y, dtype=np.float64)
        if y_arr.ndim == 2:
            if y_arr.shape != (self._n, self._n):
                raise ValueError(
                    f"y matrix must be ({self._n}, {self._n}), got {y_arr.shape}."
                )
            self._y = y_arr.ravel()
        elif y_arr.ndim == 1:
            if len(y_arr) != self._N:
                raise ValueError(
                    f"y vector must have length N={self._N} (= n²), got {len(y_arr)}."
                )
            self._y = y_arr
        else:
            raise ValueError("y must be a 1-D or 2-D array.")

        # Validate X and build design matrix
        if isinstance(X, pd.DataFrame):
            if col_names is None:
                col_names = list(X.columns)
            X_arr = X.to_numpy(dtype=np.float64)
        else:
            X_arr = np.asarray(X, dtype=np.float64)

        if X_arr.ndim == 1:
            X_arr = X_arr[:, None]
        if X_arr.shape[0] != self._N:
            raise ValueError(
                f"X must have {self._N} rows (= n² = {self._n}²), got {X_arr.shape[0]}."
            )

        self._X: np.ndarray = X_arr  # (N, p)
        if col_names is not None:
            self._feature_names: list[str] = list(col_names)
        elif X_arr.shape[1] == 0:
            self._feature_names = []
        else:
            self._feature_names = [f"x{i}" for i in range(X_arr.shape[1])]

        # Infer k_d and k_o (number of regional attribute columns) for effects computation.
        # Standard LeSage layout: [intercept, intra_indicator, dest_*, orig_*, intra_*, (dist)]
        if k is not None:
            self._k: int = k
            self._k_d: int = k
            self._k_o: int = k
        else:
            dest_cols = [
                name for name in self._feature_names if name.startswith("dest_")
            ]
            orig_cols = [
                name for name in self._feature_names if name.startswith("orig_")
            ]
            self._k_d = len(dest_cols)
            self._k_o = len(orig_cols)
            self._k = self._k_d  # backward compat alias
            if self._k_d == 0 and self._k_o == 0:
                # Fallback: cannot infer k from column names; effects decomposition
                # will not be available.  Set k=0 as a sentinel.
                self._k = 0

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
        # identical (the symmetric Xo = Xd case).  When asymmetric the
        # Thomas-Agnan & LeSage (2014, §83.5.2) shortcut of summing dest and
        # orig effects is not appropriate, and `spatial_effects(mode="auto")`
        # falls back to reporting both sides separately.
        if (
            symmetric_xo_xd is None
            and self._k_d > 0
            and self._k_d == self._k_o
            and X_arr.shape[1] >= 2 + self._k_d + self._k_o
        ):
            dest_block = X_arr[:, 2 : 2 + self._k_d]
            orig_block = X_arr[:, 2 + self._k_d : 2 + self._k_d + self._k_o]
            self._symmetric_xo_xd: bool = bool(np.array_equal(dest_block, orig_block))
        else:
            self._symmetric_xo_xd = (
                bool(symmetric_xo_xd)
                if symmetric_xo_xd is not None
                else (self._k_d == self._k_o)
            )

        # Pre-compute logdet data for separable constraint: log|Lo⊗Ld| = n*f(ρ_d) + n*f(ρ_o).
        # Also keep _W_eigs for backward compatibility.  ``_W_eigs`` is
        # populated only by the eigenvalue logdet path.
        # exposed as a property below that returns ``None`` when eigenvalues
        # were not pre-computed (mirroring :class:`SpatialModel`).
        self._W_eigs: Optional[np.ndarray] = None
        self._separable_logdet_fn = None
        self._separable_logdet_numpy_fn = None
        _SEPARABLE_METHODS = {
            "eigenvalue",
            "chebyshev",
            "cheb_cholesky",
            "aaa",
            "cheb_stochastic",
        }
        if logdet_method is None or logdet_method in _SEPARABLE_METHODS:
            from ..._logdet._config import resolve_logdet_method

            self._separable_logdet_fn = make_flow_separable_logdet(
                self._W_sparse,
                self._n,
                method=logdet_method,
            )
            self._separable_logdet_numpy_fn = make_flow_separable_logdet_numpy(
                self._W_sparse,
                self._n,
                method=logdet_method,
            )
            # Populate ``_W_eigs`` only when the resolved method is eigenvalue
            # (auto-selection may resolve None to eigenvalue for small n).
            resolved = resolve_logdet_method(logdet_method, n=self._n, W=self._W_sparse)
            if resolved == "eigenvalue":
                self._W_eigs = np.linalg.eigvals(
                    self._W_sparse.toarray().astype(np.float64)
                ).real

        # Pre-compute spatial lags: Wd_y, Wo_y, Ww_y
        # Matrix-free: W⊗W alone would hold nnz(W)² entries.
        self._Wd_y, self._Wo_y, self._Ww_y = flow_lags(self._W_sparse, self._y)

        # Aliases used by some downstream code and tests
        self._y_vec = self._y
        self._spatial_lag = self._Wd_y

        # Cache the symmetric 3x3 Kronecker trace matrix used by Bayesian
        # LM diagnostics: T[i,j] = tr(W_i' W_j) + tr(W_i W_j) for
        # (W_d, W_o, W_w).  Computed in O(nnz) from the base n x n graph.
        self._T_flow_traces: np.ndarray = flow_trace_blocks(self._W_sparse)

        # The unrestricted 3-parameter flow log-determinant is handled by the
        # resolvent-Kronecker gradient sampler; the old "traces" value method was
        # removed (it amplifies stochastic-moment noise for large directed W).
        self._traces = None

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    def _build_pymc_model(self) -> pm.Model:
        """Construct and return the PyMC model.  Implemented by subclasses."""

    @abstractmethod
    def _compute_spatial_effects_posterior(
        self, draws: Optional[int] = None
    ) -> dict[str, np.ndarray]:
        """Compute posterior spatial effects.  Implemented by subclasses."""

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
    # Pointwise log-likelihood and sparse filter helpers inherited from
    # FlowSharedMethods — see .._mixins._flow_shared
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Public diagnostics
    # ------------------------------------------------------------------

    def spatial_effects(
        self,
        draws: Optional[int] = None,
        return_posterior_samples: bool = False,
        ci: float = 0.95,
        mode: str = "auto",
    ) -> "pd.DataFrame | tuple[pd.DataFrame, dict[str, np.ndarray]]":
        """Summarize posterior origin/destination/intra/network/total effects.

        Wraps :meth:`_compute_spatial_effects_posterior` to produce a tidy
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
            (``self._symmetric_xo_xd``) and reports both sides otherwise.
            ``"combined"`` always sums; ``"separate"`` always reports both.

        Returns
        -------
        pandas.DataFrame, or (DataFrame, dict)
            Long-format summary indexed by ``(predictor, side, effect)`` where
            ``side`` is one of ``"combined"``, ``"dest"``, ``"orig"``.
        """
        from ...diagnostics.spatial_effects import _compute_bayesian_pvalue

        if self._idata is None:
            raise RuntimeError("Model has not been fit yet.  Call fit() first.")
        if self._k == 0:
            raise RuntimeError(
                "Cannot compute spatial effects: no `dest_*` columns detected "
                "in the design matrix.  Pass `k=` explicitly when constructing "
                "the model."
            )
        if mode not in {"auto", "combined", "separate"}:
            raise ValueError(
                f"mode must be 'auto', 'combined', or 'separate'; got {mode!r}."
            )

        posterior = self._compute_spatial_effects_posterior(draws=draws)

        if mode == "auto":
            effective_mode = "combined" if self._symmetric_xo_xd else "separate"
        else:
            effective_mode = mode

        if effective_mode == "combined":
            display = [("combined", eff) for eff in _EFFECT_KEYS]
        else:
            display = [(side, eff) for side in ("dest", "orig") for eff in _EFFECT_KEYS]

        # Predictor names: prefer dest_* and orig_* labels stripped of the prefix.
        dest_feature_names = [
            name[len("dest_") :] if name.startswith("dest_") else name
            for name in self._feature_names
            if name.startswith("dest_")
        ][: self._k_d]
        if len(dest_feature_names) != self._k_d:
            dest_feature_names = [f"x{i}" for i in range(self._k_d)]

        orig_feature_names = [
            name[len("orig_") :] if name.startswith("orig_") else name
            for name in self._feature_names
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
        """Draw a single posterior-predictive replicate.

        Default implementation: Gaussian SAR flow,
        ``y_rep = A^{-1} (X β + σ ε)`` with ``ε ~ N(0, I_N)``.
        Subclasses (SARNegBinFlow, SARNegBinFlowSeparable) override this.
        """
        Xb = self._X @ beta
        eps = rng.normal(scale=float(sigma), size=self._N) if sigma is not None else 0.0
        rhs = Xb + eps
        return self._solve_A(rho_d, rho_o, rho_w, rhs)

    def posterior_predictive(
        self,
        n_draws: Optional[int] = None,
        random_seed: Optional[int] = None,
    ) -> np.ndarray:
        """Draw posterior-predictive samples ``y_rep``.

        For each (subsampled) posterior draw, simulates a new flow vector
        ``y_rep`` from the implied data-generating process by solving the
        sparse system ``A(rho) y_rep = X β + ε`` (Gaussian) or
        ``y_rep ~ NegBin(exp(A^{-1} X β), α)`` (NB variants).

        Parameters
        ----------
        n_draws : int, optional
            Number of posterior draws to use.  Defaults to all available.
        random_seed : int, optional
            Seed for the posterior-predictive sampler.

        Returns
        -------
        np.ndarray
            Array of shape ``(n_draws, N)`` with posterior-predictive flows.
        """
        if self._idata is None:
            raise RuntimeError("Model has not been fit yet.  Call fit() first.")

        post = self._idata.posterior
        rho_d = post["rho_d"].values.reshape(-1)
        rho_o = post["rho_o"].values.reshape(-1)
        rho_w = post["rho_w"].values.reshape(-1)
        beta_draws = post["beta"].values.reshape(-1, len(self._feature_names))
        sigma_draws = (
            post["sigma"].values.reshape(-1) if "sigma" in post.data_vars else None
        )

        total = len(rho_d)
        if n_draws is not None:
            total = min(int(n_draws), total)
            rho_d = rho_d[:total]
            rho_o = rho_o[:total]
            rho_w = rho_w[:total]
            beta_draws = beta_draws[:total]
            if sigma_draws is not None:
                sigma_draws = sigma_draws[:total]

        rng = np.random.default_rng(random_seed)
        out = np.empty((total, self._N), dtype=np.float64)
        for g in range(total):
            sigma_g = float(sigma_draws[g]) if sigma_draws is not None else None
            out[g] = self._simulate_y_rep(
                float(rho_d[g]),
                float(rho_o[g]),
                float(rho_w[g]),
                beta_draws[g],
                sigma_g,
                rng,
            )
        return out


# ---------------------------------------------------------------------------
# Model 1: SARFlow — unrestricted 3-ρ
# ---------------------------------------------------------------------------


class SARFlow(FlowModel):
    """Bayesian SAR flow model with three free spatial autoregressive parameters.

    .. math::

        y = \\rho_d W_d y + \\rho_o W_o y + \\rho_w W_w y + X\\beta + \\varepsilon,
        \\quad \\varepsilon \\sim \\mathcal{N}(0, \\sigma^2 I_N)

    where :math:`W_d = I_n \\otimes W`, :math:`W_o = W \\otimes I_n`,
    :math:`W_w = W \\otimes W`.

    Parameters
    ----------
    y : array-like, shape (n, n) or (N,)
        Observed origin-destination flow matrix or its vec-form. Must be
        a square matrix or a flat vector of length :math:`N = n^2`.
    W : libpysal.graph.Graph or scipy.sparse / dense (n×n) matrix
        Row-standardized regional weights on *n* units (Graph or matrix).
    X : np.ndarray or pandas.DataFrame, shape (N, p)
        Full origin-destination design matrix with :math:`N = n^2` rows.
        Typically produced by :func:`~neighbayes.graph.flow_design_matrix`
        or :func:`~neighbayes.graph.flow_design_matrix_with_orig`.
        DataFrame columns are preserved as feature names.
    col_names : list of str, optional
        Column labels for ``X``. Inferred from a DataFrame if omitted;
        otherwise defaults to ``["x0", "x1", ...]``.
    k : int, optional
        Number of regional attribute columns (destination/origin variable
        pairs). Inferred from ``dest_*``/``orig_*`` column names when the
        standard LeSage layout is used.
    logdet_method : str, default "resolvent"
        Log-determinant method.  The default ``"resolvent"`` samples via the
        resolvent-Kronecker gradient sampler (recommended).
    restrict_positive : bool, default True
        If True, use ``pm.Dirichlet("rho_simplex", a=ones(4))`` to enforce
        :math:`\\rho_d, \\rho_o, \\rho_w \\geq 0` and
        :math:`\\rho_d + \\rho_o + \\rho_w \\leq 1`. NUTS-safe via the
        stick-breaking bijection and appropriate when competitive
        (negative) spillovers are not expected. If False, three
        independent ``pm.Uniform(rho_lower, rho_upper)`` priors are used
        together with a differentiable quadratic-wall stability potential.
    symmetric_xo_xd : bool, optional
        If ``None`` (default), origin and destination design blocks are
        compared and symmetry is auto-detected. Set explicitly to override
        the heuristic.
    priors : dict, optional
        Override default priors. Supported keys:

        - ``beta_mu`` : float, default 0.0 — Normal prior mean for ``beta``.
        - ``beta_sigma`` : float, default 1e6 — Normal prior std for ``beta``.
        - ``sigma_sigma`` : float, default 10.0 — HalfNormal prior std for ``sigma``.
        - ``rho_lower`` : float, default -1.0 — Lower bound of Uniform prior on each ρ (only when ``restrict_positive=False``).
        - ``rho_upper`` : float, default 1.0 — Upper bound of Uniform prior on each ρ (only when ``restrict_positive=False``).

    Notes
    -----
    The unrestricted flow log-determinant is sampled with the scalable
    resolvent-Kronecker gradient (see
    :mod:`neighbayes.samplers.gaussian._flow_resolvent`), whose accuracy improves
    with the flow sample size ``N``.  The legacy ``"traces"`` value method — which
    amplifies stochastic-moment noise for large directed ``W`` — is retired but
    remains reachable via ``logdet_method="traces"`` (PyMC/NUTS path).
    """

    def __init__(self, y, X, W, **kwargs):
        # Default to the resolvent-gradient sampler; "traces" stays as an
        # explicit (deprecated) opt-in routed through the legacy PyMC path.
        kwargs.setdefault("logdet_method", "resolvent")
        super().__init__(y, X, W, **kwargs)

    def fit(
        self,
        draws: int = 2000,
        tune: int = 1000,
        chains: int = 4,
        random_seed: Optional[int] = None,
        *,
        step_size: float = 5e-4,
        n_probes: int = 48,
        logdet_method: str = "auto",
        n_quad: int = 8,
        progressbar: bool = True,
        n_jobs: int = -1,
        idata_kwargs: Optional[dict] = None,
        **sample_kwargs,
    ) -> az.InferenceData:
        """Sample the posterior with the resolvent-Kronecker gradient sampler.

        MALA-on-ρ within conjugate Gibbs for ``β, σ²``; the flow log-determinant
        gradient and value come from the scalable, eigenvalue-free resolvent
        estimator (whose accuracy improves with the flow sample size ``N``).

        Parameters
        ----------
        logdet_method : {"auto", "kron_traces", "jax", "numpy"}, default "auto"
            Flow log-determinant backend.  ``"auto"`` is exact (trace moments of
            ``W``, no ``N × N`` work) for undirected ``W`` and the stochastic
            resolvent estimator for directed ``W``; see
            :func:`~neighbayes.samplers.gaussian._flow_resolvent.resolve_flow_logdet`.
        idata_kwargs : dict, optional
            ``{"log_likelihood": True}`` stores the pointwise log-likelihood
            (one value per draw, chain, and flow) for ``az.loo`` / ``az.waic``.
            Off by default, as in PyMC.
        """
        from ...samplers.gaussian._flow_resolvent import sample_flow_resolvent

        self._pymc_model = None
        self._idata = sample_flow_resolvent(
            self._W_sparse,
            self._y,
            self._X,
            draws=draws,
            tune=tune,
            chains=chains,
            step_size=step_size,
            n_probes=n_probes,
            coord_names=list(self._feature_names) or None,
            random_seed=random_seed,
            logdet_method=logdet_method,
            n_quad=n_quad,
            progressbar=progressbar,
            n_jobs=n_jobs,
            restrict_positive=self.restrict_positive,
            compute_log_likelihood=bool(
                (idata_kwargs or {}).get("log_likelihood", False)
            ),
        )
        return self._idata

    def _build_pymc_model(self) -> pm.Model:
        # The unrestricted flow no longer uses a PyMC/NUTS model; ``fit`` samples
        # via the resolvent-Kronecker gradient sampler
        # (:func:`neighbayes.samplers.gaussian._flow_resolvent.sample_flow_resolvent`).
        raise NotImplementedError(
            "SARFlow samples via the resolvent-gradient sampler, not PyMC; "
            "the legacy 'traces' Jacobian was removed. Use SARFlow.fit()."
        )

    def _compute_spatial_effects_posterior(
        self, draws: Optional[int] = None
    ) -> dict[str, np.ndarray]:
        """Compute posterior origin / destination / intra / network / total effects.

        Implements the LeSage (2008) effects decomposition: for each posterior
        draw of :math:`(\\rho_d, \\rho_o, \\rho_w, \\beta_d, \\beta_o)` and each
        regional predictor *p*, builds an :math:`N\\times n` shock matrix whose
        column ``j`` contains :math:`\\beta_d^{(p)}` at flows with destination
        ``j``, :math:`\\beta_o^{(p)}` at flows with origin ``j``, and
        :math:`\\beta_d^{(p)} + \\beta_o^{(p)}` at the intra flow ``(j, j)``.
        The system :math:`A\\,T = \\text{shock}` is solved with one sparse
        :math:`LU` factorization per draw (re-used for all ``n`` columns and all
        ``k`` predictors), and scalar effects are obtained by averaging
        :math:`T` over the appropriate masks.  Mirrors LeSage's
        ``calc_effects.m`` reference implementation.

        Parameters
        ----------
        draws : int, optional
            Number of posterior draws to use.  Defaults to all draws.

        Returns
        -------
        dict[str, np.ndarray]
            Keys: ``"origin"``, ``"destination"``, ``"intra"``, ``"network"``,
            ``"total"``.  Each value has shape ``(draws, k)``.
        """
        if self._idata is None:
            raise RuntimeError("Model has not been fit yet.  Call fit() first.")

        idata = self._idata
        k_d = self._k_d
        k_o = self._k_o

        rho_d_draws = idata.posterior["rho_d"].values.reshape(-1)
        rho_o_draws = idata.posterior["rho_o"].values.reshape(-1)
        rho_w_draws = idata.posterior["rho_w"].values.reshape(-1)
        beta_draws = idata.posterior["beta"].values.reshape(
            -1, len(self._feature_names)
        )

        dest_start = 2
        orig_start = 2 + k_d
        intra_start = 2 + k_d + k_o
        has_intra = (
            self._intra_idx is not None and beta_draws.shape[1] >= intra_start + k_d
        )

        n_draws_total = len(rho_d_draws)
        if draws is not None:
            n_draws_total = min(draws, n_draws_total)
            rho_d_draws = rho_d_draws[:n_draws_total]
            rho_o_draws = rho_o_draws[:n_draws_total]
            rho_w_draws = rho_w_draws[:n_draws_total]
            beta_draws = beta_draws[:n_draws_total]

        # Exact LeSage decomposition from W-only moments (no n²-sized arrays).
        return self._flow_effects_for_draws(
            rho_d_draws,
            rho_o_draws,
            rho_w_draws,
            beta_draws[:, dest_start : dest_start + k_d],
            beta_draws[:, orig_start : orig_start + k_o],
            beta_draws[:, intra_start : intra_start + k_d] if has_intra else None,
        )


# ---------------------------------------------------------------------------
# Model 2: SARFlowSeparable — constrained ρ_w = −ρ_d·ρ_o
# ---------------------------------------------------------------------------


class SARFlowSeparable(FlowModel):
    """Bayesian separable SAR flow model with ρ_w = −ρ_d · ρ_o.

    The separability constraint :math:`\\rho_w = -\\rho_d \\rho_o` reduces
    the flow weight matrix to a Kronecker structure whose log-determinant
    factors as:

    .. math::

        \\log|I_N - \\rho_d W_d - \\rho_o W_o + \\rho_d \\rho_o W_w|
        = n \\log|I_n - \\rho_d W| + n \\log|I_n - \\rho_o W|

    enabling exact O(n) log-det evaluation via eigenvalues — no trace
    estimation required.

    Parameters
    ----------
    y : array-like, shape (n, n) or (N,)
        Observed origin-destination flow matrix or its vec-form. Must be
        a square matrix or a flat vector of length :math:`N = n^2`.
    W : libpysal.graph.Graph or scipy.sparse / dense (n×n) matrix
        Row-standardized regional weights on *n* units (Graph or matrix).
    X : np.ndarray or pandas.DataFrame, shape (N, p)
        Full origin-destination design matrix with :math:`N = n^2` rows.
        Typically produced by :func:`~neighbayes.graph.flow_design_matrix`
        or :func:`~neighbayes.graph.flow_design_matrix_with_orig`.
        DataFrame columns are preserved as feature names.
    col_names : list of str, optional
        Column labels for ``X``. Inferred from a DataFrame if omitted;
        otherwise defaults to ``["x0", "x1", ...]``.
    k : int, optional
        Number of regional attribute columns (destination/origin variable
        pairs). Inferred from ``dest_*``/``orig_*`` column names when the
        standard LeSage layout is used.
    logdet_method : {"eigenvalue", "chebyshev", "cheb_cholesky", "aaa", "cheb_stochastic"} or None, default None
        Method for the :math:`n \\times n` log-determinant used via the exact
        Kronecker factorization.  ``None`` (default) auto-selects
        (``"aaa"`` for directed/non-symmetric ``W``).
    symmetric_xo_xd : bool, optional
        If ``None`` (default), origin and destination design blocks are
        compared and symmetry is auto-detected. Set explicitly to override
        the heuristic.
    priors : dict, optional
        Override default priors. Supported keys:

        - ``beta_mu`` : float, default 0.0 — Normal prior mean for ``beta``.
        - ``beta_sigma`` : float, default 1e6 — Normal prior std for ``beta``.
        - ``sigma_sigma`` : float, default 10.0 — HalfNormal prior std for ``sigma``.
        - ``rho_lower`` : float, default -0.999 — Lower bound of Uniform prior on ``rho_d`` and ``rho_o``.
        - ``rho_upper`` : float, default 0.999 — Upper bound of Uniform prior on ``rho_d`` and ``rho_o``.

    Notes
    -----
    The ``restrict_positive`` argument inherited from :class:`FlowModel`
    has no effect on this class — separable variants always use Uniform
    priors on the individual :math:`\rho` components.
    """

    def __init__(self, y, X, W, **kwargs):
        method = kwargs.pop("logdet_method", None)
        _VALID = {"eigenvalue", "chebyshev", "cheb_cholesky", "aaa", "cheb_stochastic"}
        if method is not None and method not in _VALID:
            raise ValueError(
                f"SARFlowSeparable logdet_method must be None (auto) or one of "
                f"{sorted(_VALID)}; got {method!r}."
            )
        kwargs["logdet_method"] = method
        super().__init__(y, X, W, **kwargs)

    def _build_pymc_model(self) -> pm.Model:
        beta_mu = self.priors.get("beta_mu", 0.0)
        beta_sigma = self.priors.get("beta_sigma", 1e6)
        sigma_sigma = self.priors.get("sigma_sigma", 10.0)
        rho_lower = self.priors.get("rho_lower", -0.999)
        rho_upper = self.priors.get("rho_upper", 0.999)

        if self._separable_logdet_fn is None:
            raise RuntimeError(
                "SARFlowSeparable requires precomputed logdet data; "
                "initialize with a separable logdet_method (None/auto, "
                "eigenvalue, chebyshev, cheb_cholesky, aaa, or cheb_stochastic)"
            )
        Wd_y_t = pt.as_tensor_variable(self._Wd_y.astype(np.float64))
        Wo_y_t = pt.as_tensor_variable(self._Wo_y.astype(np.float64))
        Ww_y_t = pt.as_tensor_variable(self._Ww_y.astype(np.float64))
        X_t = pt.as_tensor_variable(self._X.astype(np.float64))
        y_t = pt.as_tensor_variable(self._y.astype(np.float64))

        with pm.Model(coords=self._model_coords()) as model:
            rho_d = pm.Uniform("rho_d", lower=rho_lower, upper=rho_upper)
            rho_o = pm.Uniform("rho_o", lower=rho_lower, upper=rho_upper)
            # rho_w is deterministic: -rho_d * rho_o  (must appear in posterior)
            rho_w = pm.Deterministic("rho_w", -rho_d * rho_o)

            beta = pm.Normal("beta", mu=beta_mu, sigma=beta_sigma, dims="coefficient")
            sigma = pm.HalfNormal("sigma", sigma=sigma_sigma)

            mu = rho_d * Wd_y_t + rho_o * Wo_y_t + rho_w * Ww_y_t + pt.dot(X_t, beta)
            pm.Normal("obs", mu=mu, sigma=sigma, observed=y_t)

            # Jacobian: n*log|I_n - rho_d*W| + n*log|I_n - rho_o*W|
            # factorization holds exactly for the separable constraint.
            pm.Potential(
                "jacobian",
                self._separable_logdet_fn(rho_d, rho_o),
            )

        return model

    def _compute_jacobian_log_det(self, posterior) -> np.ndarray:
        rho_d = np.asarray(posterior["rho_d"].values.reshape(-1), dtype=np.float64)
        rho_o = np.asarray(posterior["rho_o"].values.reshape(-1), dtype=np.float64)
        if self._separable_logdet_numpy_fn is None:
            raise RuntimeError(
                "Missing separable numeric logdet evaluator. "
                "Initialize with a separable logdet_method (None/auto, "
                "eigenvalue, chebyshev, cheb_cholesky, aaa, or cheb_stochastic)"
            )
        return self._separable_logdet_numpy_fn(rho_d, rho_o)

    def _compute_spatial_effects_posterior(
        self, draws: Optional[int] = None
    ) -> dict[str, np.ndarray]:
        """Compute posterior effects using Kronecker-factored solve.

        Implements the LeSage (2008) effects decomposition (see
        :meth:`SARFlow._compute_spatial_effects_posterior`) but exploits
        :math:`A = L_o \\otimes L_d` to replace the
        :math:`N\\times N` sparse factorization with two :math:`n\\times n`
        sparse solves per predictor via
        :func:`~neighbayes._ops.kron_solve_matrix`.

        Parameters
        ----------
        draws : int, optional
            Number of posterior draws to use.  Defaults to all.

        Returns
        -------
        dict[str, np.ndarray]
            Same keys and shapes as :meth:`SARFlow._compute_spatial_effects_posterior`.
        """
        if self._idata is None:
            raise RuntimeError("Model has not been fit yet.  Call fit() first.")

        idata = self._idata
        k_d = self._k_d
        k_o = self._k_o

        rho_d_draws = idata.posterior["rho_d"].values.reshape(-1)
        rho_o_draws = idata.posterior["rho_o"].values.reshape(-1)
        beta_draws = idata.posterior["beta"].values.reshape(
            -1, len(self._feature_names)
        )

        dest_start = 2
        orig_start = 2 + k_d
        intra_start = 2 + k_d + k_o
        has_intra = (
            self._intra_idx is not None and beta_draws.shape[1] >= intra_start + k_d
        )

        n_draws_total = len(rho_d_draws)
        if draws is not None:
            n_draws_total = min(draws, n_draws_total)
            rho_d_draws = rho_d_draws[:n_draws_total]
            rho_o_draws = rho_o_draws[:n_draws_total]
            beta_draws = beta_draws[:n_draws_total]

        # Exact LeSage decomposition from W-only moments (no n²-sized arrays).
        return self._flow_effects_for_draws(
            rho_d_draws,
            rho_o_draws,
            -rho_d_draws * rho_o_draws,  # separable: ρ_w = −ρ_d·ρ_o
            beta_draws[:, dest_start : dest_start + k_d],
            beta_draws[:, orig_start : orig_start + k_o],
            beta_draws[:, intra_start : intra_start + k_d] if has_intra else None,
        )


def _compute_ols_flow_effects(
    idata: "az.InferenceData",
    *,
    n: int,
    k_d: int,
    k_o: int,
    feature_names: list[str],
    intra_idx: Optional[np.ndarray],
    draws: Optional[int],
) -> dict[str, np.ndarray]:
    """Closed-form Thomas-Agnan & LeSage (2014, Table 83.1) effects.

    Shared between :class:`OLSFlow`, :class:`SEMFlow`,
    :class:`SEMFlowSeparable` (cross-section), and :class:`OLSFlowPanel`,
    :class:`SEMFlowPanel`, :class:`SEMFlowSeparablePanel` (panel) — all of
    which have :math:`\\mathbb{E}[y] = X\\beta` (no :math:`X`-mediated
    spillovers).
    """
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
    out["dest_intra"] = (bd + bi) / n
    out["dest_origin"] = zeros_d.copy()
    out["dest_network"] = zeros_d.copy()

    out["orig_total"] = bo
    out["orig_origin"] = bo * (n - 1) / n
    out["orig_intra"] = bo / n
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
# Model 3: OLSFlow — non-spatial gravity baseline (Thomas-Agnan & LeSage 2014)
# ---------------------------------------------------------------------------


class OLSFlow(FlowModel):
    r"""Non-spatial Bayesian OD-flow gravity model (independence baseline).

    Implements the conventional log-linear gravity model from
    :cite:t:`thomas-agnan2014SpatialEconometric` (eq. 83.2):

    .. math::

        y = \alpha \iota_{N} + X_o \beta_o + X_d \beta_d + g\gamma + \varepsilon,
        \quad \varepsilon \sim \mathcal{N}(0, \sigma^{2} I_{N})

    with no spatial-lag terms.  Provided as a baseline for comparison with
    :class:`SARFlow` / :class:`SARFlowSeparable` and to reproduce Table 83.1
    of the chapter.

    Parameters
    ----------
    y : array-like, shape (n, n) or (N,)
        Observed O-D flow matrix (or its vec-form). Must be a square
        matrix or a flat vector of length :math:`N = n^2`.
    W : libpysal.graph.Graph or scipy.sparse / dense (n×n) matrix
        Row-standardized regional weights on *n* units (Graph or matrix). Required for API
        symmetry with the spatial flow models, but the graph weights
        are not used in estimation.
    X : np.ndarray or pandas.DataFrame, shape (N, p)
        Full origin-destination design matrix.
    col_names : list[str], optional
        Column labels for *X*. Defaults to ``["x0", "x1", ...]`` when
        not provided and *X* is not a DataFrame.
    k : int, optional
        Number of regional attribute columns. Inferred from column names
        when they follow the ``dest_*``/``orig_*`` convention.
    priors : dict, optional
        Override default priors. Supported keys:

        - ``beta_mu`` : float, default 0.0 — Normal prior mean for ``beta``.
        - ``beta_sigma`` : float, default 1e6 — Normal prior std for ``beta``.
        - ``sigma_sigma`` : float, default 10.0 — HalfNormal prior std for ``sigma``.

        Spatial keys (``rho_*``) are ignored.
    symmetric_xo_xd : bool, optional
        If ``None`` (default), origin/destination design symmetry is
        auto-detected. Set explicitly to override.

    Notes
    -----
    No spatial-lag term enters the likelihood, so no log-determinant
    is required and ``logdet_method`` is ignored if passed.
    """

    def __init__(self, y, X, W, **kwargs):
        # Skip log-determinant precomputation: A = I_N has |A| = 1.
        kwargs.pop("logdet_method", None)
        kwargs.pop("restrict_positive", None)
        super().__init__(y, X, W, logdet_method="none", **kwargs)

    def _build_pymc_model(self) -> pm.Model:
        beta_mu = self.priors.get("beta_mu", 0.0)
        beta_sigma = self.priors.get("beta_sigma", 1e6)
        sigma_sigma = self.priors.get("sigma_sigma", 10.0)

        X_t = pt.as_tensor_variable(self._X.astype(np.float64))
        y_t = pt.as_tensor_variable(self._y.astype(np.float64))

        with pm.Model(coords=self._model_coords()) as model:
            beta = pm.Normal("beta", mu=beta_mu, sigma=beta_sigma, dims="coefficient")
            sigma = pm.HalfNormal("sigma", sigma=sigma_sigma)
            mu = pt.dot(X_t, beta)
            pm.Normal("obs", mu=mu, sigma=sigma, observed=y_t)

        return model

    def _simulate_y_rep(
        self,
        rho_d: float,  # unused
        rho_o: float,  # unused
        rho_w: float,  # unused
        beta: np.ndarray,
        sigma: Optional[float],
        rng: np.random.Generator,
    ) -> np.ndarray:
        Xb = self._X @ beta
        if sigma is None:
            return Xb
        return Xb + rng.normal(scale=float(sigma), size=self._N)

    def posterior_predictive(
        self,
        n_draws: Optional[int] = None,
        random_seed: Optional[int] = None,
    ) -> np.ndarray:
        """Draw posterior-predictive flows for the OLS gravity model."""
        if self._idata is None:
            raise RuntimeError("Model has not been fit yet.  Call fit() first.")

        post = self._idata.posterior
        beta_draws = post["beta"].values.reshape(-1, len(self._feature_names))
        sigma_draws = (
            post["sigma"].values.reshape(-1) if "sigma" in post.data_vars else None
        )

        total = beta_draws.shape[0]
        if n_draws is not None:
            total = min(int(n_draws), total)
            beta_draws = beta_draws[:total]
            if sigma_draws is not None:
                sigma_draws = sigma_draws[:total]

        rng = np.random.default_rng(random_seed)
        out = np.empty((total, self._N), dtype=np.float64)
        for g in range(total):
            sigma_g = float(sigma_draws[g]) if sigma_draws is not None else None
            out[g] = self._simulate_y_rep(0.0, 0.0, 0.0, beta_draws[g], sigma_g, rng)
        return out

    def _compute_spatial_effects_posterior(
        self, draws: Optional[int] = None
    ) -> dict[str, np.ndarray]:
        r"""Closed-form Thomas-Agnan & LeSage (2014, Table 83.1) effects.

        With :math:`A = I_N` the response to any shock equals the shock
        itself, so the Thomas-Agnan decomposition simplifies analytically to:

        .. math::

            \mathrm{TE} = \beta_d + \beta_o, \qquad
            \mathrm{NE} = 0, \qquad
            \mathrm{IE} = (\beta_d + \beta_o + \beta_{\text{intra}}) / n,

        with :math:`\mathrm{OE} = \beta_o (n-1)/n` and
        :math:`\mathrm{DE} = \beta_d (n-1)/n`, and the symmetric
        contributions ``β_intra / n`` distributed to the destination side.
        """
        if self._idata is None:
            raise RuntimeError("Model has not been fit yet.  Call fit() first.")

        return _compute_ols_flow_effects(
            self._idata,
            n=self._n,
            k_d=self._k_d,
            k_o=self._k_o,
            feature_names=self._feature_names,
            intra_idx=self._intra_idx,
            draws=draws,
        )


# ---------------------------------------------------------------------------
# Model 4: NegBin SAR/OLS flow variants
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

    def fit(
        self,
        draws: int = 2000,
        tune: int = 1000,
        chains: int = 4,
        random_seed: Optional[int] = None,
        sampler: str = "gibbs",
        gibbs_backend: str = "numpy",
        store_lambda: bool = False,
        idata_kwargs: Optional[dict] = None,
        progressbar: bool = True,
        attach_log_abs_det: bool = True,
        n_jobs: int = -1,
        **sample_kwargs,
    ) -> az.InferenceData:
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
        gibbs_backend : {"numpy", "jax", "auto"}, default "numpy"
            Execution backend for the Gibbs sampler (only used when
            ``sampler="gibbs"``).  ``"jax"`` compiles the sweep with JAX and
            sparsax sparse LU solves, chains on threads; ``"numpy"`` uses the
            host CHOLMOD/KLU path.  ``"auto"`` currently resolves to
            ``"numpy"``.
        store_lambda : bool, default False
            If True, include the high-dimensional fitted mean ``lambda`` in the
            stored posterior (NUTS only).
        idata_kwargs : dict, optional
            ``{"log_likelihood": True}`` stores the pointwise log-likelihood
            (one value per draw, chain, and flow) for ``az.loo`` / ``az.waic``,
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
        arviz.InferenceData
        """
        if sampler == "gibbs":
            if gibbs_backend not in {"numpy", "jax", "auto"}:
                raise ValueError(
                    "Negative-Binomial flow Gibbs supports "
                    "gibbs_backend in {'numpy', 'jax', 'auto'}; "
                    f"got {gibbs_backend!r}."
                )
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
        elif sampler == "nuts":
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
                **sample_kwargs,
            )
        else:
            raise ValueError(f"sampler must be 'nuts' or 'gibbs', got {sampler!r}")
        if attach_log_abs_det:
            self._attach_flow_log_abs_det(idata)
        return idata


class SARNegBinFlow(_NegBinFlowMixin, SARFlow):
    r"""Bayesian SAR flow model with NB2 observation noise.

    This class extends :class:`SARFlow` with a Negative Binomial likelihood:

    .. math::

        y_{ij} \sim \operatorname{NegBin}(\mu_{ij}, \alpha),

    where ``alpha`` is an overdispersion parameter sampled from a
    HalfNormal prior.
    """

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

        beta_mu = self.priors.get("beta_mu", 0.0)
        beta_sigma = self.priors.get("beta_sigma", 10.0)
        alpha_sigma = self.priors.get("alpha_sigma", 2.5)
        alpha_nu = self.priors.get("alpha_nu", 3.0)

        X_t = pt.as_tensor_variable(self._X.astype(np.float64))

        with pm.Model(coords=self._model_coords()) as model:
            if self.restrict_positive:
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
            alpha = pm.HalfStudentT("alpha", nu=alpha_nu, sigma=alpha_sigma)

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
        krylov_reuse: bool = True,
        sample_kwargs: dict[str, Any] | None = None,
        log_likelihood: bool = False,
    ) -> az.InferenceData:
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
            krylov_reuse=krylov_reuse,
            log_likelihood=log_likelihood,
        )


class SARNegBinFlowSeparable(_NegBinFlowMixin, SARFlowSeparable):
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

        beta_mu = self.priors.get("beta_mu", 0.0)
        beta_sigma = self.priors.get("beta_sigma", 10.0)
        alpha_sigma = self.priors.get("alpha_sigma", 2.5)
        alpha_nu = self.priors.get("alpha_nu", 3.0)
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
            alpha = pm.HalfStudentT("alpha", nu=alpha_nu, sigma=alpha_sigma)

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
        krylov_reuse: bool = True,
        sample_kwargs: dict[str, Any] | None = None,
        log_likelihood: bool = False,
    ) -> az.InferenceData:
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
            krylov_reuse=krylov_reuse,
            log_likelihood=log_likelihood,
        )


class NegBinFlow(_NegBinFlowMixin, OLSFlow):
    """Aspatial OD-flow Negative Binomial gravity baseline."""

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
        beta_mu = self.priors.get("beta_mu", 0.0)
        beta_sigma = self.priors.get("beta_sigma", 10.0)
        alpha_sigma = self.priors.get("alpha_sigma", 2.5)
        alpha_nu = self.priors.get("alpha_nu", 3.0)

        X_t = pt.as_tensor_variable(self._X.astype(np.float64))

        with pm.Model(coords=self._model_coords()) as model:
            beta = pm.Normal("beta", mu=beta_mu, sigma=beta_sigma, dims="coefficient")
            alpha = pm.HalfStudentT("alpha", nu=alpha_nu, sigma=alpha_sigma)
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
    ) -> az.InferenceData:
        """Sample posterior via aspatial PG-Gibbs (no spatial parameters).

        Three blocks per sweep: ω (Pólya–Gamma), β (conjugate normal),
        α (slice on log(α)).
        """
        from ...models._base._shared import gelman_default_beta_prior
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
        default_beta_mu, default_beta_sigma = gelman_default_beta_prior(
            self._y, X, list(self._feature_names)
        )
        priors = ReducedGibbsPriors(
            beta_mu=self.priors.get("beta_mu", default_beta_mu),
            beta_sigma=self.priors.get("beta_sigma", default_beta_sigma),
            alpha_sigma=self.priors.get("alpha_sigma", 2.5),
            alpha_nu=self.priors.get("alpha_nu", 3.0),
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
                    sigma2=1.0,
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

        # --- Assemble InferenceData ---
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
# Model 5: SEMFlow — Spatial-error analogue of SARFlow (unrestricted 3-rho)
# ---------------------------------------------------------------------------


class _PoissonFlowMixin:
    """Shared ``fit`` dispatch for Poisson flow models.

    Mirrors :class:`_NegBinFlowMixin`, but dispatches to the auxiliary-mixture
    Gibbs sampler (Frühwirth-Schnatter & Wagner 2006) rather than Pólya–Gamma.
    Poisson admits no exact PG representation, and the NB-with-large-alpha
    approximation degenerates precisely in the Poisson limit — the PG working
    precision outruns the Fisher information without bound, collapsing ESS.

    Gibbs-only: unlike the NB classes there is no ``sampler="nuts"`` path yet.
    """

    def fit(
        self,
        draws: int = 2000,
        tune: int = 1000,
        chains: int = 4,
        random_seed: Optional[int] = None,
        sampler: str = "gibbs",
        progressbar: bool = True,
        n_jobs: int = -1,
        idata_kwargs: Optional[dict] = None,
        **sample_kwargs,
    ) -> az.InferenceData:
        """Draw samples from the posterior.

        Parameters
        ----------
        draws, tune, chains : int
            Post-warmup draws, warmup sweeps, and number of chains.
        random_seed : int, optional
            Seed for reproducibility.
        sampler : {"gibbs"}, default "gibbs"
            Only the auxiliary-mixture Gibbs sampler is implemented.
        progressbar : bool, default True
            Show a progress bar.
        n_jobs : int, default -1
            Chain-level parallelism.
        idata_kwargs : dict, optional
            ``{"log_likelihood": True}`` stores the pointwise log-likelihood
            (one value per draw, chain, and flow) for ``az.loo`` / ``az.waic``.
            Off by default, as in PyMC.
        """
        if sampler != "gibbs":
            raise NotImplementedError(
                f"{type(self).__name__} supports sampler='gibbs' only; "
                f"got {sampler!r}.  There is no NUTS path for the "
                "auxiliary-mixture Poisson flow models yet."
            )
        if sample_kwargs:
            raise TypeError(f"Unexpected keyword arguments: {sorted(sample_kwargs)}")
        return self._fit_gibbs(
            draws=draws,
            tune=tune,
            chains=chains,
            random_seed=random_seed,
            progressbar=progressbar,
            n_jobs=n_jobs,
            log_likelihood=bool((idata_kwargs or {}).get("log_likelihood", False)),
        )


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


class SARPoissonFlow(_PoissonFlowMixin, SARFlow):
    """Unrestricted 3-ρ SAR flow model with Poisson observation noise.

    .. warning::

       The unrestricted 3-ρ parameterization mixes poorly at moderate ``n`` for
       *both* likelihoods — ρ_d, ρ_o and ρ_w trade off along a ridge that
       one-at-a-time slice updates cannot traverse.  Measured at n=36
       (N=1296): ESS ≈ 4 of 2000 draws with rhat ≈ 1.4 for this sampler *and*
       for :class:`SARNegBinFlow` on the same data.  Prefer
       :class:`SARPoissonFlowSeparable` unless ρ_w is genuinely of interest.
    """

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
            eta = self._solve_A(
                rho_d_draws[g], rho_o_draws[g], rho_w_draws[g], self._X @ beta_draws[g]
            )
            lam = np.exp(np.clip(eta, -50.0, 50.0))
            out[g] = rng.poisson(lam).astype(np.float64)
        return out

    def _fit_gibbs(
        self,
        draws: int = 2000,
        tune: int = 1000,
        chains: int = 4,
        random_seed: Optional[int] = None,
        progressbar: bool = True,
        n_jobs: int = -1,
        log_likelihood: bool = False,
    ) -> az.InferenceData:
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


class SARPoissonFlowSeparable(_PoissonFlowMixin, SARFlowSeparable):
    """Separable SAR flow model with Poisson observation noise.

    ``rho_w = -rho_d * rho_o`` is deterministic, which removes the ρ ridge that
    makes the unrestricted variant intractable.  This is the recommended
    Poisson flow model.
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
        n = self._n
        I_n = sp.eye(n, format="csr", dtype=np.float64)
        for g in range(total):
            Ld = I_n - float(rho_d_draws[g]) * self._W_sparse
            Lo = I_n - float(rho_o_draws[g]) * self._W_sparse
            eta = kron_solve_vec(Lo, Ld, self._X @ beta_draws[g], n)
            lam = np.exp(np.clip(eta, -50.0, 50.0))
            out[g] = rng.poisson(lam).astype(np.float64)
        return out

    def _fit_gibbs(
        self,
        draws: int = 2000,
        tune: int = 1000,
        chains: int = 4,
        random_seed: Optional[int] = None,
        progressbar: bool = True,
        n_jobs: int = -1,
        log_likelihood: bool = False,
    ) -> az.InferenceData:
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


class SEMFlow(FlowModel):
    r"""Bayesian spatial-error flow model with three free spatial parameters.

    .. math::

        y = X\beta + u, \qquad
        B u = \varepsilon, \qquad
        B = I_N - \lambda_d W_d - \lambda_o W_o - \lambda_w W_w,
        \quad \varepsilon \sim \mathcal{N}(0, \sigma^2 I_N)

    where :math:`W_d = I_n \otimes W`, :math:`W_o = W \otimes I_n`,
    :math:`W_w = W \otimes W`.  The Kronecker spatial structure is identical
    to :class:`SARFlow`, but the spatial filter acts on the *disturbance*
    rather than the dependent variable.  Equivalently the model implies a
    Gaussian likelihood with covariance :math:`\sigma^2 (B^\top B)^{-1}`.

    Marginal mean is :math:`\mathbb{E}[y] = X\beta`, so there are no
    :math:`X`-mediated spatial spillovers — the LeSage / Thomas-Agnan
    decomposition reduces to the closed-form expressions used by
    :class:`OLSFlow` (direct effect equals :math:`\beta`, network effect
    equals zero).  Use :class:`SARFlow` if spillovers from observed
    covariates are of interest.

    Parameters
    ----------
    y : array-like, shape (n, n) or (N,)
        Observed origin-destination flow matrix or its vec-form.
    W : libpysal.graph.Graph or scipy.sparse / dense (n×n) matrix
        Row-standardized regional weights on *n* units (Graph or matrix).
    X : np.ndarray or pandas.DataFrame, shape (N, p)
        Full origin-destination design matrix with :math:`N = n^2` rows.
        DataFrame columns are preserved as feature names.
    col_names : list of str, optional
        Column labels for ``X``. Inferred from a DataFrame if omitted;
        otherwise defaults to ``["x0", "x1", ...]``.
    k : int, optional
        Number of regional attribute columns (destination/origin variable
        pairs). Inferred from ``dest_*``/``orig_*`` column names when the
        standard LeSage layout is used.
    logdet_method : str, default "resolvent"
        Log-determinant method.  The default ``"resolvent"`` samples via the
        resolvent-gradient sampler (recommended).
    restrict_positive : bool, default True
        If True, use ``pm.Dirichlet("lam_simplex", a=ones(4))`` to enforce
        :math:`\\lambda_d, \\lambda_o, \\lambda_w \\geq 0` and
        :math:`\\lambda_d + \\lambda_o + \\lambda_w \\leq 1`. If False,
        three independent ``pm.Uniform(lam_lower, lam_upper)`` priors are
        used with a differentiable quadratic-wall stability potential.
    symmetric_xo_xd : bool, optional
        If ``None`` (default), origin and destination design blocks are
        compared and symmetry is auto-detected.
    priors : dict, optional
        Override default priors. Supported keys:

        - ``beta_mu`` : float, default 0.0 — Normal prior mean for ``beta``.
        - ``beta_sigma`` : float, default 1e6 — Normal prior std for ``beta``.
        - ``sigma_sigma`` : float, default 10.0 — HalfNormal prior std for ``sigma``.
        - ``lam_lower`` : float, default -1.0 — Lower bound of Uniform prior on each λ (only when ``restrict_positive=False``).
        - ``lam_upper`` : float, default 1.0 — Upper bound of Uniform prior on each λ (only when ``restrict_positive=False``).

    Notes
    -----
    Implementation: PyMC body uses precomputed lags of both ``y`` and
    ``X`` (``self._Wd``, ``self._Wo``, ``self._Ww`` applied to
    ``self._X``) so that the residual
    :math:`B u = B y - B X \\beta` is expressible as a linear combination
    of fixed quantities — no symbolic sparse mat-vec is required. The
    Jacobian :math:`\\log|B|` reuses the same trace-based polynomial as
    :class:`SARFlow`.
    """

    def __init__(self, y, X, W, **kwargs):
        # Default to the resolvent-gradient SEM sampler (separable subclasses pass
        # their own separable logdet_method and route to the PyMC path).
        kwargs.setdefault("logdet_method", "resolvent")
        super().__init__(y, X, W, **kwargs)
        # Precompute lags of the design matrix (constant — no parameter dependence).
        self._Wd_X, self._Wo_X, self._Ww_X = flow_lags(self._W_sparse, self._X)

    def fit(
        self,
        draws: int = 2000,
        tune: int = 1000,
        chains: int = 4,
        random_seed: Optional[int] = None,
        *,
        step_size: float = 5e-4,
        n_probes: int = 48,
        logdet_method: str = "auto",
        n_quad: int = 8,
        progressbar: bool = True,
        n_jobs: int = -1,
        idata_kwargs: Optional[dict] = None,
        **sample_kwargs,
    ) -> az.InferenceData:
        """Sample the SEM-flow posterior.

        Uses the resolvent-Kronecker gradient sampler (MALA-on-λ within GLS Gibbs
        for ``β, σ²``) by default; the separable subclass (which sets a separable
        ``logdet_method``) routes to the PyMC/NUTS path instead.

        Parameters
        ----------
        logdet_method : {"auto", "kron_traces", "jax", "numpy"}, default "auto"
            Flow log-determinant backend.  ``"auto"`` is exact (trace moments of
            ``W``, no ``N × N`` work) for undirected ``W`` and the stochastic
            resolvent estimator for directed ``W``; see
            :func:`~neighbayes.samplers.gaussian._flow_resolvent.resolve_flow_logdet`.
        idata_kwargs : dict, optional
            ``{"log_likelihood": True}`` stores the pointwise log-likelihood
            (one value per draw, chain, and flow) for ``az.loo`` / ``az.waic``.
            Off by default, as in PyMC.
        """
        if self.logdet_method != "resolvent":
            return super().fit(
                draws=draws,
                tune=tune,
                chains=chains,
                random_seed=random_seed,
                progressbar=progressbar,
                idata_kwargs=idata_kwargs,
                **sample_kwargs,
            )
        from ...samplers.gaussian._flow_resolvent import sample_sem_flow_resolvent

        self._pymc_model = None
        self._idata = sample_sem_flow_resolvent(
            self._W_sparse,
            self._y,
            self._X,
            draws=draws,
            tune=tune,
            chains=chains,
            step_size=step_size,
            n_probes=n_probes,
            coord_names=list(self._feature_names) or None,
            random_seed=random_seed,
            logdet_method=logdet_method,
            n_quad=n_quad,
            progressbar=progressbar,
            n_jobs=n_jobs,
            restrict_positive=self.restrict_positive,
            compute_log_likelihood=bool(
                (idata_kwargs or {}).get("log_likelihood", False)
            ),
        )
        return self._idata

    def _build_pymc_model(self) -> pm.Model:
        # The unrestricted SEM flow samples via the resolvent-gradient sampler
        # (``fit`` → ``sample_sem_flow_resolvent``); the legacy "traces" Jacobian
        # was removed.  Only reached if a non-resolvent, non-separable
        # ``logdet_method`` is forced.
        raise NotImplementedError(
            "SEMFlow samples via the resolvent-gradient sampler (the default "
            "logdet_method='resolvent'); the legacy 'traces' PyMC path was removed."
        )

    def _simulate_y_rep(
        self,
        lam_d: float,
        lam_o: float,
        lam_w: float,
        beta: np.ndarray,
        sigma: Optional[float],
        rng: np.random.Generator,
    ) -> np.ndarray:
        """SEM posterior-predictive: ``y_rep = X β + B^{-1} ε``."""
        Xb = self._X @ beta
        if sigma is None:
            return Xb
        eps = rng.normal(scale=float(sigma), size=self._N)
        u = self._solve_A(lam_d, lam_o, lam_w, eps)
        return Xb + u

    def _compute_spatial_effects_posterior(
        self, draws: Optional[int] = None
    ) -> dict[str, np.ndarray]:
        r"""Closed-form Thomas-Agnan & LeSage (2014, Table 83.1) effects.

        The marginal mean :math:`\mathbb{E}[y] = X\beta` is unaffected by the
        spatial-error filter, so the LeSage decomposition collapses to the
        same closed form used by :class:`OLSFlow`: direct effect equals
        :math:`\beta`, network effect equals zero, intra/origin/destination
        contributions split :math:`\beta` per Table 83.1.
        """
        if self._idata is None:
            raise RuntimeError("Model has not been fit yet.  Call fit() first.")

        return _compute_ols_flow_effects(
            self._idata,
            n=self._n,
            k_d=self._k_d,
            k_o=self._k_o,
            feature_names=self._feature_names,
            intra_idx=self._intra_idx,
            draws=draws,
        )


# ---------------------------------------------------------------------------
# Model 6: SEMFlowSeparable — separable SEM with lam_w = -lam_d * lam_o
# ---------------------------------------------------------------------------


class SEMFlowSeparable(SEMFlow):
    r"""Bayesian separable spatial-error flow model with :math:`\lambda_w = -\lambda_d \lambda_o`.

    Spatial-error analogue of :class:`SARFlowSeparable`.  The separability
    constraint reduces :math:`\log|B|` to the eigenvalue / Chebyshev factored
    form

    .. math::

        \log|B| = n \log|I_n - \lambda_d W| + n \log|I_n - \lambda_o W|

    enabling :math:`O(n)` log-determinant evaluation per draw.  All other
    properties (no :math:`X`-mediated spillovers, closed-form effects, etc.)
    are identical to :class:`SEMFlow`.

    Parameters
    ----------
    y : array-like, shape (n, n) or (N,)
        Observed origin-destination flow matrix or its vec-form.
    W : libpysal.graph.Graph or scipy.sparse / dense (n×n) matrix
        Row-standardized regional weights on *n* units (Graph or matrix).
    X : np.ndarray or pandas.DataFrame, shape (N, p)
        Full origin-destination design matrix with :math:`N = n^2` rows.
        DataFrame columns are preserved as feature names.
    col_names : list of str, optional
        Column labels for ``X``. Inferred from a DataFrame if omitted.
    k : int, optional
        Number of regional attribute columns. Inferred from
        ``dest_*``/``orig_*`` column names when the standard LeSage
        layout is used.
    logdet_method : {"eigenvalue", "chebyshev", "cheb_cholesky", "aaa", "cheb_stochastic"} or None, default None
        ``None`` auto-selects (``aaa`` for directed W, ``cheb_cholesky`` for
        symmetric, ``eigenvalue`` for small n).
        Method for the Kronecker-factored log-determinant.
    symmetric_xo_xd : bool, optional
        If ``None`` (default), origin and destination design blocks are
        compared and symmetry is auto-detected.
    priors : dict, optional
        Override default priors. Supported keys:

        - ``beta_mu`` : float, default 0.0 — Normal prior mean for ``beta``.
        - ``beta_sigma`` : float, default 1e6 — Normal prior std for ``beta``.
        - ``sigma_sigma`` : float, default 10.0 — HalfNormal prior std for ``sigma``.
        - ``lam_lower`` : float, default -0.999 — Lower bound of Uniform prior on ``lam_d`` and ``lam_o``.
        - ``lam_upper`` : float, default 0.999 — Upper bound of Uniform prior on ``lam_d`` and ``lam_o``.

    Notes
    -----
    The ``restrict_positive`` argument inherited from :class:`FlowModel`
    has no effect on this class — separable variants always use Uniform
    priors on the individual :math:`\\lambda` components.
    """

    def __init__(self, y, X, W, **kwargs):
        method = kwargs.pop("logdet_method", None)
        _VALID = {"eigenvalue", "chebyshev", "cheb_cholesky", "aaa", "cheb_stochastic"}
        if method is not None and method not in _VALID:
            raise ValueError(
                f"SEMFlowSeparable logdet_method must be None (auto) or one of "
                f"{sorted(_VALID)}; got {method!r}."
            )
        kwargs["logdet_method"] = method
        super().__init__(y, X, W, **kwargs)

    def _build_pymc_model(self) -> pm.Model:
        beta_mu = self.priors.get("beta_mu", 0.0)
        beta_sigma = self.priors.get("beta_sigma", 1e6)
        sigma_sigma = self.priors.get("sigma_sigma", 10.0)
        lam_lower = self.priors.get("lam_lower", -0.999)
        lam_upper = self.priors.get("lam_upper", 0.999)

        if self._separable_logdet_fn is None:
            raise RuntimeError(
                "SEMFlowSeparable requires precomputed logdet data; "
                "initialize with a separable logdet_method (None/auto, "
                "eigenvalue, chebyshev, cheb_cholesky, aaa, or cheb_stochastic)"
            )
        Wd_y_t = pt.as_tensor_variable(self._Wd_y.astype(np.float64))
        Wo_y_t = pt.as_tensor_variable(self._Wo_y.astype(np.float64))
        Ww_y_t = pt.as_tensor_variable(self._Ww_y.astype(np.float64))
        Wd_X_t = pt.as_tensor_variable(self._Wd_X.astype(np.float64))
        Wo_X_t = pt.as_tensor_variable(self._Wo_X.astype(np.float64))
        Ww_X_t = pt.as_tensor_variable(self._Ww_X.astype(np.float64))
        X_t = pt.as_tensor_variable(self._X.astype(np.float64))
        y_t = pt.as_tensor_variable(self._y.astype(np.float64))

        with pm.Model(coords=self._model_coords()) as model:
            lam_d = pm.Uniform("lam_d", lower=lam_lower, upper=lam_upper)
            lam_o = pm.Uniform("lam_o", lower=lam_lower, upper=lam_upper)
            lam_w = pm.Deterministic("lam_w", -lam_d * lam_o)

            beta = pm.Normal("beta", mu=beta_mu, sigma=beta_sigma, dims="coefficient")
            sigma = pm.HalfNormal("sigma", sigma=sigma_sigma)

            mu = (
                lam_d * Wd_y_t
                + lam_o * Wo_y_t
                + lam_w * Ww_y_t
                + pt.dot(X_t, beta)
                - lam_d * pt.dot(Wd_X_t, beta)
                - lam_o * pt.dot(Wo_X_t, beta)
                - lam_w * pt.dot(Ww_X_t, beta)
            )
            pm.Normal("obs", mu=mu, sigma=sigma, observed=y_t)

            pm.Potential(
                "jacobian",
                self._separable_logdet_fn(lam_d, lam_o),
            )

        return model

    def _compute_jacobian_log_det(self, posterior) -> np.ndarray:
        lam_d = np.asarray(posterior["lam_d"].values.reshape(-1), dtype=np.float64)
        lam_o = np.asarray(posterior["lam_o"].values.reshape(-1), dtype=np.float64)
        if self._separable_logdet_numpy_fn is None:
            raise RuntimeError(
                "Missing separable numeric logdet evaluator. "
                "Initialize with a separable logdet_method (None/auto, "
                "eigenvalue, chebyshev, cheb_cholesky, aaa, or cheb_stochastic)"
            )
        return self._separable_logdet_numpy_fn(lam_d, lam_o)
