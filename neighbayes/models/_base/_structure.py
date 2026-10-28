"""The spatial structure of a model's data: ``W`` applied within each period.

A cross-section is the ``T = 1`` panel, so one class serves both: the spatial
lag, the batched lag of residual draws, the ``(N*T)×(N*T)`` operator
``I_T ⊗ W`` and the ``W`` operand handed to the log-determinant factory.
Everything else that is behaviour-identical across designs lives in
:class:`neighbayes.models._base._shared.SharedSpatialMethods`.
"""

from __future__ import annotations

import numpy as np


class SpatialStructure:
    """``W ⊗ I_T`` on stacked data ordered by period, then unit.

    Accepts either an ``N×N`` ``W`` (applied per period in one batched matmul)
    or a full ``(N*T)×(N*T)`` block matrix.  The logdet factory receives the
    sparse ``W``.
    """

    def __init__(self, W_sparse, N: int, T: int = 1):
        self._W_sparse = W_sparse
        self._N = int(N)
        self._T = int(T)
        self._W_sparse_NT_cache = None

    def spatial_lag(self, v: np.ndarray) -> np.ndarray:
        """The spatial lag of a stacked vector or a column-stacked matrix."""
        W = self._W_sparse
        N, T = self._N, self._T
        v = np.asarray(v, dtype=float)
        if W.shape[0] == N:
            if v.ndim == 1:
                # Stack ordered (T, N); apply W per period in one matmul.
                chunks = v.reshape(T, N)  # (T, N)
                return np.asarray((W @ chunks.T).T, dtype=float).ravel()
            # 2-D path: (N*T, k) → reshape so all periods/columns become a
            # single dense block, perform ONE sparse matmul, then reshape back.
            k = v.shape[1]
            chunks = v.reshape(T, N, k)  # (T, N, k)
            mat = chunks.transpose(1, 0, 2).reshape(N, T * k)
            out = np.asarray(W @ mat, dtype=float)  # (N, T*k)
            return out.reshape(N, T, k).transpose(1, 0, 2).reshape(T * N, k)
        # Full (N*T)×(N*T) block matrix provided.
        return np.asarray(W @ v, dtype=float)

    def batch_spatial_lag(
        self, resid: np.ndarray, T_eff: int | None = None
    ) -> np.ndarray:
        """Apply the spatial lag to a batch of stacked residual draws.

        Parameters
        ----------
        resid : np.ndarray
            Residual draws with shape ``(n_draws, N*T_eff)``.
        T_eff : int, optional
            Effective time periods in the stacked layout.  Defaults to ``T``;
            dynamic panel paths pass ``T - 1``.

        Returns
        -------
        np.ndarray
            Spatially lagged residuals with the same shape as ``resid``.
        """
        R = np.asarray(resid, dtype=np.float64)
        if R.ndim != 2:
            raise ValueError(
                f"resid must be 2D (n_draws, N*T_eff), got shape {R.shape}."
            )
        N = self._N
        Te = int(self._T if T_eff is None else T_eff)
        expected = N * Te
        if R.shape[1] != expected:
            raise ValueError(
                "resid second dimension must equal N*T_eff; "
                f"got {R.shape[1]} and expected {expected} (N={N}, T_eff={Te})."
            )

        W = self._W_sparse
        if W.shape[0] == N:
            # Reshape (draws, N*T_eff) -> (draws*T_eff, N), apply one sparse
            # matrix multiply, then reshape back.
            draws = R.shape[0]
            R_flat = R.reshape(draws * Te, N)
            WR_flat = np.asarray(W @ R_flat.T, dtype=np.float64).T
            return WR_flat.reshape(draws, Te * N)

        # Full panel matrix path (N*T x N*T) if supplied by caller.
        if W.shape[0] != expected:
            raise ValueError(
                f"W has shape {W.shape}; expected ({N},{N}) or ({expected},{expected}) "
                "for the provided N and T_eff."
            )
        return np.asarray(W @ R.T, dtype=np.float64).T

    def logdet_W_operand(self):
        return self._W_sparse

    def W_sparse_NT(self):
        """Cached sparse ``(N*T)×(N*T)`` Kronecker block ``I_T ⊗ W``.

        Exposes a single linear operator that applies the per-period lag to a
        stacked panel vector without the ``O((N*T)²)`` dense footprint.  It is
        ``W`` itself in a cross-section, and the caller's matrix when a full
        ``(N*T)×(N*T)`` one was supplied.
        """
        if self._W_sparse_NT_cache is None:
            import scipy.sparse as sp

            W = self._W_sparse
            if self._T == 1 or W.shape[0] != self._N:
                self._W_sparse_NT_cache = W if self._T == 1 else sp.csr_matrix(W)
            else:
                # Ensure int64 indices to prevent overflow in scipy kron
                W = sp.csr_matrix(W, dtype=np.float64)
                W.indices = W.indices.astype(np.int64)
                W.indptr = W.indptr.astype(np.int64)
                # Force ``csr_matrix`` (not ``csr_array``) so the result is
                # accepted by :mod:`pytensor.sparse`, which currently only
                # supports the legacy ``scipy.sparse`` matrix API.
                self._W_sparse_NT_cache = sp.csr_matrix(
                    sp.kron(sp.eye(self._T, format="csr"), W, format="csr")
                )
        return self._W_sparse_NT_cache
