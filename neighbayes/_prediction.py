"""Conditional Gaussian kernel for out-of-sample prediction.

For the Gaussian cross-section family the joint distribution of ``y`` at
one posterior draw is :math:`N(\\mu, \\Lambda^{-1})` with precision

.. math::

    \\Lambda = (I - \\theta W)^{\\top} (I - \\theta W) / \\sigma^2
             = \\bigl(I - \\theta (W + W^{\\top}) + \\theta^2 W^{\\top} W\\bigr) / \\sigma^2,

where :math:`\\theta` is :math:`\\rho` for the lag models (SAR, SDM),
:math:`\\lambda` for the error models (SEM, SDEM), and zero for OLS and SLX.
Partitioning the units into observed :math:`S` and held-out :math:`O`, the
conditional is exact (Goulard, Laurent & Thomas-Agnan 2017, eq. 9):

.. math::

    y_O \\mid y_S \\sim N\\bigl(\\mu_O - \\Lambda_{OO}^{-1}\\Lambda_{OS}(y_S - \\mu_S),
    \\; \\Lambda_{OO}^{-1}\\bigr).

:math:`\\Lambda_{OS}` is nonzero only where :math:`W + W^{\\top} + W^{\\top}W`
links a held-out unit to an observed one, so the conditional needs the
``O`` rows of two fixed sparse matrices and one sparse Cholesky factor of
:math:`\\Lambda_{OO}` per draw.  The factor's sparsity pattern does not depend
on the draw, so its symbolic analysis is done once.

:class:`GaussianConditional` is shared by :func:`neighbayes.diagnostics.spatial_kfold`
(held-out log density) and model prediction (conditional means and draws).
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import scipy.sparse as sp

__all__ = ["GaussianConditional"]


def _values_on_pattern(M: sp.spmatrix, pattern: sp.csc_matrix) -> np.ndarray:
    """``M``'s values at the positions of ``pattern`` (zeros where ``M`` has none).

    ``pattern`` must be CSC with sorted indices, and every structural entry of
    ``M`` must lie inside it.
    """
    n = pattern.shape[0]
    pcol = np.repeat(np.arange(pattern.shape[1]), np.diff(pattern.indptr))
    pkeys = pcol.astype(np.int64) * n + pattern.indices
    Mc = sp.coo_matrix(M)
    Mc.sum_duplicates()
    out = np.zeros(pattern.nnz, dtype=np.float64)
    if Mc.nnz == 0:
        return out
    mkeys = Mc.col.astype(np.int64) * n + Mc.row
    pos = np.searchsorted(pkeys, mkeys)
    pos = np.minimum(pos, pkeys.size - 1)
    if np.any(pkeys[pos] != mkeys):
        raise ValueError("matrix has entries outside the pattern")
    out[pos] = Mc.data
    return out


class GaussianConditional:
    """The conditional :math:`y_O \\mid y_S` of a Gaussian spatial model.

    Build once per weights matrix and held-out set, then call
    :meth:`update` for each posterior draw before using :meth:`mean`,
    :meth:`logpdf` or :meth:`sample`.

    Parameters
    ----------
    W : scipy.sparse matrix or None
        Weights over all ``n`` units, observed and held out.  ``None`` gives
        the independent-errors case (OLS, SLX), where the conditional is the
        marginal :math:`N(\\mu_O, \\sigma^2 I)`.
    oos_idx : array of int
        Indices of the held-out units :math:`O`; every other unit is
        conditioned on.  Outputs follow this order.
    n : int
        Number of units.
    """

    def __init__(self, W: Optional[sp.spmatrix], oos_idx: np.ndarray, n: int):
        oos_idx = np.asarray(oos_idx, dtype=np.int64).ravel()
        if oos_idx.size == 0:
            raise ValueError("oos_idx is empty.")
        if np.unique(oos_idx).size != oos_idx.size:
            raise ValueError("oos_idx contains duplicates.")
        if oos_idx.min() < 0 or oos_idx.max() >= n:
            raise ValueError(f"oos_idx out of range for n={n}.")
        in_sample = np.ones(n, dtype=bool)
        in_sample[oos_idx] = False
        self.n = int(n)
        self.oos_idx = oos_idx
        self.obs_idx = np.flatnonzero(in_sample)
        self.n_oos = oos_idx.size
        self._theta = None
        self._sigma = None
        self._factor = None

        if W is None:
            return
        W = sp.csr_matrix(W, dtype=np.float64)
        if W.shape != (n, n):
            raise ValueError(f"W has shape {W.shape}, expected ({n}, {n}).")
        W.eliminate_zeros()
        O, S = self.oos_idx, self.obs_idx
        W_colO_T = sp.csr_matrix(W[:, O].T)  # (n_O, n): rows O of Wᵀ
        B1 = (W[O, :] + W_colO_T).tocsc()  # rows O of W + Wᵀ
        B2 = (W_colO_T @ W).tocsc()  # rows O of WᵀW
        self._B1_OS = B1[:, S].tocsr()
        self._B2_OS = B2[:, S].tocsr()
        B1_OO = B1[:, O]
        B2_OO = B2[:, O]

        # Structural pattern of Λ_OO from |W|, so no cancellation (negative
        # weights, or θ = 0 at some draw) can drop an entry from the analysis.
        A = abs(W)
        A_colO_T = sp.csr_matrix(A[:, O].T)
        struct = (
            sp.identity(self.n_oos, format="csc")
            + (A[O, :] + A_colO_T)[:, O]
            + (A_colO_T @ A)[:, O]
        ).tocsc()
        struct.sum_duplicates()
        struct.sort_indices()
        self._indptr = struct.indptr.copy()
        self._indices = struct.indices.copy()
        self._e = _values_on_pattern(sp.identity(self.n_oos), struct)
        self._b1 = _values_on_pattern(B1_OO, struct)
        self._b2 = _values_on_pattern(B2_OO, struct)

        # Analyze at a θ inside the stable region: ρ(W) ≤ min(‖W‖₁, ‖W‖∞), so
        # I - t W is nonsingular and its Gram block is SPD.
        radius = min(
            float(np.abs(W).sum(axis=0).max()), float(np.abs(W).sum(axis=1).max())
        )
        t = 0.5 / radius if radius > 0 else 0.5
        from .samplers._utils._spatial_normal import CholmodFactor

        self._factor = CholmodFactor(
            self._on_pattern(self._e - t * self._b1 + t * t * self._b2)
        )

    @property
    def is_iid(self) -> bool:
        """True when the errors are independent (no ``W``)."""
        return self._factor is None

    def _on_pattern(self, data: np.ndarray) -> sp.csc_matrix:
        return sp.csc_matrix(
            (data, self._indices, self._indptr), shape=(self.n_oos, self.n_oos)
        )

    def update(self, theta: float, sigma: float) -> None:
        """Set the draw: spatial parameter ``theta`` and error scale ``sigma``."""
        self._theta = float(theta)
        self._sigma = float(sigma)
        if self.is_iid:
            return
        s2 = self._sigma**2
        th = self._theta
        self._Lam_data = (self._e - th * self._b1 + th * th * self._b2) / s2
        self._factor.factorize(self._on_pattern(self._Lam_data))

    def _precision_OS_times(self, r_S: np.ndarray) -> np.ndarray:
        th, s2 = self._theta, self._sigma**2
        return (-th * (self._B1_OS @ r_S) + th * th * (self._B2_OS @ r_S)) / s2

    def _solve(self, rhs: np.ndarray) -> np.ndarray:
        try:
            return self._factor.solve(rhs)
        except Exception as exc:  # noqa: BLE001 - re-raised with context
            raise np.linalg.LinAlgError(
                "The held-out precision block is not numerically positive "
                f"definite at theta={self._theta!r}, sigma={self._sigma!r}."
            ) from exc

    def mean(self, mu: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Conditional mean of ``y_O`` given the observed values.

        Parameters
        ----------
        mu : ndarray of shape (n,)
            Marginal mean at this draw.
        y : ndarray of shape (n,)
            Outcomes; entries at the held-out units are ignored.
        """
        mu_O = mu[self.oos_idx]
        if self.is_iid:
            return mu_O.copy()
        r_S = y[self.obs_idx] - mu[self.obs_idx]
        return mu_O - self._solve(self._precision_OS_times(r_S))

    def logpdf(self, y_O: np.ndarray, mean_O: np.ndarray) -> float:
        """Joint log density of ``y_O`` under the conditional at this draw."""
        d = y_O - mean_O
        k = self.n_oos
        if self.is_iid:
            s = self._sigma
            return float(
                -0.5 * np.sum((d / s) ** 2)
                - k * np.log(s)
                - 0.5 * k * np.log(2 * np.pi)
            )
        quad = float(d @ (self._on_pattern(self._Lam_data) @ d))
        logdet = float(self._factor.logdet())
        return 0.5 * logdet - 0.5 * k * np.log(2.0 * np.pi) - 0.5 * quad

    def sample(self, mean_O: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """One draw of ``y_O`` from the conditional at this draw."""
        if self.is_iid:
            return mean_O + self._sigma * rng.standard_normal(self.n_oos)
        # CholmodFactor.sample draws N(Λ⁻¹ t, Λ⁻¹); t = 0 leaves the noise.
        try:
            return mean_O + self._factor.sample(np.zeros(self.n_oos), rng=rng)
        except Exception as exc:  # noqa: BLE001 - re-raised with context
            raise np.linalg.LinAlgError(
                "The held-out precision block is not numerically positive "
                f"definite at theta={self._theta!r}, sigma={self._sigma!r}."
            ) from exc
