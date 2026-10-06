r"""Auxiliary-mixture augmentation for Poisson log-link regression.

Implements the Frühwirth-Schnatter & Wagner (2006) data augmentation, which
represents a Poisson observation through the inter-arrival times of the
underlying Poisson process.  Unlike Pólya–Gamma — which has no exact Poisson
representation, and whose NB-limit approximation degenerates as the working
precision outruns the Fisher information — this scheme is exact and its working
precision converges to the Poisson information :math:`\mu`.

Construction
------------
Read :math:`y_i \sim \operatorname{Poisson}(\lambda_i)`,
:math:`\lambda_i = e^{\eta_i}`, as the number of jumps in :math:`[0, 1]` of a
Poisson process with intensity :math:`\lambda_i`.  Let :math:`t_k` be the
arrival times.  Augment with

- :math:`\tau_{i2} = t_{y_i}`, the last arrival at or before 1 (only when
  :math:`y_i > 0`).  Marginally it is :math:`\operatorname{Gamma}(y_i, \lambda_i)`,
  which gives the location model below; *given* :math:`y_i` it is the largest
  of :math:`y_i` uniforms, :math:`\operatorname{Beta}(y_i, 1)`, free of
  :math:`\lambda_i` (the Gamma density times the probability
  :math:`e^{-\lambda_i(1-\tau)}` of no further arrival before 1 leaves
  :math:`\tau^{y_i - 1}`);
- :math:`\xi_{i1}`, the inter-arrival time from :math:`\tau_{i2}` (or from 0
  when :math:`y_i = 0`) to the first arrival *after* 1, which by
  memorylessness is :math:`(1 - \tau_{i2}) + \operatorname{Exp}(\lambda_i)`.

Taking negative logs turns each into a location model in :math:`\eta_i`:

.. math::

    -\log \xi_{i1} = \eta_i + \varepsilon, \qquad
      \varepsilon \sim -\log \operatorname{Gamma}(1, 1) \\
    -\log \tau_{i2} = \eta_i + \varepsilon', \qquad
      \varepsilon' \sim -\log \operatorname{Gamma}(y_i, 1)

Each error is approximated by a finite normal mixture (see :mod:`._mixture`).
Conditional on the component indicator :math:`r`, the augmented observation is
exactly Gaussian with working response :math:`s = -\log(\cdot) - m_r` and
working precision :math:`\omega = 1/v_r` — the same ``(s, omega)`` contract the
Pólya–Gamma samplers hand to the shared β and ρ blocks.

So every observation contributes one augmented row, plus a second row when
:math:`y_i > 0`.
"""

from __future__ import annotations

import numpy as np

from ._mixture import mixture_for_shape, mixture_for_unit_shape

__all__ = ["AugmentedDesign", "draw_augmentation", "build_augmented_index"]


class AugmentedDesign:
    """Row bookkeeping for the ragged augmented design.

    The augmented design stacks the ``N`` inter-arrival rows on top of the
    ``N_pos`` last-arrival rows (one per strictly positive count), so the
    working design is ``U_aug = U[rows]`` with ``rows`` given by :attr:`rows`.

    Parameters
    ----------
    y : ndarray, shape (N,)
        Integer response vector.

    Attributes
    ----------
    N : int
        Number of observations.
    pos : ndarray of int
        Indices with ``y > 0``.
    rows : ndarray of int, shape (N + N_pos,)
        Index into the ``N``-row design for each augmented row.
    """

    __slots__ = ("N", "pos", "rows", "y_pos")

    def __init__(self, y: np.ndarray):
        y = np.asarray(y)
        self.N = int(y.shape[0])
        self.pos = np.flatnonzero(y > 0).astype(np.intp)
        self.y_pos = y[self.pos].astype(np.float64)
        self.rows = np.concatenate([np.arange(self.N, dtype=np.intp), self.pos])

    @property
    def n_aug(self) -> int:
        """Total number of augmented rows (``N + N_pos``)."""
        return int(self.rows.shape[0])


def build_augmented_index(y: np.ndarray) -> AugmentedDesign:
    """Build the augmented row index for a response vector."""
    return AugmentedDesign(y)


def draw_augmentation(
    y: np.ndarray,
    eta: np.ndarray,
    design: AugmentedDesign,
    *,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    r"""Draw the auxiliary variables and return the Gaussian working data.

    Parameters
    ----------
    y : ndarray, shape (N,)
        Integer responses.
    eta : ndarray, shape (N,)
        Current linear predictor, :math:`\log \lambda`.
    design : AugmentedDesign
        Row bookkeeping from :func:`build_augmented_index`.
    rng : numpy.random.Generator
        Random state.

    Returns
    -------
    s : ndarray, shape (N + N_pos,)
        Working response for each augmented row.
    omega : ndarray, shape (N + N_pos,)
        Working precision (``1 / v_r``) for each augmented row.
    """
    eta = np.clip(np.asarray(eta, dtype=np.float64), -30.0, 30.0)
    lam = np.exp(eta)
    pos = design.pos

    # --- tau_2: last arrival at or before 1, only for y > 0 -------------
    # Given y arrivals in [0, 1] the arrival times are uniform order
    # statistics, so the last one is Beta(y, 1) whatever lambda is.  (Drawing
    # it from Gamma(y, lambda) truncated to (0, 1) drops the factor
    # exp(-lambda (1 - tau)) for no arrival in (tau, 1]; that biased eta and
    # inflated its spread, badly so for counts in the tens and above.)
    tau2 = np.empty(pos.shape[0], dtype=np.float64)
    if pos.size:
        tau2 = np.maximum(rng.random(pos.shape[0]) ** (1.0 / design.y_pos), 1e-300)

    # --- xi_1: inter-arrival from tau_2 (or 0) to the first arrival > 1 --
    # Memorylessness: the residual wait past (1 - tau_2) is a fresh Exp(lam).
    start = np.zeros(design.N, dtype=np.float64)
    if pos.size:
        start[pos] = tau2
    xi1 = (1.0 - start) + rng.exponential(1.0, size=design.N) / lam

    # --- turn each into a Gaussian working observation -------------------
    # Row block 1: -log(xi1) = eta + eps,  eps ~ -log Gamma(1, 1)
    w1, m1, v1 = mixture_for_unit_shape()
    s1, o1 = _mixture_working_data(-np.log(xi1), eta, w1, m1, v1, rng)

    if pos.size == 0:
        return s1, o1

    # Row block 2: -log(tau2) = eta + eps', eps' ~ -log Gamma(y_i, 1)
    s2, o2 = _mixture_working_data_by_shape(-np.log(tau2), eta[pos], design.y_pos, rng)
    return np.concatenate([s1, s2]), np.concatenate([o1, o2])


def _mixture_working_data(
    z: np.ndarray,
    eta: np.ndarray,
    w: np.ndarray,
    m: np.ndarray,
    v: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample the mixture indicator, then return ``(s, omega)``.

    ``P(r = j | z, eta) ∝ w_j N(z - eta; m_j, v_j)``.
    """
    resid = z - eta
    # log w_j + log N(resid; m_j, v_j), shape (n, K)
    lp = (
        np.log(w)[None, :]
        - 0.5 * np.log(v)[None, :]
        - 0.5 * (resid[:, None] - m[None, :]) ** 2 / v[None, :]
    )
    lp -= lp.max(axis=1, keepdims=True)
    p = np.exp(lp)
    p /= p.sum(axis=1, keepdims=True)
    # Vectorised categorical draw via the inverse-CDF trick.
    idx = (p.cumsum(axis=1) < rng.random((resid.shape[0], 1))).sum(axis=1)
    idx = np.clip(idx, 0, w.shape[0] - 1)
    return z - m[idx], 1.0 / v[idx]


def _mixture_working_data_by_shape(
    z: np.ndarray,
    eta: np.ndarray,
    shapes: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """As :func:`_mixture_working_data`, with a per-observation shape.

    Groups by integer shape so each distinct ``y`` value costs one vectorised
    mixture step.  Counts above the tabulated cutoff use the single-normal
    limit, which needs no indicator draw at all.
    """
    s = np.empty(z.shape[0], dtype=np.float64)
    o = np.empty(z.shape[0], dtype=np.float64)
    for shape_val in np.unique(shapes):
        sel = shapes == shape_val
        w, m, v = mixture_for_shape(float(shape_val))
        if w.shape[0] == 1:
            s[sel] = z[sel] - m[0]
            o[sel] = 1.0 / v[0]
        else:
            s[sel], o[sel] = _mixture_working_data(z[sel], eta[sel], w, m, v, rng)
    return s, o
