"""Separable-flow log-determinant factories.

The unrestricted 3-parameter flow log-determinant is handled by the scalable
resolvent-Kronecker gradient (:mod:`neighbayes._logdet._flow_resolvent`); the old
``"traces"`` value method was removed because it amplifies stochastic-moment noise
for large directed ``W``.  What remains here is the *separable* flow factorization
``log|L_o ⊗ L_d| = n·log|I − ρ_d W| + n·log|I − ρ_o W|``, delegated to the general
single-parameter logdet factories.  Used by ``models/flow``.
"""

import numpy as np
import scipy.sparse as sp


def _as_flow_csr(W_sparse):
    """Coerce ``W_sparse`` to a float64 CSR matrix."""
    if sp.issparse(W_sparse):
        return W_sparse.tocsr().astype(np.float64)
    return sp.csr_matrix(np.asarray(W_sparse, dtype=np.float64))


def make_flow_separable_logdet(
    W_sparse,
    n: int,
    method: str | None = None,
    rho_min: float = -1.0,
    rho_max: float = 1.0,
    cheb_order: int = 20,
):
    r"""Pre-compute logdet data for separable flow models and return a pytensor callable.

    For the separable constraint :math:`\rho_w = -\rho_d \rho_o` the full
    system log-determinant factors exactly as

    .. math::

        \log|L_o \otimes L_d|
        = n\,\log|I_n - \rho_d W| + n\,\log|I_n - \rho_o W|

    Both halves are the ordinary single-parameter logdet, so this delegates to
    the general :func:`~neighbayes._logdet._factories.make_logdet_fn` factory —
    unlocking every method it supports (``eigenvalue``, ``chebyshev``,
    ``cheb_cholesky``, ``aaa``, ``cheb_stochastic``) with a single differentiable
    precompute reused for both :math:`\rho` components.  ``aaa`` is the
    auto-selected method for directed (non-symmetric) flow ``W``.

    Returns a closure ``fn(rho_d, rho_o) -> pt.TensorVariable`` suitable for
    ``pm.Potential``.

    Parameters
    ----------
    W_sparse : array-like or scipy.sparse matrix
        Row-standardized :math:`n \times n` spatial weights matrix.
    n : int
        Number of spatial units.
    method : str, optional
        Any logdet method understood by :func:`make_logdet_fn`, or ``None`` for
        auto-selection.
    rho_min, rho_max : float
        Bounds for the approximation interval.
    cheb_order : int, default 20
        Accepted for backward compatibility; ignored (the general factory uses
        the package-standard order).
    """
    from ._factories import make_logdet_fn

    # Single-parameter differentiable log|I_n - rho W|, scaled by n via T=n.
    single = make_logdet_fn(
        _as_flow_csr(W_sparse),
        method=method,
        rho_min=rho_min,
        rho_max=rho_max,
        T=n,
    )
    return lambda rho_d, rho_o: single(rho_d) + single(rho_o)


def make_flow_separable_logdet_numpy(
    W_sparse,
    n: int,
    method: str | None = None,
    rho_min: float = -1.0,
    rho_max: float = 1.0,
    cheb_order: int = 20,
):
    r"""Pre-compute numeric logdet data for separable flow models.

    Returns a vectorized numpy closure for post-fit Jacobian reconstruction:

    .. math::

        n\,\log|I_n - \rho_d W| + n\,\log|I_n - \rho_o W|

    Delegates to :func:`~neighbayes._logdet._factories.make_logdet_numpy_vec_fn`,
    which supports every method and auto-resolves ``None`` (computing eigenvalues
    only if it lands on the ``eigenvalue`` method).  ``cheb_order`` is accepted
    for backward compatibility but ignored.
    """
    from ._factories import make_logdet_numpy_vec_fn

    logdet_vec = make_logdet_numpy_vec_fn(
        _as_flow_csr(W_sparse),
        eigs=None,
        method=method,
        rho_min=rho_min,
        rho_max=rho_max,
    )

    def _eval(rho_d, rho_o) -> np.ndarray:
        rd = np.asarray(rho_d, dtype=np.float64).reshape(-1)
        ro = np.asarray(rho_o, dtype=np.float64).reshape(-1)
        if rd.shape != ro.shape:
            raise ValueError(
                "rho_d and rho_o must have the same shape for separable logdet evaluation."
            )
        return n * (logdet_vec(rd) + logdet_vec(ro))

    return _eval
