"""Model mocks for the Gaussian LM tests."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np


def exact_logdet_grad(W_sparse):
    """``r -> d log|I - rW| / dr = -tr(W (I - rW)^{-1})``, exactly (dense)."""
    Wd = W_sparse.toarray()
    n = Wd.shape[0]
    return lambda r: np.array(
        [-np.trace(Wd @ np.linalg.inv(np.eye(n) - ri * Wd)) for ri in np.atleast_1d(r)]
    )


def gaussian_mock(X, W_sparse) -> MagicMock:
    """A model mock carrying what the score engine reads beyond the data.

    It has the exact logdet gradient of :func:`exact_logdet_grad`, and every
    column of ``X`` but the first is a non-intercept one.
    """
    model = MagicMock()
    model._nonintercept_indices = list(range(1, X.shape[1]))
    model._logdet_grad_numpy_vec_fn = exact_logdet_grad(W_sparse)
    return model


def panel_mock(X, W_sparse, N: int, T: int) -> MagicMock:
    """A pooled-panel mock: :func:`gaussian_mock` with the panel dimensions.

    Periods are stacked time-first; a pooled panel counts every observation
    and applies the Jacobian ``T`` times.
    """
    model = gaussian_mock(X, W_sparse)
    model._is_panel = True
    model._design = "static"
    model._N, model._T = N, T
    model._jacobian_T, model._jacobian_shift = T, 0.0
    model._n_effective = N * T
    model._unit_idx = np.arange(N * T) % N
    return model
