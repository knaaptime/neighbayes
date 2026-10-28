"""Fast edge-case tests for logdet utilities."""

from __future__ import annotations

import numpy as np
import pytest

from neighbayes import _logdet as logdet


def _toy_w(n: int = 5) -> np.ndarray:
    W = np.zeros((n, n), dtype=float)
    for i in range(n):
        if i > 0:
            W[i, i - 1] = 1.0
        if i < n - 1:
            W[i, i + 1] = 1.0
    rs = W.sum(axis=1, keepdims=True)
    return W / np.where(rs == 0, 1, rs)


def test_make_logdet_fn_unknown_method_raises():
    W = _toy_w(4)
    with pytest.raises(ValueError, match="Unknown logdet method"):
        logdet.make_logdet_fn(W, method="not-a-method")


def test_resolve_logdet_bounds_uses_prior_by_default():
    out = logdet.resolve_logdet_bounds(
        "eigenvalue",
        n=5,
        priors={"rho_lower": -0.7, "rho_upper": 0.8},
    )
    assert out.method == "eigenvalue"
    assert out.source == "prior"
    assert out.rho_min <= -0.6
    assert out.rho_max >= 0.7


def test_resolve_logdet_bounds_rejects_bad_override_pair():
    with pytest.raises(ValueError, match="provided together"):
        logdet.resolve_logdet_bounds(
            "eigenvalue",
            n=5,
            rho_min=-0.5,
            rho_max=None,
        )
