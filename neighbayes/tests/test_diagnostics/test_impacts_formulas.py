"""Formula-level checks for impacts against explicit matrix calculations.

These tests avoid MCMC by injecting posterior means directly into model
InferenceData objects. They verify that impacts match legacy-style formulas.
"""

from __future__ import annotations

import numpy as np

from neighbayes.models import SAR, SDEM, SDM, SLX
from neighbayes.tests.helpers import (
    W_to_graph,
    make_line_W,
)
from neighbayes.tests.helpers import (
    set_posterior_means as _set_posterior_means,
)


def _build_inputs(n: int = 5) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    W = make_line_W(n)
    x1 = np.linspace(-1.0, 1.0, n)
    X = np.column_stack([np.ones(n), x1])
    y = np.zeros(n)
    return y, X, W


def test_sar_impacts_match_matrix_formula() -> None:
    y, X, W = _build_inputs()
    model = SAR(y=y, X=X, W=W_to_graph(W))

    beta = np.array([1.2, -0.7])
    rho = 0.35
    _set_posterior_means(model, beta=beta, rho=rho)

    effects = model.impacts()

    S = np.linalg.inv(np.eye(W.shape[0]) - rho * W)
    # Intercept excluded from effects, so use only non-intercept beta
    expected_direct = np.diag(S).mean() * beta[1:]
    expected_total = S.sum(axis=1).mean() * beta[1:]
    expected_indirect = expected_total - expected_direct

    assert np.allclose(effects["direct"].values, expected_direct)
    assert np.allclose(effects["total"].values, expected_total)
    assert np.allclose(effects["indirect"].values, expected_indirect)


def test_sdm_impacts_match_matrix_formula() -> None:
    y, X, W = _build_inputs()
    model = SDM(y=y, X=X, W=W_to_graph(W))

    # beta = [intercept, x1, W*x1]
    beta = np.array([0.8, -0.4, 0.25])
    rho = 0.3
    _set_posterior_means(model, beta=beta, rho=rho)

    effects = model.impacts()

    n = W.shape[0]
    M = np.linalg.inv(np.eye(n) - rho * W)
    beta1_x1 = beta[1]
    beta2_x1 = beta[2]
    Sx = M @ (beta1_x1 * np.eye(n) + beta2_x1 * W)

    expected_direct = np.array([np.diag(Sx).mean()])
    expected_total = np.array([Sx.sum(axis=1).mean()])
    expected_indirect = expected_total - expected_direct

    assert np.allclose(effects["direct"].values, expected_direct)
    assert np.allclose(effects["total"].values, expected_total)
    assert np.allclose(effects["indirect"].values, expected_indirect)


def test_slx_impacts_match_matrix_formula() -> None:
    y, X, W = _build_inputs()
    model = SLX(y=y, X=X, W=W_to_graph(W))

    # beta = [intercept, x1, W*x1]
    beta = np.array([0.5, 1.1, -0.2])
    _set_posterior_means(model, beta=beta)

    effects = model.impacts()

    beta1_x1 = beta[1]
    beta2_x1 = beta[2]
    Sx = beta1_x1 * np.eye(W.shape[0]) + beta2_x1 * W

    expected_direct = np.array([np.diag(Sx).mean()])
    expected_total = np.array([Sx.sum(axis=1).mean()])
    expected_indirect = expected_total - expected_direct

    assert np.allclose(effects["direct"].values, expected_direct)
    assert np.allclose(effects["total"].values, expected_total)
    assert np.allclose(effects["indirect"].values, expected_indirect)


def test_sdem_impacts_match_matrix_formula() -> None:
    y, X, W = _build_inputs()
    model = SDEM(y=y, X=X, W=W_to_graph(W))

    # beta = [intercept, x1, W*x1]
    beta = np.array([0.3, -1.0, 0.4])
    _set_posterior_means(model, beta=beta)

    effects = model.impacts()

    beta1_x1 = beta[1]
    beta2_x1 = beta[2]
    Sx = beta1_x1 * np.eye(W.shape[0]) + beta2_x1 * W

    expected_direct = np.array([np.diag(Sx).mean()])
    expected_total = np.array([Sx.sum(axis=1).mean()])
    expected_indirect = expected_total - expected_direct

    assert np.allclose(effects["direct"].values, expected_direct)
    assert np.allclose(effects["total"].values, expected_total)
    assert np.allclose(effects["indirect"].values, expected_indirect)
