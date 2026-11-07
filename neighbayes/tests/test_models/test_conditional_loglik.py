"""The stored Gaussian term against brute-force conditioning on the dense joint."""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp
from scipy import stats

from neighbayes.models._base._conditional import gaussian_conditional_loglik

N, T, SIGMA2, PAR = 9, 3, 1.3, 0.37
FLAT = 1e6


def _W():
    rows, cols = [], []
    for i in range(3):
        for j in range(3):
            for a, b in ((i + 1, j), (i - 1, j), (i, j + 1), (i, j - 1)):
                if 0 <= a < 3 and 0 <= b < 3:
                    rows.append(3 * i + j)
                    cols.append(3 * a + b)
    W = sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(N, N))
    return sp.csr_matrix(sp.diags(1 / np.asarray(W.sum(axis=1)).ravel()) @ W)


def _dense(y, mu, Sigma):
    out = []
    for i in range(len(y)):
        keep = np.arange(len(y)) != i
        coef = np.linalg.solve(Sigma[np.ix_(keep, keep)], Sigma[keep, i])
        m = mu[i] + coef @ (y[keep] - mu[keep])
        v = Sigma[i, i] - coef @ Sigma[keep, i]
        out.append(stats.norm(m, np.sqrt(v)).logpdf(y[i]))
    return np.array(out)


@pytest.mark.parametrize("process", ["lag", "error"])
@pytest.mark.parametrize(
    "unit_var, time_var, tol",
    [
        (0.0, 0.0, 1e-10),
        (FLAT, 0.0, 1e-3),
        (0.0, FLAT, 1e-3),
        (FLAT, FLAT, 1e-3),
        (0.9, 0.0, 1e-10),
    ],
)
def test_matches_dense_conditional(process, unit_var, time_var, tol):
    if process == "error" and unit_var == 0.9:
        pytest.skip("Random effects under an error process are conditioned on instead.")
    W = _W()
    rng = np.random.default_rng(0)
    n = N * T
    y, fitted = rng.normal(size=n), rng.normal(size=n)
    A = np.eye(N) - PAR * W.toarray()
    Ainv = np.linalg.inv(A)
    block = np.kron(np.eye(T), Ainv)
    effects = unit_var * np.kron(np.ones((T, T)), np.eye(N)) + time_var * np.kron(
        np.eye(T), np.ones((N, N))
    )
    if process == "lag":
        Sigma = block @ (SIGMA2 * np.eye(n) + effects) @ block.T
        mu = block @ fitted
        e = np.kron(np.eye(T), A) @ y - fitted
    else:
        Sigma = effects + SIGMA2 * block @ block.T
        mu = fitted
        e = np.kron(np.eye(T), A) @ (y - fitted)
    if unit_var == FLAT:
        c = 1.0 / T
    else:
        c = unit_var / (SIGMA2 + T * unit_var)
    ours = gaussian_conditional_loglik(
        e[None, :], [PAR], [SIGMA2], W, T=T, c=c, time_effects=time_var > 0
    )[0]
    np.testing.assert_allclose(ours, _dense(y, mu, Sigma), atol=tol)


def test_three_operator_filter_matches_dense():
    # A flow's filter, I - rho_d W_d - rho_o W_o - rho_w W_w, on 3 x 3 = 9 pairs.
    small = _W()[:3, :3].toarray()
    small[0, 1] = small[1, 0] = small[1, 2] = small[2, 1] = 1.0
    small = small / small.sum(axis=1, keepdims=True)
    I3 = np.eye(3)
    Ws = [
        sp.csr_matrix(np.kron(I3, small)),
        sp.csr_matrix(np.kron(small, I3)),
        sp.csr_matrix(np.kron(small, small)),
    ]
    rho = np.array([0.3, 0.2, -0.1])
    A = np.eye(9) - sum(r * w.toarray() for r, w in zip(rho, Ws))
    rng = np.random.default_rng(1)
    y, fitted = rng.normal(size=9), rng.normal(size=9)
    Ainv = np.linalg.inv(A)
    ours = gaussian_conditional_loglik(
        (A @ y - fitted)[None, :], rho[None, :], [SIGMA2], Ws
    )[0]
    dense = _dense(y, Ainv @ fitted, SIGMA2 * Ainv @ Ainv.T)
    np.testing.assert_allclose(ours, dense, atol=1e-10)


def test_student_t_tends_to_the_normal_conditional():
    from neighbayes.models._base._conditional import student_t_conditional_loglik

    W = _W()
    rng = np.random.default_rng(3)
    e = rng.normal(size=(2, N * T))
    par = np.array([0.37, -0.2])
    sigma2 = np.array([1.3, 0.7])
    normal = gaussian_conditional_loglik(e, par, sigma2, W, T=T)
    heavy = student_t_conditional_loglik(e, par, np.sqrt(sigma2), 1e7, W, T=T)
    np.testing.assert_allclose(heavy, normal, atol=1e-5)


def test_student_t_matches_adaptive_quadrature():
    from scipy import integrate

    from neighbayes.models._base._conditional import student_t_conditional_loglik

    W = _W().toarray()
    A = np.eye(N) - PAR * W
    nu, sigma = 3.0, 1.1
    rng = np.random.default_rng(4)
    y, fitted = rng.standard_t(nu, size=N) * 2, rng.normal(size=N)

    def log_joint(v, i):
        yy = y.copy()
        yy[i] = v
        return stats.t(nu, scale=sigma).logpdf(A @ yy - fitted).sum()

    ours = student_t_conditional_loglik(
        (A @ y - fitted)[None, :], [PAR], [sigma], nu, sp.csr_matrix(W)
    )[0]
    for i in range(N):
        peak = log_joint(y[i], i)
        mass, _ = integrate.quad(
            lambda v: np.exp(log_joint(v, i) - peak),
            -np.inf,
            np.inf,
            epsabs=1e-13,
            epsrel=1e-11,
            limit=400,
        )
        assert abs(ours[i] - (-np.log(mass))) < 1e-7


@pytest.mark.parametrize("time_var", [0.0, FLAT])
def test_dynamic_panel_matches_dense(time_var):
    # e_t = A y_t - L y_{t-1} - fitted_t for t = 2..T, y_1 given; L = phi I + theta W.
    from neighbayes.models._base._conditional import LinearOperator

    W = _W()
    phi, theta = 0.5, 0.2
    rng = np.random.default_rng(2)
    y1 = rng.normal(size=N)
    y = rng.normal(size=N * T)
    fitted = rng.normal(size=N * T)
    A = np.eye(N) - PAR * W.toarray()
    L = phi * np.eye(N) + theta * W.toarray()
    D = np.kron(np.eye(T), A) - np.kron(np.eye(T, k=-1), L)
    h = fitted + np.concatenate([L @ y1, np.zeros(N * (T - 1))])
    e = D @ y - h
    Dinv = np.linalg.inv(D)
    Sigma = (
        Dinv
        @ (SIGMA2 * np.eye(N * T) + time_var * np.kron(np.eye(T), np.ones((N, N))))
        @ Dinv.T
    )
    lag = LinearOperator([sp.identity(N), W], [[phi, theta]])
    ours = gaussian_conditional_loglik(
        e[None, :], [PAR], [SIGMA2], W, T=T, time_effects=time_var > 0, next_period=lag
    )[0]
    np.testing.assert_allclose(
        ours, _dense(y, Dinv @ h, Sigma), atol=1e-3 if time_var else 1e-10
    )
