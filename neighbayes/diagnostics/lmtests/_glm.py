r"""GLM helpers for the Bayesian LM tests of the logit, probit and NB families.

Score quantities
----------------
The score tests of the latent SAR and SEM (Knaap, esda_triad) are built from
per-unit quantities at the restricted fit's linear predictor :math:`\eta_i`,
with the units independent under the null:

=========  =================================  =================================================  ==========================================================
quantity   logit                              probit (:math:`\phi_i, \Phi_i` at :math:`\eta_i`)  NB2 (:math:`\mu = e^\eta`, dispersion r)
=========  =================================  =================================================  ==========================================================
``e``      :math:`y_i - p_i`                  :math:`\lambda_i`, the generalized residual        :math:`r(y_i - \mu_i)/(r + \mu_i)`
``omega``  :math:`p_i(1 - p_i)`               :math:`\phi_i^2/(\Phi_i(1 - \Phi_i))`              :math:`r\mu_i/(r + \mu_i)`
``h``      :math:`E[\varepsilon_i \mid y_i]`  :math:`\lambda_i`                                  :math:`\psi(r + y_i) - \log(r + \mu_i) - \psi(r) + \log r`
``tau``    :math:`\mathrm{Var}(h_i)`          ``omega``                                          summed over the NB2 mass function
``kappa``  :math:`\mathrm{Cov}(e_i, h_i)`     ``omega``                                          summed over the NB2 mass function
=========  =================================  =================================================  ==========================================================

``e`` is the score residual (the derivative of the unit's log-likelihood in
its linear predictor), ``omega`` its Fisher information, ``h`` the predicted
disturbance: the logistic disturbance for the logit, the normal one for the
probit, and the log of the gamma heterogeneity for the NB.  The probit's
generalized residual :math:`\lambda_i = E[\varepsilon_i \mid y_i]` is
:math:`\phi_i/\Phi_i` when :math:`y_i = 1` and :math:`-\phi_i/(1 - \Phi_i)`
when :math:`y_i = 0`, both its score residual and its predicted disturbance.
:func:`score_units` returns the per-draw pair ``(e, h)``;
:func:`information_units` the moments ``(omega, tau, kappa)``.

Pólya–Gamma working residuals
-----------------------------
The Pólya-Gamma augmentation of Polson, Scott & Windle (2013) makes the
logistic and negative-binomial likelihoods conditionally Gaussian in the
linear predictor:

.. math::
    p(y_i \mid \psi_i, \cdot) \propto \exp\!\left\{-\tfrac{\omega_i}{2}
        \bigl(\psi_i - \kappa_i/\omega_i\bigr)^2\right\} \cdot p(\omega_i),
    \qquad \omega_i \sim \mathrm{PG}(b_i, \psi_i).

Conditional on :math:`\omega`, this is a weighted Gaussian regression with
pseudo-outcome :math:`\tilde z_i = \kappa_i/\omega_i` and weights
:math:`\Omega = \mathrm{diag}(\omega_i)`, which the Moran diagnostic on working
residuals uses.  The probit has no Pólya–Gamma form; its working response is
the Fisher-scoring one, :math:`\tilde z_i = \eta_i + e_i/\omega_i` with the
Fisher weights.  :math:`\omega` is replaced by its conditional mean given the
draw,

.. math::
    E[\omega_i \mid \psi_i] = \frac{b_i}{2\psi_i}\,\tanh(\psi_i/2),
    \qquad \lim_{\psi_i\to 0} = b_i/4.

* Logit:  :math:`b_i = 1`,  :math:`\kappa_i = y_i - 1/2`.
* NegBin: :math:`b_i = y_i + \alpha`,  :math:`\kappa_i = (y_i - \alpha)/2`,
  with :math:`\alpha` the dispersion of the NB-2 logistic parameterization.

:math:`\Omega\tilde e` is then exactly the score residual ``e``; the PG weights
are not the Fisher information, so the score tests do not use them.
"""

from __future__ import annotations

import numpy as np
from scipy.special import digamma, expit, gammaln, log_ndtr

from .core import _get_posterior_draws

# ---------------------------------------------------------------------------
# Family and dispersion
# ---------------------------------------------------------------------------


def glm_family(model) -> str:
    """``"logit"``, ``"probit"`` or ``"negbin"``, from the model's likelihood and link."""
    likelihood = getattr(model, "_likelihood", "")
    if likelihood == "binary":
        return model._link
    if likelihood == "count":
        return "negbin"
    raise NotImplementedError(
        f"GLM LM tests are implemented for the logit, probit and NB families; "
        f"{type(model).__name__} has likelihood {likelihood!r}."
    )


def nb_dispersion_draws(model, idata, n_draws: int) -> np.ndarray:
    """The NB dispersion per draw: its posterior, or the value ``alpha_fixed`` holds."""
    if "alpha" in idata.posterior.data_vars:
        return _get_posterior_draws(idata, "alpha").reshape(-1)
    from ...models._base._nb import nb_alpha_fixed

    return np.full(n_draws, float(nb_alpha_fixed(model.priors)))


# ---------------------------------------------------------------------------
# Score quantities (latent SAR / SEM score tests)
# ---------------------------------------------------------------------------


def _logistic_upper_mean(a: np.ndarray) -> np.ndarray:
    """``E[eps | eps > a]`` for a standard logistic ``eps``."""
    a = np.asarray(a, dtype=np.float64)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        g = a + np.logaddexp(0.0, -a) / expit(-a)
    return np.where(a > 30.0, a + 1.0, g)


def _logit_gap(eta: np.ndarray) -> np.ndarray:
    """``E[eps | y = 1] - E[eps | y = 0]`` at linear predictor ``eta``."""
    return _logistic_upper_mean(-eta) + _logistic_upper_mean(eta)


def _probit_parts(eta: np.ndarray):
    """``log phi``, ``log Phi`` and ``log(1 - Phi)`` at ``eta``, in log space."""
    log_phi = -0.5 * eta * eta - 0.5 * np.log(2.0 * np.pi)
    return log_phi, log_ndtr(eta), log_ndtr(-eta)


def score_units(family: str, y: np.ndarray, eta: np.ndarray, r=None):
    """Score residual ``e`` and predicted disturbance ``h`` at ``eta``.

    ``eta`` may carry leading draw axes; ``r`` (NB) broadcasts against it.
    """
    if family == "logit":
        e = y - expit(eta)
        return e, e * _logit_gap(eta)
    if family == "probit":
        log_phi, log_P, log_Q = _probit_parts(eta)
        lam = np.where(y == 1.0, np.exp(log_phi - log_P), -np.exp(log_phi - log_Q))
        return lam, lam
    mu = np.exp(eta)
    e = r * (y - mu) / (r + mu)
    h = digamma(r + y) - np.log(r + mu) - digamma(r) + np.log(r)
    return e, h


def _nb_digamma_moments(mu, r, tail=1e-13, max_cells=4_000_000):
    """``Var(psi(r + Y))`` and ``Cov(Y, psi(r + Y))`` for ``Y ~ NB2(mu, r)``.

    Summed over each unit's central ``1 - 2 tail`` of mass, in blocks of
    units sorted by mean.
    """
    from scipy.stats import nbinom

    mu = np.asarray(mu, dtype=np.float64)
    n = mu.size
    p = r / (r + mu)
    lo = nbinom.ppf(tail, r, p).astype(int)
    hi = nbinom.ppf(1 - tail, r, p).astype(int) + 1
    order = np.argsort(mu)
    var_d = np.empty(n)
    cov_yd = np.empty(n)
    start = 0
    while start < n:
        stop = start + 1
        while stop < n:
            idx = order[start : stop + 1]
            if (hi[idx].max() - lo[idx].min() + 1) * idx.size > max_cells:
                break
            stop += 1
        idx = order[start:stop]
        ys = np.arange(lo[idx].min(), hi[idx].max() + 1, dtype=np.float64)[:, None]
        logpmf = (
            gammaln(ys + r)
            - gammaln(r)
            - gammaln(ys + 1)
            + r * np.log(p[idx])
            + ys * np.log1p(-p[idx])
        )
        pm = np.exp(logpmf)
        pm /= pm.sum(0)
        d = digamma(r + ys)
        Ed = (pm * d).sum(0)
        Ey = (pm * ys).sum(0)
        var_d[idx] = (pm * (d - Ed) ** 2).sum(0)
        cov_yd[idx] = (pm * (ys - Ey) * (d - Ed)).sum(0)
        start = stop
    return var_d, cov_yd


def information_units(family: str, eta: np.ndarray, r=None):
    """Fisher weight ``omega``, ``tau = Var(h)`` and ``kappa = Cov(e, h)`` at ``eta``."""
    if family == "logit":
        p = expit(eta)
        v = p * (1.0 - p)
        gap = _logit_gap(eta)
        return v, gap**2 * v, gap * v
    if family == "probit":
        log_phi, log_P, log_Q = _probit_parts(eta)
        omega = np.exp(2.0 * log_phi - log_P - log_Q)
        return omega, omega, omega
    mu = np.exp(eta)
    var_d, cov_yd = _nb_digamma_moments(mu, r)
    return r * mu / (r + mu), var_d, r / (r + mu) * cov_yd


# ---------------------------------------------------------------------------
# PG conditional-mean weight
# ---------------------------------------------------------------------------


def _pg_mean_weight(psi: np.ndarray, b: np.ndarray | float) -> np.ndarray:
    r"""Conditional mean :math:`E[\omega \mid \psi]` for :math:`\omega \sim \mathrm{PG}(b,\psi)`.

    .. math::
        E[\omega \mid \psi] = \frac{b}{2\psi}\,\tanh(\psi/2),
        \qquad \lim_{\psi \to 0} = b/4.

    Computed via ``0.5 * b * tanh(psi/2) / psi`` with a numerically safe
    branch at :math:`\psi \approx 0` using the Taylor expansion
    :math:`\tanh(x)/x \approx 1 - x^2/3 + 2x^4/15`.
    """
    psi = np.asarray(psi, dtype=np.float64)
    b_arr = np.asarray(b, dtype=np.float64)
    b_arr = np.broadcast_to(b_arr, psi.shape)
    half = 0.5 * psi
    small = np.abs(psi) < 1e-6
    # Default branch
    with np.errstate(divide="ignore", invalid="ignore"):
        w = b_arr * np.tanh(half) / (2.0 * psi)
    # Taylor: tanh(x)/x ≈ 1 - x²/3 + 2x⁴/15  →  b/(4) * (1 - psi²/12 + ...)
    if np.any(small):
        x2 = half[small] ** 2
        w_small = 0.25 * b_arr[small] * (1.0 - x2 / 3.0 + 2.0 * x2 * x2 / 15.0)
        w[small] = w_small
    return w


# ---------------------------------------------------------------------------
# Per-model working response builders
# ---------------------------------------------------------------------------


def _glm_eta_draws(model, beta_draws: np.ndarray) -> np.ndarray:
    """Linear predictor draws ``eta = X @ beta``.

    Shape: ``(draws, n)``.
    """
    X = model._X
    return beta_draws @ X.T


def _logit_working_response(
    model, beta_draws: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r"""Working response, working residual and PG weights for a logit model.

    Returns
    -------
    z_tilde : (draws, n) array
        Pseudo-outcome :math:`\kappa_i/\omega_i = (y_i - 1/2)/\omega_i`.
    e_tilde : (draws, n) array
        Working residual :math:`\tilde z - X\beta`.
    omega : (draws, n) array
        Conditional-mean PG weight :math:`E[\omega_i\mid\psi_i]` with
        :math:`b_i = 1`.
    """
    y = np.asarray(model._y, dtype=np.float64)
    psi = _glm_eta_draws(model, beta_draws)  # (draws, n)
    omega = _pg_mean_weight(psi, b=1.0)  # (draws, n)
    kappa = y - 0.5  # (n,)
    z_tilde = kappa[None, :] / omega
    e_tilde = z_tilde - psi
    return z_tilde, e_tilde, omega


def _probit_working_response(
    model, beta_draws: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r"""Fisher-scoring working response, residual and weights for a probit model.

    Returns
    -------
    z_tilde : (draws, n) array
        :math:`\eta_i + e_i/\omega_i`, :math:`e` the score residual.
    e_tilde : (draws, n) array
        Working residual :math:`e_i/\omega_i`.
    omega : (draws, n) array
        The Fisher weight :math:`\phi_i^2/(\Phi_i(1 - \Phi_i))`.
    """
    y = np.asarray(model._y, dtype=np.float64)
    eta = _glm_eta_draws(model, beta_draws)
    e, _ = score_units("probit", y, eta)
    omega, _, _ = information_units("probit", eta)
    e_tilde = e / omega
    return eta + e_tilde, e_tilde, omega


def _negbin_working_response(
    model, beta_draws: np.ndarray, alpha_draws: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r"""Working response, working residual and PG weights for an NB-2 model.

    Uses the logistic parameterization :math:`\psi_i = \log(\mu_i/\alpha)`
    so :math:`\psi_i = x_i'\beta - \log\alpha`.  PG shape is
    :math:`b_i = y_i + \alpha` and :math:`\kappa_i = (y_i - \alpha)/2`.

    Parameters
    ----------
    model : NegBin-like
        Must expose ``_X``, ``_y`` (raw counts).
    beta_draws : (draws, k) array
    alpha_draws : (draws,) array

    Returns
    -------
    z_tilde, e_tilde, omega : each (draws, n)
    """
    y = np.asarray(model._y, dtype=np.float64)
    eta = _glm_eta_draws(model, beta_draws)  # (draws, n) — log-mu
    alpha = alpha_draws[:, None]  # (draws, 1)
    psi = eta - np.log(alpha)  # logistic-parameterized linear predictor
    b = y[None, :] + alpha  # (draws, n)
    kappa = 0.5 * (y[None, :] - alpha)  # (draws, n)
    omega = _pg_mean_weight(psi, b=b)
    z_tilde = kappa / omega
    e_tilde = z_tilde - psi
    return z_tilde, e_tilde, omega


def glm_working_response(
    model, beta_draws: np.ndarray, idata
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The working response, residual and weights for the model's family."""
    family = glm_family(model)
    if family == "logit":
        return _logit_working_response(model, beta_draws)
    if family == "probit":
        return _probit_working_response(model, beta_draws)
    alpha_draws = nb_dispersion_draws(model, idata, beta_draws.shape[0])
    return _negbin_working_response(model, beta_draws, alpha_draws)


# ---------------------------------------------------------------------------
# Weighted M_X projection helpers
# ---------------------------------------------------------------------------


def _weighted_mx_quadratic(X: np.ndarray, v: np.ndarray, w: np.ndarray) -> float:
    r"""Compute :math:`v^\top \Omega M_X^\Omega v` where
    :math:`M_X^\Omega = I - X(X^\top \Omega X)^{-1} X^\top \Omega`.

    Equivalently :math:`v^\top \Omega v - (X^\top \Omega v)^\top
    (X^\top \Omega X)^{-1} (X^\top \Omega v)` — the residual sum of
    squares from a weighted least-squares regression of ``v`` on ``X``.
    """
    Wv = w * v
    XtWv = X.T @ Wv
    XtWX = X.T @ (w[:, None] * X)
    sol, *_ = np.linalg.lstsq(XtWX, XtWv, rcond=None)
    return float(v @ Wv) - float(XtWv @ sol)


def _weighted_mx_cross(
    X: np.ndarray, U: np.ndarray, V: np.ndarray, w: np.ndarray
) -> np.ndarray:
    r"""Generalises :func:`_weighted_mx_quadratic` to arbitrary left/right factors.

    Returns :math:`U^\top \Omega V - U^\top \Omega X (X^\top \Omega X)^{-1}
    X^\top \Omega V`.
    """
    Wv = w[:, None] * V if V.ndim == 2 else w * V
    Wu = w[:, None] * U if U.ndim == 2 else w * U
    XtWX = X.T @ (w[:, None] * X)
    XtWU = X.T @ Wu
    XtWV = X.T @ Wv
    UtWV = U.T @ Wv
    sol, *_ = np.linalg.lstsq(XtWX, XtWV, rcond=None)
    return UtWV - XtWU.T @ sol


def _weighted_T_ww(W_sp, w: np.ndarray, *, tr_W2: float | None = None) -> float:
    r"""Weighted analog of :math:`T_{WW} = \mathrm{tr}(W^2 + W^\top W)`
    for PG-augmented GLM LM-Error tests.

    Derivation: for the augmented working model :math:`\tilde z = X\beta + u`
    with :math:`u \sim \mathcal{N}(0, \Omega^{-1})`, the LM-Error score is
    :math:`S = u^\top \Omega W u` and its variance under H₀ is

    .. math::
        \mathrm{Var}(S) = \mathrm{tr}\!\bigl((\Omega W \Omega^{-1})^2\bigr)
            + \mathrm{tr}\!\bigl(\Omega W \Omega^{-1} W^\top\bigr)
            = \mathrm{tr}(W^2) + \sum_{i,j}\!\frac{\omega_i}{\omega_j}\, W_{ij}^2.

    The first term reduces to :math:`\mathrm{tr}(W^2) = \sum_{ij} W_{ij} W_{ji}`
    because the diagonal-similarity weights cancel; only the
    :math:`\mathrm{tr}(W^\top W)` analog acquires the weight ratio.

    Parameters
    ----------
    W_sp : scipy.sparse matrix
    w : (n,) array
        Per-observation working weights :math:`\omega_i`.
    tr_W2 : float, optional
        Pre-computed :math:`\mathrm{tr}(W^2) = \sum_{ij} W_{ij} W_{ji}`.
        If ``None`` it is computed from ``W_sp``.
    """
    import scipy.sparse as sp

    W = sp.csr_matrix(W_sp)
    if tr_W2 is None:
        tr_W2 = float(W.multiply(W.T).sum())
    # Sum_{ij} (w_i / w_j) W_ij^2  via COO iteration
    Wsq = W.multiply(W).tocoo()
    omega_ratio = w[Wsq.row] / w[Wsq.col]
    weighted_F2 = float((Wsq.data * omega_ratio).sum())
    return tr_W2 + weighted_F2
