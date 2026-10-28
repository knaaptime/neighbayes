r"""Zero-truncated NB2 pieces shared by the hurdle samplers.

The positive half of a hurdle model is NB2 truncated at zero,

.. math::

    p(y \mid y > 0) = \frac{\mathrm{NB}(y; \mu, \alpha)}{1 - p_0},
    \qquad p_0 = \mathrm{NB}(0; \mu, \alpha) = \Big(\frac{\alpha}{\mu + \alpha}\Big)^{\alpha}.

**Augmentation.**  :math:`1/(1 - p_0) = \sum_{m \ge 0} p_0^m`, so
:math:`p(y, m) \propto \mathrm{NB}(y)\,p_0^m`: ``m`` extra zero draws (the
zeros the truncation hid) restore the untruncated likelihood, and given ``m``
the cell is Pólya–Gamma conjugate, ``NB(0)`` being :math:`(1 + e^{\psi})^{-\alpha}`
with :math:`\psi = \eta - \log\alpha`:

.. math::

    m \mid y, \eta, \alpha &\sim \mathrm{Geometric}(1 - p_0)
    \text{ on } \{0, 1, \dots\}, \\
    \omega \mid m &\sim \mathrm{PG}(y + \alpha(1 + m), \psi), \qquad
    z = \frac{y - \alpha(1 + m)}{2\omega} + \log\alpha .

α is drawn from the truncated likelihood with ``m`` integrated out, so ``m``
is redrawn before its next use (the partially collapsed ordering).
"""

from __future__ import annotations

import functools

import numpy as np
from scipy.special import gammaln

from .._utils._polyagamma import sample_polyagamma
from .._utils._slice import slice_sample_1d

#: Cap on the missed-zero count; reached only when ``p₀ → 1`` (``μ → 0``).
_M_MAX = 1e8


def log_nb_zero(eta: np.ndarray, alpha: float) -> np.ndarray:
    """``log NB(0; e^η, α) = α (log α − log(e^η + α))``."""
    la = np.log(alpha)
    return alpha * (la - np.logaddexp(eta, la))


def draw_missed_zeros(eta: np.ndarray, alpha: float, rng) -> np.ndarray:
    """``m ~ Geometric(1 − p₀)`` on ``{0, 1, …}``: ``⌊log U / log p₀⌋``."""
    log_p0 = np.minimum(log_nb_zero(eta, alpha), -1e-300)
    m = np.floor(np.log(rng.random(np.shape(eta))) / log_p0)
    return np.minimum(m, _M_MAX)


def truncated_working_data(
    y: np.ndarray, eta: np.ndarray, alpha: float, rng
) -> tuple[np.ndarray, np.ndarray]:
    """PG weights and working response of the positive cells: ``(ω, z)``."""
    m = draw_missed_zeros(eta, alpha, rng)
    h = y + alpha * (1.0 + m)
    la = np.log(alpha)
    omega = sample_polyagamma(h, np.clip(eta - la, -20.0, 20.0), rng=rng)
    return omega, 0.5 * (y - alpha * (1.0 + m)) / omega + la


def _log1mexp(x: np.ndarray) -> np.ndarray:
    """``log(1 − eˣ)`` for ``x < 0``, accurate at both ends."""
    return np.where(x > -0.6931471805599453, np.log(-np.expm1(x)), np.log1p(-np.exp(x)))


def truncated_nb_eta_loglik(y: np.ndarray, eta: np.ndarray, alpha: float) -> float:
    """``Σ log tNB(y; e^η, α)`` up to terms free of η (for slices in η)."""
    la = np.log(alpha)
    log_mu_a = np.logaddexp(eta, la)
    log_p0 = np.minimum(alpha * (la - log_mu_a), -1e-300)
    return float(np.sum(y * eta - (y + alpha) * log_mu_a - _log1mexp(log_p0)))


def bernoulli_logit_loglik(d: np.ndarray, eta: np.ndarray) -> float:
    """``Σ [d η − log(1 + e^η)]``."""
    return float(np.sum(d * eta - np.logaddexp(0.0, eta)))


def truncated_nb_loglik_pointwise(y: np.ndarray, eta: np.ndarray, alpha) -> np.ndarray:
    """``log NB(y; e^η, α) − log(1 − p₀)`` for positive ``y`` (full pmf)."""
    alpha = np.asarray(alpha, dtype=np.float64)
    la = np.log(alpha)
    log_mu_a = np.logaddexp(eta, la)
    log_p0 = alpha * (la - log_mu_a)
    ll = (
        gammaln(y + alpha)
        - gammaln(alpha)
        - gammaln(y + 1.0)
        + y * (eta - log_mu_a)
        + log_p0
    )
    return ll - _log1mexp(np.minimum(log_p0, -1e-300))


def hurdle_loglik_pointwise(
    y: np.ndarray, eta_bin: np.ndarray, eta_cnt: np.ndarray, alpha
) -> np.ndarray:
    """Pointwise hurdle log-pmf: ``log(1 − π)`` at zero, else ``log π + log tNB``.

    Broadcasts like :func:`~neighbayes.samplers.zinb._core._zinb_loglik_pointwise`
    (a leading draw axis with ``alpha`` shaped ``(n_draws, 1)``).
    """
    y = np.asarray(y, dtype=np.float64)
    pos = y > 0
    log_pi = -np.logaddexp(0.0, -eta_bin)
    log_1m_pi = -np.logaddexp(0.0, eta_bin)
    tnb = truncated_nb_loglik_pointwise(np.where(pos, y, 1.0), eta_cnt, alpha)
    return np.where(pos, log_pi + tnb, log_1m_pi)


def sample_alpha_truncated(
    alpha: float,
    y: np.ndarray,
    eta: np.ndarray,
    *,
    alpha_sigma: float,
    alpha_nu: float,
    alpha_fixed=None,
    rng,
) -> float:
    """α from the zero-truncated likelihood of the positive cells.

    Slice on ``log α`` with the half-Student-t(ν, σ) prior and the ``log α``
    Jacobian, as :func:`~neighbayes.samplers.negbin._core._sample_alpha`.
    """
    if alpha_fixed is not None:
        return float(alpha_fixed)
    y = np.asarray(y, dtype=np.float64)
    if y.size == 0:
        return float(alpha)
    y_vals, y_counts = np.unique(y, return_counts=True)
    n_obs = y.size
    y_dot_eta = float(y @ eta)

    def log_density(log_a: float) -> float:
        a = np.exp(log_a)
        log_mu_a = np.logaddexp(eta, log_a)
        log_p0 = a * (log_a - log_mu_a)
        ll = (
            float(y_counts @ gammaln(y_vals + a))
            - n_obs * gammaln(a)
            + y_dot_eta
            - float(y @ log_mu_a)
            + float(log_p0.sum())
            - float(_log1mexp(np.minimum(log_p0, -1e-300)).sum())
        )
        prior = -0.5 * (alpha_nu + 1.0) * np.log1p(a * a / (alpha_nu * alpha_sigma**2))
        val = log_a + ll + prior
        return val if np.isfinite(val) else -np.inf

    new, _ = slice_sample_1d(
        log_density, np.log(alpha), lower=-10.0, upper=10.0, w=0.5, rng=rng
    )
    return float(np.exp(new))


def _alpha_log_prior(a: float, alpha_sigma: float, alpha_nu: float) -> float:
    return -0.5 * (alpha_nu + 1.0) * np.log1p(a * a / (alpha_nu * alpha_sigma**2))


def factor_slice(
    logf, x, dirs, widths, rng, max_steps: int = 10, max_shrink: int = 200
):
    """One factor-slice cycle: a univariate slice along each column of ``dirs``.

    Tibbits et al. (2014): slicing along the eigenvectors of a warmup estimate
    of the posterior covariance follows a correlated ridge that coordinate
    updates crawl along.  Stepping out is capped at ``max_steps`` widths per
    side (still a valid slice sampler).
    """
    x = np.asarray(x, dtype=np.float64).copy()
    fx = logf(x)
    for j in rng.permutation(dirs.shape[1]):
        d, w = dirs[:, j], widths[j]
        logy = fx + np.log(rng.uniform())
        u = rng.uniform()
        lo, hi = -u * w, (1.0 - u) * w
        for _ in range(max_steps):
            if logf(x + lo * d) <= logy:
                break
            lo -= w
        for _ in range(max_steps):
            if logf(x + hi * d) <= logy:
                break
            hi += w
        for _ in range(max_shrink):
            s = rng.uniform(lo, hi)
            fn = logf(x + s * d)
            if fn > logy:
                x, fx = x + s * d, fn
                break
            if s < 0:
                lo = s
            else:
                hi = s
    return x


class LevelAlphaMove:
    r"""Joint slice on the count level and ``log α`` along their ridge.

    When the positive counts are mostly ones only ``E[y | y > 0] = μ/(1 − p₀)``
    is pinned, and a smaller α with a smaller μ fits as well: the level and
    ``log α`` trade off along a ridge that the alternating PG and α updates
    barely move along.  Each sweep this slices ``(t, log α)`` — the count
    predictor moved to ``η + t·v`` along the level direction ``v`` — under the
    exact truncated likelihood, along the eigenvectors of the warmup
    covariance of ``(level, log α)``.

    The host supplies, per sweep, ``η`` and ``v`` on the positive cells and the
    log prior of the level carrier as a function of ``t``, and applies the
    returned ``t``.
    """

    #: Warmup sweeps (as fractions of ``tune``) at which the directions refresh.
    _REFRESH = (0.2, 0.4, 0.6, 0.8)

    def __init__(self, tune: int, alpha_sigma: float, alpha_nu: float):
        self.tune = int(tune)
        self.alpha_sigma, self.alpha_nu = float(alpha_sigma), float(alpha_nu)
        self.refresh = {int(f * tune) for f in self._REFRESH}
        self.dirs = np.eye(2)
        self.widths = np.array([0.5, 0.5])
        self.history: list[tuple[float, float]] = []

    def step(self, it, y, eta, v, carrier_log_prior, alpha, rng):
        """Return ``(t, α)`` for this sweep; ``it`` is the sweep index."""
        sig, nu = self.alpha_sigma, self.alpha_nu

        def logf(th):
            t, la = th
            if not -10.0 < la < 10.0:
                return -np.inf
            a = np.exp(la)
            ll = float(truncated_nb_loglik_pointwise(y, eta + t * v, a).sum())
            val = ll + carrier_log_prior(t) + la + _alpha_log_prior(a, sig, nu)
            return val if np.isfinite(val) else -np.inf

        if it in self.refresh and len(self.history) > 20:
            H = np.asarray(self.history[len(self.history) // 2 :])
            cov = np.cov(H.T) + 1e-10 * np.eye(2)
            ev, self.dirs = np.linalg.eigh(cov)
            self.widths = 2.0 * np.sqrt(np.maximum(ev, 1e-12))
        t, la = factor_slice(logf, [0.0, np.log(alpha)], self.dirs, self.widths, rng)
        if it < self.tune:
            vbar = float(np.mean(v)) if v.size else 1.0
            level = float(np.mean(eta)) / vbar if vbar else 0.0
            self.history.append((level + t, la))
        return float(t), float(np.exp(la))


# ---------------------------------------------------------------------------
# JAX
# ---------------------------------------------------------------------------


@functools.lru_cache(maxsize=None)
def make_truncated_jax():
    """JAX twins: ``(working_data, alpha_update, loglik_pointwise)``.

    ``working_data(y, eta, pos, alpha, key, draw_pg) -> (ω, z)`` with ``ω = 0``
    and ``z = 0`` off the positive cells; ``alpha_update(alpha, y, eta, pos,
    key, a_sigma, a_nu)`` slices ``log α`` on the truncated likelihood of the
    positive cells under a half-t(``a_nu``, ``a_sigma``) prior.  The prior's
    values are arguments, so one set of functions serves every model.
    """
    import jax
    import jax.numpy as jnp
    from jax.scipy.special import gammaln as jgammaln

    from .._utils._jax_slice import jax_slice_sample_1d

    def log1mexp(x):
        return jnp.where(
            x > -0.6931471805599453, jnp.log(-jnp.expm1(x)), jnp.log1p(-jnp.exp(x))
        )

    def working_data(y, eta, pos, alpha, key, draw_pg):
        k_m, k_pg = jax.random.split(key)
        la = jnp.log(alpha)
        log_p0 = jnp.minimum(alpha * (la - jnp.logaddexp(eta, la)), -1e-300)
        u = jax.random.uniform(k_m, eta.shape, minval=1e-300, maxval=1.0)
        m = jnp.minimum(jnp.floor(jnp.log(u) / log_p0), _M_MAX)
        h = jnp.where(pos, y + alpha * (1.0 + m), 1.0)
        om = draw_pg(h, jnp.clip(eta - la, -20.0, 20.0), k_pg)
        om = jnp.where(pos, om, 0.0)
        safe = jnp.where(pos, om, 1.0)
        z = jnp.where(pos, 0.5 * (y - alpha * (1.0 + m)) / safe + la, 0.0)
        return om, z

    def log_density(log_a, y, eta, pos, a_sigma, a_nu):
        a = jnp.exp(log_a)
        log_mu_a = jnp.logaddexp(eta, log_a)
        log_p0 = jnp.minimum(a * (log_a - log_mu_a), -1e-300)
        ll = (
            jgammaln(y + a)
            - jgammaln(a)
            + y * (eta - log_mu_a)
            + log_p0
            - log1mexp(log_p0)
        )
        prior = -0.5 * (a_nu + 1.0) * jnp.log1p(a * a / (a_nu * a_sigma**2))
        return log_a + jnp.sum(jnp.where(pos, ll, 0.0)) + prior

    def alpha_update(alpha, y, eta, pos, key, a_sigma, a_nu):
        new, _ = jax_slice_sample_1d(
            lambda la: log_density(la, y, eta, pos, a_sigma, a_nu),
            jnp.log(alpha),
            -10.0,
            10.0,
            key=key,
            w=0.5,
        )
        return jnp.exp(new)

    def loglik_pointwise(y, eta_bin, eta_cnt, alpha):
        pos = y > 0
        yy = jnp.where(pos, y, 1.0)
        la = jnp.log(alpha)
        log_mu_a = jnp.logaddexp(eta_cnt, la)
        log_p0 = jnp.minimum(alpha * (la - log_mu_a), -1e-300)
        tnb = (
            jgammaln(yy + alpha)
            - jgammaln(alpha)
            - jgammaln(yy + 1.0)
            + yy * (eta_cnt - log_mu_a)
            + log_p0
            - log1mexp(log_p0)
        )
        return jnp.where(
            pos, -jnp.logaddexp(0.0, -eta_bin) + tnb, -jnp.logaddexp(0.0, eta_bin)
        )

    return working_data, alpha_update, loglik_pointwise
