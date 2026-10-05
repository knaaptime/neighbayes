"""Count log-likelihoods as functions of the log-mean, up to terms free of it.

Used by the exact-likelihood non-centred moves of the learned effect sds
(:func:`~._group_effects.exact_noncentered_sd`), which slice a scalar while
the predictor moves; terms constant in the predictor are dropped.
"""

from __future__ import annotations

import numpy as np


def nb_eta_loglik(y: np.ndarray, eta: np.ndarray, alpha: float) -> float:
    """``Σ log NB2(y; e^η, α)`` up to terms free of η."""
    return float(np.sum(y * eta - (y + alpha) * np.logaddexp(eta, np.log(alpha))))


def zinb_count_eta_loglik(
    y: np.ndarray, eta_sel: np.ndarray, eta_cnt: np.ndarray, alpha: float
) -> float:
    """``Σ log ZINB(y)`` as a function of the count predictor (selection held).

    The structural zeros are summed out: ``log π + log NB(y)`` for ``y > 0``,
    ``log(1 − π + π NB(0))`` at zero.
    """
    la = np.log(alpha)
    log_mu_a = np.logaddexp(eta_cnt, la)
    log_pi = -np.logaddexp(0.0, -eta_sel)
    log_1m_pi = -np.logaddexp(0.0, eta_sel)
    pos = y > 0
    ll_pos = y * (eta_cnt - log_mu_a) + alpha * (la - log_mu_a)
    ll_zero = np.logaddexp(log_1m_pi, log_pi + alpha * (la - log_mu_a))
    return float(np.sum(np.where(pos, ll_pos, ll_zero)))
