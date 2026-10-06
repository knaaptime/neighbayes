r"""Structured separable hurdle NB flow sweep (cross-section and panel).

Two separable flow equations on the ``n × n`` flow array, each a
:class:`~neighbayes.samplers.negbin_reduced._flow_structured_fe.StructuredEquation`
(``n × n`` solves only, nothing ``n² × n²``), each with its own pair and
period effects:

* **binary**, every cell — ``η_b,t = (L_o^b ⊗ L_d^b)⁻¹ Z_t γ + τ_b,t + C_b``
  with PG(1) weights and working response ``(d − ½)/ω``;
* **count**, positive cells — ``η_c,t = (L_o ⊗ L_d)⁻¹ X_t β + τ_t + C`` with
  the truncation augmented (:mod:`._truncated`) and ``Ω = 0`` elsewhere,
  which removes the zero cells from every moment exactly;

then the count level and ``log α`` jointly
(:class:`~._truncated.LevelAlphaMove`).
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from ..negbin_reduced._flow_structured_fe import StructuredEquation
from ._truncated import (
    LevelAlphaMove,
    bernoulli_logit_loglik,
    hurdle_loglik_pointwise,
    sample_alpha_truncated,
    truncated_nb_eta_loglik,
    truncated_working_data,
)


def _const_col(M: np.ndarray):
    """``(index, value)`` of the first constant non-zero column, or ``(None, 1.0)``."""
    const = np.flatnonzero(np.all(M == M[:1], axis=0) & (M[0] != 0))
    return (int(const[0]), float(M[0, const[0]])) if const.size else (None, 1.0)


def _start_logits(d, n, T, Z, pair):
    """Starting γ (least squares on shrunk logits) and binary pair effects."""
    target = np.where(d > 0, np.log(3.0), -np.log(3.0))
    g0 = np.linalg.lstsq(Z, target, rcond=None)[0] if Z.shape[1] else np.zeros(0)
    C0 = None
    if pair:
        R = (target - Z @ g0).reshape(T, n, n)
        C0 = R.mean(axis=0)
    return g0, C0


def run_chain_hurdle_flow_structured(
    y: np.ndarray,
    X: np.ndarray,
    Z: np.ndarray,
    W_csc,
    W_sel_csc,
    n: int,
    T: int,
    priors,
    draws: int,
    tune: int,
    *,
    pair_effects: Optional[tuple[float, float]] = None,
    period_effects: Optional[tuple[float, float, int]] = None,
    sel_pair_effects: Optional[tuple[float, float]] = None,
    sel_period_effects: Optional[tuple[float, float, int]] = None,
    rho_bounds: tuple[float, float] = (-0.999, 0.999),
    lam_bounds: tuple[float, float] = (-0.999, 0.999),
    thin: int = 1,
    rng: Optional[np.random.Generator] = None,
    chain_id: int = 0,
    progress_manager: object | None = None,
    store_log_lik: bool = False,
    store_group_draws: bool = True,
) -> dict[str, np.ndarray]:
    """One chain of the structured separable hurdle flow sampler.

    ``priors`` carries ``beta_mu``, ``beta_sigma``, ``gamma_mu``,
    ``gamma_sigma``, ``alpha_sigma``, ``alpha_nu`` and ``alpha_fixed``.

    Returns
    -------
    dict
        ``lam_d, lam_o, lam_w, gamma, sel_time_effect`` and the binary pair
        effects (``sel_group_effect``), ``rho_d, rho_o, rho_w, beta,
        time_effect, alpha`` and the count pair effects (``group_effect``),
        and ``log_lik``.
    """
    from .._utils._polyagamma import sample_polyagamma
    from ..count_panel._core import _GroupStore

    rng = np.random.default_rng() if rng is None else rng
    y = np.asarray(y, dtype=np.float64)
    X = np.asarray(X, dtype=np.float64)
    Z = np.asarray(Z, dtype=np.float64)
    shape = (T, n, n)
    pos = y > 0
    if not pos.any():
        raise ValueError("A hurdle model needs at least one positive count.")
    d = pos.astype(np.float64)
    y_pos = y[pos]
    alpha_fixed = getattr(priors, "alpha_fixed", None)
    alpha = 1.0 if alpha_fixed is None else float(alpha_fixed)

    def jitter():
        return {"rho_d": rng.uniform(-0.1, 0.1), "rho_o": rng.uniform(-0.1, 0.1)}

    g0, Cb0 = _start_logits(d, n, T, Z, sel_pair_effects is not None)
    rows = pos if pos.sum() > X.shape[1] else np.ones(y.size, bool)
    b0 = np.linalg.lstsq(X[rows], np.log(y[rows] + 0.5), rcond=None)[0]
    Cc0 = None
    if pair_effects is not None:
        Y = y.reshape(shape)
        P = pos.reshape(shape)
        cnt = P.sum(axis=0)
        mean_pos = np.where(cnt > 0, (Y * P).sum(axis=0) / np.maximum(cnt, 1), 1.0)
        Cc0 = np.log(mean_pos) - np.log(y_pos.mean())

    binary = StructuredEquation(
        Z, W_sel_csc, n, T, priors.gamma_mu, priors.gamma_sigma,
        init_beta=g0 + rng.normal(0.0, 0.1, size=g0.size), init_rho=jitter(),
        pair_effects=sel_pair_effects, period_effects=sel_period_effects,
        rho_lower=lam_bounds[0], rho_upper=lam_bounds[1], C_init=Cb0,
    )  # fmt: skip
    count = StructuredEquation(
        X, W_csc, n, T, priors.beta_mu, priors.beta_sigma,
        init_beta=b0 + rng.normal(0.0, 0.1, size=b0.size), init_rho=jitter(),
        pair_effects=pair_effects, period_effects=period_effects,
        rho_lower=rho_bounds[0], rho_upper=rho_bounds[1], C_init=Cc0,
    )  # fmt: skip
    level_col, level_val = _const_col(X)
    mover = (
        LevelAlphaMove(tune, priors.alpha_sigma, priors.alpha_nu)
        if alpha_fixed is None
        else None
    )

    n_keep = draws // thin if thin > 0 else draws
    out = {
        name: np.empty(n_keep)
        for name in ("lam_d", "lam_o", "lam_w", "rho_d", "rho_o", "rho_w", "alpha")
    }
    out["gamma"] = np.empty((n_keep, binary.k))
    out["sel_time_effect"] = np.empty((n_keep, binary.n_tau))
    out["beta"] = np.empty((n_keep, count.k))
    out["time_effect"] = np.empty((n_keep, count.n_tau))
    out["log_lik"] = np.empty((n_keep, y.size)) if store_log_lik else None
    for key, eq in (("sel_group_sd", binary), ("group_sd", count)):
        if eq.pair_sd_scale is not None:
            out[key] = np.empty(n_keep)
    store_b = (
        _GroupStore(n_keep, n * n, store_group_draws) if binary.has_pairs else None
    )
    store_c = _GroupStore(n_keep, n * n, store_group_draws) if count.has_pairs else None

    for it in range(tune + draws):
        tuning = it < tune
        # binary: PG-logit on every cell
        om_b = sample_polyagamma(np.ones(y.size), binary.eta.ravel(), rng=rng)
        binary.update(
            om_b.reshape(shape), ((d - 0.5) / om_b).reshape(shape), rng, tuning
        )
        d3 = d.reshape(shape)
        binary.exact_sd_move(lambda e: bernoulli_logit_loglik(d3, e), rng)

        # count: positive cells, truncation augmented; Ω = 0 elsewhere
        om_c = np.zeros(y.size)
        z_c = np.zeros(y.size)
        om_c[pos], z_c[pos] = truncated_working_data(
            y_pos, count.eta.ravel()[pos], alpha, rng
        )
        count.update(om_c.reshape(shape), z_c.reshape(shape), rng, tuning)
        count.exact_sd_move(
            lambda e: truncated_nb_eta_loglik(y_pos, e.ravel()[pos], alpha), rng
        )

        # level and dispersion jointly
        if mover is not None:
            level = count.level_direction(level_col, level_val)
            eta_pos = count.eta.ravel()[pos]
            if level is None:
                alpha = sample_alpha_truncated(
                    alpha, y_pos, eta_pos, alpha_sigma=priors.alpha_sigma,
                    alpha_nu=priors.alpha_nu, rng=rng,
                )  # fmt: skip
            else:
                v, log_prior, apply = level
                v_pos = np.broadcast_to(v, shape).ravel()[pos]
                t, alpha = mover.step(it, y_pos, eta_pos, v_pos, log_prior, alpha, rng)
                apply(t)

        if not tuning and (it - tune) % thin == 0:
            j = (it - tune) // thin
            if j < n_keep:
                for eq, a, b, w in (
                    (binary, "lam_d", "lam_o", "lam_w"),
                    (count, "rho_d", "rho_o", "rho_w"),
                ):
                    out[a][j] = eq.rho["rho_d"]
                    out[b][j] = eq.rho["rho_o"]
                    out[w][j] = -eq.rho["rho_d"] * eq.rho["rho_o"]
                out["gamma"][j] = binary.beta
                out["sel_time_effect"][j] = binary.tau
                out["beta"][j] = count.beta
                out["time_effect"][j] = count.tau
                out["alpha"][j] = alpha
                for key, eq in (("sel_group_sd", binary), ("group_sd", count)):
                    if eq.pair_sd_scale is not None:
                        out[key][j] = eq.s_pair
                if store_b is not None:
                    store_b.add(j, binary.C.ravel().copy())
                if store_c is not None:
                    store_c.add(j, count.C.ravel().copy())
                if store_log_lik:
                    out["log_lik"][j] = hurdle_loglik_pointwise(
                        y, binary.eta.ravel(), count.eta.ravel(), alpha
                    )
        if progress_manager is not None:
            progress_manager.update(chain_id, it, tuning=tuning)

    if store_b is not None:
        out.update({"sel_" + key: val for key, val in store_b.result().items()})
    if store_c is not None:
        out.update(store_c.result())
    return out
