r"""Structured separable ZINB flow sweep (cross-section and panel).

Two separable flow equations on the ``n × n`` flow array, each a
:class:`~neighbayes.samplers.negbin_reduced._flow_structured_fe.StructuredEquation`
(``n × n`` solves only, nothing ``n² × n²``):

* **selection** — ``η_sel,t = (L_o^s ⊗ L_d^s)⁻¹ Z_t γ`` with PG(1) weights
  and working response ``(z − ½)/ω``;
* **count** — ``η_cnt,t = (L_o ⊗ L_d)⁻¹ X_t β + τ_t + C`` with PG(y + α, ·)
  weights on the active cells and ``Ω = 0`` on the structural zeros, which
  removes them from every moment exactly;

linked by the per-cell allocation ``z`` (``_sample_z``).  Pair and period
effects enter the count equation only.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from .._utils._count_loglik import zinb_count_eta_loglik
from ..negbin._core import GibbsState as _AlphaState
from ..negbin._core import _sample_alpha
from ..negbin_reduced._core import ReducedGibbsPriors, _sample_omega
from ..negbin_reduced._flow_structured_fe import StructuredEquation
from ._core import _sample_z, _zinb_loglik_pointwise


def run_chain_zinb_flow_structured(
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
    rho_bounds: tuple[float, float] = (-0.999, 0.999),
    lam_bounds: tuple[float, float] = (-0.999, 0.999),
    thin: int = 1,
    rng: Optional[np.random.Generator] = None,
    chain_id: int = 0,
    progress_manager: object | None = None,
    store_log_lik: bool = False,
    store_group_draws: bool = True,
) -> dict[str, np.ndarray]:
    """One chain of the structured separable ZINB flow sampler.

    ``priors`` carries ``beta_mu``, ``beta_sigma``, ``gamma_mu``,
    ``gamma_sigma``, ``alpha_sigma``, ``alpha_nu`` and ``alpha_fixed``.

    Returns
    -------
    dict
        ``lam_d, lam_o, lam_w, gamma, rho_d, rho_o, rho_w, beta, time_effect,
        alpha``, the pair effects and ``log_lik``.
    """
    from .._utils._polyagamma import sample_polyagamma
    from ..count_panel._core import _GroupStore

    rng = np.random.default_rng() if rng is None else rng
    y = np.asarray(y, dtype=np.float64)
    shape = (T, n, n)
    positive = y > 0
    alpha_fixed = getattr(priors, "alpha_fixed", None)
    alpha = 1.0 if alpha_fixed is None else float(alpha_fixed)
    alpha_priors = ReducedGibbsPriors(
        alpha_sigma=priors.alpha_sigma,
        alpha_nu=priors.alpha_nu,
        alpha_fixed=alpha_fixed,
    )

    # Starting values: selection at the active share, count on the positives.
    share = float(np.clip(positive.mean() * 1.5, 0.05, 0.95))
    p = Z.shape[1]
    g0 = np.zeros(p)
    const = np.flatnonzero(np.all(Z == Z[:1], axis=0) & (Z[0] != 0))
    if const.size:
        g0[const[0]] = np.log(share / (1.0 - share)) / Z[0, const[0]]
    rows = positive if positive.sum() > X.shape[1] else np.ones(y.size, bool)
    b0 = np.linalg.lstsq(X[rows], np.log(y[rows] + 0.5), rcond=None)[0]
    jitter = lambda: {  # noqa: E731
        "rho_d": rng.uniform(-0.1, 0.1),
        "rho_o": rng.uniform(-0.1, 0.1),
    }
    C_init = None
    if pair_effects is not None:
        Y = y.reshape(shape)
        C_init = np.log(Y.mean(axis=0) + 0.5) - np.log(Y.mean() + 0.5)

    sel = StructuredEquation(
        Z,
        W_sel_csc,
        n,
        T,
        priors.gamma_mu,
        priors.gamma_sigma,
        init_beta=g0 + rng.normal(0.0, 0.1, size=p),
        init_rho=jitter(),
        rho_lower=lam_bounds[0],
        rho_upper=lam_bounds[1],
    )
    cnt = StructuredEquation(
        X,
        W_csc,
        n,
        T,
        priors.beta_mu,
        priors.beta_sigma,
        init_beta=b0 + rng.normal(0.0, 0.1, size=b0.size),
        init_rho=jitter(),
        pair_effects=pair_effects,
        period_effects=period_effects,
        rho_lower=rho_bounds[0],
        rho_upper=rho_bounds[1],
        C_init=C_init,
    )
    z = np.where(positive, 1, rng.binomial(1, 0.5, size=y.size)).astype(np.int8)

    n_keep = draws // thin if thin > 0 else draws
    out = {
        name: np.empty(n_keep)
        for name in ("lam_d", "lam_o", "lam_w", "rho_d", "rho_o", "rho_w", "alpha")
    }
    out["gamma"] = np.empty((n_keep, sel.k))
    out["beta"] = np.empty((n_keep, cnt.k))
    out["time_effect"] = np.empty((n_keep, cnt.n_tau))
    out["log_lik"] = np.empty((n_keep, y.size)) if store_log_lik else None
    if cnt.pair_sd_scale is not None:
        out["group_sd"] = np.empty(n_keep)
    gstore = _GroupStore(n_keep, n * n, store_group_draws) if cnt.has_pairs else None

    for it in range(tune + draws):
        tuning = it < tune
        # selection: PG-logit on the latent allocation
        om_s = sample_polyagamma(np.ones(y.size), sel.eta.ravel(), rng=rng)
        sel.update(om_s.reshape(shape), ((z - 0.5) / om_s).reshape(shape), rng, tuning)

        # zero allocation
        z = _sample_z(y, cnt.eta.ravel(), sel.eta.ravel(), alpha, rng=rng)
        act = z == 1

        # count: active cells only (Ω = 0 on the structural zeros)
        if np.any(act):
            eta_c = cnt.eta.ravel()
            om_c = np.zeros(y.size)
            zc = np.zeros(y.size)
            om_c[act] = _sample_omega(
                y[act], alpha, eta_c[act] - np.log(alpha), rng=rng
            )
            zc[act] = 0.5 * (y[act] - alpha) / om_c[act] + np.log(alpha)
            cnt.update(om_c.reshape(shape), zc.reshape(shape), rng, tuning)
            y3 = y.reshape(shape)
            cnt.exact_sd_move(
                lambda e: zinb_count_eta_loglik(y3, sel.eta, e, alpha), rng
            )
            eta_c = cnt.eta.ravel()
            st = _AlphaState(
                eta=eta_c[act],
                beta=cnt.beta,
                sigma2=1.0,
                rho=0.0,
                alpha=alpha,
                omega=om_c[act],
            )
            alpha = _sample_alpha(st, y[act], alpha_priors, rng=rng)

        if not tuning and (it - tune) % thin == 0:
            j = (it - tune) // thin
            if j < n_keep:
                for eq, a, b, w in (
                    (sel, "lam_d", "lam_o", "lam_w"),
                    (cnt, "rho_d", "rho_o", "rho_w"),
                ):
                    out[a][j] = eq.rho["rho_d"]
                    out[b][j] = eq.rho["rho_o"]
                    out[w][j] = -eq.rho["rho_d"] * eq.rho["rho_o"]
                out["gamma"][j] = sel.beta
                out["beta"][j] = cnt.beta
                out["time_effect"][j] = cnt.tau
                out["alpha"][j] = alpha
                if cnt.pair_sd_scale is not None:
                    out["group_sd"][j] = cnt.s_pair
                if gstore is not None:
                    gstore.add(j, cnt.C.ravel().copy())
                if store_log_lik:
                    out["log_lik"][j] = _zinb_loglik_pointwise(
                        y, sel.eta.ravel(), cnt.eta.ravel(), alpha
                    )
        if progress_manager is not None:
            progress_manager.update(chain_id, it, tuning=tuning)

    if gstore is not None:
        out.update(gstore.result())
    return out
