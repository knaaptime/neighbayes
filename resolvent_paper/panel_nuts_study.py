r"""Experiment C — panel Gaussian: NUTS (+ fast differentiable logdet) vs Gibbs.

The cross-section shootout (`nuts_study.py`) found Gibbs beats native NUTS on
CPU because the collapsed slice sampler mixes near-perfectly on a 1-D ρ.  Panel
models are the opposite regime: unit random effects (RE) or many fixed effects
(FE) give a high-dimensional posterior where gradient-based NUTS is often the
better mixer — *if* the logdet in the leapfrog is cheap.  That is exactly what
the resolvent / chol-cheb surrogate provides (the pytensor logdet is
differentiable, so PyMC NUTS already rides it).

This benchmarks SAR **panel** FE and RE both ways — NUTS with the chol-cheb
logdet vs the package Gibbs sampler — and reports ESS/sec for ρ and β.

Run: ``conda run -n bayespreg python resolvent_paper/panel_nuts_study.py [--quick]``
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import arviz as az
import numpy as np

from neighbayes.dgp.panel_fe import simulate_panel_sar_fe
from neighbayes.models.panel import SARPanelFE, SARPanelRE

RESULTS = Path(__file__).resolve().parent / "results"
RESULTS.mkdir(exist_ok=True)

PRIORS = {"beta_mu": 0.0, "beta_sigma": 10.0, "sigma2_beta": 1.0}
TRUE_RHO = 0.5
TRUE_BETA = np.array([1.0, 2.0])


def _min_ess(post, var):
    da = np.asarray(post[var])
    if da.ndim == 3:  # (chain, draw, k)
        return min(float(az.ess(da[..., j])) for j in range(da.shape[-1]))
    return float(az.ess(da))


def _run(model_cls, y, X, W, N, T, sampler, draws, tune, chains, **fit_kw):
    # Warmup: trigger JAX JIT / numba JIT / pytensor compilation with a tiny
    # run so first-call overhead doesn't inflate the timed run.
    gibbs_backend = fit_kw.pop("gibbs_backend", None)
    nuts_sampler = fit_kw.get("nuts_sampler", "pymc")
    m_warm = model_cls(y=y, X=X, W=W, N=N, T=T, logdet_method="cheb_cholesky")
    warm_kw = dict(draws=2, tune=2, chains=1, progressbar=False)
    if gibbs_backend:
        warm_kw["gibbs_backend"] = gibbs_backend
    if sampler == "nuts":
        warm_kw["nuts_sampler"] = nuts_sampler
    m_warm.fit(sampler=sampler, **warm_kw)

    m = model_cls(y=y, X=X, W=W, N=N, T=T, logdet_method="cheb_cholesky")
    t0 = time.perf_counter()
    fit_kw_out = dict(draws=draws, tune=tune, chains=chains, progressbar=False)
    if gibbs_backend:
        fit_kw_out["gibbs_backend"] = gibbs_backend
    fit_kw_out.update(fit_kw)
    idata = m.fit(sampler=sampler, **fit_kw_out)
    elapsed = time.perf_counter() - t0
    post = idata.posterior
    rho_ess = _min_ess(post, "rho")
    beta_ess = _min_ess(post, "beta")
    label = sampler
    if sampler == "gibbs" and gibbs_backend:
        label = f"gibbs-{gibbs_backend}"
    elif sampler == "nuts":
        label = f"nuts-{nuts_sampler}"
    return dict(
        model=model_cls.__name__,
        sampler=label,
        N=N,
        T=T,
        n_obs=N * T,
        draws=draws,
        chains=chains,
        wall_s=round(elapsed, 2),
        rho_ess=round(rho_ess, 1),
        rho_ess_per_s=round(rho_ess / elapsed, 2),
        beta_ess=round(beta_ess, 1),
        beta_ess_per_s=round(beta_ess / elapsed, 2),
        rho_mean=round(float(post["rho"].mean()), 4),
    )


def run_size(side, T, draws, tune, chains, rows, with_nutpie=False):
    N = side * side
    d = simulate_panel_sar_fe(
        N=N, T=T, rho=TRUE_RHO, beta=TRUE_BETA, sigma=1.0, n_side=side, seed=7
    )
    y, X, W = d["y"], d["X"], d["W_sparse"]
    print(
        f"\n{'=' * 72}\npanel N={N} T={T} n_obs={N * T}  draws={draws} chains={chains}"
        f"\n{'=' * 72}"
    )

    for model_cls in (SARPanelFE, SARPanelRE):
        # Gibbs: numpy + jax backends
        arms = [
            ("gibbs", dict(gibbs_backend="numpy")),
            ("gibbs", dict(gibbs_backend="jax")),
            ("nuts", dict(nuts_sampler="pymc")),
            ("nuts", dict(nuts_sampler="blackjax")),
        ]
        if with_nutpie:
            arms.append(("nuts", dict(nuts_sampler="nutpie")))
        for sampler, kw in arms:
            try:
                row = _run(model_cls, y, X, W, N, T, sampler, draws, tune, chains, **kw)
                rows.append(row)
                print(
                    f"  {row['model']:12} {row['sampler']:16} "
                    f"wall {row['wall_s']:6.1f}s  ρ-ESS/s {row['rho_ess_per_s']:8.1f}  "
                    f"β-ESS/s {row['beta_ess_per_s']:8.1f}  ρ={row['rho_mean']}"
                )
            except Exception as e:  # noqa: BLE001
                print(
                    f"  {model_cls.__name__} {sampler} {kw}: "
                    f"skipped ({type(e).__name__}: {e})"
                )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--with-nutpie", action="store_true", default=True)
    ap.add_argument("--draws", type=int, default=1000)
    ap.add_argument("--tune", type=int, default=1000)
    ap.add_argument("--chains", type=int, default=4)
    args = ap.parse_args()

    if args.quick:
        sizes = [(6, 8)]  # N=36, T=8
        draws, tune, chains = 500, 500, 2
    else:
        sizes = [(10, 10), (15, 10)]  # N=100/225, T=10
        draws, tune, chains = args.draws, args.tune, args.chains

    rows: list = []
    for side, T in sizes:
        run_size(side, T, draws, tune, chains, rows, with_nutpie=args.with_nutpie)

    fields = [
        "model",
        "sampler",
        "N",
        "T",
        "n_obs",
        "draws",
        "chains",
        "wall_s",
        "rho_ess",
        "rho_ess_per_s",
        "beta_ess",
        "beta_ess_per_s",
        "rho_mean",
    ]
    out = RESULTS / "panel_shootout.csv"
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})
    print(f"\nWrote {len(rows)} rows → {out}")


if __name__ == "__main__":
    main()
