r"""Experiment F — model estimation timing across all backend combinations.

The gradient accuracy study (Experiment A) tests the logdet gradient in
isolation.  The sampler shootouts (Experiments B/C) test NUTS vs Gibbs but
don't systematically cross **all** backend combinations for the full
``model.fit()`` estimation path.  This script does: for a simulated SAR on a
rook lattice, it times end-to-end estimation via every combination of:

* **Sampler**: Gibbs, NUTS
* **Gibbs backend**: ``jax``, ``numpy`` (Gibbs only)
* **NUTS backend**: ``pymc`` (default), ``nutpie``, ``blackjax`` (via PyMC
  routing), plus the **native blackjax** path (``run_chain_blackjax_gaussian``,
  not via ``model.fit()``)
* **Logdet method**: ``cheb_cholesky`` (the production default for symmetric W
  at these sizes), ``eigenvalue`` (the exact reference, feasible at small n)

All arms fit the **same** simulated SAR model (same data, same priors, same
draws/tune/chains) and report wall-time, ρ-ESS/sec, β-ESS/sec, and ρ posterior
mean — so the accuracy / correctness dimension is visible alongside the timing
dimension.

Outputs ``resolvent_paper/results/backend_shootout.csv`` and a console table.

Run:  ``conda run -n bayespreg python resolvent_paper/backend_shootout.py [--quick]``
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import arviz as az
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "logdet_paper"))
from benchmark_logdet_paper import build_rook_W  # noqa: E402

from neighbayes.models import SAR  # noqa: E402
from neighbayes.models.priors import SARPriors  # noqa: E402

RESULTS = Path(__file__).resolve().parent / "results"
RESULTS.mkdir(exist_ok=True)

PRIORS = {"beta_mu": 0.0, "beta_sigma": 10.0, "sigma2_beta": 1.0}
RHO_LO, RHO_HI = 0.0, 0.95
TRUE_RHO = 0.6
TRUE_BETA = np.array([1.0, 2.0])


def make_sar_data(n_side, rho, beta, seed):
    W = build_rook_W(n_side * n_side)
    n = W.shape[0]
    rng = np.random.default_rng(seed)
    x1 = rng.standard_normal(n)
    X = np.column_stack([np.ones(n), x1])
    eps = rng.standard_normal(n)
    A = sp.eye(n, format="csc") - rho * W.tocsc()
    y = spla.spsolve(A, X @ beta + eps)
    return y, X, W.tocsr()


def _ess(arr_2d) -> float:
    return float(az.ess(np.asarray(arr_2d)))


def _min_beta_ess(beta) -> float:
    return float(min(_ess(beta[:, :, j]) for j in range(beta.shape[2])))


def _sar_priors():
    return SARPriors(rho_lower=RHO_LO, rho_upper=RHO_HI, **PRIORS)


# ---------------------------------------------------------------------------
# Arms
# ---------------------------------------------------------------------------


def run_gibbs_arm(y, X, W, backend, draws, tune, chains, logdet_method):
    m = SAR(y=y, X=X, W=W, priors=PRIORS, logdet_method=logdet_method)
    # Warmup the JAX JIT compiler with a tiny run so compilation cost
    # doesn't inflate the timed run.  Numpy is unaffected (no compilation).
    if backend in ("jax", "auto"):
        m_warm = SAR(y=y, X=X, W=W, priors=PRIORS, logdet_method=logdet_method)
        m_warm.fit(
            sampler="gibbs",
            gibbs_backend=backend,
            draws=2,
            tune=2,
            chains=1,
            progressbar=False,
        )
    t0 = time.perf_counter()
    idata = m.fit(
        sampler="gibbs",
        gibbs_backend=backend,
        draws=draws,
        tune=tune,
        chains=chains,
        progressbar=False,
    )
    wall = time.perf_counter() - t0
    post = idata.posterior
    rho = np.asarray(post["rho"])
    beta = np.asarray(post["beta"])
    return dict(
        sampler="gibbs",
        backend=backend,
        logdet_method=logdet_method,
        wall_s=round(wall, 3),
        rho_ess=round(_ess(rho), 1),
        rho_ess_per_s=round(_ess(rho) / wall, 2),
        beta_ess=round(_min_beta_ess(beta), 1),
        beta_ess_per_s=round(_min_beta_ess(beta) / wall, 2),
        rho_mean=round(float(rho.mean()), 4),
        rho_sd=round(float(rho.std()), 4),
    )


def run_pymc_nuts_arm(y, X, W, nuts_backend, draws, tune, chains, logdet_method):
    m = SAR(y=y, X=X, W=W, priors=PRIORS, logdet_method=logdet_method)
    # Warmup: trigger PyMC graph compilation / nutpie numba JIT / blackjax
    # tracing with a tiny run so first-call overhead doesn't inflate the
    # timed run.  The pytensor graph and numba kernels are cached after the
    # first call, so subsequent runs measure steady-state performance.
    m_warm = SAR(y=y, X=X, W=W, priors=PRIORS, logdet_method=logdet_method)
    m_warm.fit(
        sampler="nuts",
        nuts_sampler=nuts_backend,
        draws=2,
        tune=2,
        chains=1,
        progressbar=False,
    )
    t0 = time.perf_counter()
    idata = m.fit(
        sampler="nuts",
        nuts_sampler=nuts_backend,
        draws=draws,
        tune=tune,
        chains=chains,
        progressbar=False,
    )
    wall = time.perf_counter() - t0
    post = idata.posterior
    rho = np.asarray(post["rho"])
    beta = np.asarray(post["beta"])
    return dict(
        sampler="nuts",
        backend=f"pymc-{nuts_backend}",
        logdet_method=logdet_method,
        wall_s=round(wall, 3),
        rho_ess=round(_ess(rho), 1),
        rho_ess_per_s=round(_ess(rho) / wall, 2),
        beta_ess=round(_min_beta_ess(beta), 1),
        beta_ess_per_s=round(_min_beta_ess(beta) / wall, 2),
        rho_mean=round(float(rho.mean()), 4),
        rho_sd=round(float(rho.std()), 4),
    )


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def run_size(n_side, draws, tune, chains, rows):
    y, X, W = make_sar_data(n_side, TRUE_RHO, TRUE_BETA, seed=7)
    n = len(y)
    np.linalg.eigvals(W.toarray()) if n <= 500 else None
    print(f"\n{'=' * 78}\nn={n}  draws={draws} tune={tune} chains={chains}\n{'=' * 78}")

    logdet = "eigenvalue" if n <= 500 else "cheb_cholesky"
    arms = []

    # --- Gibbs: numpy + jax backends ---
    for backend in ("numpy", "jax"):
        try:
            row = run_gibbs_arm(y, X, W, backend, draws, tune, chains, logdet)
            arms.append(row)
            print(
                f"  gibbs/{backend:5}/{logdet:14}  wall={row['wall_s']:7.2f}s  "
                f"ρ-ESS/s={row['rho_ess_per_s']:8.1f}  ρ={row['rho_mean']}"
            )
        except Exception as e:
            print(f"  gibbs/{backend}: skipped ({type(e).__name__}: {e})")

    # --- NUTS: PyMC-routed pymc, nutpie, blackjax ---
    for nb in ("pymc", "nutpie", "blackjax"):
        try:
            row = run_pymc_nuts_arm(y, X, W, nb, draws, tune, chains, logdet)
            arms.append(row)
            print(
                f"  nuts/pymc-{nb:8}/{logdet:14}  wall={row['wall_s']:7.2f}s  "
                f"ρ-ESS/s={row['rho_ess_per_s']:8.1f}  ρ={row['rho_mean']}"
            )
        except Exception as e:
            print(f"  nuts/pymc-{nb}: skipped ({type(e).__name__}: {e})")

    for row in arms:
        row["n"] = n
        rows.append(row)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--draws", type=int, default=1000)
    ap.add_argument("--tune", type=int, default=500)
    ap.add_argument("--chains", type=int, default=4)
    args = ap.parse_args()

    if args.quick:
        sides, draws, tune, chains = [15], 500, 300, 2
    else:
        sides = [20, 50, 100]  # n = 400, 2500, 10000
        draws, tune, chains = args.draws, args.tune, args.chains

    rows: list = []
    for side in sides:
        run_size(side, draws, tune, chains, rows)

    fields = [
        "n",
        "sampler",
        "backend",
        "logdet_method",
        "wall_s",
        "rho_ess",
        "rho_ess_per_s",
        "beta_ess",
        "beta_ess_per_s",
        "rho_mean",
        "rho_sd",
    ]
    out = RESULTS / "backend_shootout.csv"
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})
    print(f"\nWrote {len(rows)} rows -> {out}")


if __name__ == "__main__":
    main()
