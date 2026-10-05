r"""Experiment B — "spatial at the speed of linear" sampler shootout.

Headline question: does native blackjax NUTS on the joint ``(β, σ², ρ)`` SAR
posterior — with a µs differentiable logdet — reach ESS/sec comparable to
**Gibbs on CPU**, and how close does it get to the OLS "speed of light"?

Arms (per n, simulated rook-lattice SAR data, identical priors):

* **OLS blackjax**   — joint ``(β, σ²)`` NUTS; the linear-model yardstick.
* **SAR blackjax**   — joint ``(β, σ², ρ)`` NUTS with logdet from
  eigenvalue (exact reference; n ≤ 10k), chol-cheb (µs surrogate), or SLQ
  (matvec-only precompute; the only arm whose setup survives n ≫ 10⁵).
* **SAR Gibbs**      — the package's blocked Gibbs sampler; the CPU parity bar.
* **SAR PyMC-NUTS**  — optional (``--with-pymc``): ``SAR.fit(sampler="nuts",
  nuts_sampler="blackjax"/"nutpie")`` — the current graph-routed path, to
  contrast against the native one and to confirm the pytensor logdet already
  carries a gradient for nutpie.

Metrics → ``resolvent_paper/results/sampler_shootout.csv``: ESS/sec for ρ and β
(the headline ratio vs Gibbs and vs OLS), draws/sec, mean leapfrog steps and
adapted step size (geometry), and — for surrogate arms — KS distance and
posterior mean/sd deltas of ρ vs the eigenvalue reference (frozen-probe bias for
SLQ as a function of p).

Run: ``conda run -n bayespreg python resolvent_paper/nuts_study.py [--quick]``
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
from scipy.stats import ks_2samp

# Reuse the sparse rook-W builder from the logdet-paper benchmark so no dense
# n×n matrix is ever materialised (n=50k would be ~20 GB dense).
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "logdet_paper"))
from benchmark_logdet_paper import build_rook_W  # noqa: E402

from neighbayes.models import SAR  # noqa: E402
from neighbayes.models.priors import OLSPriors, SARPriors  # noqa: E402
from neighbayes.samplers.gaussian._blackjax import (  # noqa: E402
    run_chain_blackjax_gaussian,
)

RESULTS = Path(__file__).resolve().parent / "results"
RESULTS.mkdir(exist_ok=True)

PRIORS = {"beta_mu": 0.0, "beta_sigma": 10.0, "sigma2_beta": 1.0}
RHO_LO, RHO_HI = 0.0, 0.95
TRUE_RHO = 0.6
TRUE_BETA = np.array([1.0, 2.0])
EIG_CAP = 10000  # eigenvalue reference / densification only at or below this n


def make_sar_data(n_side, rho, beta, seed):
    """Simulate SAR data on a sparse rook lattice — never densifies W.

    y = (I − ρW)⁻¹(Xβ + ε) via a sparse solve.  Returns
    ``(y, X, W_csr, eigs_or_None)``; eigenvalues are computed (densifying once)
    only when n ≤ EIG_CAP so the exact reference stays feasible.
    """
    W = build_rook_W(n_side * n_side)
    n = W.shape[0]
    rng = np.random.default_rng(seed)
    x1 = rng.standard_normal(n)
    X = np.column_stack([np.ones(n), x1])
    eps = rng.standard_normal(n)
    rhs = X @ beta + eps
    A = sp.eye(n, format="csc") - rho * W.tocsc()
    y = spla.spsolve(A, rhs)
    eigs = np.linalg.eigvals(W.toarray()) if n <= EIG_CAP else None
    return y, X, W.tocsr(), eigs


def _sar_priors():
    return SARPriors(rho_lower=RHO_LO, rho_upper=RHO_HI, **PRIORS)


def _ess(arr_2d) -> float:
    """ESS from a (chains, draws) array via arviz."""
    return float(az.ess(np.asarray(arr_2d)))


def _min_beta_ess(beta) -> float:
    # beta: (chains, draws, k) → worst-case ESS across coefficients.
    return float(min(_ess(beta[:, :, j]) for j in range(beta.shape[2])))


# ---------------------------------------------------------------------------
# Arms
# ---------------------------------------------------------------------------


def run_blackjax_arm(
    name,
    y,
    X,
    W,
    priors,
    *,
    spatial,
    method,
    draws,
    tune,
    chains,
    eigs=None,
    rho_lo=RHO_LO,
    rho_hi=RHO_HI,
):
    res = run_chain_blackjax_gaussian(
        y,
        X,
        W,
        priors,
        spatial=spatial,
        logdet_method=method,
        eigs=eigs,
        draws=draws,
        tune=tune,
        chains=chains,
        seed=0,
        rho_lower=rho_lo,
        rho_upper=rho_hi,
    )
    total = res.warmup_time + res.sampling_time
    row = dict(
        arm=name,
        n=len(y),
        draws=draws,
        chains=chains,
        warmup_s=round(res.warmup_time, 3),
        sampling_s=round(res.sampling_time, 3),
        total_s=round(total, 3),
        draws_per_s=round(draws * chains / res.sampling_time, 1),
        beta_ess=round(_min_beta_ess(res.beta), 1),
        beta_ess_per_s=round(_min_beta_ess(res.beta) / res.sampling_time, 2),
        mean_leapfrog=round(float(res.num_integration_steps.mean()), 2),
        step_size=round(res.step_size, 4),
        accept=round(float(res.acceptance_rate.mean()), 3),
        divergent=res.num_divergent,
    )
    if spatial:
        row.update(
            rho_mean=round(float(res.rho.mean()), 4),
            rho_sd=round(float(res.rho.std()), 4),
            rho_ess=round(_ess(res.rho), 1),
            rho_ess_per_s=round(_ess(res.rho) / res.sampling_time, 2),
        )
    return row, res


def run_gibbs_arm(y, X, W, draws, tune, chains):
    # Auto logdet: the realistic CPU parity bar (eigenvalue for tiny n,
    # cheb_cholesky for the sizes here) — never the O(n³) path at n=50k.
    m = SAR(y=y, X=X, W=W, priors=PRIORS, logdet_method=None)
    t0 = time.perf_counter()
    idata = m.fit(
        sampler="gibbs", draws=draws, tune=tune, chains=chains, progressbar=False
    )
    elapsed = time.perf_counter() - t0
    post = idata.posterior
    rho = np.asarray(post["rho"])  # (chains, draws)
    beta = np.asarray(post["beta"])  # (chains, draws, k)
    rho_ess = _ess(rho)
    beta_ess = min(_ess(beta[:, :, j]) for j in range(beta.shape[2]))
    row = dict(
        arm="gibbs",
        n=len(y),
        draws=draws,
        chains=chains,
        warmup_s=None,
        sampling_s=round(elapsed, 3),
        total_s=round(elapsed, 3),
        draws_per_s=round(draws * chains / elapsed, 1),
        beta_ess=round(beta_ess, 1),
        beta_ess_per_s=round(beta_ess / elapsed, 2),
        mean_leapfrog=None,
        step_size=None,
        accept=None,
        divergent=None,
        rho_mean=round(float(rho.mean()), 4),
        rho_sd=round(float(rho.std()), 4),
        rho_ess=round(rho_ess, 1),
        rho_ess_per_s=round(rho_ess / elapsed, 2),
    )
    return row, rho.ravel()


def run_pymc_nuts_arm(sampler_backend, y, X, W, draws, tune, chains):
    m = SAR(y=y, X=X, W=W, priors=PRIORS, logdet_method="cheb_cholesky")
    t0 = time.perf_counter()
    idata = m.fit(
        sampler="nuts",
        nuts_sampler=sampler_backend,
        draws=draws,
        tune=tune,
        chains=chains,
        progressbar=False,
    )
    elapsed = time.perf_counter() - t0
    post = idata.posterior
    rho = np.asarray(post["rho"])
    beta = np.asarray(post["beta"])
    rho_ess = _ess(rho)
    beta_ess = min(_ess(beta[:, :, j]) for j in range(beta.shape[2]))
    return dict(
        arm=f"pymc-{sampler_backend}",
        n=len(y),
        draws=draws,
        chains=chains,
        warmup_s=None,
        sampling_s=round(elapsed, 3),
        total_s=round(elapsed, 3),
        draws_per_s=round(draws * chains / elapsed, 1),
        beta_ess=round(beta_ess, 1),
        beta_ess_per_s=round(beta_ess / elapsed, 2),
        mean_leapfrog=None,
        step_size=None,
        accept=None,
        divergent=None,
        rho_mean=round(float(rho.mean()), 4),
        rho_sd=round(float(rho.std()), 4),
        rho_ess=round(rho_ess, 1),
        rho_ess_per_s=round(rho_ess / elapsed, 2),
    )


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def run_size(n_side, draws, tune, chains, slq_probes, with_pymc, rows):
    y, X, W, eigs = make_sar_data(n_side, TRUE_RHO, TRUE_BETA, seed=7)
    n = len(y)
    print(
        f"\n{'=' * 70}\nn={n}  (side={n_side})  draws={draws} tune={tune} "
        f"chains={chains}\n{'=' * 70}"
    )

    ref_rho = None

    # OLS yardstick (same X, no spatial term).
    row, _ = run_blackjax_arm(
        "ols_blackjax",
        y,
        X,
        None,
        OLSPriors(**PRIORS),
        spatial=False,
        method=None,
        draws=draws,
        tune=tune,
        chains=chains,
    )
    rows.append(row)
    print(
        f"  ols_blackjax        β-ESS/s {row['beta_ess_per_s']:>8}  "
        f"draws/s {row['draws_per_s']:>7}"
    )

    # SAR eigenvalue reference (only feasible n ≤ EIG_CAP).
    if eigs is not None:
        row, res = run_blackjax_arm(
            "sar_eig",
            y,
            X,
            W,
            _sar_priors(),
            spatial=True,
            method="eigenvalue",
            draws=draws,
            tune=tune,
            chains=chains,
            eigs=eigs,
        )
        rows.append(row)
        ref_rho = res.rho.ravel()
        print(
            f"  sar_eig  (ref)      ρ-ESS/s {row['rho_ess_per_s']:>8}  "
            f"β-ESS/s {row['beta_ess_per_s']:>8}  ρ={row['rho_mean']} "
            f"leapfrog {row['mean_leapfrog']}"
        )

    # SAR chol-cheb surrogate.
    row, res = run_blackjax_arm(
        "sar_chol_cheb",
        y,
        X,
        W,
        _sar_priors(),
        spatial=True,
        method="cheb_cholesky",
        draws=draws,
        tune=tune,
        chains=chains,
    )
    if ref_rho is not None:
        row["ks_vs_ref"] = round(ks_2samp(res.rho.ravel(), ref_rho).statistic, 4)
    rows.append(row)
    print(
        f"  sar_chol_cheb       ρ-ESS/s {row['rho_ess_per_s']:>8}  "
        f"β-ESS/s {row['beta_ess_per_s']:>8}  ρ={row['rho_mean']} "
        f"KS {row.get('ks_vs_ref', '—')}"
    )

    # SAR SLQ (frozen-probe) at each probe count.
    for p in slq_probes:
        # Build a probe-specific logdet by monkeypatching the precompute default
        # is awkward; instead use the "slq" method (default 50 probes) for the
        # canonical arm and note p in the arm name via a dedicated method call.
        row, res = run_blackjax_arm(
            f"sar_slq_p{p}",
            y,
            X,
            W,
            _sar_priors(),
            spatial=True,
            method="slq",
            draws=draws,
            tune=tune,
            chains=chains,
        )
        if ref_rho is not None:
            row["ks_vs_ref"] = round(ks_2samp(res.rho.ravel(), ref_rho).statistic, 4)
        row["slq_probes"] = p
        rows.append(row)
        print(
            f"  sar_slq (p={p:>3})     ρ-ESS/s {row['rho_ess_per_s']:>8}  "
            f"β-ESS/s {row['beta_ess_per_s']:>8}  ρ={row['rho_mean']} "
            f"KS {row.get('ks_vs_ref', '—')}"
        )
        break  # default precompute is 50 probes; one SLQ arm unless extended

    # SAR Gibbs — CPU parity bar.
    row, gibbs_rho = run_gibbs_arm(y, X, W, draws, tune, chains)
    if ref_rho is not None:
        row["ks_vs_ref"] = round(ks_2samp(gibbs_rho, ref_rho).statistic, 4)
    rows.append(row)
    print(
        f"  gibbs (parity bar)  ρ-ESS/s {row['rho_ess_per_s']:>8}  "
        f"β-ESS/s {row['beta_ess_per_s']:>8}  ρ={row['rho_mean']} "
        f"KS {row.get('ks_vs_ref', '—')}"
    )

    # Optional PyMC-routed NUTS arms.
    if with_pymc and n <= 10000:
        for backend in ("nutpie", "blackjax"):
            try:
                row = run_pymc_nuts_arm(backend, y, X, W, draws, tune, chains)
                rows.append(row)
                print(
                    f"  pymc-{backend:8}     ρ-ESS/s {row['rho_ess_per_s']:>8}  "
                    f"draws/s {row['draws_per_s']:>7}"
                )
            except Exception as e:  # noqa: BLE001
                print(f"  pymc-{backend}: skipped ({type(e).__name__}: {e})")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="small n, few draws")
    ap.add_argument(
        "--with-pymc",
        action="store_true",
        help="include PyMC-routed nutpie/blackjax arms",
    )
    ap.add_argument("--draws", type=int, default=2000)
    ap.add_argument("--tune", type=int, default=1000)
    ap.add_argument("--chains", type=int, default=4)
    args = ap.parse_args()

    if args.quick:
        sides, draws, tune, chains = [15], 800, 500, 2
    else:
        # side² observations: 2500, 10000, 50000(≈224²).
        sides, draws, tune, chains = [50, 100, 224], args.draws, args.tune, args.chains

    rows: list = []
    for side in sides:
        run_size(
            side,
            draws,
            tune,
            chains,
            slq_probes=(50,),
            with_pymc=args.with_pymc,
            rows=rows,
        )

    # Union of keys across arms for a stable CSV header.
    fields = [
        "arm",
        "n",
        "draws",
        "chains",
        "warmup_s",
        "sampling_s",
        "total_s",
        "draws_per_s",
        "beta_ess",
        "beta_ess_per_s",
        "rho_mean",
        "rho_sd",
        "rho_ess",
        "rho_ess_per_s",
        "mean_leapfrog",
        "step_size",
        "accept",
        "divergent",
        "ks_vs_ref",
        "slq_probes",
    ]
    out = RESULTS / "sampler_shootout.csv"
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})
    print(f"\nWrote {len(rows)} rows → {out}")


if __name__ == "__main__":
    main()
