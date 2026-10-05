r"""Experiment B2 — does a warmup-refit window change the NUTS-vs-Gibbs verdict?

Experiment B found native blackjax NUTS on the joint ``(β, σ², ρ)`` SAR posterior
losing to blocked Gibbs on CPU.  This asks whether a *warmup-adaptive* ρ interval
moves that result, and it is worth asking because the interval enters the NUTS
arm twice, in ways that pull in the same direction:

**Geometry.**  ρ is sampled through
``ρ = lower + (upper − lower)·sigmoid(u)``.  With the prior interval
``[0, 0.95]`` and a posterior of width ~0.01, essentially all of the posterior
maps into a sliver of ``u``-space, and the sampler must take small steps to
resolve it.  Narrowing ``[lower, upper]`` to the region the posterior occupies
rescales ``u`` so the posterior spans an O(1) range — better conditioning at no
cost.

**Jacobian.**  A narrow interval needs fewer interpolation nodes at the same
accuracy, and at a tight tolerance drives the error over the posterior's support
to the factorisation's roundoff floor.

Design — three arms per size, all on the same simulated data and priors:

* ``gibbs``          — the package's blocked Gibbs sampler; the CPU parity bar.
* ``nuts_prior``     — blackjax NUTS on the prior interval (Experiment B's arm).
* ``nuts_refit``     — a scouting run on the prior interval, then a full run on
  the padded post-warmup window.  **The scouting run's wall-clock is charged to
  this arm**, so the comparison is honest: this is what a user would pay.

Metrics → ``resolvent_paper/results/nuts_refit_study.csv``: ESS/sec for ρ and β,
mean leapfrog steps and adapted step size (the geometry channel), the refit
window and its node count, and posterior mean/sd of ρ against the prior-interval
arm so any distortion from the narrowed support is visible.

Result (2026-07, CPU, 4 chains × 1,000 draws)
---------------------------------------------

The answer is no, and the geometry channel is the reason it is no.

======  ==========  ==================  ==========  ==========  =============
n       narrowing   ρ ESS/s vs prior    charged     vs Gibbs    leapfrog
======  ==========  ==================  ==========  ==========  =============
900     1.9×        0.97×               0.29×       0.25×       8.2 → 8.6
2,500   3.1×        0.91×               0.29×       0.19×       9.4 → 9.7
10,000  6.0×        1.08×               0.40×       0.27×       9.3 → 9.1
======  ==========  ==================  ==========  ==========  =============

Even at six-fold narrowing the trajectory length and adapted step size barely
move (9.3 → 9.1 leapfrog steps, step size 0.388 → 0.398) and ESS/sec lands
within ±10% — noise.  The explanation is that **NUTS window adaptation already
does this**: the diagonal mass matrix learns precisely the rescaling a narrow
interval would have supplied, so pre-conditioning the sigmoid is redundant.
Charging the scouting run, the refit arm is a straight 2.5–3.5× loss.

This closes the question rather than leaving it open.  The warmup refit is worth
doing for what it does to the *Jacobian* — fewer factorisations and a near-exact
interpolant over the posterior's support, which is what
``neighbayes`` ships it for — and not as a way to make Hamiltonian methods
competitive with Gibbs on CPU for spatial models.  Experiment B's verdict is
unchanged: NUTS trails Gibbs by 3.7–5.3× here.

Run: ``conda run -n bayespreg python resolvent_paper/nuts_refit_study.py [--quick]``
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from nuts_study import (  # noqa: E402
    RHO_HI,
    RHO_LO,
    TRUE_BETA,
    TRUE_RHO,
    _sar_priors,
    make_sar_data,
    run_blackjax_arm,
    run_gibbs_arm,
)

from neighbayes._logdet._refit import refit_window  # noqa: E402

RESULTS = Path(__file__).resolve().parent / "results"
RESULTS.mkdir(exist_ok=True)
OUT = RESULTS / "nuts_refit_study.csv"

#: Fraction of the full run's warmup spent scouting for the window.  Short on
#: purpose — locating ρ to within a few standard deviations is a much easier
#: problem than sampling it, and the scouting cost is charged to the arm.
SCOUT_FRACTION = 0.5

PAD_SD = 10.0


def run_refit_arm(y, X, W, priors, *, method, draws, tune, chains, eigs=None):
    """Scout on the prior interval, then sample on the padded warmup window.

    Returns ``(row, res)`` shaped like :func:`nuts_study.run_blackjax_arm`, with
    the scouting run's time folded into every timing column so ESS/sec is what a
    user would actually observe.
    """
    scout_tune = max(50, int(tune * SCOUT_FRACTION))
    scout_draws = max(50, int(draws * 0.1))

    t0 = time.perf_counter()
    scout_row, scout = run_blackjax_arm(
        "scout",
        y,
        X,
        W,
        priors,
        spatial=True,
        method=method,
        draws=scout_draws,
        tune=scout_tune,
        chains=chains,
        eigs=eigs,
    )
    scout_s = time.perf_counter() - t0

    window = refit_window(np.asarray(scout.rho).ravel(), RHO_LO, RHO_HI, pad_sd=PAD_SD)
    if window is None:
        return None, None
    lo, hi = window

    row, res = run_blackjax_arm(
        "nuts_refit",
        y,
        X,
        W,
        priors,
        spatial=True,
        method=method,
        draws=draws,
        tune=tune,
        chains=chains,
        eigs=eigs,
        rho_lo=lo,
        rho_hi=hi,
    )

    # Charge the scouting run: it is part of the price of the window.
    row["scout_s"] = round(scout_s, 3)
    row["warmup_s"] = round(row["warmup_s"] + scout_s, 3)
    row["total_s"] = round(row["total_s"] + scout_s, 3)
    charged = res.sampling_time + scout_s
    row["rho_ess_per_s_charged"] = round(row["rho_ess"] / charged, 2)
    row["beta_ess_per_s_charged"] = round(row["beta_ess"] / charged, 2)
    row["window_lo"] = round(lo, 5)
    row["window_hi"] = round(hi, 5)
    row["window_width"] = round(hi - lo, 5)
    row["narrowing"] = round((RHO_HI - RHO_LO) / (hi - lo), 2)
    return row, res


def run_size(n_side, draws, tune, chains, method, rows):
    y, X, W, eigs = make_sar_data(n_side, TRUE_RHO, TRUE_BETA, seed=0)
    n = len(y)
    priors = _sar_priors()
    print(f"\n=== n = {n:,} (rook {n_side}x{n_side}), logdet={method} ===")

    gibbs, _ = run_gibbs_arm(y, X, W, draws, tune, chains)
    gibbs["logdet_method"] = method
    rows.append(gibbs)
    print(
        f"  {'gibbs':<12} rho_ess/s={gibbs['rho_ess_per_s']:>9.2f}  "
        f"rho={gibbs['rho_mean']:.4f}+-{gibbs['rho_sd']:.4f}"
    )

    prior_row, _ = run_blackjax_arm(
        "nuts_prior",
        y,
        X,
        W,
        priors,
        spatial=True,
        method=method,
        draws=draws,
        tune=tune,
        chains=chains,
        eigs=eigs,
    )
    prior_row["logdet_method"] = method
    rows.append(prior_row)
    print(
        f"  {'nuts_prior':<12} rho_ess/s={prior_row['rho_ess_per_s']:>9.2f}  "
        f"rho={prior_row['rho_mean']:.4f}+-{prior_row['rho_sd']:.4f}  "
        f"leapfrog={prior_row['mean_leapfrog']:.1f}  step={prior_row['step_size']:.4f}"
    )

    refit_row, _ = run_refit_arm(
        y,
        X,
        W,
        priors,
        method=method,
        draws=draws,
        tune=tune,
        chains=chains,
        eigs=eigs,
    )
    if refit_row is None:
        print("  nuts_refit   scouting produced no usable window; skipped")
        return
    refit_row["logdet_method"] = method
    rows.append(refit_row)
    print(
        f"  {'nuts_refit':<12} rho_ess/s={refit_row['rho_ess_per_s']:>9.2f}  "
        f"(charged {refit_row['rho_ess_per_s_charged']:>9.2f})  "
        f"rho={refit_row['rho_mean']:.4f}+-{refit_row['rho_sd']:.4f}  "
        f"leapfrog={refit_row['mean_leapfrog']:.1f}  "
        f"step={refit_row['step_size']:.4f}"
    )
    print(
        f"  {'':12} window=[{refit_row['window_lo']}, {refit_row['window_hi']}] "
        f"({refit_row['narrowing']}x narrower)   "
        f"vs prior-interval NUTS: "
        f"ESS/s x{refit_row['rho_ess_per_s'] / max(prior_row['rho_ess_per_s'], 1e-9):.2f}"
        f" (charged x"
        f"{refit_row['rho_ess_per_s_charged'] / max(prior_row['rho_ess_per_s'], 1e-9):.2f})"
        f"   vs Gibbs: x"
        f"{refit_row['rho_ess_per_s_charged'] / max(gibbs['rho_ess_per_s'], 1e-9):.2f}"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="one small size, short chains")
    args = ap.parse_args()

    if args.quick:
        sizes, draws, tune, chains = [(30, "cheb_cholesky")], 400, 400, 2
    else:
        sizes = [(30, "cheb_cholesky"), (50, "cheb_cholesky"), (100, "cheb_cholesky")]
        draws, tune, chains = 1000, 1000, 4

    rows: list[dict] = []
    for n_side, method in sizes:
        run_size(n_side, draws, tune, chains, method, rows)

    fields: list[str] = []
    for r in rows:
        for kk in r:
            if kk not in fields:
                fields.append(kk)
    with OUT.open("w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=fields)
        wr.writeheader()
        wr.writerows(rows)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
