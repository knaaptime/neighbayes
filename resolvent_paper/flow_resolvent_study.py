r"""Experiment D (sketch) — unrestricted flow logdet at scale via the resolvent.

The flow model's system matrix is ``A = I_N - W_F``,
``W_F = ρ_d(I⊗W) + ρ_o(W⊗I) + ρ_w(W⊗W)`` with ``N = n²`` and **directed** ``W``
(so eigen-decomposition is out past n≈2k, and the multinomial ``"traces"`` value
method is noise-amplified — both dead ends for real flow sizes).  What *does*
scale is the resolvent-Kronecker gradient

    g_k(ρ) = −tr(W_k (I_N − W_F)⁻¹),   k ∈ {d, o, w},

a single resolvent trace estimated matvec-only (Hutchinson probes + Kronecker
GMRES solves), with the flow value recovered by integrating the gradient along
the ray ``0→ρ`` (:mod:`neighbayes._logdet._flow_resolvent`).  This script sketches
the two gates that matter for the paper:

  (A) **gradient accuracy improves with N.**  For fixed probe count ``P`` the
      relative error of ``g`` follows ``~1/√(N·P)`` — it *sharpens* as the flow
      grows.  We report std% of the dominant components against the exact
      eigenvalue gradient on a sequence of directed W of growing ``n``.

  (B) **ρ-ESS/sec of the sampler.**  MALA-on-ρ within conjugate β,σ² Gibbs
      (:func:`neighbayes.samplers.gaussian._flow_resolvent.sample_flow_resolvent`),
      timed end-to-end, with ESS from ArviZ.  A small-``n`` run is validated
      against an exact-eigenvalue log-det backend so the posterior is known-good
      before the estimator replaces it.

The CPU numbers here are the baseline; the payoff is the **GPU** path — the
Kronecker matvecs are batched ``n×n`` matmuls, GPU-native — which is left as the
`_grad_backend="jax"` hook below (the team's single-param verdict was that GPU is
the open gate, and Kronecker structure is exactly where the CPU deficit can
invert).

Outputs ``resolvent_paper/results/flow_resolvent_study.csv`` + a console table.

Run:  ``conda run -n bayespreg python resolvent_paper/flow_resolvent_study.py``
"""

from __future__ import annotations

import csv
import time
from pathlib import Path

import numpy as np

from neighbayes._logdet._flow_resolvent import (
    flow_logdet_grad,
    flow_logdet_grad_exact,
)

RESULTS = Path(__file__).resolve().parent / "results"
RESULTS.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# Directed flow data generator (row-standardised, always directed).
# ---------------------------------------------------------------------------
def directed_flow_W(n: int, density: float = 0.2, seed: int = 0):
    """Row-standardised directed ``n×n`` weights (asymmetric; complex spectrum)."""
    rng = np.random.default_rng(seed)
    A = (rng.uniform(size=(n, n)) < density).astype(float)
    np.fill_diagonal(A, 0.0)
    A[A.sum(1) == 0, 0] = 1.0
    return A / A.sum(1, keepdims=True)


def simulate_flow(n: int, rho, beta, sigma: float, seed: int = 0):
    """Draw ``A(ρ)y = Xβ + ε`` for a directed flow of size ``N = n²``."""
    W = directed_flow_W(n, seed=seed)
    rng = np.random.default_rng(seed + 1)
    Nf = n * n
    Ide = np.eye(n)
    WF = rho[0] * np.kron(Ide, W) + rho[1] * np.kron(W, Ide) + rho[2] * np.kron(W, W)
    X = np.column_stack([np.ones(Nf), rng.standard_normal(Nf)])
    y = np.linalg.solve(np.eye(Nf) - WF, X @ beta + sigma * rng.standard_normal(Nf))
    return W, y, X


# ---------------------------------------------------------------------------
# Gate A — gradient accuracy vs N (the "sharpens with scale" claim).
# ---------------------------------------------------------------------------
def gate_gradient_accuracy(rho=(0.3, 0.2, -0.05), n_probes=20, reps=8):
    """std% of the stochastic gradient vs the exact eigenvalue gradient, per N."""
    rows = []
    for n in (20, 30, 40):  # N = 400, 900, 1600 (dense-eigval reference feasible)
        W = directed_flow_W(n, seed=1)
        g_exact = flow_logdet_grad_exact(W, *rho)
        ests = np.array(
            [
                flow_logdet_grad(
                    W, *rho, n_probes=n_probes, rng=np.random.default_rng(r)
                )
                for r in range(reps)
            ]
        )
        std_pct = 100.0 * ests.std(0) / np.abs(g_exact)
        bias_pct = 100.0 * np.abs(ests.mean(0) - g_exact) / np.abs(g_exact)
        rows.append(
            {
                "N": n * n,
                "std_pct_g_d": std_pct[0],
                "std_pct_g_o": std_pct[1],
                "bias_pct_g_d": bias_pct[0],
                "bias_pct_g_o": bias_pct[1],
            }
        )
    return rows


# ---------------------------------------------------------------------------
# Gate B — ρ-ESS/sec of the sampler (validated vs an exact-eigenvalue backend).
# ---------------------------------------------------------------------------
def _exact_value_and_grad(W):
    lam = np.linalg.eigvals(W)
    li, lj = lam[:, None], lam[None, :]

    def _vg(rd, ro, rw):
        mu = ro * li + rd * lj + rw * (li * lj)
        return float(np.sum(np.log(np.abs(1.0 - mu)))), flow_logdet_grad_exact(
            W, rd, ro, rw
        )

    return _vg


def gate_ess_per_sec(n=18, draws=800, tune=800, chains=2, backend="exact"):
    """Time ``sample_flow_resolvent`` and report ρ-ESS/sec.

    ``backend="exact"`` injects the eigenvalue log-det (known-good posterior,
    the validation gate); ``backend="resolvent"`` uses the scalable estimator
    (frozen probes) — the production path.  ``backend="jax"`` is the GPU hook.
    """
    import arviz as az

    from neighbayes.samplers.gaussian._flow_resolvent import sample_flow_resolvent

    rho_true = (0.3, 0.2, -0.05)
    W, y, X = simulate_flow(n, rho_true, np.array([1.0, -0.5]), 0.4, seed=3)

    if backend == "exact":
        vg = _exact_value_and_grad(W)
    elif backend == "resolvent":
        vg = None  # sampler builds the frozen-probe resolvent value+grad
    elif backend == "jax":
        raise NotImplementedError(
            "GPU/JAX flow logdet gradient (jax.scipy.sparse.linalg GMRES + vmap "
            "over probes) — the open gate; wire flow_logdet_grad's JAX sibling here."
        )
    else:
        raise ValueError(backend)

    t0 = time.perf_counter()
    idata = sample_flow_resolvent(
        W,
        y,
        X,
        draws=draws,
        tune=tune,
        chains=chains,
        step_size=6e-4,
        coord_names=["const", "x1"],
        logdet_value_and_grad=vg,
        random_seed=1,
    )
    wall = time.perf_counter() - t0

    ess = az.ess(idata, var_names=["rho_d", "rho_o", "rho_w"])
    ess_min = float(min(float(ess[v]) for v in ("rho_d", "rho_o", "rho_w")))
    means = {v: float(idata.posterior[v].mean()) for v in ("rho_d", "rho_o", "rho_w")}
    return {
        "n": n,
        "N": n * n,
        "backend": backend,
        "wall_s": wall,
        "ess_min_rho": ess_min,
        "ess_per_sec": ess_min / wall,
        "rho_d_mean": means["rho_d"],
        "rho_o_mean": means["rho_o"],
        "rho_w_mean": means["rho_w"],
        "rho_true": rho_true,
    }


def main():
    print("=== Gate A: flow gradient accuracy vs N (std% and bias% vs exact) ===")
    grad_rows = gate_gradient_accuracy()
    for r in grad_rows:
        print(
            f"  N={r['N']:>5d}  std%(g_d)={r['std_pct_g_d']:5.1f}  "
            f"std%(g_o)={r['std_pct_g_o']:5.1f}  "
            f"bias%(g_d)={r['bias_pct_g_d']:4.1f}  bias%(g_o)={r['bias_pct_g_o']:4.1f}"
        )

    print("\n=== Gate B: ρ-ESS/sec (exact-logdet validation run) ===")
    ess_row = gate_ess_per_sec(backend="exact")
    print(
        f"  n={ess_row['n']} (N={ess_row['N']})  wall={ess_row['wall_s']:.1f}s  "
        f"ESS_min(ρ)={ess_row['ess_min_rho']:.0f}  "
        f"ESS/sec={ess_row['ess_per_sec']:.1f}"
    )
    print(
        f"  recovered ρ ≈ ({ess_row['rho_d_mean']:.2f}, {ess_row['rho_o_mean']:.2f}, "
        f"{ess_row['rho_w_mean']:.2f})  true ρ = {ess_row['rho_true']}"
    )

    with open(RESULTS / "flow_resolvent_study.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["gate", "key", "value"])
        for r in grad_rows:
            for k, v in r.items():
                w.writerow(["gradient_accuracy", f"N{r['N']}_{k}", v])
        for k, v in ess_row.items():
            w.writerow(["ess_per_sec", k, v])
    print(f"\nWrote {RESULTS / 'flow_resolvent_study.csv'}")
    print(
        "\nNote: CPU baseline. The GPU/JAX path (gate_ess_per_sec(backend='jax')) "
        "is the open lever — Kronecker matvecs are batched n×n matmuls, GPU-native."
    )


if __name__ == "__main__":
    main()
