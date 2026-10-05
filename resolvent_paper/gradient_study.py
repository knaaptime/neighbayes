r"""Experiment A — resolvent-trace / logdet-gradient accuracy and cost.

Question: how accurately and how cheaply can each method deliver

    g(ρ) = d/dρ log|I − ρW| = −tr(W (I − ρW)⁻¹),

the quantity gradient-based samplers (NUTS/MALA) need?  Compared here:

* **surrogate derivatives** (deterministic given the precompute) — the analytic
  ``d/dρ`` of the chol-cheb / AAA / SLQ / cheb-stochastic logdet surrogates,
  from :mod:`neighbayes._logdet._resolvent`;
* **CG-resolvent** (fresh Hutchinson probes, matvec-only) — an unbiased but
  noisy estimator, the "no-truncation" alternative the original proposal
  described, incl. a Perron-deflated variant for the ρ→1 regime;
* **exact ground truth** — dense eigenvalues at small n, else a
  central finite-difference of the sparse-Cholesky/LU exact logdet.

An internal gate first confirms the package numpy ``_resolvent`` core agrees
with ``jax.grad`` of the JAX logdet closures to ~1e-11 — i.e. the numpy path and
the autodiff paths compute the same object.

Outputs ``resolvent_paper/results/gradient_accuracy.csv`` and a console table.

Run:  ``conda run -n bayespreg python resolvent_paper/gradient_study.py``
"""

from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

# Reuse the graph builders + D-symmetrisation from the logdet-paper benchmark.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "logdet_paper"))
from benchmark_logdet_paper import (  # noqa: E402
    build_knn_W,
    build_rook_W,
    exact_logdets_cholesky,
    exact_logdets_lu,
)

from neighbayes._logdet import (  # noqa: E402
    aaa_logdet_precompute,
    chol_cheb_logdet_precompute,
    logdet_grad_aaa,
    logdet_grad_chebyshev,
    logdet_grad_eigenvalue,
    logdet_grad_slq,
    slq_logdet_precompute,
)
from neighbayes._logdet._slq import _recover_symmetrizing_diagonal  # noqa: E402

RESULTS = Path(__file__).resolve().parent / "results"
RESULTS.mkdir(exist_ok=True)

# ρ grid: dense in the usual band plus the hard near-singular tail.
RHOS = np.array([0.0, 0.3, 0.6, 0.8, 0.9, 0.95, 0.99])
DENSE_EIG_CAP = 6000  # n above which we finite-difference the exact logdet
SLQ_PROBES = (10, 30, 50, 100)
CG_PROBES = (8, 32, 128)
SIZES = (2500, 10000)


# ---------------------------------------------------------------------------
# Ground truth
# ---------------------------------------------------------------------------


def ground_truth_grad(
    W: sp.csr_matrix, rhos: np.ndarray, symmetric: bool
) -> np.ndarray:
    """Exact g(ρ): dense eigenvalues at small n, else central FD of exact logdet."""
    n = W.shape[0]
    if n <= DENSE_EIG_CAP:
        eigs = np.linalg.eigvals(W.toarray())
        return np.array([float(logdet_grad_eigenvalue(r, eigs)) for r in rhos])
    # Finite-difference the sparse exact logdet (O(nnz) factorisations).
    h = 1e-4
    exact = exact_logdets_cholesky if symmetric else exact_logdets_lu
    lo = exact(W, rhos - h)
    hi = exact(W, rhos + h)
    return (hi - lo) / (2 * h)


# ---------------------------------------------------------------------------
# CG-resolvent estimators (fresh Hutchinson probes, matvec-only)
# ---------------------------------------------------------------------------


def _symmetrized_operators(W: sp.csr_matrix):
    """Return (W_sym as LinearOperator matvec, n) for symmetric-sparsity W."""
    D = _recover_symmetrizing_diagonal(W)
    if D is None:
        return None
    sqrtD = np.sqrt(D)
    inv_sqrtD = 1.0 / sqrtD

    def matvec(x):
        return sqrtD * (W @ (inv_sqrtD * x))

    return matvec, W.shape[0]


def cg_resolvent_grad(
    W: sp.csr_matrix, rho: float, p: int, rng, symmetric: bool, tol=1e-8
) -> float:
    """Unbiased Hutchinson estimate of g(ρ) = −tr(W(I−ρW)⁻¹) via iterative solves.

    Symmetric W: D-symmetrise → SPD ``(I − ρW_sym)`` → CG.  Directed W:
    BiCGSTAB on ``(I − ρW)``.  Per probe: one solve + one matvec.
    """
    n = W.shape[0]
    Z = rng.integers(0, 2, size=(n, p)).astype(np.float64) * 2 - 1  # Rademacher

    if symmetric:
        matvec, _ = _symmetrized_operators(W)
        A = spla.LinearOperator((n, n), matvec=lambda x: x - rho * matvec(x))
        Wop = matvec
    else:
        eye = sp.eye(n, format="csr")
        A = (eye - rho * W).tocsr()
        Wop = lambda x: W @ x  # noqa: E731

    acc = 0.0
    for j in range(p):
        z = Z[:, j]
        if symmetric:
            v, _ = spla.cg(A, z, rtol=tol, maxiter=2000)
        else:
            v, _ = spla.bicgstab(A, z, rtol=tol, maxiter=2000)
        acc += float(z @ Wop(v))
    return -acc / p


def cg_resolvent_grad_deflated(
    W: sp.csr_matrix, rho: float, p: int, rng, tol=1e-8
) -> float:
    """Perron-deflated CG estimate (symmetric W only).

    Row-standardised W has a λ=1 Perron mode; on the D-symmetrised operator its
    eigenvector is ``u ∝ D^{1/2}·1``.  Its exact contribution to the resolvent
    trace is ``1/(1−ρ)`` (which diverges as ρ→1); the remainder is estimated
    with probes projected off ``u`` so the noisy part stays bounded.
    """
    D = _recover_symmetrizing_diagonal(W)
    if D is None:
        raise ValueError("deflation requires symmetric-sparsity W")
    n = W.shape[0]
    sqrtD = np.sqrt(D)
    inv_sqrtD = 1.0 / sqrtD
    u = sqrtD / np.linalg.norm(sqrtD)  # Perron eigvec of W_sym (eigenvalue 1)

    def w_sym(x):
        return sqrtD * (W @ (inv_sqrtD * x))

    A = spla.LinearOperator((n, n), matvec=lambda x: x - rho * w_sym(x))

    Z = rng.integers(0, 2, size=(n, p)).astype(np.float64) * 2 - 1
    acc = 0.0
    for j in range(p):
        z = Z[:, j]
        z = z - u * (u @ z)  # project off the Perron direction
        v, _ = spla.cg(A, z, rtol=tol, maxiter=2000)
        v = v - u * (u @ v)
        acc += float(z @ w_sym(v))
    remainder = -acc / p
    perron = -1.0 / (1.0 - rho)  # exact λ=1 contribution to g(ρ)
    return remainder + perron


# ---------------------------------------------------------------------------
# Internal gate: numpy _resolvent  ==  jax.grad(jax closure)
# ---------------------------------------------------------------------------


def autodiff_parity_gate(W_sym: sp.csr_matrix, W_dir: sp.csr_matrix) -> float:
    """Return worst |Δ| between numpy _resolvent and jax.grad (also prints)."""
    try:
        import jax

        jax.config.update("jax_enable_x64", True)
        import jax.numpy as jnp
    except Exception:
        print("  [gate] JAX not available — skipping autodiff-parity check")
        return float("nan")
    from neighbayes._logdet import make_logdet_jax_fn

    checks = []
    # chol-cheb (symmetric)
    pre = chol_cheb_logdet_precompute(W_sym, order=None, rho_min=0.0, rho_max=0.95)
    fn = make_logdet_jax_fn(W_sym, method="cheb_cholesky", rho_min=0.0, rho_max=0.95)
    for r in (0.2, 0.5, 0.8):
        a = float(logdet_grad_chebyshev(r, pre.coeffs, pre.rho_min, pre.rho_max))
        b = float(jax.grad(fn)(jnp.float64(r)))
        checks.append(abs(a - b))
    # aaa (directed)
    pra = aaa_logdet_precompute(W_dir, rho_min=0.0, rho_max=0.95)
    fna = make_logdet_jax_fn(W_dir, method="aaa", rho_min=0.0, rho_max=0.95)
    for r in (0.2, 0.5, 0.8):
        a = float(
            logdet_grad_aaa(r, pra.support_points, pra.support_values, pra.weights)
        )
        b = float(jax.grad(fna)(jnp.float64(r)))
        checks.append(abs(a - b))
    # slq (symmetric)
    prs = slq_logdet_precompute(W_sym)
    fns = make_logdet_jax_fn(W_sym, method="slq")
    for r in (0.2, 0.5, 0.8):
        a = float(logdet_grad_slq(r, prs.nodes, prs.weights, prs.n_probes))
        b = float(jax.grad(fns)(jnp.float64(r)))
        checks.append(abs(a - b))
    worst = max(checks)
    status = "PASS" if worst < 1e-8 else "FAIL"
    print(f"  [gate] numpy _resolvent vs jax.grad: worst |Δ| = {worst:.2e}  [{status}]")
    return worst


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def _relerr(approx, truth):
    return abs(approx - truth) / (abs(truth) + 1e-12)


def _record(rows, *, family, n, rho, method, param, grad, truth, setup_s, eval_s):
    rows.append(
        dict(
            family=family,
            n=n,
            rho=float(rho),
            method=method,
            param=param,
            grad=grad,
            truth=truth,
            abserr=abs(grad - truth),
            relerr=_relerr(grad, truth),
            setup_s=setup_s,
            eval_s=eval_s,
        )
    )


def run_family(family: str, builder, symmetric: bool, rows: list) -> None:
    for n in SIZES:
        W = builder(n)
        n_actual = W.shape[0]
        print(f"\n[{family}] n={n_actual}  (symmetric={symmetric})")
        truth = ground_truth_grad(W, RHOS, symmetric)

        # --- deterministic surrogate derivatives ---
        # chol-cheb / aaa depend on symmetry.
        if symmetric:
            t0 = time.perf_counter()
            pre = chol_cheb_logdet_precompute(W, order=None, rho_min=0.0, rho_max=0.99)
            setup_surr = time.perf_counter() - t0
            grad_surr = lambda r: float(  # noqa: E731
                logdet_grad_chebyshev(r, pre.coeffs, pre.rho_min, pre.rho_max)
            )
            surr_name = "chol_cheb"
        else:
            t0 = time.perf_counter()
            pra = aaa_logdet_precompute(W, rho_min=0.0, rho_max=0.99)
            setup_surr = time.perf_counter() - t0
            grad_surr = lambda r: float(  # noqa: E731
                logdet_grad_aaa(r, pra.support_points, pra.support_values, pra.weights)
            )
            surr_name = "aaa"

        for i, r in enumerate(RHOS):
            t0 = time.perf_counter()
            g = grad_surr(float(r))
            ev = time.perf_counter() - t0
            _record(
                rows,
                family=family,
                n=n_actual,
                rho=r,
                method=surr_name,
                param="",
                grad=g,
                truth=truth[i],
                setup_s=setup_surr,
                eval_s=ev,
            )

        # --- JAX jax.grad of the value closure (autodiff path) ---
        try:
            import jax

            jax.config.update("jax_enable_x64", True)
            import jax.numpy as jnp

            from neighbayes._logdet import make_logdet_jax_fn

            jax_method = "cheb_cholesky" if symmetric else "aaa"
            t0 = time.perf_counter()
            jax_fn = make_logdet_jax_fn(W, method=jax_method, rho_min=0.0, rho_max=0.99)
            jax_grad_fn = jax.grad(jax_fn)
            setup_jax = time.perf_counter() - t0
            for i, r in enumerate(RHOS):
                t0 = time.perf_counter()
                g = float(jax_grad_fn(jnp.float64(r)))
                ev = time.perf_counter() - t0
                _record(
                    rows,
                    family=family,
                    n=n_actual,
                    rho=r,
                    method=f"jax_grad_{surr_name}",
                    param="",
                    grad=g,
                    truth=truth[i],
                    setup_s=setup_jax,
                    eval_s=ev,
                )
        except Exception as e:  # noqa: BLE001
            print(f"  [jax_grad] skipped for {family}: {type(e).__name__}: {e}")

        # --- SLQ analytic derivative (probe sweep) ---
        for p in SLQ_PROBES:
            t0 = time.perf_counter()
            prs = slq_logdet_precompute(W, n_probes=p)
            setup = time.perf_counter() - t0
            for i, r in enumerate(RHOS):
                t0 = time.perf_counter()
                g = float(logdet_grad_slq(r, prs.nodes, prs.weights, prs.n_probes))
                ev = time.perf_counter() - t0
                _record(
                    rows,
                    family=family,
                    n=n_actual,
                    rho=r,
                    method="slq",
                    param=f"p={p}",
                    grad=g,
                    truth=truth[i],
                    setup_s=setup,
                    eval_s=ev,
                )

        # --- CG-resolvent (fresh probes, matvec-only) ---
        rng = np.random.default_rng(0)
        for p in CG_PROBES:
            for i, r in enumerate(RHOS):
                t0 = time.perf_counter()
                g = cg_resolvent_grad(W, float(r), p, rng, symmetric)
                ev = time.perf_counter() - t0
                _record(
                    rows,
                    family=family,
                    n=n_actual,
                    rho=r,
                    method="cg",
                    param=f"p={p}",
                    grad=g,
                    truth=truth[i],
                    setup_s=0.0,
                    eval_s=ev,
                )

        # --- Perron-deflated CG (symmetric, ρ→1 regime) ---
        if symmetric:
            for p in CG_PROBES:
                for i, r in enumerate(RHOS):
                    if r < 0.85:
                        continue
                    t0 = time.perf_counter()
                    g = cg_resolvent_grad_deflated(W, float(r), p, rng)
                    ev = time.perf_counter() - t0
                    _record(
                        rows,
                        family=family,
                        n=n_actual,
                        rho=r,
                        method="cg_deflated",
                        param=f"p={p}",
                        grad=g,
                        truth=truth[i],
                        setup_s=0.0,
                        eval_s=ev,
                    )


def print_summary(rows: list) -> None:
    print("\n" + "=" * 82)
    print("Gradient accuracy: max relerr over 0.3≤ρ≤0.9  |  relerr at ρ=0.99")
    print("(ρ=0 excluded from relerr — g(0)=−tr(W)=0 makes it degenerate)")
    print("=" * 82)
    keys = sorted({(r["family"], r["n"], r["method"], r["param"]) for r in rows})
    header = (
        f"{'family':6} {'n':>6} {'method':12} {'param':7} "
        f"{'relerr.3-.9':>11} {'relerr@.99':>11} {'eval µs':>10}"
    )
    print(header)
    for fam, n, method, param in keys:
        sub = [
            r
            for r in rows
            if r["family"] == fam
            and r["n"] == n
            and r["method"] == method
            and r["param"] == param
        ]
        lo = [r["relerr"] for r in sub if 0.3 <= r["rho"] <= 0.9]
        hi = [r["relerr"] for r in sub if r["rho"] == 0.99]
        ev = np.mean([r["eval_s"] for r in sub]) * 1e6
        lo_s = f"{max(lo):.2e}" if lo else "—"
        hi_s = f"{hi[0]:.2e}" if hi else "—"
        print(f"{fam:6} {n:>6} {method:12} {param:7} {lo_s:>11} {hi_s:>11} {ev:>10.1f}")


def main() -> None:
    print("Experiment A — logdet-gradient accuracy & cost")
    print("Autodiff-parity gate:")
    gate_worst = autodiff_parity_gate(build_rook_W(2500), build_knn_W(2500))

    rows: list = []
    run_family("rook", build_rook_W, True, rows)
    run_family("knn", lambda n: build_knn_W(n, k=8), False, rows)

    # Record the parity gate as a special row so it lands in the CSV.
    if not np.isnan(gate_worst):
        rows.append(
            dict(
                family="parity_gate",
                n=2500,
                rho=0.0,
                method="numpy_vs_jax_grad",
                param="",
                grad=0.0,
                truth=0.0,
                abserr=gate_worst,
                relerr=gate_worst,
                setup_s=0.0,
                eval_s=0.0,
            )
        )

    out = RESULTS / "gradient_accuracy.csv"
    with out.open("w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "family",
                "n",
                "rho",
                "method",
                "param",
                "grad",
                "truth",
                "abserr",
                "relerr",
                "setup_s",
                "eval_s",
            ],
        )
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {len(rows)} rows → {out}")
    print_summary(rows)


if __name__ == "__main__":
    main()
