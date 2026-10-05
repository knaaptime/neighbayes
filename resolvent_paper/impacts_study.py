r"""Experiment E — spatial impacts without an eigendecomposition.

The resolvent identity (§7 of ``notes.md``, shipped in
``neighbayes.models._base._shared``) routes the direct-effect trace

    (1/n) tr((I - rho W)^{-1}) = 1 - (rho/n) * g(rho),   g = d/drho log|I-rho W|

and the SDM cross-direct trace ``(1/n) tr((I-rho W)^{-1} W) = -g(rho)/n``
through the model's fast logdet **gradient** surrogate, so a chol-cheb /
AAA model computes ``spatial_effects()`` with **no** O(n^3)
eigendecomposition.  Previously both were built from the full spectrum via
``_chunked_eig_means(rho, self._W_eigs)``.

This script times ``SAR.spatial_effects()`` two ways at n = 2.5k, 10k, 50k
on a row-standardised rook lattice:

* **eig path**  — ``logdet_method="eigenvalue"`` forces the eigenvalue
  logdet (and thus the eigenvalue gradient), so impacts go through
  ``_chunked_eig_means`` with the pre-computed spectrum;
* **resolvent path** — ``logdet_method="cheb_cholesky"`` (the default for
  this size range): the gradient rides the µs Clenshaw surrogate and
  ``np.linalg.eig`` is never called (verified by poisoning it to raise).

It records wall-time and the max abs difference of the direct/indirect/
total posterior means between the two paths (the ground-truth agreement
check from §7 of the notes).

Outputs ``resolvent_paper/results/impacts_study.csv`` and a console table.

Run:  ``conda run -n bayespreg python resolvent_paper/impacts_study.py [--quick]``
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

# Reuse the sparse rook-W builder from the logdet-paper benchmark so no
# dense n x n matrix is materialised at n=50k.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "logdet_paper"))
from benchmark_logdet_paper import build_rook_W  # noqa: E402

from neighbayes.models import SAR  # noqa: E402

RESULTS = Path(__file__).resolve().parent / "results"
RESULTS.mkdir(exist_ok=True)

PRIORS = {"beta_mu": 0.0, "beta_sigma": 10.0, "sigma2_beta": 1.0}
TRUE_RHO = 0.6
TRUE_BETA = np.array([1.0, 2.0])
DRAW_ARGS = dict(draws=500, tune=500, chains=2)


def make_sar_data(n_side, rho, beta, seed):
    """Simulate SAR data on a sparse rook lattice (never densifies W)."""
    W = build_rook_W(n_side * n_side)
    n = W.shape[0]
    rng = np.random.default_rng(seed)
    x1 = rng.standard_normal(n)
    X = np.column_stack([np.ones(n), x1])
    eps = rng.standard_normal(n)
    A = sp.eye(n, format="csc") - rho * W.tocsc()
    y = spla.spsolve(A, X @ beta + eps)
    return y, X, W.tocsr()


def _poison_eig():
    """Make ``np.linalg.eig``/``eigvals`` raise if called (resolvent-path check).

    Returns a restore callable that reinstalls the original routines.
    """
    import numpy.linalg as nl

    orig_eig = nl.eig
    orig_eigvals = nl.eigvals

    def _boom(*a, **k):
        raise AssertionError(
            "np.linalg.eig/eigvals called — resolvent path must not eigendecompose"
        )

    nl.eig = _boom
    nl.eigvals = _boom

    def restore():
        nl.eig = orig_eig
        nl.eigvals = orig_eigvals

    return restore


def _effects_means(model):
    """Return (direct, indirect, total) posterior-mean arrays for covariate 1."""
    df = model.spatial_effects()
    # DataFrame is indexed by feature; columns include ``mean``.
    means = {col: np.asarray(df[col]) for col in ("direct", "indirect", "total")}
    return means


def run_size_small(n_side, rows):
    """Small n where the O(n^3) eigendecomposition is feasible: both paths
    are fit and their posterior-mean impacts compared for agreement."""
    y, X, W = make_sar_data(n_side, TRUE_RHO, TRUE_BETA, seed=7)
    n = len(y)
    print(f"\n{'=' * 70}\nn={n}  (side={n_side})  [agreement check]\n{'=' * 70}")

    # --- eig path (ground truth + timing) ---
    m_eig = SAR(y=y, X=X, W=W, priors=PRIORS, logdet_method="eigenvalue")
    m_eig.fit(sampler="gibbs", progressbar=False, **DRAW_ARGS)
    t0 = time.perf_counter()
    eig_means = _effects_means(m_eig)
    t_eig = time.perf_counter() - t0
    print(f"  eig       wall={t_eig:6.3f}s  direct={eig_means['direct']}")

    # --- resolvent path (chol-cheb) with eig poisoned only around effects ---
    m_cheb = SAR(y=y, X=X, W=W, priors=PRIORS, logdet_method="cheb_cholesky")
    m_cheb.fit(sampler="gibbs", progressbar=False, **DRAW_ARGS)
    restore = _poison_eig()
    try:
        t0 = time.perf_counter()
        cheb_means = _effects_means(m_cheb)
        t_cheb = time.perf_counter() - t0
        eig_called = False
    except AssertionError as e:
        t_cheb = float("nan")
        cheb_means = {k: np.full_like(v, np.nan) for k, v in eig_means.items()}
        eig_called = True
        print(f"  [FAIL] resolvent path called eig: {e}")
    finally:
        restore()

    max_abs_diff = max(
        float(np.max(np.abs(cheb_means[k] - eig_means[k])))
        for k in ("direct", "indirect", "total")
    )
    print(
        f"  resolvent wall={t_cheb:6.3f}s  direct={cheb_means['direct']}  "
        f"max|Δ|={max_abs_diff:.2e}  eig_called={eig_called}"
    )
    rows.append(
        dict(
            n=n,
            side=n_side,
            path="eigenvalue",
            wall_s=round(t_eig, 4),
            direct=float(eig_means["direct"][0]),
            indirect=float(eig_means["indirect"][0]),
            total=float(eig_means["total"][0]),
            max_abs_diff_vs_eig=0.0,
            eig_called=False,
        )
    )
    rows.append(
        dict(
            n=n,
            side=n_side,
            path="cheb_cholesky",
            wall_s=round(t_cheb, 4) if not np.isnan(t_cheb) else "",
            direct=float(cheb_means["direct"][0]),
            indirect=float(cheb_means["indirect"][0]),
            total=float(cheb_means["total"][0]),
            max_abs_diff_vs_eig=max_abs_diff,
            eig_called=eig_called,
        )
    )


def run_size_large(n_side, rows):
    """Large n where the eigendecomposition is infeasible: resolvent path
    only, timing + the eig-poison check (no eig comparison row)."""
    y, X, W = make_sar_data(n_side, TRUE_RHO, TRUE_BETA, seed=7)
    n = len(y)
    print(
        f"\n{'=' * 70}\nn={n}  (side={n_side})  [resolvent only, eig infeasible]\n{'=' * 70}"
    )

    m_cheb = SAR(y=y, X=X, W=W, priors=PRIORS, logdet_method="cheb_cholesky")
    m_cheb.fit(sampler="gibbs", progressbar=False, **DRAW_ARGS)
    restore = _poison_eig()
    try:
        t0 = time.perf_counter()
        cheb_means = _effects_means(m_cheb)
        t_cheb = time.perf_counter() - t0
        eig_called = False
    except AssertionError as e:
        t_cheb = float("nan")
        cheb_means = {k: np.array([np.nan]) for k in ("direct", "indirect", "total")}
        eig_called = True
        print(f"  [FAIL] resolvent path called eig: {e}")
    finally:
        restore()

    print(
        f"  resolvent wall={t_cheb:6.3f}s  direct={cheb_means['direct']}  "
        f"eig_called={eig_called}"
    )
    rows.append(
        dict(
            n=n,
            side=n_side,
            path="cheb_cholesky",
            wall_s=round(t_cheb, 4) if not np.isnan(t_cheb) else "",
            direct=float(cheb_means["direct"][0]),
            indirect=float(cheb_means["indirect"][0]),
            total=float(cheb_means["total"][0]),
            max_abs_diff_vs_eig="",  # eig path infeasible at this n
            eig_called=eig_called,
        )
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    rows: list = []
    if args.quick:
        run_size_small(15, rows)
    else:
        # Small n: agreement check (both paths feasible).
        for side in (20, 50):  # n = 400, 2500
            run_size_small(side, rows)
        # Large n: resolvent only (eigendecomposition infeasible).
        for side in (100, 224):  # n = 10000, 50176
            run_size_large(side, rows)
    out = RESULTS / "impacts_study.csv"
    fields = [
        "n",
        "side",
        "path",
        "wall_s",
        "direct",
        "indirect",
        "total",
        "max_abs_diff_vs_eig",
        "eig_called",
    ]
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})
    print(f"\nWrote {len(rows)} rows -> {out}")


if __name__ == "__main__":
    main()
