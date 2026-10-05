# Resolvent-trace logdet gradient — working notes

Draft notes for a prospective paper. Companion code lives in the package
(`neighbayes/_logdet/_resolvent.py`, `neighbayes/samplers/gaussian/_blackjax.py`)
and the experiments here (`gradient_study.py`, `nuts_study.py`).

## 1. The idea, corrected

The SAR/SDM likelihood needs `log|I − ρW|`. Gradient-based samplers need its
ρ-derivative, which by Jacobi's formula is the **resolvent trace**

    g(ρ) = d/dρ log|I − ρW| = −tr(W (I − ρW)⁻¹) = −Σᵢ λᵢ/(1 − ρλᵢ).

The original pitch was: estimate this unbiasedly with Hutchinson probes and
sparse solves (no polynomial truncation), plug into NUTS, get exact posterior
targeting. Three corrections turn that into what we actually built.

**(a) NUTS needs the value too, and fresh probes break it.** NUTS evaluates the
log-density throughout tree building and in the Metropolis correction, so a
value is required, not just a gradient. And re-drawing Hutchinson probes every
leapfrog step makes the gradient a *different* function each step — the
Hamiltonian is no longer conserved and detailed balance is lost. The coherent
statement is: **freeze the probes.** With fixed probes the surrogate log-density
is a smooth, deterministic function whose gradient is its exact analytic
derivative; NUTS then samples a *surrogate posterior* whose log-density differs
from the truth by O(1/√p), with **no ρ-truncation anywhere**.

**(b) The surrogate value AND gradient are already in the package.** Every
logdet method the package ships (`cheb_cholesky`, `aaa`, `slq`,
`cheb_stochastic`, `eigenvalue`) evaluates a smooth surrogate `L̂(ρ)`. Its
derivative `L̂'(ρ)` is the resolvent-trace estimate — available in closed form,
in **microseconds**, with no per-step solves. `_resolvent.py` implements those
derivatives; the pytensor and jax logdet closures already expose them through
autodiff. So the "p solves per leapfrog step" is a *reference tool*, not the
workhorse (see §3).

**(c) MALA with an exact-value Metropolis accept is the clean way to use a
noisy gradient.** If one *does* want fresh-probe unbiased gradients (the
CG-resolvent estimator), pair them with a Metropolis accept that uses the exact
log-density: the proposal may be driven by a noisy gradient, but the accept step
keeps the target exact. This isolates "what does an unbiased per-step gradient
buy/cost" from "is the target right." (Experiment B, secondary arm.)

## 2. One object, three consumers

`g(ρ) = −tr(W(I−ρW)⁻¹)` is a single quantity the library needs in three places:

| consumer | needs | today | with `_resolvent` |
|---|---|---|---|
| gradient samplers (NUTS/MALA), any backend | `g(ρ)` | autodiff of pytensor/jax closures (implicit) | explicit closed form + numpy factory |
| spatial impacts (`spatial_effects._chunked_eig_means`) | `mean(λ/(1−ρλ)) = −g(ρ)/n` | dense eigenvalues, O(n³), forms V, V⁻¹ | same quantity at any n, no eigendecomposition |
| the joint log-density | `g(ρ)` as its only extra ρ-term over OLS | — | — |

The unification is the reason the core is **backend-parametrized** (`xp` = numpy
or jax.numpy) rather than a blackjax-specific closure: the numpy path feeds
nutpie-custom / MALA / impacts; the pytensor path already carries it for PyMC
and nutpie-via-PyMC; the jax path for blackjax/numpyro. Same formulas, one
implementation.

## 3. Experiment A — gradient accuracy & cost (`results/gradient_accuracy.csv`)

Internal gate: numpy `_resolvent` vs `jax.grad` of the jax closures agree to
**3.4e-13** — the numpy and autodiff paths compute the same object.

Headline (max relative error over 0.3 ≤ ρ ≤ 0.9 | at ρ = 0.99 | eval time):

| method | rel-err band | rel-err ρ=0.99 | eval |
|---|---|---|---|
| chol-cheb (sym) / AAA (dir) — surrogate deriv | ~3e-5 | ~4e-3 | **9–13 µs** |
| SLQ analytic deriv, p=30–50 (matvec-only setup) | ~1–4 % | ~0.3–2 % | ~6–10 µs |
| CG-resolvent, fresh probes, p=32–128 | ~1–2 % | ~0.4–3 % | **35 ms – 0.5 s** |
| CG-resolvent + Perron deflation (ρ→1) | — | ~2e-4 (best) | 0.1–0.8 s |

Takeaways:
- **Exact surrogate derivatives dominate**: µs-cheap, ~1e-5 accurate, exact by
  construction (chol-cheb/AAA are exact logdet interpolants).
- **CG-resolvent is 10⁴× slower** for similar-or-worse accuracy — it confirms
  the frozen-surrogate route is right and relegates per-step solves to the
  ρ→1 reference regime.
- **Perron deflation** (exact λ=1 mode + probes projected off it) recovers ~10×
  accuracy at ρ=0.99 — the tool if a near-singular application ever needs the
  unbiased estimator.

## 4. Experiment B — "spatial at the speed of linear" (`results/sampler_shootout.csv`)

Native blackjax NUTS on the joint `(β, σ², ρ)` posterior, `Wy` precomputed as
data (residual `y − ρ·Wy − Xβ` is one vector op over OLS). Validated: it targets
the **same posterior** as the package's own SAR Gibbs sampler (ρ, β, σ² means
agree within MC error; KS ≤ 0.02 vs the eigenvalue reference for chol-cheb).

Full run — ρ-ESS/sec (8000 draws = 4 chains × 2000), CPU:

| n | OLS-NUTS β-ESS/s | SAR-NUTS chol-cheb | SAR-NUTS SLQ p50 | SAR-NUTS eig | **Gibbs** |
|---|---|---|---|---|---|
| 2,500  | 4760 | 1432 | 1084 | 1136 | **5015** |
| 10,000 | 2826 | 814  | 691  | 381  | **3081** |
| 50,176 | 1283 | 266  | 311  |  —   | **1627** |

Honest reading — **the CPU-parity heuristic is NOT met**:

- **Blocked Gibbs wins decisively on CPU**: 3.5× (n=2.5k), 3.8× (10k), ~5–6×
  (50k) more ρ-ESS/sec than the best NUTS arm. Two compounding reasons: its
  collapsed slice sampler yields near-independent draws (ρ-ESS ≈ 8000/8000, i.e.
  ESS/draw ≈ 1.0, vs NUTS ≈ 0.45), *and* its wall-time is comparable or faster
  (1.6s vs 2.6–12.7s). It wins on both mixing-per-draw and cost-per-draw. The
  n≈225 pilot showed false parity because fixed NUTS overhead dominates at tiny n.

Two real wins survive, and they're the reusable part:

- **The µs surrogate logdet is the right NUTS ingredient.** chol-cheb NUTS beats
  the eigenvalue-logdet NUTS by 2.1× at n=10k (814 vs 381 ρ-ESS/s) at identical
  ESS/draw — because the eigenvalue logdet costs O(n) per leapfrog (Σ over 10k
  eigenvalues) while Clenshaw costs O(order)≈15. This validates
  `_resolvent`/the differentiable surrogate over the naïve eigenvalue path.
- **The whole gap is the spatial penalty, not NUTS.** OLS-NUTS β-ESS/s (4760 @
  2.5k) ≈ Gibbs-SAR (5015): NUTS on a *linear* model already matches Gibbs on a
  *spatial* one. SAR-NUTS runs ~3× slower than its own OLS baseline — the cost
  of the logdet term plus ρ-geometry (mean tree ≈ 8.5 leapfrog steps vs OLS
  ≈ 4.5). "Spatial at the speed of linear" is the unmet target; closing it is a
  geometry problem (ρ reparameterisation), not a logdet one.
- SLQ's frozen-probe bias grows with n (KS 0.010 → 0.043 from 2.5k→10k) but
  stays small; it's the arm whose *precompute* survives n ≫ 10⁵ / non-planar W.

**GPU is the open door (unmeasured).** blackjax is pure-JAX: it vmaps across
chains and its per-leapfrog cost is an O(nk) matmul plus a µs Clenshaw — both
GPU-native. Gibbs' sequential conjugate draws + stepping-out slice sampler do
not parallelise the same way. So the CPU deficit could narrow or invert on GPU;
this run does not measure it and the claim stays a hypothesis.

## 5. Lineage — references to acquire

The method is new to spatial econometrics but sits in a clear stochastic-trace /
GP lineage. `ubaru2017` (SLQ) is already in `logdet_paper/references.bib`. To
add:
- **Dong et al. 2017** — "Scalable log determinants for GP kernel learning"
  (stochastic Lanczos + probes for GP marginal likelihood).
- **Gardner et al. 2018** — GPyTorch / BBMM (matrix-free GP inference,
  Lanczos-based logdet gradients).
- **Filippone & Girolami 2014** — pseudo-marginal MCMC for GP hyperparameters
  with unbiased-ish likelihood estimates.
- **Lyne et al. 2015** — "Russian roulette" unbiased estimators inside exact
  MCMC (the pseudo-marginal fallback if frozen-probe bias ever bites).

## 6. Decision gates — verdict

- **Headline gate NOT met on CPU.** Blocked Gibbs is 3.5–6× faster in ρ-ESS/sec
  than the best native-NUTS arm across n=2.5k–50k. Do **not** make native
  blackjax NUTS the default Gaussian-SAR path on CPU, and do not claim CPU
  throughput parity. The native path ships as a correct, validated, reusable
  prototype (it matches the Gibbs posterior and is the substrate for the GPU
  question), not as the recommended CPU sampler.
- **What is validated and worth keeping regardless:**
  - `_resolvent` + the numpy gradient factory (impacts at scale without O(n³)
    eigendecomposition; nutpie-custom / MALA gradients) — tested, matched-pair
    correct.
  - The jax SLQ bug fix (a real defect: dense densification + wrong weights).
  - The surrogate-logdet-for-NUTS result: chol-cheb/`_resolvent` is 2× faster
    inside the leapfrog than the eigenvalue logdet — the right ingredient for
    *any* future gradient-based spatial sampler.
- **The one thing that would change the verdict: GPU.** Re-run Experiment B on a
  GPU (blackjax `vmap` over chains; larger n). If SAR-NUTS ESS/sec there reaches
  or beats Gibbs, wire the native path into `fit` dispatch as the GPU fast path
  and do the impacts refactor. Until measured, this is the sole open gate.
- Secondary findings: SLQ frozen-probe bias is small at feasible p (no pivot to
  CG-resolvent MALA / pseudo-marginal needed in the planar regime); the
  SAR↔OLS-NUTS gap is geometry (ρ reparameterisation), not logdet cost.

## 7. Realised win — impacts without eigendecomposition (shipped)

The "one object, three consumers" unification (§2) is now realised for impacts.
The two direct-effect trace quantities are exactly the logdet gradient:

* average direct multiplier  `(1/n) tr((I−ρW)⁻¹) = 1 − (ρ/n)·g(ρ)`
* SDM cross direct           `(1/n) tr((I−ρW)⁻¹W) = −g(ρ)/n`

Previously both were built from the full spectrum via `_chunked_eig_means(rho,
self._W_eigs)`, which triggered the O(n³) dense eigendecomposition **even for
row-standardised W** (where the total-effect row sum already had the closed form
`1/(1−ρ)`). They now ride the model's fast logdet gradient
(`make_logdet_grad_numpy_vec_fn` → `_batch_mean_diag` / `_batch_mean_diag_MW`),
so a chol-cheb / AAA model computes impacts with **no eigendecomposition at all**.

Verified: resolvent traces match the eigenvalue reference to ~1e-5 (chol-cheb
gradient accuracy) cross-section and ~7e-5 panel; exact to machine precision when
the resolved method is `eigenvalue`. A row-standardised chol-cheb SAR computes
`spatial_effects()` with `np.linalg.eig`/`eigvals` poisoned to raise — it never
touches them. Wired across cross-section SAR/SDM (`models/base.py`) and panel
FE/RE/dynamic (`panel_base.py`, `panel/_re.py`, `panel/_fe.py`,
`panel/_dynamic.py`). Non-row-standardised *total* effects still use the
eigenvector bilinear form (`1'S1` is not a trace, so no logdet-gradient identity
applies) — the remaining, and rare, eigen dependency.

**Nonlinear SAR-GLMs (follow-up, same day).** The link-scale (log-odds / log-mean)
direct effect for the SAR-GLMs is the *same* trace `1 − (ρ/n)·g(ρ)`, so `SARLogit`,
`SARNegBin`, `SARNegBinStructural`, and `SARZINB` (count equation) now inherit
`_batch_mean_diag` in place of `_chunked_eig_means(rho, _W_eigs)`. ZINB's selection
equation, which may carry a distinct `W_sel`, gets its own resolvent evaluator
(`_sel_logdet_grad_numpy_vec_fn` / `_sel_batch_mean_diag`, reusing the count helper
when `W_sel is W`); the dead `_W_sel_eigs` was dropped. The **probability/count-scale**
response impacts are deliberately untouched: those weight the resolvent *diagonal*
per observation by the link derivative (Bernoulli variance / `μ`), which is not a
trace, so they keep the eigendecomp (small n) / sparse-Hutchinson (large n) path.
Regression: `tests/test_diagnostics/test_nonlinear_impacts_resolvent.py` (7 tests).

## 8. Panel Gaussian — NUTS vs Gibbs (Experiment C, `panel_shootout.csv`)

The cross-section verdict (Gibbs ≫ NUTS on CPU) does not obviously carry to
panels: unit random effects (RE) / many fixed effects (FE) make the posterior
high-dimensional, the regime where gradient NUTS is often the better mixer — and
the chol-cheb logdet keeps its per-leapfrog cost low (the pytensor logdet is
differentiable, so PyMC NUTS already uses it). `panel_nuts_study.py` benchmarks
SAR panel FE and RE both ways.

Full run (SAR panel, T=10, 1500 draws × 4 chains, PyMC NUTS with chol-cheb):

| model | n_obs | Gibbs ρ-ESS/s | NUTS ρ-ESS/s | Gibbs β-ESS/s | NUTS β-ESS/s | Gibbs ρ̄ | NUTS ρ̄ |
|---|---|---|---|---|---|---|---|
| FE | 1000 | 2509 | 123 | 2466 | 115 | **0.194** ✗ | 0.491 ✓ |
| FE | 2250 | 2988 | 129 | 2796 | 134 | **0.205** ✗ | 0.521 ✓ |
| RE | 1000 | 1099 | 505 | 320 | 281 | 0.491 ✓ | 0.495 ✓ |
| RE | 2250 | 1041 | 162 | 233 |  79 | 0.523 ✓ | 0.527 ✓ |

Two findings:

**(1) ESS/sec: Gibbs still wins on CPU — but the gap is PyMC overhead, not NUTS.**
NUTS wall-time is 20–45 s vs Gibbs 2–6 s, yet NUTS's *per-draw* mixing is
excellent (ρ-ESS ≈ draws; at RE n=1000 raw ρ-ESS 10.5k > Gibbs 6.4k). The
throughput gap is PyMC/pytensor NUTS fixed cost (graph compile + C backend +
multiprocess) — the same "current-path overhead" the cross-section study flagged.
A **native panel NUTS** (the blackjax analog of `gaussian/_blackjax.py`, not yet
built) is the real test; PyMC-routed NUTS is dominated by fixed overhead here. So
"panels sample faster with NUTS" is **not confirmed on CPU via PyMC**, and stays
open pending a native panel path (and GPU).

**(2) 🐞→✅ Bug discovered and FIXED — `SARPanelFE` JAX-Gibbs ρ double-counted T.**
FE Gibbs reported ρ≈0.19–0.21 (T=10) and **0.137** on a clean N=100/T=15 panel
(tight CI [0.114, 0.159] excluding true ρ=0.5), worsening with T; FE NUTS gave
ρ=0.475 ✓. Root cause (JAX backend only): the panel passes the NT×NT block-diagonal
lag `I_T ⊗ W` as the sampler's `W_sparse`, whose logdet *already* equals
`T·log|I_N−ρW|`; `_build_logdet_jax` then multiplied by `T=self.T` again →
**T²·log**, an over-strong Jacobian pushing ρ→0 (T² vs T ⇒ worse with T). The numpy
backend used the model's N×N `W` and was always correct; `SARPanelRE` only supports
numpy; the eigenvalue method used length-N `W_eigs` — so only FE with `auto`→JAX and
a non-eigenvalue logdet was hit. **Fix**: extract the per-period block `W[:N,:N]` in
`samplers/gaussian/_estimation.py::_build_logdet_jax` (no-op for cross-section T=1),
so `T=self.T` applies once. Verified: FE-jax ρ now 0.519 = FE-numpy = truth.
Regression `tests/test_samplers/test_panel_fe_jax_logdet.py` pins the JAX logdet to
T×exact. (The existing slow recovery test missed it — tol 0.25 too loose, the biased
estimate landed just inside the band.) *Aside:* this is the strongest illustration of
why a trustworthy gradient path matters — NUTS was right where Gibbs was silently
wrong.
