# Architecture

This page describes how `neighbayes` is put together and why. It is aimed at
contributors, and at users who want to know what the library is doing on their behalf
before they trust a posterior. For the models themselves see
[Supported Models](models), and for measured numbers see
[Performance](performance/index).

## Design goals

The package is organised around four commitments.

**One implementation of each hard thing.** The log-determinant, the spatial lag, the
effects decomposition, and the diagnostics battery are each written once and shared by
every model that needs them. A new model should be a declaration of its structure, not a
new copy of the machinery.

**Exact by default, approximate on request.** Where an exact computation is affordable,
the defaults take it. Stochastic estimators are available, but a user has to ask for
them — noise in the log-density is a property of the posterior you sample, not an
implementation detail to be chosen silently on a user's behalf.

**Automatic choices are overridable.** The library picks a log-determinant method, a
sampler, a sparse backend, and a set of priors for you. Every one of them can be
replaced by an explicit argument — `logdet_method=`, `sampler=`, `gibbs_backend=`,
`priors=` — and the resolved log-determinant method is recorded on the model as
`_resolved_logdet_method`. Nothing is chosen in a way you cannot see or change.

**No bespoke result objects.** `fit()` returns an `arviz.InferenceData`. The posterior
belongs to the wider PyData ecosystem, not to this package.

## The layers

```
models/          SAR, SDM, panel FE, flow …  — thin, declarative
  _base/         shared methods + structure strategies
  _mixins/       likelihood-specific model building
  priors.py      frozen dataclasses, inheritance mirrors the models
samplers/        Gibbs runners, one per (likelihood, structure)
  _registry.py   dispatch: (likelihood, structure) -> runner
_logdet/         log|I - rho W|: methods, factories, auto-selection
_ops/            sparse/dense linear algebra, Kronecker helpers, backends
diagnostics/     LM tests, decision trees, effects, Bayes factors, spatial CV
dgp/             a simulator for every model
```

Dependencies run downward. `_logdet` and `_ops` know nothing about models; the models
know nothing about which sampler will be used until `fit()` is called.

## Models are declarations

A model class carries almost no code. It states what it *is* through class attributes,
and inherits the behaviour from `SharedSpatialMethods` in
[`models/_base/_shared.py`](https://github.com/pysal/neighbayes/blob/main/neighbayes/models/_base/_shared.py).
The whole of `SAR`'s structural definition is:

```python
class SAR(GaussianLikelihoodMixin, SpatialModel):
    _priors_cls = SARPriors
    _spatial_params = ("rho",)        # autoregressive parameters
    _lag_terms = ("Wy",)              # which lags appear in the specification
    _jacobian_param = "rho"           # the parameter inside log|I - .W|, or None
    _has_wx_in_beta = False           # is the design matrix [X, WX] or just X?
    _model_type = "sar"
    _likelihood = "gaussian"
    _gibbs_key = ("gaussian", "cross_section")
```

Those attributes are what the shared machinery reads. `_jacobian_param` tells the base
class whether to build a log-determinant at all and which parameter to feed it —
`"rho"` for SAR and SDM, `"lam"` for SEM and SDEM, `None` for OLS and SLX, which
therefore skip the Jacobian entirely. `_lag_terms` and `_has_wx_in_beta` drive design
matrix assembly and coefficient labelling. `_spatial_params` drives which diagnostics
and effects decompositions apply.

The payoff is that the differences between the fifty-odd model classes are legible in a
few lines each, and a bug fixed in the shared path is fixed everywhere at once. The cost
is that adding a genuinely novel structure means extending the shared machinery rather
than writing a self-contained class, which is the right trade for a package whose models
are variations on a small number of themes.

### Cross-section and panel share one implementation

Cross-sectional and panel models differ in only a few operations: what the spatial lag
means, what operand the log-determinant factory receives, and what the symbolic sparse
operator looks like inside a PyMC model. Those three are isolated behind the
`SpatialStructure` strategy in
[`models/_base/_structure.py`](https://github.com/pysal/neighbayes/blob/main/neighbayes/models/_base/_structure.py):

:::{list-table}
:header-rows: 1
:widths: 22 39 39

* - operation
  - `CrossSectionStructure`
  - `PanelStructure`
* - `spatial_lag(x)`
  - `W @ x`
  - `W` applied per period via the Kronecker structure
* - `logdet_W_operand()`
  - dense `W`
  - sparse `W` (the panel logdet path stays sparse)
* - `W_pt_sparse()`
  - the `N×N` `W`
  - the `(N·T)×(N·T)` block `I_T ⊗ W`, so one symbolic multiply lags a stacked panel
:::

Everything that is behaviour-identical across the two lives in the shared base. This is
why panel models inherit the diagnostics, effects, and prior handling of their
cross-sectional counterparts without re-implementation.

### Priors mirror model inheritance

Priors are frozen dataclasses in
[`models/priors.py`](https://github.com/pysal/neighbayes/blob/main/neighbayes/models/priors.py),
and their inheritance graph tracks the models': `SDMPriors` extends `SARPriors`,
`SDEMPriors` extends `SEMPriors`, and `SARNegBinPriors` extends both `SARPriors` and
`NegBinPriors`. A model declares its prior class as `_priors_cls` and users override
individual hyperparameters through the `priors=` dict. Because the dataclasses are
frozen and typed, an unrecognised prior key is an error rather than a silently ignored
keyword.

## The log-determinant subsystem

The Jacobian $\log|I - \rho W|$ is evaluated at every $\rho$ the sampler proposes, which
makes it the dominant cost in a spatial fit and the reason `_logdet` is a subsystem
rather than a function. Three design decisions shape it.

**Precompute once, evaluate many times.** Every method splits into a `*_precompute`
step that does the expensive linear algebra and an `*_eval` step that answers a single
$\rho$. The sampler pays the setup once per model and then evaluates in microseconds.
This is what makes exact interpolation competitive: `cheb_cholesky` factorises at a
handful of Chebyshev nodes and then costs ~1.3 μs per $\rho$ via Clenshaw recurrence;
`aaa` factorises on an adaptive coarse grid, selects ~7 support points, and costs ~5 μs.
Both are accurate to near machine precision, and both need far fewer factorisations than
the common practice of factorising a dense grid and splining between the results.

**One term, several computational dialects.** The same log-determinant has to be
available as a plain Python float (for the NumPy Gibbs sampler), as a vectorised array
operation, as a gradient, as a PyTensor symbolic expression (for PyMC and NUTS), and
under JAX. The factories in
[`_logdet/_factories.py`](https://github.com/pysal/neighbayes/blob/main/neighbayes/_logdet/_factories.py)
emit each of these from the same precomputed data:

:::{list-table}
:header-rows: 1
:widths: 45 55

* - factory
  - returns
* - `make_logdet_numpy_fn`
  - `(rho: float) -> float`
* - `make_logdet_numpy_vec_fn`
  - `(rho_arr) -> ndarray`
* - `make_logdet_grad_numpy_fn`
  - `(rho: float) -> float`, the derivative
* - `make_logdet_fn`
  - `(rho) -> pytensor` symbolic expression
:::

The gradient factory matters beyond sampling: the direct-effects computation rides the
resolvent identity $\operatorname{tr}(S)/n = 1 - (\rho/n)\,g$ with
$g = \tfrac{d}{d\rho}\log|I-\rho W|$ and $S = (I-\rho W)^{-1}$, so the effects
decomposition needs no $O(n^3)$ eigendecomposition either.

Built callables are cached by a signature of $W$ and the resolved bounds
(`get_cached_logdet_fn`), so refitting a model, or fitting several models on the same
graph, reuses the setup.

**Auto-selection from the shape of the problem.** `resolve_logdet_method` picks from the
size and symmetry of $W$: exact eigendecomposition below $n=500$; `cheb_cholesky` for
symmetric $W$ (undirected graphs) up to $n=20{,}000$; `aaa` for non-symmetric $W$ (KNN,
travel time, migration flows) in the same range; a stochastic method above it, where
factorisation fill-in stops being affordable. The symmetry test uses
`Graph.asymmetry(intrinsic=False)` for libpysal graphs and never densifies a sparse
matrix to check. Both cutoffs are environment variables, and `logdet_method=` overrides
the choice outright.

## Sampler dispatch

Sampler selection is a module-level dict keyed by `(likelihood, structure)` in
[`samplers/_registry.py`](https://github.com/pysal/neighbayes/blob/main/neighbayes/samplers/_registry.py)
— no plugin system and no entry points, because three structures times seven likelihoods
is well below the size at which anything fancier pays for itself. A model names its key
as `_gibbs_key`; a model with no entry simply has no Gibbs sampler, and `fit()` routes
it to NUTS.

Each entry is a `GibbsEntry` recording the runner plus what the base `fit()` needs to
know about it:

`run`
: The callable, which takes the model and resolved controls and returns a **finished**
  `InferenceData`. Returning finished output rather than raw chains is deliberate: it
  lets class-based families keep their own `fit` and function-based families keep
  `run_chains` + `gibbs_to_inference_data`. The registry unifies dispatch, not sampler
  internals.

`backends` / `auto_backend`
: Which of `{"jax", "numpy"}` the family supports, and which one `"auto"` should prefer.
  The Gaussian families prefer JAX, where a vmapped Gibbs is the fast path. The
  Pólya–Gamma families pin `auto` to NumPy, because there the CHOLMOD `factorize` path
  is fastest and the dense JAX path is an opt-in for GPU work.

`options`
: The family-specific `fit` keywords this runner accepts, e.g. `slice_width`,
  `krylov_degree`. The base `fit()` pops exactly these and raises on anything left over,
  replacing a silent `**kwargs` swallow — a misspelled tuning parameter is an error, not
  a setting that quietly did nothing.

`supports_robust`
: Whether the family handles a Student-t likelihood, so the base class can raise one
  clear error instead of each runner re-checking.

### Two samplers behind one object

Most models are estimated by a blocked Gibbs sampler written for this posterior:
conjugate draws for $\beta$ and $\sigma^2$, and $\rho$ updated by slice sampling the
collapsed log-density with $\beta$ and $\sigma^2$ integrated out. Collapsing is what
makes the sampler robust to the strong $\rho$–$\beta$–$\sigma^2$ correlation that
frustrates generic samplers, and it is also why per-$\rho$ log-determinant cost is the
thing worth optimising.

Models without a registered Gibbs sampler are built as PyMC models and sampled with
NUTS. Both paths are reachable from the same object through `fit(sampler=...)`, which
exists so that a Gibbs result can be checked against an independent NUTS result on
identical data — the check the package's own validation notebooks run.

## Backends are a runtime choice, not a fork

There is one model code path. Acceleration is selected at run time and degrades to a
working default rather than an error:

- **JAX** is optional. `gibbs_backend="auto"` uses it when it is installed and the
  family prefers it, NumPy otherwise. Requesting `"jax"` explicitly without JAX
  installed raises, because a silent downgrade of an explicit request would be
  misleading.
- **Sparse solves** prefer KLU, then UMFPACK (both from `scikit-sparse`), then SciPy's
  SuperLU, with a one-time advisory warning when it falls back so the performance loss
  is visible.
- **A dense LAPACK fast path** handles small problems: below `NEIGHBAYES_KRON_DENSE_MAX`
  (512), `lu_factor` on a dense $I - \rho W$ beats `splu`, because SuperLU spends most of
  its time in symbolic-factorisation overhead at that size while `dgetrf` is a single
  BLAS-3 kernel. One factorisation serves both the forward and adjoint solves.

Heavy dependencies load lazily. PyMC and ArviZ together cost roughly three seconds to
import and are needed only when a model is *fit*, never when a model class is
*imported*, so they are deferred behind proxies in `_lazy_deps.py` following
[SPEC 1](https://scientific-python.org/specs/spec-0001/). JAX imports stay inside
function bodies for the same reason.

## Every model has a simulator

`neighbayes.dgp` mirrors `neighbayes.models`: each model has a generator that samples
from its data-generating process. This is a deliberate testing strategy rather than a
convenience. Bayesian spatial models have no closed-form answer to check against, so
correctness is established by simulating data with known parameters and confirming the
posterior recovers them, across the whole catalog. The same generators are what make the
documentation reproducible, and what users should reach for when they want to know
whether a specification can identify the effect they care about at their sample size.

## Diagnostics are part of the model, not an add-on

Every fitted model exposes `spatial_diagnostics()`, which runs the battery of Bayesian LM
tests appropriate to its specification — lag, error, WX, joint, and the Neyman-orthogonal
robust variants — defined separately for cross-sectional, GLM, panel, and flow models and
resolved through a registry of test suites. `spatial_diagnostics_decision()` walks those
results through a decision tree and reports both the favoured model and the path it took
to reach it, so the recommendation can be audited rather than taken on faith.

`spatial_effects()` decomposes a change in $X$ through the spatial multiplier
$(I - \rho W)^{-1}$ into direct, indirect, and total effects, computed from the posterior
draws. These are distributions, not delta-method approximations — which is much of the
argument for estimating these models in a Bayesian framework in the first place.

## Configuration

Runtime knobs are environment variables so they can be set for a session or a job
without touching model code:

:::{list-table}
:header-rows: 1
:widths: 42 58

* - variable
  - effect
* - `NEIGHBAYES_LOGDET_EIGEN_MAX_N`
  - largest `n` auto-routed to exact eigendecomposition (default 500)
* - `NEIGHBAYES_LOGDET_CHEB_MAX_N`
  - largest `n` auto-routed to an exact interpolating method (default 60000)
* - `NEIGHBAYES_SPARSE_BACKEND`
  - force a sparse solver instead of auto-detecting
* - `NEIGHBAYES_SPARSE_STRICT`
  - raise instead of falling back when the requested backend is unavailable
* - `NEIGHBAYES_KRON_DENSE_MAX`
  - largest `n` using dense LAPACK over SuperLU (default 512)
* - `NEIGHBAYES_OP_INSTRUMENT`
  - record per-operation timings for profiling
:::

All of these are read at the point of use except `NEIGHBAYES_OP_INSTRUMENT`, which is
snapshotted at import time in `_config.py`.
