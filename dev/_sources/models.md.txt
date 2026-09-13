# Supported Models


## Model suite

The package organizes models along three column dimensions — **likelihood** (linear / non-linear), **temporal structure** (cross-section / panel), and **outcome structure** (single / flow). Each cell lists the spatial structures implemented for that combination.

| | Linear · Cross-section | Linear · Panel | Non-linear · Cross-section | Non-linear · Panel |
|---|---|---|---|---|
| **Single** | Aspatial, SLX, SAR, SEM, SDM, SDEM | Aspatial, SLX, SAR, SEM, SDM, SDEM | Aspatial, SAR, SEM, SDM | SAR, SEM |
| **Flow** | Aspatial, SAR, SEM | Aspatial, SAR, SEM | Aspatial, SAR | Aspatial, SAR |

Panel models come in fixed-effects and random-effects variants, and the linear
panel family additionally has dynamic (lagged-dependent-variable) forms. Every
spatial flow model has a **separable** counterpart that pins
$\rho_w = -\rho_d \rho_o$; the non-linear flow cell covers both negative
binomial and Poisson observation models. The sections below list every class
individually.

## Cross Sectional Models

### OLS

$$y = X\beta + \epsilon$$

### SLX

$$y = X\beta + WX\theta + \epsilon$$

### SAR

$$y = \rho Wy + X\beta + \epsilon$$

### SEM

$$y = X\beta + u, \quad u = \lambda Wu + \epsilon$$

### SDM

$$y = \rho Wy + X\beta + WX\theta + \epsilon$$

### SDEM

$$y = X\beta + WX\theta + u, \quad u = \lambda Wu + \epsilon$$

## Panel Models

### OLS panel

$$y_{it} = x_{it}' \beta + a_i + \tau_t + \epsilon_{it}$$

### SAR panel

$$y_{it} = \rho Wy_{it} + x_{it}' \beta + a_i + \tau_t + \epsilon_{it}$$

### SEM panel

$$y_{it} = x_{it}' \beta + a_i + \tau_t + u_{it}, \quad u_{it} = \lambda Wu_{it} + \epsilon_{it}$$

### SDM panel

$$y_{it} = \rho Wy_{it} + x_{it}' \beta + Wx_{it}' \theta + a_i + \tau_t + \epsilon_{it}$$

### SDEM panel

$$y_{it} = x_{it}' \beta + Wx_{it}' \theta + a_i + \tau_t + u_{it}, \quad u_{it} = \lambda Wu_{it} + \epsilon_{it}$$

### SLX panel

$$y_{it} = x_{it}' \beta + Wx_{it}' \theta + a_i + \tau_t + \epsilon_{it}$$

### OLS panel (Random Effects)

$$y_{it} = x_{it}' \beta + \alpha_i + \tau_t + \epsilon_{it}, \quad \alpha_i \sim N(0, \sigma_\alpha^2)$$

### SAR panel (Random Effects)

$$y_{it} = \rho W y_{it} + x_{it}' \beta + \alpha_i + \tau_t + \epsilon_{it}, \quad \alpha_i \sim N(0, \sigma_\alpha^2)$$

### SEM panel (Random Effects)

$$y_{it} = x_{it}' \beta + \alpha_i + \tau_t + u_{it}, \quad u_{it} = \lambda W u_{it} + \epsilon_{it}, \quad \alpha_i \sim N(0, \sigma_\alpha^2)$$

### SDEM panel (Random Effects)

$$y_{it} = x_{it}' \beta + W x_{it}' \theta + \alpha_i + u_{it}, \quad u_{it} = \lambda W u_{it} + \epsilon_{it}, \quad \alpha_i \sim N(0, \sigma_\alpha^2)$$

## Dynamic Panel Models

### OLSPanelDynamic (Dynamic Linear Model)

$$y_{it} = \phi y_{i,t-1} + x_{it}' \beta + a_i + \tau_t + \epsilon_{it}$$

### SDMRPanelDynamic (Dynamic Restricted Spatial Durbin)

$$y_{it} = \phi y_{i,t-1} + \rho W y_{it} - \rho \phi W y_{i,t-1} + x_{it}' \beta + W x_{it}' \theta + a_i + \tau_t + \epsilon_{it}$$

### SDMUPanelDynamic (Dynamic Unrestricted Spatial Durbin)

$$y_{it} = \phi y_{i,t-1} + \rho W y_{it} + \theta W y_{i,t-1} + x_{it}' \beta + W x_{it}' \theta + a_i + \tau_t + \epsilon_{it}$$

### SARPanelDynamic (Dynamic SAR)

$$y_{it} = \phi y_{i,t-1} + \rho W y_{it} + x_{it}' \beta + a_i + \tau_t + \epsilon_{it}$$

### SEMPanelDynamic (Dynamic SEM)

$$y_{it} = \phi y_{i,t-1} + x_{it}' \beta + a_i + \tau_t + u_{it}, \quad u_{it} = \lambda W u_{it} + \epsilon_{it}$$

### SDEMPanelDynamic (Dynamic SDEM)

$$y_{it} = \phi y_{i,t-1} + x_{it}' \beta + W x_{it}' \theta + a_i + \tau_t + u_{it}, \quad u_{it} = \lambda W u_{it} + \epsilon_{it}$$

### SLXPanelDynamic (Dynamic SLX)

$$y_{it} = \phi y_{i,t-1} + x_{it}' \beta + W x_{it}' \theta + a_i + \tau_t + \epsilon_{it}$$

## Non-Linear Models

### SARProbit

$$y^* = \rho W y^* + X\beta + a + \varepsilon, \quad \varepsilon \sim \mathcal{N}(0, I), \quad y_i = \mathbf{1}[y_i^* > 0]$$

Note the $a$ term: this is a **region-random-effects** specification, in which
$\rho$ acts on region-level latent utilities and observations are nested within
regions via `region_ids`. It is not the standard spatial probit of the LeSage
toolbox. For spatial binary outcomes prefer the Pólya–Gamma logit classes
(`SARLogit`, `SARLogitStructural`, `SEMLogit`), which are conjugate and have a
Gibbs sampler.

### Tobit (SAR Tobit)

$$y_i = \max(c, y_i^*), \quad y^* = \rho W y^* + X\beta + \varepsilon, \quad \varepsilon \sim \mathcal{N}(0, \sigma^2 I)$$

### Tobit (SEM Tobit)

$$y_i = \max(c, y_i^*), \quad y^* = X\beta + u, \quad u = \lambda Wu + \varepsilon, \quad \varepsilon \sim \mathcal{N}(0, \sigma^2 I)$$

### Tobit (SDM Tobit)

$$y_i = \max(c, y_i^*), \quad y^* = \rho W y^* + X\beta + WX\theta + \varepsilon, \quad \varepsilon \sim \mathcal{N}(0, \sigma^2 I)$$

### Panel Tobit (SAR)

$$y_{it} = \max(c, y_{it}^*), \quad y_t^* = \rho W y_t^* + X_t\beta + \varepsilon_t$$

### Panel Tobit (SEM)

$$y_{it} = \max(c, y_{it}^*), \quad y_t^* = X_t\beta + u_t, \quad u_t = \lambda W u_t + \varepsilon_t$$

### SARNegBin (Reduced Form)

$$y_i \sim \operatorname{NegBin}(\mu_i, \alpha), \quad \mu = \exp(\eta), \quad \eta = (I - \rho W)^{-1} X\beta$$

No latent $\sigma$ — spatial dependence enters only through the mean propagator. Supports both NUTS and Pólya–Gamma Gibbs sampling.

### SARNegBinStructural (Structural Form)

$$y_i \sim \operatorname{NegBin}(\mu_i, \alpha), \quad \eta = \rho W \eta + X\beta + \nu, \quad \nu \sim \mathcal{N}(0, \sigma^2 I)$$

Includes latent $\sigma$ — structural form with explicit noise. Gibbs sampling only (PG augmentation).

### SARZINB

$$y_i \sim \operatorname{ZINB}(\mu_i, \alpha, \pi), \quad \mu = \exp(\eta), \quad \eta = (I - \rho W)^{-1} X\beta$$

Zero-inflated negative binomial with spatial lag on the log-mean.

### Logit

$$y_i \sim \operatorname{Bernoulli}(p_i), \quad \operatorname{logit}(p) = X\beta$$

Non-spatial logistic regression baseline.

### NegBin

$$y_i \sim \operatorname{NegBin}(\mu_i, \alpha), \quad \log \boldsymbol{\mu} = X\beta$$

Non-spatial negative binomial baseline.

### SARLogit (Reduced Form)

$$y_i \sim \operatorname{Bernoulli}(p_i), \quad \eta = (I - \rho W)^{-1}(X\beta + \nu), \quad \nu \sim \mathcal{N}(0, I)$$

Spatial lag on the latent log-odds, written with the multiplier applied to the
mean. Pólya–Gamma Gibbs sampler only — no NUTS path.

### SARLogitStructural (Structural Form)

$$y_i \sim \operatorname{Bernoulli}(p_i), \quad \eta = \rho W \eta + X\beta + \nu, \quad \nu \sim \mathcal{N}(0, I)$$

The same model as `SARLogit`, parameterised without inverting $(I - \rho W)$.
Pólya–Gamma Gibbs sampler only.

### SEMLogit

$$y_i \sim \operatorname{Bernoulli}(p_i), \quad \eta = X\beta + u, \quad u = \lambda W u + \nu, \quad \nu \sim \mathcal{N}(0, I)$$

Spatial error on the latent log-odds. The logit link fixes $\sigma^2 = 1$, so it
does not appear in the posterior. Pólya–Gamma Gibbs sampler only.

## Flow Models

Vectorize the origin-destination flow matrix to $y \in \mathbb{R}^{N}$ with $N = n^2$, and define destination, origin, and network weight matrices as $W_d$, $W_o$, and $W_w$.

### OLSFlow

$$y = X\beta + \varepsilon$$

### NegBinFlow

$$y_{ij} \sim \operatorname{NegBin}(\mu_{ij}, \alpha), \quad \log \boldsymbol{\mu} = X\beta$$

### SARFlow

$$y = \rho_d W_d y + \rho_o W_o y + \rho_w W_w y + X\beta + \varepsilon$$

### SARFlowSeparable

$$y = \rho_d W_d y + \rho_o W_o y - \rho_d \rho_o W_w y + X\beta + \varepsilon$$

### SARNegBinFlow

$$y_{ij} \sim \operatorname{NegBin}(\mu_{ij}, \alpha), \quad \log \boldsymbol{\mu} = A(\boldsymbol{\rho})^{-1} X\beta$$

### SARNegBinFlowSeparable

$$y_{ij} \sim \operatorname{NegBin}(\mu_{ij}, \alpha), \quad \log \boldsymbol{\mu} = A(\boldsymbol{\rho})^{-1} X\beta, \quad \rho_w = -\rho_d \rho_o$$

### SARPoissonFlow

$$y_{ij} \sim \operatorname{Poisson}(\mu_{ij}), \quad \log \boldsymbol{\mu} = A(\boldsymbol{\rho})^{-1} X\beta$$

No dispersion parameter. Sampled by auxiliary-mixture Gibbs
(Frühwirth-Schnatter & Wagner 2006) rather than Pólya–Gamma, which admits no
exact Poisson representation.

### SARPoissonFlowSeparable

$$y_{ij} \sim \operatorname{Poisson}(\mu_{ij}), \quad \log \boldsymbol{\mu} = A(\boldsymbol{\rho})^{-1} X\beta, \quad \rho_w = -\rho_d \rho_o$$

The recommended Poisson flow model — the separable restriction removes the
weakly-identified $\rho$ ridge of the unrestricted variant.

### SEMFlow

$$y = X\beta + u, \quad u = \lambda_d W_d u + \lambda_o W_o u + \lambda_w W_w u + \varepsilon, \quad \varepsilon \sim \mathcal{N}(0, \sigma^2 I)$$

### SEMFlowSeparable

$$y = X\beta + u, \quad u = \lambda_d W_d u + \lambda_o W_o u - \lambda_d \lambda_o W_w u + \varepsilon, \quad \varepsilon \sim \mathcal{N}(0, \sigma^2 I)$$

## Panel Flow Models

Stack the flow models above across $T$ periods in time-first order. The NB panel variants currently operate in pooled mode.

### OLSFlowPanel

$$y_t = X_t\beta + \varepsilon_t, \quad \varepsilon_t \sim \mathcal{N}(0, \sigma^2 I_N)$$

### NegBinFlowPanel

$$y_{ij,t} \sim \operatorname{NegBin}(\mu_{ij,t}, \alpha), \quad \log \boldsymbol{\mu}_t = X_t\beta$$

### SARFlowPanel

$$y_t = \rho_d W_d y_t + \rho_o W_o y_t + \rho_w W_w y_t + X_t\beta + \varepsilon_t$$

### SARFlowSeparablePanel

$$y_t = \rho_d W_d y_t + \rho_o W_o y_t - \rho_d \rho_o W_w y_t + X_t\beta + \varepsilon_t$$

### SARNegBinFlowPanel

$$y_{ij,t} \sim \operatorname{NegBin}(\mu_{ij,t}, \alpha), \quad \log \boldsymbol{\mu}_t = A(\boldsymbol{\rho})^{-1} X_t\beta$$

### SARNegBinFlowSeparablePanel

$$y_{ij,t} \sim \operatorname{NegBin}(\mu_{ij,t}, \alpha), \quad \log \boldsymbol{\mu}_t = A(\boldsymbol{\rho})^{-1} X_t\beta, \quad \rho_w = -\rho_d \rho_o$$

### SEMFlowPanel

$$y_t = X_t\beta + u_t, \quad u_t = \lambda_d W_d u_t + \lambda_o W_o u_t + \lambda_w W_w u_t + \varepsilon_t, \quad \varepsilon_t \sim \mathcal{N}(0, \sigma^2 I_N)$$

### SEMFlowSeparablePanel

$$y_t = X_t\beta + u_t, \quad u_t = \lambda_d W_d u_t + \lambda_o W_o u_t - \lambda_d \lambda_o W_w u_t + \varepsilon_t, \quad \varepsilon_t \sim \mathcal{N}(0, \sigma^2 I_N)$$

## Specification Tests

Lagrange-Multiplier tests for choosing a spatial specification. Each statistic
is evaluated at every posterior draw, giving a posterior distribution rather
than a point estimate. Call them directly from
`neighbayes.diagnostics.lmtests`, or through `spatial_diagnostics()` and
`spatial_diagnostics_decision()` on any fitted model.

| Test | $H_0$ | Alternative | df | Null model |
|------|-------|-------------|----|----|
| LM-Lag | $\rho = 0$ | SAR | 1 | OLS |
| LM-Error | $\lambda = 0$ | SEM | 1 | OLS |
| LM-WX | $\gamma = 0$ | SLX | $k_{wx}$ | SAR |
| LM-SDM (joint) | $\rho = \gamma = 0$ | SDM | $1 + k_{wx}$ | OLS |
| LM-SLX-Error (joint) | $\lambda = \gamma = 0$ | SDEM | $1 + k_{wx}$ | OLS |
| LM-WX-SEM | $\gamma = 0$ in SEM | SDEM | $k_{wx}$ | SEM |
| LM-Error-SDM | $\lambda = 0$ in SDM | SDARAR | 1 | SDM |
| LM-Lag-SDEM | $\rho = 0$ in SDEM | SDARAR | 1 | SDEM |
| Robust LM-Lag | $\rho = 0$, robust to $\lambda$ | SAR vs SEM | 1 | OLS |
| Robust LM-Error | $\lambda = 0$, robust to $\rho$ | SEM vs SAR | 1 | OLS |
| Robust LM-Lag-SDM | $\rho = 0$, robust to $\gamma$ | SDM | 1 | SLX |
| Robust LM-WX | $\gamma = 0$, robust to $\rho$ | SDM | $k_{wx}$ | SAR |
| Robust LM-Error-SDEM | $\lambda = 0$, robust to $\gamma$ | SDEM | 1 | SLX |

The robust variants use the **Neyman orthogonal score** of Doğan, Taşpınar &
Bera (2021), which removes the correlation between the test score and the
nuisance score:

$$g_\psi^* = g_\psi - J_{\psi\phi \cdot \sigma}\,J_{\phi\phi\cdot\sigma}^{-1}\,g_\phi.$$

The same machinery extends to balanced panels — with a $T$ multiplier on the
information matrix, under the `bayesian_panel_lm_*` prefix — and to
origin–destination flow models on Kronecker weights $W_d$, $W_o$, $W_w$, under
`bayesian_lm_flow_*`.

### Sources

- Doğan, O., Taşpınar, S., Bera, A.K. (2021). "A Bayesian robust chi-squared
  test for testing simple hypotheses." *Journal of Econometrics*, 222(2),
  933–958.
- Koley, M., Bera, A.K. (2024). "To Use, or Not to Use the Spatial Durbin
  Model? – That Is the Question." *Spatial Economic Analysis*, 19(1), 30–56.
- Bera, A.K., Yoon, M.J. (1993). "Specification testing with locally
  misspecified alternatives." *Econometric Theory*, 9(4), 649–658.
- Anselin, L., Bera, A.K., Florax, R., Yoon, M.J. (1996). "Simple diagnostic
  tests for spatial dependence." *Regional Science and Urban Economics*, 26(1),
  77–104.
- LeSage, J.P., Pace, R.K. (2008). "Spatial Econometric Modeling of
  Origin–Destination Flows." *Journal of Regional Science*, 48(5), 941–967.

## Sampling Backends

### Choosing a sampler

`fit()` takes `sampler={"gibbs", "nuts", None}`, and `None` — the default —
**selects Gibbs whenever the model has a registered Gibbs sampler, and NUTS
otherwise.** For SAR/SEM/SDM/SDEM, the count families, the ZINB model and the
Gaussian panel families, that means Gibbs unless you ask for something else.

NUTS is not universally available. The Pólya–Gamma logit classes
(`SARLogit`, `SARLogitStructural`, `SEMLogit`) and the auxiliary-mixture
Poisson flow classes build no PyMC graph at all, and `fit(sampler="nuts")`
raises `NotImplementedError`. Robust (Student-t) models are the mirror case:
no Gibbs sampler supports them, so `robust=True` requires NUTS.

`target_accept` is NUTS-only and raises `TypeError` if passed with Gibbs.

### Execution backends

For any Gibbs sampler, `gibbs_backend` selects the execution path:

| Value | Behaviour |
|---|---|
| `"auto"` | **default** — JAX when installed and supported by the family, else NumPy |
| `"jax"` | the sweep JIT-compiled into one XLA kernel; chains vectorised under `jax.vmap`, controlled by `chain_method` |
| `"numpy"` | pure NumPy/SciPy; chains as separate processes via `joblib`, controlled by `n_jobs` |

Both backends implement the same sampler and target the same posterior.

### Gibbs Sampler (Gaussian models)

Gaussian cross-sectional models (SAR, SEM, SDM, SDEM) and the Gaussian panel
families exploit conditional conjugacy with a 3-block strategy:

| Block | Full conditional | Update |
|---|---|---|
| β \| ρ, σ², y | Normal | Direct draw (conjugate) |
| σ² \| β, ρ, y | Inverse-Gamma | Direct draw (conjugate) |
| ρ/λ \| β, σ², y | 1-D non-conjugate | Adaptive slice sampling |

SAR and SDM update the spatial parameter with β and σ² integrated out
(a collapsed conditional); SEM and SDEM update it conditional on them.

```python
model = SAR(y=y, X=X, W=W)
idata = model.fit(draws=2000, tune=1000, chains=4)   # Gibbs, by default
```

The family accepts two options beyond the shared `fit()` arguments:
`slice_width` (initial slice interval for ρ/λ) and `chain_method` (JAX
backend chain mapping). See the
[Gibbs sampler how-to](how-to/gibbs_sampler.ipynb) for details.

### Gibbs Sampler (SAR Negative Binomial)

`SARNegBin` (reduced form) supports a Pólya–Gamma Gibbs sampler via `sampler="gibbs"` (the default):

```python
model = SARNegBin(y=y_int, X=X, W=W)
idata = model.fit(draws=2000, tune=1000, chains=4)
```

The reduced form has no latent σ² — spatial dependence enters only through the mean propagator $(I - \rho W)^{-1}$:

| Block | Full conditional | Update |
|---|---|---|
| ω \| β, ρ, α, y | Pólya–Gamma | Direct draw (conjugate augmentation) |
| β \| ρ, ω, y | Normal | Direct draw (conjugate, via $\tilde{X} = (I-\rho W)^{-1}X$) |
| ρ \| ω, y | 1-D non-conjugate | Adaptive slice sampling (β marginalised) |
| α \| y, η | 1-D non-conjugate | Slice sampling on log(α) |

`SARNegBinStructural` (structural form) adds latent η and σ² blocks via a separate Gibbs sampler in `neighbayes.samplers.negbin`.

### Gibbs Sampler (NB flow models)

NB flow models (`SARNegBinFlow`, `SARNegBinFlowSeparable`, `NegBinFlow`) support a Pólya–Gamma Gibbs sampler via `sampler="gibbs"`:

```python
model = SARNegBinFlow(y_int, X, G)   # positional: (y, X, W)
idata = model.fit(sampler="gibbs", draws=2000, tune=1000, chains=4)
```

The sampler uses a reduced-form Pólya–Gamma augmentation strategy with no σ² parameter — spatial dependence enters only through the mean propagator $A^{-1}$:

| Block | Full conditional | Update |
|---|---|---|
| ω \| β, α, y | Pólya–Gamma | Direct draw (conjugate augmentation) |
| β \| ρ, ω, y | Normal | Direct draw (conjugate, via $\tilde{X} = A^{-1}X$) |
| ρ \| ω, y | 1-D non-conjugate | Adaptive slice sampling (β marginalised) |
| α \| y, η | 1-D non-conjugate | Slice sampling on log(α) |

For the unrestricted model (`SARNegBinFlow`), each ρ parameter (ρ_d, ρ_o, ρ_w) is updated via independent 1-D slice sampling with β marginalised out. For the separable model (`SARNegBinFlowSeparable`), ρ_w = −ρ_d·ρ_o is deterministic and only ρ_d and ρ_o are sampled. The aspatial `NegBinFlow` omits the ρ block entirely.

## Log-Determinant Methods

The spatial Jacobian $\log|I - \rho W|$ is evaluated at every MCMC draw, and is
the term that makes large problems expensive. `logdet_method` is set on the
**model** constructor, not on `fit()`, and both samplers honour it. Leaving it
at `None` auto-selects by size, by whether $W$ is symmetric, and by how much
fill-in a sparse factorization would incur.

### Auto-selection

| $n$ | $W$ | Chosen | Why |
|---|---|---|---|
| ≤ 500 | any | `eigenvalue` | one $O(n^3)$ eigendecomposition, then $O(n)$ per ρ — exact and cheap at this size |
| ≤ 60000 | symmetric | `chol_aaa` | sparse Cholesky at adaptively-chosen AAA support points; exact, root-exponential convergence |
| ≤ 60000 | non-symmetric | `aaa` | the same rational scheme over sparse LU (KLU), for directed graphs — k-nearest-neighbour, travel time, migration |
| > 60000 | any | `cheb_stochastic` | stochastic Chebyshev expansion; no factorization, at the cost of stochastic error |

**Fill-in guard.** Size alone does not predict factorization cost — a dense or
hub-dominated graph blows up under Cholesky regardless of $n$. Before
committing to an exact path the selector estimates
$\mathrm{nnz}(W^2)/\mathrm{nnz}(W)$ in $O(\mathrm{nnz})$; if that exceeds 20 it
warns and falls back to `cheb_stochastic`. A KNN-50 graph or a fully dense $W$
at moderate $n$ takes that branch.

### The full set

| Method | Exact | Notes |
|---|---|---|
| `eigenvalue` | ✅ | full eigendecomposition; the reference answer at small $n$ |
| `chol_aaa` | ✅ | CHOLMOD factorizations at AAA support points; auto choice for symmetric $W$ |
| `aaa` | ✅ | AAA rational approximation over sparse LU; handles non-symmetric $W$ |
| `cheb_cholesky` | ✅ | sparse Cholesky at Chebyshev nodes |
| `lu_cheb` | ✅ | sparse LU at Chebyshev nodes |
| `chebyshev` | ✅ | deterministic Chebyshev from exact eigenvalues |
| `cholmod` | ✅ | JAX-native sparse CHOLMOD; requires `sparsax` |
| `grid_spline` | ≈ | spline interpolation over a precomputed ρ grid |
| `cheb_stochastic` | ✗ | stochastic Chebyshev (Han et al. 2015); auto choice above the cutoff |
| `slq` | ✗ | Stochastic Lanczos Quadrature, D-symmetrised |
| `traces` | ✗ | truncated trace series; legacy, retained for the flow NUTS path |

Flow models take an additional value, `"resolvent"`, which is their default for
the unrestricted three-ρ case: it samples via the resolvent-Kronecker gradient
rather than evaluating a scalar log-determinant.

Cutoffs are configurable through the environment:
`NEIGHBAYES_LOGDET_EIGEN_MAX_N` (default 500),
`NEIGHBAYES_LOGDET_CHEB_MAX_N` (default 60000), and
`NEIGHBAYES_LOGDET_MAX_FILLIN_RATIO` (default 20).

The constructor reports the valid names on a bad value, so the list above can
be checked against any installation:

```python
SAR(y=y, X=X, W=W, logdet_method="?")   # ValueError lists every valid option
```
