# Supported Models


## Model suite

The package organizes models along three column dimensions — **likelihood** (linear / non-linear), **temporal structure** (cross-section / panel), and **outcome structure** (single / flow). Each cell lists the spatial structures implemented for that combination.

| | Linear · Cross-section | Linear · Panel | Non-linear · Cross-section | Non-linear · Panel |
|---|---|---|---|---|
| **Single** | Aspatial, SLX, SAR, SEM, SDM, SDEM | Aspatial, SLX, SAR, SEM, SDM, SDEM | Aspatial, SAR, SEM, SDM | Aspatial, SAR, SEM |
| **Flow** | Aspatial, SAR, SEM | Aspatial, SAR, SEM | Aspatial, SAR | Aspatial, SAR |

Panel models come in fixed-effects and random-effects variants, and the linear
panel family additionally has dynamic (lagged-dependent-variable) forms. Every
spatial flow model has a **separable** counterpart that pins
$\rho_w = -\rho_d \rho_o$. The non-linear cells include negative binomial,
zero-inflated NB and hurdle NB count models (and Poisson for cross-sectional
flows). Outside the table, `SpatialMultilevel` nests units in groups in larger
groups, with a graph and a process at every level
([multilevel models](multilevel-models)). The sections below list every class
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

The random-effects Gibbs sampler (this and the spatial variants below) updates
$\sigma_\alpha$ by interweaving a centred step ($\sigma_\alpha \mid \alpha$) with
a non-centred one ($\sigma_\alpha \mid \alpha/\sigma_\alpha$, exact under the
Gaussian likelihood). The centred step alone mixes at about 1% of draws when the
unit effects are small against $\sigma/\sqrt{T}$, the common many-units,
few-periods spatial panel.

### SAR panel (Random Effects)

$$y_{it} = \rho W y_{it} + x_{it}' \beta + \alpha_i + \tau_t + \epsilon_{it}, \quad \alpha_i \sim N(0, \sigma_\alpha^2)$$

### SEM panel (Random Effects)

$$y_{it} = x_{it}' \beta + \alpha_i + \tau_t + u_{it}, \quad u_{it} = \lambda W u_{it} + \epsilon_{it}, \quad \alpha_i \sim N(0, \sigma_\alpha^2)$$

### SDEM panel (Random Effects)

$$y_{it} = x_{it}' \beta + W x_{it}' \theta + \alpha_i + u_{it}, \quad u_{it} = \lambda W u_{it} + \epsilon_{it}, \quad \alpha_i \sim N(0, \sigma_\alpha^2)$$

(multilevel-models)=
## Multilevel Models

### SpatialMultilevel

$$\theta_\ell = \rho_\ell W_\ell \theta_\ell + X_\ell \beta_\ell + \Delta_\ell \theta_{\ell+1} + \varepsilon_\ell, \quad \varepsilon_\ell \sim N(0, \sigma_\ell^2 I), \quad \ell = 0, \dots, L, \quad \theta_0 \equiv y$$

Units (level 0) nest in groups (level 1), which nest in larger groups, up to
level $L$: schools in districts in states, tracts in counties in states. Each
level has its own graph $W_\ell$, covariates $X_\ell$, innovation sd
$\sigma_\ell$ and process, and each level's effect $\theta_\ell$ enters the
equation of the level below through $\Delta_\ell$, which maps a row to its
parent. In place of the lag, any level may carry an error process,
$\theta_\ell = X_\ell\beta_\ell + \Delta_\ell\theta_{\ell+1} + v_\ell$ with
$v_\ell = \lambda_\ell W_\ell v_\ell + \varepsilon_\ell$, or none, and any level
may add the Durbin terms $W_\ell X_\ell$.

Every graph spans all groups at its level, whatever their parent: a West Texas
county neighbours New Mexico counties. Because each level's effect passes
through the filter of the level below, a shift at the top reaches the units
through every filter beneath it. In the reduced form with a lag at every level,

$$y = S_0\big(X_0\beta_0 + \varepsilon_0 + \Delta_0 S_1(X_1\beta_1 + \varepsilon_1 + \Delta_1 S_2(X_2\beta_2 + \varepsilon_2))\big), \qquad S_\ell = (I - \rho_\ell W_\ell)^{-1},$$

so a Texas policy spreads into Oklahoma's border districts and from there into
their schools. With two levels this chain and an additive entry of the effects
coincide, and the model nests the two-level hierarchical SAR models: Dong &
Harris's HSAR (a lag at the units, an error process with no covariates above),
the dual spatial-error model of Wolf et al. (spvcm), and Lacombe & McIntyre's
upper-level lag or error with Durbin terms. The units carry the only intercept;
upper levels drop theirs, since through row-standardized filters constants at
different levels are confounded.

```python
from neighbayes.models import Level, SpatialMultilevel

m = SpatialMultilevel(
    Level("score ~ frl + log_enroll", data=schools, W=W_schools),         # level 0
    Level("~ income + segregation", data=districts, W=W_districts,
          key="district_id"),                                             # level 1
    Level("~ spending", data=states, W=W_states, key="fips",
          process="error"),                                               # level 2
)
idata = m.fit()
```

A level's position is its index $\ell$. `key` names the id column shared with
the level below; it is looked up in the level below's data, then in the units'
data, where strict nesting is checked. A Graph's ids order a level's groups (its
data are reindexed to them). In matrix mode, `groups` gives each lower row's
parent label. The posterior follows the levels: `rho_ℓ` (or `lam_ℓ` for an
error process), `beta_ℓ`, `sigma_ℓ`, and the effects `theta_ℓ` (coordinate
`group_ℓ`).

**Priors.** $\sigma_0^2 \sim \text{Inv-}\Gamma(2, \operatorname{Var} y)$; each
upper level's $\sigma_\ell \sim \text{half-}t_3(0, \operatorname{sd} y)$, since
every level's effects are in the outcome's units; Gelman et al. (2008) priors on
each level's $\beta_\ell$; uniform priors on each graph's stability bounds for
$\rho_\ell$. See `MultilevelPriors`.

**Sampling.** Given the $\rho$'s and $\sigma$'s, the coefficients of every level
and all the effects are jointly Gaussian with a sparse precision, so the Gibbs
sampler draws them in one sparse Cholesky block. Drawing them together removes
the ridge between the units' intercept and the mean of the effects. Every
$\rho_\ell$ and upper-level $\sigma_\ell$ is drawn from its conditional with that
block integrated out (`parametrization="collapsed"`, the default), at one
refactorization per evaluation. Three other schemes update the upper levels'
$\rho_\ell$ and $\sigma_\ell$ given the block instead:
- `"centred"`, given the effects;
- `"noncentred"`, given the standardized innovations $\varepsilon_\ell/\sigma_\ell$, a move that needs no Jacobian;
- `"interweave"`, both in turn (Yu & Meng 2011).

These need only the block's conditional, so they carry over to non-Gaussian
likelihoods.

On a three-level lag model with 16 units per group, the collapsed scheme gave
7–14× the effective sample size of interweaving for the upper levels' $\rho$
and $\sigma$, and the most per second.

`gibbs_backend="jax"`, the default when JAX is installed, runs the same sweep
compiled, using sparsax's CHOLMOD and LU. When the top level is small (a few
dozen states, say), its $\rho$ and $\sigma$ are updated through a dense Schur
complement, which spares the large factorizations. The sweep is compiled once
per model structure and reused by later fits, including fits to new data of
the same shape, so a simulation study pays for compilation once. Per sweep it
ran:
- 2–10× faster than NumPy with about 600 units;
- 1.5–4× faster with 14,400 units, where the collapsed scheme's cost is
  CHOLMOD's own factorization. `sampler="nuts"` fits the same model
non-centred. Gibbs and NUTS agree across lag, error and no-process
configurations. The units' intercept is heavy-tailed: its spread grows as the top
level's $\rho$ nears 1, where NUTS diverges.

**Effects.** `spatial_effects(level=ℓ)` reports the effects of level $\ell$'s
covariates on the units, through the composed multiplier
$S_0\Delta_0 S_1 \cdots \Delta_{\ell-1}S_\ell$, where $S_m$ is the identity for a
level with an error process or none:
- *direct*: the mean effect on a unit of a shift in its own group;
- *total*: the mean effect of a shift in every group;
- *indirect*: the difference, which is what reaches a unit from other groups through the filters.

`on="level"` gives the effects on $\theta_\ell$ itself, through $S_\ell$ alone.
The own-group trace is exact up to `exact_max` groups and estimated with
Rademacher probes above.

- **SpatialMultilevel**: Gaussian, cross-sectional, $L \ge 1$ nested levels.
  Panels, count likelihoods and crossed (non-nested) classifications are
  planned.

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

### Panel count models (SARNegBinPanel, NegBinPanel)

$$y_{it} \sim \operatorname{NegBin}(\mu_{it}, \alpha), \quad \log \boldsymbol{\mu}_t = (I - \rho W)^{-1} X_t\beta + c + \tau_t$$

Unit effects $c$ and period effects $\tau_t$ are selected by `effects`
(`"pooled"`, `"unit"`, `"time"`, `"two_way"`). The Gaussian panels' within
transform does not carry to a log link, so the effects are parameters, held as
a unit index rather than dummy columns. Unit effects sit outside the spatial
filter: $(I_T \otimes A^{-1})$ commutes with the unit dummies, so this is the
inside-filter model reparameterized. The Gibbs sampler integrates the unit
effects out of every $\rho$ update by an ω-weighted analogue of the within
transform, recomputed each sweep from the Pólya–Gamma weights.

The unit effects are **partially pooled**: $c_i \sim N(\mu, \sigma^2)$ with
$\sigma$ learned (a half-$t_3$ prior with scale `group_effect_sd_scale`,
default 1), so a unit seen in few periods is shrunk toward the common mean in
proportion to how little its data say. Spatial panels typically have many units
and few periods (Elhorst 2014), and there a fixed, wide prior leaves each
unit's level pinned only by its own few periods, which biases the dispersion $\alpha$ and $\rho$ (the incidental-parameter problem)
and lets $\rho$ drift toward 1, where the filter's amplification of an
unanchored level goes unchecked. The unit effects absorb no column: the
intercept and time-invariant covariates stay in the design. Period effects keep
a fixed prior and absorb the columns that vary only over time. Pooling assumes
the effects are unrelated to $X$; when they are not, `mundlak=True` adds the
unit means of the time-varying columns to the design inside the filter (Mundlak
1978), which absorbs the correlation (in a nonlinear model only approximately).
The Gibbs sampler updates $\sigma$ by interweaving a centred step
($\sigma \mid c$) with a non-centred one ($\sigma \mid \tilde c$,
$\tilde c = (c - \mu)/\sigma$; Yu & Meng 2011), the latter on the exact
likelihood rather than the Pólya–Gamma working one, so no augmentation layer
stands between $\sigma$ and the data (Papaspiliopoulos, Roberts & Sermaidis
2011). That keeps $\sigma$ mixing in the large-$N$, small-$T$ regime, where each
unit's data say little about its effect.
Both classes default to Gibbs; `sampler="nuts"` runs the PyMC model. `priors={"alpha_fixed": a}` holds $\alpha$ fixed; a large value
(10-20 times the typical mean count) gives an essentially Poisson model that
keeps the exact Pólya–Gamma sampler.

### SARNegBin (Reduced Form)

$$y_i \sim \operatorname{NegBin}(\mu_i, \alpha), \quad \mu = \exp(\eta), \quad \eta = (I - \rho W)^{-1} X\beta$$

No latent $\sigma$ — spatial dependence enters only through the mean propagator. Supports both NUTS and Pólya–Gamma Gibbs sampling.

### SARNegBinStructural (Structural Form)

$$y_i \sim \operatorname{NegBin}(\mu_i, \alpha), \quad \eta = \rho W \eta + X\beta + \nu, \quad \nu \sim \mathcal{N}(0, \sigma^2 I)$$

Includes latent $\sigma$ — structural form with explicit noise. Gibbs sampling only (PG augmentation).

### SARZINB

$$y_i \sim \operatorname{ZINB}(\mu_i, \alpha, \pi_i), \quad \log \boldsymbol{\mu} = (I - \rho W)^{-1} X\beta, \quad \operatorname{logit} \boldsymbol{\pi} = (I - \lambda W_{\mathrm{sel}})^{-1} Z\gamma$$

Zero-inflated negative binomial: a spatial-lag logit selection (is the unit
active?) and a spatial-lag NB count, both in reduced form. Pólya–Gamma Gibbs
by default; `sampler="nuts"` fits the same model in PyMC. Corridor
probabilities, zero attribution and fitted means are posterior expectations;
`posterior_predictive` simulates the zero process too.

**Identification.** Only the shape of the count distribution separates
structural zeros from sampling zeros. When active units average about one
count, NB with a smaller α explains the zeros nearly as well as zero inflation
does. The selection equation is then weakly identified, the posterior of λ is
close to its prior, and the Gibbs chain mixes slowly. Repeated periods fix this:
in a panel with unit (or pair) effects, a unit's other periods pin down its
count distribution, and the zeros in excess of it are structural. For "any
versus none" questions on sparse counts, use a hurdle model
({ref}`below <hurdle-models>`), whose binary half is fit to the observed
zeros.

### ZINB panels (SARZINBPanel, ZINBPanel)

The same two equations per period, $T$ periods stacked time-first, with
per-period structural zeros and unit and period effects on the count
equation (`effects`, pooled as for the NB panels). The selection design `Z` may vary
over time.

### Flow ZINB (SARZINBFlowSeparable, SARZINBFlowSeparablePanel)

Both equations are separable flow models,
$\eta^{\mathrm{sel}} = (L^{\lambda}_o \otimes L^{\lambda}_d)^{-1} Z\gamma$ and
$\eta^{\mathrm{cnt}} = (L^{\rho}_o \otimes L^{\rho}_d)^{-1} X\beta$, sampled by a
structured $n \times n$ sweep (NumPy or `gibbs_backend="jax"`) that never forms
an $n^2 \times n^2$ matrix. The panel adds pair and period effects on the count
equation; the cross-section is the panel with $T = 1$. Flow counts are usually
sparse, so the identification caveat above bites hardest here: use the panel
with pair effects, or the flow hurdle.

(hurdle-models)=
### Hurdle models (SARHurdleNB and family)

$$P(y > 0) = \operatorname{logit}^{-1}(\eta^{\mathrm{b}}), \quad y \mid y > 0 \sim \operatorname{NB}(\mu, \alpha) \text{ truncated at } 0, \quad \eta^{\mathrm{b}} = (I - \lambda W_{\mathrm{sel}})^{-1} Z\gamma, \quad \log \boldsymbol{\mu} = (I - \rho W)^{-1} X\beta$$

A hurdle separates *whether* a count is positive from *how large* it is when
it is. Both halves are reduced form: the spatial lag acts on the linear
predictor and no latent noise field enters, so neither half has a Jacobian.

**Hurdle or ZINB.** Both give positive counts the same zero-truncated NB
distribution; they differ only in the probability of a positive count. The
hurdle models it directly, $P(y > 0) = \pi$. ZINB splits it into an
activation probability and the NB's own chance of a positive count,
$\pi\,(1 - \operatorname{NB}(0))$, and so can call a zero structural. The data
see that split only through the shape of the positive counts, which on sparse
counts barely constrains it (see SARZINB). The hurdle's binary half is fit to
the observed zeros, so it is identified however sparse the counts are. What it
gives up is the split: its γ and λ describe whether any count occurs, not
whether a unit is "open". Use the hurdle when the question is "any versus none",
and ZINB, in a panel with unit or pair effects, when the question is
"structurally closed versus open but quiet".

**Sampling.** The binary half is a Pólya–Gamma logit. In the count half the
truncation is augmented exactly: each positive cell gets a geometric count of
the zeros the truncation hid, after which the cell is Pólya–Gamma conjugate.
When the positive counts are mostly ones, the count level and α trade off along
a ridge, so they are drawn jointly by a slice sampler that moves along it.
`sampler="nuts"` fits the same model in PyMC (`HurdleNegativeBinomial`).

- **SARHurdleNB**: cross-section, with separate `Z` (or `sel_formula`) and
  `W_sel` for the binary half.
- **SARHurdleNBPanel**, **HurdleNBPanel**: $T$ periods stacked time-first,
  with unit and period effects (`effects`) in **both** halves. In the binary
  half a unit effect absorbs the unit's baseline chance of a positive count,
  so γ comes mostly from within-unit switches between zero and positive
  periods. Both halves' unit effects are partially pooled, with their own
  learned sds; the pooling is what keeps α and ρ unbiased when units have few
  positive periods (about three per unit in simulations, where a fixed, wide
  prior gave α at half its true value).
- **SARHurdleNBFlowSeparable**, **SARHurdleNBFlowSeparablePanel**: both halves
  are separable flow models, sampled by the structured $n \times n$ sweep
  (NumPy or `gibbs_backend="jax"`), with pair and period effects in both
  halves.

The halves share no parameters, so the posterior factorizes. Linking them
through correlated unit (pair) effects is a planned extension.

### Logit

$$y_i \sim \operatorname{Bernoulli}(p_i), \quad \operatorname{logit}(p) = X\beta$$

Non-spatial logistic regression baseline.

### NegBin

$$y_i \sim \operatorname{NegBin}(\mu_i, \alpha), \quad \log \boldsymbol{\mu} = X\beta$$

Non-spatial negative binomial baseline.

### SARLogit (Reduced Form)

$$y_i \sim \operatorname{Bernoulli}(p_i), \quad \eta = (I - \rho W)^{-1}X\beta$$

The reduced form: the spatial multiplier acts on the mean of the log-odds, with
no latent noise field (that is `SARLogitStructural`). Pólya–Gamma Gibbs sampler
only — no NUTS path.

### SARLogitStructural (Structural Form)

$$y_i \sim \operatorname{Bernoulli}(p_i), \quad \eta = \rho W \eta + X\beta + \nu, \quad \nu \sim \mathcal{N}(0, I)$$

A latent-field model: unlike `SARLogit`, the log-odds carry a noise term $\nu$
that the spatial multiplier also propagates. Pólya–Gamma Gibbs sampler only.

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
exact Poisson representation. Use it for counts close to Poisson: the normal
mixture behind the sampler has lighter tails than the error it replaces, so
under overdispersion it is biased toward large counts (`fit` reports the
Pearson dispersion and warns above 2). For overdispersed counts use the NB
flow models; an NB with `priors={"alpha_fixed": a}`, `a` 10-20 times the
typical mean, is an essentially Poisson model on the exact Pólya–Gamma
sampler.

### SARPoissonFlowSeparable

$$y_{ij} \sim \operatorname{Poisson}(\mu_{ij}), \quad \log \boldsymbol{\mu} = A(\boldsymbol{\rho})^{-1} X\beta, \quad \rho_w = -\rho_d \rho_o$$

The recommended Poisson flow model — the separable restriction removes the
weakly-identified $\rho$ ridge of the unrestricted variant.

### SEMFlow

$$y = X\beta + u, \quad u = \lambda_d W_d u + \lambda_o W_o u + \lambda_w W_w u + \varepsilon, \quad \varepsilon \sim \mathcal{N}(0, \sigma^2 I)$$

### SEMFlowSeparable

$$y = X\beta + u, \quad u = \lambda_d W_d u + \lambda_o W_o u - \lambda_d \lambda_o W_w u + \varepsilon, \quad \varepsilon \sim \mathcal{N}(0, \sigma^2 I)$$

## Panel Flow Models

Stack the flow models above across $T$ periods in time-first order. The NB
panels take pair and period effects as parameters, the pair effects partially
pooled,
$\log \boldsymbol{\mu}_t = A(\boldsymbol{\rho})^{-1} X_t\beta + c + \tau_t$,
with one effect per origin-destination pair held as an index, never as $n^2$
dummy columns; see the panel count models above. The separable NB panel keeps
its structured $n \times n$ sweep with effects.

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
raises `NotImplementedError`. Robust (Student-t) SAR/SEM/SDM/SDEM and
Gaussian panel FE models keep Gibbs: the Student-t error is sampled as a
normal scale mixture with the same fixed `nu` the NUTS path uses. Robust
random-effects and Tobit models still require NUTS.

`target_accept` is NUTS-only and raises `TypeError` if passed with Gibbs.

### Execution backends

For any Gibbs sampler, `gibbs_backend` selects the execution path:

| Value | Behaviour |
|---|---|
| `"auto"` | **default** — JAX when installed and supported by the family, else NumPy |
| `"jax"` | the sweep JIT-compiled into one XLA kernel; chains vectorised under `jax.vmap`, controlled by `chain_method` |
| `"numpy"` | pure NumPy/SciPy; chains as separate processes via `joblib`, controlled by `n_jobs` |

Both backends implement the same sampler and target the same posterior.

A JAX sweep is compiled once per model structure (its dimensions and its
process, effect and prior options) and reused for the rest of the session. The
first fit pays for compilation, from under a second to a few seconds; a refit,
a longer run, or new data of the same dimensions on the same graph does not, so
a simulation study compiles once rather than once per replicate.

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
