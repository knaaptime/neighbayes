# neighbayes

[![Continuous Integration](https://github.com/knaaptime/neighbayes/actions/workflows/unittests.yml/badge.svg)](https://github.com/knaaptime/neighbayes/actions/workflows/unittests.yml)
[![codecov](https://codecov.io/gh/knaaptime/neighbayes/branch/main/graph/badge.svg?token=XO4SilfBEb)](https://codecov.io/gh/knaaptime/neighbayes)

`neighbayes` is a Python package for Bayesian estimation of spatial econometric models. Every model is a class, and specified with a Wilkinson formula via
[`formulaic`](https://matthew.wardrop.casa/formulaic/latest/) and a PySAL
[`Graph`](https://pysal.org/libpysal/stable/generated/libpysal.graph.Graph.html)
(or any `scipy.sparse` matrix) for the spatial weights $W$; `fit()` returns an
ArviZ `DataTree` (an `xarray.DataTree`), so the standard posterior tooling
(`az.plot_trace_dist`, `az.compare`, `az.loo`) works without translation (though
exercise caution as LOO diagnostics aren't valid for spatial models).

```python
import libpysal
from neighbayes.dgp import simulate_sar
from neighbayes.models import OLS, SAR

gdf = simulate_sar(n_side=20, beta=[1, 0.4, 2.5], rho=0.6, create_gdf=True)
G = libpysal.graph.Graph.build_contiguity(gdf).transform("r")
form = "y ~ -1 + X_0 + X_1 + X_2"

# start with a baseline OLS and test the residuals for spatial dependence
ols = OLS(formula=form, W=G, data=gdf)
ols.fit()
print(ols.spatial_diagnostics_decision(format="ascii"))

# the diagnostics point to SAR here, so fit it and decompose the spillovers
sar = SAR(formula=form, W=G, data=gdf)
sar.fit(draws=2000, chains=4, random_seed=42)
sar.summary()
sar.spatial_effects()
```

`neighbayes` provides roughly fifty model classes behind one formula interface, each
with a matching simulator in `neighbayes.dgp`. Every fit comes with Bayesian LM
diagnostics (with a decision tree for specification search) and posterior decompositions
of direct, indirect, and total effects. Every model (almost) has a custom Gibbs sampler
for [better mixing](https://onlinelibrary.wiley.com/doi/abs/10.1111/gean.12135) (with both numby and jax implementations), but can fall back to PyMC if requested.

## Installation

```bash
conda install -c conda-forge neighbayes   # or: pip install neighbayes
```

Requires Python 3.12+. The [installation guide](https://pysal.org/neighbayes/installation.html)
covers development installs and the optional JAX backend.

## Documentation

Model-by-model user guides, the diagnostics battery, effects interpretation, and the
API reference are at [pysal.org/neighbayes](https://pysal.org/neighbayes).

## Validation

Every model includes thorough parameter recovery tests, and samplers have been validated
against maximum likelihood estimators as well as published results (see docs).
