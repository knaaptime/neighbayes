"""Bayesian diagnostics for spatial econometric models.

This sub-package provides:

- **Bayesian LM tests** — Lagrange-multiplier-style specification tests
  evaluated over posterior draws rather than point estimates.
- **Bayes factor comparison** — exact (quadrature), bridge-sampling and
  BIC-based model comparison.

Both modules operate on fitted Bayesian model objects (i.e. models with
``inference_data`` attached) and are fully posterior-aware.

Symbols are loaded lazily via ``lazy_loader`` (SPEC 1).
"""

import lazy_loader as _lazy

__getattr__, __dir__, __all__ = _lazy.attach_stub(__name__, __file__)
