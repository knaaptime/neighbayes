"""Custom Gibbs samplers for spatial econometric models.

Subpackages
-----------
gaussian
    Gaussian (SAR/SEM/SDM/SDEM) Gibbs samplers.
panel
    Panel (FE/RE) Gibbs samplers.
negbin
    The \u03b1 step and log-likelihood the negative binomial samplers share.
_utils
    Internal utilities (not public API).

Import from the subpackages directly::

    from neighbayes.samplers.gaussian import GaussianSARGibbs
    from neighbayes.samplers.panel import REGibbsEstimation
    from neighbayes.samplers.negbin_reduced import run_chain

This module intentionally re-exports nothing: sampler families register
their Gibbs entries on *their own* import, and every model module imports
the family it needs \u2014 keeping ``neighbayes.models`` imports fast.
"""
