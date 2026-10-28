"""Shared pieces of the Pólya–Gamma negative binomial samplers.

The α step and the pointwise log-likelihood that the reduced-form, panel,
ZINB, hurdle and flow samplers share (:mod:`._core`).  No model samples
through this package directly, so it registers no Gibbs entry.
"""

from ._core import GibbsState

__all__ = ["GibbsState"]
