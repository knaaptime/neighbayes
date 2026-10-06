"""Gibbs sampler for the Gaussian spatial multilevel model.

See :mod:`._core` for the model, the sweep and its references.  The model
class (:class:`neighbayes.models.SpatialMultilevel`) registers the runner.
"""

from ._core import (
    PARAMETRIZATIONS,
    PROCESSES,
    LevelSpec,
    MultilevelGibbsPriors,
    MultilevelStructure,
    log_marginal,
    run_multilevel_chain,
)

__all__ = [
    "PARAMETRIZATIONS",
    "PROCESSES",
    "LevelSpec",
    "MultilevelGibbsPriors",
    "MultilevelStructure",
    "log_marginal",
    "run_multilevel_chain",
]
