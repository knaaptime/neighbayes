"""The NB2 dispersion α, sampled or held fixed, shared by every NB model.

``priors={"alpha_fixed": a}`` holds α at ``a``: the Gibbs samplers skip the α
update and the PyMC models record α as a constant.  A large fixed α (10-20 times
the typical mean count) makes the NB an essentially Poisson model that keeps the
exact Pólya–Gamma sampler; very large values (≳ 1000) mix poorly, because the
Pólya–Gamma working precision grows with α while the information in the data
does not.
"""

from __future__ import annotations

from typing import Mapping, Optional

import numpy as np


def nb_alpha_fixed(priors: Mapping) -> Optional[float]:
    """The validated ``alpha_fixed`` value of a priors mapping, or ``None``."""
    value = priors.get("alpha_fixed")
    if value is None:
        return None
    value = float(value)
    if not (np.isfinite(value) and value > 0.0):
        raise ValueError(f"alpha_fixed must be a positive number, got {value!r}.")
    return value


def nb_alpha_rv(alpha_fixed: Optional[float], alpha_nu: float, alpha_sigma: float):
    """Inside a ``pm.Model``: ``α ~ Half-t(ν, σ)``, or a constant when held fixed.

    The fixed case is still recorded as ``alpha`` (a ``Deterministic``), so
    downstream code reading ``posterior["alpha"]`` works either way.
    """
    import pytensor.tensor as pt

    from ..._lazy_deps import pm

    if alpha_fixed is None:
        return pm.HalfStudentT("alpha", nu=alpha_nu, sigma=alpha_sigma)
    return pm.Deterministic("alpha", pt.constant(float(alpha_fixed), dtype="float64"))
