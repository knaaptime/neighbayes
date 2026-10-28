"""Free helpers for resolving the NUTS sampling path.

NUTS backend selection (``"pymc"``, ``"numpyro"``, ``"blackjax"``,
``"nutpie"``) is handled directly by PyMC via ``pm.sample(nuts_sampler=...)`` —
there is no backend object to construct.  This package only exposes the small
helper functions the model ``fit`` / ``_build_pymc_model`` paths need to
normalize ``pm.sample`` kwargs and decide whether a JAX-native likelihood
(:class:`pymc.CustomDist`) path is required for a given ``nuts_sampler``.
"""

from __future__ import annotations

from .sampler_helpers import (
    enforce_c_backend,
    jax_available,
    prepare_compile_kwargs,
    prepare_idata_kwargs,
    select_backend,
    use_jax_likelihood,
)

__all__ = [
    "enforce_c_backend",
    "jax_available",
    "prepare_compile_kwargs",
    "prepare_idata_kwargs",
    "select_backend",
    "use_jax_likelihood",
]
