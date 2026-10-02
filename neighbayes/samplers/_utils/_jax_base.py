"""Factory for JAX-compatible Gibbs state classes.

The pattern ``if _eqx_available(): class JAX...State(eqx.Module): ... else: class JAX...State: raise ImportError``
is repeated across the negbin, logit and gaussian samplers.
This module provides a factory that creates the equinox.Module subclass
(or stub) once, caching the result so ``jax.lax.scan`` sees a stable type.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Type

import numpy as np

from ._base import GibbsBaseState


def _eqx_available() -> bool:
    """Return True when optional ``equinox`` is importable."""
    import importlib.util

    return importlib.util.find_spec("equinox") is not None


def _jax_available() -> bool:
    """Return True when optional ``jax`` is importable."""
    import importlib.util

    return importlib.util.find_spec("jax") is not None


@lru_cache(maxsize=None)
def make_jax_state_class(
    name: str,
    fields: tuple[str, ...],
    numpy_state_cls: Type | None = None,
) -> type:
    """Create (or return cached) a JAX-compatible equinox.Module state class.

    Parameters
    ----------
    name : str
        Class name (e.g. ``"JAXGibbsState"``).
    fields : tuple[str, ...]
        Field names for the equinox.Module (e.g. ``("eta", "beta", "sigma2", "rho", "alpha", "omega")``).
    numpy_state_cls : type, optional
        The corresponding numpy state dataclass.  When provided, a
        ``to_numpy()`` method is added that converts JAX arrays back to
        the numpy state.  When ``None``, no ``to_numpy()`` is generated.

    Returns
    -------
    type
        An ``equinox.Module`` subclass when equinox+JAX are available,
        or a stub class that raises ``ImportError`` on instantiation.
    """
    if not (_eqx_available() and _jax_available()):
        # Stub class — should never be instantiated.
        def __init__(self, *args, **kwargs):
            raise ImportError(
                "equinox and jax are required for the JAX Gibbs sampler path. "
                "Install with: pip install equinox jax"
            )

        return type(name, (), {"__init__": __init__})

    import equinox as eqx
    import jax.numpy as jnp

    # Build the class dynamically as an eqx.Module subclass.
    # We use __init_subclass__ via eqx.Module's metaclass, so we need
    # to create the class with annotations set up properly.
    annotations = {f: jnp.ndarray for f in fields}
    cls_dict: dict = {"__annotations__": annotations}

    if numpy_state_cls is not None:

        def to_numpy(self) -> GibbsBaseState:
            """Convert JAX arrays to the corresponding numpy state."""
            kwargs = {}
            for f in fields:
                val = getattr(self, f)
                if f in (
                    "rho",
                    "sigma2",
                    "alpha",
                    "sigma2_u",
                    "sigma2_y",
                    "rho_d",
                    "rho_o",
                    "gamma",
                ):
                    kwargs[f] = float(val)
                else:
                    kwargs[f] = np.asarray(val)
            return numpy_state_cls(**kwargs)

        cls_dict["to_numpy"] = to_numpy

    cls = type(name, (eqx.Module,), cls_dict)
    return cls
