"""Lazily-loaded heavy dependencies shared across the library.

``pymc`` is only needed when a model is *fit* (NUTS model build, ``pm.sample``),
never when a model *class* is imported.  Loading it lazily (SPEC 1,
https://scientific-python.org/specs/spec-0001/) keeps
``from neighbayes.models import SAR`` fast — the real import is triggered on
first attribute access (e.g. ``pm.Model`` inside ``_build_pymc_model``).

Import these as ``from ..._lazy_deps import pm, xr`` (adjust the leading dots
to the module's depth) instead of ``import pymc`` so the whole package shares
one deferred proxy.

ArviZ is not here: import it inside the functions that use it.  ``arviz_base``
asks ``lazy_loader`` for JAX and NumPyro when it is imported, and for a missing
package ``lazy_loader`` calls ``inspect.stack()``, which runs every lazily
loaded module waiting in ``sys.modules``.  A waiting ``arviz`` then imports in
the middle of ``arviz_base``'s own import and fails, so without the JAX extras
the first NUTS fit, or ``import arviz_plots``, raised ``ImportError``
(``tests/test_imports.py``).  Neither ``pymc`` nor ``xarray`` imports
``arviz_base`` when loaded, so they stay here.
"""

from __future__ import annotations

import lazy_loader as _lazy

pm = _lazy.load("pymc")
# Posterior containers are ``xarray.DataTree`` (ArviZ >= 1).
xr = _lazy.load("xarray")
