"""Lazily-loaded heavy optional dependencies shared across the library.

``pymc`` and ``arviz`` together cost roughly three seconds to import but are
only needed when a model is *fit* (NUTS model build, ``pm.sample``, idata
assembly), never when a model *class* is imported.  Loading them lazily
(SPEC 1, https://scientific-python.org/specs/spec-0001/) keeps
``from neighbayes.models import SAR`` fast — the real import is triggered on
first attribute access (e.g. ``pm.Model`` inside ``_build_pymc_model``).

Import these as ``from ..._lazy_deps import pm, az, xr`` (adjust the leading dots
to the module's depth) instead of ``import pymc`` so the whole package shares
one deferred proxy.
"""

from __future__ import annotations

import lazy_loader as _lazy

pm = _lazy.load("pymc")
az = _lazy.load("arviz")
# Posterior containers are ``xarray.DataTree`` (ArviZ >= 1).
xr = _lazy.load("xarray")
