"""Bayesian spatial econometric models and diagnostics.

The package exposes cross-sectional and panel spatial regression model
classes and Bayesian specification tests.

Submodules and attributes are loaded lazily following SPEC 1
(https://scientific-python.org/specs/spec-0001/) so that ``import neighbayes``
is cheap and does not eagerly import ``pymc``/``pytensor``/``arviz``. The
public API surface is declared in the sibling ``__init__.pyi`` stub for
static type checkers and IDE autocomplete.

Examples
--------
Import a model class from the ``models`` submodule::

        from neighbayes.models import SAR
"""

from importlib.metadata import PackageNotFoundError, version

import lazy_loader as _lazy

__getattr__, __dir__, __all__ = _lazy.attach_stub(__name__, __file__)

try:
    __version__ = version("neighbayes")
except PackageNotFoundError:  # running from a source tree with no metadata
    __version__ = "unknown"
