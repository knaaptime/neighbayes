"""Importing neighbayes must not break ArviZ on a machine without the JAX extras.

``arviz_base`` asks ``lazy_loader`` for JAX and NumPyro when it is imported.
For a missing package ``lazy_loader`` calls ``inspect.stack()``, which reads
``__file__`` from every entry in ``sys.modules`` and so runs any lazily loaded
module still waiting there.  A lazily loaded ``arviz`` left in ``sys.modules``
by neighbayes would then import itself in the middle of ``arviz_base``'s import
and fail on the half-initialized module.  PyMC imports ``arviz_base`` after
every NUTS fit, and users import ``arviz_plots`` to plot.
"""

import subprocess
import sys

import pytest

_SCRIPT = """
import sys

for name in ("jax", "numpyro"):
    sys.modules[name] = None  # the import fails and find_spec returns None
from neighbayes.models import SAR
import {module}
"""


@pytest.mark.parametrize("module", ["arviz_base", "arviz_plots"])
def test_arviz_imports_after_a_model_class_without_jax(module):
    result = subprocess.run(
        [sys.executable, "-c", _SCRIPT.format(module=module)],
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, result.stderr[-2000:]
