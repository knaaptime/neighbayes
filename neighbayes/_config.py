"""Import-time configuration snapshot for :mod:`neighbayes`.

This module snapshots the single ``NEIGHBAYES_*`` toggle that has no
call-time consumer: the Op instrumentation switch.

Env vars
--------
``NEIGHBAYES_OP_INSTRUMENT`` (default ``"0"``)
    Set to ``"1"`` to enable per-Op callback timing instrumentation.
    Off by default for zero overhead.

The other runtime knobs — ``NEIGHBAYES_SPARSE_BACKEND``,
``NEIGHBAYES_SPARSE_STRICT``, ``NEIGHBAYES_KRON_DENSE_MAX``,
``NEIGHBAYES_LOGDET_EIGEN_MAX_N``, ``NEIGHBAYES_LOGDET_FACTOR_MAX_N`` — are
read **at call time** by their consumers
(:mod:`neighbayes._ops._backend`, :mod:`neighbayes._logdet._config`) with
``functools.lru_cache``, so tests can override them via
``monkeypatch.setenv`` + ``cache_clear()``.  Snapshotting them here would
break that pattern, so this module deliberately owns only the toggle above.
"""

from __future__ import annotations

import os
from dataclasses import dataclass


def _env_bool(key: str, default: str = "0") -> bool:
    return os.environ.get(key, default).strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    """Snapshot of import-time ``NEIGHBAYES_*`` toggles."""

    # _ops / _instrument
    op_instrument: bool


settings: Settings = Settings(
    op_instrument=_env_bool("NEIGHBAYES_OP_INSTRUMENT", "0"),
)
