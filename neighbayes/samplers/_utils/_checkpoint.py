"""On-disk checkpoints for :func:`._jax_utils.run_chains_chunked`.

Turned on by :func:`neighbayes.config.checkpoints`.  A run is named by a
fingerprint of everything that determines its draws: its lengths and chunk
length, the sweep's structure key, every chain's initial state and keys, and
the data in ``consts``.  Its directory under the checkpoint root holds
``state.pkl`` (the chains' states and keys, the phase and the sweeps done in
it, and the warmup traces a midpoint hook has kept so far or was given),
replaced atomically, and the retained draws in numbered ``draws_*.pkl`` files,
each written before the state that counts it.  A run resumes from its last
state at a chunk boundary, so its draws are the uninterrupted run's; one that
finishes removes its directory.

A midpoint hook's substitutions are not stored.  Its input is, and on resume
the hook is run on it again: an adaptation is deterministic in its input, so
this rebuilds both the ``consts`` it substituted and the hook's own state,
which its caller reads after the run.  The chains' states are saved after the
substitution and are not touched again.
"""

from __future__ import annotations

import contextvars
import hashlib
import os
import pickle
import shutil
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import numpy as np

#: Bump when the saved layout changes, so older checkpoints are not resumed.
_FORMAT = 1


@dataclass(frozen=True)
class _Settings:
    root: Path
    every: float


_ACTIVE: contextvars.ContextVar = contextvars.ContextVar(
    "neighbayes_checkpoints", default=None
)


@contextmanager
def checkpoints(path, *, every: float = 300.0):
    """Checkpoint the JAX Gibbs runs started inside the block under ``path``.

    See :func:`neighbayes.config.checkpoints`.
    """
    root = Path(path)
    root.mkdir(parents=True, exist_ok=True)
    token = _ACTIVE.set(_Settings(root, float(every)))
    try:
        yield root
    finally:
        _ACTIVE.reset(token)


def fingerprint(*, sweep_key: str, lengths: tuple, trees: tuple) -> str:
    """SHA-256 of the run's description and of every leaf of ``trees``."""
    import jax

    h = hashlib.sha256(repr((_FORMAT, sweep_key, lengths)).encode())
    for tree in trees:
        leaves, treedef = jax.tree_util.tree_flatten(tree)
        h.update(str(treedef).encode())
        for leaf in leaves:
            a = np.ascontiguousarray(np.asarray(leaf))
            h.update(repr((a.dtype.str, a.shape)).encode())
            h.update(a.tobytes())
    return h.hexdigest()


def _write(path: Path, obj) -> None:
    """Pickle ``obj`` to ``path`` through a temporary file renamed over it."""
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, path)


def _read(path: Path):
    with open(path, "rb") as f:
        return pickle.load(f)


def _to_host(tree):
    import jax

    return jax.tree_util.tree_map(np.asarray, tree)


class RunCheckpoint:
    """One run's checkpoint directory."""

    def __init__(self, settings: _Settings, fp: str):
        self.fp = fp
        self.dir = settings.root / fp[:32]
        self.every = settings.every
        self.n_files = 0
        self._pending: list = []
        self._last = time.monotonic()

    @classmethod
    def for_run(cls, **fingerprint_kw) -> "RunCheckpoint | None":
        """The checkpoint of a run, or ``None`` outside :func:`checkpoints`."""
        settings = _ACTIVE.get()
        if settings is None:
            return None
        return cls(settings, fingerprint(**fingerprint_kw))

    def load(self) -> dict | None:
        """The saved run, or ``None`` when there is none.

        Keys: ``phase`` (``"warmup"`` or ``"draws"``), ``done`` (sweeps done in
        it), ``keys`` and ``states`` (per chain, on the host), ``kept`` (the
        midpoint hook's warmup traces so far), ``midpoint_input`` (what the
        hook was given, once it has run), and ``parts`` (the retained draws so
        far, per chunk and chain).
        """
        path = self.dir / "state.pkl"
        if not path.exists():
            return None
        saved = _read(path)
        if saved["fingerprint"] != self.fp:
            raise ValueError(
                f"The checkpoint in {self.dir} belongs to another run; remove it."
            )
        self.n_files = saved["n_files"]
        saved["parts"] = [
            part
            for k in range(self.n_files)
            for part in _read(self.dir / f"draws_{k:05d}.pkl")
        ]
        return saved

    def add_draws(self, part) -> None:
        """Queue one chunk's retained draws (per chain) for the next save."""
        self._pending.append(_to_host(part))

    def save(self, *, phase, done, keys, states, kept, midpoint_input) -> None:
        """Write the run's state, if ``every`` seconds have passed since the last."""
        now = time.monotonic()
        if now - self._last < self.every:
            return
        self.dir.mkdir(parents=True, exist_ok=True)
        if self._pending:
            _write(self.dir / f"draws_{self.n_files:05d}.pkl", self._pending)
            self.n_files += 1
            self._pending = []
        _write(
            self.dir / "state.pkl",
            {
                "fingerprint": self.fp,
                "phase": phase,
                "done": done,
                "keys": [np.asarray(k) for k in keys],
                "states": [_to_host(s) for s in states],
                "kept": kept,
                "midpoint_input": midpoint_input,
                "n_files": self.n_files,
            },
        )
        self._last = now

    def finish(self) -> None:
        """Remove the directory of a run that has finished."""
        if self.dir.exists():
            shutil.rmtree(self.dir)
