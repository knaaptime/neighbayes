"""Progress bar manager for Gibbs samplers.

Uses rich.progress for terminal rendering with per-chain bars
showing iteration count, speed, and timing.

Following PyMC, the run is *announced before* the bars start so the user
knows what is about to happen; only the elapsed time and throughput are
reported afterwards.

There is no accept-rate column: every Gibbs block here is either a
conjugate draw or a slice step, both of which accept by construction, so
the column only ever showed ``--`` or a vacuous 100%.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Iterable
from contextlib import nullcontext
from multiprocessing import shared_memory
from typing import Any

import numpy as np
from rich.box import SIMPLE_HEAD
from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    Task,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Column, Table
from rich.theme import Theme

default_gibbs_theme = Theme(
    {
        "bar.complete": "#1764f4",
        "bar.finished": "#1764f4",
        "progress.remaining": "none",
        "progress.elapsed": "none",
    }
)


def _announcement(chains: int, tune: int, draws: int, model_type: str) -> str:
    """Line printed *before* the bars, telling the user what is starting.

    Mirrors PyMC, which announces the run up front; the previous behaviour
    printed this only on exit, by which point it could no longer inform
    anyone about what they had been waiting on.
    """
    total = chains * (tune + draws)
    plural = "s" if chains > 1 else ""
    label = f" ({model_type})" if model_type else ""
    return (
        f"Gibbs sampling{label}: {chains} chain{plural} "
        f"for {tune:,} tune and {draws:,} draw iterations "
        f"({chains} x {tune + draws:,} = {total:,} draws total)"
    )


def _timing_summary(total_draws: int, elapsed: float) -> str:
    """Line printed *after* the bars: elapsed time and throughput."""
    speed = total_draws / elapsed if elapsed > 0 else 0.0
    return f"Sampling took {elapsed:.0f}s ({speed:,.0f} draws/s)"


class _GibbsProgress(Progress):
    """Custom Progress subclass that renders column headers.

    Overrides ``make_tasks_table`` to return a proper ``Table`` with
    ``show_header=True`` instead of the default ``Table.grid()`` which
    hides headers.  Mimics PyMC's ``CustomProgress`` pattern.
    """

    def make_tasks_table(self, tasks: Iterable[Task]) -> Table:
        table_columns = (
            Column(no_wrap=True)
            if isinstance(_column, str)
            else _column.get_table_column().copy()
            for _column in self.columns
        )
        table = Table(
            *table_columns,
            padding=(0, 1),
            expand=self.expand,
            show_header=True,
            show_edge=True,
            box=SIMPLE_HEAD,
        )
        for task in tasks:
            if task.visible:
                table.add_row(
                    *(
                        column.format(task=task)
                        if isinstance(column, str)
                        else (
                            column(task)
                            if hasattr(column, "callbacks")
                            else column(task)
                        )
                        for column in self.columns
                    )
                )
        return table


class GibbsProgressBarManager:
    """Progress bar manager for Gibbs sampling.

    Shows per-chain rich progress bars with iteration count, speed, and
    timing, preceded by a one-line announcement of the run.

    The bar advances across the full chain (tune + draw) so notebook
    users see continuous progress rather than a long flat 0% period.

    Parameters
    ----------
    chains : int
        Number of chains.
    draws : int
        Post-warmup draws per chain.
    tune : int
        Warmup draws per chain.
    progressbar : bool, default True
        Show progress bars.
    model_type : str, default "sar"
        Model type (for display).
    """

    def __init__(
        self,
        chains: int,
        draws: int,
        tune: int,
        progressbar: bool = True,
        model_type: str = "sar",
    ):
        self.chains = chains
        self.draws = draws
        self.tune = tune
        self.total = draws + tune
        self._show = progressbar
        self.model_type = model_type

        # Per-chain start times for accurate speed reporting.
        # Without this, sequential chains inherit elapsed time from
        # task creation (not chain start), making later chains appear
        # slower than they really are.
        self._chain_start_times: dict[int, float] = {}

        # Throttle rich updates so per-iteration overhead stays
        # negligible even when samplers exceed thousands of iters/sec.
        # We aim for ~200 redraws per chain plus a forced update on
        # the final iteration of each phase.
        self._update_every = max(1, self.total // 200)

        if self._show:
            self._progress = _GibbsProgress(
                BarColumn(bar_width=None, table_column=Column("Progress", ratio=2)),
                TextColumn(
                    "{task.fields[iter_count]:>5d}/{task.fields[total_iters]}",
                    table_column=Column("Iter", ratio=2),
                ),
                TextColumn(
                    "{task.fields[speed]}",
                    table_column=Column("Speed", ratio=2),
                ),
                TimeElapsedColumn(table_column=Column("Elapsed", ratio=1)),
                TimeRemainingColumn(table_column=Column("Remaining", ratio=1)),
                console=Console(theme=default_gibbs_theme),
                expand=True,
            )
            self._tasks: list[Any] = []
        else:
            self._progress = nullcontext()
            self._tasks = []

    def __enter__(self):
        if self._show:
            # Announce *before* the bars start, as PyMC does, so the user
            # knows what is running while it runs rather than learning it
            # only once the bars have finished.
            self._progress.console.print(
                _announcement(self.chains, self.tune, self.draws, self.model_type)
            )
            self._progress.__enter__()
            # Add one task per chain
            for c in range(self.chains):
                task_id = self._progress.add_task(
                    f"Chain {c + 1}",
                    total=self.total,
                    phase="tune",
                    iter_count=0,
                    total_iters=self.total,
                    speed="--",
                )
                self._tasks.append(task_id)
        return self

    def __exit__(self, *args):
        if self._show:
            # Timing only — the run itself was announced in __enter__.
            total_draws = self.chains * (self.draws + self.tune)
            if self._tasks:
                elapsed = self._progress.tasks[self._tasks[0]].elapsed or 0.0
            else:
                elapsed = 0.0
            summary = _timing_summary(total_draws, elapsed)
            # Live.stop() skips a final refresh in Jupyter, so force one
            # before exit to render fully completed bars.
            self._progress.refresh()
            # In Jupyter, print inside the ipy_widget before exiting
            # so the summary appears in the same output block as the
            # progress bar (avoids creating a separate rich block).
            ipy_widget = getattr(self._progress.live, "ipy_widget", None)
            if ipy_widget is not None:
                with ipy_widget:
                    self._progress.console.print(summary)
                self._progress.__exit__(*args)
            else:
                self._progress.__exit__(*args)
                self._progress.console.print(summary)

    def start_chain(self, chain_idx: int) -> None:
        """Record the start time for a chain.

        Must be called just before the chain begins sampling so that
        the speed column reflects actual per-chain sampling time rather
        than wall-clock time since task creation.

        Parameters
        ----------
        chain_idx : int
            Chain index (0-based).
        """
        self._chain_start_times[chain_idx] = time.monotonic()

    def update(
        self,
        chain_idx: int,
        iteration: int,
        tuning: bool,
    ):
        """Update progress for a chain.

        Parameters
        ----------
        chain_idx : int
            Chain index (0-based).
        iteration : int
            Current iteration (0-based, counting both tune and draw).
        tuning : bool
            Whether in warmup phase.
        """
        if not self._show:
            return

        iter1 = iteration + 1
        is_phase_end = iter1 == self.tune or iter1 == self.total
        if iter1 % self._update_every != 0 and not is_phase_end:
            return

        task_id = self._tasks[chain_idx]
        phase = "tune" if tuning else "draw"

        chain_start = self._chain_start_times.get(chain_idx)
        if chain_start is not None:
            elapsed = time.monotonic() - chain_start
        else:
            elapsed = self._progress.tasks[task_id].elapsed or 0.0
        speed_str = f"{iter1 / elapsed:.1f} it/s" if elapsed > 0 else "--"

        self._progress.update(
            task_id,
            completed=iter1,
            phase=phase,
            iter_count=iter1,
            speed=speed_str,
        )

    def refresh(self):
        """Force a refresh of the progress bar display."""
        if self._show:
            self._progress.refresh()


# ---------------------------------------------------------------------------
# Parallel progress bar support
# ---------------------------------------------------------------------------


class _SharedCounterReporter:
    """Picklable shared-memory progress reporter for worker processes.

    Implements the same interface as :class:`GibbsProgressBarManager`
    (``update``, ``start_chain``, ``refresh``)
    but writes per-iteration progress into a shared-memory block
    instead of sending IPC messages.  Each chain owns two ``int64``
    slots ``[iteration, tuning_flag]`` in a flat ``(n_chains, 2)``
    array.  Workers perform plain stores (no locks — single writer
    per slot, single reader on the main process), so ``update()``
    costs two memory writes rather than a pickle + syscall.

    Parameters
    ----------
    shm_name : str
        Name of the :class:`multiprocessing.shared_memory.SharedMemory`
        block allocated by the parent process.
    chain_id : int
        0-based chain index that this reporter represents.
    n_chains : int
        Total number of chains (needed to compute the buffer shape on
        the worker side).
    """

    def __init__(self, shm_name: str, chain_id: int, n_chains: int):
        self._shm_name = shm_name
        self._chain_id = chain_id
        self._n_chains = n_chains
        # Lazy-opened on first use in the worker process so that the
        # reporter pickles cleanly (open SharedMemory handles are not
        # picklable across spawn/loky).
        self._shm: shared_memory.SharedMemory | None = None
        self._buf: np.ndarray | None = None

    # -- pickle protocol --------------------------------------------------

    def __reduce__(self):
        return (
            self.__class__,
            (self._shm_name, self._chain_id, self._n_chains),
        )

    # -- internals --------------------------------------------------------

    def _ensure_open(self) -> None:
        if self._buf is None:
            self._shm = shared_memory.SharedMemory(name=self._shm_name)
            self._buf = np.ndarray(
                (self._n_chains, 2), dtype=np.int64, buffer=self._shm.buf
            )

    def __del__(self):
        # Best-effort close of the worker-side handle.  The parent
        # owns the lifetime of the underlying segment (unlink).
        self.close()

    def close(self) -> None:
        """Explicitly close the shared-memory handle.

        Call this when the reporter is no longer needed (e.g. at the
        end of a chain) to free the file descriptor promptly.  In loky
        worker processes, ``__del__`` is not guaranteed to fire, so
        relying on it alone leaks FDs across many parallel calls.
        """
        shm = getattr(self, "_shm", None)
        if shm is not None:
            try:
                shm.close()
            except (BufferError, OSError):
                pass  # a view still holds the buffer, or it is already gone
            self._shm = None
            self._buf = None

    # -- same interface as GibbsProgressBarManager -----------------------

    def update(
        self,
        chain_idx: int,
        iteration: int,
        tuning: bool,
    ):
        # Two int64 stores.  No syscalls, no locks, ~10 ns.
        self._ensure_open()
        buf = self._buf
        assert buf is not None
        buf[self._chain_id, 0] = iteration + 1
        buf[self._chain_id, 1] = 1 if tuning else 0

    def start_chain(self, chain_idx: int):
        # No-op: the renderer infers the chain start from the first
        # non-zero iteration count in its periodic poll.
        self._ensure_open()

    def refresh(self):
        # No-op: the renderer's polling thread refreshes the display.
        pass


class _ParallelProgressRenderer:
    """Main-process renderer that polls shared memory and updates rich bars.

    Owns a :class:`_GibbsProgress` instance with one task per chain.
    A daemon thread calls :meth:`poll` in a loop to snapshot a
    shared-memory counter block written by worker processes and update
    the corresponding tasks.

    Parameters
    ----------
    n_chains : int
        Number of chains.
    draws : int
        Post-warmup draws per chain.
    tune : int
        Warmup draws per chain.
    model_type : str
        Model type (for display).
    """

    def __init__(
        self,
        n_chains: int,
        draws: int,
        tune: int,
        model_type: str = "sar",
    ):
        self.n_chains = n_chains
        self.draws = draws
        self.tune = tune
        self.model_type = model_type

        # Per-chain start times for speed reporting (populated on first
        # observed iteration during polling).
        self._chain_start_times: dict[int, float] = {}

        # Create the rich Progress instance (same layout as sequential)
        self._progress = _GibbsProgress(
            BarColumn(bar_width=None, table_column=Column("Progress", ratio=2)),
            TextColumn(
                "{task.fields[iter_count]:>5d}/{task.fields[total_iters]}",
                table_column=Column("Iter", ratio=2),
            ),
            TextColumn(
                "{task.fields[speed]}",
                table_column=Column("Speed", ratio=2),
            ),
            TimeElapsedColumn(table_column=Column("Elapsed", ratio=1)),
            TimeRemainingColumn(table_column=Column("Remaining", ratio=1)),
            console=Console(theme=default_gibbs_theme),
            expand=True,
        )
        self._tasks: list[Any] = []

    def __enter__(self):
        # Announce before the bars start (see GibbsProgressBarManager).
        self._progress.console.print(
            _announcement(self.n_chains, self.tune, self.draws, self.model_type)
        )
        self._progress.__enter__()
        for c in range(self.n_chains):
            task_id = self._progress.add_task(
                f"Chain {c + 1}",
                total=self.draws + self.tune,
                phase="tune",
                iter_count=0,
                total_iters=self.draws + self.tune,
                speed="--",
            )
            self._tasks.append(task_id)
        return self

    def __exit__(self, *args):
        # Timing only — the run itself was announced in __enter__.
        total_draws = self.n_chains * (self.draws + self.tune)
        if self._tasks:
            elapsed = self._progress.tasks[self._tasks[0]].elapsed or 0.0
        else:
            elapsed = 0.0
        summary = _timing_summary(total_draws, elapsed)
        # Live.stop() skips a final refresh in Jupyter, so force one
        # before exit to render fully completed bars.
        self._progress.refresh()
        # In Jupyter, print inside the ipy_widget before exiting
        # so the summary appears in the same output block as the
        # progress bar (avoids creating a separate rich block).
        ipy_widget = getattr(self._progress.live, "ipy_widget", None)
        if ipy_widget is not None:
            with ipy_widget:
                self._progress.console.print(summary)
            self._progress.__exit__(*args)
        else:
            self._progress.__exit__(*args)
            self._progress.console.print(summary)

    def _apply_snapshot(
        self,
        snapshot: np.ndarray,
        last_iter: np.ndarray,
    ) -> None:
        """Apply a shared-memory snapshot to the rich progress bars.

        Updates each chain's task to absolute position ``draw_iter``
        rather than advancing by a delta, so missed polls are
        self-correcting.

        Parameters
        ----------
        snapshot : ndarray, shape (n_chains, 2), dtype int64
            Copy of the shared-memory counter block.  Column 0 is the
            1-based current iteration (0 = not started); column 1 is
            the tuning flag (1 = tuning, 0 = sampling).
        last_iter : ndarray, shape (n_chains,), dtype int64
            Per-chain last-observed iteration count.  Mutated in place.
        """
        now = time.monotonic()
        for c in range(self.n_chains):
            cur_iter = int(snapshot[c, 0])
            cur_tuning = int(snapshot[c, 1])

            if cur_iter == 0 and last_iter[c] == 0:
                continue  # chain has not started yet

            # Record start time on first observed iteration.
            if last_iter[c] == 0 and cur_iter > 0:
                self._chain_start_times[c] = now

            task_id = self._tasks[c]

            chain_start = self._chain_start_times.get(c)
            if chain_start is not None and now > chain_start:
                speed_str = f"{cur_iter / (now - chain_start):.1f} it/s"
            else:
                speed_str = "--"

            phase = "tune" if cur_tuning else "draw"
            self._progress.update(
                task_id,
                completed=cur_iter,
                phase=phase,
                iter_count=cur_iter,
                speed=speed_str,
                refresh=True,
            )

            last_iter[c] = cur_iter

    def poll(
        self,
        shm_name: str,
        stop_event: threading.Event,
        interval: float = 0.1,
    ) -> None:
        """Poll the shared-memory counter block until *stop_event* is set.

        Intended to be run in a daemon thread.  Snapshots the buffer at
        ``interval`` seconds and updates each chain's rich task to its
        current absolute position.

        Parameters
        ----------
        shm_name : str
            Name of the shared-memory block allocated by the parent.
        stop_event : threading.Event
            When set, the loop performs one final snapshot and exits.
        interval : float, default 0.1
            Polling interval in seconds (10 Hz default).
        """
        shm = shared_memory.SharedMemory(name=shm_name)
        try:
            buf = np.ndarray((self.n_chains, 2), dtype=np.int64, buffer=shm.buf)
            last_iter = np.zeros(self.n_chains, dtype=np.int64)

            while not stop_event.wait(interval):
                self._apply_snapshot(buf.copy(), last_iter)
            # One final snapshot after stop to catch any iterations
            # completed between the last poll and the stop signal.
            self._apply_snapshot(buf.copy(), last_iter)
            self._progress.refresh()
        finally:
            shm.close()
