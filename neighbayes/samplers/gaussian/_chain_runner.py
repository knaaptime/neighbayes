"""Parallel chain dispatch for Gibbs samplers.

Supports sequential execution with rich progress bars and process-based
parallelism via ``joblib.Parallel`` (with per-chain progress bars fed via a
shared-memory counter block).  Warnings a worker raises are relayed to the
parent, which raises each distinct one once.
"""

from __future__ import annotations

import gc
import logging
import threading
import warnings
from multiprocessing import shared_memory
from typing import Callable

import numpy as np

_log = logging.getLogger(__name__)


def _call_recording(fn, *args, **kwargs):
    """``(fn(*args, **kwargs), warnings)``, the warnings as picklable pairs.

    A loky worker's warnings go to its own stderr, which a notebook never
    shows; returning them lets :func:`run_chains` raise them in the parent,
    under the parent's warning filters.
    """
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("default")
        result = fn(*args, **kwargs)
    return result, [(w.category, str(w.message)) for w in caught]


def _unpack_relayed(outputs) -> list:
    """The chains' results, raising each distinct relayed warning once."""
    seen = set()
    for _, records in outputs:
        for category, text in records:
            if (category, text) in seen:
                continue
            seen.add((category, text))
            # stacklevel 3: this function, run_chains, then run_chains' caller.
            warnings.warn(text, category, stacklevel=3)
    return [result for result, _ in outputs]


def run_chains(
    chain_fn: Callable,
    n_chains: int,
    seeds: list[int] | None = None,
    n_jobs: int = -1,
    progressbar: bool = True,
    parallel: bool = False,
    draws: int = 2000,
    tune: int = 1000,
    model_type: str = "sar",
    timeout: float | None = 600,
    reuse_workers: bool = False,
) -> list[dict]:
    """Run n_chains independent Gibbs chains.

    Parameters
    ----------
    chain_fn : callable
        Function with signature
        ``(chain_id, seed, progress_manager=None, chain_id=0) -> dict``
        that runs a single chain and returns a dict of arrays.
    n_chains : int
        Number of independent chains.
    seeds : list of int, optional
        Per-chain seeds. If None, derived from a parent
        ``numpy.random.SeedSequence``.
    n_jobs : int, default -1
        Number of parallel workers when ``parallel=True``.
        ``-1`` uses all CPUs. Ignored when ``parallel=False``.
    progressbar : bool, default True
        Show per-chain progress bars.  In sequential mode
        (``parallel=False``), uses ``rich`` progress bars directly.
        In parallel mode (``parallel=True``), uses ``rich`` progress
        bars updated via a multiprocessing Queue.
    parallel : bool, default False
        If True, run chains in parallel via ``joblib.Parallel``.
        If False, run chains sequentially with progress bars.
    draws : int, default 2000
        Post-warmup draws per chain (used for progress bar setup).
    tune : int, default 1000
        Warmup draws per chain (used for progress bar setup).
    model_type : str, default "sar"
        Model type (used for progress bar display).
    reuse_workers : bool, default False
        Skip the worker-pool teardown before spawning.  Set this only for a
        second call that immediately follows a first with the same
        ``n_chains``/``n_jobs`` — the workers already carry the right
        thread-limiting environment, so respawning them costs a process launch
        and a re-pickle of the closure's data for nothing.  The warmup-refit
        path uses it to split one run into two phases without paying twice.
    timeout : float or None, default None
        Maximum wall-clock seconds to wait for **all** chains to
        finish when ``parallel=True``.  If any worker has not
        returned by this deadline, a :class:`TimeoutError` is raised.
        ``None`` waits indefinitely (backward-compatible default).
        Ignored when ``parallel=False``.

    Returns
    -------
    list of dict
        One dict per chain, each containing parameter trace arrays.
    """
    if seeds is None:
        from .._utils._seeds import spawn_chain_seeds

        seeds = spawn_chain_seeds(None, n_chains)
    elif len(seeds) != n_chains:
        raise ValueError(
            f"len(seeds) must equal n_chains, got {len(seeds)} != {n_chains}."
        )

    if parallel:
        # Process-based parallelism via joblib (handles closures)
        import os

        from joblib import Parallel, delayed, parallel_config

        # Cap workers at n_chains: launching more processes than chains
        # wastes spawn/import cost (the extra workers sit idle) and risks
        # BLAS oversubscription.
        if n_jobs is None or n_jobs < 0:
            n_workers = min(n_chains, os.cpu_count() or 1)
        else:
            n_workers = min(n_jobs, n_chains)

        # Limit BLAS/OpenMP threads per worker to avoid oversubscription.
        # With n_workers processes each using cpu_count threads, total
        # threads would be n_workers * cpu_count — far exceeding actual
        # cores.  Setting inner_max_num_threads = cpu_count // n_workers
        # gives each worker a proportional share, keeping total threads
        # near cpu_count.  This prevents one worker from starving others
        # while still allowing multi-threaded BLAS within each worker.
        cpu_count = os.cpu_count() or 1
        inner_threads = max(1, cpu_count // n_workers)

        # Kill any existing loky workers so that new ones are spawned
        # with the correct thread-limiting environment variables.
        # Without this, reused workers keep their original (unlimited)
        # thread settings, which can cause BLAS deadlocks on macOS
        # with Apple Accelerate after many parallel calls.
        if not reuse_workers:
            from joblib.externals.loky import get_reusable_executor

            get_reusable_executor(reuse=True).shutdown(wait=True)

            # Force garbage collection before spawning workers.  Each fit()
            # call creates CholmodFactor objects that hold C-level resources
            # (CHOLMOD common structs, SuiteSparse memory).  Python's GC may
            # not collect these promptly, and accumulated C resources can
            # cause issues after many calls.
            gc.collect()

        if progressbar:
            # Per-chain progress bars via shared-memory counters + rich.
            # Workers do two int64 stores per iteration (no IPC); a
            # daemon thread on the main process polls the buffer at
            # 10 Hz and updates the rich tasks.
            from .._utils._progress import (
                _ParallelProgressRenderer,
                _SharedCounterReporter,
            )

            # Layout: (n_chains, 2) int64 — [iteration, tuning_flag].
            nbytes = n_chains * 2 * 8
            shm = shared_memory.SharedMemory(create=True, size=nbytes)
            try:
                # Initialise: iteration=0 (not started), tuning_flag=1.
                init_buf = np.ndarray((n_chains, 2), dtype=np.int64, buffer=shm.buf)
                init_buf[:, 0] = 0
                init_buf[:, 1] = 1
                del init_buf

                renderer = _ParallelProgressRenderer(
                    n_chains=n_chains,
                    draws=draws,
                    tune=tune,
                    model_type=model_type,
                )
                reporters = [
                    _SharedCounterReporter(shm.name, c, n_chains)
                    for c in range(n_chains)
                ]

                stop_event = threading.Event()
                poll_thread = threading.Thread(
                    target=renderer.poll,
                    args=(shm.name, stop_event),
                    kwargs={"interval": 0.1},
                    daemon=True,
                )

                with renderer:
                    poll_thread.start()
                    try:
                        with parallel_config(
                            backend="loky", inner_max_num_threads=inner_threads
                        ):
                            # Wrap each chain call to ensure the shared-memory
                            # reporter is closed in the worker process, even if
                            # the chain raises an exception.
                            def _run_with_cleanup(cid, seed, pm):
                                try:
                                    return _call_recording(
                                        chain_fn,
                                        cid,
                                        seed,
                                        progress_manager=pm,
                                        chain_id_kw=cid,
                                    )
                                finally:
                                    if pm is not None and hasattr(pm, "close"):
                                        pm.close()

                            results = Parallel(n_jobs=n_workers, timeout=timeout)(
                                delayed(_run_with_cleanup)(
                                    chain_id, seed, reporters[chain_id]
                                )
                                for chain_id, seed in enumerate(seeds)
                            )
                    finally:
                        stop_event.set()
                        poll_thread.join(timeout=5.0)
                return _unpack_relayed(results)
            finally:
                shm.close()
                try:
                    shm.unlink()
                except FileNotFoundError:
                    pass
        else:
            # No progress bar — silent parallel execution.
            with parallel_config(backend="loky", inner_max_num_threads=inner_threads):
                results = Parallel(n_jobs=n_workers, timeout=timeout)(
                    delayed(_call_recording)(chain_fn, chain_id, seed)
                    for chain_id, seed in enumerate(seeds)
                )
            return _unpack_relayed(results)

    # Sequential execution with progress bars
    from .._utils._progress import GibbsProgressBarManager

    with GibbsProgressBarManager(
        chains=n_chains,
        draws=draws,
        tune=tune,
        progressbar=progressbar,
        model_type=model_type,
    ) as pm:
        results = []
        for chain_id, seed in enumerate(seeds):
            if pm is not None:
                pm.start_chain(chain_id)
            result = chain_fn(
                chain_id,
                seed,
                progress_manager=pm,
                chain_id_kw=chain_id,
            )
            results.append(result)
        return results
