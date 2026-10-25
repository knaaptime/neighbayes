"""The draw loop of the NumPy samplers written as a chain object.

A chain object holds one chain's state and has ``sweep(tuning)``, one Gibbs
sweep, and ``draw(store_latent)``, its current values by name;
:func:`run_sweeps` runs it and keeps the draws.  Two samplers' chains compose
into a joint one by sweeping each in turn.

:func:`run_sweeps_with_midpoint` runs several chains with an adaptation halfway
through warmup on what every chain did there: the NumPy counterpart of the hook
in :func:`._jax_utils.run_chains_chunked`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np


def run_sweeps(
    chain,
    *,
    draws: int,
    tune: int,
    thin: int = 1,
    chain_id: int = 0,
    progress_manager=None,
    store_latent: bool = False,
) -> dict:
    """``tune`` warmup sweeps, then ``draws``, keeping every ``thin``-th.

    Returns each of ``chain.draw``'s values stacked over the kept draws.
    """
    n_keep = draws // thin
    kept = {
        name: np.empty((n_keep,) + np.shape(value))
        for name, value in chain.draw(store_latent).items()
    }
    for it in range(tune + draws):
        chain.sweep(tuning=it < tune)
        if it >= tune and (it - tune) % thin == 0:
            idx = (it - tune) // thin
            if idx < n_keep:
                for name, value in chain.draw(store_latent).items():
                    kept[name][idx] = value
        if progress_manager is not None:
            progress_manager.update(chain_id, it, tuning=it < tune)
    return kept


@dataclass(frozen=True)
class SweepMidpoint:
    """An adaptation :func:`run_sweeps_with_midpoint` makes after ``tune // 2`` sweeps.

    ``keep(chain)`` is what to keep from each warmup sweep, a tuple of floats.
    ``adapt(kept, *, draws, chains)`` receives them as ``(chains, tune // 2,
    len(keep))``, chains apart, and returns ``apply(chain)``, the change made
    to every chain before the run resumes, or ``None``.
    """

    keep: Callable
    adapt: Callable


def run_sweeps_with_midpoint(
    make_chain,
    *,
    seeds,
    draws: int,
    tune: int,
    thin: int = 1,
    midpoint: Optional[SweepMidpoint] = None,
    store_latent: bool = False,
    finish: Optional[Callable] = None,
    n_jobs: int = -1,
    progressbar: bool = True,
    model_type: str = "gibbs",
) -> list[dict]:
    """:func:`run_sweeps` on one chain per seed, with ``midpoint`` halfway through warmup.

    ``make_chain(rng)`` builds a chain object.  With a midpoint, and ``tune >=
    2``, every chain runs its first ``tune // 2`` warmup sweeps and comes back
    with what ``midpoint.keep`` recorded; ``midpoint.adapt`` decides on all of
    it at once, and each chain resumes where it stopped, on its own random
    stream.  A midpoint that changes nothing leaves the draws as they would
    have been without it.  ``finish(chain)``, what a chain accumulated beyond
    its draws, is merged into its result.
    """
    from ..gaussian._chain_runner import run_chains

    chains = len(seeds)
    parallel = n_jobs != 1
    tune_a = tune // 2 if midpoint is not None and tune >= 2 else 0
    started = [None] * chains
    if tune_a:
        # Only ``keep`` goes to the workers; ``adapt`` may hold large state.
        keep = midpoint.keep

        def _warm(chain_id, seed, progress_manager=None, chain_id_kw=None):
            chain = make_chain(np.random.default_rng(seed))
            kept = []
            for _ in range(tune_a):
                chain.sweep(tuning=True)
                kept.append(keep(chain))
            return {"chain": chain, "kept": np.asarray(kept, dtype=np.float64)}

        warm = run_chains(
            chain_fn=_warm,
            n_chains=chains,
            seeds=seeds,
            n_jobs=n_jobs,
            progressbar=False,
            parallel=parallel,
            draws=tune_a,
            tune=0,
            model_type=model_type,
        )
        started = [w["chain"] for w in warm]
        apply = midpoint.adapt(
            np.stack([w["kept"] for w in warm]), draws=draws, chains=chains
        )
        if apply is not None:
            for chain in started:
                apply(chain)

    def _run(chain_id, seed, progress_manager=None, chain_id_kw=None):
        chain = started[chain_id]
        if chain is None:
            chain = make_chain(np.random.default_rng(seed))
        out = run_sweeps(
            chain,
            draws=draws,
            tune=tune - tune_a,
            thin=thin,
            chain_id=chain_id_kw if chain_id_kw is not None else chain_id,
            progress_manager=progress_manager,
            store_latent=store_latent,
        )
        if finish is not None:
            out.update(finish(chain))
        return out

    return run_chains(
        chain_fn=_run,
        n_chains=chains,
        seeds=seeds,
        n_jobs=n_jobs,
        progressbar=progressbar,
        parallel=parallel,
        draws=draws,
        tune=tune - tune_a,
        model_type=model_type,
        # The warm phase just spawned this pool with the same worker count.
        reuse_workers=bool(tune_a),
    )


def probe_sweep_midpoint(arms) -> SweepMidpoint:
    """The :class:`SweepMidpoint` that sizes stochastic log-determinant probe pools.

    ``arms`` lists ``(adapter, rho_of, set_logdet)``: a
    :class:`~neighbayes._logdet._probe_check.WarmupProbes`, ``rho_of(chain)``
    a spatial parameter, and ``set_logdet(chain, fn)`` installing its
    log-determinant of one period.  Arms that share an adapter, two spatial
    parameters on one W, size one pool for both.  The first half of the kept
    warmup is transient and dropped; a pool that grows replaces the evaluator
    in every chain.
    """
    groups: dict = {}
    for i, (adapter, _, _) in enumerate(arms):
        groups.setdefault(id(adapter), (adapter, []))[1].append(i)
    rho_ofs = [rho_of for _, rho_of, _ in arms]

    def keep(chain):
        return tuple(float(rho_of(chain)) for rho_of in rho_ofs)

    def adapt(kept, *, draws, chains):
        settled = kept[:, kept.shape[1] // 2 :]
        changes = []
        for adapter, idx in groups.values():
            by_param = np.moveaxis(settled[..., idx], -1, 0)
            if adapter.adapt(by_param, draws=draws, chains=chains) is not None:
                fn = adapter.period_fn()
                changes.extend((arms[i][2], fn) for i in idx)
        if not changes:
            return None

        def apply(chain):
            for set_logdet, fn in changes:
                set_logdet(chain, fn)

        return apply

    return SweepMidpoint(keep=keep, adapt=adapt)
