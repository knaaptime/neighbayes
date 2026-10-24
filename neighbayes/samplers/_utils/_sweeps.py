"""The draw loop of the NumPy samplers written as a chain object.

A chain object holds one chain's state and has ``sweep(tuning)``, one Gibbs
sweep, and ``draw(store_latent)``, its current values by name;
:func:`run_sweeps` runs it and keeps the draws.  Two samplers' chains compose
into a joint one by sweeping each in turn.
"""

from __future__ import annotations

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
