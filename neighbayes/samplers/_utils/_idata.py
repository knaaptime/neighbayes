"""Assemble xarray.DataTree from Gibbs sampler output.

Model-agnostic: takes a dict of arrays and metadata, returns DataTree
with proper warmup/posterior split, log-likelihood, and observed data.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from ..._lazy_deps import az, xr


def gibbs_to_inference_data(
    *,
    posterior_samples: dict[str, np.ndarray],
    warmup_samples: dict[str, np.ndarray] | None = None,
    log_likelihood: dict[str, np.ndarray] | None = None,
    observed_data: dict[str, np.ndarray] | None = None,
    coords: dict[str, Sequence] | None = None,
    dims: dict[str, list[str]] | None = None,
    sample_stats: dict[str, np.ndarray] | None = None,
) -> xr.DataTree:
    """Build an ArviZ ``DataTree`` from Gibbs sampler chain output.

    Parameters
    ----------
    posterior_samples : dict
        ``{var_name: array of shape (chains, draws, ...)}``.
        The post-warmup draws for each parameter.
    warmup_samples : dict, optional
        Same structure as ``posterior_samples`` for the warmup phase.
        Stored in the ``warmup_posterior`` group.
    log_likelihood : dict, optional
        ``{obs_name: array of shape (chains, draws, n)}`` for
        LOO/WAIC computation.
    observed_data : dict, optional
        ``{obs_name: array of shape (n,)}`` observed data.
    coords : dict, optional
        ArviZ coordinate mappings, e.g. ``{"coefficient": ["x1", "x2"]}``.
    dims : dict, optional
        ArviZ dimension mappings, e.g. ``{"beta": ["coefficient"]}``.
    sample_stats : dict, optional
        Additional per-draw statistics (e.g., acceptance rates).

    Returns
    -------
    xr.DataTree
        With ``posterior``, ``warmup_posterior`` (if provided),
        ``log_likelihood`` (if provided), ``observed_data`` (if provided),
        and ``sample_stats`` (if provided) groups.
    """
    groups: dict[str, dict[str, np.ndarray]] = {"posterior": posterior_samples}
    if warmup_samples is not None:
        groups["warmup_posterior"] = warmup_samples
    if log_likelihood is not None:
        groups["log_likelihood"] = log_likelihood
    if sample_stats is not None:
        groups["sample_stats"] = sample_stats
    idata = az.from_dict(groups, coords=coords, dims=dims)

    # Observed data carries no chain/draw dimensions, so it is attached as its
    # own node rather than passed through from_dict's sample-dim conventions.
    if observed_data is not None:
        import xarray

        obs = {}
        for name, arr in observed_data.items():
            arr = np.asarray(arr)
            obs[name] = xarray.DataArray(
                arr, dims=["obs_dim"] if arr.ndim == 1 else None
            )
        idata["observed_data"] = xarray.DataTree(xarray.Dataset(obs))

    return idata
