"""Data-generating process for the spatial multilevel model.

Units sit on a square grid; each level's groups are square blocks of the
level below, so the levels nest, and each level's graph is the rook adjacency
of its own blocks, which crosses parent boundaries.  Effects are generated
from the top down, each entering the equation of the level below.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.linalg import splu

from .utils import _hetero_scale, ensure_rng, rook_grid_weights


def _filter(W: sp.csr_matrix, coef: float, rhs: np.ndarray) -> np.ndarray:
    A = (sp.eye(W.shape[0], format="csc") - coef * W).tocsc()
    return splu(A).solve(rhs)


def simulate_spatial_multilevel(
    n_side: int = 24,
    blocks: Sequence[int] = (4, 2),
    processes: Sequence[str] = ("lag", "lag", "lag"),
    rhos: Sequence[float] = (0.4, 0.5, 0.4),
    betas: Sequence[Sequence[float]] | None = None,
    sigmas: Sequence[float] = (1.0, 0.5, 0.5),
    gdf=None,
    err_hetero: bool = False,
    rng: np.random.Generator | None = None,
    seed: int | None = None,
) -> dict:
    r"""Simulate nested levels with a graph and a process at each.

    Level 0 is an ``n_side × n_side`` grid of units.  Level ℓ's groups are
    ``blocks[ℓ−1] × blocks[ℓ−1]`` blocks of level ℓ−1's groups, so level ℓ
    has ``(n_side / (b_1⋯b_ℓ))²`` groups.  Every level's graph is the
    row-standardized rook adjacency of its own grid.  From the top down,

    .. math::

        \theta_\ell = S_\ell(X_\ell\beta_\ell + \Delta_\ell\theta_{\ell+1}
                      + \sigma_\ell\varepsilon_\ell)

    for a lag (:math:`S_\ell = (I-\rho_\ell W_\ell)^{-1}`); an error process
    filters only the innovation, and ``"none"`` filters nothing.
    :math:`y = \theta_0`.

    Parameters
    ----------
    n_side : int, default 24
        Side of the unit grid; must be divisible by ``prod(blocks)``.
    blocks : sequence of int, default (4, 2)
        Block side of each upper level, in groups of the level below.
    processes : sequence of {"lag", "error", "none"}
        One per level, the units first.
    rhos : sequence of float
        The autoregressive parameter of each level (ignored for ``"none"``).
    betas : sequence of sequences, optional
        Coefficients per level.  The units' include the intercept first;
        upper levels have none.  Default: ``[1, 1, -1]`` at the units and
        ``[0.5]`` above.
    sigmas : sequence of float
        Innovation sd per level.
    gdf : None
        Not supported (the levels are built on a grid); accepted for the
        common simulator signature.
    err_hetero : bool, default False
        Heteroskedastic unit innovations, ``σ_0 sqrt(1 + ‖x_i‖²)``.
    rng, seed
        Randomness.

    Returns
    -------
    dict
        ``y``, ``X`` (units' design), ``W`` / ``W_sparse`` (units' graph),
        ``levels`` (per upper level: ``X``, ``W``, ``W_sparse``, ``parent`` —
        the index of each lower row's group — and ``theta``), ``frames``
        (one DataFrame per level for formula use: the units with ``y``,
        ``x1``… and a key ``g1``, ``g2``… for every level above; each upper
        level with its key, ``z1``… and the key of its parent), and
        ``params``.
    """
    if gdf is not None:
        raise NotImplementedError(
            "simulate_spatial_multilevel builds nested grids; pass n_side."
        )
    rng = ensure_rng(rng, seed)
    L = len(blocks)
    if not (len(processes) == len(rhos) == len(sigmas) == L + 1):
        raise ValueError("processes, rhos and sigmas need one entry per level.")
    if n_side % int(np.prod(blocks)):
        raise ValueError("n_side must be divisible by the product of blocks.")
    if betas is None:
        betas = [[1.0, 1.0, -1.0]] + [[0.5]] * L
    betas = [np.asarray(b, dtype=np.float64) for b in betas]

    sides = [n_side]
    for b in blocks:
        sides.append(sides[-1] // b)
    graphs = [rook_grid_weights(s) for s in sides]

    # parent of each level-ℓ cell in level ℓ+1's grid
    parents = []
    for ell, b in enumerate(blocks):
        s = sides[ell]
        r, c = np.divmod(np.arange(s * s), s)
        parents.append((r // b) * sides[ell + 1] + c // b)

    designs = []
    for ell in range(L + 1):
        J = sides[ell] ** 2
        k = len(betas[ell]) - (1 if ell == 0 else 0)
        Z = rng.standard_normal((J, k))
        designs.append(np.column_stack([np.ones(J), Z]) if ell == 0 else Z)

    thetas: list[np.ndarray] = [None] * (L + 1)
    for ell in range(L, -1, -1):
        J = sides[ell] ** 2
        W = graphs[ell][0]
        mean = designs[ell] @ betas[ell]
        if ell < L:
            mean = mean + thetas[ell + 1][parents[ell]]
        sd = sigmas[ell]
        if ell == 0 and err_hetero:
            sd = _hetero_scale(designs[0], sigmas[0])
        eps = sd * rng.standard_normal(J)
        proc = processes[ell]
        if proc == "lag":
            thetas[ell] = _filter(W, rhos[ell], mean + eps)
        elif proc == "error":
            thetas[ell] = mean + _filter(W, rhos[ell], eps)
        else:
            thetas[ell] = mean + eps

    # Frames for formula use, with keys g1, g2, ...
    ancestors = [np.arange(n_side * n_side)]
    for ell in range(L):
        ancestors.append(parents[ell][ancestors[-1]])
    units = pd.DataFrame({"y": thetas[0]})
    for j in range(1, designs[0].shape[1]):
        units[f"x{j}"] = designs[0][:, j]
    for ell in range(1, L + 1):
        units[f"g{ell}"] = ancestors[ell]
    frames = [units]
    for ell in range(1, L + 1):
        df = pd.DataFrame({f"g{ell}": np.arange(sides[ell] ** 2)})
        for j in range(designs[ell].shape[1]):
            df[f"z{j + 1}"] = designs[ell][:, j]
        if ell < L:
            df[f"g{ell + 1}"] = parents[ell]
        frames.append(df)

    levels = [
        {
            "X": designs[ell],
            "W": graphs[ell][1],
            "W_sparse": graphs[ell][0],
            "parent": parents[ell] if ell < L else None,
            "theta": thetas[ell],
        }
        for ell in range(1, L + 1)
    ]
    return {
        "y": thetas[0],
        "X": designs[0],
        "W": graphs[0][1],
        "W_sparse": graphs[0][0],
        "parent": parents[0],
        "levels": levels,
        "frames": frames,
        "params": {
            "processes": tuple(processes),
            "rhos": tuple(rhos),
            "betas": betas,
            "sigmas": tuple(sigmas),
        },
    }
