"""Panel random-effects style DGP functions.

These generators share the same observation equations as FE variants but are
named for compatibility with random-effects model tests and workflows.
"""

from __future__ import annotations

from .panel_fe import (
    simulate_panel_ols_fe,
    simulate_panel_sar_fe,
    simulate_panel_sem_fe,
)


def simulate_panel_ols_re(*, err_hetero: bool = False, **kwargs) -> dict:
    """Simulate OLS random-effects panel style data.

    Parameters
    ----------
    err_hetero : bool, default=False
        If True, generate heteroskedastic innovations. Forwarded to
        :func:`neighbayes.dgp.panel_fe.simulate_panel_ols_fe`.
    **kwargs
        Forwarded to :func:`neighbayes.dgp.panel_fe.simulate_panel_ols_fe`.

    Returns
    -------
    dict
        Simulation output with keys ``y``, ``X``, ``unit``, ``time``,
        ``W_sparse``, ``W_graph``, and ``params_true``.

    Notes
    -----
    Uses the same DGP form as :func:`simulate_panel_ols_fe` with unit effects
    sampled from ``N(0, sigma_alpha^2)``.
    """
    return simulate_panel_ols_fe(err_hetero=err_hetero, **kwargs)


def simulate_panel_sar_re(*, err_hetero: bool = False, **kwargs) -> dict:
    """Simulate SAR random-effects panel style data.

    Parameters
    ----------
    err_hetero : bool, default=False
        If True, generate heteroskedastic innovations. Forwarded to
        :func:`neighbayes.dgp.panel_fe.simulate_panel_sar_fe`.
    **kwargs
        Forwarded to :func:`neighbayes.dgp.panel_fe.simulate_panel_sar_fe`.

    Returns
    -------
    dict
        Simulation output with keys ``y``, ``X``, ``unit``, ``time``,
        ``W_sparse``, ``W_graph``, and ``params_true``.

    Notes
    -----
    Uses the same DGP form as :func:`simulate_panel_sar_fe` with unit effects.
    """
    return simulate_panel_sar_fe(err_hetero=err_hetero, **kwargs)


def simulate_panel_sem_re(*, err_hetero: bool = False, **kwargs) -> dict:
    """Simulate SEM random-effects panel style data.

    Parameters
    ----------
    err_hetero : bool, default=False
        If True, generate heteroskedastic innovations. Forwarded to
        :func:`neighbayes.dgp.panel_fe.simulate_panel_sem_fe`.
    **kwargs
        Forwarded to :func:`neighbayes.dgp.panel_fe.simulate_panel_sem_fe`.

    Returns
    -------
    dict
        Simulation output with keys ``y``, ``X``, ``unit``, ``time``,
        ``W_sparse``, ``W_graph``, and ``params_true``.

    Notes
    -----
    Uses the same DGP form as :func:`simulate_panel_sem_fe` with unit effects.
    """
    return simulate_panel_sem_fe(err_hetero=err_hetero, **kwargs)
