"""Panel censored (Tobit-style) DGP functions.

Simulates left-censored panel data from SAR and SEM spatial Tobit models
with fixed-effects transformations.
"""

from __future__ import annotations

from .panel_fe import simulate_sar_panel_fe, simulate_sem_panel_fe
from .utils import _left_censor, make_panel_output_geodataframe


def simulate_sar_tobit_panel(
    *, censoring: float = 0.0, err_hetero: bool = False, **kwargs
) -> dict:
    """Simulate left-censored panel SAR FE data.

    Parameters
    ----------
    censoring : float, default=0.0
        Left-censoring threshold ``c`` where observed ``y = max(c, y*)``.
    err_hetero : bool, default=False
        If True, generate heteroskedastic innovations. Forwarded to
        :func:`neighbayes.dgp.panel_fe.simulate_sar_panel_fe`.
    **kwargs
        Forwarded to :func:`neighbayes.dgp.panel_fe.simulate_sar_panel_fe`.

    Returns
    -------
    dict
        Adds ``y_latent`` and ``censored_mask`` to panel SAR FE simulation output.
    """
    create_gdf = kwargs.pop("create_gdf", False)
    wide = kwargs.pop("wide", False)
    geometry_type = kwargs.pop("geometry_type", "polygon")
    gdf = kwargs.get("gdf", None)
    out = simulate_sar_panel_fe(err_hetero=err_hetero, **kwargs)
    y_obs, mask = _left_censor(out["y"], censoring)
    out["y_latent"] = out["y"]
    out["y"] = y_obs
    out["censored_mask"] = mask
    out["params_true"]["censoring"] = censoring
    if create_gdf or gdf is not None or wide:
        N = kwargs.get("N")
        T = kwargs.get("T")
        return make_panel_output_geodataframe(
            out["y"],
            out["X"],
            out["unit"],
            out["time"],
            N,
            T,
            gdf=gdf,
            geometry_type=geometry_type,
            wide=wide,
        )
    return out


def simulate_sem_tobit_panel(
    *, censoring: float = 0.0, err_hetero: bool = False, **kwargs
) -> dict:
    """Simulate left-censored panel SEM FE data.

    Parameters
    ----------
    censoring : float, default=0.0
        Left-censoring threshold ``c`` where observed ``y = max(c, y*)``.
    err_hetero : bool, default=False
        If True, generate heteroskedastic innovations. Forwarded to
        :func:`neighbayes.dgp.panel_fe.simulate_sem_panel_fe`.
    **kwargs
        Forwarded to :func:`neighbayes.dgp.panel_fe.simulate_sem_panel_fe`.

    Returns
    -------
    dict
        Adds ``y_latent`` and ``censored_mask`` to panel SEM FE simulation output.
    """
    create_gdf = kwargs.pop("create_gdf", False)
    wide = kwargs.pop("wide", False)
    geometry_type = kwargs.pop("geometry_type", "polygon")
    gdf = kwargs.get("gdf", None)
    out = simulate_sem_panel_fe(err_hetero=err_hetero, **kwargs)
    y_obs, mask = _left_censor(out["y"], censoring)
    out["y_latent"] = out["y"]
    out["y"] = y_obs
    out["censored_mask"] = mask
    out["params_true"]["censoring"] = censoring
    if create_gdf or gdf is not None or wide:
        N = kwargs.get("N")
        T = kwargs.get("T")
        return make_panel_output_geodataframe(
            out["y"],
            out["X"],
            out["unit"],
            out["time"],
            N,
            T,
            gdf=gdf,
            geometry_type=geometry_type,
            wide=wide,
        )
    return out
