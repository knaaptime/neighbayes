r"""Bayesian LM diagnostic tests — panel variants.

Each Gaussian panel test is the score test of the panel general nesting model
at the fitted null, on the engine of :mod:`._gaussian`: pooled panels stack
their periods, fixed effects use the Lee and Yu (2010) transformed
likelihood the demeaned data give, random unit effects are coefficients
carrying their prior, and a dynamic panel conditions on its first period with
the lagged outcome as a regressor.  The flow panels' tests follow.

All public test functions return :class:`BayesianLMTestResult`.
"""

from neighbayes.diagnostics.lmtests._gaussian import orthogonal_lm
from neighbayes.diagnostics.lmtests.core import (
    BayesianLMTestResult,
    _finalize_lm,
)

from .flow import _flow_test


def _panel_test(
    model, tested, robust=(), *, test_type: str, wx: bool = False
) -> BayesianLMTestResult:
    LM, statistic, df = orthogonal_lm(model, tested, robust, test_name=test_type)
    details = {"N": int(model._N), "T": int(model._T)}
    if wx:
        details["k_wx"] = int(model._WX.shape[1])
    return _finalize_lm(
        LM, test_type=test_type, df=df, details=details, statistic=statistic
    )


def bayesian_panel_lm_lag_test(model) -> BayesianLMTestResult:
    r"""LM test for an omitted spatial lag in a panel (H₀: ρ = 0).

    At a pooled, fixed- or random-effects OLS, SLX or SEM panel, or their dynamic forms (:cite:p:`anselin2008SpatialPanel`; :cite:p:`elhorst2014SpatialEconometrics`).

    Returns
    -------
    BayesianLMTestResult
        Per-draw LM samples, summary statistics and ``df = 1``.
    """
    return _panel_test(model, ("rho",), test_type="bayesian_panel_lm_lag")


def bayesian_panel_lm_error_test(model) -> BayesianLMTestResult:
    r"""LM test for an omitted spatial error process in a panel (H₀: λ = 0).

    At a pooled, fixed- or random-effects OLS, SLX, SAR or SDM panel (:cite:p:`anselin2008SpatialPanel`).

    Returns
    -------
    BayesianLMTestResult
        Per-draw LM samples, summary statistics and ``df = 1``.
    """
    return _panel_test(model, ("lam",), test_type="bayesian_panel_lm_error")


def bayesian_panel_robust_lm_lag_test(model) -> BayesianLMTestResult:
    r"""Robust panel LM-lag (H₀: ρ = 0, robust to a local λ).

    The panel analogue of :func:`~neighbayes.diagnostics.lmtests.bayesian_robust_lm_lag_test` (:cite:p:`elhorst2014SpatialEconometrics`).

    Returns
    -------
    BayesianLMTestResult
        Per-draw LM samples, summary statistics and ``df = 1``.
    """
    return _panel_test(
        model, ("rho",), ("lam",), test_type="bayesian_panel_robust_lm_lag"
    )


def bayesian_panel_robust_lm_error_test(model) -> BayesianLMTestResult:
    r"""Robust panel LM-error (H₀: λ = 0, robust to a local ρ).

    The panel analogue of :func:`~neighbayes.diagnostics.lmtests.bayesian_robust_lm_error_test`.

    Returns
    -------
    BayesianLMTestResult
        Per-draw LM samples, summary statistics and ``df = 1``.
    """
    return _panel_test(
        model, ("lam",), ("rho",), test_type="bayesian_panel_robust_lm_error"
    )


def bayesian_panel_lm_wx_test(model) -> BayesianLMTestResult:
    r"""Panel LM test for omitted :math:`WX` at a SAR panel (H₀: θ = 0 | SAR).

    Whether the SAR panel should be the SDM panel.

    Returns
    -------
    BayesianLMTestResult
        Per-draw LM samples, summary statistics and ``df = k_wx``.
    """
    return _panel_test(model, ("theta",), test_type="bayesian_panel_lm_wx", wx=True)


def bayesian_panel_lm_wx_sem_test(model) -> BayesianLMTestResult:
    r"""Panel LM test for omitted :math:`WX` at a SEM panel (H₀: θ = 0 | SEM).

    Whether the SEM panel should be the SDEM panel.

    Returns
    -------
    BayesianLMTestResult
        Per-draw LM samples, summary statistics and ``df = k_wx``.
    """
    return _panel_test(model, ("theta",), test_type="bayesian_panel_lm_wx_sem", wx=True)


def bayesian_panel_lm_sdm_joint_test(model) -> BayesianLMTestResult:
    r"""Joint panel LM test for the SDM at an OLS panel (H₀: ρ = 0 and θ = 0).

    The panel analogue of :func:`~neighbayes.diagnostics.lmtests.bayesian_lm_sdm_joint_test`.

    Returns
    -------
    BayesianLMTestResult
        Per-draw LM samples, summary statistics and ``df = 1 + k_wx``.
    """
    return _panel_test(
        model, ("rho", "theta"), test_type="bayesian_panel_lm_sdm_joint", wx=True
    )


def bayesian_panel_lm_slx_error_joint_test(model) -> BayesianLMTestResult:
    r"""Joint panel LM test for the SDEM at an OLS panel (H₀: λ = 0 and θ = 0).

    The panel analogue of :func:`~neighbayes.diagnostics.lmtests.bayesian_lm_slx_error_joint_test`.

    Returns
    -------
    BayesianLMTestResult
        Per-draw LM samples, summary statistics and ``df = 1 + k_wx``.
    """
    return _panel_test(
        model, ("lam", "theta"), test_type="bayesian_panel_lm_slx_error_joint", wx=True
    )


def bayesian_panel_robust_lm_lag_sdm_test(model) -> BayesianLMTestResult:
    r"""Robust panel LM-lag at an SLX panel (H₀: ρ = 0, robust to a local λ).

    Whether the SLX panel should be the SDM panel rather than the SDEM panel.

    Returns
    -------
    BayesianLMTestResult
        Per-draw LM samples, summary statistics and ``df = 1``.
    """
    return _panel_test(
        model, ("rho",), ("lam",), test_type="bayesian_panel_robust_lm_lag_sdm", wx=True
    )


def bayesian_panel_robust_lm_wx_test(model) -> BayesianLMTestResult:
    r"""Robust panel LM-WX at a SAR panel (H₀: θ = 0, robust to a local λ).

    The :math:`WX` score net of the error direction.

    Returns
    -------
    BayesianLMTestResult
        Per-draw LM samples, summary statistics and ``df = k_wx``.
    """
    return _panel_test(
        model, ("theta",), ("lam",), test_type="bayesian_panel_robust_lm_wx", wx=True
    )


def bayesian_panel_robust_lm_error_sdem_test(model) -> BayesianLMTestResult:
    r"""Robust panel LM-error at an SLX panel (H₀: λ = 0, robust to a local ρ).

    Whether the SLX panel should be the SDEM panel rather than the SDM panel.

    Returns
    -------
    BayesianLMTestResult
        Per-draw LM samples, summary statistics and ``df = 1``.
    """
    return _panel_test(
        model,
        ("lam",),
        ("rho",),
        test_type="bayesian_panel_robust_lm_error_sdem",
        wx=True,
    )


def bayesian_panel_lm_error_sdm_test(model) -> BayesianLMTestResult:
    r"""Panel LM test for a spatial error process at an SDM panel (H₀: λ = 0 | SDM).

    Whether the SDM panel should be the general nesting panel.

    Returns
    -------
    BayesianLMTestResult
        Per-draw LM samples, summary statistics and ``df = 1``.
    """
    return _panel_test(model, ("lam",), test_type="bayesian_panel_lm_error_sdm")


def bayesian_panel_lm_lag_sdem_test(model) -> BayesianLMTestResult:
    r"""Panel LM test for a spatial lag at an SEM or SDEM panel (H₀: ρ = 0).

    Whether the panel should add the lag; the dynamic SEM panel's suite uses it.

    Returns
    -------
    BayesianLMTestResult
        Per-draw LM samples, summary statistics and ``df = 1``.
    """
    return _panel_test(model, ("rho",), test_type="bayesian_panel_lm_lag_sdem")


def bayesian_panel_robust_lm_lag_sdem_test(model) -> BayesianLMTestResult:
    r"""Robust panel LM-lag at an SEM or SDEM panel (H₀: ρ = 0).

    Robust to the error process, which the null estimates, so this is :func:`bayesian_panel_lm_lag_sdem_test`.

    Returns
    -------
    BayesianLMTestResult
        Per-draw LM samples, summary statistics and ``df = 1``.
    """
    return _panel_test(model, ("rho",), test_type="bayesian_panel_robust_lm_lag_sdem")


def _flow_panel_test(model, tested, robust=(), *, test_type: str):
    details = {"T": int(model._T)}
    if len(tested) == 1:
        details["direction"] = tested[0]
    return _flow_test(model, tested, robust, test_type=test_type, details=details)


def bayesian_panel_lm_flow_destination_test(model) -> BayesianLMTestResult:
    r"""LM test for an omitted destination lag at an ``OLSFlowPanelFE`` fit.

    H₀: :math:`\rho_d = 0`.  The test of
    :func:`~neighbayes.diagnostics.lmtests.bayesian_lm_flow_destination_test`
    on the Lee and Yu (2010) transformed likelihood of the demeaned panel.

    Returns
    -------
    BayesianLMTestResult
        Per-draw LM samples, summary statistics and ``df = 1``.
    """
    return _flow_panel_test(
        model, ("d",), test_type="bayesian_panel_lm_flow_destination"
    )


def bayesian_panel_lm_flow_origin_test(model) -> BayesianLMTestResult:
    r"""LM test for an omitted origin lag at an ``OLSFlowPanelFE`` fit (H₀: ρ_o = 0).

    See :func:`bayesian_panel_lm_flow_destination_test`.
    """
    return _flow_panel_test(model, ("o",), test_type="bayesian_panel_lm_flow_origin")


def bayesian_panel_lm_flow_od_test(model) -> BayesianLMTestResult:
    r"""LM test for an omitted network lag at an ``OLSFlowPanelFE`` fit (H₀: ρ_w = 0).

    See :func:`bayesian_panel_lm_flow_destination_test`.
    """
    return _flow_panel_test(model, ("w",), test_type="bayesian_panel_lm_flow_od")


def bayesian_panel_lm_flow_joint_test(model) -> BayesianLMTestResult:
    r"""Joint LM test at an ``OLSFlowPanelFE`` fit (H₀: ρ_d = ρ_o = ρ_w = 0), df 3.

    See :func:`bayesian_panel_lm_flow_destination_test`.
    """
    return _flow_panel_test(
        model, ("d", "o", "w"), test_type="bayesian_panel_lm_flow_joint"
    )


def bayesian_panel_robust_lm_flow_destination_test(model) -> BayesianLMTestResult:
    r"""Robust LM test for the destination lag at an ``OLSFlowPanelFE`` fit.

    H₀: :math:`\rho_d = 0`, robust to local origin and network lags; see
    :func:`~neighbayes.diagnostics.lmtests.bayesian_robust_lm_flow_destination_test`.
    """
    return _flow_panel_test(
        model, ("d",), ("o", "w"),
        test_type="bayesian_panel_robust_lm_flow_destination",
    )  # fmt: skip


def bayesian_panel_robust_lm_flow_origin_test(model) -> BayesianLMTestResult:
    r"""Robust LM test for the origin lag at an ``OLSFlowPanelFE`` fit.

    See :func:`bayesian_panel_robust_lm_flow_destination_test`.
    """
    return _flow_panel_test(
        model, ("o",), ("d", "w"), test_type="bayesian_panel_robust_lm_flow_origin"
    )


def bayesian_panel_robust_lm_flow_od_test(model) -> BayesianLMTestResult:
    r"""Robust LM test for the network lag at an ``OLSFlowPanelFE`` fit.

    See :func:`bayesian_panel_robust_lm_flow_destination_test`.
    """
    return _flow_panel_test(
        model, ("w",), ("d", "o"), test_type="bayesian_panel_robust_lm_flow_od"
    )
