r"""Bayesian LM diagnostic tests — cross-sectional variants.

Each test is a score test of the Gaussian general nesting model at the fitted
null (:mod:`._gaussian`): the score of the tested direction at every posterior
draw, Neyman-orthogonal to every parameter the null estimates, and the
efficient information at the posterior mean (Doğan, Taşpınar and Bera 2021).
A robust test also removes the directions its alternative leaves at zero
(Bera and Yoon 1993).  The null is read from the fit, so one function serves
every null at which its direction is not estimated: ``bayesian_lm_lag_test``
is the lag test at the OLS, SLX and SEM nulls alike.

At a degenerate posterior on the OLS fit the tests are the classical ones:
Anselin's LM-lag and LM-error and their robust pair
(:cite:p:`anselin1996SimpleDiagnostic`), and the WX and joint tests of
:cite:t:`koley2024UseNot`.

All public test functions return :class:`BayesianLMTestResult`.
"""

from neighbayes.diagnostics.lmtests._gaussian import orthogonal_lm
from neighbayes.diagnostics.lmtests.core import (
    BayesianLMTestResult,
    _finalize_lm,
)


def _score_test(
    model, tested, robust=(), *, test_type: str, wx: bool = False
) -> BayesianLMTestResult:
    LM, statistic, df = orthogonal_lm(model, tested, robust, test_name=test_type)
    details = {"k_wx": int(model._WX.shape[1])} if wx else None
    return _finalize_lm(
        LM, test_type=test_type, df=df, details=details, statistic=statistic
    )


def bayesian_lm_lag_test(model) -> BayesianLMTestResult:
    r"""LM test for an omitted spatial lag (H₀: ρ = 0).

    The score test of :math:`\rho` at an OLS, SLX or SEM fit.  At a
    degenerate posterior on the OLS estimate it is Anselin's LM-lag,
    :math:`(e^\top W y)^2 / (\sigma^4 T_{WW} + \sigma^2 \|M_X WX\beta\|^2)`
    (:cite:p:`anselin1996SimpleDiagnostic`, eq. 13).

    Returns
    -------
    BayesianLMTestResult
        Per-draw LM samples, summary statistics and ``df = 1``.
    """
    return _score_test(model, ("rho",), test_type="bayesian_lm_lag")


def bayesian_lm_error_test(model) -> BayesianLMTestResult:
    r"""LM test for an omitted spatial error process (H₀: λ = 0).

    The score test of :math:`\lambda` at an OLS, SLX, SAR or SDM fit (the
    Tobit suites use it at their SAR and SDM nulls).  At a degenerate
    posterior on the OLS estimate it is Anselin's LM-error,
    :math:`(e^\top W e)^2/(\sigma^4 T_{WW})`.

    Returns
    -------
    BayesianLMTestResult
        Per-draw LM samples, summary statistics and ``df = 1``.
    """
    return _score_test(model, ("lam",), test_type="bayesian_lm_error")


def bayesian_lm_wx_test(model) -> BayesianLMTestResult:
    r"""LM test for omitted :math:`WX` at a SAR fit (H₀: θ = 0 | SAR).

    Whether the SAR should be the SDM; the score of the :math:`WX` block,
    orthogonal to :math:`(\beta, \rho, \sigma^2)`
    (:cite:p:`koley2024UseNot`).

    Returns
    -------
    BayesianLMTestResult
        Per-draw LM samples, summary statistics and ``df = k_wx``.
    """
    return _score_test(model, ("theta",), test_type="bayesian_lm_wx", wx=True)


def bayesian_lm_wx_sem_test(model) -> BayesianLMTestResult:
    r"""LM test for omitted :math:`WX` at a SEM fit (H₀: θ = 0 | SEM).

    Whether the SEM should be the SDEM; the score of the :math:`WX` block,
    orthogonal to :math:`(\beta, \lambda, \sigma^2)`.

    Returns
    -------
    BayesianLMTestResult
        Per-draw LM samples, summary statistics and ``df = k_wx``.
    """
    return _score_test(model, ("theta",), test_type="bayesian_lm_wx_sem", wx=True)


def bayesian_lm_sdm_joint_test(model) -> BayesianLMTestResult:
    r"""Joint LM test for the SDM at an OLS fit (H₀: ρ = 0 and θ = 0).

    The joint score of :math:`(\rho, \theta)`; at a degenerate posterior on
    the OLS estimate it is the ``lm_spdurbin`` test of
    :cite:t:`koley2024UseNot`.

    Returns
    -------
    BayesianLMTestResult
        Per-draw LM samples, summary statistics and ``df = 1 + k_wx``.
    """
    return _score_test(
        model, ("rho", "theta"), test_type="bayesian_lm_sdm_joint", wx=True
    )


def bayesian_lm_slx_error_joint_test(model) -> BayesianLMTestResult:
    r"""Joint LM test for the SDEM at an OLS fit (H₀: λ = 0 and θ = 0).

    The joint score of :math:`(\lambda, \theta)`, whose information is block
    diagonal at the OLS null, so that at a degenerate posterior it is
    ``LM-Error + LM-WX``, the ``lm_slxerr`` test of :cite:t:`koley2024UseNot`.

    Returns
    -------
    BayesianLMTestResult
        Per-draw LM samples, summary statistics and ``df = 1 + k_wx``.
    """
    return _score_test(
        model, ("lam", "theta"), test_type="bayesian_lm_slx_error_joint", wx=True
    )


def bayesian_robust_lm_lag_test(model) -> BayesianLMTestResult:
    r"""Robust LM-lag at an OLS fit (H₀: ρ = 0, robust to a local λ).

    The lag score net of the error score's direction (Bera and Yoon 1993;
    :cite:p:`anselin1996SimpleDiagnostic`).

    Returns
    -------
    BayesianLMTestResult
        Per-draw LM samples, summary statistics and ``df = 1``.
    """
    return _score_test(model, ("rho",), ("lam",), test_type="bayesian_robust_lm_lag")


def bayesian_robust_lm_error_test(model) -> BayesianLMTestResult:
    r"""Robust LM-error at an OLS fit (H₀: λ = 0, robust to a local ρ).

    The error score net of the lag score's direction
    (:cite:p:`anselin1996SimpleDiagnostic`).

    Returns
    -------
    BayesianLMTestResult
        Per-draw LM samples, summary statistics and ``df = 1``.
    """
    return _score_test(model, ("lam",), ("rho",), test_type="bayesian_robust_lm_error")


def bayesian_robust_lm_lag_sdm_test(model) -> BayesianLMTestResult:
    r"""Robust LM-lag at an SLX fit (H₀: ρ = 0, robust to a local λ).

    Whether the SLX should be the SDM rather than the SDEM: the lag score,
    orthogonal to the fitted :math:`(\beta, \theta, \sigma^2)` and net of
    the error direction (:cite:p:`koley2024UseNot`).

    Returns
    -------
    BayesianLMTestResult
        Per-draw LM samples, summary statistics and ``df = 1``.
    """
    return _score_test(
        model, ("rho",), ("lam",), test_type="bayesian_robust_lm_lag_sdm", wx=True
    )


def bayesian_robust_lm_error_sdem_test(model) -> BayesianLMTestResult:
    r"""Robust LM-error at an SLX fit (H₀: λ = 0, robust to a local ρ).

    Whether the SLX should be the SDEM rather than the SDM: the error score,
    orthogonal to the fitted :math:`(\beta, \theta, \sigma^2)` and net of
    the lag direction.

    Returns
    -------
    BayesianLMTestResult
        Per-draw LM samples, summary statistics and ``df = 1``.
    """
    return _score_test(
        model,
        ("lam",),
        ("rho",),
        test_type="bayesian_robust_lm_error_sdem",
        wx=True,
    )


def bayesian_lm_error_from_sar_test(model) -> BayesianLMTestResult:
    r"""LM test for a spatial error process at a SAR fit (H₀: λ = 0 | SAR).

    Whether the SAR should be the SARAR; the error score, orthogonal to
    :math:`(\beta, \rho, \sigma^2)`, so that the estimated lag is accounted
    for (:cite:p:`anselin1988SpatialEconometrics`).

    Returns
    -------
    BayesianLMTestResult
        Per-draw LM samples, summary statistics and ``df = 1``.
    """
    return _score_test(model, ("lam",), test_type="bayesian_lm_error_from_sar")


def bayesian_robust_lm_wx_test(model) -> BayesianLMTestResult:
    r"""Robust LM-WX at a SAR fit (H₀: θ = 0, robust to a local λ).

    The :math:`WX` score of :func:`bayesian_lm_wx_test` net of the error
    direction, so that clustered disturbances the SAR leaves out do not pass
    for omitted spillovers.

    Returns
    -------
    BayesianLMTestResult
        Per-draw LM samples, summary statistics and ``df = k_wx``.
    """
    return _score_test(
        model, ("theta",), ("lam",), test_type="bayesian_robust_lm_wx", wx=True
    )


def bayesian_robust_lm_error_sar_test(model) -> BayesianLMTestResult:
    r"""Robust LM-error at a SAR fit (H₀: λ = 0, robust to a local θ).

    The error score of :func:`bayesian_lm_error_from_sar_test` net of the
    :math:`WX` direction, so that omitted spillovers do not pass for
    clustered disturbances.  Without :math:`WX` columns it is that test.

    Returns
    -------
    BayesianLMTestResult
        Per-draw LM samples, summary statistics and ``df = 1``.
    """
    return _score_test(
        model, ("lam",), ("theta",), test_type="bayesian_robust_lm_error_sar"
    )


def bayesian_lm_error_sdm_test(model) -> BayesianLMTestResult:
    r"""LM test for a spatial error process at an SDM fit (H₀: λ = 0 | SDM).

    Whether the SDM should be the general nesting model; the error score,
    orthogonal to :math:`(\beta, \theta, \rho, \sigma^2)`.

    Returns
    -------
    BayesianLMTestResult
        Per-draw LM samples, summary statistics and ``df = 1``.
    """
    return _score_test(model, ("lam",), test_type="bayesian_lm_error_sdm")


def bayesian_robust_lm_error_sdm_test(model) -> BayesianLMTestResult:
    r"""Robust LM-error at an SDM fit (H₀: λ = 0 | SDM).

    Robust to the lag, which the SDM estimates: the efficient score is
    orthogonal to it already, and the general nesting model leaves no other
    direction at zero, so this is :func:`bayesian_lm_error_sdm_test`.

    Returns
    -------
    BayesianLMTestResult
        Per-draw LM samples, summary statistics and ``df = 1``.
    """
    return _score_test(
        model, ("lam",), test_type="bayesian_robust_lm_error_sdm", wx=True
    )


def bayesian_lm_lag_sdem_test(model) -> BayesianLMTestResult:
    r"""LM test for a spatial lag at an SDEM fit (H₀: ρ = 0 | SDEM).

    Whether the SDEM should be the general nesting model; the lag score,
    orthogonal to :math:`(\beta, \theta, \lambda, \sigma^2)`.

    Returns
    -------
    BayesianLMTestResult
        Per-draw LM samples, summary statistics and ``df = 1``.
    """
    return _score_test(model, ("rho",), test_type="bayesian_lm_lag_sdem")


def bayesian_robust_lm_lag_sdem_test(model) -> BayesianLMTestResult:
    r"""Robust LM-lag at an SDEM fit (H₀: ρ = 0 | SDEM).

    Robust to the error process, which the SDEM estimates, so this is
    :func:`bayesian_lm_lag_sdem_test`.

    Returns
    -------
    BayesianLMTestResult
        Per-draw LM samples, summary statistics and ``df = 1``.
    """
    return _score_test(model, ("rho",), test_type="bayesian_robust_lm_lag_sdem")


def bayesian_robust_lm_lag_sem_test(model) -> BayesianLMTestResult:
    r"""Robust LM-lag at a SEM fit (H₀: ρ = 0, robust to a local θ).

    The lag score of :func:`bayesian_lm_lag_test` at the SEM, net of the
    :math:`WX` direction.

    Returns
    -------
    BayesianLMTestResult
        Per-draw LM samples, summary statistics and ``df = 1``.
    """
    return _score_test(
        model, ("rho",), ("theta",), test_type="bayesian_robust_lm_lag_sem", wx=True
    )


def bayesian_robust_lm_wx_sem_test(model) -> BayesianLMTestResult:
    r"""Robust LM-WX at a SEM fit (H₀: θ = 0, robust to a local ρ).

    The :math:`WX` score of :func:`bayesian_lm_wx_sem_test` net of the lag
    direction.

    Returns
    -------
    BayesianLMTestResult
        Per-draw LM samples, summary statistics and ``df = k_wx``.
    """
    return _score_test(
        model, ("theta",), ("rho",), test_type="bayesian_robust_lm_wx_sem", wx=True
    )
