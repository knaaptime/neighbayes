"""Ground-truth pin for every model family's default prior hyperparameters.

Phase 5a prerequisite: before any restructuring of ``models/priors.py`` this
snapshot locks the resolved default hyperparameters (``priors_as_dict`` of the
zero-argument dataclass) for *every* exported priors class.  Any refactor must
reproduce this table exactly, so a changed default fails loudly here.

Note the deliberate entanglement this captures — it is why the priors do not
factor into three orthogonal axes:

* ``rho``/``lam`` bounds differ by likelihood and static/dynamic:
  SAR ``[-1, 1]``, SARProbit ``[-0.95, 0.95]``, SARLogit ``[-0.999, 0.999]``,
  PanelSARDynamic ``[-0.95, 0.95]``.
* ``beta`` defaults are data-scaled (omitted) for every family, resolved on
  the link scale (identity, log, logit or probit) at fit time.
* the noise block differs: Gaussian/NB/Tobit carry ``sigma2_alpha``;
  SARProbit carries ``sigma_a_sigma``; Logit carries neither.
"""

from __future__ import annotations

import pytest

from neighbayes.models import priors as P
from neighbayes.models.priors import priors_as_dict

NU = 4.0  # fixed Student-t df (LeSage's rval)

EXPECTED: dict[str, dict[str, float]] = {
    "BasePriors": {"nu": NU, "sigma2_alpha": 2.0},
    "OLSPriors": {"nu": NU, "sigma2_alpha": 2.0},
    "SLXPriors": {"nu": NU, "sigma2_alpha": 2.0},
    "SARPriors": {
        "nu": NU,
        "sigma2_alpha": 2.0,
        "rho_lower": -1.0,
        "rho_upper": 1.0,
    },
    "SEMPriors": {
        "nu": NU,
        "sigma2_alpha": 2.0,
        "lam_lower": -1.0,
        "lam_upper": 1.0,
    },
    "SDMPriors": {
        "nu": NU,
        "sigma2_alpha": 2.0,
        "rho_lower": -1.0,
        "rho_upper": 1.0,
    },
    "SDEMPriors": {
        "nu": NU,
        "sigma2_alpha": 2.0,
        "lam_lower": -1.0,
        "lam_upper": 1.0,
    },
    "MultilevelPriors": {"nu": NU, "sigma2_alpha": 2.0, "sigma_nu": 3.0},
    "NegBinPriors": {
        "nu": NU,
        "sigma2_alpha": 2.0,
        "alpha_sigma": 2.5,
        "alpha_nu": 3.0,
    },
    "SARNegBinPriors": {
        "nu": NU,
        "sigma2_alpha": 2.0,
        "alpha_sigma": 2.5,
        "alpha_nu": 3.0,
        "rho_lower": -1.0,
        "rho_upper": 1.0,
    },
    "SARTobitPriors": {
        "nu": NU,
        "sigma2_alpha": 2.0,
        "rho_lower": -1.0,
        "rho_upper": 1.0,
    },
    "SEMTobitPriors": {
        "nu": NU,
        "sigma2_alpha": 2.0,
        "lam_lower": -1.0,
        "lam_upper": 1.0,
    },
    "SDMTobitPriors": {
        "nu": NU,
        "sigma2_alpha": 2.0,
        "rho_lower": -1.0,
        "rho_upper": 1.0,
    },
    "SARProbitPriors": {
        "rho_lower": -0.95,
        "rho_upper": 0.95,
        "sigma_a_sigma": 2.0,
    },
    "SARLogitPriors": {
        "rho_lower": -0.999,
        "rho_upper": 0.999,
    },
    "SEMLogitPriors": {
        "lam_lower": -0.999,
        "lam_upper": 0.999,
    },
    "PanelBasePriors": {"nu": NU, "sigma2_alpha": 2.0},
    "PanelOLSPriors": {"nu": NU, "sigma2_alpha": 2.0},
    "PanelSLXPriors": {"nu": NU, "sigma2_alpha": 2.0},
    "PanelSARPriors": {
        "nu": NU,
        "sigma2_alpha": 2.0,
        "rho_lower": -1.0,
        "rho_upper": 1.0,
    },
    "PanelSEMPriors": {
        "nu": NU,
        "sigma2_alpha": 2.0,
        "lam_lower": -1.0,
        "lam_upper": 1.0,
    },
    "PanelSDMPriors": {
        "nu": NU,
        "sigma2_alpha": 2.0,
        "rho_lower": -1.0,
        "rho_upper": 1.0,
    },
    "PanelSDEMPriors": {
        "nu": NU,
        "sigma2_alpha": 2.0,
        "lam_lower": -1.0,
        "lam_upper": 1.0,
    },
    "PanelOLSREPriors": {
        "nu": NU,
        "sigma2_alpha": 2.0,
        "sigma_alpha_nu": 1.0,
    },
    "PanelSARREPriors": {
        "nu": NU,
        "sigma2_alpha": 2.0,
        "sigma_alpha_nu": 1.0,
        "rho_lower": -1.0,
        "rho_upper": 1.0,
    },
    "PanelSEMREPriors": {
        "nu": NU,
        "sigma2_alpha": 2.0,
        "sigma_alpha_nu": 1.0,
        "lam_lower": -1.0,
        "lam_upper": 1.0,
    },
    "PanelSDEMREPriors": {
        "nu": NU,
        "sigma2_alpha": 2.0,
        "sigma_alpha_nu": 1.0,
        "lam_lower": -1.0,
        "lam_upper": 1.0,
    },
    "PanelSARTobitPriors": {
        "nu": NU,
        "sigma2_alpha": 2.0,
        "rho_lower": -1.0,
        "rho_upper": 1.0,
    },
    "PanelSEMTobitPriors": {
        "nu": NU,
        "sigma2_alpha": 2.0,
        "lam_lower": -1.0,
        "lam_upper": 1.0,
    },
    "PanelDynamicBasePriors": {
        "nu": NU,
        "sigma2_alpha": 2.0,
        "phi_lower": -0.95,
        "phi_upper": 0.95,
    },
    "PanelOLSDynamicPriors": {
        "nu": NU,
        "sigma2_alpha": 2.0,
        "phi_lower": -0.95,
        "phi_upper": 0.95,
    },
    "PanelSLXDynamicPriors": {
        "nu": NU,
        "sigma2_alpha": 2.0,
        "phi_lower": -0.95,
        "phi_upper": 0.95,
    },
    "PanelSARDynamicPriors": {
        "nu": NU,
        "sigma2_alpha": 2.0,
        "phi_lower": -0.95,
        "phi_upper": 0.95,
        "rho_lower": -0.95,
        "rho_upper": 0.95,
    },
    "PanelSEMDynamicPriors": {
        "nu": NU,
        "sigma2_alpha": 2.0,
        "phi_lower": -0.95,
        "phi_upper": 0.95,
        "lam_lower": -0.95,
        "lam_upper": 0.95,
    },
    "PanelSDMRDynamicPriors": {
        "nu": NU,
        "sigma2_alpha": 2.0,
        "phi_lower": -0.95,
        "phi_upper": 0.95,
        "rho_lower": -0.95,
        "rho_upper": 0.95,
    },
    "PanelSDMUDynamicPriors": {
        "nu": NU,
        "sigma2_alpha": 2.0,
        "phi_lower": -0.95,
        "phi_upper": 0.95,
        "rho_lower": -0.95,
        "rho_upper": 0.95,
        "theta_lower": -0.95,
        "theta_upper": 0.95,
    },
    "PanelSDEMDynamicPriors": {
        "nu": NU,
        "sigma2_alpha": 2.0,
        "phi_lower": -0.95,
        "phi_upper": 0.95,
        "lam_lower": -0.95,
        "lam_upper": 0.95,
    },
}


@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_default_prior_hyperparams_pinned(name):
    cls = getattr(P, name)
    assert priors_as_dict(cls()) == EXPECTED[name]


# Internal Gibbs-sampler prior structs (resolved-hyperparameter containers the
# numpy/JAX kernels consume) also live in ``priors.py``, but they are not
# user-facing validated priors and are exercised by the sampler recovery tests
# rather than this default-hyperparameter snapshot.
_GIBBS_PRIOR_STRUCTS = {
    "GibbsBasePriors",
    "GaussianGibbsPriors",
    "GibbsPriors",
    "ReducedGibbsPriors",
    "FlowReducedGibbsPriors",
    "ZINBGibbsPriors",
    "LogitGibbsPriors",
    "SEMLogitGibbsPriors",
    "REGibbsPriors",
}


def test_snapshot_covers_every_user_facing_priors_class():
    """Guard: every user-facing ``*Priors`` dataclass in ``__all__`` is pinned above."""
    exported = {
        n for n in P.__all__ if n.endswith("Priors") and n != "PriorsLike"
    } - _GIBBS_PRIOR_STRUCTS
    assert exported == set(EXPECTED), (
        "Priors classes not covered by the pin: "
        f"{sorted(exported - set(EXPECTED))}; stale entries: "
        f"{sorted(set(EXPECTED) - exported)}"
    )
