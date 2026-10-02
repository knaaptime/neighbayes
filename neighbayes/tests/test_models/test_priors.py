"""Tests for the typed-priors layer added in Phase 2."""

from __future__ import annotations

import numpy as np
import pytest
from libpysal.graph import Graph
from scipy import sparse as sp

from neighbayes.models import OLS, SAR, SDEM, SDM, SEM, SLX
from neighbayes.models.cross_section.sar_probit import SARProbit
from neighbayes.models.cross_section.tobit import SARTobit
from neighbayes.models.priors import (
    BasePriors,
    OLSPriors,
    SARPriors,
    SARProbitPriors,
    SDEMPriors,
    SDMPriors,
    SEMPriors,
    SLXPriors,
    priors_as_dict,
    resolve_priors,
)


def _make_w(n: int = 6) -> Graph:
    rng = np.random.default_rng(0)
    A = (rng.random((n, n)) < 0.4).astype(float)
    np.fill_diagonal(A, 0)
    rs = A.sum(axis=1)
    rs[rs == 0] = 1
    A = A / rs[:, None]
    return Graph.from_sparse(sp.csr_matrix(A))


@pytest.fixture
def xy():
    rng = np.random.default_rng(1)
    n = 6
    X = rng.normal(size=(n, 2))
    y = rng.normal(size=n)
    return y, X, _make_w(n)


# ---------------------------------------------------------------------------
# resolve_priors / priors_as_dict
# ---------------------------------------------------------------------------


def test_resolve_priors_defaults():
    p = resolve_priors(None, SARPriors)
    assert isinstance(p, SARPriors)
    assert p.rho_lower == -1.0
    assert p.rho_upper == 1.0


def test_resolve_priors_dict_equals_dataclass():
    a = resolve_priors({"rho_lower": 0.0}, SARPriors)
    b = resolve_priors(SARPriors(rho_lower=0.0), SARPriors)
    assert priors_as_dict(a) == priors_as_dict(b)


def test_resolve_priors_unknown_key_raises():
    with pytest.raises(TypeError, match="Unknown prior key"):
        resolve_priors({"not_a_prior": 1.0}, SARPriors)


def test_resolve_priors_wrong_dataclass_type_raises():
    with pytest.raises(TypeError):
        resolve_priors(SEMPriors(), SARPriors)


# ---------------------------------------------------------------------------
# Model-level wiring
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "model_cls, priors_cls",
    [
        (OLS, OLSPriors),
        (SLX, SLXPriors),
        (SAR, SARPriors),
        (SEM, SEMPriors),
        (SDM, SDMPriors),
        (SDEM, SDEMPriors),
    ],
)
def test_model_default_priors(model_cls, priors_cls, xy):
    y, X, W = xy
    m = model_cls(y=y, X=X, W=W)
    assert isinstance(m.priors_obj, priors_cls)
    assert isinstance(m.priors, dict)
    assert m.priors == priors_as_dict(m.priors_obj)


def test_model_dict_and_dataclass_equivalent(xy):
    y, X, W = xy
    a = SAR(y=y, X=X, W=W, priors={"rho_lower": 0.0})
    b = SAR(y=y, X=X, W=W, priors=SARPriors(rho_lower=0.0))
    assert a.priors == b.priors


def test_model_unknown_prior_key_raises(xy):
    y, X, W = xy
    with pytest.raises(TypeError, match="Unknown prior key"):
        SAR(y=y, X=X, W=W, priors={"bogus": 1.0})


def test_sar_tobit_priors(xy):
    """Tobit shares the Gaussian priors; the old fixed-scale keys are gone."""
    y, X, W = xy
    m = SARTobit(y=np.clip(y, 0, None), X=X, W=W)
    assert m.priors["sigma2_alpha"] == 2.0
    for key in ("censor_sigma", "sigma_sigma"):
        with pytest.raises(TypeError, match="Unknown prior key"):
            SARTobit(y=np.clip(y, 0, None), X=X, W=W, priors={key: 10.0})


def test_spatial_probit_priors_defaults():
    n = 4
    rng = np.random.default_rng(2)
    X = rng.normal(size=(n, 2))
    y = (rng.normal(size=n) > 0).astype(int)
    W = _make_w(n)
    m = SARProbit(y=y, X=X, W=W, region_ids=np.arange(n))
    assert isinstance(m.priors_obj, SARProbitPriors)
    assert m.priors["rho_lower"] == -0.95
    assert m.priors["rho_upper"] == 0.95


def test_basepriors_is_abstract_default():
    # The base default applies to OLS-like models without subclassing.
    assert issubclass(OLSPriors, BasePriors)
