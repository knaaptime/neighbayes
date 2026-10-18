"""``fit(sampler=..., backend=...)``: one argument names what executes the sampler.

The sampler is the algorithm (Gibbs or NUTS); the backend is what runs it
(GLOSSARY.md), so the valid backends depend on the sampler.
"""

from __future__ import annotations

import pytest

from neighbayes import dgp
from neighbayes._backends import select_backend
from neighbayes.models import SAR, SARNegBinFlow, SEMProbit


def test_defaults_follow_the_sampler():
    assert select_backend("gibbs", None, {}) == "auto"
    assert select_backend("nuts", None, {}) == "pymc"
    assert select_backend("nuts", "nutpie", {}) == "nutpie"


@pytest.mark.parametrize(
    ("sampler", "backend"), [("gibbs", "pymc"), ("nuts", "jax"), ("nuts", "numpy")]
)
def test_a_backend_belongs_to_its_sampler(sampler, backend):
    with pytest.raises(ValueError, match=f"sampler={sampler!r}"):
        select_backend(sampler, backend, {})


def test_unknown_sampler_raises():
    with pytest.raises(ValueError, match="sampler must be"):
        select_backend("mala", None, {})


def _sar():
    d = dgp.simulate_sar(n_side=4, seed=0)
    return SAR(y=d["y"], X=d["X"], W=d["W_sparse"])


@pytest.mark.parametrize(
    ("sampler", "old"), [("gibbs", "gibbs_backend"), ("nuts", "nuts_sampler")]
)
def test_retired_names_raise_before_sampling(sampler, old):
    with pytest.raises(TypeError, match=f"{old}= is replaced by backend="):
        _sar().fit(sampler=sampler, **{old: "numpy"})


def test_flow_and_nuts_only_models_check_the_backend():
    f = dgp.simulate_sar_negbin_flow(n=4, seed=0)
    with pytest.raises(TypeError, match="replaced by backend="):
        SARNegBinFlow(f["y_vec"], f["X"], f["G"]).fit(gibbs_backend="numpy")
    p = dgp.simulate_sem_probit(n_side=3, seed=0, group_effects="random")
    m = SEMProbit(
        group_effects="random", y=p["y"], X=p["X"], W=p["W_sparse"], groups=p["groups"]
    )
    with pytest.raises(NotImplementedError, match="no Gibbs sampler"):
        m.fit(sampler="gibbs")
    with pytest.raises(ValueError, match="sampler='nuts'"):
        m.fit(backend="jax")
