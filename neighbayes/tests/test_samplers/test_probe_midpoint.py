"""The probe check halfway through warmup, outside the Gaussian family.

The NumPy driver (:func:`run_sweeps_with_midpoint`) pauses every chain at the
midpoint and resumes it; the hook factories size a stochastic log-determinant's
probe pool there and install it, through the chain objects (NumPy) or by
substituting the sweep's parameters (JAX).  A fit records the pool, and the
impacts' gradient reuses it.
"""

from __future__ import annotations

import importlib.util
import warnings

import numpy as np
import pytest

from neighbayes.samplers._utils._sweeps import (
    SweepMidpoint,
    probe_sweep_midpoint,
    run_sweeps,
    run_sweeps_with_midpoint,
)

SEEDS, TUNE, DRAWS = [11, 12, 13], 40, 30
requires_jax = pytest.mark.skipif(
    importlib.util.find_spec("jax") is None, reason="JAX not installed"
)


class _Walk:
    """A toy chain: ``x`` takes a standard normal step plus a drift each sweep."""

    def __init__(self, rng):
        self.rng, self.x, self.drift, self.sweeps = rng, 0.0, 0.0, 0

    def sweep(self, tuning):
        self.x += self.drift + self.rng.normal()
        self.sweeps += 1

    def draw(self, store_latent):
        return {"x": self.x}


def _run(midpoint=None, n_jobs=1, **kw):
    out = run_sweeps_with_midpoint(
        _Walk,
        seeds=SEEDS,
        draws=DRAWS,
        tune=TUNE,
        midpoint=midpoint,
        n_jobs=n_jobs,
        progressbar=False,
        **kw,
    )
    return np.stack([r["x"] for r in out])


@pytest.mark.parametrize("n_jobs", [1, 2])
def test_a_midpoint_that_changes_nothing_leaves_the_draws(n_jobs):
    seen = {}

    def adapt(kept, *, draws, chains):
        seen.update(kept=kept, draws=draws, chains=chains)

    hooked = _run(SweepMidpoint(keep=lambda ch: (ch.x, ch.sweeps), adapt=adapt), n_jobs)
    plain = [
        run_sweeps(_Walk(np.random.default_rng(s)), draws=DRAWS, tune=TUNE)["x"]
        for s in SEEDS
    ]
    np.testing.assert_array_equal(hooked, np.stack(plain))
    np.testing.assert_array_equal(_run(None, n_jobs), hooked)
    # Every chain's own first half of warmup, chains apart.
    assert seen["kept"].shape == (len(SEEDS), TUNE // 2, 2)
    np.testing.assert_array_equal(
        seen["kept"][:, :, 1], np.tile(np.arange(1, 21), (3, 1))
    )
    assert not np.allclose(seen["kept"][0], seen["kept"][1])
    assert (seen["draws"], seen["chains"]) == (DRAWS, len(SEEDS))


@pytest.mark.parametrize("n_jobs", [1, 2])
def test_a_change_at_the_midpoint_reaches_every_chain(n_jobs):
    def drift(chain):
        chain.drift = 100.0

    x = _run(SweepMidpoint(keep=lambda ch: (ch.x,), adapt=lambda k, **_: drift), n_jobs)
    assert (np.diff(x, axis=1) > 90.0).all()


def test_finish_adds_what_the_chain_accumulated():
    out = run_sweeps_with_midpoint(
        _Walk,
        seeds=SEEDS,
        draws=DRAWS,
        tune=TUNE,
        finish=lambda ch: {"sweeps": ch.sweeps},
        n_jobs=1,
        progressbar=False,
    )
    assert [r["sweeps"] for r in out] == [TUNE + DRAWS] * len(SEEDS)


class _Probes:
    """Stands in for WarmupProbes: records each adapt; grows when told to."""

    def __init__(self, grow: bool):
        self.grow, self.calls = grow, []

    def adapt(self, settled, *, draws, chains, jax=False):
        from neighbayes._logdet._warmup import Evaluators

        self.calls.append(np.shape(settled))
        if not self.grow:
            return None
        params = None
        if jax:
            import jax.numpy as jnp

            params = (jnp.ones(5), jnp.float64(-0.5), jnp.float64(0.9))
        return Evaluators(None, None, -0.5, 0.9, params)

    def period_fn(self):
        return "grown"


class _TwoParts:
    def __init__(self):
        self.a, self.b, self.fn_a, self.fn_b = 0.1, 0.2, None, None


def test_two_parameters_on_one_W_size_one_pool():
    probes = _Probes(grow=True)
    mid = probe_sweep_midpoint(
        [
            (probes, lambda ch: ch.a, lambda ch, fn: setattr(ch, "fn_a", fn)),
            (probes, lambda ch: ch.b, lambda ch, fn: setattr(ch, "fn_b", fn)),
        ]
    )
    chain = _TwoParts()
    assert mid.keep(chain) == (0.1, 0.2)
    apply = mid.adapt(np.zeros((3, 20, 2)), draws=DRAWS, chains=3)
    # One sizing, on both parameters' settled halves, chains apart.
    assert probes.calls == [(2, 3, 10)]
    apply(chain)
    assert chain.fn_a == chain.fn_b == "grown"


def test_a_pool_that_does_not_grow_changes_nothing():
    mid = probe_sweep_midpoint(
        [(_Probes(grow=False), lambda ch: ch.a, lambda ch, fn: None)]
    )
    assert mid.adapt(np.zeros((3, 20, 1)), draws=DRAWS, chains=3) is None


@requires_jax
def test_the_jax_hook_substitutes_every_arms_parameters():
    import jax.numpy as jnp

    from neighbayes._jax_dispatch import ensure_x64
    from neighbayes.samplers._utils._jax_utils import _substitute, probe_midpoint

    ensure_x64()
    start = (jnp.zeros(5), jnp.float64(-1.0), jnp.float64(1.0))
    consts = {"zero": {"logdet": start}, "count": {"logdet": start}, "y": jnp.ones(3)}
    probes = _Probes(grow=True)
    hook = probe_midpoint(
        [
            (probes, lambda t: t[0], lambda c: c["zero"]["logdet"]),
            (probes, lambda t: t[1], lambda c: c["count"]["logdet"]),
        ]
    )
    assert hook.keep(("rz", "rc", "beta")) == ("rz", "rc")
    update = hook.adapt((np.zeros((2, 20)), np.zeros((2, 20))), draws=DRAWS, chains=2)
    assert probes.calls == [(2, 2, 10)]
    _, new = _substitute(update, [{}], consts)
    for part in ("zero", "count"):
        coeffs, lo, hi = new[part]["logdet"]
        np.testing.assert_array_equal(coeffs, np.ones(5))
        assert (float(lo), float(hi)) == (-0.5, 0.9)
    np.testing.assert_array_equal(new["y"], consts["y"])


# ── Fits ─────────────────────────────────────────────────────────────────────

FIT = dict(draws=60, tune=60, chains=2, random_seed=3, progressbar=False)
# A bar no 50-probe pool meets on these lattices: in the bulk of the interval
# their probes agree to a few thousandths of a posterior sd.
TIGHT = 2e-4


@pytest.fixture
def tight_bar(monkeypatch):
    """Size every pool against ``TIGHT`` rather than the run's own error."""
    monkeypatch.setattr(
        "neighbayes._logdet._probe_check.run_bar", lambda *a, **k: TIGHT
    )


def _negbin_latent(seed=9):
    from neighbayes import dgp
    from neighbayes.models import SARNegBin

    d = dgp.simulate_sar_negbin(n_side=8, seed=seed, latent=True)
    return SARNegBin(
        y=d["y"], X=d["X"], W=d["W_sparse"], latent=True, logdet_method="stochastic"
    )


def _fit(model, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return model.fit(**{**FIT, **kw})


def _assert_pool_recorded(model, idata, suffix=""):
    probes = model._fitted_probes if not suffix else model._fitted_probes_zero
    assert idata.attrs[f"logdet_probes{suffix}"] == probes.pool.n_probes > 50
    assert idata.attrs[f"logdet_probe_bar{suffix}"] == pytest.approx(TIGHT)


def _tilt(mp):
    """Make a pool's series fall steeply in ρ once it has been sized.

    Sweeps that run under the sized pool feel a strong pull toward the lower
    bound; the pool warmup started on, and so nothing before the midpoint,
    feels none.  A slice sampler's draws barely notice the few-thousandths
    change a real resizing makes, so this is what shows the pool arrived.
    """
    from neighbayes._logdet._probe_check import WarmupProbes

    original = WarmupProbes.coefficients

    def coefficients(self):
        coeffs, lo, hi = original(self)
        if self.probe_check is None:
            return coeffs, lo, hi
        tilted = np.array(coeffs)
        tilted[1] -= 300.0
        return tilted, lo, hi

    mp.setattr(WarmupProbes, "coefficients", coefficients)


def _samples_under_the_sized_pool(make, params, **kw):
    """Fit ``make()`` twice, the second with the sized pool tilted; ``params`` fall."""
    m = make()
    plain = _fit(m, **kw)
    _assert_pool_recorded(m, plain)
    with pytest.MonkeyPatch.context() as mp:
        _tilt(mp)
        tilted = _fit(make(), **kw)
    for name in params:
        before = float(plain.posterior[name].mean())
        assert float(tilted.posterior[name].mean()) < before - 0.3, name
    return m, plain


@pytest.mark.parametrize("backend", ["numpy", pytest.param("jax", marks=requires_jax)])
def test_latent_fit_samples_under_the_sized_pool(backend, tight_bar):
    n_jobs = 2 if backend == "numpy" else 1
    _samples_under_the_sized_pool(
        _negbin_latent, ["rho"], backend=backend, n_jobs=n_jobs
    )


def test_impacts_reuse_the_fitted_pool(tight_bar):
    m = _negbin_latent()
    _fit(m, backend="numpy", n_jobs=1)
    r = np.linspace(-0.5, 0.8, 7)
    np.testing.assert_array_equal(
        m._logdet_grad_numpy_vec_fn(r), m._fitted_probes.grad_vec_fn()(r)
    )
    W = m._W_sparse.toarray()
    exact = [-np.trace(W @ np.linalg.inv(np.eye(len(W)) - ri * W)) for ri in r]
    np.testing.assert_allclose(m._logdet_grad_numpy_vec_fn(r), exact, rtol=0.02)


@requires_jax
def test_a_grown_pool_compiles_nothing_new(tight_bar):
    import jax.monitoring

    from neighbayes.tests.test_samplers.test_jax_compile_once import _COMPILE_EVENT

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr("neighbayes._logdet._probe_check.run_bar", lambda *a, **k: 10.0)
        _fit(_negbin_latent(), backend="jax")
    events = []

    def listener(name, _secs, **_kw):
        if name == _COMPILE_EVENT:
            events.append(name)

    jax.monitoring.register_event_duration_secs_listener(listener)
    try:
        m = _negbin_latent(seed=10)
        idata = _fit(m, backend="jax", draws=70, tune=50)
    finally:
        jax.monitoring.unregister_event_duration_listener(listener)
    _assert_pool_recorded(m, idata)
    assert not events


@pytest.mark.parametrize("backend", ["numpy", pytest.param("jax", marks=requires_jax)])
def test_two_part_parts_on_one_W_share_the_pool(backend, tight_bar):
    from neighbayes import dgp
    from neighbayes.models import SARHurdleNegBin

    d = dgp.simulate_sar_hurdle_negbin(n_side=8, seed=12, latent=True)

    def make():
        return SARHurdleNegBin(
            y=d["y"],
            X=d["X"],
            X_zero=d["X_zero"],
            W=d["W_sparse"],
            latent=True,
            logdet_method="stochastic",
        )

    m, idata = _samples_under_the_sized_pool(
        make, ["rho", "rho_zero"], backend=backend, n_jobs=2
    )
    assert m._fitted_probes is m._fitted_probes_zero
    assert "logdet_probes_zero" not in idata.attrs


def test_random_effects_panel_samples_under_the_sized_pool(tight_bar):
    from neighbayes import dgp
    from neighbayes.models import SAR

    d = dgp.simulate_sar(N=36, T=3, n_side=6, seed=4, unit_effects="random")

    def make():
        return SAR(
            unit_effects="random",
            y=d["y"],
            X=d["X"],
            W=d["W_sparse"],
            N=36,
            T=3,
            logdet_method="stochastic",
        )

    _samples_under_the_sized_pool(make, ["rho"], n_jobs=2)


@pytest.mark.parametrize("backend", ["numpy", pytest.param("jax", marks=requires_jax)])
def test_multilevel_sizes_one_pool_per_level(backend, monkeypatch):
    import neighbayes._logdet as ld
    from neighbayes import dgp
    from neighbayes.models import Level, Multilevel

    # The units' probes agree to under a thousandth of a posterior sd here.
    monkeypatch.setattr("neighbayes._logdet._probe_check.run_bar", lambda *a, **k: 2e-5)
    # Upper levels choose their own method, exact on W this small.
    resolve = ld.resolve_logdet_bounds
    monkeypatch.setattr(
        ld,
        "resolve_logdet_bounds",
        lambda method, **kw: resolve("stoch_cheb", **kw),
    )
    d = dgp.simulate_multilevel(
        n_side=8,
        blocks=(2,),
        processes=("SAR", "SAR"),
        rhos=(0.4, 0.5),
        sigmas=(1.0, 0.6),
        seed=5,
    )

    def make():
        f = d["frames"]
        return Multilevel(
            Level("y ~ x1 + x2", data=f[0], W=d["W"], process="SAR"),
            Level("~ z1", data=f[1], W=d["levels"][0]["W"], key="g1", process="SAR"),
            logdet_method="stochastic",
        )

    kw = dict(backend=backend, n_jobs=2 if backend == "numpy" else 1)
    m = make()
    plain = _fit(m, **kw)
    r = np.linspace(-0.5, 0.8, 5)
    for ell in (0, 1):
        probes = m._fitted_level_probes[ell]
        assert plain.attrs[f"logdet_probes_{ell}"] == probes.pool.n_probes > 50
        np.testing.assert_array_equal(
            m._level_logdet_grad(ell)(r), probes.grad_vec_fn()(r)
        )
    assert m._fitted_level_probes[0] is not m._fitted_level_probes[1]
    assert m._fitted_probes is m._fitted_level_probes[0]
    with pytest.MonkeyPatch.context() as mp:
        _tilt(mp)
        tilted = _fit(make(), **kw)
    for name in ("rho_0", "rho_1"):
        before = float(plain.posterior[name].mean())
        assert float(tilted.posterior[name].mean()) < before - 0.3, name
