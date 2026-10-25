"""On-disk checkpoints of the generic JAX chunk runner.

A run killed at a chunk boundary and started again inside the same
``checkpoints`` block must return the draws of an uninterrupted run, bit for
bit, wherever it was killed: in warmup before the midpoint hook, after it, or
among the draws.  The toy sweep is the one ``test_warmup_midpoint.py`` uses,
with a hook whose substitution depends on what it was given, so a resumed run
that failed to replay it would draw differently.
"""

from __future__ import annotations

import numpy as np
import pytest

pytestmark = pytest.mark.requires_jax

#: 128-sweep chunks: with the hook, warmup runs 128, 22 (to the midpoint), 128,
#: 22; without it 128, 128, 44.  Draws run 128, 72.
TUNE, DRAWS, CHAINS = 300, 200, 3


class _Killed(Exception):
    """Stands in for the process dying."""


def _sweep(st, key, tuning, consts):
    import jax
    import jax.numpy as jnp

    x = st["x"] + consts["scale"] * jax.random.normal(key) + jnp.sum(st["params"])
    return dict(st, x=x), (x, st["params"][0])


def _hook(calls):
    """Sets each chain's params, and the noise scale, from what warmup kept."""
    from neighbayes.samplers._utils._jax_utils import MidpointUpdate, WarmupMidpoint

    def adapt(kept, *, draws, chains):
        import jax.numpy as jnp

        calls.append(np.array(kept))
        level = float(np.abs(kept).mean())
        return MidpointUpdate(
            state=(
                lambda st: st["params"],
                [jnp.full(4, 1e-3 * c) for c in range(chains)],
            ),
            consts=(lambda c: c["scale"], jnp.float64(1.0 / (1.0 + level))),
        )

    return WarmupMidpoint(keep=lambda trace: trace[0], adapt=adapt)


def _run(hook=None, scale=1.0):
    import jax
    import jax.numpy as jnp

    from neighbayes._jax_dispatch import ensure_x64
    from neighbayes.samplers._utils._jax_utils import run_chains_chunked

    ensure_x64()
    states = [{"x": jnp.float64(0.0), "params": jnp.zeros(4)} for _ in range(CHAINS)]
    warm = [jax.random.PRNGKey(c) for c in range(CHAINS)]
    draw = [jax.random.fold_in(jax.random.PRNGKey(c), 1) for c in range(CHAINS)]
    return run_chains_chunked(
        _sweep,
        states,
        warm,
        draw,
        tune=TUNE,
        draws=DRAWS,
        consts={"scale": jnp.float64(scale)},
        on_warmup_midpoint=hook,
    )


def _kill_after(monkeypatch, n_chunks):
    """Make the chunk runner die once ``n_chunks`` chunks have finished."""
    from neighbayes.samplers._utils import _jax_utils

    real, calls = _jax_utils.run_chains_in_threads, []

    def dying(*args, **kwargs):
        if len(calls) == n_chunks:
            raise _Killed
        calls.append(None)
        return real(*args, **kwargs)

    monkeypatch.setattr(_jax_utils, "run_chains_in_threads", dying)


@pytest.mark.parametrize(
    "n_chunks", [1, 3, 5], ids=["before-the-midpoint", "after-it", "in-the-draws"]
)
def test_a_killed_run_resumes_to_the_uninterrupted_draws(
    tmp_path, monkeypatch, n_chunks
):
    from neighbayes.config import checkpoints

    calls = []
    final, plain = _run(_hook(calls))
    expected_calls = len(calls)

    calls.clear()
    with checkpoints(tmp_path, every=0):
        with monkeypatch.context() as mp:
            _kill_after(mp, n_chunks)
            with pytest.raises(_Killed):
                _run(_hook(calls))
        assert any(tmp_path.iterdir()), "nothing was saved before the kill"
        resumed_final, resumed = _run(_hook(calls))

    for a, b in zip(plain, resumed):
        np.testing.assert_array_equal(a, b)
    for st, rs in zip(final, resumed_final):
        np.testing.assert_array_equal(
            np.asarray(st["params"]), np.asarray(rs["params"])
        )
    # Killed after the midpoint, the resumed run replays the hook on the input
    # it saved; before it, the hook runs once, in the resumed run.
    assert len(calls) == expected_calls + (n_chunks >= 2)
    for c in calls[1:]:
        np.testing.assert_array_equal(c, calls[0])
    # A finished run leaves nothing behind.
    assert not any(tmp_path.iterdir())


def test_a_run_on_other_data_does_not_resume(tmp_path, monkeypatch):
    from neighbayes.config import checkpoints

    _, other = _run(scale=2.0)
    with checkpoints(tmp_path, every=0):
        with monkeypatch.context() as mp:
            _kill_after(mp, 4)  # among the draws
            with pytest.raises(_Killed):
                _run(scale=1.0)
        saved = sorted(tmp_path.iterdir())
        _, resumed = _run(scale=2.0)
    for a, b in zip(other, resumed):
        np.testing.assert_array_equal(a, b)
    # The interrupted run's checkpoint is untouched.
    assert sorted(tmp_path.iterdir()) == saved and len(saved) == 1


def test_saves_wait_for_their_interval(tmp_path, monkeypatch):
    from neighbayes.config import checkpoints

    with checkpoints(tmp_path, every=1e9):
        with monkeypatch.context() as mp:
            _kill_after(mp, 4)  # among the draws
            with pytest.raises(_Killed):
                _run()
    assert not any(tmp_path.iterdir())


def test_a_fit_resumes_with_its_probe_check(tmp_path, monkeypatch):
    """End to end: a latent NB fit whose midpoint grows the probe pool."""
    import warnings

    from neighbayes import dgp
    from neighbayes.config import checkpoints
    from neighbayes.models import SARNegBin

    # A bar no 50-probe pool meets, so the midpoint substitutes a grown pool.
    monkeypatch.setattr("neighbayes._logdet._probe_check.run_bar", lambda *a, **k: 2e-4)
    d = dgp.simulate_sar_negbin(n_side=8, seed=9, latent=True)

    def fit():
        m = SARNegBin(
            y=d["y"], X=d["X"], W=d["W_sparse"], latent=True, logdet_method="stochastic"
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            idata = m.fit(
                draws=300, tune=300, chains=2, random_seed=3, progressbar=False,
                backend="jax",
            )  # fmt: skip
        return idata

    plain = fit()
    with checkpoints(tmp_path, every=0):
        with monkeypatch.context() as mp:
            _kill_after(mp, 5)
            with pytest.raises(_Killed):
                fit()
        resumed = fit()
    for name in plain.posterior.data_vars:
        np.testing.assert_array_equal(
            plain.posterior[name].values, resumed.posterior[name].values
        )
    assert resumed.attrs["logdet_probes"] == plain.attrs["logdet_probes"] > 50
