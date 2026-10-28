"""The warmup-midpoint hook of the generic JAX chunk runner.

A toy sweep stands in for a Gibbs family: its state carries a scalar ``x``, a
fixed-capacity parameter vector and the chain's index, and ``consts`` a
scale.  The hook must leave a run's draws untouched when it substitutes
nothing, must be seen by every sweep after it when it does, must see each
chain's warmup trace apart from the others', and must not recompile.
"""

from __future__ import annotations

import numpy as np
import pytest

pytestmark = pytest.mark.requires_jax

TUNE, DRAWS, CHAINS = 300, 200, 3
_COMPILE_EVENT = "/jax/core/compile/backend_compile_duration"


def _sweep(st, key, tuning, consts):
    import jax
    import jax.numpy as jnp

    x = st["x"] + consts["scale"] * jax.random.normal(key) + jnp.sum(st["params"])
    return dict(st, x=x), (x, st["params"][0], st["chain"])


def _run(hook=None):
    import jax
    import jax.numpy as jnp

    from neighbayes._jax_dispatch import ensure_x64
    from neighbayes.samplers._utils._jax_utils import run_chains_chunked

    ensure_x64()
    states = [
        {"x": jnp.float64(0.0), "params": jnp.zeros(4), "chain": jnp.float64(c)}
        for c in range(CHAINS)
    ]
    warm = [jax.random.PRNGKey(c) for c in range(CHAINS)]
    draw = [jax.random.fold_in(jax.random.PRNGKey(c), 1) for c in range(CHAINS)]
    return run_chains_chunked(
        _sweep,
        states,
        warm,
        draw,
        tune=TUNE,
        draws=DRAWS,
        consts={"scale": jnp.float64(1.0)},
        on_warmup_midpoint=hook,
    )


def _hook(adapt):
    from neighbayes.samplers._utils._jax_utils import WarmupMidpoint

    return WarmupMidpoint(keep=lambda trace: trace[2], adapt=adapt)


def _known_update(kept, *, draws, chains):
    """Every chain's parameters to ones, and the noise off."""
    import jax.numpy as jnp

    from neighbayes.samplers._utils._jax_utils import MidpointUpdate

    return MidpointUpdate(
        state=(lambda st: st["params"], [jnp.ones(4)] * chains),
        consts=(lambda c: c["scale"], jnp.float64(0.0)),
    )


def test_a_hook_that_substitutes_nothing_leaves_the_draws_unchanged():
    _, plain = _run()
    seen = {}

    def adapt(kept, *, draws, chains):
        seen.update(kept=kept, draws=draws, chains=chains)

    _, hooked = _run(_hook(adapt))
    for a, b in zip(plain, hooked):
        np.testing.assert_array_equal(a, b)
    # The hook saw each chain's own first-half trace, chains apart.
    assert seen["kept"].shape == (CHAINS, TUNE // 2)
    np.testing.assert_array_equal(
        seen["kept"], np.repeat(np.arange(CHAINS, dtype=float)[:, None], TUNE // 2, 1)
    )
    assert (seen["draws"], seen["chains"]) == (DRAWS, CHAINS)


def test_a_substituted_value_is_seen_by_the_next_sweep():
    final, (x, p0, _) = _run(_hook(_known_update))
    # Draws ran under the new parameters and the new consts: x climbs by
    # exactly sum(ones(4)) a sweep, without noise.
    np.testing.assert_array_equal(p0, np.ones((CHAINS, DRAWS)))
    np.testing.assert_allclose(np.diff(x, axis=1), 4.0)
    for st in final:
        np.testing.assert_array_equal(np.asarray(st["params"]), np.ones(4))


def test_substituting_leaves_compiles_nothing_new():
    import jax.monitoring

    _run()  # compile, and cache the chunk on the sweep
    # Build the update's arrays once, so creating them is not counted.
    _known_update(None, draws=DRAWS, chains=CHAINS)
    events = []

    def listener(name, _secs, **_kw):
        if name == _COMPILE_EVENT:
            events.append(name)

    jax.monitoring.register_event_duration_secs_listener(listener)
    try:
        _run(_hook(_known_update))
    finally:
        jax.monitoring.unregister_event_duration_listener(listener)
    assert not events


def test_an_update_that_changes_a_shape_is_refused():
    import jax.numpy as jnp

    from neighbayes.samplers._utils._jax_utils import MidpointUpdate

    def grow(kept, *, draws, chains):
        return MidpointUpdate(state=(lambda st: st["params"], [jnp.ones(5)] * chains))

    with pytest.raises(ValueError, match="fixed capacity"):
        _run(_hook(grow))
