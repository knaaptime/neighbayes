"""Gibbs samplers for the reduced-form SAR hurdle NB models.

A hurdle splits each count into a binary half, ``d = 1(y > 0)`` with a
reduced-form spatial logit, and a positive half, a zero-truncated reduced-form
spatial NB2 on the cells with ``y > 0``.  With unlinked halves the likelihood
factorizes; the samplers draw both halves in one sweep.

* :mod:`._generic` — materialized ``U = A⁻¹X`` for any filter pair, unit and
  period effects in both halves (place-based cross-sections and panels).
* :mod:`._flow_structured` / :mod:`._flow_structured_jax` — the separable flow
  hurdle on ``n × n`` solves, pair and period effects in both halves.
* :mod:`._truncated` — the zero-truncated NB2 pieces they share.
"""

from .._registry import register


def _run_hurdle_gibbs(
    model,
    *,
    draws,
    tune,
    chains,
    random_seed,
    thin,
    n_jobs,
    progressbar,
    backend,
    log_likelihood=False,
    **options,
):
    """Registry runner for the hurdle models: a thin adapter over ``_fit_gibbs``."""
    return model._fit_gibbs(
        draws=draws,
        tune=tune,
        chains=chains,
        random_seed=random_seed,
        thin=thin,
        n_jobs=n_jobs,
        progressbar=progressbar,
        log_likelihood=log_likelihood,
        **options,
    )


register(
    "hurdle",
    "cross_section",
    run=_run_hurdle_gibbs,
    backends={"numpy"},
    skips_log_likelihood=True,
)
register(
    "hurdle",
    "panel",
    run=_run_hurdle_gibbs,
    backends={"numpy"},
    options={"store_group_effects"},
    skips_log_likelihood=True,
)
