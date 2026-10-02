"""Pin every stored pointwise log-likelihood against a reference pmf.

The stored ``log_likelihood`` group is what ``az.loo`` and the
BIC bridge factor read, so each count family must store the **full log-pmf
of the observed data** — every normalizing constant included, and for ZINB
the latent allocation integrated out.  A missing draw-independent constant
shifts elpd by that constant and corrupts pointwise LOO while leaving shape
and finiteness checks perfectly happy; these tests exist because a
shape/finite test cannot catch a constant.

Regression guard for the ``-log Γ(y+1)`` term that was absent from every NB
storage site, and for the ZINB arm that stored ``NB(y)·1(z=1) + logit(d)``
— a quantity conditional on the imputed ``z`` rather than a marginal pmf.
"""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest
from scipy.stats import nbinom, poisson

from neighbayes.samplers.negbin._core import _nb_loglik_pointwise
from neighbayes.samplers.poisson_reduced._core import _poisson_loglik_pointwise
from neighbayes.samplers.zinb._core import _zinb_loglik_pointwise

_HAS_JAX = importlib.util.find_spec("jax") is not None
requires_jax = pytest.mark.skipif(not _HAS_JAX, reason="JAX not installed")


def _case(seed=0, n=40):
    """Small synthetic count data with zeros, small and moderate counts."""
    rng = np.random.default_rng(seed)
    eta = rng.uniform(-1.5, 2.5, size=n)
    y = rng.poisson(np.exp(eta)).astype(np.float64)
    y[:5] = 0.0  # guarantee zeros
    return y, eta


def _nb_reference(y, eta, alpha):
    """``scipy`` NB2 log-pmf: mean ``exp(eta)``, dispersion ``alpha``."""
    mu = np.exp(eta)
    return nbinom.logpmf(y, n=alpha, p=alpha / (alpha + mu))


# ---------------------------------------------------------------------------
# Poisson
# ---------------------------------------------------------------------------


def test_poisson_pointwise_matches_scipy():
    y, eta = _case()
    got = _poisson_loglik_pointwise(y, eta)
    np.testing.assert_allclose(got, poisson.logpmf(y, np.exp(eta)), rtol=0, atol=1e-10)


# ---------------------------------------------------------------------------
# Negative binomial — every storage site, numpy and jax
# ---------------------------------------------------------------------------


def _nb_storage_fns():
    """(id, callable) for every function whose output is *stored* as log_lik."""
    fns = [("numpy:_nb_loglik_pointwise", _nb_loglik_pointwise)]
    if _HAS_JAX:
        import jax.numpy as jnp

        from neighbayes.samplers.negbin._jax import (
            _nb_loglik_pointwise_jax,
            _nb_loglik_pointwise_jax_op,
        )

        def _as_np(fn):
            def _wrapped(y, eta, alpha):
                return np.asarray(
                    fn(jnp.asarray(y), jnp.asarray(eta), jnp.float64(alpha))
                )

            return _wrapped

        fns += [
            ("jax:negbin._nb_loglik_pointwise_jax", _nb_loglik_pointwise_jax),
            (
                "jax:negbin._nb_loglik_pointwise_jax_op",
                _as_np(_nb_loglik_pointwise_jax_op),
            ),
        ]
    return fns


@pytest.mark.parametrize("alpha", [0.3, 1.0, 7.5])
@pytest.mark.parametrize(
    "name,fn", _nb_storage_fns(), ids=lambda v: v if isinstance(v, str) else ""
)
def test_nb_pointwise_matches_scipy(name, fn, alpha):
    """Every NB storage site must be the full NB2 log-pmf."""
    y, eta = _case()
    np.testing.assert_allclose(
        fn(y, eta, alpha), _nb_reference(y, eta, alpha), atol=1e-8
    )


def test_nb_pointwise_is_a_normalized_pmf():
    """Exponentiated, the stored NB values must sum to 1 over the support."""
    eta = np.full(1, 1.2)
    alpha = 2.0
    support = np.arange(0, 4000, dtype=np.float64)
    ll = _nb_loglik_pointwise(support, np.full(support.shape, eta[0]), alpha)
    assert np.exp(ll).sum() == pytest.approx(1.0, abs=1e-8)


def test_nb_storage_is_not_the_alpha_sampler_density():
    """The α slice sampler's internal density may omit ``-log Γ(y+1)``.

    It is a target up to a ρ,α-independent constant, so the omission does not
    move the sampled α.  Pin the relationship so the two are never conflated:
    the stored value is the sampler's density minus ``log(y!)``.
    """
    from scipy.special import gammaln

    y, eta = _case()
    alpha = 1.7
    mu = np.exp(eta)
    sampler_density = (
        gammaln(y + alpha)
        - gammaln(alpha)
        + y * np.log(mu / (mu + alpha))
        + alpha * np.log(alpha / (mu + alpha))
    )
    np.testing.assert_allclose(
        _nb_loglik_pointwise(y, eta, alpha),
        sampler_density - gammaln(y + 1.0),
        atol=1e-12,
    )


# ---------------------------------------------------------------------------
# ZINB — marginal over the latent allocation
# ---------------------------------------------------------------------------


def _zinb_reference(y, eta_sel, eta_cnt, alpha):
    """Closed-form ZINB marginal log-pmf, built from scipy's NB."""
    pi = 1.0 / (1.0 + np.exp(-eta_sel))
    nb = _nb_reference(y, eta_cnt, alpha)
    nb0 = _nb_reference(np.zeros_like(y), eta_cnt, alpha)
    return np.where(y > 0, np.log(pi) + nb, np.log(pi * np.exp(nb0) + (1.0 - pi)))


@pytest.mark.parametrize("alpha", [0.4, 2.0])
def test_zinb_pointwise_matches_closed_form(alpha):
    y, eta_cnt = _case()
    _, eta_sel = _case(seed=1)
    np.testing.assert_allclose(
        _zinb_loglik_pointwise(y, eta_sel, eta_cnt, alpha),
        _zinb_reference(y, eta_sel, eta_cnt, alpha),
        atol=1e-10,
    )


def test_zinb_pointwise_is_a_normalized_pmf():
    """Exponentiated, the marginal must sum to 1 over the count support."""
    alpha, eta_cnt, eta_sel = 1.5, 0.8, 0.35
    support = np.arange(0, 4000, dtype=np.float64)
    ll = _zinb_loglik_pointwise(
        support,
        np.full(support.shape, eta_sel),
        np.full(support.shape, eta_cnt),
        alpha,
    )
    assert np.exp(ll).sum() == pytest.approx(1.0, abs=1e-8)


def test_zinb_pointwise_broadcasts_over_draws():
    """The JAX backend calls the helper with a leading draw axis."""
    y, eta_cnt_1 = _case()
    _, eta_sel_1 = _case(seed=1)
    n_draws = 3
    eta_cnt = np.tile(eta_cnt_1, (n_draws, 1))
    eta_sel = np.tile(eta_sel_1, (n_draws, 1))
    alpha = np.array([0.5, 1.0, 3.0])[:, None]

    got = _zinb_loglik_pointwise(y, eta_sel, eta_cnt, alpha)
    assert got.shape == (n_draws, y.size)
    for d in range(n_draws):
        np.testing.assert_allclose(
            got[d],
            _zinb_loglik_pointwise(y, eta_sel_1, eta_cnt_1, float(alpha[d, 0])),
            atol=1e-12,
        )


def test_zinb_reduces_to_nb_when_activation_is_certain():
    """As ``pi -> 1`` the marginal collapses onto the plain NB log-pmf."""
    y, eta_cnt = _case()
    alpha = 1.3
    eta_sel = np.full(y.shape, 40.0)  # pi ~ 1 - 4e-18
    np.testing.assert_allclose(
        _zinb_loglik_pointwise(y, eta_sel, eta_cnt, alpha),
        _nb_loglik_pointwise(y, eta_cnt, alpha),
        atol=1e-12,
    )


# ---------------------------------------------------------------------------
# Downstream: the BIC bridge factor reads the same stored group
# ---------------------------------------------------------------------------


def test_bic_logml_inherits_the_normalizing_constant():
    """``_bic_logml`` sums the stored group, so the NB log-ML shifts by
    ``-Σ log(y!)`` relative to the pre-fix (unnormalized) storage.

    The shift is real and intended: Bayes factors between two NB models are
    unchanged (the constant cancels), but an NB-vs-Poisson comparison is now
    on a common scale.  Pinned here so the change is documented, not silent.
    """
    from scipy.special import gammaln

    from neighbayes.diagnostics.bayesfactor import _bic_logml

    az = pytest.importorskip("arviz")

    y, eta = _case()
    alpha = 1.4
    n_chains, n_draws = 2, 25
    rng = np.random.default_rng(7)
    etas = eta + 0.05 * rng.standard_normal((n_chains, n_draws, y.size))

    stored = _nb_loglik_pointwise(y, etas, alpha)
    unnormalized = stored + gammaln(y + 1.0)  # what the site used to store

    def _logml(ll):
        idata = az.from_dict(
            {
                "posterior": {"beta": rng.standard_normal((n_chains, n_draws, 2))},
                "log_likelihood": {"obs": ll},
                "observed_data": {"y": y},
            }
        )
        return _bic_logml(idata)

    assert _logml(stored) == pytest.approx(
        _logml(unnormalized) - gammaln(y + 1.0).sum()
    )
