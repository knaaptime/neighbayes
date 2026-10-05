"""Unit tests for the Pólya–Gamma sampler wrapper."""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest

from neighbayes.samplers._utils._polyagamma import sample_polyagamma


class TestSamplePolyagamma:
    """Tests for the PG draw wrapper."""

    def test_pg1_mean(self, rng):
        """PG(1, 0) has known mean ≈ 0.25."""
        h = np.ones(5000)
        z = np.zeros(5000)
        omega = sample_polyagamma(h, z, rng=rng)
        assert omega.shape == (5000,)
        assert np.all(omega > 0)
        assert abs(np.mean(omega) - 0.25) < 0.05

    def test_pg5_positive(self, rng):
        """PG(5, 2) draws are all positive."""
        h = np.full(100, 5.0)
        z = np.full(100, 2.0)
        omega = sample_polyagamma(h, z, rng=rng)
        assert omega.shape == (100,)
        assert np.all(omega > 0)

    def test_shape_mismatch_raises(self):
        """h and z with different shapes raises ValueError."""
        with pytest.raises(ValueError, match="same shape"):
            sample_polyagamma(np.ones(10), np.ones(5))

    def test_h_nonpositive_raises(self):
        """h <= 0 raises ValueError."""
        with pytest.raises(ValueError, match="positive"):
            sample_polyagamma(np.array([0.0]), np.array([1.0]))

    def test_scalar_broadcast(self, rng):
        """Scalar h and z work correctly."""
        omega = sample_polyagamma(np.array([3.0]), np.array([1.0]), rng=rng)
        assert omega.shape == (1,)
        assert omega[0] > 0

    @pytest.fixture
    def rng(self):
        return np.random.default_rng(42)


class TestPolyagammaMethodDispatch:
    """Tests that sample_polyagamma dispatches on h integrality."""

    @patch("polyagamma.random_polyagamma")
    def test_integer_h_uses_hybrid(self, mock_pg, rng):
        """When all h are integer, method=None (hybrid) is used."""
        mock_pg.return_value = np.ones(10)

        h = np.ones(10)  # all integer (logit case)
        z = np.zeros(10)
        sample_polyagamma(h, z, rng=rng)

        mock_pg.assert_called_once()
        call_kwargs = mock_pg.call_args
        assert call_kwargs.kwargs["method"] is None

    @patch("polyagamma.random_polyagamma")
    def test_noninteger_h_uses_hybrid(self, mock_pg, rng):
        """Non-integer h uses hybrid method (saddle for large h)."""
        mock_pg.return_value = np.ones(10)

        h = np.array([1.5, 2.3, 3.7, 4.1, 5.9, 6.2, 7.8, 8.0, 9.5, 10.1])
        z = np.zeros(10)
        sample_polyagamma(h, z, rng=rng)

        mock_pg.assert_called_once()
        call_kwargs = mock_pg.call_args
        assert call_kwargs.kwargs["method"] is None

    @patch("polyagamma.random_polyagamma")
    def test_mixed_h_uses_hybrid(self, mock_pg, rng):
        """Mixed integer/non-integer h uses hybrid method."""
        mock_pg.return_value = np.ones(5)

        h = np.array([1.0, 2.5, 3.0, 4.7, 5.0])  # some integer, some not
        z = np.zeros(5)
        sample_polyagamma(h, z, rng=rng)

        mock_pg.assert_called_once()
        call_kwargs = mock_pg.call_args
        assert call_kwargs.kwargs["method"] is None

    @patch("polyagamma.random_polyagamma")
    def test_large_integer_h_uses_hybrid(self, mock_pg, rng):
        """Large integer h values (e.g. h=5) also use hybrid method."""
        mock_pg.return_value = np.ones(10)

        h = np.full(10, 5.0)  # all integer, h=5
        z = np.zeros(10)
        sample_polyagamma(h, z, rng=rng)

        mock_pg.assert_called_once()
        call_kwargs = mock_pg.call_args
        assert call_kwargs.kwargs["method"] is None

    def test_logit_h_produces_valid_draws(self, rng):
        """End-to-end: logit-style h=1 draws are valid PG(1, z) samples."""
        n = 5000
        h = np.ones(n)  # logit case
        z = rng.normal(size=n)
        omega = sample_polyagamma(h, z, rng=rng)
        assert omega.shape == (n,)
        assert np.all(omega > 0)
        # PG(1, 0) has mean 0.25; with z != 0 the mean shifts but stays positive
        assert np.mean(omega) > 0

    def test_negbin_h_produces_valid_draws(self, rng):
        """End-to-end: negbin-style non-integer h draws are valid."""
        n = 5000
        y = rng.integers(0, 10, size=n)
        alpha = 2.5
        h = y + alpha  # non-integer
        z = rng.normal(size=n)
        omega = sample_polyagamma(h, z, rng=rng)
        assert omega.shape == (n,)
        assert np.all(omega > 0)


class TestBiasWarning:
    """polyagamma 2.0.2 is biased for h < 8 outside Devroye's exact cases."""

    @pytest.mark.parametrize(
        ("h", "z", "biased"),
        [
            (1.0, 0.0, False),  # Devroye
            (1.0, 3.0, False),  # Devroye at any z for h = 1
            (3.0, 0.5, False),  # Devroye: integer h, z <= 1
            (3.0, 2.0, True),  # alternate: integer h, z > 1
            (2.5, 0.5, True),  # alternate: non-integer h
            (0.5, 0.0, True),  # alternate below h = 1
            (5.0, 2.0, True),  # saddlepoint, 4 < h < 8
            (2.5, 6.0, False),  # bias gone by |z| = 4
            (12.0, 0.0, False),  # h >= 8
        ],
    )
    def test_region_matches_the_hybrid_dispatch(self, h, z, biased):
        from neighbayes.samplers._utils import _polyagamma as pg

        if not pg._installed_release_is_biased():
            pytest.skip("installed polyagamma is newer than the biased releases")
        assert pg.draws_are_biased(np.array([h]), np.array([z])) is biased

    def test_warns_for_biased_draws_only(self, rng):
        from neighbayes.samplers._utils import _polyagamma as pg

        if not pg._installed_release_is_biased():
            pytest.skip("installed polyagamma is newer than the biased releases")
        with pytest.warns(RuntimeWarning, match="gibbs_backend='jax'"):
            sample_polyagamma(np.full(4, 2.5), np.zeros(4), rng=rng)
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            sample_polyagamma(np.ones(4), np.zeros(4), rng=rng)  # logit: exact
            sample_polyagamma(np.full(4, 2.5), np.zeros(4), rng=rng, warn=False)

    def test_a_fixed_release_does_not_warn(self, rng, monkeypatch):
        import warnings

        from neighbayes.samplers._utils import _polyagamma as pg

        monkeypatch.setattr(pg, "_installed_release_is_biased", lambda: False)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            sample_polyagamma(np.full(4, 2.5), np.zeros(4), rng=rng)


def _warning_chain(chain_id, seed, progress_manager=None, chain_id_kw=0):
    import warnings

    warnings.warn("relayed from a worker", RuntimeWarning)
    return {"chain": chain_id}


def test_parallel_chains_relay_worker_warnings_once():
    """A loky worker's warning reaches the parent, once however many chains."""
    import warnings

    from neighbayes.samplers.gaussian._chain_runner import run_chains

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        results = run_chains(
            _warning_chain, 2, seeds=[1, 2], parallel=True, progressbar=False
        )
    assert [r["chain"] for r in results] == [0, 1]
    relayed = [w for w in caught if str(w.message) == "relayed from a worker"]
    assert len(relayed) == 1
    assert relayed[0].category is RuntimeWarning


def _nb_model():
    from neighbayes import dgp
    from neighbayes.models import SARNegBin

    d = dgp.simulate_sar_negbin(n_side=6, seed=3)
    return SARNegBin(y=d["y"], X=d["X"], W=d["W_sparse"])


@pytest.mark.parametrize("parallel_jobs", [1, 2])
def test_numpy_count_fit_warns(parallel_jobs):
    from neighbayes.samplers._utils import _polyagamma as pg

    if not pg._installed_release_is_biased():
        pytest.skip("installed polyagamma is newer than the biased releases")
    with pytest.warns(RuntimeWarning, match="gibbs_backend='jax'"):
        _nb_model().fit(
            draws=5, tune=5, chains=2, n_jobs=parallel_jobs, progressbar=False,
            gibbs_backend="numpy", random_seed=1,
        )  # fmt: skip


@pytest.mark.requires_jax
def test_jax_fits_do_not_warn():
    """Starting values drawn in the parent must not raise the NumPy warning."""
    import warnings

    pytest.importorskip("pgjax")
    from neighbayes import dgp
    from neighbayes.models import SARZINB, SARNegBinStructural

    d = dgp.simulate_sar_negbin(n_side=6, seed=3)
    dz = dgp.simulate_sar_zinb(n_side=6, seed=3)
    models = [
        _nb_model(),
        SARNegBinStructural(y=d["y"], X=d["X"], W=d["W_sparse"]),
        SARZINB(y=dz["y"], X=dz["X"], W=dz["W_sparse"]),
    ]
    for model in models:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model.fit(
                draws=5, tune=5, chains=1, progressbar=False, gibbs_backend="jax",
                random_seed=1,
            )  # fmt: skip
        assert not [w for w in caught if "Pólya–Gamma" in str(w.message)], type(
            model
        ).__name__
