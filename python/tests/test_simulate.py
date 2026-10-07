"""Unit tests for synthetic data generators."""

from __future__ import annotations

import numpy as np
import pytest

import greenwood as gw
from greenwood import Surv


class TestSimulateCompetingRisks:
    def test_default_returns_surv(self) -> None:
        sim = gw.simulate_competing_risks(seed=0)
        assert isinstance(sim.surv, Surv)
        assert sim.surv.n == 500

    def test_n_subjects(self) -> None:
        sim = gw.simulate_competing_risks(n=100, seed=0)
        assert sim.surv.n == 100
        assert sim.latent_times.shape[0] == 100
        assert sim.censoring_time.shape[0] == 100

    def test_n_causes(self) -> None:
        sim = gw.simulate_competing_risks(n=50, n_causes=3, seed=0)
        assert sim.latent_times.shape == (50, 3)
        unique_events = np.unique(sim.surv.status[sim.surv.status > 0])
        assert set(unique_events.tolist()).issubset({1, 2, 3})

    def test_states_names(self) -> None:
        sim = gw.simulate_competing_risks(n=50, n_causes=2, states=("relapse", "death"), seed=0)
        assert sim.surv.states == ("relapse", "death")

    def test_no_censoring(self) -> None:
        sim = gw.simulate_competing_risks(n=200, censoring_scale=None, seed=0)
        assert (sim.surv.status > 0).all()
        np.testing.assert_array_equal(sim.censoring_time, np.inf)

    def test_zero_censoring_scale(self) -> None:
        sim = gw.simulate_competing_risks(n=200, censoring_scale=0, seed=0)
        assert (sim.surv.status > 0).all()

    def test_heavy_censoring(self) -> None:
        sim = gw.simulate_competing_risks(n=1000, censoring_scale=0.3, seed=0)
        censored_frac = (sim.surv.status == 0).mean()
        assert censored_frac > 0.3

    def test_no_covariates_by_default(self) -> None:
        sim = gw.simulate_competing_risks(seed=0)
        assert sim.covariates is None

    def test_covariates_shape(self) -> None:
        sim = gw.simulate_competing_risks(n=100, n_covariates=5, seed=0)
        assert sim.covariates is not None
        assert sim.covariates.shape == (100, 5)

    def test_custom_coef(self) -> None:
        coef = np.array([[0.5, -0.3], [0.0, 0.8]])
        sim = gw.simulate_competing_risks(n=200, n_causes=2, n_covariates=2, coef=coef, seed=0)
        assert sim.covariates is not None
        assert sim.covariates.shape == (200, 2)

    def test_coef_shape_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="shape"):
            gw.simulate_competing_risks(
                n=50, n_causes=2, n_covariates=3, coef=np.zeros((2, 2)), seed=0
            )

    def test_shapes_length_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="shapes must have length"):
            gw.simulate_competing_risks(n=50, n_causes=2, shapes=(1.0,), seed=0)

    def test_scales_length_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="scales must have length"):
            gw.simulate_competing_risks(n=50, n_causes=2, scales=(10.0,), seed=0)

    def test_states_length_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="states must have length"):
            gw.simulate_competing_risks(n=50, n_causes=2, states=("a",), seed=0)

    def test_negative_shape_raises(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            gw.simulate_competing_risks(n=50, n_causes=2, shapes=(-1.0, 1.0), seed=0)

    def test_negative_scale_raises(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            gw.simulate_competing_risks(n=50, n_causes=2, scales=(10.0, -5.0), seed=0)

    def test_invalid_n_raises(self) -> None:
        with pytest.raises(ValueError, match="n must be"):
            gw.simulate_competing_risks(n=0)

    def test_invalid_n_causes_raises(self) -> None:
        with pytest.raises(ValueError, match="n_causes must be"):
            gw.simulate_competing_risks(n_causes=0)

    def test_seed_reproducibility(self) -> None:
        sim1 = gw.simulate_competing_risks(n=50, seed=23)
        sim2 = gw.simulate_competing_risks(n=50, seed=23)
        np.testing.assert_array_equal(sim1.surv.stop, sim2.surv.stop)
        np.testing.assert_array_equal(sim1.surv.status, sim2.surv.status)
        np.testing.assert_array_equal(sim1.latent_times, sim2.latent_times)

    def test_different_seeds_differ(self) -> None:
        sim1 = gw.simulate_competing_risks(n=50, seed=1)
        sim2 = gw.simulate_competing_risks(n=50, seed=2)
        assert not np.array_equal(sim1.surv.stop, sim2.surv.stop)

    def test_custom_shapes_affect_distribution(self) -> None:
        # Increasing hazard (shape > 1) should produce more events concentrated later
        sim_inc = gw.simulate_competing_risks(
            n=1000,
            n_causes=1,
            shapes=(3.0,),
            scales=(10.0,),
            censoring_scale=None,
            seed=0,
        )
        # Decreasing hazard (shape < 1) should produce more events concentrated early
        sim_dec = gw.simulate_competing_risks(
            n=1000,
            n_causes=1,
            shapes=(0.5,),
            scales=(10.0,),
            censoring_scale=None,
            seed=0,
        )
        # Increasing hazard has higher median time (events occur later)
        median_inc = float(np.median(sim_inc.surv.stop))
        median_dec = float(np.median(sim_dec.surv.stop))
        assert median_inc > median_dec

    def test_observed_time_is_minimum(self) -> None:
        sim = gw.simulate_competing_risks(n=100, seed=0)
        min_latent = sim.latent_times.min(axis=1)
        min_overall = np.minimum(min_latent, sim.censoring_time)
        np.testing.assert_allclose(sim.surv.stop, min_overall)

    def test_covariates_affect_event_distribution(self) -> None:
        coef = np.array([[1.0, -1.0]])  # cause 1 hazard up, cause 2 down
        sim = gw.simulate_competing_risks(
            n=2000,
            n_causes=2,
            n_covariates=1,
            coef=coef,
            censoring_scale=None,
            seed=0,
        )
        assert sim.covariates is not None
        high_x = sim.covariates[:, 0] > 0.5
        low_x = sim.covariates[:, 0] < -0.5
        # High covariate -> more cause 1 events (coef > 0 increases hazard)
        frac_cause1_high = (sim.surv.status[high_x] == 1).mean()
        frac_cause1_low = (sim.surv.status[low_x] == 1).mean()
        assert frac_cause1_high > frac_cause1_low

    def test_works_with_aalen_johansen(self) -> None:
        sim = gw.simulate_competing_risks(n=200, seed=0)
        aj = gw.AalenJohansen().fit(sim.surv)
        df = aj.to_frame(format="polars")
        assert df.shape[0] > 0

    def test_works_with_metrics(self) -> None:
        sim = gw.simulate_competing_risks(n=200, seed=0)
        rng = np.random.default_rng(0)
        times = np.array([3.0, 6.0, 10.0])
        probs = rng.uniform(0, 0.3, size=(sim.surv.n, len(times)))
        bs = gw.brier_score_incidence(sim.surv, probs, times, cause=1)
        assert bs.shape == (3,)
        assert np.all(bs >= 0)
