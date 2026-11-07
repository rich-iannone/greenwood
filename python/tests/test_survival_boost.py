"""Unit tests for SurvivalBoost."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import greenwood as gw
from greenwood import Surv


@pytest.fixture()
def competing_data() -> gw.SimulatedCompetingRisks:
    return gw.simulate_competing_risks(
        n=300, n_causes=2, n_covariates=3, censoring_scale=1.5, seed=23
    )


@pytest.fixture()
def fitted_model(competing_data: gw.SimulatedCompetingRisks) -> gw.SurvivalBoost:
    sb = gw.SurvivalBoost(
        n_estimators=20, learning_rate=0.1, max_depth=2, min_samples_leaf=5, random_state=0
    )
    sb.fit(competing_data.surv, competing_data.covariates)
    return sb


class TestSurvivalBoostInit:
    def test_repr_unfitted(self) -> None:
        sb = gw.SurvivalBoost(n_estimators=50)
        assert "unfitted" in repr(sb)

    def test_invalid_n_estimators(self) -> None:
        with pytest.raises(ValueError, match="n_estimators"):
            gw.SurvivalBoost(n_estimators=0)

    def test_invalid_learning_rate(self) -> None:
        with pytest.raises(ValueError, match="learning_rate"):
            gw.SurvivalBoost(learning_rate=0.0)

    def test_invalid_max_depth(self) -> None:
        with pytest.raises(ValueError, match="max_depth"):
            gw.SurvivalBoost(max_depth=0)

    def test_invalid_subsample(self) -> None:
        with pytest.raises(ValueError, match="subsample"):
            gw.SurvivalBoost(subsample=0.0)

    def test_invalid_hard_zero_fraction(self) -> None:
        with pytest.raises(ValueError, match="hard_zero_fraction"):
            gw.SurvivalBoost(hard_zero_fraction=1.0)


class TestSurvivalBoostFit:
    def test_requires_multistate(self) -> None:
        y = Surv(time=[1, 2, 3], event=[1, 0, 1])
        x = np.random.default_rng(0).standard_normal((3, 2))
        sb = gw.SurvivalBoost(n_estimators=5)
        with pytest.raises(NotImplementedError, match="multi-state"):
            sb.fit(y, x)

    def test_fit_sets_attributes(
        self, fitted_model: gw.SurvivalBoost, competing_data: gw.SimulatedCompetingRisks
    ) -> None:
        assert fitted_model.n_ == competing_data.surv.n
        assert fitted_model.n_causes_ == 2
        assert fitted_model.n_classes_ == 3
        assert fitted_model.n_features_in_ == 3
        assert len(fitted_model.trees_) == 20
        assert len(fitted_model.time_grid_) > 0

    def test_repr_fitted(self, fitted_model: gw.SurvivalBoost) -> None:
        r = repr(fitted_model)

        assert "SurvivalBoost" in r
        assert "causes = 2" in r

    def test_fit_no_events_raises(self) -> None:
        y = Surv(
            time=[1, 2, 3],
            event=pd.Categorical.from_codes(np.array([0, 0, 0]), categories=["censor", "a"]),
        )
        x = np.ones((3, 1))
        sb = gw.SurvivalBoost(n_estimators=5)
        with pytest.raises(ValueError, match="No events"):
            sb.fit(y, x)

    def test_covariate_length_mismatch_raises(self) -> None:
        sim = gw.simulate_competing_risks(n=50, n_causes=2, seed=0)
        x = np.ones((10, 2))
        sb = gw.SurvivalBoost(n_estimators=5)
        with pytest.raises(ValueError, match="same number"):
            sb.fit(sim.surv, x)


class TestSurvivalBoostPredict:
    def test_cif_shape(
        self, fitted_model: gw.SurvivalBoost, competing_data: gw.SimulatedCompetingRisks
    ) -> None:
        times = np.array([2.0, 5.0, 10.0])
        cif = fitted_model.predict_cumulative_incidence(competing_data.covariates, times=times)

        assert cif.shape == (competing_data.surv.n, 3, 3)

    def test_cif_probabilities_sum_to_one(
        self, fitted_model: gw.SurvivalBoost, competing_data: gw.SimulatedCompetingRisks
    ) -> None:
        times = np.array([2.0, 5.0, 10.0])
        cif = fitted_model.predict_cumulative_incidence(competing_data.covariates, times=times)
        totals = cif.sum(axis=1)

        np.testing.assert_allclose(totals, 1.0, atol=1e-10)

    def test_survival_shape(
        self, fitted_model: gw.SurvivalBoost, competing_data: gw.SimulatedCompetingRisks
    ) -> None:
        times = np.array([2.0, 5.0, 10.0])
        surv = fitted_model.predict_survival_function(competing_data.covariates, times=times)

        assert surv.shape == (competing_data.surv.n, 3)

    def test_survival_in_unit_range(
        self, fitted_model: gw.SurvivalBoost, competing_data: gw.SimulatedCompetingRisks
    ) -> None:
        surv = fitted_model.predict_survival_function(competing_data.covariates)

        assert np.all(surv >= 0)
        assert np.all(surv <= 1)

    def test_cif_nonnegative(
        self, fitted_model: gw.SurvivalBoost, competing_data: gw.SimulatedCompetingRisks
    ) -> None:
        cif = fitted_model.predict_cumulative_incidence(competing_data.covariates)

        assert np.all(cif >= 0)

    def test_predict_proba_shape(
        self, fitted_model: gw.SurvivalBoost, competing_data: gw.SimulatedCompetingRisks
    ) -> None:
        proba = fitted_model.predict_proba(competing_data.covariates, time_horizon=5.0)

        assert proba.shape == (competing_data.surv.n, 3)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-10)

    def test_predict_requires_newdata(self, fitted_model: gw.SurvivalBoost) -> None:
        with pytest.raises(ValueError, match="newdata"):
            fitted_model.predict_cumulative_incidence(None)

    def test_default_time_grid_used(
        self, fitted_model: gw.SurvivalBoost, competing_data: gw.SimulatedCompetingRisks
    ) -> None:
        cif = fitted_model.predict_cumulative_incidence(competing_data.covariates)

        assert cif.shape[2] == len(fitted_model.time_grid_)

    def test_survival_decreasing_over_time(
        self, fitted_model: gw.SurvivalBoost, competing_data: gw.SimulatedCompetingRisks
    ) -> None:
        times = np.linspace(0.5, 20.0, 20)
        surv = fitted_model.predict_survival_function(competing_data.covariates[:5], times=times)
        for i in range(surv.shape[0]):
            diffs = np.diff(surv[i, :])

            assert np.all(diffs <= 0.05), "Survival should generally decrease over time"

    def test_cif_at_zero_near_zero(
        self, fitted_model: gw.SurvivalBoost, competing_data: gw.SimulatedCompetingRisks
    ) -> None:
        cif = fitted_model.predict_cumulative_incidence(
            competing_data.covariates[:5], times=np.array([0.0])
        )
        for k in range(1, cif.shape[1]):
            assert np.all(cif[:, k, 0] < 0.3), "CIF at t=0 should be low"


class TestSurvivalBoostFormat:
    def test_cif_polars_format(
        self, fitted_model: gw.SurvivalBoost, competing_data: gw.SimulatedCompetingRisks
    ) -> None:
        times = np.array([2.0, 5.0])
        df = fitted_model.predict_cumulative_incidence(
            competing_data.covariates[:3], times=times, format="polars"
        )

        assert df.shape[0] > 0
        assert "time" in df.columns
        assert "cause" in df.columns
        assert "probability" in df.columns

    def test_survival_polars_format(
        self, fitted_model: gw.SurvivalBoost, competing_data: gw.SimulatedCompetingRisks
    ) -> None:
        times = np.array([2.0, 5.0])
        df = fitted_model.predict_survival_function(
            competing_data.covariates[:3], times=times, format="polars"
        )

        assert df.shape[0] > 0
        assert "time" in df.columns


class TestSurvivalBoostBroom:
    def test_tidy(self, fitted_model: gw.SurvivalBoost) -> None:
        df = gw.tidy(fitted_model, format="polars")

        assert "term" in df.columns
        assert df.shape[0] == fitted_model.n_features_in_

    def test_glance(self, fitted_model: gw.SurvivalBoost) -> None:
        df = gw.glance(fitted_model, format="polars")

        assert df.shape[0] == 1
        assert "n" in df.columns
        assert "n_causes" in df.columns


class TestSurvivalBoostIntegration:
    def test_works_with_brier_score(self, competing_data: gw.SimulatedCompetingRisks) -> None:
        sb = gw.SurvivalBoost(n_estimators=10, learning_rate=0.1, max_depth=2, random_state=0)
        sb.fit(competing_data.surv, competing_data.covariates)
        times = np.array([3.0, 6.0, 10.0])
        cif = sb.predict_cumulative_incidence(competing_data.covariates, times=times)
        probs_cause1 = cif[:, 1, :]
        bs = gw.brier_score_incidence(competing_data.surv, probs_cause1, times, cause=1)

        assert bs.shape == (3,)
        assert np.all(bs >= 0)

    def test_works_with_concordance(self, competing_data: gw.SimulatedCompetingRisks) -> None:
        sb = gw.SurvivalBoost(n_estimators=10, learning_rate=0.1, max_depth=2, random_state=0)
        sb.fit(competing_data.surv, competing_data.covariates)
        tau = float(np.median(competing_data.surv.stop))
        cif = sb.predict_cumulative_incidence(competing_data.covariates, times=np.array([tau]))
        probs = cif[:, 1, 0]
        c = gw.concordance_index_incidence(competing_data.surv, probs, cause=1, tau=tau)

        assert 0.0 <= c <= 1.0

    def test_three_causes(self) -> None:
        sim = gw.simulate_competing_risks(n=200, n_causes=3, n_covariates=2, seed=7)
        sb = gw.SurvivalBoost(n_estimators=10, learning_rate=0.1, max_depth=2, random_state=0)
        sb.fit(sim.surv, sim.covariates)

        assert sb.n_causes_ == 3
        assert sb.n_classes_ == 4

        cif = sb.predict_cumulative_incidence(sim.covariates, times=np.array([5.0]))

        assert cif.shape == (200, 4, 1)
        np.testing.assert_allclose(cif.sum(axis=1), 1.0, atol=1e-10)

    def test_subsample(self, competing_data: gw.SimulatedCompetingRisks) -> None:
        sb = gw.SurvivalBoost(
            n_estimators=10, learning_rate=0.1, max_depth=2, subsample=0.8, random_state=0
        )
        sb.fit(competing_data.surv, competing_data.covariates)
        surv = sb.predict_survival_function(competing_data.covariates, times=np.array([5.0]))

        assert surv.shape == (competing_data.surv.n, 1)

    def test_seed_reproducibility(self, competing_data: gw.SimulatedCompetingRisks) -> None:
        sb1 = gw.SurvivalBoost(n_estimators=10, random_state=23)
        sb1.fit(competing_data.surv, competing_data.covariates)
        sb2 = gw.SurvivalBoost(n_estimators=10, random_state=23)
        sb2.fit(competing_data.surv, competing_data.covariates)
        p1 = sb1.predict_survival_function(competing_data.covariates, times=np.array([5.0]))
        p2 = sb2.predict_survival_function(competing_data.covariates, times=np.array([5.0]))

        np.testing.assert_array_equal(p1, p2)


class TestSurvivalBoostRounds:
    """Each round's trees are rebuilt from the rows and gradients they were grown on."""

    @staticmethod
    def _capture(monkeypatch: pytest.MonkeyPatch) -> list[tuple[np.ndarray, np.ndarray]]:
        from greenwood import _survival_boost

        calls: list[tuple[np.ndarray, np.ndarray]] = []
        original = _survival_boost._RegressionTree.fit

        def fit(
            self: _survival_boost._RegressionTree, x: np.ndarray, y: np.ndarray, imp: np.ndarray
        ) -> _survival_boost._RegressionTree:
            calls.append((x.copy(), y.copy()))
            return original(self, x, y, imp)

        monkeypatch.setattr(_survival_boost._RegressionTree, "fit", fit)
        return calls

    def test_gradient_is_the_current_model_at_each_sampled_row(
        self, monkeypatch: pytest.MonkeyPatch, competing_data: gw.SimulatedCompetingRisks
    ) -> None:
        from greenwood._competing import _censoring_km
        from greenwood._survival_boost import _ensemble_scores, _ipcw_targets, _softmax

        calls = self._capture(monkeypatch)
        sb = gw.SurvivalBoost(
            n_estimators=4, learning_rate=0.3, max_depth=2, min_samples_leaf=5, random_state=1
        ).fit(competing_data.surv, competing_data.covariates)
        x = np.asarray(competing_data.covariates, dtype=float)
        time = np.asarray(competing_data.surv.stop, dtype=float)
        status = np.asarray(competing_data.surv.status, dtype=int)
        g_times, g_surv = _censoring_km(time, status)
        n_classes = sb.n_classes_
        assert len(calls) == 4 * n_classes

        for r in range(4):
            xa, _ = calls[r * n_classes]
            # Every row is a subject at one horizon. The covariates identify the subject.
            subject = np.array(
                [int(np.flatnonzero((x == row).all(axis=1))[0]) for row in xa[:, 1:]]
            )
            targets, weights = _ipcw_targets(
                status[subject], time[subject], xa[:, 0], g_times, g_surv, n_classes, 0.05
            )
            probs = _softmax(_ensemble_scores(sb.trees_[:r], xa, sb.learning_rate, n_classes))
            for k in range(n_classes):
                xk, yk = calls[r * n_classes + k]
                np.testing.assert_array_equal(xk, xa)
                expected = weights * ((targets == k).astype(float) - probs[:, k])
                np.testing.assert_allclose(yk, expected, rtol=1e-12, atol=1e-12)

    @pytest.mark.parametrize(("fraction", "n_zero"), [(0.0, 0), (0.1, 30), (0.05, 15)])
    def test_hard_zero_fraction_rounds_down(
        self,
        monkeypatch: pytest.MonkeyPatch,
        competing_data: gw.SimulatedCompetingRisks,
        fraction: float,
        n_zero: int,
    ) -> None:
        calls = self._capture(monkeypatch)
        gw.SurvivalBoost(
            n_estimators=2,
            n_horizons_per_observation=2,
            hard_zero_fraction=fraction,
            random_state=0,
        ).fit(competing_data.surv, competing_data.covariates)
        # A horizon of 0 is before every follow-up time, so its row is never left out.
        for xa, _ in calls:
            assert int((xa[:, 0] == 0.0).sum()) == 2 * n_zero

    def test_time_grid_has_distinct_times(self) -> None:
        rng = np.random.default_rng(3)
        n = 200
        time = rng.integers(1, 6, size=n).astype(float)
        cause = rng.choice(["censor", "a", "b"], size=n)
        y = Surv(time=time, event=pd.Categorical(cause, categories=["censor", "a", "b"]))
        x = rng.standard_normal((n, 2))

        quantiles = gw.SurvivalBoost(n_estimators=1, n_time_grid_steps=50, random_state=0)
        quantiles.fit(y, x)
        assert np.all(np.diff(quantiles.time_grid_) > 0)
        assert quantiles.time_grid_[0] == time[cause != "censor"].min()

        event_times = gw.SurvivalBoost(n_estimators=1, n_time_grid_steps=1000, random_state=0)
        event_times.fit(y, x)
        np.testing.assert_array_equal(event_times.time_grid_, np.unique(time[cause != "censor"]))
