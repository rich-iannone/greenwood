"""Unit tests for prediction-performance metrics."""

from __future__ import annotations

import numpy as np
import pytest

import greenwood as gw
from greenwood import Surv


def test_concordance_perfect_ordering() -> None:
    # Larger risk fails sooner -> perfect concordance.
    time = [1, 2, 3, 4]
    event = [1, 1, 1, 1]
    risk = [4.0, 3.0, 2.0, 1.0]

    assert gw.concordance_index(Surv.right(time, event), risk) == pytest.approx(1.0)


def test_concordance_reversed_ordering() -> None:
    time = [1, 2, 3, 4]
    event = [1, 1, 1, 1]
    risk = [1.0, 2.0, 3.0, 4.0]  # larger risk fails later -> fully discordant

    assert gw.concordance_index(Surv.right(time, event), risk) == pytest.approx(0.0)


def test_concordance_length_checked() -> None:
    with pytest.raises(ValueError, match="same length"):
        gw.concordance_index(Surv.right([1, 2, 3], [1, 1, 1]), [0.1, 0.2])


def test_brier_shape_checked() -> None:
    y = Surv.right([1, 2, 3], [1, 0, 1])
    with pytest.raises(ValueError, match="shape"):
        gw.brier_score(y, np.zeros((3, 1)), times=[1.0, 2.0])


def test_brier_perfect_prediction_is_zero() -> None:
    # No censoring; predicting survival 1 before the event and 0 after is perfect.
    time = [1.0, 2.0, 3.0]
    event = [1, 1, 1]
    y = Surv.right(time, event)
    times = np.array([1.5, 2.5])

    # subject i alive at t iff time_i > t.
    probs = np.array([[float(ti > t) for t in times] for ti in time])

    np.testing.assert_allclose(gw.brier_score(y, probs, times), [0.0, 0.0], atol=1e-12)


def test_brier_in_unit_range() -> None:
    df = gw.load_dataset("lung", backend="pandas")
    y = Surv.right(df["time"], event=(df["status"] == 2))
    cox = gw.CoxPH().fit(y, df[["age", "sex"]])
    times = np.array([180.0, 365.0])
    pred = cox.predict(df[["age", "sex"]], type="survival", times=times, format="pandas")
    probs = pred[[f"subject_{i + 1}" for i in range(len(df))]].to_numpy().T
    bs = gw.brier_score(y, probs, times)

    assert np.all((bs >= 0) & (bs <= 1))


def test_integrated_brier_needs_two_times() -> None:
    y = Surv.right([1, 2, 3], [1, 0, 1])
    with pytest.raises(ValueError, match="at least two times"):
        gw.integrated_brier_score(y, np.zeros((3, 1)), times=[1.0])


def test_integrated_brier_between_pointwise() -> None:
    df = gw.load_dataset("lung", backend="pandas")
    y = Surv.right(df["time"], event=(df["status"] == 2))
    cox = gw.CoxPH().fit(y, df[["age", "sex"]])
    times = np.array([180.0, 365.0, 540.0])
    pred = cox.predict(df[["age", "sex"]], type="survival", times=times, format="pandas")
    probs = pred[[f"subject_{i + 1}" for i in range(len(df))]].to_numpy().T
    bs = gw.brier_score(y, probs, times)
    ibs = gw.integrated_brier_score(y, probs, times)

    assert bs.min() <= ibs <= bs.max()


def test_calibration_structure_and_coverage() -> None:
    df = gw.load_dataset("lung", backend="pandas")
    y = Surv.right(df["time"], event=(df["status"] == 2))
    cox = gw.CoxPH().fit(y, df[["age", "sex"]])
    pred = (
        cox.predict(df[["age", "sex"]], type="survival", times=[365.0], format="pandas")
        .iloc[0, 1:]
        .to_numpy()
    )
    cal = gw.calibration(y, pred, 365.0, n_bins=5, format="pandas")

    assert list(cal.columns) == [
        "bin",
        "n",
        "predicted",
        "observed",
        "observed_lower",
        "observed_upper",
    ]
    assert cal["n"].sum() == len(df)  # bins partition the subjects
    assert list(cal["predicted"]) == sorted(cal["predicted"])  # bins ordered by prediction
    assert ((cal["observed"] >= 0) & (cal["observed"] <= 1)).all()


def test_calibration_single_bin_is_overall_km() -> None:
    # A constant prediction collapses to one bin; the observed is the overall KM at the time.
    df = gw.load_dataset("lung", backend="pandas")
    y = Surv.right(df["time"], event=(df["status"] == 2))
    cal = gw.calibration(y, np.full(len(df), 0.5), 365.0, n_bins=3, format="pandas")

    assert len(cal) == 1

    km_at = float(gw.KaplanMeier().fit(y).predict([365.0])[0])

    np.testing.assert_allclose(cal["observed"].iloc[0], km_at)


def test_calibration_diagonal_on_well_specified_model() -> None:
    # Simulate from an exponential Cox model; predicted survival should track observed.
    rng = np.random.default_rng(0)
    n = 4000
    x = rng.normal(size=n)
    baseline = 0.02
    event_time = rng.exponential(1.0 / (baseline * np.exp(0.7 * x)))
    censor_time = rng.exponential(1.0 / 0.01, size=n)
    time = np.minimum(event_time, censor_time)
    event = (event_time <= censor_time).astype(int)
    y = Surv.right(time, event=event)
    cox = gw.CoxPH().fit(y, x.reshape(-1, 1))
    horizon = float(np.quantile(time, 0.4))
    pred = (
        cox.predict(x.reshape(-1, 1), type="survival", times=[horizon], format="pandas")
        .iloc[0, 1:]
        .to_numpy()
    )
    cal = gw.calibration(y, pred, horizon, n_bins=10, format="pandas")

    # A correctly specified model is close to the diagonal on average.
    assert np.mean(np.abs(cal["predicted"] - cal["observed"])) < 0.05


def test_calibration_input_validation() -> None:
    y = Surv.right([1, 2, 3, 4], [1, 1, 1, 1])
    with pytest.raises(ValueError, match="one value per subject"):
        gw.calibration(y, [0.5, 0.5], 2.0)
    with pytest.raises(ValueError, match="n_bins"):
        gw.calibration(y, [0.1, 0.2, 0.3, 0.4], 2.0, n_bins=1)


# -- IPCW concordance -----------------------------------------------------------


def test_ipcw_concordance_perfect_ordering() -> None:
    time = [1, 2, 3, 4]
    event = [1, 1, 1, 1]
    risk = [4.0, 3.0, 2.0, 1.0]

    assert gw.concordance_index_ipcw(Surv.right(time, event), risk) == pytest.approx(1.0)


def test_ipcw_concordance_reversed_ordering() -> None:
    time = [1, 2, 3, 4]
    event = [1, 1, 1, 1]
    risk = [1.0, 2.0, 3.0, 4.0]

    assert gw.concordance_index_ipcw(Surv.right(time, event), risk) == pytest.approx(0.0)


def test_ipcw_concordance_no_censoring_equals_harrell() -> None:
    rng = np.random.default_rng(23)
    time = rng.exponential(1.0, size=50)
    event = np.ones(50, dtype=int)
    risk = rng.normal(size=50)
    y = Surv.right(time, event)
    c_harrell = gw.concordance_index(y, risk)
    c_ipcw = gw.concordance_index_ipcw(y, risk)

    assert c_ipcw == pytest.approx(c_harrell)


def test_ipcw_concordance_length_checked() -> None:
    with pytest.raises(ValueError, match="same length"):
        gw.concordance_index_ipcw(Surv.right([1, 2, 3], [1, 1, 1]), [0.1, 0.2])


def test_ipcw_concordance_no_events_raises() -> None:
    with pytest.raises(ValueError, match="No events"):
        gw.concordance_index_ipcw(Surv.right([1, 2, 3], [0, 0, 0]), [0.1, 0.2, 0.3])


def test_ipcw_concordance_no_events_before_tau_raises() -> None:
    with pytest.raises(ValueError, match="No events before tau"):
        gw.concordance_index_ipcw(Surv.right([10, 20, 30], [1, 1, 1]), [0.1, 0.2, 0.3], tau=5.0)


def test_ipcw_concordance_tau_restricts_events() -> None:
    df = gw.load_dataset("lung", backend="pandas")
    y = Surv.right(df["time"], event=(df["status"] == 2))
    cox = gw.CoxPH().fit(y, df[["age", "sex"]])
    lp = cox.predict(type="lp")
    c_full = gw.concordance_index_ipcw(y, lp)
    c_tau = gw.concordance_index_ipcw(y, lp, tau=365.0)

    assert 0 < c_tau < 1
    assert c_full != pytest.approx(c_tau, abs=1e-6)


def test_ipcw_concordance_in_unit_range() -> None:
    df = gw.load_dataset("lung", backend="pandas")
    y = Surv.right(df["time"], event=(df["status"] == 2))
    cox = gw.CoxPH().fit(y, df[["age", "sex"]])
    lp = cox.predict(type="lp")
    c = gw.concordance_index_ipcw(y, lp)

    assert 0 <= c <= 1


# -- time_dependent_auc --------------------------------------------------------


class TestTimeDependentAUC:
    @pytest.fixture(scope="class")
    def lung_cox(self):  # type: ignore[no-untyped-def]
        df = gw.load_dataset("lung", backend="pandas")
        y = Surv.right(df["time"], event=(df["status"] == 2))
        cox = gw.CoxPH().fit(y, df[["age", "sex"]])
        lp = cox.predict(type="lp")
        return y, lp

    def test_returns_array_of_correct_shape(self, lung_cox) -> None:
        y, lp = lung_cox
        auc = gw.time_dependent_auc(y, lp, times=[180, 365, 540])

        assert isinstance(auc, np.ndarray)
        assert auc.shape == (3,)

    def test_values_in_unit_interval(self, lung_cox) -> None:
        y, lp = lung_cox
        auc = gw.time_dependent_auc(y, lp, times=[180, 365, 540])

        assert np.all((auc >= 0) & (auc <= 1))

    def test_perfect_marker_gives_one(self) -> None:
        time = np.array([1.0, 2.0, 3.0, 4.0])
        event = np.array([1, 1, 1, 1])
        risk = np.array([4.0, 3.0, 2.0, 1.0])
        y = Surv.right(time, event)
        auc = gw.time_dependent_auc(y, risk, times=[1.5, 2.5, 3.5])

        np.testing.assert_allclose(auc, [1.0, 1.0, 1.0])

    def test_reversed_marker_gives_zero(self) -> None:
        time = np.array([1.0, 2.0, 3.0, 4.0])
        event = np.array([1, 1, 1, 1])
        risk = np.array([1.0, 2.0, 3.0, 4.0])
        y = Surv.right(time, event)
        auc = gw.time_dependent_auc(y, risk, times=[1.5, 2.5, 3.5])

        np.testing.assert_allclose(auc, [0.0, 0.0, 0.0])

    def test_random_marker_near_half(self) -> None:
        rng = np.random.default_rng(42)
        n = 500
        time = rng.exponential(1.0, size=n)
        event = np.ones(n, dtype=int)
        risk = rng.normal(size=n)
        y = Surv.right(time, event)
        med = float(np.median(time))
        auc = gw.time_dependent_auc(y, risk, times=[med])

        assert abs(auc[0] - 0.5) < 0.1

    def test_time_before_first_event_is_nan(self) -> None:
        time = np.array([5.0, 6.0, 7.0])
        event = np.array([1, 1, 1])
        y = Surv.right(time, event)
        auc = gw.time_dependent_auc(y, [3.0, 2.0, 1.0], times=[1.0])

        assert np.isnan(auc[0])

    def test_time_after_all_observations_is_nan(self) -> None:
        time = np.array([1.0, 2.0, 3.0])
        event = np.array([1, 1, 1])
        y = Surv.right(time, event)
        auc = gw.time_dependent_auc(y, [3.0, 2.0, 1.0], times=[10.0])

        assert np.isnan(auc[0])

    def test_marker_length_mismatch_raises(self) -> None:
        y = Surv.right([1, 2, 3], [1, 1, 1])
        with pytest.raises(ValueError, match="same length"):
            gw.time_dependent_auc(y, [1.0, 2.0], times=[1.5])

    def test_model_beats_random(self, lung_cox) -> None:
        y, lp = lung_cox
        rng = np.random.default_rng(99)
        auc_model = gw.time_dependent_auc(y, lp, times=[365])
        auc_random = gw.time_dependent_auc(y, rng.normal(size=len(lp)), times=[365])

        assert auc_model[0] > auc_random[0]

    def test_no_censoring_matches_empirical_auc(self) -> None:
        rng = np.random.default_rng(0)
        n = 200
        time = rng.exponential(1.0, size=n)
        event = np.ones(n, dtype=int)
        risk = -time + rng.normal(0, 0.1, size=n)
        y = Surv.right(time, event)
        med = float(np.median(time))
        auc = gw.time_dependent_auc(y, risk, times=[med])

        assert auc[0] > 0.8


# -- integrated_auc ------------------------------------------------------------


class TestIntegratedAUC:
    @pytest.fixture(scope="class")
    def lung_cox(self):  # type: ignore[no-untyped-def]
        df = gw.load_dataset("lung", backend="pandas")
        y = Surv.right(df["time"], event=(df["status"] == 2))
        cox = gw.CoxPH().fit(y, df[["age", "sex"]])
        lp = cox.predict(type="lp")
        return y, lp

    def test_returns_scalar(self, lung_cox) -> None:
        y, lp = lung_cox
        iauc = gw.integrated_auc(y, lp, times=[180, 365, 540])

        assert isinstance(iauc, float)

    def test_in_unit_interval(self, lung_cox) -> None:
        y, lp = lung_cox
        iauc = gw.integrated_auc(y, lp, times=[180, 365, 540])

        assert 0 <= iauc <= 1

    def test_between_pointwise_extremes(self, lung_cox) -> None:
        y, lp = lung_cox
        times = [180, 365, 540]
        auc = gw.time_dependent_auc(y, lp, times=times)
        iauc = gw.integrated_auc(y, lp, times=times)

        assert auc.min() <= iauc + 1e-12
        assert iauc <= auc.max() + 1e-12

    def test_needs_at_least_two_times(self, lung_cox) -> None:
        y, lp = lung_cox
        with pytest.raises(ValueError, match="at least two times"):
            gw.integrated_auc(y, lp, times=[180])

    def test_perfect_marker_gives_one(self) -> None:
        time = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        event = np.array([1, 1, 1, 1, 1])
        risk = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
        y = Surv.right(time, event)
        iauc = gw.integrated_auc(y, risk, times=[1.5, 2.5, 3.5, 4.5])

        assert iauc == pytest.approx(1.0)

    def test_model_beats_chance(self, lung_cox) -> None:
        y, lp = lung_cox
        iauc = gw.integrated_auc(y, lp, times=[180, 365, 540])

        assert iauc > 0.5
