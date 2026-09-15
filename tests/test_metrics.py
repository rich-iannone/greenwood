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
        rng = np.random.default_rng(23)
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


# ---------------------------------------------------------------------------
# Cause-specific (incidence) Brier score
# ---------------------------------------------------------------------------


@pytest.fixture()
def competing_data():
    """Competing-risks dataset: 2 causes, some censoring."""
    rng = np.random.default_rng(23)
    n = 200
    time = rng.exponential(scale=10, size=n)
    status = rng.choice([0, 1, 2], size=n, p=[0.3, 0.4, 0.3])
    y = Surv.multistate(time, status, states=("cause1", "cause2"))
    return y


class TestBrierScoreIncidence:
    def test_shape_checked(self, competing_data) -> None:
        y = competing_data
        with pytest.raises(ValueError, match="shape"):
            gw.brier_score_incidence(y, np.zeros((y.n, 1)), times=[1.0, 2.0], cause=1)

    def test_invalid_cause_raises(self, competing_data) -> None:
        y = competing_data
        times = [5.0, 10.0]
        probs = np.full((y.n, 2), 0.3)
        with pytest.raises(ValueError, match="cause=99"):
            gw.brier_score_incidence(y, probs, times, cause=99)

    def test_perfect_prediction_is_zero(self) -> None:
        time = np.array([1.0, 2.0, 3.0, 4.0])
        status = np.array([1, 2, 1, 2])
        y = Surv.multistate(time, status, states=("a", "b"))
        t_eval = np.array([1.5, 2.5, 3.5])

        # Perfect CIF for cause 1: indicator that cause 1 happened before t
        probs = np.array(
            [[float((status[i] == 1) and (time[i] <= t)) for t in t_eval] for i in range(len(time))]
        )

        bs = gw.brier_score_incidence(y, probs, t_eval, cause=1)
        np.testing.assert_allclose(bs, 0.0, atol=1e-12)

    def test_scores_are_nonnegative(self, competing_data) -> None:
        y = competing_data
        rng = np.random.default_rng(123)
        times = [3.0, 6.0, 10.0, 15.0]
        probs = rng.uniform(0, 0.5, size=(y.n, len(times)))

        bs = gw.brier_score_incidence(y, probs, times, cause=1)
        assert np.all(bs >= 0.0)

    def test_scores_bounded_by_one(self, competing_data) -> None:
        y = competing_data
        times = [3.0, 6.0, 10.0]
        probs = np.full((y.n, 3), 0.5)

        bs = gw.brier_score_incidence(y, probs, times, cause=1)
        assert np.all(bs <= 1.0)

    def test_wrong_cause_gives_higher_score(self) -> None:
        rng = np.random.default_rng(7)
        n = 300
        time = rng.exponential(scale=10, size=n)
        status = rng.choice([0, 1, 2], size=n, p=[0.2, 0.5, 0.3])
        y = Surv.multistate(time, status, states=("a", "b"))
        times = np.array([5.0, 10.0, 15.0])

        # Good predictions: approximately correct CIF for cause 1
        from greenwood._nonparametric import KaplanMeier

        km_all = KaplanMeier().fit(Surv.right(time, (status > 0).astype(int)))
        cause1_frac = (status == 1).sum() / (status > 0).sum()
        good_probs = np.array(
            [
                [
                    cause1_frac * (1.0 - float(np.interp(t, km_all.time_, km_all.survival_)))
                    for t in times
                ]
                for _ in range(n)
            ]
        )

        # Bad predictions: constant 0.9 for everyone
        bad_probs = np.full((n, len(times)), 0.9)

        bs_good = gw.brier_score_incidence(y, good_probs, times, cause=1)
        bs_bad = gw.brier_score_incidence(y, bad_probs, times, cause=1)
        assert np.all(bs_good < bs_bad)

    def test_reduces_to_survival_brier_single_cause(self) -> None:
        rng = np.random.default_rng(55)
        n = 150
        time = rng.exponential(scale=8, size=n)
        event_binary = rng.choice([0, 1], size=n, p=[0.3, 0.7])
        y_right = Surv.right(time, event_binary)
        y_multi = Surv.multistate(time, event_binary, states=("death",))

        times = np.array([3.0, 6.0, 10.0])
        surv_probs = np.column_stack([rng.uniform(0.3, 0.9, size=n) for _ in times])
        incidence_probs = 1.0 - surv_probs

        bs_surv = gw.brier_score(y_right, surv_probs, times)
        bs_inc = gw.brier_score_incidence(y_multi, incidence_probs, times, cause=1)
        np.testing.assert_allclose(bs_inc, bs_surv, atol=1e-10)

    def test_each_cause_independently_scored(self, competing_data) -> None:
        y = competing_data
        rng = np.random.default_rng(99)
        times = [5.0, 10.0]
        probs = rng.uniform(0, 0.5, size=(y.n, 2))

        bs1 = gw.brier_score_incidence(y, probs, times, cause=1)
        bs2 = gw.brier_score_incidence(y, probs, times, cause=2)
        # Same predictions but different causes should yield different scores
        assert not np.allclose(bs1, bs2)


class TestIntegratedBrierScoreIncidence:
    def test_needs_at_least_two_times(self, competing_data) -> None:
        y = competing_data
        probs = np.full((y.n, 1), 0.3)
        with pytest.raises(ValueError, match="at least two times"):
            gw.integrated_brier_score_incidence(y, probs, times=[5.0], cause=1)

    def test_nonnegative(self, competing_data) -> None:
        y = competing_data
        rng = np.random.default_rng(77)
        times = [3.0, 6.0, 10.0, 15.0]
        probs = rng.uniform(0, 0.5, size=(y.n, len(times)))

        ibs = gw.integrated_brier_score_incidence(y, probs, times, cause=1)
        assert ibs >= 0.0

    def test_consistent_with_pointwise(self, competing_data) -> None:
        y = competing_data
        rng = np.random.default_rng(88)
        times = np.array([3.0, 6.0, 10.0, 15.0])
        probs = rng.uniform(0, 0.5, size=(y.n, len(times)))

        ibs = gw.integrated_brier_score_incidence(y, probs, times, cause=1)
        bs = gw.brier_score_incidence(y, probs, times, cause=1)
        manual_ibs = float(np.trapezoid(bs, times)) / (times[-1] - times[0])
        assert ibs == pytest.approx(manual_ibs, abs=1e-12)

    def test_better_model_lower_ibs(self) -> None:
        rng = np.random.default_rng(33)
        n = 250
        time = rng.exponential(scale=10, size=n)
        status = rng.choice([0, 1, 2], size=n, p=[0.2, 0.5, 0.3])
        y = Surv.multistate(time, status, states=("a", "b"))
        times = np.linspace(2.0, 20.0, 10)

        good_probs = np.column_stack(
            [
                np.clip(
                    ((status == 1) & (time <= t)).astype(float) + rng.normal(0, 0.05, size=n),
                    0.0,
                    1.0,
                )
                for t in times
            ]
        )
        bad_probs = np.full((n, len(times)), 0.8)

        ibs_good = gw.integrated_brier_score_incidence(y, good_probs, times, cause=1)
        ibs_bad = gw.integrated_brier_score_incidence(y, bad_probs, times, cause=1)
        assert ibs_good < ibs_bad


# ---------------------------------------------------------------------------
# Concordance index for incidence
# ---------------------------------------------------------------------------


class TestConcordanceIndexIncidence:
    def test_length_checked(self, competing_data) -> None:
        y = competing_data
        with pytest.raises(ValueError, match="same length"):
            gw.concordance_index_incidence(y, np.zeros(5), cause=1)

    def test_invalid_cause_raises(self, competing_data) -> None:
        y = competing_data
        with pytest.raises(ValueError, match="cause=99"):
            gw.concordance_index_incidence(y, np.zeros(y.n), cause=99)

    def test_no_events_before_tau_raises(self) -> None:
        time = np.array([10.0, 20.0, 30.0])
        status = np.array([1, 2, 1])
        y = Surv.multistate(time, status, states=("a", "b"))
        with pytest.raises(ValueError, match="No events of cause 1 before tau"):
            gw.concordance_index_incidence(y, np.zeros(3), cause=1, tau=5.0)

    def test_in_unit_range(self, competing_data) -> None:
        y = competing_data
        rng = np.random.default_rng(23)
        pred = rng.uniform(0, 1, size=y.n)
        c = gw.concordance_index_incidence(y, pred, cause=1)
        assert 0.0 <= c <= 1.0

    def test_perfect_discrimination(self) -> None:
        time = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        status = np.array([1, 1, 2, 2, 0, 0])
        y = Surv.multistate(time, status, states=("a", "b"))
        # Subjects with cause 1 (times 1, 2) get highest predicted CIF
        pred = np.array([0.9, 0.8, 0.3, 0.2, 0.1, 0.05])
        c = gw.concordance_index_incidence(y, pred, cause=1)
        assert c == pytest.approx(1.0)

    def test_constant_prediction_near_half(self, competing_data) -> None:
        y = competing_data
        pred = np.full(y.n, 0.5)
        c = gw.concordance_index_incidence(y, pred, cause=1)
        assert c == pytest.approx(0.5)

    def test_tau_restricts_events(self, competing_data) -> None:
        y = competing_data
        rng = np.random.default_rng(11)
        pred = rng.uniform(0, 1, size=y.n)
        c_full = gw.concordance_index_incidence(y, pred, cause=1)
        tau = float(np.median(y.stop))
        c_tau = gw.concordance_index_incidence(y, pred, cause=1, tau=tau)
        assert 0.0 <= c_tau <= 1.0
        assert c_full != pytest.approx(c_tau, abs=1e-6)


# ---------------------------------------------------------------------------
# Calibration for incidence (AJ calibration)
# ---------------------------------------------------------------------------


class TestCalibrationIncidence:
    def test_shape_checked(self, competing_data) -> None:
        y = competing_data
        with pytest.raises(ValueError, match="shape"):
            gw.calibration_incidence(y, np.zeros((y.n, 1)), times=[1.0, 2.0], cause=1)

    def test_invalid_cause_raises(self, competing_data) -> None:
        y = competing_data
        probs = np.full((y.n, 2), 0.3)
        with pytest.raises(ValueError, match="cause=99"):
            gw.calibration_incidence(y, probs, times=[5.0, 10.0], cause=99)

    def test_marginal_model_perfectly_calibrated(self, competing_data) -> None:
        y = competing_data
        aj = gw.AalenJohansen().fit(y)
        cif_df = aj.to_frame(format="polars")
        c1_cif = cif_df.filter(cif_df["cause"] == "cause1")

        times = np.array([3.0, 6.0, 10.0])
        marginal = np.array(
            [
                float(np.interp(t, c1_cif["time"].to_numpy(), c1_cif["estimate"].to_numpy()))
                for t in times
            ]
        )
        probs = np.tile(marginal, (y.n, 1))

        cal_err = gw.calibration_incidence(y, probs, times, cause=1)
        np.testing.assert_allclose(cal_err, 0.0, atol=1e-12)

    def test_errors_nonnegative(self, competing_data) -> None:
        y = competing_data
        rng = np.random.default_rng(55)
        times = [3.0, 6.0, 10.0]
        probs = rng.uniform(0, 0.5, size=(y.n, len(times)))
        cal_err = gw.calibration_incidence(y, probs, times, cause=1)
        assert np.all(cal_err >= 0.0)

    def test_biased_model_has_large_error(self, competing_data) -> None:
        y = competing_data
        times = np.array([3.0, 6.0, 10.0])
        # Predict 0.9 for everyone (highly biased)
        probs = np.full((y.n, len(times)), 0.9)
        cal_err = gw.calibration_incidence(y, probs, times, cause=1)
        assert np.all(cal_err > 0.1)

    def test_returns_correct_shape(self, competing_data) -> None:
        y = competing_data
        times = [3.0, 6.0, 10.0, 15.0]
        probs = np.full((y.n, len(times)), 0.3)
        cal_err = gw.calibration_incidence(y, probs, times, cause=1)
        assert cal_err.shape == (4,)


# ---------------------------------------------------------------------------
# Accuracy in time
# ---------------------------------------------------------------------------


class TestAccuracyInTime:
    def test_3d_required(self, competing_data) -> None:
        y = competing_data
        with pytest.raises(ValueError, match="3-D"):
            gw.accuracy_in_time(y, np.zeros((y.n, 2)), times=[5.0])

    def test_wrong_n_subjects_raises(self, competing_data) -> None:
        y = competing_data
        with pytest.raises(ValueError, match="subjects"):
            gw.accuracy_in_time(y, np.zeros((5, 2, 1)), times=[5.0])

    def test_wrong_n_causes_raises(self, competing_data) -> None:
        y = competing_data
        with pytest.raises(ValueError, match="causes"):
            gw.accuracy_in_time(y, np.zeros((y.n, 5, 1)), times=[5.0])

    def test_wrong_n_times_raises(self, competing_data) -> None:
        y = competing_data
        with pytest.raises(ValueError, match="times"):
            gw.accuracy_in_time(y, np.zeros((y.n, 2, 3)), times=[5.0])

    def test_in_unit_range(self, competing_data) -> None:
        y = competing_data
        rng = np.random.default_rng(23)
        times = [3.0, 6.0, 10.0]
        probs = rng.uniform(0, 0.3, size=(y.n, 2, len(times)))
        acc = gw.accuracy_in_time(y, probs, times)
        assert np.all((acc >= 0.0) & (acc <= 1.0))

    def test_perfect_prediction_is_one(self) -> None:
        time = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        status = np.array([1, 2, 1, 0, 2, 1])
        y = Surv.multistate(time, status, states=("a", "b"))
        t_eval = np.array([3.5])

        # At t=3.5: subject 0 (cause 1 at 1.0), 1 (cause 2 at 2.0), 2 (cause 1 at 3.0),
        # 3 (censored at 4.0 > 3.5, excluded=no, survived), 4 (cause 2 at 5.0 > 3.5, survived),
        # 5 (cause 1 at 6.0 > 3.5, survived)
        # Wait, subject 3 is censored at 4.0 > 3.5, so not censored before t. Included.
        # Observed classes at t=3.5: [1, 2, 1, 0, 0, 0]
        # Perfect predictions: CIF_1, CIF_2 that give correct argmax
        probs = np.zeros((6, 2, 1))
        # Subject 0: cause 1 -> CIF_1 high
        probs[0, 0, 0] = 0.8
        probs[0, 1, 0] = 0.1
        # Subject 1: cause 2 -> CIF_2 high
        probs[1, 0, 0] = 0.1
        probs[1, 1, 0] = 0.8
        # Subject 2: cause 1 -> CIF_1 high
        probs[2, 0, 0] = 0.8
        probs[2, 1, 0] = 0.1
        # Subject 3: survived -> survival = 1 - sum(CIFs) should be highest
        probs[3, 0, 0] = 0.1
        probs[3, 1, 0] = 0.1
        # Subject 4: survived
        probs[4, 0, 0] = 0.1
        probs[4, 1, 0] = 0.1
        # Subject 5: survived
        probs[5, 0, 0] = 0.1
        probs[5, 1, 0] = 0.1

        acc = gw.accuracy_in_time(y, probs, t_eval)

        assert acc[0] == pytest.approx(1.0)

    def test_returns_correct_shape(self, competing_data) -> None:
        y = competing_data
        times = [3.0, 6.0, 10.0, 15.0]
        probs = np.full((y.n, 2, len(times)), 0.1)
        acc = gw.accuracy_in_time(y, probs, times)

        assert acc.shape == (4,)

    def test_early_times_high_accuracy(self) -> None:
        rng = np.random.default_rng(77)
        n = 300
        time = rng.exponential(scale=50, size=n)
        status = rng.choice([0, 1, 2], size=n, p=[0.3, 0.4, 0.3])
        y = Surv.multistate(time, status, states=("a", "b"))

        # At very early times, most subjects survived; predicting low CIF = survival
        times = np.array([0.5, 1.0])
        probs = np.full((n, 2, len(times)), 0.01)
        acc = gw.accuracy_in_time(y, probs, times)

        # Most subjects survived at early times, so accuracy should be high
        assert np.all(acc > 0.5)
