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
    pred = cox.predict(df[["age", "sex"]], type="survival", times=times)
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
    pred = cox.predict(df[["age", "sex"]], type="survival", times=times)
    probs = pred[[f"subject_{i + 1}" for i in range(len(df))]].to_numpy().T
    bs = gw.brier_score(y, probs, times)
    ibs = gw.integrated_brier_score(y, probs, times)
    assert bs.min() <= ibs <= bs.max()


def test_calibration_structure_and_coverage() -> None:
    df = gw.load_dataset("lung", backend="pandas")
    y = Surv.right(df["time"], event=(df["status"] == 2))
    cox = gw.CoxPH().fit(y, df[["age", "sex"]])
    pred = cox.predict(df[["age", "sex"]], type="survival", times=[365.0]).iloc[0, 1:].to_numpy()
    cal = gw.calibration(y, pred, 365.0, n_bins=5)
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
    cal = gw.calibration(y, np.full(len(df), 0.5), 365.0, n_bins=3)
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
    pred = cox.predict(x.reshape(-1, 1), type="survival", times=[horizon]).iloc[0, 1:].to_numpy()
    cal = gw.calibration(y, pred, horizon, n_bins=10)
    # A correctly specified model is close to the diagonal on average.
    assert np.mean(np.abs(cal["predicted"] - cal["observed"])) < 0.05


def test_calibration_input_validation() -> None:
    y = Surv.right([1, 2, 3, 4], [1, 1, 1, 1])
    with pytest.raises(ValueError, match="one value per subject"):
        gw.calibration(y, [0.5, 0.5], 2.0)
    with pytest.raises(ValueError, match="n_bins"):
        gw.calibration(y, [0.1, 0.2, 0.3, 0.4], 2.0, n_bins=1)
