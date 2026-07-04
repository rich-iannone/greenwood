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
    df = gw.data.load_dataset("lung", backend="pandas")
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
    df = gw.data.load_dataset("lung", backend="pandas")
    y = Surv.right(df["time"], event=(df["status"] == 2))
    cox = gw.CoxPH().fit(y, df[["age", "sex"]])
    times = np.array([180.0, 365.0, 540.0])
    pred = cox.predict(df[["age", "sex"]], type="survival", times=times)
    probs = pred[[f"subject_{i + 1}" for i in range(len(df))]].to_numpy().T
    bs = gw.brier_score(y, probs, times)
    ibs = gw.integrated_brier_score(y, probs, times)
    assert bs.min() <= ibs <= bs.max()
