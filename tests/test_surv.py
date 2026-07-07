"""Tests for the `Surv` response object: construction, validation, round-trip."""

from __future__ import annotations

import numpy as np
import pytest

from greenwood import CensoringType, Surv


def test_right_basic() -> None:
    y = Surv.right([5, 6, 4], [1, 0, 1])
    assert y.type is CensoringType.RIGHT
    assert y.n == 3
    assert y.n_events == 2
    assert y.n_censored == 1
    assert not y.is_truncated


def test_right_event_defaults_to_all_events() -> None:
    y = Surv.right([5, 6, 4])
    assert y.n_events == 3


def test_right_accepts_boolean_event() -> None:
    y = Surv.right([5, 6, 4], [True, False, True])
    assert y.n_events == 2


def test_event_rejects_r_1_2_coding() -> None:
    # survival::lung uses 1 = censored, 2 = dead; must be converted explicitly.
    with pytest.raises(ValueError, match="1/2 coding"):
        Surv.right([5, 6, 4], [1, 2, 2])


def test_counting_left_truncation() -> None:
    y = Surv.counting(start=[0, 2, 1], stop=[5, 6, 4], event=[1, 0, 1])
    assert y.type is CensoringType.COUNTING
    assert y.is_truncated
    np.testing.assert_array_equal(y.entry, [0.0, 2.0, 1.0])


def test_counting_requires_start_lt_stop() -> None:
    with pytest.raises(ValueError, match="strictly less"):
        Surv.counting(start=[0, 5], stop=[5, 5], event=[1, 1])


def test_right_entry_is_neg_inf() -> None:
    y = Surv.right([5, 6], [1, 1])
    assert np.all(np.isneginf(y.entry))


def test_negative_time_rejected() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        Surv.right([-1, 5], [1, 1])


def test_length_mismatch_rejected() -> None:
    with pytest.raises(ValueError, match="same length"):
        Surv.right([5, 6, 4], [1, 0])


def test_weights_must_be_positive() -> None:
    with pytest.raises(ValueError, match="strictly positive"):
        Surv.right([5, 6], [1, 1], weights=[1.0, 0.0])


def test_interval_encodes_censoring() -> None:
    y = Surv.interval(lower=[1, 2, 3], upper=[1, np.inf, 5])
    # exact (1==1) -> event, inf upper -> right-censored, (3,5] -> interval
    np.testing.assert_array_equal(y.status, [1, 0, 2])


def test_multistate_states_and_codes() -> None:
    y = Surv.multistate([5, 6, 7], event=[0, 1, 2], states=("relapse", "death"))
    assert y.is_multistate
    assert y.states == ("relapse", "death")
    assert y.n_events == 2  # codes 1 and 2 are events; 0 is censored


def test_multistate_rejects_code_above_states() -> None:
    with pytest.raises(ValueError, match="event states"):
        Surv.multistate([5, 6], event=[1, 3], states=("relapse", "death"))


def test_repr_is_informative() -> None:
    y = Surv.counting(start=[0, 1], stop=[5, 6], event=[1, 0])
    assert "truncated" in repr(y)
    assert "n=2" in repr(y)


def test_roundtrip_json_right() -> None:
    y = Surv.right([5, 6, 4], [1, 0, 1], weights=[1.0, 2.0, 1.5])
    restored = Surv.from_json(y.to_json())
    assert restored.to_json() == y.to_json()
    np.testing.assert_array_equal(restored.stop, y.stop)


def test_roundtrip_json_counting() -> None:
    y = Surv.counting(start=[0, 1], stop=[5, 6], event=[1, 0])
    restored = Surv.from_json(y.to_json())
    assert restored.type is CensoringType.COUNTING
    np.testing.assert_array_equal(restored.start, y.start)


def test_accepts_pandas_series() -> None:
    pd = pytest.importorskip("pandas")
    df = pd.DataFrame({"t": [5, 6, 4], "e": [1, 0, 1]})
    y = Surv.right(df["t"], df["e"])
    assert y.n == 3


def test_accepts_polars_series() -> None:
    pl = pytest.importorskip("polars")
    df = pl.DataFrame({"t": [5, 6, 4], "e": [1, 0, 1]})
    y = Surv.right(df["t"], df["e"])
    assert y.n == 3


def test_to_pandas() -> None:
    y = Surv.counting(start=[0, 1], stop=[5, 6], event=[1, 0])
    df = y.to_frame(format="pandas")
    assert list(df.columns) == ["start", "stop", "status"]


def test_to_polars() -> None:
    y = Surv.counting(start=[0, 1], stop=[5, 6], event=[1, 0])
    df = y.to_frame(format="polars")
    assert set(df.columns) == {"start", "stop", "status"}


def test_to_arrow() -> None:
    y = Surv.counting(start=[0, 1], stop=[5, 6], event=[1, 0])
    table = y.to_frame(format="pyarrow")
    assert set(table.column_names) == {"start", "stop", "status"}
