"""Tests for the risk-set / event-table kernel."""

from __future__ import annotations

import numpy as np
import pytest

from greenwood import Surv, event_table


def test_event_table_simple_counts() -> None:
    # times 4,5,6 with an event at each; no censoring, no truncation.
    y = Surv.right([4, 5, 6], [1, 1, 1])
    et = event_table(y)
    np.testing.assert_array_equal(et.time, [4, 5, 6])
    np.testing.assert_array_equal(et.n_risk, [3, 2, 1])
    np.testing.assert_array_equal(et.n_event, [1, 1, 1])
    np.testing.assert_array_equal(et.n_censor, [0, 0, 0])


def test_event_table_with_censoring() -> None:
    y = Surv.right([4, 4, 6], [1, 0, 1])  # a tie: one event, one censor at t=4
    et = event_table(y)
    np.testing.assert_array_equal(et.time, [4, 6])
    np.testing.assert_array_equal(et.n_risk, [3, 1])
    np.testing.assert_array_equal(et.n_event, [1, 1])
    np.testing.assert_array_equal(et.n_censor, [1, 0])


def test_event_table_weights() -> None:
    y = Surv.right([4, 5], [1, 1], weights=[2.0, 3.0])
    et = event_table(y)
    np.testing.assert_allclose(et.n_risk, [5.0, 3.0])
    np.testing.assert_allclose(et.n_event, [2.0, 3.0])


def test_event_table_grouped_strata_order_is_first_appearance() -> None:
    y = Surv.right([5, 4, 6, 4], [1, 1, 1, 1])
    et = event_table(y, group=["b", "a", "b", "a"])
    assert et.strata is not None
    # "b" appears first, so its rows come first.
    assert list(dict.fromkeys(et.strata.tolist())) == ["b", "a"]


def test_event_table_group_length_checked() -> None:
    y = Surv.right([5, 4, 6], [1, 1, 1])
    with pytest.raises(ValueError, match="same length"):
        event_table(y, group=["a", "b"])


def test_event_table_interval_not_supported() -> None:
    y = Surv.interval(lower=[1, 2], upper=[3, 4])
    with pytest.raises(NotImplementedError, match="counting-process"):
        event_table(y)


def test_event_table_to_pandas() -> None:
    y = Surv.right([4, 5], [1, 1])
    df = event_table(y).to_pandas()
    assert list(df.columns) == ["time", "n_risk", "n_event", "n_censor"]
