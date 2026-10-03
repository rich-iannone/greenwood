"""Python-side behaviour of `Surv` that the R conformance fixtures don't cover.

`tests/test_surv_conformance.py` checks results, errors, and warnings against R's
`survival::Surv()`. This file covers inputs from Python backends, the vector protocol,
serialization, `first_event()`, and `as_surv()`.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd
import polars as pl
import pyarrow as pa
import pytest

import greenwood as gw
from greenwood import CensoringType, Surv

# -- inputs ---------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("time", "event"),
    [
        ([5, 6, 4], [1, 0, 1]),
        (np.array([5.0, 6.0, 4.0]), np.array([True, False, True])),
        (pd.Series([5, 6, 4]), pd.Series([2, 1, 2])),
        (pl.Series([5, 6, 4]), pl.Series([True, False, True])),
        (pa.chunked_array([[5, 6, 4]]), pa.chunked_array([[1, 0, 1]])),
    ],
    ids=["list", "numpy", "pandas", "polars", "pyarrow"],
)
def test_inputs_from_every_backend_agree(time: Any, event: Any) -> None:
    y = Surv(time=time, event=event)
    assert y.type == "right"
    np.testing.assert_array_equal(y.stop, [5.0, 6.0, 4.0])
    np.testing.assert_array_equal(y.status, [1.0, 0.0, 1.0])


def test_nulls_from_backends_are_missing() -> None:
    y = Surv(time=pl.Series([5, None]), event=pd.Series([1.0, None]))
    np.testing.assert_array_equal(y.is_na(), [False, True])
    assert np.isnan(y.status[1])


@pytest.mark.parametrize(
    "event",
    [
        pd.Categorical(["b", "a", "c"], categories=["a", "b", "c"]),
        pd.Series(pd.Categorical(["b", "a", "c"], categories=["a", "b", "c"])),
        pl.Series(["b", "a", "c"], dtype=pl.Enum(["a", "b", "c"])),
    ],
    ids=["pandas-categorical", "pandas-series", "polars-enum"],
)
def test_categorical_event_is_multistate(event: Any) -> None:
    y = Surv(time=[5, 6, 7], event=event)
    assert y.type is CensoringType.MRIGHT
    assert y.states == ("b", "c")
    np.testing.assert_array_equal(y.status, [1.0, 0.0, 2.0])


def test_categorical_event_with_start_is_multistate_counting() -> None:
    event = pd.Categorical(["b", "a"], categories=["a", "b"])
    y = Surv(time=[0, 1], time2=[5, 6], event=event)
    assert y.type is CensoringType.MCOUNTING
    assert y.is_truncated and y.is_multistate


def test_string_event_is_rejected_like_r() -> None:
    with pytest.raises(TypeError, match="Invalid status value, must be logical or numeric"):
        Surv(time=[5, 6], event=["a", "b"])


def test_unknown_type_is_rejected() -> None:
    with pytest.raises(ValueError, match="`type` must be one of"):
        Surv(time=[5], event=[1], type="rightish")


def test_censoring_type_compares_to_r_strings() -> None:
    assert CensoringType.RIGHT == "right"
    assert Surv(time=[1, 2], time2=[3, 4], event=[1, 0]).type == "counting"


# -- derived views --------------------------------------------------------------------------


def test_event_counts_ignore_missing_rows() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        y = Surv(time=[5, 6, 4, 3], event=[1, 0, 1, 7])
    assert (y.n, y.n_events, y.n_censored) == (4, 2, 1)
    np.testing.assert_array_equal(y.event, [True, False, True, False])


def test_entry_is_start_or_minus_infinity() -> None:
    np.testing.assert_array_equal(Surv(time=[5, 6], event=[1, 0]).entry, [-np.inf, -np.inf])
    np.testing.assert_array_equal(Surv(time=[0, 2], time2=[5, 6], event=[1, 0]).entry, [0, 2])


def test_is_na_checks_every_column() -> None:
    y = Surv(time=[1, np.nan, 3], time2=[2, 4, np.inf], type="interval2")
    np.testing.assert_array_equal(y.is_na(), [False, False, False])
    z = Surv(time=[np.nan, np.nan], time2=[np.nan, 2], type="interval2")
    np.testing.assert_array_equal(z.is_na(), [True, False])


# -- vector protocol ------------------------------------------------------------------------


def test_indexing() -> None:
    y = Surv(time=[0, 2, 1], time2=[5, 6, 4], event=[1, 0, 1])
    assert len(y[1]) == 1
    np.testing.assert_array_equal(y[1:].stop, [6.0, 4.0])
    np.testing.assert_array_equal(y[[2, 0]].start, [1.0, 0.0])
    np.testing.assert_array_equal(y[np.array([True, False, True])].status, [1.0, 1.0])
    assert y[-1] == y[2]
    with pytest.raises(IndexError):
        y[3]
    with pytest.raises(TypeError):
        y[np.array([0.5])]


def test_indexing_keeps_states() -> None:
    y = Surv(time=[5, 6], event=pd.Categorical(["b", "c"], categories=["a", "b", "c"]))
    assert y[0:1].states == ("b", "c")


def test_equality() -> None:
    y = Surv(time=[5, 6], event=[1, 0])
    assert y == Surv(time=[5, 6], event=[2, 1])
    assert y != Surv(time=[5, 6], event=[1, 1])
    assert y != Surv(time=[5, 6], event=[1, 0], type="left")
    assert y.__eq__([5, 6]) is NotImplemented
    with pytest.raises(TypeError):
        hash(y)


def test_repr() -> None:
    assert repr(Surv(time=[5, 6], event=[1, 0])) == "Surv(type=right, n=2, events=1)"
    y = Surv(time=[5, 6], event=pd.Categorical(["b", "a"], categories=["a", "b"]))
    assert "states=('b',)" in repr(y)
    assert "missing=1" in repr(Surv(time=[5, np.nan], event=[1, 0]))


# -- serialization and export -----------------------------------------------------------------


@pytest.mark.parametrize(
    "y",
    [
        Surv(time=[5, 6], event=[1, 0]),
        Surv(time=[5, 6], event=[1, 0], type="left"),
        Surv(time=[0, 2], time2=[5, 6], event=[1, 0]),
        Surv(time=[1, 2, np.nan], time2=[1, np.inf, 4], type="interval2"),
        Surv(time=[5, 6], event=pd.Categorical(["b", "a"], categories=["a", "b"])),
        Surv(time=[5, np.nan], event=[1, 0]),
    ],
    ids=["right", "left", "counting", "interval", "mright", "missing"],
)
def test_json_round_trip(y: Surv) -> None:
    assert Surv.from_json(y.to_json()) == y
    assert Surv.from_dict(y.to_dict()) == y


def test_from_dict_rejects_other_layouts() -> None:
    with pytest.raises(ValueError, match="layout version"):
        Surv.from_dict({"type": "right", "stop": [1.0], "status": [1]})


@pytest.mark.parametrize("format", ["pandas", "polars", "pyarrow"])
def test_to_frame_backends(format: str) -> None:
    frame = Surv(time=[0, 2], time2=[5, 6], event=[1, 0]).to_frame(format=format)
    columns = frame.column_names if format == "pyarrow" else list(frame.columns)
    assert list(columns) == ["start", "stop", "status"]


def test_to_frame_uses_nulls_for_missing_values() -> None:
    frame = Surv(time=[5, np.nan], event=[1, 0]).to_frame(format="polars")
    assert frame["time"].null_count() == 1


# -- first_event ----------------------------------------------------------------------------


def test_first_event_earliest_wins_and_censoring_time() -> None:
    y = gw.first_event(
        endpoints={"a": ([5.0, 9.0, 4.0], [1, 0, 0]), "b": ([3.0, 8.0, 6.0], [1, 1, 0])}
    )
    assert y.type is CensoringType.MRIGHT
    assert y.states == ("a", "b")
    np.testing.assert_array_equal(y.stop, [3.0, 8.0, 6.0])
    np.testing.assert_array_equal(y.status, [2.0, 2.0, 0.0])


def test_first_event_ties_follow_endpoint_order() -> None:
    a_first = gw.first_event(endpoints={"a": ([5.0], [1]), "b": ([5.0], [1])})
    b_first = gw.first_event(endpoints={"b": ([5.0], [1]), "a": ([5.0], [1])})
    assert a_first.states is not None and b_first.states is not None
    assert a_first.states[int(a_first.status[0]) - 1] == "a"
    assert b_first.states[int(b_first.status[0]) - 1] == "b"


def test_first_event_censor_at_and_start() -> None:
    y = gw.first_event(
        endpoints={"a": ([5.0, 9.0], [0, 0]), "b": ([3.0, 8.0], [0, 0])},
        censor_at=[10.0, 11.0],
        start=[1.0, 2.0],
    )
    assert y.type is CensoringType.MCOUNTING
    np.testing.assert_array_equal(y.stop, [10.0, 11.0])
    np.testing.assert_array_equal(y.entry, [1.0, 2.0])


def test_first_event_warns_on_event_after_follow_up_ends() -> None:
    with pytest.warns(UserWarning, match="after another endpoint's follow-up"):
        gw.first_event(endpoints={"pcm": ([80.0], [1]), "death": ([60.0], [0])})


@pytest.mark.parametrize(
    ("endpoints", "message"),
    [
        ({}, "at least one endpoint"),
        ({"a": ([1.0], [1], 1)}, "must be a `\\(time, event\\)` pair"),
        ({"a": ([1.0], [3])}, "must be logical or 0/1"),
        ({"a": ([1.0, 2.0], [1, 0]), "b": ([1.0], [1])}, "same length"),
    ],
)
def test_first_event_validation(endpoints: Any, message: str) -> None:
    with pytest.raises(ValueError, match=message), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gw.first_event(endpoints=endpoints)


# -- as_surv --------------------------------------------------------------------------------


def test_as_surv_picks_the_type_from_the_codes() -> None:
    assert gw.as_surv(gw.event_time(time=[7, 5], status=["e", "r"])).type == "right"
    assert gw.as_surv(gw.event_time(time=[7, 5], status=["e", "l"])).type == "left"
    assert gw.as_surv(gw.event_time(time=[7, 5], status=["r", "l"])).type == "interval"


def test_estimators_accept_event_time() -> None:
    lung = gw.load_dataset("lung")
    status = np.where(lung["status"].to_numpy() == 2, "e", "r")
    et = gw.event_time(time=lung["time"], status=status)
    from_et = gw.KaplanMeier().fit(et)
    from_surv = gw.KaplanMeier().fit(Surv(time=lung["time"], event=lung["status"]))
    np.testing.assert_allclose(from_et.survival_, from_surv.survival_)
