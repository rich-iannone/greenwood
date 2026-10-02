"""Python-side behavior of `EventTime` that the etd conformance fixtures don't cover.

`tests/test_event_time_conformance.py` checks every observable result against etd itself. This
file covers what only exists in Python: DataFrame and NumPy inputs, the error attributes, the
vector protocol (indexing, combining, equality), `repr()`, and `to_frame()` backends.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import polars as pl
import pytest

import greenwood as gw


@pytest.fixture
def x() -> gw.EventTime:
    return gw.event_time(
        time=[7, 5, 3, 2, None],
        status=["e", "r", "l", "i", None],
        time_max=[None, None, None, 4, None],
    )


# -- inputs ---------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("time", "status"),
    [
        ([1.0, None], ["e", None]),
        (np.array([1.0, np.nan]), np.array(["e", None], dtype=object)),
        (pd.Series([1.0, None]), pd.Series(["e", None])),
        (pl.Series([1, None]), pl.Series(["e", None])),
    ],
    ids=["list", "numpy", "pandas", "polars"],
)
def test_inputs_from_every_backend_agree(time: Any, status: Any) -> None:
    assert gw.event_time(time, status).format() == ["1 ", None]


def test_scalar_inputs_are_length_one() -> None:
    assert gw.event_time(3, "e").format() == ["3 "]
    assert gw.event_time(2, "i", 4).format() == ["[2, 4]"]


def test_numpy_string_status_is_accepted() -> None:
    assert gw.event_time(np.array([1, 2]), np.array(["e", "r"])).format() == ["1 ", "2+"]


@pytest.mark.parametrize(
    "status",
    [pl.Series(["e"], dtype=pl.Categorical), pd.Series(["e"], dtype="category")],
    ids=["polars", "pandas"],
)
def test_categorical_status_is_rejected_like_an_r_factor(status: Any) -> None:
    with pytest.raises(TypeError) as excinfo:
        gw.event_time([1.0], status)
    assert excinfo.value.check == "status_not_character"  # pyright: ignore[reportAttributeAccessIssue]


def test_dates_are_not_numeric() -> None:
    with pytest.raises(TypeError) as excinfo:
        gw.event_time(np.array(["2020-01-01"], dtype="datetime64[D]"), ["e"])
    assert excinfo.value.check == "time_not_numeric"  # pyright: ignore[reportAttributeAccessIssue]


def test_two_dimensional_time_is_rejected() -> None:
    with pytest.raises(TypeError):
        gw.event_time(np.ones((2, 2)), ["e", "e"])


# -- errors ---------------------------------------------------------------------------------


def test_locations_are_zero_based_and_indexable() -> None:
    time = np.array([-1.0, 2.0, -3.0])
    with pytest.raises(ValueError) as excinfo:
        gw.event_time(time, ["e", "e", "e"])
    error: Any = excinfo.value
    assert error.check == "time_negative"
    assert error.locations.dtype == np.int64
    assert list(time[error.locations]) == [-1.0, -3.0]
    assert "locations 0 and 2" in str(error)


def test_checks_without_locations_have_an_empty_array() -> None:
    with pytest.raises(ValueError) as excinfo:
        gw.event_time([1.0, 2.0], ["e"])
    error: Any = excinfo.value
    assert error.check == "status_length"
    assert error.locations.shape == (0,)


def test_long_location_lists_are_summarized() -> None:
    with pytest.raises(ValueError) as excinfo:
        gw.event_time(list(range(30)), ["x"] * 30)
    error: Any = excinfo.value
    assert len(error.locations) == 30
    assert str(error).endswith("19, and 10 more.")


def test_extractors_reject_other_types() -> None:
    with pytest.raises(TypeError):
        gw.extract_time([1.0])
    with pytest.raises(TypeError):
        gw.extract_status([1.0])


@pytest.mark.parametrize("time", [1.0, None, "abc"])
def test_new_event_time_requires_a_sequence(time: Any) -> None:
    with pytest.raises(TypeError) as excinfo:
        gw.new_event_time(time=time, status=["e"])
    assert excinfo.value.check == "time_not_list"  # pyright: ignore[reportAttributeAccessIssue]


def test_new_event_time_rejects_elements_longer_than_a_pair() -> None:
    with pytest.raises(TypeError) as excinfo:
        gw.new_event_time(time=[(1, 2, 3)], status=["i"])
    assert excinfo.value.check == "time_not_list"  # pyright: ignore[reportAttributeAccessIssue]


# -- vector protocol ------------------------------------------------------------------------


def test_integer_index_returns_a_length_one_vector(x: gw.EventTime) -> None:
    assert x[1].format() == ["5+"]
    assert x[-2].format() == ["[2, 4]"]


def test_slices_arrays_and_masks(x: gw.EventTime) -> None:
    assert x[1:4].format() == ["5+", "3-", "[2, 4]"]
    assert x[[0, 3]].format() == ["7 ", "[2, 4]"]
    assert x[np.array([True, False, True, False, True])].format() == ["7 ", "3-", None]
    assert len(x[0:0]) == 0


def test_out_of_bounds_index_raises(x: gw.EventTime) -> None:
    with pytest.raises(IndexError):
        x[5]


def test_non_integer_index_raises(x: gw.EventTime) -> None:
    with pytest.raises(TypeError):
        x[np.array([0.5])]


def test_concat(x: gw.EventTime) -> None:
    combined = gw.EventTime.concat([x[0:2], x[3:4]])
    assert combined.format() == ["7 ", "5+", "[2, 4]"]
    assert len(gw.EventTime.concat([])) == 0
    with pytest.raises(TypeError):
        gw.EventTime.concat([x, [1.0]])  # pyright: ignore[reportArgumentType]


def test_equality_compares_whole_vectors(x: gw.EventTime) -> None:
    assert x == x[0:5]
    assert x != x[0:4]
    assert x.__eq__([1.0]) is NotImplemented


def test_event_time_is_unhashable(x: gw.EventTime) -> None:
    with pytest.raises(TypeError):
        hash(x)


# -- display and export ---------------------------------------------------------------------


def test_repr(x: gw.EventTime) -> None:
    assert repr(x) == "<event_time[5]>\n7      5+     3-     [2, 4] <NA>"
    assert repr(x[0:0]) == "<event_time[0]>"


def test_repr_wraps_at_80_columns() -> None:
    lines = repr(gw.event_time(np.arange(40) * 1.5, ["r"] * 40)).splitlines()
    assert lines[0] == "<event_time[40]>"
    assert all(len(line) <= 80 for line in lines)
    assert len(lines) > 2


@pytest.mark.parametrize("format", ["pandas", "polars", "pyarrow"])
def test_to_frame_backends(x: gw.EventTime, format: str) -> None:
    frame = x.to_frame(format=format)
    assert list(frame.columns if format != "pyarrow" else frame.column_names) == [
        "time",
        "status",
        "time_max",
    ]


def test_to_frame_uses_nulls_for_missing_values(x: gw.EventTime) -> None:
    frame = x.to_frame(format="polars")
    assert frame["time"].null_count() == 1
    assert frame["time_max"].null_count() == 4
    assert frame["time"].is_nan().sum() == 0
