"""Tests for the repr formatting helpers, including the missing-value branches."""

from __future__ import annotations

import math

from greenwood._repr import align_table, fixed, num, whole


def test_num_formats_and_handles_missing() -> None:
    assert num(1.23456, digits=3) == "1.23"
    assert num(0.0001234) == "0.0001234"
    assert num(None) == "NA"
    assert num(math.nan) == "NA"


def test_fixed_formats_and_handles_missing() -> None:
    assert fixed(1.23456, digits=2) == "1.23"
    assert fixed(-3.0, digits=1) == "-3.0"
    assert fixed(None) == "NA"
    assert fixed(math.nan) == "NA"


def test_whole_integers_floats_and_missing() -> None:
    assert whole(5.0) == "5"
    assert whole(310.0) == "310"
    assert whole(2.5) == "2.5"  # non-integer falls back to significant figures
    assert whole(None) == "NA"
    assert whole(math.nan) == "NA"


def test_align_table_with_and_without_labels() -> None:
    plain = align_table(["a", "bb"], [["1", "22"], ["333", "4"]])
    # Columns are right-aligned to the widest cell; header row present.
    assert plain.splitlines()[0].endswith("bb")
    assert "333" in plain

    labeled = align_table(["coef"], [["1.0"], ["2.0"]], row_labels=["age", "sex"])
    lines = labeled.splitlines()
    assert lines[1].startswith("age")
    assert lines[2].startswith("sex")
