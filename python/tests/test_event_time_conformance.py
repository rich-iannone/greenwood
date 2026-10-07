"""Conformance of Greenwood's `event_time` port against the pinned etd package.

Greenwood follows etd (https://github.com/topepo/etd) exactly. The cases under
`spec/conformance/event_time/` are generated from etd itself by
`scripts/generate_event_time_conformance.R`. See `spec/event_time.md` for the contract and the
documented Python adaptations.

Two kinds of test live here:

- Fixture integrity tests check the fixtures are pinned to the expected etd commit and are
  internally consistent, so a bad regeneration is caught immediately.
- Behaviour tests replay every case against `greenwood.event_time` and friends.
"""

from __future__ import annotations

from typing import Any

import pytest

import greenwood as gw

from ._conformance import (
    SPEC_DIR,
    decode_input,
    floats_equal,
    load_conformance,
    r_floats,
    to_python_locations,
)

pytestmark = pytest.mark.rparity

# The etd commit the port targets. It must match the fixtures' metadata and spec/event_time.md.
ETD_SHA = "3d5caff4f03d771a6b65b9605c9b0e48a912482f"

# Every check id the generator may record, mapped to the Python exception type that reports it.
# Type problems raise TypeError and value problems raise ValueError (spec/event_time.md).
CHECKS: dict[str, type[Exception]] = {
    "time_not_numeric": TypeError,
    "time_max_not_numeric": TypeError,
    "status_not_character": TypeError,
    "status_length": ValueError,
    "time_max_length": ValueError,
    "time_negative": ValueError,
    "status_missing": ValueError,
    "status_invalid_value": ValueError,
    "time_max_not_allowed": ValueError,
    "time_max_missing": ValueError,
    "interval_bounds_order": ValueError,
    "time_not_list": TypeError,
    "fields_size": ValueError,
    "as_surv_unsupported": TypeError,
}

# Cases whose R input has no faithful Python equivalent. These are documented adaptations
# (spec/event_time.md, "Python adaptations"), not failures.
ADAPTATIONS: dict[str, str] = {
    "constructor/status_logical": (
        "R distinguishes a logical NA from a character NA. In Python both are None, so a status "
        "of [None] is a missing character status, not a non-character one."
    ),
    "new_event_time/time_not_list": (
        "R distinguishes an atomic vector from a list. A Python list of floats is a valid "
        "per-element time sequence."
    ),
}

# Observations where etd itself fails, keyed as "<fixture>/<case>/<observation>". These are
# reported upstream (spec/event_time.md, "Upstream issues") rather than mirrored, so Python is not
# compared on them. Any other etd failure in the fixtures is treated as a regeneration problem.
UPSTREAM_ISSUES: dict[str, str] = {
    "constructor/empty/format": "etd's format() errors on a zero-length vector.",
    "new_event_time/empty/format": "etd's format() errors on a zero-length vector.",
}

OBSERVATIONS = ("is_na", "format", "extract_time", "extract_status", "as_surv", "as_tibble")


def _cases(name: str) -> list[dict[str, Any]]:
    return load_conformance("event_time", name)


def _params(name: str, *, ok: bool | None = None) -> list[Any]:
    """Parametrize over the cases of one fixture file, skipping documented adaptations."""
    params: list[Any] = []
    for case in _cases(name):
        if ok is not None and case["result"]["ok"] is not ok:
            continue
        key = f"{name}/{case['id']}"
        marks = [pytest.mark.skip(reason=ADAPTATIONS[key])] if key in ADAPTATIONS else []
        params.append(pytest.param(case, id=case["id"], marks=marks))
    return params


# -- fixture integrity ---------------------------------------------------------------


def test_fixtures_are_pinned_to_the_expected_etd_commit() -> None:
    metadata = load_conformance("event_time", "metadata")
    assert metadata["etd_sha"] == ETD_SHA
    assert ETD_SHA in (SPEC_DIR / "event_time.md").read_text()


@pytest.mark.parametrize("name", ["constructor", "new_event_time", "conversion", "vector_ops"])
def test_case_ids_are_unique(name: str) -> None:
    ids = [case["id"] for case in _cases(name)]
    assert len(ids) == len(set(ids))


@pytest.mark.parametrize("name", ["constructor", "new_event_time", "conversion"])
def test_error_cases_use_known_checks_and_valid_locations(name: str) -> None:
    for case in _cases(name):
        result = case["result"]
        if result["ok"]:
            continue
        assert result["check"] in CHECKS, case["id"]
        for location in result["locations"] or []:
            assert isinstance(location, int) and location >= 1, case["id"]


def test_every_check_is_exercised() -> None:
    seen = {
        case["result"]["check"]
        for name in ("constructor", "new_event_time", "conversion")
        for case in _cases(name)
        if not case["result"]["ok"]
    }
    assert seen == set(CHECKS)


@pytest.mark.parametrize("name", ["constructor", "new_event_time"])
def test_etd_failures_on_valid_vectors_are_all_known_upstream_issues(name: str) -> None:
    for case in _cases(name):
        result = case["result"]
        if not result["ok"]:
            continue
        for observation in OBSERVATIONS:
            key = f"{name}/{case['id']}/{observation}"
            failed = not result[observation]["ok"]
            # A new failure means a bad regeneration (or a new etd bug to report). A known issue
            # that no longer fails means etd fixed it: drop it here and in spec/event_time.md.
            assert failed == (key in UPSTREAM_ISSUES), key


def test_format_battery_has_no_etd_failures() -> None:
    assert all(case["format"]["ok"] for case in _cases("format"))


# -- behaviour -------------------------------------------------


def _build(case: dict[str, Any]) -> Any:
    inputs = case["input"]
    return gw.event_time(  # pyright: ignore[reportAttributeAccessIssue]
        decode_input(inputs["time"]),
        decode_input(inputs["status"]),
        decode_input(inputs["time_max"]),
    )


def _new_time_input(encoded: dict[str, Any]) -> Any:
    # etd's per-element form: a length-1 double is a single time, a length-2 double an interval.
    elements: list[Any] = []
    for element in encoded["values"]:
        values = decode_input(element)
        elements.append(values[0] if len(values) == 1 else tuple(values))
    return elements


def _build_new(case: dict[str, Any]) -> Any:
    inputs = case["input"]
    time = inputs["time"]
    time_value = _new_time_input(time) if time["r_type"] == "list" else decode_input(time)
    return gw.new_event_time(  # pyright: ignore[reportAttributeAccessIssue]
        time_value, decode_input(inputs["status"])
    )


def _assert_error(excinfo: pytest.ExceptionInfo[Exception], result: dict[str, Any]) -> None:
    assert isinstance(excinfo.value, CHECKS[result["check"]])
    error: Any = excinfo.value
    assert error.check == result["check"]
    assert list(error.locations) == to_python_locations(result["locations"])


def _assert_observations(x: Any, result: dict[str, Any], case_key: str) -> None:
    assert len(x) == result["length"]
    assert list(x.is_na()) == result["is_na"]["value"]

    if f"{case_key}/format" not in UPSTREAM_ISSUES:
        assert list(x.format()) == result["format"]["value"]

    extract = result["extract_time"]["value"]
    actual = gw.extract_time(x)  # pyright: ignore[reportAttributeAccessIssue]
    if extract["shape"] == "vector":
        assert actual.ndim == 1
        assert floats_equal(list(actual), r_floats(extract["values"]))
    else:
        assert actual.ndim == 2 and actual.shape[1] == 2
        assert floats_equal(list(actual[:, 0]), r_floats(extract["columns"]["time"]))
        assert floats_equal(list(actual[:, 1]), r_floats(extract["columns"]["time_max"]))

    assert list(gw.extract_status(x)) == result["extract_status"]["value"]  # pyright: ignore[reportAttributeAccessIssue]

    frame = result["as_tibble"]["value"]
    columns = x.to_frame(format="polars").to_dict(as_series=False)
    assert list(columns) == ["time", "status", "time_max"]
    assert floats_equal(columns["time"], r_floats(frame["time"]))
    assert columns["status"] == frame["status"]
    assert floats_equal(columns["time_max"], r_floats(frame["time_max"]))


@pytest.mark.parametrize("case", _params("constructor", ok=False))
def test_event_time_validation_matches_etd(case: dict[str, Any]) -> None:
    with pytest.raises(Exception) as excinfo:
        _build(case)
    _assert_error(excinfo, case["result"])


@pytest.mark.parametrize("case", _params("constructor", ok=True))
def test_event_time_observations_match_etd(case: dict[str, Any]) -> None:
    _assert_observations(_build(case), case["result"], f"constructor/{case['id']}")


@pytest.mark.parametrize("case", _params("new_event_time", ok=False))
def test_new_event_time_validation_matches_etd(case: dict[str, Any]) -> None:
    with pytest.raises(Exception) as excinfo:
        _build_new(case)
    _assert_error(excinfo, case["result"])


@pytest.mark.parametrize("case", _params("new_event_time", ok=True))
def test_new_event_time_observations_match_etd(case: dict[str, Any]) -> None:
    _assert_observations(_build_new(case), case["result"], f"new_event_time/{case['id']}")


@pytest.mark.parametrize("case", _params("format"))
def test_format_matches_etd(case: dict[str, Any]) -> None:
    expected = case["format"]
    assert expected["ok"], case["id"]
    assert list(_build(case).format()) == expected["value"]


@pytest.mark.parametrize("case", _params("conversion"))
def test_as_surv_rejects_unsupported_input(case: dict[str, Any]) -> None:
    with pytest.raises(Exception) as excinfo:
        gw.as_surv(decode_input(case["input"]["x"]))  # pyright: ignore[reportAttributeAccessIssue]
    _assert_error(excinfo, case["result"])


@pytest.mark.parametrize("case", _params("constructor", ok=True))
def test_as_surv_matches_etd(case: dict[str, Any]) -> None:
    expected = case["result"]["as_surv"]
    if not expected["ok"]:
        pytest.skip(f"etd's as_surv fails here: {expected['message']}")
    surv = gw.as_surv(_build(case))  # pyright: ignore[reportAttributeAccessIssue]
    assert surv.type == expected["value"]["type"]
    columns = surv.to_frame(format="polars").to_dict(as_series=False)
    assert list(columns) == list(expected["value"]["columns"])
    for name, values in expected["value"]["columns"].items():
        assert floats_equal(columns[name], r_floats(values)), name
