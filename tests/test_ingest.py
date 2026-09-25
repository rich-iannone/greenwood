"""Tests for building `Surv` responses from column names and declared event encodings.

The column-name form (`data=`) with `event_value=` / `censor_value=` / mapped `states=` must give
the same response as the array form, on every supported backend, eager or lazy.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

import greenwood as gw
from greenwood import CensoringType, Surv


@pytest.fixture(scope="module")
def lung_pd() -> Any:
    return gw.load_dataset("lung", backend="pandas")


@pytest.fixture(scope="module")
def reference(lung_pd: Any) -> Surv:
    return Surv.right(lung_pd["time"].to_numpy(), (lung_pd["status"] == 2).to_numpy())


def _backend_frames(lung_pd: Any) -> dict[str, Any]:
    import polars as pl
    import pyarrow as pa

    frames: dict[str, Any] = {
        "pandas": lung_pd,
        "polars": pl.from_pandas(lung_pd),
        "polars-lazy": pl.from_pandas(lung_pd).lazy(),
        "pyarrow": pa.Table.from_pandas(lung_pd),
        "dict": {c: lung_pd[c].to_numpy() for c in lung_pd.columns},
    }
    try:
        import duckdb  # pyright: ignore[reportMissingImports]

        frames["duckdb"] = duckdb.from_arrow(pa.Table.from_pandas(lung_pd))  # pyright: ignore
    except ImportError:
        pass
    return frames


def _assert_same(a: Surv, b: Surv) -> None:
    assert a.type is b.type
    np.testing.assert_array_equal(a.stop, b.stop)
    np.testing.assert_array_equal(a.status, b.status)
    assert a.states == b.states


# -- backend parity -----------------------------------------------------------


@pytest.mark.parametrize("backend", ["pandas", "polars", "polars-lazy", "pyarrow", "dict"])
def test_right_by_name_matches_array_form(lung_pd: Any, reference: Surv, backend: str) -> None:
    data = _backend_frames(lung_pd)[backend]
    y = Surv.right("time", "status", data=data, event_value=2)

    _assert_same(y, reference)


def test_duckdb_relation(lung_pd: Any, reference: Surv) -> None:
    frames = _backend_frames(lung_pd)
    if "duckdb" not in frames:
        pytest.skip("duckdb is not installed")

    _assert_same(Surv.right("time", "status", data=frames["duckdb"], event_value=2), reference)


def test_names_and_values_can_mix(lung_pd: Any, reference: Surv) -> None:
    y = Surv.right(lung_pd["time"], "status", data=lung_pd, event_value=2)

    _assert_same(y, reference)


def test_fits_the_same_model(lung_pd: Any, reference: Surv) -> None:
    import polars as pl

    y = Surv.right("time", "status", data=pl.from_pandas(lung_pd), event_value=2)
    km_new = gw.KaplanMeier().fit(y)
    km_ref = gw.KaplanMeier().fit(reference)

    np.testing.assert_allclose(km_new.survival_, km_ref.survival_, rtol=1e-12)


# -- event encodings ----------------------------------------------------------


def test_event_value_list() -> None:
    y = Surv.right([1, 2, 3, 4], [0, 1, 2, 3], event_value=[1, 2])

    np.testing.assert_array_equal(y.status, [0, 1, 1, 0])


def test_censor_value_makes_everything_else_an_event() -> None:
    y = Surv.right([1, 2, 3], [0, 1, 2], censor_value=0)

    np.testing.assert_array_equal(y.status, [0, 1, 1])


def test_string_event_value() -> None:
    import polars as pl

    df = pl.DataFrame({"t": [3.0, 5.0, 2.0], "outcome": ["died", "censored", "withdrew"]})
    y = Surv.right("t", "outcome", data=df, event_value="died")

    np.testing.assert_array_equal(y.status, [1, 0, 0])


def test_boolean_event_value() -> None:
    y = Surv.right([1, 2], [True, False], event_value=False)

    np.testing.assert_array_equal(y.status, [0, 1])


def test_default_still_accepts_bool_and_01() -> None:
    np.testing.assert_array_equal(Surv.right([1, 2], [True, False]).status, [1, 0])
    np.testing.assert_array_equal(Surv.right([1, 2], [1, 0]).status, [1, 0])


def test_r_coding_error_names_the_fix() -> None:
    with pytest.raises(ValueError, match="event_value=2"):
        Surv.right([5, 6, 4], [1, 2, 2])


def test_multilevel_status_error_names_the_options() -> None:
    with pytest.raises(ValueError, match="censor_value"):
        Surv.right([5, 6, 4], [0, 1, 2])


def test_string_column_without_encoding_errors() -> None:
    with pytest.raises(ValueError, match="event_value='died'"):
        Surv.right([5, 6], ["died", "alive"])


def test_event_and_censor_value_are_exclusive() -> None:
    with pytest.raises(ValueError, match="not both"):
        Surv.right([1, 2], [0, 1], event_value=1, censor_value=0)


def test_encoding_without_event_column_errors() -> None:
    with pytest.raises(ValueError, match="need an `event` column"):
        Surv.right([1, 2], event_value=1)


def test_kind_mismatch_errors() -> None:
    with pytest.raises(TypeError, match="same kind"):
        Surv.right([1, 2], [1, 2], event_value="2")
    with pytest.raises(TypeError, match="same kind"):
        Surv.right([1, 2], ["a", "b"], event_value=1)


def test_no_matching_rows_warns() -> None:
    with pytest.warns(UserWarning, match="No rows"):
        y = Surv.right([1, 2], ["alive", "alive"], event_value="Died")

    assert y.n_events == 0


def test_missing_event_values_error() -> None:
    with pytest.raises(ValueError, match="1 missing"):
        Surv.right([1, 2, 3], [1.0, np.nan, 2.0], event_value=2)
    with pytest.raises(ValueError, match="1 missing"):
        Surv.right([1, 2], ["died", None], event_value="died")


# -- column resolution errors -------------------------------------------------


def test_column_name_without_data_errors() -> None:
    with pytest.raises(ValueError, match="no `data=` was given"):
        Surv.right("time", "status")


def test_unknown_column_suggests_close_match(lung_pd: Any) -> None:
    with pytest.raises(KeyError, match="Did you mean 'status'"):
        Surv.right("time", "stauts", data=lung_pd, event_value=2)


def test_non_frame_data_errors() -> None:
    with pytest.raises(TypeError, match="must be a DataFrame"):
        Surv.right("time", data=23)


# -- other constructors -------------------------------------------------------


def test_weights_by_name() -> None:
    data = {"t": [1.0, 2.0, 3.0], "e": [2, 1, 2], "w": [0.5, 1.0, 2.0]}
    y = Surv.right("t", "e", weights="w", data=data, event_value=2)

    assert y.weights is not None
    np.testing.assert_array_equal(y.weights, [0.5, 1.0, 2.0])


def test_left_by_name() -> None:
    data = {"t": [5.0, 6.0, 4.0], "e": ["yes", "no", "yes"]}
    y = Surv.left("t", "e", data=data, event_value="yes")

    assert y.type is CensoringType.LEFT
    np.testing.assert_array_equal(y.status, [1, 0, 1])


def test_counting_by_name() -> None:
    data = {"a": [0.0, 2.0, 1.0], "b": [5.0, 6.0, 4.0], "e": [2, 1, 2]}
    y = Surv.counting("a", "b", "e", data=data, event_value=2)

    _assert_same(y, Surv.counting([0, 2, 1], [5, 6, 4], [1, 0, 1]))


def test_interval_by_name() -> None:
    data = {"lo": [1.0, 2.0, 3.0], "hi": [2.0, np.inf, 5.0]}
    y = Surv.interval("lo", "hi", data=data)

    _assert_same(y, Surv.interval([1, 2, 3], [2, np.inf, 5]))


# -- multistate ---------------------------------------------------------------


def test_multistate_mapping_matches_legacy() -> None:
    legacy = Surv.multistate([5, 6, 7, 8], [1, 2, 0, 1], states=("relapse", "death"))
    mapped = Surv.multistate([5, 6, 7, 8], [1, 2, 0, 1], states={"relapse": 1, "death": 2})

    _assert_same(mapped, legacy)


def test_multistate_mapping_reorders_codes() -> None:
    # Label order sets the internal code, independent of the raw values.
    y = Surv.multistate([5, 6, 7], [2, 1, 0], states={"death": 1, "relapse": 2})

    assert y.states == ("death", "relapse")
    np.testing.assert_array_equal(y.status, [2, 1, 0])


def test_multistate_string_mapping_by_name() -> None:
    import polars as pl

    df = pl.DataFrame({"t": [5.0, 6.0, 7.0, 8.0], "out": ["rel", "dth", "alive", "lost"]})
    y = Surv.multistate(
        "t",
        "out",
        data=df,
        states={"relapse": "rel", "death": "dth"},
        censor_value=["alive", "lost"],
    )

    np.testing.assert_array_equal(y.status, [1, 2, 0, 0])


def test_multistate_state_can_list_several_values() -> None:
    y = Surv.multistate([1, 2, 3], [1, 2, 3], states={"a": 1, "b": [2, 3]})

    np.testing.assert_array_equal(y.status, [1, 2, 2])


def test_multistate_unassigned_value_errors() -> None:
    with pytest.raises(ValueError, match="not assigned"):
        Surv.multistate([1, 2, 3], [0, 1, 9], states={"a": 1})


def test_multistate_overlap_errors() -> None:
    with pytest.raises(ValueError, match="one outcome"):
        Surv.multistate([1, 2], [0, 1], states={"a": 1, "b": 1})


def test_multistate_censor_value_needs_mapping() -> None:
    with pytest.raises(ValueError, match="only applies"):
        Surv.multistate([1, 2], [0, 1], states=("a",), censor_value=0)


def test_duckdb_dotted_column_names() -> None:
    # SQL backends read `ph.ecog` as table `ph`, column `ecog` unless selected eagerly.
    duckdb = pytest.importorskip("duckdb")
    import pyarrow as pa

    rel = duckdb.from_arrow(pa.table({"fu.time": [5.0, 6.0, 4.0], "ph.status": [2, 1, 2]}))
    y = Surv.right("fu.time", "ph.status", data=rel, event_value=2)

    np.testing.assert_array_equal(y.status, [1, 0, 1])
