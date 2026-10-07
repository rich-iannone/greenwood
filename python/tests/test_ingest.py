"""Tests for building responses from column names and expressions.

An `Outcome` bound to data (`Outcome.surv(...).bind(data)`) evaluates its expressions
(`"status == 2"`, `"status %in% c(1, 2)"`, `"factor(cause, ...)"`) and passes the results to
`Surv()`. It must give the same response as `Surv()` on the array form, on every supported backend,
eager or lazy, and follow R's rules for status codes and missing values.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd
import pytest

import greenwood as gw
from greenwood import CensoringType, Outcome, Surv

DEATH = Outcome.surv(time="time", event="status == 2")
BACKENDS = ["pandas", "polars", "polars-lazy", "pyarrow", "duckdb", "dict"]


@pytest.fixture(scope="module")
def lung_pd() -> Any:
    return gw.load_dataset("lung", backend="pandas")


@pytest.fixture(scope="module")
def reference(lung_pd: Any) -> Surv:
    return Surv(time=lung_pd["time"].to_numpy(), event=(lung_pd["status"] == 2).to_numpy())


def _as_backend(frame: pd.DataFrame, backend: str) -> Any:
    import polars as pl
    import pyarrow as pa

    if backend == "pandas":
        return frame
    if backend == "polars":
        return pl.from_pandas(frame)
    if backend == "polars-lazy":
        return pl.from_pandas(frame).lazy()
    if backend == "pyarrow":
        return pa.Table.from_pandas(frame, preserve_index=False)
    if backend == "duckdb":
        duckdb = pytest.importorskip("duckdb")
        return duckdb.from_arrow(pa.Table.from_pandas(frame, preserve_index=False))
    return {c: frame[c].to_numpy() for c in frame.columns}


def _assert_same(a: Surv, b: Surv) -> None:
    assert a.type is b.type
    np.testing.assert_array_equal(a.stop, b.stop)
    np.testing.assert_array_equal(a.status, b.status)
    assert a.states == b.states


# -- backend parity -----------------------------------------------------------


@pytest.mark.parametrize("backend", BACKENDS)
def test_comparison_by_name_matches_array_form(lung_pd: Any, reference: Surv, backend: str) -> None:
    y = DEATH.bind(_as_backend(lung_pd, backend))

    _assert_same(y, reference)


@pytest.mark.parametrize("backend", BACKENDS)
def test_r_one_two_status_by_name_matches_array_form(
    lung_pd: Any, reference: Surv, backend: str
) -> None:
    # R's 1/2 coding (as in `lung`) passes straight through `Surv()`, so no expression is needed.
    y = Outcome.surv(time="time", event="status").bind(_as_backend(lung_pd, backend))

    _assert_same(y, reference)


def test_r_one_two_status_passes_straight_to_surv(lung_pd: Any, reference: Surv) -> None:
    y = Surv(time=lung_pd["time"], event=lung_pd["status"])

    _assert_same(y, reference)


def test_fits_the_same_model(lung_pd: Any, reference: Surv) -> None:
    import polars as pl

    y = DEATH.bind(pl.from_pandas(lung_pd))
    km_new = gw.KaplanMeier().fit(y)
    km_ref = gw.KaplanMeier().fit(reference)

    np.testing.assert_allclose(km_new.survival_, km_ref.survival_, rtol=1e-12)


# -- event expressions --------------------------------------------------------


def test_in_list() -> None:
    data = {"t": [1, 2, 3, 4], "e": [0, 1, 2, 3]}

    for event in ("e in (1, 2)", "e %in% c(1, 2)", "e in [1, 2]", "e %in% 1:2"):
        y = Outcome.surv(time="t", event=event).bind(data)

        np.testing.assert_array_equal(y.status, [0, 1, 1, 0])


def test_not_equal_makes_everything_else_an_event() -> None:
    y = Outcome.surv(time="t", event="e != 0").bind({"t": [1, 2, 3], "e": [0, 1, 2]})

    np.testing.assert_array_equal(y.status, [0, 1, 1])


def test_not_in_list() -> None:
    y = Outcome.surv(time="t", event="e not in (0, 9)").bind({"t": [1, 2, 3, 4], "e": [0, 1, 9, 2]})

    np.testing.assert_array_equal(y.status, [0, 1, 0, 1])


def test_string_comparison() -> None:
    import polars as pl

    df = pl.DataFrame({"t": [3.0, 5.0, 2.0], "outcome": ["died", "censored", "withdrew"]})
    y = Outcome.surv(time="t", event="outcome == 'died'").bind(df)

    np.testing.assert_array_equal(y.status, [1, 0, 0])


def test_boolean_comparison() -> None:
    y = Outcome.surv(time="t", event="e == False").bind({"t": [1, 2], "e": [True, False]})

    np.testing.assert_array_equal(y.status, [0, 1])


def test_plain_column_accepts_bool_and_01() -> None:
    o = Outcome.surv(time="t", event="e")

    np.testing.assert_array_equal(o.bind({"t": [1, 2], "e": [True, False]}).status, [1, 0])
    np.testing.assert_array_equal(o.bind({"t": [1, 2], "e": [1, 0]}).status, [1, 0])


def test_r_one_two_coding_passes_through() -> None:
    y = Outcome.surv(time="t", event="e").bind({"t": [5, 6, 4], "e": [1, 2, 2]})

    np.testing.assert_array_equal(y.status, [0, 1, 1])


def test_multilevel_status_warns_as_r_does() -> None:
    # R's `Surv()` turns a status outside 0/1 (or 1/2) into NA with a warning.
    with pytest.warns(UserWarning, match="Invalid status value"):
        y = Outcome.surv(time="t", event="e").bind({"t": [5, 6, 4], "e": [0, 1, 2]})

    np.testing.assert_array_equal(y.status, [np.nan, 0, 1])


def test_string_column_without_expression_errors() -> None:
    with pytest.raises(TypeError, match="must be logical or numeric"):
        Outcome.surv(time="t", event="e").bind({"t": [5, 6], "e": ["died", "alive"]})


def test_unsupported_expression_is_read_as_a_column_name() -> None:
    # Outside a formula, a string that isn't a supported expression is a column name, so `<`
    # surfaces as an unknown column when the outcome is bound.
    o = Outcome.surv(time="t", event="e < 2")

    assert o.column_names == ("t", "e < 2")
    with pytest.raises(KeyError, match="not found"):
        o.bind({"t": [1, 2], "e": [0, 1]})


def test_no_matching_rows_gives_no_events() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        y = Outcome.surv(time="t", event="e == 'Died'").bind({"t": [1, 2], "e": ["alive", "alive"]})

    assert y.n_events == 0


def test_comparison_with_missing_value_is_missing() -> None:
    # R: `NA == 2` is NA, so the status is missing and the row is dropped at `fit()`.
    y = Outcome.surv(time="t", event="e == 2").bind({"t": [1, 2, 3], "e": [1.0, np.nan, 2.0]})

    np.testing.assert_array_equal(y.status, [0, np.nan, 1])

    y = Outcome.surv(time="t", event="e != 'died'").bind({"t": [1, 2], "e": ["died", None]})

    np.testing.assert_array_equal(y.status, [0, np.nan])


def test_in_with_missing_value_is_false() -> None:
    # R: `NA %in% c(1, 2)` is FALSE, so the row is censored rather than missing.
    data = {"t": [1, 2, 3], "e": [1.0, np.nan, 2.0]}

    y = Outcome.surv(time="t", event="e %in% c(1, 2)").bind(data)
    np.testing.assert_array_equal(y.status, [1, 0, 1])

    y = Outcome.surv(time="t", event="e not in (1, 2)").bind(data)
    np.testing.assert_array_equal(y.status, [0, 1, 0])


@pytest.mark.parametrize("backend", ["pandas", "polars", "pyarrow", "duckdb"])
def test_missing_value_rules_on_every_backend(backend: str) -> None:
    frame = pd.DataFrame({"t": [1.0, 2.0, 3.0], "e": [1.0, None, 2.0]})
    data = _as_backend(frame, backend)

    np.testing.assert_array_equal(
        Outcome.surv(time="t", event="e == 2").bind(data).status, [0, np.nan, 1]
    )
    np.testing.assert_array_equal(
        Outcome.surv(time="t", event="e %in% c(1, 2)").bind(data).status, [1, 0, 1]
    )


# -- column resolution errors -------------------------------------------------


def test_surv_does_not_resolve_column_names() -> None:
    # Like R's `Surv()`, column names are not looked up. Use `Outcome` to bind names to data.
    with pytest.raises(TypeError, match="not numeric"):
        Surv(time="time", event="status")


def test_unknown_column_suggests_close_match(lung_pd: Any) -> None:
    with pytest.raises(KeyError, match="Did you mean 'status'"):
        Outcome.surv(time="time", event="stauts == 2").bind(lung_pd)


def test_unknown_column_in_mapping_errors() -> None:
    with pytest.raises(KeyError, match="Did you mean 'stat'"):
        Outcome.surv(time="time", event="status").bind({"time": [1.0], "stat": [1]})


def test_non_frame_data_errors() -> None:
    with pytest.raises(TypeError, match="must be a DataFrame"):
        Outcome.surv(time="time").bind(23)


# -- other Surv types ---------------------------------------------------------


def test_weights_by_name_at_fit() -> None:
    # Weights are not part of the response. Name the column in `fit(weights=)` with `data=`.
    df = pd.DataFrame({"t": [1.0, 2.0, 3.0, 4.0], "e": [2, 1, 2, 2], "w": [0.5, 1.0, 2.0, 1.5]})
    by_name = gw.KaplanMeier().fit(Outcome.surv(time="t", event="e == 2"), weights="w", data=df)
    by_formula = gw.KaplanMeier().fit("Surv(t, e == 2)", weights="w", data=df)
    direct = gw.KaplanMeier().fit(Surv(time=df["t"], event=df["e"] == 2), weights=df["w"])
    unweighted = gw.KaplanMeier().fit(Surv(time=df["t"], event=df["e"] == 2))

    np.testing.assert_allclose(by_name.survival_, direct.survival_, rtol=1e-12)
    np.testing.assert_allclose(by_formula.survival_, direct.survival_, rtol=1e-12)
    assert not np.allclose(by_name.survival_, unweighted.survival_)


def test_left_by_name() -> None:
    data = {"t": [5.0, 6.0, 4.0], "e": ["yes", "no", "yes"]}
    y = Outcome.surv(time="t", event="e == 'yes'", type="left").bind(data)

    assert y.type is CensoringType.LEFT
    np.testing.assert_array_equal(y.status, [1, 0, 1])


def test_counting_by_name() -> None:
    data = {"a": [0.0, 2.0, 1.0], "b": [5.0, 6.0, 4.0], "e": [2, 1, 2]}
    y = Outcome.surv(time="a", time2="b", event="e == 2").bind(data)

    _assert_same(y, Surv(time=[0, 2, 1], time2=[5, 6, 4], event=[1, 0, 1]))


def test_interval_by_name() -> None:
    data = {"lo": [1.0, 2.0, 3.0], "hi": [2.0, np.inf, 5.0]}
    y = Outcome.surv(time="lo", time2="hi", type="interval2").bind(data)

    _assert_same(y, Surv(time=[1, 2, 3], time2=[2, np.inf, 5], type="interval2"))


# -- multi-state with factor() ------------------------------------------------


def test_factor_matches_categorical_surv() -> None:
    data = {"t": [5, 6, 7, 8], "e": [1, 2, 0, 1]}
    y = Outcome.surv(time="t", event="factor(e, c(0, 1, 2), c('censor', 'relapse', 'death'))").bind(
        data
    )
    event = pd.Categorical.from_codes([1, 2, 0, 1], categories=["censor", "relapse", "death"])

    _assert_same(y, Surv(time=data["t"], event=event))
    assert y.states == ("relapse", "death")


def test_factor_accepts_r_ranges() -> None:
    data = {"t": [5, 6, 7, 8], "e": [1, 2, 0, 1]}
    spelled = Outcome.surv(time="t", event="factor(e, c(0, 1, 2), c('censor', 'a', 'b'))")
    ranged = Outcome.surv(time="t", event="factor(e, 0:2, c('censor', 'a', 'b'))")

    assert ranged == spelled
    _assert_same(ranged.bind(data), spelled.bind(data))


def test_factor_label_order_sets_the_state_order() -> None:
    # Labels follow the order of `levels`, independent of the raw values.
    y = Outcome.surv(time="t", event="factor(e, c(0, 2, 1), c('censor', 'relapse', 'death'))").bind(
        {"t": [5, 6, 7], "e": [2, 1, 0]}
    )

    assert y.states == ("relapse", "death")
    np.testing.assert_array_equal(y.status, [1, 2, 0])


def test_factor_string_levels_with_several_censoring_values() -> None:
    import polars as pl

    df = pl.DataFrame({"t": [5.0, 6.0, 7.0, 8.0], "out": ["rel", "dth", "alive", "lost"]})
    event = (
        "factor(out, c('alive', 'lost', 'rel', 'dth'), c('censor', 'censor', 'relapse', 'death'))"
    )
    y = Outcome.surv(time="t", event=event).bind(df)

    assert y.states == ("relapse", "death")
    np.testing.assert_array_equal(y.status, [1, 2, 0, 0])


def test_factor_repeated_labels_merge_levels() -> None:
    # As in R, two levels with the same label become one state.
    y = Outcome.surv(time="t", event="factor(e, 0:3, c('censor', 'a', 'b', 'b'))").bind(
        {"t": [1, 2, 3, 4], "e": [1, 2, 3, 0]}
    )

    assert y.states == ("a", "b")
    np.testing.assert_array_equal(y.status, [1, 2, 2, 0])


def test_factor_value_outside_levels_is_missing() -> None:
    # R's `factor()` gives NA for a value not in `levels`.
    y = Outcome.surv(time="t", event="factor(e, c(0, 1), c('censor', 'a'))").bind(
        {"t": [1, 2, 3], "e": [0, 1, 9]}
    )

    np.testing.assert_array_equal(y.status, [0, 1, np.nan])


def test_factor_without_levels_sorts_the_values() -> None:
    # R's default levels are the sorted distinct values, and the first means censored.
    y = Outcome.surv(time="t", event="factor(e)").bind({"t": [1, 2, 3], "e": ["x", "a", "b"]})

    assert y.states == ("b", "x")
    np.testing.assert_array_equal(y.status, [2, 0, 1])


def test_factor_level_label_length_mismatch_errors() -> None:
    o = Outcome.surv(time="t", event="factor(e, c(0, 1, 2), c('censor', 'a'))")

    with pytest.raises(ValueError, match="3 levels but 2 labels"):
        o.bind({"t": [1, 2], "e": [0, 1]})


@pytest.mark.parametrize("backend", ["pandas", "polars", "polars-lazy", "pyarrow", "duckdb"])
def test_factor_on_every_backend(backend: str) -> None:
    frame = pd.DataFrame({"t": [1.0, 2.0, 3.0, 4.0], "e": [0.0, 1.0, 2.0, None]})
    y = Outcome.surv(time="t", event="factor(e, 0:2, c('censor', 'a', 'b'))").bind(
        _as_backend(frame, backend)
    )

    assert y.states == ("a", "b")
    np.testing.assert_array_equal(y.status, [0, 1, 2, np.nan])


def test_duckdb_dotted_column_names() -> None:
    # SQL backends read `ph.ecog` as table `ph`, column `ecog` unless selected eagerly.
    duckdb = pytest.importorskip("duckdb")
    import pyarrow as pa

    rel = duckdb.from_arrow(pa.table({"fu.time": [5.0, 6.0, 4.0], "ph.status": [2, 1, 2]}))
    y = Outcome.surv(time="fu.time", event="ph.status == 2").bind(rel)

    np.testing.assert_array_equal(y.status, [1, 0, 1])


def test_factor_rejects_duplicated_levels_like_r() -> None:
    with pytest.raises(ValueError, match=r"factor level \[3\] is duplicated"):
        Outcome.surv(time="t", event="factor(e, c(0, 1, 1), c('censor', 'a', 'b'))").bind(
            {"t": [1.0, 2.0], "e": [0, 1]}
        )


def test_comparisons_coerce_numbers_and_strings_like_r() -> None:
    data = {"t": [5.0, 6.0, 7.0], "e": [2, 1, 2]}
    for event in ("e == '2'", "e %in% c('2')"):
        y = Outcome.surv(time="t", event=event).bind(data)
        np.testing.assert_array_equal(y.status, [1.0, 0.0, 1.0])
