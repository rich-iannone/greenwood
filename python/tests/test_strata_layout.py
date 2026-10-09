"""The one-curve / many-curve layout rule (spec/results.md).

A fit's tables have the same columns, in the same order and with the same dtypes, whether it
has one curve or many. Every curve-level and time-level table starts with a string `strata`
column. A single curve is labeled "all" and grouped curves are labeled "name=value".
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import polars as pl
import pytest

import greenwood as gw
from greenwood import AalenJohansen, KaplanMeier, NelsonAalen, Surv, Turnbull

DEATH = gw.Outcome.surv(time="time", event="status")
MGUS_CR = gw.Outcome.first_event(
    endpoints={"pcm": ("ptime", "pstat"), "death": ("futime", "death")}
)


@pytest.fixture(scope="module")
def lung() -> pl.DataFrame:
    return gw.load_dataset("lung", backend="polars")


@pytest.fixture(scope="module")
def mgus2() -> pl.DataFrame:
    return gw.load_dataset("mgus2", backend="polars")


def _interval_data() -> tuple[Surv, np.ndarray]:
    y = Surv(
        time=[0, 4, 7, 0, 3, 5, 1, 2, 0, 6, 3, 0],
        time2=[4, float("inf"), 7, 2.5, 6, 5, 3, 2, 5, 9, float("inf"), 4],
        type="interval2",
    )
    sex = np.array([1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2])
    return y, sex


def _assert_same_layout(one: pl.DataFrame, many: pl.DataFrame) -> None:
    assert one.columns[0] == "strata"
    assert one.schema == many.schema
    assert one.schema["strata"] == pl.String
    assert set(one["strata"].to_list()) == {"all"}


# -- Kaplan-Meier ------------------------------------------------------------------------

KM_SUMMARIES: dict[str, Callable[[KaplanMeier], Any]] = {
    "to_frame": lambda m: m.to_frame(format="polars"),
    "glance": lambda m: gw.glance(m, format="polars"),
    "tidy": lambda m: gw.tidy(m, format="polars"),
    "median": lambda m: m.median(format="polars"),
    "quantile": lambda m: m.quantile([0.25, 0.5], format="polars"),
    "rmst": lambda m: m.rmst(365.0, format="polars"),
    "rmrl": lambda m: m.rmrl(100.0, 365.0, format="polars"),
    "predict": lambda m: m.predict([100.0, 365.0], format="polars"),
    "predict_cumhaz": lambda m: m.predict([100.0, 365.0], what="cumhaz", format="polars"),
}


@pytest.mark.parametrize("summary", list(KM_SUMMARIES))
def test_km_one_and_two_curves_share_layout(lung: pl.DataFrame, summary: str) -> None:
    one = KM_SUMMARIES[summary](KaplanMeier().fit(DEATH, data=lung))
    many = KM_SUMMARIES[summary](KaplanMeier().fit(DEATH, data=lung, by="sex"))

    _assert_same_layout(one, many)
    assert list(dict.fromkeys(many["strata"].to_list())) == ["sex=1", "sex=2"]


def test_km_summaries_have_one_row_per_curve_and_argument(lung: pl.DataFrame) -> None:
    km = KaplanMeier().fit(DEATH, data=lung, by="sex")

    assert km.median(format="polars").height == 2
    assert km.quantile([0.25, 0.5, 0.75], format="polars").height == 6
    assert km.predict([100.0, 200.0, 300.0], format="polars").height == 6
    assert km.rmst(365.0, format="polars").height == 2
    assert gw.glance(km, format="polars").height == 2


def test_km_summary_columns(lung: pl.DataFrame) -> None:
    km = KaplanMeier().fit(DEATH, data=lung)

    assert km.median(format="polars").columns == ["strata", "time", "conf_low", "conf_high"]
    assert km.quantile(0.5, format="polars").columns == [
        "strata",
        "prob",
        "time",
        "conf_low",
        "conf_high",
    ]
    assert km.rmst(365.0, format="polars").columns == [
        "strata",
        "tau",
        "estimate",
        "std_error",
        "conf_low",
        "conf_high",
    ]
    assert km.rmrl(100.0, 365.0, format="polars").columns == [
        "strata",
        "s",
        "tau",
        "estimate",
        "std_error",
        "conf_low",
        "conf_high",
    ]
    assert km.predict([100.0], format="polars").columns == ["strata", "time", "estimate"]


def test_km_strata_property_is_always_an_array(lung: pl.DataFrame) -> None:
    km = KaplanMeier().fit(DEATH, data=lung)

    assert km.strata_ is not None
    assert set(np.asarray(km.strata_).tolist()) == {"all"}


def test_km_formula_label_matches_by(lung: pl.DataFrame) -> None:
    formula = KaplanMeier().fit("Surv(time, status) ~ sex", data=lung)
    by_name = KaplanMeier().fit(DEATH, data=lung, by="sex")

    assert formula.median(format="polars")["strata"].to_list() == ["sex=1", "sex=2"]
    assert formula.median(format="polars").equals(by_name.median(format="polars"))


def test_km_formula_two_variables_labels(lung: pl.DataFrame) -> None:
    km = KaplanMeier().fit("Surv(time, status) ~ sex + ph.ecog", data=lung)
    labels = km.median(format="polars")["strata"].to_list()

    assert "sex=1, ph.ecog=0" in labels
    assert "sex=2, ph.ecog=1" in labels
    # Integer-valued floats print without a trailing ".0"
    assert not any(".0" in label for label in labels)


def test_km_unnamed_numpy_by_uses_argument_name(lung: pl.DataFrame) -> None:
    y = Surv(time=lung["time"], event=lung["status"] == 2)
    km = KaplanMeier().fit(y, by=lung["sex"].to_numpy())

    assert km.median(format="polars")["strata"].to_list() == ["by=1", "by=2"]


def test_km_named_series_uses_series_name(lung: pl.DataFrame) -> None:
    y = Surv(time=lung["time"], event=lung["status"] == 2)

    polars_km = KaplanMeier().fit(y, by=lung["sex"].alias("gender"))
    assert polars_km.median(format="polars")["strata"].to_list() == ["gender=1", "gender=2"]

    pandas_sex = lung.to_pandas()["sex"]
    pandas_km = KaplanMeier().fit(y, by=pandas_sex)
    assert pandas_km.median(format="polars")["strata"].to_list() == ["sex=1", "sex=2"]


def test_km_bool_group_prints_r_style() -> None:
    y = Surv(time=[1, 2, 3, 4], event=[1, 1, 1, 1])
    km = KaplanMeier().fit(y, by=np.array([True, True, False, False]))

    assert km.median(format="polars")["strata"].to_list() == ["by=TRUE", "by=FALSE"]


def test_km_group_order_is_first_appearance() -> None:
    y = Surv(time=[1, 2, 3, 4], event=[1, 1, 1, 1])
    km = KaplanMeier().fit(y, by=["b", "a", "b", "a"])

    assert km.median(format="polars")["strata"].to_list() == ["by=b", "by=a"]


def test_km_summary_formats_agree(lung: pl.DataFrame) -> None:
    km = KaplanMeier().fit(DEATH, data=lung, by="sex")
    pl_med = km.median(format="polars")
    pd_med = km.median(format="pandas")
    pa_med = km.median(format="pyarrow")

    assert list(pd_med.columns) == pl_med.columns
    assert pa_med.column_names == pl_med.columns
    assert pd_med["strata"].tolist() == pl_med["strata"].to_list()


# -- Nelson-Aalen ------------------------------------------------------------------------

NA_SUMMARIES: dict[str, Callable[[NelsonAalen], Any]] = {
    "to_frame": lambda m: m.to_frame(format="polars"),
    "glance": lambda m: gw.glance(m, format="polars"),
    "tidy": lambda m: gw.tidy(m, format="polars"),
}


@pytest.mark.parametrize("summary", list(NA_SUMMARIES))
def test_na_one_and_two_curves_share_layout(lung: pl.DataFrame, summary: str) -> None:
    one = NA_SUMMARIES[summary](NelsonAalen().fit(DEATH, data=lung))
    many = NA_SUMMARIES[summary](NelsonAalen().fit(DEATH, data=lung, by="sex"))

    _assert_same_layout(one, many)
    assert list(dict.fromkeys(many["strata"].to_list())) == ["sex=1", "sex=2"]


# -- Turnbull ----------------------------------------------------------------------------

TB_SUMMARIES: dict[str, Callable[[Turnbull], Any]] = {
    "to_frame": lambda m: m.to_frame(format="polars"),
    "glance": lambda m: gw.glance(m, format="polars"),
    "tidy": lambda m: gw.tidy(m, format="polars"),
    "median": lambda m: m.median(format="polars"),
    "quantile": lambda m: m.quantile([0.25, 0.5], format="polars"),
    "rmst": lambda m: m.rmst(5.0, format="polars"),
    "rmrl": lambda m: m.rmrl(1.0, 5.0, format="polars"),
    "predict": lambda m: m.predict([1.0, 3.0, 6.0], format="polars"),
}


@pytest.mark.parametrize("summary", list(TB_SUMMARIES))
def test_turnbull_one_and_two_curves_share_layout(summary: str) -> None:
    y, sex = _interval_data()
    one = TB_SUMMARIES[summary](Turnbull().fit(y))
    many = TB_SUMMARIES[summary](Turnbull().fit(y, by=sex))

    _assert_same_layout(one, many)
    assert list(dict.fromkeys(many["strata"].to_list())) == ["by=1", "by=2"]


def test_turnbull_summary_columns() -> None:
    y, _ = _interval_data()
    tb = Turnbull().fit(y)

    assert tb.median(format="polars").columns == ["strata", "time", "time_low", "time_high"]
    assert tb.quantile(0.5, format="polars").columns == [
        "strata",
        "prob",
        "time",
        "time_low",
        "time_high",
    ]
    assert tb.rmst(5.0, format="polars").columns == ["strata", "tau", "estimate"]
    assert tb.rmrl(1.0, 5.0, format="polars").columns == ["strata", "s", "tau", "estimate"]
    assert tb.predict([1.0], format="polars").columns == ["strata", "time", "estimate"]


def test_turnbull_by_column_name_label(lung: pl.DataFrame) -> None:
    tb = Turnbull().fit(DEATH, data=lung, by="sex")

    assert tb.median(format="polars")["strata"].to_list() == ["sex=1", "sex=2"]
    assert set(np.asarray(tb.strata_).tolist()) == {"sex=1", "sex=2"}


# -- Aalen-Johansen ----------------------------------------------------------------------

AJ_SUMMARIES: dict[str, Callable[[AalenJohansen], Any]] = {
    "to_frame": lambda m: m.to_frame(format="polars"),
    "glance": lambda m: gw.glance(m, format="polars"),
    "tidy": lambda m: gw.tidy(m, format="polars"),
}


@pytest.mark.parametrize("summary", list(AJ_SUMMARIES))
def test_aj_one_and_two_curves_share_layout(mgus2: pl.DataFrame, summary: str) -> None:
    one = AJ_SUMMARIES[summary](AalenJohansen().fit(MGUS_CR, data=mgus2))
    many = AJ_SUMMARIES[summary](AalenJohansen().fit(MGUS_CR, data=mgus2, by="sex"))

    _assert_same_layout(one, many)
    assert set(many["strata"].to_list()) == {"sex=F", "sex=M"}


def test_aj_blocks_keyed_by_label(mgus2: pl.DataFrame) -> None:
    one = AalenJohansen().fit(MGUS_CR, data=mgus2)
    many = AalenJohansen().fit(MGUS_CR, data=mgus2, by="sex")

    assert list(one._blocks) == ["all"]  # pyright: ignore[reportPrivateUsage]
    assert set(many._blocks) == {"sex=F", "sex=M"}  # pyright: ignore[reportPrivateUsage]


# -- event_table -------------------------------------------------------------------------


def test_event_table_one_and_two_groups_share_layout() -> None:
    y = Surv(time=[5, 4, 6, 4, 7, 3], event=[1, 1, 0, 1, 1, 0])
    one = gw.event_table(y).to_frame(format="polars")
    many = gw.event_table(y, group=[1, 2, 1, 2, 1, 2]).to_frame(format="polars")

    _assert_same_layout(one, many)
    assert list(dict.fromkeys(many["strata"].to_list())) == ["group=1", "group=2"]
