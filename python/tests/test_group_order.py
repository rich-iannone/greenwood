"""The group order rule (spec/results.md).

Groups come in the order of their values, never the order of the rows: a pandas `Categorical` or a
Polars `Enum` keeps its declared levels, and other values are sorted, numbers in numeric order and
text in code-point order. With several grouping columns the first varies slowest. Every estimator
that fits one curve per group, and every test that compares groups, uses the same order.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import polars as pl
import pyarrow as pa
import pytest

import greenwood as gw
from greenwood import KaplanMeier, Surv

LEVELS = ["squamous", "smallcell", "adeno", "large"]
CELLTYPE = [f"celltype={level}" for level in LEVELS]
SORTED_CELLTYPE = sorted(CELLTYPE)


@pytest.fixture(scope="module")
def veteran() -> pd.DataFrame:
    """The cell type as plain text, so its values are sorted."""
    out = gw.load_dataset("veteran", backend="pandas")
    out["celltype"] = out["celltype"].astype(object)
    return out


@pytest.fixture(scope="module")
def veteran_cat() -> pd.DataFrame:
    """The cell type as loaded, a categorical with R's levels."""
    out = gw.load_dataset("veteran", backend="pandas")
    assert list(out["celltype"].cat.categories) == LEVELS
    return out


def _km_strata(fit: KaplanMeier) -> list[str]:
    return fit.median(format="pandas")["strata"].tolist()


def test_numbers_sort_numerically() -> None:
    y = Surv(time=[1, 2, 3, 4, 5, 6], event=[1, 1, 1, 1, 1, 1])
    km = KaplanMeier().fit(y, by=[10, 2, 1, 10, 2, 1])
    assert _km_strata(km) == ["by=1", "by=2", "by=10"]
    km = KaplanMeier().fit(y, by=[0.5, 10.0, 2.5, 0.5, 10.0, 2.5])
    assert _km_strata(km) == ["by=0.5", "by=2.5", "by=10"]


def test_text_sorts_by_code_point() -> None:
    y = Surv(time=[1, 2, 3, 4], event=[1, 1, 1, 1])
    km = KaplanMeier().fit(y, by=["a", "B", "a", "B"])
    assert _km_strata(km) == ["by=B", "by=a"]


def test_row_order_does_not_change_the_result(veteran: pd.DataFrame) -> None:
    shuffled = veteran.sample(frac=1.0, random_state=3).reset_index(drop=True)
    formula = "Surv(time, status) ~ celltype"
    original = KaplanMeier().fit(formula, data=veteran).to_frame(format="pandas")
    reordered = KaplanMeier().fit(formula, data=shuffled).to_frame(format="pandas")
    pd.testing.assert_frame_equal(original, reordered)
    assert list(dict.fromkeys(original["strata"])) == SORTED_CELLTYPE


def test_pandas_categorical_keeps_its_levels(veteran_cat: pd.DataFrame) -> None:
    formula = "Surv(time, status) ~ celltype"
    assert _km_strata(KaplanMeier().fit(formula, data=veteran_cat)) == CELLTYPE
    y = Surv(time=veteran_cat["time"], event=veteran_cat["status"])
    assert _km_strata(KaplanMeier().fit(y, by="celltype", data=veteran_cat)) == CELLTYPE
    assert _km_strata(KaplanMeier().fit(y, by=veteran_cat["celltype"])) == CELLTYPE
    assert _km_strata(KaplanMeier().fit(y, by=veteran_cat["celltype"].array)) == [
        f"by={level}" for level in LEVELS
    ]


def test_categorical_levels_that_do_not_occur_are_skipped(veteran: pd.DataFrame) -> None:
    data = veteran.copy()
    data["celltype"] = pd.Categorical(data["celltype"], categories=["other", *LEVELS])
    assert _km_strata(KaplanMeier().fit("Surv(time, status) ~ celltype", data=data)) == CELLTYPE


def test_polars_enum_keeps_its_levels(veteran: pd.DataFrame) -> None:
    data = pl.from_pandas(veteran).with_columns(pl.col("celltype").cast(pl.Enum(LEVELS)))
    assert _km_strata(KaplanMeier().fit("Surv(time, status) ~ celltype", data=data)) == CELLTYPE


def test_polars_categorical_and_arrow_dictionary_sort_by_value(veteran: pd.DataFrame) -> None:
    # Both libraries sort these by value, so their values are sorted here too.
    formula = "Surv(time, status) ~ celltype"
    data = pl.from_pandas(veteran).with_columns(pl.col("celltype").cast(pl.Categorical))
    assert _km_strata(KaplanMeier().fit(formula, data=data)) == SORTED_CELLTYPE
    table = pa.Table.from_pandas(veteran, preserve_index=False)
    index = table.schema.get_field_index("celltype")
    table = table.set_column(index, "celltype", table["celltype"].dictionary_encode())
    assert _km_strata(KaplanMeier().fit(formula, data=table)) == SORTED_CELLTYPE


def test_several_columns_vary_the_first_slowest() -> None:
    data = pd.DataFrame(
        {
            "time": np.arange(1.0, 9.0),
            "status": np.ones(8),
            "dose": [10, 2, 10, 2, 10, 2, 10, 2],
            "arm": ["b", "b", "a", "a", "b", "b", "a", "a"],
        }
    )
    km = KaplanMeier().fit("Surv(time, status) ~ dose + arm", data=data)
    assert _km_strata(km) == ["dose=2, arm=a", "dose=2, arm=b", "dose=10, arm=a", "dose=10, arm=b"]

    data["arm"] = pd.Categorical(data["arm"], categories=["b", "a"])
    km = KaplanMeier().fit("Surv(time, status) ~ arm + dose", data=data)
    assert _km_strata(km) == ["arm=b, dose=2", "arm=b, dose=10", "arm=a, dose=2", "arm=a, dose=10"]


def test_every_curve_estimator_uses_the_same_order(veteran_cat: pd.DataFrame) -> None:
    formula = "Surv(time, status) ~ celltype"
    na = gw.NelsonAalen().fit(formula, data=veteran_cat).to_frame(format="pandas")
    assert list(dict.fromkeys(na["strata"])) == CELLTYPE
    et = gw.event_table(formula, data=veteran_cat).to_frame(format="pandas")
    assert list(dict.fromkeys(et["strata"])) == CELLTYPE

    cox = gw.CoxPH().fit("Surv(time, status) ~ karno + strata(celltype)", data=veteran_cat)
    assert list(dict.fromkeys(cox.baseline_hazard(format="pandas")["strata"])) == CELLTYPE
    plain = gw.CoxPH().fit(
        "Surv(time, status) ~ karno + strata(celltype)",
        data=veteran_cat.assign(celltype=veteran_cat["celltype"].astype(str)),
    )
    np.testing.assert_allclose(cox.coef_, plain.coef_, rtol=1e-12)

    event = veteran_cat["status"] == 1
    interval = veteran_cat.assign(
        lower=np.where(event, veteran_cat["time"] * 0.8, veteran_cat["time"]),
        upper=np.where(event, veteran_cat["time"], np.inf),
    )
    tb = (
        gw.Turnbull()
        .fit("Surv(lower, upper, type='interval2') ~ celltype", data=interval)
        .to_frame(format="pandas")
    )
    assert list(dict.fromkeys(tb["strata"])) == CELLTYPE


def test_aalen_johansen_uses_the_same_order() -> None:
    mgus2 = gw.load_dataset("mgus2", backend="pandas")
    mgus2["sex"] = pd.Categorical(mgus2["sex"], categories=["M", "F"])
    outcome = gw.Outcome.first_event(
        endpoints={"pcm": ("ptime", "pstat"), "death": ("futime", "death")}
    )
    aj = gw.AalenJohansen().fit(outcome, by="sex", data=mgus2).to_frame(format="pandas")
    assert list(dict.fromkeys(aj["strata"])) == ["sex=M", "sex=F"]


def test_comparisons_use_the_group_order(veteran: pd.DataFrame, veteran_cat: pd.DataFrame) -> None:
    formula = "Surv(time, status) ~ celltype"
    by_level = gw.logrank_test(formula, data=veteran_cat)
    by_value = gw.logrank_test(formula, data=veteran)
    assert list(by_level.observed) == LEVELS
    assert list(by_value.observed) == sorted(LEVELS)
    assert by_level.observed == by_value.observed
    assert by_level.statistic == pytest.approx(by_value.statistic, rel=1e-12)

    # The default trend scores are 0, 1, 2, 3 in group order.
    trend = gw.trend_test(formula, data=veteran_cat)
    explicit = gw.trend_test(
        formula, scores={level: i for i, level in enumerate(LEVELS)}, data=veteran
    )
    assert trend.statistic == pytest.approx(explicit.statistic, rel=1e-12)


def test_two_group_differences_take_the_first_group_first() -> None:
    lung = gw.load_dataset("lung", backend="pandas")
    reversed_sex = lung.assign(sex=pd.Categorical(lung["sex"], categories=[2, 1]))
    formula = "Surv(time, status) ~ sex"

    forward = gw.rmst_test(formula, tau=365, data=lung)
    backward = gw.rmst_test(formula, tau=365, data=reversed_sex)
    assert (forward.group1, forward.group2) == (1, 2)
    assert (backward.group1, backward.group2) == (2, 1)
    assert backward.estimate == pytest.approx(-forward.estimate, rel=1e-12)

    forward = gw.bootstrap(formula, statistic="median_diff", n_boot=20, seed=1, data=lung)
    backward = gw.bootstrap(formula, statistic="median_diff", n_boot=20, seed=1, data=reversed_sex)
    assert forward.estimate == pytest.approx(270.0 - 426.0)
    assert backward.estimate == pytest.approx(426.0 - 270.0)
    assert np.nanmean(backward.distribution) > 0 > np.nanmean(forward.distribution)
