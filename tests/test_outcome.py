"""Tests for `Outcome` (Tier 2) and formula responses (Tier 3).

An `Outcome` or a formula bound at `fit()` must give exactly the fit that the array form gives on
the same complete-case rows, on every backend.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest

import greenwood as gw
from greenwood import CensoringType, Outcome, Surv

DEATH = Outcome.surv(time="time", event="status == 2")


@pytest.fixture(scope="module")
def lung() -> Any:
    return gw.load_dataset("lung", backend="pandas")


def _array_cox(lung: Any, columns: list[str], **kwargs: Any) -> Any:
    """The reference fit: drop incomplete rows by hand, then use the array form."""
    frame = lung.dropna(subset=["time", "status", *columns, *kwargs.values()])
    y = Surv(time=frame["time"].to_numpy(), event=(frame["status"] == 2).to_numpy())
    labels = {k: frame[v].to_numpy() for k, v in kwargs.items()}
    return gw.CoxPH().fit(y, frame[columns], **labels)


def _frames(lung: Any) -> dict[str, Any]:
    import duckdb
    import polars as pl
    import pyarrow as pa

    table = pa.Table.from_pandas(lung, preserve_index=False)
    return {
        "pandas": lung,
        "polars": pl.from_pandas(lung),
        "polars-lazy": pl.from_pandas(lung).lazy(),
        "pyarrow": table,
        "duckdb": duckdb.from_arrow(table),
    }


# -- Outcome construction -----------------------------------------------------


def test_repr_round_trips_the_constructor() -> None:
    assert repr(DEATH) == "Outcome.surv(time='time', event='status == 2')"

    ms = Outcome.surv(
        time="t0", time2="t", event="factor(cause, c(0, 1, 2), c('censor', 'pcm', 'death'))"
    )
    odd = Outcome.surv(time="follow up", event="the status != 0", type="right")
    timed = Outcome.surv(time=gw.duration(start="a", end="b", unit="weeks"), event="e")
    namespace = {"Outcome": Outcome, "duration": gw.duration}

    for o in (DEATH, ms, odd, timed, Outcome.event_time(time="t", status="code")):
        assert eval(repr(o), namespace) == o


def test_values_are_rejected() -> None:
    with pytest.raises(TypeError, match="use `Surv\\(\\)` or `event_time\\(\\)` directly"):
        Outcome.surv([1, 2, 3], event="status")  # pyright: ignore[reportArgumentType]


def test_unknown_surv_type_errors() -> None:
    with pytest.raises(ValueError, match="`type` must be one of"):
        Outcome.surv(time="t", event="e", type="weird")


def test_list_values_are_frozen() -> None:
    o = Outcome.surv(time="time", event="status in [1, 2]")

    assert o == Outcome.surv(time="time", event="status %in% c(1, 2)")
    assert o == Outcome.surv(time="time", event="status in 1:2")

    hash(o)


def test_column_names_and_arguments() -> None:
    o = Outcome.surv(time="a", time2="b", event="e == 1")

    assert o.column_names == ("a", "b", "e")
    assert [arg for arg, _ in o.arguments] == ["time", "time2", "event"]
    assert o.kind == "surv"

    timed = Outcome.surv(time=gw.duration(start="enter", end="exit"), event="factor(cause)")

    assert timed.column_names == ("enter", "exit", "cause")


def test_second_argument_without_event_is_the_status() -> None:
    # R's `Surv(time, x)` reads `x` as the status for right-censored data.
    assert Outcome.surv(time="time", time2="status == 2") == DEATH
    assert Outcome.surv("time", "status == 2") == DEATH


# -- bind ---------------------------------------------------------------------


@pytest.mark.parametrize("backend", ["pandas", "polars", "polars-lazy", "pyarrow", "duckdb"])
def test_bind_matches_array_form(lung: Any, backend: str) -> None:
    y = DEATH.bind(_frames(lung)[backend])
    ref = Surv(time=lung["time"].to_numpy(), event=(lung["status"] == 2).to_numpy())

    np.testing.assert_array_equal(y.stop, ref.stop)
    np.testing.assert_array_equal(y.status, ref.status)


@pytest.mark.parametrize("backend", ["pyarrow", "duckdb"])
def test_bind_factor_on_arrow_and_duckdb(lung: Any, backend: str) -> None:
    # A comparison on a PyArrow column with `==` gives a single `False`. Expressions don't.
    o = Outcome.surv(time="time", event="factor(status, 1:2, c('censor', 'death'))")
    y = o.bind(_frames(lung)[backend])

    assert y.states == ("death",)
    np.testing.assert_array_equal(y.status, (lung["status"] == 2).to_numpy().astype(float))


def test_bind_keeps_missing_values() -> None:
    # As with `Surv()`, `bind()` keeps missing values. `fit()` drops those rows.
    y = Outcome.surv(time="t", event="e").bind({"t": [1.0, np.nan], "e": [1, 0]})

    assert y.n == 2
    np.testing.assert_array_equal(y.stop, [1.0, np.nan])


def test_bind_multistate_and_interval() -> None:
    o = Outcome.surv(time="t", event="factor(e, c('c', 'x', 'y'), c('censor', 'a', 'b'))")
    y = o.bind({"t": [1.0, 2.0, 3.0], "e": ["x", "y", "c"]})

    assert y.states == ("a", "b")
    np.testing.assert_array_equal(y.status, [1, 2, 0])

    iv = Outcome.surv(time="lo", time2="hi", type="interval2").bind(
        {"lo": [1.0, 2.0], "hi": [2.0, np.inf]}
    )

    assert iv.type is CensoringType.INTERVAL


def test_bind_event_time() -> None:
    data = {
        "time": [1.0, 2.0, 3.0, 4.0],
        "code": ["e", "r", "l", "i"],
        "upper": [np.nan, np.nan, np.nan, 5.0],
    }
    y = Outcome.event_time(time="time", status="code", time_max="upper").bind(data)
    ref = gw.as_surv(gw.event_time(time=data["time"], status=data["code"], time_max=data["upper"]))

    assert y.type is ref.type
    np.testing.assert_array_equal(y.status, ref.status)
    np.testing.assert_array_equal(y.stop, ref.stop)


# -- fit with an Outcome ------------------------------------------------------


@pytest.mark.parametrize("backend", ["pandas", "polars", "polars-lazy", "pyarrow", "duckdb"])
def test_cox_outcome_matches_array_form(lung: Any, backend: str) -> None:
    columns = ["age", "sex", "ph.ecog"]
    ref = _array_cox(lung, columns)
    cox = gw.CoxPH().fit(DEATH, columns, data=_frames(lung)[backend])

    np.testing.assert_allclose(cox.coef_, ref.coef_, rtol=1e-10)
    np.testing.assert_allclose(cox.std_error_, ref.std_error_, rtol=1e-10)


def test_cox_strata_by_name_drops_its_missing_rows(lung: Any) -> None:
    ref = _array_cox(lung, ["age", "sex"], strata="inst")
    cox = gw.CoxPH().fit(DEATH, ["age", "sex"], data=lung, strata="inst")

    np.testing.assert_allclose(cox.coef_, ref.coef_, rtol=1e-10)


def test_response_rows_with_missing_values_are_dropped() -> None:
    import polars as pl

    df = pl.DataFrame(
        {
            "time": [5.0, None, 3.0, 8.0, 2.0, 9.0],
            "status": [2, 2, 1, 2, 2, 1],
            "x": [1.0, 2.0, 0.5, 3.0, 1.5, 0.2],
        }
    )
    km = gw.KaplanMeier().fit(DEATH, data=df)
    ref = gw.KaplanMeier().fit(Surv(time=[5.0, 3.0, 8.0, 2.0, 9.0], event=[1, 0, 1, 1, 0]))

    np.testing.assert_allclose(km.survival_, ref.survival_)


def test_kaplan_meier_by_name(lung: Any) -> None:
    y = Surv(time=lung["time"].to_numpy(), event=(lung["status"] == 2).to_numpy())
    ref = gw.KaplanMeier().fit(y, by=lung["sex"].to_numpy())
    via_outcome = gw.KaplanMeier().fit(DEATH, by="sex", data=lung)
    via_surv = gw.KaplanMeier().fit(y, by="sex", data=lung)

    np.testing.assert_allclose(via_outcome.survival_, ref.survival_)
    np.testing.assert_allclose(via_surv.survival_, ref.survival_)


def test_arrays_alongside_outcome_are_filtered_too() -> None:
    import polars as pl

    df = pl.DataFrame({"time": [5.0, None, 3.0, 8.0], "status": [2, 2, 1, 2]})
    groups = np.array(["a", "b", "a", "b"])
    km = gw.KaplanMeier().fit(DEATH, by=groups, data=df)
    ref = gw.KaplanMeier().fit(Surv(time=[5.0, 3.0, 8.0], event=[1, 0, 1]), by=["a", "a", "b"])

    np.testing.assert_allclose(km.survival_, ref.survival_)


def test_misaligned_array_alongside_outcome_errors(lung: Any) -> None:
    with pytest.raises(ValueError, match="must line up"):
        gw.KaplanMeier().fit(DEATH, by=np.zeros(3), data=lung)


def test_outcome_without_data_errors() -> None:
    with pytest.raises(ValueError, match="Pass `data=`"):
        gw.KaplanMeier().fit(DEATH)


def test_mapping_data_is_rejected_at_fit() -> None:
    with pytest.raises(TypeError, match="needs a DataFrame"):
        gw.KaplanMeier().fit(DEATH, data={"time": [1.0], "status": [2]})


def test_column_name_without_data_errors() -> None:
    y = Surv(time=[1, 2, 3], event=[1, 0, 1])
    with pytest.raises(ValueError, match="no `data=` was given"):
        gw.KaplanMeier().fit(y, by="sex")


def test_regression_needs_covariates() -> None:
    with pytest.raises(ValueError, match="needs `covariates`"):
        gw.CoxPH().fit(DEATH, data={"time": [1.0]})


def test_bad_response_type_errors() -> None:
    with pytest.raises(TypeError, match="must be a `Surv`, an `EventTime`, an `Outcome`"):
        gw.KaplanMeier().fit(23)  # pyright: ignore[reportArgumentType]


# -- formulas -----------------------------------------------------------------


def test_from_formula_equals_the_constructor() -> None:
    assert Outcome.from_formula("Surv(time, status == 2)") == Outcome.surv(
        time="time", event="status == 2"
    )


def test_from_formula_forms() -> None:
    def surv(event: str) -> Outcome:
        return Outcome.surv(time="time", event=event)

    assert Outcome.from_formula("Surv(time, status == 2)") == DEATH
    assert Outcome.from_formula("Surv(time, status != 0)") == surv("status != 0")
    assert Outcome.from_formula("Surv(time, status in (1, 2))") == surv("status in (1, 2)")
    assert Outcome.from_formula("Surv(time, status %in% c(1, 2))") == surv("status in (1, 2)")
    assert Outcome.from_formula("Surv(time, status %in% 1:2)") == surv("status in (1, 2)")
    assert Outcome.from_formula("Surv(time, status not in [0])") == surv("status not in (0,)")
    assert Outcome.from_formula("Surv(time, outcome == 'died')") == surv("outcome == 'died'")
    assert Outcome.from_formula("Surv(time, event=status == 2)") == DEATH
    assert Outcome.from_formula("Surv(time)") == Outcome.surv(time="time")
    assert Outcome.from_formula("Surv(time, status == 2) ~ 1") == DEATH


def test_from_formula_other_types() -> None:
    assert Outcome.from_formula("Surv(a, b, e == 1)") == Outcome.surv(
        time="a", time2="b", event="e == 1"
    )
    assert Outcome.from_formula('Surv(t, e, type="left")') == Outcome.surv(
        time="t", event="e", type="left"
    )
    assert Outcome.from_formula('Surv(lo, hi, type="interval2")') == Outcome.surv(
        time="lo", time2="hi", type="interval2"
    )
    assert Outcome.from_formula("Surv(time, status, origin=10)").origin == 10.0

    ms = Outcome.from_formula("Surv(t, factor(cause, 0:2, c('censor', 'pcm', 'death')))")

    assert ms == Outcome.surv(
        time="t", event="factor(cause, c(0, 1, 2), c('censor', 'pcm', 'death'))"
    )


def test_from_formula_event_time() -> None:
    parsed = Outcome.from_formula("event_time(time, code, upper)")

    assert parsed == Outcome.event_time(time="time", status="code", time_max="upper")
    assert parsed.kind == "event_time"
    assert Outcome.from_formula("event_time(time, status=code)") == Outcome.event_time(
        time="time", status="code"
    )


def test_event_time_formula_fits(lung: Any) -> None:
    import polars as pl

    df = pl.DataFrame({"time": [5.0, 3.0, 8.0, 2.0, 9.0], "code": ["e", "r", "e", "e", "r"]})
    km = gw.KaplanMeier().fit("event_time(time, code)", data=df)
    ref = gw.KaplanMeier().fit(Surv(time=[5.0, 3.0, 8.0, 2.0, 9.0], event=[1, 0, 1, 1, 0]))

    np.testing.assert_allclose(km.survival_, ref.survival_)


def test_from_formula_names() -> None:
    assert Outcome.from_formula("Surv(`follow up`, `the status` == 2)").column_names == (
        "follow up",
        "the status",
    )
    assert Outcome.from_formula("Surv(fu.time, dead)").column_names == ("fu.time", "dead")
    assert Outcome.from_formula("Surv(t0, t1, factor(cause))").column_names == (
        "t0",
        "t1",
        "cause",
    )


@pytest.mark.parametrize(
    ("formula", "message"),
    [
        ("Surv(time, status > 1)", "is not supported"),
        ("Surv(time, 1 < status < 3)", "Chained comparisons"),
        ("Surv(time, status ==)", "Could not parse"),
        ("surv(time, status)", "call to `Surv"),
        ("time + status", "call to `Surv"),
        ("Surv(time, status, foo=1)", "Unknown or repeated `Surv\\(\\)` argument `foo`"),
        ("Surv(time, status, event_value=2)", "argument `event_value`"),
        ("Surv(time, status, censor_value=0)", "argument `censor_value`"),
        ("Surv(t, e, states={'a': 1, 'b': 2})", "argument `states`"),
        ("Surv(time, status, weights=wt)", "argument `weights`"),
        ("Surv(time, status == x)", "literal value"),
        ("Surv(a, b, c, d)", "at most 3 positional"),
        ("Surv(type='left')", "needs a `time`"),
        ("event_time(time)", "needs `time` and `status`"),
        ("event_time(time, code, type='left')", "argument `type`"),
        ("Surv(time, factor(status, c(0, 1), c('a', 'b'), ordered=TRUE))", "factor\\(\\)` takes"),
        ('Surv(t, e, type="weird")', "`type` must be one of"),
        ("~ age", "has no response"),
        ("Surv(time, status == 2) ~ age", "takes only a response"),
        ("Surv(time, status > 1)", "not supported in the status"),
        ("Surv(time, duration(a, b))", "can't be a `duration\\(\\)`"),
        ("Surv(time, __import__('os'))", "column name"),
    ],
)
def test_from_formula_errors(formula: str, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        Outcome.from_formula(formula)


def test_fit_with_removed_keyword_errors(lung: Any) -> None:
    with pytest.raises(ValueError, match="argument `event_value`"):
        gw.KaplanMeier().fit("Surv(time, status, event_value=2) ~ sex", data=lung)


def test_cox_full_formula_matches_array_form(lung: Any) -> None:
    ref = _array_cox(lung, ["age", "sex"])
    cox = gw.CoxPH().fit("Surv(time, status == 2) ~ age + sex", data=lung)

    np.testing.assert_allclose(cox.coef_, ref.coef_, rtol=1e-10)


def test_cox_formula_with_dotted_names(lung: Any) -> None:
    ref = _array_cox(lung, ["age", "ph.ecog"])
    cox = gw.CoxPH().fit("Surv(time, status == 2) ~ age + ph.ecog", data=lung)

    np.testing.assert_allclose(cox.coef_, ref.coef_, rtol=1e-10)


def test_formula_on_duckdb(lung: Any) -> None:
    ref = _array_cox(lung, ["age", "sex"])
    cox = gw.CoxPH().fit("Surv(time, status == 2) ~ age + sex", data=_frames(lung)["duckdb"])

    np.testing.assert_allclose(cox.coef_, ref.coef_, rtol=1e-10)


def test_outcome_with_formula_covariates(lung: Any) -> None:
    ref = _array_cox(lung, ["age", "sex"])
    cox = gw.CoxPH().fit(DEATH, "age + sex", data=lung)

    np.testing.assert_allclose(cox.coef_, ref.coef_, rtol=1e-10)


def test_covariates_twice_errors(lung: Any) -> None:
    with pytest.raises(ValueError, match="both in the formula"):
        gw.CoxPH().fit("Surv(time, status == 2) ~ age", ["sex"], data=lung)


def test_kaplan_meier_formula_groups(lung: Any) -> None:
    km_one = gw.KaplanMeier().fit("Surv(time, status == 2) ~ 1", data=lung)

    assert not km_one._grouped  # pyright: ignore[reportPrivateUsage]

    km_sex = gw.KaplanMeier().fit("Surv(time, status == 2) ~ sex", data=lung)
    ref = gw.KaplanMeier().fit(DEATH, by="sex", data=lung)

    np.testing.assert_allclose(km_sex.survival_, ref.survival_)


def test_kaplan_meier_formula_combines_several_groups(lung: Any) -> None:
    km = gw.KaplanMeier().fit("Surv(time, status == 2) ~ sex + ph.ecog", data=lung)
    labels = set(np.asarray(km.strata_).tolist())

    assert "sex=1, ph.ecog=0.0" in labels


def test_kaplan_meier_formula_rejects_expressions(lung: Any) -> None:
    with pytest.raises(ValueError, match="grouping columns only"):
        gw.KaplanMeier().fit("Surv(time, status == 2) ~ C(sex)", data=lung)


def test_univariate_rejects_covariates(lung: Any) -> None:
    with pytest.raises(ValueError, match="takes no covariates"):
        gw.Parametric("weibull").fit("Surv(time, status == 2) ~ age", data=lung)
    fitted = gw.Parametric("weibull").fit("Surv(time, status == 2)", data=lung)
    ref = gw.Parametric("weibull").fit(DEATH.bind(lung))

    assert fitted.params_ == pytest.approx(ref.params_)


def test_competing_risks_formula() -> None:
    mg = gw.load_dataset("mgus2", backend="polars")
    import polars as pl

    mg = mg.with_columns(
        etime=pl.when(pl.col("pstat") == 1).then(pl.col("ptime")).otherwise(pl.col("futime")),
        cause=pl.when(pl.col("pstat") == 1)
        .then(pl.lit("pcm"))
        .when(pl.col("death") == 1)
        .then(pl.lit("death"))
        .otherwise(pl.lit("censor")),
    )
    formula = "Surv(etime, factor(cause, c('censor', 'pcm', 'death')))"
    aj = gw.AalenJohansen().fit(formula, data=mg)
    y = Surv(
        time=mg["etime"],
        event=pd.Categorical(mg["cause"].to_list(), categories=["censor", "pcm", "death"]),
    )
    ref = gw.AalenJohansen().fit(y)
    np.testing.assert_allclose(aj.to_frame()["estimate"], ref.to_frame()["estimate"])

    fg = gw.FineGray("pcm").fit(formula + " ~ age + sex", data=mg)
    fg_ref = gw.FineGray("pcm").fit(
        Surv(
            time=mg.drop_nulls(["age", "sex"])["etime"],
            event=pd.Categorical(
                mg.drop_nulls(["age", "sex"])["cause"].to_list(),
                categories=["censor", "pcm", "death"],
            ),
        ),
        mg.drop_nulls(["age", "sex"]).select(["age", "sex"]),
    )

    np.testing.assert_allclose(fg.coef_, fg_ref.coef_, rtol=1e-10)
