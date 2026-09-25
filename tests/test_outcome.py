"""Tests for `Outcome` (Tier 2) and formula responses (Tier 3).

An `Outcome` or a formula bound at `fit()` must give exactly the fit that the array form gives on
the same complete-case rows, on every backend.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

import greenwood as gw
from greenwood import CensoringType, Outcome, Surv

DEATH = Outcome.right("time", "status", event_value=2)


@pytest.fixture(scope="module")
def lung() -> Any:
    return gw.load_dataset("lung", backend="pandas")


def _array_cox(lung: Any, columns: list[str], **kwargs: Any) -> Any:
    """The reference fit: drop incomplete rows by hand, then use the array form."""
    frame = lung.dropna(subset=["time", "status", *columns, *kwargs.values()])
    y = Surv.right(frame["time"].to_numpy(), (frame["status"] == 2).to_numpy())
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
    assert repr(DEATH) == "Outcome.right(time='time', event='status', event_value=2)"

    ms = Outcome.multistate("t", "cause", states={"pcm": 1, "death": 2}, start="t0")

    assert repr(ms) == (
        "Outcome.multistate(time='t', event='cause', states={'pcm': 1, 'death': 2}, start='t0')"
    )


def test_values_are_rejected() -> None:
    with pytest.raises(TypeError, match="use `Surv` directly"):
        Outcome.right([1, 2, 3], "status")  # pyright: ignore[reportArgumentType]


def test_encoding_needs_event_column() -> None:
    with pytest.raises(ValueError, match="need an `event` column"):
        Outcome.right("time", event_value=2)


def test_list_values_are_frozen() -> None:
    o = Outcome.right("time", "status", event_value=[1, 2])

    assert o.event_value == (1, 2)

    hash(o)


def test_column_names() -> None:
    o = Outcome.counting("a", "b", "e", weights="w")

    assert o.column_names == ("a", "b", "e", "w")


# -- bind ---------------------------------------------------------------------


@pytest.mark.parametrize("backend", ["pandas", "polars", "polars-lazy", "pyarrow", "duckdb"])
def test_bind_matches_array_form(lung: Any, backend: str) -> None:
    y = DEATH.bind(_frames(lung)[backend])
    ref = Surv.right(lung["time"].to_numpy(), (lung["status"] == 2).to_numpy())

    np.testing.assert_array_equal(y.stop, ref.stop)
    np.testing.assert_array_equal(y.status, ref.status)


def test_bind_is_strict_about_missing_values() -> None:
    with pytest.raises(ValueError, match="finite"):
        Outcome.right("t", "e").bind({"t": [1.0, np.nan], "e": [1, 0]})


def test_bind_multistate_and_interval() -> None:
    y = Outcome.multistate("t", "e", states={"a": "x", "b": "y"}, censor_value="c").bind(
        {"t": [1.0, 2.0, 3.0], "e": ["x", "y", "c"]}
    )

    assert y.states == ("a", "b")
    np.testing.assert_array_equal(y.status, [1, 2, 0])

    iv = Outcome.interval("lo", "hi").bind({"lo": [1.0, 2.0], "hi": [2.0, np.inf]})

    assert iv.type is CensoringType.INTERVAL


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
    ref = gw.KaplanMeier().fit(Surv.right([5.0, 3.0, 8.0, 2.0, 9.0], [1, 0, 1, 1, 0]))

    np.testing.assert_allclose(km.survival_, ref.survival_)


def test_kaplan_meier_by_name(lung: Any) -> None:
    y = Surv.right(lung["time"].to_numpy(), (lung["status"] == 2).to_numpy())
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
    ref = gw.KaplanMeier().fit(Surv.right([5.0, 3.0, 8.0], [1, 0, 1]), by=["a", "a", "b"])

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
    y = Surv.right([1, 2, 3], [1, 0, 1])
    with pytest.raises(ValueError, match="no `data=` was given"):
        gw.KaplanMeier().fit(y, by="sex")


def test_regression_needs_covariates() -> None:
    with pytest.raises(ValueError, match="needs `covariates`"):
        gw.CoxPH().fit(DEATH, data={"time": [1.0]})


def test_bad_response_type_errors() -> None:
    with pytest.raises(TypeError, match="must be a `Surv`, an `Outcome`"):
        gw.KaplanMeier().fit(23)  # pyright: ignore[reportArgumentType]


# -- formulas -----------------------------------------------------------------


def test_from_formula_forms() -> None:
    assert Outcome.from_formula("Surv(time, status == 2)") == DEATH
    assert Outcome.from_formula("Surv(time, status != 0)").censor_value == 0
    assert Outcome.from_formula("Surv(time, status in (1, 2))").event_value == (1, 2)
    assert Outcome.from_formula("Surv(time, status %in% c(1, 2))").event_value == (1, 2)
    assert Outcome.from_formula("Surv(time, status not in [0])").censor_value == (0,)
    assert Outcome.from_formula("Surv(time, outcome == 'died')").event_value == "died"
    assert Outcome.from_formula("Surv(time, status, event_value=2)") == DEATH
    assert Outcome.from_formula("Surv(time)") == Outcome.right("time")
    assert Outcome.from_formula("Surv(time, status == 2) ~ 1") == DEATH


def test_from_formula_other_types() -> None:
    assert Outcome.from_formula("Surv(a, b, e == 1)") == Outcome.counting(
        "a", "b", "e", event_value=1
    )
    assert Outcome.from_formula('Surv(t, e, type="left")') == Outcome.left("t", "e")
    assert Outcome.from_formula('Surv(lo, hi, type="interval2")') == Outcome.interval("lo", "hi")

    ms = Outcome.from_formula('Surv(t, cause, states={"pcm": 1, "death": 2})')

    assert ms == Outcome.multistate("t", "cause", states={"pcm": 1, "death": 2})


def test_from_formula_names() -> None:
    assert Outcome.from_formula("Surv(`follow up`, `the status` == 2)").column_names == (
        "follow up",
        "the status",
    )
    assert Outcome.from_formula("Surv(fu.time, dead)").column_names == ("fu.time", "dead")
    assert Outcome.from_formula("Surv(time, status, weights=wt)").column_names == (
        "time",
        "status",
        "wt",
    )


@pytest.mark.parametrize(
    ("formula", "message"),
    [
        ("Surv(time, status > 1)", "Encode the event by value"),
        ("surv(time, status)", "call to `Surv"),
        ("time + status", "call to `Surv"),
        ("Surv(time, status, foo=1)", "Unknown `Surv\\(\\)` argument"),
        ("Surv(time, status == x)", "literal value"),
        ("Surv(time, status == 2, event_value=2)", "given twice"),
        ('Surv(t, e, type="weird")', "Unsupported"),
        ("Surv(time, status == 2) ~ age", "response only"),
        ("~ age", "no response"),
        ("Surv(time, __import__('os'))", "column name"),
    ],
)
def test_from_formula_errors(formula: str, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        Outcome.from_formula(formula)


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
    formula = 'Surv(etime, cause, states={"pcm": "pcm", "death": "death"}, censor_value="censor")'
    aj = gw.AalenJohansen().fit(formula, data=mg)
    y = Surv.multistate(
        mg["etime"],
        mg["cause"],
        states={"pcm": "pcm", "death": "death"},
        censor_value="censor",
    )
    ref = gw.AalenJohansen().fit(y)
    np.testing.assert_allclose(aj.to_frame()["estimate"], ref.to_frame()["estimate"])

    fg = gw.FineGray("pcm").fit(formula + " ~ age + sex", data=mg)
    fg_ref = gw.FineGray("pcm").fit(
        Surv.multistate(
            mg.drop_nulls(["age", "sex"])["etime"],
            mg.drop_nulls(["age", "sex"])["cause"],
            states={"pcm": "pcm", "death": "death"},
            censor_value="censor",
        ),
        mg.drop_nulls(["age", "sex"]).select(["age", "sex"]),
    )

    np.testing.assert_allclose(fg.coef_, fg_ref.coef_, rtol=1e-10)
