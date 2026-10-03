"""Tests for the ingest extras: dropped-row reporting, `strata()` / `cluster()` formula terms,
`Outcome` and formula support in tests / RMST / metrics / `cross_validate`, `first_event()`, and
`duration()`.
"""

from __future__ import annotations

from datetime import date
from typing import Any

import numpy as np
import pandas as pd
import pytest

import greenwood as gw
from greenwood import Outcome, Surv

DEATH = Outcome.surv(time="time", event="status == 2")
MGUS_ENDPOINTS = {"pcm": ("ptime", "pstat"), "death": ("futime", "death")}


@pytest.fixture(scope="module")
def lung() -> Any:
    return gw.load_dataset("lung", backend="pandas")


@pytest.fixture(scope="module")
def mgus2() -> Any:
    return gw.load_dataset("mgus2", backend="pandas")


def _y(frame: Any) -> Surv:
    return Surv(time=frame["time"].to_numpy(), event=(frame["status"] == 2).to_numpy())


def _mgus_first_event(mgus2: Any) -> Surv:
    return gw.first_event(
        endpoints={
            "pcm": (mgus2["ptime"], mgus2["pstat"]),
            "death": (mgus2["futime"], mgus2["death"]),
        }
    )


# -- dropped-row reporting ----------------------------------------------------


def test_outcome_fit_reports_dropped_rows(lung: Any) -> None:
    cox = gw.CoxPH().fit(DEATH, ["age", "ph.ecog"], data=lung)

    assert cox.n_dropped_ == 1
    assert cox.n_ == 227
    assert "(1 observation deleted due to missingness)" in repr(cox)


def test_formula_with_dotted_names_is_filtered_by_the_binder(lung: Any) -> None:
    cox = gw.CoxPH().fit("Surv(time, status == 2) ~ age + ph.ecog + wt.loss", data=lung)

    assert cox.n_dropped_ == int(lung[["ph.ecog", "wt.loss"]].isna().any(axis=1).sum())


def test_plain_surv_reports_the_models_own_drops(lung: Any) -> None:
    cox = gw.CoxPH().fit(_y(lung), lung[["age", "meal.cal"]])

    assert cox.n_dropped_ == int(lung["meal.cal"].isna().sum())
    assert "47 observations deleted due to missingness" in repr(cox)


def test_no_drops_means_no_note(lung: Any) -> None:
    cox = gw.CoxPH().fit(DEATH, ["age", "sex"], data=lung)

    assert cox.n_dropped_ == 0
    assert "missingness" not in repr(cox)


def test_nonparametric_repr_footer(lung: Any) -> None:
    km = gw.KaplanMeier().fit(DEATH, by="ph.ecog", data=lung)

    assert km.n_dropped_ == 1
    assert repr(km).endswith("(1 observation deleted due to missingness)")


def test_cause_specific_cox_combines_both_drop_counts(mgus2: Any) -> None:
    csc = gw.CauseSpecificCox("pcm").fit(
        Outcome.first_event(MGUS_ENDPOINTS), ["age", "hgb"], data=mgus2
    )

    assert csc.n_dropped_ == int(mgus2["hgb"].isna().sum())
    assert "deleted due to missingness" in repr(csc)


# -- strata() and cluster() terms ---------------------------------------------


def test_cox_strata_term_matches_strata_argument(lung: Any) -> None:
    ref = gw.CoxPH().fit(DEATH, ["age"], data=lung, strata="sex")
    cox = gw.CoxPH().fit("Surv(time, status == 2) ~ age + strata(sex)", data=lung)

    np.testing.assert_allclose(cox.coef_, ref.coef_, rtol=1e-12)


def test_cox_strata_with_several_columns(lung: Any) -> None:
    frame = lung.dropna(subset=["ph.ecog"])
    combined = [
        f"sex={s}, ph.ecog={e}" for s, e in zip(frame["sex"], frame["ph.ecog"], strict=True)
    ]
    ref = gw.CoxPH().fit(_y(frame), frame[["age"]], strata=np.array(combined))
    cox = gw.CoxPH().fit("Surv(time, status == 2) ~ age + strata(sex, ph.ecog)", data=lung)

    np.testing.assert_allclose(cox.coef_, ref.coef_, rtol=1e-12)


def test_cox_cluster_term_matches_cluster_argument(lung: Any) -> None:
    frame = lung.dropna(subset=["inst"])
    ref = gw.CoxPH().fit(_y(frame), frame[["age"]], cluster=frame["inst"].to_numpy())
    cox = gw.CoxPH().fit("Surv(time, status == 2) ~ age + cluster(inst)", data=lung)

    np.testing.assert_allclose(cox.std_error_, ref.std_error_, rtol=1e-12)


def test_strata_term_groups_a_kaplan_meier_curve(lung: Any) -> None:
    km = gw.KaplanMeier().fit("Surv(time, status == 2) ~ strata(sex)", data=lung)
    ref = gw.KaplanMeier().fit(DEATH, by="sex", data=lung)

    np.testing.assert_allclose(km.survival_, ref.survival_)


def test_unsupported_special_term_errors(lung: Any) -> None:
    with pytest.raises(ValueError, match="does not support `strata\\(\\)`"):
        gw.AFT("weibull").fit("Surv(time, status == 2) ~ age + strata(sex)", data=lung)


def test_strata_given_twice_errors(lung: Any) -> None:
    with pytest.raises(ValueError, match="`strata` was given both"):
        gw.CoxPH().fit("Surv(time, status == 2) ~ age + strata(sex)", data=lung, strata="inst")


def test_split_terms_respects_parentheses() -> None:
    from greenwood._outcome import split_terms

    assert split_terms("a + C(b, Treatment(1)) + strata(c, d)") == [
        "a",
        "C(b, Treatment(1))",
        "strata(c, d)",
    ]


# -- hypothesis tests and RMST ------------------------------------------------


def test_logrank_formula_matches_array_form(lung: Any) -> None:
    ref = gw.logrank_test(_y(lung), lung["sex"].to_numpy())
    res = gw.logrank_test("Surv(time, status == 2) ~ sex", data=lung)

    assert res.statistic == pytest.approx(ref.statistic, rel=1e-12)

    res2 = gw.logrank_test(DEATH, "sex", data=lung)

    assert res2.statistic == pytest.approx(ref.statistic, rel=1e-12)


def test_logrank_formula_with_strata(lung: Any) -> None:
    frame = lung.dropna(subset=["ph.ecog"])
    ref = gw.logrank_test(_y(frame), frame["sex"].to_numpy(), strata=frame["ph.ecog"].to_numpy())
    res = gw.logrank_test("Surv(time, status == 2) ~ sex + strata(ph.ecog)", data=lung)

    assert res.statistic == pytest.approx(ref.statistic, rel=1e-12)


def test_test_needs_a_group(lung: Any) -> None:
    with pytest.raises(ValueError, match="needs `group`"):
        gw.logrank_test(DEATH, data=lung)


def test_rmst_formula_matches_array_form(lung: Any) -> None:
    ref = gw.rmst_test(_y(lung), 365, lung["sex"].to_numpy())
    res = gw.rmst_test("Surv(time, status == 2) ~ sex", 365, data=lung)

    assert res.estimate == pytest.approx(ref.estimate, rel=1e-12)


def test_pairwise_logrank_by_name(lung: Any) -> None:
    frame = lung.dropna(subset=["ph.ecog"])
    ref = gw.pairwise_logrank_test(_y(frame), frame["ph.ecog"].to_numpy(), format="polars")
    res = gw.pairwise_logrank_test(DEATH, "ph.ecog", data=lung, format="polars")

    np.testing.assert_allclose(res["p_value"].to_numpy(), ref["p_value"].to_numpy())


def test_grays_test_with_first_event(mgus2: Any) -> None:
    y = _mgus_first_event(mgus2)
    ref = gw.grays_test(y, mgus2["sex"].to_numpy(), cause="pcm")
    res = gw.grays_test(Outcome.first_event(MGUS_ENDPOINTS), "sex", data=mgus2, cause="pcm")

    assert res.statistic == pytest.approx(ref.statistic, rel=1e-12)


# -- metrics ------------------------------------------------------------------


def test_concordance_with_outcome_filters_the_predictions() -> None:
    import polars as pl

    df = pl.DataFrame(
        {
            "time": [5.0, None, 3.0, 8.0, 2.0],
            "status": [2, 2, 1, 2, 2],
            "risk": [0.2, 0.9, 0.5, 0.1, 0.8],
        }
    )
    ref = gw.concordance_index(
        Surv(time=[5.0, 3.0, 8.0, 2.0], event=[1, 0, 1, 1]), np.array([0.2, 0.5, 0.1, 0.8])
    )
    by_array = gw.concordance_index(DEATH, np.array([0.2, 0.9, 0.5, 0.1, 0.8]), data=df)
    by_name = gw.concordance_index(DEATH, "risk", data=df)

    assert by_array == pytest.approx(ref)
    assert by_name == pytest.approx(ref)


def test_brier_score_filters_a_2d_prediction_matrix() -> None:
    import polars as pl

    df = pl.DataFrame({"time": [5.0, None, 3.0, 8.0, 2.0], "status": [2, 2, 1, 2, 2]})
    probs = np.linspace(0.1, 0.9, 10).reshape(5, 2)
    times = [2.5, 4.0]
    ref = gw.brier_score(
        Surv(time=[5.0, 3.0, 8.0, 2.0], event=[1, 0, 1, 1]), probs[[0, 2, 3, 4]], times
    )
    res = gw.brier_score(DEATH, probs, times, data=df)

    np.testing.assert_allclose(res, ref)


def test_metric_formula_takes_no_covariates(lung: Any) -> None:
    with pytest.raises(ValueError, match="takes no covariates"):
        gw.concordance_index("Surv(time, status == 2) ~ age", np.zeros(len(lung)), data=lung)


def test_cross_validate_formula(lung: Any) -> None:
    ref = gw.cross_validate(gw.CoxPH(), _y(lung), lung[["age", "sex"]], seed=3)
    res = gw.cross_validate(gw.CoxPH(), "Surv(time, status == 2) ~ age + sex", data=lung, seed=3)

    np.testing.assert_allclose(res["scores"], ref["scores"])


# -- first_event --------------------------------------------------------------


def test_first_event_matches_the_r_recipe(mgus2: Any) -> None:
    etime = np.where(mgus2["pstat"] == 1, mgus2["ptime"], mgus2["futime"])
    cause = np.where(mgus2["pstat"] == 1, 1, 2 * mgus2["death"])
    status = pd.Categorical.from_codes(
        np.asarray(cause, dtype=int), categories=["censor", "pcm", "death"]
    )
    ref = Surv(time=etime, event=status)
    y = _mgus_first_event(mgus2)

    np.testing.assert_array_equal(y.stop, ref.stop)
    np.testing.assert_array_equal(y.status, ref.status)
    assert y.states == ("pcm", "death")


def test_first_event_ties_follow_endpoint_order() -> None:
    a = ([5.0], [1])
    b = ([5.0], [1])
    a_first = gw.first_event(endpoints={"a": a, "b": b})
    b_first = gw.first_event(endpoints={"b": b, "a": a})

    assert a_first.states is not None and b_first.states is not None
    assert a_first.states[int(a_first.status[0]) - 1] == "a"
    assert b_first.states[int(b_first.status[0]) - 1] == "b"


def test_first_event_earliest_wins_and_censoring_time() -> None:
    data = {
        "t1": [3.0, 9.0, 4.0],
        "e1": [1, 1, 0],
        "t2": [5.0, 2.0, 6.0],
        "e2": ["yes", "yes", "no"],
        "last": [7.0, 7.0, 10.0],
    }
    endpoints = {
        "x": (data["t1"], data["e1"]),
        "y": (data["t2"], np.asarray(data["e2"]) == "yes"),
    }
    y = gw.first_event(endpoints=endpoints)

    np.testing.assert_array_equal(y.stop, [3.0, 2.0, 6.0])
    np.testing.assert_array_equal(y.status, [1, 2, 0])

    y2 = gw.first_event(endpoints=endpoints, censor_at=data["last"])

    np.testing.assert_array_equal(y2.stop, [3.0, 2.0, 10.0])


def test_first_event_bad_spec_errors() -> None:
    with pytest.raises(ValueError, match="must be a `\\(time, event\\)` pair"):
        gw.first_event(endpoints={"a": ([1.0],)})  # pyright: ignore[reportArgumentType]
    with pytest.raises(ValueError, match="at least one"):
        gw.first_event(endpoints={})


def test_outcome_first_event_round_trip(mgus2: Any) -> None:
    cr = Outcome.first_event(
        endpoints={"pcm": ("ptime", "pstat"), "death": ("futime", "death == 1")}
    )

    assert repr(cr) == (
        "Outcome.first_event(endpoints={'pcm': ('ptime', 'pstat'), "
        "'death': ('futime', 'death == 1')})"
    )
    assert eval(repr(cr), {"Outcome": Outcome}) == cr
    assert cr.column_names == ("ptime", "pstat", "futime", "death")
    assert cr.kind == "first_event"

    hash(cr)
    y = cr.bind(mgus2)
    ref = _mgus_first_event(mgus2)

    np.testing.assert_array_equal(y.status, ref.status)


def test_first_event_fits_aalen_johansen(mgus2: Any) -> None:
    aj = gw.AalenJohansen().fit(Outcome.first_event(MGUS_ENDPOINTS), data=mgus2)
    ref = gw.AalenJohansen().fit(_mgus_first_event(mgus2))

    np.testing.assert_allclose(aj.to_frame()["estimate"], ref.to_frame()["estimate"])


# -- duration -----------------------------------------------------------------


def test_duration_units() -> None:
    data = {"a": [date(2020, 1, 1)], "b": [date(2021, 1, 1)]}
    for unit, expected in [
        ("days", 366.0),
        ("weeks", 366 / 7),
        ("months", 366 / 30.4375),
        ("years", 366 / 365.25),
    ]:
        y = Outcome.surv(time=gw.duration("a", "b", unit=unit)).bind(data)

        assert y.stop[0] == pytest.approx(expected)


@pytest.mark.parametrize("backend", ["pandas", "polars", "pyarrow", "duckdb"])
def test_duration_across_backends(backend: str) -> None:
    frame = pd.DataFrame(
        {
            "enroll": pd.to_datetime(["2021-03-01", "2021-03-15"]),
            "exit": pd.to_datetime(["2021-03-11", "2021-04-14"]),
            "died": [1, 0],
        }
    )
    if backend == "polars":
        import polars as pl

        data: Any = pl.from_pandas(frame)
    elif backend == "pyarrow":
        import pyarrow as pa

        data = pa.Table.from_pandas(frame)
    elif backend == "duckdb":
        import duckdb
        import pyarrow as pa

        data = duckdb.from_arrow(pa.Table.from_pandas(frame))
    else:
        data = frame
    y = Outcome.surv(time=gw.duration("enroll", "exit"), event="died").bind(data)

    np.testing.assert_allclose(y.stop, [10.0, 30.0])


def test_duration_accepts_iso_strings() -> None:
    y = Outcome.surv(time=gw.duration("a", "b")).bind({"a": ["2020-01-01"], "b": ["2020-03-01"]})

    assert y.stop[0] == 60.0


def test_duration_errors() -> None:
    with pytest.raises(ValueError, match="negative duration"):
        Outcome.surv(time=gw.duration("a", "b")).bind(
            {"a": [date(2021, 1, 1)], "b": [date(2020, 1, 1)]}
        )
    with pytest.raises(TypeError, match="not numbers"):
        Outcome.surv(time=gw.duration("a", "b")).bind({"a": [1], "b": [2]})
    with pytest.raises(ValueError, match="Unknown unit"):
        gw.duration("a", "b", unit="fortnights")
    with pytest.raises(ValueError, match="Pass `data=`"):
        gw.KaplanMeier().fit(Outcome.surv(time=gw.duration("a", "b")))


def test_duration_in_outcome_drops_missing_dates() -> None:
    import polars as pl

    df = pl.DataFrame(
        {
            "enroll": [date(2021, 1, 1), None, date(2021, 1, 1)],
            "exit": [date(2021, 1, 11), date(2021, 2, 1), date(2021, 1, 21)],
            "outcome": ["died", "died", "alive"],
        }
    )
    o = Outcome.surv(time=gw.duration("enroll", "exit"), event="outcome == 'died'")

    assert repr(o) == (
        "Outcome.surv(time=\"duration('enroll', 'exit', unit='days')\", "
        "event=\"outcome == 'died'\")"
    )
    assert o.column_names == ("enroll", "exit", "outcome")

    km = gw.KaplanMeier().fit(o, data=df)

    assert km.n_dropped_ == 1

    ref = gw.KaplanMeier().fit(Surv(time=[10.0, 20.0], event=[1, 0]))

    np.testing.assert_allclose(km.survival_, ref.survival_)


# -- remaining entry points ---------------------------------------------------


def test_event_table_formula_and_names(lung: Any) -> None:
    ref = gw.event_table(_y(lung), group=lung["sex"].to_numpy()).to_frame(format="pandas")
    by_formula = gw.event_table("Surv(time, status == 2) ~ sex", data=lung).to_frame(
        format="pandas"
    )
    by_name = gw.event_table(DEATH, group="sex", data=lung).to_frame(format="pandas")

    np.testing.assert_array_equal(by_formula["n_risk"], ref["n_risk"])
    np.testing.assert_array_equal(by_name["n_event"], ref["n_event"])


def test_bootstrap_with_outcome(lung: Any) -> None:
    ref = gw.bootstrap(_y(lung), "median_diff", by=lung["sex"].to_numpy(), n_boot=30, seed=7)
    res = gw.bootstrap(DEATH, "median_diff", by="sex", data=lung, n_boot=30, seed=7)

    np.testing.assert_allclose(np.asarray(res.estimate), np.asarray(ref.estimate))
    np.testing.assert_allclose(np.asarray(res.se), np.asarray(ref.se))


def test_cv_coxnet_formula(lung: Any) -> None:
    frame = lung.dropna(subset=["ph.ecog"])
    ref = gw.cv_coxnet(_y(frame), frame[["age", "sex", "ph.ecog"]], k=3, seed=2)
    res = gw.cv_coxnet("Surv(time, status == 2) ~ age + sex + ph.ecog", data=lung, k=3, seed=2)

    assert res.best_penalizer_ == pytest.approx(ref.best_penalizer_)
    np.testing.assert_allclose(res.mean_scores_, ref.mean_scores_)


def test_compare_distributions_formula(lung: Any) -> None:
    ref = gw.compare_distributions(_y(lung), format="pandas")
    res = gw.compare_distributions("Surv(time, status == 2)", data=lung, format="pandas")

    np.testing.assert_allclose(res["aic"], ref["aic"])


# -- split_episodes column selection ------------------------------------------


@pytest.fixture(scope="module")
def pbcseq() -> Any:
    return gw.load_dataset("pbcseq", backend="pandas")


def test_split_episodes_long_table_matches_two_frame_form(pbcseq: Any) -> None:
    base = pbcseq.drop_duplicates("id")[["id", "futime", "status"]]
    old = gw.split_episodes(
        baseline=base,
        visits=pbcseq[["id", "day", "bili", "albumin"]],
        id="id",
        time="futime",
        event="status",
        visit_time="day",
        format="pandas",
    )
    new = gw.split_episodes(
        baseline=pbcseq,
        visits=pbcseq,
        id="id",
        time="futime",
        event="status",
        visit_time="day",
        covariates=["bili", "albumin"],
        format="pandas",
    )
    pd.testing.assert_frame_equal(old.reset_index(drop=True), new.reset_index(drop=True))


def test_split_episodes_baseline_covariates_and_lazy_input(pbcseq: Any) -> None:
    import polars as pl

    long = gw.split_episodes(
        baseline=pl.from_pandas(pbcseq).lazy(),
        visits=pl.from_pandas(pbcseq),
        id="id",
        time="futime",
        event="status",
        visit_time="day",
        covariates=["bili"],
        baseline_covariates=["trt", "age"],
        format="polars",
    )
    assert long.columns == ["id", "tstart", "tstop", "status", "trt", "age", "bili"]


def test_split_episodes_errors(pbcseq: Any) -> None:
    common: dict[str, Any] = {"id": "id", "time": "futime", "event": "status", "visit_time": "day"}
    with pytest.raises(ValueError, match="more than one value of 'chol'"):
        gw.split_episodes(
            baseline=pbcseq,
            visits=pbcseq,
            covariates=["bili"],
            baseline_covariates=["chol"],
            **common,
        )
    with pytest.raises(ValueError, match="both time-fixed"):
        gw.split_episodes(
            baseline=pbcseq,
            visits=pbcseq,
            covariates=["age"],
            baseline_covariates=["age"],
            **common,
        )
    with pytest.raises(ValueError, match="'bilirubin' not found in visits"):
        gw.split_episodes(baseline=pbcseq, visits=pbcseq, covariates=["bilirubin"], **common)


# -- MultiState column names ----------------------------------------------------


def test_multistate_fit_by_column_names() -> None:
    import polars as pl

    intervals = pl.DataFrame(
        {
            "tstart": [0.0, 0.0, 5.0, 0.0],
            "tstop": [5.0, 8.0, 9.0, 4.0],
            "from": ["a", "a", "b", "a"],
            "to": ["b", None, "c", "c"],
        }
    )
    by_name = gw.MultiState().fit(
        start="tstart", stop="tstop", state="from", event="to", data=intervals
    )
    by_value = gw.MultiState().fit(
        start=intervals["tstart"],
        stop=intervals["tstop"],
        state=intervals["from"],
        event=intervals["to"],
    )
    a, b = by_name.to_frame(format="pandas"), by_value.to_frame(format="pandas")
    np.testing.assert_allclose(a.drop(columns="time").to_numpy(), b.drop(columns="time").to_numpy())
    with pytest.raises(ValueError, match="no `data=` was given"):
        gw.MultiState().fit(start="tstart", stop="tstop", state="from", event="to")


# -- duration() in formulas -----------------------------------------------------


def test_duration_in_formula() -> None:
    import polars as pl

    df = pl.DataFrame(
        {
            "enroll": [date(2021, 1, 1), date(2021, 1, 1), date(2021, 2, 1)],
            "exit": [date(2021, 1, 11), date(2021, 1, 21), date(2021, 3, 3)],
            "outcome": ["died", "alive", "died"],
        }
    )
    parsed = Outcome.from_formula('Surv(duration(enroll, exit, unit="weeks"), outcome == "died")')
    built = Outcome.surv(
        time=gw.duration(start="enroll", end="exit", unit="weeks"),
        event="outcome == 'died'",
    )
    spelled = Outcome.surv(time="duration(enroll, exit, unit='weeks')", event="outcome == 'died'")
    assert parsed == built == spelled
    km = gw.KaplanMeier().fit('Surv(duration(enroll, exit), outcome == "died")', data=df)
    ref = gw.KaplanMeier().fit(gw.Surv(time=[10.0, 20.0, 30.0], event=[1, 0, 1]))
    np.testing.assert_allclose(km.survival_, ref.survival_)


@pytest.mark.parametrize(
    "formula",
    [
        "Surv(duration(enroll), died)",
        "Surv(duration(enroll, exit, days), died)",
        'Surv(duration(enroll, exit, scale="days"), died)',
    ],
)
def test_bad_duration_in_formula(formula: str) -> None:
    with pytest.raises(ValueError):
        Outcome.from_formula(formula)


# -- frailty() terms ------------------------------------------------------------


@pytest.mark.parametrize(
    ("term", "dist"),
    [
        ("frailty(inst)", "gamma"),
        ("frailty.gamma(inst)", "gamma"),
        ('frailty(inst, distribution="gaussian")', "lognormal"),
        ("frailty.gaussian(inst)", "lognormal"),
    ],
)
def test_frailty_term_matches_arguments(lung: Any, term: str, dist: str) -> None:
    ref = gw.CoxPH(ties="breslow").fit(
        DEATH, covariates=["age", "sex"], data=lung, frailty=dist, frailty_cluster="inst"
    )
    cox = gw.CoxPH(ties="breslow").fit(f"Surv(time, status == 2) ~ age + sex + {term}", data=lung)
    np.testing.assert_allclose(cox.coef_, ref.coef_, rtol=1e-10)


def test_frailty_term_errors(lung: Any) -> None:
    formula = "Surv(time, status == 2) ~ age + frailty(inst)"
    with pytest.raises(ValueError, match="also given"):
        gw.CoxPH(ties="breslow").fit(formula, data=lung, frailty="lognormal")
    with pytest.raises(ValueError, match="not supported"):
        gw.CoxPH(ties="breslow").fit(
            "Surv(time, status == 2) ~ age + frailty(inst, theta=1)", data=lung
        )
    with pytest.raises(ValueError, match="two different frailty"):
        gw.CoxPH(ties="breslow").fit(
            'Surv(time, status == 2) ~ age + frailty.gamma(inst, distribution="gaussian")',
            data=lung,
        )
    with pytest.raises(ValueError, match="does not support `frailty\\(\\)`"):
        gw.AFT(dist="weibull").fit(formula, data=lung)


# -- first_event() sanity warning ------------------------------------------------


def test_first_event_warns_when_an_event_follows_ended_follow_up() -> None:
    data = {"p": [80.0, 5.0], "ps": [1, 0], "f": [60.0, 9.0], "d": [0, 0]}
    with pytest.warns(UserWarning, match="follow-up stops at 60"):
        gw.first_event(endpoints={"pcm": (data["p"], data["ps"]), "death": (data["f"], data["d"])})


def test_first_event_is_quiet_on_mgus2(mgus2: Any) -> None:
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _mgus_first_event(mgus2)
