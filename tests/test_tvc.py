"""Tests for split_episodes() (episode splitting for time-varying covariates)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import greenwood as gw
from greenwood import CoxPH, Surv
from tests._r_parity import assert_allclose_to_r, load_fixture

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def simple_baseline() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "id": [1, 2, 3],
            "time": [10.0, 8.0, 12.0],
            "event": [1, 0, 1],
            "sex": ["m", "f", "m"],
        }
    )


@pytest.fixture
def simple_visits() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "id": [1, 1, 2, 3, 3, 3],
            "day": [0.0, 5.0, 0.0, 0.0, 4.0, 8.0],
            "bili": [1.2, 2.4, 0.8, 3.1, 2.9, 4.0],
        }
    )


# ---------------------------------------------------------------------------
# Basic structural checks
# ---------------------------------------------------------------------------


def test_basic_shape(simple_baseline: pd.DataFrame, simple_visits: pd.DataFrame) -> None:
    out = gw.split_episodes(
        simple_baseline,
        simple_visits,
        id="id",
        time="time",
        event="event",
        visit_time="day",
        format="pandas",
    )
    assert isinstance(out, pd.DataFrame)
    # subject 1: visits at 0, 5 → intervals (0,5), (5,10) = 2
    # subject 2: visit at 0 → interval (0,8) = 1
    # subject 3: visits at 0, 4, 8 → intervals (0,4), (4,8), (8,12) = 3
    assert len(out) == 6


def test_required_columns_present(
    simple_baseline: pd.DataFrame, simple_visits: pd.DataFrame
) -> None:
    out = gw.split_episodes(
        simple_baseline,
        simple_visits,
        id="id",
        time="time",
        event="event",
        visit_time="day",
        format="pandas",
    )
    assert set(out.columns) >= {"id", "tstart", "tstop", "event", "sex", "bili"}


def test_tstart_tstop_ordering(simple_baseline: pd.DataFrame, simple_visits: pd.DataFrame) -> None:
    out = gw.split_episodes(
        simple_baseline,
        simple_visits,
        id="id",
        time="time",
        event="event",
        visit_time="day",
        format="pandas",
    )
    assert (out["tstart"] < out["tstop"]).all()


# ---------------------------------------------------------------------------
# Event indicator placement
# ---------------------------------------------------------------------------


def test_event_only_on_last_interval(
    simple_baseline: pd.DataFrame, simple_visits: pd.DataFrame
) -> None:
    out = gw.split_episodes(
        simple_baseline,
        simple_visits,
        id="id",
        time="time",
        event="event",
        visit_time="day",
        format="pandas",
    )
    for subj_id, base_time, base_event in [(1, 10.0, 1), (2, 8.0, 0), (3, 12.0, 1)]:
        rows = out[out["id"] == subj_id]
        last = rows[rows["tstop"] == base_time]
        non_last = rows[rows["tstop"] != base_time]
        assert len(last) == 1
        assert int(last["event"].iloc[0]) == base_event
        assert (non_last["event"] == 0).all()


def test_censored_subject_event_zero(
    simple_baseline: pd.DataFrame, simple_visits: pd.DataFrame
) -> None:
    out = gw.split_episodes(
        simple_baseline,
        simple_visits,
        id="id",
        time="time",
        event="event",
        visit_time="day",
        format="pandas",
    )
    subj2 = out[out["id"] == 2]
    assert (subj2["event"] == 0).all()


# ---------------------------------------------------------------------------
# Covariate values
# ---------------------------------------------------------------------------


def test_covariate_values_carried_correctly(
    simple_baseline: pd.DataFrame, simple_visits: pd.DataFrame
) -> None:
    out = gw.split_episodes(
        simple_baseline,
        simple_visits,
        id="id",
        time="time",
        event="event",
        visit_time="day",
        format="pandas",
    )
    subj1 = out[out["id"] == 1].sort_values("tstart")
    assert float(subj1.iloc[0]["bili"]) == pytest.approx(1.2)  # visit at day 0
    assert float(subj1.iloc[1]["bili"]) == pytest.approx(2.4)  # visit at day 5

    subj3 = out[out["id"] == 3].sort_values("tstart")
    assert float(subj3.iloc[0]["bili"]) == pytest.approx(3.1)
    assert float(subj3.iloc[1]["bili"]) == pytest.approx(2.9)
    assert float(subj3.iloc[2]["bili"]) == pytest.approx(4.0)


def test_static_covariate_propagated(
    simple_baseline: pd.DataFrame, simple_visits: pd.DataFrame
) -> None:
    out = gw.split_episodes(
        simple_baseline,
        simple_visits,
        id="id",
        time="time",
        event="event",
        visit_time="day",
        format="pandas",
    )
    for subj_id, sex in [(1, "m"), (2, "f"), (3, "m")]:
        rows = out[out["id"] == subj_id]
        assert (rows["sex"] == sex).all()


# ---------------------------------------------------------------------------
# carry_forward=False
# ---------------------------------------------------------------------------


def test_carry_forward_false_drops_trailing_interval() -> None:
    baseline = pd.DataFrame({"id": [1], "time": [10.0], "event": [1]})
    visits = pd.DataFrame({"id": [1, 1], "day": [0.0, 5.0], "bili": [1.2, 2.4]})

    out_cf = gw.split_episodes(
        baseline,
        visits,
        id="id",
        time="time",
        event="event",
        visit_time="day",
        carry_forward=True,
        format="pandas",
    )
    out_no_cf = gw.split_episodes(
        baseline,
        visits,
        id="id",
        time="time",
        event="event",
        visit_time="day",
        carry_forward=False,
        format="pandas",
    )
    # carry_forward=True: (0,5) and (5,10) → 2 rows
    # carry_forward=False: (0,5) only → 1 row (trailing interval dropped)
    assert len(out_cf) == 2
    assert len(out_no_cf) == 1
    assert float(out_no_cf["tstop"].iloc[0]) == pytest.approx(5.0)


def test_carry_forward_false_single_visit_at_zero_preserved() -> None:
    """A single visit at day 0 must produce one interval even with carry_forward=False."""
    baseline = pd.DataFrame({"id": [1, 2], "time": [10.0, 8.0], "event": [1, 0]})
    visits = pd.DataFrame({"id": [1], "day": [0.0], "bili": [1.5]})

    out = gw.split_episodes(
        baseline,
        visits,
        id="id",
        time="time",
        event="event",
        visit_time="day",
        carry_forward=False,
        format="pandas",
    )
    # Subject 2 has no visits → dropped (CF=False)
    # Subject 1 has single visit at day 0: the only possible interval (0,10) is retained
    assert set(out["id"].tolist()) == {1}
    assert len(out) == 1


def test_carry_forward_false_no_visits_drops_subject() -> None:
    baseline = pd.DataFrame({"id": [1, 2], "time": [10.0, 8.0], "event": [1, 0]})
    visits = pd.DataFrame({"id": [1, 1], "day": [0.0, 5.0], "bili": [1.5, 2.0]})

    out = gw.split_episodes(
        baseline,
        visits,
        id="id",
        time="time",
        event="event",
        visit_time="day",
        carry_forward=False,
        format="pandas",
    )
    # Subject 2 has no visits → not in output
    assert 2 not in out["id"].tolist()
    # Subject 1 has two visits: (0,5) included, (5,10) trailing → dropped
    assert set(out["id"].tolist()) == {1}
    assert len(out) == 1


# ---------------------------------------------------------------------------
# Pre-visit gap (first visit not at time 0)
# ---------------------------------------------------------------------------


def test_pre_visit_gap_carry_forward_true() -> None:
    baseline = pd.DataFrame({"id": [1], "time": [10.0], "event": [1]})
    visits = pd.DataFrame({"id": [1], "day": [3.0], "bili": [2.0]})

    out = gw.split_episodes(
        baseline,
        visits,
        id="id",
        time="time",
        event="event",
        visit_time="day",
        carry_forward=True,
        format="pandas",
    )
    out_sorted = out.sort_values("tstart")
    # Expect (0,3) with NaN bili, then (3,10) with bili=2.0
    assert len(out_sorted) == 2
    assert float(out_sorted.iloc[0]["tstart"]) == pytest.approx(0.0)
    assert float(out_sorted.iloc[0]["tstop"]) == pytest.approx(3.0)
    assert np.isnan(float(out_sorted.iloc[0]["bili"]))
    assert float(out_sorted.iloc[1]["bili"]) == pytest.approx(2.0)


def test_pre_visit_gap_carry_forward_false() -> None:
    """With carry_forward=False, the pre-visit NaN gap is excluded.

    Single visit at day 3: no pre-gap interval (CF=False excludes it), and since
    intervals_added_for_subject==0 at the trailing interval, that interval IS kept.
    """
    baseline = pd.DataFrame({"id": [1], "time": [10.0], "event": [1]})
    visits = pd.DataFrame({"id": [1], "day": [3.0], "bili": [2.0]})

    out = gw.split_episodes(
        baseline,
        visits,
        id="id",
        time="time",
        event="event",
        visit_time="day",
        carry_forward=False,
        format="pandas",
    )
    # Pre-gap excluded, trailing (3,10) kept because it is the only interval
    assert len(out) == 1
    assert float(out["tstart"].iloc[0]) == pytest.approx(3.0)
    assert float(out["tstop"].iloc[0]) == pytest.approx(10.0)
    assert float(out["bili"].iloc[0]) == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_visit_at_exactly_end_time_skipped() -> None:
    """A visit at exactly t_end creates a degenerate (tstart==tstop) interval that is dropped."""
    baseline = pd.DataFrame({"id": [1], "time": [10.0], "event": [1]})
    visits = pd.DataFrame({"id": [1, 1], "day": [0.0, 10.0], "bili": [1.0, 2.0]})

    out = gw.split_episodes(
        baseline,
        visits,
        id="id",
        time="time",
        event="event",
        visit_time="day",
        format="pandas",
    )
    assert (out["tstart"] < out["tstop"]).all()
    assert float(out["tstop"].max()) == pytest.approx(10.0)


def test_duplicate_visit_times_degenerate_dropped() -> None:
    baseline = pd.DataFrame({"id": [1], "time": [10.0], "event": [1]})
    visits = pd.DataFrame({"id": [1, 1, 1], "day": [0.0, 5.0, 5.0], "bili": [1.0, 2.0, 3.0]})

    out = gw.split_episodes(
        baseline,
        visits,
        id="id",
        time="time",
        event="event",
        visit_time="day",
        format="pandas",
    )
    assert (out["tstart"] < out["tstop"]).all()


def test_no_visits_with_carry_forward() -> None:
    baseline = pd.DataFrame({"id": [1], "time": [5.0], "event": [1]})
    visits = pd.DataFrame(
        {
            "id": pd.Series([], dtype=int),
            "day": pd.Series([], dtype=float),
            "bili": pd.Series([], dtype=float),
        }
    )

    out = gw.split_episodes(
        baseline,
        visits,
        id="id",
        time="time",
        event="event",
        visit_time="day",
        carry_forward=True,
        format="pandas",
    )
    assert len(out) == 1
    assert float(out["tstart"].iloc[0]) == pytest.approx(0.0)
    assert float(out["tstop"].iloc[0]) == pytest.approx(5.0)
    assert np.isnan(float(out["bili"].iloc[0]))


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


def test_missing_id_column_in_baseline() -> None:
    baseline = pd.DataFrame({"subj": [1], "time": [5.0], "event": [1]})
    visits = pd.DataFrame({"id": [1], "day": [0.0], "bili": [1.0]})
    with pytest.raises(ValueError, match="id"):
        gw.split_episodes(baseline, visits, id="id", time="time", event="event", visit_time="day")


def test_missing_visit_time_column_in_visits() -> None:
    baseline = pd.DataFrame({"id": [1], "time": [5.0], "event": [1]})
    visits = pd.DataFrame({"id": [1], "t": [0.0], "bili": [1.0]})
    with pytest.raises(ValueError, match="day"):
        gw.split_episodes(baseline, visits, id="id", time="time", event="event", visit_time="day")


def test_visit_after_end_time_raises() -> None:
    baseline = pd.DataFrame({"id": [1], "time": [5.0], "event": [1]})
    visits = pd.DataFrame({"id": [1], "day": [6.0], "bili": [1.0]})
    with pytest.raises(ValueError, match="exceed follow-up"):
        gw.split_episodes(baseline, visits, id="id", time="time", event="event", visit_time="day")


def test_no_matching_ids_raises() -> None:
    baseline = pd.DataFrame({"id": [1, 2], "time": [5.0, 8.0], "event": [1, 0]})
    visits = pd.DataFrame({"id": [99], "day": [0.0], "bili": [1.0]})
    with pytest.raises(ValueError, match="no rows"):
        gw.split_episodes(
            baseline,
            visits,
            id="id",
            time="time",
            event="event",
            visit_time="day",
            carry_forward=False,
        )


# ---------------------------------------------------------------------------
# Output format
# ---------------------------------------------------------------------------


def test_format_polars(simple_baseline: pd.DataFrame, simple_visits: pd.DataFrame) -> None:
    polars = pytest.importorskip("polars")
    out = gw.split_episodes(
        simple_baseline,
        simple_visits,
        id="id",
        time="time",
        event="event",
        visit_time="day",
        format="polars",
    )
    assert isinstance(out, polars.DataFrame)


def test_format_pyarrow(simple_baseline: pd.DataFrame, simple_visits: pd.DataFrame) -> None:
    pyarrow = pytest.importorskip("pyarrow")
    out = gw.split_episodes(
        simple_baseline,
        simple_visits,
        id="id",
        time="time",
        event="event",
        visit_time="day",
        format="pyarrow",
    )
    assert isinstance(out, pyarrow.Table)


def test_polars_input_polars_output() -> None:
    polars = pytest.importorskip("polars")
    baseline = polars.DataFrame(
        {
            "id": [1, 2],
            "time": [10.0, 8.0],
            "event": [1, 0],
        }
    )
    visits = polars.DataFrame(
        {
            "id": [1, 1, 2],
            "day": [0.0, 5.0, 0.0],
            "x": [1.0, 2.0, 3.0],
        }
    )
    out = gw.split_episodes(
        baseline,
        visits,
        id="id",
        time="time",
        event="event",
        visit_time="day",
    )
    # auto-detect: polars input → polars output
    assert isinstance(out, polars.DataFrame)


# ---------------------------------------------------------------------------
# Integration: result feeds directly into Surv.counting + CoxPH
# ---------------------------------------------------------------------------


def test_split_then_cox_fit(simple_baseline: pd.DataFrame, simple_visits: pd.DataFrame) -> None:
    out = gw.split_episodes(
        simple_baseline,
        simple_visits,
        id="id",
        time="time",
        event="event",
        visit_time="day",
        format="pandas",
    )
    # All first visits are at day 0, so no NaN bili values expected here
    out_clean = out.dropna(subset=["bili"])
    y = Surv.counting(out_clean["tstart"], out_clean["tstop"], out_clean["event"])
    cox = CoxPH().fit(y, out_clean[["bili"]])
    assert len(cox.coef_) == 1


def test_split_with_two_tvc_columns() -> None:
    """Multiple TVC columns are all passed through correctly."""
    baseline = pd.DataFrame({"id": [1, 2], "time": [10.0, 8.0], "event": [1, 0]})
    visits = pd.DataFrame(
        {
            "id": [1, 1, 2, 2],
            "day": [0.0, 5.0, 0.0, 4.0],
            "bili": [1.2, 2.4, 0.8, 1.0],
            "albumin": [3.5, 3.2, 4.0, 3.8],
        }
    )
    out = gw.split_episodes(
        baseline,
        visits,
        id="id",
        time="time",
        event="event",
        visit_time="day",
        format="pandas",
    )
    assert "bili" in out.columns
    assert "albumin" in out.columns
    # subject 1: two visits → (0,5) and (5,10)
    subj1 = out[out["id"] == 1].sort_values("tstart")
    assert float(subj1.iloc[0]["albumin"]) == pytest.approx(3.5)
    assert float(subj1.iloc[1]["albumin"]) == pytest.approx(3.2)


# ---------------------------------------------------------------------------
# R-parity: pbcseq TVC Cox model
# ---------------------------------------------------------------------------


@pytest.mark.rparity
def test_tvc_pbcseq_cox_rparity() -> None:
    """split_episodes + CoxPH on pbcseq must match R's tmerge + coxph to 1e-6."""
    fx = load_fixture("tvc_pbcseq")

    pbcseq = gw.load_dataset("pbcseq", backend="pandas")
    base = (
        pbcseq.drop_duplicates("id")[["id", "futime", "status"]]
        .rename(columns={"futime": "time"})
    )
    long = gw.split_episodes(
        base,
        pbcseq[["id", "day", "bili", "albumin", "protime"]],
        id="id",
        time="time",
        event="status",
        visit_time="day",
        format="pandas",
    )
    long = long.dropna(subset=["bili", "albumin", "protime"])
    long["event_bin"] = (long["status"] == 2).astype(int)

    assert len(long) == fx["n"]
    assert int(long["event_bin"].sum()) == fx["nevent"]

    y = Surv.counting(long["tstart"], long["tstop"], long["event_bin"])
    cox = CoxPH().fit(y, long[["bili", "albumin", "protime"]])

    assert_allclose_to_r(cox.coef_, fx["coef"], rtol=1e-6, atol=1e-6, what="TVC Cox coef")
    assert_allclose_to_r(
        [cox.loglik_], [fx["loglik"]], rtol=1e-6, atol=1e-6, what="TVC Cox loglik"
    )
