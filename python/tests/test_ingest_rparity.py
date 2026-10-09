"""R-parity for the ingest API: formulas, `Outcome`, and column names checked against R fixtures.

The array-form tests in `test_r_parity.py` pin Greenwood's numerics to R. These tests build the
same models through the new front doors (formula strings, `Outcome` bound at fit time, `strata()` /
`cluster()` terms, `first_event()`, column names with `data=`) and check them against the same
fixtures, on more than one backend. They catch any difference in how the new paths select rows,
code covariates, or encode events.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

import greenwood as gw
from greenwood import Outcome

from ._r_parity import assert_allclose_to_r, load_fixture
from .test_r_parity import (  # pyright: ignore[reportPrivateUsage]
    _check_cox,
    _check_km,
)

pytestmark = pytest.mark.rparity

DEATH = Outcome.surv(time="time", event="status == 2")
MGUS_ENDPOINTS = {"pcm": ("ptime", "pstat"), "death": ("futime", "death")}


def _backend(name: str, frame: Any) -> Any:
    import polars as pl
    import pyarrow as pa

    if name == "pandas":
        return frame
    if name == "polars":
        return pl.from_pandas(frame)
    import duckdb

    return duckdb.from_arrow(pa.Table.from_pandas(frame, preserve_index=False))


@pytest.fixture(scope="module")
def lung() -> Any:
    return gw.load_dataset("lung", backend="pandas")


# -- Cox ---------------------------------------------------------------------


@pytest.mark.parametrize("backend", ["pandas", "polars", "duckdb"])
@pytest.mark.parametrize("ties", ["efron", "breslow"])
def test_cox_formula_matches_r(lung: Any, backend: str, ties: str) -> None:
    cox = gw.CoxPH(ties=ties).fit(
        "Surv(time, status == 2) ~ age + sex", data=_backend(backend, lung)
    )
    _check_cox(cox, load_fixture(f"cox_lung_age_sex_{ties}"), f"formula {ties} ({backend})")


def test_cox_outcome_with_missing_covariate_matches_r(lung: Any) -> None:
    # R drops the one patient with no ph.ecog; the binder must drop the same row.
    cox = gw.CoxPH().fit(DEATH, covariates=["age", "sex", "ph.ecog"], data=lung)
    _check_cox(cox, load_fixture("cox_lung_three_efron"), "outcome three covariates")
    assert cox.n_dropped_ == 1


def test_cox_strata_term_matches_r(lung: Any) -> None:
    fixture = load_fixture("cox_strata")
    cox = gw.CoxPH().fit("Surv(time, status == 2) ~ age + ph.ecog + strata(sex)", data=lung)
    assert cox.term_names_ == fixture["terms"]
    assert cox.n_ == fixture["n"]
    assert cox.n_event_ == fixture["nevent"]
    assert_allclose_to_r(cox.coef_, fixture["coef"], what="strata() coef")
    assert_allclose_to_r(cox.std_error_, fixture["se"], what="strata() se")
    assert_allclose_to_r(cox.loglik_, fixture["loglik"], atol=1e-6, what="strata() loglik")


def test_cox_cluster_term_matches_r(lung: Any) -> None:
    # R's coxph(..., cluster = inst) drops the patient with no institution code.
    fixture = load_fixture("cox_cluster")
    cox = gw.CoxPH(ties="breslow").fit(
        "Surv(time, status == 2) ~ age + sex + cluster(inst)", data=lung
    )
    assert cox.n_ == fixture["n"]
    assert_allclose_to_r(cox.coef_, fixture["coef"], what="cluster() coef")
    assert_allclose_to_r(cox.std_error_, fixture["robust_se"], what="cluster() robust se")


# -- Kaplan-Meier and log-rank --------------------------------------------------


def test_km_formula_matches_r(lung: Any) -> None:
    km = gw.KaplanMeier(conf_type="log").fit("Surv(time, status == 2) ~ 1", data=lung)
    _check_km(km, load_fixture("km_lung_overall")["overall"], "formula KM")


@pytest.mark.parametrize("backend", ["pandas", "polars", "duckdb"])
def test_km_by_sex_formula_matches_r(lung: Any, backend: str) -> None:
    fixture = load_fixture("km_lung_by_sex")
    km = gw.KaplanMeier(conf_type="log-log").fit(
        "Surv(time, status == 2) ~ sex", data=_backend(backend, lung)
    )
    for block in km._blocks:  # pyright: ignore[reportPrivateUsage]
        assert block.label.startswith("sex=")
        expected = fixture[block.label.removeprefix("sex=")]
        assert_allclose_to_r(block.surv, expected["surv"], what=f"{block.label} surv")
        assert_allclose_to_r(block.conf_low, expected["lower_loglog"], what="lower")
        assert_allclose_to_r(block.conf_high, expected["upper_loglog"], what="upper")


def test_logrank_formula_matches_r(lung: Any) -> None:
    result = gw.logrank_test("Surv(time, status == 2) ~ sex", data=lung)
    fixture = load_fixture("logrank_lung_sex")
    assert result.df == fixture["df"]
    assert_allclose_to_r(result.statistic, fixture["chisq"], what="formula log-rank chisq")
    assert_allclose_to_r(result.p_value, fixture["p"], what="formula log-rank p")
    # Group keys are the column's own values (here the integers 1 and 2)
    observed = {str(k): v for k, v in result.observed.items()}
    expected = {str(k): v for k, v in result.expected.items()}
    for g, obs, exp in zip(fixture["groups"], fixture["obs"], fixture["exp"], strict=True):
        assert_allclose_to_r(observed[str(g)], obs, what=f"observed[{g}]")
        assert_allclose_to_r(expected[str(g)], exp, what=f"expected[{g}]")


def test_stratified_logrank_formula_matches_r(lung: Any) -> None:
    # R's survdiff drops rows with no ph.ecog; the binder drops them from the formula's columns.
    fixture = load_fixture("logrank_stratified_lung_sex_ecog")
    result = gw.logrank_test("Surv(time, status == 2) ~ sex + strata(ph.ecog)", data=lung)
    assert_allclose_to_r(result.statistic, fixture["chisq"], what="stratified chisq")
    assert_allclose_to_r(result.p_value, fixture["p"], what="stratified p")
    assert result.df == fixture["df"]


# -- competing risks and multi-state ----------------------------------------------


@pytest.mark.parametrize("backend", ["pandas", "polars", "duckdb"])
def test_fine_gray_first_event_matches_r(backend: str) -> None:
    mg = gw.load_dataset("mgus2", backend="pandas")
    fixture = load_fixture("finegray_mgus2_pcm")
    fg = gw.FineGray(cause="pcm").fit(
        Outcome.first_event(endpoints=MGUS_ENDPOINTS),
        covariates=["age", "sex"],
        data=_backend(backend, mg),
    )
    assert fg.term_names_ == fixture["terms"]
    assert_allclose_to_r(fg.coef_, fixture["coef"], what="fine-gray coef")
    assert_allclose_to_r(fg.naive_std_error_, fixture["naive_se"], what="fine-gray naive se")
    assert_allclose_to_r(fg.std_error_, fixture["robust_se"], what="fine-gray robust se")


def test_multistate_column_names_match_r() -> None:
    import pandas as pd

    from .test_r_parity import _mgus2_illness_death  # pyright: ignore[reportPrivateUsage]

    t0, t1, frm, evt = _mgus2_illness_death()  # pyright: ignore[reportUnknownVariableType]
    intervals = pd.DataFrame({"tstart": t0, "tstop": t1, "from": frm, "to": evt})
    fixture = load_fixture("multistate_mgus2")
    ms = gw.MultiState().fit(
        start="tstart",
        stop="tstop",
        state="from",
        event="to",
        states=("mgus", "pcm", "death"),
        data=intervals,
    )
    table = ms.to_frame(format="pandas")
    assert_allclose_to_r(table["time"].to_numpy(), fixture["time"], what="ms time")
    for state in ("mgus", "pcm", "death"):
        assert_allclose_to_r(table[state].to_numpy(), fixture[state], what=f"occupancy {state}")


# -- time-varying covariates -------------------------------------------------------


def test_tvc_split_episodes_long_table_matches_r() -> None:
    # One long table serves as both baseline and visits, and the lab columns are named.
    fx = load_fixture("tvc_pbcseq")
    pbcseq = gw.load_dataset("pbcseq", backend="pandas")
    long = gw.split_episodes(
        baseline=pbcseq,
        visits=pbcseq,
        id="id",
        time="futime",
        event="status",
        visit_time="day",
        covariates=["bili", "albumin", "protime"],
        format="pandas",
    )
    tvc = Outcome.surv(time="tstart", time2="tstop", event="status == 2")
    cox = gw.CoxPH().fit(tvc, covariates=["bili", "albumin", "protime"], data=long)
    assert cox.n_ == fx["n"]
    assert cox.n_event_ == fx["nevent"]
    assert_allclose_to_r(cox.coef_, fx["coef"], rtol=1e-6, atol=1e-6, what="TVC coef")
    assert_allclose_to_r(np.asarray([cox.loglik_]), [fx["loglik"]], rtol=1e-6, atol=1e-6)
