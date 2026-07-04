"""R-parity validation of the risk-set / event-table kernel against `survfit`.

This proves the kernel matches R's `survival::survfit` tabulation (time, n.risk, n.event,
n.censor) on real datasets, including a left-truncation case. The Kaplan-Meier estimator
builds directly on this kernel.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

import greenwood as gw
from greenwood import Surv, event_table

from ._r_parity import assert_allclose_to_r, load_fixture

pytestmark = pytest.mark.rparity


def _check_block(et: gw.EventTable, expected: dict[str, list[float]], label: str) -> None:
    assert_allclose_to_r(et.time, expected["time"], what=f"{label} time")
    assert_allclose_to_r(et.n_risk, expected["n_risk"], what=f"{label} n_risk")
    assert_allclose_to_r(et.n_event, expected["n_event"], what=f"{label} n_event")
    assert_allclose_to_r(et.n_censor, expected["n_censor"], what=f"{label} n_censor")


def test_lung_km_overall_matches_r() -> None:
    df = gw.data.load_dataset("lung", backend="pandas")
    # survival::lung codes status 1 = censored, 2 = dead.
    y = Surv.right(df["time"], event=(df["status"] == 2))
    et = event_table(y)
    _check_block(et, load_fixture("lung_km_overall")["overall"], "lung overall")


def test_lung_km_by_sex_matches_r() -> None:
    df = gw.data.load_dataset("lung", backend="pandas")
    y = Surv.right(df["time"], event=(df["status"] == 2))
    et = event_table(y, group=df["sex"])
    fixture = load_fixture("lung_km_by_sex")

    assert et.strata is not None
    for level, block in fixture.items():
        mask = et.strata.astype(int).astype(str) == str(level)
        sub = gw.EventTable(
            time=et.time[mask],
            n_risk=et.n_risk[mask],
            n_event=et.n_event[mask],
            n_censor=et.n_censor[mask],
        )
        _check_block(sub, block, f"lung sex={level}")


def test_veteran_km_overall_matches_r() -> None:
    df = gw.data.load_dataset("veteran", backend="pandas")
    y = Surv.right(df["time"], event=df["status"])
    et = event_table(y)
    _check_block(et, load_fixture("veteran_km_overall")["overall"], "veteran overall")


def test_counting_left_truncation_matches_r() -> None:
    fixture = load_fixture("counting_truncation")
    data = fixture["data"]
    y = Surv.counting(start=data["start"], stop=data["stop"], event=data["event"])
    et = event_table(y)
    _check_block(et, fixture["overall"], "counting truncation")


# -- Kaplan-Meier / Nelson-Aalen against survfit --------------------------------

_CONF = [("plain", "plain"), ("log", "log"), ("log-log", "loglog")]


def _check_km(km: gw.KaplanMeier, expected: dict[str, Any], label: str) -> None:
    assert_allclose_to_r(km.survival_, expected["surv"], what=f"{label} surv")
    assert_allclose_to_r(km.std_error_, expected["se"], what=f"{label} se(S)")
    assert_allclose_to_r(km.cumhaz_, expected["cumhaz"], what=f"{label} cumhaz")


def test_km_lung_overall_matches_r() -> None:
    df = gw.data.load_dataset("lung", backend="pandas")
    y = Surv.right(df["time"], event=(df["status"] == 2))
    expected = load_fixture("km_lung_overall")["overall"]

    for conf_type, key in _CONF:
        km = gw.KaplanMeier(conf_type=conf_type).fit(y)
        _check_km(km, expected, f"lung/{conf_type}")
        assert_allclose_to_r(km.conf_low_, expected[f"lower_{key}"], what=f"lung {conf_type} lower")
        assert_allclose_to_r(
            km.conf_high_, expected[f"upper_{key}"], what=f"lung {conf_type} upper"
        )


def test_km_median_matches_r() -> None:
    df = gw.data.load_dataset("lung", backend="pandas")
    y = Surv.right(df["time"], event=(df["status"] == 2))
    expected = load_fixture("km_lung_overall")["overall"]
    point, lower, upper = gw.KaplanMeier(conf_type="log").fit(y).median(ci=True)
    assert point == expected["median"]
    assert lower == expected["median_lower"]
    assert upper == expected["median_upper"]


def test_km_by_sex_matches_r() -> None:
    df = gw.data.load_dataset("lung", backend="pandas")
    y = Surv.right(df["time"], event=(df["status"] == 2))
    fixture = load_fixture("km_lung_by_sex")
    km = gw.KaplanMeier(conf_type="log-log").fit(y, by=df["sex"])
    for block in km._blocks:
        expected = fixture[str(block.label)]
        assert_allclose_to_r(block.surv, expected["surv"], what=f"sex={block.label} surv")
        assert_allclose_to_r(block.conf_low, expected["lower_loglog"], what="lower")
        assert_allclose_to_r(block.conf_high, expected["upper_loglog"], what="upper")


def test_km_veteran_overall_matches_r() -> None:
    df = gw.data.load_dataset("veteran", backend="pandas")
    y = Surv.right(df["time"], event=df["status"])
    km = gw.KaplanMeier(conf_type="log").fit(y)
    _check_km(km, load_fixture("km_veteran_overall")["overall"], "veteran")


def test_nelson_aalen_matches_r() -> None:
    df = gw.data.load_dataset("lung", backend="pandas")
    y = Surv.right(df["time"], event=(df["status"] == 2))
    expected = load_fixture("km_lung_overall")["overall"]
    na = gw.NelsonAalen().fit(y)
    assert_allclose_to_r(na.cumhaz_, expected["cumhaz"], what="NA cumhaz")
    assert_allclose_to_r(na.std_error_**2, expected["cumhaz_var"], what="NA cumhaz var")


def test_rmst_matches_r() -> None:
    fixture = load_fixture("rmst")
    for name, event_is_2 in [("lung", True), ("veteran", False)]:
        df = gw.data.load_dataset(name, backend="pandas")
        event = (df["status"] == 2) if event_is_2 else df["status"]
        y = Surv.right(df["time"], event=event)
        expected = fixture[name]
        value, lower, upper = gw.KaplanMeier().fit(y).rmst(expected["tau"], ci=True)
        assert_allclose_to_r(value, expected["rmst"], what=f"{name} rmst")
        # Recover the se from the symmetric normal interval and compare.
        z = 1.959963984540054
        assert_allclose_to_r((value - lower) / z, expected["se"], what=f"{name} rmst se")


# -- Log-rank / G-rho against survdiff ------------------------------------------


def _check_logrank(result: gw.TestResult, fixture: dict[str, Any], label: str) -> None:
    assert result.df == fixture["df"]
    assert_allclose_to_r(result.statistic, fixture["chisq"], what=f"{label} chisq")
    assert_allclose_to_r(result.p_value, fixture["p"], what=f"{label} p")
    for g, obs, exp in zip(fixture["groups"], fixture["obs"], fixture["exp"], strict=True):
        assert_allclose_to_r(result.observed[str(g)], obs, what=f"{label} observed[{g}]")
        assert_allclose_to_r(result.expected[str(g)], exp, what=f"{label} expected[{g}]")


def test_logrank_lung_sex_matches_r() -> None:
    df = gw.data.load_dataset("lung", backend="pandas")
    y = Surv.right(df["time"], event=(df["status"] == 2))
    result = gw.logrank_test(y, group=df["sex"].astype(str))
    _check_logrank(result, load_fixture("logrank_lung_sex"), "log-rank lung/sex")


def test_grho_lung_sex_rho1_matches_r() -> None:
    df = gw.data.load_dataset("lung", backend="pandas")
    y = Surv.right(df["time"], event=(df["status"] == 2))
    result = gw.logrank_test(y, group=df["sex"].astype(str), rho=1)
    _check_logrank(result, load_fixture("grho_lung_sex_rho1"), "G-rho lung/sex")


def test_logrank_veteran_celltype_matches_r() -> None:
    df = gw.data.load_dataset("veteran", backend="pandas")
    y = Surv.right(df["time"], event=df["status"])
    result = gw.logrank_test(y, group=df["celltype"])
    _check_logrank(result, load_fixture("logrank_veteran_celltype"), "log-rank veteran/celltype")


def _check_cox(cox: gw.CoxPH, fixture: dict[str, Any], label: str) -> None:
    assert cox.term_names_ == fixture["terms"]
    assert cox.n_ == fixture["n"]
    assert cox.n_event_ == fixture["nevent"]
    assert_allclose_to_r(cox.coef_, fixture["coef"], what=f"{label} coef")
    assert_allclose_to_r(cox.std_error_, fixture["se"], what=f"{label} se")
    assert_allclose_to_r(cox.z_, fixture["z"], rtol=1e-6, atol=1e-6, what=f"{label} z")
    assert_allclose_to_r(cox.p_value_, fixture["p"], rtol=1e-6, atol=1e-8, what=f"{label} p")
    assert_allclose_to_r(cox.hazard_ratio_, fixture["exp_coef"], what=f"{label} HR")
    assert_allclose_to_r(cox.loglik_null_, fixture["loglik_null"], atol=1e-6, what=f"{label} ll0")
    assert_allclose_to_r(cox.loglik_, fixture["loglik"], atol=1e-6, what=f"{label} ll")
    # HR confidence limits (exponentiated scale).
    tidy_exp = cox.to_dataframe(exponentiate=True)
    assert_allclose_to_r(tidy_exp["conf_low"].to_numpy(), fixture["conf_low"], what=f"{label} lo")
    assert_allclose_to_r(tidy_exp["conf_high"].to_numpy(), fixture["conf_high"], what=f"{label} hi")
    # Global tests: LR and score are exact; R stores the Wald test rounded to 2 decimals.
    assert_allclose_to_r(cox.lr_stat_, fixture["lr"]["stat"], atol=1e-4, what=f"{label} LR")
    assert_allclose_to_r(
        cox.score_stat_, fixture["score"]["stat"], atol=1e-4, what=f"{label} score"
    )
    assert_allclose_to_r(cox.wald_stat_, fixture["wald"]["stat"], atol=1e-2, what=f"{label} Wald")


def test_cox_lung_age_sex_efron_matches_r() -> None:
    df = gw.data.load_dataset("lung", backend="pandas")
    y = Surv.right(df["time"], event=(df["status"] == 2))
    cox = gw.CoxPH(ties="efron").fit(y, df[["age", "sex"]])
    _check_cox(cox, load_fixture("cox_lung_age_sex_efron"), "cox efron")


def test_cox_lung_age_sex_breslow_matches_r() -> None:
    df = gw.data.load_dataset("lung", backend="pandas")
    y = Surv.right(df["time"], event=(df["status"] == 2))
    cox = gw.CoxPH(ties="breslow").fit(y, df[["age", "sex"]])
    _check_cox(cox, load_fixture("cox_lung_age_sex_breslow"), "cox breslow")


def test_cox_lung_three_covariates_matches_r() -> None:
    df = gw.data.load_dataset("lung", backend="pandas")
    y = Surv.right(df["time"], event=(df["status"] == 2))
    cox = gw.CoxPH(ties="efron").fit(y, df[["age", "sex", "ph.ecog"]])
    _check_cox(cox, load_fixture("cox_lung_three_efron"), "cox three")


@pytest.mark.parametrize("ties", ["breslow", "efron"])
def test_cox_baseline_and_prediction_match_r(ties: str) -> None:
    import pandas as pd

    df = gw.data.load_dataset("lung", backend="pandas")
    y = Surv.right(df["time"], event=(df["status"] == 2))
    fixture = load_fixture(f"cox_diag_{ties}")
    cox = gw.CoxPH(ties=ties).fit(y, df[["age", "sex"]])

    assert_allclose_to_r(
        cox.baseline_hazard()["cumhaz"].to_numpy(), fixture["basehaz_cumhaz"], what="basehaz"
    )
    assert_allclose_to_r(cox.predict(type="lp"), fixture["lp"], what="lp")

    newdata = pd.DataFrame({"age": fixture["surv_newdata_age"], "sex": fixture["surv_newdata_sex"]})
    surv = cox.predict(newdata, type="survival", times=fixture["surv_times"])
    assert_allclose_to_r(surv["subject_1"].to_numpy(), fixture["surv"]["subj1"], what="surv 1")
    assert_allclose_to_r(surv["subject_2"].to_numpy(), fixture["surv"]["subj2"], what="surv 2")


@pytest.mark.parametrize("ties", ["breslow", "efron"])
def test_cox_schoenfeld_matches_r(ties: str) -> None:
    df = gw.data.load_dataset("lung", backend="pandas")
    y = Surv.right(df["time"], event=(df["status"] == 2))
    fixture = load_fixture(f"cox_diag_{ties}")
    sch = gw.CoxPH(ties=ties).fit(y, df[["age", "sex"]]).residuals("schoenfeld")
    # Row order within tied event times is arbitrary; compare as sorted columns.
    for col in ("age", "sex"):
        assert_allclose_to_r(
            np.sort(sch[col].to_numpy()),
            np.sort(fixture["schoenfeld"][col]),
            what=f"schoenfeld {col}",
        )


def test_cox_martingale_matches_r_breslow() -> None:
    df = gw.data.load_dataset("lung", backend="pandas")
    y = Surv.right(df["time"], event=(df["status"] == 2))
    fixture = load_fixture("cox_diag_breslow")
    resid = gw.CoxPH(ties="breslow").fit(y, df[["age", "sex"]]).residuals("martingale")
    assert_allclose_to_r(resid, fixture["martingale"], what="martingale")


@pytest.mark.parametrize("ties", ["breslow", "efron"])
@pytest.mark.parametrize("transform", ["identity", "log"])
def test_cox_zph_matches_r(ties: str, transform: str) -> None:
    df = gw.data.load_dataset("lung", backend="pandas")
    y = Surv.right(df["time"], event=(df["status"] == 2))
    fixture = load_fixture(f"cox_diag_{ties}")[f"zph_{transform}"]
    z = gw.CoxPH(ties=ties).fit(y, df[["age", "sex"]]).cox_zph(transform=transform)
    for term in ("age", "sex"):
        assert_allclose_to_r(z.per_term[term]["chisq"], fixture[term]["chisq"], what=f"zph {term}")
        assert_allclose_to_r(z.per_term[term]["p_value"], fixture[term]["p"], what=f"zph {term} p")
    assert_allclose_to_r(z.global_test["chisq"], fixture["global"]["chisq"], what="zph global")


@pytest.mark.parametrize("ties", ["breslow", "efron"])
def test_cox_concordance_matches_r(ties: str) -> None:
    df = gw.data.load_dataset("lung", backend="pandas")
    y = Surv.right(df["time"], event=(df["status"] == 2))
    fixture = load_fixture(f"cox_diag_{ties}")
    cox = gw.CoxPH(ties=ties).fit(y, df[["age", "sex"]])
    assert_allclose_to_r(cox.concordance(), fixture["concordance"], what="concordance")


def test_cox_stratified_matches_r() -> None:
    df = gw.data.load_dataset("lung", backend="pandas")
    y = Surv.right(df["time"], event=(df["status"] == 2))
    fixture = load_fixture("cox_strata")
    cox = gw.CoxPH().fit(y, df[["age", "ph.ecog"]], strata=df["sex"])
    assert cox.term_names_ == fixture["terms"]
    assert cox.n_ == fixture["n"]
    assert cox.n_event_ == fixture["nevent"]
    assert_allclose_to_r(cox.coef_, fixture["coef"], what="strata coef")
    assert_allclose_to_r(cox.std_error_, fixture["se"], what="strata se")
    assert_allclose_to_r(cox.loglik_, fixture["loglik"], atol=1e-6, what="strata loglik")
    assert_allclose_to_r(cox.lr_stat_, fixture["lr"], atol=1e-4, what="strata LR")
    assert_allclose_to_r(cox.score_stat_, fixture["score"], atol=1e-4, what="strata score")


def test_cox_robust_variance_matches_r() -> None:
    df = gw.data.load_dataset("lung", backend="pandas")
    y = Surv.right(df["time"], event=(df["status"] == 2))
    fixture = load_fixture("cox_robust")
    cox = gw.CoxPH(ties="breslow").fit(y, df[["age", "sex"]], robust=True)
    assert_allclose_to_r(cox.coef_, fixture["coef"], what="robust coef")
    assert_allclose_to_r(cox.naive_std_error_, fixture["naive_se"], what="naive se")
    assert_allclose_to_r(cox.std_error_, fixture["robust_se"], what="robust se")


def test_cox_cluster_robust_variance_matches_r() -> None:
    df = gw.data.load_dataset("lung", backend="pandas")
    y = Surv.right(df["time"], event=(df["status"] == 2))
    fixture = load_fixture("cox_cluster")
    cox = gw.CoxPH(ties="breslow").fit(y, df[["age", "sex"]], cluster=df["inst"])
    assert cox.n_ == fixture["n"]
    assert_allclose_to_r(cox.coef_, fixture["coef"], what="cluster coef")
    assert_allclose_to_r(cox.std_error_, fixture["robust_se"], what="cluster robust se")


@pytest.mark.parametrize("dist", ["weibull", "exponential", "lognormal", "loglogistic"])
def test_aft_matches_r_survreg(dist: str) -> None:
    df = gw.data.load_dataset("lung", backend="pandas")
    y = Surv.right(df["time"], event=(df["status"] == 2))
    fixture = load_fixture(f"aft_{dist}")
    model = gw.AFT(dist).fit(y, df[["age", "sex"]])
    assert model.term_names_ == fixture["terms"]
    # Coefficients agree with survreg to ~5 significant figures (optimizer tolerance).
    assert_allclose_to_r(model.coef_, fixture["coef"], atol=1e-4, what=f"{dist} coef")
    assert_allclose_to_r(model.std_error_, fixture["coef_se"], atol=1e-4, what=f"{dist} se")
    assert_allclose_to_r(model.scale_, fixture["scale"], atol=1e-4, what=f"{dist} scale")
    assert_allclose_to_r(model.loglik_, fixture["loglik"], atol=1e-3, what=f"{dist} loglik")
    if dist != "exponential":
        assert_allclose_to_r(
            model.log_scale_se_, fixture["log_scale_se"], atol=1e-4, what=f"{dist} log-scale se"
        )

    # Predicted survival-time quantiles match survreg's predict(type="quantile").
    # Agreement is at the coefficient optimizer floor (relative ~1e-5 on day-scale values).
    import pandas as pd

    newdata = pd.DataFrame(fixture["pred_newdata"])
    pred = model.predict(newdata, type="quantile", p=fixture["pred_p"])
    ours = pred[[c for c in pred.columns if c != "p"]].to_numpy().T  # subjects x p
    assert_allclose_to_r(
        ours.ravel(),
        np.array(fixture["pred_quantile"]).ravel(),
        rtol=1e-3,
        atol=1e-2,
        what=f"{dist} quantiles",
    )


def test_aalen_johansen_cif_matches_r() -> None:
    df = gw.data.load_dataset("mgus2", backend="pandas")
    etime = np.where(df["pstat"] == 1, df["ptime"], df["futime"])
    event = np.where(df["pstat"] == 1, 1, 2 * df["death"])  # 0 censor, 1 pcm, 2 death
    y = Surv.multistate(etime, event=event, states=("pcm", "death"))
    fixture = load_fixture("cif_mgus2")

    table = gw.AalenJohansen().fit(y).to_dataframe()
    pcm = table[table["cause"] == "pcm"].sort_values("time")
    death = table[table["cause"] == "death"].sort_values("time")

    assert_allclose_to_r(pcm["n_risk"].to_numpy(), fixture["n_risk"], what="n_risk")
    assert_allclose_to_r(pcm["estimate"].to_numpy(), fixture["cif_pcm"], what="CIF pcm")
    assert_allclose_to_r(death["estimate"].to_numpy(), fixture["cif_death"], what="CIF death")
    assert_allclose_to_r(pcm["std_error"].to_numpy(), fixture["se_pcm"], atol=1e-8, what="se pcm")
    assert_allclose_to_r(
        death["std_error"].to_numpy(), fixture["se_death"], atol=1e-8, what="se death"
    )


def test_finegray_matches_r() -> None:
    df = gw.data.load_dataset("mgus2", backend="pandas")
    etime = np.where(df["pstat"] == 1, df["ptime"], df["futime"])
    cause = np.where(df["pstat"] == 1, 1, 2 * df["death"])
    y = Surv.multistate(etime, event=cause, states=("pcm", "death"))
    fixture = load_fixture("finegray_mgus2_pcm")
    fg = gw.FineGray("pcm").fit(y, df[["age", "sex"]])
    assert fg.term_names_ == fixture["terms"]
    assert_allclose_to_r(fg.coef_, fixture["coef"], what="fine-gray coef")
    assert_allclose_to_r(fg.naive_std_error_, fixture["naive_se"], what="fine-gray naive se")
    assert_allclose_to_r(fg.std_error_, fixture["robust_se"], what="fine-gray robust se")


def _mgus2_illness_death():  # type: ignore[no-untyped-def]
    """Build the mgus -> pcm -> death counting-process intervals from mgus2."""
    df = gw.data.load_dataset("mgus2", backend="pandas")
    t0: list[float] = []
    t1: list[float] = []
    frm: list[str] = []
    evt: list[Any] = []
    for i in range(len(df)):
        pt, ft = df["ptime"][i], df["futime"][i]
        prog, died = df["pstat"][i] == 1, df["death"][i] == 1
        if prog and pt < ft:
            t0 += [0, pt]
            t1 += [pt, ft]
            frm += ["mgus", "pcm"]
            evt += ["pcm", "death" if died else None]
        else:
            t0 += [0]
            t1 += [ft]
            frm += ["mgus"]
            evt += ["death" if died else ("pcm" if prog else None)]
    keep = [b > a for a, b in zip(t0, t1, strict=True)]
    sel = lambda xs: [x for x, k in zip(xs, keep, strict=True) if k]  # noqa: E731
    return sel(t0), sel(t1), sel(frm), sel(evt)


def test_multistate_occupancy_matches_r() -> None:
    t0, t1, frm, evt = _mgus2_illness_death()
    fixture = load_fixture("multistate_mgus2")
    ms = gw.MultiState().fit(t0, t1, frm, evt, states=("mgus", "pcm", "death"))
    table = ms.to_dataframe()
    assert_allclose_to_r(table["time"].to_numpy(), fixture["time"], what="ms time")
    for state in ("mgus", "pcm", "death"):
        assert_allclose_to_r(table[state].to_numpy(), fixture[state], what=f"occupancy {state}")


def test_brier_score_matches_r() -> None:
    df = gw.data.load_dataset("lung", backend="pandas")
    y = Surv.right(df["time"], event=(df["status"] == 2))
    fixture = load_fixture("brier_lung")
    times = np.asarray(fixture["times"], dtype=float)
    cox = gw.CoxPH().fit(y, df[["age", "sex"]])
    pred = cox.predict(df[["age", "sex"]], type="survival", times=times)
    probs = pred[[f"subject_{i + 1}" for i in range(len(df))]].to_numpy().T
    assert_allclose_to_r(gw.brier_score(y, probs, times), fixture["brier"], what="brier")


def test_concordance_index_matches_r() -> None:
    # concordance_index of the Cox linear predictor equals the model's concordance.
    df = gw.data.load_dataset("lung", backend="pandas")
    y = Surv.right(df["time"], event=(df["status"] == 2))
    fixture = load_fixture("cox_diag_efron")
    cox = gw.CoxPH(ties="efron").fit(y, df[["age", "sex"]])
    c = gw.concordance_index(y, cox.predict(type="lp"))
    assert_allclose_to_r(c, fixture["concordance"], what="concordance_index")


def test_risk_table_numbers_match_r() -> None:
    # risk_table_data needs only numpy/pandas (no plotnine), so it runs here.
    fixture = load_fixture("risk_table_lung_sex")
    df = gw.data.load_dataset("lung", backend="pandas")
    y = Surv.right(df["time"], event=(df["status"] == 2))
    km = gw.KaplanMeier().fit(y, by=df["sex"])
    rtd = gw.viz.risk_table_data(km, times=fixture["times"])
    for label, expected in fixture["n_risk"].items():
        sub = rtd[rtd["strata"] == label].sort_values("time")
        assert_allclose_to_r(sub["n_risk"].to_numpy(), expected, what=f"n_risk sex={label}")
