"""R-parity validation of the risk-set / event-table kernel against `survfit`.

This proves the kernel matches R's `survival::survfit` tabulation (time, n.risk, n.event,
n.censor) on real datasets, including a left-truncation case. The Kaplan-Meier estimator
builds directly on this kernel.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
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
    df = gw.load_dataset("lung", backend="pandas")
    # survival::lung codes status 1 = censored, 2 = dead.
    y = Surv(time=df["time"], event=df["status"] == 2)
    et = event_table(y)
    _check_block(et, load_fixture("lung_km_overall")["overall"], "lung overall")


def test_lung_km_by_sex_matches_r() -> None:
    df = gw.load_dataset("lung", backend="pandas")
    y = Surv(time=df["time"], event=df["status"] == 2)
    et = event_table(y, group=df["sex"])
    fixture = load_fixture("lung_km_by_sex")

    assert et.strata is not None
    for level, block in fixture.items():
        mask = et.strata == f"sex={level}"
        sub = gw.EventTable(
            time=et.time[mask],
            n_risk=et.n_risk[mask],
            n_event=et.n_event[mask],
            n_censor=et.n_censor[mask],
        )
        _check_block(sub, block, f"lung sex={level}")


def test_veteran_km_overall_matches_r() -> None:
    df = gw.load_dataset("veteran", backend="pandas")
    y = Surv(time=df["time"], event=df["status"])
    et = event_table(y)
    _check_block(et, load_fixture("veteran_km_overall")["overall"], "veteran overall")


def test_counting_left_truncation_matches_r() -> None:
    fixture = load_fixture("counting_truncation")
    data = fixture["data"]
    y = Surv(time=data["start"], time2=data["stop"], event=data["event"])
    et = event_table(y)
    _check_block(et, fixture["overall"], "counting truncation")


# -- Kaplan-Meier / Nelson-Aalen against survfit --------------------------------

_CONF = [("plain", "plain"), ("log", "log"), ("log-log", "loglog")]


def _check_km(km: gw.KaplanMeier, expected: dict[str, Any], label: str) -> None:
    assert_allclose_to_r(km.survival_, expected["surv"], what=f"{label} surv")
    assert_allclose_to_r(km.std_error_, expected["se"], what=f"{label} se(S)")
    assert_allclose_to_r(km.cumhaz_, expected["cumhaz"], what=f"{label} cumhaz")


def test_km_lung_overall_matches_r() -> None:
    df = gw.load_dataset("lung", backend="pandas")
    y = Surv(time=df["time"], event=df["status"] == 2)
    expected = load_fixture("km_lung_overall")["overall"]

    for conf_type, key in _CONF:
        km = gw.KaplanMeier(conf_type=conf_type).fit(y)
        _check_km(km, expected, f"lung/{conf_type}")
        assert_allclose_to_r(km.conf_low_, expected[f"lower_{key}"], what=f"lung {conf_type} lower")
        assert_allclose_to_r(
            km.conf_high_, expected[f"upper_{key}"], what=f"lung {conf_type} upper"
        )


def test_km_median_matches_r() -> None:
    df = gw.load_dataset("lung", backend="pandas")
    y = Surv(time=df["time"], event=df["status"] == 2)
    expected = load_fixture("km_lung_overall")["overall"]
    med = gw.KaplanMeier(conf_type="log").fit(y).median(format="polars")
    assert med["strata"].to_list() == ["all"]
    assert med["time"][0] == expected["median"]
    assert med["conf_low"][0] == expected["median_lower"]
    assert med["conf_high"][0] == expected["median_upper"]


def test_km_by_sex_matches_r() -> None:
    df = gw.load_dataset("lung", backend="pandas")
    y = Surv(time=df["time"], event=df["status"] == 2)
    fixture = load_fixture("km_lung_by_sex")
    km = gw.KaplanMeier(conf_type="log-log").fit(y, by=df["sex"])
    for block in km._blocks:
        assert block.label.startswith("sex=")
        expected = fixture[block.label.removeprefix("sex=")]
        assert_allclose_to_r(block.surv, expected["surv"], what=f"{block.label} surv")
        assert_allclose_to_r(block.conf_low, expected["lower_loglog"], what="lower")
        assert_allclose_to_r(block.conf_high, expected["upper_loglog"], what="upper")


def test_km_veteran_overall_matches_r() -> None:
    df = gw.load_dataset("veteran", backend="pandas")
    y = Surv(time=df["time"], event=df["status"])
    km = gw.KaplanMeier(conf_type="log").fit(y)
    _check_km(km, load_fixture("km_veteran_overall")["overall"], "veteran")


def test_km_robust_lung_overall_matches_r() -> None:
    df = gw.load_dataset("lung", backend="pandas")
    y = Surv(time=df["time"], event=df["status"] == 2)
    expected = load_fixture("km_robust_lung_overall")["overall"]

    for conf_type, key in _CONF:
        km = gw.KaplanMeier(conf_type=conf_type, robust=True).fit(y)
        assert_allclose_to_r(km.survival_, expected["surv"], what=f"robust lung/{conf_type} surv")
        assert_allclose_to_r(km.std_error_, expected["se"], what=f"robust lung/{conf_type} se(S)")
        assert_allclose_to_r(
            km.conf_low_, expected[f"lower_{key}"], what=f"robust lung {conf_type} lower"
        )
        assert_allclose_to_r(
            km.conf_high_, expected[f"upper_{key}"], what=f"robust lung {conf_type} upper"
        )


def test_km_robust_weighted_matches_r() -> None:
    df = gw.load_dataset("lung", backend="pandas")
    y = Surv(time=df["time"], event=df["status"] == 2)
    fixture = load_fixture("km_robust_lung_weighted")
    weights = np.array(fixture["weights"])
    expected = fixture["overall"]

    km = gw.KaplanMeier(robust=True).fit(y, weights=weights)
    assert_allclose_to_r(km.survival_, expected["surv"], what="robust weighted surv")
    assert_allclose_to_r(km.std_error_, expected["se"], what="robust weighted se(S)")
    assert_allclose_to_r(km.conf_low_, expected["lower_log"], what="robust weighted lower")
    assert_allclose_to_r(km.conf_high_, expected["upper_log"], what="robust weighted upper")


def test_km_robust_by_sex_matches_r() -> None:
    df = gw.load_dataset("lung", backend="pandas")
    y = Surv(time=df["time"], event=df["status"] == 2)
    fixture = load_fixture("km_robust_lung_by_sex")
    km = gw.KaplanMeier(robust=True).fit(y, by=df["sex"])
    for block in km._blocks:
        assert block.label.startswith("sex=")
        expected = fixture[block.label.removeprefix("sex=")]
        assert_allclose_to_r(block.surv, expected["surv"], what=f"robust {block.label} surv")
        assert_allclose_to_r(block.std_error, expected["se"], what=f"robust {block.label} se(S)")
        assert_allclose_to_r(
            block.conf_low, expected["lower_log"], what=f"robust {block.label} lower"
        )
        assert_allclose_to_r(
            block.conf_high, expected["upper_log"], what=f"robust {block.label} upper"
        )


def test_km_cluster_lung_inst_matches_r() -> None:
    df = gw.load_dataset("lung", backend="pandas")
    keep = ~df["inst"].isna()
    y = Surv(time=df.loc[keep, "time"], event=df.loc[keep, "status"] == 2)
    cluster_ids = df.loc[keep, "inst"].values
    expected = load_fixture("km_cluster_lung_inst")["overall"]

    for conf_type, key in _CONF:
        km = gw.KaplanMeier(conf_type=conf_type).fit(y, cluster=cluster_ids)
        assert_allclose_to_r(km.survival_, expected["surv"], what=f"cluster lung/{conf_type} surv")
        assert_allclose_to_r(km.std_error_, expected["se"], what=f"cluster lung/{conf_type} se(S)")
        assert_allclose_to_r(
            km.conf_low_, expected[f"lower_{key}"], what=f"cluster lung {conf_type} lower"
        )
        assert_allclose_to_r(
            km.conf_high_, expected[f"upper_{key}"], what=f"cluster lung {conf_type} upper"
        )


def test_nelson_aalen_matches_r() -> None:
    df = gw.load_dataset("lung", backend="pandas")
    y = Surv(time=df["time"], event=df["status"] == 2)
    expected = load_fixture("km_lung_overall")["overall"]
    na = gw.NelsonAalen().fit(y)
    assert_allclose_to_r(na.cumhaz_, expected["cumhaz"], what="NA cumhaz")
    assert_allclose_to_r(na.std_error_**2, expected["cumhaz_var"], what="NA cumhaz var")


def test_rmst_matches_r() -> None:
    fixture = load_fixture("rmst")
    for name, event_is_2 in [("lung", True), ("veteran", False)]:
        df = gw.load_dataset(name, backend="pandas")
        event = (df["status"] == 2) if event_is_2 else df["status"]
        y = Surv(time=df["time"], event=event)
        expected = fixture[name]
        res = gw.KaplanMeier().fit(y).rmst(expected["tau"], format="polars")
        value, lower = float(res["estimate"][0]), float(res["conf_low"][0])
        assert_allclose_to_r(value, expected["rmst"], what=f"{name} rmst")
        # Recover the se from the symmetric normal interval and compare.
        z = 1.959963984540054
        assert_allclose_to_r((value - lower) / z, expected["se"], what=f"{name} rmst se")


def test_rmst_twogroup_matches_r() -> None:
    """Two-group rmst_test must match per-group survfit RMST values from R."""
    fix = load_fixture("rmst_twogroup_lung")
    df = gw.load_dataset("lung", backend="pandas")
    y = Surv(time=df["time"], event=df["status"] == 2)
    result = gw.rmst_test(y, tau=fix["tau"], group=df["sex"].astype(int))

    assert_allclose_to_r(result.rmst1, fix["rmst1"], what="rmst twogroup lung rmst1")
    assert_allclose_to_r(result.rmst2, fix["rmst2"], what="rmst twogroup lung rmst2")
    assert_allclose_to_r(result.se1, fix["se1"], what="rmst twogroup lung se1")
    assert_allclose_to_r(result.se2, fix["se2"], what="rmst twogroup lung se2")
    assert_allclose_to_r(result.estimate, fix["difference"], what="rmst twogroup lung difference")
    assert_allclose_to_r(result.se, fix["se_difference"], what="rmst twogroup lung se_difference")


def test_rmst_twogroup_stratified_matches_r() -> None:
    """Stratified rmst_test must match inverse-variance pooled values from R."""
    fix = load_fixture("rmst_twogroup_lung")["stratified"]
    df = gw.load_dataset("lung", backend="pandas")
    df_clean = df[df["ph.ecog"].notna()].copy()
    y = Surv(time=df_clean["time"], event=df_clean["status"] == 2)
    result = gw.rmst_test(
        y,
        tau=365,
        group=df_clean["sex"].astype(int),
        strata=df_clean["ph.ecog"].astype(int),
    )

    assert result.stratified is True
    assert_allclose_to_r(
        result.estimate, fix["difference"], atol=1e-6, what="stratified rmst difference"
    )
    assert_allclose_to_r(result.se, fix["se"], atol=1e-6, what="stratified rmst se")
    assert_allclose_to_r(
        result.statistic, fix["statistic"], atol=1e-6, what="stratified rmst z-statistic"
    )
    assert_allclose_to_r(result.p_value, fix["p_value"], atol=1e-6, what="stratified rmst p-value")


# -- Log-rank / G-rho against survdiff ------------------------------------------


def _check_logrank(result: gw.TestResult, fixture: dict[str, Any], label: str) -> None:
    assert result.df == fixture["df"]
    assert_allclose_to_r(result.statistic, fixture["chisq"], what=f"{label} chisq")
    assert_allclose_to_r(result.p_value, fixture["p"], what=f"{label} p")
    for g, obs, exp in zip(fixture["groups"], fixture["obs"], fixture["exp"], strict=True):
        assert_allclose_to_r(result.observed[str(g)], obs, what=f"{label} observed[{g}]")
        assert_allclose_to_r(result.expected[str(g)], exp, what=f"{label} expected[{g}]")


def test_logrank_lung_sex_matches_r() -> None:
    df = gw.load_dataset("lung", backend="pandas")
    y = Surv(time=df["time"], event=df["status"] == 2)
    result = gw.logrank_test(y, group=df["sex"].astype(str))
    _check_logrank(result, load_fixture("logrank_lung_sex"), "log-rank lung/sex")


def test_grho_lung_sex_rho1_matches_r() -> None:
    df = gw.load_dataset("lung", backend="pandas")
    y = Surv(time=df["time"], event=df["status"] == 2)
    result = gw.logrank_test(y, group=df["sex"].astype(str), rho=1)
    _check_logrank(result, load_fixture("grho_lung_sex_rho1"), "G-rho lung/sex")


def test_logrank_veteran_celltype_matches_r() -> None:
    df = gw.load_dataset("veteran", backend="pandas")
    y = Surv(time=df["time"], event=df["status"])
    result = gw.logrank_test(y, group=df["celltype"])
    _check_logrank(result, load_fixture("logrank_veteran_celltype"), "log-rank veteran/celltype")


def test_stratified_logrank_matches_r() -> None:
    df = gw.load_dataset("lung", backend="pandas")
    df = df[df["ph.ecog"].notna()]
    y = Surv(time=df["time"], event=df["status"] == 2)
    result = gw.logrank_test(y, group=df["sex"], strata=df["ph.ecog"])
    fixture = load_fixture("logrank_stratified_lung_sex_ecog")
    assert_allclose_to_r(result.statistic, fixture["chisq"], what="stratified logrank chisq")
    assert_allclose_to_r(result.p_value, fixture["p"], what="stratified logrank p")
    assert result.df == fixture["df"]


def test_trend_lung_ecog_matches_r() -> None:
    df = gw.load_dataset("lung", backend="pandas")
    df = df[df["ph.ecog"].notna()]
    y = Surv(time=df["time"], event=df["status"] == 2)
    fixture = load_fixture("trend_lung_ecog")
    cases: dict[str, dict[str, Any]] = {
        "linear": {},
        "quadratic": {"scores": [0, 1, 4, 9]},
        "peto": {"rho": 1},
        "stratified": {"strata": df["sex"]},
    }
    for name, options in cases.items():
        result = gw.trend_test(y, group=df["ph.ecog"], **options)
        expected = fixture[name]
        assert_allclose_to_r(
            list(result.expected.values()), expected["expected"], what=f"trend {name} E"
        )
        assert_allclose_to_r(result.statistic, expected["statistic"], what=f"trend {name} chisq")
        assert_allclose_to_r(result.p_value, expected["p"], what=f"trend {name} p")
    # Shifting the scores changes nothing.
    shifted = gw.trend_test(y, group=df["ph.ecog"], scores=[5, 6, 7, 8])
    assert_allclose_to_r(shifted.statistic, fixture["linear"]["statistic"], what="shifted scores")


def test_pairwise_logrank_matches_r() -> None:
    df = gw.load_dataset("veteran", backend="pandas")
    y = Surv(time=df["time"], event=df["status"])
    fixture = load_fixture("pairwise_logrank_veteran")
    for correction, key in [("holm", "holm"), ("bh", "bh"), ("bonferroni", "bonferroni")]:
        pw = gw.pairwise_logrank_test(y, df["celltype"], correction=correction, format="pandas")
        # The pairs come in level order. The fixture's are in sorted order, so match them up.
        pairs = zip(pw["group1"], pw["group2"], strict=True)
        row_of = {frozenset(pair): i for i, pair in enumerate(pairs)}
        rows = [
            row_of[frozenset(pair)]
            for pair in zip(fixture["group1"], fixture["group2"], strict=True)
        ]
        assert_allclose_to_r(
            pw["p_value"].to_numpy()[rows], fixture["p_value"], what="pairwise raw p"
        )
        assert_allclose_to_r(
            pw["p_adjusted"].to_numpy()[rows], fixture[key], what=f"pairwise {correction} p"
        )


@pytest.mark.rparity
def test_maxcombo_matches_r() -> None:
    df = gw.load_dataset("lung", backend="pandas")
    y = Surv(time=df["time"], event=df["status"] == 2)
    fixture = load_fixture("maxcombo_lung_sex")
    result = gw.maxcombo_test(y, group=df["sex"])

    weights_keys = [(0, 0), (1, 0), (0, 1), (1, 1)]
    fixture_z = [fixture["z_statistics"][f"{r}_{g}"] for r, g in weights_keys]
    result_z = [result.z_statistics[w] for w in weights_keys]
    assert_allclose_to_r(result_z, fixture_z, what="maxcombo Z")
    assert_allclose_to_r(
        result.correlation.ravel(),
        np.array(fixture["correlation"]).ravel(),
        what="maxcombo correlation",
    )
    assert_allclose_to_r(result.statistic, fixture["statistic"], what="maxcombo statistic")


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
    tidy_exp = cox.to_frame(format="pandas", exponentiate=True)
    assert_allclose_to_r(tidy_exp["conf_low"].to_numpy(), fixture["conf_low"], what=f"{label} lo")
    assert_allclose_to_r(tidy_exp["conf_high"].to_numpy(), fixture["conf_high"], what=f"{label} hi")
    # Global tests: LR and score are exact; R stores the Wald test rounded to 2 decimals.
    assert_allclose_to_r(cox.lr_stat_, fixture["lr"]["stat"], atol=1e-4, what=f"{label} LR")
    assert_allclose_to_r(
        cox.score_stat_, fixture["score"]["stat"], atol=1e-4, what=f"{label} score"
    )
    assert_allclose_to_r(cox.wald_stat_, fixture["wald"]["stat"], atol=1e-2, what=f"{label} Wald")


def test_cox_lung_age_sex_efron_matches_r() -> None:
    df = gw.load_dataset("lung", backend="pandas")
    y = Surv(time=df["time"], event=df["status"] == 2)
    cox = gw.CoxPH(ties="efron").fit(y, df[["age", "sex"]])
    _check_cox(cox, load_fixture("cox_lung_age_sex_efron"), "cox efron")


def test_cox_lung_age_sex_breslow_matches_r() -> None:
    df = gw.load_dataset("lung", backend="pandas")
    y = Surv(time=df["time"], event=df["status"] == 2)
    cox = gw.CoxPH(ties="breslow").fit(y, df[["age", "sex"]])
    _check_cox(cox, load_fixture("cox_lung_age_sex_breslow"), "cox breslow")


def test_cox_lung_three_covariates_matches_r() -> None:
    df = gw.load_dataset("lung", backend="pandas")
    y = Surv(time=df["time"], event=df["status"] == 2)
    cox = gw.CoxPH(ties="efron").fit(y, df[["age", "sex", "ph.ecog"]])
    _check_cox(cox, load_fixture("cox_lung_three_efron"), "cox three")


@pytest.mark.parametrize("ties", ["breslow", "efron"])
def test_cox_baseline_and_prediction_match_r(ties: str) -> None:
    import pandas as pd

    df = gw.load_dataset("lung", backend="pandas")
    y = Surv(time=df["time"], event=df["status"] == 2)
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
    df = gw.load_dataset("lung", backend="pandas")
    y = Surv(time=df["time"], event=df["status"] == 2)
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
    df = gw.load_dataset("lung", backend="pandas")
    y = Surv(time=df["time"], event=df["status"] == 2)
    fixture = load_fixture("cox_diag_breslow")
    resid = gw.CoxPH(ties="breslow").fit(y, df[["age", "sex"]]).residuals("martingale")
    assert_allclose_to_r(resid, fixture["martingale"], what="martingale")


@pytest.mark.parametrize("ties", ["breslow", "efron"])
@pytest.mark.parametrize("transform", ["identity", "log", "km", "rank"])
def test_cox_zph_matches_r(ties: str, transform: str) -> None:
    df = gw.load_dataset("lung", backend="pandas")
    y = Surv(time=df["time"], event=df["status"] == 2)
    fixture = load_fixture(f"cox_diag_{ties}")[f"zph_{transform}"]
    z = gw.CoxPH(ties=ties).fit(y, df[["age", "sex"]]).cox_zph(transform=transform)
    for term in ("age", "sex"):
        assert_allclose_to_r(z.per_term[term]["chisq"], fixture[term]["chisq"], what=f"zph {term}")
        assert_allclose_to_r(z.per_term[term]["p_value"], fixture[term]["p"], what=f"zph {term} p")
    assert_allclose_to_r(z.global_test["chisq"], fixture["global"]["chisq"], what="zph global")


def test_cox_zph_windowed_matches_r() -> None:
    df = gw.load_dataset("lung", backend="pandas")
    y = Surv(time=df["time"], event=df["status"] == 2)
    fixture = load_fixture("cox_zph_windowed")
    z = gw.CoxPH().fit(y, df[["age", "sex"]]).cox_zph(breaks=[180, 365])
    assert z.windows is not None
    assert len(z.windows) == len(fixture["windows"])
    for w, fw in zip(z.windows, fixture["windows"], strict=True):
        assert w.n_events == fw["n_events"]
        for term in ("age", "sex"):
            assert_allclose_to_r(
                w.per_term[term]["chisq"],
                fw["per_term"][term]["chisq"],
                what=f"zph window {fw['interval']} {term}",
            )
            assert_allclose_to_r(
                w.per_term[term]["p_value"],
                fw["per_term"][term]["p_value"],
                what=f"zph window {fw['interval']} {term} p",
            )
        assert_allclose_to_r(
            w.global_test["chisq"],
            fw["global_test"]["chisq"],
            what=f"zph window {fw['interval']} global",
        )


@pytest.mark.parametrize("ties", ["breslow", "efron"])
def test_cox_concordance_matches_r(ties: str) -> None:
    df = gw.load_dataset("lung", backend="pandas")
    y = Surv(time=df["time"], event=df["status"] == 2)
    fixture = load_fixture(f"cox_diag_{ties}")
    cox = gw.CoxPH(ties=ties).fit(y, df[["age", "sex"]])
    assert_allclose_to_r(cox.concordance(), fixture["concordance"], what="concordance")


def test_cox_stratified_matches_r() -> None:
    df = gw.load_dataset("lung", backend="pandas")
    y = Surv(time=df["time"], event=df["status"] == 2)
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
    df = gw.load_dataset("lung", backend="pandas")
    y = Surv(time=df["time"], event=df["status"] == 2)
    fixture = load_fixture("cox_robust")
    cox = gw.CoxPH(ties="breslow").fit(y, df[["age", "sex"]], robust=True)
    assert_allclose_to_r(cox.coef_, fixture["coef"], what="robust coef")
    assert_allclose_to_r(cox.naive_std_error_, fixture["naive_se"], what="naive se")
    assert_allclose_to_r(cox.std_error_, fixture["robust_se"], what="robust se")


def test_cox_cluster_robust_variance_matches_r() -> None:
    df = gw.load_dataset("lung", backend="pandas")
    y = Surv(time=df["time"], event=df["status"] == 2)
    fixture = load_fixture("cox_cluster")
    cox = gw.CoxPH(ties="breslow").fit(y, df[["age", "sex"]], cluster=df["inst"])
    assert cox.n_ == fixture["n"]
    assert_allclose_to_r(cox.coef_, fixture["coef"], what="cluster coef")
    assert_allclose_to_r(cox.std_error_, fixture["robust_se"], what="cluster robust se")


def test_cox_time_varying_matches_r() -> None:
    # Counting-process (start, stop] Cox with a time-varying covariate, vs R coxph on `heart`.
    import pandas as pd

    fixture = load_fixture("cox_timevarying")
    y = Surv(time=fixture["start"], time2=fixture["stop"], event=fixture["event"])
    x = pd.DataFrame(
        {"age": fixture["age"], "surgery": fixture["surgery"], "transplant": fixture["transplant"]}
    )
    cox = gw.CoxPH(ties="breslow").fit(y, x)
    assert cox.n_ == fixture["n"] and cox.n_event_ == fixture["nevent"]
    assert_allclose_to_r(cox.coef_, fixture["coef"], what="time-varying coef")
    assert_allclose_to_r(cox.std_error_, fixture["se"], what="time-varying se")
    assert_allclose_to_r(cox.loglik_, fixture["loglik"], what="time-varying loglik")


@pytest.mark.parametrize("ties", ["efron", "breslow"])
def test_cox_weights_and_residuals_match_r(ties: str) -> None:
    # Case weights, strata, and counting-process data with the robust variance and the
    # martingale, score, and dfbeta residuals. Efron's tie correction enters every one of them.
    import pandas as pd

    fixture = load_fixture("cox_weighted_residuals")
    lung = gw.load_dataset("lung", backend="pandas")
    lung["w"] = fixture["weights"]
    lung = lung.dropna(subset=["time", "status", "age", "sex", "ph.ecog"])
    heart = pd.DataFrame(fixture["heart"])
    fits = {
        "plain": gw.CoxPH(ties=ties).fit("Surv(time, status) ~ age + sex", data=lung),
        "weighted": gw.CoxPH(ties=ties).fit(
            "Surv(time, status) ~ age + sex", data=lung, weights="w", robust=True
        ),
        "strata": gw.CoxPH(ties=ties).fit(
            "Surv(time, status) ~ age + ph.ecog + strata(sex)", data=lung, weights="w", robust=True
        ),
        "counting": gw.CoxPH(ties=ties).fit(
            "Surv(start, stop, event) ~ age + surgery + transplant", data=heart, cluster="id"
        ),
    }
    for name, cox in fits.items():
        expected = fixture[ties][name]
        what = f"{ties} {name}"
        assert_allclose_to_r(cox.coef_, expected["coef"], what=f"{what} coef")
        assert_allclose_to_r(cox.std_error_, expected["se"], what=f"{what} se")
        assert_allclose_to_r(cox.naive_std_error_, expected["naive_se"], what=f"{what} naive se")
        assert_allclose_to_r(
            [cox.loglik_null_, cox.loglik_], expected["loglik"], what=f"{what} loglik"
        )
        assert_allclose_to_r(cox.score_stat_, expected["score_test"], what=f"{what} score test")
        assert_allclose_to_r(
            np.asarray(cox.residuals("martingale")), expected["martingale"], what=f"{what} mart"
        )
        for kind in ("score", "dfbeta"):
            values = cox.residuals(kind, format="pandas").to_numpy()
            # jsonlite writes a matrix as a list of rows.
            rows = np.array(expected[kind], dtype=float)
            assert_allclose_to_r(values.ravel(), rows.ravel(), what=f"{what} {kind}")


def test_cox_survival_ci_matches_r() -> None:
    import pandas as pd

    df = gw.load_dataset("lung", backend="pandas")
    y = Surv(time=df["time"], event=df["status"] == 2)
    fixture = load_fixture("cox_survci_breslow")
    cox = gw.CoxPH(ties="breslow").fit(y, df[["age", "sex"]])
    newdata = pd.DataFrame({"age": fixture["newdata_age"], "sex": fixture["newdata_sex"]})
    pred = cox.predict(newdata, type="survival", times=fixture["times"], ci=True)
    for j, subj in enumerate(("subj1", "subj2"), start=1):
        assert_allclose_to_r(
            pred[f"subject_{j}"].to_numpy(), fixture["surv"][subj], what=f"{subj} surv"
        )
        assert_allclose_to_r(
            pred[f"subject_{j}_lower"].to_numpy(), fixture["lower"][subj], what=f"{subj} lower"
        )
        assert_allclose_to_r(
            pred[f"subject_{j}_upper"].to_numpy(), fixture["upper"][subj], what=f"{subj} upper"
        )


def test_cox_baseline_hazard_ci_matches_r() -> None:
    df = gw.load_dataset("lung", backend="pandas")
    y = Surv(time=df["time"], event=df["status"] == 2)
    fixture = load_fixture("cox_basehaz_ci")
    cox = gw.CoxPH(ties="breslow").fit(y, df[["age", "sex"]])
    bh = cox.baseline_hazard(ci=True, conf_type="log-log", format="pandas")
    times = np.asarray(fixture["times"])
    idx = np.searchsorted(bh["time"].to_numpy(), times, side="right") - 1
    assert_allclose_to_r(bh["cumhaz"].to_numpy()[idx], fixture["cumhaz"], what="basehaz cumhaz")
    assert_allclose_to_r(
        bh["survival"].to_numpy()[idx], fixture["survival"], what="basehaz survival"
    )
    assert_allclose_to_r(
        bh["survival_lower"].to_numpy()[idx],
        fixture["lower_loglog"],
        what="basehaz survival lower (log-log)",
    )
    assert_allclose_to_r(
        bh["survival_upper"].to_numpy()[idx],
        fixture["upper_loglog"],
        what="basehaz survival upper (log-log)",
    )


def test_cox_residuals_breslow_match_r() -> None:
    df = gw.load_dataset("lung", backend="pandas")
    y = Surv(time=df["time"], event=df["status"] == 2)
    fixture = load_fixture("cox_residuals_breslow")
    cox = gw.CoxPH(ties="breslow").fit(y, df[["age", "sex"]])

    dev = cox.residuals("deviance")
    assert_allclose_to_r(dev, fixture["deviance"], what="deviance residuals")

    score = cox.residuals("score", format="pandas")
    assert_allclose_to_r(score["age"].to_numpy(), fixture["score_age"], what="score age")
    assert_allclose_to_r(score["sex"].to_numpy(), fixture["score_sex"], what="score sex")

    dfb = cox.residuals("dfbeta", format="pandas")
    assert_allclose_to_r(dfb["age"].to_numpy(), fixture["dfbeta_age"], what="dfbeta age")
    assert_allclose_to_r(dfb["sex"].to_numpy(), fixture["dfbeta_sex"], what="dfbeta sex")

    dfbs = cox.residuals("dfbetas", format="pandas")
    assert_allclose_to_r(dfbs["age"].to_numpy(), fixture["dfbetas_age"], what="dfbetas age")
    assert_allclose_to_r(dfbs["sex"].to_numpy(), fixture["dfbetas_sex"], what="dfbetas sex")

    ssch = cox.residuals("scaledsch", format="pandas")
    assert_allclose_to_r(
        ssch["age"].to_numpy(), fixture["scaledsch_age"], what="scaled schoenfeld age"
    )
    assert_allclose_to_r(
        ssch["sex"].to_numpy(), fixture["scaledsch_sex"], what="scaled schoenfeld sex"
    )

    lev = cox.residuals("leverage")
    assert_allclose_to_r(lev, fixture["leverage"], what="leverage")


@pytest.mark.parametrize("dist", ["weibull", "exponential", "lognormal", "loglogistic"])
def test_aft_matches_r_survreg(dist: str) -> None:
    df = gw.load_dataset("lung", backend="pandas")
    y = Surv(time=df["time"], event=df["status"] == 2)
    fixture = load_fixture(f"aft_{dist}")
    model = gw.AFT(dist).fit(y, df[["age", "sex"]])
    assert model.term_names_ == fixture["terms"]
    # Newton-Raphson and survreg reach the same maximum, so the fits agree to about 1e-10.
    assert_allclose_to_r(model.coef_, fixture["coef"], what=f"{dist} coef")
    assert_allclose_to_r(model.std_error_, fixture["coef_se"], what=f"{dist} se")
    assert_allclose_to_r(model.scale_, fixture["scale"], what=f"{dist} scale")
    assert_allclose_to_r(model.loglik_, fixture["loglik"], what=f"{dist} loglik")
    if dist != "exponential":
        assert_allclose_to_r(
            model.log_scale_se_, fixture["log_scale_se"], what=f"{dist} log-scale se"
        )

    # Predicted survival-time quantiles match survreg's predict(type="quantile").
    newdata = pd.DataFrame(fixture["pred_newdata"])
    pred = model.predict(newdata, type="quantile", p=fixture["pred_p"])
    ours = pred[[c for c in pred.columns if c != "p"]].to_numpy().T  # subjects x p
    assert_allclose_to_r(
        ours.ravel(), np.array(fixture["pred_quantile"]).ravel(), what=f"{dist} quantiles"
    )

    # The limits come from the delta method with the full variance, scale included:
    # q * exp(+/- z * se(q) / q), with se(q) from survreg's predict(se.fit = TRUE).
    q_r = np.array(fixture["pred_quantile"])
    se_r = np.array(fixture["pred_quantile_se"])
    z = 1.959963984540054
    with_ci = model.predict_quantile(newdata, p=fixture["pred_p"], ci=True, format="pandas")
    n_subj = q_r.shape[0]
    for bound, sign in (("lower", -1.0), ("upper", 1.0)):
        ours_ci = np.array([with_ci[f"subject_{i + 1}_{bound}"] for i in range(n_subj)])
        assert_allclose_to_r(
            ours_ci.ravel(),
            (q_r * np.exp(sign * z * se_r / q_r)).ravel(),
            what=f"{dist} quantile {bound} limits",
        )

    # Survival and restricted-mean standard errors also use the full variance. The fixture's
    # gradients are numerical (numDeriv), so they agree to about 1e-8.
    _assert_survival_and_rmst_match(model, fixture, label=dist)


def _assert_survival_and_rmst_match(
    model: Any, fixture: dict[str, Any], *, label: str, rtol: float = 1e-8
) -> None:
    """Survival at `pred_times` and the restricted mean to `tau`, with their standard errors.

    The plain survival limits are `S +/- z se(S)`, and the restricted-mean limits
    `rmst * exp(+/- z se / rmst)`, so both carry the fixture's standard errors. `rtol` is for the
    estimates, and the limits get ten times as much.
    """
    newdata = pd.DataFrame(fixture["pred_newdata"])
    n_subj = len(newdata)
    z = 1.959963984540054
    s_r = np.array(fixture["pred_survival"])
    s_se_r = np.array(fixture["pred_survival_se"])
    s = model.predict(
        newdata,
        type="survival",
        times=fixture["pred_times"],
        ci=True,
        conf_type="plain",
        format="pandas",
    )
    ours_s = np.array([s[f"subject_{i + 1}"] for i in range(n_subj)])
    assert_allclose_to_r(ours_s.ravel(), s_r.ravel(), rtol=rtol, what=f"{label} survival")
    ours_low = np.array([s[f"subject_{i + 1}_lower"] for i in range(n_subj)])
    assert_allclose_to_r(
        ours_low.ravel(),
        (s_r - z * s_se_r).ravel(),
        rtol=10 * rtol,
        what=f"{label} survival limits",
    )

    rmst_r = np.array(fixture["rmst"])
    rmst_se_r = np.array(fixture["rmst_se"])
    e = model.predict_expectation(newdata, tau=fixture["tau"], ci=True, format="pandas")
    assert_allclose_to_r(
        [e[f"subject_{i + 1}"][0] for i in range(n_subj)], rmst_r, rtol=rtol, what=f"{label} rmst"
    )
    for bound, sign in (("lower", -1.0), ("upper", 1.0)):
        assert_allclose_to_r(
            [e[f"subject_{i + 1}_{bound}"][0] for i in range(n_subj)],
            rmst_r * np.exp(sign * z * rmst_se_r / rmst_r),
            rtol=10 * rtol,
            what=f"{label} rmst {bound} limits",
        )


def test_aft_gengamma_matches_r() -> None:
    """The generalized gamma against flexsurv's likelihood, maximized to about 1e-8.

    The fixture comes from `scripts/generate_gengamma_fixture.R`. Its derivatives are numerical,
    so the variance and the standard errors agree to about 1e-7.
    """
    df = gw.load_dataset("lung", backend="pandas")
    y = Surv(time=df["time"], event=df["status"] == 2)
    fixture = load_fixture("aft_gengamma")
    model = gw.AFT("gengamma").fit(y, df[["age", "sex"]])
    tol = {"rtol": 1e-7, "atol": 0.0}
    assert model.term_names_ == fixture["terms"]
    assert_allclose_to_r(model.coef_, fixture["coef"], **tol, what="gengamma coef")
    assert_allclose_to_r(model.std_error_, fixture["coef_se"], rtol=1e-6, what="gengamma se")
    assert_allclose_to_r(model.scale_, fixture["scale"], **tol, what="gengamma scale")
    assert_allclose_to_r(model.Q_, fixture["Q"], **tol, what="gengamma Q")
    assert_allclose_to_r(model.Q_se_, fixture["Q_se"], rtol=1e-6, what="gengamma Q se")
    assert_allclose_to_r(
        model.log_scale_se_, fixture["log_scale_se"], rtol=1e-6, what="gengamma log-scale se"
    )
    assert_allclose_to_r(
        model.vcov_.ravel(),
        np.array(fixture["vcov"]).ravel(),
        rtol=1e-5,
        atol=1e-9,
        what="gengamma vcov",
    )
    assert_allclose_to_r(model.loglik_, fixture["loglik"], what="gengamma loglik")

    newdata = pd.DataFrame(fixture["pred_newdata"])
    n_subj = len(newdata)
    q = model.predict_quantile(newdata, p=fixture["pred_p"], ci=True, format="pandas")
    ours_q = np.array([q[f"subject_{i + 1}"] for i in range(n_subj)])
    assert_allclose_to_r(
        ours_q.ravel(), np.array(fixture["pred_quantile"]).ravel(), **tol, what="gengamma quantiles"
    )
    q_r = np.array(fixture["pred_quantile"])
    se_r = np.array(fixture["pred_quantile_se"])
    z = 1.959963984540054
    for bound, sign in (("lower", -1.0), ("upper", 1.0)):
        ours_ci = np.array([q[f"subject_{i + 1}_{bound}"] for i in range(n_subj)])
        assert_allclose_to_r(
            ours_ci.ravel(),
            (q_r * np.exp(sign * z * se_r / q_r)).ravel(),
            rtol=1e-7,
            atol=0.0,
            what=f"gengamma quantile {bound} limits",
        )

    s = model.predict(newdata, type="survival", times=fixture["pred_times"], format="pandas")
    ours_s = np.array([s[f"subject_{i + 1}"] for i in range(n_subj)])
    assert_allclose_to_r(
        ours_s.ravel(), np.array(fixture["pred_survival"]).ravel(), **tol, what="gengamma survival"
    )
    assert_allclose_to_r(
        model.predict(newdata, type="mean"), fixture["mean"], **tol, what="gengamma mean"
    )
    assert_allclose_to_r(
        model.predict(newdata, type="rmst", tau=fixture["tau"]),
        fixture["rmst"],
        **tol,
        what="gengamma rmst",
    )
    _assert_survival_and_rmst_match(model, fixture, label="gengamma", rtol=1e-7)
    assert_allclose_to_r(
        model.residuals("cox_snell"), fixture["cox_snell"], **tol, what="gengamma cox-snell"
    )
    # The fixture's score has a column per parameter, and the residuals one per coefficient. Each
    # column agrees to about 1e-8 of its largest value.
    score = model.residuals("score", format="pandas").to_numpy()
    expected_score = np.array(fixture["score"])[:, :3]
    for j, term in enumerate(model.term_names_):
        assert_allclose_to_r(
            score[:, j],
            expected_score[:, j],
            rtol=0.0,
            atol=1e-7 * float(np.max(np.abs(expected_score[:, j]))),
            what=f"gengamma score {term}",
        )

    nested = model.test_distributions(format="pandas")
    expected = fixture["nested"]
    assert list(nested["dist"]) == [row["dist"] for row in expected]
    assert list(nested["df"]) == [row["df"] for row in expected]
    for column in ("loglik", "aic"):
        assert_allclose_to_r(
            nested[column].to_numpy(),
            [row[column] for row in expected],
            what=f"gengamma nested {column}",
        )
    for column in ("lr_statistic", "p_value"):
        assert_allclose_to_r(
            nested[column].to_numpy(),
            [row[column] for row in expected],
            rtol=1e-6,
            atol=0.0,
            what=f"gengamma nested {column}",
        )


@pytest.mark.parametrize("dist", ["weibull", "exponential", "lognormal", "loglogistic"])
def test_aft_residuals_match_r(dist: str) -> None:
    df = gw.load_dataset("lung", backend="pandas")
    y = Surv(time=df["time"], event=df["status"] == 2)
    fixture = load_fixture(f"aft_residuals_{dist}")
    model = gw.AFT(dist).fit(y, df[["age", "sex"]])

    resp = model.residuals("response")
    assert_allclose_to_r(resp, fixture["response"], atol=1e-2, what=f"{dist} response")

    dev = model.residuals("deviance")
    assert_allclose_to_r(dev, fixture["deviance"], atol=1e-4, what=f"{dist} deviance")

    dfb = model.residuals("dfbeta", format="pandas")
    assert_allclose_to_r(
        dfb["(Intercept)"].to_numpy(),
        fixture["dfbeta_intercept"],
        atol=2e-2,
        what=f"{dist} dfbeta intercept",
    )
    assert_allclose_to_r(
        dfb["age"].to_numpy(), fixture["dfbeta_age"], atol=1e-3, what=f"{dist} dfbeta age"
    )
    assert_allclose_to_r(
        dfb["sex"].to_numpy(), fixture["dfbeta_sex"], atol=2e-2, what=f"{dist} dfbeta sex"
    )

    dfbs = model.residuals("dfbetas", format="pandas")
    assert_allclose_to_r(
        dfbs["(Intercept)"].to_numpy(),
        fixture["dfbetas_intercept"],
        atol=1e-1,
        what=f"{dist} dfbetas intercept",
    )
    assert_allclose_to_r(
        dfbs["age"].to_numpy(), fixture["dfbetas_age"], atol=1e-1, what=f"{dist} dfbetas age"
    )
    assert_allclose_to_r(
        dfbs["sex"].to_numpy(), fixture["dfbetas_sex"], atol=1e-1, what=f"{dist} dfbetas sex"
    )


def test_aalen_johansen_cif_matches_r() -> None:
    df = gw.load_dataset("mgus2", backend="pandas")
    etime = np.where(df["pstat"] == 1, df["ptime"], df["futime"])
    event = np.where(df["pstat"] == 1, 1, 2 * df["death"])  # 0 censor, 1 pcm, 2 death
    y = Surv(
        time=etime,
        event=pd.Categorical.from_codes(event, categories=["censor", "pcm", "death"]),
    )
    fixture = load_fixture("cif_mgus2")

    table = gw.AalenJohansen().fit(y).to_frame(format="pandas")
    pcm = table[table["cause"] == "pcm"].sort_values("time")
    death = table[table["cause"] == "death"].sort_values("time")

    assert_allclose_to_r(pcm["n_risk"].to_numpy(), fixture["n_risk"], what="n_risk")
    assert_allclose_to_r(pcm["estimate"].to_numpy(), fixture["cif_pcm"], what="CIF pcm")
    assert_allclose_to_r(death["estimate"].to_numpy(), fixture["cif_death"], what="CIF death")
    assert_allclose_to_r(pcm["std_error"].to_numpy(), fixture["se_pcm"], atol=1e-8, what="se pcm")
    assert_allclose_to_r(
        death["std_error"].to_numpy(), fixture["se_death"], atol=1e-8, what="se death"
    )


_CIF_CONF = [("plain", "plain"), ("log", "log"), ("log-log", "loglog")]


def test_aalen_johansen_cif_ci_matches_r() -> None:
    df = gw.load_dataset("mgus2", backend="pandas")
    etime = np.where(df["pstat"] == 1, df["ptime"], df["futime"])
    event = np.where(df["pstat"] == 1, 1, 2 * df["death"])
    y = Surv(
        time=etime,
        event=pd.Categorical.from_codes(event, categories=["censor", "pcm", "death"]),
    )
    fixture = load_fixture("cif_mgus2")

    for conf_type, key in _CIF_CONF:
        table = gw.AalenJohansen(conf_type=conf_type).fit(y).to_frame(format="pandas")
        pcm = table[table["cause"] == "pcm"].sort_values("time")
        death = table[table["cause"] == "death"].sort_values("time")
        assert_allclose_to_r(
            pcm["conf_low"].to_numpy(),
            fixture[f"lower_pcm_{key}"],
            atol=1e-8,
            what=f"pcm lower {conf_type}",
        )
        assert_allclose_to_r(
            pcm["conf_high"].to_numpy(),
            fixture[f"upper_pcm_{key}"],
            atol=1e-8,
            what=f"pcm upper {conf_type}",
        )
        assert_allclose_to_r(
            death["conf_low"].to_numpy(),
            fixture[f"lower_death_{key}"],
            atol=1e-8,
            what=f"death lower {conf_type}",
        )
        assert_allclose_to_r(
            death["conf_high"].to_numpy(),
            fixture[f"upper_death_{key}"],
            atol=1e-8,
            what=f"death upper {conf_type}",
        )


def test_calibration_incidence_reads_aj_steps() -> None:
    df = gw.load_dataset("mgus2", backend="pandas")
    etime = np.where(df["pstat"] == 1, df["ptime"], df["futime"])
    cause = np.where(df["pstat"] == 1, 1, 2 * df["death"])
    y = Surv(
        time=etime,
        event=pd.Categorical.from_codes(cause, categories=["censor", "pcm", "death"]),
    )
    fixture = load_fixture("calibration_incidence_mgus2")
    times = np.array(fixture["times"])
    # With every prediction 0, the error is the Aalen-Johansen reference itself, read as a
    # right-continuous step function between the event times.
    zeros = np.zeros((y.n, times.shape[0]))
    err = gw.calibration_incidence(y, zeros, times, cause=1)
    assert_allclose_to_r(err, fixture["cif_pcm"], what="calibration_incidence reference")


def test_finegray_matches_r() -> None:
    df = gw.load_dataset("mgus2", backend="pandas")
    etime = np.where(df["pstat"] == 1, df["ptime"], df["futime"])
    cause = np.where(df["pstat"] == 1, 1, 2 * df["death"])
    y = Surv(
        time=etime,
        event=pd.Categorical.from_codes(cause, categories=["censor", "pcm", "death"]),
    )
    fixture = load_fixture("finegray_mgus2_pcm")
    fg = gw.FineGray("pcm").fit(y, df[["age", "sex"]])
    assert fg.term_names_ == fixture["terms"]
    assert_allclose_to_r(fg.coef_, fixture["coef"], what="fine-gray coef")
    assert_allclose_to_r(fg.naive_std_error_, fixture["naive_se"], what="fine-gray naive se")
    assert_allclose_to_r(fg.std_error_, fixture["robust_se"], what="fine-gray robust se")

    # The predicted CIF is 1 - S(t | x) from the same weighted Cox fit.
    newdata = pd.DataFrame({"age": fixture["newdata_age"], "sex": fixture["newdata_sex"]})
    cif = fg.predict_cumulative_incidence(newdata, times=fixture["times"], format="pandas")
    assert_allclose_to_r(cif["subject_1"], fixture["cif"]["subj1"], what="fine-gray cif 1")
    assert_allclose_to_r(cif["subject_2"], fixture["cif"]["subj2"], what="fine-gray cif 2")


def _mgus2_illness_death():  # type: ignore[no-untyped-def]
    """Build the mgus -> pcm -> death counting-process intervals from mgus2."""
    df = gw.load_dataset("mgus2", backend="pandas")
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


def test_grays_test_matches_r() -> None:
    df = gw.load_dataset("mgus2", backend="pandas")
    etime = np.where(df["pstat"] == 1, df["ptime"], df["futime"])
    cause = np.where(df["pstat"] == 1, 1, 2 * df["death"])
    y = Surv(
        time=etime,
        event=pd.Categorical.from_codes(cause, categories=["censor", "pcm", "death"]),
    )
    fixture = load_fixture("grays_test_mgus2_sex")

    tol = dict(rtol=1e-6, atol=1e-6)
    result_pcm = gw.grays_test(y, group=df["sex"], cause="pcm")
    assert_allclose_to_r(
        result_pcm.statistic, fixture["pcm"]["statistic"], what="gray pcm stat", **tol
    )
    assert_allclose_to_r(result_pcm.p_value, fixture["pcm"]["p_value"], what="gray pcm p", **tol)
    assert result_pcm.df == int(fixture["pcm"]["df"])

    result_death = gw.grays_test(y, group=df["sex"], cause="death")
    assert_allclose_to_r(
        result_death.statistic, fixture["death"]["statistic"], what="gray death stat", **tol
    )
    assert_allclose_to_r(
        result_death.p_value, fixture["death"]["p_value"], what="gray death p", **tol
    )
    assert result_death.df == int(fixture["death"]["df"])


def test_grays_test_three_groups_matches_r() -> None:
    df = gw.load_dataset("mgus2", backend="pandas")
    etime = np.where(df["pstat"] == 1, df["ptime"], df["futime"])
    cause = np.where(df["pstat"] == 1, 1, 2 * df["death"])
    y = Surv(
        time=etime,
        event=pd.Categorical.from_codes(cause, categories=["censor", "pcm", "death"]),
    )
    fixture = load_fixture("grays_test_mgus2_age")

    age_group = np.where(df["age"] < 60, "young", np.where(df["age"] < 70, "mid", "old"))
    tol = dict(rtol=1e-6, atol=1e-6)
    result = gw.grays_test(y, group=age_group, cause="pcm")
    assert_allclose_to_r(result.statistic, fixture["pcm"]["statistic"], what="gray age stat", **tol)
    assert_allclose_to_r(result.p_value, fixture["pcm"]["p_value"], what="gray age p", **tol)
    assert result.df == int(fixture["pcm"]["df"])


def test_multistate_occupancy_matches_r() -> None:
    t0, t1, frm, evt = _mgus2_illness_death()
    fixture = load_fixture("multistate_mgus2")
    ms = gw.MultiState().fit(t0, t1, frm, evt, states=("mgus", "pcm", "death"))
    table = ms.to_frame(format="pandas")
    assert_allclose_to_r(table["time"].to_numpy(), fixture["time"], what="ms time")
    for state in ("mgus", "pcm", "death"):
        assert_allclose_to_r(table[state].to_numpy(), fixture[state], what=f"occupancy {state}")


def test_brier_score_matches_r() -> None:
    df = gw.load_dataset("lung", backend="pandas")
    y = Surv(time=df["time"], event=df["status"] == 2)
    fixture = load_fixture("brier_lung")
    times = np.asarray(fixture["times"], dtype=float)
    cox = gw.CoxPH().fit(y, df[["age", "sex"]])
    pred = cox.predict(df[["age", "sex"]], type="survival", times=times)
    probs = pred[[f"subject_{i + 1}" for i in range(len(df))]].to_numpy().T
    assert_allclose_to_r(gw.brier_score(y, probs, times), fixture["brier"], what="brier")


def test_time_dependent_auc_matches_r() -> None:
    # Validates the Uno IPCW formula against an independent R implementation
    # (uno_td_auc() in scripts/regenerate_r_fixtures.R, no third-party packages).
    # The marker values from R's coxph are stored in the fixture so that the test
    # isolates the AUC formula from Cox LP estimation precision.
    df = gw.load_dataset("lung", backend="pandas")
    y = Surv(time=df["time"], event=df["status"] == 2)
    fixture = load_fixture("td_auc_lung")
    times = np.asarray(fixture["times"], dtype=float)
    marker = np.asarray(fixture["marker"], dtype=float)
    auc = gw.time_dependent_auc(y, marker, times=times)
    assert_allclose_to_r(auc, fixture["auc"], atol=1e-9, what="td_auc")


def test_concordance_index_matches_r() -> None:
    # concordance_index() of the Cox linear predictor equals the model's concordance.
    df = gw.load_dataset("lung", backend="pandas")
    y = Surv(time=df["time"], event=df["status"] == 2)
    fixture = load_fixture("cox_diag_efron")
    cox = gw.CoxPH(ties="efron").fit(y, df[["age", "sex"]])
    c = gw.concordance_index(y, cox.predict(type="lp"))
    assert_allclose_to_r(c, fixture["concordance"], what="concordance_index")


@pytest.mark.rparity
def test_concordance_index_ipcw_matches_r() -> None:
    df = gw.load_dataset("lung", backend="pandas")
    y = Surv(time=df["time"], event=df["status"] == 2)
    fixture = load_fixture("concordance_ipcw_lung")
    marker = np.asarray(load_fixture("td_auc_lung")["marker"], dtype=float)
    c_default = gw.concordance_index_ipcw(y, marker)
    assert_allclose_to_r(c_default, fixture["c_ipcw_default"], what="ipcw_c_default")
    c_tau365 = gw.concordance_index_ipcw(y, marker, tau=365.0)
    assert_allclose_to_r(c_tau365, fixture["c_ipcw_tau365"], what="ipcw_c_tau365")


@pytest.mark.rparity
def test_concordance_index_ci_se_matches_r_jackknife() -> None:
    fixture = load_fixture("concordance_ci_ij")
    # The R standard errors are infinitesimal-jackknife values from central differences, so they
    # agree to about 1e-9.
    df = gw.load_dataset("lung", backend="pandas")
    y = Surv(time=df["time"], event=df["status"] == 2)
    marker = np.asarray(load_fixture("td_auc_lung")["marker"], dtype=float)
    for key, tau in (("lung_default", None), ("lung_tau365", 365.0)):
        res = gw.concordance_index_ci(y, marker, tau=tau)
        assert_allclose_to_r(res.estimate, fixture[key]["estimate"], what=f"{key} C")
        assert_allclose_to_r(res.se, fixture[key]["se"], rtol=1e-7, what=f"{key} se")

    mg = gw.load_dataset("mgus2", backend="pandas")
    etime = np.where(mg["pstat"] == 1, mg["ptime"], mg["futime"])
    cause = np.where(mg["pstat"] == 1, 1, 2 * mg["death"])
    y_cr = Surv(
        time=etime,
        event=pd.Categorical.from_codes(cause, categories=["censor", "pcm", "death"]),
    )
    res = gw.concordance_index_ci(y_cr, mg["age"].to_numpy(dtype=float), cause=1)
    expected = fixture["mgus2_pcm_age"]
    assert_allclose_to_r(res.estimate, expected["estimate"], what="mgus2 pcm C")
    assert_allclose_to_r(res.se, expected["se"], rtol=1e-7, what="mgus2 pcm se")

    cmp = gw.concordance_index_compare(y, marker, df["age"].to_numpy(dtype=float))
    expected = fixture["compare_lung_lp_age"]
    assert_allclose_to_r(cmp.delta, expected["difference"], what="lung lp - age")
    assert_allclose_to_r(cmp.se, expected["se"], rtol=1e-7, what="lung lp - age se")
    male = (mg["sex"] == "M").to_numpy(dtype=float)
    cmp = gw.concordance_index_compare(y_cr, mg["age"].to_numpy(dtype=float), male, cause=1)
    expected = fixture["compare_mgus2_age_sex"]
    assert_allclose_to_r(cmp.delta, expected["difference"], what="mgus2 age - sex")
    assert_allclose_to_r(cmp.se, expected["se"], rtol=1e-7, what="mgus2 age - sex se")


def test_risk_table_numbers_match_r() -> None:
    # get_risk_table_frame() needs only numpy/pandas (no plotnine), so it runs here.
    fixture = load_fixture("risk_table_lung_sex")
    df = gw.load_dataset("lung", backend="pandas")
    y = Surv(time=df["time"], event=df["status"] == 2)
    km = gw.KaplanMeier().fit(y, by=df["sex"])
    rtd = gw.get_risk_table_frame(km, times=fixture["times"], format="pandas")
    for label, expected in fixture["n_risk"].items():
        sub = rtd[rtd["strata"] == f"sex={label}"].sort_values("time")
        assert_allclose_to_r(sub["n_risk"].to_numpy(), expected, what=f"n_risk sex={label}")


def test_cox_conditional_after_ci_matches_r() -> None:
    import pandas as pd

    df = gw.load_dataset("lung", backend="pandas")
    y = Surv(time=df["time"], event=df["status"] == 2)
    fixture = load_fixture("cox_conditional_ci")
    cox = gw.CoxPH(ties="breslow").fit(y, df[["age", "sex"]])
    newdata = pd.DataFrame({"age": fixture["newdata_age"], "sex": fixture["newdata_sex"]})
    pred = cox.predict(
        newdata,
        type="survival",
        times=fixture["times"],
        conditional_after=fixture["start_time"],
        ci=True,
        format="pandas",
    )
    for j, subj in enumerate(("subj1", "subj2"), start=1):
        assert_allclose_to_r(
            pred[f"subject_{j}"].to_numpy(),
            fixture["surv"][subj],
            what=f"conditional {subj} surv",
        )
        assert_allclose_to_r(
            pred[f"subject_{j}_lower"].to_numpy(),
            fixture["lower"][subj],
            what=f"conditional {subj} lower",
        )
        assert_allclose_to_r(
            pred[f"subject_{j}_upper"].to_numpy(),
            fixture["upper"][subj],
            what=f"conditional {subj} upper",
        )
