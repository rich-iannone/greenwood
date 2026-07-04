"""Unit tests for the Cox proportional hazards model."""

from __future__ import annotations

import numpy as np
import pytest

import greenwood as gw
from greenwood import CoxPH, Surv


@pytest.fixture
def lung_surv():  # type: ignore[no-untyped-def]
    df = gw.data.load_dataset("lung", backend="pandas")
    return df, Surv.right(df["time"], event=(df["status"] == 2))


def test_invalid_ties() -> None:
    with pytest.raises(ValueError, match="ties"):
        CoxPH(ties="exact")


def test_predict_conditional_after_identity(lung_surv) -> None:  # type: ignore[no-untyped-def]
    # S(t | T > c) * S(c) == S(t) for t >= c, and conditioning on c=0 is a no-op.
    df, y = lung_surv
    cox = CoxPH().fit(y, df[["age", "sex"]])
    nd = df[["age", "sex"]].iloc[:2]
    times = [200, 400, 600]
    c = 150.0
    s_t = cox.predict(nd, type="survival", times=times)
    s_c = cox.predict(nd, type="survival", times=[c])
    s_cond = cox.predict(nd, type="survival", times=times, conditional_after=c)
    for col in ("subject_1", "subject_2"):
        np.testing.assert_allclose(
            s_cond[col].to_numpy() * float(s_c[col].iloc[0]), s_t[col].to_numpy(), atol=1e-12
        )
    s0 = cox.predict(nd, type="survival", times=times, conditional_after=0.0)
    np.testing.assert_allclose(
        s0[["subject_1", "subject_2"]].to_numpy(), s_t[["subject_1", "subject_2"]].to_numpy()
    )


def test_predict_conditional_after_per_subject(lung_surv) -> None:  # type: ignore[no-untyped-def]
    df, y = lung_surv
    cox = CoxPH().fit(y, df[["age", "sex"]])
    nd = df[["age", "sex"]].iloc[:2]
    out = cox.predict(nd, type="survival", times=[300, 600], conditional_after=[100.0, 250.0])
    assert out.shape == (2, 3)
    assert ((out.iloc[:, 1:] >= 0) & (out.iloc[:, 1:] <= 1)).all().all()


def test_invalid_conf_level() -> None:
    with pytest.raises(ValueError, match="conf_level"):
        CoxPH(conf_level=0.0)


def test_length_mismatch(lung_surv) -> None:  # type: ignore[no-untyped-def]
    df, y = lung_surv
    with pytest.raises(ValueError, match="same number of rows"):
        CoxPH().fit(y, df[["age"]].iloc[:-1])


def test_higher_risk_score_direction() -> None:
    # A covariate perfectly ordered with earlier failure should get a positive coefficient.
    time = [1, 2, 3, 4, 5, 6]
    event = [1, 1, 1, 1, 1, 1]
    x = np.array([[6.0], [5.0], [4.0], [3.0], [2.0], [1.0]])  # larger x fails sooner
    cox = CoxPH().fit(Surv.right(time, event), x)
    assert cox.coef_[0] > 0


def test_array_covariates_default_names() -> None:
    x = np.random.default_rng(0).normal(size=(50, 2))
    time = np.arange(1, 51, dtype=float)
    event = np.ones(50)
    cox = CoxPH().fit(Surv.right(time, event), x)
    assert cox.term_names_ == ["x0", "x1"]


def test_categorical_covariate_dummy_encoding(lung_surv) -> None:  # type: ignore[no-untyped-def]
    df, _ = lung_surv
    vt = gw.data.load_dataset("veteran", backend="pandas")
    y = Surv.right(vt["time"], event=vt["status"])
    cox = CoxPH().fit(y, vt[["celltype"]])
    # celltype has 4 levels; drop-first leaves 3 dummy terms, all prefixed "celltype".
    assert len(cox.term_names_) == 3
    assert all(name.startswith("celltype") for name in cox.term_names_)


def test_tidy_exponentiate(lung_surv) -> None:  # type: ignore[no-untyped-def]
    df, y = lung_surv
    cox = CoxPH().fit(y, df[["age", "sex"]])
    tidy = gw.tidy.tidy(cox, exponentiate=True)
    # Exponentiated estimate equals the hazard ratio.
    np.testing.assert_allclose(tidy["estimate"].to_numpy(), cox.hazard_ratio_)


def test_glance_fields(lung_surv) -> None:  # type: ignore[no-untyped-def]
    df, y = lung_surv
    cox = CoxPH().fit(y, df[["age", "sex"]])
    glance = gw.tidy.glance(cox)
    row = glance.iloc[0]
    assert row["nevent"] == 165
    assert row["df"] == 2
    # AIC = -2 loglik + 2 p.
    assert row["aic"] == pytest.approx(-2 * cox.loglik_ + 2 * 2)


def test_to_dataframe_columns(lung_surv) -> None:  # type: ignore[no-untyped-def]
    df, y = lung_surv
    cox = CoxPH().fit(y, df[["age", "sex"]])
    assert list(cox.to_dataframe().columns) == [
        "term",
        "estimate",
        "std_error",
        "statistic",
        "p_value",
        "conf_low",
        "conf_high",
    ]


def test_baseline_hazard_is_monotone(lung_surv) -> None:  # type: ignore[no-untyped-def]
    df, y = lung_surv
    bh = CoxPH().fit(y, df[["age", "sex"]]).baseline_hazard()
    assert list(bh.columns) == ["time", "cumhaz", "survival"]
    assert np.all(np.diff(bh["cumhaz"]) >= 0)  # cumulative hazard is non-decreasing
    assert np.all(np.diff(bh["survival"]) <= 1e-12)  # survival is non-increasing


def test_predict_risk_is_exp_lp(lung_surv) -> None:  # type: ignore[no-untyped-def]
    df, y = lung_surv
    cox = CoxPH().fit(y, df[["age", "sex"]])
    np.testing.assert_allclose(cox.predict(type="risk"), np.exp(cox.predict(type="lp")))


def test_predict_unknown_type(lung_surv) -> None:  # type: ignore[no-untyped-def]
    df, y = lung_surv
    cox = CoxPH().fit(y, df[["age", "sex"]])
    with pytest.raises(ValueError, match="Unknown predict type"):
        cox.predict(type="hazard")


def test_predict_survival_in_unit_interval(lung_surv) -> None:  # type: ignore[no-untyped-def]
    df, y = lung_surv
    cox = CoxPH().fit(y, df[["age", "sex"]])
    surv = cox.predict(df[["age", "sex"]].head(3), type="survival", times=[100, 500])
    values = surv.drop(columns="time").to_numpy()
    assert np.all((values >= 0) & (values <= 1))


def test_residuals_unknown_type(lung_surv) -> None:  # type: ignore[no-untyped-def]
    df, y = lung_surv
    cox = CoxPH().fit(y, df[["age", "sex"]])
    with pytest.raises(ValueError, match="Unknown residual type"):
        cox.residuals("deviance")


def test_cox_zph_transform_validation(lung_surv) -> None:  # type: ignore[no-untyped-def]
    df, y = lung_surv
    cox = CoxPH().fit(y, df[["age", "sex"]])
    with pytest.raises(ValueError, match="transform"):
        cox.cox_zph(transform="km")


def test_cox_zph_result_to_dataframe(lung_surv) -> None:  # type: ignore[no-untyped-def]
    df, y = lung_surv
    z = CoxPH().fit(y, df[["age", "sex"]]).cox_zph()
    table = z.to_dataframe()
    assert list(table["term"]) == ["age", "sex", "GLOBAL"]
    assert "chisq" in table.columns


def test_concordance_in_unit_interval(lung_surv) -> None:  # type: ignore[no-untyped-def]
    df, y = lung_surv
    c = CoxPH().fit(y, df[["age", "sex"]]).concordance()
    assert 0.0 <= c <= 1.0


def test_stratified_has_robust_flag_false(lung_surv) -> None:  # type: ignore[no-untyped-def]
    df, y = lung_surv
    cox = CoxPH().fit(y, df[["age"]], strata=df["sex"])
    assert cox.robust is False
    # Stratified baseline hazard carries a strata column, one baseline per stratum.
    bh = cox.baseline_hazard()
    assert "strata" in bh.columns
    assert set(bh["strata"]) == {1, 2}


def test_robust_sets_flag_and_naive_available(lung_surv) -> None:  # type: ignore[no-untyped-def]
    df, y = lung_surv
    cox = CoxPH(ties="breslow").fit(y, df[["age", "sex"]], robust=True)
    assert cox.robust is True
    # Robust and naive standard errors are both available and generally differ.
    assert not np.allclose(cox.std_error_, cox.naive_std_error_)


def test_cluster_implies_robust(lung_surv) -> None:  # type: ignore[no-untyped-def]
    df, y = lung_surv
    cox = CoxPH(ties="breslow").fit(y, df[["age", "sex"]], cluster=df["inst"])
    assert cox.robust is True


def test_stratified_survival_prediction_not_supported(lung_surv) -> None:  # type: ignore[no-untyped-def]
    df, y = lung_surv
    cox = CoxPH().fit(y, df[["age"]], strata=df["sex"])
    with pytest.raises(NotImplementedError, match="stratified"):
        cox.predict(type="survival")


def test_strata_length_checked(lung_surv) -> None:  # type: ignore[no-untyped-def]
    df, y = lung_surv
    with pytest.raises(ValueError, match="strata"):
        CoxPH().fit(y, df[["age"]], strata=df["sex"].iloc[:-1])
