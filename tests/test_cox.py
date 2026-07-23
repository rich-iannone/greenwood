"""Unit tests for the Cox proportional hazards model."""

from __future__ import annotations

import contextlib

import numpy as np
import pytest

import greenwood as gw
from greenwood import CoxPH, Surv


@pytest.fixture
def lung_surv():  # type: ignore[no-untyped-def]
    df = gw.load_dataset("lung", backend="pandas")
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
    s_t = cox.predict(nd, type="survival", times=times, format="pandas")
    s_c = cox.predict(nd, type="survival", times=[c], format="pandas")
    s_cond = cox.predict(nd, type="survival", times=times, conditional_after=c, format="pandas")
    for col in ("subject_1", "subject_2"):
        np.testing.assert_allclose(
            s_cond[col].to_numpy() * float(s_c[col].iloc[0]), s_t[col].to_numpy(), atol=1e-12
        )
    s0 = cox.predict(nd, type="survival", times=times, conditional_after=0.0, format="pandas")
    np.testing.assert_allclose(
        s0[["subject_1", "subject_2"]].to_numpy(), s_t[["subject_1", "subject_2"]].to_numpy()
    )


def test_predict_conditional_after_per_subject(lung_surv) -> None:  # type: ignore[no-untyped-def]
    df, y = lung_surv
    cox = CoxPH().fit(y, df[["age", "sex"]])
    nd = df[["age", "sex"]].iloc[:2]
    out = cox.predict(
        nd,
        type="survival",
        times=[300, 600],
        conditional_after=[100.0, 250.0],
        format="pandas",
    )
    assert out.shape == (2, 3)
    assert ((out.iloc[:, 1:] >= 0) & (out.iloc[:, 1:] <= 1)).all().all()


def test_predict_survival_ci_columns_and_ordering(lung_surv) -> None:  # type: ignore[no-untyped-def]
    df, y = lung_surv
    cox = CoxPH().fit(y, df[["age", "sex"]])
    nd = df[["age", "sex"]].iloc[:2]
    pred = cox.predict(nd, type="survival", times=[180, 365], ci=True)
    assert list(pred.columns) == [
        "time",
        "subject_1",
        "subject_1_lower",
        "subject_1_upper",
        "subject_2",
        "subject_2_lower",
        "subject_2_upper",
    ]
    # The band brackets the point estimate at every time.
    for j in (1, 2):
        assert (pred[f"subject_{j}_lower"] <= pred[f"subject_{j}"]).all()
        assert (pred[f"subject_{j}"] <= pred[f"subject_{j}_upper"]).all()


def test_predict_ci_with_conditional_after_raises(lung_surv) -> None:  # type: ignore[no-untyped-def]
    df, y = lung_surv
    cox = CoxPH().fit(y, df[["age", "sex"]])
    with pytest.raises(NotImplementedError, match="conditional_after"):
        cox.predict(
            df[["age", "sex"]].iloc[:1],
            type="survival",
            times=[180],
            ci=True,
            conditional_after=50.0,
        )


def test_formula_matches_explicit(lung_surv) -> None:  # type: ignore[no-untyped-def]
    # A right-hand-side formula gives the same fit as passing the columns directly.
    df, y = lung_surv
    explicit = CoxPH().fit(y, df[["age", "sex"]])
    formula = CoxPH().fit(y, "age + sex", data=df)
    assert formula.term_names_ == ["age", "sex"]
    np.testing.assert_allclose(formula.coef_, explicit.coef_)
    np.testing.assert_allclose(formula.std_error_, explicit.std_error_)


def test_formula_interaction_and_categorical(lung_surv) -> None:  # type: ignore[no-untyped-def]
    df, y = lung_surv
    inter = CoxPH().fit(y, "age * sex", data=df)
    assert inter.term_names_ == ["age", "sex", "age:sex"]
    vet = gw.load_dataset("veteran", backend="pandas")
    yv = Surv.right(vet["time"], event=vet["status"])
    # Categorical: same model as explicit dummy coding, up to the reference level, so the
    # log-likelihood (reference-invariant) agrees exactly.
    by_formula = CoxPH(ties="breslow").fit(yv, "celltype", data=vet)
    explicit = CoxPH(ties="breslow").fit(yv, vet[["celltype"]])
    assert len(by_formula.term_names_) == 3
    np.testing.assert_allclose(by_formula.loglik_, explicit.loglik_)


def test_formula_complete_case_alignment(lung_surv) -> None:  # type: ignore[no-untyped-def]
    # ph.ecog has one missing value; the formula path drops that row like the response.
    df, y = lung_surv
    assert CoxPH().fit(y, "age + ph.ecog", data=df).n_ == 227


def test_formula_requires_data(lung_surv) -> None:  # type: ignore[no-untyped-def]
    df, y = lung_surv
    with pytest.raises(ValueError, match="data"):
        CoxPH().fit(y, "age + sex")


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
    vt = gw.load_dataset("veteran", backend="pandas")
    y = Surv.right(vt["time"], event=vt["status"])
    cox = CoxPH().fit(y, vt[["celltype"]])
    # celltype has 4 levels; drop-first leaves 3 dummy terms, all prefixed "celltype".
    assert len(cox.term_names_) == 3
    assert all(name.startswith("celltype") for name in cox.term_names_)


def test_tidy_exponentiate(lung_surv) -> None:  # type: ignore[no-untyped-def]
    df, y = lung_surv
    cox = CoxPH().fit(y, df[["age", "sex"]])
    tidy = gw.tidy(cox, exponentiate=True)
    # Exponentiated estimate equals the hazard ratio.
    np.testing.assert_allclose(tidy["estimate"].to_numpy(), cox.hazard_ratio_)


def test_glance_fields(lung_surv) -> None:  # type: ignore[no-untyped-def]
    df, y = lung_surv
    cox = CoxPH().fit(y, df[["age", "sex"]])
    glance = gw.glance(cox, format="pandas")
    row = glance.iloc[0]
    assert row["nevent"] == 165
    assert row["df"] == 2
    # AIC = -2 loglik + 2 p.
    assert row["aic"] == pytest.approx(-2 * cox.loglik_ + 2 * 2)


def test_glance_includes_frailty_lrt_fields(lung_surv) -> None:  # type: ignore[no-untyped-def]
    df, y = lung_surv
    cox = CoxPH(ties="breslow").fit(
        y,
        df[["age", "sex"]],
        frailty="gamma",
        frailty_cluster=df["inst"],
    )
    row = gw.glance(cox, format="pandas").iloc[0]
    assert row["frailty_theta"] > 0.0
    assert row["frailty_lrt_statistic"] >= 0.0
    assert 0.0 <= row["frailty_lrt_p_value"] <= 1.0


def test_to_dataframe_columns(lung_surv) -> None:  # type: ignore[no-untyped-def]
    df, y = lung_surv
    cox = CoxPH().fit(y, df[["age", "sex"]])
    expected_columns = [
        "term",
        "estimate",
        "std_error",
        "statistic",
        "p_value",
        "conf_low",
        "conf_high",
    ]
    # Test to_pandas
    assert list(cox.to_frame(format="pandas").columns) == expected_columns
    # Test to_polars
    pytest.importorskip("polars")
    assert list(cox.to_frame(format="polars").columns) == expected_columns
    # Test to_arrow
    pytest.importorskip("pyarrow")
    assert list(cox.to_frame(format="pyarrow").column_names) == expected_columns


def test_baseline_hazard_is_monotone(lung_surv) -> None:  # type: ignore[no-untyped-def]
    df, y = lung_surv
    bh = CoxPH().fit(y, df[["age", "sex"]]).baseline_hazard()
    assert list(bh.columns) == ["time", "cumhaz", "survival"]
    assert np.all(np.diff(bh["cumhaz"]) >= 0)  # cumulative hazard is non-decreasing
    assert np.all(np.diff(bh["survival"]) <= 1e-12)  # survival is non-increasing


def test_baseline_hazard_ci_without_ci_unchanged(lung_surv) -> None:  # type: ignore[no-untyped-def]
    df, y = lung_surv
    cox = CoxPH().fit(y, df[["age", "sex"]])
    bh_no_ci = cox.baseline_hazard()
    bh_ci = cox.baseline_hazard(ci=True)
    np.testing.assert_allclose(bh_ci["cumhaz"], bh_no_ci["cumhaz"])
    np.testing.assert_allclose(bh_ci["survival"], bh_no_ci["survival"])


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
    surv = cox.predict(
        df[["age", "sex"]].head(3), type="survival", times=[100, 500], format="pandas"
    )
    values = surv.drop(columns="time").to_numpy()
    assert np.all((values >= 0) & (values <= 1))


def test_residuals_unknown_type(lung_surv) -> None:  # type: ignore[no-untyped-def]
    df, y = lung_surv
    cox = CoxPH().fit(y, df[["age", "sex"]])
    with pytest.raises(ValueError, match="Unknown residual type"):
        cox.residuals("hazard")


def test_cox_zph_transform_validation(lung_surv) -> None:  # type: ignore[no-untyped-def]
    df, y = lung_surv
    cox = CoxPH().fit(y, df[["age", "sex"]])
    with pytest.raises(ValueError, match="transform"):
        cox.cox_zph(transform="km")


def test_cox_zph_result_to_dataframe(lung_surv) -> None:  # type: ignore[no-untyped-def]
    df, y = lung_surv
    z = CoxPH().fit(y, df[["age", "sex"]]).cox_zph()

    # Test to_pandas
    table = z.to_frame(format="pandas")
    assert list(table["term"]) == ["age", "sex", "GLOBAL"]
    assert "chisq" in table.columns

    # Test to_polars
    pytest.importorskip("polars")
    table_pl = z.to_frame(format="polars")
    assert list(table_pl["term"]) == ["age", "sex", "GLOBAL"]
    assert "chisq" in table_pl.columns

    # Test to_arrow
    pytest.importorskip("pyarrow")
    table_pa = z.to_frame(format="pyarrow")
    # PyArrow returns scalar objects, convert to Python objects
    terms = [str(t) for t in table_pa["term"].to_pylist()]
    assert terms == ["age", "sex", "GLOBAL"]
    assert "chisq" in table_pa.column_names


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


def test_gamma_frailty_requires_cluster(lung_surv) -> None:  # type: ignore[no-untyped-def]
    df, y = lung_surv
    with pytest.raises(ValueError, match="frailty_cluster"):
        CoxPH(ties="breslow").fit(y, df[["age", "sex"]], frailty="gamma")


def test_gamma_frailty_requires_breslow_ties(lung_surv) -> None:  # type: ignore[no-untyped-def]
    df, y = lung_surv
    with pytest.raises(NotImplementedError, match="ties='breslow'"):
        CoxPH(ties="efron").fit(y, df[["age", "sex"]], frailty="gamma", frailty_cluster=df["inst"])


def test_gamma_frailty_fit_sets_attributes(lung_surv) -> None:  # type: ignore[no-untyped-def]
    df, y = lung_surv
    cox = CoxPH(ties="breslow").fit(
        y,
        df[["age", "sex"]],
        frailty="gamma",
        frailty_cluster=df["inst"],
    )
    assert cox.frailty_ == "gamma"
    assert cox.frailty_theta_ is not None and cox.frailty_theta_ > 0
    assert cox.frailty_effect_ is not None
    assert cox.frailty_levels_ is not None
    assert len(cox.frailty_effect_) == len(cox.frailty_levels_)
    assert np.all(cox.frailty_effect_ > 0)
    assert cox.coef_.shape == (2,)
    assert cox.frailty_lrt_stat_ is not None and cox.frailty_lrt_stat_ >= 0.0
    assert cox.frailty_lrt_p_value_ is not None and 0.0 <= cox.frailty_lrt_p_value_ <= 1.0


def test_gamma_frailty_variance_test_api(lung_surv) -> None:  # type: ignore[no-untyped-def]
    df, y = lung_surv
    cox = CoxPH(ties="breslow").fit(
        y,
        df[["age", "sex"]],
        frailty="gamma",
        frailty_cluster=df["inst"],
    )
    out = cox.frailty_test()
    assert out["theta"] > 0.0
    assert out["lr_statistic"] >= 0.0
    assert out["df"] == 1.0
    assert 0.0 <= out["p_value"] <= 1.0


def test_frailty_test_requires_frailty_fit(lung_surv) -> None:  # type: ignore[no-untyped-def]
    df, y = lung_surv
    cox = CoxPH().fit(y, df[["age", "sex"]])
    with pytest.raises(ValueError, match="frailty"):
        cox.frailty_test()


def test_stratified_survival_prediction_not_supported(lung_surv) -> None:  # type: ignore[no-untyped-def]
    df, y = lung_surv
    cox = CoxPH().fit(y, df[["age"]], strata=df["sex"])
    with pytest.raises(NotImplementedError, match="stratified"):
        cox.predict(type="survival")


def test_strata_length_checked(lung_surv) -> None:  # type: ignore[no-untyped-def]
    df, y = lung_surv
    with pytest.raises(ValueError, match="strata"):
        CoxPH().fit(y, df[["age"]], strata=df["sex"].iloc[:-1])


def test_counting_process_proper_data_no_warning() -> None:
    """Counting-process data with subjects starting at 0 should not warn."""
    import warnings

    import pandas as pd

    df = pd.DataFrame(
        {
            "start": [0, 3, 0, 4, 0, 2],
            "stop": [3, 10, 4, 12, 2, 8],
            "event": [0, 1, 0, 1, 0, 1],
            "x": [1.0, 1.0, 2.0, 2.0, 3.0, 3.0],
        }
    )

    surv = Surv.counting(start=df["start"], stop=df["stop"], event=df["event"])

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        CoxPH().fit(surv, df[["x"]])
        our_warnings = [x for x in w if "start time" in str(x.message).lower()]
        assert len(our_warnings) == 0, "Should not warn for proper data starting at 0"


def test_counting_process_mixed_start_times_warns() -> None:
    """Counting-process data with mixed start times should warn."""
    import pandas as pd

    df = pd.DataFrame(
        {
            "start": [0, 5, 0, 4, 100, 105],  # Subject 3 starts at 100
            "stop": [5, 15, 4, 12, 105, 115],
            "event": [0, 1, 0, 1, 0, 1],
            "x": [1.0, 1.0, 2.0, 2.0, 3.0, 3.0],
        }
    )

    surv = Surv.counting(start=df["start"], stop=df["stop"], event=df["event"])

    with pytest.warns(UserWarning, match="start time.*calendar time"):
        CoxPH().fit(surv, df[["x"]])


def test_counting_process_large_gaps_warns() -> None:
    """Counting-process data with large gaps in start times should warn."""
    import pandas as pd

    df = pd.DataFrame(
        {
            "start": [0, 5, 200, 205],  # Large gap from 5 to 200
            "stop": [5, 15, 205, 215],
            "event": [0, 1, 0, 1],
            "x": [1.0, 1.0, 2.0, 2.0],
        }
    )

    surv = Surv.counting(start=df["start"], stop=df["stop"], event=df["event"])

    with (
        pytest.warns(UserWarning, match="start time.*calendar time"),
        contextlib.suppress(np.linalg.LinAlgError),
    ):
        # Large gaps may cause numerical issues; that's expected
        CoxPH().fit(surv, df[["x"]])


def test_counting_process_negative_start_warns() -> None:
    """Counting-process data with negative start times should warn."""
    import pandas as pd

    df = pd.DataFrame(
        {
            "start": [-5, 0, 0, 5],
            "stop": [0, 10, 5, 15],
            "event": [0, 1, 0, 1],
            "x": [1.0, 1.0, 2.0, 2.0],
        }
    )

    surv = Surv.counting(start=df["start"], stop=df["stop"], event=df["event"])

    with pytest.warns(UserWarning, match="negative.*start time"):
        CoxPH().fit(surv, df[["x"]])


def test_right_censored_data_no_warning() -> None:
    """Right-censored data should not trigger counting-process warnings."""
    import warnings

    import pandas as pd

    df = pd.DataFrame(
        {
            "time": [10, 20, 15, 25, 30],
            "event": [1, 1, 0, 1, 1],
            "x": [1.0, 2.0, 1.5, 2.5, 1.2],
        }
    )

    surv = Surv.right(df["time"], event=df["event"])

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        CoxPH().fit(surv, df[["x"]])
        our_warnings = [x for x in w if "start time" in str(x.message).lower()]
        assert len(our_warnings) == 0, "Right-censored data should not warn about start times"


def test_weight_invariance(lung_surv) -> None:  # type: ignore[no-untyped-def]
    """Coefficients should be invariant to uniform weight scaling."""
    df, y = lung_surv

    # Fit unweighted model
    cox_unweighted = CoxPH().fit(y, df[["age", "sex"]])
    coef_unweighted = cox_unweighted.coef_

    # Fit with uniform weights of 1 (should be identical to unweighted)
    y_weight1 = Surv.right(df["time"], event=(df["status"] == 2), weights=np.ones(len(df)))
    cox_weight1 = CoxPH().fit(y_weight1, df[["age", "sex"]])
    np.testing.assert_allclose(cox_weight1.coef_, coef_unweighted, rtol=1e-10)

    # Fit with uniform weights of 2 (should be identical to unweighted)
    y_weight2 = Surv.right(df["time"], event=(df["status"] == 2), weights=2.0 * np.ones(len(df)))
    cox_weight2 = CoxPH().fit(y_weight2, df[["age", "sex"]])
    np.testing.assert_allclose(cox_weight2.coef_, coef_unweighted, rtol=1e-10)

    # Fit with uniform weights of 5 (should be identical to unweighted)
    y_weight5 = Surv.right(df["time"], event=(df["status"] == 2), weights=5.0 * np.ones(len(df)))
    cox_weight5 = CoxPH().fit(y_weight5, df[["age", "sex"]])
    np.testing.assert_allclose(cox_weight5.coef_, coef_unweighted, rtol=1e-10)


def test_weight_invariance_efron_ties(lung_surv) -> None:  # type: ignore[no-untyped-def]
    """Test weight invariance with Efron tie-handling method."""
    df, y = lung_surv

    # Fit with Efron ties (non-default)
    cox_unweighted = CoxPH(ties="efron").fit(y, df[["age", "sex"]])
    coef_unweighted = cox_unweighted.coef_

    # Scale weights by different factors
    for scale in [1.0, 2.0, 0.5, 10.0]:
        y_weighted = Surv.right(
            df["time"], event=(df["status"] == 2), weights=scale * np.ones(len(df))
        )
        cox_weighted = CoxPH(ties="efron").fit(y_weighted, df[["age", "sex"]])
        np.testing.assert_allclose(cox_weighted.coef_, coef_unweighted, rtol=1e-10)


def test_weight_invariance_robust_variance(lung_surv) -> None:  # type: ignore[no-untyped-def]
    """Test weight invariance with robust/sandwich variance."""
    df, y = lung_surv

    # Fit with robust variance
    cox_unweighted = CoxPH(conf_level=0.95).fit(y, df[["age", "sex"]], robust=True)
    coef_unweighted = cox_unweighted.coef_
    var_unweighted = cox_unweighted.vcov_

    # Fit with scaled weights
    y_weighted = Surv.right(df["time"], event=(df["status"] == 2), weights=2.0 * np.ones(len(df)))
    cox_weighted = CoxPH(conf_level=0.95).fit(y_weighted, df[["age", "sex"]], robust=True)

    # Coefficients should match
    np.testing.assert_allclose(cox_weighted.coef_, coef_unweighted, rtol=1e-10)

    # Sandwich variance should also match (up to numerical precision)
    np.testing.assert_allclose(cox_weighted.vcov_, var_unweighted, rtol=1e-10)


def test_weight_invariance_stratified(lung_surv) -> None:  # type: ignore[no-untyped-def]
    """Test weight invariance with stratified Cox model."""
    df, y = lung_surv

    # Fit stratified model (by sex)
    cox_unweighted = CoxPH().fit(y, df[["age"]], strata=df["sex"])
    coef_unweighted = cox_unweighted.coef_

    # Fit with scaled weights
    y_weighted = Surv.right(df["time"], event=(df["status"] == 2), weights=3.0 * np.ones(len(df)))
    cox_weighted = CoxPH().fit(y_weighted, df[["age"]], strata=df["sex"])

    np.testing.assert_allclose(cox_weighted.coef_, coef_unweighted, rtol=1e-10)


def test_weight_invariance_predictions_and_residuals(lung_surv) -> None:  # type: ignore[no-untyped-def]
    """Test that predictions and residuals are invariant to weight scaling."""
    df, y = lung_surv
    test_data = df[["age", "sex"]].iloc[:5]

    # Fit unweighted and weighted models
    cox_unweighted = CoxPH().fit(y, df[["age", "sex"]])
    y_weighted = Surv.right(df["time"], event=(df["status"] == 2), weights=2.5 * np.ones(len(df)))
    cox_weighted = CoxPH().fit(y_weighted, df[["age", "sex"]])

    # Linear predictors should match
    pred_unweighted = cox_unweighted.predict(test_data, type="lp")
    pred_weighted = cox_weighted.predict(test_data, type="lp")
    np.testing.assert_allclose(pred_unweighted, pred_weighted, rtol=1e-10)

    # Risk scores should match
    risk_unweighted = cox_unweighted.predict(test_data, type="risk")
    risk_weighted = cox_weighted.predict(test_data, type="risk")
    np.testing.assert_allclose(risk_unweighted, risk_weighted, rtol=1e-10)

    # Survival predictions should match
    times = [100, 200, 365]
    surv_unweighted = cox_unweighted.predict(
        test_data, type="survival", times=times, format="pandas"
    )
    surv_weighted = cox_weighted.predict(test_data, type="survival", times=times, format="pandas")
    np.testing.assert_allclose(surv_unweighted.to_numpy(), surv_weighted.to_numpy(), rtol=1e-10)


def test_weight_invariance_concordance(lung_surv) -> None:  # type: ignore[no-untyped-def]
    """Test that concordance index is consistent with weight scaling."""
    df, y = lung_surv

    # Fit unweighted and weighted models
    cox_unweighted = CoxPH().fit(y, df[["age", "sex"]])
    y_weighted = Surv.right(df["time"], event=(df["status"] == 2), weights=2.0 * np.ones(len(df)))
    cox_weighted = CoxPH().fit(y_weighted, df[["age", "sex"]])

    # Concordance should match
    c_unweighted = cox_unweighted.concordance()
    c_weighted = cox_weighted.concordance()
    np.testing.assert_allclose(c_unweighted, c_weighted, rtol=1e-10)


def test_weight_invariance_extreme_scales(lung_surv) -> None:  # type: ignore[no-untyped-def]
    """Test weight invariance with very large and very small scale factors."""
    df, y = lung_surv

    # Fit unweighted
    cox_unweighted = CoxPH().fit(y, df[["age", "sex"]])
    coef_unweighted = cox_unweighted.coef_

    # Test with very small weights
    y_small = Surv.right(df["time"], event=(df["status"] == 2), weights=1e-6 * np.ones(len(df)))
    cox_small = CoxPH().fit(y_small, df[["age", "sex"]])
    np.testing.assert_allclose(cox_small.coef_, coef_unweighted, rtol=1e-9)

    # Test with very large weights
    y_large = Surv.right(df["time"], event=(df["status"] == 2), weights=1e6 * np.ones(len(df)))
    cox_large = CoxPH().fit(y_large, df[["age", "sex"]])
    np.testing.assert_allclose(cox_large.coef_, coef_unweighted, rtol=1e-9)


def test_weight_invariance_single_covariate(lung_surv) -> None:  # type: ignore[no-untyped-def]
    """Test weight invariance with a single covariate."""
    df, y = lung_surv

    # Fit with single covariate
    cox_unweighted = CoxPH().fit(y, df[["age"]])
    coef_unweighted = cox_unweighted.coef_

    # Test multiple weight scales
    for scale in [0.1, 1.0, 5.0, 100.0]:
        y_weighted = Surv.right(
            df["time"], event=(df["status"] == 2), weights=scale * np.ones(len(df))
        )
        cox_weighted = CoxPH().fit(y_weighted, df[["age"]])
        np.testing.assert_allclose(
            cox_weighted.coef_,
            coef_unweighted,
            rtol=1e-10,
            err_msg=f"Failed for weight scale {scale}",
        )


def test_weight_invariance_many_covariates(lung_surv) -> None:  # type: ignore[no-untyped-def]
    """Test weight invariance with many covariates."""
    df, y = lung_surv

    # Use all available numeric covariates
    covariates = ["age", "sex", "ph.ecog", "ph.karno", "wt.loss"]
    cox_unweighted = CoxPH().fit(y, df[covariates])
    coef_unweighted = cox_unweighted.coef_

    # Fit with scaled weights
    y_weighted = Surv.right(df["time"], event=(df["status"] == 2), weights=1.5 * np.ones(len(df)))
    cox_weighted = CoxPH().fit(y_weighted, df[covariates])

    np.testing.assert_allclose(cox_weighted.coef_, coef_unweighted, rtol=1e-10)


def test_poorly_scaled_covariates_warns() -> None:
    """CoxPH.fit() should warn when covariates have very different standard deviations."""

    import numpy as np

    lung = gw.load_dataset("lung", backend="pandas")
    y = Surv.right(lung["time"], event=(lung["status"] == 2))

    # 'age' has std ~10; add an independent covariate with std ~10,000 (income-like).
    # Using a random column ensures no collinearity with age.
    rng = np.random.default_rng(99)
    x = lung[["age"]].copy()
    x["income"] = rng.normal(50_000, 15_000, size=len(lung))  # std ~15,000

    with pytest.warns(UserWarning, match="very different scales"):
        CoxPH().fit(y, x)


def test_cox_frailty_repr(lung_surv) -> None:  # type: ignore[no-untyped-def]
    df, y = lung_surv
    cox = CoxPH(ties="breslow").fit(
        y, df[["age", "sex"]], frailty="gamma", frailty_cluster=df["inst"]
    )
    text = repr(cox)
    assert "Shared frailty" in text
    assert "theta" in text


def test_cox_frailty_invalid_type(lung_surv) -> None:  # type: ignore[no-untyped-def]
    df, y = lung_surv
    with pytest.raises(ValueError, match="frailty must be None or 'gamma'"):
        CoxPH().fit(y, df[["age", "sex"]], frailty="invalid")


def test_cox_frailty_cluster_without_frailty(lung_surv) -> None:  # type: ignore[no-untyped-def]
    df, y = lung_surv
    with pytest.raises(ValueError, match="frailty_cluster requires frailty='gamma'"):
        CoxPH().fit(y, df[["age", "sex"]], frailty_cluster=df["inst"])


def test_cox_frailty_theta_nonpositive(lung_surv) -> None:  # type: ignore[no-untyped-def]
    df, y = lung_surv
    with pytest.raises(ValueError, match="frailty_theta must be > 0"):
        CoxPH().fit(
            y,
            df[["age", "sex"]],
            frailty="gamma",
            frailty_cluster=df["inst"],
            frailty_theta=-1.0,
        )


def test_cox_frailty_max_iter_zero(lung_surv) -> None:  # type: ignore[no-untyped-def]
    df, y = lung_surv
    with pytest.raises(ValueError, match="frailty_max_iter must be >= 1"):
        CoxPH().fit(
            y,
            df[["age", "sex"]],
            frailty="gamma",
            frailty_cluster=df["inst"],
            frailty_max_iter=0,
        )


def test_cox_frailty_counting_process_not_supported(lung_surv) -> None:  # type: ignore[no-untyped-def]
    df, _ = lung_surv
    y_cp = Surv.counting(np.zeros(df.shape[0]), df["time"].values, (df["status"] == 2).values)
    with pytest.raises(NotImplementedError, match="right-censored"):
        CoxPH(ties="breslow").fit(
            y_cp, df[["age", "sex"]], frailty="gamma", frailty_cluster=df["inst"]
        )


def test_cox_frailty_with_strata_not_supported(lung_surv) -> None:  # type: ignore[no-untyped-def]
    df, y = lung_surv
    with pytest.raises(NotImplementedError, match="strata"):
        CoxPH(ties="breslow").fit(
            y,
            df[["age", "sex"]],
            frailty="gamma",
            frailty_cluster=df["inst"],
            strata=df["sex"],
        )


def test_cox_frailty_with_robust_not_supported(lung_surv) -> None:  # type: ignore[no-untyped-def]
    df, y = lung_surv
    with pytest.raises(NotImplementedError, match="robust"):
        CoxPH(ties="breslow").fit(
            y,
            df[["age", "sex"]],
            frailty="gamma",
            frailty_cluster=df["inst"],
            robust=True,
        )


def test_cox_frailty_test_unavailable_guard() -> None:
    cox = CoxPH()
    cox.frailty_ = "gamma"
    cox.frailty_theta_ = 0.5
    cox.frailty_lrt_stat_ = None
    cox.frailty_lrt_p_value_ = None
    with pytest.raises(ValueError, match="unavailable"):
        cox.frailty_test()


def test_residuals_deviance_shape_and_sign(lung_surv) -> None:  # type: ignore[no-untyped-def]
    df, y = lung_surv
    cox = CoxPH(ties="breslow").fit(y, df[["age", "sex"]])
    dev = cox.residuals("deviance")
    assert dev.shape == (cox.n_,)
    assert np.isfinite(dev).all()


def test_residuals_score_shape(lung_surv) -> None:  # type: ignore[no-untyped-def]
    df, y = lung_surv
    cox = CoxPH(ties="breslow").fit(y, df[["age", "sex"]])
    score = cox.residuals("score", format="pandas")
    assert score.shape == (cox.n_, 2)
    assert list(score.columns) == ["age", "sex"]


def test_residuals_dfbeta_shape(lung_surv) -> None:  # type: ignore[no-untyped-def]
    df, y = lung_surv
    cox = CoxPH(ties="breslow").fit(y, df[["age", "sex"]])
    dfb = cox.residuals("dfbeta", format="pandas")
    assert dfb.shape == (cox.n_, 2)
    assert list(dfb.columns) == ["age", "sex"]


def test_residuals_dfbetas_shape(lung_surv) -> None:  # type: ignore[no-untyped-def]
    df, y = lung_surv
    cox = CoxPH(ties="breslow").fit(y, df[["age", "sex"]])
    dfbs = cox.residuals("dfbetas", format="pandas")
    assert dfbs.shape == (cox.n_, 2)
    assert list(dfbs.columns) == ["age", "sex"]


def test_residuals_scaledsch_shape(lung_surv) -> None:  # type: ignore[no-untyped-def]
    df, y = lung_surv
    cox = CoxPH(ties="breslow").fit(y, df[["age", "sex"]])
    ssch = cox.residuals("scaledsch", format="pandas")
    assert ssch.shape[1] == 2
    assert list(ssch.columns) == ["age", "sex"]
    assert ssch.shape[0] == cox.n_event_


def test_residuals_dfbetas_is_standardized_dfbeta(lung_surv) -> None:  # type: ignore[no-untyped-def]
    df, y = lung_surv
    cox = CoxPH(ties="breslow").fit(y, df[["age", "sex"]])
    dfb = cox.residuals("dfbeta", format="pandas").to_numpy()
    dfbs = cox.residuals("dfbetas", format="pandas").to_numpy()
    np.testing.assert_allclose(dfbs, dfb / cox.naive_std_error_[None, :], atol=1e-14)


def test_well_scaled_covariates_no_warning() -> None:
    """CoxPH.fit() should not warn when covariates are on comparable scales."""
    import warnings

    lung = gw.load_dataset("lung", backend="pandas")
    y = Surv.right(lung["time"], event=(lung["status"] == 2))

    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        # age (std ~10) and sex (std ~0.5): ratio ~20, well below the 100 threshold
        CoxPH().fit(y, lung[["age", "sex"]])
