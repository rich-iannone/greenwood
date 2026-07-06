"""Unit tests for the Cox proportional hazards model."""

from __future__ import annotations

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


def test_predict_survival_ci_columns_and_ordering(lung_surv) -> None:  # type: ignore[no-untyped-def]
    df, y = lung_surv
    cox = CoxPH().fit(y, df[["age", "sex"]])
    nd = df[["age", "sex"]].iloc[:2]
    pred = cox.predict(nd, type="survival", times=[180, 365], ci=True)
    assert list(pred.columns) == [
        "time",
        "subject_1", "subject_1_lower", "subject_1_upper",
        "subject_2", "subject_2_lower", "subject_2_upper",
    ]
    # The band brackets the point estimate at every time.
    for j in (1, 2):
        assert (pred[f"subject_{j}_lower"] <= pred[f"subject_{j}"]).all()
        assert (pred[f"subject_{j}"] <= pred[f"subject_{j}_upper"]).all()


def test_predict_ci_with_conditional_after_raises(lung_surv) -> None:  # type: ignore[no-untyped-def]
    df, y = lung_surv
    cox = CoxPH().fit(y, df[["age", "sex"]])
    with pytest.raises(NotImplementedError, match="conditional_after"):
        cox.predict(df[["age", "sex"]].iloc[:1], type="survival", times=[180], ci=True,
                    conditional_after=50.0)


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
    glance = gw.glance(cox)
    row = glance.iloc[0]
    assert row["nevent"] == 165
    assert row["df"] == 2
    # AIC = -2 loglik + 2 p.
    assert row["aic"] == pytest.approx(-2 * cox.loglik_ + 2 * 2)


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
    assert list(cox.to_pandas().columns) == expected_columns
    # Test to_polars
    pytest.importorskip("polars")
    assert list(cox.to_polars().columns) == expected_columns
    # Test to_arrow
    pytest.importorskip("pyarrow")
    assert list(cox.to_arrow().column_names) == expected_columns


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
    
    # Test to_pandas
    table = z.to_pandas()
    assert list(table["term"]) == ["age", "sex", "GLOBAL"]
    assert "chisq" in table.columns
    
    # Test to_polars
    pytest.importorskip("polars")
    table_pl = z.to_polars()
    assert list(table_pl["term"]) == ["age", "sex", "GLOBAL"]
    assert "chisq" in table_pl.columns
    
    # Test to_arrow
    pytest.importorskip("pyarrow")
    table_pa = z.to_arrow()
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
    import pandas as pd
    import warnings
    
    df = pd.DataFrame({
        "start": [0, 3, 0, 4, 0, 2],
        "stop": [3, 10, 4, 12, 2, 8],
        "event": [0, 1, 0, 1, 0, 1],
        "x": [1.0, 1.0, 2.0, 2.0, 3.0, 3.0],
    })
    
    surv = Surv.counting(start=df["start"], stop=df["stop"], event=df["event"])
    
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        CoxPH().fit(surv, df[["x"]])
        our_warnings = [x for x in w if "start time" in str(x.message).lower()]
        assert len(our_warnings) == 0, "Should not warn for proper data starting at 0"


def test_counting_process_mixed_start_times_warns() -> None:
    """Counting-process data with mixed start times should warn."""
    import pandas as pd
    
    df = pd.DataFrame({
        "start": [0, 5, 0, 4, 100, 105],  # Subject 3 starts at 100
        "stop": [5, 15, 4, 12, 105, 115],
        "event": [0, 1, 0, 1, 0, 1],
        "x": [1.0, 1.0, 2.0, 2.0, 3.0, 3.0],
    })
    
    surv = Surv.counting(start=df["start"], stop=df["stop"], event=df["event"])
    
    with pytest.warns(UserWarning, match="start time.*calendar time"):
        CoxPH().fit(surv, df[["x"]])


def test_counting_process_large_gaps_warns() -> None:
    """Counting-process data with large gaps in start times should warn."""
    import pandas as pd
    
    df = pd.DataFrame({
        "start": [0, 5, 200, 205],  # Large gap from 5 to 200
        "stop": [5, 15, 205, 215],
        "event": [0, 1, 0, 1],
        "x": [1.0, 1.0, 2.0, 2.0],
    })
    
    surv = Surv.counting(start=df["start"], stop=df["stop"], event=df["event"])
    
    with pytest.warns(UserWarning, match="start time.*calendar time"):
        try:
            CoxPH().fit(surv, df[["x"]])
        except np.linalg.LinAlgError:
            # Large gaps may cause numerical issues; that's expected
            pass


def test_counting_process_negative_start_warns() -> None:
    """Counting-process data with negative start times should warn."""
    import pandas as pd
    
    df = pd.DataFrame({
        "start": [-5, 0, 0, 5],
        "stop": [0, 10, 5, 15],
        "event": [0, 1, 0, 1],
        "x": [1.0, 1.0, 2.0, 2.0],
    })
    
    surv = Surv.counting(start=df["start"], stop=df["stop"], event=df["event"])
    
    with pytest.warns(UserWarning, match="negative.*start time"):
        CoxPH().fit(surv, df[["x"]])


def test_right_censored_data_no_warning() -> None:
    """Right-censored data should not trigger counting-process warnings."""
    import pandas as pd
    import warnings
    
    df = pd.DataFrame({
        "time": [10, 20, 15, 25, 30],
        "event": [1, 1, 0, 1, 1],
        "x": [1.0, 2.0, 1.5, 2.5, 1.2],
    })
    
    surv = Surv.right(df["time"], event=df["event"])
    
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        CoxPH().fit(surv, df[["x"]])
        our_warnings = [x for x in w if "start time" in str(x.message).lower()]
        assert len(our_warnings) == 0, "Right-censored data should not warn about start times"
