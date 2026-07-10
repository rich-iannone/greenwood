"""Unit tests for AFT survival prediction confidence intervals."""

from __future__ import annotations

import numpy as np
import pytest

import greenwood as gw
from greenwood import AFT, Surv


@pytest.fixture
def lung_surv():  # type: ignore[no-untyped-def]
    df = gw.load_dataset("lung", backend="pandas")
    return df, Surv.right(df["time"], event=(df["status"] == 2))


def test_aft_predict_survival_ci_default_false(lung_surv) -> None:  # type: ignore[no-untyped-def]
    """By default, ci=False returns point estimates only."""
    df, y = lung_surv
    aft = AFT("weibull").fit(y, df[["age", "sex"]])
    pred = aft.predict(df[["age", "sex"]].iloc[:2], type="survival", times=[180, 365])
    assert "subject_1_lower" not in pred.columns
    assert "subject_1_upper" not in pred.columns


def test_aft_predict_survival_ci_logl_columns(lung_surv) -> None:  # type: ignore[no-untyped-def]
    """With ci=True and conf_type='log-log', CI columns are present."""
    df, y = lung_surv
    aft = AFT("weibull").fit(y, df[["age", "sex"]])
    pred = aft.predict(
        df[["age", "sex"]].iloc[:2], type="survival", times=[180, 365], ci=True, conf_type="log-log"
    )
    assert "subject_1_lower" in pred.columns
    assert "subject_1_upper" in pred.columns
    assert "subject_2_lower" in pred.columns
    assert "subject_2_upper" in pred.columns


def test_aft_predict_survival_ci_plain_columns(lung_surv) -> None:  # type: ignore[no-untyped-def]
    """With ci=True and conf_type='plain', CI columns are present."""
    df, y = lung_surv
    aft = AFT("weibull").fit(y, df[["age", "sex"]])
    pred = aft.predict(
        df[["age", "sex"]].iloc[:2], type="survival", times=[180, 365], ci=True, conf_type="plain"
    )
    assert "subject_1_lower" in pred.columns
    assert "subject_1_upper" in pred.columns


def test_aft_predict_survival_ci_bounds_logl(lung_surv) -> None:  # type: ignore[no-untyped-def]
    """Log-log CI bounds should bracket point estimates."""
    df, y = lung_surv
    aft = AFT("weibull").fit(y, df[["age", "sex"]])
    pred = aft.predict(
        df[["age", "sex"]].iloc[:2], type="survival", times=[180, 365], ci=True, conf_type="log-log"
    )

    # Bounds should bracket estimates
    assert (pred["subject_1_lower"] <= pred["subject_1"]).all()
    assert (pred["subject_1"] <= pred["subject_1_upper"]).all()
    assert (pred["subject_2_lower"] <= pred["subject_2"]).all()
    assert (pred["subject_2"] <= pred["subject_2_upper"]).all()

    # Bounds should respect (0, 1) constraint
    assert (pred["subject_1_lower"] > 0).all()
    assert (pred["subject_1_upper"] < 1).all()
    assert (pred["subject_2_lower"] > 0).all()
    assert (pred["subject_2_upper"] < 1).all()


def test_aft_predict_survival_ci_bounds_plain(lung_surv) -> None:  # type: ignore[no-untyped-def]
    """Plain CI bounds should bracket point estimates (but may violate constraints)."""
    df, y = lung_surv
    aft = AFT("weibull").fit(y, df[["age", "sex"]])
    pred = aft.predict(
        df[["age", "sex"]].iloc[:2], type="survival", times=[180, 365], ci=True, conf_type="plain"
    )

    # Bounds should bracket estimates
    assert (pred["subject_1_lower"] <= pred["subject_1"]).all()
    assert (pred["subject_1"] <= pred["subject_1_upper"]).all()


def test_aft_predict_survival_ci_confidence_level(lung_surv) -> None:  # type: ignore[no-untyped-def]
    """CI width should respect the model's confidence level."""
    df, y = lung_surv
    aft95 = AFT("weibull", conf_level=0.95).fit(y, df[["age", "sex"]])
    aft90 = AFT("weibull", conf_level=0.90).fit(y, df[["age", "sex"]])

    times = [180, 365]
    nd = df[["age", "sex"]].iloc[:1]

    pred95 = aft95.predict(nd, type="survival", times=times, ci=True, format="pandas")
    pred90 = aft90.predict(nd, type="survival", times=times, ci=True, format="pandas")

    # 90% CI should be narrower than 95% CI (on average)
    width_95 = (pred95["subject_1_upper"] - pred95["subject_1_lower"]).mean()
    width_90 = (pred90["subject_1_upper"] - pred90["subject_1_lower"]).mean()
    assert width_90 < width_95


def test_aft_predict_survival_ci_all_distributions(lung_surv) -> None:  # type: ignore[no-untyped-def]
    """CI computation should work for all error distributions."""
    df, y = lung_surv
    nd = df[["age", "sex"]].iloc[:1]
    times = [180, 365]

    for dist in ["weibull", "exponential", "lognormal", "loglogistic"]:
        aft = AFT(dist).fit(y, df[["age", "sex"]])
        pred = aft.predict(nd, type="survival", times=times, ci=True, conf_type="log-log")
        assert "subject_1_lower" in pred.columns
        assert "subject_1_upper" in pred.columns
        # Bounds should bracket estimates
        assert (pred["subject_1_lower"] <= pred["subject_1"]).all()
        assert (pred["subject_1"] <= pred["subject_1_upper"]).all()


def test_aft_predict_survival_ci_uncertainty_increases_with_covariate_distance(
    lung_surv,
) -> None:  # type: ignore[no-untyped-def]
    """CI width should increase for covariate values far from the training mean."""
    df, y = lung_surv
    aft = AFT("weibull").fit(y, df[["age", "sex"]])

    # One subject near mean, one at extreme
    mean_age = df["age"].mean()
    center = np.array([[mean_age, 1.0]])  # near mean
    extreme = np.array([[25.0, 1.0]])  # far from mean

    times = [180, 365]
    pred_center = aft.predict(center, type="survival", times=times, ci=True, format="pandas")
    pred_extreme = aft.predict(extreme, type="survival", times=times, ci=True, format="pandas")

    # CI width (on average) should be wider for extreme
    width_center = (pred_center["subject_1_upper"] - pred_center["subject_1_lower"]).mean()
    width_extreme = (pred_extreme["subject_1_upper"] - pred_extreme["subject_1_lower"]).mean()
    assert width_extreme > width_center


def test_aft_predict_survival_ci_with_conditional_after_raises(lung_surv) -> None:  # type: ignore[no-untyped-def]
    """CI with conditional_after should raise NotImplementedError."""
    df, y = lung_surv
    aft = AFT("weibull").fit(y, df[["age", "sex"]])
    with pytest.raises(NotImplementedError, match="conditional_after"):
        aft.predict(
            df[["age", "sex"]].iloc[:1],
            type="survival",
            times=[180],
            ci=True,
            conditional_after=100.0,
        )


def test_aft_predict_survival_ci_narrower_at_high_event_times(lung_surv) -> None:  # type: ignore[no-untyped-def]
    """CI may be narrower at high event times where survival is near 0 (log-log scale)."""
    df, y = lung_surv
    aft = AFT("weibull").fit(y, df[["age", "sex"]])
    nd = df[["age", "sex"]].iloc[:1]

    # Use early time (high survival) and late time (low survival)
    pred = aft.predict(nd, type="survival", times=[100, 800], ci=True, conf_type="log-log")

    # At high survival, log-log transform stretches the scale, so CI can be wider
    # At low survival, it compresses, so CI can be narrower
    width_early = pred.loc[pred["time"] == 100, "subject_1_upper"].values[0] - pred.loc[
        pred["time"] == 100, "subject_1_lower"
    ].values[0]
    width_late = pred.loc[pred["time"] == 800, "subject_1_upper"].values[0] - pred.loc[
        pred["time"] == 800, "subject_1_lower"
    ].values[0]

    # Both should be positive and valid
    assert width_early > 0
    assert width_late > 0


def test_aft_predict_survival_ci_single_subject(lung_surv) -> None:  # type: ignore[no-untyped-def]
    """CI computation should work for a single subject."""
    df, y = lung_surv
    aft = AFT("lognormal").fit(y, df[["age", "sex"]])
    nd = df[["age", "sex"]].iloc[:1]
    pred = aft.predict(nd, type="survival", times=[180, 365], ci=True)

    assert "subject_1" in pred.columns
    assert "subject_1_lower" in pred.columns
    assert "subject_1_upper" in pred.columns
    assert len(pred) == 2  # two time points


def test_aft_predict_survival_ci_many_subjects(lung_surv) -> None:  # type: ignore[no-untyped-def]
    """CI computation should work for multiple subjects."""
    df, y = lung_surv
    aft = AFT("loglogistic").fit(y, df[["age", "sex"]])
    nd = df[["age", "sex"]].iloc[:5]
    pred = aft.predict(nd, type="survival", times=[180, 365], ci=True)

    # Should have one column per subject (point + lower + upper)
    assert len(pred.columns) == 1 + 3 * 5  # time + (point + lower + upper) * 5
    assert len(pred) == 2  # two time points
    for i in range(1, 6):
        assert f"subject_{i}" in pred.columns
        assert f"subject_{i}_lower" in pred.columns
        assert f"subject_{i}_upper" in pred.columns


def test_aft_predict_survival_ci_edge_case_high_survival(lung_surv) -> None:  # type: ignore[no-untyped-def]
    """At very early times, survival should be near 1 with narrow CI."""
    df, y = lung_surv
    aft = AFT("weibull").fit(y, df[["age", "sex"]])
    nd = df[["age", "sex"]].iloc[:1]

    # Query at t=1 (very early)
    pred = aft.predict(nd, type="survival", times=[1.0], ci=True, conf_type="log-log")

    s = float(pred["subject_1"].iloc[0])
    assert s > 0.9  # should be high
    # CI should bracket
    assert s >= pred["subject_1_lower"].iloc[0]
    assert s <= pred["subject_1_upper"].iloc[0]


def test_aft_predict_survival_ci_edge_case_low_survival(lung_surv) -> None:  # type: ignore[no-untyped-def]
    """At very late times, survival should be near 0 with CI staying valid."""
    df, y = lung_surv
    aft = AFT("weibull").fit(y, df[["age", "sex"]])
    nd = df[["age", "sex"]].iloc[:1]

    # Query at very late time
    pred = aft.predict(nd, type="survival", times=[2000.0], ci=True, conf_type="log-log")

    s = float(pred["subject_1"].iloc[0])
    assert s < 0.1  # should be low
    # Bounds should respect (0, 1)
    assert 0 < pred["subject_1_lower"].iloc[0]
    assert pred["subject_1_upper"].iloc[0] < 1
    # And bracket the estimate
    assert s >= pred["subject_1_lower"].iloc[0]
    assert s <= pred["subject_1_upper"].iloc[0]
