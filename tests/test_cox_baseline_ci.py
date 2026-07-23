"""Unit tests for Cox baseline hazard confidence intervals."""

from __future__ import annotations

import pytest

import greenwood as gw
from greenwood import CoxPH, Surv


@pytest.fixture
def lung_surv():  # type: ignore[no-untyped-def]
    df = gw.load_dataset("lung", backend="pandas")
    return df, Surv.right(df["time"], event=(df["status"] == 2))


def test_baseline_hazard_ci_default_false(lung_surv) -> None:  # type: ignore[no-untyped-def]
    """By default, ci=False returns point estimates only."""
    df, y = lung_surv
    cox = CoxPH().fit(y, df[["age", "sex"]])
    bh = cox.baseline_hazard(format="pandas")
    expected_cols = {"time", "cumhaz", "survival"}
    assert set(bh.columns) == expected_cols


def test_baseline_hazard_ci_logl_columns(lung_surv) -> None:  # type: ignore[no-untyped-def]
    """With ci=True and conf_type='log-log', CI columns are present."""
    df, y = lung_surv
    cox = CoxPH().fit(y, df[["age", "sex"]])
    bh = cox.baseline_hazard(ci=True, conf_type="log-log", format="pandas")
    expected_cols = {
        "time",
        "cumhaz",
        "cumhaz_lower",
        "cumhaz_upper",
        "survival",
        "survival_lower",
        "survival_upper",
    }
    assert set(bh.columns) == expected_cols


def test_baseline_hazard_ci_plain_columns(lung_surv) -> None:  # type: ignore[no-untyped-def]
    """With ci=True and conf_type='plain', CI columns are present."""
    df, y = lung_surv
    cox = CoxPH().fit(y, df[["age", "sex"]])
    bh = cox.baseline_hazard(ci=True, conf_type="plain", format="pandas")
    expected_cols = {
        "time",
        "cumhaz",
        "cumhaz_lower",
        "cumhaz_upper",
        "survival",
        "survival_lower",
        "survival_upper",
    }
    assert set(bh.columns) == expected_cols


def test_baseline_hazard_ci_logl_bounds(lung_surv) -> None:  # type: ignore[no-untyped-def]
    """Log-log CI bounds should bracket point estimates."""
    df, y = lung_surv
    cox = CoxPH().fit(y, df[["age", "sex"]])
    bh = cox.baseline_hazard(ci=True, conf_type="log-log", format="pandas")

    # Cumulative hazard bounds
    assert (bh["cumhaz_lower"] <= bh["cumhaz"]).all()
    assert (bh["cumhaz"] <= bh["cumhaz_upper"]).all()

    # Survival bounds
    assert (bh["survival_lower"] <= bh["survival"]).all()
    assert (bh["survival"] <= bh["survival_upper"]).all()

    # Bounds should respect (0, 1) constraint for survival
    assert (bh["survival_lower"] > 0).all()
    assert (bh["survival_upper"] < 1).all()


def test_baseline_hazard_ci_plain_bounds(lung_surv) -> None:  # type: ignore[no-untyped-def]
    """Plain CI bounds should bracket point estimates (but may violate constraints)."""
    df, y = lung_surv
    cox = CoxPH().fit(y, df[["age", "sex"]])
    bh = cox.baseline_hazard(ci=True, conf_type="plain", format="pandas")

    # Cumulative hazard bounds
    assert (bh["cumhaz_lower"] <= bh["cumhaz"]).all()
    assert (bh["cumhaz"] <= bh["cumhaz_upper"]).all()

    # Survival bounds
    assert (bh["survival_lower"] <= bh["survival"]).all()
    assert (bh["survival"] <= bh["survival_upper"]).all()


def test_baseline_hazard_ci_confidence_level(lung_surv) -> None:  # type: ignore[no-untyped-def]
    """CI width should respect the model's confidence level."""
    df, y = lung_surv
    cox95 = CoxPH(conf_level=0.95).fit(y, df[["age", "sex"]])
    cox90 = CoxPH(conf_level=0.90).fit(y, df[["age", "sex"]])

    bh95 = cox95.baseline_hazard(ci=True, conf_type="log-log", format="pandas")
    bh90 = cox90.baseline_hazard(ci=True, conf_type="log-log", format="pandas")

    # 90% CI should be narrower than 95% CI (on average)
    width_95 = (bh95["cumhaz_upper"] - bh95["cumhaz_lower"]).mean()
    width_90 = (bh90["cumhaz_upper"] - bh90["cumhaz_lower"]).mean()
    assert width_90 < width_95


def test_baseline_hazard_ci_at_first_event(lung_surv) -> None:  # type: ignore[no-untyped-def]
    """At first event time, CI bounds should bracket the point estimate."""
    df, y = lung_surv
    cox = CoxPH().fit(y, df[["age", "sex"]])
    bh = cox.baseline_hazard(ci=True, conf_type="log-log", format="pandas")

    first_row = bh.iloc[0]
    assert first_row["cumhaz_lower"] >= 0
    assert first_row["cumhaz_lower"] <= first_row["cumhaz"]
    assert first_row["cumhaz_upper"] >= first_row["cumhaz"]


def test_baseline_hazard_ci_absolute_width_increases(lung_surv) -> None:  # type: ignore[no-untyped-def]
    """Absolute CI width should generally increase over time (cumulative uncertainty)."""
    df, y = lung_surv
    cox = CoxPH().fit(y, df[["age", "sex"]])
    bh = cox.baseline_hazard(ci=True, conf_type="log-log", format="pandas")

    abs_width = bh["cumhaz_upper"] - bh["cumhaz_lower"]
    n = len(bh)
    if n > 10:
        first_width = abs_width.iloc[:5].mean()
        last_width = abs_width.iloc[-5:].mean()
        assert last_width > first_width


def test_baseline_hazard_ci_stratified(lung_surv) -> None:  # type: ignore[no-untyped-def]
    """Stratified Cox model should have per-stratum CIs."""
    df, y = lung_surv
    cox = CoxPH().fit(y, df[["age"]], strata=df["sex"])
    bh = cox.baseline_hazard(ci=True, conf_type="log-log", format="pandas")

    # Should have 'strata' column
    assert "strata" in bh.columns
    assert set(bh["strata"].unique()) == {1.0, 2.0}

    # Each stratum should have its own CI
    for strata_val in bh["strata"].unique():
        subset = bh[bh["strata"] == strata_val]
        assert (subset["cumhaz_lower"] <= subset["cumhaz"]).all()
        assert (subset["cumhaz"] <= subset["cumhaz_upper"]).all()


def test_baseline_hazard_ci_monotonicity(lung_surv) -> None:  # type: ignore[no-untyped-def]
    """Cumulative hazard (point and bounds) should be non-decreasing."""
    df, y = lung_surv
    cox = CoxPH().fit(y, df[["age", "sex"]])
    bh = cox.baseline_hazard(ci=True, conf_type="log-log", format="pandas")

    # Point estimate should be non-decreasing
    assert (bh["cumhaz"].diff().dropna() >= 0).all()

    # Lower and upper bounds should also be non-decreasing
    assert (bh["cumhaz_lower"].diff().dropna() >= -1e-14).all()  # allow small numerical error
    assert (bh["cumhaz_upper"].diff().dropna() >= -1e-14).all()
