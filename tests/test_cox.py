"""Unit tests for the Cox proportional hazards model."""

from __future__ import annotations

import numpy as np
import pytest

import greenwood as gw
from greenwood import CoxPH, Surv


@pytest.fixture
def lung_surv():  # type: ignore[no-untyped-def]
    df = gw.data.load_dataset("lung")
    return df, Surv.right(df["time"], event=(df["status"] == 2))


def test_invalid_ties() -> None:
    with pytest.raises(ValueError, match="ties"):
        CoxPH(ties="exact")


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
    vt = gw.data.load_dataset("veteran")
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
