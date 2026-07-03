"""Unit tests for the parametric AFT models."""

from __future__ import annotations

import numpy as np
import pytest

import greenwood as gw
from greenwood import AFT, Surv


@pytest.fixture
def lung_surv():  # type: ignore[no-untyped-def]
    df = gw.data.load_dataset("lung")
    return df, Surv.right(df["time"], event=(df["status"] == 2))


def test_invalid_dist() -> None:
    with pytest.raises(ValueError, match="dist"):
        AFT("gaussian")


def test_invalid_conf_level() -> None:
    with pytest.raises(ValueError, match="conf_level"):
        AFT(conf_level=1.0)


def test_intercept_is_added(lung_surv) -> None:  # type: ignore[no-untyped-def]
    df, y = lung_surv
    model = AFT("weibull").fit(y, df[["age", "sex"]])
    assert model.term_names_ == ["(Intercept)", "age", "sex"]


def test_exponential_scale_is_one(lung_surv) -> None:  # type: ignore[no-untyped-def]
    df, y = lung_surv
    model = AFT("exponential").fit(y, df[["age", "sex"]])
    assert model.scale_ == 1.0
    assert model.log_scale_se_ is None


def test_weibull_scale_estimated(lung_surv) -> None:  # type: ignore[no-untyped-def]
    df, y = lung_surv
    model = AFT("weibull").fit(y, df[["age", "sex"]])
    assert model.scale_ > 0.0
    assert model.log_scale_se_ is not None


def test_length_mismatch(lung_surv) -> None:  # type: ignore[no-untyped-def]
    df, y = lung_surv
    with pytest.raises(ValueError, match="same number of rows"):
        AFT().fit(y, df[["age"]].iloc[:-1])


def test_interval_censoring_not_supported() -> None:
    y = Surv.interval(lower=[1, 2, 3], upper=[2, 3, 4])
    with pytest.raises(NotImplementedError, match="right-censored"):
        AFT().fit(y, np.zeros((3, 1)))


def test_tidy_and_glance_via_registry(lung_surv) -> None:  # type: ignore[no-untyped-def]
    df, y = lung_surv
    model = AFT("weibull").fit(y, df[["age", "sex"]])
    tidy = gw.tidy.tidy(model)
    assert list(tidy["term"]) == ["(Intercept)", "age", "sex"]
    glance = gw.tidy.glance(model)
    row = glance.iloc[0]
    assert row["dist"] == "weibull"
    assert row["nevent"] == 165
    assert row["aic"] == pytest.approx(-2 * model.loglik_ + 2 * 4)  # 3 coef + log(scale)


def test_to_dataframe_columns(lung_surv) -> None:  # type: ignore[no-untyped-def]
    df, y = lung_surv
    model = AFT().fit(y, df[["age", "sex"]])
    assert list(model.to_dataframe().columns) == [
        "term",
        "estimate",
        "std_error",
        "statistic",
        "p_value",
        "conf_low",
        "conf_high",
    ]


def test_array_covariates(lung_surv) -> None:  # type: ignore[no-untyped-def]
    df, y = lung_surv
    x = df[["age", "sex"]].to_numpy(dtype=float)
    model = AFT("weibull").fit(y, x)
    assert model.term_names_ == ["(Intercept)", "x0", "x1"]
