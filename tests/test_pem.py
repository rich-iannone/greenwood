"""Unit tests for the piecewise exponential model."""

from __future__ import annotations

import numpy as np
import pytest

import greenwood as gw
from greenwood import PiecewiseExponential, Surv


@pytest.fixture
def lung_data():  # type: ignore[no-untyped-def]
    df = gw.load_dataset("lung", backend="pandas")
    y = Surv.right(df["time"], event=(df["status"] == 2))
    return df, y


def test_manual_breaks_basic(lung_data) -> None:  # type: ignore[no-untyped-def]
    df, y = lung_data
    pem = PiecewiseExponential(breaks=[180, 365]).fit(y, df[["age", "sex"]])
    assert pem.n_ == 228
    assert pem.n_event_ == 165
    assert pem._n_intervals == 3
    assert len(pem.coef_) == 2
    assert len(pem.std_error_) == 2
    assert pem.aic_ == pytest.approx(-2.0 * pem.loglik_ + 2.0 * pem.df_)


def test_auto_breaks(lung_data) -> None:  # type: ignore[no-untyped-def]
    df, y = lung_data
    pem = PiecewiseExponential(knot_strategy="aic").fit(y, df[["age", "sex"]])
    assert pem._n_intervals >= 1
    assert len(pem.breaks_) == pem._n_intervals - 1


def test_bic_strategy(lung_data) -> None:  # type: ignore[no-untyped-def]
    df, y = lung_data
    pem = PiecewiseExponential(knot_strategy="bic").fit(y, df[["age", "sex"]])
    assert pem._n_intervals >= 1


def test_invalid_strategy() -> None:
    with pytest.raises(ValueError, match="knot_strategy"):
        PiecewiseExponential(knot_strategy="invalid")


def test_to_frame(lung_data) -> None:  # type: ignore[no-untyped-def]
    df, y = lung_data
    pem = PiecewiseExponential(breaks=[180, 365]).fit(y, df[["age", "sex"]])
    result = pem.to_frame(format="pandas")
    assert list(result.columns) == [
        "term",
        "estimate",
        "std_error",
        "statistic",
        "p_value",
        "conf_low",
        "conf_high",
    ]
    assert len(result) == 2
    assert list(result["term"]) == ["age", "sex"]


def test_baseline_hazard(lung_data) -> None:  # type: ignore[no-untyped-def]
    df, y = lung_data
    pem = PiecewiseExponential(breaks=[180, 365]).fit(y, df[["age", "sex"]])
    bh = pem.baseline_hazard(format="pandas")
    assert list(bh.columns) == ["start", "stop", "hazard", "log_hazard"]
    assert len(bh) == 3
    assert bh["hazard"].iloc[0] > 0
    np.testing.assert_allclose(bh["log_hazard"], np.log(bh["hazard"]))


def test_predict_survival(lung_data) -> None:  # type: ignore[no-untyped-def]
    df, y = lung_data
    pem = PiecewiseExponential(breaks=[180, 365]).fit(y, df[["age", "sex"]])
    times = [100, 200, 365, 500]
    pred = pem.predict(type="survival", times=times, format="pandas")
    assert pred.shape[0] == 4
    assert pred.shape[1] == pem.n_ + 1

    surv_vals = pred.iloc[:, 1:].values
    assert np.all(surv_vals >= 0)
    assert np.all(surv_vals <= 1)
    assert np.all(np.diff(surv_vals, axis=0) <= 0)


def test_predict_cumhaz(lung_data) -> None:  # type: ignore[no-untyped-def]
    df, y = lung_data
    pem = PiecewiseExponential(breaks=[180, 365]).fit(y, df[["age", "sex"]])
    pred = pem.predict(type="cumhaz", times=[100, 300], format="pandas")
    cumhaz_vals = pred.iloc[:, 1:].values
    assert np.all(cumhaz_vals >= 0)
    assert np.all(np.diff(cumhaz_vals, axis=0) >= 0)


def test_predict_lp_risk(lung_data) -> None:  # type: ignore[no-untyped-def]
    df, y = lung_data
    pem = PiecewiseExponential(breaks=[180, 365]).fit(y, df[["age", "sex"]])
    lp = pem.predict(type="lp")
    risk = pem.predict(type="risk")
    np.testing.assert_allclose(np.exp(lp), risk)


def test_predict_times_required(lung_data) -> None:  # type: ignore[no-untyped-def]
    df, y = lung_data
    pem = PiecewiseExponential(breaks=[180, 365]).fit(y, df[["age", "sex"]])
    with pytest.raises(ValueError, match="times="):
        pem.predict(type="survival")


def test_predict_newdata(lung_data) -> None:  # type: ignore[no-untyped-def]
    df, y = lung_data
    pem = PiecewiseExponential(breaks=[180, 365]).fit(y, df[["age", "sex"]])
    newdata = df[["age", "sex"]].iloc[:3]
    pred = pem.predict(newdata=newdata, type="survival", times=[100, 200], format="pandas")
    assert pred.shape == (2, 4)


def test_repr_unfitted() -> None:
    pem = PiecewiseExponential(breaks=[180, 365])
    r = repr(pem)
    assert "unfitted" in r


def test_repr_fitted(lung_data) -> None:  # type: ignore[no-untyped-def]
    df, y = lung_data
    pem = PiecewiseExponential(breaks=[180, 365]).fit(y, df[["age", "sex"]])
    r = repr(pem)
    assert "3 intervals" in r
    assert "age" in r
    assert "sex" in r
    assert "Log-likelihood" in r


def test_lr_statistic(lung_data) -> None:  # type: ignore[no-untyped-def]
    df, y = lung_data
    pem = PiecewiseExponential(breaks=[180, 365]).fit(y, df[["age", "sex"]])
    assert pem.lr_stat_ > 0
    assert pem.loglik_ > pem.loglik_null_


def test_no_covariates(lung_data) -> None:  # type: ignore[no-untyped-def]
    _, y = lung_data
    pem = PiecewiseExponential(breaks=[180]).fit(y, np.zeros((228, 0)))
    assert len(pem.coef_) == 0
    assert pem._n_intervals == 2


def test_single_break(lung_data) -> None:  # type: ignore[no-untyped-def]
    df, y = lung_data
    pem = PiecewiseExponential(breaks=[365]).fit(y, df[["age", "sex"]])
    assert pem._n_intervals == 2
    assert len(pem.breaks_) == 1


def test_tidy_glance(lung_data) -> None:  # type: ignore[no-untyped-def]
    df, y = lung_data
    pem = PiecewiseExponential(breaks=[180, 365]).fit(y, df[["age", "sex"]])
    t = gw.tidy(pem, format="pandas")
    assert len(t) == 2
    g = gw.glance(pem, format="pandas")
    assert "loglik" in g.columns
    assert g["n_intervals"].iloc[0] == 3


def test_coef_close_to_cox(lung_data) -> None:  # type: ignore[no-untyped-def]
    """PEM covariate effects should be close to Cox model estimates."""
    df, y = lung_data
    cox = gw.CoxPH().fit(y, df[["age", "sex"]])
    pem = PiecewiseExponential(breaks=[180, 365]).fit(y, df[["age", "sex"]])
    np.testing.assert_allclose(pem.coef_, cox.coef_, atol=0.05)


def test_survival_cumhaz_consistency(lung_data) -> None:  # type: ignore[no-untyped-def]
    """S(t) = exp(-H(t))."""
    df, y = lung_data
    pem = PiecewiseExponential(breaks=[180, 365]).fit(y, df[["age", "sex"]])
    times = [50, 100, 200, 365, 500]
    surv = pem.predict(type="survival", times=times, format="pandas")
    cumhaz = pem.predict(type="cumhaz", times=times, format="pandas")
    surv_vals = surv.iloc[:, 1:].values
    cumhaz_vals = cumhaz.iloc[:, 1:].values
    np.testing.assert_allclose(surv_vals, np.exp(-cumhaz_vals), rtol=1e-10)


def test_counting_process(lung_data) -> None:  # type: ignore[no-untyped-def]
    """PEM should accept counting-process Surv input."""
    df, _ = lung_data
    y_cp = Surv.counting(
        start=np.zeros(len(df)),
        stop=df["time"].values,
        event=(df["status"].values == 2),
    )
    pem = PiecewiseExponential(breaks=[180, 365]).fit(y_cp, df[["age", "sex"]])
    assert pem.n_ == 228
