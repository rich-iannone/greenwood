"""Edge cases for the array helpers and for missing rows at fit time."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import polars as pl
import pytest

import greenwood as gw
from greenwood import Surv
from greenwood._ingest import to_1d_array


def test_to_1d_array_none_and_shape() -> None:
    with pytest.raises(ValueError, match="Expected an array-like"):
        to_1d_array(None)
    with pytest.raises(ValueError, match="1-D"):
        to_1d_array(np.ones((2, 2)))


def test_to_1d_array_non_series_fallback() -> None:
    np.testing.assert_array_equal(to_1d_array(range(3)), [0.0, 1.0, 2.0])


def test_zero_length_response() -> None:
    y = Surv(time=[], event=[])
    assert (y.n, y.n_events, y.type) == (0, 0, "right")


def test_fit_drops_rows_with_a_missing_response() -> None:
    lung = gw.load_dataset("lung", backend="pandas")
    time = lung["time"].astype(float).copy()
    time.iloc[:3] = np.nan
    model = gw.CoxPH().fit(Surv(time=time, event=lung["status"]), "age + sex", data=lung)
    assert model.n_dropped_ == 3

    complete = lung.iloc[3:]
    reference = gw.CoxPH().fit(
        Surv(time=complete["time"], event=complete["status"]), "age + sex", data=complete
    )
    np.testing.assert_allclose(model.coef_, reference.coef_)


def test_fit_drops_rows_with_an_invalid_status() -> None:
    times = pl.Series([5.0, 6.0, 4.0, 9.0, 3.0])
    with pytest.warns(UserWarning, match="Invalid status value"):
        y = Surv(time=times, event=[1, 0, 1, 0, 5])
    km = gw.KaplanMeier().fit(y)
    assert km.n_dropped_ == 1


def test_fit_drops_rows_from_array_covariates_too() -> None:
    rng = np.random.default_rng(1)
    n = 60
    x = rng.normal(size=(n, 1))
    time = rng.exponential(size=n)
    time[0] = np.nan
    event = rng.integers(0, 2, size=n)
    model = gw.CoxPH().fit(Surv(time=time, event=event), x)
    reference = gw.CoxPH().fit(Surv(time=time[1:], event=event[1:]), x[1:])
    assert model.n_dropped_ == 1
    np.testing.assert_allclose(model.coef_, reference.coef_)


def test_weights_must_be_positive_and_aligned() -> None:
    y = Surv(time=[5.0, 6.0, 4.0, 9.0], event=[1, 0, 1, 0])
    with pytest.raises(ValueError, match="strictly positive"):
        gw.KaplanMeier().fit(y, weights=[1.0, 0.0, 1.0, 1.0])
    with pytest.raises(ValueError, match="has 2 values"):
        gw.KaplanMeier().fit(y, weights=[1.0, 1.0])


def test_counting_start_after_stop_is_dropped_at_fit() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        y = Surv(time=[0.0, 7.0, 1.0], time2=[5.0, 6.0, 4.0], event=[1, 0, 1])
    assert y.start is not None and int(np.isnan(y.start).sum()) == 1
    assert gw.KaplanMeier().fit(y).n_dropped_ == 1


def test_pandas_categorical_with_missing_value() -> None:
    event = pd.Categorical(["a", None, "b"], categories=["censor", "a", "b"])
    y = Surv(time=[5, 6, 7], event=event)
    assert np.isnan(y.status[1])
    assert y.states == ("a", "b")
