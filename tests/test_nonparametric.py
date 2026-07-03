"""Unit tests for the Kaplan-Meier and Nelson-Aalen estimators."""

from __future__ import annotations

import numpy as np
import pytest

import greenwood as gw
from greenwood import KaplanMeier, NelsonAalen, Surv


def test_km_simple_survival() -> None:
    # Three ordered events, no censoring: S steps 2/3, 1/3, 0.
    km = KaplanMeier().fit(Surv.right([1, 2, 3], [1, 1, 1]))
    np.testing.assert_allclose(km.survival_, [2 / 3, 1 / 3, 0.0])
    np.testing.assert_array_equal(km.time_, [1, 2, 3])


def test_km_censoring_holds_survival_flat() -> None:
    # A censor at t=2 does not drop survival, but reduces the risk set afterward.
    km = KaplanMeier().fit(Surv.right([1, 2, 3], [1, 0, 1]))
    # events only at t=1 and t=3; at t=1 S=1-1/3=2/3, at t=3 n=1 so S=0.
    df = km.to_dataframe()
    assert list(df["n_event"]) == [1, 0, 1]
    np.testing.assert_allclose(km.survival_, [2 / 3, 2 / 3, 0.0])


def test_km_median() -> None:
    km = KaplanMeier().fit(Surv.right([1, 2, 3, 4], [1, 1, 1, 1]))
    # S = 0.75, 0.5, 0.25, 0; first time S <= 0.5 is t=2.
    assert km.median() == 2.0


def test_km_predict_step_function() -> None:
    km = KaplanMeier().fit(Surv.right([1, 2, 3], [1, 1, 1]))
    pred = km.predict([0.5, 1.0, 1.5, 2.0, 3.0, 5.0])
    np.testing.assert_allclose(pred, [1.0, 2 / 3, 2 / 3, 1 / 3, 0.0, 0.0])


def test_km_predict_cumhaz() -> None:
    km = KaplanMeier().fit(Surv.right([1, 2, 3], [1, 1, 1]))
    pred = km.predict([1.0, 2.0], what="cumhaz")
    np.testing.assert_allclose(pred, [1 / 3, 1 / 3 + 1 / 2])


def test_km_grouped_returns_dict() -> None:
    km = KaplanMeier().fit(Surv.right([1, 2, 1, 2], [1, 1, 1, 1]), by=["a", "a", "b", "b"])
    med = km.median()
    assert set(med) == {"a", "b"}
    assert km.strata_ is not None


def test_km_confidence_bracket_estimate() -> None:
    km = KaplanMeier(conf_type="log-log").fit(Surv.right([1, 2, 3, 4, 5], [1, 1, 1, 1, 0]))
    assert np.all(km.conf_low_ <= km.survival_ + 1e-12)
    assert np.all(km.survival_ <= km.conf_high_ + 1e-12)
    assert np.all((km.conf_low_ >= 0) & (km.conf_high_ <= 1))


def test_km_invalid_conf_type() -> None:
    with pytest.raises(ValueError, match="conf_type"):
        KaplanMeier(conf_type="bogus")


def test_km_invalid_conf_level() -> None:
    with pytest.raises(ValueError, match="conf_level"):
        KaplanMeier(conf_level=1.5)


def test_km_to_dataframe_columns() -> None:
    km = KaplanMeier().fit(Surv.right([1, 2], [1, 1]))
    df = km.to_dataframe()
    assert list(df.columns) == [
        "time",
        "n_risk",
        "n_event",
        "n_censor",
        "estimate",
        "std_error",
        "conf_low",
        "conf_high",
    ]


def test_km_to_dataframe_grouped_has_strata() -> None:
    km = KaplanMeier().fit(Surv.right([1, 2, 1, 2], [1, 1, 1, 1]), by=["a", "a", "b", "b"])
    assert "strata" in km.to_dataframe().columns


def test_nelson_aalen_cumhaz() -> None:
    na = NelsonAalen().fit(Surv.right([1, 2, 3], [1, 1, 1]))
    np.testing.assert_allclose(na.cumhaz_, [1 / 3, 1 / 3 + 1 / 2, 1 / 3 + 1 / 2 + 1.0])
    np.testing.assert_allclose(na.std_error_**2, [1 / 9, 1 / 9 + 1 / 4, 1 / 9 + 1 / 4 + 1.0])


def test_km_rmst_equals_area_under_curve() -> None:
    # All events at 1,2,3: S = 2/3, 1/3, 0. Area to tau=3 is
    # 1*(1-0) + (2/3)*(2-1) + (1/3)*(3-2) = 1 + 2/3 + 1/3 = 2.
    km = KaplanMeier().fit(Surv.right([1, 2, 3], [1, 1, 1]))
    assert km.rmst(3.0) == pytest.approx(2.0)


def test_km_rmst_truncates_at_tau() -> None:
    km = KaplanMeier().fit(Surv.right([1, 2, 3], [1, 1, 1]))
    # Up to tau=1.5: 1*(1) + (2/3)*(0.5) = 1.3333...
    assert km.rmst(1.5) == pytest.approx(1.0 + (2 / 3) * 0.5)


def test_km_rmst_grouped_and_ci() -> None:
    km = KaplanMeier().fit(Surv.right([1, 2, 1, 2], [1, 1, 1, 1]), by=["a", "a", "b", "b"])
    out = km.rmst(2.0, ci=True)
    assert set(out) == {"a", "b"}
    value, lower, upper = out["a"]
    assert lower <= value <= upper


def test_km_tidy_and_glance_via_registry() -> None:
    km = KaplanMeier().fit(Surv.right([1, 2, 3, 4], [1, 1, 1, 1]))
    tidy_df = gw.tidy.tidy(km)
    assert "estimate" in tidy_df.columns
    glance_df = gw.tidy.glance(km)
    assert float(glance_df["events"].iloc[0]) == 4.0
    assert float(glance_df["median"].iloc[0]) == 2.0


def test_km_weights_scale_risk_set() -> None:
    km = KaplanMeier().fit(Surv.right([1, 2], [1, 1], weights=[2.0, 2.0]))
    # Weighted n at t=1 is 4, one weighted event of 2 -> S = 1 - 2/4 = 0.5.
    np.testing.assert_allclose(km.survival_[0], 0.5)
