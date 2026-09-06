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
    df = km.to_frame(format="pandas")
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


def test_km_to_pandas_columns() -> None:
    km = KaplanMeier().fit(Surv.right([1, 2], [1, 1]))
    df = km.to_frame(format="pandas")
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


def test_km_to_pandas_grouped_has_strata() -> None:
    km = KaplanMeier().fit(Surv.right([1, 2, 1, 2], [1, 1, 1, 1]), by=["a", "a", "b", "b"])
    assert "strata" in km.to_frame(format="pandas").columns


def test_nelson_aalen_cumhaz() -> None:
    na = NelsonAalen().fit(Surv.right([1, 2, 3], [1, 1, 1]))
    np.testing.assert_allclose(na.cumhaz_, [1 / 3, 1 / 3 + 1 / 2, 1 / 3 + 1 / 2 + 1.0])
    np.testing.assert_allclose(na.std_error_**2, [1 / 9, 1 / 9 + 1 / 4, 1 / 9 + 1 / 4 + 1.0])


# ---------------------------------------------------------------------------
# Nelson-Aalen tidy / glance
# ---------------------------------------------------------------------------


class TestNelsonAalenTidy:
    def test_tidy_columns(self) -> None:
        lung = gw.load_dataset("lung", backend="pandas")
        y = Surv.right(lung["time"], event=(lung["status"] == 2))
        na = NelsonAalen().fit(y)
        t = gw.tidy(na, format="pandas")
        expected = ["time", "n_risk", "n_event", "estimate", "std_error", "conf_low", "conf_high"]
        assert list(t.columns) == expected

    def test_tidy_matches_to_frame(self) -> None:
        na = NelsonAalen().fit(Surv.right([1, 2, 3, 4], [1, 0, 1, 1]))
        t = gw.tidy(na, format="pandas")
        f = na.to_frame(format="pandas")
        assert t.equals(f)

    def test_tidy_stratified_has_strata(self) -> None:
        na = NelsonAalen().fit(Surv.right([1, 2, 1, 2], [1, 1, 1, 1]), by=["a", "a", "b", "b"])
        t = gw.tidy(na, format="pandas")
        assert "strata" in t.columns
        assert set(t["strata"]) == {"a", "b"}

    def test_tidy_format_polars(self) -> None:
        import polars as pl

        na = NelsonAalen().fit(Surv.right([1, 2, 3], [1, 1, 1]))
        t = gw.tidy(na, format="polars")
        assert isinstance(t, pl.DataFrame)

    def test_tidy_estimate_is_cumhaz(self) -> None:
        na = NelsonAalen().fit(Surv.right([1, 2, 3], [1, 1, 1]))
        t = gw.tidy(na, format="pandas")
        np.testing.assert_allclose(t["estimate"].values, na.cumhaz_)


class TestNelsonAalenGlance:
    def test_glance_columns(self) -> None:
        lung = gw.load_dataset("lung", backend="pandas")
        y = Surv.right(lung["time"], event=(lung["status"] == 2))
        na = NelsonAalen().fit(y)
        g = gw.glance(na, format="pandas")
        assert list(g.columns) == ["n_start", "events", "max_cumhaz"]
        assert g.shape[0] == 1

    def test_glance_values(self) -> None:
        na = NelsonAalen().fit(Surv.right([1, 2, 3], [1, 1, 1]))
        g = gw.glance(na, format="pandas")
        assert g["n_start"].iloc[0] == 3.0
        assert g["events"].iloc[0] == 3.0
        np.testing.assert_allclose(g["max_cumhaz"].iloc[0], na.cumhaz_[-1])

    def test_glance_stratified(self) -> None:
        na = NelsonAalen().fit(Surv.right([1, 2, 1, 2], [1, 1, 1, 1]), by=["a", "a", "b", "b"])
        g = gw.glance(na, format="pandas")
        assert "strata" in g.columns
        assert g.shape[0] == 2
        assert list(g["strata"]) == ["a", "b"]

    def test_glance_format_polars(self) -> None:
        import polars as pl

        na = NelsonAalen().fit(Surv.right([1, 2, 3], [1, 1, 1]))
        g = gw.glance(na, format="polars")
        assert isinstance(g, pl.DataFrame)


# ---------------------------------------------------------------------------
# KaplanMeier.quantile() tests
# ---------------------------------------------------------------------------


class TestKaplanMeierQuantile:
    def test_quantile_median_matches_median(self) -> None:
        km = KaplanMeier().fit(Surv.right([1, 2, 3, 4], [1, 1, 1, 1]))
        assert km.quantile(0.5) == km.median()

    def test_quantile_first_quartile(self) -> None:
        km = KaplanMeier().fit(Surv.right([1, 2, 3, 4], [1, 1, 1, 1]))
        # S = 0.75, 0.5, 0.25, 0. First time S <= 0.75 is t=1.
        assert km.quantile(0.25) == 1.0

    def test_quantile_third_quartile(self) -> None:
        km = KaplanMeier().fit(Surv.right([1, 2, 3, 4], [1, 1, 1, 1]))
        # First time S <= 0.25 is t=3.
        assert km.quantile(0.75) == 3.0

    def test_quantile_never_reached_returns_nan(self) -> None:
        km = KaplanMeier().fit(Surv.right([1, 2, 3], [1, 0, 0]))
        # S = 2/3, 2/3, 2/3. Never drops to 0.5, so median is nan.
        assert np.isnan(km.quantile(0.5))

    def test_quantile_with_ci(self) -> None:
        km = KaplanMeier().fit(Surv.right([1, 2, 3, 4], [1, 1, 1, 1]))
        result = km.quantile(0.5, ci=True)
        assert isinstance(result, tuple)
        assert len(result) == 3
        estimate, lower, upper = result
        assert estimate == 2.0
        assert lower <= estimate
        assert upper >= estimate or np.isnan(upper)

    def test_quantile_ci_matches_median_ci(self) -> None:
        km = KaplanMeier().fit(Surv.right([1, 2, 3, 4, 5, 6, 7, 8], [1, 1, 1, 1, 1, 0, 0, 0]))
        q_est, q_lo, q_hi = km.quantile(0.5, ci=True)
        m_est, m_lo, m_hi = km.median(ci=True)
        assert q_est == m_est
        assert q_lo == m_lo
        assert (q_hi == m_hi) or (np.isnan(q_hi) and np.isnan(m_hi))

    def test_quantile_grouped(self) -> None:
        km = KaplanMeier().fit(
            Surv.right([1, 2, 3, 4, 1, 2, 3, 4], [1, 1, 1, 1, 1, 1, 1, 1]),
            by=["a", "a", "a", "a", "b", "b", "b", "b"],
        )
        result = km.quantile(0.5)
        assert isinstance(result, dict)
        assert set(result) == {"a", "b"}
        assert result["a"] == 2.0
        assert result["b"] == 2.0

    def test_quantile_grouped_with_ci(self) -> None:
        km = KaplanMeier().fit(
            Surv.right([1, 2, 3, 4, 1, 2, 3, 4], [1, 1, 1, 1, 1, 1, 1, 1]),
            by=["a", "a", "a", "a", "b", "b", "b", "b"],
        )
        result = km.quantile(0.25, ci=True)
        assert isinstance(result, dict)
        for label in ("a", "b"):
            est, lower, upper = result[label]
            assert est == 1.0
            assert lower <= est
            assert upper >= est or np.isnan(upper)

    def test_quantile_ordering(self) -> None:
        km = KaplanMeier().fit(Surv.right([1, 2, 3, 4, 5], [1, 1, 1, 1, 1]))
        q25 = km.quantile(0.25)
        q50 = km.quantile(0.50)
        q75 = km.quantile(0.75)
        assert q25 <= q50 <= q75

    def test_quantile_real_data(self) -> None:
        lung = gw.load_dataset("lung", backend="pandas")
        y = Surv.right(lung["time"], event=(lung["status"] == 2))
        km = KaplanMeier().fit(y)
        q25 = km.quantile(0.25)
        q50 = km.quantile(0.50)
        q75 = km.quantile(0.75)
        assert q25 > 0
        assert q25 <= q50
        if not np.isnan(q75):
            assert q50 <= q75


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


def test_rmrl_at_zero_equals_rmst() -> None:
    # RMRL(0; tau) is exactly the restricted mean survival time (value and CI).
    km = KaplanMeier().fit(Surv.right([5, 6, 4, 9, 3, 7, 2, 8], [1, 0, 1, 0, 1, 1, 1, 0]))
    for tau in (4.0, 7.0, 9.0):
        assert km.rmrl(0.0, tau) == pytest.approx(km.rmst(tau))
        np.testing.assert_allclose(km.rmrl(0.0, tau, ci=True), km.rmst(tau, ci=True))


def test_rmrl_matches_conditional_area() -> None:
    # RMRL(s; tau) = integral_s^tau S(u) du / S(s). Check against a fine numerical integral.
    km = KaplanMeier().fit(Surv.right([5, 6, 4, 9, 3, 7, 2, 8, 10], [1, 0, 1, 0, 1, 1, 1, 0, 1]))
    s, tau = 3.0, 9.0
    grid = np.linspace(s, tau, 200001)
    expected = float(np.trapezoid(km.predict(grid), grid)) / float(km.predict([s])[0])
    assert km.rmrl(s, tau) == pytest.approx(expected, abs=1e-3)


def test_rmrl_grouped_and_ci() -> None:
    km = KaplanMeier().fit(
        Surv.right([2, 4, 6, 3, 5, 7], [1, 1, 1, 1, 1, 1]), by=["a", "a", "a", "b", "b", "b"]
    )
    out = km.rmrl(1.0, 6.0, ci=True)
    assert set(out) == {"a", "b"}
    value, lower, upper = out["a"]
    assert lower <= value <= upper
    assert value <= 6.0 - 1.0  # bounded by the window width


def test_rmrl_undefined_when_all_failed_before_s() -> None:
    km = KaplanMeier().fit(Surv.right([1, 2, 3], [1, 1, 1]))  # S drops to 0 at t=3
    assert np.isnan(km.rmrl(5.0, 10.0))


def test_rmrl_argument_validation() -> None:
    km = KaplanMeier().fit(Surv.right([1, 2, 3], [1, 1, 1]))
    with pytest.raises(ValueError, match="tau"):
        km.rmrl(5.0, 3.0)
    with pytest.raises(ValueError, match="non-negative"):
        km.rmrl(-1.0, 3.0)


def test_km_tidy_and_glance_via_registry() -> None:
    km = KaplanMeier().fit(Surv.right([1, 2, 3, 4], [1, 1, 1, 1]))
    tidy_df = gw.tidy(km)
    assert "estimate" in tidy_df.columns
    glance_df = gw.glance(km, format="pandas")
    assert float(glance_df["events"].iloc[0]) == 4.0
    assert float(glance_df["median"].iloc[0]) == 2.0


def test_km_weights_scale_risk_set() -> None:
    km = KaplanMeier().fit(Surv.right([1, 2], [1, 1], weights=[2.0, 2.0]))
    # Weighted n at t=1 is 4, one weighted event of 2 -> S = 1 - 2/4 = 0.5.
    np.testing.assert_allclose(km.survival_[0], 0.5)


def test_km_robust_se_differs_from_greenwood() -> None:
    km_green = KaplanMeier().fit(Surv.right([1, 2, 3, 4, 5], [1, 0, 1, 0, 1]))
    km_robust = KaplanMeier(robust=True).fit(Surv.right([1, 2, 3, 4, 5], [1, 0, 1, 0, 1]))
    # Survival estimates are identical regardless of variance method.
    np.testing.assert_allclose(km_robust.survival_, km_green.survival_)
    # Robust SE differs from Greenwood SE.
    assert not np.allclose(km_robust.std_error_, km_green.std_error_)


def test_km_robust_ci_brackets_estimate() -> None:
    km = KaplanMeier(robust=True).fit(
        Surv.right([1, 2, 3, 4, 5, 6, 7, 8], [1, 0, 1, 0, 1, 0, 1, 0])
    )
    valid = km.survival_ > 0
    assert np.all(km.conf_low_[valid] <= km.survival_[valid])
    assert np.all(km.survival_[valid] <= km.conf_high_[valid])


def test_km_robust_weighted_se_smaller_with_unit_weights() -> None:
    y = Surv.right([1, 2, 3, 4, 5], [1, 0, 1, 0, 1])
    km_unit = KaplanMeier(robust=True).fit(y)
    km_double = KaplanMeier(robust=True).fit(y, weights=[2.0, 2.0, 2.0, 2.0, 2.0])
    # Doubling all weights doesn't change survival (proportional scaling).
    np.testing.assert_allclose(km_double.survival_, km_unit.survival_)


def test_km_robust_grouped() -> None:
    y = Surv.right([1, 2, 3, 4, 5, 6], [1, 1, 1, 1, 1, 1])
    group = [0, 0, 0, 1, 1, 1]
    km = KaplanMeier(robust=True).fit(y, by=group)
    assert len(km._blocks) == 2
    for block in km._blocks:
        assert len(block.std_error) == len(block.surv)


def test_km_cluster_implies_robust() -> None:
    y = Surv.right([1, 2, 3, 4, 5, 6], [1, 0, 1, 0, 1, 0])
    cluster = ["A", "A", "B", "B", "C", "C"]
    km_cluster = KaplanMeier().fit(y, cluster=cluster)
    km_robust = KaplanMeier(robust=True).fit(y)
    # Survival estimates are identical; only SE differs due to clustering.
    np.testing.assert_allclose(km_cluster.survival_, km_robust.survival_)
    # Clustered SE differs from unclustered robust SE.
    assert not np.allclose(km_cluster.std_error_, km_robust.std_error_)


def test_km_cluster_se_larger_than_robust() -> None:
    y = Surv.right([1, 2, 3, 4, 5, 6, 7, 8], [1, 1, 1, 1, 1, 1, 1, 1])
    cluster = [0, 0, 1, 1, 2, 2, 3, 3]
    km_cluster = KaplanMeier().fit(y, cluster=cluster)
    km_robust = KaplanMeier(robust=True).fit(y)
    # With correlated subjects, clustered SE should generally differ from unclustered.
    assert km_cluster.std_error_.shape == km_robust.std_error_.shape


def test_km_cluster_singleton_clusters_equal_robust() -> None:
    y = Surv.right([1, 2, 3, 4, 5], [1, 0, 1, 0, 1])
    # Each subject is its own cluster — should equal unclustered robust.
    cluster = [0, 1, 2, 3, 4]
    km_cluster = KaplanMeier().fit(y, cluster=cluster)
    km_robust = KaplanMeier(robust=True).fit(y)
    np.testing.assert_allclose(km_cluster.std_error_, km_robust.std_error_)
