"""Tests for time_dependent_auc and integrated_auc."""

from __future__ import annotations

import numpy as np
import pytest

import greenwood as gw
from greenwood import Surv


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def lung():
    return gw.load_dataset("lung", backend="pandas")


@pytest.fixture(scope="module")
def lung_surv(lung):
    return Surv.right(lung["time"], event=(lung["status"] == 2))


# ---------------------------------------------------------------------------
# Perfect and reversed discrimination
# ---------------------------------------------------------------------------


def test_auc_perfect_marker_no_censoring() -> None:
    # Higher marker fails sooner → perfect discrimination at any time > max event time.
    # With no censoring, AUC should equal 1.0 when marker perfectly orders event times.
    time = [1.0, 2.0, 3.0, 4.0, 5.0]
    event = [1, 1, 1, 1, 1]
    marker = [5.0, 4.0, 3.0, 2.0, 1.0]  # perfectly inversely correlated with time
    y = Surv.right(time, event)
    auc = gw.time_dependent_auc(y, marker, times=[3.0])
    # At t=3: cases are times 1,2,3 (markers 5,4,3); controls are times 4,5 (markers 2,1).
    # All cases have higher marker than all controls → AUC = 1.0.
    assert auc[0] == pytest.approx(1.0)


def test_auc_reversed_marker_no_censoring() -> None:
    # Marker increases with time → worst discrimination (marker inverted relative to risk).
    time = [1.0, 2.0, 3.0, 4.0, 5.0]
    event = [1, 1, 1, 1, 1]
    marker = [1.0, 2.0, 3.0, 4.0, 5.0]  # perfectly positively correlated with time
    y = Surv.right(time, event)
    auc = gw.time_dependent_auc(y, marker, times=[3.0])
    # All cases have lower marker than all controls → AUC = 0.0.
    assert auc[0] == pytest.approx(0.0)


def test_auc_equal_marker_gives_half() -> None:
    # Uniform marker → AUC = 0.5 at every time.
    time = [1.0, 2.0, 3.0, 4.0, 5.0]
    event = [1, 1, 1, 1, 1]
    marker = [0.5, 0.5, 0.5, 0.5, 0.5]
    y = Surv.right(time, event)
    auc = gw.time_dependent_auc(y, marker, times=[2.0, 4.0])
    np.testing.assert_allclose(auc, [0.5, 0.5])


# ---------------------------------------------------------------------------
# NaN edge cases
# ---------------------------------------------------------------------------


def test_auc_nan_when_no_cases() -> None:
    # t=0.5 is before all events → no cases → nan.
    time = [1.0, 2.0, 3.0]
    event = [1, 1, 1]
    y = Surv.right(time, event)
    auc = gw.time_dependent_auc(y, [1.0, 2.0, 3.0], times=[0.5])
    assert np.isnan(auc[0])


def test_auc_nan_when_no_controls() -> None:
    # t beyond all observation times → no controls → nan.
    time = [1.0, 2.0, 3.0]
    event = [1, 1, 1]
    y = Surv.right(time, event)
    auc = gw.time_dependent_auc(y, [3.0, 2.0, 1.0], times=[100.0])
    assert np.isnan(auc[0])


# ---------------------------------------------------------------------------
# In-range on real data
# ---------------------------------------------------------------------------


def test_auc_in_unit_range(lung, lung_surv) -> None:
    cox = gw.CoxPH().fit(lung_surv, lung[["age", "sex"]])
    lp = cox.predict(type="lp")
    auc = gw.time_dependent_auc(lung_surv, lp, times=[180.0, 365.0, 540.0])
    assert auc.shape == (3,)
    assert np.all((auc >= 0.0) & (auc <= 1.0))


def test_auc_better_than_random_on_real_data(lung, lung_surv) -> None:
    # A fitted Cox model should discriminate better than random at standard horizons.
    cox = gw.CoxPH().fit(lung_surv, lung[["age", "sex"]])
    lp = cox.predict(type="lp")
    auc = gw.time_dependent_auc(lung_surv, lp, times=[365.0])
    assert auc[0] > 0.5


def test_auc_multiple_times_shape(lung, lung_surv) -> None:
    cox = gw.CoxPH().fit(lung_surv, lung[["age", "sex"]])
    lp = cox.predict(type="lp")
    times = [100.0, 200.0, 300.0, 400.0, 500.0]
    auc = gw.time_dependent_auc(lung_surv, lp, times=times)
    assert auc.shape == (5,)


# ---------------------------------------------------------------------------
# IPCW weights matter when censoring is present
# ---------------------------------------------------------------------------


def test_auc_with_censoring_is_finite(lung, lung_surv) -> None:
    # lung has censored subjects; result should still be a valid probability.
    n_censored = int((~lung_surv.event.astype(bool)).sum())
    assert n_censored > 0, "Need censored subjects for this test"
    cox = gw.CoxPH().fit(lung_surv, lung[["age", "sex"]])
    lp = cox.predict(type="lp")
    auc = gw.time_dependent_auc(lung_surv, lp, times=[365.0])
    assert np.isfinite(auc[0])
    assert 0.0 <= auc[0] <= 1.0


# ---------------------------------------------------------------------------
# integrated_auc
# ---------------------------------------------------------------------------


def test_integrated_auc_in_range(lung, lung_surv) -> None:
    cox = gw.CoxPH().fit(lung_surv, lung[["age", "sex"]])
    lp = cox.predict(type="lp")
    iauc = gw.integrated_auc(lung_surv, lp, times=[180.0, 365.0, 540.0])
    assert 0.0 <= iauc <= 1.0


def test_integrated_auc_between_pointwise(lung, lung_surv) -> None:
    cox = gw.CoxPH().fit(lung_surv, lung[["age", "sex"]])
    lp = cox.predict(type="lp")
    times = [180.0, 365.0, 540.0]
    auc_pts = gw.time_dependent_auc(lung_surv, lp, times=times)
    iauc = gw.integrated_auc(lung_surv, lp, times=times)
    assert auc_pts.min() <= iauc <= auc_pts.max()


def test_integrated_auc_needs_two_times(lung_surv) -> None:
    with pytest.raises(ValueError, match="at least two times"):
        gw.integrated_auc(lung_surv, np.zeros(lung_surv.n), times=[365.0])


def test_integrated_auc_better_than_random(lung, lung_surv) -> None:
    cox = gw.CoxPH().fit(lung_surv, lung[["age", "sex"]])
    lp = cox.predict(type="lp")
    iauc = gw.integrated_auc(lung_surv, lp, times=[180.0, 365.0, 540.0])
    assert iauc > 0.5


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


def test_auc_length_mismatch(lung_surv) -> None:
    with pytest.raises(ValueError, match="same length"):
        gw.time_dependent_auc(lung_surv, np.zeros(5), times=[365.0])


def test_auc_accepts_polars_series() -> None:
    lung = gw.load_dataset("lung", backend="polars")
    y = Surv.right(lung["time"], event=(lung["status"] == 2))
    cox = gw.CoxPH().fit(y, lung[["age", "sex"]])
    lp = cox.predict(type="lp")
    # Should not raise; Polars series accepted via _to_1d_array.
    auc = gw.time_dependent_auc(y, lp, times=[365.0])
    assert auc.shape == (1,)


# ---------------------------------------------------------------------------
# Consistency: time_dependent_auc at a single very late time ≈ concordance
# ---------------------------------------------------------------------------


def test_integrated_auc_exceeds_random_for_strong_marker() -> None:
    # A perfectly ranked marker on uncensored data should yield integrated AUC close to 1.
    rng = np.random.default_rng(42)
    n = 200
    x = rng.normal(size=n)
    time = np.abs(rng.normal(loc=3.0 - 0.8 * x, size=n)) + 0.1
    event = np.ones(n, dtype=int)
    y = Surv.right(time, event)
    marker = x  # positively correlated with hazard (shorter time)

    times = np.linspace(float(np.percentile(time, 10)), float(np.percentile(time, 90)), 10)
    iauc = gw.integrated_auc(y, marker, times=times)
    # Strong marker should give clearly better-than-random AUC.
    assert iauc > 0.65
