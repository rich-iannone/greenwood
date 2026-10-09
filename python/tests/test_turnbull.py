"""Unit tests for the Turnbull NPMLE (interval-censored data)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import greenwood as gw
from greenwood import KaplanMeier, Surv, Turnbull
from greenwood._turnbull import _build_alpha, _em_turnbull


def test_turnbull_reduces_to_km_right_censored() -> None:
    # No genuine interval ambiguity (exact events + right-censoring): must match KM exactly,
    # including a censored tail (KM leaves that mass unresolved rather than forcing S to 0).
    y = Surv(time=[1, 2, 3, 4, 5], event=[1, 1, 0, 1, 0])
    tb = Turnbull().fit(y)
    km = KaplanMeier().fit(y)
    np.testing.assert_allclose(tb.survival_, km.survival_, atol=1e-8)


def test_turnbull_reduces_to_km_all_exact() -> None:
    y = Surv(time=[1, 2, 3], event=[1, 1, 1])
    tb = Turnbull().fit(y)
    km = KaplanMeier().fit(y)
    np.testing.assert_allclose(tb.survival_, km.survival_, atol=1e-10)


def test_turnbull_mixed_censoring_sums_to_one() -> None:
    y = Surv(time=[0, 4, 7, 0, 3, 5], time2=[4, float("inf"), 7, 2.5, 6, 5], type="interval2")
    tb = Turnbull().fit(y)

    assert tb.prob_mass_.sum() == pytest.approx(1.0, abs=1e-6)

    # Survival is monotone non-increasing from 1 down to (near) 0.

    assert np.all(np.diff(tb.survival_) <= 1e-9)
    assert tb.survival_[-1] == pytest.approx(0.0, abs=1e-6)


def test_turnbull_exact_events_resolve_to_points() -> None:
    # Subjects with an exact event at t=5 and t=7 must show up as degenerate point atoms.
    y = Surv(time=[0, 4, 7, 0, 3, 5], time2=[4, float("inf"), 7, 2.5, 6, 5], type="interval2")
    tb = Turnbull().fit(y)
    resolved = tb.interval_low_ == tb.interval_high_

    assert 5.0 in tb.interval_high_[resolved]
    assert 7.0 in tb.interval_high_[resolved]


def test_turnbull_left_censored_mass_within_bound() -> None:
    # A left-censored subject (event before t=3) can only place mass in (0, 3].
    y = gw.as_surv(gw.event_time(time=[3, 10], status=np.where(np.asarray([1, 0]) == 1, "l", "r")))
    tb = Turnbull().fit(y)

    assert np.all(tb.interval_high_ <= 3.0 + 1e-9)


def test_turnbull_ambiguous_quantile_returns_bracket() -> None:
    # Two subjects with the same wide, fully overlapping window: nothing in the data can
    # resolve where within it the mass belongs, so the whole window is one ambiguous atom.
    y = Surv(time=[0, 0], time2=[10, 10], type="interval2")
    tb = Turnbull().fit(y)
    estimate, lower, upper = tb.quantile(0.5, format="polars").row(0)[2:]

    assert np.isnan(estimate)
    assert lower == 0.0
    assert upper == 10.0


def test_turnbull_unambiguous_quantile_returns_equal_bounds() -> None:
    y = Surv(time=[1, 2, 3, 4], event=[1, 1, 1, 1])
    tb = Turnbull().fit(y)
    med = tb.median(format="polars")

    assert med.columns == ["strata", "time", "time_low", "time_high"]
    assert med["strata"].to_list() == ["all"]

    estimate, lower, upper = med.row(0)[1:]

    assert estimate == lower == upper == 2.0


def test_turnbull_quantile_always_returns_same_columns() -> None:
    # Type-stable return: always the same frame columns, ambiguous or not.
    y = Surv(time=[0, 4, 7, 0, 3, 5], time2=[4, float("inf"), 7, 2.5, 6, 5], type="interval2")
    tb = Turnbull().fit(y)
    for p in (0.1, 0.25, 0.5, 0.75, 0.9):
        result = tb.quantile(p, format="polars")

        assert result.columns == ["strata", "prob", "time", "time_low", "time_high"]
        assert result.height == 1
        assert result["prob"][0] == p


def test_turnbull_quantile_never_reached_is_all_nan() -> None:
    y = Surv(time=[1, 2], event=[1, 0])  # survival never drops to 0
    tb = Turnbull().fit(y)
    estimate, lower, upper = tb.quantile(0.99, format="polars").row(0)[2:]

    assert np.isnan(estimate)
    assert np.isnan(lower)
    assert np.isnan(upper)


def test_turnbull_predict_nan_inside_ambiguous_interval() -> None:
    y = Surv(time=[0, 4, 7, 0, 3, 5], time2=[4, float("inf"), 7, 2.5, 6, 5], type="interval2")
    tb = Turnbull().fit(y)
    pred = tb.predict([1.0, 2.5, 5.0, 7.0], format="polars")["estimate"].to_numpy()

    assert np.isnan(pred[0])  # strictly inside the ambiguous (0, 2.5) region
    assert not np.isnan(pred[1])  # exactly at the boundary: defined
    assert not np.isnan(pred[2])
    assert not np.isnan(pred[3])


def test_turnbull_predict_before_first_atom_is_one() -> None:
    y = Surv(time=[2, 3], event=[1, 1])
    tb = Turnbull().fit(y)

    pred = tb.predict([0.0], format="polars")

    assert pred.columns == ["strata", "time", "estimate"]
    np.testing.assert_allclose(pred["estimate"].to_numpy(), [1.0])


def test_turnbull_rmst_reduces_to_km() -> None:
    y = Surv(time=[1, 2, 3, 4, 5], event=[1, 1, 0, 1, 0])
    tb = Turnbull().fit(y)
    km = KaplanMeier().fit(y)

    tb_rmst = tb.rmst(4, format="polars")["estimate"][0]

    assert tb_rmst == pytest.approx(km.rmst(4, format="polars")["estimate"][0], abs=1e-8)


def test_turnbull_rmrl_reduces_to_km() -> None:
    y = Surv(time=[1, 2, 3, 4, 5], event=[1, 1, 0, 1, 0])
    tb = Turnbull().fit(y)
    km = KaplanMeier().fit(y)

    tb_rmrl = tb.rmrl(2, 4, format="polars")["estimate"][0]

    assert tb_rmrl == pytest.approx(km.rmrl(2, 4, format="polars")["estimate"][0], abs=1e-8)


def test_turnbull_rmst_right_endpoint_convention() -> None:
    # A single wide ambiguous atom (0, 10]: the right-endpoint convention treats all of its
    # mass as resolving exactly at t=10, so S=1 throughout [0, 10) and RMST(tau) == tau for
    # any tau <= 10.
    y = Surv(time=[0, 0], time2=[10, 10], type="interval2")
    tb = Turnbull().fit(y)

    assert tb.rmst(10, format="polars")["estimate"][0] == pytest.approx(10.0)
    assert tb.rmst(5, format="polars")["estimate"][0] == pytest.approx(5.0)


def test_turnbull_rmrl_equals_rmst_at_zero() -> None:
    y = Surv(time=[0, 4, 7, 0, 3, 5], time2=[4, float("inf"), 7, 2.5, 6, 5], type="interval2")
    tb = Turnbull().fit(y)

    rmrl = tb.rmrl(0.0, 7.0, format="polars")["estimate"][0]

    assert rmrl == pytest.approx(tb.rmst(7.0, format="polars")["estimate"][0])


def test_turnbull_rmrl_nan_when_fully_resolved() -> None:
    y = Surv(time=[1, 2], event=[1, 1])
    tb = Turnbull().fit(y)

    assert np.isnan(tb.rmrl(2.0, 5.0, format="polars")["estimate"][0])


def test_turnbull_rmrl_invalid_tau() -> None:
    y = Surv(time=[1, 2], event=[1, 1])
    tb = Turnbull().fit(y)
    with pytest.raises(ValueError, match="tau"):
        tb.rmrl(5.0, 4.0)


def test_turnbull_rmrl_invalid_s() -> None:
    y = Surv(time=[1, 2], event=[1, 1])
    tb = Turnbull().fit(y)
    with pytest.raises(ValueError, match="non-negative"):
        tb.rmrl(-1.0, 4.0)


def test_turnbull_rmst_grouped_returns_one_row_per_curve() -> None:
    y = Surv(time=[1, 2, 1, 2], event=[1, 1, 1, 1])
    tb = Turnbull().fit(y, by=["a", "a", "b", "b"])
    result = tb.rmst(2, format="polars")

    assert result.columns == ["strata", "tau", "estimate"]
    assert result["strata"].to_list() == ["by=a", "by=b"]


def test_turnbull_grouped_returns_one_row_per_curve() -> None:
    y = Surv(time=[1, 2, 1, 2], event=[1, 1, 1, 1])
    tb = Turnbull().fit(y, by=["a", "a", "b", "b"])
    med = tb.median(format="polars")

    assert med["strata"].to_list() == ["by=a", "by=b"]
    assert set(tb.strata_) == {"by=a", "by=b"}


def test_turnbull_weights_equivalent_to_duplication() -> None:
    lower = [0.0, 4.0, 0.0, 3.0]
    upper = [4.0, float("inf"), 2.5, 6.0]
    y_dup = Surv(time=lower + [0.0], time2=upper + [2.5], type="interval2")  # duplicate row 3
    y_wt = Surv(time=lower, time2=upper, type="interval2")
    tb_dup = Turnbull().fit(y_dup)
    tb_wt = Turnbull().fit(y_wt, weights=[1.0, 1.0, 2.0, 1.0])

    np.testing.assert_allclose(tb_dup.survival_, tb_wt.survival_, atol=1e-6)


def test_turnbull_counting_process_not_supported() -> None:
    y = Surv(time=[0, 1], time2=[2, 3], event=[1, 0])
    with pytest.raises(NotImplementedError, match="truncation"):
        Turnbull().fit(y)


def test_turnbull_multistate_not_supported() -> None:
    y = Surv(time=[1, 2], event=pd.Categorical.from_codes([1, 2], categories=["censor", "a", "b"]))
    with pytest.raises(NotImplementedError, match="multi-state"):
        Turnbull().fit(y)


def test_turnbull_invalid_tol() -> None:
    with pytest.raises(ValueError, match="tol"):
        Turnbull(tol=0.0)


def test_turnbull_invalid_max_iter() -> None:
    with pytest.raises(ValueError, match="max_iter"):
        Turnbull(max_iter=0)


def test_turnbull_to_frame_columns() -> None:
    y = Surv(time=[1, 2], event=[1, 1])
    tb = Turnbull().fit(y)
    df = tb.to_frame(format="pandas")

    assert list(df.columns) == ["strata", "interval_low", "interval_high", "prob_mass", "estimate"]
    assert set(df["strata"]) == {"all"}


def test_turnbull_repr_unfit_and_fit() -> None:
    tb = Turnbull()

    assert "unfitted" in repr(tb)

    tb.fit(Surv(time=[1, 2], event=[1, 1]))

    assert "Turnbull" in repr(tb)


def test_turnbull_em_likelihood_nondecreasing() -> None:
    # The self-consistency EM must monotonically increase the observed-data log-likelihood.
    lower = np.array([0.0, 4.0, 7.0, 0.0, 3.0, 5.0])
    upper = np.array([4.0, np.inf, 7.0, 2.5, 6.0, 5.0])
    weight = np.ones(6)
    _, _, alpha = _build_alpha(lower, upper)

    def loglik(s: np.ndarray) -> float:
        return float(np.sum(weight * np.log(alpha.astype(float) @ s)))

    prev = -np.inf
    s = np.full(alpha.shape[1], 1.0 / alpha.shape[1])
    for max_iter in range(1, 30):
        s, _, _ = _em_turnbull(alpha, weight, tol=1e-14, max_iter=max_iter)
        current = loglik(s)

        assert current >= prev - 1e-10

        prev = current


def test_turnbull_em_converges() -> None:
    lower = np.array([0.0, 4.0, 7.0, 0.0, 3.0, 5.0])
    upper = np.array([4.0, np.inf, 7.0, 2.5, 6.0, 5.0])
    _, _, alpha = _build_alpha(lower, upper)
    s, n_iter, converged = _em_turnbull(alpha, np.ones(6), tol=1e-10, max_iter=10000)

    assert converged
    assert n_iter < 10000
    assert s.sum() == pytest.approx(1.0)


def test_turnbull_glance_and_tidy() -> None:
    import greenwood as gw

    y = Surv(time=[1, 2, 3], event=[1, 1, 1])
    tb = Turnbull().fit(y)
    g = gw.glance(tb, format="pandas")

    assert "n_iter" in g.columns
    assert "converged" in g.columns

    t = gw.tidy(tb, format="pandas")

    assert list(t.columns) == ["strata", "interval_low", "interval_high", "prob_mass", "estimate"]
    assert g["strata"].iloc[0] == "all"
