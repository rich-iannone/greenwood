"""Unit tests for the Turnbull NPMLE (interval-censored data)."""

from __future__ import annotations

import numpy as np
import pytest

from greenwood import KaplanMeier, Surv, Turnbull
from greenwood._turnbull import _build_alpha, _em_turnbull


def test_turnbull_reduces_to_km_right_censored() -> None:
    # No genuine interval ambiguity (exact events + right-censoring): must match KM exactly,
    # including a censored tail (KM leaves that mass unresolved rather than forcing S to 0).
    y = Surv.right([1, 2, 3, 4, 5], [1, 1, 0, 1, 0])
    tb = Turnbull().fit(y)
    km = KaplanMeier().fit(y)
    np.testing.assert_allclose(tb.survival_, km.survival_, atol=1e-8)


def test_turnbull_reduces_to_km_all_exact() -> None:
    y = Surv.right([1, 2, 3], [1, 1, 1])
    tb = Turnbull().fit(y)
    km = KaplanMeier().fit(y)
    np.testing.assert_allclose(tb.survival_, km.survival_, atol=1e-10)


def test_turnbull_mixed_censoring_sums_to_one() -> None:
    y = Surv.interval(lower=[0, 4, 7, 0, 3, 5], upper=[4, float("inf"), 7, 2.5, 6, 5])
    tb = Turnbull().fit(y)

    assert tb.prob_mass_.sum() == pytest.approx(1.0, abs=1e-6)

    # Survival is monotone non-increasing from 1 down to (near) 0.

    assert np.all(np.diff(tb.survival_) <= 1e-9)
    assert tb.survival_[-1] == pytest.approx(0.0, abs=1e-6)


def test_turnbull_exact_events_resolve_to_points() -> None:
    # Subjects with an exact event at t=5 and t=7 must show up as degenerate point atoms.
    y = Surv.interval(lower=[0, 4, 7, 0, 3, 5], upper=[4, float("inf"), 7, 2.5, 6, 5])
    tb = Turnbull().fit(y)
    resolved = tb.interval_low_ == tb.interval_high_

    assert 5.0 in tb.interval_high_[resolved]
    assert 7.0 in tb.interval_high_[resolved]


def test_turnbull_left_censored_mass_within_bound() -> None:
    # A left-censored subject (event before t=3) can only place mass in (0, 3].
    y = Surv.left([3, 10], [1, 0])
    tb = Turnbull().fit(y)

    assert np.all(tb.interval_high_ <= 3.0 + 1e-9)


def test_turnbull_ambiguous_quantile_returns_bracket() -> None:
    # Two subjects with the same wide, fully overlapping window: nothing in the data can
    # resolve where within it the mass belongs, so the whole window is one ambiguous atom.
    y = Surv.interval(lower=[0, 0], upper=[10, 10])
    tb = Turnbull().fit(y)
    estimate, lower, upper = tb.quantile(0.5)

    assert np.isnan(estimate)
    assert lower == 0.0
    assert upper == 10.0


def test_turnbull_unambiguous_quantile_returns_equal_triple() -> None:
    y = Surv.right([1, 2, 3, 4], [1, 1, 1, 1])
    tb = Turnbull().fit(y)
    estimate, lower, upper = tb.median()

    assert estimate == lower == upper == 2.0


def test_turnbull_quantile_always_returns_triple() -> None:
    # Type-stable return: always a 3-tuple, never a bare float (ambiguous or not).
    y = Surv.interval(lower=[0, 4, 7, 0, 3, 5], upper=[4, float("inf"), 7, 2.5, 6, 5])
    tb = Turnbull().fit(y)
    for p in (0.1, 0.25, 0.5, 0.75, 0.9):
        result = tb.quantile(p)

        assert isinstance(result, tuple)
        assert len(result) == 3


def test_turnbull_quantile_never_reached_is_nan_triple() -> None:
    y = Surv.right([1, 2], [1, 0])  # survival never drops to 0
    tb = Turnbull().fit(y)
    estimate, lower, upper = tb.quantile(0.99)

    assert np.isnan(estimate)
    assert np.isnan(lower)
    assert np.isnan(upper)


def test_turnbull_predict_nan_inside_ambiguous_interval() -> None:
    y = Surv.interval(lower=[0, 4, 7, 0, 3, 5], upper=[4, float("inf"), 7, 2.5, 6, 5])
    tb = Turnbull().fit(y)
    pred = tb.predict([1.0, 2.5, 5.0, 7.0])

    assert np.isnan(pred[0])  # strictly inside the ambiguous (0, 2.5) region
    assert not np.isnan(pred[1])  # exactly at the boundary: defined
    assert not np.isnan(pred[2])
    assert not np.isnan(pred[3])


def test_turnbull_predict_before_first_atom_is_one() -> None:
    y = Surv.right([2, 3], [1, 1])
    tb = Turnbull().fit(y)

    np.testing.assert_allclose(tb.predict([0.0]), [1.0])


def test_turnbull_grouped_returns_dict() -> None:
    y = Surv.right([1, 2, 1, 2], [1, 1, 1, 1])
    tb = Turnbull().fit(y, by=["a", "a", "b", "b"])
    med = tb.median()

    assert set(med) == {"a", "b"}
    assert tb.strata_ is not None


def test_turnbull_weights_equivalent_to_duplication() -> None:
    lower = [0.0, 4.0, 0.0, 3.0]
    upper = [4.0, float("inf"), 2.5, 6.0]
    y_dup = Surv.interval(lower=lower + [0.0], upper=upper + [2.5])  # duplicate row 3
    y_wt = Surv.interval(lower=lower, upper=upper, weights=[1.0, 1.0, 2.0, 1.0])
    tb_dup = Turnbull().fit(y_dup)
    tb_wt = Turnbull().fit(y_wt)

    np.testing.assert_allclose(tb_dup.survival_, tb_wt.survival_, atol=1e-6)


def test_turnbull_counting_process_not_supported() -> None:
    y = Surv.counting(start=[0, 1], stop=[2, 3], event=[1, 0])
    with pytest.raises(NotImplementedError, match="truncation"):
        Turnbull().fit(y)


def test_turnbull_multistate_not_supported() -> None:
    y = Surv.multistate(time=[1, 2], event=[1, 2], states=("a", "b"))
    with pytest.raises(NotImplementedError, match="multi-state"):
        Turnbull().fit(y)


def test_turnbull_invalid_tol() -> None:
    with pytest.raises(ValueError, match="tol"):
        Turnbull(tol=0.0)


def test_turnbull_invalid_max_iter() -> None:
    with pytest.raises(ValueError, match="max_iter"):
        Turnbull(max_iter=0)


def test_turnbull_to_frame_columns() -> None:
    y = Surv.right([1, 2], [1, 1])
    tb = Turnbull().fit(y)
    df = tb.to_frame(format="pandas")

    assert list(df.columns) == ["interval_low", "interval_high", "prob_mass", "estimate"]


def test_turnbull_repr_unfit_and_fit() -> None:
    tb = Turnbull()

    assert "unfitted" in repr(tb)

    tb.fit(Surv.right([1, 2], [1, 1]))

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

    y = Surv.right([1, 2, 3], [1, 1, 1])
    tb = Turnbull().fit(y)
    g = gw.glance(tb, format="pandas")

    assert "n_iter" in g.columns
    assert "converged" in g.columns

    t = gw.tidy(tb, format="pandas")

    assert list(t.columns) == ["interval_low", "interval_high", "prob_mass", "estimate"]
