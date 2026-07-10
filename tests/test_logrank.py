"""Unit tests for the log-rank / G-rho group comparison test."""

from __future__ import annotations

import numpy as np
import pytest

from greenwood import Surv, logrank_test, pairwise_logrank_test, trend_test


def test_identical_groups_give_near_zero_statistic() -> None:
    # Two groups with identical event patterns: no evidence of a difference.
    time = [1, 2, 3, 4, 1, 2, 3, 4]
    event = [1, 1, 1, 1, 1, 1, 1, 1]
    group = ["a", "a", "a", "a", "b", "b", "b", "b"]
    result = logrank_test(Surv.right(time, event), group)
    assert result.statistic == pytest.approx(0.0, abs=1e-12)
    assert result.p_value == pytest.approx(1.0)
    assert result.df == 1


def test_separated_groups_give_large_statistic() -> None:
    # Group b fails much later than group a.
    time = [1, 2, 3, 10, 11, 12]
    event = [1, 1, 1, 1, 1, 1]
    group = ["a", "a", "a", "b", "b", "b"]
    result = logrank_test(Surv.right(time, event), group)
    assert result.statistic > 3.0


def test_method_string_reflects_weights() -> None:
    y = Surv.right([1, 2, 3, 4], [1, 1, 1, 1])
    g = ["a", "a", "b", "b"]
    assert logrank_test(y, g).method == "Log-rank test"
    assert "rho=1" in logrank_test(y, g, rho=1).method


def test_observed_events_sum_to_total() -> None:
    y = Surv.right([1, 2, 3, 4], [1, 0, 1, 1])
    g = ["a", "a", "b", "b"]
    result = logrank_test(y, g)
    # Standard log-rank (weight 1): observed events per group are plain counts.
    assert sum(result.observed.values()) == pytest.approx(3.0)
    np.testing.assert_allclose(sum(result.observed.values()), sum(result.expected.values()))


def test_requires_two_groups() -> None:
    with pytest.raises(ValueError, match="at least two groups"):
        logrank_test(Surv.right([1, 2, 3], [1, 1, 1]), ["a", "a", "a"])


def test_group_length_checked() -> None:
    with pytest.raises(ValueError, match="same length"):
        logrank_test(Surv.right([1, 2, 3], [1, 1, 1]), ["a", "b"])


def test_repr_is_informative() -> None:
    y = Surv.right([1, 2, 3, 4], [1, 1, 1, 1])
    text = repr(logrank_test(y, ["a", "a", "b", "b"]))
    assert "TestResult" in text
    assert "p_value" in text


def test_stratified_reduces_to_unstratified_with_one_stratum() -> None:
    y = Surv.right([1, 2, 3, 4, 5, 6], [1, 1, 1, 1, 1, 1])
    group = ["a", "a", "a", "b", "b", "b"]
    plain = logrank_test(y, group)
    one_stratum = logrank_test(y, group, strata=["s"] * 6)
    np.testing.assert_allclose(one_stratum.statistic, plain.statistic)
    assert one_stratum.method.startswith("Stratified")


def test_pairwise_shape_and_correction() -> None:
    y = Surv.right([1, 2, 3, 4, 5, 6, 7, 8, 9], [1, 1, 1, 1, 1, 1, 1, 1, 1])
    group = ["a", "a", "a", "b", "b", "b", "c", "c", "c"]
    pw = pairwise_logrank_test(y, group, correction="none")
    assert list(pw.columns) == ["group1", "group2", "statistic", "p_value", "p_adjusted"]
    assert len(pw) == 3  # C(3, 2)
    np.testing.assert_allclose(pw["p_value"], pw["p_adjusted"])  # 'none' leaves them equal


def test_pairwise_bonferroni_scales_raw() -> None:
    y = Surv.right([1, 2, 3, 4, 5, 6, 7, 8, 9], [1, 1, 1, 1, 1, 1, 1, 1, 1])
    group = ["a", "a", "a", "b", "b", "b", "c", "c", "c"]
    pw = pairwise_logrank_test(y, group, correction="bonferroni")
    np.testing.assert_allclose(pw["p_adjusted"], np.minimum(pw["p_value"] * 3, 1.0))


def test_pairwise_invalid_correction_raises() -> None:
    y = Surv.right([1, 2, 3, 4], [1, 1, 1, 1])
    with pytest.raises(ValueError, match="correction"):
        pairwise_logrank_test(y, ["a", "a", "b", "b"], correction="nope")


# ============================================================================
# Trend Test Tests
# ============================================================================


def test_trend_identical_groups_give_zero_statistic() -> None:
    """Linear trend across identical groups should show no trend."""
    # Groups with identical event patterns: no evidence of trend.
    time = [1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4]
    event = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    group = [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2]
    result = trend_test(Surv.right(time, event), group)
    assert result.statistic == pytest.approx(0.0, abs=1e-12)
    assert result.p_value == pytest.approx(1.0)
    assert result.df == 1


def test_trend_linear_deterioration_gives_significant_result() -> None:
    """Survival decreases linearly across groups: strong trend."""
    # Group 0: events at 10,11,12 (late, good survival)
    # Group 1: events at 5,6,7 (medium survival)
    # Group 2: events at 1,2,3 (early, poor survival)
    time = [10, 11, 12, 5, 6, 7, 1, 2, 3]
    event = [1, 1, 1, 1, 1, 1, 1, 1, 1]
    group = [0, 0, 0, 1, 1, 1, 2, 2, 2]
    result = trend_test(Surv.right(time, event), group)
    # Strong negative trend: higher group number = earlier events
    assert result.statistic > 3.0
    assert result.p_value < 0.05


def test_trend_default_scores_are_zero_one_two() -> None:
    """Default scores should be 0, 1, 2, ... for sorted groups."""
    y = Surv.right([1, 2, 3, 4, 5, 6], [1, 1, 1, 1, 1, 1])
    # Group 'a': events at 1,2 | Group 'b': events at 3,4 | Group 'c': events at 5,6
    g = ["a", "a", "b", "b", "c", "c"]
    result_default = trend_test(y, g)
    
    # Same data with explicit scores matching default
    result_explicit = trend_test(y, g, scores={"a": 0, "b": 1, "c": 2})
    
    np.testing.assert_allclose(result_default.statistic, result_explicit.statistic)
    np.testing.assert_allclose(result_default.p_value, result_explicit.p_value)


def test_trend_custom_scores_are_scale_invariant() -> None:
    """Chi-square statistic is scale-invariant to score scaling."""
    time = [1, 2, 3, 5, 6, 7]
    event = [1, 1, 1, 1, 1, 1]
    group = ["low", "low", "low", "high", "high", "high"]
    
    # Linear scores: 0, 1
    result_linear = trend_test(Surv.right(time, event), group, scores={"low": 0, "high": 1})
    
    # Scaled scores: 0, 10 (10x scaling)
    result_scaled = trend_test(Surv.right(time, event), group, scores={"low": 0, "high": 10})
    
    # Chi-square should be identical (scale-invariant)
    np.testing.assert_allclose(result_scaled.statistic, result_linear.statistic, rtol=1e-10)


def test_trend_negative_scores_work() -> None:
    """Negative scores should work correctly."""
    y = Surv.right([1, 2, 3, 4, 5, 6], [1, 1, 1, 1, 1, 1])
    g = [1, 1, 2, 2, 3, 3]
    
    # All positive scores
    result_pos = trend_test(y, g, scores={1: 0, 2: 1, 3: 2})
    
    # Mix of negative and positive
    result_neg = trend_test(y, g, scores={1: -2, 2: -1, 3: 0})
    
    # Both should give chi-square > 0 (some evidence of trend)
    assert result_pos.statistic > 0
    assert result_neg.statistic > 0


def test_trend_method_string_reflects_weights() -> None:
    """Method description should reflect weights and stratification."""
    y = Surv.right([1, 2, 3, 4, 5, 6], [1, 1, 1, 1, 1, 1])
    g = [0, 0, 1, 1, 2, 2]
    
    assert trend_test(y, g).method == "Linear trend test"
    assert "rho=1" in trend_test(y, g, rho=1).method
    assert "Stratified" in trend_test(y, g, strata=[0, 0, 0, 0, 1, 1]).method


def test_trend_with_fleming_harrington_weights() -> None:
    """Trend test should work with Fleming-Harrington weights."""
    time = [1, 2, 3, 10, 11, 12, 20, 21, 22]
    event = [1, 1, 1, 1, 1, 1, 1, 1, 1]
    group = [0, 0, 0, 1, 1, 1, 2, 2, 2]
    
    result_default = trend_test(Surv.right(time, event), group)
    result_peto = trend_test(Surv.right(time, event), group, rho=1)  # Peto-Peto
    
    # Different weights should give different statistics
    assert result_default.statistic != pytest.approx(result_peto.statistic)


def test_trend_stratified_reduces_to_unstratified() -> None:
    """Stratified trend test with one stratum should match unstratified."""
    # One stratum with clear trend
    time = [1, 2, 3, 10, 11, 12]
    event = [1, 1, 1, 1, 1, 1]
    group = [0, 0, 0, 1, 1, 1]
    
    result_unstrat = trend_test(Surv.right(time, event), group)
    result_strat = trend_test(Surv.right(time, event), group, strata=[0] * 6)
    
    # Should be identical
    np.testing.assert_allclose(result_strat.statistic, result_unstrat.statistic)
    assert result_strat.method.startswith("Stratified")


def test_trend_requires_two_groups() -> None:
    """Trend test needs at least two groups."""
    with pytest.raises(ValueError, match="at least two"):
        trend_test(Surv.right([1, 2, 3], [1, 1, 1]), [0, 0, 0])


def test_trend_requires_events() -> None:
    """Trend test needs at least one event."""
    with pytest.raises(ValueError, match="No events"):
        trend_test(Surv.right([1, 2, 3], [0, 0, 0]), [0, 1, 2])


def test_trend_group_length_checked() -> None:
    """Group length must match response length."""
    with pytest.raises(ValueError, match="same length"):
        trend_test(Surv.right([1, 2, 3], [1, 1, 1]), [0, 1])


def test_trend_strata_length_checked() -> None:
    """Strata length must match response length."""
    with pytest.raises(ValueError, match="same length"):
        trend_test(Surv.right([1, 2, 3], [1, 1, 1]), [0, 1, 2], strata=[0, 1])


def test_trend_scores_dict_validation() -> None:
    """Scores dict must contain all group labels."""
    y = Surv.right([1, 2, 3, 4], [1, 1, 1, 1])
    g = [0, 0, 1, 1]
    
    # Missing group 1
    with pytest.raises(KeyError):
        trend_test(y, g, scores={0: 0})


def test_trend_scores_array_length_validation() -> None:
    """Scores array must have same length as number of groups."""
    y = Surv.right([1, 2, 3, 4], [1, 1, 1, 1])
    g = [0, 0, 1, 1]
    
    # Wrong length
    with pytest.raises(ValueError, match="length"):
        trend_test(y, g, scores=[0, 1, 2])  # 3 scores but only 2 groups


def test_trend_observed_expected_in_result() -> None:
    """Result should contain observed and expected event counts."""
    y = Surv.right([1, 2, 3, 4, 5, 6], [1, 1, 1, 1, 1, 1])
    g = [0, 0, 1, 1, 2, 2]
    result = trend_test(y, g)
    
    # Should have observed and expected for each group
    assert len(result.observed) == 3
    assert len(result.expected) == 3
    assert 0 in result.observed and 1 in result.observed and 2 in result.observed


def test_trend_always_df_1() -> None:
    """Trend test should always have df=1."""
    y = Surv.right([1, 2, 3, 4, 5, 6], [1, 1, 1, 1, 1, 1])
    
    # 2 groups
    result2 = trend_test(y, [0, 0, 1, 1, 1, 1])
    assert result2.df == 1
    
    # 3 groups
    result3 = trend_test(y, [0, 0, 1, 1, 2, 2])
    assert result3.df == 1
    
    # 4 groups
    result4 = trend_test(y, [0, 1, 1, 2, 2, 3])
    assert result4.df == 1


def test_trend_with_right_censored_data() -> None:
    """Trend test with typical right-censored survival data."""
    y = Surv.right(
        [10, 20, 30, 15, 25, 35, 12, 22, 32],
        event=[1, 0, 1, 1, 0, 0, 1, 1, 1]
    )
    groups = [0, 0, 0, 1, 1, 1, 2, 2, 2]
    
    result = trend_test(y, groups)
    assert result.df == 1
    assert result.p_value >= 0.0 and result.p_value <= 1.0
    assert result.statistic >= 0.0


def test_trend_with_counting_process_data() -> None:
    """Trend test should work with counting-process (time-varying) data."""
    entry = [0, 1, 0, 1, 0, 2]
    exit = [1, 2, 1, 2, 2, 3]
    event = [0, 1, 1, 1, 0, 1]
    y = Surv.counting(entry, exit, event)
    g = [0, 0, 1, 1, 2, 2]
    
    result = trend_test(y, g)
    assert result.df == 1
    assert result.p_value >= 0.0 and result.p_value <= 1.0


def test_trend_with_weighted_observations() -> None:
    """Trend test should handle weighted observations."""
    y = Surv.right(
        time=[1, 2, 3, 4, 5, 6],
        event=[1, 1, 1, 1, 1, 1],
        weights=[0.5, 1.0, 1.5, 0.5, 1.0, 1.5]
    )
    g = [0, 0, 0, 1, 1, 1]
    
    result = trend_test(y, g)
    assert result.df == 1
    assert result.statistic > 0


def test_trend_identical_to_logrank_for_two_groups() -> None:
    """Trend test with two groups should be similar in behavior to logrank."""
    y = Surv.right([1, 2, 3, 4, 5, 6], [1, 1, 1, 1, 1, 1])
    g = [0, 0, 0, 1, 1, 1]
    
    trend_result = trend_test(y, g)
    logrank_result = logrank_test(y, g)
    
    # Both should test the same difference with 1 df for trend, varies for logrank
    # For 2 groups, logrank has df=1, so results should be very similar
    assert trend_result.df == 1
    assert logrank_result.df == 1
    np.testing.assert_allclose(trend_result.statistic, logrank_result.statistic, rtol=1e-10)
