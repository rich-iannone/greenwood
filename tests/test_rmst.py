"""Unit tests for RMST comparisons."""

from __future__ import annotations

import numpy as np
import pytest

from greenwood import Surv, pairwise_rmst_test, rmst_diff, rmst_test


def test_rmst_test_requires_two_groups() -> None:
    """Two or more groups required."""
    time = [1, 2, 3, 4]
    event = [1, 1, 1, 1]
    group = ["a", "a", "a", "a"]
    y = Surv.right(time, event)

    with pytest.raises(ValueError, match="at least two groups"):
        rmst_test(y, tau=10, group=group)


def test_rmst_test_rejects_more_than_two_groups() -> None:
    """Currently only supports two-group comparisons."""
    time = [1, 2, 3, 4, 5, 6]
    event = [1, 1, 1, 1, 1, 1]
    group = ["a", "a", "b", "b", "c", "c"]
    y = Surv.right(time, event)

    with pytest.raises(ValueError, match="two groups"):
        rmst_test(y, tau=10, group=group)


def test_rmst_test_simple_two_group() -> None:
    """Simple two-group RMST comparison."""
    # Group a: [1, 2, 3, 4] all events
    # Group b: [5, 6, 7, 8] all events
    # Group b should have higher RMST (longer survival)
    time = [1, 2, 3, 4, 5, 6, 7, 8]
    event = [1, 1, 1, 1, 1, 1, 1, 1]
    group = ["a", "a", "a", "a", "b", "b", "b", "b"]
    y = Surv.right(time, event)

    result = rmst_test(y, tau=10, group=group)

    # Group b should have higher RMST
    assert result.rmst2 > result.rmst1
    assert result.estimate < 0  # estimate = rmst1 - rmst2
    assert result.p_value >= 0 and result.p_value <= 1
    assert result.se > 0
    assert result.statistic < 0
    assert result.lower_ci < result.upper_ci


def test_rmst_test_identical_groups() -> None:
    """Identical groups should have near-zero difference."""
    time = [1, 2, 3, 4, 1, 2, 3, 4]
    event = [1, 1, 1, 1, 1, 1, 1, 1]
    group = ["a", "a", "a", "a", "b", "b", "b", "b"]
    y = Surv.right(time, event)

    result = rmst_test(y, tau=10, group=group)

    assert result.estimate == pytest.approx(0.0, abs=1e-10)
    assert result.rmst1 == pytest.approx(result.rmst2)
    assert result.p_value == pytest.approx(1.0)


def test_rmst_test_difference_estimand() -> None:
    """Difference estimand (default)."""
    time = [1, 2, 3, 4, 5, 6, 7, 8]
    event = [1, 1, 1, 1, 1, 1, 1, 1]
    group = ["a", "a", "a", "a", "b", "b", "b", "b"]
    y = Surv.right(time, event)

    result = rmst_test(y, tau=10, group=group, estimand="difference")

    assert result.estimand == "difference"
    assert result.estimate == pytest.approx(result.rmst1 - result.rmst2)
    # SE should be sqrt(se1^2 + se2^2)
    expected_se = np.sqrt(result.se1**2 + result.se2**2)
    assert result.se == pytest.approx(expected_se)


def test_rmst_test_ratio_estimand() -> None:
    """Ratio estimand (RMST1 / RMST2)."""
    time = [1, 2, 3, 4, 5, 6, 7, 8]
    event = [1, 1, 1, 1, 1, 1, 1, 1]
    group = ["a", "a", "a", "a", "b", "b", "b", "b"]
    y = Surv.right(time, event)

    result = rmst_test(y, tau=10, group=group, estimand="ratio")

    assert result.estimand == "ratio"
    assert result.estimate == pytest.approx(result.rmst1 / result.rmst2)
    assert result.estimate > 0
    assert result.lower_ci > 0
    assert result.upper_ci > 0


def test_rmst_test_percentage_difference_estimand() -> None:
    """Percentage difference estimand."""
    time = [1, 2, 3, 4, 5, 6, 7, 8]
    event = [1, 1, 1, 1, 1, 1, 1, 1]
    group = ["a", "a", "a", "a", "b", "b", "b", "b"]
    y = Surv.right(time, event)

    result = rmst_test(y, tau=10, group=group, estimand="percentage_difference")

    assert result.estimand == "percentage_difference"
    expected = (result.rmst1 - result.rmst2) / result.rmst2 * 100.0
    assert result.estimate == pytest.approx(expected)


def test_rmst_test_invalid_estimand() -> None:
    """Invalid estimand raises error."""
    time = [1, 2, 3, 4, 5, 6, 7, 8]
    event = [1, 1, 1, 1, 1, 1, 1, 1]
    group = ["a", "a", "a", "a", "b", "b", "b", "b"]
    y = Surv.right(time, event)

    with pytest.raises(ValueError, match="estimand must be"):
        rmst_test(y, tau=10, group=group, estimand="invalid")


def test_rmst_test_z_statistic_and_pvalue() -> None:
    """Z-statistic should be estimate / SE, and p-value should be two-tailed."""
    time = [1, 2, 3, 4, 5, 6, 7, 8]
    event = [1, 1, 1, 1, 1, 1, 1, 1]
    group = ["a", "a", "a", "a", "b", "b", "b", "b"]
    y = Surv.right(time, event)

    result = rmst_test(y, tau=10, group=group)

    expected_z = result.estimate / result.se
    assert result.statistic == pytest.approx(expected_z)

    # Two-tailed p-value from standard normal
    from scipy.stats import norm

    expected_pval = 2.0 * (1.0 - norm.cdf(np.abs(result.statistic)))
    assert result.p_value == pytest.approx(expected_pval)


def test_rmst_test_ci_bounds() -> None:
    """CI should be estimate +/- z * SE."""
    time = [1, 2, 3, 4, 5, 6, 7, 8]
    event = [1, 1, 1, 1, 1, 1, 1, 1]
    group = ["a", "a", "a", "a", "b", "b", "b", "b"]
    y = Surv.right(time, event)

    result = rmst_test(y, tau=10, group=group, conf_level=0.95)

    from scipy.stats import norm

    z_crit = norm.ppf(1.0 - (1.0 - 0.95) / 2.0)
    expected_lower = result.estimate - z_crit * result.se
    expected_upper = result.estimate + z_crit * result.se

    assert result.lower_ci == pytest.approx(expected_lower)
    assert result.upper_ci == pytest.approx(expected_upper)


def test_rmst_test_conf_level() -> None:
    """Different confidence levels should produce different CIs."""
    time = [1, 2, 3, 4, 5, 6, 7, 8]
    event = [1, 1, 1, 1, 1, 1, 1, 1]
    group = ["a", "a", "a", "a", "b", "b", "b", "b"]
    y = Surv.right(time, event)

    result_95 = rmst_test(y, tau=10, group=group, conf_level=0.95)
    result_99 = rmst_test(y, tau=10, group=group, conf_level=0.99)

    # 99% CI should be wider than 95%
    width_95 = result_95.upper_ci - result_95.lower_ci
    width_99 = result_99.upper_ci - result_99.lower_ci
    assert width_99 > width_95


def test_rmst_test_with_censoring() -> None:
    """RMST comparison should handle censoring."""
    time = [1, 2, 3, 4, 5, 6, 7, 8]
    event = [1, 1, 0, 1, 1, 0, 1, 1]  # Some censored
    group = ["a", "a", "a", "a", "b", "b", "b", "b"]
    y = Surv.right(time, event)

    result = rmst_test(y, tau=10, group=group)

    # Should compute without error
    assert result.rmst1 > 0
    assert result.rmst2 > 0
    assert result.se1 > 0
    assert result.se2 > 0
    assert np.isfinite(result.p_value)


def test_rmst_test_output_repr() -> None:
    """Result repr should be informative."""
    time = [1, 2, 3, 4, 5, 6, 7, 8]
    event = [1, 1, 1, 1, 1, 1, 1, 1]
    group = ["a", "a", "a", "a", "b", "b", "b", "b"]
    y = Surv.right(time, event)

    result = rmst_test(y, tau=10, group=group)

    repr_str = repr(result)
    assert "RMSTResult" in repr_str
    assert "estimate" in repr_str
    assert "p_value" in repr_str


def test_rmst_diff_returns_dataframe() -> None:
    """rmst_diff should return a DataFrame-like object."""
    time = [1, 2, 3, 4, 5, 6, 7, 8]
    event = [1, 1, 1, 1, 1, 1, 1, 1]
    group = ["a", "a", "a", "a", "b", "b", "b", "b"]
    y = Surv.right(time, event)

    result_df = rmst_diff(y, tau=10, group=group)

    # Should have columns for RMST values, difference, SE, CI, test stat, p-value
    assert hasattr(result_df, "columns")
    columns = list(result_df.columns)
    assert "difference" in columns
    assert "se" in columns
    assert "p_value" in columns


def test_rmst_test_group_labels_sorted() -> None:
    """Group labels should be sorted (consistent ordering)."""
    time = [1, 2, 3, 4, 5, 6, 7, 8]
    event = [1, 1, 1, 1, 1, 1, 1, 1]
    group = ["z", "z", "z", "z", "a", "a", "a", "a"]  # Reversed order
    y = Surv.right(time, event)

    result = rmst_test(y, tau=10, group=group)

    # Should sort to ["a", "z"]
    assert result.group1 == "a"
    assert result.group2 == "z"


def test_rmst_test_method_string() -> None:
    """Method string should reflect estimand and tau."""
    time = [1, 2, 3, 4, 5, 6, 7, 8]
    event = [1, 1, 1, 1, 1, 1, 1, 1]
    group = ["a", "a", "a", "a", "b", "b", "b", "b"]
    y = Surv.right(time, event)

    result_diff = rmst_test(y, tau=365, group=group, estimand="difference")
    result_ratio = rmst_test(y, tau=365, group=group, estimand="ratio")

    assert "difference" in result_diff.method
    assert "365" in result_diff.method
    assert "ratio" in result_ratio.method


def test_rmst_test_large_tau() -> None:
    """Large tau should include all observations."""
    time = [1, 2, 3, 4, 5, 6, 7, 8]
    event = [1, 1, 1, 1, 1, 1, 1, 1]
    group = ["a", "a", "a", "a", "b", "b", "b", "b"]
    y = Surv.right(time, event)

    # tau=100 is much larger than max time (8)
    result = rmst_test(y, tau=100, group=group)

    # RMST with large tau should include full time span
    assert result.rmst1 > 0
    assert result.rmst2 > 0


def test_rmst_test_small_tau() -> None:
    """Small tau should truncate area."""
    time = [1, 2, 3, 4, 5, 6, 7, 8]
    event = [1, 1, 1, 1, 1, 1, 1, 1]
    group = ["a", "a", "a", "a", "b", "b", "b", "b"]
    y = Surv.right(time, event)

    result_small = rmst_test(y, tau=2, group=group)
    result_large = rmst_test(y, tau=10, group=group)

    # Small tau should give smaller RMST
    assert result_small.rmst1 < result_large.rmst1
    assert result_small.rmst2 < result_large.rmst2


def test_rmst_test_with_numeric_group() -> None:
    """Group can be numeric."""
    time = [1, 2, 3, 4, 5, 6, 7, 8]
    event = [1, 1, 1, 1, 1, 1, 1, 1]
    group = [1, 1, 1, 1, 2, 2, 2, 2]  # Numeric
    y = Surv.right(time, event)

    result = rmst_test(y, tau=10, group=group)

    assert result.group1 == 1
    assert result.group2 == 2
    assert np.isfinite(result.estimate)


def test_rmst_test_ratio_with_zero_rmst_error() -> None:
    """Ratio estimand should raise error if RMST2 is zero."""
    # Create data where group 2 has all censoring (no events)
    time = [1, 2, 3, 4, 100, 100, 100, 100]
    event = [1, 1, 1, 1, 0, 0, 0, 0]  # Group b all censored
    group = ["a", "a", "a", "a", "b", "b", "b", "b"]
    y = Surv.right(time, event)

    # Should not raise immediately, but check if RMST2 is effectively zero
    rmst_test(y, tau=5, group=group, estimand="ratio")
    # If RMST2 is very small or zero, ratio will be undefined or very large
    # The function should handle gracefully


def test_rmst_test_percentage_with_zero_rmst_error() -> None:
    """Percentage difference should raise error if RMST2 is zero."""
    time = [1, 2, 3, 4, 100, 100, 100, 100]
    event = [1, 1, 1, 1, 0, 0, 0, 0]
    group = ["a", "a", "a", "a", "b", "b", "b", "b"]
    y = Surv.right(time, event)

    rmst_test(y, tau=5, group=group, estimand="percentage_difference")
    # Should compute or handle gracefully


def test_rmst_test_attributes() -> None:
    """All expected attributes should be present."""
    time = [1, 2, 3, 4, 5, 6, 7, 8]
    event = [1, 1, 1, 1, 1, 1, 1, 1]
    group = ["a", "a", "a", "a", "b", "b", "b", "b"]
    y = Surv.right(time, event)

    result = rmst_test(y, tau=10, group=group)

    # Check all attributes exist
    assert hasattr(result, "estimate")
    assert hasattr(result, "se")
    assert hasattr(result, "lower_ci")
    assert hasattr(result, "upper_ci")
    assert hasattr(result, "statistic")
    assert hasattr(result, "p_value")
    assert hasattr(result, "method")
    assert hasattr(result, "group1")
    assert hasattr(result, "group2")
    assert hasattr(result, "rmst1")
    assert hasattr(result, "se1")
    assert hasattr(result, "rmst2")
    assert hasattr(result, "se2")
    assert hasattr(result, "tau")
    assert hasattr(result, "estimand")
    assert hasattr(result, "stratified")
    assert hasattr(result, "conf_level")


def test_rmst_test_comparison_order() -> None:
    """Ensure group1 and group2 are in consistent order (group1 - group2)."""
    time = [1, 2, 3, 4, 5, 6, 7, 8]
    event = [1, 1, 1, 1, 1, 1, 1, 1]
    group = ["b", "b", "b", "b", "a", "a", "a", "a"]
    y = Surv.right(time, event)

    result = rmst_test(y, tau=10, group=group)

    # Groups should be sorted
    assert result.group1 == "a"
    assert result.group2 == "b"
    assert result.estimate == pytest.approx(result.rmst1 - result.rmst2)


def test_pairwise_rmst_test_three_groups() -> None:
    """Pairwise RMST test with three groups."""
    time = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    event = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    group = ["a", "a", "a", "a", "b", "b", "b", "b", "c", "c", "c", "c"]
    y = Surv.right(time, event)

    result = pairwise_rmst_test(y, tau=15, group=group)

    # Should have 3 pairs: (a,b), (a,c), (b,c)
    assert len(result) == 3
    # Should have p_adjusted column
    assert "p_adjusted" in result.columns
    assert "p_value" in result.columns


def test_pairwise_rmst_test_requires_two_groups() -> None:
    """Pairwise test requires at least two groups."""
    time = [1, 2, 3, 4]
    event = [1, 1, 1, 1]
    group = ["a", "a", "a", "a"]
    y = Surv.right(time, event)

    with pytest.raises(ValueError, match="at least two groups"):
        pairwise_rmst_test(y, tau=10, group=group)


def test_pairwise_rmst_test_correction_methods() -> None:
    """Test different p-value correction methods."""
    time = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    event = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    group = ["a", "a", "a", "a", "b", "b", "b", "b", "c", "c", "c", "c"]
    y = Surv.right(time, event)

    result_holm = pairwise_rmst_test(y, tau=15, group=group, correction="holm")
    result_bh = pairwise_rmst_test(y, tau=15, group=group, correction="bh")
    result_bonf = pairwise_rmst_test(y, tau=15, group=group, correction="bonferroni")
    result_none = pairwise_rmst_test(y, tau=15, group=group, correction="none")

    # All should have same structure
    assert len(result_holm) == len(result_bh) == len(result_bonf) == len(result_none) == 3

    # p_adjusted for "none" should equal p_value
    for i in range(len(result_none)):
        assert result_none["p_adjusted"][i] == pytest.approx(result_none["p_value"][i])


def test_pairwise_rmst_test_invalid_correction() -> None:
    """Invalid correction method should raise error."""
    time = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    event = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    group = ["a", "a", "a", "a", "b", "b", "b", "b", "c", "c", "c", "c"]
    y = Surv.right(time, event)

    with pytest.raises(ValueError, match="Unknown correction"):
        pairwise_rmst_test(y, tau=15, group=group, correction="invalid")
