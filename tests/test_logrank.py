"""Unit tests for the log-rank / G-rho group comparison test."""

from __future__ import annotations

import numpy as np
import pytest

from greenwood import Surv, logrank_test, pairwise_logrank_test


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
