"""Tests for log-rank sample size and power calculators (Schoenfeld's method).

No R power package is installed, so correctness is pinned to the well-known Schoenfeld
textbook constants, the exact inverse relationship between events and power, and the expected
monotonic behavior.
"""

from __future__ import annotations

import pytest

from greenwood import logrank_n_events, logrank_power, logrank_sample_size


def test_textbook_event_counts() -> None:
    # Classic Schoenfeld results for a two-sided 5% test.
    assert logrank_n_events(0.5, power=0.9) == 88
    assert logrank_n_events(0.5, power=0.8) == 66


def test_symmetric_in_hazard_ratio() -> None:
    assert logrank_n_events(0.5) == logrank_n_events(2.0)
    assert logrank_power(0.5, 80) == pytest.approx(logrank_power(2.0, 80))


def test_events_and_power_are_inverse() -> None:
    for hr in (0.4, 0.6, 0.75):
        for target in (0.7, 0.8, 0.9):
            d = logrank_n_events(hr, power=target)
            # The rounded-up event count achieves at least the target power.
            assert logrank_power(hr, d) >= target
            # And one event short falls below it.
            assert logrank_power(hr, d - 1) < target


def test_monotonicity() -> None:
    # A hazard ratio nearer 1 is harder to detect, needing more events.
    assert logrank_n_events(0.8) > logrank_n_events(0.5)
    # Higher power needs more events.
    assert logrank_n_events(0.5, power=0.9) > logrank_n_events(0.5, power=0.8)
    # A balanced design minimizes the events required.
    assert logrank_n_events(0.5, allocation=0.5) <= logrank_n_events(0.5, allocation=0.3)


def test_one_sided_needs_fewer_events() -> None:
    assert logrank_n_events(0.5, sides=1) < logrank_n_events(0.5, sides=2)


def test_sample_size_scales_with_event_probability() -> None:
    assert logrank_sample_size(0.5, 0.5, power=0.9) == 175  # ceil(87.48 / 0.5)
    # Rarer events require a larger sample for the same number of events.
    assert logrank_sample_size(0.5, 0.25) > logrank_sample_size(0.5, 0.5)


def test_input_validation() -> None:
    with pytest.raises(ValueError, match="hazard_ratio"):
        logrank_n_events(1.0)
    with pytest.raises(ValueError, match="hazard_ratio"):
        logrank_n_events(-0.5)
    with pytest.raises(ValueError, match="power"):
        logrank_n_events(0.5, power=1.5)
    with pytest.raises(ValueError, match="alpha"):
        logrank_n_events(0.5, alpha=0)
    with pytest.raises(ValueError, match="sides"):
        logrank_n_events(0.5, sides=3)
    with pytest.raises(ValueError, match="n_events"):
        logrank_power(0.5, 0)
    with pytest.raises(ValueError, match="prob_event"):
        logrank_sample_size(0.5, 1.5)
