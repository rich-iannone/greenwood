"""Tests for CIF plot data preparation and visualization."""

from __future__ import annotations

import numpy as np
import pytest

import greenwood as gw
from greenwood import AalenJohansen, Surv
from greenwood.viz._cif import _step_data


def _make_aj(*, grouped: bool = False) -> AalenJohansen:
    """Minimal AalenJohansen fit: 4 subjects, 2 competing causes."""
    y = Surv.multistate([1, 2, 3, 4], event=[1, 2, 1, 0], states=("pcm", "death"))
    if grouped:
        by = np.array(["A", "A", "B", "B"])
        return AalenJohansen().fit(y, by=by)
    return AalenJohansen().fit(y)


def test_step_data_columns() -> None:
    """_step_data returns the expected column keys."""
    data = _step_data(_make_aj())
    assert set(data.keys()) == {"time", "estimate", "conf_low", "conf_high", "cause", "group"}


def test_step_data_t0_anchor() -> None:
    """Each (group, cause) pair is prepended with a t=0, CIF=0 anchor row."""
    aj = _make_aj()
    data = _step_data(aj)
    n_causes = len(aj._causes)
    n_groups = len(aj._blocks)
    # Each series starts at t=0
    pairs = zip(data["time"], data["estimate"], strict=True)
    zeros = [t for t, e in pairs if t == 0.0 and e == 0.0]
    assert len(zeros) == n_causes * n_groups


def test_step_data_group_labels_unstratified() -> None:
    """Unstratified fit labels the single group 'Overall'."""
    data = _step_data(_make_aj())
    assert set(data["group"]) == {"Overall"}


def test_step_data_group_labels_stratified() -> None:
    """Stratified fit uses the by= values as group labels."""
    data = _step_data(_make_aj(grouped=True))
    assert set(data["group"]) == {"A", "B"}


def test_step_data_cause_labels() -> None:
    """Cause labels match the state names from the Surv response."""
    data = _step_data(_make_aj())
    assert set(data["cause"]) == {"pcm", "death"}


def test_step_data_monotone_per_series() -> None:
    """CIF values are non-decreasing within each (group, cause) series."""
    data = _step_data(_make_aj())
    for group in set(data["group"]):
        for cause in set(data["cause"]):
            vals = [
                e
                for g, c, e in zip(data["group"], data["cause"], data["estimate"], strict=True)
                if g == group and c == cause
            ]
            assert all(b >= a - 1e-12 for a, b in zip(vals, vals[1:], strict=False))


def test_step_data_estimates_bounded() -> None:
    """All CIF estimates are in [0, 1]."""
    data = _step_data(_make_aj())
    assert all(0.0 <= e <= 1.0 for e in data["estimate"])


def test_plot_cif_invalid_backend() -> None:
    """An unknown backend raises ValueError."""
    with pytest.raises(ValueError, match="backend"):
        gw.plot_cif(_make_aj(), backend="invalid")
