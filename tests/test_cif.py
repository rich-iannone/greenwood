"""Tests for CIF plot visualization."""

from __future__ import annotations

import numpy as np
import pytest

from greenwood.viz._cif import _cif_plot_data, cif_plot


def test_cif_plot_data_single_group() -> None:
    """Test cif_plot_data with single group."""
    times = np.array([30, 60, 90])
    cif = np.array(
        [
            [0.05, 0.02],
            [0.12, 0.05],
            [0.20, 0.10],
        ]
    )

    data = _cif_plot_data(time=times, cif=cif, event_names=["Event A", "Event B"])

    assert "data" in data
    assert len(data["data"]) == 6  # 3 times × 2 events
    assert data["events"] == ["Event A", "Event B"]

    # Check structure
    for row in data["data"]:
        assert "time" in row
        assert "cif" in row
        assert "event" in row
        assert "group" in row


def test_cif_plot_data_multiple_groups() -> None:
    """Test cif_plot_data with multiple groups."""
    times = np.array([30, 60])
    cif = {
        "Group A": np.array([[0.05, 0.02], [0.12, 0.05]]),
        "Group B": np.array([[0.02, 0.01], [0.06, 0.03]]),
    }

    data = _cif_plot_data(time=times, cif=cif, event_names=["Event 1", "Event 2"])

    assert len(data["data"]) == 8  # 2 groups × 2 times × 2 events
    assert data["groups"] == ["Group A", "Group B"]


def test_cif_plot_data_default_names() -> None:
    """Test that default event/group names are created."""
    times = np.array([30, 60])
    cif = np.array([[0.05, 0.02], [0.12, 0.05]])

    data = _cif_plot_data(time=times, cif=cif)

    assert data["events"] == ["Event 1", "Event 2"]
    assert "Overall" in data["groups"]


def test_cif_plot_data_group_names_mapping() -> None:
    """Test custom group name mapping."""
    times = np.array([30])
    cif = {
        1: np.array([[0.05, 0.02]]),
        2: np.array([[0.02, 0.01]]),
    }

    group_names = {1: "Control", 2: "Treatment"}
    data = _cif_plot_data(time=times, cif=cif, group_names=group_names)

    assert data["groups"] == ["Control", "Treatment"]


def test_cif_plot_data_time_dimension() -> None:
    """Test that time dimension must match CIF."""
    times = np.array([30, 60])
    cif = np.array([[0.05, 0.02]])  # Wrong: only 1 time point

    with pytest.raises(ValueError, match="must have 2 rows"):
        _cif_plot_data(time=times, cif=cif)


def test_cif_plot_data_event_names_length() -> None:
    """Test event_names length must match CIF."""
    times = np.array([30])
    cif = np.array([[0.05, 0.02]])  # 2 events

    with pytest.raises(ValueError, match="event_names length"):
        _cif_plot_data(time=times, cif=cif, event_names=["Only one"])


def test_cif_plot_data_time_format() -> None:
    """Test that time is converted to 1-D."""
    times = np.array([[30], [60]])  # 2-D, should raise
    cif = np.array([[0.05, 0.02], [0.12, 0.05]])

    with pytest.raises(ValueError, match="time must be 1-D"):
        _cif_plot_data(time=times, cif=cif)


def test_cif_plot_data_values_range() -> None:
    """Test that CIF values are properly captured in tidy format."""
    times = np.array([30, 60])
    cif = np.array([[0.05, 0.02], [0.12, 0.05]])

    data = _cif_plot_data(time=times, cif=cif, event_names=["A", "B"])

    # Check first time point, first event
    row1 = [r for r in data["data"] if r["time"] == 30 and r["event"] == "A"][0]
    assert row1["cif"] == 0.05

    # Check second time point, second event
    row2 = [r for r in data["data"] if r["time"] == 60 and r["event"] == "B"][0]
    assert row2["cif"] == 0.05


def test_cif_plot_altair_basic() -> None:
    """Test cif_plot returns an Altair chart with raw parameters."""
    try:
        import altair  # noqa: F401
    except ImportError:
        pytest.skip("altair not installed")

    times = np.array([30, 60])
    cif = np.array([[0.05, 0.02], [0.12, 0.05]])

    chart = cif_plot(time=times, cif=cif, event_names=["A", "B"], title="Test CIF")
    assert chart is not None


def test_cif_plot_backend_parameter() -> None:
    """Test cif_plot backend parameter validation."""
    times = np.array([30, 60])
    cif = np.array([[0.05, 0.02], [0.12, 0.05]])

    with pytest.raises(ValueError, match="backend must be 'altair'"):
        cif_plot(time=times, cif=cif, event_names=["A", "B"], backend="invalid")
