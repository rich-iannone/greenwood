"""Tests for forest plot visualization."""

from __future__ import annotations

import numpy as np
import pytest

from greenwood.viz._forest import _forest_plot_data, forest_plot


def test_forest_plot_data_array_input() -> None:
    """Test forest_plot_data with array inputs."""
    data = _forest_plot_data(
        estimates=[0.85, 1.02, 1.15],
        ci_lower=[0.71, 0.89, 0.98],
        ci_upper=[1.01, 1.17, 1.35],
        labels=["Age", "Sex", "ECOG"],
        scale="log",
    )

    assert data["scale"] == "log"
    assert len(data["data"]) == 3
    assert data["data"][0]["label"] == "Age"
    assert "estimate" in data["data"][0]
    assert "ci_lower" in data["data"][0]
    assert "ci_upper" in data["data"][0]


def test_forest_plot_data_dict_input() -> None:
    """Test forest_plot_data with dict input."""
    input_dict = {
        "estimate": [0.85, 1.02],
        "ci_lower": [0.71, 0.89],
        "ci_upper": [1.01, 1.17],
        "labels": ["Age", "Sex"],
    }
    data = _forest_plot_data(input_dict, scale="log")

    assert len(data["data"]) == 2
    assert data["data"][0]["label"] == "Age"


def test_forest_plot_data_log_scale() -> None:
    """Test that log scale correctly transforms estimates."""
    # HR = 1 (no effect) should become log(1) = 0
    data = _forest_plot_data(
        estimates=[1.0],
        ci_lower=[0.8],
        ci_upper=[1.2],
        scale="log",
    )

    # log(1) = 0
    assert abs(data["data"][0]["estimate"] - 0.0) < 1e-10
    # log(0.8) < 0, log(1.2) > 0
    assert data["data"][0]["ci_lower"] < 0
    assert data["data"][0]["ci_upper"] > 0


def test_forest_plot_data_linear_scale() -> None:
    """Test that linear scale does no transformation."""
    data = _forest_plot_data(
        estimates=[5.0, -3.0],
        ci_lower=[2.0, -5.0],
        ci_upper=[8.0, -1.0],
        scale="linear",
    )

    assert data["data"][0]["estimate"] == 5.0
    assert data["data"][0]["ci_lower"] == 2.0
    assert data["data"][0]["ci_upper"] == 8.0


def test_forest_plot_data_reference_line() -> None:
    """Test reference line defaults."""
    log_data = _forest_plot_data(
        estimates=[0.9],
        ci_lower=[0.7],
        ci_upper=[1.1],
        scale="log",
    )
    # Log scale defaults to 0 (log(1) = 0)
    assert log_data["reference_line"] == 0.0

    linear_data = _forest_plot_data(
        estimates=[5.0],
        ci_lower=[2.0],
        ci_upper=[8.0],
        scale="linear",
    )
    # Linear scale defaults to 0
    assert linear_data["reference_line"] == 0.0


def test_forest_plot_data_missing_ci() -> None:
    """Test error when CI bounds are missing."""
    with pytest.raises(ValueError, match="ci_lower and ci_upper must be provided"):
        _forest_plot_data(estimates=[0.9, 1.0])


def test_forest_plot_data_shape_mismatch() -> None:
    """Test error when shapes don't match."""
    with pytest.raises(ValueError, match="same shape"):
        _forest_plot_data(
            estimates=[0.9, 1.0],
            ci_lower=[0.7],  # Wrong length
            ci_upper=[1.1, 1.2],
        )


def test_forest_plot_data_label_length() -> None:
    """Test error when label length doesn't match."""
    with pytest.raises(ValueError, match="same length"):
        _forest_plot_data(
            estimates=[0.9, 1.0],
            ci_lower=[0.7, 0.8],
            ci_upper=[1.1, 1.2],
            labels=["Only one"],  # Wrong length
        )


def test_forest_plot_data_invalid_scale() -> None:
    """Test error for invalid scale."""
    with pytest.raises(ValueError, match="must be 'log' or 'linear'"):
        _forest_plot_data(
            estimates=[0.9],
            ci_lower=[0.7],
            ci_upper=[1.1],
            scale="invalid",
        )


def test_forest_plot_altair_basic() -> None:
    """Test forest_plot returns an Altair chart with raw parameters."""
    try:
        import altair  # noqa: F401
    except ImportError:
        pytest.skip("altair not installed")

    chart = forest_plot(
        estimates=[0.85, 1.02],
        ci_lower=[0.71, 0.89],
        ci_upper=[1.01, 1.17],
        labels=["Age", "Sex"],
        scale="log",
        title="Test",
    )
    assert chart is not None


def test_forest_plot_backend_parameter() -> None:
    """Test forest_plot backend parameter validation."""
    with pytest.raises(ValueError, match="backend must be 'altair'"):
        forest_plot(
            estimates=[0.85], ci_lower=[0.71], ci_upper=[1.01], scale="log", backend="invalid"
        )


def test_forest_plot_missing_altair_import() -> None:
    """Test error when altair is not available."""
    import sys

    altair_backup = sys.modules.get("altair")
    sys.modules["altair"] = None  # type: ignore

    try:
        with pytest.raises(ImportError, match="altair"):
            forest_plot(estimates=[0.85], ci_lower=[0.71], ci_upper=[1.01], scale="log")
    finally:
        if altair_backup is None:
            del sys.modules["altair"]
        else:
            sys.modules["altair"] = altair_backup
