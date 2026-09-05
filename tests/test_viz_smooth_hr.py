"""Tests for the smooth hazard ratio visualization."""

from __future__ import annotations

import pytest

alt = pytest.importorskip("altair")

import greenwood as gw  # noqa: E402
from greenwood import CoxPH, Surv  # noqa: E402


@pytest.fixture
def shr():
    df = gw.load_dataset("lung", backend="pandas")
    y = Surv.right(df["time"], event=(df["status"] == 2))
    cox = CoxPH().fit(y, df[["age", "sex"]])
    return cox.smooth_hr("age")


def test_returns_altair_chart(shr) -> None:
    chart = gw.plot_smooth_hr(shr)
    assert isinstance(chart, alt.LayerChart | alt.Chart | alt.VConcatChart | alt.HConcatChart)
    chart.to_dict()


def test_default_scale_log_hr(shr) -> None:
    chart = gw.plot_smooth_hr(shr)
    chart.to_dict()


def test_scale_hr(shr) -> None:
    chart = gw.plot_smooth_hr(shr, scale="hr")
    chart.to_dict()


def test_with_title(shr) -> None:
    chart = gw.plot_smooth_hr(shr, title="Smooth HR for age")
    spec = chart.to_dict()
    assert spec.get("title") == "Smooth HR for age" or any(
        "Smooth HR for age" in str(v) for v in spec.values()
    )


def test_custom_xlab(shr) -> None:
    chart = gw.plot_smooth_hr(shr, xlab="Age (years)")
    chart.to_dict()


def test_custom_ylab(shr) -> None:
    chart = gw.plot_smooth_hr(shr, ylab="log(HR)")
    chart.to_dict()


def test_custom_dimensions(shr) -> None:
    chart = gw.plot_smooth_hr(shr, width=400, height=200)
    chart.to_dict()


def test_invalid_backend(shr) -> None:
    with pytest.raises(ValueError, match="backend"):
        gw.plot_smooth_hr(shr, backend="matplotlib")  # type: ignore[arg-type]


def test_plotnine_backend(shr) -> None:
    p9 = pytest.importorskip("plotnine")
    p = gw.plot_smooth_hr(shr, backend="plotnine")
    assert isinstance(p, p9.ggplot)


def test_plotnine_hr_scale(shr) -> None:
    p9 = pytest.importorskip("plotnine")
    p = gw.plot_smooth_hr(shr, scale="hr", backend="plotnine")
    assert isinstance(p, p9.ggplot)


def test_plotnine_with_title(shr) -> None:
    p9 = pytest.importorskip("plotnine")
    p = gw.plot_smooth_hr(shr, title="Test title", backend="plotnine")
    assert isinstance(p, p9.ggplot)
