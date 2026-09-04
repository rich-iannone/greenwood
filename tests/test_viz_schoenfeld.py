"""Tests for the Schoenfeld residual diagnostic visualization."""

from __future__ import annotations

import pytest

alt = pytest.importorskip("altair")

import greenwood as gw  # noqa: E402
from greenwood import CoxPH, Surv  # noqa: E402


@pytest.fixture
def cox_lung() -> CoxPH:
    df = gw.load_dataset("lung", backend="pandas")
    y = Surv.right(df["time"], event=(df["status"] == 2))
    return CoxPH().fit(y, df[["age", "sex"]])


def test_plot_schoenfeld_returns_hconcat(cox_lung: CoxPH) -> None:
    chart = gw.plot_schoenfeld(cox_lung)
    assert isinstance(chart, alt.HConcatChart)
    chart.to_dict()


def test_plot_schoenfeld_two_panels(cox_lung: CoxPH) -> None:
    spec = gw.plot_schoenfeld(cox_lung).to_dict()
    assert len(spec["hconcat"]) == 2


def test_plot_schoenfeld_log_transform(cox_lung: CoxPH) -> None:
    chart = gw.plot_schoenfeld(cox_lung, transform="log")
    chart.to_dict()


def test_plot_schoenfeld_km_transform(cox_lung: CoxPH) -> None:
    chart = gw.plot_schoenfeld(cox_lung, transform="km")
    chart.to_dict()


def test_plot_schoenfeld_rank_transform(cox_lung: CoxPH) -> None:
    chart = gw.plot_schoenfeld(cox_lung, transform="rank")
    chart.to_dict()


def test_plot_schoenfeld_no_zph(cox_lung: CoxPH) -> None:
    chart = gw.plot_schoenfeld(cox_lung, show_zph=False)
    chart.to_dict()


def test_plot_schoenfeld_with_title(cox_lung: CoxPH) -> None:
    chart = gw.plot_schoenfeld(cox_lung, title="PH check")
    spec = chart.to_dict()
    assert spec.get("title") == "PH check"


def test_plot_schoenfeld_invalid_backend(cox_lung: CoxPH) -> None:
    with pytest.raises(ValueError, match="backend"):
        gw.plot_schoenfeld(cox_lung, backend="matplotlib")  # type: ignore[arg-type]


def test_plot_schoenfeld_invalid_transform(cox_lung: CoxPH) -> None:
    with pytest.raises(ValueError, match="transform"):
        gw.plot_schoenfeld(cox_lung, transform="invalid")


def test_plot_schoenfeld_plotnine_backend(cox_lung: CoxPH) -> None:
    p9 = pytest.importorskip("plotnine")
    p = gw.plot_schoenfeld(cox_lung, backend="plotnine")
    assert isinstance(p, p9.ggplot)
