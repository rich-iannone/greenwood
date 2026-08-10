"""Tests for the influence diagnostic visualization."""

from __future__ import annotations

import pytest

alt = pytest.importorskip("altair")

import greenwood as gw  # noqa: E402
from greenwood import CoxPH, Surv  # noqa: E402


@pytest.fixture
def cox_lung() -> CoxPH:
    df = gw.load_dataset("lung", backend="pandas")
    y = Surv.right(df["time"], event=(df["status"] == 2))
    return CoxPH(ties="breslow").fit(y, df[["age", "sex"]])


def test_plot_influence_returns_hconcat(cox_lung: CoxPH) -> None:
    chart = gw.plot_influence(cox_lung)
    assert isinstance(chart, alt.HConcatChart)
    chart.to_dict()


def test_plot_influence_default_three_panels(cox_lung: CoxPH) -> None:
    spec = gw.plot_influence(cox_lung).to_dict()
    assert len(spec["hconcat"]) == 3


def test_plot_influence_custom_panels(cox_lung: CoxPH) -> None:
    chart = gw.plot_influence(cox_lung, panels=["deviance", "leverage"])
    spec = chart.to_dict()
    assert len(spec["hconcat"]) == 2
    chart.to_dict()


def test_plot_influence_single_panel(cox_lung: CoxPH) -> None:
    chart = gw.plot_influence(cox_lung, panels=["ld"])
    spec = chart.to_dict()
    assert len(spec["hconcat"]) == 1


def test_plot_influence_highlight_zero(cox_lung: CoxPH) -> None:
    chart = gw.plot_influence(cox_lung, highlight=0)
    chart.to_dict()


def test_plot_influence_highlight_five(cox_lung: CoxPH) -> None:
    chart = gw.plot_influence(cox_lung, highlight=5)
    chart.to_dict()


def test_plot_influence_with_title(cox_lung: CoxPH) -> None:
    chart = gw.plot_influence(cox_lung, title="Lung model diagnostics")
    spec = chart.to_dict()
    assert spec.get("title") == "Lung model diagnostics"


def test_plot_influence_invalid_backend(cox_lung: CoxPH) -> None:
    with pytest.raises(ValueError, match="backend"):
        gw.plot_influence(cox_lung, backend="matplotlib")  # type: ignore[arg-type]


def test_plot_influence_plotnine_backend(cox_lung: CoxPH) -> None:
    p9 = pytest.importorskip("plotnine")
    p = gw.plot_influence(cox_lung, backend="plotnine")
    assert isinstance(p, p9.ggplot)


def test_plot_influence_martingale_panel(cox_lung: CoxPH) -> None:
    chart = gw.plot_influence(cox_lung, panels=["martingale"])
    chart.to_dict()
