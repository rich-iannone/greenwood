"""Tests for the plotnine visualization layer.

These need plotnine (the `viz` extra); the module is skipped when it is absent. Rendering
uses a non-interactive backend, and figures are closed after each test.
"""

from __future__ import annotations

import pytest

p9 = pytest.importorskip("plotnine")

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import greenwood as gw  # noqa: E402
from greenwood import Surv  # noqa: E402


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


@pytest.fixture
def km_grouped() -> gw.KaplanMeier:
    df = gw.data.load_dataset("lung")
    y = Surv.right(df["time"], event=(df["status"] == 2))
    return gw.KaplanMeier(conf_type="log-log").fit(y, by=df["sex"])


@pytest.fixture
def km_overall() -> gw.KaplanMeier:
    df = gw.data.load_dataset("lung")
    y = Surv.right(df["time"], event=(df["status"] == 2))
    return gw.KaplanMeier().fit(y)


def test_plot_survival_returns_ggplot(km_overall: gw.KaplanMeier) -> None:
    plot = gw.viz.plot_survival(km_overall)
    assert isinstance(plot, p9.ggplot)


def test_plot_survival_draws_without_error(km_grouped: gw.KaplanMeier) -> None:
    gw.viz.plot_survival(km_grouped).draw(show=False)


def test_plot_survival_layers(km_overall: gw.KaplanMeier) -> None:
    # A ribbon (CI), a line (survival), and points (censor marks) are present.
    plot = gw.viz.plot_survival(km_overall, conf_int=True, censor_marks=True)
    geoms = {type(layer.geom).__name__ for layer in plot.layers}
    assert {"geom_ribbon", "geom_line", "geom_point"} <= geoms


def test_plot_survival_without_ci_has_no_ribbon(km_overall: gw.KaplanMeier) -> None:
    plot = gw.viz.plot_survival(km_overall, conf_int=False)
    geoms = {type(layer.geom).__name__ for layer in plot.layers}
    assert "geom_ribbon" not in geoms


def test_risk_table_composition_stacks(km_grouped: gw.KaplanMeier) -> None:
    from plotnine.composition import Compose

    comp = gw.viz.plot_survival(km_grouped, risk_table=True)
    assert isinstance(comp, Compose)
    comp.draw()


def test_risk_table_plot_returns_ggplot(km_grouped: gw.KaplanMeier) -> None:
    assert isinstance(gw.viz.risk_table(km_grouped), p9.ggplot)


def test_risk_table_data_shape(km_grouped: gw.KaplanMeier) -> None:
    rtd = gw.viz.risk_table_data(km_grouped, times=[0, 250, 500])
    assert list(rtd.columns) == ["strata", "time", "n_risk"]
    assert set(rtd["strata"]) == {"1", "2"}
    assert len(rtd) == 6
