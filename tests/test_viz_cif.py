"""Tests for `plot_cif()` visualizations."""

from __future__ import annotations

import numpy as np
import pytest

import greenwood as gw
from greenwood import AalenJohansen, Surv


def _mgus_fit(*, by_sex: bool = False) -> AalenJohansen:
    """Aalen-Johansen fit on the bundled mgus2 dataset."""
    mg = gw.load_dataset("mgus2", backend="polars")
    etime = np.where(mg["pstat"] == 1, mg["ptime"], mg["futime"])
    cause = np.where(mg["pstat"] == 1, 1, 2 * mg["death"])
    y = Surv.multistate(etime, event=cause, states=("pcm", "death"))
    if by_sex:
        return AalenJohansen().fit(y, by=mg["sex"])
    return AalenJohansen().fit(y)


def test_plot_cif_returns_chart() -> None:
    """Unstratified fit produces a faceted Altair chart."""
    import altair as alt

    chart = gw.plot_cif(_mgus_fit())
    assert isinstance(chart, alt.FacetChart)


def test_plot_cif_grouped_returns_chart() -> None:
    """Stratified fit produces a faceted Altair chart."""
    import altair as alt

    chart = gw.plot_cif(_mgus_fit(by_sex=True))
    assert isinstance(chart, alt.FacetChart)


def test_plot_cif_no_conf_int() -> None:
    """conf_int=False still returns a valid chart."""
    import altair as alt

    chart = gw.plot_cif(_mgus_fit(), conf_int=False)
    assert isinstance(chart, alt.FacetChart)


def test_plot_cif_with_title() -> None:
    """title= is passed through to the chart properties."""
    chart = gw.plot_cif(_mgus_fit(by_sex=True), title="By sex")
    assert chart.title == "By sex"


def test_plot_cif_risk_table_returns_vconcat() -> None:
    """risk_table=True wraps the chart in a VConcatChart."""
    import altair as alt

    chart = gw.plot_cif(_mgus_fit(), risk_table=True)
    assert isinstance(chart, alt.VConcatChart)


def test_plot_cif_risk_table_grouped() -> None:
    """risk_table=True works with a stratified fit."""
    import altair as alt

    chart = gw.plot_cif(_mgus_fit(by_sex=True), risk_table=True)
    assert isinstance(chart, alt.VConcatChart)


def test_plot_cif_risk_table_custom_times() -> None:
    """Custom times= are passed to the risk table without error."""
    import altair as alt

    chart = gw.plot_cif(_mgus_fit(), risk_table=True, times=[0, 60, 120, 240])
    assert isinstance(chart, alt.VConcatChart)


def test_plot_cif_invalid_backend() -> None:
    """An unknown backend raises ValueError."""
    with pytest.raises(ValueError, match="backend"):
        gw.plot_cif(_mgus_fit(), backend="plotnine")
