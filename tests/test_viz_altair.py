"""Tests for the Altair visualization backend.

These need altair (the `altair` extra); the module is skipped when it is absent. No rendering
engine is required as building the chart and serializing its Vega-Lite spec (`to_dict`)
validates it without pulling in Pandas or Matplotlib.
"""

from __future__ import annotations

import pytest

alt = pytest.importorskip("altair")

import greenwood as gw  # noqa: E402
from greenwood import Surv  # noqa: E402


@pytest.fixture
def km_grouped() -> gw.KaplanMeier:
    df = gw.load_dataset("lung", backend="polars")
    y = Surv.right(df["time"], event=(df["status"] == 2))
    return gw.KaplanMeier(conf_type="log-log").fit(y, by=df["sex"])


@pytest.fixture
def km_overall() -> gw.KaplanMeier:
    df = gw.load_dataset("lung", backend="polars")
    y = Surv.right(df["time"], event=(df["status"] == 2))
    return gw.KaplanMeier().fit(y)


def test_plot_survival_returns_layer_chart(km_overall: gw.KaplanMeier) -> None:
    chart = gw.viz.altair.plot_survival(km_overall)
    assert isinstance(chart, alt.LayerChart)
    chart.to_dict()  # validates the Vega-Lite spec


def test_altair_is_the_default_backend(km_overall: gw.KaplanMeier) -> None:
    # The top-level and `gw.viz` entry points default to Altair, not plotnine.
    assert gw.plot_survival is gw.viz.plot_survival is gw.viz.altair.plot_survival
    assert isinstance(gw.plot_survival(km_overall), alt.LayerChart)


def test_plot_survival_grouped_serializes(km_grouped: gw.KaplanMeier) -> None:
    gw.viz.altair.plot_survival(km_grouped).to_dict()


def test_risk_table_returns_vconcat(km_grouped: gw.KaplanMeier) -> None:
    chart = gw.viz.altair.plot_survival(km_grouped, risk_table=True)
    assert isinstance(chart, alt.VConcatChart)
    chart.to_dict()


def test_risk_table_shares_one_zoom_param(km_grouped: gw.KaplanMeier) -> None:
    # Curve and table must pan/zoom together: a single top-level x-zoom param, so both
    # views read the one shared x scale it rewrites (no desync between curve and table).
    spec = gw.viz.altair.plot_survival(km_grouped, risk_table=True).to_dict()
    params = spec.get("params", [])
    assert len(params) == 1
    assert params[0]["bind"] == "scales"
    assert params[0]["select"]["encodings"] == ["x"]


def test_risk_table_drops_redundant_curve_axis(km_grouped: gw.KaplanMeier) -> None:
    # With a table below, the curve's x-axis title is dropped (the table carries Time).
    spec = gw.viz.altair.plot_survival(km_grouped, risk_table=True).to_dict()
    curve = spec["vconcat"][0]
    x_enc = curve["layer"][0]["encoding"]["x"]
    assert x_enc["axis"]["title"] is None


def test_layers_present(km_overall: gw.KaplanMeier) -> None:
    # A step area (CI), a step line (survival), and censor points.
    chart = gw.viz.altair.plot_survival(km_overall, conf_int=True, censor_marks=True)
    marks = {layer.to_dict()["mark"]["type"] for layer in chart.layer}
    assert {"area", "line", "point"} <= marks


def test_censor_marks_are_angled_notches(km_overall: gw.KaplanMeier) -> None:
    # Censoring marks render as `/` notches: a custom symbol path anchored at (0,0) so the
    # bottom tip rests on the curve, slanting up and to the right (negative SVG y).
    from greenwood.viz._altair import _CENSOR_NOTCH

    chart = gw.viz.altair.plot_survival(km_overall, conf_int=False)
    point = next(layer for layer in chart.layer if layer.to_dict()["mark"]["type"] == "point")
    assert point.to_dict()["mark"]["shape"] == _CENSOR_NOTCH
    assert _CENSOR_NOTCH == "M 0,0 L 1,-1"


def test_censor_color_is_darker_than_line(km_overall: gw.KaplanMeier) -> None:
    # The notch color is a darker tint of the (solid) curve color.
    from greenwood.viz._altair import _SOLID, _darken

    chart = gw.viz.altair.plot_survival(km_overall, conf_int=False)
    point = next(layer for layer in chart.layer if layer.to_dict()["mark"]["type"] == "point")
    assert point.to_dict()["encoding"]["stroke"]["value"] == _darken(_SOLID)


def test_no_ci_drops_area(km_overall: gw.KaplanMeier) -> None:
    chart = gw.viz.altair.plot_survival(km_overall, conf_int=False)
    marks = {layer.to_dict()["mark"]["type"] for layer in chart.layer}
    assert "area" not in marks


def test_step_interpolation(km_overall: gw.KaplanMeier) -> None:
    # The line must use step-after interpolation to draw a right-continuous KM step.
    chart = gw.viz.altair.plot_survival(km_overall, conf_int=False, censor_marks=False)
    line = chart.layer[0].to_dict()["mark"]
    assert line["interpolate"] == "step-after"


def test_standalone_risk_table(km_grouped: gw.KaplanMeier) -> None:
    gt = pytest.importorskip("great_tables")

    table = gw.risk_table(km_grouped, times=[0, 250, 500])
    assert isinstance(table, gt.GT)


# ---------------------------------------------------------------------------
# risk_table() GT content tests
# ---------------------------------------------------------------------------


class TestRiskTableContent:
    @pytest.fixture(autouse=True)
    def _require_gt(self) -> None:
        pytest.importorskip("great_tables")

    def test_grouped_row_count(self, km_grouped: gw.KaplanMeier) -> None:
        table = gw.risk_table(km_grouped, times=[0, 250, 500])
        assert table._tbl_data.shape[0] == 2

    def test_unstratified_row_count(self, km_overall: gw.KaplanMeier) -> None:
        table = gw.risk_table(km_overall, times=[0, 250, 500])
        assert table._tbl_data.shape[0] == 1

    def test_columns_match_times(self, km_grouped: gw.KaplanMeier) -> None:
        table = gw.risk_table(km_grouped, times=[0, 250, 500])
        assert table._tbl_data.columns == ["strata", "0.0", "250.0", "500.0"]

    def test_custom_times_columns(self, km_overall: gw.KaplanMeier) -> None:
        table = gw.risk_table(km_overall, times=[0, 100, 200, 400])
        expected = ["strata", "0.0", "100.0", "200.0", "400.0"]
        assert table._tbl_data.columns == expected

    def test_default_times_six_columns(self, km_overall: gw.KaplanMeier) -> None:
        table = gw.risk_table(km_overall)
        n_time_cols = len([c for c in table._tbl_data.columns if c != "strata"])
        assert n_time_cols == 6

    def test_header_title(self, km_grouped: gw.KaplanMeier) -> None:
        table = gw.risk_table(km_grouped, times=[0, 250, 500])
        assert table._heading.title == "Numbers at Risk"

    def test_header_subtitle(self, km_grouped: gw.KaplanMeier) -> None:
        table = gw.risk_table(km_grouped, times=[0, 250, 500])
        assert table._heading.subtitle == "Count of subjects at risk at each time point"

    def test_unstratified_strata_label(self, km_overall: gw.KaplanMeier) -> None:
        table = gw.risk_table(km_overall, times=[0, 250, 500])
        assert table._tbl_data["strata"][0] == "Overall"

    def test_grouped_strata_labels(self, km_grouped: gw.KaplanMeier) -> None:
        table = gw.risk_table(km_grouped, times=[0, 250, 500])
        assert set(table._tbl_data["strata"].to_list()) == {"1", "2"}

    def test_risk_values_match_frame(self, km_grouped: gw.KaplanMeier) -> None:
        import polars as pl

        table = gw.risk_table(km_grouped, times=[0, 250, 500])
        data = table._tbl_data
        frame = gw.get_risk_table_frame(km_grouped, times=[0, 250, 500], format="polars")
        for row_idx in range(data.shape[0]):
            stratum = data["strata"][row_idx]
            for col in ["0.0", "250.0", "500.0"]:
                gt_val = data[col][row_idx]
                frame_val = frame.filter(
                    (pl.col("strata") == stratum) & (pl.col("time") == float(col))
                )["n_risk"][0]
                assert gt_val == frame_val

    def test_fmt_integer_applied(self, km_grouped: gw.KaplanMeier) -> None:
        html = gw.risk_table(km_grouped, times=[0, 250, 500]).as_raw_html()
        assert "138" in html
        assert "138.0" not in html
