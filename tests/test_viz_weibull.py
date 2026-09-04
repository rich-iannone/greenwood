"""Tests for the Weibull diagnostic plot."""

from __future__ import annotations

import numpy as np
import pytest

alt = pytest.importorskip("altair")

import greenwood as gw  # noqa: E402
from greenwood import AFT, KaplanMeier, Surv  # noqa: E402
from greenwood.viz._weibull import _transform_y, _weibull_columns  # noqa: E402


@pytest.fixture
def lung_surv() -> tuple[Surv, gw.data.DataFrameT]:
    df = gw.load_dataset("lung", backend="pandas")
    y = Surv.right(df["time"], event=(df["status"] == 2))
    return y, df


@pytest.fixture
def km_overall(lung_surv: tuple[Surv, gw.data.DataFrameT]) -> KaplanMeier:
    y, _ = lung_surv
    return KaplanMeier().fit(y)


@pytest.fixture
def km_grouped(lung_surv: tuple[Surv, gw.data.DataFrameT]) -> KaplanMeier:
    y, df = lung_surv
    return KaplanMeier().fit(y, by=df["sex"])


@pytest.fixture
def aft_weibull(lung_surv: tuple[Surv, gw.data.DataFrameT]) -> AFT:
    y, df = lung_surv
    return AFT("weibull").fit(y, df[["age", "sex"]])


def test_plot_weibull_returns_chart(km_overall: KaplanMeier) -> None:
    chart = gw.plot_weibull(km_overall)

    assert isinstance(chart, (alt.LayerChart, alt.Chart))

    chart.to_dict()


def test_plot_weibull_grouped(km_grouped: KaplanMeier) -> None:
    chart = gw.plot_weibull(km_grouped)
    spec = chart.to_dict()

    assert "layer" in spec

    chart.to_dict()


def test_plot_weibull_with_aft_overlay(km_overall: KaplanMeier, aft_weibull: AFT) -> None:
    chart = gw.plot_weibull(km_overall, aft=aft_weibull)
    spec = chart.to_dict()

    assert len(spec["layer"]) == 3


def test_plot_weibull_aft_stratified_raises(km_grouped: KaplanMeier, aft_weibull: AFT) -> None:
    with pytest.raises(ValueError, match="unstratified"):
        gw.plot_weibull(km_grouped, aft=aft_weibull)


def test_plot_weibull_dist_lognormal(km_overall: KaplanMeier) -> None:
    chart = gw.plot_weibull(km_overall, dist="lognormal")
    chart.to_dict()


def test_plot_weibull_dist_loglogistic(km_overall: KaplanMeier) -> None:
    chart = gw.plot_weibull(km_overall, dist="loglogistic")
    chart.to_dict()


def test_plot_weibull_invalid_dist_raises(km_overall: KaplanMeier) -> None:
    with pytest.raises(ValueError, match="dist must be one of"):
        gw.plot_weibull(km_overall, dist="gamma")


def test_plot_weibull_dist_mismatch_raises(km_overall: KaplanMeier, aft_weibull: AFT) -> None:
    with pytest.raises(ValueError, match="does not match"):
        gw.plot_weibull(km_overall, aft=aft_weibull, dist="lognormal")


def test_plot_weibull_exponential_aft_accepted(
    lung_surv: tuple[Surv, gw.data.DataFrameT],
    km_overall: KaplanMeier,
) -> None:
    y, df = lung_surv
    aft_exp = AFT("exponential").fit(y, df[["age", "sex"]])
    chart = gw.plot_weibull(km_overall, aft=aft_exp, dist="weibull")
    chart.to_dict()


def test_plot_weibull_with_title(km_overall: KaplanMeier) -> None:
    chart = gw.plot_weibull(km_overall, title="Weibull check")
    spec = chart.to_dict()

    assert spec.get("title") == "Weibull check"


def test_plot_weibull_custom_labels(km_overall: KaplanMeier) -> None:
    chart = gw.plot_weibull(km_overall, xlab="log(T)", ylab="cloglog")
    spec = chart.to_dict()
    x_titles = [
        layer.get("encoding", {}).get("x", {}).get("title") for layer in spec.get("layer", [])
    ]
    y_titles = [
        layer.get("encoding", {}).get("y", {}).get("title") for layer in spec.get("layer", [])
    ]

    assert "log(T)" in x_titles
    assert "cloglog" in y_titles


def test_weibull_columns_empty_block() -> None:
    """A single-event KM where S jumps from 1 to 0 produces no plottable points."""
    y = Surv.right(np.array([5.0]), event=np.array([True]))
    km = KaplanMeier().fit(y)
    cols = _weibull_columns(km, "weibull")

    assert cols["log_time"] == []
    assert cols["y"] == []


def test_weibull_columns_filters_boundary() -> None:
    y = Surv.right(
        np.array([1.0, 2.0, 3.0, 4.0]),
        event=np.array([True, True, True, True]),
    )
    km = KaplanMeier().fit(y)
    cols = _weibull_columns(km, "weibull")
    surv_at_events = km._blocks[0].surv
    boundary_count = int(np.sum(surv_at_events == 0.0)) + int(np.sum(surv_at_events == 1.0))

    assert len(cols["log_time"]) == len(surv_at_events) - boundary_count


def test_transform_y_weibull() -> None:
    s = np.array([0.9, 0.5, 0.1])
    y = _transform_y(s, "weibull")
    expected = np.log(-np.log(s))

    np.testing.assert_allclose(y, expected)


def test_transform_y_lognormal() -> None:
    from scipy.stats import norm

    s = np.array([0.9, 0.5, 0.1])
    y = _transform_y(s, "lognormal")
    expected = norm.ppf(1.0 - s)

    np.testing.assert_allclose(y, expected)


def test_transform_y_loglogistic() -> None:
    s = np.array([0.9, 0.5, 0.1])
    y = _transform_y(s, "loglogistic")
    expected = np.log((1.0 - s) / s)

    np.testing.assert_allclose(y, expected)


def test_weibull_transform_linearity() -> None:
    """Data drawn from a Weibull distribution should be approximately linear on cloglog axes."""
    rng = np.random.default_rng(23)
    shape, scale = 1.5, 100.0
    times = rng.weibull(shape, size=500) * scale
    event = np.ones(500, dtype=bool)
    y = Surv.right(times, event=event)
    km = KaplanMeier().fit(y)
    cols = _weibull_columns(km, "weibull")
    x = np.array(cols["log_time"])
    y_vals = np.array(cols["y"])
    coeffs = np.polyfit(x, y_vals, 1)
    fitted = np.polyval(coeffs, x)
    ss_res = np.sum((y_vals - fitted) ** 2)
    ss_tot = np.sum((y_vals - np.mean(y_vals)) ** 2)
    r_squared = 1.0 - ss_res / ss_tot

    assert r_squared > 0.99
