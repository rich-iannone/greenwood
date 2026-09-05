"""Tests for the forest plot visualization."""

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


def test_returns_altair_chart(cox_lung: CoxPH) -> None:
    chart = gw.plot_forest(cox_lung)
    assert isinstance(chart, alt.LayerChart | alt.Chart | alt.VConcatChart | alt.HConcatChart)
    chart.to_dict()


def test_default_exponentiate(cox_lung: CoxPH) -> None:
    spec = gw.plot_forest(cox_lung).to_dict()
    assert spec is not None


def test_exponentiate_false(cox_lung: CoxPH) -> None:
    chart = gw.plot_forest(cox_lung, exponentiate=False)
    chart.to_dict()


def test_scale_log(cox_lung: CoxPH) -> None:
    chart = gw.plot_forest(cox_lung, scale="log")
    chart.to_dict()


def test_scale_linear(cox_lung: CoxPH) -> None:
    chart = gw.plot_forest(cox_lung, scale="linear")
    chart.to_dict()


def test_with_title(cox_lung: CoxPH) -> None:
    chart = gw.plot_forest(cox_lung, title="My Forest Plot")
    spec = chart.to_dict()
    assert spec.get("title") == "My Forest Plot" or any(
        "My Forest Plot" in str(v) for v in spec.values()
    )


def test_term_labels(cox_lung: CoxPH) -> None:
    chart = gw.plot_forest(cox_lung, term_labels={"age": "Age (years)", "sex": "Sex"})
    chart.to_dict()


def test_custom_xlab(cox_lung: CoxPH) -> None:
    chart = gw.plot_forest(cox_lung, xlab="Hazard Ratio (95% CI)")
    chart.to_dict()


def test_custom_dimensions(cox_lung: CoxPH) -> None:
    chart = gw.plot_forest(cox_lung, width=400, height=200)
    chart.to_dict()


def test_from_dataframe(cox_lung: CoxPH) -> None:
    import pandas as pd

    df = pd.DataFrame(
        {
            "term": ["age", "sex"],
            "estimate": [1.02, 0.59],
            "conf_low": [0.99, 0.43],
            "conf_high": [1.05, 0.82],
        }
    )
    chart = gw.plot_forest(df, scale="log")
    chart.to_dict()


def test_from_dict() -> None:
    data = {
        "term": ["A", "B"],
        "estimate": [1.5, 0.8],
        "conf_low": [1.1, 0.5],
        "conf_high": [2.0, 1.2],
    }
    chart = gw.plot_forest(data, scale="linear")
    chart.to_dict()


def test_from_polars_dataframe() -> None:
    import polars as pl

    df = pl.DataFrame(
        {
            "term": ["age", "sex"],
            "estimate": [1.02, 0.59],
            "conf_low": [0.99, 0.43],
            "conf_high": [1.05, 0.82],
        }
    )
    chart = gw.plot_forest(df, scale="log")
    chart.to_dict()


def test_invalid_backend(cox_lung: CoxPH) -> None:
    with pytest.raises(ValueError, match="backend"):
        gw.plot_forest(cox_lung, backend="matplotlib")  # type: ignore[arg-type]


def test_invalid_input() -> None:
    with pytest.raises(TypeError):
        gw.plot_forest("not a model")  # type: ignore[arg-type]


def test_plotnine_backend(cox_lung: CoxPH) -> None:
    p9 = pytest.importorskip("plotnine")
    p = gw.plot_forest(cox_lung, backend="plotnine")
    assert isinstance(p, p9.ggplot)


def test_plotnine_with_term_labels(cox_lung: CoxPH) -> None:
    p9 = pytest.importorskip("plotnine")
    p = gw.plot_forest(cox_lung, backend="plotnine", term_labels={"age": "Age", "sex": "Sex"})
    assert isinstance(p, p9.ggplot)
