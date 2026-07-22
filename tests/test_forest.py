"""Tests for forest plot visualization."""

from __future__ import annotations

import pandas as pd
import pytest

import greenwood as gw
from greenwood import CoxPH, Surv
from greenwood.viz._forest import (
    _extract_forest_frame,
    _fmt_pvalue,
    plot_forest,
    theme_forest,
)

# ---------------------------------------------------------------------------
# plot_forest(): helpers
# ---------------------------------------------------------------------------


def test_fmt_pvalue_tiny() -> None:
    assert _fmt_pvalue(0.0001) == "<0.001"


def test_fmt_pvalue_small() -> None:
    assert _fmt_pvalue(0.005) == "0.005"


def test_fmt_pvalue_moderate() -> None:
    assert _fmt_pvalue(0.04) == "0.04"


def test_fmt_pvalue_large() -> None:
    assert _fmt_pvalue(0.5) == "0.50"


# ---------------------------------------------------------------------------
# plot_forest(): _extract_forest_frame with CoxPH
# ---------------------------------------------------------------------------


@pytest.fixture
def lung_cox() -> CoxPH:
    lung = gw.load_dataset("lung", backend="pandas")
    y = Surv.right(lung["time"], event=(lung["status"] == 2))
    return CoxPH().fit(y, lung[["age", "sex"]])


@pytest.fixture
def three_term_cox() -> CoxPH:
    lung = gw.load_dataset("lung", backend="pandas")
    y = Surv.right(lung["time"], event=(lung["status"] == 2))
    return CoxPH().fit(y, lung[["age", "sex", "ph.ecog"]])


@pytest.fixture
def subgroup_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "term": ["Age < 60", "Age ≥ 60", "Male", "Female"],
            "estimate": [0.72, 0.91, 0.85, 0.68],
            "ci_lower": [0.51, 0.74, 0.68, 0.50],
            "ci_upper": [1.01, 1.12, 1.06, 0.92],
        }
    )


def test_extract_from_cox_exponentiate_true(lung_cox: CoxPH) -> None:
    df = _extract_forest_frame(lung_cox, exponentiate=True, term_labels=None)
    assert set(df.columns) >= {"term", "estimate", "ci_lower", "ci_upper", "p_value"}
    assert len(df) == 2
    assert (df["estimate"] > 0).all()
    assert (df["ci_lower"] <= df["estimate"]).all()
    assert (df["ci_upper"] >= df["estimate"]).all()


def test_extract_from_cox_exponentiate_false(lung_cox: CoxPH) -> None:
    df = _extract_forest_frame(lung_cox, exponentiate=False, term_labels=None)
    assert (df["ci_lower"] <= df["estimate"]).all()
    assert (df["ci_upper"] >= df["estimate"]).all()


def test_extract_from_cox_term_labels(lung_cox: CoxPH) -> None:
    df = _extract_forest_frame(
        lung_cox,
        exponentiate=True,
        term_labels={"age": "Age (years)", "sex": "Sex (F vs M)"},
    )
    assert "Age (years)" in df["term"].tolist()
    assert "Sex (F vs M)" in df["term"].tolist()


def test_extract_from_cox_partial_term_labels(lung_cox: CoxPH) -> None:
    df = _extract_forest_frame(lung_cox, exponentiate=True, term_labels={"age": "Age (years)"})
    terms = df["term"].tolist()
    assert "Age (years)" in terms
    assert "sex" in terms


def test_extract_from_cox_row_count(three_term_cox: CoxPH) -> None:
    df = _extract_forest_frame(three_term_cox, exponentiate=True, term_labels=None)
    assert len(df) == 3


def test_extract_from_dataframe(subgroup_df: pd.DataFrame) -> None:
    df = _extract_forest_frame(subgroup_df, exponentiate=True, term_labels=None)
    assert len(df) == 4
    assert set(df.columns) >= {"term", "estimate", "ci_lower", "ci_upper"}


def test_extract_from_dataframe_conf_low_alias() -> None:
    df_in = pd.DataFrame(
        {
            "term": ["A", "B"],
            "estimate": [0.8, 1.2],
            "conf_low": [0.6, 0.9],
            "conf_high": [1.0, 1.6],
        }
    )
    df = _extract_forest_frame(df_in, exponentiate=True, term_labels=None)
    assert "ci_lower" in df.columns
    assert "ci_upper" in df.columns


def test_extract_from_dataframe_missing_column_raises() -> None:
    bad_df = pd.DataFrame({"term": ["A"], "estimate": [0.8]})
    with pytest.raises(ValueError, match="missing required column"):
        _extract_forest_frame(bad_df, exponentiate=True, term_labels=None)


def test_extract_from_invalid_type_raises() -> None:
    with pytest.raises(TypeError, match="Expected a CoxPH result"):
        _extract_forest_frame([1, 2, 3], exponentiate=True, term_labels=None)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# plot_forest(): Altair backend (default)
# ---------------------------------------------------------------------------


def test_plot_forest_returns_altair_chart(lung_cox: CoxPH) -> None:
    pytest.importorskip("altair")
    chart = plot_forest(lung_cox)
    assert hasattr(chart, "to_dict")


def test_plot_forest_altair_explicit(lung_cox: CoxPH) -> None:
    pytest.importorskip("altair")
    chart = plot_forest(lung_cox, backend="altair")
    assert hasattr(chart, "to_dict")


def test_plot_forest_altair_serialises(lung_cox: CoxPH) -> None:
    pytest.importorskip("altair")
    chart_dict = plot_forest(lung_cox).to_dict()
    assert chart_dict is not None


def test_plot_forest_altair_title(lung_cox: CoxPH) -> None:
    pytest.importorskip("altair")
    chart = plot_forest(lung_cox, title="My Model")
    assert "My Model" in str(chart.to_dict())


def test_plot_forest_altair_term_labels(lung_cox: CoxPH) -> None:
    pytest.importorskip("altair")
    chart = plot_forest(lung_cox, term_labels={"age": "Age (years)"})
    assert "Age (years)" in str(chart.to_dict())


def test_plot_forest_altair_exponentiate_false(lung_cox: CoxPH) -> None:
    pytest.importorskip("altair")
    chart = plot_forest(lung_cox, exponentiate=False)
    assert hasattr(chart, "to_dict")


def test_plot_forest_altair_from_dataframe(subgroup_df: pd.DataFrame) -> None:
    pytest.importorskip("altair")
    chart = plot_forest(subgroup_df)
    assert hasattr(chart, "to_dict")


def test_plot_forest_altair_three_terms(three_term_cox: CoxPH) -> None:
    pytest.importorskip("altair")
    chart = plot_forest(three_term_cox)
    assert hasattr(chart, "to_dict")


# ---------------------------------------------------------------------------
# plot_forest(): plotnine backend
# ---------------------------------------------------------------------------


def test_plot_forest_plotnine_returns_ggplot(lung_cox: CoxPH) -> None:
    p9 = pytest.importorskip("plotnine")
    chart = plot_forest(lung_cox, backend="plotnine")
    assert isinstance(chart, p9.ggplot)


def test_plot_forest_plotnine_from_dataframe(subgroup_df: pd.DataFrame) -> None:
    p9 = pytest.importorskip("plotnine")
    chart = plot_forest(subgroup_df, backend="plotnine")
    assert isinstance(chart, p9.ggplot)


def test_plot_forest_plotnine_exponentiate_false(lung_cox: CoxPH) -> None:
    p9 = pytest.importorskip("plotnine")
    chart = plot_forest(lung_cox, backend="plotnine", exponentiate=False)
    assert isinstance(chart, p9.ggplot)


def test_plot_forest_plotnine_composable(lung_cox: CoxPH) -> None:
    """plot_forest + theme_forest() works via plotnine + operator."""
    p9 = pytest.importorskip("plotnine")
    chart = plot_forest(lung_cox, backend="plotnine") + theme_forest()
    assert isinstance(chart, p9.ggplot)


# ---------------------------------------------------------------------------
# plot_forest(): invalid backend
# ---------------------------------------------------------------------------


def test_plot_forest_invalid_backend_raises(lung_cox: CoxPH) -> None:
    with pytest.raises(ValueError, match="backend"):
        plot_forest(lung_cox, backend="matplotlib")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# theme_forest()
# ---------------------------------------------------------------------------


def test_theme_forest_returns_theme() -> None:
    p9 = pytest.importorskip("plotnine")
    assert isinstance(theme_forest(), p9.theme)


# ---------------------------------------------------------------------------
# Public API surface
# ---------------------------------------------------------------------------


def test_plot_forest_altair_scale_log_dataframe(subgroup_df: pd.DataFrame) -> None:
    """scale='log' on a DataFrame gives log-scale display."""
    pytest.importorskip("altair")
    chart = plot_forest(subgroup_df, scale="log")
    assert hasattr(chart, "to_dict")
    chart.to_dict()  # must not raise


def test_plot_forest_altair_scale_linear_cox(lung_cox: CoxPH) -> None:
    """scale='linear' overrides the CoxPH default of log."""
    pytest.importorskip("altair")
    chart = plot_forest(lung_cox, scale="linear")
    assert hasattr(chart, "to_dict")


def test_plot_forest_in_gw_namespace() -> None:
    assert hasattr(gw, "plot_forest")
    assert "plot_forest" in gw.__all__
    assert not hasattr(gw, "forest_plot")  # consolidated into plot_forest


def test_plot_forest_dict_input() -> None:
    data = {
        "term": ["age", "sex"],
        "estimate": [1.02, 0.58],
        "ci_lower": [1.00, 0.42],
        "ci_upper": [1.04, 0.81],
        "p_value": [0.01, 0.001],
    }
    chart = plot_forest(data)
    assert chart is not None


def test_theme_forest_in_gw_namespace() -> None:
    assert hasattr(gw, "theme_forest")
    assert "theme_forest" in gw.__all__
