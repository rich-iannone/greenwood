"""Tests for `plot_predicted_survival` (per-subject predicted survival/cumulative-hazard curves)."""

from __future__ import annotations

import pytest

alt = pytest.importorskip("altair")

import greenwood as gw  # noqa: E402
from greenwood import RandomSurvivalForest, Surv  # noqa: E402


@pytest.fixture(scope="module")
def forest_data():
    lung = gw.load_dataset("lung", backend="pandas").dropna(
        subset=["ph.ecog", "ph.karno", "wt.loss"]
    )
    y = Surv.right(lung["time"], event=(lung["status"] == 2))
    cols = ["age", "sex", "ph.ecog", "ph.karno", "wt.loss"]
    rsf = RandomSurvivalForest(n_estimators=40, random_state=0).fit(y, lung[cols])
    return rsf, lung[cols]


def test_returns_valid_chart(forest_data) -> None:
    rsf, x = forest_data
    chart = gw.plot_predicted_survival(rsf, x[:4])

    assert isinstance(chart, alt.Chart)

    chart.to_dict()  # validates the Vega-Lite spec


def test_exported_at_top_level_and_viz() -> None:
    assert gw.plot_predicted_survival is gw.viz.plot_predicted_survival


def test_cumulative_hazard_type(forest_data) -> None:
    rsf, x = forest_data
    chart = gw.plot_predicted_survival(rsf, x[:3], type="cumulative_hazard")
    spec = chart.to_dict()

    assert spec is not None


def test_custom_labels_and_times(forest_data) -> None:
    rsf, x = forest_data
    chart = gw.plot_predicted_survival(
        rsf, x[:2], labels=["patient A", "patient B"], times=[100, 300, 600]
    )
    chart.to_dict()


def test_works_for_tree_and_cox(forest_data) -> None:
    _, x = forest_data
    lung = gw.load_dataset("lung", backend="pandas").dropna(
        subset=["ph.ecog", "ph.karno", "wt.loss"]
    )
    y = Surv.right(lung["time"], event=(lung["status"] == 2))
    tree = gw.SurvivalTree(max_depth=3, random_state=0).fit(y, x)
    cox = gw.CoxPH().fit(y, x)
    gw.plot_predicted_survival(tree, x[:2]).to_dict()
    gw.plot_predicted_survival(cox, x[:2]).to_dict()


def test_bad_type_raises(forest_data) -> None:
    rsf, x = forest_data
    with pytest.raises(ValueError, match="type must be"):
        gw.plot_predicted_survival(rsf, x[:2], type="lp")


def test_bad_backend_raises(forest_data) -> None:
    rsf, x = forest_data
    with pytest.raises(ValueError, match="backend"):
        gw.plot_predicted_survival(rsf, x[:2], backend="plotnine")


def test_label_count_mismatch_raises(forest_data) -> None:
    rsf, x = forest_data
    with pytest.raises(ValueError, match="labels"):
        gw.plot_predicted_survival(rsf, x[:3], labels=["only one"])
