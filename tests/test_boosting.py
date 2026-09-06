"""Tests for `GradientBoostingSurvivalAnalysis` (Cox-loss gradient boosting).

Correctness is checked by structural and statistical properties: predicted survival curves are valid
step functions, `type="lp"` and `type="risk"` are consistent, fits are reproducible under a fixed
seed, discrimination beats chance, and adding trees drives the training concordance up (the model is
learning).
"""

from __future__ import annotations

import numpy as np
import pytest

import greenwood as gw
from greenwood import GradientBoostingSurvivalAnalysis as GBSA
from greenwood import Surv


@pytest.fixture(scope="module")
def data():
    lung = gw.load_dataset("lung", backend="pandas").dropna(
        subset=["ph.ecog", "ph.karno", "wt.loss"]
    )
    y = Surv.right(lung["time"], event=(lung["status"] == 2))
    cols = ["age", "sex", "ph.ecog", "ph.karno", "wt.loss"]
    return y, lung[cols]


def _curve_matrix(frame) -> np.ndarray:
    cols = [c for c in frame.columns if c != "time"]
    return frame[cols].to_numpy()


def test_fit_reports_structure(data) -> None:
    y, x = data
    gbm = GBSA(n_estimators=50, random_state=0).fit(y, x)
    assert gbm.n_ == x.shape[0]
    assert gbm.n_event_ == int(y.n_events)
    assert gbm.n_features_in_ == x.shape[1]
    assert gbm.feature_names_in_ == list(x.columns)
    assert len(gbm.trees_) == 50
    assert "GradientBoostingSurvivalAnalysis" in repr(gbm)


def test_lp_and_risk_are_consistent(data) -> None:
    y, x = data
    gbm = GBSA(n_estimators=80, learning_rate=0.05, max_depth=2, random_state=0).fit(y, x)
    lp = gbm.predict(x, type="lp")
    risk = gbm.predict(x, type="risk")
    assert np.allclose(risk, np.exp(lp))


def test_survival_curves_are_valid(data) -> None:
    y, x = data
    gbm = GBSA(n_estimators=100, learning_rate=0.05, max_depth=2, random_state=0).fit(y, x)
    surv = _curve_matrix(gbm.predict(x[:8], type="survival", format="pandas"))
    chf = _curve_matrix(gbm.predict(x[:8], type="cumulative_hazard", format="pandas"))
    assert np.all(surv >= 0.0) and np.all(surv <= 1.0 + 1e-12)
    assert np.all(np.diff(surv, axis=0) <= 1e-12)
    assert np.all(np.diff(chf, axis=0) >= -1e-12)


def test_reproducible_under_seed(data) -> None:
    y, x = data
    r1 = GBSA(n_estimators=60, max_depth=2, random_state=23).fit(y, x).predict(x)
    r2 = GBSA(n_estimators=60, max_depth=2, random_state=23).fit(y, x).predict(x)
    assert np.allclose(r1, r2)


def test_beats_chance(data) -> None:
    y, x = data
    gbm = GBSA(n_estimators=200, learning_rate=0.05, max_depth=2, random_state=0).fit(y, x)
    assert gw.concordance_index(y, gbm.predict(x)) > 0.6


def test_more_trees_improve_training_fit(data) -> None:
    y, x = data
    c_few = gw.concordance_index(
        y, GBSA(n_estimators=5, learning_rate=0.1, max_depth=2, random_state=0).fit(y, x).predict(x)
    )
    c_many = gw.concordance_index(
        y,
        GBSA(n_estimators=200, learning_rate=0.1, max_depth=2, random_state=0).fit(y, x).predict(x),
    )
    assert c_many > c_few


def test_subsample_runs(data) -> None:
    y, x = data
    gbm = GBSA(n_estimators=60, subsample=0.7, max_depth=2, random_state=1).fit(y, x)
    assert gbm.predict(x).shape == (x.shape[0],)


def test_predict_custom_times(data) -> None:
    y, x = data
    gbm = GBSA(n_estimators=40, max_depth=2, random_state=0).fit(y, x)
    frame = gbm.predict(x[:3], type="survival", times=[100, 300, 600], format="pandas")
    assert list(frame["time"]) == [100.0, 300.0, 600.0]


def test_rejects_non_right_censored() -> None:
    y = Surv.counting(start=[0, 1, 2], stop=[5, 6, 7], event=[1, 0, 1])
    x = np.array([[1.0], [2.0], [3.0]])
    with pytest.raises(NotImplementedError, match="right-censored"):
        GBSA().fit(y, x)


def test_predict_rejects_unknown_type(data) -> None:
    y, x = data
    gbm = GBSA(n_estimators=10, random_state=0).fit(y, x)
    with pytest.raises(ValueError, match="Unknown predict type"):
        gbm.predict(x, type="bogus")


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"n_estimators": 0}, "n_estimators"),
        ({"learning_rate": 0.0}, "learning_rate"),
        ({"max_depth": 0}, "max_depth"),
        ({"min_samples_leaf": 0}, "min_samples_leaf"),
        ({"subsample": 0.0}, "subsample"),
        ({"subsample": 1.5}, "subsample"),
    ],
)
def test_invalid_params_raise(kwargs, match) -> None:
    with pytest.raises(ValueError, match=match):
        GBSA(**kwargs)


def test_unfitted_repr() -> None:
    assert "unfitted" in repr(GBSA())


@pytest.mark.parametrize("max_features", [None, "sqrt", "log2", 2, 0.5])
def test_max_features_variants(data, max_features) -> None:
    y, x = data
    gbm = GBSA(n_estimators=10, max_features=max_features, random_state=0).fit(y, x)
    assert gbm.predict(x).shape == (x.shape[0],)


def test_invalid_max_features_string_raises(data) -> None:
    y, x = data
    with pytest.raises(ValueError, match="max_features"):
        GBSA(n_estimators=5, max_features="bogus", random_state=0).fit(y, x)


def test_invalid_max_features_type_raises(data) -> None:
    y, x = data
    with pytest.raises(TypeError, match="max_features"):
        GBSA(n_estimators=5, max_features=[1, 2], random_state=0).fit(y, x)


def test_mismatched_rows_raises(data) -> None:
    y, x = data
    with pytest.raises(ValueError, match="same number of rows"):
        GBSA(n_estimators=5, random_state=0).fit(y, x[:-5])


def test_no_events_raises() -> None:
    y = Surv.right(time=[5, 6, 7, 8], event=[0, 0, 0, 0])
    x = np.array([[1.0], [2.0], [3.0], [4.0]])
    with pytest.raises(ValueError, match="No events remain"):
        GBSA(n_estimators=5).fit(y, x)


def test_summaries(data) -> None:
    y, x = data
    gbm = GBSA(n_estimators=40, max_depth=2, random_state=0).fit(y, x)
    g = gw.glance(gbm, format="pandas")
    assert g.loc[0, "n"] == x.shape[0]
    assert g.loc[0, "n_estimators"] == 40
    t = gw.tidy(gbm, format="pandas")
    assert set(t["term"]) == set(x.columns)
    assert abs(float(t["importance"].sum()) - 1.0) < 1e-9
    assert list(t["importance"]) == sorted(t["importance"], reverse=True)


# ---------------------------------------------------------------------------
# variable_importance() tests
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def gbm(data):  # type: ignore[no-untyped-def]
    y, x = data
    return GBSA(n_estimators=100, learning_rate=0.05, max_depth=2, random_state=0).fit(y, x)


class TestVariableImportance:
    def test_returns_dataframe_with_expected_columns(self, gbm) -> None:  # type: ignore[no-untyped-def]
        vi = gbm.variable_importance(format="pandas")
        assert list(vi.columns) == ["term", "importance"]

    def test_all_features_present(self, data, gbm) -> None:  # type: ignore[no-untyped-def]
        _, x = data
        vi = gbm.variable_importance(format="pandas")
        assert set(vi["term"]) == set(x.columns)

    def test_importance_sums_to_one(self, gbm) -> None:  # type: ignore[no-untyped-def]
        vi = gbm.variable_importance(format="pandas")
        assert abs(float(vi["importance"].sum()) - 1.0) < 1e-9

    def test_importance_non_negative(self, gbm) -> None:  # type: ignore[no-untyped-def]
        vi = gbm.variable_importance(format="pandas")
        assert (vi["importance"] >= 0).all()

    def test_sorted_descending(self, gbm) -> None:  # type: ignore[no-untyped-def]
        vi = gbm.variable_importance(format="pandas")
        vals = vi["importance"].tolist()
        assert vals == sorted(vals, reverse=True)

    def test_n_rows_equals_n_features(self, gbm) -> None:  # type: ignore[no-untyped-def]
        vi = gbm.variable_importance(format="pandas")
        assert vi.shape[0] == gbm.n_features_in_

    def test_matches_feature_importances_attr(self, gbm) -> None:  # type: ignore[no-untyped-def]
        vi = gbm.variable_importance(format="pandas")
        order = np.argsort(gbm.feature_importances_)[::-1]
        np.testing.assert_allclose(
            vi["importance"].values, gbm.feature_importances_[order]
        )

    def test_format_polars(self, gbm) -> None:  # type: ignore[no-untyped-def]
        import polars as pl

        vi = gbm.variable_importance(format="polars")
        assert isinstance(vi, pl.DataFrame)
        assert vi.columns == ["term", "importance"]

    def test_format_default(self, gbm) -> None:  # type: ignore[no-untyped-def]
        vi = gbm.variable_importance()
        assert vi.shape[0] == gbm.n_features_in_

    def test_matches_tidy(self, gbm) -> None:  # type: ignore[no-untyped-def]
        vi = gbm.variable_importance(format="pandas")
        t = gw.tidy(gbm, format="pandas")
        np.testing.assert_allclose(vi["importance"].values, t["importance"].values)
        assert list(vi["term"]) == list(t["term"])

    def test_single_feature(self) -> None:
        lung = gw.load_dataset("lung", backend="pandas").dropna(subset=["ph.ecog"])
        y = Surv.right(lung["time"], event=(lung["status"] == 2))
        gbm1 = GBSA(n_estimators=20, max_depth=1, random_state=0).fit(y, lung[["ph.ecog"]])
        vi = gbm1.variable_importance(format="pandas")
        assert vi.shape == (1, 2)
        assert vi["term"].iloc[0] == "ph.ecog"
        assert abs(float(vi["importance"].iloc[0]) - 1.0) < 1e-9
