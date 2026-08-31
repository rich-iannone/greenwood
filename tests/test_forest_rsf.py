"""Tests for the tree-based survival estimators (`SurvivalTree`, `RandomSurvivalForest`).

Correctness is checked here by structural and statistical properties: predicted survival curves are
valid step functions (monotone, bounded in `[0, 1]`), the risk score is consistent with the survival
curves, fits are reproducible under a fixed seed, the ensemble out-of-bag concordance beats chance,
and a forest sharpens a single tree.
"""

from __future__ import annotations

import numpy as np
import pytest

import greenwood as gw
import greenwood._forest as forest
from greenwood import ExtraSurvivalTrees, RandomSurvivalForest, Surv, SurvivalTree
from greenwood._forest import (
    _HAS_NUMBA,
    _best_logrank_split,
    _best_logrank_split_numba,
    _compact_grid,
    _logrank_kernel_impl,
    _random_logrank_split,
    _resolve_engine,
    _resolve_max_features,
    _RiskEventGrid,
)


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


def _small():
    time = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    event = np.array([True, True, False, True, True, False])
    col = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    return time, event, col


# -- SurvivalTree ---------------------------------------------------------------------------


def test_tree_fits_and_reports_structure(data) -> None:
    y, x = data
    tree = SurvivalTree(max_depth=3, random_state=0).fit(y, x)
    assert tree.n_ == x.shape[0]
    assert tree.n_event_ == int(y.n_events)
    assert tree.n_features_in_ == x.shape[1]
    assert tree.feature_names_in_ == list(x.columns)
    assert tree._n_leaves_ >= 2
    assert "SurvivalTree" in repr(tree)


def test_tree_risk_shape_and_finiteness(data) -> None:
    y, x = data
    tree = SurvivalTree(max_depth=4, random_state=0).fit(y, x)
    risk = tree.predict(x)
    assert risk.shape == (x.shape[0],)
    assert np.all(np.isfinite(risk))
    assert np.all(risk >= 0)


def test_tree_survival_is_valid_step_function(data) -> None:
    y, x = data
    tree = SurvivalTree(max_depth=4, random_state=0).fit(y, x)
    surv = _curve_matrix(tree.predict(x[:10], type="survival", format="pandas"))
    assert np.all(surv >= 0.0) and np.all(surv <= 1.0 + 1e-12)
    assert np.all(np.diff(surv, axis=0) <= 1e-12)  # non-increasing over time


def test_tree_predict_requires_newdata(data) -> None:
    y, x = data
    tree = SurvivalTree(random_state=0).fit(y, x)
    with pytest.raises(ValueError, match="newdata"):
        tree.predict()


def test_tree_rejects_non_right_censored() -> None:
    y = Surv.counting(start=[0, 1, 2], stop=[5, 6, 7], event=[1, 0, 1])
    x = np.array([[1.0], [2.0], [3.0]])
    with pytest.raises(NotImplementedError, match="right-censored"):
        SurvivalTree().fit(y, x)


# -- RandomSurvivalForest -------------------------------------------------------------------


def test_forest_curves_are_valid(data) -> None:
    y, x = data
    rsf = RandomSurvivalForest(n_estimators=30, random_state=0).fit(y, x)
    surv = _curve_matrix(rsf.predict(x[:8], type="survival", format="pandas"))
    chf = _curve_matrix(rsf.predict(x[:8], type="cumulative_hazard", format="pandas"))
    assert np.all(surv >= 0.0) and np.all(surv <= 1.0 + 1e-12)
    assert np.all(np.diff(surv, axis=0) <= 1e-12)  # survival non-increasing
    assert np.all(np.diff(chf, axis=0) >= -1e-12)  # cumulative hazard non-decreasing


def test_forest_risk_matches_summed_cumulative_hazard(data) -> None:
    y, x = data
    rsf = RandomSurvivalForest(n_estimators=25, random_state=1).fit(y, x)
    risk = rsf.predict(x)
    chf = _curve_matrix(rsf.predict(x, type="cumulative_hazard", format="pandas"))
    # The risk score is the summed ensemble cumulative hazard over the full event-time grid.
    assert np.allclose(risk, chf.sum(axis=0))


def test_forest_risk_is_reproducible_under_seed(data) -> None:
    y, x = data
    r1 = RandomSurvivalForest(n_estimators=30, random_state=23).fit(y, x).predict(x)
    r2 = RandomSurvivalForest(n_estimators=30, random_state=23).fit(y, x).predict(x)
    assert np.allclose(r1, r2)
    r3 = RandomSurvivalForest(n_estimators=30, random_state=7).fit(y, x).predict(x)
    assert not np.allclose(r1, r3)


def test_forest_beats_chance_out_of_bag(data) -> None:
    y, x = data
    rsf = RandomSurvivalForest(n_estimators=100, oob_score=True, random_state=0).fit(y, x)
    assert rsf.oob_score_ is not None
    assert rsf.oob_score_ > 0.55


def test_forest_sharpens_single_tree(data) -> None:
    y, x = data
    tree_risk = SurvivalTree(random_state=0).fit(y, x).predict(x)
    forest_risk = RandomSurvivalForest(n_estimators=100, random_state=0).fit(y, x).predict(x)
    tree_c = gw.concordance_index(y, tree_risk)
    forest_c = gw.concordance_index(y, forest_risk)
    assert forest_c >= tree_c


def test_forest_higher_risk_means_lower_survival(data) -> None:
    y, x = data
    rsf = RandomSurvivalForest(n_estimators=40, random_state=0).fit(y, x)
    risk = rsf.predict(x)
    late = _curve_matrix(rsf.predict(x, type="survival", times=[500], format="pandas")).ravel()
    assert np.corrcoef(risk, late)[0, 1] < 0.0


def test_forest_predict_at_custom_times(data) -> None:
    y, x = data
    rsf = RandomSurvivalForest(n_estimators=20, random_state=0).fit(y, x)
    frame = rsf.predict(x[:3], type="survival", times=[100, 300, 600], format="pandas")
    assert list(frame["time"]) == [100.0, 300.0, 600.0]
    assert [c for c in frame.columns if c != "time"] == ["subject_1", "subject_2", "subject_3"]


def test_forest_oob_requires_bootstrap() -> None:
    with pytest.raises(ValueError, match="bootstrap"):
        RandomSurvivalForest(bootstrap=False, oob_score=True)


def test_forest_predict_rejects_unknown_type(data) -> None:
    y, x = data
    rsf = RandomSurvivalForest(n_estimators=10, random_state=0).fit(y, x)
    with pytest.raises(ValueError, match="Unknown predict type"):
        rsf.predict(x, type="lp")


# -- summaries adapters ---------------------------------------------------------------------


def test_glance_forest(data) -> None:
    y, x = data
    rsf = RandomSurvivalForest(n_estimators=20, oob_score=True, random_state=0).fit(y, x)
    g = gw.glance(rsf, format="pandas")
    assert g.loc[0, "n"] == x.shape[0]
    assert g.loc[0, "n_estimators"] == 20
    assert g.loc[0, "n_features"] == x.shape[1]
    assert 0.0 <= g.loc[0, "oob_concordance"] <= 1.0


def test_tidy_forest_variable_importance(data) -> None:
    y, x = data
    rsf = RandomSurvivalForest(n_estimators=40, random_state=0).fit(y, x)
    t = gw.tidy(rsf, format="pandas")
    assert set(t["term"]) == set(x.columns)
    assert "importance" in t.columns
    # Importances are returned in descending order.
    assert list(t["importance"]) == sorted(t["importance"], reverse=True)


def test_variable_importance_requires_bootstrap(data) -> None:
    y, x = data
    rsf = RandomSurvivalForest(n_estimators=10, bootstrap=False, random_state=0).fit(y, x)
    with pytest.raises(ValueError, match="bootstrap"):
        rsf.variable_importance()


# -- ExtraSurvivalTrees ---------------------------------------------------------------------


def test_extra_trees_defaults_to_no_bootstrap() -> None:
    ext = ExtraSurvivalTrees()
    assert ext.bootstrap is False
    assert ext._splitter == "random"


def test_extra_trees_curves_are_valid(data) -> None:
    y, x = data
    ext = ExtraSurvivalTrees(n_estimators=40, random_state=0).fit(y, x)
    surv = _curve_matrix(ext.predict(x[:8], type="survival", format="pandas"))
    chf = _curve_matrix(ext.predict(x[:8], type="cumulative_hazard", format="pandas"))
    assert np.all(surv >= 0.0) and np.all(surv <= 1.0 + 1e-12)
    assert np.all(np.diff(surv, axis=0) <= 1e-12)
    assert np.all(np.diff(chf, axis=0) >= -1e-12)


def test_extra_trees_reproducible_under_seed(data) -> None:
    y, x = data
    r1 = ExtraSurvivalTrees(n_estimators=30, random_state=3).fit(y, x).predict(x)
    r2 = ExtraSurvivalTrees(n_estimators=30, random_state=3).fit(y, x).predict(x)
    assert np.allclose(r1, r2)
    r3 = ExtraSurvivalTrees(n_estimators=30, random_state=9).fit(y, x).predict(x)
    assert not np.allclose(r1, r3)


def test_extra_trees_beats_chance_out_of_bag(data) -> None:
    y, x = data
    ext = ExtraSurvivalTrees(n_estimators=150, bootstrap=True, oob_score=True, random_state=0).fit(
        y, x
    )
    assert ext.oob_score_ is not None
    assert ext.oob_score_ > 0.55


def test_extra_trees_summaries(data) -> None:
    y, x = data
    ext = ExtraSurvivalTrees(n_estimators=40, bootstrap=True, oob_score=True, random_state=0).fit(
        y, x
    )
    g = gw.glance(ext, format="pandas")
    assert g.loc[0, "n_estimators"] == 40
    assert 0.0 <= g.loc[0, "oob_concordance"] <= 1.0
    t = gw.tidy(ext, format="pandas")
    assert set(t["term"]) == set(x.columns)


def test_extra_trees_repr_uses_class_name(data) -> None:
    y, x = data
    ext = ExtraSurvivalTrees(n_estimators=10, random_state=0).fit(y, x)
    assert "ExtraSurvivalTrees" in repr(ext)
    assert "RandomSurvivalForest" not in repr(ext)


def test_survival_tree_random_splitter_runs(data) -> None:
    y, x = data
    tree = SurvivalTree(splitter="random", max_depth=4, random_state=0).fit(y, x)
    risk = tree.predict(x)
    assert risk.shape == (x.shape[0],)
    assert np.all(np.isfinite(risk))


def test_survival_tree_rejects_bad_splitter() -> None:
    with pytest.raises(ValueError, match="splitter"):
        SurvivalTree(splitter="nonsense")


# -- engine selection -----------------------------------------------------------------------


def test_bad_engine_raises() -> None:
    with pytest.raises(ValueError, match="engine"):
        RandomSurvivalForest(engine="bogus")


def test_engine_numpy_is_default() -> None:
    assert RandomSurvivalForest().engine == "numpy"


@pytest.mark.skipif(_HAS_NUMBA, reason="numba is installed")
def test_engine_numba_errors_without_numba(data) -> None:
    y, x = data
    with pytest.raises(ImportError, match="numba"):
        RandomSurvivalForest(n_estimators=5, engine="numba").fit(y, x)


@pytest.mark.skipif(not _HAS_NUMBA, reason="requires numba")
def test_engine_numba_matches_numpy_discrimination(data) -> None:
    y, x = data
    r_np = (
        RandomSurvivalForest(n_estimators=80, engine="numpy", random_state=0).fit(y, x).predict(x)
    )
    r_nb = (
        RandomSurvivalForest(n_estimators=80, engine="numba", random_state=0).fit(y, x).predict(x)
    )
    c_np = gw.concordance_index(y, r_np)
    c_nb = gw.concordance_index(y, r_nb)
    # The Numba kernel accumulates the log-rank statistic in a different order, so it yields a
    # statistically equivalent (not bitwise identical) fit.
    assert abs(c_np - c_nb) < 0.02
    assert np.corrcoef(r_np, r_nb)[0, 1] > 0.95


@pytest.mark.skipif(not _HAS_NUMBA, reason="requires numba")
def test_engine_auto_runs(data) -> None:
    y, x = data
    rsf = RandomSurvivalForest(n_estimators=20, engine="auto", random_state=0).fit(y, x)
    assert rsf.predict(x).shape == (x.shape[0],)


# -- engine and max-features resolvers ------------------------------------------------------


def test_resolve_engine_variants() -> None:
    assert _resolve_engine("numpy") is False
    assert _resolve_engine("auto") == forest._HAS_NUMBA
    with pytest.raises(ValueError, match="engine must be one of"):
        _resolve_engine("bogus")


def test_resolve_engine_numba_without_numba(monkeypatch) -> None:
    monkeypatch.setattr(forest, "_HAS_NUMBA", False)
    with pytest.raises(ImportError, match="numba"):
        _resolve_engine("numba")


@pytest.mark.parametrize(
    "spec, n_features, expected",
    [
        (None, 5, 5),
        ("sqrt", 9, 3),
        ("log2", 8, 3),
        (2, 5, 2),
        (0.5, 8, 4),
        (100, 5, 5),  # clipped to n_features
        (0, 5, 1),  # clipped up to 1
    ],
)
def test_resolve_max_features(spec, n_features, expected) -> None:
    assert _resolve_max_features(spec, n_features) == expected


def test_resolve_max_features_bad_string() -> None:
    with pytest.raises(ValueError, match="Unknown max_features"):
        _resolve_max_features("nonsense", 5)


def test_resolve_max_features_bad_type() -> None:
    with pytest.raises(TypeError, match="max_features must be"):
        _resolve_max_features(object(), 5)


# -- split-search helpers on degenerate inputs ----------------------------------------------


def test_compact_grid_no_events() -> None:
    time = np.array([1.0, 2.0, 3.0])
    event = np.array([False, False, False])
    event_times, n_tot, d_tot, var_factor = _compact_grid(time, event)
    assert event_times.size == 0
    assert n_tot.size == 0 and d_tot.size == 0 and var_factor.size == 0


def test_best_logrank_split_empty_grid() -> None:
    time = np.array([1.0, 2.0, 3.0])
    event = np.array([False, False, False])
    grid = _RiskEventGrid(time, event)
    x = np.array([[0.0], [1.0], [2.0]])
    assert _best_logrank_split(grid, x, np.array([0]), 1) is None


def test_best_logrank_split_constant_feature() -> None:
    time = np.array([1.0, 2.0, 3.0, 4.0])
    event = np.array([True, True, True, False])
    grid = _RiskEventGrid(time, event)
    x_const = np.zeros((4, 1))
    # No valid split when the only feature is constant.
    assert _best_logrank_split(grid, x_const, np.array([0]), 1) is None


def test_random_logrank_split_empty_and_constant() -> None:
    rng = np.random.default_rng(0)
    # Empty grid (no events).
    grid_empty = _RiskEventGrid(np.array([1.0, 2.0]), np.array([False, False]))
    assert _random_logrank_split(grid_empty, np.zeros((2, 1)), np.array([0]), 1, rng) is None
    # Constant feature yields no split.
    grid = _RiskEventGrid(np.array([1.0, 2.0, 3.0, 4.0]), np.array([True, True, True, False]))
    assert _random_logrank_split(grid, np.zeros((4, 1)), np.array([0]), 1, rng) is None


def test_random_logrank_split_finds_split() -> None:
    rng = np.random.default_rng(1)
    time, event, col = _small()
    grid = _RiskEventGrid(time, event)
    result = _random_logrank_split(grid, col[:, None], np.array([0]), 1, rng)
    assert result is not None
    feat, threshold, chi = result
    assert feat == 0 and np.isfinite(threshold) and chi >= 0


def test_best_logrank_split_numba_empty_grid() -> None:
    # Reaches the early return before the (possibly absent) compiled kernel is called.
    time = np.array([1.0, 2.0, 3.0])
    event = np.array([False, False, False])
    x = np.array([[0.0], [1.0], [2.0]])
    assert _best_logrank_split_numba(x, np.array([0]), 1, time, event) is None


def test_logrank_kernel_impl_pure_python() -> None:
    # Exercises the reference (un-jitted) kernel body directly.
    time, event, col = _small()
    event_times, n_tot, d_tot, var_factor = _compact_grid(time, event)
    time_idx_le = np.searchsorted(event_times, time, side="right").astype(np.int64)

    thr, chi = _logrank_kernel_impl(col, time_idx_le, event, n_tot, d_tot, var_factor, 1)
    assert np.isfinite(thr) and chi >= 0.0

    # No valid split when the leaf minimum exceeds the sample size.
    thr_none, chi_none = _logrank_kernel_impl(col, time_idx_le, event, n_tot, d_tot, var_factor, 10)
    assert np.isnan(thr_none) and chi_none == -1.0


# -- SurvivalTree constructor and fit validation --------------------------------------------


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"min_samples_split": 1}, "min_samples_split"),
        ({"min_samples_leaf": 0}, "min_samples_leaf"),
        ({"max_depth": 0}, "max_depth"),
        ({"splitter": "bogus"}, "splitter"),
    ],
)
def test_survival_tree_invalid_params(kwargs, match) -> None:
    with pytest.raises(ValueError, match=match):
        SurvivalTree(**kwargs)


def test_survival_tree_unfitted_repr() -> None:
    assert "<unfitted>" in repr(SurvivalTree())


def test_survival_tree_fit_requires_response() -> None:
    x = np.array([[1.0], [2.0], [3.0]])
    with pytest.raises(ValueError, match="Surv` response is required"):
        SurvivalTree().fit(None, x)


def test_survival_tree_fit_row_mismatch(data) -> None:
    y, x = data
    with pytest.raises(ValueError, match="same number of rows"):
        SurvivalTree(random_state=0).fit(y, x[:-3])


def test_survival_tree_fit_no_events() -> None:
    y = Surv.right(time=[5, 6, 7, 8], event=[0, 0, 0, 0])
    x = np.array([[1.0], [2.0], [3.0], [4.0]])
    with pytest.raises(ValueError, match="No events remain"):
        SurvivalTree().fit(y, x)


def test_survival_tree_engine_bad_raises_on_fit(data) -> None:
    y, x = data
    with pytest.raises(ValueError, match="engine must be one of"):
        SurvivalTree(engine="bogus", random_state=0).fit(y, x)


def test_survival_curve_before_first_event(data) -> None:
    # Query time below the first event time hits the step-function boundary branch.
    y, x = data
    tree = SurvivalTree(max_depth=3, random_state=0).fit(y, x)
    frame = tree.predict(x[:2], type="survival", times=[0.0], format="pandas")
    vals = frame[[c for c in frame.columns if c != "time"]].to_numpy()
    assert np.allclose(vals, 1.0)  # survival is 1 before any event


def test_survival_tree_predict_cumulative_hazard(data) -> None:
    y, x = data
    tree = SurvivalTree(max_depth=3, random_state=0).fit(y, x)
    frame = tree.predict(x[:3], type="cumulative_hazard", times=[100, 300], format="pandas")
    vals = frame[[c for c in frame.columns if c != "time"]].to_numpy()
    assert np.all(np.diff(vals, axis=0) >= -1e-12)  # non-decreasing over time


def test_survival_tree_predict_rejects_unknown_type(data) -> None:
    y, x = data
    tree = SurvivalTree(max_depth=3, random_state=0).fit(y, x)
    with pytest.raises(ValueError, match="Unknown predict type"):
        tree.predict(x, type="lp")


# -- forest constructor, repr, and fit validation -------------------------------------------


def test_forest_invalid_n_estimators() -> None:
    with pytest.raises(ValueError, match="n_estimators"):
        RandomSurvivalForest(n_estimators=0)


def test_forest_unfitted_repr() -> None:
    assert "<unfitted>" in repr(RandomSurvivalForest())
    assert "<unfitted>" in repr(ExtraSurvivalTrees())


def test_forest_fitted_repr_with_oob(data) -> None:
    y, x = data
    rsf = RandomSurvivalForest(n_estimators=20, oob_score=True, random_state=0).fit(y, x)
    assert "out-of-bag concordance" in repr(rsf)


def test_forest_fit_row_mismatch(data) -> None:
    y, x = data
    with pytest.raises(ValueError, match="same number of rows"):
        RandomSurvivalForest(n_estimators=3, random_state=0).fit(y, x[:-3])


def test_forest_fit_no_events() -> None:
    y = Surv.right(time=[5, 6, 7, 8, 9, 10], event=[0, 0, 0, 0, 0, 0])
    x = np.arange(6.0).reshape(6, 1)
    with pytest.raises(ValueError, match="No events remain"):
        RandomSurvivalForest(n_estimators=3, random_state=0).fit(y, x)


def test_forest_predict_survival_before_first_event(data) -> None:
    y, x = data
    rsf = RandomSurvivalForest(n_estimators=15, random_state=0).fit(y, x)
    frame = rsf.predict(x[:2], type="survival", times=[0.0], format="pandas")
    vals = frame[[c for c in frame.columns if c != "time"]].to_numpy()
    assert np.allclose(vals, 1.0)


def test_compute_oob_score_returns_none_without_oob_subjects(data) -> None:
    y, x = data
    rsf = RandomSurvivalForest(n_estimators=5, random_state=0).fit(y, x)
    # Force every tree to have an empty out-of-bag set: each tree is skipped and no subject is
    # scored, so the estimate is undefined.
    rsf._oob_masks = [np.zeros(rsf.n_, dtype=bool) for _ in rsf._oob_masks]
    assert rsf._compute_oob_score() is None


def test_permutation_importance_skips_tree_without_oob(data) -> None:
    y, x = data
    rsf = RandomSurvivalForest(n_estimators=8, random_state=0).fit(y, x)
    # Blank out one tree's OOB set so it is skipped inside the permutation-importance loop, while
    # the remaining trees still score enough subjects to compute importances.
    rsf._oob_masks[0] = np.zeros(rsf.n_, dtype=bool)
    result = rsf.variable_importance(n_repeats=2, random_state=0, format="pandas")
    assert set(result["term"]) == set(x.columns)
