"""Tests for CompetingRiskForest."""

from __future__ import annotations

import numpy as np
import pytest

import greenwood as gw

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def cr_data():
    """Build a competing-risks dataset from mgus2."""
    mg = gw.load_dataset("mgus2", backend="pandas")
    etime = np.where(mg["pstat"] == 1, mg["ptime"], mg["futime"])
    cause = np.where(mg["pstat"] == 1, 1, 2 * mg["death"])
    y = gw.Surv.multistate(etime, event=cause, states=("pcm", "death"))
    X = mg[["age", "sex"]]
    return y, X, mg


@pytest.fixture(scope="module")
def fitted_forest(cr_data):
    """A small fitted CompetingRiskForest for reuse across tests."""
    y, X, _ = cr_data
    return gw.CompetingRiskForest(n_estimators=10, max_depth=4, random_state=23).fit(y, X)


# ---------------------------------------------------------------------------
# Basic fit / repr / attributes
# ---------------------------------------------------------------------------


class TestFitAndRepr:
    def test_unfitted_repr(self):
        crf = gw.CompetingRiskForest(n_estimators=5)

        assert "<unfitted>" in repr(crf)

    def test_fitted_repr(self, fitted_forest):
        r = repr(fitted_forest)

        assert "logrankCR" in r
        assert "pcm" in r and "death" in r

    def test_fitted_attributes(self, fitted_forest):
        crf = fitted_forest

        assert crf.n_ == 1384
        assert crf.n_event_ > 0
        assert crf.n_features_in_ == 2
        assert len(crf.trees_) == 10
        assert crf.event_times_.shape[0] > 0
        assert crf.states_ == ("pcm", "death")
        assert crf.cause_codes_ == [1, 2]

    def test_feature_names(self, fitted_forest):
        assert "age" in fitted_forest.feature_names_in_


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


class TestValidation:
    def test_right_censored_rejected(self):
        y = gw.Surv.right([1, 2, 3], event=[True, False, True])
        with pytest.raises(NotImplementedError, match="multistate"):
            gw.CompetingRiskForest(n_estimators=2).fit(y, np.array([[1], [2], [3]]))

    def test_row_mismatch(self, cr_data):
        y, X, _ = cr_data
        with pytest.raises(ValueError, match="same number"):
            gw.CompetingRiskForest(n_estimators=2).fit(y, X.head(10))

    def test_n_estimators_zero(self):
        with pytest.raises(ValueError, match="n_estimators"):
            gw.CompetingRiskForest(n_estimators=0)

    def test_oob_without_bootstrap(self):
        with pytest.raises(ValueError, match="oob_score"):
            gw.CompetingRiskForest(oob_score=True, bootstrap=False)

    def test_max_depth_zero(self):
        with pytest.raises(ValueError, match="max_depth"):
            gw.CompetingRiskForest(max_depth=0)


# ---------------------------------------------------------------------------
# Predict CIF
# ---------------------------------------------------------------------------


class TestPredictCIF:
    def test_output_shape(self, fitted_forest, cr_data):
        _, X, _ = cr_data
        cif = fitted_forest.predict(X.head(5), cause="pcm", format="pandas")

        assert cif.shape[0] == fitted_forest.event_times_.shape[0]
        assert "time" in cif.columns
        assert "subject_1" in cif.columns
        assert "subject_5" in cif.columns

    def test_cif_bounded(self, fitted_forest, cr_data):
        _, X, _ = cr_data
        cif = fitted_forest.predict(X.head(10), cause="pcm", format="pandas")
        vals = cif.drop(columns="time").values

        assert np.all(vals >= 0.0)
        assert np.all(vals <= 1.0)

    def test_cif_monotone(self, fitted_forest, cr_data):
        _, X, _ = cr_data
        cif = fitted_forest.predict(X.head(5), cause="pcm", format="pandas")
        vals = cif.drop(columns="time").values
        diffs = np.diff(vals, axis=0)

        assert np.all(diffs >= -1e-12)

    def test_cause_by_integer(self, fitted_forest, cr_data):
        _, X, _ = cr_data
        cif_str = fitted_forest.predict(X.head(3), cause="pcm", format="pandas")
        cif_int = fitted_forest.predict(X.head(3), cause=1, format="pandas")

        np.testing.assert_array_equal(
            cif_str.drop(columns="time").values,
            cif_int.drop(columns="time").values,
        )

    def test_death_cause(self, fitted_forest, cr_data):
        _, X, _ = cr_data
        cif = fitted_forest.predict(X.head(3), cause="death", format="pandas")
        vals = cif.drop(columns="time").values

        assert np.all(vals >= 0.0)
        assert np.all(vals <= 1.0)

    def test_unknown_cause_string(self, fitted_forest, cr_data):
        _, X, _ = cr_data
        with pytest.raises(ValueError, match="Unknown cause"):
            fitted_forest.predict(X.head(3), cause="bogus")

    def test_unknown_cause_int(self, fitted_forest, cr_data):
        _, X, _ = cr_data
        with pytest.raises(ValueError, match="Unknown cause"):
            fitted_forest.predict(X.head(3), cause=99)

    def test_cause_required_multicause(self, fitted_forest, cr_data):
        _, X, _ = cr_data
        with pytest.raises(ValueError, match="Multiple causes"):
            fitted_forest.predict(X.head(3))

    def test_specific_times(self, fitted_forest, cr_data):
        _, X, _ = cr_data
        times = [365, 730, 1095]
        cif = fitted_forest.predict(X.head(3), cause="pcm", times=times, format="pandas")

        assert cif.shape[0] == 3
        np.testing.assert_array_equal(cif["time"].values, times)

    def test_polars_format(self, fitted_forest, cr_data):
        _, X, _ = cr_data
        cif = fitted_forest.predict(X.head(3), cause="pcm", format="polars")

        assert type(cif).__name__ == "DataFrame"
        assert "time" in cif.columns


# ---------------------------------------------------------------------------
# Predict risk
# ---------------------------------------------------------------------------


class TestPredictRisk:
    def test_risk_shape(self, fitted_forest, cr_data):
        _, X, _ = cr_data
        risk = fitted_forest.predict_risk(X.head(10), cause="pcm")

        assert risk.shape == (10,)

    def test_risk_nonnegative(self, fitted_forest, cr_data):
        _, X, _ = cr_data
        risk = fitted_forest.predict_risk(X.head(10), cause="pcm")

        assert np.all(risk >= 0.0)

    def test_different_covariates_different_risk(self, cr_data):
        y, X, _ = cr_data
        crf = gw.CompetingRiskForest(n_estimators=20, max_depth=5, random_state=0).fit(y, X)
        X_low = np.array([[30.0, 1.0]] * 5)
        X_high = np.array([[80.0, 1.0]] * 5)
        risk_low = crf.predict_risk(X_low, cause="death")
        risk_high = crf.predict_risk(X_high, cause="death")

        assert risk_low.shape == (5,)
        assert risk_high.shape == (5,)

        # Older subjects should generally have higher death risk
        assert risk_high.mean() > risk_low.mean()


# ---------------------------------------------------------------------------
# OOB score
# ---------------------------------------------------------------------------


class TestOOB:
    def test_oob_score(self, cr_data):
        y, X, _ = cr_data
        crf = gw.CompetingRiskForest(
            n_estimators=10, max_depth=3, oob_score=True, random_state=23
        ).fit(y, X)

        assert crf.oob_score_ is not None
        assert 0.0 < crf.oob_score_ < 1.0

    def test_oob_none_without_flag(self, fitted_forest):
        assert fitted_forest.oob_score_ is None


# ---------------------------------------------------------------------------
# Variable importance
# ---------------------------------------------------------------------------


class TestVariableImportance:
    def test_vimp_shape(self, cr_data):
        y, X, _ = cr_data
        crf = gw.CompetingRiskForest(
            n_estimators=10, max_depth=3, bootstrap=True, random_state=23
        ).fit(y, X)
        vimp = crf.variable_importance(cause="pcm", n_repeats=2, format="pandas")

        assert vimp.shape == (2, 2)
        assert list(vimp.columns) == ["term", "importance"]

    def test_vimp_requires_bootstrap(self, cr_data):
        y, X, _ = cr_data
        crf = gw.CompetingRiskForest(
            n_estimators=5, max_depth=3, bootstrap=False, random_state=23
        ).fit(y, X)
        with pytest.raises(ValueError, match="bootstrap"):
            crf.variable_importance()


# ---------------------------------------------------------------------------
# Tidy / glance
# ---------------------------------------------------------------------------


class TestTidyGlance:
    def test_tidy(self, cr_data):
        y, X, _ = cr_data
        crf = gw.CompetingRiskForest(
            n_estimators=10, max_depth=3, bootstrap=True, random_state=23
        ).fit(y, X)
        t = gw.tidy(crf, format="pandas")

        assert "term" in t.columns
        assert "importance" in t.columns

    def test_glance(self, cr_data):
        y, X, _ = cr_data
        crf = gw.CompetingRiskForest(n_estimators=10, max_depth=3, random_state=23).fit(y, X)
        g = gw.glance(crf, format="pandas")

        assert g.shape[0] == 1
        assert "n" in g.columns
        assert "n_causes" in g.columns


# ---------------------------------------------------------------------------
# Splitting criterion correctness
# ---------------------------------------------------------------------------


class TestSplittingCriterion:
    def test_trees_have_splits(self, fitted_forest):
        """Trees with max_depth > 1 should have internal (non-leaf) nodes."""
        for root in fitted_forest.trees_:
            if not root.is_leaf:
                assert root.left is not None
                assert root.right is not None

    def test_leaf_cif_per_cause(self, fitted_forest):
        """Every leaf should have CIF arrays for all causes."""

        def check_leaves(node):
            if node.is_leaf:
                assert node.cif is not None

                for k in fitted_forest.cause_codes_:
                    assert k in node.cif
                    assert node.cif[k].shape == (fitted_forest.event_times_.shape[0],)
            else:
                check_leaves(node.left)
                check_leaves(node.right)

        for root in fitted_forest.trees_:
            check_leaves(root)

    def test_leaf_cif_bounded(self, fitted_forest):
        """All leaf CIF values should be in [0, 1]."""

        def check_bounds(node):
            if node.is_leaf:
                for k in fitted_forest.cause_codes_:
                    assert np.all(node.cif[k] >= 0.0)
                    assert np.all(node.cif[k] <= 1.0 + 1e-10)
            else:
                check_bounds(node.left)
                check_bounds(node.right)

        for root in fitted_forest.trees_:
            check_bounds(root)

    def test_leaf_cif_monotone(self, fitted_forest):
        """Each leaf's CIF should be non-decreasing."""

        def check_mono(node):
            if node.is_leaf:
                for k in fitted_forest.cause_codes_:
                    diffs = np.diff(node.cif[k])

                    assert np.all(diffs >= -1e-12)
            else:
                check_mono(node.left)
                check_mono(node.right)

        for root in fitted_forest.trees_:
            check_mono(root)

    def test_cif_sum_leq_one(self, fitted_forest):
        """At each leaf, sum of CIF across all causes should be <= 1."""

        def check_sum(node):
            if node.is_leaf:
                total = sum(node.cif[k] for k in fitted_forest.cause_codes_)

                assert np.all(total <= 1.0 + 1e-10)
            else:
                check_sum(node.left)
                check_sum(node.right)

        for root in fitted_forest.trees_:
            check_sum(root)


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------


class TestReproducibility:
    def test_same_seed_same_result(self, cr_data):
        y, X, _ = cr_data
        crf1 = gw.CompetingRiskForest(n_estimators=5, max_depth=3, random_state=123).fit(y, X)
        crf2 = gw.CompetingRiskForest(n_estimators=5, max_depth=3, random_state=123).fit(y, X)
        cif1 = crf1.predict(X.head(3), cause="pcm", format="pandas")
        cif2 = crf2.predict(X.head(3), cause="pcm", format="pandas")

        np.testing.assert_array_equal(
            cif1.drop(columns="time").values,
            cif2.drop(columns="time").values,
        )


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_single_tree(self, cr_data):
        y, X, _ = cr_data
        crf = gw.CompetingRiskForest(n_estimators=1, max_depth=2, random_state=0).fit(y, X)
        cif = crf.predict(X.head(3), cause="pcm", format="pandas")

        assert cif.shape[0] > 0

    def test_no_bootstrap(self, cr_data):
        y, X, _ = cr_data
        crf = gw.CompetingRiskForest(
            n_estimators=3, max_depth=2, bootstrap=False, random_state=0
        ).fit(y, X)
        cif = crf.predict(X.head(3), cause="pcm", format="pandas")

        assert cif.shape[0] > 0

    def test_max_depth_one(self, cr_data):
        y, X, _ = cr_data
        crf = gw.CompetingRiskForest(n_estimators=3, max_depth=1, random_state=0).fit(y, X)
        cif = crf.predict(X.head(3), cause="pcm", format="pandas")

        assert cif.shape[0] > 0

    def test_predict_training_data(self, fitted_forest):
        """predict() with newdata=None should use training data."""
        cif = fitted_forest.predict(cause="pcm", format="pandas")
        n_subjects = cif.shape[1] - 1  # minus the time column

        assert n_subjects == fitted_forest.n_
