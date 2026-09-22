"""Tests for concordance_index_ci, concordance_index_compare, and score_cr."""

from __future__ import annotations

import numpy as np
import pytest

import greenwood as gw

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def lung_data():
    """Right-censored lung data with two Cox models."""
    lung = gw.load_dataset("lung", backend="pandas").dropna(subset=["ph.ecog"])
    y = gw.Surv.right(lung["time"], event=(lung["status"] == 2))
    cox_full = gw.CoxPH().fit(y, lung[["age", "sex"]])
    cox_age = gw.CoxPH().fit(y, lung[["age"]])
    lp_full = cox_full.predict(type="lp")
    lp_age = cox_age.predict(type="lp")
    return y, lp_full, lp_age, lung


@pytest.fixture(scope="module")
def cr_data():
    """Competing-risks data with a fitted Fine-Gray model."""
    mg = gw.load_dataset("mgus2", backend="pandas")
    etime = np.where(mg["pstat"] == 1, mg["ptime"], mg["futime"])
    cause = np.where(mg["pstat"] == 1, 1, 2 * mg["death"])
    y = gw.Surv.multistate(etime, event=cause, states=("pcm", "death"))
    fg = gw.FineGray("pcm").fit(y, mg[["age", "sex"]])
    return y, fg, mg


# ---------------------------------------------------------------------------
# concordance_index_ci
# ---------------------------------------------------------------------------


class TestConcordanceCI:
    def test_returns_result_object(self, lung_data):
        y, lp, _, _ = lung_data
        result = gw.concordance_index_ci(y, lp)

        assert isinstance(result, gw.ConcordanceResult)

    def test_estimate_matches_ipcw(self, lung_data):
        y, lp, _, _ = lung_data
        result = gw.concordance_index_ci(y, lp)
        c_ipcw = gw.concordance_index_ipcw(y, lp)
        np.testing.assert_allclose(result.estimate, c_ipcw, atol=1e-10)

    def test_ci_contains_estimate(self, lung_data):
        y, lp, _, _ = lung_data
        result = gw.concordance_index_ci(y, lp)

        assert result.ci_low <= result.estimate <= result.ci_high

    def test_ci_bounded(self, lung_data):
        y, lp, _, _ = lung_data
        result = gw.concordance_index_ci(y, lp)

        assert 0.0 <= result.ci_low <= 1.0
        assert 0.0 <= result.ci_high <= 1.0

    def test_se_positive(self, lung_data):
        y, lp, _, _ = lung_data
        result = gw.concordance_index_ci(y, lp)

        assert result.se > 0

    def test_n_matches(self, lung_data):
        y, lp, _, _ = lung_data
        result = gw.concordance_index_ci(y, lp)

        assert result.n == y.n

    def test_repr(self, lung_data):
        y, lp, _, _ = lung_data
        result = gw.concordance_index_ci(y, lp)
        r = repr(result)

        assert "ConcordanceResult" in r
        assert "CI=" in r

    def test_to_frame(self, lung_data):
        y, lp, _, _ = lung_data
        result = gw.concordance_index_ci(y, lp)
        df = result.to_frame(format="pandas")

        assert df.shape == (1, 5)
        assert "estimate" in df.columns
        assert "se" in df.columns

    def test_identity_transform(self, lung_data):
        y, lp, _, _ = lung_data
        result = gw.concordance_index_ci(y, lp, transform="identity")

        assert result.ci_low <= result.estimate <= result.ci_high

    def test_log_transform(self, lung_data):
        y, lp, _, _ = lung_data
        result = gw.concordance_index_ci(y, lp, transform="log")

        assert result.ci_low <= result.estimate <= result.ci_high

    def test_invalid_transform(self, lung_data):
        y, lp, _, _ = lung_data
        with pytest.raises(ValueError, match="transform"):
            gw.concordance_index_ci(y, lp, transform="bogus")

    def test_custom_conf_level(self, lung_data):
        y, lp, _, _ = lung_data
        r95 = gw.concordance_index_ci(y, lp, conf_level=0.95)
        r99 = gw.concordance_index_ci(y, lp, conf_level=0.99)

        assert r99.ci_high - r99.ci_low > r95.ci_high - r95.ci_low

    def test_with_tau(self, lung_data):
        y, lp, _, _ = lung_data
        result = gw.concordance_index_ci(y, lp, tau=365.0)

        assert result.estimate > 0

    def test_cause_specific(self, cr_data):
        y, fg, mg = cr_data
        eval_times = np.array([360.0])
        cif = fg.predict_cumulative_incidence(mg[["age", "sex"]], times=eval_times, format="pandas")
        cif_at_tau = cif.drop(columns="time").values.flatten()
        result = gw.concordance_index_ci(y, cif_at_tau, cause=1, tau=360.0)

        assert isinstance(result, gw.ConcordanceResult)
        assert result.ci_low <= result.estimate <= result.ci_high

    def test_length_mismatch(self, lung_data):
        y, lp, _, _ = lung_data
        with pytest.raises(ValueError, match="same length"):
            gw.concordance_index_ci(y, lp[:10])


# ---------------------------------------------------------------------------
# concordance_index_compare
# ---------------------------------------------------------------------------


class TestConcordanceCompare:
    def test_returns_compare_result(self, lung_data):
        y, lp_full, lp_age, _ = lung_data
        result = gw.concordance_index_compare(y, lp_full, lp_age)

        assert isinstance(result, gw.ConcordanceCompareResult)

    def test_delta_sign(self, lung_data):
        y, lp_full, lp_age, _ = lung_data
        result = gw.concordance_index_compare(y, lp_full, lp_age)
        c_full = gw.concordance_index_ipcw(y, lp_full)
        c_age = gw.concordance_index_ipcw(y, lp_age)

        np.testing.assert_allclose(result.delta, c_full - c_age, atol=1e-10)

    def test_ci_contains_delta(self, lung_data):
        y, lp_full, lp_age, _ = lung_data
        result = gw.concordance_index_compare(y, lp_full, lp_age)

        assert result.ci_low <= result.delta <= result.ci_high

    def test_pvalue_bounded(self, lung_data):
        y, lp_full, lp_age, _ = lung_data
        result = gw.concordance_index_compare(y, lp_full, lp_age)

        assert 0.0 <= result.pvalue <= 1.0

    def test_same_model_delta_zero(self, lung_data):
        y, lp_full, _, _ = lung_data
        result = gw.concordance_index_compare(y, lp_full, lp_full)

        assert abs(result.delta) < 1e-10
        assert result.pvalue > 0.9

    def test_repr(self, lung_data):
        y, lp_full, lp_age, _ = lung_data
        result = gw.concordance_index_compare(y, lp_full, lp_age)

        assert "ConcordanceCompareResult" in repr(result)
        assert "p=" in repr(result)

    def test_to_frame(self, lung_data):
        y, lp_full, lp_age, _ = lung_data
        result = gw.concordance_index_compare(y, lp_full, lp_age)
        df = result.to_frame(format="polars")

        assert "delta" in df.columns
        assert "pvalue" in df.columns

    def test_length_mismatch(self, lung_data):
        y, lp_full, _, _ = lung_data
        with pytest.raises(ValueError, match="same length"):
            gw.concordance_index_compare(y, lp_full, lp_full[:10])


# ---------------------------------------------------------------------------
# score_cr
# ---------------------------------------------------------------------------


class TestScoreCR:
    def test_returns_dataframe(self, cr_data):
        y, fg, mg = cr_data
        eval_times = np.array([120, 240, 360])
        cif = fg.predict_cumulative_incidence(mg[["age", "sex"]], times=eval_times, format="pandas")
        cif_pred = cif.drop(columns="time").values.T
        result = gw.score_cr(y, cif_pred, cause=1, times=eval_times, format="pandas")

        assert "metric" in result.columns
        assert "estimate" in result.columns

    def test_correct_metrics(self, cr_data):
        y, fg, mg = cr_data
        eval_times = np.array([120, 240, 360])
        cif = fg.predict_cumulative_incidence(mg[["age", "sex"]], times=eval_times, format="pandas")
        cif_pred = cif.drop(columns="time").values.T
        result = gw.score_cr(y, cif_pred, cause=1, times=eval_times, format="pandas")
        metrics = list(result["metric"])

        assert metrics[0] == "concordance"
        assert "brier(120)" in metrics
        assert "brier(240)" in metrics
        assert "brier(360)" in metrics
        assert "integrated_brier" in metrics

    def test_concordance_has_ci(self, cr_data):
        y, fg, mg = cr_data
        eval_times = np.array([120, 240, 360])
        cif = fg.predict_cumulative_incidence(mg[["age", "sex"]], times=eval_times, format="pandas")
        cif_pred = cif.drop(columns="time").values.T
        result = gw.score_cr(y, cif_pred, cause=1, times=eval_times, format="pandas")
        conc_row = result[result["metric"] == "concordance"].iloc[0]

        assert not np.isnan(conc_row["se"])
        assert not np.isnan(conc_row["conf_low"])
        assert not np.isnan(conc_row["conf_high"])

    def test_brier_no_ci(self, cr_data):
        y, fg, mg = cr_data
        eval_times = np.array([120, 240])
        cif = fg.predict_cumulative_incidence(mg[["age", "sex"]], times=eval_times, format="pandas")
        cif_pred = cif.drop(columns="time").values.T
        result = gw.score_cr(y, cif_pred, cause=1, times=eval_times, format="pandas")
        brier_rows = result[result["metric"].str.startswith("brier")]

        assert brier_rows["se"].isna().all()

    def test_brier_nonnegative(self, cr_data):
        y, fg, mg = cr_data
        eval_times = np.array([120, 240, 360])
        cif = fg.predict_cumulative_incidence(mg[["age", "sex"]], times=eval_times, format="pandas")
        cif_pred = cif.drop(columns="time").values.T
        result = gw.score_cr(y, cif_pred, cause=1, times=eval_times, format="pandas")
        for _, row in result.iterrows():
            if row["metric"].startswith("brier") or row["metric"] == "integrated_brier":
                assert row["estimate"] >= 0

    def test_shape_mismatch(self, cr_data):
        y, fg, mg = cr_data
        eval_times = np.array([120, 240])
        cif_pred = np.random.rand(y.n, 3)  # wrong number of times
        with pytest.raises(ValueError, match="n_times"):
            gw.score_cr(y, cif_pred, cause=1, times=eval_times)

    def test_polars_format(self, cr_data):
        y, fg, mg = cr_data
        eval_times = np.array([120, 240])
        cif = fg.predict_cumulative_incidence(mg[["age", "sex"]], times=eval_times, format="pandas")
        cif_pred = cif.drop(columns="time").values.T
        result = gw.score_cr(y, cif_pred, cause=1, times=eval_times, format="polars")

        assert type(result).__name__ == "DataFrame"
        assert "metric" in result.columns
