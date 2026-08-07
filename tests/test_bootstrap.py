"""Tests for bootstrap confidence intervals."""

from __future__ import annotations

import numpy as np
import pytest

from greenwood import BootstrapResult, KaplanMeier, Surv, bootstrap

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def lung_surv() -> Surv:
    """Lung dataset as a Surv response."""
    import greenwood as gw

    lung = gw.load_dataset("lung", backend="polars")
    return gw.Surv.right(lung["time"], event=(lung["status"] == 2))


@pytest.fixture()
def lung_sex(lung_surv: Surv) -> np.ndarray:
    """Sex grouping variable for the lung dataset."""
    import greenwood as gw

    lung = gw.load_dataset("lung", backend="polars")
    return np.array(lung["sex"].to_list())


@pytest.fixture()
def simple_surv() -> Surv:
    return Surv.right(
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        [1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
    )


# ---------------------------------------------------------------------------
# Basic functionality
# ---------------------------------------------------------------------------


class TestBootstrapMedian:
    def test_returns_bootstrap_result(self, simple_surv: Surv) -> None:
        result = bootstrap(simple_surv, "median", n_boot=50, seed=1)
        assert isinstance(result, BootstrapResult)
        assert result.n_boot == 50
        assert result.ci_type == "percentile"

    def test_estimate_matches_km_median(self, simple_surv: Surv) -> None:
        result = bootstrap(simple_surv, "median", n_boot=50, seed=1)
        km = KaplanMeier().fit(simple_surv)
        assert result.estimate == km.median()

    def test_ci_brackets_estimate(self, simple_surv: Surv) -> None:
        result = bootstrap(simple_surv, "median", n_boot=200, seed=1)
        assert result.conf_low <= result.estimate <= result.conf_high

    def test_se_positive(self, simple_surv: Surv) -> None:
        result = bootstrap(simple_surv, "median", n_boot=200, seed=1)
        assert result.se > 0

    def test_distribution_length(self, simple_surv: Surv) -> None:
        result = bootstrap(simple_surv, "median", n_boot=123, seed=1)
        assert len(result.distribution) == 123

    def test_reproducible_with_seed(self, simple_surv: Surv) -> None:
        r1 = bootstrap(simple_surv, "median", n_boot=100, seed=23)
        r2 = bootstrap(simple_surv, "median", n_boot=100, seed=23)
        np.testing.assert_array_equal(r1.distribution, r2.distribution)

    def test_different_seeds_differ(self, simple_surv: Surv) -> None:
        r1 = bootstrap(simple_surv, "median", n_boot=100, seed=1)
        r2 = bootstrap(simple_surv, "median", n_boot=100, seed=2)
        assert not np.array_equal(r1.distribution, r2.distribution)


class TestBootstrapRMST:
    def test_estimate_matches_km_rmst(self, simple_surv: Surv) -> None:
        result = bootstrap(simple_surv, "rmst", tau=8.0, n_boot=50, seed=1)
        km = KaplanMeier().fit(simple_surv)
        assert result.estimate == pytest.approx(km.rmst(8.0))

    def test_requires_tau(self, simple_surv: Surv) -> None:
        with pytest.raises(ValueError, match="tau="):
            bootstrap(simple_surv, "rmst", n_boot=10, seed=1)

    def test_ci_brackets_estimate(self, simple_surv: Surv) -> None:
        result = bootstrap(simple_surv, "rmst", tau=8.0, n_boot=200, seed=1)
        assert result.conf_low <= result.estimate <= result.conf_high


class TestBootstrapQuantile:
    def test_estimate_matches_km_quantile(self, simple_surv: Surv) -> None:
        result = bootstrap(simple_surv, "quantile", p=0.25, n_boot=50, seed=1)
        km = KaplanMeier().fit(simple_surv)
        assert result.estimate == km.quantile(0.25)

    def test_requires_p(self, simple_surv: Surv) -> None:
        with pytest.raises(ValueError, match="p="):
            bootstrap(simple_surv, "quantile", n_boot=10, seed=1)


class TestBootstrapSurvival:
    def test_estimate_matches_km_predict(self, simple_surv: Surv) -> None:
        result = bootstrap(simple_surv, "survival", times=5.0, n_boot=50, seed=1)
        km = KaplanMeier().fit(simple_surv)
        assert result.estimate == pytest.approx(float(km.predict([5.0])[0]))

    def test_requires_times(self, simple_surv: Surv) -> None:
        with pytest.raises(ValueError, match="times="):
            bootstrap(simple_surv, "survival", n_boot=10, seed=1)


# ---------------------------------------------------------------------------
# Grouped (diff) statistics
# ---------------------------------------------------------------------------


class TestBootstrapMedianDiff:
    def test_estimate_is_difference(self) -> None:
        y = Surv.right(
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        )
        group = ["A"] * 6 + ["B"] * 6
        result = bootstrap(y, "median_diff", by=group, n_boot=50, seed=1)
        km = KaplanMeier().fit(y, by=group)
        medians = km.median()
        expected = medians["A"] - medians["B"]
        assert result.estimate == pytest.approx(expected)

    def test_requires_by(self, simple_surv: Surv) -> None:
        with pytest.raises(ValueError, match="by="):
            bootstrap(simple_surv, "median_diff", n_boot=10, seed=1)

    def test_requires_exactly_two_groups(self) -> None:
        y = Surv.right([1, 2, 3, 4, 5, 6], [1, 1, 1, 1, 1, 1])
        group = ["A", "A", "B", "B", "C", "C"]
        with pytest.raises(ValueError, match="2 groups"):
            bootstrap(y, "median_diff", by=group, n_boot=10, seed=1)


class TestBootstrapRMSTDiff:
    def test_estimate_is_rmst_difference(self) -> None:
        y = Surv.right(
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
            [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
        )
        group = ["A"] * 6 + ["B"] * 6
        result = bootstrap(y, "rmst_diff", by=group, tau=10.0, n_boot=50, seed=1)
        km = KaplanMeier().fit(y, by=group)
        rmsts = km.rmst(10.0)
        expected = rmsts["A"] - rmsts["B"]
        assert result.estimate == pytest.approx(expected)


class TestBootstrapSurvivalDiff:
    def test_estimate_is_survival_difference(self) -> None:
        y = Surv.right(
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
            [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
        )
        group = ["A"] * 6 + ["B"] * 6
        result = bootstrap(y, "survival_diff", by=group, times=5.0, n_boot=50, seed=1)
        km = KaplanMeier().fit(y, by=group)
        preds = km.predict([5.0])
        expected = float(preds["A"][0]) - float(preds["B"][0])
        assert result.estimate == pytest.approx(expected)


# ---------------------------------------------------------------------------
# CI types
# ---------------------------------------------------------------------------


class TestCITypes:
    def test_percentile_is_default(self, simple_surv: Surv) -> None:
        result = bootstrap(simple_surv, "median", n_boot=100, seed=1)
        assert result.ci_type == "percentile"

    def test_normal_ci(self, simple_surv: Surv) -> None:
        result = bootstrap(simple_surv, "median", n_boot=200, seed=1, ci_type="normal")
        assert result.ci_type == "normal"
        # Normal CI should be symmetric around estimate
        assert result.estimate - result.conf_low == pytest.approx(
            result.conf_high - result.estimate, abs=1e-10
        )

    def test_bca_ci(self, simple_surv: Surv) -> None:
        result = bootstrap(simple_surv, "median", n_boot=200, seed=1, ci_type="bca")
        assert result.ci_type == "bca"
        assert np.isfinite(result.conf_low)
        assert np.isfinite(result.conf_high)

    def test_invalid_ci_type(self, simple_surv: Surv) -> None:
        with pytest.raises(ValueError, match="ci_type"):
            bootstrap(simple_surv, "median", n_boot=10, seed=1, ci_type="bogus")


# ---------------------------------------------------------------------------
# Custom callables
# ---------------------------------------------------------------------------


class TestCustomStatistic:
    def test_callable_receives_fitted_km(self, simple_surv: Surv) -> None:
        result = bootstrap(simple_surv, lambda km: km.rmst(8.0), n_boot=50, seed=1)
        km = KaplanMeier().fit(simple_surv)
        assert result.estimate == pytest.approx(km.rmst(8.0))

    def test_callable_rmst_difference(self) -> None:
        y = Surv.right(
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
            [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
        )
        group = ["A"] * 6 + ["B"] * 6

        def rmst_diff_fn(km: KaplanMeier) -> float:
            r = km.rmst(10.0)
            return r["A"] - r["B"]

        result = bootstrap(y, rmst_diff_fn, by=group, n_boot=100, seed=1)
        assert np.isfinite(result.estimate)
        assert result.se > 0


# ---------------------------------------------------------------------------
# Weights
# ---------------------------------------------------------------------------


class TestWeights:
    def test_weighted_bootstrap(self, simple_surv: Surv) -> None:
        weights = np.ones(10) * 2.0
        result = bootstrap(simple_surv, "median", weights=weights, n_boot=100, seed=1)
        assert isinstance(result, BootstrapResult)
        assert np.isfinite(result.estimate)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


class TestValidation:
    def test_invalid_statistic_name(self, simple_surv: Surv) -> None:
        with pytest.raises(ValueError, match="Unknown statistic"):
            bootstrap(simple_surv, "bogus", n_boot=10, seed=1)

    def test_invalid_conf_level(self, simple_surv: Surv) -> None:
        with pytest.raises(ValueError, match="conf_level"):
            bootstrap(simple_surv, "median", n_boot=10, seed=1, conf_level=1.5)

    def test_invalid_n_boot(self, simple_surv: Surv) -> None:
        with pytest.raises(ValueError, match="n_boot"):
            bootstrap(simple_surv, "median", n_boot=0, seed=1)

    def test_single_stat_with_by_raises(self, simple_surv: Surv) -> None:
        group = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
        with pytest.raises(ValueError, match="grouped"):
            bootstrap(simple_surv, "median", by=group, n_boot=10, seed=1)


# ---------------------------------------------------------------------------
# Output / repr / to_frame
# ---------------------------------------------------------------------------


class TestOutput:
    def test_repr(self, simple_surv: Surv) -> None:
        result = bootstrap(simple_surv, "median", n_boot=50, seed=1)
        r = repr(result)
        assert "BootstrapResult" in r
        assert "n_boot=50" in r

    def test_to_frame_pandas(self, simple_surv: Surv) -> None:
        result = bootstrap(simple_surv, "median", n_boot=50, seed=1)
        df = result.to_frame(format="pandas")
        assert list(df.columns) == ["estimate", "se", "conf_low", "conf_high"]
        assert len(df) == 1
        assert float(df["estimate"].iloc[0]) == result.estimate


# ---------------------------------------------------------------------------
# Larger-sample behavior (lung dataset)
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestLungDataset:
    def test_bootstrap_median_close_to_analytical(self, lung_surv: Surv) -> None:
        result = bootstrap(lung_surv, "median", n_boot=500, seed=23)
        km = KaplanMeier().fit(lung_surv)
        analytical_median = km.median()
        assert result.estimate == analytical_median
        assert result.conf_low < analytical_median < result.conf_high

    def test_bootstrap_rmst_se_close_to_analytical(self, lung_surv: Surv) -> None:
        result = bootstrap(lung_surv, "rmst", tau=365.0, n_boot=500, seed=23)
        km = KaplanMeier().fit(lung_surv)
        from greenwood._nonparametric import _rmst_block

        _, analytical_se = _rmst_block(km._blocks[0], 365.0)
        ratio = result.se / analytical_se
        assert 0.5 < ratio < 2.0

    def test_bootstrap_median_diff(self, lung_surv: Surv, lung_sex: np.ndarray) -> None:
        result = bootstrap(lung_surv, "median_diff", by=lung_sex, n_boot=500, seed=23)
        km = KaplanMeier().fit(lung_surv, by=lung_sex)
        medians = km.median()
        keys = sorted(medians.keys(), key=str)
        expected_diff = medians[keys[0]] - medians[keys[1]]
        assert result.estimate == pytest.approx(expected_diff)
        assert result.se > 0

    def test_bootstrap_rmst_diff(self, lung_surv: Surv, lung_sex: np.ndarray) -> None:
        result = bootstrap(lung_surv, "rmst_diff", by=lung_sex, tau=365.0, n_boot=500, seed=23)
        assert np.isfinite(result.estimate)
        assert result.se > 0
        assert result.conf_low < result.estimate < result.conf_high
