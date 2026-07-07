"""Tests for cross_validate with different DataFrame formats (pandas, polars, pyarrow)."""

from __future__ import annotations

import numpy as np
import pytest

import greenwood as gw
from greenwood import CoxPH, Surv, cross_validate


@pytest.fixture
def lung_surv():  # type: ignore[no-untyped-def]
    df = gw.load_dataset("lung", backend="pandas")
    return df, Surv.right(df["time"], event=(df["status"] == 2))


class TestCrossValidatePandas:
    """Test cross_validate with pandas DataFrames (default)."""

    def test_cross_validate_cox_pandas(self, lung_surv) -> None:
        """Test cross_validate works with CoxPH and pandas."""
        df, y = lung_surv
        result = cross_validate(
            CoxPH(), y, df[["age", "sex"]], metric="brier", times=[180, 365], seed=42
        )
        assert result["metric"] == "brier"
        assert len(result["scores"]) > 0
        assert all(0.0 <= s <= 1.0 for s in result["scores"])
        # Pandas DataFrames should work
        assert isinstance(result["scores"], list)


class TestCrossValidatePolars:
    """Test cross_validate with polars DataFrames."""

    def test_cross_validate_cox_polars_data(self, lung_surv) -> None:
        """Test cross_validate with Polars data input."""
        pytest.importorskip("polars")
        import polars as pl

        df, y = lung_surv
        # Convert to polars
        df_polars = pl.from_pandas(df)
        result = cross_validate(
            CoxPH(), y, df_polars[["age", "sex"]], metric="brier", times=[180, 365], seed=42
        )
        assert result["metric"] == "brier"
        assert len(result["scores"]) > 0
        assert all(0.0 <= s <= 1.0 for s in result["scores"])

    def test_cross_validate_cox_polars_prediction_format(self, lung_surv) -> None:
        """Test cross_validate when predict returns polars DataFrame."""
        pytest.importorskip("polars")
        df, y = lung_surv
        # Use CoxPH which now returns polars by default (format=None)
        result = cross_validate(
            CoxPH(), y, df[["age", "sex"]], metric="brier", times=[180, 365], seed=42, k=3
        )
        assert result["metric"] == "brier"
        assert len(result["scores"]) == 3  # k=3 folds
        assert all(0.0 <= s <= 1.0 for s in result["scores"])


class TestCrossValidatePolyArrow:
    """Test cross_validate with pyarrow tables."""

    def test_cross_validate_cox_pyarrow_data(self, lung_surv) -> None:
        """Test cross_validate with PyArrow data input."""
        pytest.importorskip("pyarrow")
        import pyarrow as pa

        df, y = lung_surv
        # Convert to pyarrow
        df_arrow = pa.table(df[["age", "sex"]])
        # Note: pyarrow tables don't support iloc, but cross_validate should handle it
        # This tests the fallback column selection logic
        result = cross_validate(
            CoxPH(), y, df_arrow, metric="brier", times=[180, 365], seed=42, k=3
        )
        assert result["metric"] == "brier"
        assert len(result["scores"]) == 3
        assert all(0.0 <= s <= 1.0 for s in result["scores"])


class TestCrossValidateConsistency:
    """Test that cross_validate gives consistent results across formats."""

    def test_cross_validate_results_consistent_pandas_vs_polars(self, lung_surv) -> None:
        """Test that results are consistent between pandas and polars inputs."""
        pytest.importorskip("polars")
        import polars as pl

        df, y = lung_surv

        # Run with pandas
        result_pandas = cross_validate(
            CoxPH(), y, df[["age", "sex"]], metric="brier", times=[180, 365], seed=42, k=3
        )

        # Run with polars
        df_polars = pl.from_pandas(df)
        result_polars = cross_validate(
            CoxPH(), y, df_polars[["age", "sex"]], metric="brier", times=[180, 365], seed=42, k=3
        )

        # Results should be identical (same seed and folds)
        np.testing.assert_allclose(result_pandas["scores"], result_polars["scores"])
        assert result_pandas["mean"] == result_polars["mean"]
        assert result_pandas["std"] == result_polars["std"]

    def test_cross_validate_results_consistent_pandas_vs_pyarrow(self, lung_surv) -> None:
        """Test that results are consistent between pandas and pyarrow inputs."""
        pytest.importorskip("pyarrow")
        import pyarrow as pa

        df, y = lung_surv

        # Run with pandas
        result_pandas = cross_validate(
            CoxPH(), y, df[["age", "sex"]], metric="brier", times=[180, 365], seed=42, k=3
        )

        # Run with pyarrow
        df_arrow = pa.table(df[["age", "sex"]])
        result_arrow = cross_validate(
            CoxPH(), y, df_arrow, metric="brier", times=[180, 365], seed=42, k=3
        )

        # Results should be identical
        np.testing.assert_allclose(result_pandas["scores"], result_arrow["scores"])
        assert result_pandas["mean"] == result_arrow["mean"]
        assert result_pandas["std"] == result_arrow["std"]


class TestCrossValidateAFTWithFormats:
    """Test that cross_validate works with AFT and various formats."""

    def test_cross_validate_aft_pandas(self, lung_surv) -> None:
        """Test cross_validate works with AFT model and pandas."""
        df, y = lung_surv
        result = cross_validate(
            gw.AFT("weibull"), y, df[["age", "sex"]], metric="concordance", seed=42, k=3
        )
        assert result["metric"] == "concordance"
        assert len(result["scores"]) == 3
        assert all(0.0 <= s <= 1.0 for s in result["scores"])

    def test_cross_validate_aft_polars(self, lung_surv) -> None:
        """Test cross_validate works with AFT model and polars."""
        pytest.importorskip("polars")
        import polars as pl

        df, y = lung_surv
        df_polars = pl.from_pandas(df)
        # AFT doesn't have format parameter, so it returns pandas DataFrames
        # cross_validate should handle this gracefully
        result = cross_validate(
            gw.AFT("weibull"), y, df_polars[["age", "sex"]], metric="concordance", seed=42, k=3
        )
        assert result["metric"] == "concordance"
        assert len(result["scores"]) == 3
        assert all(0.0 <= s <= 1.0 for s in result["scores"])
