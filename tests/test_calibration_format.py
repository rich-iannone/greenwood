"""Tests for format parameter in calibration() function."""

from __future__ import annotations

import numpy as np
import pytest

import greenwood as gw
from greenwood import CoxPH, Surv


@pytest.fixture
def lung_surv():  # type: ignore[no-untyped-def]
    df = gw.load_dataset("lung", backend="pandas")
    return df, Surv.right(df["time"], event=(df["status"] == 2))


class TestCalibrationFormat:
    """Test format parameter for calibration() function."""

    def test_calibration_format_pandas(self, lung_surv) -> None:
        """Test that format='pandas' returns pandas DataFrame."""
        df, y = lung_surv
        cox = CoxPH().fit(y, df[["age", "sex"]])
        pred = (
            cox.predict(df[["age", "sex"]], type="survival", times=[365.0], format="pandas")
            .iloc[0, 1:]
            .to_numpy()
        )
        result = gw.calibration(y, pred, 365.0, n_bins=5, format="pandas")
        import pandas as pd

        assert isinstance(result, pd.DataFrame)
        assert list(result.columns) == [
            "bin",
            "n",
            "predicted",
            "observed",
            "observed_lower",
            "observed_upper",
        ]

    def test_calibration_format_polars(self, lung_surv) -> None:
        """Test that format='polars' returns polars DataFrame."""
        pytest.importorskip("polars")
        df, y = lung_surv
        cox = CoxPH().fit(y, df[["age", "sex"]])
        pred = (
            cox.predict(df[["age", "sex"]], type="survival", times=[365.0], format="pandas")
            .iloc[0, 1:]
            .to_numpy()
        )
        result = gw.calibration(y, pred, 365.0, n_bins=5, format="polars")
        import polars as pl

        assert isinstance(result, pl.DataFrame)
        assert list(result.columns) == [
            "bin",
            "n",
            "predicted",
            "observed",
            "observed_lower",
            "observed_upper",
        ]

    def test_calibration_format_pyarrow(self, lung_surv) -> None:
        """Test that format='pyarrow' returns pyarrow Table."""
        pytest.importorskip("pyarrow")
        df, y = lung_surv
        cox = CoxPH().fit(y, df[["age", "sex"]])
        pred = (
            cox.predict(df[["age", "sex"]], type="survival", times=[365.0], format="pandas")
            .iloc[0, 1:]
            .to_numpy()
        )
        result = gw.calibration(y, pred, 365.0, n_bins=5, format="pyarrow")
        import pyarrow as pa

        assert isinstance(result, pa.Table)
        assert list(result.column_names) == [
            "bin",
            "n",
            "predicted",
            "observed",
            "observed_lower",
            "observed_upper",
        ]

    def test_calibration_format_none_default(self, lung_surv) -> None:
        """Test that format=None auto-detects (prefers polars)."""
        df, y = lung_surv
        cox = CoxPH().fit(y, df[["age", "sex"]])
        pred = (
            cox.predict(df[["age", "sex"]], type="survival", times=[365.0], format="pandas")
            .iloc[0, 1:]
            .to_numpy()
        )
        result = gw.calibration(y, pred, 365.0, n_bins=5, format=None)
        # Check if polars is installed; if so, should return polars DataFrame, else pandas
        try:
            import polars as pl

            assert isinstance(result, pl.DataFrame)
        except ImportError:
            import pandas as pd

            assert isinstance(result, pd.DataFrame)

    def test_calibration_data_consistent_across_formats(self, lung_surv) -> None:
        """Test that calibration data is consistent across formats."""
        df, y = lung_surv
        cox = CoxPH().fit(y, df[["age", "sex"]])
        pred = (
            cox.predict(df[["age", "sex"]], type="survival", times=[365.0], format="pandas")
            .iloc[0, 1:]
            .to_numpy()
        )

        result_pandas = gw.calibration(y, pred, 365.0, n_bins=5, format="pandas")
        pandas_predicted = result_pandas["predicted"].to_numpy()
        pandas_observed = result_pandas["observed"].to_numpy()

        try:
            result_polars = gw.calibration(y, pred, 365.0, n_bins=5, format="polars")
            polars_predicted = result_polars["predicted"].to_numpy()
            polars_observed = result_polars["observed"].to_numpy()
            np.testing.assert_allclose(pandas_predicted, polars_predicted)
            np.testing.assert_allclose(pandas_observed, polars_observed)
        except ImportError:
            pass
