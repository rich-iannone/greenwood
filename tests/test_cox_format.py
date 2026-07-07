"""Tests for format parameter in CoxPH DataFrame-returning methods."""

from __future__ import annotations

import numpy as np
import pytest

import greenwood as gw
from greenwood import CoxPH, Surv


@pytest.fixture
def lung_surv():  # type: ignore[no-untyped-def]
    df = gw.load_dataset("lung", backend="pandas")
    return df, Surv.right(df["time"], event=(df["status"] == 2))


class TestPredictFormat:
    """Test format parameter for predict() method."""

    def test_predict_format_pandas(self, lung_surv) -> None:
        """Test that format='pandas' returns pandas DataFrame."""
        df, y = lung_surv
        cox = CoxPH().fit(y, df[["age", "sex"]])
        result = cox.predict(
            df[["age", "sex"]].iloc[:2], type="survival", times=[100, 300], format="pandas"
        )
        import pandas as pd

        assert isinstance(result, pd.DataFrame)
        assert list(result.columns) == ["time", "subject_1", "subject_2"]

    def test_predict_format_polars(self, lung_surv) -> None:
        """Test that format='polars' returns polars DataFrame."""
        pytest.importorskip("polars")
        df, y = lung_surv
        cox = CoxPH().fit(y, df[["age", "sex"]])
        result = cox.predict(
            df[["age", "sex"]].iloc[:2], type="survival", times=[100, 300], format="polars"
        )
        import polars as pl

        assert isinstance(result, pl.DataFrame)
        assert list(result.columns) == ["time", "subject_1", "subject_2"]

    def test_predict_format_pyarrow(self, lung_surv) -> None:
        """Test that format='pyarrow' returns pyarrow Table."""
        pytest.importorskip("pyarrow")
        df, y = lung_surv
        cox = CoxPH().fit(y, df[["age", "sex"]])
        result = cox.predict(
            df[["age", "sex"]].iloc[:2], type="survival", times=[100, 300], format="pyarrow"
        )
        import pyarrow as pa

        assert isinstance(result, pa.Table)
        assert list(result.column_names) == ["time", "subject_1", "subject_2"]

    def test_predict_format_none_default(self, lung_surv) -> None:
        """Test that format=None auto-detects (prefers polars)."""
        df, y = lung_surv
        cox = CoxPH().fit(y, df[["age", "sex"]])
        result = cox.predict(
            df[["age", "sex"]].iloc[:2], type="survival", times=[100, 300], format=None
        )
        # Check if polars is installed; if so, should return polars DataFrame, else pandas
        try:
            import polars as pl

            assert isinstance(result, pl.DataFrame)
        except ImportError:
            import pandas as pd

            assert isinstance(result, pd.DataFrame)

    def test_predict_data_consistent_across_formats(self, lung_surv) -> None:
        """Test that data is consistent regardless of format."""
        df, y = lung_surv
        cox = CoxPH().fit(y, df[["age", "sex"]])
        nd = df[["age", "sex"]].iloc[:2]
        times = [100, 300, 500]

        result_pandas = cox.predict(nd, type="survival", times=times, format="pandas")
        result_pandas_vals = result_pandas[["subject_1", "subject_2"]].to_numpy()

        try:
            cox.predict(nd, type="survival", times=times, format="polars")
            result_polars = cox.predict(nd, type="survival", times=times, format="polars")
            result_polars_vals = result_polars[["subject_1", "subject_2"]].to_numpy()
            np.testing.assert_allclose(result_pandas_vals, result_polars_vals)
        except ImportError:
            pass


class TestBaselineHazardFormat:
    """Test format parameter for baseline_hazard() method."""

    def test_baseline_hazard_format_pandas(self, lung_surv) -> None:
        """Test that format='pandas' returns pandas DataFrame."""
        df, y = lung_surv
        cox = CoxPH().fit(y, df[["age", "sex"]])
        result = cox.baseline_hazard(format="pandas")
        import pandas as pd

        assert isinstance(result, pd.DataFrame)
        assert "cumhaz" in result.columns

    def test_baseline_hazard_format_polars(self, lung_surv) -> None:
        """Test that format='polars' returns polars DataFrame."""
        pytest.importorskip("polars")
        df, y = lung_surv
        cox = CoxPH().fit(y, df[["age", "sex"]])
        result = cox.baseline_hazard(format="polars")
        import polars as pl

        assert isinstance(result, pl.DataFrame)
        assert "cumhaz" in result.columns

    def test_baseline_hazard_format_pyarrow(self, lung_surv) -> None:
        """Test that format='pyarrow' returns pyarrow Table."""
        pytest.importorskip("pyarrow")
        df, y = lung_surv
        cox = CoxPH().fit(y, df[["age", "sex"]])
        result = cox.baseline_hazard(format="pyarrow")
        import pyarrow as pa

        assert isinstance(result, pa.Table)
        assert "cumhaz" in result.column_names

    def test_baseline_hazard_data_consistent_across_formats(self, lung_surv) -> None:
        """Test that baseline hazard data is consistent across formats."""
        df, y = lung_surv
        cox = CoxPH().fit(y, df[["age", "sex"]])

        result_pandas = cox.baseline_hazard(format="pandas")
        pandas_cumhaz = result_pandas["cumhaz"].to_numpy()

        try:
            result_polars = cox.baseline_hazard(format="polars")
            polars_cumhaz = result_polars["cumhaz"].to_numpy()
            np.testing.assert_allclose(pandas_cumhaz, polars_cumhaz)
        except ImportError:
            pass


class TestResidualsFormat:
    """Test format parameter for residuals() method."""

    def test_residuals_format_pandas(self, lung_surv) -> None:
        """Test that format='pandas' returns pandas DataFrame."""
        df, y = lung_surv
        cox = CoxPH().fit(y, df[["age", "sex"]])
        result = cox.residuals(type="schoenfeld", format="pandas")
        import pandas as pd

        assert isinstance(result, pd.DataFrame)

    def test_residuals_format_polars(self, lung_surv) -> None:
        """Test that format='polars' returns polars DataFrame."""
        pytest.importorskip("polars")
        df, y = lung_surv
        cox = CoxPH().fit(y, df[["age", "sex"]])
        result = cox.residuals(type="schoenfeld", format="polars")
        import polars as pl

        assert isinstance(result, pl.DataFrame)

    def test_residuals_format_pyarrow(self, lung_surv) -> None:
        """Test that format='pyarrow' returns pyarrow Table."""
        pytest.importorskip("pyarrow")
        df, y = lung_surv
        cox = CoxPH().fit(y, df[["age", "sex"]])
        result = cox.residuals(type="schoenfeld", format="pyarrow")
        import pyarrow as pa

        assert isinstance(result, pa.Table)

    def test_residuals_data_consistent_across_formats(self, lung_surv) -> None:
        """Test that residuals data is consistent across formats."""
        df, y = lung_surv
        cox = CoxPH().fit(y, df[["age", "sex"]])

        result_pandas = cox.residuals(type="schoenfeld", format="pandas")
        pandas_vals = result_pandas.to_numpy()

        try:
            result_polars = cox.residuals(type="schoenfeld", format="polars")
            polars_vals = result_polars.to_numpy()
            np.testing.assert_allclose(pandas_vals, polars_vals)
        except ImportError:
            pass
