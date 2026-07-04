"""Tests for the bundled survival datasets."""

from __future__ import annotations

import pytest

from greenwood import data


def test_available_datasets() -> None:
    assert set(data.available_datasets()) == {
        "lung",
        "veteran",
        "ovarian",
        "pbc",
        "colon",
        "mgus2",
    }


def test_load_lung() -> None:
    df = data.load_dataset("lung")
    assert df.shape == (228, 10)
    assert "time" in df.columns
    assert "status" in df.columns


def test_load_veteran() -> None:
    assert data.load_dataset("veteran").shape == (137, 8)


def test_load_polars_backend() -> None:
    df = data.load_dataset("lung", backend="polars")
    assert df.shape == (228, 10)


def test_load_pandas_backend() -> None:
    pd = pytest.importorskip("pandas")
    df = data.load_dataset("lung", backend="pandas")
    assert isinstance(df, pd.DataFrame)
    assert df.shape == (228, 10)


def test_default_backend_prefers_polars() -> None:
    # With both installed (as in dev), the agnostic default resolves to Polars.
    pl = pytest.importorskip("polars")
    assert isinstance(data.load_dataset("lung"), pl.DataFrame)


def test_unknown_dataset_raises() -> None:
    with pytest.raises(ValueError, match="Unknown dataset"):
        data.load_dataset("nope")


def test_unknown_backend_raises() -> None:
    with pytest.raises(ValueError, match="Unknown backend"):
        data.load_dataset("lung", backend="numpy")
