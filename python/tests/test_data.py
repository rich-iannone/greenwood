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
        "pbcseq",
        "colon",
        "mgus2",
        "e1684",
    }


def test_load_lung() -> None:
    df = data.load_dataset("lung")
    assert df.shape == (228, 10)
    assert "time" in df.columns
    assert "status" in df.columns


def test_load_veteran() -> None:
    assert data.load_dataset("veteran").shape == (137, 8)


def test_load_pbcseq() -> None:
    df = data.load_dataset("pbcseq")
    assert df.shape == (1945, 19)
    # key columns for TVC usage
    for col in ("id", "day", "bili", "albumin", "futime", "status"):
        assert col in df.columns


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


FACTORS = {
    ("veteran", "celltype"): ["squamous", "smallcell", "adeno", "large"],
    ("colon", "rx"): ["Obs", "Lev", "Lev+5FU"],
    ("pbc", "sex"): ["m", "f"],
    ("pbcseq", "sex"): ["m", "f"],
    ("mgus2", "sex"): ["F", "M"],
}


@pytest.mark.parametrize(("name", "column"), list(FACTORS))
def test_factor_columns_load_with_r_levels(name: str, column: str) -> None:
    pd = pytest.importorskip("pandas")
    pl = pytest.importorskip("polars")
    levels = FACTORS[(name, column)]
    polars_df = data.load_dataset(name, backend="polars")
    assert polars_df.schema[column] == pl.Enum(levels)
    pandas_df = data.load_dataset(name, backend="pandas")
    assert isinstance(pandas_df[column].dtype, pd.CategoricalDtype)
    assert list(pandas_df[column].cat.categories) == levels
    assert not pandas_df[column].cat.ordered
    assert pandas_df[column].astype(str).tolist() == polars_df[column].cast(pl.String).to_list()


@pytest.mark.parametrize("name", sorted(data.available_datasets()))
def test_other_columns_are_numeric_in_both_backends(name: str) -> None:
    pd = pytest.importorskip("pandas")
    pl = pytest.importorskip("polars")
    factors = {column for dataset, column in FACTORS if dataset == name}
    polars_df = data.load_dataset(name, backend="polars")
    pandas_df = data.load_dataset(name, backend="pandas")
    for column in set(polars_df.columns) - factors:
        assert polars_df.schema[column].is_numeric(), f"{name}${column}"
        assert pd.api.types.is_numeric_dtype(pandas_df[column]), f"{name}${column}"
        # Missing values are missing in both, not text.
        assert polars_df[column].null_count() == int(pandas_df[column].isna().sum())
    pd.testing.assert_frame_equal(
        polars_df.with_columns(pl.col(c).cast(pl.String) for c in factors).to_pandas(),
        pandas_df.astype(dict.fromkeys(factors, object)),
        check_dtype=False,
    )


def test_unknown_dataset_raises() -> None:
    with pytest.raises(ValueError, match="Unknown dataset"):
        data.load_dataset("nope")


def test_unknown_backend_raises() -> None:
    with pytest.raises(ValueError, match="Unknown backend"):
        data.load_dataset("lung", backend="numpy")
