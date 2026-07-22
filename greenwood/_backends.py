"""DataFrame-agnostic output helpers.

Greenwood never assumes a single DataFrame library is installed. Result objects and the
functions that return tabular data build a plain ``dict`` of columns and hand it to
`to_dataframe`, which materializes it in the requested backend (or auto-detects one). This
keeps Pandas from being an unconditional import: a Polars-only user can get Polars output,
and nothing pulls in Pandas unless it is actually asked for.
"""

from __future__ import annotations

from typing import Any

__all__ = ["to_dataframe"]

VALID_FORMATS = ("pandas", "polars", "pyarrow")


def to_dataframe(data: dict[str, Any], *, format: str | None = None) -> Any:
    """Materialize a dict of columns as a DataFrame in the requested backend.

    This is the single gateway through which all Greenwood functions return tabular output.
    It keeps DataFrame library imports lazy so that users who only have Polars installed
    never trigger a Pandas import (and vice versa).

    Parameters
    ----------
    data
        Mapping of column names to arrays or lists. All values must be the same length.
    format
        Output format: `None` (default), `"pandas"`, `"polars"`, or `"pyarrow"`.

        - `None` (default): auto-detect, trying Polars first, then Pandas, then PyArrow.
          Raises `ImportError` if none is installed.
        - `"pandas"`: return a `pandas.DataFrame`.
        - `"polars"`: return a `polars.DataFrame`.
        - `"pyarrow"`: return a `pyarrow.Table`.

    Returns
    -------
    pandas.DataFrame, polars.DataFrame, or pyarrow.Table
        The data in the requested (or auto-detected) format.

    Raises
    ------
    ImportError
        If the requested backend is not installed, or if `format=None` and no backend is
        available at all.
    ValueError
        If `format` is not one of the recognised strings.

    Examples
    --------
    Create a small table and materialise it as a Polars DataFrame:

    ```{python}
    from greenwood._backends import to_dataframe

    to_dataframe({"name": ["Alice", "Bob"], "age": [30, 25]}, format="polars")
    ```
    """
    if format is None:
        # Prefer Polars (most efficient), then Pandas, then PyArrow.
        try:
            import polars as pl  # pyright: ignore[reportMissingImports]

            return pl.DataFrame(data)
        except ImportError:
            pass
        try:
            import pandas as pd

            return pd.DataFrame(data)
        except ImportError:
            pass
        try:
            import pyarrow as pa
        except ImportError as e:
            raise ImportError(
                "No DataFrame library found. Install one of: pandas, polars, or pyarrow"
            ) from e
        return pa.table(data)

    if format == "pandas":
        try:
            import pandas as pd
        except ImportError as e:
            raise ImportError(
                "pandas is required for format='pandas'. Install it with: pip install pandas"
            ) from e
        return pd.DataFrame(data)

    if format == "polars":
        try:
            import polars as pl  # pyright: ignore[reportMissingImports]
        except ImportError as e:
            raise ImportError(
                "polars is required for format='polars'. Install it with: pip install polars"
            ) from e
        return pl.DataFrame(data)

    if format == "pyarrow":
        try:
            import pyarrow as pa
        except ImportError as e:
            raise ImportError(
                "pyarrow is required for format='pyarrow'. Install it with: pip install pyarrow"
            ) from e
        return pa.table(data)

    raise ValueError(
        f"Unknown format {format!r}; use 'pandas', 'polars', 'pyarrow', or None for auto-detect"
    )
