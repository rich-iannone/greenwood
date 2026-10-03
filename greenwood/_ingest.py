"""Data ingest for `Outcome` responses: column resolution, durations, and array helpers.

An `Outcome` names its columns, and those names are resolved against a frame when the outcome
is bound to data. Event codings are written as expressions on column names (`"status == 2"`,
`"factor(cause, c(0, 1, 2), c('censor', 'pcm', 'death'))"`) rather than with DataFrame
expressions, so the same outcome works unchanged on every backend. The columns are pulled out
through Narwhals and compared in NumPy.

The data may be any Narwhals-compatible frame, eager (pandas, Polars, PyArrow) or lazy (Polars
`LazyFrame`, DuckDB, Ibis, ...), or a plain mapping of column names to values. Only the referenced
columns are selected, and lazy frames are collected once.
"""

from __future__ import annotations

import difflib
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import narwhals as nw  # pyright: ignore[reportMissingImports]  # installed + typed; pyright quirk
import numpy as np
import numpy.typing as npt

__all__ = ["Duration", "duration"]

Array = npt.NDArray[Any]


def as_1d(x: Any) -> Array:
    """Coerce a Narwhals series, NumPy array, or sequence to a 1-D array, keeping its dtype."""
    if x is None:
        raise ValueError("Expected an array-like, got None.")
    if isinstance(x, (np.ndarray, list, tuple)):
        arr = np.asarray(x)
    else:
        # A Narwhals-native series (pandas, Polars, ...) or anything else array-like.
        try:
            arr = np.asarray(nw.from_native(x, series_only=True).to_numpy())
        except TypeError:
            arr = np.asarray(x)
    if arr.ndim != 1:
        raise ValueError(f"Expected a 1-D array-like, got shape {arr.shape}.")
    return arr


def to_1d_array(x: Any, *, dtype: Any = float) -> Array:
    """Coerce a Narwhals series, NumPy array, or sequence to a 1-D NumPy array of `dtype`."""
    return np.asarray(as_1d(x), dtype=dtype)


# -- durations ----------------------------------------------------------------

# Seconds per unit. Months and years use the average Gregorian lengths (30.4375 and 365.25 days).
_UNIT_SECONDS: dict[str, float] = {
    "seconds": 1.0,
    "minutes": 60.0,
    "hours": 3600.0,
    "days": 86400.0,
    "weeks": 7 * 86400.0,
    "months": 30.4375 * 86400.0,
    "years": 365.25 * 86400.0,
}


@dataclass(frozen=True)
class Duration:
    """A time argument computed as the elapsed time between two date or datetime columns.

    A `Duration` is a frozen specification that names two date or datetime columns and a
    time unit. When passed to an `Outcome` constructor as the `time` (or `time2`) argument,
    Greenwood computes the elapsed time automatically when the data is read. Inside a formula,
    write the same thing as `duration(start, end, unit="days")`.

    Create a `Duration` with the `duration()` helper rather than constructing it directly.

    Parameters
    ----------
    start
        Column name holding the origin date or datetime (e.g., enrollment date).
    end
        Column name holding the exit date or datetime (e.g., date of event or last contact).
    unit
        Time unit for the elapsed duration: `"days"` (default), `"weeks"`, `"months"`,
        `"years"`, `"hours"`, `"minutes"`, or `"seconds"`. Months and years use average
        Gregorian lengths (30.4375 and 365.25 days).

    Examples
    --------
    Create a `Duration` that computes follow-up time in days between two date columns,
    and use it as the time argument of an `Outcome`:

    ```{python}
    import greenwood as gw
    import polars as pl

    df = pl.DataFrame({
        "enroll": ["2020-01-01", "2020-03-15", "2020-06-01"],
        "last_contact": ["2021-06-15", "2020-12-01", "2022-01-10"],
        "event": [1, 0, 1],
    }).with_columns(
        pl.col("enroll").str.to_date(),
        pl.col("last_contact").str.to_date(),
    )

    d = gw.duration(start="enroll", end="last_contact", unit="days")
    d
    ```

    ```{python}
    y = gw.Outcome.surv(time=d, event="event").bind(df)
    y
    ```
    """

    start: str
    end: str
    unit: str = "days"

    def __post_init__(self) -> None:
        if not isinstance(self.start, str) or not isinstance(self.end, str):  # pyright: ignore[reportUnnecessaryIsInstance]  # runtime guard
            raise TypeError("`duration()` takes the names of two date or datetime columns.")
        if self.unit not in _UNIT_SECONDS:
            units = ", ".join(repr(u) for u in _UNIT_SECONDS)
            raise ValueError(f"Unknown unit {self.unit!r}. Use one of {units}.")

    @property
    def columns(self) -> tuple[str, str]:
        """The two columns the duration reads."""
        return (self.start, self.end)

    def compute(self, start: Any, end: Any) -> Array:
        """The elapsed time from `start` to `end` in `unit` (`NaN` where either is missing)."""
        t0 = _as_datetime(as_1d(start), self.start)
        t1 = _as_datetime(as_1d(end), self.end)
        elapsed = (t1 - t0) / np.timedelta64(1, "s") / _UNIT_SECONDS[self.unit]
        elapsed = np.asarray(elapsed, dtype=float)
        negative = np.isfinite(elapsed) & (elapsed < 0)
        if negative.any():
            raise ValueError(
                f"{int(negative.sum())} row(s) have `{self.end}` earlier than `{self.start}`, "
                "which gives a negative duration."
            )
        return elapsed

    def __repr__(self) -> str:
        return f"duration(start={self.start!r}, end={self.end!r}, unit={self.unit!r})"


def duration(start: str, end: str, *, unit: str = "days") -> Duration:
    """Describe a time as the elapsed time between two date or datetime columns.

    Study exports often record calendar dates (enrollment, last contact, death) rather than
    durations. `duration()` names the two columns and the time unit, and Greenwood computes the
    difference when it reads the data. Pass the result as the `time` (or `time2`) argument of an
    `Outcome` constructor. Inside a formula, write `duration(enroll, exit, unit="years")` instead.

    Parameters
    ----------
    start
        The column holding the origin, such as an enrollment date.
    end
        The column holding the exit, such as the date of the event or of last contact.
    unit
        The time unit: `"days"` (the default), `"weeks"`, `"months"`, `"years"`, `"hours"`,
        `"minutes"`, or `"seconds"`. Months and years use average lengths (30.4375 and 365.25 days).

    Returns
    -------
    Duration
        A frozen `Duration` spec holding the two column names and the unit. It is computed when the
        data is read, and it prints as the call that made it, such as
        `duration(start='enroll', end='exit', unit='days')`. The `Duration` class is exported
        for type annotations.

    Examples
    --------
    Compute follow-up time in years from two date columns:

    ```{python}
    import greenwood as gw
    import polars as pl
    from datetime import date

    study = pl.DataFrame({
        "enroll": [date(2021, 3, 1), date(2021, 3, 15), date(2021, 4, 1)],
        "exit": [date(2022, 1, 10), date(2021, 9, 22), date(2023, 1, 1)],
        "outcome": ["died", "censored", "died"],
    })

    # The time is the gap between the two dates, in years
    gw.Outcome.surv(
        time=gw.duration(start="enroll", end="exit", unit="years"),
        event="outcome == 'died'",
    ).bind(study)
    ```

    The same response written as a formula, fitted with `data=`:

    ```{python}
    gw.KaplanMeier().fit(
        "Surv(duration(enroll, exit, unit='years'), outcome == 'died')", data=study
    )
    ```
    """
    return Duration(start, end, unit)


def _as_datetime(arr: Array, name: str) -> Array:
    if arr.dtype.kind == "M":
        return arr
    if arr.dtype.kind in "iufbm":
        raise TypeError(
            f"Column {name!r} must hold dates or datetimes for `duration()`, not numbers. If it "
            "already holds elapsed times, pass it directly as the time column."
        )
    try:
        return np.asarray(arr, dtype="datetime64[us]")
    except (TypeError, ValueError) as error:
        raise TypeError(f"Column {name!r} could not be read as dates or datetimes.") from error


# -- column resolution --------------------------------------------------------


def resolve_columns(data: Any, arguments: dict[str, Any]) -> dict[str, Any]:
    """Replace column-name arguments with the matching columns of `data`.

    `arguments=` maps each constructor argument name to what the caller passed. String values are
    column names and are looked up in `data`, and a `Duration` is computed from its two columns.
    Anything else (values, or `None`) passes through unchanged, so names and values can be mixed in
    one call.
    """
    named = {arg: v for arg, v in arguments.items() if isinstance(v, (str, Duration))}
    if not named:
        return dict(arguments)
    if data is None:
        arg, value = next(iter(named.items()))
        what = "names columns" if isinstance(value, Duration) else "is a column name"
        raise ValueError(
            f"`{arg}={value!r}` {what}, but no `data=` was given. Pass the frame that holds it "
            "(e.g., `data=df`) or pass the values directly."
        )
    wanted: list[str] = []
    for value in named.values():
        wanted += list(value.columns) if isinstance(value, Duration) else [value]
    columns = _fetch_columns(data, list(dict.fromkeys(wanted)))

    def resolve(value: Any) -> Any:
        if isinstance(value, Duration):
            return value.compute(columns[value.start], columns[value.end])
        if isinstance(value, str):
            return columns[value]
        return value

    return {arg: resolve(value) for arg, value in arguments.items()}


def _fetch_columns(data: Any, names: list[str]) -> dict[str, Array]:
    """Pull the named columns out of `data` as 1-D NumPy arrays."""
    if isinstance(data, Mapping):
        mapping: Mapping[Any, Any] = data  # pyright: ignore[reportUnknownVariableType]
        available = [str(k) for k in mapping]
        _check_names(names, available)
        return {name: as_1d(mapping[name]) for name in names}

    try:
        frame: Any = nw.from_native(data)
    except TypeError as error:
        raise TypeError(
            "`data` must be a DataFrame (pandas, Polars, PyArrow, DuckDB, ...) or a mapping of "
            f"column names to values. Got {type(data).__name__}."
        ) from error
    if not isinstance(frame, (nw.DataFrame, nw.LazyFrame)):
        raise TypeError(f"`data` must be a DataFrame, not a {type(data).__name__}.")

    available = [str(c) for c in frame.collect_schema().names()]
    _check_names(names, available)
    selected = select_columns(frame, names)
    return {name: as_1d(selected.get_column(name).to_numpy()) for name in names}


_PLAIN_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def select_columns(frame: Any, names: list[str] | None) -> Any:
    """Select `names` from a Narwhals frame and return an eager Narwhals `DataFrame`.

    With `names=None`, every column is kept. For a lazy frame the selection is pushed down to the
    backend before collecting, so only the needed columns are read. SQL backends such as DuckDB
    parse a dotted name like `ph.ecog` as a table-qualified reference, so when any name is not a
    plain identifier the frame is collected first and the selection happens eagerly.
    """
    if isinstance(frame, nw.LazyFrame):
        if names is None:
            return frame.collect()
        if all(_PLAIN_NAME.fullmatch(n) for n in names):
            return frame.select(names).collect()
        frame = frame.collect()
    return frame if names is None else frame.select(names)


def _check_names(names: list[str], available: list[str]) -> None:
    missing = [name for name in names if name not in available]
    if not missing:
        return
    name = missing[0]
    close = difflib.get_close_matches(name, available, n=1)
    hint = f" Did you mean {close[0]!r}?" if close else ""
    raise KeyError(f"Column {name!r} was not found in `data`.{hint}")


# -- small helpers ------------------------------------------------------------


def missing_mask(arr: Array) -> Array:
    """Return a boolean mask of missing entries (NaN, None, NaT, or `pandas.NA`)."""
    if arr.dtype.kind == "f":
        return np.isnan(arr)
    if arr.dtype.kind in "Mm":
        return np.isnat(arr)
    if arr.dtype.kind in "iub":
        return np.zeros(arr.shape[0], dtype=bool)
    return np.fromiter((_is_missing(v) for v in arr.tolist()), dtype=bool, count=arr.shape[0])


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float):
        return math.isnan(value)
    # pandas.NA and pandas.NaT, detected without importing pandas.
    return type(value).__name__ in {"NAType", "NaTType"}
