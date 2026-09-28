"""Data ingest for `Surv` responses: column resolution and event encoding.

The `Surv` constructors accept either values (a series, NumPy array, or sequence) or column names
resolved against a `data=` frame. Event encodings are declared with values (`event_value=2`,
`censor_value=0`, `states={"pcm": 1, "death": 2}`) rather than with DataFrame expressions, so the
same call works unchanged on every backend. Comparisons happen here, in NumPy, after the columns
have been pulled out through Narwhals.

`data=` may be any Narwhals-compatible frame, eager (pandas, Polars, PyArrow) or lazy (Polars
`LazyFrame`, DuckDB, Ibis, ...), or a plain mapping of column names to values. Only the referenced
columns are selected, and lazy frames are collected once.
"""

from __future__ import annotations

import difflib
import math
import re
import warnings
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import narwhals as nw  # pyright: ignore[reportMissingImports]  # installed + typed; pyright quirk
import numpy as np
import numpy.typing as npt

__all__ = ["Duration", "duration"]

Array = npt.NDArray[Any]

# How many distinct values to show in error messages before truncating.
_MAX_SHOWN = 8


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
    time unit. When passed to a `Surv` or `Outcome` constructor as the `time` (or `start`
    / `stop`) argument, Greenwood computes the elapsed time automatically when the data
    is read.

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
    and use it as the time argument in a `Surv` response:

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
    y = gw.Surv.right(time=d, event="event", data=df)
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
    difference when it reads the data. Pass the result as the `time` (or `start` / `stop`) argument
    of a `Surv` or `Outcome` constructor.

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
    gw.Surv.right(
        time=gw.duration(start="enroll", end="exit", unit="years"),
        event="outcome",
        data=study,
        event_value="died",
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


# -- event encoding -----------------------------------------------------------


def encode_event(
    values: Any,
    n: int,
    *,
    event_value: Any = None,
    censor_value: Any = None,
) -> Array:
    """Encode an event column as an int status array (`0` = censored, `1` = event).

    With `event_value=`, rows matching any listed value are events and every other row is censored.
    With `censor_value=`, it is the other way around. With neither, the column must already be
    boolean or `0`/`1`. R's `1`/`2` coding is never guessed. It gets a targeted error that names the
    fix.
    """
    if event_value is not None and censor_value is not None:
        raise ValueError(
            "Pass `event_value` or `censor_value`, not both. Every value not listed is assigned "
            "to the other outcome."
        )
    if values is None:
        if event_value is not None or censor_value is not None:
            raise ValueError("`event_value` and `censor_value` need an `event` column to encode.")
        return np.ones(n, dtype=np.int64)

    arr = as_1d(values)
    _check_no_missing(arr, "event")

    if event_value is None and censor_value is None:
        return _strict_indicator(arr)

    listed = event_value if event_value is not None else censor_value
    which = "event_value" if event_value is not None else "censor_value"
    targets = _value_list(listed)
    _check_comparable(arr, targets, which)
    hit = _isin(arr, targets)
    if not hit.any():
        warnings.warn(
            f"No rows of the event column match `{which}={listed!r}`. "
            f"Values present: {_show_values(arr)}.",
            UserWarning,
            stacklevel=3,
        )
    status = hit if event_value is not None else ~hit
    return status.astype(np.int64)


def encode_states(
    values: Any,
    states: Any,
    *,
    censor_value: Any = None,
) -> tuple[Array, tuple[str, ...]]:
    """Encode a multi-state event column as int codes plus a tuple of state labels.

    `states` is either a sequence of labels, in which case the column must already hold integer
    codes `0..len(states)`, or a mapping of label to the column value (or values) marking that
    state. With a mapping, rows matching `censor_value` (default `0`) are censored and any other
    value raises, so a typo in the mapping cannot silently become censoring.
    """
    arr = as_1d(values)
    _check_no_missing(arr, "event")

    if not isinstance(states, Mapping):
        if censor_value is not None:
            raise ValueError(
                "`censor_value` only applies when `states` is a mapping of label to value. "
                "With a sequence of labels, the event column must already hold codes "
                "0 (censored), 1, 2, ..."
            )
        seq_labels = tuple(str(s) for s in states)  # pyright: ignore[reportUnknownVariableType, reportUnknownArgumentType]
        codes = np.asarray(arr, dtype=float)
        if np.any(codes != np.round(codes)):
            raise ValueError(
                "With a sequence of `states`, the event column must hold integer codes. To map "
                'other values, pass a mapping such as `states={"relapse": "rel", "death": "dth"}`.'
            )
        return codes.astype(np.int64), seq_labels

    mapping: Mapping[Any, Any] = states  # pyright: ignore[reportUnknownVariableType]
    if not mapping:
        raise ValueError("`states` must name at least one state.")
    if censor_value is not None:
        censor_targets = _value_list(censor_value)
        _check_comparable(arr, censor_targets, "censor_value")
    else:
        # The default censor code `0` only makes sense for a numeric column.
        censor_targets = [0] if _is_numeric_column(arr) else []

    status = np.zeros(arr.shape[0], dtype=np.int64)
    claimed = _isin(arr, censor_targets)

    labels: list[str] = []
    for code, (label, value) in enumerate(mapping.items(), start=1):
        targets = _value_list(value)
        _check_comparable(arr, targets, f"states[{label!r}]")
        hit = _isin(arr, targets)
        overlap = hit & claimed
        if overlap.any():
            raise ValueError(
                f"Value {_first(arr[overlap])!r} is assigned to state {label!r} and also to "
                "another state or to `censor_value`. Each value must map to one outcome."
            )
        status[hit] = code
        claimed |= hit
        labels.append(str(label))

    if not claimed.all():
        stray = arr[~claimed]
        raise ValueError(
            f"Event values {_show_values(stray)} are not assigned to any state or to censoring. "
            "Add them to `states`, or list them in `censor_value=` if they mean censored."
        )
    return status, tuple(labels)


def _strict_indicator(arr: Array) -> Array:
    """Accept only a boolean or `0`/`1` indicator, with a helpful error otherwise."""
    if arr.dtype.kind not in "iufb" and not all(
        isinstance(v, (bool, np.bool_)) for v in arr.tolist()
    ):
        raise ValueError(
            f"The event column holds non-numeric values {_show_values(arr)}. Say which value "
            "marks an event with `event_value=` (e.g., `event_value='died'`), or which marks "
            "censoring with `censor_value=`."
        )
    num = np.asarray(arr, dtype=float)
    uniq = set(np.unique(num).tolist())
    if uniq <= {0.0, 1.0}:
        return num.astype(np.int64)
    if uniq == {1.0, 2.0}:
        raise ValueError(
            "The event column must be boolean or 0/1, but it holds [1, 2]. This looks like R's "
            "1/2 coding (1 = censored, 2 = event, as in `survival::lung`). Pass `event_value=2` "
            "to use it."
        )
    raise ValueError(
        f"The event column must be boolean or 0/1. Got values {_show_values(arr)}. Say which "
        "values mark an event with `event_value=` (e.g., `event_value=[1, 2]`), or which mark "
        "censoring with `censor_value=` (e.g., `censor_value=0`)."
    )


# -- small helpers ------------------------------------------------------------


def _value_list(value: Any) -> list[Any]:
    """Normalize a scalar or collection of target values to a list."""
    if isinstance(value, np.ndarray):
        return list(value.tolist())  # pyright: ignore[reportUnknownArgumentType]
    if isinstance(value, (list, tuple, set, frozenset)):
        items: list[Any] = list(value)  # pyright: ignore[reportUnknownArgumentType]
        if not items:
            raise ValueError("An empty collection of values was given for an event encoding.")
        return items
    return [value]


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


def _check_no_missing(arr: Array, name: str) -> None:
    if arr.dtype.kind == "f":
        n_missing = int((~np.isfinite(arr)).sum())
    elif arr.dtype.kind in "iub":
        n_missing = 0
    else:
        n_missing = sum(_is_missing(v) for v in arr.tolist())
    if n_missing:
        raise ValueError(
            f"The {name} column contains {n_missing} missing/non-finite value(s). Drop or impute "
            "those rows before building the response."
        )


def _is_numeric_column(arr: Array) -> bool:
    if arr.dtype.kind in "iufb":
        return True
    return all(
        isinstance(v, (bool, int, float, np.integer, np.floating, np.bool_)) for v in arr.tolist()
    )


def _comparable(arr: Array, targets: list[Any]) -> bool:
    numeric_targets = all(
        isinstance(t, (bool, int, float, np.integer, np.floating, np.bool_)) for t in targets
    )
    return numeric_targets == _is_numeric_column(arr)


def _check_comparable(arr: Array, targets: list[Any], which: str) -> None:
    """Catch the common mismatch of string targets against a numeric column, or vice versa."""
    if _comparable(arr, targets):
        return
    kind = "numeric" if _is_numeric_column(arr) else "non-numeric"
    raise TypeError(
        f"`{which}` gives {targets!r}, but the event column is {kind} with values "
        f"{_show_values(arr)}. The values must be of the same kind as the column."
    )


def _isin(arr: Array, targets: list[Any]) -> Array:
    if arr.dtype.kind in "iufb":
        return np.isin(arr, np.asarray(targets))
    target_set = set(targets)
    return np.fromiter((v in target_set for v in arr.tolist()), dtype=bool, count=arr.shape[0])


def _distinct(arr: Array) -> list[Any]:
    seen: dict[Any, None] = dict.fromkeys(arr.tolist())
    try:
        return sorted(seen)
    except TypeError:
        return list(seen)


def _show_values(arr: Array) -> str:
    vals = _distinct(arr)
    shown = ", ".join(repr(v) for v in vals[:_MAX_SHOWN])
    more = f", ... ({len(vals)} distinct)" if len(vals) > _MAX_SHOWN else ""
    return f"[{shown}{more}]"


def _first(arr: Array) -> Any:
    return arr.tolist()[0]
