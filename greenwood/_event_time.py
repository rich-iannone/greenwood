"""`EventTime`: a vector of potentially censored event times, ported from R's etd package.

This is a faithful port of Max Kuhn's etd (https://github.com/topepo/etd). Each element is an
event time with a single-letter status:

- `"e"`: an exact (observed) event time.
- `"r"`: right-censored. The event happens after `time`.
- `"l"`: left-censored. The event happened before `time`.
- `"i"`: interval-censored. The event happened between `time` and `time_max`.

Names, arguments, validation rules and their order, missing-value handling, and printed
formatting all follow etd. Where Python can't match R literally it matches the meaning instead
(0-based error locations, `nan` and `None` for missing values). The contract and every adaptation
are recorded in `spec/event_time.md`, and `tests/test_event_time_conformance.py` replays cases
generated from etd itself.
"""

from __future__ import annotations

import math
import numbers
from collections.abc import Iterable, Sequence
from functools import singledispatch
from typing import Any

import narwhals as nw  # pyright: ignore[reportMissingImports]  # installed + typed; pyright quirk
import numpy as np
import numpy.typing as npt
from typing_extensions import Self

from ._backends import to_dataframe
from ._rformat import format_real

__all__ = ["EventTime", "event_time", "new_event_time", "extract_time", "extract_status"]

FloatArray = npt.NDArray[np.float64]
BoolArray = npt.NDArray[np.bool_]
ObjectArray = npt.NDArray[np.object_]

STATUS_VALUES = ("e", "r", "l", "i")

# Printed after each value: nothing extra for events, "+" for right, "-" for left. Intervals are
# bracketed instead.
_SUFFIX = {"e": " ", "r": "+", "l": "-", "i": ""}

# How many locations to list in an error message before summarizing the rest.
_MAX_LOCATIONS = 20

# Line width for `repr()`, matching R's default `width` option.
_REPR_WIDTH = 80


# -- errors ---------------------------------------------------------------------------------


class _EventTimeTypeError(TypeError):
    """A type problem in event-time input. Carries a stable `check` id and `locations`."""

    def __init__(self, message: str, *, check: str, locations: Sequence[int] = ()) -> None:
        super().__init__(message)
        self.check = check
        self.locations = np.asarray(locations, dtype=np.int64)


class _EventTimeValueError(ValueError):
    """A value problem in event-time input. Carries a stable `check` id and `locations`."""

    def __init__(self, message: str, *, check: str, locations: Sequence[int] = ()) -> None:
        super().__init__(message)
        self.check = check
        self.locations = np.asarray(locations, dtype=np.int64)


def _describe_locations(locations: Sequence[int]) -> str:
    """Write 0-based locations the way etd (via cli) lists them: "1, 2, and 3"."""
    shown = [str(loc) for loc in locations[:_MAX_LOCATIONS]]
    extra = len(locations) - len(shown)
    if extra:
        return f"locations {', '.join(shown)}, and {extra} more"
    if len(shown) == 1:
        return f"location {shown[0]}"
    if len(shown) == 2:
        return f"locations {shown[0]} and {shown[1]}"
    return f"locations {', '.join(shown[:-1])}, and {shown[-1]}"


def _check_locations(mask: BoolArray, check: str, message: str) -> None:
    """Raise a located `ValueError` if any element of `mask` is true."""
    locations = [int(i) for i in np.flatnonzero(mask)]
    if locations:
        raise _EventTimeValueError(
            f"{message} Problem at {_describe_locations(locations)}.",
            check=check,
            locations=locations,
        )


def _type_name(value: Any) -> str:
    return f"<{type(value).__name__}>"


# -- input coercion -------------------------------------------------------------------------


def _is_scalar(x: Any) -> bool:
    return isinstance(x, (str, bytes, numbers.Number, np.generic))


def _to_values(x: Any) -> tuple[Sequence[Any] | np.ndarray[Any, Any], bool]:
    """Turn a list, tuple, NumPy array, or DataFrame series into something iterable.

    Returns the values and whether they came from a categorical series, which callers reject the
    way etd rejects R factors. A bare scalar becomes a length-1 sequence, since R has no scalars
    and `event_time(3, "e")` is the usual way to write one value.
    """
    if _is_scalar(x):
        return [x], False
    if isinstance(x, (list, tuple, np.ndarray)):
        return x, False  # pyright: ignore[reportUnknownVariableType]
    try:
        series = nw.from_native(x, series_only=True)
    except TypeError:
        return np.asarray(x), False
    categorical = series.dtype in (nw.Categorical, nw.Enum)
    # Backends spell missing values differently (pandas uses NaN even for strings), so normalize
    # every null to None.
    nulls = series.is_null().to_list()
    values = [None if null else v for v, null in zip(series.to_list(), nulls, strict=True)]
    return values, categorical


def _cast_double(x: Any, arg: str) -> FloatArray:
    """Cast to float64 the way `vctrs::vec_cast(x, double())` does.

    Numbers and booleans are accepted, `None` and NaN are missing, and anything else (strings,
    nested lists, dates) is an error.
    """
    if x is None:
        return np.empty(0, dtype=np.float64)
    if isinstance(x, np.ndarray):
        arr: np.ndarray[Any, Any] = x  # pyright: ignore[reportUnknownVariableType]
        if arr.ndim != 1:
            raise _EventTimeTypeError(
                f"Can't convert `{arg}` with shape {arr.shape} to a 1-D <double>.",
                check=f"{arg}_not_numeric",
            )
        if arr.dtype.kind in "fiub":
            return arr.astype(np.float64)
        if arr.dtype.kind != "O":
            raise _EventTimeTypeError(
                f"Can't convert `{arg}` <{arr.dtype}> to <double>.", check=f"{arg}_not_numeric"
            )
        values: Iterable[Any] = arr.tolist()
    else:
        seq, categorical = _to_values(x)
        if categorical:
            raise _EventTimeTypeError(
                f"Can't convert `{arg}` <categorical> to <double>.", check=f"{arg}_not_numeric"
            )
        if isinstance(seq, np.ndarray):
            return _cast_double(seq, arg)
        values = seq

    out: list[float] = []
    for v in values:
        if v is None:
            out.append(math.nan)
        elif isinstance(v, (bool, np.bool_, numbers.Real)) and not isinstance(v, np.datetime64):
            out.append(float(v))  # pyright: ignore[reportArgumentType]
        else:
            raise _EventTimeTypeError(
                f"Can't convert `{arg}` {_type_name(v)} to <double>.", check=f"{arg}_not_numeric"
            )
    return np.asarray(out, dtype=np.float64)


def _as_character(x: Any, arg: str) -> ObjectArray:
    """Require a character vector: strings, with `None` for missing values.

    Categorical input is rejected, as etd rejects R factors.
    """
    check = "status_not_character"
    if x is None or (_is_scalar(x) and not isinstance(x, str)):
        raise _EventTimeTypeError(
            f"`{arg}` must be a character vector, not {_type_name(x)}.", check=check
        )
    seq, categorical = _to_values(x)
    if categorical:
        raise _EventTimeTypeError(
            f"`{arg}` must be a character vector, not a categorical.", check=check
        )
    if isinstance(seq, np.ndarray):
        if seq.ndim != 1:
            raise _EventTimeTypeError(
                f"`{arg}` must be a 1-D character vector, not shape {seq.shape}.", check=check
            )
        if seq.dtype.kind not in "UO":
            raise _EventTimeTypeError(
                f"`{arg}` must be a character vector, not <{seq.dtype}>.", check=check
            )
        values: list[Any] = seq.tolist()
    else:
        values = list(seq)
    for v in values:
        if v is not None and not isinstance(v, str):
            raise _EventTimeTypeError(
                f"`{arg}` must be a character vector, not one containing {_type_name(v)}.",
                check=check,
            )
    out = np.empty(len(values), dtype=object)
    out[:] = values
    return out


# -- the vector -----------------------------------------------------------------------------


class EventTime:
    """A vector of event times that may be exact, or right-, left-, or interval-censored.

    `EventTime` is the Python counterpart of etd's `event_time` vector. Create one with
    `event_time()`, which validates its input, or with `new_event_time()`, which only checks
    types. Direct instantiation is for internal use.

    Each element has a time and a single-letter status: `"e"` (exact event), `"r"`
    (right-censored), `"l"` (left-censored), or `"i"` (interval-censored, with an upper bound in
    `time_max`). Elements print the way etd prints them: `7` for an event, `5+` for right
    censoring, `3-` for left censoring, and `[2, 4]` for an interval.

    The vector supports `len()`, indexing with an integer, a slice, an integer array, or a boolean
    mask (always returning an `EventTime`), and `EventTime.concat()` to combine vectors. Use
    `extract_time()` and `extract_status()` to get the components back, and `to_frame()` for a
    table with `time`, `status`, and `time_max` columns.

    Examples
    --------
    ```{python}
    import greenwood as gw

    x = gw.event_time(
        time=[7, 5, 3, 2, None],
        status=["e", "r", "l", "i", None],
        time_max=[None, None, None, 4, None],
    )
    x
    ```

    Indexing returns another `EventTime`:

    ```{python}
    x[1:4]
    ```
    """

    __slots__ = ("_time", "_time_max", "_paired", "_status")

    _time: FloatArray
    _time_max: FloatArray
    _paired: BoolArray
    _status: ObjectArray

    def __init__(
        self,
        *,
        time: FloatArray,
        time_max: FloatArray,
        paired: BoolArray,
        status: ObjectArray,
    ) -> None:
        # Columnar storage of etd's list field: `time` is each element's first number, and
        # `time_max` its second number where `paired` is true (an etd element of length 2).
        self._time = time
        self._time_max = time_max
        self._paired = paired
        self._status = status

    # -- vector protocol ------------------------------------------------------------------

    def __len__(self) -> int:
        return int(self._time.shape[0])

    def __getitem__(self, key: Any) -> EventTime:
        if isinstance(key, (int, np.integer)):
            n = len(self)
            i = int(key)  # pyright: ignore[reportUnknownArgumentType]
            if not -n <= i < n:
                raise IndexError(f"Index {i} is out of bounds for an EventTime of length {n}.")
            index: Any = [i]
        elif isinstance(key, slice):
            index = key
        else:
            index = np.asarray(key)
            if index.dtype.kind not in "iub" or index.ndim != 1:
                raise TypeError(
                    "EventTime indices must be an integer, a slice, an integer array, or a "
                    "boolean mask."
                )
        return EventTime(
            time=self._time[index],
            time_max=self._time_max[index],
            paired=self._paired[index],
            status=self._status[index],
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, EventTime):
            return NotImplemented
        return (
            np.array_equal(self._time, other._time, equal_nan=True)
            and np.array_equal(self._time_max, other._time_max, equal_nan=True)
            and np.array_equal(self._paired, other._paired)
            and list(self._status) == list(other._status)
        )

    __hash__ = None  # pyright: ignore[reportAssignmentType]  # mutable-free but array-valued

    @classmethod
    def concat(cls, items: Iterable[EventTime]) -> Self:
        """Combine event-time vectors end to end, as `c()` does in R.

        Parameters
        ----------
        items
            The `EventTime` vectors to combine, in order.

        Returns
        -------
        EventTime
            A vector holding every element of `items`.

        Examples
        --------
        ```{python}
        import greenwood as gw

        a = gw.event_time(time=[7, 5], status=["e", "r"])
        b = gw.event_time(time=[1], status=["i"], time_max=[3])
        gw.EventTime.concat([a, b])
        ```
        """
        parts = list(items)
        for part in parts:
            if not isinstance(part, EventTime):  # pyright: ignore[reportUnnecessaryIsInstance]
                raise TypeError(f"Can't combine an EventTime with {_type_name(part)}.")
        if not parts:
            return cls(
                time=np.empty(0, dtype=np.float64),
                time_max=np.empty(0, dtype=np.float64),
                paired=np.empty(0, dtype=np.bool_),
                status=np.empty(0, dtype=object),
            )
        return cls(
            time=np.concatenate([p._time for p in parts]),
            time_max=np.concatenate([p._time_max for p in parts]),
            paired=np.concatenate([p._paired for p in parts]),
            status=np.concatenate([p._status for p in parts]),
        )

    # -- observations -----------------------------------------------------------------------

    def is_na(self) -> BoolArray:
        """Return which elements are missing.

        An element is missing when any of its times is missing (`nan`), matching etd's
        `is.na()`.

        Returns
        -------
        numpy.ndarray
            A boolean array, one entry per element.

        Examples
        --------
        ```{python}
        import greenwood as gw

        x = gw.event_time(time=[7, None, 2], status=["e", None, "i"], time_max=[None, None, 4])
        x.is_na()
        ```
        """
        return np.isnan(self._time) | (self._paired & np.isnan(self._time_max))

    def format(self) -> list[str | None]:
        """Format each element as etd does, with `None` for missing elements.

        Every number in the vector is formatted together, with R's rules: at most 7 significant
        digits, a shared number of decimal places, and scientific notation when it is narrower.
        Events get a trailing space, right-censored values a `+`, left-censored values a `-`, and
        intervals are written as `[lower, upper]`.

        Returns
        -------
        list of str or None
            One string per element, or `None` where the element is missing.

        Examples
        --------
        ```{python}
        import greenwood as gw

        x = gw.event_time(time=[7, 5.5, 3], status=["e", "r", "i"], time_max=[None, None, 4])
        x.format()
        ```
        """
        flat: list[float] = []
        for t, upper, paired in zip(self._time, self._time_max, self._paired, strict=True):
            flat.append(float(t))
            if paired:
                flat.append(float(upper))
        numbers_ = format_real(flat)

        out: list[str | None] = []
        missing = self.is_na()
        k = 0
        for i in range(len(self)):
            width = 2 if self._paired[i] else 1
            text = ", ".join(numbers_[k : k + width])
            k += width
            status = self._status[i]
            # R pastes a missing suffix as the literal "NA", which only an unvalidated
            # new_event_time() vector can show.
            text += _SUFFIX.get(status, "NA") if isinstance(status, str) else "NA"
            if status == "i":
                text = f"[{text}]"
            out.append(None if missing[i] else text)
        return out

    def __repr__(self) -> str:
        header = f"<event_time[{len(self)}]>"
        if len(self) == 0:
            return header
        tokens = ["<NA>" if s is None else s for s in self.format()]
        width = max(len(t) for t in tokens)
        per_line = max(1, (_REPR_WIDTH + 1) // (width + 1))
        lines = [header]
        for start in range(0, len(tokens), per_line):
            chunk = tokens[start : start + per_line]
            lines.append(" ".join(t.ljust(width) for t in chunk).rstrip())
        return "\n".join(lines)

    def _lower(self) -> FloatArray:
        return self._time.copy()

    def _upper(self) -> FloatArray:
        # Missing for elements that aren't stored as a pair, as in etd's event_time_upper().
        return np.where(self._paired, self._time_max, np.nan)

    def to_frame(self, *, format: str | None = None) -> Any:
        """Split the vector into a table with `time`, `status`, and `time_max` columns.

        This is the counterpart of etd's `as_tibble()`. `time_max` is always included, and is
        missing for elements that aren't interval-censored. Missing values are nulls.

        Parameters
        ----------
        format
            Output format: `None` (default), `"pandas"`, `"polars"`, or `"pyarrow"`. When `None`,
            a backend is auto-detected (Polars, then Pandas, then PyArrow).

        Returns
        -------
        pandas.DataFrame, polars.DataFrame, or pyarrow.Table
            One row per element.

        Examples
        --------
        ```{python}
        import greenwood as gw

        x = gw.event_time(time=[7, 5, 2], status=["e", "r", "i"], time_max=[None, None, 4])
        x.to_frame(format="polars")
        ```
        """

        def nullable(values: FloatArray) -> list[float | None]:
            return [None if math.isnan(v) else float(v) for v in values]

        return to_dataframe(
            {
                "time": nullable(self._lower()),
                "status": list(self._status),
                "time_max": nullable(self._upper()),
            },
            format=format,
        )


# -- constructors ---------------------------------------------------------------------------


def event_time(time: Any, status: Any, time_max: Any = None) -> EventTime:
    """Create a vector of event times.

    `event_time()` creates a vector of event times that may be exact, or right-, left-, or
    interval-censored. It is a port of etd's `event_time()` and validates its input with the same
    rules, in the same order.

    Parameters
    ----------
    time
        Non-negative event times (numbers or booleans, with `None` or `nan` for missing values).
        For interval-censored values, this is the lower end of the interval. Accepts a list, a
        NumPy array, or a DataFrame series. A single number is treated as a vector of length 1.
    status
        Status codes, the same length as `time`: `"e"` (exact event), `"r"` (right-censored),
        `"l"` (left-censored), or `"i"` (interval-censored). It can be missing (`None`) only where
        `time` is missing. Codes are case-sensitive, and categorical input is rejected.
    time_max
        Upper ends of the intervals, the same length as `time`. Must be missing for values that
        are not interval-censored, and greater than `time` otherwise. The default `None` sets all
        values to missing.

    Returns
    -------
    EventTime
        The event-time vector.

    Raises
    ------
    TypeError
        If `time` or `time_max` isn't numeric, or `status` isn't a character vector.
    ValueError
        If the lengths differ, or a value breaks one of the rules above.

    Notes
    -----
    Errors carry two extra attributes: `check`, a stable identifier for the rule that failed
    (such as `"status_invalid_value"`), and `locations`, a 0-based integer array of the offending
    positions (empty when the rule isn't about particular positions). Only the first failing rule
    is reported, but with every location that breaks it.

    Examples
    --------
    ```{python}
    import greenwood as gw

    x = gw.event_time(
        time=[7, 5, 3, 2, None],
        status=["e", "r", "l", "i", None],
        time_max=[None, None, None, 4, None],
    )
    x
    ```

    Columns of a data frame work directly:

    ```{python}
    import polars as pl

    df = pl.DataFrame({
        "times": [0.14, 0.15, 0.44, 0.76, 1.18],
        "status": ["l", "e", "e", "e", "r"],
    })
    gw.event_time(time=df["times"], status=df["status"])
    ```
    """
    time_a = _cast_double(time, "time")
    n = time_a.shape[0]
    time_max_a = np.full(n, np.nan) if time_max is None else _cast_double(time_max, "time_max")
    status_a = _as_character(status, "status")

    if status_a.shape[0] != n:
        raise _EventTimeValueError(
            f"`status` must have length {n} (the length of `time`), not {status_a.shape[0]}.",
            check="status_length",
        )
    if time_max_a.shape[0] != n:
        raise _EventTimeValueError(
            f"`time_max` must have length {n} (the length of `time`), not {time_max_a.shape[0]}.",
            check="time_max_length",
        )

    present = ~np.isnan(time_a)
    status_missing = np.array([s is None for s in status_a], dtype=np.bool_)
    _check_locations(present & (time_a < 0), "time_negative", "`time` must be non-negative.")
    _check_locations(
        status_missing & present,
        "status_missing",
        "`status` must not be missing when `time` is not missing.",
    )
    invalid = np.array([s is not None and s not in STATUS_VALUES for s in status_a], dtype=np.bool_)
    _check_locations(
        invalid,
        "status_invalid_value",
        '`status` must be one of "e", "r", "l", and "i".',
    )

    is_interval = np.array([s == "i" for s in status_a], dtype=np.bool_)
    has_max = ~np.isnan(time_max_a)
    _check_locations(
        ~is_interval & has_max,
        "time_max_not_allowed",
        "`time_max` must be missing for values that are not interval censored.",
    )
    _check_locations(
        is_interval & ~has_max,
        "time_max_missing",
        "`time_max` must not be missing for interval-censored values.",
    )
    with np.errstate(invalid="ignore"):
        reversed_ = is_interval & (time_a >= time_max_a)
    _check_locations(
        reversed_,
        "interval_bounds_order",
        "`time` must be smaller than `time_max` for interval-censored values.",
    )

    return EventTime(time=time_a, time_max=time_max_a, paired=is_interval, status=status_a)


def new_event_time(time: Any = (), status: Any = ()) -> EventTime:
    """Create an event-time vector with only minimal type checks.

    The low-level constructor, a port of etd's `new_event_time()`. Use `event_time()` to create a
    validated vector.

    Parameters
    ----------
    time
        A sequence with one entry per element: a single number, or a `(lower, upper)` pair for an
        interval-censored value.
    status
        Status codes, one per element. Only the type is checked, so any string is accepted.

    Returns
    -------
    EventTime
        The event-time vector.

    Raises
    ------
    TypeError
        If `time` isn't a sequence of numbers and pairs, or `status` isn't a character vector.
    ValueError
        If `time` and `status` differ in length.

    Examples
    --------
    ```{python}
    import greenwood as gw

    gw.new_event_time(time=[7, (2, 4)], status=["e", "i"])
    ```
    """
    not_list = "`time` must be a list of numbers and (lower, upper) pairs"
    if _is_scalar(time) or time is None:
        raise _EventTimeTypeError(f"{not_list}, not {_type_name(time)}.", check="time_not_list")
    seq, _ = _to_values(time)
    elements: list[Any] = seq.tolist() if isinstance(seq, np.ndarray) else list(seq)
    status_a = _as_character(status, "status")

    lower: list[float] = []
    upper: list[float] = []
    paired: list[bool] = []
    for element in elements:
        if element is None or isinstance(element, (bool, numbers.Real)):
            parts: list[Any] = [element]
        elif isinstance(element, (list, tuple, np.ndarray)) and len(element) in (1, 2):  # pyright: ignore[reportUnknownArgumentType]
            parts = list(element)  # pyright: ignore[reportUnknownArgumentType]
        else:
            raise _EventTimeTypeError(
                f"{not_list}, not one containing {_type_name(element)}.", check="time_not_list"
            )
        values = _cast_double(parts, "time")
        lower.append(float(values[0]))
        upper.append(float(values[1]) if len(values) == 2 else math.nan)
        paired.append(len(values) == 2)

    if len(elements) != status_a.shape[0]:
        raise _EventTimeValueError(
            f"`time` and `status` must have the same length, not {len(elements)} and "
            f"{status_a.shape[0]}.",
            check="fields_size",
        )
    return EventTime(
        time=np.asarray(lower, dtype=np.float64),
        time_max=np.asarray(upper, dtype=np.float64),
        paired=np.asarray(paired, dtype=np.bool_),
        status=status_a,
    )


# -- extractors -----------------------------------------------------------------------------


@singledispatch
def extract_time(x: Any) -> FloatArray:
    """Extract the times from an event-time vector.

    A port of etd's `extract_time()`. Like etd, the shape depends on the data: a 1-D array when no
    element is interval-censored, otherwise a 2-D array with columns `time` and `time_max`, where
    `time_max` is missing (`nan`) for elements that aren't interval-censored.

    Parameters
    ----------
    x
        An `EventTime` vector.

    Returns
    -------
    numpy.ndarray
        A 1-D float array of times, or a 2-D float array of shape `(n, 2)` holding `time` and
        `time_max` when any element is interval-censored.

    Examples
    --------
    ```{python}
    import greenwood as gw

    x = gw.event_time(time=[7, 5, 2], status=["e", "r", "i"], time_max=[None, None, 4])
    gw.extract_time(x)
    ```

    Without intervals, the result is 1-D:

    ```{python}
    gw.extract_time(x[0:2])
    ```
    """
    raise TypeError(f"extract_time() has no method for {_type_name(x)}.")


@extract_time.register
def _(x: EventTime) -> FloatArray:
    lower = x._lower()  # pyright: ignore[reportPrivateUsage]
    if not any(s == "i" for s in x._status):  # pyright: ignore[reportPrivateUsage]
        return lower
    return np.column_stack([lower, x._upper()])  # pyright: ignore[reportPrivateUsage]


@singledispatch
def extract_status(x: Any) -> list[str | None]:
    """Extract the status codes from an event-time vector.

    A port of etd's `extract_status()`.

    Parameters
    ----------
    x
        An `EventTime` vector.

    Returns
    -------
    list of str or None
        One status code per element (`"e"`, `"r"`, `"l"`, or `"i"`), with `None` where it is
        missing.

    Examples
    --------
    ```{python}
    import greenwood as gw

    x = gw.event_time(time=[7, 5, 2], status=["e", "r", "i"], time_max=[None, None, 4])
    gw.extract_status(x)
    ```
    """
    raise TypeError(f"extract_status() has no method for {_type_name(x)}.")


@extract_status.register
def _(x: EventTime) -> list[str | None]:
    return list(x._status)  # pyright: ignore[reportPrivateUsage]
