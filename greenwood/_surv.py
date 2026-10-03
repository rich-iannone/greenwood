"""The `Surv` response object, a port of R's `survival::Surv()`.

`Surv()` takes the same arguments as R's `Surv(time, time2, event, type, origin)` and follows the
same rules: it infers the type from the arguments given, accepts the same status codings, turns
invalid values into missing values with the same warnings, and raises the same errors. The result
holds the same columns as R's `Surv` matrix.

Types:

- `"right"`: `Surv(time, event)`, the common case.
- `"left"`: `Surv(time, event, type="left")`. Status `1` is an exact event, `0` is left-censored.
- `"counting"`: `Surv(start, stop, event)`, for late entry and time-varying covariates.
- `"interval"`: `Surv(time, time2, event, type="interval")` with codes `0` right, `1` exact,
  `2` left, `3` interval. `type="interval2"` takes lower and upper bounds instead, with missing or
  infinite bounds marking censoring, and converts to `"interval"`.
- `"mright"` / `"mcounting"`: a categorical `event` gives a multi-state response. The first level
  is censoring and the remaining levels are the states.

For exact, right-, left-, and interval-censored data, `event_time()` is the more readable way to
build a response. `as_surv()` converts it to a `Surv`. The contract is recorded in
`spec/surv.md`, and `tests/test_surv_conformance.py` replays cases generated from R.
"""

from __future__ import annotations

import json
import math
import numbers
import warnings
from collections.abc import Mapping
from enum import Enum
from typing import Any

import narwhals as nw  # pyright: ignore[reportMissingImports]  # installed + typed; pyright quirk
import numpy as np
import numpy.typing as npt
from typing_extensions import Self

from ._backends import to_dataframe

__all__ = ["Surv", "CensoringType", "first_event"]

Array = npt.NDArray[Any]
FloatArray = npt.NDArray[np.float64]

_WRONG_ARGS = "Wrong number of args for this type of survival data"

# How to build a multi-state response, for estimators that need one.
MULTISTATE_HINT = (
    "Describe it with a formula such as "
    '\'Surv(time, factor(cause, 0:2, c("censor", "pcm", "death")))\', with '
    "`Outcome.first_event()`, or with `Surv()` and a categorical event."
)
_TYPES = ("right", "left", "interval", "counting", "interval2", "mstate")

# Version of the `to_dict()` layout. Version 2 stores R's Surv columns.
_DICT_VERSION = 2


class CensoringType(str, Enum):
    """The type of a `Surv` response, using R's type names.

    - `RIGHT`: right-censored, `Surv(time, event)`.
    - `LEFT`: left-censored, `Surv(time, event, type="left")`.
    - `INTERVAL`: interval-censored, from `type="interval"` or `type="interval2"`.
    - `COUNTING`: counting-process `(start, stop]` intervals, `Surv(start, stop, event)`.
    - `MRIGHT`: multi-state right-censored, from a categorical `event=`.
    - `MCOUNTING`: multi-state counting-process, from a categorical `event=` with start times.

    A response reports its type through `Surv(...).type`. Because `CensoringType` is a `str` enum,
    it compares equal to R's type strings, such as `"right"`.

    Examples
    --------
    ```{python}
    from greenwood import CensoringType

    list(CensoringType)
    ```
    """

    RIGHT = "right"
    """Right-censored data: the event, if any, happens after the recorded time.

    The most common type. Each row has a follow-up time and a status: `1` if the event was seen
    at that time, `0` if the subject was event-free when last seen. `Surv()` infers it from a time
    and a status, `Surv(time=t, event=d)`, and `as_surv()` gives it for an event-time vector with
    only `"e"` and `"r"` codes. The response's columns are `time` and `status`, as in R.

    Examples
    --------
    ```{python}
    import greenwood as gw

    gw.Surv(time=[5, 6, 4], event=[1, 0, 1]).type
    ```
    """

    LEFT = "left"
    """Left-censored data: the event may have happened before the recorded time.

    Each row has a time and a status, using R's meaning: `1` is an event seen at that time, `0`
    means the event had already happened by then (left-censored). It must be asked for with
    `Surv(time=t, event=d, type="left")`, and `as_surv()` gives it for an event-time vector with
    only `"e"` and `"l"` codes. The response's columns are `time` and `status`, as in R.

    Examples
    --------
    ```{python}
    import greenwood as gw

    gw.Surv(time=[5, 6, 4], event=[1, 0, 1], type="left").type
    ```
    """

    INTERVAL = "interval"
    """Interval data: each row may be exact, right-, left-, or interval-censored.

    The most general single-event type. The status uses R's interval codes: `0` right-censored,
    `1` exact, `2` left-censored, and `3` interval-censored. It comes from
    `Surv(time, time2, event, type="interval")` with those codes, from
    `Surv(time=lower, time2=upper, type="interval2")` with open bounds marking censoring, or
    from `as_surv()` for an event-time vector that mixes codes in any other way. The response's
    columns are `time1`, `time2`, and `status`, as in R, where `time2` is `1` for rows that are
    not interval-censored. Fit it with `Turnbull`.

    Examples
    --------
    ```{python}
    import numpy as np
    import greenwood as gw

    # The event happened in (1, 2], after 2, and in (3, 5]
    gw.Surv(time=[1, 2, 3], time2=[2, np.inf, 5], type="interval2").type
    ```
    """

    COUNTING = "counting"
    """Counting-process data: each row is an interval `(start, stop]` at risk.

    Used for late entry (left truncation), where subjects join the study after time zero, and
    for time-varying covariates, where each subject's follow-up is split into several rows. The
    status is `1` if the event happened at the stop time. `Surv()` infers it from a start time, a
    stop time, and a status: `Surv(time=start, time2=stop, event=d)`. The response's columns are
    `start`, `stop`, and `status`, as in R.

    Examples
    --------
    ```{python}
    import greenwood as gw

    gw.Surv(time=[0, 2, 1], time2=[5, 6, 4], event=[1, 0, 1]).type
    ```
    """

    MRIGHT = "mright"
    """Multi-state right-censored data: several kinds of event that compete.

    Used for competing risks, where each subject can experience at most one of several events
    (for example progression or death). The status is `0` for censored, or `k` for the `k`-th
    state name in the response's `states`. `Surv()` gives this type when the status is a
    categorical whose first category means censored, and `first_event()` builds it from one
    `(time, event)` column pair per endpoint. In a formula, write the status with `factor()`. The
    response's columns are `time` and `status`, as in R.

    Examples
    --------
    ```{python}
    import pandas as pd
    import greenwood as gw

    cause = pd.Categorical(["pcm", "censor", "death"], categories=["censor", "pcm", "death"])
    gw.Surv(time=[5, 6, 7], event=cause).type
    ```
    """

    MCOUNTING = "mcounting"
    """Multi-state counting-process data: competing events with late entry.

    The counting-process form of `MRIGHT`: each row is an interval `(start, stop]` at risk, and
    the status is `0` for censored or `k` for the `k`-th state name in `states`. `Surv()` gives
    this type for a start time, a stop time, and a categorical status, and `first_event()` gives
    it when `start=` is supplied. The response's columns are `start`, `stop`, and `status`, as in
    R.

    Examples
    --------
    ```{python}
    import pandas as pd
    import greenwood as gw

    cause = pd.Categorical(["pcm", "censor"], categories=["censor", "pcm", "death"])
    gw.Surv(time=[0, 2], time2=[5, 6], event=cause).type
    ```
    """


# -- input classification -------------------------------------------------------------------


def _values(x: Any) -> tuple[list[Any], str, list[str] | None]:
    """Classify an input the way R's type checks see it.

    Returns the values (with every missing value as `None`), a kind (`"numeric"`, `"logical"`,
    `"categorical"`, `"character"`, or `"other"`), and the levels for categorical input.
    """
    if isinstance(x, (bool, np.bool_, numbers.Number, str, np.generic)):
        x = [x]
    if hasattr(x, "categories") and hasattr(x, "codes"):
        # A bare pandas Categorical (not wrapped in a Series), recognized without importing pandas.
        levels = [str(level) for level in x.categories]
        codes = [int(code) for code in x.codes]
        return [None if code < 0 else levels[code] for code in codes], "categorical", levels
    if isinstance(x, np.ndarray):
        arr: np.ndarray[Any, Any] = x  # pyright: ignore[reportUnknownVariableType]
        if arr.ndim != 1:
            return [], "other", None
        if arr.dtype.kind == "b":
            return arr.tolist(), "logical", None
        if arr.dtype.kind in "iuf":
            return [None if math.isnan(v) else float(v) for v in arr.astype(float)], "numeric", None
        if arr.dtype.kind == "U":
            return arr.tolist(), "character", None
        if arr.dtype.kind != "O":
            return [], "other", None
        values: list[Any] = arr.tolist()
    elif isinstance(x, (list, tuple)):
        values = list(x)  # pyright: ignore[reportUnknownArgumentType]
    else:
        try:
            series = nw.from_native(x, series_only=True)
        except TypeError:
            return _values(np.asarray(x))
        nulls = series.is_null().to_list()
        raw = series.to_list()
        values = [None if null else v for v, null in zip(raw, nulls, strict=True)]
        if series.dtype in (nw.Categorical, nw.Enum):
            levels = [str(level) for level in series.cat.get_categories().to_list()]
            return values, "categorical", levels

    present = [v for v in values if v is not None and not (isinstance(v, float) and math.isnan(v))]
    if all(isinstance(v, (bool, np.bool_)) for v in present) and present:
        return [None if v is None else bool(v) for v in values], "logical", None
    if all(isinstance(v, numbers.Real) and not isinstance(v, (bool, np.bool_)) for v in present):
        return [None if v is None or v != v else float(v) for v in values], "numeric", None
    if all(isinstance(v, str) for v in present):
        return values, "character", None
    return values, "other", None


def _float_array(values: list[Any]) -> FloatArray:
    return np.asarray([math.nan if v is None else float(v) for v in values], dtype=np.float64)


def _numeric(x: Any, message: str) -> FloatArray:
    """Return `x` as float64 if R would call it numeric, else raise with R's message."""
    values, kind, _ = _values(x)
    if kind != "numeric" and not (kind == "logical" and not values):
        raise TypeError(message)
    return _float_array(values)


def _r_label(value: Any) -> str:
    """Write a value the way R's `as.character()` does, for factor levels."""
    if isinstance(value, (bool, np.bool_)):
        return "TRUE" if value else "FALSE"
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def _as_factor(
    values: list[Any], kind: str, levels: list[str] | None
) -> tuple[FloatArray, list[str]]:
    """Port of `as.numeric(as.factor(event)) - 1`: 0-based level codes, NaN for missing."""
    if kind == "categorical":
        assert levels is not None
        labels = [None if v is None else str(v) for v in values]
    else:
        present = sorted({v for v in values if v is not None})
        levels = [_r_label(v) for v in present]
        labels = [None if v is None else _r_label(v) for v in values]
    index = {level: i for i, level in enumerate(levels)}
    codes = np.asarray(
        [math.nan if label is None else float(index[label]) for label in labels],
        dtype=np.float64,
    )
    return codes, levels


def _numeric_status(event: Any, message: str) -> FloatArray:
    """Port of R's status handling for right, left, and counting data.

    Logical status becomes 0/1. Numeric status that peaks at 2 is shifted down by one (so 1/2
    coding becomes 0/1). Anything other than 0 or 1 becomes missing, with a warning.
    """
    values, kind, _ = _values(event)
    if kind == "logical" or (kind == "numeric" and not values):
        return np.asarray([math.nan if v is None else float(v) for v in values], dtype=np.float64)
    if kind != "numeric":
        raise TypeError(message)
    status = _float_array(values)
    observed = ~np.isnan(status)
    if observed.any() and status[observed].max() == 2:
        status = status - 1
    valid = (status == 0) | (status == 1)
    if not valid[observed].all():
        warnings.warn("Invalid status value, converted to NA", UserWarning, stacklevel=4)
    return np.where(valid, status, np.nan)


# -- the response ---------------------------------------------------------------------------


class Surv:
    """A survival response, built exactly as R's `survival::Surv()` builds one.

    `Surv()` records a time-to-event outcome in one of R's forms. The type is inferred from the
    arguments, as in R:

    - `Surv(time, event)` is right-censored. `event=` may be logical, `0`/`1`, or `1`/`2` (as in
      the `lung` data, where `2` means died).
    - `Surv(start, stop, event)` is a counting-process response, for late entry (left truncation)
      and time-varying covariates.
    - A categorical `event=` (a pandas or Polars categorical, or an Arrow dictionary) gives a
      multi-state response, whose first level means censored.
    - `type="left"`, `type="interval"`, and `type="interval2"` must be asked for.

    Status values R doesn't accept become missing, with a warning. Missing values are kept, and
    rows with any missing value are dropped when a model is fitted. For exact, right-, left-, and
    interval-censored data, `event_time()` is often clearer, and `as_surv()` converts it.

    `Surv()` takes values: lists, arrays, or series already in hand. To start from a data frame,
    name the columns instead, in a formula such as `"Surv(time, status == 2) ~ age"` or with
    `Outcome.surv(time="time", event="status == 2")`, and pass `data=` when fitting. That form is
    evaluated on the frame's own backend, so it works the same on pandas, Polars, PyArrow, DuckDB,
    and lazy frames.

    Parameters
    ----------
    time
        Follow-up time for right-censored data, or the start time for counting-process data, or
        the lower bound for interval data. Numeric, with `None` or `nan` for missing values.
    time2
        Stop time for counting-process data, or the upper bound for interval data. When `event=` is
        not given and `type=` is not an interval type, a second argument is the status (R's rule),
        so `Surv(time, status)` works positionally.
    event
        The status. Logical, `0`/`1`, or `1`/`2` for right, left, and counting data. Codes `0`
        (right), `1` (exact), `2` (left), and `3` (interval) for `type="interval"`. A categorical
        for a multi-state response.
    type
        One of `"right"`, `"left"`, `"interval"`, `"counting"`, `"interval2"`, or `"mstate"`. The
        default infers `"right"` or `"counting"` from the number of arguments.
    origin
        Subtracted from every time, as in R.

    Raises
    ------
    TypeError
        If a time or the status isn't of a type R accepts. The messages match R's.
    ValueError
        If lengths differ, or the arguments don't fit the requested type.

    Warns
    -----
    UserWarning
        When status values are invalid or intervals run backwards. Those rows become missing, as
        in R.

    Examples
    --------
    Four subjects: events at times 5 and 4, censored at 6 and 9:

    ```{python}
    import greenwood as gw

    gw.Surv(time=[5, 6, 4, 9], event=[1, 0, 1, 0])
    ```

    `1`/`2` status coding works as it does in R, where `2` means an event:

    ```{python}
    gw.Surv(time=[5, 6, 4, 9], event=[2, 1, 2, 1])
    ```

    A start time gives a counting-process response:

    ```{python}
    gw.Surv(time=[0, 2, 1], time2=[5, 6, 4], event=[1, 0, 1])
    ```

    A categorical status gives a multi-state response, with the first level meaning censored:

    ```{python}
    import pandas as pd

    outcome = pd.Categorical(
        ["death", "censor", "relapse", "death"], categories=["censor", "relapse", "death"]
    )
    gw.Surv(time=[5, 6, 7, 8], event=outcome)
    ```

    Interval data uses R's coding, or lower and upper bounds with `type="interval2"`:

    ```{python}
    import numpy as np

    # The event happened in (1, 2], after 2, and in (3, 5]
    gw.Surv(time=[1, 2, 3], time2=[2, np.inf, 5], type="interval2")
    ```

    Starting from a data frame, name the columns and let the model read them. We'll use the bundled
    `pbc` dataset, from a Mayo Clinic trial in primary biliary cholangitis, where status is `0`
    (censored), `1` (transplant), or `2` (died):

    ```{python}
    pbc = gw.load_dataset("pbc")

    # The same as Surv(time, status == 2) in R, evaluated on the frame's own backend
    gw.CoxPH().fit("Surv(time, status == 2) ~ age + bili", data=pbc)
    ```
    """

    __slots__ = ("_type", "_stop", "_status", "_start", "_lower", "_states")

    _type: CensoringType
    _stop: FloatArray
    _status: FloatArray
    _start: FloatArray | None
    _lower: FloatArray | None
    _states: tuple[str, ...] | None

    def __init__(
        self,
        time: Any,
        time2: Any = None,
        event: Any = None,
        *,
        type: str | None = None,
        origin: float = 0.0,
    ) -> None:
        if type is not None and type not in _TYPES:
            raise ValueError(f"`type` must be one of {', '.join(map(repr, _TYPES))}, not {type!r}.")
        _check_not_column_names(time, time2, event)
        time_a = _numeric(time, "Time variable is not numeric")
        _check_not_scalar_comparison(time_a.shape[0], time2, event)
        n = time_a.shape[0]
        ng = 1 + (time2 is not None) + (event is not None)

        resolved: str
        if type is None or type == "mstate":
            resolved = "right" if ng in (1, 2) else "counting"
        else:
            resolved = type
            if ng != 3 and resolved in ("interval", "counting"):
                raise ValueError(_WRONG_ARGS)
            if ng != 2 and resolved in ("right", "left", "interval2"):
                raise ValueError(_WRONG_ARGS)
        mstate = type == "mstate"

        if ng == 1:
            self._set(CensoringType.RIGHT, stop=time_a - origin, status=np.ones(n))
        elif resolved in ("right", "left"):
            if event is None:
                event, time2 = time2, None
            self._init_right_left(time_a, event, resolved, mstate, origin)
        elif resolved == "counting":
            self._init_counting(time_a, time2, event, mstate, origin)
        else:
            self._init_interval(time_a, time2, event, resolved, origin)

    def _init_right_left(
        self, time: FloatArray, event: Any, kind: str, mstate: bool, origin: float
    ) -> None:
        values, value_kind, levels = _values(event)
        if len(values) != time.shape[0]:
            raise ValueError("Time and status are different lengths")
        if mstate or value_kind == "categorical":
            status, state_levels = _as_factor(values, value_kind, levels)
            self._set(
                CensoringType.MRIGHT,
                stop=time - origin,
                status=status,
                states=_states(state_levels),
            )
            return
        status = _numeric_status(event, "Invalid status value, must be logical or numeric")
        ctype = CensoringType.RIGHT if kind == "right" else CensoringType.LEFT
        self._set(ctype, stop=time - origin, status=status)

    def _init_counting(
        self, start: FloatArray, stop: Any, event: Any, mstate: bool, origin: float
    ) -> None:
        n = start.shape[0]
        stop_values, _, _ = _values(stop)
        event_values, event_kind, levels = _values(event)
        if len(stop_values) != n:
            raise ValueError("Start and stop are different lengths")
        if len(event_values) != n:
            raise ValueError("Start and event are different lengths")
        stop_a = _numeric(stop, "Stop time is not numeric")
        with np.errstate(invalid="ignore"):
            backwards = start >= stop_a
        if backwards.any():
            start = np.where(backwards, np.nan, start)
            warnings.warn("Stop time must be > start time, NA created", UserWarning, stacklevel=3)
        if mstate or event_kind == "categorical":
            status, state_levels = _as_factor(event_values, event_kind, levels)
            self._set(
                CensoringType.MCOUNTING,
                stop=stop_a - origin,
                status=status,
                start=start - origin,
                states=_states(state_levels),
            )
            return
        status = _numeric_status(event, "Invalid status value")
        self._set(CensoringType.COUNTING, stop=stop_a - origin, status=status, start=start - origin)

    def _init_interval(
        self, time: FloatArray, time2: Any, event: Any, kind: str, origin: float
    ) -> None:
        n = time.shape[0]
        if kind == "interval2":
            upper = _numeric(time2, "Time2 must be numeric")
            if upper.shape[0] != n:
                raise ValueError("time and time2 are different lengths")
            with np.errstate(invalid="ignore"):
                backwards = ~np.isnan(time) & ~np.isnan(upper) & (time > upper)
            lower = np.where(np.isfinite(time), time, np.nan)
            upper = np.where(np.isfinite(upper), upper, np.nan)
            unknown = np.isnan(lower) & np.isnan(upper)
            status = np.where(
                np.isnan(lower),
                2.0,
                np.where(np.isnan(upper), 0.0, np.where(lower == upper, 1.0, 3.0)),
            )
            lower = np.where(status != 2, lower, upper)
            if backwards.any():
                warnings.warn(
                    "Invalid interval: start > stop, NA created", UserWarning, stacklevel=3
                )
                status = np.where(backwards, np.nan, status)
            status = np.where(unknown, np.nan, status)
            time, time2_a = lower, upper
        else:
            values, value_kind, _ = _values(event)
            if len(values) != n:
                raise ValueError("Time and status are different lengths")
            if value_kind != "numeric":
                raise TypeError("Invalid status value, must be logical or numeric")
            codes = _float_array(values)
            valid = np.isin(codes, (0.0, 1.0, 2.0, 3.0))
            status = np.where(valid, codes, np.nan)
            if not valid[~np.isnan(codes)].all():
                warnings.warn(
                    "Status must be 0, 1, 2 or 3; converted to NA", UserWarning, stacklevel=3
                )
            if (codes == 3).any():
                time2_a = _numeric(time2, "Time2 must be numeric")
                if time2_a.shape[0] != n:
                    raise ValueError("time and time2 are different lengths")
                with np.errstate(invalid="ignore"):
                    backwards = (status == 3) & (time > time2_a)
                if backwards.any():
                    status = np.where(backwards, np.nan, status)
                    warnings.warn(
                        "Invalid interval: start > stop, NA created", UserWarning, stacklevel=3
                    )
            else:
                time2_a = np.ones(n)
        # R stores 1 in time2 for rows that aren't interval-censored.
        time2_col = np.where(status == 3, time2_a - origin, 1.0)
        self._set(CensoringType.INTERVAL, stop=time2_col, status=status, lower=time - origin)

    def _set(
        self,
        ctype: CensoringType,
        *,
        stop: FloatArray,
        status: FloatArray,
        start: FloatArray | None = None,
        lower: FloatArray | None = None,
        states: tuple[str, ...] | None = None,
    ) -> None:
        self._type = ctype
        self._stop = np.asarray(stop, dtype=np.float64)
        self._status = np.asarray(status, dtype=np.float64)
        self._start = None if start is None else np.asarray(start, dtype=np.float64)
        self._lower = None if lower is None else np.asarray(lower, dtype=np.float64)
        self._states = states

    @classmethod
    def _from_fields(
        cls,
        type: CensoringType,
        *,
        stop: Any,
        status: Any,
        start: Any = None,
        lower: Any = None,
        states: tuple[str, ...] | None = None,
    ) -> Self:
        """Build a response from its internal fields, with no R-style processing."""
        obj = cls.__new__(cls)
        obj._set(
            type,
            stop=stop,
            status=status,
            start=start,
            lower=lower,
            states=states,
        )
        return obj

    # -- fields -------------------------------------------------------------------------------

    @property
    def type(self) -> CensoringType:
        """The kind of response, using R's type names.

        One of `CensoringType.RIGHT`, `LEFT`, `INTERVAL`, `COUNTING`, `MRIGHT` (multi-state), or
        `MCOUNTING` (multi-state with start times). The type is chosen when the response is built:
        inferred from the arguments, as in R, or set with `type=`. A `type="interval2"` response
        reports `INTERVAL`, because R converts it to interval coding. Because `CensoringType` is a
        `str` enum, the value also compares equal to R's type strings.

        Returns
        -------
        CensoringType
            The response type.

        Examples
        --------
        ```{python}
        import greenwood as gw

        y = gw.Surv(time=[0, 2, 1], time2=[5, 6, 4], event=[1, 0, 1])
        y.type, y.type == "counting"
        ```
        """
        return self._type

    @property
    def stop(self) -> FloatArray:
        """The main time column, one value per row.

        What it holds depends on the type, following R's `Surv` matrix:

        - Right, left, and multi-state data: the follow-up time (R's time column).
        - Counting-process data: the time each interval ends (R's stop column).
        - Interval data: the upper bound (R's time2 column). As in R, rows that are not
          interval-censored (status other than `3`) hold the placeholder `1`, and their time is in
          `lower`.

        Any origin offset given to `Surv(origin=...)` has already been subtracted. Missing times
        are `nan`.

        Returns
        -------
        numpy.ndarray
            A float array with one time per row.

        Examples
        --------
        ```{python}
        import greenwood as gw

        gw.Surv(time=[5, 6, 4, 9], event=[1, 0, 1, 0]).stop
        ```
        """
        return self._stop

    @property
    def status(self) -> FloatArray:
        """The status column, one code per row, stored as floats as R stores it.

        The codes depend on the type:

        - Right and counting data: `1` for an event, `0` for censored. Logical status and R's
          `1`/`2` coding are converted to these codes when the response is built.
        - Left data: `1` for an exact event, `0` for left-censored.
        - Interval data: `0` right-censored, `1` exact, `2` left-censored, `3` interval-censored.
        - Multi-state data: `0` for censored, `k` for the `k`-th entry of `states`.

        Codes that R doesn't accept become missing (`nan`) when the response is built, with a
        warning, as do backwards intervals. Use `int(y.status[i])` when a code is used as an index.

        Returns
        -------
        numpy.ndarray
            A float array with one code per row, `nan` where it is missing.

        Examples
        --------
        R's `1`/`2` coding (as in the `lung` data) becomes `0`/`1`:

        ```{python}
        import greenwood as gw

        gw.Surv(time=[5, 6, 4], event=[2, 1, 2]).status
        ```
        """
        return self._status

    @property
    def start(self) -> FloatArray | None:
        """The time each row starts being at risk, for counting-process data.

        A counting-process response, `Surv(time=start, time2=stop, event=...)`, describes each row
        as an interval `(start, stop]`. This is how late entry (left truncation) and time-varying
        covariates are expressed. A row whose start is not before its stop has its start set to
        `nan`, with a warning, as in R. For every other type there are no start times and this is
        `None`. Use `entry` for a value that is defined for every type.

        Returns
        -------
        numpy.ndarray or None
            A float array of start times, or `None` when the response has none.

        Examples
        --------
        ```{python}
        import greenwood as gw

        # Subject 2 joins the study at time 2
        gw.Surv(time=[0, 2, 1], time2=[5, 6, 4], event=[1, 0, 1]).start
        ```
        """
        return self._start

    @property
    def lower(self) -> FloatArray | None:
        """The first time column of interval data (R's `time1`), otherwise `None`.

        For interval data this is the time that goes with each row's status: the censoring time
        for right-censored rows, the event time for exact rows, the time by which the event had
        happened for left-censored rows, and the lower bound for interval-censored rows (whose
        upper bound is in `stop`). For every other type it is `None`.

        Returns
        -------
        numpy.ndarray or None
            A float array with one time per row, or `None` for types other than interval.

        Examples
        --------
        ```{python}
        import numpy as np
        import greenwood as gw

        # Rows: an interval (1, 2], right-censored at 2, an interval (3, 5]
        y = gw.Surv(time=[1, 2, 3], time2=[2, np.inf, 5], type="interval2")
        y.lower, y.stop, y.status
        ```
        """
        return self._lower

    @property
    def states(self) -> tuple[str, ...] | None:
        """The names of the event states of a multi-state response, otherwise `None`.

        A multi-state response comes from a categorical `event=` (or `first_event()`). Its first
        category means censored, and the remaining categories, in order, are the states. A row
        with status `k` reached `states[k - 1]`. For single-event responses this is `None`.

        Returns
        -------
        tuple of str or None
            The state names in code order, or `None` for a single-event response.

        Examples
        --------
        ```{python}
        import pandas as pd
        import greenwood as gw

        outcome = pd.Categorical(
            ["death", "censor", "relapse"], categories=["censor", "relapse", "death"]
        )
        y = gw.Surv(time=[5, 6, 7], event=outcome)
        y.states, y.status
        ```
        """
        return self._states

    # -- derived views ------------------------------------------------------------------------

    @property
    def n(self) -> int:
        """The number of rows, including any with missing values.

        Each row is one observation: one subject for most data, or one interval of a subject's
        follow-up for counting-process data. Rows with missing values are counted here but dropped
        when a model is fitted, so a fit may use fewer rows. `len(y)` gives the same number.

        Returns
        -------
        int
            The number of rows.

        Examples
        --------
        ```{python}
        import greenwood as gw

        gw.Surv(time=[5, 6, 4, 9], event=[1, 0, 1, 0]).n
        ```
        """
        return int(self._stop.shape[0])

    @property
    def entry(self) -> FloatArray:
        """The time each row enters the risk set, defined for every type.

        For counting-process data this is the start time of each interval, the same as `start`.
        For every other type, subjects are at risk from the beginning, so every value is minus
        infinity (`-inf`). Estimators use `entry` to build risk sets without having to check
        whether the response has start times.

        Returns
        -------
        numpy.ndarray
            A float array with one entry time per row.

        Examples
        --------
        ```{python}
        import greenwood as gw

        gw.Surv(time=[5, 6], event=[1, 0]).entry
        ```

        ```{python}
        gw.Surv(time=[0, 2], time2=[5, 6], event=[1, 0]).entry
        ```
        """
        if self._start is not None:
            return self._start
        return np.full(self.n, -np.inf)

    @property
    def event(self) -> npt.NDArray[np.bool_]:
        """Whether each row ended in an event, as a boolean array.

        A row counts as an event when its status is `1` or more. For multi-state data that means
        any state, not a particular one. For interval data it means the event is known to have
        happened (exactly, before a time, or within an interval), so only right-censored rows are
        `False`. Rows with a missing status are `False`.

        Returns
        -------
        numpy.ndarray
            A boolean array with one value per row.

        Examples
        --------
        ```{python}
        import greenwood as gw

        gw.Surv(time=[5, 6, 4, 9], event=[1, 0, 1, 0]).event
        ```
        """
        with np.errstate(invalid="ignore"):
            return self._status >= 1

    @property
    def is_truncated(self) -> bool:
        """Whether the response has start times (late entry or time-varying data).

        `True` for counting-process responses, built as `Surv(time=start, time2=stop,
        event=...)`, including multi-state ones. Estimators that can't handle delayed entry check
        this and raise an error rather than silently ignoring the start times.

        Returns
        -------
        bool
            `True` if the response has start times.

        Examples
        --------
        ```{python}
        import greenwood as gw

        gw.Surv(time=[0, 2], time2=[5, 6], event=[1, 0]).is_truncated
        ```
        """
        return self._start is not None

    @property
    def is_multistate(self) -> bool:
        """Whether the response has more than one kind of event.

        `True` for multi-state and competing-risks responses (types `MRIGHT` and `MCOUNTING`),
        which come from a categorical `event=` or from `first_event()`. Their state names are in
        `states`. Estimators for a single event type, such as `KaplanMeier` and `CoxPH`, reject
        these responses. Competing-risks estimators such as `AalenJohansen` require them.

        Returns
        -------
        bool
            `True` if the response has several event states.

        Examples
        --------
        ```{python}
        import pandas as pd
        import greenwood as gw

        outcome = pd.Categorical(["pcm", "censor"], categories=["censor", "pcm", "death"])
        gw.Surv(time=[5, 6], event=outcome).is_multistate
        ```
        """
        return self._type in (CensoringType.MRIGHT, CensoringType.MCOUNTING)

    @property
    def n_events(self) -> int:
        """The number of rows that ended in an event.

        This counts the `True` values of `event`: rows with status `1` or more, in any state for
        multi-state data. Rows with a missing status are not counted.

        Returns
        -------
        int
            The number of events.

        Examples
        --------
        ```{python}
        import greenwood as gw

        gw.Surv(time=[5, 6, 4, 9], event=[1, 0, 1, 0]).n_events
        ```
        """
        return int(np.count_nonzero(self.event))

    @property
    def n_censored(self) -> int:
        """The number of censored rows (status `0`).

        For right, counting, and multi-state data these are the rows that were event-free when
        last seen. For left data, status `0` means left-censored, and for interval data it means
        right-censored. Rows with a missing status are not counted, so `n_events + n_censored`
        is less than `n` when some statuses are missing.

        Returns
        -------
        int
            The number of censored rows.

        Examples
        --------
        ```{python}
        import greenwood as gw

        y = gw.Surv(time=[5, 6, 4, 9], event=[1, 0, 1, 0])
        y.n_censored, y.n_events + y.n_censored == y.n
        ```
        """
        return int(np.count_nonzero(self._status == 0))

    def is_na(self) -> npt.NDArray[np.bool_]:
        """Return which rows have a missing value in any column, as R's `is.na()` does.

        A row is missing when any of its columns is `nan`: a missing time, a missing status, or (for
        counting data) a missing start time. `Surv()` keeps such rows rather than dropping them, as
        R does. They come from missing input values, and also from values R turns into missing when
        the response is built: an invalid status code, a start time that is not before its stop
        time, or an interval that runs backwards. Estimators drop these rows when fitting and report
        how many in `n_dropped_`.

        Returns
        -------
        numpy.ndarray
            A boolean array with one value per row, `True` where the row has a missing value.

        Examples
        --------
        The second subject's time is missing:

        ```{python}
        import greenwood as gw

        y = gw.Surv(time=[5, None, 4], event=[1, 0, 1])
        y.is_na()
        ```

        Select the complete rows with the inverted mask:

        ```{python}
        y[~y.is_na()]
        ```
        """
        missing = np.zeros(self.n, dtype=np.bool_)
        for column in self._columns().values():
            missing |= np.isnan(column)
        return missing

    # -- vector protocol ------------------------------------------------------------------------

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, key: Any) -> Surv:
        """Select rows, returning a new response of the same type.

        Indexing works like indexing a NumPy array and always returns a `Surv`, even for a single
        row (as subsetting an R `Surv` with `y[i]` does). Every column is subset together, and the
        type and state names are kept. Negative integers count from the end, as usual in Python.

        Parameters
        ----------
        key
            An integer, a slice, an array of integer positions, or a boolean mask with one value
            per row.

        Returns
        -------
        Surv
            The selected rows.

        Raises
        ------
        IndexError
            If an integer position is out of range.
        TypeError
            If `key` is not an integer, a slice, an integer array, or a boolean mask.

        Examples
        --------
        ```{python}
        import greenwood as gw

        y = gw.Surv(time=[5, 6, 4, 9], event=[1, 0, 1, 0])
        y[1:3]
        ```

        A boolean mask keeps the rows where it is `True`, here the events:

        ```{python}
        y[y.event]
        ```
        """
        if isinstance(key, (int, np.integer)):
            i = int(key)  # pyright: ignore[reportUnknownArgumentType]
            if not -self.n <= i < self.n:
                raise IndexError(f"Index {i} is out of bounds for a Surv of length {self.n}.")
            index: Any = [i]
        elif isinstance(key, slice):
            index = key
        else:
            index = np.asarray(key)
            if index.dtype.kind not in "iub" or index.ndim != 1:
                raise TypeError(
                    "Surv indices must be an integer, a slice, an integer array, or a boolean mask."
                )

        def pick(a: FloatArray | None) -> FloatArray | None:
            return None if a is None else a[index]

        return Surv._from_fields(
            self._type,
            stop=self._stop[index],
            status=self._status[index],
            start=pick(self._start),
            lower=pick(self._lower),
            states=self._states,
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Surv):
            return NotImplemented
        if self._type is not other._type or self._states != other._states:
            return False
        mine, theirs = self._columns(), other._columns()
        return list(mine) == list(theirs) and all(
            np.array_equal(mine[k], theirs[k], equal_nan=True) for k in mine
        )

    __hash__ = None  # pyright: ignore[reportAssignmentType]

    def __repr__(self) -> str:
        extra = ""
        if self.is_multistate:
            extra += f", states={self._states}"
        n_na = int(self.is_na().sum())
        if n_na:
            extra += f", missing={n_na}"
        return f"Surv(type={self._type.value}, n={self.n}, events={self.n_events}{extra})"

    # -- interop --------------------------------------------------------------------------------

    def _columns(self) -> dict[str, FloatArray]:
        """R's matrix columns for this type."""
        if self._type in (CensoringType.COUNTING, CensoringType.MCOUNTING):
            assert self._start is not None
            return {"start": self._start, "stop": self._stop, "status": self._status}
        if self._type is CensoringType.INTERVAL:
            assert self._lower is not None
            return {"time1": self._lower, "time2": self._stop, "status": self._status}
        return {"time": self._stop, "status": self._status}

    def to_frame(self, *, format: str | None = None) -> Any:
        """Return the response as a table with the same columns as R's `Surv` matrix.

        Each row of the table is one row of the response. The columns depend on the type, and match
        the columns of the matrix R's `Surv()` returns:

        - Right, left, and multi-state data: `time` and `status`.
        - Counting-process data: `start`, `stop`, and `status`.
        - Interval data: `time1`, `time2`, and `status`. As in R, `time2` is `1` for rows that are
          not interval-censored.

        Status is stored as a float, as in R. Missing values are nulls. Use this table to inspect a
        response, to join it back onto other data, or to export it.

        Parameters
        ----------
        format
            Output format: `None` (default), `"pandas"`, `"polars"`, or `"pyarrow"`. When `None`,
            a backend is auto-detected (Polars, then Pandas, then PyArrow).

        Returns
        -------
        pandas.DataFrame, polars.DataFrame, or pyarrow.Table
            One row per observation, with the columns listed above.

        Raises
        ------
        ImportError
            If the requested DataFrame library (or, with `format=None`, any of them) is not
            installed.

        Examples
        --------
        ```{python}
        import greenwood as gw

        gw.Surv(time=[0, 2, 1], time2=[5, 6, 4], event=[1, 0, 1]).to_frame(format="polars")
        ```

        Interval data uses R's interval columns:

        ```{python}
        import numpy as np

        gw.Surv(time=[1, 2, 3], time2=[2, np.inf, 5], type="interval2").to_frame(format="polars")
        ```
        """
        return to_dataframe(
            {name: _nullable(values) for name, values in self._columns().items()}, format=format
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-ready mapping that fully describes the response.

        The mapping holds only plain Python values (strings, floats, lists, and `None`), so it can
        be written with `json.dumps()`, stored, or sent to another process, and turned back into
        an equal response with `~~greenwood.Surv.from_dict()`. Missing values become `None`. The
        layout is versioned: version 2 stores R's `Surv` columns.

        Returns
        -------
        dict
            A mapping with the keys `"version"` (the layout version), `"type"` (the response type,
            as one of R's type names), `"states"` (the state names, or `None`), and `"columns"` (R's
            column names mapped to lists of values).

        Examples
        --------
        ```{python}
        import greenwood as gw

        gw.Surv(time=[5, 6], event=[1, 0]).to_dict()
        ```
        """
        return {
            "version": _DICT_VERSION,
            "type": self._type.value,
            "states": None if self._states is None else list(self._states),
            "columns": {name: _nullable(values) for name, values in self._columns().items()},
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Self:
        """Rebuild a response from the output of `~~greenwood.Surv.to_dict()`.

        The mapping is read as it was written, without applying `Surv()`'s rules again: the stored
        type, columns, and state names are used directly. The result compares equal to the
        response that produced the mapping.

        Parameters
        ----------
        data
            A mapping produced by `~~greenwood.Surv.to_dict()`, possibly after a round trip through
            JSON.

        Returns
        -------
        Surv
            The rebuilt response.

        Raises
        ------
        ValueError
            If the mapping has an unsupported layout version, such as one written by an earlier
            Greenwood release.

        Examples
        --------
        ```{python}
        import greenwood as gw

        y = gw.Surv(time=[5, 6], event=[1, 0])
        data = y.to_dict()
        gw.Surv.from_dict(data) == y
        ```
        """
        if data.get("version") != _DICT_VERSION:
            raise ValueError(
                f"Unsupported Surv layout version {data.get('version')!r}; expected "
                f"{_DICT_VERSION}."
            )
        ctype = CensoringType(data["type"])
        columns = {name: _float_array(list(values)) for name, values in data["columns"].items()}
        states = None if data.get("states") is None else tuple(data["states"])
        if ctype in (CensoringType.COUNTING, CensoringType.MCOUNTING):
            return cls._from_fields(
                ctype,
                stop=columns["stop"],
                status=columns["status"],
                start=columns["start"],
                states=states,
            )
        if ctype is CensoringType.INTERVAL:
            return cls._from_fields(
                ctype, stop=columns["time2"], status=columns["status"], lower=columns["time1"]
            )
        return cls._from_fields(
            ctype, stop=columns["time"], status=columns["status"], states=states
        )

    def to_json(self, *, indent: int | None = 2) -> str:
        """Serialize the response to a JSON string.

        The string is the JSON encoding of `~~greenwood.Surv.to_dict()`: the layout version, the
        type, the state names, and R's columns, with missing values as `null`. Turn it back into a
        response with `~~greenwood.Surv.from_json()`.

        Parameters
        ----------
        indent
            Indentation passed to `json.dumps()`. The default `2` is readable, and `None` gives a
            compact single-line string.

        Returns
        -------
        str
            The JSON text.

        Examples
        --------
        ```{python}
        import greenwood as gw

        print(gw.Surv(time=[5, 6], event=[1, 0]).to_json(indent=None))
        ```
        """
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_json(cls, text: str) -> Self:
        """Rebuild a response from the output of `~~greenwood.Surv.to_json()`.

        The text is parsed with `json.loads()` and passed to `~~greenwood.Surv.from_dict()`, so the
        same rules apply: the stored type, columns, and state names are used as they are, and the
        result compares equal to the response that was serialized.

        Parameters
        ----------
        text
            A JSON string produced by `~~greenwood.Surv.to_json()`.

        Returns
        -------
        Surv
            The rebuilt response.

        Raises
        ------
        ValueError
            If the text is not valid JSON, or has an unsupported layout version.

        Examples
        --------
        ```{python}
        import greenwood as gw

        y = gw.Surv(time=[5, 6], event=[1, 0])
        text = y.to_json()
        gw.Surv.from_json(text) == y
        ```
        """
        return cls.from_dict(json.loads(text))


_NAME_COLUMNS = (
    "`Surv()` takes values, not column names. To name columns, write a formula such as "
    '\'Surv(time, status == 2) ~ age\' or use `Outcome.surv(time="time", event="status == 2")`, '
    "and pass `data=` to `fit()`. Both work on every DataFrame backend."
)


def _check_not_column_names(*values: Any) -> None:
    """A bare string where values belong is almost always a column name."""
    for arg, value in zip(("time", "time2", "event"), values, strict=True):
        if isinstance(value, str):
            prefix = "Time variable is not numeric" if arg == "time" else "Invalid argument"
            raise TypeError(f"{prefix}. `{arg}` is the string {value!r}. {_NAME_COLUMNS}")


def _check_not_scalar_comparison(n: int, *values: Any) -> None:
    """A single boolean where a column belongs usually comes from comparing a PyArrow or DuckDB
    column with `==`, which compares whole objects instead of rows."""
    if n <= 1:
        return
    for arg, value in zip(("time2", "event"), values, strict=True):
        if isinstance(value, (bool, np.bool_)):
            raise ValueError(
                f"Time and status are different lengths. `{arg}` is a single {bool(value)}, but "
                f"`time` has {n} values. Comparing a PyArrow or DuckDB column with `==` gives one "
                "True/False rather than one per row. Write the comparison in a formula, such as "
                "'Surv(time, status == 2) ~ age', or in `Outcome.surv(event=\"status == 2\")`, "
                "and pass `data=` to `fit()`."
            )


def _states(levels: list[str]) -> tuple[str, ...]:
    """The state names of a factor: every level but the first (censoring), none blank."""
    states = tuple(levels[1:])
    if any(state == "" for state in states):
        raise ValueError("each state must have a non-blank name")
    return states


def _nullable(values: FloatArray) -> list[float | None]:
    return [None if math.isnan(v) else float(v) for v in values]


# -- first_event ----------------------------------------------------------------------------


def first_event(
    endpoints: Mapping[str, tuple[Any, Any]],
    *,
    censor_at: Any = None,
    start: Any = None,
) -> Surv:
    """Build a competing-risks response from several endpoints: the earliest observed event wins.

    Some datasets record each endpoint in its own pair of columns rather than in one outcome
    column. `mgus2` is an example: `ptime`/`pstat` hold the time to plasma cell malignancy (PCM) and
    whether it occurred, and `futime`/`death` do the same for death. A competing-risks analysis
    needs one time and one cause per subject: the first endpoint observed, or censoring if none
    was.

    For each row, the endpoint with the earliest observed event becomes the cause and its time
    becomes the event time. Ties go to the endpoint listed first. Rows with no observed event are
    censored at the latest endpoint time (the last time the subject was known to be event-free), or
    at `censor_at` if given. This helper is Greenwood's own: R's `survival`{.gd-no-link} has no
    direct equivalent.

    Parameters
    ----------
    endpoints
        A mapping of state name to a `(time, event)` pair, in priority order for ties.
        `event`{.gd-no-link} is logical or `0`/`1`.
    censor_at
        The time at which rows with no observed event are censored. The default is the latest of
        the endpoint times.
    start
        Entry times for late entry. When given, the result is a multi-state counting-process
        response.

    Returns
    -------
    Surv
        A multi-state response whose states are the endpoint names, in order.

    Warns
    -----
    UserWarning
        When a row's event is recorded after another endpoint's follow-up had already ended
        without an event. This often means the time columns don't share an origin or a unit.

    Examples
    --------
    Two endpoints for three subjects. Subject 1 has endpoint `"a"` at time 5, subject 2 has `"b"`
    at time 8, and subject 3 is censored at time 9:

    ```{python}
    import greenwood as gw

    gw.first_event(
        endpoints={
            "a": ([5.0, 9.0, 4.0], [1, 0, 0]),
            "b": ([7.0, 8.0, 9.0], [0, 1, 0]),
        }
    )
    ```

    For a data frame such as the bundled `mgus2` (one column pair per endpoint), name the columns
    with `Outcome.first_event()` and pass `data=` when fitting. PCM is listed first, so a PCM
    diagnosed at the same time as death counts as PCM:

    ```{python}
    mgus2 = gw.load_dataset("mgus2")

    cr = gw.Outcome.first_event(endpoints={"pcm": ("ptime", "pstat"), "death": ("futime", "death")})
    gw.AalenJohansen().fit(cr, data=mgus2)
    ```
    """
    if not endpoints:
        raise ValueError("`endpoints` must name at least one endpoint.")
    labels = [str(label) for label in endpoints]
    times: list[FloatArray] = []
    events: list[npt.NDArray[np.bool_]] = []
    for label, spec in endpoints.items():
        if isinstance(spec, str) or len(spec) != 2:  # pyright: ignore[reportUnnecessaryIsInstance]
            raise ValueError(f"Endpoint {label!r} must be a `(time, event)` pair.")
        time = _numeric(spec[0], f"The time of endpoint {label!r} is not numeric")
        event = _numeric_status(spec[1], f"The event of endpoint {label!r} must be logical or 0/1")
        if np.isnan(event).any():
            raise ValueError(f"The event of endpoint {label!r} must be logical or 0/1.")
        times.append(time)
        events.append(event == 1)
    n = times[0].shape[0]
    if any(t.shape[0] != n for t in times) or any(e.shape[0] != n for e in events):
        raise ValueError("All endpoint columns must have the same length.")

    time_matrix = np.column_stack(times)
    event_matrix = np.column_stack(events)
    # argmin picks the first column among equal times, so ties follow the endpoint order.
    event_times = np.where(event_matrix, time_matrix, np.inf)
    first = np.argmin(event_times, axis=1)
    observed = event_matrix.any(axis=1)
    if censor_at is not None:
        censor_time = _numeric(censor_at, "`censor_at` is not numeric")
    else:
        censor_time = time_matrix.max(axis=1)
    stop = np.where(observed, event_times[np.arange(n), first], censor_time)
    status = np.where(observed, first + 1, 0).astype(np.float64)
    _warn_event_after_follow_up(labels, time_matrix, event_matrix, observed, first, stop)
    if start is None:
        return Surv._from_fields(  # pyright: ignore[reportPrivateUsage]
            CensoringType.MRIGHT, stop=stop, status=status, states=tuple(labels)
        )
    return Surv._from_fields(  # pyright: ignore[reportPrivateUsage]
        CensoringType.MCOUNTING,
        stop=stop,
        status=status,
        start=_numeric(start, "`start` is not numeric"),
        states=tuple(labels),
    )


def _warn_event_after_follow_up(
    labels: list[str],
    times: Array,
    events: Array,
    observed: Array,
    first: Array,
    stop: Array,
) -> None:
    """Warn when an event is recorded after another endpoint's follow-up had already ended."""
    ended_early = (~events) & (times < stop[:, None])
    bad = observed & ended_early.any(axis=1)
    if not bad.any():
        return
    row = int(np.flatnonzero(bad)[0])
    other = int(np.flatnonzero(ended_early[row])[0])
    warnings.warn(
        f"{int(bad.sum())} row(s) have an event recorded after another endpoint's follow-up had "
        f"already ended. For example, row {row} has {labels[int(first[row])]!r} at "
        f"{stop[row]:g}, but its {labels[other]!r} follow-up stops at {times[row, other]:g}, so "
        f"{labels[other]!r} was not observed in between. Check that the endpoint time columns "
        "share an origin and a unit. If one endpoint was simply followed for less time, the "
        "response treats it as not having occurred before the recorded event.",
        UserWarning,
        stacklevel=3,
    )
