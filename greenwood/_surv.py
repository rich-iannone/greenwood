"""The `Surv` response object: the spine of every survival analysis.

`Surv` mirrors R's `Surv()`. It captures a time-to-event outcome in one of several
censoring flavors and validates it eagerly, so every downstream estimator can rely on a
clean, consistent representation.

Censoring types supported:

- **right** (`Surv.right`): the common case, `(time, event)`.
- **left** (`Surv.left`): left-censored `(time, event)`.
- **interval** (`Surv.interval`): the event is known only to lie in `(lower, upper]`;
  open bounds encode left/right censoring.
- **counting** (`Surv.counting`): the counting-process form `(start, stop, event]`, which
  also expresses **left truncation / late entry** and **time-varying covariates**.

Multi-state and competing-risks endpoints are expressed by an integer `status` with more
than one event code plus a `states` label tuple (`Surv.multistate`); the estimators that
consume them arrive in later phases, but construction and validation live here now.

Inputs may be any Narwhals-compatible series (pandas, Polars, ...), a NumPy array, or a
plain sequence; everything is coerced to NumPy internally.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from typing import Any

import narwhals as nw  # pyright: ignore[reportMissingImports]  # installed + typed; pyright quirk
import numpy as np
import numpy.typing as npt
from typing_extensions import Self

__all__ = ["Surv", "CensoringType"]

Array = npt.NDArray[Any]


class CensoringType(str, Enum):
    """The censoring flavor of a `Surv` response.

    Examples
    --------
    The members correspond to the `Surv` constructors. A constructed response reports its
    flavor through `Surv(...).type`, which is one of these values.

    ```{python}
    from greenwood import CensoringType

    list(CensoringType)
    ```
    """

    RIGHT = "right"
    LEFT = "left"
    INTERVAL = "interval"
    COUNTING = "counting"


def _to_1d_array(x: Any, *, dtype: Any = float) -> Array:
    """Coerce a Narwhals series, NumPy array, or sequence to a 1-D NumPy array."""
    if x is None:
        raise ValueError("Expected an array-like, got None.")
    if isinstance(x, (np.ndarray, list, tuple)):
        arr = np.asarray(x)
    else:
        # A Narwhals-native series (pandas, Polars, ...) or anything else array-like.
        try:
            arr = nw.from_native(x, series_only=True).to_numpy()
        except TypeError:
            arr = np.asarray(x)
    arr = np.asarray(arr, dtype=dtype)
    if arr.ndim != 1:
        raise ValueError(f"Expected a 1-D array-like, got shape {arr.shape}.")
    return arr


def _coerce_event(event: Any, n: int) -> Array:
    """Coerce an event indicator to an int status array (0 = censored, 1 = event).

    Accepts booleans or 0/1 integers. R's 1/2 coding (as in `survival::lung`) is *not*
    auto-detected; convert it explicitly, e.g. `event=(status == 2)`.
    """
    if event is None:
        return np.ones(n, dtype=np.int64)
    arr = _to_1d_array(event, dtype=float)
    if not np.all(np.isfinite(arr)):
        raise ValueError("Event indicator contains missing/non-finite values.")
    uniq = set(np.unique(arr).tolist())
    if not uniq <= {0.0, 1.0}:
        raise ValueError(
            "Event indicator must be boolean or 0/1. Got values "
            f"{sorted(uniq)}. R's 1/2 coding must be converted (e.g. event=(status == 2))."
        )
    return arr.astype(np.int64)


@dataclass(frozen=True)
class Surv:
    """A validated time-to-event response for survival analysis.

    `Surv` represents the outcome in survival models: a time at which each subject either
    experienced an event (observed) or was censored (did not experience the event during
    follow-up). `Surv` supports multiple censoring types:

    - **Right-censored** (most common): The event time is at or after the recorded time.
      Use `Surv.right(time, event)`.
    - **Left-censored**: The event time is before the recorded time.
      Use `Surv.left(time, event)`.
    - **Counting-process** (left truncation, time-varying): Each subject enters the risk set
      at `start` and exits at `stop`. Use `Surv.counting(start, stop, event)`.
    - **Interval-censored**: The event occurred within a time interval `[lower, upper)`.
      Use `Surv.interval(lower, upper)`.
    - **Multi-state / competing risks**: Multiple mutually exclusive events.
      Use `Surv.multistate(time, event, states)`.

    **Use the class methods** (`right`, `left`, `counting`, `interval`, `multistate`)
    **to construct Surv objects.** They validate your input and set the censoring type
    appropriately. Direct instantiation is not recommended.

    Attributes
    ----------
    type
        The `CensoringType` enum indicating the censoring mechanism.
    stop
        Exit time (for interval censoring, the upper bound).
    status
        Integer event code per observation: 0 = censored, 1+ = event code (for multi-state,
        codes >= 1 index into `states`).
    start
        Entry time for the counting-process form (left truncation); `None` otherwise.
    lower
        Lower bound for interval censoring; `None` otherwise.
    states
        Event-state labels for multi-state/competing-risks endpoints; `None` for the
        single-event case.
    weights
        Optional case weights (strictly positive); `None` if no weights provided.

    Examples
    --------
    Here's an example of direct instantiation of `Surv`:

    ```{python}
    from greenwood import Surv, CensoringType
    import numpy as np

    y = Surv(
        type=CensoringType.RIGHT,
        stop=np.array([5, 6, 4, 9]),
        status=np.array([1, 0, 1, 0])
    )
    y
    ```

    While this is fine, the preferred approach is to use the class method constructors for each
    censoring type. They handle validation and conversion automatically.

    Right-censored (the most common case): each subject has an exit time and an event
    indicator.

    ```{python}
    y = Surv.right(time=[5, 6, 4, 9], event=[1, 0, 1, 0])
    y
    ```

    Counting-process form with left truncation (late entry):

    ```{python}
    y = Surv.counting(start=[0, 2, 1], stop=[5, 6, 4], event=[1, 0, 1])
    y
    ```

    Interval-censored (event known to occur in a time window):

    ```{python}
    y = Surv.interval(lower=[1, 3], upper=[3, 8])
    y
    ```

    Multi-state (competing risks, multiple mutually exclusive events):

    ```{python}
    y = Surv.multistate(time=[5, 6, 4], event=[1, 0, 2], states=("pcm", "death"))
    y
    ```

    See Also
    --------
    Surv.right : Right-censored response constructor
    Surv.left : Left-censored response constructor
    Surv.counting : Counting-process response constructor (with left truncation)
    Surv.interval : Interval-censored response constructor
    Surv.multistate : Multi-state / competing-risks response constructor
    CensoringType : Enumeration of censoring types
    """

    type: CensoringType
    stop: Array
    status: Array
    start: Array | None = None
    lower: Array | None = None
    states: tuple[str, ...] | None = None
    weights: Array | None = None

    # -- validation -----------------------------------------------------------

    def __post_init__(self) -> None:
        n = self.stop.shape[0]
        if self.status.shape[0] != n:
            raise ValueError("`stop` and `status` must have the same length.")
        if not np.all(np.isfinite(self.stop)):
            raise ValueError("`stop` times must be finite.")
        if np.any(self.stop < 0):
            raise ValueError("`stop` times must be non-negative.")
        if np.any(self.status < 0):
            raise ValueError("`status` codes must be non-negative integers.")

        # For interval censoring, `status` encodes the censoring kind (0/1/2), not an
        # event state, so the state-count check does not apply.
        if self.type is not CensoringType.INTERVAL:
            n_states = 1 if self.states is None else len(self.states)
            if np.any(self.status > n_states):
                raise ValueError("A `status` code exceeds the number of event states.")

        if self.start is not None:
            if self.start.shape[0] != n:
                raise ValueError("`start` and `stop` must have the same length.")
            if not np.all(np.isfinite(self.start)):
                raise ValueError("`start` times must be finite.")
            if np.any(self.start >= self.stop):
                raise ValueError("Each `start` must be strictly less than its `stop`.")

        if self.lower is not None:
            if self.lower.shape[0] != n:
                raise ValueError("`lower` and `stop` must have the same length.")
            if np.any(self.lower > self.stop):
                raise ValueError("Each interval `lower` must be <= its `upper`.")

        if self.weights is not None:
            if self.weights.shape[0] != n:
                raise ValueError("`weights` and `stop` must have the same length.")
            if np.any(self.weights <= 0):
                raise ValueError("`weights` must be strictly positive.")

    # -- constructors ---------------------------------------------------------

    @classmethod
    def right(cls, time: Any, event: Any = None, *, weights: Any = None) -> Self:
        """Right-censored response `(time, event)`. `event=None` means all events.

        Parameters
        ----------
        time
            Array-like of exit times (one per subject). Must be finite and non-negative.
        event
            Array-like of event indicators (1 = event observed, 0 = censored). If `None`,
            all subjects are treated as having experienced the event.
        weights
            Optional array-like of case weights (strictly positive). One weight per subject;
            used in weighted survival analysis. Default is `None` (all weights = 1).

        Returns
        -------
        Surv
            A right-censored Surv response object.

        Examples
        --------
        The common case: each subject has an exit `time` and an `event` indicator (1 if the
        event was observed, 0 if censored).

        ```{python}
        Surv.right(time=[5, 6, 4, 9], event=[1, 0, 1, 0])
        ```

        The output shows 4 observations with 2 events and 2 censored. The `+` mark denotes
        censored observations (where the event was not observed within follow-up).
        """
        stop = _to_1d_array(time)
        status = _coerce_event(event, stop.shape[0])
        w = _to_1d_array(weights) if weights is not None else None
        return cls(type=CensoringType.RIGHT, stop=stop, status=status, weights=w)

    @classmethod
    def left(cls, time: Any, event: Any = None, *, weights: Any = None) -> Self:
        """Left-censored response `(time, event)`.

        Parameters
        ----------
        time
            Array-like of observation times. Must be finite and non-negative.
        event
            Array-like of event indicators (1 = event occurred before `time`, 0 = censored).
            If `None`, all subjects are treated as having experienced the event.
        weights
            Optional array-like of case weights (strictly positive). One weight per subject;
            used in weighted survival analysis. Default is `None` (all weights = 1).

        Returns
        -------
        Surv
            A left-censored Surv response object.

        Examples
        --------
        Left censoring marks subjects known to have had the event before the recorded
        `time` rather than after it.

        ```{python}
        Surv.left(time=[5, 6, 4], event=[1, 0, 1])
        ```

        The output shows 3 observations with 2 events and 1 censored. For the 2 subjects
        with events (event=1), the event occurred before their recorded time. The `+` mark
        indicates the censored subject (event=0).
        """
        stop = _to_1d_array(time)
        status = _coerce_event(event, stop.shape[0])
        w = _to_1d_array(weights) if weights is not None else None
        return cls(type=CensoringType.LEFT, stop=stop, status=status, weights=w)

    @classmethod
    def counting(cls, start: Any, stop: Any, event: Any = None, *, weights: Any = None) -> Self:
        """Counting-process response `(start, stop, event]` (left truncation / time-varying).

        Parameters
        ----------
        start
            Array-like of entry times (one per subject). Must be finite and non-negative.
            Represents when the subject enters the risk set (late entry / left truncation).
        stop
            Array-like of exit times (one per subject). Must be finite, non-negative, and
            strictly greater than the corresponding `start`.
        event
            Array-like of event indicators (1 = event observed, 0 = censored). If `None`,
            all subjects are treated as having experienced the event.
        weights
            Optional array-like of case weights (strictly positive). One weight per subject;
            used in weighted survival analysis. Default is `None` (all weights = 1).

        Returns
        -------
        Surv
            A counting-process Surv response object (with left truncation if applicable).

        Examples
        --------
        Each subject enters the risk set at `start` and exits at `stop`. This form expresses
        left truncation / late entry and time-varying covariates.

        ```{python}
        Surv.counting(start=[0, 2, 1], stop=[5, 6, 4], event=[1, 0, 1])
        ```

        The output shows 3 observations with 2 events and 1 censored, represented in the
        counting-process form. Subject 1 entered at time 0 and exited at time 5 with an event;
        subject 2 entered at time 2, exited at time 6, and was censored; subject 3 entered at
        time 1 and experienced an event at time 4.
        """
        start_a = _to_1d_array(start)
        stop_a = _to_1d_array(stop)
        status = _coerce_event(event, stop_a.shape[0])
        w = _to_1d_array(weights) if weights is not None else None
        return cls(
            type=CensoringType.COUNTING, stop=stop_a, status=status, start=start_a, weights=w
        )

    @classmethod
    def interval(cls, lower: Any, upper: Any, *, weights: Any = None) -> Self:
        """Interval-censored response; the event lies in `(lower, upper]`.

        Use `numpy.inf` for `upper` to mark right-censoring and `0` for `lower` to mark
        left-censoring. `lower == upper` denotes an exact observation.

        Parameters
        ----------
        lower
            Array-like of interval lower bounds (one per subject). Must be finite and
            non-negative. Set to 0 to mark left-censored subjects.
        upper
            Array-like of interval upper bounds (one per subject). Must be finite,
            non-negative, and >= `lower`. Set to `numpy.inf` to mark right-censored subjects.
        weights
            Optional array-like of case weights (strictly positive). One weight per subject;
            used in weighted survival analysis. Default is `None` (all weights = 1).

        Returns
        -------
        Surv
            An interval-censored Surv response object.

        Examples
        --------
        Each event is known only to lie in `(lower, upper]`. Here the second subject is
        right-censored (`upper` is infinite).

        ```{python}
        import numpy as np

        Surv.interval(lower=[1, 2, 3], upper=[2, np.inf, 5])
        ```

        The output shows 3 observations: subject 1 had an exact event at time 2 (lower=upper);
        subject 2 was right-censored at time 2 (upper=infinity means the event happened after
        time 2); subject 3 had an interval-censored event between times 3 and 5.
        """
        lower_a = _to_1d_array(lower)
        upper_a = _to_1d_array(upper)
        if lower_a.shape[0] != upper_a.shape[0]:
            raise ValueError("`lower` and `upper` must have the same length.")
        # status: 1 = exact event, 0 = right-censored (upper = inf), 2 = interval.
        status = np.where(~np.isfinite(upper_a), 0, np.where(lower_a == upper_a, 1, 2)).astype(
            np.int64
        )
        finite_upper = np.where(np.isfinite(upper_a), upper_a, lower_a)
        w = _to_1d_array(weights) if weights is not None else None
        return cls(
            type=CensoringType.INTERVAL,
            stop=finite_upper,
            status=status,
            lower=lower_a,
            weights=w,
        )

    @classmethod
    def multistate(
        cls,
        time: Any,
        event: Any,
        states: tuple[str, ...],
        *,
        start: Any = None,
        weights: Any = None,
    ) -> Self:
        """Multi-state / competing-risks response.

        `event` holds integer codes (0 = censored, `k` = transition to `states[k-1]`).

        Parameters
        ----------
        time
            Array-like of event/censoring times (one per subject). Must be finite and
            non-negative.
        event
            Array-like of event codes (integer, one per subject): 0 = censored, 1 = transition
            to `states[0]`, 2 = transition to `states[1]`, etc. Must be in the range
            [0, len(states)].
        states
            Tuple of state labels (strings). Event codes in `event` index into this tuple.
            For example, `states=("relapse", "death")` means event code 1 → relapse,
            event code 2 → death.
        start
            Optional array-like of entry times (for late entry / left truncation).
            If `None` (default), all subjects enter at time 0. Must be non-negative and
            strictly less than `time`.
        weights
            Optional array-like of case weights (strictly positive). One weight per subject;
            used in weighted survival analysis. Default is `None` (all weights = 1).

        Returns
        -------
        Surv
            A multi-state / competing-risks Surv response object.

        Examples
        --------
        The `states` labels name the transitions; `event` codes index into them (1 =
        `states[0]`, 2 = `states[1]`, and so on), with 0 for censored.

        ```{python}
        Surv.multistate(time=[5, 6, 7, 8], event=[1, 2, 0, 1], states=("relapse", "death"))
        ```

        The output shows 4 observations with 3 events and 1 censored. Subject 1 transitioned
        to "relapse" (event code 1) at time 5; subject 2 transitioned to "death" (event code 2)
        at time 6; subject 3 was censored (event code 0); subject 4 transitioned to "relapse"
        at time 8. The two competing event types are mutually exclusive—each subject can only
        experience one.
        """
        stop = _to_1d_array(time)
        status = _to_1d_array(event, dtype=int).astype(np.int64)
        start_a = _to_1d_array(start) if start is not None else None
        w = _to_1d_array(weights) if weights is not None else None
        ctype = CensoringType.COUNTING if start is not None else CensoringType.RIGHT
        return cls(
            type=ctype, stop=stop, status=status, start=start_a, states=tuple(states), weights=w
        )

    # -- derived views used by the kernel -------------------------------------

    @property
    def n(self) -> int:
        """Number of observations."""
        return int(self.stop.shape[0])

    @property
    def entry(self) -> Array:
        """Entry times; `-inf` where there is no left truncation."""
        if self.start is not None:
            return self.start
        return np.full(self.n, -np.inf)

    @property
    def event(self) -> Array:
        """Boolean "any event occurred" indicator (`status >= 1`)."""
        return self.status >= 1

    @property
    def is_truncated(self) -> bool:
        """Whether the response carries left-truncation entry times."""
        return self.start is not None

    @property
    def is_multistate(self) -> bool:
        """Whether the response has more than one event state."""
        return self.states is not None

    @property
    def n_events(self) -> int:
        """Count of observations with an event (any state)."""
        return int(np.count_nonzero(self.event))

    @property
    def n_censored(self) -> int:
        """Count of censored observations."""
        return self.n - self.n_events

    def __len__(self) -> int:
        return self.n

    def __repr__(self) -> str:
        extra = ""
        if self.is_truncated:
            extra += ", truncated"
        if self.is_multistate:
            extra += f", states={self.states}"
        return f"Surv(type={self.type.value}, n={self.n}, events={self.n_events}{extra})"

    # -- interop --------------------------------------------------------------

    def to_pandas(self) -> Any:
        """Return the response as a pandas DataFrame (one row per observation).

        This method exports the Surv object to a tidy pandas DataFrame format, where
        each row represents one observation. The DataFrame includes the `stop` and
        `status` columns, plus optional columns for `start` (entry time in counting
        process), `lower` (lower bound for interval censoring), and `weight` (case
        weights). This format is convenient for inspection, export to CSV or other
        file formats, or integration with other pandas workflows.

        Returns
        -------
        pandas.DataFrame
            A tidy DataFrame with one row per observation, including columns for `stop`,
            `status`, and optional `start`, `lower`, `weight` columns.

        Raises
        ------
        ImportError
            If pandas is not installed.

        Examples
        --------
        Export to pandas DataFrame. Each row represents one observation with its
        event time and status:

        ```{python}
        y.to_pandas()
        ```

        The resulting DataFrame can be saved to CSV, used with pandas functions, or
        integrated into standard data science workflows.
        """
        try:
            import pandas as pd
        except ImportError as e:
            raise ImportError(
                "pandas is required for to_pandas(). Install it with: pip install pandas"
            ) from e

        cols: dict[str, Array] = {}
        if self.start is not None:
            cols["start"] = self.start
        if self.lower is not None:
            cols["lower"] = self.lower
        cols["stop"] = self.stop
        cols["status"] = self.status
        if self.weights is not None:
            cols["weight"] = self.weights
        return pd.DataFrame(cols)

    def to_polars(self) -> Any:
        """Return the response as a Polars DataFrame (one row per observation).

        This method exports the Surv object to a tidy Polars DataFrame format, where
        each row represents one observation. Polars provides superior performance and
        memory efficiency compared to pandas for larger datasets. The DataFrame includes
        the `stop` and `status` columns, plus optional columns for `start` (entry time
        in counting process), `lower` (lower bound for interval censoring), and `weight`
        (case weights). This format is ideal for efficient data manipulation and
        integration with Polars-based workflows.

        Returns
        -------
        polars.DataFrame
            A tidy DataFrame with one row per observation, including columns for `stop`,
            `status`, and optional `start`, `lower`, `weight` columns.

        Raises
        ------
        ImportError
            If polars is not installed.

        Examples
        --------
        Export to Polars DataFrame for high-performance data processing:

        ```{python}
        y.to_polars()
        ```

        Polars DataFrames are efficient and support lazy evaluation, making them ideal
        for larger datasets and complex transformations.
        """
        try:
            import polars as pl
        except ImportError as e:
            raise ImportError(
                "polars is required for to_polars(). Install it with: pip install polars"
            ) from e

        cols: dict[str, Array] = {}
        if self.start is not None:
            cols["start"] = self.start
        if self.lower is not None:
            cols["lower"] = self.lower
        cols["stop"] = self.stop
        cols["status"] = self.status
        if self.weights is not None:
            cols["weight"] = self.weights
        return pl.DataFrame(cols)

    def to_arrow(self) -> Any:
        """Return the response as a PyArrow Table (one row per observation).

        This method exports the Surv object to a PyArrow Table, a columnar data structure
        designed for efficient data interchange and analytics. PyArrow Tables are ideal
        for integration with tools that work with the Apache Arrow memory format, including
        Polars, DuckDB, and many other data processing libraries. The table includes the
        `stop` and `status` columns, plus optional columns for `start` (entry time in
        counting process), `lower` (lower bound for interval censoring), and `weight`
        (case weights).

        Returns
        -------
        pyarrow.Table
            A table with one row per observation, including columns for `stop`,
            `status`, and optional `start`, `lower`, `weight` columns.

        Raises
        ------
        ImportError
            If pyarrow is not installed.

        Examples
        --------
        Export to PyArrow Table for interoperability with Arrow-based tools:

        ```{python}
        y.to_arrow()
        ```

        PyArrow Tables are the standard format for efficient data interchange between
        different tools and libraries in the modern data stack.
        """
        try:
            import pyarrow as pa
        except ImportError as e:
            raise ImportError(
                "pyarrow is required for to_arrow(). Install it with: pip install pyarrow"
            ) from e

        cols: dict[str, Any] = {}
        if self.start is not None:
            cols["start"] = self.start
        if self.lower is not None:
            cols["lower"] = self.lower
        cols["stop"] = self.stop
        cols["status"] = self.status
        if self.weights is not None:
            cols["weight"] = self.weights
        return pa.table(cols)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-ready mapping fully describing the response.

        This method serializes the entire Surv object into a plain Python dictionary,
        making it suitable for JSON serialization, storage, or transmission. All array
        data is converted to plain Python lists. The dictionary captures the censoring
        type and every array (time, status, optional fields), with `None` for fields
        that a given censoring flavor does not use.

        Returns
        -------
        dict[str, Any]
            A dictionary with keys: `type` (CensoringType as string), `stop`, `status`,
            and optional keys `start`, `lower`, `states`, `weights` (as lists or None).

        Examples
        --------
        The mapping structure varies by censoring type, but always includes `type`,
        `stop`, and `status`. Unused fields are `None`. Here we convert `y` to a
        dictionary:

        ```{python}
        y.to_dict()
        ```

        This is the serialized form that underpins `to_json()` and enables
        round-tripping via `from_dict()`.
        """

        def _list(a: Array | None) -> list[Any] | None:
            return None if a is None else a.tolist()

        return {
            "type": self.type.value,
            "stop": self.stop.tolist(),
            "status": self.status.tolist(),
            "start": _list(self.start),
            "lower": _list(self.lower),
            "states": list(self.states) if self.states is not None else None,
            "weights": _list(self.weights),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        """Rebuild a response from `to_dict` output.

        This is the inverse of `to_dict()`: it reconstructs an equivalent Surv object
        from a dictionary previously created by `to_dict()`. Useful for deserializing
        stored or transmitted data, or for round-tripping through storage formats.

        Parameters
        ----------
        data : dict[str, Any]
            A dictionary produced by `to_dict()` containing keys `type`, `stop`, `status`,
            and optional keys for `start`, `lower`, `states`, `weights`.

        Returns
        -------
        Surv
            A new Surv object with the same data and structure as the input dictionary.

        Examples
        --------
        Rebuild an equivalent response from its dictionary representation. Here we
        serialize `y` and immediately deserialize it:

        ```{python}
        reconstructed = gw.Surv.from_dict(y.to_dict())
        print("Objects equal:", y.to_dict() == reconstructed.to_dict())
        ```

        The reconstructed object is equivalent to the original in every way.
        """

        def _arr(key: str, dtype: Any = float) -> Array | None:
            value = data.get(key)
            return None if value is None else np.asarray(value, dtype=dtype)

        return cls(
            type=CensoringType(data["type"]),
            stop=np.asarray(data["stop"], dtype=float),
            status=np.asarray(data["status"], dtype=np.int64),
            start=_arr("start"),
            lower=_arr("lower"),
            states=tuple(data["states"]) if data.get("states") is not None else None,
            weights=_arr("weights"),
        )

    def to_json(self, *, indent: int | None = 2) -> str:
        """Serialize to a deterministic JSON string.

        This method converts the entire Surv object to a compact, JSON-formatted string
        suitable for storage in files, databases, or transmission over APIs. By default,
        the output is human-readable with indentation; pass `indent=None` for a compact
        form. The serialization is deterministic: the same Surv object always produces
        the identical JSON string.

        Parameters
        ----------
        indent : int | None, optional
            Number of spaces to use for indentation. If `None`, produces compact JSON
            without whitespace. Default is 2 (human-readable).

        Returns
        -------
        str
            A JSON string representing the Surv object, including censoring type and
            all arrays.

        Examples
        --------
        Serialize to JSON. By default, output is indented for readability. Here we show
        just the first 120 characters of compact JSON:

        ```{python}
        json_compact = y.to_json(indent=None)
        print(json_compact[:120])
        ```

        The full JSON includes all data in a structured format that can be parsed back
        with `from_json()`.
        """
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_json(cls, text: str) -> Self:
        """Deserialize from `to_json` output.

        This is the inverse of `to_json()`: it reconstructs a Surv object from a JSON
        string previously created by `to_json()`. Useful for loading data from stored
        files, API responses, or any other JSON source. The reconstructed object is
        guaranteed to be equivalent to the original.

        Parameters
        ----------
        text : str
            A JSON string produced by `to_json()` containing the serialized Surv data.

        Returns
        -------
        Surv
            A new Surv object restored from the JSON representation.

        Examples
        --------
        Deserialize from JSON. Round-trip through `to_json()` and back:

        ```{python}
        json_text = y.to_json()
        restored = gw.Surv.from_json(json_text)
        print("Round-trip successful:", y.to_json() == restored.to_json())
        ```

        The restored object is an exact copy of the original Surv object.
        """
        return cls.from_dict(json.loads(text))
