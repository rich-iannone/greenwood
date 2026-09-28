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
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np
import numpy.typing as npt
from typing_extensions import Self

from ._backends import to_dataframe
from ._ingest import as_1d, encode_event, encode_states, resolve_columns

__all__ = ["Surv", "CensoringType"]

Array = npt.NDArray[Any]


class CensoringType(str, Enum):
    """The censoring mechanism of a `Surv` response.

    Each member corresponds to one of the `Surv` constructors and describes how incomplete
    observations are handled:

    - `RIGHT`: Classic right-censoring. All subjects enter at time 0, and some are lost before the
    event occurs (`Surv.right()`).
    - `LEFT`: Left-censoring. The event is known to have occurred before the observation time
    (`Surv.left()`).
    - `INTERVAL`: Interval-censoring. The event is known to lie within a time bracket
    (`Surv.interval()`).
    - `COUNTING`: Counting-process (start, stop] intervals with possible late entry and time-varying
    covariates (`Surv.counting()`).

    A constructed response reports its censoring type through `Surv(...).type`, which is one of
    these values.

    Examples
    --------
    List all available censoring types:

    ```{python}
    from greenwood import CensoringType

    # List all supported censoring mechanisms
    list(CensoringType)
    ```
    """

    RIGHT = "right"
    LEFT = "left"
    INTERVAL = "interval"
    COUNTING = "counting"


def _to_1d_array(x: Any, *, dtype: Any = float) -> Array:
    """Coerce a Narwhals series, NumPy array, or sequence to a 1-D NumPy array."""
    return np.asarray(as_1d(x), dtype=dtype)


def _coerce_event(
    event: Any, n: int, *, event_value: Any = None, censor_value: Any = None
) -> Array:
    """Coerce an event indicator to an int status array (`0` = censored, `1` = event).

    Accepts booleans or `0`/`1` integers as they are. Any other coding (R's `1`/`2`, strings, a
    compound status) needs `event_value=` or `censor_value=` to say which rows are events.
    """
    return encode_event(event, n, event_value=event_value, censor_value=censor_value)


@dataclass(frozen=True)
class Surv:
    """A validated time-to-event response for survival analysis.

    `Surv` represents the outcome in survival models: a time at which each subject either
    experienced an event (observed) or was censored (did not experience the event during follow-up).
    Build one with the class method for its censoring type:

    - **Right-censored** (most common): the event time is at or after the recorded time. Use
    `Surv.right()`.
    - **Left-censored**: the event time is before the recorded time. Use `Surv.left()`.
    - **Counting-process** (left truncation, time-varying covariates): each row enters the risk set
    at `start` and exits at `stop`. Use `Surv.counting()`.
    - **Interval-censored**: the event occurred within a time interval `[lower, upper)`. Use
    `Surv.interval()`.
    - **Multi-state / competing risks**: several mutually exclusive events. Use
    `Surv.multistate()`, or `Surv.first_event()` when each endpoint has its own pair of columns.

    The usual way to call them is with column names and the frame that holds them, as in
    `Surv.right(time="time", event="status", data=lung, event_value=2)`. The event coding is
    declared as a value (`event_value=` or `censor_value=`) rather than written as a data frame
    expression, so the same call works on pandas, Polars, PyArrow, DuckDB, and lazy frames. Every
    constructor also accepts plain lists or arrays, which suit small examples and values computed
    by hand. To describe a response once and bind it to data when a model is fit (dropping
    incomplete rows together with the covariates), use `Outcome` instead.

    The class methods validate their input and set the censoring type. Direct instantiation with
    the parameters below is possible but rarely needed.

    Parameters
    ----------
    type
        The `CensoringType` of the response: `RIGHT`, `LEFT`, `COUNTING`, or `INTERVAL`.
    stop
        Exit time of each observation (for interval censoring, the upper bound).
    status
        Integer event code per observation: `0` = censored, `1` or more = event. For a multi-state
        response, code `k` is the `k`-th entry of `states`. For interval censoring, `status` holds
        the kind of observation (`0` right-censored, `1` exact, `2` interval).
    start
        Entry time of each observation, for the counting-process form (left truncation). `None`
        otherwise.
    lower
        Lower bound of each interval, for interval censoring. `None` otherwise.
    states
        Event-state labels for a multi-state or competing-risks response. `None` for a single
        event type.
    weights
        Case weights (strictly positive), or `None` for unit weights.

    Notes
    -----
    Each parameter is stored as an attribute of the same name. Derived views such as `n`,
    `n_events`, `n_censored`, `entry`, `event`, and `is_truncated` are listed under Attributes
    below.

    Examples
    --------
    Build a right-censored response from a data frame. In the `lung` dataset, a `status` of `2`
    means the patient died:

    ```{python}
    import greenwood as gw

    lung = gw.load_dataset("lung")

    # Name the columns, and say which status value marks the event
    y = gw.Surv.right(time="time", event="status", data=lung, event_value=2)
    y
    ```

    A multi-state response maps each state to the value that marks it. In `pbc`, `status` is `0`
    for censored, `1` for transplant, and `2` for death:

    ```{python}
    pbc = gw.load_dataset("pbc")

    # Map each competing outcome to its code, and 0 is censored
    gw.Surv.multistate(time="time", event="status", data=pbc, states={"transplant": 1, "death": 2})
    ```

    When each endpoint has its own time and event columns, as in `mgus2`, `first_event()` combines
    them. The first observed endpoint becomes the cause:

    ```{python}
    mgus2 = gw.load_dataset("mgus2")

    # One (time, event) column pair per endpoint
    gw.Surv.first_event(
        endpoints={"pcm": ("ptime", "pstat"), "death": ("futime", "death")}, data=mgus2
    )
    ```

    Counting-process and interval-censored responses name their columns the same way:

    ```{python}
    import numpy as np
    import polars as pl

    visits = pl.DataFrame({"tstart": [0.0, 2.0, 1.0], "tstop": [5.0, 6.0, 4.0], "died": [1, 0, 1]})
    windows = pl.DataFrame({"last_negative": [1.0, 3.0], "first_positive": [3.0, np.inf]})

    # Late entry: each row is at risk from tstart to tstop
    print(gw.Surv.counting(start="tstart", stop="tstop", event="died", data=visits))

    # The event happened somewhere in each window, and inf marks right censoring
    print(gw.Surv.interval(lower="last_negative", upper="first_positive", data=windows))
    ```

    For small examples, pass the values directly as lists or arrays:

    ```{python}
    # Four subjects: events at 5 and 4, censored at 6 and 9
    gw.Surv.right(time=[5, 6, 4, 9], event=[1, 0, 1, 0])
    ```

    Direct instantiation skips the constructors' coercion and conveniences, so it expects NumPy
    arrays with the internal coding described under Parameters:

    ```{python}
    # Build the same response from its fields
    gw.Surv(type=gw.CensoringType.RIGHT, stop=np.array([5, 6, 4, 9]), status=np.array([1, 0, 1, 0]))
    ```
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
    def right(
        cls,
        time: Any,
        event: Any = None,
        *,
        weights: Any = None,
        data: Any = None,
        event_value: Any = None,
        censor_value: Any = None,
    ) -> Self:
        """Right-censored response: the standard and most common form of survival data.

        Right censoring is the default in survival analysis. It occurs when follow-up ends before
        the event happens (a subject is still event-free when we last observed them). This is the
        most common censoring mechanism in practice:

        - **Study ends**: A clinical trial concludes while some patients are still healthy
        - **Loss to follow-up**: A subject drops out, moves away, or stops visiting the clinic
        - **Administrative censoring**: Follow-up ends at a fixed time regardless of status

        Right censoring is called "censoring from the right" because we know the event happened
        *after* the censoring time. We record that a subject was event-free at their last
        observation but don't know how much longer they could have gone.

        This simple form assumes all subjects enter follow-up at the same reference time (typically
        time 0). If subjects enter at different times or follow-up is complex, use `counting()` or
        `interval()` instead.

        Parameters
        ----------
        time
            Exit times when follow-up ends (one per subject), or the name of a column in `data`.
            Must be finite and non-negative. This is the time of either the event or censoring,
            whichever came first.
        event
            Event indicators, or the name of a column in `data`. Without `event_value` or
            `censor_value`, the values must be boolean or `0`/`1`:

            - `1` = event occurred (fully observed)
            - `0` = censored (event time unknown but > time)

            If `None`, all subjects are treated as having experienced the event (useful for testing
            or descriptive purposes).
        weights
            Case weights (strictly positive, one per subject), or the name of a column in `data`.
            Used to weight subjects differently in survival analysis (e.g., inverse probability
            weighting). The default is `None` (all weights are `1`).
        data
            A data frame to look up column names in. Any Narwhals-compatible frame works (pandas,
            Polars, PyArrow, a Polars `LazyFrame`, DuckDB, ...), as does a plain mapping of column
            names to values. Only the named columns are read.
        event_value
            The value (or list of values) in `event` that marks an event. Every other row is
            censored. For example, `event_value=2` for R's 1/2 coding, `event_value="died"` for a
            string outcome, or `event_value=[1, 2]` to count two outcomes as the event.
        censor_value
            The value (or list of values) in `event` that marks censoring. Every other row is an
            event. For example, `censor_value=0` treats any nonzero status as an event. Pass this or
            `event_value`, not both.

        Returns
        -------
        Surv
            A right-censored `Surv` response object (the most common type).

        Examples
        --------
        Build a right-censored response from a data frame by naming its columns. In the `lung`
        dataset, a `status` of `2` means the patient died (R's 1/2 coding), so `event_value=2`
        says which rows are events:

        ```{python}
        import greenwood as gw

        lung = gw.load_dataset("lung")

        # Name the columns, and say which status value marks a death
        y = gw.Surv.right(time="time", event="status", data=lung, event_value=2)
        y
        ```

        The same call works for any data frame backend (pandas, Polars, PyArrow, DuckDB), because
        the encoding is a value rather than a data frame expression. When several values count as
        an event, name the censoring value instead. In `pbc`, `status` is `0` (censored), `1`
        (transplant), or `2` (death), so an all-cause endpoint is:

        ```{python}
        pbc = gw.load_dataset("pbc")

        # Anything other than 0 is an event
        gw.Surv.right(time="time", event="status", data=pbc, censor_value=0)
        ```

        For small examples, pass the values directly. Here two subjects have events (at times 5
        and 4) and two are censored (at 6 and 9):

        ```{python}
        gw.Surv.right(time=[5, 6, 4, 9], event=[1, 0, 1, 0])
        ```
        """
        cols = resolve_columns(data, {"time": time, "event": event, "weights": weights})
        stop = _to_1d_array(cols["time"])
        status = _coerce_event(
            cols["event"], stop.shape[0], event_value=event_value, censor_value=censor_value
        )
        w = _to_1d_array(cols["weights"]) if cols["weights"] is not None else None
        return cls(type=CensoringType.RIGHT, stop=stop, status=status, weights=w)

    @classmethod
    def left(
        cls,
        time: Any,
        event: Any = None,
        *,
        weights: Any = None,
        data: Any = None,
        event_value: Any = None,
        censor_value: Any = None,
    ) -> Self:
        """Left-censored response: event occurred before the observation time.

        Left censoring occurs when all you know is that an event happened *before* you observed the
        subject. For example, an infection that must have occurred before a patient was tested, or a
        failure that was known to have happened sometime before equipment was inspected. The exact
        event time is unknown, but you know it was no later than the recorded `time`.

        This is less common than right censoring, but important in scenarios where you cannot
        pinpoint when something happened, only that it already had.

        Parameters
        ----------
        time
            Observation times (the upper bound on when the event occurred), or the name of a
            column in `data`. Must be finite and non-negative. Each value represents "the event
            happened by this time".
        event
            Event indicators, or the name of a column in `data`. Without `event_value` or
            `censor_value`, the values must be boolean or `0`/`1`:

            - `1` = event occurred before `time` (left-censored)
            - `0` = subject was event-free at `time` (not censored)

            If `None`, all subjects are treated as having experienced the event.
        weights
            Case weights (strictly positive, one per subject), or the name of a column in `data`.
            Default is `None` (all weights are `1`).
        data
            A data frame to look up column names in. Any of `time`, `event`, and `weights` may be a
            column name. See `Surv.right()` for the accepted frame types.
        event_value
            The value (or list of values) in `event` that marks an event (one that occurred by
            `time`). Every other row is event-free at `time`. See `Surv.right()`.
        censor_value
            The value (or list of values) in `event` that marks an event-free row. Every other row
            is an event. Pass this or `event_value`, not both.

        Returns
        -------
        Surv
            A left-censored `Surv` response object.

        Examples
        --------
        Left censoring arises when a test shows the event has already happened, without saying
        when. Here each subject is tested once. A positive test means the event occurred by
        `test_day`, and a negative test means the subject was still event-free then:

        ```{python}
        import greenwood as gw
        import polars as pl

        screening = pl.DataFrame({
            "test_day": [5.0, 6.0, 4.0, 9.0],
            "result": ["positive", "negative", "positive", "negative"],
        })

        # A positive result marks an event that happened by test_day
        gw.Surv.left(time="test_day", event="result", data=screening, event_value="positive")
        ```

        For small examples, pass the values directly:

        ```{python}
        gw.Surv.left(time=[5, 6, 4], event=[1, 0, 1])
        ```
        """
        cols = resolve_columns(data, {"time": time, "event": event, "weights": weights})
        stop = _to_1d_array(cols["time"])
        status = _coerce_event(
            cols["event"], stop.shape[0], event_value=event_value, censor_value=censor_value
        )
        w = _to_1d_array(cols["weights"]) if cols["weights"] is not None else None
        return cls(type=CensoringType.LEFT, stop=stop, status=status, weights=w)

    @classmethod
    def counting(
        cls,
        start: Any,
        stop: Any,
        event: Any = None,
        *,
        weights: Any = None,
        data: Any = None,
        event_value: Any = None,
        censor_value: Any = None,
    ) -> Self:
        """Counting-process response: each row enters and leaves the risk set at its own times.

        The counting-process form handles two important real-world complexities:

        1. **Late entry (left truncation)**: Not all subjects start being at risk at time 0.
           For example, a study might enroll subjects at different ages, or you might analyze
           a subset of follow-up time after some subjects are already older. The `start` time
           marks when each subject becomes eligible to experience the event.

        2. **Time-varying covariates**: The counting-process form naturally accommodates
           covariates that change over time. Each row represents one interval of time for
           a subject, allowing you to track how covariate values change.

        Each subject contributes one or more (start, stop] intervals. The subject is at risk
        only during their interval(s) and cannot experience the event before entering at `start`.

        Parameters
        ----------
        start
            Entry times (when each row becomes at risk), or the name of a column in `data`. Must be
            finite and non-negative. In standard studies this is `0`. In studies with late entry,
            it is the age or time at enrollment.
        stop
            Exit times (when follow-up for the row ends), or the name of a column in `data`. Must
            be finite, non-negative, and strictly greater than the corresponding `start`.
        event
            Event indicators, or the name of a column in `data`. Without `event_value` or
            `censor_value`, the values must be boolean or `0`/`1` (`1` = event at `stop`,
            `0` = censored at `stop`). If `None`, every row is treated as ending in an event.
        weights
            Case weights (strictly positive, one per row), or the name of a column in `data`.
            Default is `None` (all weights are `1`).
        data
            A data frame to look up column names in. Any of `start`, `stop`, `event`, and
            `weights` may be a column name. See `Surv.right()` for the accepted frame types.
        event_value
            The value (or list of values) in `event` that marks an event. Every other row is
            censored. See `Surv.right()`.
        censor_value
            The value (or list of values) in `event` that marks censoring. Every other row is an
            event. Pass this or `event_value`, not both.

        Returns
        -------
        Surv
            A counting-process `Surv` response object with potential left truncation.

        Examples
        --------
        The counting-process form is what `split_episodes()` produces from repeated measurements:
        one row per `(tstart, tstop]` interval, with the event on each subject's last interval.
        In `pbcseq`, a `status` of `2` means death:

        ```{python}
        import greenwood as gw

        pbcseq = gw.load_dataset("pbcseq", backend="pandas")

        # One row per interval between lab visits
        long = gw.split_episodes(
            baseline=pbcseq, visits=pbcseq, id="id", time="futime", event="status",
            visit_time="day", covariates=["bili"], format="pandas",
        )

        # Name the interval columns, and say which status value marks a death
        gw.Surv.counting(start="tstart", stop="tstop", event="status", data=long, event_value=2)
        ```

        The same form expresses late entry (left truncation), where subjects join the risk set
        after time 0. For small examples, pass the values directly. Here subject 1 enters at 0 and
        has an event at 5, subject 2 enters late at 2 and is censored at 6, and subject 3 enters
        at 1 and has an event at 4:

        ```{python}
        gw.Surv.counting(start=[0, 2, 1], stop=[5, 6, 4], event=[1, 0, 1])
        ```
        """
        cols = resolve_columns(
            data, {"start": start, "stop": stop, "event": event, "weights": weights}
        )
        start_a = _to_1d_array(cols["start"])
        stop_a = _to_1d_array(cols["stop"])
        status = _coerce_event(
            cols["event"], stop_a.shape[0], event_value=event_value, censor_value=censor_value
        )
        w = _to_1d_array(cols["weights"]) if cols["weights"] is not None else None
        return cls(
            type=CensoringType.COUNTING, stop=stop_a, status=status, start=start_a, weights=w
        )

    @classmethod
    def interval(cls, lower: Any, upper: Any, *, weights: Any = None, data: Any = None) -> Self:
        """Interval-censored response: event time is known to lie within a range.

        Interval censoring occurs when you know the event happened sometime between two observation
        times, but not exactly when. Common in:

        - **Medical follow-up**: Disease detection between clinic visits. You might know a patient's
        disease status at two checkups, but not the exact time of onset.
        - **Equipment reliability**: Failure detected between inspections. You know failure happened
        between the last working inspection and the current failed one.
        - **Longitudinal surveys**: Event reported between survey waves but exact timing unknown.

        The interval-censored form captures this uncertainty. The event happened somewhere in the
        interval (lower, upper]. If lower == upper, it's an exact (uncensored) event. Use
        `numpy.inf` for upper to represent right-censoring, and `0` for lower to represent
        left-censoring.

        Parameters
        ----------
        lower
            Interval lower bounds (one per subject), or the name of a column in `data`. Must be
            finite and non-negative. The event happened *after* this time (possibly at this time).
            Set to `0` to mark left-censored subjects (event happened before first observation).
        upper
            Interval upper bounds (one per subject), or the name of a column in `data`. Must be
            non-negative and >= `lower`. The event happened *by* this time. Use `numpy.inf` to mark
            right-censored subjects (no event observed by the end of the study), and set it equal
            to `lower` for an exactly observed event.
        weights
            Case weights (strictly positive, one per subject), or the name of a column in `data`.
            Default is `None` (all weights are `1`).
        data
            A data frame to look up column names in. Any of `lower`, `upper`, and `weights` may be
            a column name. See `Surv.right()` for the accepted frame types.

        Returns
        -------
        Surv
            An interval-censored `Surv` response object.

        Examples
        --------
        Interval censoring arises with periodic inspections: the event is known only to have
        happened between the last visit where it was absent and the first where it was present.
        Here the columns hold those two visit times, with `inf` for a subject whose event was
        never seen:

        ```{python}
        import greenwood as gw
        import numpy as np
        import polars as pl

        inspections = pl.DataFrame({
            "last_negative": [1.0, 2.0, 3.0],
            "first_positive": [2.0, np.inf, 5.0],
        })

        # The event happened somewhere in (last_negative, first_positive]
        gw.Surv.interval(lower="last_negative", upper="first_positive", data=inspections)
        ```

        Subject 1's event fell between times 1 and 2, subject 2 was still event-free at the last
        visit (time 2, so right-censored), and subject 3's event fell between 3 and 5. For small
        examples, pass the values directly, with `lower == upper` for an exactly observed event:

        ```{python}
        gw.Surv.interval(lower=[1, 2, 3], upper=[1, np.inf, 5])
        ```
        """
        cols = resolve_columns(data, {"lower": lower, "upper": upper, "weights": weights})
        weights = cols["weights"]
        lower_a = _to_1d_array(cols["lower"])
        upper_a = _to_1d_array(cols["upper"])
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
        states: Sequence[str] | Mapping[str, Any],
        *,
        start: Any = None,
        weights: Any = None,
        data: Any = None,
        censor_value: Any = None,
    ) -> Self:
        """Multi-state or competing-risks response: track which of multiple outcomes occurs.

        Real-world studies often involve multiple competing outcomes. A patient in a cancer
        study might relapse, die from cancer, or die from other causes. Each subject can only
        experience one outcome, and once it happens, no other outcome is possible.

        The multi-state framework elegantly handles this by:

        1. **Defining possible states**: You specify the labeled outcomes (e.g., "relapse",
           "death from cancer", "death from other causes") that are mutually exclusive.
        2. **Recording which state occurred**: Rather than a simple 0/1 event, you record
           which specific state the subject transitioned to (or 0 if censored).
        3. **Separate risk estimation**: You can estimate the risk of each state independently,
           accounting for the fact that other states prevent each outcome.

        This is essential for realistic survival modeling: accounting for competing risks often
        substantially changes the estimated risk curves compared to treating all non-events
        identically.

        Parameters
        ----------
        time
            Event or censoring times (one per subject), or the name of a column in `data`. Must be
            finite and non-negative.
        event
            The outcome of each subject, or the name of a column in `data`. With a `states`
            mapping, the column may use any coding (numbers or strings), and `censor_value` marks
            censoring. With a sequence of `states`, it must hold integer codes: `0` for censored,
            `1` for `states[0]`, `2` for `states[1]`, and so on.
        states
            The possible outcomes, given in one of two ways.

            A mapping of label to the `event` value (or list of values) that marks it, such as
            `states={"relapse": 1, "death": 2}` or `states={"relapse": "rel", "death": "dth"}`.
            The column can then use any coding. Rows matching `censor_value` are censored, and any
            value not listed raises an error.

            A sequence of labels, such as `states=("relapse", "death")`. The `event` column must
            then hold integer codes that index into it:

            - event code 1 → relapse occurred
            - event code 2 → death occurred

            Labels are arbitrary strings describing what the transition represents.
        start
            Entry times for late entry (left truncation), or the name of a column in `data`. If
            given, each subject is at risk from `start` until `time`. Default is `None` (all
            subjects enter at time 0).
        weights
            Case weights (strictly positive, one per subject), or the name of a column in `data`.
            Default is `None` (all weights are `1`).
        data
            A data frame to look up column names in. Any of `time`, `event`, `start`, and `weights`
            may be a column name. See `Surv.right()` for the accepted frame types.
        censor_value
            The value (or list of values) in `event` that marks censoring, used when `states` is a
            mapping. The default is `0`.

        Returns
        -------
        Surv
            A multi-state / competing-risks `Surv` response object.

        Examples
        --------
        Build a competing-risks response from a data frame by mapping each outcome to the value
        that marks it. In `pbc`, `status` is `0` (censored), `1` (transplant), or `2` (death):

        ```{python}
        import greenwood as gw

        pbc = gw.load_dataset("pbc")

        # Map each competing outcome to its code, and 0 is censored
        y = gw.Surv.multistate(
            time="time", event="status", data=pbc, states={"transplant": 1, "death": 2}
        )
        y
        ```

        The column can use any coding, strings included. List every value that means censored in
        `censor_value=`. A value that is assigned to no state and not listed as censoring raises an
        error, so a typo cannot silently become censoring:

        ```{python}
        import polars as pl

        trial = pl.DataFrame({
            "months": [5.0, 6.0, 7.0, 8.0],
            "outcome": ["relapse", "death", "alive", "lost"],
        })

        # "alive" and "lost" both mean the subject was censored
        gw.Surv.multistate(
            time="months",
            event="outcome",
            data=trial,
            states={"relapse": "relapse", "death": "death"},
            censor_value=["alive", "lost"],
        )
        ```

        When each outcome has its own time and event columns instead of one outcome column, use
        `Surv.first_event()`. For small examples with integer codes already in place, pass the
        values and a sequence of state labels, where code `k` means `states[k - 1]`:

        ```{python}
        gw.Surv.multistate(time=[5, 6, 7, 8], event=[1, 2, 0, 1], states=("relapse", "death"))
        ```
        """
        cols = resolve_columns(
            data, {"time": time, "event": event, "start": start, "weights": weights}
        )
        stop = _to_1d_array(cols["time"])
        status, labels = encode_states(cols["event"], states, censor_value=censor_value)
        start_a = _to_1d_array(cols["start"]) if cols["start"] is not None else None
        w = _to_1d_array(cols["weights"]) if cols["weights"] is not None else None
        ctype = CensoringType.COUNTING if start_a is not None else CensoringType.RIGHT
        return cls(type=ctype, stop=stop, status=status, start=start_a, states=labels, weights=w)

    @classmethod
    def first_event(
        cls,
        endpoints: Mapping[str, Sequence[Any]],
        *,
        data: Any = None,
        censor_at: Any = None,
        start: Any = None,
        weights: Any = None,
    ) -> Self:
        """Competing-risks response from several endpoints: the earliest observed event wins.

        Some datasets record each endpoint in its own pair of columns rather than in one outcome
        column. `mgus2` is an example: `ptime`/`pstat` hold the time to plasma cell malignancy
        (PCM) and whether it occurred, and `futime`/`death` do the same for death. A
        competing-risks analysis needs one time and one cause per subject: the first endpoint
        observed, or censoring if none was.

        `first_event()` combines the pairs. For each row, the endpoint with the earliest observed
        event becomes the cause and its time becomes the event time. Ties go to the endpoint listed
        first. Rows with no observed event are censored at the latest endpoint time (the last time
        the subject was known to be event-free), or at `censor_at` if given.

        Parameters
        ----------
        endpoints
            A mapping of state label to `(time, event)` or `(time, event, event_value)`, in
            priority order for ties. Each `time` and `event` is a column name in `data` or an
            array. Without an `event_value`, each event column must be boolean or `0`/`1`.
        data
            A data frame to look up column names in. See `Surv.right()` for the accepted frame
            types.
        censor_at
            The time (a column name or an array) at which rows with no observed event are
            censored. The default is the latest of the endpoint times.
        start
            Entry times for late entry (optional).
        weights
            Case weights (optional).

        Returns
        -------
        Surv
            A multi-state `Surv` response whose states are the endpoint labels, in order.

        Warns
        -----
        UserWarning
            When a row's event is recorded after another endpoint's follow-up had already ended
            without an event. For example, a progression at month 80 for a subject whose survival
            follow-up stopped at month 60. This often means the time columns do not share an
            origin or a unit. It can also mean one endpoint was followed for less time, in which
            case the response assumes that endpoint did not occur before the recorded event.

        Examples
        --------
        Build the `mgus2` competing-risks response from its two column pairs. PCM is listed first,
        so a PCM diagnosed at the same time as death counts as PCM:

        ```{python}
        import greenwood as gw

        mgus2 = gw.load_dataset("mgus2")

        # The first observed endpoint wins, and ties go to the one listed first
        y = gw.Surv.first_event(
            endpoints={"pcm": ("ptime", "pstat"), "death": ("futime", "death")},
            data=mgus2,
        )
        y
        ```
        """
        if not endpoints:
            raise ValueError("`endpoints` must name at least one endpoint.")
        labels = [str(label) for label in endpoints]
        arguments: dict[str, Any] = {"censor_at": censor_at, "start": start, "weights": weights}
        event_values: list[Any] = []
        for i, (label, spec) in enumerate(endpoints.items()):
            if isinstance(spec, str) or len(spec) not in (2, 3):
                raise ValueError(
                    f"Endpoint {label!r} must be `(time, event)` or `(time, event, event_value)`."
                )
            arguments[f"time{i}"] = spec[0]
            arguments[f"event{i}"] = spec[1]
            event_values.append(spec[2] if len(spec) == 3 else None)
        cols = resolve_columns(data, arguments)

        times = [_to_1d_array(cols[f"time{i}"]) for i in range(len(labels))]
        n = times[0].shape[0]
        if any(t.shape[0] != n for t in times):
            raise ValueError("All endpoint columns must have the same length.")
        events = [
            _coerce_event(cols[f"event{i}"], n, event_value=event_values[i]).astype(bool)
            for i in range(len(labels))
        ]
        time_matrix = np.column_stack(times)
        event_matrix = np.column_stack(events)

        # argmin picks the first column among equal times, so ties follow the endpoint order.
        event_times = np.where(event_matrix, time_matrix, np.inf)
        first = np.argmin(event_times, axis=1)
        observed = event_matrix.any(axis=1)
        if cols["censor_at"] is not None:
            censor_time = _to_1d_array(cols["censor_at"])
        else:
            censor_time = time_matrix.max(axis=1)
        stop = np.where(observed, event_times[np.arange(n), first], censor_time)
        status = np.where(observed, first + 1, 0).astype(np.int64)
        _warn_event_after_follow_up(labels, time_matrix, event_matrix, observed, first, stop)
        return cls.multistate(
            stop, status, tuple(labels), start=cols["start"], weights=cols["weights"]
        )

    # -- derived views used by the kernel -------------------------------------

    @property
    def n(self) -> int:
        """Number of observations in the response.

        Returns the total count of subjects/observations, regardless of event status. Equivalent to
        `len(surv_object)`.

        Returns
        -------
        int
            Number of observations.

        Examples
        --------
        ```{python}
        import greenwood as gw

        y = gw.Surv.right(time=[5, 6, 4, 9], event=[1, 0, 1, 0])

        # Get the number of observations
        y.n
        ```

        This is useful for loops, validation, or allocating arrays. Often used to determine sample
        size or for sanity checks on data shape.
        """
        return int(self.stop.shape[0])

    @property
    def entry(self) -> Array:
        r"""Entry times for each observation (or $-\infty$ if no left truncation).

        For counting-process data (late entry), this returns the `start` time when each subject
        became at risk. For standard right-censored data with no left truncation, all values are
        $-\infty$, indicating subjects entered at the beginning of follow-up.

        Returns
        -------
        Array
            Entry times with shape (n,). Contains start times for counting-process form
            or $-\infty$ where there is no left truncation.

        Examples
        --------
        Right-censored data (no left truncation) has all $-\infty$ entry times:

        ```{python}
        import greenwood as gw

        y_right = gw.Surv.right(time=[5, 6, 4], event=[1, 0, 1])

        # Entry times are -inf when there is no left truncation
        y_right.entry
        ```

        Counting-process data shows each subject's entry time:

        ```{python}
        import greenwood as gw

        y_counting = gw.Surv.counting(start=[0, 2, 1], stop=[5, 6, 4], event=[1, 0, 1])

        # Entry times reflect each subject's actual start
        y_counting.entry
        ```

        The `entry` property is primarily used internally by survival estimators to correctly
        compute risk sets. You rarely need it directly, but it's available for custom analyses.
        """
        if self.start is not None:
            return self.start
        return np.full(self.n, -np.inf)

    @property
    def event(self) -> Array:
        """Boolean event indicator: True if any event occurred, False if censored.

        Converts the integer `status` codes to a simple boolean: `1` or more -> True (event),
        `0` -> False (censored). This is a convenient summary when you only care about
        event occurrence, not which specific state occurred in multi-state data.

        Returns
        -------
        Array
            Boolean array with shape (n,). `True` where status >= `1`, `False` otherwise.

        Examples
        --------
        ```{python}
        import greenwood as gw

        y = gw.Surv.right(time=[5, 6, 4, 9], event=[1, 0, 1, 0])

        # Boolean indicator: True where any event occurred
        y.event
        ```

        The `True`/`False` values indicate which subjects experienced any event. This is useful
        for filtering, counting events, or checking data quality. For multi-state data,
        this collapses all states into a single "any event" indicator:

        ```{python}
        import greenwood as gw

        y_multi = gw.Surv.multistate(
            time=[5, 6, 7, 8],
            event=[1, 2, 0, 1],
            states=("relapse", "death")
        )

        # Collapses all competing causes into a single indicator
        y_multi.event
        ```
        """
        return self.status >= 1

    @property
    def is_truncated(self) -> bool:
        """Whether the response has left truncation (late entry).

        Left truncation occurs in counting-process data when subjects enter the risk set
        at different times (late entry). This is common in studies with age-based entry
        or complex follow-up patterns. When True, the `entry()` property contains the
        actual start times. When False, all subjects implicitly start at time 0.

        Returns
        -------
        bool
            `True` if the response has left-truncation entry times, `False` otherwise.

        Examples
        --------
        Right-censored data has no left truncation:

        ```{python}
        import greenwood as gw

        # Right-censored data has no left truncation
        y_right = gw.Surv.right(time=[5, 6, 4], event=[1, 0, 1])
        y_right.is_truncated
        ```

        Counting-process data with late entry is truncated:

        ```{python}
        # Counting-process data with late entry is truncated
        y_counting = gw.Surv.counting(start=[0, 2, 1], stop=[5, 6, 4], event=[1, 0, 1])
        y_counting.is_truncated
        ```

        This property is useful for understanding data structure and for conditional logic
        that handles truncated vs. non-truncated data differently.
        """
        return self.start is not None

    @property
    def is_multistate(self) -> bool:
        """Whether the response has multiple competing event states.

        Multi-state responses track which of several competing outcomes occurred
        (e.g., "relapse" vs. "death"). When False, there is only one event type
        (censored or not). When True, the `states` property contains the outcome labels.

        Returns
        -------
        bool
            `True` if the response has multiple event states, `False` for single-event data.

        Examples
        --------
        Right-censored data has a single outcome:

        ```{python}
        import greenwood as gw

        # Single-event data is not multi-state
        y_right = gw.Surv.right(time=[5, 6, 4], event=[1, 0, 1])
        y_right.is_multistate
        ```

        Multi-state data with competing risks:

        ```{python}
        # Competing-risks data is multi-state
        y_multi = gw.Surv.multistate(
            time=[5, 6, 7, 8],
            event=[1, 2, 0, 1],
            states=("relapse", "death")
        )

        y_multi.is_multistate
        ```

        This property is useful for determining how to interpret the event codes and
        what kind of survival estimation is needed.
        """
        return self.states is not None

    @property
    def n_events(self) -> int:
        """Count of observations where an event occurred (any state in multi-state data).

        Counts all observations with `status >= 1`. For multi-state responses, this counts
        all events regardless of which specific state occurred. For single-event data, this
        is the count of subjects who experienced the event.

        Returns
        -------
        int
            Number of observations with an event. Equals `n - n_censored`.

        Examples
        --------
        ```{python}
        import greenwood as gw

        y = gw.Surv.right(time=[5, 6, 4, 9], event=[1, 0, 1, 0])

        # Count of observed events
        y.n_events
        ```

        This is useful for descriptive statistics, event rate calculations, or validating
        data: `assert y.n_events + y.n_censored == y.n`.

        For multi-state data, this gives the total event count across all states:

        ```{python}
        y_multi = gw.Surv.multistate(
            time=[5, 6, 7, 8],
            event=[1, 2, 0, 1],
            states=("relapse", "death")
        )

        # Total events across all competing causes
        y_multi.n_events
        ```
        """
        return int(np.count_nonzero(self.event))

    @property
    def n_censored(self) -> int:
        """Count of censored observations.

        Counts all observations where the event was not observed (status == 0).
        These are subjects whose true event time is unknown but exceeds their
        observation time.

        Returns
        -------
        int
            Number of censored observations. Equals `n - n_events`.

        Examples
        --------
        ```{python}
        import greenwood as gw

        y = gw.Surv.right(time=[5, 6, 4, 9], event=[1, 0, 1, 0])

        # Count of censored observations
        y.n_censored
        ```

        Often used for descriptive summary: "We observed 2 events and 2 censored subjects
        out of 4 total." Can validate data quality:

        ```{python}
        # Verify that events plus censorings equal the total
        assert y.n_events + y.n_censored == y.n
        ```

        Higher censoring rates reduce the information available for estimation and may
        require larger sample sizes for stable inference.
        """
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

    def _frame_columns(self) -> dict[str, Any]:
        cols: dict[str, Any] = {}
        if self.start is not None:
            cols["start"] = self.start
        if self.lower is not None:
            cols["lower"] = self.lower
        cols["stop"] = self.stop
        cols["status"] = self.status
        if self.weights is not None:
            cols["weight"] = self.weights
        return cols

    def to_frame(self, *, format: str | None = None) -> Any:
        """Return the response as a DataFrame (one row per observation).

        Exports the `Surv` object to a tidy table where each row represents one
        observation. The table includes the `stop` and `status` columns, plus optional
        columns for `start` (entry time in counting-process form), `lower` (lower bound
        for interval censoring), and `weight` (case weights). This is convenient for
        inspection, export to CSV, or integration with other DataFrame workflows.

        Parameters
        ----------
        format
            Output format: `None` (default), `"pandas"`, `"polars"`, or `"pyarrow"`. When
            `None`, a backend is auto-detected (Polars, then Pandas, then PyArrow).

        Returns
        -------
        pandas.DataFrame, polars.DataFrame, or pyarrow.Table
            A tidy table with one row per observation, including columns for `stop`,
            `status`, and optional `start`, `lower`, `weight` columns.

        Raises
        ------
        ImportError
            If the requested (or, when auto-detecting, any) DataFrame library is not
            installed.

        Examples
        --------
        Build a right-censored response and export it as a Polars frame. Each row
        represents one observation with its event time and status:

        ```{python}
        import greenwood as gw

        y = gw.Surv.right(time=[5, 6, 4, 9], event=[1, 0, 1, 0])

        # Export the response as a Polars DataFrame
        y.to_frame(format="polars")
        ```

        Request a different backend with `format=`:

        ```{python}
        # Export as a Pandas DataFrame instead
        y.to_frame(format="pandas")
        ```
        """
        return to_dataframe(self._frame_columns(), format=format)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-ready mapping fully describing the response.

        This method serializes the entire `Surv` object into a plain Python dictionary,
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
        `stop`, and `status`. Unused fields are `None`. Here we build a response and
        convert it to a dictionary:

        ```{python}
        import greenwood as gw

        y = gw.Surv.right(time=[5, 6, 4, 9], event=[1, 0, 1, 0])

        # Serialize the response to a plain dictionary
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

        This is the inverse of `to_dict()`: it reconstructs an equivalent `Surv` object
        from a dictionary previously created by `to_dict()`. Useful for deserializing
        stored or transmitted data, or for round-tripping through storage formats.

        Parameters
        ----------
        data
            A dictionary produced by `to_dict()` containing keys `type`, `stop`, `status`,
            and optional keys for `start`, `lower`, `states`, `weights`.

        Returns
        -------
        Surv
            A new `Surv` object with the same data and structure as the input dictionary.

        Examples
        --------
        Rebuild an equivalent response from its dictionary representation. Here we
        serialize a response and immediately deserialize it:

        ```{python}
        import greenwood as gw

        y = gw.Surv.right(time=[5, 6, 4, 9], event=[1, 0, 1, 0])

        # Round-trip through dictionary serialization
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

        This method converts the entire `Surv` object to a compact, JSON-formatted string
        suitable for storage in files, databases, or transmission over APIs. By default,
        the output is human-readable with indentation; pass `indent=None` for a compact
        form. The serialization is deterministic: the same `Surv` object always produces
        the identical JSON string.

        Parameters
        ----------
        indent
            Number of spaces to use for indentation. If `None`, produces compact JSON
            without whitespace. Default is 2 (human-readable).

        Returns
        -------
        str
            A JSON string representing the `Surv` object, including censoring type and
            all arrays.

        Examples
        --------
        Serialize to JSON. By default, output is indented for readability. Here we build
        a response and show just the first 120 characters of compact JSON:

        ```{python}
        import greenwood as gw

        y = gw.Surv.right(time=[5, 6, 4, 9], event=[1, 0, 1, 0])

        # Serialize to compact JSON
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

        This is the inverse of `to_json()`: it reconstructs a `Surv` object from a JSON
        string previously created by `to_json()`. Useful for loading data from stored
        files, API responses, or any other JSON source. The reconstructed object is
        guaranteed to be equivalent to the original.

        Parameters
        ----------
        text
            A JSON string produced by `to_json()` containing the serialized `Surv` data.

        Returns
        -------
        Surv
            A new `Surv` object restored from the JSON representation.

        Examples
        --------
        Deserialize from JSON. Build a response, then round-trip through `to_json()`
        and back:

        ```{python}
        import greenwood as gw

        y = gw.Surv.right(time=[5, 6, 4, 9], event=[1, 0, 1, 0])

        # Round-trip through JSON serialization
        json_text = y.to_json()
        restored = gw.Surv.from_json(json_text)
        print("Round-trip successful:", y.to_json() == restored.to_json())
        ```

        The restored object is an exact copy of the original `Surv` object.
        """
        return cls.from_dict(json.loads(text))


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
    import warnings

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
