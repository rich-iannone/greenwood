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
    to construct `Surv` objects. They validate your input and set the censoring type
    appropriately. As such, direct instantiation is not recommended.

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
    import greenwood as gw
    import numpy as np

    y = gw.Surv(
        type=gw.CensoringType.RIGHT,
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
    y = gw.Surv.right(time=[5, 6, 4, 9], event=[1, 0, 1, 0])
    y
    ```

    Counting-process form with left truncation (late entry):

    ```{python}
    y = gw.Surv.counting(start=[0, 2, 1], stop=[5, 6, 4], event=[1, 0, 1])
    y
    ```

    Interval-censored (event known to occur in a time window):

    ```{python}
    y = gw.Surv.interval(lower=[1, 3], upper=[3, 8])
    y
    ```

    Multi-state (competing risks, multiple mutually exclusive events):

    ```{python}
    y = gw.Surv.multistate(time=[5, 6, 4], event=[1, 0, 2], states=("pcm", "death"))
    y
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
    def right(cls, time: Any, event: Any = None, *, weights: Any = None) -> Self:
        """Right-censored response: the standard and most common form of survival data.

        Right censoring is the default in survival analysis. It occurs when follow-up 
        ends before the event happens—a subject is still event-free when we last observed 
        them. This is the most common censoring mechanism in practice:
        
        - **Study ends**: A clinical trial concludes while some patients are still healthy
        - **Loss to follow-up**: A subject drops out, moves away, or stops visiting the clinic
        - **Administrative censoring**: Follow-up ends at a fixed time regardless of status
        
        Right censoring is called "censoring from the right" because we know the event 
        happened *after* the censoring time. We record that a subject was event-free at 
        their last observation but don't know how much longer they could have gone.

        This simple form assumes all subjects enter follow-up at the same reference time 
        (typically time 0). If subjects enter at different times or follow-up is complex, 
        use `counting()` or `interval()` instead.

        Parameters
        ----------
        time
            Exit times when follow-up ends (one per subject). Must be finite and 
            non-negative. This is the time of either the event or censoring, whichever 
            came first.
        event
            Event indicators:

            - 1 = event occurred (fully observed)
            - 0 = censored (event time unknown but > time)
            
            If `None`, all subjects are treated as having experienced the event 
            (useful for testing or descriptive purposes).
        weights
            Case weights (strictly positive, one per subject). Used to weight subjects 
            differently in survival analysis (e.g., inverse probability weighting). 
            Default is `None` (all weights = 1).

        Returns
        -------
        Surv
            A right-censored `Surv` response object (the most common type).

        Examples
        --------
        The most common case: subjects have an exit `time` and an `event` indicator 
        (1 if event occurred, 0 if censored):

        ```{python}
        import greenwood as gw
        
        y = gw.Surv.right(time=[5, 6, 4, 9], event=[1, 0, 1, 0])
        y
        ```

        The display shows 4 observations with 2 events and 2 censored observations:
        - Subjects 1 and 3: Event observed (no marker or `*` depending on visualization)
        - Subjects 2 and 4: Censored (indicated by `+` marker)—still event-free at times 6 and 9
        
        This is the default input format for nearly all survival analysis methods. 
        Right-censored data is so ubiquitous that "survival data" often refers specifically 
        to right-censored observations.

        See Also
        --------
        left : Event occurred before the observation time.
        interval : Event time is known to lie in a range.
        counting : Late entry and time-varying covariates.
        multistate : Track multiple competing outcomes.
        """
        stop = _to_1d_array(time)
        status = _coerce_event(event, stop.shape[0])
        w = _to_1d_array(weights) if weights is not None else None
        return cls(type=CensoringType.RIGHT, stop=stop, status=status, weights=w)

    @classmethod
    def left(cls, time: Any, event: Any = None, *, weights: Any = None) -> Self:
        """Left-censored response: event occurred before the observation time.

        Left censoring occurs when all you know is that an event happened *before* you 
        observed the subject. For example, an infection that must have occurred before a
        patient was tested, or a failure that was known to have happened sometime before
        equipment was inspected. The exact event time is unknown, but you know it was no 
        later than the recorded `time`.

        This is less common than right censoring, but important in scenarios where you 
        cannot pinpoint when something happened, only that it already had.

        Parameters
        ----------
        time
            Observation times (the upper bound on when the event occurred). Must be 
            finite and non-negative. Each value represents "the event happened by this time".
        event
            Event indicators:

            - 1 = event occurred before `time` (left-censored)
            - 0 = subject was event-free at `time` (not censored)
            
            If `None`, all subjects are treated as having experienced the event.
        weights
            Case weights (strictly positive, one per subject). Used to weight subjects 
            differently in survival analysis. Default is `None` (all weights = 1).

        Returns
        -------
        Surv
            A left-censored `Surv` response object.

        Examples
        --------
        Here we have 3 subjects. Two experienced the event before the recorded time 
        (event=1), and one was event-free at observation (event=0):

        ```{python}
        import greenwood as gw
        
        y = gw.Surv.left(time=[5, 6, 4], event=[1, 0, 1])
        y
        ```

        The display shows the data structure:

        - The `<` symbol indicates left-censored observations (event occurred before time)
        - The `+` symbol indicates subjects who were still event-free at the observation time
        - The left-censoring type `"left"` is displayed at the top

        See Also
        --------
        right : Right-censored response (event after the observation time).
        counting : Time intervals with late entry.
        interval : Event lies in a known interval.
        """
        stop = _to_1d_array(time)
        status = _coerce_event(event, stop.shape[0])
        w = _to_1d_array(weights) if weights is not None else None
        return cls(type=CensoringType.LEFT, stop=stop, status=status, weights=w)

    @classmethod
    def counting(cls, start: Any, stop: Any, event: Any = None, *, weights: Any = None) -> Self:
        """Counting-process response: track subjects entering and exiting the risk set at different times.

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
            Entry times (when each subject becomes at risk). Must be finite and non-negative.
            Represents when the subject enters the risk set. In standard studies, this is 0;
            in studies with late entry, it's the age/time at enrollment.
        stop
            Exit times (when follow-up ends). Must be finite, non-negative, and strictly 
            greater than the corresponding `start`. Represents when the subject leaves follow-up
            (event, censoring, or end of study).
        event
            Event indicators (1 = event occurred, 0 = censored at `stop` time).
            If `None`, all subjects are treated as having experienced the event.
        weights
            Case weights (strictly positive, one per subject). Used to weight subjects 
            differently in survival analysis. Default is `None` (all weights = 1).

        Returns
        -------
        Surv
            A counting-process `Surv` response object with potential left truncation.

        Examples
        --------
        Here we have 3 subjects with different entry times:

        ```{python}
        import greenwood as gw
        
        y = gw.Surv.counting(start=[0, 2, 1], stop=[5, 6, 4], event=[1, 0, 1])
        y
        ```

        The display shows:

        - Subject 1: Entered at time 0, exited with an event at time 5
        - Subject 2: Entered at time 2 (late entry), exited censored at time 6
        - Subject 3: Entered at time 1, experienced an event at time 4
        
        Only subjects 2 and 3 benefit from the late entry handling, but the counting-process 
        form elegantly handles all cases uniformly. This representation is also essential 
        for studies with time-varying covariates, where you create multiple rows per subject 
        as their covariate values change.

        See Also
        --------
        right : Simple right-censored response (all subjects start at time 0).
        interval : Event lies in a known interval.
        multistate : Track transitions to multiple competing states.
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
        """Interval-censored response: event time is known to lie within a range.

        Interval censoring occurs when you know the event happened sometime between two 
        observation times, but not exactly when. Common in:
        
        - **Medical follow-up**: Disease detection between clinic visits. You might know 
          a patient's disease status at two checkups, but not the exact time of onset.
        - **Equipment reliability**: Failure detected between inspections. You know failure 
          happened between the last working inspection and the current failed one.
        - **Longitudinal surveys**: Event reported between survey waves but exact timing unknown.

        The interval-censored form captures this uncertainty. The event happened somewhere 
        in the interval (lower, upper]. If lower == upper, it's an exact (uncensored) event.
        Use infinity for upper to represent right-censoring, and 0 for lower to represent 
        left-censoring.

        Parameters
        ----------
        lower
            Interval lower bounds (one per subject). Must be finite and non-negative. 
            Event happened *after* this time (possibly at this time).
            Set to 0 to mark left-censored subjects (event happened before first observation).
        upper
            Interval upper bounds (one per subject). Must be finite, non-negative, and >= `lower`.
            Event happened *by* this time. Set to `numpy.inf` to mark right-censored subjects (no
            event observed by end of study).
        weights
            Case weights (strictly positive, one per subject). Used to weight subjects 
            differently in survival analysis. Default is `None` (all weights = 1).

        Returns
        -------
        Surv
            An interval-censored `Surv` response object.

        Examples
        --------
        Here we have 3 subjects with different levels of observation precision:

        ```{python}
        import numpy as np

        y = gw.Surv.interval(lower=[1, 2, 3], upper=[2, np.inf, 5])
        y
        ```

        The display shows:
        - Subject 1: Exact event at time 2 (lower == upper)
        - Subject 2: Right-censored at time 2 (upper = infinity means event never observed)
        - Subject 3: Interval-censored between times 3 and 5 (event happened somewhere in that window)

        Interval censoring gives you more information than right censoring alone. Rather than 
        just knowing "no event by time X," you may know "event was definitely before time Y 
        but after time X," which allows for more precise estimation when multiple observations 
        bracket the event.

        See Also
        --------
        left : Event occurred before the observation time.
        right : Event occurred after the observation time.
        counting : Track subjects entering and exiting at different times.
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
            Event or censoring times (one per subject). Must be finite and non-negative.
            Represents when the subject experienced an outcome (or was censored).
        event
            Event codes indicating which state occurred:

            - 0 = censored (no event observed)
            - 1 = transitioned to states[0] (first outcome)
            - 2 = transitioned to states[1] (second outcome)
            - ... and so on for each defined state
            
            Must be in range [0, len(states)].
        states : tuple[str, ...]
            Labels for the possible outcomes. Event codes index into this tuple.
            Example: states=("relapse", "death") means:

            - event code 1 → relapse occurred
            - event code 2 → death occurred
            
            Labels are arbitrary strings describing what the transition represents.
        start : array-like, optional
            Optional entry times (for late entry / left truncation). If provided, each subject
            is only at risk from `start` until `time`. Default is `None` (all subjects enter at time 0).
        weights : array-like, optional
            Case weights (strictly positive, one per subject). Used to weight subjects 
            differently in survival analysis. Default is `None` (all weights = 1).

        Returns
        -------
        Surv
            A multi-state / competing-risks `Surv` response object.

        Examples
        --------
        Here we have 4 subjects with 2 competing outcomes (relapse and death):

        ```{python}
        import greenwood as gw
        
        y = gw.Surv.multistate(
            time=[5, 6, 7, 8],
            event=[1, 2, 0, 1],
            states=("relapse", "death")
        )
        
        y
        ```

        The display shows:

        - Subject 1: Transitioned to "relapse" (event code 1) at time 5
        - Subject 2: Transitioned to "death" (event code 2) at time 6
        - Subject 3: Censored (event code 0) at time 7
        - Subject 4: Transitioned to "relapse" (event code 1) at time 8

        You can then estimate the probability of each outcome separately, capturing the 
        full picture: not just "will something happen?" but "which specific outcome is most likely?"
        This avoids the bias of artificially grouping competing outcomes together.

        See Also
        --------
        right : Simple right-censored `Surv` response object (only one possible outcome).
        counting : Time intervals with late entry.
        left : Event occurred before the observation time.
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
        """Number of observations in the response.

        Returns the total count of subjects/observations, regardless of event status.
        Equivalent to `len(surv_object)`.

        Returns
        -------
        int
            Number of observations.

        Examples
        --------
        ```{python}
        import greenwood as gw
        
        y = gw.Surv.right(time=[5, 6, 4, 9], event=[1, 0, 1, 0])
        y.n
        ```

        This is useful for loops, validation, or allocating arrays. Often used to determine
        sample size or for sanity checks on data shape.
        """
        return int(self.stop.shape[0])

    @property
    def entry(self) -> Array:
        """Entry times for each observation (or -∞ if no left truncation).

        For counting-process data (late entry), this returns the `start` time when each 
        subject became at risk. For standard right-censored data with no left truncation, 
        all values are -∞, indicating subjects entered at the beginning of follow-up.

        Returns
        -------
        Array
            Entry times with shape (n,). Contains start times for counting-process form
            or -∞ where there is no left truncation.

        Examples
        --------
        Right-censored data (no left truncation) has all -∞ entry times:

        ```{python}
        import greenwood as gw
        
        y_right = gw.Surv.right(time=[5, 6, 4], event=[1, 0, 1])
        y_right.entry
        ```

        Counting-process data shows each subject's entry time:

        ```{python}
        import greenwood as gw
        
        y_counting = gw.Surv.counting(start=[0, 2, 1], stop=[5, 6, 4], event=[1, 0, 1])
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

        Converts the integer `status` codes to a simple boolean: 1 or more -> True (event),
        0 -> False (censored). This is a convenient summary when you only care about 
        event occurrence, not which specific state occurred in multi-state data.

        Returns
        -------
        Array
            Boolean array with shape (n,). `True` where status >= 1, `False` otherwise.

        Examples
        --------
        ```{python}
        import greenwood as gw
        
        y = gw.Surv.right(time=[5, 6, 4, 9], event=[1, 0, 1, 0])
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
        actual start times; when False, all subjects implicitly start at time 0.

        Returns
        -------
        bool
            `True` if the response has left-truncation entry times, `False` otherwise.

        Examples
        --------
        Right-censored data has no left truncation:

        ```{python}
        import greenwood as gw
        
        y_right = gw.Surv.right(time=[5, 6, 4], event=[1, 0, 1])
        y_right.is_truncated
        ```

        Counting-process data with late entry is truncated:

        ```{python}
        import greenwood as gw
        
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
        
        y_right = gw.Surv.right(time=[5, 6, 4], event=[1, 0, 1])
        y_right.is_multistate
        ```

        Multi-state data with competing risks:

        ```{python}
        import greenwood as gw
        
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
        y.n_events
        ```

        This is useful for descriptive statistics, event rate calculations, or validating
        data: `assert y.n_events + y.n_censored == y.n`.

        For multi-state data, this gives the total event count across all states:

        ```{python}
        import greenwood as gw
        
        y_multi = gw.Surv.multistate(
            time=[5, 6, 7, 8],
            event=[1, 2, 0, 1],
            states=("relapse", "death")
        )
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
        y.n_censored
        ```

        Often used for descriptive summary: "We observed 2 events and 2 censored subjects
        out of 4 total." Can validate data quality:

        ```{python}
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

    def to_pandas(self) -> Any:
        """Return the response as a Pandas DataFrame (one row per observation).

        This method exports the `Surv` object to a tidy Pandas DataFrame format, where
        each row represents one observation. The DataFrame includes the `stop` and
        `status` columns, plus optional columns for `start` (entry time in counting
        process), `lower` (lower bound for interval censoring), and `weight` (case
        weights). This format is convenient for inspection, export to CSV or other
        file formats, or integration with other Pandas workflows.

        Returns
        -------
        pandas.DataFrame
            A tidy DataFrame with one row per observation, including columns for `stop`,
            `status`, and optional `start`, `lower`, `weight` columns.

        Raises
        ------
        ImportError
            If Pandas is not installed.

        Examples
        --------
        Export to Pandas DataFrame. Each row represents one observation with its
        event time and status:

        ```{python}
        y.to_pandas()
        ```

        The resulting DataFrame can be saved to CSV, used with Pandas functions, or
        integrated into standard data science workflows.
        """
        try:
            import pandas as pd
        except ImportError as e:
            raise ImportError(
                "Pandas is required for to_pandas(). Install it with: pip install pandas"
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

        This method exports the `Surv` object to a tidy Polars DataFrame format, where
        each row represents one observation. Polars provides superior performance and
        memory efficiency compared to Pandas for larger datasets. The DataFrame includes
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
            If Polars is not installed.

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
                "Polars is required for to_polars(). Install it with: pip install polars"
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

        This method exports the `Surv` object to a PyArrow Table, a columnar data structure
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
            If PyArrow is not installed.

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
                "PyArrow is required for to_arrow(). Install it with: pip install pyarrow"
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
        serialize `y` and immediately deserialize it:

        ```{python}
        import greenwood as gw

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

        This is the inverse of `to_json()`: it reconstructs a `Surv` object from a JSON
        string previously created by `to_json()`. Useful for loading data from stored
        files, API responses, or any other JSON source. The reconstructed object is
        guaranteed to be equivalent to the original.

        Parameters
        ----------
        text : str
            A JSON string produced by `to_json()` containing the serialized `Surv` data.

        Returns
        -------
        `Surv`
            A new `Surv` object restored from the JSON representation.

        Examples
        --------
        Deserialize from JSON. Round-trip through `to_json()` and back:

        ```{python}
        json_text = y.to_json()
        restored = gw.Surv.from_json(json_text)
        print("Round-trip successful:", y.to_json() == restored.to_json())
        ```

        The restored object is an exact copy of the original `Surv` object.
        """
        return cls.from_dict(json.loads(text))
