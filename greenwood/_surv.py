class CensoringType(str, Enum):
    """The censoring flavor of a `Surv` response."""

    RIGHT = "right"
    LEFT = "left"
    INTERVAL = "interval"
    COUNTING = "counting"


def _to_1d_array(x: Any, *, dtype: Any = float) -> Array:
    """Coerce a narwhals series, NumPy array, or sequence to a 1-D NumPy array."""
    if x is None:
        raise ValueError("Expected an array-like, got None.")
    if isinstance(x, (np.ndarray, list, tuple)):
        arr = np.asarray(x)
    else:
        # A narwhals-native series (pandas, Polars, ...) or anything else array-like.
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
    """A validated time-to-event response.

    Prefer the constructors (`Surv.right`, `Surv.counting`, `Surv.interval`,
    `Surv.left`, `Surv.multistate`) over building this directly.

    Attributes
    ----------
    type
        The `CensoringType`.
    stop
        Exit time (for interval censoring, the upper bound).
    status
        Integer event code per observation: 0 = censored, 1 = event (for multi-state,
        codes >= 1 index into `states`).
    start
        Entry time for the counting-process form (left truncation); `None` otherwise.
    lower
        Lower bound for interval censoring; `None` otherwise.
    states
        Event-state labels for multi-state/competing-risks endpoints; `None` for the
        single-event case.
    weights
        Optional case weights (strictly positive).
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
        """Right-censored response `(time, event)`. `event=None` means all events."""
        stop = _to_1d_array(time)
        status = _coerce_event(event, stop.shape[0])
        w = _to_1d_array(weights) if weights is not None else None
        return cls(type=CensoringType.RIGHT, stop=stop, status=status, weights=w)

    @classmethod
    def left(cls, time: Any, event: Any = None, *, weights: Any = None) -> Self:
        """Left-censored response `(time, event)`."""
        stop = _to_1d_array(time)
        status = _coerce_event(event, stop.shape[0])
        w = _to_1d_array(weights) if weights is not None else None
        return cls(type=CensoringType.LEFT, stop=stop, status=status, weights=w)

    @classmethod
    def counting(cls, start: Any, stop: Any, event: Any = None, *, weights: Any = None) -> Self:
        """Counting-process response `(start, stop, event]` (left truncation / time-varying)."""
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
        """
        lower_a = _to_1d_array(lower)
        upper_a = _to_1d_array(upper)
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

    def as_dataframe(self, backend: str = "pandas") -> Any:
        """Return the response as a tidy dataframe (one row per observation)."""
        cols: dict[str, Array] = {}
        if self.start is not None:
            cols["start"] = self.start
        if self.lower is not None:
            cols["lower"] = self.lower
        cols["stop"] = self.stop
        cols["status"] = self.status
        if self.weights is not None:
            cols["weight"] = self.weights
        if backend == "pandas":
            import pandas as pd

            return pd.DataFrame(cols)
        if backend == "polars":
            import polars as pl

            return pl.DataFrame(cols)
        raise ValueError(f"Unknown backend {backend!r}; use 'pandas' or 'polars'.")

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-ready mapping fully describing the response."""

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

