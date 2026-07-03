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

