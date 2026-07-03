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


