@dataclass(frozen=True)
class EventTable:
    """Per-time risk-set tabulation (optionally within strata).

    Every array is aligned row-wise. When `strata` is not `None`, rows are grouped by
    stratum (each stratum's times are ascending). Counts are weighted when case weights
    are supplied, so they may be floats.
    """

    time: Array
    n_risk: Array
    n_event: Array
    n_censor: Array
    strata: Array | None = None

    def __len__(self) -> int:
        return int(self.time.shape[0])

    def to_dataframe(self, backend: str = "pandas") -> Any:
        """Return the tabulation as a tidy dataframe."""
        cols: dict[str, Array] = {}
        if self.strata is not None:
            cols["strata"] = self.strata
        cols["time"] = self.time
        cols["n_risk"] = self.n_risk
        cols["n_event"] = self.n_event
        cols["n_censor"] = self.n_censor
        if backend == "pandas":
            import pandas as pd

            return pd.DataFrame(cols)
        if backend == "polars":
            import polars as pl

            return pl.DataFrame(cols)
        raise ValueError(f"Unknown backend {backend!r}; use 'pandas' or 'polars'.")


def _tabulate_block(
    entry: Array,
    exit_: Array,
    event: Array,
    weight: Array,
) -> tuple[Array, Array, Array, Array]:
    """Tabulate one homogeneous block (no groups). Returns time/n_risk/n_event/n_censor."""
    times = np.unique(exit_)

    # Events and censorings at each unique exit time, via weighted bincount.
    idx = np.searchsorted(times, exit_)
    n_time = times.shape[0]
    n_event = np.bincount(idx[event], weights=weight[event], minlength=n_time)
    n_censor = np.bincount(idx[~event], weights=weight[~event], minlength=n_time)

    # n_risk(t) = (weight with entry < t) - (weight with exit < t).
    e_order = np.argsort(entry, kind="stable")
    e_sorted = entry[e_order]
    e_cumw = np.concatenate(([0.0], np.cumsum(weight[e_order])))
    entered_before = e_cumw[np.searchsorted(e_sorted, times, side="left")]

    x_order = np.argsort(exit_, kind="stable")
    x_sorted = exit_[x_order]
    x_cumw = np.concatenate(([0.0], np.cumsum(weight[x_order])))
    exited_before = x_cumw[np.searchsorted(x_sorted, times, side="left")]

    n_risk = entered_before - exited_before
    return times, n_risk, n_event, n_censor


