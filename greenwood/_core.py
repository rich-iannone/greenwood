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


