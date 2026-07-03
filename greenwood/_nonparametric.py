class KaplanMeier:
    """Kaplan-Meier product-limit estimator of the survival function.

    Parameters
    ----------
    conf_type
        Confidence-interval transform: `"log"` (default, as in R's `survfit`), `"plain"`,
        or `"log-log"`.
    conf_level
        Confidence level for the interval (default 0.95).

    Notes
    -----
    Call `fit` with a `Surv` response. Results are exposed as aligned arrays (`time_`,
    `survival_`, `std_error_`, `conf_low_`, `conf_high_`, `strata_`), as a tidy frame via
    `to_dataframe`, and through `median`, `quantile`, and `predict`.
    """

    def __init__(self, *, conf_type: str = "log", conf_level: float = 0.95) -> None:
        if conf_type not in _CONF_TYPES:
            raise ValueError(f"conf_type must be one of {sorted(_CONF_TYPES)}, got {conf_type!r}.")
        if not 0.0 < conf_level < 1.0:
            raise ValueError(f"conf_level must be in (0, 1), got {conf_level}.")
        self.conf_type = conf_type
        self.conf_level = conf_level

    def fit(self, surv: Surv, *, by: Any = None, weights: Any = None) -> KaplanMeier:
        """Fit the estimator to a `Surv` response, optionally stratified by `by`."""
        z = float(norm.ppf(1.0 - (1.0 - self.conf_level) / 2.0))
        self._blocks = _fit_blocks(surv, by, weights, self.conf_type, z)
        self._grouped = by is not None
        return self

    # -- aligned-array accessors ---------------------------------------------

    def _concat(self, attr: str) -> Array:
        return np.concatenate([getattr(b, attr) for b in self._blocks])

    @property
    def time_(self) -> Array:
        return self._concat("time")

    @property
    def survival_(self) -> Array:
        return self._concat("surv")

    @property
    def std_error_(self) -> Array:
        return self._concat("std_error")

    @property
    def conf_low_(self) -> Array:
        return self._concat("conf_low")

    @property
    def conf_high_(self) -> Array:
        return self._concat("conf_high")

    @property
    def cumhaz_(self) -> Array:
        return self._concat("cumhaz")

    @property
    def strata_(self) -> Array | None:
        if not self._grouped:
            return None
        return np.concatenate(
            [np.full(b.time.shape[0], b.label, dtype=object) for b in self._blocks]
        )

    # -- quantiles ------------------------------------------------------------

    def quantile(self, p: float, *, ci: bool = False) -> Any:
        """Return the `p`-quantile survival time per stratum.

        With `ci=True`, also return confidence limits obtained by inverting the survival
        confidence band (the R `quantile.survfit` convention). For a single stratum a
        scalar (or 3-tuple) is returned; when stratified, a dict keyed by stratum label.
        """
        level = 1.0 - p

        def one(b: _Block) -> Any:
            point = _crossing_time(b.time, b.surv, level)
            if not ci:
                return point
            lower = _crossing_time(b.time, b.conf_low, level)
            upper = _crossing_time(b.time, b.conf_high, level)
            return (point, lower, upper)

        if not self._grouped:
            return one(self._blocks[0])
        return {b.label: one(b) for b in self._blocks}

    def median(self, *, ci: bool = False) -> Any:
        """Median survival time per stratum (the 0.5-quantile)."""
        return self.quantile(0.5, ci=ci)

    # -- prediction -----------------------------------------------------------

    def predict(self, times: Any, *, what: str = "survival") -> Any:
        """Evaluate the step function at `times` (survival or cumulative hazard).

        Returns an array for a single stratum, or a dict of arrays when stratified.
        """
        if what not in ("survival", "cumhaz"):
            raise ValueError(f"what must be 'survival' or 'cumhaz', got {what!r}.")
        query = np.atleast_1d(np.asarray(times, dtype=float))

        def one(b: _Block) -> Array:
            curve = b.surv if what == "survival" else b.cumhaz
            baseline = 1.0 if what == "survival" else 0.0
            # Right-continuous step function: value at t is the last step at time <= t.
            idx = np.searchsorted(b.time, query, side="right") - 1
            out = np.where(idx >= 0, curve[idx.clip(min=0)], baseline)
            return out

        if not self._grouped:
            return one(self._blocks[0])
        return {b.label: one(b) for b in self._blocks}

    # -- interop --------------------------------------------------------------

    def to_dataframe(self, backend: str = "pandas") -> Any:
        """Return the fitted curve(s) as a tidy frame, one row per time point."""
        cols: dict[str, Array] = {}
        if self._grouped:
            cols["strata"] = self.strata_  # type: ignore[assignment]
        cols["time"] = self.time_
        cols["n_risk"] = self._concat("n_risk")
        cols["n_event"] = self._concat("n_event")
        cols["n_censor"] = self._concat("n_censor")
        cols["estimate"] = self.survival_
        cols["std_error"] = self.std_error_
        cols["conf_low"] = self.conf_low_
        cols["conf_high"] = self.conf_high_
        if backend == "pandas":
            import pandas as pd

            return pd.DataFrame(cols)
        if backend == "polars":
            import polars as pl

            return pl.DataFrame(cols)
        raise ValueError(f"Unknown backend {backend!r}; use 'pandas' or 'polars'.")


