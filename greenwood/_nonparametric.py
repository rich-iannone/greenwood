"""Non-parametric estimators: Kaplan-Meier survival and Nelson-Aalen cumulative hazard.

Both are built on the risk-set / event-table kernel in `_core`, so they inherit its
left-truncation and case-weight handling and its R-validated tabulation. The Kaplan-Meier
survival function uses Greenwood's variance, with a choice of confidence-interval
transforms matching R's `survfit` (`plain`, `log`, `log-log`).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt
from scipy.stats import norm

from ._core import event_table

if TYPE_CHECKING:
    from ._surv import Surv

__all__ = ["KaplanMeier", "NelsonAalen"]

Array = npt.NDArray[Any]

_CONF_TYPES = frozenset({"plain", "log", "log-log"})


def _km_confidence(surv: Array, sigma: Array, conf_type: str, z: float) -> tuple[Array, Array]:
    """Confidence limits for the survival function on the requested scale.

    `sigma` is the standard error of `log(S)` (the square root of Greenwood's sum), so
    `se(S) = S * sigma`. Limits are clipped to `[0, 1]`, matching R's `survfit`.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        if conf_type == "plain":
            se = surv * sigma
            lower = surv - z * se
            upper = surv + z * se
        elif conf_type == "log":
            lower = surv * np.exp(-z * sigma)
            upper = surv * np.exp(z * sigma)
        else:  # "log-log"
            log_s = np.log(surv)
            a = np.log(-log_s)
            se_a = sigma / np.abs(log_s)
            lower = np.exp(-np.exp(a + z * se_a))
            upper = np.exp(-np.exp(a - z * se_a))

    # Degenerate points: S == 1 (no events yet) gives a [1, 1] interval; S == 0 (all
    # remaining failed) has undefined variance, so the interval is NaN, matching R.
    at_one = surv >= 1.0
    at_zero = surv <= 0.0
    lower = np.where(at_one, 1.0, lower)
    upper = np.where(at_one, 1.0, upper)
    lower = np.clip(lower, 0.0, 1.0)
    upper = np.clip(upper, 0.0, 1.0)
    lower = np.where(at_zero, np.nan, lower)
    upper = np.where(at_zero, np.nan, upper)
    return lower, upper


@dataclass(frozen=True)
class _Block:
    """One stratum's fitted curve."""

    label: object
    time: Array
    n_risk: Array
    n_event: Array
    n_censor: Array
    surv: Array
    std_error: Array  # se(S), Greenwood
    conf_low: Array
    conf_high: Array
    cumhaz: Array
    cumhaz_var: Array


def _fit_blocks(surv: Surv, by: Any, weights: Any, conf_type: str, z: float) -> list[_Block]:
    et = event_table(surv, group=by, weights=weights)
    if et.strata is None:
        labels = [None]
        masks = [np.ones(len(et), dtype=bool)]
    else:
        labels = list(dict.fromkeys(et.strata.tolist()))
        masks = [et.strata == lab for lab in labels]

    blocks: list[_Block] = []
    for label, mask in zip(labels, masks, strict=True):
        n = et.n_risk[mask].astype(float)
        d = et.n_event[mask].astype(float)
        c = et.n_censor[mask].astype(float)
        t = et.time[mask].astype(float)

        with np.errstate(divide="ignore", invalid="ignore"):
            factor = np.where(n > 0, 1.0 - d / n, 1.0)
        surv_hat = np.cumprod(factor)

        denom = n * (n - d)
        with np.errstate(divide="ignore", invalid="ignore"):
            # When everyone remaining fails (n == d, S -> 0) the Greenwood term is
            # infinite, so the variance and se are undefined there, matching R.
            increment = np.where(denom > 0, d / denom, np.inf)
        greenwood = np.cumsum(increment)  # Var(log S)
        sigma = np.sqrt(greenwood)
        with np.errstate(invalid="ignore"):
            std_error = surv_hat * sigma  # 0 * inf -> nan at S == 0, as in R

        conf_low, conf_high = _km_confidence(surv_hat, sigma, conf_type, z)

        cumhaz = np.cumsum(np.where(n > 0, d / n, 0.0))
        cumhaz_var = np.cumsum(np.where(n > 0, d / n**2, 0.0))

        blocks.append(
            _Block(
                label=label,
                time=t,
                n_risk=n,
                n_event=d,
                n_censor=c,
                surv=surv_hat,
                std_error=std_error,
                conf_low=conf_low,
                conf_high=conf_high,
                cumhaz=cumhaz,
                cumhaz_var=cumhaz_var,
            )
        )
    return blocks


def _crossing_time(time: Array, curve: Array, level: float) -> float:
    """First time at which a monotone-decreasing `curve` drops to <= `level`."""
    hit = np.nonzero(curve <= level)[0]
    return float(time[hit[0]]) if hit.size else float("nan")


def _rmst_block(block: _Block, tau: float) -> tuple[float, float]:
    """Restricted mean survival time up to `tau` and its standard error.

    RMST is the area under the Kaplan-Meier curve on `[0, tau]`. The variance uses the
    standard estimator `sum_i A_i^2 d_i / (n_i (n_i - d_i))` over event times `t_i <=
    tau`, where `A_i` is the area under the curve from `t_i` to `tau` (as in R's
    `survival::survfit` restricted mean).
    """
    t = block.time
    s = block.surv
    n = block.n_risk
    d = block.n_event

    # Piecewise-constant segments: height on [start, next_start). The curve is 1 on
    # [0, t[0]) and s[i] on [t[i], t[i+1]); the final segment runs to tau.
    starts = np.concatenate([[0.0], t])
    heights = np.concatenate([[1.0], s])
    next_starts = np.concatenate([t, [tau]])
    widths = np.clip(np.minimum(next_starts, tau) - np.minimum(starts, tau), 0.0, None)
    seg_area = heights * widths
    rmst = float(seg_area.sum())

    # A_i = area from each event time to tau (reverse-cumulative segment areas).
    area_from = np.cumsum(seg_area[::-1])[::-1][1:]  # aligned with t
    with np.errstate(divide="ignore", invalid="ignore"):
        contrib = np.where((t <= tau) & (n - d > 0), area_from**2 * d / (n * (n - d)), 0.0)
    se = float(np.sqrt(contrib.sum()))
    return rmst, se


def _rmrl_block(block: _Block, s: float, tau: float) -> tuple[float, float]:
    """Restricted mean residual life at `s` over the window `(s, tau]`, and its SE.

    RMRL(s; tau) = (1 / S(s)) * integral_s^tau S(u) du, the expected additional survival
    beyond `s` restricted to `tau`, given survival to `s`. The variance is the restricted-mean
    (Greenwood) estimator applied to the conditional curve, summed over event times in
    `(s, tau]`; at `s = 0` this reduces exactly to `_rmst_block` (S(0) = 1).
    """
    t = block.time
    surv = block.surv
    n = block.n_risk
    d = block.n_event

    # Survival at s (right-continuous KM value at the last event time <= s).
    idx = int(np.searchsorted(t, s, side="right")) - 1
    s_at = float(surv[idx]) if idx >= 0 else 1.0
    if s_at <= 0.0:
        return float("nan"), float("nan")  # everyone has failed by s; RMRL undefined

    starts = np.concatenate([[0.0], t])
    heights = np.concatenate([[1.0], surv])
    next_starts = np.concatenate([t, [tau]])

    # Area under S(u) over the window [s, tau].
    lo = np.clip(starts, s, tau)
    hi = np.clip(next_starts, s, tau)
    area_window = float((heights * np.clip(hi - lo, 0.0, None)).sum())
    rmrl = area_window / s_at

    # A_i = area from each event time to tau (full curve); variance uses times in (s, tau].
    seg_area = heights * np.clip(
        np.minimum(next_starts, tau) - np.minimum(starts, tau), 0.0, None
    )
    area_from = np.cumsum(seg_area[::-1])[::-1][1:]  # aligned with t
    with np.errstate(divide="ignore", invalid="ignore"):
        contrib = np.where(
            (t > s) & (t <= tau) & (n - d > 0), area_from**2 * d / (n * (n - d)), 0.0
        )
    se = float(np.sqrt(contrib.sum())) / s_at
    return rmrl, se


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

    def __repr__(self) -> str:
        if getattr(self, "_blocks", None) is None:
            return f"KaplanMeier(conf_type={self.conf_type!r}) <unfitted>"
        from ._repr import align_table, whole

        df = self.to_dataframe()
        lcl, ucl = f"{self.conf_level}LCL", f"{self.conf_level}UCL"
        headers = ["n", "events", "median", lcl, ucl]
        if self._grouped:
            med = self.median(ci=True)
            labels, rows = [], []
            for label, g in df.groupby("strata", sort=False):
                m, lo, hi = med[label]
                labels.append(str(label))
                rows.append(
                    [
                        whole(g["n_risk"].iloc[0]),
                        whole(g["n_event"].sum()),
                        whole(m),
                        whole(lo),
                        whole(hi),
                    ]
                )
            table = align_table(headers, rows, labels)
        else:
            m, lo, hi = self.median(ci=True)
            row = [
                whole(df["n_risk"].iloc[0]),
                whole(df["n_event"].sum()),
                whole(m),
                whole(lo),
                whole(hi),
            ]
            table = align_table(headers, [row])
        return "KaplanMeier (Kaplan-Meier survival estimate)\n\n" + table

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

    def rmst(self, tau: float, *, ci: bool = False) -> Any:
        """Restricted mean survival time up to `tau` (area under the curve).

        With `ci=True`, return `(rmst, lower, upper)` using a normal approximation
        (`rmst +/- z * se`, lower bounded at 0). For a single stratum a scalar (or
        3-tuple) is returned; when stratified, a dict keyed by stratum label.
        """
        z = float(norm.ppf(1.0 - (1.0 - self.conf_level) / 2.0))

        def one(b: _Block) -> Any:
            value, se = _rmst_block(b, float(tau))
            if not ci:
                return value
            return (value, max(0.0, value - z * se), value + z * se)

        if not self._grouped:
            return one(self._blocks[0])
        return {b.label: one(b) for b in self._blocks}

    def rmrl(self, s: float, tau: float, *, ci: bool = False) -> Any:
        """Restricted mean residual life at time `s`, over the window `(s, tau]`.

        This is the expected additional survival time beyond `s`, restricted to `tau` and
        conditional on having survived to `s`: `RMRL(s; tau) = integral_s^tau S(u) du / S(s)`.
        It generalizes `rmst` (which is `rmrl(0, tau)`) to a later landmark time.

        With `ci=True`, return `(rmrl, lower, upper)` using a normal approximation
        (`rmrl +/- z * se`, lower bounded at 0). For a single stratum a scalar (or 3-tuple)
        is returned; when stratified, a dict keyed by stratum label. If everyone has failed by
        `s`, the value is `nan`.
        """
        if tau <= s:
            raise ValueError(f"tau ({tau}) must be greater than s ({s}).")
        if s < 0.0:
            raise ValueError(f"s must be non-negative, got {s}.")
        z = float(norm.ppf(1.0 - (1.0 - self.conf_level) / 2.0))

        def one(b: _Block) -> Any:
            value, se = _rmrl_block(b, float(s), float(tau))
            if not ci:
                return value
            return (value, max(0.0, value - z * se), value + z * se)

        if not self._grouped:
            return one(self._blocks[0])
        return {b.label: one(b) for b in self._blocks}

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


def _tidy_kaplan_meier(km: KaplanMeier, **_: Any) -> Any:
    """broom-style `tidy`: one row per time point (`estimate` is survival)."""
    return km.to_dataframe()


def _glance_kaplan_meier(km: KaplanMeier, **_: Any) -> Any:
    """broom-style `glance`: one row per stratum with counts and median survival."""
    import pandas as pd

    rows: list[dict[str, Any]] = []
    for b in km._blocks:
        row: dict[str, Any] = {}
        if km._grouped:
            row["strata"] = b.label
        row["n_start"] = float(b.n_risk[0]) if b.n_risk.size else float("nan")
        row["events"] = float(b.n_event.sum())
        row["median"] = _crossing_time(b.time, b.surv, 0.5)
        row["median_lower"] = _crossing_time(b.time, b.conf_low, 0.5)
        row["median_upper"] = _crossing_time(b.time, b.conf_high, 0.5)
        rows.append(row)
    return pd.DataFrame(rows)


class NelsonAalen:
    """Nelson-Aalen estimator of the cumulative hazard.

    The cumulative hazard is the running sum of `d / n` over event times, with Aalen
    variance `sum(d / n^2)`. Confidence limits use the `conf_type` transform (`"plain"`
    or `"log"`).
    """

    def __init__(self, *, conf_type: str = "log", conf_level: float = 0.95) -> None:
        if conf_type not in ("plain", "log"):
            raise ValueError(f"conf_type must be 'plain' or 'log', got {conf_type!r}.")
        if not 0.0 < conf_level < 1.0:
            raise ValueError(f"conf_level must be in (0, 1), got {conf_level}.")
        self.conf_type = conf_type
        self.conf_level = conf_level

    def __repr__(self) -> str:
        if getattr(self, "_blocks", None) is None:
            return f"NelsonAalen(conf_type={self.conf_type!r}) <unfitted>"
        from ._repr import align_table, num, whole

        df = self.to_dataframe()
        headers = ["n", "events", "max cumhaz"]
        if self._grouped:
            labels, rows = [], []
            for label, g in df.groupby("strata", sort=False):
                labels.append(str(label))
                rows.append(
                    [
                        whole(g["n_risk"].iloc[0]),
                        whole(g["n_event"].sum()),
                        num(g["estimate"].iloc[-1]),
                    ]
                )
            table = align_table(headers, rows, labels)
        else:
            row = [
                whole(df["n_risk"].iloc[0]),
                whole(df["n_event"].sum()),
                num(df["estimate"].iloc[-1]),
            ]
            table = align_table(headers, [row])
        return "NelsonAalen (Nelson-Aalen cumulative hazard estimate)\n\n" + table

    def fit(self, surv: Surv, *, by: Any = None, weights: Any = None) -> NelsonAalen:
        """Fit the estimator to a `Surv` response, optionally stratified by `by`."""
        z = float(norm.ppf(1.0 - (1.0 - self.conf_level) / 2.0))
        self._blocks = _fit_blocks(surv, by, weights, "log", z)
        self._grouped = by is not None
        self._z = z
        return self

    def _concat(self, attr: str) -> Array:
        return np.concatenate([getattr(b, attr) for b in self._blocks])

    @property
    def time_(self) -> Array:
        return self._concat("time")

    @property
    def cumhaz_(self) -> Array:
        return self._concat("cumhaz")

    @property
    def std_error_(self) -> Array:
        return np.sqrt(self._concat("cumhaz_var"))

    @property
    def strata_(self) -> Array | None:
        if not self._grouped:
            return None
        return np.concatenate(
            [np.full(b.time.shape[0], b.label, dtype=object) for b in self._blocks]
        )

    def to_dataframe(self, backend: str = "pandas") -> Any:
        """Return the fitted cumulative hazard as a tidy frame."""
        cumhaz = self.cumhaz_
        se = self.std_error_
        with np.errstate(divide="ignore", invalid="ignore"):
            if self.conf_type == "plain":
                lower = cumhaz - self._z * se
                upper = cumhaz + self._z * se
            else:  # log
                factor = np.where(cumhaz > 0, np.exp(self._z * se / cumhaz), 1.0)
                lower = cumhaz / factor
                upper = cumhaz * factor
        lower = np.clip(lower, 0.0, None)

        cols: dict[str, Array] = {}
        if self._grouped:
            cols["strata"] = self.strata_  # type: ignore[assignment]
        cols["time"] = self.time_
        cols["n_risk"] = self._concat("n_risk")
        cols["n_event"] = self._concat("n_event")
        cols["estimate"] = cumhaz
        cols["std_error"] = se
        cols["conf_low"] = lower
        cols["conf_high"] = upper
        if backend == "pandas":
            import pandas as pd

            return pd.DataFrame(cols)
        if backend == "polars":
            import polars as pl

            return pl.DataFrame(cols)
        raise ValueError(f"Unknown backend {backend!r}; use 'pandas' or 'polars'.")


# Register the tidy/glance adapters so `greenwood.tidy.tidy(km)` and
# `great_summaries.tidy` can consume a fitted Kaplan-Meier curve.
def _register_adapters() -> None:
    from .tidy import register_glance, register_tidier

    register_tidier("greenwood._nonparametric.KaplanMeier", _tidy_kaplan_meier)
    register_glance("greenwood._nonparametric.KaplanMeier", _glance_kaplan_meier)


_register_adapters()
