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

from ._backends import to_dataframe
from ._core import event_table

if TYPE_CHECKING:
    from ._surv import Surv

__all__ = ["KaplanMeier", "NelsonAalen"]

Array = npt.NDArray[Any]

_CONF_TYPES = frozenset({"plain", "log", "log-log"})


def _km_confidence(surv: Array, sigma: Array, conf_type: str, z: float) -> tuple[Array, Array]:
    r"""Confidence limits for the survival function on the requested scale.

    `sigma` is the standard error of $\log S$ (the square root of Greenwood's sum), so
    $\mathrm{se}(S) = S \cdot \sigma$. Limits are clipped to $[0, 1]$, matching R's `survfit`.
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
    r"""Restricted mean survival time up to `tau` and its standard error.

    RMST is the area under the Kaplan-Meier curve on $[0, \tau]$. The variance uses the
    standard estimator

    $$
    \sum_i \frac{A_i^2 \, d_i}{n_i (n_i - d_i)}
    $$

    over event times $t_i \le \tau$, where $A_i$ is the area under the curve from $t_i$ to
    $\tau$ (as in R's `survival::survfit` restricted mean).
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
    r"""Restricted mean residual life at `s` over the window $(s, \tau]$, and its SE.

    $$
    \mathrm{RMRL}(s; \tau) = \frac{1}{S(s)} \int_s^\tau S(u) \, du
    $$

    the expected additional survival beyond $s$ restricted to $\tau$, given survival to $s$.
    The variance is the restricted-mean (Greenwood) estimator applied to the conditional
    curve, summed over event times in $(s, \tau]$; at $s = 0$ this reduces exactly to
    `_rmst_block` ($S(0) = 1$).
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
    seg_area = heights * np.clip(np.minimum(next_starts, tau) - np.minimum(starts, tau), 0.0, None)
    area_from = np.cumsum(seg_area[::-1])[::-1][1:]  # aligned with t
    with np.errstate(divide="ignore", invalid="ignore"):
        contrib = np.where(
            (t > s) & (t <= tau) & (n - d > 0), area_from**2 * d / (n * (n - d)), 0.0
        )
    se = float(np.sqrt(contrib.sum())) / s_at
    return rmrl, se


class KaplanMeier:
    r"""Kaplan-Meier product-limit estimator of the survival function.

    The Kaplan-Meier estimator is a non-parametric method to estimate the survival function
    from right-censored data. It computes the survival probability at each observed event time
    as the product of conditional survival probabilities, accounting for subjects still at risk.
    This is the most widely used method for survival analysis and is the starting point for
    comparing survival between groups or assessing model fit.

    To use this estimator, call `fit()` with a right-censored `Surv` response (built with
    `Surv.right()`). The estimator computes survival probabilities, standard errors, and
    confidence intervals at each unique event time. Results can be accessed as aligned
    arrays, exported to pandas/polars/pyarrow DataFrames, or queried through methods like
    `median()`, `quantile()`, and `predict()`.

    The implementation uses the product-limit formula

    $$
    S(t) = \prod_{t_i \le t} \frac{n_i - d_i}{n_i}
    $$

    where $n_i$ is the number at risk and $d_i$ is the number of events at time $t_i$.
    Variance uses Greenwood's formula, and confidence intervals can be constructed on the
    log, log-log, or identity scale.

    Parameters
    ----------
    conf_type
        Confidence-interval transform: `"log"` (default, as in R's `survfit`), `"plain"`,
        or `"log-log"`.
    conf_level
        Confidence level for the interval (default 0.95).

    Returns
    -------
    Fitted estimator
        Call `fit()` to produce a fitted estimator with cached results (`time_`, `surv_`,
        `std_error_`, `conf_low_`, `conf_high_`, `n_risk_`, `n_event_`, `n_censor_`),
        accessible as aligned arrays or exported to DataFrames.

    Details
    -------
    Call `fit` with a `Surv` response. Results are exposed as aligned arrays (`time_`,
    `survival_`, `std_error_`, `conf_low_`, `conf_high_`, `strata_`), as tidy frames via
    `to_frame()` (optionally `format=`), and through `median`, `quantile`, and `predict`.

    Examples
    --------
    Build a `Surv` response from the bundled `lung` dataset and fit the estimator. Printing
    the fitted object reports the median survival and its confidence interval.

    ```{python}
    import greenwood as gw

    lung = gw.load_dataset("lung", backend="polars")
    y = gw.Surv.right(lung["time"], event=(lung["status"] == 2))
    km = gw.KaplanMeier().fit(y)
    km
    ```

    The full step function, one row per event time, is available with `to_frame`. Pass
    `format=` to choose the backend (here, Polars):

    ```{python}
    km.to_frame(format="polars")
    ```
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

        lcl, ucl = f"{self.conf_level}LCL", f"{self.conf_level}UCL"
        headers = ["n", "events", "median", lcl, ucl]
        if self._grouped:
            med = self.median(ci=True)
            labels, rows = [], []
            for b in self._blocks:
                m, lo, hi = med[b.label]
                labels.append(str(b.label))
                rows.append(
                    [
                        whole(b.n_risk[0]),
                        whole(b.n_event.sum()),
                        whole(m),
                        whole(lo),
                        whole(hi),
                    ]
                )
            table = align_table(headers, rows, labels)
        else:
            m, lo, hi = self.median(ci=True)
            b = self._blocks[0]
            row = [
                whole(b.n_risk[0]),
                whole(b.n_event.sum()),
                whole(m),
                whole(lo),
                whole(hi),
            ]
            table = align_table(headers, [row])
        return "KaplanMeier (Kaplan-Meier survival estimate)\n\n" + table

    def fit(self, surv: Surv, *, by: Any = None, weights: Any = None) -> KaplanMeier:
        r"""Fit the Kaplan-Meier estimator to survival data.

        Computes the product-limit survival estimate from a `Surv` response (time-to-event
        data, possibly right-censored). The estimator remains in the fitted object after
        calling `fit()`. Access it via attributes like `surv`, `time`, `n_risk`, etc., or
        access raw tables with `to_frame()` (optionally `format=`). Pass `by=` to
        produce separate curves per group (stratified analysis). Each group's fit is stored
        independently and can be visualized with `plot_survival()`.

        The fit is exact and no distributional assumptions are made. Optionally supply
        `weights=` (e.g., inverse-probability-of-censoring weights from the survey literature)
        to adjust for selection bias or survey design. Confidence intervals use the method
        specified at instantiation (`conf_type`), typically Greenwood's variance estimator.

        Parameters
        ----------
        surv
            A `Surv` response (typically right-censored, but supports counting-process and
            other forms). Built from data using `Surv.right()`, `Surv.interval()`, etc.
        by
            Optional grouping variable (e.g., a column or array). Produces one fit (one curve)
            per unique value of `by`, enabling stratified Kaplan-Meier analysis. Each group's
            results are stored and can be accessed separately via `to_frame()`, or
            visualized as separate curves via `plot_survival()`. Default (`None`): fit a
            single, unstratified curve.
        weights
            Optional weights (e.g., from survey design or inverse-probability-of-censoring
            adjustments). Must have the same length as `surv`. Default (`None`): unit weights.

        Returns
        -------
        KaplanMeier
            The fitted estimator object itself (for method chaining) with cached results
            (`time_`, `surv_`, `conf_low_`, `conf_high_`, `n_risk_`, `n_event_`, etc. as
            attributes).

        Details
        -------
        The Kaplan-Meier estimator is a non-parametric maximum likelihood estimator of the
        survival function $S(t)$. It is defined as the product of $(1 - d/n)$ over all event
        times up to $t$, where $d$ is the number of events and $n$ is the number at risk at
        each time. Confidence intervals are point-wise. They do not guarantee that the true
        curve lies entirely within the band.

        Examples
        --------
        Fit a single (unstratified) survival curve on the bundled `lung` dataset:

        ```{python}
        import greenwood as gw

        lung = gw.load_dataset("lung", backend="polars")
        y = gw.Surv.right(lung["time"], event=(lung["status"] == 2))
        km = gw.KaplanMeier().fit(y)
        km
        ```

        Fit stratified curves by sex by passing `by=lung["sex"]`. This produces one curve per
        group. The results are stored and can be visualized separately:

        ```{python}
        km_stratified = gw.KaplanMeier().fit(y, by=lung["sex"])
        gw.plot_survival(km_stratified)
        ```
        """
        z = float(norm.ppf(1.0 - (1.0 - self.conf_level) / 2.0))
        self._blocks = _fit_blocks(surv, by, weights, self.conf_type, z)
        self._grouped = by is not None
        return self

    # -- aligned-array accessors ---------------------------------------------

    def _concat(self, attr: str) -> Array:
        return np.concatenate([getattr(b, attr) for b in self._blocks])

    @property
    def time_(self) -> Array:
        """Event times at which the survival estimate changes (one entry per step)."""
        return self._concat("time")

    @property
    def survival_(self) -> Array:
        """Kaplan–Meier survival estimates at each event time."""
        return self._concat("surv")

    @property
    def std_error_(self) -> Array:
        """Standard errors of the survival estimates (Greenwood's formula)."""
        return self._concat("std_error")

    @property
    def conf_low_(self) -> Array:
        """Lower confidence limits for the survival estimates."""
        return self._concat("conf_low")

    @property
    def conf_high_(self) -> Array:
        """Upper confidence limits for the survival estimates."""
        return self._concat("conf_high")

    @property
    def cumhaz_(self) -> Array:
        """Nelson–Aalen cumulative hazard estimates at each event time."""
        return self._concat("cumhaz")

    @property
    def strata_(self) -> Array | None:
        """Stratum labels for each row, or `None` for unstratified fits."""
        if not self._grouped:
            return None
        return np.concatenate(
            [np.full(b.time.shape[0], b.label, dtype=object) for b in self._blocks]
        )

    # -- quantiles ------------------------------------------------------------

    def quantile(self, p: float, *, ci: bool = False) -> Any:
        r"""Return the `p`-quantile survival time per stratum.

        Computes the quantile (percentile) of the survival time distribution, i.e., the time
        at which the survival curve first drops to (1 - p). For example, `p=0.25` returns the
        25th percentile (first-quartile time: the time by which 25% of subjects have
        experienced the event). Useful for reporting clinically meaningful landmarks.

        Parameters
        ----------
        p
            Quantile level between 0 and 1. For example, `p=0.5` is the median, `p=0.25` is
            the first quartile, `p=0.75` is the third quartile.
        ci
            If `True`, return (estimate, lower, upper) confidence limits by inverting the
            survival confidence band (follows R's `quantile.survfit` convention). If `False`
            (default), return only the point estimate.

        Returns
        -------
        float or tuple or dict
            For a single stratum: a float (point estimate) or 3-tuple of floats
            (estimate, lower, upper) if `ci=True`. For stratified fits: a dict keyed by
            stratum label, with values as above. If the survival curve never drops to
            (1 - p), the quantile is `nan`.

        Details
        -------
        The quantile is found by inverting the step-function survival curve: the smallest time
        $t$ such that $S(t) \le (1 - p)$. Confidence intervals are obtained by inverting the
        pointwise confidence band, following R's convention. These are not simultaneous
        confidence intervals.

        Examples
        --------
        Any quantile of the survival distribution is available. Here is the first-quartile
        survival time (the time by which a quarter of subjects have had the event), with its
        confidence limits:

        ```{python}
        import greenwood as gw

        lung = gw.load_dataset("lung", backend="polars")
        y = gw.Surv.right(lung["time"], event=(lung["status"] == 2))
        km = gw.KaplanMeier().fit(y)
        km.quantile(0.25, ci=True)
        ```
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
        """Median survival time per stratum (the 0.5-quantile).

        Computes the median survival time: the time at which the survival curve first drops
        to 0.5, meaning 50% of subjects have experienced the event. A key clinical summary
        statistic when comparing survival across groups or evaluating prognosis.

        Parameters
        ----------
        ci
            If `True`, return (estimate, lower, upper) confidence limits by inverting the
            survival confidence band. If `False` (default), return only the point estimate.

        Returns
        -------
        float or tuple or dict
            For a single stratum: a float (point estimate) or 3-tuple of floats
            (estimate, lower, upper) if `ci=True`. For stratified fits: a dict keyed by
            stratum label, with values as above. If the survival curve never drops to 0.5,
            the median is `nan`.

        Details
        -------
        The median is a convenience wrapper around `quantile(0.5, ci=ci)`. It is the
        time-to-event value that divides the cohort into two equal halves (in terms of
        probability of experiencing the event). Unlike parametric models, the non-parametric
        median may not be uniquely defined if the curve jumps over 0.5; by convention, the
        first time the curve reaches or falls below 0.5 is returned.

        Examples
        --------
        The median is the time at which the survival curve first drops to 0.5. Pass
        `ci=True` for its confidence limits:

        ```{python}
        import greenwood as gw

        lung = gw.load_dataset("lung", backend="polars")
        y = gw.Surv.right(lung["time"], event=(lung["status"] == 2))
        km = gw.KaplanMeier().fit(y)
        km.median(ci=True)
        ```
        """
        return self.quantile(0.5, ci=ci)

    def rmst(self, tau: float, *, ci: bool = False) -> Any:
        r"""Restricted mean survival time up to `tau` (area under the survival curve).

        Computes the restricted mean survival time: the expected survival time over a fixed
        time window [0, tau], calculated as the area under the survival curve up to tau.
        Unlike median or quantiles, RMST uses all available follow-up information in the
        window, making it robust and easily interpretable as the average survival time over
        tau (e.g., 1-year mean survival, 5-year mean survival).

        Parameters
        ----------
        tau
            The upper time limit for the restriction. Must be positive. Typically chosen as
            a clinically relevant horizon (e.g., 1, 5, or 10 years).
        ci
            If `True`, return (estimate, lower, upper) confidence limits using a normal
            approximation ($\text{estimate} \pm z \cdot \text{se}$, with lower bound at 0).
            If `False` (default), return only the point estimate.

        Returns
        -------
        float or tuple or dict
            For a single stratum: a float (point estimate) or 3-tuple of floats
            (estimate, lower, upper) if `ci=True`. For stratified fits: a dict keyed by
            stratum label, with values as above.

        Details
        -------
        The restricted mean survival time is computed as the definite integral of $S(t)$
        from 0 to $\tau$:

        $$
        \mathrm{RMST}(\tau) = \int_0^\tau S(t) \, dt
        $$

        It is estimated numerically by integrating the step-function survival curve. Unlike
        the median, RMST is defined even when the survival curve does not reach 0.5, and is
        easily comparable across groups. Confidence intervals use the normal approximation
        with Greenwood-style variance estimation.

        Examples
        --------
        The restricted mean survival time is the average survival time over a fixed window,
        computed as the area under the curve up to `tau`. Here it is over the first 365 days,
        with its confidence limits:

        ```{python}
        import greenwood as gw

        lung = gw.load_dataset("lung", backend="polars")
        y = gw.Surv.right(lung["time"], event=(lung["status"] == 2))
        km = gw.KaplanMeier().fit(y)
        km.rmst(365, ci=True)
        ```
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
        r"""Restricted mean residual life at time `s`, over the window $(s, \tau]$.

        Computes the expected additional survival time beyond a landmark time `s`, conditional
        on having survived to `s`, restricted to an upper time limit `tau`. Mathematically:

        $$
        \mathrm{RMRL}(s; \tau) = \frac{\int_s^\tau S(u) \, du}{S(s)}
        $$

        This is a generalization of RMST to a later landmark point, useful for assessing
        prognosis or remaining life expectancy for subjects who have already reached a
        specific milestone.

        Parameters
        ----------
        s
            The landmark time. Must be non-negative. Represents the time at which subjects
            are assessed (e.g., time to remission, time at clinic visit, etc.).
        tau
            The upper time limit for the restriction. Must be greater than $s$. Typically a
            clinically relevant horizon beyond the landmark (e.g., $s = 180$ days landmark,
            $\tau = 730$ days endpoint).
        ci
            If `True`, return (estimate, lower, upper) confidence limits using a normal
            approximation ($\text{estimate} \pm z \cdot \text{se}$, with lower bound at 0).
            If `False` (default), return only the point estimate.

        Returns
        -------
        float or tuple or dict
            For a single stratum: a float (point estimate) or 3-tuple of floats
            (estimate, lower, upper) if `ci=True`. For stratified fits: a dict keyed by
            stratum label, with values as above. If everyone has failed by time `s`
            (i.e., S(s) = 0), the value is `nan`.

        Details
        -------
        The restricted mean residual life at a landmark time `s` measures the expected
        additional survival time for subjects who have survived to `s`, restricted to time
        `tau`. It generalizes RMST (which is equivalently rmrl(0, tau)). This is useful in
        clinical follow-up: given that a patient has survived to time `s`, what is the
        expected additional survival time? Variance estimation accounts for the conditioning
        on $S(s)$.

        Examples
        --------
        Restricted mean residual life is the expected additional survival time for subjects
        who have already survived to a landmark. Here it is at 180 days, over the window out
        to 730 days, with confidence limits:

        ```{python}
        import greenwood as gw

        lung = gw.load_dataset("lung", backend="polars")
        y = gw.Surv.right(lung["time"], event=(lung["status"] == 2))
        km = gw.KaplanMeier().fit(y)
        km.rmrl(180, 730, ci=True)
        ```
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
        r"""Evaluate the survival or cumulative hazard curve at specified times.

        Reads the estimated survival function or cumulative hazard off the step-function
        curve at any set of query times. Useful for extracting survival probabilities or
        hazard accumulation at clinically relevant time points (e.g., 1-year, 5-year
        survival).

        Parameters
        ----------
        times
            Query times at which to evaluate the curve. Can be a scalar or array-like of
            floats. Results are returned as a scalar or array matching the input shape.
        what
            Quantity to evaluate: `"survival"` (default) for survival probability $S(t)$, or
            `"cumhaz"` for cumulative hazard $H(t)$. Raises `ValueError` if any other value.

        Returns
        -------
        ndarray or dict
            For a single stratum: an array (or scalar if `times` is scalar) of estimated values
            at the query times. For stratified fits: a dict keyed by stratum label, with values
            as above.

        Details
        -------
        The survival and cumulative hazard curves are step functions defined only at observed
        event times. Values at times between events are interpolated using the right-continuous
        step-function convention: the value at time $t$ is the last step at time $\le t$. Times
        before the first event (or after the last observed time with non-zero survival) may
        return baseline values (1.0 for survival, 0.0 for cumulative hazard) or the last
        estimated value, respectively.

        Examples
        --------
        Read the survival probability off the curve at any set of times. Here are the
        estimated survival probabilities at 180, 365, and 730 days:

        ```{python}
        import greenwood as gw

        lung = gw.load_dataset("lung", backend="polars")
        y = gw.Surv.right(lung["time"], event=(lung["status"] == 2))
        km = gw.KaplanMeier().fit(y)
        km.predict([180, 365, 730])
        ```

        Pass `what="cumhaz"` instead to evaluate the cumulative hazard at those same times:

        ```{python}
        km.predict([180, 365, 730], what="cumhaz")
        ```
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

    def _table_columns(self) -> dict[str, Array]:
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
        return cols

    def to_frame(self, *, format: str | None = None) -> Any:
        """Return the fitted survival curve(s) as a DataFrame.

        Exports the Kaplan-Meier step function with one row per time point, including
        risk-set counts, the survival estimate, its standard error, confidence limits, and
        optional strata labels.

        Parameters
        ----------
        format
            Output format: `None` (default), `"pandas"`, `"polars"`, or `"pyarrow"`. When
            `None`, a backend is auto-detected (Polars, then Pandas, then PyArrow).

        Returns
        -------
        pandas.DataFrame, polars.DataFrame, or pyarrow.Table
            A tidy table with columns `time`, `n_risk`, `n_event`, `n_censor`, `estimate`,
            `std_error`, `conf_low`, `conf_high`, and optionally `strata`.

        Raises
        ------
        ImportError
            If the requested (or, when auto-detecting, any) DataFrame library is not
            installed.

        Examples
        --------
        Fit a Kaplan-Meier estimator on the bundled `lung` dataset, then export the fitted
        curve as a Polars frame:

        ```{python}
        import greenwood as gw

        lung = gw.load_dataset("lung", backend="polars")
        y = gw.Surv.right(lung["time"], event=(lung["status"] == 2))
        km = gw.KaplanMeier().fit(y)
        km.to_frame(format="polars")
        ```

        Pass a different `format=` for pandas or PyArrow output:

        ```{python}
        km.to_frame(format="pandas")
        ```
        """
        return to_dataframe(self._table_columns(), format=format)


def _tidy_kaplan_meier(km: KaplanMeier, *, format: str | None = None, **_: Any) -> Any:
    """broom-style `tidy`: one row per time point (`estimate` is survival)."""
    return km.to_frame(format=format)


def _glance_kaplan_meier(km: KaplanMeier, *, format: str | None = None, **_: Any) -> Any:
    """broom-style `glance`: one row per stratum with counts and median survival."""
    cols: dict[str, list[Any]] = {}
    if km._grouped:
        cols["strata"] = [b.label for b in km._blocks]
    cols["n_start"] = [float(b.n_risk[0]) if b.n_risk.size else float("nan") for b in km._blocks]
    cols["events"] = [float(b.n_event.sum()) for b in km._blocks]
    cols["median"] = [_crossing_time(b.time, b.surv, 0.5) for b in km._blocks]
    cols["median_lower"] = [_crossing_time(b.time, b.conf_low, 0.5) for b in km._blocks]
    cols["median_upper"] = [_crossing_time(b.time, b.conf_high, 0.5) for b in km._blocks]
    return to_dataframe(cols, format=format)


class NelsonAalen:
    r"""Nelson-Aalen estimator of the cumulative hazard.

    The Nelson-Aalen estimator provides a non-parametric estimate of the cumulative hazard
    function, which represents the total "accumulated risk" up to a given time. Unlike the
    Kaplan-Meier estimator which models survival directly, this approach models the force of
    mortality. The cumulative hazard at each event time is computed as a running sum of the
    ratio of events to subjects at risk:

    $$
    H(t) = \sum_{t_i \le t} \frac{d_i}{n_i}
    $$

    This estimator is useful when you want to examine the hazard directly rather than survival
    probabilities, and is often used as the basis for other analyses. You can convert the
    cumulative hazard to a survival estimate via $S(t) = \exp(-H(t))$, though the Kaplan-Meier
    estimator is typically preferred for direct survival estimation. Call `fit()` with a
    right-censored `Surv` response to compute cumulative hazard at each event time.

    The variance of the cumulative hazard estimate uses Aalen's formula:

    $$
    \mathrm{Var}(H(t)) = \sum_{t_i \le t} \frac{d_i}{n_i^2}
    $$

    Confidence intervals can be constructed on the plain or log scale, with the log scale
    providing better coverage in the tails.

    Parameters
    ----------
    conf_type
        Confidence-interval transform: `"plain"` (default for Nelson-Aalen) or `"log"`.
    conf_level
        Confidence level for the interval (default 0.95).

    Returns
    -------
    Fitted estimator
        Call `fit()` to produce a fitted estimator with cached results (`time_`,
        `cumulative_hazard_`, `std_error_`, `conf_low_`, `conf_high_`, `n_risk_`, `n_event_`,
        `n_censor_`), accessible as aligned arrays or exported to DataFrames.

    Details
    -------
    Call `fit()` with a `Surv` response. Results are exposed as aligned arrays, as tidy
    frames via `to_frame()` (optionally `format=`), and through the `predict()`,
    `quantile()`, and other methods.

    Examples
    --------
    Build a `Surv` response from the bundled `lung` dataset and fit the estimator. Printing
    the fitted object reports the counts and the maximum cumulative hazard reached.

    ```{python}
    import greenwood as gw

    lung = gw.load_dataset("lung", backend="polars")
    y = gw.Surv.right(lung["time"], event=(lung["status"] == 2))
    na = gw.NelsonAalen().fit(y)
    na
    ```
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

        headers = ["n", "events", "max cumhaz"]
        if self._grouped:
            labels, rows = [], []
            for b in self._blocks:
                labels.append(str(b.label))
                rows.append(
                    [
                        whole(b.n_risk[0]),
                        whole(b.n_event.sum()),
                        num(b.cumhaz[-1]),
                    ]
                )
            table = align_table(headers, rows, labels)
        else:
            b = self._blocks[0]
            row = [
                whole(b.n_risk[0]),
                whole(b.n_event.sum()),
                num(b.cumhaz[-1]),
            ]
            table = align_table(headers, [row])
        return "NelsonAalen (Nelson-Aalen cumulative hazard estimate)\n\n" + table

    def fit(self, surv: Surv, *, by: Any = None, weights: Any = None) -> NelsonAalen:
        r"""Fit the Nelson-Aalen estimator to survival data.

        Computes the cumulative hazard function $H(t)$ from a `Surv` response (time-to-event
        data). Like Kaplan-Meier, this is a non-parametric estimate requiring no distributional
        assumptions. The Nelson-Aalen estimator is an alternative to Kaplan-Meier. It estimates
        the cumulative hazard directly (sum of $d/n$ at each event time), from which the survival
        probability can be derived via $S(t) = \exp(-H(t))$. Results are stored in the fitted
        object. Access them via attributes or export to a DataFrame with `to_frame()`
        (optionally `format=`).

        Pass `by=` to produce separate cumulative hazard curves per group (stratified analysis),
        enabling covariate-free comparison of hazard accumulation across groups. Optionally
        supply `weights` to adjust for selection bias or survey design.

        Parameters
        ----------
        surv
            A `Surv` response (typically right-censored). Built from data using `Surv.right()`,
            `Surv.interval()`, etc.
        by
            Optional grouping variable (e.g., a column or array). Produces one fit (one
            cumulative hazard curve) per unique value of `by`, enabling stratified
            Nelson-Aalen analysis. Default (`None`): fit a single, unstratified curve.
        weights
            Optional weights (e.g., from survey design or inverse-probability-of-censoring
            adjustments). Must have the same length as `surv`. Default (`None`): unit weights.

        Returns
        -------
        NelsonAalen
            The fitted estimator object itself (for method chaining) with cached results
            (`time_`, `cumulative_hazard_`, `conf_low_`, `conf_high_`, `n_risk_`, `n_event_`
            as attributes).

        Details
        -------
        The Nelson-Aalen estimator is

        $$
        H(t) = \sum_{t_i \le t} \frac{d_i}{n_i}
        $$

        where $d_i$ and $n_i$ are events and number at risk at time $t_i$. Its variance is
        estimated using Aalen's formula:

        $$
        \mathrm{Var}(H) = \sum \frac{d_i}{n_i^2}
        $$

        The survival function can be recovered as $S(t) = \exp(-H(t))$. Confidence intervals
        are point-wise.

        Examples
        --------
        Fit a single (unstratified) cumulative hazard curve on the bundled `lung` dataset:

        ```{python}
        import greenwood as gw

        lung = gw.load_dataset("lung", backend="polars")
        y = gw.Surv.right(lung["time"], event=(lung["status"] == 2))
        na = gw.NelsonAalen().fit(y)
        na
        ```

        Fit stratified curves by sex to compare cumulative hazard accumulation:

        ```{python}
        na_stratified = gw.NelsonAalen().fit(y, by=lung["sex"])
        na_stratified
        ```
        """
        z = float(norm.ppf(1.0 - (1.0 - self.conf_level) / 2.0))
        self._blocks = _fit_blocks(surv, by, weights, "log", z)
        self._grouped = by is not None
        self._z = z
        return self

    def _concat(self, attr: str) -> Array:
        return np.concatenate([getattr(b, attr) for b in self._blocks])

    @property
    def time_(self) -> Array:
        """Event times at which the cumulative hazard estimate changes."""
        return self._concat("time")

    @property
    def cumhaz_(self) -> Array:
        """Nelson–Aalen cumulative hazard estimates at each event time."""
        return self._concat("cumhaz")

    @property
    def std_error_(self) -> Array:
        """Standard errors of the cumulative hazard estimates."""
        return np.sqrt(self._concat("cumhaz_var"))

    @property
    def strata_(self) -> Array | None:
        """Stratum labels for each row, or `None` for unstratified fits."""
        if not self._grouped:
            return None
        return np.concatenate(
            [np.full(b.time.shape[0], b.label, dtype=object) for b in self._blocks]
        )

    def _table_columns(self) -> dict[str, Array]:
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
        return cols

    def to_frame(self, *, format: str | None = None) -> Any:
        """Return the fitted cumulative hazard as a DataFrame.

        Exports the Nelson-Aalen estimate with one row per event time, including risk-set
        counts, the cumulative hazard estimate, its standard error, confidence limits, and
        optional strata labels.

        Parameters
        ----------
        format
            Output format: `None` (default), `"pandas"`, `"polars"`, or `"pyarrow"`. When
            `None`, a backend is auto-detected (Polars, then Pandas, then PyArrow).

        Returns
        -------
        pandas.DataFrame, polars.DataFrame, or pyarrow.Table
            A tidy table with columns `time`, `n_risk`, `n_event`, `estimate`,
            `std_error`, `conf_low`, `conf_high`, and optionally `strata`.

        Raises
        ------
        ImportError
            If the requested (or, when auto-detecting, any) DataFrame library is not
            installed.

        Examples
        --------
        Fit a Nelson-Aalen estimator on the bundled `lung` dataset, then export the fitted
        cumulative-hazard curve as a Polars frame:

        ```{python}
        import greenwood as gw

        lung = gw.load_dataset("lung", backend="polars")
        y = gw.Surv.right(lung["time"], event=(lung["status"] == 2))
        na = gw.NelsonAalen().fit(y)
        na.to_frame(format="polars")
        ```

        Pass a different `format=` for pandas or PyArrow output:

        ```{python}
        na.to_frame(format="pandas")
        ```
        """
        return to_dataframe(self._table_columns(), format=format)


# Register the tidy/glance adapters so `greenwood.tidy()` and
# `great_summaries.tidy` can consume a fitted Kaplan-Meier curve.
def _register_adapters() -> None:
    from .summaries import register_glance, register_tidier

    register_tidier("greenwood._nonparametric.KaplanMeier", _tidy_kaplan_meier)
    register_glance("greenwood._nonparametric.KaplanMeier", _glance_kaplan_meier)


_register_adapters()
