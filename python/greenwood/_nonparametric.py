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
from ._core import tabulate_groups
from ._outcome import bind_fit_inputs
from ._strata import resolve_strata

if TYPE_CHECKING:
    from ._outcome import Outcome
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

    label: str
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


def _robust_sigma(
    entry: Array,
    exit_: Array,
    event: Array,
    weight: Array,
    event_times: Array,
    n_risk: Array,
    n_event: Array,
    cluster_labels: Array | None = None,
) -> Array:
    r"""Infinitesimal-jackknife (IJ) standard error of $\log S$ at each event time.

    Replaces Greenwood's formula when `robust=True`. The influence of subject $i$ on $\log S(t)$ is

    $$
    U_i(t) = -\sum_{t_j \le t} \frac{\mathrm{d}M_i(t_j)}{n_j - d_j}
    $$

    where $\mathrm{d}M_i(t_j) = w_i \bigl[\mathbf{1}(\text{event}_i = t_j)
    - \mathbf{1}(\text{at risk at } t_j) \cdot d_j / n_j\bigr]$. The robust variance is
    $\sum_i U_i(t)^2$.

    When `cluster_labels=` is provided, per-subject influences are summed within each cluster before
    squaring, accounting for within-cluster correlation.
    """
    hazard = n_event / n_risk  # (K,)
    survivor = n_risk - n_event  # (K,)

    # (n, K) indicator matrices
    at_risk = (entry[:, None] < event_times[None, :]) & (exit_[:, None] >= event_times[None, :])
    event_here = (exit_[:, None] == event_times[None, :]) & event[:, None]

    # Martingale increments: dM_i(t_k) = w_i * [event_here - at_risk * hazard]
    dm = weight[:, None] * (event_here.astype(float) - at_risk.astype(float) * hazard[None, :])

    # Scale by 1/(n_k - d_k), setting to 0 where everyone fails (survivor == 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        scale = np.where(survivor > 0, 1.0 / survivor, 0.0)
    scaled = dm * scale[None, :]  # (n, K)

    # Cumulative influence on log S: IJ_i(t) = -cumsum of scaled increments
    ij = -np.cumsum(scaled, axis=1)  # (n, K)

    # For clustered data, sum per-subject influences within each cluster
    if cluster_labels is not None:
        levels = list(dict.fromkeys(cluster_labels.tolist()))
        ij = np.array([ij[cluster_labels == lev].sum(axis=0) for lev in levels])

    # Robust variance of log S at each event time
    var_logs = np.sum(ij**2, axis=0)  # (K,)
    return np.sqrt(var_logs)


def _resolve_weights(surv: Surv, weights: Any) -> Array:
    """Resolve weights from the explicit argument, or unit weights."""
    if weights is not None:
        from ._ingest import to_1d_array as _to_1d_array

        return _to_1d_array(weights)
    return np.ones(surv.n)


def _fit_blocks(
    surv: Surv,
    strata: Array,
    weights: Any,
    conf_type: str,
    z: float,
    *,
    robust: bool = False,
    cluster: Any = None,
) -> list[_Block]:
    # `strata` holds one curve label per subject ("all" for an ungrouped fit), so every fit has at
    # least one labeled block.
    et = tabulate_groups(surv, strata, weights)
    assert et.strata is not None
    labels: list[str] = list(dict.fromkeys(et.strata.tolist()))
    masks = [et.strata == lab for lab in labels]

    # Per-subject arrays needed for robust variance
    subj_entry: Array = np.empty(0)
    subj_exit: Array = np.empty(0)
    subj_event: Array = np.empty(0, dtype=bool)
    subj_weight: Array = np.empty(0)
    subj_cluster: Array | None = None
    if robust:
        subj_entry = surv.entry.astype(float)
        subj_exit = surv.stop.astype(float)
        subj_event = surv.event.astype(bool)
        subj_weight = _resolve_weights(surv, weights)
        if cluster is not None:
            from ._ingest import to_1d_array as _to_1d

            subj_cluster = _to_1d(cluster, dtype=object)

    blocks: list[_Block] = []
    for label, mask in zip(labels, masks, strict=True):
        n = et.n_risk[mask].astype(float)
        d = et.n_event[mask].astype(float)
        c = et.n_censor[mask].astype(float)
        t = et.time[mask].astype(float)

        with np.errstate(divide="ignore", invalid="ignore"):
            factor = np.where(n > 0, 1.0 - d / n, 1.0)
        surv_hat = np.cumprod(factor)

        if robust:
            smask = strata == label
            s_entry = subj_entry[smask]
            s_exit = subj_exit[smask]
            s_event = subj_event[smask]
            s_weight = subj_weight[smask]
            s_cluster = subj_cluster[smask] if subj_cluster is not None else None
            sigma = _robust_sigma(
                s_entry, s_exit, s_event, s_weight, t, n, d, cluster_labels=s_cluster
            )
        else:
            denom = n * (n - d)
            with np.errstate(divide="ignore", invalid="ignore"):
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


def _step_values(time: Array, curve: Array, query: Array, baseline: float) -> Array:
    """Read a right-continuous step function at `query`: the last step at a time <= t."""
    idx = np.searchsorted(time, query, side="right") - 1
    return np.where(idx >= 0, curve[idx.clip(min=0)], baseline)


def _crossing_time(time: Array, curve: Array, level: float) -> float:
    """First time at which a monotone-decreasing `curve` drops to <= `level`."""
    hit = np.nonzero(curve <= level)[0]
    return float(time[hit[0]]) if hit.size else float("nan")


def _block_quantile(block: _Block, p: float) -> tuple[float, float, float]:
    """The `p` quantile of one curve and its confidence limits."""
    level = 1.0 - p
    return (
        _crossing_time(block.time, block.surv, level),
        _crossing_time(block.time, block.conf_low, level),
        _crossing_time(block.time, block.conf_high, level),
    )


def _rmst_block(block: _Block, tau: float) -> tuple[float, float]:
    r"""Restricted mean survival time up to `tau` and its standard error.

    RMST is the area under the Kaplan-Meier curve on $[0, \tau]$. The variance uses the standard
    estimator

    $$
    \sum_i \frac{A_i^2 \, d_i}{n_i (n_i - d_i)}
    $$

    over event times $t_i \le \tau$, where $A_i$ is the area under the curve from $t_i$ to $\tau$
    (as in R's `survival::survfit` restricted mean).
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

    the expected additional survival beyond $s$ restricted to $\tau$, given survival to $s$. The
    variance is the restricted-mean (Greenwood) estimator applied to the conditional curve, summed
    over event times in $(s, \tau]$; at $s = 0$ this reduces exactly to `_rmst_block` ($S(0) = 1$).
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

    To use this estimator, call `~~greenwood.KaplanMeier.fit()` with a right-censored response: an
    `Outcome` such as `gw.Outcome.surv(time="time", event="status")` or a formula such as
    `"Surv(time, status) ~ sex"` together with `data=`, or a `Surv` built from values in hand. The
    estimator computes survival probabilities, standard errors, and confidence intervals at each
    unique event time. Results can be accessed as aligned arrays, exported to pandas/polars/pyarrow
    DataFrames, or queried through methods like `~~greenwood.KaplanMeier.median()`,
    `~~greenwood.KaplanMeier.quantile()`, and `~~greenwood.KaplanMeier.predict()`.

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
    robust
        Use the infinitesimal-jackknife (sandwich) variance instead of Greenwood's formula (the
        default is `False`). When `True`, standard errors and confidence intervals account for the
        correlation structure induced by weighted or correlated observations, matching R's
        `survfit(..., robust = TRUE)`.

    Returns
    -------
    Fitted estimator
        Call `~~greenwood.KaplanMeier.fit()` to produce a fitted estimator with cached results
        (`time_`, `surv_`, `std_error_`, `conf_low_`, `conf_high_`, `n_risk_`, `n_event_`,
        `n_censor_`), accessible as aligned arrays or exported to DataFrames.

    Details
    -------
    Call `~~greenwood.KaplanMeier.fit` with a `Surv` response. Results are exposed as aligned arrays
    (`time_`, `~~greenwood.KaplanMeier.survival_`, `std_error_`, `conf_low_`, `conf_high_`,
    `~~greenwood.KaplanMeier.strata_`), as tidy frames via `~~greenwood.KaplanMeier.to_frame()`
    (optionally `format=`), and through `~~greenwood.KaplanMeier.median`,
    `~~greenwood.KaplanMeier.quantile`, and `~~greenwood.KaplanMeier.predict`.

    Examples
    --------
    Name the response columns of the bundled `lung` dataset with `gw.Outcome.surv()` and fit the
    estimator with `data=`. Printing the fitted object reports the median survival and its
    confidence interval.

    ```{python}
    import greenwood as gw

    # Load data and name the response columns
    lung = gw.load_dataset("lung", backend="polars")
    death = gw.Outcome.surv(time="time", event="status")

    # Fit the Kaplan-Meier estimator
    km = gw.KaplanMeier().fit(death, data=lung)
    km
    ```

    The full step function, one row per event time, is available with
    `~~greenwood.KaplanMeier.to_frame`. Pass `format=` to choose the backend (here, Polars):

    ```{python}
    # Export the survival curve as a Polars DataFrame
    km.to_frame(format="polars")
    ```
    """

    def __init__(
        self, *, conf_type: str = "log", conf_level: float = 0.95, robust: bool = False
    ) -> None:
        if conf_type not in _CONF_TYPES:
            raise ValueError(f"conf_type must be one of {sorted(_CONF_TYPES)}, got {conf_type!r}.")
        if not 0.0 < conf_level < 1.0:
            raise ValueError(f"conf_level must be in (0, 1), got {conf_level}.")
        self.conf_type = conf_type
        self.conf_level = conf_level
        self.robust = robust

    def __repr__(self) -> str:
        if getattr(self, "_blocks", None) is None:
            return f"KaplanMeier(conf_type={self.conf_type!r}) <unfitted>"
        from ._repr import align_table, dropped_footer, whole

        lcl, ucl = f"{self.conf_level}LCL", f"{self.conf_level}UCL"
        headers = ["n", "events", "median", lcl, ucl]
        labels, rows = [], []
        for b in self._blocks:
            m, lo, hi = _block_quantile(b, 0.5)
            labels.append(b.label)
            rows.append(
                [whole(b.n_risk[0]), whole(b.n_event.sum()), whole(m), whole(lo), whole(hi)]
            )
        table = align_table(headers, rows, labels)
        return "KaplanMeier (Kaplan-Meier survival estimate)\n\n" + table + dropped_footer(self)

    def fit(
        self,
        surv: Surv | Outcome | str,
        *,
        data: Any = None,
        by: Any = None,
        weights: Any = None,
        cluster: Any = None,
    ) -> KaplanMeier:
        r"""Fit the Kaplan-Meier estimator to survival data.

        Computes the product-limit survival estimate from a `Surv` response (time-to-event
        data, possibly right-censored). The estimator remains in the fitted object after
        calling `~~greenwood.KaplanMeier.fit()`. Access it via attributes like `surv`{.gd-no-link},
        `time`, `n_risk`, etc., or access raw tables with `~~greenwood.KaplanMeier.to_frame()`
        (optionally `format=`). Pass `by=` to produce separate curves per group (stratified
        analysis). Each group's fit is stored independently and can be visualized with
        `plot_survival()`.

        The fit is exact and no distributional assumptions are made. Optionally supply
        `weights=` (e.g., inverse-probability-of-censoring weights from the survey literature)
        to adjust for selection bias or survey design. Confidence intervals use the method
        specified at instantiation (`conf_type`), typically Greenwood's variance estimator.

        Parameters
        ----------
        surv
            A `Surv` response (typically right-censored, but supports counting-process and
            other forms). Built with `Surv()`.
            An `Outcome` or a formula string such as `'Surv(time, status == 2) ~ sex'` is also
            accepted, with its columns read from `data`. The right-hand side names the `by`
            column(s).
        by
            Optional grouping variable (e.g., a column or array). Produces one fit (one curve) per
            unique value of `by`, enabling stratified Kaplan-Meier analysis. Each group's results
            are stored and can be accessed separately via `~~greenwood.KaplanMeier.to_frame()`, or
            visualized as separate curves via `plot_survival()`. Default (`None`) means to fit a
            single, unstratified curve.
        weights
            Optional weights (e.g., from survey design or inverse-probability-of-censoring
            adjustments). Must have the same length as `surv`{.gd-no-link}. Default (`None`): unit
            weights.
        cluster
            Optional cluster labels for grouped robust variance estimation. When provided,
            `robust=True` is implied. Per-subject influences are summed within each cluster before
            squaring, accounting for within-cluster correlation. Default (`None`) means no
            clustering.
        data
            A data frame (pandas, Polars, PyArrow, DuckDB, a lazy frame, ...) holding the columns
            named by the response, `by`, `weights`, and `cluster`. When `surv`{.gd-no-link} is an
            `Outcome` or a formula, rows with a missing value in any column used are dropped before
            fitting.

        Returns
        -------
        KaplanMeier
            The fitted estimator object itself (for method chaining) with cached results (`time_`,
            `surv_`, `conf_low_`, `conf_high_`, `n_risk_`, `n_event_`, etc. as attributes).

        Details
        -------
        The Kaplan-Meier estimator is a non-parametric maximum likelihood estimator of the survival
        function $S(t)$. It is defined as the product of $(1 - d/n)$ over all event times up to $t$,
        where $d$ is the number of events and $n$ is the number at risk at each time. Confidence
        intervals are point-wise. They do not guarantee that the true curve lies entirely within the
        band.

        Examples
        --------
        Fit a single (unstratified) survival curve on the bundled `lung` dataset:

        ```{python}
        import greenwood as gw

        # Load data and name the response columns
        lung = gw.load_dataset("lung", backend="polars")
        death = gw.Outcome.surv(time="time", event="status")

        # Fit a single unstratified survival curve
        km = gw.KaplanMeier().fit(death, data=lung)
        km
        ```

        Fit stratified curves by sex by passing `by="sex"`. This produces one curve per group, and
        the formula `"Surv(time, status) ~ sex"` gives the same fit. The results are stored and can
        be visualized separately:

        ```{python}
        # Fit stratified curves by sex and plot them
        km_stratified = gw.KaplanMeier().fit(death, by="sex", data=lung)
        gw.plot_survival(km_stratified)
        ```
        """
        bound = bind_fit_inputs(
            surv,
            data=data,
            labels={"by": by, "weights": weights, "cluster": cluster},
            rhs_to="by",
            estimator="KaplanMeier",
        )
        surv = bound.surv
        strata, self._grouped = resolve_strata(bound, "by")
        weights = bound.labels["weights"]
        cluster = bound.labels["cluster"]
        self._n_input = bound.n_input
        self.n_dropped_ = bound.n_dropped

        z = float(norm.ppf(1.0 - (1.0 - self.conf_level) / 2.0))
        use_robust = self.robust or cluster is not None
        self._blocks = _fit_blocks(
            surv, strata, weights, self.conf_type, z, robust=use_robust, cluster=cluster
        )
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
    def strata_(self) -> Array:
        """The curve label of each row: `"all"` for one curve, `"sex=1"` and so on for groups."""
        return np.concatenate(
            [np.full(b.time.shape[0], b.label, dtype=object) for b in self._blocks]
        )

    # -- quantiles ------------------------------------------------------------

    def quantile(self, p: Any, *, format: str | None = None) -> Any:
        r"""Return survival-time quantiles, with confidence limits, for each curve.

        The `p` quantile is the time by which a proportion `p` of subjects have had the event: the
        time at which the survival curve first drops to `1 - p`. For example, `p=0.25` gives the
        first-quartile time. Confidence limits come from inverting the pointwise confidence band,
        as R's `quantile.survfit()` does.

        Parameters
        ----------
        p
            One or more proportions between 0 and 1. For example, `0.5` is the median, and
            `[0.25, 0.5, 0.75]` gives the three quartiles.
        format
            Output format: `None` (default), `"pandas"`, `"polars"`, or `"pyarrow"`. When `None`,
            a backend is auto-detected (Polars, then Pandas, then PyArrow).

        Returns
        -------
        pandas.DataFrame, polars.DataFrame, or pyarrow.Table
            One row per curve and proportion, with columns `strata`, `prob`, `time`, `conf_low`,
            and `conf_high`. The layout is the same for one curve (`strata` is `"all"`) as for
            several. A quantile is `nan` when the curve (or a confidence limit) never drops to
            `1 - p`.

        Details
        -------
        The quantile is the smallest time $t$ with $S(t) \le 1 - p$. Its confidence limits are
        the same crossing times for the lower and upper confidence curves. These are pointwise,
        not simultaneous, intervals.

        Examples
        --------
        The three quartiles of survival time, with their confidence limits:

        ```{python}
        import greenwood as gw

        # Load data and fit the Kaplan-Meier estimator
        lung = gw.load_dataset("lung", backend="polars")
        death = gw.Outcome.surv(time="time", event="status")
        km = gw.KaplanMeier().fit(death, data=lung)

        # Survival-time quartiles with confidence limits
        km.quantile(p=[0.25, 0.5, 0.75], format="polars")
        ```
        """
        probs = [float(v) for v in np.atleast_1d(np.asarray(p, dtype=float))]
        if any(not 0.0 <= v <= 1.0 for v in probs):
            raise ValueError("p must be between 0 and 1.")
        cols: dict[str, list[Any]] = {
            k: [] for k in ("strata", "prob", "time", "conf_low", "conf_high")
        }
        for b in self._blocks:
            for prob in probs:
                point, lower, upper = _block_quantile(b, prob)
                cols["strata"].append(b.label)
                cols["prob"].append(prob)
                cols["time"].append(point)
                cols["conf_low"].append(lower)
                cols["conf_high"].append(upper)
        return to_dataframe(cols, format=format)

    def median(self, *, format: str | None = None) -> Any:
        """Median survival time, with confidence limits, for each curve.

        The median survival time is the time at which the survival curve first drops to 0.5,
        when half of subjects have had the event. It is a key summary when comparing survival
        across groups.

        Parameters
        ----------
        format
            Output format: `None` (default), `"pandas"`, `"polars"`, or `"pyarrow"`. When `None`,
            a backend is auto-detected (Polars, then Pandas, then PyArrow).

        Returns
        -------
        pandas.DataFrame, polars.DataFrame, or pyarrow.Table
            One row per curve, with columns `strata`, `time`, `conf_low`, and `conf_high`. The
            layout is the same for one curve (`strata` is `"all"`) as for several. The median is
            `nan` when the curve never drops to 0.5.

        Details
        -------
        This is `~~greenwood.KaplanMeier.quantile()` at `p=0.5`, without the `prob` column. When
        the curve jumps over 0.5, the first time it reaches or falls below 0.5 is used.

        Examples
        --------
        The median survival time for each sex, with confidence limits:

        ```{python}
        import greenwood as gw

        # Load data and fit one curve per sex
        lung = gw.load_dataset("lung", backend="polars")
        km = gw.KaplanMeier().fit("Surv(time, status) ~ sex", data=lung)

        # Median survival time with confidence limits
        km.median(format="polars")
        ```
        """
        cols: dict[str, list[Any]] = {k: [] for k in ("strata", "time", "conf_low", "conf_high")}
        for b in self._blocks:
            point, lower, upper = _block_quantile(b, 0.5)
            cols["strata"].append(b.label)
            cols["time"].append(point)
            cols["conf_low"].append(lower)
            cols["conf_high"].append(upper)
        return to_dataframe(cols, format=format)

    def rmst(self, tau: float, *, format: str | None = None) -> Any:
        r"""Restricted mean survival time up to `tau` (the area under the survival curve).

        The restricted mean survival time is the expected survival time over the window
        $[0, \tau]$: the area under the survival curve up to `tau`. Unlike the median, it uses
        all the follow-up in the window, is defined even when the curve never reaches 0.5, and
        reads as an average (for example, mean survival over the first year).

        Parameters
        ----------
        tau
            The end of the window. Typically a clinically relevant horizon (for example, 1, 5,
            or 10 years).
        format
            Output format: `None` (default), `"pandas"`, `"polars"`, or `"pyarrow"`. When `None`,
            a backend is auto-detected (Polars, then Pandas, then PyArrow).

        Returns
        -------
        pandas.DataFrame, polars.DataFrame, or pyarrow.Table
            One row per curve, with columns `strata`, `tau`, `estimate`, `std_error`, `conf_low`,
            and `conf_high`. The layout is the same for one curve (`strata` is `"all"`) as for
            several.

        Details
        -------
        $$
        \mathrm{RMST}(\tau) = \int_0^\tau S(t) \, dt
        $$

        is computed exactly from the step-function curve. The standard error is R's
        `survfit` restricted-mean estimator, and the confidence limits are
        $\text{estimate} \pm z \cdot \text{se}$, with the lower limit floored at 0.

        Examples
        --------
        The mean survival time over the first 365 days, with confidence limits:

        ```{python}
        import greenwood as gw

        # Load data and fit the Kaplan-Meier estimator
        lung = gw.load_dataset("lung", backend="polars")
        death = gw.Outcome.surv(time="time", event="status")
        km = gw.KaplanMeier().fit(death, data=lung)

        # Restricted mean survival time over 365 days
        km.rmst(tau=365, format="polars")
        ```
        """
        z = float(norm.ppf(1.0 - (1.0 - self.conf_level) / 2.0))
        cols: dict[str, list[Any]] = {
            k: [] for k in ("strata", "tau", "estimate", "std_error", "conf_low", "conf_high")
        }
        for b in self._blocks:
            value, se = _rmst_block(b, float(tau))
            cols["strata"].append(b.label)
            cols["tau"].append(float(tau))
            cols["estimate"].append(value)
            cols["std_error"].append(se)
            cols["conf_low"].append(max(0.0, value - z * se))
            cols["conf_high"].append(value + z * se)
        return to_dataframe(cols, format=format)

    def rmrl(self, s: float, tau: float, *, format: str | None = None) -> Any:
        r"""Restricted mean residual life at time `s`, over the window $(s, \tau]$.

        The expected additional survival time beyond a landmark `s`, for subjects who have
        survived to `s`, restricted to `tau`:

        $$
        \mathrm{RMRL}(s; \tau) = \frac{\int_s^\tau S(u) \, du}{S(s)}
        $$

        It generalizes the restricted mean survival time (which is `rmrl(0, tau)`) to a later
        landmark, for example the remaining life expectancy of patients who reached a milestone.

        Parameters
        ----------
        s
            The landmark time. Must be non-negative.
        tau
            The end of the window. Must be greater than `s`.
        format
            Output format: `None` (default), `"pandas"`, `"polars"`, or `"pyarrow"`. When `None`,
            a backend is auto-detected (Polars, then Pandas, then PyArrow).

        Returns
        -------
        pandas.DataFrame, polars.DataFrame, or pyarrow.Table
            One row per curve, with columns `strata`, `s`, `tau`, `estimate`, `std_error`,
            `conf_low`, and `conf_high`. The layout is the same for one curve (`strata` is
            `"all"`) as for several. The estimate is `nan` for a curve where everyone has had the
            event by `s`.

        Details
        -------
        The variance is the restricted-mean estimator applied to the conditional curve, summed
        over event times in $(s, \tau]$. The confidence limits are
        $\text{estimate} \pm z \cdot \text{se}$, with the lower limit floored at 0.

        Examples
        --------
        The expected additional survival time at 180 days, over the window out to 730 days:

        ```{python}
        import greenwood as gw

        # Load data and fit the Kaplan-Meier estimator
        lung = gw.load_dataset("lung", backend="polars")
        death = gw.Outcome.surv(time="time", event="status")
        km = gw.KaplanMeier().fit(death, data=lung)

        # Restricted mean residual life at 180 days
        km.rmrl(s=180, tau=730, format="polars")
        ```
        """
        if tau <= s:
            raise ValueError(f"tau ({tau}) must be greater than s ({s}).")
        if s < 0.0:
            raise ValueError(f"s must be non-negative, got {s}.")
        z = float(norm.ppf(1.0 - (1.0 - self.conf_level) / 2.0))
        cols: dict[str, list[Any]] = {
            k: [] for k in ("strata", "s", "tau", "estimate", "std_error", "conf_low", "conf_high")
        }
        for b in self._blocks:
            value, se = _rmrl_block(b, float(s), float(tau))
            cols["strata"].append(b.label)
            cols["s"].append(float(s))
            cols["tau"].append(float(tau))
            cols["estimate"].append(value)
            cols["std_error"].append(se)
            cols["conf_low"].append(max(0.0, value - z * se))
            cols["conf_high"].append(value + z * se)
        return to_dataframe(cols, format=format)

    # -- prediction -----------------------------------------------------------

    def predict(self, times: Any, *, what: str = "survival", format: str | None = None) -> Any:
        r"""Read the survival or cumulative hazard curve at given times, for each curve.

        Useful for survival probabilities at clinically relevant times (for example, 1-year and
        5-year survival) and for the at-risk tables under survival plots.

        Parameters
        ----------
        times
            One or more times at which to read the curve.
        what
            `"survival"` (default) for the survival probability $S(t)$, or `"cumhaz"` for the
            cumulative hazard $H(t)$.
        format
            Output format: `None` (default), `"pandas"`, `"polars"`, or `"pyarrow"`. When `None`,
            a backend is auto-detected (Polars, then Pandas, then PyArrow).

        Returns
        -------
        pandas.DataFrame, polars.DataFrame, or pyarrow.Table
            One row per curve and time, with columns `strata`, `time`, and `estimate`. The layout
            is the same for one curve (`strata` is `"all"`) as for several.

        Details
        -------
        The curves are right-continuous step functions, so the value at time $t$ is the last
        step at a time $\le t$. Before the first step it is 1 for survival and 0 for the
        cumulative hazard.

        Examples
        --------
        Survival probabilities at 180, 365, and 730 days:

        ```{python}
        import greenwood as gw

        # Load data and fit the Kaplan-Meier estimator
        lung = gw.load_dataset("lung", backend="polars")
        death = gw.Outcome.surv(time="time", event="status")
        km = gw.KaplanMeier().fit(death, data=lung)

        # Read survival probabilities at specific time points
        km.predict(times=[180, 365, 730], format="polars")
        ```

        Pass `what="cumhaz"` to read the cumulative hazard at the same times:

        ```{python}
        # Read the cumulative hazard at the same time points
        km.predict(times=[180, 365, 730], what="cumhaz", format="polars")
        ```
        """
        if what not in ("survival", "cumhaz"):
            raise ValueError(f"what must be 'survival' or 'cumhaz', got {what!r}.")
        query = np.atleast_1d(np.asarray(times, dtype=float))
        cols: dict[str, list[Any]] = {"strata": [], "time": [], "estimate": []}
        for b in self._blocks:
            curve = b.surv if what == "survival" else b.cumhaz
            baseline = 1.0 if what == "survival" else 0.0
            cols["strata"].extend([b.label] * query.shape[0])
            cols["time"].extend(query.tolist())
            cols["estimate"].extend(_step_values(b.time, curve, query, baseline).tolist())
        return to_dataframe(cols, format=format)

    # -- interop --------------------------------------------------------------

    def _table_columns(self) -> dict[str, Array]:
        cols: dict[str, Array] = {"strata": self.strata_}
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
        curve labels.

        Parameters
        ----------
        format
            Output format: `None` (default), `"pandas"`, `"polars"`, or `"pyarrow"`. When
            `None`, a backend is auto-detected (Polars, then Pandas, then PyArrow).

        Returns
        -------
        pandas.DataFrame, polars.DataFrame, or pyarrow.Table
            A tidy table with columns `strata`, `time`, `n_risk`, `n_event`, `n_censor`,
            `estimate`, `std_error`, `conf_low`, and `conf_high`. The layout is the same for one
            curve (`strata` is `"all"`) as for several (`"sex=1"`, `"sex=2"`).

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

        # Load data and fit the Kaplan-Meier estimator
        lung = gw.load_dataset("lung", backend="polars")
        death = gw.Outcome.surv(time="time", event="status")
        km = gw.KaplanMeier().fit(death, data=lung)

        # Export the survival curve as a Polars DataFrame
        km.to_frame(format="polars")
        ```

        Pass a different `format=` for pandas or PyArrow output:

        ```{python}
        # Export as a pandas DataFrame instead
        km.to_frame(format="pandas")
        ```
        """
        return to_dataframe(self._table_columns(), format=format)


def _tidy_kaplan_meier(km: KaplanMeier, *, format: str | None = None, **_: Any) -> Any:
    """broom-style `tidy`: one row per time point (`estimate` is survival)."""
    return km.to_frame(format=format)


def _glance_kaplan_meier(km: KaplanMeier, *, format: str | None = None, **_: Any) -> Any:
    """broom-style `glance`: one row per stratum with counts and median survival."""
    cols: dict[str, list[Any]] = {"strata": [b.label for b in km._blocks]}
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
    estimator is typically preferred for direct survival estimation. Call
    `~~greenwood.NelsonAalen.fit()` with a right-censored `Surv` response to compute cumulative
    hazard at each event time.

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
        Call `~~greenwood.NelsonAalen.fit()` to produce a fitted estimator with cached results
        (`~~greenwood.NelsonAalen.time_`, `cumulative_hazard_`,
        `~~greenwood.NelsonAalen.std_error_`, `conf_low_`{.gd-no-link}, `conf_high_`{.gd-no-link},
        `n_risk_`, `n_event_`, `n_censor_`), accessible as aligned arrays or exported to DataFrames.

    Details
    -------
    Call `~~greenwood.NelsonAalen.fit()` with a `Surv` response. Results are exposed as aligned
    arrays, as tidy frames via `~~greenwood.NelsonAalen.to_frame()` (optionally `format=`), and
    through the `predict()`{.gd-no-link}, `quantile()`{.gd-no-link}, and other methods.

    Examples
    --------
    Name the response columns of the bundled `lung` dataset with `gw.Outcome.surv()` and fit the
    estimator with `data=`. Printing the fitted object reports the counts and the maximum
    cumulative hazard reached.

    ```{python}
    import greenwood as gw

    # Load data and name the response columns
    lung = gw.load_dataset("lung", backend="polars")
    death = gw.Outcome.surv(time="time", event="status")

    # Fit the Nelson-Aalen estimator
    na = gw.NelsonAalen().fit(death, data=lung)
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
        from ._repr import align_table, dropped_footer, num, whole

        headers = ["n", "events", "max cumhaz"]
        labels, rows = [], []
        for b in self._blocks:
            labels.append(b.label)
            rows.append([whole(b.n_risk[0]), whole(b.n_event.sum()), num(b.cumhaz[-1])])
        table = align_table(headers, rows, labels)
        return (
            "NelsonAalen (Nelson-Aalen cumulative hazard estimate)\n\n"
            + table
            + dropped_footer(self)
        )

    def fit(
        self, surv: Surv | Outcome | str, *, data: Any = None, by: Any = None, weights: Any = None
    ) -> NelsonAalen:
        r"""Fit the Nelson-Aalen estimator to survival data.

        Computes the cumulative hazard function $H(t)$ from a `Surv` response (time-to-event
        data). Like Kaplan-Meier, this is a non-parametric estimate requiring no distributional
        assumptions. The Nelson-Aalen estimator is an alternative to Kaplan-Meier. It estimates
        the cumulative hazard directly (sum of $d/n$ at each event time), from which the survival
        probability can be derived via $S(t) = \exp(-H(t))$. Results are stored in the fitted
        object. Access them via attributes or export to a DataFrame with
        `~~greenwood.NelsonAalen.to_frame()` (optionally `format=`).

        Pass `by=` to produce separate cumulative hazard curves per group (stratified analysis),
        enabling covariate-free comparison of hazard accumulation across groups. Optionally
        supply `weights` to adjust for selection bias or survey design.

        Parameters
        ----------
        surv
            A `Surv` response (typically right-censored). Built with `Surv()`.
            An `Outcome` or a formula string such as `'Surv(time, status == 2) ~ sex'` is also
            accepted, with its columns read from `data`. The right-hand side names the `by`
            column(s).
        by
            Optional grouping variable (e.g., a column or array). Produces one fit (one
            cumulative hazard curve) per unique value of `by`, enabling stratified
            Nelson-Aalen analysis. Default (`None`): fit a single, unstratified curve.
        weights
            Optional weights (e.g., from survey design or inverse-probability-of-censoring
            adjustments). Must have the same length as `surv`{.gd-no-link}. Default (`None`): unit
            weights.
        data
            A data frame (pandas, Polars, PyArrow, DuckDB, a lazy frame, ...) holding the columns
            named by the response, `by`, and `weights`. When `surv`{.gd-no-link} is an `Outcome` or
            a formula, rows with a missing value in any column used are dropped before fitting.

        Returns
        -------
        NelsonAalen
            The fitted estimator object itself (for method chaining) with cached results
            (`~~greenwood.NelsonAalen.time_`, `cumulative_hazard_`, `conf_low_`{.gd-no-link},
            `conf_high_`{.gd-no-link}, `n_risk_`, `n_event_` as attributes).

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

        # Load data and name the response columns
        lung = gw.load_dataset("lung", backend="polars")
        death = gw.Outcome.surv(time="time", event="status")

        # Fit a single unstratified cumulative hazard curve
        na = gw.NelsonAalen().fit(death, data=lung)
        na
        ```

        Fit stratified curves by sex to compare cumulative hazard accumulation:

        ```{python}
        # Fit stratified cumulative hazard curves by sex
        na_stratified = gw.NelsonAalen().fit(death, by="sex", data=lung)
        na_stratified
        ```
        """
        bound = bind_fit_inputs(
            surv,
            data=data,
            labels={"by": by, "weights": weights},
            rhs_to="by",
            estimator="NelsonAalen",
        )
        surv = bound.surv
        strata, self._grouped = resolve_strata(bound, "by")
        weights = bound.labels["weights"]
        self._n_input = bound.n_input
        self.n_dropped_ = bound.n_dropped

        z = float(norm.ppf(1.0 - (1.0 - self.conf_level) / 2.0))
        self._blocks = _fit_blocks(surv, strata, weights, "log", z)
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
    def strata_(self) -> Array:
        """The curve label of each row: `"all"` for one curve, `"sex=1"` and so on for groups."""
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

        cols: dict[str, Array] = {"strata": self.strata_}
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

        Exports the Nelson-Aalen estimate with one row per event time, including risk-set counts,
        the cumulative hazard estimate, its standard error, confidence limits, and curve labels.

        Parameters
        ----------
        format
            Output format: `None` (default), `"pandas"`, `"polars"`, or `"pyarrow"`. When `None`, a
            backend is auto-detected (Polars, then Pandas, then PyArrow).

        Returns
        -------
        pandas.DataFrame, polars.DataFrame, or pyarrow.Table
            A tidy table with columns `strata`, `time`, `n_risk`, `n_event`, `estimate`,
            `std_error`, `conf_low`, and `conf_high`. The layout is the same for one curve
            (`strata` is `"all"`) as for several.

        Raises
        ------
        ImportError
            If the requested (or, when auto-detecting, any) DataFrame library is not installed.

        Examples
        --------
        Fit a Nelson-Aalen estimator on the bundled `lung` dataset, then export the fitted
        cumulative-hazard curve as a Polars frame:

        ```{python}
        import greenwood as gw

        # Load data and fit the Nelson-Aalen estimator
        lung = gw.load_dataset("lung", backend="polars")
        death = gw.Outcome.surv(time="time", event="status")
        na = gw.NelsonAalen().fit(death, data=lung)

        # Export the cumulative hazard as a Polars DataFrame
        na.to_frame(format="polars")
        ```

        Pass a different `format=` for pandas or PyArrow output:

        ```{python}
        # Export as a pandas DataFrame instead
        na.to_frame(format="pandas")
        ```
        """
        return to_dataframe(self._table_columns(), format=format)


def _tidy_nelson_aalen(na: NelsonAalen, *, format: str | None = None, **_: Any) -> Any:
    """broom-style `tidy`: one row per time point (`estimate` is cumulative hazard)."""
    return na.to_frame(format=format)


def _glance_nelson_aalen(na: NelsonAalen, *, format: str | None = None, **_: Any) -> Any:
    """broom-style `glance`: one row per stratum with counts and max cumulative hazard."""
    cols: dict[str, list[Any]] = {"strata": [b.label for b in na._blocks]}
    cols["n_start"] = [float(b.n_risk[0]) if b.n_risk.size else float("nan") for b in na._blocks]
    cols["events"] = [float(b.n_event.sum()) for b in na._blocks]
    cols["max_cumhaz"] = [
        float(b.cumhaz[-1]) if b.cumhaz.size else float("nan") for b in na._blocks
    ]
    return to_dataframe(cols, format=format)


def _register_adapters() -> None:
    from .summaries import register_glance, register_tidier

    register_tidier("greenwood._nonparametric.KaplanMeier", _tidy_kaplan_meier)
    register_glance("greenwood._nonparametric.KaplanMeier", _glance_kaplan_meier)
    register_tidier("greenwood._nonparametric.NelsonAalen", _tidy_nelson_aalen)
    register_glance("greenwood._nonparametric.NelsonAalen", _glance_nelson_aalen)


_register_adapters()
