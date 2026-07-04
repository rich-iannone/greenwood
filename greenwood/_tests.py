"""Group comparison tests: the log-rank statistic and the G-rho (Fleming-Harrington) family.

`logrank_test` compares survival across two or more groups using the weighted log-rank
statistic. The weight is the Fleming-Harrington `S(t-)^rho * (1 - S(t-))^gamma` evaluated
on the pooled Kaplan-Meier estimate, so `rho=0, gamma=0` is the standard log-rank test and
`rho=1, gamma=0` is the Peto-Peto (Wilcoxon-type) test. This matches R's
`survival::survdiff` (which exposes the `rho` parameter).
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt
from scipy.stats import chi2

if TYPE_CHECKING:
    from ._surv import Surv

__all__ = ["logrank_test", "pairwise_logrank_test", "TestResult"]

Array = npt.NDArray[Any]


@dataclass(frozen=True)
class TestResult:
    """The outcome of a group comparison test.

    Attributes
    ----------
    statistic
        The chi-square test statistic.
    df
        Degrees of freedom (number of groups minus one).
    p_value
        Upper-tail chi-square p-value.
    method
        Human-readable description of the test and its weights.
    observed, expected
        Weighted observed and expected event counts per group, keyed by group label.
    """

    statistic: float
    df: int
    p_value: float
    method: str
    observed: dict[Any, float] = field(default_factory=dict)
    expected: dict[Any, float] = field(default_factory=dict)

    def __repr__(self) -> str:
        return (
            f"TestResult(method={self.method!r}, statistic={self.statistic:.4f}, "
            f"df={self.df}, p_value={self.p_value:.4g})"
        )


def _at_risk(entry: Array, exit_: Array, weight: Array, times: Array) -> Array:
    """Weighted number at risk at each of `times`: entered before t and not yet exited."""
    e_order = np.argsort(entry, kind="stable")
    e_cumw = np.concatenate(([0.0], np.cumsum(weight[e_order])))
    entered_before = e_cumw[np.searchsorted(entry[e_order], times, side="left")]

    x_order = np.argsort(exit_, kind="stable")
    x_cumw = np.concatenate(([0.0], np.cumsum(weight[x_order])))
    exited_before = x_cumw[np.searchsorted(exit_[x_order], times, side="left")]
    return entered_before - exited_before


def _logrank_uv(
    entry: Array,
    exit_: Array,
    event: Array,
    weight: Array,
    labels: Array,
    groups: list[Any],
    rho: float,
    gamma: float,
) -> tuple[Array, Array, Array]:
    """Observed, expected, and the group variance-covariance for one (sub)sample.

    Uses the pooled, left-continuous Kaplan-Meier for the Fleming-Harrington weight and the
    hypergeometric variance. Returns `(observed, expected, var)` over `groups`; a group with
    no members contributes zeros.
    """
    n_groups = len(groups)
    times = np.unique(exit_[event])
    if times.size == 0:
        z = np.zeros(n_groups)
        return z, z, np.zeros((n_groups, n_groups))

    n_risk_g = np.empty((n_groups, times.size))
    n_event_g = np.zeros((n_groups, times.size))
    for j, label in enumerate(groups):
        mask = labels == label
        n_risk_g[j] = _at_risk(entry[mask], exit_[mask], weight[mask], times)
        ev = mask & event
        idx = np.searchsorted(times, exit_[ev])
        np.add.at(n_event_g[j], idx, weight[ev])

    n_risk = n_risk_g.sum(axis=0)
    n_event = n_event_g.sum(axis=0)

    with np.errstate(divide="ignore", invalid="ignore"):
        surv_pool = np.cumprod(np.where(n_risk > 0, 1.0 - n_event / n_risk, 1.0))
    surv_left = np.concatenate(([1.0], surv_pool[:-1]))
    w = surv_left**rho * (1.0 - surv_left) ** gamma

    with np.errstate(divide="ignore", invalid="ignore"):
        expected_rate = np.where(n_risk > 0, n_event / n_risk, 0.0)
    observed = (w * n_event_g).sum(axis=1)
    expected = (w * n_risk_g * expected_rate).sum(axis=1)

    with np.errstate(divide="ignore", invalid="ignore"):
        common = np.where(
            n_risk > 1,
            w**2 * n_event * (n_risk - n_event) / (n_risk**2 * (n_risk - 1.0)),
            0.0,
        )
    var = np.zeros((n_groups, n_groups))
    for j in range(n_groups):
        for k in range(n_groups):
            if j == k:
                var[j, k] = np.sum(common * n_risk_g[j] * (n_risk - n_risk_g[j]))
            else:
                var[j, k] = -np.sum(common * n_risk_g[j] * n_risk_g[k])
    return observed, expected, var


def _logrank_statistic(
    entry: Array,
    exit_: Array,
    event: Array,
    weight: Array,
    labels: Array,
    groups: list[Any],
    strata: Array | None,
    rho: float,
    gamma: float,
) -> tuple[float, int, Array, Array]:
    """Compute the (optionally stratified) log-rank statistic over `groups`.

    A stratified test sums the observed, expected, and variance contributions across strata
    before forming `chisq = U' V^-1 U` on a dropped-one basis (matching R `survdiff` with a
    `strata()` term).
    """
    n_groups = len(groups)
    observed = np.zeros(n_groups)
    expected = np.zeros(n_groups)
    var = np.zeros((n_groups, n_groups))
    if strata is None:
        observed, expected, var = _logrank_uv(
            entry, exit_, event, weight, labels, groups, rho, gamma
        )
    else:
        for s in np.unique(strata):
            m = strata == s
            o, e, v = _logrank_uv(
                entry[m], exit_[m], event[m], weight[m], labels[m], groups, rho, gamma
            )
            observed += o
            expected += e
            var += v

    u = observed - expected
    statistic = float(u[:-1] @ np.linalg.solve(var[:-1, :-1], u[:-1]))
    return statistic, n_groups - 1, observed, expected


def logrank_test(
    surv: Surv, group: Any, *, rho: float = 0.0, gamma: float = 0.0, strata: Any = None
) -> TestResult:
    """Compare survival across groups with the weighted log-rank (G-rho) test.

    Parameters
    ----------
    surv
        A `Surv` response (right-censored or counting-process).
    group
        Group labels, one per observation (any narwhals series, array, or sequence).
    rho, gamma
        Fleming-Harrington weight exponents on the pooled survival `S(t-)`. Defaults
        `(0, 0)` give the standard log-rank test; `(1, 0)` gives Peto-Peto.
    strata
        Optional stratifying labels. When given, the test is computed within each stratum
        and combined, matching R `survdiff(... + strata(s))`. Use this to control for a
        nuisance variable while comparing `group`.

    Returns
    -------
    TestResult
        Statistic, degrees of freedom, p-value, and per-group observed/expected counts.
    """
    from ._surv import CensoringType, _to_1d_array

    if surv.type not in (CensoringType.RIGHT, CensoringType.COUNTING):
        raise NotImplementedError(
            f"logrank_test supports right-censored and counting-process responses, "
            f"not {surv.type.value!r}."
        )

    labels_all = _to_1d_array(group, dtype=object)
    if labels_all.shape[0] != surv.n:
        raise ValueError("`group` must have the same length as the response.")
    strata_arr = None
    if strata is not None:
        strata_arr = _to_1d_array(strata, dtype=object)
        if strata_arr.shape[0] != surv.n:
            raise ValueError("`strata` must have the same length as the response.")

    entry = surv.entry
    exit_ = surv.stop
    event = surv.event
    weight = surv.weights if surv.weights is not None else np.ones(surv.n)

    groups = sorted(set(labels_all.tolist()), key=lambda v: (str(type(v)), v))
    if len(groups) < 2:
        raise ValueError("logrank_test needs at least two groups.")
    if np.count_nonzero(event) == 0:
        raise ValueError("No events in the data; the test is undefined.")

    statistic, df, observed, expected = _logrank_statistic(
        entry, exit_, event, weight, labels_all, groups, strata_arr, rho, gamma
    )
    p_value = float(chi2.sf(statistic, df))

    if rho == 0.0 and gamma == 0.0:
        method = "Log-rank test"
    else:
        method = f"G-rho test (rho={rho}, gamma={gamma})"
    if strata is not None:
        method = f"Stratified {method[0].lower()}{method[1:]}"

    return TestResult(
        statistic=statistic,
        df=df,
        p_value=p_value,
        method=method,
        observed=dict(zip(groups, observed.tolist(), strict=True)),
        expected=dict(zip(groups, expected.tolist(), strict=True)),
    )


def _p_adjust(pvalues: list[float], method: str) -> list[float]:
    """Adjust p-values for multiple comparisons: `holm`, `bh`, `bonferroni`, or `none`."""
    p = np.asarray(pvalues, dtype=float)
    m = p.size
    if method == "none":
        return p.tolist()
    if method == "bonferroni":
        return np.minimum(p * m, 1.0).tolist()
    order = np.argsort(p)
    out = np.empty(m)
    if method == "holm":
        running = 0.0
        for rank, i in enumerate(order):
            running = max(running, (m - rank) * p[i])
            out[i] = min(running, 1.0)
        return out.tolist()
    if method == "bh":  # Benjamini-Hochberg
        running = 1.0
        for rank in range(m - 1, -1, -1):
            i = order[rank]
            running = min(running, m / (rank + 1) * p[i])
            out[i] = min(running, 1.0)
        return out.tolist()
    raise ValueError(f"Unknown correction {method!r}; use 'holm', 'bh', 'bonferroni', or 'none'.")


def pairwise_logrank_test(
    surv: Surv,
    group: Any,
    *,
    rho: float = 0.0,
    gamma: float = 0.0,
    strata: Any = None,
    correction: str = "holm",
) -> Any:
    """Log-rank test for every pair of groups, with a multiple-comparison correction.

    Runs the (optionally stratified) log-rank test on each pair of `group` levels and adjusts
    the p-values across the pairs. This mirrors R's `pairwise_survdiff`.

    Parameters
    ----------
    surv, group, rho, gamma, strata
        As in `logrank_test`.
    correction
        Multiple-comparison adjustment across the pairwise tests: `"holm"` (default), `"bh"`
        (Benjamini-Hochberg), `"bonferroni"`, or `"none"`.

    Returns
    -------
    A pandas DataFrame with one row per pair: `group1`, `group2`, `statistic`, `p_value`,
    and `p_adjusted`.
    """
    import pandas as pd

    from ._surv import CensoringType, _to_1d_array

    if surv.type not in (CensoringType.RIGHT, CensoringType.COUNTING):
        raise NotImplementedError(
            f"pairwise_logrank_test supports right-censored and counting-process responses, "
            f"not {surv.type.value!r}."
        )
    labels_all = _to_1d_array(group, dtype=object)
    if labels_all.shape[0] != surv.n:
        raise ValueError("`group` must have the same length as the response.")
    strata_arr = None
    if strata is not None:
        strata_arr = _to_1d_array(strata, dtype=object)
        if strata_arr.shape[0] != surv.n:
            raise ValueError("`strata` must have the same length as the response.")

    entry, exit_, event = surv.entry, surv.stop, surv.event
    weight = surv.weights if surv.weights is not None else np.ones(surv.n)
    groups = sorted(set(labels_all.tolist()), key=lambda v: (str(type(v)), v))
    if len(groups) < 2:
        raise ValueError("pairwise_logrank_test needs at least two groups.")

    rows: list[dict[str, Any]] = []
    raw_p: list[float] = []
    for a, b in itertools.combinations(groups, 2):
        m = (labels_all == a) | (labels_all == b)
        stat, dfree, _, _ = _logrank_statistic(
            entry[m],
            exit_[m],
            event[m],
            weight[m],
            labels_all[m],
            [a, b],
            None if strata_arr is None else strata_arr[m],
            rho,
            gamma,
        )
        p = float(chi2.sf(stat, dfree))
        raw_p.append(p)
        rows.append({"group1": a, "group2": b, "statistic": stat, "p_value": p})

    for row, p_adj in zip(rows, _p_adjust(raw_p, correction), strict=True):
        row["p_adjusted"] = p_adj
    return pd.DataFrame(rows)
