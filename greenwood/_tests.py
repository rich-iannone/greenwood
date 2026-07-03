"""Group comparison tests: the log-rank statistic and the G-rho (Fleming-Harrington) family.

`logrank_test` compares survival across two or more groups using the weighted log-rank
statistic. The weight is the Fleming-Harrington `S(t-)^rho * (1 - S(t-))^gamma` evaluated
on the pooled Kaplan-Meier estimate, so `rho=0, gamma=0` is the standard log-rank test and
`rho=1, gamma=0` is the Peto-Peto (Wilcoxon-type) test. This matches R's
`survival::survdiff` (which exposes the `rho` parameter).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt
from scipy.stats import chi2

if TYPE_CHECKING:
    from ._surv import Surv

__all__ = ["logrank_test", "TestResult"]

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


def logrank_test(surv: Surv, group: Any, *, rho: float = 0.0, gamma: float = 0.0) -> TestResult:
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

    entry = surv.entry
    exit_ = surv.stop
    event = surv.event
    weight = surv.weights if surv.weights is not None else np.ones(surv.n)

    groups = sorted(set(labels_all.tolist()), key=lambda v: (str(type(v)), v))
    n_groups = len(groups)
    if n_groups < 2:
        raise ValueError("logrank_test needs at least two groups.")

    # Pooled event times (times with at least one event).
    times = np.unique(exit_[event])
    if times.size == 0:
        raise ValueError("No events in the data; the test is undefined.")

    # Per-group at-risk and event counts on the pooled event-time grid.
    n_risk_g = np.empty((n_groups, times.size))
    n_event_g = np.zeros((n_groups, times.size))
    for j, label in enumerate(groups):
        mask = labels_all == label
        n_risk_g[j] = _at_risk(entry[mask], exit_[mask], weight[mask], times)
        ev = mask & event
        idx = np.searchsorted(times, exit_[ev])
        np.add.at(n_event_g[j], idx, weight[ev])

    n_risk = n_risk_g.sum(axis=0)
    n_event = n_event_g.sum(axis=0)

    # Fleming-Harrington weights from the pooled, left-continuous survival.
    with np.errstate(divide="ignore", invalid="ignore"):
        surv_pool = np.cumprod(np.where(n_risk > 0, 1.0 - n_event / n_risk, 1.0))
    surv_left = np.concatenate(([1.0], surv_pool[:-1]))
    w = surv_left**rho * (1.0 - surv_left) ** gamma

    # Observed and expected (weighted) events per group.
    with np.errstate(divide="ignore", invalid="ignore"):
        expected_rate = np.where(n_risk > 0, n_event / n_risk, 0.0)
    observed = (w * n_event_g).sum(axis=1)
    expected = (w * n_risk_g * expected_rate).sum(axis=1)
    u = observed - expected

    # Hypergeometric variance-covariance across groups.
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

    # Drop one group to make V full rank, then chisq = U' V^-1 U.
    u_sub = u[:-1]
    var_sub = var[:-1, :-1]
    statistic = float(u_sub @ np.linalg.solve(var_sub, u_sub))
    df = n_groups - 1
    p_value = float(chi2.sf(statistic, df))

    if rho == 0.0 and gamma == 0.0:
        method = "Log-rank test"
    else:
        method = f"G-rho test (rho={rho}, gamma={gamma})"

    return TestResult(
        statistic=statistic,
        df=df,
        p_value=p_value,
        method=method,
        observed=dict(zip(groups, observed.tolist(), strict=True)),
        expected=dict(zip(groups, expected.tolist(), strict=True)),
    )
