"""Core survival primitives: the risk-set and event-table kernel.

This is the shared numeric foundation behind Kaplan-Meier, the log-rank test, and Cox.
Given a `Surv` response (and optionally group labels and case weights), it tabulates, at
each unique exit time, the number at risk, the number of events, and the number censored,
handling left truncation via the counting-process convention.

Risk-set convention (matching R's `survfit`): an observation is at risk at time `t` if it
has entered strictly before `t` and has not yet exited before `t`, i.e. `entry < t <= exit`.
For right-censored data with no truncation, `entry = -inf`, so this reduces to `exit >= t`.
Equivalently, `n_risk(t) = (weight entered before t) - (weight exited before t)`, which is
what we compute with sorted cumulative weights.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    from ._surv import Surv

__all__ = ["EventTable", "event_table"]

Array = npt.NDArray[Any]


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


def event_table(surv: Surv, *, group: Any = None, weights: Any = None) -> EventTable:
    """Tabulate the risk set at each unique exit time.

    Parameters
    ----------
    surv
        A `Surv` response (right-censored or counting-process; interval and multi-state
        endpoints are tabulated later, once their estimators land).
    group
        Optional group labels (any narwhals series / array / sequence, length `n`). When
        given, the table is stratified and carries a `strata` column.
    weights
        Optional case weights. Defaults to the response's weights, or 1.

    Returns
    -------
    EventTable
        The per-time tabulation, ascending in time within each stratum.
    """
    from ._surv import CensoringType, _to_1d_array

    if surv.type not in (CensoringType.RIGHT, CensoringType.COUNTING):
        raise NotImplementedError(
            f"event_table currently supports right-censored and counting-process "
            f"responses, not {surv.type.value!r}."
        )

    entry = surv.entry
    exit_ = surv.stop
    event = surv.event

    if weights is not None:
        weight = _to_1d_array(weights)
    elif surv.weights is not None:
        weight = surv.weights
    else:
        weight = np.ones(surv.n)

    if group is None:
        time, n_risk, n_event, n_censor = _tabulate_block(entry, exit_, event, weight)
        return EventTable(time=time, n_risk=n_risk, n_event=n_event, n_censor=n_censor)

    labels = _to_1d_array(group, dtype=object)
    if labels.shape[0] != surv.n:
        raise ValueError("`group` must have the same length as the response.")

    # Stable, deterministic stratum order: first appearance.
    _, first_idx = np.unique(labels, return_index=True)
    ordered_levels = labels[np.sort(first_idx)]

    times_all: list[Array] = []
    risk_all: list[Array] = []
    event_all: list[Array] = []
    censor_all: list[Array] = []
    strata_all: list[Array] = []
    for level in ordered_levels:
        mask = labels == level
        t, r, e, c = _tabulate_block(entry[mask], exit_[mask], event[mask], weight[mask])
        times_all.append(t)
        risk_all.append(r)
        event_all.append(e)
        censor_all.append(c)
        strata_all.append(np.full(t.shape[0], level, dtype=object))

    return EventTable(
        time=np.concatenate(times_all),
        n_risk=np.concatenate(risk_all),
        n_event=np.concatenate(event_all),
        n_censor=np.concatenate(censor_all),
        strata=np.concatenate(strata_all),
    )
