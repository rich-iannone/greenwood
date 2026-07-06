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

    Examples
    --------
    An `EventTable` is produced by `event_table`. Build one from the bundled `lung`
    dataset and view it as a frame with `to_pandas`. The table built here is reused by
    the export-method examples below.

    ```{python}
    import greenwood as gw

    lung = gw.load_dataset("lung")
    y = gw.Surv.right(lung["time"], event=(lung["status"] == 2))
    et = gw.event_table(y)
    et.to_pandas()
    ```
    """

    time: Array
    n_risk: Array
    n_event: Array
    n_censor: Array
    strata: Array | None = None

    def __len__(self) -> int:
        return int(self.time.shape[0])

    def _table_columns(self) -> dict[str, Array]:
        cols: dict[str, Array] = {}
        if self.strata is not None:
            cols["strata"] = self.strata
        cols["time"] = self.time
        cols["n_risk"] = self.n_risk
        cols["n_event"] = self.n_event
        cols["n_censor"] = self.n_censor
        return cols

    def to_pandas(self) -> Any:
        """Return the tabulation as a pandas DataFrame.

        This method exports the event-table rows to pandas with one row per unique exit
        time and columns for the risk set, events, censorings, and optional strata labels.

        Returns
        -------
        pandas.DataFrame
            A tidy DataFrame containing `time`, `n_risk`, `n_event`, `n_censor`, and
            optionally `strata`.

        Raises
        ------
        ImportError
            If pandas is not installed.

        Examples
        --------
        Convert the event table to pandas for inspection or downstream analysis:

        ```{python}
        et.to_pandas()
        ```
        """
        try:
            import pandas as pd
        except ImportError as e:
            raise ImportError(
                "pandas is required for to_pandas(). Install it with: pip install pandas"
            ) from e

        return pd.DataFrame(self._table_columns())

    def to_polars(self) -> Any:
        """Return the tabulation as a Polars DataFrame.

        This method exports the event-table rows to Polars with one row per unique exit
        time and columns for the risk set, events, censorings, and optional strata labels.

        Returns
        -------
        polars.DataFrame
            A tidy DataFrame containing `time`, `n_risk`, `n_event`, `n_censor`, and
            optionally `strata`.

        Raises
        ------
        ImportError
            If polars is not installed.

        Examples
        --------
        Convert the event table to Polars for fast columnar processing:

        ```{python}
        et.to_polars()
        ```
        """
        try:
            import polars as pl
        except ImportError as e:
            raise ImportError(
                "polars is required for to_polars(). Install it with: pip install polars"
            ) from e

        return pl.DataFrame(self._table_columns())

    def to_arrow(self) -> Any:
        """Return the tabulation as a PyArrow Table.

        This method exports the event-table rows to an Arrow table, preserving the same
        columns used in the pandas and Polars exports for efficient interoperability.

        Returns
        -------
        pyarrow.Table
            A table containing `time`, `n_risk`, `n_event`, `n_censor`, and optionally
            `strata`.

        Raises
        ------
        ImportError
            If pyarrow is not installed.

        Examples
        --------
        Convert the event table to Arrow for interchange with Arrow-based tools:

        ```{python}
        et.to_arrow()
        ```
        """
        try:
            import pyarrow as pa
        except ImportError as e:
            raise ImportError(
                "pyarrow is required for to_arrow(). Install it with: pip install pyarrow"
            ) from e

        return pa.table(self._table_columns())


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
        Optional group labels (any Narwhals series / array / sequence, length `n`). When
        given, the table is stratified and carries a `strata` column.
    weights
        Optional case weights. Defaults to the response's weights, or 1.

    Returns
    -------
    EventTable
        The per-time tabulation, ascending in time within each stratum.

    Examples
    --------
    Tabulate the risk set from the bundled `lung` dataset. Each row is a unique exit
    `time` with the number still at risk (`n_risk`), the number of events (`n_event`), and
    the number censored (`n_censor`) at that time.

    ```{python}
    import greenwood as gw

    lung = gw.load_dataset("lung")
    y = gw.Surv.right(lung["time"], event=(lung["status"] == 2))
    gw.event_table(y).to_pandas()
    ```

    Passing `group=` stratifies the table, adding a `strata` column with one block of rows
    per group.

    ```{python}
    import greenwood as gw

    gw.event_table(y, group=lung["sex"]).to_pandas()
    ```
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
