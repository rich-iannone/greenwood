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
    """Tabulate the event history: risk sets and events at each observed time.

    Creates a structured summary of the survival data at each unique event time. The table
    shows how many subjects were at risk, how many experienced events, and how many were
    censored at each time point. This is the foundational data structure used by
    non-parametric estimators (Kaplan-Meier, Nelson-Aalen) and the log-rank test.

    **Uses**:

    - **Verification**: Inspect risk sets to understand data structure and check for
      censoring patterns.
    - **Manual calculations**: Compute survival estimates, cumulative event rates, or
      other summaries directly from the risk-set counts.
    - **Understanding censoring**: See censoring patterns by time and stratification.
    - **Reporting**: Present summary tables in publications (common in clinical trials).

    Parameters
    ----------
    surv
        A `Surv` response (time-to-event data). Supports right-censored or counting-process
        format. Weighted responses are supported; weights are incorporated into risk-set
        counts.
    group
        Optional grouping variable for stratification, one value per subject. Can be a
        Pandas/Polars series, 1-D array, or Python sequence. When provided, the table is
        split into blocks with a `strata` column, one group per block. Groups appear in
        order of first appearance in the data.
    weights
        Optional case weights. Can be a 1-D array or series. If `None` (default), uses
        weights from the `surv` response if present, otherwise treats all subjects as
        weight 1.

    Returns
    -------
    EventTable
        A structured result with the following attributes:

        - `time`: Unique event times (ascending, per stratum if grouped).
        - `n_risk`: Weighted number of subjects at risk (alive/uncensored and under follow-up)
          at each time.
        - `n_event`: Weighted number of events at each time.
        - `n_censor`: Weighted number of censored subjects at each time.
        - `strata` (if grouped): Stratum label for each row.

        Access columns via `.to_pandas()`, `.to_polars()`, `.to_arrow()`, or iterate directly.

    Notes
    -----
    **Risk-set definition**: At time t, subjects "at risk" are those with:

    - Entry time ≤ t (for counting-process data)
    - Exit time > t (not yet having an event or censoring)

    For right-censored data, entry is always 0, so the condition simplifies.

    **Censoring and events at the same time**: Subjects censored at time t are handled
    carefully. A subject censored at exactly time t is at risk for any event at t (following
    convention in survival analysis). The table counts them in `n_risk` at time t, but
    removes them from `n_risk` at times > t.

    **Stratification order**: If grouped, strata appear in the order of first appearance in
    the input data, not alphabetically. This allows meaningful orderings (e.g., control,
    then treatment).

    **Weights**: If weights are provided, all counts (n_risk, n_event, n_censor) are sums
    of weights, not subject counts. This handles case weights or frequency weights.

    Examples
    --------
    View the basic event table for the `lung` dataset: how many subjects are at risk,
    events, and censoring at each time.

    ```{python}
    import greenwood as gw

    lung = gw.load_dataset("lung")
    y = gw.Surv.right(lung["time"], event=(lung["status"] == 2))
    et = gw.event_table(y)
    et.to_pandas().head(10)
    ```

    The first row shows the first event time: how many subjects were at risk, how many
    experienced an event, and how many were censored. Note that `n_risk` decreases over
    time as subjects leave the risk set (events or censoring).

    Stratify by sex to see event patterns for each group separately:

    ```{python}
    et_sex = gw.event_table(y, group=lung["sex"])
    et_sex.to_pandas().head(15)
    ```

    Now each unique time appears twice (once per stratum) with a `strata` column indicating
    the group. This is useful for inspecting whether event rates and censoring patterns
    differ by group.

    Compute a manual survival estimate from risk-set counts. The survival probability at
    time t is the product of (1 - n_event / n_risk) over all times ≤ t:

    ```{python}
    import numpy as np
    et = gw.event_table(y)
    df = et.to_pandas()
    # Kaplan-Meier survival at each time
    df["surv"] = np.cumprod(1 - df["n_event"] / df["n_risk"])
    df[["time", "n_risk", "n_event", "surv"]].head(10)
    ```

    This manual calculation matches the Kaplan-Meier estimate from `KaplanMeier().fit()`.
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
