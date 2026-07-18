"""Cumulative Incidence Function (CIF) plots for competing risks.

CIF plots display the probability of each competing event over time, allowing visual
comparison of cumulative incidence between groups or for different event types.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from .._backends import to_dataframe

if TYPE_CHECKING:
    pass

__all__ = ["cif_plot"]


def _cif_plot_data(
    time: Any,
    cif: Any,
    event_names: list[str] | None = None,
    group: Any | None = None,
    group_names: dict[Any, str] | None = None,
) -> dict[str, Any]:
    r"""Prepare cumulative incidence function (CIF) data for visualization.

    Formats time-indexed CIF estimates into a tidy structure for plotting. Supports
    stratification by group and multiple competing events.

    Parameters
    ----------
    time
        Event times (1-D array), sorted ascending.
    cif
        Cumulative incidence estimates. Either:

        - A 2-D array of shape (n_times, n_events) for a single group
        - A dict with group keys, each containing a (n_times, n_events) array

    event_names
        Names for each competing event. If `None`, labeled "Event 1", "Event 2", etc.
    group
        Optional group membership (1-D array, same length as number of observations).
        If provided, CIF should be a dict keyed by group.
    group_names
        Dict mapping group values to display names. If `None`, group values used as is.

    Returns
    -------
    dict
        Dictionary with key `"data"` containing a list of dicts, each with:

        - `"time"`: Event time
        - `"cif"`: Cumulative incidence value
        - `"event"`: Event name
        - `"group"`: Group name (if stratified)

        Plus metadata keys `"times"`, `"events"`, `"groups"`.

    Examples
    --------
    Single-group CIF (two competing events):

    ```python
    import greenwood as gw
    import numpy as np

    # CIF over time for two events
    times = np.array([30, 60, 90, 120])
    cif = np.array([
        [0.05, 0.02],
        [0.12, 0.05],
        [0.20, 0.10],
        [0.28, 0.16],
    ])

    cif_data = gw.viz._cif_plot_data(
        time=times,
        cif=cif,
        event_names=["Relapse", "Death"],
    )
    ```

    Stratified by group:

    ```python
    # CIF for two groups and two events
    cif_by_group = {
        "Control": np.array([
            [0.05, 0.02],
            [0.12, 0.05],
            [0.20, 0.10],
            [0.28, 0.16],
        ]),
        "Treatment": np.array([
            [0.02, 0.01],
            [0.06, 0.03],
            [0.12, 0.07],
            [0.18, 0.12],
        ]),
    }

    cif_data = gw.viz._cif_plot_data(
        time=times,
        cif=cif_by_group,
        event_names=["Relapse", "Death"],
    )
    ```
    """
    time_array = np.asarray(time, dtype=float)
    if time_array.ndim != 1:
        raise ValueError("time must be 1-D.")

    n_times = len(time_array)

    # Check CIF structure
    cif_array: np.ndarray | None = None
    if isinstance(cif, dict):
        # Stratified by group
        if group is not None:
            raise ValueError("If cif is a dict, group parameter should not be provided.")
        groups = list(cif.keys())
        group_data_map = cif
        # Extract shape from first group for n_events calculation
        if cif:
            cif_array = list(cif.values())[0]
    else:
        # Single group
        cif_array = np.asarray(cif, dtype=float)
        if cif_array.ndim != 2:
            raise ValueError("cif must be 2-D or a dict of 2-D arrays.")
        if cif_array.shape[0] != n_times:
            raise ValueError(f"cif must have {n_times} rows matching time length.")
        groups = [None]
        group_data_map = {None: cif_array}

    n_events = cif_array.shape[1] if cif_array is not None else 0

    if event_names is None:
        event_names = [f"Event {i + 1}" for i in range(n_events)]
    else:
        if len(event_names) != n_events:
            raise ValueError(
                f"event_names length {len(event_names)} must match number of events {n_events}."
            )

    if group_names is None:
        group_names = {g: str(g) if g is not None else "Overall" for g in groups}

    # Build tidy data
    data_list = []
    for group_key in groups:
        group_cif = group_data_map[group_key]
        group_label = group_names.get(group_key, str(group_key))
        for event_idx, event_name in enumerate(event_names):
            cif_values = group_cif[:, event_idx]
            for t, cif_val in zip(time_array, cif_values, strict=True):
                data_list.append(
                    {
                        "time": float(t),
                        "cif": float(cif_val),
                        "event": event_name,
                        "group": group_label,
                    }
                )

    return {
        "data": data_list,
        "times": list(time_array),
        "events": event_names,
        "groups": [group_names.get(g, str(g)) for g in groups],
    }


def plot_cif(
    time: Any,
    cif: Any,
    event_names: list[str] | None = None,
    group: Any | None = None,
    group_names: dict[Any, str] | None = None,
    title: str | None = None,
    width: int = 400,
    height: int = 300,
    backend: str = "altair",
) -> Any:
    r"""Create an interactive cumulative incidence function (CIF) plot.

    Visualizes cumulative incidence curves for competing events as interactive Altair
    charts with separate panels for each event type.

    Parameters
    ----------
    time
        Event times (1-D array), sorted ascending.
    cif
        Cumulative incidence estimates. Either:

        - A 2-D array of shape (n_times, n_events) for a single group
        - A dict with group keys, each containing a (n_times, n_events) array

    event_names
        Names for each competing event. If `None`, labeled "Event 1", "Event 2", etc.
    group
        Optional group membership. If provided, data is stratified by group.
    group_names
        Dict mapping group values to display names. If `None`, group values used as is.
    title
        Plot title. If `None`, no title.
    width
        Plot width in pixels (default 400).
    height
        Plot height in pixels (default 300).
    backend
        Plotting backend (default `"altair"`). Currently only `"altair"` is supported.

    Returns
    -------
    altair.Chart
        An Altair chart object, interactive and composable.

    Examples
    --------
    Single-group CIF (two competing events):

    ```{python}
    import numpy as np
    import greenwood as gw

    times = np.array([30, 60, 90, 120])
    cif = np.array([
        [0.05, 0.02],
        [0.12, 0.05],
        [0.20, 0.10],
        [0.28, 0.16],
    ])

    gw.cif_plot(
        time=times,
        cif=cif,
        event_names=["Relapse", "Death"],
        title="Cumulative Incidence",
    )
    ```

    Stratified by group:

    ```{python}
    cif_by_group = {
        "Control": np.array([
            [0.05, 0.02],
            [0.12, 0.05],
            [0.20, 0.10],
            [0.28, 0.16],
        ]),
        "Treatment": np.array([
            [0.02, 0.01],
            [0.06, 0.03],
            [0.12, 0.07],
            [0.18, 0.12],
        ]),
    }

    gw.cif_plot(
        time=times,
        cif=cif_by_group,
        event_names=["Relapse", "Death"],
        title="Cumulative Incidence by Group",
    )
    ```
    """
    if backend != "altair":
        raise ValueError(f"backend must be 'altair', got {backend!r}")

    # Prepare data internally
    cif_data = _cif_plot_data(
        time=time,
        cif=cif,
        event_names=event_names,
        group=group,
        group_names=group_names,
    )

    try:
        import altair as alt
    except ImportError as exc:
        raise ImportError("altair required; install with `pip install greenwood[altair]`.") from exc

    data = cif_data["data"]
    # Convert to dict format for to_dataframe (transpose the list of dicts)
    data_dict = {k: [d[k] for d in data] for k in data[0]}
    # to_dataframe returns pandas/polars/pyarrow; Altair accepts all via Narwhals
    df = to_dataframe(data_dict)

    # Base chart for lines
    lines = (
        alt.Chart(df)
        .mark_line()
        .encode(
            x=alt.X("time:Q", title="Time"),
            y=alt.Y("cif:Q", title="Cumulative Incidence", scale=alt.Scale(domain=[0, 1])),
            color=alt.Color("group:N", title="Group"),
            tooltip=["time:Q", "cif:Q", "event:N", "group:N"],
        )
    )

    # Points
    points = (
        alt.Chart(df)
        .mark_point(size=50)
        .encode(
            x="time:Q",
            y="cif:Q",
            color="group:N",
            tooltip=["time:Q", "cif:Q", "event:N", "group:N"],
        )
    )

    # Combine lines and points, set width/height on the spec before faceting
    spec = (lines + points).properties(width=width, height=height)

    # Facet by event, optionally add title
    chart = spec.facet(column="event:N")

    # Add title if provided (use transform on the facet)
    if title is not None:
        chart = chart.properties(title=title)

    return chart
