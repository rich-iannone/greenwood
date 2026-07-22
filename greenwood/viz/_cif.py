"""Cumulative Incidence Function (CIF) plots for competing risks."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from .._backends import to_dataframe

if TYPE_CHECKING:
    from .._competing import AalenJohansen

__all__ = ["plot_cif"]

# Vega-Lite tableau10 palette — kept in sync with _altair.py.
_PALETTE = (
    "#4c78a8",
    "#f58518",
    "#e45756",
    "#72b7b2",
    "#54a24b",
    "#eeca3b",
    "#b279a2",
    "#ff9da6",
    "#9d755d",
    "#bab0ac",
)
_SOLID = "#20558A"
_VALID_CI = "isValid(datum.conf_low) && isValid(datum.conf_high)"


def _require_altair() -> Any:
    try:
        import altair as alt  # pyright: ignore[reportMissingImports]
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "Altair visualization requires altair. Install it with `pip install greenwood[altair]`."
        ) from exc
    return alt


def _max_time_aj(aj: AalenJohansen) -> float:
    """Largest event time across all blocks."""
    first_cause = aj._causes[0]
    return max(
        (
            float(block[first_cause]["time"][-1])
            for block in aj._blocks.values()
            if len(block[first_cause]["time"])
        ),
        default=0.0,
    )


def _default_times_aj(aj: AalenJohansen) -> list[float]:
    """Six evenly spaced, rounded times from 0 to the largest observed time."""
    max_t = _max_time_aj(aj)
    raw = np.linspace(0.0, max_t, 6)
    return sorted({round(float(t)) for t in raw})


def _n_at_risk_aj(aj: AalenJohansen, query: np.ndarray) -> dict[str, np.ndarray]:
    """At-risk counts per group at each query time.

    Because all competing causes share one at-risk set, we use the first cause's `n_risk` array.
    Logic mirrors `_shared._n_at_risk` for KaplanMeier.
    """
    first_cause = aj._causes[0]
    out: dict[str, np.ndarray] = {}
    for label, block in aj._blocks.items():
        group_name = str(label) if label is not None else "Overall"
        event_times = block[first_cause]["time"]
        n_risk = block[first_cause]["n_risk"]
        idx = np.searchsorted(event_times, query, side="left")
        counts = np.zeros(len(query))
        valid = idx < len(event_times)
        counts[valid] = n_risk[idx[valid]]
        out[group_name] = counts
    return out


def _risk_table_chart_cif(
    alt: Any,
    aj: AalenJohansen,
    times: Any,
    xlab: str,
    x_domain: list[float],
    total_width: int,
) -> Any:
    """Numbers-at-risk text chart to place below the CIF facet."""
    query = np.asarray(_default_times_aj(aj) if times is None else times, dtype=float)
    counts = _n_at_risk_aj(aj, query)

    labels = list(counts.keys())
    grouped = aj._grouped
    palette = list(_PALETTE[: len(labels)]) if grouped else [_SOLID]
    color_scale = alt.Scale(domain=labels, range=palette)

    group_col: list[str] = []
    time_col: list[float] = []
    label_col: list[str] = []
    for group_name, cnt_arr in counts.items():
        for t, n in zip(query, cnt_arr, strict=True):
            group_col.append(group_name)
            time_col.append(float(t))
            label_col.append(f"{int(n):g}")

    frame = to_dataframe({"group": group_col, "time": time_col, "label": label_col})

    return (
        alt.Chart(frame)
        .mark_text(size=11)
        .encode(
            x=alt.X("time:Q", title=xlab, scale=alt.Scale(domain=x_domain, nice=False)),
            y=alt.Y("group:N", title=None),
            text="label:N",
            color=alt.Color("group:N", scale=color_scale, legend=None),
        )
        .properties(width=total_width, height=20 * max(1, len(labels)))
    )


def _step_data(aj: AalenJohansen) -> dict[str, list[Any]]:
    """Build tidy step-function columns from a fitted AalenJohansen.

    Each (group, cause) pair is prepended with a t=0, CIF=0 anchor so that Vega-Lite's step-after
    interpolation renders a correct right-continuous step.
    """
    time_col: list[float] = []
    estimate_col: list[float] = []
    conf_low_col: list[float] = []
    conf_high_col: list[float] = []
    cause_col: list[str] = []
    group_col: list[str] = []

    for label, block in aj._blocks.items():
        group_name = str(label) if label is not None else "Overall"
        for cause_int in aj._causes:
            cause_name = str(aj.states_[cause_int - 1])
            data = block[cause_int]
            times = data["time"]
            estimates = data["estimate"]
            conf_lows = data["conf_low"]
            conf_highs = data["conf_high"]

            # t=0 anchor: CIF starts at zero.
            time_col.append(0.0)
            estimate_col.append(0.0)
            conf_low_col.append(0.0)
            conf_high_col.append(0.0)
            cause_col.append(cause_name)
            group_col.append(group_name)

            for i in range(len(times)):
                time_col.append(float(times[i]))
                estimate_col.append(float(estimates[i]))
                conf_low_col.append(float(conf_lows[i]))
                conf_high_col.append(float(conf_highs[i]))
                cause_col.append(cause_name)
                group_col.append(group_name)

    return {
        "time": time_col,
        "estimate": estimate_col,
        "conf_low": conf_low_col,
        "conf_high": conf_high_col,
        "cause": cause_col,
        "group": group_col,
    }


def plot_cif(
    aj: AalenJohansen,
    *,
    conf_int: bool = True,
    risk_table: bool = False,
    times: Any = None,
    title: str | None = None,
    xlab: str = "Time",
    ylab: str = "Cumulative incidence",
    width: int = 400,
    height: int = 300,
    backend: str = "altair",
) -> Any:
    r"""Plot cumulative incidence functions from a fitted Aalen-Johansen estimator.

    Renders one cumulative incidence curve per competing cause as a right-continuous step function.
    The chart is faceted by cause (one panel per event type). Stratified fits (produced by passing
    `by=` to `AalenJohansen.fit()`) draw one colored line per group within each panel. An optional
    shaded confidence band shows the point-wise uncertainty.

    Parameters
    ----------
    aj
        A fitted `AalenJohansen` object. Unstratified fits draw a single curve per panel. Stratified
        fits draw one colored curve per group.
    conf_int
        If `True` (default), draw a shaded point-wise confidence band around each curve.
    risk_table
        If `True`, stack an aligned numbers-at-risk table beneath the curves.
    times
        Query times for the numbers-at-risk table (used only when `risk_table=True`). Defaults to
        six evenly spaced times from `0` to the largest observed follow-up time.
    title
        Optional overall plot title.
    xlab
        X-axis label (default `"Time"`).
    ylab
        Y-axis label (default `"Cumulative incidence"`).
    width
        Width of each cause panel in pixels (default `400`).
    height
        Height of each cause panel in pixels (default `300`).
    backend
        Plotting backend. Currently only `"altair"` is supported.

    Returns
    -------
    altair.Chart
        An interactive Altair chart. Each panel corresponds to one competing cause. Groups are
        distinguished by color.

    Examples
    --------
    Fit the Aalen-Johansen estimator on the bundled mgus2 dataset, where patients may progress to
    malignancy (PCM) or die first:

    ```{python}
    import numpy as np
    import greenwood as gw

    # Load data and build a competing-risks response
    mg = gw.load_dataset("mgus2", backend="polars")
    etime = np.where(mg["pstat"] == 1, mg["ptime"], mg["futime"])
    cause = np.where(mg["pstat"] == 1, 1, 2 * mg["death"])
    y = gw.Surv.multistate(etime, event=cause, states=("pcm", "death"))

    # Fit the Aalen-Johansen estimator and plot cumulative incidence
    aj = gw.AalenJohansen().fit(y)
    gw.plot_cif(aj)
    ```

    Pass `by=` to stratify by a covariate and compare groups across causes:

    ```{python}
    # Stratify by sex and compare groups across causes
    aj_sex = gw.AalenJohansen().fit(y, by=mg["sex"])
    gw.plot_cif(aj_sex, title="Cumulative incidence by sex")
    ```

    Add an aligned numbers-at-risk table beneath the curves:

    ```{python}
    # Add a numbers-at-risk table beneath the curves
    gw.plot_cif(aj_sex, risk_table=True)
    ```
    """
    if backend != "altair":
        raise ValueError(f"backend must be 'altair', got {backend!r}")

    alt = _require_altair()

    data = _step_data(aj)
    df = to_dataframe(data)

    grouped = aj._grouped
    labels = [str(k) if k is not None else "Overall" for k in aj._blocks]

    if grouped:
        color_scale = alt.Scale(domain=labels, range=list(_PALETTE[: len(labels)]))
        line_color: Any = alt.Color("group:N", title="Group", scale=color_scale)
        fill_color: Any = alt.Color("group:N", title="Group", scale=color_scale)
    else:
        line_color = alt.value(_SOLID)
        fill_color = alt.value(_SOLID)

    # Pin the x domain so the risk table (if requested) can share the same scale.
    x_domain = [0.0, _max_time_aj(aj)]
    x = alt.X("time:Q", title=xlab, scale=alt.Scale(domain=x_domain, nice=False, padding=16))
    y_enc = alt.Y("estimate:Q", title=ylab, scale=alt.Scale(domain=[0.0, 1.0]))

    base = alt.Chart(df)
    layers: list[Any] = []

    if conf_int:
        area = (
            base.transform_filter(_VALID_CI)
            .mark_area(interpolate="step-after", opacity=0.18)
            .encode(
                x=x,
                y=alt.Y("conf_low:Q", title=ylab, scale=alt.Scale(domain=[0.0, 1.0])),
                y2=alt.Y2("conf_high:Q"),
                fill=fill_color,
            )
        )
        layers.append(area)

    tooltip: list[Any] = [
        alt.Tooltip("time:Q", title=xlab),
        alt.Tooltip("estimate:Q", title=ylab, format=".3f"),
        alt.Tooltip("cause:N", title="Cause"),
    ]
    if grouped:
        tooltip.insert(0, alt.Tooltip("group:N", title="Group"))

    line = base.mark_line(interpolate="step-after").encode(
        x=x,
        y=y_enc,
        color=line_color,
        tooltip=tooltip,
    )
    layers.append(line)

    n_causes = len(aj._causes)

    if risk_table:
        # Suppress the curve's own x-axis labels; the table's axis serves both.
        x_no_label = alt.X(
            "time:Q",
            scale=alt.Scale(domain=x_domain, nice=False, padding=16),
            axis=alt.Axis(title=None, labels=False, ticks=False),
        )
        layers_no_ax = []
        if conf_int:
            area_no_ax = (
                base.transform_filter(_VALID_CI)
                .mark_area(interpolate="step-after", opacity=0.18)
                .encode(
                    x=x_no_label,
                    y=alt.Y("conf_low:Q", title=ylab, scale=alt.Scale(domain=[0.0, 1.0])),
                    y2=alt.Y2("conf_high:Q"),
                    fill=fill_color,
                )
            )
            layers_no_ax.append(area_no_ax)
        line_no_ax = base.mark_line(interpolate="step-after").encode(
            x=x_no_label,
            y=y_enc,
            color=line_color,
            tooltip=tooltip,
        )
        layers_no_ax.append(line_no_ax)
        spec = alt.layer(*layers_no_ax).properties(width=width, height=height).interactive()
    else:
        spec = alt.layer(*layers).properties(width=width, height=height).interactive()

    chart = spec.facet(
        column=alt.Column(
            "cause:N",
            title=None,
            header=alt.Header(title=None, labelFontSize=13),
        )
    )

    if title is not None:
        chart = chart.properties(title=title)

    if not risk_table:
        return chart

    # Approximate total width of the faceted chart for risk table alignment.
    total_width = width * n_causes
    table = _risk_table_chart_cif(alt, aj, times, xlab, x_domain, total_width)
    return alt.vconcat(chart, table).resolve_scale(x="shared")
