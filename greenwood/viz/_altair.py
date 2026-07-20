"""Kaplan-Meier survival curves and numbers-at-risk tables, drawn with Altair.

A Narwhals-native alternative to the plotnine backend: everything returns composable Altair
objects, so no Pandas or Matplotlib is pulled in. `plot_survival` gives an `alt.Chart` (an
interactive Vega-Lite spec), and with `risk_table=True` it returns an `alt.VConcatChart`
stacking the curve over an aligned numbers-at-risk table (the x-axes share a scale). Altair
is an optional dependency (the `altair` extra), so it is imported lazily.

The step geometry is expressed with Vega-Lite's native ``interpolate="step-after"`` rather
than the manual point-doubling the plotnine backend uses, so the frames handed to Altair hold
one row per event time. Those frames are built through `to_dataframe`, so a Polars-only user
gets Polars all the way through (Altair consumes it via Narwhals).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .._backends import to_dataframe
from ._shared import _risk_table_columns, _strata_label

if TYPE_CHECKING:
    from .._nonparametric import KaplanMeier

__all__ = ["plot_survival"]

_SOLID = "#20558A"
_VALID_CI = "isValid(datum.conf_low) && isValid(datum.conf_high)"

# Custom symbol path for a censoring notch. Vega anchors a custom symbol's path origin (0,0)
# at the data point, so this draws a `/` whose bottom-left tip rests on the curve and slants up
# and to the right (SVG y grows downward, hence the negative y).
_CENSOR_NOTCH = "M 0,0 L 1,-1"

# Vega-Lite's default "tableau10" categorical range. We pin it explicitly (rather than relying
# on the default) so we know each stratum's exact color and can derive a matching darker tint
# for the censoring notches. Lines keep the same colors they had before.
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


def _darken(hex_color: str, factor: float = 0.62) -> str:
    """A darker tint of a hex color, scaling each RGB channel toward black."""
    h = hex_color.lstrip("#")
    r, g, b = (int(int(h[i : i + 2], 16) * factor) for i in (0, 2, 4))
    return f"#{r:02x}{g:02x}{b:02x}"


def _require_altair() -> Any:
    try:
        import altair as alt  # pyright: ignore[reportMissingImports]
    except ImportError as exc:  # pragma: no cover (exercised only without the extra)
        raise ImportError(
            "Altair visualization requires altair. Install it with `pip install greenwood[altair]`."
        ) from exc
    return alt


def _step_columns(km: KaplanMeier) -> dict[str, list[Any]]:
    """One row per event time (plus a leading point at t=0), for all blocks combined.

    Vega-Lite's ``step-after`` interpolation draws the right-continuous step, so unlike the
    plotnine backend we do not double the coordinates by hand.
    """
    time: list[float] = []
    estimate: list[float] = []
    conf_low: list[float] = []
    conf_high: list[float] = []
    strata: list[str] = []
    for block in km._blocks:
        label = _strata_label(block)
        time.append(0.0)
        estimate.append(1.0)
        conf_low.append(1.0)
        conf_high.append(1.0)
        strata.append(label)
        for i in range(block.time.shape[0]):
            time.append(float(block.time[i]))
            estimate.append(float(block.surv[i]))
            conf_low.append(float(block.conf_low[i]))
            conf_high.append(float(block.conf_high[i]))
            strata.append(label)
    return {
        "time": time,
        "estimate": estimate,
        "conf_low": conf_low,
        "conf_high": conf_high,
        "strata": strata,
    }


def _censor_columns(km: KaplanMeier) -> dict[str, list[Any]]:
    time: list[float] = []
    estimate: list[float] = []
    strata: list[str] = []
    for block in km._blocks:
        mask = block.n_censor > 0
        label = _strata_label(block)
        for t, s in zip(block.time[mask], block.surv[mask], strict=True):
            time.append(float(t))
            estimate.append(float(s))
            strata.append(label)
    return {"time": time, "estimate": estimate, "strata": strata}


def _max_time(km: KaplanMeier) -> float:
    return max((float(b.time[-1]) for b in km._blocks if b.time.size), default=0.0)


def _x_scale(alt: Any, km: KaplanMeier) -> Any:
    """A shared x scale for the curve and risk table.

    `padding` reserves pixels at both ends of the range so risk-table counts centered on the
    first and last times are not clipped by the plot edge. The curve and table use the same
    scale and width, so their axes line up pixel-for-pixel.
    """
    return alt.Scale(domain=[0.0, _max_time(km)], nice=False, padding=16)


def plot_survival(
    km: KaplanMeier,
    *,
    conf_int: bool = True,
    censor_marks: bool = True,
    risk_table: bool = False,
    times: Any = None,
    xlab: str = "Time",
    ylab: str = "Survival probability",
    width: int = 500,
    height: int = 300,
    backend: str = "altair",
) -> Any:
    """Plot Kaplan-Meier survival curve(s).

    Renders one or more Kaplan-Meier survival curves as a publication-ready visualization.
    By default uses interactive Altair (Vega-Lite) charts with optional plotnine (ggplot2)
    support. Each curve shows the proportion of subjects surviving (event-free) over time as a
    right-continuous step function, with an optional shaded confidence band and censoring
    marks. Stratified fits produce one colored curve per group with a legend.

    Parameters
    ----------
    km
        A fitted `KaplanMeier` object, unstratified (single curve) or stratified.
    conf_int
        If `True` (default), draw the point-wise confidence band.
    censor_marks
        If `True` (default), mark censoring times with `+` symbols on the curve.
    risk_table
        If `True`, return a visualization stacking the curve over an aligned
        numbers-at-risk table. If `False` (default), return only the curve.
    times
        Query times for the numbers-at-risk table (used only if `risk_table=True`). Defaults
        to six evenly spaced, rounded times from 0 to the maximum observed follow-up time.
    xlab, ylab
        Axis labels (defaults `"Time"` and `"Survival probability"`).
    width, height
        Plot dimensions (in pixels for Altair, inches for plotnine; defaults 500x300 pixels).
    backend
        Plotting backend: `"altair"` (default, interactive Vega-Lite) or `"plotnine"`
        (ggplot2-style). Requires the corresponding extra: `pip install greenwood[altair]`
        or `pip install greenwood[plotnine]`.

    Returns
    -------
    An Altair `alt.LayerChart` or `alt.VConcatChart` (if `backend="altair"`), or a plotnine
    `ggplot` object (if `backend="plotnine"`).

    Examples
    --------
    ```{python}
    import greenwood as gw

    lung = gw.load_dataset("lung", backend="polars")
    y = gw.Surv.right(lung["time"], event=(lung["status"] == 2))
    km = gw.KaplanMeier().fit(y, by=lung["sex"])

    # Interactive Altair (default)
    gw.plot_survival(km, risk_table=True)

    # ggplot2-style (if plotnine is installed)
    gw.plot_survival(km, backend="plotnine", risk_table=True)
    ```
    """
    if backend == "altair":
        return _plot_survival_altair(
            km,
            conf_int=conf_int,
            censor_marks=censor_marks,
            risk_table=risk_table,
            times=times,
            xlab=xlab,
            ylab=ylab,
            width=width,
            height=height,
        )
    elif backend == "plotnine":
        # Import from _curves module
        from . import _curves

        return _curves.plot_survival(
            km,
            conf_int=conf_int,
            censor_marks=censor_marks,
            risk_table=risk_table,
            times=times,
            xlab=xlab,
            ylab=ylab,
        )
    else:
        raise ValueError(f"backend must be 'altair' or 'plotnine', got {backend!r}")


def _plot_survival_altair(
    km: KaplanMeier,
    *,
    conf_int: bool = True,
    censor_marks: bool = True,
    risk_table: bool = False,
    times: Any = None,
    xlab: str = "Time",
    ylab: str = "Survival probability",
    width: int = 500,
    height: int = 300,
) -> Any:
    """Internal Altair implementation of plot_survival."""
    alt = _require_altair()

    steps = to_dataframe(_step_columns(km))
    grouped = km._grouped

    # Pin the palette so lines and their (darker) censoring notches map through known colors.
    labels = [_strata_label(b) for b in km._blocks]
    if grouped:
        color_scale = alt.Scale(domain=labels, range=list(_PALETTE[: len(labels)]))
        dark_scale = alt.Scale(domain=labels, range=[_darken(c) for c in _PALETTE[: len(labels)]])
        stroke_color = alt.Color("strata:N", scale=dark_scale, legend=None)
    else:
        color_scale = alt.Scale(domain=labels, range=[_SOLID])
        stroke_color = alt.value(_darken(_SOLID))

    def _line_color() -> Any:
        if grouped:
            return alt.Color("strata:N", title="Group", scale=color_scale)
        return alt.value(_SOLID)

    # With a risk table below, the table's axis carries the Time scale for both, so the
    # curve's own x-axis (title and tick labels) would only duplicate it; keep the gridlines.
    if risk_table:
        x = alt.X(
            "time:Q", scale=_x_scale(alt, km), axis=alt.Axis(title=None, labels=False, ticks=False)
        )
    else:
        x = alt.X("time:Q", title=xlab, scale=_x_scale(alt, km))
    y = alt.Y("estimate:Q", title=ylab, scale=alt.Scale(domain=[0.0, 1.0]))

    base = alt.Chart(steps)
    layers: list[Any] = []

    if conf_int:
        area = (
            base.transform_filter(_VALID_CI)
            .mark_area(interpolate="step-after", opacity=0.18)
            .encode(
                x=x,
                y=alt.Y("conf_low:Q", title=ylab, scale=alt.Scale(domain=[0.0, 1.0])),
                y2=alt.Y2("conf_high:Q"),
                fill=_line_color(),
            )
        )
        layers.append(area)

    line_enc: dict[str, Any] = {
        "x": x,
        "y": y,
        "tooltip": [
            alt.Tooltip("strata:N", title="Group"),
            alt.Tooltip("time:Q", title=xlab),
            alt.Tooltip("estimate:Q", title=ylab, format=".3f"),
        ],
    }
    line_enc["color"] = _line_color()
    line = base.mark_line(interpolate="step-after").encode(**line_enc)
    layers.append(line)

    if censor_marks:
        censors = _censor_columns(km)
        if censors["time"]:
            # Censoring marks are thin `/` notches in a darker tint of each curve's color:
            # distinctive, and readable where many cluster together. The custom symbol path
            # anchors its origin (0,0) (the notch's bottom-left tip) at the data point, so the
            # mark rests *on* the curve and slants up and to the right, away from the
            # descending trace, rather than being centered across it.
            marks = (
                alt.Chart(to_dataframe(censors))
                .mark_point(shape=_CENSOR_NOTCH, size=140, strokeWidth=2, opacity=0.9)
                .encode(
                    x=x,
                    y=alt.Y("estimate:Q", scale=alt.Scale(domain=[0.0, 1.0])),
                    stroke=stroke_color,
                )
            )
            layers.append(marks)

    if not risk_table:
        return alt.layer(*layers).properties(width=width, height=height).interactive()

    # An x-only zoom on the curve, driving a *shared* x scale, keeps the curve and the table
    # locked together when the user pans or zooms: both views read the one x scale the zoom
    # rewrites. (y is fixed to [0, 1], so there is nothing to zoom vertically.)
    zoom = alt.selection_interval(bind="scales", encodings=["x"])
    curve = alt.layer(*layers).properties(width=width, height=height).add_params(zoom)
    table = _risk_table_chart(km, times=times, xlab=xlab, width=width)
    return alt.vconcat(curve, table).resolve_scale(x="shared")


def _risk_table_chart(
    km: KaplanMeier, *, times: Any = None, xlab: str = "Time", width: int = 500
) -> Any:
    alt = _require_altair()
    cols = _risk_table_columns(km, times)
    # Whole counts as integers (90, not 90.0); weighted counts pass through unchanged.
    frame = to_dataframe(
        {
            "strata": cols["strata"],
            "time": cols["time"],
            "label": [f"{v:g}" for v in cols["n_risk"]],
        }
    )

    # Match the counts' colors to the curve's lines: the same pinned palette (or the solid
    # color when there is a single, unstratified curve).
    labels = [_strata_label(b) for b in km._blocks]
    palette = list(_PALETTE[: len(labels)]) if km._grouped else [_SOLID]
    color_scale = alt.Scale(domain=labels, range=palette)

    return (
        alt.Chart(frame)
        .mark_text(size=11)
        .encode(
            x=alt.X("time:Q", title=xlab, scale=_x_scale(alt, km)),
            y=alt.Y("strata:N", title=None),
            text="label:N",
            color=alt.Color("strata:N", scale=color_scale, legend=None),
        )
        .properties(width=width, height=20 * max(1, len(km._blocks)))
    )
