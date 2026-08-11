"""Smooth non-linear hazard ratio curves for continuous covariates.

Plots the estimated log-hazard ratio (or hazard ratio) as a function of a continuous
covariate, with a pointwise confidence band. The curve is computed by refitting the Cox
model with a B-spline basis expansion for the covariate of interest.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from .._cox import SmoothHRResult

__all__ = ["plot_smooth_hr"]

_LINE_COLOR = "#20558A"
_BAND_COLOR = "#20558A"


def plot_smooth_hr(
    result: Any,
    *,
    scale: str = "log_hr",
    title: str | None = None,
    xlab: str | None = None,
    ylab: str | None = None,
    backend: Literal["altair", "plotnine"] = "altair",
    width: int = 500,
    height: int = 300,
) -> Any:
    r"""Plot a smooth hazard ratio curve for a continuous covariate.

    Draws the estimated (log) hazard ratio as a smooth curve across the range of a
    continuous covariate, with a shaded pointwise confidence band. A horizontal reference
    line marks HR = 1 (log-HR = 0). The curve is produced by ``CoxPH.smooth_hr()``.

    Parameters
    ----------
    result
        A ``SmoothHRResult`` from ``CoxPH.smooth_hr()``.
    scale
        ``"log_hr"`` (default) plots the log hazard ratio. ``"hr"`` plots the hazard ratio.
    title
        Optional title for the chart.
    xlab
        Label for the x-axis. Defaults to the covariate name.
    ylab
        Label for the y-axis. Defaults to ``"Log hazard ratio"`` or ``"Hazard ratio"``.
    backend
        Plotting backend: ``"altair"`` (default) or ``"plotnine"``.
    width
        Width in pixels (Altair) or approximate inches (plotnine).
    height
        Height in pixels (Altair) or approximate inches (plotnine).

    Returns
    -------
    alt.Chart or plotnine.ggplot

    Examples
    --------
    ```{python}
    import greenwood as gw

    lung = gw.load_dataset("lung", backend="polars")
    y = gw.Surv.right(lung["time"], event=(lung["status"] == 2))
    cox = gw.CoxPH().fit(y, lung[["age", "sex"]])

    gw.plot_smooth_hr(cox.smooth_hr("age"))
    ```

    Plot on the hazard ratio scale instead:

    ```{python}
    gw.plot_smooth_hr(cox.smooth_hr("age"), scale="hr")
    ```
    """
    x_label: str = result.term if xlab is None else xlab
    y_label: str = (
        ("Hazard ratio" if scale == "hr" else "Log hazard ratio") if ylab is None else ylab
    )

    if backend == "altair":
        return _plot_altair(
            result, scale=scale, title=title, xlab=x_label, ylab=y_label, width=width, height=height
        )
    if backend == "plotnine":
        return _plot_plotnine(
            result,
            scale=scale,
            title=title,
            xlab=x_label,
            ylab=y_label,
            width=width,
            height=height,
        )
    raise ValueError(f"backend must be 'altair' or 'plotnine', got {backend!r}")


def _plot_altair(
    result: SmoothHRResult,
    *,
    scale: str,
    title: str | None,
    xlab: str,
    ylab: str,
    width: int,
    height: int,
) -> Any:
    try:
        import altair as alt
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "plot_smooth_hr() with backend='altair' requires altair. "
            "Install with `pip install greenwood[altair]`."
        ) from exc

    from .._backends import to_dataframe

    if scale == "hr":
        data = {
            result.term: result.grid.tolist(),
            "y": result.hr.tolist(),
            "lower": result.hr_lower.tolist(),
            "upper": result.hr_upper.tolist(),
        }
        ref_y = 1.0
    else:
        data = {
            result.term: result.grid.tolist(),
            "y": result.log_hr.tolist(),
            "lower": result.log_hr_lower.tolist(),
            "upper": result.log_hr_upper.tolist(),
        }
        ref_y = 0.0

    frame = to_dataframe(data)
    x_enc = alt.X(f"{result.term}:Q", title=xlab)
    y_enc = alt.Y("y:Q", title=ylab)

    band = (
        alt.Chart(frame)
        .mark_area(opacity=0.18, color=_BAND_COLOR)
        .encode(
            x=x_enc,
            y=alt.Y("lower:Q", title=ylab),
            y2="upper:Q",
        )
    )

    line = (
        alt.Chart(frame)
        .mark_line(color=_LINE_COLOR, strokeWidth=2)
        .encode(
            x=x_enc,
            y=y_enc,
        )
    )

    ref = (
        alt.Chart(to_dataframe({"y": [ref_y]}))
        .mark_rule(strokeDash=[4, 4], color="#888888", strokeWidth=1)
        .encode(y="y:Q")
    )

    chart = (ref + band + line).properties(width=width, height=height)
    if title:
        chart = chart.properties(title=title)
    return chart


def _plot_plotnine(
    result: SmoothHRResult,
    *,
    scale: str,
    title: str | None,
    xlab: str,
    ylab: str,
    width: int,
    height: int,
) -> Any:
    try:
        import plotnine as p9
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "plot_smooth_hr() with backend='plotnine' requires plotnine. "
            "Install with `pip install greenwood[plotnine]`."
        ) from exc

    import pandas as pd

    if scale == "hr":
        y_vals = result.hr
        lower_vals = result.hr_lower
        upper_vals = result.hr_upper
        ref_y = 1.0
    else:
        y_vals = result.log_hr
        lower_vals = result.log_hr_lower
        upper_vals = result.log_hr_upper
        ref_y = 0.0

    df = pd.DataFrame(
        {
            "x": result.grid,
            "y": y_vals,
            "lower": lower_vals,
            "upper": upper_vals,
        }
    )

    p = (
        p9.ggplot(df, p9.aes("x", "y"))
        + p9.geom_hline(yintercept=ref_y, linetype="dashed", color="#888888", size=0.5)
        + p9.geom_ribbon(p9.aes(ymin="lower", ymax="upper"), alpha=0.18, fill=_BAND_COLOR)
        + p9.geom_line(color=_LINE_COLOR, size=1.2)
        + p9.labs(x=xlab, y=ylab)
        + p9.theme_minimal()
    )

    if title:
        p = p + p9.ggtitle(title)

    return p
