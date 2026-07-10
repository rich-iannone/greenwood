"""Kaplan-Meier survival curves and numbers-at-risk tables, drawn with plotnine.

Everything returns composable plotnine objects: `plot_survival` gives a `ggplot`, and with
`risk_table=True` it returns a `plotnine.composition` stacking the curve over an aligned
numbers-at-risk table (the x-axes line up). plotnine is an optional dependency (the `plotnine`
extra), so it is imported lazily.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ._shared import _strata_label, get_risk_table_frame

if TYPE_CHECKING:
    from .._nonparametric import KaplanMeier

__all__ = ["plot_survival", "risk_table", "theme_survival"]


def _require_plotnine() -> Any:
    try:
        import plotnine as p9
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise ImportError(
            "Visualization requires plotnine. Install it with `pip install greenwood[plotnine]`."
        ) from exc
    return p9


def _step_frame(block: Any) -> Any:
    """Expand a block into right-continuous step coordinates for line and ribbon.

    Rows with a NaN confidence limit (the point where survival reaches 0) are kept for the
    line but dropped from the ribbon by the caller.
    """
    import pandas as pd

    xs = [0.0]
    est = [1.0]
    low = [1.0]
    high = [1.0]
    prev_e, prev_l, prev_h = 1.0, 1.0, 1.0
    for i in range(block.time.shape[0]):
        t = float(block.time[i])
        xs.extend([t, t])
        est.extend([prev_e, float(block.surv[i])])
        low.extend([prev_l, float(block.conf_low[i])])
        high.extend([prev_h, float(block.conf_high[i])])
        prev_e, prev_l, prev_h = (
            float(block.surv[i]),
            float(block.conf_low[i]),
            float(block.conf_high[i]),
        )
    return pd.DataFrame(
        {
            "time": xs,
            "estimate": est,
            "conf_low": low,
            "conf_high": high,
            "strata": _strata_label(block),
        }
    )


def _censor_frame(block: Any) -> Any:
    import pandas as pd

    mask = block.n_censor > 0
    return pd.DataFrame(
        {
            "time": block.time[mask],
            "estimate": block.surv[mask],
            "strata": _strata_label(block),
        }
    )


def theme_survival() -> Any:
    """A light publication theme for survival plots."""
    p9 = _require_plotnine()
    return p9.theme_minimal() + p9.theme(
        legend_position="top",
        panel_grid_minor=p9.element_blank(),
    )


def plot_survival(
    km: KaplanMeier,
    *,
    conf_int: bool = True,
    censor_marks: bool = True,
    risk_table: bool = False,
    times: Any = None,
    xlab: str = "Time",
    ylab: str = "Survival probability",
) -> Any:
    """Plot Kaplan-Meier survival curve(s) with plotnine.

    Renders one or more Kaplan-Meier survival curves as a publication-ready ggplot
    visualization. Each curve shows the proportion of subjects surviving (event-free) over
    time, with a shaded confidence band indicating uncertainty. Censoring events (subjects
    who exit the study without experiencing the event) are marked with tick points on the
    curve. If the fit is stratified (by groups), separate curves appear with distinct colors
    and a legend.

    The function returns a composable plotnine `ggplot` object: add layers, facets, scales,
    or themes using the `+` operator. Pass `risk_table=True` to stack an aligned
    numbers-at-risk table beneath the curve, showing how many subjects remain at-risk at each
    time point (a standard element in publication-quality survival plots). Confidence bands
    use point-wise confidence intervals (as computed by `KaplanMeier`) and are drawn as
    right-continuous step functions to match the step-function nature of the Kaplan-Meier
    estimator.

    The plot uses a light, minimal theme suitable for publications. Requires plotnine
    (install with `pip install greenwood[plotnine]`).

    Parameters
    ----------
    km
        A fitted `KaplanMeier` object, either unstratified (single curve) or stratified
        (multiple curves, one per group).
    conf_int
        If `True` (default), draw the confidence band as a shaded ribbon around the curve.
        Set `False` to hide the confidence band and show only the point estimate.
    censor_marks
        If `True` (default), mark censoring times with `+` symbols on the curve. Set `False`
        to hide these marks.
    risk_table
        If `True`, return a `plotnine.composition` stacking the curve over an aligned
        numbers-at-risk table; the table's x-axis aligns with the curve. If `False` (default),
        return only the survival curve plot.
    times
        Query times for the numbers-at-risk table (used only if `risk_table=True`). Defaults
        to six evenly spaced, rounded times from 0 to the maximum observed follow-up time.
        Specify a list of times (e.g., `[0, 100, 200, 300]`) to customize the table rows.
    xlab
        X-axis label (default `"Time"`).
    ylab
        Y-axis label (default `"Survival probability"`).

    Returns
    -------
    A plotnine `ggplot` object (or a `plotnine.composition` combining the curve and table if
    `risk_table=True`). The returned object is composable: you can add layers, scales, themes,
    and facets using plotnine's `+` operator or `/` operator for arrangement.

    Details
    -------
    The survival curve is drawn as a right-continuous step function, reflecting how the
    Kaplan-Meier estimate changes only at observed event times (not between them). Confidence
    bands are point-wise (not uniform/simultaneous), so they do not guarantee that the true
    survival curve lies entirely within the band.

    For stratified fits, group labels appear in the legend as `"Group"`. Censoring marks
    appear at the same survival level as the curve but with a `+` shape for visibility.

    Examples
    --------
    Fit a stratified Kaplan-Meier estimator on the bundled `lung` dataset and draw one
    survival curve per group (by sex). The result is a composable plotnine object, so you can
    add layers, scales, or themes to it.

    ```{python}
    import greenwood as gw

    lung = gw.load_dataset("lung", backend="polars")
    y = gw.Surv.right(lung["time"], event=(lung["status"] == 2))
    km = gw.KaplanMeier().fit(y, by=lung["sex"])

    gw.plot_survival(km)
    ```

    Add a numbers-at-risk table by passing `risk_table=True`:

    ```{python}
    gw.plot_survival(km, risk_table=True)
    ```

    Customize the plot by dropping the confidence band or censoring marks, specifying
    custom times for the risk table, or adding plotnine layers:

    ```{python}
    gw.plot_survival(km, conf_int=False, censor_marks=False, risk_table=True,
                     times=[0, 200, 400, 600])
    ```
    """
    import pandas as pd

    p9 = _require_plotnine()

    steps = pd.concat([_step_frame(b) for b in km._blocks], ignore_index=True)
    grouped = km._grouped

    plot = p9.ggplot(steps, p9.aes(x="time", y="estimate"))
    if conf_int:
        ribbon = steps.dropna(subset=["conf_low", "conf_high"])
        if grouped:
            plot = plot + p9.geom_ribbon(
                ribbon,
                p9.aes(ymin="conf_low", ymax="conf_high", fill="strata"),
                alpha=0.18,
                show_legend=False,
            )
        else:
            plot = plot + p9.geom_ribbon(
                ribbon,
                p9.aes(ymin="conf_low", ymax="conf_high"),
                alpha=0.18,
                fill="#20558A",
            )
    if grouped:
        plot = plot + p9.geom_line(p9.aes(color="strata"))
    else:
        plot = plot + p9.geom_line(color="#20558A")

    if censor_marks:
        censors = pd.concat([_censor_frame(b) for b in km._blocks], ignore_index=True)
        if len(censors):
            plot = plot + p9.geom_point(
                censors, p9.aes(x="time", y="estimate"), shape="+", size=3, show_legend=False
            )

    plot = (
        plot
        + p9.scale_y_continuous(limits=[0.0, 1.0])
        + p9.labs(x=xlab, y=ylab, color="Group", fill="Group")
        + theme_survival()
    )

    if not risk_table:
        return plot

    return plot / _risk_table_plot(km, times=times, xlab=xlab)


def _risk_table_plot(km: KaplanMeier, *, times: Any = None, xlab: str = "Time") -> Any:
    """A compact numbers-at-risk table as a plotnine plot (x aligned with the curve)."""
    p9 = _require_plotnine()
    # plotnine consumes pandas, so build the risk-table frame in pandas regardless of
    # which DataFrame libraries happen to be installed.
    data = get_risk_table_frame(km, times=times, format="pandas")
    # Show whole counts as integers (e.g. 90, not 90.0); keep weighted counts as-is.
    data["label"] = [f"{v:g}" for v in data["n_risk"]]
    grouped = km._grouped

    aes_kwargs = {"color": "strata"} if grouped else {}
    plot = (
        p9.ggplot(data, p9.aes(x="time", y="strata", label="label"))
        + p9.geom_text(p9.aes(**aes_kwargs), size=9, show_legend=False)
        + p9.labs(x=xlab, y="", title="Number at risk")
        + p9.theme_minimal()
        + p9.theme(
            panel_grid=p9.element_blank(),
            axis_ticks=p9.element_blank(),
            plot_title=p9.element_text(size=10),
        )
    )
    return plot


def risk_table(km: KaplanMeier, times: Any = None) -> Any:
    """Return the numbers-at-risk table as a standalone plotnine plot.

    Examples
    --------
    Fit a stratified Kaplan-Meier estimator on the bundled `lung` dataset, then render the
    numbers at risk at a chosen set of times as its own plotnine plot (whose x-axis lines up
    with a survival curve drawn over the same range).

    ```{python}
    import greenwood as gw

    lung = gw.load_dataset("lung", backend="polars")
    y = gw.Surv.right(lung["time"], event=(lung["status"] == 2))
    km = gw.KaplanMeier().fit(y, by=lung["sex"])

    gw.risk_table(km, times=[0, 250, 500, 750, 1000])
    ```
    """
    return _risk_table_plot(km, times=times)
