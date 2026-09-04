"""Schoenfeld residual plots for assessing the proportional hazards assumption.

Plots scaled Schoenfeld residuals against event time, one panel per covariate, with a loess smooth
and an optional reference line at the overall coefficient estimate. This is the visual companion to
`CoxPH.cox_zph()`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

import numpy as np

if TYPE_CHECKING:
    from .._cox import CoxPH

__all__ = ["plot_schoenfeld"]

_POINT_COLOR = "#20558A"
_SMOOTH_COLOR = "#E45756"
_REF_COLOR = "#888888"


def _prepare_schoenfeld_data(
    cox: CoxPH,
    *,
    transform: str,
) -> tuple[dict[str, Any], list[str]]:
    """Compute scaled Schoenfeld residuals and event times.

    Returns a dict with 'time' plus one column per covariate, and the list of term names.
    """
    residuals_list, times_list, _ = cox._event_contributions()
    t = np.array(times_list)
    arr = np.array(residuals_list)
    n_events = len(t)

    scaled = arr @ cox.naive_vcov_ * n_events + cox.coef_[None, :]

    if transform == "identity":
        x_vals = t
    elif transform == "log":
        x_vals = np.log(t)
    elif transform == "km":
        from .._nonparametric import KaplanMeier
        from .._surv import Surv

        surv = Surv.right(cox._exit, cox._event)
        km = KaplanMeier().fit(surv)
        km_surv = np.interp(t, km.time_, km.survival_, left=1.0, right=float(km.survival_[-1]))
        x_vals = 1.0 - km_surv
    elif transform == "rank":
        x_vals = np.argsort(np.argsort(t)).astype(float) / (n_events - 1)
    else:
        raise ValueError(
            f"transform must be 'identity', 'log', 'km', or 'rank', got {transform!r}."
        )

    terms = cox.term_names_
    data: dict[str, Any] = {"time": x_vals}
    for j, name in enumerate(terms):
        data[name] = scaled[:, j]

    return data, terms


def plot_schoenfeld(
    cox: Any,
    *,
    transform: str = "identity",
    show_zph: bool = True,
    title: str | None = None,
    backend: Literal["altair", "plotnine"] = "altair",
    width: int = 280,
    height: int = 200,
) -> Any:
    r"""Scaled Schoenfeld residual plots for assessing proportional hazards.

    Plots scaled Schoenfeld residuals against event time, one panel per covariate, with a loess
    smooth line and a dashed reference line at the overall coefficient estimate. Under proportional
    hazards, the smooth should be approximately flat. A trend indicates that the covariate's effect
    changes over time.

    This is the visual equivalent of R's `plot(cox.zph(fit))`.

    Parameters
    ----------
    cox
        A fitted `CoxPH` model.
    transform
        Time-axis transformation: `"identity"` (default), `"log"`, `"km"` (Kaplan-Meier failure
        probability), or `"rank"`.
    show_zph
        If `True` (default), annotate each panel with the Grambsch-Therneau test p-value from
        `cox_zph()`.
    title
        Optional supertitle for the combined chart.
    backend
        Plotting backend: `"altair"` (default) or `"plotnine"`.
    width
        Width of each panel in pixels (Altair) or approximate inches × 72 (plotnine).
    height
        Height of each panel in pixels (Altair) or approximate inches × 72 (plotnine).

    Returns
    -------
    alt.Chart or plotnine.ggplot
        A composite chart (Altair `HConcatChart`) or a faceted plotnine plot.

    Examples
    --------
    ```{python}
    import greenwood as gw

    lung = gw.load_dataset("lung", backend="polars")
    y = gw.Surv.right(lung["time"], event=(lung["status"] == 2))
    cox = gw.CoxPH().fit(y, lung[["age", "sex"]])

    gw.plot_schoenfeld(cox)
    ```

    Use a log time axis and suppress the p-value annotation:

    ```{python}
    gw.plot_schoenfeld(cox, transform="log", show_zph=False)
    ```

    Use the plotnine backend for a static ggplot object:

    ```{python}
    gw.plot_schoenfeld(cox, backend="plotnine")
    ```
    """
    data, terms = _prepare_schoenfeld_data(cox, transform=transform)

    zph_p: dict[str, float] = {}
    if show_zph:
        zph = cox.cox_zph(transform=transform)
        for name in terms:
            zph_p[name] = zph.per_term[name]["p_value"]

    _XLAB = {
        "identity": "Time",
        "log": "log(Time)",
        "km": "KM failure probability",
        "rank": "Rank(Time)",
    }
    xlab = _XLAB.get(transform, "Time")

    if backend == "altair":
        return _plot_schoenfeld_altair(
            data,
            terms=terms,
            zph_p=zph_p,
            coef=cox.coef_,
            xlab=xlab,
            title=title,
            width=width,
            height=height,
        )
    if backend == "plotnine":
        return _plot_schoenfeld_plotnine(
            data,
            terms=terms,
            zph_p=zph_p,
            coef=cox.coef_,
            xlab=xlab,
            title=title,
            width=width,
            height=height,
        )
    raise ValueError(f"backend must be 'altair' or 'plotnine', got {backend!r}")


def _plot_schoenfeld_altair(
    data: dict[str, Any],
    *,
    terms: list[str],
    zph_p: dict[str, float],
    coef: Any,
    xlab: str,
    title: str | None,
    width: int,
    height: int,
) -> Any:
    try:
        import altair as alt
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "plot_schoenfeld() with backend='altair' requires altair. "
            "Install with `pip install greenwood[altair]`."
        ) from exc

    from .._backends import to_dataframe

    charts: list[Any] = []
    for j, name in enumerate(terms):
        panel_data = {"time": data["time"], "residual": data[name]}
        frame = to_dataframe(panel_data)

        base = alt.Chart(frame).encode(
            x=alt.X("time:Q", title=xlab),
            y=alt.Y("residual:Q", title=f"Beta(t) for {name}"),
        )

        points = base.mark_circle(size=20, opacity=0.35, color=_POINT_COLOR)

        smooth = base.transform_loess("time", "residual", bandwidth=0.6).mark_line(
            color=_SMOOTH_COLOR, strokeWidth=2
        )

        ref_line = (
            alt.Chart(to_dataframe({"y": [float(coef[j])]}))
            .mark_rule(strokeDash=[4, 4], color=_REF_COLOR)
            .encode(y="y:Q")
        )

        panel_title = f"Beta(t) for {name}"
        if name in zph_p:
            p_val = zph_p[name]
            p_text = f"p = {p_val:.3f}" if p_val >= 0.001 else f"p = {p_val:.1e}"
            panel_title = f"{panel_title}  ({p_text})"

        panel = (points + smooth + ref_line).properties(
            width=width, height=height, title=panel_title
        )

        charts.append(panel)

    combined = alt.hconcat(*charts)
    if title:
        combined = combined.properties(title=title)
    return combined


def _plot_schoenfeld_plotnine(
    data: dict[str, Any],
    *,
    terms: list[str],
    zph_p: dict[str, float],
    coef: Any,
    xlab: str,
    title: str | None,
    width: int,
    height: int,
) -> Any:
    try:
        import plotnine as p9
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "plot_schoenfeld() with backend='plotnine' requires plotnine. "
            "Install with `pip install greenwood[plotnine]`."
        ) from exc

    import pandas as pd

    n = len(data["time"])
    rows: list[dict[str, Any]] = []
    for name in terms:
        for i in range(n):
            rows.append(
                {
                    "time": float(data["time"][i]),
                    "residual": float(data[name][i]),
                    "term": name,
                }
            )
    long = pd.DataFrame(rows)
    long["term"] = pd.Categorical(long["term"], categories=terms, ordered=True)

    ref_df = pd.DataFrame(
        {
            "term": pd.Categorical(terms, categories=terms, ordered=True),
            "yintercept": [float(coef[j]) for j in range(len(terms))],
        }
    )

    p = (
        p9.ggplot(long, p9.aes("time", "residual"))
        + p9.geom_point(size=1, alpha=0.35, color=_POINT_COLOR)
        + p9.geom_smooth(method="lowess", color=_SMOOTH_COLOR, se=False, span=0.6)
        + p9.geom_hline(
            p9.aes(yintercept="yintercept"), data=ref_df, linetype="dashed", color=_REF_COLOR
        )
        + p9.facet_wrap("term", scales="free_y", ncol=len(terms))
        + p9.labs(x=xlab, y="Scaled Schoenfeld residual")
        + p9.theme_minimal()
        + p9.theme(strip_text=p9.element_text(size=10, weight="bold"))
    )

    if zph_p:
        label_rows = []
        for name in terms:
            if name in zph_p:
                p_val = zph_p[name]
                p_text = f"p = {p_val:.3f}" if p_val >= 0.001 else f"p = {p_val:.1e}"
                label_rows.append({"term": name, "label": p_text})
        if label_rows:
            label_df = pd.DataFrame(label_rows)
            label_df["term"] = pd.Categorical(label_df["term"], categories=terms, ordered=True)
            label_df["time"] = float(np.max(data["time"]))
            label_df["residual"] = [float(np.max(data[row["term"]])) for row in label_rows]
            p = p + p9.geom_text(
                p9.aes(label="label"),
                data=label_df,
                ha="right",
                va="top",
                size=8,
                fontstyle="italic",
                color=_REF_COLOR,
            )

    if title:
        p = p + p9.ggtitle(title)

    return p
