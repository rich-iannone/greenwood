"""Diagnostic plots for identifying influential and outlying observations in Cox models.

Produces scatter-plot panels of deviance residuals, leverage, and likelihood
displacement against the linear predictor, with the most influential observations
labeled.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

import numpy as np

if TYPE_CHECKING:
    from .._cox import CoxPH

__all__ = ["plot_influence"]

_POINT_COLOR = "#20558A"
_HIGHLIGHT_COLOR = "#E45756"


def _prepare_data(cox: CoxPH, highlight: int) -> dict[str, Any]:
    """Build a dict of numpy arrays with influence diagnostics and highlight flags."""
    scores = cox._score_residuals(cox.coef_)
    dfb = scores @ cox.naive_vcov_
    leverage = np.sum(dfb * scores, axis=1)
    mart = cox._martingale_residuals()
    event = cox._event.astype(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        log_term = np.where(event > mart, np.log(event - mart), 0.0)
    dev = np.sign(mart) * np.sqrt(-2.0 * (mart + event * log_term))
    info_inv = np.linalg.solve(cox.naive_vcov_, np.eye(cox.naive_vcov_.shape[0]))
    ld = np.sum(dfb @ info_inv * dfb, axis=1)
    lp = np.asarray(cox.predict(type="lp")).ravel()
    obs = np.arange(1, len(lp) + 1)

    threshold = float(np.sort(ld)[-highlight]) if highlight > 0 else float("inf")
    influential = ld >= threshold

    return {
        "leverage": leverage,
        "martingale": mart,
        "deviance": dev,
        "ld": ld,
        "lp": lp,
        "obs": obs,
        "influential": influential,
    }


def plot_influence(
    cox: Any,
    *,
    highlight: int = 3,
    panels: tuple[str, ...] | list[str] | None = None,
    title: str | None = None,
    backend: Literal["altair", "plotnine"] = "altair",
    width: int = 240,
    height: int = 200,
) -> Any:
    r"""Diagnostic scatter plots for identifying influential observations in a Cox model.

    Produces a horizontal row of panels plotting key diagnostics against the linear predictor. The
    most influential observations (by likelihood displacement) are highlighted in red and labeled
    with their observation number.

    Parameters
    ----------
    cox
        A fitted `CoxPH` model.
    highlight
        Number of most influential observations to highlight and label (default `3`). Set to `0` to
        disable highlighting.
    panels
        Which diagnostic panels to show. Each name is a column from `influence_diagnostics()`. The
        default is `("deviance", "leverage", "ld")`.
    title
        Optional supertitle for the combined chart.
    backend
        Plotting backend: `"altair"` (default) or `"plotnine"`.
    width
        Width of each panel in pixels (Altair) or inches (plotnine).
    height
        Height of each panel in pixels (Altair) or inches (plotnine).

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

    gw.plot_influence(cox)
    ```

    Show only deviance residuals and leverage, highlighting the top 5:

    ```{python}
    gw.plot_influence(cox, panels=["deviance", "leverage"], highlight=5)
    ```

    Use the plotnine backend for a static ggplot object:

    ```{python}
    gw.plot_influence(cox, backend="plotnine")
    ```
    """
    if panels is None:
        panels = ["deviance", "leverage", "ld"]

    data = _prepare_data(cox, highlight)

    if backend == "altair":
        return _plot_influence_altair(data, panels=panels, title=title, width=width, height=height)
    if backend == "plotnine":
        return _plot_influence_plotnine(
            data, panels=panels, title=title, width=width, height=height
        )
    raise ValueError(f"backend must be 'altair' or 'plotnine', got {backend!r}")


_Y_LABELS: dict[str, str] = {
    "deviance": "Deviance residual",
    "leverage": "Leverage",
    "ld": "Likelihood displacement",
    "martingale": "Martingale residual",
}


def _plot_influence_altair(
    data: dict[str, Any],
    *,
    panels: list[str] | tuple[str, ...],
    title: str | None,
    width: int,
    height: int,
) -> Any:
    try:
        import altair as alt
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "plot_influence() with backend='altair' requires altair. "
            "Install with `pip install greenwood[altair]`."
        ) from exc

    from .._backends import to_dataframe

    frame = to_dataframe(data)

    charts: list[Any] = []
    for panel_name in panels:
        ylab = _Y_LABELS.get(panel_name, panel_name)

        base = alt.Chart(frame).encode(
            x=alt.X("lp:Q", title="Linear predictor"),
            y=alt.Y(f"{panel_name}:Q", title=ylab),
        )

        points = base.mark_circle(size=30, opacity=0.5).encode(
            color=alt.condition(
                alt.datum.influential,
                alt.value(_HIGHLIGHT_COLOR),
                alt.value(_POINT_COLOR),
            ),
        )

        labels = (
            base.transform_filter(alt.datum.influential)
            .mark_text(align="left", dx=6, fontSize=10, color=_HIGHLIGHT_COLOR)
            .encode(text="obs:N")
        )

        chart = (points + labels).properties(width=width, height=height)
        charts.append(chart)

    combined = alt.hconcat(*charts).resolve_scale(color="independent")
    if title:
        combined = combined.properties(title=title)
    return combined


def _plot_influence_plotnine(
    data: dict[str, Any],
    *,
    panels: list[str] | tuple[str, ...],
    title: str | None,
    width: int,
    height: int,
) -> Any:
    try:
        import plotnine as p9
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "plot_influence() with backend='plotnine' requires plotnine. "
            "Install with `pip install greenwood[plotnine]`."
        ) from exc

    import pandas as pd

    n = len(data["lp"])
    rows: list[dict[str, Any]] = []
    for panel_name in panels:
        ylab = _Y_LABELS.get(panel_name, panel_name)
        for i in range(n):
            rows.append(
                {
                    "lp": float(data["lp"][i]),
                    "value": float(data[panel_name][i]),
                    "panel": ylab,
                    "obs": int(data["obs"][i]),
                    "influential": bool(data["influential"][i]),
                }
            )
    long = pd.DataFrame(rows)
    long["panel"] = pd.Categorical(
        long["panel"],
        categories=[_Y_LABELS.get(p, p) for p in panels],
        ordered=True,
    )

    p = (
        p9.ggplot(long, p9.aes("lp", "value"))
        + p9.geom_point(
            p9.aes(color="influential"),
            size=1.5,
            alpha=0.5,
        )
        + p9.scale_color_manual(values={True: _HIGHLIGHT_COLOR, False: _POINT_COLOR})
        + p9.facet_wrap("panel", scales="free_y", ncol=len(panels))
        + p9.labs(x="Linear predictor", y="")
        + p9.theme_minimal()
        + p9.theme(
            legend_position="none",
            strip_text=p9.element_text(size=10, weight="bold"),
        )
    )

    highlighted = pd.DataFrame(long[long["influential"]])
    if not highlighted.empty:
        p = p + p9.geom_text(
            p9.aes(label="obs"),
            data=highlighted,
            size=7,
            ha="left",
            nudge_x=0.05,
            color=_HIGHLIGHT_COLOR,
        )

    if title:
        p = p + p9.ggtitle(title)

    return p
