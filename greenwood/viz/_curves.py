def _require_plotnine() -> Any:
    try:
        import plotnine as p9
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise ImportError(
            "Visualization requires plotnine. Install it with `pip install greenwood[viz]`."
        ) from exc
    return p9


def _strata_label(block: Any) -> str:
    return _OVERALL if block.label is None else str(block.label)


def _default_times(km: KaplanMeier) -> list[float]:
    """Six evenly spaced, rounded times from 0 to the largest observed time."""
    max_t = max(float(b.time[-1]) for b in km._blocks if b.time.size)
    raw = np.linspace(0.0, max_t, 6)
    return sorted({round(float(t)) for t in raw})


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


def _n_at_risk(block: Any, times: Array) -> Array:
    """Number at risk at each query time (as in R's `summary(fit, times=)$n.risk`)."""
    idx = np.searchsorted(block.time, times, side="left")
    out = np.zeros(times.shape[0])
    valid = idx < block.time.shape[0]
    out[valid] = block.n_risk[idx[valid]]
    return out


def risk_table_data(km: KaplanMeier, times: Any = None) -> Any:
    """Return a tidy frame of the number at risk per stratum at each of `times`."""
    import pandas as pd

    query = np.asarray(_default_times(km) if times is None else times, dtype=float)
    rows: list[dict[str, Any]] = []
    for block in km._blocks:
        counts = _n_at_risk(block, query)
        for t, n in zip(query, counts, strict=True):
            rows.append({"strata": _strata_label(block), "time": float(t), "n_risk": float(n)})
    return pd.DataFrame(rows)


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

    Parameters
    ----------
    km
        A fitted `KaplanMeier`.
    conf_int
        Draw the confidence band as a stepped ribbon.
    censor_marks
        Mark censoring times with tick points on the curve.
    risk_table
        If true, return a `plotnine.composition` stacking the curve over an aligned
        numbers-at-risk table instead of a bare `ggplot`.
    times
        Times for the risk table (defaults to six evenly spaced, rounded times).
    xlab, ylab
        Axis labels.

    Returns
    -------
    A plotnine `ggplot` (or a composition when `risk_table=True`).
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
    data = risk_table_data(km, times=times)
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
    """Return the numbers-at-risk table as a standalone plotnine plot."""
    return _risk_table_plot(km, times=times)
