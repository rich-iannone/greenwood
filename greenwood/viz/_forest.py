"""Forest plots for hazard ratios, RMST differences, and other contrasts.

Forest plots display point estimates and confidence intervals for multiple effects or
comparisons, enabling visual comparison of precision and direction across contrasts.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import numpy.typing as npt

from .._backends import to_dataframe

if TYPE_CHECKING:
    pass

__all__ = ["plot_forest", "theme_forest"]

Array = npt.NDArray[Any]


def _forest_plot_data(
    estimates: Array | dict[str, Any],
    ci_lower: Array | None = None,
    ci_upper: Array | None = None,
    labels: Array | None = None,
    scale: Literal["log", "linear"] = "log",
    reference_line: float | None = None,
) -> dict[str, Any]:
    r"""Prepare forest plot data from estimates and confidence intervals.

    Formats point estimates and confidence intervals into a tidy structure suitable for
    visualization. Supports both hazard ratios (log scale) and contrasts like RMST
    differences (linear scale).

    This is a data preparation layer—it returns a structured dict that can be passed to
    plotnine or Altair visualization functions (see `plot_forest_plotnine` and
    `plot_forest_altair`).

    Parameters
    ----------
    estimates
        Either:

        - A 1-D array of point estimates (hazard ratios, RMST diffs, etc.)
        - A dict with keys `"estimate"`, `"ci_lower"`, `"ci_upper"`, and optionally
          `"labels"` (when provided, other params are ignored)

    ci_lower
        Lower bounds of confidence intervals. Required if `estimates` is an array.
    ci_upper
        Upper bounds of confidence intervals. Required if `estimates` is an array.
    labels
        Labels for each estimate (e.g., covariate names, group names). If `None`,
        labeled as "Estimate 1", "Estimate 2", etc.
    scale
        Transformation for the x-axis:

        - `"log"` (default): for hazard ratios, odds ratios, etc. On a log scale, symmetric
          CIs appear symmetric, and HR=1 (no effect) aligns with x=0. Usually called a
          "log scale" forest plot.
        - `"linear"`: for RMST differences, mean differences, etc. No transformation;
          the reference line typically at 0.

    reference_line
        Location of the reference line (e.g., 1 for HR on log scale, 0 for difference).
        If `None`, inferred from scale: 0 for log scale, 0 for linear scale. For HR on
        log scale, recommend `reference_line=0` (since log(1) = 0).

    Returns
    -------
    dict
        Dictionary with keys:

        - `"data"`: a list of dicts, each with keys `"label"`, `"estimate"`, `"ci_lower"`,
          `"ci_upper"` (useful for DataFrames or manual plotting).
        - `"scale"`: the scale used (`"log"` or `"linear"`).
        - `"reference_line"`: the reference value on the display scale.

    Details
    -------
    **Log scale for HR**: When `scale="log"`, estimates and CIs are log-transformed
    internally (if not already). This is the standard for forest plots of hazard ratios,
    where HR=1 (no effect) becomes log(1)=0 on the plot.

    **RMST differences**: Use `scale="linear"`, `reference_line=0`. No transformation is
    applied; the plot displays raw differences.

    Examples
    --------
    Forest plot data for three hazard ratios:

    ```python
    import greenwood as gw

    # Hazard ratios and 95% CIs for three covariates
    hr = gw.viz._forest_plot_data(
        estimates=[0.85, 1.02, 1.15],
        ci_lower=[0.71, 0.89, 0.98],
        ci_upper=[1.01, 1.17, 1.35],
        labels=["Age (per 10 years)", "Sex (F vs M)", "ECOG (0 vs 1)"],
        scale="log",
    )
    hr
    ```

    RMST differences (e.g., from group comparisons):

    ```python
    rmst_diff = gw.viz._forest_plot_data(
        estimates=[15.3, -8.5, 5.2],
        ci_lower=[2.1, -20.3, -5.1],
        ci_upper=[28.5, 3.3, 15.5],
        labels=["Drug A vs Placebo", "Drug B vs Placebo", "Drug A vs Drug B"],
        scale="linear",
        reference_line=0,
    )
    rmst_diff
    ```
    """
    # Parse input
    if isinstance(estimates, dict):
        data_dict = estimates
        est_array = np.asarray(data_dict["estimate"], dtype=float)
        ci_lower_array = np.asarray(data_dict["ci_lower"], dtype=float)
        ci_upper_array = np.asarray(data_dict["ci_upper"], dtype=float)
        labels_list = data_dict.get("labels", None)
    else:
        est_array = np.asarray(estimates, dtype=float)
        if ci_lower is None or ci_upper is None:
            raise ValueError("If estimates is an array, ci_lower and ci_upper must be provided.")
        ci_lower_array = np.asarray(ci_lower, dtype=float)
        ci_upper_array = np.asarray(ci_upper, dtype=float)
        labels_list = labels

    if est_array.ndim != 1:
        raise ValueError("estimates must be 1-D.")
    if ci_lower_array.shape != est_array.shape or ci_upper_array.shape != est_array.shape:
        raise ValueError("ci_lower and ci_upper must have the same shape as estimates.")

    n = len(est_array)
    if labels_list is None:
        labels_list = [f"Estimate {i + 1}" for i in range(n)]
    else:
        labels_list = list(labels_list)
        if len(labels_list) != n:
            raise ValueError("labels must have the same length as estimates.")

    # Apply log transform if needed
    if scale == "log":
        # Log transform: estimates and CIs
        est_display = np.log(est_array)
        ci_lower_display = np.log(ci_lower_array)
        ci_upper_display = np.log(ci_upper_array)
        if reference_line is None:
            reference_line = 0.0  # log(1) = 0
    elif scale == "linear":
        est_display = est_array
        ci_lower_display = ci_lower_array
        ci_upper_display = ci_upper_array
        if reference_line is None:
            reference_line = 0.0
    else:
        raise ValueError(f"scale must be 'log' or 'linear', got {scale!r}.")

    # Build data list
    data_list = [
        {
            "label": lbl,
            "estimate": float(est),
            "ci_lower": float(ci_l),
            "ci_upper": float(ci_u),
        }
        for lbl, est, ci_l, ci_u in zip(
            labels_list, est_display, ci_lower_display, ci_upper_display, strict=True
        )
    ]

    return {
        "data": data_list,
        "scale": scale,
        "reference_line": float(reference_line),
    }


def _forest_plot(
    estimates: Array | None = None,
    ci_lower: Array | None = None,
    ci_upper: Array | None = None,
    labels: Array | None = None,
    scale: Literal["log", "linear"] = "log",
    reference_line: float | None = None,
    title: str | None = None,
    width: int = 600,
    height: int = 400,
    backend: str = "altair",
) -> Any:
    r"""Create an interactive forest plot.

    Visualizes point estimates with confidence intervals as an interactive Altair chart
    with hoverable confidence intervals and reference line.

    Parameters
    ----------
    estimates
        A 1-D array of point estimates (hazard ratios, RMST diffs, etc.)
    ci_lower
        Lower bounds of confidence intervals.
    ci_upper
        Upper bounds of confidence intervals.
    labels
        Labels for each estimate (e.g., covariate names, group names). If `None`,
        labeled as "Estimate 1", "Estimate 2", etc.
    scale
        Transformation for the x-axis:

        - `"log"` (default): for hazard ratios, odds ratios, etc. On a log scale, symmetric
          CIs appear symmetric, and HR=1 (no effect) aligns with x=0.
        - `"linear"`: for RMST differences, mean differences, etc. No transformation;
          the reference line typically at 0.

    reference_line
        Location of the reference line (e.g., `1` for HR on log scale, `0` for difference).
        If `None`, defaults to `0` (which is log(1) for log scale).
    title
        Plot title. If `None`, no title.
    width
        Plot width in pixels (default 600).
    height
        Plot height in pixels (default 400).
    backend
        Plotting backend (default `"altair"`). Currently only `"altair"` is supported.

    Returns
    -------
    altair.Chart
        An Altair chart object, interactive and composable.

    Examples
    --------
    Hazard ratios from a Cox model:

    ```python
    import greenwood as gw

    # Hazard ratios and 95% CIs for three covariates
    gw.forest_plot(
        estimates=[0.85, 1.02, 1.15],
        ci_lower=[0.71, 0.89, 0.98],
        ci_upper=[1.01, 1.17, 1.35],
        labels=["Age (per 10 years)", "Sex (F vs M)", "ECOG (0 vs 1)"],
        scale="log",
        title="Hazard Ratios",
    )
    ```

    RMST differences (linear scale):

    ```python
    gw.forest_plot(
        estimates=[15.3, -8.5, 5.2],
        ci_lower=[2.1, -20.3, -5.1],
        ci_upper=[28.5, 3.3, 15.5],
        labels=["Drug A vs Placebo", "Drug B vs Placebo", "Drug A vs Drug B"],
        scale="linear",
        reference_line=0,
        title="RMST Differences",
    )
    ```
    """
    if backend != "altair":
        raise ValueError(f"backend must be 'altair', got {backend!r}")

    # Validate estimates is provided and not None
    if estimates is None:
        raise ValueError("estimates must be provided and cannot be None")

    # Prepare data internally
    forest_data = _forest_plot_data(
        estimates=estimates,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        labels=labels,
        scale=scale,
        reference_line=reference_line,
    )

    try:
        import altair as alt
    except ImportError as exc:
        raise ImportError("altair required; install with `pip install greenwood[altair]`.") from exc

    data_list = forest_data["data"]
    # Convert to dict format for to_dataframe (transpose the list of dicts)
    data_dict = {k: [d[k] for d in data_list] for k in data_list[0]}
    # to_dataframe returns pandas/polars/pyarrow; Altair accepts all via Narwhals
    df = to_dataframe(data_dict)

    # Create base chart for CIs
    ci_lines = (
        alt.Chart(df)
        .mark_rule()
        .encode(
            y=alt.Y("label:N", axis=alt.Axis(labelAngle=0)),
            x=alt.X(
                "ci_lower:Q",
                title="Estimate (log scale)" if forest_data["scale"] == "log" else "Estimate",
            ),
            x2="ci_upper:Q",
            tooltip=["label:N", "estimate:Q", "ci_lower:Q", "ci_upper:Q"],
        )
    )

    # Point estimates
    points = (
        alt.Chart(df)
        .mark_point(size=100)
        .encode(
            y=alt.Y("label:N", axis=alt.Axis(labelAngle=0)),
            x="estimate:Q",
            tooltip=["label:N", "estimate:Q", "ci_lower:Q", "ci_upper:Q"],
            color=alt.value("steelblue"),
        )
    )

    # Reference line
    ref_df = to_dataframe({"ref": [forest_data["reference_line"]]})
    ref_line = (
        alt.Chart(ref_df).mark_rule(color="gray", strokeDash=[5, 5], opacity=0.5).encode(x="ref:Q")
    )

    # Build properties dict, only including title if not None (Altair validates title type)
    props: dict[str, int | str] = {"width": width, "height": height}
    if title is not None:
        props["title"] = title
    chart = (ci_lines + points + ref_line).properties(**props)

    return chart


# ---------------------------------------------------------------------------
# plot_forest — Cox-aware high-level wrapper
# ---------------------------------------------------------------------------


def _fmt_pvalue(p: float) -> str:
    """Format a p-value for display in forest plot annotations."""
    if p < 0.001:
        return "<0.001"
    if p < 0.01:
        return f"{p:.3f}"
    return f"{p:.2f}"


def _extract_forest_frame(
    result: Any,
    *,
    exponentiate: bool,
    term_labels: dict[str, str] | None,
) -> Any:
    """Return a Pandas DataFrame with columns: term, estimate, ci_lower, ci_upper, p_value.

    Accepts a fitted CoxPH object or a tidy DataFrame that already has the required columns
    (`term`, `estimate`, `ci_lower`/`conf_low`, `ci_upper`/`conf_high`). When *result* is a
    CoxPH object, *exponentiate* controls whether hazard ratios (`True`) or log-hazard
    coefficients (`False`) are extracted. For DataFrames the columns are used as-is.
    """
    import pandas as pd

    if hasattr(result, "term_names_") and hasattr(result, "hazard_ratio_"):
        # CoxPH result object
        terms = list(result.term_names_)
        if exponentiate:
            estimates = np.exp(result.coef_).tolist()
            ci_lower = np.exp(result.conf_low_).tolist()
            ci_upper = np.exp(result.conf_high_).tolist()
        else:
            estimates = result.coef_.tolist()
            ci_lower = result.conf_low_.tolist()
            ci_upper = result.conf_high_.tolist()
        p_values = result.p_value_.tolist()
        df = pd.DataFrame(
            {
                "term": terms,
                "estimate": estimates,
                "ci_lower": ci_lower,
                "ci_upper": ci_upper,
                "p_value": p_values,
            }
        )
    elif hasattr(result, "__dataframe__") or isinstance(result, dict):
        # Tidy DataFrame (pandas / polars / pyarrow) or plain dict
        if isinstance(result, dict):
            df = pd.DataFrame(result)
        else:
            try:
                df = result.to_pandas()  # polars / pyarrow
            except AttributeError:
                df = pd.DataFrame(result)
        # Normalise column names
        df = df.rename(
            columns={"conf_low": "ci_lower", "conf_high": "ci_upper", "std_error": "se"},
            errors="ignore",
        )
        required = {"term", "estimate", "ci_lower", "ci_upper"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(
                f"DataFrame is missing required column(s): {sorted(missing)}. "
                "Expected: term, estimate, ci_lower (or conf_low), ci_upper (or conf_high)."
            )
    else:
        raise TypeError(
            f"Expected a CoxPH result or a tidy DataFrame, got {type(result).__name__!r}."
        )

    if term_labels:
        df["term"] = [term_labels.get(t, t) for t in df["term"]]

    return df


def theme_forest() -> Any:
    r"""A minimal plotnine theme for forest plots.

    Returns a composable plotnine theme object suitable for forest plots of hazard ratios
    or other contrasts. Horizontal grid lines are suppressed; the y-axis line is removed
    so the term labels stand alone.

    Returns
    -------
    plotnine.theme
        A composable theme object. Add it to a `ggplot` with `+`.

    Examples
    --------
    ```python
    import greenwood as gw

    lung = gw.load_dataset("lung", backend="polars")
    y = gw.Surv.right(lung["time"], event=(lung["status"] == 2))
    cox = gw.CoxPH().fit(y, lung[["age", "sex"]])

    gw.plot_forest(cox, backend="plotnine") + gw.theme_forest()
    ```
    """
    try:
        import plotnine as p9
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "theme_forest() requires plotnine. Install with `pip install greenwood[plotnine]`."
        ) from exc
    return p9.theme_minimal() + p9.theme(
        axis_line_y=p9.element_blank(),
        panel_grid_minor=p9.element_blank(),
        panel_grid_major_y=p9.element_blank(),
        legend_position="none",
    )


def plot_forest(
    result: Any,
    *,
    scale: Literal["log", "linear"] | None = None,
    exponentiate: bool = True,
    title: str | None = None,
    term_labels: dict[str, str] | None = None,
    xlab: str | None = None,
    backend: Literal["altair", "plotnine"] = "altair",
    width: int = 600,
    height: int = 400,
) -> Any:
    r"""Forest plot of hazard ratios (or other contrasts) with confidence intervals.

    Accepts a fitted `~greenwood.CoxPH` object or any tidy DataFrame that contains
    `term`, `estimate`, `ci_lower` (or `conf_low`), and `ci_upper` (or
    `conf_high`) columns. The latter supports subgroup forest plots where you already
    have the summary estimates (e.g., from running Cox per subgroup and calling
    `~greenwood.tidy` on each).

    Parameters
    ----------
    result
        A fitted `CoxPH` result object, or a tidy DataFrame with one row per term.
    scale
        X-axis scale: `"log"` for hazard ratios (reference line at HR = 1) or
        `"linear"` for differences (reference line at 0). When `None` (default),
        the scale is inferred: `"log"` for a CoxPH object with `exponentiate=True`,
        `"linear"` otherwise. Pass `scale="log"` explicitly when supplying a
        DataFrame of hazard ratios.
    exponentiate
        When *result* is a `CoxPH` object: if `True` (default) extract hazard ratios;
        if `False` extract log-hazard coefficients. Ignored when *result* is a
        DataFrame or when *scale* is set explicitly.
    title
        Plot title. Defaults to `None` (no title).
    term_labels
        Optional mapping from internal term names to display labels, e.g.
        `{"age": "Age (years)", "sex": "Female vs. Male"}`. Only the terms listed are
        renamed; others keep their original names.
    xlab
        X-axis label. Defaults to `"Hazard Ratio"` for log scale and `"Estimate"`
        for linear scale.
    backend
        Plotting backend: `"altair"` (default, interactive) or `"plotnine"`
        (composable ggplot2-style object).
    width
        Plot width in pixels (default 600). Altair only.
    height
        Plot height in pixels (default 400). Altair only.

    Returns
    -------
    altair.Chart or plotnine.ggplot
        A composable chart object. Add layers, scales, or themes using the `+` operator
        (plotnine) or `.properties()` / `.interactive()` (Altair).

    Examples
    --------
    Fit a Cox model and draw a forest plot of hazard ratios:

    ```{python}
    import greenwood as gw

    lung = gw.load_dataset("lung", backend="polars")
    y = gw.Surv.right(lung["time"], event=(lung["status"] == 2))
    cox = gw.CoxPH().fit(y, lung[["age", "sex", "ph.ecog"]])

    gw.plot_forest(cox)
    ```

    Rename terms for publication display:

    ```{python}
    gw.plot_forest(
        cox,
        term_labels={"age": "Age (years)", "sex": "Female vs. Male", "ph.ecog": "ECOG PS"},
        title="Cox Model: Lung Cancer",
    )
    ```

    Build a subgroup forest plot from a tidy DataFrame. Pass `scale="log"` when the
    estimates are hazard ratios:

    ```{python}
    import pandas as pd

    subgroups = pd.DataFrame({
        "term": ["Age < 60", "Age \u2265 60", "Male", "Female"],
        "estimate": [0.72, 0.91, 0.85, 0.68],
        "ci_lower": [0.51, 0.74, 0.68, 0.50],
        "ci_upper": [1.01, 1.12, 1.06, 0.92],
    })
    gw.plot_forest(subgroups, scale="log")
    ```

    For RMST differences or other linear-scale contrasts, omit `scale` (or pass
    `scale="linear"` explicitly):

    ```{python}
    rmst = pd.DataFrame({
        "term": ["Drug A vs Placebo", "Drug B vs Placebo"],
        "estimate": [45, 25],
        "ci_lower": [-10, -20],
        "ci_upper": [95, 70],
    })
    gw.plot_forest(rmst)
    ```

    Use a plotnine backend for a static, composable ggplot object:

    ```{python}
    gw.plot_forest(cox, backend="plotnine")
    ```
    """
    df = _extract_forest_frame(result, exponentiate=exponentiate, term_labels=term_labels)

    _is_cox = hasattr(result, "term_names_") and hasattr(result, "hazard_ratio_")
    if scale is not None:
        use_log = scale == "log"
    elif _is_cox:
        use_log = exponentiate
    else:
        use_log = False
    vline_x = 1.0 if use_log else 0.0
    x_label = xlab or ("Hazard Ratio" if use_log else "Estimate")

    if backend == "altair":
        return _plot_forest_altair(
            df,
            use_log=use_log,
            vline_x=vline_x,
            x_label=x_label,
            title=title,
            width=width,
            height=height,
        )
    elif backend == "plotnine":
        return _plot_forest_plotnine(
            df,
            use_log=use_log,
            vline_x=vline_x,
            x_label=x_label,
            title=title,
        )
    else:
        raise ValueError(f"backend must be 'altair' or 'plotnine', got {backend!r}")


def _plot_forest_altair(
    df: Any,
    *,
    use_log: bool,
    vline_x: float,
    x_label: str,
    title: str | None,
    width: int,
    height: int,
) -> Any:
    """Altair implementation of plot_forest."""
    try:
        import altair as alt
    except ImportError as exc:
        raise ImportError(
            "plot_forest() with backend='altair' requires altair. "
            "Install with `pip install greenwood[altair]`."
        ) from exc

    has_pvalue = "p_value" in df.columns

    # Pre-log the estimates for display (Altair doesn't natively do log axes the same way)
    display = df.copy()
    if use_log:
        display["est_display"] = np.log(display["estimate"])
        display["ci_lower_display"] = np.log(display["ci_lower"])
        display["ci_upper_display"] = np.log(display["ci_upper"])
        ref_display = 0.0  # log(1)
        # Build tick labels: powers of 2 spanning the data
        all_vals = np.concatenate(
            [display["ci_lower"].values, display["ci_upper"].values, [1.0]]
        )
        exp_range = np.log(all_vals[np.isfinite(all_vals) & (all_vals > 0)])
        ticks_log = np.arange(np.floor(exp_range.min()), np.ceil(exp_range.max()) + 1, 0.5)
        ticks_hr = np.exp(ticks_log)
        axis_values = ticks_log.tolist()
        axis_labels = [f"{v:.2g}" for v in ticks_hr]
    else:
        display["est_display"] = display["estimate"]
        display["ci_lower_display"] = display["ci_lower"]
        display["ci_upper_display"] = display["ci_upper"]
        ref_display = vline_x
        axis_values = None
        axis_labels = None

    # Format text columns for tooltip
    display["hr_ci"] = [
        f"{e:.2f} ({l:.2f}\u2013{u:.2f})"
        for e, l, u in zip(display["estimate"], display["ci_lower"], display["ci_upper"])
    ]
    if has_pvalue:
        display["p_fmt"] = [_fmt_pvalue(p) for p in display["p_value"]]

    # Reverse row order so first term appears at top of chart
    display = display.iloc[::-1].reset_index(drop=True)

    tooltip_fields = ["term:N", "hr_ci:N"]
    if has_pvalue:
        tooltip_fields.append("p_fmt:N")

    x_axis: alt.Axis
    if axis_values is not None and axis_labels is not None:
        x_axis = alt.Axis(
            values=axis_values,
            labelExpr="{"
            + ", ".join(f'"{v:.3f}": "{l}"' for v, l in zip(axis_values, axis_labels))
            + "}[format(datum.value, '.3f')]",
        )
    else:
        x_axis = alt.Axis()

    base = alt.Chart(display)

    ci_bars = (
        base.mark_rule(strokeWidth=1.5)
        .encode(
            y=alt.Y("term:N", sort=None, axis=alt.Axis(labelAngle=0), title=""),
            x=alt.X("ci_lower_display:Q", title=x_label, axis=x_axis),
            x2="ci_upper_display:Q",
            tooltip=tooltip_fields,
        )
    )

    points = (
        base.mark_point(size=80, filled=True)
        .encode(
            y=alt.Y("term:N", sort=None),
            x=alt.X("est_display:Q"),
            color=alt.value("#20558A"),
            tooltip=tooltip_fields,
        )
    )

    ref_df = to_dataframe({"ref": [ref_display]})
    ref_line = (
        alt.Chart(ref_df)
        .mark_rule(color="#888888", strokeDash=[4, 4], opacity=0.7)
        .encode(x="ref:Q")
    )

    props: dict[str, Any] = {"width": width, "height": height}
    if title:
        props["title"] = title

    return (ci_bars + points + ref_line).properties(**props)


def _plot_forest_plotnine(
    df: Any,
    *,
    use_log: bool,
    vline_x: float,
    x_label: str,
    title: str | None,
) -> Any:
    """plotnine implementation of plot_forest."""
    try:
        import plotnine as p9
    except ImportError as exc:
        raise ImportError(
            "plot_forest() with backend='plotnine' requires plotnine. "
            "Install with `pip install greenwood[plotnine]`."
        ) from exc

    import pandas as pd

    # Reverse order so first term is at top
    plot_df = df.iloc[::-1].reset_index(drop=True)
    # Make term an ordered categorical to lock the y-axis order
    plot_df["term"] = pd.Categorical(plot_df["term"], categories=plot_df["term"].tolist(), ordered=True)

    plot = (
        p9.ggplot(plot_df, p9.aes(y="term", x="estimate"))
        + p9.geom_errorbarh(
            p9.aes(xmin="ci_lower", xmax="ci_upper"),
            height=0.25,
            color="#555555",
        )
        + p9.geom_point(size=3, color="#20558A")
        + p9.geom_vline(xintercept=vline_x, linetype="dashed", color="#888888", alpha=0.8)
        + p9.labs(x=x_label, y="", title=title or "")
        + theme_forest()
    )

    if use_log:
        plot = plot + p9.scale_x_log10()

    return plot

