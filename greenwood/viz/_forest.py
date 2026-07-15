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

__all__ = ["forest_plot"]

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


def forest_plot(
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
        Location of the reference line (e.g., 1 for HR on log scale, 0 for difference).
        If `None`, defaults to 0 (which is log(1) for log scale).
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

    ```{python}
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

    ```{python}
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


