"""Numbers-at-risk tables with Great Tables.

Publication-ready risk tables formatted with Great Tables (GT), supporting
interactive display and export to HTML and other formats.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ._shared import get_risk_table_frame

if TYPE_CHECKING:
    from .._nonparametric import KaplanMeier

__all__ = ["risk_table"]


def risk_table(km: KaplanMeier, times: Any = None) -> Any:
    """Return the numbers-at-risk table as a Great Tables object.

    Creates a publication-ready table showing the number of subjects at risk at specified
    time points, stratified by group (if fitted with strata). The table is formatted with
    Great Tables for interactive viewing and export to various formats.

    Parameters
    ----------
    km
        A fitted `KaplanMeier` object.
    times
        Query times for the numbers-at-risk table. Defaults to six evenly spaced, rounded
        times from 0 to the maximum observed follow-up time.

    Returns
    -------
    great_tables.GT
        A Great Tables object, which can be displayed in notebooks or exported.

    Examples
    --------
    Fit a stratified Kaplan-Meier estimator and produce a publication-ready risk table
    using Great Tables:

    ```{python}
    import greenwood as gw

    # Load data and fit a stratified Kaplan-Meier estimator
    lung = gw.load_dataset("lung", backend="polars")
    y = gw.Surv.right(lung["time"], event=(lung["status"] == 2))
    km = gw.KaplanMeier().fit(y, by=lung["sex"])

    # Produce a publication-ready numbers-at-risk table
    gw.risk_table(km, times=[0, 250, 500, 750, 1000])
    ```
    """
    try:
        import great_tables as gt  # pyright: ignore[reportMissingImports]
    except ImportError as exc:  # pragma: no cover
        raise ImportError(  # pragma: no cover
            "Great Tables required for risk_table. Install it with `pip install great-tables`."
        ) from exc

    frame = get_risk_table_frame(km, times=times, format="polars")

    # Use Narwhals for backend-agnostic pivot operation (works with Polars, pandas, PyArrow)
    import narwhals as nw

    df_nw = nw.from_native(frame)
    pivot_df = df_nw.pivot(index="strata", on="time", values="n_risk").to_native()

    # Create GT object
    gt_table = gt.GT(pivot_df)

    # Format the title
    gt_table = gt_table.tab_header(
        title="Numbers at Risk",
        subtitle="Count of subjects at risk at each time point",
    )

    # Format as integers (no decimals): exclude the strata column
    numeric_columns = [col for col in pivot_df.columns if col != "strata"]
    gt_table = gt_table.fmt_integer(columns=numeric_columns)

    return gt_table
