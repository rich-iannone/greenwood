"""Formatting helpers for the R-style `__repr__` of the estimators.

These build compact, aligned, printout-style summaries modeled on R's `print` methods
for `coxph`, `survreg`, and `survfit`. Everything here is deterministic (no memory
addresses, no locale-dependent formatting) so reprs are stable across runs and builds.
"""

from __future__ import annotations

import math
from typing import Any


def num(x: Any, digits: int = 4) -> str:
    """Format a number with `digits` significant figures, or `"NA"` for missing values.

    Used throughout the estimator `__repr__` methods to produce compact, stable numeric
    output. `None` and `NaN` are both rendered as `"NA"`.

    Parameters
    ----------
    x
        A numeric value (or `None`).
    digits
        Number of significant figures (default `4`).

    Returns
    -------
    str
        The formatted string.

    Examples
    --------
    ```{python}
    from greenwood._repr import num

    num(0.001234567)
    ```

    ```{python}
    num(None)
    ```
    """
    if x is None:
        return "NA"
    xf = float(x)
    if math.isnan(xf):
        return "NA"
    return f"{xf:.{digits}g}"


def fixed(x: Any, digits: int = 3) -> str:
    """Format a number with a fixed number of decimal places, or `"NA"` for missing values.

    Unlike `num()`, which uses significant figures, this always shows exactly `digits`
    decimal places. Useful for p-values and other quantities where a fixed format is
    conventional.

    Parameters
    ----------
    x
        A numeric value (or `None`).
    digits
        Number of decimal places (default `3`).

    Returns
    -------
    str
        The formatted string.

    Examples
    --------
    ```{python}
    from greenwood._repr import fixed

    fixed(0.04217)
    ```

    ```{python}
    fixed(float("nan"))
    ```
    """
    if x is None:
        return "NA"
    xf = float(x)
    if math.isnan(xf):
        return "NA"
    return f"{xf:.{digits}f}"


def whole(x: Any) -> str:
    """Format a value that is conceptually an integer (times, counts), or `"NA"`.

    If the value is a whole number (e.g., `365.0`), the decimal is dropped and it is
    rendered as `"365"`. Non-integer floats fall back to `num()` formatting.

    Parameters
    ----------
    x
        A numeric value (or `None`).

    Returns
    -------
    str
        The formatted string.

    Examples
    --------
    ```{python}
    from greenwood._repr import whole

    whole(365.0)
    ```

    ```{python}
    whole(3.14)
    ```
    """
    if x is None:
        return "NA"
    xf = float(x)
    if math.isnan(xf):
        return "NA"
    if xf.is_integer():
        return str(int(xf))
    return num(xf)


def align_table(
    headers: list[str], rows: list[list[str]], row_labels: list[str] | None = None
) -> str:
    """Render a right-aligned numeric table with an optional left label column.

    Parameters
    ----------
    headers
        Column headers for the data columns.
    rows
        One list of preformatted string cells per row, each the length of `headers`.
    row_labels
        Optional left-hand labels (term or stratum names), left-justified.
    """
    widths = [len(h) for h in headers]
    for row in rows:
        for j, cell in enumerate(row):
            widths[j] = max(widths[j], len(cell))
    label_w = max((len(x) for x in (row_labels or [])), default=0)

    lines = []
    head = " " * label_w
    for j, header in enumerate(headers):
        head += "  " + header.rjust(widths[j])
    lines.append(head)
    for i, row in enumerate(rows):
        line = row_labels[i].ljust(label_w) if row_labels else ""
        for j, cell in enumerate(row):
            line += "  " + cell.rjust(widths[j])
        lines.append(line)
    return "\n".join(lines)
