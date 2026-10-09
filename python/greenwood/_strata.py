"""Curve labels for estimators that fit one curve per group.

Every such estimator follows `spec/results.md`: its tables always start with a `strata` column, a
single curve is labeled `"all"`, and grouped curves are labeled with `name=value` pairs joined by
`", "` (`"sex=1"`, `"sex=1, ph.ecog=0"`). The same labels appear in tables, printed output, and plot
legends, so code that consumes results never branches on the number of curves.
"""

from __future__ import annotations

import math
import numbers
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    from ._outcome import BoundInputs

__all__ = ["ALL", "format_value", "resolve_strata", "strata_labels"]

ALL = "all"

Array = npt.NDArray[Any]


def format_value(value: Any) -> str:
    """Write one grouping value the way R prints it, so labels match across languages."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "NA"
    if isinstance(value, (bool, np.bool_)):
        return "TRUE" if value else "FALSE"
    if isinstance(value, numbers.Real) and not isinstance(value, numbers.Integral):
        as_float = float(value)
        if as_float.is_integer():
            return str(int(as_float))
        return repr(as_float)
    if isinstance(value, numbers.Integral):
        return str(int(value))
    return str(value)


def strata_labels(values: Any, name: str) -> Array:
    """Label each row `name=value`."""
    from ._ingest import to_1d_array

    raw = to_1d_array(values, dtype=object)
    return np.array([f"{name}={format_value(v)}" for v in raw.tolist()], dtype=object)


def _series_name(value: Any) -> str | None:
    """The name of a pandas or Polars Series (or similar), if it has one."""
    name = getattr(value, "name", None)
    return name if isinstance(name, str) and name else None


def resolve_strata(bound: BoundInputs, key: str) -> tuple[Array, bool]:
    """Curve labels for one row each, and whether the fit is grouped.

    The name in each label is the column name (from the formula or a column-name argument), else
    the name of a Series passed as the argument, else the argument's own name (for example `by`).
    Labels combined from several columns arrive already formatted and pass through unchanged.
    """
    values = bound.labels.get(key)
    if values is None:
        return np.full(bound.surv.n, ALL, dtype=object), False
    names = bound.label_names.get(key)
    if names is not None and len(names) > 1:
        from ._ingest import to_1d_array

        labels = to_1d_array(values, dtype=object)
    else:
        name = names[0] if names else (_series_name(values) or key)
        labels = strata_labels(values, name)
    if labels.shape[0] != bound.surv.n:
        raise ValueError(f"`{key}` must have the same length as the response.")
    return labels, True
