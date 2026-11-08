"""Curve labels for estimators that fit one curve per group.

Every such estimator follows `spec/results.md`: its tables always start with a `strata` column, a
single curve is labeled `"all"`, and grouped curves are labeled with `name=value` pairs joined by
`", "` (`"sex=1"`, `"sex=1, ph.ecog=0"`). The same labels appear in tables, printed output, and plot
legends, so code that consumes results never branches on the number of curves.

Groups come in the order of their values: a categorical column's declared levels (a pandas
`Categorical` or a Polars `Enum`), and otherwise sorted values, with numbers in numeric order and
text in code-point order. With several grouping columns the first varies slowest.
"""

from __future__ import annotations

import math
import numbers
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import narwhals as nw  # pyright: ignore[reportMissingImports]
import narwhals.stable.v2 as nw_v2  # pyright: ignore[reportMissingImports]
import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    from ._outcome import BoundInputs

__all__ = [
    "ALL",
    "OrderedGroups",
    "category_levels",
    "format_value",
    "group_levels",
    "group_order",
    "resolve_strata",
    "strata_labels",
]

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


def category_levels(values: Any) -> list[Any] | None:
    """The declared levels of a categorical column, or `None` when its values should be sorted.

    A pandas `Categorical` (bare or in a Series) and a Polars `Enum` declare their order, the one
    their own libraries sort by. A Polars `Categorical` and an Arrow dictionary sort by value, so
    their values are sorted here too.
    """
    if hasattr(values, "categories") and hasattr(values, "codes"):
        return list(values.categories)
    try:
        series: Any = nw.from_native(values, series_only=True)
    except TypeError:
        return None
    if series.dtype == nw.Enum or (
        series.dtype == nw.Categorical and nw.dependencies.is_pandas_like_series(series.to_native())
    ):
        # The stable API's get_categories() returns the declared levels, not just those present.
        stable = nw_v2.from_native(series.to_native(), series_only=True)
        return list(stable.cat.get_categories().to_list())
    return None


def _sort_key(value: Any) -> tuple[int, Any]:
    """Numbers (and logicals) in numeric order, then text in code-point order, missing last."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return (3, 0)
    if isinstance(value, (numbers.Real, np.bool_)):
        return (0, float(value))
    if isinstance(value, str):
        return (1, value)
    return (2, str(value))


def group_order(values: Array, declared: Sequence[Any] | None = None) -> list[Any]:
    """The distinct values of `values` in group order.

    Declared levels come first, in their order, keeping only the levels that occur. Any other value
    follows in sorted order.
    """
    present: dict[Any, None] = dict.fromkeys(values.tolist())
    ordered = [level for level in declared if level in present] if declared is not None else []
    seen = set(ordered)
    ordered += sorted((v for v in present if v not in seen), key=_sort_key)
    return ordered


class OrderedGroups:
    """Group values that carry their group order, so a refit on some of the rows keeps it.

    It reads as a categorical (`categories` and `codes`), the way `category_levels()` recognizes a
    pandas `Categorical`, and as an array of the values everywhere else.
    """

    def __init__(self, values: Array, levels: Sequence[Any]) -> None:
        self.values = values
        self.categories = list(levels)

    @property
    def codes(self) -> list[int]:
        index = {level: i for i, level in enumerate(self.categories)}
        return [index.get(v, -1) for v in self.values.tolist()]

    def __array__(self, dtype: Any = None, copy: bool | None = None) -> Array:
        return np.asarray(self.values, dtype=dtype)

    def __len__(self) -> int:
        return int(self.values.shape[0])

    def __getitem__(self, index: Any) -> OrderedGroups:
        return OrderedGroups(self.values[index], self.categories)


def group_levels(bound: BoundInputs, key: str) -> list[Any]:
    """The distinct values of the label argument `key`, in group order."""
    from ._ingest import to_1d_array

    values = bound.labels[key]
    declared = bound.label_levels.get(key)
    if declared is None:
        declared = category_levels(values)
    return group_order(to_1d_array(values, dtype=object), declared)


def resolve_strata(bound: BoundInputs, key: str) -> tuple[Array, list[str], bool]:
    """Curve labels for one row each, the labels in curve order, and whether the fit is grouped.

    The name in each label is the column name (from the formula or a column-name argument), else
    the name of a Series passed as the argument, else the argument's own name (for example `by`).
    Labels combined from several columns arrive already formatted and pass through unchanged.
    """
    values = bound.labels.get(key)
    if values is None:
        return np.full(bound.surv.n, ALL, dtype=object), [ALL], False
    names = bound.label_names.get(key)
    order = group_levels(bound, key)
    if names is not None and len(names) > 1:
        from ._ingest import to_1d_array

        labels = to_1d_array(values, dtype=object)
        levels = [str(level) for level in order]
    else:
        name = names[0] if names else key
        labels = strata_labels(values, name)
        levels = list(dict.fromkeys(f"{name}={format_value(v)}" for v in order))
    if labels.shape[0] != bound.surv.n:
        raise ValueError(f"`{key}` must have the same length as the response.")
    return labels, levels, True
