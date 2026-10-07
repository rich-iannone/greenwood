"""Helpers for replaying the language-neutral conformance fixtures under `spec/conformance/`.

The fixtures are generated in R from the reference implementation (for `event_time`, the pinned
etd package) by scripts such as `scripts/generate_event_time_conformance.R`. Each case records an
R input, tagged with its R type, and what the reference implementation did with it. This module
turns those R inputs into their Python equivalents and normalizes R's missing-value tokens, so
test files can compare Greenwood's behaviour case by case.

Locations in the fixtures are 1-based, exactly as R reports them. `to_python_locations()`
converts them to Python's 0-based indexing.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

# `spec/` lives at the repo root, shared with the R package: python/tests -> repo root.
SPEC_DIR = Path(__file__).resolve().parents[2] / "spec"
CONFORMANCE_DIR = SPEC_DIR / "conformance"

# R's missing and non-finite tokens as written by jsonlite (`na = "string"`).
_FLOAT_TOKENS = {
    "NA": math.nan,
    "NaN": math.nan,
    "Inf": math.inf,
    "-Inf": -math.inf,
}


def load_conformance(area: str, name: str) -> Any:
    """Load one conformance fixture file, such as `load_conformance("event_time", "format")`."""
    path = CONFORMANCE_DIR / area / f"{name}.json"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing conformance fixture {path}. Regenerate with "
            f"`Rscript scripts/generate_{area}_conformance.R`."
        )
    return json.loads(path.read_text())


def r_float(value: Any) -> float:
    """Convert one R numeric value (a number or a jsonlite token) to a Python float."""
    if value is None:
        return math.nan
    if isinstance(value, str):
        return _FLOAT_TOKENS[value]
    return float(value)


def r_floats(values: list[Any]) -> list[float]:
    """Convert an R numeric vector to a list of Python floats, with NA as NaN."""
    return [r_float(v) for v in values]


def r_int_or_none(value: Any) -> int | None:
    """Convert one R integer value to a Python int, with NA as None."""
    if value is None or value == "NA":
        return None
    return int(value)


def decode_input(encoded: dict[str, Any] | None) -> Any:
    """Build the Python value that corresponds to an R input recorded by a generator.

    | R type      | Python value                                   |
    |-------------|------------------------------------------------|
    | `NULL`      | `None`                                         |
    | double      | `list[float]`, NA and NaN become `nan`         |
    | integer     | `list[int | None]`                             |
    | logical     | `list[bool | None]`                            |
    | character   | `list[str | None]`                             |
    | list        | `list` of the decoded elements                 |
    | factor      | pandas `Series` with a categorical dtype       |

    Greenwood treats categorical input the way etd treats R factors (spec/event_time.md).
    """
    if encoded is None:
        return None
    r_type = encoded["r_type"]
    values = encoded["values"]
    if r_type == "double":
        return r_floats(values)
    if r_type == "integer":
        return [r_int_or_none(v) for v in values]
    if r_type in ("logical", "character"):
        return list(values)
    if r_type == "list":
        return [decode_input(v) for v in values]
    if r_type == "factor":
        import pandas as pd

        return pd.Series(pd.Categorical(values, categories=encoded["levels"]))
    raise ValueError(f"Unknown R input type {r_type!r}.")


def to_python_locations(locations: list[int] | None) -> list[int]:
    """Convert R's 1-based locations to Python's 0-based indexing."""
    return [] if locations is None else [loc - 1 for loc in locations]


def floats_equal(actual: list[float | None], expected: list[float]) -> bool:
    """Compare float lists exactly, treating NaN (and None, a frame null) as equal to NaN."""
    if len(actual) != len(expected):
        return False
    for a, e in zip(actual, expected, strict=True):
        a_missing = a is None or math.isnan(a)
        if a_missing or math.isnan(e):
            if not (a_missing and math.isnan(e)):
                return False
        elif a != e:
            return False
    return True
