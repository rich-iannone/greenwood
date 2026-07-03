"""Helpers for validating Greenwood numerics against R fixtures.

Correctness against R's `survival` is non-negotiable, so statistics are checked to
tolerance against values exported by `scripts/regenerate_r_fixtures.R` and checked in under
`tests/fixtures/r/`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "r"

DEFAULT_RTOL = 1e-9
DEFAULT_ATOL = 1e-9


def load_fixture(name: str) -> dict[str, Any]:
    """Load an R-parity fixture by base name (without the `.json` suffix)."""
    path = FIXTURE_DIR / f"{name}.json"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing R fixture {path}. Regenerate with `Rscript scripts/regenerate_r_fixtures.R`."
        )
    return json.loads(path.read_text())


def assert_allclose_to_r(
    actual: Any,
    expected: Any,
    *,
    rtol: float = DEFAULT_RTOL,
    atol: float = DEFAULT_ATOL,
    what: str = "value",
) -> None:
    """Assert an array of Greenwood values matches R to tolerance.

    R's non-finite tokens (`NA`, `NaN`, `Inf`, `-Inf`, JSON `null`) are coerced to their
    NumPy equivalents, and NaN positions must line up on both sides.
    """
    a = np.atleast_1d(np.asarray(actual, dtype=float))
    e = _to_float_array(expected)
    if a.shape != e.shape:
        raise AssertionError(f"{what}: shape {a.shape} != R shape {e.shape}.")
    if not np.array_equal(np.isnan(a), np.isnan(e)):
        raise AssertionError(f"{what}: NaN positions differ from R.")
    if not np.allclose(a, e, rtol=rtol, atol=atol, equal_nan=True):
        finite = ~np.isnan(a)
        diff = np.max(np.abs(a[finite] - e[finite])) if finite.any() else float("nan")
        raise AssertionError(f"{what}: does not match R (max abs diff {diff}).")


def _to_float_array(values: Any) -> Any:
    """Coerce an R-exported list to floats, mapping NA/NaN/Inf tokens and null."""

    def one(v: Any) -> float:
        if v is None:
            return float("nan")
        if isinstance(v, str):
            token = v.strip()
            if token in ("NA", "NaN", "nan", "null", ""):
                return float("nan")
            if token in ("Inf", "inf", "Infinity"):
                return float("inf")
            if token in ("-Inf", "-inf", "-Infinity"):
                return float("-inf")
            return float(token)
        return float(v)

    return np.array([one(v) for v in np.atleast_1d(values)], dtype=float)
