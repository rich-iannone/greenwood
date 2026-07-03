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
    """Assert an array of Greenwood values matches R to tolerance."""
    a = np.asarray(actual, dtype=float)
    e = np.asarray(expected, dtype=float)
    if a.shape != e.shape:
        raise AssertionError(f"{what}: shape {a.shape} != R shape {e.shape}.")
    if not np.allclose(a, e, rtol=rtol, atol=atol):
        diff = np.max(np.abs(a - e)) if a.size else float("nan")
        raise AssertionError(f"{what}: does not match R (max abs diff {diff}).")
