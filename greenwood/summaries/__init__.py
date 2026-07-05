"""Greenwood's tidy / `broom` layer.

Standardized `tidy`/`glance`/`augment` views over fitted estimators, broom-compatible and
aligned with `great_summaries.tidy` so the two ecosystems share one contract.
"""

from __future__ import annotations

from ._registry import (
    augment,
    glance,
    register_augment,
    register_glance,
    register_tidier,
    tidy,
)

__all__ = [
    "tidy",
    "glance",
    "augment",
    "register_tidier",
    "register_glance",
    "register_augment",
]
