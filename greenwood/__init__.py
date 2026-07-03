"""Greenwood: modern survival analysis for Python.

narwhals-native, validated against R's `survival`, visualized with plotnine, and a
first-class citizen of the Great Tables ecosystem.

This foundational release exposes the `Surv` response object and the risk-set /
event-table kernel that Kaplan-Meier, the log-rank test, and Cox will build on. The
estimators themselves arrive in later releases (see `ROADMAP.md`).
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

from . import data, tidy, viz
from ._core import EventTable, event_table
from ._surv import CensoringType, Surv

try:
    __version__ = version("greenwood")
except PackageNotFoundError:  # pragma: no cover - source tree without metadata
    __version__ = "0.0.0.dev0"

__all__ = [
    "__version__",
    "Surv",
    "CensoringType",
    "EventTable",
    "event_table",
    "data",
    "tidy",
    "viz",
]
