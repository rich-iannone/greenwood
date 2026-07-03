"""Greenwood: modern survival analysis for Python.

narwhals-native, validated against R's `survival`, visualized with plotnine, and a
first-class citizen of the Great Tables ecosystem.

This release provides the `Surv` response object, the risk-set / event-table kernel, and
the non-parametric estimators (`KaplanMeier`, `NelsonAalen`). Regression models, group
tests, and visualization arrive in later releases (see `ROADMAP.md`).
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

from . import data, tidy, viz
from ._core import EventTable, event_table
from ._nonparametric import KaplanMeier, NelsonAalen
from ._surv import CensoringType, Surv
from ._tests import TestResult, logrank_test

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
    "KaplanMeier",
    "NelsonAalen",
    "logrank_test",
    "TestResult",
    "data",
    "tidy",
    "viz",
]
