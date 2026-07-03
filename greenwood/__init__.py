"""Greenwood: modern survival analysis for Python.

narwhals-native, validated against R's `survival`, visualized with plotnine, and a
first-class citizen of the Great Tables ecosystem.

This release provides the `Surv` response object, the risk-set / event-table kernel, the
non-parametric estimators (`KaplanMeier`, `NelsonAalen`), group comparison tests
(`logrank_test`), and plotnine visualization (`plot_survival` with aligned risk tables).
Regression and parametric models arrive in later releases (see `ROADMAP.md`).
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

from . import data, tidy, viz
from ._competing import AalenJohansen, FineGray, MultiState
from ._core import EventTable, event_table
from ._cox import CoxPH, ZPHResult
from ._nonparametric import KaplanMeier, NelsonAalen
from ._parametric import AFT
from ._surv import CensoringType, Surv
from ._tests import TestResult, logrank_test
from .viz import plot_survival, risk_table

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
    "CoxPH",
    "ZPHResult",
    "AFT",
    "AalenJohansen",
    "FineGray",
    "MultiState",
    "logrank_test",
    "TestResult",
    "plot_survival",
    "risk_table",
    "data",
    "tidy",
    "viz",
]
