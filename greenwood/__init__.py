"""Greenwood: modern survival analysis for Python.

Narwhals-native, validated against R's `survival`, visualized with Altair (or plotnine), and
a first-class citizen of the Great Tables ecosystem.

This release provides the `Surv` response object, the risk-set / event-table kernel, the
non-parametric estimators (`KaplanMeier`, `NelsonAalen`), group comparison tests
(`logrank_test`), and interactive visualization (`plot_survival` with aligned risk tables).
Regression and parametric models arrive in later releases (see `ROADMAP.md`).
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

from . import data, summaries, viz
from ._competing import AalenJohansen, FineGray, MultiState
from ._core import EventTable, event_table
from ._cox import CoxPH, ZPHResult
from ._flexible import RoystonParmar
from ._metrics import brier_score, calibration, concordance_index, integrated_brier_score
from ._nonparametric import KaplanMeier, NelsonAalen
from ._parametric import AFT
from ._penalized import CoxNet
from ._power import logrank_n_events, logrank_power, logrank_sample_size
from ._resample import cross_validate
from ._rmst import RMSTResult, pairwise_rmst_test, rmst_diff, rmst_test
from ._surv import CensoringType, Surv
from ._tests import TestResult, logrank_test, pairwise_logrank_test, trend_test
from .data import available_datasets, load_dataset
from .summaries import augment, glance, tidy
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
    "CoxNet",
    "RoystonParmar",
    "ZPHResult",
    "AFT",
    "AalenJohansen",
    "FineGray",
    "MultiState",
    "calibration",
    "concordance_index",
    "cross_validate",
    "logrank_n_events",
    "logrank_power",
    "logrank_sample_size",
    "brier_score",
    "integrated_brier_score",
    "logrank_test",
    "pairwise_logrank_test",
    "trend_test",
    "TestResult",
    "rmst_test",
    "rmst_diff",
    "pairwise_rmst_test",
    "RMSTResult",
    "plot_survival",
    "risk_table",
    "load_dataset",
    "available_datasets",
    "tidy",
    "glance",
    "augment",
    "data",
    "summaries",
    "viz",
]
