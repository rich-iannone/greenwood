"""Greenwood: modern survival analysis for Python.

Narwhals-native, validated against R's `survival`, visualized with Altair and Great Tables.

This release provides the `Surv` response object, the risk-set / event-table kernel, the
non-parametric estimators (`KaplanMeier`, `NelsonAalen`), group comparison tests
(`logrank_test()`), and interactive visualization (`plot_survival()` with aligned risk
tables).
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

from . import data, summaries, viz
from ._competing import AalenJohansen, FineGray, MultiState
from ._core import EventTable, event_table
from ._cox import CoxPH, ZPHResult
from ._flexible import RoystonParmar
from ._metrics import (
    brier_score,
    calibration,
    concordance_index,
    integrated_auc,
    integrated_brier_score,
    time_dependent_auc,
)
from ._nonparametric import KaplanMeier, NelsonAalen
from ._parametric import AFT
from ._penalized import CoxNet, CoxNetCVResult, cv_coxnet
from ._power import logrank_n_events, logrank_power, logrank_sample_size
from ._resample import cross_validate
from ._rmst import RMSTResult, pairwise_rmst_test, rmst_diff, rmst_test
from ._surv import CensoringType, Surv
from ._tests import TestResult, logrank_test, pairwise_logrank_test, trend_test
from ._univariate import Parametric, compare_distributions
from .data import available_datasets, load_dataset
from .summaries import augment, glance, tidy
from .viz import (
    get_risk_table_frame,
    plot_cif,
    plot_forest,
    plot_survival,
    risk_table,
    theme_forest,
)

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
    "Parametric",
    "CoxPH",
    "CoxNet",
    "CoxNetCVResult",
    "cv_coxnet",
    "RoystonParmar",
    "ZPHResult",
    "AFT",
    "AalenJohansen",
    "FineGray",
    "MultiState",
    "calibration",
    "compare_distributions",
    "concordance_index",
    "cross_validate",
    "logrank_n_events",
    "logrank_power",
    "logrank_sample_size",
    "brier_score",
    "integrated_auc",
    "integrated_brier_score",
    "time_dependent_auc",
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
    "get_risk_table_frame",
    "plot_forest",
    "theme_forest",
    "plot_cif",
    "load_dataset",
    "available_datasets",
    "tidy",
    "glance",
    "augment",
    "data",
    "summaries",
    "viz",
]
