"""Visualization for Greenwood.

Survival curves (with confidence ribbons and censoring marks) and aligned numbers-at-risk
tables. `plot_survival` and `risk_table` here are the default backend — interactive,
Narwhals-native Altair charts that pull in no Pandas or Matplotlib. Two backends are available
explicitly, each imported lazily so importing `greenwood` never requires either:

- `greenwood.viz.altair` (the `altair` extra) — the default; returns interactive Altair charts.
- `greenwood.viz.plotnine` (the `plotnine` extra) — returns composable plotnine objects for a
  grammar-of-graphics workflow.

`risk_table_data` returns the numbers at risk as a tidy DataFrame and is backend-neutral.
"""

from __future__ import annotations

from . import _altair as altair
from . import _curves as plotnine
from ._altair import plot_survival, risk_table
from ._shared import risk_table_data

__all__ = [
    "altair",
    "plotnine",
    "plot_survival",
    "risk_table",
    "risk_table_data",
]
