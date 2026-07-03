"""plotnine-based visualization for Greenwood.

Survival curves (with confidence ribbons and censoring marks) and aligned numbers-at-risk
tables, returned as composable plotnine objects. plotnine is an optional dependency (the
`viz` extra); the functions import it lazily, so importing `greenwood` never requires it.
"""

from __future__ import annotations

from ._curves import (
    plot_survival,
    risk_table,
    risk_table_data,
    theme_survival,
)

__all__ = [
    "plot_survival",
    "risk_table",
    "risk_table_data",
    "theme_survival",
]
