"""Backend-agnostic helpers shared by the Altair and plotnine visualization backends.

These build plain arrays and dicts of columns from a fitted `KaplanMeier` (the numbers a
survival plot is drawn from) without importing any plotting library. `get_risk_table_frame()`
is the one public function here; it is re-exported at the top level as
`greenwood.get_risk_table_frame`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt

from .._backends import to_dataframe

if TYPE_CHECKING:
    from .._nonparametric import KaplanMeier

__all__ = ["get_risk_table_frame"]

Array = npt.NDArray[Any]

_OVERALL = "Overall"


def _strata_label(block: Any) -> str:
    return _OVERALL if block.label is None else str(block.label)


def _default_times(km: KaplanMeier) -> list[float]:
    """Six evenly spaced, rounded times from 0 to the largest observed time."""
    max_t = max(float(b.time[-1]) for b in km._blocks if b.time.size)
    raw = np.linspace(0.0, max_t, 6)
    return sorted({round(float(t)) for t in raw})


def _n_at_risk(block: Any, times: Array) -> Array:
    """Number at risk at each query time (as in R's `summary(fit, times=)$n.risk`)."""
    idx = np.searchsorted(block.time, times, side="left")
    out = np.zeros(times.shape[0])
    valid = idx < block.time.shape[0]
    out[valid] = block.n_risk[idx[valid]]
    return out


def _risk_table_columns(km: KaplanMeier, times: Any = None) -> dict[str, list[Any]]:
    """The numbers-at-risk table as a plain dict of columns (backend-agnostic)."""
    query = np.asarray(_default_times(km) if times is None else times, dtype=float)
    strata: list[str] = []
    time_col: list[float] = []
    n_risk: list[float] = []
    for block in km._blocks:
        counts = _n_at_risk(block, query)
        for t, n in zip(query, counts, strict=True):
            strata.append(_strata_label(block))
            time_col.append(float(t))
            n_risk.append(float(n))
    return {"strata": strata, "time": time_col, "n_risk": n_risk}


def get_risk_table_frame(km: KaplanMeier, times: Any = None, *, format: str | None = None) -> Any:
    """Return a tidy frame of the number at risk per stratum at each of `times`.

    Parameters
    ----------
    km
        A fitted `KaplanMeier` estimator.
    times
        Query times for the numbers-at-risk table. Defaults to an automatic grid.
    format
        Output format: `None` (default), `"pandas"`, `"polars"`, or `"pyarrow"`. When
        `None`, a backend is auto-detected (Polars, then Pandas, then PyArrow).

    Returns
    -------
    DataFrame
        A tidy frame with columns `strata`, `time`, and `n_risk` (one row per stratum per
        time point).

    Examples
    --------
    Fit a stratified Kaplan-Meier estimator on the bundled `lung` dataset, then tabulate the
    number of subjects still at risk in each group at a chosen set of times. This returns the
    numbers as a tidy frame (one row per stratum and time).

    ```{python}
    import greenwood as gw

    lung = gw.load_dataset("lung", backend="polars")
    y = gw.Surv.right(lung["time"], event=(lung["status"] == 2))
    km = gw.KaplanMeier().fit(y, by=lung["sex"])

    gw.get_risk_table_frame(km, times=[0, 250, 500, 750, 1000], format="polars")
    ```
    """
    return to_dataframe(_risk_table_columns(km, times), format=format)
