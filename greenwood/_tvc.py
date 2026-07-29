"""Time-varying covariate utilities: episode splitting."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np
import numpy.typing as npt

from ._backends import to_dataframe

__all__ = ["split_episodes"]

Array = npt.NDArray[Any]


def split_episodes(
    baseline: Any,
    visits: Any,
    *,
    id: str,
    time: str,
    event: str,
    visit_time: str,
    carry_forward: bool = True,
    format: str | None = None,
) -> Any:
    r"""Convert repeated-measurement data into counting-process (episode-split) format.

    Takes a subject-level baseline table and a long-format visits table, and merges them
    into the interval-per-row counting-process layout required by `Surv.counting()` and
    `CoxPH`. Each row in the output represents a constant-covariate interval `(tstart,
    tstop]` for one subject.

    The covariate value measured at visit time `v` applies to the interval `[v, next_v)`.
    The event indicator is 1 only on the final interval for subjects who experienced the
    event; all earlier intervals carry 0.

    Parameters
    ----------
    baseline
        One row per subject. Must contain the `id`, `time`, and `event` columns. Any
        additional columns are treated as time-fixed covariates and carried through to every
        output row for that subject.
    visits
        One row per (subject, visit). Must contain the `id` and `visit_time` columns. Every
        other column is treated as a time-varying covariate.
    id
        Column name linking `baseline` and `visits`.
    time
        Column in `baseline` giving the end of follow-up (right-censored exit time).
    event
        Column in `baseline` giving the event indicator.
    visit_time
        Column in `visits` giving the measurement time for each row.
    carry_forward
        If `True` (default), the last observed covariate value is carried forward to the
        end of follow-up (LOCF), producing a final interval `(last_visit, time]`. If
        `False`, follow-up is truncated at the last visit; the trailing interval is dropped
        and the subject is effectively administratively censored there.
    format
        Output format: `None` (default), `"pandas"`, `"polars"`, or `"pyarrow"`.

        - `None` (default): auto-detect, trying Polars first, then Pandas, then PyArrow.
        - `"pandas"`: return a `pandas.DataFrame`.
        - `"polars"`: return a `polars.DataFrame`.
        - `"pyarrow"`: return a `pyarrow.Table`.

    Returns
    -------
    pandas.DataFrame, polars.DataFrame, or pyarrow.Table
        Counting-process dataset with columns `(id, tstart, tstop, event,
        [time-fixed covariates], [time-varying covariates])`. Ready to pass directly to
        `Surv.counting(tstart, tstop, event)` and `CoxPH.fit()`.

    Raises
    ------
    ValueError
        If required columns are missing, if any visit time exceeds the subject's follow-up
        end, or if the resulting dataset is empty.

    Examples
    --------
    Build a minimal TVC dataset and fit a Cox model:

    ```{python}
    import pandas as pd
    import greenwood as gw

    baseline = pd.DataFrame({
        "id":    [1, 2, 3],
        "time":  [10.0, 8.0, 12.0],
        "event": [1, 0, 1],
    })
    visits = pd.DataFrame({
        "id":       [1, 1, 2, 3, 3, 3],
        "day":      [0.0, 5.0, 0.0, 0.0, 4.0, 8.0],
        "bili":     [1.2, 2.4, 0.8, 3.1, 2.9, 4.0],
    })

    long = gw.split_episodes(
        baseline, visits, id="id", time="time", event="event", visit_time="day"
    )
    y = gw.Surv.counting(long["tstart"], long["tstop"], long["event"])
    cox = gw.CoxPH().fit(y, long[["bili"]])
    ```
    """
    try:
        import narwhals as nw  # pyright: ignore[reportMissingImports]
    except ImportError as exc:  # pragma: no cover
        raise ImportError("narwhals is required for split_episodes.") from exc

    base_nw = nw.from_native(baseline, eager_only=True)
    vis_nw = nw.from_native(visits, eager_only=True)

    # --- validate column presence ---
    for col in (id, time, event):
        if col not in base_nw.columns:
            raise ValueError(f"Column {col!r} not found in baseline (columns: {base_nw.columns}).")
    for col in (id, visit_time):
        if col not in vis_nw.columns:
            raise ValueError(f"Column {col!r} not found in visits (columns: {vis_nw.columns}).")

    static_cols = [c for c in base_nw.columns if c not in (id, time, event)]
    tvc_cols = [c for c in vis_nw.columns if c not in (id, visit_time)]

    # --- extract arrays ---
    base_ids: list[Any] = base_nw[id].to_list()
    base_times = np.asarray(base_nw[time].to_list(), dtype=float)
    base_events: list[Any] = base_nw[event].to_list()
    base_static: dict[str, list[Any]] = {col: base_nw[col].to_list() for col in static_cols}

    vis_ids: list[Any] = vis_nw[id].to_list()
    vis_times = np.asarray(vis_nw[visit_time].to_list(), dtype=float)
    vis_tvc: dict[str, list[Any]] = {col: vis_nw[col].to_list() for col in tvc_cols}

    # group visit row indices by subject id
    vis_by_id: dict[Any, list[int]] = defaultdict(list)
    for idx, subj_id in enumerate(vis_ids):
        vis_by_id[subj_id].append(idx)

    # --- build output row by row ---
    out_id: list[Any] = []
    out_tstart: list[float] = []
    out_tstop: list[float] = []
    out_event: list[Any] = []
    out_static: dict[str, list[Any]] = {col: [] for col in static_cols}
    out_tvc: dict[str, list[Any]] = {col: [] for col in tvc_cols}

    def _append(
        subj_id: Any,
        tstart: float,
        tstop: float,
        ev: Any,
        static_vals: dict[str, Any],
        tvc_vals: dict[str, Any],
    ) -> None:
        out_id.append(subj_id)
        out_tstart.append(tstart)
        out_tstop.append(tstop)
        out_event.append(ev)
        for col in static_cols:
            out_static[col].append(static_vals[col])
        for col in tvc_cols:
            out_tvc[col].append(tvc_vals[col])

    for i, subj_id in enumerate(base_ids):
        t_end = float(base_times[i])
        ev = base_events[i]
        static_vals = {col: base_static[col][i] for col in static_cols}

        row_indices = vis_by_id.get(subj_id, [])

        if not row_indices:
            # No visits: single interval (0, t_end] with NaN TVC (only if carry_forward)
            if carry_forward:
                _append(
                    subj_id, 0.0, t_end, ev, static_vals, {col: float("nan") for col in tvc_cols}
                )
            continue

        # sort visits by time
        row_indices = sorted(row_indices, key=lambda j: vis_times[j])
        v_times = [float(vis_times[j]) for j in row_indices]
        v_tvc: dict[str, list[Any]] = {
            col: [vis_tvc[col][j] for j in row_indices] for col in tvc_cols
        }

        # validate: no visit after end of follow-up
        late = [vt for vt in v_times if vt > t_end]
        if late:
            raise ValueError(
                f"Subject {subj_id!r}: visit time(s) {late} exceed follow-up end {t_end}. "
                "Check that visit times are on the same scale as the follow-up time."
            )

        n_visits = len(v_times)

        # Count intervals added so far for this subject. carry_forward=False only drops
        # the trailing interval when at least one earlier interval already exists (a
        # subject whose only measurement is at time 0 should still produce one row).
        intervals_added_for_subject = 0

        # pre-first-visit interval: (0, v_times[0]) with NaN TVC
        if v_times[0] > 0.0 and carry_forward:
            _append(
                subj_id, 0.0, v_times[0], 0, static_vals, {col: float("nan") for col in tvc_cols}
            )
            intervals_added_for_subject += 1

        # intervals anchored at each visit
        for k in range(n_visits):
            tstart = v_times[k]
            tstop = v_times[k + 1] if k + 1 < n_visits else t_end
            tvc_here = {col: v_tvc[col][k] for col in tvc_cols}

            if tstart >= tstop:
                continue  # degenerate: duplicate visit time or visit at t_end

            is_last = k == n_visits - 1

            if is_last:
                # Drop the trailing interval when carry_forward=False, but only if there
                # is already at least one earlier interval for this subject.
                if carry_forward or intervals_added_for_subject == 0:
                    _append(subj_id, tstart, tstop, ev, static_vals, tvc_here)
            else:
                # Intermediate intervals are always included; event=0
                _append(subj_id, tstart, tstop, 0, static_vals, tvc_here)
                intervals_added_for_subject += 1

    if not out_id and not any(vis_by_id.get(subj_id) for subj_id in base_ids):
        raise ValueError(
            "split_episodes produced no rows. "
            "Check that the id columns in baseline and visits contain matching values."
        )

    out: dict[str, Any] = {
        id: out_id,
        "tstart": out_tstart,
        "tstop": out_tstop,
        event: out_event,
    }
    for col in static_cols:
        out[col] = out_static[col]
    for col in tvc_cols:
        out[col] = out_tvc[col]

    return to_dataframe(out, format=format)
