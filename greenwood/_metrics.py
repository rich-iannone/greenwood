"""Prediction-performance metrics for survival models.

- `concordance_index`: Harrell's C-statistic for an arbitrary risk score, validated against
  R's `survival::concordance`.
- `brier_score` / `integrated_brier_score`: the inverse-probability-of-censoring-weighted
  (Graf) Brier score at fixed times and its time integral, validated against R's
  `survival:::brier`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    from ._surv import Surv

__all__ = ["concordance_index", "brier_score", "integrated_brier_score"]

Array = npt.NDArray[Any]


def concordance_index(surv: Surv, risk: Any) -> float:
    """Harrell's concordance index between a risk score and observed survival.

    A higher `risk` should correspond to earlier failure. A subject that dies at time `t`
    is treated as failing before another still under observation at `t` (including one
    censored exactly at `t`); pairs tied in event time are excluded. Matches R's
    `survival::concordance`.
    """
    from ._surv import _to_1d_array

    scores = _to_1d_array(risk)
    exit_ = surv.stop
    event = surv.event
    if scores.shape[0] != surv.n:
        raise ValueError("`risk` must have the same length as the response.")

    concordant = 0.0
    comparable = 0.0
    for i in range(exit_.shape[0]):
        if not event[i]:
            continue
        later = (exit_ > exit_[i]) | ((exit_ == exit_[i]) & ~event)
        if not later.any():
            continue
        comparable += float(later.sum())
        concordant += float(np.sum(scores[i] > scores[later]))
        concordant += 0.5 * float(np.sum(scores[i] == scores[later]))
    return concordant / comparable


def _censoring_survival(surv: Surv) -> tuple[Array, Array]:
    from ._competing import _censoring_km

    status = np.where(surv.event, 1, 0)
    return _censoring_km(surv.stop, status)


def brier_score(surv: Surv, survival_prob: Any, times: Any) -> Array:
    """IPCW (Graf) Brier score of predicted survival probabilities at each of `times`.

    Parameters
    ----------
    surv
        A right-censored `Surv` response.
    survival_prob
        Predicted survival probabilities, shape `(n_obs, len(times))`.
    times
        The evaluation times.

    Returns
    -------
    The Brier score at each time (lower is better).
    """
    query = np.atleast_1d(np.asarray(times, dtype=float))
    probs = np.asarray(survival_prob, dtype=float)
    if probs.shape != (surv.n, query.shape[0]):
        raise ValueError(
            f"survival_prob must have shape (n_obs, len(times)) = "
            f"({surv.n}, {query.shape[0]}), got {probs.shape}."
        )

    exit_ = surv.stop
    event = surv.event
    drop_times, drop_surv = _censoring_survival(surv)

    def g_left(t: Array) -> Array:  # censoring survival just before t (1 if no censoring)
        if drop_times.shape[0] == 0:
            return np.ones_like(np.atleast_1d(t), dtype=float)
        idx = np.searchsorted(drop_times, t, side="left") - 1
        return np.where(idx >= 0, drop_surv[idx.clip(min=0)], 1.0)

    g_at_exit = g_left(exit_)
    out = np.empty(query.shape[0])
    with np.errstate(divide="ignore", invalid="ignore"):
        for j, t in enumerate(query):
            s = probs[:, j]
            dead = (exit_ <= t) & event
            alive = exit_ > t
            g_t = float(g_left(np.array([t]))[0])
            contrib = np.where(
                dead,
                s**2 / g_at_exit,
                np.where(alive, (1.0 - s) ** 2 / g_t, 0.0),
            )
            out[j] = float(contrib.mean())
    return out


def integrated_brier_score(surv: Surv, survival_prob: Any, times: Any) -> float:
    """Integrated Brier score: the time-average of `brier_score` over `times` (trapezoidal)."""
    query = np.atleast_1d(np.asarray(times, dtype=float))
    if query.shape[0] < 2:
        raise ValueError("integrated_brier_score needs at least two times.")
    scores = brier_score(surv, survival_prob, query)
    area = float(np.sum(np.diff(query) * (scores[:-1] + scores[1:]) / 2.0))
    return area / float(query[-1] - query[0])


def _survival_at(km: Any, time: float) -> tuple[float, float, float]:
    """Read a Kaplan-Meier estimate and its confidence limits at a single time."""
    frame = km.to_dataframe()
    grid = frame["time"].to_numpy()
    idx = int(np.searchsorted(grid, time, side="right")) - 1
    if idx < 0:
        return 1.0, 1.0, 1.0
    row = frame.iloc[idx]
    return float(row["estimate"]), float(row["conf_low"]), float(row["conf_high"])


