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

__all__ = ["concordance_index", "brier_score", "integrated_brier_score", "calibration"]

Array = npt.NDArray[Any]


def concordance_index(surv: Surv, risk: Any) -> float:
    """Harrell's concordance index between a risk score and observed survival.

    A higher `risk` should correspond to earlier failure. A subject that dies at time `t`
    is treated as failing before another still under observation at `t` (including one
    censored exactly at `t`); pairs tied in event time are excluded. Matches R's
    `survival::concordance`.

    Examples
    --------
    Fit a Cox model on the bundled `lung` dataset, then score its linear predictor. A higher
    linear predictor (`type="lp"`) means higher risk, which should correspond to earlier
    events, so a well-discriminating model scores above 0.5.

    ```{python}
    import greenwood as gw

    lung = gw.load_dataset("lung")
    y = gw.Surv.right(lung["time"], event=(lung["status"] == 2))
    cox = gw.CoxPH().fit(y, lung[["age", "sex"]])

    gw.concordance_index(y, cox.predict(type="lp"))
    ```
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

    Examples
    --------
    Fit a Cox model on the bundled `lung` dataset, read predicted survival probabilities at a
    few horizons, and score them. Lower is better; the score uses inverse-probability-of-censoring
    weighting so that censored subjects contribute honestly.

    ```{python}
    import greenwood as gw

    lung = gw.load_dataset("lung")
    y = gw.Surv.right(lung["time"], event=(lung["status"] == 2))
    cox = gw.CoxPH().fit(y, lung[["age", "sex"]])

    times = [180, 365, 540]
    surv = cox.predict(lung[["age", "sex"]], type="survival", times=times)
    probs = surv.iloc[:, 1:].to_numpy().T   # shape (n_subjects, n_times)
    gw.brier_score(y, probs, times)
    ```
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
    """Integrated Brier score: the time-average of `brier_score` over `times` (trapezoidal).

    Examples
    --------
    Fit a Cox model on the bundled `lung` dataset and reduce the Brier scores across several
    horizons to a single time-averaged number (lower is better).

    ```{python}
    import greenwood as gw

    lung = gw.load_dataset("lung")
    y = gw.Surv.right(lung["time"], event=(lung["status"] == 2))
    cox = gw.CoxPH().fit(y, lung[["age", "sex"]])

    times = [180, 365, 540]
    surv = cox.predict(lung[["age", "sex"]], type="survival", times=times)
    probs = surv.iloc[:, 1:].to_numpy().T   # shape (n_subjects, n_times)
    gw.integrated_brier_score(y, probs, times)
    ```
    """
    query = np.atleast_1d(np.asarray(times, dtype=float))
    if query.shape[0] < 2:
        raise ValueError("integrated_brier_score needs at least two times.")
    scores = brier_score(surv, survival_prob, query)
    area = float(np.sum(np.diff(query) * (scores[:-1] + scores[1:]) / 2.0))
    return area / float(query[-1] - query[0])


def _survival_at(km: Any, time: float) -> tuple[float, float, float]:
    """Read a Kaplan-Meier estimate and its confidence limits at a single time."""
    frame = km.to_pandas()
    grid = frame["time"].to_numpy()
    idx = int(np.searchsorted(grid, time, side="right")) - 1
    if idx < 0:
        return 1.0, 1.0, 1.0
    row = frame.iloc[idx]
    return float(row["estimate"]), float(row["conf_low"]), float(row["conf_high"])


def calibration(
    surv: Surv, predicted: Any, time: float, *, n_bins: int = 10, conf_level: float = 0.95
) -> Any:
    """Assess calibration of predicted survival probabilities at a fixed time.

    Subjects are grouped into `n_bins` bins by their predicted survival probability at
    `time`. Within each bin the mean prediction is compared against the observed survival, a
    Kaplan-Meier estimate at `time` for that bin's subjects. A well-calibrated model has the
    observed values close to the predicted ones (points near the diagonal).

    Parameters
    ----------
    surv
        The `Surv` response (right-censored or counting-process).
    predicted
        Predicted survival probability at `time`, one per subject (for example a column of
        `CoxPH.predict(newdata, type="survival", times=[time])`).
    time
        The horizon at which predictions are assessed.
    n_bins
        Number of prediction bins (default 10). Bins are quantile-based; empty bins are
        dropped, so ties in `predicted` may yield fewer rows.
    conf_level
        Confidence level for the observed (Kaplan-Meier) interval.

    Returns
    -------
    A pandas DataFrame with one row per bin: `bin`, `n`, `predicted` (mean), `observed`,
    `observed_lower`, `observed_upper`.

    Examples
    --------
    Fit a Cox model on the bundled `lung` dataset and read the predicted survival at a single
    horizon (one value per subject). Subjects are grouped into bins by their predicted survival,
    and each bin's mean `predicted` value is compared against the observed Kaplan-Meier survival
    for that bin's subjects.

    ```{python}
    import greenwood as gw

    lung = gw.load_dataset("lung")
    y = gw.Surv.right(lung["time"], event=(lung["status"] == 2))
    cox = gw.CoxPH().fit(y, lung[["age", "sex"]])

    surv = cox.predict(lung[["age", "sex"]], type="survival", times=[365.0])
    predicted = surv.iloc[0, 1:].to_numpy()
    gw.calibration(y, predicted, 365.0, n_bins=5)
    ```
    """
    import pandas as pd

    from ._nonparametric import KaplanMeier
    from ._resample import _subset_surv

    pred = np.asarray(predicted, dtype=float)
    if pred.shape[0] != surv.n:
        raise ValueError("`predicted` must have one value per subject.")
    if n_bins < 2:
        raise ValueError("n_bins must be at least 2.")
    horizon = float(time)

    edges = np.quantile(pred, np.linspace(0.0, 1.0, n_bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    bin_idx = np.clip(np.searchsorted(edges, pred, side="right") - 1, 0, n_bins - 1)

    rows: list[dict[str, Any]] = []
    for b in range(n_bins):
        members = np.where(bin_idx == b)[0]
        if members.size == 0:
            continue
        km = KaplanMeier(conf_level=conf_level).fit(_subset_surv(surv, members))
        observed, lower, upper = _survival_at(km, horizon)
        rows.append(
            {
                "bin": len(rows) + 1,
                "n": int(members.size),
                "predicted": float(pred[members].mean()),
                "observed": observed,
                "observed_lower": lower,
                "observed_upper": upper,
            }
        )
    return pd.DataFrame(rows)
