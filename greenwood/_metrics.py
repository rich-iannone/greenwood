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
    """Harrell's concordance index: discrimination of risk scores against observed survival.

    Computes Harrell's C-statistic, a measure of how well a risk score discriminates between
    subjects who experience early events and those who survive longer. The concordance index
    compares all comparable pairs of subjects: those with an observed event are compared to
    those still under observation at the same time or later. Higher risk should correspond to
    earlier failure; if predictions match reality better than chance, the index exceeds 0.5.

    **Interpretation**:

    - 0.5: Random discrimination (no better than a coin flip)
    - 0.6-0.7: Moderate discrimination (clinically useful)
    - 0.7-0.8: Strong discrimination
    - 0.8+: Excellent discrimination (rare in practice)

    **Practical use**: Validates a model's ability to rank subjects by risk. After fitting a
    Cox model or other survival model, compute the concordance index of its predictions to
    assess out-of-sample discrimination. A model with high concordance index generalizes well
    to ranking future subjects by risk.

    Parameters
    ----------
    surv
        A right-censored `Surv` response (time-to-event data).
    risk
        Risk score for each subject, one per observation. Can be a 1-D array, Pandas/Polars
        series, or Python sequence. Higher values indicate higher risk (earlier expected
        failure). Examples: Cox model linear predictor, predicted log-hazard, or predicted
        cumulative incidence.

    Returns
    -------
    float
        Concordance index between 0 and 1. Values above 0.5 indicate the model discriminates
        better than random; below 0.5 indicates worse-than-random discrimination (possibly
        inverted risk scale).

    Notes
    -----
    **Pair comparison rule**:

    - A subject with an observed event at time t is compared to all subjects still under
      observation at time t or later (including censored subjects at exactly t).
    - Pairs are concordant if the subject with the event has higher risk than the subject
      without.
    - Pairs with tied risk scores are counted as 0.5 (half-concordant).
    - Pairs with the same event time are excluded.

    **Censoring handling**: Censored subjects are handled through the comparable pairs
    definition. A censored subject at time t can only be compared to subjects with events at
    times strictly greater than t. This avoids artificial inflation of concordance from
    censored subjects and matches R's `survival::concordance`.

    **Relationship to other metrics**: The concordance index is related to the rank correlation
    between risk and observed survival. It's invariant to monotonic transformations of risk
    (e.g., exp(lp) vs. lp both give the same concordance).

    Examples
    --------
    Fit a Cox model on the `lung` dataset and evaluate its discrimination. Higher linear
    predictor values should correspond to earlier deaths.

    ```{python}
    import greenwood as gw

    lung = gw.load_dataset("lung", backend="polars")
    y = gw.Surv.right(lung["time"], event=(lung["status"] == 2))
    cox = gw.CoxPH().fit(y, lung[["age", "sex"]])

    lp = cox.predict(type="lp")
    c_index = gw.concordance_index(y, lp)
    c_index
    ```

    A concordance index of ~0.6 indicates moderate discrimination. Compare with baseline
    (naive model assuming all subjects are at equal risk):

    ```{python}
    import numpy as np
    baseline_c = gw.concordance_index(y, np.zeros(len(y)))
    print(f"Baseline: {baseline_c:.3f}")
    print(f"Cox model: {c_index:.3f}")
    print(f"Improvement: {c_index - baseline_c:.3f}")
    ```

    Evaluate on hold-out test data to assess generalization (use Cox model fit on training
    data, predict on test data):

    ```{python}
    # cox = CoxPH().fit(y_train, x_train)
    # lp_test = cox.predict(x_test, type="lp")
    # c_test = gw.concordance_index(y_test, lp_test)
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
    """IPCW (Graf) Brier score of predicted survival probabilities at specified times.

    Measures calibration and accuracy of predicted survival probabilities at fixed time points
    using inverse-probability-of-censoring-weighted (IPCW) averaging. The Brier score is the
    mean squared difference between predicted and observed outcomes, weighted so censored
    subjects contribute honestly without bias.

    **Interpretation**:

    - Ranges from 0 (perfect predictions) to 1 (worst possible).
    - Lower is better. A Brier score of 0.25 means, on average, predictions are off by ±0.5
      in terms of squared deviation.
    - **Null model baseline**: A model predicting 50% survival at every time has Brier score
      ~0.25. Compare your model to this baseline to assess practical improvement.
    - Score typically increases with time (harder to predict farther into the future).

    **Practical use**: After fitting a survival model (Cox, parametric, flexible), evaluate
    calibration at important clinical horizons (e.g., 1-year, 5-year survival). Compute Brier
    scores at multiple times, then use `integrated_brier_score()` for a single summary.

    Parameters
    ----------
    surv
        A right-censored `Surv` response (time-to-event data).
    survival_prob
        Predicted survival probabilities, shape `(n_subjects, n_times)`. Each entry is a
        predicted probability of surviving beyond the corresponding time. Must be between 0
        and 1. Example: columns from `CoxPH.predict(type="survival", times=...)` (excluding
        the `time` column), transposed so rows are times and columns are subjects.
    times
        Evaluation times where Brier scores are computed. 1-D array-like. Must have length
        equal to the second dimension of `survival_prob`.

    Returns
    -------
    ndarray
        Brier score at each time, shape `(len(times),)`. Lower is better.

    Notes
    -----
    **Graf (IPCW) Brier score**: The unbiased Brier score under censoring is

        BS(t) = E[(S(t) - hat_S(t))^2 * weights],

    where:

    - S(t) is the true survival status at time t (1 if alive, 0 if dead)
    - hat_S(t) is the predicted survival probability
    - weights are inverse-probability-of-censoring: inverse of the censoring survival function

    Mathematically:

        BS(t) = (1/n) * [sum over dead at t: (hat_S_i(t))^2 / G(t_i^-)]
                 + (1/n) * [sum over alive at t: (1 - hat_S_i(t))^2 / G(t)]

    where G(u) is the Kaplan-Meier estimate of the censoring distribution (probability of not
    being censored).

    **Advantages over MSE**: IPCW weighting makes the Brier score unbiased under censoring,
    unlike naive MSE which would be biased (censored subjects look artificially "correct").

    **Time dependence**: Brier scores typically increase with time in chronic-disease settings
    (longer prediction horizons are harder); they may decrease in acute-illness settings
    (events cluster early, predictions stabilize).

    Examples
    --------
    Fit a Cox model on the `lung` dataset and evaluate its calibration (how accurately does
    it predict survival?) at a few horizons.

    ```{python}
    import greenwood as gw
    import numpy as np

    lung = gw.load_dataset("lung", backend="polars")
    y = gw.Surv.right(lung["time"], event=(lung["status"] == 2))
    cox = gw.CoxPH().fit(y, lung[["age", "sex"]])

    times = [180, 365, 540]
    surv_pred = cox.predict(lung[["age", "sex"]], type="survival", times=times, format="pandas")
    probs = surv_pred.iloc[:, 1:].to_numpy().T
    brier = gw.brier_score(y, probs, times)
    brier
    ```

    Brier scores at three time points. Scores typically increase over time. Compare to a
    null model (all subjects at 50% survival) to assess improvement:

    ```{python}
    null_probs = np.full_like(probs, 0.5)
    null_brier = gw.brier_score(y, null_probs, times)
    print(f"Null model Brier: {null_brier}")
    print(f"Cox model Brier: {brier}")
    print(f"Improvement: {null_brier - brier}")
    ```

    Summarize Brier scores across times into a single summary via time-averaged Brier score:

    ```{python}
    ibs = gw.integrated_brier_score(y, probs, times)
    print(f"Integrated Brier Score: {ibs:.3f}")
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
    """Integrated (time-averaged) Brier score across multiple time points.

    Summarizes Brier scores computed at multiple time horizons into a single summary metric
    via trapezoidal integration. This provides an overall calibration measure that doesn't
    emphasize any particular time point.

    **Use this to**: Reduce multiple Brier scores (one per time point) to a single number
    for model comparison. A single IBS score makes it easier to compare two models or report
    a single "calibration quality" metric.

    **Interpretation**: Same scale as Brier score (0 = perfect, 1 = worst). Values of
    0.15-0.25 are typical for reasonable survival models; values > 0.30 suggest poor
    calibration.

    Parameters
    ----------
    surv
        A right-censored `Surv` response.
    survival_prob
        Predicted survival probabilities, shape `(n_subjects, n_times)`.
    times
        Evaluation times (must be at least 2 to define an interval). The IBS is computed as
        the area under the Brier-score curve from times[0] to times[-1], normalized by the
        time span.

    Returns
    -------
    float
        Integrated Brier score (time-averaged). Lower is better.

    Notes
    -----
    **Computation**: The integrated Brier score is

        IBS = (1 / (t_max - t_min)) * integral over t of BS(t) dt

    Using trapezoidal rule to approximate the integral. This ensures the summary score
    balances contributions from all times without emphasizing early or late horizons.

    **Time scale sensitivity**: IBS weights contribution proportionally to time intervals.
    If you have many time points clustered early (e.g., 10 times in [0, 100] and 1 time at
    [1000]), early times dominate. Use evenly-spaced time points for balanced assessment.

    Examples
    --------
    Fit a Cox model and compute its integrated Brier score over a range of times:

    ```{python}
    import greenwood as gw

    lung = gw.load_dataset("lung", backend="polars")
    y = gw.Surv.right(lung["time"], event=(lung["status"] == 2))
    cox = gw.CoxPH().fit(y, lung[["age", "sex"]])

    times = [180, 365, 540]
    surv_pred = cox.predict(lung[["age", "sex"]], type="survival", times=times, format="pandas")
    probs = surv_pred.iloc[:, 1:].to_numpy().T
    ibs = gw.integrated_brier_score(y, probs, times)
    ibs
    ```

    Compare two models via their integrated Brier scores. Lower is better:

    ```{python}
    # cox2 = CoxPH().fit(y, lung[["age", "sex", "ph.ecog"]])  # More covariates
    # surv_pred2 = cox2.predict(...)
    # ibs2 = gw.integrated_brier_score(y, probs2, times)
    # print(f"Model 1 IBS: {ibs:.3f}")
    # print(f"Model 2 IBS: {ibs2:.3f}")
    # print(f"Better model: {'Model 2' if ibs2 < ibs else 'Model 1'}")
    ```

    Compute integrated Brier score over a wide range of times to get an overall calibration
    assessment:

    ```{python}
    times_wide = list(range(100, 700, 50))
    surv_pred_wide = cox.predict(
        lung[["age", "sex"]], type="survival", times=times_wide, format="pandas"
    )
    probs_wide = surv_pred_wide.iloc[:, 1:].to_numpy().T
    ibs_wide = gw.integrated_brier_score(y, probs_wide, times_wide)
    print(f"IBS over {len(times_wide)} time points: {ibs_wide:.3f}")
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
    # Read straight off the fitted arrays, so no DataFrame library is required here.
    grid = km.time_
    idx = int(np.searchsorted(grid, time, side="right")) - 1
    if idx < 0:
        return 1.0, 1.0, 1.0
    return float(km.survival_[idx]), float(km.conf_low_[idx]), float(km.conf_high_[idx])


def calibration(
    surv: Surv,
    predicted: Any,
    time: float,
    *,
    n_bins: int = 10,
    conf_level: float = 0.95,
    format: str | None = None,
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
    format
        Output format for the returned DataFrame: `None` (default), `"pandas"`, `"polars"`,
        or `"pyarrow"`.

        - `None` (default): Auto-detects and prefers Polars if available, falls back to
          Pandas, then Pyarrow. Raises error if no DataFrame library is available.
        - `"pandas"`: returns pandas.DataFrame.
        - `"polars"`: returns polars.DataFrame.
        - `"pyarrow"`: returns pyarrow.Table.

    Returns
    -------
    A DataFrame with one row per bin: `bin`, `n`, `predicted` (mean), `observed`,
    `observed_lower`, `observed_upper`. Format depends on `format` parameter.

    Examples
    --------
    Fit a Cox model on the bundled `lung` dataset and read the predicted survival at a single
    horizon (one value per subject). Subjects are grouped into bins by their predicted survival,
    and each bin's mean `predicted` value is compared against the observed Kaplan-Meier survival
    for that bin's subjects.

    ```{python}
    import greenwood as gw

    lung = gw.load_dataset("lung", backend="polars")
    y = gw.Surv.right(lung["time"], event=(lung["status"] == 2))
    cox = gw.CoxPH().fit(y, lung[["age", "sex"]])

    surv = cox.predict(lung[["age", "sex"]], type="survival", times=[365.0], format="pandas")
    predicted = surv.iloc[0, 1:].to_numpy()
    gw.calibration(y, predicted, 365.0, n_bins=5, format="polars")
    ```
    """
    from ._backends import to_dataframe
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
    return to_dataframe(
        {col: [row[col] for row in rows] for col in rows[0]}
        if rows
        else {
            "bin": [],
            "n": [],
            "predicted": [],
            "observed": [],
            "observed_lower": [],
            "observed_upper": [],
        },
        format=format,
    )
