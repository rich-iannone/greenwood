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

__all__ = [
    "concordance_index",
    "brier_score",
    "integrated_brier_score",
    "calibration",
    "time_dependent_auc",
    "integrated_auc",
]

Array = npt.NDArray[Any]


def concordance_index(surv: Surv, risk: Any) -> float:
    r"""Harrell's concordance index: discrimination of risk scores against observed survival.

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

    Details
    -------
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
    (e.g., $\exp(\text{lp})$ vs. $\text{lp}$ both give the same concordance).

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
    r"""IPCW (Graf) Brier score of predicted survival probabilities at specified times.

    Measures calibration and accuracy of predicted survival probabilities at fixed time points
    using inverse-probability-of-censoring-weighted (IPCW) averaging. The Brier score is the
    mean squared difference between predicted and observed outcomes, weighted so censored
    subjects contribute honestly without bias.

    **Interpretation**:

    - Ranges from 0 (perfect predictions) to 1 (worst possible).
    - Lower is better. A Brier score of 0.25 means, on average, predictions are off by
      $\pm 0.5$ in terms of squared deviation.
    - **Null model baseline**: A model predicting 50% survival at every time has Brier score
      $\approx 0.25$. Compare your model to this baseline to assess practical improvement.
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

    Details
    -------
    **Graf (IPCW) Brier score**: The unbiased Brier score under censoring is

    $$
    BS(t) = E\left[(S(t) - \hat{S}(t))^2 \cdot \text{weights}\right],
    $$

    where:

    - $S(t)$ is the true survival status at time $t$ (1 if alive, 0 if dead)
    - $\hat{S}(t)$ is the predicted survival probability
    - $\text{weights}$ are inverse-probability-of-censoring: inverse of the censoring survival
      function

    Mathematically:

    $$
    BS(t) = \frac{1}{n} \sum_{\text{dead at } t} \frac{(\hat{S}_i(t))^2}{G(t_i^-)}
            + \frac{1}{n} \sum_{\text{alive at } t} \frac{(1 - \hat{S}_i(t))^2}{G(t)}
    $$

    where $G(u)$ is the Kaplan-Meier estimate of the censoring distribution (probability of not
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
    r"""Integrated (time-averaged) Brier score across multiple time points.

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

    Details
    -------
    **Computation**: The integrated Brier score is

    $$
    IBS = \frac{1}{t_{\max} - t_{\min}} \int BS(t) \, dt
    $$

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
        or `"pyarrow"`. `None` (the default) will auto-detects and prefer Polars if available (falls
        back to Pandas, then Pyarrow, and raises an error if no DataFrame library is available).

    Returns
    -------
    DataFrame
        One row per bin with columns `bin`, `n`, `predicted` (mean), `observed`,
        `observed_lower`, `observed_upper`. Format depends on the `format` parameter.

    Details
    -------
    Bins are formed by splitting subjects into `n_bins` quantile-based groups of their
    predicted survival probability. Within each bin the Kaplan–Meier estimate at `time`
    gives the observed survival, and the mean of the predictions gives the predicted value.
    A well-calibrated model produces points that lie along the 45-degree diagonal when
    plotted as observed vs. predicted. Systematic departures from the diagonal indicate
    over- or under-prediction in that probability range.

    Examples
    --------
    Fit a Cox model on the bundled `lung` dataset and assess calibration at one year. Each
    bin's mean predicted survival is compared against the observed Kaplan–Meier survival for
    that bin's subjects:

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


def time_dependent_auc(surv: Surv, marker: Any, times: Any) -> Array:
    r"""IPCW (Uno) time-dependent AUC at specified times.

    Computes the cumulative-dynamic AUC at each requested time using the inverse-probability-
    of-censoring-weighted (IPCW) estimator of Uno et al. (2011). At each time $t$, *cases*
    are subjects who experienced an event by $t$ and *controls* are subjects still at risk
    after $t$; the AUC measures how well the marker separates the two groups, correcting for
    censoring bias via IPCW weights.

    **Interpretation**:

    - 0.5: Random discrimination (marker carries no prognostic information at $t$).
    - > 0.5: Better-than-random; the marker ranks earlier-failing subjects higher.
    - 1.0: Perfect discrimination at $t$.
    - AUC tends to vary with $t$; use `integrated_auc()` for a single summary.

    **Higher marker = higher risk** convention: the marker should be on a scale where larger
    values indicate greater hazard (e.g., a Cox linear predictor, a predicted cumulative
    incidence, or a biomarker positively associated with failure). To use a *lower-is-worse*
    marker (e.g., predicted survival probability), negate it first.

    Parameters
    ----------
    surv
        A right-censored `Surv` response.
    marker
        Risk score for each subject (one value per observation). Higher values should
        indicate higher risk (earlier expected failure). Accepts a 1-D array, pandas/Polars
        Series, or Python sequence.
    times
        Evaluation times where the AUC is computed. 1-D array-like. Times before the first
        event or after the last observation yield `nan`.

    Returns
    -------
    ndarray
        AUC at each requested time, shape `(len(times),)`. Values are in [0, 1] or `nan`
        when a time has no cases or no controls.

    Details
    -------
    **Uno et al. (2011) estimator**: For time $t$ let

    - cases: $\mathcal{C}(t) = \{i : T_i \le t,\; \Delta_i = 1\}$
    - controls: $\mathcal{K}(t) = \{j : T_j > t\}$
    - IPCW weight for case $i$: $w_i = \hat{G}(T_i-)^{-2}$, where $\hat{G}$ is the
      Kaplan-Meier estimate of the *censoring* survival function.

    $$
    \widehat{AUC}(t) =
    \frac{\displaystyle\sum_{i \in \mathcal{C}(t)} w_i
          \Bigl[\#\{j\in\mathcal{K}(t):\eta_j < \eta_i\}
               + \tfrac{1}{2}\#\{j\in\mathcal{K}(t):\eta_j = \eta_i\}\Bigr]}
         {|\mathcal{K}(t)| \cdot \displaystyle\sum_{i \in \mathcal{C}(t)} w_i}
    $$

    When $G(t) = 1$ (no censoring) the estimator reduces to the empirical AUC of the
    binary problem "case vs. control at $t$".

    **Relationship to concordance**: The Harrell C-statistic is closely related to the
    time-averaged AUC across all event times; use `integrated_auc()` to obtain a single
    time-averaged summary that is directly comparable to the C-index.

    References
    ----------
    Uno H., Cai T., Pencina M.J., D'Agostino R.B., Wei L.J. (2011). On the C-statistics
    for evaluating overall adequacy of risk prediction procedures with censored survival
    data. *Statistics in Medicine*, 30(10), 1105-1117.

    Examples
    --------
    Fit a Cox model on the `lung` dataset and compute its time-dependent AUC using the
    linear predictor as the risk marker.

    ```{python}
    import greenwood as gw

    lung = gw.load_dataset("lung", backend="polars")
    y = gw.Surv.right(lung["time"], event=(lung["status"] == 2))
    cox = gw.CoxPH().fit(y, lung[["age", "sex"]])

    lp = cox.predict(type="lp")
    auc = gw.time_dependent_auc(y, lp, times=[180, 365, 540])
    auc
    ```

    Compare discrimination of two models via `integrated_auc()`:

    ```{python}
    ibs = gw.integrated_auc(y, lp, times=[180, 365, 540])
    ibs
    ```
    """
    from ._surv import _to_1d_array

    scores = _to_1d_array(marker)
    query = np.atleast_1d(np.asarray(times, dtype=float))
    if scores.shape[0] != surv.n:
        raise ValueError("`marker` must have the same length as the response.")

    T = surv.stop
    evt = surv.event.astype(bool)

    # Censoring KM: G(t-) is the censoring survival just before t.
    g_times, g_surv = _censoring_survival(surv)

    def _g_left(t_arr: Array) -> Array:
        if g_times.shape[0] == 0:
            return np.ones(len(t_arr))
        idx = np.searchsorted(g_times, t_arr, side="left") - 1
        return np.where(idx >= 0, g_surv[idx.clip(min=0)], 1.0)

    out = np.empty(query.shape[0])
    for j, t in enumerate(query):
        case_mask = (t >= T) & evt
        ctrl_mask = t < T
        n_ctrl = int(ctrl_mask.sum())

        if case_mask.sum() == 0 or n_ctrl == 0:
            out[j] = np.nan
            continue

        # IPCW weights: 1 / G(T_i-)^2 for each case
        g_case = _g_left(T[case_mask])
        with np.errstate(divide="ignore", invalid="ignore"):
            w = np.where(g_case > 0, 1.0 / g_case**2, 0.0)

        eta_case = scores[case_mask]  # (n_cases,)
        eta_ctrl = scores[ctrl_mask]  # (n_controls,)

        # For each case i, count controls j where η_j < η_i (concordant) plus
        # 0.5 * those where η_j == η_i (tied).  Broadcast shape: (n_cases, n_controls).
        diff = eta_case[:, None] - eta_ctrl[None, :]
        pairwise = (diff > 0).astype(float) + 0.5 * (diff == 0).astype(float)

        num = float(np.dot(w, pairwise.sum(axis=1)))
        denom = float(w.sum()) * n_ctrl
        out[j] = num / denom if denom > 0.0 else np.nan

    return out


def integrated_auc(surv: Surv, marker: Any, times: Any) -> float:
    r"""Time-averaged IPCW AUC across multiple time points.

    Summarises `time_dependent_auc()` into a single number via trapezoidal integration
    over the supplied time range. This provides a discrimination summary analogous to
    Harrell's C-statistic but with explicit IPCW bias-correction for censoring.

    **Interpretation**: Same scale as `time_dependent_auc()` (0.5 = random, 1.0 = perfect).
    Values of 0.6-0.7 indicate moderate and 0.7+ indicate strong discrimination.

    Parameters
    ----------
    surv
        A right-censored `Surv` response.
    marker
        Risk score for each subject. Higher values indicate higher risk.
    times
        Evaluation times (at least 2). The integrated AUC is computed as the area under the
        AUC curve from `times[0]` to `times[-1]`, normalized by the time span. `nan` time
        points (no cases or controls) are dropped before integration.

    Returns
    -------
    float
        Time-averaged AUC in [0, 1]. Higher is better.

    Examples
    --------
    ```{python}
    import greenwood as gw

    lung = gw.load_dataset("lung", backend="polars")
    y = gw.Surv.right(lung["time"], event=(lung["status"] == 2))
    cox = gw.CoxPH().fit(y, lung[["age", "sex"]])

    lp = cox.predict(type="lp")
    gw.integrated_auc(y, lp, times=[180, 365, 540])
    ```
    """
    query = np.atleast_1d(np.asarray(times, dtype=float))
    if query.shape[0] < 2:
        raise ValueError("integrated_auc needs at least two times.")
    auc = time_dependent_auc(surv, marker, query)
    valid = ~np.isnan(auc)
    if valid.sum() < 2:
        raise ValueError(
            "integrated_auc: fewer than two valid AUC values after dropping nan time points."
        )
    t_v = query[valid]
    a_v = auc[valid]
    area = float(np.sum(np.diff(t_v) * (a_v[:-1] + a_v[1:]) / 2.0))
    return area / float(t_v[-1] - t_v[0])
