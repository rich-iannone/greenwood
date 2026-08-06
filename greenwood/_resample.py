"""K-fold cross-validation for honest, out-of-sample survival-model evaluation.

Fitting a model and scoring it on the same data is optimistic. `cross_validate` splits the
data into folds, fits on the training folds, and scores predictions on the held-out fold,
using the censoring-aware metrics in `greenwood._metrics`.
"""

from __future__ import annotations

import copy
import warnings
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    from ._surv import Surv

__all__ = ["cross_validate"]

Array = npt.NDArray[Any]

_VALID_METRICS = frozenset({"concordance", "brier", "auc"})


def _subset_surv(surv: Surv, idx: Array) -> Surv:
    """Rebuild a `Surv` response from a row subset (right-censored or counting-process)."""
    from ._surv import CensoringType, Surv

    weights = None if surv.weights is None else surv.weights[idx]
    if surv.type is CensoringType.RIGHT:
        return Surv.right(surv.stop[idx], event=surv.event[idx], weights=weights)
    if surv.type is CensoringType.COUNTING:
        return Surv.counting(
            surv.entry[idx], surv.stop[idx], event=surv.event[idx], weights=weights
        )
    raise NotImplementedError(
        "cross_validate supports right-censored and counting-process responses, "
        f"not {surv.type.value!r}."
    )


def _stratified_kfold_indices(surv: Surv, k: int, seed: int | None = None) -> list[Array]:
    """Create k-fold indices stratified by event status.

    For survival data, stratification ensures each fold has approximately the same
    proportion of events and censored observations as the overall dataset. This is
    critical for imbalanced survival data (e.g., rare events) to prevent singular
    matrix errors and biased CV estimates.

    Parameters
    ----------
    surv
        A Surv response object containing event indicators.
    k
        Number of folds.
    seed
        Random seed for reproducibility.

    Returns
    -------
    list of arrays
        k arrays, each containing row indices for a fold. Folds are stratified by
        event status (censored vs. event).
    """
    rng = np.random.default_rng(seed)

    # For multi-state (multiple events), stratify by event type; for binary (event/censoring),
    # stratify by event indicator.
    if surv.event.dtype == object or (  # pragma: no cover
        hasattr(surv.event, "dtype") and surv.event.dtype.kind in ("U", "O")
    ):
        stratify_by = surv.event  # pragma: no cover
    else:
        # Binary event indicator: stratify by event status
        stratify_by = surv.event

    # Group indices by stratum
    unique_strata = np.unique(stratify_by)
    stratum_indices = {s: np.where(stratify_by == s)[0] for s in unique_strata}

    # For each stratum, shuffle and split into k folds
    fold_lists = [[] for _ in range(k)]
    for stratum_idx in unique_strata:
        indices = stratum_indices[stratum_idx]
        shuffled = rng.permutation(indices)
        stratum_folds = np.array_split(shuffled, k)
        for fold_idx, fold_indices in enumerate(stratum_folds):
            fold_lists[fold_idx].extend(fold_indices)

    # Shuffle within each fold to break any remaining structure
    folds = [rng.permutation(np.array(f)) for f in fold_lists]
    return folds


def _risk_score(model: Any, x: Array) -> Array:
    """A risk score where larger means higher risk (earlier event), for concordance."""
    from ._cox import CoxPH
    from ._flexible import RoystonParmar
    from ._parametric import AFT
    from ._penalized import CoxNet

    if isinstance(model, (CoxPH, CoxNet)):
        return model.predict(x, type="lp")
    if isinstance(model, AFT):
        return -model.predict(x, type="lp")
    if isinstance(model, RoystonParmar):
        x_design, _ = _design_matrix_for_rp(x)
        beta = model.coef_[model._n_spline :]
        if beta.size == 0:
            return np.zeros(x_design.shape[0])
        return x_design @ beta
    raise TypeError(
        f"cross_validate with metric='concordance' or 'auc' needs a CoxPH, CoxNet, AFT, "
        f"or RoystonParmar model, got {type(model).__name__}."
    )


def _design_matrix_for_rp(covariates: Any) -> tuple[Array, Any]:
    """Build a design matrix suitable for RoystonParmar (no intercept)."""
    from ._cox import _design_matrix

    return _design_matrix(covariates)


def _extract_survival_probs(model: Any, x: Array, times_list: list[float]) -> Array:
    """Extract survival probabilities as (n_subjects, n_times) array."""
    import narwhals as nw  # pyright: ignore[reportMissingImports]

    frame = model.predict(x, type="survival", times=times_list)
    native = nw.from_native(frame)
    return native.drop(native.columns[0]).to_numpy().T


def _score_fold(
    model: Any,
    surv_test: Surv,
    x_test: Array,
    metric_name: str,
    times_list: list[float],
) -> float:
    """Compute a single metric on a held-out fold."""
    from ._metrics import concordance_index, integrated_auc, integrated_brier_score

    if metric_name == "concordance":
        return float(concordance_index(surv_test, _risk_score(model, x_test)))
    if metric_name == "brier":
        probs = _extract_survival_probs(model, x_test, times_list)
        return float(integrated_brier_score(surv_test, probs, times_list))
    # auc
    return float(integrated_auc(surv_test, _risk_score(model, x_test), times_list))


def cross_validate(
    model: Any,
    surv: Surv,
    covariates: Any,
    *,
    data: Any = None,
    k: int = 5,
    metric: str | None = None,
    metrics: list[str] | None = None,
    times: Any = None,
    stratified: bool = True,
    seed: int | None = None,
) -> dict[str, Any]:
    r"""Evaluate a survival model's out-of-sample performance using k-fold cross-validation.

    Provides an honest, unbiased estimate of model performance by splitting data into folds,
    fitting on training folds, and evaluating on held-out test folds. This avoids overfitting
    bias that occurs when fitting and scoring on the same data.

    **Why cross-validate?** Fitting and scoring on the training data gives overly optimistic
    performance estimates. A model may fit the training data well due to overfitting, not
    true predictive ability. Cross-validation repeatedly fits on different training splits
    and evaluates on held-out data, simulating performance on new subjects.

    **Metrics**:

    - `"concordance"`: Harrell's C-statistic on the test fold. Higher is better
      (0.5 = random, 1.0 = perfect).
    - `"brier"`: Integrated IPCW Brier score over specified times. Lower is better
      (0 = perfect calibration, 1 = worst). Requires explicit `times=` parameter.
    - `"auc"`: Integrated time-dependent AUC (Uno estimator). Higher is better
      (0.5 = random, 1.0 = perfect). Requires explicit `times=` parameter.

    Parameters
    ----------
    model
        An unfitted estimator instance (e.g., `CoxPH()`, `CoxNet()`, `AFT("weibull")`,
        `RoystonParmar(df=3)`). A fresh copy is fit on each training fold, leaving the passed object
        unchanged.
    surv
        A `Surv` response (time-to-event data). Can be right-censored or counting-process. Weights
        in the response are carried through the cross-validation.
    covariates
        Covariates/predictors for the model. Can be:

        - A 2-D array or pandas/Polars DataFrame with one row per subject
        - A formula string (as in `CoxPH.fit()`), evaluated against `data`

    data
        If `covariates` is a formula string, the data frame to evaluate it against.
    k
        Number of folds (default 5). Each fold serves as test data once; subjects are split randomly
        and evenly across folds. Typical choices: 5 or 10.
    metric
        A single performance metric (backward-compatible). Use `metrics` instead to evaluate
        multiple metrics in a single CV run. If neither is provided, defaults to
        `"concordance"`.
    metrics
        A list of metrics to evaluate in a single CV run. The model is fit once per fold and scored
        on all requested metrics. Cannot be used together with `metric`. Supported options are:
        `"concordance"`, `"brier"`, and `"auc"`.
    times
        Evaluation time points for `"brier"` and `"auc"` metrics (1-D array-like, length $\ge 2$).
        Required when using those metrics. Example: `times=[365, 730, 1095]` for 1-, 2-, and 3-year
        predictions.
    stratified
        If `True` (default), use stratified k-fold ensuring balanced event/censoring representation
        across folds. This prevents singular matrix errors and biased CV estimates on imbalanced
        survival data (rare events). If `False`, use simple random k-fold shuffling.
    seed
        Random seed for fold shuffling, ensures reproducibility. If `None`, results may vary between
        runs. Use a fixed seed for consistent comparisons.

    Returns
    -------
    dict
        When `metric` (singular) is used, returns a flat dictionary:

        - `"metric"`: Metric name used.
        - `"k"`: Number of folds.
        - `"scores"`: List of per-fold scores.
        - `"mean"`: Mean score across folds.
        - `"std"`: Standard deviation of scores.

        When `metrics` (plural) is used, returns a keyed dictionary:

        - `"k"`: Number of folds.
        - `"results"`: Dict keyed by metric name, each with `"scores"`, `"mean"`, and
          `"std"`.

    Details
    -------
    **How folds work**: By default (`stratified=True`), subjects are grouped by event status
    (censored vs. event, or multiple event types), then randomly shuffled within each stratum and
    split into k roughly equal-sized groups. This ensures each fold has approximately the same
    proportion of events and censored observations as the overall dataset. This is crucial for
    imbalanced data (e.g., rare events) to prevent singular matrix errors and ensures unbiased
    cross-validation estimates.

    If `stratified=False`, subjects are simply shuffled and split randomly, which may lead to folds
    with very different event rates and can destabilize model fitting on sparse data.

    **Multi-metric mode**: When `metrics` is a list, the model is fit once per fold and all
    requested metrics are computed on the same held-out data. This is more efficient than calling
    `cross_validate` separately for each metric (which would refit the model each time) and ensures
    all metrics see the same folds.

    **Completeness**: Subjects with missing covariates are dropped before folding. This ensures all
    folds use the same cleaned data, avoiding alignment issues.

    **AFT model note**: For AFT, concordance uses the negated linear predictor (since in AFT, larger
    lp means longer survival, opposite to Cox). This is handled automatically.

    **Reproducibility**: Set `seed=` to ensure the same folds are used across runs. This is
    important for comparing different models or reporting consistent results.

    Examples
    --------
    Evaluate a Cox model with 5-fold cross-validation using concordance:

    ```{python}
    import greenwood as gw

    # Load data and build a right-censored response
    lung = gw.load_dataset("lung", backend="polars")
    y = gw.Surv.right(lung["time"], event=(lung["status"] == 2))

    # Run 5-fold cross-validation with concordance
    result = gw.cross_validate(
        gw.CoxPH(), y, lung[["age", "sex"]], k=5, metric="concordance", seed=1
    )
    result
    ```

    Access individual components. The mean concordance across folds:

    ```{python}
    # Mean concordance across folds
    result["mean"]
    ```

    Per-fold scores (variability check):

    ```{python}
    # Per-fold concordance scores
    result["scores"]
    ```

    Standard deviation (estimate of generalization uncertainty):

    ```{python}
    # Standard deviation of fold scores
    result["std"]
    ```

    Evaluate multiple metrics in a single CV run. The model is fit once per fold and scored on all
    requested metrics:

    ```{python}
    # Evaluate concordance, Brier, and AUC in one pass
    result_multi = gw.cross_validate(
        gw.CoxPH(), y, lung[["age", "sex"]], k=5,
        metrics=["concordance", "brier", "auc"],
        times=[180, 365, 540], seed=1
    )
    result_multi
    ```

    Access results for a specific metric:

    ```{python}
    # Mean concordance
    result_multi["results"]["concordance"]["mean"]
    ```

    ```{python}
    # Mean integrated Brier score
    result_multi["results"]["brier"]["mean"]
    ```
    """
    from ._cox import CoxPH, _design_matrix
    from ._flexible import RoystonParmar
    from ._parametric import AFT
    from ._penalized import CoxNet

    if not isinstance(model, (CoxPH, CoxNet, AFT, RoystonParmar)):
        raise TypeError(
            f"cross_validate needs a CoxPH, CoxNet, AFT, or RoystonParmar model, "
            f"got {type(model).__name__}."
        )
    if k < 2:
        raise ValueError("k must be at least 2.")

    if metric is not None and metrics is not None:
        raise ValueError("Specify either `metric` or `metrics`, not both.")

    multi_mode = metrics is not None
    if metrics is not None:
        metric_list = list(metrics)
    elif metric is not None:
        metric_list = [metric]
    else:
        metric_list = ["concordance"]

    unknown = set(metric_list) - _VALID_METRICS
    if unknown:
        raise ValueError(
            f"Unknown metric(s) {sorted(unknown)}; use 'concordance', 'brier', or 'auc'."
        )

    design, _ = _design_matrix(covariates, data)
    if design.shape[0] != surv.n:
        raise ValueError("Covariates and response must have the same number of rows.")

    keep = ~np.isnan(design).any(axis=1)
    if not keep.all():
        design = design[keep]
        surv = _subset_surv(surv, np.nonzero(keep)[0])

    times_list: list[float] = []
    needs_times = {"brier", "auc"}
    if needs_times & set(metric_list):
        if times is None:
            raise ValueError(
                "The 'brier' and 'auc' metrics require `times` (at least two time points)."
            )
        times_list = [float(t) for t in np.atleast_1d(np.asarray(times, dtype=float))]
        if len(times_list) < 2:
            raise ValueError(
                "The 'brier' and 'auc' metrics require `times` with at least two time points."
            )

    folds = (
        _stratified_kfold_indices(surv, k, seed)
        if stratified
        else np.array_split(np.random.default_rng(seed).permutation(surv.n), k)
    )

    n_events = int(surv.event.astype(bool).sum())
    if n_events < 2 * k:
        warnings.warn(
            f"Only {n_events} events found for {k}-fold cross-validation "
            f"(fewer than 2 × k = {2 * k}). "
            "Some folds may contain too few events for reliable evaluation. "
            "Consider reducing k or collecting more data with observed events.",
            UserWarning,
            stacklevel=2,
        )

    all_scores: dict[str, list[float]] = {m: [] for m in metric_list}

    for i in range(k):
        test = folds[i]
        train = np.concatenate([folds[j] for j in range(k) if j != i])
        fold_model = copy.deepcopy(model)
        fold_model.fit(_subset_surv(surv, train), design[train])
        surv_test = _subset_surv(surv, test)
        x_test = design[test]

        for m in metric_list:
            all_scores[m].append(_score_fold(fold_model, surv_test, x_test, m, times_list))

    if multi_mode:
        results: dict[str, dict[str, Any]] = {}
        for m in metric_list:
            arr = np.asarray(all_scores[m])
            results[m] = {
                "scores": all_scores[m],
                "mean": float(arr.mean()),
                "std": float(arr.std(ddof=1)),
            }
        return {"k": k, "results": results}

    m = metric_list[0]
    arr = np.asarray(all_scores[m])
    return {
        "metric": m,
        "k": k,
        "scores": all_scores[m],
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=1)),
    }
