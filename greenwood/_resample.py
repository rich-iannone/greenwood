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
    from ._parametric import AFT
    from ._penalized import CoxNet

    if isinstance(model, (CoxPH, CoxNet)):
        return model.predict(x, type="lp")
    if isinstance(model, AFT):
        # The AFT linear predictor is the log-time location: larger means longer survival,
        # so negate it to get a risk score.
        return -model.predict(x, type="lp")
    raise TypeError(
        f"cross_validate with metric='concordance' needs a CoxPH, CoxNet, or AFT model, "
        f"got {type(model).__name__}."
    )


def cross_validate(
    model: Any,
    surv: Surv,
    covariates: Any,
    *,
    data: Any = None,
    k: int = 5,
    metric: str = "concordance",
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

    - `"concordance"` (default): Harrell's C-statistic on the test fold. Higher is better
      (0.5 = random, 1.0 = perfect). Requires CoxPH, CoxNet, or AFT model.
    - `"brier"`: Integrated IPCW Brier score over specified times. Lower is better
      (0 = perfect calibration, 1 = worst). Requires explicit `times=` parameter.

    Parameters
    ----------
    model
        An unfitted estimator instance (e.g., `CoxPH()`, `CoxNet()`, `AFT("weibull")`).
        A fresh copy is fit on each training fold, leaving the passed object unchanged.
        Supported: CoxPH, CoxNet, AFT (for concordance) and any of those (for Brier).
    surv
        A `Surv` response (time-to-event data). Can be right-censored or counting-process.
        Weights in the response are carried through the cross-validation.
    covariates
        Covariates/predictors for the model. Can be:

        - A 2-D array or pandas/Polars DataFrame with one row per subject
        - A formula string (as in `CoxPH.fit()`), evaluated against `data`

    data
        If `covariates` is a formula string, the data frame to evaluate it against.
    k
        Number of folds (default 5). Each fold serves as test data once; subjects are split
        randomly and evenly across folds. Typical choices: 5 or 10.
    metric
        Performance metric for evaluation:

        - `"concordance"` (default): Harrell's C-statistic. Requires CoxPH, CoxNet, or AFT.
        - `"brier"`: Integrated inverse-probability-of-censoring-weighted (IPCW) Brier
          score. Requires `times=` with at least 2 time points.

    times
        For `metric="brier"`, evaluation time points (1-D array-like, length $\ge 2$). The Brier
        score is computed at each time, then integrated (time-averaged). Example:
        `times=[365, 730, 1095]` for 1, 2, 3-year predictions.
    stratified
        If `True` (default), use stratified k-fold ensuring balanced event/censoring
        representation across folds. This prevents singular matrix errors and biased CV
        estimates on imbalanced survival data (rare events). If `False`, use simple random
        k-fold shuffling.
    seed
        Random seed for fold shuffling, ensures reproducibility. If `None`, results may vary
        between runs. Use a fixed seed for consistent comparisons.

    Returns
    -------
    dict
        Dictionary with keys:

        - `"metric"`: Metric name used (`"concordance"` or `"brier"`).
        - `"k"`: Number of folds.
        - `"scores"`: List of per-fold scores (one per fold).
        - `"mean"`: Mean score across folds (primary summary).
        - `"std"`: Standard deviation of scores (variability estimate).

        For concordance, higher mean is better. For Brier, lower mean is better.

    Details
    -------
    **How folds work**: By default (`stratified=True`), subjects are grouped by event status
    (censored vs. event, or multiple event types), then randomly shuffled within each stratum
    and split into k roughly equal-sized groups. This ensures each fold has approximately the
    same proportion of events and censored observations as the overall dataset. This is crucial
    for imbalanced data (e.g., rare events) to prevent singular matrix errors and ensures
    unbiased cross-validation estimates.

    If `stratified=False`, subjects are simply shuffled and split randomly, which may lead to
    folds with very different event rates and can destabilize model fitting on sparse data.

    **Completeness**: Subjects with missing covariates are dropped before folding. This
    ensures all folds use the same cleaned data, avoiding alignment issues.

    **AFT model note**: For AFT, concordance uses the negated linear predictor (since in AFT,
    larger lp means longer survival, opposite to Cox). This is handled automatically.

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

    Use Brier score (calibration) instead of concordance (discrimination):

    ```{python}
    # Evaluate calibration with the integrated Brier score
    result_brier = gw.cross_validate(
        gw.CoxPH(), y, lung[["age", "sex"]], k=5,
        metric="brier", times=[180, 365, 540], seed=1
    )
    result_brier
    ```

    Compare two models via cross-validation. Model with higher mean concordance (or lower
    mean Brier) generalizes better:

    ```{python}
    # Compare a simple vs. complex model (uncomment to run)
    # simple_model = gw.CoxPH()
    # complex_model = gw.CoxPH()
    # simple_cv = gw.cross_validate(simple_model, y, lung[["age"]], seed=1)
    # complex_cv = gw.cross_validate(complex_model, y, lung[["age", "sex", "ph.ecog"]], seed=1)
    # print(f"Simple model C-index: {simple_cv['mean']:.3f} ± {simple_cv['std']:.3f}")
    # print(f"Complex model C-index: {complex_cv['mean']:.3f} ± {complex_cv['std']:.3f}")
    ```
    """
    from ._cox import CoxPH, _design_matrix
    from ._metrics import concordance_index, integrated_brier_score
    from ._parametric import AFT
    from ._penalized import CoxNet

    if not isinstance(model, (CoxPH, CoxNet, AFT)):
        raise TypeError(
            f"cross_validate needs a CoxPH, CoxNet, or AFT model, got {type(model).__name__}."
        )
    if k < 2:
        raise ValueError("k must be at least 2.")
    if metric not in ("concordance", "brier"):
        raise ValueError(f"Unknown metric {metric!r}; use 'concordance' or 'brier'.")

    design, _ = _design_matrix(covariates, data)
    if design.shape[0] != surv.n:
        raise ValueError("Covariates and response must have the same number of rows.")

    # Complete-case: drop rows with a missing covariate once, up front, so every fold's
    # train and test sets are aligned with the response.
    keep = ~np.isnan(design).any(axis=1)
    if not keep.all():
        design = design[keep]
        surv = _subset_surv(surv, np.nonzero(keep)[0])

    brier_times: list[float] = []
    if metric == "brier":
        brier_times = [float(t) for t in np.atleast_1d(np.asarray(times, dtype=float))]
        if len(brier_times) < 2:
            raise ValueError("metric='brier' requires `times` with at least two time points.")

    folds = (
        _stratified_kfold_indices(surv, k, seed)
        if stratified
        else np.array_split(np.random.default_rng(seed).permutation(surv.n), k)
    )

    # Warn when there are very few events relative to the number of folds.  With fewer
    # than k events in total, some test folds will contain zero events, making concordance
    # undefined and Brier score unreliable.  The practical threshold for a reliable
    # per-fold estimate is at least ~5 events per fold, so we warn at < 2 * k.
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

    scores: list[float] = []
    for i in range(k):
        test = folds[i]
        train = np.concatenate([folds[j] for j in range(k) if j != i])
        fold_model = copy.deepcopy(model)
        fold_model.fit(_subset_surv(surv, train), design[train])
        surv_test = _subset_surv(surv, test)
        x_test = design[test]
        if metric == "concordance":
            scores.append(float(concordance_index(surv_test, _risk_score(fold_model, x_test))))
        else:
            frame = fold_model.predict(x_test, type="survival", times=brier_times)
            # Extract subject columns (skip first column which is "time")
            # Works with pandas, polars, or pyarrow without requiring pandas
            try:
                import polars as pl

                if isinstance(frame, pl.DataFrame):
                    probs = frame[:, 1:].to_numpy().T
                else:
                    probs = frame.iloc[:, 1:].to_numpy().T  # pragma: no cover
            except (ImportError, AttributeError):  # pragma: no cover
                cols = list(frame.columns)  # pragma: no cover
                probs = frame[cols[1:]].to_numpy().T  # pragma: no cover
            scores.append(float(integrated_brier_score(surv_test, probs, brier_times)))

    arr = np.asarray(scores)
    return {
        "metric": metric,
        "k": k,
        "scores": scores,
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=1)),
    }
