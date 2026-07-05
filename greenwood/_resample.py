"""K-fold cross-validation for honest, out-of-sample survival-model evaluation.

Fitting a model and scoring it on the same data is optimistic. `cross_validate` splits the
data into folds, fits on the training folds, and scores predictions on the held-out fold,
using the censoring-aware metrics in `greenwood._metrics`.
"""

from __future__ import annotations

import copy
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
    seed: int | None = None,
) -> dict[str, Any]:
    """Evaluate a survival model out-of-sample with k-fold cross-validation.

    Parameters
    ----------
    model
        An unfitted estimator instance (for example `CoxPH()` or `AFT("weibull")`). A fresh
        copy is fit on each training split, so the passed object is left untouched.
    surv
        The `Surv` response (right-censored or counting-process).
    covariates
        A dataframe or 2-D array, or a right-hand-side formula string evaluated against
        `data` (as in `CoxPH.fit`). The design is resolved once, then split by fold.
    data
        The data frame a formula string is evaluated against.
    k
        Number of folds (default 5).
    metric
        `"concordance"` (Harrell's C on the held-out fold, needs a CoxPH or AFT model) or
        `"brier"` (integrated IPCW Brier score, lower is better; requires `times`).
    times
        For `metric="brier"`, at least two evaluation times.
    seed
        Seed for the fold shuffle, for reproducibility.

    Returns
    -------
    dict
        `{"metric", "k", "scores" (per fold), "mean", "std"}`.

    Examples
    --------
    Fit-and-score on the same data is optimistic; cross-validation gives an honest,
    out-of-sample estimate. Here five-fold concordance for a Cox model on the bundled `lung`
    dataset. The returned dict carries the per-fold `scores` alongside their `mean` and `std`.

    ```{python}
    import greenwood as gw
    from greenwood import Surv

    lung = gw.load_dataset("lung")
    y = Surv.right(lung["time"], event=(lung["status"] == 2))

    gw.cross_validate(gw.CoxPH(), y, lung[["age", "sex"]], k=5, metric="concordance", seed=1)
    ```

    Pass `metric="brier"` with a `times=` grid instead to score the integrated
    inverse-probability-of-censoring-weighted Brier score (lower is better).
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

    folds = np.array_split(np.random.default_rng(seed).permutation(surv.n), k)
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
            probs = frame.iloc[:, 1:].to_numpy().T  # (n_test, n_times)
            scores.append(float(integrated_brier_score(surv_test, probs, brier_times)))

    arr = np.asarray(scores)
    return {
        "metric": metric,
        "k": k,
        "scores": scores,
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=1)),
    }
