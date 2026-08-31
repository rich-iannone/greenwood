"""Gradient-boosted survival analysis: `GradientBoostingSurvivalAnalysis`.

Tree-based gradient boosting that minimizes the negative Cox partial log-likelihood. At each
boosting iteration a squared-error regression tree is fit to the negative gradient of the Cox loss
(the martingale residuals under the current model), and its scaled predictions are added to the
additive risk score `F(x)`. The final `F` plays the role of the Cox linear predictor: a Breslow
baseline hazard estimated from the training data then turns it into per-subject survival and
cumulative-hazard functions.

The implementation is pure NumPy (no scikit-learn / scikit-survival dependency) and follows the
component of Ridgeway (1999) / Ishwaran-style boosting used by scikit-survival's
`GradientBoostingSurvivalAnalysis(loss="coxph")`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt
from typing_extensions import Self

from ._backends import to_dataframe
from ._cox import _design_matrix

if TYPE_CHECKING:
    from ._surv import Surv

__all__ = ["GradientBoostingSurvivalAnalysis"]

Array = npt.NDArray[Any]

_PREDICT_TYPES = ("risk", "lp", "survival", "cumulative_hazard")


def _resolve_max_features(max_features: Any, n_features: int) -> int:
    """Translate a `max_features` specification into a concrete feature count in `[1, n]`."""
    if max_features is None:
        value = n_features
    elif isinstance(max_features, str):
        if max_features == "sqrt":
            value = int(np.sqrt(n_features))
        elif max_features == "log2":
            value = int(np.log2(n_features))
        else:
            raise ValueError(
                f"Unknown max_features {max_features!r}; use 'sqrt', 'log2', an int, a float, "
                "or None."
            )
    elif isinstance(max_features, (int, np.integer)):
        value = int(max_features)
    elif isinstance(max_features, float):
        value = int(max_features * n_features)
    else:
        raise TypeError("max_features must be 'sqrt', 'log2', an int, a float, or None.")
    return int(np.clip(value, 1, n_features))


def _cox_gradient_baseline(
    scores: Array, time: Array, event: Array
) -> tuple[Array, Array, Array, Array]:
    """Cox martingale residuals and Breslow baseline for the current additive scores.

    Returns `(residuals, event_times, baseline_cumhaz, is_event_time)`, where `residuals` are the
    negative gradient of the negative Cox partial log-likelihood (i.e. the martingale residuals),
    and `baseline_cumhaz` is the Breslow cumulative hazard at each unique time.
    """
    exp_f = np.exp(scores)
    order = np.argsort(time, kind="mergesort")
    t_sorted = time[order]
    exp_sorted = exp_f[order]
    event_sorted = event[order].astype(float)

    uniq, starts = np.unique(t_sorted, return_index=True)

    # Risk-set sum at each unique time = sum of exp(score) over subjects with t_j >= t_k.
    suffix = np.cumsum(exp_sorted[::-1])[::-1]
    risk_sum = suffix[starts]
    d = np.add.reduceat(event_sorted, starts)
    with np.errstate(divide="ignore", invalid="ignore"):
        dh0 = np.where(risk_sum > 0, d / risk_sum, 0.0)
    baseline_cumhaz = np.cumsum(dh0)

    inv = np.searchsorted(uniq, time)
    cumhaz_i = baseline_cumhaz[inv]
    residuals = event.astype(float) - exp_f * cumhaz_i
    return residuals, uniq, baseline_cumhaz, d > 0


class _RegressionNode:
    """A node of a squared-error regression tree: an internal split or a leaf value."""

    __slots__ = ("feature", "threshold", "left", "right", "value")

    def __init__(self) -> None:
        self.feature: int = -1
        self.threshold: float = np.nan
        self.left: _RegressionNode | None = None
        self.right: _RegressionNode | None = None
        self.value: float = 0.0

    @property
    def is_leaf(self) -> bool:
        return self.left is None


def _best_sse_split(
    x: Array, y: Array, features: Array, min_samples_leaf: int
) -> tuple[int, float, float] | None:
    """Best squared-error split over candidate features, returning `(feature, threshold, gain)`.

    `gain` is the reduction in total squared error, `sum_L^2/n_L + sum_R^2/n_R - sum^2/n`.
    """
    n = x.shape[0]
    total = float(y.sum())
    parent = total * total / n
    best: tuple[int, float, float] | None = None
    for feat in features:
        col = x[:, feat]
        order = np.argsort(col, kind="mergesort")
        col_sorted = col[order]
        y_sorted = y[order]
        csum = np.cumsum(y_sorted)[:-1]
        n_left = np.arange(1, n)
        n_right = n - n_left
        objective = csum * csum / n_left + (total - csum) ** 2 / n_right
        distinct = col_sorted[:-1] < col_sorted[1:]
        valid = distinct & (n_left >= min_samples_leaf) & (n_right >= min_samples_leaf)
        if not valid.any():
            continue
        masked = np.where(valid, objective, -np.inf)
        p = int(np.argmax(masked))
        gain = float(masked[p] - parent)
        if best is None or gain > best[2]:
            threshold = 0.5 * (col_sorted[p] + col_sorted[p + 1])
            best = (int(feat), float(threshold), gain)
    return best


class _RegressionTree:
    """A minimal squared-error regression tree used as a boosting base learner."""

    def __init__(
        self,
        *,
        max_depth: int,
        min_samples_leaf: int,
        n_features_split: int,
        rng: np.random.Generator,
    ) -> None:
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.n_features_split = n_features_split
        self.rng = rng
        self.n_features_in_ = 0

    def fit(self, x: Array, y: Array, importances: Array) -> _RegressionTree:
        self.n_features_in_ = x.shape[1]
        self._root = self._grow(x, y, depth=0, importances=importances)
        return self

    def _grow(self, x: Array, y: Array, *, depth: int, importances: Array) -> _RegressionNode:
        node = _RegressionNode()
        n = x.shape[0]
        node.value = float(y.mean()) if n else 0.0
        if depth < self.max_depth and n >= 2 * self.min_samples_leaf and np.ptp(y) > 0:
            features = self.rng.choice(
                self.n_features_in_, size=self.n_features_split, replace=False
            )
            split = _best_sse_split(x, y, features, self.min_samples_leaf)
            if split is not None and split[2] > 0:
                feat, threshold, gain = split
                importances[feat] += gain
                mask = x[:, feat] <= threshold
                node.feature = feat
                node.threshold = threshold
                node.left = self._grow(x[mask], y[mask], depth=depth + 1, importances=importances)
                node.right = self._grow(
                    x[~mask], y[~mask], depth=depth + 1, importances=importances
                )
        return node

    def predict(self, x: Array) -> Array:
        out = np.empty(x.shape[0])
        for i in range(x.shape[0]):
            node = self._root
            row = x[i]
            while not node.is_leaf:
                assert node.left is not None and node.right is not None
                node = node.left if row[node.feature] <= node.threshold else node.right
            out[i] = node.value
        return out


class GradientBoostingSurvivalAnalysis:
    """Gradient-boosted survival model minimizing the Cox partial-likelihood loss.

    Builds an additive risk score by sequentially fitting shallow squared-error regression trees to
    the negative gradient of the negative Cox partial log-likelihood (the martingale residuals under
    the current model) and adding a shrunken version of each tree. The result is a flexible,
    non-linear generalization of the Cox model: strong predictive performance on structured tabular
    data, without a global proportional-hazards assumption on individual covariates. A Breslow
    baseline hazard turns the additive log-risk score into per-subject survival and
    cumulative-hazard functions.

    Parameters
    ----------
    n_estimators
        Number of boosting iterations (trees).
    learning_rate
        Shrinkage applied to each tree's contribution. Smaller values need more trees but
        generalize better.
    max_depth
        Maximum depth of each regression tree (interaction depth).
    min_samples_leaf
        Minimum number of samples in each leaf of a tree.
    subsample
        Fraction of the training rows sampled (without replacement) to grow each tree. Values
        below `1.0` give stochastic gradient boosting.
    max_features
        Covariates considered per split: `None` (all, default), `"sqrt"`, `"log2"`, an int, or a
        float fraction.
    random_state
        Seed or `numpy.random.Generator` for subsampling and per-node feature selection.

    Examples
    --------
    Fit a gradient-boosted survival model on the bundled `lung` dataset:

    ```{python}
    import greenwood as gw

    # Load data and build a right-censored response
    lung = gw.load_dataset("lung", backend="pandas").dropna(subset=["ph.ecog", "ph.karno"])
    y = gw.Surv.right(lung["time"], event=(lung["status"] == 2))
    cols = ["age", "sex", "ph.ecog", "ph.karno", "wt.loss"]

    # Fit the model and score the first five subjects
    gbm = gw.GradientBoostingSurvivalAnalysis(
        n_estimators=200, learning_rate=0.05, max_depth=2, random_state=0
    ).fit(y, lung[cols])
    gbm.predict(lung[cols])[:5]
    ```
    """

    def __init__(
        self,
        *,
        n_estimators: int = 100,
        learning_rate: float = 0.1,
        max_depth: int = 3,
        min_samples_leaf: int = 3,
        subsample: float = 1.0,
        max_features: Any = None,
        random_state: Any = None,
    ) -> None:
        if n_estimators < 1:
            raise ValueError("n_estimators must be >= 1.")
        if learning_rate <= 0.0:
            raise ValueError("learning_rate must be > 0.")
        if max_depth < 1:
            raise ValueError("max_depth must be >= 1.")
        if min_samples_leaf < 1:
            raise ValueError("min_samples_leaf must be >= 1.")
        if not 0.0 < subsample <= 1.0:
            raise ValueError("subsample must be in (0, 1].")
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.subsample = subsample
        self.max_features = max_features
        self.random_state = random_state

    def __repr__(self) -> str:
        if getattr(self, "trees_", None) is None:
            return f"GradientBoostingSurvivalAnalysis(n_estimators={self.n_estimators}) <unfitted>"
        return (
            f"GradientBoostingSurvivalAnalysis ({self.n_estimators} trees, "
            f"learning_rate={self.learning_rate}, max_depth={self.max_depth})\n"
            f"n = {self.n_}, events = {self.n_event_}, features = {self.n_features_in_}"
        )

    def fit(self, surv: Surv, covariates: Any, *, data: Any = None) -> Self:
        """Fit the gradient-boosted survival model to a right-censored response.

        Parameters
        ----------
        surv
            A right-censored `Surv` response (built with `Surv.right()`).
        covariates
            A dataframe, a 2-D array, or a right-hand-side formula string evaluated against `data`.
        data
            DataFrame used to evaluate a formula string.

        Returns
        -------
        self
            The fitted estimator, with cached attributes including `trees_`, `event_times_`,
            `feature_importances_`, `n_features_in_`, and `feature_names_in_`.
        """
        from ._surv import CensoringType

        if surv.type != CensoringType.RIGHT:
            raise NotImplementedError(
                f"GradientBoostingSurvivalAnalysis supports right-censored responses, "
                f"not {surv.type.value!r}."
            )
        x, names = _design_matrix(covariates, data)
        time = np.asarray(surv.stop, dtype=float)
        event = np.asarray(surv.event, dtype=bool)
        if x.shape[0] != surv.n:
            raise ValueError("Covariates and response must have the same number of rows.")
        keep = ~np.isnan(x).any(axis=1)
        x, time, event = x[keep], time[keep], event[keep]
        n = x.shape[0]
        if not event.any():
            raise ValueError("No events remain after dropping missing rows.")

        rng = np.random.default_rng(self.random_state)
        n_features_split = _resolve_max_features(self.max_features, x.shape[1])
        importances = np.zeros(x.shape[1])

        scores = np.zeros(n)
        trees: list[tuple[_RegressionTree, float]] = []
        for _ in range(self.n_estimators):
            residuals, _, _, _ = _cox_gradient_baseline(scores, time, event)
            if self.subsample < 1.0:
                m = max(2 * self.min_samples_leaf, int(round(self.subsample * n)))
                rows = rng.choice(n, size=min(m, n), replace=False)
            else:
                rows = np.arange(n)
            tree = _RegressionTree(
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
                n_features_split=n_features_split,
                rng=rng,
            )
            tree.fit(x[rows], residuals[rows], importances)
            scores += self.learning_rate * tree.predict(x)
            trees.append((tree, self.learning_rate))

        self.trees_ = trees
        self.feature_names_in_ = list(names)
        self.n_features_in_ = x.shape[1]
        self.n_ = int(n)
        self.n_event_ = int(event.sum())
        total = importances.sum()
        self.feature_importances_ = importances / total if total > 0 else importances

        # Breslow baseline from the final additive scores, restricted to event times.
        _, uniq, baseline_cumhaz, is_event = _cox_gradient_baseline(scores, time, event)
        self.event_times_ = uniq[is_event]
        self._baseline_cumhaz = baseline_cumhaz[is_event]
        self._train_scores = scores
        return self

    def _score(self, x: Array) -> Array:
        scores = np.zeros(x.shape[0])
        for tree, lr in self.trees_:
            scores += lr * tree.predict(x)
        return scores

    def predict(
        self,
        newdata: Any = None,
        *,
        type: str = "risk",
        times: Any = None,
        format: str | None = None,
    ) -> Any:
        r"""Predict a risk score, linear predictor, or survival / cumulative-hazard curves.

        1. `type="risk"` (default): the relative hazard $\exp(F(x))$, a 1-D array where higher
        values indicate higher risk.
        2. `type="lp"`: the additive log-risk score $F(x)$ (the boosted Cox linear predictor).
        3. `type="survival"`: survival $S(t \mid x) = \exp(-H_0(t)\,e^{F(x)})$ at `times`.
        4. `type="cumulative_hazard"`: $H(t \mid x) = H_0(t)\,e^{F(x)}$ at `times`.

        Parameters
        ----------
        newdata
            Covariates to predict for. `None` predicts for the training subjects.
        type
            One of `"risk"`, `"lp"`, `"survival"`, or `"cumulative_hazard"`.
        times
            Times at which to evaluate curves. Defaults to the training event times.
        format
            DataFrame backend for curve output.

        Returns
        -------
        numpy.ndarray or DataFrame
            A risk / linear-predictor vector, or a curve frame with a `time` column and one column
            per subject.
        """
        if type not in _PREDICT_TYPES:
            raise ValueError(f"Unknown predict type {type!r}; use one of {_PREDICT_TYPES}.")
        scores = self._train_scores if newdata is None else self._score(_design_matrix(newdata)[0])
        if type == "lp":
            return scores
        if type == "risk":
            return np.exp(scores)

        query = (
            self.event_times_ if times is None else np.atleast_1d(np.asarray(times, dtype=float))
        )
        idx = np.searchsorted(self.event_times_, query, side="right") - 1
        h0 = np.where(idx >= 0, self._baseline_cumhaz[idx.clip(min=0)], 0.0)
        relative = np.exp(scores)
        cumhaz = np.outer(h0, relative)  # (n_times, n_subjects)
        curves = np.exp(-cumhaz) if type == "survival" else cumhaz
        columns: dict[str, Any] = {"time": query}
        columns.update({f"subject_{i + 1}": curves[:, i] for i in range(curves.shape[1])})
        return to_dataframe(columns, format=format)

    def variable_importance(self, *, format: str | None = None) -> Any:
        """Impurity-based variable importance (normalized total squared-error reduction).

        Each split's reduction in squared error is credited to its covariate and summed across all
        trees, then normalized to sum to one. Returns a table sorted by importance.
        """
        order = np.argsort(self.feature_importances_)[::-1]
        columns = {
            "term": [self.feature_names_in_[i] for i in order],
            "importance": self.feature_importances_[order],
        }
        return to_dataframe(columns, format=format)


def _tidy_gbm(
    model: GradientBoostingSurvivalAnalysis, *, format: str | None = None, **_: Any
) -> Any:
    """broom-style `tidy`: one row per covariate with its impurity-based importance."""
    return model.variable_importance(format=format)


def _glance_gbm(
    model: GradientBoostingSurvivalAnalysis, *, format: str | None = None, **_: Any
) -> Any:
    """broom-style `glance`: one-row model summary."""
    return to_dataframe(
        {
            "n": [model.n_],
            "nevent": [model.n_event_],
            "n_estimators": [model.n_estimators],
            "learning_rate": [model.learning_rate],
            "max_depth": [model.max_depth],
            "n_features": [model.n_features_in_],
        },
        format=format,
    )


def _register_adapters() -> None:
    from .summaries import register_glance, register_tidier

    register_tidier("greenwood._boosting.GradientBoostingSurvivalAnalysis", _tidy_gbm)
    register_glance("greenwood._boosting.GradientBoostingSurvivalAnalysis", _glance_gbm)


_register_adapters()
