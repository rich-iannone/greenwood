"""Gradient-boosted competing-risks model: `SurvivalBoost`.

Estimates cause-specific cumulative incidence functions (CIFs) and the any-event survival function
using multi-class softmax gradient boosting with IPCW-weighted targets and stochastic time-horizon
sampling. Inspired by hazardous (Alberge et al. 2025). The implementation is pure NumPy and reuses
Greenwood's existing regression-tree infrastructure.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt
from typing_extensions import Self

from ._backends import to_dataframe
from ._boosting import _RegressionTree, _resolve_max_features
from ._competing import _censoring_km
from ._cox import _design_matrix

if TYPE_CHECKING:
    from ._surv import Surv

__all__ = ["SurvivalBoost"]

Array = npt.NDArray[Any]


def _softmax(scores: Array) -> Array:
    """Row-wise softmax: *scores* has shape `(n, K+1)`."""
    shifted = scores - scores.max(axis=1, keepdims=True)
    exp_s = np.exp(shifted)
    return exp_s / exp_s.sum(axis=1, keepdims=True)


def _ipcw_targets(
    event: Array,
    duration: Array,
    time_horizons: Array,
    g_times: Array,
    g_surv: Array,
    n_classes: int,
    epsilon: float,
) -> tuple[Array, Array]:
    """Compute IPCW-weighted multi-class targets for a batch of time horizons.

    Parameters
    ----------
    event
        Observed event codes, shape `(n,)`. `0` = censored, `1..K` = event types.
    duration
        Observed times, shape `(n,)`.
    time_horizons
        Sampled time horizons, shape `(n,)`.
    g_times, g_surv
        Censoring KM: times and corresponding survival probabilities.
    n_classes
        Number of classes (`K + 1`: survival class + `K` event types).
    epsilon
        Floor for censoring probabilities to avoid division by zero.

    Returns
    -------
    targets
        Integer class labels, shape `(n,)`. `0` = event-free at horizon.
    weights
        IPCW sample weights, shape `(n,)`. Zero for censored-before-horizon subjects.
    """
    n = len(event)
    targets = np.zeros(n, dtype=int)
    weights = np.zeros(n, dtype=float)

    for i in range(n):
        tau = time_horizons[i]
        t_i = duration[i]
        e_i = event[i]

        if t_i > tau:
            targets[i] = 0
            g_tau = _interp_censoring(tau, g_times, g_surv, epsilon)
            weights[i] = 1.0 / g_tau
        elif e_i > 0:
            targets[i] = int(e_i)
            g_ti = _interp_censoring(t_i, g_times, g_surv, epsilon)
            weights[i] = 1.0 / g_ti
        else:
            weights[i] = 0.0

    return targets, weights


def _interp_censoring(t: float, g_times: Array, g_surv: Array, epsilon: float) -> float:
    """Evaluate the censoring survival function G(t) by left-continuous step interpolation."""
    if t <= 0 or len(g_times) == 0:
        return 1.0
    idx = int(np.searchsorted(g_times, t, side="right")) - 1
    val = float(g_surv[idx]) if idx >= 0 else 1.0
    return max(val, epsilon)


class SurvivalBoost:
    """Gradient-boosted cumulative incidence model for competing risks.

    Builds an ensemble of regression trees that estimates the cause-specific cumulative incidence
    functions (CIFs) and the any-event survival function via multi-class softmax gradient boosting.
    At each boosting iteration, random time horizons are sampled and concatenated as an extra input
    feature. IPCW weights correct for right censoring. The result is a flexible, nonparametric
    competing-risks model that learns CIFs directly from the data.

    Parameters
    ----------
    n_estimators
        Number of boosting iterations. Each iteration fits one tree per class (event type +
        survival).
    learning_rate
        Shrinkage applied to each tree's contribution.
    max_depth
        Maximum depth of each regression tree.
    min_samples_leaf
        Minimum number of samples required in each leaf.
    subsample
        Fraction of the training rows sampled (without replacement) per iteration.
    max_features
        Features considered per split: `None` (all), `"sqrt"`, `"log2"`, an int, or a float.
    n_time_grid_steps
        Number of time points in the default prediction grid. The grid is built from quantiles of
        observed event times.
    hard_zero_fraction
        Fraction of sampled time horizons forced to zero at each iteration, ensuring the model
        learns zero incidence at `t = 0`.
    n_horizons_per_observation
        Number of independent time horizons sampled per subject at each iteration. Higher values
        give more stable gradients at the cost of larger per-iteration datasets.
    epsilon_censoring
        Floor for censoring survival probabilities to prevent extreme IPCW weights.
    random_state
        Seed or `numpy.random.Generator` for reproducibility.

    Examples
    --------
    Fit to synthetic competing-risks data and predict cumulative incidence:

    ```{python}
    import greenwood as gw

    sim = gw.simulate_competing_risks(n=500, n_causes=2, n_covariates=3, seed=42)

    sb = gw.SurvivalBoost(n_estimators=50, learning_rate=0.1, max_depth=3, random_state=0)
    sb.fit(sim.surv, sim.covariates)
    sb
    ```

    ```{python}
    # Predict survival and CIFs at specific times
    import numpy as np

    times = np.array([2.0, 5.0, 10.0])
    cif = sb.predict_cumulative_incidence(sim.covariates, times=times)
    cif.shape  # (n_subjects, n_events, n_times)
    ```

    ```{python}
    # Survival function
    surv = sb.predict_survival_function(sim.covariates, times=times)
    surv.shape  # (n_subjects, n_times)
    ```
    """

    def __init__(
        self,
        *,
        n_estimators: int = 100,
        learning_rate: float = 0.05,
        max_depth: int = 3,
        min_samples_leaf: int = 10,
        subsample: float = 1.0,
        max_features: Any = None,
        n_time_grid_steps: int = 100,
        hard_zero_fraction: float = 0.1,
        n_horizons_per_observation: int = 3,
        epsilon_censoring: float = 0.05,
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
        if n_time_grid_steps < 1:
            raise ValueError("n_time_grid_steps must be >= 1.")
        if not 0.0 <= hard_zero_fraction < 1.0:
            raise ValueError("hard_zero_fraction must be in [0, 1).")
        if n_horizons_per_observation < 1:
            raise ValueError("n_horizons_per_observation must be >= 1.")
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.subsample = subsample
        self.max_features = max_features
        self.n_time_grid_steps = n_time_grid_steps
        self.hard_zero_fraction = hard_zero_fraction
        self.n_horizons_per_observation = n_horizons_per_observation
        self.epsilon_censoring = epsilon_censoring
        self.random_state = random_state

    def __repr__(self) -> str:
        if not hasattr(self, "trees_"):
            return f"SurvivalBoost(n_estimators={self.n_estimators}) <unfitted>"
        return (
            f"SurvivalBoost ({self.n_estimators} rounds, "
            f"learning_rate={self.learning_rate}, max_depth={self.max_depth})\n"
            f"n = {self.n_}, events = {self.n_event_}, "
            f"causes = {self.n_causes_}, features = {self.n_features_in_}"
        )

    def fit(self, surv: Surv, covariates: Any, *, data: Any = None) -> Self:
        """Fit the gradient-boosted cumulative incidence model.

        Parameters
        ----------
        surv
            A multi-state `Surv` response (built with `Surv.multistate()`).
        covariates
            A dataframe, a 2-D array, or a right-hand-side formula string evaluated against `data`.
        data
            DataFrame used to evaluate a formula string.

        Returns
        -------
        self
            The fitted estimator.
        """
        if surv.states is None:
            raise NotImplementedError(
                "SurvivalBoost requires a multi-state response from Surv.multistate()."
            )

        x, names = _design_matrix(covariates, data)
        time = np.asarray(surv.stop, dtype=float)
        status = np.asarray(surv.status, dtype=int)
        if x.shape[0] != surv.n:
            raise ValueError("Covariates and response must have the same number of rows.")

        keep = ~np.isnan(x).any(axis=1)
        x, time, status = x[keep], time[keep], status[keep]
        n = x.shape[0]
        if not (status > 0).any():
            raise ValueError("No events remain after dropping missing rows.")

        states = surv.states
        assert states is not None  # guaranteed by check above
        n_causes = len(states)
        n_classes = n_causes + 1

        g_times, g_surv = _censoring_km(time, status)

        any_event_mask = status > 0
        observed_event_times = time[any_event_mask]
        if len(observed_event_times) > self.n_time_grid_steps:
            self.time_grid_ = np.quantile(
                observed_event_times, np.linspace(0, 1, num=self.n_time_grid_steps)
            )
        else:
            self.time_grid_ = np.sort(observed_event_times.copy())

        rng = np.random.default_rng(self.random_state)
        n_features_aug = x.shape[1] + 1
        n_features_split = _resolve_max_features(self.max_features, n_features_aug)

        trees: list[list[_RegressionTree]] = []
        scores = np.zeros((n, n_classes))

        t_max = float(time.max())

        for _ in range(self.n_estimators):
            x_aug_all = np.empty((0, n_features_aug))
            targets_all = np.empty(0, dtype=int)
            weights_all = np.empty(0)
            scores_all = np.empty((0, n_classes))

            for _ in range(self.n_horizons_per_observation):
                sampled = rng.uniform(0.0, t_max, size=n)
                n_hard = max(int(self.hard_zero_fraction * n), 1)
                hard_idx = rng.choice(n, size=n_hard, replace=False)
                sampled[hard_idx] = 0.0

                targets, weights = _ipcw_targets(
                    status, time, sampled, g_times, g_surv, n_classes, self.epsilon_censoring
                )

                x_aug = np.column_stack([sampled, x])
                x_aug_all = np.vstack([x_aug_all, x_aug])
                targets_all = np.concatenate([targets_all, targets])
                weights_all = np.concatenate([weights_all, weights])
                scores_all = np.vstack([scores_all, scores])

            mask = weights_all > 0
            if mask.sum() < 2:
                continue

            x_aug_active = x_aug_all[mask]
            targets_active = targets_all[mask]
            weights_active = weights_all[mask]
            scores_active = scores_all[mask]

            probs = _softmax(scores_active)

            if self.subsample < 1.0:
                m = max(2 * self.min_samples_leaf, int(round(self.subsample * mask.sum())))
                rows = rng.choice(mask.sum(), size=min(m, mask.sum()), replace=False)
            else:
                rows = np.arange(mask.sum())

            importances = np.zeros(n_features_aug)
            round_trees: list[_RegressionTree] = []

            for k in range(n_classes):
                one_hot = (targets_active == k).astype(float)
                residuals = weights_active * (one_hot - probs[:, k])

                tree = _RegressionTree(
                    max_depth=self.max_depth,
                    min_samples_leaf=self.min_samples_leaf,
                    n_features_split=n_features_split,
                    rng=rng,
                )
                tree.fit(x_aug_active[rows], residuals[rows], importances)
                round_trees.append(tree)

            trees.append(round_trees)

            for h_idx in range(self.n_horizons_per_observation):
                start = h_idx * n
                end = (h_idx + 1) * n
                x_aug_chunk = x_aug_all[start:end]
                for k in range(n_classes):
                    scores[:, k] += self.learning_rate * round_trees[k].predict(x_aug_chunk)

        self.trees_ = trees
        self.n_classes_ = n_classes
        self.n_causes_ = n_causes
        self.n_ = int(n)
        self.n_event_ = int((status > 0).sum())
        self.n_features_in_ = x.shape[1]
        self.feature_names_in_ = list(names)
        self.states_ = states
        return self

    def predict_cumulative_incidence(
        self,
        newdata: Any = None,
        *,
        times: Any = None,
        format: str | None = None,
    ) -> Any:
        """Predict cause-specific cumulative incidence functions.

        Parameters
        ----------
        newdata
            Covariates for prediction. A dataframe, 2-D array, or formula string. `None` is not
            supported (training covariates are not cached).
        times
            Time points at which to evaluate the CIFs. Defaults to the training-time grid.
        format
            When `None`, returns a 3-D NumPy array of shape `(n_subjects, n_events, n_times)`. Event
            index `0` is the survival probability (complement of all CIFs). Indices `1..K`
            correspond to cause-specific CIFs. When a DataFrame backend name is given (e.g.,
            `"polars"`), returns a long-format table with columns `time`, `cause`, `subject`, and
            `probability`.

        Returns
        -------
        numpy.ndarray or DataFrame
            A 3-D array or a long-format DataFrame.
        """
        if newdata is None:
            raise ValueError("newdata is required for SurvivalBoost prediction.")
        x = _design_matrix(newdata)[0]
        query = self.time_grid_ if times is None else np.atleast_1d(np.asarray(times, dtype=float))

        n_subj = x.shape[0]
        n_times = len(query)
        result = np.zeros((n_subj, self.n_classes_, n_times))

        for t_idx, t in enumerate(query):
            x_aug = np.column_stack([np.full(n_subj, t), x])
            scores = np.zeros((n_subj, self.n_classes_))
            for round_trees in self.trees_:
                for k in range(self.n_classes_):
                    scores[:, k] += self.learning_rate * round_trees[k].predict(x_aug)
            result[:, :, t_idx] = _softmax(scores)

        if format is None:
            return result

        rows: dict[str, list[Any]] = {"time": [], "cause": [], "subject": [], "probability": []}
        cause_names = ["survival"] + list(self.states_)
        for t_idx, t in enumerate(query):
            for k, cname in enumerate(cause_names):
                for i in range(n_subj):
                    rows["time"].append(float(t))
                    rows["cause"].append(cname)
                    rows["subject"].append(i + 1)
                    rows["probability"].append(float(result[i, k, t_idx]))
        return to_dataframe(
            {col: np.array(vals) if col != "cause" else vals for col, vals in rows.items()},
            format=format,
        )

    def predict_survival_function(
        self,
        newdata: Any = None,
        *,
        times: Any = None,
        format: str | None = None,
    ) -> Any:
        """Predict the any-event survival function.

        Parameters
        ----------
        newdata
            Covariates for prediction.
        times
            Time points at which to evaluate. Defaults to the training-time grid.
        format
            DataFrame backend for output. `None` returns a 2-D NumPy array of shape
            `(n_subjects, n_times)`.

        Returns
        -------
        numpy.ndarray or DataFrame
            Survival probabilities.
        """
        cif = self.predict_cumulative_incidence(newdata, times=times)
        survival = cif[:, 0, :]  # class 0 is "no event" = survival

        if format is None:
            return survival

        query = self.time_grid_ if times is None else np.atleast_1d(np.asarray(times, dtype=float))
        columns: dict[str, Any] = {"time": query}
        columns.update({f"subject_{i + 1}": survival[i, :] for i in range(survival.shape[0])})
        return to_dataframe(columns, format=format)

    def predict_proba(
        self,
        newdata: Any,
        *,
        time_horizon: float,
    ) -> Array:
        """Predict class probabilities at a single time horizon.

        Returns an array of shape `(n_subjects, n_classes)` where column 0 is the probability of
        remaining event-free and columns `1..K` are cause-specific CIFs at `time_horizon`.

        Parameters
        ----------
        newdata
            Covariates for prediction.
        time_horizon
            The time at which to evaluate probabilities.

        Returns
        -------
        numpy.ndarray
            Probabilities of shape `(n_subjects, n_classes)`.
        """
        cif = self.predict_cumulative_incidence(newdata, times=np.array([time_horizon]))
        return cif[:, :, 0]


def _tidy_sb(model: SurvivalBoost, *, format: str | None = None, **_: Any) -> Any:
    """broom-style `tidy`: feature names."""
    return to_dataframe(
        {"term": model.feature_names_in_},
        format=format,
    )


def _glance_sb(model: SurvivalBoost, *, format: str | None = None, **_: Any) -> Any:
    """broom-style `glance`: one-row model summary."""
    return to_dataframe(
        {
            "n": [model.n_],
            "nevent": [model.n_event_],
            "n_causes": [model.n_causes_],
            "n_estimators": [model.n_estimators],
            "learning_rate": [model.learning_rate],
            "max_depth": [model.max_depth],
            "n_features": [model.n_features_in_],
        },
        format=format,
    )


def _register_adapters() -> None:
    from .summaries import register_glance, register_tidier

    register_tidier("greenwood._survival_boost.SurvivalBoost", _tidy_sb)
    register_glance("greenwood._survival_boost.SurvivalBoost", _glance_sb)


_register_adapters()
