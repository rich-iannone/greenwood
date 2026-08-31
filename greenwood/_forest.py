"""Tree-based survival estimators: `SurvivalTree`, `RandomSurvivalForest`, `ExtraSurvivalTrees`.

These are fully non-parametric, machine-learning survival models. A `SurvivalTree` recursively
partitions the covariate space using the two-sample log-rank statistic as its split criterion
(the Ishwaran rule): at each node it chooses the covariate and threshold that most strongly
separate survival between the two child groups. Each leaf stores non-parametric estimates for the
subjects that fall into it: a Nelson-Aalen cumulative hazard and a Kaplan-Meier survival curve,
evaluated on the training set's event-time grid so that trees share a common set of time points.

A `RandomSurvivalForest` is a bootstrap-aggregated ensemble of such trees, each grown on a
bootstrap sample and considering a random subset of covariates at each split. `ExtraSurvivalTrees`
is an extremely-randomized variant that also draws the split threshold at random, trading a little
bias for lower variance and faster fits. Predictions average the per-tree cumulative-hazard
functions; the scalar risk score is Ishwaran's ensemble mortality, the sum of the ensemble
cumulative hazard over the event-time grid. An out-of-bag concordance estimate and
permutation-based variable importance are available.

The implementation is pure NumPy (no scikit-learn / scikit-survival dependency) and follows the
model of Ishwaran et al. (2008), "Random Survival Forests", Annals of Applied Statistics.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt
from typing_extensions import Self

from ._backends import to_dataframe
from ._cox import _design_matrix

if TYPE_CHECKING:
    from ._surv import Surv

__all__ = ["SurvivalTree", "RandomSurvivalForest", "ExtraSurvivalTrees"]

Array = npt.NDArray[Any]

_PREDICT_TYPES = ("risk", "survival", "cumulative_hazard")

# Optional Numba acceleration for the log-rank split search. When Numba is installed (the `fast`
# extra) and requested via `engine=`, the best-split search runs a compiled kernel that avoids
# materializing the O(n x m) risk/event matrices, which is the dominant cost on large datasets.
# The default `engine="numpy"` keeps the deterministic, environment-independent vectorized path.
_njit: Any = None
try:
    _njit = importlib.import_module("numba").njit
except Exception:  # pragma: no cover - exercised only when Numba is absent
    _njit = None
_HAS_NUMBA = _njit is not None

_ENGINES = ("numpy", "numba", "auto")


def _resolve_engine(engine: str) -> bool:
    """Resolve an `engine` selection to a boolean: use the Numba kernel (`True`) or NumPy (`False`).

    `"numpy"` always uses the vectorized NumPy path; `"numba"` requires Numba to be installed;
    `"auto"` uses Numba when available and falls back to NumPy otherwise.
    """
    if engine not in _ENGINES:
        raise ValueError(f"engine must be one of {_ENGINES}, got {engine!r}.")
    if engine == "numpy":
        return False
    if engine == "numba":
        if not _HAS_NUMBA:
            raise ImportError(
                "engine='numba' requires the `numba` package. Install it with "
                "`pip install greenwood[fast]`, or use engine='numpy' (the default)."
            )
        return True
    return _HAS_NUMBA  # "auto"


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


def _prepare_response(surv: Surv) -> tuple[Array, Array]:
    """Return `(time, event)` NumPy arrays from a right-censored `Surv`, validating support."""
    from ._surv import CensoringType

    if surv.type != CensoringType.RIGHT:
        raise NotImplementedError(
            f"Survival trees support right-censored responses, not {surv.type.value!r}."
        )
    return np.asarray(surv.stop, dtype=float), np.asarray(surv.event, dtype=bool)


class _RiskEventGrid:
    """At-risk and event counts for a set of samples on a fixed event-time grid.

    Precomputes, for the samples in a node, the boolean at-risk matrix and event matrix against the
    node's own unique event times. The log-rank split search reuses these to evaluate every
    candidate threshold of a feature in a single vectorized pass.
    """

    __slots__ = ("times", "at_risk", "events", "n_tot", "d_tot", "var_factor")

    def __init__(self, time: Array, event: Array) -> None:
        event_times = np.unique(time[event])
        self.times = event_times
        # at_risk[j, i] = subject j is at risk at event time i (time_j >= tau_i)
        self.at_risk = time[:, None] >= event_times[None, :]
        # events[j, i] = subject j has an event exactly at tau_i
        self.events = (time[:, None] == event_times[None, :]) & event[:, None]
        self.n_tot = self.at_risk.sum(axis=0).astype(float)
        self.d_tot = self.events.sum(axis=0).astype(float)
        # Hypergeometric variance factor d_i (n_i - d_i) / (n_i - 1), zero when n_i <= 1.
        with np.errstate(divide="ignore", invalid="ignore"):
            factor = self.d_tot * (self.n_tot - self.d_tot) / (self.n_tot - 1.0)
        self.var_factor = np.where(self.n_tot > 1.0, factor, 0.0)


def _best_logrank_split(
    grid: _RiskEventGrid,
    x: Array,
    features: Array,
    min_samples_leaf: int,
) -> tuple[int, float, float] | None:
    """Find the (feature, threshold) maximizing the log-rank chi-square over candidate features.

    Returns `(feature_index, threshold, chi_square)` for the best split, or `None` if no valid
    split (respecting `min_samples_leaf` and requiring distinct feature values) exists.
    """
    n = x.shape[0]
    if grid.times.size == 0:
        return None
    n_tot, d_tot, var_factor = grid.n_tot, grid.d_tot, grid.var_factor
    best: tuple[int, float, float] | None = None
    for feat in features:
        col = x[:, feat]
        order = np.argsort(col, kind="mergesort")
        col_sorted = col[order]
        # Cumulative at-risk / event counts for the "left" group = first p+1 sorted samples.
        n1 = np.cumsum(grid.at_risk[order], axis=0).astype(float)  # (n, m)
        d1 = np.cumsum(grid.events[order], axis=0).astype(float)  # (n, m)
        frac = np.divide(n1, n_tot, out=np.zeros_like(n1), where=n_tot > 0)
        u = (d1 - frac * d_tot).sum(axis=1)  # (n,) log-rank numerator per split
        v = (var_factor * frac * (1.0 - frac)).sum(axis=1)  # (n,) variance per split
        chi = np.divide(u * u, v, out=np.zeros_like(u), where=v > 0)
        # A split after sorted position p is valid when the feature value strictly increases
        # there (so the threshold separates the two groups) and both sides meet the leaf minimum.
        distinct = col_sorted[:-1] < col_sorted[1:]
        left_size = np.arange(1, n)
        valid = distinct & (left_size >= min_samples_leaf) & (n - left_size >= min_samples_leaf)
        if not valid.any():
            continue
        chi_valid = np.where(valid, chi[:-1], -np.inf)
        p = int(np.argmax(chi_valid))
        if not np.isfinite(chi_valid[p]):  # pragma: no cover - argmax over valid is always finite
            continue
        score = float(chi_valid[p])
        if best is None or score > best[2]:
            threshold = 0.5 * (col_sorted[p] + col_sorted[p + 1])
            best = (int(feat), float(threshold), score)
    return best


def _random_logrank_split(
    grid: _RiskEventGrid,
    x: Array,
    features: Array,
    min_samples_leaf: int,
    rng: np.random.Generator,
) -> tuple[int, float, float] | None:
    """Extra-trees split: draw one random threshold per feature, keep the best log-rank score.

    For each candidate feature a single cut-point is sampled uniformly between the feature's minimum
    and maximum at the node (the extremely-randomized-trees rule). The feature whose random split
    yields the largest log-rank chi-square is returned, or `None` if none is valid.
    """
    n = x.shape[0]
    if grid.times.size == 0:
        return None
    n_tot, d_tot, var_factor = grid.n_tot, grid.d_tot, grid.var_factor
    best: tuple[int, float, float] | None = None
    for feat in features:
        col = x[:, feat]
        lo, hi = float(col.min()), float(col.max())
        if lo >= hi:
            continue  # constant feature at this node
        threshold = float(rng.uniform(lo, hi))
        mask = col <= threshold
        left_size = int(mask.sum())
        if left_size < min_samples_leaf or n - left_size < min_samples_leaf:
            continue
        n1 = grid.at_risk[mask].sum(axis=0).astype(float)
        d1 = grid.events[mask].sum(axis=0).astype(float)
        frac = np.divide(n1, n_tot, out=np.zeros_like(n1), where=n_tot > 0)
        u = float((d1 - frac * d_tot).sum())
        v = float((var_factor * frac * (1.0 - frac)).sum())
        if v <= 0:
            continue
        score = u * u / v
        if best is None or score > best[2]:
            best = (int(feat), threshold, score)
    return best


def _compact_grid(time: Array, event: Array) -> tuple[Array, Array, Array, Array]:
    """Per-event-time totals without materializing the O(n x m) risk/event matrices.

    Returns `(event_times, n_tot, d_tot, var_factor)`, where `n_tot[i]` is the number at risk at
    the i-th unique event time, `d_tot[i]` the number of events there, and `var_factor` the
    hypergeometric variance factor used by the log-rank statistic.
    """
    event_times = np.unique(time[event])
    if event_times.size == 0:
        empty = np.empty(0)
        return event_times, empty, empty, empty
    sorted_time = np.sort(time)
    n_tot = (time.shape[0] - np.searchsorted(sorted_time, event_times, side="left")).astype(float)
    positions = np.searchsorted(event_times, time[event])
    d_tot = np.bincount(positions, minlength=event_times.size).astype(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        factor = d_tot * (n_tot - d_tot) / (n_tot - 1.0)
    var_factor = np.where(n_tot > 1.0, factor, 0.0)
    return event_times, n_tot, d_tot, var_factor


def _logrank_kernel_impl(
    col: Array,
    time_idx_le: Array,
    is_event: Array,
    n_tot: Array,
    d_tot: Array,
    var_factor: Array,
    min_leaf: int,
) -> tuple[Any, Any]:
    """Best log-rank split for one feature via an incremental sweep (Numba-compiled when available).

    Sweeps subjects in ascending feature order, maintaining running sums for the log-rank numerator
    and variance as each subject joins the left group, so it never allocates the full risk matrix.
    Returns `(threshold, chi_square)`, with `chi_square < 0` signalling that no valid split exists.
    """
    n = col.shape[0]
    m = n_tot.shape[0]
    order = np.argsort(col)
    n1 = np.zeros(m)
    running_e = 0.0  # sum_i d_tot_i * n1_i / n_tot_i
    running_v = 0.0  # sum_i var_factor_i * f_i (1 - f_i)
    left_events = 0.0
    best_chi = -1.0
    best_thr = np.nan
    for p in range(n):
        j = order[p]
        k = time_idx_le[j]
        for i in range(k):
            f_old = n1[i] / n_tot[i]
            running_e -= d_tot[i] * f_old
            running_v -= var_factor[i] * f_old * (1.0 - f_old)
            n1[i] += 1.0
            f_new = n1[i] / n_tot[i]
            running_e += d_tot[i] * f_new
            running_v += var_factor[i] * f_new * (1.0 - f_new)
        if is_event[j]:
            left_events += 1.0
        if p < n - 1:
            left = p + 1
            nxt = order[p + 1]
            if (
                left >= min_leaf
                and (n - left) >= min_leaf
                and col[j] < col[nxt]
                and running_v > 0.0
            ):
                u = left_events - running_e
                chi = u * u / running_v
                if chi > best_chi:
                    best_chi = chi
                    best_thr = 0.5 * (col[j] + col[nxt])
    return best_thr, best_chi


_logrank_kernel = _njit(cache=True)(_logrank_kernel_impl) if _HAS_NUMBA else None


def _best_logrank_split_numba(
    x: Array,
    features: Array,
    min_samples_leaf: int,
    time: Array,
    event: Array,
) -> tuple[int, float, float] | None:
    """Numba-accelerated `_best_logrank_split`, driving the compiled per-feature kernel."""
    event_times, n_tot, d_tot, var_factor = _compact_grid(time, event)
    if event_times.size == 0:
        return None
    time_idx_le = np.searchsorted(event_times, time, side="right").astype(np.int64)
    is_event = np.ascontiguousarray(event.astype(np.bool_))
    best: tuple[int, float, float] | None = None
    for feat in features:
        col = np.ascontiguousarray(x[:, feat], dtype=np.float64)
        thr, chi = _logrank_kernel(  # type: ignore[misc]
            col, time_idx_le, is_event, n_tot, d_tot, var_factor, int(min_samples_leaf)
        )
        if chi >= 0.0 and (best is None or chi > best[2]):
            best = (int(feat), float(thr), float(chi))
    return best


def _leaf_curves(time: Array, event: Array, grid_times: Array) -> tuple[Array, Array]:
    """Nelson-Aalen cumulative hazard and Kaplan-Meier survival on `grid_times` for leaf samples.

    Counts are taken against the global `grid_times`, so where no leaf sample is at risk the
    increments are zero and the curves carry their previous value forward.
    """
    at_risk = (time[:, None] >= grid_times[None, :]).sum(axis=0).astype(float)  # n_i
    events = ((time[:, None] == grid_times[None, :]) & event[:, None]).sum(axis=0).astype(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        na_inc = np.where(at_risk > 0, events / at_risk, 0.0)
        km_inc = np.where(at_risk > 0, 1.0 - events / at_risk, 1.0)
    cumhaz = np.cumsum(na_inc)
    survival = np.cumprod(km_inc)
    return cumhaz, survival


class _Node:
    """A single node of a survival tree: either an internal split or a leaf with cached curves."""

    __slots__ = ("feature", "threshold", "left", "right", "cumhaz", "survival")

    def __init__(self) -> None:
        self.feature: int = -1
        self.threshold: float = np.nan
        self.left: _Node | None = None
        self.right: _Node | None = None
        self.cumhaz: Array | None = None
        self.survival: Array | None = None

    @property
    def is_leaf(self) -> bool:
        return self.left is None


class SurvivalTree:
    """A survival decision tree grown with the log-rank splitting rule.

    The tree recursively partitions subjects by covariate thresholds, at each step choosing the
    split that maximizes the two-sample log-rank statistic between the resulting child groups. Each
    leaf holds a Nelson-Aalen cumulative hazard and a Kaplan-Meier survival curve estimated from its
    training subjects, evaluated on the full training set's event-time grid. Prediction routes a
    subject to a leaf and reads off that leaf's curves; the scalar risk score is the summed
    cumulative hazard (ensemble mortality).

    A single tree is a high-variance estimator and is most useful as an interpretable building block
    or the base learner of a `RandomSurvivalForest`.

    Parameters
    ----------
    max_depth
        Maximum depth of the tree. `None` grows until other stopping rules apply.
    min_samples_split
        Minimum number of subjects a node must have to be eligible for splitting.
    min_samples_leaf
        Minimum number of subjects required in each child of a split.
    max_features
        Number of covariates considered at each split: `"sqrt"`, `"log2"`, an int, a float
        fraction, or `None` (all features). A random subset is drawn per node.
    splitter
        How thresholds are chosen: `"best"` (default) scans every candidate cut-point for the
        optimal log-rank split; `"random"` draws a single random threshold per candidate feature
        (the extremely-randomized-trees rule used by `ExtraSurvivalTrees`).
    engine
        Compute backend for the best-split search: `"numpy"` (default) uses the deterministic
        vectorized path; `"numba"` uses a compiled kernel (requires the `fast` extra) that is
        faster on large datasets; `"auto"` uses Numba when available. The Numba path produces a
        statistically equivalent fit but, because it accumulates the log-rank statistic in a
        different order, may differ from the NumPy path in tie-breaking.
    random_state
        Seed or `numpy.random.Generator` controlling the per-node feature subsampling.

    Examples
    --------
    Grow a survival tree on the bundled `lung` dataset and predict a risk score:

    ```{python}
    import greenwood as gw

    # Load data and build a right-censored response
    lung = gw.load_dataset("lung", backend="pandas").dropna(subset=["ph.ecog", "ph.karno"])
    y = gw.Surv.right(lung["time"], event=(lung["status"] == 2))
    cols = ["age", "sex", "ph.ecog", "ph.karno", "wt.loss"]

    # Fit a survival tree and score the first five subjects
    tree = gw.SurvivalTree(max_depth=3, random_state=0).fit(y, lung[cols])
    tree.predict(lung[cols])[:5]
    ```
    """

    def __init__(
        self,
        *,
        max_depth: int | None = None,
        min_samples_split: int = 6,
        min_samples_leaf: int = 3,
        max_features: Any = None,
        splitter: str = "best",
        engine: str = "numpy",
        random_state: Any = None,
    ) -> None:
        if min_samples_split < 2:
            raise ValueError("min_samples_split must be >= 2.")
        if min_samples_leaf < 1:
            raise ValueError("min_samples_leaf must be >= 1.")
        if max_depth is not None and max_depth < 1:
            raise ValueError("max_depth must be >= 1 or None.")
        if splitter not in ("best", "random"):
            raise ValueError(f"splitter must be 'best' or 'random', got {splitter!r}.")
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.splitter = splitter
        self.engine = engine
        self.random_state = random_state

    def __repr__(self) -> str:
        if getattr(self, "_root", None) is None:
            return "SurvivalTree() <unfitted>"
        return (
            f"SurvivalTree (log-rank splits, {self._n_nodes_} nodes, {self._n_leaves_} leaves)\n"
            f"n = {self.n_}, events = {self.n_event_}, features = {self.n_features_in_}"
        )

    def fit(
        self,
        surv: Surv | None,
        covariates: Any,
        *,
        data: Any = None,
        _rng: np.random.Generator | None = None,
        _event_times: Array | None = None,
        _time: Array | None = None,
        _event: Array | None = None,
    ) -> SurvivalTree:
        """Fit the survival tree to a right-censored response and covariate design.

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
        SurvivalTree
            The fitted estimator (for method chaining), with cached attributes such as
            `event_times_`, `n_features_in_`, and `feature_names_in_`.
        """
        x, names = _design_matrix(covariates, data)
        # The forest passes already-aligned time/event arrays (e.g. from a bootstrap sample);
        # otherwise derive them from the response and check row alignment.
        if _time is not None and _event is not None:
            time, event = _time, _event
        else:
            if surv is None:
                raise ValueError("A `Surv` response is required.")
            time, event = _prepare_response(surv)
            if x.shape[0] != surv.n:
                raise ValueError("Covariates and response must have the same number of rows.")
        keep = ~np.isnan(x).any(axis=1)
        x, time, event = x[keep], time[keep], event[keep]
        if not event.any():
            raise ValueError("No events remain after dropping missing rows.")

        rng = _rng if _rng is not None else np.random.default_rng(self.random_state)
        self._use_numba = _resolve_engine(self.engine)
        self.event_times_ = np.unique(time[event]) if _event_times is None else _event_times
        self.feature_names_in_ = list(names)
        self.n_features_in_ = x.shape[1]
        self.n_ = int(x.shape[0])
        self.n_event_ = int(event.sum())
        self._n_features_split = _resolve_max_features(self.max_features, x.shape[1])

        self._counter = [0]
        self._root = self._grow(x, time, event, depth=0, rng=rng)
        self._n_nodes_ = self._counter[0]
        self._n_leaves_ = self._count_leaves(self._root)
        return self

    def _grow(
        self, x: Array, time: Array, event: Array, *, depth: int, rng: np.random.Generator
    ) -> _Node:
        node = _Node()
        self._counter[0] += 1
        n = x.shape[0]
        can_split = (
            n >= self.min_samples_split
            and event.any()
            and (self.max_depth is None or depth < self.max_depth)
        )
        if can_split:
            features = rng.choice(self.n_features_in_, size=self._n_features_split, replace=False)
            if self.splitter == "random":
                grid = _RiskEventGrid(time, event)
                split = _random_logrank_split(grid, x, features, self.min_samples_leaf, rng)
            elif self._use_numba:
                split = _best_logrank_split_numba(x, features, self.min_samples_leaf, time, event)
            else:
                grid = _RiskEventGrid(time, event)
                split = _best_logrank_split(grid, x, features, self.min_samples_leaf)
            if split is not None:
                feat, threshold, _ = split
                mask = x[:, feat] <= threshold
                node.feature = feat
                node.threshold = threshold
                node.left = self._grow(x[mask], time[mask], event[mask], depth=depth + 1, rng=rng)
                node.right = self._grow(
                    x[~mask], time[~mask], event[~mask], depth=depth + 1, rng=rng
                )
                return node
        node.cumhaz, node.survival = _leaf_curves(time, event, self.event_times_)
        return node

    def _count_leaves(self, node: _Node) -> int:
        if node.is_leaf:
            return 1
        assert node.left is not None and node.right is not None
        return self._count_leaves(node.left) + self._count_leaves(node.right)

    def _leaf_indices(self, x: Array) -> list[_Node]:
        return [self._route(row) for row in x]

    def _route(self, row: Array) -> _Node:
        node = self._root
        while not node.is_leaf:
            assert node.left is not None and node.right is not None
            node = node.left if row[node.feature] <= node.threshold else node.right
        return node

    def _curves(self, x: Array) -> tuple[Array, Array]:
        """Return per-subject `(cumhaz, survival)` matrices of shape `(n_subjects, n_times)`."""
        leaves = self._leaf_indices(x)
        grid = self.event_times_.shape[0]
        cumhaz = np.empty((len(leaves), grid))
        survival = np.empty((len(leaves), grid))
        for i, leaf in enumerate(leaves):
            assert leaf.cumhaz is not None and leaf.survival is not None
            cumhaz[i] = leaf.cumhaz
            survival[i] = leaf.survival
        return cumhaz, survival

    def predict(
        self,
        newdata: Any = None,
        *,
        type: str = "risk",
        times: Any = None,
        format: str | None = None,
    ) -> Any:
        r"""Predict a risk score, survival curves, or cumulative-hazard curves.

        Three prediction types are available:

        1. `type="risk"` (default): Ishwaran's ensemble mortality, the sum of the leaf cumulative
        hazard over the training event times. Higher values indicate higher risk. Returned as a
        1-D NumPy array.

        2. `type="survival"`: Kaplan-Meier survival probabilities $S(t \mid x)$ at `times`, one
        column per subject. Returned as a DataFrame.

        3. `type="cumulative_hazard"`: Nelson-Aalen cumulative hazard $H(t \mid x)$ at `times`.
        Returned as a DataFrame.

        Parameters
        ----------
        newdata
            Covariates to predict for. `None` predicts for the training subjects.
        type
            One of `"risk"`, `"survival"`, or `"cumulative_hazard"`.
        times
            Times at which to evaluate survival / cumulative hazard. Defaults to the training
            event times. Ignored for `type="risk"`.
        format
            DataFrame backend for curve output: `None`, `"pandas"`, `"polars"`, or `"pyarrow"`.

        Returns
        -------
        numpy.ndarray or DataFrame
            A risk vector, or a curve frame with a `time` column and one column per subject.
        """
        if type not in _PREDICT_TYPES:
            raise ValueError(f"Unknown predict type {type!r}; use one of {_PREDICT_TYPES}.")
        x = self._design(newdata)
        cumhaz, survival = self._curves(x)
        if type == "risk":
            return cumhaz.sum(axis=1)
        if type == "cumulative_hazard":
            return self._curve_frame(cumhaz, times, format, boundary=0.0)
        return self._curve_frame(survival, times, format, boundary=1.0)

    def _design(self, newdata: Any) -> Array:
        if newdata is None:
            raise ValueError("Provide `newdata` to predict; the tree does not retain training X.")
        return _design_matrix(newdata)[0]

    def _curve_frame(self, curves: Array, times: Any, format: str | None, boundary: float) -> Any:
        query = (
            self.event_times_ if times is None else np.atleast_1d(np.asarray(times, dtype=float))
        )
        idx = np.searchsorted(self.event_times_, query, side="right") - 1
        sampled = _sample_step(curves, idx, boundary)
        columns: dict[str, Any] = {"time": query}
        columns.update({f"subject_{i + 1}": sampled[i] for i in range(sampled.shape[0])})
        return to_dataframe(columns, format=format)


def _sample_step(curves: Array, idx: Array, boundary: float) -> Array:
    """Evaluate right-continuous step-function `curves` (n_subjects x n_grid) at grid indices `idx`.

    Query indices before the first grid point (`idx < 0`) take the `boundary` value (0 for a
    cumulative hazard, 1 for survival).
    """
    n_grid = curves.shape[1]
    safe = np.clip(idx, 0, n_grid - 1)
    sampled = curves[:, safe].copy()
    before = idx < 0
    if before.any():
        sampled[:, before] = boundary
    return sampled


class _BaseSurvivalForest:
    """Shared implementation for bagged ensembles of log-rank survival trees.

    Concrete subclasses set the class attribute `_splitter` to control how each tree chooses split
    thresholds (`"best"` for `RandomSurvivalForest`, `"random"` for `ExtraSurvivalTrees`). This base
    provides fitting, ensemble averaging of the per-tree cumulative-hazard and survival curves,
    out-of-bag concordance, and permutation variable importance.
    """

    _splitter: str = "best"

    def __init__(
        self,
        *,
        n_estimators: int = 100,
        max_depth: int | None = None,
        min_samples_split: int = 6,
        min_samples_leaf: int = 3,
        max_features: Any = "sqrt",
        bootstrap: bool = True,
        oob_score: bool = False,
        engine: str = "numpy",
        random_state: Any = None,
    ) -> None:
        if n_estimators < 1:
            raise ValueError("n_estimators must be >= 1.")
        if oob_score and not bootstrap:
            raise ValueError("oob_score=True requires bootstrap=True.")
        if engine not in _ENGINES:
            raise ValueError(f"engine must be one of {_ENGINES}, got {engine!r}.")
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.oob_score = oob_score
        self.engine = engine
        self.random_state = random_state

    def __repr__(self) -> str:
        name = type(self).__name__
        if getattr(self, "trees_", None) is None:
            return f"{name}(n_estimators={self.n_estimators}) <unfitted>"
        lines = [
            f"{name} ({self.n_estimators} log-rank trees, max_features={self.max_features!r})",
            f"n = {self.n_}, events = {self.n_event_}, features = {self.n_features_in_}",
        ]
        if self.oob_score_ is not None:
            lines.append(f"out-of-bag concordance = {self.oob_score_:.4f}")
        return "\n".join(lines)

    def fit(self, surv: Surv, covariates: Any, *, data: Any = None) -> Self:
        """Fit the forest to a right-censored response and covariate design.

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
            `oob_score_`, `n_features_in_`, and `feature_names_in_`.
        """
        x, names = _design_matrix(covariates, data)
        time, event = _prepare_response(surv)
        if x.shape[0] != surv.n:
            raise ValueError("Covariates and response must have the same number of rows.")
        keep = ~np.isnan(x).any(axis=1)
        x, time, event = x[keep], time[keep], event[keep]
        n = x.shape[0]
        if not event.any():
            raise ValueError("No events remain after dropping missing rows.")

        rng = np.random.default_rng(self.random_state)
        event_times = np.unique(time[event])
        self.event_times_ = event_times
        self.feature_names_in_ = list(names)
        self.n_features_in_ = x.shape[1]
        self.n_ = int(n)
        self.n_event_ = int(event.sum())
        # Retain training data for OOB scoring and variable importance.
        self._x_train, self._time_train, self._event_train = x, time, event

        trees: list[SurvivalTree] = []
        oob_masks: list[Array] = []
        for _ in range(self.n_estimators):
            if self.bootstrap:
                idx = rng.integers(0, n, size=n)
                in_bag = np.zeros(n, dtype=bool)
                in_bag[idx] = True
                oob_masks.append(~in_bag)
                xb, tb, eb = x[idx], time[idx], event[idx]
            else:
                oob_masks.append(np.zeros(n, dtype=bool))
                xb, tb, eb = x, time, event
            tree = SurvivalTree(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                max_features=self.max_features,
                splitter=self._splitter,
                engine=self.engine,
            )
            tree.fit(None, xb, _rng=rng, _event_times=event_times, _time=tb, _event=eb)
            trees.append(tree)
        self.trees_ = trees
        self._oob_masks = oob_masks
        self.oob_score_ = self._compute_oob_score() if self.oob_score else None
        return self

    def _compute_oob_score(self) -> float | None:
        """Out-of-bag concordance: score each subject using only trees where it was out of bag."""
        from ._metrics import concordance_index
        from ._surv import Surv

        n = self.n_
        risk_sum = np.zeros(n)
        tree_count = np.zeros(n)
        for tree, oob in zip(self.trees_, self._oob_masks, strict=False):
            if not oob.any():
                continue
            cumhaz, _ = tree._curves(self._x_train[oob])
            risk_sum[oob] += cumhaz.sum(axis=1)
            tree_count[oob] += 1
        scored = tree_count > 0
        if scored.sum() < 2:
            return None
        risk = risk_sum[scored] / tree_count[scored]
        y = Surv.right(self._time_train[scored], event=self._event_train[scored])
        return float(concordance_index(y, risk))

    def _ensemble_curves(self, x: Array) -> tuple[Array, Array]:
        """Average per-tree cumulative-hazard and survival curves across the forest."""
        cumhaz = np.zeros((x.shape[0], self.event_times_.shape[0]))
        survival = np.zeros_like(cumhaz)
        for tree in self.trees_:
            tc, ts = tree._curves(x)
            cumhaz += tc
            survival += ts
        return cumhaz / self.n_estimators, survival / self.n_estimators

    def predict(
        self,
        newdata: Any = None,
        *,
        type: str = "risk",
        times: Any = None,
        format: str | None = None,
    ) -> Any:
        r"""Predict a risk score, survival curves, or cumulative-hazard curves from the forest.

        Prediction types mirror `SurvivalTree.predict`, but each curve is the average over all
        trees in the ensemble:

        1. `type="risk"` (default): summed ensemble cumulative hazard (mortality), a 1-D array.
        2. `type="survival"`: ensemble survival $S(t \mid x)$ at `times`, one column per subject.
        3. `type="cumulative_hazard"`: ensemble cumulative hazard $H(t \mid x)$ at `times`.

        Parameters
        ----------
        newdata
            Covariates to predict for. `None` predicts for the training subjects.
        type
            One of `"risk"`, `"survival"`, or `"cumulative_hazard"`.
        times
            Times at which to evaluate curves. Defaults to the training event times.
        format
            DataFrame backend for curve output.

        Returns
        -------
        numpy.ndarray or DataFrame
            A risk vector, or a curve frame with a `time` column and one column per subject.
        """
        if type not in _PREDICT_TYPES:
            raise ValueError(f"Unknown predict type {type!r}; use one of {_PREDICT_TYPES}.")
        x = self._x_train if newdata is None else _design_matrix(newdata)[0]
        cumhaz, survival = self._ensemble_curves(x)
        if type == "risk":
            return cumhaz.sum(axis=1)
        curves, boundary = (cumhaz, 0.0) if type == "cumulative_hazard" else (survival, 1.0)
        query = (
            self.event_times_ if times is None else np.atleast_1d(np.asarray(times, dtype=float))
        )
        idx = np.searchsorted(self.event_times_, query, side="right") - 1
        sampled = _sample_step(curves, idx, boundary)
        columns: dict[str, Any] = {"time": query}
        columns.update({f"subject_{i + 1}": sampled[i] for i in range(sampled.shape[0])})
        return to_dataframe(columns, format=format)

    def variable_importance(
        self, *, n_repeats: int = 5, random_state: Any = None, format: str | None = None
    ) -> Any:
        """Permutation variable importance on the out-of-bag samples.

        For each covariate, its values are randomly permuted and the increase in out-of-bag
        prediction error (`1 - concordance`) is measured; larger increases indicate more important
        covariates. Importances are averaged over `n_repeats` permutations.

        Parameters
        ----------
        n_repeats
            Number of random permutations per covariate.
        random_state
            Seed or generator controlling the permutations.
        format
            DataFrame backend for the returned table.

        Returns
        -------
        DataFrame
            One row per covariate with columns `term` and `importance`, sorted by importance.
        """
        importances = self._permutation_importance(n_repeats=n_repeats, random_state=random_state)
        order = np.argsort(importances)[::-1]
        columns = {
            "term": [self.feature_names_in_[i] for i in order],
            "importance": importances[order],
        }
        return to_dataframe(columns, format=format)

    def _permutation_importance(self, *, n_repeats: int, random_state: Any) -> Array:
        from ._metrics import concordance_index
        from ._surv import Surv

        if not self.bootstrap:
            raise ValueError("variable_importance requires bootstrap=True.")
        rng = np.random.default_rng(random_state)
        x, time, event = self._x_train, self._time_train, self._event_train

        def oob_risk(matrix: Array) -> tuple[Array, Array]:
            risk_sum = np.zeros(x.shape[0])
            count = np.zeros(x.shape[0])
            for tree, oob in zip(self.trees_, self._oob_masks, strict=False):
                if not oob.any():
                    continue
                cumhaz, _ = tree._curves(matrix[oob])
                risk_sum[oob] += cumhaz.sum(axis=1)
                count[oob] += 1
            scored = count > 0
            return risk_sum[scored] / np.where(count[scored] == 0, 1, count[scored]), scored

        base_risk, scored = oob_risk(x)
        y = Surv.right(time[scored], event=event[scored])
        base_error = 1.0 - concordance_index(y, base_risk)

        importances = np.zeros(x.shape[1])
        for feat in range(x.shape[1]):
            drop = 0.0
            for _ in range(n_repeats):
                permuted = x.copy()
                permuted[:, feat] = x[rng.permutation(x.shape[0]), feat]
                risk, sc = oob_risk(permuted)
                y_p = Surv.right(time[sc], event=event[sc])
                drop += (1.0 - concordance_index(y_p, risk)) - base_error
            importances[feat] = drop / n_repeats
        return importances


class RandomSurvivalForest(_BaseSurvivalForest):
    """A random survival forest: a bagged ensemble of log-rank survival trees.

    Each tree is grown on a bootstrap sample of the data and considers a random subset of covariates
    at every split, following Ishwaran et al. (2008). At each node the split threshold is chosen to
    maximize the log-rank separation between the child groups. Predictions average the per-tree
    cumulative-hazard functions across the ensemble; the scalar risk score is the summed ensemble
    cumulative hazard (mortality). Averaging many decorrelated trees yields a low-variance,
    well-calibrated non-parametric survival model that captures non-linearities and interactions
    without a proportional-hazards assumption.

    Parameters
    ----------
    n_estimators
        Number of trees in the forest.
    max_depth
        Maximum depth of each tree (`None` for unlimited).
    min_samples_split
        Minimum node size eligible for splitting.
    min_samples_leaf
        Minimum subjects in each child of a split.
    max_features
        Covariates considered per split: `"sqrt"` (default), `"log2"`, an int, a float, or `None`.
    bootstrap
        Whether to grow each tree on a bootstrap sample (required for `oob_score` and
        `variable_importance`).
    oob_score
        Whether to compute an out-of-bag concordance estimate after fitting.
    engine
        Compute backend for the split search: `"numpy"` (default, deterministic) or `"numba"`
        (faster on large datasets, requires the `fast` extra; produces a statistically equivalent
        fit that may differ in tie-breaking). `"auto"` uses Numba when available.
    random_state
        Seed or `numpy.random.Generator` for bootstrap sampling and per-node feature selection.

    Examples
    --------
    Fit a forest on the bundled `lung` dataset and inspect the out-of-bag concordance:

    ```{python}
    import greenwood as gw

    # Load data and build a right-censored response
    lung = gw.load_dataset("lung", backend="pandas").dropna(subset=["ph.ecog", "ph.karno"])
    y = gw.Surv.right(lung["time"], event=(lung["status"] == 2))
    cols = ["age", "sex", "ph.ecog", "ph.karno", "wt.loss"]

    # Fit the forest with an out-of-bag score
    rsf = gw.RandomSurvivalForest(
        n_estimators=100, oob_score=True, random_state=0
    ).fit(y, lung[cols])
    rsf
    ```
    """

    _splitter = "best"


class ExtraSurvivalTrees(_BaseSurvivalForest):
    """An extremely-randomized survival forest (extra survival trees).

    Like `RandomSurvivalForest`, but at each node the split threshold for every candidate covariate
    is drawn at random (uniformly between the covariate's minimum and maximum at that node) rather
    than optimized. The extra randomization further decorrelates the trees, often lowering variance
    and speeding up fitting at the cost of a little bias. Following the extra-trees convention, each
    tree is grown on the full sample (`bootstrap=False`) by default; set `bootstrap=True` to enable
    out-of-bag scoring and permutation variable importance.

    Parameters
    ----------
    n_estimators
        Number of trees in the forest.
    max_depth
        Maximum depth of each tree (`None` for unlimited).
    min_samples_split
        Minimum node size eligible for splitting.
    min_samples_leaf
        Minimum subjects in each child of a split.
    max_features
        Covariates considered per split: `"sqrt"` (default), `"log2"`, an int, a float, or `None`.
    bootstrap
        Whether to grow each tree on a bootstrap sample. `False` by default (the extra-trees
        convention); required to be `True` for `oob_score` and `variable_importance`.
    oob_score
        Whether to compute an out-of-bag concordance estimate after fitting (requires
        `bootstrap=True`).
    engine
        Compute backend for the split search: `"numpy"` (default, deterministic) or `"numba"`
        (faster on large datasets, requires the `fast` extra). `"auto"` uses Numba when available.
    random_state
        Seed or `numpy.random.Generator` for split randomization and per-node feature selection.

    Examples
    --------
    Fit extra survival trees on the bundled `lung` dataset:

    ```{python}
    import greenwood as gw

    # Load data and build a right-censored response
    lung = gw.load_dataset("lung", backend="pandas").dropna(subset=["ph.ecog", "ph.karno"])
    y = gw.Surv.right(lung["time"], event=(lung["status"] == 2))
    cols = ["age", "sex", "ph.ecog", "ph.karno", "wt.loss"]

    # Fit with bootstrap sampling so an out-of-bag score is available
    ext = gw.ExtraSurvivalTrees(
        n_estimators=100, bootstrap=True, oob_score=True, random_state=0
    ).fit(y, lung[cols])
    ext
    ```
    """

    _splitter = "random"

    def __init__(
        self,
        *,
        n_estimators: int = 100,
        max_depth: int | None = None,
        min_samples_split: int = 6,
        min_samples_leaf: int = 3,
        max_features: Any = "sqrt",
        bootstrap: bool = False,
        oob_score: bool = False,
        engine: str = "numpy",
        random_state: Any = None,
    ) -> None:
        super().__init__(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            bootstrap=bootstrap,
            oob_score=oob_score,
            engine=engine,
            random_state=random_state,
        )


def _tidy_forest(model: _BaseSurvivalForest, *, format: str | None = None, **_: Any) -> Any:
    """broom-style `tidy`: one row per covariate with its permutation importance."""
    return model.variable_importance(format=format)


def _glance_forest(model: _BaseSurvivalForest, *, format: str | None = None, **_: Any) -> Any:
    """broom-style `glance`: one-row forest summary."""
    return to_dataframe(
        {
            "n": [model.n_],
            "nevent": [model.n_event_],
            "n_estimators": [model.n_estimators],
            "n_features": [model.n_features_in_],
            "oob_concordance": [model.oob_score_],
        },
        format=format,
    )


def _register_adapters() -> None:
    from .summaries import register_glance, register_tidier

    for path in (
        "greenwood._forest.RandomSurvivalForest",
        "greenwood._forest.ExtraSurvivalTrees",
    ):
        register_tidier(path, _tidy_forest)
        register_glance(path, _glance_forest)


_register_adapters()
