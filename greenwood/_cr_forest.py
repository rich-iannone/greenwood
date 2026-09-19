"""Competing-risk random survival forest with native CIF-based splitting.

A `CompetingRiskForest` grows an ensemble of trees that split nodes using the log-rank CR statistic,
which modifies the standard log-rank test with a Lau-inclusive risk set so that subjects who
experienced a competing event remain at risk under the subdistribution framework. Each leaf stores
non-parametric Aalen-Johansen cumulative incidence curves for every cause, and the forest prediction
averages these per-cause CIF curves across the ensemble.
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

__all__ = ["CompetingRiskForest"]

Array = npt.NDArray[Any]


# ---------------------------------------------------------------------------
# Response helpers
# ---------------------------------------------------------------------------


def _prepare_cr_response(surv: Surv) -> tuple[Array, Array, tuple[str, ...], list[int]]:
    """Extract `(time, status, state_labels, cause_codes)` from a multistate `Surv`.

    Returns
    -------
    time : ndarray
        Event/censoring times.
    status : ndarray of int
        0 = censored, 1..K = cause codes.
    state_labels : tuple of str
        Human-readable cause names.
    cause_codes : list of int
        Integer cause codes `[1, 2, ..., K]`.
    """
    if surv.states is None:
        raise NotImplementedError(
            "CompetingRiskForest requires a multistate response built with "
            "`Surv.multistate()`, not a plain right-censored response."
        )
    time = np.asarray(surv.stop, dtype=float)
    status = np.asarray(surv.status, dtype=int)
    states = surv.states
    cause_codes = sorted(set(status[status > 0]))
    return time, status, states, cause_codes


# ---------------------------------------------------------------------------
# Splitting criterion: log-rank CR (Lau-inclusive risk set)
# ---------------------------------------------------------------------------


class _CREventGrid:
    """Pre-computed event/risk counts for the logrankCR split search.

    For each unique event time, stores total at-risk (standard and Lau-inclusive), per-cause event
    counts, and the per-subject membership arrays needed to compute left-child counts for any
    candidate split.
    """

    __slots__ = (
        "times",
        "at_risk",
        "events_by_cause",
        "causes",
        "n_tot",
        "d_by_cause",
        "d_other_cumsum",
        "subj_at_risk",
        "subj_events_by_cause",
    )

    def __init__(self, time: Array, status: Array, causes: list[int]) -> None:
        self.causes = causes
        event_mask = status > 0
        event_times = np.unique(time[event_mask])
        self.times = event_times
        m = event_times.size
        if m == 0:
            return

        # Subject-level boolean matrices: (n_subjects, n_times)
        self.subj_at_risk = time[:, None] >= event_times[None, :]  # (n, m)
        self.n_tot = self.subj_at_risk.sum(axis=0).astype(float)

        # Per-cause event matrices
        self.subj_events_by_cause: dict[int, Array] = {}
        self.d_by_cause: dict[int, Array] = {}
        for k in causes:
            ev_k = (time[:, None] == event_times[None, :]) & (status == k)[:, None]
            self.subj_events_by_cause[k] = ev_k
            dk = ev_k.sum(axis=0).astype(float)
            self.d_by_cause[k] = dk

        # Total events at each time
        d_any = np.zeros(m, dtype=float)
        for k in causes:
            d_any += self.d_by_cause[k]

        # Cumulative competing events (for Lau-inclusive risk set)
        # For each cause k, "other" = d_any - d_k
        # Lau-inclusive at-risk for cause k at time m:
        #   Y_inc_k(m) = Y(m) + cumsum(d_other_k)[m-1]
        # where d_other_k = d_any - d_k
        self.d_other_cumsum: dict[int, Array] = {}
        for k in causes:
            d_other_k = d_any - self.d_by_cause[k]
            # Strict prefix sum: at time index i, sum of other-cause events at indices < i
            cs = np.cumsum(d_other_k)
            prefix = np.concatenate(([0.0], cs[:-1]))
            self.d_other_cumsum[k] = prefix


def _best_logrankcr_split(
    grid: _CREventGrid,
    x: Array,
    features: Array,
    min_samples_leaf: int,
) -> tuple[int, float, float] | None:
    """Find the (feature, threshold) maximizing the logrankCR statistic.

    Pools signed numerators across all causes before squaring.
    """
    n = x.shape[0]
    m = grid.times.size
    if m == 0:
        return None

    causes = grid.causes
    best: tuple[int, float, float] | None = None

    for feat in features:
        col = x[:, feat]
        order = np.argsort(col, kind="mergesort")
        col_sorted = col[order]

        # Cumulative left-child at-risk counts (standard)
        n1_std = np.cumsum(grid.subj_at_risk[order], axis=0).astype(float)  # (n, m)

        # Cumulative left-child events per cause (compute once, reuse)
        d1_per_cause: dict[int, Array] = {}
        for k in causes:
            d1_per_cause[k] = np.cumsum(grid.subj_events_by_cause[k][order], axis=0).astype(
                float
            )  # (n, m)

        # Left-child all-cause events (sum across causes)
        d1_any = np.zeros((n, m), dtype=float)
        for k in causes:
            d1_any += d1_per_cause[k]

        pooled_num = np.zeros(n)
        pooled_var = np.zeros(n)

        for k in causes:
            d1_k = d1_per_cause[k]
            d_k = grid.d_by_cause[k]  # (m,)

            # Left-child other-cause events and their strict prefix sum
            d1_other_k = d1_any - d1_k
            d1_other_prefix = np.concatenate(
                [np.zeros((n, 1)), np.cumsum(d1_other_k, axis=1)[:, :-1]], axis=1
            )

            # Lau-inclusive at-risk: left and total
            n1_inc = n1_std + d1_other_prefix
            n_tot_inc = grid.n_tot + grid.d_other_cumsum[k]  # (m,)

            # Expected left-child events
            with np.errstate(divide="ignore", invalid="ignore"):
                frac_inc = np.divide(
                    n1_inc,
                    n_tot_inc[None, :],
                    out=np.zeros_like(n1_inc),
                    where=n_tot_inc[None, :] > 0,
                )
            expected_k = (frac_inc * d_k[None, :]).sum(axis=1)  # (n,)
            obs_k = d1_k.sum(axis=1)  # (n,) observed left-child cause-k events
            num_k = obs_k - expected_k

            # Variance (hypergeometric, guarded on standard n_tot >= 2)
            with np.errstate(divide="ignore", invalid="ignore"):
                var_factor = np.where(
                    grid.n_tot[None, :] >= 2,
                    d_k[None, :]
                    * frac_inc
                    * (1.0 - frac_inc)
                    * (n_tot_inc[None, :] - d_k[None, :])
                    / (n_tot_inc[None, :] - 1.0),
                    0.0,
                )
            var_k = var_factor.sum(axis=1)  # (n,)

            pooled_num += num_k
            pooled_var += var_k

        # Composite statistic: (sum_k num_k)^2 / sum_k var_k
        with np.errstate(divide="ignore", invalid="ignore"):
            chi = np.divide(pooled_num**2, pooled_var, out=np.zeros(n), where=pooled_var > 0)

        # Valid splits
        distinct = col_sorted[:-1] < col_sorted[1:]
        left_size = np.arange(1, n)
        valid = distinct & (left_size >= min_samples_leaf) & (n - left_size >= min_samples_leaf)
        if not valid.any():
            continue
        chi_valid = np.where(valid, chi[:-1], -np.inf)
        p = int(np.argmax(chi_valid))
        if not np.isfinite(chi_valid[p]):
            continue
        score = float(chi_valid[p])
        if best is None or score > best[2]:
            threshold = 0.5 * (col_sorted[p] + col_sorted[p + 1])
            best = (int(feat), float(threshold), score)

    return best


# ---------------------------------------------------------------------------
# Leaf CIF computation
# ---------------------------------------------------------------------------


def _leaf_cif(time: Array, status: Array, causes: list[int], grid_times: Array) -> dict[int, Array]:
    """Aalen-Johansen CIF for each cause, evaluated on `grid_times`.

    Returns a dict mapping cause code to a 1-D CIF array of length `len(grid_times)`.
    """
    # Compute CIF on the leaf's own event times
    leaf_event_times = np.unique(time[status > 0])
    if leaf_event_times.size == 0:
        return {k: np.zeros(grid_times.size) for k in causes}

    n_risk = np.array([float((time >= t).sum()) for t in leaf_event_times])
    d_any = np.array([float(((time == t) & (status > 0)).sum()) for t in leaf_event_times])
    surv = np.cumprod(1.0 - d_any / n_risk)
    surv_left = np.concatenate(([1.0], surv[:-1]))

    result: dict[int, Array] = {}
    for k in causes:
        d_k = np.array([float(((time == t) & (status == k)).sum()) for t in leaf_event_times])
        cif_leaf = np.cumsum(surv_left * d_k / n_risk)

        # Map to the global grid via step-function evaluation
        cif_on_grid = np.zeros(grid_times.size)
        for i, t in enumerate(grid_times):
            idx = np.searchsorted(leaf_event_times, t, side="right") - 1
            if idx >= 0:
                cif_on_grid[i] = cif_leaf[idx]
        result[k] = cif_on_grid

    return result


# ---------------------------------------------------------------------------
# Tree node
# ---------------------------------------------------------------------------


class _CRNode:
    """A node in a competing-risk tree: internal split or leaf with per-cause CIF."""

    __slots__ = ("feature", "threshold", "left", "right", "cif")

    def __init__(self) -> None:
        self.feature: int = -1
        self.threshold: float = np.nan
        self.left: _CRNode | None = None
        self.right: _CRNode | None = None
        self.cif: dict[int, Array] | None = None

    @property
    def is_leaf(self) -> bool:
        return self.left is None


# ---------------------------------------------------------------------------
# CompetingRiskForest
# ---------------------------------------------------------------------------


class CompetingRiskForest:
    """A random survival forest for competing risks with native CIF-based splitting.

    Grows an ensemble of trees that partition subjects using the log-rank CR statistic. At each
    candidate split, the at-risk set follows the subdistribution convention: subjects who
    experienced a competing event remain at risk, so the split directly targets separation of the
    cumulative incidence functions. Each leaf estimates per-cause CIF via the Aalen-Johansen
    estimator, and the forest averages these curves across all trees.

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
        Whether to grow each tree on a bootstrap sample. Required for
        `oob_score` and `variable_importance`.
    oob_score
        Whether to compute an out-of-bag concordance estimate after fitting.
    random_state
        Seed or `numpy.random.Generator` for reproducibility.

    Examples
    --------
    ```{python}
    import greenwood as gw
    import numpy as np

    mg = gw.load_dataset("mgus2", backend="pandas")
    etime = np.where(mg["pstat"] == 1, mg["ptime"], mg["futime"])
    cause = np.where(mg["pstat"] == 1, 1, 2 * mg["death"])
    y = gw.Surv.multistate(etime, event=cause, states=("pcm", "death"))

    crf = gw.CompetingRiskForest(n_estimators=50, random_state=0).fit(
        y, mg[["age", "sex"]]
    )
    crf
    ```
    """

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
        random_state: Any = None,
    ) -> None:
        if n_estimators < 1:
            raise ValueError("n_estimators must be >= 1.")
        if oob_score and not bootstrap:
            raise ValueError("oob_score=True requires bootstrap=True.")
        if max_depth is not None and max_depth < 1:
            raise ValueError("max_depth must be >= 1 or None.")
        if min_samples_split < 2:
            raise ValueError("min_samples_split must be >= 2.")
        if min_samples_leaf < 1:
            raise ValueError("min_samples_leaf must be >= 1.")
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.oob_score = oob_score
        self.random_state = random_state

    def __repr__(self) -> str:
        if getattr(self, "trees_", None) is None:
            return f"CompetingRiskForest(n_estimators={self.n_estimators}) <unfitted>"
        lines = [
            f"CompetingRiskForest ({self.n_estimators} logrankCR trees, "
            f"max_features={self.max_features!r})",
            f"n = {self.n_}, events = {self.n_event_}, features = {self.n_features_in_}",
            f"causes: {', '.join(str(s) for s in self.states_)}",
        ]
        if self.oob_score_ is not None:
            lines.append(f"out-of-bag concordance = {self.oob_score_:.4f}")
        return "\n".join(lines)

    def fit(self, surv: Surv, covariates: Any, *, data: Any = None) -> Self:
        """Fit the competing-risk forest.

        Parameters
        ----------
        surv
            A multistate `Surv` response built with `Surv.multistate()`.
        covariates
            A dataframe, a 2-D array, or a formula string evaluated against `data`.
        data
            DataFrame used to evaluate a formula string.

        Returns
        -------
        self
            The fitted estimator with attributes `trees_`, `event_times_`,
            `states_`, `cause_codes_`, and `oob_score_`.
        """
        x, names = _design_matrix(covariates, data)
        time, status, states, cause_codes = _prepare_cr_response(surv)
        if x.shape[0] != surv.n:
            raise ValueError("Covariates and response must have the same number of rows.")

        keep = ~np.isnan(x).any(axis=1)
        x, time, status = x[keep], time[keep], status[keep]
        n = x.shape[0]
        if not (status > 0).any():
            raise ValueError("No events remain after dropping missing rows.")

        rng = np.random.default_rng(self.random_state)
        event_times = np.unique(time[status > 0])
        self.event_times_ = event_times
        self.states_ = states
        self.cause_codes_ = cause_codes
        self.feature_names_in_ = list(names)
        self.n_features_in_ = x.shape[1]
        self.n_ = int(n)
        self.n_event_ = int((status > 0).sum())
        self._n_features_split = self._resolve_max_features(x.shape[1])
        self._x_train = x
        self._time_train = time
        self._status_train = status

        trees: list[_CRNode] = []
        oob_masks: list[Array] = []
        for _ in range(self.n_estimators):
            if self.bootstrap:
                idx = rng.integers(0, n, size=n)
                in_bag = np.zeros(n, dtype=bool)
                in_bag[idx] = True
                oob_masks.append(~in_bag)
                xb, tb, sb = x[idx], time[idx], status[idx]
            else:
                oob_masks.append(np.zeros(n, dtype=bool))
                xb, tb, sb = x, time, status

            root = self._grow(xb, tb, sb, cause_codes, depth=0, rng=rng)
            trees.append(root)

        self.trees_ = trees
        self._oob_masks = oob_masks
        self.oob_score_ = self._compute_oob_score() if self.oob_score else None
        return self

    def _resolve_max_features(self, n_features: int) -> int:
        mf = self.max_features
        if mf is None:
            value = n_features
        elif isinstance(mf, str):
            if mf == "sqrt":
                value = int(np.sqrt(n_features))
            elif mf == "log2":
                value = int(np.log2(n_features))
            else:
                raise ValueError(f"Unknown max_features {mf!r}.")
        elif isinstance(mf, (int, np.integer)):
            value = int(mf)
        elif isinstance(mf, float):
            value = int(mf * n_features)
        else:
            raise TypeError("max_features must be 'sqrt', 'log2', an int, a float, or None.")
        return int(np.clip(value, 1, n_features))

    def _grow(
        self,
        x: Array,
        time: Array,
        status: Array,
        causes: list[int],
        *,
        depth: int,
        rng: np.random.Generator,
    ) -> _CRNode:
        node = _CRNode()
        n = x.shape[0]
        can_split = (
            n >= self.min_samples_split
            and (status > 0).any()
            and (self.max_depth is None or depth < self.max_depth)
        )
        if can_split:
            features = rng.choice(self.n_features_in_, size=self._n_features_split, replace=False)
            grid = _CREventGrid(time, status, causes)
            split = _best_logrankcr_split(grid, x, features, self.min_samples_leaf)
            if split is not None:
                feat, threshold, _ = split
                mask = x[:, feat] <= threshold
                node.feature = feat
                node.threshold = threshold
                node.left = self._grow(
                    x[mask], time[mask], status[mask], causes, depth=depth + 1, rng=rng
                )
                node.right = self._grow(
                    x[~mask], time[~mask], status[~mask], causes, depth=depth + 1, rng=rng
                )
                return node

        node.cif = _leaf_cif(time, status, causes, self.event_times_)
        return node

    def _route(self, row: Array) -> _CRNode:
        node = self.trees_[0]  # placeholder; caller iterates trees
        while not node.is_leaf:
            assert node.left is not None and node.right is not None
            node = node.left if row[node.feature] <= node.threshold else node.right
        return node

    def _route_tree(self, root: _CRNode, row: Array) -> _CRNode:
        node = root
        while not node.is_leaf:
            assert node.left is not None and node.right is not None
            node = node.left if row[node.feature] <= node.threshold else node.right
        return node

    def _ensemble_cif(self, x: Array) -> dict[int, Array]:
        """Average per-cause CIF across all trees for each subject.

        Returns dict mapping cause code to array of shape `(n_subjects, n_times)`.
        """
        n_subjects = x.shape[0]
        n_times = self.event_times_.shape[0]
        result: dict[int, Array] = {k: np.zeros((n_subjects, n_times)) for k in self.cause_codes_}
        for root in self.trees_:
            for i in range(n_subjects):
                leaf = self._route_tree(root, x[i])
                assert leaf.cif is not None
                for k in self.cause_codes_:
                    result[k][i] += leaf.cif[k]
        for k in self.cause_codes_:
            result[k] /= self.n_estimators
        return result

    def predict(
        self,
        newdata: Any = None,
        *,
        cause: int | str | None = None,
        times: Any = None,
        format: str | None = None,
    ) -> Any:
        r"""Predict cumulative incidence functions from the forest.

        Returns a DataFrame with a `time` column and one column per subject,
        containing the ensemble-averaged CIF $F_k(t \mid x)$ for the requested
        cause.  If `cause` is `None` and there is only one cause, that cause
        is used; otherwise `cause` is required.

        Parameters
        ----------
        newdata
            Covariates to predict for.
        cause
            Which cause to predict. A string label or integer code. Required when
            there are multiple causes.
        times
            Times at which to evaluate the CIF. Defaults to the training event times.
        format
            DataFrame backend for the returned table.

        Returns
        -------
        DataFrame
            Columns `time`, `subject_1`, `subject_2`, ...
        """
        cause_code = self._resolve_cause(cause)
        x = self._x_train if newdata is None else _design_matrix(newdata)[0]
        cif_dict = self._ensemble_cif(x)
        cif_k = cif_dict[cause_code]  # (n_subjects, n_times)

        query = (
            self.event_times_ if times is None else np.atleast_1d(np.asarray(times, dtype=float))
        )
        # Sample the step function at query times
        idx = np.searchsorted(self.event_times_, query, side="right") - 1
        n_grid = self.event_times_.shape[0]
        safe = np.clip(idx, 0, n_grid - 1)
        sampled = cif_k[:, safe].copy()
        before = idx < 0
        if before.any():
            sampled[:, before] = 0.0

        columns: dict[str, Any] = {"time": query}
        columns.update({f"subject_{i + 1}": sampled[i] for i in range(sampled.shape[0])})
        return to_dataframe(columns, format=format)

    def predict_risk(
        self,
        newdata: Any = None,
        *,
        cause: int | str | None = None,
    ) -> Array:
        """Scalar risk score: summed CIF over the event-time grid for each subject.

        Higher values indicate higher predicted cumulative incidence for the given cause.
        """
        cause_code = self._resolve_cause(cause)
        x = self._x_train if newdata is None else _design_matrix(newdata)[0]
        cif_dict = self._ensemble_cif(x)
        return cif_dict[cause_code].sum(axis=1)

    def _resolve_cause(self, cause: int | str | None) -> int:
        if cause is None:
            if len(self.cause_codes_) == 1:
                return self.cause_codes_[0]
            raise ValueError(
                "Multiple causes present; specify `cause` as a string label or integer code. "
                f"Available causes: {list(self.states_)}"
            )
        if isinstance(cause, str):
            if cause not in self.states_:
                raise ValueError(f"Unknown cause {cause!r}. Available: {list(self.states_)}")
            return list(self.states_).index(cause) + 1
        code = int(cause)
        if code not in self.cause_codes_:
            raise ValueError(f"Unknown cause code {cause}. Available: {self.cause_codes_}")
        return code

    def _compute_oob_score(self) -> float | None:
        """OOB concordance using summed CIF as the risk score for the first cause."""
        from ._metrics import concordance_index
        from ._surv import Surv

        n = self.n_
        target = self.cause_codes_[0]
        risk_sum = np.zeros(n)
        tree_count = np.zeros(n)
        for root, oob in zip(self.trees_, self._oob_masks, strict=False):
            if not oob.any():
                continue
            oob_idx = np.where(oob)[0]
            for i in oob_idx:
                leaf = self._route_tree(root, self._x_train[i])
                assert leaf.cif is not None
                risk_sum[i] += leaf.cif[target].sum()
                tree_count[i] += 1

        scored = tree_count > 0
        if scored.sum() < 2:
            return None
        risk = risk_sum[scored] / tree_count[scored]
        # Build right-censored response for cause 1 (competing events = censored)
        cs_event = self._status_train[scored] == target
        y = Surv.right(self._time_train[scored], event=cs_event)
        return float(concordance_index(y, risk))

    def variable_importance(
        self,
        *,
        cause: int | str | None = None,
        n_repeats: int = 5,
        random_state: Any = None,
        format: str | None = None,
    ) -> Any:
        """Permutation variable importance on out-of-bag samples.

        For each covariate, its values are randomly permuted and the increase in
        OOB prediction error (`1 - concordance`) is measured.

        Parameters
        ----------
        cause
            Which cause to compute importance for. Defaults to the first cause.
        n_repeats
            Number of random permutations per covariate.
        random_state
            Seed or generator controlling permutations.
        format
            DataFrame backend for the returned table.

        Returns
        -------
        DataFrame
            One row per covariate with columns `term` and `importance`.
        """
        if not self.bootstrap:
            raise ValueError("variable_importance requires bootstrap=True.")

        cause_code = self._resolve_cause(cause) if cause is not None else self.cause_codes_[0]
        vimp_rng = np.random.default_rng(random_state)
        x = self._x_train
        time = self._time_train
        status = self._status_train

        def oob_risk(matrix: Array) -> tuple[Array, Array]:
            risk_sum = np.zeros(x.shape[0])
            count = np.zeros(x.shape[0])
            for root, oob in zip(self.trees_, self._oob_masks, strict=False):
                if not oob.any():
                    continue
                oob_idx = np.where(oob)[0]
                for i in oob_idx:
                    leaf = self._route_tree(root, matrix[i])
                    assert leaf.cif is not None
                    risk_sum[i] += leaf.cif[cause_code].sum()
                    count[i] += 1
            scored = count > 0
            return (
                risk_sum[scored] / np.where(count[scored] == 0, 1, count[scored]),
                scored,
            )

        from ._metrics import concordance_index
        from ._surv import Surv

        base_risk, scored = oob_risk(x)
        cs_event = status[scored] == cause_code
        y = Surv.right(time[scored], event=cs_event)
        base_error = 1.0 - concordance_index(y, base_risk)

        importances = np.zeros(x.shape[1])
        for feat in range(x.shape[1]):
            drop = 0.0
            for _ in range(n_repeats):
                permuted = x.copy()
                permuted[:, feat] = x[vimp_rng.permutation(x.shape[0]), feat]
                risk, sc = oob_risk(permuted)
                y_p = Surv.right(time[sc], event=(status[sc] == cause_code))
                drop += (1.0 - concordance_index(y_p, risk)) - base_error
            importances[feat] = drop / n_repeats

        order = np.argsort(importances)[::-1]
        columns = {
            "term": [self.feature_names_in_[i] for i in order],
            "importance": importances[order],
        }
        return to_dataframe(columns, format=format)


# ---------------------------------------------------------------------------
# Tidy / glance adapters
# ---------------------------------------------------------------------------


def _register_adapters() -> None:
    from .summaries import register_glance, register_tidier

    def _tidy_crf(model: CompetingRiskForest, *, format: str | None = None, **_: Any) -> Any:
        return model.variable_importance(format=format)

    def _glance_crf(model: CompetingRiskForest, *, format: str | None = None, **_: Any) -> Any:
        return to_dataframe(
            {
                "n": [model.n_],
                "nevent": [model.n_event_],
                "n_estimators": [model.n_estimators],
                "n_features": [model.n_features_in_],
                "n_causes": [len(model.cause_codes_)],
                "causes": [", ".join(str(s) for s in model.states_)],
                "oob_concordance": [model.oob_score_],
            },
            format=format,
        )

    register_tidier("greenwood._cr_forest.CompetingRiskForest", _tidy_crf)
    register_glance("greenwood._cr_forest.CompetingRiskForest", _glance_crf)


_register_adapters()
