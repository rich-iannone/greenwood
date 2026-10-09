r"""Turnbull's self-consistent NPMLE for interval-censored survival data.

Kaplan-Meier and Nelson-Aalen (`_nonparametric.py`) assume the event time is either exactly observed
or right-censored. Neither is valid when the event is only known to lie in a window `(lower, upper]`
(periodic follow-up, inspection data). Turnbull showed that the nonparametric maximum-likelihood
estimator (NPMLE) for that setting places its probability mass on a specific, data-determined set of
intervals, the *maximal intersection intervals*, computed here via the equivalent
"identical coverage" construction of Gentleman & Geyer (1994): the finest partition induced by the
data's endpoints is grouped into runs of consecutive atoms that every subject's `(lower, upper]`
constraint either fully includes or fully excludes, since only the total mass on such a run is
identified.

Outside these intervals the survival curve is exactly known (flat). Inside a non-degenerate one it
is not identified by the data. `Turnbull` reports both bounds of every such interval rather than
picking an arbitrary representative point, and `predict()`/`quantile()` propagate that ambiguity as
`nan` instead of silently interpolating.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt

from ._backends import to_dataframe
from ._outcome import bind_fit_inputs
from ._strata import resolve_strata

if TYPE_CHECKING:
    from ._outcome import Outcome
    from ._surv import Surv

__all__ = ["Turnbull"]

Array = npt.NDArray[Any]


def _candidate_atoms(lower: Array, upper: Array) -> tuple[Array, Array]:
    """Alternating point/open-interval atoms spanning the data's breakpoints.

    Returns `(atom_low, atom_high)`. For a point atom (a candidate exact-time mass point)
    `atom_low[k] == atom_high[k]`. For an open-interval atom (a candidate ambiguous region strictly
    between two consecutive breakpoints) `atom_low[k] < atom_high[k]`. Every atom from every
    subject's true event time must fall on one of these candidates, since the breakpoints are
    exactly the observed lower/upper bounds.
    """
    finite_upper = upper[np.isfinite(upper)]
    bp = np.unique(np.concatenate([lower, finite_upper]))
    if bp.size == 0:
        return np.empty(0), np.empty(0)
    atoms_low = [bp[0]]
    atoms_high = [bp[0]]
    for k in range(bp.size - 1):
        atoms_low.append(bp[k])
        atoms_high.append(bp[k + 1])
        atoms_low.append(bp[k + 1])
        atoms_high.append(bp[k + 1])
    return np.array(atoms_low), np.array(atoms_high)


def _build_alpha(lower: Array, upper: Array) -> tuple[Array, Array, Array]:
    """Maximal intersection intervals and the `(n, m)` containment indicator matrix.

    `alpha[i, j]` is `True` when Turnbull interval `j` is fully contained in subject `i`'s
    `(lower[i], upper[i]]`. Candidate atoms with no subject overlapping them at all are dropped
    (regions no observation touches can never carry mass). Adjacent atoms that every subject treats
    identically are merged, since the likelihood only depends on their combined mass. An exact
    observation (`lower[i] == upper[i]`) is a degenerate `(t, t]` interval, so it is matched to its
    own point atom by direct equality rather than the general open-closed containment rule.

    A right-censored subject (`upper[i] == inf`) additionally covers an implicit "never" atom beyond
    every finite breakpoint, appended as the last column of `alpha` (with no matching entry in
    `interval_low`/`interval_high`). Without it, the EM would be forced to resolve every censored
    subject's mass into whatever finite atoms exist, which breaks the classical identity between
    this estimator and Kaplan-Meier whenever the tail is censored: KM leaves that mass unresolved
    (the curve plateaus above 0) rather than forcing it onto the last observed atom.
    """
    n = lower.shape[0]
    has_inf_atom = bool(np.any(np.isinf(upper)))

    atoms_low, atoms_high = _candidate_atoms(lower, upper)
    if atoms_low.size == 0:
        interval_low, interval_high = np.empty(0), np.empty(0)
        alpha = np.zeros((n, 0), dtype=bool)
    else:
        is_point = atoms_low == atoms_high
        point_cover = (lower[:, None] < atoms_low[None, :]) & (atoms_low[None, :] <= upper[:, None])
        open_cover = (lower[:, None] <= atoms_low[None, :]) & (
            upper[:, None] >= atoms_high[None, :]
        )
        signature = np.where(is_point[None, :], point_cover, open_cover)

        exact = lower == upper
        exact_match = exact[:, None] & (lower[:, None] == atoms_low[None, :]) & is_point[None, :]
        signature = signature | exact_match

        covered = signature.any(axis=0)
        atoms_low, atoms_high = atoms_low[covered], atoms_high[covered]
        signature = signature[:, covered]
        if signature.shape[1] == 0:
            interval_low, interval_high = np.empty(0), np.empty(0)
            alpha = np.zeros((n, 0), dtype=bool)
        else:
            same_as_prev = np.all(signature[:, 1:] == signature[:, :-1], axis=0)
            group_start = np.concatenate(([True], ~same_as_prev))
            group_id = np.cumsum(group_start) - 1
            n_groups = int(group_id[-1]) + 1

            interval_low = np.empty(n_groups)
            interval_high = np.empty(n_groups)
            alpha = np.zeros((n, n_groups), dtype=bool)
            for g in range(n_groups):
                cols = np.nonzero(group_id == g)[0]
                interval_low[g] = atoms_low[cols[0]]
                interval_high[g] = atoms_high[cols[-1]]
                alpha[:, g] = signature[:, cols[0]]

    if has_inf_atom:
        alpha = np.concatenate([alpha, np.isinf(upper)[:, None]], axis=1)
    return interval_low, interval_high, alpha


def _em_turnbull(
    alpha: Array, weight: Array, *, tol: float, max_iter: int
) -> tuple[Array, int, bool]:
    r"""Self-consistency EM for the atom probabilities `s_j`.

    $$
    \mu_{ij} = \frac{\alpha_{ij} s_j}{\sum_k \alpha_{ik} s_k}, \qquad
    s_j \leftarrow \frac{\sum_i w_i \mu_{ij}}{\sum_i w_i}
    $$

    Monotone in the observed-data log-likelihood. Converges to the NPMLE.
    """
    _, m = alpha.shape
    if m == 0:
        return np.empty(0), 0, True

    s = np.full(m, 1.0 / m)
    alpha_f = alpha.astype(float)
    total_weight = float(weight.sum())
    n_iter = 0
    converged = False
    for it in range(1, max_iter + 1):
        n_iter = it
        numer = alpha_f * s[None, :]
        denom = numer.sum(axis=1, keepdims=True)
        with np.errstate(invalid="ignore", divide="ignore"):
            mu = np.where(denom > 0, numer / denom, 0.0)
        s_new = (weight[:, None] * mu).sum(axis=0) / total_weight
        if np.max(np.abs(s_new - s)) < tol:
            s = s_new
            converged = True
            break
        s = s_new
    return s, n_iter, converged


def _to_interval_bounds(surv: Surv) -> tuple[Array, Array]:
    """Reduce a supported `Surv` response to `(lower, upper]` bounds for every subject."""
    from ._surv import CensoringType

    if surv.is_multistate:
        raise NotImplementedError("Turnbull does not support multi-state/competing-risks data.")
    if surv.is_truncated:
        raise NotImplementedError(
            "Turnbull does not support left truncation (counting-process responses)."
        )

    # Left-censored rows are bounded below by time 0.
    if surv.type is CensoringType.INTERVAL:
        # R's interval coding: 0 right, 1 exact, 2 left, 3 interval (time1, time2].
        assert surv.lower is not None
        time1, code = surv.lower, surv.status
        lower = np.where(code == 2, 0.0, time1)
        upper = np.select([code == 0, code == 3], [np.inf, surv.stop], default=time1)
        return lower, upper
    if surv.type is CensoringType.RIGHT:
        event = surv.event
        lower = surv.stop.copy()
        upper = np.where(event, surv.stop, np.inf)
        return lower, upper
    if surv.type is CensoringType.LEFT:
        # R's meaning: status 1 is an exact event, status 0 is left-censored at `time`.
        event = surv.event
        lower = np.where(event, surv.stop, 0.0)
        return lower, surv.stop.copy()
    raise NotImplementedError(  # pragma: no cover - exhaustive over CensoringType
        f"Turnbull does not support {surv.type.value!r} responses."
    )


def _resolve_weights(surv: Surv, weights: Any) -> Array:
    """Resolve weights from the explicit argument, or unit weights."""
    if weights is not None:
        from ._ingest import to_1d_array as _to_1d_array

        return _to_1d_array(weights)
    return np.ones(surv.n)


@dataclass(frozen=True)
class _TurnbullBlock:
    """One stratum's fitted NPMLE."""

    label: str
    n: int
    interval_low: Array
    interval_high: Array
    prob_mass: Array
    survival: Array
    n_iter: int
    converged: bool


def _fit_turnbull_blocks(
    surv: Surv, strata: Array, weights: Any, tol: float, max_iter: int
) -> list[_TurnbullBlock]:
    """One NPMLE per curve label in `strata` ("all" for an ungrouped fit), in first-seen order."""
    lower, upper = _to_interval_bounds(surv)
    weight = _resolve_weights(surv, weights)

    labels: list[str] = list(dict.fromkeys(strata.tolist()))
    masks = [strata == label for label in labels]

    blocks: list[_TurnbullBlock] = []
    for label, mask in zip(labels, masks, strict=True):
        interval_low, interval_high, alpha = _build_alpha(lower[mask], upper[mask])
        s, n_iter, converged = _em_turnbull(alpha, weight[mask], tol=tol, max_iter=max_iter)
        # `s` may carry a trailing "never" (at-infinity) component beyond the finite atoms
        # (see `_build_alpha`). Only the finite atoms are reported as prob_mass/survival.
        n_finite = interval_low.shape[0]
        prob_mass = s[:n_finite]
        survival = 1.0 - np.cumsum(prob_mass)
        blocks.append(
            _TurnbullBlock(
                label=label,
                n=int(mask.sum()),
                interval_low=interval_low,
                interval_high=interval_high,
                prob_mass=prob_mass,
                survival=survival,
                n_iter=n_iter,
                converged=converged,
            )
        )
    return blocks


def _crossing(block: _TurnbullBlock, level: float) -> tuple[float, float, float]:
    """First atom at which survival drops to `<= level`, as `(estimate, lower, upper)`.

    `estimate` is `nan` when the crossing falls inside a non-degenerate (ambiguous) interval.
    `lower`/`upper` are then that interval's bounds rather than a confidence bound. When the
    crossing is exact (a point atom, or no ambiguity at that step), all three values coincide.
    """
    hit = np.nonzero(block.survival <= level)[0]
    if hit.size == 0:
        return float("nan"), float("nan"), float("nan")
    j = int(hit[0])
    lo = float(block.interval_low[j])
    hi = float(block.interval_high[j])
    if lo == hi:
        return lo, lo, lo
    return float("nan"), lo, hi


def _rmst_value(interval_high: Array, survival: Array, tau: float) -> float:
    r"""Area under the right-endpoint-convention curve on $[0, \tau]$.

    Every atom's mass, ambiguous or not, is treated as resolving exactly at
    `interval_high` (Turnbull's own convention for reporting a plottable curve), so the
    curve here is a well-defined step function even though `survival_`/`predict()` leave
    ambiguous regions as `nan`.
    """
    t = interval_high
    s = survival
    starts = np.concatenate([[0.0], t])
    heights = np.concatenate([[1.0], s])
    next_starts = np.concatenate([t, [tau]])
    widths = np.clip(np.minimum(next_starts, tau) - np.minimum(starts, tau), 0.0, None)
    return float((heights * widths).sum())


def _rmrl_value(interval_high: Array, survival: Array, s_time: float, tau: float) -> float:
    r"""Right-endpoint-convention restricted mean residual life at `s_time`, over $(s, \tau]$."""
    t = interval_high
    surv = survival
    idx = int(np.searchsorted(t, s_time, side="right")) - 1
    s_at = float(surv[idx]) if idx >= 0 else 1.0
    if s_at <= 0.0:
        return float("nan")

    starts = np.concatenate([[0.0], t])
    heights = np.concatenate([[1.0], surv])
    next_starts = np.concatenate([t, [tau]])
    lo = np.clip(starts, s_time, tau)
    hi = np.clip(next_starts, s_time, tau)
    area_window = float((heights * np.clip(hi - lo, 0.0, None)).sum())
    return area_window / s_at


def _predict_block(b: _TurnbullBlock, query: Array) -> Array:
    """Survival at `query`, `nan` strictly inside an ambiguous interval."""
    m = b.interval_high.shape[0]
    if m == 0:
        return np.ones_like(query)
    idx = np.searchsorted(b.interval_high, query, side="right") - 1
    out = np.where(idx >= 0, b.survival[idx.clip(min=0)], 1.0)

    idx2 = np.searchsorted(b.interval_high, query, side="left")
    valid = idx2 < m
    in_gap = np.zeros(query.shape, dtype=bool)
    clipped = idx2[valid].clip(max=m - 1)
    lo = b.interval_low[clipped]
    hi = b.interval_high[clipped]
    in_gap[valid] = (query[valid] > lo) & (query[valid] < hi) & (lo != hi)
    return np.where(in_gap, np.nan, out)


class Turnbull:
    r"""Turnbull's self-consistent NPMLE for interval-censored survival data.

    Kaplan-Meier requires knowing each subject's event time exactly (up to right-censoring).
    When follow-up is periodic instead (clinic visits, equipment inspections), all that is known
    is that the event fell in a window `(lower, upper]`. Turnbull's nonparametric maximum likelihood
    estimator handles this directly, together with mixtures of exact, left-, right-, and
    interval-censored observations in the same fit.

    Unlike Kaplan-Meier, the NPMLE's support is not identified everywhere: the data determine a
    set of *maximal intersection intervals*, and only the total probability mass on each one is
    identified, not its placement inside. `Turnbull` reports both bounds of every such interval.
    Where an interval degenerates to a single point (an exact death, or a region every subject's
    constraint resolves unambiguously), survival is known exactly there. Otherwise it is
    genuinely unidentified and `~~greenwood.Turnbull.predict()`/`quantile()` return `nan` rather
    than interpolate. `rmst()`/`rmrl()` need a single number, so they instead fall back to
    Turnbull's own right-endpoint convention (every atom's mass resolves at `interval_high_`); see
    their docstrings for the resulting conservative (upper-bound) bias.

    To use this estimator, call `~~greenwood.Turnbull.fit()` with an interval-censored `Surv`
    response, built with `gw.Surv(time=lower, time2=upper, type="interval2")` or from a
    `gw.event_time()` vector via `gw.as_surv()`. With a data frame, name its columns in a formula
    such as `"Surv(lower, upper, type='interval2')"` and pass `data=`. Right- and left-censored
    responses are degenerate cases. Fitting a right-censored response through `Turnbull` reproduces
    `KaplanMeier` exactly, since there is then no genuine interval ambiguity. Left-truncated
    (counting-process) and multi-state responses are not supported.

    Parameters
    ----------
    tol
        Convergence tolerance on the largest change in any atom's probability mass between EM
        iterations (default `1e-9`).
    max_iter
        Maximum number of EM iterations (default `10000`).

    Returns
    -------
    Fitted estimator
        Call `~~greenwood.Turnbull.fit()` to produce a fitted estimator with cached results
        (`interval_low_`, `interval_high_`, `prob_mass_`, `survival_`), accessible as aligned arrays
        or exported to DataFrames.

    Details
    -------
    Call `~~greenwood.Turnbull.fit` with a `Surv` response. The maximal intersection intervals are
    found via the Gentleman & Geyer (1994) construction: the finest partition induced by every
    subject's `lower`{.gd-no-link}/`upper` bounds is grouped into runs of consecutive atoms that
    every subject's constraint either fully includes or fully excludes (only their combined mass is
    identified). Probabilities on that support are then found by the EM self-consistency algorithm,
    which is monotone in the likelihood and converges to the NPMLE.

    Examples
    --------
    Six subjects observed at irregular follow-up visits, so some events are only known to lie
    in a window rather than at an exact time. Two are exact deaths, one is right-censored (no
    event observed by the last visit), and three are genuinely interval-censored:

    ```{python}
    import greenwood as gw

    # Build an interval-censored response: (lower, upper] windows, inf marks right-censoring
    y = gw.Surv(
        time=[0, 4, 7, 0, 3, 5],
        time2=[4, float("inf"), 7, 2.5, 6, 5],
        type="interval2",
    )

    # Fit the Turnbull NPMLE
    tb = gw.Turnbull().fit(y)
    tb
    ```

    The fitted curve, one row per maximal intersection interval, is available via
    `~~greenwood.Turnbull.to_frame`:

    ```{python}
    tb.to_frame(format="polars")
    ```
    """

    def __init__(self, *, tol: float = 1e-9, max_iter: int = 10000) -> None:
        if tol <= 0.0:
            raise ValueError(f"tol must be positive, got {tol}.")
        if max_iter <= 0:
            raise ValueError(f"max_iter must be positive, got {max_iter}.")
        self.tol = tol
        self.max_iter = max_iter

    def __repr__(self) -> str:
        if getattr(self, "_blocks", None) is None:
            return f"Turnbull(tol={self.tol!r}, max_iter={self.max_iter!r}) <unfitted>"
        from ._repr import align_table, dropped_footer, whole

        headers = ["n", "atoms", "ambiguous", "iters", "converged"]

        def row_for(b: _TurnbullBlock) -> list[str]:
            n_atoms = b.interval_low.shape[0]
            n_ambiguous = int(np.sum(b.interval_low != b.interval_high))
            return [
                whole(b.n),
                whole(n_atoms),
                whole(n_ambiguous),
                whole(b.n_iter),
                str(b.converged),
            ]

        labels = [b.label for b in self._blocks]
        table = align_table(headers, [row_for(b) for b in self._blocks], labels)
        return (
            "Turnbull (self-consistent NPMLE for interval-censored data)\n\n"
            + table
            + dropped_footer(self)
        )

    def fit(
        self, surv: Surv | Outcome | str, *, data: Any = None, by: Any = None, weights: Any = None
    ) -> Turnbull:
        r"""Fit Turnbull's NPMLE to interval-censored survival data.

        Computes the maximal intersection intervals and their probability masses from a
        `Surv` response, via EM self-consistency. Pass `by=` to fit separate curves per group
        (stratified analysis).

        Parameters
        ----------
        surv
            An interval-, right-, or left-censored `Surv` response built with `Surv()`.
            Left-truncated (counting-process) and multi-state responses raise
            `NotImplementedError`.
            An `Outcome` or a formula string such as `"Surv(lower, upper, type='interval2') ~ sex"`
            is also accepted, with its columns read from `data`. The right-hand side names the `by`
            column(s).
        by
            Optional grouping variable (e.g., a column or array). Produces one fit per unique
            value of `by`. Default (`None`): a single, unstratified fit.
        weights
            Optional case weights. Must have the same length as `surv`{.gd-no-link}. Default
            (`None`): unit weights.
        data
            A data frame (pandas, Polars, PyArrow, DuckDB, a lazy frame, ...) holding the columns
            named by the response, `by`, and `weights`. When `surv`{.gd-no-link} is an `Outcome` or
            a formula, rows with a missing value in any column used are dropped before fitting.

        Returns
        -------
        Turnbull
            The fitted estimator itself (for method chaining), with cached results
            (`interval_low_`, `interval_high_`, `prob_mass_`, `survival_`, ...) as attributes.

        Examples
        --------
        ```{python}
        import greenwood as gw

        y = gw.Surv(
            time=[0, 4, 7, 0, 3, 5],
            time2=[4, float("inf"), 7, 2.5, 6, 5],
            type="interval2",
        )
        tb = gw.Turnbull().fit(y)
        tb.survival_
        ```
        """
        bound = bind_fit_inputs(
            surv,
            data=data,
            labels={"by": by, "weights": weights},
            rhs_to="by",
            estimator="Turnbull",
        )
        surv = bound.surv
        strata, self._grouped = resolve_strata(bound, "by")
        weights = bound.labels["weights"]
        self._n_input = bound.n_input
        self.n_dropped_ = bound.n_dropped

        self._blocks = _fit_turnbull_blocks(surv, strata, weights, self.tol, self.max_iter)
        return self

    # -- aligned-array accessors ---------------------------------------------

    def _concat(self, attr: str) -> Array:
        return np.concatenate([getattr(b, attr) for b in self._blocks])

    @property
    def interval_low_(self) -> Array:
        """Lower bound of each maximal intersection interval."""
        return self._concat("interval_low")

    @property
    def interval_high_(self) -> Array:
        """Upper bound of each maximal intersection interval."""
        return self._concat("interval_high")

    @property
    def prob_mass_(self) -> Array:
        """Probability mass assigned to each maximal intersection interval."""
        return self._concat("prob_mass")

    @property
    def survival_(self) -> Array:
        """Survival estimate just after each interval (`1 - cumsum(prob_mass_)`)."""
        return self._concat("survival")

    @property
    def strata_(self) -> Array:
        """The curve label of each row: `"all"` for one curve, `"sex=1"` and so on for groups."""
        return np.concatenate(
            [np.full(b.interval_low.shape[0], b.label, dtype=object) for b in self._blocks]
        )

    # -- quantiles ------------------------------------------------------------

    def quantile(self, p: Any, *, format: str | None = None) -> Any:
        r"""Return survival-time quantiles, with their identifiability bounds, for each curve.

        Parameters
        ----------
        p
            One or more proportions between 0 and 1 (for example, `0.5` for the median).
        format
            Output format: `None` (default), `"pandas"`, `"polars"`, or `"pyarrow"`. When `None`,
            a backend is auto-detected (Polars, then Pandas, then PyArrow).

        Returns
        -------
        pandas.DataFrame, polars.DataFrame, or pyarrow.Table
            One row per curve and proportion, with columns `strata`, `prob`, `time`, `time_low`,
            and `time_high`. When the crossing falls inside a non-degenerate (ambiguous) interval,
            `time` is `nan` and `time_low`/`time_high` are that interval's bounds. They bound what
            the data can identify and are not confidence limits. Otherwise all three are equal.
            The layout is the same for one curve (`strata` is `"all"`) as for several.

        Details
        -------
        The quantile is found by inverting the step-function survival curve, exactly as for
        `KaplanMeier`, but reporting the identifiability bracket rather than guessing a point
        inside it.

        Examples
        --------
        ```{python}
        import greenwood as gw

        y = gw.Surv(
            time=[0, 4, 7, 0, 3, 5],
            time2=[4, float("inf"), 7, 2.5, 6, 5],
            type="interval2",
        )
        tb = gw.Turnbull().fit(y)

        # The first-quartile survival time (or its ambiguity bracket)
        tb.quantile(p=0.25, format="polars")
        ```
        """
        probs = [float(v) for v in np.atleast_1d(np.asarray(p, dtype=float))]
        if any(not 0.0 <= v <= 1.0 for v in probs):
            raise ValueError("p must be between 0 and 1.")
        cols: dict[str, list[Any]] = {
            k: [] for k in ("strata", "prob", "time", "time_low", "time_high")
        }
        for b in self._blocks:
            for prob in probs:
                point, low, high = _crossing(b, 1.0 - prob)
                cols["strata"].append(b.label)
                cols["prob"].append(prob)
                cols["time"].append(point)
                cols["time_low"].append(low)
                cols["time_high"].append(high)
        return to_dataframe(cols, format=format)

    def median(self, *, format: str | None = None) -> Any:
        """Median survival time, with its identifiability bounds, for each curve.

        Parameters
        ----------
        format
            Output format: `None` (default), `"pandas"`, `"polars"`, or `"pyarrow"`. When `None`,
            a backend is auto-detected (Polars, then Pandas, then PyArrow).

        Returns
        -------
        pandas.DataFrame, polars.DataFrame, or pyarrow.Table
            One row per curve, with columns `strata`, `time`, `time_low`, and `time_high`. See
            `~~greenwood.Turnbull.quantile()` for what the bounds mean.

        Examples
        --------
        ```{python}
        import greenwood as gw

        y = gw.Surv(
            time=[0, 4, 7, 0, 3, 5],
            time2=[4, float("inf"), 7, 2.5, 6, 5],
            type="interval2",
        )
        tb = gw.Turnbull().fit(y)
        tb.median(format="polars")
        ```
        """
        cols: dict[str, list[Any]] = {k: [] for k in ("strata", "time", "time_low", "time_high")}
        for b in self._blocks:
            point, low, high = _crossing(b, 0.5)
            cols["strata"].append(b.label)
            cols["time"].append(point)
            cols["time_low"].append(low)
            cols["time_high"].append(high)
        return to_dataframe(cols, format=format)

    # -- restricted mean survival / residual life ------------------------------

    def rmst(self, tau: float, *, format: str | None = None) -> Any:
        r"""Restricted mean survival time up to `tau`, under the right-endpoint convention.

        Parameters
        ----------
        tau
            The end of the window. Must be positive.
        format
            Output format: `None` (default), `"pandas"`, `"polars"`, or `"pyarrow"`. When `None`,
            a backend is auto-detected (Polars, then Pandas, then PyArrow).

        Returns
        -------
        pandas.DataFrame, polars.DataFrame, or pyarrow.Table
            One row per curve, with columns `strata`, `tau`, and `estimate`. The layout is the
            same for one curve (`strata` is `"all"`) as for several.

        Details
        -------
        RMST is the area under the survival curve on $[0, \tau]$. Unlike
        `~~greenwood.Turnbull.predict()` and `quantile()`, which report the genuine
        identifiability gap as `nan`, RMST needs a single number, so every atom's probability
        mass, ambiguous or not, is treated as resolving exactly at that atom's *right* endpoint
        (`interval_high_`). This is Turnbull's own convention for reporting a plottable curve
        from an otherwise partially-unidentified NPMLE. It is also the most conservative choice
        for RMST: placing mass as late as possible maximizes the area under the curve, so this
        reports the *largest* RMST consistent with the data, not an unbiased point estimate.
        There is no variance estimator for it, so no standard error or confidence limits are
        reported.

        Examples
        --------
        ```{python}
        import greenwood as gw

        y = gw.Surv(
            time=[0, 4, 7, 0, 3, 5],
            time2=[4, float("inf"), 7, 2.5, 6, 5],
            type="interval2",
        )
        tb = gw.Turnbull().fit(y)
        tb.rmst(tau=7, format="polars")
        ```
        """
        cols: dict[str, list[Any]] = {"strata": [], "tau": [], "estimate": []}
        for b in self._blocks:
            cols["strata"].append(b.label)
            cols["tau"].append(float(tau))
            cols["estimate"].append(_rmst_value(b.interval_high, b.survival, float(tau)))
        return to_dataframe(cols, format=format)

    def rmrl(self, s: float, tau: float, *, format: str | None = None) -> Any:
        r"""Restricted mean residual life at `s`, under the right-endpoint convention.

        Parameters
        ----------
        s
            The landmark time. Must be non-negative.
        tau
            The end of the window. Must be greater than `s`.
        format
            Output format: `None` (default), `"pandas"`, `"polars"`, or `"pyarrow"`. When `None`,
            a backend is auto-detected (Polars, then Pandas, then PyArrow).

        Returns
        -------
        pandas.DataFrame, polars.DataFrame, or pyarrow.Table
            One row per curve, with columns `strata`, `s`, `tau`, and `estimate`. The estimate is
            `nan` for a curve where everyone has resolved (under the same right-endpoint
            convention) by `s`.

        Details
        -------
        Generalizes `rmst` to a later landmark:
        $\mathrm{RMRL}(s; \tau) = \int_s^\tau S(u)\,du / S(s)$, under the same right-endpoint
        convention (every atom's mass is treated as resolving at `interval_high_`). See `rmst`
        for why, and for the resulting conservative (upper-bound) bias.

        Examples
        --------
        ```{python}
        import greenwood as gw

        y = gw.Surv(
            time=[0, 4, 7, 0, 3, 5],
            time2=[4, float("inf"), 7, 2.5, 6, 5],
            type="interval2",
        )
        tb = gw.Turnbull().fit(y)
        tb.rmrl(s=4, tau=7, format="polars")
        ```
        """
        if tau <= s:
            raise ValueError(f"tau ({tau}) must be greater than s ({s}).")
        if s < 0.0:
            raise ValueError(f"s must be non-negative, got {s}.")
        cols: dict[str, list[Any]] = {"strata": [], "s": [], "tau": [], "estimate": []}
        for b in self._blocks:
            cols["strata"].append(b.label)
            cols["s"].append(float(s))
            cols["tau"].append(float(tau))
            cols["estimate"].append(_rmrl_value(b.interval_high, b.survival, float(s), float(tau)))
        return to_dataframe(cols, format=format)

    # -- prediction -----------------------------------------------------------

    def predict(self, times: Any, *, format: str | None = None) -> Any:
        r"""Read the survival curve at given times, for each curve.

        Parameters
        ----------
        times
            One or more times at which to read the curve.
        format
            Output format: `None` (default), `"pandas"`, `"polars"`, or `"pyarrow"`. When `None`,
            a backend is auto-detected (Polars, then Pandas, then PyArrow).

        Returns
        -------
        pandas.DataFrame, polars.DataFrame, or pyarrow.Table
            One row per curve and time, with columns `strata`, `time`, and `estimate`. The
            estimate is `nan` at any time strictly inside a non-degenerate (ambiguous) interval.
            The layout is the same for one curve (`strata` is `"all"`) as for several.

        Details
        -------
        Outside every ambiguous interval, the curve is a well-defined, right-continuous step
        function, exactly as for `KaplanMeier`. Strictly inside a non-degenerate interval
        `(lower, upper)`, the true survival value is not identified by the data, so `nan` is
        returned there rather than an arbitrary interpolated guess.

        Examples
        --------
        ```{python}
        import greenwood as gw

        y = gw.Surv(
            time=[0, 4, 7, 0, 3, 5],
            time2=[4, float("inf"), 7, 2.5, 6, 5],
            type="interval2",
        )
        tb = gw.Turnbull().fit(y)

        # nan at t=1 (strictly inside the ambiguous (0, 2.5) region)
        tb.predict(times=[1, 2.5, 5, 7], format="polars")
        ```
        """
        query = np.atleast_1d(np.asarray(times, dtype=float))
        cols: dict[str, list[Any]] = {"strata": [], "time": [], "estimate": []}
        for b in self._blocks:
            cols["strata"].extend([b.label] * query.shape[0])
            cols["time"].extend(query.tolist())
            cols["estimate"].extend(_predict_block(b, query).tolist())
        return to_dataframe(cols, format=format)

    # -- interop --------------------------------------------------------------

    def _table_columns(self) -> dict[str, Array]:
        cols: dict[str, Array] = {"strata": self.strata_}
        cols["interval_low"] = self.interval_low_
        cols["interval_high"] = self.interval_high_
        cols["prob_mass"] = self.prob_mass_
        cols["estimate"] = self.survival_
        return cols

    def to_frame(self, *, format: str | None = None) -> Any:
        """Return the fitted NPMLE as a DataFrame.

        Exports one row per maximal intersection interval, with its bounds, probability mass, and
        the survival estimate just after it.

        Parameters
        ----------
        format
            Output format: `None` (default), `"pandas"`, `"polars"`, or `"pyarrow"`. When `None`, a
            backend is auto-detected (Polars, then Pandas, then PyArrow).

        Returns
        -------
        pandas.DataFrame, polars.DataFrame, or pyarrow.Table
            A tidy table with columns `strata`, `interval_low`, `interval_high`, `prob_mass`, and
            `estimate`. The layout is the same for one curve (`strata` is `"all"`) as for several.

        Raises
        ------
        ImportError
            If the requested (or, when auto-detecting, any) DataFrame library is not installed.

        Examples
        --------
        ```{python}
        import greenwood as gw

        y = gw.Surv(
            time=[0, 4, 7, 0, 3, 5],
            time2=[4, float("inf"), 7, 2.5, 6, 5],
            type="interval2",
        )
        tb = gw.Turnbull().fit(y)
        tb.to_frame(format="polars")
        ```
        """
        return to_dataframe(self._table_columns(), format=format)


def _tidy_turnbull(tb: Turnbull, *, format: str | None = None, **_: Any) -> Any:
    """broom-style `tidy`: one row per maximal intersection interval."""
    return tb.to_frame(format=format)


def _glance_turnbull(tb: Turnbull, *, format: str | None = None, **_: Any) -> Any:
    """broom-style `glance`: one row per stratum with counts and EM convergence info."""
    cols: dict[str, list[Any]] = {"strata": [b.label for b in tb._blocks]}
    cols["n"] = [float(b.n) for b in tb._blocks]
    cols["n_atoms"] = [float(b.interval_low.shape[0]) for b in tb._blocks]
    cols["n_ambiguous"] = [float(np.sum(b.interval_low != b.interval_high)) for b in tb._blocks]
    cols["n_iter"] = [float(b.n_iter) for b in tb._blocks]
    cols["converged"] = [bool(b.converged) for b in tb._blocks]
    return to_dataframe(cols, format=format)


def _register_adapters() -> None:
    from .summaries import register_glance, register_tidier

    register_tidier("greenwood._turnbull.Turnbull", _tidy_turnbull)
    register_glance("greenwood._turnbull.Turnbull", _glance_turnbull)


_register_adapters()
