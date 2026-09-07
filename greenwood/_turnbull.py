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

if TYPE_CHECKING:
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

    if surv.type is CensoringType.INTERVAL:
        assert surv.lower is not None
        return surv.lower, surv.stop
    if surv.type is CensoringType.RIGHT:
        event = surv.event
        lower = surv.stop.copy()
        upper = np.where(event, surv.stop, np.inf)
        return lower, upper
    if surv.type is CensoringType.LEFT:
        event = surv.event
        lower = np.where(event, 0.0, surv.stop)
        upper = np.where(event, surv.stop, np.inf)
        return lower, upper
    raise NotImplementedError(  # pragma: no cover - exhaustive over CensoringType
        f"Turnbull does not support {surv.type.value!r} responses."
    )


def _resolve_weights(surv: Surv, weights: Any) -> Array:
    """Resolve weights from explicit argument, `Surv.weights`, or unit weights."""
    if weights is not None:
        from ._surv import _to_1d_array

        return _to_1d_array(weights)
    if surv.weights is not None:
        return surv.weights
    return np.ones(surv.n)


@dataclass(frozen=True)
class _TurnbullBlock:
    """One stratum's fitted NPMLE."""

    label: object
    n: int
    interval_low: Array
    interval_high: Array
    prob_mass: Array
    survival: Array
    n_iter: int
    converged: bool


def _fit_turnbull_blocks(
    surv: Surv, by: Any, weights: Any, tol: float, max_iter: int
) -> list[_TurnbullBlock]:
    lower, upper = _to_interval_bounds(surv)
    weight = _resolve_weights(surv, weights)

    if by is None:
        labels: list[Any] = [None]
        masks = [np.ones(surv.n, dtype=bool)]
    else:
        from ._surv import _to_1d_array

        group_labels = _to_1d_array(by, dtype=object)
        if group_labels.shape[0] != surv.n:
            raise ValueError("`by` must have the same length as the response.")
        _, first_idx = np.unique(group_labels, return_index=True)
        ordered_levels = group_labels[np.sort(first_idx)]
        labels = list(ordered_levels)
        masks = [group_labels == lev for lev in ordered_levels]

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
    genuinely unidentified and `predict()`/`quantile()` return `nan` rather than interpolate.
    `rmst()`/`rmrl()` need a single number, so they instead fall back to Turnbull's own
    right-endpoint convention (every atom's mass resolves at `interval_high_`); see their
    docstrings for the resulting conservative (upper-bound) bias.

    To use this estimator, call `fit()` with a `Surv` response built via `Surv.interval()`
    (the general case), or `Surv.right()`/`Surv.left()` (degenerate cases: fitting a
    right-censored response through `Turnbull` reproduces `KaplanMeier` exactly, since there is
    then no genuine interval ambiguity). Left-truncated (`Surv.counting()`) and multi-state
    responses are not supported.

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
        Call `fit()` to produce a fitted estimator with cached results (`interval_low_`,
        `interval_high_`, `prob_mass_`, `survival_`), accessible as aligned arrays or exported
        to DataFrames.

    Details
    -------
    Call `fit` with a `Surv` response. The maximal intersection intervals are found via the
    Gentleman & Geyer (1994) construction: the finest partition induced by every subject's
    `lower`/`upper` bounds is grouped into runs of consecutive atoms that every subject's
    constraint either fully includes or fully excludes (only their combined mass is
    identified). Probabilities on that support are then found by the EM self-consistency
    algorithm, which is monotone in the likelihood and converges to the NPMLE.

    Examples
    --------
    Six subjects observed at irregular follow-up visits, so some events are only known to lie
    in a window rather than at an exact time. Two are exact deaths, one is right-censored (no
    event observed by the last visit), and three are genuinely interval-censored:

    ```{python}
    import greenwood as gw

    # Build an interval-censored response: (lower, upper] windows, inf marks right-censoring
    y = gw.Surv.interval(
        lower=[0, 4, 7, 0, 3, 5],
        upper=[4, float("inf"), 7, 2.5, 6, 5],
    )

    # Fit the Turnbull NPMLE
    tb = gw.Turnbull().fit(y)
    tb
    ```

    The fitted curve, one row per maximal intersection interval, is available via `to_frame`:

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
        from ._repr import align_table, whole

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

        if self._grouped:
            labels = [str(b.label) for b in self._blocks]
            table = align_table(headers, [row_for(b) for b in self._blocks], labels)
        else:
            table = align_table(headers, [row_for(self._blocks[0])])
        return "Turnbull (self-consistent NPMLE for interval-censored data)\n\n" + table

    def fit(self, surv: Surv, *, by: Any = None, weights: Any = None) -> Turnbull:
        r"""Fit Turnbull's NPMLE to interval-censored survival data.

        Computes the maximal intersection intervals and their probability masses from a
        `Surv` response, via EM self-consistency. Pass `by=` to fit separate curves per group
        (stratified analysis).

        Parameters
        ----------
        surv
            A `Surv` response built with `Surv.interval()`, `Surv.right()`, or `Surv.left()`.
            Left-truncated (`Surv.counting()`) and multi-state responses raise
            `NotImplementedError`.
        by
            Optional grouping variable (e.g., a column or array). Produces one fit per unique
            value of `by`. Default (`None`): a single, unstratified fit.
        weights
            Optional case weights. Must have the same length as `surv`. Default (`None`): uses
            `surv.weights` if present, otherwise unit weights.

        Returns
        -------
        Turnbull
            The fitted estimator itself (for method chaining), with cached results
            (`interval_low_`, `interval_high_`, `prob_mass_`, `survival_`, ...) as attributes.

        Examples
        --------
        ```{python}
        import greenwood as gw

        y = gw.Surv.interval(lower=[0, 4, 7, 0, 3, 5], upper=[4, float("inf"), 7, 2.5, 6, 5])
        tb = gw.Turnbull().fit(y)
        tb.survival_
        ```
        """
        self._blocks = _fit_turnbull_blocks(surv, by, weights, self.tol, self.max_iter)
        self._grouped = by is not None
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
    def strata_(self) -> Array | None:
        """Stratum labels for each row, or `None` for unstratified fits."""
        if not self._grouped:
            return None
        return np.concatenate(
            [np.full(b.interval_low.shape[0], b.label, dtype=object) for b in self._blocks]
        )

    # -- quantiles ------------------------------------------------------------

    def quantile(self, p: float) -> Any:
        r"""Return the `p`-quantile survival time per stratum.

        Parameters
        ----------
        p
            Quantile level between 0 and 1 (e.g. `p=0.5` for the median).

        Returns
        -------
        tuple or dict
            For a single stratum: `(estimate, lower, upper)`. `estimate` is `nan` when the
            crossing falls inside a non-degenerate (ambiguous) interval, in which case
            `lower`/`upper` are that interval's bounds rather than a sampling confidence bound.
            Otherwise `estimate == lower == upper`. For stratified fits: a `dict` keyed by
            stratum label, with values as above.

        Details
        -------
        The quantile is found by inverting the step-function survival curve, exactly as for
        `KaplanMeier`, but returning the *identifiability* bracket rather than ever guessing a
        point inside it. This is always a 3-tuple regardless of whether the crossing is
        ambiguous, so the return type does not depend on the data.

        Examples
        --------
        ```{python}
        import greenwood as gw

        y = gw.Surv.interval(lower=[0, 4, 7, 0, 3, 5], upper=[4, float("inf"), 7, 2.5, 6, 5])
        tb = gw.Turnbull().fit(y)

        # The first-quartile survival time (or its ambiguity bracket)
        tb.quantile(0.25)
        ```
        """
        level = 1.0 - p
        if not self._grouped:
            return _crossing(self._blocks[0], level)
        return {b.label: _crossing(b, level) for b in self._blocks}

    def median(self) -> Any:
        """Median survival time per stratum (the 0.5-quantile).

        A convenience wrapper around `quantile(0.5)`. See `quantile` for the return shape.

        Returns
        -------
        tuple or dict
            `(estimate, lower, upper)` for a single stratum, or a `dict` keyed by stratum label
            for stratified fits.

        Examples
        --------
        ```{python}
        import greenwood as gw

        y = gw.Surv.interval(lower=[0, 4, 7, 0, 3, 5], upper=[4, float("inf"), 7, 2.5, 6, 5])
        tb = gw.Turnbull().fit(y)
        tb.median()
        ```
        """
        return self.quantile(0.5)

    # -- restricted mean survival / residual life ------------------------------

    def rmst(self, tau: float) -> Any:
        r"""Restricted mean survival time up to `tau`, under the right-endpoint convention.

        Parameters
        ----------
        tau
            The upper time limit for the restriction. Must be positive.

        Returns
        -------
        float or dict
            The restricted mean survival time for a single stratum, or a `dict` keyed by stratum
            label for stratified fits.

        Details
        -------
        RMST is the area under the survival curve on $[0, \tau]$. Unlike `predict()` and
        `quantile()`, which report the genuine identifiability gap as `nan`, RMST needs a single
        number, so every atom's probability mass, ambiguous or not, is treated as resolving exactly
        at that atom's *right* endpoint (`interval_high_`). This is Turnbull's own convention for
        reporting a plottable curve from an otherwise partially-unidentified NPMLE. It is also the
        most conservative choice for RMST: placing mass as late as possible maximizes the area under
        the curve, so this systematically reports the *largest* RMST consistent with the data, not
        an unbiased point estimate. There is no variance estimator for it (no confidence interval is
        returned).

        Examples
        --------
        ```{python}
        import greenwood as gw

        y = gw.Surv.interval(lower=[0, 4, 7, 0, 3, 5], upper=[4, float("inf"), 7, 2.5, 6, 5])
        tb = gw.Turnbull().fit(y)
        tb.rmst(7)
        ```
        """
        if not self._grouped:
            b = self._blocks[0]
            return _rmst_value(b.interval_high, b.survival, float(tau))
        return {b.label: _rmst_value(b.interval_high, b.survival, float(tau)) for b in self._blocks}

    def rmrl(self, s: float, tau: float) -> Any:
        r"""Restricted mean residual life at `s`, under the right-endpoint convention.

        Parameters
        ----------
        s
            The landmark time. Must be non-negative.
        tau
            The upper time limit for the restriction. Must be greater than `s`.

        Returns
        -------
        float or dict
            The restricted mean residual life for a single stratum, or a `dict` keyed by stratum
            label for stratified fits. `nan` if everyone has resolved (under the same right-endpoint
            convention) by time `s`.

        Details
        -------
        Generalizes `rmst` to a later landmark:
        $\mathrm{RMRL}(s; \tau) = \int_s^\tau S(u)\,du / S(s)$, under the same right-endpoint
        convention (every atom's mass is treated as resolving at `interval_high_`); see `rmst` for
        why, and for the resulting conservative (upper-bound) bias.

        Examples
        --------
        ```{python}
        import greenwood as gw

        y = gw.Surv.interval(lower=[0, 4, 7, 0, 3, 5], upper=[4, float("inf"), 7, 2.5, 6, 5])
        tb = gw.Turnbull().fit(y)
        tb.rmrl(4, 7)
        ```
        """
        if tau <= s:
            raise ValueError(f"tau ({tau}) must be greater than s ({s}).")
        if s < 0.0:
            raise ValueError(f"s must be non-negative, got {s}.")

        if not self._grouped:
            b = self._blocks[0]
            return _rmrl_value(b.interval_high, b.survival, float(s), float(tau))
        return {
            b.label: _rmrl_value(b.interval_high, b.survival, float(s), float(tau))
            for b in self._blocks
        }

    # -- prediction -----------------------------------------------------------

    def predict(self, times: Any) -> Any:
        r"""Evaluate the survival curve at specified times.

        Parameters
        ----------
        times
            Query times at which to evaluate the curve. Can be a scalar or array-like of floats.

        Returns
        -------
        ndarray or dict
            For a single stratum: an array (matching `times`' shape) of survival estimates, with
            `nan` at any query time that falls strictly inside a non-degenerate (ambiguous)
            interval. For stratified fits: a `dict` keyed by stratum label.

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

        y = gw.Surv.interval(lower=[0, 4, 7, 0, 3, 5], upper=[4, float("inf"), 7, 2.5, 6, 5])
        tb = gw.Turnbull().fit(y)

        # nan at t=1 (strictly inside the ambiguous (0, 2.5) region)
        tb.predict([1, 2.5, 5, 7])
        ```
        """
        query = np.atleast_1d(np.asarray(times, dtype=float))

        def one(b: _TurnbullBlock) -> Array:
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

        if not self._grouped:
            return one(self._blocks[0])
        return {b.label: one(b) for b in self._blocks}

    # -- interop --------------------------------------------------------------

    def _table_columns(self) -> dict[str, Array]:
        cols: dict[str, Array] = {}
        if self._grouped:
            cols["strata"] = self.strata_  # type: ignore[assignment]
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
            A tidy table with columns `interval_low`, `interval_high`, `prob_mass`, `estimate`, and
            optionally `strata`.

        Raises
        ------
        ImportError
            If the requested (or, when auto-detecting, any) DataFrame library is not installed.

        Examples
        --------
        ```{python}
        import greenwood as gw

        y = gw.Surv.interval(lower=[0, 4, 7, 0, 3, 5], upper=[4, float("inf"), 7, 2.5, 6, 5])
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
    cols: dict[str, list[Any]] = {}
    if tb._grouped:
        cols["strata"] = [b.label for b in tb._blocks]
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
