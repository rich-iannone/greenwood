"""Group comparison tests: the log-rank statistic and the G-rho (Fleming-Harrington) family.

`logrank_test` compares survival across two or more groups using the weighted log-rank
statistic. The weight is the Fleming-Harrington `S(t-)^rho * (1 - S(t-))^gamma` evaluated
on the pooled Kaplan-Meier estimate, so `rho=0, gamma=0` is the standard log-rank test and
`rho=1, gamma=0` is the Peto-Peto (Wilcoxon-type) test. This matches R's
`survival::survdiff` (which exposes the `rho` parameter).
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt
from scipy.stats import chi2

from ._backends import to_dataframe

if TYPE_CHECKING:
    from ._surv import Surv

__all__ = ["logrank_test", "pairwise_logrank_test", "TestResult"]

Array = npt.NDArray[Any]


@dataclass(frozen=True)
class TestResult:
    """The outcome of a log-rank group comparison test.

    This class stores the results of `logrank_test` or `pairwise_logrank_test` in a
    structured format. Access test statistics, significance (p-value), and per-group
    observed vs. expected event counts.

    Attributes
    ----------
    statistic
        The chi-square test statistic. Larger values indicate stronger evidence against the
        null hypothesis of equal survival across groups.
    df
        Degrees of freedom for the chi-square distribution (number of groups minus one for
        `logrank_test`, always 1 for pairwise tests).
    p_value
        Upper-tail chi-square p-value. The probability of observing a chi-square statistic
        this large or larger under the null hypothesis of equal survival. Small p-values
        (typically p < 0.05) indicate significant differences between groups.
    method
        Human-readable description of the test method and its configuration, e.g.,
        "Log-rank test", "Stratified log-rank test", "G-rho test (rho=1, gamma=0)".
    observed
        Dictionary mapping each group label to its observed (actual) weighted event count.
        Useful for understanding which groups contribute more events.
    expected
        Dictionary mapping each group label to its expected event count under the null
        hypothesis of equal survival. Comparison of observed vs. expected reveals which
        groups have more or fewer events than expected.

    Notes
    -----
    For a significant result (p_value < 0.05), examine the `observed` and `expected`
    dictionaries to see which groups experienced more or fewer events than expected. Groups
    with observed > expected have worse (shorter) survival; groups with observed < expected
    have better (longer) survival.

    Examples
    --------
    Run a log-rank test and examine results:

    ```{python}
    import greenwood as gw

    lung = gw.load_dataset("lung")
    y = gw.Surv.right(lung["time"], event=(lung["status"] == 2))
    result = gw.logrank_test(y, group=lung["sex"])
    result
    ```

    Access individual components. The chi-square statistic:

    ```{python}
    result.statistic
    ```

    The p-value for significance:

    ```{python}
    result.p_value
    ```

    Observed event counts per group (actual events in data):

    ```{python}
    result.observed
    ```

    Expected event counts per group (under null hypothesis):

    ```{python}
    result.expected
    ```

    Test description:

    ```{python}
    result.method
    ```
    """

    statistic: float
    df: int
    p_value: float
    method: str
    observed: dict[Any, float] = field(default_factory=dict)
    expected: dict[Any, float] = field(default_factory=dict)

    def __repr__(self) -> str:
        return (
            f"TestResult(method={self.method!r}, statistic={self.statistic:.4f}, "
            f"df={self.df}, p_value={self.p_value:.4g})"
        )


def _at_risk(entry: Array, exit_: Array, weight: Array, times: Array) -> Array:
    """Weighted number at risk at each of `times`: entered before t and not yet exited."""
    e_order = np.argsort(entry, kind="stable")
    e_cumw = np.concatenate(([0.0], np.cumsum(weight[e_order])))
    entered_before = e_cumw[np.searchsorted(entry[e_order], times, side="left")]

    x_order = np.argsort(exit_, kind="stable")
    x_cumw = np.concatenate(([0.0], np.cumsum(weight[x_order])))
    exited_before = x_cumw[np.searchsorted(exit_[x_order], times, side="left")]
    return entered_before - exited_before


def _logrank_uv(
    entry: Array,
    exit_: Array,
    event: Array,
    weight: Array,
    labels: Array,
    groups: list[Any],
    rho: float,
    gamma: float,
) -> tuple[Array, Array, Array]:
    """Observed, expected, and the group variance-covariance for one (sub)sample.

    Uses the pooled, left-continuous Kaplan-Meier for the Fleming-Harrington weight and the
    hypergeometric variance. Returns `(observed, expected, var)` over `groups`; a group with
    no members contributes zeros.
    """
    n_groups = len(groups)
    times = np.unique(exit_[event])
    if times.size == 0:
        z = np.zeros(n_groups)
        return z, z, np.zeros((n_groups, n_groups))

    n_risk_g = np.empty((n_groups, times.size))
    n_event_g = np.zeros((n_groups, times.size))
    for j, label in enumerate(groups):
        mask = labels == label
        n_risk_g[j] = _at_risk(entry[mask], exit_[mask], weight[mask], times)
        ev = mask & event
        idx = np.searchsorted(times, exit_[ev])
        np.add.at(n_event_g[j], idx, weight[ev])

    n_risk = n_risk_g.sum(axis=0)
    n_event = n_event_g.sum(axis=0)

    with np.errstate(divide="ignore", invalid="ignore"):
        surv_pool = np.cumprod(np.where(n_risk > 0, 1.0 - n_event / n_risk, 1.0))
    surv_left = np.concatenate(([1.0], surv_pool[:-1]))
    w = surv_left**rho * (1.0 - surv_left) ** gamma

    with np.errstate(divide="ignore", invalid="ignore"):
        expected_rate = np.where(n_risk > 0, n_event / n_risk, 0.0)
    observed = (w * n_event_g).sum(axis=1)
    expected = (w * n_risk_g * expected_rate).sum(axis=1)

    with np.errstate(divide="ignore", invalid="ignore"):
        common = np.where(
            n_risk > 1,
            w**2 * n_event * (n_risk - n_event) / (n_risk**2 * (n_risk - 1.0)),
            0.0,
        )
    var = np.zeros((n_groups, n_groups))
    for j in range(n_groups):
        for k in range(n_groups):
            if j == k:
                var[j, k] = np.sum(common * n_risk_g[j] * (n_risk - n_risk_g[j]))
            else:
                var[j, k] = -np.sum(common * n_risk_g[j] * n_risk_g[k])
    return observed, expected, var


def _logrank_statistic(
    entry: Array,
    exit_: Array,
    event: Array,
    weight: Array,
    labels: Array,
    groups: list[Any],
    strata: Array | None,
    rho: float,
    gamma: float,
) -> tuple[float, int, Array, Array]:
    """Compute the (optionally stratified) log-rank statistic over `groups`.

    A stratified test sums the observed, expected, and variance contributions across strata
    before forming `chisq = U' V^-1 U` on a dropped-one basis (matching R `survdiff` with a
    `strata()` term).
    """
    n_groups = len(groups)
    observed = np.zeros(n_groups)
    expected = np.zeros(n_groups)
    var = np.zeros((n_groups, n_groups))
    if strata is None:
        observed, expected, var = _logrank_uv(
            entry, exit_, event, weight, labels, groups, rho, gamma
        )
    else:
        for s in np.unique(strata):
            m = strata == s
            o, e, v = _logrank_uv(
                entry[m], exit_[m], event[m], weight[m], labels[m], groups, rho, gamma
            )
            observed += o
            expected += e
            var += v

    u = observed - expected
    statistic = float(u[:-1] @ np.linalg.solve(var[:-1, :-1], u[:-1]))
    return statistic, n_groups - 1, observed, expected


def logrank_test(
    surv: Surv, group: Any, *, rho: float = 0.0, gamma: float = 0.0, strata: Any = None
) -> TestResult:
    """Compare survival across groups using the weighted log-rank (G-rho) test.

    Tests whether survival curves differ significantly across two or more groups using a
    chi-square test based on weighted event counts. The test is flexible: with default weights
    (Fleming-Harrington rho=0, gamma=0), it gives equal weight to all event times (standard
    log-rank). With rho=1, gamma=0 (Peto-Peto), it emphasizes early events where more subjects
    are at risk. Other rho/gamma combinations allow custom emphasis on different phases of
    follow-up.

    The test compares observed vs. expected event counts under the null hypothesis of equal
    survival. A large chi-square statistic indicates the groups differ; p-values are
    interpreted as the probability of seeing such a statistic or larger if survival is truly
    equal.

    **Stratification**: Optionally stratify by a nuisance variable (e.g., site, gender) to
    compute the test within each stratum, then combine results. This controls for confounding
    while testing group differences.

    Parameters
    ----------
    surv
        A `Surv` response object representing censored survival times. Supports right-censored
        data (standard time-to-event) or counting-process format (interval-based data with
        entry/exit times). Constructed with `Surv.right()`, `Surv.counting()`, or
        `Surv.multistate()`.
    group
        Group labels, one per observation. Can be a Narwhals series (Polars/Pandas), 1-D
        array, or Python sequence. Labels can be strings, integers, or other hashable types.
        Must have the same length as `surv`.
    rho, gamma
        Fleming-Harrington weight exponents applied to the pooled Kaplan-Meier survival
        `S(t-)` at each event time. The weight is `S(t-)^rho * (1-S(t-))^gamma`.

        - `rho=0, gamma=0` (default): Standard log-rank test. Equal weight across all times.
        - `rho=1, gamma=0`: Peto-Peto (Wilcoxon) test. Emphasizes early events.
        - `rho=0, gamma=1`: Tarone-Ware. Alternative early-event emphasis.
        - Other (rho, gamma): Flexible emphasis. Higher values emphasize the chosen phase.

    strata
        Optional stratifying factor, one per observation. Same length as `surv`. When
        provided, the test is computed separately within each stratum, then combined
        (stratified test). Use to control for confounding or variable that affects baseline
        hazard but not group differences. Example: stratify by site to account for
        site-specific differences in survival while testing an overall group effect.

    Returns
    -------
    TestResult
        A result object with attributes:

        - `statistic`: Chi-square test statistic.
        - `df`: Degrees of freedom (number of groups minus one).
        - `p_value`: Upper-tail chi-square p-value. Small values indicate survival curves differ.
        - `method`: Description of the test (e.g., "Log-rank test", "Stratified log-rank test",
          "G-rho test (rho=1, gamma=0)").
        - `observed`: Dictionary mapping group labels to observed weighted event counts.
        - `expected`: Dictionary mapping group labels to expected event counts under null.

    Notes
    -----
    The log-rank test uses the hypergeometric variance for the chi-square statistic, matching
    R's `survival::survdiff`. The pooled Kaplan-Meier survivor curve from all groups combined
    is used to compute the Fleming-Harrington weights, ensuring the test is consistently
    weighted regardless of group sample sizes.

    Counting-process data (with entry times) are fully supported, allowing stratification and
    left-truncation (delayed entry).

    Examples
    --------
    Test whether survival differs between the two sexes in the bundled `lung` dataset:

    ```{python}
    import greenwood as gw

    lung = gw.load_dataset("lung")
    y = gw.Surv.right(lung["time"], event=(lung["status"] == 2))
    result = gw.logrank_test(y, group=lung["sex"])
    result
    ```

    Extract individual components from the result:

    ```{python}
    result.statistic  # Chi-square statistic
    ```

    ```{python}
    result.p_value  # P-value for significance
    ```

    ```{python}
    result.observed  # Observed event counts per group
    ```

    Use the Peto-Peto (Wilcoxon) weighted test to emphasize differences in early survival:

    ```{python}
    gw.logrank_test(y, group=lung["sex"], rho=1, gamma=0)
    ```

    Run a stratified test to control for institution (if available in data):

    ```{python}
    # gw.logrank_test(y, group=lung["sex"], strata=lung["institution"])
    ```
    """
    from ._surv import CensoringType, _to_1d_array

    if surv.type not in (CensoringType.RIGHT, CensoringType.COUNTING):
        raise NotImplementedError(
            f"logrank_test supports right-censored and counting-process responses, "
            f"not {surv.type.value!r}."
        )

    labels_all = _to_1d_array(group, dtype=object)
    if labels_all.shape[0] != surv.n:
        raise ValueError("`group` must have the same length as the response.")
    strata_arr = None
    if strata is not None:
        strata_arr = _to_1d_array(strata, dtype=object)
        if strata_arr.shape[0] != surv.n:
            raise ValueError("`strata` must have the same length as the response.")

    entry = surv.entry
    exit_ = surv.stop
    event = surv.event
    weight = surv.weights if surv.weights is not None else np.ones(surv.n)

    groups = sorted(set(labels_all.tolist()), key=lambda v: (str(type(v)), v))
    if len(groups) < 2:
        raise ValueError("logrank_test needs at least two groups.")
    if np.count_nonzero(event) == 0:
        raise ValueError("No events in the data; the test is undefined.")

    statistic, df, observed, expected = _logrank_statistic(
        entry, exit_, event, weight, labels_all, groups, strata_arr, rho, gamma
    )
    p_value = float(chi2.sf(statistic, df))

    if rho == 0.0 and gamma == 0.0:
        method = "Log-rank test"
    else:
        method = f"G-rho test (rho={rho}, gamma={gamma})"
    if strata is not None:
        method = f"Stratified {method[0].lower()}{method[1:]}"

    return TestResult(
        statistic=statistic,
        df=df,
        p_value=p_value,
        method=method,
        observed=dict(zip(groups, observed.tolist(), strict=True)),
        expected=dict(zip(groups, expected.tolist(), strict=True)),
    )


def _p_adjust(pvalues: list[float], method: str) -> list[float]:
    """Adjust p-values for multiple comparisons: `holm`, `bh`, `bonferroni`, or `none`."""
    p = np.asarray(pvalues, dtype=float)
    m = p.size
    if method == "none":
        return p.tolist()
    if method == "bonferroni":
        return np.minimum(p * m, 1.0).tolist()
    order = np.argsort(p)
    out = np.empty(m)
    if method == "holm":
        running = 0.0
        for rank, i in enumerate(order):
            running = max(running, (m - rank) * p[i])
            out[i] = min(running, 1.0)
        return out.tolist()
    if method == "bh":  # Benjamini-Hochberg
        running = 1.0
        for rank in range(m - 1, -1, -1):
            i = order[rank]
            running = min(running, m / (rank + 1) * p[i])
            out[i] = min(running, 1.0)
        return out.tolist()
    raise ValueError(f"Unknown correction {method!r}; use 'holm', 'bh', 'bonferroni', or 'none'.")


def pairwise_logrank_test(
    surv: Surv,
    group: Any,
    *,
    rho: float = 0.0,
    gamma: float = 0.0,
    strata: Any = None,
    correction: str = "holm",
    format: str | None = None,
) -> Any:
    """Pairwise log-rank tests for all group pairs with multiple-comparison correction.

    Runs the log-rank test on every pair of groups, then adjusts p-values to control for
    multiple testing. This answers the question: "Which pairs of groups have significantly
    different survival?" when you have more than two groups.

    After the global log-rank test (via `logrank_test`) indicates groups differ, this
    pairwise test reveals which pairs are significantly different and by how much. P-values
    are adjusted across all pairs using a chosen correction method to control the false
    discovery rate or family-wise error rate.

    **Typical workflow**: First run `logrank_test` to test overall group differences. If
    significant, use `pairwise_logrank_test` to identify which pairs differ. The adjusted
    p-values account for testing multiple pairs from the same data.

    Parameters
    ----------
    surv
        A `Surv` response object representing censored survival times. Supports right-censored
        data or counting-process format. Constructed with `Surv.right()`, `Surv.counting()`,
        or `Surv.multistate()`.
    group
        Group labels, one per observation. Can be a Narwhals series, 1-D array, or Python
        sequence. Must have at least 3 unique levels (to create multiple pairs). Must have
        the same length as `surv`.
    rho, gamma
        Fleming-Harrington weight exponents for the log-rank test (same as `logrank_test`).
        Default `(0, 0)` gives standard log-rank; `(1, 0)` gives Peto-Peto (emphasizes early
        events).
    strata
        Optional stratifying factor. When provided, each pairwise test is stratified by this
        factor (computed within each stratum, then combined). Use to control for confounding.
    correction
        Multiple-comparison adjustment method applied across all pairwise p-values:

        - `"holm"` (default): Controls family-wise error rate. Conservative; recommended for
          small numbers of pairs (fewer than ~10).
        - `"bh"`: Benjamini-Hochberg false-discovery rate. Less conservative; recommended for
          many pairs. Allows more false positives but focuses on their rate.
        - `"bonferroni"`: Bonferroni correction. Very conservative; adjusted p = raw p × m,
          where m is the number of pairs.
        - `"none"`: No adjustment. Use only if you're testing a single pre-planned pair
          (though use `logrank_test` directly in that case).
    format
        Output format: `None` (default), `"pandas"`, `"polars"`, or `"pyarrow"`. When
        `None`, a backend is auto-detected (Polars, then Pandas, then PyArrow).

    Returns
    -------
    pandas.DataFrame, polars.DataFrame, or pyarrow.Table
        One row per pair of groups with columns:

        - `group1`, `group2`: The pair of group labels being compared.
        - `statistic`: Chi-square test statistic for the pair.
        - `p_value`: Raw (unadjusted) log-rank p-value for the pair.
        - `p_adjusted`: Adjusted p-value after multiple-comparison correction. Use this for
          significance testing (e.g., p_adjusted < 0.05).

    Notes
    -----
    The number of pairs tested is C(k, 2) = k(k-1)/2, where k is the number of groups. For
    k=3, that's 3 pairs; for k=5, that's 10 pairs. Larger numbers of pairs can reduce power
    per comparison (wider adjusted confidence intervals), so keep the number of groups
    reasonable when possible.

    The adjustment method affects stringency: Holm controls false discovery more strictly
    (lower type-I error, higher type-II error), while Benjamini-Hochberg is more permissive
    (higher type-I error rate overall, but controls the proportion of false discoveries).

    Examples
    --------
    Test pairwise survival differences among the four cell types in the `veteran` dataset.
    A global log-rank test first shows that cell types differ overall, but doesn't say which
    pairs differ:

    ```{python}
    import greenwood as gw

    vet = gw.load_dataset("veteran")
    y = gw.Surv.right(vet["time"], event=vet["status"])
    gw.logrank_test(y, group=vet["celltype"])
    ```

    The pairwise test compares all six pairs of cell types and returns a DataFrame with the
    test statistic, raw p-value, and adjusted p-value for each pair. Use `p_adjusted` for
    significance testing:

    ```{python}
    pairs = gw.pairwise_logrank_test(y, group=vet["celltype"])
    pairs
    ```

    Filter to significant pairs (adjusted p-value < 0.05):

    ```{python}
    pairs = gw.pairwise_logrank_test(y, group=vet["celltype"], format="pandas")
    pairs[pairs["p_adjusted"] < 0.05]
    ```

    Use the Peto-Peto (Wilcoxon) weighting to emphasize early survival differences:

    ```{python}
    gw.pairwise_logrank_test(y, group=vet["celltype"], rho=1)
    ```

    Use Benjamini-Hochberg adjustment (less conservative) if you're interested in which pairs
    show evidence of differences (false-discovery rate control rather than family-wise error):

    ```{python}
    gw.pairwise_logrank_test(y, group=vet["celltype"], correction="bh")
    ```
    """
    from ._surv import CensoringType, _to_1d_array

    if surv.type not in (CensoringType.RIGHT, CensoringType.COUNTING):
        raise NotImplementedError(
            f"pairwise_logrank_test supports right-censored and counting-process responses, "
            f"not {surv.type.value!r}."
        )
    labels_all = _to_1d_array(group, dtype=object)
    if labels_all.shape[0] != surv.n:
        raise ValueError("`group` must have the same length as the response.")
    strata_arr = None
    if strata is not None:
        strata_arr = _to_1d_array(strata, dtype=object)
        if strata_arr.shape[0] != surv.n:
            raise ValueError("`strata` must have the same length as the response.")

    entry, exit_, event = surv.entry, surv.stop, surv.event
    weight = surv.weights if surv.weights is not None else np.ones(surv.n)
    groups = sorted(set(labels_all.tolist()), key=lambda v: (str(type(v)), v))
    if len(groups) < 2:
        raise ValueError("pairwise_logrank_test needs at least two groups.")

    rows: list[dict[str, Any]] = []
    raw_p: list[float] = []
    for a, b in itertools.combinations(groups, 2):
        m = (labels_all == a) | (labels_all == b)
        stat, dfree, _, _ = _logrank_statistic(
            entry[m],
            exit_[m],
            event[m],
            weight[m],
            labels_all[m],
            [a, b],
            None if strata_arr is None else strata_arr[m],
            rho,
            gamma,
        )
        p = float(chi2.sf(stat, dfree))
        raw_p.append(p)
        rows.append({"group1": a, "group2": b, "statistic": stat, "p_value": p})

    for row, p_adj in zip(rows, _p_adjust(raw_p, correction), strict=True):
        row["p_adjusted"] = p_adj
    columns = {key: [row[key] for row in rows] for key in rows[0]}
    return to_dataframe(columns, format=format)
