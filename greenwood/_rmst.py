r"""Restricted Mean Survival Time (RMST) comparisons between groups.

Compares RMST across two or more groups, providing:

- RMST differences with confidence intervals
- Hypothesis tests for RMST equality
- Stratified RMST comparisons
- Ratio and percentage difference estimands

All methods use Greenwood-based variance estimation, consistent with the
Kaplan-Meier RMST calculations in `_nonparametric`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt
from scipy.stats import norm

from ._backends import to_dataframe
from ._nonparametric import _Block, _rmst_block

if TYPE_CHECKING:
    from ._surv import Surv

__all__ = ["rmst_test", "rmst_diff", "RMSTResult", "pairwise_rmst_test"]

Array = npt.NDArray[Any]


@dataclass(frozen=True)
class RMSTResult:
    """Results of an RMST comparison test or difference calculation.

    This class stores the results of RMST group comparisons in a structured format,
    including point estimates, confidence intervals, and hypothesis test statistics.

    Attributes
    ----------
    estimate
        The point estimate of RMST difference, ratio, or percentage difference between groups.
    lower_ci
        Lower bound of the confidence interval for the estimate.
    upper_ci
        Upper bound of the confidence interval for the estimate.
    se
        Standard error of the estimate.
    statistic
        Test statistic (z-score for Wald test) for the null hypothesis of no difference.
    p_value
        Two-tailed p-value for the hypothesis test. Small values (typically < 0.05)
        indicate significant differences between groups.
    method
        Human-readable description of the comparison method, e.g.,
        `"RMST difference (tau=365)"`.
    group1
        Label of the first group (minuend in difference).
    group2
        Label of the second group (subtrahend in difference).
    rmst1
        RMST estimate for group 1 at tau.
    se1
        Standard error of RMST for group 1.
    rmst2
        RMST estimate for group 2 at tau.
    se2
        Standard error of RMST for group 2.
    tau
        The restriction time tau used in the RMST calculation.
    estimand
        The type of estimand: `"difference"`, `"ratio"`, or `"percentage_difference"`.
    stratified
        Whether this is a stratified comparison (True) or pooled (False).
    conf_level
        Confidence level used for interval estimation (the default is `0.95`).

    Examples
    --------
    Compare RMST between two groups:

    ```{python}
    import greenwood as gw

    lung = gw.load_dataset("lung", backend="polars")
    y = gw.Surv.right(lung["time"], event=(lung["status"] == 2))

    result = gw.rmst_test(y, tau=365, group=lung["sex"])
    result
    ```

    Access individual components:

    ```{python}
    result.estimate  # difference between groups
    result.p_value   # significance
    result.se        # standard error
    ```
    """

    estimate: float
    se: float
    lower_ci: float
    upper_ci: float
    statistic: float
    p_value: float
    method: str
    group1: Any
    group2: Any
    rmst1: float
    se1: float
    rmst2: float
    se2: float
    tau: float
    estimand: str = "difference"
    stratified: bool = False
    conf_level: float = 0.95

    def __repr__(self) -> str:
        ci_pct = int(self.conf_level * 100)
        return (
            f"RMSTResult(method={self.method!r}, estimate={self.estimate:.4f}, "
            f"se={self.se:.4f}, {ci_pct}% CI=[{self.lower_ci:.4f}, {self.upper_ci:.4f}], "
            f"p_value={self.p_value:.4g})"
        )


def _subset_surv(surv: Surv, mask: npt.NDArray[np.bool_]) -> Surv:
    """Subset a Surv object by a boolean mask."""
    from ._surv import Surv as _Surv

    if surv.type.value == "right":
        return _Surv.right(surv.stop[mask], surv.event[mask])
    if surv.type.value == "counting":
        return _Surv.counting(surv.entry[mask], surv.stop[mask], surv.event[mask])
    raise NotImplementedError(f"_subset_surv does not support Surv type {surv.type.value!r}")


def _stratified_rmst_group_values(
    surv: Surv,
    tau: float,
    group: Any,
    strata: Any,
    label1: Any,
    label2: Any,
) -> tuple[float, float, float, float]:
    """Return inverse-variance-pooled (rmst1, se1, rmst2, se2) across strata.

    Per-stratum differences are pooled with inverse-variance weights based on the
    variance of the difference (`var_s = se1_s^2 + se2_s^2`). This matches the
    standard stratified RMST estimator (e.g. survRM2):

        w_s     = 1 / (se1_s^2 + se2_s^2)
        W       = sum(w_s)
        rmst1   = sum(w_s * rmst1_s) / W          (consistent display value)
        rmst2   = sum(w_s * rmst2_s) / W          (consistent display value)
        se_k    = sqrt(sum(w_s^2 * se_k_s^2)) / W (propagated SE)

    Strata where either group is absent are skipped.
    """
    from ._surv import _to_1d_array

    group_arr = _to_1d_array(group, dtype=object)
    strata_arr = _to_1d_array(strata, dtype=object)
    strata_levels = sorted(set(strata_arr.tolist()), key=lambda v: (str(type(v)), v))

    rmst1_vals: list[float] = []
    se1_vals: list[float] = []
    rmst2_vals: list[float] = []
    se2_vals: list[float] = []

    for s in strata_levels:
        s_mask = strata_arr == s
        s_group = group_arr[s_mask]
        present = set(s_group.tolist())
        if label1 not in present or label2 not in present:
            continue  # Skip strata that lack one of the two groups

        surv_sub = _subset_surv(surv, s_mask)
        rmst_dict, _ = _rmst_group_values(surv_sub, tau, s_group)

        rmst1_s, se1_s = rmst_dict[label1]
        rmst2_s, se2_s = rmst_dict[label2]

        if se1_s <= 0 or se2_s <= 0:
            continue  # Degenerate stratum (all events at one time point)

        rmst1_vals.append(rmst1_s)
        se1_vals.append(se1_s)
        rmst2_vals.append(rmst2_s)
        se2_vals.append(se2_s)

    if not rmst1_vals:
        raise ValueError(
            "No usable strata found: each stratum must contain both group levels with "
            "positive standard errors."
        )

    se1_arr = np.array(se1_vals)
    se2_arr = np.array(se2_vals)
    rmst1_arr = np.array(rmst1_vals)
    rmst2_arr = np.array(rmst2_vals)

    # Weights based on variance of the per-stratum difference
    var_diff = se1_arr**2 + se2_arr**2
    w = 1.0 / var_diff
    W = float(w.sum())

    # Pooled RMST display values (same weights for both groups → difference is consistent)
    rmst1_pooled = float(np.dot(w, rmst1_arr) / W)
    rmst2_pooled = float(np.dot(w, rmst2_arr) / W)

    # Propagated SEs using the difference-based weights
    se1_pooled = float(np.sqrt(np.dot(w**2, se1_arr**2)) / W)
    se2_pooled = float(np.sqrt(np.dot(w**2, se2_arr**2)) / W)

    return rmst1_pooled, se1_pooled, rmst2_pooled, se2_pooled


def _rmst_group_values(
    surv: Surv, tau: float, group: Any
) -> tuple[dict[Any, tuple[float, float]], list[Any]]:
    """Compute RMST and SE for each group.

    Returns (rmst_dict, group_labels) where rmst_dict maps group label to (rmst, se) tuple.
    """
    from ._core import event_table

    et = event_table(surv, group=group, weights=None)

    # The event table has a 'strata' column when group is provided
    if et.strata is None:
        raise ValueError("event_table should have strata when group is provided")

    # Get unique group labels from strata
    group_labels = list(dict.fromkeys(et.strata.tolist()))  # Preserve order of first appearance

    rmst_dict: dict[Any, tuple[float, float]] = {}

    # For each group, build a _Block and compute RMST
    for label in group_labels:
        mask = et.strata == label

        # Extract data for this group from event table
        group_time = et.time[mask]
        group_n_risk = et.n_risk[mask]
        group_n_event = et.n_event[mask]
        group_n_censor = et.n_censor[mask]

        # Build survival curve for this group
        with np.errstate(divide="ignore", invalid="ignore"):
            factor = np.where(group_n_risk > 0, 1.0 - group_n_event / group_n_risk, 1.0)
        surv_hat = np.cumprod(factor)

        # Build a _Block for this group
        block = _Block(
            label=label,
            time=group_time,
            n_risk=group_n_risk,
            n_event=group_n_event,
            n_censor=group_n_censor,
            surv=surv_hat,
            std_error=np.full_like(group_time, np.nan),  # Not used for RMST
            conf_low=np.full_like(group_time, np.nan),  # Not used for RMST
            conf_high=np.full_like(group_time, np.nan),  # Not used for RMST
            cumhaz=np.full_like(group_time, np.nan),  # Not used for RMST
            cumhaz_var=np.full_like(group_time, np.nan),  # Not used for RMST
        )

        rmst_val, se_val = _rmst_block(block, float(tau))
        rmst_dict[label] = (rmst_val, se_val)

    return rmst_dict, group_labels


def rmst_test(
    surv: Surv,
    tau: float,
    group: Any,
    *,
    estimand: str = "difference",
    strata: Any | None = None,
    conf_level: float = 0.95,
) -> RMSTResult:
    r"""Test for equality of RMST across two or more groups.

    Compares restricted mean survival time (RMST) up to a fixed time tau across groups
    using a z-test or t-test. Provides point estimate, standard error, confidence interval,
    and p-value for the null hypothesis of equal RMST.

    For two groups, this is equivalent to a z-test on the RMST difference (default) or
    log-ratio (if estimand="ratio").

    Parameters
    ----------
    surv
        A right-censored `Surv` response (time-to-event data).
    tau
        The restriction time, typically a clinically relevant horizon (e.g., 365, 1825).
    group
        Group membership for each observation. Can be array-like or categorical variable.
    estimand
        Type of estimand: `"difference"` (default, RMST1 - RMST2), `"ratio"` (RMST1 / RMST2),
        or `"percentage_difference"` ((RMST1 - RMST2) / RMST2 * 100).
    strata
        (Optional) Stratification variable for stratified RMST comparison. If provided,
        per-group RMST estimates are computed separately within each stratum and then
        combined using inverse-variance weights (Kaplan-Meier Greenwood variance).
        Strata in which either group is absent are skipped.
    conf_level
        Confidence level for confidence intervals (the default is `0.95` for 95% CI).

    Returns
    -------
    RMSTResult
        A result object containing estimate, standard error, confidence interval,
        test statistic, and p-value.

    Details
    -------
    For two groups (i=1,2), the RMST difference is:

    $$
    \Delta = \mathrm{RMST}_1(\tau) - \mathrm{RMST}_2(\tau)
    $$

    with standard error:

    $$
    \mathrm{SE}(\Delta) = \sqrt{\mathrm{SE}(\mathrm{RMST}_1)^2 + \mathrm{SE}(\mathrm{RMST}_2)^2}
    $$

    assuming independence. The z-statistic is $Z = \Delta / \mathrm{SE}(\Delta)$, with
    two-tailed p-value from the standard normal.

    For ratio estimand, the log-ratio variance uses the delta method:

    $$
    \mathrm{SE}(\log R) = \sqrt{\frac{\mathrm{SE}_1^2}{\mathrm{RMST}_1^2} +
    \frac{\mathrm{SE}_2^2}{\mathrm{RMST}_2^2}}
    $$

    Examples
    --------
    Test RMST difference between two treatment groups:

    ```{python}
    import greenwood as gw

    lung = gw.load_dataset("lung", backend="polars")
    y = gw.Surv.right(lung["time"], event=(lung["status"] == 2))
    result = gw.rmst_test(y, tau=365, group=lung["sex"])
    result.estimate   # RMST difference
    result.p_value    # significance
    ```

    Using ratio estimand:

    ```{python}
    result_ratio = gw.rmst_test(y, tau=365, group=lung["sex"], estimand="ratio")
    ```
    """
    # Get RMST values and SEs for each group
    rmst_dict, group_labels_ordered = _rmst_group_values(surv, tau, group, strata)

    # Sort group labels for consistent ordering
    group_labels = sorted(set(group_labels_ordered), key=lambda v: (str(type(v)), v))

    if len(group_labels) < 2:
        raise ValueError("RMST comparison requires at least two groups")

    if len(group_labels) > 2:
        raise ValueError(
            "rmst_test supports two groups; use rmst_pairwise_test for multiple groups"
        )

    if estimand not in {"difference", "ratio", "percentage_difference"}:
        raise ValueError(
            f"estimand must be 'difference', 'ratio', or 'percentage_difference', got {estimand!r}"
        )

    # Get RMST values and SEs for each group
    rmst_dict, _ = _rmst_group_values(surv, tau, group, strata)

    label1, label2 = group_labels
    rmst1, se1 = rmst_dict[label1]
    rmst2, se2 = rmst_dict[label2]

    # Compute estimate and SE based on estimand
    z_critical = norm.ppf(1.0 - (1.0 - conf_level) / 2.0)

    lower_ci: float
    upper_ci: float
    statistic: float

    if estimand == "difference":
        estimate = rmst1 - rmst2
        se = np.sqrt(se1**2 + se2**2)
        lower_ci = estimate - z_critical * se
        upper_ci = estimate + z_critical * se
        statistic = estimate / se if se > 0 else np.inf
    elif estimand == "ratio":
        if rmst2 <= 0:
            raise ValueError("RMST2 must be positive for ratio estimand")
        estimate = rmst1 / rmst2
        # Log-ratio SE using delta method
        se_log_ratio = np.sqrt((se1 / rmst1) ** 2 + (se2 / rmst2) ** 2)
        se = estimate * se_log_ratio
        # For CI, use log scale then exponentiate
        log_estimate = np.log(estimate)
        log_lower = log_estimate - z_critical * se_log_ratio
        log_upper = log_estimate + z_critical * se_log_ratio
        lower_ci = float(np.exp(log_lower))
        upper_ci = float(np.exp(log_upper))
        statistic = float(log_estimate / se_log_ratio)
    else:  # percentage_difference
        if rmst2 <= 0:
            raise ValueError("RMST2 must be positive for percentage_difference estimand")
        estimate = (rmst1 - rmst2) / rmst2 * 100.0
        # SE = (1/RMST2) * SE(RMST1 - RMST2) * 100
        se = np.sqrt(se1**2 + se2**2) / rmst2 * 100.0
        lower_ci = estimate - z_critical * se
        upper_ci = estimate + z_critical * se
        statistic = estimate / se if se > 0 else np.inf

    p_value = float(2.0 * (1.0 - norm.cdf(np.abs(statistic))))

    method = f"RMST {estimand} (tau={tau})"
    if strata is not None:
        method = f"Stratified {method}"

    return RMSTResult(
        estimate=estimate,
        se=se,
        lower_ci=lower_ci,
        upper_ci=upper_ci,
        statistic=statistic,
        p_value=p_value,
        method=method,
        group1=label1,
        group2=label2,
        rmst1=rmst1,
        se1=se1,
        rmst2=rmst2,
        se2=se2,
        tau=tau,
        estimand=estimand,
        stratified=strata is not None,
        conf_level=conf_level,
    )


def rmst_diff(
    surv: Surv,
    tau: float,
    group: Any,
    *,
    strata: Any | None = None,
    conf_level: float = 0.95,
) -> Any:
    """Compute RMST difference between two groups with confidence interval.

    Convenience function that calls `rmst_test()` with `estimand="difference"` and returns
    a DataFrame with the comparison results.

    Parameters
    ----------
    surv
        A right-censored `Surv` response.
    tau
        The restriction time.
    group
        Group membership.
    strata
        (Optional) Stratification variable.
    conf_level
        Confidence level for intervals.

    Returns
    -------
    DataFrame or dict
        Comparison results in tabular format.
    """
    result = rmst_test(
        surv, tau, group, estimand="difference", strata=strata, conf_level=conf_level
    )

    data = {
        "group1": [result.group1],
        "group2": [result.group2],
        "rmst1": [result.rmst1],
        "rmst2": [result.rmst2],
        "difference": [result.estimate],
        "se": [result.se],
        "lower_ci": [result.lower_ci],
        "upper_ci": [result.upper_ci],
        "statistic": [result.statistic],
        "p_value": [result.p_value],
    }

    return to_dataframe(data)


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


def pairwise_rmst_test(
    surv: Surv,
    tau: float,
    group: Any,
    *,
    estimand: str = "difference",
    strata: Any | None = None,
    correction: str = "holm",
    conf_level: float = 0.95,
    format: str | None = None,
) -> Any:
    r"""Pairwise RMST tests for all group pairs with multiple-comparison correction.

    Compares RMST between all pairs of groups, with optional multiple-comparison adjustment.
    This answers the question: "Which pairs of groups have significantly different RMST?"
    when you have more than two groups.

    Parameters
    ----------
    surv
        A right-censored `Surv` response (time-to-event data).
    tau
        The restriction time for RMST calculation.
    group
        Group labels, one per observation. Can be array-like or categorical variable.
        Must have at least 2 unique levels to create pairs.
    estimand
        Type of estimand: `"difference"` (default), `"ratio"`, or `"percentage_difference"`.
    strata
        (Optional) Stratification variable. Each pairwise test is stratified by this factor.
    correction
        Multiple-comparison adjustment: `"holm"` (default), `"bh"`, `"bonferroni"`, or `"none"`.
    conf_level
        Confidence level for intervals (the default is `0.95`).
    format
        Output format: None (auto-detect), `"pandas"`, `"polars"`, or `"pyarrow"`.

    Returns
    -------
    DataFrame
        One row per pair of groups with columns for group1, group2, RMST estimates,
        difference/ratio, confidence interval, test statistic, p-value, and adjusted p-value.

    Examples
    --------
    Compare RMST across multiple groups with pairwise comparisons:

    ```{python}
    import greenwood as gw

    lung = gw.load_dataset("lung", backend="polars")
    y = gw.Surv.right(lung["time"], event=(lung["status"] == 2))
    # If there are multiple groups, e.g., by stage:
    # result = gw.pairwise_rmst_test(y, tau=365, group=lung["stage"])
    ```
    """
    import itertools

    from ._surv import Surv, _to_1d_array

    group_arr = _to_1d_array(group, dtype=object)
    groups = sorted(set(group_arr.tolist()), key=lambda v: (str(type(v)), v))

    if len(groups) < 2:
        raise ValueError("pairwise_rmst_test needs at least two groups.")

    rows: list[dict[str, Any]] = []
    raw_p: list[float] = []

    for g1, g2 in itertools.combinations(groups, 2):
        # Create a subset with only the two groups
        mask = (group_arr == g1) | (group_arr == g2)

        # Create subset Surv object based on type
        if surv.type.value == "right":
            surv_sub = Surv.right(surv.stop[mask], surv.event[mask])
        elif surv.type.value == "counting":
            surv_sub = Surv.counting(surv.entry[mask], surv.stop[mask], surv.event[mask])
        else:
            raise NotImplementedError(f"pairwise_rmst_test does not support {surv.type.value}")

        group_sub = group_arr[mask]

        if strata is not None:
            from ._surv import _to_1d_array as to_1d

            strata_arr = to_1d(strata, dtype=object)
            strata_sub = strata_arr[mask]
        else:
            strata_sub = None

        # Run test on subset
        result = rmst_test(
            surv_sub,
            tau,
            group_sub,
            estimand=estimand,
            strata=strata_sub,
            conf_level=conf_level,
        )

        raw_p.append(result.p_value)
        rows.append(
            {
                "group1": result.group1,
                "group2": result.group2,
                "rmst1": result.rmst1,
                "rmst2": result.rmst2,
                "estimate": result.estimate,
                "se": result.se,
                "lower_ci": result.lower_ci,
                "upper_ci": result.upper_ci,
                "statistic": result.statistic,
                "p_value": result.p_value,
            }
        )

    for row, p_adj in zip(rows, _p_adjust(raw_p, correction), strict=True):
        row["p_adjusted"] = p_adj

    columns = {key: [row[key] for row in rows] for key in rows[0]}
    return to_dataframe(columns, format=format)
