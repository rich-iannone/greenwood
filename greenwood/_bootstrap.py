"""Bootstrap confidence intervals for Kaplan-Meier summary statistics.

Provides non-parametric bootstrap resampling for quantities that lack simple closed-form confidence
intervals, such as median survival differences, RMST differences between groups, and survival
probabilities at fixed time points. Supports percentile, normal, and BCa (bias-corrected and
accelerated) interval types.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt
from scipy.stats import norm

if TYPE_CHECKING:
    from ._nonparametric import KaplanMeier
    from ._surv import Surv

Array = npt.NDArray[Any]

__all__ = ["bootstrap", "BootstrapResult"]

_VALID_STATISTICS = frozenset(
    {"median", "rmst", "quantile", "survival", "median_diff", "rmst_diff", "survival_diff"}
)
_DIFF_STATISTICS = frozenset({"median_diff", "rmst_diff", "survival_diff"})
_VALID_CI_TYPES = frozenset({"percentile", "normal", "bca"})


@dataclass(frozen=True)
class BootstrapResult:
    """Result of a bootstrap confidence interval computation.

    Attributes
    ----------
    estimate
        Point estimate from the original (non-resampled) data.
    se
        Bootstrap standard error (standard deviation of the bootstrap distribution).
    conf_low
        Lower confidence limit.
    conf_high
        Upper confidence limit.
    conf_level
        Confidence level used for the interval.
    ci_type
        Type of confidence interval (`"percentile"`, `"normal"`, or `"bca"`).
    n_boot
        Number of bootstrap replicates.
    distribution
        The full bootstrap distribution (array of length `n_boot=`).
    """

    estimate: float
    se: float
    conf_low: float
    conf_high: float
    conf_level: float
    ci_type: str
    n_boot: int
    distribution: Array

    def __repr__(self) -> str:
        from ._repr import align_table, fixed

        headers = ["estimate", "se", f"{self.conf_level}LCL", f"{self.conf_level}UCL"]
        row = [
            fixed(self.estimate),
            fixed(self.se),
            fixed(self.conf_low),
            fixed(self.conf_high),
        ]
        table = align_table(headers, [row])
        return f"BootstrapResult (n_boot={self.n_boot}, ci_type={self.ci_type!r})\n\n" + table

    def to_frame(self, *, format: str | None = None) -> Any:
        """Export the result as a single-row DataFrame.

        Parameters
        ----------
        format
            Backend format: `"pandas"`, `"polars"`, or `"pyarrow"`. Default (`None`) uses the first
            available backend.
        """
        from ._backends import to_dataframe

        return to_dataframe(
            {
                "estimate": [self.estimate],
                "se": [self.se],
                "conf_low": [self.conf_low],
                "conf_high": [self.conf_high],
            },
            format=format,
        )


def _resolve_statistic(
    name: str,
    *,
    tau: float | None,
    p: float | None,
    times: float | None,
    is_grouped: bool,
) -> Callable[[KaplanMeier], float]:
    """Convert a built-in statistic name to a callable on a fitted KaplanMeier."""
    if name == "median":
        if is_grouped:
            raise ValueError("Use 'median_diff' for grouped bootstrap (by= is set).")
        return lambda km: float(km.median())

    if name == "rmst":
        if tau is None:
            raise ValueError("tau= is required for statistic='rmst'.")
        if is_grouped:
            raise ValueError("Use 'rmst_diff' for grouped bootstrap (by= is set).")
        _tau = tau
        return lambda km: float(km.rmst(_tau))

    if name == "quantile":
        if p is None:
            raise ValueError("p= is required for statistic='quantile'.")
        if is_grouped:
            raise ValueError("statistic='quantile' is not supported with by=.")
        _p = p
        return lambda km: float(km.quantile(_p))

    if name == "survival":
        if times is None:
            raise ValueError("times= is required for statistic='survival'.")
        if is_grouped:
            raise ValueError("Use 'survival_diff' for grouped bootstrap (by= is set).")
        _t = float(times)
        return lambda km: float(km.predict([_t])[0])

    if name == "median_diff":
        if not is_grouped:
            raise ValueError("by= is required for statistic='median_diff'.")
        return lambda km: _grouped_diff(km.median())

    if name == "rmst_diff":
        if not is_grouped:
            raise ValueError("by= is required for statistic='rmst_diff'.")
        if tau is None:
            raise ValueError("tau= is required for statistic='rmst_diff'.")
        _tau = tau
        return lambda km: _grouped_diff(km.rmst(_tau))

    if name == "survival_diff":
        if not is_grouped:
            raise ValueError("by= is required for statistic='survival_diff'.")
        if times is None:
            raise ValueError("times= is required for statistic='survival_diff'.")
        _t = float(times)

        def _surv_diff(km: KaplanMeier) -> float:
            pred_dict = km.predict([_t])
            keys = sorted(pred_dict.keys(), key=str)
            return float(pred_dict[keys[0]][0]) - float(pred_dict[keys[1]][0])

        return _surv_diff

    raise ValueError(f"Unknown statistic {name!r}. Choose from: {sorted(_VALID_STATISTICS)}.")


def _grouped_diff(result: dict[Any, float] | Any) -> float:
    """Extract the difference (group1 - group2) from a dict-keyed result."""
    if not isinstance(result, dict):
        raise TypeError("Expected a dict from a grouped KaplanMeier, got a scalar.")
    keys = sorted(result.keys(), key=str)
    if len(keys) != 2:
        raise ValueError(f"Diff statistics require exactly 2 groups, got {len(keys)}: {keys}.")
    return float(result[keys[0]]) - float(result[keys[1]])


def _percentile_ci(distribution: Array, conf_level: float) -> tuple[float, float]:
    alpha = 1.0 - conf_level
    return (
        float(np.nanpercentile(distribution, 100 * alpha / 2)),
        float(np.nanpercentile(distribution, 100 * (1 - alpha / 2))),
    )


def _normal_ci(estimate: float, se: float, conf_level: float) -> tuple[float, float]:
    alpha = 1.0 - conf_level
    z = float(norm.ppf(1 - alpha / 2))
    return (estimate - z * se, estimate + z * se)


def _bca_ci(
    distribution: Array, estimate: float, jackknife_values: Array, conf_level: float
) -> tuple[float, float]:
    """Bias-corrected and accelerated (BCa) confidence interval."""
    alpha = 1.0 - conf_level

    valid = np.isfinite(distribution)
    if valid.sum() < 2:
        return (float("nan"), float("nan"))

    z0 = float(norm.ppf(np.mean(distribution[valid] < estimate)))

    jk_mean = np.mean(jackknife_values)
    jk_diff = jk_mean - jackknife_values
    numer = np.sum(jk_diff**3)
    denom = 6.0 * np.sum(jk_diff**2) ** 1.5
    a = float(numer / denom) if denom > 0 else 0.0

    z_lo = float(norm.ppf(alpha / 2))
    z_hi = float(norm.ppf(1 - alpha / 2))

    def adjusted_quantile(z_alpha: float) -> float:
        numer_val = z0 + z_alpha
        adj = z0 + numer_val / (1 - a * numer_val)
        return float(norm.cdf(adj))

    p_lo = adjusted_quantile(z_lo)
    p_hi = adjusted_quantile(z_hi)

    p_lo = np.clip(p_lo, 0.0, 1.0)
    p_hi = np.clip(p_hi, 0.0, 1.0)

    return (
        float(np.nanpercentile(distribution, 100 * p_lo)),
        float(np.nanpercentile(distribution, 100 * p_hi)),
    )


def bootstrap(
    surv: Surv,
    statistic: str | Callable[[KaplanMeier], float],
    *,
    by: Any = None,
    weights: Any = None,
    n_boot: int = 1000,
    conf_level: float = 0.95,
    ci_type: str = "percentile",
    seed: int | None = None,
    tau: float | None = None,
    p: float | None = None,
    times: float | None = None,
) -> BootstrapResult:
    """Bootstrap confidence interval for a Kaplan-Meier summary statistic.

    Resamples subjects with replacement, fits a Kaplan-Meier estimator on each resample, and
    extracts the statistic of interest. The bootstrap distribution is then used to construct a
    confidence interval.

    Parameters
    ----------
    surv
        A `Surv` response (right-censored or counting-process).
    statistic
        The quantity to bootstrap. Pass a string for built-in statistics:

        - `"median"`: median survival time.
        - `"rmst"`: restricted mean survival time (requires `tau=`).
        - `"quantile"`: survival quantile (requires `p=`).
        - `"survival"`: survival probability at a fixed time (requires `times=`).
        - `"median_diff"`: difference in median survival between two groups (requires `by=`).
        - `"rmst_diff"`: difference in RMST between two groups (requires `by=` and `tau=`).
        - `"survival_diff"`: difference in survival probability at a fixed time between two groups
        (requires `by=` and `times=`).

        Alternatively, pass a callable that takes a fitted `KaplanMeier` and returns a float.
    by
        Grouping variable for two-sample comparisons. Required for `"_diff"` statistics. Must define
        exactly two groups.
    weights
        Optional case weights passed through to each bootstrap KM fit.
    n_boot
        Number of bootstrap replicates (default 1000).
    conf_level
        Confidence level (default 0.95).
    ci_type
        Confidence interval type: `"percentile"` (default), `"normal"`, or `"bca"` (bias-corrected
        and accelerated).
    seed
        Random seed for reproducibility.
    tau
        Upper time limit for RMST statistics.
    p
        Quantile level for `"quantile"` statistic.
    times
        Time point for `"survival"` and `"survival_diff"` statistics.

    Returns
    -------
    BootstrapResult
        Point estimate, standard error, confidence interval, and the full bootstrap distribution.
    """
    from ._nonparametric import KaplanMeier as _KM
    from ._resample import _subset_surv
    from ._surv import _to_1d_array

    if ci_type not in _VALID_CI_TYPES:
        raise ValueError(f"ci_type must be one of {sorted(_VALID_CI_TYPES)}, got {ci_type!r}.")
    if not 0.0 < conf_level < 1.0:
        raise ValueError(f"conf_level must be in (0, 1), got {conf_level}.")
    if n_boot < 1:
        raise ValueError(f"n_boot must be >= 1, got {n_boot}.")

    is_grouped = by is not None
    is_diff = False

    if isinstance(statistic, str):
        if statistic not in _VALID_STATISTICS:
            raise ValueError(
                f"Unknown statistic {statistic!r}. Choose from: {sorted(_VALID_STATISTICS)}."
            )
        is_diff = statistic in _DIFF_STATISTICS
        stat_fn = _resolve_statistic(statistic, tau=tau, p=p, times=times, is_grouped=is_grouped)
    else:
        stat_fn = statistic

    n = surv.n
    rng = np.random.default_rng(seed)

    weights_arr: Array | None = None
    if weights is not None:
        weights_arr = _to_1d_array(weights)

    by_arr: Array | None = None
    if by is not None:
        by_arr = _to_1d_array(by, dtype=object)

    km_orig = _KM()
    if is_grouped:
        km_orig.fit(surv, by=by, weights=weights)
    else:
        km_orig.fit(surv, weights=weights)
    estimate = float(stat_fn(km_orig))

    theta_boot = np.empty(n_boot)

    if is_diff and by_arr is not None:
        levels = sorted(dict.fromkeys(by_arr.tolist()).keys(), key=str)
        if len(levels) != 2:
            raise ValueError(
                f"Diff statistics require exactly 2 groups, got {len(levels)}: {levels}."
            )
        idx_g1 = np.where(by_arr == levels[0])[0]
        idx_g2 = np.where(by_arr == levels[1])[0]

        for b in range(n_boot):
            boot_i1 = rng.choice(idx_g1, size=len(idx_g1), replace=True)
            boot_i2 = rng.choice(idx_g2, size=len(idx_g2), replace=True)
            boot_idx = np.concatenate([boot_i1, boot_i2])
            boot_surv = _subset_surv(surv, boot_idx)
            boot_by = np.array([levels[0]] * len(idx_g1) + [levels[1]] * len(idx_g2), dtype=object)
            boot_w = weights_arr[boot_idx] if weights_arr is not None else None
            try:
                km_b = _KM().fit(boot_surv, by=boot_by, weights=boot_w)
                theta_boot[b] = stat_fn(km_b)
            except Exception:
                theta_boot[b] = float("nan")
    else:
        for b in range(n_boot):
            boot_idx = rng.choice(n, size=n, replace=True)
            boot_surv = _subset_surv(surv, boot_idx)
            boot_w = weights_arr[boot_idx] if weights_arr is not None else None
            try:
                km_b = _KM()
                if is_grouped and by_arr is not None:
                    km_b.fit(boot_surv, by=by_arr[boot_idx], weights=boot_w)
                else:
                    km_b.fit(boot_surv, weights=boot_w)
                theta_boot[b] = stat_fn(km_b)
            except Exception:
                theta_boot[b] = float("nan")

    valid_mask = np.isfinite(theta_boot)
    se = float(np.nanstd(theta_boot, ddof=1)) if valid_mask.sum() > 1 else float("nan")

    if ci_type == "percentile":
        conf_low, conf_high = _percentile_ci(theta_boot, conf_level)
    elif ci_type == "normal":
        conf_low, conf_high = _normal_ci(estimate, se, conf_level)
    elif ci_type == "bca":
        jk_values = _jackknife_values(surv, stat_fn, by=by, weights=weights, is_diff=is_diff)
        conf_low, conf_high = _bca_ci(theta_boot, estimate, jk_values, conf_level)
    else:
        raise ValueError(f"ci_type must be one of {sorted(_VALID_CI_TYPES)}, got {ci_type!r}.")

    return BootstrapResult(
        estimate=estimate,
        se=se,
        conf_low=conf_low,
        conf_high=conf_high,
        conf_level=conf_level,
        ci_type=ci_type,
        n_boot=n_boot,
        distribution=theta_boot,
    )


def _jackknife_values(
    surv: Surv,
    stat_fn: Callable[[KaplanMeier], float],
    *,
    by: Any,
    weights: Any,
    is_diff: bool,
) -> Array:
    """Leave-one-out jackknife values for BCa acceleration constant."""
    from ._nonparametric import KaplanMeier as _KM
    from ._resample import _subset_surv
    from ._surv import _to_1d_array

    n = surv.n
    jk = np.empty(n)
    by_arr = _to_1d_array(by, dtype=object) if by is not None else None
    weights_arr = _to_1d_array(weights) if weights is not None else None

    all_idx = np.arange(n)
    for i in range(n):
        loo_idx = np.delete(all_idx, i)
        loo_surv = _subset_surv(surv, loo_idx)
        loo_w = weights_arr[loo_idx] if weights_arr is not None else None
        try:
            km_i = _KM()
            if by_arr is not None:
                km_i.fit(loo_surv, by=by_arr[loo_idx], weights=loo_w)
            else:
                km_i.fit(loo_surv, weights=loo_w)
            jk[i] = stat_fn(km_i)
        except Exception:
            jk[i] = float("nan")
    return jk
