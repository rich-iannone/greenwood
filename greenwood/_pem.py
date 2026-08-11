"""Piecewise exponential survival model.

`PiecewiseExponential` fits a survival model where the baseline hazard is constant within each time
interval but changes across intervals. Covariates shift the hazard proportionally (like Cox), but
the baseline is fully specified rather than left nonparametric. The model is fit via Poisson maximum
likelihood on the interval-expanded dataset, which is equivalent to the piecewise exponential
likelihood.

Knot placement can be manual (`breaks=`) or automatic via AIC minimization (`knot_strategy="aic"`).
Results are validated for consistency against the Cox model's partial likelihood estimates and
parametric exponential/Weibull fits.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt
from scipy.stats import norm

from ._backends import to_dataframe
from ._cox import _design_matrix

if TYPE_CHECKING:
    from ._surv import Surv

__all__ = ["PiecewiseExponential"]

Array = npt.NDArray[Any]


def _expand_data(
    entry: Array,
    exit_: Array,
    event: Array,
    weight: Array,
    x: Array,
    breaks: Array,
) -> tuple[Array, Array, Array, Array, Array]:
    """Expand subject-level data into interval rows for the Poisson GLM.

    Each subject's follow-up `(entry, exit]` is split at the breakpoints. The result has one row per
    subject-interval with columns for exposure time (`log_offset`), event indicator, interval index,
    weight, and covariates.

    Returns (y, log_offset, interval_idx, weights, x_expanded).
    """
    n = len(exit_)
    n_intervals = len(breaks) + 1
    boundaries = np.concatenate([[0.0], breaks, [np.inf]])

    rows_y: list[float] = []
    rows_offset: list[float] = []
    rows_interval: list[int] = []
    rows_weight: list[float] = []
    rows_x_idx: list[int] = []

    for i in range(n):
        e_in = entry[i] if np.isfinite(entry[i]) else 0.0
        e_out = exit_[i]
        ev = event[i]
        w = weight[i]

        for j in range(n_intervals):
            lo, hi = boundaries[j], boundaries[j + 1]
            start = max(e_in, lo)
            stop = min(e_out, hi)
            if stop <= start:
                continue
            exposure = stop - start
            is_event = 1.0 if (ev and stop == e_out and e_out <= hi) else 0.0
            rows_y.append(is_event)
            rows_offset.append(np.log(exposure))
            rows_interval.append(j)
            rows_weight.append(w)
            rows_x_idx.append(i)

    y = np.array(rows_y)
    log_offset = np.array(rows_offset)
    interval_idx = np.array(rows_interval, dtype=int)
    weights = np.array(rows_weight)
    x_expanded = x[np.array(rows_x_idx, dtype=int)]

    return y, log_offset, interval_idx, weights, x_expanded


def _poisson_fit(
    y: Array, X: Array, log_offset: Array, weights: Array, max_iter: int, tol: float
) -> tuple[Array, float, Array]:
    """Fit Poisson GLM via iteratively reweighted least squares (IRLS).

    Returns (coefficients, log_likelihood, variance-covariance matrix).
    """
    p = X.shape[1]
    beta = np.zeros(p)

    for _ in range(max_iter):
        eta = X @ beta + log_offset
        mu = np.exp(eta)
        wmu = weights * mu

        ll = float(np.sum(weights * (y * eta - mu)))

        grad = X.T @ (weights * (y - mu))
        info = (X * wmu[:, None]).T @ X

        try:
            step = np.linalg.solve(info, grad)
        except np.linalg.LinAlgError:
            break

        beta_new = beta + step
        if np.max(np.abs(step)) < tol:
            beta = beta_new
            break
        beta = beta_new

    eta = X @ beta + log_offset
    mu = np.exp(eta)
    ll = float(np.sum(weights * (y * eta - mu)))
    wmu_final = weights * mu
    info = (X * wmu_final[:, None]).T @ X
    vcov = np.linalg.inv(info)
    return beta, ll, vcov


def _select_knots(
    entry: Array,
    exit_: Array,
    event: Array,
    weight: Array,
    x: Array,
    cov_names: list[str],
    max_knots: int,
    strategy: str,
    conf_level: float,
    max_iter: int,
    tol: float,
) -> Array:
    """Select optimal knot positions by minimizing AIC or BIC."""
    event_times = np.sort(np.unique(exit_[event.astype(bool)]))
    if len(event_times) < 3:
        return np.array([])

    best_ic = np.inf
    best_breaks: Array = np.array([])

    for n_knots in range(0, max_knots + 1):
        if n_knots == 0:
            breaks = np.array([])
        else:
            quantiles = np.linspace(0, 1, n_knots + 2)[1:-1]
            breaks = np.quantile(event_times, quantiles)
            breaks = np.unique(breaks)

        n_intervals = len(breaks) + 1
        y, log_offset, interval_idx, weights, x_exp = _expand_data(
            entry, exit_, event, weight, x, breaks
        )
        interval_dummies = np.zeros((len(interval_idx), n_intervals))
        interval_dummies[np.arange(len(interval_idx)), interval_idx] = 1.0
        X = np.column_stack([interval_dummies, x_exp])

        beta, ll, _ = _poisson_fit(y, X, log_offset, weights, max_iter, tol)
        n_params = len(beta)

        if strategy == "aic":
            ic = -2.0 * ll + 2.0 * n_params
        else:
            n_obs = float(np.sum(weights))
            ic = -2.0 * ll + np.log(n_obs) * n_params

        if ic < best_ic:
            best_ic = ic
            best_breaks = breaks

    return best_breaks


class PiecewiseExponential:
    """Piecewise exponential survival model.

    Assumes the hazard is constant within each time interval but can differ between intervals.
    Covariates enter multiplicatively (proportional hazards within each interval). The model is fit
    by maximum likelihood, equivalent to a Poisson GLM on the interval-expanded dataset with
    `log(exposure)` as offset.

    This sits between the fully nonparametric Cox model and the fully parametric AFT: it estimates
    the baseline hazard but restricts it to a step function. Fewer intervals give a smoother hazard.
    More intervals approach the Cox model's flexibility.

    Parameters
    ----------
    breaks
        Time points at which the hazard is allowed to change. For example, `breaks=[180, 365]`
        creates three intervals: `(0, 180]`, `(180, 365]`, and `(365, inf]`. When `None` (default),
        knots are chosen automatically by minimizing AIC.
    knot_strategy
        Strategy for automatic knot selection when `breaks` is `None`. `"aic"` (default) minimizes
        the Akaike information criterion. `"bic"` minimizes the Bayesian information criterion.
        Ignored when `breaks` is provided.
    max_knots
        Maximum number of interior knots to consider during automatic selection (the default is
        `10`). Ignored when `breaks` is provided.
    conf_level
        Confidence level for coefficient intervals (the default is `0.95`).

    Examples
    --------
    ```{python}
    import greenwood as gw

    lung = gw.load_dataset("lung", backend="polars")
    y = gw.Surv.right(lung["time"], event=(lung["status"] == 2))

    pem = gw.PiecewiseExponential().fit(y, lung[["age", "sex"]])
    pem
    ```

    With manual break points:

    ```{python}
    pem_manual = gw.PiecewiseExponential(breaks=[180, 365]).fit(y, lung[["age", "sex"]])
    pem_manual.to_frame(format="polars")
    ```
    """

    def __init__(
        self,
        *,
        breaks: list[float] | tuple[float, ...] | None = None,
        knot_strategy: str = "aic",
        max_knots: int = 10,
        conf_level: float = 0.95,
    ) -> None:
        if knot_strategy not in ("aic", "bic"):
            raise ValueError(f"knot_strategy must be 'aic' or 'bic', got {knot_strategy!r}")
        self.breaks_input = breaks
        self.knot_strategy = knot_strategy
        self.max_knots = max_knots
        self.conf_level = conf_level

    def __repr__(self) -> str:
        if getattr(self, "coef_", None) is None:
            return (
                f"PiecewiseExponential(breaks={self.breaks_input!r}, "
                f"conf_level={self.conf_level}) <unfitted>"
            )
        from ._repr import align_table, fixed, num

        rows = [
            [num(c), num(se), fixed(z, 3), num(p)]
            for c, se, z, p in zip(self.coef_, self.std_error_, self.z_, self.p_value_, strict=True)
        ]
        table = align_table(["coef", "se(coef)", "z", "p"], rows, list(self.term_names_))
        intervals = [
            f"  ({self._boundaries[j]:.0f}, {self._boundaries[j + 1]:.0f}]"
            if np.isfinite(self._boundaries[j + 1])
            else f"  ({self._boundaries[j]:.0f}, inf]"
            for j in range(self._n_intervals)
        ]
        return "\n".join(
            [
                f"PiecewiseExponential ({self._n_intervals} intervals)",
                "",
                table,
                "",
                "Intervals:",
                *intervals,
                "",
                f"n = {self.n_}, events = {self.n_event_}",
                f"Log-likelihood = {num(self.loglik_)}",
                f"AIC = {num(self.aic_)}",
            ]
        )

    def fit(
        self,
        surv: Surv,
        covariates: Any,
        *,
        data: Any = None,
        max_iter: int = 50,
        tol: float = 1e-9,
    ) -> PiecewiseExponential:
        """Fit the piecewise exponential model.

        Parameters
        ----------
        surv
            A right-censored or counting-process `Surv` response.
        covariates
            A dataframe (pandas or polars), a 2-D array, or a formula string.
        data
            A dataframe for formula evaluation (ignored otherwise).
        max_iter
            Maximum IRLS iterations (default 50).
        tol
            Convergence tolerance on the step size (default 1e-9).

        Returns
        -------
        PiecewiseExponential
            The fitted model (for method chaining).

        Examples
        --------
        ```{python}
        import greenwood as gw

        lung = gw.load_dataset("lung", backend="polars")
        y = gw.Surv.right(lung["time"], event=(lung["status"] == 2))
        pem = gw.PiecewiseExponential(breaks=[180, 365]).fit(y, lung[["age", "sex"]])
        pem.to_frame(format="polars")
        ```
        """
        from ._surv import CensoringType

        if surv.type not in (CensoringType.RIGHT, CensoringType.COUNTING):
            raise NotImplementedError(
                f"PiecewiseExponential supports right-censored or counting-process data, "
                f"not {surv.type.value!r}."
            )

        design, cov_names = _design_matrix(covariates, data)
        if design.shape[0] != surv.n:
            raise ValueError("Covariates and response must have the same number of rows.")

        exit_ = surv.stop
        entry = surv.entry
        event = surv.event.astype(float)
        weight = surv.weights if surv.weights is not None else np.ones(surv.n)

        keep = ~np.isnan(design).any(axis=1) & (exit_ > 0)
        if np.isfinite(entry).all():
            keep &= exit_ > entry
        design = design[keep]
        exit_ = exit_[keep]
        entry_vals = entry[keep]
        event = event[keep]
        weight = weight[keep]

        if self.breaks_input is not None:
            breaks = np.sort(np.unique(np.array(self.breaks_input, dtype=float)))
        else:
            breaks = _select_knots(
                entry_vals,
                exit_,
                event,
                weight,
                design,
                cov_names,
                self.max_knots,
                self.knot_strategy,
                self.conf_level,
                max_iter,
                tol,
            )

        n_intervals = len(breaks) + 1
        y, log_offset, interval_idx, weights, x_exp = _expand_data(
            entry_vals,
            exit_,
            event,
            weight,
            design,
            breaks,
        )

        interval_dummies = np.zeros((len(interval_idx), n_intervals))
        interval_dummies[np.arange(len(interval_idx)), interval_idx] = 1.0
        X = np.column_stack([interval_dummies, x_exp])

        beta, ll, vcov = _poisson_fit(y, X, log_offset, weights, max_iter, tol)

        self.breaks_ = breaks
        self._boundaries = np.concatenate([[0.0], breaks, [np.inf]])
        self._n_intervals = n_intervals

        self._baseline_log_hazard = beta[:n_intervals]
        self._baseline_vcov = vcov[:n_intervals, :n_intervals]

        n_cov = design.shape[1]
        if n_cov > 0:
            cov_beta = beta[n_intervals:]
            cov_vcov = vcov[n_intervals:, n_intervals:]
            cov_se = np.sqrt(np.diag(cov_vcov))
        else:
            cov_beta = np.array([])
            cov_vcov = np.zeros((0, 0))
            cov_se = np.array([])

        self.term_names_ = list(cov_names)
        self.coef_ = cov_beta
        self.vcov_ = cov_vcov
        self.std_error_ = cov_se
        self.z_ = cov_beta / cov_se if len(cov_se) > 0 else np.array([])
        self.p_value_ = 2.0 * norm.sf(np.abs(self.z_)) if len(self.z_) > 0 else np.array([])
        self.hazard_ratio_ = np.exp(cov_beta)
        self.loglik_ = ll
        self.n_ = int(keep.sum())
        self.n_event_ = int(event.sum())
        self.df_ = len(beta)
        self.aic_ = -2.0 * ll + 2.0 * self.df_
        self.bic_ = -2.0 * ll + np.log(float(self.n_)) * self.df_

        z_val = float(norm.ppf(1.0 - (1.0 - self.conf_level) / 2.0))
        half = z_val * cov_se
        self.conf_low_ = cov_beta - half
        self.conf_high_ = cov_beta + half

        self._full_beta = beta
        self._full_vcov = vcov
        self._x = design
        self._exit = exit_
        self._entry = entry_vals
        self._event = event
        self._weight = weight

        lr_null = self._null_loglik(entry_vals, exit_, event, weight, breaks, max_iter, tol)
        self.lr_stat_ = 2.0 * (ll - lr_null)
        self.loglik_null_ = lr_null

        return self

    @staticmethod
    def _null_loglik(
        entry: Array,
        exit_: Array,
        event: Array,
        weight: Array,
        breaks: Array,
        max_iter: int,
        tol: float,
    ) -> float:
        """Log-likelihood of the null model (no covariates, only interval dummies)."""
        n_intervals = len(breaks) + 1
        y, log_offset, interval_idx, weights, _ = _expand_data(
            entry,
            exit_,
            event,
            weight,
            np.zeros((len(exit_), 0)),
            breaks,
        )
        interval_dummies = np.zeros((len(interval_idx), n_intervals))
        interval_dummies[np.arange(len(interval_idx)), interval_idx] = 1.0
        _, ll, _ = _poisson_fit(y, interval_dummies, log_offset, weights, max_iter, tol)
        return ll

    def baseline_hazard(self, *, format: str | None = None) -> Any:
        """Return the piecewise-constant baseline hazard as a DataFrame.

        Each row gives the interval boundaries and the constant hazard rate within that interval.

        Parameters
        ----------
        format
            Output format: `None`, `"pandas"`, `"polars"`, or `"pyarrow"`.

        Returns
        -------
        pandas.DataFrame, polars.DataFrame, or pyarrow.Table

        Examples
        --------
        ```{python}
        import greenwood as gw

        lung = gw.load_dataset("lung", backend="polars")
        y = gw.Surv.right(lung["time"], event=(lung["status"] == 2))
        pem = gw.PiecewiseExponential(breaks=[180, 365]).fit(y, lung[["age", "sex"]])
        pem.baseline_hazard(format="polars")
        ```
        """
        hazards = np.exp(self._baseline_log_hazard)
        starts = self._boundaries[:-1].tolist()
        stops = self._boundaries[1:].tolist()
        return to_dataframe(
            {
                "start": starts,
                "stop": stops,
                "hazard": hazards.tolist(),
                "log_hazard": self._baseline_log_hazard.tolist(),
            },
            format=format,
        )

    def predict(
        self,
        newdata: Any = None,
        *,
        type: str = "survival",
        times: list[float] | Array | None = None,
        format: str | None = None,
    ) -> Any:
        """Predict survival or cumulative hazard for new subjects.

        Parameters
        ----------
        newdata
            Covariate values for new subjects. If `None`, uses the training data.
        type
            `"survival"` (default), `"cumhaz"` (cumulative hazard), `"lp"` (linear predictor), or
            `"risk"` (exp of linear predictor).
        times
            Times at which to evaluate survival or cumulative hazard. Required for `"survival"` and
            `"cumhaz"`.
        format
            Output format for tabular results.

        Returns
        -------
        Array or DataFrame
            For `"lp"` and `"risk"`, a 1-D array. For `"survival"` and `"cumhaz"`, a DataFrame with
            one column per subject and one row per time.
        """
        if newdata is None:
            x = self._x
        else:
            x, _ = _design_matrix(newdata)

        lp = x @ self.coef_ if len(self.coef_) > 0 else np.zeros(x.shape[0])

        if type == "lp":
            return lp
        if type == "risk":
            return np.exp(lp)

        if times is None:
            raise ValueError("times= is required for type='survival' or type='cumhaz'")
        t_arr = np.asarray(times, dtype=float)

        baseline_hazard = np.exp(self._baseline_log_hazard)
        boundaries = self._boundaries

        cumhaz_base = np.zeros(len(t_arr))
        for k in range(len(t_arr)):
            t = t_arr[k]
            ch = 0.0
            for j in range(self._n_intervals):
                lo, hi = boundaries[j], boundaries[j + 1]
                if t <= lo:
                    break
                width = min(t, hi) - lo
                ch += baseline_hazard[j] * width
            cumhaz_base[k] = ch

        n_subjects = x.shape[0]
        cols: dict[str, Any] = {"time": t_arr.tolist()}
        for i in range(n_subjects):
            risk_i = float(np.exp(lp[i]))
            cumhaz_i = cumhaz_base * risk_i
            if type == "survival":
                cols[f"subject_{i + 1}"] = np.exp(-cumhaz_i).tolist()
            else:
                cols[f"subject_{i + 1}"] = cumhaz_i.tolist()

        return to_dataframe(cols, format=format)

    def _coefficient_columns(self) -> dict[str, Any]:
        return {
            "term": self.term_names_,
            "estimate": self.coef_.tolist(),
            "std_error": self.std_error_.tolist(),
            "statistic": self.z_.tolist(),
            "p_value": self.p_value_.tolist(),
            "conf_low": self.conf_low_.tolist(),
            "conf_high": self.conf_high_.tolist(),
        }

    def to_frame(self, *, format: str | None = None) -> Any:
        """Return the coefficient table as a DataFrame.

        One row per covariate (excluding interval parameters). Includes coefficient estimates,
        standard errors, Wald statistics, p-values, and confidence limits.

        Parameters
        ----------
        format
            Output format: `None`, `"pandas"`, `"polars"`, or `"pyarrow"`.

        Returns
        -------
        pandas.DataFrame, polars.DataFrame, or pyarrow.Table

        Examples
        --------
        ```{python}
        import greenwood as gw

        lung = gw.load_dataset("lung", backend="polars")
        y = gw.Surv.right(lung["time"], event=(lung["status"] == 2))
        pem = gw.PiecewiseExponential(breaks=[180, 365]).fit(y, lung[["age", "sex"]])
        pem.to_frame(format="polars")
        ```
        """
        return to_dataframe(self._coefficient_columns(), format=format)


def _tidy_pem(model: PiecewiseExponential, *, format: str | None = None, **_: Any) -> Any:
    return model.to_frame(format=format)


def _glance_pem(model: PiecewiseExponential, *, format: str | None = None, **_: Any) -> Any:
    return to_dataframe(
        {
            "n_intervals": [model._n_intervals],
            "n": [model.n_],
            "nevent": [model.n_event_],
            "loglik": [model.loglik_],
            "aic": [model.aic_],
            "bic": [model.bic_],
            "df": [model.df_],
            "lr_statistic": [model.lr_stat_],
        },
        format=format,
    )


def _register_adapters() -> None:
    from .summaries import register_glance, register_tidier

    register_tidier("greenwood._pem.PiecewiseExponential", _tidy_pem)
    register_glance("greenwood._pem.PiecewiseExponential", _glance_pem)


_register_adapters()
