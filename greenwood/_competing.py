"""Competing-risks estimation: the Aalen-Johansen cumulative incidence function.

For competing risks (each subject starts in one state and makes a single transition to one
of several absorbing causes), the cumulative incidence function (CIF) for cause `k` is

    CIF_k(t) = sum_{t_i <= t} S(t_i^-) * d_{ki} / n_i,

where `S` is the all-cause Kaplan-Meier survival, `d_{ki}` the cause-`k` events at `t_i`, and
`n_i` the number at risk. The standard error uses the Aalen (Marubini-Valsecchi)
delta-method estimator. Both are validated to tolerance against R's `survfit`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt
from scipy.stats import norm

if TYPE_CHECKING:
    from ._surv import Surv

__all__ = ["AalenJohansen", "FineGray", "MultiState"]

Array = npt.NDArray[Any]


def _censoring_km(time: Array, cause: Array) -> tuple[Array, Array]:
    """Nudged censoring Kaplan-Meier: (drop times, survival after each drop).

    Events (any cause) are treated as leaving just before a tied censoring, matching R's
    `finegray`. Returns the censoring times where the curve drops and the survival value
    just after each drop.
    """
    censor_times = np.unique(time[cause == 0])
    surv = 1.0
    drop_times: list[float] = []
    drop_surv: list[float] = []
    for c in censor_times:
        # At-risk for censoring excludes events tied at c (they are nudged just before).
        n_risk = float((time > c).sum() + ((cause == 0) & (time == c)).sum())
        d = float(((cause == 0) & (time == c)).sum())
        surv *= 1.0 - d / n_risk
        drop_times.append(float(c))
        drop_surv.append(surv)
    return np.array(drop_times), np.array(drop_surv)


def _cif_block(
    exit_: Array, status: Array, causes: list[int], z: float
) -> dict[int, dict[str, Array]]:
    """Cumulative incidence, delta-method SE, and CI for each cause in one group."""
    times = np.unique(exit_)
    n_risk = np.array([float((exit_ >= t).sum()) for t in times])
    d_any = np.array([float(((exit_ == t) & (status > 0)).sum()) for t in times])

    surv = np.cumprod(1.0 - d_any / n_risk)
    surv_left = np.concatenate(([1.0], surv[:-1]))

    out: dict[int, dict[str, Array]] = {}
    for cause in causes:
        d_k = np.array([float(((exit_ == t) & (status == cause)).sum()) for t in times])
        cif = np.cumsum(surv_left * d_k / n_risk)

        # Aalen (Marubini-Valsecchi) delta-method variance via cumulative sums.
        with np.errstate(divide="ignore", invalid="ignore"):
            a = np.where(n_risk > d_any, d_any / (n_risk * (n_risk - d_any)), 0.0)
        b = surv_left**2 * (n_risk - d_k) / n_risk * d_k / n_risk**2
        c = surv_left * d_k / n_risk**2
        c_a = np.cumsum(a)
        c_ac = np.cumsum(a * cif)
        c_ac2 = np.cumsum(a * cif**2)
        c_b = np.cumsum(b)
        c_c = np.cumsum(c)
        c_cc = np.cumsum(c * cif)
        var = cif**2 * c_a - 2 * cif * c_ac + c_ac2 + c_b - 2 * cif * c_c + 2 * c_cc
        se = np.sqrt(np.clip(var, 0.0, None))

        out[cause] = {
            "time": times,
            "n_risk": n_risk,
            "estimate": cif,
            "std_error": se,
            "conf_low": np.clip(cif - z * se, 0.0, 1.0),
            "conf_high": np.clip(cif + z * se, 0.0, 1.0),
        }
    return out


class AalenJohansen:
    """Aalen-Johansen estimator of cumulative incidence functions for competing risks.

    In survival analysis, competing risks occur when subjects can experience multiple types
    of events (e.g., progression to malignancy vs. death from other causes), and experiencing
    one event precludes the others. The Aalen-Johansen estimator extends the Kaplan-Meier
    approach to this setting by estimating the cumulative incidence function (CIF) for each
    cause: the probability of experiencing that specific cause by time t, accounting for
    competition from other causes.

    Unlike naive estimates that ignore censoring or competing events, the Aalen-Johansen CIF
    correctly accounts for both. It is computed using transition probabilities between states
    via generalized Kaplan-Meier estimates. Call `fit()` with a multi-state `Surv` response
    (built with `Surv.multistate()`) to obtain estimates for each competing cause. Results
    are returned as tidy DataFrames with one row per combination of stratum, cause, and time.

    The estimator uses the formula: CIF_j(t) = Σ_{s≤t} S(s-) P_{0j}(s), where S(s-) is the
    probability of being event-free before time s, and P_{0j}(s) is the transition probability
    from censoring to cause j. Confidence intervals use the Greenwood-style variance estimator
    on the complementary log-log scale for improved coverage.

    Parameters
    ----------
    conf_level
        Confidence level for the (Wald) confidence intervals (default 0.95).

    Returns
    -------
    Fitted estimator
        Call `fit()` to produce a fitted estimator with cached results (`states_`, and
        internal transition matrices), accessible as tidy DataFrames.

    Notes
    -----
    Call `fit(surv, by=...)` with a multi-state `Surv` response (built with
    `Surv.multistate`, where `event` codes are 0 for censoring and `1..K` for the competing
    causes). Results are tidy frames via `to_pandas()`, `to_polars()`, `to_arrow()` with one row
    per stratum, cause, and time.

    Examples
    --------
    The bundled `mgus2` dataset follows monoclonal-gammopathy patients who may progress to
    plasma-cell malignancy (`"pcm"`) or die first, a competing-risks setup. Build the
    competing-risks response by combining the progression and death indicators into a single
    cause code (0 censored, 1 progression, 2 death), then fit the estimator. Printing the
    fitted object reports the final cumulative incidence for each cause.

    ```{python}
    import numpy as np
    import greenwood as gw

    mg = gw.load_dataset("mgus2")
    etime = np.where(mg["pstat"] == 1, mg["ptime"], mg["futime"])
    cause = np.where(mg["pstat"] == 1, 1, 2 * mg["death"])
    y = gw.Surv.multistate(etime, event=cause, states=("pcm", "death"))
    aj = gw.AalenJohansen().fit(y)
    aj
    ```

    The `aj` object fit here, along with the `y` response, is reused by the method examples
    below.
    """

    def __init__(self, *, conf_level: float = 0.95) -> None:
        if not 0.0 < conf_level < 1.0:
            raise ValueError(f"conf_level must be in (0, 1), got {conf_level}.")
        self.conf_level = conf_level

    def __repr__(self) -> str:
        if getattr(self, "states_", None) is None:
            return "AalenJohansen() <unfitted>"
        states = ", ".join(str(s) for s in self.states_)
        head = [
            "AalenJohansen (Aalen-Johansen cumulative incidence)",
            "",
            f"states: {states}",
        ]
        df = self.to_pandas()
        if self._grouped:
            n_strata = df["strata"].nunique()
            head.append(f"strata: {n_strata}")
            return "\n".join(head)
        from ._repr import align_table, num

        head.append(f"n = {int(df['n_risk'].iloc[0])}")
        labels, rows = [], []
        for cause, g in df.groupby("cause", sort=False):
            labels.append(str(cause))
            rows.append([num(g["estimate"].iloc[-1])])
        table = align_table(["final CIF"], rows, labels)
        return "\n".join(head) + "\n\n" + table

    def fit(self, surv: Surv, *, by: Any = None) -> AalenJohansen:
        """Fit cumulative incidence functions to a competing-risks response.

        Computes the cumulative incidence function (CIF) for each cause-of-interest from a
        multi-state `Surv` response. Unlike Kaplan-Meier (which handles only a single event
        type), the Aalen-Johansen estimator accounts for competing events: subjects who
        experience a competing cause are removed from the risk set, preventing overly optimistic
        estimates of the probability of experiencing the target cause. Results are stored in
        the fitted object; access them via `to_pandas()`, `to_polars()`, or `to_arrow()`.

        The Aalen-Johansen estimator generalizes both Kaplan-Meier and Nelson-Aalen to the
        competing-risks setting. For each cause j, it estimates F_j(t), the cumulative
        probability of experiencing cause j by time t, accounting for all competing causes. The
        CIFs sum to the overall event probability at any time. Pass `by=` to produce separate
        CIF estimates per group (stratified competing-risks analysis).

        Parameters
        ----------
        surv
            A multi-state `Surv` response built with `Surv.multistate()`. Must have multiple
            causes-of-interest. Raises `ValueError` if a single-event response is passed
            (use `KaplanMeier` for that).
        by
            Optional grouping variable (e.g., a column or array). Produces one set of
            cumulative incidence functions per unique value of `by`. Default (`None`): fit
            a single, unstratified set of CIFs.

        Returns
        -------
        AalenJohansen
            The fitted estimator object itself (for method chaining) with cached cumulative
            incidence results (time arrays, CIF per cause, confidence bands) accessible via
            `to_pandas()` and related methods.

        Notes
        -----
        The Aalen-Johansen estimator is a product-integral estimator of the CIF:
        F_j(t) = integral S_{-}(u) dM_j(u), where S_{-}(u) is the estimated probability of
        surviving (remaining uncensored) just before u, and M_j(u) is the counting process for
        cause j. It reduces to Kaplan-Meier when there is only one cause and no censoring.

        Left truncation is not yet supported. Multi-state responses must be built with
        `Surv.multistate()`.

        Examples
        --------
        Fit cumulative incidence functions on the competing-risks `mgus2` dataset, where
        subjects can experience plasma-cell malignancy (pcm) or death:

        ```{python}
        import numpy as np
        import greenwood as gw

        mg = gw.load_dataset("mgus2")
        etime = np.where(mg["pstat"] == 1, mg["ptime"], mg["futime"])
        cause = np.where(mg["pstat"] == 1, 1, 2 * mg["death"])
        y = gw.Surv.multistate(etime, event=cause, states=("pcm", "death"))
        aj = gw.AalenJohansen().fit(y)
        aj
        ```

        Fit stratified cumulative incidence functions by sex to compare risk accumulation:

        ```{python}
        aj_stratified = gw.AalenJohansen().fit(y, by=mg["sex"])
        aj_stratified
        ```
        """
        if not surv.is_multistate:
            raise ValueError(
                "AalenJohansen needs a multi-state response; build it with Surv.multistate "
                "(use KaplanMeier for a single event type)."
            )
        if surv.is_truncated:
            raise NotImplementedError(
                "Left truncation is not yet supported for cumulative incidence."
            )

        assert surv.states is not None
        self.states_ = surv.states
        causes = list(range(1, len(surv.states) + 1))
        z = float(norm.ppf(1.0 - (1.0 - self.conf_level) / 2.0))

        exit_ = surv.stop
        status = surv.status

        if by is None:
            self._grouped = False
            self._blocks = {None: _cif_block(exit_, status, causes, z)}
        else:
            from ._surv import _to_1d_array

            labels = _to_1d_array(by, dtype=object)
            if labels.shape[0] != surv.n:
                raise ValueError("`by` must have the same length as the response.")
            self._grouped = True
            self._blocks = {}
            for level in dict.fromkeys(labels.tolist()):
                mask = labels == level
                self._blocks[level] = _cif_block(exit_[mask], status[mask], causes, z)
        self._causes = causes
        return self

    def _table_columns(self) -> dict[str, list[Any]]:
        cols: dict[str, list[Any]] = {
            k: []
            for k in (
                "strata",
                "cause",
                "time",
                "n_risk",
                "estimate",
                "std_error",
                "conf_low",
                "conf_high",
            )
        }
        for label, block in self._blocks.items():
            for cause in self._causes:
                data = block[cause]
                m = data["time"].shape[0]
                cols["strata"].extend([label] * m)
                cols["cause"].extend([self.states_[cause - 1]] * m)
                for key in ("time", "n_risk", "estimate", "std_error", "conf_low", "conf_high"):
                    cols[key].extend(data[key].tolist())
        if not self._grouped:
            cols.pop("strata")
        return cols

    def to_pandas(self) -> Any:
        """Return cumulative-incidence estimates as a pandas DataFrame.

        This method exports the Aalen-Johansen fit with one row per cause and time point,
        including the risk set, cumulative-incidence estimate, standard error, confidence
        limits, and optional strata labels.

        Returns
        -------
        pandas.DataFrame
            A tidy DataFrame with columns `cause`, `time`, `n_risk`, `estimate`,
            `std_error`, `conf_low`, `conf_high`, and optionally `strata`.

        Raises
        ------
        ImportError
            If pandas is not installed.

        Examples
        --------
        Export the fitted cumulative-incidence functions to pandas:

        ```{python}
        aj.to_pandas()
        ```
        """
        try:
            import pandas as pd
        except ImportError as e:
            raise ImportError(
                "pandas is required for to_pandas(). Install it with: pip install pandas"
            ) from e

        return pd.DataFrame(self._table_columns())

    def to_polars(self) -> Any:
        """Return cumulative-incidence estimates as a Polars DataFrame.

        This method exports the Aalen-Johansen fit with one row per cause and time point,
        including the risk set, cumulative-incidence estimate, standard error, confidence
        limits, and optional strata labels.

        Returns
        -------
        polars.DataFrame
            A tidy DataFrame with columns `cause`, `time`, `n_risk`, `estimate`,
            `std_error`, `conf_low`, `conf_high`, and optionally `strata`.

        Raises
        ------
        ImportError
            If polars is not installed.

        Examples
        --------
        Export the fitted cumulative-incidence functions to Polars:

        ```{python}
        aj.to_polars()
        ```
        """
        try:
            import polars as pl
        except ImportError as e:
            raise ImportError(
                "polars is required for to_polars(). Install it with: pip install polars"
            ) from e

        return pl.DataFrame(self._table_columns())

    def to_arrow(self) -> Any:
        """Return cumulative-incidence estimates as a PyArrow Table.

        This method exports the Aalen-Johansen fit to Arrow, preserving the same columns
        as the pandas and Polars exports for efficient interchange.

        Returns
        -------
        pyarrow.Table
            A table with columns `cause`, `time`, `n_risk`, `estimate`, `std_error`,
            `conf_low`, `conf_high`, and optionally `strata`.

        Raises
        ------
        ImportError
            If pyarrow is not installed.

        Examples
        --------
        Export the fitted cumulative-incidence functions to Arrow:

        ```{python}
        aj.to_arrow()
        ```
        """
        try:
            import pyarrow as pa
        except ImportError as e:
            raise ImportError(
                "pyarrow is required for to_arrow(). Install it with: pip install pyarrow"
            ) from e

        return pa.table(self._table_columns())


class FineGray:
    """Fine-Gray subdistribution hazard model for a competing-risks endpoint.

    The Fine-Gray model extends Cox regression to competing-risks settings where multiple
    event types (e.g., disease progression vs. death) can occur and only the first is observed.
    Rather than modeling the cause-specific hazard (as in standard Cox), it models the
    subdistribution hazard: the rate at which subjects experience the target cause as if
    competing events did not occur. This allows inference directly on the cumulative incidence
    function, making it ideal for policy-relevant questions like "what is the effect of
    treatment on my probability of experiencing event A?"

    Technically, the Fine-Gray model uses a weighted Cox-like approach: subjects who experience
    a competing event remain in the risk set but with decreasing inverse-probability-of-censoring
    weights, reflecting their reduced ability to contribute information about the target cause.
    Call `fit()` with a multi-state `Surv` response (built with `Surv.multistate()`) and specify
    the target cause of interest. Coefficients, hazard ratios, and standard errors are computed
    via weighted partial likelihood, with robust (clustered) standard errors accounting for the
    weighting scheme.

    The implementation automatically computes event weights and handles censoring. Standard errors
    use the Lin-Wei robust (sandwich) estimator, validated against R's survival package. Unlike
    the Aalen-Johansen non-parametric approach, Fine-Gray provides covariate-adjusted estimates
    and supports prediction of cumulative incidence at specified times.

    Parameters
    ----------
    cause
        The target cause-of-interest label from the multi-state `Surv` response.
    conf_level
        Confidence level for coefficient intervals (default 0.95).

    Returns
    -------
    Fitted estimator
        Call `fit()` to produce a fitted estimator with cached results (`coef_`,
        `hazard_ratio_`, `std_error_`, `z_`, `p_value_`, `conf_low_`, `conf_high_`),
        accessible as arrays or exported to DataFrames.

    Notes
    -----
    Call `fit(surv, covariates)` with a multi-state `Surv` response (built with
    `Surv.multistate()`) and specify the target `cause`. The model uses weighted Cox-like
    optimization with robust standard errors. Results can be tidy frames via `to_pandas()`,
    `to_polars()`, `to_arrow()`.

    Examples
    --------
    Using the same competing-risks setup as `AalenJohansen`, model the cumulative incidence of
    plasma-cell malignancy (`"pcm"`) as a function of age and sex. The `age` and `sex` columns
    of the `mgus2` frame form the covariate design. Printing the fitted object reports the
    subdistribution-hazard coefficient table.

    ```{python}
    import numpy as np
    import greenwood as gw

    mg = gw.load_dataset("mgus2")
    etime = np.where(mg["pstat"] == 1, mg["ptime"], mg["futime"])
    cause = np.where(mg["pstat"] == 1, 1, 2 * mg["death"])
    y = gw.Surv.multistate(etime, event=cause, states=("pcm", "death"))
    fg = gw.FineGray("pcm").fit(y, mg[["age", "sex"]])
    fg
    ```

    Passing `exponentiate=True` to `tidy` reports the subdistribution hazard ratios (with
    their confidence limits) instead of the log-scale coefficients. The `fg` object fit here
    is reused by the method examples below.

    ```{python}
    import greenwood as gw

    gw.tidy(fg, exponentiate=True)
    ```
    """

    def __init__(self, cause: Any, *, conf_level: float = 0.95) -> None:
        if not 0.0 < conf_level < 1.0:
            raise ValueError(f"conf_level must be in (0, 1), got {conf_level}.")
        self.cause = cause
        self.conf_level = conf_level

    def __repr__(self) -> str:
        if getattr(self, "coef_", None) is None:
            return f"FineGray(cause={self.cause!r}) <unfitted>"
        from ._repr import align_table, fixed, num

        rows = [
            [num(c), num(hr), num(se), fixed(z, 3), num(p)]
            for c, hr, se, z, p in zip(
                self.coef_, self.hazard_ratio_, self.std_error_, self.z_, self.p_value_, strict=True
            )
        ]
        table = align_table(
            ["coef", "exp(coef)", "se(coef)", "z", "p"], rows, list(self.term_names_)
        )
        return "\n".join(
            [
                f"FineGray (Fine-Gray subdistribution hazard model, cause={self.cause!r})",
                "",
                table,
                "",
                f"n = {self.n_}, events = {self.n_event_}",
                "Standard errors: robust (clustered)",
            ]
        )

    def fit(
        self, surv: Surv, covariates: Any, *, max_iter: int = 30, tol: float = 1e-9
    ) -> FineGray:
        """Fit the Fine-Gray subdistribution hazard model to competing-risks data.

        Fits a Cox-like regression model to the subdistribution hazard of a target cause in
        a competing-risks setting. Subjects who experience a competing event remain in the
        risk set with inverse-probability-of-censoring weights, allowing direct inference on
        cumulative incidence of the target cause. The model reports coefficients, hazard
        ratios, and clustered robust (Lin-Wei) standard errors.

        Parameters
        ----------
        surv
            A multi-state `Surv` response built with `Surv.multistate()`. Must have multiple
            causes-of-interest. Raises `ValueError` if a single-event response is passed.
        covariates
            A dataframe (pandas or polars) or 2-D array of covariates to adjust for in the
            subdistribution hazard. An intercept is added automatically. Must have the same
            number of rows as `surv`.
        max_iter
            Maximum number of Newton-Raphson iterations (default 30).
        tol
            Convergence tolerance for coefficient changes (default 1e-9).

        Returns
        -------
        FineGray
            The fitted estimator object itself (for method chaining) with cached coefficient
            arrays (`coef_`, `std_error_`, `hazard_ratio_`, `z_`, `p_value_`), event counts,
            and model fit statistics.

        Notes
        -----
        The Fine-Gray model estimates the subdistribution hazard h_j(t) using weighted partial
        likelihood. Weights are the inverse of the Kaplan-Meier estimate of the censoring
        distribution, computed just before each target event time. Subjects with competing
        events receive time-decreasing weights reflecting reduced ability to contribute
        information.

        Coefficients are interpreted as log-hazard ratios on the subdistribution hazard scale,
        not the cause-specific hazard. The model directly targets cumulative incidence (F_j),
        making it ideal for policy-relevant questions about the probability of experiencing
        a specific cause.

        Examples
        --------
        Fit a Fine-Gray model on the competing-risks `mgus2` dataset, modeling the cumulative
        incidence of plasma-cell malignancy (pcm) as a function of age and sex:

        ```{python}
        import numpy as np
        import greenwood as gw

        mg = gw.load_dataset("mgus2")
        etime = np.where(mg["pstat"] == 1, mg["ptime"], mg["futime"])
        cause = np.where(mg["pstat"] == 1, 1, 2 * mg["death"])
        y = gw.Surv.multistate(etime, event=cause, states=("pcm", "death"))
        fg = gw.FineGray("pcm").fit(y, mg[["age", "sex"]])
        fg
        ```

        Extract hazard ratios with exponentiation for interpretation:

        ```{python}
        gw.tidy(fg, exponentiate=True)
        ```
        """
        from ._cox import _design_matrix

        if not surv.is_multistate:
            raise ValueError(
                "FineGray needs a multi-state response; build it with Surv.multistate."
            )
        assert surv.states is not None
        if self.cause in surv.states:
            target = surv.states.index(self.cause) + 1
        elif isinstance(self.cause, int) and 1 <= self.cause <= len(surv.states):
            target = self.cause
        else:
            raise ValueError(f"cause {self.cause!r} is not one of the states {surv.states}.")

        x, names = _design_matrix(covariates)
        if x.shape[0] != surv.n:
            raise ValueError("Covariates and response must have the same number of rows.")

        time = surv.stop
        cause = surv.status
        keep = ~np.isnan(x).any(axis=1)
        x, time, cause = x[keep], time[keep], cause[keep]

        drop_times, drop_surv = _censoring_km(time, cause)

        def g_before(t: Array) -> Array:
            """Censoring survival just before `t` (drops strictly before `t`).

            Target and competing events are nudged just before their time (as in R's
            `finegray`), so the censoring drop tied at an event time is not counted.
            """
            idx = np.searchsorted(drop_times, t, side="left") - 1
            return np.where(idx >= 0, drop_surv[idx.clip(min=0)], 1.0)

        competing = (cause != target) & (cause != 0)
        g_before_i = g_before(time)  # denominator per subject
        target_times = np.unique(time[cause == target])
        p = x.shape[1]

        def _weights(tj: float) -> Array:
            w = np.zeros(time.shape[0])
            w[time >= tj] = 1.0
            mask = competing & (time < tj)
            w[mask] = float(g_before(np.array([tj]))[0]) / g_before_i[mask]
            return w

        def terms(beta: Array) -> tuple[float, Array, Array]:
            r = np.exp(x @ beta)
            loglik = 0.0
            grad = np.zeros(p)
            info = np.zeros((p, p))
            for tj in target_times:
                w = _weights(float(tj))
                rw = w * r
                s0 = rw.sum()
                s1 = (x * rw[:, None]).sum(axis=0)
                s2 = (x * rw[:, None]).T @ x
                dy = (time == tj) & (cause == target)
                d = float(dy.sum())
                z1 = s1 / s0
                loglik += float((x[dy] @ beta).sum()) - d * np.log(s0)
                grad += x[dy].sum(axis=0) - d * z1
                info += d * (s2 / s0 - np.outer(z1, z1))
            return loglik, grad, info

        beta = np.zeros(p)
        loglik = terms(beta)[0]
        for _ in range(max_iter):
            ll, grad, info = terms(beta)
            step = np.linalg.solve(info, grad)
            beta = beta + step
            new_ll = terms(beta)[0]
            if abs(new_ll - ll) <= tol * (abs(new_ll) + tol):
                loglik = new_ll
                break
            loglik = new_ll

        _, _, info = terms(beta)
        naive_vcov = np.linalg.inv(info)

        # Robust (Lin-Wei) sandwich from per-subject score residuals.
        scores = self._score_residuals(
            beta, x, time, cause, target, target_times, competing, g_before_i, g_before
        )
        robust_vcov = naive_vcov @ (scores.T @ scores) @ naive_vcov

        z = float(norm.ppf(1.0 - (1.0 - self.conf_level) / 2.0))
        self.term_names_ = names
        self.coef_ = beta
        self.hazard_ratio_ = np.exp(beta)
        self.naive_vcov_ = naive_vcov
        self.naive_std_error_ = np.sqrt(np.diag(naive_vcov))
        self.vcov_ = robust_vcov
        self.std_error_ = np.sqrt(np.diag(robust_vcov))
        self.z_ = beta / self.std_error_
        self.p_value_ = 2.0 * norm.sf(np.abs(self.z_))
        self.conf_low_ = beta - z * self.std_error_
        self.conf_high_ = beta + z * self.std_error_
        self.loglik_ = float(loglik)
        self.n_ = int(keep.sum())
        self.n_event_ = int((cause == target).sum())
        return self

    def _score_residuals(
        self,
        beta: Array,
        x: Array,
        time: Array,
        cause: Array,
        target: int,
        target_times: Array,
        competing: Array,
        g_before_i: Array,
        g_before: Any,
    ) -> Array:
        """Per-subject score residuals of the weighted subdistribution partial likelihood."""
        n, p = x.shape
        r = np.exp(x @ beta)
        scores = np.zeros((n, p))
        for tj in target_times:
            w = np.zeros(n)
            w[time >= tj] = 1.0
            mask = competing & (time < tj)
            w[mask] = float(g_before(np.array([tj]))[0]) / g_before_i[mask]
            rw = w * r
            s0 = rw.sum()
            xbar = (x * rw[:, None]).sum(axis=0) / s0
            dy = (time == tj) & (cause == target)
            d = float(dy.sum())
            dlambda = d / s0
            # Event term for the subjects failing (target) at tj.
            scores[dy] += x[dy] - xbar
            # Compensator for every weighted member of the risk set.
            member = w > 0
            scores[member] -= w[member, None] * (x[member] - xbar) * (r[member] * dlambda)[:, None]
        return scores

    def _coefficient_columns(self, *, exponentiate: bool = False) -> dict[str, Any]:
        estimate = self.hazard_ratio_ if exponentiate else self.coef_
        low = np.exp(self.conf_low_) if exponentiate else self.conf_low_
        high = np.exp(self.conf_high_) if exponentiate else self.conf_high_
        return {
            "term": self.term_names_,
            "estimate": estimate,
            "std_error": self.std_error_,
            "statistic": self.z_,
            "p_value": self.p_value_,
            "conf_low": low,
            "conf_high": high,
        }

    def to_pandas(self, *, exponentiate: bool = False) -> Any:
        """Return the Fine-Gray coefficient table as a pandas DataFrame.

        This method exports one row per term with coefficient estimates, robust standard
        errors, test statistics, p-values, and confidence limits. Set
        `exponentiate=True` to return subdistribution hazard ratios and exponentiated
        confidence limits.

        Parameters
        ----------
        exponentiate : bool, default False
            Whether to return subdistribution hazard ratios instead of log-scale
            coefficients.

        Returns
        -------
        pandas.DataFrame
            A tidy DataFrame with columns `term`, `estimate`, `std_error`, `statistic`,
            `p_value`, `conf_low`, and `conf_high`.

        Raises
        ------
        ImportError
            If pandas is not installed.

        Examples
        --------
        Export the fitted coefficient table to pandas:

        ```{python}
        fg.to_pandas()
        ```

        To report subdistribution hazard ratios instead of coefficients:

        ```{python}
        fg.to_pandas(exponentiate=True)
        ```
        """
        try:
            import pandas as pd
        except ImportError as e:
            raise ImportError(
                "pandas is required for to_pandas(). Install it with: pip install pandas"
            ) from e

        return pd.DataFrame(self._coefficient_columns(exponentiate=exponentiate))

    def to_polars(self, *, exponentiate: bool = False) -> Any:
        """Return the Fine-Gray coefficient table as a Polars DataFrame.

        This method exports one row per term with coefficient estimates, robust standard
        errors, test statistics, p-values, and confidence limits. Set
        `exponentiate=True` to return subdistribution hazard ratios and exponentiated
        confidence limits.

        Parameters
        ----------
        exponentiate : bool, default False
            Whether to return subdistribution hazard ratios instead of log-scale
            coefficients.

        Returns
        -------
        polars.DataFrame
            A tidy DataFrame with columns `term`, `estimate`, `std_error`, `statistic`,
            `p_value`, `conf_low`, and `conf_high`.

        Raises
        ------
        ImportError
            If polars is not installed.

        Examples
        --------
        Export the fitted coefficient table to Polars:

        ```{python}
        fg.to_polars()
        ```
        """
        try:
            import polars as pl
        except ImportError as e:
            raise ImportError(
                "polars is required for to_polars(). Install it with: pip install polars"
            ) from e

        return pl.DataFrame(self._coefficient_columns(exponentiate=exponentiate))

    def to_arrow(self, *, exponentiate: bool = False) -> Any:
        """Return the Fine-Gray coefficient table as a PyArrow Table.

        This method exports one row per term with coefficient estimates, robust standard
        errors, test statistics, p-values, and confidence limits. Set
        `exponentiate=True` to return subdistribution hazard ratios and exponentiated
        confidence limits.

        Parameters
        ----------
        exponentiate : bool, default False
            Whether to return subdistribution hazard ratios instead of log-scale
            coefficients.

        Returns
        -------
        pyarrow.Table
            A table with columns `term`, `estimate`, `std_error`, `statistic`, `p_value`,
            `conf_low`, and `conf_high`.

        Raises
        ------
        ImportError
            If pyarrow is not installed.

        Examples
        --------
        Export the fitted coefficient table to Arrow:

        ```{python}
        fg.to_arrow()
        ```
        """
        try:
            import pyarrow as pa
        except ImportError as e:
            raise ImportError(
                "pyarrow is required for to_arrow(). Install it with: pip install pyarrow"
            ) from e

        return pa.table(self._coefficient_columns(exponentiate=exponentiate))


def _register_finegray() -> None:
    from .summaries import register_glance, register_tidier

    def _tidy(model: FineGray, *, exponentiate: bool = False, **_: Any) -> Any:
        return model.to_pandas(exponentiate=exponentiate)

    def _glance(model: FineGray, **_: Any) -> Any:
        import pandas as pd

        return pd.DataFrame([{"n": model.n_, "nevent": model.n_event_, "loglik": model.loglik_}])

    register_tidier("greenwood._competing.FineGray", _tidy)
    register_glance("greenwood._competing.FineGray", _glance)


_register_finegray()


class MultiState:
    """Aalen-Johansen estimator of multi-state transition and occupancy probabilities.

    Given counting-process intervals `(start, stop]` each labelled with the state occupied
    (`state`) and the state transitioned to at `stop` (`event`, or a censoring marker), this
    forms the Aalen-Johansen product `P(0, t) = prod (I + dA(s))` and reports the state
    occupancy probabilities over time. Occupancy probabilities are validated to tolerance
    against R's `survfit` multi-state `pstate`. (Competing risks and Kaplan-Meier are special
    cases handled by `AalenJohansen` and `KaplanMeier`.)

    Returns
    -------
    Fitted estimator
        Call `fit()` to produce a fitted estimator with cached results (`states_`,
        `time_`, `occupancy_`, and internal transition matrices), accessible as tidy
        DataFrames.

    Examples
    --------
    The `mgus2` patients occupy three states in turn: `"mgus"` at entry, then possibly
    `"pcm"` (plasma-cell malignancy), then `"death"`. Reshape the wide dataset into
    counting-process intervals `(start, stop]`, one interval per state occupied, labelled with
    the state entered next. A patient who progresses before dying contributes two intervals;
    everyone else contributes one. Fitting reports the occupancy probability of each state
    over time.

    ```{python}
    import greenwood as gw

    mg = gw.load_dataset("mgus2")
    start, stop, state, event = [], [], [], []
    for i in range(len(mg)):
        pt, ft = mg["ptime"][i], mg["futime"][i]
        progressed, died = mg["pstat"][i] == 1, mg["death"][i] == 1
        if progressed and pt < ft:
            start += [0, pt]; stop += [pt, ft]; state += ["mgus", "pcm"]
            event += ["pcm", "death" if died else None]
        else:
            start += [0]; stop += [ft]; state += ["mgus"]
            event += ["death" if died else ("pcm" if progressed else None)]
    rows = [(a, b, s, e) for a, b, s, e in zip(start, stop, state, event) if b > a]
    start, stop, state, event = map(list, zip(*rows))
    ms = gw.MultiState().fit(start, stop, state, event, states=("mgus", "pcm", "death"))
    ms.to_pandas()
    ```

    The `ms` object fit here, along with the interval arrays, is reused by the method examples
    below.
    """

    def __repr__(self) -> str:
        if getattr(self, "states_", None) is None:
            return "MultiState() <unfitted>"
        from ._repr import align_table, num

        states = ", ".join(str(s) for s in self.states_)
        final = self.occupancy_[-1]
        rows = [[num(p)] for p in final]
        table = align_table(["final occupancy"], rows, [str(s) for s in self.states_])
        return "\n".join(
            [
                "MultiState (Aalen-Johansen multi-state model)",
                "",
                f"states: {states}",
                f"times: {len(self.time_)}",
                "",
                table,
            ]
        )

    def fit(
        self, start: Any, stop: Any, state: Any, event: Any, *, states: Any = None
    ) -> MultiState:
        """Fit a multi-state model using counting-process intervals.

        Estimates state-occupancy probabilities and transition dynamics over time from
        counting-process data (multiple overlapping intervals per subject). The model tracks
        how subjects move between states and computes the probability of being in each state
        at any given time, accounting for censoring and competing transitions.

        This estimator is ideal for:

        - **Panel data**: Subjects observed at discrete times, with state changes recorded
          between observations.
        - **Chronic-disease progression**: Modeling progression through stages (e.g., MGUS →
          PCM → death).
        - **Multi-event data**: Non-absorbing or semi-absorbing intermediate states.
        - **Irregular follow-up**: Each subject's observation times may differ.

        The model estimates without distributional assumptions via non-parametric maximum
        likelihood. Occupancy probabilities are computed as a product of transition
        matrices evaluated at each event time.

        Parameters
        ----------
        start
            Start time of each interval. Can be a 1-D array-like (or Polars/Pandas
            Series). Intervals are half-open: (start, stop].
        stop
            Stop (end) time of each interval. Must have the same length as `start`.
            Intervals define subject-time windows.
        state
            The state occupied during each interval (the "from" state). Can be string, int,
            or other hashable label. Must have the same length as `start` and `stop`.
        event
            The state transitioned to at the stop time. If `None`, NaN, or 0, the subject
            was censored (no transition). Otherwise, must be a valid state label. Must have
            the same length as `start` and `stop`.
        states
            Optional ordered sequence of all state labels (default: auto-detected from data).
            If provided, must include all unique states in `state` and `event`. Useful for
            enforcing a specific state ordering (e.g., disease progression order) or
            including states with no observed transitions.

        Returns
        -------
        MultiState
            The fitted estimator object itself (for method chaining) with cached results
            (`states_`, `time_`, `occupancy_`, `transition_`) accessible via `to_pandas()`
            and related methods. Occupancy probabilities and transition probabilities can
            be queried at any time via `predict()`.

        Notes
        -----
        **Data format**: Intervals are half-open (start, stop]. Each row represents a
        subject-interval: the period during which the subject was in `state` and either
        remained (censored) or transitioned to `event` at `stop`.

        **State labels**: States can be strings, integers, or other hashable types (e.g.,
        tuples). Mixed types are not allowed. Transitions between the same state (self-loops)
        treated as censoring.

        **Handling censoring**: Censored intervals (event = None/NaN/0) contribute
        right-censored data. Subjects re-enter their original state after censoring (common
        in discrete-time or periodic follow-up studies).

        **Computational method**: Non-parametric maximum likelihood. At each distinct event
        time, transition intensities are estimated from risk sets (subjects at risk to
        transition from each state), and occupancy is updated by matrix multiplication of
        transition probabilities.

        Examples
        --------
        Build a multi-state model from counting-process intervals. First, prepare interval
        data from a chronic-disease cohort:

        ```{python}
        import greenwood as gw
        import numpy as np

        mg = gw.load_dataset("mgus2")
        start, stop, state, event = [], [], [], []
        for i in range(len(mg)):
            pt, ft = mg["ptime"][i], mg["futime"][i]
            progressed, died = mg["pstat"][i] == 1, mg["death"][i] == 1
            if progressed and pt < ft:
                start += [0, pt]; stop += [pt, ft]; state += ["mgus", "pcm"]
                event += ["pcm", "death" if died else None]
            else:
                start += [0]; stop += [ft]; state += ["mgus"]
                event += ["death" if died else ("pcm" if progressed else None)]
        rows = [(a, b, s, e) for a, b, s, e in zip(start, stop, state, event) if b > a]
        start, stop, state, event = map(list, zip(*rows))
        ms = gw.MultiState().fit(start, stop, state, event, states=("mgus", "pcm", "death"))
        ms
        ```

        Query occupancy probabilities at specific follow-up times (60, 120, and 240 months):

        ```{python}
        ms.predict([60, 120, 240])
        ```
        """
        from ._surv import _to_1d_array

        t0 = _to_1d_array(start)
        t1 = _to_1d_array(stop)
        frm = _to_1d_array(state, dtype=object)
        evt = _to_1d_array(event, dtype=object)
        if not (t0.shape[0] == t1.shape[0] == frm.shape[0] == evt.shape[0]):
            raise ValueError("start, stop, state, and event must have the same length.")

        def is_censor(v: Any) -> bool:
            return v is None or v == 0 or (isinstance(v, float) and v != v)

        if states is None:
            labels = set(frm.tolist()) | {v for v in evt.tolist() if not is_censor(v)}
            self.states_ = tuple(sorted(labels, key=str))
        else:
            self.states_ = tuple(states)
        index = {s: i for i, s in enumerate(self.states_)}
        n_states = len(self.states_)

        from_idx = np.array([index[s] for s in frm], dtype=int)
        to_idx = np.array([-1 if is_censor(v) else index[v] for v in evt], dtype=int)

        # Initial distribution from the entry intervals (earliest start).
        entry = t0 == t0.min()
        p0 = np.array([float((from_idx[entry] == j).sum()) for j in range(n_states)])
        p0 /= p0.sum()

        times = np.unique(t1)
        prob = np.eye(n_states)
        occupancy = np.empty((times.shape[0], n_states))
        transition = np.empty((times.shape[0], n_states, n_states))
        for row, t in enumerate(times):
            increment = np.zeros((n_states, n_states))
            for j in range(n_states):
                y_j = float(((from_idx == j) & (t0 < t) & (t1 >= t)).sum())
                if y_j == 0:
                    continue
                for k in range(n_states):
                    if k != j:
                        d = float(((from_idx == j) & (t1 == t) & (to_idx == k)).sum())
                        increment[j, k] = d / y_j
                increment[j, j] = -increment[j].sum()
            prob = prob @ (np.eye(n_states) + increment)
            transition[row] = prob
            occupancy[row] = p0 @ prob

        self.time_ = times
        self.occupancy_ = occupancy
        self.transition_ = transition  # P(0, t): row = time, [from, to]
        self._p0 = p0
        return self

    def predict(self, times: Any) -> Any:
        """State occupancy probabilities at specified times.

        Evaluates the state occupancy probabilities (probability of being in each state) at
        requested times. The occupancy probability for each state is a right-continuous step
        function, defined at the fitted time points and interpolated or held constant elsewhere.

        Parameters
        ----------
        times
            Time points at which to evaluate state occupancy. Can be a scalar or array-like of
            floats. Values before the first transition time use the initial distribution;
            values after the last transition time use the final distribution.

        Returns
        -------
        DataFrame
            A DataFrame with a `time` column and one column per state, containing occupancy
            probabilities (values between 0 and 1) at each query time. Row indices match the
            input times.

        Examples
        --------
        Evaluate occupancy probabilities at specific times. The result shows how the
        probability of being in each state changes over time:

        ```{python}
        import greenwood as gw

        mg = gw.load_dataset("mgus2")
        start, stop, state, event = [], [], [], []
        for i in range(len(mg)):
            pt, ft = mg["ptime"][i], mg["futime"][i]
            progressed, died = mg["pstat"][i] == 1, mg["death"][i] == 1
            if progressed and pt < ft:
                start += [0, pt]; stop += [pt, ft]; state += ["mgus", "pcm"]
                event += ["pcm", "death" if died else None]
            else:
                start += [0]; stop += [ft]; state += ["mgus"]
                event += ["death" if died else ("pcm" if progressed else None)]
        rows = [(a, b, s, e) for a, b, s, e in zip(start, stop, state, event) if b > a]
        start, stop, state, event = map(list, zip(*rows))
        ms = gw.MultiState().fit(start, stop, state, event, states=("mgus", "pcm", "death"))
        ms.predict([60, 120, 240])
        ```
        """
        import pandas as pd

        query = np.atleast_1d(np.asarray(times, dtype=float))
        idx = np.searchsorted(self.time_, query, side="right") - 1
        out = np.where(idx[:, None] >= 0, self.occupancy_[idx.clip(min=0)], self._p0[None, :])
        frame = pd.DataFrame({str(s): out[:, j] for j, s in enumerate(self.states_)})
        frame.insert(0, "time", query)
        return frame

    def _table_columns(self) -> dict[str, Array]:
        cols: dict[str, Array] = {"time": self.time_}
        for j, state in enumerate(self.states_):
            cols[str(state)] = self.occupancy_[:, j]
        return cols

    def to_pandas(self) -> Any:
        """Return occupancy probabilities over time as a pandas DataFrame.

        This method exports one row per distinct time and one column per state, where each
        state column contains its occupancy probability at that time.

        Returns
        -------
        pandas.DataFrame
            A tidy DataFrame with a `time` column and one probability column per state.

        Raises
        ------
        ImportError
            If pandas is not installed.

        Examples
        --------
        Export the state-occupancy probabilities to pandas:

        ```{python}
        ms.to_pandas()
        ```
        """
        try:
            import pandas as pd
        except ImportError as e:
            raise ImportError(
                "pandas is required for to_pandas(). Install it with: pip install pandas"
            ) from e

        return pd.DataFrame(self._table_columns())

    def to_polars(self) -> Any:
        """Return occupancy probabilities over time as a Polars DataFrame.

        This method exports one row per distinct time and one column per state, where each
        state column contains its occupancy probability at that time.

        Returns
        -------
        polars.DataFrame
            A tidy DataFrame with a `time` column and one probability column per state.

        Raises
        ------
        ImportError
            If polars is not installed.

        Examples
        --------
        Export the state-occupancy probabilities to Polars:

        ```{python}
        ms.to_polars()
        ```
        """
        try:
            import polars as pl
        except ImportError as e:
            raise ImportError(
                "polars is required for to_polars(). Install it with: pip install polars"
            ) from e

        return pl.DataFrame(self._table_columns())

    def to_arrow(self) -> Any:
        """Return occupancy probabilities over time as a PyArrow Table.

        This method exports one row per distinct time and one column per state to Arrow
        for efficient interchange with Arrow-based tools.

        Returns
        -------
        pyarrow.Table
            A table with a `time` column and one probability column per state.

        Raises
        ------
        ImportError
            If pyarrow is not installed.

        Examples
        --------
        Export the state-occupancy probabilities to Arrow:

        ```{python}
        ms.to_arrow()
        ```
        """
        try:
            import pyarrow as pa
        except ImportError as e:
            raise ImportError(
                "pyarrow is required for to_arrow(). Install it with: pip install pyarrow"
            ) from e

        return pa.table(self._table_columns())
