"""Cox proportional hazards regression.

`CoxPH` fits the semiparametric proportional-hazards model by maximizing the partial
likelihood with Newton-Raphson, supporting the Efron (default) and Breslow tie corrections.
It reports coefficients, hazard ratios, model-based standard errors, Wald z-tests, and the
three global tests (likelihood-ratio, Wald, score), all validated to tolerance against R's
`survival::coxph`.

It also provides the baseline (Breslow/Efron) cumulative hazard, survival prediction,
martingale and Schoenfeld residuals, the Grambsch-Therneau proportional-hazards test
(`cox_zph`), and the concordance index, all validated against R.

Stratification (per-stratum baselines with shared coefficients) and the robust (Lin-Wei)
sandwich variance, with optional clustering, are supported. The risk sets use the same
entry/exit convention as the rest of Greenwood, so left truncation and counting-process
data are handled.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt
from scipy.stats import chi2, norm

if TYPE_CHECKING:
    from ._surv import Surv

__all__ = ["CoxPH", "ZPHResult"]

Array = npt.NDArray[Any]


@dataclass(frozen=True)
class ZPHResult:
    """Proportional-hazards test results (Grambsch-Therneau).

    A key assumption of the Cox proportional hazards model is that the hazard ratio between
    any two subjects is constant over time (hence "proportional"). When this assumption is
    violated—for example, if a treatment effect diminishes over time—the Cox model may produce
    biased estimates. The Grambsch-Therneau proportional hazards test checks this assumption
    by testing whether scaled residuals are correlated with time.

    `ZPHResult` holds the test results obtained from a fitted Cox model's `cox_zph()` method.
    It provides both per-term tests (one for each covariate) and a global test (jointly across
    all terms). Each test includes a chi-squared test statistic, degrees of freedom, and
    p-value. Results can be printed, accessed via dictionary keys, or exported to pandas/polars/
    pyarrow DataFrames for further analysis or visualization.

    The test uses scaled Schoenfeld residuals, which have a known asymptotic distribution under
    the proportional hazards assumption. Large chi-squared values or small p-values (typically
    p < 0.05) suggest violation of the assumption. When the assumption is violated, stratified
    analysis or time-dependent covariate models may be more appropriate.

    Attributes
    ----------
    transform
        The transformation applied to time when computing the test (e.g., identity, log, rank).
    per_term
        Dictionary mapping each covariate name to `{chisq, df, p_value}` dict.
    global_test
        Dictionary with `{chisq, df, p_value}` for the joint test across all terms.

    Examples
    --------
    A `ZPHResult` comes from a fitted model's `cox_zph` method. Fit a Cox model to the
    bundled `lung` dataset, run the proportional-hazards test, and print the result:

    ```{python}
    import greenwood as gw

    lung = gw.load_dataset("lung")
    y = gw.Surv.right(lung["time"], event=(lung["status"] == 2))
    cox = gw.CoxPH().fit(y, lung[["age", "sex"]])
    zph = cox.cox_zph()
    zph
    ```
    """

    transform: str
    per_term: dict[str, dict[str, float]]
    global_test: dict[str, float]

    def __repr__(self) -> str:
        rows = ", ".join(f"{k}: p={v['p_value']:.4g}" for k, v in self.per_term.items())
        return (
            f"ZPHResult(transform={self.transform!r}, {rows}, "
            f"GLOBAL p={self.global_test['p_value']:.4g})"
        )

    def to_pandas(self) -> Any:
        """Return the test table as a pandas DataFrame (one row per term plus GLOBAL).

        The table contains proportional hazards test statistics for each covariate plus
        a global test across all terms. One row represents one term in the model.

        Returns
        -------
        pandas.DataFrame
            A DataFrame with columns for term, test statistic, p-value, and other
            diagnostics. Includes a GLOBAL row.

        Raises
        ------
        ImportError
            If pandas is not installed.

        Examples
        --------
        ```{python}
        df = zph.to_pandas()
        df
        ```

        The table shows the proportional hazards assumption test results for each term,
        with the GLOBAL row testing the overall assumption.
        """
        try:
            import pandas as pd
        except ImportError as e:
            raise ImportError(
                "pandas is required for to_pandas(). Install it with: pip install pandas"
            ) from e

        rows = [{"term": k, **v} for k, v in self.per_term.items()]
        rows.append({"term": "GLOBAL", **self.global_test})
        return pd.DataFrame(rows)

    def to_polars(self) -> Any:
        """Return the test table as a Polars DataFrame (one row per term plus GLOBAL).

        The table contains proportional hazards test statistics for each covariate plus
        a global test across all terms. One row represents one term in the model.

        Returns
        -------
        polars.DataFrame
            A DataFrame with columns for term, test statistic, p-value, and other
            diagnostics. Includes a GLOBAL row.

        Raises
        ------
        ImportError
            If polars is not installed.

        Examples
        --------
        ```{python}
        df = zph.to_polars()
        df
        ```

        Polars provides better performance and memory efficiency for larger datasets.
        """
        try:
            import polars as pl
        except ImportError as e:
            raise ImportError(
                "polars is required for to_polars(). Install it with: pip install polars"
            ) from e

        rows = [{"term": k, **v} for k, v in self.per_term.items()]
        rows.append({"term": "GLOBAL", **self.global_test})
        return pl.DataFrame(rows)

    def to_arrow(self) -> Any:
        """Return the test table as a PyArrow Table (one row per term plus GLOBAL).

        The table contains proportional hazards test statistics for each covariate plus
        a global test across all terms. One row represents one term in the model.

        Returns
        -------
        pyarrow.Table
            A Table with columns for term, test statistic, p-value, and other
            diagnostics. Includes a GLOBAL row.

        Raises
        ------
        ImportError
            If pyarrow is not installed.

        Examples
        --------
        ```{python}
        tbl = zph.to_arrow()
        tbl
        ```

        PyArrow Tables are useful for interoperability with other Arrow-based tools.
        """
        try:
            import pyarrow as pa
        except ImportError as e:
            raise ImportError(
                "pyarrow is required for to_arrow(). Install it with: pip install pyarrow"
            ) from e

        rows = [{"term": k, **v} for k, v in self.per_term.items()]
        rows.append({"term": "GLOBAL", **self.global_test})
        # Convert list of dicts to proper PyArrow table with column names
        if rows:
            column_names = list(rows[0].keys())
            columns = {name: [row[name] for row in rows] for name in column_names}
            return pa.table(columns)
        return pa.table({})


_TIES = frozenset({"efron", "breslow"})


def _missing_mask(labels: Array) -> Array:
    """Boolean mask of missing entries (None or NaN) in an object array."""
    return np.array([v is None or (isinstance(v, float) and v != v) for v in labels], dtype=bool)


def _to_labels(values: Any, n: int, name: str) -> Array:
    """Coerce group labels (Narwhals series, array, or sequence) to a length-`n` array."""
    from ._surv import _to_1d_array

    labels = _to_1d_array(values, dtype=object)
    if labels.shape[0] != n:
        raise ValueError(f"`{name}` must have the same length as the response ({n}).")
    return labels


def _formula_design(formula: str, data: Any) -> tuple[Array, list[str]]:
    """Build a design matrix from a Wilkinson formula (right-hand side) via formulaic.

    `formula` is the right-hand side only (no `~`), for example
    `"age + sex + ph.ecog"`, `"age + C(celltype)"`, or `"age * sex"`. The intercept column
    that formulaic adds is dropped, so the result matches the no-intercept design the models
    expect (an AFT adds its own intercept). Missing values are preserved so the caller's
    complete-case handling drops the same rows as the response.
    """
    try:
        from formulaic import model_matrix  # pyright: ignore[reportMissingImports]
    except ImportError as error:  # pragma: no cover
        raise ImportError(
            "A formula requires the `formulaic` package. Install it with "
            "`pip install greenwood[formula]`."
        ) from error
    if data is None:
        raise ValueError("A formula string requires the `data` argument.")
    import narwhals as nw  # pyright: ignore[reportMissingImports]  # installed + typed; pyright quirk

    # Normalize any backend (pandas, Polars, PyArrow, ...) to pandas for formulaic.
    frame = nw.from_native(data, eager_only=True).to_pandas()
    matrix = model_matrix(f"~ {formula}", frame, na_action="ignore")
    names = [c for c in matrix.columns if c != "Intercept"]
    if not names:
        raise ValueError("The formula produced no covariates.")
    return np.asarray(matrix[names].to_numpy(), dtype=float), list(names)


def _design_matrix(covariates: Any, data: Any = None) -> tuple[Array, list[str]]:
    """Build a numeric design matrix and term names from covariates.

    Accepts a right-hand-side formula string (with `data`), a 2-D NumPy array, or any
    Narwhals-compatible dataframe. Numeric columns pass through; non-numeric columns are
    treatment-coded (drop-first dummies) with names like `celltypesmallcell`.
    """
    if isinstance(covariates, str):
        return _formula_design(covariates, data)
    if isinstance(covariates, np.ndarray):
        x = np.asarray(covariates, dtype=float)
        if x.ndim != 2:
            raise ValueError("A covariate array must be 2-D (n_obs x n_features).")
        return x, [f"x{i}" for i in range(x.shape[1])]

    import narwhals as nw  # pyright: ignore[reportMissingImports]  # installed + typed; pyright quirk

    frame = nw.from_native(covariates, eager_only=True)
    columns: list[Array] = []
    names: list[str] = []
    for name in frame.columns:
        values = frame[name].to_numpy()
        if values.dtype.kind in "iufb":
            columns.append(values.astype(float))
            names.append(name)
        else:
            levels = sorted({v for v in values.tolist() if v is not None})
            for level in levels[1:]:  # drop the first level as the reference
                columns.append((values == level).astype(float))
                names.append(f"{name}{level}")
    if not columns:
        raise ValueError("No covariates found.")
    return np.column_stack(columns), names


def _to_dataframe(data: dict[str, Any], format: str | None = None) -> Any:
    """Convert a dict to a DataFrame in the requested format.

    Parameters
    ----------
    data
       Dictionary of column names to arrays/lists.
    format
       Output format: `None` (default), `"pandas"`, `"polars"`, or `"pyarrow"`.

       - `None` (default): Auto-detects and tries Polars first, falls back to Pandas,
         then Pyarrow. Raises error if no DataFrame library is available.
       - `"pandas"`: returns pandas.DataFrame.
       - `"polars"`: returns polars.DataFrame.
       - `"pyarrow"`: returns pyarrow.Table.

    Returns
    -------
    pandas.DataFrame, polars.DataFrame, or pyarrow.Table
       The data in the requested format.
    """
    if format is None:
        # Try Polars first (most efficient), then pandas, then pyarrow
        try:
            import polars as pl  # pyright: ignore[reportMissingImports]

            return pl.DataFrame(data)
        except ImportError:
            try:
                import pandas as pd

                return pd.DataFrame(data)
            except ImportError:
                try:
                    import pyarrow as pa

                    return pa.table(data)
                except ImportError as e:
                    raise ImportError(
                        "No DataFrame library found. Install one of: pandas, polars, or pyarrow"
                    ) from e

    if format == "pandas":
        try:
            import pandas as pd
        except ImportError as e:
            raise ImportError(
                "pandas is required for format='pandas'. Install it with: pip install pandas"
            ) from e
        return pd.DataFrame(data)

    if format == "polars":
        try:
            import polars as pl  # pyright: ignore[reportMissingImports]
        except ImportError as e:
            raise ImportError(
                "polars is required for format='polars'. Install it with: pip install polars"
            ) from e
        return pl.DataFrame(data)

    if format == "pyarrow":
        try:
            import pyarrow as pa
        except ImportError as e:
            raise ImportError(
                "pyarrow is required for format='pyarrow'. Install it with: pip install pyarrow"
            ) from e
        return pa.table(data)

    raise ValueError(
        f"Unknown format {format!r}; use 'pandas', 'polars', 'pyarrow', or None for auto-detect"
    )


def _cox_terms(
    beta: Array,
    x: Array,
    entry: Array,
    exit_: Array,
    event: Array,
    weight: Array,
    strata_groups: list[tuple[Array, Array]],
    ties: str,
) -> tuple[float, Array, Array]:
    """Partial log-likelihood, gradient, and observed information at `beta`.

    `strata_groups` is a list of `(member_index, event_times)` pairs; risk sets are
    confined to a stratum, and the terms are summed across strata (the coefficients are
    shared). An unstratified model is a single group.
    """
    p = beta.shape[0]
    loglik = 0.0
    grad = np.zeros(p)
    info = np.zeros((p, p))

    for members, event_times in strata_groups:
        xs = x[members]
        es = entry[members]
        xx = exit_[members]
        ev = event[members]
        ws = weight[members]
        eta = xs @ beta
        risk_score = np.exp(eta) * ws

        for t in event_times:
            at_risk = (es < t) & (xx >= t)
            dying = (xx == t) & ev

            rx = xs[at_risk]
            rr = risk_score[at_risk]
            s0 = rr.sum()
            s1 = rx.T @ rr
            s2 = (rx * rr[:, None]).T @ rx

            w_d = ws[dying]
            loglik += float((w_d * eta[dying]).sum())
            grad += (xs[dying] * w_d[:, None]).sum(axis=0)

            if ties == "breslow":
                d_weight = float(w_d.sum())
                z1 = s1 / s0
                loglik -= d_weight * np.log(s0)
                grad -= d_weight * z1
                info += d_weight * (s2 / s0 - np.outer(z1, z1))
            else:  # efron
                dx = xs[dying]
                dr = risk_score[dying]
                d0 = dr.sum()
                d1 = dx.T @ dr
                d2 = (dx * dr[:, None]).T @ dx
                m = int(dying.sum())
                for tie in range(m):
                    f = tie / m
                    denom = s0 - f * d0
                    z1 = (s1 - f * d1) / denom
                    z2 = (s2 - f * d2) / denom
                    loglik -= float(np.log(denom))
                    grad -= z1
                    info += z2 - np.outer(z1, z1)

    return loglik, grad, info


class CoxPH:
    """Cox proportional hazards model.

    The Cox proportional hazards model is the most widely used regression method for survival
    data. It models the hazard (instantaneous risk of an event) as a multiplicative function
    of covariates: h(t | x) = h₀(t) exp(β'x). The model is semi-parametric: the baseline
    hazard h₀(t) is left unspecified (estimated non-parametrically), while covariate effects
    are estimated parametrically through the log-hazard-ratio coefficients β.

    To use this model, call `fit()` with a right-censored or counting-process `Surv` response
    and a design matrix of covariates (2-D array or DataFrame). The model automatically handles
    stratification (via `by=` in fit), tied event times (via configurable tie-handling methods),
    and can compute predictions, baseline hazards, and diagnostic residuals. Results include
    coefficient estimates with confidence intervals, hazard ratios, standard errors, and
    global significance tests.

    The implementation uses maximum partial likelihood to estimate coefficients. Variance
    estimates use the observed information matrix (Hessian). The model assumes proportional
    hazards—the ratio of hazards between two subjects remains constant over time. This can
    be checked using the `cox_zph()` method for formal tests or diagnostic plots.

    Parameters
    ----------
    ties
        Tie-handling method: `"efron"` (default, as in R) or `"breslow"`.
    conf_level
        Confidence level for coefficient and hazard-ratio intervals (default 0.95).

    Returns
    -------
    Fitted estimator
        Call `fit()` to produce a fitted estimator with cached results (`coef_`,
        `hazard_ratio_`, `std_error_`, `z_`, `p_value_`, `conf_low_`, `conf_high_`,
        `concordance_`, `lr_stat_`, `df_`), accessible as arrays or exported to DataFrames.

    Notes
    -----
    Call `fit(surv, covariates)` with a `Surv` response and a design (a 2-D array or a
    dataframe of covariates). Rows with missing values are dropped (complete-case, as in R's
    default `na.omit`). Results are exposed as arrays (`coef_`, `std_error_`, `hazard_ratio_`,
    …) and as tidy frames via `to_pandas()`, `to_polars()`, `to_arrow()`, and `greenwood.tidy`.

    Examples
    --------
    Build a `Surv` response from the bundled `lung` dataset and fit the model on `age` and
    `sex`. Printing the fitted object reports the coefficient table and global tests in the
    style of R's `summary.coxph`.

    ```{python}
    import greenwood as gw

    lung = gw.load_dataset("lung")
    y = gw.Surv.right(lung["time"], event=(lung["status"] == 2))
    cox = gw.CoxPH().fit(y, lung[["age", "sex"]])
    cox
    ```

    Hazard ratios (and their confidence limits) come from `tidy` with `exponentiate=True`.
    The `cox` object fit here, along with `y` and `lung`, is reused by the method examples
    below.

    ```{python}
    gw.tidy(cox, exponentiate=True)
    ```
    """

    def __init__(self, *, ties: str = "efron", conf_level: float = 0.95) -> None:
        if ties not in _TIES:
            raise ValueError(f"ties must be one of {sorted(_TIES)}, got {ties!r}.")
        if not 0.0 < conf_level < 1.0:
            raise ValueError(f"conf_level must be in (0, 1), got {conf_level}.")
        self.ties = ties
        self.conf_level = conf_level

    def __repr__(self) -> str:
        if getattr(self, "coef_", None) is None:
            return f"CoxPH(ties={self.ties!r}, conf_level={self.conf_level}) <unfitted>"
        from scipy.stats import chi2

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
        lr_p = float(chi2.sf(self.lr_stat_, self.df_))
        lines = [
            f"CoxPH (Cox proportional hazards model, ties={self.ties!r})",
            "",
            table,
            "",
            f"n = {self.n_}, events = {self.n_event_}",
            f"Likelihood ratio test = {num(self.lr_stat_)} on {self.df_} df, p = {num(lr_p)}",
        ]
        if self.robust:
            lines.append("Standard errors: robust (sandwich)")
        return "\n".join(lines)

    def fit(
        self,
        surv: Surv,
        covariates: Any,
        *,
        data: Any = None,
        strata: Any = None,
        robust: bool = False,
        cluster: Any = None,
        max_iter: int = 30,
        tol: float = 1e-9,
    ) -> CoxPH:
        """Fit the model to a `Surv` response and a covariate design.

        `covariates` is a dataframe or 2-D array, or a right-hand-side formula string (for
        example `"age + sex + C(ph.ecog)"`) evaluated against `data`. `strata` gives
        per-stratum baseline hazards with shared coefficients. `robust=True` (or providing
        `cluster` ids) reports the Lin-Wei sandwich variance; `cluster` sums the score
        residuals within groups before forming the sandwich.

        Parameters
        ----------
        surv
            A `Surv` object representing the response (censoring type must be
            right-censored or counting-process).
        covariates
            Covariate design, either a 2-D array, dataframe, or formula string.
        data
            DataFrame to evaluate formula strings against (required if `covariates=`
            is a formula string).
        strata
            Optional stratification variable, giving each stratum its own baseline
            hazard while sharing coefficients. Can be a 1-D array or series.
        robust
            If `True`, report Lin-Wei sandwich variance (robust standard errors).
            Default is `False`.
        cluster
            Optional cluster labels for grouped robust variance estimation.
            Sums score residuals within groups before forming the sandwich.
        max_iter
            Maximum number of iterations for the Newton-Raphson solver. Default is `30`.
        tol
            Convergence tolerance for the optimization. Default is `1e-9`.

        Returns
        -------
        CoxPH
            Returns self with fitted attributes including `coef_`, `std_error_`,
            `hazard_ratio_`, `z_`, `p_value_`, and other model diagnostics.

        Examples
        --------
        Passing `strata=` gives each stratum its own baseline hazard while sharing the
        coefficients. Here we fit `age` and `ph.ecog` stratified by sex, reusing the `y`
        response and `lung` data from the class example above:

        ```{python}
        import greenwood as gw

        gw.CoxPH().fit(y, lung[["age", "ph.ecog"]], strata=lung["sex"]).to_pandas()
        ```

        The `covariates` argument also accepts a right-hand-side formula string (for example
        `"age + sex + C(ph.ecog)"`), and `robust=True` reports the Lin-Wei sandwich variance.
        """
        from ._surv import CensoringType

        if surv.type not in (CensoringType.RIGHT, CensoringType.COUNTING):
            raise NotImplementedError(
                f"CoxPH supports right-censored and counting-process responses, "
                f"not {surv.type.value!r}."
            )

        x, names = _design_matrix(covariates, data)
        if x.shape[0] != surv.n:
            raise ValueError("Covariates and response must have the same number of rows.")

        entry = surv.entry
        exit_ = surv.stop
        event = surv.event
        weight = surv.weights if surv.weights is not None else np.ones(surv.n)

        strata_labels = None if strata is None else _to_labels(strata, surv.n, "strata")
        cluster_labels = None if cluster is None else _to_labels(cluster, surv.n, "cluster")

        # Complete-case analysis: drop rows with any missing covariate, stratum, or cluster.
        keep = ~np.isnan(x).any(axis=1)
        if strata_labels is not None:
            keep &= ~_missing_mask(strata_labels)
        if cluster_labels is not None:
            keep &= ~_missing_mask(cluster_labels)
        x, entry, exit_, event, weight = (
            x[keep],
            entry[keep],
            exit_[keep],
            event[keep],
            weight[keep],
        )
        if strata_labels is not None:
            strata_labels = strata_labels[keep]
        if cluster_labels is not None:
            cluster_labels = cluster_labels[keep]
        if not event.any():
            raise ValueError("No events remain after dropping missing rows.")

        # Group members by stratum (a single group when unstratified).
        # Check for counting-process data with subjects not starting at time 0
        # (a common data preparation error when converting from calendar time)
        if surv.type == CensoringType.COUNTING and entry.min() < 0:
            warnings.warn(
                "Some subjects have negative start times in counting-process data. "
                "Start times must be non-negative.",
                UserWarning,
                stacklevel=2,
            )

        # Warn if not all rows with the same minimum entry time are 0
        # (indicates possible calendar time instead of subject-relative time)
        if surv.type == CensoringType.COUNTING:
            min_entry = entry.min()
            if min_entry == 0 and entry.max() > 0:  # At least one subject enters at 0
                # Check if there are subjects entering at times other than 0
                # by looking for gaps in the entry times that are large
                unique_entries = np.unique(entry)
                if len(unique_entries) > 1:
                    # Check if the pattern looks like calendar time entries
                    # (large differences between entry times like 100, 200, 300)
                    diffs = np.diff(unique_entries)
                    large_diffs = diffs[diffs > 10]  # Threshold for "large" gaps
                    if len(large_diffs) > 0:
                        warnings.warn(
                            "Subjects in counting-process data have different start times, "
                            "some much larger than 0. This may indicate that start/stop times are "
                            "calendar time rather than subject-relative time. "
                            "Each subject's timeline should begin at 0. "
                            "If you have calendar dates, subtract each subject's entry date from "
                            "their start/stop times before fitting.",
                            UserWarning,
                            stacklevel=2,
                        )

        if strata_labels is None:
            strata_groups = [(np.arange(x.shape[0]), np.unique(exit_[event]))]
        else:
            strata_groups = []
            for level in dict.fromkeys(strata_labels.tolist()):
                members = np.nonzero(strata_labels == level)[0]
                ev_times = np.unique(exit_[members][event[members]])
                strata_groups.append((members, ev_times))
        p = x.shape[1]

        def terms(beta: Array) -> tuple[float, Array, Array]:
            return _cox_terms(beta, x, entry, exit_, event, weight, strata_groups, self.ties)

        beta = np.zeros(p)
        loglik_null, grad0, info0 = terms(beta)
        loglik = loglik_null
        for _ in range(max_iter):
            _, grad, info = terms(beta)
            step = np.linalg.solve(info, grad)
            # Newton with step-halving to guarantee the likelihood increases.
            halving = 0
            while True:
                candidate = beta + step
                new_loglik, _, _ = terms(candidate)
                if new_loglik >= loglik - 1e-12 or halving >= 20:
                    break
                step = step / 2.0
                halving += 1
            converged = abs(new_loglik - loglik) <= tol * (abs(new_loglik) + tol)
            beta, loglik = candidate, new_loglik
            if converged:
                break

        _, _, info = terms(beta)
        naive_var = np.linalg.inv(info)

        # Retain the fitted design for diagnostics, baseline hazard, and prediction.
        self._x = x
        self._entry = entry
        self._exit = exit_
        self._event = event
        self._weight = weight
        self._strata_groups = strata_groups
        self._strata_labels = strata_labels
        self._xbar = (weight[:, None] * x).sum(axis=0) / weight.sum()

        self.term_names_ = names
        self.coef_ = beta
        self.naive_vcov_ = naive_var
        self.naive_std_error_ = np.sqrt(np.diag(naive_var))
        self.robust = robust or cluster is not None

        if self.robust:
            scores = self._score_residuals(beta)
            if cluster_labels is not None:
                levels = list(dict.fromkeys(cluster_labels.tolist()))
                scores = np.array([scores[cluster_labels == lev].sum(axis=0) for lev in levels])
            meat = scores.T @ scores
            self.vcov_ = naive_var @ meat @ naive_var
        else:
            self.vcov_ = naive_var

        self.std_error_ = np.sqrt(np.diag(self.vcov_))
        self.hazard_ratio_ = np.exp(beta)
        self.z_ = beta / self.std_error_
        self.p_value_ = 2.0 * norm.sf(np.abs(self.z_))
        self.loglik_ = float(loglik)
        self.loglik_null_ = float(loglik_null)
        self.n_ = int(keep.sum())
        self.n_event_ = int(event.sum())

        z = float(norm.ppf(1.0 - (1.0 - self.conf_level) / 2.0))
        half = z * self.std_error_
        self.conf_low_ = beta - half
        self.conf_high_ = beta + half

        # Global tests (all chi-square on p degrees of freedom).
        self.df_ = p
        self.lr_stat_ = 2.0 * (loglik - loglik_null)
        self.wald_stat_ = float(beta @ np.linalg.solve(self.vcov_, beta))
        self.score_stat_ = float(grad0 @ np.linalg.solve(info0, grad0))
        return self

    # -- baseline hazard & prediction ----------------------------------------

    def _group_label(self, members: Array) -> Any:
        return None if self._strata_labels is None else self._strata_labels[members[0]]

    def _baseline(self) -> list[tuple[Any, Array, Array]]:
        """Uncentered baseline cumulative hazard per stratum.

        Returns `(label, times, cumhaz)` per stratum group, reported at all unique exit
        times (matching R's `basehaz`); the hazard increments only at event times.
        """
        risk_score = np.exp(self._x @ self.coef_) * self._weight
        out: list[tuple[Any, Array, Array]] = []
        for members, _ in self._strata_groups:
            exit_s = self._exit[members]
            event_s = self._event[members]
            times = np.unique(exit_s)
            increments = np.zeros(times.shape[0])
            for i, t in enumerate(times):
                dying = (exit_s == t) & event_s
                if not dying.any():
                    continue
                at_risk = (self._entry[members] < t) & (exit_s >= t)
                s0 = risk_score[members][at_risk].sum()
                dw = self._weight[members][dying].sum()
                if self.ties == "breslow":
                    increments[i] = dw / s0
                else:  # efron
                    d0 = risk_score[members][dying].sum()
                    m = int(dying.sum())
                    increments[i] = sum((dw / m) / (s0 - (tie / m) * d0) for tie in range(m))
            out.append((self._group_label(members), times, np.cumsum(increments)))
        return out

    def baseline_hazard(self, *, format: str | None = None) -> Any:
        """Return the uncentered baseline cumulative hazard and survival as a frame.

        The baseline hazard represents the hazard rate for a reference subject with all
        covariates at their mean values. It is useful for understanding the underlying
        time-to-event distribution estimated by the model, and can be combined with
        individual covariate values to compute predicted survival probabilities for
        specific subjects.

        In Cox proportional hazards models, the hazard for an individual is modeled as:
        h(t | x) = h₀(t) * exp(x'β), where h₀(t) is the baseline hazard. This method
        returns the estimated cumulative baseline hazard H₀(t) at each observed event time,
        evaluated using the Breslow estimator (non-parametric).

        Parameters
        ----------
        format
            Output format: `None` (default), `"pandas"`, `"polars"`, or `"pyarrow"`.

            - `None` (default): Auto-detects and tries Polars first, falls back to
            Pandas, then Pyarrow. Raises an error if no DataFrame library is installed.
            - `"pandas"`: returns pandas.DataFrame.
            - `"polars"`: returns polars.DataFrame.
            - `"pyarrow"`: returns pyarrow.Table.

        Returns
        -------
        pandas.DataFrame, polars.DataFrame, or pyarrow.Table
            A DataFrame with one row per event time containing:

            - `time`: Event times at which the baseline hazard is evaluated.
            - `cumhaz`: Cumulative baseline hazard H₀(t) at each time.
            - `survival`: Baseline survival probability S₀(t) = exp(-H₀(t)).
            - `strata` (if stratified): Stratum label, one baseline hazard per stratum.

        Notes
        -----
        The baseline hazard is evaluated only at the event times in the training data.
        The cumulative hazard is non-decreasing by construction. For stratified models,
        each stratum has its own baseline hazard while coefficients are shared across
        strata, allowing different baseline risks for different groups.

        The baseline survival S₀(t) is computed from the cumulative hazard using the
        relationship S₀(t) = exp(-H₀(t)), consistent with the exponential survival model.

        Examples
        --------
        The baseline cumulative hazard (and the implied baseline survival) is reported at
        every event time, reusing the `cox` fit from the class example above:

        ```{python}
        cox.baseline_hazard()
        ```

        The returned DataFrame shows the estimated hazard and survival trajectory for the
        reference population (covariates at their means). For stratified models, a separate
        baseline is provided for each stratum:

        ```{python}
        cox_stratified = gw.CoxPH().fit(y, lung[["age", "ph.ecog"]], strata=lung["sex"])
        cox_stratified.baseline_hazard()
        ```

        The baseline hazard can be combined with individual predictions to compute
        personalized survival curves (see `predict(type="survival")`).
        """
        # Collect all data into lists
        times_list = []
        cumhaz_list = []
        survival_list = []
        strata_list = []

        for label, times, cumhaz in self._baseline():
            times_list.extend(times)
            cumhaz_list.extend(cumhaz)
            survival_list.extend(np.exp(-cumhaz))
            if self._strata_labels is not None:
                strata_list.extend([label] * len(times))

        # Build data dict
        data = {
            "time": times_list,
            "cumhaz": cumhaz_list,
            "survival": survival_list,
        }
        if self._strata_labels is not None:
            data["strata"] = strata_list

        return _to_dataframe(data, format=format)

    def _linear_predictor(self, x: Array) -> Array:
        """Centered linear predictor `(x - xbar) . beta` (as in R `predict(type='lp')`)."""
        return (x - self._xbar) @ self.coef_

    def predict(
        self,
        newdata: Any = None,
        *,
        type: str = "lp",
        times: Any = None,
        conditional_after: Any = None,
        ci: bool = False,
        format: str | None = None,
    ) -> Any:
        """Predict from the fitted model.

        `type` is one of `"lp"` (centered linear predictor), `"risk"` (`exp(lp)`), or
        `"survival"`. For `"survival"`, returns a frame of survival probabilities at `times`
        (defaulting to the event times), one column per row of `newdata`. Survival prediction
        for stratified models is not yet supported.

        `conditional_after` (a scalar or one value per subject) predicts survival conditional
        on having already survived to that time: the returned value at time `t` is
        `P(T > t | T > c) = S(t) / S(c)`, and is 1 for `t <= c`.

        With `ci=True` (survival only), the frame also carries `_lower` and `_upper` columns
        per subject: a pointwise confidence band from the cumulative-hazard standard error
        (the log transform used by R's `survfit`), at the model's `conf_level`.

        Parameters
        ----------
        newdata
            Covariate design for prediction. If None, predictions are made on the
            fitted data. Can be a 2-D array or dataframe.
        type
            Type of prediction: `"lp"` (centered linear predictor, default),
            `"risk"` (exp of linear predictor), or `"survival"` (survival probability).
        times
            Time points at which to compute survival probabilities (for `type="survival"`).
            Defaults to the event times from the fitted model.
        conditional_after
            Optional scalar or per-subject time for conditional survival prediction.
            Computes P(T > t | T > c) where c is the conditional_after time.
        ci
            If `True` (survival only), include confidence intervals (`_lower` and `_upper`
            columns). Default is `False`.
        format
            Output format (for `type="survival"` only): `None` (default), `"pandas"`,
            `"polars"`, or `"pyarrow"`.

            - `None` (default): Auto-detects and tries Polars first, falls back to Pandas,
              then Pyarrow. Raises an error if no DataFrame library is installed.
            - `"pandas"`: returns pandas.DataFrame.
            - `"polars"`: returns polars.DataFrame.
            - `"pyarrow"`: returns pyarrow.Table.

        Returns
        -------
        ndarray or DataFrame
            For `type="lp"` or `"risk"`, returns a 1-D array with one prediction per row.
            For `type="survival"`, returns a DataFrame with rows for each time point
            and columns for each subject (named `subject_1`, `subject_2`, etc.), optionally
            with `_lower` and `_upper` columns for confidence intervals.

        Examples
        --------
        The default `type="lp"` returns the centered linear predictor, one value per fitted
        subject (reusing the `cox` fit from the class example above):

        ```{python}
        cox.predict(type="lp")[:5]
        ```

        With `type="survival"` and `newdata`, the result is a frame of survival
        probabilities at the requested `times`, one column per new subject. Row slicing with
        `[:3]` works on both pandas and Polars frames:

        ```{python}
        cox.predict(lung[["age", "sex"]][:3], type="survival", times=[180, 365])
        ```

        Passing `ci=True` adds pointwise confidence bands, and `conditional_after=` gives
        survival conditional on having already survived to a landmark time.
        """
        if newdata is None:
            x = self._x
        else:
            x, _ = _design_matrix(newdata)

        if type == "lp":
            return self._linear_predictor(x)
        if type == "risk":
            return np.exp(self._linear_predictor(x))
        if type == "survival":
            if self._strata_labels is not None:
                raise NotImplementedError(
                    "Survival prediction for stratified models is not yet supported."
                )
            _, base_times, base_cumhaz = self._baseline()[0]
            query = base_times if times is None else np.atleast_1d(np.asarray(times, dtype=float))
            idx = np.searchsorted(base_times, query, side="right") - 1
            h0 = np.where(idx >= 0, base_cumhaz[idx.clip(min=0)], 0.0)
            risk = np.exp(x @ self.coef_)  # uncentered: S(t|x) = exp(-H0(t) exp(x.beta))
            if conditional_after is None:
                surv = np.exp(-np.outer(h0, risk))
            else:
                if ci:
                    raise NotImplementedError(
                        "Confidence intervals are not supported with conditional_after."
                    )
                h0_c = self._baseline_cumhaz_at(base_times, base_cumhaz, conditional_after, x)
                delta = np.clip(h0[:, None] - h0_c[None, :], 0.0, None)  # (n_times, n_subj)
                surv = np.exp(-delta * risk[None, :])
            columns = {"time": query}
            if ci:
                se_h = self._cumhaz_se(x, query)  # (n_times, n_subj)
                z = float(norm.ppf(1.0 - (1.0 - self.conf_level) / 2.0))
                lower = surv * np.exp(-z * se_h)
                upper = surv * np.exp(z * se_h)
                for i in range(x.shape[0]):
                    columns[f"subject_{i + 1}"] = surv[:, i]
                    columns[f"subject_{i + 1}_lower"] = lower[:, i]
                    columns[f"subject_{i + 1}_upper"] = upper[:, i]
            else:
                for i in range(x.shape[0]):
                    columns[f"subject_{i + 1}"] = surv[:, i]
            return _to_dataframe(columns, format=format)
        raise ValueError(f"Unknown predict type {type!r}; use 'lp', 'risk', or 'survival'.")

    def _cumhaz_se(self, x_new: Array, query: Array) -> Array:
        """Standard error of the cumulative hazard `H(t | x)` at `query` times, per subject.

        Uses the two-part Breslow-form variance (baseline variability plus the delta-method
        term for coefficient uncertainty), matching R `survfit.coxph`'s `std.chaz` for a
        Breslow fit (approximate for Efron ties, as with the score-residual variance).
        """
        xr, entry, exit_, event, w = self._x, self._entry, self._exit, self._event, self._weight
        ev = event.astype(bool)
        rs = np.exp(xr @ self.coef_) * w
        et = np.unique(exit_[ev])
        p = xr.shape[1]
        s0 = np.empty(len(et))
        xbar = np.empty((len(et), p))
        d = np.empty(len(et))
        for k, t in enumerate(et):
            at_risk = (entry < t) & (exit_ >= t)
            s0[k] = rs[at_risk].sum()
            xbar[k] = (xr[at_risk] * rs[at_risk, None]).sum(axis=0) / s0[k]
            d[k] = w[(exit_ == t) & ev].sum()
        dl0 = d / s0
        cum_part1 = np.cumsum(d / s0**2)  # baseline variance, cumulative over event times
        cum_dl0 = np.cumsum(dl0)
        cum_xbar_dl0 = np.cumsum(xbar * dl0[:, None], axis=0)

        r0 = np.exp(x_new @ self.coef_)  # (n_subj,)
        vcov = self.naive_vcov_
        se = np.zeros((query.shape[0], x_new.shape[0]))
        qi = np.searchsorted(et, query, side="right") - 1
        for j, k in enumerate(qi):
            if k < 0:
                continue  # before the first event: H = 0, se = 0
            # q_subject = r0 * cumsum((x0 - xbar) dLambda0) up to k
            qmat = r0[:, None] * (x_new * cum_dl0[k] - cum_xbar_dl0[k][None, :])  # (n_subj, p)
            var_h = r0**2 * cum_part1[k] + np.einsum("sp,pq,sq->s", qmat, vcov, qmat)
            se[j] = np.sqrt(np.clip(var_h, 0.0, None))
        return se

    @staticmethod
    def _baseline_cumhaz_at(
        base_times: Array, base_cumhaz: Array, conditional_after: Any, x: Array
    ) -> Array:
        """Baseline cumulative hazard at `conditional_after`, broadcast to one value per subject."""
        c = np.asarray(conditional_after, dtype=float)
        if c.ndim == 0:
            c = np.full(x.shape[0], float(c))
        if c.shape[0] != x.shape[0]:
            raise ValueError("conditional_after must be a scalar or one value per subject.")
        idx_c = np.searchsorted(base_times, c, side="right") - 1
        return np.where(idx_c >= 0, base_cumhaz[idx_c.clip(min=0)], 0.0)

    # -- residuals & diagnostics ---------------------------------------------

    def residuals(self, type: str = "martingale", *, format: str | None = None) -> Any:
        """Return `"martingale"` or `"schoenfeld"` residuals.

        Martingale residuals are one per observation; Schoenfeld residuals are one row per
        event (columns are the covariates), ordered by stratum and then event time.

        Parameters
        ----------
        type
            Type of residuals to return: `"martingale"` (default) or `"schoenfeld"`.
            Martingale residuals are one value per observation. Schoenfeld residuals
            are one row per event with one column per covariate.
        format
            Output format (for `type="schoenfeld"` only): `None` (default), `"pandas"`,
            `"polars"`, or `"pyarrow"`.

            - `None` (default): Auto-detects and tries Polars first, falls back to Pandas,
              then Pyarrow. Raises an error if no DataFrame library is installed.
            - `"pandas"`: returns pandas.DataFrame.
            - `"polars"`: returns polars.DataFrame.
            - `"pyarrow"`: returns pyarrow.Table.

            Returns an array for `type="martingale"`.

        Returns
        -------
        ndarray or DataFrame
            For `type="martingale"`, returns a 1-D array with one residual per observation.
            For `type="schoenfeld"`, returns a DataFrame with one row per event and
            one column per covariate, ordered by stratum and then event time.

        Examples
        --------
        Martingale residuals are returned as one value per observation (reusing the `cox`
        fit from the class example above):

        ```{python}
        cox.residuals("martingale")[:5]
        ```

        Passing `"schoenfeld"` instead returns a DataFrame with one row per event and one
        column per covariate.
        """
        if type == "martingale":
            risk = np.exp(self._x @ self.coef_)
            cumhaz_i = np.zeros(self._x.shape[0])
            for (members, _), (_, times, cumhaz) in zip(
                self._strata_groups, self._baseline(), strict=True
            ):
                idx = np.searchsorted(times, self._exit[members], side="right") - 1
                h0 = np.where(idx >= 0, cumhaz[idx.clip(min=0)], 0.0)
                cumhaz_i[members] = h0 * risk[members]
            return self._event.astype(float) - cumhaz_i
        if type == "schoenfeld":
            residuals, _, _ = self._event_contributions()
            arr = np.array(residuals)
            data = {name: arr[:, j] for j, name in enumerate(self.term_names_)}
            return _to_dataframe(data, format=format)
        raise ValueError(f"Unknown residual type {type!r}; use 'martingale' or 'schoenfeld'.")

    def _event_contributions(self) -> tuple[list[Array], list[float], list[Array]]:
        """Per-event Schoenfeld residual, event time, and risk-set covariance share.

        Iterates strata then event times; risk sets are confined to the stratum. For tied
        Efron events, the risk mean is averaged and the covariance split across the ties.
        """
        risk_score = np.exp(self._x @ self.coef_) * self._weight
        residuals: list[Array] = []
        times: list[float] = []
        covariances: list[Array] = []
        for members, event_times in self._strata_groups:
            xs = self._x[members]
            es = self._entry[members]
            xx = self._exit[members]
            ev = self._event[members]
            rs = risk_score[members]
            for t in event_times:
                at_risk = (es < t) & (xx >= t)
                dying = (xx == t) & ev
                rr = rs[at_risk]
                rx = xs[at_risk]
                s0 = rr.sum()
                s1 = (rx * rr[:, None]).sum(axis=0)
                s2 = (rx * rr[:, None]).T @ rx
                if self.ties == "breslow":
                    xbar = s1 / s0
                    cov = s2 / s0 - np.outer(xbar, xbar)
                else:  # efron: average over the tie-adjusted denominators
                    dr = rs[dying]
                    dx = xs[dying]
                    d0 = dr.sum()
                    d1 = (dx * dr[:, None]).sum(axis=0)
                    d2 = (dx * dr[:, None]).T @ dx
                    m = int(dying.sum())
                    means = [(s1 - (tie / m) * d1) / (s0 - (tie / m) * d0) for tie in range(m)]
                    xbar = np.mean(means, axis=0)
                    cov = np.zeros_like(s2)
                    for tie in range(m):
                        f = tie / m
                        z1 = (s1 - f * d1) / (s0 - f * d0)
                        cov += (s2 - f * d2) / (s0 - f * d0) - np.outer(z1, z1)
                    cov = cov / m
                for xi in xs[dying]:
                    residuals.append(xi - xbar)
                    times.append(float(t))
                    covariances.append(cov)
        return residuals, times, covariances

    def cox_zph(self, *, transform: str = "identity") -> ZPHResult:
        """Test the proportional-hazards assumption (Grambsch-Therneau).

        Regresses the scaled Schoenfeld residuals on a transform of time. `transform` is
        `"identity"` (default) or `"log"`, both validated against R's `cox.zph`. (R defaults
        to a Kaplan-Meier transform; `"km"` and `"rank"` are planned.)

        Examples
        --------
        The test returns a `ZPHResult`; printing it summarizes the per-term and global
        p-values (reusing the `cox` fit from the class example above):

        ```{python}
        cox.cox_zph()
        ```

        The full statistics are available as a tidy frame, one row per term plus a `GLOBAL`
        row:

        ```{python}
        cox.cox_zph().to_pandas()
        ```
        """
        residuals, times, covariances = self._event_contributions()
        t = np.array(times)
        if transform == "identity":
            g = t
        elif transform == "log":
            g = np.log(t)
        else:
            raise ValueError(f"transform must be 'identity' or 'log', got {transform!r}.")

        centered = g - g.mean()
        p = self.coef_.shape[0]
        u = np.zeros(p)
        a = np.zeros((p, p))
        c = np.zeros((p, p))
        b = np.zeros((p, p))
        for k in range(len(residuals)):
            gc = centered[k]
            v = covariances[k]
            u += gc * residuals[k]
            a += gc * gc * v
            c += gc * v
            b += v
        # Correct for beta having been estimated (the Schoenfeld residuals are constrained).
        var = a - c @ np.linalg.solve(b, c)

        per_term: dict[str, dict[str, float]] = {}
        for j, name in enumerate(self.term_names_):
            stat = float(u[j] ** 2 / var[j, j])
            per_term[name] = {"chisq": stat, "df": 1, "p_value": float(chi2.sf(stat, 1))}
        global_stat = float(u @ np.linalg.solve(var, u))
        global_test = {
            "chisq": global_stat,
            "df": self.df_,
            "p_value": float(chi2.sf(global_stat, self.df_)),
        }
        return ZPHResult(transform=transform, per_term=per_term, global_test=global_test)

    def _score_residuals(self, beta: Array) -> Array:
        """Breslow-form score (dfbeta-precursor) residuals, one per observation.

        Confined to strata; summed over the event times at which each subject is at risk.
        Used to form the robust (Lin-Wei) sandwich variance.
        """
        n, p = self._x.shape
        scores = np.zeros((n, p))
        for members, event_times in self._strata_groups:
            xs = self._x[members]
            es = self._entry[members]
            xx = self._exit[members]
            ev = self._event[members]
            ws = self._weight[members]
            ri = np.exp(xs @ beta)
            order = np.argsort(event_times)
            etimes = event_times[order]
            xbar = np.zeros((etimes.shape[0], p))
            dlambda = np.zeros(etimes.shape[0])
            for k, t in enumerate(etimes):
                at_risk = (es < t) & (xx >= t)
                dying = (xx == t) & ev
                rr = (ri * ws)[at_risk]
                s0 = rr.sum()
                xbar[k] = (xs[at_risk] * rr[:, None]).sum(axis=0) / s0
                dlambda[k] = ws[dying].sum() / s0
            index = {float(t): k for k, t in enumerate(etimes)}
            for local, gi in enumerate(members):
                x_i = xs[local]
                at = (es[local] < etimes) & (etimes <= xx[local])
                compensator = ri[local] * (
                    x_i * (dlambda[at]).sum() - (xbar[at] * dlambda[at][:, None]).sum(axis=0)
                )
                score = -compensator
                if ev[local]:
                    score = score + (x_i - xbar[index[float(xx[local])]])
                scores[gi] = ws[local] * score
        return scores

    def concordance(self) -> float:
        """Harrell's concordance index (C-statistic) of the fitted risk scores.

        Matches R's `survival::concordance`: a subject that dies at time `t` is treated as
        having failed before another subject still under observation at `t` (including one
        censored exactly at `t`); pairs tied in event time are excluded. For stratified
        models, only within-stratum pairs are compared.

        Examples
        --------
        Harrell's C is returned as a single number, where 0.5 is chance and 1.0 is perfect
        discrimination (reusing the `cox` fit from the class example above):

        ```{python}
        cox.concordance()
        ```
        """
        risk = self._x @ self.coef_
        exit_ = self._exit
        event = self._event
        concordant = 0.0
        comparable = 0.0
        for members, _ in self._strata_groups:
            rk = risk[members]
            ex = exit_[members]
            ev = event[members]
            for i in range(ex.shape[0]):
                if not ev[i]:
                    continue
                later = (ex > ex[i]) | ((ex == ex[i]) & ~ev)
                if not later.any():
                    continue
                comparable += float(later.sum())
                concordant += float(np.sum(rk[i] > rk[later]))
                concordant += 0.5 * float(np.sum(rk[i] == rk[later]))
        return concordant / comparable

    # -- interop --------------------------------------------------------------

    def to_pandas(self, *, exponentiate: bool = False) -> Any:
        """Return a tidy coefficient table as pandas DataFrame (one row per term).

        The table contains coefficient estimates, standard errors, test statistics,
        p-values, and confidence limits. If `exponentiate=True`, returns hazard ratios
        instead of log-hazards.

        Parameters
        ----------
        exponentiate : bool, optional
            If True, return hazard ratios (exp of coefficients). Default is False.

        Returns
        -------
        pandas.DataFrame
            One row per term with columns: term, estimate, std_error, statistic,
            p_value, conf_low, conf_high.

        Raises
        ------
        ImportError
            If pandas is not installed.

        Examples
        --------
        ```{python}
        df = cox.to_pandas()
        df
        ```

        With `exponentiate=True`, estimates become hazard ratios:

        ```{python}
        df_hr = cox.to_pandas(exponentiate=True)
        df_hr
        ```
        """
        try:
            import pandas as pd
        except ImportError as e:
            raise ImportError(
                "pandas is required for to_pandas(). Install it with: pip install pandas"
            ) from e

        if exponentiate:
            estimate = self.hazard_ratio_
            conf_low = np.exp(self.conf_low_)
            conf_high = np.exp(self.conf_high_)
        else:
            estimate = self.coef_
            conf_low = self.conf_low_
            conf_high = self.conf_high_
        return pd.DataFrame(
            {
                "term": self.term_names_,
                "estimate": estimate,
                "std_error": self.std_error_,
                "statistic": self.z_,
                "p_value": self.p_value_,
                "conf_low": conf_low,
                "conf_high": conf_high,
            }
        )

    def to_polars(self, *, exponentiate: bool = False) -> Any:
        """Return a tidy coefficient table as Polars DataFrame (one row per term).

        The table contains coefficient estimates, standard errors, test statistics,
        p-values, and confidence limits. If `exponentiate=True`, returns hazard ratios
        instead of log-hazards.

        Parameters
        ----------
        exponentiate : bool, optional
            If True, return hazard ratios (exp of coefficients). Default is False.

        Returns
        -------
        polars.DataFrame
            One row per term with columns: term, estimate, std_error, statistic,
            p_value, conf_low, conf_high.

        Raises
        ------
        ImportError
            If polars is not installed.

        Examples
        --------
        ```{python}
        df = cox.to_polars()
        df
        ```

        Polars is efficient for larger coefficient tables or further processing.
        """
        try:
            import polars as pl
        except ImportError as e:
            raise ImportError(
                "polars is required for to_polars(). Install it with: pip install polars"
            ) from e

        if exponentiate:
            estimate = self.hazard_ratio_
            conf_low = np.exp(self.conf_low_)
            conf_high = np.exp(self.conf_high_)
        else:
            estimate = self.coef_
            conf_low = self.conf_low_
            conf_high = self.conf_high_
        return pl.DataFrame(
            {
                "term": self.term_names_,
                "estimate": estimate,
                "std_error": self.std_error_,
                "statistic": self.z_,
                "p_value": self.p_value_,
                "conf_low": conf_low,
                "conf_high": conf_high,
            }
        )

    def to_arrow(self, *, exponentiate: bool = False) -> Any:
        """Return a tidy coefficient table as PyArrow Table (one row per term).

        The table contains coefficient estimates, standard errors, test statistics,
        p-values, and confidence limits. If `exponentiate=True`, returns hazard ratios
        instead of log-hazards.

        Parameters
        ----------
        exponentiate : bool, optional
            If True, return hazard ratios (exp of coefficients). Default is False.

        Returns
        -------
        pyarrow.Table
            One row per term with columns: term, estimate, std_error, statistic,
            p_value, conf_low, conf_high.

        Raises
        ------
        ImportError
            If pyarrow is not installed.

        Examples
        --------
        ```{python}
        tbl = cox.to_arrow()
        tbl
        ```

        PyArrow is useful for interoperability with other Arrow-based tools.
        """
        try:
            import pyarrow as pa
        except ImportError as e:
            raise ImportError(
                "pyarrow is required for to_arrow(). Install it with: pip install pyarrow"
            ) from e

        if exponentiate:
            estimate = self.hazard_ratio_
            conf_low = np.exp(self.conf_low_)
            conf_high = np.exp(self.conf_high_)
        else:
            estimate = self.coef_
            conf_low = self.conf_low_
            conf_high = self.conf_high_
        return pa.table(
            {
                "term": self.term_names_,
                "estimate": estimate,
                "std_error": self.std_error_,
                "statistic": self.z_,
                "p_value": self.p_value_,
                "conf_low": conf_low,
                "conf_high": conf_high,
            }
        )


def _tidy_cox(model: CoxPH, *, exponentiate: bool = False, **_: Any) -> Any:
    """broom-style `tidy`: one row per term; `exponentiate` gives hazard ratios."""
    return model.to_pandas(exponentiate=exponentiate)


def _glance_cox(model: CoxPH, **_: Any) -> Any:
    """broom-style `glance`: one-row model summary."""
    import pandas as pd

    return pd.DataFrame(
        [
            {
                "n": model.n_,
                "nevent": model.n_event_,
                "loglik": model.loglik_,
                "aic": -2.0 * model.loglik_ + 2.0 * model.df_,
                "lr_statistic": model.lr_stat_,
                "df": model.df_,
                "lr_p_value": float(chi2.sf(model.lr_stat_, model.df_)),
            }
        ]
    )


def _register_adapters() -> None:
    from .summaries import register_glance, register_tidier

    register_tidier("greenwood._cox.CoxPH", _tidy_cox)
    register_glance("greenwood._cox.CoxPH", _glance_cox)


_register_adapters()
