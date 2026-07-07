"""Royston-Parmar flexible parametric survival models.

A Royston-Parmar model puts a restricted cubic spline in `log(time)` on the log cumulative
hazard:

    log H(t | x) = s(log t; gamma) + x . beta,

so `S(t | x) = exp(-exp(s(log t) + x.beta))`. The spline `s` has boundary knots at the
extreme uncensored log times and internal knots at quantiles in between; its flexibility is
set by `df` (the number of spline terms, equivalently one more than the number of internal
knots). With `df=1` the spline is linear in `log t`, which is exactly a Weibull proportional
hazards model, so `RoystonParmar(df=1)` reproduces a Weibull fit; larger `df` relaxes the
parametric shape while keeping smooth, extrapolatable survival and hazard curves.

Coefficients are fit by maximum likelihood for right-censored data. This is the proportional
hazards (log cumulative hazard) scale, matching R's `flexsurv::flexsurvspline(scale="hazard")`
and Stata's `stpm2`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt
from scipy.optimize import minimize
from scipy.stats import norm

from ._cox import _design_matrix
from ._parametric import _num_hessian

if TYPE_CHECKING:
    from ._surv import Surv

__all__ = ["RoystonParmar"]

Array = npt.NDArray[Any]


def _cube_plus(a: Array) -> Array:
    return np.where(a > 0.0, a, 0.0) ** 3


def _square_plus(a: Array) -> Array:
    return np.where(a > 0.0, a, 0.0) ** 2


def _rcs_basis(u: Array, knots: Array) -> tuple[Array, Array]:
    """Restricted cubic spline basis and its derivative at `u`.

    `knots` is the sorted knot vector (boundary knots first and last, internal in between).
    Returns `(basis, deriv)`, each with columns `[1, u, v_1, ..., v_m]` where `v_j` are the
    Royston-Parmar internal-knot terms; `deriv` holds the columnwise `d/du`.
    """
    kmin, kmax = float(knots[0]), float(knots[-1])
    internal = knots[1:-1]
    n = u.shape[0]
    ncol = 2 + internal.shape[0]
    basis = np.empty((n, ncol))
    deriv = np.empty((n, ncol))
    basis[:, 0], deriv[:, 0] = 1.0, 0.0
    basis[:, 1], deriv[:, 1] = u, 1.0
    for j, kj in enumerate(internal):
        lam = (kmax - float(kj)) / (kmax - kmin)
        basis[:, 2 + j] = (
            _cube_plus(u - kj) - lam * _cube_plus(u - kmin) - (1.0 - lam) * _cube_plus(u - kmax)
        )
        deriv[:, 2 + j] = 3.0 * (
            _square_plus(u - kj)
            - lam * _square_plus(u - kmin)
            - (1.0 - lam) * _square_plus(u - kmax)
        )
    return basis, deriv


class RoystonParmar:
    """Royston-Parmar flexible parametric survival model (proportional hazards scale).

    The Royston-Parmar model offers a middle ground between rigid parametric models (like
    Weibull) and fully non-parametric methods (like Kaplan-Meier). It models the log baseline
    cumulative hazard as a smooth spline function on the log-time scale, combined with
    proportional-hazards covariate effects. This allows flexible baseline shapes while
    maintaining interpretable proportional-hazards covariate coefficients—a key advantage over
    fully parametric AFT models.

    The model uses restricted cubic splines with a fixed number of degrees of freedom (controlled
    by knots placed at quantiles of event times). A low df value (e.g., df=1) approaches a
    Weibull fit; higher df values (e.g., df=3 or 4) provide greater flexibility. Call `fit()`
    with a right-censored `Surv` response and a design matrix. The model reports spline and
    covariate coefficients, fitted knot locations, log-likelihood, and supports predictions of
    survival at specified times and covariate values.

    The implementation uses maximum likelihood estimation with constraints that ensure monotone
    increasing log cumulative hazard (valid hazard functions). The flexible baseline makes this
    model useful when baseline hazard shape is unknown but important, yet you want interpretable
    proportional-hazards effects of covariates. Results can be exported to tidy DataFrames or
    accessed as coefficient arrays.

    Parameters
    ----------
    df
        Spline degrees of freedom: the number of spline terms beyond the intercept, equal to
        one more than the number of internal knots. `df=1` is a Weibull model; `df=3`
        (two internal knots) is a common flexible default.
    conf_level
        Confidence level for coefficient intervals (default 0.95).

    Returns
    -------
    Not applicable at instantiation. Call `fit()` to produce a fitted estimator with
    cached results (`coef_`, `std_error_`, `z_`, `p_value_`, `conf_low_`, `conf_high_`,
    `knots_`, `loglik_`, `aic_`, `bic_`), accessible as arrays or exported to DataFrames
    via `tidy()` or `to_pandas()`/`to_polars()`/`to_arrow()`.

    Notes
    -----
    Call `fit(surv, covariates)` with a right-censored `Surv` response and a covariate design
    (a dataframe, a 2-D array, or a formula string with `data`). Results are exposed as arrays
    (`coef_`, `std_error_`, ...), the fitted `knots_`, and tidy frames via `to_pandas()`,
    `to_polars()`, `to_arrow()`.

    Examples
    --------
    Build a `Surv` response from the bundled `lung` dataset and fit a flexible model with three
    spline degrees of freedom and `age` and `sex` as covariates. Printing the fitted object
    reports the spline and covariate coefficients and the log-likelihood.

    ```{python}
    import greenwood as gw

    lung = gw.load_dataset("lung")
    y = gw.Surv.right(lung["time"], event=(lung["status"] == 2))
    rp = gw.RoystonParmar(df=3).fit(y, lung[["age", "sex"]])
    rp
    ```

    The `rp` object fit here is reused by the method examples below.
    """

    def __init__(self, df: int = 3, *, conf_level: float = 0.95) -> None:
        if df < 1:
            raise ValueError(f"df must be at least 1, got {df}.")
        if not 0.0 < conf_level < 1.0:
            raise ValueError(f"conf_level must be in (0, 1), got {conf_level}.")
        self.df = df
        self.conf_level = conf_level

    def __repr__(self) -> str:
        if getattr(self, "coef_", None) is None:
            return f"RoystonParmar(df={self.df}, conf_level={self.conf_level}) <unfitted>"
        from ._repr import align_table, fixed, num

        rows = [
            [num(c), num(se), fixed(z, 3), num(p)]
            for c, se, z, p in zip(self.coef_, self.std_error_, self.z_, self.p_value_, strict=True)
        ]
        table = align_table(["coef", "se(coef)", "z", "p"], rows, list(self.term_names_))
        return "\n".join(
            [
                f"RoystonParmar (flexible parametric survival, df={self.df})",
                "",
                table,
                "",
                f"n = {self.n_}, events = {self.n_event_}",
                f"Log-likelihood = {num(self.loglik_)}",
            ]
        )

    def fit(self, surv: Surv, covariates: Any = None, *, data: Any = None) -> RoystonParmar:
        """Fit the Royston-Parmar flexible parametric model to survival data.

        Fits a flexible parametric survival model to a right-censored response and optional
        covariates. The model uses restricted cubic splines on the log-time scale to flexibly
        estimate the baseline cumulative hazard, combined with proportional-hazards covariate
        effects. This combines the interpretability of proportional-hazards regression with the
        flexibility of non-parametric methods.

        The spline flexibility is controlled by `df` (degrees of freedom): `df=1` recovers a
        Weibull model; higher `df` values provide more flexibility to capture non-standard
        baseline hazard shapes. An intercept is added automatically. Covariates are optional;
        if omitted, the fit is a flexible univariate survival model (baseline hazard only).

        Parameters
        ----------
        surv
            A right-censored `Surv` response. Built with `Surv.right()`.
        covariates
            Optional. A dataframe (pandas or polars), a 2-D array, or a formula string
            (e.g., `"age + sex"`) evaluated against the `data` argument. If `None` (default),
            fits a univariate model with no covariates.
        data
            A dataframe to evaluate the formula string (ignored if `covariates` is a
            dataframe, array, or `None`).

        Returns
        -------
        The fitted `RoystonParmar` object itself (for method chaining), now with coefficient
        arrays (`coef_`, `std_error_`, `z_`, `p_value_`), fitted knot locations (`knots_`),
        and summary statistics like log-likelihood.

        Notes
        -----
        The Royston-Parmar model parameterizes the log cumulative hazard as a restricted
        cubic spline in log-time, with proportional-hazards covariate effects added linearly.
        Knots are placed at quantiles of observed event times. Maximum likelihood estimation
        is used; constraints ensure that the log cumulative hazard is monotone increasing
        (required for a valid hazard function).

        The model is useful when baseline hazard shape is unknown but important, yet you want
        interpretable proportional-hazards effects of covariates.

        Examples
        --------
        Fit a flexible Royston-Parmar model with three degrees of freedom (two internal knots)
        on the bundled `lung` dataset with `age` and `sex` as covariates:

        ```{python}
        import greenwood as gw

        lung = gw.load_dataset("lung")
        y = gw.Surv.right(lung["time"], event=(lung["status"] == 2))
        rp = gw.RoystonParmar(df=3).fit(y, lung[["age", "sex"]])
        rp
        ```

        Fit a more flexible model with five degrees of freedom:

        ```{python}
        rp_flexible = gw.RoystonParmar(df=5).fit(y, lung[["age", "sex"]])
        rp_flexible
        ```

        Fit a univariate flexible model without covariates:

        ```{python}
        rp_univariate = gw.RoystonParmar(df=3).fit(y)
        rp_univariate
        ```
        """
        from ._nonparametric import NelsonAalen
        from ._surv import CensoringType

        if surv.type is not CensoringType.RIGHT:
            raise NotImplementedError(
                f"RoystonParmar currently supports right-censored responses, not "
                f"{surv.type.value!r}."
            )

        time = surv.stop
        event = surv.event
        if covariates is None:
            design = np.empty((surv.n, 0))
            cov_names: list[str] = []
        else:
            design, cov_names = _design_matrix(covariates, data)
            if design.shape[0] != surv.n:
                raise ValueError("Covariates and response must have the same number of rows.")

        complete = ~np.isnan(design).any(axis=1) if design.shape[1] else np.ones(surv.n, bool)
        keep = (time > 0) & complete
        time, event, design = time[keep], event[keep], design[keep]
        if not event.any():
            raise ValueError("No events remain after dropping missing rows.")

        u = np.log(time)
        event_u = u[event.astype(bool)]
        n_internal = self.df - 1
        probs = np.linspace(0.0, 1.0, n_internal + 2)  # includes 0 and 1 (boundary)
        knots = np.quantile(event_u, probs)
        self.knots_ = knots

        basis, deriv = _rcs_basis(u, knots)
        n_spline = basis.shape[1]
        n_cov = design.shape[1]

        def neg_loglik(theta: Array) -> float:
            gamma = theta[:n_spline]
            beta = theta[n_spline:]
            eta = basis @ gamma + (design @ beta if n_cov else 0.0)
            sprime = deriv @ gamma
            if np.any(sprime <= 0.0):
                return 1e12
            ll = event * (eta + np.log(sprime) - u) - np.exp(eta)
            return -float(ll.sum())

        # Initialize the spline from a least-squares fit of log(Nelson-Aalen) on the basis.
        na = NelsonAalen().fit(surv).to_pandas()
        pos = na["estimate"].to_numpy() > 0
        b_init, _ = _rcs_basis(np.log(na["time"].to_numpy()[pos]), knots)
        gamma0, *_ = np.linalg.lstsq(b_init, np.log(na["estimate"].to_numpy()[pos]), rcond=None)
        x0 = np.concatenate([gamma0, np.zeros(n_cov)])

        result = minimize(neg_loglik, x0, method="BFGS", options={"gtol": 1e-7, "maxiter": 2000})
        theta = result.x
        vcov = np.linalg.inv(_num_hessian(neg_loglik, theta))

        self._knots = knots
        self.term_names_ = [f"gamma{i}" for i in range(n_spline)] + cov_names
        self.coef_ = theta
        self.vcov_ = vcov
        self.std_error_ = np.sqrt(np.diag(vcov))
        self.loglik_ = -float(result.fun)
        self.n_ = int(keep.sum())
        self.n_event_ = int(event.sum())
        self._n_spline = n_spline

        z = float(norm.ppf(1.0 - (1.0 - self.conf_level) / 2.0))
        self.z_ = self.coef_ / self.std_error_
        self.p_value_ = 2.0 * norm.sf(np.abs(self.z_))
        self.conf_low_ = self.coef_ - z * self.std_error_
        self.conf_high_ = self.coef_ + z * self.std_error_
        return self

    def _eta(self, times: Array, x_row: Array) -> tuple[Array, Array]:
        """Log cumulative hazard and spline derivative at `times` for one covariate row."""
        u = np.log(times)
        basis, deriv = _rcs_basis(u, self._knots)
        gamma = self.coef_[: self._n_spline]
        beta = self.coef_[self._n_spline :]
        lp = float(x_row @ beta) if beta.size else 0.0
        return basis @ gamma + lp, deriv @ gamma

    def predict(self, newdata: Any = None, *, type: str = "survival", times: Any = None) -> Any:
        """Predict survival probability, hazard, or cumulative hazard from the fitted model.

        Generates predictions from a fitted Royston-Parmar flexible parametric model. Pass
        `newdata=None` to predict for a baseline subject (all covariates set to 0, or training
        data mean if covariates are centered).

        The Royston-Parmar model flexibly estimates the baseline cumulative hazard via splines,
        then multiplies by exp(eta) for each subject's covariate-adjusted log-hazard eta. This
        produces smooth, covariate-adjusted survival and hazard curves.

        Three prediction types are available:

        1. **Survival** (`type="survival"`): Survival probabilities S(t|x) at specified times.
           Useful for survival curves and prognosis.

        2. **Hazard** (`type="hazard"`): Instantaneous hazard h(t|x) at specified times. Shows
           the rate of events at each time.

        3. **Cumulative hazard** (`type="cumhaz"`): Cumulative hazard H(t|x) at specified times.
           Useful for risk quantification and comparisons.

        Parameters
        ----------
        newdata
            Covariate values for prediction. A DataFrame (Pandas or Polars), 2-D array, or
            `None` (the default). If `None`, uses baseline (all covariates `0` or the training
            data mean). Must have the same columns/features as the training data.
        type
            Prediction type (default `"survival"`):

            - `"survival"`: Survival probabilities S(t|x) = exp(-H(t|x)). Returns a frame
              with `time` column and one column per subject.
            - `"hazard"`: Instantaneous hazard h(t|x) = dH(t|x)/dt. Returns a frame with
              `time` column and one column per subject.
            - `"cumhaz"`: Cumulative hazard H(t|x). Returns a frame with `time` column and
              one column per subject.

        times
            Query times at which to evaluate curves. An array-like of floats. Required unless
            a default grid is used. If None, may raise an error or use a default grid.

        Returns
        -------
        DataFrame with columns `time` (query times) and `subject_1`, `subject_2`, etc.
        (predictions for each subject, one row per query time). All three types return the
        same DataFrame structure.

        Raises
        ------
        ValueError
            If `type=` is not one of `"survival"`, `"hazard"`, or `"cumhaz"`.

        Notes
        -----
        The Royston-Parmar model represents log cumulative hazard as a smooth spline function
        in log-time, with proportional-hazards covariate effects: H(t|x) = exp(eta(t, x)),
        where eta(t, x) = spline(log t) + x*beta. The spline basis and knot locations are
        fitted to the training data; predictions use these fixed basis functions.

        Hazard is computed numerically as the derivative of cumulative hazard, so predictions
        may be slightly noisy if times are coarsely spaced. For smooth hazard predictions,
        use a fine query grid.

        Predictions assume the model is well-specified and fit the training data adequately.

        Examples
        --------
        Read predicted survival probabilities off the fitted curves at chosen times. Here are
        the estimates at 180 and 365 days for the first two subjects:

        ```{python}
        rp.predict(lung[["age", "sex"]][:2], type="survival", times=[180, 365])
        ```

        Predict the instantaneous hazard (force of mortality) at those same times:

        ```{python}
        rp.predict(lung[["age", "sex"]][:2], type="hazard", times=[180, 365])
        ```

        Predict cumulative hazard (total risk accumulated by time t):

        ```{python}
        rp.predict(lung[["age", "sex"]][:2], type="cumhaz", times=[180, 365])
        ```

        Predict for a baseline subject (covariates all zero):

        ```{python}
        rp.predict(type="survival", times=[180, 365])
        ```
        """
        import pandas as pd

        if newdata is None:
            x = np.zeros((1, max(self.coef_.size - self._n_spline, 0)))
        else:
            x, _ = _design_matrix(newdata)
        query = np.atleast_1d(np.asarray(times, dtype=float))
        columns: dict[str, Array] = {}
        for i in range(x.shape[0]):
            eta, sprime = self._eta(query, x[i])
            cumhaz = np.exp(eta)
            if type == "cumhaz":
                columns[f"subject_{i + 1}"] = cumhaz
            elif type == "survival":
                columns[f"subject_{i + 1}"] = np.exp(-cumhaz)
            elif type == "hazard":
                columns[f"subject_{i + 1}"] = cumhaz * sprime / query
            else:
                raise ValueError(
                    f"Unknown predict type {type!r}; use 'survival', 'hazard', or 'cumhaz'."
                )
        frame = pd.DataFrame(columns)
        frame.insert(0, "time", query)
        return frame

    def _coefficient_columns(self) -> dict[str, Any]:
        return {
            "term": self.term_names_,
            "estimate": self.coef_,
            "std_error": self.std_error_,
            "statistic": self.z_,
            "p_value": self.p_value_,
            "conf_low": self.conf_low_,
            "conf_high": self.conf_high_,
        }

    def to_pandas(self) -> Any:
        """Return the coefficient table as a pandas DataFrame.

        This method exports one row per spline or covariate term with coefficient
        estimates, standard errors, Wald statistics, p-values, and confidence limits.

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
        Export the fitted Royston-Parmar coefficients to pandas:

        ```{python}
        rp.to_pandas()
        ```
        """
        try:
            import pandas as pd
        except ImportError as e:
            raise ImportError(
                "pandas is required for to_pandas(). Install it with: pip install pandas"
            ) from e

        return pd.DataFrame(self._coefficient_columns())

    def to_polars(self) -> Any:
        """Return the coefficient table as a Polars DataFrame.

        This method exports one row per spline or covariate term with coefficient
        estimates, standard errors, Wald statistics, p-values, and confidence limits.

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
        Export the fitted Royston-Parmar coefficients to Polars:

        ```{python}
        rp.to_polars()
        ```
        """
        try:
            import polars as pl
        except ImportError as e:
            raise ImportError(
                "polars is required for to_polars(). Install it with: pip install polars"
            ) from e

        return pl.DataFrame(self._coefficient_columns())

    def to_arrow(self) -> Any:
        """Return the coefficient table as a PyArrow Table.

        This method exports one row per spline or covariate term with coefficient
        estimates, standard errors, Wald statistics, p-values, and confidence limits.

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
        Export the fitted Royston-Parmar coefficients to Arrow:

        ```{python}
        rp.to_arrow()
        ```
        """
        try:
            import pyarrow as pa
        except ImportError as e:
            raise ImportError(
                "pyarrow is required for to_arrow(). Install it with: pip install pyarrow"
            ) from e

        return pa.table(self._coefficient_columns())
