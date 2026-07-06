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

    Parameters
    ----------
    df
        Spline degrees of freedom: the number of spline terms beyond the intercept, equal to
        one more than the number of internal knots. `df=1` is a Weibull model; `df=3`
        (two internal knots) is a common flexible default.
    conf_level
        Confidence level for coefficient intervals (default 0.95).

    Notes
    -----
    Call `fit(surv, covariates)` with a right-censored `Surv` response and a covariate design
    (a dataframe, a 2-D array, or a formula string with `data`). Results are exposed as arrays
    (`coef_`, `std_error_`, ...), the fitted `knots_`, and tidy frames via `to_pandas()`, `to_polars()`, `to_arrow()`.

    Examples
    --------
    Build a `Surv` response from the bundled `lung` dataset and fit a flexible model with three
    spline degrees of freedom and `age` and `sex` as covariates. Printing the fitted object
    reports the spline and covariate coefficients and the log-likelihood.

    ```{python}
    import greenwood as gw
    from greenwood import Surv

    lung = gw.load_dataset("lung")
    y = Surv.right(lung["time"], event=(lung["status"] == 2))
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
        """Fit the model to a right-censored `Surv` response and an optional covariate design.

        Examples
        --------
        The spline flexibility is set by `df`: `df=1` is a Weibull model, and larger values
        relax the parametric shape. Here is a more flexible fit with two extra degrees of
        freedom, using the same `y` response and `lung` data from the class example above:

        ```{python}
import greenwood as gw
        gw.RoystonParmar(df=5).fit(y, lung[["age", "sex"]])
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
        """Predict `"survival"`, `"hazard"`, or `"cumhaz"` at `times` for each row of `newdata`.

        Returns a frame with a `time` column and one column per subject.

        Examples
        --------
        Read predicted survival probabilities off the fitted curves at chosen times. Here are
        the estimates at 180 and 365 days for the first two subjects (reusing the `rp` fit
        above):

        ```{python}
        rp.predict(lung[["age", "sex"]][:2], type="survival", times=[180, 365])
        ```

        Pass `type="hazard"` or `type="cumhaz"` for the hazard or cumulative hazard at those
        same times instead.
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

        Parameters
        ----------
        None

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

        Parameters
        ----------
        None

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

        Parameters
        ----------
        None

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
