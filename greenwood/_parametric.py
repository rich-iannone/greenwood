"""Parametric accelerated failure time (AFT) models.

`AFT` fits the log-linear (accelerated failure time) model `log(T) = X beta + sigma * W`,
where the error `W` follows a standard distribution chosen by `dist`:

- `"weibull"` and `"exponential"` (exponential fixes `sigma = 1`): minimum extreme value,
- `"lognormal"`: standard normal,
- `"loglogistic"`: standard logistic.

Coefficients are on the log-time scale (with an intercept), matching R's `survreg`. The
model is fit by maximum likelihood for right-censored data, and the coefficients, scale,
standard errors, and log-likelihood are validated to tolerance against `survreg`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt
from scipy.optimize import minimize
from scipy.stats import logistic, norm

from ._cox import _design_matrix

if TYPE_CHECKING:
    from ._surv import Surv

__all__ = ["AFT"]

Array = npt.NDArray[Any]

_DISTS = frozenset({"weibull", "exponential", "lognormal", "loglogistic"})


def _log_density_survival(dist: str, z: Array) -> tuple[Array, Array]:
    """Standardized log-density and log-survival of the error term at `z`."""
    if dist in ("weibull", "exponential"):  # minimum extreme value
        return z - np.exp(z), -np.exp(z)
    if dist == "lognormal":
        return norm.logpdf(z), norm.logsf(z)
    # loglogistic
    return logistic.logpdf(z), logistic.logsf(z)


def _error_quantile(dist: str, p: Array) -> Array:
    """Standardized quantile of the error term `W` at probabilities `p`.

    `p` is the cumulative probability of failure, so the returned `w` satisfies
    `F_W(w) = p`; the corresponding time quantile is `exp(mu + sigma * w)`.
    """
    if dist in ("weibull", "exponential"):  # minimum extreme value
        return np.log(-np.log1p(-p))
    if dist == "lognormal":
        return norm.ppf(p)
    return logistic.ppf(p)  # loglogistic


def _num_hessian(fn: Any, x: Array, rel_step: float = 1e-5) -> Array:
    """Central-difference Hessian of a scalar function at `x`."""
    n = x.shape[0]
    h = rel_step * (np.abs(x) + rel_step)
    hess = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            xi = np.zeros(n)
            xj = np.zeros(n)
            xi[i] = h[i]
            xj[j] = h[j]
            f_pp = fn(x + xi + xj)
            f_pm = fn(x + xi - xj)
            f_mp = fn(x - xi + xj)
            f_mm = fn(x - xi - xj)
            value = (f_pp - f_pm - f_mp + f_mm) / (4.0 * h[i] * h[j])
            hess[i, j] = value
            hess[j, i] = value
    return hess


class AFT:
    """Parametric accelerated failure time model.

    Parameters
    ----------
    dist
        Error distribution: `"weibull"` (default), `"exponential"`, `"lognormal"`, or
        `"loglogistic"`.
    conf_level
        Confidence level for coefficient intervals (default 0.95).

    Notes
    -----
    Call `fit(surv, covariates)` with a right-censored `Surv` response and a covariate design
    (a 2-D array or a dataframe). An intercept is added automatically; rows with missing
    covariates are dropped. Results are exposed as arrays (`coef_`, `scale_`, `std_error_`,
    …) and a tidy frame via `to_dataframe`, and feed `greenwood.tidy`.

    Examples
    --------
    Build a `Surv` response from the bundled `lung` dataset and fit a Weibull AFT model with
    `age` and `sex` as covariates. Printing the fitted object reports the coefficients (on the
    log-time scale), the scale, and the log-likelihood.

    ```{python}
    import greenwood as gw
    from greenwood import Surv

    lung = gw.data.load_dataset("lung")
    y = Surv.right(lung["time"], event=(lung["status"] == 2))
    aft = gw.AFT("weibull").fit(y, lung[["age", "sex"]])
    aft
    ```

    The `aft` object fit here is reused by the method examples below.
    """

    def __init__(self, dist: str = "weibull", *, conf_level: float = 0.95) -> None:
        if dist not in _DISTS:
            raise ValueError(f"dist must be one of {sorted(_DISTS)}, got {dist!r}.")
        if not 0.0 < conf_level < 1.0:
            raise ValueError(f"conf_level must be in (0, 1), got {conf_level}.")
        self.dist = dist
        self.conf_level = conf_level

    def __repr__(self) -> str:
        if getattr(self, "coef_", None) is None:
            return f"AFT(dist={self.dist!r}, conf_level={self.conf_level}) <unfitted>"
        from ._repr import align_table, fixed, num

        rows = [
            [num(c), num(se), fixed(z, 3), num(p)]
            for c, se, z, p in zip(self.coef_, self.std_error_, self.z_, self.p_value_, strict=True)
        ]
        table = align_table(["coef", "se(coef)", "z", "p"], rows, list(self.term_names_))
        return "\n".join(
            [
                f"AFT (accelerated failure time model, dist={self.dist!r})",
                "",
                table,
                "",
                f"Scale = {num(self.scale_)}",
                f"n = {self.n_}, events = {self.n_event_}",
                f"Log-likelihood = {num(self.loglik_)}",
            ]
        )

    def fit(self, surv: Surv, covariates: Any, *, data: Any = None) -> AFT:
        """Fit the model to a right-censored `Surv` response and a covariate design.

        `covariates` is a dataframe or 2-D array, or a right-hand-side formula string (for
        example `"age + sex"`) evaluated against `data`. An intercept is added automatically.

        Examples
        --------
        The error distribution is chosen with `dist=`, one of `"weibull"`, `"exponential"`,
        `"lognormal"`, or `"loglogistic"`. Here is a log-normal fit to the same `y` response and
        `lung` data from the class example above:

        ```{python}
        gw.AFT(dist="lognormal").fit(y, lung[["age", "sex"]])
        ```
        """
        from ._surv import CensoringType

        if surv.type is not CensoringType.RIGHT:
            raise NotImplementedError(
                f"AFT currently supports right-censored responses, not {surv.type.value!r}."
            )

        design, cov_names = _design_matrix(covariates, data)
        if design.shape[0] != surv.n:
            raise ValueError("Covariates and response must have the same number of rows.")

        time = surv.stop
        event = surv.event
        keep = (~np.isnan(design).any(axis=1)) & (time > 0)
        design, time, event = design[keep], time[keep], event[keep]

        x = np.column_stack([np.ones(design.shape[0]), design])
        names = ["(Intercept)", *cov_names]
        log_t = np.log(time)
        n_coef = x.shape[1]
        has_scale = self.dist != "exponential"

        def neg_loglik(params: Array) -> float:
            beta = params[:n_coef]
            log_sigma = params[n_coef] if has_scale else 0.0
            sigma = np.exp(log_sigma)
            z = (log_t - x @ beta) / sigma
            log_f, log_s = _log_density_survival(self.dist, z)
            ll = event * (log_f - log_sigma - log_t) + (1.0 - event) * log_s
            return -float(ll.sum())

        x0 = np.zeros(n_coef + (1 if has_scale else 0))
        x0[0] = float(log_t.mean())
        result = minimize(neg_loglik, x0, method="BFGS", options={"gtol": 1e-8, "maxiter": 1000})
        params = result.x
        vcov = np.linalg.inv(_num_hessian(neg_loglik, params))

        self._x = x
        self.term_names_ = names
        self.coef_ = params[:n_coef]
        self.vcov_ = vcov
        self.std_error_ = np.sqrt(np.diag(vcov))[:n_coef]
        self.scale_ = float(np.exp(params[n_coef])) if has_scale else 1.0
        self.log_scale_se_ = float(np.sqrt(vcov[n_coef, n_coef])) if has_scale else None
        self.loglik_ = -float(result.fun)
        self.n_ = int(keep.sum())
        self.n_event_ = int(event.sum())

        z = float(norm.ppf(1.0 - (1.0 - self.conf_level) / 2.0))
        self.z_ = self.coef_ / self.std_error_
        self.p_value_ = 2.0 * norm.sf(np.abs(self.z_))
        self.conf_low_ = self.coef_ - z * self.std_error_
        self.conf_high_ = self.coef_ + z * self.std_error_
        return self

    def _design(self, newdata: Any) -> Array:
        """Build the intercept-prepended design matrix for `newdata` (or the training data)."""
        if newdata is None:
            return self._x
        design, _ = _design_matrix(newdata)
        return np.column_stack([np.ones(design.shape[0]), design])

    def predict(
        self,
        newdata: Any = None,
        *,
        type: str = "survival",
        times: Any = None,
        p: Any = 0.5,
        conditional_after: Any = None,
    ) -> Any:
        """Predict from the fitted AFT model.

        `type` is one of:

        - `"lp"`: the linear predictor `X . beta` (the fitted `log(T)` location).
        - `"quantile"`: predicted survival-time quantiles at the failure probabilities `p`
          (default the median). Matches R `predict(survreg, type="quantile")`.
        - `"survival"`: survival probabilities `S(t | x)` at each time in `times` (defaulting
          to a grid), returned as a frame with a `time` column and one column per subject.

        For `type="survival"`, `conditional_after` (a scalar or one value per subject)
        predicts conditional on having already survived to that time: the value at time `t` is
        `P(T > t | T > c) = S(t) / S(c)`, and is 1 for `t <= c`.

        Examples
        --------
        Predicted survival-time quantiles for the first two subjects, at the lower quartile,
        median, and upper quartile (reusing the `aft` fit above):

        ```{python}
        aft.predict(lung[["age", "sex"]][:2], type="quantile", p=[0.25, 0.5, 0.75])
        ```

        With `type="survival"`, read survival probabilities off the fitted curves at chosen
        times. Here are the estimates at 180 and 365 days for those same two subjects:

        ```{python}
        aft.predict(lung[["age", "sex"]][:2], type="survival", times=[180, 365])
        ```
        """
        x = self._design(newdata)
        mu = x @ self.coef_
        sigma = self.scale_

        if type == "lp":
            return mu
        if type == "quantile":
            import pandas as pd

            p_arr = np.atleast_1d(np.asarray(p, dtype=float))
            w = _error_quantile(self.dist, p_arr)
            quantiles = np.exp(mu[:, None] + sigma * w[None, :])  # (n_subjects, n_p)
            frame = pd.DataFrame(
                {f"subject_{i + 1}": quantiles[i] for i in range(quantiles.shape[0])}
            )
            frame.insert(0, "p", p_arr)
            return frame
        if type == "survival":
            import pandas as pd

            if times is None:
                w = _error_quantile(self.dist, np.linspace(0.01, 0.99, 50))
                query = np.unique(np.round(np.exp(mu.mean() + sigma * w), 6))
            else:
                query = np.atleast_1d(np.asarray(times, dtype=float))
            z = (np.log(query)[:, None] - mu[None, :]) / sigma  # (n_times, n_subjects)
            _, log_s = _log_density_survival(self.dist, z)
            if conditional_after is None:
                surv = np.exp(log_s)
            else:
                c = np.asarray(conditional_after, dtype=float)
                if c.ndim == 0:
                    c = np.full(mu.shape[0], float(c))
                if c.shape[0] != mu.shape[0]:
                    raise ValueError("conditional_after must be a scalar or one value per subject.")
                with np.errstate(divide="ignore"):
                    zc = (np.log(c) - mu) / sigma
                _, log_s_c = _log_density_survival(self.dist, zc)
                log_s_c = np.where(c > 0, log_s_c, 0.0)  # S(c) = 1 for c <= 0
                surv = np.exp(np.minimum(log_s - log_s_c[None, :], 0.0))  # ratio capped at 1
            frame = pd.DataFrame({f"subject_{i + 1}": surv[:, i] for i in range(surv.shape[1])})
            frame.insert(0, "time", query)
            return frame
        raise ValueError(f"Unknown predict type {type!r}; use 'lp', 'quantile', or 'survival'.")

    def to_dataframe(self) -> Any:
        """Return a tidy coefficient table (one row per term, including the intercept).

        Examples
        --------
        The coefficient table gives one row per term, including the intercept, with standard
        errors, test statistics, p-values, and confidence limits (reusing the `aft` fit above):

        ```{python}
        aft.to_dataframe()
        ```
        """
        import pandas as pd

        return pd.DataFrame(
            {
                "term": self.term_names_,
                "estimate": self.coef_,
                "std_error": self.std_error_,
                "statistic": self.z_,
                "p_value": self.p_value_,
                "conf_low": self.conf_low_,
                "conf_high": self.conf_high_,
            }
        )


def _tidy_aft(model: AFT, **_: Any) -> Any:
    return model.to_dataframe()


def _glance_aft(model: AFT, **_: Any) -> Any:
    import pandas as pd

    n_params = len(model.term_names_) + (0 if model.dist == "exponential" else 1)
    return pd.DataFrame(
        [
            {
                "dist": model.dist,
                "n": model.n_,
                "nevent": model.n_event_,
                "scale": model.scale_,
                "loglik": model.loglik_,
                "aic": -2.0 * model.loglik_ + 2.0 * n_params,
            }
        ]
    )


def _register_adapters() -> None:
    from .tidy import register_glance, register_tidier

    register_tidier("greenwood._parametric.AFT", _tidy_aft)
    register_glance("greenwood._parametric.AFT", _glance_aft)


_register_adapters()
