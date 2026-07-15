r"""Parametric accelerated failure time (AFT) models.

`AFT` fits the log-linear (accelerated failure time) model $\log(T) = X\beta + \sigma W$,
where the error $W$ follows a standard distribution chosen by `dist`:

- `"weibull"` and `"exponential"` (exponential fixes $\sigma = 1$): minimum extreme value,
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

from ._backends import to_dataframe
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
    r"""Standardized quantile of the error term `W` at probabilities `p`.

    `p` is the cumulative probability of failure, so the returned `w` satisfies
    $F_W(w) = p$; the corresponding time quantile is $\exp(\mu + \sigma w)$.
    """
    if dist in ("weibull", "exponential"):  # minimum extreme value
        return np.log(-np.log1p(-p))
    if dist == "lognormal":
        return norm.ppf(p)
    return logistic.ppf(p)  # loglogistic


def _mean_survival_aft(dist: str, mu: Array, sigma: float) -> Array:
    r"""Closed-form $E[T]$ for each subject under the AFT distribution.

    Returns `np.inf` for log-logistic when `sigma >= 1` (mean is undefined).
    """
    if dist in ("weibull", "exponential"):
        return np.exp(mu) * _sp_gamma(1.0 + sigma)
    if dist == "lognormal":
        return np.exp(mu + 0.5 * sigma**2)
    # loglogistic
    if sigma < 1.0:
        return np.exp(mu) * np.pi * sigma / np.sin(np.pi * sigma)
    return np.full_like(mu, np.inf)


def _tail_partial_moment(dist: str, mu: Array, sigma: float, t0: Array) -> Array:
    r"""Compute $\int_{t_0}^\infty S(t)\,dt$ for each subject.

    This is the *tail partial moment* starting at `t0`. When `t0 = 0` it
    equals `E[T]`, and therefore `_mean_survival_aft` is a special case.

    The key identities derived from this quantity are:

    - $E[T - t_0 \mid T > t_0] = \text{tail}(t_0) / S(t_0)$
    - $E[T \mid T > t_0] = t_0 + \text{tail}(t_0) / S(t_0)$
    - $E[\min(T, \tau)] = E[T] - \text{tail}(\tau)$  (when $E[T] < \infty$)

    Closed-form formulas by distribution:

    - **Weibull / Exponential**:
      $e^\mu \Gamma(1+\sigma) \cdot \bar{\Gamma}(\sigma,\, u_0)$
      where $u_0 = (t_0 / e^\mu)^{1/\sigma}$ and $\bar{\Gamma}$ is the
      regularized upper incomplete gamma function.
    - **Log-normal**:
      $e^{\mu+\sigma^2/2} \Phi(\sigma - z_0) - t_0 \Phi(-z_0)$
      where $z_0 = (\log t_0 - \mu)/\sigma$.
    - **Log-logistic** ($\sigma < 1$):
      $E[T] \cdot I_{S(t_0)}(1-\sigma,\, \sigma)$
      where $I_x$ is the regularized incomplete Beta function and
      $S(t_0) = \operatorname{logistic.sf}(z_0)$.
    - **Log-logistic** ($\sigma \ge 1$): numerical integration via
      `scipy.integrate.quad`.
    """
    t0 = np.broadcast_to(np.asarray(t0, dtype=float), mu.shape).copy()

    if dist in ("weibull", "exponential"):
        lam = np.exp(mu)
        u0 = (t0 / lam) ** (1.0 / sigma)
        return lam * _sp_gamma(1.0 + sigma) * _sp_gammaincc(sigma, u0)

    if dist == "lognormal":
        # Avoid log(0) = -inf warnings; handle t0 = 0 as a special case
        with np.errstate(divide="ignore", invalid="ignore"):
            log_t0 = np.log(np.where(t0 > 0.0, t0, 1.0))
        log_t0 = np.where(t0 > 0.0, log_t0, -np.inf)
        z0 = (log_t0 - mu) / sigma
        term1 = np.exp(mu + 0.5 * sigma**2) * norm.cdf(sigma - z0)
        term2 = np.where(t0 > 0.0, t0 * norm.sf(z0), 0.0)
        return term1 - term2

    # loglogistic
    with np.errstate(divide="ignore", invalid="ignore"):
        log_t0 = np.log(np.where(t0 > 0.0, t0, 1.0))
    log_t0 = np.where(t0 > 0.0, log_t0, -np.inf)
    z0 = (log_t0 - mu) / sigma
    s0 = logistic.sf(z0)  # S(t0) per subject
    if sigma < 1.0:
        e_t = np.exp(mu) * np.pi * sigma / np.sin(np.pi * sigma)
        return e_t * _sp_betainc(1.0 - sigma, sigma, s0)
    # sigma >= 1: no closed-form; per-subject numerical integration
    from scipy.integrate import quad

    out = np.empty_like(mu)
    for i in range(mu.shape[0]):
        mu_i = float(mu[i])
        t0_i = float(t0[i])

        def _sf(t: float, _mu: float = mu_i) -> float:
            z = (np.log(t) - _mu) / sigma
            return float(logistic.sf(z))

        out[i], _ = quad(_sf, t0_i, np.inf, limit=200)
    return out


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
    r"""Parametric accelerated failure time model.

    While the Cox proportional hazards model leaves the baseline hazard unspecified, AFT models
    assume a fully parametric distribution for survival times and model how covariates
    accelerate or decelerate the "clock" of failure. Specifically, $\log(T) = \mu + \beta^\top x
    + \sigma\varepsilon$, where $T$ is survival time, $\beta$ are log-time-scale coefficients,
    $\sigma$ is a scale parameter, and $\varepsilon$ follows a specified error distribution
    (e.g., extreme-value, logistic, normal). This means a unit increase in covariate $x$
    multiplies survival time by $\exp(\beta)$.

    AFT models are useful when you want explicit, interpretable survival time predictions or when
    the parametric assumptions are reasonable. Unlike Cox models, they require choosing a
    distributional family (Weibull, exponential, lognormal, or loglogistic). Call `fit()` with
    a right-censored `Surv` response and a design matrix. The model automatically adds an
    intercept and estimates coefficients (on the log-time scale), the scale parameter, and
    standard errors via maximum likelihood.

    The implementation uses numerical optimization (typically Newton-Raphson) to maximize the
    likelihood. Coefficients on the log-time scale can be exponentiated to obtain time-
    acceleration ratios: $\exp(\beta)$ is the multiplicative effect on median or mean survival.
    The model also supports prediction of survival probabilities and quantiles at future times
    given covariate values.

    Parameters
    ----------
    dist
        Error distribution: `"weibull"` (default), `"exponential"`, `"lognormal"`, or
        `"loglogistic"`.
    conf_level
        Confidence level for coefficient intervals (default is `0.95`).

    Returns
    -------
    Fitted estimator
        Call `fit()` to produce a fitted estimator with cached results (`coef_`,
        `scale_`, `std_error_`, `z_`, `p_value_`, `conf_low_`, `conf_high_`, `loglik_`,
        `aic_`, `bic_`), accessible as arrays or exported to DataFrames.

    Details
    -------
    Call `fit(surv, covariates)` with a right-censored `Surv` response and a covariate design
    (a 2-D array or a dataframe). An intercept is added automatically; rows with missing
    covariates are dropped. Results are exposed as arrays (`coef_`, `scale_`, `std_error_`,
    `z_`, `p_value_`) and as tidy frames via `to_frame()` (optionally `format=`) and
    `greenwood.tidy`.

    Examples
    --------
    Build a `Surv` response from the bundled `lung` dataset and fit a Weibull AFT model with
    `age` and `sex` as covariates. Printing the fitted object reports the coefficients (on the
    log-time scale), the scale, and the log-likelihood.

    ```{python}
    import greenwood as gw

    lung = gw.load_dataset("lung", backend="polars")
    y = gw.Surv.right(lung["time"], event=(lung["status"] == 2))
    aft = gw.AFT("weibull").fit(y, lung[["age", "sex"]])
    aft
    ```
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
        r"""Fit the accelerated failure time model to survival data.

        Fits a parametric accelerated failure time (AFT) model to a right-censored response
        and covariates. The AFT models the log-survival time as a linear regression on
        covariates plus a random error from a specified parametric distribution (Weibull,
        exponential, log-normal, or log-logistic). An intercept is added automatically.

        The AFT is a parametric alternative to Cox regression, providing a fully specified
        survival distribution at the cost of stronger distributional assumptions. Unlike Cox,
        AFT supports median survival predictions and is naturally interpreted on the
        log-time scale: a coefficient of 0.1 means the covariate multiplies survival time by
        $\exp(0.1)$. Results are stored in the fitted object as coefficient arrays and can be
        exported to DataFrames.

        Parameters
        ----------
        surv
            A right-censored `Surv` response. Built with `Surv.right()`. Interval-censored
            or other response types raise `NotImplementedError`.
        covariates
            A dataframe (pandas or polars), a 2-D array, or a formula string (e.g.,
            `"age + sex"`) evaluated against the `data` argument.
        data
            A dataframe to evaluate the formula string (ignored if `covariates` is a
            dataframe or array).

        Returns
        -------
        AFT
            The fitted estimator object itself (for method chaining) with cached coefficient
            arrays (`coef_`, `std_error_`, `z_`, `p_value_`), scale parameter (`scale_`),
            and log-likelihood (`loglik_`).

        Details
        -------
        The AFT model parameterizes log-survival time as $\log(T) = X\beta + \sigma\varepsilon$,
        where $X$ is the design matrix, $\beta$ are coefficients, $\sigma$ is a scale parameter,
        and $\varepsilon$ is an error term from the chosen distribution. The survival function
        is then $S(t \mid X) = P(T > t \mid X) = G((\log(t) - X\beta) / \sigma)$, where $G$ is
        the survival function of the error distribution.

        Estimation uses maximum likelihood via numerical optimization. Exponential and
        Weibull models are nested special cases; log-normal and log-logistic offer different
        tail behaviors.

        Examples
        --------
        Fit a log-normal AFT model on the bundled `lung` dataset with `age` and `sex` as
        covariates:

        ```{python}
        import greenwood as gw

        lung = gw.load_dataset("lung", backend="polars")
        y = gw.Surv.right(lung["time"], event=(lung["status"] == 2))
        aft = gw.AFT(dist="lognormal").fit(y, lung[["age", "sex"]])
        aft
        ```

        Use a formula string with the `data` argument:

        ```{python}
        aft_formula = gw.AFT(dist="weibull").fit(y, "age + sex", data=lung)
        aft_formula
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

    def _survival_pdf(self, z: Array) -> Array:
        """PDF of the standardized error distribution at z-values.

        Parameters
        ----------
        z
            Standardized values where z = (log(t) - mu) / sigma.
            Can be any shape.

        Returns
        -------
        Array
            PDF values, same shape as input.
        """
        if self.dist in ("weibull", "exponential"):  # minimum extreme value
            return np.exp(z - np.exp(z))
        if self.dist == "lognormal":
            return norm.pdf(z)
        # loglogistic
        return logistic.pdf(z)

    def _survival_se(self, x: Array, times: Array) -> Array:
        r"""Standard error of survival predictions via delta-method.

        Computes SE of $S(t|x)$ using the delta-method, propagating coefficient uncertainty
        through the survival function. The derivative of survival w.r.t. the linear predictor
        (mu = X @ beta) is:

        $$\frac{\partial S}{\partial \mu} = -f_\varepsilon(z) \cdot \frac{1}{\sigma}$$

        where f_varepsilon is the PDF of the error distribution and z = (log(t) - mu) / sigma.

        Parameters
        ----------
        x
            Design matrix (n_subjects, n_features). Typically includes intercept.
        times
            Query times (n_times,).

        Returns
        -------
        Array
            Standard errors of survival predictions, shape (n_times, n_subjects).
        """
        mu = x @ self.coef_  # (n_subjects,)
        sigma = self.scale_
        z = (np.log(times)[:, None] - mu[None, :]) / sigma  # (n_times, n_subjects)

        # Derivative of survival w.r.t. mu: dS/dmu = -f(z) / sigma
        pdf_z = self._survival_pdf(z)
        ds_dmu = -pdf_z / sigma  # (n_times, n_subjects)

        # SE(mu) for each subject: sqrt(diag(x @ vcov @ x.T))
        # vcov_ includes scale parameter; extract only the coefficient part
        n_coef = x.shape[1]
        vcov_coef = self.vcov_[:n_coef, :n_coef]
        se_mu_sq = np.diag(x @ vcov_coef @ x.T)  # (n_subjects,)
        se_mu = np.sqrt(np.clip(se_mu_sq, 0.0, None))

        # SE(S) = |dS/dmu| * SE(mu)
        se_s = np.abs(ds_dmu) * se_mu[None, :]  # (n_times, n_subjects)
        return se_s

    def predict(
        self,
        newdata: Any = None,
        *,
        type: str = "survival",
        times: Any = None,
        p: Any = 0.5,
        tau: Any = None,
        conditional_after: Any = None,
        ci: bool = False,
        conf_type: str = "log-log",
        format: str | None = None,
    ) -> Any:
        r"""Predict survival times, quantiles, or survival probabilities from the AFT model.

        Generates predictions from a fitted accelerated failure time model. The AFT is a fully
        parametric survival model, so predictions require specifying both the predictor values
        (via `newdata`) and the type of prediction desired. Pass `newdata=None` to predict
        for the training data (fitted subjects).

        Six prediction types are available:

        1. **Linear predictor** (`type="lp"`): the log-time location $X\beta$, showing how
           covariates shift the log-survival time distribution.

        2. **Quantile** (`type="quantile"`): predicted survival-time quantiles at specified
           failure probabilities (e.g., median survival when $p=0.5$). Useful for clinical
           summaries like "50% of subjects with these covariates survive to time X."

        3. **Survival** (`type="survival"`): survival probabilities $S(t \mid x)$ at specified
           times, returned as a DataFrame for easy visualization. Optionally condition on already
           having survived to a landmark time (`conditional_after`) for landmark-based
           predictions. With `ci=True`, confidence intervals are included.

        4. **Mean** (`type="mean"`): the expected survival time $E[T]$ (unconditional) or the
           conditional mean $E[T \mid T > t_0]$ when `conditional_after` is provided. Computed
           via closed-form formulas for all distributions. Returns an array of shape
           (n_subjects,).

        5. **Mean remaining** (`type="mean_remaining"`): expected remaining lifetime
           $E[T - t_0 \mid T > t_0]$ for subjects known to have survived past `t0`.
           Requires `conditional_after`. Returns an array of shape (n_subjects,).

        6. **RMST** (`type="rmst"`): restricted mean survival time $E[\min(T, \tau)]$ up to
           the restriction time `tau`. Requires the `tau` argument. Returns an array of
           shape (n_subjects,).

        Parameters
        ----------
        newdata
            Covariate values for prediction. A DataFrame (Pandas or Polars), 2-D array, or
            `None` (default). If `None`, uses the training data (design matrix used at fit time).
            Must have the same columns/features as the training data.
        type
            Prediction type (default `"survival"`):

            - `"lp"`: Linear predictor $X\beta$ (log-time location). Returns an array.
            - `"quantile"`: Survival-time quantiles at failure probabilities `p`. Returns a
              frame with `p` column and one column per subject.
            - `"survival"`: Survival probabilities $S(t \mid x)$ at times in `times`. Returns a
              frame with `time` column and one column per subject (one per query time, one
              per subject in newdata).

        times
            Query times for `type="survival"` (ignored for other types). An array-like of
            floats. If None (default), uses an automatic grid based on the fitted distribution
            (50 equally spaced times on the log scale, rounded).
        p
            Failure probabilities for `type="quantile"` (ignored for other types). Can be a
            scalar (e.g., `0.5` for median) or array-like. Default `0.5` (median). Must be in
            (0, 1).
        tau
            Restriction time for `type="rmst"`. Scalar float giving the upper limit of
            integration. Required when `type="rmst"`, ignored otherwise.
        conditional_after
            For `type="survival"`, optionally compute conditional survival
            $P(T > t \mid T > c) = S(t) / S(c)$. Also used with `type="mean"` and
            `type="mean_remaining"` to condition on surviving past a landmark time $t_0$.
            Scalar (same conditioning time for all subjects) or array-like (one per
            subject). Default `None` (unconditional).
        ci
            If `True` (survival only), include confidence intervals (`_lower` and `_upper`
            columns per subject). Default is `False`. Not supported with `conditional_after`.
        conf_type
            Confidence interval transform (used only if `ci=True` and `type="survival"`):

            - `"log-log"` (default): Log-log transform. Bounds respect the constraint that
              survival $S(t) \in (0, 1)$. Recommended.
            - `"plain"`: Wald bounds without transform. Simple but may produce invalid bounds
              (survival < 0 or > 1).

        format
            Output format for the returned frame (`type="quantile"` or `"survival"`): `None`
            (default), `"pandas"`, `"polars"`, or `"pyarrow"`. When `None`, a backend is
            auto-detected (Polars, then Pandas, then PyArrow). Ignored for `type="lp"`,
            `"mean"`, `"mean_remaining"`, and `"rmst"` (which always return arrays).

        Returns
        -------
        ndarray or DataFrame
            If `type="lp"`: an array of shape (n_subjects,) containing log-time locations.
            If `type="quantile"`: a DataFrame with columns `p` (failure probabilities) and
            `subject_1`, `subject_2`, etc. containing survival times at each p.
            If `type="survival"`: a DataFrame with columns `time` (query times) and
            `subject_1`, `subject_2`, etc. containing survival probabilities at each time,
            optionally with `_lower` and `_upper` columns for confidence intervals.
            If `type="mean"`: an array of shape (n_subjects,) with $E[T]$ or
            $E[T \mid T > t_0]$.
            If `type="mean_remaining"`: an array of shape (n_subjects,) with
            $E[T - t_0 \mid T > t_0]$.
            If `type="rmst"`: an array of shape (n_subjects,) with $E[\min(T, \tau)]$.

        Details
        -------
        The AFT model assumes $\log(T) = X\beta + \sigma\varepsilon$, where $\varepsilon$
        follows a parametric error distribution (Weibull, lognormal, etc.). Predictions are made
        by evaluating the CDF/survival function of this distribution at covariate-adjusted
        locations. All predictions respect the fitted distribution and scale parameter.

        Predictions assume the model is well-specified. For flexible models, consider
        parametric bootstrap to quantify uncertainty.

        Examples
        --------
        Fit a Weibull AFT model on the bundled `lung` dataset, then predict the linear
        predictor (log-time location) for the first two subjects:

        ```{python}
        import greenwood as gw

        lung = gw.load_dataset("lung", backend="polars")
        y = gw.Surv.right(lung["time"], event=(lung["status"] == 2))
        aft = gw.AFT("weibull").fit(y, lung[["age", "sex"]])

        aft.predict(lung[["age", "sex"]][:2], type="lp")
        ```

        Predicted survival-time quantiles for the first two subjects at the lower quartile,
        median, and upper quartile (a table, so pass `format=`):

        ```{python}
        aft.predict(lung[["age", "sex"]][:2], type="quantile", p=[0.25, 0.5, 0.75],
                    format="polars")
        ```

        Read survival probabilities off the fitted curves at chosen times. Here are the
        estimates at 180 and 365 days for those same two subjects:

        ```{python}
        aft.predict(lung[["age", "sex"]][:2], type="survival", times=[180, 365],
                    format="polars")
        ```

        Predict conditional survival given already having survived to 100 days:

        ```{python}
        aft.predict(lung[["age", "sex"]][:2], type="survival", times=[180, 365],
                    conditional_after=100, format="polars")
        ```

        Add confidence intervals with `ci=True`:

        ```{python}
        aft.predict(lung[["age", "sex"]][:2], type="survival", times=[180, 365],
                    ci=True, format="polars")
        ```
        """
        x = self._design(newdata)
        mu = x @ self.coef_
        sigma = self.scale_

        if type == "lp":
            return mu
        if type == "quantile":
            p_arr = np.atleast_1d(np.asarray(p, dtype=float))
            w = _error_quantile(self.dist, p_arr)
            quantiles = np.exp(mu[:, None] + sigma * w[None, :])  # (n_subjects, n_p)
            cols: dict[str, Any] = {"p": p_arr}
            cols.update({f"subject_{i + 1}": quantiles[i] for i in range(quantiles.shape[0])})
            return to_dataframe(cols, format=format)
        if type == "survival":
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
                if ci:
                    raise NotImplementedError(
                        "Confidence intervals are not supported with conditional_after."
                    )
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
            cols = {"time": query}
            if ci:
                se_s = self._survival_se(x, query)  # (n_times, n_subjects)
                z_val = float(norm.ppf(1.0 - (1.0 - self.conf_level) / 2.0))

                if conf_type == "log-log":
                    # Log-log transform: Y = log(-log(S))
                    # SE(Y) = SE(S) / |S * log(S)|
                    with np.errstate(divide="ignore", invalid="ignore"):
                        log_surv = np.log(surv)
                        se_logl = se_s / np.abs(surv * log_surv)  # (n_times, n_subjects)
                        se_logl = np.where(np.isfinite(se_logl), se_logl, 0.0)  # handle 0 or 1
                        logl = np.log(-log_surv)
                        logl = np.where(np.isfinite(logl), logl, 0.0)
                        logl_lower = logl - z_val * se_logl
                        logl_upper = logl + z_val * se_logl
                        # note: upper on log scale -> lower survival
                        surv_lower = np.exp(-np.exp(logl_upper))
                        surv_upper = np.exp(-np.exp(logl_lower))
                else:  # conf_type == "plain"
                    surv_lower = surv - z_val * se_s
                    surv_upper = surv + z_val * se_s

                for i in range(surv.shape[1]):
                    cols[f"subject_{i + 1}"] = surv[:, i]
                    cols[f"subject_{i + 1}_lower"] = surv_lower[:, i]
                    cols[f"subject_{i + 1}_upper"] = surv_upper[:, i]
            else:
                cols.update({f"subject_{i + 1}": surv[:, i] for i in range(surv.shape[1])})
            return to_dataframe(cols, format=format)
        raise ValueError(f"Unknown predict type {type!r}; use 'lp', 'quantile', or 'survival'.")

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

    def to_frame(self, *, format: str | None = None) -> Any:
        """Return the coefficient table as a DataFrame.

        Exports one row per term, including the intercept, with coefficient estimates,
        standard errors, Wald statistics, p-values, and confidence limits.

        Parameters
        ----------
        format
            Output format: `None` (default), `"pandas"`, `"polars"`, or `"pyarrow"`. When
            `None`, a backend is auto-detected (Polars, then Pandas, then PyArrow).

        Returns
        -------
        pandas.DataFrame, polars.DataFrame, or pyarrow.Table
            A tidy table with columns `term`, `estimate`, `std_error`, `statistic`,
            `p_value`, `conf_low`, and `conf_high`.

        Raises
        ------
        ImportError
            If the requested (or, when auto-detecting, any) DataFrame library is not
            installed.

        Examples
        --------
        Fit a Weibull AFT model on the bundled `lung` dataset, then export its coefficient
        table as a Polars frame:

        ```{python}
        import greenwood as gw

        lung = gw.load_dataset("lung", backend="polars")
        y = gw.Surv.right(lung["time"], event=(lung["status"] == 2))
        aft = gw.AFT("weibull").fit(y, lung[["age", "sex"]])

        aft.to_frame(format="polars")
        ```

        Request a different backend with `format=`:

        ```{python}
        aft.to_frame(format="pandas")
        ```
        """
        return to_dataframe(self._coefficient_columns(), format=format)


def _tidy_aft(model: AFT, *, format: str | None = None, **_: Any) -> Any:
    return model.to_frame(format=format)


def _glance_aft(model: AFT, *, format: str | None = None, **_: Any) -> Any:
    n_params = len(model.term_names_) + (0 if model.dist == "exponential" else 1)
    return to_dataframe(
        {
            "dist": [model.dist],
            "n": [model.n_],
            "nevent": [model.n_event_],
            "scale": [model.scale_],
            "loglik": [model.loglik_],
            "aic": [-2.0 * model.loglik_ + 2.0 * n_params],
        },
        format=format,
    )


def _register_adapters() -> None:
    from .summaries import register_glance, register_tidier

    register_tidier("greenwood._parametric.AFT", _tidy_aft)
    register_glance("greenwood._parametric.AFT", _glance_aft)


_register_adapters()
