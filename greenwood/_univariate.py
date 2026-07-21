r"""Univariate parametric survival distributions.

`Parametric` fits a parametric survival distribution to right-censored data by maximum
likelihood, without covariates. This is the one-sample analogue of `AFT`: instead of
modelling how covariates shift the survival-time distribution, `Parametric` estimates the
distribution parameters themselves (e.g., Weibull shape and scale).

Four families are supported:

- `"weibull"`: shape `k` and scale `\lambda`; $S(t) = \exp\!\bigl(-(t/\lambda)^k\bigr)$.
- `"exponential"`: rate `\lambda`; constant hazard $h(t) = \lambda$.
- `"lognormal"`: location `\mu` and scale `\sigma` of $\log T$.
- `"loglogistic"`: scale `\alpha` and shape `\beta`; $S(t) = 1/(1+(t/\alpha)^\beta)$.

Internally every family is fitted in the AFT location-scale parameterisation
($\log T = \mu + \sigma\varepsilon$) and results are reported in the natural
parameterisation of each distribution.

The standalone helper `compare_distributions()` fits all four families to the same data and
returns an AIC/BIC comparison table for model selection.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt
from scipy.optimize import minimize
from scipy.stats import norm

from ._backends import to_dataframe
from ._parametric import (
    _DISTS,
    _error_quantile,
    _log_density_survival,
    _mean_survival_aft,
    _num_hessian,
)

if TYPE_CHECKING:
    from ._surv import Surv

__all__ = ["Parametric", "compare_distributions"]

Array = npt.NDArray[Any]

# Distribution parameter names in display order.
_PARAM_NAMES: dict[str, list[str]] = {
    "weibull": ["shape", "scale"],
    "exponential": ["rate"],
    "lognormal": ["mu", "sigma"],
    "loglogistic": ["alpha", "beta"],
}


class Parametric:
    r"""Univariate parametric survival distribution.

    Fits a parametric survival distribution to right-censored data by maximum likelihood, without
    covariates. The result is a fully specified distribution that provides survival, hazard,
    density, quantile, and mean-life predictions.

    Use this when you want to explore which distributional family best describes your data (Weibull
    vs. log-normal vs. log-logistic) before moving to regression modelling, or when the scientific
    question is simply "what is the median survival time and its uncertainty?"

    Parameters
    ----------
    dist
        Distribution family: `"weibull"` (default), `"exponential"`, `"lognormal"`, or
        `"loglogistic"`.
    conf_level
        Confidence level for parameter intervals (default `0.95`).

    Returns
    -------
    Parametric
        The fitted estimator (after calling `fit()`), with attributes `params_`, `std_error_`,
        `conf_low_`, `conf_high_`, `loglik_`, `aic_`, `bic_`, `n_`, and `n_event_`.

    Details
    -------
    Internally every family is fitted in the AFT location–scale parameterisation
    ($\log T = \mu + \sigma\varepsilon$), and results are reported back in the natural
    parameterisation of each distribution (e.g., Weibull shape and scale). Standard errors
    are obtained via the delta method applied to the observed information matrix.

    Model selection between families is straightforward: lower AIC (or BIC) indicates a
    better fit. Use `compare_distributions()` to rank all four families at once.

    Examples
    --------
    Fit a Weibull distribution to the lung cancer dataset and inspect the parameter
    estimates:

    ```{python}
    import greenwood as gw

    lung = gw.load_dataset("lung", backend="polars")
    y = gw.Surv.right(lung["time"], event=(lung["status"] == 2))
    fit = gw.Parametric("weibull").fit(y)
    fit
    ```

    To quickly compare all four distributions and pick the best-fitting family, use the
    companion function `compare_distributions()`:

    ```{python}
    gw.compare_distributions(y, format="polars")
    ```
    """

    def __init__(self, dist: str = "weibull", *, conf_level: float = 0.95) -> None:
        if dist not in _DISTS:
            raise ValueError(f"dist must be one of {sorted(_DISTS)}, got {dist!r}.")
        if not 0.0 < conf_level < 1.0:
            raise ValueError(f"conf_level must be in (0, 1), got {conf_level}.")
        self.dist = dist
        self.conf_level = conf_level

    # -- repr ----------------------------------------------------------------

    def __repr__(self) -> str:
        if not hasattr(self, "params_"):
            return f"Parametric(dist={self.dist!r}, conf_level={self.conf_level}) <unfitted>"
        from ._repr import align_table, num

        names = list(self.params_.keys())
        rows = [[num(self.params_[n]), num(self.std_error_[n])] for n in names]
        table = align_table(["estimate", "std_error"], rows, names)
        return "\n".join(
            [
                f"Parametric ({self.dist} distribution)",
                "",
                table,
                "",
                f"n = {self.n_}, events = {self.n_event_}",
                f"Log-likelihood = {num(self.loglik_)}",
                f"AIC = {num(self.aic_)}, BIC = {num(self.bic_)}",
            ]
        )

    # -- fit -----------------------------------------------------------------

    def fit(self, surv: Surv) -> Parametric:
        r"""Fit the distribution to right-censored survival data by maximum likelihood.

        Maximises the likelihood of the parametric model $\log T = \mu + \sigma\varepsilon$
        (intercept-only AFT) over the observed and censored times. The result is stored on the
        fitted object and reported in the natural parameterisation of each distribution.

        Parameters
        ----------
        surv
            A right-censored `Surv` response built with `Surv.right()`.

        Returns
        -------
        Parametric
            The fitted object (for method chaining), with attributes `params_`, `std_error_`,
            `loglik_`, `aic_`, `bic_`, etc.

        Examples
        --------
        Fit a log-normal distribution and view the estimated parameters with their standard
        errors:

        ```{python}
        import greenwood as gw

        lung = gw.load_dataset("lung", backend="polars")
        y = gw.Surv.right(lung["time"], event=(lung["status"] == 2))
        gw.Parametric("lognormal").fit(y)
        ```
        """
        from ._surv import CensoringType

        if surv.type is not CensoringType.RIGHT:
            raise NotImplementedError(
                f"Parametric currently supports right-censored responses, not {surv.type.value!r}."
            )

        time = surv.stop
        event = surv.event
        keep = time > 0
        time, event = time[keep], event[keep]
        log_t = np.log(time)

        has_scale = self.dist != "exponential"

        def neg_loglik(params: Array) -> float:
            mu = params[0]
            log_sigma = params[1] if has_scale else 0.0
            sigma = np.exp(log_sigma)
            z = (log_t - mu) / sigma
            log_f, log_s = _log_density_survival(self.dist, z)
            ll = event * (log_f - log_sigma - log_t) + (1.0 - event) * log_s
            return -float(ll.sum())

        x0 = np.array([float(log_t.mean())] + ([0.0] if has_scale else []))
        result = minimize(neg_loglik, x0, method="BFGS", options={"gtol": 1e-8, "maxiter": 1000})
        params = result.x

        # Variance via numerical Hessian of the negative log-likelihood.
        vcov_raw = np.linalg.inv(_num_hessian(neg_loglik, params))

        # Store the AFT location-scale parameterisation.
        self._mu = float(params[0])
        self._log_sigma = float(params[1]) if has_scale else 0.0
        self._sigma = float(np.exp(self._log_sigma))
        self._vcov_raw = vcov_raw  # on (mu, log_sigma) scale
        self.loglik_ = -float(result.fun)
        self.n_ = int(keep.sum())
        self.n_event_ = int(event.sum())

        # Natural parameters, SEs, and CIs.
        self._compute_natural_params()

        # AIC / BIC.
        n_params = 2 if has_scale else 1
        self.aic_ = -2.0 * self.loglik_ + 2.0 * n_params
        self.bic_ = -2.0 * self.loglik_ + np.log(self.n_) * n_params

        return self

    def _compute_natural_params(self) -> None:
        """Derive natural distribution parameters and SEs from the AFT fit."""
        mu = self._mu
        sigma = self._sigma
        vcov = self._vcov_raw
        z_val = float(norm.ppf(1.0 - (1.0 - self.conf_level) / 2.0))

        if self.dist == "weibull":
            shape = 1.0 / sigma
            scale = np.exp(mu)
            # Delta method SEs: shape = exp(-log_sigma), scale = exp(mu).
            se_shape = shape * np.sqrt(vcov[1, 1])
            se_scale = scale * np.sqrt(vcov[0, 0])
            self.params_ = {"shape": shape, "scale": scale}
            self.std_error_ = {"shape": se_shape, "scale": se_scale}
            self.conf_low_ = {"shape": shape - z_val * se_shape, "scale": scale - z_val * se_scale}
            self.conf_high_ = {
                "shape": shape + z_val * se_shape,
                "scale": scale + z_val * se_scale,
            }

        elif self.dist == "exponential":
            rate = np.exp(-mu)
            se_rate = rate * np.sqrt(vcov[0, 0])
            self.params_ = {"rate": rate}
            self.std_error_ = {"rate": se_rate}
            self.conf_low_ = {"rate": rate - z_val * se_rate}
            self.conf_high_ = {"rate": rate + z_val * se_rate}

        elif self.dist == "lognormal":
            se_mu = np.sqrt(vcov[0, 0])
            se_sigma = sigma * np.sqrt(vcov[1, 1])
            self.params_ = {"mu": mu, "sigma": sigma}
            self.std_error_ = {"mu": se_mu, "sigma": se_sigma}
            self.conf_low_ = {"mu": mu - z_val * se_mu, "sigma": sigma - z_val * se_sigma}
            self.conf_high_ = {"mu": mu + z_val * se_mu, "sigma": sigma + z_val * se_sigma}

        else:  # loglogistic
            alpha = np.exp(mu)
            beta = 1.0 / sigma
            se_alpha = alpha * np.sqrt(vcov[0, 0])
            se_beta = beta * np.sqrt(vcov[1, 1])
            self.params_ = {"alpha": alpha, "beta": beta}
            self.std_error_ = {"alpha": se_alpha, "beta": se_beta}
            self.conf_low_ = {
                "alpha": alpha - z_val * se_alpha,
                "beta": beta - z_val * se_beta,
            }
            self.conf_high_ = {
                "alpha": alpha + z_val * se_alpha,
                "beta": beta + z_val * se_beta,
            }

        self.vcov_ = vcov

    # -- prediction methods --------------------------------------------------

    def survival(self, times: Any) -> Array:
        r"""Survival function $S(t) = P(T > t)$ at the given times.

        Returns the probability that a subject survives beyond each requested time under the
        fitted parametric distribution. Unlike the non-parametric Kaplan–Meier estimate, this
        gives a smooth curve that can be evaluated at any time point — including beyond the
        last observed event.

        Parameters
        ----------
        times
            Query times (array-like of positive floats).

        Returns
        -------
        ndarray
            Survival probabilities, same length as `times`.

        Details
        -------
        The survival probability is computed from the fitted location–scale model as
        $S(t) = 1 - F_\varepsilon\!\bigl((\log t - \mu)/\sigma\bigr)$, where
        $F_\varepsilon$ is the CDF of the standardised error distribution.

        Examples
        --------
        Evaluate the fitted Weibull survival function at a few clinically relevant time
        points (100, 200, 365, and 500 days):

        ```{python}
        import greenwood as gw

        lung = gw.load_dataset("lung", backend="polars")
        y = gw.Surv.right(lung["time"], event=(lung["status"] == 2))
        fit = gw.Parametric("weibull").fit(y)
        fit.survival([100, 200, 365, 500])
        ```
        """
        t = np.atleast_1d(np.asarray(times, dtype=float))
        z = (np.log(t) - self._mu) / self._sigma
        _, log_s = _log_density_survival(self.dist, z)
        return np.exp(log_s)

    def cumulative_hazard(self, times: Any) -> Array:
        r"""Cumulative hazard function $H(t) = -\log S(t)$ at the given times.

        The cumulative hazard summarises the total accumulated risk up to time $t$. It
        increases monotonically from zero and is unbounded. A steeper rise indicates higher
        event rates over that interval.

        Parameters
        ----------
        times
            Query times (array-like of positive floats).

        Returns
        -------
        ndarray
            Cumulative hazard values, same length as `times`.

        Details
        -------
        Computed as the negative log of the survival function: $H(t) = -\log S(t)$. For
        a Weibull distribution this simplifies to $H(t) = (t/\lambda)^k$.

        Examples
        --------
        Compute the cumulative hazard at several time points under a fitted Weibull model:

        ```{python}
        import greenwood as gw

        lung = gw.load_dataset("lung", backend="polars")
        y = gw.Surv.right(lung["time"], event=(lung["status"] == 2))
        fit = gw.Parametric("weibull").fit(y)
        fit.cumulative_hazard([100, 200, 365, 500])
        ```
        """
        return -np.log(self.survival(times))

    def hazard(self, times: Any) -> Array:
        r"""Hazard function $h(t) = f(t) / S(t)$ at the given times.

        The hazard (or hazard rate) gives the instantaneous risk of the event at time $t$,
        conditional on survival up to that time. Its shape reveals whether risk is increasing,
        decreasing, or constant over time — a signature feature of each distribution family.

        Parameters
        ----------
        times
            Query times (array-like of positive floats).

        Returns
        -------
        ndarray
            Hazard rate values, same length as `times=`.

        Examples
        --------
        ```{python}
        import greenwood as gw

        lung = gw.load_dataset("lung", backend="polars")
        y = gw.Surv.right(lung["time"], event=(lung["status"] == 2))
        fit = gw.Parametric("weibull").fit(y)
        fit.hazard([100, 200, 365, 500])
        ```
        """
        t = np.atleast_1d(np.asarray(times, dtype=float))
        z = (np.log(t) - self._mu) / self._sigma
        log_f, log_s = _log_density_survival(self.dist, z)
        # f_T(t) = f_eps(z) / (sigma * t), h(t) = f_T(t) / S(t)
        log_hazard = log_f - log_s - np.log(self._sigma) - np.log(t)
        return np.exp(log_hazard)

    def density(self, times: Any) -> Array:
        r"""Probability density function $f(t)$ at the given times.

        Parameters
        ----------
        times
            Query times (array-like of positive floats).

        Returns
        -------
        ndarray
            Density values, same length as `times=`.
        """
        t = np.atleast_1d(np.asarray(times, dtype=float))
        z = (np.log(t) - self._mu) / self._sigma
        log_f, _ = _log_density_survival(self.dist, z)
        # f_T(t) = f_eps(z) / (sigma * t)
        return np.exp(log_f - np.log(self._sigma) - np.log(t))

    def quantile(self, p: Any) -> Array:
        r"""Quantile function: survival-time quantiles at failure probabilities `p`.

        The quantile $t_p$ satisfies $P(T \le t_p) = p$, so $p = 0.5$ gives the median survival
        time.

        Parameters
        ----------
        p
            Failure probabilities in `(0, 1)`. Scalar or array-like.

        Returns
        -------
        ndarray
            Survival times corresponding to each probability.

        Examples
        --------
        ```{python}
        import greenwood as gw

        lung = gw.load_dataset("lung", backend="polars")
        y = gw.Surv.right(lung["time"], event=(lung["status"] == 2))
        fit = gw.Parametric("weibull").fit(y)
        fit.quantile([0.25, 0.5, 0.75])
        ```
        """
        p_arr = np.atleast_1d(np.asarray(p, dtype=float))
        w = _error_quantile(self.dist, p_arr)
        return np.exp(self._mu + self._sigma * w)

    def mean(self) -> float:
        r"""Expected survival time $E[T]$.

        Returns `inf` for log-logistic when $\beta \le 1$ (mean undefined).

        Returns
        -------
        float
            The mean survival time under the fitted distribution.
        """
        mu_arr = np.array([self._mu])
        return float(_mean_survival_aft(self.dist, mu_arr, self._sigma)[0])

    def median(self) -> float:
        """Median survival time (the 50th-percentile time).

        Returns
        -------
        float
            The time at which $S(t) = 0.5$.
        """
        return float(self.quantile(0.5)[0])

    # -- export --------------------------------------------------------------

    def to_frame(self, *, format: str | None = None) -> Any:
        """Return the parameter estimates as a tidy DataFrame.

        One row per parameter, with columns `param`, `estimate`, `std_error`, `conf_low`, and
        `conf_high`.

        Parameters
        ----------
        format
            Output format: `None` (auto-detect), `"pandas"`, `"polars"`, or `"pyarrow"`.

        Returns
        -------
        DataFrame
            A tidy parameter table.

        Examples
        --------
        ```{python}
        import greenwood as gw

        lung = gw.load_dataset("lung", backend="polars")
        y = gw.Surv.right(lung["time"], event=(lung["status"] == 2))
        fit = gw.Parametric("weibull").fit(y)
        fit.to_frame(format="polars")
        ```
        """
        names = list(self.params_.keys())
        return to_dataframe(
            {
                "param": names,
                "estimate": [self.params_[n] for n in names],
                "std_error": [self.std_error_[n] for n in names],
                "conf_low": [self.conf_low_[n] for n in names],
                "conf_high": [self.conf_high_[n] for n in names],
            },
            format=format,
        )


# ---------------------------------------------------------------------------
# Model comparison
# ---------------------------------------------------------------------------


def compare_distributions(
    surv: Any,
    *,
    dists: list[str] | None = None,
    format: str | None = None,
) -> Any:
    """Fit multiple parametric distributions and return an AIC/BIC comparison table.

    This is the primary model-selection helper for univariate parametric survival analysis. It fits
    each distribution by maximum likelihood, then ranks them by AIC (lower is better).

    Parameters
    ----------
    surv
        A right-censored `Surv` response.
    dists
        Distribution families to compare. The default is all four:
        `["weibull", "exponential", "lognormal", "loglogistic"]`.
    format
        Output format: `None` (auto-detect), `"pandas"`, `"polars"`, or `"pyarrow"`.

    Returns
    -------
    DataFrame
        One row per distribution, sorted by AIC, with columns `dist`, `n_params`, `loglik`, `aic`,
        and `bic`.

    Examples
    --------
    ```{python}
    import greenwood as gw

    lung = gw.load_dataset("lung", backend="polars")
    y = gw.Surv.right(lung["time"], event=(lung["status"] == 2))
    gw.compare_distributions(y, format="polars")
    ```
    """
    if dists is None:
        dists = ["weibull", "exponential", "lognormal", "loglogistic"]

    fits = [Parametric(d).fit(surv) for d in dists]

    rows_dist: list[str] = []
    rows_npar: list[int] = []
    rows_ll: list[float] = []
    rows_aic: list[float] = []
    rows_bic: list[float] = []
    for fit in fits:
        rows_dist.append(fit.dist)
        rows_npar.append(len(fit.params_))
        rows_ll.append(fit.loglik_)
        rows_aic.append(fit.aic_)
        rows_bic.append(fit.bic_)

    # Sort by AIC (best first).
    order = sorted(range(len(rows_aic)), key=lambda i: rows_aic[i])

    return to_dataframe(
        {
            "dist": [rows_dist[i] for i in order],
            "n_params": [rows_npar[i] for i in order],
            "loglik": [rows_ll[i] for i in order],
            "aic": [rows_aic[i] for i in order],
            "bic": [rows_bic[i] for i in order],
        },
        format=format,
    )


# ---------------------------------------------------------------------------
# Tidy / glance adapters
# ---------------------------------------------------------------------------


def _tidy_parametric(model: Parametric, *, format: str | None = None, **_: Any) -> Any:
    return model.to_frame(format=format)


def _glance_parametric(model: Parametric, *, format: str | None = None, **_: Any) -> Any:
    return to_dataframe(
        {
            "dist": [model.dist],
            "n": [model.n_],
            "nevent": [model.n_event_],
            "loglik": [model.loglik_],
            "aic": [model.aic_],
            "bic": [model.bic_],
        },
        format=format,
    )


def _register_adapters() -> None:
    from .summaries import register_glance, register_tidier

    register_tidier("greenwood._univariate.Parametric", _tidy_parametric)
    register_glance("greenwood._univariate.Parametric", _glance_parametric)


_register_adapters()
