r"""Buckley-James rank-based accelerated failure time regression.

The Buckley-James estimator (Buckley & James, 1979) fits the same log-linear model as `AFT`,
$\log(T) = X\beta + \varepsilon$, but without assuming a parametric family for $\varepsilon$.
Instead it alternates two steps until $\beta$ stabilizes:

1. Given the current $\beta$, compute residuals $e_i = \log(T_i) - x_i^\top\beta$. For a censored
   observation, $e_i$ is itself right-censored, so it is replaced by its conditional mean given
   that the true residual exceeds it, using the Kaplan-Meier estimator of the residual
   distribution (self-consistent redistribution, the same principle behind Kaplan-Meier itself).
2. Refit $\beta$ by ordinary least squares of these imputed pseudo-responses on $X$.

This predates and is superseded in practice by the maximum-likelihood `AFT`, which is more
efficient when its distributional assumption holds and has a closed-form likelihood and standard
errors. Buckley-James trades that efficiency for not needing a distributional assumption at all,
at the cost of an iterative, sometimes slow-to-converge fit and no closed-form standard errors
(estimated here by bootstrap resampling when requested).
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt
from scipy.stats import norm

from ._backends import to_dataframe
from ._cox import _design_matrix

if TYPE_CHECKING:
    from ._surv import Surv

__all__ = ["BuckleyJames"]

Array = npt.NDArray[Any]


def _bj_pseudo_response(log_time: Array, event: Array, x: Array, beta: Array) -> Array:
    r"""One Buckley-James imputation step: pseudo-responses for the next OLS refit.

    Exact observations keep their observed $\log T_i$. For a censored observation, the residual
    $e_i = \log T_i - x_i^\top\beta$ is only known to exceed the observed value, so it is replaced
    by its conditional mean under the Kaplan-Meier estimate of the residual distribution:
    $e_i \to e_i + \mathrm{rmrl}(e_i, \tau)$, where $\tau$ is the largest residual. The largest
    residual is temporarily treated as uncensored when fitting that Kaplan-Meier curve (a standard
    Buckley-James tail correction), since otherwise the residual survival curve never reaches 0
    and the conditional mean beyond the last censored residual is undefined; residuals are
    shifted to be positive first since `Surv.right` requires non-negative times (the restricted
    mean residual life used here is invariant to that shift).
    """
    from ._nonparametric import KaplanMeier
    from ._surv import Surv

    resid = log_time - x @ beta
    shift = float(-resid.min() + 1.0)
    resid_shifted = resid + shift

    km_event = event.copy()
    idx_max = int(np.argmax(resid_shifted))
    km_event[idx_max] = True  # tail correction: ensure the residual KM curve reaches 0
    km = KaplanMeier().fit(Surv.right(resid_shifted, event=km_event))
    tau = float(resid_shifted[idx_max])

    y_star = log_time.copy()
    for i in np.nonzero(~event)[0]:
        s_i = float(resid_shifted[i])
        if s_i >= tau:
            continue  # the tail-corrected observation itself: no additional residual to add
        y_star[i] = log_time[i] + km.rmrl(s_i, tau)
    return y_star


def _fit_bj_core(
    log_time: Array, event: Array, x: Array, *, tol: float, max_iter: int
) -> tuple[Array, int, bool]:
    """The Buckley-James iteration: alternates imputation and OLS refitting until convergence.

    Buckley-James is well known to sometimes settle into a 2-cycle (alternating between two
    values) rather than a single fixed point; when that happens, the average of the two
    alternating estimates is reported as converged, a standard practical resolution.
    """
    beta, *_ = np.linalg.lstsq(x, log_time, rcond=None)
    history = [beta]
    converged = False
    n_iter = 0
    for it in range(1, max_iter + 1):
        n_iter = it
        y_star = _bj_pseudo_response(log_time, event, x, beta)
        beta_new, *_ = np.linalg.lstsq(x, y_star, rcond=None)
        if np.max(np.abs(beta_new - beta)) < tol:
            beta = beta_new
            converged = True
            break
        if it >= 3 and np.max(np.abs(beta_new - history[-2])) < tol:
            beta = 0.5 * (beta_new + history[-1])
            converged = True
            break
        beta = beta_new
        history.append(beta)
    return beta, n_iter, converged


class BuckleyJames:
    r"""Buckley-James rank-based accelerated failure time regression.

    Fits $\log(T) = X\beta + \varepsilon$ without assuming a distribution for $\varepsilon$, by
    alternating Kaplan-Meier-based imputation of censored log-times with ordinary least squares
    refitting until $\beta$ stabilizes. This is a semiparametric alternative to `AFT`: it avoids
    committing to a parametric error distribution, at the cost of an iterative fit, occasional
    slow or oscillating convergence, and no closed-form standard errors (available here only via
    optional bootstrap resampling, `n_boot=`).

    When there is no censoring at all, every residual is exact and the algorithm converges in a
    single step to the ordinary least squares fit of $\log(T)$ on $X$.

    Parameters
    ----------
    max_iter
        Maximum number of Buckley-James iterations (default `200`).
    tol
        Convergence tolerance on the largest change in any coefficient between iterations
        (default `1e-6`).
    n_boot
        Number of bootstrap resamples for standard errors and confidence intervals. If `None`
        (the default), `std_error_`, `conf_low_`, and `conf_high_` are `nan` (Buckley-James has
        no closed-form variance).
    seed
        Random seed for bootstrap resampling (ignored if `n_boot` is `None`).
    conf_level
        Confidence level for bootstrap coefficient intervals (default `0.95`).

    Returns
    -------
    Fitted estimator
        Call `fit()` to produce a fitted estimator with cached results (`coef_`, `std_error_`,
        `z_`, `p_value_`, `conf_low_`, `conf_high_`, `n_iter_`, `converged_`, `loglik_`-free since
        there is no likelihood), accessible as arrays or exported to DataFrames.

    Details
    -------
    Call `fit(surv, covariates)` with a right-censored `Surv` response and a covariate design (a
    2-D array, a dataframe, or a formula string with `data`). An intercept is added automatically.

    Examples
    --------
    ```{python}
    import greenwood as gw

    # Load data and build a right-censored response
    lung = gw.load_dataset("lung", backend="polars")
    y = gw.Surv.right(lung["time"], event=(lung["status"] == 2))

    # Fit a Buckley-James model
    bj = gw.BuckleyJames().fit(y, lung[["age", "sex"]])
    bj
    ```
    """

    def __init__(
        self,
        *,
        max_iter: int = 200,
        tol: float = 1e-6,
        n_boot: int | None = None,
        seed: int | None = None,
        conf_level: float = 0.95,
    ) -> None:
        if max_iter <= 0:
            raise ValueError(f"max_iter must be positive, got {max_iter}.")
        if tol <= 0.0:
            raise ValueError(f"tol must be positive, got {tol}.")
        if n_boot is not None and n_boot <= 0:
            raise ValueError(f"n_boot must be positive or None, got {n_boot}.")
        if not 0.0 < conf_level < 1.0:
            raise ValueError(f"conf_level must be in (0, 1), got {conf_level}.")
        self.max_iter = max_iter
        self.tol = tol
        self.n_boot = n_boot
        self.seed = seed
        self.conf_level = conf_level

    def __repr__(self) -> str:
        if getattr(self, "coef_", None) is None:
            return f"BuckleyJames(max_iter={self.max_iter}, tol={self.tol!r}) <unfitted>"
        from ._repr import align_table, fixed, num

        rows = [
            [num(c), num(se), fixed(z, 3), num(p)]
            for c, se, z, p in zip(self.coef_, self.std_error_, self.z_, self.p_value_, strict=True)
        ]
        table = align_table(["coef", "se(coef)", "z", "p"], rows, list(self.term_names_))
        return "\n".join(
            [
                "BuckleyJames (rank-based accelerated failure time regression)",
                "",
                table,
                "",
                f"n = {self.n_}, events = {self.n_event_}",
                f"iterations = {self.n_iter_}, converged = {self.converged_}",
            ]
        )

    def fit(self, surv: Surv, covariates: Any, *, data: Any = None) -> BuckleyJames:
        """Fit the Buckley-James model to survival data.

        Parameters
        ----------
        surv
            A right-censored `Surv` response. Built with `Surv.right()`.
        covariates
            A dataframe (pandas or polars), a 2-D array, or a formula string (e.g., `"age + sex"`)
            evaluated against the `data` argument.
        data
            A dataframe to evaluate the formula string (ignored if `covariates` is a dataframe or
            array).

        Returns
        -------
        BuckleyJames
            The fitted estimator itself (for method chaining) with cached coefficient arrays
            (`coef_`, `std_error_`, `z_`, `p_value_`) and convergence diagnostics (`n_iter_`,
            `converged_`).

        Examples
        --------
        ```{python}
        import greenwood as gw

        lung = gw.load_dataset("lung", backend="polars")
        y = gw.Surv.right(lung["time"], event=(lung["status"] == 2))

        bj = gw.BuckleyJames().fit(y, lung[["age", "sex"]])
        bj
        ```
        """
        from ._surv import CensoringType

        if surv.type is not CensoringType.RIGHT:
            raise NotImplementedError(
                f"BuckleyJames currently supports right-censored responses, not "
                f"{surv.type.value!r}."
            )

        design, cov_names = _design_matrix(covariates, data)
        if design.shape[0] != surv.n:
            raise ValueError("Covariates and response must have the same number of rows.")

        time = surv.stop
        event = surv.event
        keep = (~np.isnan(design).any(axis=1)) & (time > 0)
        design, time, event = design[keep], time[keep], event[keep]
        if not event.any():
            raise ValueError("No events remain after dropping missing rows.")

        x = np.column_stack([np.ones(design.shape[0]), design])
        names = ["(Intercept)", *cov_names]
        log_time = np.log(time)

        beta, n_iter, converged = _fit_bj_core(
            log_time, event, x, tol=self.tol, max_iter=self.max_iter
        )
        if not converged:
            warnings.warn(
                f"BuckleyJames did not converge within max_iter={self.max_iter} iterations "
                "(and did not settle into a detectable 2-cycle either); coef_ is the last "
                "iterate, not a stable fixed point.",
                stacklevel=2,
            )

        self._x = x
        self.term_names_ = names
        self.coef_ = beta
        self.n_ = int(keep.sum())
        self.n_event_ = int(event.sum())
        self.n_iter_ = n_iter
        self.converged_ = converged
        self._log_time = log_time
        self._event = event

        n_coef = x.shape[1]
        if self.n_boot is not None:
            rng = np.random.default_rng(self.seed)
            boot_coefs = []
            n = x.shape[0]
            for _ in range(self.n_boot):
                idx = rng.integers(0, n, size=n)
                if not event[idx].any():
                    continue
                b, _, _ = _fit_bj_core(
                    log_time[idx], event[idx], x[idx], tol=self.tol, max_iter=self.max_iter
                )
                boot_coefs.append(b)
            boot_arr = np.array(boot_coefs)
            self.std_error_ = np.std(boot_arr, axis=0, ddof=1)
            self._boot_coefs = boot_arr
        else:
            self.std_error_ = np.full(n_coef, np.nan)
            self._boot_coefs = None

        z = float(norm.ppf(1.0 - (1.0 - self.conf_level) / 2.0))
        self.z_ = self.coef_ / self.std_error_
        self.p_value_ = 2.0 * norm.sf(np.abs(self.z_))
        self.conf_low_ = self.coef_ - z * self.std_error_
        self.conf_high_ = self.coef_ + z * self.std_error_

        # Residual Kaplan-Meier curve at the final beta, kept for survival predictions.
        resid = log_time - x @ beta
        self._resid_shift = float(-resid.min() + 1.0)
        resid_shifted = resid + self._resid_shift
        km_event = event.copy()
        idx_max = int(np.argmax(resid_shifted))
        km_event[idx_max] = True
        from ._nonparametric import KaplanMeier
        from ._surv import Surv

        self._resid_km = KaplanMeier().fit(Surv.right(resid_shifted, event=km_event))
        return self

    def _design(self, newdata: Any) -> Array:
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
        format: str | None = None,
    ) -> Any:
        r"""Predict the linear predictor or survival probabilities from the fitted model.

        Parameters
        ----------
        newdata
            Covariate values for prediction. A DataFrame (Pandas or Polars), 2-D array, or `None`
            (the default, uses the training data).
        type
            `"survival"` (default): survival probabilities $S(t \mid x)$ at `times`, evaluated by
            shifting the fitted residual Kaplan-Meier curve by each subject's linear predictor.
            `"lp"`: the linear predictor $X\beta$ (log-time location) only; returns an array.
        times
            Query times for `type="survival"` (ignored for `"lp"`). Required for `"survival"`.
        format
            Output format for `type="survival"`: `None` (auto-detect), `"pandas"`, `"polars"`, or
            `"pyarrow"`. Ignored for `type="lp"` (which always returns an array).

        Returns
        -------
        ndarray or DataFrame
            An array of shape `(n_subjects,)` for `type="lp"`. A DataFrame with a `time` column
            and one column per subject for `type="survival"`.

        Details
        -------
        Because Buckley-James makes no distributional assumption, survival predictions reuse the
        empirical Kaplan-Meier curve of the fitted residuals rather than a closed-form survival
        function: $S(t \mid x) = \hat{S}_{\text{resid}}(\log t - x^\top\hat\beta)$. This curve is
        only as reliable as the residual sample size and is not smooth like `AFT`'s or
        `RoystonParmar`'s predictions.

        Examples
        --------
        ```{python}
        import greenwood as gw

        lung = gw.load_dataset("lung", backend="polars")
        y = gw.Surv.right(lung["time"], event=(lung["status"] == 2))
        bj = gw.BuckleyJames().fit(y, lung[["age", "sex"]])

        bj.predict(lung[["age", "sex"]][:2], type="survival", times=[180, 365, 730],
                   format="polars")
        ```
        """
        x = self._design(newdata)
        mu = x @ self.coef_
        if type == "lp":
            return mu
        if type != "survival":
            raise ValueError(f"Unknown predict type {type!r}; use 'survival' or 'lp'.")
        if times is None:
            raise ValueError("times is required for type='survival'.")
        query = np.atleast_1d(np.asarray(times, dtype=float))
        log_query = np.log(query)
        columns: dict[str, Array] = {"time": query}
        for i in range(x.shape[0]):
            shifted = log_query - mu[i] + self._resid_shift
            columns[f"subject_{i + 1}"] = self._resid_km.predict(shifted)
        return to_dataframe(columns, format=format)

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

        Exports one row per term, including the intercept. `std_error`, `p_value`, `conf_low`,
        and `conf_high` are `nan` unless the model was fit with `n_boot=`.

        Parameters
        ----------
        format
            Output format: `None` (default), `"pandas"`, `"polars"`, or `"pyarrow"`.

        Returns
        -------
        pandas.DataFrame, polars.DataFrame, or pyarrow.Table
            A tidy table with columns `term`, `estimate`, `std_error`, `statistic`, `p_value`,
            `conf_low`, and `conf_high`.

        Examples
        --------
        ```{python}
        import greenwood as gw

        lung = gw.load_dataset("lung", backend="polars")
        y = gw.Surv.right(lung["time"], event=(lung["status"] == 2))
        bj = gw.BuckleyJames(n_boot=200, seed=0).fit(y, lung[["age", "sex"]])
        bj.to_frame(format="polars")
        ```
        """
        return to_dataframe(self._coefficient_columns(), format=format)


def _tidy_bj(model: BuckleyJames, *, format: str | None = None, **_: Any) -> Any:
    return model.to_frame(format=format)


def _glance_bj(model: BuckleyJames, *, format: str | None = None, **_: Any) -> Any:
    return to_dataframe(
        {
            "n": [model.n_],
            "nevent": [model.n_event_],
            "n_iter": [model.n_iter_],
            "converged": [model.converged_],
            "n_boot": [model.n_boot if model.n_boot is not None else 0],
        },
        format=format,
    )


def _register_adapters() -> None:
    from .summaries import register_glance, register_tidier

    register_tidier("greenwood._buckley_james.BuckleyJames", _tidy_bj)
    register_glance("greenwood._buckley_james.BuckleyJames", _glance_bj)


_register_adapters()
