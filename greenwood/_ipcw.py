r"""IPC-weighted survival utilities and estimators.

`CensoringDistribution` exposes the Kaplan-Meier estimate of the censoring distribution as a
first-class utility, providing inverse-probability-of-censoring (IPC) weights that correct for
censoring bias. These weights are used internally by `brier_score`, `concordance_index_ipcw`, and
other IPCW metrics, but are now available for manual reweighting in custom analyses.

`IPCRidge` is an IPC-weighted ridge regression for survival data. Instead of optimizing the Cox
partial likelihood, it fits a weighted least-squares objective on (possibly transformed survival
times, where censored observations are downweighted by their inverse probability of being censored.
This corrects for censoring bias without assuming proportional hazards.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt

from ._backends import to_dataframe

if TYPE_CHECKING:
    from ._surv import Surv

__all__ = ["CensoringDistribution", "IPCRidge"]

Array = npt.NDArray[Any]


def _censoring_km_standard(time: Array, event: Array) -> tuple[Array, Array]:
    """Standard Kaplan-Meier for the censoring distribution, matching R's survfit.

    Treats true events as censoring and censoring as the event of interest. At tied times, censoring
    events are processed before true events leave the risk set, matching R's default
    `survfit(Surv(time, 1 - event) ~ 1)` convention.
    """
    cens_indicator = ~event.astype(bool)
    unique_times = np.unique(time[cens_indicator])
    surv_val = 1.0
    drop_times: list[float] = []
    drop_surv: list[float] = []
    for t in unique_times:
        n_risk = float((time >= t).sum())
        d = float((cens_indicator & (time == t)).sum())
        if n_risk > 0:
            surv_val *= 1.0 - d / n_risk
        drop_times.append(float(t))
        drop_surv.append(surv_val)
    return np.array(drop_times), np.array(drop_surv)


class CensoringDistribution:
    r"""Kaplan-Meier estimate of the censoring distribution, providing IPC weights.

    The censoring distribution $\hat{G}(t) = P(\text{not censored by time } t)$ is the Kaplan-Meier
    estimator applied to the censoring process (where censoring is treated as the "event" and true
    events are treated as censoring). This is the foundation of all IPCW methods.

    IPC weights $w_i = 1 / \hat{G}(t_i^-)$ reweight observed subjects to represent the full
    population, correcting for the information lost to censoring. Subjects who experienced an event
    at a time when censoring was heavy receive larger weights (they represent more unobserved
    subjects), while subjects at times with little censoring receive weights near 1.

    Parameters
    ----------
    surv
        A right-censored `Surv` response. The censoring distribution is estimated from the observed
        censoring pattern.

    Examples
    --------
    Estimate the censoring distribution for the lung dataset:

    ```{python}
    import greenwood as gw

    lung = gw.load_dataset("lung", backend="polars")
    y = gw.Surv.right(lung["time"], event=(lung["status"] == 2))

    # Fit the censoring distribution
    cens = gw.CensoringDistribution(y)
    cens
    ```

    Retrieve IPC weights for all subjects:

    ```{python}
    # IPC weights: larger where censoring is heavier
    weights = cens.weights()
    weights[:10]
    ```
    """

    def __init__(self, surv: Surv) -> None:
        self._times, self._surv = _censoring_km_standard(surv.stop, surv.event)
        self._surv_obj = surv

    def __repr__(self) -> str:
        n_cens = int((~self._surv_obj.event.astype(bool)).sum())
        n = self._surv_obj.n
        tail = f"{self._surv[-1]:.4f}" if self._surv.shape[0] > 0 else "1.0000"
        return (
            f"CensoringDistribution (n={n}, censored={n_cens})\n"
            f"  {self._times.shape[0]} unique censoring times, "
            f"tail G = {tail}"
        )

    def survival(self, times: Any) -> Array:
        r"""Evaluate the censoring survival function $\hat{G}(t)$ at arbitrary times.

        Returns the probability of not being censored by each requested time, estimated via the
        Kaplan-Meier estimator of the censoring distribution.

        Parameters
        ----------
        times
            1-D array-like of query times.

        Returns
        -------
        ndarray
            $\hat{G}(t)$ at each query time, shape `(len(times),)`.
        """
        t = np.atleast_1d(np.asarray(times, dtype=float))
        if self._times.shape[0] == 0:
            return np.ones(t.shape[0])
        idx = np.searchsorted(self._times, t, side="right") - 1
        return np.where(idx >= 0, self._surv[idx.clip(min=0)], 1.0)

    def survival_left(self, times: Any) -> Array:
        r"""Evaluate $\hat{G}(t^-)$, the left-continuous censoring survival.

        This is the censoring survival just before each time point, used for IPC weight
        construction. At the first observation time, $\hat{G}(t^-) = 1$.

        Parameters
        ----------
        times
            1-D array-like of query times.

        Returns
        -------
        ndarray
            $\hat{G}(t^-)$ at each query time, shape `(len(times),)`.
        """
        t = np.atleast_1d(np.asarray(times, dtype=float))
        if self._times.shape[0] == 0:
            return np.ones(t.shape[0])
        idx = np.searchsorted(self._times, t, side="left") - 1
        return np.where(idx >= 0, self._surv[idx.clip(min=0)], 1.0)

    def weights(self, *, tau: float | None = None) -> Array:
        r"""IPC weights $w_i = \delta_i / \hat{G}(t_i^-)$ for each subject.

        Uncensored subjects receive weight $1 / \hat{G}(t_i^-)$. Censored subjects receive weight 0
        (they are excluded from the reweighted analysis). Subjects with events after `tau` also
        receive weight 0.

        Parameters
        ----------
        tau
            Truncation time. Subjects with event times after `tau` receive zero weight. Defaults to
            the maximum observed time.

        Returns
        -------
        ndarray
            IPC weights, shape `(n,)`.
        """
        surv = self._surv_obj
        event = surv.event.astype(bool)
        times = surv.stop

        g_left = self.survival_left(times)

        with np.errstate(divide="ignore", invalid="ignore"):
            w = np.where((event) & (g_left > 0), 1.0 / g_left, 0.0)

        if tau is not None:
            w = np.where(times <= tau, w, 0.0)

        return w

    def to_frame(self, *, format: str | None = None) -> Any:
        """Return the censoring survival curve as a DataFrame.

        Parameters
        ----------
        format
            Output format: `None` (default), `"pandas"`, `"polars"`, or `"pyarrow"`.

        Returns
        -------
        DataFrame
            Columns: `time`, `survival`.
        """
        return to_dataframe(
            {"time": self._times, "survival": self._surv},
            format=format,
        )


class IPCRidge:
    r"""IPC-weighted ridge regression for survival data.

    A linear survival model that corrects for censoring bias via inverse-probability-of-censoring
    (IPC) reweighting rather than the Cox partial likelihood. The model fits a weighted ridge
    regression on observed (uncensored) log-times:

    $$
    \hat{\beta} = \arg\min_{\beta}
    \sum_{i:\,\delta_i=1} w_i \bigl(\log T_i - X_i \beta\bigr)^2
    + \alpha \|\beta\|_2^2
    $$

    where $w_i = 1 / \hat{G}(T_i^-)$ are IPC weights from the Kaplan-Meier estimate of the censoring
    distribution. This eliminates censoring bias from the least-squares objective without assuming
    proportional hazards.

    Centering and optional standardization are applied before fitting, and coefficients are returned
    on the original covariate scale. The intercept is always estimated.

    Parameters
    ----------
    alpha
        Ridge penalty strength. `0` gives unpenalized IPC-weighted OLS.
    standardize
        Standardize covariates to unit variance before penalizing (default `True`). Coefficients are
        returned on the original scale.

    Examples
    --------
    Fit an IPC-weighted ridge on the lung dataset:

    ```{python}
    import greenwood as gw

    lung = gw.load_dataset("lung", backend="polars")
    y = gw.Surv.right(lung["time"], event=(lung["status"] == 2))
    cols = ["age", "sex"]

    ridge = gw.IPCRidge(alpha=1.0).fit(y, lung[cols])
    ridge
    ```
    """

    def __init__(
        self,
        alpha: float = 1.0,
        *,
        standardize: bool = True,
    ) -> None:
        if alpha < 0.0:
            raise ValueError(f"alpha must be non-negative, got {alpha}.")
        self.alpha = alpha
        self.standardize = standardize

    def __repr__(self) -> str:
        if getattr(self, "coef_", None) is None:
            return f"IPCRidge(alpha={self.alpha}) <unfitted>"
        from ._repr import align_table, num

        labels = ["(Intercept)"] + list(self.term_names_)
        coefs = np.concatenate([[self.intercept_], self.coef_])
        rows = [[num(c)] for c in coefs]
        table = align_table(["coef"], rows, labels)
        return "\n".join(
            [
                f"IPCRidge (IPC-weighted ridge, alpha={self.alpha})",
                "",
                table,
                "",
                f"n = {self.n_}, events = {self.n_event_} (used for fitting)",
            ]
        )

    def fit(self, surv: Surv, covariates: Any, *, data: Any = None) -> IPCRidge:
        r"""Fit the IPC-weighted ridge regression to survival data.

        Only uncensored subjects contribute to the fit. Each is weighted by $1/\hat{G}(T_i^-)$ from
        the Kaplan-Meier censoring estimate, so subjects who experienced events when censoring was
        heavy receive greater influence.

        Parameters
        ----------
        surv
            A right-censored `Surv` response (built with `Surv.right()`).
        covariates
            A dataframe (pandas or polars), a 2-D array, or a formula string (e.g., `"age + sex"`)
            evaluated against `data`.
        data
            A dataframe to evaluate the formula string (ignored if `covariates` is a dataframe or
            array).

        Returns
        -------
        IPCRidge
            The fitted estimator with cached coefficient arrays.
        """
        from ._cox import _design_matrix
        from ._surv import CensoringType

        if surv.type != CensoringType.RIGHT:
            raise NotImplementedError(
                f"IPCRidge supports right-censored responses, not {surv.type.value!r}."
            )

        x, names = _design_matrix(covariates, data)
        if x.shape[0] != surv.n:
            raise ValueError("Covariates and response must have the same number of rows.")

        keep = ~np.isnan(x).any(axis=1)
        x = x[keep]
        time = surv.stop[keep]
        event = surv.event[keep].astype(bool)

        if not event.any():
            raise ValueError("No events remain after dropping missing rows.")

        cens = CensoringDistribution(surv)
        ipc_w = cens.weights()
        ipc_w = ipc_w[keep]

        mask = event & (ipc_w > 0)
        x_fit = x[mask]
        y_fit = np.log(time[mask])
        w_fit = ipc_w[mask]

        _, p = x_fit.shape
        center = x_fit.mean(axis=0)
        scale = x_fit.std(axis=0, ddof=1) if self.standardize else np.ones(p)
        scale = np.where(scale > 0, scale, 1.0)
        xs = (x_fit - center) / scale

        sw = np.sqrt(w_fit)
        xw = xs * sw[:, np.newaxis]
        yw = y_fit * sw

        gram = xw.T @ xw + self.alpha * np.eye(p)
        rhs = xw.T @ yw
        beta_s = np.linalg.solve(gram, rhs)

        self.coef_ = beta_s / scale
        self.intercept_ = float(np.mean(y_fit * w_fit) / np.mean(w_fit) - center @ self.coef_)
        self.term_names_ = names
        self.n_ = int(keep.sum())
        self.n_event_ = int(mask.sum())
        self._center = center
        self._scale = scale
        self._x = x
        self._time = time
        self._event = event
        self._ipc_weights = ipc_w
        return self

    def predict(
        self,
        newdata: Any = None,
        *,
        type: str = "lp",
    ) -> Array:
        r"""Predict from the fitted IPC-weighted ridge model.

        Parameters
        ----------
        newdata
            Covariate values for prediction. `None` uses the training data.
        type
            Prediction type:

            - `"lp"`: the linear predictor $\hat{\beta}_0 + X\hat{\beta}$ (predicted log-time).
              Higher values indicate longer expected survival.
            - `"response"`: $\exp(\text{lp})$, predicted survival time on the original scale.

        Returns
        -------
        ndarray
            Predicted values, shape `(n_subjects,)`.
        """
        from ._cox import _design_matrix

        if type not in ("lp", "response"):
            raise ValueError(f"Unknown predict type {type!r}; use 'lp' or 'response'.")

        x = self._x if newdata is None else _design_matrix(newdata)[0]
        lp = self.intercept_ + x @ self.coef_

        if type == "lp":
            return lp
        return np.exp(lp)

    def _coefficient_columns(self) -> dict[str, Any]:
        return {
            "term": ["(Intercept)"] + list(self.term_names_),
            "estimate": np.concatenate([[self.intercept_], self.coef_]).tolist(),
        }

    def to_frame(self, *, format: str | None = None) -> Any:
        """Return the coefficient table as a DataFrame.

        Parameters
        ----------
        format
            Output format: `None` (default), `"pandas"`, `"polars"`, or `"pyarrow"`.

        Returns
        -------
        DataFrame
            Columns: `term`, `estimate`.
        """
        return to_dataframe(self._coefficient_columns(), format=format)


def _tidy_ipcridge(model: IPCRidge, *, format: str | None = None, **_: Any) -> Any:
    return model.to_frame(format=format)


def _glance_ipcridge(model: IPCRidge, *, format: str | None = None, **_: Any) -> Any:
    return to_dataframe(
        {
            "n": [model.n_],
            "nevent": [model.n_event_],
            "alpha": [model.alpha],
            "n_features": [len(model.term_names_)],
        },
        format=format,
    )


def _register_adapters() -> None:
    from .summaries import register_glance, register_tidier

    register_tidier("greenwood._ipcw.IPCRidge", _tidy_ipcridge)
    register_glance("greenwood._ipcw.IPCRidge", _glance_ipcridge)


_register_adapters()
