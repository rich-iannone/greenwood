"""Elastic-net penalized Cox regression.

`CoxNet` fits the Cox proportional-hazards model with an elastic-net penalty, minimizing the
negative partial log-likelihood (per observation) plus

    penalizer * [ l1_ratio * ||b||_1 + (1 - l1_ratio) / 2 * ||b||_2^2 ]

on standardized covariates, the objective used by glmnet's `coxnet`. `l1_ratio=1` is the
lasso (sparse, selects variables), `l1_ratio=0` is ridge (shrinks smoothly), and values in
between blend the two. The partial likelihood uses the Breslow tie handling, as glmnet does.

Penalized coefficients are biased by design, so `CoxNet` reports point estimates for
prediction and variable selection but not standard errors or p-values. With `penalizer=0` it
reduces to the ordinary `CoxPH` (Breslow) fit.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt

from ._cox import _cox_terms, _design_matrix

if TYPE_CHECKING:
    from ._surv import Surv

__all__ = ["CoxNet"]

Array = npt.NDArray[Any]


def _soft_threshold(v: Array, thr: float) -> Array:
    """Elementwise soft-thresholding, the proximal operator of the L1 norm."""
    return np.sign(v) * np.maximum(np.abs(v) - thr, 0.0)


class CoxNet:
    """Elastic-net penalized Cox proportional hazards model.

    Parameters
    ----------
    penalizer
        Overall penalty strength (`lambda`). `0` recovers the unpenalized Breslow Cox fit.
    l1_ratio
        Elastic-net mixing in `[0, 1]`: `1` is lasso, `0` is ridge.
    standardize
        Standardize covariates to unit variance before penalizing (default `True`, as in
        glmnet). Coefficients are returned on the original scale.
    max_iter, tol
        Maximum FISTA iterations and the relative-change convergence tolerance.

    Notes
    -----
    Call `fit(surv, covariates)` with a right-censored or counting-process `Surv` response.
    `covariates` may be a dataframe, a 2-D array, or a formula string with `data`. Stratified
    penalized fits are not supported.

    Examples
    --------
    Build a `Surv` response from the bundled `lung` dataset and fit a lasso (`l1_ratio=1.0`)
    elastic-net Cox model over several covariates. The `ph.ecog`, `ph.karno`, and `wt.loss`
    columns have missing values, which `CoxNet` drops automatically. Printing the fitted object
    shows the penalized coefficients and how many were driven to zero.

    ```{python}
    import greenwood as gw
    from greenwood import Surv

    lung = gw.load_dataset("lung")
    y = Surv.right(lung["time"], event=(lung["status"] == 2))
    cols = ["age", "sex", "ph.ecog", "ph.karno", "wt.loss"]
    coxnet = gw.CoxNet(penalizer=0.05, l1_ratio=1.0).fit(y, lung[cols])
    coxnet
    ```

    The `coxnet` object fit here is reused by the method examples below.
    """

    def __init__(
        self,
        penalizer: float = 0.1,
        l1_ratio: float = 0.5,
        *,
        standardize: bool = True,
        max_iter: int = 1000,
        tol: float = 1e-7,
    ) -> None:
        if penalizer < 0.0:
            raise ValueError(f"penalizer must be non-negative, got {penalizer}.")
        if not 0.0 <= l1_ratio <= 1.0:
            raise ValueError(f"l1_ratio must be in [0, 1], got {l1_ratio}.")
        self.penalizer = penalizer
        self.l1_ratio = l1_ratio
        self.standardize = standardize
        self.max_iter = max_iter
        self.tol = tol

    def __repr__(self) -> str:
        if getattr(self, "coef_", None) is None:
            return f"CoxNet(penalizer={self.penalizer}, l1_ratio={self.l1_ratio}) <unfitted>"
        from ._repr import align_table, num

        rows = [[num(c)] for c in self.coef_]
        table = align_table(["coef"], rows, list(self.term_names_))
        n_nonzero = int(np.count_nonzero(self.coef_))
        return "\n".join(
            [
                f"CoxNet (elastic-net Cox, penalizer={self.penalizer}, l1_ratio={self.l1_ratio})",
                "",
                table,
                "",
                f"n = {self.n_}, events = {self.n_event_}, nonzero coefficients = {n_nonzero}",
            ]
        )

    def fit(self, surv: Surv, covariates: Any, *, data: Any = None) -> CoxNet:
        """Fit the penalized model to a `Surv` response and a covariate design.

        Examples
        --------
        `penalizer` sets the overall penalty strength and `l1_ratio` the elastic-net mixing:
        `l1_ratio=1` is a lasso (sparse, selects variables), `l1_ratio=0` is ridge (smooth
        shrinkage), and `penalizer=0` recovers the ordinary Cox fit. Here is a ridge fit to the
        same `y` response and `lung` data from the class example above:

        ```{python}
import greenwood as gw
        gw.CoxNet(penalizer=0.05, l1_ratio=0.0).fit(y, lung[cols])
        ```
        """
        from ._surv import CensoringType

        if surv.type not in (CensoringType.RIGHT, CensoringType.COUNTING):
            raise NotImplementedError(
                f"CoxNet supports right-censored and counting-process responses, "
                f"not {surv.type.value!r}."
            )

        x, names = _design_matrix(covariates, data)
        if x.shape[0] != surv.n:
            raise ValueError("Covariates and response must have the same number of rows.")

        entry, exit_, event = surv.entry, surv.stop, surv.event
        weight = surv.weights if surv.weights is not None else np.ones(surv.n)
        keep = ~np.isnan(x).any(axis=1)
        x, entry, exit_ = x[keep], entry[keep], exit_[keep]
        event, weight = event[keep], weight[keep]
        if not event.any():
            raise ValueError("No events remain after dropping missing rows.")

        n, p = x.shape
        center = x.mean(axis=0)
        scale = x.std(axis=0) if self.standardize else np.ones(p)
        scale = np.where(scale > 0, scale, 1.0)
        xs = (x - center) / scale  # centering is free for Cox; scaling defines the penalty

        groups = [(np.arange(n), np.unique(exit_[event]))]
        lam, alpha = self.penalizer, self.l1_ratio

        def smooth(b: Array) -> tuple[float, Array]:
            loglik, grad, _ = _cox_terms(b, xs, entry, exit_, event, weight, groups, "breslow")
            h = -loglik / n + 0.5 * lam * (1.0 - alpha) * float(b @ b)
            grad_h = -grad / n + lam * (1.0 - alpha) * b
            return h, grad_h

        beta = np.zeros(p)
        momentum = beta.copy()
        t_acc = 1.0
        step = 1.0
        for _ in range(self.max_iter):
            h_z, grad_z = smooth(momentum)
            # Backtracking line search for the proximal-gradient step size.
            while True:
                candidate = _soft_threshold(momentum - step * grad_z, step * lam * alpha)
                diff = candidate - momentum
                h_c, _ = smooth(candidate)
                if h_c <= h_z + float(grad_z @ diff) + float(diff @ diff) / (2.0 * step):
                    break
                step *= 0.5
                if step < 1e-12:
                    break
            t_next = (1.0 + np.sqrt(1.0 + 4.0 * t_acc**2)) / 2.0
            momentum = candidate + ((t_acc - 1.0) / t_next) * (candidate - beta)
            change = np.linalg.norm(candidate - beta) / (np.linalg.norm(beta) + self.tol)
            beta, t_acc = candidate, t_next
            if change < self.tol:
                break

        self._x = x
        self._entry, self._exit, self._event, self._weight = entry, exit_, event, weight
        self._center = center
        self._event_times = np.unique(exit_[event])
        self.term_names_ = names
        self.coef_ = beta / scale  # back to the original covariate scale
        self.hazard_ratio_ = np.exp(self.coef_)
        loglik, _, _ = _cox_terms(beta, xs, entry, exit_, event, weight, groups, "breslow")
        self.loglik_ = float(loglik)
        self.n_ = n
        self.n_event_ = int(event.sum())
        return self

    def _baseline(self) -> tuple[Array, Array]:
        """Breslow baseline cumulative hazard (uncentered), at the event times."""
        risk = np.exp(self._x @ self.coef_) * self._weight
        cumhaz = np.empty(self._event_times.shape[0])
        total = 0.0
        for i, t in enumerate(self._event_times):
            at_risk = (self._entry < t) & (self._exit >= t)
            dying = (self._exit == t) & self._event
            total += self._weight[dying].sum() / risk[at_risk].sum()
            cumhaz[i] = total
        return self._event_times, cumhaz

    def predict(self, newdata: Any = None, *, type: str = "lp", times: Any = None) -> Any:
        """Predict `"lp"` (centered linear predictor), `"risk"` (`exp(lp)`), or `"survival"`.

        Examples
        --------
        The default `type="lp"` returns the centered linear predictor. Here are the values for
        the first five subjects (reusing the `coxnet` fit above):

        ```{python}
        coxnet.predict(lung[cols], type="lp")[:5]
        ```

        Pass `type="risk"` for the relative risk `exp(lp)`, or `type="survival"` for predicted
        survival curves.
        """
        x = self._x if newdata is None else _design_matrix(newdata)[0]
        lp = (x - self._center) @ self.coef_
        if type == "lp":
            return lp
        if type == "risk":
            return np.exp(lp)
        if type == "survival":
            import pandas as pd

            base_times, base_cumhaz = self._baseline()
            query = base_times if times is None else np.atleast_1d(np.asarray(times, dtype=float))
            idx = np.searchsorted(base_times, query, side="right") - 1
            h0 = np.where(idx >= 0, base_cumhaz[idx.clip(min=0)], 0.0)
            risk = np.exp(x @ self.coef_)
            surv = np.exp(-np.outer(h0, risk))
            frame = pd.DataFrame({f"subject_{i + 1}": surv[:, i] for i in range(x.shape[0])})
            frame.insert(0, "time", query)
            return frame
        raise ValueError(f"Unknown predict type {type!r}; use 'lp', 'risk', or 'survival'.")

    def _coefficient_columns(self) -> dict[str, Any]:
        return {
            "term": self.term_names_,
            "estimate": self.coef_,
            "hazard_ratio": self.hazard_ratio_,
        }

    def to_pandas(self) -> Any:
        """Return the penalized coefficient table as a pandas DataFrame.

        This method exports one row per term with the penalized coefficient estimate and
        its hazard ratio. Terms set to zero by the lasso remain in the table with zero
        estimates.

        Returns
        -------
        pandas.DataFrame
            A tidy DataFrame with columns `term`, `estimate`, and `hazard_ratio`.

        Raises
        ------
        ImportError
            If pandas is not installed.

        Examples
        --------
        Export the fitted CoxNet coefficients to pandas:

        ```{python}
        coxnet.to_pandas()
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
        """Return the penalized coefficient table as a Polars DataFrame.

        This method exports one row per term with the penalized coefficient estimate and
        its hazard ratio. Terms set to zero by the lasso remain in the table with zero
        estimates.

        Returns
        -------
        polars.DataFrame
            A tidy DataFrame with columns `term`, `estimate`, and `hazard_ratio`.

        Raises
        ------
        ImportError
            If polars is not installed.

        Examples
        --------
        Export the fitted CoxNet coefficients to Polars:

        ```{python}
        coxnet.to_polars()
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
        """Return the penalized coefficient table as a PyArrow Table.

        This method exports one row per term with the penalized coefficient estimate and
        its hazard ratio for Arrow-based interoperability.

        Returns
        -------
        pyarrow.Table
            A table with columns `term`, `estimate`, and `hazard_ratio`.

        Raises
        ------
        ImportError
            If pyarrow is not installed.

        Examples
        --------
        Export the fitted CoxNet coefficients to Arrow:

        ```{python}
        coxnet.to_arrow()
        ```
        """
        try:
            import pyarrow as pa
        except ImportError as e:
            raise ImportError(
                "pyarrow is required for to_arrow(). Install it with: pip install pyarrow"
            ) from e

        return pa.table(self._coefficient_columns())
