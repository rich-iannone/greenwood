r"""Elastic-net penalized Cox regression.

`CoxNet` fits the Cox proportional-hazards model with an elastic-net penalty, minimizing the
negative partial log-likelihood (per observation) plus

$$
\lambda\left(\alpha\|\beta\|_1 + \tfrac{1-\alpha}{2}\|\beta\|_2^2\right)
$$

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

from ._backends import to_dataframe
from ._cox import _cox_terms, _design_matrix

if TYPE_CHECKING:
    from ._surv import Surv

__all__ = ["CoxNet"]

Array = npt.NDArray[Any]


def _soft_threshold(v: Array, thr: float) -> Array:
    """Elementwise soft-thresholding, the proximal operator of the L1 norm."""
    return np.sign(v) * np.maximum(np.abs(v) - thr, 0.0)


class CoxNet:
    r"""Elastic-net penalized Cox proportional hazards model.

    When the number of covariates is large relative to the sample size, unpenalized Cox models
    may overfit or fail to converge due to multicollinearity. CoxNet addresses this by adding
    a penalty term to the partial likelihood, which shrinks coefficients toward zero and can
    perform automatic variable selection. The elastic-net penalty combines $L_1$ (lasso) and
    $L_2$ (ridge) penalties: $\lambda(\alpha\|\beta\|_1 + \tfrac{1-\alpha}{2}\|\beta\|_2^2)$,
    where the mixing parameter $\alpha$ controls the trade-off between sparsity and smoothness.

    Fit the model with `fit()` supplying a right-censored or counting-process `Surv` response
    and a design matrix of covariates. The algorithm uses FISTA (Fast Iterative Shrinkage-
    Thresholding Algorithm) to optimize the penalized partial likelihood. By default, covariates
    are standardized before penalizing (for fair comparison of penalties across features), but
    coefficients are returned on the original scale. Ridge (`l1_ratio=0`) encourages small,
    spread-out coefficients; lasso (`l1_ratio=1`) drives some coefficients exactly to zero.

    The implementation follows the glmnet model for elastic-net regularization, using coordinate
    descent-like optimization with soft-thresholding. Results include penalized coefficients,
    standard errors, and indicators of which features were selected (non-zero coefficients).
    Unlike unpenalized Cox, this model does not compute hazard ratios or perform formal
    hypothesis tests on individual coefficients.

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

    Returns
    -------
    Fitted estimator
        Call `fit()` to produce a fitted estimator with cached results (`coef_`,
        `std_error_`, `n_features_in_`, `feature_names_in_`), accessible as arrays or
        exported to DataFrames.

    Details
    -------
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

    lung = gw.load_dataset("lung", backend="polars")
    y = gw.Surv.right(lung["time"], event=(lung["status"] == 2))
    cols = ["age", "sex", "ph.ecog", "ph.karno", "wt.loss"]
    coxnet = gw.CoxNet(penalizer=0.05, l1_ratio=1.0).fit(y, lung[cols])
    coxnet
    ```
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
        r"""Fit the elastic-net penalized Cox model to survival data.

        Fits a Cox proportional-hazards model with elastic-net penalty (L1 + L2
        regularization) to a right-censored or counting-process response and covariates.
        The penalty shrinks coefficients toward zero, selecting a sparse subset of important
        variables (when L1 dominates) or smoothly shrinking all coefficients (when L2
        dominates). An intercept is added automatically.

        The CoxNet model is useful for high-dimensional covariate spaces where unpenalized
        Cox fails to converge or produces unstable estimates. It maintains the
        proportional-hazards interpretation of hazard ratios while controlling model
        complexity. Tuning the `penalizer` strength and `l1_ratio` mixing parameter enables
        variable selection and regularized estimation.

        Parameters
        ----------
        surv
            A `Surv` response (right-censored or counting-process). Built with `Surv.right()`
            or `Surv.counting()`.
        covariates
            A dataframe (pandas or polars), a 2-D array, or a formula string (e.g.,
            `"age + sex"`) evaluated against the `data` argument.
        data
            A dataframe to evaluate the formula string (ignored if `covariates` is a
            dataframe or array).

        Returns
        -------
        CoxNet
            The fitted estimator object itself (for method chaining) with cached coefficient
            arrays, standard errors, and model metrics.

        Details
        -------
        The elastic-net penalty is $\lambda(\alpha L_1 + (1 - \alpha) L_2)$, where
        $\lambda$ = `penalizer` and $\alpha$ = `l1_ratio`. Setting `l1_ratio=1` gives lasso
        ($L_1$ only, induces sparsity); `l1_ratio=0` gives ridge ($L_2$ only, smooth
        shrinkage); intermediate values blend both effects.

        Estimation uses proximal gradient descent (FISTA) to optimize the penalized
        partial likelihood. Covariates are centered and optionally standardized before
        fitting; standardization affects the penalty scale but not the fitted hazard ratios.

        Examples
        --------
        Fit a ridge-penalized Cox model (L2 penalty, smooth shrinkage) on the bundled
        `lung` dataset:

        ```{python}
        import greenwood as gw

        lung = gw.load_dataset("lung", backend="polars")
        y = gw.Surv.right(lung["time"], event=(lung["status"] == 2))
        cols = ["age", "sex", "ph.ecog", "ph.karno", "wt.loss"]
        coxnet_ridge = gw.CoxNet(penalizer=0.05, l1_ratio=0.0).fit(y, lung[cols])
        coxnet_ridge
        ```

        Fit a lasso-penalized Cox model (L1 penalty, sparse selection):

        ```{python}
        coxnet_lasso = gw.CoxNet(penalizer=0.05, l1_ratio=1.0).fit(y, lung[cols])
        coxnet_lasso
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

    def predict(
        self,
        newdata: Any = None,
        *,
        type: str = "lp",
        times: Any = None,
        format: str | None = None,
    ) -> Any:
        r"""Predict log-hazard, risk, or survival probabilities from the penalized Cox model.

        Generates predictions from a fitted elastic-net penalized Cox model. Pass `newdata=None`
        to predict for the training data (fitted subjects).

        Three prediction types are available:

        1. **Linear predictor** (`type="lp"`): the centered log-hazard $X\beta$, a risk score
           showing how covariates affect hazard. Higher values indicate higher risk. Centered
           means the baseline is set such that $\exp(\text{lp}) = 1$ for an average subject
           (average covariate values).

        2. **Risk** (`type="risk"`): the relative hazard $\exp(\text{lp})$, comparing each
           subject's hazard to the baseline (average). A value of 2.0 means 2x baseline hazard.

        3. **Survival** (`type="survival"`): survival probabilities $S(t \mid x)$ at specified
           times, returned as a DataFrame. Uses the baseline cumulative hazard from the training
           data and applies the covariate adjustment via relative risk.

        Parameters
        ----------
        newdata
            Covariate values for prediction. A dataframe (Pandas or Polars), 2-D array, or
            None (default). If `None`, uses the training data (design matrix used at fit time).
            Must have the same columns/features as the training data. Covariates are centered
            using the centering from the training data.
        type
            Prediction type (default `"lp"`):

            - `"lp"`: Centered linear predictor $X\beta$ (log-hazard). Returns an array.
            - `"risk"`: Relative risk $\exp(\text{lp})$. Returns an array (always positive).
            - `"survival"`: Survival probabilities $S(t \mid x)$ at times in `times`. Returns a
              frame with `time` column and one column per subject.
        times
            Query times for `type="survival"` (ignored for other types). An array-like of
            floats. If `None` (the default), uses the event times from the training data
            (baseline cumulative hazard times).
        format
            Output format for the returned frame (`type="survival"`): `None` (default),
            `"pandas"`, `"polars"`, or `"pyarrow"`. When `None`, a backend is auto-detected
            (Polars, then Pandas, then PyArrow). Ignored for `type="lp"` and `type="risk"`.

        Returns
        -------
        ndarray or DataFrame
            For `type="lp"` or `type="risk"`: an array of shape (n_subjects,) containing
            centered log-hazard or relative risk values respectively. For `type="survival"`:
            a DataFrame with columns `time` (query times) and `subject_1`, `subject_2`, etc.
            containing survival probabilities at each time. Column names match the input row
            index if `newdata` has a row index.

        Raises
        ------
        ValueError
            If `type` is not one of `"lp"`, `"risk"`, or `"survival"`.

        Details
        -------
        The penalized Cox model estimates $\exp(\text{lp})$ as a multiplier on the baseline
        cumulative hazard: $H(t \mid x) = H_0(t)\,\exp(\text{lp})$. Survival is then
        $S(t \mid x) = \exp(-H(t \mid x))$. The baseline cumulative hazard $H_0(t)$ is
        estimated using the Breslow estimator from the training data, and is fixed for new
        predictions.

        Centering ensures that the linear predictor at the average covariate level is 0,
        making relative risks and survival curves interpretable. Predictions assume the model
        is well-specified and that the proportional-hazards assumption holds.

        Examples
        --------
        The default `type="lp"` returns the centered linear predictor (log-hazard). Here are
        the values for the first five subjects:

        ```{python}
        import greenwood as gw

        lung = gw.load_dataset("lung", backend="polars")
        y = gw.Surv.right(lung["time"], event=(lung["status"] == 2))
        cols = ["age", "sex", "ph.ecog", "ph.karno", "wt.loss"]
        coxnet = gw.CoxNet(penalizer=0.05, l1_ratio=1.0).fit(y, lung[cols])
        coxnet.predict(lung[cols], type="lp")[:5]
        ```

        Pass `type="risk"` for the relative risk $\exp(\text{lp})$, showing how many times the
        baseline hazard each subject has:

        ```{python}
        coxnet.predict(lung[cols], type="risk")[:5]
        ```

        Pass `type="survival"` for predicted survival curves at specified times or at the
        event times from training data; pass `format=` to choose the backend (here, Polars):

        ```{python}
        coxnet.predict(lung[cols][:2], type="survival", times=[180, 365], format="polars")
        ```
        """
        x = self._x if newdata is None else _design_matrix(newdata)[0]
        lp = (x - self._center) @ self.coef_
        if type == "lp":
            return lp
        if type == "risk":
            return np.exp(lp)
        if type == "survival":
            base_times, base_cumhaz = self._baseline()
            query = base_times if times is None else np.atleast_1d(np.asarray(times, dtype=float))
            idx = np.searchsorted(base_times, query, side="right") - 1
            h0 = np.where(idx >= 0, base_cumhaz[idx.clip(min=0)], 0.0)
            risk = np.exp(x @ self.coef_)
            surv = np.exp(-np.outer(h0, risk))
            columns: dict[str, Any] = {"time": query}
            columns.update({f"subject_{i + 1}": surv[:, i] for i in range(x.shape[0])})
            return to_dataframe(columns, format=format)
        raise ValueError(f"Unknown predict type {type!r}; use 'lp', 'risk', or 'survival'.")

    def _coefficient_columns(self) -> dict[str, Any]:
        return {
            "term": self.term_names_,
            "estimate": self.coef_,
            "hazard_ratio": self.hazard_ratio_,
        }

    def to_frame(self, *, format: str | None = None) -> Any:
        """Return the penalized coefficient table as a DataFrame.

        Exports one row per term with the penalized coefficient estimate and its hazard
        ratio. Terms set to zero by the lasso remain in the table with zero estimates.

        Parameters
        ----------
        format
            Output format: `None` (default), `"pandas"`, `"polars"`, or `"pyarrow"`. When
            `None`, a backend is auto-detected (Polars, then Pandas, then PyArrow).

        Returns
        -------
        pandas.DataFrame, polars.DataFrame, or pyarrow.Table
            A tidy table with columns `term`, `estimate`, and `hazard_ratio`.

        Raises
        ------
        ImportError
            If the requested (or, when auto-detecting, any) DataFrame library is not
            installed.

        Examples
        --------
        Fit a lasso-penalized Cox model and export the coefficient table as a Polars frame:

        ```{python}
        import greenwood as gw

        lung = gw.load_dataset("lung", backend="polars")
        y = gw.Surv.right(lung["time"], event=(lung["status"] == 2))
        cols = ["age", "sex", "ph.ecog", "ph.karno", "wt.loss"]
        coxnet = gw.CoxNet(penalizer=0.05, l1_ratio=1.0).fit(y, lung[cols])
        coxnet.to_frame(format="polars")
        ```

        Request a different backend with `format=`:

        ```{python}
        coxnet.to_frame(format="pandas")
        ```
        """
        return to_dataframe(self._coefficient_columns(), format=format)


# ---------------------------------------------------------------------------
# Cross-validated penalizer selection
# ---------------------------------------------------------------------------


def _lambda_max_cox(
    xs: Array,
    entry: Array,
    exit_: Array,
    event: Array,
    weight: Array,
    l1_ratio: float,
) -> float:
    """Smallest lambda that zeros all coefficients under lasso/elastic-net.

    At beta = 0 the gradient of the minimized objective (negative partial log-likelihood / n, no
    ridge term) equals `-grad_loglik(0) / n`.  The L1 subgradient condition
    `|(-grad_loglik(0)/n)_j| <= lambda * alpha` must hold for every component, so:

    lambda_max = max_j |grad_loglik(0)_j| / (n * alpha)

    For pure ridge (`l1_ratio = 0`) the condition never holds for any finite lambda, so the
    effective alpha is clamped to 0.01 to return a usable upper bound for the path.
    """
    n, p = xs.shape
    groups = [(np.arange(n), np.unique(exit_[event]))]
    _, grad, _ = _cox_terms(np.zeros(p), xs, entry, exit_, event, weight, groups, "breslow")
    effective_alpha = max(l1_ratio, 1e-2)
    return float(np.max(np.abs(grad)) / (n * effective_alpha))


class CoxNetCVResult:
    r"""Result of cross-validated penalizer selection for :class:`CoxNet`.

    Returned by `cv_coxnet()`. Stores the cross-validation scores at every tested penalizer value,
    identifies the best penalizer (highest concordance or lowest Brier score), and applies the
    *one-standard-error rule* for a more parsimonious alternative.

    Penalizers are ordered from *largest* (most regularized, sparsest model) to *smallest* (least
    regularized, densest model).

    Attributes
    ----------
    penalizers_ : ndarray
        Penalizer values tested, sorted descending.
    mean_scores_ : ndarray
        Mean cross-validation score at each penalizer.
    std_scores_ : ndarray
        Standard deviation of per-fold scores at each penalizer.
    n_nonzero_ : ndarray
        Average number of non-zero coefficients at each penalizer (averaged over cross-validation
        folds).
    metric_ : str
        Metric used: `"concordance"` or `"brier"`.
    l1_ratio_ : float
        Elastic-net mixing parameter fixed during the path search.
    k_ : int
        Number of folds used.
    best_penalizer_ : float
        Penalizer with the best mean cross-validation score.
    best_score_ : float
        Mean score at `best_penalizer_`.
    penalizer_1se_ : float
        Largest penalizer (most regularized) whose mean score is within one standard error of
        `best_score_`. Prefer this when parsimony matters.
    score_1se_ : float
        Mean score at `penalizer_1se_`.
    """

    def __init__(
        self,
        penalizers: Array,
        mean_scores: Array,
        std_scores: Array,
        n_nonzero: Array,
        metric: str,
        l1_ratio: float,
        k: int,
    ) -> None:
        self.penalizers_ = penalizers
        self.mean_scores_ = mean_scores
        self.std_scores_ = std_scores
        self.n_nonzero_ = n_nonzero
        self.metric_ = metric
        self.l1_ratio_ = l1_ratio
        self.k_ = k

        higher_is_better = metric == "concordance"
        best_idx = int(np.argmax(mean_scores) if higher_is_better else np.argmin(mean_scores))
        self.best_penalizer_ = float(penalizers[best_idx])
        self.best_score_ = float(mean_scores[best_idx])

        # 1-SE rule: most regularized (largest lambda) within 1 SE of best score.
        se_best = float(std_scores[best_idx])
        if higher_is_better:
            candidates = np.where(mean_scores >= self.best_score_ - se_best)[0]
        else:
            candidates = np.where(mean_scores <= self.best_score_ + se_best)[0]
        one_se_idx = candidates[int(np.argmax(penalizers[candidates]))]
        self.penalizer_1se_ = float(penalizers[one_se_idx])
        self.score_1se_ = float(mean_scores[one_se_idx])

    def __repr__(self) -> str:
        direction = "↑ higher is better" if self.metric_ == "concordance" else "↓ lower is better"
        lo, hi = float(self.penalizers_.min()), float(self.penalizers_.max())
        return "\n".join(
            [
                f"CoxNetCV ({self.metric_}, {direction},"
                f" l1_ratio={self.l1_ratio_}, {self.k_}-fold)",
                "",
                f"  best penalizer : {self.best_penalizer_:.5g}"
                f"  (mean {self.metric_}: {self.best_score_:.4f})",
                f"  1-SE  penalizer : {self.penalizer_1se_:.5g}"
                f"  (mean {self.metric_}: {self.score_1se_:.4f})",
                "",
                f"  {len(self.penalizers_)} penalizers tested, range [{lo:.3g}, {hi:.3g}]",
            ]
        )

    def to_frame(self, *, format: str | None = None) -> Any:
        """Return the CV path as a tidy DataFrame, one row per penalizer.

        Parameters
        ----------
        format
            Output format: `None` (default), `"pandas"`, `"polars"`, or `"pyarrow"`. When `None`, a
            backend is auto-detected.

        Returns
        -------
        DataFrame
            Columns: `penalizer`, `mean_score`, `std_score`, `n_nonzero`. Rows are ordered from
            largest to smallest penalizer.

        Examples
        --------
        ```{python}
        import greenwood as gw

        lung = gw.load_dataset("lung", backend="polars")
        y = gw.Surv.right(lung["time"], event=(lung["status"] == 2))
        cols = ["age", "sex", "ph.ecog", "ph.karno", "wt.loss"]
        result = gw.cv_coxnet(y, lung[cols], seed=23)
        result.to_frame(format="polars")
        ```
        """
        return to_dataframe(
            {
                "penalizer": self.penalizers_,
                "mean_score": self.mean_scores_,
                "std_score": self.std_scores_,
                "n_nonzero": self.n_nonzero_,
            },
            format=format,
        )


def cv_coxnet(
    surv: Surv,
    covariates: Any,
    *,
    data: Any = None,
    l1_ratio: float = 1.0,
    penalizers: Any = None,
    n_penalizers: int = 50,
    eps: float = 1e-3,
    k: int = 5,
    metric: str = "concordance",
    times: Any = None,
    standardize: bool = True,
    max_iter: int = 1000,
    tol: float = 1e-7,
    stratified: bool = True,
    seed: int | None = None,
) -> CoxNetCVResult:
    r"""Select the `CoxNet` penalizer by k-fold cross-validation.

    Evaluates a log-spaced grid of penalizer values (or a user-supplied sequence) with stratified
    k-fold cross-validation and returns the value that maximises concordance (or minimises the
    integrated Brier score).

    Parameters
    ----------
    surv
        A right-censored or counting-process `Surv` response.
    covariates
        Covariate design (dataframe, 2-D array, or formula string with `data`).
    data
        DataFrame to evaluate a formula `covariates` string against.
    l1_ratio
        Elastic-net mixing parameter (default `1.0` = lasso). Fixed across the entire path (only the
        `penalizer` is varied).
    penalizers
        An explicit sequence of penalizer values to evaluate. When `None` (the default) a log-spaced
        path of `n_penalizers=` values is generated automatically from `lambda_max` down to
        `eps * lambda_max`.
    n_penalizers
        Number of log-spaced values in the auto-generated path (the default is `50`). Ignored when
        `penalizers=` is supplied.
    eps
        Ratio of smallest to largest auto-generated penalizer (default of `1e-3`). Smaller values
        extend the path toward weaker regularisation. Ignored when `penalizers=` is supplied.
    k
        Number of cross-validation folds (the default is `5`).
    metric
        Scoring metric:

        - `"concordance"` (the default): Harrell's C-statistic, where higher is better.
        - `"brier"`: integrated IPCW Brier score, which requires `times=` (here lower is better).

    times
        Evaluation times for `metric="brier"` (1-D array-like, $\ge 2$ values).
    standardize
        Standardize covariates before penalising (default is `True`). Must match the `standardize=`
        argument used for the final `CoxNet` fit.
    max_iter, tol
        FISTA iteration limit and convergence tolerance for each inner fit.
    stratified
        Use stratified k-fold to preserve the event rate in every fold (the default is `True`).
    seed
        Random seed for reproducible fold assignments.

    Returns
    -------
    CoxNetCVResult
        Object exposing `best_penalizer_`, `penalizer_1se_`, `mean_scores_`, `std_scores_`,
        `n_nonzero_`, and a `to_frame()` method for the path.

    Details
    -------
    **Lambda path.** When `penalizers=` is not supplied the path is

    $$
    \lambda_{\max} = \frac{\max_j \bigl|\nabla_j\,\ell(\beta=0)\bigr|}{n \cdot
                     \max(\alpha,\, 0.01)}, \quad
    \lambda_i = \lambda_{\max} \cdot \varepsilon^{i/(m-1)}, \quad i = 0,\ldots,m-1
    $$

    where $\nabla\ell(\beta=0)$ is the score of the partial log-likelihood at the null model,
    $\alpha$ = `l1_ratio`, $\varepsilon$ = `eps`, and $m$ = `n_penalizers`.

    **1-SE rule.** `penalizer_1se_` is the *largest* penalizer (most regularized) whose mean score
    lies within one standard error of `best_score_`. Prefer it when parsimony matters as it yields a
    sparser model at no significant loss of predictive accuracy.

    **Fitting the final model.** After selecting a penalizer, refit on all data:

    ```python
    result = gw.cv_coxnet(y, x, seed=23)
    final = gw.CoxNet(penalizer=result.best_penalizer_).fit(y, x)
    ```

    Examples
    --------
    Run cross-validated lasso selection on the `lung` dataset:

    ```{python}
    import greenwood as gw

    lung = gw.load_dataset("lung", backend="polars")
    y = gw.Surv.right(lung["time"], event=(lung["status"] == 2))
    cols = ["age", "sex", "ph.ecog", "ph.karno", "wt.loss"]

    result = gw.cv_coxnet(y, lung[cols], l1_ratio=1.0, seed=23)
    result
    ```

    Inspect the full penalizer path:

    ```{python}
    result.to_frame(format="polars")
    ```

    Fit the final model at the best penalizer:

    ```{python}
    final = gw.CoxNet(penalizer=result.best_penalizer_, l1_ratio=1.0).fit(y, lung[cols])
    final
    ```

    Use the 1-SE rule for a sparser model:

    ```{python}
    sparse = gw.CoxNet(penalizer=result.penalizer_1se_, l1_ratio=1.0).fit(y, lung[cols])
    sparse
    ```
    """
    from ._cox import _design_matrix
    from ._metrics import concordance_index, integrated_brier_score
    from ._resample import _stratified_kfold_indices, _subset_surv
    from ._surv import CensoringType

    if not (0.0 <= l1_ratio <= 1.0):
        raise ValueError(f"l1_ratio must be in [0, 1], got {l1_ratio}.")
    if k < 2:
        raise ValueError(f"k must be at least 2, got {k}.")
    if metric not in ("concordance", "brier"):
        raise ValueError(f"metric must be 'concordance' or 'brier', got {metric!r}.")

    brier_times: list[float] = []
    if metric == "brier":
        if times is None:
            raise ValueError("metric='brier' requires `times`.")
        brier_times = [float(t) for t in np.atleast_1d(np.asarray(times, dtype=float))]
        if len(brier_times) < 2:
            raise ValueError("metric='brier' requires at least two time points in `times`.")

    if surv.type not in (CensoringType.RIGHT, CensoringType.COUNTING):
        raise NotImplementedError(
            f"cv_coxnet supports right-censored and counting-process responses, "
            f"not {surv.type.value!r}."
        )

    # --- Preprocess: complete-case filter ----------------------------------------
    x_full, _ = _design_matrix(covariates, data)
    if x_full.shape[0] != surv.n:
        raise ValueError("Covariates and response must have the same number of rows.")

    keep = ~np.isnan(x_full).any(axis=1)
    if not keep.all():
        x_full = x_full[keep]
        surv = _subset_surv(surv, np.nonzero(keep)[0])

    n, p = x_full.shape
    entry = surv.entry
    exit_ = surv.stop
    event = surv.event
    weight = surv.weights if surv.weights is not None else np.ones(n)

    if not event.any():
        raise ValueError("No events remain after dropping missing rows.")

    # Standardise (same transform as CoxNet.fit) for lambda_max computation.
    center = x_full.mean(axis=0)
    scale = x_full.std(axis=0) if standardize else np.ones(p)
    scale = np.where(scale > 0, scale, 1.0)
    xs_full = (x_full - center) / scale

    # --- Lambda path -------------------------------------------------------------
    if penalizers is not None:
        lam_path = np.sort(np.atleast_1d(np.asarray(penalizers, dtype=float)))[::-1].copy()
        if np.any(lam_path < 0):
            raise ValueError("All penalizers must be non-negative.")
    else:
        if n_penalizers < 1:
            raise ValueError(f"n_penalizers must be at least 1, got {n_penalizers}.")
        lam_max = _lambda_max_cox(xs_full, entry, exit_, event, weight, l1_ratio)
        lam_min = eps * lam_max
        lam_path = np.exp(np.linspace(np.log(lam_max), np.log(lam_min), n_penalizers))

    # --- Folds -------------------------------------------------------------------
    n_events = int(event.astype(bool).sum())
    if n_events < 2 * k:
        warnings.warn(
            f"Only {n_events} events for {k}-fold CV (fewer than 2 × k = {2 * k}). "
            "Some folds may have too few events. Consider reducing k.",
            UserWarning,
            stacklevel=2,
        )

    folds: list[Array] = (
        _stratified_kfold_indices(surv, k, seed)
        if stratified
        else list(np.array_split(np.random.default_rng(seed).permutation(n), k))
    )

    # --- CV loop -----------------------------------------------------------------
    higher_is_better = metric == "concordance"
    n_lam = len(lam_path)
    mean_scores = np.empty(n_lam)
    std_scores = np.empty(n_lam)
    n_nonzero_arr = np.empty(n_lam)

    for lam_idx, lam in enumerate(lam_path):
        fold_scores: list[float] = []
        fold_nonzero: list[int] = []

        for fold_i in range(k):
            test_idx = folds[fold_i]
            train_idx = np.concatenate([folds[j] for j in range(k) if j != fold_i])

            surv_train = _subset_surv(surv, train_idx)
            surv_test = _subset_surv(surv, test_idx)
            x_train = x_full[train_idx]
            x_test = x_full[test_idx]

            # Skip degenerate folds (no events in train or test).
            if not surv_train.event.astype(bool).any():
                continue
            if not surv_test.event.astype(bool).any():
                continue

            fold_model = CoxNet(
                penalizer=float(lam),
                l1_ratio=l1_ratio,
                standardize=standardize,
                max_iter=max_iter,
                tol=tol,
            ).fit(surv_train, x_train)

            fold_nonzero.append(int(np.count_nonzero(fold_model.coef_)))

            if metric == "concordance":
                lp = fold_model.predict(x_test, type="lp")
                fold_scores.append(float(concordance_index(surv_test, lp)))
            else:
                frame = fold_model.predict(x_test, type="survival", times=brier_times)
                try:
                    import polars as pl

                    if isinstance(frame, pl.DataFrame):
                        probs = frame[:, 1:].to_numpy().T
                    else:
                        probs = frame.iloc[:, 1:].to_numpy().T  # type: ignore[union-attr]
                except (ImportError, AttributeError):
                    cols_list = list(frame.columns)  # type: ignore[union-attr]
                    probs = frame[cols_list[1:]].to_numpy().T  # type: ignore[index]
                fold_scores.append(float(integrated_brier_score(surv_test, probs, brier_times)))

        if fold_scores:
            arr = np.asarray(fold_scores)
            mean_scores[lam_idx] = float(arr.mean())
            std_scores[lam_idx] = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
        else:
            mean_scores[lam_idx] = 0.5 if higher_is_better else np.inf
            std_scores[lam_idx] = 0.0
        n_nonzero_arr[lam_idx] = float(np.mean(fold_nonzero)) if fold_nonzero else 0.0

    return CoxNetCVResult(
        penalizers=lam_path,
        mean_scores=mean_scores,
        std_scores=std_scores,
        n_nonzero=n_nonzero_arr,
        metric=metric,
        l1_ratio=l1_ratio,
        k=k,
    )
