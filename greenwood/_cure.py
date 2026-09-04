r"""Mixture cure models for survival data with long-term survivors.

A mixture cure model decomposes the population into a cured fraction (who will never
experience the event) and a susceptible fraction (who follow a latent survival distribution):

$$
S_{\text{pop}}(t \mid x, z) = 1 - \pi(x) + \pi(x)\,S_u(t \mid z)
$$

where $\pi(x) = \text{expit}(x'\gamma)$ is the probability of being susceptible (uncured)
and $S_u(t \mid z)$ is the latency survival for susceptible subjects.

Estimation uses the EM algorithm. Validated against R's `smcure` package.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt
from scipy.optimize import minimize
from scipy.stats import norm as norm_dist

from ._backends import to_dataframe

if TYPE_CHECKING:
    from ._surv import Surv

__all__ = ["MixtureCure"]

Array = npt.NDArray[Any]


def _expit(x: Array) -> npt.NDArray[np.floating[Any]]:
    result: npt.NDArray[np.floating[Any]] = 1.0 / (1.0 + np.exp(-x))
    return result


def _logistic_negll(gamma: Array, z: Array, w: Array) -> float:
    """Negative quasi-binomial log-likelihood for the incidence submodel."""
    eta = z @ gamma
    pi_ = _expit(eta)
    pi_ = np.clip(pi_, 1e-15, 1.0 - 1e-15)
    return -float(np.sum(w * np.log(pi_) + (1.0 - w) * np.log(1.0 - pi_)))


def _logistic_grad(gamma: Array, z: Array, w: Array) -> Array:
    """Gradient of the negative quasi-binomial log-likelihood."""
    eta = z @ gamma
    pi_ = _expit(eta)
    return np.asarray(-(z.T @ (w - pi_)))


def _fit_logistic(z: Array, w: Array, gamma_init: Array) -> Array:
    """Fit weighted logistic regression via L-BFGS-B."""
    result = minimize(
        _logistic_negll,
        gamma_init,
        args=(z, w),
        jac=_logistic_grad,
        method="L-BFGS-B",
        options={"ftol": 1e-15, "gtol": 1e-12, "maxiter": 500},
    )
    return np.asarray(result.x, dtype=float)


def _cox_weighted_negpl(beta: Array, x: Array, time: Array, status: Array, w: Array) -> float:
    """Negative weighted Cox partial log-likelihood (Breslow ties).

    Uses offset(log(w)) formulation: each subject's risk contribution is
    w_i * exp(beta' x_i). All events at the same time share the same
    risk set denominator (Breslow method).
    """
    lp = x @ beta
    risk = w * np.exp(lp)

    order = np.argsort(-time, kind="stable")
    time_sorted = time[order]
    risk_sorted = risk[order]
    status_sorted = status[order]
    lp_sorted = lp[order]
    w_sorted = w[order]

    cumrisk = np.cumsum(risk_sorted)

    event_mask = status_sorted == 1
    event_times = time_sorted[event_mask]
    event_lp = lp_sorted[event_mask]
    event_w = w_sorted[event_mask]

    unique_times = np.unique(event_times)
    log_risk_at_time = np.empty(len(unique_times))
    for j, ut in enumerate(unique_times):
        last_idx = np.searchsorted(-time_sorted, -ut, side="right") - 1
        log_risk_at_time[j] = np.log(max(cumrisk[last_idx], 1e-300))

    time_to_logrisk = dict(zip(unique_times, log_risk_at_time, strict=True))
    log_denom = np.array([time_to_logrisk[t] for t in event_times])

    nll = -float(np.sum(np.log(np.maximum(event_w, 1e-300)) + event_lp - log_denom))
    return nll


def _cox_weighted_grad(beta: Array, x: Array, time: Array, status: Array, w: Array) -> Array:
    """Gradient of the negative weighted Cox partial log-likelihood (Breslow ties)."""
    lp = x @ beta
    risk = w * np.exp(lp)

    order = np.argsort(-time, kind="stable")
    time_sorted = time[order]
    risk_sorted = risk[order]
    status_sorted = status[order]
    x_sorted = x[order]

    cumrisk = np.cumsum(risk_sorted)
    weighted_x_cumsum = np.cumsum(risk_sorted[:, np.newaxis] * x_sorted, axis=0)

    event_mask = status_sorted == 1
    event_times = time_sorted[event_mask]
    event_x = x_sorted[event_mask]

    unique_times = np.unique(event_times)
    xbar_at_time: dict[float, Array] = {}
    for ut in unique_times:
        last_idx = int(np.searchsorted(-time_sorted, -ut, side="right")) - 1
        xbar_at_time[ut] = weighted_x_cumsum[last_idx] / max(cumrisk[last_idx], 1e-300)

    xbar = np.array([xbar_at_time[t] for t in event_times])
    grad = -(event_x - xbar).sum(axis=0)
    return grad


def _fit_cox_weighted(x: Array, time: Array, status: Array, w: Array, beta_init: Array) -> Array:
    """Fit weighted Cox PH via L-BFGS-B."""
    mask = w > 0
    x_sub = x[mask]
    time_sub = time[mask]
    status_sub = status[mask]
    w_sub = w[mask]

    result = minimize(
        _cox_weighted_negpl,
        beta_init,
        args=(x_sub, time_sub, status_sub, w_sub),
        jac=_cox_weighted_grad,
        method="L-BFGS-B",
        options={"ftol": 1e-15, "gtol": 1e-12, "maxiter": 500},
    )
    return np.asarray(result.x, dtype=float)


def _breslow_baseline(
    time: Array, status: Array, x: Array, beta: Array, w: Array
) -> tuple[Array, Array]:
    """Weighted Breslow baseline survival at unique event times.

    Returns (unique_event_times, baseline_survival_at_those_times).
    The numerator counts raw events (not weighted), while the denominator
    uses w_i * exp(beta' x_i) as risk weights.
    """
    risk = w * np.exp(x @ beta)
    death_times = np.sort(np.unique(time[status == 1]))

    lambdas = np.empty(len(death_times))
    for i, dt in enumerate(death_times):
        d_i = np.sum(status[time == dt])
        at_risk = time >= dt
        lambdas[i] = d_i / np.maximum(np.sum(risk[at_risk]), 1e-300)

    cum_hazard = np.cumsum(lambdas)
    baseline_surv = np.exp(-cum_hazard)
    return death_times, baseline_surv


def _baseline_survival_at(t: Array, death_times: Array, baseline_surv: Array) -> Array:
    """Evaluate baseline survival S_0(t) for arbitrary time points."""
    out = np.ones(len(t))
    for i, ti in enumerate(t):
        if ti >= death_times[-1]:
            if ti > death_times[-1]:
                out[i] = 0.0
            else:
                out[i] = baseline_surv[-1]
        elif ti < death_times[0]:
            out[i] = 1.0
        else:
            idx = np.searchsorted(death_times, ti, side="right") - 1
            out[i] = baseline_surv[idx]
    return out


class MixtureCure:
    r"""Mixture cure model with logistic incidence and Cox PH latency.

    The population survival is decomposed as:

    $$
    S_{\text{pop}}(t \mid x, z) = 1 - \pi(x) + \pi(x)\,S_u(t \mid z)
    $$

    where $\pi(x) = \text{expit}(\gamma_0 + \gamma' x)$ is the probability of being
    susceptible (uncured) and $S_u(t \mid z) = S_0(t)^{\exp(\beta' z)}$ is the latency
    survival under a Cox proportional hazards model.

    Parameters are estimated via the EM algorithm.

    Parameters
    ----------
    emmax
        Maximum number of EM iterations (default 50).
    eps
        Convergence tolerance for the EM algorithm (default 1e-7).

    Examples
    --------
    Fit a mixture cure model on the e1684 melanoma dataset:

    ```{python}
    import greenwood as gw

    e1684 = gw.load_dataset("e1684", backend="polars")
    y = gw.Surv.right(e1684["FAILTIME"], event=e1684["FAILCENS"])

    cure = gw.MixtureCure().fit(y, latency=e1684[["TRT"]], cure=e1684[["TRT"]])
    cure
    ```
    """

    def __init__(
        self,
        *,
        emmax: int = 50,
        eps: float = 1e-7,
    ) -> None:
        self.emmax = emmax
        self.eps = eps

    def __repr__(self) -> str:
        if getattr(self, "cure_coef_", None) is None:
            return "MixtureCure() <unfitted>"
        from ._repr import align_table, num

        cure_rows = [
            [num(c), num(se), num(z), num(p, digits=3)]
            for c, se, z, p in zip(
                self.cure_coef_,
                self.cure_se_,
                self.cure_z_,
                self.cure_p_,
                strict=True,
            )
        ]
        cure_table = align_table(
            ["coef", "se(coef)", "z", "p"],
            cure_rows,
            list(self.cure_term_names_),
        )

        lat_rows = [
            [num(c), num(se), num(z), num(p, digits=3)]
            for c, se, z, p in zip(
                self.latency_coef_,
                self.latency_se_,
                self.latency_z_,
                self.latency_p_,
                strict=True,
            )
        ]
        lat_table = align_table(
            ["coef", "se(coef)", "z", "p"],
            lat_rows,
            list(self.latency_term_names_),
        )

        return "\n".join(
            [
                "MixtureCure (logistic incidence + Cox PH latency)",
                "",
                "Cure probability model:",
                cure_table,
                "",
                "Failure time distribution model:",
                lat_table,
                "",
                f"n = {self.n_}, events = {self.n_event_}, EM iterations = {self.n_iter_}",
            ]
        )

    def fit(
        self,
        surv: Surv,
        latency: Any,
        cure: Any,
        *,
        data: Any = None,
        nboot: int = 100,
    ) -> MixtureCure:
        r"""Fit the mixture cure model via EM.

        Parameters
        ----------
        surv
            A right-censored `Surv` response.
        latency
            Covariates for the latency (survival) submodel. A dataframe, 2-D array,
            or formula string evaluated against `data`.
        cure
            Covariates for the incidence (cure) submodel. A dataframe, 2-D array,
            or formula string evaluated against `data`.
        data
            DataFrame for formula evaluation.
        nboot
            Number of bootstrap resamples for standard errors (default 100).

        Returns
        -------
        MixtureCure
            The fitted estimator.
        """
        from ._cox import _design_matrix
        from ._surv import CensoringType

        if surv.type != CensoringType.RIGHT:
            raise NotImplementedError(
                f"MixtureCure supports right-censored responses, not {surv.type.value!r}."
            )

        x_raw, latency_names = _design_matrix(latency, data)
        z_raw, cure_names = _design_matrix(cure, data)

        if x_raw.shape[0] != surv.n:
            raise ValueError("Latency covariates and response must have the same number of rows.")
        if z_raw.shape[0] != surv.n:
            raise ValueError("Cure covariates and response must have the same number of rows.")

        keep = ~(np.isnan(x_raw).any(axis=1) | np.isnan(z_raw).any(axis=1))
        x_raw = x_raw[keep]
        z_raw = z_raw[keep]
        time = surv.stop[keep]
        status = surv.event[keep].astype(float)

        n = x_raw.shape[0]
        if not status.any():
            raise ValueError("No events in the data.")

        z = np.column_stack([np.ones(n), z_raw])
        cure_term_names = ["(Intercept)"] + list(cure_names)
        latency_term_names = list(latency_names)

        gamma, beta, base_times, base_surv, n_iter = _em_fit(
            time, status, x_raw, z, self.emmax, self.eps
        )

        cure_se, latency_se = _bootstrap_se(
            time, status, x_raw, z, gamma, beta, nboot, self.emmax, self.eps
        )

        cure_z = gamma / cure_se
        cure_p: Array = 2.0 * norm_dist.sf(np.abs(cure_z))
        latency_z = beta / latency_se
        latency_p: Array = 2.0 * norm_dist.sf(np.abs(latency_z))

        self.cure_coef_ = gamma
        self.cure_se_ = cure_se
        self.cure_z_ = cure_z
        self.cure_p_ = cure_p
        self.cure_term_names_ = cure_term_names

        self.latency_coef_ = beta
        self.latency_se_ = latency_se
        self.latency_z_ = latency_z
        self.latency_p_ = latency_p
        self.latency_term_names_ = latency_term_names

        self.baseline_times_ = base_times
        self.baseline_survival_ = base_surv

        self.n_ = n
        self.n_event_ = int(status.sum())
        self.n_iter_ = n_iter

        self._time = time
        self._status = status
        self._x = x_raw
        self._z = z

        return self

    def predict_cure_prob(self, cure_covariates: Any, *, data: Any = None) -> Array:
        """Predict the probability of being susceptible (uncured).

        Parameters
        ----------
        cure_covariates
            Covariates for the incidence submodel (same format as `cure` in `fit`).
        data
            DataFrame for formula evaluation.

        Returns
        -------
        numpy.ndarray
            Array of susceptibility probabilities (1 = certainly uncured).
        """
        from ._cox import _design_matrix

        z_raw, _ = _design_matrix(cure_covariates, data)
        z = np.column_stack([np.ones(z_raw.shape[0]), z_raw])
        return _expit(z @ self.cure_coef_)

    def predict_survival(
        self,
        times: Any,
        latency_covariates: Any,
        cure_covariates: Any,
        *,
        data: Any = None,
    ) -> Array:
        r"""Predict population survival $S_{\text{pop}}(t \mid x, z)$.

        Parameters
        ----------
        times
            Time points at which to evaluate survival.
        latency_covariates
            Covariates for the latency submodel.
        cure_covariates
            Covariates for the incidence submodel.
        data
            DataFrame for formula evaluation.

        Returns
        -------
        numpy.ndarray
            Array of shape `(n_subjects, n_times)` with population survival probabilities.
        """
        from ._cox import _design_matrix

        t = np.atleast_1d(np.asarray(times, dtype=float))
        x_raw, _ = _design_matrix(latency_covariates, data)
        z_raw, _ = _design_matrix(cure_covariates, data)
        z = np.column_stack([np.ones(z_raw.shape[0]), z_raw])

        pi_ = _expit(z @ self.cure_coef_)
        lp = x_raw @ self.latency_coef_

        s0 = _baseline_survival_at(t, self.baseline_times_, self.baseline_survival_)

        n_subj = x_raw.shape[0]
        n_t = len(t)
        result = np.empty((n_subj, n_t))
        for i in range(n_subj):
            s_u = s0 ** np.exp(lp[i])
            result[i] = (1.0 - pi_[i]) + pi_[i] * s_u
        return result

    def to_frame(self, *, format: str | None = None) -> Any:
        """Return the summary table as a DataFrame.

        Two sections (cure and latency) are stacked, distinguished by a `submodel` column.

        Parameters
        ----------
        format
            Output format: `None` (default), `"pandas"`, `"polars"`, or `"pyarrow"`.

        Returns
        -------
        DataFrame
            Columns: `submodel`, `term`, `coef`, `se`, `z`, `p`.
        """
        terms = list(self.cure_term_names_) + list(self.latency_term_names_)
        submodels = ["cure"] * len(self.cure_term_names_) + ["latency"] * len(
            self.latency_term_names_
        )
        coefs = np.concatenate([self.cure_coef_, self.latency_coef_])
        ses = np.concatenate([self.cure_se_, self.latency_se_])
        zs = np.concatenate([self.cure_z_, self.latency_z_])
        ps = np.concatenate([self.cure_p_, self.latency_p_])

        return to_dataframe(
            {
                "submodel": submodels,
                "term": terms,
                "coef": coefs,
                "se": ses,
                "z": zs,
                "p": ps,
            },
            format=format,
        )


def _em_fit(
    time: Array,
    status: Array,
    x: Array,
    z: Array,
    emmax: int,
    eps: float,
) -> tuple[Array, Array, Array, Array, int]:
    """Run the EM algorithm for the mixture cure model (PH latency).

    Returns (gamma, beta, baseline_times, baseline_survival, n_iterations).
    """
    w = status.copy()
    gamma = _fit_logistic(z, w, np.zeros(z.shape[1]))
    mask_w = w > 0
    beta_init = np.zeros(x.shape[1])
    beta = _fit_cox_weighted(x[mask_w], time[mask_w], status[mask_w], w[mask_w], beta_init)
    base_times, base_surv = _breslow_baseline(time, status, x, beta, w)
    s_per_subject = _baseline_survival_at(time, base_times, base_surv)

    for i in range(emmax):
        uncure_prob = _expit(z @ gamma)
        surv_latency = s_per_subject ** np.exp(x @ beta)

        w_new = status + (1.0 - status) * (uncure_prob * surv_latency) / (
            (1.0 - uncure_prob) + uncure_prob * surv_latency
        )

        gamma_new = _fit_logistic(z, w_new, gamma)
        beta_new = _fit_cox_weighted(x, time, status, w_new, beta)
        base_times_new, base_surv_new = _breslow_baseline(time, status, x, beta, w_new)
        s_new = _baseline_survival_at(time, base_times_new, base_surv_new)

        convergence = float(
            np.sum((gamma_new - gamma) ** 2)
            + np.sum((beta_new - beta) ** 2)
            + np.sum((s_new - s_per_subject) ** 2)
        )

        gamma = gamma_new
        beta = beta_new
        base_times = base_times_new
        base_surv = base_surv_new
        s_per_subject = s_new

        if convergence <= eps:
            return gamma, beta, base_times, base_surv, i + 1

    return gamma, beta, base_times, base_surv, emmax


def _bootstrap_se(
    time: Array,
    status: Array,
    x: Array,
    z: Array,
    gamma_init: Array,
    beta_init: Array,
    nboot: int,
    emmax: int,
    eps: float,
) -> tuple[Array, Array]:
    """Compute standard errors via stratified bootstrap (events/censored separately)."""
    rng = np.random.default_rng(seed=42)
    event_idx = np.nonzero(status == 1)[0]
    censor_idx = np.nonzero(status == 0)[0]
    n1, n0 = len(event_idx), len(censor_idx)

    gamma_boots: list[Array] = []
    beta_boots: list[Array] = []

    attempts = 0
    while len(gamma_boots) < nboot and attempts < nboot * 5:
        attempts += 1
        boot_e = rng.choice(event_idx, size=n1, replace=True)
        boot_c = rng.choice(censor_idx, size=n0, replace=True)
        boot_idx = np.concatenate([boot_e, boot_c])

        try:
            g, b, _, _, _ = _em_fit(
                time[boot_idx],
                status[boot_idx],
                x[boot_idx],
                z[boot_idx],
                emmax,
                eps,
            )
            gamma_boots.append(g)
            beta_boots.append(b)
        except Exception:
            continue

    if len(gamma_boots) < 2:
        return np.full(len(gamma_init), np.nan), np.full(len(beta_init), np.nan)

    gamma_arr = np.array(gamma_boots)
    beta_arr = np.array(beta_boots)
    return np.std(gamma_arr, axis=0, ddof=1), np.std(beta_arr, axis=0, ddof=1)


def _tidy_cure(model: MixtureCure, *, format: str | None = None, **_: Any) -> Any:
    return model.to_frame(format=format)


def _glance_cure(model: MixtureCure, *, format: str | None = None, **_: Any) -> Any:
    return to_dataframe(
        {
            "n": [model.n_],
            "nevent": [model.n_event_],
            "n_iter": [model.n_iter_],
        },
        format=format,
    )


def _register_adapters() -> None:
    from .summaries import register_glance, register_tidier

    register_tidier("greenwood._cure.MixtureCure", _tidy_cure)
    register_glance("greenwood._cure.MixtureCure", _glance_cure)


_register_adapters()
