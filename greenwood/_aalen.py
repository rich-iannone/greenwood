r"""Aalen additive hazards model.

The Aalen additive model specifies the hazard as a linear (additive) function of covariates
with time-varying coefficients:

$$
h(t \mid x) = \beta_0(t) + \beta_1(t)\,x_1 + \cdots + \beta_p(t)\,x_p
$$

Unlike the Cox model, which assumes a multiplicative (proportional) effect, the Aalen model
allows each covariate to have a completely non-parametric, time-varying influence on the hazard.
The primary estimands are the cumulative regression coefficients
$B_j(t) = \int_0^t \beta_j(s)\,ds$, estimated by ordinary least squares at each event time.

Validated against R's `survival::aareg`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt
from scipy.stats import norm as norm_dist

from ._backends import to_dataframe

if TYPE_CHECKING:
    from ._surv import Surv

__all__ = ["AalenAdditive"]

Array = npt.NDArray[Any]


class AalenAdditive:
    r"""Aalen additive hazards model with time-varying coefficients.

    The additive hazards model is an alternative to Cox regression when the proportional
    hazards assumption fails. Instead of modeling the log-hazard ratio as constant over time,
    it models the hazard itself as a sum of covariate-specific, time-varying functions:

    $$
    h(t \mid x) = \beta_0(t) + \sum_{j=1}^{p} \beta_j(t)\,x_j
    $$

    Estimation proceeds by ordinary least squares at each event time: for each death, the
    coefficient increment $d\hat{B}(t)$ is obtained by regressing the event indicator on
    the at-risk covariate matrix. Cumulative coefficients $\hat{B}(t)$ are obtained by
    summing these increments.

    Parameters
    ----------
    nmin
        Minimum risk-set size. Event times where the risk set has fewer than `nmin`
        subjects are dropped (late-tail instability). Defaults to `3 * (p + 1)` where
        `p` is the number of covariates.
    test
        Weighting scheme for the summary test statistics. `"aalen"` (default) uses
        equal weights, `"variance"` uses inverse-variance weights, and `"nrisk"` uses
        the number at risk.
    qrtol
        QR decomposition tolerance for detecting rank deficiency (default `1e-7`).

    Examples
    --------
    Fit the Aalen additive model on the lung dataset:

    ```{python}
    import greenwood as gw

    lung = gw.load_dataset("lung", backend="polars")
    y = gw.Surv.right(lung["time"], event=(lung["status"] == 2))

    aalen = gw.AalenAdditive().fit(y, lung[["age", "sex"]])
    aalen
    ```
    """

    def __init__(
        self,
        *,
        nmin: int | None = None,
        test: str = "aalen",
        qrtol: float = 1e-7,
    ) -> None:
        if test not in ("aalen", "variance", "nrisk"):
            raise ValueError(f"test must be 'aalen', 'variance', or 'nrisk', got {test!r}.")
        self.nmin = nmin
        self.test = test
        self.qrtol = qrtol

    def __repr__(self) -> str:
        if getattr(self, "coef_increments_", None) is None:
            return "AalenAdditive() <unfitted>"
        from ._repr import align_table, num

        rows = [
            [num(sl), num(co), num(se), num(z), num(p, digits=3)]
            for sl, co, se, z, p in zip(
                self.summary_slope_,
                self.summary_coef_,
                self.summary_se_,
                self.summary_z_,
                self.summary_p_,
            )
        ]
        table = align_table(
            ["slope", "coef", "se(coef)", "z", "p"],
            rows,
            list(self.term_names_),
        )
        return "\n".join(
            [
                f"AalenAdditive (additive hazards, test={self.test!r})",
                "",
                table,
                "",
                f"n = {self.n_}, events = {self.n_event_}, "
                f"event times used = {self.n_event_times_used_}",
            ]
        )

    def fit(
        self, surv: Surv, covariates: Any, *, data: Any = None
    ) -> AalenAdditive:
        r"""Fit the Aalen additive hazards model.

        At each event time, an OLS regression of the event indicator on the at-risk design
        matrix produces one coefficient increment per covariate. The cumulative sum of
        these increments gives the cumulative regression function $\hat{B}(t)$.

        Parameters
        ----------
        surv
            A right-censored or counting-process `Surv` response.
        covariates
            A dataframe (pandas or polars), 2-D array, or formula string evaluated against
            `data`.
        data
            DataFrame for formula evaluation (ignored if `covariates` is a dataframe or array).

        Returns
        -------
        AalenAdditive
            The fitted estimator.
        """
        from ._cox import _design_matrix
        from ._surv import CensoringType

        if surv.type not in (CensoringType.RIGHT, CensoringType.COUNTING):
            raise NotImplementedError(
                f"AalenAdditive supports right-censored and counting-process responses, "
                f"not {surv.type.value!r}."
            )

        x_raw, covariate_names = _design_matrix(covariates, data)
        if x_raw.shape[0] != surv.n:
            raise ValueError("Covariates and response must have the same number of rows.")

        keep = ~np.isnan(x_raw).any(axis=1)
        x_raw = x_raw[keep]
        entry = surv.entry[keep]
        exit_ = surv.stop[keep]
        event = surv.event[keep].astype(bool)

        if not event.any():
            raise ValueError("No events remain after dropping missing rows.")

        n = x_raw.shape[0]
        p = x_raw.shape[1]
        nmin = self.nmin if self.nmin is not None else 3 * p

        term_names = ["Intercept"] + list(covariate_names)
        pp1 = p + 1

        event_order = np.argsort(exit_[event], kind="stable")
        event_indices = np.nonzero(event)[0][event_order]

        times_list: list[float] = []
        nrisk_list: list[int] = []
        increments_list: list[Array] = []
        tweight_list: list[Array] = []

        cached_time: float | None = None
        cached_vmat: Array | None = None
        cached_means: Array | None = None
        cached_nrisk: int = 0
        cached_twt: Array = np.zeros(pp1)
        cached_x_centered: Array = np.zeros((0, p))
        risk_mask: Array = np.zeros(0, dtype=bool)

        for idx in event_indices:
            t = float(exit_[idx])

            if t != cached_time:
                at_risk = (entry < t) & (exit_ >= t)
                n_risk = int(at_risk.sum())
                if n_risk < nmin:
                    cached_time = t
                    cached_vmat = None
                    continue

                x_risk = x_raw[at_risk]
                means = x_risk.mean(axis=0)
                x_centered = x_risk - means
                cov_x = (x_centered.T @ x_centered) / n_risk

                try:
                    vmat = np.linalg.solve(cov_x, np.eye(p))
                except np.linalg.LinAlgError:
                    cached_time = t
                    cached_vmat = None
                    continue

                diag_v = np.diag(vmat)
                if np.any(diag_v <= 0):
                    cached_time = t
                    cached_vmat = None
                    continue

                twt_intercept = n_risk / (1.0 + float(means @ vmat @ means))
                twt_covs = n_risk / diag_v
                twt = np.concatenate([[twt_intercept], twt_covs])

                cached_time = t
                cached_vmat = vmat
                cached_means = means
                cached_nrisk = n_risk
                risk_mask = at_risk
                cached_twt = twt
                cached_x_centered = x_centered
            else:
                if cached_vmat is None:
                    continue

            vmat = cached_vmat
            means = cached_means
            assert means is not None
            n_risk = cached_nrisk

            local_idx = int(np.searchsorted(np.nonzero(risk_mask)[0], idx))

            x_death_centered = cached_x_centered[local_idx]
            coef_slopes = (vmat @ x_death_centered) / n_risk
            b0 = 1.0 / n_risk - float(means @ coef_slopes)
            db = np.concatenate([[b0], coef_slopes])

            times_list.append(t)
            nrisk_list.append(n_risk)
            increments_list.append(db)
            tweight_list.append(cached_twt)

        if not increments_list:
            raise ValueError("No usable event times (risk sets too small or rank-deficient).")

        self.event_times_ = np.array(times_list)
        self.nrisk_ = np.array(nrisk_list, dtype=int)
        self.coef_increments_ = np.array(increments_list)
        self.cumulative_coefs_ = np.cumsum(self.coef_increments_, axis=0)
        tweight = np.array(tweight_list)
        self.tweight_ = tweight

        n_events_used = len(times_list)

        if self.test == "nrisk":
            tw = self.nrisk_.astype(float)[:, np.newaxis] * np.ones((1, pp1))
        else:
            tw = tweight

        tx = tw * self.coef_increments_
        test_statistic = tx.sum(axis=0)
        test_var: Array = tx.T @ tx

        scale = tw.sum(axis=0)

        ctx = np.cumsum(tx, axis=0)
        times_col = self.event_times_
        if tw.ndim == 2 and tw.shape[1] > 1:
            tempwt = (tw * times_col[:, np.newaxis] ** 2).sum(axis=0)
        else:
            tempwt = np.array([(tw.ravel() * times_col**2).sum()])
        with np.errstate(divide="ignore", invalid="ignore"):
            slope = np.where(
                tempwt > 0,
                (ctx * times_col[:, np.newaxis]).sum(axis=0) / tempwt,
                0.0,
            )

        se1 = np.sqrt(np.diag(test_var))
        with np.errstate(divide="ignore", invalid="ignore"):
            coef = np.where(scale > 0, test_statistic / scale, 0.0)
            se_coef = np.where(scale > 0, se1 / scale, 0.0)
            z = np.where(se1 > 0, test_statistic / se1, 0.0)
        p_values: Array = 2.0 * norm_dist.sf(np.abs(z))

        self.term_names_ = term_names
        self.n_ = n
        self.n_event_ = int(event.sum())
        self.n_unique_event_times_ = len(np.unique(exit_[event]))
        self.n_event_times_used_ = n_events_used
        self.summary_slope_ = slope
        self.summary_coef_ = coef
        self.summary_se_ = se_coef
        self.summary_z_ = z
        self.summary_p_ = p_values
        self.test_statistic_ = test_statistic
        self.test_var_ = test_var

        self._x = x_raw
        self._entry = entry
        self._exit = exit_
        self._event = event

        return self

    def cumulative_coefficients(self, *, format: str | None = None) -> Any:
        r"""Return the cumulative regression coefficients $\hat{B}(t)$ as a DataFrame.

        Each row corresponds to an event time. The cumulative coefficient for covariate $j$
        at time $t$ is $\hat{B}_j(t) = \sum_{s \le t} d\hat{B}_j(s)$, the running total
        of OLS increments up to time $t$.

        Parameters
        ----------
        format
            Output format: `None` (default), `"pandas"`, `"polars"`, or `"pyarrow"`.

        Returns
        -------
        DataFrame
            Columns: `time` and one column per term (Intercept plus covariates).
        """
        columns: dict[str, Any] = {"time": self.event_times_}
        for j, name in enumerate(self.term_names_):
            columns[name] = self.cumulative_coefs_[:, j]
        return to_dataframe(columns, format=format)

    def to_frame(self, *, format: str | None = None) -> Any:
        """Return the summary test table as a DataFrame.

        One row per term with the slope (weighted average of increments), its standard error,
        z-statistic, and p-value.

        Parameters
        ----------
        format
            Output format: `None` (default), `"pandas"`, `"polars"`, or `"pyarrow"`.

        Returns
        -------
        DataFrame
            Columns: `term`, `slope`, `coef`, `se`, `z`, `p`.
        """
        return to_dataframe(
            {
                "term": self.term_names_,
                "slope": self.summary_slope_,
                "coef": self.summary_coef_,
                "se": self.summary_se_,
                "z": self.summary_z_,
                "p": self.summary_p_,
            },
            format=format,
        )


def _tidy_aalen(model: AalenAdditive, *, format: str | None = None, **_: Any) -> Any:
    return model.to_frame(format=format)


def _glance_aalen(model: AalenAdditive, *, format: str | None = None, **_: Any) -> Any:
    return to_dataframe(
        {
            "n": [model.n_],
            "nevent": [model.n_event_],
            "n_event_times_used": [model.n_event_times_used_],
            "test": [model.test],
        },
        format=format,
    )


def _register_adapters() -> None:
    from .summaries import register_glance, register_tidier

    register_tidier("greenwood._aalen.AalenAdditive", _tidy_aalen)
    register_glance("greenwood._aalen.AalenAdditive", _glance_aalen)


_register_adapters()
