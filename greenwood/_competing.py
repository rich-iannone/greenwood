"""Competing-risks estimation: the Aalen-Johansen cumulative incidence function.

For competing risks (each subject starts in one state and makes a single transition to one
of several absorbing causes), the cumulative incidence function (CIF) for cause `k` is

    CIF_k(t) = sum_{t_i <= t} S(t_i^-) * d_{ki} / n_i,

where `S` is the all-cause Kaplan-Meier survival, `d_{ki}` the cause-`k` events at `t_i`, and
`n_i` the number at risk. The standard error uses the Aalen (Marubini-Valsecchi)
delta-method estimator. Both are validated to tolerance against R's `survfit`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt
from scipy.stats import norm

if TYPE_CHECKING:
    from ._surv import Surv

__all__ = ["AalenJohansen", "FineGray"]

Array = npt.NDArray[Any]


def _censoring_km(time: Array, cause: Array) -> tuple[Array, Array]:
    """Nudged censoring Kaplan-Meier: (drop times, survival after each drop).

    Events (any cause) are treated as leaving just before a tied censoring, matching R's
    `finegray`. Returns the censoring times where the curve drops and the survival value
    just after each drop.
    """
    censor_times = np.unique(time[cause == 0])
    surv = 1.0
    drop_times: list[float] = []
    drop_surv: list[float] = []
    for c in censor_times:
        # At-risk for censoring excludes events tied at c (they are nudged just before).
        n_risk = float((time > c).sum() + ((cause == 0) & (time == c)).sum())
        d = float(((cause == 0) & (time == c)).sum())
        surv *= 1.0 - d / n_risk
        drop_times.append(float(c))
        drop_surv.append(surv)
    return np.array(drop_times), np.array(drop_surv)


def _cif_block(
    exit_: Array, status: Array, causes: list[int], z: float
) -> dict[int, dict[str, Array]]:
    """Cumulative incidence, delta-method SE, and CI for each cause in one group."""
    times = np.unique(exit_)
    n_risk = np.array([float((exit_ >= t).sum()) for t in times])
    d_any = np.array([float(((exit_ == t) & (status > 0)).sum()) for t in times])

    surv = np.cumprod(1.0 - d_any / n_risk)
    surv_left = np.concatenate(([1.0], surv[:-1]))

    out: dict[int, dict[str, Array]] = {}
    for cause in causes:
        d_k = np.array([float(((exit_ == t) & (status == cause)).sum()) for t in times])
        cif = np.cumsum(surv_left * d_k / n_risk)

        # Aalen (Marubini-Valsecchi) delta-method variance via cumulative sums.
        with np.errstate(divide="ignore", invalid="ignore"):
            a = np.where(n_risk > d_any, d_any / (n_risk * (n_risk - d_any)), 0.0)
        b = surv_left**2 * (n_risk - d_k) / n_risk * d_k / n_risk**2
        c = surv_left * d_k / n_risk**2
        c_a = np.cumsum(a)
        c_ac = np.cumsum(a * cif)
        c_ac2 = np.cumsum(a * cif**2)
        c_b = np.cumsum(b)
        c_c = np.cumsum(c)
        c_cc = np.cumsum(c * cif)
        var = cif**2 * c_a - 2 * cif * c_ac + c_ac2 + c_b - 2 * cif * c_c + 2 * c_cc
        se = np.sqrt(np.clip(var, 0.0, None))

        out[cause] = {
            "time": times,
            "n_risk": n_risk,
            "estimate": cif,
            "std_error": se,
            "conf_low": np.clip(cif - z * se, 0.0, 1.0),
            "conf_high": np.clip(cif + z * se, 0.0, 1.0),
        }
    return out


class AalenJohansen:
    """Aalen-Johansen estimator of cumulative incidence functions for competing risks.

    Parameters
    ----------
    conf_level
        Confidence level for the (Wald) confidence intervals (default 0.95).

    Notes
    -----
    Call `fit(surv, by=...)` with a multi-state `Surv` response (built with
    `Surv.multistate`, where `event` codes are 0 for censoring and `1..K` for the competing
    causes). Results are a tidy frame via `to_dataframe` with one row per stratum, cause, and
    time.
    """

    def __init__(self, *, conf_level: float = 0.95) -> None:
        if not 0.0 < conf_level < 1.0:
            raise ValueError(f"conf_level must be in (0, 1), got {conf_level}.")
        self.conf_level = conf_level

    def fit(self, surv: Surv, *, by: Any = None) -> AalenJohansen:
        """Fit cumulative incidence functions to a competing-risks `Surv` response."""
        if not surv.is_multistate:
            raise ValueError(
                "AalenJohansen needs a multi-state response; build it with Surv.multistate "
                "(use KaplanMeier for a single event type)."
            )
        if surv.is_truncated:
            raise NotImplementedError(
                "Left truncation is not yet supported for cumulative incidence."
            )

        assert surv.states is not None
        self.states_ = surv.states
        causes = list(range(1, len(surv.states) + 1))
        z = float(norm.ppf(1.0 - (1.0 - self.conf_level) / 2.0))

        exit_ = surv.stop
        status = surv.status

        if by is None:
            self._grouped = False
            self._blocks = {None: _cif_block(exit_, status, causes, z)}
        else:
            from ._surv import _to_1d_array

            labels = _to_1d_array(by, dtype=object)
            if labels.shape[0] != surv.n:
                raise ValueError("`by` must have the same length as the response.")
            self._grouped = True
            self._blocks = {}
            for level in dict.fromkeys(labels.tolist()):
                mask = labels == level
                self._blocks[level] = _cif_block(exit_[mask], status[mask], causes, z)
        self._causes = causes
        return self

    def to_dataframe(self, backend: str = "pandas") -> Any:
        """Return a tidy frame: [strata,] cause, time, n_risk, estimate, std_error, CI."""
        cols: dict[str, list[Any]] = {
            k: []
            for k in (
                "strata",
                "cause",
                "time",
                "n_risk",
                "estimate",
                "std_error",
                "conf_low",
                "conf_high",
            )
        }
        for label, block in self._blocks.items():
            for cause in self._causes:
                data = block[cause]
                m = data["time"].shape[0]
                cols["strata"].extend([label] * m)
                cols["cause"].extend([self.states_[cause - 1]] * m)
                for key in ("time", "n_risk", "estimate", "std_error", "conf_low", "conf_high"):
                    cols[key].extend(data[key].tolist())
        if not self._grouped:
            cols.pop("strata")

        if backend == "pandas":
            import pandas as pd

            return pd.DataFrame(cols)
        if backend == "polars":
            import polars as pl

            return pl.DataFrame(cols)
        raise ValueError(f"Unknown backend {backend!r}; use 'pandas' or 'polars'.")


class FineGray:
    """Fine-Gray subdistribution hazard model for a competing-risks endpoint.

    The Fine-Gray model is a Cox-like regression on the subdistribution hazard of a target
    cause. Subjects who experience a competing event remain in the risk set with a
    time-decreasing inverse-probability-of-censoring weight, so the coefficients describe the
    covariate effects on the cumulative incidence of the target cause. Coefficients and both
    the model-based and clustered robust (Lin-Wei) standard errors are validated against R's
    `survival::finegray` plus `coxph`.
    """

    def __init__(self, cause: Any, *, conf_level: float = 0.95) -> None:
        if not 0.0 < conf_level < 1.0:
            raise ValueError(f"conf_level must be in (0, 1), got {conf_level}.")
        self.cause = cause
        self.conf_level = conf_level

    def fit(
        self, surv: Surv, covariates: Any, *, max_iter: int = 30, tol: float = 1e-9
    ) -> FineGray:
        """Fit the model to a competing-risks `Surv` response and a covariate design."""
        from ._cox import _design_matrix

        if not surv.is_multistate:
            raise ValueError(
                "FineGray needs a multi-state response; build it with Surv.multistate."
            )
        assert surv.states is not None
        if self.cause in surv.states:
            target = surv.states.index(self.cause) + 1
        elif isinstance(self.cause, int) and 1 <= self.cause <= len(surv.states):
            target = self.cause
        else:
            raise ValueError(f"cause {self.cause!r} is not one of the states {surv.states}.")

        x, names = _design_matrix(covariates)
        if x.shape[0] != surv.n:
            raise ValueError("Covariates and response must have the same number of rows.")

        time = surv.stop
        cause = surv.status
        keep = ~np.isnan(x).any(axis=1)
        x, time, cause = x[keep], time[keep], cause[keep]

        drop_times, drop_surv = _censoring_km(time, cause)

        def g_before(t: Array) -> Array:
            """Censoring survival just before `t` (drops strictly before `t`).

            Target and competing events are nudged just before their time (as in R's
            `finegray`), so the censoring drop tied at an event time is not counted.
            """
            idx = np.searchsorted(drop_times, t, side="left") - 1
            return np.where(idx >= 0, drop_surv[idx.clip(min=0)], 1.0)

        competing = (cause != target) & (cause != 0)
        g_before_i = g_before(time)  # denominator per subject
        target_times = np.unique(time[cause == target])
        p = x.shape[1]

        def _weights(tj: float) -> Array:
            w = np.zeros(time.shape[0])
            w[time >= tj] = 1.0
            mask = competing & (time < tj)
            w[mask] = float(g_before(np.array([tj]))[0]) / g_before_i[mask]
            return w

        def terms(beta: Array) -> tuple[float, Array, Array]:
            r = np.exp(x @ beta)
            loglik = 0.0
            grad = np.zeros(p)
            info = np.zeros((p, p))
            for tj in target_times:
                w = _weights(float(tj))
                rw = w * r
                s0 = rw.sum()
                s1 = (x * rw[:, None]).sum(axis=0)
                s2 = (x * rw[:, None]).T @ x
                dy = (time == tj) & (cause == target)
                d = float(dy.sum())
                z1 = s1 / s0
                loglik += float((x[dy] @ beta).sum()) - d * np.log(s0)
                grad += x[dy].sum(axis=0) - d * z1
                info += d * (s2 / s0 - np.outer(z1, z1))
            return loglik, grad, info

        beta = np.zeros(p)
        loglik = terms(beta)[0]
        for _ in range(max_iter):
            ll, grad, info = terms(beta)
            step = np.linalg.solve(info, grad)
            beta = beta + step
            new_ll = terms(beta)[0]
            if abs(new_ll - ll) <= tol * (abs(new_ll) + tol):
                loglik = new_ll
                break
            loglik = new_ll

        _, _, info = terms(beta)
        naive_vcov = np.linalg.inv(info)

        # Robust (Lin-Wei) sandwich from per-subject score residuals.
        scores = self._score_residuals(
            beta, x, time, cause, target, target_times, competing, g_before_i, g_before
        )
        robust_vcov = naive_vcov @ (scores.T @ scores) @ naive_vcov

        z = float(norm.ppf(1.0 - (1.0 - self.conf_level) / 2.0))
        self.term_names_ = names
        self.coef_ = beta
        self.hazard_ratio_ = np.exp(beta)
        self.naive_vcov_ = naive_vcov
        self.naive_std_error_ = np.sqrt(np.diag(naive_vcov))
        self.vcov_ = robust_vcov
        self.std_error_ = np.sqrt(np.diag(robust_vcov))
        self.z_ = beta / self.std_error_
        self.p_value_ = 2.0 * norm.sf(np.abs(self.z_))
        self.conf_low_ = beta - z * self.std_error_
        self.conf_high_ = beta + z * self.std_error_
        self.loglik_ = float(loglik)
        self.n_ = int(keep.sum())
        self.n_event_ = int((cause == target).sum())
        return self

    def _score_residuals(
        self,
        beta: Array,
        x: Array,
        time: Array,
        cause: Array,
        target: int,
        target_times: Array,
        competing: Array,
        g_before_i: Array,
        g_before: Any,
    ) -> Array:
        """Per-subject score residuals of the weighted subdistribution partial likelihood."""
        n, p = x.shape
        r = np.exp(x @ beta)
        scores = np.zeros((n, p))
        for tj in target_times:
            w = np.zeros(n)
            w[time >= tj] = 1.0
            mask = competing & (time < tj)
            w[mask] = float(g_before(np.array([tj]))[0]) / g_before_i[mask]
            rw = w * r
            s0 = rw.sum()
            xbar = (x * rw[:, None]).sum(axis=0) / s0
            dy = (time == tj) & (cause == target)
            d = float(dy.sum())
            dlambda = d / s0
            # Event term for the subjects failing (target) at tj.
            scores[dy] += x[dy] - xbar
            # Compensator for every weighted member of the risk set.
            member = w > 0
            scores[member] -= w[member, None] * (x[member] - xbar) * (r[member] * dlambda)[:, None]
        return scores

    def to_dataframe(self, *, exponentiate: bool = False) -> Any:
        """Return a tidy coefficient table (subdistribution hazard ratios if exponentiated)."""
        import pandas as pd

        estimate = self.hazard_ratio_ if exponentiate else self.coef_
        low = np.exp(self.conf_low_) if exponentiate else self.conf_low_
        high = np.exp(self.conf_high_) if exponentiate else self.conf_high_
        return pd.DataFrame(
            {
                "term": self.term_names_,
                "estimate": estimate,
                "std_error": self.std_error_,
                "statistic": self.z_,
                "p_value": self.p_value_,
                "conf_low": low,
                "conf_high": high,
            }
        )


def _register_finegray() -> None:
    from .tidy import register_glance, register_tidier

    def _tidy(model: FineGray, *, exponentiate: bool = False, **_: Any) -> Any:
        return model.to_dataframe(exponentiate=exponentiate)

    def _glance(model: FineGray, **_: Any) -> Any:
        import pandas as pd

        return pd.DataFrame([{"n": model.n_, "nevent": model.n_event_, "loglik": model.loglik_}])

    register_tidier("greenwood._competing.FineGray", _tidy)
    register_glance("greenwood._competing.FineGray", _glance)


_register_finegray()
