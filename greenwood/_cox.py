"""Cox proportional hazards regression.

`CoxPH` fits the semiparametric proportional-hazards model by maximizing the partial
likelihood with Newton-Raphson, supporting the Efron (default) and Breslow tie corrections.
It reports coefficients, hazard ratios, model-based standard errors, Wald z-tests, and the
three global tests (likelihood-ratio, Wald, score), all validated to tolerance against R's
`survival::coxph`.

It also provides the baseline (Breslow/Efron) cumulative hazard, survival prediction,
martingale and Schoenfeld residuals, the Grambsch-Therneau proportional-hazards test
(`cox_zph`), and the concordance index, all validated against R.

Stratification (per-stratum baselines with shared coefficients) and the robust (Lin-Wei)
sandwich variance, with optional clustering, are supported. The risk sets use the same
entry/exit convention as the rest of Greenwood, so left truncation and counting-process
data are handled.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt
from scipy.stats import chi2, norm

if TYPE_CHECKING:
    from ._surv import Surv

__all__ = ["CoxPH", "ZPHResult"]

Array = npt.NDArray[Any]


@dataclass(frozen=True)
class ZPHResult:
    """Proportional-hazards test results (Grambsch-Therneau).

    `per_term` maps each covariate to a `{chisq, df, p_value}` dict; `global_test` holds
    the same for the overall test.
    """

    transform: str
    per_term: dict[str, dict[str, float]]
    global_test: dict[str, float]

    def __repr__(self) -> str:
        rows = ", ".join(f"{k}: p={v['p_value']:.4g}" for k, v in self.per_term.items())
        return (
            f"ZPHResult(transform={self.transform!r}, {rows}, "
            f"GLOBAL p={self.global_test['p_value']:.4g})"
        )

    def to_dataframe(self) -> Any:
        """Return the test table (one row per term plus GLOBAL)."""
        import pandas as pd

        rows = [{"term": k, **v} for k, v in self.per_term.items()]
        rows.append({"term": "GLOBAL", **self.global_test})
        return pd.DataFrame(rows)


_TIES = frozenset({"efron", "breslow"})


def _missing_mask(labels: Array) -> Array:
    """Boolean mask of missing entries (None or NaN) in an object array."""
    return np.array([v is None or (isinstance(v, float) and v != v) for v in labels], dtype=bool)


def _to_labels(values: Any, n: int, name: str) -> Array:
    """Coerce group labels (narwhals series, array, or sequence) to a length-`n` array."""
    from ._surv import _to_1d_array

    labels = _to_1d_array(values, dtype=object)
    if labels.shape[0] != n:
        raise ValueError(f"`{name}` must have the same length as the response ({n}).")
    return labels


def _design_matrix(covariates: Any) -> tuple[Array, list[str]]:
    """Build a numeric design matrix and term names from covariates.

    Accepts a 2-D NumPy array or any narwhals-compatible dataframe. Numeric columns pass
    through; non-numeric columns are treatment-coded (drop-first dummies) with names like
    `celltypesmallcell`.
    """
    if isinstance(covariates, np.ndarray):
        x = np.asarray(covariates, dtype=float)
        if x.ndim != 2:
            raise ValueError("A covariate array must be 2-D (n_obs x n_features).")
        return x, [f"x{i}" for i in range(x.shape[1])]

    import narwhals as nw  # pyright: ignore[reportMissingImports]  # installed + typed; pyright quirk

    frame = nw.from_native(covariates, eager_only=True)
    columns: list[Array] = []
    names: list[str] = []
    for name in frame.columns:
        values = frame[name].to_numpy()
        if values.dtype.kind in "iufb":
            columns.append(values.astype(float))
            names.append(name)
        else:
            levels = sorted({v for v in values.tolist() if v is not None})
            for level in levels[1:]:  # drop the first level as the reference
                columns.append((values == level).astype(float))
                names.append(f"{name}{level}")
    if not columns:
        raise ValueError("No covariates found.")
    return np.column_stack(columns), names


def _cox_terms(
    beta: Array,
    x: Array,
    entry: Array,
    exit_: Array,
    event: Array,
    weight: Array,
    strata_groups: list[tuple[Array, Array]],
    ties: str,
) -> tuple[float, Array, Array]:
    """Partial log-likelihood, gradient, and observed information at `beta`.

    `strata_groups` is a list of `(member_index, event_times)` pairs; risk sets are
    confined to a stratum, and the terms are summed across strata (the coefficients are
    shared). An unstratified model is a single group.
    """
    p = beta.shape[0]
    loglik = 0.0
    grad = np.zeros(p)
    info = np.zeros((p, p))

    for members, event_times in strata_groups:
        xs = x[members]
        es = entry[members]
        xx = exit_[members]
        ev = event[members]
        ws = weight[members]
        eta = xs @ beta
        risk_score = np.exp(eta) * ws

        for t in event_times:
            at_risk = (es < t) & (xx >= t)
            dying = (xx == t) & ev

            rx = xs[at_risk]
            rr = risk_score[at_risk]
            s0 = rr.sum()
            s1 = rx.T @ rr
            s2 = (rx * rr[:, None]).T @ rx

            w_d = ws[dying]
            loglik += float((w_d * eta[dying]).sum())
            grad += (xs[dying] * w_d[:, None]).sum(axis=0)

            if ties == "breslow":
                d_weight = float(w_d.sum())
                z1 = s1 / s0
                loglik -= d_weight * np.log(s0)
                grad -= d_weight * z1
                info += d_weight * (s2 / s0 - np.outer(z1, z1))
            else:  # efron
                dx = xs[dying]
                dr = risk_score[dying]
                d0 = dr.sum()
                d1 = dx.T @ dr
                d2 = (dx * dr[:, None]).T @ dx
                m = int(dying.sum())
                for tie in range(m):
                    f = tie / m
                    denom = s0 - f * d0
                    z1 = (s1 - f * d1) / denom
                    z2 = (s2 - f * d2) / denom
                    loglik -= float(np.log(denom))
                    grad -= z1
                    info += z2 - np.outer(z1, z1)

    return loglik, grad, info


class CoxPH:
    """Cox proportional hazards model.

    Parameters
    ----------
    ties
        Tie-handling method: `"efron"` (default, as in R) or `"breslow"`.
    conf_level
        Confidence level for coefficient and hazard-ratio intervals (default 0.95).

    Notes
    -----
    Call `fit(surv, covariates)` with a `Surv` response and a design (a 2-D array or a
    dataframe of covariates). Rows with missing values are dropped (complete-case, as in R's
    default `na.omit`). Results are exposed as arrays (`coef_`, `std_error_`, `hazard_ratio_`,
    …) and as a tidy frame via `to_dataframe`, and feed `greenwood.tidy`.
    """

    def __init__(self, *, ties: str = "efron", conf_level: float = 0.95) -> None:
        if ties not in _TIES:
            raise ValueError(f"ties must be one of {sorted(_TIES)}, got {ties!r}.")
        if not 0.0 < conf_level < 1.0:
            raise ValueError(f"conf_level must be in (0, 1), got {conf_level}.")
        self.ties = ties
        self.conf_level = conf_level

    def __repr__(self) -> str:
        if getattr(self, "coef_", None) is None:
            return f"CoxPH(ties={self.ties!r}, conf_level={self.conf_level}) <unfitted>"
        from scipy.stats import chi2

        from ._repr import align_table, fixed, num

        rows = [
            [num(c), num(hr), num(se), fixed(z, 3), num(p)]
            for c, hr, se, z, p in zip(
                self.coef_, self.hazard_ratio_, self.std_error_, self.z_, self.p_value_, strict=True
            )
        ]
        table = align_table(
            ["coef", "exp(coef)", "se(coef)", "z", "p"], rows, list(self.term_names_)
        )
        lr_p = float(chi2.sf(self.lr_stat_, self.df_))
        lines = [
            f"CoxPH (Cox proportional hazards model, ties={self.ties!r})",
            "",
            table,
            "",
            f"n = {self.n_}, events = {self.n_event_}",
            f"Likelihood ratio test = {num(self.lr_stat_)} on {self.df_} df, p = {num(lr_p)}",
        ]
        if self.robust:
            lines.append("Standard errors: robust (sandwich)")
        return "\n".join(lines)

    def fit(
        self,
        surv: Surv,
        covariates: Any,
        *,
        strata: Any = None,
        robust: bool = False,
        cluster: Any = None,
        max_iter: int = 30,
        tol: float = 1e-9,
    ) -> CoxPH:
        """Fit the model to a `Surv` response and a covariate design.

        `strata` gives per-stratum baseline hazards with shared coefficients. `robust=True`
        (or providing `cluster` ids) reports the Lin-Wei sandwich variance; `cluster` sums
        the score residuals within groups before forming the sandwich.
        """
        from ._surv import CensoringType

        if surv.type not in (CensoringType.RIGHT, CensoringType.COUNTING):
            raise NotImplementedError(
                f"CoxPH supports right-censored and counting-process responses, "
                f"not {surv.type.value!r}."
            )

        x, names = _design_matrix(covariates)
        if x.shape[0] != surv.n:
            raise ValueError("Covariates and response must have the same number of rows.")

        entry = surv.entry
        exit_ = surv.stop
        event = surv.event
        weight = surv.weights if surv.weights is not None else np.ones(surv.n)

        strata_labels = None if strata is None else _to_labels(strata, surv.n, "strata")
        cluster_labels = None if cluster is None else _to_labels(cluster, surv.n, "cluster")

        # Complete-case analysis: drop rows with any missing covariate, stratum, or cluster.
        keep = ~np.isnan(x).any(axis=1)
        if strata_labels is not None:
            keep &= ~_missing_mask(strata_labels)
        if cluster_labels is not None:
            keep &= ~_missing_mask(cluster_labels)
        x, entry, exit_, event, weight = (
            x[keep],
            entry[keep],
            exit_[keep],
            event[keep],
            weight[keep],
        )
        if strata_labels is not None:
            strata_labels = strata_labels[keep]
        if cluster_labels is not None:
            cluster_labels = cluster_labels[keep]
        if not event.any():
            raise ValueError("No events remain after dropping missing rows.")

        # Group members by stratum (a single group when unstratified).
        if strata_labels is None:
            strata_groups = [(np.arange(x.shape[0]), np.unique(exit_[event]))]
        else:
            strata_groups = []
            for level in dict.fromkeys(strata_labels.tolist()):
                members = np.nonzero(strata_labels == level)[0]
                ev_times = np.unique(exit_[members][event[members]])
                strata_groups.append((members, ev_times))
        p = x.shape[1]

        def terms(beta: Array) -> tuple[float, Array, Array]:
            return _cox_terms(beta, x, entry, exit_, event, weight, strata_groups, self.ties)

        beta = np.zeros(p)
        loglik_null, grad0, info0 = terms(beta)
        loglik = loglik_null
        for _ in range(max_iter):
            _, grad, info = terms(beta)
            step = np.linalg.solve(info, grad)
            # Newton with step-halving to guarantee the likelihood increases.
            halving = 0
            while True:
                candidate = beta + step
                new_loglik, _, _ = terms(candidate)
                if new_loglik >= loglik - 1e-12 or halving >= 20:
                    break
                step = step / 2.0
                halving += 1
            converged = abs(new_loglik - loglik) <= tol * (abs(new_loglik) + tol)
            beta, loglik = candidate, new_loglik
            if converged:
                break

        _, _, info = terms(beta)
        naive_var = np.linalg.inv(info)

        # Retain the fitted design for diagnostics, baseline hazard, and prediction.
        self._x = x
        self._entry = entry
        self._exit = exit_
        self._event = event
        self._weight = weight
        self._strata_groups = strata_groups
        self._strata_labels = strata_labels
        self._xbar = (weight[:, None] * x).sum(axis=0) / weight.sum()

        self.term_names_ = names
        self.coef_ = beta
        self.naive_vcov_ = naive_var
        self.naive_std_error_ = np.sqrt(np.diag(naive_var))
        self.robust = robust or cluster is not None

        if self.robust:
            scores = self._score_residuals(beta)
            if cluster_labels is not None:
                levels = list(dict.fromkeys(cluster_labels.tolist()))
                scores = np.array([scores[cluster_labels == lev].sum(axis=0) for lev in levels])
            meat = scores.T @ scores
            self.vcov_ = naive_var @ meat @ naive_var
        else:
            self.vcov_ = naive_var

        self.std_error_ = np.sqrt(np.diag(self.vcov_))
        self.hazard_ratio_ = np.exp(beta)
        self.z_ = beta / self.std_error_
        self.p_value_ = 2.0 * norm.sf(np.abs(self.z_))
        self.loglik_ = float(loglik)
        self.loglik_null_ = float(loglik_null)
        self.n_ = int(keep.sum())
        self.n_event_ = int(event.sum())

        z = float(norm.ppf(1.0 - (1.0 - self.conf_level) / 2.0))
        half = z * self.std_error_
        self.conf_low_ = beta - half
        self.conf_high_ = beta + half

        # Global tests (all chi-square on p degrees of freedom).
        self.df_ = p
        self.lr_stat_ = 2.0 * (loglik - loglik_null)
        self.wald_stat_ = float(beta @ np.linalg.solve(self.vcov_, beta))
        self.score_stat_ = float(grad0 @ np.linalg.solve(info0, grad0))
        return self

    # -- baseline hazard & prediction ----------------------------------------

    def _group_label(self, members: Array) -> Any:
        return None if self._strata_labels is None else self._strata_labels[members[0]]

    def _baseline(self) -> list[tuple[Any, Array, Array]]:
        """Uncentered baseline cumulative hazard per stratum.

        Returns `(label, times, cumhaz)` per stratum group, reported at all unique exit
        times (matching R's `basehaz`); the hazard increments only at event times.
        """
        risk_score = np.exp(self._x @ self.coef_) * self._weight
        out: list[tuple[Any, Array, Array]] = []
        for members, _ in self._strata_groups:
            exit_s = self._exit[members]
            event_s = self._event[members]
            times = np.unique(exit_s)
            increments = np.zeros(times.shape[0])
            for i, t in enumerate(times):
                dying = (exit_s == t) & event_s
                if not dying.any():
                    continue
                at_risk = (self._entry[members] < t) & (exit_s >= t)
                s0 = risk_score[members][at_risk].sum()
                dw = self._weight[members][dying].sum()
                if self.ties == "breslow":
                    increments[i] = dw / s0
                else:  # efron
                    d0 = risk_score[members][dying].sum()
                    m = int(dying.sum())
                    increments[i] = sum((dw / m) / (s0 - (tie / m) * d0) for tie in range(m))
            out.append((self._group_label(members), times, np.cumsum(increments)))
        return out

    def baseline_hazard(self) -> Any:
        """Return the uncentered baseline cumulative hazard and survival as a frame.

        When the model is stratified, rows carry a `strata` column.
        """
        import pandas as pd

        frames = []
        for label, times, cumhaz in self._baseline():
            frame = pd.DataFrame({"time": times, "cumhaz": cumhaz, "survival": np.exp(-cumhaz)})
            if self._strata_labels is not None:
                frame.insert(0, "strata", label)
            frames.append(frame)
        return pd.concat(frames, ignore_index=True)

    def _linear_predictor(self, x: Array) -> Array:
        """Centered linear predictor `(x - xbar) . beta` (as in R `predict(type='lp')`)."""
        return (x - self._xbar) @ self.coef_

    def predict(
        self,
        newdata: Any = None,
        *,
        type: str = "lp",
        times: Any = None,
        conditional_after: Any = None,
    ) -> Any:
        """Predict from the fitted model.

        `type` is one of `"lp"` (centered linear predictor), `"risk"` (`exp(lp)`), or
        `"survival"`. For `"survival"`, returns a frame of survival probabilities at `times`
        (defaulting to the event times), one column per row of `newdata`. Survival prediction
        for stratified models is not yet supported.

        `conditional_after` (a scalar or one value per subject) predicts survival conditional
        on having already survived to that time: the returned value at time `t` is
        `P(T > t | T > c) = S(t) / S(c)`, and is 1 for `t <= c`.
        """
        if newdata is None:
            x = self._x
        else:
            x, _ = _design_matrix(newdata)

        if type == "lp":
            return self._linear_predictor(x)
        if type == "risk":
            return np.exp(self._linear_predictor(x))
        if type == "survival":
            import pandas as pd

            if self._strata_labels is not None:
                raise NotImplementedError(
                    "Survival prediction for stratified models is not yet supported."
                )
            _, base_times, base_cumhaz = self._baseline()[0]
            query = base_times if times is None else np.atleast_1d(np.asarray(times, dtype=float))
            idx = np.searchsorted(base_times, query, side="right") - 1
            h0 = np.where(idx >= 0, base_cumhaz[idx.clip(min=0)], 0.0)
            risk = np.exp(x @ self.coef_)  # uncentered: S(t|x) = exp(-H0(t) exp(x.beta))
            if conditional_after is None:
                surv = np.exp(-np.outer(h0, risk))
            else:
                h0_c = self._baseline_cumhaz_at(base_times, base_cumhaz, conditional_after, x)
                delta = np.clip(h0[:, None] - h0_c[None, :], 0.0, None)  # (n_times, n_subj)
                surv = np.exp(-delta * risk[None, :])
            frame = pd.DataFrame({f"subject_{i + 1}": surv[:, i] for i in range(x.shape[0])})
            frame.insert(0, "time", query)
            return frame
        raise ValueError(f"Unknown predict type {type!r}; use 'lp', 'risk', or 'survival'.")

    @staticmethod
    def _baseline_cumhaz_at(
        base_times: Array, base_cumhaz: Array, conditional_after: Any, x: Array
    ) -> Array:
        """Baseline cumulative hazard at `conditional_after`, broadcast to one value per subject."""
        c = np.asarray(conditional_after, dtype=float)
        if c.ndim == 0:
            c = np.full(x.shape[0], float(c))
        if c.shape[0] != x.shape[0]:
            raise ValueError("conditional_after must be a scalar or one value per subject.")
        idx_c = np.searchsorted(base_times, c, side="right") - 1
        return np.where(idx_c >= 0, base_cumhaz[idx_c.clip(min=0)], 0.0)

    # -- residuals & diagnostics ---------------------------------------------

    def residuals(self, type: str = "martingale") -> Any:
        """Return `"martingale"` or `"schoenfeld"` residuals.

        Martingale residuals are one per observation; Schoenfeld residuals are one row per
        event (columns are the covariates), ordered by stratum and then event time.
        """
        if type == "martingale":
            risk = np.exp(self._x @ self.coef_)
            cumhaz_i = np.zeros(self._x.shape[0])
            for (members, _), (_, times, cumhaz) in zip(
                self._strata_groups, self._baseline(), strict=True
            ):
                idx = np.searchsorted(times, self._exit[members], side="right") - 1
                h0 = np.where(idx >= 0, cumhaz[idx.clip(min=0)], 0.0)
                cumhaz_i[members] = h0 * risk[members]
            return self._event.astype(float) - cumhaz_i
        if type == "schoenfeld":
            import pandas as pd

            residuals, _, _ = self._event_contributions()
            arr = np.array(residuals)
            return pd.DataFrame({name: arr[:, j] for j, name in enumerate(self.term_names_)})
        raise ValueError(f"Unknown residual type {type!r}; use 'martingale' or 'schoenfeld'.")

    def _event_contributions(self) -> tuple[list[Array], list[float], list[Array]]:
        """Per-event Schoenfeld residual, event time, and risk-set covariance share.

        Iterates strata then event times; risk sets are confined to the stratum. For tied
        Efron events, the risk mean is averaged and the covariance split across the ties.
        """
        risk_score = np.exp(self._x @ self.coef_) * self._weight
        residuals: list[Array] = []
        times: list[float] = []
        covariances: list[Array] = []
        for members, event_times in self._strata_groups:
            xs = self._x[members]
            es = self._entry[members]
            xx = self._exit[members]
            ev = self._event[members]
            rs = risk_score[members]
            for t in event_times:
                at_risk = (es < t) & (xx >= t)
                dying = (xx == t) & ev
                rr = rs[at_risk]
                rx = xs[at_risk]
                s0 = rr.sum()
                s1 = (rx * rr[:, None]).sum(axis=0)
                s2 = (rx * rr[:, None]).T @ rx
                if self.ties == "breslow":
                    xbar = s1 / s0
                    cov = s2 / s0 - np.outer(xbar, xbar)
                else:  # efron: average over the tie-adjusted denominators
                    dr = rs[dying]
                    dx = xs[dying]
                    d0 = dr.sum()
                    d1 = (dx * dr[:, None]).sum(axis=0)
                    d2 = (dx * dr[:, None]).T @ dx
                    m = int(dying.sum())
                    means = [(s1 - (tie / m) * d1) / (s0 - (tie / m) * d0) for tie in range(m)]
                    xbar = np.mean(means, axis=0)
                    cov = np.zeros_like(s2)
                    for tie in range(m):
                        f = tie / m
                        z1 = (s1 - f * d1) / (s0 - f * d0)
                        cov += (s2 - f * d2) / (s0 - f * d0) - np.outer(z1, z1)
                    cov = cov / m
                for xi in xs[dying]:
                    residuals.append(xi - xbar)
                    times.append(float(t))
                    covariances.append(cov)
        return residuals, times, covariances

    def cox_zph(self, *, transform: str = "identity") -> ZPHResult:
        """Test the proportional-hazards assumption (Grambsch-Therneau).

        Regresses the scaled Schoenfeld residuals on a transform of time. `transform` is
        `"identity"` (default) or `"log"`, both validated against R's `cox.zph`. (R defaults
        to a Kaplan-Meier transform; `"km"` and `"rank"` are planned.)
        """
        residuals, times, covariances = self._event_contributions()
        t = np.array(times)
        if transform == "identity":
            g = t
        elif transform == "log":
            g = np.log(t)
        else:
            raise ValueError(f"transform must be 'identity' or 'log', got {transform!r}.")

        centered = g - g.mean()
        p = self.coef_.shape[0]
        u = np.zeros(p)
        a = np.zeros((p, p))
        c = np.zeros((p, p))
        b = np.zeros((p, p))
        for k in range(len(residuals)):
            gc = centered[k]
            v = covariances[k]
            u += gc * residuals[k]
            a += gc * gc * v
            c += gc * v
            b += v
        # Correct for beta having been estimated (the Schoenfeld residuals are constrained).
        var = a - c @ np.linalg.solve(b, c)

        per_term: dict[str, dict[str, float]] = {}
        for j, name in enumerate(self.term_names_):
            stat = float(u[j] ** 2 / var[j, j])
            per_term[name] = {"chisq": stat, "df": 1, "p_value": float(chi2.sf(stat, 1))}
        global_stat = float(u @ np.linalg.solve(var, u))
        global_test = {
            "chisq": global_stat,
            "df": self.df_,
            "p_value": float(chi2.sf(global_stat, self.df_)),
        }
        return ZPHResult(transform=transform, per_term=per_term, global_test=global_test)

    def _score_residuals(self, beta: Array) -> Array:
        """Breslow-form score (dfbeta-precursor) residuals, one per observation.

        Confined to strata; summed over the event times at which each subject is at risk.
        Used to form the robust (Lin-Wei) sandwich variance.
        """
        n, p = self._x.shape
        scores = np.zeros((n, p))
        for members, event_times in self._strata_groups:
            xs = self._x[members]
            es = self._entry[members]
            xx = self._exit[members]
            ev = self._event[members]
            ws = self._weight[members]
            ri = np.exp(xs @ beta)
            order = np.argsort(event_times)
            etimes = event_times[order]
            xbar = np.zeros((etimes.shape[0], p))
            dlambda = np.zeros(etimes.shape[0])
            for k, t in enumerate(etimes):
                at_risk = (es < t) & (xx >= t)
                dying = (xx == t) & ev
                rr = (ri * ws)[at_risk]
                s0 = rr.sum()
                xbar[k] = (xs[at_risk] * rr[:, None]).sum(axis=0) / s0
                dlambda[k] = ws[dying].sum() / s0
            index = {float(t): k for k, t in enumerate(etimes)}
            for local, gi in enumerate(members):
                x_i = xs[local]
                at = (es[local] < etimes) & (etimes <= xx[local])
                compensator = ri[local] * (
                    x_i * (dlambda[at]).sum() - (xbar[at] * dlambda[at][:, None]).sum(axis=0)
                )
                score = -compensator
                if ev[local]:
                    score = score + (x_i - xbar[index[float(xx[local])]])
                scores[gi] = ws[local] * score
        return scores

    def concordance(self) -> float:
        """Harrell's concordance index (C-statistic) of the fitted risk scores.

        Matches R's `survival::concordance`: a subject that dies at time `t` is treated as
        having failed before another subject still under observation at `t` (including one
        censored exactly at `t`); pairs tied in event time are excluded. For stratified
        models, only within-stratum pairs are compared.
        """
        risk = self._x @ self.coef_
        exit_ = self._exit
        event = self._event
        concordant = 0.0
        comparable = 0.0
        for members, _ in self._strata_groups:
            rk = risk[members]
            ex = exit_[members]
            ev = event[members]
            for i in range(ex.shape[0]):
                if not ev[i]:
                    continue
                later = (ex > ex[i]) | ((ex == ex[i]) & ~ev)
                if not later.any():
                    continue
                comparable += float(later.sum())
                concordant += float(np.sum(rk[i] > rk[later]))
                concordant += 0.5 * float(np.sum(rk[i] == rk[later]))
        return concordant / comparable

    # -- interop --------------------------------------------------------------

    def to_dataframe(self, *, exponentiate: bool = False) -> Any:
        """Return a tidy coefficient table (one row per term)."""
        import pandas as pd

        if exponentiate:
            estimate = self.hazard_ratio_
            conf_low = np.exp(self.conf_low_)
            conf_high = np.exp(self.conf_high_)
        else:
            estimate = self.coef_
            conf_low = self.conf_low_
            conf_high = self.conf_high_
        return pd.DataFrame(
            {
                "term": self.term_names_,
                "estimate": estimate,
                "std_error": self.std_error_,
                "statistic": self.z_,
                "p_value": self.p_value_,
                "conf_low": conf_low,
                "conf_high": conf_high,
            }
        )


def _tidy_cox(model: CoxPH, *, exponentiate: bool = False, **_: Any) -> Any:
    """broom-style `tidy`: one row per term; `exponentiate` gives hazard ratios."""
    return model.to_dataframe(exponentiate=exponentiate)


def _glance_cox(model: CoxPH, **_: Any) -> Any:
    """broom-style `glance`: one-row model summary."""
    import pandas as pd

    return pd.DataFrame(
        [
            {
                "n": model.n_,
                "nevent": model.n_event_,
                "loglik": model.loglik_,
                "aic": -2.0 * model.loglik_ + 2.0 * model.df_,
                "lr_statistic": model.lr_stat_,
                "df": model.df_,
                "lr_p_value": float(chi2.sf(model.lr_stat_, model.df_)),
            }
        ]
    )


def _register_adapters() -> None:
    from .tidy import register_glance, register_tidier

    register_tidier("greenwood._cox.CoxPH", _tidy_cox)
    register_glance("greenwood._cox.CoxPH", _glance_cox)


_register_adapters()
