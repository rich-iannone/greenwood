def _design_matrix(covariates: Any) -> tuple[Array, list[str]]:
    """Build a numeric design matrix and term names from covariates.

    Accepts a 2-D NumPy array or any narwhals-compatible dataframe. Numeric columns pass
    through; non-numeric columns are treatment-coded (drop-first dummies) with names like
    ``celltypesmallcell``.
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
    event_times: Array,
    ties: str,
) -> tuple[float, Array, Array]:
    """Partial log-likelihood, gradient, and observed information at `beta`."""
    p = beta.shape[0]
    eta = x @ beta
    risk_score = np.exp(eta) * weight

    loglik = 0.0
    grad = np.zeros(p)
    info = np.zeros((p, p))

    for t in event_times:
        at_risk = (entry < t) & (exit_ >= t)
        dying = (exit_ == t) & event

        rx = x[at_risk]
        rr = risk_score[at_risk]
        s0 = rr.sum()
        s1 = rx.T @ rr
        s2 = (rx * rr[:, None]).T @ rx

        w_d = weight[dying]
        loglik += float((w_d * eta[dying]).sum())
        grad += (x[dying] * w_d[:, None]).sum(axis=0)

        if ties == "breslow":
            d_weight = float(w_d.sum())
            z1 = s1 / s0
            loglik -= d_weight * np.log(s0)
            grad -= d_weight * z1
            info += d_weight * (s2 / s0 - np.outer(z1, z1))
        else:  # efron
            dx = x[dying]
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

    def fit(self, surv: Surv, covariates: Any, *, max_iter: int = 30, tol: float = 1e-9) -> CoxPH:
        """Fit the model to a `Surv` response and a covariate design."""
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

        # Complete-case analysis: drop rows with any missing covariate.
        keep = ~np.isnan(x).any(axis=1)
        x, entry, exit_, event, weight = (
            x[keep],
            entry[keep],
            exit_[keep],
            event[keep],
            weight[keep],
        )
        if not event.any():
            raise ValueError("No events remain after dropping missing rows.")

        event_times = np.unique(exit_[event])
        p = x.shape[1]

        def terms(beta: Array) -> tuple[float, Array, Array]:
            return _cox_terms(beta, x, entry, exit_, event, weight, event_times, self.ties)

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
        var = np.linalg.inv(info)

        self.term_names_ = names
        self.coef_ = beta
        self.vcov_ = var
        self.std_error_ = np.sqrt(np.diag(var))
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
        self.wald_stat_ = float(beta @ np.linalg.solve(var, beta))
        self.score_stat_ = float(grad0 @ np.linalg.solve(info0, grad0))
        return self

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


