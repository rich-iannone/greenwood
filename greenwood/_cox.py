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


