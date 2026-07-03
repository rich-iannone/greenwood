def _require_plotnine() -> Any:
    try:
        import plotnine as p9
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise ImportError(
            "Visualization requires plotnine. Install it with `pip install greenwood[viz]`."
        ) from exc
    return p9


def _strata_label(block: Any) -> str:
    return _OVERALL if block.label is None else str(block.label)


def _default_times(km: KaplanMeier) -> list[float]:
    """Six evenly spaced, rounded times from 0 to the largest observed time."""
    max_t = max(float(b.time[-1]) for b in km._blocks if b.time.size)
    raw = np.linspace(0.0, max_t, 6)
    return sorted({round(float(t)) for t in raw})


def _step_frame(block: Any) -> Any:
    """Expand a block into right-continuous step coordinates for line and ribbon.

    Rows with a NaN confidence limit (the point where survival reaches 0) are kept for the
    line but dropped from the ribbon by the caller.
    """
    import pandas as pd

    xs = [0.0]
    est = [1.0]
    low = [1.0]
    high = [1.0]
    prev_e, prev_l, prev_h = 1.0, 1.0, 1.0
    for i in range(block.time.shape[0]):
        t = float(block.time[i])
        xs.extend([t, t])
        est.extend([prev_e, float(block.surv[i])])
        low.extend([prev_l, float(block.conf_low[i])])
        high.extend([prev_h, float(block.conf_high[i])])
        prev_e, prev_l, prev_h = (
            float(block.surv[i]),
            float(block.conf_low[i]),
            float(block.conf_high[i]),
        )
    return pd.DataFrame(
        {
            "time": xs,
            "estimate": est,
            "conf_low": low,
            "conf_high": high,
            "strata": _strata_label(block),
        }
    )


def _censor_frame(block: Any) -> Any:
    import pandas as pd

    mask = block.n_censor > 0
    return pd.DataFrame(
        {
            "time": block.time[mask],
            "estimate": block.surv[mask],
            "strata": _strata_label(block),
        }
    )


def _n_at_risk(block: Any, times: Array) -> Array:
    """Number at risk at each query time (as in R's `summary(fit, times=)$n.risk`)."""
    idx = np.searchsorted(block.time, times, side="left")
    out = np.zeros(times.shape[0])
    valid = idx < block.time.shape[0]
    out[valid] = block.n_risk[idx[valid]]
    return out


def risk_table_data(km: KaplanMeier, times: Any = None) -> Any:
    """Return a tidy frame of the number at risk per stratum at each of `times`."""
    import pandas as pd

    query = np.asarray(_default_times(km) if times is None else times, dtype=float)
    rows: list[dict[str, Any]] = []
    for block in km._blocks:
        counts = _n_at_risk(block, query)
        for t, n in zip(query, counts, strict=True):
            rows.append({"strata": _strata_label(block), "time": float(t), "n_risk": float(n)})
    return pd.DataFrame(rows)


def theme_survival() -> Any:
    """A light publication theme for survival plots."""
    p9 = _require_plotnine()
    return p9.theme_minimal() + p9.theme(
        legend_position="top",
        panel_grid_minor=p9.element_blank(),
    )


