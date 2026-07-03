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


