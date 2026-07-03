def _require_plotnine() -> Any:
    try:
        import plotnine as p9
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise ImportError(
            "Visualization requires plotnine. Install it with `pip install greenwood[viz]`."
        ) from exc
    return p9


