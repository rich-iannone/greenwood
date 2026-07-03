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


