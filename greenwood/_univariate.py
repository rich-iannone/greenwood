class Parametric:
    r"""Univariate parametric survival distribution.

    Fits a parametric survival distribution to right-censored data by maximum likelihood, without
    covariates. The result is a fully specified distribution that provides survival, hazard,
    density, quantile, and mean-life predictions.

    Use this when you want to explore which distributional family best describes your data (Weibull
    vs. log-normal vs. log-logistic) before moving to regression modelling, or when the scientific
    question is simply "what is the median survival time and its uncertainty?"

    Parameters
    ----------
    dist
        Distribution family: `"weibull"` (default), `"exponential"`, `"lognormal"`, or
        `"loglogistic"`.
    conf_level
        Confidence level for parameter intervals (default `0.95`).

    Examples
    --------
    Fit a Weibull distribution to the lung cancer dataset:

    ```{python}
    import greenwood as gw

    lung = gw.load_dataset("lung", backend="polars")
    y = gw.Surv.right(lung["time"], event=(lung["status"] == 2))
    fit = gw.Parametric("weibull").fit(y)
    fit
    ```

    Compare all four distributions:

    ```{python}
    gw.compare_distributions(y, format="polars")
    ```
    """

    def __init__(self, dist: str = "weibull", *, conf_level: float = 0.95) -> None:
        if dist not in _DISTS:
            raise ValueError(f"dist must be one of {sorted(_DISTS)}, got {dist!r}.")
        if not 0.0 < conf_level < 1.0:
            raise ValueError(f"conf_level must be in (0, 1), got {conf_level}.")
        self.dist = dist
        self.conf_level = conf_level

    # -- repr ----------------------------------------------------------------

    def __repr__(self) -> str:
        if not hasattr(self, "params_"):
            return f"Parametric(dist={self.dist!r}, conf_level={self.conf_level}) <unfitted>"
        from ._repr import align_table, num

        names = list(self.params_.keys())
        rows = [[num(self.params_[n]), num(self.std_error_[n])] for n in names]
        table = align_table(["estimate", "std_error"], rows, names)
        return "\n".join(
            [
                f"Parametric ({self.dist} distribution)",
                "",
                table,
                "",
                f"n = {self.n_}, events = {self.n_event_}",
                f"Log-likelihood = {num(self.loglik_)}",
                f"AIC = {num(self.aic_)}, BIC = {num(self.bic_)}",
            ]
        )

