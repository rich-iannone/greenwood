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

    # -- fit -----------------------------------------------------------------

    def fit(self, surv: Surv) -> Parametric:
        r"""Fit the distribution to right-censored survival data by maximum likelihood.

        Maximises the likelihood of the parametric model $\log T = \mu + \sigma\varepsilon$
        (intercept-only AFT) over the observed and censored times. The result is stored on the
        fitted object and reported in the natural parameterisation of each distribution.

        Parameters
        ----------
        surv
            A right-censored `Surv` response built with `Surv.right()`.

        Returns
        -------
        Parametric
            The fitted object (for method chaining), with attributes `params_`, `std_error_`,
            `loglik_`, `aic_`, `bic_`, etc.

        Examples
        --------
        ```{python}
        import greenwood as gw

        lung = gw.load_dataset("lung", backend="polars")
        y = gw.Surv.right(lung["time"], event=(lung["status"] == 2))
        gw.Parametric("lognormal").fit(y)
        ```
        """
        from ._surv import CensoringType

        if surv.type is not CensoringType.RIGHT:
            raise NotImplementedError(
                f"Parametric currently supports right-censored responses, not {surv.type.value!r}."
            )

        time = surv.stop
        event = surv.event
        keep = time > 0
        time, event = time[keep], event[keep]
        log_t = np.log(time)

        has_scale = self.dist != "exponential"

        def neg_loglik(params: Array) -> float:
            mu = params[0]
            log_sigma = params[1] if has_scale else 0.0
            sigma = np.exp(log_sigma)
            z = (log_t - mu) / sigma
            log_f, log_s = _log_density_survival(self.dist, z)
            ll = event * (log_f - log_sigma - log_t) + (1.0 - event) * log_s
            return -float(ll.sum())

        x0 = np.array([float(log_t.mean())] + ([0.0] if has_scale else []))
        result = minimize(neg_loglik, x0, method="BFGS", options={"gtol": 1e-8, "maxiter": 1000})
        params = result.x

        # Variance via numerical Hessian of the negative log-likelihood.
        vcov_raw = np.linalg.inv(_num_hessian(neg_loglik, params))

        # Store the AFT location-scale parameterisation.
        self._mu = float(params[0])
        self._log_sigma = float(params[1]) if has_scale else 0.0
        self._sigma = float(np.exp(self._log_sigma))
        self._vcov_raw = vcov_raw  # on (mu, log_sigma) scale
        self.loglik_ = -float(result.fun)
        self.n_ = int(keep.sum())
        self.n_event_ = int(event.sum())

        # Natural parameters, SEs, and CIs.
        self._compute_natural_params()

        # AIC / BIC.
        n_params = 2 if has_scale else 1
        self.aic_ = -2.0 * self.loglik_ + 2.0 * n_params
        self.bic_ = -2.0 * self.loglik_ + np.log(self.n_) * n_params

        return self

