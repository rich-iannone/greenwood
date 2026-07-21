"""Registry and dispatch for Greenwood's tidy / `broom` layer.

Fitted Greenwood estimators expose standardized `tidy`/`glance`/`augment` views via this
registry, using the same broom-compatible protocol as `great_summaries.tidy`. That shared
contract is what lets Great Summaries build `tbl_survfit`/`tbl_regression` on top of
Greenwood without touching its internals.

This is the initial skeleton: the dispatch machinery and the extension protocol. Concrete
adapters are registered as each estimator lands (Kaplan-Meier, Cox, and the rest).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

__all__ = [
    "tidy",
    "glance",
    "augment",
    "register_tidier",
    "register_glance",
    "register_augment",
]

Tidier = Callable[..., Any]

# Registries keyed by fully-qualified class name (e.g. "greenwood._cox.CoxPH"), so
# registration never needs to import the estimator eagerly.
_TIDIERS: dict[str, Tidier] = {}
_GLANCERS: dict[str, Tidier] = {}
_AUGMENTERS: dict[str, Tidier] = {}


def _qualname(obj: object) -> str:
    cls = type(obj)
    return f"{cls.__module__}.{cls.__qualname__}"


def _lookup(registry: dict[str, Tidier], model: object, kind: str) -> Tidier:
    # Walk the MRO so registering a base class covers subclasses.
    for cls in type(model).__mro__:
        key = f"{cls.__module__}.{cls.__qualname__}"
        if key in registry:
            return registry[key]
    raise NotImplementedError(
        f"No {kind} adapter registered for {_qualname(model)!r}. "
        f"Register one with greenwood.tidy.register_{kind}()."
    )


def register_tidier(class_path: str, fn: Tidier) -> None:
    """Register a `tidy` adapter for a model class.

    This is the extension point for adding `tidy()` support to new estimator classes. Each
    adapter is a callable that accepts a fitted model as its first argument (plus optional
    keyword arguments like `format=`) and returns a tidy DataFrame.

    Parameters
    ----------
    class_path
        Fully qualified class name used as the registry key, e.g.,
        `"greenwood._cox.CoxPH"`. Using the string path avoids eager imports.
    fn
        Callable with signature `fn(model, *, format=None, **kwargs) -> DataFrame`.

    Examples
    --------
    Register a custom tidy adapter for a new estimator class:

    ```{python}
    from greenwood.summaries import register_tidier

    def _tidy_my_model(model, *, format=None, **kwargs):
        return model.to_frame(format=format)

    register_tidier("mypackage.MyModel", _tidy_my_model)
    ```
    """
    _TIDIERS[class_path] = fn


def register_glance(class_path: str, fn: Tidier) -> None:
    """Register a `glance` adapter for a model class.

    This is the extension point for adding `glance()` support to new estimator classes. Each
    adapter is a callable that accepts a fitted model and returns a one-row summary DataFrame
    with model-level statistics (e.g., log-likelihood, AIC, number of observations).

    Parameters
    ----------
    class_path
        Fully qualified class name used as the registry key, e.g.,
        `"greenwood._cox.CoxPH"`.
    fn
        Callable with signature `fn(model, *, format=None, **kwargs) -> DataFrame`.

    Examples
    --------
    Register a custom glance adapter:

    ```{python}
    from greenwood.summaries import register_glance

    def _glance_my_model(model, *, format=None, **kwargs):
        from greenwood._backends import to_dataframe

        return to_dataframe({"n": [model.n_], "loglik": [model.loglik_]}, format=format)

    register_glance("mypackage.MyModel", _glance_my_model)
    ```
    """
    _GLANCERS[class_path] = fn


def register_augment(class_path: str, fn: Tidier) -> None:
    """Register an `augment` adapter for a model class.

    This is the extension point for adding `augment()` support to new estimator classes. Each
    adapter is a callable that accepts a fitted model and (optionally) the original data, and
    returns an observation-level DataFrame with per-row predictions or residuals appended.

    Parameters
    ----------
    class_path
        Fully qualified class name used as the registry key, e.g.,
        `"greenwood._cox.CoxPH"`.
    fn
        Callable with signature `fn(model, data=None, *, format=None, **kwargs) -> DataFrame`.

    Examples
    --------
    Register a custom augment adapter:

    ```{python}
    from greenwood.summaries import register_augment

    def _augment_my_model(model, data=None, *, format=None, **kwargs):
        from greenwood._backends import to_dataframe

        preds = model.predict(data)
        return to_dataframe({"prediction": preds}, format=format)

    register_augment("mypackage.MyModel", _augment_my_model)
    ```
    """
    _AUGMENTERS[class_path] = fn


def tidy(model: object, **kwargs: Any) -> Any:
    """Return a standardised term-level DataFrame for a fitted model.

    Produces one row per model term (coefficient, parameter, or stratum) with columns for the
    estimate, standard error, and confidence limits. This is the Python equivalent of
    R's `broom::tidy()`.

    Parameters
    ----------
    model
        A fitted Greenwood estimator (e.g., `KaplanMeier`, `CoxPH`, `AFT`, `Parametric`).
    **kwargs
        Forwarded to the registered adapter. Common options include `format=` to choose the
        output backend (`"pandas"`, `"polars"`, or `"pyarrow"`).

    Returns
    -------
    DataFrame
        A tidy, term-level summary of the fitted model.

    Examples
    --------
    Tidy a fitted Cox model into a Polars DataFrame:

    ```{python}
    import greenwood as gw

    lung = gw.load_dataset("lung", backend="polars")
    y = gw.Surv.right(lung["time"], event=(lung["status"] == 2))
    cox = gw.CoxPH().fit(y, covariates=lung[["age", "sex"]])
    gw.tidy(cox, format="polars")
    ```
    """
    return _lookup(_TIDIERS, model, "tidier")(model, **kwargs)


def glance(model: object, **kwargs: Any) -> Any:
    """Return a one-row model-summary DataFrame for a fitted model.

    Produces a single row of model-level statistics such as the number of observations,
    number of events, log-likelihood, AIC, and concordance. This is the Python equivalent
    of R's `broom::glance()`.

    Parameters
    ----------
    model
        A fitted Greenwood estimator (e.g., `CoxPH`, `AFT`, `Parametric`).
    **kwargs
        Forwarded to the registered adapter. Common options include `format=`.

    Returns
    -------
    DataFrame
        A one-row model-level summary.

    Examples
    --------
    Glance at a fitted Cox model for its overall fit statistics:

    ```{python}
    import greenwood as gw

    lung = gw.load_dataset("lung", backend="polars")
    y = gw.Surv.right(lung["time"], event=(lung["status"] == 2))
    cox = gw.CoxPH().fit(y, covariates=lung[["age", "sex"]])
    gw.glance(cox, format="polars")
    ```
    """
    return _lookup(_GLANCERS, model, "glance")(model, **kwargs)


def augment(model: object, data: Any = None, **kwargs: Any) -> Any:
    """Return an observation-level DataFrame for a fitted model.

    Produces one row per observation, appending model-derived columns (e.g., fitted values,
    residuals, or predicted survival probabilities) to the original data. This is the Python
    equivalent of R's `broom::augment()`.

    Parameters
    ----------
    model
        A fitted Greenwood estimator (e.g., `CoxPH`).
    data
        The original data used to fit the model. Required by some adapters (e.g., Cox
        residuals need the covariate matrix); optional for others.
    **kwargs
        Forwarded to the registered adapter. Common options include `format=`.

    Returns
    -------
    DataFrame
        An observation-level summary with predictions or residuals.

    Examples
    --------
    Once an augment adapter is registered for a model class, call `augment()` to get
    observation-level predictions or residuals:

    ```python
    import greenwood as gw

    lung = gw.load_dataset("lung", backend="polars")
    y = gw.Surv.right(lung["time"], event=(lung["status"] == 2))
    cox = gw.CoxPH().fit(y, covariates=lung[["age", "sex"]])
    gw.augment(cox, data=lung, format="polars")
    ```
    """
    return _lookup(_AUGMENTERS, model, "augment")(model, data, **kwargs)
