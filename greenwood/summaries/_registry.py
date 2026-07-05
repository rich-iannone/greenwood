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
    """Register a `tidy` adapter for the model class named `class_path`."""
    _TIDIERS[class_path] = fn


def register_glance(class_path: str, fn: Tidier) -> None:
    """Register a `glance` adapter for the model class named `class_path`."""
    _GLANCERS[class_path] = fn


def register_augment(class_path: str, fn: Tidier) -> None:
    """Register an `augment` adapter for the model class named `class_path`."""
    _AUGMENTERS[class_path] = fn


def tidy(model: object, **kwargs: Any) -> Any:
    """Return a standardized term-level frame for `model` (`broom::tidy`)."""
    return _lookup(_TIDIERS, model, "tidier")(model, **kwargs)


def glance(model: object, **kwargs: Any) -> Any:
    """Return a one-row model-summary frame for `model` (`broom::glance`)."""
    return _lookup(_GLANCERS, model, "glance")(model, **kwargs)


def augment(model: object, data: Any = None, **kwargs: Any) -> Any:
    """Return an observation-level frame for `model` (`broom::augment`)."""
    return _lookup(_AUGMENTERS, model, "augment")(model, data, **kwargs)
