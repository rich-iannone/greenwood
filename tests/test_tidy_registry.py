"""Tests for the tidy-layer registry and dispatch skeleton."""

from __future__ import annotations

import pytest

from greenwood import summaries as tidy


class _FakeFit:
    pass


class _FakeSubFit(_FakeFit):
    pass


def test_tidy_raises_for_unregistered_model() -> None:
    with pytest.raises(NotImplementedError, match="No tidier adapter registered"):
        tidy.tidy(object())


def test_register_and_dispatch() -> None:
    path = f"{_FakeFit.__module__}.{_FakeFit.__qualname__}"
    tidy.register_tidier(path, lambda m, **kw: {"ok": True, "kwargs": kw})
    try:
        assert tidy.tidy(_FakeFit(), conf_level=0.9) == {"ok": True, "kwargs": {"conf_level": 0.9}}
    finally:
        from greenwood.summaries._registry import _TIDIERS

        _TIDIERS.pop(path, None)


def test_dispatch_walks_mro() -> None:
    path = f"{_FakeFit.__module__}.{_FakeFit.__qualname__}"
    tidy.register_glance(path, lambda m, **kw: "glanced")
    try:
        assert tidy.glance(_FakeSubFit()) == "glanced"
    finally:
        from greenwood.summaries._registry import _GLANCERS

        _GLANCERS.pop(path, None)


def test_augment_unregistered_raises() -> None:
    with pytest.raises(NotImplementedError, match="No augment adapter registered"):
        tidy.augment(object())
