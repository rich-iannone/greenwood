"""Tests for k-fold cross-validation."""

from __future__ import annotations

import numpy as np
import pytest

import greenwood as gw
from greenwood import Surv, cross_validate


@pytest.fixture(scope="module")
def lung():
    return gw.data.load_dataset("lung", backend="pandas")


@pytest.fixture(scope="module")
def y(lung):
    return Surv.right(lung["time"], event=(lung["status"] == 2))


def test_concordance_is_deterministic_and_reasonable(lung, y) -> None:
    r1 = cross_validate(gw.CoxPH(), y, lung[["age", "sex"]], k=5, seed=7)
    r2 = cross_validate(gw.CoxPH(), y, lung[["age", "sex"]], k=5, seed=7)
    assert r1["scores"] == r2["scores"]  # same seed -> same folds/scores
    assert len(r1["scores"]) == 5
    assert 0.4 < r1["mean"] < 1.0
    assert set(r1) == {"metric", "k", "scores", "mean", "std"}


def test_brier_scores_in_range(lung, y) -> None:
    r = cross_validate(
        gw.CoxPH(), y, lung[["age", "sex"]], metric="brier", times=[180, 365, 540], seed=1
    )
    assert r["metric"] == "brier"
    assert all(0.0 <= s <= 0.5 for s in r["scores"])


def test_aft_concordance_direction(lung, y) -> None:
    # A useful model beats a coin flip; this checks the AFT risk sign is correct.
    r = cross_validate(gw.AFT("weibull"), y, lung[["age", "sex"]], k=5, seed=3)
    assert r["mean"] > 0.5


def test_formula_matches_frame(lung, y) -> None:
    frame = cross_validate(gw.CoxPH(), y, lung[["age", "sex"]], seed=2)
    formula = cross_validate(gw.CoxPH(), y, "age + sex", data=lung, seed=2)
    np.testing.assert_allclose(formula["scores"], frame["scores"])


def test_does_not_mutate_input_model(lung, y) -> None:
    model = gw.CoxPH()
    cross_validate(model, y, lung[["age", "sex"]], seed=0)
    assert getattr(model, "coef_", None) is None  # the passed model stays unfitted


def test_invalid_arguments(lung, y) -> None:
    with pytest.raises(ValueError, match="k must be at least 2"):
        cross_validate(gw.CoxPH(), y, lung[["age", "sex"]], k=1)
    with pytest.raises(ValueError, match="metric"):
        cross_validate(gw.CoxPH(), y, lung[["age", "sex"]], metric="nope")
    with pytest.raises(ValueError, match="two time points"):
        cross_validate(gw.CoxPH(), y, lung[["age", "sex"]], metric="brier", times=[365])


def test_concordance_requires_cox_or_aft(lung, y) -> None:
    with pytest.raises(TypeError, match="CoxPH or AFT"):
        cross_validate(gw.KaplanMeier(), y, lung[["age", "sex"]])
