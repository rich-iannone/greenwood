"""Tests for k-fold cross-validation."""

from __future__ import annotations

import numpy as np
import pytest

import greenwood as gw
from greenwood import Surv, cross_validate


@pytest.fixture(scope="module")
def lung():
    return gw.load_dataset("lung", backend="pandas")


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
    cross_validate(model, y, lung[["age", "sex"]], seed=23)
    assert getattr(model, "coef_", None) is None  # the passed model stays unfitted


def test_invalid_arguments(lung, y) -> None:
    with pytest.raises(ValueError, match="k must be at least 2"):
        cross_validate(gw.CoxPH(), y, lung[["age", "sex"]], k=1)
    with pytest.raises(ValueError, match="metric"):
        cross_validate(gw.CoxPH(), y, lung[["age", "sex"]], metric="nope")
    with pytest.raises(ValueError, match="two time points"):
        cross_validate(gw.CoxPH(), y, lung[["age", "sex"]], metric="brier", times=[365])


def test_concordance_requires_cox_or_aft(lung, y) -> None:
    with pytest.raises(TypeError, match="CoxPH, CoxNet, or AFT"):
        cross_validate(gw.KaplanMeier(), y, lung[["age", "sex"]])


def test_stratified_kfold_balances_events(lung, y) -> None:
    """Stratified k-fold should balance event representation across folds."""
    from greenwood._resample import _stratified_kfold_indices

    overall_event_rate = y.event.mean()
    folds = _stratified_kfold_indices(y, k=5, seed=23)

    # Check each fold has similar event representation
    fold_event_rates = []
    for fold_idx in folds:
        fold_events = y.event[fold_idx]
        fold_rate = fold_events.mean()
        fold_event_rates.append(fold_rate)

    # Event rates across folds should be similar (within 15% tolerance)
    min_rate = min(fold_event_rates)
    max_rate = max(fold_event_rates)
    tolerance = 0.15
    assert max_rate - min_rate < tolerance, (
        f"Event rate imbalance: {fold_event_rates}, overall rate {overall_event_rate}"
    )


def test_stratified_vs_random_kfold(lung, y) -> None:
    """Stratified k-fold should produce different (more balanced) folds than random."""
    from greenwood._resample import _stratified_kfold_indices

    stratified = _stratified_kfold_indices(y, k=5, seed=23)
    random_perm = np.random.default_rng(23).permutation(y.n)
    random_folds = np.array_split(random_perm, 5)

    # Stratified should have more balanced event rates
    strat_rates = [y.event[f].mean() for f in stratified]
    random_rates = [y.event[f].mean() for f in random_folds]

    strat_spread = max(strat_rates) - min(strat_rates)
    random_spread = max(random_rates) - min(random_rates)

    # Stratified should typically have less spread (more balanced)
    assert strat_spread <= random_spread + 0.05


def test_stratified_kfold_reproducible(lung, y) -> None:
    """Stratified k-fold with same seed should produce same folds."""
    from greenwood._resample import _stratified_kfold_indices

    folds1 = _stratified_kfold_indices(y, k=5, seed=23)
    folds2 = _stratified_kfold_indices(y, k=5, seed=23)

    for f1, f2 in zip(folds1, folds2, strict=True):
        np.testing.assert_array_equal(f1, f2)


def test_cross_validate_stratified_parameter(lung, y) -> None:
    """cross_validate should support stratified=True and stratified=False."""
    # Both should work without errors
    r_strat = cross_validate(gw.CoxPH(), y, lung[["age", "sex"]], stratified=True, seed=10)
    r_random = cross_validate(gw.CoxPH(), y, lung[["age", "sex"]], stratified=False, seed=10)

    assert len(r_strat["scores"]) == 5
    assert len(r_random["scores"]) == 5
    # Results may be different due to different fold assignments, but both valid
    assert all(0.4 < s < 1.0 for s in r_strat["scores"])
    assert all(0.4 < s < 1.0 for s in r_random["scores"])


def test_low_event_rate_warns(lung, y) -> None:
    """cross_validate should warn when there are very few events relative to k."""
    # Build a highly imbalanced dataset: 200 subjects, only 6 events.
    rng = np.random.default_rng(0)
    n = 200
    time = rng.exponential(500, size=n)
    event = np.zeros(n, dtype=bool)
    event[rng.choice(n, size=6, replace=False)] = True  # 3% event rate
    y_sparse = Surv.right(time, event=event)
    x_sparse = rng.standard_normal((n, 2))

    with pytest.warns(UserWarning, match="fewer than 2"):
        cross_validate(gw.CoxPH(), y_sparse, x_sparse, k=5, seed=1)


def test_imbalanced_data_no_error(lung, y) -> None:
    """cross_validate on low-event data should produce valid scores (not crash)."""
    rng = np.random.default_rng(42)
    n = 300
    time = rng.exponential(400, size=n)
    event = np.zeros(n, dtype=bool)
    # ~8% event rate — just above the 2*k=10 threshold for k=5
    event[rng.choice(n, size=25, replace=False)] = True
    y_sparse = Surv.right(time, event=event)
    x_sparse = rng.standard_normal((n, 2))

    result = cross_validate(gw.CoxPH(), y_sparse, x_sparse, k=5, seed=42)
    assert len(result["scores"]) == 5
    assert all(np.isfinite(s) for s in result["scores"])
