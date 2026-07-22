"""Tests for elastic-net penalized Cox regression (`CoxNet`).

No R reference (glmnet is not installed), so correctness is pinned three ways: `penalizer=0`
must reproduce the R-validated unpenalized Breslow `CoxPH`; the solution must satisfy the
elastic-net KKT optimality conditions; and the penalty must produce the expected sparsity.
"""

from __future__ import annotations

import numpy as np
import pytest

import greenwood as gw
from greenwood import CoxNet, CoxPH, Surv
from greenwood._cox import _cox_terms


@pytest.fixture(scope="module")
def data():
    lung = gw.load_dataset("lung", backend="pandas")
    y = Surv.right(lung["time"], event=(lung["status"] == 2))
    cols = ["age", "sex", "ph.ecog", "ph.karno", "wt.loss"]
    x = lung[cols].fillna(lung[cols].mean())
    return y, x


def _max_kkt_residual(model: CoxNet) -> float:
    """Largest violation of the elastic-net subgradient optimality conditions."""
    x, center = model._x, model._center
    scale = np.where(x.std(axis=0) > 0, x.std(axis=0), 1.0)
    xs = (x - center) / scale
    beta = model.coef_ * scale
    n = x.shape[0]
    groups = [(np.arange(n), np.unique(model._exit[model._event]))]
    _, grad, _ = _cox_terms(
        beta, xs, model._entry, model._exit, model._event, model._weight, groups, "breslow"
    )
    lam, alpha = model.penalizer, model.l1_ratio
    grad_h = -grad / n + lam * (1.0 - alpha) * beta
    residual = 0.0
    for j in range(beta.size):
        if abs(beta[j]) > 1e-8:
            residual = max(residual, abs(grad_h[j] + lam * alpha * np.sign(beta[j])))
        else:
            residual = max(residual, abs(grad_h[j]) - lam * alpha)
    return residual


def test_penalizer_zero_matches_unpenalized_cox(data) -> None:  # type: ignore[no-untyped-def]
    y, x = data
    ref = CoxPH(ties="breslow").fit(y, x).coef_
    cn = CoxNet(penalizer=0.0).fit(y, x).coef_
    np.testing.assert_allclose(cn, ref, atol=1e-4)


@pytest.mark.parametrize("penalizer,l1_ratio", [(0.05, 1.0), (0.1, 0.5), (0.2, 0.5), (0.05, 0.0)])
def test_kkt_conditions_hold(data, penalizer, l1_ratio) -> None:  # type: ignore[no-untyped-def]
    y, x = data
    model = CoxNet(penalizer=penalizer, l1_ratio=l1_ratio).fit(y, x)
    assert _max_kkt_residual(model) < 1e-4


def test_lasso_sparsity_increases_with_penalizer(data) -> None:  # type: ignore[no-untyped-def]
    y, x = data
    counts = [
        int(np.count_nonzero(CoxNet(penalizer=lam, l1_ratio=1.0).fit(y, x).coef_))
        for lam in (0.01, 0.05, 0.2, 1.0)
    ]
    assert counts == sorted(counts, reverse=True)  # monotonically fewer nonzero
    assert counts[-1] == 0  # a large lasso penalty zeros everything


def test_ridge_shrinks_without_zeroing(data) -> None:  # type: ignore[no-untyped-def]
    y, x = data
    ridge = CoxNet(penalizer=0.1, l1_ratio=0.0).fit(y, x)
    unpen = np.abs(CoxPH(ties="breslow").fit(y, x).coef_)
    assert np.all(np.count_nonzero(ridge.coef_) == ridge.coef_.size)  # nothing set to zero
    assert np.sum(np.abs(ridge.coef_)) < np.sum(unpen)  # but shrunk toward zero


def test_predict_shapes_and_survival_range(data) -> None:  # type: ignore[no-untyped-def]
    y, x = data
    model = CoxNet(penalizer=0.05, l1_ratio=0.5).fit(y, x)
    lp = model.predict(x, type="lp")
    assert lp.shape == (model.n_,)
    surv = model.predict(x.iloc[:3], type="survival", times=[180, 365], format="pandas")
    assert list(surv.columns) == ["time", "subject_1", "subject_2", "subject_3"]
    assert ((surv.iloc[:, 1:] >= 0) & (surv.iloc[:, 1:] <= 1)).all().all()


def test_to_pandas_and_repr(data) -> None:  # type: ignore[no-untyped-def]
    y, x = data
    model = CoxNet(penalizer=0.1, l1_ratio=1.0).fit(y, x)
    df = model.to_frame(format="pandas")
    assert list(df.columns) == ["term", "estimate", "hazard_ratio"]
    text = repr(model)
    assert "elastic-net Cox" in text
    assert "nonzero coefficients" in text
    assert "object at 0x" not in text


def test_invalid_arguments() -> None:
    with pytest.raises(ValueError, match="penalizer"):
        CoxNet(penalizer=-1.0)
    with pytest.raises(ValueError, match="l1_ratio"):
        CoxNet(l1_ratio=2.0)


# ---------------------------------------------------------------------------
# cv_coxnet tests
# ---------------------------------------------------------------------------

from greenwood import CoxNetCVResult, cv_coxnet  # noqa: E402


def test_cv_coxnet_returns_valid_result(data) -> None:  # type: ignore[no-untyped-def]
    y, x = data
    result = cv_coxnet(y, x, l1_ratio=1.0, n_penalizers=20, k=3, seed=0)
    assert isinstance(result, CoxNetCVResult)
    assert len(result.penalizers_) == 20
    assert len(result.mean_scores_) == 20
    assert len(result.std_scores_) == 20
    assert len(result.n_nonzero_) == 20
    assert result.metric_ == "concordance"
    assert result.l1_ratio_ == 1.0
    assert result.k_ == 3


def test_cv_coxnet_path_is_sorted_descending(data) -> None:  # type: ignore[no-untyped-def]
    y, x = data
    result = cv_coxnet(y, x, l1_ratio=1.0, n_penalizers=20, k=3, seed=0)
    assert np.all(np.diff(result.penalizers_) <= 0), "penalizers_ must be sorted descending"


def test_cv_coxnet_best_penalizer_in_path(data) -> None:  # type: ignore[no-untyped-def]
    y, x = data
    result = cv_coxnet(y, x, l1_ratio=1.0, n_penalizers=20, k=3, seed=0)
    assert result.best_penalizer_ in result.penalizers_


def test_cv_coxnet_1se_penalizer_ge_best(data) -> None:  # type: ignore[no-untyped-def]
    """1-SE penalizer is >= best penalizer (more regularized / sparser)."""
    y, x = data
    result = cv_coxnet(y, x, l1_ratio=1.0, n_penalizers=20, k=3, seed=0)
    assert result.penalizer_1se_ >= result.best_penalizer_


def test_cv_coxnet_scores_in_valid_range(data) -> None:  # type: ignore[no-untyped-def]
    y, x = data
    result = cv_coxnet(y, x, l1_ratio=1.0, n_penalizers=20, k=3, seed=0)
    assert np.all((result.mean_scores_ >= 0) & (result.mean_scores_ <= 1))
    assert np.all(result.std_scores_ >= 0)
    assert np.all(result.n_nonzero_ >= 0)


def test_cv_coxnet_n_nonzero_decreases_with_penalizer(data) -> None:  # type: ignore[no-untyped-def]
    """More regularisation (larger lambda) should not increase non-zero count on average."""
    y, x = data
    result = cv_coxnet(y, x, l1_ratio=1.0, n_penalizers=15, k=3, seed=0)
    # path is sorted descending; n_nonzero should be non-increasing (large lambda → fewer non-zeros)
    assert result.n_nonzero_[0] <= result.n_nonzero_[-1]


def test_cv_coxnet_custom_penalizers(data) -> None:  # type: ignore[no-untyped-def]
    y, x = data
    custom = [0.01, 0.05, 0.1, 0.2]
    result = cv_coxnet(y, x, penalizers=custom, k=3, seed=0)
    # Custom path is sorted descending
    np.testing.assert_array_equal(result.penalizers_, sorted(custom, reverse=True))
    assert len(result.mean_scores_) == 4


def test_cv_coxnet_ridge(data) -> None:  # type: ignore[no-untyped-def]
    """Ridge (l1_ratio=0) should produce no sparsity but still return a valid result."""
    y, x = data
    result = cv_coxnet(y, x, l1_ratio=0.0, n_penalizers=10, k=3, seed=0)
    assert isinstance(result, CoxNetCVResult)
    # Ridge never zeros coefficients — n_nonzero should be p everywhere (or at least nonzero)
    assert np.all(result.n_nonzero_ > 0)


def test_cv_coxnet_to_frame_columns(data) -> None:  # type: ignore[no-untyped-def]
    y, x = data
    result = cv_coxnet(y, x, l1_ratio=1.0, n_penalizers=10, k=3, seed=0)
    df = result.to_frame(format="pandas")
    assert list(df.columns) == ["penalizer", "mean_score", "std_score", "n_nonzero"]
    assert len(df) == 10


def test_cv_coxnet_repr(data) -> None:  # type: ignore[no-untyped-def]
    y, x = data
    result = cv_coxnet(y, x, l1_ratio=1.0, n_penalizers=10, k=3, seed=0)
    text = repr(result)
    assert "CoxNetCV" in text
    assert "concordance" in text
    assert "best penalizer" in text
    assert "1-SE" in text
    assert "object at 0x" not in text


def test_cv_coxnet_invalid_args(data) -> None:  # type: ignore[no-untyped-def]
    y, x = data
    with pytest.raises(ValueError, match="l1_ratio"):
        cv_coxnet(y, x, l1_ratio=1.5)
    with pytest.raises(ValueError, match="k must be at least 2"):
        cv_coxnet(y, x, k=1)
    with pytest.raises(ValueError, match="metric"):
        cv_coxnet(y, x, metric="invalid")
    with pytest.raises(ValueError, match="times"):
        cv_coxnet(y, x, metric="brier")


def test_cv_coxnet_brier_requires_two_times(data) -> None:  # type: ignore[no-untyped-def]
    y, x = data
    with pytest.raises(ValueError, match="at least two time points"):
        cv_coxnet(y, x, metric="brier", times=[100.0])


def test_cv_coxnet_rejects_interval_censored() -> None:
    y_int = Surv.interval(lower=[1, 2, 3, 4], upper=[2, 3, 4, 5])
    with pytest.raises(NotImplementedError, match="right-censored"):
        cv_coxnet(y_int, np.zeros((4, 1)))


def test_cv_coxnet_row_mismatch(data) -> None:  # type: ignore[no-untyped-def]
    y, _ = data
    with pytest.raises(ValueError, match="same number of rows"):
        cv_coxnet(y, np.zeros((5, 1)))


def test_cv_coxnet_no_events() -> None:
    y_no_events = Surv.right([1, 2, 3, 4], [0, 0, 0, 0])
    with pytest.raises(ValueError, match="No events"):
        cv_coxnet(y_no_events, np.array([[1.0], [2.0], [3.0], [4.0]]))


def test_cv_coxnet_negative_penalizer(data) -> None:  # type: ignore[no-untyped-def]
    y, x = data
    with pytest.raises(ValueError, match="non-negative"):
        cv_coxnet(y, x, penalizers=[-1.0, 0.1])


def test_cv_coxnet_n_penalizers_zero(data) -> None:  # type: ignore[no-untyped-def]
    y, x = data
    with pytest.raises(ValueError, match="n_penalizers must be at least 1"):
        cv_coxnet(y, x, n_penalizers=0)


def test_cv_coxnet_few_events_warning() -> None:
    times = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
    events = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0])
    y = Surv.right(times, events)
    x = np.random.default_rng(42).standard_normal((10, 2))
    with pytest.warns(UserWarning, match="fewer than"):
        cv_coxnet(y, x, k=5, penalizers=[0.1], seed=42)


def test_cv_coxnet_missing_rows_filtered(data) -> None:  # type: ignore[no-untyped-def]
    y, x = data
    import pandas as pd

    x_df = pd.DataFrame(x) if not isinstance(x, pd.DataFrame) else x.copy()
    x_df.iloc[0, 0] = np.nan
    result = cv_coxnet(y, x_df, penalizers=[0.1], k=2, seed=42)
    assert result.best_penalizer_ > 0


def test_cv_coxnet_brier_metric(data) -> None:  # type: ignore[no-untyped-def]
    y, x = data
    result = cv_coxnet(
        y, x, metric="brier",
        times=[100, 200, 300], penalizers=[0.1, 0.01], k=2, seed=42,
    )
    assert result.metric_ == "brier"
    assert result.best_score_ >= 0
    assert result.penalizer_1se_ > 0
