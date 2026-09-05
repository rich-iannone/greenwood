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


# ---------------------------------------------------------------------------
# Detailed predict tests
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def coxnet(data):  # type: ignore[no-untyped-def]
    y, x = data
    return CoxNet(penalizer=0.05, l1_ratio=0.5).fit(y, x)


class TestPredictLP:
    def test_lp_shape(self, data, coxnet) -> None:  # type: ignore[no-untyped-def]
        _, x = data
        lp = coxnet.predict(x, type="lp")
        assert lp.shape == (coxnet.n_,)

    def test_lp_newdata_none(self, coxnet) -> None:  # type: ignore[no-untyped-def]
        lp = coxnet.predict(type="lp")
        assert lp.shape == (coxnet.n_,)

    def test_lp_numpy_input(self, data, coxnet) -> None:  # type: ignore[no-untyped-def]
        _, x = data
        lp_df = coxnet.predict(x, type="lp")
        lp_np = coxnet.predict(x.values, type="lp")
        np.testing.assert_allclose(lp_df, lp_np)

    def test_lp_polars_input(self, data, coxnet) -> None:  # type: ignore[no-untyped-def]
        import polars as pl

        _, x = data
        lp_pd = coxnet.predict(x, type="lp")
        lp_pl = coxnet.predict(pl.from_pandas(x), type="lp")
        np.testing.assert_allclose(lp_pd, lp_pl)

    def test_lp_single_subject(self, data, coxnet) -> None:  # type: ignore[no-untyped-def]
        _, x = data
        lp = coxnet.predict(x.iloc[:1], type="lp")
        assert lp.shape == (1,)

    def test_lp_centered(self, data, coxnet) -> None:  # type: ignore[no-untyped-def]
        _, x = data
        lp = coxnet.predict(x, type="lp")
        assert np.isclose(lp.mean(), 0.0, atol=0.5)


class TestPredictRisk:
    def test_risk_positive(self, data, coxnet) -> None:  # type: ignore[no-untyped-def]
        _, x = data
        risk = coxnet.predict(x, type="risk")
        assert np.all(risk > 0)

    def test_risk_equals_exp_lp(self, data, coxnet) -> None:  # type: ignore[no-untyped-def]
        _, x = data
        lp = coxnet.predict(x, type="lp")
        risk = coxnet.predict(x, type="risk")
        np.testing.assert_allclose(risk, np.exp(lp))

    def test_risk_shape(self, data, coxnet) -> None:  # type: ignore[no-untyped-def]
        _, x = data
        risk = coxnet.predict(x, type="risk")
        assert risk.shape == (coxnet.n_,)

    def test_risk_newdata_none(self, coxnet) -> None:  # type: ignore[no-untyped-def]
        risk = coxnet.predict(type="risk")
        assert risk.shape == (coxnet.n_,)
        assert np.all(risk > 0)

    def test_risk_single_subject(self, data, coxnet) -> None:  # type: ignore[no-untyped-def]
        _, x = data
        risk = coxnet.predict(x.iloc[:1], type="risk")
        assert risk.shape == (1,)
        assert risk[0] > 0


class TestPredictSurvival:
    def test_survival_range(self, data, coxnet) -> None:  # type: ignore[no-untyped-def]
        _, x = data
        surv = coxnet.predict(x.iloc[:3], type="survival", times=[180, 365], format="pandas")
        vals = surv.iloc[:, 1:].values
        assert np.all((vals >= 0) & (vals <= 1))

    def test_survival_monotone_decreasing(self, data, coxnet) -> None:  # type: ignore[no-untyped-def]
        _, x = data
        surv = coxnet.predict(
            x.iloc[:3], type="survival", times=[90, 180, 365, 730], format="pandas"
        )
        for col in ["subject_1", "subject_2", "subject_3"]:
            vals = surv[col].values
            assert np.all(np.diff(vals) <= 1e-10)

    def test_survival_default_times(self, data, coxnet) -> None:  # type: ignore[no-untyped-def]
        _, x = data
        surv = coxnet.predict(x.iloc[:2], type="survival", format="pandas")
        assert surv.shape[0] > 2
        assert list(surv.columns[:2]) == ["time", "subject_1"]

    def test_survival_newdata_none(self, coxnet) -> None:  # type: ignore[no-untyped-def]
        surv = coxnet.predict(type="survival", times=[180], format="pandas")
        assert surv.shape == (1, coxnet.n_ + 1)

    def test_survival_single_subject(self, data, coxnet) -> None:  # type: ignore[no-untyped-def]
        _, x = data
        surv = coxnet.predict(x.iloc[:1], type="survival", times=[180, 365], format="pandas")
        assert list(surv.columns) == ["time", "subject_1"]
        assert surv.shape == (2, 2)

    def test_survival_format_polars(self, data, coxnet) -> None:  # type: ignore[no-untyped-def]
        import polars as pl

        _, x = data
        surv = coxnet.predict(x.iloc[:2], type="survival", times=[180, 365], format="polars")
        assert isinstance(surv, pl.DataFrame)
        assert surv.columns == ["time", "subject_1", "subject_2"]

    def test_survival_format_pandas(self, data, coxnet) -> None:  # type: ignore[no-untyped-def]
        import pandas as pd

        _, x = data
        surv = coxnet.predict(x.iloc[:2], type="survival", times=[180, 365], format="pandas")
        assert isinstance(surv, pd.DataFrame)

    def test_survival_numpy_input(self, data, coxnet) -> None:  # type: ignore[no-untyped-def]
        _, x = data
        surv_df = coxnet.predict(x.iloc[:2], type="survival", times=[180], format="pandas")
        surv_np = coxnet.predict(x.iloc[:2].values, type="survival", times=[180], format="pandas")
        np.testing.assert_allclose(surv_df.iloc[:, 1:].values, surv_np.iloc[:, 1:].values)

    def test_survival_at_time_zero(self, data, coxnet) -> None:  # type: ignore[no-untyped-def]
        _, x = data
        surv = coxnet.predict(x.iloc[:2], type="survival", times=[0.0], format="pandas")
        vals = surv.iloc[:, 1:].values
        np.testing.assert_allclose(vals, 1.0)


class TestPredictConsistency:
    def test_zero_penalty_matches_coxph(self, data) -> None:  # type: ignore[no-untyped-def]
        y, x = data
        cn = CoxNet(penalizer=0.0).fit(y, x)
        ref = CoxPH(ties="breslow").fit(y, x)
        lp_cn = cn.predict(x.iloc[:5], type="lp")
        lp_ref = ref.predict(x.iloc[:5], type="lp")
        np.testing.assert_allclose(lp_cn, lp_ref, atol=1e-4)

    def test_zero_penalty_survival_matches_coxph(self, data) -> None:  # type: ignore[no-untyped-def]
        y, x = data
        cn = CoxNet(penalizer=0.0).fit(y, x)
        ref = CoxPH(ties="breslow").fit(y, x)
        times = [180, 365]
        s_cn = cn.predict(x.iloc[:3], type="survival", times=times, format="pandas")
        s_ref = ref.predict(x.iloc[:3], type="survival", times=times, format="pandas")
        np.testing.assert_allclose(s_cn.iloc[:, 1:].values, s_ref.iloc[:, 1:].values, atol=1e-3)

    def test_invalid_type_raises(self, data, coxnet) -> None:  # type: ignore[no-untyped-def]
        _, x = data
        with pytest.raises(ValueError, match="Unknown predict type"):
            coxnet.predict(x, type="hazard")


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
    x = np.random.default_rng(23).standard_normal((10, 2))
    with pytest.warns(UserWarning, match="fewer than"):
        cv_coxnet(y, x, k=5, penalizers=[0.1], seed=23)


def test_cv_coxnet_missing_rows_filtered(data) -> None:  # type: ignore[no-untyped-def]
    y, x = data
    import pandas as pd

    x_df = pd.DataFrame(x) if not isinstance(x, pd.DataFrame) else x.copy()
    x_df.iloc[0, 0] = np.nan
    result = cv_coxnet(y, x_df, penalizers=[0.1], k=2, seed=23)
    assert result.best_penalizer_ > 0


def test_cv_coxnet_brier_metric(data) -> None:  # type: ignore[no-untyped-def]
    y, x = data
    result = cv_coxnet(
        y,
        x,
        metric="brier",
        times=[100, 200, 300],
        penalizers=[0.1, 0.01],
        k=2,
        seed=23,
    )
    assert result.metric_ == "brier"
    assert result.best_score_ >= 0
    assert result.penalizer_1se_ > 0


# ── AIC / BIC ──


def test_aic_bic_unpenalized_matches_r(data) -> None:  # type: ignore[no-untyped-def]
    """penalizer=0 should reproduce R's AIC/BIC for the same Breslow Cox fit."""
    lung = gw.load_dataset("lung", backend="pandas")
    y = Surv.right(lung["time"], event=(lung["status"] == 2))
    cn = CoxNet(penalizer=0.0).fit(y, lung[["age", "sex"]])
    np.testing.assert_allclose(cn.effective_df(), 2.0)
    np.testing.assert_allclose(cn.aic(), 1490.159, atol=0.01)
    np.testing.assert_allclose(cn.bic(), 1496.371, atol=0.01)


def test_effective_df_lasso_equals_nonzero(data) -> None:  # type: ignore[no-untyped-def]
    """Pure lasso (l1_ratio=1): edf equals the count of non-zero coefficients."""
    y, x = data
    lasso = CoxNet(penalizer=0.05, l1_ratio=1.0).fit(y, x)
    assert lasso.effective_df() == float(np.count_nonzero(lasso.coef_))


def test_effective_df_ridge_less_than_p(data) -> None:  # type: ignore[no-untyped-def]
    """Ridge (l1_ratio=0) edf should be strictly less than p (the number of covariates)."""
    y, x = data
    ridge = CoxNet(penalizer=0.1, l1_ratio=0.0).fit(y, x)
    p = x.shape[1]
    edf = ridge.effective_df()
    assert 0 < edf < p


def test_effective_df_heavy_penalty_is_zero(data) -> None:  # type: ignore[no-untyped-def]
    """Heavy lasso penalty should zero all coefficients, giving edf=0."""
    y, x = data
    heavy = CoxNet(penalizer=10.0, l1_ratio=1.0).fit(y, x)
    assert heavy.effective_df() == 0.0
    assert heavy.aic() == heavy.bic()


def test_aic_decreases_then_increases(data) -> None:  # type: ignore[no-untyped-def]
    """AIC should have a minimum: too little penalty overfits, too much underfits."""
    y, x = data
    aics = [
        CoxNet(penalizer=lam, l1_ratio=0.0).fit(y, x).aic() for lam in [0.0, 0.01, 0.05, 0.5, 5.0]
    ]
    best_idx = int(np.argmin(aics))
    assert 0 < best_idx < len(aics) - 1


def test_bic_penalizes_more_than_aic(data) -> None:  # type: ignore[no-untyped-def]
    """BIC uses log(n_events) > 2 as the complexity multiplier, so BIC >= AIC."""
    y, x = data
    model = CoxNet(penalizer=0.05, l1_ratio=0.5).fit(y, x)
    assert model.bic() >= model.aic()


def test_tidy_columns(data) -> None:  # type: ignore[no-untyped-def]
    y, x = data
    model = CoxNet(penalizer=0.01).fit(y, x)
    tidy = gw.tidy(model, format="pandas")
    assert list(tidy.columns) == ["term", "estimate", "hazard_ratio"]
    assert len(tidy) == x.shape[1]


def test_tidy_exponentiate(data) -> None:  # type: ignore[no-untyped-def]
    y, x = data
    model = CoxNet(penalizer=0.01).fit(y, x)
    tidy = gw.tidy(model, exponentiate=True, format="pandas")
    np.testing.assert_allclose(tidy["estimate"].to_numpy(), model.hazard_ratio_)


def test_glance_fields(data) -> None:  # type: ignore[no-untyped-def]
    y, x = data
    model = CoxNet(penalizer=0.05, l1_ratio=0.5).fit(y, x)
    row = gw.glance(model, format="pandas").iloc[0]
    assert row["nevent"] == 165
    assert row["aic"] == pytest.approx(model.aic())
    assert row["bic"] == pytest.approx(model.bic())
    assert row["penalizer"] == 0.05
    assert row["l1_ratio"] == 0.5
    assert row["n_nonzero"] == int(np.count_nonzero(model.coef_))
