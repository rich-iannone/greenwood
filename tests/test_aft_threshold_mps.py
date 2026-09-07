"""Unit tests for AFT's threshold (three-parameter) models and MPS estimation."""

from __future__ import annotations

import warnings

import numpy as np
import pytest
from scipy.stats import weibull_min

import greenwood as gw
from greenwood import AFT, Surv
from greenwood._parametric import _error_quantile, _mean_survival_aft


@pytest.fixture
def lung_surv():  # type: ignore[no-untyped-def]
    df = gw.load_dataset("lung", backend="pandas")
    return df, Surv.right(df["time"], event=(df["status"] == 2))


@pytest.fixture(scope="module")
def three_param_data():  # type: ignore[no-untyped-def]
    # Shape < 1 so the ordinary likelihood is unbounded as the threshold approaches the
    # smallest observed time: the classic case method="mps" is meant to handle.
    rng = np.random.default_rng(1)
    n = 300
    gamma_true, shape_true, scale_true = 5.0, 0.7, 10.0
    t_raw = weibull_min.rvs(shape_true, scale=scale_true, size=n, random_state=rng)
    time = gamma_true + t_raw
    y = Surv.right(time, event=np.ones(n))
    return y, dict(gamma=gamma_true, shape=shape_true, scale=scale_true)


# -- constructor validation ------------------------------------------------------------


def test_invalid_method() -> None:
    with pytest.raises(ValueError, match="method"):
        AFT(method="bogus")


def test_threshold_gengamma_not_supported() -> None:
    with pytest.raises(ValueError, match="gengamma"):
        AFT(dist="gengamma", threshold=True)


def test_threshold_mle_warns() -> None:
    with pytest.warns(UserWarning, match="method='mps'"):
        AFT(threshold=True, method="mle")


def test_threshold_mps_no_warning() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        AFT(threshold=True, method="mps")


def test_defaults_no_warning() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        AFT()


# -- regression guard: threshold=False must be untouched --------------------------------


def test_threshold_false_unchanged(lung_surv) -> None:  # type: ignore[no-untyped-def]
    df, y = lung_surv
    a = AFT("weibull").fit(y, df[["age", "sex"]])
    b = AFT("weibull", method="mle", threshold=False).fit(y, df[["age", "sex"]])
    np.testing.assert_allclose(a.coef_, b.coef_)
    assert a.scale_ == b.scale_
    assert a.loglik_ == b.loglik_
    assert b.threshold_ == 0.0


# -- MPS without a threshold should track MLE closely on real data ----------------------


def test_mps_no_threshold_close_to_mle(lung_surv) -> None:  # type: ignore[no-untyped-def]
    df, y = lung_surv
    mle = AFT("weibull").fit(y, df[["age", "sex"]])
    mps = AFT("weibull", method="mps").fit(y, df[["age", "sex"]])
    np.testing.assert_allclose(mps.coef_, mle.coef_, atol=0.02)
    np.testing.assert_allclose(mps.scale_, mle.scale_, atol=0.05)
    assert mps.threshold_ == 0.0


# -- recovering a genuine three-parameter model ------------------------------------------


def test_threshold_mle_recovers_true_params(three_param_data) -> None:  # type: ignore[no-untyped-def]
    y, truth = three_param_data
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        aft = AFT("weibull", threshold=True).fit(y, np.zeros((y.n, 0)))
    assert aft.threshold_ == pytest.approx(truth["gamma"], abs=0.5)
    assert 1.0 / aft.scale_ == pytest.approx(truth["shape"], abs=0.2)
    assert np.exp(aft.coef_[0]) == pytest.approx(truth["scale"], abs=2.0)


def test_threshold_mps_recovers_true_params(three_param_data) -> None:  # type: ignore[no-untyped-def]
    y, truth = three_param_data
    aft = AFT("weibull", method="mps", threshold=True).fit(y, np.zeros((y.n, 0)))
    assert aft.threshold_ == pytest.approx(truth["gamma"], abs=0.5)
    assert 1.0 / aft.scale_ == pytest.approx(truth["shape"], abs=0.2)
    assert np.exp(aft.coef_[0]) == pytest.approx(truth["scale"], abs=2.0)


def test_threshold_mps_less_boundary_biased_than_mle(three_param_data) -> None:  # type: ignore[no-untyped-def]
    # The classic qualitative signature: MLE's threshold estimate sits closer to the smallest
    # observed time than MPS's, since the (unbounded, for shape < 1) likelihood keeps improving
    # as the threshold approaches that boundary, while MPS's bounded objective does not.
    y, _ = three_param_data
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        aft_mle = AFT("weibull", threshold=True).fit(y, np.zeros((y.n, 0)))
    aft_mps = AFT("weibull", method="mps", threshold=True).fit(y, np.zeros((y.n, 0)))
    t_min = float(y.stop.min())
    assert (t_min - aft_mle.threshold_) < (t_min - aft_mps.threshold_)


# -- prediction formulas with a threshold -------------------------------------------------


@pytest.fixture
def threshold_fit(lung_surv):  # type: ignore[no-untyped-def]
    df, y = lung_surv
    aft = AFT("weibull", method="mps", threshold=True).fit(y, df[["age", "sex"]])
    return df, aft


def test_survival_is_one_below_threshold(threshold_fit) -> None:  # type: ignore[no-untyped-def]
    df, aft = threshold_fit
    below = aft.threshold_ * np.array([0.1, 0.5, 0.999])
    sf = aft.predict(df[["age", "sex"]][:1], type="survival", times=below, format="pandas")
    np.testing.assert_allclose(sf["subject_1"], 1.0)


def test_quantile_shifted_by_threshold(threshold_fit) -> None:  # type: ignore[no-untyped-def]
    df, aft = threshold_fit
    x_row = df[["age", "sex"]][:1]
    mu = float(aft.predict(x_row, type="lp")[0])
    w = float(_error_quantile("weibull", np.array([0.5]))[0])
    expected = aft.threshold_ + np.exp(mu + aft.scale_ * w)
    q = aft.predict(x_row, type="quantile", p=0.5, format="pandas")
    assert float(q["subject_1"][0]) == pytest.approx(expected)


def test_mean_shifted_by_threshold(threshold_fit) -> None:  # type: ignore[no-untyped-def]
    df, aft = threshold_fit
    x_row = df[["age", "sex"]][:1]
    mu = float(aft.predict(x_row, type="lp")[0])
    mean_2p = float(_mean_survival_aft("weibull", np.array([mu]), aft.scale_)[0])
    expected = aft.threshold_ + mean_2p
    mean_pred = float(aft.predict(x_row, type="mean")[0])
    assert mean_pred == pytest.approx(expected)


def test_rmst_below_threshold_equals_tau(threshold_fit) -> None:  # type: ignore[no-untyped-def]
    df, aft = threshold_fit
    x_row = df[["age", "sex"]][:1]
    tau_small = aft.threshold_ * 0.5
    rmst = float(aft.predict(x_row, type="rmst", tau=tau_small)[0])
    assert rmst == pytest.approx(tau_small)


def test_predict_quantile_ci_shifted(threshold_fit) -> None:  # type: ignore[no-untyped-def]
    df, aft = threshold_fit
    out = aft.predict_quantile(df[["age", "sex"]][:2], p=0.5, ci=True, format="pandas")
    assert np.all(out["subject_1_lower"] >= aft.threshold_)
    assert np.all(out["subject_1"] >= aft.threshold_)


# -- glance / repr / to_frame -------------------------------------------------------------


def test_glance_includes_threshold_and_method(lung_surv) -> None:  # type: ignore[no-untyped-def]
    df, y = lung_surv
    aft = AFT("weibull", method="mps", threshold=True).fit(y, df[["age", "sex"]])
    g = gw.glance(aft, format="pandas")
    assert "threshold" in g.columns
    assert "method" in g.columns
    assert g["method"][0] == "mps"
    assert g["threshold"][0] == pytest.approx(aft.threshold_)


def test_glance_omits_threshold_when_not_used(lung_surv) -> None:  # type: ignore[no-untyped-def]
    df, y = lung_surv
    aft = AFT("weibull").fit(y, df[["age", "sex"]])
    g = gw.glance(aft, format="pandas")
    assert "threshold" not in g.columns


def test_repr_shows_threshold(lung_surv) -> None:  # type: ignore[no-untyped-def]
    df, y = lung_surv
    aft = AFT("weibull", method="mps", threshold=True).fit(y, df[["age", "sex"]])
    assert "Threshold" in repr(aft)
    assert "method='mps'" in repr(aft)


def test_to_frame_unaffected_by_threshold(lung_surv) -> None:  # type: ignore[no-untyped-def]
    df, y = lung_surv
    aft = AFT("weibull", method="mps", threshold=True).fit(y, df[["age", "sex"]])
    tf = aft.to_frame(format="pandas")
    assert list(tf["term"]) == ["(Intercept)", "age", "sex"]


# -- residuals still work with a threshold -------------------------------------------------


def test_residuals_finite_with_threshold(threshold_fit) -> None:  # type: ignore[no-untyped-def]
    _, aft = threshold_fit
    for kind in ("response", "cox_snell", "martingale", "deviance"):
        vals = aft.residuals(kind)
        assert np.all(np.isfinite(vals))
