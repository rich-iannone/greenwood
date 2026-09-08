"""Unit tests for the Buckley-James rank-based AFT estimator.

There is no R package for Buckley-James available in this environment (`rms::bj` is not
installable for this R version), so correctness is validated internally: the algorithm must
reduce exactly to ordinary least squares when there is no censoring (every residual is exact,
so no Kaplan-Meier imputation ever happens), and should land close to a parametric `AFT` fit on
data reasonably well described by that distribution.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

import greenwood as gw
from greenwood import AFT, BuckleyJames, Surv


@pytest.fixture(scope="module")
def lung():
    return gw.load_dataset("lung", backend="pandas")


@pytest.fixture(scope="module")
def y(lung):
    return Surv.right(lung["time"], event=(lung["status"] == 2))


@pytest.fixture(scope="module")
def uncensored_data():
    rng = np.random.default_rng(0)
    n = 200
    age = rng.normal(60, 10, n)
    sex = rng.integers(0, 2, n)
    log_t = 6.0 + 0.01 * age + 0.3 * sex + rng.normal(0, 0.5, n)
    time = np.exp(log_t)
    design = np.column_stack([age, sex])
    return time, design, log_t


def test_invalid_max_iter() -> None:
    with pytest.raises(ValueError, match="max_iter"):
        BuckleyJames(max_iter=0)


def test_invalid_tol() -> None:
    with pytest.raises(ValueError, match="tol"):
        BuckleyJames(tol=0.0)


def test_invalid_n_boot() -> None:
    with pytest.raises(ValueError, match="n_boot"):
        BuckleyJames(n_boot=0)


def test_invalid_conf_level() -> None:
    with pytest.raises(ValueError, match="conf_level"):
        BuckleyJames(conf_level=1.5)


def test_no_censoring_matches_ols_exactly(uncensored_data) -> None:
    time, design, log_t = uncensored_data
    y_full = Surv.right(time, event=np.ones(time.shape[0]))
    bj = BuckleyJames().fit(y_full, design)

    assert bj.n_iter_ == 1
    assert bj.converged_

    x = np.column_stack([np.ones(design.shape[0]), design])
    ols, *_ = np.linalg.lstsq(x, log_t, rcond=None)
    np.testing.assert_allclose(bj.coef_, ols, atol=1e-9)


def test_converges_on_lung(lung, y) -> None:
    bj = BuckleyJames().fit(y, lung[["age", "sex"]])
    assert bj.converged_
    assert bj.n_iter_ >= 1


def test_close_to_aft_lognormal(lung, y) -> None:
    bj = BuckleyJames().fit(y, lung[["age", "sex"]])
    aft = AFT("lognormal").fit(y, lung[["age", "sex"]])
    np.testing.assert_allclose(bj.coef_, aft.coef_, atol=0.1)


def test_no_bootstrap_gives_nan_se(lung, y) -> None:
    bj = BuckleyJames().fit(y, lung[["age", "sex"]])
    assert np.all(np.isnan(bj.std_error_))
    assert np.all(np.isnan(bj.p_value_))
    assert np.all(np.isnan(bj.conf_low_))
    assert np.all(np.isnan(bj.conf_high_))


def test_bootstrap_gives_finite_se(lung, y) -> None:
    bj = BuckleyJames(n_boot=50, seed=0).fit(y, lung[["age", "sex"]])
    assert np.all(np.isfinite(bj.std_error_))
    assert np.all(bj.std_error_ > 0.0)
    assert np.all(bj.conf_low_ < bj.coef_)
    assert np.all(bj.conf_high_ > bj.coef_)


def test_non_convergence_warns(lung, y) -> None:
    with pytest.warns(UserWarning, match="did not converge"):
        bj = BuckleyJames(max_iter=1).fit(y, lung[["age", "sex"]])
    assert not bj.converged_


def test_predict_lp(lung, y) -> None:
    bj = BuckleyJames().fit(y, lung[["age", "sex"]])
    lp = bj.predict(lung[["age", "sex"]][:3], type="lp")
    x = np.column_stack([np.ones(3), lung[["age", "sex"]][:3].to_numpy()])
    np.testing.assert_allclose(lp, x @ bj.coef_)


def test_predict_survival_monotonic_and_bounded(lung, y) -> None:
    bj = BuckleyJames().fit(y, lung[["age", "sex"]])
    times = np.linspace(10, 900, 30)
    surv = bj.predict(lung[["age", "sex"]][:1], type="survival", times=times, format="pandas")
    s = surv["subject_1"].to_numpy()
    finite = s[~np.isnan(s)]
    assert np.all(np.diff(finite) <= 1e-9)
    assert np.all((finite >= 0.0) & (finite <= 1.0))


def test_predict_survival_requires_times(lung, y) -> None:
    bj = BuckleyJames().fit(y, lung[["age", "sex"]])
    with pytest.raises(ValueError, match="times"):
        bj.predict(lung[["age", "sex"]][:1], type="survival")


def test_predict_invalid_type(lung, y) -> None:
    bj = BuckleyJames().fit(y, lung[["age", "sex"]])
    with pytest.raises(ValueError, match="Unknown predict type"):
        bj.predict(lung[["age", "sex"]][:1], type="bogus")


def test_to_frame_columns(lung, y) -> None:
    bj = BuckleyJames().fit(y, lung[["age", "sex"]])
    df = bj.to_frame(format="pandas")
    assert list(df["term"]) == ["(Intercept)", "age", "sex"]
    assert list(df.columns) == [
        "term",
        "estimate",
        "std_error",
        "statistic",
        "p_value",
        "conf_low",
        "conf_high",
    ]


def test_glance_and_tidy(lung, y) -> None:
    bj = BuckleyJames().fit(y, lung[["age", "sex"]])
    g = gw.glance(bj, format="pandas")
    assert list(g.columns) == ["n", "nevent", "n_iter", "converged", "n_boot"]
    t = gw.tidy(bj, format="pandas")
    assert list(t["term"]) == ["(Intercept)", "age", "sex"]


def test_repr_unfit_and_fit() -> None:
    bj = BuckleyJames()
    assert "unfitted" in repr(bj)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        bj_fit = BuckleyJames().fit(
            Surv.right([1.0, 2.0, 3.0, 4.0], event=[1, 0, 1, 1]),
            np.array([[1.0], [2.0], [3.0], [4.0]]),
        )
    assert "BuckleyJames" in repr(bj_fit)


def test_counting_process_not_supported() -> None:
    y_counting = Surv.counting(start=[0, 1], stop=[2, 3], event=[1, 0])
    with pytest.raises(NotImplementedError, match="right-censored"):
        BuckleyJames().fit(y_counting, np.array([[1.0], [2.0]]))
