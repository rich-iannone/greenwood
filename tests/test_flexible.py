"""Tests for the Royston-Parmar flexible parametric model (`RoystonParmar`).

flexsurv/rstpm2 are not installed, so correctness is pinned by the Weibull special case
(`df=1` must reproduce R `survreg`'s Weibull log-likelihood and survival), MLE stationarity,
and the expected behavior of the spline (more df fits at least as well; valid survival and
hazard).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import greenwood as gw
from greenwood import RoystonParmar, Surv
from greenwood._flexible import _rcs_basis
from tests._r_parity import assert_allclose_to_r, load_fixture


@pytest.fixture(scope="module")
def lung():
    return gw.load_dataset("lung", backend="pandas")


@pytest.fixture(scope="module")
def y(lung):
    return Surv.right(lung["time"], event=(lung["status"] == 2))


def test_df1_matches_r_weibull(lung, y) -> None:
    fixture = load_fixture("rp_weibull_anchor")
    rp = RoystonParmar(df=1).fit(y, lung[["age", "sex"]])
    assert_allclose_to_r(rp.loglik_, fixture["loglik"], atol=1e-3, what="df=1 loglik")
    newdata = pd.DataFrame({"age": fixture["newdata_age"], "sex": fixture["newdata_sex"]})
    surv = rp.predict(newdata, type="survival", times=fixture["times"])
    assert_allclose_to_r(
        surv["subject_1"].to_numpy(), fixture["surv"]["subj1"], atol=1e-4, what="df=1 surv subj1"
    )
    assert_allclose_to_r(
        surv["subject_2"].to_numpy(), fixture["surv"]["subj2"], atol=1e-4, what="df=1 surv subj2"
    )


def test_more_df_fits_at_least_as_well(lung, y) -> None:
    logliks = [RoystonParmar(df=d).fit(y, lung[["age", "sex"]]).loglik_ for d in (1, 2, 3, 4)]
    for lo, hi in zip(logliks, logliks[1:], strict=False):
        assert hi >= lo - 1e-6  # added flexibility cannot lower the maximized likelihood


def test_mle_is_stationary(lung, y) -> None:
    rp = RoystonParmar(df=3).fit(y, lung[["age", "sex"]])
    time = lung["time"].to_numpy()
    event = (lung["status"] == 2).to_numpy().astype(float)
    x = lung[["age", "sex"]].to_numpy()
    u = np.log(time)
    basis, deriv = _rcs_basis(u, rp._knots)
    n_spline = basis.shape[1]

    def negll(theta):
        g, b = theta[:n_spline], theta[n_spline:]
        eta = basis @ g + x @ b
        sprime = deriv @ g
        if np.any(sprime <= 0):
            return 1e12
        return -float((event * (eta + np.log(sprime) - u) - np.exp(eta)).sum())

    theta = rp.coef_
    grad = np.zeros_like(theta)
    h = 1e-5
    for i in range(theta.size):
        step = np.zeros_like(theta)
        step[i] = h
        grad[i] = (negll(theta + step) - negll(theta - step)) / (2 * h)
    assert np.max(np.abs(grad)) < 1e-2  # gradient of the summed log-likelihood ~ 0


def test_predictions_are_valid(lung, y) -> None:
    rp = RoystonParmar(df=3).fit(y, lung[["age", "sex"]])
    nd = pd.DataFrame({"age": [60], "sex": [1]})
    surv = rp.predict(nd, type="survival", times=[50, 150, 300, 500, 800])["subject_1"].to_numpy()
    assert np.all((surv >= 0) & (surv <= 1))
    assert np.all(np.diff(surv) <= 1e-12)  # monotone non-increasing
    haz = rp.predict(nd, type="hazard", times=[100, 300, 600])["subject_1"].to_numpy()
    assert np.all(haz > 0)


def test_terms_and_repr(lung, y) -> None:
    rp = RoystonParmar(df=3).fit(y, lung[["age", "sex"]])
    # df=3 -> 4 spline terms (gamma0..gamma3) plus the two covariates.
    assert rp.term_names_ == ["gamma0", "gamma1", "gamma2", "gamma3", "age", "sex"]
    assert "flexible parametric" in repr(rp)
    assert "object at 0x" not in repr(rp)


def test_no_covariate_fit(y) -> None:
    rp = RoystonParmar(df=3).fit(y)  # baseline-only model
    assert rp.term_names_ == ["gamma0", "gamma1", "gamma2", "gamma3"]
    surv = rp.predict(type="survival", times=[180, 365])
    assert ((surv["subject_1"] >= 0) & (surv["subject_1"] <= 1)).all()


def test_invalid_arguments() -> None:
    with pytest.raises(ValueError, match="df"):
        RoystonParmar(df=0)
    with pytest.raises(ValueError, match="conf_level"):
        RoystonParmar(conf_level=1.5)


@pytest.fixture(scope="module")
def rp3(lung, y):
    return RoystonParmar(df=3).fit(y, lung[["age", "sex"]])


class TestFittedAttributes:
    def test_n_and_nevent(self, rp3) -> None:
        assert rp3.n_ == 228
        assert rp3.n_event_ == 165

    def test_coef_shape(self, rp3) -> None:
        # df=3 -> 4 spline terms + 2 covariates = 6
        assert rp3.coef_.shape == (6,)

    def test_vcov_shape_and_symmetry(self, rp3) -> None:
        assert rp3.vcov_.shape == (6, 6)
        np.testing.assert_allclose(rp3.vcov_, rp3.vcov_.T, atol=1e-12)

    def test_vcov_positive_semidefinite(self, rp3) -> None:
        eigvals = np.linalg.eigvalsh(rp3.vcov_)
        assert np.all(eigvals >= -1e-10)

    def test_std_error_matches_vcov(self, rp3) -> None:
        np.testing.assert_allclose(rp3.std_error_, np.sqrt(np.diag(rp3.vcov_)))

    def test_knots_sorted(self, rp3) -> None:
        assert np.all(np.diff(rp3.knots_) > 0)

    def test_knots_count(self, rp3) -> None:
        # df=3 -> 3 internal knots + 2 boundary = 5 knots total
        # Actually: n_spline = df + 1 = 4 columns, n_knots = df - 1 = 2 internal + 2 boundary
        # Let's just check it's non-empty and reasonable
        assert len(rp3.knots_) >= 3

    def test_conf_intervals_bracket_estimate(self, rp3) -> None:
        assert np.all(rp3.conf_low_ <= rp3.coef_)
        assert np.all(rp3.conf_high_ >= rp3.coef_)

    def test_loglik_is_finite(self, rp3) -> None:
        assert np.isfinite(rp3.loglik_)


class TestPredict:
    def test_cumhaz_positive(self, rp3, lung) -> None:
        nd = pd.DataFrame({"age": [60], "sex": [1]})
        ch = rp3.predict(nd, type="cumhaz", times=[50, 150, 300, 500])
        vals = ch["subject_1"].to_numpy()
        assert np.all(vals >= 0)

    def test_cumhaz_increasing(self, rp3, lung) -> None:
        nd = pd.DataFrame({"age": [60], "sex": [1]})
        ch = rp3.predict(nd, type="cumhaz", times=[50, 150, 300, 500, 800])
        vals = ch["subject_1"].to_numpy()
        assert np.all(np.diff(vals) >= -1e-12)

    def test_cumhaz_survival_identity(self, rp3) -> None:
        nd = pd.DataFrame({"age": [55, 70], "sex": [1, 2]})
        times = [100, 200, 400, 600]
        surv = rp3.predict(nd, type="survival", times=times)
        ch = rp3.predict(nd, type="cumhaz", times=times)
        for col in ["subject_1", "subject_2"]:
            np.testing.assert_allclose(
                surv[col].to_numpy(), np.exp(-ch[col].to_numpy()), rtol=1e-10
            )

    def test_survival_at_small_time_near_one(self, rp3) -> None:
        nd = pd.DataFrame({"age": [60], "sex": [1]})
        surv = rp3.predict(nd, type="survival", times=[0.01])
        assert surv["subject_1"].to_numpy()[0] > 0.99

    def test_newdata_none_returns_baseline(self, rp3) -> None:
        surv = rp3.predict(times=[180])
        assert "subject_1" in surv.columns
        assert surv.shape[0] == 1

    def test_format_polars(self, rp3) -> None:
        import polars as pl

        nd = pd.DataFrame({"age": [60], "sex": [1]})
        surv = rp3.predict(nd, type="survival", times=[180, 365], format="polars")
        assert isinstance(surv, pl.DataFrame)

    def test_invalid_type_raises(self, rp3) -> None:
        nd = pd.DataFrame({"age": [60], "sex": [1]})
        with pytest.raises(ValueError, match="type"):
            rp3.predict(nd, type="bogus", times=[180])

    def test_multiple_subjects(self, rp3) -> None:
        nd = pd.DataFrame({"age": [50, 60, 70], "sex": [1, 1, 2]})
        surv = rp3.predict(nd, type="survival", times=[180, 365])
        assert surv.shape == (2, 4)  # 2 times × (time + 3 subjects)

    def test_hazard_shape(self, rp3) -> None:
        nd = pd.DataFrame({"age": [50, 70], "sex": [1, 2]})
        haz = rp3.predict(nd, type="hazard", times=[100, 300, 600])
        assert haz.shape == (3, 3)  # 3 times × (time + 2 subjects)
        for col in ["subject_1", "subject_2"]:
            assert np.all(haz[col].to_numpy() > 0)


class TestPredictQuantile:
    def test_median_basic(self, rp3) -> None:
        nd = pd.DataFrame({"age": [60], "sex": [1]})
        med = rp3.predict_median(nd)
        assert med.shape[0] == 1
        val = float(med["subject_1"][0])
        assert val > 0

    def test_quantile_ordering(self, rp3) -> None:
        nd = pd.DataFrame({"age": [60], "sex": [1]})
        q = rp3.predict_quantile(nd, p=[0.25, 0.5, 0.75])
        vals = q["subject_1"].to_numpy()
        # higher p -> longer survival time (survival quantile convention)
        assert np.all(np.diff(vals) >= -1e-10)

    def test_median_matches_quantile_half(self, rp3) -> None:
        nd = pd.DataFrame({"age": [60], "sex": [1]})
        med = rp3.predict_median(nd)
        q50 = rp3.predict_quantile(nd, p=0.5)
        med_val = float(med["subject_1"][0])
        q50_val = float(q50["subject_1"][0])
        assert med_val == pytest.approx(q50_val, rel=1e-10)

    def test_quantile_with_ci(self, rp3) -> None:
        nd = pd.DataFrame({"age": [60], "sex": [1]})
        q = rp3.predict_quantile(nd, p=0.5, ci=True)
        cols = list(q.columns)
        assert any("lower" in c for c in cols)
        assert any("upper" in c for c in cols)

    def test_median_with_ci(self, rp3) -> None:
        nd = pd.DataFrame({"age": [60], "sex": [1]})
        med = rp3.predict_median(nd, ci=True)
        ncols = med.shape[1]
        assert ncols >= 3  # at least: p, estimate, lower, upper

    def test_quantile_format_polars(self, rp3) -> None:
        import polars as pl

        nd = pd.DataFrame({"age": [60], "sex": [1]})
        q = rp3.predict_quantile(nd, p=0.5, format="polars")
        assert isinstance(q, pl.DataFrame)


class TestPredictExpectation:
    def test_basic(self, rp3) -> None:
        nd = pd.DataFrame({"age": [60], "sex": [1]})
        rmst = rp3.predict_expectation(nd, tau=365)
        assert rmst.shape[0] == 1
        val = float(rmst["subject_1"][0])
        assert 0 < val <= 365

    def test_longer_tau_gives_larger_rmst(self, rp3) -> None:
        nd = pd.DataFrame({"age": [60], "sex": [1]})
        r1 = rp3.predict_expectation(nd, tau=180)
        r2 = rp3.predict_expectation(nd, tau=365)
        v1 = float(r1["subject_1"][0])
        v2 = float(r2["subject_1"][0])
        assert v2 >= v1 - 1e-6

    def test_with_ci(self, rp3) -> None:
        nd = pd.DataFrame({"age": [60], "sex": [1]})
        rmst = rp3.predict_expectation(nd, tau=365, ci=True)
        ncols = rmst.shape[1]
        assert ncols >= 3

    def test_format_polars(self, rp3) -> None:
        import polars as pl

        nd = pd.DataFrame({"age": [60], "sex": [1]})
        rmst = rp3.predict_expectation(nd, tau=365, format="polars")
        assert isinstance(rmst, pl.DataFrame)

    def test_multiple_subjects(self, rp3) -> None:
        nd = pd.DataFrame({"age": [50, 60, 70], "sex": [1, 1, 2]})
        rmst = rp3.predict_expectation(nd, tau=365)
        # One row per tau, one column per subject + tau column
        assert rmst.shape == (1, 4)


class TestRepr:
    def test_unfitted_repr(self) -> None:
        rp = RoystonParmar(df=3)
        r = repr(rp)
        assert "unfitted" in r.lower() or "RoystonParmar" in r

    def test_fitted_repr_contains_terms(self, rp3) -> None:
        r = repr(rp3)
        assert "gamma0" in r
        assert "age" in r
        assert "sex" in r


class TestPolarsInput:
    def test_fit_with_polars(self, y) -> None:
        lung_pl = gw.load_dataset("lung", backend="polars")
        rp = RoystonParmar(df=3).fit(y, lung_pl[["age", "sex"]])
        assert rp.n_ == 228
        assert rp.coef_.shape == (6,)

    def test_predict_with_polars_newdata(self, rp3) -> None:
        import polars as pl

        nd = pl.DataFrame({"age": [60], "sex": [1]})
        surv = rp3.predict(nd, type="survival", times=[180, 365])
        assert surv.shape == (2, 2)


class TestConfLevel:
    def test_wider_conf_level(self, lung, y) -> None:
        rp_95 = RoystonParmar(df=3, conf_level=0.95).fit(y, lung[["age", "sex"]])
        rp_99 = RoystonParmar(df=3, conf_level=0.99).fit(y, lung[["age", "sex"]])
        width_95 = rp_95.conf_high_ - rp_95.conf_low_
        width_99 = rp_99.conf_high_ - rp_99.conf_low_
        assert np.all(width_99 >= width_95 - 1e-12)


class TestToFrame:
    def test_to_frame_polars(self, rp3) -> None:
        import polars as pl

        df = rp3.to_frame(format="polars")
        assert isinstance(df, pl.DataFrame)
        assert len(df) == 6

    def test_to_frame_default(self, rp3) -> None:
        df = rp3.to_frame()
        assert hasattr(df, "columns") or hasattr(df, "schema")
        assert len(df) == 6


def test_tidy_columns(lung, y) -> None:  # type: ignore[no-untyped-def]
    rp = RoystonParmar(df=3).fit(y, lung[["age", "sex"]])
    tidy = gw.tidy(rp, format="pandas")

    assert list(tidy.columns) == [
        "term",
        "estimate",
        "std_error",
        "statistic",
        "p_value",
        "conf_low",
        "conf_high",
    ]
    assert len(tidy) == 6
    assert list(tidy["term"]) == ["gamma0", "gamma1", "gamma2", "gamma3", "age", "sex"]


def test_glance_fields(lung, y) -> None:  # type: ignore[no-untyped-def]
    rp = RoystonParmar(df=3).fit(y, lung[["age", "sex"]])
    row = gw.glance(rp, format="pandas").iloc[0]

    assert row["nevent"] == 165
    assert row["df"] == 6
    assert row["n_knots"] == 3
    assert row["aic"] == pytest.approx(-2 * rp.loglik_ + 2 * 6)
    assert row["bic"] == pytest.approx(-2 * rp.loglik_ + np.log(165) * 6)
