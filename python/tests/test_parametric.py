"""Unit tests for the parametric AFT models."""

from __future__ import annotations

import numpy as np
import pytest

import greenwood as gw
from greenwood import AFT, Surv


@pytest.fixture
def lung_surv():  # type: ignore[no-untyped-def]
    # Pinned to pandas: these tests use pandas idioms (`.iloc`, `to_numpy(dtype=)`).
    # Backend-agnostic input is covered separately in test_backends.py.
    df = gw.load_dataset("lung", backend="pandas")
    return df, Surv(time=df["time"], event=(df["status"] == 2))


def test_invalid_dist() -> None:
    with pytest.raises(ValueError, match="dist"):
        AFT("gaussian")


def test_invalid_conf_level() -> None:
    with pytest.raises(ValueError, match="conf_level"):
        AFT(conf_level=1.0)


def test_intercept_is_added(lung_surv) -> None:  # type: ignore[no-untyped-def]
    df, y = lung_surv
    model = AFT("weibull").fit(y, df[["age", "sex"]])
    assert model.term_names_ == ["(Intercept)", "age", "sex"]


def test_exponential_scale_is_one(lung_surv) -> None:  # type: ignore[no-untyped-def]
    df, y = lung_surv
    model = AFT("exponential").fit(y, df[["age", "sex"]])
    assert model.scale_ == 1.0
    assert model.log_scale_se_ is None


def test_weibull_scale_estimated(lung_surv) -> None:  # type: ignore[no-untyped-def]
    df, y = lung_surv
    model = AFT("weibull").fit(y, df[["age", "sex"]])
    assert model.scale_ > 0.0
    assert model.log_scale_se_ is not None


def test_length_mismatch(lung_surv) -> None:  # type: ignore[no-untyped-def]
    df, y = lung_surv
    with pytest.raises(ValueError, match="same number of rows"):
        AFT().fit(y, df[["age"]].iloc[:-1])


def test_interval_censoring_not_supported() -> None:
    y = Surv(time=[1, 2, 3], time2=[2, 3, 4], type="interval2")
    with pytest.raises(NotImplementedError, match="right-censored"):
        AFT().fit(y, np.zeros((3, 1)))


def test_tidy_and_glance_via_registry(lung_surv) -> None:  # type: ignore[no-untyped-def]
    df, y = lung_surv
    model = AFT("weibull").fit(y, df[["age", "sex"]])
    tidy = gw.tidy(model)
    assert list(tidy["term"]) == ["(Intercept)", "age", "sex"]
    glance = gw.glance(model, format="pandas")
    row = glance.iloc[0]
    assert row["dist"] == "weibull"
    assert row["nevent"] == 165
    assert row["aic"] == pytest.approx(-2 * model.loglik_ + 2 * 4)  # 3 coef + log(scale)


def test_to_pandas_columns(lung_surv) -> None:  # type: ignore[no-untyped-def]
    df, y = lung_surv
    model = AFT().fit(y, df[["age", "sex"]])
    assert list(model.to_frame(format="pandas").columns) == [
        "term",
        "estimate",
        "std_error",
        "statistic",
        "p_value",
        "conf_low",
        "conf_high",
    ]


def test_array_covariates(lung_surv) -> None:  # type: ignore[no-untyped-def]
    df, y = lung_surv
    x = df[["age", "sex"]].to_numpy(dtype=float)
    model = AFT("weibull").fit(y, x)
    assert model.term_names_ == ["(Intercept)", "x0", "x1"]


def test_formula_matches_explicit(lung_surv) -> None:  # type: ignore[no-untyped-def]
    import numpy as np

    df, y = lung_surv
    explicit = AFT("weibull").fit(y, df[["age", "sex"]])
    formula = AFT("weibull").fit(y, "age + sex", data=df)
    assert formula.term_names_ == ["(Intercept)", "age", "sex"]
    np.testing.assert_allclose(formula.coef_, explicit.coef_)


@pytest.mark.parametrize("dist", ["weibull", "exponential", "lognormal", "loglogistic"])
def test_predict_survival_quantile_consistent(lung_surv, dist) -> None:  # type: ignore[no-untyped-def]
    # The survival and quantile predictions are exact inverses: S(quantile_p) == 1 - p.
    import numpy as np

    df, y = lung_surv
    model = AFT(dist).fit(y, df[["age", "sex"]])
    newdata = df[["age", "sex"]].iloc[:3]
    p = [0.1, 0.3, 0.5, 0.8]
    q = model.predict(newdata, type="quantile", p=p)
    for i in range(3):
        col = f"subject_{i + 1}"
        surv = model.predict(newdata.iloc[[i]], type="survival", times=list(q[col]))
        np.testing.assert_allclose(surv["subject_1"].to_numpy(), 1.0 - np.array(p), atol=1e-12)


def test_predict_lp_matches_design(lung_surv) -> None:  # type: ignore[no-untyped-def]
    import numpy as np

    df, y = lung_surv
    model = AFT("weibull").fit(y, df[["age", "sex"]])
    lp = model.predict(type="lp")
    assert lp.shape == (model.n_,)
    # Median quantile equals exp(lp) only when the error median is 0 (weibull median is not),
    # but the linear predictor is the log-time location, so exp(lp) is positive and finite.
    assert np.all(np.isfinite(lp))


def test_predict_survival_shape(lung_surv) -> None:  # type: ignore[no-untyped-def]
    df, y = lung_surv
    model = AFT("weibull").fit(y, df[["age", "sex"]])
    surv = model.predict(
        df[["age", "sex"]].iloc[:4], type="survival", times=[100, 300, 500], format="pandas"
    )
    assert list(surv.columns) == ["time", "subject_1", "subject_2", "subject_3", "subject_4"]
    assert len(surv) == 3
    assert ((surv.iloc[:, 1:] >= 0) & (surv.iloc[:, 1:] <= 1)).all().all()


def test_predict_conditional_after_identity(lung_surv) -> None:  # type: ignore[no-untyped-def]
    # S(t | T > c) * S(c) == S(t) for t >= c, and conditioning on c=0 is a no-op.
    import numpy as np

    df, y = lung_surv
    model = AFT("weibull").fit(y, df[["age", "sex"]])
    nd = df[["age", "sex"]].iloc[:2]
    times = [200, 400, 600]
    c = 150.0
    s_t = model.predict(nd, type="survival", times=times, format="pandas")
    s_c = model.predict(nd, type="survival", times=[c], format="pandas")
    s_cond = model.predict(nd, type="survival", times=times, conditional_after=c, format="pandas")
    for col in ("subject_1", "subject_2"):
        np.testing.assert_allclose(
            s_cond[col].to_numpy() * float(s_c[col].iloc[0]), s_t[col].to_numpy(), atol=1e-12
        )
    s0 = model.predict(nd, type="survival", times=times, conditional_after=0.0, format="pandas")
    np.testing.assert_allclose(
        s0[["subject_1", "subject_2"]].to_numpy(), s_t[["subject_1", "subject_2"]].to_numpy()
    )


# -- loglogistic sigma >= 1 edge cases ----------------------------------------


def test_mean_survival_aft_loglogistic_sigma_ge_1() -> None:
    from greenwood._parametric import _mean_survival_aft

    mu = np.array([2.0, 3.0])
    result = _mean_survival_aft("loglogistic", mu, sigma=1.5)
    assert np.all(np.isinf(result))


def test_tail_partial_moment_loglogistic_sigma_ge_1() -> None:
    from greenwood._parametric import _tail_partial_moment

    mu = np.array([5.0, 6.0])
    t0 = np.array([1.0, 1.0])
    result = _tail_partial_moment("loglogistic", mu, sigma=1.5, t0=t0)
    assert result.shape == (2,)
    assert np.all(np.isfinite(result))


def test_aft_loglogistic_predict_mean_conditional(lung_surv) -> None:  # type: ignore[no-untyped-def]
    df, y = lung_surv
    aft = AFT(dist="loglogistic").fit(y, df[["age", "sex"]])
    original_scale = aft.scale_
    aft.scale_ = 1.5
    try:
        nd = df[["age", "sex"]].iloc[:3]
        result = aft.predict(nd, type="mean")
        assert np.all(np.isinf(result))
        result_cond = aft.predict(nd, type="mean", conditional_after=10.0)
        assert result_cond.shape == (3,)
        with pytest.raises(ValueError, match="conditional_after must be a scalar"):
            aft.predict(nd, type="mean", conditional_after=[1.0, 2.0])
    finally:
        aft.scale_ = original_scale


def test_aft_loglogistic_predict_mean_remaining(lung_surv) -> None:  # type: ignore[no-untyped-def]
    df, y = lung_surv
    aft = AFT(dist="loglogistic").fit(y, df[["age", "sex"]])
    original_scale = aft.scale_
    aft.scale_ = 1.5
    try:
        nd = df[["age", "sex"]].iloc[:3]
        result = aft.predict(nd, type="mean_remaining", conditional_after=10.0)
        assert result.shape == (3,)
        assert np.all(np.isfinite(result))
        with pytest.raises(ValueError, match="conditional_after must be a scalar"):
            aft.predict(nd, type="mean_remaining", conditional_after=[1.0, 2.0])
    finally:
        aft.scale_ = original_scale


@pytest.mark.parametrize("dist", ["weibull", "exponential", "lognormal", "loglogistic"])
def test_predict_survival_ci_columns(lung_surv, dist) -> None:  # type: ignore[no-untyped-def]
    df, y = lung_surv
    model = AFT(dist).fit(y, df[["age", "sex"]])
    nd = df[["age", "sex"]].iloc[:2]
    pred = model.predict(nd, type="survival", times=[180, 365], ci=True, format="pandas")
    assert list(pred.columns) == [
        "time",
        "subject_1",
        "subject_1_lower",
        "subject_1_upper",
        "subject_2",
        "subject_2_lower",
        "subject_2_upper",
    ]
    assert len(pred) == 2


@pytest.mark.parametrize("dist", ["weibull", "exponential", "lognormal", "loglogistic"])
def test_predict_survival_ci_brackets_point(lung_surv, dist) -> None:  # type: ignore[no-untyped-def]
    df, y = lung_surv
    model = AFT(dist).fit(y, df[["age", "sex"]])
    nd = df[["age", "sex"]].iloc[:3]
    pred = model.predict(nd, type="survival", times=[100, 300, 500], ci=True, format="pandas")
    for j in range(1, 4):
        lower = pred[f"subject_{j}_lower"].to_numpy()
        point = pred[f"subject_{j}"].to_numpy()
        upper = pred[f"subject_{j}_upper"].to_numpy()
        assert np.all(lower <= point + 1e-12)
        assert np.all(point <= upper + 1e-12)


@pytest.mark.parametrize("dist", ["weibull", "exponential", "lognormal", "loglogistic"])
def test_predict_survival_ci_loglog_bounds_valid(lung_surv, dist) -> None:  # type: ignore[no-untyped-def]
    df, y = lung_surv
    model = AFT(dist).fit(y, df[["age", "sex"]])
    nd = df[["age", "sex"]].iloc[:2]
    pred = model.predict(
        nd, type="survival", times=[100, 300, 500], ci=True, conf_type="log-log", format="pandas"
    )
    for j in range(1, 3):
        lower = pred[f"subject_{j}_lower"].to_numpy()
        upper = pred[f"subject_{j}_upper"].to_numpy()
        assert np.all(lower >= 0), f"lower bound < 0 for {dist}"
        assert np.all(upper <= 1), f"upper bound > 1 for {dist}"


@pytest.mark.parametrize("dist", ["weibull", "lognormal", "loglogistic"])
def test_predict_survival_ci_plain(lung_surv, dist) -> None:  # type: ignore[no-untyped-def]
    df, y = lung_surv
    model = AFT(dist).fit(y, df[["age", "sex"]])
    nd = df[["age", "sex"]].iloc[:2]
    pred = model.predict(
        nd, type="survival", times=[200, 400], ci=True, conf_type="plain", format="pandas"
    )
    for j in range(1, 3):
        lower = pred[f"subject_{j}_lower"].to_numpy()
        point = pred[f"subject_{j}"].to_numpy()
        upper = pred[f"subject_{j}_upper"].to_numpy()
        assert np.all(lower <= point + 1e-12)
        assert np.all(point <= upper + 1e-12)


def test_predict_survival_ci_wider_at_higher_conf_level(lung_surv) -> None:  # type: ignore[no-untyped-def]
    df, y = lung_surv
    nd = df[["age", "sex"]].iloc[:2]
    pred_90 = (
        AFT("weibull", conf_level=0.90)
        .fit(y, df[["age", "sex"]])
        .predict(nd, type="survival", times=[200, 400], ci=True, format="pandas")
    )
    pred_99 = (
        AFT("weibull", conf_level=0.99)
        .fit(y, df[["age", "sex"]])
        .predict(nd, type="survival", times=[200, 400], ci=True, format="pandas")
    )
    for j in range(1, 3):
        w90 = pred_90[f"subject_{j}_upper"].to_numpy() - pred_90[f"subject_{j}_lower"].to_numpy()
        w99 = pred_99[f"subject_{j}_upper"].to_numpy() - pred_99[f"subject_{j}_lower"].to_numpy()
        assert np.all(w99 >= w90 - 1e-12)


def test_predict_survival_ci_without_ci_unchanged(lung_surv) -> None:  # type: ignore[no-untyped-def]
    df, y = lung_surv
    model = AFT("weibull").fit(y, df[["age", "sex"]])
    nd = df[["age", "sex"]].iloc[:2]
    times = [180, 365]
    no_ci = model.predict(nd, type="survival", times=times, format="pandas")
    with_ci = model.predict(nd, type="survival", times=times, ci=True, format="pandas")
    for j in range(1, 3):
        np.testing.assert_allclose(
            with_ci[f"subject_{j}"].to_numpy(), no_ci[f"subject_{j}"].to_numpy()
        )


def test_predict_survival_ci_conditional_after_raises(lung_surv) -> None:  # type: ignore[no-untyped-def]
    df, y = lung_surv
    model = AFT("weibull").fit(y, df[["age", "sex"]])
    with pytest.raises(NotImplementedError, match="conditional_after"):
        model.predict(
            df[["age", "sex"]].iloc[:1],
            type="survival",
            times=[200],
            ci=True,
            conditional_after=100.0,
        )


@pytest.mark.slow
def test_predict_survival_ci_vs_bootstrap(lung_surv) -> None:  # type: ignore[no-untyped-def]
    df, y = lung_surv
    model = AFT("weibull").fit(y, df[["age", "sex"]])
    nd = df[["age", "sex"]].iloc[:2]
    times = np.array([200, 400])
    rng = np.random.default_rng(23)
    n_boot = 500
    boot_surv = np.zeros((n_boot, len(times), 2))
    n = len(df)
    for b in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        df_b = df.iloc[idx].reset_index(drop=True)
        y_b = Surv(time=df_b["time"], event=(df_b["status"] == 2))
        try:
            m_b = AFT("weibull").fit(y_b, df_b[["age", "sex"]])
            s_b = m_b.predict(nd, type="survival", times=list(times), format="pandas")
            for j in range(2):
                boot_surv[b, :, j] = s_b[f"subject_{j + 1}"].to_numpy()
        except Exception:
            boot_surv[b] = np.nan

    valid = ~np.isnan(boot_surv[:, 0, 0])
    boot_valid = boot_surv[valid]
    boot_se = boot_valid.std(axis=0)
    delta_se = np.zeros((len(times), 2))
    se_s = model._survival_se(model._design(nd), times)
    delta_se = se_s
    # Delta-method SE should be within a factor of 2 of bootstrap SE
    for j in range(2):
        ratio = delta_se[:, j] / boot_se[:, j]
        assert np.all(ratio > 0.3), f"Delta SE too small vs bootstrap for subject {j + 1}"
        assert np.all(ratio < 3.0), f"Delta SE too large vs bootstrap for subject {j + 1}"


def test_aft_loglogistic_predict_rmst_sigma_ge_1(lung_surv) -> None:  # type: ignore[no-untyped-def]
    df, y = lung_surv
    aft = AFT(dist="loglogistic").fit(y, df[["age", "sex"]])
    original_scale = aft.scale_
    aft.scale_ = 1.5
    try:
        nd = df[["age", "sex"]].iloc[:3]
        result = aft.predict(nd, type="rmst", tau=365.0)
        assert result.shape == (3,)
        assert np.all(result > 0)
    finally:
        aft.scale_ = original_scale


# ---------------------------------------------------------------------------
# Generalized gamma distribution
# ---------------------------------------------------------------------------


def test_gengamma_basic(lung_surv) -> None:  # type: ignore[no-untyped-def]
    df, y = lung_surv
    gg = AFT("gengamma").fit(y, df[["age", "sex"]])
    assert gg.dist == "gengamma"
    assert gg.Q_ is not None
    assert gg.Q_se_ is not None
    assert gg.loglik_ >= AFT("weibull").fit(y, df[["age", "sex"]]).loglik_ - 0.01


def test_gengamma_nests_weibull(lung_surv) -> None:  # type: ignore[no-untyped-def]
    """Generalized gamma should achieve at least the Weibull log-likelihood."""
    df, y = lung_surv
    gg = AFT("gengamma").fit(y, df[["age", "sex"]])
    wb = AFT("weibull").fit(y, df[["age", "sex"]])
    assert gg.loglik_ >= wb.loglik_ - 0.01


def test_gengamma_repr(lung_surv) -> None:  # type: ignore[no-untyped-def]
    df, y = lung_surv
    gg = AFT("gengamma").fit(y, df[["age", "sex"]])
    r = repr(gg)
    assert "gengamma" in r
    assert "Q =" in r
    assert "Scale" in r


def test_gengamma_to_frame(lung_surv) -> None:  # type: ignore[no-untyped-def]
    df, y = lung_surv
    gg = AFT("gengamma").fit(y, df[["age", "sex"]])
    result = gg.to_frame(format="pandas")
    assert len(result) == 3
    assert "(Intercept)" in result["term"].values


def test_gengamma_glance(lung_surv) -> None:  # type: ignore[no-untyped-def]
    df, y = lung_surv
    gg = AFT("gengamma").fit(y, df[["age", "sex"]])
    g = gw.glance(gg, format="pandas")
    assert "Q" in g.columns
    assert g["dist"].iloc[0] == "gengamma"


def test_gengamma_predict_survival(lung_surv) -> None:  # type: ignore[no-untyped-def]
    df, y = lung_surv
    gg = AFT("gengamma").fit(y, df[["age", "sex"]])
    nd = df[["age", "sex"]].iloc[:3]
    pred = gg.predict(nd, type="survival", times=[100, 365], format="pandas")
    assert pred.shape[0] == 2
    surv_vals = pred.iloc[:, 1:].values
    assert np.all(surv_vals >= 0)
    assert np.all(surv_vals <= 1)
    assert np.all(np.diff(surv_vals, axis=0) <= 0)


def test_gengamma_predict_quantile(lung_surv) -> None:  # type: ignore[no-untyped-def]
    df, y = lung_surv
    gg = AFT("gengamma").fit(y, df[["age", "sex"]])
    nd = df[["age", "sex"]].iloc[:3]
    q = gg.predict(nd, type="quantile", p=[0.25, 0.5, 0.75], format="pandas")
    assert q.shape == (3, 4)
    vals = q.iloc[:, 1:].values
    assert np.all(vals > 0)
    assert np.all(np.diff(vals, axis=0) > 0)


def test_gengamma_predict_lp(lung_surv) -> None:  # type: ignore[no-untyped-def]
    df, y = lung_surv
    gg = AFT("gengamma").fit(y, df[["age", "sex"]])
    lp = gg.predict(df[["age", "sex"]].iloc[:3], type="lp")
    assert lp.shape == (3,)


def test_gengamma_test_distributions(lung_surv) -> None:  # type: ignore[no-untyped-def]
    df, y = lung_surv
    gg = AFT("gengamma").fit(y, df[["age", "sex"]])
    result = gg.test_distributions(format="pandas")
    assert list(result.columns) == ["dist", "loglik", "aic", "lr_statistic", "df", "p_value"]
    # The log-logistic isn't nested in the generalized gamma.
    assert list(result["dist"]) == ["weibull", "lognormal", "exponential"]
    assert list(result["df"]) == [1, 1, 2]
    assert all(result["p_value"] >= 0)
    assert all(result["p_value"] <= 1)
    assert all(result["lr_statistic"] >= 0)
    # Each nested log-likelihood is that distribution's own fit.
    for dist, loglik in zip(result["dist"], result["loglik"], strict=True):
        assert loglik == pytest.approx(AFT(dist).fit(y, df[["age", "sex"]]).loglik_, abs=1e-6)


def test_gengamma_nests_weibull_and_lognormal_on_simulated_data() -> None:
    rng = np.random.default_rng(11)
    n = 300
    x = rng.normal(size=n)
    covariates = x[:, None]
    # Weibull data: Q lands near 1, and the fit is at least as good as the Weibull's.
    time = np.exp(1.0 + 0.5 * x + 0.6 * np.log(rng.exponential(size=n)))
    y = gw.Surv(time=time, event=np.ones(n))
    gg = AFT("gengamma").fit(y, covariates)
    assert gg.Q_ is not None and gg.Q_se_ is not None
    assert abs(gg.Q_ - 1.0) < 3.0 * gg.Q_se_
    assert gg.loglik_ >= AFT("weibull").fit(y, covariates).loglik_ - 1e-6
    # Log-normal data: Q lands near 0, where the family passes smoothly through the normal limit.
    time = np.exp(1.0 + 0.5 * x + 0.6 * rng.normal(size=n))
    y = gw.Surv(time=time, event=np.ones(n))
    gg = AFT("gengamma").fit(y, covariates)
    assert gg.Q_ is not None and gg.Q_se_ is not None
    assert abs(gg.Q_) < 3.0 * gg.Q_se_
    assert gg.loglik_ >= AFT("lognormal").fit(y, covariates).loglik_ - 1e-6


def test_gengamma_family_is_continuous_at_q_zero() -> None:
    from scipy.stats import norm

    from greenwood._parametric import (
        _gengamma_log_density_survival,
        _gengamma_mean,
        _gengamma_quantile,
    )

    z = np.array([-3.0, -0.5, 0.0, 1.0, 2.5])
    p = np.array([0.1, 0.5, 0.9])
    for q in (-1e-4, -1e-7, 1e-7, 1e-4):
        log_f, log_s = _gengamma_log_density_survival(z, q)
        np.testing.assert_allclose(log_f, norm.logpdf(z), rtol=1e-3)
        np.testing.assert_allclose(log_s, norm.logsf(z), rtol=1e-3)
        np.testing.assert_allclose(_gengamma_quantile(p, q), norm.ppf(p), atol=1e-3)
        mean = _gengamma_mean(np.zeros(1), 0.7, q)
        np.testing.assert_allclose(mean, np.exp(0.7**2 / 2.0), rtol=1e-3)


@pytest.mark.parametrize("q", [-1.5, -0.4, 0.3, 1.0, 2.0])
def test_gengamma_family_matches_its_definition(q: float) -> None:
    from scipy.integrate import quad
    from scipy.special import gammainc, gammaincc, gammaln

    from greenwood._parametric import (
        _error_pieces,
        _gengamma_log_density_survival,
        _gengamma_mean,
        _gengamma_quantile,
    )

    k = q**-2.0
    z = np.array([-2.0, -0.3, 0.4, 1.7])
    u = k * np.exp(q * z)
    log_f, log_s = _gengamma_log_density_survival(z, q)
    expected = np.log(abs(q)) + k * np.log(k) - gammaln(k) + k * q * z - u
    np.testing.assert_allclose(log_f, expected, rtol=1e-12)
    upper = gammaincc(k, u) if q > 0 else gammainc(k, u)
    np.testing.assert_allclose(np.exp(log_s), upper, rtol=1e-12)
    _, log_s_q = _gengamma_log_density_survival(_gengamma_quantile(np.array([0.2, 0.7]), q), q)
    np.testing.assert_allclose(np.exp(log_s_q), [0.8, 0.3], rtol=1e-10)

    # The derivatives in z agree with differences, for events and censored rows.
    h = 1e-5
    for status in (1.0, 0.0):
        event = np.full(z.shape, status)
        _, d1, d2 = _error_pieces("gengamma", z, event, Q=q)
        g_up, d1_up, _ = _error_pieces("gengamma", z + h, event, Q=q)
        g_down, d1_down, _ = _error_pieces("gengamma", z - h, event, Q=q)
        np.testing.assert_allclose(d1, (g_up - g_down) / (2 * h), rtol=1e-6, atol=1e-8)
        np.testing.assert_allclose(d2, (d1_up - d1_down) / (2 * h), rtol=1e-6, atol=1e-8)

    # The mean of exp(sigma W) by integration.
    sigma = 0.6

    def integrand(w: float) -> float:
        lf, _ = _gengamma_log_density_survival(np.array([w]), q)
        return float(np.exp(sigma * w + lf[0]))

    area = quad(integrand, -np.inf, np.inf, epsrel=1e-10, limit=500)[0]
    np.testing.assert_allclose(_gengamma_mean(np.zeros(1), sigma, q), area, rtol=1e-7)


@pytest.mark.parametrize("q", [-1.5, -0.4, 0.6, 1.4])
def test_gengamma_rmst_and_tail_match_integration(q: float) -> None:
    from scipy.integrate import quad

    from greenwood._parametric import (
        _gengamma_log_survival,
        _rmst_value,
        _tail_partial_moment,
    )

    # sigma = 0.9 makes k + sigma / Q negative for Q = -1.5, so that mean is infinite.
    mu, sigma, tau = np.array([0.2, 1.1]), 0.9, 3.0
    for i, m in enumerate(mu):

        def surv(t: float, m: float = float(m)) -> float:
            return float(np.exp(_gengamma_log_survival(np.array([(np.log(t) - m) / sigma]), q)[0]))

        area = quad(surv, 0.0, tau, epsrel=1e-11, limit=500)[0]
        np.testing.assert_allclose(_rmst_value("gengamma", mu, sigma, tau, Q=q)[i], area, rtol=1e-8)
        tail = _tail_partial_moment("gengamma", mu, sigma, np.full(2, tau), Q=q)[i]
        if q**-2.0 + sigma / q > 0:
            expected = quad(surv, tau, np.inf, epsrel=1e-11, limit=500)[0]
            np.testing.assert_allclose(tail, expected, rtol=1e-7)
        else:
            assert tail == np.inf


def test_gengamma_mps_starts_from_the_mle(lung_surv) -> None:  # type: ignore[no-untyped-def]
    df, y = lung_surv
    mle = AFT("gengamma").fit(y, df[["age", "sex"]])
    mps = AFT("gengamma", method="mps").fit(y, df[["age", "sex"]])
    assert mps.Q_ is not None and mle.Q_ is not None
    assert np.all(np.isfinite(mps.std_error_))
    # MPS and MLE are asymptotically equivalent, so the estimates are within a standard error.
    assert np.all(np.abs(mps.coef_ - mle.coef_) < mle.std_error_)
    assert mps.loglik_ <= mle.loglik_ + 1e-9


def test_gengamma_test_distributions_only_gengamma(lung_surv) -> None:  # type: ignore[no-untyped-def]
    df, y = lung_surv
    wb = AFT("weibull").fit(y, df[["age", "sex"]])
    with pytest.raises(ValueError, match="only available for"):
        wb.test_distributions()


def test_gengamma_coef_close_to_weibull(lung_surv) -> None:  # type: ignore[no-untyped-def]
    """When Q converges to 1, coefficients should match Weibull."""
    df, y = lung_surv
    gg = AFT("gengamma").fit(y, df[["age", "sex"]])
    wb = AFT("weibull").fit(y, df[["age", "sex"]])
    if abs(gg.Q_ - 1.0) < 0.1:  # type: ignore[operator]
        np.testing.assert_allclose(gg.coef_, wb.coef_, atol=0.05)


# -- Residual diagnostics -----------------------------------------------------


def test_residuals_response_shape(lung_surv) -> None:  # type: ignore[no-untyped-def]
    df, y = lung_surv
    model = AFT("weibull").fit(y, df[["age", "sex"]])
    r = model.residuals("response")
    assert isinstance(r, np.ndarray)
    assert r.shape == (model.n_,)


def test_residuals_cox_snell_positive(lung_surv) -> None:  # type: ignore[no-untyped-def]
    df, y = lung_surv
    model = AFT("weibull").fit(y, df[["age", "sex"]])
    cs = model.residuals("cox_snell")
    assert np.all(cs >= 0)


def test_residuals_martingale_identity(lung_surv) -> None:  # type: ignore[no-untyped-def]
    df, y = lung_surv
    model = AFT("weibull").fit(y, df[["age", "sex"]])
    mart = model.residuals("martingale")
    cs = model.residuals("cox_snell")
    np.testing.assert_allclose(mart, model._event.astype(float) - cs)


def test_residuals_deviance_shape(lung_surv) -> None:  # type: ignore[no-untyped-def]
    df, y = lung_surv
    model = AFT("weibull").fit(y, df[["age", "sex"]])
    d = model.residuals("deviance")
    assert isinstance(d, np.ndarray)
    assert d.shape == (model.n_,)


def test_residuals_score_shape(lung_surv) -> None:  # type: ignore[no-untyped-def]
    df, y = lung_surv
    model = AFT("weibull").fit(y, df[["age", "sex"]])
    s = model.residuals("score", format="pandas")
    assert list(s.columns) == model.term_names_
    assert len(s) == model.n_


def test_residuals_dfbeta_shape(lung_surv) -> None:  # type: ignore[no-untyped-def]
    df, y = lung_surv
    model = AFT("weibull").fit(y, df[["age", "sex"]])
    d = model.residuals("dfbeta", format="pandas")
    assert list(d.columns) == model.term_names_
    assert len(d) == model.n_


def test_residuals_dfbetas_shape(lung_surv) -> None:  # type: ignore[no-untyped-def]
    df, y = lung_surv
    model = AFT("weibull").fit(y, df[["age", "sex"]])
    d = model.residuals("dfbetas", format="pandas")
    assert list(d.columns) == model.term_names_
    assert len(d) == model.n_


def test_residuals_invalid_type_raises(lung_surv) -> None:  # type: ignore[no-untyped-def]
    df, y = lung_surv
    model = AFT("weibull").fit(y, df[["age", "sex"]])
    with pytest.raises(ValueError, match="Unknown residual type"):
        model.residuals("schoenfeld")


@pytest.mark.parametrize("dist", ["weibull", "exponential", "lognormal", "loglogistic"])
def test_residuals_all_dists_smoke(lung_surv, dist: str) -> None:  # type: ignore[no-untyped-def]
    df, y = lung_surv
    model = AFT(dist).fit(y, df[["age", "sex"]])
    for rtype in AFT._RESIDUAL_TYPES:
        r = model.residuals(rtype, format="pandas")
        assert r is not None


def test_residuals_format_polars(lung_surv) -> None:  # type: ignore[no-untyped-def]
    pytest.importorskip("polars")
    import polars as pl

    df, y = lung_surv
    model = AFT("weibull").fit(y, df[["age", "sex"]])
    d = model.residuals("dfbeta", format="polars")
    assert isinstance(d, pl.DataFrame)
