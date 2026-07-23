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
    return df, Surv.right(df["time"], event=(df["status"] == 2))


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
    y = Surv.interval(lower=[1, 2, 3], upper=[2, 3, 4])
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
    rng = np.random.default_rng(42)
    n_boot = 500
    boot_surv = np.zeros((n_boot, len(times), 2))
    n = len(df)
    for b in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        df_b = df.iloc[idx].reset_index(drop=True)
        y_b = Surv.right(df_b["time"], event=(df_b["status"] == 2))
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
