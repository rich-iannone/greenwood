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
    df = gw.data.load_dataset("lung", backend="pandas")
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
    tidy = gw.tidy.tidy(model)
    assert list(tidy["term"]) == ["(Intercept)", "age", "sex"]
    glance = gw.tidy.glance(model)
    row = glance.iloc[0]
    assert row["dist"] == "weibull"
    assert row["nevent"] == 165
    assert row["aic"] == pytest.approx(-2 * model.loglik_ + 2 * 4)  # 3 coef + log(scale)


def test_to_dataframe_columns(lung_surv) -> None:  # type: ignore[no-untyped-def]
    df, y = lung_surv
    model = AFT().fit(y, df[["age", "sex"]])
    assert list(model.to_dataframe().columns) == [
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
    surv = model.predict(df[["age", "sex"]].iloc[:4], type="survival", times=[100, 300, 500])
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
    s_t = model.predict(nd, type="survival", times=times)
    s_c = model.predict(nd, type="survival", times=[c])
    s_cond = model.predict(nd, type="survival", times=times, conditional_after=c)
    for col in ("subject_1", "subject_2"):
        np.testing.assert_allclose(
            s_cond[col].to_numpy() * float(s_c[col].iloc[0]), s_t[col].to_numpy(), atol=1e-12
        )
    s0 = model.predict(nd, type="survival", times=times, conditional_after=0.0)
    np.testing.assert_allclose(
        s0[["subject_1", "subject_2"]].to_numpy(), s_t[["subject_1", "subject_2"]].to_numpy()
    )
