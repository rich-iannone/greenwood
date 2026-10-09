"""Edge-case, error-path, and alternate-backend coverage across the package.

These exercise the branches that the feature tests do not: validation errors, unfitted reprs,
grouped/ungrouped returns, and default-argument paths.
"""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest

import greenwood as gw
from greenwood import (
    AFT,
    CoxNet,
    CoxPH,
    KaplanMeier,
    NelsonAalen,
    RoystonParmar,
    Surv,
    cross_validate,
    event_table,
    logrank_test,
    pairwise_logrank_test,
)


@pytest.fixture(scope="module")
def lung():
    return gw.load_dataset("lung", backend="pandas")


@pytest.fixture(scope="module")
def y(lung):
    return Surv(time=lung["time"], event=lung["status"] == 2)


# -- data backend resolution -----------------------------------------------------------


def test_resolve_backend_fallback_and_error(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    from greenwood.data import _resolve_backend

    real = importlib.util.find_spec
    monkeypatch.setattr(importlib.util, "find_spec", lambda n: None if n == "polars" else real(n))
    assert _resolve_backend(None) == "pandas"  # prefers Polars, falls back to pandas
    monkeypatch.setattr(
        importlib.util, "find_spec", lambda n: None if n in ("polars", "pandas") else real(n)
    )
    with pytest.raises(ImportError, match="polars or pandas"):
        _resolve_backend(None)


# -- event table -----------------------------------------------------------------------


def test_event_table_weights_grouped_and_backends(y, lung) -> None:
    et = event_table(y, weights=np.ones(y.n))
    assert et.to_frame(format="polars").shape[1] == 5
    assert et.to_frame(format="pyarrow").num_columns == 5
    grouped = event_table(y, group=lung["sex"]).to_frame(format="pandas")
    assert set(grouped["strata"]) == {"sex=1", "sex=2"}
    with pytest.raises(NotImplementedError, match="event_table"):
        event_table(Surv(time=[1, 2], time2=[2, 3], type="interval2"))


# -- Kaplan-Meier / Nelson-Aalen -------------------------------------------------------


def test_km_strata_predict_and_backends(y, lung) -> None:
    km = KaplanMeier().fit(y)
    assert set(km.strata_) == {"all"}  # ungrouped
    assert km.to_frame(format="polars").shape[0] > 0
    assert km.to_frame(format="pyarrow").num_rows > 0
    with pytest.raises(ValueError, match="survival' or 'cumhaz"):
        km.predict([100], what="nonsense")
    grouped = KaplanMeier().fit(y, by=lung["sex"])
    assert set(grouped.strata_) == {"sex=1", "sex=2"}
    assert "strata" in grouped.to_frame(format="pandas").columns
    pred = grouped.predict([100, 200], format="pandas")
    assert list(pred["strata"]) == ["sex=1", "sex=1", "sex=2", "sex=2"]  # one row per curve x time


def test_nelson_aalen_variants(y, lung) -> None:
    with pytest.raises(ValueError, match="conf_type"):
        NelsonAalen(conf_type="bad")
    with pytest.raises(ValueError, match="conf_level"):
        NelsonAalen(conf_level=2.0)
    na_plain = NelsonAalen(conf_type="plain").fit(y).to_frame(format="pandas")
    assert {"conf_low", "conf_high"} <= set(na_plain.columns)
    na = NelsonAalen().fit(y)
    assert set(na.strata_) == {"all"}
    assert na.to_frame(format="polars").shape[0] > 0
    assert na.to_frame(format="pyarrow").num_rows > 0
    grouped = NelsonAalen().fit(y, by=lung["sex"])
    assert set(grouped.strata_) == {"sex=1", "sex=2"}
    assert "strata" in grouped.to_frame(format="pandas").columns


# -- log-rank family -------------------------------------------------------------------


def test_logrank_error_paths() -> None:
    y = Surv(time=[1, 2, 3, 4], event=[1, 1, 1, 1])
    with pytest.raises(NotImplementedError, match="logrank_test"):
        logrank_test(Surv(time=[1, 2], time2=[2, 3], type="interval2"), ["a", "b"])
    with pytest.raises(ValueError, match="`strata`"):
        logrank_test(y, ["a", "a", "b", "b"], strata=["s"])
    with pytest.raises(ValueError, match="No events"):
        logrank_test(Surv(time=[1, 2, 3, 4], event=[0, 0, 0, 0]), ["a", "a", "b", "b"])


def test_stratified_logrank_with_empty_stratum() -> None:
    # A stratum with no events contributes zeros (exercises the empty-times branch).
    y = Surv(time=[1, 2, 3, 4], event=[1, 1, 0, 0])
    result = logrank_test(y, ["a", "b", "a", "b"], strata=["s1", "s1", "s2", "s2"])
    assert result.df == 1


def test_pairwise_error_paths() -> None:
    y = Surv(time=[1, 2, 3, 4], event=[1, 1, 1, 1])
    with pytest.raises(NotImplementedError, match="pairwise"):
        pairwise_logrank_test(Surv(time=[1, 2], time2=[2, 3], type="interval2"), ["a", "b"])
    with pytest.raises(ValueError, match="same length"):
        pairwise_logrank_test(y, ["a", "b"])
    with pytest.raises(ValueError, match="`strata`"):
        pairwise_logrank_test(y, ["a", "a", "b", "b"], strata=["s"])
    with pytest.raises(ValueError, match="at least two groups"):
        pairwise_logrank_test(y, ["a", "a", "a", "a"])


# -- power calculators -----------------------------------------------------------------


def test_power_validation() -> None:
    from greenwood import logrank_n_events, logrank_sample_size

    with pytest.raises(ValueError, match="allocation"):
        logrank_n_events(0.5, allocation=1.5)
    with pytest.raises(ValueError, match="power"):
        logrank_sample_size(0.5, 0.4, power=1.5)


# -- AFT / Royston-Parmar / CoxNet predict and repr paths ------------------------------


def test_aft_predict_paths(y, lung) -> None:
    aft = AFT("weibull").fit(y, lung[["age", "sex"]])
    default_grid = aft.predict(lung[["age", "sex"]].iloc[:1], type="survival")  # times=None
    assert "time" in default_grid.columns and len(default_grid) > 1
    with pytest.raises(ValueError, match="conditional_after must be"):
        aft.predict(
            lung[["age", "sex"]].iloc[:2], type="survival", times=[100], conditional_after=[1.0]
        )
    with pytest.raises(ValueError, match="Unknown predict type"):
        aft.predict(lung[["age", "sex"]].iloc[:1], type="nonsense")


def test_royston_parmar_paths(y, lung) -> None:
    assert "<unfitted>" in repr(RoystonParmar())
    with pytest.raises(NotImplementedError, match="right-censored"):
        RoystonParmar().fit(Surv(time=[0, 1], time2=[5, 6], event=[1, 1]))
    with pytest.raises(ValueError, match="same number of rows"):
        RoystonParmar().fit(y, lung[["age"]].iloc[:-1])
    with pytest.raises(ValueError, match="No events remain"):
        RoystonParmar().fit(Surv(time=[1, 2, 3, 4], event=[0, 0, 0, 0]))
    rp = RoystonParmar(df=2).fit(y, lung[["age", "sex"]])
    nd = lung[["age", "sex"]].iloc[:1]
    assert (rp.predict(nd, type="cumhaz", times=[180, 365], format="pandas").iloc[:, 1] >= 0).all()
    assert list(rp.to_frame(format="pandas").columns)[0] == "term"
    with pytest.raises(ValueError, match="Unknown predict type"):
        rp.predict(nd, type="nonsense", times=[180])


def test_coxnet_paths(y, lung) -> None:
    assert "<unfitted>" in repr(CoxNet())
    with pytest.raises(NotImplementedError, match="right-censored"):
        CoxNet().fit(Surv(time=[1, 2], time2=[2, 3], type="interval2"), np.zeros((2, 1)))
    with pytest.raises(ValueError, match="same number of rows"):
        CoxNet().fit(y, lung[["age"]].iloc[:-1])
    with pytest.raises(ValueError, match="No events remain"):
        CoxNet().fit(Surv(time=[1, 2, 3, 4], event=[0, 0, 0, 0]), np.zeros((4, 1)))
    cn = CoxNet(penalizer=0.05).fit(y, lung[["age", "sex"]])
    assert np.all(cn.predict(lung[["age", "sex"]], type="risk") > 0)
    with pytest.raises(ValueError, match="Unknown predict type"):
        cn.predict(lung[["age", "sex"]], type="nonsense")


# -- cross-validation edge paths -------------------------------------------------------


def test_cross_validate_counting_and_errors(lung) -> None:
    # Counting-process response exercises the counting branch of the fold subsetter.
    yc = Surv(time=[0, 0, 0, 0, 0, 0], time2=[2, 4, 6, 3, 5, 7], event=[1, 1, 1, 1, 0, 1])
    x = np.arange(6.0).reshape(-1, 1)
    out = cross_validate(CoxPH(), yc, x, k=2, seed=23)
    assert len(out["scores"]) == 2
    y = Surv(time=lung["time"], event=lung["status"] == 2)
    with pytest.raises(ValueError, match="same number of rows"):
        cross_validate(CoxPH(), y, lung[["age", "sex"]].iloc[:-1])


def test_cross_validate_drops_missing_rows(lung) -> None:
    y = Surv(time=lung["time"], event=lung["status"] == 2)
    out = cross_validate(CoxPH(), y, lung[["age", "ph.ecog"]], k=3, seed=23)  # ph.ecog has a NaN
    assert len(out["scores"]) == 3


def test_subset_surv_rejects_interval() -> None:
    from greenwood._resample import _risk_score, _subset_surv

    with pytest.raises(NotImplementedError, match="right-censored"):
        _subset_surv(Surv(time=[1, 2], time2=[2, 3], type="interval2"), np.array([0, 1]))
    with pytest.raises(TypeError, match="CoxPH, CoxNet, AFT, or RoystonParmar"):
        _risk_score(KaplanMeier(), np.zeros((2, 1)))


# -- tidy registry augment -------------------------------------------------------------


class _DummyModel:
    """A stand-in model for exercising the augment registry."""


def test_register_and_dispatch_augment() -> None:
    from greenwood.summaries import augment, register_augment

    key = f"{_DummyModel.__module__}.{_DummyModel.__qualname__}"
    register_augment(key, lambda m, data, **k: {"ok": True})
    assert augment(_DummyModel()) == {"ok": True}


# -- R left type, interval interop, and calibration-before-first-event -------------------


def test_surv_left_and_interval_dataframe() -> None:
    # R's left type: status 1 is an exact event, status 0 is left-censored.
    left = Surv(time=[3, 5, 7], event=[1, 0, 1], type="left")
    assert left.type.value == "left" and len(left) == 3
    np.testing.assert_array_equal(left.status, [1.0, 0.0, 1.0])
    iv = Surv(time=[1, 2, 3], time2=[2, 4, 6], type="interval2")
    assert list(iv.to_frame(format="pandas").columns) == ["time1", "time2", "status"]


def test_calibration_before_first_event(y, lung) -> None:
    cox = CoxPH().fit(y, lung[["age", "sex"]])
    pred = (
        cox.predict(lung[["age", "sex"]], type="survival", times=[365.0], format="pandas")
        .iloc[0, 1:]
        .to_numpy()
    )
    cal = gw.calibration(y, pred, 1.0, n_bins=3)  # earlier than any event -> observed is 1
    assert (cal["observed"] == 1.0).all()


# -- Cox error and prediction branches -------------------------------------------------


def test_cox_fit_error_paths(y, lung) -> None:
    with pytest.raises(NotImplementedError, match="right-censored"):
        CoxPH().fit(Surv(time=[1, 2], time2=[2, 3], type="interval2"), np.zeros((2, 1)))
    with pytest.raises(ValueError, match="No events remain"):
        CoxPH().fit(Surv(time=[1, 2, 3, 4], event=[0, 0, 0, 0]), np.zeros((4, 1)))
    with pytest.raises(ValueError, match="produced no covariates"):
        CoxPH().fit(y, "1", data=lung)
    with pytest.raises(ValueError, match="2-D"):
        CoxPH().fit(y, np.array([1.0, 2.0, 3.0]))
    with pytest.raises(ValueError, match="No covariates found"):
        CoxPH().fit(y, lung[[]])


def test_zph_repr(y, lung) -> None:
    assert "ZPHResult" in repr(CoxPH().fit(y, lung[["age", "sex"]]).cox_zph())


def test_cox_predict_ci_before_event_and_conditional_mismatch(y, lung) -> None:
    cox = CoxPH().fit(y, lung[["age", "sex"]])
    nd = lung[["age", "sex"]].iloc[:2]
    before = cox.predict(nd, type="survival", times=[1.0], ci=True)  # before first event
    assert (before["subject_1"] == 1.0).all()
    with pytest.raises(ValueError, match="conditional_after must be"):
        cox.predict(nd, type="survival", times=[100], conditional_after=[1.0, 2.0, 3.0])


# -- Competing-risks / multi-state backends and validation -----------------------------


def test_competing_backends_and_validation() -> None:
    import pandas as pd

    from greenwood import AalenJohansen, FineGray, MultiState

    status = pd.Categorical.from_codes(
        np.array([1, 2, 1, 2, 0, 1, 2, 1]), categories=["censor", "pcm", "death"]
    )
    y_cr = Surv(time=[5, 6, 7, 8, 9, 10, 11, 12], event=status)
    aj = AalenJohansen().fit(y_cr)
    assert aj.to_frame(format="polars").shape[0] > 0
    assert aj.to_frame(format="pyarrow").num_rows > 0
    with pytest.raises(ValueError, match="conf_level"):
        FineGray("pcm", conf_level=2.0)

    ms = MultiState().fit(
        start=[0, 0, 2, 0, 0, 3],
        stop=[2, 5, 5, 4, 3, 6],
        state=["mgus", "mgus", "pcm", "mgus", "mgus", "pcm"],
        event=["pcm", "death", "death", None, "pcm", "death"],
        states=("mgus", "pcm", "death"),
    )
    assert ms.to_frame(format="polars").shape[0] > 0
    assert ms.to_frame(format="pyarrow").num_rows > 0


# -- Grouped tidy / repr paths ---------------------------------------------------------


def test_km_tidy_grouped_has_strata(y, lung) -> None:
    km = KaplanMeier().fit(y, by=lung["sex"])
    assert "strata" in gw.tidy(km).columns
    assert "strata" in gw.glance(km).columns  # grouped glance carries the stratum label


def test_grouped_nelson_aalen_repr(y, lung) -> None:
    text = repr(NelsonAalen().fit(y, by=lung["sex"]))
    assert "Nelson-Aalen" in text and "\n" in text


def test_concordance_last_event_has_no_comparable() -> None:
    # The largest event time has no later subject, exercising the skip branch.
    cox = CoxPH().fit(Surv(time=[1, 2, 3], event=[1, 1, 1]), np.array([[1.0], [2.0], [3.0]]))
    c = cox.concordance()
    assert 0.0 <= c <= 1.0
