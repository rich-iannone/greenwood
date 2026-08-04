"""Tests for predict_quantile and predict_median across CoxPH, AFT, and RoystonParmar."""

from __future__ import annotations

import numpy as np
import pytest

import greenwood as gw
from greenwood import Surv

from ._r_parity import assert_allclose_to_r, load_fixture

pytestmark = pytest.mark.rparity


@pytest.fixture()
def lung_data() -> tuple[Surv, np.ndarray]:
    df = gw.load_dataset("lung", backend="pandas")
    y = Surv.right(df["time"], event=(df["status"] == 2))
    x = df[["age", "sex"]].values
    return y, x


@pytest.fixture()
def median_fixture() -> dict:
    return load_fixture("predict_median")


@pytest.fixture()
def quantile_fixture() -> dict:
    return load_fixture("predict_quantile")


@pytest.fixture()
def newdata(median_fixture: dict) -> np.ndarray:
    return np.column_stack([median_fixture["newdata"]["age"], median_fixture["newdata"]["sex"]])


# ── CoxPH predict_quantile ──


def test_cox_predict_quantile(
    lung_data: tuple[Surv, np.ndarray],
    quantile_fixture: dict,
    newdata: np.ndarray,
) -> None:
    y, x = lung_data
    cox = gw.CoxPH(ties="breslow").fit(y, x)
    result = cox.predict_quantile(newdata, p=[0.25, 0.5, 0.75], format="pandas")

    assert "p" in result.columns
    assert list(result["p"].values) == [0.25, 0.5, 0.75]

    r = quantile_fixture["cox_breslow"]["quantiles"]
    for k in range(3):
        for i in range(3):
            assert_allclose_to_r(
                [result[f"subject_{i + 1}"].values[k]],
                [r[k][i]],
                what=f"cox quantile p={[0.25, 0.5, 0.75][k]} subject {i + 1}",
                rtol=1e-6,
                atol=1.0,
            )


def test_cox_predict_quantile_ci(lung_data: tuple[Surv, np.ndarray], newdata: np.ndarray) -> None:
    y, x = lung_data
    cox = gw.CoxPH(ties="breslow").fit(y, x)
    result = cox.predict_quantile(newdata, p=0.5, ci=True, format="pandas")

    for i in range(3):
        col = f"subject_{i + 1}"
        assert f"{col}_lower" in result.columns
        assert f"{col}_upper" in result.columns


# ── CoxPH predict_median (delegates to predict_quantile) ──


def test_cox_predict_median(
    lung_data: tuple[Surv, np.ndarray],
    median_fixture: dict,
    newdata: np.ndarray,
) -> None:
    y, x = lung_data
    cox = gw.CoxPH(ties="breslow").fit(y, x)
    result = cox.predict_median(newdata, format="pandas")

    r = median_fixture["cox_breslow"]
    for i in range(3):
        assert_allclose_to_r(
            result[f"subject_{i + 1}"].values,
            [r["median"][i]],
            what=f"cox median subject {i + 1}",
            rtol=1e-6,
            atol=1.0,
        )


def test_cox_predict_median_ci(
    lung_data: tuple[Surv, np.ndarray],
    median_fixture: dict,
    newdata: np.ndarray,
) -> None:
    y, x = lung_data
    cox = gw.CoxPH(ties="breslow").fit(y, x)
    result = cox.predict_median(newdata, ci=True, format="pandas")

    r = median_fixture["cox_breslow"]
    for i in range(3):
        col = f"subject_{i + 1}"
        assert_allclose_to_r(
            result[f"{col}_lower"].values,
            [r["lower"][i]],
            what=f"cox median lower subject {i + 1}",
            rtol=1e-6,
            atol=1.0,
        )
        if not np.isnan(r["upper"][i]):
            assert_allclose_to_r(
                result[f"{col}_upper"].values,
                [r["upper"][i]],
                what=f"cox median upper subject {i + 1}",
                rtol=1e-6,
                atol=1.0,
            )


def test_cox_predict_median_no_newdata(
    lung_data: tuple[Surv, np.ndarray],
) -> None:
    y, x = lung_data
    cox = gw.CoxPH(ties="breslow").fit(y, x)
    result = cox.predict_median(format="pandas")
    assert result.shape[0] == 1
    assert f"subject_{x.shape[0]}" in result.columns


# ── AFT predict_quantile ──


def test_aft_predict_quantile(
    lung_data: tuple[Surv, np.ndarray],
    quantile_fixture: dict,
    newdata: np.ndarray,
) -> None:
    y, x = lung_data
    aft = gw.AFT("weibull").fit(y, x)
    result = aft.predict_quantile(newdata, p=[0.25, 0.5, 0.75], format="pandas")

    assert "p" in result.columns
    r = quantile_fixture["aft_weibull"]["quantiles"]
    for k in range(3):
        for i in range(3):
            assert_allclose_to_r(
                [result[f"subject_{i + 1}"].values[k]],
                [r[k][i]],
                what=f"aft quantile p={[0.25, 0.5, 0.75][k]} subj {i + 1}",
                rtol=1e-4,
            )


def test_aft_predict_quantile_ci(
    lung_data: tuple[Surv, np.ndarray],
    quantile_fixture: dict,
    newdata: np.ndarray,
) -> None:
    y, x = lung_data
    aft = gw.AFT("weibull").fit(y, x)
    result = aft.predict_quantile(newdata, p=[0.25, 0.5, 0.75], ci=True, format="pandas")

    r_lo = quantile_fixture["aft_weibull"]["lower"]
    r_hi = quantile_fixture["aft_weibull"]["upper"]
    for k in range(3):
        for i in range(3):
            assert_allclose_to_r(
                [result[f"subject_{i + 1}_lower"].values[k]],
                [r_lo[k][i]],
                what=f"aft quantile lower p={k} subj {i + 1}",
                rtol=1e-4,
            )
            assert_allclose_to_r(
                [result[f"subject_{i + 1}_upper"].values[k]],
                [r_hi[k][i]],
                what=f"aft quantile upper p={k} subj {i + 1}",
                rtol=1e-4,
            )


# ── AFT predict_median (all distributions) ──


def test_aft_weibull_predict_median(
    lung_data: tuple[Surv, np.ndarray],
    median_fixture: dict,
    newdata: np.ndarray,
) -> None:
    y, x = lung_data
    aft = gw.AFT("weibull").fit(y, x)
    result = aft.predict_median(newdata, format="pandas")

    r = median_fixture["aft_weibull"]
    for i in range(3):
        assert_allclose_to_r(
            result[f"subject_{i + 1}"].values,
            [r["median"][i]],
            what=f"aft weibull median subject {i + 1}",
            rtol=1e-4,
        )


def test_aft_weibull_predict_median_ci(
    lung_data: tuple[Surv, np.ndarray],
    median_fixture: dict,
    newdata: np.ndarray,
) -> None:
    y, x = lung_data
    aft = gw.AFT("weibull").fit(y, x)
    result = aft.predict_median(newdata, ci=True, format="pandas")

    r = median_fixture["aft_weibull"]
    for i in range(3):
        assert_allclose_to_r(
            result[f"subject_{i + 1}_lower"].values,
            [r["lower"][i]],
            what=f"aft weibull lower subject {i + 1}",
            rtol=1e-4,
        )
        assert_allclose_to_r(
            result[f"subject_{i + 1}_upper"].values,
            [r["upper"][i]],
            what=f"aft weibull upper subject {i + 1}",
            rtol=1e-4,
        )


def test_aft_lognormal_predict_median(
    lung_data: tuple[Surv, np.ndarray],
    median_fixture: dict,
    newdata: np.ndarray,
) -> None:
    y, x = lung_data
    aft = gw.AFT("lognormal").fit(y, x)
    result = aft.predict_median(newdata, format="pandas")

    r = median_fixture["aft_lognormal"]
    for i in range(3):
        assert_allclose_to_r(
            result[f"subject_{i + 1}"].values,
            [r["median"][i]],
            what=f"aft lognormal median subject {i + 1}",
            rtol=1e-4,
        )


def test_aft_loglogistic_predict_median(
    lung_data: tuple[Surv, np.ndarray],
    median_fixture: dict,
    newdata: np.ndarray,
) -> None:
    y, x = lung_data
    aft = gw.AFT("loglogistic").fit(y, x)
    result = aft.predict_median(newdata, format="pandas")

    r = median_fixture["aft_loglogistic"]
    for i in range(3):
        assert_allclose_to_r(
            result[f"subject_{i + 1}"].values,
            [r["median"][i]],
            what=f"aft loglogistic median subject {i + 1}",
            rtol=1e-4,
        )


# ── RoystonParmar predict_quantile ──


def test_rp_predict_quantile(lung_data: tuple[Surv, np.ndarray], newdata: np.ndarray) -> None:
    y, x = lung_data
    rp = gw.RoystonParmar(df=3).fit(y, x)
    result = rp.predict_quantile(newdata, p=[0.25, 0.5, 0.75], format="pandas")

    assert "p" in result.columns
    assert result.shape[0] == 3
    for k in range(3):
        for i in range(3):
            val = result[f"subject_{i + 1}"].values[k]
            assert np.isfinite(val) and val > 0


def test_rp_predict_quantile_ci(lung_data: tuple[Surv, np.ndarray], newdata: np.ndarray) -> None:
    y, x = lung_data
    rp = gw.RoystonParmar(df=3).fit(y, x)
    result = rp.predict_quantile(newdata, p=[0.25, 0.5, 0.75], ci=True, format="pandas")
    for k in range(3):
        for i in range(3):
            col = f"subject_{i + 1}"
            lo = result[f"{col}_lower"].values[k]
            val = result[col].values[k]
            hi = result[f"{col}_upper"].values[k]
            assert lo < val < hi, (
                f"CI ordering violated for p={[0.25, 0.5, 0.75][k]} subject {i + 1}"
            )


# ── RoystonParmar predict_median ──


def test_rp_predict_median(lung_data: tuple[Surv, np.ndarray], newdata: np.ndarray) -> None:
    y, x = lung_data
    rp = gw.RoystonParmar(df=3).fit(y, x)
    result = rp.predict_median(newdata, format="pandas")
    assert result.shape[0] == 1
    for i in range(3):
        val = result[f"subject_{i + 1}"].values[0]
        assert np.isfinite(val) and val > 0


def test_rp_predict_median_consistency(
    lung_data: tuple[Surv, np.ndarray], newdata: np.ndarray
) -> None:
    """Median should match the point where the survival curve crosses 0.5."""
    y, x = lung_data
    rp = gw.RoystonParmar(df=3).fit(y, x)
    median_df = rp.predict_median(newdata, format="pandas")
    times = np.linspace(1, 1500, 5000)
    surv_df = rp.predict(newdata, type="survival", times=times, format="pandas")
    for i in range(3):
        med = median_df[f"subject_{i + 1}"].values[0]
        s_vals = surv_df[f"subject_{i + 1}"].values
        idx = np.searchsorted(-s_vals, -0.5)
        if idx < len(times):
            np.testing.assert_allclose(med, times[idx], atol=1.0)


# ── Edge cases ──


def test_predict_quantile_nan_when_curve_never_crosses() -> None:
    """If S(t) never drops to 1-p, quantile should be NaN."""
    rng = np.random.default_rng(42)
    n = 50
    times = rng.exponential(100, n)
    events = np.zeros(n, dtype=int)
    events[0] = 1
    y = Surv.right(times, events)
    x = rng.standard_normal((n, 2))
    cox = gw.CoxPH().fit(y, x)
    result = cox.predict_quantile(p=0.5, format="pandas")
    n_nan = sum(np.isnan(result[f"subject_{i + 1}"].values[0]) for i in range(n))
    assert n_nan >= 1


def test_aft_predict_quantile_no_newdata(
    lung_data: tuple[Surv, np.ndarray],
) -> None:
    y, x = lung_data
    aft = gw.AFT("weibull").fit(y, x)
    result = aft.predict_quantile(p=[0.25, 0.75], format="pandas")
    assert result.shape[0] == 2
    assert f"subject_{x.shape[0]}" in result.columns


def test_predict_quantile_invalid_p(
    lung_data: tuple[Surv, np.ndarray],
) -> None:
    y, x = lung_data
    cox = gw.CoxPH().fit(y, x)
    with pytest.raises(ValueError, match="p must be in"):
        cox.predict_quantile(p=0.0)
    with pytest.raises(ValueError, match="p must be in"):
        cox.predict_quantile(p=1.0)
    with pytest.raises(ValueError, match="p must be in"):
        cox.predict_quantile(p=[0.5, 1.5])


def test_predict_quantile_monotone(lung_data: tuple[Surv, np.ndarray], newdata: np.ndarray) -> None:
    """Quantiles should be monotonically increasing in p."""
    y, x = lung_data
    cox = gw.CoxPH(ties="breslow").fit(y, x)
    result = cox.predict_quantile(newdata, p=[0.1, 0.25, 0.5, 0.75, 0.9], format="pandas")
    for i in range(3):
        vals = result[f"subject_{i + 1}"].values
        finite = vals[~np.isnan(vals)]
        assert np.all(np.diff(finite) >= 0), f"Non-monotone quantiles for subject {i + 1}"
