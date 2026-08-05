"""Tests for predict_expectation across CoxPH, AFT, and RoystonParmar."""

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
def fixture() -> dict:
    return load_fixture("predict_expectation")


@pytest.fixture()
def newdata(fixture: dict) -> np.ndarray:
    return np.column_stack([fixture["newdata"]["age"], fixture["newdata"]["sex"]])


# ── CoxPH predict_expectation ──


def test_cox_predict_expectation_365(
    lung_data: tuple[Surv, np.ndarray],
    fixture: dict,
    newdata: np.ndarray,
) -> None:
    y, x = lung_data
    cox = gw.CoxPH(ties="breslow").fit(y, x)
    result = cox.predict_expectation(newdata, tau=365, format="pandas")

    assert "tau" in result.columns
    assert result["tau"].values[0] == 365.0
    assert result.shape[0] == 1

    r = fixture["cox_breslow"]["rmst_365"]
    for i in range(3):
        assert_allclose_to_r(
            result[f"subject_{i + 1}"].values,
            [r[i]],
            what=f"cox rmst tau=365 subject {i + 1}",
            rtol=1e-4,
            atol=1.0,
        )


def test_cox_predict_expectation_730(
    lung_data: tuple[Surv, np.ndarray],
    fixture: dict,
    newdata: np.ndarray,
) -> None:
    y, x = lung_data
    cox = gw.CoxPH(ties="breslow").fit(y, x)
    result = cox.predict_expectation(newdata, tau=730, format="pandas")

    r = fixture["cox_breslow"]["rmst_730"]
    for i in range(3):
        assert_allclose_to_r(
            result[f"subject_{i + 1}"].values,
            [r[i]],
            what=f"cox rmst tau=730 subject {i + 1}",
            rtol=1e-4,
            atol=1.0,
        )


def test_cox_predict_expectation_ci(
    lung_data: tuple[Surv, np.ndarray], newdata: np.ndarray
) -> None:
    y, x = lung_data
    cox = gw.CoxPH(ties="breslow").fit(y, x)
    result = cox.predict_expectation(newdata, tau=365, ci=True, format="pandas")

    for i in range(3):
        col = f"subject_{i + 1}"
        assert f"{col}_lower" in result.columns
        assert f"{col}_upper" in result.columns
        lo = result[f"{col}_lower"].values[0]
        val = result[col].values[0]
        hi = result[f"{col}_upper"].values[0]
        assert lo < val < hi, f"CI ordering violated for subject {i + 1}"


def test_cox_predict_expectation_no_newdata(
    lung_data: tuple[Surv, np.ndarray],
) -> None:
    y, x = lung_data
    cox = gw.CoxPH(ties="breslow").fit(y, x)
    result = cox.predict_expectation(tau=365, format="pandas")
    assert result.shape[0] == 1
    assert f"subject_{x.shape[0]}" in result.columns


# ── AFT predict_expectation ──


def test_aft_predict_expectation_365(
    lung_data: tuple[Surv, np.ndarray],
    fixture: dict,
    newdata: np.ndarray,
) -> None:
    y, x = lung_data
    aft = gw.AFT("weibull").fit(y, x)
    result = aft.predict_expectation(newdata, tau=365, format="pandas")

    r = fixture["aft_weibull"]["rmst_365"]
    for i in range(3):
        assert_allclose_to_r(
            result[f"subject_{i + 1}"].values,
            [r[i]],
            what=f"aft rmst tau=365 subject {i + 1}",
            rtol=1e-3,
        )


def test_aft_predict_expectation_730(
    lung_data: tuple[Surv, np.ndarray],
    fixture: dict,
    newdata: np.ndarray,
) -> None:
    y, x = lung_data
    aft = gw.AFT("weibull").fit(y, x)
    result = aft.predict_expectation(newdata, tau=730, format="pandas")

    r = fixture["aft_weibull"]["rmst_730"]
    for i in range(3):
        assert_allclose_to_r(
            result[f"subject_{i + 1}"].values,
            [r[i]],
            what=f"aft rmst tau=730 subject {i + 1}",
            rtol=1e-3,
        )


def test_aft_predict_expectation_ci(
    lung_data: tuple[Surv, np.ndarray], newdata: np.ndarray
) -> None:
    y, x = lung_data
    aft = gw.AFT("weibull").fit(y, x)
    result = aft.predict_expectation(newdata, tau=365, ci=True, format="pandas")

    for i in range(3):
        col = f"subject_{i + 1}"
        lo = result[f"{col}_lower"].values[0]
        val = result[col].values[0]
        hi = result[f"{col}_upper"].values[0]
        assert lo < val < hi, f"CI ordering violated for subject {i + 1}"


def test_aft_predict_expectation_no_newdata(
    lung_data: tuple[Surv, np.ndarray],
) -> None:
    y, x = lung_data
    aft = gw.AFT("weibull").fit(y, x)
    result = aft.predict_expectation(tau=365, format="pandas")
    assert result.shape[0] == 1
    assert f"subject_{x.shape[0]}" in result.columns


# ── RoystonParmar predict_expectation ──


def test_rp_predict_expectation(lung_data: tuple[Surv, np.ndarray], newdata: np.ndarray) -> None:
    y, x = lung_data
    rp = gw.RoystonParmar(df=3).fit(y, x)
    result = rp.predict_expectation(newdata, tau=365, format="pandas")

    assert "tau" in result.columns
    assert result.shape[0] == 1
    for i in range(3):
        val = result[f"subject_{i + 1}"].values[0]
        assert np.isfinite(val) and 0.0 < val <= 365.0


def test_rp_predict_expectation_ci(lung_data: tuple[Surv, np.ndarray], newdata: np.ndarray) -> None:
    y, x = lung_data
    rp = gw.RoystonParmar(df=3).fit(y, x)
    result = rp.predict_expectation(newdata, tau=365, ci=True, format="pandas")

    for i in range(3):
        col = f"subject_{i + 1}"
        lo = result[f"{col}_lower"].values[0]
        val = result[col].values[0]
        hi = result[f"{col}_upper"].values[0]
        assert lo < val < hi, f"CI ordering violated for subject {i + 1}"


def test_rp_predict_expectation_consistency(
    lung_data: tuple[Surv, np.ndarray], newdata: np.ndarray
) -> None:
    """RMST should equal the area under the survival curve (trapezoidal check)."""
    y, x = lung_data
    rp = gw.RoystonParmar(df=3).fit(y, x)
    tau = 365.0
    rmst_df = rp.predict_expectation(newdata, tau=tau, format="pandas")
    times = np.linspace(0.1, tau, 2000)
    surv_df = rp.predict(newdata, type="survival", times=times, format="pandas")
    for i in range(3):
        rmst_val = rmst_df[f"subject_{i + 1}"].values[0]
        s_vals = surv_df[f"subject_{i + 1}"].values
        trap_area = float(np.trapezoid(s_vals, times))
        np.testing.assert_allclose(rmst_val, trap_area, rtol=1e-3)


# ── Edge cases ──


def test_predict_expectation_invalid_tau(
    lung_data: tuple[Surv, np.ndarray],
) -> None:
    y, x = lung_data
    cox = gw.CoxPH().fit(y, x)
    with pytest.raises(ValueError, match="tau must be positive"):
        cox.predict_expectation(tau=0.0)
    with pytest.raises(ValueError, match="tau must be positive"):
        cox.predict_expectation(tau=-10.0)


def test_predict_expectation_monotone_in_tau(
    lung_data: tuple[Surv, np.ndarray], newdata: np.ndarray
) -> None:
    """RMST should be monotonically increasing in tau."""
    y, x = lung_data
    cox = gw.CoxPH(ties="breslow").fit(y, x)
    taus = [100, 200, 365, 500, 730]
    prev = np.zeros(3)
    for t in taus:
        result = cox.predict_expectation(newdata, tau=t, format="pandas")
        for i in range(3):
            val = result[f"subject_{i + 1}"].values[0]
            assert val >= prev[i], f"Non-monotone RMST at tau={t} subject {i + 1}"
            prev[i] = val


def test_predict_expectation_bounded_by_tau(
    lung_data: tuple[Surv, np.ndarray], newdata: np.ndarray
) -> None:
    """RMST must be in (0, tau]."""
    y, x = lung_data
    aft = gw.AFT("weibull").fit(y, x)
    for tau in [100.0, 365.0, 1000.0]:
        result = aft.predict_expectation(newdata, tau=tau, format="pandas")
        for i in range(3):
            val = result[f"subject_{i + 1}"].values[0]
            assert 0.0 < val <= tau, f"RMST out of bounds at tau={tau} subject {i + 1}"
