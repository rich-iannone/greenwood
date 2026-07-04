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
    return gw.data.load_dataset("lung", backend="pandas")


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
