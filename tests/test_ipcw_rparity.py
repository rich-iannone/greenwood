"""R-parity tests for CensoringDistribution and IPCRidge.

Validates IPC weights, censoring survival, and ridge coefficients against
values computed by R's survival package (fixture: ipcridge_lung.json).
"""

from __future__ import annotations

import numpy as np
import pytest

import greenwood as gw
from greenwood import CensoringDistribution, IPCRidge, Surv

from ._r_parity import assert_allclose_to_r, load_fixture


@pytest.fixture(scope="module")
def fixture():
    return load_fixture("ipcridge_lung")


@pytest.fixture(scope="module")
def lung_data():
    lung = gw.load_dataset("lung", backend="pandas")
    y = Surv.right(lung["time"], event=(lung["status"] == 2))
    return y, lung


@pytest.mark.rparity
class TestCensoringDistributionRParity:
    def test_censoring_survival_matches_r(self, fixture, lung_data) -> None:
        y, _ = lung_data
        cens = CensoringDistribution(y)
        r_times = np.array(fixture["censoring_times"], dtype=float)
        r_surv = np.array(fixture["censoring_surv"], dtype=float)
        py_surv = cens.survival(r_times)
        assert_allclose_to_r(py_surv, r_surv, what="censoring survival G(t)", rtol=1e-6, atol=1e-6)

    def test_ipc_weights_match_r(self, fixture, lung_data) -> None:
        y, _ = lung_data
        cens = CensoringDistribution(y)
        r_weights = np.array(fixture["ipc_weights"], dtype=float)
        py_weights = cens.weights()
        assert_allclose_to_r(py_weights, r_weights, what="IPC weights", rtol=1e-6, atol=1e-6)


@pytest.mark.rparity
class TestIPCRidgeRParity:
    def test_coefficients_match_r(self, fixture, lung_data) -> None:
        y, lung = lung_data
        alpha = fixture["ridge_alpha"]
        cols = fixture["covariates"]
        model = IPCRidge(alpha=alpha).fit(y, lung[cols])

        r_coef = np.array(fixture["ridge_coef"], dtype=float)
        r_intercept = fixture["ridge_intercept"]

        assert_allclose_to_r(
            model.coef_, r_coef, what="IPCRidge coefficients", rtol=1e-6, atol=1e-6
        )
        assert_allclose_to_r(
            np.array([model.intercept_]),
            np.array([r_intercept]),
            what="IPCRidge intercept",
            rtol=1e-6,
            atol=1e-6,
        )

    def test_predictions_match_r(self, fixture, lung_data) -> None:
        y, lung = lung_data
        alpha = fixture["ridge_alpha"]
        cols = fixture["covariates"]
        model = IPCRidge(alpha=alpha).fit(y, lung[cols])

        r_lp = np.array(fixture["ridge_lp"], dtype=float)
        py_lp = model.predict(lung[cols], type="lp")

        assert_allclose_to_r(py_lp, r_lp, what="IPCRidge linear predictor", rtol=1e-6, atol=1e-6)

    def test_n_and_nevents_match_r(self, fixture, lung_data) -> None:
        y, lung = lung_data
        alpha = fixture["ridge_alpha"]
        cols = fixture["covariates"]
        model = IPCRidge(alpha=alpha).fit(y, lung[cols])

        assert model.n_ == fixture["n"]
        assert model.n_event_ == fixture["n_event"]
