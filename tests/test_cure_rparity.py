"""R-parity tests for the mixture cure model.

Validates EM coefficients, baseline survival, and predictions against
values computed by R's `smcure` package (fixture: cure_ph_e1684.json).
"""

from __future__ import annotations

import numpy as np
import pytest

import greenwood as gw
from greenwood import MixtureCure, Surv

from ._r_parity import assert_allclose_to_r, load_fixture


@pytest.fixture(scope="module")
def fixture():
    return load_fixture("cure_ph_e1684")


@pytest.fixture(scope="module")
def fitted_model():
    e1684 = gw.load_dataset("e1684", backend="pandas")
    y = Surv.right(e1684["FAILTIME"], event=e1684["FAILCENS"].astype(bool))
    return MixtureCure(emmax=50, eps=1e-7).fit(
        y, latency=e1684[["TRT"]], cure=e1684[["TRT"]], nboot=0
    )


@pytest.mark.rparity
class TestMixtureCureRParity:
    def test_n(self, fitted_model, fixture) -> None:
        assert fitted_model.n_ == fixture["n"]

    def test_n_events(self, fitted_model, fixture) -> None:
        assert fitted_model.n_event_ == fixture["n_events"]

    def test_cure_coef(self, fitted_model, fixture) -> None:
        assert_allclose_to_r(
            fitted_model.cure_coef_,
            fixture["cure_coef"],
            what="cure coefficients",
            rtol=1e-6,
            atol=1e-6,
        )

    def test_latency_coef(self, fitted_model, fixture) -> None:
        assert_allclose_to_r(
            fitted_model.latency_coef_,
            fixture["latency_coef"],
            what="latency coefficients",
            rtol=1e-6,
            atol=1e-6,
        )

    def test_baseline_times(self, fitted_model, fixture) -> None:
        assert_allclose_to_r(
            fitted_model.baseline_times_,
            fixture["baseline_times"],
            what="baseline event times",
        )

    def test_baseline_survival(self, fitted_model, fixture) -> None:
        r_surv = np.array([float(x) for x in fixture["baseline_survival"]])
        assert_allclose_to_r(
            fitted_model.baseline_survival_,
            r_surv,
            what="baseline survival",
            rtol=1e-6,
            atol=1e-6,
        )
