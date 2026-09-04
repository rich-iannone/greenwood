"""R-parity tests for the Aalen additive hazards model.

Validates coefficient increments, cumulative coefficients, and summary
statistics against values computed by R's ``survival::aareg`` (fixtures:
aalen_lung_age_sex.json, aalen_veteran.json).
"""

from __future__ import annotations

import numpy as np
import pytest

import greenwood as gw
from greenwood import AalenAdditive, Surv

from ._r_parity import assert_allclose_to_r, load_fixture


@pytest.fixture(scope="module")
def lung_fixture():
    return load_fixture("aalen_lung_age_sex")


@pytest.fixture(scope="module")
def veteran_fixture():
    return load_fixture("aalen_veteran")


@pytest.fixture(scope="module")
def lung_fit(lung_fixture):
    lung = gw.load_dataset("lung", backend="pandas")
    y = Surv.right(lung["time"], event=(lung["status"] == 2))
    return AalenAdditive().fit(y, lung[["age", "sex"]])


@pytest.fixture(scope="module")
def veteran_fit(veteran_fixture):
    vet = gw.load_dataset("veteran", backend="pandas")
    y = Surv.right(vet["time"], event=(vet["status"] == 1))
    return AalenAdditive().fit(y, vet[["trt", "karno", "diagtime", "age"]])


def _stack_coef_columns(fixture: dict, key: str) -> np.ndarray:
    cols = fixture[key]
    return np.column_stack([np.array(v, dtype=float) for v in cols.values()])


@pytest.mark.rparity
class TestAalenLungRParity:
    def test_n_and_events(self, lung_fit, lung_fixture) -> None:
        assert lung_fit.n_ == lung_fixture["n"][0]

    def test_event_times(self, lung_fit, lung_fixture) -> None:
        assert_allclose_to_r(
            lung_fit.event_times_,
            lung_fixture["event_times"],
            what="event times",
        )

    def test_nrisk(self, lung_fit, lung_fixture) -> None:
        assert_allclose_to_r(
            lung_fit.nrisk_,
            lung_fixture["nrisk"],
            what="number at risk",
        )

    def test_coef_increments(self, lung_fit, lung_fixture) -> None:
        r_inc = _stack_coef_columns(lung_fixture, "coef_increments")
        assert_allclose_to_r(
            lung_fit.coef_increments_.ravel(),
            r_inc.ravel(),
            what="coefficient increments",
        )

    def test_cumulative_coefs(self, lung_fit, lung_fixture) -> None:
        r_cum = _stack_coef_columns(lung_fixture, "cumulative_coefs")
        assert_allclose_to_r(
            lung_fit.cumulative_coefs_.ravel(),
            r_cum.ravel(),
            what="cumulative coefficients",
        )

    def test_summary_slope(self, lung_fit, lung_fixture) -> None:
        assert_allclose_to_r(
            lung_fit.summary_slope_,
            lung_fixture["summary_slope"],
            what="summary slope",
            rtol=1e-6,
            atol=1e-9,
        )

    def test_summary_coef(self, lung_fit, lung_fixture) -> None:
        assert_allclose_to_r(
            lung_fit.summary_coef_,
            lung_fixture["summary_coef"],
            what="summary coef",
        )

    def test_summary_se(self, lung_fit, lung_fixture) -> None:
        assert_allclose_to_r(
            lung_fit.summary_se_,
            lung_fixture["summary_se"],
            what="summary se(coef)",
        )

    def test_summary_z(self, lung_fit, lung_fixture) -> None:
        assert_allclose_to_r(
            lung_fit.summary_z_,
            lung_fixture["summary_z"],
            what="summary z",
        )

    def test_summary_p(self, lung_fit, lung_fixture) -> None:
        assert_allclose_to_r(
            lung_fit.summary_p_,
            lung_fixture["summary_p"],
            what="summary p",
            rtol=1e-6,
            atol=1e-9,
        )

    def test_test_statistic(self, lung_fit, lung_fixture) -> None:
        assert_allclose_to_r(
            lung_fit.test_statistic_,
            lung_fixture["test_statistic"],
            what="test statistic",
        )

    def test_test_var(self, lung_fit, lung_fixture) -> None:
        r_var = _stack_coef_columns(lung_fixture, "test_var")
        assert_allclose_to_r(
            lung_fit.test_var_.ravel(),
            r_var.ravel(),
            what="test variance matrix",
            rtol=1e-6,
            atol=1e-6,
        )


@pytest.mark.rparity
class TestAalenVeteranRParity:
    def test_summary_coef(self, veteran_fit, veteran_fixture) -> None:
        assert_allclose_to_r(
            veteran_fit.summary_coef_,
            veteran_fixture["summary_coef"],
            what="veteran summary coef",
        )

    def test_summary_se(self, veteran_fit, veteran_fixture) -> None:
        assert_allclose_to_r(
            veteran_fit.summary_se_,
            veteran_fixture["summary_se"],
            what="veteran summary se(coef)",
        )

    def test_summary_z(self, veteran_fit, veteran_fixture) -> None:
        assert_allclose_to_r(
            veteran_fit.summary_z_,
            veteran_fixture["summary_z"],
            what="veteran summary z",
        )

    def test_test_statistic(self, veteran_fit, veteran_fixture) -> None:
        assert_allclose_to_r(
            veteran_fit.test_statistic_,
            veteran_fixture["test_statistic"],
            what="veteran test statistic",
        )
