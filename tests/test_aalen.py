"""Unit tests for the Aalen additive hazards model."""

from __future__ import annotations

import numpy as np
import pytest

import greenwood as gw
from greenwood import AalenAdditive, Surv


@pytest.fixture(scope="module")
def lung_data():
    lung = gw.load_dataset("lung", backend="pandas")
    y = Surv.right(lung["time"], event=(lung["status"] == 2))
    return y, lung


@pytest.fixture(scope="module")
def fitted_model(lung_data):
    y, lung = lung_data
    return AalenAdditive().fit(y, lung[["age", "sex"]])


class TestInit:
    def test_defaults(self) -> None:
        m = AalenAdditive()
        assert m.nmin is None
        assert m.test == "aalen"
        assert m.qrtol == 1e-7

    def test_custom_params(self) -> None:
        m = AalenAdditive(nmin=10, test="nrisk", qrtol=1e-5)
        assert m.nmin == 10
        assert m.test == "nrisk"
        assert m.qrtol == 1e-5

    def test_invalid_test(self) -> None:
        with pytest.raises(ValueError, match="test must be"):
            AalenAdditive(test="bogus")


class TestFit:
    def test_basic_attributes(self, fitted_model) -> None:
        m = fitted_model
        assert m.n_ == 228
        assert m.n_event_ == 165
        assert m.n_event_times_used_ > 0
        assert len(m.term_names_) == 3
        assert m.term_names_[0] == "Intercept"

    def test_coef_increments_shape(self, fitted_model) -> None:
        m = fitted_model
        assert m.coef_increments_.shape == (m.n_event_times_used_, 3)

    def test_cumulative_is_cumsum(self, fitted_model) -> None:
        m = fitted_model
        np.testing.assert_allclose(m.cumulative_coefs_, np.cumsum(m.coef_increments_, axis=0))

    def test_event_times_sorted(self, fitted_model) -> None:
        m = fitted_model
        assert np.all(np.diff(m.event_times_) >= 0)

    def test_nmin_default_is_3p(self, lung_data) -> None:
        y, lung = lung_data
        m6 = AalenAdditive(nmin=6).fit(y, lung[["age", "sex"]])
        m_default = AalenAdditive().fit(y, lung[["age", "sex"]])
        assert m_default.n_event_times_used_ == m6.n_event_times_used_

    def test_custom_nmin_reduces_events(self, lung_data) -> None:
        y, lung = lung_data
        m_large = AalenAdditive(nmin=100).fit(y, lung[["age", "sex"]])
        m_default = AalenAdditive().fit(y, lung[["age", "sex"]])
        assert m_large.n_event_times_used_ <= m_default.n_event_times_used_

    def test_nrisk_test(self, lung_data) -> None:
        y, lung = lung_data
        m = AalenAdditive(test="nrisk").fit(y, lung[["age", "sex"]])
        assert m.test == "nrisk"
        assert len(m.summary_z_) == 3

    def test_numpy_array_input(self, lung_data) -> None:
        y, lung = lung_data
        x = lung[["age", "sex"]].to_numpy()
        m = AalenAdditive().fit(y, x)
        assert m.n_ == 228

    def test_no_events_raises(self) -> None:
        y = Surv.right(np.array([1.0, 2.0, 3.0]), event=np.array([False, False, False]))
        x = np.array([[1.0], [2.0], [3.0]])
        with pytest.raises(ValueError, match="No events"):
            AalenAdditive().fit(y, x)

    def test_shape_mismatch_raises(self) -> None:
        y = Surv.right(np.array([1.0, 2.0, 3.0]), event=np.array([True, False, True]))
        x = np.array([[1.0], [2.0]])
        with pytest.raises(ValueError, match="same number of rows"):
            AalenAdditive().fit(y, x)

    def test_unsupported_surv_type_raises(self) -> None:
        y = Surv.left(np.array([1.0, 2.0, 3.0]))
        x = np.array([[1.0], [2.0], [3.0]])
        with pytest.raises(NotImplementedError, match="right-censored"):
            AalenAdditive().fit(y, x)

    def test_tweight_stored(self, fitted_model) -> None:
        m = fitted_model
        assert m.tweight_.shape == (m.n_event_times_used_, 3)
        assert np.all(m.tweight_ > 0)


class TestOutputMethods:
    def test_cumulative_coefficients_default(self, fitted_model) -> None:
        df = fitted_model.cumulative_coefficients()
        assert "time" in str(type(df)) or hasattr(df, "columns")

    def test_cumulative_coefficients_polars(self, fitted_model) -> None:
        df = fitted_model.cumulative_coefficients(format="polars")
        import polars as pl

        assert isinstance(df, pl.DataFrame)
        assert "time" in df.columns
        assert "Intercept" in df.columns
        assert len(df) == fitted_model.n_event_times_used_

    def test_to_frame_default(self, fitted_model) -> None:
        df = fitted_model.to_frame()
        assert hasattr(df, "columns") or hasattr(df, "schema")

    def test_to_frame_polars(self, fitted_model) -> None:
        df = fitted_model.to_frame(format="polars")
        import polars as pl

        assert isinstance(df, pl.DataFrame)
        assert set(df.columns) == {"term", "slope", "coef", "se", "z", "p"}
        assert len(df) == 3


class TestRepr:
    def test_unfitted_repr(self) -> None:
        m = AalenAdditive()
        assert "unfitted" in repr(m)

    def test_fitted_repr(self, fitted_model) -> None:
        r = repr(fitted_model)
        assert "AalenAdditive" in r
        assert "Intercept" in r
        assert "age" in r
        assert "sex" in r
        assert "n = 228" in r


class TestPredict:
    def test_survival_shape(self, fitted_model, lung_data) -> None:
        _, lung = lung_data
        surv = fitted_model.predict(lung[["age", "sex"]][:3], times=[180, 365])
        assert surv.shape == (2, 4)

    def test_cumhaz_shape(self, fitted_model, lung_data) -> None:
        _, lung = lung_data
        ch = fitted_model.predict(lung[["age", "sex"]][:3], type="cumhaz", times=[180, 365])
        assert ch.shape == (2, 4)

    def test_survival_in_unit_interval(self, fitted_model, lung_data) -> None:
        _, lung = lung_data
        surv = fitted_model.predict(lung[["age", "sex"]], times=[100, 200, 365, 500])
        for col in surv.columns[1:]:
            vals = surv[col].to_numpy()
            assert np.all(vals >= 0.0) and np.all(vals <= 1.0)

    def test_cumhaz_nonnegative(self, fitted_model, lung_data) -> None:
        _, lung = lung_data
        ch = fitted_model.predict(lung[["age", "sex"]], type="cumhaz", times=[100, 365])
        for col in ch.columns[1:]:
            assert np.all(ch[col].to_numpy() >= 0.0)

    def test_survival_decreasing(self, fitted_model, lung_data) -> None:
        _, lung = lung_data
        surv = fitted_model.predict(lung[["age", "sex"]][:1], times=[50, 100, 200, 365, 500])
        vals = surv["subject_1"].to_numpy()
        assert np.all(np.diff(vals) <= 1e-12)

    def test_cumhaz_increasing(self, fitted_model, lung_data) -> None:
        _, lung = lung_data
        ch = fitted_model.predict(
            lung[["age", "sex"]][:1], type="cumhaz", times=[50, 100, 200, 365, 500]
        )
        vals = ch["subject_1"].to_numpy()
        assert np.all(np.diff(vals) >= -1e-12)

    def test_survival_exp_neg_cumhaz(self, fitted_model, lung_data) -> None:
        _, lung = lung_data
        nd = lung[["age", "sex"]][:2]
        surv = fitted_model.predict(nd, times=[180, 365])
        ch = fitted_model.predict(nd, type="cumhaz", times=[180, 365])
        for col in ["subject_1", "subject_2"]:
            np.testing.assert_allclose(
                surv[col].to_numpy(), np.exp(-ch[col].to_numpy()), rtol=1e-12
            )

    def test_newdata_none_uses_training(self, fitted_model) -> None:
        surv = fitted_model.predict(times=[180, 365])
        assert surv.shape[1] == 228 + 1

    def test_default_times_uses_event_times(self, fitted_model, lung_data) -> None:
        _, lung = lung_data
        surv = fitted_model.predict(lung[["age", "sex"]][:1])
        assert len(surv) == len(fitted_model.event_times_)

    def test_before_first_event_time(self, fitted_model, lung_data) -> None:
        _, lung = lung_data
        surv = fitted_model.predict(lung[["age", "sex"]][:1], times=[1.0])
        assert surv["subject_1"].to_numpy()[0] == pytest.approx(1.0)

    def test_invalid_type_raises(self, fitted_model, lung_data) -> None:
        _, lung = lung_data
        with pytest.raises(ValueError, match="Unknown predict type"):
            fitted_model.predict(lung[["age", "sex"]][:1], type="hazard", times=[180])

    def test_format_polars(self, fitted_model, lung_data) -> None:
        _, lung = lung_data
        import polars as pl

        surv = fitted_model.predict(lung[["age", "sex"]][:2], times=[180], format="polars")
        assert isinstance(surv, pl.DataFrame)


class TestTidyGlance:
    def test_tidy(self, fitted_model) -> None:
        df = gw.tidy(fitted_model, format="polars")
        import polars as pl

        assert isinstance(df, pl.DataFrame)
        assert len(df) == 3

    def test_glance(self, fitted_model) -> None:
        df = gw.glance(fitted_model, format="polars")
        import polars as pl

        assert isinstance(df, pl.DataFrame)
        assert len(df) == 1
        assert df["n"][0] == 228
