"""Unit tests for the mixture cure model."""

from __future__ import annotations

import numpy as np
import pytest

import greenwood as gw
from greenwood import MixtureCure, Surv


@pytest.fixture(scope="module")
def e1684_data():
    e1684 = gw.load_dataset("e1684", backend="pandas")
    y = Surv.right(e1684["FAILTIME"], event=e1684["FAILCENS"].astype(bool))
    return y, e1684


@pytest.fixture(scope="module")
def fitted_model(e1684_data):
    y, e1684 = e1684_data
    return MixtureCure(emmax=50, eps=1e-7).fit(
        y, latency=e1684[["TRT"]], cure=e1684[["TRT"]], nboot=0
    )


class TestInit:
    def test_defaults(self) -> None:
        m = MixtureCure()
        assert m.emmax == 50
        assert m.eps == 1e-7

    def test_custom_params(self) -> None:
        m = MixtureCure(emmax=100, eps=1e-5)
        assert m.emmax == 100
        assert m.eps == 1e-5


class TestFit:
    def test_basic_attributes(self, fitted_model) -> None:
        m = fitted_model
        assert m.n_ == 284
        assert m.n_event_ == 196
        assert m.n_iter_ > 0
        assert len(m.cure_term_names_) == 2
        assert len(m.latency_term_names_) == 1

    def test_cure_coef_shape(self, fitted_model) -> None:
        assert fitted_model.cure_coef_.shape == (2,)

    def test_latency_coef_shape(self, fitted_model) -> None:
        assert fitted_model.latency_coef_.shape == (1,)

    def test_baseline_survival_monotone(self, fitted_model) -> None:
        assert np.all(np.diff(fitted_model.baseline_survival_) <= 0)

    def test_baseline_survival_range(self, fitted_model) -> None:
        assert np.all(fitted_model.baseline_survival_ >= 0)
        assert np.all(fitted_model.baseline_survival_ <= 1)

    def test_baseline_times_sorted(self, fitted_model) -> None:
        assert np.all(np.diff(fitted_model.baseline_times_) > 0)

    def test_numpy_array_input(self, e1684_data) -> None:
        y, e1684 = e1684_data
        x = e1684[["TRT"]].to_numpy()
        m = MixtureCure().fit(y, latency=x, cure=x, nboot=0)
        assert m.n_ == 284

    def test_no_events_raises(self) -> None:
        y = Surv.right(np.array([1.0, 2.0, 3.0]), event=np.array([False, False, False]))
        x = np.array([[1.0], [2.0], [3.0]])
        with pytest.raises(ValueError, match="No events"):
            MixtureCure().fit(y, latency=x, cure=x, nboot=0)

    def test_shape_mismatch_raises(self) -> None:
        y = Surv.right(np.array([1.0, 2.0, 3.0]), event=np.array([True, False, True]))
        x = np.array([[1.0], [2.0]])
        z = np.array([[1.0], [2.0], [3.0]])
        with pytest.raises(ValueError, match="same number of rows"):
            MixtureCure().fit(y, latency=x, cure=z, nboot=0)

    def test_unsupported_surv_type_raises(self) -> None:
        y = Surv.left(np.array([1.0, 2.0, 3.0]))
        x = np.array([[1.0], [2.0], [3.0]])
        with pytest.raises(NotImplementedError, match="right-censored"):
            MixtureCure().fit(y, latency=x, cure=x, nboot=0)


class TestPredict:
    def test_cure_prob_range(self, fitted_model, e1684_data) -> None:
        _, e1684 = e1684_data
        probs = fitted_model.predict_cure_prob(e1684[["TRT"]])
        assert np.all(probs >= 0)
        assert np.all(probs <= 1)

    def test_cure_prob_shape(self, fitted_model, e1684_data) -> None:
        _, e1684 = e1684_data
        probs = fitted_model.predict_cure_prob(e1684[["TRT"]])
        assert probs.shape == (284,)

    def test_cure_prob_trt_effect(self, fitted_model) -> None:
        x0 = np.array([[0]])
        x1 = np.array([[1]])
        p0 = fitted_model.predict_cure_prob(x0)
        p1 = fitted_model.predict_cure_prob(x1)
        assert p0[0] != p1[0]

    def test_survival_shape(self, fitted_model) -> None:
        x = np.array([[0], [1]])
        times = np.array([0.5, 1.0, 2.0, 5.0])
        surv = fitted_model.predict_survival(times, x, x)
        assert surv.shape == (2, 4)

    def test_survival_range(self, fitted_model) -> None:
        x = np.array([[0], [1]])
        times = np.array([0.5, 1.0, 2.0, 5.0])
        surv = fitted_model.predict_survival(times, x, x)
        assert np.all(surv >= 0)
        assert np.all(surv <= 1)

    def test_survival_decreasing(self, fitted_model) -> None:
        x = np.array([[0]])
        times = np.linspace(0.1, 10, 50)
        surv = fitted_model.predict_survival(times, x, x)
        assert np.all(np.diff(surv[0]) <= 1e-10)

    def test_survival_plateau(self, fitted_model) -> None:
        x = np.array([[0]])
        times = np.array([50.0, 100.0])
        surv = fitted_model.predict_survival(times, x, x)
        cure_frac = 1.0 - fitted_model.predict_cure_prob(x)[0]
        np.testing.assert_allclose(surv[0], cure_frac, atol=1e-6)


class TestOutputMethods:
    def test_to_frame_polars(self, fitted_model) -> None:
        df = fitted_model.to_frame(format="polars")
        import polars as pl

        assert isinstance(df, pl.DataFrame)
        assert set(df.columns) == {"submodel", "term", "coef", "se", "z", "p"}
        assert len(df) == 3

    def test_to_frame_submodel_labels(self, fitted_model) -> None:
        df = fitted_model.to_frame(format="polars")
        submodels = df["submodel"].to_list()
        assert submodels == ["cure", "cure", "latency"]


class TestRepr:
    def test_unfitted_repr(self) -> None:
        m = MixtureCure()
        assert "unfitted" in repr(m)

    def test_fitted_repr(self, fitted_model) -> None:
        r = repr(fitted_model)
        assert "MixtureCure" in r
        assert "Cure probability model" in r
        assert "Failure time distribution model" in r
        assert "(Intercept)" in r
        assert "TRT" in r
        assert "n = 284" in r


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
        assert df["n"][0] == 284
