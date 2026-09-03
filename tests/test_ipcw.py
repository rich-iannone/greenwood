"""Tests for `CensoringDistribution` and `IPCRidge`.

Validates IPC weights, censoring survival evaluation, and the IPC-weighted ridge estimator.
Structural and statistical properties are checked: weights are non-negative, zero for censored
subjects, IPC-weighted ridge beats random discrimination, predictions are consistent, and the
model is reproducible.
"""

from __future__ import annotations

import numpy as np
import pytest

import greenwood as gw
from greenwood import CensoringDistribution, IPCRidge, Surv


@pytest.fixture(scope="module")
def lung_data():
    lung = gw.load_dataset("lung", backend="pandas").dropna(
        subset=["ph.ecog", "ph.karno", "wt.loss"]
    )
    y = Surv.right(lung["time"], event=(lung["status"] == 2))
    return y, lung


# ---------------------------------------------------------------------------
# CensoringDistribution
# ---------------------------------------------------------------------------


class TestCensoringDistribution:
    def test_survival_starts_at_one(self, lung_data) -> None:
        y, _ = lung_data
        cens = CensoringDistribution(y)
        assert cens.survival(np.array([0.0]))[0] == 1.0

    def test_survival_is_nonincreasing(self, lung_data) -> None:
        y, _ = lung_data
        cens = CensoringDistribution(y)
        times = np.linspace(0, float(y.stop.max()), 100)
        g = cens.survival(times)
        assert np.all(np.diff(g) <= 1e-12)

    def test_survival_left_precedes_survival(self, lung_data) -> None:
        y, _ = lung_data
        cens = CensoringDistribution(y)
        event_times = np.sort(np.unique(y.stop))
        g = cens.survival(event_times)
        g_left = cens.survival_left(event_times)
        assert np.all(g_left >= g - 1e-12)

    def test_weights_zero_for_censored(self, lung_data) -> None:
        y, _ = lung_data
        cens = CensoringDistribution(y)
        w = cens.weights()
        censored = ~y.event.astype(bool)
        assert np.all(w[censored] == 0.0)

    def test_weights_positive_for_events(self, lung_data) -> None:
        y, _ = lung_data
        cens = CensoringDistribution(y)
        w = cens.weights()
        events = y.event.astype(bool)
        assert np.all(w[events] > 0.0)

    def test_weights_ge_one_for_events(self, lung_data) -> None:
        y, _ = lung_data
        cens = CensoringDistribution(y)
        w = cens.weights()
        events = y.event.astype(bool)
        assert np.all(w[events] >= 1.0 - 1e-12)

    def test_tau_truncates_weights(self, lung_data) -> None:
        y, _ = lung_data
        tau = float(np.median(y.stop[y.event.astype(bool)]))
        cens = CensoringDistribution(y)
        w = cens.weights(tau=tau)
        assert np.all(w[y.stop > tau] == 0.0)

    def test_to_frame(self, lung_data) -> None:
        y, _ = lung_data
        cens = CensoringDistribution(y)
        df = cens.to_frame(format="pandas")
        assert "time" in df.columns
        assert "survival" in df.columns
        assert len(df) == cens._times.shape[0]

    def test_repr(self, lung_data) -> None:
        y, _ = lung_data
        cens = CensoringDistribution(y)
        r = repr(cens)
        assert "CensoringDistribution" in r
        assert "censored=" in r

    def test_no_censoring(self) -> None:
        y = Surv.right(np.array([1.0, 2.0, 3.0]), event=np.array([True, True, True]))
        cens = CensoringDistribution(y)
        w = cens.weights()
        assert np.allclose(w, 1.0)


# ---------------------------------------------------------------------------
# IPCRidge
# ---------------------------------------------------------------------------


class TestIPCRidge:
    def test_fit_basic(self, lung_data) -> None:
        y, lung = lung_data
        model = IPCRidge(alpha=1.0).fit(y, lung[["age", "sex"]])
        assert model.n_ > 0
        assert model.n_event_ > 0
        assert len(model.coef_) == 2
        assert model.intercept_ != 0.0

    def test_repr_unfitted(self) -> None:
        assert "unfitted" in repr(IPCRidge(alpha=1.0))

    def test_repr_fitted(self, lung_data) -> None:
        y, lung = lung_data
        model = IPCRidge(alpha=1.0).fit(y, lung[["age", "sex"]])
        r = repr(model)
        assert "IPCRidge" in r
        assert "age" in r
        assert "sex" in r

    def test_predict_lp_and_response_consistent(self, lung_data) -> None:
        y, lung = lung_data
        cols = ["age", "sex"]
        model = IPCRidge(alpha=1.0).fit(y, lung[cols])
        lp = model.predict(lung[cols], type="lp")
        resp = model.predict(lung[cols], type="response")
        assert np.allclose(resp, np.exp(lp))

    def test_predict_none_uses_training_data(self, lung_data) -> None:
        y, lung = lung_data
        cols = ["age", "sex"]
        model = IPCRidge(alpha=1.0).fit(y, lung[cols])
        lp_default = model.predict(type="lp")
        lp_explicit = model.predict(lung[cols], type="lp")
        assert lp_default.shape[0] == lp_explicit.shape[0]

    def test_predict_invalid_type(self, lung_data) -> None:
        y, lung = lung_data
        model = IPCRidge(alpha=1.0).fit(y, lung[["age", "sex"]])
        with pytest.raises(ValueError, match="Unknown predict type"):
            model.predict(type="survival")

    def test_to_frame(self, lung_data) -> None:
        y, lung = lung_data
        model = IPCRidge(alpha=1.0).fit(y, lung[["age", "sex"]])
        df = model.to_frame(format="pandas")
        assert "term" in df.columns
        assert "estimate" in df.columns
        assert len(df) == 3  # intercept + 2 covariates

    def test_beats_random_discrimination(self, lung_data) -> None:
        y, lung = lung_data
        cols = ["age", "sex", "ph.ecog", "ph.karno", "wt.loss"]
        model = IPCRidge(alpha=1.0).fit(y, lung[cols])
        lp = model.predict(lung[cols], type="lp")
        c = gw.concordance_index(y, -lp)
        assert c > 0.55

    def test_alpha_zero_is_unpenalized(self, lung_data) -> None:
        y, lung = lung_data
        cols = ["age", "sex"]
        m0 = IPCRidge(alpha=0.0).fit(y, lung[cols])
        m1 = IPCRidge(alpha=10.0).fit(y, lung[cols])
        assert np.linalg.norm(m0.coef_) > np.linalg.norm(m1.coef_)

    def test_higher_alpha_shrinks_coefficients(self, lung_data) -> None:
        y, lung = lung_data
        cols = ["age", "sex"]
        norms = []
        for alpha in [0.01, 1.0, 100.0]:
            m = IPCRidge(alpha=alpha).fit(y, lung[cols])
            norms.append(np.linalg.norm(m.coef_))
        assert norms[0] > norms[1] > norms[2]

    def test_negative_alpha_raises(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            IPCRidge(alpha=-1.0)

    def test_no_events_raises(self) -> None:
        y = Surv.right(np.array([1.0, 2.0, 3.0]), event=np.array([False, False, False]))
        with pytest.raises(ValueError, match="No events"):
            IPCRidge(alpha=1.0).fit(y, np.array([[1.0], [2.0], [3.0]]))

    def test_right_censored_only(self) -> None:
        y = Surv.counting(
            start=np.array([0.0, 0.0]),
            stop=np.array([1.0, 2.0]),
            event=np.array([True, False]),
        )
        with pytest.raises(NotImplementedError, match="right-censored"):
            IPCRidge(alpha=1.0).fit(y, np.array([[1.0], [2.0]]))

    def test_mismatched_rows_raises(self, lung_data) -> None:
        y, _ = lung_data
        with pytest.raises(ValueError, match="same number of rows"):
            IPCRidge(alpha=1.0).fit(y, np.array([[1.0, 2.0]]))

    def test_tidy_and_glance(self, lung_data) -> None:
        y, lung = lung_data
        model = IPCRidge(alpha=1.0).fit(y, lung[["age", "sex"]])
        tidy_df = gw.tidy(model, format="pandas")
        assert "term" in tidy_df.columns
        glance_df = gw.glance(model, format="pandas")
        assert "n" in glance_df.columns
        assert "alpha" in glance_df.columns

    def test_standardize_false(self, lung_data) -> None:
        y, lung = lung_data
        cols = ["age", "sex"]
        m_std = IPCRidge(alpha=1.0, standardize=True).fit(y, lung[cols])
        m_raw = IPCRidge(alpha=1.0, standardize=False).fit(y, lung[cols])
        assert not np.allclose(m_std.coef_, m_raw.coef_)

    def test_handles_missing_values(self, lung_data) -> None:
        y, lung = lung_data
        cols = ["age", "sex", "ph.ecog", "ph.karno", "wt.loss"]
        lung_with_na = gw.load_dataset("lung", backend="pandas")
        y_full = Surv.right(lung_with_na["time"], event=(lung_with_na["status"] == 2))
        model = IPCRidge(alpha=1.0).fit(y_full, lung_with_na[cols])
        assert model.n_ > 0
        assert model.n_ <= len(lung_with_na)
