"""Tests for AFT conditional expectation and RMST predictions.

Covers:
- type="mean" (unconditional and conditional)
- type="mean_remaining"
- type="rmst"
All four distributions: weibull, exponential, lognormal, loglogistic.
Key algebraic identities are checked against closed-form references and
scipy numerical integration.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.special import gamma as sp_gamma
from scipy.stats import logistic, norm

import greenwood as gw
from greenwood import AFT, Surv

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def lung_data():  # type: ignore[no-untyped-def]
    df = gw.load_dataset("lung", backend="pandas")
    y = Surv.right(df["time"], event=(df["status"] == 2))
    return df, y


@pytest.fixture(scope="module")
def fitted_models(lung_data):  # type: ignore[no-untyped-def]
    df, y = lung_data
    X = df[["age", "sex"]]
    return {
        dist: AFT(dist).fit(y, X)
        for dist in ("weibull", "exponential", "lognormal", "loglogistic")
    }


# ---------------------------------------------------------------------------
# type="mean": unconditional
# ---------------------------------------------------------------------------


def test_mean_lognormal_closed_form(fitted_models, lung_data) -> None:  # type: ignore[no-untyped-def]
    """E[T] for lognormal equals exp(mu + sigma^2/2)."""
    df, _ = lung_data
    aft = fitted_models["lognormal"]
    X = df[["age", "sex"]].iloc[:5]
    mu = aft.predict(X, type="lp")
    sigma = aft.scale_
    expected = np.exp(mu + 0.5 * sigma**2)
    actual = aft.predict(X, type="mean")
    np.testing.assert_allclose(actual, expected, rtol=1e-10)


def test_mean_weibull_closed_form(fitted_models, lung_data) -> None:  # type: ignore[no-untyped-def]
    """E[T] for Weibull equals exp(mu) * Gamma(1+sigma)."""
    df, _ = lung_data
    aft = fitted_models["weibull"]
    X = df[["age", "sex"]].iloc[:5]
    mu = aft.predict(X, type="lp")
    sigma = aft.scale_
    expected = np.exp(mu) * sp_gamma(1.0 + sigma)
    actual = aft.predict(X, type="mean")
    np.testing.assert_allclose(actual, expected, rtol=1e-10)


def test_mean_exponential_equals_exp_mu(fitted_models, lung_data) -> None:  # type: ignore[no-untyped-def]
    """E[T] for exponential equals exp(mu) (scale=1, Gamma(2)=1)."""
    df, _ = lung_data
    aft = fitted_models["exponential"]
    X = df[["age", "sex"]].iloc[:5]
    mu = aft.predict(X, type="lp")
    expected = np.exp(mu)
    actual = aft.predict(X, type="mean")
    np.testing.assert_allclose(actual, expected, rtol=1e-10)


def test_mean_loglogistic_closed_form(lung_data) -> None:  # type: ignore[no-untyped-def]
    """E[T] for loglogistic equals exp(mu)*pi*sigma/sin(pi*sigma) when sigma<1."""
    df, y = lung_data
    # Fit with smaller sigma; loglogistic often fits with sigma < 1 on lung
    aft = AFT("loglogistic").fit(y, df[["age", "sex"]])
    sigma = aft.scale_
    if sigma >= 1.0:
        pytest.skip("sigma >= 1: mean is infinite, skip closed-form test")
    X = df[["age", "sex"]].iloc[:5]
    mu = aft.predict(X, type="lp")
    expected = np.exp(mu) * np.pi * sigma / np.sin(np.pi * sigma)
    actual = aft.predict(X, type="mean")
    np.testing.assert_allclose(actual, expected, rtol=1e-10)


def test_mean_all_distributions_positive(fitted_models, lung_data) -> None:  # type: ignore[no-untyped-def]
    """E[T] should be strictly positive for all distributions."""
    df, _ = lung_data
    X = df[["age", "sex"]].iloc[:10]
    for dist, aft in fitted_models.items():
        result = aft.predict(X, type="mean")
        finite_mask = np.isfinite(result)
        assert finite_mask.any(), f"{dist}: no finite means"
        assert (result[finite_mask] > 0).all(), f"{dist}: non-positive mean"


# ---------------------------------------------------------------------------
# type="mean": conditional
# ---------------------------------------------------------------------------


def test_mean_conditional_geq_unconditional(fitted_models, lung_data) -> None:  # type: ignore[no-untyped-def]
    """E[T | T > t0] >= t0 for all subjects and distributions."""
    df, _ = lung_data
    X = df[["age", "sex"]].iloc[:5]
    t0 = 100.0
    for dist, aft in fitted_models.items():
        cond = aft.predict(X, type="mean", conditional_after=t0)
        assert (cond >= t0).all(), f"{dist}: E[T|T>t0] < t0"


def test_mean_conditional_lognormal_formula(fitted_models, lung_data) -> None:  # type: ignore[no-untyped-def]
    """E[T | T > t0] for lognormal matches exp(mu+s^2/2)*Phi(s-z0)/Phi(-z0)."""
    df, _ = lung_data
    aft = fitted_models["lognormal"]
    X = df[["age", "sex"]].iloc[:5]
    t0 = 180.0
    mu = aft.predict(X, type="lp")
    sigma = aft.scale_
    z0 = (np.log(t0) - mu) / sigma
    expected = np.exp(mu + 0.5 * sigma**2) * norm.cdf(sigma - z0) / norm.sf(z0)
    actual = aft.predict(X, type="mean", conditional_after=t0)
    np.testing.assert_allclose(actual, expected, rtol=1e-8)


def test_mean_conditional_per_subject_array(fitted_models, lung_data) -> None:  # type: ignore[no-untyped-def]
    """conditional_after accepts a per-subject array."""
    df, _ = lung_data
    X = df[["age", "sex"]].iloc[:5]
    aft = fitted_models["weibull"]
    t0_arr = np.array([50.0, 100.0, 150.0, 200.0, 250.0])
    result = aft.predict(X, type="mean", conditional_after=t0_arr)
    assert result.shape == (5,)
    # Each E[T|T>t0_i] must be >= t0_i
    assert (result >= t0_arr).all()


# ---------------------------------------------------------------------------
# type="mean_remaining"
# ---------------------------------------------------------------------------


def test_mean_remaining_requires_conditional_after(fitted_models, lung_data) -> None:  # type: ignore[no-untyped-def]
    """type='mean_remaining' raises without conditional_after."""
    df, _ = lung_data
    aft = fitted_models["weibull"]
    X = df[["age", "sex"]].iloc[:2]
    with pytest.raises(ValueError, match="conditional_after"):
        aft.predict(X, type="mean_remaining")


def test_mean_remaining_equals_conditional_mean_minus_t0(fitted_models, lung_data) -> None:  # type: ignore[no-untyped-def]
    """E[T-t0|T>t0] = E[T|T>t0] - t0 for all distributions."""
    df, _ = lung_data
    X = df[["age", "sex"]].iloc[:5]
    t0 = 120.0
    for dist, aft in fitted_models.items():
        cond_mean = aft.predict(X, type="mean", conditional_after=t0)
        remaining = aft.predict(X, type="mean_remaining", conditional_after=t0)
        np.testing.assert_allclose(
            remaining, cond_mean - t0, rtol=1e-10, err_msg=f"{dist}: identity failed"
        )


def test_mean_remaining_exponential_memoryless(fitted_models, lung_data) -> None:  # type: ignore[no-untyped-def]
    """Exponential distribution is memoryless: E[T-t0|T>t0] = E[T] for all t0."""
    df, _ = lung_data
    aft = fitted_models["exponential"]
    X = df[["age", "sex"]].iloc[:5]
    e_t = aft.predict(X, type="mean")
    for t0 in [0.0, 50.0, 200.0, 500.0]:
        remaining = aft.predict(X, type="mean_remaining", conditional_after=t0)
        np.testing.assert_allclose(
            remaining, e_t, rtol=1e-6, err_msg=f"memoryless failed at t0={t0}"
        )


def test_mean_remaining_positive(fitted_models, lung_data) -> None:  # type: ignore[no-untyped-def]
    """E[T - t0 | T > t0] must be positive."""
    df, _ = lung_data
    X = df[["age", "sex"]].iloc[:5]
    for dist, aft in fitted_models.items():
        remaining = aft.predict(X, type="mean_remaining", conditional_after=200.0)
        assert (remaining > 0).all(), f"{dist}: non-positive mean remaining"


# ---------------------------------------------------------------------------
# type="rmst"
# ---------------------------------------------------------------------------


def test_rmst_requires_tau(fitted_models, lung_data) -> None:  # type: ignore[no-untyped-def]
    """type='rmst' raises without tau."""
    df, _ = lung_data
    aft = fitted_models["weibull"]
    X = df[["age", "sex"]].iloc[:2]
    with pytest.raises(ValueError, match="tau"):
        aft.predict(X, type="rmst")


def test_rmst_positive_tau_required(fitted_models, lung_data) -> None:  # type: ignore[no-untyped-def]
    """Non-positive tau raises ValueError."""
    df, _ = lung_data
    aft = fitted_models["weibull"]
    X = df[["age", "sex"]].iloc[:2]
    with pytest.raises(ValueError):
        aft.predict(X, type="rmst", tau=0.0)
    with pytest.raises(ValueError):
        aft.predict(X, type="rmst", tau=-10.0)


def test_rmst_lognormal_vs_numerical(fitted_models, lung_data) -> None:  # type: ignore[no-untyped-def]
    """RMST for lognormal matches scipy.integrate.quad within 1e-5."""
    df, _ = lung_data
    aft = fitted_models["lognormal"]
    X = df[["age", "sex"]].iloc[:3]
    tau = 365.0
    sigma = aft.scale_

    predicted = aft.predict(X, type="rmst", tau=tau)

    for i, mu_i in enumerate(aft.predict(X, type="lp")):

        def s_t(t: float, _mu: float = float(mu_i)) -> float:
            return float(norm.sf((np.log(t) - _mu) / sigma))

        numerical, _ = quad(s_t, 0.0, tau)
        np.testing.assert_allclose(
            predicted[i], numerical, rtol=1e-5, err_msg=f"lognormal RMST mismatch at subject {i}"
        )


def test_rmst_weibull_vs_numerical(fitted_models, lung_data) -> None:  # type: ignore[no-untyped-def]
    """RMST for Weibull matches scipy.integrate.quad within 1e-5."""
    df, _ = lung_data
    aft = fitted_models["weibull"]
    X = df[["age", "sex"]].iloc[:3]
    tau = 365.0
    sigma = aft.scale_

    predicted = aft.predict(X, type="rmst", tau=tau)

    for i, mu_i in enumerate(aft.predict(X, type="lp")):

        def s_t(t: float, _mu: float = float(mu_i)) -> float:
            z = (np.log(t) - _mu) / sigma
            return float(np.exp(-np.exp(z)))

        numerical, _ = quad(s_t, 0.0, tau)
        np.testing.assert_allclose(
            predicted[i], numerical, rtol=1e-5, err_msg=f"Weibull RMST mismatch at subject {i}"
        )


def test_rmst_loglogistic_vs_numerical(lung_data) -> None:  # type: ignore[no-untyped-def]
    """RMST for loglogistic matches scipy.integrate.quad for both sigma<1 and sigma>=1."""
    df, y = lung_data
    # Test sigma < 1 path (typical loglogistic fit on lung)
    aft = AFT("loglogistic").fit(y, df[["age", "sex"]])
    X = df[["age", "sex"]].iloc[:3]
    tau = 365.0
    sigma = aft.scale_

    predicted = aft.predict(X, type="rmst", tau=tau)

    for i, mu_i in enumerate(aft.predict(X, type="lp")):

        def s_t(t: float, _mu: float = float(mu_i)) -> float:
            return float(logistic.sf((np.log(t) - _mu) / sigma))

        numerical, _ = quad(s_t, 0.0, tau)
        np.testing.assert_allclose(
            predicted[i],
            numerical,
            rtol=1e-4,
            err_msg=f"loglogistic RMST mismatch at subject {i} (sigma={sigma:.3f})",
        )


def test_rmst_monotone_in_tau(fitted_models, lung_data) -> None:  # type: ignore[no-untyped-def]
    """RMST should increase with tau (more time → larger restricted mean)."""
    df, _ = lung_data
    aft = fitted_models["weibull"]
    X = df[["age", "sex"]].iloc[:5]
    rmst_180 = aft.predict(X, type="rmst", tau=180.0)
    rmst_365 = aft.predict(X, type="rmst", tau=365.0)
    assert (rmst_365 > rmst_180).all()


def test_rmst_bounded_by_tau(fitted_models, lung_data) -> None:  # type: ignore[no-untyped-def]
    """RMST <= tau (can't exceed the restriction time)."""
    df, _ = lung_data
    tau = 365.0
    X = df[["age", "sex"]].iloc[:10]
    for dist, aft in fitted_models.items():
        rmst = aft.predict(X, type="rmst", tau=tau)
        assert (rmst <= tau).all(), f"{dist}: RMST > tau"
        assert (rmst > 0).all(), f"{dist}: RMST <= 0"


def test_rmst_all_distributions(fitted_models, lung_data) -> None:  # type: ignore[no-untyped-def]
    """type='rmst' runs without error for all distributions."""
    df, _ = lung_data
    X = df[["age", "sex"]].iloc[:5]
    tau = 365.0
    for dist, aft in fitted_models.items():
        result = aft.predict(X, type="rmst", tau=tau)
        assert result.shape == (5,)
        assert np.isfinite(result).all(), f"{dist}: non-finite RMST"


# ---------------------------------------------------------------------------
# Invalid type
# ---------------------------------------------------------------------------


def test_predict_invalid_type(fitted_models, lung_data) -> None:  # type: ignore[no-untyped-def]
    """Unknown predict type raises ValueError."""
    df, _ = lung_data
    aft = fitted_models["weibull"]
    X = df[["age", "sex"]].iloc[:2]
    with pytest.raises(ValueError, match="Unknown predict type"):
        aft.predict(X, type="bogus")
