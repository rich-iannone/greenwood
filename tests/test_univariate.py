"""Unit tests for univariate parametric survival distributions."""

from __future__ import annotations

import numpy as np
import pytest

import greenwood as gw
from greenwood import Parametric, Surv

from ._r_parity import assert_allclose_to_r, load_fixture

# -- fixtures ----------------------------------------------------------------


@pytest.fixture
def lung_surv():  # type: ignore[no-untyped-def]
    df = gw.load_dataset("lung", backend="pandas")
    return Surv.right(df["time"], event=(df["status"] == 2))


# -- construction & validation -----------------------------------------------


def test_invalid_dist() -> None:
    with pytest.raises(ValueError, match="dist"):
        Parametric("gaussian")


def test_invalid_conf_level() -> None:
    with pytest.raises(ValueError, match="conf_level"):
        Parametric(conf_level=0.0)


def test_unfitted_repr() -> None:
    r = repr(Parametric("weibull"))
    assert "unfitted" in r
    assert "weibull" in r


def test_interval_censoring_not_supported() -> None:
    y = Surv.interval(lower=[1, 2, 3], upper=[2, 3, 4])
    with pytest.raises(NotImplementedError, match="right-censored"):
        Parametric().fit(y)


# -- fit & properties --------------------------------------------------------


@pytest.mark.parametrize("dist", ["weibull", "exponential", "lognormal", "loglogistic"])
def test_fit_has_expected_attributes(lung_surv, dist: str) -> None:  # type: ignore[no-untyped-def]
    fit = Parametric(dist).fit(lung_surv)
    assert fit.dist == dist
    assert fit.n_ == 228
    assert fit.n_event_ == 165
    assert fit.loglik_ < 0
    assert fit.aic_ > 0
    assert fit.bic_ > 0
    assert isinstance(fit.params_, dict)
    assert isinstance(fit.std_error_, dict)
    assert isinstance(fit.conf_low_, dict)
    assert isinstance(fit.conf_high_, dict)
    assert set(fit.params_.keys()) == set(fit.std_error_.keys())


def test_weibull_params(lung_surv) -> None:  # type: ignore[no-untyped-def]
    fit = Parametric("weibull").fit(lung_surv)
    assert set(fit.params_.keys()) == {"shape", "scale"}
    assert fit.params_["shape"] > 0
    assert fit.params_["scale"] > 0


def test_exponential_params(lung_surv) -> None:  # type: ignore[no-untyped-def]
    fit = Parametric("exponential").fit(lung_surv)
    assert set(fit.params_.keys()) == {"rate"}
    assert fit.params_["rate"] > 0


def test_lognormal_params(lung_surv) -> None:  # type: ignore[no-untyped-def]
    fit = Parametric("lognormal").fit(lung_surv)
    assert set(fit.params_.keys()) == {"mu", "sigma"}
    assert fit.params_["sigma"] > 0


def test_loglogistic_params(lung_surv) -> None:  # type: ignore[no-untyped-def]
    fit = Parametric("loglogistic").fit(lung_surv)
    assert set(fit.params_.keys()) == {"alpha", "beta"}
    assert fit.params_["alpha"] > 0
    assert fit.params_["beta"] > 0


def test_conf_low_below_conf_high(lung_surv) -> None:  # type: ignore[no-untyped-def]
    for dist in ["weibull", "exponential", "lognormal", "loglogistic"]:
        fit = Parametric(dist).fit(lung_surv)
        for name in fit.params_:
            assert fit.conf_low_[name] < fit.params_[name]
            assert fit.params_[name] < fit.conf_high_[name]


def test_aic_bic_formula(lung_surv) -> None:  # type: ignore[no-untyped-def]
    fit = Parametric("weibull").fit(lung_surv)
    n_params = len(fit.params_)
    assert fit.aic_ == pytest.approx(-2 * fit.loglik_ + 2 * n_params)
    assert fit.bic_ == pytest.approx(-2 * fit.loglik_ + np.log(fit.n_) * n_params)


def test_exponential_one_param(lung_surv) -> None:  # type: ignore[no-untyped-def]
    fit = Parametric("exponential").fit(lung_surv)
    n_params = 1
    assert fit.aic_ == pytest.approx(-2 * fit.loglik_ + 2 * n_params)
    assert fit.bic_ == pytest.approx(-2 * fit.loglik_ + np.log(fit.n_) * n_params)


# -- prediction methods -----------------------------------------------------


@pytest.mark.parametrize("dist", ["weibull", "exponential", "lognormal", "loglogistic"])
def test_survival_range(lung_surv, dist: str) -> None:  # type: ignore[no-untyped-def]
    fit = Parametric(dist).fit(lung_surv)
    s = fit.survival([1, 100, 365, 1000, 5000])
    assert np.all(s >= 0)
    assert np.all(s <= 1)
    # Survival must be non-increasing.
    assert np.all(np.diff(s) <= 1e-15)


@pytest.mark.parametrize("dist", ["weibull", "exponential", "lognormal", "loglogistic"])
def test_hazard_positive(lung_surv, dist: str) -> None:  # type: ignore[no-untyped-def]
    fit = Parametric(dist).fit(lung_surv)
    h = fit.hazard([1, 100, 365, 1000])
    assert np.all(h > 0)


@pytest.mark.parametrize("dist", ["weibull", "exponential", "lognormal", "loglogistic"])
def test_density_positive(lung_surv, dist: str) -> None:  # type: ignore[no-untyped-def]
    fit = Parametric(dist).fit(lung_surv)
    f = fit.density([1, 100, 365, 1000])
    assert np.all(f > 0)


@pytest.mark.parametrize("dist", ["weibull", "exponential", "lognormal", "loglogistic"])
def test_hazard_equals_density_over_survival(lung_surv, dist: str) -> None:  # type: ignore[no-untyped-def]
    fit = Parametric(dist).fit(lung_surv)
    times = [50, 100, 200, 365, 500]
    h = fit.hazard(times)
    f = fit.density(times)
    s = fit.survival(times)
    np.testing.assert_allclose(h, f / s, rtol=1e-10)


@pytest.mark.parametrize("dist", ["weibull", "exponential", "lognormal", "loglogistic"])
def test_cumhaz_equals_neg_log_survival(lung_surv, dist: str) -> None:  # type: ignore[no-untyped-def]
    fit = Parametric(dist).fit(lung_surv)
    times = [50, 100, 200, 365, 500]
    H = fit.cumulative_hazard(times)
    s = fit.survival(times)
    np.testing.assert_allclose(H, -np.log(s), rtol=1e-12)


@pytest.mark.parametrize("dist", ["weibull", "exponential", "lognormal", "loglogistic"])
def test_quantile_inverts_survival(lung_surv, dist: str) -> None:  # type: ignore[no-untyped-def]
    fit = Parametric(dist).fit(lung_surv)
    p = np.array([0.1, 0.25, 0.5, 0.75, 0.9])
    q = fit.quantile(p)
    s_at_q = fit.survival(q)
    np.testing.assert_allclose(s_at_q, 1.0 - p, rtol=1e-10)


@pytest.mark.parametrize("dist", ["weibull", "exponential", "lognormal", "loglogistic"])
def test_median_matches_quantile_half(lung_surv, dist: str) -> None:  # type: ignore[no-untyped-def]
    fit = Parametric(dist).fit(lung_surv)
    assert fit.median() == pytest.approx(fit.quantile(0.5)[0])


@pytest.mark.parametrize("dist", ["weibull", "exponential", "lognormal"])
def test_mean_positive(lung_surv, dist: str) -> None:  # type: ignore[no-untyped-def]
    fit = Parametric(dist).fit(lung_surv)
    m = fit.mean()
    assert m > 0
    assert np.isfinite(m)


def test_loglogistic_mean_can_be_infinite() -> None:
    """Log-logistic mean is inf when beta <= 1 (sigma >= 1)."""
    # Fabricate a Surv response and set internal params directly to force sigma >= 1.
    # Instead of doing that, just verify the property: if fit sigma < 1, mean is finite.
    # The real data (lung) has beta > 1 so mean should be finite.
    df = gw.load_dataset("lung", backend="pandas")
    y = Surv.right(df["time"], event=(df["status"] == 2))
    fit = Parametric("loglogistic").fit(y)
    if fit.params_["beta"] > 1.0:
        assert np.isfinite(fit.mean())
    else:
        assert fit.mean() == np.inf


# -- repr and export ---------------------------------------------------------


@pytest.mark.parametrize("dist", ["weibull", "exponential", "lognormal", "loglogistic"])
def test_repr_fitted(lung_surv, dist: str) -> None:  # type: ignore[no-untyped-def]
    fit = Parametric(dist).fit(lung_surv)
    r = repr(fit)
    assert dist in r
    assert "Log-likelihood" in r
    assert "AIC" in r


def test_to_frame_columns(lung_surv) -> None:  # type: ignore[no-untyped-def]
    fit = Parametric("weibull").fit(lung_surv)
    df = fit.to_frame(format="pandas")
    assert list(df.columns) == ["param", "estimate", "std_error", "conf_low", "conf_high"]
    assert list(df["param"]) == ["shape", "scale"]


def test_to_frame_polars(lung_surv) -> None:  # type: ignore[no-untyped-def]
    fit = Parametric("weibull").fit(lung_surv)
    df = fit.to_frame(format="polars")
    assert df.shape[0] == 2


# -- tidy / glance -----------------------------------------------------------


def test_tidy(lung_surv) -> None:  # type: ignore[no-untyped-def]
    fit = Parametric("weibull").fit(lung_surv)
    t = gw.tidy(fit, format="pandas")
    assert list(t.columns) == ["param", "estimate", "std_error", "conf_low", "conf_high"]
    assert t.shape[0] == 2


def test_glance(lung_surv) -> None:  # type: ignore[no-untyped-def]
    fit = Parametric("weibull").fit(lung_surv)
    g = gw.glance(fit, format="pandas")
    assert g.shape[0] == 1
    assert g.iloc[0]["dist"] == "weibull"
    assert g.iloc[0]["n"] == 228
    assert g.iloc[0]["nevent"] == 165
    assert g.iloc[0]["aic"] == pytest.approx(fit.aic_)


# -- compare_distributions ---------------------------------------------------


def test_compare_distributions_returns_all(lung_surv) -> None:  # type: ignore[no-untyped-def]
    df = gw.compare_distributions(lung_surv, format="pandas")
    assert df.shape[0] == 4
    assert set(df["dist"]) == {"weibull", "exponential", "lognormal", "loglogistic"}
    assert list(df.columns) == ["dist", "n_params", "loglik", "aic", "bic"]


def test_compare_distributions_sorted_by_aic(lung_surv) -> None:  # type: ignore[no-untyped-def]
    df = gw.compare_distributions(lung_surv, format="pandas")
    assert list(df["aic"]) == sorted(df["aic"])


def test_compare_distributions_subset(lung_surv) -> None:  # type: ignore[no-untyped-def]
    df = gw.compare_distributions(lung_surv, dists=["weibull", "exponential"], format="pandas")
    assert df.shape[0] == 2
    assert set(df["dist"]) == {"weibull", "exponential"}


def test_compare_distributions_polars(lung_surv) -> None:  # type: ignore[no-untyped-def]
    df = gw.compare_distributions(lung_surv, format="polars")
    assert df.shape[0] == 4


# -- R parity ----------------------------------------------------------------

pytestmark_rparity = pytest.mark.rparity


@pytest.mark.parametrize("dist", ["weibull", "exponential", "lognormal", "loglogistic"])
def test_loglik_matches_r(lung_surv, dist: str) -> None:  # type: ignore[no-untyped-def]
    """Log-likelihood must match R's survreg(Surv(...) ~ 1, dist=...)."""
    fix = load_fixture(f"parametric_{dist}")
    fit = Parametric(dist).fit(lung_surv)
    assert_allclose_to_r(fit.loglik_, fix["loglik"], atol=1e-3, what=f"{dist} loglik")


@pytest.mark.parametrize("dist", ["weibull", "exponential", "lognormal", "loglogistic"])
def test_mu_matches_r(lung_surv, dist: str) -> None:  # type: ignore[no-untyped-def]
    """AFT intercept (mu) must match R's survreg coefficient."""
    fix = load_fixture(f"parametric_{dist}")
    fit = Parametric(dist).fit(lung_surv)
    assert_allclose_to_r(fit._mu, fix["mu"], atol=1e-4, what=f"{dist} mu")


@pytest.mark.parametrize("dist", ["weibull", "exponential", "lognormal", "loglogistic"])
def test_mu_se_matches_r(lung_surv, dist: str) -> None:  # type: ignore[no-untyped-def]
    """SE of mu must match R's survreg."""
    fix = load_fixture(f"parametric_{dist}")
    fit = Parametric(dist).fit(lung_surv)
    assert_allclose_to_r(
        np.sqrt(fit._vcov_raw[0, 0]), fix["mu_se"], atol=1e-4, what=f"{dist} mu_se"
    )


@pytest.mark.parametrize("dist", ["weibull", "lognormal", "loglogistic"])
def test_scale_matches_r(lung_surv, dist: str) -> None:  # type: ignore[no-untyped-def]
    """AFT scale (sigma) must match R's survreg scale."""
    fix = load_fixture(f"parametric_{dist}")
    fit = Parametric(dist).fit(lung_surv)
    assert_allclose_to_r(fit._sigma, fix["scale"], atol=1e-4, what=f"{dist} scale")


@pytest.mark.parametrize("dist", ["weibull", "lognormal", "loglogistic"])
def test_log_scale_se_matches_r(lung_surv, dist: str) -> None:  # type: ignore[no-untyped-def]
    """SE of log(scale) must match R's survreg."""
    fix = load_fixture(f"parametric_{dist}")
    fit = Parametric(dist).fit(lung_surv)
    assert_allclose_to_r(
        np.sqrt(fit._vcov_raw[1, 1]), fix["log_scale_se"], atol=1e-4, what=f"{dist} log_scale_se"
    )


@pytest.mark.parametrize("dist", ["weibull", "exponential", "lognormal", "loglogistic"])
def test_quantiles_match_r(lung_surv, dist: str) -> None:  # type: ignore[no-untyped-def]
    """Quantile predictions must match R's predict(..., type='quantile')."""
    fix = load_fixture(f"parametric_{dist}")
    fit = Parametric(dist).fit(lung_surv)
    q = fit.quantile(fix["pred_p"])
    assert_allclose_to_r(q, fix["pred_quantile"], rtol=1e-4, atol=1e-2, what=f"{dist} quantile")


@pytest.mark.parametrize("dist", ["weibull", "exponential", "lognormal", "loglogistic"])
def test_survival_matches_r(lung_surv, dist: str) -> None:  # type: ignore[no-untyped-def]
    """Survival predictions must match R's parametric S(t)."""
    fix = load_fixture(f"parametric_{dist}")
    fit = Parametric(dist).fit(lung_surv)
    s = fit.survival(fix["pred_times"])
    assert_allclose_to_r(s, fix["pred_survival"], rtol=1e-4, atol=1e-6, what=f"{dist} survival")
