r"""Parametric accelerated failure time (AFT) models.

`AFT` fits the log-linear (accelerated failure time) model $\log(T) = X\beta + \sigma W$, where the
error $W$ follows a standard distribution chosen by `dist`:

- `"weibull"` and `"exponential"` (exponential fixes $\sigma = 1$): minimum extreme value,
- `"lognormal"`: standard normal,
- `"loglogistic"`: standard logistic,
- `"gengamma"`: Prentice's generalized gamma, with a shape `Q` that nests the Weibull (`Q = 1`) and
  the log-normal (`Q = 0`).

Coefficients are on the log-time scale (with an intercept), matching R's `survreg`. The model is fit
by maximum likelihood for right-censored data, and the coefficients, scale, standard errors, and
log-likelihood are validated to tolerance against `survreg` (and the generalized gamma against
flexsurv's likelihood).
"""

from __future__ import annotations

import warnings
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt
from scipy.optimize import minimize
from scipy.special import betainc as _sp_betainc
from scipy.special import gamma as _sp_gamma
from scipy.special import gammainc as _sp_gammainc
from scipy.special import gammaincc as _sp_gammaincc
from scipy.special import gammaln as _sp_gammaln
from scipy.stats import logistic, norm

from ._backends import to_dataframe
from ._cox import _design_matrix_spec
from ._outcome import bind_fit_inputs
from ._repr import dropped_note

if TYPE_CHECKING:
    from ._outcome import Outcome
    from ._surv import Surv

__all__ = ["AFT"]

Array = npt.NDArray[Any]
# A log-likelihood with its gradient and Hessian, as a function of the parameters.
_Terms = Callable[[Array], tuple[float, Array, Array]]

_DISTS = frozenset({"weibull", "exponential", "lognormal", "loglogistic", "gengamma"})


def _log_density_survival(dist: str, z: Array, *, Q: float = 0.0) -> tuple[Array, Array]:
    """Standardized log-density and log-survival of the error term at `z`."""
    if dist in ("weibull", "exponential"):  # minimum extreme value
        return z - np.exp(z), -np.exp(z)
    if dist == "lognormal":
        return norm.logpdf(z), norm.logsf(z)
    if dist == "gengamma":
        return _gengamma_log_density_survival(z, Q)
    # loglogistic
    return logistic.logpdf(z), logistic.logsf(z)


def _cdf(dist: str, z: Array, *, Q: float = 0.0) -> Array:
    """Standardized CDF of the error term at `z`, derived from the log-survival.

    Used by the maximum-product-of-spacings objective, which needs $F(z) = 1 - S(z)$ rather
    than $S(z)$ itself. `-expm1(log_s)` is numerically stable for `log_s` close to 0 (i.e. `S`
    close to 1, where a naive `1 - exp(log_s)` would lose precision to cancellation).
    """
    _, log_s = _log_density_survival(dist, z, Q=Q)
    return -np.expm1(log_s)


def _dlogf_dz(dist: str, z: Array, *, Q: float = 0.0) -> Array:
    """Derivative of the standardized log-density w.r.t. `z`."""
    if dist in ("weibull", "exponential"):
        return 1.0 - np.exp(z)
    if dist == "lognormal":
        return -z
    if dist == "gengamma":
        return _gengamma_slope(z, Q)
    # loglogistic
    return 1.0 - 2.0 * logistic.cdf(z)


def _gengamma_log_density(z: Array, Q: float) -> Array:
    r"""Log-density of the generalized gamma error at standardized `z`.

    In Prentice's parameterization, with $k = Q^{-2}$ and $u = k e^{Qz}$, the log-density is
    $\log|Q| + k \log k - \log\Gamma(k) + kQz - u$. It is computed as a constant minus
    $(e^{Qz} - 1 - Qz) / Q^2$, with both parts kept accurate as $Q$ nears 0. $Q = 0$ is the
    standard normal limit.
    """
    if Q == 0.0:
        return norm.logpdf(z)
    k = Q ** (-2.0)
    if k > 1e4:
        # Stirling's series for k log k - k - log Gamma(k) + log|Q|.
        constant = -0.5 * np.log(2.0 * np.pi) - 1.0 / (12.0 * k) + 1.0 / (360.0 * k**3)
    else:
        constant = np.log(abs(Q)) + k * np.log(k) - k - float(_sp_gammaln(k))
    a = Q * z
    with np.errstate(over="ignore"):
        excess = np.where(
            np.abs(a) < 1e-3,
            z**2 * (0.5 + a / 6.0 + a**2 / 24.0 + a**3 / 120.0),
            (np.expm1(a) - a) / Q**2,
        )
    return constant - excess


def _gengamma_log_survival(z: Array, Q: float) -> Array:
    """Log-survival of the generalized gamma error at standardized `z`.

    The survival is the upper regularized incomplete gamma function at $(k, u)$ for $Q > 0$ and
    the lower one for $Q < 0$.
    """
    if Q == 0.0:
        return norm.logsf(z)
    k = Q ** (-2.0)
    with np.errstate(over="ignore"):
        u = np.exp(Q * z + np.log(k))
    tail = _sp_gammaincc(k, u) if Q > 0 else _sp_gammainc(k, u)
    with np.errstate(divide="ignore"):
        return np.log(np.clip(tail, 1e-300, None))


def _gengamma_slope(z: Array, Q: float) -> Array:
    r"""Derivative of the generalized gamma log-density in `z`, $-(e^{Qz} - 1) / Q$."""
    if Q == 0.0:
        return -np.asarray(z, dtype=float)
    with np.errstate(over="ignore"):
        return -np.expm1(Q * z) / Q


def _gengamma_log_density_survival(z: Array, Q: float) -> tuple[Array, Array]:
    """Log-density and log-survival for the generalized gamma error at standardized `z`."""
    return _gengamma_log_density(z, Q), _gengamma_log_survival(z, Q)


def _gengamma_log_moment(k: float, shift: float) -> float:
    r"""$\log\Gamma(k + c) - \log\Gamma(k) - c \log k$ for $c = \sigma/Q$.

    This is the log of $E[e^{\sigma W}]$. For a large `k`, the difference of Stirling's series
    avoids the cancellation of the direct form.
    """
    if min(k, k + shift) < 1e5:
        return float(_sp_gammaln(k + shift) - _sp_gammaln(k) - shift * np.log(k))

    def correction(v: float) -> float:
        return 1.0 / (12.0 * v) - 1.0 / (360.0 * v**3)

    return float(
        (k + shift - 0.5) * np.log1p(shift / k) - shift + correction(k + shift) - correction(k)
    )


def _error_pieces(
    dist: str, z: Array, event: Array, *, Q: float = 0.0
) -> tuple[Array, Array, Array]:
    """Each row's log-likelihood term on the `z` scale, with its first two derivatives in `z`.

    Events contribute the error's log-density and censored rows its log-survival.
    """
    is_event = event == 1
    if dist in ("weibull", "exponential"):
        e = np.exp(z)
        return np.where(is_event, z - e, -e), np.where(is_event, 1.0 - e, -e), -e
    if dist == "loglogistic":
        f = logistic.cdf(z)
        g = np.where(is_event, logistic.logpdf(z), logistic.logsf(z))
        d1 = np.where(is_event, 1.0 - 2.0 * f, -f)
        d2 = np.where(is_event, -2.0 * f * (1.0 - f), -f * (1.0 - f))
        return g, d1, d2
    if dist == "lognormal":
        log_f, log_s, slope, curve = norm.logpdf(z), norm.logsf(z), -z, -np.ones_like(z)
    else:
        log_f, log_s = _gengamma_log_density_survival(z, Q)
        slope = _gengamma_slope(z, Q)
        with np.errstate(over="ignore"):
            curve = -np.exp(Q * z)
    with np.errstate(over="ignore", invalid="ignore"):
        h = np.exp(log_f - log_s)
        d2 = np.where(is_event, curve, -h * (h + slope))
    g = np.where(is_event, log_f, log_s)
    d1 = np.where(is_event, slope, -h)
    return g, d1, d2


def _aft_terms(
    theta: Array, x: Array, log_t: Array, event: Array, dist: str, *, Q: float = 0.0
) -> tuple[float, Array, Array]:
    """Log-likelihood of the AFT model with its gradient and Hessian in `theta`.

    `theta` is `(beta, log sigma)`, or `beta` alone for the exponential. The density is on the
    time scale, so events carry the Jacobian `-log(sigma) - log(t)`.
    """
    p = x.shape[1]
    fixed_scale = dist == "exponential"
    s = 0.0 if fixed_scale else float(theta[p])
    sigma = np.exp(s)
    z = (log_t - x @ theta[:p]) / sigma
    g, d1, d2 = _error_pieces(dist, z, event, Q=Q)
    loglik = float(np.sum(g - event * (s + log_t)))
    if not np.isfinite(loglik):
        return -np.inf, np.zeros(0), np.zeros((0, 0))
    gradient = x.T @ (-d1 / sigma)
    hessian = (x * (d2 / sigma**2)[:, None]).T @ x
    if not fixed_scale:
        h_bs = x.T @ ((d2 * z + d1) / sigma)
        h_ss = float(np.sum(d2 * z**2 + d1 * z))
        gradient = np.append(gradient, np.sum(-d1 * z - event))
        hessian = np.block([[hessian, h_bs[:, None]], [h_bs[None, :], np.array([[h_ss]])]])
    return loglik, gradient, hessian


def _gengamma_terms(
    theta: Array, x: Array, log_t: Array, event: Array
) -> tuple[float, Array, Array]:
    """Generalized gamma log-likelihood with its gradient and Hessian in `(beta, log sigma, Q)`.

    The derivatives in `(beta, log sigma)` are exact at each `Q`, and those in `Q` are five-point
    differences of them.
    """
    Q = float(theta[-1])
    rest = theta[:-1]
    mid = _aft_terms(rest, x, log_t, event, "gengamma", Q=Q)
    h = 1e-3
    sides = [_aft_terms(rest, x, log_t, event, "gengamma", Q=Q + d * h) for d in (-2, -1, 1, 2)]
    values = [side[0] for side in sides]
    if not np.isfinite(mid[0]) or not np.all(np.isfinite(values)):
        return -np.inf, np.zeros(0), np.zeros((0, 0))

    def stencil(v: list[Any]) -> Any:
        return (v[0] - 8.0 * v[1] + 8.0 * v[2] - v[3]) / (12.0 * h)

    d_q = stencil(values)
    d_qq = (-values[0] + 16.0 * values[1] - 30.0 * mid[0] + 16.0 * values[2] - values[3]) / (
        12.0 * h**2
    )
    d_q_other = stencil([side[1] for side in sides])
    gradient = np.append(mid[1], d_q)
    hessian = np.block([[mid[2], d_q_other[:, None]], [d_q_other[None, :], np.array([[d_qq]])]])
    return mid[0], gradient, hessian


def _newton_step(info: Array, gradient: Array) -> Array:
    """Solve `info @ step = gradient`, adding a ridge until `info` is positive definite."""
    ridge = 0.0
    scale = max(float(np.max(np.abs(np.diag(info)))), 1.0)
    eye = np.eye(info.shape[0])
    for _ in range(30):
        try:
            factor = np.linalg.cholesky(info + ridge * eye)
        except np.linalg.LinAlgError:
            ridge = 1e-8 * scale if ridge == 0.0 else ridge * 10.0
            continue
        return np.linalg.solve(factor.T, np.linalg.solve(factor, gradient))
    return gradient / scale


def _newton_maximize(
    theta: Array, terms: _Terms, *, max_iter: int = 50, tol: float = 1e-10
) -> tuple[Array, float, Array, bool]:
    """Newton-Raphson from `theta`, halving the step whenever it lowers the log-likelihood.

    `terms(theta)` gives the log-likelihood with its gradient and Hessian. Returns the estimates,
    the log-likelihood, the observed information, and whether the iterations converged.
    """
    current = terms(theta)
    if not np.isfinite(current[0]):
        raise ValueError("The log-likelihood isn't finite at the starting values.")
    converged = False
    for _ in range(max_iter):
        step = _newton_step(-current[2], current[1])
        candidate = terms(theta + step)
        halvings = 0
        while (
            not (np.isfinite(candidate[0]) and candidate[0] >= current[0] - 1e-12) and halvings < 30
        ):
            step = step / 2.0
            halvings += 1
            candidate = terms(theta + step)
        change = candidate[0] - current[0]
        theta = theta + step
        current = candidate
        if abs(change) <= tol * (abs(current[0]) + tol):
            converged = True
            break
    return theta, float(current[0]), -current[2], converged


def _aft_newton(
    x: Array, log_t: Array, event: Array, dist: str, *, Q: float = 0.0
) -> tuple[Array, float, Array, bool]:
    """Fit the AFT model by Newton-Raphson, starting from least squares on all the rows."""
    beta, *_ = np.linalg.lstsq(x, log_t, rcond=None)
    theta = np.asarray(beta, dtype=float)
    if dist != "exponential":
        spread = float(np.sqrt(np.mean((log_t - x @ theta) ** 2)))
        theta = np.append(theta, np.log(max(spread, 1e-3)))
    return _newton_maximize(theta, lambda th: _aft_terms(th, x, log_t, event, dist, Q=Q))


def _gengamma_newton(x: Array, log_t: Array, event: Array) -> tuple[Array, float, Array, bool]:
    """Fit the generalized gamma, with `theta = (beta, log sigma, Q)`.

    The start is the best of the fits over a grid of fixed values of `Q`, and Newton-Raphson then
    moves all the parameters together.
    """
    best: tuple[float, Array, float] | None = None
    for q in np.arange(-6, 7) * 0.5:
        try:
            theta_q, loglik_q, _, _ = _aft_newton(x, log_t, event, "gengamma", Q=float(q))
        except (ValueError, np.linalg.LinAlgError):
            continue
        if np.isfinite(loglik_q) and (best is None or loglik_q > best[0]):
            best = (loglik_q, theta_q, float(q))
    if best is None:
        raise ValueError("The generalized gamma likelihood isn't finite for any starting Q.")
    start = np.append(best[1], best[2])
    return _newton_maximize(start, lambda th: _gengamma_terms(th, x, log_t, event))


def _error_quantile(dist: str, p: Array, *, Q: float = 0.0) -> Array:
    r"""Standardized quantile of the error term `W` at probabilities `p`.

    `p` is the cumulative probability of failure, so the returned `w` satisfies $F_W(w) = p$. The
    corresponding time quantile is $\exp(\mu + \sigma w)$.
    """
    if dist in ("weibull", "exponential"):  # minimum extreme value
        return np.log(-np.log1p(-p))
    if dist == "lognormal":
        return norm.ppf(p)
    if dist == "gengamma":
        return _gengamma_quantile(p, Q)
    return logistic.ppf(p)  # loglogistic


def _gengamma_quantile(p: Array, Q: float) -> Array:
    """Quantile of the standardized generalized gamma error distribution."""
    from scipy.special import gammainccinv, gammaincinv

    if Q == 0.0:
        return norm.ppf(p)

    k = Q ** (-2.0)
    # F(w) = p is the lower tail of u = k exp(Q w) for Q > 0 and the upper tail for Q < 0.
    u = gammaincinv(k, p) if Q > 0 else gammainccinv(k, p)
    return np.log(u / k) / Q


def _mean_survival_aft(dist: str, mu: Array, sigma: float, *, Q: float = 0.0) -> Array:
    r"""Closed-form $E[T]$ for each subject under the AFT distribution.

    Returns `np.inf` for log-logistic when `sigma >= 1` (mean is undefined).
    """
    if dist in ("weibull", "exponential"):
        return np.exp(mu) * _sp_gamma(1.0 + sigma)
    if dist == "lognormal":
        return np.exp(mu + 0.5 * sigma**2)
    if dist == "gengamma":
        return _gengamma_mean(mu, sigma, Q)
    # loglogistic
    if sigma < 1.0:
        return np.exp(mu) * np.pi * sigma / np.sin(np.pi * sigma)
    return np.full_like(mu, np.inf)


def _gengamma_mean(mu: Array, sigma: float, Q: float) -> Array:
    r"""$E[T]$ for the generalized gamma distribution.

    With $c = \sigma / Q$, $E[T] = e^\mu \Gamma(k + c) / (k^c \Gamma(k))$ when $k + c > 0$. The
    mean is infinite otherwise.
    """
    if Q == 0.0:
        return np.exp(mu + 0.5 * sigma**2)
    k = Q ** (-2.0)
    shift = sigma / Q
    if k + shift <= 0:
        return np.full_like(mu, np.inf)
    return np.exp(mu + _gengamma_log_moment(k, shift))


def _gengamma_partial_moment(mu: Array, sigma: float, Q: float, t0: Array, *, upper: bool) -> Array:
    r"""$E[T \, 1\{T > t_0\}]$ (`upper=True`) or $E[T \, 1\{T \le t_0\}]$ for the generalized gamma.

    With $c = \sigma / Q$, $T = e^\mu (U/k)^c$ for $U \sim \text{Gamma}(k)$, and
    $E[U^c 1\{U \in A\}] = \Gamma(k + c) / \Gamma(k) \cdot P_{k+c}(A)$, so each part is the mean
    times a regularized incomplete gamma function of shape $k + c$. Needs $k + c > 0$.
    """
    k = Q ** (-2.0)
    shift = sigma / Q
    with np.errstate(divide="ignore"):
        z0 = (np.log(t0) - mu) / sigma
    with np.errstate(over="ignore"):
        u0 = np.exp(Q * z0 + np.log(k))
    # T > t0 is U > u0 for Q > 0 and U < u0 for Q < 0.
    share = _sp_gammaincc(k + shift, u0) if upper == (Q > 0) else _sp_gammainc(k + shift, u0)
    return np.exp(mu + _gengamma_log_moment(k, shift)) * share


def _tail_partial_moment(dist: str, mu: Array, sigma: float, t0: Array, *, Q: float = 0.0) -> Array:
    r"""Compute $\int_{t_0}^\infty S(t)\,dt$ for each subject.

    This is the *tail partial moment* starting at `t0`. When `t0 = 0` it equals `E[T]`, and
    therefore `_mean_survival_aft` is a special case.

    The key identities derived from this quantity are:

    - $E[T - t_0 \mid T > t_0] = \text{tail}(t_0) / S(t_0)$
    - $E[T \mid T > t_0] = t_0 + \text{tail}(t_0) / S(t_0)$
    - $E[\min(T, \tau)] = E[T] - \text{tail}(\tau)$  (when $E[T] < \infty$)

    Closed-form formulas by distribution:

    - **Weibull / Exponential**:
      $e^\mu \Gamma(1+\sigma) \cdot \bar{\Gamma}(\sigma,\, u_0)$ where
      $u_0 = (t_0 / e^\mu)^{1/\sigma}$ and $\bar{\Gamma}$ is the regularized upper incomplete gamma
      function.
    - **Log-normal**:
      $e^{\mu+\sigma^2/2} \Phi(\sigma - z_0) - t_0 \Phi(-z_0)$ where
      $z_0 = (\log t_0 - \mu)/\sigma$.
    - **Log-logistic** ($\sigma < 1$):
      $E[T] \cdot I_{S(t_0)}(1-\sigma,\, \sigma)$ where $I_x$ is the regularized incomplete Beta
      function and $S(t_0) = \operatorname{logistic.sf}(z_0)$.
    - **Log-logistic** ($\sigma \ge 1$): numerical integration via `scipy.integrate.quad`.
    - **Generalized gamma** ($k + \sigma/Q > 0$): $E[T \, 1\{T > t_0\}] - t_0 S(t_0)$, the first
      part a regularized incomplete gamma function of shape $k + \sigma/Q$ (see
      `_gengamma_partial_moment`). The tail is infinite otherwise.
    """
    t0 = np.broadcast_to(np.asarray(t0, dtype=float), mu.shape).copy()

    if dist in ("weibull", "exponential"):
        lam = np.exp(mu)
        u0 = (t0 / lam) ** (1.0 / sigma)
        return lam * _sp_gamma(1.0 + sigma) * _sp_gammaincc(sigma, u0)

    if dist == "lognormal":
        # Avoid log(0) = -inf warnings; handle t0 = 0 as a special case
        with np.errstate(divide="ignore", invalid="ignore"):
            log_t0 = np.log(np.where(t0 > 0.0, t0, 1.0))
        log_t0 = np.where(t0 > 0.0, log_t0, -np.inf)
        z0 = (log_t0 - mu) / sigma
        term1 = np.exp(mu + 0.5 * sigma**2) * norm.cdf(sigma - z0)
        term2 = np.where(t0 > 0.0, t0 * norm.sf(z0), 0.0)
        return term1 - term2

    if dist == "gengamma":
        if Q == 0.0:
            return _tail_partial_moment("lognormal", mu, sigma, t0)
        if Q ** (-2.0) + sigma / Q <= 0:
            return np.full_like(mu, np.inf)
        with np.errstate(divide="ignore"):
            z0 = (np.log(t0) - mu) / sigma
        s0 = np.exp(_gengamma_log_survival(z0, Q))
        moment = _gengamma_partial_moment(mu, sigma, Q, t0, upper=True)
        return moment - np.where(t0 > 0.0, t0 * s0, 0.0)

    # loglogistic
    with np.errstate(divide="ignore", invalid="ignore"):
        log_t0 = np.log(np.where(t0 > 0.0, t0, 1.0))
    log_t0 = np.where(t0 > 0.0, log_t0, -np.inf)
    z0 = (log_t0 - mu) / sigma
    s0 = logistic.sf(z0)  # S(t0) per subject
    if sigma < 1.0:
        e_t = np.exp(mu) * np.pi * sigma / np.sin(np.pi * sigma)
        return e_t * _sp_betainc(1.0 - sigma, sigma, s0)
    # sigma >= 1: no closed-form; per-subject numerical integration
    from scipy.integrate import quad

    out = np.empty_like(mu)
    for i in range(mu.shape[0]):
        mu_i = float(mu[i])
        t0_i = float(t0[i])

        def _sf(t: float, _mu: float = mu_i) -> float:
            z = (np.log(t) - _mu) / sigma
            return float(logistic.sf(z))

        out[i], _ = quad(_sf, t0_i, np.inf, limit=200)
    return out


def _rmst_value(
    dist: str, mu: Array, sigma: float, tau: float, *, threshold: float = 0.0, Q: float = 0.0
) -> Array:
    r"""$E[\min(T, \tau)]$ per subject, under an optional threshold shift.

    With a threshold $\gamma$, $T = \gamma + T'$ where $T'$ follows the 2-parameter error
    distribution. If $\tau \le \gamma$, $T > \gamma \ge \tau$ almost surely, so
    $E[\min(T, \tau)] = \tau$ trivially. Otherwise $E[\min(T, \tau)] = \gamma +
    E[\min(T', \tau - \gamma)] = \gamma + E[T'] - \text{tail}'(\tau - \gamma)$, reusing the
    2-parameter closed forms on the shifted horizon.
    """
    if tau <= threshold:
        return np.full_like(mu, tau)
    tau_shifted = tau - threshold
    gengamma_finite = dist == "gengamma" and Q != 0.0 and Q ** (-2.0) + sigma / Q > 0
    if gengamma_finite:
        # E[min(T, tau)] = E[T 1{T <= tau}] + tau S(tau), which avoids subtracting the tail from
        # the mean.
        tau_arr = np.full_like(mu, tau_shifted)
        s_tau = np.exp(_gengamma_log_survival((np.log(tau_shifted) - mu) / sigma, Q))
        lower = _gengamma_partial_moment(mu, sigma, Q, tau_arr, upper=False)
        return threshold + lower + tau_shifted * s_tau
    if (dist == "loglogistic" and sigma >= 1.0) or (dist == "gengamma" and Q != 0.0):
        # E[T'] = inf; integrate S' from 0 to tau_shifted directly.
        from scipy.integrate import quad

        out = np.empty_like(mu)
        for i in range(mu.shape[0]):
            mu_i = float(mu[i])

            def _sf(t: float, _mu: float = mu_i) -> float:
                z = np.atleast_1d((np.log(t) - _mu) / sigma)
                _, log_s = _log_density_survival(dist, z, Q=Q)
                return float(np.exp(log_s[0]))

            val, _ = quad(_sf, 0.0, tau_shifted, limit=200)
            out[i] = val
        return threshold + out
    tau_arr = np.full_like(mu, tau_shifted)
    return (
        threshold
        + _mean_survival_aft(dist, mu, sigma, Q=Q)
        - _tail_partial_moment(dist, mu, sigma, tau_arr, Q=Q)
    )


def _num_hessian(fn: Any, x: Array, rel_step: float = 1e-5) -> Array:
    """Central-difference Hessian of a scalar function at `x`."""
    n = x.shape[0]
    h = rel_step * (np.abs(x) + rel_step)
    hess = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            xi = np.zeros(n)
            xj = np.zeros(n)
            xi[i] = h[i]
            xj[j] = h[j]
            f_pp = fn(x + xi + xj)
            f_pm = fn(x + xi - xj)
            f_mp = fn(x - xi + xj)
            f_mm = fn(x - xi - xj)
            value = (f_pp - f_pm - f_mp + f_mm) / (4.0 * h[i] * h[j])
            hess[i, j] = value
            hess[j, i] = value
    return hess


class AFT:
    r"""Parametric accelerated failure time model.

    While the Cox proportional hazards model leaves the baseline hazard unspecified, AFT models
    assume a fully parametric distribution for survival times and model how covariates accelerate or
    decelerate the "clock" of failure. Specifically,
    $\log(T) = \mu + \beta^\top x + \sigma\varepsilon$, where $T$ is survival time, $\beta$ are
    log-time-scale coefficients, $\sigma$ is a scale parameter, and $\varepsilon$ follows a
    specified error distribution (e.g., extreme-value, logistic, normal). This means a unit increase
    in covariate $x$ multiplies survival time by $\exp(\beta)$.

    AFT models are useful when you want explicit, interpretable survival time predictions or when
    the parametric assumptions are reasonable. Unlike Cox models, they require choosing a
    distributional family (Weibull, exponential, lognormal, loglogistic, or generalized gamma).
    Call `fit()` with a
    right-censored `Surv` response and a design matrix. The model automatically adds an intercept
    and estimates coefficients (on the log-time scale), the scale parameter, and standard errors via
    maximum likelihood.

    The likelihood is maximized by Newton-Raphson with exact derivatives in the coefficients and
    the log scale. Coefficients on the log-time scale can be exponentiated to obtain
    time-acceleration ratios: $\exp(\beta)$ is the multiplicative effect on median or mean survival.
    The model also supports prediction of survival probabilities and quantiles at future times given
    covariate values.

    Parameters
    ----------
    dist
        Error distribution: `"weibull"` (default), `"exponential"`, `"lognormal"`, `"loglogistic"`,
        or `"gengamma"` (generalized gamma). The generalized gamma encompasses Weibull and
        log-normal as special cases, making it useful for distribution selection via likelihood
        ratio tests (see `test_distributions()`). Its shape $Q$ is estimated along with the
        coefficients and the scale, and its estimate and standard error are in `Q_` and `Q_se_`.
    conf_level
        Confidence level for coefficient intervals (the default is `0.95`).
    method
        Estimation method: `"mle"` (default, ordinary maximum likelihood) or `"mps"` (maximum
        product of spacings). MPS replaces each exact observation's density contribution with
        the gap between consecutive order statistics on the CDF scale, which stays bounded where
        MLE's density can diverge; see `threshold` for when this matters.
    threshold
        If `True`, fit a three-parameter model with a location/threshold parameter $\gamma$: $T =
        \gamma + T'$, where $T'$ follows the ordinary 2-parameter `dist`. Not supported for
        `dist="gengamma"`. `method="mle"` with `threshold=True` can drive $\gamma$ toward the
        smallest observed time, where the likelihood is unbounded for some distributions;
        `method="mps"` is recommended in that case (a warning is raised otherwise).

    Returns
    -------
    Fitted estimator
        Call `fit()` to produce a fitted estimator with cached results (`coef_`, `scale_`,
        `threshold_`, `std_error_`{.gd-no-link}, `z_`, `p_value_`, `conf_low_`{.gd-no-link},
        `conf_high_`{.gd-no-link}, `loglik_`, `aic_`, `bic_`), accessible as arrays or exported to
        DataFrames.

    Details
    -------
    Call `fit(surv, covariates)` with a right-censored `Surv` response and a covariate design (a 2-D
    array or a dataframe). An intercept is added automatically; rows with missing covariates are
    dropped. Results are exposed as arrays (`coef_`, `scale_`, `std_error_`{.gd-no-link}, `z_`,
    `p_value_`) and as tidy frames via `to_frame()` (optionally `format=`) and `greenwood.tidy`.

    Examples
    --------
    Fit a Weibull AFT model to the bundled `lung` dataset with `age` and `sex` as covariates.
    Printing the fitted object reports the coefficients (on the log-time scale), the scale, and the
    log-likelihood.

    ```{python}
    import greenwood as gw

    lung = gw.load_dataset("lung", backend="polars")

    # Fit a Weibull AFT model with age and sex as covariates
    aft = gw.AFT(dist="weibull").fit("Surv(time, status) ~ age + sex", data=lung)
    aft
    ```
    """

    def __init__(
        self,
        dist: str = "weibull",
        *,
        conf_level: float = 0.95,
        method: str = "mle",
        threshold: bool = False,
    ) -> None:
        if dist not in _DISTS:
            raise ValueError(f"dist must be one of {sorted(_DISTS)}, got {dist!r}.")
        if not 0.0 < conf_level < 1.0:
            raise ValueError(f"conf_level must be in (0, 1), got {conf_level}.")
        if method not in ("mle", "mps"):
            raise ValueError(f"method must be 'mle' or 'mps', got {method!r}.")
        if threshold and dist == "gengamma":
            raise ValueError("threshold=True is not supported for dist='gengamma'.")
        if threshold and method == "mle":
            warnings.warn(
                "threshold=True with method='mle' can drive the threshold toward the smallest "
                "observed time, where the likelihood is unbounded for some distributions. "
                "method='mps' is recommended instead.",
                stacklevel=2,
            )
        self.dist = dist
        self.conf_level = conf_level
        self.method = method
        self.threshold = threshold

    def __repr__(self) -> str:
        if getattr(self, "coef_", None) is None:
            return (
                f"AFT(dist={self.dist!r}, conf_level={self.conf_level}, "
                f"method={self.method!r}, threshold={self.threshold}) <unfitted>"
            )
        from ._repr import align_table, fixed, num

        rows = [
            [num(c), num(se), fixed(z, 3), num(p)]
            for c, se, z, p in zip(self.coef_, self.std_error_, self.z_, self.p_value_, strict=True)
        ]
        table = align_table(["coef", "se(coef)", "z", "p"], rows, list(self.term_names_))
        header = f"AFT (accelerated failure time model, dist={self.dist!r}"
        if self.method != "mle":
            header += f", method={self.method!r}"
        header += ")"
        lines = [
            header,
            "",
            table,
            "",
            f"Scale = {num(self.scale_)}",
        ]
        if self.Q_ is not None:
            lines.append(f"Q = {num(self.Q_)}")
        if self.threshold:
            lines.append(f"Threshold = {num(self.threshold_)}")
        lines.extend(
            [
                f"n = {self.n_}, events = {self.n_event_}{dropped_note(self)}",
                f"Log-likelihood = {num(self.loglik_)}",
            ]
        )
        return "\n".join(lines)

    def fit(self, surv: Surv | Outcome | str, covariates: Any = None, *, data: Any = None) -> AFT:
        r"""Fit the accelerated failure time model to survival data.

        Fits a parametric accelerated failure time (AFT) model to a right-censored response and
        covariates. The AFT models the log-survival time as a linear regression on covariates plus a
        random error from a specified parametric distribution (Weibull, exponential, log-normal,
        log-logistic, or generalized gamma). An intercept is added automatically.

        The AFT is a parametric alternative to Cox regression, providing a fully specified survival
        distribution at the cost of stronger distributional assumptions. Unlike Cox, AFT supports
        median survival predictions and is naturally interpreted on the log-time scale: a
        coefficient of 0.1 means the covariate multiplies survival time by $\exp(0.1)$. Results are
        stored in the fitted object as coefficient arrays and can be exported to DataFrames.

        Parameters
        ----------
        surv
            A right-censored `Surv` response, built with `gw.Surv(time=..., event=...)`.
            Interval-censored or other response types raise `NotImplementedError`.
            An `Outcome` or a formula string such as `'Surv(time, status == 2) ~ age + sex'` is also
            accepted, with its columns read from `data`. The right-hand side sets `covariates`.
        covariates
            A dataframe (pandas or polars), a 2-D array, or a formula string (e.g., `"age + sex"`)
            evaluated against the `data` argument.
            A list of column names in `data` also works.
        data
            A data frame (pandas, Polars, PyArrow, DuckDB, a lazy frame, ...) holding the columns
            named by the response and `covariates`. When `surv`{.gd-no-link} is an `Outcome` or a
            formula, rows with a missing value in any column used are dropped before fitting.
            Covariate formula strings and lists of column names are also resolved here.

        Returns
        -------
        AFT
            The fitted estimator object itself (for method chaining) with cached coefficient arrays
            (`coef_`, `std_error_`{.gd-no-link}, `z_`, `p_value_`), scale parameter (`scale_`),
            threshold (`threshold_`, `0.0` unless `threshold=True`), and log-likelihood (`loglik_`).

        Details
        -------
        The AFT model parameterizes log-survival time as $\log(T) = X\beta + \sigma\varepsilon$,
        where $X$ is the design matrix, $\beta$ are coefficients, $\sigma$ is a scale parameter, and
        $\varepsilon$ is an error term from the chosen distribution. The survival function is then
        $S(t \mid X) = P(T > t \mid X) = G((\log(t) - X\beta) / \sigma)$, where $G$ is the survival
        function of the error distribution.

        Estimation uses maximum likelihood by default (`method="mle"`), maximized by Newton-Raphson.
        A three-parameter fit uses a bounded quasi-Newton search instead, since the threshold has to
        stay below the smallest observed time.
        Exponential and Weibull models are nested special cases; log-normal and log-logistic offer
        different tail behaviors. With `threshold=True`, $T = \gamma + T'$ for a location parameter
        $\gamma \ge 0$ and 2-parameter `dist` $T'$; `method="mps"` (maximum product of spacings)
        replaces each exact observation's density with the gap between consecutive order
        statistics on the CDF scale, which stays bounded where the ordinary likelihood can diverge
        as $\gamma$ approaches the smallest observed time.

        Examples
        --------
        Fit a log-normal AFT model on the bundled `lung` dataset with `age` and `sex` as covariates:

        ```{python}
        import greenwood as gw

        lung = gw.load_dataset("lung", backend="polars")

        # Fit a log-normal AFT model
        aft = gw.AFT(dist="lognormal").fit("Surv(time, status) ~ age + sex", data=lung)
        aft
        ```

        The response and covariates can also be given separately, with an `Outcome` naming the
        response columns and a right-hand-side formula for the covariates:

        ```{python}
        death = gw.Outcome.surv(time="time", event="status")
        aft_weibull = gw.AFT(dist="weibull").fit(death, covariates="age + sex", data=lung)
        aft_weibull
        ```
        """
        bound = bind_fit_inputs(
            surv,
            data=data,
            designs={"covariates": covariates},
            rhs_to="covariates",
            required=("covariates",),
            estimator="AFT",
        )
        surv = bound.surv
        covariates = bound.designs["covariates"]
        data = bound.data
        self._n_input = bound.n_input
        self.n_dropped_ = bound.n_dropped

        from ._surv import CensoringType

        if surv.type is not CensoringType.RIGHT:
            raise NotImplementedError(
                f"AFT currently supports right-censored responses, not {surv.type.value!r}."
            )

        design, cov_names, self._design_spec = _design_matrix_spec(covariates, data)
        if design.shape[0] != surv.n:
            raise ValueError("Covariates and response must have the same number of rows.")

        time = surv.stop
        event = surv.event
        keep = (~np.isnan(design).any(axis=1)) & (time > 0)
        design, time, event = design[keep], time[keep], event[keep]

        x = np.column_stack([np.ones(design.shape[0]), design])
        names = ["(Intercept)", *cov_names]
        n_coef = x.shape[1]
        has_scale = self.dist != "exponential"
        has_Q = self.dist == "gengamma"
        has_threshold = self.threshold
        t_min = float(time.min())
        # Keep the threshold strictly below every observed time (censored or exact): the
        # likelihood/spacings need log(time - threshold) to be defined for all of them.
        gamma_bound = t_min - 1e-6 * max(t_min, 1.0)

        def _shifted_log_time(gamma: float) -> Array:
            return np.log(np.clip(time - gamma, 1e-300, None))

        def _unpack(params: Array) -> tuple[Array, float, float, float, float]:
            beta = params[:n_coef]
            log_sigma = float(params[n_coef]) if has_scale else 0.0
            sigma = float(np.exp(log_sigma))
            q_val = float(params[n_coef + 1]) if has_Q else 0.0
            gamma = float(params[-1]) if has_threshold else 0.0
            return beta, sigma, log_sigma, q_val, gamma

        def neg_loglik(params: Array) -> float:
            beta, sigma, log_sigma, q_val, gamma = _unpack(params)
            log_t = _shifted_log_time(gamma)
            z = (log_t - x @ beta) / sigma
            log_f, log_s = _log_density_survival(self.dist, z, Q=q_val)
            ll = event * (log_f - log_sigma - log_t) + (1.0 - event) * log_s
            return -float(ll.sum())

        def neg_mps(params: Array) -> float:
            r"""Negative maximum-product-of-spacings objective (Cheng & Amin 1983).

            Exact observations contribute consecutive gaps between sorted CDF values (bounded,
            unlike a raw density, so a threshold parameter cannot drive this to $-\infty$);
            censored observations keep the ordinary $\log S$ contribution (Cheng & Stephens 1989).
            Every subject's error is standardized to the same distribution regardless of
            covariates, so sorting by $z$ generalizes the univariate spacing construction to
            the regression setting.
            """
            beta, sigma, _log_sigma, q_val, gamma = _unpack(params)
            log_t = _shifted_log_time(gamma)
            z = (log_t - x @ beta) / sigma
            exact = event.astype(bool)
            obj = 0.0
            if (~exact).any():
                _, log_s_c = _log_density_survival(self.dist, z[~exact], Q=q_val)
                obj += float(log_s_c.sum())
            if exact.any():
                z_exact = np.sort(z[exact])
                cdf = _cdf(self.dist, z_exact, Q=q_val)
                u = np.concatenate(([0.0], cdf, [1.0]))
                spacing = np.clip(np.diff(u), 1e-300, None)
                obj += float(np.log(spacing).sum())
            return -obj

        objective = neg_mps if self.method == "mps" else neg_loglik

        n_extra = (1 if has_scale else 0) + (1 if has_Q else 0)
        x0 = np.zeros(n_coef + n_extra)
        x0[0] = float(np.log(time).mean())
        mle_info: Array | None = None
        if not has_threshold:
            # Newton-Raphson on the exact derivatives. The generalized gamma starts from the best
            # of a grid of fixed-Q fits, since its likelihood is often flat in Q. Under
            # method="mps", the MLE is the starting point: MPS and MLE are asymptotically
            # equivalent, and the MPS objective's sort-induced kinks make a naive start harder.
            if has_Q:
                x0, _, mle_info, converged = _gengamma_newton(x, np.log(time), event)
            else:
                x0, _, mle_info, converged = _aft_newton(x, np.log(time), event, self.dist)
            if not converged:
                warnings.warn(f"The {self.dist} AFT fit didn't converge.", stacklevel=2)
        elif self.method == "mps":
            # A threshold fit under method="mps" starts from a quasi-Newton MLE.
            res_mle_warm = minimize(
                neg_loglik, x0, method="BFGS", options={"gtol": 1e-8, "maxiter": 1000}
            )
            x0 = res_mle_warm.x

        if has_threshold:
            # The threshold parameter is bounded (0 <= gamma < t_min), which BFGS cannot express,
            # and its objective can be poorly behaved near the boundary (that's the whole reason
            # method="mps" exists), so a bounded optimizer with a few restarts is used instead of
            # plain BFGS.
            bounds = [(None, None)] * (n_coef + n_extra) + [(0.0, gamma_bound)]

            def _fit_bounded(obj_fn: Any, x0_base: Array) -> Any:
                best_fit = None
                for frac in (0.0, 0.3, 0.6, 0.9):
                    gamma0 = frac * gamma_bound
                    x0_t = x0_base.copy()
                    x0_t[0] = float(np.log(np.clip(time - gamma0, 1e-6, None)).mean())
                    x0_t = np.concatenate([x0_t, [gamma0]])
                    res_t = minimize(
                        obj_fn,
                        x0_t,
                        method="L-BFGS-B",
                        bounds=bounds,
                        options={"maxiter": 2000, "ftol": 1e-12, "gtol": 1e-10},
                    )
                    if best_fit is None or res_t.fun < best_fit.fun:
                        best_fit = res_t
                return best_fit

            # Joint (covariates, threshold) estimation can have more than one near-tied local
            # optimum, and picking whichever has the lowest raw objective across a generic grid
            # can land on a degenerate one (coefficients collapsed toward 0, threshold absorbing
            # their explanatory power). The MLE fit's own multi-start grid is trusted as the
            # sensible region; method="mps" only refines *from* it rather than re-searching the
            # grid under its own objective, so it inherits MLE's answer everywhere except where
            # MPS is specifically meant to correct it (the threshold hugging the boundary).
            mle_threshold_fit = _fit_bounded(neg_loglik, x0)
            if self.method == "mle":
                result = mle_threshold_fit
            else:
                result = minimize(
                    objective,
                    mle_threshold_fit.x,
                    method="L-BFGS-B",
                    bounds=bounds,
                    options={"maxiter": 2000, "ftol": 1e-12, "gtol": 1e-10},
                )
            params = result.x
        elif self.method == "mle":
            params = x0
        else:
            params = minimize(
                objective, x0, method="BFGS", options={"gtol": 1e-8, "maxiter": 1000}
            ).x
        if mle_info is not None and self.method == "mle":
            vcov = np.linalg.inv((mle_info + mle_info.T) / 2.0)
        else:
            vcov = np.linalg.inv(_num_hessian(objective, params))

        self._x = x
        self.term_names_ = names
        self.coef_ = params[:n_coef]
        self.vcov_ = vcov
        diag = np.diag(vcov)
        self.std_error_ = np.sqrt(np.clip(diag[:n_coef], 0.0, None))
        self.scale_ = float(np.exp(params[n_coef])) if has_scale else 1.0
        if has_scale:
            scale_var = diag[n_coef]
            self.log_scale_se_ = float(np.sqrt(max(scale_var, 0.0)))
        else:
            self.log_scale_se_ = None
        if has_Q:
            self.Q_ = float(params[n_coef + 1])
            q_var = diag[n_coef + 1]
            self.Q_se_ = float(np.sqrt(max(q_var, 0.0)))
        else:
            self.Q_: float | None = None  # type: ignore[no-redef]
            self.Q_se_: float | None = None  # type: ignore[no-redef]
        self.threshold_ = float(params[-1]) if has_threshold else 0.0
        # loglik_ is always the ordinary AFT log-likelihood at the final estimates, even when
        # method="mps" optimized a different objective, so AIC/BIC stay comparable across methods.
        self.loglik_ = -neg_loglik(params)
        self.n_ = int(keep.sum())
        self.n_dropped_ = max(getattr(self, "_n_input", self.n_) - self.n_, 0)
        self.n_event_ = int(event.sum())
        self._log_t = _shifted_log_time(self.threshold_)
        self._event = event

        z = float(norm.ppf(1.0 - (1.0 - self.conf_level) / 2.0))
        self.z_ = self.coef_ / self.std_error_
        self.p_value_ = 2.0 * norm.sf(np.abs(self.z_))
        self.conf_low_ = self.coef_ - z * self.std_error_
        self.conf_high_ = self.coef_ + z * self.std_error_
        return self

    @property
    def _q(self) -> float:
        """Return Q for gengamma, 0.0 otherwise."""
        return self.Q_ if self.Q_ is not None else 0.0

    def _design(self, newdata: Any) -> Array:
        """Build the intercept-prepended design matrix for `newdata` (or the training data)."""
        if newdata is None:
            return self._x
        design = self._design_spec.transform(newdata)
        return np.column_stack([np.ones(design.shape[0]), design])

    def _survival_pdf(self, z: Array) -> Array:
        """PDF of the standardized error distribution at z-values.

        Parameters
        ----------
        z
            Standardized values where z = (log(t) - mu) / sigma. Can be any shape.

        Returns
        -------
        Array
            PDF values, same shape as input.
        """
        if self.dist in ("weibull", "exponential"):  # minimum extreme value
            return np.exp(z - np.exp(z))
        if self.dist == "lognormal":
            return norm.pdf(z)
        if self.dist == "gengamma":
            log_f, _ = _gengamma_log_density_survival(z, self._q)
            return np.exp(log_f)
        # loglogistic
        return logistic.pdf(z)

    def _theta(self) -> Array:
        """The fitted parameters in the order of `vcov_`: beta, log sigma, Q, and the threshold."""
        parts = [self.coef_]
        if self.dist != "exponential":
            parts.append(np.array([np.log(self.scale_)]))
        if self.Q_ is not None:
            parts.append(np.array([self.Q_]))
        if self.threshold:
            parts.append(np.array([self.threshold_]))
        return np.concatenate(parts)

    def _unpack_theta(self, theta: Array) -> tuple[Array, float, float, float]:
        """Split a parameter vector into the coefficients, sigma, Q, and the threshold."""
        n_coef = self.coef_.shape[0]
        sigma = 1.0 if self.dist == "exponential" else float(np.exp(theta[n_coef]))
        q = float(theta[n_coef + 1]) if self.Q_ is not None else 0.0
        gamma = float(theta[-1]) if self.threshold else 0.0
        return theta[:n_coef], sigma, q, gamma

    def _survival_se(self, x: Array, times: Array) -> Array:
        r"""Standard error of the survival predictions, by the delta method.

        With $z = (\log(t - \gamma) - \mu) / \sigma$ and $S(t) = S_0(z)$, the standard error is
        $f_0(z) \, \text{SE}(z)$, where $f_0$ is the error's density. $\text{SE}(z)$ uses the full
        parameter variance. The gradient of $z$ is $-x / \sigma$ in the coefficients, $-z$ in
        $\log \sigma$, and $-1 / ((t - \gamma) \sigma)$ in the threshold. For the generalized
        gamma, a change in $Q$ is carried onto $z$ as the change in $z$ that moves survival as
        much at the estimated $Q$, $(\partial \log S_0 / \partial Q) / (\partial \log S_0 /
        \partial z)$.

        Parameters
        ----------
        x
            Design matrix (n_subjects, n_features), with the intercept.
        times
            Query times (n_times,).

        Returns
        -------
        Array
            Standard errors of survival predictions, shape (n_times, n_subjects).
        """
        z, se_z, live = self._z_se(x, times)
        return np.where(live[:, None], self._survival_pdf(z) * se_z, 0.0)

    def _z_se(self, x: Array, times: Array) -> tuple[Array, Array, Array]:
        """The standardized `z` at each time and subject, its standard error, and which times
        are past the threshold (where `z` is defined)."""
        n_coef = x.shape[1]
        sigma = self.scale_
        mu = x @ self.coef_
        shifted = times - self.threshold_
        live = shifted > 0.0
        safe = np.where(live, shifted, 1.0)
        z = (np.log(safe)[:, None] - mu[None, :]) / sigma  # (n_times, n_subjects)

        grad = np.zeros((*z.shape, self.vcov_.shape[0]))
        grad[:, :, :n_coef] = -x[None, :, :] / sigma
        if self.dist != "exponential":
            grad[:, :, n_coef] = -z
        if self.Q_ is not None:
            h = 1e-3
            sides = [_gengamma_log_survival(z, self.Q_ + d * h) for d in (-2, -1, 1, 2)]
            d_log_s = (sides[0] - 8.0 * sides[1] + 8.0 * sides[2] - sides[3]) / (12.0 * h)
            log_f, log_s = _gengamma_log_density_survival(z, self.Q_)
            with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                dz_dq = d_log_s / -np.exp(log_f - log_s)
            grad[:, :, n_coef + 1] = np.where(np.isfinite(dz_dq), dz_dq, 0.0)
        if self.threshold:
            grad[:, :, -1] = (-1.0 / (safe * sigma))[:, None]
        grad[~live] = 0.0
        var = np.einsum("tsi,ij,tsj->ts", grad, self.vcov_, grad)
        return z, np.sqrt(np.clip(var, 0.0, None)), live

    def _rmst_se(self, x: Array, tau: float) -> Array:
        """Standard error of the restricted mean at `tau` for each subject, by the delta method
        with a central-difference gradient in all the parameters."""
        theta = self._theta()

        def rmst_at(par: Array) -> Array:
            beta, sigma, q, gamma = self._unpack_theta(par)
            return _rmst_value(self.dist, x @ beta, sigma, tau, threshold=gamma, Q=q)

        grad = np.zeros((x.shape[0], theta.shape[0]))
        for j in range(theta.shape[0]):
            h = 1e-5 * max(abs(float(theta[j])), 1.0)
            up = theta.copy()
            down = theta.copy()
            up[j] += h
            down[j] -= h
            grad[:, j] = (rmst_at(up) - rmst_at(down)) / (2.0 * h)
        var = np.einsum("si,ij,sj->s", grad, self.vcov_, grad)
        return np.sqrt(np.clip(var, 0.0, None))

    def _quantile_se(self, x: Array, p: Array) -> Array:
        r"""Standard error of the time quantiles $\gamma + \exp(\mu + \sigma w_p)$.

        The delta method uses the full parameter variance. The quantile's gradient is $q x$ in
        the coefficients, $q \sigma w_p$ in $\log \sigma$, $q \sigma \partial w_p / \partial Q$ in
        the generalized gamma's $Q$, and 1 in the threshold, where $q = \exp(\mu + \sigma w_p)$.

        Returns
        -------
        Array
            Standard errors, shape (n_subjects, n_p).
        """
        n_coef = x.shape[1]
        n_params = self.vcov_.shape[0]
        sigma = self.scale_
        w = _error_quantile(self.dist, p, Q=self._q)
        q = np.exp((x @ self.coef_)[:, None] + sigma * w[None, :])  # (n_subjects, n_p)

        grad = np.zeros((*q.shape, n_params))
        grad[:, :, :n_coef] = q[:, :, None] * x[:, None, :]
        if self.dist != "exponential":
            grad[:, :, n_coef] = q * sigma * w[None, :]
        if self.Q_ is not None:
            h = 1e-6 * max(1.0, abs(self.Q_))
            dw = (
                _error_quantile(self.dist, p, Q=self.Q_ + h)
                - _error_quantile(self.dist, p, Q=self.Q_ - h)
            ) / (2.0 * h)
            grad[:, :, n_coef + 1] = q * sigma * dw[None, :]
        if self.threshold:
            grad[:, :, -1] = 1.0
        var = np.einsum("spi,ij,spj->sp", grad, self.vcov_, grad)
        return np.sqrt(np.clip(var, 0.0, None))

    # -- residuals & diagnostics ---------------------------------------------

    _RESIDUAL_TYPES = frozenset(
        {
            "response",
            "cox_snell",
            "martingale",
            "deviance",
            "score",
            "dfbeta",
            "dfbetas",
        }
    )

    def residuals(self, type: str = "martingale", *, format: str | None = None) -> Any:
        r"""Return diagnostic residuals from the fitted AFT model.

        Parameters
        ----------
        type
            Type of residuals to return:

            - `"response"`: Raw residuals on the log-time scale, $\log(t) - X\hat\beta$. One per
            observation.
            - `"cox_snell"`: $-\log \hat S(t_i \mid x_i)$, the estimated cumulative hazard at the
            observed time. Under a correct model, these follow $\text{Exp}(1)$ for uncensored
            observations.
            - `"martingale"` (default): $\delta_i - r^{CS}_i$ where $\delta$ is the event indicator
            and $r^{CS}$ is the Cox-Snell residual. Range $(-\infty, 1]$.
            - `"deviance"`: Signed likelihood-ratio residuals. Positive for censored observations;
            signed by $z$ for events. More symmetrically distributed than martingale residuals.
            - `"score"`: Efficient score residuals (derivative of each observation's log-likelihood
            contribution w.r.t. the coefficients). One row per observation, one column per term.
            - `"dfbeta"`: Approximate change in each coefficient if observation $i$ were deleted.
            One row per observation, one column per term.
            - `"dfbetas"`: Standardized dfbeta (dfbeta divided by the coefficient SE). Comparable
            across covariates on different scales.

        format
            Output format for multi-column residual types: `None` (auto-detect), `"pandas"`,
            `"polars"`, or `"pyarrow"`. Returns a numpy array for `"response"`, `"cox_snell"`,
            `"martingale"`, and `"deviance"`.

        Returns
        -------
        ndarray or DataFrame
            For `"response"`, `"cox_snell"`, `"martingale"`, and `"deviance"`: a 1-D array (one
            value per observation). For all other types: a DataFrame with one column per term.

        Examples
        --------
        ```{python}
        import greenwood as gw

        lung = gw.load_dataset("lung", backend="polars")
        aft = gw.AFT(dist="weibull").fit("Surv(time, status) ~ age + sex", data=lung)

        aft.residuals(type="martingale")[:5]
        ```

        ```{python}
        aft.residuals(type="dfbeta", format="polars")
        ```
        """
        if type == "response":
            return np.exp(self._log_t) - np.exp(self._x @ self.coef_)
        if type == "cox_snell":
            return self._cox_snell_residuals()
        if type == "martingale":
            return self._event.astype(float) - self._cox_snell_residuals()
        if type == "deviance":
            z = (self._log_t - self._x @ self.coef_) / self.scale_
            log_f, log_s = _log_density_survival(self.dist, z, Q=self._q)
            event = self._event.astype(float)
            log_f_0, _ = _log_density_survival(self.dist, np.zeros(1), Q=self._q)
            d2_event = 2.0 * (float(log_f_0[0]) - log_f)
            d2_censor = -2.0 * log_s
            d2 = np.where(event == 1, d2_event, d2_censor)
            sign = np.where(event == 1, np.sign(z), 1.0)
            return sign * np.sqrt(np.clip(d2, 0.0, None))
        if type == "score":
            scores = self._score_residuals()
            data = {name: scores[:, j] for j, name in enumerate(self.term_names_)}
            return to_dataframe(data, format=format)
        if type == "dfbeta":
            scores = self._score_residuals()
            n = len(self.coef_)
            vcov_beta = self.vcov_[:n, :n]
            dfb = scores @ vcov_beta
            data = {name: dfb[:, j] for j, name in enumerate(self.term_names_)}
            return to_dataframe(data, format=format)
        if type == "dfbetas":
            scores = self._score_residuals()
            n = len(self.coef_)
            vcov_beta = self.vcov_[:n, :n]
            dfb = scores @ vcov_beta
            dfbs = dfb / self.std_error_[None, :]
            data = {name: dfbs[:, j] for j, name in enumerate(self.term_names_)}
            return to_dataframe(data, format=format)
        valid = "', '".join(sorted(self._RESIDUAL_TYPES))
        raise ValueError(f"Unknown residual type {type!r}; use one of '{valid}'.")

    def _cox_snell_residuals(self) -> Array:
        """Cox-Snell residuals: estimated cumulative hazard at observed time."""
        z = (self._log_t - self._x @ self.coef_) / self.scale_
        _, log_s = _log_density_survival(self.dist, z, Q=self._q)
        return -log_s

    def _score_residuals(self) -> Array:
        """Per-observation score residuals w.r.t. the coefficient vector."""
        z = (self._log_t - self._x @ self.coef_) / self.scale_
        log_f, log_s = _log_density_survival(self.dist, z, Q=self._q)
        dlogf = _dlogf_dz(self.dist, z, Q=self._q)
        dlogs = -np.exp(log_f - log_s)
        w = self._event * dlogf + (1.0 - self._event) * dlogs
        return w[:, None] * (-self._x / self.scale_)

    def predict(
        self,
        newdata: Any = None,
        *,
        type: str = "survival",
        times: Any = None,
        p: Any = 0.5,
        tau: Any = None,
        conditional_after: Any = None,
        ci: bool = False,
        conf_type: str = "log-log",
        format: str | None = None,
    ) -> Any:
        r"""Predict survival times, quantiles, or survival probabilities from the AFT model.

        Generates predictions from a fitted accelerated failure time model. The AFT is a fully
        parametric survival model, so predictions require specifying both the predictor values (via
        `newdata=`) and the type of prediction desired. Pass `newdata=None` to predict for the
        training data (fitted subjects).

        Six prediction types are available:

        1. **Linear predictor** (`type="lp"`): the log-time location $X\beta$, showing how
        covariates shift the log-survival time distribution.

        2. **Quantile** (`type="quantile"`): predicted survival-time quantiles at specified failure
        probabilities (e.g., median survival when $p=0.5$). Useful for clinical summaries like
        "50% of subjects with these covariates survive to time X."

        3. **Survival** (`type="survival"`): survival probabilities $S(t \mid x)$ at specified
        times, returned as a DataFrame for easy visualization. Optionally condition on already
        having survived to a landmark time (`conditional_after=`) for landmark-based predictions.
        With `ci=True`, confidence intervals are included.

        4. **Mean** (`type="mean"`): the expected survival time $E[T]$ (unconditional) or the
        conditional mean $E[T \mid T > t_0]$ when `conditional_after=` is provided. Computed via
        closed-form formulas for all distributions. Returns an array of shape (n_subjects,).

        5. **Mean remaining** (`type="mean_remaining"`): expected remaining lifetime
        $E[T - t_0 \mid T > t_0]$ for subjects known to have survived past `t_0`. Requires
        `conditional_after=`. Returns an array of shape (n_subjects,).

        6. **RMST** (`type="rmst"`): restricted mean survival time $E[\min(T, \tau)]$ up to the
        restriction time `tau`. Requires the `tau` argument. Returns an array of shape
        (n_subjects,).

        Parameters
        ----------
        newdata
            Covariate values for prediction. A DataFrame (Pandas or Polars), 2-D array, or `None`
            (the default). If `None`, uses the training data (design matrix used at fit time). Must
            have the same columns/features as the training data.
            A data frame may also hold other columns. The covariates are picked out by name and
            coded as they were at fit time, so the frame the model was fit on can be passed as is.
        type
            Prediction type (default `"survival"`):

            - `"lp"`: Linear predictor $X\beta$ (log-time location). Returns an array.
            - `"quantile"`: Survival-time quantiles at failure probabilities `p`. Returns a frame
            with `p` column and one column per subject.
            - `"survival"`: Survival probabilities $S(t \mid x)$ at times in `times`. Returns a
            frame with `time` column and one column per subject (one per query time, one per subject
            in newdata).

        times
            Query times for `type="survival"` (ignored for other types). An array-like of floats.
            If `None` (the default), uses an automatic grid based on the fitted distribution (50
            equally spaced times on the log scale, rounded).
        p
            Failure probabilities for `type="quantile"` (ignored for other types). Can be a scalar
            (e.g., `0.5` for median) or array-like. Default is `0.5` (median). Must be in (0, 1).
        tau
            Restriction time for `type="rmst"`. Scalar float giving the upper limit of integration.
            Required when `type="rmst"`, ignored otherwise.
        conditional_after
            For `type="survival"`, optionally compute conditional survival
            $P(T > t \mid T > c) = S(t) / S(c)$. Also used with `type="mean"` and
            `type="mean_remaining"` to condition on surviving past a landmark time $t_0$. Scalar
            (same conditioning time for all subjects) or array-like (one per subject). The default
            is `None` (unconditional).
        ci
            If `True` (survival only), include confidence intervals (`_lower` and `_upper` columns
            per subject). Default is `False`. Not supported with `conditional_after`. The standard
            error of $S(t)$ comes from the delta method with the full parameter variance (the
            coefficients, the log scale, and $Q$ and the threshold when fitted).
        conf_type
            Confidence interval transform (used only if `ci=True` and `type="survival"`):

            - `"log-log"` (default): Log-log transform. Bounds respect the constraint that survival
            $S(t) \in (0, 1)$. Recommended.
            - `"plain"`: Wald bounds without transform. Simple but may produce invalid bounds
            (survival < 0 or > 1).

        format
            Output format for the returned frame (`type="quantile"` or `"survival"`): `None`
            (the default), `"pandas"`, `"polars"`, or `"pyarrow"`. When `None`, a backend is
            auto-detected (Polars, then Pandas, then PyArrow). Ignored for `type="lp"`, `"mean"`,
            `"mean_remaining"`, and `"rmst"` (which always return arrays).

        Returns
        -------
        ndarray or DataFrame
            If `type="lp"`: an array of shape (n_subjects,) containing log-time locations.
            If `type="quantile"`: a DataFrame with columns `p` (failure probabilities) and
            `subject_1`, `subject_2`, etc. containing survival times at each p.
            If `type="survival"`: a DataFrame with columns `time` (query times) and
            `subject_1`, `subject_2`, etc. containing survival probabilities at each time,
            optionally with `_lower` and `_upper` columns for confidence intervals.
            If `type="mean"`: an array of shape (n_subjects,) with $E[T]$ or
            $E[T \mid T > t_0]$.
            If `type="mean_remaining"`: an array of shape (n_subjects,) with
            $E[T - t_0 \mid T > t_0]$.
            If `type="rmst"`: an array of shape (n_subjects,) with $E[\min(T, \tau)]$.

        Details
        -------
        The AFT model assumes $\log(T) = X\beta + \sigma\varepsilon$, where $\varepsilon$
        follows a parametric error distribution (Weibull, lognormal, etc.). Predictions are made
        by evaluating the CDF/survival function of this distribution at covariate-adjusted
        locations. All predictions respect the fitted distribution and scale parameter. If the
        model was fit with `threshold=True`, every time-scale prediction (quantiles, survival
        times, means) is shifted by `threshold_`, and query times at or below `threshold_` report
        $S(t) = 1$ exactly (the event cannot yet have occurred).

        Predictions assume the model is well-specified. For flexible models, consider
        parametric bootstrap to quantify uncertainty.

        Examples
        --------
        Fit a Weibull AFT model on the bundled `lung` dataset, then predict the linear predictor
        (log-time location) for the first two subjects:

        ```{python}
        import greenwood as gw

        # Load data and fit a Weibull AFT model
        lung = gw.load_dataset("lung", backend="polars")
        aft = gw.AFT(dist="weibull").fit("Surv(time, status) ~ age + sex", data=lung)

        # Predict the linear predictor for the first two subjects
        aft.predict(lung[:2], type="lp")
        ```

        Predicted survival-time quantiles for the first two subjects at the lower quartile, median,
        and upper quartile (a table, so pass `format=`):

        ```{python}
        # Predict survival-time quartiles for the first two subjects
        aft.predict(lung[:2], type="quantile", p=[0.25, 0.5, 0.75],
                    format="polars")
        ```

        Read survival probabilities off the fitted curves at chosen times. Here are the estimates at
        180 and 365 days for those same two subjects:

        ```{python}
        # Predict survival probabilities at 180 and 365 days
        aft.predict(lung[:2], type="survival", times=[180, 365],
                    format="polars")
        ```

        Predict conditional survival given already having survived to 100 days:

        ```{python}
        # Predict conditional survival given survival to 100 days
        aft.predict(lung[:2], type="survival", times=[180, 365],
                    conditional_after=100, format="polars")
        ```

        Add confidence intervals with `ci=True`:

        ```{python}
        # Include confidence intervals for the survival predictions
        aft.predict(lung[:2], type="survival", times=[180, 365],
                    ci=True, format="polars")
        ```

        Unconditional mean survival time per subject:

        ```{python}
        # Predict unconditional mean survival time
        aft.predict(lung[:2], type="mean")
        ```

        Expected remaining lifetime given the subject has already survived 100 days:

        ```{python}
        # Predict expected remaining lifetime given survival past 100 days
        aft.predict(lung[:2], type="mean_remaining",
                    conditional_after=100)
        ```

        Restricted mean survival time up to 365 days:

        ```{python}
        # Predict restricted mean survival time up to 365 days
        aft.predict(lung[:2], type="rmst", tau=365)
        ```
        """
        x = self._design(newdata)
        mu = x @ self.coef_
        sigma = self.scale_
        gamma = self.threshold_

        def _log_shifted(t: Array) -> Array:
            """log(t - gamma), with S(t) := 1 (encoded as z = -inf) for t <= gamma."""
            resolvable = t > gamma
            shifted = np.where(resolvable, t - gamma, 1.0)
            return np.where(resolvable, np.log(shifted), -np.inf)

        if type == "lp":
            return mu
        if type == "quantile":
            p_arr = np.atleast_1d(np.asarray(p, dtype=float))
            w = _error_quantile(self.dist, p_arr, Q=self._q)
            quantiles = gamma + np.exp(mu[:, None] + sigma * w[None, :])  # (n_subjects, n_p)
            cols: dict[str, Any] = {"p": p_arr}
            cols.update({f"subject_{i + 1}": quantiles[i] for i in range(quantiles.shape[0])})
            return to_dataframe(cols, format=format)
        if type == "survival":
            if times is None:
                w = _error_quantile(self.dist, np.linspace(0.01, 0.99, 50), Q=self._q)
                query = np.unique(np.round(gamma + np.exp(mu.mean() + sigma * w), 6))
            else:
                query = np.atleast_1d(np.asarray(times, dtype=float))
            z = (_log_shifted(query)[:, None] - mu[None, :]) / sigma  # (n_times, n_subjects)
            _, log_s = _log_density_survival(self.dist, z, Q=self._q)
            if conditional_after is None:
                surv = np.exp(log_s)
            else:
                if ci:
                    raise NotImplementedError(
                        "Confidence intervals are not supported with conditional_after."
                    )
                c = np.asarray(conditional_after, dtype=float)
                if c.ndim == 0:
                    c = np.full(mu.shape[0], float(c))
                if c.shape[0] != mu.shape[0]:
                    raise ValueError("conditional_after must be a scalar or one value per subject.")
                with np.errstate(divide="ignore"):
                    zc = (_log_shifted(c) - mu) / sigma
                _, log_s_c = _log_density_survival(self.dist, zc, Q=self._q)
                log_s_c = np.where(c > gamma, log_s_c, 0.0)  # S(c) = 1 for c <= gamma
                surv = np.exp(np.minimum(log_s - log_s_c[None, :], 0.0))  # ratio capped at 1
            cols = {"time": query}
            if ci:
                se_s = self._survival_se(x, query)  # (n_times, n_subjects)
                z_val = float(norm.ppf(1.0 - (1.0 - self.conf_level) / 2.0))

                if conf_type == "log-log":
                    # Log-log transform: Y = log(-log(S))
                    # SE(Y) = SE(S) / |S * log(S)|
                    with np.errstate(divide="ignore", invalid="ignore"):
                        log_surv = np.log(surv)
                        se_logl = se_s / np.abs(surv * log_surv)  # (n_times, n_subjects)
                        se_logl = np.where(np.isfinite(se_logl), se_logl, 0.0)  # handle 0 or 1
                        logl = np.log(-log_surv)
                        logl = np.where(np.isfinite(logl), logl, 0.0)
                        logl_lower = logl - z_val * se_logl
                        logl_upper = logl + z_val * se_logl
                        # note: upper on log scale -> lower survival
                        surv_lower = np.exp(-np.exp(logl_upper))
                        surv_upper = np.exp(-np.exp(logl_lower))
                else:  # conf_type == "plain"
                    surv_lower = surv - z_val * se_s
                    surv_upper = surv + z_val * se_s

                for i in range(surv.shape[1]):
                    cols[f"subject_{i + 1}"] = surv[:, i]
                    cols[f"subject_{i + 1}_lower"] = surv_lower[:, i]
                    cols[f"subject_{i + 1}_upper"] = surv_upper[:, i]
            else:
                cols.update({f"subject_{i + 1}": surv[:, i] for i in range(surv.shape[1])})
            return to_dataframe(cols, format=format)
        if type == "mean":
            if conditional_after is None:
                return gamma + _mean_survival_aft(self.dist, mu, sigma, Q=self._q)
            c = np.asarray(conditional_after, dtype=float)
            if c.ndim == 0:
                c = np.full(mu.shape[0], float(c))
            if c.shape[0] != mu.shape[0]:
                raise ValueError("conditional_after must be a scalar or one value per subject.")
            c_eff = np.clip(c - gamma, 0.0, None)
            with np.errstate(divide="ignore"):
                zc = (np.log(np.where(c_eff > 0.0, c_eff, 1.0)) - mu) / sigma
            _, log_s_c = _log_density_survival(self.dist, zc, Q=self._q)
            s_c = np.where(c_eff > 0.0, np.exp(log_s_c), 1.0)
            tail = _tail_partial_moment(self.dist, mu, sigma, c_eff, Q=self._q)
            return c + tail / s_c
        if type == "mean_remaining":
            if conditional_after is None:
                raise ValueError("type='mean_remaining' requires conditional_after.")
            c = np.asarray(conditional_after, dtype=float)
            if c.ndim == 0:
                c = np.full(mu.shape[0], float(c))
            if c.shape[0] != mu.shape[0]:
                raise ValueError("conditional_after must be a scalar or one value per subject.")
            c_eff = np.clip(c - gamma, 0.0, None)
            with np.errstate(divide="ignore"):
                zc = (np.log(np.where(c_eff > 0.0, c_eff, 1.0)) - mu) / sigma
            _, log_s_c = _log_density_survival(self.dist, zc, Q=self._q)
            s_c = np.where(c_eff > 0.0, np.exp(log_s_c), 1.0)
            tail = _tail_partial_moment(self.dist, mu, sigma, c_eff, Q=self._q)
            return tail / s_c
        if type == "rmst":
            if tau is None:
                raise ValueError("type='rmst' requires tau.")
            tau_val = float(tau)
            if tau_val <= 0.0:
                raise ValueError(f"tau must be positive, got {tau_val}.")
            return _rmst_value(self.dist, mu, sigma, tau_val, threshold=gamma, Q=self._q)
        raise ValueError(
            f"Unknown predict type {type!r}; use "
            "'lp', 'quantile', 'survival', 'mean', 'mean_remaining', or 'rmst'."
        )

    def predict_quantile(
        self,
        newdata: Any = None,
        *,
        p: Any = 0.5,
        ci: bool = False,
        format: str | None = None,
    ) -> Any:
        r"""Predict survival-time quantiles for each subject.

        The quantile at failure probability $p$ is
        $\hat{t}_p = \exp(\hat{\mu} + \hat{\sigma} w_p)$, where $w_p$ is the $p$-th quantile of the
        standardized error distribution and $\hat{\mu} = x^\top \hat{\beta}$ is the fitted linear
        predictor. This is a closed-form computation.

        When `ci=True`, the standard error of $\hat{t}_p$ comes from the delta method with the full
        parameter variance, so the uncertainty in the scale (and in $Q$ and the threshold, when
        fitted) counts as well as the coefficients'. The limits are
        $\hat{t}_p \exp(\pm z_{\alpha/2} \, \text{SE}(\hat{t}_p) / \hat{t}_p)$, which keeps them
        positive.

        Parameters
        ----------
        newdata
            Covariate values for prediction. A DataFrame (Pandas or Polars), 2-D array, or `None`
            (the default). If `None`, uses the training data.
            A data frame may also hold other columns. The covariates are picked out by name and
            coded as they were at fit time, so the frame the model was fit on can be passed as is.
        p
            Failure probability or probabilities at which to compute quantiles. Can be a scalar
            (e.g., `0.5` for median) or array-like (e.g., `[0.25, 0.5, 0.75]` for quartiles). Must
            be in (0, 1). Default is `0.5`.
        ci
            If `True`, include confidence intervals. Default is `False`.
        format
            Output format: `None` (auto-detect), `"pandas"`, `"polars"`, or `"pyarrow"`.

        Returns
        -------
        DataFrame
            A DataFrame with a `p` column listing the failure probabilities, and columns
            `subject_1`, `subject_2`, etc. containing the corresponding quantile for each subject.
            With `ci=True`, additional `subject_N_lower` and `subject_N_upper` columns are included.

        Examples
        --------
        ```{python}
        import greenwood as gw

        lung = gw.load_dataset("lung", backend="polars")
        aft = gw.AFT(dist="weibull").fit("Surv(time, status) ~ age + sex", data=lung)

        # Predicted survival-time quartiles for three subjects
        aft.predict_quantile(lung[:3], p=[0.25, 0.5, 0.75], format="polars")
        ```

        With confidence intervals at the median:

        ```{python}
        aft.predict_quantile(lung[:3], p=0.5, ci=True, format="polars")
        ```
        """
        p_arr = np.atleast_1d(np.asarray(p, dtype=float))
        if np.any(p_arr <= 0.0) or np.any(p_arr >= 1.0):
            raise ValueError("p must be in (0, 1).")

        x = self._design(newdata)
        mu = x @ self.coef_
        sigma = self.scale_
        gamma = self.threshold_

        w = _error_quantile(self.dist, p_arr, Q=self._q)
        log_q = mu[:, None] + sigma * w[None, :]  # (n_subjects, n_p)
        quantiles = gamma + np.exp(log_q)

        columns: dict[str, Any] = {"p": p_arr}

        if ci:
            z_val = float(norm.ppf(1.0 - (1.0 - self.conf_level) / 2.0))
            se_q = self._quantile_se(x, p_arr)  # (n_subjects, n_p)
            se_log = se_q / quantiles

            for i in range(quantiles.shape[0]):
                columns[f"subject_{i + 1}"] = quantiles[i]
                columns[f"subject_{i + 1}_lower"] = quantiles[i] * np.exp(-z_val * se_log[i])
                columns[f"subject_{i + 1}_upper"] = quantiles[i] * np.exp(z_val * se_log[i])
        else:
            for i in range(quantiles.shape[0]):
                columns[f"subject_{i + 1}"] = quantiles[i]

        return to_dataframe(columns, format=format)

    def predict_median(
        self,
        newdata: Any = None,
        *,
        ci: bool = False,
        format: str | None = None,
    ) -> Any:
        r"""Predict the median survival time for each subject.

        Convenience wrapper around `predict_quantile(p=0.5)`. See `predict_quantile` for
        full documentation.

        Parameters
        ----------
        newdata
            Covariate values for prediction. A DataFrame (Pandas or Polars), 2-D array,
            or `None` (the default). If `None`, uses the training data.
            A data frame may also hold other columns. The covariates are picked out by name and
            coded as they were at fit time, so the frame the model was fit on can be passed as is.
        ci
            If `True`, include confidence intervals. Default is `False`.
        format
            Output format: `None` (auto-detect), `"pandas"`, `"polars"`, or `"pyarrow"`.

        Returns
        -------
        DataFrame
            A single-row DataFrame with columns `p`, `subject_1`, `subject_2`, etc.
            containing the median survival time for each subject. With `ci=True`,
            additional `subject_N_lower` and `subject_N_upper` columns are included.

        Examples
        --------
        ```{python}
        import greenwood as gw

        lung = gw.load_dataset("lung", backend="polars")
        aft = gw.AFT(dist="weibull").fit("Surv(time, status) ~ age + sex", data=lung)

        aft.predict_median(lung[:3], format="polars")
        ```

        With confidence intervals:

        ```{python}
        aft.predict_median(lung[:3], ci=True, format="polars")
        ```
        """
        return self.predict_quantile(newdata, p=0.5, ci=ci, format=format)

    def predict_expectation(
        self,
        newdata: Any = None,
        *,
        tau: float,
        ci: bool = False,
        format: str | None = None,
    ) -> Any:
        r"""Predict the expected (restricted mean) survival time for each subject.

        The restricted mean survival time (RMST) up to horizon $\tau$ is
        $\mathrm{RMST}_i(\tau) = \int_0^{\tau} S(t \mid x_i)\, dt$. For the AFT model with Weibull
        or log-normal errors, this is computed in closed form using the mean minus tail partial
        moment, and for the generalized gamma through incomplete gamma functions. Numerical
        integration is used where the mean is infinite (the log-logistic with $\sigma \ge 1$, and
        the generalized gamma with $k + \sigma/Q \le 0$).

        When `ci=True`, the standard error of the RMST comes from the delta method with the full
        parameter variance, so the uncertainty in the scale (and in $Q$ and the threshold, when
        fitted) counts as well as the coefficients'. The limits are
        $\mathrm{RMST} \exp(\pm z_{\alpha/2} \, \text{SE} / \mathrm{RMST})$, which keeps them
        positive.

        Parameters
        ----------
        newdata
            Covariate values for prediction. A DataFrame (Pandas or Polars), 2-D array, or `None`
            (the default). If `None`, uses the training data.
            A data frame may also hold other columns. The covariates are picked out by name and
            coded as they were at fit time, so the frame the model was fit on can be passed as is.
        tau
            The restriction time (time horizon). Must be positive.
        ci
            If `True`, include confidence intervals. Default is `False`.
        format
            Output format: `None` (auto-detect), `"pandas"`, `"polars"`, or `"pyarrow"`.

        Returns
        -------
        DataFrame
            A DataFrame with a `tau` column and columns `subject_1`, `subject_2`, etc. containing
            the RMST for each subject. With `ci=True`, additional `subject_N_lower` and
            `subject_N_upper` columns are included.

        Examples
        --------
        ```{python}
        import greenwood as gw

        lung = gw.load_dataset("lung", backend="polars")
        aft = gw.AFT(dist="weibull").fit("Surv(time, status) ~ age + sex", data=lung)

        # Expected survival time up to one year for three subjects
        aft.predict_expectation(lung[:3], tau=365, format="polars")
        ```

        With confidence intervals:

        ```{python}
        aft.predict_expectation(lung[:3], tau=365, ci=True, format="polars")
        ```
        """
        tau_val = float(tau)
        if tau_val <= 0.0:
            raise ValueError(f"tau must be positive, got {tau_val}.")

        rmst_vals = self.predict(newdata, type="rmst", tau=tau_val)

        tau_arr = np.atleast_1d(np.asarray(tau_val, dtype=float))
        columns: dict[str, Any] = {"tau": tau_arr}

        if ci:
            z_val = float(norm.ppf(1.0 - (1.0 - self.conf_level) / 2.0))
            se_log = self._rmst_se(self._design(newdata), tau_val) / rmst_vals
            for i in range(rmst_vals.shape[0]):
                columns[f"subject_{i + 1}"] = np.array([rmst_vals[i]])
                columns[f"subject_{i + 1}_lower"] = np.array(
                    [rmst_vals[i] * np.exp(-z_val * se_log[i])]
                )
                columns[f"subject_{i + 1}_upper"] = np.array(
                    [rmst_vals[i] * np.exp(z_val * se_log[i])]
                )
        else:
            for i in range(rmst_vals.shape[0]):
                columns[f"subject_{i + 1}"] = np.array([rmst_vals[i]])

        return to_dataframe(columns, format=format)

    def test_distributions(self, *, format: str | None = None) -> Any:
        r"""Compare the generalized gamma fit against the distributions nested in it.

        Fits the Weibull ($Q = 1$), the log-normal ($Q = 0$), and the exponential ($Q = 1$ with
        $\sigma = 1$) with the same covariates, and tests each against the generalized gamma by a
        likelihood ratio test. The test has 1 degree of freedom for the Weibull and the log-normal,
        and 2 for the exponential. A small p-value says the extra flexibility of the generalized
        gamma is warranted, and a large one that the simpler distribution is adequate. The
        log-logistic isn't nested in the generalized gamma, so it isn't tested.

        Only available when `dist="gengamma"`.

        Parameters
        ----------
        format
            Output format: `None`, `"pandas"`, `"polars"`, or `"pyarrow"`.

        Returns
        -------
        pandas.DataFrame, polars.DataFrame, or pyarrow.Table
            One row per sub-model with columns for the distribution name, log-likelihood, AIC,
            likelihood ratio statistic, degrees of freedom, and p-value.

        Examples
        --------
        ```{python}
        import greenwood as gw

        lung = gw.load_dataset("lung", backend="polars")
        gg = gw.AFT(dist="gengamma").fit("Surv(time, status) ~ age + sex", data=lung)
        gg.test_distributions(format="polars")
        ```
        """
        if self.dist != "gengamma":
            raise ValueError("test_distributions() is only available for dist='gengamma'.")

        from scipy.stats import chi2

        n_coef = self._x.shape[1]
        # The Weibull is Q = 1 and the log-normal Q = 0, each one constraint. The exponential is
        # also sigma = 1, a second constraint.
        sub_dists = {"weibull": 1, "lognormal": 1, "exponential": 2}
        logliks: list[float] = []
        aics: list[float] = []
        lr_stats: list[float] = []
        p_values: list[float] = []
        for dist, df in sub_dists.items():
            _, loglik, _, _ = _aft_newton(self._x, self._log_t, self._event, dist)
            n_params = n_coef + (0 if dist == "exponential" else 1)
            lr = max(2.0 * (self.loglik_ - loglik), 0.0)
            logliks.append(loglik)
            aics.append(-2.0 * loglik + 2.0 * n_params)
            lr_stats.append(lr)
            p_values.append(float(chi2.sf(lr, df)))

        return to_dataframe(
            {
                "dist": list(sub_dists),
                "loglik": logliks,
                "aic": aics,
                "lr_statistic": lr_stats,
                "df": list(sub_dists.values()),
                "p_value": p_values,
            },
            format=format,
        )

    def _coefficient_columns(self) -> dict[str, Any]:
        return {
            "term": self.term_names_,
            "estimate": self.coef_,
            "std_error": self.std_error_,
            "statistic": self.z_,
            "p_value": self.p_value_,
            "conf_low": self.conf_low_,
            "conf_high": self.conf_high_,
        }

    def to_frame(self, *, format: str | None = None) -> Any:
        """Return the coefficient table as a DataFrame.

        Exports one row per term, including the intercept, with coefficient estimates, standard
        errors, Wald statistics, p-values, and confidence limits.

        Parameters
        ----------
        format
            Output format: `None` (default), `"pandas"`, `"polars"`, or `"pyarrow"`. When `None`, a
            backend is auto-detected (Polars, then Pandas, then PyArrow).

        Returns
        -------
        pandas.DataFrame, polars.DataFrame, or pyarrow.Table
            A tidy table with columns `term`, `estimate`, `std_error`, `statistic`, `p_value`,
            `conf_low`, and `conf_high`.

        Raises
        ------
        ImportError
            If the requested (or, when auto-detecting, any) DataFrame library is not installed.

        Examples
        --------
        Fit a Weibull AFT model on the bundled `lung` dataset, then export its coefficient table as
        a Polars frame:

        ```{python}
        import greenwood as gw

        # Load data and fit a Weibull AFT model
        lung = gw.load_dataset("lung", backend="polars")
        aft = gw.AFT(dist="weibull").fit("Surv(time, status) ~ age + sex", data=lung)

        # Export the coefficient table as a Polars DataFrame
        aft.to_frame(format="polars")
        ```

        Request a different backend with `format=`:

        ```{python}
        # Export as a Pandas DataFrame instead
        aft.to_frame(format="pandas")
        ```
        """
        return to_dataframe(self._coefficient_columns(), format=format)


def _tidy_aft(model: AFT, *, format: str | None = None, **_: Any) -> Any:
    return model.to_frame(format=format)


def _glance_aft(model: AFT, *, format: str | None = None, **_: Any) -> Any:
    n_params = len(model.term_names_) + (0 if model.dist == "exponential" else 1)
    if model.dist == "gengamma":
        n_params += 1
    if model.threshold:
        n_params += 1
    cols: dict[str, Any] = {
        "dist": [model.dist],
        "method": [model.method],
        "n": [model.n_],
        "nevent": [model.n_event_],
        "scale": [model.scale_],
    }
    if model.Q_ is not None:
        cols["Q"] = [model.Q_]
    if model.threshold:
        cols["threshold"] = [model.threshold_]
    cols["loglik"] = [model.loglik_]
    cols["aic"] = [-2.0 * model.loglik_ + 2.0 * n_params]
    return to_dataframe(cols, format=format)


def _register_adapters() -> None:
    from .summaries import register_glance, register_tidier

    register_tidier("greenwood._parametric.AFT", _tidy_aft)
    register_glance("greenwood._parametric.AFT", _glance_aft)


_register_adapters()
