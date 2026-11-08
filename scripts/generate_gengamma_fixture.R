#!/usr/bin/env Rscript
# Generate the R-parity fixture for the generalized gamma AFT model.
#
# The reference is flexsurv's generalized gamma (dgengamma() and pgengamma(), Prentice's
# parameterization). flexsurvreg()'s own optimizer stops short of the maximum on lung (its
# log-likelihood is below the nested Weibull's), so the likelihood built from those functions is
# maximized here by Newton-Raphson with numDeriv's Richardson-extrapolated derivatives, starting
# from flexsurvreg()'s estimates. The estimates are pinned down to about 1e-8.
#
# The parameters are ordered as in Greenwood: the coefficients (intercept first), log(sigma), Q.
#
# Run from the repo root:
#   Rscript scripts/generate_gengamma_fixture.R

suppressPackageStartupMessages({
  library(survival)
  library(flexsurv)
  if (!requireNamespace("jsonlite", quietly = TRUE)) {
    stop("jsonlite is required: install.packages('jsonlite')")
  }
  if (!requireNamespace("numDeriv", quietly = TRUE)) {
    stop("numDeriv is required: install.packages('numDeriv')")
  }
})

out_dir <- file.path("fixtures", "r")
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

write_json_fixture <- function(obj, name) {
  path <- file.path(out_dir, paste0(name, ".json"))
  jsonlite::write_json(obj, path, auto_unbox = TRUE, digits = NA, pretty = TRUE)
  cat(sprintf("wrote %s\n", path))
}

time <- lung$time
event <- lung$status == 2

# Per-row log-likelihood contributions on the time scale.
row_loglik <- function(par, x) {
  p <- ncol(x)
  mu <- drop(x %*% par[seq_len(p)])
  sigma <- exp(par[p + 1L])
  Q <- par[p + 2L]
  ifelse(
    event,
    dgengamma(time, mu, sigma, Q, log = TRUE),
    pgengamma(time, mu, sigma, Q, lower.tail = FALSE, log.p = TRUE)
  )
}

maximize <- function(x, start) {
  loglik <- function(par) sum(row_loglik(par, x))
  theta <- start
  for (i in 1:40) {
    step <- solve(numDeriv::hessian(loglik, theta), numDeriv::grad(loglik, theta))
    theta <- theta - step
    if (max(abs(step)) < 1e-9) break
  }
  info <- -numDeriv::hessian(loglik, theta)
  list(theta = theta, loglik = loglik(theta), var = solve((info + t(info)) / 2))
}

x <- cbind("(Intercept)" = 1, age = lung$age, sex = lung$sex)
p <- ncol(x)
start_fit <- flexsurvreg(Surv(time, status) ~ age + sex, data = lung, dist = "gengamma")
start <- unname(coef(start_fit)[c("mu", "age", "sex", "sigma", "Q")])
fit <- maximize(x, start)
theta <- fit$theta
v <- fit$var

null_start <- flexsurvreg(Surv(time, status) ~ 1, data = lung, dist = "gengamma")
null_fit <- maximize(x[, 1L, drop = FALSE], unname(coef(null_start)[c("mu", "sigma", "Q")]))

# The delta method with numDeriv's gradient of `value(theta)` and the variance above.
delta_se <- function(value) {
  g <- numDeriv::jacobian(value, theta)
  sqrt(pmax(rowSums((g %*% v) * g), 0))
}
z_975 <- qnorm(0.975)

newdata <- cbind(1, age = c(50, 70, 60), sex = c(1, 2, 1))
pred_p <- c(0.1, 0.25, 0.5, 0.75, 0.9)
pred_times <- c(100, 200, 365, 500, 730)
tau <- 365

location <- function(par, i) sum(newdata[i, ] * par[seq_len(p)])
sigma_of <- function(par) exp(par[p + 1L])
q_of <- function(par) par[p + 2L]

subjects <- lapply(seq_len(nrow(newdata)), function(i) {
  quantile_at <- function(par) qgengamma(pred_p, location(par, i), sigma_of(par), q_of(par))
  survival_at <- function(par) {
    pgengamma(pred_times, location(par, i), sigma_of(par), q_of(par), lower.tail = FALSE)
  }
  mean_at <- function(par) mean_gengamma(location(par, i), sigma_of(par), q_of(par))
  rmst_at <- function(par) rmst_gengamma(tau, location(par, i), sigma_of(par), q_of(par))

  quantile <- quantile_at(theta)
  quantile_se <- delta_se(quantile_at)
  survival <- survival_at(theta)
  survival_se <- delta_se(survival_at)
  # Survival limits on the scale of the error's z at the estimated Q: z +/- 1.96 se(S) / f(z).
  z <- (log(pred_times) - location(theta, i)) / sigma_of(theta)
  density <- dgengamma(exp(z), 0, 1, q_of(theta)) * exp(z)
  se_z <- survival_se / density
  error_survival <- function(zz) pgengamma(exp(zz), 0, 1, q_of(theta), lower.tail = FALSE)
  list(
    quantile = quantile,
    quantile_se = quantile_se,
    survival = survival,
    survival_se = survival_se,
    survival_low = error_survival(z + z_975 * se_z),
    survival_high = error_survival(z - z_975 * se_z),
    mean = mean_at(theta),
    mean_se = delta_se(mean_at),
    rmst = rmst_at(theta),
    rmst_se = delta_se(rmst_at)
  )
})
pick <- function(name) lapply(subjects, `[[`, name)

# Nested distributions: the Weibull is Q = 1, the log-normal Q = 0, and the exponential Q = 1 with
# sigma = 1.
nested <- lapply(c("weibull", "lognormal", "exponential"), function(d) {
  m <- survreg(Surv(time, status) ~ age + sex, data = lung, dist = d)
  df <- if (d == "exponential") 2L else 1L
  lr <- 2 * (fit$loglik - m$loglik[2])
  list(
    dist = d,
    loglik = m$loglik[2],
    aic = -2 * m$loglik[2] + 2 * (length(m$coefficients) + (d != "exponential")),
    lr_statistic = lr,
    df = df,
    p_value = pchisq(lr, df, lower.tail = FALSE)
  )
})

score <- numDeriv::jacobian(function(par) row_loglik(par, x), theta)
z_rows <- (log(time) - drop(x %*% theta[seq_len(p)])) / sigma_of(theta)

write_json_fixture(
  list(
    dist = "gengamma",
    reference = sprintf("flexsurv %s, maximized with numDeriv", packageVersion("flexsurv")),
    terms = colnames(x),
    coef = theta[seq_len(p)],
    coef_se = sqrt(diag(v))[seq_len(p)],
    scale = sigma_of(theta),
    log_scale_se = sqrt(v[p + 1L, p + 1L]),
    Q = q_of(theta),
    Q_se = sqrt(v[p + 2L, p + 2L]),
    vcov = unname(v),
    loglik = fit$loglik,
    null_loglik = null_fit$loglik,
    n = length(time),
    pred_newdata = list(age = newdata[, "age"], sex = newdata[, "sex"]),
    pred_p = pred_p,
    pred_quantile = pick("quantile"),
    pred_quantile_se = pick("quantile_se"),
    pred_times = pred_times,
    pred_survival = pick("survival"),
    pred_survival_se = pick("survival_se"),
    pred_survival_low = pick("survival_low"),
    pred_survival_high = pick("survival_high"),
    mean = unlist(pick("mean")),
    mean_se = unlist(pick("mean_se")),
    tau = tau,
    rmst = unlist(pick("rmst")),
    rmst_se = unlist(pick("rmst_se")),
    cox_snell = -pgengamma(
      time, drop(x %*% theta[seq_len(p)]), sigma_of(theta), q_of(theta),
      lower.tail = FALSE, log.p = TRUE
    ),
    score = unname(score),
    nested = nested
  ),
  "aft_gengamma"
)
