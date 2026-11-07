# The piecewise exponential model: piecewise_exponential() and its methods. No R fixture covers
# it, so tests/testthat/test-piecewise-exponential.R checks it against glm() on the split data,
# the exponential aft(), and Greenwood's Python values.

#' Piecewise exponential model
#'
#' `piecewise_exponential()` fits a proportional hazards model whose baseline hazard is constant
#' between break points:
#' \deqn{h(t \mid x) = \lambda_j \exp(x^\top \beta) \quad \textrm{for } t \in (b_{j-1}, b_j],}
#' with \eqn{b_0 = 0} and the last interval open-ended. Unlike [cox_ph()], the baseline hazard is
#' estimated along with \eqn{\beta}, as a step function, so the model gives smooth predictions
#' beyond the event times. More intervals give a more flexible hazard.
#'
#' The model is fit by maximum likelihood. Each subject's follow-up is split at the breaks, and
#' the events in each piece are modeled as Poisson counts with the piece's length as the exposure,
#' which gives the same estimates as the survival likelihood. The log-likelihood reported is the
#' survival one, \eqn{\sum_i \delta_i \log h(T_i \mid x_i) - H(T_i \mid x_i)}, so with no breaks it
#' is that of the exponential [aft()].
#'
#' With `breaks = NULL` (the default), the breaks are chosen from the data. Each candidate places
#' \eqn{k = 0, 1, \ldots,} `max_knots` breaks at equally spaced quantiles of the distinct event
#' times, and the one with the lowest AIC (or BIC, with `knot_strategy = "bic"`) is kept. With
#' fewer than 3 distinct event times, there are no breaks.
#'
#' @param formula A formula with a right-censored or counting-process [Surv()] response on the
#'   left and the covariates on the right, such as `Surv(time, status) ~ age + sex`. Factors are
#'   coded with treatment contrasts.
#' @param data A data frame holding the variables in `formula`.
#' @param weights Optional case weights, a column of `data` or a numeric vector.
#' @param subset An optional expression selecting the rows of `data` to use.
#' @param na.action What to do with rows that have missing values. The default drops them.
#' @param breaks The times at which the hazard can change, such as `c(180, 365)` for the intervals
#'   \eqn{(0, 180]}, \eqn{(180, 365]}, and \eqn{(365, \infty)}. Every interval needs some
#'   follow-up time. The default, `NULL`, chooses them from the data.
#' @param knot_strategy How to choose the breaks when `breaks = NULL`: `"aic"` (the default) or
#'   `"bic"`.
#' @param max_knots The largest number of breaks to consider when `breaks = NULL`. The default is
#'   `10`.
#' @param conf_level The confidence level, between 0 and 1. The default is `0.95`.
#' @param max_iter The largest number of Newton-Raphson iterations. The default is `50`.
#' @param tol The convergence tolerance on the largest change in a parameter. The default is
#'   `1e-9`.
#'
#' @return A `piecewise_exponential` object. Use [coef()] or [as.data.frame()] for the
#'   covariates' log hazard ratios, [baseline_hazard()] for the hazard in each interval, and
#'   [predict()] for survival curves. [logLik()], [AIC()], [BIC()], [vcov()], and [nobs()] work as
#'   usual. The object also holds `breaks`, `lr_test` (the likelihood ratio test of all the
#'   covariates), `n`, and `n_event`.
#'
#' @examples
#' lung <- data.frame(
#'   time = c(306, 455, 1010, 210, 883, 1022, 310, 361, 218, 166, 170, 654, 728, 71, 567, 144,
#'     613, 707, 61, 88),
#'   status = c(2, 2, 1, 2, 2, 1, 2, 2, 2, 2, 2, 2, 1, 2, 2, 1, 2, 2, 2, 2),
#'   age = c(74, 68, 56, 57, 60, 74, 68, 71, 53, 61, 57, 68, 68, 60, 57, 67, 70, 63, 56, 57),
#'   sex = c(1, 1, 1, 1, 1, 1, 2, 2, 1, 1, 1, 2, 2, 1, 1, 1, 1, 1, 1, 1)
#' )
#'
#' fit <- piecewise_exponential(Surv(time, status) ~ age + sex, data = lung, breaks = c(180, 365))
#' fit
#' baseline_hazard(fit)
#'
#' # Breaks chosen by AIC
#' piecewise_exponential(Surv(time, status) ~ age + sex, data = lung)
#' @export
piecewise_exponential <- function(formula,
                                  data,
                                  weights,
                                  subset,
                                  na.action = stats::na.omit,
                                  breaks = NULL,
                                  knot_strategy = c("aic", "bic"),
                                  max_knots = 10,
                                  conf_level = 0.95,
                                  max_iter = 50,
                                  tol = 1e-9) {
  knot_strategy <- match.arg(knot_strategy)
  if (!is.null(breaks) && (!is.numeric(breaks) || any(!is.finite(breaks)) || any(breaks <= 0))) {
    stop("`breaks` must be positive numbers, or `NULL`.", call. = FALSE)
  }
  check_number(max_knots, "max_knots", min = 0)
  if (max_knots != round(max_knots)) {
    stop("`max_knots` must be a whole number.", call. = FALSE)
  }
  check_conf_level(conf_level)
  check_number(max_iter, "max_iter", min = 1)
  check_number(tol, "tol", min = 0)
  if (tol <= 0) {
    stop("`tol` must be a single number above 0.", call. = FALSE)
  }
  frame <- gw_model_frame(
    match.call(expand.dots = FALSE), parent.frame(), formula, if (missing(data)) NULL else data
  )
  if (length(frame$strata) > 0L || !is.null(frame$cluster)) {
    stop(
      "`piecewise_exponential()` doesn't support `strata()` or `cluster()` terms.",
      call. = FALSE
    )
  }
  if (!is.null(split_formula(formula, if (missing(data)) NULL else data)$offset)) {
    stop("`piecewise_exponential()` doesn't support `offset()` terms.", call. = FALSE)
  }
  parts <- surv_parts(frame$response, "piecewise_exponential()")
  if (any(parts$stop <= 0)) {
    stop("`piecewise_exponential()` needs positive times.", call. = FALSE)
  }
  design <- covariate_design(frame)
  x <- design$x
  y <- list(
    entry = pmax(parts$start, 0), exit = parts$stop, event = as.numeric(parts$event == 1),
    w = frame$weights
  )
  if (!any(y$event == 1 & y$w > 0)) {
    stop("`piecewise_exponential()` needs at least one event.", call. = FALSE)
  }

  chosen <- is.null(breaks)
  breaks <- if (chosen) {
    pem_select_breaks(y, x, max_knots, knot_strategy, max_iter, tol)
  } else {
    sort(unique(as.numeric(breaks)))
  }
  split <- pem_split(y, x, breaks)
  all_events <- pem_check_intervals(split, breaks)
  fit <- pem_poisson(split, ncol(x), max_iter, tol, variance = TRUE)
  # An interval with no events already has its warning, and explains the lack of convergence.
  if (!fit$converged && all_events) {
    warning(
      sprintf(
        "`piecewise_exponential()` didn't converge in %d iterations.", as.integer(max_iter)
      ),
      call. = FALSE
    )
  }
  null <- pem_poisson(pem_split(y, x[, 0L, drop = FALSE], breaks), 0L, max_iter, tol)

  k <- length(breaks) + 1L
  p <- ncol(x)
  covariate <- k + seq_len(p)
  names(fit$beta) <- colnames(fit$var) <- rownames(fit$var) <- c(
    pem_interval_labels(breaks), colnames(x)
  )
  lr_statistic <- 2 * (fit$loglik - null$loglik)

  structure(
    list(
      coefficients = fit$beta[covariate],
      var = fit$var[covariate, covariate, drop = FALSE],
      log_hazard = fit$beta[seq_len(k)],
      full_var = fit$var,
      breaks = breaks,
      knot_strategy = if (chosen) knot_strategy else NULL,
      loglik = fit$loglik,
      loglik_null = null$loglik,
      lr_test = list(
        statistic = lr_statistic,
        df = p,
        p_value = if (p > 0L) stats::pchisq(lr_statistic, p, lower.tail = FALSE) else NA_real_
      ),
      iterations = fit$iterations,
      converged = fit$converged,
      conf_level = conf_level,
      n = length(y$exit),
      n_event = sum(y$event),
      x = x,
      terms = design$terms,
      xlevels = design$xlevels,
      contrasts = design$contrasts,
      has_weights = frame$has_weights,
      n_dropped = frame$n_dropped,
      formula = formula,
      call = match.call()
    ),
    class = "piecewise_exponential"
  )
}

# Each subject's follow-up (entry, exit], split at the breaks: one piece per subject and interval
# it's followed in, with the piece's interval, its length, whether the event falls in it, and the
# subject's covariates and weight.
pem_split <- function(y, x, breaks) {
  boundaries <- c(0, breaks, Inf)
  pieces <- lapply(seq_len(length(boundaries) - 1L), function(j) {
    start <- pmax(y$entry, boundaries[j])
    stop <- pmin(y$exit, boundaries[j + 1L])
    rows <- which(stop > start)
    list(
      row = rows,
      interval = rep(j, length(rows)),
      exposure = stop[rows] - start[rows],
      event = y$event[rows] * (y$exit[rows] <= boundaries[j + 1L])
    )
  })
  row <- unlist(lapply(pieces, `[[`, "row"))
  interval <- unlist(lapply(pieces, `[[`, "interval"))
  dummies <- matrix(0, length(row), length(boundaries) - 1L)
  dummies[cbind(seq_along(row), interval)] <- 1
  list(
    design = cbind(dummies, x[row, , drop = FALSE]),
    interval = interval,
    exposure = unlist(lapply(pieces, `[[`, "exposure")),
    event = unlist(lapply(pieces, `[[`, "event")),
    w = y$w[row]
  )
}

# Every interval needs follow-up time, and an interval with no events has a hazard of 0, which
# the fit can only approach. Gives whether every interval has events.
pem_check_intervals <- function(split, breaks) {
  k <- length(breaks) + 1L
  exposure <- tabulate_sum(split$interval, split$w * split$exposure, k)
  events <- tabulate_sum(split$interval, split$w * split$event, k)
  labels <- pem_interval_labels(breaks)
  if (any(exposure <= 0)) {
    stop(
      sprintf(
        "No one is followed in the interval %s. Remove a break so every interval has follow-up.",
        labels[which(exposure <= 0)[1L]]
      ),
      call. = FALSE
    )
  }
  if (any(events <= 0)) {
    warning(
      sprintf(
        "The interval %s has no events, so its hazard is estimated as about 0.",
        labels[which(events <= 0)[1L]]
      ),
      call. = FALSE
    )
  }
  all(events > 0)
}

pem_interval_labels <- function(breaks) {
  boundaries <- c(0, breaks, Inf)
  k <- length(boundaries) - 1L
  label <- function(b) vapply(b, format, character(1), digits = 6L)
  paste0(
    "(", label(boundaries[-(k + 1L)]), ", ", label(boundaries[-1L]),
    ifelse(is.finite(boundaries[-1L]), "]", ")")
  )
}

# The Poisson fit by Newton-Raphson from 0. The log-likelihood leaves out the events' log
# exposures, which turns the Poisson log-likelihood into the survival one.
pem_poisson <- function(split, p, max_iter, tol, variance = FALSE) {
  design <- split$design
  offset <- log(split$exposure)
  y <- split$event
  w <- split$w
  beta <- numeric(ncol(design))
  converged <- FALSE
  iterations <- 0L
  for (it in seq_len(max_iter)) {
    iterations <- it
    mu <- exp(drop(design %*% beta) + offset)
    gradient <- crossprod(design, w * (y - mu))
    information <- crossprod(design * (w * mu), design)
    step <- tryCatch(drop(solve(information, gradient, tol = 0)), error = function(e) NULL)
    if (is.null(step)) {
      break
    }
    beta <- beta + step
    if (max(abs(step)) < tol) {
      converged <- TRUE
      break
    }
  }
  eta <- drop(design %*% beta) + offset
  mu <- exp(eta)
  var <- NULL
  if (variance) {
    var <- solve(crossprod(design * (w * mu), design), tol = 0)
  }
  list(
    beta = beta,
    loglik = sum(w * (y * (eta - offset) - mu)),
    var = var,
    converged = converged,
    iterations = iterations
  )
}

# The breaks with the lowest AIC or BIC among 0 to `max_knots` breaks at equally spaced
# quantiles of the distinct event times. A tie keeps the fewer breaks.
pem_select_breaks <- function(y, x, max_knots, strategy, max_iter, tol) {
  event_times <- sort(unique(y$exit[y$event == 1 & y$w > 0]))
  if (length(event_times) < 3L) {
    return(numeric(0))
  }
  penalty <- if (strategy == "aic") 2 else log(length(y$exit))
  best <- Inf
  best_breaks <- numeric(0)
  for (k in seq(0, max_knots)) {
    breaks <- if (k == 0) {
      numeric(0)
    } else {
      probs <- seq(0, 1, length.out = k + 2L)[-c(1L, k + 2L)]
      unique(stats::quantile(event_times, probs, names = FALSE, type = 7))
    }
    fit <- pem_poisson(pem_split(y, x, breaks), ncol(x), max_iter, tol)
    criterion <- -2 * fit$loglik + penalty * length(fit$beta)
    if (criterion < best) {
      best <- criterion
      best_breaks <- breaks
    }
  }
  best_breaks
}

#' @export
coef.piecewise_exponential <- function(object, ...) {
  object$coefficients
}

#' @export
vcov.piecewise_exponential <- function(object, ...) {
  object$var
}

#' @export
nobs.piecewise_exponential <- function(object, ...) {
  object$n
}

#' @export
logLik.piecewise_exponential <- function(object, ...) {
  structure(
    object$loglik,
    df = nrow(object$full_var),
    nobs = object$n,
    class = "logLik"
  )
}

#' @describeIn baseline_hazard For a [piecewise_exponential()] fit, the hazard of a subject with
#'   every covariate at 0, one row per interval: `start`, `stop`, `hazard`, `log_hazard`,
#'   `std_error` (of the log hazard), and `conf_low` and `conf_high` (limits for the hazard, built
#'   on the log scale).
#'
#' @export
baseline_hazard.piecewise_exponential <- function(object, conf_level = object$conf_level, ...) {
  check_conf_level(conf_level)
  k <- length(object$log_hazard)
  boundaries <- c(0, object$breaks, Inf)
  log_hazard <- unname(object$log_hazard)
  std_error <- unname(sqrt(diag(object$full_var))[seq_len(k)])
  z <- z_value(conf_level)
  data.frame(
    start = boundaries[-(k + 1L)],
    stop = boundaries[-1L],
    hazard = exp(log_hazard),
    log_hazard = log_hazard,
    std_error = std_error,
    conf_low = exp(log_hazard - z * std_error),
    conf_high = exp(log_hazard + z * std_error)
  )
}

#' Predictions and coefficients of a piecewise exponential model
#'
#' `predict()` gives, for each row of `newdata` (or of the data the model was fit to), the
#' survival or cumulative hazard curve, the linear predictor \eqn{x^\top \beta}, or the risk
#' score \eqn{\exp(x^\top \beta)}. The cumulative hazard is
#' \eqn{H(t \mid x) = \exp(x^\top \beta) \sum_j \lambda_j \ell_j(t)}, where \eqn{\ell_j(t)} is the
#' time spent in interval \eqn{j} by time \eqn{t}, and survival is \eqn{\exp(-H)}. Their
#' standard errors use the delta method with the variance of all the parameters, and the limits
#' are built on the scale of \eqn{\log H}, so survival stays within 0 and 1.
#'
#' `as.data.frame()` gives the covariates' log hazard ratios, with Wald tests and limits.
#'
#' @param object,x A [piecewise_exponential()] fit.
#' @param newdata A data frame with the covariates of the subjects to predict for. The default is
#'   the data the model was fit to.
#' @param type What to predict: `"survival"` (the default), `"cumhaz"`, `"lp"`, or `"risk"`.
#' @param times For `"survival"` and `"cumhaz"`, the times to read the curves at.
#' @param conf_level The confidence level. The default is the level the model was fit with.
#' @param row.names,optional Unused. They're part of the [as.data.frame()] generic.
#' @param ... Unused.
#'
#' @return For `"lp"` and `"risk"`, a numeric vector with one value per row of `newdata`. For
#'   `"survival"` and `"cumhaz"`, a data frame with one row per subject and time: `subject` (the
#'   row of `newdata`), `time`, `estimate`, `std_error`, `conf_low`, and `conf_high`.
#'   `as.data.frame()` gives a data frame with the columns `term`, `estimate`, `std_error`,
#'   `statistic`, `p_value`, `conf_low`, and `conf_high`.
#'
#' @examples
#' lung <- data.frame(
#'   time = c(306, 455, 1010, 210, 883, 1022, 310, 361, 218, 166, 170, 654, 728, 71, 567, 144,
#'     613, 707, 61, 88),
#'   status = c(2, 2, 1, 2, 2, 1, 2, 2, 2, 2, 2, 2, 1, 2, 2, 1, 2, 2, 2, 2),
#'   age = c(74, 68, 56, 57, 60, 74, 68, 71, 53, 61, 57, 68, 68, 60, 57, 67, 70, 63, 56, 57),
#'   sex = c(1, 1, 1, 1, 1, 1, 2, 2, 1, 1, 1, 2, 2, 1, 1, 1, 1, 1, 1, 1)
#' )
#' fit <- piecewise_exponential(Surv(time, status) ~ age + sex, data = lung, breaks = c(180, 365))
#' as.data.frame(fit)
#' newdata <- data.frame(age = c(55, 70), sex = c(1, 2))
#' predict(fit, newdata = newdata, type = "survival", times = c(100, 300, 500))
#' predict(fit, newdata = newdata, type = "risk")
#' @name piecewise_exponential_predict
#' @export
predict.piecewise_exponential <- function(object,
                                          newdata,
                                          type = c("survival", "cumhaz", "lp", "risk"),
                                          times,
                                          conf_level = object$conf_level,
                                          ...) {
  type <- match.arg(type)
  check_conf_level(conf_level)
  if (missing(newdata)) {
    x <- object$x
  } else {
    if (!is.data.frame(newdata)) {
      stop("`newdata` must be a data frame.", call. = FALSE)
    }
    x <- cox_new_design(
      list(
        terms = object$terms, xlevels = object$xlevels, contrasts = object$contrasts,
        coefficients = stats::setNames(numeric(ncol(object$x)), colnames(object$x))
      ),
      newdata
    )
  }
  lp <- linear_predictor(x, object$coefficients)
  if (type == "lp") {
    return(lp)
  }
  if (type == "risk") {
    return(exp(lp))
  }
  if (missing(times)) {
    stop(sprintf("`type = \"%s\"` needs `times`.", type), call. = FALSE)
  }
  times <- check_times(times)
  if (any(times < 0)) {
    stop("`times` can't be negative.", call. = FALSE)
  }
  # The time spent in each interval by each time, one row per time.
  boundaries <- c(0, object$breaks, Inf)
  k <- length(boundaries) - 1L
  spent <- vapply(seq_len(k), function(j) {
    pmax(pmin(times, boundaries[j + 1L]) - boundaries[j], 0)
  }, numeric(length(times)))
  spent <- matrix(spent, nrow = length(times))
  pieces <- spent * rep(exp(object$log_hazard), each = length(times))
  cumhaz0 <- rowSums(pieces)
  positive <- cumhaz0 > 0
  z <- z_value(conf_level)
  rows <- lapply(seq_along(lp), function(i) {
    # The gradient of log H in the interval log hazards and then the coefficients.
    gradient <- cbind(
      pieces / ifelse(positive, cumhaz0, 1),
      matrix(x[i, ], length(times), ncol(x), byrow = TRUE)
    )
    se <- sqrt(pmax(rowSums((gradient %*% object$full_var) * gradient), 0))
    log_h <- log(cumhaz0) + lp[[i]]
    cumhaz <- exp(log_h)
    out <- if (type == "survival") {
      data.frame(
        time = times,
        estimate = exp(-cumhaz),
        std_error = exp(-cumhaz) * cumhaz * se,
        conf_low = exp(-exp(log_h + z * se)),
        conf_high = exp(-exp(log_h - z * se))
      )
    } else {
      data.frame(
        time = times,
        estimate = cumhaz,
        std_error = cumhaz * se,
        conf_low = exp(log_h - z * se),
        conf_high = exp(log_h + z * se)
      )
    }
    start <- if (type == "survival") 1 else 0
    out[!positive, -1L] <- data.frame(
      estimate = start, std_error = 0, conf_low = start, conf_high = start
    )
    cbind(data.frame(subject = i), out)
  })
  out <- do.call(rbind, rows)
  rownames(out) <- NULL
  out
}

#' @rdname piecewise_exponential_predict
#' @export
as.data.frame.piecewise_exponential <- function(x,
                                                row.names = NULL,
                                                optional = FALSE,
                                                ...,
                                                conf_level = x$conf_level) {
  check_conf_level(conf_level)
  z <- z_value(conf_level)
  estimate <- x$coefficients
  std_error <- sqrt(diag(x$var))
  statistic <- estimate / std_error
  data.frame(
    term = names(estimate),
    estimate = unname(estimate),
    std_error = unname(std_error),
    statistic = unname(statistic),
    p_value = unname(2 * stats::pnorm(-abs(statistic))),
    conf_low = unname(estimate - z * std_error),
    conf_high = unname(estimate + z * std_error),
    stringsAsFactors = FALSE
  )
}

#' @export
print.piecewise_exponential <- function(x, digits = getOption("digits"), ...) {
  cat("<piecewise_exponential> ", deparse_formula(x$formula), "\n", sep = "")
  k <- length(x$log_hazard)
  chosen <- if (is.null(x$knot_strategy)) "" else {
    sprintf(", breaks chosen by %s", toupper(x$knot_strategy))
  }
  cat(sprintf(
    "%d %s%s. n = %d, events = %s\n\n",
    k, if (k == 1L) "interval" else "intervals", chosen, x$n, format(x$n_event)
  ))
  short <- max(3L, digits - 4L)
  if (length(x$coefficients) > 0L) {
    table <- as.data.frame(x)
    coefs <- data.frame(
      estimate = table$estimate,
      hazard_ratio = exp(table$estimate),
      std_error = table$std_error,
      statistic = table$statistic,
      p_value = table$p_value
    )
    rownames(coefs) <- table$term
    print(coefs, digits = max(3L, digits - 3L))
    cat("\n")
  }
  hazard <- baseline_hazard(x)
  baseline <- data.frame(hazard = hazard$hazard, std_error = hazard$std_error * hazard$hazard)
  rownames(baseline) <- pem_interval_labels(x$breaks)
  cat("Baseline hazard:\n")
  print(baseline, digits = max(3L, digits - 3L))
  cat(sprintf(
    "\nLog-likelihood: %s, AIC: %s\n",
    format(x$loglik, digits = max(short, 6L)),
    format(stats::AIC(x), digits = max(short, 6L))
  ))
  if (x$lr_test$df > 0) {
    cat(sprintf(
      "Likelihood ratio test: %s on %d df, p = %s\n",
      format(x$lr_test$statistic, digits = short), as.integer(x$lr_test$df),
      format_p(x$lr_test$p_value)
    ))
  }
  cat_dropped(x$n_dropped)
  invisible(x)
}
