# Bootstrap confidence intervals for Kaplan-Meier summaries. Tested by
# tests/testthat/test-bootstrap.R against the closed-form estimates, the Kaplan-Meier standard
# errors, and the bootstrap standard errors from the Python package.

#' Bootstrap confidence intervals for Kaplan-Meier summaries
#'
#' `bootstrap()` resamples the subjects with replacement, fits a Kaplan-Meier curve to each
#' resample, and reads a summary statistic from it. The spread of the statistic over the resamples
#' gives its standard error and a confidence interval. This is useful for quantities without a
#' simple closed-form interval, such as the difference in median survival between two groups.
#'
#' The built-in statistics are the median survival time (`"median"`), the `p` quantile of survival
#' time (`"quantile"`), the restricted mean survival time up to `tau` (`"rmst"`), and the survival
#' probability at `times` (`"survival"`). Each of the median, RMST, and survival probability also
#' has a two-group form (`"median_diff"`, `"rmst_diff"`, and `"survival_diff"`) that gives the
#' value in the first group minus the value in the second. The groups come from the right-hand side
#' of `formula`, in the order of the curves of [kaplan_meier()], and the two-group forms resample
#' within each group so every resample keeps the group sizes. Any other summary can be given as a
#' function that takes a [kaplan_meier()] fit and returns one number.
#'
#' A resample whose statistic can't be computed (an error, or a median that the curve never
#' reaches) counts as missing and is left out of the standard error and the interval. The standard
#' error is the standard deviation of the remaining replicates.
#'
#' The `"percentile"` interval takes the \eqn{\alpha / 2} and \eqn{1 - \alpha / 2} quantiles of the
#' replicates, and the `"normal"` interval is the estimate plus or minus \eqn{z} standard errors.
#' The `"bca"` (bias-corrected and accelerated) interval shifts the percentiles by the bias
#' correction \eqn{z_0 = \Phi^{-1}(\text{share of replicates below the estimate})} and the
#' acceleration \eqn{a = \sum d_i^3 / (6 (\sum d_i^2)^{3/2})}, where \eqn{d_i} are the differences
#' between the mean of the leave-one-out (jackknife) values and each of them. The acceleration is 0
#' when a jackknife value is missing. The jackknife refits the curve once per subject.
#'
#' Resampling draws from R's random number generator, so call [set.seed()] first for results that
#' can be repeated.
#'
#' @param formula A formula with a [Surv()] response on the left and either `1` or one or more
#'   grouping variables on the right, such as `Surv(time, status) ~ sex`. The response is
#'   right-censored or counting-process data.
#' @param data A data frame holding the variables in `formula`.
#' @param statistic The quantity to bootstrap: `"median"`, `"quantile"`, `"rmst"`, `"survival"`,
#'   `"median_diff"`, `"rmst_diff"`, or `"survival_diff"`, or a function that takes a
#'   [kaplan_meier()] fit and returns one number. The `"_diff"` statistics need a formula with
#'   exactly two groups, and the others (apart from a function) need a formula without groups.
#' @param tau The end of the window for `"rmst"` and `"rmst_diff"`.
#' @param p The proportion of subjects who have had the event, between 0 and 1, for `"quantile"`.
#' @param times The time of the survival probability for `"survival"` and `"survival_diff"`.
#' @param weights Optional case weights, a column of `data` or a numeric vector. Each resampled
#'   subject keeps its weight.
#' @param subset An optional expression selecting the rows of `data` to use.
#' @param na.action What to do with rows that have missing values. The default drops them.
#' @param n_boot The number of resamples. The default is `1000`.
#' @param conf_level The confidence level, between 0 and 1. The default is `0.95`.
#' @param ci_type The kind of interval: `"percentile"` (the default), `"normal"`, or `"bca"`.
#'
#' @return A `greenwood_bootstrap` object with the `estimate` from the original data, its
#'   bootstrap `std_error`, `conf_low`, `conf_high`, `conf_level`, `ci_type`, `n_boot`, the
#'   `statistic`, the two `groups` compared by a `"_diff"` statistic, and the replicates
#'   (`distribution`, with `NA` for any that failed). [as.data.frame()] gives the estimate, the
#'   standard error, and the limits as a one-row data frame.
#'
#' @references Efron, B., and Tibshirani, R. J. (1993). *An Introduction to the Bootstrap*.
#'   Chapman and Hall.
#'
#' @examples
#' lung <- data.frame(
#'   time = c(306, 455, 1010, 210, 883, 1022, 310, 361, 218, 166, 170, 654, 728, 71, 567),
#'   status = c(2, 2, 1, 2, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2),
#'   sex = c(1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2)
#' )
#'
#' set.seed(1)
#' bootstrap(Surv(time, status) ~ 1, data = lung, statistic = "median", n_boot = 200)
#'
#' # The difference in restricted mean survival time between the sexes
#' bootstrap(Surv(time, status) ~ sex,
#'   data = lung, statistic = "rmst_diff", tau = 365, n_boot = 200
#' )
#'
#' # Any summary of the curve, here the survival probability halfway through a year
#' half_year <- function(fit) predict(fit, times = 182.5)$estimate
#' bootstrap(Surv(time, status) ~ 1,
#'   data = lung, statistic = half_year, n_boot = 200, ci_type = "bca"
#' )
#' @export
bootstrap <- function(formula,
                      data,
                      statistic,
                      tau = NULL,
                      p = NULL,
                      times = NULL,
                      weights,
                      subset,
                      na.action = stats::na.omit,
                      n_boot = 1000,
                      conf_level = 0.95,
                      ci_type = c("percentile", "normal", "bca")) {
  ci_type <- match.arg(ci_type)
  check_conf_level(conf_level)
  check_number(n_boot, "n_boot", min = 1)
  if (n_boot != round(n_boot)) {
    stop("`n_boot` must be a whole number.", call. = FALSE)
  }
  if (missing(statistic)) {
    stop("`bootstrap()` needs a `statistic`.", call. = FALSE)
  }

  frame <- gw_model_frame(
    match.call(expand.dots = FALSE), parent.frame(), formula, if (missing(data)) NULL else data
  )
  if (!is.null(frame$strata) || !is.null(frame$cluster)) {
    stop("`bootstrap()` doesn't take `strata()` or `cluster()` terms.", call. = FALSE)
  }
  parts <- surv_parts(frame$response, "bootstrap()")
  vars <- frame$vars
  stat <- boot_statistic(statistic, tau, p, times, grouped = !is.null(vars))
  n <- length(parts$stop)
  refit <- function(rows) boot_fit(parts, frame$weights, vars, rows, formula)

  fit <- refit(seq_len(n))
  estimate <- stat$fn(fit)
  if (!is.numeric(estimate) || length(estimate) != 1L) {
    stop("`statistic` must return a single number.", call. = FALSE)
  }
  estimate <- as.numeric(estimate)

  # A failed replicate is missing rather than an error, as is a value that isn't a finite number.
  value_of <- function(rows) {
    value <- tryCatch(stat$fn(refit(rows)), error = function(e) NA_real_)
    if (is.numeric(value) && length(value) == 1L && is.finite(value)) as.numeric(value) else NA_real_
  }

  groups <- NULL
  if (stat$diff) {
    if (length(fit$curves) != 2L) {
      stop(
        sprintf(
          "`statistic = \"%s\"` compares two groups, not %d.", stat$label, length(fit$curves)
        ),
        call. = FALSE
      )
    }
    groups <- fit$groups$strata
    blocks <- split(seq_len(n), group_keys(vars, n)$index)
    draw <- function() {
      unlist(lapply(blocks, function(r) r[sample.int(length(r), length(r), replace = TRUE)]),
        use.names = FALSE
      )
    }
  } else {
    draw <- function() sample.int(n, n, replace = TRUE)
  }
  distribution <- vapply(seq_len(n_boot), function(b) value_of(draw()), numeric(1))

  valid <- distribution[!is.na(distribution)]
  std_error <- if (length(valid) > 1L) stats::sd(valid) else NA_real_
  alpha <- 1 - conf_level
  limits <- switch(ci_type,
    percentile = boot_quantiles(valid, c(alpha / 2, 1 - alpha / 2)),
    normal = estimate + c(-1, 1) * z_value(conf_level) * std_error,
    bca = {
      jackknife <- vapply(seq_len(n), function(i) value_of(seq_len(n)[-i]), numeric(1))
      boot_bca(valid, estimate, jackknife, alpha)
    }
  )

  structure(
    list(
      estimate = estimate,
      std_error = std_error,
      conf_low = limits[[1L]],
      conf_high = limits[[2L]],
      conf_level = conf_level,
      ci_type = ci_type,
      n_boot = as.integer(n_boot),
      statistic = stat$label,
      groups = groups,
      distribution = distribution,
      n_dropped = frame$n_dropped,
      formula = formula,
      call = match.call()
    ),
    class = "greenwood_bootstrap"
  )
}

# The statistic as a function of a fit, whether it compares two groups, and its label.
boot_statistic <- function(statistic, tau, p, times, grouped) {
  if (is.function(statistic)) {
    return(list(fn = statistic, diff = FALSE, label = "custom"))
  }
  choices <- c("median", "quantile", "rmst", "survival", "median_diff", "rmst_diff", "survival_diff")
  if (!is.character(statistic) || length(statistic) != 1L || !statistic %in% choices) {
    stop(
      sprintf(
        "`statistic` must be a function or one of %s.",
        paste0("\"", choices, "\"", collapse = ", ")
      ),
      call. = FALSE
    )
  }
  diff <- endsWith(statistic, "_diff")
  base <- sub("_diff$", "", statistic)
  needs <- function(value, name) {
    if (is.null(value)) {
      stop(sprintf("`statistic = \"%s\"` needs `%s`.", statistic, name), call. = FALSE)
    }
  }
  value <- switch(base,
    median = function(curve) km_crossing(curve$time, curve$estimate, 0.5),
    quantile = {
      needs(p, "p")
      check_probability(p, "p")
      function(curve) km_crossing(curve$time, curve$estimate, 1 - p)
    },
    rmst = {
      needs(tau, "tau")
      check_number(tau, "tau")
      function(curve) km_rmst(curve, tau)[["estimate"]]
    },
    survival = {
      needs(times, "times")
      check_number(times, "times")
      function(curve) step_at(curve$time, curve$estimate, times, 1)
    }
  )
  if (diff && !grouped) {
    stop(
      sprintf(
        "`statistic = \"%s\"` compares two groups, so `formula` needs a grouping variable.",
        statistic
      ),
      call. = FALSE
    )
  }
  if (!diff && grouped) {
    stop(
      if (base == "quantile") {
        "`statistic = \"quantile\"` takes no grouping variables."
      } else {
        sprintf("With grouping variables, use `statistic = \"%s_diff\"`.", base)
      },
      call. = FALSE
    )
  }
  fn <- if (diff) {
    function(fit) value(fit$curves[[1L]]) - value(fit$curves[[2L]])
  } else {
    function(fit) value(fit$curves[[1L]])
  }
  list(fn = fn, diff = diff, label = statistic)
}

# A Kaplan-Meier fit to the given rows (with repeats), with the default settings of
# kaplan_meier(). Built from the curves directly so each replicate skips the model frame.
boot_fit <- function(parts, w, vars, rows, formula) {
  groups <- group_keys(if (is.null(vars)) NULL else vars[rows, , drop = FALSE], length(rows))
  z <- z_value(0.95)
  curves <- lapply(seq_len(nrow(groups$keys)), function(g) {
    keep <- rows[groups$index == g]
    km_curve(parts$start[keep], parts$stop[keep], parts$event[keep], w[keep], "log", z)
  })
  structure(
    list(
      curves = curves,
      groups = groups$keys,
      conf_type = "log",
      conf_level = 0.95,
      robust = FALSE,
      n_dropped = 0L,
      formula = formula,
      call = NULL
    ),
    class = "kaplan_meier"
  )
}

boot_quantiles <- function(values, probs) {
  if (length(values) == 0L || anyNA(probs)) {
    return(c(NA_real_, NA_real_))
  }
  stats::quantile(values, probs = probs, type = 7, names = FALSE)
}

# The bias-corrected and accelerated limits.
boot_bca <- function(values, estimate, jackknife, alpha) {
  if (length(values) < 2L || is.na(estimate)) {
    return(c(NA_real_, NA_real_))
  }
  z0 <- stats::qnorm(mean(values < estimate))
  d <- mean(jackknife) - jackknife
  denom <- 6 * sum(d^2)^1.5
  a <- if (!anyNA(d) && denom > 0) sum(d^3) / denom else 0
  z <- stats::qnorm(c(alpha / 2, 1 - alpha / 2))
  shifted <- stats::pnorm(z0 + (z0 + z) / (1 - a * (z0 + z)))
  boot_quantiles(values, pmin(pmax(shifted, 0), 1))
}

#' @export
as.data.frame.greenwood_bootstrap <- function(x, ...) {
  data.frame(
    estimate = x$estimate,
    std_error = x$std_error,
    conf_low = x$conf_low,
    conf_high = x$conf_high
  )
}

#' @export
print.greenwood_bootstrap <- function(x, digits = getOption("digits"), ...) {
  cat("<bootstrap> ", deparse_formula(x$formula), "\n", sep = "")
  compared <- if (is.null(x$groups)) "" else sprintf(" (%s minus %s)", x$groups[1L], x$groups[2L])
  cat(sprintf(
    "Statistic: %s%s, %d resamples, %s interval\n\n",
    x$statistic, compared, x$n_boot, x$ci_type
  ))
  table <- as.data.frame(x)
  names(table)[3:4] <- paste0(format(x$conf_level), c(" LCL", " UCL"))
  rownames(table) <- ""
  print(table, digits = digits)
  failed <- sum(is.na(x$distribution))
  if (failed > 0) {
    cat(sprintf(
      "\n%d %s couldn't be computed and %s left out.\n",
      failed, if (failed == 1) "resample" else "resamples", if (failed == 1) "was" else "were"
    ))
  }
  cat_dropped(x$n_dropped)
  invisible(x)
}
