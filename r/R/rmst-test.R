# Comparisons of restricted mean survival time between groups: rmst_test(), rmst_diff(), and
# pairwise_rmst_test(). The numeric fixture (rmst_twogroup_lung) is replayed by
# tests/testthat/test-rmst-test.R.

#' Compare restricted mean survival time between two groups
#'
#' `rmst_test()` compares the restricted mean survival time (RMST) up to `tau` of two groups. The
#' RMST is the area under the Kaplan-Meier curve on \eqn{[0, \tau]} (see [rmst()]). The comparison
#' is a Wald test of the difference, the ratio, or the percentage difference between the first
#' group and the second. Unlike the hazard ratio, the RMST difference doesn't depend on
#' proportional hazards and reads directly as time gained or lost.
#'
#' With standard errors \eqn{s_1} and \eqn{s_2} of the two groups' RMSTs \eqn{r_1} and \eqn{r_2},
#' the difference \eqn{r_1 - r_2} has standard error \eqn{\sqrt{s_1^2 + s_2^2}}. The ratio
#' \eqn{r_1 / r_2} is tested on the log scale, where the delta method gives the standard error
#' \eqn{\sqrt{(s_1 / r_1)^2 + (s_2 / r_2)^2}}, and its confidence limits are exponentiated. The
#' reported standard error of the ratio itself is that times the ratio. The
#' percentage difference is \eqn{100 (r_1 - r_2) / r_2}, with the standard error of the difference
#' scaled by \eqn{100 / r_2}.
#'
#' A `strata()` term on the right-hand side gives a stratified comparison. Within each stratum
#' that holds both groups, the two RMSTs and the variance of their difference \eqn{v_s} are
#' computed, and the strata are pooled with weights \eqn{w_s = 1 / v_s}. The pooled difference has
#' variance \eqn{1 / \sum w_s}, and the pooled RMSTs reported for each group use the same weights.
#'
#' @param formula A formula with a [Surv()] response on the left and a grouping variable on the
#'   right, such as `Surv(time, status) ~ sex`. A `strata()` term, such as
#'   `Surv(time, status) ~ sex + strata(ph.ecog)`, stratifies the comparison.
#' @param data A data frame holding the variables in `formula`.
#' @param tau The end of the window for the restricted mean.
#' @param subset An optional expression selecting the rows of `data` to use.
#' @param na.action What to do with rows that have missing values. The default drops them.
#' @param estimand What to compare: `"difference"` (the default), `"ratio"`, or
#'   `"percentage_difference"`.
#' @param conf_level The confidence level, between 0 and 1. The default is `0.95`.
#'
#' @return `rmst_test()` gives an `rmst_test` object with the `estimate`, its `std_error`,
#'   `conf_low`, `conf_high`, the Wald `statistic`, the two-sided `p_value`, `method`, the two groups' labels (`group1` and `group2`), their RMSTs and standard
#'   errors (`rmst1`, `se1`, `rmst2`, and `se2`), `tau`, `estimand`, `stratified`, and
#'   `conf_level`. [as.data.frame()] gives it as a one-row data frame.
#'
#'   `rmst_diff()` gives that one-row data frame for the difference directly: `group1`, `group2`,
#'   `rmst1`, `rmst2`, `difference`, `std_error`, `conf_low`, `conf_high`, `statistic`, and
#'   `p_value`.
#'
#' @seealso [pairwise_rmst_test()] for more than two groups.
#'
#' @examples
#' lung <- data.frame(
#'   time = c(306, 455, 1010, 210, 883, 1022, 310, 361, 218, 166, 170, 654),
#'   status = c(2, 2, 1, 2, 2, 1, 2, 2, 2, 2, 2, 2),
#'   sex = c(1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2)
#' )
#' rmst_test(Surv(time, status) ~ sex, data = lung, tau = 365)
#' rmst_test(Surv(time, status) ~ sex, data = lung, tau = 365, estimand = "ratio")
#' rmst_diff(Surv(time, status) ~ sex, data = lung, tau = 365)
#' @export
rmst_test <- function(formula,
                      data,
                      tau,
                      subset,
                      na.action = stats::na.omit,
                      estimand = c("difference", "ratio", "percentage_difference"),
                      conf_level = 0.95) {
  estimand <- match.arg(estimand)
  check_number(tau, "tau")
  check_conf_level(conf_level)
  frame <- gw_model_frame(
    match.call(expand.dots = FALSE), parent.frame(), formula, if (missing(data)) NULL else data
  )
  input <- rmst_input(frame, "rmst_test()")
  if (nrow(input$keys) > 2L) {
    stop(
      "`rmst_test()` compares two groups. Use `pairwise_rmst_test()` for more.",
      call. = FALSE
    )
  }
  rmst_compare(input, 1L, 2L, tau, estimand, conf_level)
}

#' @rdname rmst_test
#' @export
rmst_diff <- function(formula,
                      data,
                      tau,
                      subset,
                      na.action = stats::na.omit,
                      conf_level = 0.95) {
  call <- match.call()
  call[[1L]] <- quote(greenwood::rmst_test)
  call$estimand <- "difference"
  result <- eval(call, parent.frame())
  out <- as.data.frame(result)
  out <- out[c(
    "group1", "group2", "rmst1", "rmst2", "estimate", "std_error", "conf_low", "conf_high",
    "statistic", "p_value"
  )]
  names(out)[names(out) == "estimate"] <- "difference"
  out
}

#' Compare restricted mean survival time between every pair of groups
#'
#' `pairwise_rmst_test()` runs [rmst_test()] on each pair of groups, using only the rows of the two
#' groups, and adjusts the p-values for the number of comparisons.
#'
#' @inheritParams rmst_test
#' @param formula A formula with a [Surv()] response on the left and the grouping variables on the
#'   right, such as `Surv(time, status) ~ celltype`. A `strata()` term stratifies each comparison.
#' @param correction The adjustment for multiple comparisons: `"holm"` (the default), `"bh"`
#'   (Benjamini-Hochberg), `"bonferroni"`, or `"none"`.
#'
#' @return A data frame with one row per pair of groups, in the order of the groups: `group1`,
#'   `group2`, `rmst1`, `rmst2`, `estimate`, `std_error`, `conf_low`, `conf_high`, `statistic`,
#'   `p_value` (unadjusted), and `p_adjusted`.
#'
#' @examples
#' lung <- data.frame(
#'   time = c(306, 455, 1010, 210, 883, 1022, 310, 361, 218, 166, 170, 654),
#'   status = c(2, 2, 1, 2, 2, 1, 2, 2, 2, 2, 2, 2),
#'   ecog = c(0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2)
#' )
#' pairwise_rmst_test(Surv(time, status) ~ ecog, data = lung, tau = 500)
#' @export
pairwise_rmst_test <- function(formula,
                               data,
                               tau,
                               subset,
                               na.action = stats::na.omit,
                               estimand = c("difference", "ratio", "percentage_difference"),
                               correction = c("holm", "bh", "bonferroni", "none"),
                               conf_level = 0.95) {
  estimand <- match.arg(estimand)
  correction <- match.arg(correction)
  check_number(tau, "tau")
  check_conf_level(conf_level)
  frame <- gw_model_frame(
    match.call(expand.dots = FALSE), parent.frame(), formula, if (missing(data)) NULL else data
  )
  input <- rmst_input(frame, "pairwise_rmst_test()")
  n_groups <- nrow(input$keys)
  rows <- list()
  for (a in seq_len(n_groups - 1L)) {
    for (b in seq(a + 1L, n_groups)) {
      result <- as.data.frame(rmst_compare(input, a, b, tau, estimand, conf_level))
      rows[[length(rows) + 1L]] <- result[c(
        "group1", "group2", "rmst1", "rmst2", "estimate", "std_error", "conf_low", "conf_high",
        "statistic", "p_value"
      )]
    }
  }
  out <- do.call(rbind, rows)
  out$p_adjusted <- adjust_p(out$p_value, correction)
  rownames(out) <- NULL
  out
}

rmst_input <- function(frame, fn) {
  if (is.null(frame$vars)) {
    stop(
      sprintf("`%s` needs a grouping variable on the right-hand side of `formula`.", fn),
      call. = FALSE
    )
  }
  parts <- surv_parts(frame$response, fn)
  groups <- group_keys(frame$vars, nrow(frame$mf))
  if (nrow(groups$keys) < 2L) {
    stop(sprintf("`%s` needs at least 2 groups.", fn), call. = FALSE)
  }
  list(
    start = parts$start,
    stop = parts$stop,
    event = parts$event,
    w = frame$weights,
    group = groups$index,
    keys = groups$keys,
    labels = lr_group_labels(groups$keys),
    stratum = if (is.null(frame$strata)) NULL else group_keys(frame$strata, nrow(frame$mf))$index,
    formula = frame$formula
  )
}

# The RMST and its standard error for the rows `rows`.
rmst_of <- function(input, rows, tau) {
  counts <- count_events(input$start[rows], input$stop[rows], input$event[rows], input$w[rows])
  counts$estimate <- cumprod(ifelse(counts$n_risk > 0, 1 - counts$n_event / counts$n_risk, 1))
  km_rmst(counts, tau)
}

# Compare groups `a` and `b` of `input`.
rmst_compare <- function(input, a, b, tau, estimand, conf_level) {
  if (is.null(input$stratum)) {
    one <- rmst_of(input, input$group == a, tau)
    two <- rmst_of(input, input$group == b, tau)
    rmst1 <- one[["estimate"]]
    se1 <- one[["std_error"]]
    rmst2 <- two[["estimate"]]
    se2 <- two[["std_error"]]
  } else {
    per_stratum <- lapply(sort(unique(input$stratum)), function(s) {
      in_s <- input$stratum == s
      if (!any(in_s & input$group == a) || !any(in_s & input$group == b)) {
        return(NULL)
      }
      one <- rmst_of(input, in_s & input$group == a, tau)
      two <- rmst_of(input, in_s & input$group == b, tau)
      if (!(one[["std_error"]] > 0 && two[["std_error"]] > 0)) {
        return(NULL)
      }
      c(one, two)
    })
    per_stratum <- do.call(rbind, per_stratum)
    if (is.null(per_stratum)) {
      stop(
        paste(
          "No stratum holds both groups with positive standard errors, so the stratified",
          "comparison is undefined."
        ),
        call. = FALSE
      )
    }
    w <- 1 / (per_stratum[, 2L]^2 + per_stratum[, 4L]^2)
    total <- sum(w)
    rmst1 <- sum(w * per_stratum[, 1L]) / total
    rmst2 <- sum(w * per_stratum[, 3L]) / total
    se1 <- sqrt(sum(w^2 * per_stratum[, 2L]^2)) / total
    se2 <- sqrt(sum(w^2 * per_stratum[, 4L]^2)) / total
  }

  z <- z_value(conf_level)
  if (estimand == "ratio") {
    if (!(rmst1 > 0 && rmst2 > 0)) {
      stop("The ratio needs both restricted means to be positive.", call. = FALSE)
    }
    estimate <- rmst1 / rmst2
    se_log <- sqrt((se1 / rmst1)^2 + (se2 / rmst2)^2)
    std_error <- estimate * se_log
    conf_low <- exp(log(estimate) - z * se_log)
    conf_high <- exp(log(estimate) + z * se_log)
    statistic <- log(estimate) / se_log
  } else {
    scale <- 1
    if (estimand == "percentage_difference") {
      if (!(rmst2 > 0)) {
        stop("The percentage difference needs a positive restricted mean in the second group.",
          call. = FALSE
        )
      }
      scale <- 100 / rmst2
    }
    estimate <- (rmst1 - rmst2) * scale
    std_error <- sqrt(se1^2 + se2^2) * scale
    conf_low <- estimate - z * std_error
    conf_high <- estimate + z * std_error
    statistic <- if (std_error > 0) estimate / std_error else Inf * sign(estimate)
  }

  method <- sprintf("RMST %s (tau = %s)", sub("_", " ", estimand), format(tau))
  if (!is.null(input$stratum)) {
    method <- paste("Stratified", method)
  }
  structure(
    list(
      estimate = estimate,
      std_error = std_error,
      conf_low = conf_low,
      conf_high = conf_high,
      statistic = statistic,
      p_value = 2 * stats::pnorm(-abs(statistic)),
      method = method,
      group1 = input$labels[[a]],
      group2 = input$labels[[b]],
      strata = input$keys$strata[c(a, b)],
      rmst1 = rmst1,
      se1 = se1,
      rmst2 = rmst2,
      se2 = se2,
      tau = tau,
      estimand = estimand,
      stratified = !is.null(input$stratum),
      conf_level = conf_level,
      formula = input$formula
    ),
    class = "rmst_test"
  )
}

#' @export
as.data.frame.rmst_test <- function(x, ...) {
  fields <- c(
    "method", "group1", "group2", "tau", "rmst1", "se1", "rmst2", "se2", "estimate",
    "std_error", "conf_low", "conf_high", "statistic", "p_value"
  )
  as.data.frame(x[fields], stringsAsFactors = FALSE)
}

#' @export
print.rmst_test <- function(x, digits = getOption("digits"), ...) {
  cat("<rmst_test> ", deparse_formula(x$formula), "\n", x$method, "\n\n", sep = "")
  groups <- data.frame(rmst = c(x$rmst1, x$rmst2), std_error = c(x$se1, x$se2))
  rownames(groups) <- x$strata
  print(groups, digits = digits)
  short <- max(3L, digits - 3L)
  label <- switch(x$estimand,
    difference = "Difference",
    ratio = "Ratio",
    percentage_difference = "Percentage difference"
  )
  cat(sprintf(
    "\n%s (%s vs %s): %s, %s%% CI [%s, %s]\nz = %s, p = %s\n",
    label, x$group1, x$group2,
    format(x$estimate, digits = short),
    format(100 * x$conf_level),
    format(x$conf_low, digits = short),
    format(x$conf_high, digits = short),
    format(x$statistic, digits = short),
    format_p(x$p_value)
  ))
  invisible(x)
}
