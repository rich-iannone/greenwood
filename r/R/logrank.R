# Weighted log-rank tests: logrank_test(), pairwise_logrank_test(), trend_test(), and
# maxcombo_test(). The numeric fixtures (logrank_*, grho_*, pairwise_logrank_*, maxcombo_*) are
# replayed by tests/testthat/test-logrank.R.

#' Log-rank test for differences in survival
#'
#' `logrank_test()` tests whether survival differs between two or more groups. It compares the
#' number of events observed in each group with the number expected if every group had the same
#' hazard. With the default weights this is the log-rank test. Other weights give the
#' Fleming-Harrington \eqn{G^{\rho,\gamma}} family.
#'
#' At each event time \eqn{t}, the weight is \eqn{S(t-)^\rho (1 - S(t-))^\gamma}, where \eqn{S} is
#' the Kaplan-Meier estimate of the pooled sample just before \eqn{t}. `rho = 0, gamma = 0` (the
#' default) weights every event time equally. `rho = 1, gamma = 0` (the Peto-Peto test) gives more
#' weight to early times, and `rho = 0, gamma = 1` to late ones. The variance is the
#' hypergeometric one, and the statistic is the quadratic form of the observed minus expected
#' counts, which has a chi-square distribution with one fewer degrees of freedom than there are
#' groups.
#'
#' A `strata()` term on the right-hand side gives a stratified test: the observed and expected
#' counts and their variances are computed within each stratum (with the weights from that
#' stratum's own pooled curve) and summed before the statistic is formed.
#'
#' @param formula A formula with a [Surv()] response on the left and the grouping variables on the
#'   right, such as `Surv(time, status) ~ sex`. Several variables give one group per combination.
#'   A `strata()` term, such as `Surv(time, status) ~ sex + strata(inst)`, stratifies the test.
#' @param data A data frame holding the variables in `formula`.
#' @param weights Optional case weights, a column of `data` or a numeric vector.
#' @param subset An optional expression selecting the rows of `data` to use.
#' @param na.action What to do with rows that have missing values. The default drops them.
#' @param rho,gamma The Fleming-Harrington weight exponents, each at least 0. The default of 0 for
#'   both is the log-rank test.
#'
#' @return A `greenwood_test` object with the `statistic`, the degrees of freedom `df`, the
#'   `p_value`, a description of the test (`method`), and `groups`, a data frame with one row per
#'   group: `group` (the group's label), the grouping variables, `n` (the number of rows),
#'   `observed` and `expected` (the weighted event counts). [as.data.frame()] gives the test as a
#'   one-row data frame.
#'
#' @seealso [pairwise_logrank_test()] to compare every pair of groups, [trend_test()] for ordered
#'   groups, and [maxcombo_test()] when the hazards may not be proportional.
#'
#' @examples
#' lung <- data.frame(
#'   time = c(306, 455, 1010, 210, 883, 1022, 310, 361, 218, 166, 170, 654),
#'   status = c(2, 2, 1, 2, 2, 1, 2, 2, 2, 2, 2, 2),
#'   sex = c(1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2),
#'   ecog = c(0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1)
#' )
#'
#' logrank_test(Surv(time, status) ~ sex, data = lung)
#'
#' # The Peto-Peto test, which weights early differences more
#' logrank_test(Surv(time, status) ~ sex, data = lung, rho = 1)
#'
#' # Compare the sexes within each ECOG grade
#' logrank_test(Surv(time, status) ~ sex + strata(ecog), data = lung)
#' @export
logrank_test <- function(formula,
                         data,
                         weights,
                         subset,
                         na.action = stats::na.omit,
                         rho = 0,
                         gamma = 0) {
  check_number(rho, "rho", min = 0)
  check_number(gamma, "gamma", min = 0)
  frame <- gw_model_frame(
    match.call(expand.dots = FALSE), parent.frame(), formula, if (missing(data)) NULL else data
  )
  input <- lr_input(frame, "logrank_test()")
  sums <- lr_sums(input, list(c(rho, gamma)))
  observed <- sums$observed[, 1L]
  expected <- sums$expected[, 1L]
  test <- lr_chisq(observed, expected, sums$var[[1L]])

  method <- if (rho == 0 && gamma == 0) {
    "Log-rank test"
  } else {
    sprintf("G-rho test (rho = %s, gamma = %s)", format(rho), format(gamma))
  }
  lr_result(input, frame, test$statistic, test$df, method, observed, expected)
}

# Everything the tests need from a model frame: the response, the group of each row (an integer
# into `keys`), and the stratum of each row.
lr_input <- function(frame, fn, min_groups = 2L) {
  if (is.null(frame$vars)) {
    stop(
      sprintf("`%s` needs grouping variables on the right-hand side of `formula`.", fn),
      call. = FALSE
    )
  }
  parts <- surv_parts(frame$response, fn)
  groups <- group_keys(frame$vars, nrow(frame$mf))
  if (nrow(groups$keys) < min_groups) {
    stop(sprintf("`%s` needs at least %d groups.", fn, min_groups), call. = FALSE)
  }
  if (!any(parts$event == 1)) {
    stop(sprintf("`%s` needs at least one event.", fn), call. = FALSE)
  }
  stratum <- if (is.null(frame$strata)) {
    rep(1L, nrow(frame$mf))
  } else {
    group_keys(frame$strata, nrow(frame$mf))$index
  }
  list(
    start = parts$start,
    stop = parts$stop,
    event = parts$event,
    w = frame$weights,
    group = groups$index,
    keys = groups$keys,
    stratum = stratum,
    stratified = !is.null(frame$strata)
  )
}

# The rows of `input` in the given groups, with the group numbers renumbered 1..k.
lr_subset <- function(input, which) {
  rows <- input$group %in% which
  out <- input
  for (name in c("start", "stop", "event", "w", "stratum")) {
    out[[name]] <- input[[name]][rows]
  }
  out$group <- match(input$group[rows], which)
  out$keys <- input$keys[which, , drop = FALSE]
  out
}

# Observed and expected counts and the variance of their differences for each weight function in
# `weights` (a list of c(rho, gamma)), summed over strata. `observed` and `expected` are
# groups-by-weights matrices, `var` is a list of groups-by-groups matrices (one per weight), and
# `cov1` is the weights-by-weights covariance of the first group's observed minus expected.
lr_sums <- function(input, weights) {
  n_groups <- nrow(input$keys)
  n_weights <- length(weights)
  observed <- matrix(0, n_groups, n_weights)
  expected <- matrix(0, n_groups, n_weights)
  var <- rep(list(matrix(0, n_groups, n_groups)), n_weights)
  cov1 <- matrix(0, n_weights, n_weights)

  for (s in unique(input$stratum)) {
    rows <- input$stratum == s
    start <- input$start[rows]
    stop <- input$stop[rows]
    event <- input$event[rows]
    w <- input$w[rows]
    group <- input$group[rows]

    time <- sort(unique(stop[event == 1]))
    if (length(time) == 0L) {
      next
    }
    n_risk_g <- matrix(0, length(time), n_groups)
    n_event_g <- matrix(0, length(time), n_groups)
    for (g in seq_len(n_groups)) {
      in_g <- group == g
      if (!any(in_g)) {
        next
      }
      n_risk_g[, g] <- weight_at_risk(start[in_g], stop[in_g], w[in_g], time)
      died <- in_g & event == 1
      n_event_g[, g] <- tabulate_sum(match(stop[died], time), w[died], length(time))
    }
    n_risk <- rowSums(n_risk_g)
    n_event <- rowSums(n_event_g)
    rate <- ifelse(n_risk > 0, n_event / n_risk, 0)
    # The hypergeometric variance per unit weight. It is 0 when one subject is at risk.
    base <- ifelse(
      n_risk > 1, n_event * (n_risk - n_event) / (n_risk^2 * (n_risk - 1)), 0
    )
    pooled <- cumprod(ifelse(n_risk > 0, 1 - n_event / n_risk, 1))
    left <- c(1, pooled[-length(pooled)])

    fh <- vapply(weights, function(rg) left^rg[[1L]] * (1 - left)^rg[[2L]], numeric(length(time)))
    fh <- matrix(fh, nrow = length(time))
    for (k in seq_len(n_weights)) {
      observed[, k] <- observed[, k] + colSums(fh[, k] * n_event_g)
      expected[, k] <- expected[, k] + colSums(fh[, k] * n_risk_g * rate)
      common <- fh[, k]^2 * base
      var[[k]] <- var[[k]] + diag(colSums(common * n_risk * n_risk_g), n_groups) -
        crossprod(n_risk_g, common * n_risk_g)
    }
    spread <- base * n_risk_g[, 1L] * (n_risk - n_risk_g[, 1L])
    cov1 <- cov1 + crossprod(fh, spread * fh)
  }
  list(observed = observed, expected = expected, var = var, cov1 = cov1)
}

# The chi-square statistic for observed minus expected. Groups with nothing expected (no one at
# risk at any event time) carry no information and are left out, and one more group is dropped
# because the differences sum to zero.
lr_chisq <- function(observed, expected, var) {
  keep <- which(expected > 0)
  if (length(keep) < 2L) {
    stop("The test needs at least two groups with subjects at risk at an event time.", call. = FALSE)
  }
  keep <- keep[-1L]
  u <- (observed - expected)[keep]
  v <- var[keep, keep, drop = FALSE]
  list(statistic = sum(u * solve(v, u)), df = length(keep))
}

lr_result <- function(input, frame, statistic, df, method, observed, expected) {
  if (input$stratified) {
    method <- paste("Stratified", sub("^L", "l", method))
  }
  groups <- cbind(
    data.frame(group = input$keys$strata, stringsAsFactors = FALSE),
    input$keys[-1L],
    data.frame(
      n = as.numeric(tabulate(input$group, nrow(input$keys))),
      observed = observed,
      expected = expected
    )
  )
  structure(
    list(
      statistic = statistic,
      df = df,
      p_value = stats::pchisq(statistic, df, lower.tail = FALSE),
      method = method,
      groups = groups,
      n_dropped = frame$n_dropped,
      formula = frame$formula
    ),
    class = "greenwood_test"
  )
}

#' @export
as.data.frame.greenwood_test <- function(x, ...) {
  data.frame(
    method = x$method,
    statistic = x$statistic,
    df = x$df,
    p_value = x$p_value,
    stringsAsFactors = FALSE
  )
}

#' @export
print.greenwood_test <- function(x, digits = getOption("digits"), ...) {
  cat("<greenwood_test> ", deparse_formula(x$formula), "\n", x$method, "\n\n", sep = "")
  table <- x$groups[intersect(c("n", "observed", "expected", "score"), names(x$groups))]
  rownames(table) <- x$groups$group
  print(table, digits = digits)
  cat(sprintf(
    "\nChi-squared = %s on %d %s, p = %s\n",
    format(x$statistic, digits = max(3L, digits - 3L)),
    x$df,
    if (x$df == 1L) "degree of freedom" else "degrees of freedom",
    format_p(x$p_value)
  ))
  cat_dropped(x$n_dropped)
  invisible(x)
}

#' Log-rank tests between every pair of groups
#'
#' `pairwise_logrank_test()` runs [logrank_test()] on each pair of groups, using only the rows of
#' the two groups, and adjusts the p-values for the number of comparisons. It answers which groups
#' differ once a test across all groups has found that some do.
#'
#' Holm's method (the default) and Bonferroni's control the chance of any false positive among the
#' comparisons, Holm's with more power. The Benjamini-Hochberg method (`"bh"`) controls the
#' expected proportion of false positives among the comparisons called significant instead, and
#' is less conservative.
#'
#' @inheritParams logrank_test
#' @param correction The adjustment for multiple comparisons: `"holm"` (the default), `"bh"`,
#'   `"bonferroni"`, or `"none"`.
#'
#' @return A data frame with one row per pair of groups, in the order of the groups: `group1`,
#'   `group2` (the groups' labels), `statistic`, `p_value` (unadjusted), and `p_adjusted`.
#'
#' @examples
#' lung <- data.frame(
#'   time = c(306, 455, 1010, 210, 883, 1022, 310, 361, 218, 166, 170, 654),
#'   status = c(2, 2, 1, 2, 2, 1, 2, 2, 2, 2, 2, 2),
#'   ecog = c(0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2)
#' )
#' pairwise_logrank_test(Surv(time, status) ~ ecog, data = lung)
#' pairwise_logrank_test(Surv(time, status) ~ ecog, data = lung, correction = "bh")
#' @export
pairwise_logrank_test <- function(formula,
                                  data,
                                  weights,
                                  subset,
                                  na.action = stats::na.omit,
                                  rho = 0,
                                  gamma = 0,
                                  correction = c("holm", "bh", "bonferroni", "none")) {
  correction <- match.arg(correction)
  check_number(rho, "rho", min = 0)
  check_number(gamma, "gamma", min = 0)
  frame <- gw_model_frame(
    match.call(expand.dots = FALSE), parent.frame(), formula, if (missing(data)) NULL else data
  )
  input <- lr_input(frame, "pairwise_logrank_test()")
  labels <- lr_group_labels(input$keys)
  n_groups <- nrow(input$keys)
  pairs <- rbind(
    rep(seq_len(n_groups), rev(seq_len(n_groups) - 1L)),
    unlist(lapply(seq_len(n_groups), function(g) seq_len(n_groups)[-seq_len(g)]))
  )
  statistic <- apply(pairs, 2L, function(pair) {
    pair_input <- lr_subset(input, pair)
    if (!any(pair_input$event == 1)) {
      return(NA_real_)
    }
    sums <- lr_sums(pair_input, list(c(rho, gamma)))
    lr_chisq(sums$observed[, 1L], sums$expected[, 1L], sums$var[[1L]])$statistic
  })
  p_value <- stats::pchisq(statistic, 1, lower.tail = FALSE)
  data.frame(
    group1 = labels[pairs[1L, ]],
    group2 = labels[pairs[2L, ]],
    statistic = statistic,
    p_value = p_value,
    p_adjusted = adjust_p(p_value, correction),
    stringsAsFactors = FALSE
  )
}

# Plain group labels for a table of pairs: the value itself with one grouping variable, such as
# "adeno", and the values joined with ", " with several.
lr_group_labels <- function(keys) {
  values <- keys[-1L]
  do.call(paste, c(lapply(values, as.character), sep = ", "))
}

#' Log-rank test for a trend across ordered groups
#'
#' `trend_test()` tests for a monotone trend in the hazard across groups that have a natural
#' order, such as disease stages or dose levels. Each group gets a numeric score, and the test
#' asks whether the observed minus expected event counts line up with the scores. When the groups
#' are ordered, it has more power than [logrank_test()], which looks for any difference.
#'
#' With scores \eqn{c}, observed minus expected counts \eqn{U}, and their variance \eqn{V} (as in
#' [logrank_test()], with the same `rho`, `gamma`, and strata), the statistic is
#' \eqn{(c^\top U)^2 / c^\top V c}, which has a chi-square distribution with 1 degree of freedom.
#' Without ties it equals the score test for the scores as a covariate in a Cox model. Adding a
#' constant to every score, or multiplying them all by one, doesn't change the test.
#'
#' @inheritParams logrank_test
#' @param formula A formula with a [Surv()] response on the left and one grouping variable on the
#'   right, such as `Surv(time, status) ~ ph.ecog`. A `strata()` term stratifies the test.
#' @param scores The score of each group, in the order of the groups (factor levels or sorted
#'   values), or named by the group values. The default gives the groups the scores 0, 1, 2, and so
#'   on.
#'
#' @return A `greenwood_test` object, as from [logrank_test()], with 1 degree of freedom. Its
#'   `groups` table also has the `score` of each group.
#'
#' @examples
#' lung <- data.frame(
#'   time = c(306, 455, 1010, 210, 883, 1022, 310, 361, 218, 166, 170, 654),
#'   status = c(2, 2, 1, 2, 2, 1, 2, 2, 2, 2, 2, 2),
#'   ecog = c(0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2)
#' )
#' trend_test(Surv(time, status) ~ ecog, data = lung)
#'
#' # Scores that grow faster from one grade to the next
#' trend_test(Surv(time, status) ~ ecog, data = lung, scores = c(0, 1, 4))
#' @export
trend_test <- function(formula,
                       data,
                       weights,
                       subset,
                       na.action = stats::na.omit,
                       scores = NULL,
                       rho = 0,
                       gamma = 0) {
  check_number(rho, "rho", min = 0)
  check_number(gamma, "gamma", min = 0)
  frame <- gw_model_frame(
    match.call(expand.dots = FALSE), parent.frame(), formula, if (missing(data)) NULL else data
  )
  if (!is.null(frame$vars) && ncol(frame$vars) > 1L) {
    stop("`trend_test()` needs one grouping variable on the right-hand side.", call. = FALSE)
  }
  input <- lr_input(frame, "trend_test()")
  scores <- trend_scores(scores, lr_group_labels(input$keys))

  sums <- lr_sums(input, list(c(rho, gamma)))
  observed <- sums$observed[, 1L]
  expected <- sums$expected[, 1L]
  u <- sum(scores * (observed - expected))
  v <- drop(crossprod(scores, sums$var[[1L]] %*% scores))
  if (!(v > 0)) {
    stop("The variance of the trend statistic is 0, so the test is undefined.", call. = FALSE)
  }

  method <- if (rho == 0 && gamma == 0) {
    "Log-rank test for trend"
  } else {
    sprintf("G-rho test for trend (rho = %s, gamma = %s)", format(rho), format(gamma))
  }
  out <- lr_result(input, frame, u^2 / v, 1L, method, observed, expected)
  out$groups$score <- scores
  out
}

trend_scores <- function(scores, labels) {
  if (is.null(scores)) {
    return(seq_along(labels) - 1)
  }
  if (!is.numeric(scores) || anyNA(scores) || any(!is.finite(scores))) {
    stop("`scores` must be numbers.", call. = FALSE)
  }
  if (!is.null(names(scores))) {
    missing <- setdiff(labels, names(scores))
    if (length(missing) > 0L) {
      stop(
        sprintf("`scores` has no score for %s.", paste0("\"", missing, "\"", collapse = ", ")),
        call. = FALSE
      )
    }
    return(unname(scores[labels]))
  }
  if (length(scores) != length(labels)) {
    stop(
      sprintf("`scores` must have one score per group (%d), not %d.", length(labels), length(scores)),
      call. = FALSE
    )
  }
  as.numeric(scores)
}

#' MaxCombo test for differences in survival
#'
#' `maxcombo_test()` compares the survival of two groups with several Fleming-Harrington weighted
#' log-rank tests at once and takes the largest absolute standardized statistic. It keeps good
#' power whether the difference between the groups is early, late, or constant over time, which
#' makes it a common choice when the hazards may not be proportional, such as with a delayed
#' treatment effect.
#'
#' Each pair of `rho[i]` and `gamma[i]` defines one weight function, as in [logrank_test()]. The
#' default four are the log-rank test (0, 0), early emphasis (1, 0), late emphasis (0, 1), and
#' middle emphasis (1, 1). The standardized statistics \eqn{Z_i} for the first group are jointly
#' normal under the null hypothesis, with a correlation that follows from the shared risk sets.
#' The p-value is \eqn{P(\max_i |Z_i| \ge T)} for the observed maximum \eqn{T}, computed from that
#' multivariate normal distribution by deterministic quasi-Monte Carlo integration (accurate to
#' about 4 decimal places).
#'
#' @inheritParams logrank_test
#' @param formula A formula with a [Surv()] response on the left and a grouping variable with two
#'   groups on the right, such as `Surv(time, status) ~ sex`. A `strata()` term stratifies each of
#'   the tests.
#' @param rho,gamma Numeric vectors of the same length, at least 0, giving the weight functions.
#'
#' @return A `maxcombo_test` object with the `statistic` (the largest \eqn{|Z_i|}), the `p_value`,
#'   `z` (a data frame of `rho`, `gamma`, and `z` for each weight function), `correlation` (the
#'   correlation matrix of the statistics), and `method`. [as.data.frame()] gives the test as a
#'   one-row data frame.
#'
#' @references Lee, S.-H. (2007). On the versatility of the combination of the weighted log-rank
#'   statistics. *Computational Statistics & Data Analysis*, 51(12), 6557-6564.
#'
#' @examples
#' lung <- data.frame(
#'   time = c(306, 455, 1010, 210, 883, 1022, 310, 361, 218, 166, 170, 654),
#'   status = c(2, 2, 1, 2, 2, 1, 2, 2, 2, 2, 2, 2),
#'   sex = c(1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2)
#' )
#' maxcombo_test(Surv(time, status) ~ sex, data = lung)
#'
#' # Only the log-rank and late-emphasis weights
#' maxcombo_test(Surv(time, status) ~ sex, data = lung, rho = c(0, 0), gamma = c(0, 1))
#' @export
maxcombo_test <- function(formula,
                          data,
                          weights,
                          subset,
                          na.action = stats::na.omit,
                          rho = c(0, 1, 0, 1),
                          gamma = c(0, 0, 1, 1)) {
  if (!is.numeric(rho) || !is.numeric(gamma) || length(rho) == 0L ||
    length(rho) != length(gamma) || anyNA(c(rho, gamma)) || any(c(rho, gamma) < 0) ||
    any(!is.finite(c(rho, gamma)))) {
    stop("`rho` and `gamma` must be non-negative numbers of the same length.", call. = FALSE)
  }
  frame <- gw_model_frame(
    match.call(expand.dots = FALSE), parent.frame(), formula, if (missing(data)) NULL else data
  )
  input <- lr_input(frame, "maxcombo_test()")
  if (nrow(input$keys) != 2L) {
    stop("`maxcombo_test()` needs exactly two groups.", call. = FALSE)
  }

  sums <- lr_sums(input, Map(c, rho, gamma))
  u <- sums$observed[1L, ] - sums$expected[1L, ]
  sd <- sqrt(diag(sums$cov1))
  z <- ifelse(sd > 0, u / sd, 0)
  correlation <- sums$cov1 / outer(sd, sd)
  correlation[!is.finite(correlation)] <- 0
  diag(correlation) <- 1
  names <- paste0("(", rho, ", ", gamma, ")")
  dimnames(correlation) <- list(names, names)

  statistic <- max(abs(z))
  p_value <- max(0, 1 - pmvnorm_box(statistic, correlation))
  method <- paste0("MaxCombo test of ", paste(names, collapse = ", "))
  if (input$stratified) {
    method <- paste("Stratified", method)
  }
  structure(
    list(
      statistic = statistic,
      p_value = p_value,
      z = data.frame(rho = rho, gamma = gamma, z = z),
      correlation = correlation,
      method = method,
      groups = lr_group_labels(input$keys),
      n_dropped = frame$n_dropped,
      formula = frame$formula
    ),
    class = "maxcombo_test"
  )
}

# P(|Z_i| <= c for all i) for Z ~ N(0, R), by Genz's separation of variables. The unit cube is
# covered by a rank-1 (Richtmyer) lattice with fixed shifts and the baker's transform, so the
# result is deterministic and the random number generator is left alone.
pmvnorm_box <- function(c, corr, n_points = 4096L, n_shifts = 16L) {
  k <- nrow(corr)
  if (k == 1L) {
    return(2 * stats::pnorm(c) - 1)
  }
  # A tiny ridge keeps the Cholesky factor defined for perfectly correlated statistics.
  eig <- min(eigen(corr, symmetric = TRUE, only.values = TRUE)$values)
  chol_l <- t(chol(corr + diag(max(0, 1e-10 - eig), k)))

  primes <- c(2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71)
  if (k - 1L > length(primes)) {
    stop("The MaxCombo p-value supports at most 21 weight functions.", call. = FALSE)
  }
  generator <- sqrt(primes[seq_len(k - 1L)])
  j <- seq_len(n_points)
  estimates <- vapply(seq_len(n_shifts), function(r) {
    shift <- (r * sqrt(c(97, 101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163,
      167, 173, 179, 181, 191, 193)[seq_len(k - 1L)])) %% 1
    u <- (outer(j, generator) + matrix(shift, n_points, k - 1L, byrow = TRUE)) %% 1
    u <- abs(2 * u - 1)
    lower <- stats::pnorm(-c / chol_l[1L, 1L])
    upper <- stats::pnorm(c / chol_l[1L, 1L])
    f <- rep(upper - lower, n_points)
    y <- matrix(0, n_points, k - 1L)
    d <- rep(lower, n_points)
    e <- rep(upper, n_points)
    for (i in 2:k) {
      p <- pmin(pmax(d + u[, i - 1L] * (e - d), 1e-300), 1 - 1e-16)
      y[, i - 1L] <- stats::qnorm(p)
      s <- drop(y[, seq_len(i - 1L), drop = FALSE] %*% chol_l[i, seq_len(i - 1L)])
      d <- stats::pnorm((-c - s) / chol_l[i, i])
      e <- stats::pnorm((c - s) / chol_l[i, i])
      f <- f * (e - d)
    }
    mean(f)
  }, numeric(1))
  mean(estimates)
}

#' @export
as.data.frame.maxcombo_test <- function(x, ...) {
  data.frame(
    method = x$method,
    statistic = x$statistic,
    p_value = x$p_value,
    stringsAsFactors = FALSE
  )
}

#' @export
print.maxcombo_test <- function(x, digits = getOption("digits"), ...) {
  cat("<maxcombo_test> ", deparse_formula(x$formula), "\n", x$method, "\n\n", sep = "")
  print(x$z, digits = digits, row.names = FALSE)
  cat(sprintf(
    "\nmax |z| = %s, p = %s\n",
    format(x$statistic, digits = max(3L, digits - 3L)),
    format_p(x$p_value)
  ))
  cat_dropped(x$n_dropped)
  invisible(x)
}
