# Sample size and power for the two-group log-rank test, from Schoenfeld's formula. Tested by
# tests/testthat/test-power.R against values from the Python package and the closed forms.

#' Sample size and power for the log-rank test
#'
#' Under proportional hazards, the power of the two-group log-rank test depends on the data only
#' through the total number of events (Schoenfeld, 1981). The number of events needed to detect a
#' hazard ratio \eqn{HR} is
#' \deqn{d = \frac{(z_{1 - \alpha / s} + z_{\mathrm{power}})^2}{p (1 - p) (\log HR)^2},}
#' where \eqn{s} is the number of sides of the test and \eqn{p} the fraction of subjects in one
#' group. A hazard ratio and its inverse need the same number of events.
#'
#' `logrank_n_events()` gives \eqn{d}, rounded up. `logrank_power()` inverts the formula, giving
#' the power \eqn{\Phi(\sqrt{d p (1 - p)} |\log HR| - z_{1 - \alpha / s})} reached with `n_events`
#' events. `logrank_sample_size()` divides \eqn{d} by the probability that a subject has the event
#' during follow-up, then rounds up, giving the number of subjects to enroll.
#'
#' @param hazard_ratio The hazard ratio to detect, positive and not 1.
#' @param n_events The total number of events, across both groups.
#' @param prob_event The probability that a subject has the event during the study, greater than 0
#'   and at most 1.
#' @param power The target power, between 0 and 1. The default is `0.8`.
#' @param alpha The significance level, between 0 and 1. The default is `0.05`.
#' @param allocation The fraction of subjects in one of the groups, between 0 and 1. The default,
#'   `0.5`, is a balanced design, which needs the fewest events.
#' @param sides `2` (the default) for a two-sided test or `1` for a one-sided test.
#'
#' @return `logrank_n_events()` and `logrank_sample_size()` give a whole number (an integer).
#'   `logrank_power()` gives the power, between 0 and 1.
#'
#' @references Schoenfeld, D. A. (1981). The asymptotic properties of nonparametric tests for
#'   comparing survival distributions. *Biometrika*, 68(1), 316-319.
#'
#' @examples
#' # Events needed to detect a halving of the hazard with 80% power
#' logrank_n_events(hazard_ratio = 0.5)
#' logrank_n_events(hazard_ratio = 0.5, power = 0.9, allocation = 1 / 3)
#'
#' # The power reached with 60 events
#' logrank_power(hazard_ratio = 0.5, n_events = 60)
#'
#' # Subjects to enroll when 40% of them are expected to have the event
#' logrank_sample_size(hazard_ratio = 0.5, prob_event = 0.4, power = 0.9)
#' @export
logrank_n_events <- function(hazard_ratio, power = 0.8, alpha = 0.05, allocation = 0.5, sides = 2) {
  check_power_design(hazard_ratio, alpha, allocation, sides)
  check_probability(power, "power")
  as.integer(ceiling(schoenfeld_events(hazard_ratio, power, alpha, allocation, sides)))
}

#' @rdname logrank_n_events
#' @export
logrank_power <- function(hazard_ratio, n_events, alpha = 0.05, allocation = 0.5, sides = 2) {
  check_power_design(hazard_ratio, alpha, allocation, sides)
  check_number(n_events, "n_events")
  if (!(n_events > 0)) {
    stop("`n_events` must be positive.", call. = FALSE)
  }
  z_alpha <- stats::qnorm(1 - alpha / sides)
  stats::pnorm(sqrt(n_events * allocation * (1 - allocation)) * abs(log(hazard_ratio)) - z_alpha)
}

#' @rdname logrank_n_events
#' @export
logrank_sample_size <- function(hazard_ratio,
                                prob_event,
                                power = 0.8,
                                alpha = 0.05,
                                allocation = 0.5,
                                sides = 2) {
  check_power_design(hazard_ratio, alpha, allocation, sides)
  check_probability(power, "power")
  check_number(prob_event, "prob_event")
  if (!(prob_event > 0 && prob_event <= 1)) {
    stop("`prob_event` must be greater than 0 and at most 1.", call. = FALSE)
  }
  events <- schoenfeld_events(hazard_ratio, power, alpha, allocation, sides)
  as.integer(ceiling(events / prob_event))
}

# The unrounded Schoenfeld number of events.
schoenfeld_events <- function(hazard_ratio, power, alpha, allocation, sides) {
  z <- stats::qnorm(1 - alpha / sides) + stats::qnorm(power)
  z^2 / (allocation * (1 - allocation) * log(hazard_ratio)^2)
}

check_power_design <- function(hazard_ratio, alpha, allocation, sides) {
  check_number(hazard_ratio, "hazard_ratio")
  if (!(hazard_ratio > 0) || hazard_ratio == 1) {
    stop("`hazard_ratio` must be positive and not 1.", call. = FALSE)
  }
  check_probability(alpha, "alpha")
  check_probability(allocation, "allocation")
  if (!identical(sides, 1) && !identical(sides, 2) && !identical(sides, 1L) && !identical(sides, 2L)) {
    stop("`sides` must be 1 or 2.", call. = FALSE)
  }
  invisible(NULL)
}

check_probability <- function(x, name) {
  check_number(x, name)
  if (!(x > 0 && x < 1)) {
    stop(sprintf("`%s` must be between 0 and 1.", name), call. = FALSE)
  }
  invisible(x)
}
