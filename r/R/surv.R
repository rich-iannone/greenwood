# The survival response. Its contract is spec/surv.md, and the cases in spec/conformance/surv/ are
# replayed by tests/testthat/test-surv-conformance.R.

SURV_TYPES <- c("right", "left", "interval", "counting", "interval2", "mstate")
WRONG_ARGS <- "Wrong number of args for this type of survival data"

#' Create a survival response
#'
#' `Surv()` builds the response for a survival model from event times and an event indicator,
#' typically on the left-hand side of a formula such as `Surv(time, status == 2) ~ age`.
#'
#' The type is inferred from the arguments when `type` isn't given:
#'
#' - `time` only: right-censored, with every row an event.
#' - `time` and `event`: right-censored.
#' - `time`, `time2`, and `event`: counting process, with `time` the start and `time2` the stop.
#' - A factor `event`: multi-state, with the first level as censoring.
#'
#' Status may be logical, `0`/`1`, or `1`/`2`. When the largest status is `2`, one is subtracted,
#' so `1`/`2` coding becomes `0`/`1`. Any other value becomes missing with a warning.
#'
#' Missing values are kept rather than dropped. Missing inputs, invalid status codes, start times
#' that are not before their stop times, and backwards intervals all give missing rows.
#'
#' @param time For right-censored data, the follow-up time. For interval data, the first time
#'   (the lower bound). For counting-process data, the start of the interval.
#' @param time2 The stop time for counting-process data, or the upper bound for interval data.
#'   With two arguments and no `type`, the second is taken as the status, so `Surv(time, status)`
#'   works positionally.
#' @param event The status indicator: logical, `0`/`1`, or `1`/`2` for right, left, and counting
#'   data, codes `0` (right), `1` (exact), `2` (left), and `3` (interval) for `type = "interval"`,
#'   or a factor for multi-state data.
#' @param type One of `"right"`, `"left"`, `"interval"`, `"counting"`, `"interval2"`, or
#'   `"mstate"`. Inferred from the arguments when not given. With `"interval2"`, `time` and `time2`
#'   are the bounds, and a missing or infinite bound is open. With `"mstate"`, the status is
#'   converted to a factor with sorted levels.
#' @param origin A value subtracted from every time.
#'
#' @return A `Surv` object: a numeric matrix with columns `time` and `status` (right, left, and
#'   multi-state), `start`, `stop`, and `status` (counting process), or `time1`, `time2`, and
#'   `status` (interval), with the type in `attr(, "type")` and the state names of multi-state data
#'   in `attr(, "states")`.
#'
#' @examples
#' Surv(c(5, 6, 4, 9), c(1, 0, 1, 0))
#'
#' # 1/2 status coding, as in the lung data, becomes 0/1
#' Surv(c(5, 6, 4), c(2, 1, 2))
#'
#' # Counting process (start, stop]
#' Surv(c(0, 2, 1), c(5, 6, 4), c(1, 0, 1))
#'
#' # Interval censoring from bounds, with open bounds as NA
#' Surv(c(1, 2, NA), c(1, NA, 4), type = "interval2")
#'
#' # Multi-state, with the first level as censoring
#' Surv(c(5, 6, 7), factor(c("censor", "pcm", "death"), levels = c("censor", "pcm", "death")))
#' @export
Surv <- function(time, time2, event, type, origin = 0) {
  if (!missing(type)) {
    type <- match.arg(type, SURV_TYPES)
  }
  if (!is.numeric(time)) {
    stop("Time variable is not numeric", call. = FALSE)
  }
  time <- as.double(time)
  n_args <- 1L + (!missing(time2)) + (!missing(event))
  mstate <- !missing(type) && identical(type, "mstate")

  if (missing(type) || mstate) {
    resolved <- if (n_args == 3L) "counting" else "right"
  } else {
    resolved <- type
    if (n_args != 3L && resolved %in% c("interval", "counting")) {
      stop(WRONG_ARGS, call. = FALSE)
    }
    if (n_args != 2L && resolved %in% c("right", "left", "interval2")) {
      stop(WRONG_ARGS, call. = FALSE)
    }
  }

  if (n_args == 1L) {
    return(new_surv(list(time = time - origin, status = rep(1, length(time))), "right"))
  }
  # With two arguments, the second is the status (or, for interval2, the upper bound) whichever
  # name it was passed under.
  second <- if (n_args == 2L) {
    if (missing(time2)) event else time2
  }

  switch(
    resolved,
    right = ,
    left = surv_right_left(time, second, resolved, mstate, origin),
    counting = surv_counting(time, time2, event, mstate, origin),
    interval = surv_interval(time, time2, event, origin),
    interval2 = surv_interval2(time, second, origin)
  )
}

surv_right_left <- function(time, event, type, mstate, origin) {
  if (length(event) != length(time)) {
    stop("Time and status are different lengths", call. = FALSE)
  }
  if (mstate || is.factor(event)) {
    states <- surv_states(event)
    return(new_surv(
      list(time = time - origin, status = states$codes),
      "mright",
      states$states
    ))
  }
  status <- surv_status(event, "Invalid status value, must be logical or numeric")
  new_surv(list(time = time - origin, status = status), type)
}

surv_counting <- function(start, stop, event, mstate, origin) {
  n <- length(start)
  if (length(stop) != n) {
    stop("Start and stop are different lengths", call. = FALSE)
  }
  if (length(event) != n) {
    stop("Start and event are different lengths", call. = FALSE)
  }
  if (!is.numeric(stop)) {
    stop("Stop time is not numeric", call. = FALSE)
  }
  stop <- as.double(stop)
  backwards <- which(start >= stop)
  if (length(backwards) > 0) {
    start[backwards] <- NA_real_
    warning("Stop time must be > start time, NA created", call. = FALSE)
  }
  if (mstate || is.factor(event)) {
    states <- surv_states(event)
    return(new_surv(
      list(start = start - origin, stop = stop - origin, status = states$codes),
      "mcounting",
      states$states
    ))
  }
  status <- surv_status(event, "Invalid status value")
  new_surv(list(start = start - origin, stop = stop - origin, status = status), "counting")
}

surv_interval <- function(time, time2, event, origin) {
  n <- length(time)
  if (length(event) != n) {
    stop("Time and status are different lengths", call. = FALSE)
  }
  if (!is.numeric(event)) {
    stop("Invalid status value, must be logical or numeric", call. = FALSE)
  }
  codes <- as.double(event)
  valid <- !is.na(codes) & codes %in% c(0, 1, 2, 3)
  if (any(!is.na(codes) & !valid)) {
    warning("Status must be 0, 1, 2 or 3; converted to NA", call. = FALSE)
  }
  status <- ifelse(valid, codes, NA_real_)

  if (any(codes == 3, na.rm = TRUE)) {
    if (!is.numeric(time2)) {
      stop("Time2 must be numeric", call. = FALSE)
    }
    if (length(time2) != n) {
      stop("time and time2 are different lengths", call. = FALSE)
    }
    time2 <- as.double(time2)
    backwards <- which(status == 3 & time > time2)
    if (length(backwards) > 0) {
      status[backwards] <- NA_real_
      warning("Invalid interval: start > stop, NA created", call. = FALSE)
    }
  } else {
    time2 <- rep(1, n)
  }
  surv_interval_columns(time, time2, status, origin)
}

surv_interval2 <- function(time, time2, origin) {
  n <- length(time)
  if (!is.numeric(time2)) {
    stop("Time2 must be numeric", call. = FALSE)
  }
  if (length(time2) != n) {
    stop("time and time2 are different lengths", call. = FALSE)
  }
  time2 <- as.double(time2)
  backwards <- which(time > time2)

  # Missing or infinite bounds are open.
  lower <- ifelse(is.finite(time), time, NA_real_)
  upper <- ifelse(is.finite(time2), time2, NA_real_)
  unknown <- is.na(lower) & is.na(upper)
  status <- ifelse(
    is.na(lower),
    2,
    ifelse(is.na(upper), 0, ifelse(lower == upper, 1, 3))
  )
  # A left-censored row stores its upper bound as the time.
  lower <- ifelse(status != 2, lower, upper)

  if (length(backwards) > 0) {
    warning("Invalid interval: start > stop, NA created", call. = FALSE)
    status[backwards] <- NA_real_
  }
  status[unknown] <- NA_real_
  surv_interval_columns(lower, upper, status, origin)
}

# The `time2` column holds 1 for rows that aren't interval-censored.
surv_interval_columns <- function(time, time2, status, origin) {
  is_interval <- !is.na(status) & status == 3
  new_surv(
    list(
      time1 = time - origin,
      time2 = ifelse(is_interval, time2 - origin, 1),
      status = status
    ),
    "interval"
  )
}

# Status for right, left, and counting data. Logical status becomes 0/1. Numeric status that peaks
# at 2 is shifted down by one. Anything other than 0 or 1 becomes missing, with a warning.
surv_status <- function(event, message) {
  if (is.logical(event)) {
    return(as.double(event))
  }
  if (!is.numeric(event)) {
    stop(message, call. = FALSE)
  }
  status <- as.double(event)
  observed <- !is.na(status)
  # Guard max() so an all-missing status doesn't raise R's "no non-missing arguments" warning,
  # which comes from max() rather than from the contract.
  if (any(observed) && max(status[observed]) == 2) {
    status <- status - 1
  }
  valid <- observed & (status == 0 | status == 1)
  if (any(observed & !valid)) {
    warning("Invalid status value, converted to NA", call. = FALSE)
  }
  status[!valid] <- NA_real_
  status
}

# Multi-state status: 0-based codes of a factor, whose first level is censoring. Non-factor input
# (type = "mstate") becomes a factor with levels in sorted order. Sorting uses the C locale
# (radix), so levels match the Python package's on every platform.
surv_states <- function(event) {
  if (is.factor(event)) {
    levels <- levels(event)
    codes <- as.integer(event) - 1L
  } else {
    present <- sort(unique(event[!is.na(event)]), method = "radix")
    levels <- as.character(present)
    codes <- match(event, present) - 1L
  }
  states <- levels[-1]
  if (any(states == "")) {
    stop("each state must have a non-blank name", call. = FALSE)
  }
  list(codes = as.double(codes), states = states)
}

new_surv <- function(columns, type, states = NULL) {
  x <- matrix(
    unlist(columns, use.names = FALSE),
    ncol = length(columns),
    dimnames = list(NULL, names(columns))
  )
  storage.mode(x) <- "double"
  attr(x, "type") <- type
  if (!is.null(states)) {
    attr(x, "states") <- states
  }
  class(x) <- c("greenwood_surv", "Surv")
  x
}

# -- methods --------------------------------------------------------------------------------------
#
# Methods dispatch on the "greenwood_surv" subclass, so they don't replace other packages' methods
# for the "Surv" class.

#' @export
length.greenwood_surv <- function(x) {
  nrow(unclass(x))
}

#' @export
is.na.greenwood_surv <- function(x) {
  rowSums(is.na(unclass(x))) > 0
}

#' @export
`[.greenwood_surv` <- function(x, i, j, drop = FALSE) {
  m <- unclass(x)
  if (!missing(j)) {
    # Selecting columns gives a plain matrix, or a vector for one column unless `drop = FALSE`
    # is given.
    attr(m, "type") <- NULL
    attr(m, "states") <- NULL
    drop <- if (missing(drop)) TRUE else drop
    return(if (missing(i)) m[, j, drop = drop] else m[i, j, drop = drop])
  }
  rows <- if (missing(i)) m else m[i, , drop = FALSE]
  new_surv(
    lapply(colnames(m), function(name) rows[, name]) |> stats::setNames(colnames(m)),
    attr(x, "type"),
    attr(x, "states")
  )
}

#' @export
format.greenwood_surv <- function(x, ...) {
  m <- unclass(x)
  type <- attr(x, "type")
  status <- m[, "status"]
  num <- function(v) ifelse(is.na(v), NA_character_, format(v, trim = TRUE, ...))

  out <- switch(
    type,
    right = paste0(num(m[, "time"]), ifelse(status == 0, "+", "")),
    left = paste0(num(m[, "time"]), ifelse(status == 0, "-", "")),
    mright = paste0(
      num(m[, "time"]),
      ifelse(status == 0, "+", paste0(":", attr(x, "states")[pmax(status, 1)]))
    ),
    counting = paste0(
      "(", num(m[, "start"]), ",", num(m[, "stop"]), ifelse(status == 0, "+", ""), "]"
    ),
    mcounting = paste0(
      "(", num(m[, "start"]), ",", num(m[, "stop"]),
      ifelse(status == 0, "+", paste0(":", attr(x, "states")[pmax(status, 1)])), "]"
    ),
    interval = ifelse(
      status == 3,
      paste0("[", num(m[, "time1"]), ", ", num(m[, "time2"]), "]"),
      paste0(num(m[, "time1"]), c("+", "", "-", "")[status + 1])
    )
  )
  out[is.na(x)] <- NA_character_
  out
}

#' @export
print.greenwood_surv <- function(x, ...) {
  header <- sprintf("<Surv[%d]: %s>", length(x), attr(x, "type"))
  if (!is.null(attr(x, "states"))) {
    header <- sprintf("%s states: %s", header, paste(attr(x, "states"), collapse = ", "))
  }
  cat(header, "\n", sep = "")
  if (length(x) > 0) {
    print(format(x), quote = FALSE)
  }
  invisible(x)
}

#' @export
as.character.greenwood_surv <- function(x, ...) {
  format(x, ...)
}
