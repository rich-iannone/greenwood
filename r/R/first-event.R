# A competing-risks response from one pair of columns per endpoint: first_event(). Its contract is
# in spec/surv.md, and it's tested by tests/testthat/test-first-event.R.

#' Build a competing-risks response from several endpoints
#'
#' Some datasets record each endpoint in its own pair of columns rather than in one outcome
#' column. In mgus2, for example, `ptime` and `pstat` hold the time to plasma cell malignancy and
#' whether it occurred, and `futime` and `death` do the same for death. A competing-risks analysis
#' needs one time and one cause per subject: the first endpoint observed, or censoring if none
#' was. `first_event()` builds that response.
#'
#' For each row, the endpoint with the earliest observed event becomes the cause, and its time
#' becomes the event time. Ties go to the endpoint listed first. A row with no observed event is
#' censored at the latest of its endpoint times (the last time it was known to be event-free), or
#' at `censor_at` if given. A row with a missing endpoint time is missing.
#'
#' `first_event()` can be used on the left-hand side of a model formula, where its columns are
#' found in `data`, such as
#' `first_event(list(pcm = list(ptime, pstat), death = list(futime, death))) ~ age`.
#'
#' @param endpoints A named list with one element per endpoint, in priority order for ties. Each
#'   element is a list of two vectors of the same length: the time of the endpoint and whether it
#'   occurred (logical or `0`/`1`). The names are the states of the response.
#' @param censor_at The time at which rows with no observed event are censored, one number or one
#'   per row. The default is the latest of the endpoint times.
#' @param start Entry times for delayed entry, one per row. With them, the response is a
#'   multi-state counting-process response.
#'
#' @return A multi-state [Surv()] response whose states are the endpoint names, in order.
#'
#' @section Warnings:
#' A row whose event is recorded after another endpoint's follow-up had already ended without an
#' event gives a warning. This often means the time columns don't share an origin or a unit.
#'
#' @seealso [aalen_johansen()], [fine_gray()], and the other competing-risks models that take the
#'   response.
#'
#' @examples
#' # Subject 1 has endpoint "a" at time 5, subject 2 has "b" at time 8, and subject 3 is censored
#' # at time 9.
#' first_event(list(
#'   a = list(c(5, 9, 4), c(1, 0, 0)),
#'   b = list(c(7, 8, 9), c(0, 1, 0))
#' ))
#'
#' # In a formula, with the columns found in `data`
#' mgus <- data.frame(
#'   ptime = c(30, 45, 70, 72, 80, 95, 110, 120),
#'   pstat = c(1, 0, 0, 1, 0, 0, 1, 0),
#'   futime = c(40, 45, 70, 90, 80, 120, 130, 120),
#'   death = c(0, 1, 1, 0, 1, 0, 1, 0)
#' )
#' aalen_johansen(
#'   first_event(list(pcm = list(ptime, pstat), death = list(futime, death))) ~ 1,
#'   data = mgus
#' )
#' @export
first_event <- function(endpoints, censor_at = NULL, start = NULL) {
  labels <- names(endpoints)
  if (!is.list(endpoints) || length(endpoints) == 0L) {
    stop("`endpoints` must be a list that names at least one endpoint.", call. = FALSE)
  }
  if (is.null(labels) || anyNA(labels) || any(labels == "") || anyDuplicated(labels) > 0L) {
    stop("Every endpoint in `endpoints` must have its own name.", call. = FALSE)
  }
  times <- vector("list", length(endpoints))
  events <- vector("list", length(endpoints))
  for (k in seq_along(endpoints)) {
    pair <- endpoints[[k]]
    if (!is.list(pair) || length(pair) != 2L) {
      stop(sprintf("Endpoint \"%s\" must be a list of its time and event.", labels[k]),
        call. = FALSE
      )
    }
    if (!is.numeric(pair[[1L]])) {
      stop(sprintf("The time of endpoint \"%s\" isn't numeric.", labels[k]), call. = FALSE)
    }
    event <- pair[[2L]]
    if ((!is.logical(event) && !is.numeric(event)) || anyNA(event) ||
      !all(event %in% c(0, 1))) {
      stop(sprintf("The event of endpoint \"%s\" must be logical or 0/1.", labels[k]),
        call. = FALSE
      )
    }
    times[[k]] <- as.double(pair[[1L]])
    events[[k]] <- event == 1
  }
  n <- length(times[[1L]])
  if (any(lengths(times) != n) || any(lengths(events) != n)) {
    stop("All endpoint times and events must have the same length.", call. = FALSE)
  }

  stop_time <- rep(Inf, n)
  cause <- integer(n)
  latest <- rep(-Inf, n)
  for (k in seq_along(times)) {
    # A strict comparison, so a tie goes to the endpoint listed first.
    earlier <- events[[k]] & times[[k]] < stop_time
    earlier[is.na(earlier)] <- FALSE
    stop_time[earlier] <- times[[k]][earlier]
    cause[earlier] <- k
    latest <- pmax(latest, times[[k]])
  }
  observed <- cause > 0L
  if (!is.null(censor_at)) {
    if (!is.numeric(censor_at) || !(length(censor_at) %in% c(1L, n))) {
      stop("`censor_at` must be one number or one per row.", call. = FALSE)
    }
    censor_time <- rep_len(as.double(censor_at), n)
  } else {
    censor_time <- latest
  }
  stop_time[!observed] <- censor_time[!observed]
  incomplete <- Reduce(`|`, lapply(times, is.na))
  stop_time[incomplete] <- NA_real_
  warn_event_after_follow_up(labels, times, events, cause, stop_time)

  # The censoring level is never shown, so it only needs to differ from the endpoint names.
  censored <- paste0(".censored", strrep("_", max(nchar(labels))))
  status <- factor(c(censored, labels)[cause + 1L], levels = c(censored, labels))
  if (is.null(start)) {
    return(Surv(stop_time, status))
  }
  if (!is.numeric(start) || length(start) != n) {
    stop("`start` must be numbers, one per row.", call. = FALSE)
  }
  Surv(start, stop_time, status)
}

# A warning for rows whose event is recorded after another endpoint's follow-up had ended without
# an event.
warn_event_after_follow_up <- function(labels, times, events, cause, stop_time) {
  ended_early <- vapply(seq_along(times), function(k) {
    out <- !events[[k]] & times[[k]] < stop_time & cause > 0L
    out & !is.na(out)
  }, logical(length(stop_time)))
  ended_early <- matrix(ended_early, nrow = length(stop_time))
  bad <- rowSums(ended_early) > 0
  if (!any(bad)) {
    return(invisible())
  }
  row <- which(bad)[1L]
  other <- which(ended_early[row, ])[1L]
  warning(
    sprintf(
      paste(
        "%d %s an event recorded after another endpoint's follow-up had already ended. For",
        "example, row %d has \"%s\" at %s, but its \"%s\" follow-up stops at %s, so \"%s\" was",
        "not observed in between. Check that the endpoint time columns share an origin and a",
        "unit. If one endpoint was simply followed for less time, the response treats it as not",
        "having occurred before the recorded event."
      ),
      sum(bad), if (sum(bad) == 1L) "row has" else "rows have", row, labels[cause[row]],
      format(stop_time[row]), labels[other], format(times[[other]][row]), labels[other]
    ),
    call. = FALSE
  )
}
