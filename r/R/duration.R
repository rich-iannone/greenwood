# Elapsed time between two dates: duration(). It's tested by tests/testthat/test-duration.R.

UNIT_SECONDS <- c(
  seconds = 1,
  minutes = 60,
  hours = 3600,
  days = 86400,
  weeks = 7 * 86400,
  months = 30.4375 * 86400,
  years = 365.25 * 86400
)

#' Elapsed time between two dates
#'
#' Study data often record calendar dates (enrollment, last contact, death) rather than follow-up
#' times. `duration()` gives the time from `start` to `end` in the chosen unit, so it can be the
#' time of a response, such as `Surv(duration(enroll, exit, unit = "years"), died) ~ 1`.
#'
#' Dates are taken as midnight UTC, and character values are read as ISO 8601 dates or
#' date-times (such as `"2021-03-01"` or `"2021-03-01 14:30:00"`) in UTC. Months and years have
#' their average lengths, 30.4375 and 365.25 days.
#'
#' @param start The origin of each time, such as the enrollment date. A `Date`, a date-time
#'   (`POSIXct` or `POSIXlt`), or character dates.
#' @param end The end of each time, such as the date of the event or of last contact, of the same
#'   length as `start` (or length one).
#' @param unit The unit of the result: `"days"` (the default), `"weeks"`, `"months"`, `"years"`,
#'   `"hours"`, `"minutes"`, or `"seconds"`.
#'
#' @return A numeric vector of elapsed times, missing where either date is missing.
#'
#' @section Errors:
#' An `end` earlier than its `start` is an error, since a negative time can't be a survival time.
#' Numbers are an error too. If they're already elapsed times, use them directly.
#'
#' @seealso [Surv()] and [event_time()], which take the result as their time.
#'
#' @examples
#' study <- data.frame(
#'   enroll = as.Date(c("2021-03-01", "2021-03-15", "2021-04-01")),
#'   exit = as.Date(c("2022-01-10", "2021-09-22", "2023-01-01")),
#'   died = c(TRUE, FALSE, TRUE)
#' )
#' duration(study$enroll, study$exit, unit = "years")
#'
#' # In a formula, with the columns found in `data`
#' kaplan_meier(Surv(duration(enroll, exit, unit = "years"), died) ~ 1, data = study)
#' @export
duration <- function(start, end, unit = "days") {
  start_name <- deparse1(substitute(start))
  end_name <- deparse1(substitute(end))
  if (!is.character(unit) || length(unit) != 1L || !unit %in% names(UNIT_SECONDS)) {
    stop(
      sprintf(
        "`unit` must be one of %s.",
        paste0("\"", names(UNIT_SECONDS), "\"", collapse = ", ")
      ),
      call. = FALSE
    )
  }
  t0 <- epoch_seconds(start, start_name)
  t1 <- epoch_seconds(end, end_name)
  if (!length(t0) %in% c(1L, length(t1)) && !length(t1) %in% c(1L, length(t0))) {
    stop(
      sprintf(
        "`%s` and `%s` must have the same length, or one of them length one.", start_name,
        end_name
      ),
      call. = FALSE
    )
  }
  elapsed <- (t1 - t0) / UNIT_SECONDS[[unit]]
  negative <- !is.na(elapsed) & elapsed < 0
  if (any(negative)) {
    stop(
      sprintf(
        "%d %s `%s` earlier than `%s`, which gives a negative duration. The first is row %d.",
        sum(negative), if (sum(negative) == 1L) "row has" else "rows have", end_name, start_name,
        which(negative)[1L]
      ),
      call. = FALSE
    )
  }
  elapsed
}

# Seconds since 1970-01-01 UTC.
epoch_seconds <- function(x, name) {
  if (inherits(x, "Date")) {
    return(as.numeric(unclass(x)) * 86400)
  }
  if (inherits(x, "POSIXt")) {
    return(as.numeric(as.POSIXct(x)))
  }
  if (is.character(x) || is.factor(x)) {
    x <- as.character(x)
    out <- rep(NA_real_, length(x))
    given <- !is.na(x) & nzchar(x)
    formats <- c(
      "%Y-%m-%dT%H:%M:%OS", "%Y-%m-%d %H:%M:%OS", "%Y-%m-%dT%H:%M", "%Y-%m-%d %H:%M",
      "%Y-%m-%d", "%Y/%m/%d %H:%M:%OS", "%Y/%m/%d"
    )
    parsed <- rep(NA_real_, sum(given))
    for (format in formats) {
      todo <- is.na(parsed)
      if (!any(todo)) {
        break
      }
      # The longer formats come first, since strptime() reads a matching prefix.
      parsed[todo] <- as.numeric(as.POSIXct(strptime(x[given][todo], format, tz = "UTC")))
    }
    if (anyNA(parsed)) {
      bad <- x[given][is.na(parsed)][1L]
      stop(sprintf("`%s` has a value that isn't a date: \"%s\".", name, bad), call. = FALSE)
    }
    out[given] <- parsed
    return(out)
  }
  if (is.numeric(x) || is.logical(x) && !all(is.na(x))) {
    stop(
      sprintf(
        paste(
          "`%s` must hold dates or date-times for `duration()`, not numbers. If it already holds",
          "elapsed times, use it directly as the time."
        ),
        name
      ),
      call. = FALSE
    )
  }
  if (is.logical(x)) {
    return(rep(NA_real_, length(x)))
  }
  stop(sprintf("`%s` must hold dates or date-times, not %s.", name, obj_type_friendly(x)),
    call. = FALSE
  )
}
