# The event-time response: event_time() and the vector it builds. Its contract is
# spec/event_time.md, and the cases in spec/conformance/event_time/ are replayed by
# tests/testthat/test-event-time-conformance.R.

EVENT_STATUS_VALUES <- c("e", "r", "l", "i")
MAX_LOCATIONS <- 20L

#' Create an event-time response
#'
#' `event_time()` records one observed time per subject together with how it was observed: an
#' exact event, right-censored, left-censored, or interval-censored. The result is a vector, so it
#' can be subset, combined, and stored as a data frame column, and [as_surv()] turns it into the
#' [Surv()] response that the models take.
#'
#' The status codes are:
#'
#' * `"e"`: an exact event at `time`.
#' * `"r"`: right-censored at `time`. The event happened after it.
#' * `"l"`: left-censored at `time`. The event happened before it.
#' * `"i"`: interval-censored. The event happened after `time` and at or before `time_max`.
#'
#' `time_max` is given only for the interval-censored rows and is missing everywhere else. A row
#' with a missing time is missing. Its status can be missing too.
#'
#' `event_time()` can be used on the left-hand side of a model formula, where it's converted with
#' [as_surv()], such as `event_time(time, status, time_max) ~ 1`.
#'
#' @param time The event or censoring time, or the lower bound of an interval. Non-negative
#'   numbers (or logical `NA`).
#' @param status The status codes, a character vector of `"e"`, `"r"`, `"l"`, and `"i"` the same
#'   length as `time`.
#' @param time_max The upper bound for the interval-censored rows, missing for the others. The
#'   default is missing for every row.
#'
#' @return An `event_time` vector the length of `time`. It prints with the codes as suffixes: `7`
#'   for an event, `5+` for right-censoring, `3-` for left-censoring, and `[2, 4]` for an interval.
#'
#' @section Errors:
#' The checks run in a fixed order, and the first that fails is reported, naming every row where
#' it fails. The condition has the class `greenwood_event_time_error` and fields `check` (an id
#' such as `"status_invalid_value"`) and `locations` (the rows).
#'
#' @seealso [new_event_time()] to build one from its parts without checks, [extract_time()] and
#'   [extract_status()] to take it apart, and [as_surv()] to convert it.
#'
#' @examples
#' x <- event_time(
#'   time = c(7, 5, 3, 2, NA),
#'   status = c("e", "r", "l", "i", NA),
#'   time_max = c(NA, NA, NA, 4, NA)
#' )
#' x
#' x[2:4]
#' as_surv(x)
#'
#' # On the left-hand side of a formula
#' visits <- data.frame(
#'   lower = c(2, 4, 1, 6, 3, 8),
#'   upper = c(4, NA, 3, 9, NA, 11),
#'   status = c("i", "r", "i", "i", "r", "i")
#' )
#' turnbull(event_time(lower, status, time_max = upper) ~ 1, data = visits)
#' @export
event_time <- function(time, status, time_max = NULL) {
  n <- length(time)
  if (is.null(time_max)) {
    time_max <- rep(NA_real_, n)
  }
  time <- cast_double(time, "time")
  time_max <- cast_double(time_max, "time_max")
  check_status_type(status)
  if (length(status) != n) {
    event_time_abort(
      sprintf("`status` must have length %d (the length of `time`), not %d.", n, length(status)),
      "status_length"
    )
  }
  if (length(time_max) != n) {
    event_time_abort(
      sprintf(
        "`time_max` must have length %d (the length of `time`), not %d.", n, length(time_max)
      ),
      "time_max_length"
    )
  }
  check_locations(which(time < 0), "`time` must be non-negative.", "time_negative")
  check_locations(
    which(is.na(status) & !is.na(time)),
    "`status` must not be missing when `time` is not missing.",
    "status_missing"
  )
  check_locations(
    which(!is.na(status) & !status %in% EVENT_STATUS_VALUES),
    "`status` must be one of \"e\", \"r\", \"l\", and \"i\".",
    "status_invalid_value"
  )
  is_interval <- !is.na(status) & status == "i"
  check_locations(
    which(!is_interval & !is.na(time_max)),
    "`time_max` must be missing for values that are not interval censored.",
    "time_max_not_allowed"
  )
  check_locations(
    which(is_interval & is.na(time_max)),
    "`time_max` must not be missing for interval-censored values.",
    "time_max_missing"
  )
  check_locations(
    which(is_interval & time >= time_max),
    "`time` must be smaller than `time_max` for interval-censored values.",
    "interval_bounds_order"
  )
  pooled <- vector("list", n)
  for (i in seq_len(n)) {
    pooled[[i]] <- if (is_interval[i]) c(time[i], time_max[i]) else time[i]
  }
  new_event_time(pooled, status)
}

#' Build an event-time vector from its parts
#'
#' `new_event_time()` is the low-level constructor behind [event_time()]. It only checks the types
#' of its arguments, not their values, so it's for code that has already validated them.
#'
#' @param time A list with one numeric vector per element: the time (length one), or the lower
#'   and upper bounds of an interval (length two).
#' @param status A character vector of status codes, the same length as `time`.
#'
#' @return An `event_time` vector.
#'
#' @seealso [event_time()], which checks its input.
#'
#' @examples
#' new_event_time(time = list(7, c(2, 4)), status = c("e", "i"))
#' @export
new_event_time <- function(time = list(), status = character()) {
  if (!is.list(time) || is.object(time)) {
    event_time_abort(
      sprintf("`time` must be a list, not %s.", obj_type_friendly(time)),
      "time_not_list"
    )
  }
  check_status_type(status)
  if (length(time) != length(status)) {
    event_time_abort("All fields must be the same size.", "fields_size")
  }
  event_time_vector(unname(time), unname(status))
}

#' Take an event-time vector apart
#'
#' `extract_time()` gives the times of an [event_time()] vector, and `extract_status()` its status
#' codes.
#'
#' @param x An [event_time()] vector.
#' @param ... Unused.
#'
#' @return For `extract_time()`, a numeric vector of the times (the lower bounds for intervals). If
#'   any element is interval-censored, it's instead a matrix with columns `time` and `time_max`,
#'   where `time_max` is the upper bound of the intervals and missing elsewhere. For
#'   `extract_status()`, a character vector of the status codes.
#'
#' @examples
#' x <- event_time(time = c(7, 5, 2), status = c("e", "r", "i"), time_max = c(NA, NA, 4))
#' extract_time(x)
#' extract_status(x)
#' extract_time(x[1:2])
#' @export
extract_time <- function(x, ...) {
  UseMethod("extract_time")
}

#' @export
extract_time.greenwood_event_time <- function(x, ...) {
  check_no_dots(...)
  time <- event_time_lower(x)
  if (!any(event_time_status(x) == "i", na.rm = TRUE)) {
    return(time)
  }
  cbind(time = time, time_max = event_time_upper(x))
}

#' @rdname extract_time
#' @export
extract_status <- function(x, ...) {
  UseMethod("extract_status")
}

#' @export
extract_status.greenwood_event_time <- function(x, ...) {
  check_no_dots(...)
  event_time_status(x)
}

#' Convert to a survival response
#'
#' `as_surv()` turns an [event_time()] vector into the [Surv()] response the models take. The type
#' of the response depends on the status codes present, with missing elements ignored:
#'
#' * Only events and right-censoring: a right-censored response, with status `1` for an event.
#' * Only events and left-censoring: a left-censored response.
#' * Anything else: an interval-censored response, with status `0` for right-censored, `1` for an
#'   event, `2` for left-censored, and `3` for interval-censored.
#'
#' Missing elements stay missing, and a vector of length zero gives an empty right-censored
#' response. A [Surv()] response is returned as it is.
#'
#' @param x An [event_time()] vector.
#' @param ... Unused.
#'
#' @return A [Surv()] response the length of `x`.
#'
#' @examples
#' as_surv(event_time(time = c(7, 5, 3), status = c("e", "r", "r")))
#' as_surv(event_time(time = c(7, 2), status = c("e", "i"), time_max = c(NA, 4)))
#' @export
as_surv <- function(x, ...) {
  UseMethod("as_surv")
}

#' @export
as_surv.default <- function(x, ...) {
  event_time_abort(
    sprintf("Can't convert %s to a <Surv> object.", obj_type_friendly(x)),
    "as_surv_unsupported"
  )
}

#' @export
as_surv.Surv <- function(x, ...) {
  x
}

#' @export
as_surv.greenwood_event_time <- function(x, ...) {
  check_no_dots(...)
  lower <- event_time_lower(x)
  upper <- vapply(event_time_times(x), function(t) t[length(t)], double(1))
  status <- event_time_status(x)
  status[is.na(x)] <- NA_character_
  observed <- unique(status[!is.na(status)])
  if (length(x) == 0L) {
    Surv(double(), integer())
  } else if (all(observed %in% c("e", "r"))) {
    Surv(lower, as.integer(status == "e"), type = "right")
  } else if (all(observed %in% c("e", "l"))) {
    Surv(lower, as.integer(status == "e"), type = "left")
  } else {
    code <- unname(c(r = 0L, e = 1L, l = 2L, i = 3L)[status])
    Surv(lower, upper, code, type = "interval")
  }
}

# -- the vector ---------------------------------------------------------------------------------
#
# The vector is a list of two fields, `time` (a list with one numeric vector per element) and
# `status`, with a length() method. Methods dispatch on the "greenwood_event_time" subclass, so
# they don't replace other packages' methods for the "event_time" class. The fields are read with
# .subset2(), since `[[` is the element accessor.

event_time_vector <- function(time, status) {
  structure(list(time = time, status = status), class = c("greenwood_event_time", "event_time"))
}

event_time_times <- function(x) {
  .subset2(x, "time")
}

event_time_status <- function(x) {
  .subset2(x, "status")
}

event_time_lower <- function(x) {
  vapply(event_time_times(x), `[`, double(1), 1L)
}

event_time_upper <- function(x) {
  vapply(event_time_times(x), function(t) if (length(t) == 2L) t[2L] else NA_real_, double(1))
}

#' @export
length.greenwood_event_time <- function(x) {
  length(event_time_status(x))
}

#' @export
is.na.greenwood_event_time <- function(x) {
  vapply(event_time_times(x), anyNA, logical(1))
}

#' @export
anyNA.greenwood_event_time <- function(x, recursive = FALSE) {
  any(is.na(x))
}

#' @export
format.greenwood_event_time <- function(x, ...) {
  time <- event_time_times(x)
  status <- event_time_status(x)
  if (length(time) == 0L) {
    return(character())
  }
  # The numbers are formatted together, so they share digits and notation.
  flat <- format(unlist(time), trim = TRUE)
  group <- factor(rep(seq_along(time), lengths(time)), levels = seq_along(time))
  out <- unname(vapply(split(flat, group), paste, character(1), collapse = ", "))
  suffix <- c(e = " ", r = "+", l = "-", i = "")
  out <- paste0(out, suffix[status])
  is_interval <- !is.na(status) & status == "i"
  out[is_interval] <- paste0("[", out[is_interval], "]")
  out[is.na(x)] <- NA_character_
  out
}

#' @export
print.greenwood_event_time <- function(x, ...) {
  cat(sprintf("<event_time[%d]>\n", length(x)))
  if (length(x) > 0L) {
    print(format(x), quote = FALSE)
  }
  invisible(x)
}

#' @export
`[.greenwood_event_time` <- function(x, i) {
  if (missing(i)) {
    return(x)
  }
  index <- event_time_index(i, length(x))
  time <- event_time_times(x)[index]
  # A missing index gives a missing element.
  time[is.na(index)] <- list(NA_real_)
  event_time_vector(time, event_time_status(x)[index])
}

#' @export
`[[.greenwood_event_time` <- function(x, i) {
  if (length(i) != 1L) {
    stop("`[[` takes one location.", call. = FALSE)
  }
  x[i]
}

#' @export
`[<-.greenwood_event_time` <- function(x, i, value) {
  if (!inherits(value, "greenwood_event_time")) {
    stop(
      sprintf("Can't convert `value` <%s> to <event_time>.", vector_type(value)),
      call. = FALSE
    )
  }
  index <- if (missing(i)) seq_along(x) else event_time_index(i, length(x), assign = TRUE)
  if (anyNA(index)) {
    stop("Can't assign to a missing location.", call. = FALSE)
  }
  if (!length(value) %in% c(1L, length(index))) {
    stop(
      sprintf("Can't recycle input of size %d to size %d.", length(value), length(index)),
      call. = FALSE
    )
  }
  value <- value[rep_len(seq_along(value), length(index))]
  time <- event_time_times(x)
  status <- event_time_status(x)
  # Assigning past the end grows the vector with missing elements.
  grow <- setdiff(seq_len(max(c(index, 0L))), seq_along(time))
  time[grow] <- list(NA_real_)
  status[grow] <- NA_character_
  time[index] <- event_time_times(value)
  status[index] <- event_time_status(value)
  event_time_vector(time, status)
}

#' @export
`[[<-.greenwood_event_time` <- function(x, i, value) {
  if (length(i) != 1L) {
    stop("`[[<-` takes one location.", call. = FALSE)
  }
  x[i] <- value
  x
}

# The positions that index `i` selects in a vector of length `n`, following vctrs: logical indexes
# have length one or `n`, and positive locations past the end are an error (unless assigning).
event_time_index <- function(i, n, assign = FALSE) {
  if (is.character(i)) {
    stop("Can't use character names to index an `event_time` vector.", call. = FALSE)
  }
  if (is.logical(i)) {
    if (!length(i) %in% c(1L, n)) {
      stop(sprintf("Logical subscript must be size 1 or %d, not %d.", n, length(i)), call. = FALSE)
    }
    return(seq_len(n)[rep_len(i, n)])
  }
  if (!is.numeric(i)) {
    stop(sprintf("Can't index an `event_time` vector with <%s>.", vector_type(i)), call. = FALSE)
  }
  i <- trunc(i)
  past <- unique(i[!is.na(i) & i > n])
  if (length(past) > 0L && !assign) {
    stop(
      sprintf(
        "Can't subset elements past the end. %s %s %s exist. There %s only %d element%s.",
        if (length(past) == 1L) "Location" else "Locations",
        describe_values(past),
        if (length(past) == 1L) "doesn't" else "don't",
        if (n == 1L) "is" else "are",
        n,
        if (n == 1L) "" else "s"
      ),
      call. = FALSE
    )
  }
  if (any(i < 0, na.rm = TRUE)) {
    if (any(i > 0, na.rm = TRUE) || anyNA(i)) {
      stop("Can't mix negative and positive locations.", call. = FALSE)
    }
    return(seq_len(n)[i])
  }
  as.integer(i[is.na(i) | i > 0])
}

#' @export
c.greenwood_event_time <- function(...) {
  parts <- list(...)
  for (k in seq_along(parts)) {
    if (!inherits(parts[[k]], "greenwood_event_time")) {
      stop(
        sprintf(
          "Can't combine `..1` <event_time> and `..%d` <%s>.", k, vector_type(parts[[k]])
        ),
        call. = FALSE
      )
    }
  }
  event_time_vector(
    unlist(lapply(parts, event_time_times), recursive = FALSE),
    unlist(lapply(parts, event_time_status))
  )
}

#' @export
rep.greenwood_event_time <- function(x, ...) {
  x[rep(seq_along(x), ...)]
}

# One string per element, for finding duplicates. Missing times and statuses are keys too, so two
# missing elements are duplicates.
event_time_keys <- function(x) {
  times <- event_time_times(x)
  status <- event_time_status(x)
  vapply(seq_along(times), function(i) {
    paste(c(sprintf("%a", times[[i]]), status[i]), collapse = " ")
  }, character(1))
}

#' @export
duplicated.greenwood_event_time <- function(x, incomparables = FALSE, ...) {
  duplicated(event_time_keys(x))
}

#' @export
unique.greenwood_event_time <- function(x, incomparables = FALSE, ...) {
  x[!duplicated(x)]
}

#' @export
Ops.greenwood_event_time <- function(e1, e2) {
  if (!.Generic %in% c("==", "!=")) {
    stop(sprintf("`%s` isn't defined for `event_time` vectors.", .Generic), call. = FALSE)
  }
  if (!inherits(e1, "greenwood_event_time") || !inherits(e2, "greenwood_event_time")) {
    stop("An `event_time` vector can only be compared with another.", call. = FALSE)
  }
  n <- max(length(e1), length(e2))
  if (min(length(e1), length(e2)) == 0L) {
    return(logical())
  }
  if (!all(c(length(e1), length(e2)) %in% c(1L, n))) {
    stop(
      sprintf("Can't recycle vectors of sizes %d and %d.", length(e1), length(e2)),
      call. = FALSE
    )
  }
  a <- e1[rep_len(seq_along(e1), n)]
  b <- e2[rep_len(seq_along(e2), n)]
  # Field by field, as for records: elements whose times differ are unequal, and otherwise a
  # missing status makes the comparison missing.
  same_time <- mapply(identical, event_time_times(a), event_time_times(b), USE.NAMES = FALSE)
  equal <- ifelse(same_time, event_time_status(a) == event_time_status(b), FALSE)
  if (.Generic == "==") equal else !equal
}

#' @export
xtfrm.greenwood_event_time <- function(x) {
  stop(
    paste(
      "`event_time` vectors have no order, since an interval has no single time. Sort by",
      "`extract_time()` instead."
    ),
    call. = FALSE
  )
}

#' @export
as.character.greenwood_event_time <- function(x, ...) {
  stop("Can't convert `x` <event_time> to <character>.", call. = FALSE)
}

#' @export
as.double.greenwood_event_time <- function(x, ...) {
  stop("Can't convert `x` <event_time> to <double>.", call. = FALSE)
}

#' @export
as.list.greenwood_event_time <- function(x, ...) {
  lapply(seq_along(x), function(i) x[i])
}

#' @export
as.data.frame.greenwood_event_time <- function(x,
                                               row.names = NULL,
                                               optional = FALSE,
                                               ...,
                                               nm = deparse1(substitute(x))) {
  force(nm)
  value <- list(x)
  if (!optional) {
    names(value) <- nm
  }
  if (is.null(row.names)) {
    row.names <- .set_row_names(length(x))
  }
  structure(value, row.names = row.names, class = "data.frame")
}

#' @export
summary.greenwood_event_time <- function(object, ...) {
  stop("`summary.event_time()` not implemented.", call. = FALSE)
}

#' @exportS3Method tibble::as_tibble
as_tibble.greenwood_event_time <- function(x, ...) {
  check_no_dots(...)
  tibble::tibble(
    time = event_time_lower(x),
    status = event_time_status(x),
    time_max = event_time_upper(x)
  )
}

# -- checks -------------------------------------------------------------------------------------

# Every event-time error has the class "greenwood_event_time_error", with the id of the check that
# failed and the 1-based rows it failed at.
event_time_abort <- function(message, check, locations = integer()) {
  stop(structure(
    list(message = message, call = NULL, check = check, locations = locations),
    class = c("greenwood_event_time_error", "error", "condition")
  ))
}

check_locations <- function(idx, message, check) {
  if (length(idx) == 0L) {
    return(invisible())
  }
  event_time_abort(
    sprintf(
      "%s Problem at %s %s.", message, if (length(idx) == 1L) "location" else "locations",
      describe_values(idx)
    ),
    check,
    locations = idx
  )
}

# Values listed as "1", "1 and 2", or "1, 2, and 3". A long list shows its first and last values.
describe_values <- function(values) {
  values <- format(values, trim = TRUE, scientific = FALSE)
  n <- length(values)
  if (n > MAX_LOCATIONS) {
    values <- c(values[seq_len(MAX_LOCATIONS - 2L)], "...", values[c(n - 1L, n)])
    n <- length(values)
  }
  if (n == 1L) {
    return(values)
  }
  if (n == 2L) {
    return(paste(values, collapse = " and "))
  }
  paste0(paste(values[-n], collapse = ", "), ", and ", values[n])
}

# Double, integer, and logical vectors become doubles. Anything else is an error.
cast_double <- function(x, arg) {
  if (is.null(x)) {
    return(double())
  }
  if (is.object(x) || !typeof(x) %in% c("double", "integer", "logical")) {
    event_time_abort(
      sprintf("Can't convert `%s` <%s> to <double>.", arg, vector_type(x)),
      paste0(arg, "_not_numeric")
    )
  }
  as.double(x)
}

check_status_type <- function(status) {
  if (!is.character(status) || is.object(status)) {
    event_time_abort(
      sprintf("`status` must be a character vector, not %s.", obj_type_friendly(status)),
      "status_not_character"
    )
  }
}

check_no_dots <- function(...) {
  if (...length() > 0L) {
    stop("`...` must be empty.", call. = FALSE)
  }
}

# The type of a vector the way it's named in messages, such as <double> or <factor>.
vector_type <- function(x) {
  if (inherits(x, "event_time")) {
    return("event_time")
  }
  if (is.data.frame(x)) {
    return("data.frame")
  }
  if (is.object(x)) {
    return(class(x)[1L])
  }
  typeof(x)
}

# A description of an object for messages, such as "a number" or "a <factor> object".
obj_type_friendly <- function(x) {
  if (is.object(x)) {
    return(sprintf("a <%s> object", class(x)[1L]))
  }
  if (is.null(x)) {
    return("`NULL`")
  }
  if (is.function(x)) {
    return("a function")
  }
  if (is.list(x)) {
    return("a list")
  }
  if (!is.atomic(x)) {
    return(sprintf("an object of type <%s>", typeof(x)))
  }
  type <- typeof(x)
  if (!is.null(dim(x))) {
    kind <- if (length(dim(x)) == 2L) "matrix" else "array"
    return(sprintf("%s %s %s", article(type), type, kind))
  }
  if (length(x) == 1L) {
    if (is.na(x)) {
      return(switch(type,
        logical = "`NA`",
        integer = "an integer `NA`",
        double = if (is.nan(x)) "`NaN`" else "a numeric `NA`",
        character = "a character `NA`",
        sprintf("a %s `NA`", type)
      ))
    }
    return(switch(type,
      logical = if (x) "`TRUE`" else "`FALSE`",
      integer = "an integer",
      double = if (is.infinite(x)) if (x > 0) "`Inf`" else "`-Inf`" else "a number",
      character = if (nzchar(x)) "a string" else "the empty string \"\"",
      sprintf("a %s value", type)
    ))
  }
  if (length(x) == 0L) {
    return(sprintf("an empty %s vector", type))
  }
  sprintf("%s %s vector", article(type), type)
}

article <- function(word) {
  if (grepl("^[aeiou]", word)) "an" else "a"
}
