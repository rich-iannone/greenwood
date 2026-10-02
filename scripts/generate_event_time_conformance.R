#!/usr/bin/env Rscript
# Generate the event_time conformance fixtures from the pinned etd package.
#
# Greenwood ports etd (https://github.com/topepo/etd) to Python and follows it exactly. These
# fixtures are the executable form of that contract: each case records an input and what etd
# does with it (the result, or which validation check fails and where). The Python suite
# (tests/test_event_time_conformance.py) replays every case. A future R Greenwood can replay the
# same JSON. See spec/event_time.md and EVENT_TIME_PLAN.md.
#
# Install the pinned etd first, then run from the repo root:
#
#   Rscript -e 'pak::pak("topepo/etd@3d5caff4f03d771a6b65b9605c9b0e48a912482f")'
#   Rscript scripts/generate_event_time_conformance.R
#
# Set ETD_LIB to load etd from a specific library instead of the default library path.
#
# Writes JSON into spec/conformance/event_time/. Encoding conventions:
#
# - Numeric vectors are arrays. Missing values are the strings "NA" and "NaN", infinities are
#   "Inf" and "-Inf" (jsonlite's encoding, which tests/_r_parity.py already understands).
# - Character and logical missing values are JSON null.
# - R inputs are wrapped as {"r_type": ..., "values": ...} so the replaying side knows whether
#   R saw a double, integer, logical, character, factor, or list.
# - Locations are 1-based, exactly as etd reports them. Replaying code converts to its own
#   indexing convention.

ETD_SHA <- "3d5caff4f03d771a6b65b9605c9b0e48a912482f"

suppressPackageStartupMessages({
  for (pkg in c("jsonlite", "survival", "vctrs", "tibble")) {
    if (!requireNamespace(pkg, quietly = TRUE)) {
      stop(sprintf("%s is required: install.packages('%s')", pkg, pkg))
    }
  }
  etd_lib <- Sys.getenv("ETD_LIB", unset = "")
  if (nzchar(etd_lib)) {
    library(etd, lib.loc = etd_lib)
  } else {
    library(etd)
  }
})

installed_sha <- utils::packageDescription("etd")$RemoteSha
if (is.null(installed_sha) || !identical(installed_sha, ETD_SHA)) {
  stop(sprintf(
    "etd must be installed at the pinned commit %s (found %s). Run:\n  pak::pak(\"topepo/etd@%s\")",
    ETD_SHA,
    if (is.null(installed_sha)) "a non-GitHub install" else installed_sha,
    ETD_SHA
  ))
}

# Plain-text condition messages, so check-id matching never sees ANSI codes or fancy quotes.
options(cli.num_colors = 1, cli.unicode = FALSE, useFancyQuotes = FALSE, width = 80)

out_dir <- file.path("spec", "conformance", "event_time")
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

write_fixture <- function(cases, name) {
  path <- file.path(out_dir, paste0(name, ".json"))
  jsonlite::write_json(
    cases,
    path,
    auto_unbox = FALSE,
    digits = NA,
    pretty = TRUE,
    null = "null",
    na = "string"
  )
  cat(sprintf("wrote %s (%d cases)\n", path, length(cases)))
}

u <- jsonlite::unbox

# -- encoding -------------------------------------------------------------------------------

# Character vectors keep NA as JSON null (jsonlite's `na = "string"` would write "NA", which
# could be mistaken for a status value).
enc_chr <- function(x) {
  lapply(as.character(x), function(v) if (is.na(v)) NULL else u(v))
}

enc_lgl <- function(x) {
  lapply(x, function(v) if (is.na(v)) NULL else u(v))
}

# Wrap an R input with its type so the replaying side can build the equivalent value.
enc_input <- function(x) {
  if (is.null(x)) {
    return(NULL)
  }
  if (is.factor(x)) {
    return(list(r_type = u("factor"), values = enc_chr(x), levels = enc_chr(levels(x))))
  }
  if (is.list(x)) {
    return(list(r_type = u("list"), values = lapply(x, enc_input)))
  }
  values <- switch(
    typeof(x),
    double = x,
    integer = x,
    logical = enc_lgl(x),
    character = enc_chr(x),
    stop(sprintf("Unsupported input type: %s", typeof(x)))
  )
  list(r_type = u(typeof(x)), values = values)
}

# -- check ids ----------------------------------------------------------------------------

# Every etd validation failure maps to a stable id, so fixtures never depend on message text.
# An unmapped message stops the generator: map it consciously rather than recording noise.
check_patterns <- c(
  time_not_numeric = "Can't convert `time`",
  time_max_not_numeric = "Can't convert `time_max`",
  status_not_character = "`status` must be a character vector",
  status_length = "`status` must have length",
  time_max_length = "`time_max` must have length",
  time_negative = "`time` must be non-negative",
  status_missing = "`status` must not be missing",
  status_invalid_value = "`status` must be one of",
  time_max_not_allowed = "`time_max` must be missing for values that are not interval censored",
  time_max_missing = "`time_max` must not be missing for interval-censored values",
  interval_bounds_order = "`time` must be smaller than `time_max`",
  time_not_list = "`time` must be a list",
  fields_size = "same size|Can't recycle",
  as_surv_unsupported = "to a <Surv> object"
)

check_id <- function(message) {
  hits <- names(check_patterns)[vapply(
    check_patterns,
    function(p) grepl(p, message),
    logical(1)
  )]
  if (length(hits) != 1) {
    stop(sprintf("No unique check id for message:\n%s", message))
  }
  hits
}

# "Problem at locations 1, 2, and 3." -> c(1, 2, 3). NULL when the message names no locations.
check_locations <- function(message) {
  tail_text <- regmatches(message, regexpr("Problem at locations? [^.]*", message))
  if (length(tail_text) == 0) {
    return(NULL)
  }
  as.integer(regmatches(tail_text, gregexpr("[0-9]+", tail_text))[[1]])
}

describe_error <- function(e) {
  # cli wraps long messages at the console width. Collapse the wrapping so location parsing and
  # the recorded message don't depend on it.
  message <- gsub("[ \t]*\n[ \t]*", " ", conditionMessage(e))
  locations <- check_locations(message)
  list(
    ok = u(FALSE),
    check = u(check_id(message)),
    locations = if (is.null(locations)) NULL else locations,
    message = u(message)
  )
}

# -- describing results ---------------------------------------------------------------------

capture <- function(expr) {
  warnings <- character()
  value <- withCallingHandlers(
    tryCatch(
      list(ok = TRUE, value = expr),
      error = function(e) list(ok = FALSE, message = conditionMessage(e))
    ),
    warning = function(w) {
      warnings <<- c(warnings, conditionMessage(w))
      invokeRestart("muffleWarning")
    }
  )
  value$warnings <- warnings
  value
}

describe_surv <- function(s) {
  m <- unclass(s)
  columns <- lapply(seq_len(ncol(m)), function(j) unname(m[, j]))
  names(columns) <- colnames(m)
  list(type = u(attr(s, "type")), columns = columns)
}

describe_extract_time <- function(x) {
  et <- extract_time(x)
  if (is.matrix(et)) {
    list(
      shape = u("matrix"),
      columns = lapply(split(unname(et), col(et)), identity) |> stats::setNames(colnames(et))
    )
  } else {
    list(shape = u("vector"), values = et)
  }
}

# Run one observation of a valid vector. A failure is recorded as data rather than stopping the
# generator, so a bug in one etd method (say, format() on an empty vector) doesn't hide the rest.
observe <- function(expr, encode = identity) {
  out <- capture(encode(expr))
  if (out$ok) {
    list(ok = u(TRUE), value = out$value, warnings = out$warnings)
  } else {
    list(ok = u(FALSE), message = u(out$message), warnings = out$warnings)
  }
}

# Everything observable about a valid event_time vector.
describe_event_time <- function(x) {
  list(
    ok = u(TRUE),
    length = u(length(x)),
    fields = list(
      time = lapply(vctrs::field(x, "time"), function(t) t),
      status = enc_chr(vctrs::field(x, "status"))
    ),
    is_na = observe(is.na(x)),
    format = observe(format(x), enc_chr),
    print = observe(capture.output(print(x))),
    extract_time = observe(x, describe_extract_time),
    extract_status = observe(extract_status(x), enc_chr),
    as_surv = observe(as_surv(x), describe_surv),
    as_tibble = observe(tibble::as_tibble(x), function(tbl) {
      list(time = tbl$time, status = enc_chr(tbl$status), time_max = tbl$time_max)
    })
  )
}

# Construct, recording a validation failure as a check id. Only construction errors are mapped
# to check ids. Errors while describing a valid vector are recorded by observe().
construct <- function(expr) {
  value <- tryCatch(expr, error = function(e) e)
  if (inherits(value, "error")) describe_error(value) else describe_event_time(value)
}

run_event_time <- function(id, description, time, status, time_max = NULL) {
  input <- list(time = enc_input(time), status = enc_input(status), time_max = enc_input(time_max))
  result <- construct(event_time(time, status, time_max))
  list(id = u(id), description = u(description), input = input, result = result)
}

run_new_event_time <- function(id, description, time, status) {
  input <- list(time = enc_input(time), status = enc_input(status))
  result <- construct(new_event_time(time, status))
  list(id = u(id), description = u(description), input = input, result = result)
}

# -- event_time(): construction and validation --------------------------------------------

constructor_cases <- list(
  # Valid input.
  run_event_time(
    "doc_example",
    "The event_time() documentation example: one of each code plus a missing row.",
    c(7, 5, 3, 2, NA),
    c("e", "r", "l", "i", NA),
    c(NA, NA, NA, 4, NA)
  ),
  run_event_time("empty", "Zero-length input.", double(), character()),
  run_event_time("single_event", "A single exact event.", 3, "e"),
  run_event_time("events_only", "Only exact events.", c(1, 2, 3), c("e", "e", "e")),
  run_event_time("right_only", "Only right-censored values.", c(1, 2), c("r", "r")),
  run_event_time("left_only", "Only left-censored values.", c(1, 2), c("l", "l")),
  run_event_time(
    "interval_only",
    "Only interval-censored values.",
    c(1, 2),
    c("i", "i"),
    c(3, 4)
  ),
  run_event_time(
    "events_and_right",
    "Exact and right-censored values, which convert to a right-censored Surv.",
    c(5, 6, 4, 9),
    c("e", "r", "e", "r")
  ),
  run_event_time(
    "events_and_left",
    "Exact and left-censored values, which convert to a left-censored Surv.",
    c(5, 6, 4),
    c("e", "l", "e")
  ),
  run_event_time(
    "right_and_left",
    "Right- and left-censored values without intervals, which convert to an interval Surv.",
    c(5, 6),
    c("r", "l")
  ),
  run_event_time(
    "events_and_interval",
    "Exact and interval-censored values.",
    c(1, 2),
    c("e", "i"),
    c(NA, 5)
  ),
  run_event_time(
    "readme_left_right",
    "The README example: sorted exponential draws with left, exact, and right codes.",
    {
      set.seed(1)
      signif(sort(rexp(5)), 12)
    },
    c("l", "e", "e", "e", "r")
  ),
  run_event_time(
    "readme_interval",
    "The README interval example.",
    {
      set.seed(1)
      signif(sort(rexp(5)), 12)
    },
    c("i", "e", "e", "e", "r"),
    c(1.5, NA, NA, NA, NA)
  ),
  run_event_time("integer_time", "Integer time is cast to double.", 1:3, c("e", "r", "e")),
  run_event_time(
    "integer_time_max",
    "Integer time_max is cast to double.",
    c(1, 2),
    c("i", "e"),
    c(3L, NA_integer_)
  ),
  run_event_time(
    "logical_time",
    "Logical time is cast to double, as vctrs allows.",
    c(TRUE, FALSE),
    c("e", "r")
  ),
  run_event_time("all_na_logical_time", "A logical NA time.", NA, NA_character_),
  run_event_time("zero_time", "A time of exactly zero.", c(0, 1), c("e", "r")),
  run_event_time("zero_interval_lower", "An interval starting at zero.", 0, "i", 2),
  run_event_time(
    "missing_time_with_status",
    "A missing time may still carry a status.",
    c(NA, 2),
    c("e", "r")
  ),
  run_event_time(
    "missing_time_and_status",
    "A missing time with a missing status.",
    c(1, NA),
    c("e", NA)
  ),
  run_event_time(
    "missing_lower_interval",
    "An interval row with a missing lower bound.",
    c(NA, 1),
    c("i", "e"),
    c(3, NA)
  ),
  run_event_time("nan_time", "NaN time counts as missing.", c(NaN, 1), c("e", "e")),
  run_event_time("infinite_time", "An infinite right-censoring time.", c(Inf, 1), c("r", "e")),
  run_event_time(
    "infinite_interval_upper",
    "An interval with an infinite upper bound.",
    1,
    "i",
    Inf
  ),
  run_event_time(
    "explicit_all_na_time_max",
    "time_max given as all missing when no row is an interval.",
    c(1, 2),
    c("e", "r"),
    c(NA_real_, NA_real_)
  ),
  run_event_time(
    "logical_na_time_max",
    "time_max given as a logical NA vector.",
    c(1, 2),
    c("e", "r"),
    c(NA, NA)
  ),
  run_event_time(
    "mixed_magnitudes",
    "Values of very different magnitudes, formatted together.",
    c(0.001234, 12345.6, 7),
    c("e", "r", "l")
  ),

  # Validation failures, in etd's check order.
  run_event_time("time_character", "Character time cannot be cast to double.", "a", "e"),
  run_event_time("time_list", "A list time cannot be cast to double.", list(1, 2), c("e", "e")),
  run_event_time(
    "time_max_character",
    "Character time_max cannot be cast to double.",
    1,
    "i",
    "a"
  ),
  run_event_time("status_numeric", "Numeric status is rejected.", 1, 1),
  run_event_time("status_logical", "Logical status is rejected.", 1, NA),
  run_event_time("status_factor", "Factor status is rejected.", 1, factor("e")),
  run_event_time("status_short", "status shorter than time.", 1:2, "e"),
  run_event_time("status_long", "status longer than time.", 1, c("e", "e")),
  run_event_time("time_max_long", "time_max longer than time.", 1, "e", c(NA, NA)),
  run_event_time("time_max_short", "time_max shorter than time.", c(1, 2), c("e", "e"), NA),
  run_event_time(
    "time_negative",
    "Negative times, with every location reported.",
    c(-1, -2),
    c("e", "e")
  ),
  run_event_time(
    "time_negative_infinite",
    "Negative infinity is negative.",
    -Inf,
    "r"
  ),
  run_event_time(
    "status_missing",
    "A missing status where time is present.",
    c(1, 2, 3),
    c("e", NA, NA)
  ),
  run_event_time("status_unknown", "An unknown status code.", 1, "x"),
  run_event_time("status_uppercase", "Status codes are case-sensitive.", c(1, 2), c("E", "r")),
  run_event_time("status_long_form", "Long-form status words are rejected.", 1, "event"),
  run_event_time(
    "status_unknown_many",
    "Several unknown codes, with every location reported.",
    c(1, 2, 3, 4),
    c("e", "x", "y", "r")
  ),
  run_event_time(
    "time_max_on_event",
    "time_max given for a non-interval row.",
    1,
    "e",
    3
  ),
  run_event_time(
    "time_max_on_missing_status",
    "time_max given for a row with missing time and status.",
    NA,
    NA_character_,
    3
  ),
  run_event_time("time_max_missing", "An interval row without time_max.", 1, "i"),
  run_event_time(
    "time_max_missing_explicit",
    "An interval row with an explicit missing time_max.",
    c(1, 2),
    c("i", "i"),
    c(NA, 3)
  ),
  run_event_time(
    "interval_bounds_reversed",
    "An interval whose lower bound exceeds its upper bound.",
    c(5, 1),
    c("i", "i"),
    c(4, 3)
  ),
  run_event_time(
    "interval_bounds_equal",
    "An interval whose bounds are equal (must be strictly less).",
    2,
    "i",
    2
  ),

  # Several problems at once: the first check in etd's order wins.
  run_event_time(
    "order_negative_before_status",
    "A negative time and an unknown status: the negative-time check runs first.",
    c(-1, 2),
    c("e", "x")
  ),
  run_event_time(
    "order_status_missing_before_invalid",
    "A missing status and an unknown status: the missing-status check runs first.",
    c(1, 2),
    c(NA, "x")
  ),
  run_event_time(
    "order_invalid_before_time_max",
    "An unknown status and a stray time_max: the status-value check runs first.",
    c(1, 2),
    c("x", "e"),
    c(NA, 3)
  ),
  run_event_time(
    "order_length_before_negative",
    "A length mismatch and a negative time: the length check runs first.",
    c(-1, 2),
    "e"
  )
)

# -- new_event_time(): the low-level constructor ----------------------------------------------

new_event_time_cases <- list(
  run_new_event_time(
    "doc_example",
    "The new_event_time() documentation example.",
    list(7, c(2, 4)),
    c("e", "i")
  ),
  run_new_event_time("empty", "The zero-length default.", list(), character()),
  run_new_event_time(
    "unsorted_interval",
    "No validation: an interval stored in reverse order is accepted as given.",
    list(c(4, 2)),
    "i"
  ),
  run_new_event_time(
    "unknown_status",
    "No validation: an unknown status code is accepted as given.",
    list(1),
    "x"
  ),
  run_new_event_time("time_not_list", "time must be a list.", 1, "e"),
  run_new_event_time("status_not_character", "status must be character.", list(1), 1),
  run_new_event_time(
    "fields_size",
    "time and status of different lengths.",
    list(1, 2),
    "e"
  )
)

# -- as_surv() on unsupported input ---------------------------------------------------------

conversion_cases <- list(
  list(
    id = u("as_surv_numeric"),
    description = u("as_surv() on a plain number has no method."),
    input = list(x = enc_input(1)),
    result = tryCatch(as_surv(1), error = describe_error)
  ),
  list(
    id = u("as_surv_character"),
    description = u("as_surv() on a character vector has no method."),
    input = list(x = enc_input("e")),
    result = tryCatch(as_surv("e"), error = describe_error)
  )
)

# -- vector behaviour inherited from vctrs ----------------------------------------------------

# An inventory of what base and vctrs generics do with an event_time vector. Phase 1 decides,
# op by op, which ones Python mirrors and how. Each entry records the R code and its outcome.
x <- event_time(
  time = c(7, 5, 3, 2, NA),
  status = c("e", "r", "l", "i", NA),
  time_max = c(NA, NA, NA, 4, NA)
)
y <- event_time(time = c(1, 9), status = c("r", "i"), time_max = c(NA, 12))

describe_value <- function(v) {
  if (inherits(v, "event_time")) {
    return(c(list(kind = u("event_time")), describe_event_time(v)[-1]))
  }
  if (is.data.frame(v)) {
    return(list(kind = u("data.frame"), print = capture.output(print(v))))
  }
  if (is.logical(v)) {
    return(list(kind = u("logical"), values = enc_lgl(v)))
  }
  if (is.character(v)) {
    return(list(kind = u("character"), values = enc_chr(v)))
  }
  if (is.numeric(v) && length(v) <= 1) {
    return(list(kind = u(typeof(v)), values = v))
  }
  if (is.numeric(v)) {
    return(list(kind = u(typeof(v)), values = v))
  }
  list(kind = u(class(v)[1]), print = capture.output(print(v)))
}

run_op <- function(id, code) {
  env <- new.env(parent = globalenv())
  env$x <- x
  env$y <- y
  out <- capture(eval(parse(text = code), envir = env))
  result <- if (out$ok) {
    c(list(ok = u(TRUE)), describe_value(out$value))
  } else {
    list(ok = u(FALSE), message = u(out$message))
  }
  result$warnings <- out$warnings
  list(id = u(id), r_code = u(code), result = result)
}

vector_op_cases <- list(
  run_op("length", "length(x)"),
  run_op("is_na", "is.na(x)"),
  run_op("any_na", "anyNA(x)"),
  run_op("subset_range", "x[2:4]"),
  run_op("subset_scalar", "x[2]"),
  run_op("subset_na_row", "x[5]"),
  run_op("subset_logical", "x[c(TRUE, FALSE, TRUE, FALSE, TRUE)]"),
  run_op("subset_negative", "x[-1]"),
  run_op("subset_empty", "x[integer()]"),
  run_op("subset_out_of_bounds", "x[10]"),
  run_op("subset_na_index", "x[NA_integer_]"),
  run_op("subset_double_bracket", "x[[2]]"),
  run_op("combine", "c(x, y)"),
  run_op("combine_empty", "c(x[0], y)"),
  run_op("combine_double", "c(x, 1)"),
  run_op("combine_vec_c", "vctrs::vec_c(y, x)"),
  run_op("equal_self", "x == x"),
  run_op("equal_scalar", "x == x[1]"),
  run_op("not_equal_other", "x[1:2] != y"),
  run_op("identical", "identical(x, x[seq_along(x)])"),
  run_op("rev", "rev(x)"),
  run_op("rep", "rep(y, 2)"),
  run_op("unique", "unique(c(x, x))"),
  run_op("duplicated", "duplicated(c(y, y))"),
  run_op("sort", "sort(x[1:4])"),
  run_op("order", "order(x[1:4])"),
  run_op("as_character", "as.character(x)"),
  run_op("as_double", "as.double(x)"),
  run_op("assign_element", "{ z <- x; z[1] <- y[2]; z }"),
  run_op("assign_double", "{ z <- x; z[1] <- 3; z }"),
  run_op("in_data_frame", "data.frame(times = x)"),
  run_op("head", "head(x, 2)"),
  run_op("summary", "summary(x)")
)

# -- format(): R's joint numeric formatting --------------------------------------------------

# format.event_time() formats every number in the vector together with R's rules (7 significant
# digits, shared decimal places, scientific notation when shorter). The hand-picked cases hit
# known corners and the randomized battery covers the rest.
format_case <- function(id, time, status, time_max = NULL) {
  et <- event_time(time, status, time_max)
  list(
    id = u(id),
    input = list(time = enc_input(time), status = enc_input(status), time_max = enc_input(time_max)),
    format = observe(format(et), enc_chr)
  )
}

format_cases <- list(
  format_case("integers", c(1, 10, 100), c("e", "r", "l")),
  format_case("shared_decimals", c(1, 2.5, 3.25), c("e", "e", "e")),
  format_case("seven_significant", c(0.1, 0.123456789), c("e", "r")),
  format_case("tiny", 1e-10, "e"),
  format_case("large_integer", 123456789, "r"),
  format_case("very_large", 1e15, "r"),
  format_case("hundred_thousand", 1e5, "e"),
  format_case("hundred_thousand_and_one", 100001, "e"),
  format_case("mixed_scale", c(1e5, 1e-5), c("e", "e")),
  format_case("zero", 0, "e"),
  format_case("zero_and_fraction", c(0, 0.5), c("e", "r")),
  format_case("interval_shares_digits", c(1, 2), c("e", "i"), c(NA, 4.125)),
  format_case("infinite_upper", 1, "i", Inf),
  format_case("missing_middle", c(1.5, NA, 3), c("e", NA, "r")),
  format_case("all_missing", c(NA_real_, NA_real_), c(NA_character_, NA_character_)),
  format_case("rounding_up", c(0.99999999, 1), c("e", "e")),
  format_case("negative_exponent_boundary", c(0.0001, 0.00001), c("e", "e"))
)

set.seed(20261002)
codes <- c("e", "r", "l", "i")
for (k in seq_len(150)) {
  n <- sample(1:6, 1)
  exponent <- sample(-8:8, n, replace = TRUE)
  sig <- sample(1:10, n, replace = TRUE)
  time <- signif(stats::runif(n, 1, 10) * 10^exponent, sig)
  status <- sample(codes, n, replace = TRUE)
  time_max <- ifelse(
    status == "i",
    signif(time + stats::runif(n, 0.1, 10) * 10^exponent, sample(1:10, n, replace = TRUE)),
    NA_real_
  )
  # Guard against rounding collapsing an interval.
  time_max <- ifelse(status == "i" & time_max <= time, time * 2, time_max)
  if (stats::runif(1) < 0.15) {
    drop <- sample(n, 1)
    time[drop] <- NA_real_
    status[drop] <- NA_character_
    time_max[drop] <- NA_real_
  }
  format_cases[[length(format_cases) + 1]] <- format_case(
    sprintf("random_%03d", k),
    time,
    status,
    if (any(status == "i", na.rm = TRUE)) time_max else NULL
  )
}

# -- write ----------------------------------------------------------------------------------

metadata <- list(
  etd_sha = u(ETD_SHA),
  etd_version = u(as.character(utils::packageVersion("etd"))),
  vctrs_version = u(as.character(utils::packageVersion("vctrs"))),
  survival_version = u(as.character(utils::packageVersion("survival"))),
  r_version = u(R.version.string),
  digits_option = u(getOption("digits")),
  locations = u("1-based, as reported by etd"),
  generated_by = u("scripts/generate_event_time_conformance.R")
)
jsonlite::write_json(
  metadata,
  file.path(out_dir, "metadata.json"),
  auto_unbox = FALSE,
  pretty = TRUE
)
cat(sprintf("wrote %s\n", file.path(out_dir, "metadata.json")))

write_fixture(constructor_cases, "constructor")
write_fixture(new_event_time_cases, "new_event_time")
write_fixture(conversion_cases, "conversion")
write_fixture(vector_op_cases, "vector_ops")
write_fixture(format_cases, "format")
