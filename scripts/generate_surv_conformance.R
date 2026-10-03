#!/usr/bin/env Rscript
# Generate the Surv() conformance fixtures from R's survival package.
#
# Greenwood's `Surv()` mirrors `survival::Surv()` exactly: the same arguments, the same type
# inference, the same status coding, and the same errors and warnings. Each case records a call's
# arguments and what `survival::Surv()` returned (the matrix columns, type, and states) or the
# error it raised. The Python suite (tests/test_surv_conformance.py) replays every case. See
# spec/surv.md.
#
# Run from the repo root:
#
#   Rscript scripts/generate_surv_conformance.R
#
# Writes JSON into spec/conformance/surv/. Encoding conventions match
# scripts/generate_event_time_conformance.R: numeric missing values are "NA" and "NaN",
# infinities are "Inf" and "-Inf", character and logical missing values are null, and every input
# is wrapped as {"r_type": ..., "values": ...}. Arguments that were not supplied are absent.

suppressPackageStartupMessages({
  for (pkg in c("jsonlite", "survival")) {
    if (!requireNamespace(pkg, quietly = TRUE)) {
      stop(sprintf("%s is required: install.packages('%s')", pkg, pkg))
    }
  }
})

out_dir <- file.path("spec", "conformance", "surv")
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

u <- jsonlite::unbox

enc_chr <- function(x) {
  lapply(as.character(x), function(v) if (is.na(v)) NULL else u(v))
}

enc_lgl <- function(x) {
  lapply(x, function(v) if (is.na(v)) NULL else u(v))
}

enc_input <- function(x) {
  if (is.factor(x)) {
    return(list(r_type = u("factor"), values = enc_chr(x), levels = enc_chr(levels(x))))
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

describe_surv <- function(s) {
  m <- unclass(s)
  columns <- lapply(seq_len(ncol(m)), function(j) unname(m[, j]))
  names(columns) <- colnames(m)
  states <- attr(s, "states")
  list(
    ok = u(TRUE),
    type = u(attr(s, "type")),
    states = if (is.null(states)) NULL else enc_chr(states),
    columns = columns
  )
}

# `args` is a named list of the arguments to pass. Absent names are not supplied, which is how R
# tells `Surv(time, event)` apart from `Surv(time, time2, event)`.
run_case <- function(id, description, args) {
  warnings <- character()
  result <- withCallingHandlers(
    tryCatch(
      describe_surv(do.call(survival::Surv, args)),
      error = function(e) list(ok = u(FALSE), message = u(conditionMessage(e)))
    ),
    warning = function(w) {
      warnings <<- c(warnings, conditionMessage(w))
      invokeRestart("muffleWarning")
    }
  )
  result$warnings <- warnings
  list(id = u(id), description = u(description), args = lapply(args, enc_input), result = result)
}

# `type` and `origin` are scalars, so encode them directly rather than as vectors.
case <- function(id, description, ...) {
  args <- list(...)
  out <- run_case(id, description, args)
  if (!is.null(args$type)) out$args$type <- u(args$type)
  if (!is.null(args$origin)) out$args$origin <- u(args$origin)
  out
}

cases <- list(
  # -- right-censored: type inferred from (time, event) -------------------------------------
  case("right_01", "Right-censored with 0/1 status.", time = c(5, 6, 4, 9), event = c(1, 0, 1, 0)),
  case("right_logical", "Logical status.", time = c(5, 6), event = c(TRUE, FALSE)),
  case("right_12", "1/2 status (as in lung) is converted to 0/1.", time = c(5, 6, 4), event = c(2, 1, 2)),
  case(
    "right_012_invalid",
    "0/1/2 status: max is 2, so 0 becomes -1 and is set to NA with a warning.",
    time = c(5, 6, 4),
    event = c(0, 1, 2)
  ),
  case("right_invalid_code", "A status of 3 is invalid and becomes NA.", time = c(5, 6), event = c(1, 3)),
  case("right_status_na", "A missing status stays missing.", time = c(5, 6), event = c(1, NA)),
  case("right_time_na", "A missing time stays missing.", time = c(NA, 6), event = c(1, 0)),
  case("right_time_inf", "An infinite time is allowed.", time = c(Inf, 6), event = c(0, 1)),
  case("right_negative_time", "Negative times are allowed.", time = c(-1, 6), event = c(1, 0)),
  case("right_integer", "Integer time and status.", time = c(5L, 6L), event = c(1L, 0L)),
  case("right_time_only", "Only a time: every row is an event.", time = c(5, 6, 4)),
  case(
    "right_positional",
    "Two positional arguments: the second is the status, passed as time2.",
    time = c(5, 6),
    time2 = c(1, 0)
  ),
  case("right_explicit_type", "type = 'right' given explicitly.", time = c(5, 6), event = c(1, 0), type = "right"),
  case("right_origin", "origin is subtracted from the times.", time = c(5, 6), event = c(1, 0), origin = 2),
  case("right_empty", "Zero-length input.", time = double(), event = double()),
  case("right_all_na_status", "Every status missing.", time = c(5, 6), event = c(NA_real_, NA_real_)),
  case("right_character_status", "Character status is an error.", time = c(5, 6), event = c("a", "b")),
  case("right_time_character", "Character time is an error.", time = c("a", "b"), event = c(1, 0)),
  case("right_time_logical", "Logical time is an error (not numeric).", time = c(TRUE, FALSE), event = c(1, 0)),
  case("right_length_mismatch", "Time and status of different lengths.", time = c(5, 6), event = 1),
  case("right_wrong_arg_count", "type = 'right' with three arguments.", time = 1, time2 = 2, event = 1, type = "right"),

  # -- multi-state: a factor status ----------------------------------------------------------
  case(
    "mright_factor",
    "A factor status gives a multi-state response. The first level is censoring.",
    time = c(5, 6, 7, 8),
    event = factor(c("censor", "pcm", "death", "pcm"), levels = c("censor", "pcm", "death"))
  ),
  case(
    "mright_factor_na",
    "A missing factor value is a missing status.",
    time = c(5, 6),
    event = factor(c("pcm", NA), levels = c("censor", "pcm"))
  ),
  case(
    "mright_type_character",
    "type = 'mstate' with character status: levels sort alphabetically.",
    time = c(5, 6, 7),
    event = c("death", "alive", "pcm"),
    type = "mstate"
  ),
  case(
    "mright_type_numeric",
    "type = 'mstate' with numeric status: levels sort numerically, so 0 is censoring.",
    time = c(5, 6, 7),
    event = c(0, 2, 1),
    type = "mstate"
  ),
  case(
    "mright_blank_state",
    "A blank state name is an error.",
    time = c(5, 6),
    event = factor(c("a", ""), levels = c("a", ""))
  ),

  # -- left-censored --------------------------------------------------------------------------
  case("left_01", "Left-censored: 1 is an exact event, 0 is left-censored.", time = c(5, 6, 4), event = c(1, 0, 1), type = "left"),
  case("left_12", "Left-censored with 1/2 status.", time = c(5, 6), event = c(2, 1), type = "left"),
  case("left_wrong_arg_count", "type = 'left' with only a time.", time = c(5, 6), type = "left"),

  # -- counting process -----------------------------------------------------------------------
  case(
    "counting_01",
    "Counting-process (start, stop, event).",
    time = c(0, 2, 1),
    time2 = c(5, 6, 4),
    event = c(1, 0, 1)
  ),
  case(
    "counting_start_after_stop",
    "start >= stop sets start to NA with a warning.",
    time = c(0, 6, 4),
    time2 = c(5, 6, 3),
    event = c(1, 0, 1)
  ),
  case("counting_12", "Counting-process with 1/2 status.", time = c(0, 2), time2 = c(5, 6), event = c(2, 1)),
  case("counting_logical", "Counting-process with logical status.", time = c(0, 2), time2 = c(5, 6), event = c(TRUE, FALSE)),
  case("counting_origin", "origin applies to start and stop.", time = c(1, 2), time2 = c(5, 6), event = c(1, 0), origin = 1),
  case(
    "mcounting_factor",
    "Counting-process with a factor status.",
    time = c(0, 2),
    time2 = c(5, 6),
    event = factor(c("pcm", "censor"), levels = c("censor", "pcm"))
  ),
  case("counting_stop_length", "start and stop of different lengths.", time = c(0, 2), time2 = 5, event = c(1, 0)),
  case("counting_event_length", "start and event of different lengths.", time = c(0, 2), time2 = c(5, 6), event = 1),
  case("counting_stop_character", "Character stop time.", time = c(0, 2), time2 = c("a", "b"), event = c(1, 0)),
  case("counting_status_character", "Character status.", time = c(0, 2), time2 = c(5, 6), event = c("a", "b")),
  case("counting_wrong_arg_count", "type = 'counting' with two arguments.", time = c(0, 2), event = c(1, 0), type = "counting"),

  # -- interval (time1, time2, coded event) ----------------------------------------------------
  case(
    "interval_codes",
    "Interval coding: 0 right, 1 exact, 2 left, 3 interval. time2 is 1 except for interval rows.",
    time = c(5, 6, 4, 2),
    time2 = c(NA, NA, NA, 4),
    event = c(0, 1, 2, 3),
    type = "interval"
  ),
  case(
    "interval_no_intervals",
    "No interval rows: time2 is never used.",
    time = c(5, 6),
    time2 = c(NA, NA),
    event = c(0, 1),
    type = "interval"
  ),
  case(
    "interval_invalid_code",
    "A code of 4 becomes NA with a warning.",
    time = c(5, 6),
    time2 = c(7, 8),
    event = c(1, 4),
    type = "interval"
  ),
  case(
    "interval_backwards",
    "An interval with time > time2 becomes NA with a warning.",
    time = c(5, 2),
    time2 = c(3, 4),
    event = c(3, 3),
    type = "interval"
  ),
  case(
    "interval_origin",
    "origin applies to both times.",
    time = c(5, 2),
    time2 = c(NA, 4),
    event = c(1, 3),
    type = "interval",
    origin = 1
  ),
  case(
    "interval_status_na",
    "A missing code stays missing.",
    time = c(5, 2),
    time2 = c(NA, 4),
    event = c(NA, 3),
    type = "interval"
  ),
  case(
    "interval_status_character",
    "Character codes are an error.",
    time = c(5, 2),
    time2 = c(6, 4),
    event = c("a", "b"),
    type = "interval"
  ),
  case(
    "interval_time2_length",
    "time2 of the wrong length when there are interval rows.",
    time = c(5, 2),
    time2 = 4,
    event = c(1, 3),
    type = "interval"
  ),
  case(
    "interval_wrong_arg_count",
    "type = 'interval' with two arguments.",
    time = c(5, 2),
    event = c(1, 3),
    type = "interval"
  ),

  # -- interval2 (lower, upper with open bounds) -------------------------------------------------
  case(
    "interval2_bounds",
    "interval2: equal bounds are exact, an open upper is right, an open lower is left.",
    time = c(1, 2, NA, 3),
    time2 = c(1, NA, 4, 5),
    type = "interval2"
  ),
  case(
    "interval2_infinite",
    "Infinite bounds count as open.",
    time = c(1, -Inf),
    time2 = c(Inf, 4),
    type = "interval2"
  ),
  case(
    "interval2_backwards",
    "lower > upper becomes NA with a warning.",
    time = c(5, 1),
    time2 = c(3, 2),
    type = "interval2"
  ),
  case(
    "interval2_unknown",
    "Both bounds missing becomes NA.",
    time = c(NA, 1),
    time2 = c(NA, 2),
    type = "interval2"
  ),
  case(
    "interval2_origin",
    "origin applies to both bounds.",
    time = c(2, 3),
    time2 = c(4, NA),
    type = "interval2",
    origin = 1
  ),
  case(
    "interval2_upper_character",
    "Character upper bound is an error.",
    time = c(1, 2),
    time2 = c("a", "b"),
    type = "interval2"
  ),
  case(
    "interval2_length",
    "Bounds of different lengths.",
    time = c(1, 2),
    time2 = 3,
    type = "interval2"
  ),
  case(
    "interval2_wrong_arg_count",
    "type = 'interval2' with three arguments.",
    time = c(1, 2),
    time2 = c(3, 4),
    event = c(1, 1),
    type = "interval2"
  )
)

metadata <- list(
  survival_version = u(as.character(utils::packageVersion("survival"))),
  r_version = u(R.version.string),
  generated_by = u("scripts/generate_surv_conformance.R")
)
jsonlite::write_json(metadata, file.path(out_dir, "metadata.json"), auto_unbox = FALSE, pretty = TRUE)
cat(sprintf("wrote %s\n", file.path(out_dir, "metadata.json")))

path <- file.path(out_dir, "surv.json")
jsonlite::write_json(cases, path, auto_unbox = FALSE, digits = NA, pretty = TRUE, null = "null", na = "string")
cat(sprintf("wrote %s (%d cases)\n", path, length(cases)))
