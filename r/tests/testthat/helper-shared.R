# Helpers for replaying the cases shared with the Python package.
#
# The repository root holds `spec/` (contracts and conformance cases) and `fixtures/r/` (numeric
# fixtures). Both are generated in R by the scripts in `scripts/`, and the Python and R test suites
# replay the same JSON. They live outside the R package, so they are found by walking up from the
# test directory. That works for `devtools::test()` and for `R CMD check` run inside the repo. Set
# `GREENWOOD_REPO_ROOT` to point elsewhere. When the root can't be found (say, a check of the
# built tarball on its own), the shared tests skip.

repo_root <- function() {
  root <- Sys.getenv("GREENWOOD_REPO_ROOT", unset = "")
  if (nzchar(root)) {
    return(normalizePath(root, mustWork = TRUE))
  }
  dir <- normalizePath(getwd())
  repeat {
    if (dir.exists(file.path(dir, "spec", "conformance"))) {
      return(dir)
    }
    parent <- dirname(dir)
    if (identical(parent, dir)) {
      return(NULL)
    }
    dir <- parent
  }
}

skip_if_no_shared_cases <- function() {
  testthat::skip_if_not_installed("jsonlite")
  testthat::skip_if(is.null(repo_root()), "The shared spec/ directory is not available.")
}

read_conformance <- function(area, name) {
  path <- file.path(repo_root(), "spec", "conformance", area, paste0(name, ".json"))
  jsonlite::read_json(path, simplifyVector = FALSE)
}

read_fixture <- function(name) {
  path <- file.path(repo_root(), "fixtures", "r", paste0(name, ".json"))
  jsonlite::read_json(path, simplifyVector = FALSE)
}

# -- decoding -------------------------------------------------------------------------------
#
# The generators write numeric missing values as "NA" and "NaN", infinities as "Inf" and "-Inf",
# and character and logical missing values as JSON null. Inputs are wrapped as
# {"r_type": ..., "values": ...}, so each one is rebuilt as exactly the R value the generator used.

r_numbers <- function(values) {
  vapply(
    values,
    function(v) {
      if (is.null(v)) {
        return(NA_real_)
      }
      if (is.character(v)) {
        return(switch(v, "NA" = NA_real_, "NaN" = NaN, "Inf" = Inf, "-Inf" = -Inf))
      }
      as.double(v)
    },
    double(1)
  )
}

r_strings <- function(values) {
  vapply(values, function(v) if (is.null(v)) NA_character_ else v, character(1))
}

r_logicals <- function(values) {
  vapply(values, function(v) if (is.null(v)) NA else v, logical(1))
}

decode_input <- function(x) {
  if (is.null(x)) {
    return(NULL)
  }
  # Scalar arguments such as `type` and `origin` are written directly, not wrapped.
  if (!is.list(x) || is.null(x$r_type)) {
    return(x)
  }
  switch(
    x$r_type,
    double = r_numbers(x$values),
    integer = as.integer(r_numbers(x$values)),
    logical = r_logicals(x$values),
    character = r_strings(x$values),
    factor = factor(r_strings(x$values), levels = r_strings(x$levels)),
    list = lapply(x$values, decode_input),
    stop(sprintf("Unknown r_type: %s", x$r_type))
  )
}

# -- running ----------------------------------------------------------------------------------

# Evaluate `expr` the way the generators did: the plain-text cli options they set, warnings
# collected rather than printed, and an error recorded rather than raised.
capture <- function(expr) {
  old <- options(cli.num_colors = 1, cli.unicode = FALSE, useFancyQuotes = FALSE, width = 80)
  on.exit(options(old), add = TRUE)
  warnings <- character()
  out <- withCallingHandlers(
    tryCatch(
      list(ok = TRUE, value = force(expr)),
      error = function(e) {
        # cli wraps long messages at the console width, and the generators collapse the wrapping.
        list(ok = FALSE, message = gsub("[ \t]*\n[ \t]*", " ", conditionMessage(e)))
      }
    ),
    warning = function(w) {
      warnings <<- c(warnings, conditionMessage(w))
      invokeRestart("muffleWarning")
    }
  )
  out$warnings <- warnings
  out
}

# A `Surv` object as the generators describe it: its type, states, and matrix columns.
surv_columns <- function(s) {
  m <- unclass(s)
  columns <- lapply(seq_len(ncol(m)), function(j) unname(m[, j]))
  names(columns) <- colnames(m)
  columns
}

expect_surv_matches <- function(actual, expected, label) {
  testthat::expect_s3_class(actual, "Surv")
  testthat::expect_identical(attr(actual, "type"), expected$type, label = label)
  expected_states <- if (is.null(expected$states)) NULL else r_strings(expected$states)
  testthat::expect_identical(attr(actual, "states"), expected_states, label = label)
  columns <- surv_columns(actual)
  testthat::expect_identical(names(columns), names(expected$columns), label = label)
  for (name in names(expected$columns)) {
    testthat::expect_equal(
      columns[[name]],
      r_numbers(expected$columns[[name]]),
      tolerance = 1e-12,
      label = paste(label, name)
    )
  }
}
