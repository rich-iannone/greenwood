# Replays spec/conformance/event_time/, generated from the etd package by
# scripts/generate_event_time_conformance.R. The Python suite replays the same cases
# (python/tests/test_event_time_conformance.py). See spec/event_time.md.

# Observations where etd itself fails, recorded in spec/event_time.md as upstream issues. They
# aren't mirrored, so they aren't compared.
UPSTREAM_ISSUES <- c(
  "constructor/empty/format",
  "new_event_time/empty/format",
  "vector_ops/subset_empty/format"
)

# Operations in vector_ops.json that aren't mirrored, with the reason (spec/event_time.md, "Vector
# behaviour").
NOT_MIRRORED <- c(
  combine_vec_c = "vctrs::vec_c() needs a vctrs class.",
  subset_na_index = "A missing index gives a missing element rather than a broken one.",
  sort = "An event-time vector has no order, so sort() is an error.",
  order = "An event-time vector has no order, so order() is an error."
)

# The recorded messages carry cli's bullets ("x", "i") after the first sentence, on new lines in
# vector_ops.json. Greenwood's messages are plain sentences on one line.
plain_message <- function(message) {
  gsub("\\. [xi] ", ". ", gsub("[ \t]*\n[ \t]*", " ", message))
}

expect_numbers <- function(actual, expected, label) {
  expect_equal(unname(actual), r_numbers(expected), tolerance = 1e-12, label = label)
}

# Every observation of a valid vector, as describe_event_time() records it in the generator.
expect_event_time_matches <- function(x, expected, key) {
  expect_s3_class(x, "event_time")
  expect_identical(length(x), as.integer(expected$length), label = key)
  times <- .subset2(x, "time")
  expect_identical(length(times), length(expected$fields$time), label = key)
  for (i in seq_along(times)) {
    expect_numbers(times[[i]], expected$fields$time[[i]], paste(key, "time", i))
  }
  expect_identical(.subset2(x, "status"), r_strings(expected$fields$status), label = key)
  expect_identical(is.na(x), r_logicals(expected$is_na$value), label = key)
  if (!paste0(key, "/format") %in% UPSTREAM_ISSUES) {
    expect_true(expected$format$ok, label = key)
    expect_identical(format(x), r_strings(expected$format$value), label = key)
  }
  expect_identical(
    capture.output(print(x)), r_strings(expected$print$value),
    label = paste(key, "print")
  )

  extracted <- expected$extract_time$value
  if (extracted$shape == "vector") {
    expect_null(dim(extract_time(x)), label = key)
    expect_numbers(extract_time(x), extracted$values, paste(key, "extract_time"))
  } else {
    expect_identical(colnames(extract_time(x)), c("time", "time_max"), label = key)
    expect_numbers(extract_time(x)[, "time"], extracted$columns$time, key)
    expect_numbers(extract_time(x)[, "time_max"], extracted$columns$time_max, key)
  }
  expect_identical(extract_status(x), r_strings(expected$extract_status$value), label = key)

  surv <- suppressWarnings(as_surv(x))
  expect_identical(attr(surv, "type"), expected$as_surv$value$type, label = key)
  columns <- surv_columns(surv)
  expect_identical(names(columns), names(expected$as_surv$value$columns), label = key)
  for (name in names(columns)) {
    expect_numbers(columns[[name]], expected$as_surv$value$columns[[name]], paste(key, name))
  }

  if (requireNamespace("tibble", quietly = TRUE)) {
    frame <- tibble::as_tibble(x)
    expect_named(frame, c("time", "status", "time_max"))
    expect_numbers(frame$time, expected$as_tibble$value$time, key)
    expect_identical(frame$status, r_strings(expected$as_tibble$value$status), label = key)
    expect_numbers(frame$time_max, expected$as_tibble$value$time_max, key)
  }
}

expect_error_matches <- function(out, expected, key) {
  expect_false(out$ok, label = key)
  expect_identical(out$message, plain_message(expected$message), label = key)
}

# The condition of a failed construction, with its check id and locations.
condition_of <- function(expr) {
  tryCatch(
    {
      force(expr)
      NULL
    },
    greenwood_event_time_error = function(e) e
  )
}

test_that("the fixtures are pinned to the etd commit in spec/event_time.md", {
  skip_if_no_shared_cases()
  metadata <- read_conformance("event_time", "metadata")
  spec <- readLines(file.path(repo_root(), "spec", "event_time.md"))
  expect_true(any(grepl(metadata$etd_sha, spec, fixed = TRUE)))
})

test_that("event_time() matches every recorded constructor case", {
  skip_if_no_shared_cases()
  cases <- read_conformance("event_time", "constructor")
  expect_gt(length(cases), 0)
  for (case in cases) {
    key <- paste0("constructor/", case$id)
    build <- function() {
      event_time(
        decode_input(case$input$time),
        decode_input(case$input$status),
        decode_input(case$input$time_max)
      )
    }
    out <- capture(build())
    if (case$result$ok) {
      expect_true(out$ok, label = key)
      expect_event_time_matches(out$value, case$result, key)
    } else {
      expect_error_matches(out, case$result, key)
      condition <- condition_of(build())
      expect_identical(condition$check, case$result$check, label = key)
      expected_locations <- as.integer(unlist(case$result$locations))
      expect_identical(condition$locations, expected_locations, label = key)
    }
  }
})

test_that("new_event_time() matches every recorded case", {
  skip_if_no_shared_cases()
  for (case in read_conformance("event_time", "new_event_time")) {
    key <- paste0("new_event_time/", case$id)
    build <- function() {
      new_event_time(decode_input(case$input$time), decode_input(case$input$status))
    }
    out <- capture(build())
    if (case$result$ok) {
      expect_true(out$ok, label = key)
      expect_event_time_matches(out$value, case$result, key)
    } else {
      expect_error_matches(out, case$result, key)
      expect_identical(condition_of(build())$check, case$result$check, label = key)
    }
  }
})

test_that("as_surv() rejects what it can't convert", {
  skip_if_no_shared_cases()
  for (case in read_conformance("event_time", "conversion")) {
    out <- capture(as_surv(decode_input(case$input$x)))
    expect_error_matches(out, case$result, case$id)
    expect_identical(
      condition_of(as_surv(decode_input(case$input$x)))$check, case$result$check,
      label = case$id
    )
  }
})

test_that("format() matches R's joint number formatting in every recorded case", {
  skip_if_no_shared_cases()
  cases <- read_conformance("event_time", "format")
  expect_gt(length(cases), 100)
  for (case in cases) {
    x <- event_time(
      decode_input(case$input$time),
      decode_input(case$input$status),
      decode_input(case$input$time_max)
    )
    out <- capture(format(x))
    expect_identical(out$value, r_strings(case$format$value), label = case$id)
  }
})

test_that("the vector operations match the recorded ones", {
  skip_if_no_shared_cases()
  x <- event_time(
    time = c(7, 5, 3, 2, NA),
    status = c("e", "r", "l", "i", NA),
    time_max = c(NA, NA, NA, 4, NA)
  )
  y <- event_time(time = c(1, 9), status = c("r", "i"), time_max = c(NA, 12))
  for (case in read_conformance("event_time", "vector_ops")) {
    if (case$id %in% names(NOT_MIRRORED)) {
      next
    }
    key <- paste0("vector_ops/", case$id)
    env <- new.env(parent = environment())
    env$x <- x
    env$y <- y
    out <- capture(eval(str2lang(case$r_code), envir = env))
    expected <- case$result
    if (!expected$ok) {
      expect_error_matches(out, expected, key)
      next
    }
    expect_true(out$ok, label = key)
    value <- out$value
    switch(expected$kind,
      event_time = expect_event_time_matches(value, expected, key),
      logical = expect_identical(value, r_logicals(expected$values), label = key),
      integer = expect_identical(value, as.integer(unlist(expected$values)), label = key),
      data.frame = expect_identical(
        capture.output(print(value)), unlist(expected$print),
        label = key
      ),
      stop(sprintf("Unknown kind: %s", expected$kind))
    )
  }
})
