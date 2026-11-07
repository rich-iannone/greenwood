# Episode splitting for time-varying covariates: split_episodes(). Tested by
# tests/testthat/test-split-episodes.R, against the rules of each interval and a Cox fit to the
# pbcseq visits.

#' Split follow-up into episodes of constant covariates
#'
#' `split_episodes()` merges a table of subjects with a table of their visits into the
#' counting-process layout, one row per interval `(tstart, tstop]` over which a subject's
#' covariates stay the same. The result can be fit with a response such as
#' `Surv(tstart, tstop, event)`, so the covariates measured at each visit enter a [cox_ph()]
#' model as time-varying covariates.
#'
#' A value measured at a visit applies from that visit to the next one. The event of a subject is
#' kept only on its last interval, and every earlier interval is censored. A visit at the end of
#' follow-up doesn't start a new interval: the interval that ends there keeps the event and the
#' values measured before it. Before a subject's first visit (when it's after time 0) the
#' time-varying covariates are missing, and so are those of a subject with no visits.
#'
#' @param baseline A data frame of the subjects, with the `id`, `time`, and `event` columns. It can
#'   have one row per subject, or be the same long table as `visits` when that table repeats each
#'   subject's follow-up time and event on every row. Then the first row of each subject is used,
#'   after checking that `time`, `event`, and the `baseline_covariates` don't change within it.
#' @param visits A data frame with one row per visit, with the `id` and `visit_time` columns.
#' @param id The name of the column that identifies the subjects in both tables.
#' @param time The name of the column of `baseline` with the end of follow-up.
#' @param event The name of the column of `baseline` with the event indicator: logical, numeric,
#'   or a factor whose first level means censored.
#' @param visit_time The name of the column of `visits` with the time of each visit.
#' @param covariates The names of the columns of `visits` to carry as time-varying covariates. The
#'   default is every column other than `id`, `visit_time`, `time`, `event`, and the
#'   `baseline_covariates`.
#' @param baseline_covariates The names of the columns of `baseline` to carry as fixed covariates.
#'   The default is every column other than `id`, `time`, and `event` when `baseline` has one row
#'   per subject, and none when it repeats subjects (its other columns then usually change from
#'   visit to visit).
#' @param carry_forward Whether to carry each subject's last measured values to the end of
#'   follow-up. The default is `TRUE`. With `FALSE`, follow-up stops at the last visit, so the
#'   interval after it is dropped (and with it the event), and a subject with no visits is
#'   dropped. A subject whose only interval is the one after its last visit keeps it.
#'
#' @return A data frame with the columns `id`, `tstart`, `tstop`, `event` (named as in `baseline`),
#'   then the fixed covariates and the time-varying covariates. The rows are ordered by subject,
#'   in the order of `baseline`, and then by time.
#'
#' @seealso [cox_ph()] for fitting the result, and its [predict()][cox_ph_predict] method's
#'   `trajectory` argument for the survival of a subject along a path of covariates.
#'
#' @examples
#' baseline <- data.frame(id = 1:6, time = c(10, 8, 12, 6, 15, 9), event = c(1, 0, 1, 1, 0, 1))
#' visits <- data.frame(
#'   id = c(1, 1, 2, 3, 3, 3, 4, 4, 5, 5, 6),
#'   day = c(0, 5, 0, 0, 4, 8, 0, 3, 0, 7, 0),
#'   bili = c(1.2, 2.4, 0.8, 3.1, 2.9, 4, 2.2, 3.5, 0.9, 1.1, 1.8)
#' )
#' episodes <- split_episodes(baseline,
#'   visits = visits, id = "id", time = "time", event = "event", visit_time = "day"
#' )
#' episodes
#'
#' cox_ph(Surv(tstart, tstop, event) ~ bili, data = episodes)
#' @export
split_episodes <- function(baseline,
                           visits,
                           id,
                           time,
                           event,
                           visit_time,
                           covariates = NULL,
                           baseline_covariates = NULL,
                           carry_forward = TRUE) {
  if (!is.data.frame(baseline)) {
    stop("`baseline` must be a data frame.", call. = FALSE)
  }
  if (!is.data.frame(visits)) {
    stop("`visits` must be a data frame.", call. = FALSE)
  }
  for (arg in c("id", "time", "event", "visit_time")) {
    value <- get(arg)
    if (!is.character(value) || length(value) != 1L || is.na(value)) {
      stop(sprintf("`%s` must be a column name.", arg), call. = FALSE)
    }
  }
  check_flag(carry_forward, "carry_forward")
  check_columns(baseline, c(id, time, event, baseline_covariates), "baseline")
  check_columns(visits, c(id, visit_time, covariates), "visits")

  base_ids <- baseline[[id]]
  repeated <- anyDuplicated(base_ids) > 0L
  static <- if (!is.null(baseline_covariates)) {
    baseline_covariates
  } else if (repeated) {
    character(0)
  } else {
    setdiff(names(baseline), c(id, time, event))
  }
  varying <- if (!is.null(covariates)) {
    covariates
  } else {
    setdiff(names(visits), c(id, visit_time, time, event, static))
  }
  clash <- intersect(static, varying)
  if (length(clash) > 0L) {
    stop(
      sprintf(
        paste(
          "%s %s both fixed (from `baseline`) and time-varying (from `visits`). Choose the",
          "columns with `baseline_covariates` and `covariates`."
        ),
        paste0("`", clash, "`", collapse = ", "), if (length(clash) == 1L) "is" else "are"
      ),
      call. = FALSE
    )
  }

  keep <- first_row_per_subject(baseline, id, c(time, event, static))
  subject_ids <- base_ids[keep]
  end <- baseline[[time]][keep]
  status <- baseline[[event]][keep]
  if (!is.numeric(end) || anyNA(end)) {
    stop(sprintf("`%s` must be numbers with no missing values.", time), call. = FALSE)
  }
  if (!is.logical(status) && !is.numeric(status) && !is.factor(status)) {
    stop(sprintf("`%s` must be logical, numeric, or a factor.", event), call. = FALSE)
  }
  visit_ids <- visits[[id]]
  at <- visits[[visit_time]]
  if (!is.numeric(at) || anyNA(at)) {
    stop(sprintf("`%s` must be numbers with no missing values.", visit_time), call. = FALSE)
  }
  if (anyNA(base_ids) || anyNA(visit_ids)) {
    stop(sprintf("`%s` has missing values.", id), call. = FALSE)
  }

  # Each subject's visits, ordered by time (ties in the order of `visits`).
  owner <- match(visit_ids, subject_ids)
  ordered <- order(owner, at, method = "radix")
  ordered <- ordered[!is.na(owner[ordered])]
  by_subject <- split(ordered, factor(owner[ordered], levels = seq_along(subject_ids)))

  pieces <- lapply(seq_along(subject_ids), function(i) {
    rows <- episode_rows(by_subject[[i]], at, end[i], subject_ids[i], carry_forward)
    rows$subject <- rep(i, nrow(rows))
    rows
  })
  if (!any(lengths(by_subject) > 0L) && all(vapply(pieces, nrow, integer(1)) == 0L)) {
    stop(
      paste(
        "`split_episodes()` produced no rows. Check that the `id` columns of `baseline` and",
        "`visits` share values."
      ),
      call. = FALSE
    )
  }
  rows <- do.call(rbind, pieces)
  subject <- rows$subject

  # Only an interval that ends with follow-up keeps the event.
  out_event <- status[subject]
  censored <- if (is.factor(status)) levels(status)[1L] else if (is.logical(status)) FALSE else 0
  out_event[rows$tstop != end[subject]] <- censored
  out <- list()
  out[[id]] <- subject_ids[subject]
  out$tstart <- rows$tstart
  out$tstop <- rows$tstop
  out[[event]] <- out_event
  for (col in static) {
    out[[col]] <- baseline[[col]][keep][subject]
  }
  for (col in varying) {
    out[[col]] <- visits[[col]][rows$visit]
  }
  as.data.frame(out, stringsAsFactors = FALSE, optional = TRUE)
}

# The intervals of the subject labeled `label`, with follow-up ending at `end`: the visit row
# whose values hold on each (NA before the first visit), the bounds, and whether the interval
# keeps the subject's event. `visits` are the rows of the subject's visits, ordered by time.
episode_rows <- function(visits, at, end, label, carry_forward) {
  times <- at[visits]
  late <- times[times > end]
  if (length(late) > 0L) {
    stop(
      sprintf(
        paste(
          "Subject %s has a visit at %s, after its follow-up ends at %s. Check that the visit",
          "times are on the same scale as the follow-up time."
        ),
        format(label), format(late[1L]), format(end)
      ),
      call. = FALSE
    )
  }
  if (length(visits) == 0L) {
    return(data.frame(visit = NA_integer_, tstart = 0, tstop = end)[carry_forward, ])
  }
  stops <- c(times[-1L], end)
  rows <- data.frame(visit = visits, tstart = times, tstop = stops)
  # A repeated visit time or a visit at the end of follow-up starts no interval.
  rows <- rows[times < stops, , drop = FALSE]
  if (carry_forward && times[1L] > 0) {
    rows <- rbind(data.frame(visit = NA_integer_, tstart = 0, tstop = times[1L]), rows)
  }
  # Without carrying forward, follow-up stops at the last visit, unless that leaves no interval.
  after_last <- rows$visit %in% visits[length(visits)]
  if (!carry_forward && any(after_last) && nrow(rows) > 1L) {
    rows <- rows[!after_last, , drop = FALSE]
  }
  rows
}

# The first row of each subject in `frame`, after checking that the `constant` columns are the
# same on every row of the subject (with missing values equal to each other).
first_row_per_subject <- function(frame, id, constant) {
  ids <- frame[[id]]
  first <- match(ids, ids)
  for (col in constant) {
    values <- frame[[col]]
    same <- values == values[first]
    same[is.na(same)] <- is.na(values)[is.na(same)] & is.na(values[first])[is.na(same)]
    if (!all(same)) {
      row <- which(!same)[1L]
      stop(
        sprintf(
          paste(
            "Subject %s has more than one value of `%s` in `baseline` (%s and %s). A baseline",
            "with repeated subjects must hold the same follow-up time, event, and baseline",
            "covariates on every row."
          ),
          format(ids[row]), col, format(values[first[row]]), format(values[row])
        ),
        call. = FALSE
      )
    }
  }
  which(first == seq_along(ids))
}

check_columns <- function(frame, columns, name) {
  absent <- setdiff(columns, names(frame))
  if (length(absent) > 0L) {
    stop(
      sprintf(
        "%s %s not in `%s`.", paste0("`", absent, "`", collapse = ", "),
        if (length(absent) == 1L) "is" else "are", name
      ),
      call. = FALSE
    )
  }
}
