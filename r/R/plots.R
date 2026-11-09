# The ggplot2 plots: survival and cumulative incidence curves with their numbers-at-risk tables,
# per-subject predicted curves, forest plots, and the Cox and parametric diagnostics. Each returns
# a ggplot object (or a patchwork of two, with a risk table) to add layers and themes to. They're
# tested by tests/testthat/test-plots.R.

#' @importFrom ggplot2 .data
NULL

PLOT_COLOR <- "#20558A"
HIGHLIGHT_COLOR <- "#E45756"
REFERENCE_COLOR <- "#888888"

# -- themes ----------------------------------------------------------------------------------------

#' Themes for survival plots and forest plots
#'
#' `theme_survival()` is the light theme of [plot_survival()] and [plot_cif()]: a minimal theme
#' with the legend on top and no minor grid lines. `theme_forest()` is the theme of
#' [plot_forest()], which also drops the horizontal grid lines, the y axis line, and the legend,
#' so the term labels stand on their own.
#'
#' @return A ggplot2 theme, to add to a plot with `+`.
#'
#' @examples
#' fit <- kaplan_meier(Surv(time, status) ~ sex, data = lung)
#' plot_survival(fit) + theme_survival()
#'
#' cox <- cox_ph(Surv(time, status) ~ age + sex, data = lung)
#' plot_forest(cox) + theme_forest()
#' @export
theme_survival <- function() {
  ggplot2::theme_minimal() +
    ggplot2::theme(
      legend.position = "top",
      panel.grid.minor = ggplot2::element_blank()
    )
}

#' @rdname theme_survival
#' @export
theme_forest <- function() {
  ggplot2::theme_minimal() +
    ggplot2::theme(
      axis.line.y = ggplot2::element_blank(),
      panel.grid.minor = ggplot2::element_blank(),
      panel.grid.major.y = ggplot2::element_blank(),
      legend.position = "none"
    )
}

# -- survival curves -------------------------------------------------------------------------------

#' Plot Kaplan-Meier survival curves
#'
#' `plot_survival()` draws the survival curves of a [kaplan_meier()] fit as step functions, with
#' their pointwise confidence bands and a `+` at each censoring time. A fit with groups has one
#' curve per group, told apart by color. The result is a ggplot object, so layers, scales, and
#' themes can be added to it.
#'
#' With `risk_table = TRUE`, the numbers at risk (see [risk_table()]) are stacked under the curves,
#' on the same time axis. This needs the patchwork package, and the result is a patchwork.
#'
#' @param fit A [kaplan_meier()] fit.
#' @param conf_int Draw the confidence bands.
#' @param censor_marks Mark the censoring times on the curves.
#' @param risk_table Stack a numbers-at-risk table under the curves.
#' @param times The times of the numbers-at-risk table. The default is six evenly spaced, rounded
#'   times from 0 to the end of follow-up.
#' @param xlab,ylab The axis labels.
#'
#' @return A ggplot object, or a patchwork with `risk_table = TRUE`.
#'
#' @seealso [theme_survival()], [risk_table()], and [plot_cif()] for competing risks.
#'
#' @examples
#' fit <- kaplan_meier(Surv(time, status) ~ sex, data = lung)
#' plot_survival(fit)
#' plot_survival(fit, conf_int = FALSE, censor_marks = FALSE)
#'
#' if (requireNamespace("patchwork", quietly = TRUE)) {
#'   plot_survival(fit, risk_table = TRUE, times = c(0, 250, 500, 750, 1000))
#' }
#' @export
plot_survival <- function(fit,
                          conf_int = TRUE,
                          censor_marks = TRUE,
                          risk_table = FALSE,
                          times = NULL,
                          xlab = "Time",
                          ylab = "Survival probability") {
  check_fit_class(fit, "kaplan_meier", "plot_survival()")
  check_flag(conf_int, "conf_int")
  check_flag(censor_marks, "censor_marks")
  check_flag(risk_table, "risk_table")
  check_label(xlab, "xlab")
  check_label(ylab, "ylab")
  if (risk_table) {
    require_patchwork("plot_survival(risk_table = TRUE)")
  }
  table <- as.data.frame(fit)
  labels <- curve_labels(fit)
  table$strata <- factor(table$strata, levels = labels)
  grouped <- !identical(labels, "all")
  steps <- do.call(rbind, lapply(split(table, table$strata), step_coordinates, start = 1))

  plot <- ggplot2::ggplot(steps, ggplot2::aes(x = .data$time, y = .data$estimate))
  if (conf_int) {
    ribbon <- steps[!is.na(steps$conf_low) & !is.na(steps$conf_high), , drop = FALSE]
    plot <- plot + if (grouped) {
      ggplot2::geom_ribbon(
        ggplot2::aes(ymin = .data$conf_low, ymax = .data$conf_high, fill = .data$strata),
        data = ribbon, alpha = 0.18, show.legend = FALSE
      )
    } else {
      ggplot2::geom_ribbon(
        ggplot2::aes(ymin = .data$conf_low, ymax = .data$conf_high),
        data = ribbon, alpha = 0.18, fill = PLOT_COLOR
      )
    }
  }
  plot <- plot + if (grouped) {
    ggplot2::geom_line(ggplot2::aes(color = .data$strata))
  } else {
    ggplot2::geom_line(color = PLOT_COLOR)
  }
  if (censor_marks) {
    censored <- table[table$n_censor > 0, , drop = FALSE]
    if (nrow(censored) > 0L) {
      plot <- plot + if (grouped) {
        ggplot2::geom_point(
          ggplot2::aes(color = .data$strata),
          data = censored, shape = 3, size = 2, show.legend = FALSE
        )
      } else {
        ggplot2::geom_point(data = censored, shape = 3, size = 2, color = PLOT_COLOR)
      }
    }
  }
  plot <- plot +
    ggplot2::scale_y_continuous(limits = c(0, 1)) +
    ggplot2::labs(x = xlab, y = ylab) +
    group_legend(labels, grouped, fill = conf_int) +
    theme_survival()
  if (!risk_table) {
    return(plot)
  }
  stack_risk_table(plot, risk_table_plot(get_risk_table_frame(fit, times), grouped, xlab), steps)
}

#' Numbers at risk
#'
#' The number of subjects still at risk in each group of a [kaplan_meier()] or [aalen_johansen()]
#' fit at a set of times, the table printed under a survival plot. `get_risk_table_frame()` gives
#' the numbers as a data frame, and `risk_table()` draws them as a ggplot object, a table of text
#' whose time axis lines up with a curve drawn over the same range.
#'
#' The number at risk at time \eqn{t} is the number at risk at the first observed time at or after
#' \eqn{t}, so it counts the subjects still under follow-up just before \eqn{t}. It's 0 after the
#' end of follow-up.
#'
#' @param fit A [kaplan_meier()] or [aalen_johansen()] fit.
#' @param times The times at which to count. The default is six evenly spaced, rounded times from
#'   0 to the end of follow-up.
#'
#' @return `get_risk_table_frame()` gives a data frame with the columns `strata`, `time`, and
#'   `n_risk`, one row per group and time. `risk_table()` gives a ggplot object.
#'
#' @seealso [plot_survival()] and [plot_cif()], which stack the table under their curves with
#'   `risk_table = TRUE`.
#'
#' @examples
#' fit <- kaplan_meier(Surv(time, status) ~ sex, data = lung)
#' get_risk_table_frame(fit, times = c(0, 250, 500, 750, 1000))
#' risk_table(fit, times = c(0, 250, 500, 750, 1000))
#' @export
get_risk_table_frame <- function(fit, times = NULL) {
  check_fit_class(fit, c("kaplan_meier", "aalen_johansen"), "get_risk_table_frame()")
  curves <- risk_set_curves(fit)
  if (is.null(times)) {
    end <- max(vapply(curves, function(curve) {
      if (nrow(curve) > 0L) max(curve$time) else 0
    }, numeric(1)))
    times <- sort(unique(round(seq(0, end, length.out = 6L))))
  }
  times <- check_times(times)
  out <- do.call(rbind, lapply(names(curves), function(label) {
    curve <- curves[[label]]
    at <- findInterval(times, curve$time, left.open = TRUE) + 1L
    n_risk <- rep(0, length(times))
    inside <- at <= nrow(curve)
    n_risk[inside] <- curve$n_risk[at[inside]]
    data.frame(strata = label, time = times, n_risk = n_risk, stringsAsFactors = FALSE)
  }))
  rownames(out) <- NULL
  out
}

#' @rdname get_risk_table_frame
#' @export
risk_table <- function(fit, times = NULL) {
  frame <- get_risk_table_frame(fit, times)
  risk_table_plot(frame, !identical(unique(frame$strata), "all"), "Time")
}

# The curves holding each group's risk set, named by the group labels. The causes of a competing
# risks fit share one risk set, so the first cause's curve serves.
risk_set_curves <- function(fit) {
  labels <- curve_labels(fit)
  curves <- if (inherits(fit, "aalen_johansen")) {
    lapply(fit$curves, function(curve) curve[[1L]])
  } else {
    fit$curves
  }
  stats::setNames(curves, labels)
}

# The numbers-at-risk table as text, with the first group on top in the curves' first color. With
# `causes`, it's repeated under each cause's panel.
risk_table_plot <- function(frame, grouped, xlab, causes = NULL) {
  labels <- unique(frame$strata)
  frame$group <- factor(frame$strata, levels = labels)
  frame$strata <- factor(frame$strata, levels = rev(labels))
  frame$label <- sprintf("%g", frame$n_risk)
  if (!is.null(causes)) {
    frame <- do.call(rbind, lapply(causes, function(cause) cbind(frame, cause = cause)))
    frame$cause <- factor(frame$cause, levels = causes)
  }
  text <- if (grouped) {
    ggplot2::geom_text(ggplot2::aes(color = .data$group), size = 3.2, show.legend = FALSE)
  } else {
    ggplot2::geom_text(size = 3.2)
  }
  plot <- ggplot2::ggplot(
    frame, ggplot2::aes(x = .data$time, y = .data$strata, label = .data$label)
  ) +
    text +
    ggplot2::labs(x = xlab, y = NULL, title = "Number at risk") +
    ggplot2::theme_minimal() +
    ggplot2::theme(
      panel.grid = ggplot2::element_blank(),
      axis.ticks = ggplot2::element_blank(),
      plot.title = ggplot2::element_text(size = 10)
    )
  if (!is.null(causes)) {
    plot <- plot +
      ggplot2::facet_wrap(ggplot2::vars(.data$cause)) +
      ggplot2::theme(strip.text = ggplot2::element_blank())
  }
  plot
}

# The legend of a plot with groups, titled `title`, or nothing without groups. The bands' fill
# shares the title when the plot has bands (`fill = TRUE`), and only then, since ggplot2 reports
# labels it has no use for. Labels too long for one line across a plot about 7 inches wide wrap
# onto more rows, in order.
group_legend <- function(labels, grouped = TRUE, fill = FALSE, title = "Group") {
  if (!grouped) {
    return(NULL)
  }
  # Each key takes about as much room as five characters of label text.
  rows <- ceiling((nchar(title) + sum(nchar(labels) + 5L)) / LEGEND_LINE_CHARS)
  legend <- ggplot2::guide_legend(nrow = rows, byrow = TRUE)
  # ggplot2 merges the color and fill legends into one, so they're laid out the same way.
  if (fill) {
    list(ggplot2::labs(color = title, fill = title), ggplot2::guides(color = legend, fill = legend))
  } else {
    list(ggplot2::labs(color = title), ggplot2::guides(color = legend))
  }
}

# The characters of legend text that fit on one line across a plot about 7 inches wide.
LEGEND_LINE_CHARS <- 75

# The curve over its risk table, on a shared time range.
stack_risk_table <- function(plot, table, steps) {
  limits <- range(c(0, steps$time, table$data$time))
  plot <- plot + ggplot2::scale_x_continuous(limits = limits)
  table <- table + ggplot2::scale_x_continuous(limits = limits)
  height <- max(1, length(unique(table$data$strata)))
  patchwork::wrap_plots(plot, table, ncol = 1L, heights = c(4, 0.6 + 0.4 * height))
}

# The coordinates of a right-continuous step function through a curve's rows, from `start` at
# time 0, so that a line and a ribbon drawn through them step at each time.
step_coordinates <- function(curve, start) {
  n <- nrow(curve)
  if (n == 0L) {
    return(NULL)
  }
  previous <- function(values) c(start, values[-n])
  interleave <- function(before, after) c(start, as.vector(rbind(before, after)))
  out <- data.frame(
    time = c(0, rep(curve$time, each = 2L)),
    estimate = interleave(previous(curve$estimate), curve$estimate),
    conf_low = interleave(previous(curve$conf_low), curve$conf_low),
    conf_high = interleave(previous(curve$conf_high), curve$conf_high)
  )
  keys <- setdiff(names(curve), c("time", "estimate", "conf_low", "conf_high"))
  for (key in intersect(keys, c("strata", "cause"))) {
    out[[key]] <- curve[[key]][1L]
  }
  out
}

# -- cumulative incidence --------------------------------------------------------------------------

#' Plot cumulative incidence curves
#'
#' `plot_cif()` draws the cumulative incidence curves of an [aalen_johansen()] fit, one panel per
#' cause, as step functions with their pointwise confidence bands. A fit with groups has one curve
#' per group in each panel, told apart by color.
#'
#' With `risk_table = TRUE`, the numbers at risk (see [risk_table()]) are stacked under the panels.
#' This needs the patchwork package, and the result is a patchwork.
#'
#' @param fit An [aalen_johansen()] fit.
#' @param conf_int Draw the confidence bands.
#' @param risk_table Stack a numbers-at-risk table under the curves.
#' @param times The times of the numbers-at-risk table. The default is six evenly spaced, rounded
#'   times from 0 to the end of follow-up.
#' @param title An optional plot title.
#' @param xlab,ylab The axis labels.
#'
#' @return A ggplot object, or a patchwork with `risk_table = TRUE`.
#'
#' @examples
#' status <- with(mgus2, ifelse(pstat == 1, 1, 2 * death))
#' time <- with(mgus2, ifelse(pstat == 1, ptime, futime))
#' mgus <- data.frame(
#'   time = time,
#'   event = factor(status, 0:2, c("censor", "progression", "death")),
#'   sex = mgus2$sex
#' )
#' fit <- aalen_johansen(Surv(time, event) ~ sex, data = mgus)
#' plot_cif(fit)
#' @export
plot_cif <- function(fit,
                     conf_int = TRUE,
                     risk_table = FALSE,
                     times = NULL,
                     title = NULL,
                     xlab = "Time",
                     ylab = "Cumulative incidence") {
  check_fit_class(fit, "aalen_johansen", "plot_cif()")
  check_flag(conf_int, "conf_int")
  check_flag(risk_table, "risk_table")
  check_label(title, "title", null = TRUE)
  check_label(xlab, "xlab")
  check_label(ylab, "ylab")
  if (risk_table) {
    require_patchwork("plot_cif(risk_table = TRUE)")
  }
  table <- as.data.frame(fit)
  labels <- curve_labels(fit)
  grouped <- !identical(labels, "all")
  causes <- unique(table$cause)
  pieces <- split(table, list(factor(table$strata, labels), factor(table$cause, causes)))
  steps <- do.call(rbind, lapply(pieces, step_coordinates, start = 0))
  steps$strata <- factor(steps$strata, levels = labels)
  steps$cause <- factor(steps$cause, levels = causes)
  rownames(steps) <- NULL

  plot <- ggplot2::ggplot(steps, ggplot2::aes(x = .data$time, y = .data$estimate))
  if (conf_int) {
    ribbon <- steps[!is.na(steps$conf_low) & !is.na(steps$conf_high), , drop = FALSE]
    plot <- plot + if (grouped) {
      ggplot2::geom_ribbon(
        ggplot2::aes(ymin = .data$conf_low, ymax = .data$conf_high, fill = .data$strata),
        data = ribbon, alpha = 0.18, show.legend = FALSE
      )
    } else {
      ggplot2::geom_ribbon(
        ggplot2::aes(ymin = .data$conf_low, ymax = .data$conf_high),
        data = ribbon, alpha = 0.18, fill = PLOT_COLOR
      )
    }
  }
  line <- if (grouped) {
    ggplot2::geom_line(ggplot2::aes(color = .data$strata))
  } else {
    ggplot2::geom_line(color = PLOT_COLOR)
  }
  plot <- plot +
    line +
    ggplot2::facet_wrap(ggplot2::vars(.data$cause)) +
    ggplot2::scale_y_continuous(limits = c(0, 1)) +
    ggplot2::labs(x = xlab, y = ylab, title = title) +
    group_legend(labels, grouped, fill = conf_int) +
    theme_survival()
  if (!risk_table) {
    return(plot)
  }
  table <- risk_table_plot(get_risk_table_frame(fit, times), grouped, xlab, causes = causes)
  stack_risk_table(plot, table, steps)
}

# -- predicted curves ------------------------------------------------------------------------------

#' Plot predicted survival curves for new subjects
#'
#' `plot_predicted_survival()` draws one step curve per row of `newdata`, from any model whose
#' [predict()] method gives per-subject curves as a data frame with the columns `subject`, `time`,
#' and `estimate`, such as [cox_ph()], [survival_tree()], [random_survival_forest()],
#' [royston_parmar()], or [aft()]. It shows how the predicted survival differs between subjects,
#' where [plot_survival()] shows the curves of groups.
#'
#' @param model A fitted model with per-subject survival predictions.
#' @param newdata A data frame of the subjects to predict for.
#' @param type `"survival"` (the default) or `"cumhaz"`, passed on to [predict()].
#' @param times Optional times at which to predict, passed on to [predict()]. Models that predict
#'   at their own event times by default don't need them.
#' @param labels Optional labels for the subjects, one per row of `newdata`. The default is the
#'   row numbers.
#' @param xlab,ylab The axis labels. The default `ylab` depends on `type`.
#'
#' @return A ggplot object.
#'
#' @examples
#' fit <- cox_ph(Surv(time, status) ~ age + sex, data = lung)
#' subjects <- data.frame(age = c(50, 70, 70), sex = c(1, 1, 2))
#' plot_predicted_survival(fit, subjects, labels = c("50, male", "70, male", "70, female"))
#' plot_predicted_survival(fit, subjects, type = "cumhaz")
#' @export
plot_predicted_survival <- function(model,
                                    newdata,
                                    type = c("survival", "cumhaz"),
                                    times = NULL,
                                    labels = NULL,
                                    xlab = "Time",
                                    ylab = NULL) {
  type <- match.arg(type)
  if (!is.data.frame(newdata)) {
    stop("`newdata` must be a data frame.", call. = FALSE)
  }
  check_label(xlab, "xlab")
  check_label(ylab, "ylab", null = TRUE)
  if (is.null(ylab)) {
    ylab <- if (type == "survival") "Survival probability" else "Cumulative hazard"
  }
  if (is.null(labels)) {
    labels <- as.character(seq_len(nrow(newdata)))
  } else if (length(labels) != nrow(newdata) || anyNA(labels) || anyDuplicated(labels)) {
    stop(
      sprintf("`labels` must have %d distinct values, one per row of `newdata`.", nrow(newdata)),
      call. = FALSE
    )
  }
  curves <- if (is.null(times)) {
    stats::predict(model, newdata = newdata, type = type)
  } else {
    stats::predict(model, newdata = newdata, type = type, times = times)
  }
  if (!is.data.frame(curves) || !all(c("subject", "time", "estimate") %in% names(curves))) {
    stop(
      sprintf(
        "`predict()` of %s doesn't give per-subject curves with the columns `subject`, `time`, %s",
        obj_type_friendly(model), "and `estimate`."
      ),
      call. = FALSE
    )
  }
  start <- if (type == "survival") 1 else 0
  first <- min(curves$time)
  steps <- do.call(rbind, lapply(split(curves, curves$subject), function(curve) {
    data.frame(
      subject = curve$subject[1L],
      time = c(first, curve$time),
      estimate = c(start, curve$estimate)
    )
  }))
  steps$subject <- factor(labels[steps$subject], levels = labels)
  plot <- ggplot2::ggplot(
    steps, ggplot2::aes(x = .data$time, y = .data$estimate, color = .data$subject)
  ) +
    ggplot2::geom_step(direction = "hv") +
    ggplot2::labs(x = xlab, y = ylab) +
    group_legend(labels, title = "Subject") +
    theme_survival()
  if (type == "survival") {
    plot <- plot + ggplot2::scale_y_continuous(limits = c(0, 1))
  }
  plot
}

# -- forest plots ----------------------------------------------------------------------------------

# Labels such as `celltype: smallcell` for the terms that a model's factors expand to, named by the
# terms (`celltypesmallcell`). Models that don't record their factor levels get none.
factor_term_labels <- function(model) {
  xlevels <- if (is.list(model)) model$xlevels
  if (!is.list(xlevels) || length(xlevels) == 0L) {
    return(character())
  }
  variables <- rep(names(xlevels), lengths(xlevels))
  levels <- unlist(xlevels, use.names = FALSE)
  stats::setNames(paste0(variables, ": ", levels), paste0(variables, levels))
}

#' Forest plot of model estimates
#'
#' `plot_forest()` draws each term's estimate as a point with its confidence interval, and a
#' dashed line at no effect. It takes a fitted model, such as a [cox_ph()] fit, whose [tidy()]
#' table has the estimates, or a data frame of estimates from anywhere, such as differences in
#' restricted mean survival.
#'
#' For a model, `exponentiate = TRUE` (the default) plots hazard ratios (or the model's other ratio
#' scale) on a log axis with the line at 1. For a data frame, the estimates are plotted as they
#' are, on a linear axis with the line at 0, unless `scale = "log"`.
#'
#' @param result A fitted model, or a data frame with the columns `term`, `estimate`, and the
#'   limits `conf_low` and `conf_high` (or `ci_lower` and `ci_upper`).
#' @param scale The axis scale, `"log"` or `"linear"`. The default is `"log"` for an
#'   exponentiated model and `"linear"` otherwise.
#' @param exponentiate For a model, plot the exponentiated estimates.
#' @param title An optional plot title.
#' @param term_labels Optional display labels for the terms, as a named character vector such as
#'   `c(age = "Age (years)")`. A model's factor terms are labelled with the variable and level, such
#'   as `celltype: smallcell` for the term `celltypesmallcell`, and other terms keep their names.
#' @param xlab The x axis label. The default is `"Hazard Ratio"` on the log scale and `"Estimate"`
#'   otherwise.
#'
#' @return A ggplot object. The first term is at the top.
#'
#' @seealso [theme_forest()].
#'
#' @examples
#' fit <- cox_ph(Surv(time, status) ~ age + sex + ph.ecog, data = lung)
#' plot_forest(fit, term_labels = c(age = "Age", sex = "Female", ph.ecog = "ECOG score"))
#'
#' rmst <- data.frame(
#'   term = c("Drug A vs placebo", "Drug B vs placebo"),
#'   estimate = c(45, 25),
#'   conf_low = c(-10, -20),
#'   conf_high = c(95, 70)
#' )
#' plot_forest(rmst, xlab = "Difference in RMST (days)")
#' @export
plot_forest <- function(result,
                        scale = NULL,
                        exponentiate = TRUE,
                        title = NULL,
                        term_labels = NULL,
                        xlab = NULL) {
  check_flag(exponentiate, "exponentiate")
  check_label(title, "title", null = TRUE)
  check_label(xlab, "xlab", null = TRUE)
  if (!is.null(scale)) {
    scale <- match.arg(scale, c("log", "linear"))
  }
  if (!is.null(term_labels) && (!is.character(term_labels) || is.null(names(term_labels)))) {
    stop("`term_labels` must be a named character vector.", call. = FALSE)
  }
  is_model <- !is.data.frame(result)
  table <- if (is_model) {
    tryCatch(
      tidy(result, exponentiate = exponentiate),
      error = function(e) {
        stop(
          sprintf(
            "`result` must be a fitted model with a `tidy()` method, or a data frame, not %s.",
            obj_type_friendly(result)
          ),
          call. = FALSE
        )
      }
    )
  } else {
    result
  }
  table <- as.data.frame(table)
  names(table)[names(table) == "ci_lower"] <- "conf_low"
  names(table)[names(table) == "ci_upper"] <- "conf_high"
  missing <- setdiff(c("term", "estimate", "conf_low", "conf_high"), names(table))
  if (length(missing) > 0L) {
    stop(
      sprintf(
        "%s has no %s column%s. It needs `term`, `estimate`, `conf_low`, and `conf_high`.",
        if (is_model) "The model's `tidy()` table" else "`result`",
        paste0("`", missing, "`", collapse = ", "), if (length(missing) > 1L) "s" else ""
      ),
      call. = FALSE
    )
  }
  table$term <- as.character(table$term)
  if (is_model) {
    # The given labels take the place of the default ones for the same terms.
    term_labels <- c(factor_term_labels(result), term_labels)
    term_labels <- term_labels[!duplicated(names(term_labels), fromLast = TRUE)]
  }
  if (length(term_labels) > 0L) {
    labelled <- table$term %in% names(term_labels)
    table$term[labelled] <- term_labels[table$term[labelled]]
  }
  table$term <- factor(table$term, levels = rev(unique(table$term)))
  use_log <- if (is.null(scale)) is_model && exponentiate else scale == "log"
  if (is.null(xlab)) {
    xlab <- if (use_log) "Hazard Ratio" else "Estimate"
  }

  plot <- ggplot2::ggplot(table, ggplot2::aes(x = .data$estimate, y = .data$term)) +
    ggplot2::geom_errorbar(
      ggplot2::aes(xmin = .data$conf_low, xmax = .data$conf_high),
      width = 0.25, color = "#555555", orientation = "y"
    ) +
    ggplot2::geom_point(size = 3, color = PLOT_COLOR) +
    ggplot2::geom_vline(
      xintercept = if (use_log) 1 else 0, linetype = "dashed", color = REFERENCE_COLOR,
      alpha = 0.8
    ) +
    ggplot2::labs(x = xlab, y = NULL, title = title) +
    theme_forest()
  if (use_log) {
    plot <- plot + ggplot2::scale_x_log10()
  }
  plot
}

# -- Cox diagnostics -------------------------------------------------------------------------------

INFLUENCE_LABELS <- c(
  deviance = "Deviance residual",
  leverage = "Leverage",
  ld = "Likelihood displacement",
  martingale = "Martingale residual"
)

#' Plot the influence diagnostics of a Cox model
#'
#' `plot_influence()` plots measures from [influence_diagnostics()] against the linear predictor
#' of a [cox_ph()] fit, one panel per measure. The `highlight` observations with the largest
#' likelihood displacement are drawn in red and labelled with their row numbers.
#'
#' @param fit A [cox_ph()] fit.
#' @param highlight The number of most influential observations to highlight.
#' @param panels The measures to plot, any of `"deviance"`, `"leverage"`, `"ld"` (the likelihood
#'   displacement), and `"martingale"`.
#' @param title An optional plot title.
#'
#' @return A ggplot object.
#'
#' @examples
#' fit <- cox_ph(Surv(time, status) ~ age + sex, data = lung)
#' plot_influence(fit)
#' plot_influence(fit, highlight = 5, panels = c("deviance", "leverage"))
#' @export
plot_influence <- function(fit,
                           highlight = 3,
                           panels = c("deviance", "leverage", "ld"),
                           title = NULL) {
  check_fit_class(fit, "cox_ph", "plot_influence()")
  check_number(highlight, "highlight", min = 0)
  check_label(title, "title", null = TRUE)
  if (!is.character(panels) || length(panels) == 0L || !all(panels %in% names(INFLUENCE_LABELS))) {
    stop(
      sprintf(
        "`panels` must hold one or more of %s.",
        paste0("\"", names(INFLUENCE_LABELS), "\"", collapse = ", ")
      ),
      call. = FALSE
    )
  }
  panels <- unique(panels)
  diagnostics <- influence_diagnostics(fit)
  n <- nrow(diagnostics)
  ranks <- rank(-diagnostics$ld, ties.method = "first")
  influential <- ranks <= min(highlight, n)
  long <- do.call(rbind, lapply(panels, function(panel) {
    data.frame(
      lp = fit$linear_predictors,
      value = diagnostics[[panel]],
      panel = INFLUENCE_LABELS[[panel]],
      obs = seq_len(n),
      influential = influential,
      stringsAsFactors = FALSE
    )
  }))
  long$panel <- factor(long$panel, levels = INFLUENCE_LABELS[panels])
  plot <- ggplot2::ggplot(long, ggplot2::aes(x = .data$lp, y = .data$value)) +
    ggplot2::geom_point(ggplot2::aes(color = .data$influential), size = 1.5, alpha = 0.5) +
    ggplot2::scale_color_manual(values = c("TRUE" = HIGHLIGHT_COLOR, "FALSE" = PLOT_COLOR)) +
    ggplot2::facet_wrap(ggplot2::vars(.data$panel), scales = "free_y", nrow = 1L) +
    ggplot2::labs(x = "Linear predictor", y = NULL, title = title) +
    ggplot2::theme_minimal() +
    ggplot2::theme(
      legend.position = "none",
      strip.text = ggplot2::element_text(size = 10, face = "bold")
    )
  highlighted <- long[long$influential, , drop = FALSE]
  if (nrow(highlighted) > 0L) {
    plot <- plot + ggplot2::geom_text(
      ggplot2::aes(label = .data$obs),
      data = highlighted, size = 2.5, hjust = -0.3, color = HIGHLIGHT_COLOR
    )
  }
  plot
}

SCHOENFELD_XLABS <- c(
  identity = "Time",
  log = "log(Time)",
  km = "KM failure probability",
  rank = "Rank(Time)"
)

#' Plot the scaled Schoenfeld residuals of a Cox model
#'
#' `plot_schoenfeld()` checks the proportional hazards assumption of a [cox_ph()] fit by plotting
#' each coefficient's scaled Schoenfeld residuals against a function of time, one panel per
#' coefficient. Under proportional hazards the residuals scatter around the coefficient (the dashed
#' line), and the loess smooth through them is flat. A smooth that trends shows the effect
#' changing over time. With `show_zph = TRUE`, each panel also shows the p-value of its term's
#' [cox_zph()] test.
#'
#' @param fit A [cox_ph()] fit.
#' @param transform The function of time on the x axis, as in [cox_zph()]: `"identity"` (the
#'   default), `"km"`, `"rank"`, or `"log"`.
#' @param show_zph Show the [cox_zph()] p-value of each term.
#' @param title An optional plot title.
#'
#' @return A ggplot object.
#'
#' @examples
#' fit <- cox_ph(Surv(time, status) ~ age + sex, data = lung)
#' plot_schoenfeld(fit)
#' plot_schoenfeld(fit, transform = "km", show_zph = FALSE)
#' @export
plot_schoenfeld <- function(fit,
                            transform = c("identity", "km", "rank", "log"),
                            show_zph = TRUE,
                            title = NULL) {
  check_fit_class(fit, "cox_ph", "plot_schoenfeld()")
  transform <- match.arg(transform)
  check_flag(show_zph, "show_zph")
  check_label(title, "title", null = TRUE)
  zph <- cox_zph(fit, transform = transform)
  coefs <- names(fit$coefficients)
  long <- data.frame(
    time = rep(zph$x, times = length(coefs)),
    residual = as.vector(zph$y),
    term = factor(rep(coefs, each = length(zph$x)), levels = coefs)
  )
  reference <- data.frame(
    term = factor(coefs, levels = coefs),
    yintercept = unname(fit$coefficients)
  )
  plot <- ggplot2::ggplot(long, ggplot2::aes(x = .data$time, y = .data$residual)) +
    ggplot2::geom_point(size = 1, alpha = 0.35, color = PLOT_COLOR) +
    ggplot2::geom_smooth(
      method = "loess", formula = y ~ x, span = 0.6, se = FALSE, color = HIGHLIGHT_COLOR
    ) +
    ggplot2::geom_hline(
      ggplot2::aes(yintercept = .data$yintercept),
      data = reference, linetype = "dashed", color = REFERENCE_COLOR
    ) +
    ggplot2::facet_wrap(ggplot2::vars(.data$term), scales = "free_y", nrow = 1L) +
    ggplot2::labs(
      x = SCHOENFELD_XLABS[[transform]], y = "Scaled Schoenfeld residual", title = title
    ) +
    ggplot2::theme_minimal() +
    ggplot2::theme(strip.text = ggplot2::element_text(size = 10, face = "bold"))
  if (show_zph) {
    p_value <- zph$table$p_value[fit$assign]
    labels <- data.frame(
      term = factor(coefs, levels = coefs),
      label = ifelse(
        p_value >= 0.001, sprintf("p = %.3f", p_value), sprintf("p = %.1e", p_value)
      ),
      time = max(zph$x),
      residual = apply(zph$y, 2L, max)
    )
    plot <- plot + ggplot2::geom_text(
      ggplot2::aes(label = .data$label),
      data = labels, hjust = 1, vjust = 1, size = 3, fontface = "italic", color = REFERENCE_COLOR
    )
  }
  plot
}

#' Plot a smooth hazard ratio curve
#'
#' `plot_smooth_hr()` draws the curve of a [smooth_hr()] result: the (log) hazard ratio across the
#' range of a continuous covariate, with its pointwise confidence band, and a dashed line at no
#' effect.
#'
#' @param result A [smooth_hr()] result.
#' @param scale `"log_hr"` (the default) for the log hazard ratio, or `"hr"` for the hazard ratio.
#' @param title An optional plot title.
#' @param xlab,ylab The axis labels. The default `xlab` is the covariate's name, and the default
#'   `ylab` depends on `scale`.
#'
#' @return A ggplot object.
#'
#' @examples
#' fit <- cox_ph(Surv(time, status) ~ age + sex, data = lung)
#' plot_smooth_hr(smooth_hr(fit, term = "age"))
#' plot_smooth_hr(smooth_hr(fit, term = "age", df = 5), scale = "hr")
#' @export
plot_smooth_hr <- function(result, scale = c("log_hr", "hr"), title = NULL, xlab = NULL,
                           ylab = NULL) {
  check_fit_class(result, "smooth_hr", "plot_smooth_hr()", what = "`smooth_hr()` result")
  scale <- match.arg(scale)
  check_label(title, "title", null = TRUE)
  check_label(xlab, "xlab", null = TRUE)
  check_label(ylab, "ylab", null = TRUE)
  table <- as.data.frame(result, scale = scale)
  names(table) <- c("x", "y", "conf_low", "conf_high")
  if (is.null(xlab)) {
    xlab <- result$term
  }
  if (is.null(ylab)) {
    ylab <- if (scale == "hr") "Hazard ratio" else "Log hazard ratio"
  }
  ggplot2::ggplot(table, ggplot2::aes(x = .data$x, y = .data$y)) +
    ggplot2::geom_hline(
      yintercept = if (scale == "hr") 1 else 0, linetype = "dashed", color = REFERENCE_COLOR,
      linewidth = 0.5
    ) +
    ggplot2::geom_ribbon(
      ggplot2::aes(ymin = .data$conf_low, ymax = .data$conf_high),
      alpha = 0.18, fill = PLOT_COLOR
    ) +
    ggplot2::geom_line(color = PLOT_COLOR, linewidth = 1.2) +
    ggplot2::labs(x = xlab, y = ylab, title = title) +
    ggplot2::theme_minimal()
}

# -- parametric diagnostics ------------------------------------------------------------------------

WEIBULL_YLABS <- c(
  weibull = "log(-log S(t))",
  lognormal = "Normal quantile",
  loglogistic = "log((1 - S(t)) / S(t))"
)

# The transform of the survival probabilities that makes each distribution a straight line in
# log time.
linearize_survival <- function(s, dist) {
  switch(dist,
    weibull = log(-log(s)),
    lognormal = stats::qnorm(1 - s),
    loglogistic = log((1 - s) / s)
  )
}

#' Probability plot for a parametric survival distribution
#'
#' `plot_weibull()` plots the Kaplan-Meier estimate on the scale where a distribution's survival
#' curve is a straight line in log time: \eqn{\log(-\log S(t))} for the Weibull,
#' \eqn{\Phi^{-1}(1 - S(t))} for the log-normal, and \eqn{\log((1 - S(t)) / S(t))} for the
#' log-logistic. Points close to a straight line support the distribution. For a fit with groups,
#' parallel lines support proportional hazards (Weibull) or proportional odds (log-logistic) as
#' well.
#'
#' An [aft()] fit of the same distribution (or an exponential one, for the Weibull) adds its
#' fitted line, dashed, at the mean linear predictor. The overlay needs a fit without groups.
#'
#' @param fit A [kaplan_meier()] fit.
#' @param aft An optional [aft()] fit to overlay.
#' @param dist The distribution: `"weibull"` (the default), `"lognormal"`, or `"loglogistic"`.
#' @param title An optional plot title.
#' @param xlab,ylab The axis labels. The default `ylab` depends on `dist`.
#'
#' @return A ggplot object.
#'
#' @examples
#' fit <- kaplan_meier(Surv(time, status) ~ sex, data = lung)
#' plot_weibull(fit)
#'
#' pooled <- kaplan_meier(Surv(time, status) ~ 1, data = lung)
#' weibull <- aft(Surv(time, status) ~ 1, data = lung, dist = "weibull")
#' plot_weibull(pooled, aft = weibull)
#' @export
plot_weibull <- function(fit,
                         aft = NULL,
                         dist = c("weibull", "lognormal", "loglogistic"),
                         title = NULL,
                         xlab = "log(time)",
                         ylab = NULL) {
  check_fit_class(fit, "kaplan_meier", "plot_weibull()")
  dist <- match.arg(dist)
  check_label(title, "title", null = TRUE)
  check_label(xlab, "xlab")
  check_label(ylab, "ylab", null = TRUE)
  if (is.null(ylab)) {
    ylab <- WEIBULL_YLABS[[dist]]
  }
  labels <- curve_labels(fit)
  grouped <- !identical(labels, "all")
  if (!is.null(aft)) {
    check_fit_class(aft, "aft", "plot_weibull()", arg = "aft")
    if (grouped) {
      stop(
        "The `aft` overlay needs a `kaplan_meier()` fit without groups, such as `~ 1`.",
        call. = FALSE
      )
    }
    allowed <- if (dist == "weibull") c("weibull", "exponential") else dist
    if (!aft$dist %in% allowed) {
      stop(
        sprintf(
          "`aft` has a %s distribution, which doesn't match `dist = \"%s\"`.", aft$dist, dist
        ),
        call. = FALSE
      )
    }
  }
  table <- as.data.frame(fit)
  table <- table[table$estimate > 0 & table$estimate < 1, , drop = FALSE]
  points <- data.frame(
    log_time = log(table$time),
    y = linearize_survival(table$estimate, dist),
    strata = factor(table$strata, levels = labels)
  )
  plot <- ggplot2::ggplot(points, ggplot2::aes(x = .data$log_time, y = .data$y))
  plot <- plot + if (grouped) {
    list(
      ggplot2::geom_line(ggplot2::aes(color = .data$strata), alpha = 0.5),
      ggplot2::geom_point(ggplot2::aes(color = .data$strata), size = 1.5, alpha = 0.8)
    )
  } else {
    list(
      ggplot2::geom_line(color = PLOT_COLOR, alpha = 0.5),
      ggplot2::geom_point(color = PLOT_COLOR, size = 1.5, alpha = 0.8)
    )
  }
  if (!is.null(aft) && nrow(points) > 0L) {
    plot <- plot + ggplot2::geom_line(
      data = aft_probability_line(aft, range(points$log_time), dist),
      linetype = "dashed", color = "#333333", linewidth = 0.8
    )
  }
  plot +
    ggplot2::labs(x = xlab, y = ylab, title = title) +
    group_legend(labels, grouped) +
    theme_survival()
}

# The fitted AFT line on the probability plot's scale, at the mean linear predictor, over the
# points' log-time range padded by 5% on each side.
aft_probability_line <- function(aft, range, dist) {
  pad <- 0.05 * diff(range)
  log_time <- seq(range[1L] - pad, range[2L] + pad, length.out = 200L)
  shifted <- exp(log_time) - aft$threshold
  live <- shifted > 0
  z <- (log(shifted[live]) - mean(aft$linear_predictors)) / aft$scale
  s <- aft_family(aft$dist)$survival(z)
  keep <- s > 0 & s < 1
  data.frame(log_time = log_time[live][keep], y = linearize_survival(s[keep], dist))
}

# -- helpers ---------------------------------------------------------------------------------------

# The curve labels of a curve fit, as in its `strata` column ("all" for a single curve).
curve_labels <- function(fit) {
  if (is.null(fit$groups)) "all" else as.character(fit$groups$strata)
}

check_fit_class <- function(x, classes, fn, arg = "fit", what = NULL) {
  if (!inherits(x, classes)) {
    if (is.null(what)) {
      what <- paste(paste0("`", classes, "()`"), collapse = " or ")
      what <- paste(what, "fit")
    }
    stop(
      sprintf("`%s` of `%s` must be a %s, not %s.", arg, fn, what, obj_type_friendly(x)),
      call. = FALSE
    )
  }
  invisible(x)
}

check_label <- function(x, name, null = FALSE) {
  if (null && is.null(x)) {
    return(invisible(x))
  }
  if (!is.character(x) || length(x) != 1L || is.na(x)) {
    stop(
      sprintf("`%s` must be a single string%s.", name, if (null) " or `NULL`" else ""),
      call. = FALSE
    )
  }
  invisible(x)
}

require_patchwork <- function(what) {
  if (!requireNamespace("patchwork", quietly = TRUE)) {
    stop(
      sprintf("`%s` needs the patchwork package: install.packages(\"patchwork\").", what),
      call. = FALSE
    )
  }
}
