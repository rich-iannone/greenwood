# The ggplot2 plots: their layers, the data they draw, and their argument checks.

km_sex <- function() kaplan_meier(Surv(time, status) ~ sex, data = lung)
cox_lung <- function() cox_ph(Surv(time, status) ~ age + sex, data = lung)
mgus_cr <- function(formula = Surv(time, event) ~ sex) {
  status <- with(mgus2, ifelse(pstat == 1, 1, 2 * death))
  mgus <- data.frame(
    time = with(mgus2, ifelse(pstat == 1, ptime, futime)),
    event = factor(status, 0:2, c("censor", "progression", "death")),
    sex = mgus2$sex
  )
  aalen_johansen(formula, data = mgus)
}

layer_geoms <- function(plot) {
  vapply(plot$layers, function(layer) class(layer$geom)[1L], character(1), USE.NAMES = FALSE)
}

test_that("plot_survival() draws steps, bands, and censoring marks", {
  fit <- km_sex()
  plot <- plot_survival(fit)
  expect_s3_class(plot, "ggplot")
  expect_identical(layer_geoms(plot), c("GeomRibbon", "GeomLine", "GeomPoint"))
  steps <- plot$data
  expect_identical(levels(steps$strata), c("sex=1", "sex=2"))
  # Each curve starts at 1 at time 0 and steps down at each of its times.
  first <- steps[steps$strata == "sex=1", ]
  curve <- fit$curves[[1]]
  expect_identical(nrow(first), 2L * nrow(curve) + 1L)
  expect_equal(first$time[1:3], c(0, curve$time[1], curve$time[1]))
  expect_equal(first$estimate[1:3], c(1, 1, curve$estimate[1]))
  # The censoring marks sit on the curve at the censoring times.
  marks <- ggplot2::layer_data(plot, 3L)
  censored <- as.data.frame(fit)
  censored <- censored[censored$n_censor > 0, ]
  expect_equal(marks$x, censored$time)
  expect_equal(marks$y, censored$estimate)
  expect_identical(plot$labels$colour, "Group")
  expect_identical(plot$labels$x, "Time")

  bare <- plot_survival(fit, conf_int = FALSE, censor_marks = FALSE, ylab = "S(t)")
  expect_identical(layer_geoms(bare), "GeomLine")
  expect_identical(bare$labels$y, "S(t)")

  single <- plot_survival(kaplan_meier(Surv(time, status) ~ 1, data = lung))
  expect_identical(as.character(unique(single$data$strata)), "all")
  expect_null(single$labels$colour)
  expect_identical(unique(ggplot2::layer_data(single, 2L)$colour), "#20558A")
  expect_silent(ggplot2::ggplot_build(single))
})

test_that("plot_survival() stacks the numbers at risk under the curves", {
  skip_if_not_installed("patchwork")
  plot <- plot_survival(km_sex(), risk_table = TRUE, times = c(0, 500, 1000))
  expect_s3_class(plot, "patchwork")
  table <- plot[[2]]
  expect_identical(table$data$label, c("138", "20", "2", "90", "21", "0"))
  # The table and the curves share the time range.
  limits <- function(p) ggplot2::ggplot_build(p)$layout$panel_scales_x[[1]]$limits
  expect_equal(limits(plot[[1]]), limits(table))
})

test_that("get_risk_table_frame() counts the subjects at risk", {
  fit <- km_sex()
  frame <- get_risk_table_frame(fit, times = c(0, 11, 12, 500, 2000))
  expect_named(frame, c("strata", "time", "n_risk"))
  expect_identical(frame$strata, rep(c("sex=1", "sex=2"), each = 5))
  # At an event time, the count is the risk set of that time.
  expect_equal(frame$n_risk[1:5], c(138, 138, 135, 20, 0))
  # By default, six rounded times from 0 to the end of follow-up.
  frame <- get_risk_table_frame(fit)
  expect_equal(unique(frame$time), round(seq(0, 1022, length.out = 6)))
  expect_equal(frame$n_risk[1:6], c(138, 77, 30, 13, 4, 1))

  single <- get_risk_table_frame(kaplan_meier(Surv(time, status) ~ 1, data = lung), times = 0)
  expect_identical(single$strata, "all")
  expect_equal(single$n_risk, 228)

  cr <- get_risk_table_frame(mgus_cr(), times = c(0, 100))
  expect_identical(cr$strata, c("sex=F", "sex=F", "sex=M", "sex=M"))
  expect_equal(cr$n_risk[c(1, 3)], c(631, 753))
  expect_error(get_risk_table_frame(cox_lung()), "kaplan_meier")
})

test_that("risk_table() draws the counts as text", {
  plot <- risk_table(km_sex(), times = c(0, 500))
  expect_s3_class(plot, "ggplot")
  expect_identical(layer_geoms(plot), "GeomText")
  expect_identical(plot$labels$title, "Number at risk")
  # The first group is on top.
  expect_identical(levels(plot$data$strata), c("sex=2", "sex=1"))
})

test_that("plot_cif() draws one panel per cause", {
  fit <- mgus_cr()
  plot <- plot_cif(fit, title = "MGUS")
  expect_s3_class(plot, "ggplot")
  expect_identical(layer_geoms(plot), c("GeomRibbon", "GeomLine"))
  expect_identical(levels(plot$data$cause), c("progression", "death"))
  expect_identical(plot$labels$title, "MGUS")
  built <- ggplot2::ggplot_build(plot)
  expect_identical(nrow(built$layout$layout), 2L)
  # Each curve starts at 0 at time 0.
  starts <- plot$data[plot$data$time == 0, ]
  expect_true(all(starts$estimate == 0))

  single <- plot_cif(mgus_cr(Surv(time, event) ~ 1), conf_int = FALSE)
  expect_identical(layer_geoms(single), "GeomLine")

  skip_if_not_installed("patchwork")
  stacked <- plot_cif(fit, risk_table = TRUE, times = c(0, 100, 200))
  expect_s3_class(stacked, "patchwork")
  expect_identical(levels(stacked[[2]]$data$cause), c("progression", "death"))
})

test_that("plot_predicted_survival() draws a curve per subject", {
  fit <- cox_lung()
  subjects <- data.frame(age = c(50, 70), sex = c(1, 2))
  plot <- plot_predicted_survival(fit, subjects, labels = c("young", "old"))
  expect_s3_class(plot, "ggplot")
  expect_identical(layer_geoms(plot), "GeomStep")
  expect_identical(levels(plot$data$subject), c("young", "old"))
  predicted <- predict(fit, newdata = subjects, type = "survival")
  expect_identical(nrow(plot$data), nrow(predicted) + 2L)
  expect_identical(plot$labels$y, "Survival probability")

  cumhaz <- plot_predicted_survival(fit, subjects, type = "cumhaz", times = c(100, 200, 300))
  expect_identical(cumhaz$labels$y, "Cumulative hazard")
  expect_equal(cumhaz$data$estimate[1], 0)
  expect_identical(nrow(cumhaz$data), 8L)

  l <- stats::na.omit(lung[c("time", "status", "age", "sex")])
  tree <- survival_tree(Surv(time, status) ~ age + sex, data = l)
  expect_s3_class(plot_predicted_survival(tree, l[1:3, ]), "ggplot")

  expect_error(plot_predicted_survival(fit, subjects, labels = "a"), "labels")
  expect_error(plot_predicted_survival(fit, as.list(subjects)), "data frame")
  expect_error(
    plot_predicted_survival(km_sex(), subjects, times = 100),
    "per-subject curves|unused argument|newdata"
  )
})

test_that("plot_forest() plots hazard ratios or a table of estimates", {
  fit <- cox_ph(Surv(time, status) ~ age + sex + ph.ecog, data = lung)
  plot <- plot_forest(fit, term_labels = c(age = "Age"), title = "Lung")
  expect_s3_class(plot, "ggplot")
  expect_identical(layer_geoms(plot), c("GeomErrorbar", "GeomPoint", "GeomVline"))
  expect_identical(levels(plot$data$term), c("ph.ecog", "sex", "Age"))
  expect_equal(plot$data$estimate, unname(exp(coef(fit))))
  expect_identical(plot$labels$x, "Hazard Ratio")
  expect_identical(ggplot2::layer_data(plot, 3L)$xintercept, 0)
  expect_s3_class(plot$scales$get_scales("x")$trans, "transform")

  linear <- plot_forest(fit, exponentiate = FALSE)
  expect_equal(linear$data$estimate, unname(coef(fit)))
  expect_identical(linear$labels$x, "Estimate")
  expect_identical(ggplot2::layer_data(linear, 3L)$xintercept, 0)

  rmst <- data.frame(
    term = c("A", "B"), estimate = c(45, 25), ci_lower = c(-10, -20), ci_upper = c(95, 70)
  )
  plot <- plot_forest(rmst, xlab = "Difference")
  expect_identical(plot$labels$x, "Difference")
  expect_identical(levels(plot$data$term), c("B", "A"))
  expect_equal(plot$data$conf_low, c(-10, -20))
  expect_error(plot_forest(rmst[c("term", "estimate")]), "conf_low")
  expect_error(plot_forest(list(1)), "tidy")
  expect_error(plot_forest(fit, scale = "sqrt"), "arg")
  expect_error(plot_forest(fit, term_labels = "Age"), "named")
})

test_that("plot_influence() highlights the most influential observations", {
  fit <- cox_lung()
  plot <- plot_influence(fit)
  expect_s3_class(plot, "ggplot")
  expect_identical(layer_geoms(plot), c("GeomPoint", "GeomText"))
  expect_identical(
    levels(plot$data$panel), c("Deviance residual", "Leverage", "Likelihood displacement")
  )
  labelled <- plot$layers[[2]]$data
  expect_setequal(labelled$obs, c(6L, 85L, 38L))
  expect_equal(plot$data$lp[1:228], unname(fit$linear_predictors))

  plot <- plot_influence(fit, highlight = 0, panels = "martingale")
  expect_identical(layer_geoms(plot), "GeomPoint")
  expect_equal(plot$data$value, residuals(fit, type = "martingale"))
  expect_error(plot_influence(fit, panels = "dfbeta"), "panels")
  expect_error(plot_influence(km_sex()), "cox_ph")
})

test_that("plot_schoenfeld() plots the scaled residuals with the zph p-values", {
  fit <- cox_lung()
  plot <- plot_schoenfeld(fit)
  expect_s3_class(plot, "ggplot")
  expect_identical(layer_geoms(plot), c("GeomPoint", "GeomSmooth", "GeomHline", "GeomText"))
  zph <- cox_zph(fit, transform = "identity")
  expect_equal(plot$data$residual, as.vector(zph$y))
  expect_equal(plot$layers[[3]]$data$yintercept, unname(coef(fit)))
  labels <- plot$layers[[4]]$data$label
  expect_identical(labels, sprintf("p = %.3f", zph$table$p_value[1:2]))
  expect_identical(plot$labels$x, "Time")

  plot <- plot_schoenfeld(fit, transform = "km", show_zph = FALSE)
  expect_identical(plot$labels$x, "KM failure probability")
  expect_identical(length(plot$layers), 3L)
  expect_silent(ggplot2::ggplot_build(plot))
})

test_that("plot_schoenfeld() gives each coefficient its term's p-value", {
  fit <- cox_ph(Surv(time, status) ~ age + factor(ph.ecog), data = lung)
  plot <- plot_schoenfeld(fit)
  expect_identical(levels(plot$data$term), names(coef(fit)))
  p_value <- cox_zph(fit, transform = "identity")$table$p_value
  expect_identical(
    plot$layers[[4]]$data$label,
    ifelse(p_value[fit$assign] >= 0.001, sprintf("p = %.3f", p_value[fit$assign]),
      sprintf("p = %.1e", p_value[fit$assign])
    )
  )
})

test_that("plot_smooth_hr() draws the curve and its band", {
  curve <- smooth_hr(cox_lung(), term = "age", n_grid = 50)
  plot <- plot_smooth_hr(curve)
  expect_s3_class(plot, "ggplot")
  expect_identical(layer_geoms(plot), c("GeomHline", "GeomRibbon", "GeomLine"))
  expect_equal(plot$data$y, curve$log_hr)
  expect_identical(plot$labels$x, "age")
  expect_identical(plot$labels$y, "Log hazard ratio")
  expect_identical(ggplot2::layer_data(plot, 1L)$yintercept, 0)

  plot <- plot_smooth_hr(curve, scale = "hr", xlab = "Age (years)")
  expect_equal(plot$data$conf_high, curve$hr_upper)
  expect_identical(plot$labels$y, "Hazard ratio")
  expect_identical(ggplot2::layer_data(plot, 1L)$yintercept, 1)
  expect_error(plot_smooth_hr(cox_lung()), "smooth_hr")
})

test_that("plot_weibull() linearizes the Kaplan-Meier estimate", {
  fit <- km_sex()
  plot <- plot_weibull(fit)
  expect_s3_class(plot, "ggplot")
  expect_identical(layer_geoms(plot), c("GeomLine", "GeomPoint"))
  table <- as.data.frame(fit)
  table <- table[table$estimate > 0 & table$estimate < 1, ]
  expect_equal(plot$data$log_time, log(table$time))
  expect_equal(plot$data$y, log(-log(table$estimate)))
  expect_identical(plot$labels$y, "log(-log S(t))")

  plot <- plot_weibull(fit, dist = "lognormal")
  expect_equal(plot$data$y, stats::qnorm(1 - table$estimate))
  plot <- plot_weibull(fit, dist = "loglogistic")
  expect_equal(plot$data$y, log((1 - table$estimate) / table$estimate))

  pooled <- kaplan_meier(Surv(time, status) ~ 1, data = lung)
  weibull <- aft(Surv(time, status) ~ 1, data = lung, dist = "weibull")
  plot <- plot_weibull(pooled, aft = weibull)
  expect_identical(layer_geoms(plot), c("GeomLine", "GeomPoint", "GeomLine"))
  line <- plot$layers[[3]]$data
  # The Weibull line is straight on this scale, with slope 1 / scale.
  slopes <- diff(line$y) / diff(line$log_time)
  expect_equal(slopes, rep(1 / weibull$scale, length(slopes)), tolerance = 1e-8)
  expect_lt(min(line$log_time), min(plot$data$log_time))

  exponential <- aft(Surv(time, status) ~ 1, data = lung, dist = "exponential")
  expect_s3_class(plot_weibull(pooled, aft = exponential), "ggplot")
  expect_error(plot_weibull(fit, aft = weibull), "without groups")
  expect_error(plot_weibull(pooled, aft = weibull, dist = "lognormal"), "doesn't match")
  expect_error(plot_weibull(pooled, aft = cox_lung()), "aft")
})

test_that("the themes are ggplot2 themes", {
  expect_s3_class(theme_survival(), "theme")
  expect_identical(theme_survival()$legend.position, "top")
  expect_s3_class(theme_forest(), "theme")
  expect_identical(theme_forest()$legend.position, "none")
})

test_that("the plots check their arguments", {
  fit <- km_sex()
  expect_error(plot_survival(cox_lung()), "kaplan_meier")
  expect_error(plot_survival(fit, conf_int = NA), "conf_int")
  expect_error(plot_survival(fit, xlab = 1), "xlab")
  expect_error(plot_cif(fit), "aalen_johansen")
  expect_error(plot_cif(mgus_cr(), title = c("a", "b")), "title")
  expect_error(plot_schoenfeld(cox_lung(), transform = "sqrt"), "arg")
})
