#!/usr/bin/env Rscript
# Generate R-parity fixtures for the Aalen additive hazards model.
#
# Validates against R's `survival::aareg`.
#
# Run from the repo root:
#   Rscript scripts/generate_aalen_fixture.R

suppressPackageStartupMessages({
  library(survival)
  if (!requireNamespace("jsonlite", quietly = TRUE)) {
    stop("jsonlite is required: install.packages('jsonlite')")
  }
})

out_dir <- file.path("tests", "fixtures", "r")
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

write_json_fixture <- function(obj, name) {
  path <- file.path(out_dir, paste0(name, ".json"))
  jsonlite::write_json(obj, path, auto_unbox = TRUE, digits = 12, pretty = TRUE)
  cat(sprintf("wrote %s\n", path))
}

# ---------------------------------------------------------------------------
# Aalen additive model on the lung dataset: time ~ age + sex
# ---------------------------------------------------------------------------

data(cancer, package = "survival")
lung_clean <- lung[complete.cases(lung[, c("time", "status", "age", "sex")]), ]

fit <- aareg(Surv(time, status) ~ age + sex, data = lung_clean)
s <- summary(fit)

# Cumulative coefficients: cumsum of the per-event-time increments
# fit$coefficient is a matrix (n_event_times x (1 + p)) where first col is intercept
event_times <- fit$times
coef_increments <- fit$coefficient  # (n_times x (1+p))
cumulative_coefs <- apply(coef_increments, 2, cumsum)

# Summary table: slope, se, robust se, z, p
slope <- s$table[, "slope"]
coef_summary <- s$table[, "coef"]
se <- s$table[, "se(coef)"]
z <- s$table[, "z"]
p_value <- s$table[, "p"]

# Test statistics
test_statistic <- fit$test.statistic
test_var <- fit$test.var

# Number at risk at each event time
nrisk <- fit$nrisk

write_json_fixture(
  list(
    n = fit$n,
    event_times = as.numeric(event_times),
    nrisk = as.numeric(nrisk),
    coef_increments = as.list(as.data.frame(coef_increments)),
    cumulative_coefs = as.list(as.data.frame(cumulative_coefs)),
    term_names = colnames(coef_increments),
    summary_slope = as.numeric(slope),
    summary_coef = as.numeric(coef_summary),
    summary_se = as.numeric(se),
    summary_z = as.numeric(z),
    summary_p = as.numeric(p_value),
    test_statistic = as.numeric(test_statistic),
    test_var = as.list(as.data.frame(test_var)),
    test_type = fit$test
  ),
  "aalen_lung_age_sex"
)

# ---------------------------------------------------------------------------
# Aalen model on the veteran dataset: time ~ trt + karno + diagtime + age
# A second fixture with more covariates for broader validation
# ---------------------------------------------------------------------------

data(veteran, package = "survival")

fit2 <- aareg(Surv(time, status) ~ trt + karno + diagtime + age, data = veteran)
s2 <- summary(fit2)

event_times2 <- fit2$times
coef_increments2 <- fit2$coefficient
cumulative_coefs2 <- apply(coef_increments2, 2, cumsum)

write_json_fixture(
  list(
    n = fit2$n,
    event_times = as.numeric(event_times2),
    nrisk = as.numeric(fit2$nrisk),
    coef_increments = as.list(as.data.frame(coef_increments2)),
    cumulative_coefs = as.list(as.data.frame(cumulative_coefs2)),
    term_names = colnames(coef_increments2),
    summary_slope = as.numeric(s2$table[, "slope"]),
    summary_coef = as.numeric(s2$table[, "coef"]),
    summary_se = as.numeric(s2$table[, "se(coef)"]),
    summary_z = as.numeric(s2$table[, "z"]),
    summary_p = as.numeric(s2$table[, "p"]),
    test_statistic = as.numeric(fit2$test.statistic),
    test_var = as.list(as.data.frame(fit2$test.var)),
    test_type = fit2$test
  ),
  "aalen_veteran"
)

cat("Done.\n")
