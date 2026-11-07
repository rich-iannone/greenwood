#!/usr/bin/env Rscript
# Generate R-parity fixtures for IPC weights and IPCRidge.
#
# Validates:
# 1. Censoring distribution (Kaplan-Meier of censoring) against survival::survfit
# 2. IPC-weighted ridge regression coefficients
#
# Run from the repo root:
#   Rscript scripts/generate_ipcridge_fixture.R

suppressPackageStartupMessages({
  library(survival)
  if (!requireNamespace("jsonlite", quietly = TRUE)) {
    stop("jsonlite is required: install.packages('jsonlite')")
  }
})

out_dir <- file.path("fixtures", "r")
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

write_json_fixture <- function(obj, name) {
  path <- file.path(out_dir, paste0(name, ".json"))
  jsonlite::write_json(obj, path, auto_unbox = TRUE, digits = 12, pretty = TRUE)
  cat(sprintf("wrote %s\n", path))
}

# ---------------------------------------------------------------------------
# Censoring distribution on the lung dataset
# ---------------------------------------------------------------------------

data(cancer, package = "survival")
lung_clean <- lung[complete.cases(lung[, c("time", "status", "age", "sex")]), ]

# Censoring KM: flip event indicator (censoring = event)
cens_event <- ifelse(lung_clean$status == 2, 0, 1) # 1 = censored (the "event" for G)
cens_fit <- survfit(Surv(lung_clean$time, cens_event) ~ 1)

# IPC weights for uncensored subjects: 1 / G(t_i^-)
# G(t^-) for each subject: the censoring survival just before their event time
# For subjects with events (status == 2), compute G(t^-)
event_mask <- lung_clean$status == 2
event_times <- lung_clean$time[event_mask]

# G(t^-) = G evaluated at the last time strictly before t
# summary(cens_fit) gives the step function; we need left-continuous version
g_times <- cens_fit$time
g_surv <- cens_fit$surv

# G(t^-): for each event time, find the last censoring-KM time < t
g_left <- function(t) {
  idx <- max(which(g_times < t), 0)
  if (idx == 0) {
    return(1.0)
  }
  return(g_surv[idx])
}

ipc_weights_events <- sapply(event_times, g_left)
ipc_weights_full <- rep(0.0, nrow(lung_clean))
ipc_weights_full[event_mask] <- 1.0 / ipc_weights_events

# ---------------------------------------------------------------------------
# IPC-weighted ridge: log(time) ~ age + sex, alpha = 1.0
# ---------------------------------------------------------------------------

# The ridge has an unpenalized intercept and penalizes alpha * sum((beta_j * scale_j)^2), where
# scale_j is the standard deviation of covariate j among the events. It's fit as weighted least
# squares on the design augmented with one pseudo-row per coefficient, sqrt(alpha) * scale_j,
# with a response of 0 and a weight of 1.
x_raw <- cbind(lung_clean$age[event_mask], lung_clean$sex[event_mask])
y_log <- log(lung_clean$time[event_mask])
w_ipc <- ipc_weights_full[event_mask]

alpha <- 1.0
p <- ncol(x_raw)
x_scale <- apply(x_raw, 2, sd)
augmented <- rbind(cbind(1, x_raw), cbind(0, diag(sqrt(alpha) * x_scale, p)))
ridge <- lm.wfit(augmented, c(y_log, rep(0, p)), c(w_ipc, rep(1, p)))
intercept <- unname(ridge$coefficients[1])
beta_orig <- unname(ridge$coefficients[-1])

# Predictions (lp) for all subjects (including censored, using full design)
x_all <- cbind(lung_clean$age, lung_clean$sex)
lp_all <- as.numeric(intercept + x_all %*% beta_orig)

write_json_fixture(
  list(
    censoring_times = g_times,
    censoring_surv = g_surv,
    ipc_weights = ipc_weights_full,
    ridge_alpha = alpha,
    ridge_coef = as.numeric(beta_orig),
    ridge_intercept = intercept,
    ridge_lp = lp_all,
    covariates = c("age", "sex"),
    n = nrow(lung_clean),
    n_event = sum(event_mask),
    event_times = event_times
  ),
  "ipcridge_lung"
)

cat("Done.\n")
