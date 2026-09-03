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

out_dir <- file.path("tests", "fixtures", "r")
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

# Fit weighted ridge by hand: beta = (X'WX + alpha*I)^{-1} X'Wy
x_raw <- cbind(lung_clean$age[event_mask], lung_clean$sex[event_mask])
y_log <- log(lung_clean$time[event_mask])
w <- 1.0 / ipc_weights_events # these are G(t^-), so 1/G(t^-) = ipc weight
w_ipc <- ipc_weights_full[event_mask]

# Standardize
x_center <- colMeans(x_raw)
x_scale <- apply(x_raw, 2, sd)
x_std <- scale(x_raw, center = x_center, scale = x_scale)

alpha <- 1.0
p <- ncol(x_std)

# Weighted ridge on standardized covariates
W <- diag(w_ipc)
gram <- t(x_std) %*% W %*% x_std + alpha * diag(p)
rhs <- t(x_std) %*% (w_ipc * y_log)
beta_std <- solve(gram, rhs)

# Back to original scale
beta_orig <- beta_std / x_scale

# Intercept: weighted mean of y - x_center' * beta_orig
intercept <- weighted.mean(y_log, w_ipc) - sum(x_center * beta_orig)

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
