#!/usr/bin/env Rscript
# Generate R-parity fixtures for the mixture cure model.
#
# Validates against R's `smcure` package.
#
# Run from the repo root:
#   Rscript scripts/generate_cure_fixture.R

suppressPackageStartupMessages({
  library(survival)
  library(smcure)
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
# Mixture cure model (PH latency) on the e1684 melanoma dataset
# ---------------------------------------------------------------------------

data(e1684, package = "smcure")
e_clean <- na.omit(e1684)

sink("/dev/null")
fit <- smcure(Surv(FAILTIME, FAILCENS) ~ TRT,
              cureform = ~ TRT,
              data = e1684,
              model = "ph",
              Var = FALSE)
sink()

# Reconstruct baseline survival at unique event times
death_times <- sort(unique(e_clean$FAILTIME[e_clean$FAILCENS == 1]))
ord <- order(fit$Time)
s_sorted <- fit$s[ord]
t_sorted <- fit$Time[ord]

base_s <- numeric(length(death_times))
for (i in seq_along(death_times)) {
  idx <- which(abs(t_sorted - death_times[i]) < 1e-10)[1]
  base_s[i] <- s_sorted[idx]
}

write_json_fixture(
  list(
    data = list(
      FAILTIME = e_clean$FAILTIME,
      FAILCENS = as.integer(e_clean$FAILCENS),
      TRT = as.integer(e_clean$TRT),
      AGE = e_clean$AGE,
      SEX = as.integer(e_clean$SEX)
    ),
    n = nrow(e_clean),
    n_events = sum(e_clean$FAILCENS),
    cure_coef = as.numeric(fit$b),
    cure_coef_names = c("(Intercept)", "TRT"),
    latency_coef = as.numeric(fit$beta),
    latency_coef_names = c("TRT"),
    baseline_times = death_times,
    baseline_survival = base_s,
    model = "ph"
  ),
  "cure_ph_e1684"
)

cat("Done.\n")
