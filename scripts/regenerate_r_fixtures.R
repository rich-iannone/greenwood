#!/usr/bin/env Rscript
# Regenerate the R-parity numeric fixtures for Greenwood.
#
# Correctness against R's `survival` is the credibility currency for Greenwood, so the
# risk-set/event-table kernel is validated against `survfit`'s tabulation. Run from the
# repo root:  Rscript scripts/regenerate_r_fixtures.R
#
# Writes JSON into tests/fixtures/r/. The Python harness (tests/_r_parity.py) loads these
# and asserts to tolerance.

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

# One survfit object -> a list of {time, n_risk, n_event, n_censor}, split by strata.
tabulate_survfit <- function(sf) {
  block <- function(t, r, e, c) {
    list(time = t, n_risk = r, n_event = e, n_censor = c)
  }
  if (is.null(sf$strata)) {
    return(list(overall = block(sf$time, sf$n.risk, sf$n.event, sf$n.censor)))
  }
  out <- list()
  ends <- cumsum(sf$strata)
  starts <- c(1L, head(ends, -1L) + 1L)
  nms <- names(sf$strata)
  for (i in seq_along(sf$strata)) {
    ix <- starts[i]:ends[i]
    # Strata names look like "sex=1"; key by the level value after "=".
    key <- sub("^[^=]*=", "", nms[i])
    out[[key]] <- block(sf$time[ix], sf$n.risk[ix], sf$n.event[ix], sf$n.censor[ix])
  }
  out
}

data(cancer, package = "survival") # lung: status 1 = censored, 2 = dead

write_json_fixture(
  tabulate_survfit(survfit(Surv(time, status) ~ 1, data = lung)),
  "lung_km_overall"
)
write_json_fixture(
  tabulate_survfit(survfit(Surv(time, status) ~ sex, data = lung)),
  "lung_km_by_sex"
)

data(veteran, package = "survival")
write_json_fixture(
  tabulate_survfit(survfit(Surv(time, status) ~ 1, data = veteran)),
  "veteran_km_overall"
)

# Left truncation / counting-process case, to validate the entry-aware risk set.
trunc <- data.frame(
  start = c(0, 2, 1, 3, 0, 4, 1, 2),
  stop  = c(5, 6, 4, 8, 7, 9, 6, 5),
  event = c(1, 0, 1, 1, 0, 1, 1, 0)
)
write_json_fixture(
  c(
    tabulate_survfit(survfit(Surv(start, stop, event) ~ 1, data = trunc)),
    list(data = as.list(trunc))
  ),
  "counting_truncation"
)

# -- Kaplan-Meier survival + Greenwood CIs, and Nelson-Aalen cumulative hazard ------

# Compute the KM/NA reference for one subset (a data.frame with `time`, `status`).
# `surv`/`se` are conf.type-independent; lower/upper are exported per conf.type.
km_one <- function(d) {
  fp <- survfit(Surv(time, status) ~ 1, data = d, conf.type = "plain")
  fl <- survfit(Surv(time, status) ~ 1, data = d, conf.type = "log")
  fll <- survfit(Surv(time, status) ~ 1, data = d, conf.type = "log-log")
  # survfit stores std.err on the log scale: se(log S) = se(S) / S, so se(S) = std.err * S.
  se_surv <- fl$std.err * fl$surv
  varh <- cumsum(fl$n.event / fl$n.risk^2) # Aalen variance of the Nelson-Aalen cumhaz
  q <- quantile(fl, probs = 0.5, conf.int = TRUE)
  list(
    time = fl$time, surv = fl$surv, se = se_surv,
    lower_plain = fp$lower, upper_plain = fp$upper,
    lower_log = fl$lower, upper_log = fl$upper,
    lower_loglog = fll$lower, upper_loglog = fll$upper,
    cumhaz = fl$cumhaz, cumhaz_var = varh,
    median = unname(q$quantile), median_lower = unname(q$lower),
    median_upper = unname(q$upper)
  )
}

write_json_fixture(list(overall = km_one(lung)), "km_lung_overall")

lung_by_sex <- list()
for (s in sort(unique(lung$sex))) {
  lung_by_sex[[as.character(s)]] <- km_one(lung[lung$sex == s, ])
}
write_json_fixture(lung_by_sex, "km_lung_by_sex")

write_json_fixture(list(overall = km_one(veteran)), "km_veteran_overall")

# -- Restricted mean survival time (RMST) up to tau ---------------------------------

rmst_one <- function(d, tau) {
  tab <- summary(survfit(Surv(time, status) ~ 1, data = d), rmean = tau)$table
  list(tau = tau, rmst = unname(tab["rmean"]), se = unname(tab["se(rmean)"]))
}

write_json_fixture(
  list(lung = rmst_one(lung, 365), veteran = rmst_one(veteran, 180)),
  "rmst"
)

# -- Two-group RMST comparison (lung, sex groups at tau=365) ------------------------

rmst_twogroup_lung <- function() {
  sf1 <- summary(survfit(Surv(time, status) ~ 1, data = lung[lung$sex == 1, ]), rmean = 365)$table
  sf2 <- summary(survfit(Surv(time, status) ~ 1, data = lung[lung$sex == 2, ]), rmean = 365)$table
  rmst1 <- unname(sf1["rmean"])
  se1 <- unname(sf1["se(rmean)"])
  rmst2 <- unname(sf2["rmean"])
  se2 <- unname(sf2["se(rmean)"])

  # Stratified by ph.ecog: inverse-variance pooling of per-stratum differences.
  l_clean <- lung[!is.na(lung$ph.ecog), ]
  strata_levels <- sort(unique(l_clean$ph.ecog))
  d_vals <- numeric(0)
  var_vals <- numeric(0)
  for (s in strata_levels) {
    ls <- l_clean[l_clean$ph.ecog == s, ]
    if (length(unique(ls$sex)) < 2) next
    s1 <- summary(survfit(Surv(time, status) ~ 1, data = ls[ls$sex == 1, ]), rmean = 365)$table
    s2 <- summary(survfit(Surv(time, status) ~ 1, data = ls[ls$sex == 2, ]), rmean = 365)$table
    d_vals <- c(d_vals, s1["rmean"] - s2["rmean"])
    var_vals <- c(var_vals, s1["se(rmean)"]^2 + s2["se(rmean)"]^2)
  }
  w <- 1 / var_vals
  W <- sum(w)

  list(
    tau = 365, group1 = 1, group2 = 2,
    rmst1 = rmst1, se1 = se1, rmst2 = rmst2, se2 = se2,
    difference = rmst1 - rmst2,
    se_difference = sqrt(se1^2 + se2^2),
    stratified = list(
      strata_var  = "ph.ecog",
      difference  = sum(w * d_vals) / W,
      se          = sqrt(1 / W),
      statistic   = (sum(w * d_vals) / W) / sqrt(1 / W),
      p_value     = 2 * (1 - pnorm(abs((sum(w * d_vals) / W) / sqrt(1 / W))))
    )
  )
}

write_json_fixture(rmst_twogroup_lung(), "rmst_twogroup_lung")

# -- Log-rank and G-rho (Fleming-Harrington) tests via survdiff ---------------------

sd_fixture <- function(sd) {
  labels <- sub("^[^=]*=", "", names(sd$n))
  df <- length(sd$n) - 1
  list(
    groups = labels,
    n = as.numeric(sd$n),
    obs = as.numeric(sd$obs),
    exp = as.numeric(sd$exp),
    chisq = as.numeric(sd$chisq),
    df = df,
    p = pchisq(as.numeric(sd$chisq), df, lower.tail = FALSE)
  )
}

write_json_fixture(
  sd_fixture(survdiff(Surv(time, status) ~ sex, data = lung, rho = 0)),
  "logrank_lung_sex"
)
write_json_fixture(
  sd_fixture(survdiff(Surv(time, status) ~ sex, data = lung, rho = 1)),
  "grho_lung_sex_rho1"
)
write_json_fixture(
  sd_fixture(survdiff(Surv(time, status) ~ celltype, data = veteran, rho = 0)),
  "logrank_veteran_celltype"
)

# Stratified log-rank: compare sex within levels of ph.ecog (survdiff with strata()).
logrank_stratified_fixture <- function() {
  l <- lung[!is.na(lung$ph.ecog), ]
  sd <- survdiff(Surv(time, status) ~ sex + strata(ph.ecog), data = l)
  df <- length(sd$n) - 1
  list(
    chisq = as.numeric(sd$chisq), df = df,
    p = pchisq(as.numeric(sd$chisq), df, lower.tail = FALSE)
  )
}
write_json_fixture(logrank_stratified_fixture(), "logrank_stratified_lung_sex_ecog")

# Pairwise log-rank across the four veteran cell types, Holm-adjusted (as pairwise_survdiff).
pairwise_logrank_fixture <- function() {
  v <- veteran
  g <- as.character(v$celltype)
  levels_sorted <- sort(unique(g))
  pairs <- combn(levels_sorted, 2)
  raw <- numeric(ncol(pairs))
  for (k in seq_len(ncol(pairs))) {
    mask <- g %in% pairs[, k]
    sdi <- survdiff(Surv(time, status) ~ celltype, data = v[mask, ])
    raw[k] <- pchisq(as.numeric(sdi$chisq), 1, lower.tail = FALSE)
  }
  list(
    group1 = pairs[1, ], group2 = pairs[2, ], p_value = raw,
    holm = p.adjust(raw, "holm"), bh = p.adjust(raw, "BH"),
    bonferroni = p.adjust(raw, "bonferroni")
  )
}
write_json_fixture(pairwise_logrank_fixture(), "pairwise_logrank_veteran")

# -- Numbers at risk at fixed times (for risk tables) -------------------------------

risk_table_fixture <- function(fit, times) {
  s <- summary(fit, times = times, extend = TRUE)
  out <- list(times = times)
  strata <- if (is.null(s$strata)) rep("overall", length(s$time)) else as.character(s$strata)
  by_stratum <- list()
  for (lev in unique(strata)) {
    key <- sub("^[^=]*=", "", lev)
    by_stratum[[key]] <- as.numeric(s$n.risk[strata == lev])
  }
  c(out, list(n_risk = by_stratum))
}

write_json_fixture(
  risk_table_fixture(
    survfit(Surv(time, status) ~ sex, data = lung), c(0, 250, 500, 750, 1000)
  ),
  "risk_table_lung_sex"
)

# -- Cox proportional hazards via coxph ---------------------------------------------

coxph_fixture <- function(formula, data, ties) {
  cm <- coxph(formula, data = data, ties = ties)
  s <- summary(cm)
  co <- s$coefficients
  ci <- s$conf.int
  list(
    terms = rownames(co),
    coef = unname(co[, "coef"]),
    se = unname(co[, "se(coef)"]),
    z = unname(co[, "z"]),
    p = unname(co[, "Pr(>|z|)"]),
    exp_coef = unname(co[, "exp(coef)"]),
    conf_low = unname(ci[, "lower .95"]),
    conf_high = unname(ci[, "upper .95"]),
    loglik_null = cm$loglik[1],
    loglik = cm$loglik[2],
    n = cm$n,
    nevent = cm$nevent,
    lr = list(stat = unname(s$logtest["test"]), df = unname(s$logtest["df"]), p = unname(s$logtest["pvalue"])),
    wald = list(stat = unname(s$waldtest["test"]), df = unname(s$waldtest["df"]), p = unname(s$waldtest["pvalue"])),
    score = list(stat = unname(s$sctest["test"]), df = unname(s$sctest["df"]), p = unname(s$sctest["pvalue"]))
  )
}

write_json_fixture(coxph_fixture(Surv(time, status) ~ age + sex, lung, "efron"), "cox_lung_age_sex_efron")
write_json_fixture(coxph_fixture(Surv(time, status) ~ age + sex, lung, "breslow"), "cox_lung_age_sex_breslow")
write_json_fixture(coxph_fixture(Surv(time, status) ~ age + sex + ph.ecog, lung, "efron"), "cox_lung_three_efron")

# -- Cox diagnostics, baseline hazard, and prediction -------------------------------

cox_diag_fixture <- function(ties) {
  cm <- coxph(Surv(time, status) ~ age + sex, data = lung, ties = ties)
  bh <- basehaz(cm, centered = FALSE)
  sch <- residuals(cm, "schoenfeld")
  zph_table <- function(tr) {
    z <- cox.zph(cm, transform = tr, global = TRUE)
    list(
      age = list(chisq = z$table["age", "chisq"], df = z$table["age", "df"], p = z$table["age", "p"]),
      sex = list(chisq = z$table["sex", "chisq"], df = z$table["sex", "df"], p = z$table["sex", "p"]),
      global = list(chisq = z$table["GLOBAL", "chisq"], df = z$table["GLOBAL", "df"], p = z$table["GLOBAL", "p"])
    )
  }
  newdata <- data.frame(age = c(50, 70), sex = c(1, 2))
  times <- c(100, 300, 500)
  sf <- summary(survfit(cm, newdata = newdata), times = times)
  conc <- summary(cm)$concordance

  list(
    ties = ties,
    basehaz_time = bh$time,
    basehaz_cumhaz = bh$hazard,
    martingale = unname(residuals(cm, "martingale")),
    schoenfeld = list(age = unname(sch[, 1]), sex = unname(sch[, 2])),
    lp = unname(predict(cm, type = "lp")),
    concordance = unname(conc["C"]),
    concordance_se = unname(conc["se(C)"]),
    zph_identity = zph_table("identity"),
    zph_log = zph_table("log"),
    surv_times = times,
    surv_newdata_age = newdata$age,
    surv_newdata_sex = newdata$sex,
    surv = list(subj1 = sf$surv[, 1], subj2 = sf$surv[, 2])
  )
}

write_json_fixture(cox_diag_fixture("breslow"), "cox_diag_breslow")
write_json_fixture(cox_diag_fixture("efron"), "cox_diag_efron")

# Predicted survival curves with confidence bands (survfit.coxph, conf.type="log").
cox_survci_fixture <- function() {
  cm <- coxph(Surv(time, status) ~ age + sex, data = lung, ties = "breslow")
  nd <- data.frame(age = c(50, 70), sex = c(1, 2))
  sf <- survfit(cm, newdata = nd)
  times <- c(180, 365, 540)
  i <- findInterval(times, sf$time)
  list(
    newdata_age = nd$age,
    newdata_sex = nd$sex,
    times = times,
    surv = list(subj1 = sf$surv[i, 1], subj2 = sf$surv[i, 2]),
    lower = list(subj1 = sf$lower[i, 1], subj2 = sf$lower[i, 2]),
    upper = list(subj1 = sf$upper[i, 1], subj2 = sf$upper[i, 2]),
    se_chaz = list(subj1 = sf$std.chaz[i, 1], subj2 = sf$std.chaz[i, 2])
  )
}
write_json_fixture(cox_survci_fixture(), "cox_survci_breslow")

# Baseline hazard CIs from survfit.coxph (no newdata = baseline survival).
cox_basehaz_ci_fixture <- function() {
  cm <- coxph(Surv(time, status) ~ age + sex, data = lung, ties = "breslow")
  sf <- survfit(cm)  # baseline survival at mean covariates
  times <- c(100, 200, 365, 500, 800)
  i <- findInterval(times, sf$time)
  list(
    times = times,
    cumhaz = sf$cumhaz[i],
    survival = sf$surv[i],
    se_cumhaz = sf$std.chaz[i],
    lower = sf$lower[i],
    upper = sf$upper[i],
    conf_type = sf$conf.type
  )
}
write_json_fixture(cox_basehaz_ci_fixture(), "cox_basehaz_ci")

# Time-varying-covariate Cox on counting-process (start, stop] data (the heart transplant
# study). `transplant` changes within a subject across intervals. The data are stored in the
# fixture so the Python test reconstructs the exact same design.
cox_timevarying_fixture <- function() {
  h <- heart
  cm <- coxph(Surv(start, stop, event) ~ age + surgery + transplant, data = h, ties = "breslow")
  list(
    start = h$start, stop = h$stop, event = h$event,
    age = h$age, surgery = h$surgery, transplant = as.integer(as.character(h$transplant)),
    terms = names(coef(cm)),
    coef = unname(coef(cm)),
    se = unname(sqrt(diag(cm$var))),
    loglik = cm$loglik[2],
    n = cm$n,
    nevent = cm$nevent
  )
}
write_json_fixture(cox_timevarying_fixture(), "cox_timevarying")

# -- Stratified Cox and robust (sandwich) variance ----------------------------------

cm_strata <- coxph(Surv(time, status) ~ age + ph.ecog + strata(sex), data = lung)
ss <- summary(cm_strata)
write_json_fixture(
  list(
    terms = rownames(ss$coefficients),
    coef = unname(ss$coefficients[, "coef"]),
    se = unname(sqrt(diag(cm_strata$var))),
    loglik_null = cm_strata$loglik[1],
    loglik = cm_strata$loglik[2],
    n = cm_strata$n,
    nevent = cm_strata$nevent,
    lr = unname(ss$logtest["test"]),
    wald = unname(ss$waldtest["test"]),
    score = unname(ss$sctest["test"])
  ),
  "cox_strata"
)

cm_robust <- coxph(Surv(time, status) ~ age + sex, data = lung, robust = TRUE, ties = "breslow")
sr <- summary(cm_robust)$coefficients
write_json_fixture(
  list(
    terms = rownames(sr),
    coef = unname(sr[, "coef"]),
    naive_se = unname(sr[, "se(coef)"]),
    robust_se = unname(sr[, "robust se"]),
    z = unname(sr[, "z"]),
    p = unname(sr[, "Pr(>|z|)"])
  ),
  "cox_robust"
)

cm_cluster <- coxph(Surv(time, status) ~ age + sex + cluster(inst), data = lung, ties = "breslow")
write_json_fixture(
  list(
    terms = names(cm_cluster$coef),
    coef = unname(cm_cluster$coef),
    robust_se = unname(sqrt(diag(cm_cluster$var))),
    n = cm_cluster$n
  ),
  "cox_cluster"
)

# -- Parametric AFT models via survreg ----------------------------------------------

aft_pred_newdata <- data.frame(age = c(50, 70, 60), sex = c(1, 2, 1))
aft_pred_p <- c(0.1, 0.25, 0.5, 0.75, 0.9)

aft_fixture <- function(dist) {
  m <- survreg(Surv(time, status) ~ age + sex, data = lung, dist = dist)
  se <- sqrt(diag(m$var))
  ncoef <- length(m$coef)
  # Predicted survival-time quantiles: rows = subjects, cols = probabilities.
  pred_q <- as.matrix(predict(m, aft_pred_newdata, type = "quantile", p = aft_pred_p))
  dimnames(pred_q) <- NULL
  list(
    dist = dist,
    terms = names(m$coef),
    coef = unname(m$coef),
    coef_se = unname(se[seq_len(ncoef)]),
    scale = unname(m$scale),
    log_scale_se = if (dist == "exponential") NA else unname(se[ncoef + 1]),
    loglik = m$loglik[2],
    n = length(m$linear.predictors),
    pred_p = aft_pred_p,
    pred_newdata = list(age = aft_pred_newdata$age, sex = aft_pred_newdata$sex),
    pred_quantile = pred_q
  )
}

for (d in c("weibull", "exponential", "lognormal", "loglogistic")) {
  write_json_fixture(aft_fixture(d), paste0("aft_", d))
}

# -- Univariate parametric distributions (intercept-only survreg) -------------------

parametric_pred_p <- c(0.1, 0.25, 0.5, 0.75, 0.9)
parametric_pred_times <- c(100, 200, 365, 500, 730)

parametric_fixture <- function(dist) {
  m <- survreg(Surv(time, status) ~ 1, data = lung, dist = dist)
  se <- sqrt(diag(m$var))
  ncoef <- length(m$coef)
  pred_q <- as.numeric(predict(m, data.frame(row.names = 1),
    type = "quantile",
    p = parametric_pred_p
  ))
  # Parametric S(t) at specified times.
  lp <- m$coefficients[1]
  sigma <- m$scale
  z <- (log(parametric_pred_times) - lp) / sigma
  if (dist %in% c("weibull", "exponential")) {
    surv <- exp(-exp(z))
  } else if (dist == "lognormal") {
    surv <- pnorm(-z)
  } else {
    surv <- 1 / (1 + exp(z))
  }
  list(
    dist = dist,
    mu = unname(m$coef[1]),
    mu_se = unname(se[1]),
    scale = unname(m$scale),
    log_scale_se = if (dist == "exponential") NA else unname(se[ncoef + 1]),
    loglik = m$loglik[2],
    n = m$df.residual + ncoef + (if (dist == "exponential") 0 else 1),
    nevent = sum(lung$status == 2),
    pred_p = parametric_pred_p,
    pred_quantile = pred_q,
    pred_times = parametric_pred_times,
    pred_survival = as.numeric(surv)
  )
}

for (d in c("weibull", "exponential", "lognormal", "loglogistic")) {
  write_json_fixture(parametric_fixture(d), paste0("parametric_", d))
}

# Weibull anchor for the Royston-Parmar model: with df=1 (no internal knots) the flexible
# parametric model is a Weibull, so it must reproduce survreg's log-likelihood and predicted
# survival. Store S(t|x) = 1 - pweibull(t, shape=1/scale, scale=exp(lp)).
rp_weibull_anchor <- function() {
  sr <- survreg(Surv(time, status) ~ age + sex, data = lung, dist = "weibull")
  nd <- data.frame(age = c(50, 70), sex = c(1, 2))
  lp <- predict(sr, nd, type = "lp")
  shape <- 1 / sr$scale
  times <- c(180, 365, 540)
  surv <- sapply(seq_len(nrow(nd)), function(i) 1 - pweibull(times, shape, scale = exp(lp[i])))
  list(
    loglik = sr$loglik[2],
    newdata_age = nd$age, newdata_sex = nd$sex, times = times,
    surv = list(subj1 = surv[, 1], subj2 = surv[, 2])
  )
}
write_json_fixture(rp_weibull_anchor(), "rp_weibull_anchor")

# -- Competing risks: Aalen-Johansen CIF and Fine-Gray -------------------------------

data(mgus2, package = "survival")
mg <- mgus2
mg$etime <- ifelse(mg$pstat == 1, mg$ptime, mg$futime)
mg$event <- ifelse(mg$pstat == 1, 1L, 2L * mg$death) # 0 = censor, 1 = pcm, 2 = death
mg$event_f <- factor(mg$event, 0:2, c("censor", "pcm", "death"))

sf <- survfit(Surv(etime, event_f) ~ 1, data = mg)
# pstate columns: (s0), pcm, death; std.err aligned.
# Export CIs for all three conf.type values so Python can validate each transform.
cif_ci_by_type <- function(ct) {
  s <- survfit(Surv(etime, event_f) ~ 1, data = mg, conf.type = ct)
  key <- gsub("-", "", ct) # "log-log" -> "loglog"
  stats::setNames(
    list(s$lower[, 2], s$upper[, 2], s$lower[, 3], s$upper[, 3]),
    paste0(c("lower_pcm_", "upper_pcm_", "lower_death_", "upper_death_"), key)
  )
}
cif_fixture <- c(
  list(
    time      = sf$time,
    n_risk    = sf$n.risk[, 1],
    cif_pcm   = sf$pstate[, 2],
    cif_death = sf$pstate[, 3],
    se_pcm    = sf$std.err[, 2],
    se_death  = sf$std.err[, 3]
  ),
  cif_ci_by_type("plain"),
  cif_ci_by_type("log"),
  cif_ci_by_type("log-log")
)
write_json_fixture(cif_fixture, "cif_mgus2")

fg <- finegray(Surv(etime, event_f) ~ age + sex + id, data = mg, etype = "pcm")
fgmod <- coxph(
  Surv(fgstart, fgstop, fgstatus) ~ age + sex + cluster(id),
  weights = fgwt, data = fg, ties = "breslow"
)
sfg <- summary(fgmod)$coefficients
write_json_fixture(
  list(
    terms = rownames(sfg),
    coef = unname(sfg[, "coef"]),
    naive_se = unname(sfg[, "se(coef)"]),
    robust_se = unname(sfg[, "robust se"])
  ),
  "finegray_mgus2_pcm"
)

# Multi-state (illness-death) occupancy probabilities: mgus -> pcm -> death.
ms_rows <- list()
k <- 1
for (i in seq_len(nrow(mg))) {
  pt <- mg$ptime[i]
  ft <- mg$futime[i]
  prog <- mg$pstat[i] == 1
  died <- mg$death[i] == 1
  if (prog && pt < ft) {
    ms_rows[[k]] <- data.frame(id = mg$id[i], t0 = 0, t1 = pt, from = "mgus", ev = "pcm")
    k <- k + 1
    ms_rows[[k]] <- data.frame(
      id = mg$id[i], t0 = pt, t1 = ft, from = "pcm", ev = if (died) "death" else "censor"
    )
    k <- k + 1
  } else {
    ev <- if (died) "death" else if (prog) "pcm" else "censor"
    ms_rows[[k]] <- data.frame(id = mg$id[i], t0 = 0, t1 = ft, from = "mgus", ev = ev)
    k <- k + 1
  }
}
msd <- do.call(rbind, ms_rows)
msd <- msd[msd$t1 > msd$t0, ]
msd$ev <- factor(msd$ev, levels = c("censor", "pcm", "death"))
msd$from <- factor(msd$from, levels = c("mgus", "pcm", "death"))
msf <- survfit(Surv(t0, t1, ev) ~ 1, data = msd, id = id, istate = from)
write_json_fixture(
  list(
    time = msf$time, states = msf$states,
    mgus = msf$pstate[, 1], pcm = msf$pstate[, 2], death = msf$pstate[, 3]
  ),
  "multistate_mgus2"
)

# Gray's test needs the cmprsk package (not in this toolchain); planned next.

# -- Prediction performance: IPCW Brier score (survival:::brier) --------------------

brier_fit <- coxph(Surv(time, status) ~ age + sex, data = lung, x = TRUE)
brier_times <- c(180, 365, 540)
brier_out <- survival:::brier(brier_fit, times = brier_times)
write_json_fixture(
  list(times = brier_times, brier = unname(brier_out$brier[, "Model"])),
  "brier_lung"
)

# -- Time-dependent AUC: Uno et al. (2011) IPCW cumulative-dynamic AUC ----------
#
# Direct R implementation of the same formula used in greenwood._metrics.time_dependent_auc.
# No third-party package is required; survival:: provides the censoring KM.
# Reference: Uno H. et al. (2011) Stat Med 30(10):1105-1117.

uno_td_auc <- function(T, delta, marker, times) {
  # KM of the censoring distribution using the same "nudged" Fine-Gray convention
  # as Python's greenwood._competing._censoring_km:
  #   At each censoring time c, events tied at c are excluded from the risk set
  #   (treated as having left just before c).  This matches the Python implementation.
  delta_int <- as.integer(delta)
  censor_times <- sort(unique(T[delta_int == 0L]))
  g_surv_val <- 1.0
  g_times_vec <- numeric(0)
  g_surv_vec <- numeric(0)
  for (c in censor_times) {
    n_risk <- sum(T > c) + sum(delta_int == 0L & T == c)
    d <- sum(delta_int == 0L & T == c)
    g_surv_val <- g_surv_val * (1.0 - d / n_risk)
    g_times_vec <- c(g_times_vec, c)
    g_surv_vec <- c(g_surv_vec, g_surv_val)
  }

  # G(t-): censoring survival just before t (left-continuous).
  g_left <- function(t_vec) {
    sapply(t_vec, function(t) {
      idx <- which(g_times_vec < t) # strictly less than t
      if (length(idx) == 0L) 1.0 else g_surv_vec[max(idx)]
    })
  }

  sapply(times, function(t) {
    cases <- which(T <= t & delta_int == 1L)
    ctrls <- which(T > t)
    if (length(cases) == 0L || length(ctrls) == 0L) {
      return(NA_real_)
    }

    g_case <- g_left(T[cases])
    w <- ifelse(g_case > 0, 1.0 / g_case^2, 0.0)

    eta_c <- marker[cases]
    eta_k <- marker[ctrls]

    conc_per_case <- vapply(seq_along(cases), function(i) {
      d <- eta_c[i] - eta_k
      sum(d > 0) + 0.5 * sum(d == 0)
    }, numeric(1))

    num <- sum(w * conc_per_case)
    denom <- sum(w) * length(ctrls)
    if (denom == 0) NA_real_ else num / denom
  })
}

# Use the Cox LP (age + sex, Efron) on the lung dataset as the risk marker —
# the same model validated elsewhere in the test suite.
auc_cm <- coxph(Surv(time, status) ~ age + sex, data = lung, ties = "efron")
auc_lp <- unname(predict(auc_cm, type = "lp"))
auc_times <- c(180, 365, 540)
auc_vals <- uno_td_auc(
  T      = lung$time,
  delta  = as.integer(lung$status == 2),
  marker = auc_lp,
  times  = auc_times
)

write_json_fixture(
  list(times = auc_times, auc = auc_vals, marker = auc_lp),
  "td_auc_lung"
)

cat("done\n")
