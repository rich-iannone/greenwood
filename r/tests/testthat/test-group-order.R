# The group order rule (spec/results.md). Groups come in the order of their values, never the order
# of the rows: a factor keeps its levels, and other values are sorted, numbers in numeric order and
# text in code-point order. With several grouping variables the first varies slowest. Every
# estimator that fits one curve per group, and every test that compares groups, uses the same order.
# The Python suite checks the same cases (python/tests/test_group_order.py).

LEVELS <- c("squamous", "smallcell", "adeno", "large")
CELLTYPE <- paste0("celltype=", LEVELS)
SORTED_CELLTYPE <- paste0("celltype=", sort(LEVELS, method = "radix"))

km_strata <- function(fit) unique(as.data.frame(fit)$strata)

veteran_text <- function() {
  out <- greenwood::veteran
  out$celltype <- as.character(out$celltype)
  out
}

test_that("numbers sort numerically", {
  d <- data.frame(time = 1:6, status = 1, g = c(10, 2, 1, 10, 2, 1))
  expect_equal(km_strata(kaplan_meier(Surv(time, status) ~ g, data = d)), c("g=1", "g=2", "g=10"))
  d$g <- c(0.5, 10, 2.5, 0.5, 10, 2.5)
  expect_equal(
    km_strata(kaplan_meier(Surv(time, status) ~ g, data = d)),
    c("g=0.5", "g=2.5", "g=10")
  )
})

test_that("text sorts by code point in every locale", {
  d <- data.frame(time = 1:4, status = 1, g = c("a", "B", "a", "B"))
  expect_equal(km_strata(kaplan_meier(Surv(time, status) ~ g, data = d)), c("g=B", "g=a"))
  old <- Sys.getlocale("LC_COLLATE")
  on.exit(Sys.setlocale("LC_COLLATE", old), add = TRUE)
  if (nzchar(suppressWarnings(Sys.setlocale("LC_COLLATE", "en_US.UTF-8")))) {
    expect_equal(km_strata(kaplan_meier(Surv(time, status) ~ g, data = d)), c("g=B", "g=a"))
  }
})

test_that("logicals put FALSE first", {
  d <- data.frame(time = 1:4, status = 1, g = c(TRUE, FALSE, TRUE, FALSE))
  expect_equal(km_strata(kaplan_meier(Surv(time, status) ~ g, data = d)), c("g=FALSE", "g=TRUE"))
})

test_that("the row order doesn't change the result", {
  vet <- veteran_text()
  set.seed(3)
  shuffled <- vet[sample(nrow(vet)), ]
  original <- as.data.frame(kaplan_meier(Surv(time, status) ~ celltype, data = vet))
  reordered <- as.data.frame(kaplan_meier(Surv(time, status) ~ celltype, data = shuffled))
  expect_equal(original, reordered)
  expect_equal(unique(original$strata), SORTED_CELLTYPE)
})

test_that("a factor keeps its levels, skipping those that don't occur", {
  vet <- greenwood::veteran
  expect_equal(km_strata(kaplan_meier(Surv(time, status) ~ celltype, data = vet)), CELLTYPE)
  vet$celltype <- factor(vet$celltype, levels = c("other", LEVELS))
  expect_equal(km_strata(kaplan_meier(Surv(time, status) ~ celltype, data = vet)), CELLTYPE)
})

test_that("with several variables the first varies slowest", {
  d <- data.frame(
    time = 1:8,
    status = 1,
    dose = c(10, 2, 10, 2, 10, 2, 10, 2),
    arm = c("b", "b", "a", "a", "b", "b", "a", "a")
  )
  expect_equal(
    km_strata(kaplan_meier(Surv(time, status) ~ dose + arm, data = d)),
    c("dose=2, arm=a", "dose=2, arm=b", "dose=10, arm=a", "dose=10, arm=b")
  )
  d$arm <- factor(d$arm, levels = c("b", "a"))
  expect_equal(
    km_strata(kaplan_meier(Surv(time, status) ~ arm + dose, data = d)),
    c("arm=b, dose=2", "arm=b, dose=10", "arm=a, dose=2", "arm=a, dose=10")
  )
})

test_that("every curve estimator uses the same order", {
  vet <- greenwood::veteran
  na <- as.data.frame(nelson_aalen(Surv(time, status) ~ celltype, data = vet))
  expect_equal(unique(na$strata), CELLTYPE)
  et <- as.data.frame(event_table(Surv(time, status) ~ celltype, data = vet))
  expect_equal(unique(et$strata), CELLTYPE)

  cox <- cox_ph(Surv(time, status) ~ karno + strata(celltype), data = vet)
  expect_equal(unique(baseline_hazard(cox)$strata), CELLTYPE)
  plain <- cox_ph(Surv(time, status) ~ karno + strata(celltype), data = veteran_text())
  expect_equal(coef(cox), coef(plain), tolerance = 1e-12)
  expect_equal(unique(baseline_hazard(plain)$strata), SORTED_CELLTYPE)

  event <- vet$status == 1
  vet$lower <- ifelse(event, vet$time * 0.8, vet$time)
  vet$upper <- ifelse(event, vet$time, Inf)
  tb <- as.data.frame(turnbull(Surv(lower, upper, type = "interval2") ~ celltype, data = vet))
  expect_equal(unique(tb$strata), CELLTYPE)
})

test_that("Aalen-Johansen uses the same order", {
  mg <- greenwood::mgus2
  mg$etime <- ifelse(mg$pstat == 1, mg$ptime, mg$futime)
  mg$event <- factor(ifelse(mg$pstat == 1, 1L, 2L * mg$death), 0:2, c("censor", "pcm", "death"))
  mg$sex <- factor(mg$sex, levels = c("M", "F"))
  aj <- as.data.frame(aalen_johansen(Surv(etime, event) ~ sex, data = mg))
  expect_equal(unique(aj$strata), c("sex=M", "sex=F"))
})

test_that("comparisons use the group order", {
  by_level <- logrank_test(Surv(time, status) ~ celltype, data = greenwood::veteran)
  by_value <- logrank_test(Surv(time, status) ~ celltype, data = veteran_text())
  expect_equal(as.character(by_level$groups$celltype), LEVELS)
  expect_equal(as.character(by_value$groups$celltype), sort(LEVELS, method = "radix"))
  expect_equal(by_level$statistic, by_value$statistic, tolerance = 1e-12)

  # The default trend scores are 0, 1, 2, 3 in group order.
  trend <- trend_test(Surv(time, status) ~ celltype, data = greenwood::veteran)
  explicit <- trend_test(
    Surv(time, status) ~ celltype,
    data = veteran_text(),
    scores = stats::setNames(0:3, LEVELS)
  )
  expect_equal(trend$statistic, explicit$statistic, tolerance = 1e-12)
})

test_that("two-group differences take the first group first", {
  lung <- greenwood::lung
  reversed <- lung
  reversed$sex <- factor(reversed$sex, levels = c(2, 1))

  forward <- rmst_test(Surv(time, status) ~ sex, data = lung, tau = 365)
  backward <- rmst_test(Surv(time, status) ~ sex, data = reversed, tau = 365)
  expect_equal(c(forward$group1, forward$group2), c("1", "2"))
  expect_equal(c(backward$group1, backward$group2), c("2", "1"))
  expect_equal(backward$estimate, -forward$estimate, tolerance = 1e-12)

  set.seed(1)
  forward <- bootstrap(Surv(time, status) ~ sex, data = lung, statistic = "median_diff", n_boot = 20)
  set.seed(1)
  backward <- bootstrap(
    Surv(time, status) ~ sex,
    data = reversed,
    statistic = "median_diff",
    n_boot = 20
  )
  expect_equal(forward$estimate, 270 - 426)
  expect_equal(backward$estimate, 426 - 270)
  expect_true(mean(backward$distribution, na.rm = TRUE) > 0)
  expect_true(mean(forward$distribution, na.rm = TRUE) < 0)
})
