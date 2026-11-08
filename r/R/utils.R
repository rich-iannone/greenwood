# Shared helpers for the estimators: building the model frame, reading the response, grouping,
# and validating common arguments.

# -- model frames -----------------------------------------------------------------------------

# Split a formula's right-hand side into ordinary terms and the arguments of the `strata()`,
# `cluster()`, and `frailty()` specials. The specials are taken apart here rather than evaluated, so
# no `strata()` function is needed and each stratifying variable keeps its own column and type.
split_formula <- function(formula, data = NULL) {
  if (!inherits(formula, "formula") || length(formula) != 3L) {
    stop("`formula` must be a two-sided formula such as `Surv(time, status) ~ x`.", call. = FALSE)
  }
  tt <- if (is.null(data) || !is.data.frame(data)) {
    stats::terms(formula)
  } else {
    stats::terms(formula, data = data)
  }
  labels <- attr(tt, "term.labels")
  exprs <- lapply(labels, str2lang)
  special <- function(name) {
    vapply(exprs, function(e) is.call(e) && identical(e[[1L]], as.name(name)), logical(1))
  }
  is_strata <- special("strata")
  is_cluster <- special("cluster")
  is_frailty <- special("frailty") | special("frailty.gamma") | special("frailty.gaussian")
  args_of <- function(which) {
    unlist(lapply(exprs[which], function(e) as.list(e)[-1L]), recursive = FALSE)
  }
  strata <- args_of(is_strata)
  cluster <- args_of(is_cluster)
  if (length(cluster) > 1L) {
    stop("A formula can have at most one `cluster()` term.", call. = FALSE)
  }
  if (sum(is_frailty) > 1L) {
    stop("A formula can have at most one `frailty()` term.", call. = FALSE)
  }
  frailty <- if (any(is_frailty)) frailty_term(exprs[[which(is_frailty)]]) else NULL
  list(
    response = formula_response(formula[[2L]]),
    terms = labels[!is_strata & !is_cluster & !is_frailty],
    strata = vapply(strata, deparse1, character(1)),
    cluster = vapply(cluster, deparse1, character(1)),
    frailty = frailty,
    intercept = attr(tt, "intercept") == 1L,
    offset = attr(tt, "offset"),
    env = environment(formula)
  )
}

# The response of a formula, with an event_time() response converted by as_surv() so the model
# frame holds a Surv() response.
formula_response <- function(expr) {
  heads <- list(quote(event_time), quote(greenwood::event_time))
  if (is.call(expr) && any(vapply(heads, identical, logical(1), expr[[1L]]))) {
    return(as.call(list(quote(greenwood::as_surv), expr)))
  }
  expr
}

# The cluster variable and distribution of a `frailty(id)` term. `frailty(id)` is gamma, and
# `frailty(id, distribution = "gaussian")`, `frailty.gaussian(id)`, and `frailty.gamma(id)` choose
# the distribution explicitly. A gaussian frailty is a lognormal one on the hazard scale.
frailty_term <- function(expr) {
  term <- deparse1(expr)
  head <- as.character(expr[[1L]])
  args <- as.list(expr)[-1L]
  named <- if (is.null(names(args))) rep(FALSE, length(args)) else names(args) != ""
  distributions <- c(gamma = "gamma", gaussian = "lognormal", lognormal = "lognormal")
  distribution <- if (head == "frailty") {
    NULL
  } else {
    distributions[[sub("frailty.", "", head, fixed = TRUE)]]
  }
  for (name in names(args)[named]) {
    value <- args[[name]]
    if (name != "distribution" || !is.character(value) || !value %in% names(distributions)) {
      stop(
        sprintf(
          paste(
            "`%s` isn't supported. Use `frailty(id)`, or choose the distribution with",
            "`distribution = \"gamma\"` or `distribution = \"gaussian\"` (lognormal)."
          ),
          term
        ),
        call. = FALSE
      )
    }
    if (!is.null(distribution) && distribution != distributions[[value]]) {
      stop(sprintf("`%s` names two different frailty distributions.", term), call. = FALSE)
    }
    distribution <- distributions[[value]]
  }
  if (sum(!named) != 1L) {
    stop(sprintf("`%s` must name exactly one cluster variable.", term), call. = FALSE)
  }
  list(
    variable = deparse1(args[!named][[1L]]),
    distribution = if (is.null(distribution)) "gamma" else distribution
  )
}

# Build the model frame for a call to one of the estimators, the way lm() does, so `weights`,
# `subset`, `cluster`, and the formula's variables are evaluated in `data` and rows missing any of
# them are dropped together. An `istate` argument (the state each row starts in, for multi-state
# data) and a `frailty_cluster` argument are carried along the same way. `call` is the estimator's
# match.call() and `frame` its parent frame. Only estimators passing `frailty = TRUE` take a
# `frailty()` term.
gw_model_frame <- function(call, frame, formula, data = NULL, frailty = FALSE) {
  parts <- split_formula(formula, data)
  if (!is.null(parts$frailty) && !frailty) {
    stop(
      sprintf("`%s()` doesn't support `frailty()` terms.", deparse1(call[[1L]])),
      call. = FALSE
    )
  }
  labels <- c(parts$terms, parts$strata, parts$cluster, parts$frailty$variable)
  rebuilt <- stats::reformulate(
    if (length(labels) == 0L) "1" else labels,
    response = parts$response,
    intercept = TRUE,
    env = parts$env
  )
  mf <- call[c(1L, match(
    c("data", "weights", "subset", "na.action", "cluster", "istate", "frailty_cluster"),
    names(call),
    0L
  ))]
  mf$formula <- rebuilt
  mf$drop.unused.levels <- TRUE
  mf[[1L]] <- quote(stats::model.frame)
  mf <- eval(mf, frame)

  response <- stats::model.response(mf)
  if (!inherits(response, "Surv")) {
    stop("The left-hand side of `formula` must be a `Surv()` response.", call. = FALSE)
  }

  w <- stats::model.weights(mf)
  if (!is.null(w) && (!is.numeric(w) || any(!is.finite(w)) || any(w < 0))) {
    stop("`weights` must be non-negative numbers.", call. = FALSE)
  }

  cluster <- mf[["(cluster)"]]
  if (length(parts$cluster) == 1L) {
    if (!is.null(cluster)) {
      stop("Give clusters either as `cluster =` or as a `cluster()` term, not both.", call. = FALSE)
    }
    cluster <- mf[[parts$cluster]]
  }

  frailty_cluster <- mf[["(frailty_cluster)"]]
  if (!is.null(parts$frailty)) {
    if (!is.null(frailty_cluster)) {
      stop(
        paste(
          "Give the frailty clusters either as `frailty_cluster =` or as a `frailty()` term,",
          "not both."
        ),
        call. = FALSE
      )
    }
    frailty_cluster <- mf[[parts$frailty$variable]]
  }

  pick <- function(names) {
    if (length(names) == 0L) {
      return(NULL)
    }
    out <- mf[unique(names)]
    attr(out, "terms") <- NULL
    out
  }
  list(
    mf = mf,
    response = response,
    weights = if (is.null(w)) rep(1, nrow(mf)) else as.numeric(w),
    has_weights = !is.null(w),
    cluster = cluster,
    istate = mf[["(istate)"]],
    frailty_cluster = frailty_cluster,
    frailty_distribution = parts$frailty$distribution,
    frailty_name = if (is.null(parts$frailty)) NULL else parts$frailty$variable,
    terms = parts$terms,
    vars = pick(parts$terms),
    strata = pick(parts$strata),
    intercept = parts$intercept,
    n_dropped = length(attr(mf, "na.action")),
    na_action = attr(mf, "na.action"),
    formula = formula
  )
}

# The covariate design matrix, without the intercept column (the model supplies its own), and
# what's needed to rebuild it for new data. It can have no columns. Collinear covariates are an
# error unless `check_rank = FALSE` (for the penalized models, which can fit them).
covariate_design <- function(frame, check_rank = TRUE) {
  tt <- attr(frame$mf, "terms")
  mm <- stats::model.matrix(tt, frame$mf)
  contrasts <- attr(mm, "contrasts")
  x <- mm[, colnames(mm) != "(Intercept)", drop = FALSE]
  if (check_rank && ncol(x) > 0L && qr(cbind(1, x))$rank < ncol(x) + 1L) {
    stop(
      "The covariates are collinear (or one is constant), so the model can't be fit.",
      call. = FALSE
    )
  }
  attr(x, "assign") <- NULL
  attr(x, "contrasts") <- NULL
  storage.mode(x) <- "double"
  rownames(x) <- NULL
  list(
    x = x,
    terms = stats::delete.response(tt),
    xlevels = stats::.getXlevels(tt, frame$mf),
    contrasts = contrasts
  )
}

# The response's start, stop, and event columns, for right-censored or counting-process data.
surv_parts <- function(response, fn, types = c("right", "counting")) {
  type <- attr(response, "type")
  if (!type %in% types) {
    stop(
      sprintf(
        "`%s` needs %s data, not \"%s\".",
        fn,
        if (identical(types, c("right", "counting"))) {
          "right-censored or counting-process"
        } else if (identical(types, "right")) {
          "right-censored"
        } else {
          paste(types, collapse = " or ")
        },
        type
      ),
      call. = FALSE
    )
  }
  y <- unclass(response)
  n <- nrow(y)
  if (type %in% c("right", "mright")) {
    list(type = type, start = rep(-Inf, n), stop = y[, "time"], event = y[, "status"])
  } else if (type %in% c("counting", "mcounting")) {
    list(type = type, start = y[, "start"], stop = y[, "stop"], event = y[, "status"])
  } else {
    list(type = type, y = y)
  }
}

# -- groups ----------------------------------------------------------------------------------

# One row per group, in group order (see group_factor(), with the first variable varying slowest),
# and each row's group number. The keys start with the `strata` label, then the grouping variables.
group_keys <- function(vars, n = NULL) {
  if (is.null(vars) || length(vars) == 0L) {
    keys <- data.frame(strata = "all", stringsAsFactors = FALSE)
    if (is.null(n)) {
      n <- if (is.null(vars)) 0L else nrow(vars)
    }
    return(list(keys = keys, index = rep(1L, n)))
  }
  key <- interaction(lapply(vars, group_factor), drop = TRUE, lex.order = TRUE)
  first <- match(levels(key), key)
  values <- vars[first, , drop = FALSE]
  rownames(values) <- NULL
  keys <- cbind(data.frame(strata = strata_labels(values), stringsAsFactors = FALSE), values)
  list(keys = keys, index = as.integer(key))
}

# A grouping variable as a factor, which sets the group order. A factor keeps its levels. Other
# values are sorted, numbers in numeric order and text in code-point order (the C locale's), so the
# order is the same in every locale.
group_factor <- function(x) {
  if (is.factor(x)) {
    return(x)
  }
  factor(x, levels = sort(unique(x), method = "radix"))
}

# Curve labels: `name=value` pairs joined with ", ", the same in every table, printout, and plot.
strata_labels <- function(values) {
  pairs <- Map(function(name, value) paste0(name, "=", as.character(value)), names(values), values)
  do.call(paste, c(unname(pairs), sep = ", "))
}

# The key columns (strata and grouping variables) of group `g`, repeated `n` times.
repeat_keys <- function(keys, g, n) {
  out <- keys[rep(g, n), , drop = FALSE]
  rownames(out) <- NULL
  out
}

# Bind one data frame per group, each prefixed with the group's keys.
bind_groups <- function(keys, tables) {
  out <- do.call(rbind, lapply(seq_along(tables), function(g) {
    cbind(repeat_keys(keys, g, nrow(tables[[g]])), tables[[g]])
  }))
  rownames(out) <- NULL
  out
}

# -- risk sets ---------------------------------------------------------------------------------

# Total weight of the subjects with x >= t, for each t.
weight_at_or_above <- function(x, w, t) {
  o <- order(x)
  cumulative <- c(0, cumsum(w[o]))
  below <- findInterval(t, x[o], left.open = TRUE)
  sum(w) - cumulative[below + 1L]
}

# Weighted number at risk at each t: start < t <= stop.
weight_at_risk <- function(start, stop, w, t) {
  weight_at_or_above(stop, w, t) - weight_at_or_above(start, w, t)
}

# The event table of one group: at each distinct stop time, the weighted number at risk
# (start < t <= stop), with events, and censored. With non-integer weights, the number at risk and
# the number of events are summed differently, so a time where everyone at risk has the event can
# leave a survivor weight of about 1e-16 instead of 0. That rounding residue is treated as no
# survivors, so estimates drop to exactly 0.
count_events <- function(start, stop, event, w) {
  time <- sort(unique(stop))
  n_risk <- weight_at_risk(start, stop, w, time)
  at <- match(stop, time)
  n_event <- tabulate_sum(at, w * (event == 1), length(time))
  n_censor <- tabulate_sum(at, w * (event == 0), length(time))
  everyone <- abs(n_risk - n_event) <= 1e-10 * n_risk
  n_risk[everyone] <- n_event[everyone]
  list(time = time, n_risk = n_risk, n_event = n_event, n_censor = n_censor)
}

# Sum of `value` at each index 1..n.
tabulate_sum <- function(index, value, n) {
  out <- numeric(n)
  if (length(index) == 0L) {
    return(out)
  }
  sums <- rowsum(value, index, reorder = TRUE)
  out[as.integer(rownames(sums))] <- sums[, 1L]
  out
}

# Read a right-continuous step function at `query`: the last step at a time <= t, or `baseline`
# before the first step.
step_at <- function(time, value, query, baseline) {
  i <- findInterval(query, time)
  out <- rep(baseline, length(query))
  out[i > 0] <- value[i[i > 0]]
  out
}

# -- linear predictors -------------------------------------------------------------------------

# The linear predictor x %*% beta, summed one row at a time in a fixed order. An optimized BLAS
# can sum some rows with fused multiply-adds and others without, so two subjects with the same
# covariates can get linear predictors that differ in the last bit. Here they always match, which
# keeps the tied pairs in the concordance tied.
linear_predictor <- function(x, beta) {
  if (ncol(x) == 0L) {
    return(numeric(nrow(x)))
  }
  out <- as.vector(x[, 1L]) * beta[[1L]]
  for (j in seq_len(ncol(x))[-1L]) {
    out <- out + as.vector(x[, j]) * beta[[j]]
  }
  # Named by row, as drop(x %*% beta) is.
  if (nrow(x) > 1L) {
    names(out) <- rownames(x)
  }
  out
}

# -- arguments -------------------------------------------------------------------------------

check_conf_level <- function(conf_level) {
  if (!is.numeric(conf_level) || length(conf_level) != 1 || is.na(conf_level) ||
    !(conf_level > 0 && conf_level < 1)) {
    stop("`conf_level` must be a single number between 0 and 1.", call. = FALSE)
  }
  invisible(conf_level)
}

check_flag <- function(x, name) {
  if (!isTRUE(x) && !isFALSE(x)) {
    stop(sprintf("`%s` must be `TRUE` or `FALSE`.", name), call. = FALSE)
  }
  invisible(x)
}

check_number <- function(x, name, min = -Inf, finite = TRUE) {
  if (!is.numeric(x) || length(x) != 1L || is.na(x) || (finite && !is.finite(x)) || x < min) {
    bound <- if (is.finite(min)) sprintf(" no less than %s", format(min)) else ""
    stop(sprintf("`%s` must be a single number%s.", name, bound), call. = FALSE)
  }
  invisible(x)
}

check_times <- function(times, name = "times") {
  if (!is.numeric(times) || length(times) == 0L || anyNA(times)) {
    stop(sprintf("`%s` must be one or more numbers.", name), call. = FALSE)
  }
  invisible(as.numeric(times))
}

z_value <- function(conf_level) {
  stats::qnorm(1 - (1 - conf_level) / 2)
}

# Adjust p-values for multiple comparisons. `"bh"` is Benjamini-Hochberg, as in Python.
adjust_p <- function(p, correction) {
  method <- switch(correction,
    holm = "holm",
    bh = "BH",
    bonferroni = "bonferroni",
    none = "none",
    stop(
      sprintf(
        "`correction` must be \"holm\", \"bh\", \"bonferroni\", or \"none\", not \"%s\".",
        correction
      ),
      call. = FALSE
    )
  )
  stats::p.adjust(p, method = method)
}

# A note for printouts about rows dropped for missing values.
cat_dropped <- function(n_dropped) {
  if (n_dropped > 0) {
    cat(sprintf(
      "\n%d %s dropped for missing values.\n",
      n_dropped, if (n_dropped == 1) "row was" else "rows were"
    ))
  }
}

format_p <- function(p) {
  format.pval(p, digits = 3, eps = 1e-16)
}

deparse_formula <- function(formula) {
  paste(deparse(formula, width.cutoff = 500L), collapse = " ")
}
