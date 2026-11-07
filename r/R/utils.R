# Shared helpers for the estimators: building the model frame, reading the response, grouping,
# and validating common arguments.

# -- model frames -----------------------------------------------------------------------------

# Split a formula's right-hand side into ordinary terms and the arguments of the `strata()` and
# `cluster()` specials. The specials are taken apart here rather than evaluated, so no `strata()`
# function is needed and each stratifying variable keeps its own column and type.
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
  args_of <- function(which) {
    unlist(lapply(exprs[which], function(e) as.list(e)[-1L]), recursive = FALSE)
  }
  strata <- args_of(is_strata)
  cluster <- args_of(is_cluster)
  if (length(cluster) > 1L) {
    stop("A formula can have at most one `cluster()` term.", call. = FALSE)
  }
  list(
    response = formula[[2L]],
    terms = labels[!is_strata & !is_cluster],
    strata = vapply(strata, deparse1, character(1)),
    cluster = vapply(cluster, deparse1, character(1)),
    intercept = attr(tt, "intercept") == 1L,
    offset = attr(tt, "offset"),
    env = environment(formula)
  )
}

# Build the model frame for a call to one of the estimators, the way lm() does, so `weights`,
# `subset`, `cluster`, and the formula's variables are evaluated in `data` and rows missing any of
# them are dropped together. An `istate` argument (the state each row starts in, for multi-state
# data) is carried along the same way. `call` is the estimator's match.call() and `frame` its parent frame.
gw_model_frame <- function(call, frame, formula, data = NULL) {
  parts <- split_formula(formula, data)
  labels <- c(parts$terms, parts$strata, parts$cluster)
  rebuilt <- stats::reformulate(
    if (length(labels) == 0L) "1" else labels,
    response = parts$response,
    intercept = TRUE,
    env = parts$env
  )
  mf <- call[c(1L, match(
    c("data", "weights", "subset", "na.action", "cluster", "istate"), names(call), 0L
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
    terms = parts$terms,
    vars = pick(parts$terms),
    strata = pick(parts$strata),
    intercept = parts$intercept,
    n_dropped = length(attr(mf, "na.action")),
    na_action = attr(mf, "na.action"),
    formula = formula
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

# One row per group (sorted by the grouping variables), and each row's group number. The keys
# start with the `strata` label, followed by the grouping variables.
group_keys <- function(vars, n = NULL) {
  if (is.null(vars) || length(vars) == 0L) {
    keys <- data.frame(strata = "all", stringsAsFactors = FALSE)
    if (is.null(n)) {
      n <- if (is.null(vars)) 0L else nrow(vars)
    }
    return(list(keys = keys, index = rep(1L, n)))
  }
  key <- interaction(lapply(vars, factor), drop = TRUE, lex.order = TRUE)
  first <- match(levels(key), key)
  values <- vars[first, , drop = FALSE]
  rownames(values) <- NULL
  keys <- cbind(data.frame(strata = strata_labels(values), stringsAsFactors = FALSE), values)
  list(keys = keys, index = as.integer(key))
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

# A single label per row for a data frame of grouping values, for tests that compare groups.
group_labels <- function(vars) {
  if (ncol(vars) == 1L) {
    values <- vars[[1L]]
    return(if (is.factor(values)) values else factor(values))
  }
  interaction(vars, drop = TRUE, lex.order = TRUE, sep = ", ")
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
