# The cause-specific Cox model: cause_specific_cox(), a cox_ph() fit to a competing-risks response
# with the other causes taken as censoring. Tested by tests/testthat/test-cause-specific-cox.R.

#' Cause-specific Cox regression
#'
#' `cause_specific_cox()` fits a Cox model to the hazard of one cause of a competing-risks
#' response: among the subjects still free of any event, how the covariates change the rate of
#' events of this cause. Subjects with an event of another cause leave the risk set at that time,
#' as if censored, and the model is [cox_ph()] fit to the recoded response.
#'
#' The cause-specific hazard ratios describe the rate of the cause, not its cumulative incidence.
#' A covariate that raises the hazard of a competing cause lowers the incidence of this one even
#' when its cause-specific hazard ratio is 1. Use [fine_gray()] to model the cumulative incidence
#' directly. Fitting one model per cause describes all of the hazards.
#'
#' @param formula A formula with a competing-risks [Surv()] response on the left and the
#'   covariates on the right. The response is `Surv(time, event)` or `Surv(start, stop, event)`
#'   with `event` a factor whose first level means censored and each other level is a cause, as in
#'   `Surv(time, factor(status, 0:2, c("censor", "pcm", "death"))) ~ age + sex`. `strata()`,
#'   `cluster()`, and `frailty()` terms work as in [cox_ph()].
#' @param data A data frame holding the variables in `formula`.
#' @param cause The cause to model, a state name such as `"pcm"` or its number (`1` for the
#'   first cause after censoring). The default is `1`.
#' @param ... Further arguments for [cox_ph()]: `weights`, `subset`, `na.action`, `ties`,
#'   `robust`, `cluster`, `frailty`, `frailty_cluster`, `conf_level`, `max_iter`, and `tol`.
#'
#' @return A `cause_specific_cox` object, which is also a `cox_ph` object, so every [cox_ph()]
#'   method works on it: [as.data.frame()], [coef()], [vcov()], [predict()] (survival curves
#'   predict being free of the cause, as if the other causes couldn't happen), [residuals()],
#'   [baseline_hazard()], and [cox_zph()]. It also holds `cause` (the state modeled) and `states`
#'   (all of the causes). `n_event` counts the events of the cause.
#'
#' @seealso [fine_gray()] for the subdistribution hazard, and [aalen_johansen()] for the
#'   cumulative incidence.
#'
#' @examples
#' mgus <- data.frame(
#'   time = c(30, 45, 60, 72, 80, 95, 110, 120, 130, 150, 160, 180, 200, 210, 220, 240),
#'   status = c(1, 2, 0, 2, 1, 2, 0, 1, 2, 2, 0, 1, 1, 2, 0, 2),
#'   age = c(62, 75, 58, 80, 66, 71, 59, 64, 77, 82, 60, 68, 55, 79, 63, 74)
#' )
#' mgus$event <- factor(mgus$status, levels = 0:2, labels = c("censor", "pcm", "death"))
#'
#' # The hazard of progression, with deaths taken as censoring
#' pcm <- cause_specific_cox(Surv(time, event) ~ age, data = mgus, cause = "pcm")
#' pcm
#'
#' # The hazard of death, as hazard ratios
#' death <- cause_specific_cox(Surv(time, event) ~ age, data = mgus, cause = "death")
#' as.data.frame(death, exponentiate = TRUE)
#' @export
cause_specific_cox <- function(formula, data, cause = 1, ...) {
  call <- match.call(expand.dots = TRUE)
  frame <- gw_model_frame(
    call, parent.frame(), formula, if (missing(data)) NULL else data, frailty = TRUE
  )
  type <- attr(frame$response, "type")
  if (!type %in% c("mright", "mcounting")) {
    stop(
      paste0(
        "`cause_specific_cox()` needs a competing-risks response, `Surv(time, event)` with ",
        "`event` a factor whose first level is censoring. Use `cox_ph()` for one kind of event."
      ),
      call. = FALSE
    )
  }
  states <- attr(frame$response, "states")
  target <- cr_cause(cause, states)
  status <- unclass(frame$response)[, "status"]
  if (!any(status == target)) {
    stop(sprintf("There are no events of cause \"%s\".", states[target]), call. = FALSE)
  }

  # cox_ph() is called as the user called this function, with the response recoded in the
  # formula, so `weights`, `subset`, and `cluster` are read from `data` the usual way.
  env <- new.env(parent = environment(formula))
  assign(".cause_specific", function(y) cause_specific_response(y, target), envir = env)
  recoded <- formula
  recoded[[2L]] <- call(".cause_specific", formula[[2L]])
  environment(recoded) <- env
  cox_call <- call
  cox_call[[1L]] <- cox_ph
  cox_call$cause <- NULL
  cox_call$formula <- recoded
  fit <- eval(cox_call, parent.frame())

  fit$cause <- states[target]
  fit$states <- states
  fit$formula <- formula
  fit$call <- call
  class(fit) <- c("cause_specific_cox", class(fit))
  fit
}

# A competing-risks response with the events of cause `target` as events and every other cause as
# censoring.
cause_specific_response <- function(y, target) {
  m <- unclass(y)
  event <- as.numeric(m[, "status"] == target)
  if (attr(y, "type") == "mright") {
    new_surv(list(time = unname(m[, "time"]), status = event), "right")
  } else {
    new_surv(list(start = unname(m[, "start"]), stop = unname(m[, "stop"]), status = event),
      "counting"
    )
  }
}
