# The response model is shared with the Python package, which ports `Surv()`. R Greenwood
# re-exports the reference implementation rather than reimplementing it, so the name matches
# across languages and the R side can never drift from survival.

#' @importFrom survival Surv
#' @export
survival::Surv
