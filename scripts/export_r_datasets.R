#!/usr/bin/env Rscript
# Export the bundled datasets for the R package, as r/data/*.rda.
#
# They are the datasets of python/greenwood/data/ (see scripts/export_datasets.R), from the same
# sources: survival for all but e1684, which is from smcure. Factor columns stay factors, and the
# values and factor levels are checked against the Python copies. Run from the repo root:
#   Rscript scripts/export_r_datasets.R

suppressPackageStartupMessages({
  library(survival)
})
if (!requireNamespace("smcure", quietly = TRUE)) {
  stop("smcure is required for e1684: install.packages('smcure')")
}

python_levels <- jsonlite::read_json(file.path("python", "greenwood", "data", "levels.json"))

out_dir <- file.path("r", "data")
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

# Drop the SAS-style label and format attributes some columns carry, keeping factors as they are.
clean <- function(df) {
  for (name in names(df)) {
    attr(df[[name]], "label") <- NULL
    attr(df[[name]], "format") <- NULL
  }
  rownames(df) <- NULL
  df
}

# Every dataset must hold the values of the Python package's copy.
check_against_csv <- function(df, name) {
  path <- file.path("python", "greenwood", "data", paste0(name, ".csv.gz"))
  csv <- utils::read.csv(gzfile(path), na.strings = c("", "NA"), stringsAsFactors = FALSE)
  stopifnot(identical(names(csv), names(df)), nrow(csv) == nrow(df))
  for (column in names(df)) {
    ours <- df[[column]]
    theirs <- csv[[column]]
    if (is.factor(ours)) {
      ours <- as.character(ours)
    }
    same <- if (is.numeric(ours)) {
      isTRUE(all.equal(as.numeric(ours), as.numeric(theirs), tolerance = 1e-12))
    } else {
      identical(ours, as.character(theirs))
    }
    if (!same) {
      stop(sprintf("%s$%s differs from %s", name, column, path))
    }
  }
  # The Python copy gives the factor columns back as categoricals with these levels.
  recorded <- python_levels[[name]]
  factors <- names(df)[vapply(df, is.factor, logical(1))]
  stopifnot(setequal(factors, names(recorded)))
  for (column in factors) {
    if (!identical(levels(df[[column]]), as.character(recorded[[column]]))) {
      stop(sprintf("the levels of %s$%s differ from levels.json", name, column))
    }
  }
}

save_dataset <- function(df, name) {
  df <- clean(df)
  check_against_csv(df, name)
  env <- new.env()
  assign(name, df, envir = env)
  path <- file.path(out_dir, paste0(name, ".rda"))
  save(list = name, file = path, envir = env, compress = "xz", version = 2)
  cat(sprintf("wrote %-20s %d x %d\n", path, nrow(df), ncol(df)))
}

data(cancer, package = "survival")
save_dataset(lung, "lung")
save_dataset(veteran, "veteran")
save_dataset(ovarian, "ovarian")
save_dataset(pbc, "pbc")
save_dataset(pbcseq, "pbcseq")
save_dataset(colon, "colon")
save_dataset(mgus2, "mgus2")

# As in the Python package, without the one row that is missing AGE and SEX.
data(e1684, package = "smcure", envir = environment())
save_dataset(e1684[stats::complete.cases(e1684), ], "e1684")

cat("done\n")
